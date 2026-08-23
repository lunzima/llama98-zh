#!/bin/bash
# --kv-rot gate.
#
#   kv_rot_gate.sh <model-dir> [n-tokens]
#
# --kv-rot was the one knob in this engine that CHANGED THE OUTPUT and
# had no gate at all. It is not a performance knob (build/perf_knob_gate.sh
# covers those), so invariance is the wrong assertion: rotating Q/K/V
# with a Hadamard before the KV cache quantizes them is supposed to make
# the quantization error SMALLER, and that is a claim with a number.
#
# The reference is --kv f32, the same cache without quantization. For a
# lossy format the rotated run must land closer to it than the
# unrotated one.
#
# WHAT THIS FOUND, AND WHAT IT GOT WRONG FIRST. The first version
# reported q8 rot off 2.963e-02 / rot on 2.943e-02 (0.68% better) and q4
# BYTE-IDENTICAL between its two arms, and concluded the knob does
# nothing for q4 - the lossier format, the one that should gain most -
# so it recorded "either a defect or an undocumented limitation" and
# pinned the identity.
#
# BOTH q4 ARMS WERE THE SAME ARM. The off arm was a bare `--kv q4`, and
# choosing q4 turns rotation ON (parse_kvfmt). Two identical
# configurations are byte-identical for a reason that carries no
# information about the knob at all.
#
# MEASURED WITH A REAL OFF ARM. kmr20, 200 tokens, 2026-08-23 - and the
# model and date are written down here because the older pair above has
# neither, so it cannot be reproduced or compared against:
#   --kv q8   rot off 4.582e-02, rot on 3.836e-02  - 16.3% better
#   --kv q4   rot off 2.882e-01, rot on 1.736e-01  - 39.7% better
# The q8 half of that has since been overtaken by conv1d's int16 exit;
# see the note above the q8 arm below. The q4 half stands.
# The shape is what the theory predicted all along: the rotation helps
# the LOSSIER format far more. q8's number was never the surprise; q4's
# zero was, and the zero was the instrument, not the engine.
#
# (q8 moved from 0.68% to 16.3% between the two runs. Not this change -
# verified: bare `--kv q8` and `--kv q8 --kv-rot off` are byte-identical,
# so the arm did not move. Something else in the engine or the model did,
# and the older figure has no provenance to chase it with.)
#
# So both formats now go through helps(), and helps() names --kv-rot on
# BOTH arms so it can never again depend on a default. What the default
# does is asserted separately, below, where it is the subject.
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
NTOK=${2:-200}

BIN=${BIN:-}
if [ -z "$BIN" ]; then
    for c in "$ROOT/llama98.exe" "$ROOT/llama98"; do
        [ -x "$c" ] && BIN=$c && break
    done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "kv_rot_gate: SKIP - no native binary (make native, or set BIN)"; exit 2; }
[ -n "$MODEL" ] && [ -d "$MODEL" ] || {
    echo "kv_rot_gate: SKIP - model dir not given or not found: '$MODEL'"; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
if ! $CC -O2 -o "$W/logit_rms" "$ROOT/build/logit_rms.c" -lm 2>"$W/cc.log"; then
    echo "kv_rot_gate: SKIP - could not build build/logit_rms.c"
    tail -3 "$W/cc.log"
    exit 2
fi

# A token list long enough for KV error to accumulate; the last
# position's logits are what --dump-logits writes, so a short context
# would be measuring almost nothing.
TOKENS=$(awk -v n="$NTOK" 'BEGIN{for(i=1;i<=n;i++){printf "%s%d",(i>1?",":""),i}}')

run() {   # run TAG ARGS...
    local tag="$1"; shift
    "$BIN" "$MODEL" --tokens "$TOKENS" --dump-logits "$W/$tag.bin" "$@" \
        >"$W/$tag.log" 2>&1 || { echo "FAIL: run '$tag' failed"
                                 tail -5 "$W/$tag.log"; return 1; }
    [ -s "$W/$tag.bin" ] || { echo "FAIL: run '$tag' wrote no logits"; return 1; }
    return 0
}

run ref --kv f32 || exit 1
FAILED=0

# helps FMT - the rotated run must be strictly closer to the f32 cache.
#
# BOTH ARMS NAME --kv-rot EXPLICITLY, and that is the whole correction
# this function once needed: the off arm used to be a bare `--kv $fmt`
# and rely on the default being off. True for q8. NOT true for q4, which
# turns rotation on as part of choosing the format (parse_kvfmt, and the
# comment above it says so) - so the q4 "off" arm was a second copy of
# the on arm, the two came out byte-identical, and this gate recorded
# that as "the knob is inert on q4, either a defect or an undocumented
# limitation". It was neither. It was one missing flag.
helps() {
    local fmt="$1" off on
    run "${fmt}off" --kv "$fmt" --kv-rot off || return 1
    run "${fmt}on"  --kv "$fmt" --kv-rot on || return 1
    off=$("$W/logit_rms" "$W/ref.bin" "$W/${fmt}off.bin" | cut -d' ' -f1)
    on=$("$W/logit_rms" "$W/ref.bin" "$W/${fmt}on.bin"  | cut -d' ' -f1)
    printf '  %-4s rot off %s  rot on %s  ' "$fmt" "$off" "$on"
    # %.9e sorts lexicographically in numeric order for same-sign values
    if [ "$on" \< "$off" ]; then echo "PASS (rotation helps)"; return 0; fi
    echo "FAIL (rotation does not help)"
    return 1
}

echo "kv_rot_gate: reference is --kv f32, $NTOK tokens"

# q4 KEEPS THE DIRECTION ASSERTION. It is the lossier format and the one
# the theory says gains most, and it does - 46% closer to the f32 cache
# at the time of writing.
helps q4 || FAILED=1

# q8 NO LONGER ASSERTS A DIRECTION, and that is a measurement rather than
# a weakening. It used to: 4.582e-02 off against 3.836e-02 on, 16%
# better. It is now 3.407e-02 off against 3.792e-02 on - rotation makes
# q8 WORSE by 11%.
#
# BISECTED, not guessed. At 2eb14d2 the old pair reproduces exactly; it
# becomes the new one at b16c77e/514fe48, conv1d's int16 exit for q, k
# and then v. What moved is the OFF arm (4.58e-02 -> 3.41e-02, 26%
# better); the ON arm barely moved. So the rotation was partly
# compensating for error the conv exit has since removed, and on q8 what
# is left no longer pays for the rotation's own rounding.
#
# That leaves --kv-rot's defaults right on both formats - off for q8, on
# for q4 - which is what the aliveness check below is for. Asserting the
# old direction would now be asserting something false, and pinning the
# CURRENT direction instead is precisely the move that produced this
# gate's earlier "q4 is inert" mistake: a recorded state read as a
# property.
alive() {   # alive FMT - the knob must still change the output
    local fmt="$1" off on
    run "${fmt}off" --kv "$fmt" --kv-rot off || return 1
    run "${fmt}on"  --kv "$fmt" --kv-rot on || return 1
    off=$("$W/logit_rms" "$W/ref.bin" "$W/${fmt}off.bin" | cut -d' ' -f1)
    on=$("$W/logit_rms" "$W/ref.bin" "$W/${fmt}on.bin"  | cut -d' ' -f1)
    printf '  %-4s rot off %s  rot on %s  ' "$fmt" "$off" "$on"
    if cmp -s "$W/${fmt}off.bin" "$W/${fmt}on.bin"; then
        echo "FAIL (the knob is inert on this format)"
        return 1
    fi
    echo "alive (direction not asserted - see the header)"
    return 0
}
alive q8 || FAILED=1

# THE DEFAULT IS PART OF THE CLAIM. helps() above pins both arms, so it
# can no longer see whether --kv q4 still turns rotation on by itself -
# and that coupling is what makes the obvious `--kv q4` invocation the
# good one. Asserted here, where it is the only thing being asserted:
# bare --kv q4 must equal --kv q4 --kv-rot on, and both must differ from
# --kv q4 --kv-rot off.
run q4dflt --kv q4 || exit 1
if ! cmp -s "$W/q4dflt.bin" "$W/q4on.bin"; then
    echo "  q4   FAIL: bare --kv q4 is no longer rotating - parse_kvfmt's"
    echo "       coupling to lz_kv_rot_enable is gone, so the obvious"
    echo "       invocation now gets the WORSE arm"
    FAILED=1
elif cmp -s "$W/q4dflt.bin" "$W/q4off.bin"; then
    echo "  q4   FAIL: --kv-rot off did not turn rotation off, so the two"
    echo "       arms compared above were one arm measured twice"
    FAILED=1
else
    echo "  q4   default couples to rotation (bare == on, both != off)"
fi

[ $FAILED -eq 0 ] || { echo "kv_rot_gate: FAIL"; exit 1; }
echo "kv_rot_gate: PASS"
exit 0
