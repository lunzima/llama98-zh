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
# WHAT THIS FOUND ON ITS FIRST RUN, and why one entry is inverted:
#   --kv q8   rot off 2.963e-02, rot on 2.943e-02  - 0.68% better
#   --kv q4   rot off 1.0035e-01, rot on 1.0035e-01 - BYTE-IDENTICAL
# The knob does NOTHING for q4, the lossier of the two formats and the
# one that would gain most. That is either a defect or an undocumented
# limitation; it is not something to leave unrecorded, and asserting
# "rot helps" for q4 would leave a permanently red gate that everyone
# learns to ignore.
#
# So q4's entry asserts the CURRENT state - identical - and says what
# that means. If the rotation ever starts reaching q4, this gate FAILS
# and someone updates it deliberately. A gate that only catches
# regressions cannot notice a knob quietly starting to work.
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

# helps FMT - the rotated run must be strictly closer to the f32 cache
helps() {
    local fmt="$1" off on
    run "${fmt}off" --kv "$fmt" || return 1
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
helps q8 || FAILED=1

# q4: asserted INERT, deliberately. See the header.
run q4off --kv q4 || exit 1
run q4on  --kv q4 --kv-rot on || exit 1
if cmp -s "$W/q4off.bin" "$W/q4on.bin"; then
    echo "  q4   rot on is BYTE-IDENTICAL to rot off - the knob is inert"
    echo "       here, which is the recorded state, not a pass of its claim"
else
    echo "  q4   rot on now CHANGES the output, which it did not when this"
    echo "       gate was written. That is not a regression - it is the knob"
    echo "       starting to reach q4. Measure it and turn this entry into"
    echo "       a helps() call."
    FAILED=1
fi

[ $FAILED -eq 0 ] || { echo "kv_rot_gate: FAIL"; exit 1; }
echo "kv_rot_gate: PASS"
exit 0
