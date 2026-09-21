#!/bin/bash
# --kv-rot gate.
#
#   kv_rot_gate.sh <model-dir> [token-file ...]
#
# --kv-rot is the one knob in this engine that CHANGES THE OUTPUT, so
# invariance is the wrong assertion (build/perf_knob_gate.sh covers the
# knobs where it is the right one): rotating Q/K/V with a Hadamard before
# the KV cache quantizes them is supposed to make the quantization error
# SMALLER, and that is a claim with a number. The reference is --kv f32,
# the same cache without quantization; for a lossy format the rotated run
# must land closer to it.
#
# ---------------------------------------------------------------------
# THIS GATE HAS BEEN WRONG THREE TIMES. All three were the instrument,
# not the engine, and all three are worth keeping written down because
# each one looked like a finding about --kv-rot.
#
# 1. BOTH q4 ARMS WERE THE SAME ARM. The off arm was a bare `--kv q4`,
#    and choosing q4 turns rotation on by itself (parse_kvfmt). Two
#    identical configurations came out byte-identical and the gate
#    recorded that as "the knob is inert on q4". helps() now names
#    --kv-rot on BOTH arms; the default's coupling is asserted
#    separately, at the bottom, where it is the only thing being tested.
#
# 2. THE METRIC WAS ONE POSITION. --dump-logits writes the LAST
#    position's logits and nothing else, so the gate was comparing a
#    single sample of a discontinuous function - on a MoE model one
#    routing flip at that position moves it by most of its value.
#    MEASURED, by flipping ONE int8 level in the V cache (kmr20, zh
#    corpus, one element out of 308,736):
#
#      metric                        spread from a one-level flip
#      --dump-logits     (last)      -75% .. +5%
#      --dump-all-logits (all)       -3.0% .. +4.1%
#
#    So the old metric's noise floor was larger than every effect this
#    gate has ever reported. It now uses --dump-all-logits, and any
#    difference below ~5% should be read as nothing.
#
# 3. THE TOKENS WERE NOT TEXT. The sequence was the ids 1,2,...,200,
#    on which kmr20 scores mean NLL 8.54 against ln(32768) = 10.4 - the
#    model is barely predicting, which is not the distribution the
#    rotation is supposed to help on. The gate now runs the two real
#    corpora the norm-int gate already ships, and REQUIRES THE SAME SIGN
#    ON BOTH: one corpus is one sample too.
#
# ---------------------------------------------------------------------
# THE ARMS MUST DIFFER ONLY IN THE KNOB, and on q8 they do not always.
# The V projection's int16 exit (forward.c, LZ_VPROJ_I16) refuses when
# the rotation is on, because rot_head reads a float row the exit
# declined to build - so on a model where that exit is live, `--kv-rot
# off` takes it and `--kv-rot on` does not, and the difference between
# the arms is the rotation MINUS the exit. Read as the rotation's own
# effect that flips the answer, which is why arm() below asserts a
# direction only where the two arms took the same exits. The proof is not
# written out here as percentages: it is this gate's own output on kmr20
# over the two corpora below, once with the exit forced off on both arms
# and once shipping, and it is reproduced by running the gate there. What
# the engine keeps is not a measurement either - it is forward.c's v_may
# comment, which is where the reason the refusal exists is written.
#
# The engine keeps the refusal - the integer rotation was built and
# measured and is worse (see forward.c's v_may comment) - so this gate
# reads the `v_proj int16` counter per arm instead and asserts a
# DIRECTION only where the counters agree. Where they do not, it asserts
# aliveness and says which arm took the exit. That rule is the general
# one: a knob gate may only attribute a difference to its knob after
# checking that nothing else moved, and the int-pipeline counters are
# what makes that checkable here.
#
# MUTATION-TESTED, all four assertions, each by its own mutation:
#   on arm stops rotating          -> FAIL (the two arms' metric is equal,
#                                     which no declared direction accepts)
#   counters forced equal          -> q8 FAIL on kmr20 ONLY (surgical:
#                                     q4 and the coupling stay green)
#   bare --kv q4 given --kv-rot off-> coupling FAILs ONLY (surgical)
#   counters forced unequal with
#     identical arms               -> FAIL (inert)
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
shift 2>/dev/null || true
CORPORA=${*:-"$ROOT/build/norm_int_gate_tokens_zh.txt $ROOT/build/norm_int_gate_tokens_en.txt"}

# WHICH WAY THIS CHECKPOINT IS EXPECTED TO GO, declared per model rather
# than assumed for all of them. THIS IS AN EMPIRICAL FACT, NOT SOMETHING
# DERIVED FROM THE SOURCE: nothing in the rotation or in the quantizer
# says which way the error has to move, and the two models below are
# evidence that it does not move the same way on both.
#
#   kmr20      helps - the checkpoint the header's measurement was taken
#                      on, with the arms matched.
#   _armgate3  hurts - measured by this gate on its q4 arms, which are
#                      matched (vproj 0/0), on both corpora. ci.sh stage
#                      12 runs on this checkpoint, so this is the entry
#                      that gives that stage a direction to assert.
#
# A checkpoint that is not listed is REPORTED, not judged: arm() prints
# the pair it measured and asserts only that the two arms are not one arm
# measured twice. That is the honest reading of an undeclared model - the
# direction nobody has established is not a direction this gate may fail
# on. Adding one here is the whole cost of extending the assertion.
want_direction() {
    case "$(basename "$1")" in
        kmr20)     echo helps ;;
        _armgate3) echo hurts ;;
        *)         echo unknown ;;
    esac
}

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
for tf in $CORPORA; do
    [ -r "$tf" ] || {
        echo "kv_rot_gate: SKIP - token file not readable: $tf"; exit 2; }
done

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
if ! $CC -O2 -o "$W/logit_rms" "$ROOT/build/logit_rms.c" -lm 2>"$W/cc.log"; then
    echo "kv_rot_gate: SKIP - could not build build/logit_rms.c"
    tail -3 "$W/cc.log"
    exit 2
fi

FAILED=0

# run TAG ARGS... - writes $W/TAG.all (every position's logits) and
# $W/TAG.log. --dump-logits is NOT requested: see note 2 in the header.
run() {
    local tag="$1"; shift
    "$BIN" "$MODEL" --tokens "$TOKENS" --dump-all-logits "$W/$tag.all" "$@" \
        >"$W/$tag.log" 2>&1 || { echo "FAIL: run '$tag' failed"
                                 tail -5 "$W/$tag.log"; return 1; }
    [ -s "$W/$tag.all" ] || { echo "FAIL: run '$tag' wrote no logits"; return 1; }
    return 0
}
rms()  { "$W/logit_rms" "$W/ref.all" "$W/$1.all" | cut -d' ' -f1; }
# The V int16 exit's consumer-side counter, per arm. An exit that ran on
# one arm and not the other is a second variable in the comparison.
exit_n() { grep -m1 'v_proj int16' "$W/$1.log" | awk '{print $3}'; }

# arm FMT - both arms of one format on the current corpus. Prints the
# pair, then asserts: a DIRECTION when the two arms took the same exits,
# ALIVENESS when they did not.
arm() {
    local fmt="$1" off on noff non verdict want
    run "${fmt}off" --kv "$fmt" --kv-rot off || return 1
    run "${fmt}on"  --kv "$fmt" --kv-rot on  || return 1
    off=$(rms "${fmt}off"); on=$(rms "${fmt}on")
    noff=$(exit_n "${fmt}off"); non=$(exit_n "${fmt}on")
    printf '  %-4s off %s  on %s  vproj %s/%s  ' \
           "$fmt" "$off" "$on" "$noff" "$non"
    if [ "$noff" != "$non" ]; then
        # Not a weakening: with the arms unmatched there is no direction
        # to assert ABOUT THE ROTATION, and pinning the observed one is
        # exactly the move that produced mistake 1 above.
        if cmp -s "$W/${fmt}off.all" "$W/${fmt}on.all"; then
            echo "FAIL (inert)"; return 1
        fi
        echo "alive (arms differ in the V int16 exit too - see header)"
        return 0
    fi
    # %.9e sorts lexicographically in numeric order for same-sign values
    verdict=helps
    [ "$on" \< "$off" ] || verdict=hurts
    # Equal metrics are not a weak direction, they are no rotation: checked
    # before the declaration so that a checkpoint declared to hurt cannot
    # pass on an arm pair the knob never moved.
    if [ "$on" = "$off" ]; then
        echo "FAIL (the two arms' metric is equal - the rotation did nothing)"
        return 1
    fi
    want=$(want_direction "$MODEL")
    if [ "$want" = unknown ]; then
        echo "observed ($verdict; nothing is declared for this checkpoint)"
        return 0
    fi
    if [ "$verdict" = "$want" ]; then
        echo "PASS (rotation $verdict, as declared for this checkpoint)"
        return 0
    fi
    echo "FAIL (rotation $verdict, but this checkpoint is declared to $want)"
    return 1
}

echo "kv_rot_gate: reference is --kv f32, whole-sequence logit rms"
for tf in $CORPORA; do
    TOKENS=$(cat "$tf")
    echo "-- $(basename "$tf")"
    run ref --kv f32 || exit 1
    # Both formats are run on EVERY corpus: agreeing signs across two
    # independent sequences is the whole reason there are two. They are
    # ASSERTED only on a checkpoint want_direction() declares - on any
    # other they are reported, both corpora or neither.
    arm q4 || FAILED=1
    arm q8 || FAILED=1

    # THE DEFAULT IS PART OF THE CLAIM. arm() pins both arms, so it can
    # no longer see whether `--kv q4` still turns rotation on by itself,
    # and that coupling is what makes the obvious invocation the good
    # one. Asserted here, where it is the only thing being asserted.
    run q4dflt --kv q4 || exit 1
    if ! cmp -s "$W/q4dflt.all" "$W/q4on.all"; then
        echo "  q4   FAIL: bare --kv q4 is no longer rotating - parse_kvfmt's"
        echo "       coupling to lz_kv_rot_enable is gone, so the obvious"
        echo "       invocation now gets the unrotated arm"
        FAILED=1
    elif cmp -s "$W/q4dflt.all" "$W/q4off.all"; then
        echo "  q4   FAIL: --kv-rot off did not turn rotation off, so the two"
        echo "       arms compared above were one arm measured twice"
        FAILED=1
    else
        echo "  q4   default couples to rotation (bare == on, both != off)"
    fi
    rm -f "$W"/*.all
done

[ $FAILED -eq 0 ] || { echo "kv_rot_gate: FAIL"; exit 1; }
echo "kv_rot_gate: PASS"
exit 0
