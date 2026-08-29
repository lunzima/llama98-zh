#!/bin/bash
# The gcc build and the Watcom build must produce the same logits.
#
#   gcc_watcom_output_gate.sh <model-dir> [tokens]
#
# NOTHING CHECKED THIS, and a 20% divergence lived in the tree because of
# it. build/gcc_watcom_pair_gate.sh compares the two toolchains' SETS of
# hand-written kernels - which cells are filled - and never runs either.
# build/kernel_isa_identity_gate.sh compares tiers WITHIN one binary
# (mmx against sse2, ref against sse2) and is blind across compilers by
# construction. So the one question that matters for a shipping Watcom
# binary - does it compute what the reference build computes - had no
# asker.
#
# What it was hiding: lz_gdn1_x4_asm's #pragma aux body zeroed mm0, mm1
# and mm2 and then accumulated into mm0, mm1, mm2 AND mm7, storing all
# four. mm7 was never initialised, so a quarter of the recurrence's
# pass-1 output was whatever the previous MMX or x87 instruction had
# left in that register. The gcc intrinsics twin zeroes all four. Result:
# on the default (fixed-point) tier the two builds' logits differed by
# 20% in sum and 30% in absmax - not a rounding difference, a wrong
# answer - while the float tier agreed exactly, because it does not
# reach that kernel.
#
# THE FLOAT TIER IS ALSO COMPARED, and it is not redundant. x87 and SSE
# are allowed to differ by a rounding step somewhere in principle; this
# tree's float arm is built so they do not, and having both rows means a
# failure says WHICH arm moved. A fixed-only comparison would have
# reported this bug without saying that the float path was fine, which
# is the fact that localised it.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP (a binary or the model is missing).
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)

# A DEFAULT MODEL, because "SKIP - model dir not given" every single run
# is a gate that has never asked its question. `make check` passes no
# MODEL, so without this the only invocation that ever did real work was
# one somebody typed by hand.
#
# _t2gate RATHER THAN kmr20, AND THAT CLOSES A RECORDED GAP.
# build/fixed_tier_xcompiler_gate.sh is pinned to kmr20 and its header
# says what that leaves out: kmr20 carries no T2 tensor, so the T2 half
# of the Q4_1/Q6_1/T2 epilogue family - which has no stored zero plane
# and reaches the fixed epilogue through its own `zneg` branch - was
# untested by every row of it. _t2gate is the same checkpoint with 12
# tensors moved from q6_1 to t2 (133,349,376 elements either way, the
# same q8_0/q4_1/q16_0 counts), so a divergence here and not there is
# attributable to the T2 path rather than to two different models.
# Two gates asking the same question of the same checkpoint would be
# one gate and a copy.
#
# Same MODELS_ROOT convention as that script, and still SKIPs where the
# tree is not there. _armgate2 stays a useful second opinion by
# argument; it is not the default because it closes nothing.
MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
MODEL=${1:-}
[ -n "$MODEL" ] || MODEL="$MODELS_ROOT/_t2gate"

# EIGHT TOKENS WAS THE WRONG DEFAULT, and this file is the place to say
# so because it shipped with it. src/lz_mathf.c's reduce_large defect
# (e7b9069) diverged the two toolchains from token 125 on kmr20 and 134
# on _armgate2, and every run shorter than that reads clean - as does
# every run shorter than ~150 for the lz_quantize_q8_int defect
# build/fixed_tier_xcompiler_gate.sh's header is about. A configuration
# that has never been run long enough to diverge has not been shown
# identical; it has only not yet been shown different. So the default is
# the long fixture when it is there, and the second argument still
# overrides for a quick manual poke.
TOK_ARG=${2:-}
LONG_FIXTURE="$ROOT/build/norm_int_gate_tokens_zh.txt"
if [ -n "$TOK_ARG" ]; then
    TOK_MODE="--tokens"; TOK_VAL="$TOK_ARG"
elif [ -f "$LONG_FIXTURE" ]; then
    TOK_MODE="--tokens-file"; TOK_VAL="$LONG_FIXTURE"
else
    TOK_MODE="--tokens"; TOK_VAL="1,2,3,4,5,6,7,8"
    echo "  note: $LONG_FIXTURE is missing, falling back to 8 tokens -" \
         "too short to expose a length-dependent divergence" >&2
fi

GBIN=""
for c in "$ROOT/llama98.exe" "$ROOT/llama98"; do
    [ -x "$c" ] && GBIN=$c && break
done
WBIN="$ROOT/llama98-w.exe"

if [ -z "$GBIN" ] || [ ! -x "$WBIN" ]; then
    echo "gcc_watcom_output_gate: SKIP - need both builds" \
         "(make && make watcom-cli); the two toolchains were NOT compared" >&2
    exit 2
fi
if [ -z "$MODEL" ] || [ ! -d "$MODEL" ]; then
    echo "gcc_watcom_output_gate: SKIP - model dir not given or not found:" \
         "'${MODEL:-<unset>}'" >&2
    exit 2
fi

# Beside the repo, not /tmp: a Windows binary cannot resolve an MSYS
# path, and --dump-all-logits would write nowhere and be believed.
W="$(dirname "$ROOT")/.gcc-watcom-out-$$"
mkdir -p "$W" || { echo "gcc_watcom_output_gate: FAIL - cannot create $W"; exit 1; }
trap 'rm -rf "$W"' EXIT

FAIL=0
for arm in all off; do
    for side in g w; do
        BIN=$GBIN; [ "$side" = w ] && BIN=$WBIN
        if ! ( cd "$W" && "$BIN" "$MODEL" --kernel ref --fixed "$arm" \
                "$TOK_MODE" "$TOK_VAL" --dump-all-logits "$side.$arm.bin" ) \
                >"$W/$side.$arm.log" 2>&1; then
            echo "gcc_watcom_output_gate: FAIL - the $side run (--fixed $arm)" \
                 "did not finish"
            tail -n 5 "$W/$side.$arm.log" | sed 's/^/      /'
            exit 1
        fi
        [ -s "$W/$side.$arm.bin" ] || {
            echo "gcc_watcom_output_gate: FAIL - the $side run (--fixed $arm)" \
                 "wrote no logits"; exit 1; }
    done
    if cmp -s "$W/g.$arm.bin" "$W/w.$arm.bin"; then
        echo "  --fixed $arm : identical ($(wc -c < "$W/g.$arm.bin") bytes)"
    else
        NDIFF=$(cmp -l "$W/g.$arm.bin" "$W/w.$arm.bin" | wc -l)
        SZ=$(wc -c < "$W/g.$arm.bin")
        echo "  --fixed $arm : DIFFER, $NDIFF of $SZ bytes"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "gcc_watcom_output_gate: FAIL"
    echo "      Integer arithmetic has no rounding freedom, so a fixed-tier"
    echo "      difference is a defect in one of the two hand-written"
    echo "      kernel sets, not a precision trade. If --fixed off agrees"
    echo "      and --fixed all does not, the fixed path is where to look."
    exit 1
fi
echo "gcc_watcom_output_gate: PASS (both tiers identical across toolchains)"
exit 0
