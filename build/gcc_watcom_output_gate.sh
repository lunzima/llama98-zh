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
MODEL=${1:-}
TOKENS=${2:-1,2,3,4,5,6,7,8}

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
                --tokens "$TOKENS" --dump-all-logits "$side.$arm.bin" ) \
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
