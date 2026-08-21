#!/bin/bash
# Cross-compiler bit-identity gate for the int-domain fixed-tier quantize
# paths (--norm-write int, --attn-write int).
#
# WHY THIS EXISTS: lz_quantize_q8_int's fast-branch element loop fed a
# bare `(float)grp[k]` straight into a multiply by an arbitrary (non
# power-of-two) ratio. gcc/SSE rounds the int32->float conversion once
# (cvtsi2ssl) and the product again (mulss) - two roundings. Watcom/x87's
# `fild` loads the int exactly and `fmul` rounds only the product - one
# rounding. They disagree whenever |grp[k]| exceeds 2^24 and the true
# product lands within half a ULP of a .5 tie, which q8_round's
# round-half-even then resolves two different ways. Fixed by routing the
# cast through lz_i32f() (src/ops_kernel_shared.h), which forces the
# float32 rounding before the multiply on both compilers.
#
# This survived because the tier's original verification ran at 8 and 16
# tokens - both far too short for the recurrent state that feeds
# lz_quantize_q8_int to cross 2^24. Concrete instance: model kmr20,
# token position 146 of 153, layer 0, group 1 - clean on every shorter
# probe and on English input, only surfacing past ~150 tokens of real
# Chinese text. The fixtures below are real project text (not synthetic
# noise): build/norm_int_gate_tokens_zh.txt is exactly the 153-token
# reproduction prefix plus 450 more tokens of real Chinese docs text
# (causal - the prefix computes identically whatever follows it);
# build/norm_int_gate_tokens_en.txt is the matching English/code fixture
# (211 known-clean tokens plus 450 more). Two corpora because the
# divergence is data-specific: English alone never triggered it here.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP (toolchain or model fixture missing)
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
MAKE=${MAKE:-make}

# ---- preflight: every reason to stop here is a SKIP, not a FAIL --------
if ! command -v gcc >/dev/null 2>&1; then
    echo "norm_int_xcompiler_gate: SKIP - gcc not found"
    exit 2
fi
if [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
    :
elif [ -x /mnt/c/WATCOM/binl64/wcc386 ]; then
    :
else
    echo "norm_int_xcompiler_gate: SKIP - wcc386 not found (checked binnt64/binl64)"
    exit 2
fi

# kmr20 is a quantized L98Z-format export (hidden 512, 8 layers
# LLLALLLA, the same architecture family as models/kunmoe-v2) kept
# outside the repo alongside the other named checkpoints ci.sh's model
# sweep already uses - not reproducible from a recipe in this tree, same
# as those. LZ_NORM_INT_GATE_MODEL overrides the path directly.
MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
MODEL=${LZ_NORM_INT_GATE_MODEL:-$MODELS_ROOT/kmr20}
if [ ! -d "$MODEL" ]; then
    echo "norm_int_xcompiler_gate: SKIP - model not found: $MODEL" \
         "(set LZ_NORM_INT_GATE_MODEL or MODELS_ROOT)"
    exit 2
fi

TOK_ZH="$ROOT/build/norm_int_gate_tokens_zh.txt"
TOK_EN="$ROOT/build/norm_int_gate_tokens_en.txt"
for f in "$TOK_ZH" "$TOK_EN"; do
    if [ ! -f "$f" ]; then
        echo "norm_int_xcompiler_gate: SKIP - fixture missing: $f"
        exit 2
    fi
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# ---- build both toolchains (relies on make's own up-to-date check, same
# as kernel_matrix_gate.sh - a stale binary from a differently-flagged
# build is a risk every gate in this tree already accepts) -------------
echo "-- gcc build --"
if ! "$MAKE" native >"$WORK/gcc.build.log" 2>&1; then
    echo "norm_int_xcompiler_gate: ERROR 'make native' failed"
    tail -n 20 "$WORK/gcc.build.log"
    exit 2
fi
GCC_BIN="$ROOT/llama98.exe"
[ -x "$GCC_BIN" ] || GCC_BIN="$ROOT/llama98"
if [ ! -x "$GCC_BIN" ]; then
    echo "norm_int_xcompiler_gate: ERROR 'make native' produced no binary"
    exit 2
fi

echo "-- Watcom build --"
if ! "$MAKE" watcom-cli >"$WORK/wat.build.log" 2>&1; then
    echo "norm_int_xcompiler_gate: ERROR 'make watcom-cli' failed"
    tail -n 20 "$WORK/wat.build.log"
    exit 2
fi
WAT_BIN="$ROOT/llama98-w.exe"
if [ ! -f "$WAT_BIN" ]; then
    echo "norm_int_xcompiler_gate: ERROR 'make watcom-cli' produced no binary"
    exit 2
fi

# ---- run one (tier, corpus) pair on both toolchains, cmp the raw
# --dump-all-logits bytes. --scalar-fixed float isolates the tier under
# test the same way the RCA's own repro does - without it, AUTO would
# also flip other tiers and a mismatch could not be attributed. --------
FAIL=0
run_one() {
    local tier="$1" label="$2" tokfile="$3"
    local safe gdump wdump glog wlog
    safe=$(printf '%s' "$tier $label" | tr ' -' '__')
    gdump="$WORK/g_$safe.bin"; wdump="$WORK/w_$safe.bin"
    glog="$WORK/g_$safe.log"; wlog="$WORK/w_$safe.log"

    if ! "$GCC_BIN" "$MODEL" --tokens-file "$tokfile" $tier \
            --scalar-fixed float --dump-all-logits "$gdump" >"$glog" 2>&1; then
        echo "FAIL: $tier / $label - gcc run failed:"
        tail -n 10 "$glog"
        FAIL=1
        return
    fi
    if ! "$WAT_BIN" "$MODEL" --tokens-file "$tokfile" $tier \
            --scalar-fixed float --dump-all-logits "$wdump" >"$wlog" 2>&1; then
        echo "FAIL: $tier / $label - watcom run failed:"
        tail -n 10 "$wlog"
        FAIL=1
        return
    fi
    if [ ! -s "$gdump" ] || [ ! -s "$wdump" ]; then
        echo "FAIL: $tier / $label - one or both --dump-all-logits files are empty"
        FAIL=1
        return
    fi
    if ! cmp -s "$gdump" "$wdump"; then
        local ndiff sz
        ndiff=$(cmp -l "$gdump" "$wdump" | wc -l)
        sz=$(wc -c < "$gdump")
        echo "FAIL: $tier / $label - gcc/watcom --dump-all-logits differ ($ndiff of $sz bytes)"
        cmp "$gdump" "$wdump"
        FAIL=1
        return
    fi
    echo "PASS: $tier / $label ($(wc -c < "$gdump") bytes, $(wc -l < "$tokfile") tokens)"
}

for tier in "--norm-write int" "--attn-write int"; do
    run_one "$tier" zh "$TOK_ZH"
    run_one "$tier" en "$TOK_EN"
done

if [ "$FAIL" -ne 0 ]; then
    echo "norm_int_xcompiler_gate: FAIL"
    exit 1
fi
echo "norm_int_xcompiler_gate: PASS"
exit 0
