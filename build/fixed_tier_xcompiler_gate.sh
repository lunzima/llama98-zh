#!/bin/bash
# Cross-compiler bit-identity gate for EVERY fixed-point write tier, not
# just the two int ones (build/ci.sh stage 6).
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
# Chinese text.
#
# THIS SCRIPT USED TO ONLY COVER --norm-write int / --attn-write int
# (the two tiers the RCA above was about). docs/fixed-tier-precision-
# audit.md then re-ran gcc<->Watcom identity for the other ten tier
# values, but on tz/te (153-token zh / 211-token en, .prof/tz.bin +
# te.bin) - the SAME short-input regime that hid the norm-write-int
# defect in its own original verification. That audit's own table shows
# why length matters here: 13 of its 14 configs read "identical" at
# 153/211 tokens, and the one that does not (--norm-write int, tz) is
# exactly the tier already known to need ~150 tokens of specific text
# before it diverges - at 8 tokens even THAT one reads clean. A
# configuration that has never been run long enough to diverge has not
# been shown identical, it has only not yet been shown different. So
# this gate now runs every tier value at the SAME long fixtures
# (norm_int_gate_tokens_{zh,en}.txt, 603 / 661 tokens - the 153/211-token
# reproduction prefixes plus 450 more tokens of real project text) that
# is the one regime known to expose this class of bug, instead of
# trusting a length that already hid one instance of it.
#
# Fixtures are real project text (not synthetic noise), not new to this
# script: build/norm_int_gate_tokens_zh.txt is the 153-token norm-int
# reproduction prefix plus 450 more tokens of real Chinese docs text
# (causal - the prefix computes identically whatever follows it);
# build/norm_int_gate_tokens_en.txt is the matching English/code fixture
# (211 known-clean tokens plus 450 more). Two corpora because the
# original divergence was data-specific: English alone never triggered
# it.
#
# TIER LIST (docs/fixed-tier-precision-audit.md's cross-check of
# cli_main.c's --help text against ops_sched.c's lz_*_select functions -
# 7 independent axes, 12 non-auto values, plus --fixed all as an
# 8th/group axis; nothing missing):
#
#   (float baseline, control - see below)
#   --recur-write fixed
#   --conv-write fixed
#   --attn-write scoring
#   --attn-write wsum
#   --attn-write fixed
#   --attn-write int      (the original RCA's other tier)
#   --epi-write fixed
#   --sig-write fixed
#   --norm-write fixed
#   --norm-write int       (the original RCA's tier)
#   --scalar-fixed fixed
#   --fixed all
#
# BASELINE: --scalar-fixed defaults to "fixed" even under auto -
# lz_scalar_mode() ignores hardware and returns 1 unless told "float"
# explicitly (every other axis's auto asks lz_soft_float_target(), which
# resolves to float on any SSE-capable host, but scalar's auto does not
# ask). So "pass no flags" is NOT a float baseline; the baseline row
# below passes --scalar-fixed float explicitly, same as every other row
# does to isolate its own tier (see ISOLATION below). It runs FIRST and
# is reported as a distinct FAIL if it ever diverges: every other row's
# meaning depends on gcc and Watcom agreeing on the float path they both
# fall back to, so a baseline divergence makes the rest of the run
# uninterpretable and the gate stops there instead of printing 24 more
# lines that cannot be trusted either way.
#
# ISOLATION: every non-baseline, non-group tier row also passes
# --scalar-fixed float alongside its own tier flag. Every OTHER axis's
# auto already resolves to float on this host (see above), so this is
# only needed for scalar - but it is added uniformly so the isolation
# rule does not depend on which host runs the gate. --scalar-fixed fixed
# and --fixed all are the two exceptions: they do not get the extra
# flag because scalar being fixed (or everything being fixed) IS what
# they are testing, and a per-tier flag WINS over --fixed regardless of
# argument order, so appending --scalar-fixed float to a --fixed all row
# would silently re-float scalar and make that row test only six axes.
#
# --dump-nll is unusable for cross-compiler comparison: its exp()/log()
# are MinGW libm vs Watcom libm and differ even when the engine's raw
# logits are identical (docs/fixed-tier-precision-audit.md 3). Compare
# --dump-all-logits raw bytes only, as before.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP (toolchain or model fixture missing)
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
MAKE=${MAKE:-make}

# ---- preflight: every reason to stop here is a SKIP, not a FAIL --------
if ! command -v gcc >/dev/null 2>&1; then
    echo "fixed_tier_xcompiler_gate: SKIP - gcc not found"
    exit 2
fi
if [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
    :
elif [ -x /mnt/c/WATCOM/binl64/wcc386 ]; then
    :
else
    echo "fixed_tier_xcompiler_gate: SKIP - wcc386 not found (checked binnt64/binl64)"
    exit 2
fi

# kmr20 is a quantized L98Z-format export (hidden 512, 8 layers
# LLLALLLA, the same architecture family as models/kunmoe-v2) kept
# outside the repo alongside the other named checkpoints ci.sh's model
# sweep already uses - not reproducible from a recipe in this tree, same
# as those. LZ_NORM_INT_GATE_MODEL overrides the path directly (kept
# under its original name - the two tiers it was named for are still in
# here, and nothing outside this script needs to know it grew).
MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
MODEL=${LZ_NORM_INT_GATE_MODEL:-$MODELS_ROOT/kmr20}
if [ ! -d "$MODEL" ]; then
    echo "fixed_tier_xcompiler_gate: SKIP - model not found: $MODEL" \
         "(set LZ_NORM_INT_GATE_MODEL or MODELS_ROOT)"
    exit 2
fi

TOK_ZH="$ROOT/build/norm_int_gate_tokens_zh.txt"
TOK_EN="$ROOT/build/norm_int_gate_tokens_en.txt"
for f in "$TOK_ZH" "$TOK_EN"; do
    if [ ! -f "$f" ]; then
        echo "fixed_tier_xcompiler_gate: SKIP - fixture missing: $f"
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
    echo "fixed_tier_xcompiler_gate: ERROR 'make native' failed"
    tail -n 20 "$WORK/gcc.build.log"
    exit 2
fi
GCC_BIN="$ROOT/llama98.exe"
[ -x "$GCC_BIN" ] || GCC_BIN="$ROOT/llama98"
if [ ! -x "$GCC_BIN" ]; then
    echo "fixed_tier_xcompiler_gate: ERROR 'make native' produced no binary"
    exit 2
fi

echo "-- Watcom build --"
if ! "$MAKE" watcom-cli >"$WORK/wat.build.log" 2>&1; then
    echo "fixed_tier_xcompiler_gate: ERROR 'make watcom-cli' failed"
    tail -n 20 "$WORK/wat.build.log"
    exit 2
fi
WAT_BIN="$ROOT/llama98-w.exe"
if [ ! -f "$WAT_BIN" ]; then
    echo "fixed_tier_xcompiler_gate: ERROR 'make watcom-cli' produced no binary"
    exit 2
fi

# ---- run one (tier, corpus) pair on both toolchains, cmp the raw
# --dump-all-logits bytes. ----------------------------------------------
FAIL=0
run_one() {
    local tier="$1" label="$2" tokfile="$3"
    local safe gdump wdump glog wlog
    safe=$(printf '%s' "$tier $label" | tr ' -' '__')
    gdump="$WORK/g_$safe.bin"; wdump="$WORK/w_$safe.bin"
    glog="$WORK/g_$safe.log"; wlog="$WORK/w_$safe.log"

    if ! "$GCC_BIN" "$MODEL" --tokens-file "$tokfile" $tier \
            --dump-all-logits "$gdump" >"$glog" 2>&1; then
        echo "FAIL: $tier / $label - gcc run failed:"
        tail -n 10 "$glog"
        FAIL=1
        return 1
    fi
    if ! "$WAT_BIN" "$MODEL" --tokens-file "$tokfile" $tier \
            --dump-all-logits "$wdump" >"$wlog" 2>&1; then
        echo "FAIL: $tier / $label - watcom run failed:"
        tail -n 10 "$wlog"
        FAIL=1
        return 1
    fi
    if [ ! -s "$gdump" ] || [ ! -s "$wdump" ]; then
        echo "FAIL: $tier / $label - one or both --dump-all-logits files are empty"
        FAIL=1
        return 1
    fi
    if ! cmp -s "$gdump" "$wdump"; then
        local ndiff sz
        ndiff=$(cmp -l "$gdump" "$wdump" | wc -l)
        sz=$(wc -c < "$gdump")
        echo "FAIL: $tier / $label - gcc/watcom --dump-all-logits differ ($ndiff of $sz bytes)"
        cmp "$gdump" "$wdump"
        FAIL=1
        return 1
    fi
    echo "PASS: $tier / $label ($(wc -c < "$gdump") bytes, $(wc -l < "$tokfile") tokens)"
    return 0
}

# ---- baseline first, and its failure reported distinctly: every other
# row below assumes gcc and Watcom already agree on the float path they
# both fall back to for the axes the row under test does not touch. ----
BASELINE="--scalar-fixed float"
echo "-- baseline (float control) --"
run_one "$BASELINE" zh "$TOK_ZH"
BASE_ZH=$?
run_one "$BASELINE" en "$TOK_EN"
BASE_EN=$?
if [ "$BASE_ZH" -ne 0 ] || [ "$BASE_EN" -ne 0 ]; then
    echo "fixed_tier_xcompiler_gate: FAIL - BASELINE (float control) diverged;" \
         "every tier result below would be uninterpretable, not running them"
    exit 1
fi

# Isolated: tested tier plus --scalar-fixed float so scalar (the one
# axis whose auto is not a float-resolving auto) cannot also move and
# blur attribution. Order matches --help / the audit's table.
TIERS_ISOLATED=(
    "--recur-write fixed"
    "--conv-write fixed"
    "--attn-write scoring"
    "--attn-write wsum"
    "--attn-write fixed"
    "--attn-write int"
    "--epi-write fixed"
    "--sig-write fixed"
    "--norm-write fixed"
    "--norm-write int"
)
for tier in "${TIERS_ISOLATED[@]}"; do
    run_one "$tier --scalar-fixed float" zh "$TOK_ZH"
    run_one "$tier --scalar-fixed float" en "$TOK_EN"
done

# Group/scalar-self tiers: NOT isolated with an extra --scalar-fixed
# float - that would win over --fixed all's own group setting (per-tier
# flags win regardless of order) and silently test six axes instead of
# seven, or make --scalar-fixed fixed test nothing at all.
TIERS_ASIS=(
    "--scalar-fixed fixed"
    "--fixed all"
)
for tier in "${TIERS_ASIS[@]}"; do
    run_one "$tier" zh "$TOK_ZH"
    run_one "$tier" en "$TOK_EN"
done

if [ "$FAIL" -ne 0 ]; then
    echo "fixed_tier_xcompiler_gate: FAIL"
    exit 1
fi
echo "fixed_tier_xcompiler_gate: PASS"
exit 0
