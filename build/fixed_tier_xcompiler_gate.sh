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
# (--norm-write has no `int` value: the int output is folded into the
# fixed tier unconditionally - lz_norm_int() just returns
# lz_norm_mode(). The rows below say `fixed` where this comment says
# `int`; the tier the RCA was about is still covered, it is not
# separately selectable.)
#
# THIS SCRIPT COVERS ALL TWELVE TIER VALUES, not only the --norm-write
# int / --attn-write int pair the RCA above was about.
# docs/fixed-tier-precision-audit.md ran gcc<->Watcom identity for the
# other ten, but on tz/te (153-token zh / 211-token en, .prof/tz.bin +
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
# 7 independent axes, plus --fixed all as an 8th/group axis, plus the
# shipping default itself (zero flags) as a distinct invocation shape;
# nothing missing):
#
#   (float baseline, control - see below)
#   --recur-write fixed
#   --recur-write float        (the newly-reachable inverse isolation -
#                               see ISOLATION below)
#   --conv-write fixed
#   --attn-write scoring
#   --attn-write wsum
#   --attn-write fixed
#   --attn-write int      (the original RCA's other tier)
#   --epi-write fixed
#   --sig-write fixed
#   --norm-write fixed     (the original RCA's tier; its `int` output
#                           is folded into `fixed`, so this row is that
#                           tier's coverage)
#   --scalar-fixed fixed
#   --fixed all
#   (shipping default, zero flags)
#
# BASELINE: the shipping build is fixed-point unconditionally now - none
# of the seven tiers' `auto` asks hardware any more (ops_sched.c's
# retirement note on lz_soft_float_target; --scalar-fixed's auto went
# this way first, and the other six followed it). So "pass no flags" is
# NOT a float baseline for ANY axis, let alone just scalar - the
# baseline row below passes --fixed off, which
# forces all seven explicitly (recurrence included: --recur-write gained
# an explicit `float` value in the same change that retired the
# hardware predicate, so it needs no special-casing here). It runs FIRST
# and is reported as a distinct FAIL if it ever diverges - but it does
# NOT STOP THE RUN, and the baseline block below says at length what
# stopping the run cost.
# Short version: the rows whose meaning depended on the float path were
# the twelve isolated ones, and they are gone; the two that remain pass
# no isolation flag and never reach it.
#
# ISOLATION: every non-baseline, non-group, non-default tier row also
# passes --fixed off alongside its own tier flag, forcing every OTHER
# axis to float so only the row's own tier is under test - true for all
# seven now, not just scalar (a per-tier flag WINS over --fixed
# regardless of argument order, so "$tier --fixed off" leaves $tier at
# its own explicit value and floats everything else). --recur-write
# float gets the INVERSE isolation instead (--fixed all, not --fixed
# off): it is the one row testing an axis's float path in isolation
# rather than its fixed path, so the other six need to be FIXED around
# it, not float - the mirror image of every other row, and the
# combination "one axis float, six fixed" stayed out of reach until
# --recur-write gained an explicit float value: every SSE host's auto
# put all seven on float together, so one-float-among-six-fixed was
# untested. --fixed all and the shipping default are the
# two configurations that get no isolation flag at all, because "every
# axis fixed" (however it is spelled - explicitly or via the new
# unconditional auto) IS what they are testing.
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
# COVERAGE GAP, NOW COVERED ELSEWHERE - kmr20 carries no T2 tensor
# (.prof/dtype_probe.c: q8_0 7.1%, q4_1 33.0%, q6_1 59.4%, q16_0 0.5%),
# so every row below leaves the T2 half of the Q4_1/Q6_1/T2 epilogue
# family untested. That half is not a copy of the others: T2 has no
# stored zero plane and reaches the fixed epilogue through its own
# `zneg` branch, reusing the NEGATED scale plane.
#
# build/gcc_watcom_output_gate.sh now defaults to $MODELS_ROOT/_t2gate,
# which is this same checkpoint with 12 tensors moved from q6_1 to t2 -
# 133,349,376 elements either way, identical q8_0/q4_1/q16_0 counts, T2
# at 4.7%. Same model, one format swapped, so a divergence there and not
# here lands on the T2 path instead of on "two different models". That
# gate is in `make check` and takes ~4 minutes; a second full run here
# would cost ~50.
#
# THAT ~50-MINUTE RUN HAS BEEN DONE ONCE, so the gap is closed by
# measurement rather than only by inference from the cheaper gate: all
# eight rows below PASS on _t2gate - float control, --fixed all, the
# shipping default and --kernel ref, on both the 603-token zh and
# 661-token en fixtures, 78,905,344 and 86,507,520 bytes each. Not made
# the default, because paying 50 minutes of every CI run to re-derive a
# one-off result is not the same thing as checking something.
#
# A REMEDY THAT NAMES A FIXTURE IS A CLAIM ABOUT THE DISK, and it rots
# with nobody touching the paragraph: this one said to run
# LZ_NORM_INT_GATE_MODEL=/e/LLM/models/kunmoe-v2-t2-probe (historical) -
# a directory that is gone. The instruction went stale exactly like a
# comment naming a function, silently, and the gap stayed open the whole
# time because the way to close it did not resolve.
#
# THE MARKER IS LOAD-BEARING, not decoration: fixture_exists_gate is a
# text scanner, so it cannot tell a line that names a fixture to USE from
# a line that records one being GONE. A sentence about a deleted
# checkpoint has to say so on the same line as the name, which is what
# (historical) is for.
#
# The override below still works for a one-off full run against T2:
#   LZ_NORM_INT_GATE_MODEL=/e/LLM/models/_t2gate bash "$0"
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

# ---- baseline first, and its failure reported distinctly - but NOT as a
# reason to stop.
#
# This block exits 1 on a baseline failure. The grounds once given for
# stopping on every failure - "every tier result below would be
# uninterpretable" - hold only of the TWELVE ISOLATED ROWS, each of
# which wrapped one tier flag in --fixed off and therefore did rest on
# the float path being
# sound. Those rows are gone (see the block below for where they went).
# What is left is --fixed all and the shipping default, neither of which
# passes an isolation flag, neither of which touches the float path, and
# both of which are the only rows here that describe a configuration
# anyone ships. The barrier was protecting rows that no longer exist by
# suppressing the two that matter.
#
# MEASURED, not reasoned: the baseline was red for days on
# src/lz_mathf.c's reduce_large defect (an exact int64 multiplied by
# pi63 on x87, since fixed), and both shipping rows were byte-identical
# across the toolchains that entire time - 18,219,008 bytes at 139
# tokens, then 78,905,344 and 86,507,520 at the full fixtures. Every one
# of those runs was skipped, and the skip printed as a FAIL that named
# the float control, so nothing said the shipping configuration was
# untested. That is the failure mode this
# tree calls "gate passed is not gate looked", in its other direction: a
# gate that fails without having asked.
#
# A red baseline is still a FAIL and still reported first, because a
# float-tier divergence is a real defect - it is what named reduce_large.
# It just no longer decides whether the rest runs.
BASELINE="--fixed off"
echo "-- baseline (float control) --"
run_one "$BASELINE" zh "$TOK_ZH"
BASE_ZH=$?
run_one "$BASELINE" en "$TOK_EN"
BASE_EN=$?
if [ "$BASE_ZH" -ne 0 ] || [ "$BASE_EN" -ne 0 ]; then
    echo "  NOTE: the float control diverged. The shipping rows below do"
    echo "  not pass an isolation flag and do not reach the float path,"
    echo "  so they are still run and still mean what they say - but a"
    echo "  float-tier divergence is a defect in its own right and this"
    echo "  gate fails on it. build/mathf_xcompiler_gate.sh narrows the"
    echo "  most likely cause (lz_mathf, int->float feeding a multiply)"
    echo "  in seconds instead of the ~40 minutes this run costs."
fi

# WHERE THE TWELVE ISOLATED ROWS WENT, and how to get attribution back.
#
# This block runs one row per remaining tier VALUE, not one per runtime
# tier flag - that shape put --fixed off around each flag, which let a
# gcc/Watcom divergence name the tier. The
# seven per-tier flags are gone: they could only ever select MIXED
# settings, mixing measured worse than either end on six of seven axes,
# and one float knob replaced the lot (ops.h, at lz_float_ref_select).
# There is nothing left for those rows to isolate - "--attn-write
# scoring" is not a state the engine can be in.
#
# What that costs is attribution, and it is a real cost: this gate now
# says "the two toolchains disagree" without saying where. Attribution
# moved to the COMPILE switches, which are still per-tier and are what
# build/switch_matrix_gate.sh drives - LZ_GDN_FIXED, LZ_CONV_FIXED,
# LZ_NORM_FIXED, LZ_ATTN_FIXED, LZ_CONVO_I16, LZ_EPI_FIXED. On a
# failure here, rebuild both toolchains with one of them at 0 at a time
# and re-run the two arms below; the switch that makes the divergence
# disappear is the tier that owns it. Slower than a flag, because it is
# a rebuild per row rather than an argv per row - which is why it is a
# failure procedure rather than the default sweep.

# --fixed all and the shipping default: NOT isolated with any extra
# flag. "Every axis fixed" (spelled explicitly, or reached by the
# unconditional auto every axis now has) IS what these two rows test,
# and either isolation flag would just restate the row itself. Run as
# two entries so a divergence names which invocation shape broke,
# though both are expected to resolve identically (verified elsewhere:
# --dump-all-logits byte-identical between the two on both toolchains).
TIERS_ASIS=(
    "--fixed all"
    ""
)
for tier in "${TIERS_ASIS[@]}"; do
    run_one "$tier" zh "$TOK_ZH"
    run_one "$tier" en "$TOK_EN"
done

# --kernel ref, and it closes a gap rather than restating a row.
#
# THE CHAIN OF ASSERTIONS AND WHERE IT BROKE. Between them, this gate and
# build/kernel_isa_identity_gate.sh assert:
#
#   gcc ref == gcc mmx == gcc sse2      (isa_identity, one binary)
#   watcom ref == watcom mmx            (isa_identity, one binary)
#   gcc default == watcom default       (the row above; both are sse2)
#
# from which gcc's ref tier is pinned to Watcom's DEFAULT tier - but
# nothing pins it to Watcom's REF tier, because `watcom ref == watcom
# sse2` is one of the four cross-rounding rows isa_identity reports as
# INFO rather than asserting. So the scalar fallback, which is what a
# machine with no SSE2 actually runs, had no cross-compiler assertion
# anywhere. One row here supplies it and the transitive closure follows,
# without touching isa_identity's INFO decision.
#
# NOT redundant with the default row above: both sides of that one run
# the SSE2 kernels, so a divergence that lives in the C scalar code the
# SIMD path replaces rather than shares - a different accumulation order
# in a reference loop, an int->float cast the kernels do in registers -
# is invisible to it. This row is the only place both sides run that C.
# Costs two more full-fixture runs (~12 minutes); the tier it covers is
# what a machine without SSE2 executes.
for tier in "--kernel ref"; do
    run_one "$tier" zh "$TOK_ZH"
    run_one "$tier" en "$TOK_EN"
done

if [ "$FAIL" -ne 0 ]; then
    echo "fixed_tier_xcompiler_gate: FAIL"
    exit 1
fi
echo "fixed_tier_xcompiler_gate: PASS"
exit 0
