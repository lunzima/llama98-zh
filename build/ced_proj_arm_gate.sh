#!/bin/bash
# CED's engine-side "proj" arm (report Sec2.2 Eq(1)): ced_kv/s->ced_hs.
#
# No model dir needed - build/ced_proj_arm_probe.c builds its own tiny
# LZModel fixture and calls the real lz_ced_kv_target/lz_forward
# straight out of src/model.c/src/forward.c - so this sits with the
# other fast, build-and-run gates (json-offset-gate, kv-share-ced-
# boundary-gate) rather than the model-dir-gated ones (kv-share-gate,
# ced-gate).
#
# UNLIKE kv_share_ced_boundary_gate.sh's own probe, this one drives a
# REAL forward pass (lz_forward on a hand-built 3-layer full-attention
# model), so it needs the whole engine body, not just model.c's config
# helpers - the full $ENG list plus the MMX/SSE2/AVX2 objects the
# dispatch code inside ops_quant.c/ops_matmul.c/ops_norm.c/ops_gdn.c/
# fwht.c references unconditionally (LZ_MMX_TU/LZ_SSE2_TU/LZ_AVX2_TU
# only gate whether OTHER files call into them, not whether the symbols
# exist - defining them to 0 to skip the three SIMD source files still
# leaves undefined references).
#
# THE TAP. Part 2 of the probe needs to see s->x right after layer 0
# finishes, inside forward_chunk's own per-layer loop - a quantity
# forward.c already writes out through its own LZ_TAP macro (the "out"
# tag) but never returns to a caller. forward.h's own comment on
# LZ_TAP says "a dump tool redefines it" - this is that mechanism, the
# same one .prof's own dump probes use, applied here via a tiny -include
# header rather than a permanent third committed file: forward.h's
# `#ifndef LZ_TAP` guard means whichever translation unit sees this
# header before forward.h gets ced_probe_tap instead of the default
# no-op, and the probe.c file both defines ced_probe_tap and declares
# it (matching signatures) for its own use.
#
# THE CONTROL is a MUTATION, not a git-history diff, for the same
# reason json_offset_gate.sh gives: a git-ref-pinned "before" copy goes
# stale the day this fix lands and HEAD catches up to the working tree.
# The mutation replaces lz_ced_kv_target's own return expression with
# a bare `return 0;` in a scratch copy of model.c and re-runs the SAME
# probe against it. Expected fallout, worked out from what the mutated
# function feeds:
#   - Part 1's row 2 (ced_kv=proj, ced_split=1): layers 1/2/3 stop
#     being targets - three rows turn FAIL.
#   - Part 1's row 5 final assert (CED claims layer 3, attn_kv_share
#     does not): the "CED claims it" half turns FAIL.
#   - Part 2's h_S-capture: the capture is itself gated on
#     lz_ced_kv_target(c, ced_split), so it never fires - s->ced_hs
#     stays at its calloc-zero value and the byte-identity check
#     against the tapped layer-0 output turns FAIL.
#   - Part 2's "logits differ" check: with the capture and the
#     resolution both dead, ced_kv=proj and ced_kv=off run IDENTICAL
#     computations (kv_in stays NULL either way) - the two logit sets
#     stop differing and this turns FAIL too.
#   - Part 1's rows 1/3/4 (all already expect 0) and Part 3's refusal
#     checks (config.json parsing, unrelated to this function) must
#     stay PASS - a mutant that breaks those as well means the probe's
#     checks are not independent of each other.
# A mutant where none of the five expected checks turn FAIL means the
# probe cannot see this defect at all - reported as a gate FAILURE.
#
# Every source file is copied into the scratch dir (not just model.c/
# model.h) for the quote-include resolution reason json_offset_gate.sh
# documents: `#include "model.h"` resolves against model.c's own
# directory before -I is ever consulted, so a mixed-directory build
# would silently link a fixed model.o against a stale model.h or vice
# versa.
#
# PART 3 of the probe (the config.json refusal check) is optional and
# best-effort: it needs a REAL config.json this tree does not own
# (kunkun98's own, which already carries "ced_kv": "proj" - this
# checkpoint's training-side config expects the substitution this task
# adds). Same default path and skip-if-absent posture as
# json_offset_gate.sh's own config.json regression (LZ_JSON_GATE_CONFIG
# env var) - its absence SKIPS that one part of the probe, not the gate.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "ced_proj_arm_gate: SKIP - no C compiler"
    exit 2
}

CFG="${LZ_JSON_GATE_CONFIG:-/e/LLM/models/kunkun98/config.json}"

ANCHOR='return c->ced_kv && c->ced_split > 0 && c->ced_split < c->n_layers && li >= c->ced_split;'

W=$(mktemp -d 2>/dev/null || echo /tmp/cedprojarmgate.$$)
mkdir -p "$W/fixed" "$W/mutant"
trap 'rm -rf "$W"' EXIT

cp src/*.c src/*.h "$W/fixed/" 2>/dev/null
cp src/*.c src/*.h "$W/mutant/" 2>/dev/null

N=$(grep -Fc "$ANCHOR" "$W/fixed/model.c")
if [ "$N" != "1" ]; then
    echo "ced_proj_arm_gate: FAIL - expected exactly 1 occurrence of" \
         "lz_ced_kv_target's return line in model.c, found $N (the gate no" \
         "longer matches the source)"
    exit 1
fi
ESCAPED=$(printf '%s' "$ANCHOR" | sed 's/[.[\*^$/&]/\\&/g')
sed -i "s/$ESCAPED/return 0;/" "$W/mutant/model.c"
if grep -qF "$ANCHOR" "$W/mutant/model.c"; then
    echo "ced_proj_arm_gate: FAIL - could not mutate lz_ced_kv_target in" \
         "the mutant's model.c"
    exit 1
fi

# The tap header: forward.h's LZ_TAP, redefined for both trees so
# Part 2 of the probe can see forward.c's own "out" tag without a
# permanent third committed file.
cat > "$W/ced_tap.h" <<'EOF'
#ifndef CED_TAP_H
#define CED_TAP_H
void ced_probe_tap(const char *tag, int li, const float *ptr, int n);
#ifndef LZ_TAP
#define LZ_TAP(tag, li, ptr, n) ced_probe_tap(tag, li, ptr, n)
#endif
#endif
EOF

# The whole engine body (Makefile's $(ENG)) plus the MMX/SSE2/AVX2
# objects every gcc build links unconditionally (see this file's own
# header comment for why they cannot just be left out).
ENGX="err compat lfn json safetensors model ops ops_kernel ops_rope ops_moe ops_epi ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn ops_conv1d fwht forward forward_attn forward_moe forward_ssm forward_kda forward_engram forward_csa2 forward_mtp forward_rope forward_ckpt forward_state sampler generate lz_mathf lz_bf16 lz4d tokenizer unicode chat gbk cpucheck"
ENGX_MMX="ops_mmx ops_mmx_sse ops_sse2"
ENGX_AVX2="ops_avx2"

RC=0

build_and_run() {
    local dir="$1" label="$2"
    local f
    for f in $ENGX_MMX; do
        $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mmmx -msse -msse2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "ced_proj_arm_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    for f in $ENGX_AVX2; do
        $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "ced_proj_arm_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    if ! $CC -O0 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -include "$W/ced_tap.h" \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/ced_proj_arm_probe.c -lm 2>"$dir/cc.log"; then
        echo "ced_proj_arm_gate: FAIL - probe did not build/link against $label"
        tail -n 30 "$dir/cc.log"
        return 2
    fi
    "$dir/probe" "$CFG" > "$dir/run.log" 2>&1
    local rc=$?
    cat "$dir/run.log"
    return $rc
}

echo "-- current tree (must PASS) --"
build_and_run "$W/fixed" "the current (fixed) tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then
    exit 1
elif [ "$FIXED_RC" -ne 0 ]; then
    echo "ced_proj_arm_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

echo ""
echo "-- mutant tree: lz_ced_kv_target forced to return 0 --"
build_and_run "$W/mutant" "the lz_ced_kv_target mutant"
MUT_RC=$?
if [ "$MUT_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT_RC" -eq 0 ]; then
    echo "ced_proj_arm_gate: FAIL - the mutant PASSED; the probe cannot see" \
         "this defect"
    RC=1
else
    EXPECT_FAIL=(
        "FAIL ced_kv=proj: layer 1 (== ced_split) is a target"
        "FAIL ced_kv=proj: layer 2 (decoder) is a target"
        "FAIL ced_kv=proj: layer 3 (decoder) is a target"
        "FAIL layer 3 (decoder, full_attention): CED claims it, attn_kv_share does not"
        "FAIL s->ced_hs is byte-identical to layer 0's own residual output"
        "FAIL ced_kv=proj vs off (same weights, same token): final logits differ"
    )
    EXPECT_STAY_PASS=(
        "PASS ced_kv=off: layer 0 is not a target"
        "PASS ced_kv=off: layer 1 (== ced_split) is not a target"
        "PASS ced_split==0: not a decoder boundary, layer 0 is not a target"
        "PASS ced_split==n_layers: not a decoder boundary, layer 3 is not a target"
    )
    missing=0
    for line in "${EXPECT_FAIL[@]}"; do
        grep -qF "$line" "$W/mutant/run.log" || {
            echo "ced_proj_arm_gate: FAIL - mutant did not turn FAIL on:" \
                 "$line"
            missing=1
        }
    done
    for line in "${EXPECT_STAY_PASS[@]}"; do
        grep -qF "$line" "$W/mutant/run.log" || {
            echo "ced_proj_arm_gate: FAIL - mutant lost an unrelated PASS" \
                 "(rows are not independent): $line"
            missing=1
        }
    done
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant failed exactly the checks lz_ced_kv_target feeds," \
             "and no others --"
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "ced_proj_arm_gate: FAIL"
    exit 1
fi
echo "ced_proj_arm_gate: PASS"
exit 0
