#!/bin/bash
# cache_idx aliasing for CSA2 REINDEX/REUSE-mode layers - Part 5 Task
# 5.3 of the DeepSeek-V4.1 report Sec2.3.1 Full/Reindex/Reuse plan
# (build/csa2_modes_config_gate.sh landed the schema half; forward_csa2.c
# reading csa2_mode at all is a later, separate task's mode dispatch -
# this gate covers ONLY forward_state.c's cache_idx fill).
#
# No model dir needed - build/csa2_cache_idx_alias_probe.c builds its
# own 4-layer LZModel fixture by hand and calls the real lz_state_alloc
# straight out of src/forward_state.c - so, like ced_proj_arm_gate.sh,
# this needs the whole engine body (lz_state_alloc's own helpers span
# ops_quant.c/ops_norm.c/fwht.c etc.), not just one translation unit.
#
# THE CONTROL is a MUTATION, not a git-history diff, same reason every
# other gate on this plan gives: a git-ref-pinned "before" copy goes
# stale the day this fix lands and HEAD catches up to the working tree.
# The mutation reverts forward_state.c's cache_idx loop to the OLD
# unconditional "every full-attention layer gets a fresh nf++" form,
# in a scratch copy, and re-runs the SAME probe against it. Expected
# fallout, worked out from what the aliasing feeds:
#   - Case A's three aliasing checks (layer 2 aliases 0, layer 3
#     aliases 2, layer 3 transitively aliases 0) turn FAIL - reverted,
#     layers 0/2/3 each get their own fresh slot (0/1/2), so none of
#     the three equalities hold any more.
#   - Every other check (case A's cache_idx-is-allocated and layer-1
#     independence, case B's control distinctness and layer-1 checks)
#     must stay PASS - case B never touched csa2_mode's REINDEX/REUSE
#     branch even before this fix, and layer 1 was never numbered by
#     nf either way.
# A mutant where the three aliasing checks stay PASS means the probe
# cannot see this defect at all - reported as a gate FAILURE. A mutant
# where any OTHER check also turns FAIL means the probe's cases are not
# independent, which is a probe defect in its own right - also a gate
# FAILURE, not folded into "the mutation worked".
#
# Every source file is copied into the scratch dir (not just
# forward_state.c/forward.h) for the quote-include resolution reason
# json_offset_gate.sh documents: `#include "forward.h"` resolves
# against forward_state.c's own directory before -I is ever consulted.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler, no python3).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_cache_idx_alias_gate: SKIP - no C compiler"
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_cache_idx_alias_gate: SKIP - no python3"
    exit 2
}

# The one authoritative copy of both forms of the loop - passed into the
# Python mutator below through the environment rather than retyped in
# its heredoc, so there is exactly one place that has to agree with
# forward_state.c's actual text, not two.
NEW_BLOCK='    for (l = 0; l < c->n_layers; l++) {
        if (c->layer_types[l] != LZ_LT_FULL) {
            s->cache_idx[l] = nl++;
        } else if (m->layers[l].csa2_mode != LZ_CSA2_FULL) {
            s->cache_idx[l] = s->cache_idx[c->csa2_source[l]];
        } else {
            s->cache_idx[l] = nf++;
        }
    }'
OLD_BLOCK='    for (l = 0; l < c->n_layers; l++)
        s->cache_idx[l] = (c->layer_types[l] == LZ_LT_FULL) ? nf++ : nl++;'
export NEW_BLOCK OLD_BLOCK

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2cacheidxaliasgate.$$)
mkdir -p "$W/fixed" "$W/mutant"
trap 'rm -rf "$W"' EXIT

cp src/*.c src/*.h "$W/fixed/" 2>/dev/null
cp src/*.c src/*.h "$W/mutant/" 2>/dev/null

N=$(grep -Fc "s->cache_idx[l] = s->cache_idx[c->csa2_source[l]];" "$W/fixed/forward_state.c")
if [ "$N" != "1" ]; then
    echo "csa2_cache_idx_alias_gate: FAIL - expected exactly 1 occurrence of" \
         "the aliasing assignment in forward_state.c, found $N (the gate no" \
         "longer matches the source)"
    exit 1
fi

python3 - "$W/mutant/forward_state.c" <<'PYEOF' || exit 1
import os, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
new_block = os.environ["NEW_BLOCK"]
old_block = os.environ["OLD_BLOCK"]
if src.count(new_block) != 1:
    print("csa2_cache_idx_alias_gate: FAIL - could not find exactly one copy "
          "of the current aliasing block to mutate", file=sys.stderr)
    sys.exit(1)
src = src.replace(new_block, old_block, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

if grep -qF "s->cache_idx[l] = s->cache_idx[c->csa2_source[l]];" "$W/mutant/forward_state.c"; then
    echo "csa2_cache_idx_alias_gate: FAIL - could not mutate the aliasing" \
         "block in the mutant's forward_state.c"
    exit 1
fi

# The whole engine body (Makefile's $(ENG)) plus the MMX/SSE2/AVX2
# objects every gcc build links unconditionally - same list
# ced_proj_arm_gate.sh uses, for the same reason (its own header
# comment): lz_state_alloc's helpers reach ops_quant.c/ops_norm.c/
# ops_gdn.c/fwht.c, whose dispatch code references those symbols
# unconditionally regardless of which kernel tier is selected.
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
            echo "csa2_cache_idx_alias_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    for f in $ENGX_AVX2; do
        $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_cache_idx_alias_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    if ! $CC -O0 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/csa2_cache_idx_alias_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_cache_idx_alias_gate: FAIL - probe did not build/link against $label"
        tail -n 30 "$dir/cc.log"
        return 2
    fi
    ( cd "$dir" && ./probe > run.log 2>&1 )
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
    echo "csa2_cache_idx_alias_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

echo ""
echo "-- mutant tree: aliasing reverted to unconditional nf++ --"
build_and_run "$W/mutant" "the unconditional-nf++ mutant"
MUT_RC=$?
if [ "$MUT_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT_RC" -eq 0 ]; then
    echo "csa2_cache_idx_alias_gate: FAIL - the mutant PASSED; the probe" \
         "cannot see this defect"
    RC=1
else
    EXPECT_FAIL=(
        "FAIL case A: layer 2 (reindex, source 0) aliases layer 0's cache_idx"
        "FAIL case A: layer 3 (reuse, source 2) aliases layer 2's cache_idx"
        "FAIL case A: transitively, layer 3 lands on layer 0's SAME physical"
    )
    EXPECT_STAY_PASS=(
        "PASS case A: cache_idx is allocated"
        "PASS case A: layer 1 (LINEAR) gets its own independent nl-numbering"
        "PASS case B: default mode (no aliasing) - layers 0/2/3 get MUTUALLY"
        "PASS case B: layer 1 (LINEAR) is unaffected by csa2_mode either way"
    )
    missing=0
    for line in "${EXPECT_FAIL[@]}"; do
        grep -qF "$line" "$W/mutant/run.log" || {
            echo "csa2_cache_idx_alias_gate: FAIL - mutant did not turn FAIL on:" \
                 "$line"
            missing=1
        }
    done
    for line in "${EXPECT_STAY_PASS[@]}"; do
        grep -qF "$line" "$W/mutant/run.log" || {
            echo "csa2_cache_idx_alias_gate: FAIL - mutant lost an unrelated PASS" \
                 "(checks are not independent): $line"
            missing=1
        }
    done
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant failed exactly the three aliasing checks, and no others --"
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_cache_idx_alias_gate: FAIL"
    exit 1
fi
echo "csa2_cache_idx_alias_gate: PASS"
exit 0
