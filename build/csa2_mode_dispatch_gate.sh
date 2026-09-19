#!/bin/bash
# Full/Reindex/Reuse mode dispatch in forward_csa2.c - Part 5 Task 5.4
# of the DeepSeek-V4.1 report Sec2.3.1 Full/Reindex/Reuse plan
# (build/csa2_modes_config_gate.sh landed the schema half, build/
# csa2_cache_idx_alias_gate.sh the cache_idx aliasing half - this is
# the task that makes forward_csa2.c actually read L->csa2_mode at all).
#
# No model dir needed - build/csa2_mode_dispatch_probe.c builds its own
# hand-crafted LZModel/LZRunState fixtures with real (synthetic) weight
# tensors and calls the real forward_csa2() directly, out of
# src/forward_csa2.c - so, like ced_proj_arm_gate.sh and
# csa2_cache_idx_alias_gate.sh, this needs the whole engine body
# (lz_state_alloc's own helpers span ops_quant.c/ops_norm.c/fwht.c
# etc.), not just one translation unit.
#
# THE CONTROL is TWO INDEPENDENT MUTATIONS, not a git-history diff, same
# reason every other gate on this plan gives: a git-ref-pinned "before"
# copy goes stale the day this fix lands and HEAD catches up to the
# working tree.
#
#   MUTATION 1, "dispatch reverted": the three csa2_mode guards this
#   task adds (step 2's `if (mode == LZ_CSA2_FULL)`, step 3's
#   `if (mode != LZ_CSA2_REUSE)`, step 6's `... && mode == LZ_CSA2_FULL`)
#   are each replaced with an always-true condition, in a scratch copy -
#   every csa2_on layer runs the pre-this-task, unconditional six-step
#   body again, regardless of what L->csa2_mode says. Expected fallout:
#     - check1's Full-only checks (1a, 1b) MUST stay PASS - a single
#       FULL-mode layer already satisfied every one of these three
#       conditions, so making them unconditional is a no-op for it.
#     - check_chain_soft's REINDEX/REUSE-specific numeric checks (layer
#       B's/C's own published counts and indices, checks 2 and 3a) and
#       check5's batch-vs-unbatched chain identity MUST turn FAIL - a
#       REINDEX layer now also runs steps 2/6 with WRONG-but-present
#       tensors (build/csa2_mode_dispatch_probe.c's own header comment
#       on why REINDEX is given a full tensor set), corrupting the
#       shared ring/pending state its aliased ci shares with its
#       source.
#     - check4b (REINDEX must never read kv_in) MUST turn FAIL - step 2
#       now runs for D2 too, and kv_in is exactly what step 2 reads.
#     - check3b (a REUSE layer owns none of the seven csa2_w_* tensors)
#       is ordered LAST and is not expected to print a PASS line at
#       all: its own C layer (build_layer_reuse_strict) has NO tensors
#       to run FULL's steps 2/3/6 with, so the mutant CRASHES there
#       (a NULL LZTensor.f dereference) - this is the correct, expected
#       outcome, not a probe defect, and build/csa2_mode_dispatch_
#       probe.c's own ok() flushes stdout after every line specifically
#       so this crash does not erase the 19 checks that already ran
#       and printed before it.
#
#   MUTATION 2, "publish buffer collapsed to per-ci": the two `+ tk`
#   terms in forward_csa2.c's own per-token publish-slot pointers
#   (`tok_idx`/`tok_n`, just inside the per-token loop) are removed,
#   reverting csa2_topk_idx_pub/csa2_topk_n_pub to a single (topk)-wide
#   slot PER CI, shared by every token of the current chunk - what this
#   pair would have stayed as if this task had reshaped the allocation
#   (forward_state.c, forward.h) without changing forward_csa2.c's own
#   addressing into it. This targets a narrower, specific claim than
#   mutation 1: not "does mode dispatch happen at all", but "does the
#   publish buffer need a per-TOKEN dimension, or would per-ci alone
#   have sufficed". Expected fallout, verified empirically against this
#   exact probe rather than assumed (a token's own attn_out is
#   overwritten by the next token within the same call, so only the
#   LAST token of any call is ever externally inspectable again - and
#   the last token's own published slot is, by construction, always
#   "the last write" under EITHER addressing scheme, which is why
#   check5's batch-vs-unbatched attn_out comparison, despite being
#   genuinely useful against mutation 1, turns out to be structurally
#   blind to this specific one and correctly stays PASS here):
#     - check_chain_soft's two DIRECT publish-buffer reads turn FAIL -
#       "layer A's own published k_eff grows 0,1,2 across the chunk"
#       (A's own tk=0/1 writes are overwritten by tk=1/2's, landing all
#       three in the same physical slot) and "layer B (reindex) sees
#       enough of A's already-finalized blocks ... for every token of
#       its own loop" (same collapse, one layer downstream).
#     - check1b's own two published-buffer match assertions ALSO turn
#       FAIL: they are not Reindex/Reuse-specific (check1b is a single
#       FULL-mode layer), but they specifically compare batched vs
#       unbatched PUBLISHED CONTENT token for token, which this
#       mutation breaks even for Full-only.
#     - every other check, including check1b's own attn_out comparison,
#       check_chain_soft's "buffer untouched by C" and check5, stays
#       PASS - see this script's own comment above on why the ones that
#       only ever inspect a call's LAST token cannot see this bug.
#
# A mutant where the expected-to-fail lines stay PASS means the probe
# cannot see that defect at all - reported as a gate FAILURE. A mutant
# where any OTHER line also changes means the probe's checks are not
# independent, which is a probe defect in its own right - also a gate
# FAILURE, not folded into "the mutation worked".
#
# Every source file is copied into the scratch dir (not just
# forward_csa2.c/forward_state.c/forward.h) for the quote-include
# resolution reason json_offset_gate.sh documents: `#include "forward.h"`
# resolves against forward_csa2.c's own directory before -I is ever
# consulted.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler, no python3).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "csa2_mode_dispatch_gate: SKIP - no C compiler"
    exit 2
}
command -v python3 >/dev/null 2>&1 || {
    echo "csa2_mode_dispatch_gate: SKIP - no python3"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/csa2modedispatchgate.$$)
mkdir -p "$W/fixed" "$W/mut1" "$W/mut2"
trap 'rm -rf "$W"' EXIT

cp src/*.c src/*.h "$W/fixed/" 2>/dev/null
cp src/*.c src/*.h "$W/mut1/" 2>/dev/null
cp src/*.c src/*.h "$W/mut2/" 2>/dev/null

# Sanity: the lines both mutations target must exist exactly once in
# the current source, or this gate no longer matches the code it names.
for needle in \
    "        if (mode == LZ_CSA2_FULL) {" \
    "        if (mode != LZ_CSA2_REUSE) {" \
    "        if (off == cm - 1 && mode == LZ_CSA2_FULL) {" \
    "        int *tok_idx = pub_idx_base + (size_t)tk * topk;" \
    "        int *tok_n = pub_n_base + tk;"
do
    N=$(grep -Fc "$needle" "$W/fixed/forward_csa2.c")
    if [ "$N" != "1" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - expected exactly 1 occurrence of" \
             "'$needle' in forward_csa2.c, found $N (the gate no longer" \
             "matches the source)"
        exit 1
    fi
done

python3 - "$W/mut1/forward_csa2.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
subs = [
    ("        if (mode == LZ_CSA2_FULL) {\n", "        if (1) {\n"),
    ("        if (mode != LZ_CSA2_REUSE) {\n", "        if (1) {\n"),
    ("        if (off == cm - 1 && mode == LZ_CSA2_FULL) {\n",
     "        if (off == cm - 1) {\n"),
]
for old, new in subs:
    if src.count(old) != 1:
        print("csa2_mode_dispatch_gate: FAIL - mutation 1 could not find "
              "exactly one copy of %r" % old, file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

python3 - "$W/mut2/forward_csa2.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
subs = [
    ("        int *tok_idx = pub_idx_base + (size_t)tk * topk;\n",
     "        int *tok_idx = pub_idx_base;\n"),
    ("        int *tok_n = pub_n_base + tk;\n",
     "        int *tok_n = pub_n_base;\n"),
]
for old, new in subs:
    if src.count(old) != 1:
        print("csa2_mode_dispatch_gate: FAIL - mutation 2 could not find "
              "exactly one copy of %r" % old, file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

if grep -qF "if (mode == LZ_CSA2_FULL) {" "$W/mut1/forward_csa2.c" || \
   grep -qF "if (mode != LZ_CSA2_REUSE) {" "$W/mut1/forward_csa2.c"; then
    echo "csa2_mode_dispatch_gate: FAIL - could not mutate mutant 1's forward_csa2.c"
    exit 1
fi
if grep -qF "pub_idx_base + (size_t)tk * topk" "$W/mut2/forward_csa2.c" || \
   grep -qF "pub_n_base + tk" "$W/mut2/forward_csa2.c"; then
    echo "csa2_mode_dispatch_gate: FAIL - could not mutate mutant 2's forward_csa2.c"
    exit 1
fi

# The whole engine body (Makefile's $(ENG)) plus the MMX/SSE2/AVX2
# objects every gcc build links unconditionally - same list ced_proj_
# arm_gate.sh/csa2_cache_idx_alias_gate.sh use, for the same reason
# (their own header comments): lz_state_alloc's helpers reach
# ops_quant.c/ops_norm.c/ops_gdn.c/fwht.c, whose dispatch code
# references those symbols unconditionally regardless of which kernel
# tier is selected.
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
            echo "csa2_mode_dispatch_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    for f in $ENGX_AVX2; do
        $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 -mavx2 \
            -c -o "$dir/$f.o" "$dir/$f.c" 2>"$dir/cc_$f.log" || {
            echo "csa2_mode_dispatch_gate: FAIL - $f.c did not build against $label"
            tail -n 20 "$dir/cc_$f.log"
            return 2
        }
    done
    if ! $CC -O0 -std=c99 -Wall -Wextra -I"$dir" -ffp-contract=off \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1 \
            -o "$dir/probe" \
            $(for f in $ENGX; do echo "$dir/$f.c"; done) \
            $(for f in $ENGX_MMX $ENGX_AVX2; do echo "$dir/$f.o"; done) \
            build/csa2_mode_dispatch_probe.c -lm 2>"$dir/cc.log"; then
        echo "csa2_mode_dispatch_gate: FAIL - probe did not build/link against $label"
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
    echo "csa2_mode_dispatch_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

check_lines() {
    local logfile="$1" tag="$2"
    shift 2
    local mode="$1"; shift   # "FAIL" or "PASS"
    local line missing=0
    for line in "$@"; do
        grep -qF "$mode $line" "$logfile" || {
            echo "csa2_mode_dispatch_gate: FAIL - $tag did not turn $mode on: $line"
            missing=1
        }
    done
    return $missing
}

echo ""
echo "-- mutant 1: dispatch reverted to unconditional Full-mode body --"
build_and_run "$W/mut1" "mutant 1 (dispatch reverted)"
MUT1_RC=$?
if [ "$MUT1_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT1_RC" -eq 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 1 PASSED; the probe cannot see this defect"
    RC=1
else
    STAY_PASS=(
        "check1a: first-ever token's published k_eff is 0"
        "check1a: k_eff==0 token's attn_out is the exact zero vector"
        "check1b: batch-vs-unbatched - published k_eff matches token for token"
        "check1b: batch-vs-unbatched - published Top-K indices match token for token"
        "check1b: batch-vs-unbatched - last token's attn_out is bit-identical"
        "check_chain: all three layers share the same physical ci (cache_idx aliasing from the prior task still holds)"
        "check_chain: layer A's own published k_eff grows 0,1,2 across the chunk"
        "check_chain: layer B's own Top-K selection is not identical across all three of its own tokens"
        "check2: layer A's own attn_out is untouched by a change to layer B's (unrelated) indexer weights"
        "check4a: layer D1 (FULL, decoder-region CSA2) still reads kv_in in its own step 2"
        "check4b: ...and so is the published k_eff"
    )
    TURN_FAIL=(
        "check_chain: layer B (reindex) sees enough of A's already-finalized blocks (n_vis >= topk) for every token of its own loop to publish a full, non-trivial Top-K (A's whole call already finished before B's began, so n_vis is already 3, constant across B's own tokens)"
        "check_chain: after layer C's (reuse) call, all three tokens' published Top-K selections are still exactly layer B's own"
        "check2: perturbing layer B's own indexer weights changes layer B's own attn_out"
        "check3a: perturbing layer A's compression weights changes layer B's attn_out"
        "check3a: perturbing layer A's compression weights also changes layer C's attn_out"
        "check5: layer C's (reuse) final attn_out over a 3-token chunk processed as one nt=3 call per layer is bit-identical to the same 3 positions processed one at a time (nt=1, layer-major per position)"
        "check4b: layer D2 (reindex, decoder-region CSA2) attn_out is byte-identical whether kv_in is NULL or a poisoned non-NULL"
        "check4b: ...and so is the published Top-K index set"
    )
    missing=0
    check_lines "$W/mut1/run.log" "mutant 1" "PASS" "${STAY_PASS[@]}" || missing=1
    check_lines "$W/mut1/run.log" "mutant 1" "FAIL" "${TURN_FAIL[@]}" || missing=1
    if grep -qF "csa2_mode_dispatch_probe: PASS" "$W/mut1/run.log"; then
        echo "csa2_mode_dispatch_gate: FAIL - mutant 1 printed a final PASS - it" \
             "should have crashed at check3b (layer C there owns none of the" \
             "seven csa2_w_* tensors and the reverted dispatch tries to use them)"
        missing=1
    fi
    if grep -qF "PASS check3b" "$W/mut1/run.log"; then
        echo "csa2_mode_dispatch_gate: FAIL - mutant 1 reached and passed check3b - it" \
             "should have crashed there instead (see this script's own header comment)"
        missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant 1 failed exactly the Reindex/Reuse-specific checks, then crashed at check3b as expected --"
    fi
fi

echo ""
echo "-- mutant 2: publish buffer collapsed from per-(ci, token) to per-ci --"
build_and_run "$W/mut2" "mutant 2 (publish buffer collapsed)"
MUT2_RC=$?
if [ "$MUT2_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT2_RC" -eq 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 2 PASSED; the probe cannot see this defect"
    RC=1
else
    STAY_PASS=(
        "check1a: first-ever token's published k_eff is 0"
        "check1a: k_eff==0 token's attn_out is the exact zero vector"
        "check1b: batch-vs-unbatched - last token's attn_out is bit-identical"
        "check_chain: all three layers share the same physical ci (cache_idx aliasing from the prior task still holds)"
        "check_chain: layer B's own Top-K selection is not identical across all three of its own tokens"
        "check_chain: after layer C's (reuse) call, all three tokens' published Top-K selections are still exactly layer B's own"
        "check2: layer A's own attn_out is untouched by a change to layer B's (unrelated) indexer weights"
        "check2: perturbing layer B's own indexer weights changes layer B's own attn_out"
        "check3a: perturbing layer A's compression weights changes layer B's attn_out"
        "check3a: perturbing layer A's compression weights also changes layer C's attn_out"
        "check5: layer C's (reuse) final attn_out over a 3-token chunk processed as one nt=3 call per layer is bit-identical to the same 3 positions processed one at a time (nt=1, layer-major per position)"
        "check4a: layer D1 (FULL, decoder-region CSA2) still reads kv_in in its own step 2"
        "check4b: layer D2 (reindex, decoder-region CSA2) attn_out is byte-identical whether kv_in is NULL or a poisoned non-NULL"
        "check4b: ...and so is the published k_eff"
        "check4b: ...and so is the published Top-K index set"
        "check3b: layer C (reuse) completed its whole 3-token call without touching any of the seven csa2_w_* tensors it does not own, and its output is finite"
    )
    TURN_FAIL=(
        "check1b: batch-vs-unbatched - published k_eff matches token for token"
        "check1b: batch-vs-unbatched - published Top-K indices match token for token"
        "check_chain: layer A's own published k_eff grows 0,1,2 across the chunk"
        "check_chain: layer B (reindex) sees enough of A's already-finalized blocks (n_vis >= topk) for every token of its own loop to publish a full, non-trivial Top-K (A's whole call already finished before B's began, so n_vis is already 3, constant across B's own tokens)"
    )
    missing=0
    check_lines "$W/mut2/run.log" "mutant 2" "PASS" "${STAY_PASS[@]}" || missing=1
    check_lines "$W/mut2/run.log" "mutant 2" "FAIL" "${TURN_FAIL[@]}" || missing=1
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant 2 failed exactly the direct per-token publish-buffer reads, and no others --"
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL"
    exit 1
fi
echo "csa2_mode_dispatch_gate: PASS"
exit 0
