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
# THE CONTROL is SIX INDEPENDENT MUTATIONS, not a git-history diff, same
# reason every other gate on this plan gives: a git-ref-pinned "before"
# copy goes stale the day this fix lands and HEAD catches up to the
# working tree. Three were written with the task's own first landing;
# three more (3, 4, 5) were added after an independent, second-round
# code review found that the first pass, on its own, could not tell
# "reads the right physical ci" apart from "reads the only ci that
# exists" (every original fixture aliased every csa2_on layer to ONE
# ci), and that neither of the two review-added validation predicates
# (lz_csa2_pub_overflows, lz_csa2_mode_source_valid) nor the later
# whole-config one (lz_csa2_last_writer_valid) had a mutation of their
# own - this file's own history is the record of both rounds; it is not
# rewritten to look like it was designed in one pass, because it was not.
#
#   MUTATION 1, "dispatch reverted": the three csa2_mode guards this
#   task adds (step 2's `if (mode == LZ_CSA2_FULL)`, step 3's
#   `if (mode != LZ_CSA2_REUSE)`, step 6's `... && mode == LZ_CSA2_FULL`)
#   are each replaced with an always-true condition, in a scratch copy -
#   every csa2_on layer runs the pre-this-task, unconditional six-step
#   body again, regardless of what L->csa2_mode says. Expected fallout,
#   verified against build/csa2_mode_dispatch_probe.c's ACTUAL current
#   check list (the exact set below, not a shape assumed from the code
#   alone):
#     STAY PASS - every Full-only check (1a, 1b), check_chain's cache-
#       idx-shared and layer-A-own-growth and layer-B-selection-differs
#       lines, check2's A-untouched line, all three check_ci_
#       independence lines (A and D are both Full, already unconditional
#       either way), check4a, and check4b's own published-k_eff line.
#     TURN FAIL - check_chain's two REINDEX/REUSE-specific lines, both
#       of check2's B/C-changed lines, both of check3a's lines, check5,
#       and check4b's attn_out and published-index-set lines - a
#       REINDEX layer now also runs steps 2/6 with WRONG-but-present
#       tensors (build_layer_reindex's own comment on why REINDEX is
#       given a full tensor set), corrupting the shared ring/pending
#       state its aliased ci shares with its source, and kv_in is read
#       again for D2.
#     CRASHES right after check4b's own three lines, before check3b
#       ever prints anything: check3b's own C layer
#       (build_layer_reuse_strict) has NO tensors to run FULL's steps
#       2/3/6 with, so the reverted dispatch dereferences a NULL
#       LZTensor.f - the correct, expected outcome, not a probe defect.
#       build_model_indep's own E is deliberately "soft" (build_layer_
#       reuse_soft), NOT strict, specifically so check_ci_independence
#       (which runs BEFORE check3b) does not crash here itself and rob
#       check4/check3b of the chance to run at all - an earlier draft of
#       this fixture used a strict E and moved the crash point earlier,
#       silently losing coverage of everything after it.
#     build/csa2_mode_dispatch_probe.c's own ok() flushes stdout after
#     every line specifically so this crash does not erase whatever
#     already ran and printed before it.
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
#     - every other check, including check_ci_independence (a DIFFERENT
#       collapse - across ci, not across tk within one ci - see mutation
#       3 below), stays PASS.
#
#   MUTATION 3, "ci term dropped from the publish base pointers": the
#   `+ (size_t)ci * LZ_BATCH_MAX * topk` / `+ (size_t)ci * LZ_BATCH_MAX`
#   terms on pub_idx_base/pub_n_base THEMSELVES (declared once, before
#   the per-token loop - a different pair of lines than mutation 2's,
#   which only ever touches the per-token tk sub-offset added onto
#   these same bases) are removed. Every csa2_on layer's publish
#   traffic now lands in the SAME physical (ci=0-shaped) region
#   regardless of its own real ci - exactly what build_model_indep's own
#   two-independent-ci fixture (check_ci_independence) exists to catch,
#   and no other fixture in this file can: every one of them aliases
#   every csa2_on layer to ONE ci, so a bug that stops distinguishing
#   ci values entirely is invisible to them by construction. Expected
#   fallout: ONLY check_ci_independence's own third line (layer D's
#   perturbation now reaches layer E, since D's writes land in the same
#   region A's/E's do) turns FAIL; every other line, including the
#   other two check_ci_independence lines (ci_a != ci_d and ci_e ==
#   ci_a are both about s->cache_idx[], untouched by this mutation -
#   only the SEPARATE csa2_topk_idx_pub/csa2_topk_n_pub addressing
#   moves), stays PASS.
#
#   MUTATION 4, "lz_csa2_pub_overflows neutered": forward_state.c's own
#   final `return (size_t)topk > cap / n;` becomes `return 0;` -
#   the function always answers "no overflow", the direction that
#   actually matters (a false "safe" lets the real bug through; a false
#   "overflow" would only ever reject a legitimate config). Expected
#   fallout: ONLY check_overflow_guard's "4 * (size_t's own max/4 + 1)
#   ... correctly flagged" line turns FAIL (the one assertion that
#   requires the function to answer "yes, this overflows"); the "stays
#   under the ceiling" control, the production-worst-case check, and
#   both nf==0/batch_max==0 short-circuit checks (which never reach the
#   mutated return at all) stay PASS.
#
#   MUTATION 5, "lz_csa2_mode_source_valid's source half neutered":
#   model.c's own final
#     `return src >= 0 && src < li && c->csa2_on[src] &&
#              c->layer_types[src] == LZ_LT_FULL;`
#   becomes `return 1;` - the SOURCE-validity half of the predicate only
#   (the mode-code check and the `!c->csa2_on[li]` check above it, both
#   untouched, still run first and can still return 0 on their own).
#   Expected fallout: exactly the five check_bin_validation lines that
#   test a BAD SOURCE (sentinel -1 paired with non-FULL mode, self-
#   reference, forward-reference, wrong-namespace/LINEAR, never-csa2_on
#   source) turn FAIL; the mode-code-outside-{0,1,2} line, the ordinary-
#   non-csa2_on-layer control, and both F9 whole-config lines (which
#   depend on lz_csa2_last_writer_valid, a different function) stay
#   PASS.
#
#   MUTATION 6, "lz_csa2_last_writer_valid neutered": model.c's own
#   `if (c->csa2_source[i] != last_writer[r]) {` becomes
#   `if (0 && c->csa2_source[i] != last_writer[r]) {` - the walk and its
#   last_writer bookkeeping still run, but the violation it exists to
#   detect can never fire. Expected fallout: ONLY the F9 "whole-config
#   predicate refuses it" line turns FAIL; the F9 "well-formed chain
#   passes" line (nothing to refuse there) and the F9 "per-layer
#   predicate alone still accepts it" line (a DIFFERENT function,
#   lz_csa2_mode_source_valid, untouched) both stay PASS, along with
#   everything else.
#
# A mutant where the expected-to-fail lines stay PASS means the probe
# cannot see that defect at all - reported as a gate FAILURE. A mutant
# where any OTHER line also changes means the probe's checks are not
# independent, which is a probe defect in its own right - also a gate
# FAILURE, not folded into "the mutation worked".
#
# Every source file is copied into the scratch dir (not just
# forward_csa2.c/forward_state.c/model.c/forward.h/model.h) for the
# quote-include resolution reason json_offset_gate.sh documents:
# `#include "forward.h"` resolves against forward_csa2.c's own
# directory before -I is ever consulted.
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
mkdir -p "$W/fixed" "$W/mut1" "$W/mut2" "$W/mut3" "$W/mut4" "$W/mut5" "$W/mut6"
trap 'rm -rf "$W"' EXIT

for d in fixed mut1 mut2 mut3 mut4 mut5 mut6; do
    cp src/*.c src/*.h "$W/$d/" 2>/dev/null
done

# Sanity: the lines every mutation targets must exist exactly once in
# the current source, or this gate no longer matches the code it names.
for needle_file in \
    "forward_csa2.c:        if (mode == LZ_CSA2_FULL) {" \
    "forward_csa2.c:        if (mode != LZ_CSA2_REUSE) {" \
    "forward_csa2.c:        if (off == cm - 1 && mode == LZ_CSA2_FULL) {" \
    "forward_csa2.c:        int *tok_idx = pub_idx_base + (size_t)tk * topk;" \
    "forward_csa2.c:        int *tok_n = pub_n_base + tk;" \
    "forward_csa2.c:    int *pub_idx_base = s->csa2_topk_idx_pub + (size_t)ci * LZ_BATCH_MAX * topk;" \
    "forward_csa2.c:    int *pub_n_base = s->csa2_topk_n_pub + (size_t)ci * LZ_BATCH_MAX;" \
    "forward_state.c:    return (size_t)topk > cap / n;"
do
    file=${needle_file%%:*}
    needle=${needle_file#*:}
    N=$(grep -Fc "$needle" "$W/fixed/$file")
    if [ "$N" != "1" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - expected exactly 1 occurrence of" \
             "'$needle' in $file, found $N (the gate no longer matches the source)"
        exit 1
    fi
done
N=$(grep -Fc "        if (c->csa2_source[i] != last_writer[r]) {" "$W/fixed/model.c")
if [ "$N" != "1" ]; then
    echo "csa2_mode_dispatch_gate: FAIL - expected exactly 1 occurrence of the" \
         "last_writer check in model.c, found $N"
    exit 1
fi

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

python3 - "$W/mut3/forward_csa2.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
subs = [
    ("    int *pub_idx_base = s->csa2_topk_idx_pub + (size_t)ci * LZ_BATCH_MAX * topk;\n",
     "    int *pub_idx_base = s->csa2_topk_idx_pub;\n"),
    ("    int *pub_n_base = s->csa2_topk_n_pub + (size_t)ci * LZ_BATCH_MAX;\n",
     "    int *pub_n_base = s->csa2_topk_n_pub;\n"),
]
for old, new in subs:
    if src.count(old) != 1:
        print("csa2_mode_dispatch_gate: FAIL - mutation 3 could not find "
              "exactly one copy of %r" % old, file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

python3 - "$W/mut4/forward_state.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = "    return (size_t)topk > cap / n;\n"
new = "    return 0;\n"
if src.count(old) != 1:
    print("csa2_mode_dispatch_gate: FAIL - mutation 4 could not find "
          "exactly one copy of %r" % old, file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

python3 - "$W/mut5/model.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = ("    return src >= 0 && src < li && c->csa2_on[src] &&\n"
       "           c->layer_types[src] == LZ_LT_FULL;\n")
new = "    return 1;\n"
if src.count(old) != 1:
    print("csa2_mode_dispatch_gate: FAIL - mutation 5 could not find "
          "exactly one copy of %r" % old, file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

python3 - "$W/mut6/model.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
old = "        if (c->csa2_source[i] != last_writer[r]) {\n"
new = "        if (0 && c->csa2_source[i] != last_writer[r]) {\n"
if src.count(old) != 1:
    print("csa2_mode_dispatch_gate: FAIL - mutation 6 could not find "
          "exactly one copy of %r" % old, file=sys.stderr)
    sys.exit(1)
src = src.replace(old, new, 1)
with open(path, "w", encoding="utf-8") as f:
    f.write(src)
PYEOF

check_mutated() {
    local dir="$1" file="$2" needle="$3" label="$4"
    if grep -qF "$needle" "$dir/$file"; then
        echo "csa2_mode_dispatch_gate: FAIL - could not mutate $label ($dir/$file still has $needle)"
        exit 1
    fi
}
check_mutated "$W/mut1" forward_csa2.c "if (mode == LZ_CSA2_FULL) {" "mutant 1"
check_mutated "$W/mut1" forward_csa2.c "if (mode != LZ_CSA2_REUSE) {" "mutant 1"
check_mutated "$W/mut2" forward_csa2.c "pub_idx_base + (size_t)tk * topk" "mutant 2"
check_mutated "$W/mut2" forward_csa2.c "pub_n_base + tk" "mutant 2"
check_mutated "$W/mut3" forward_csa2.c "+ (size_t)ci * LZ_BATCH_MAX * topk;" "mutant 3"
check_mutated "$W/mut3" forward_csa2.c "+ (size_t)ci * LZ_BATCH_MAX;" "mutant 3"
check_mutated "$W/mut4" forward_state.c "return (size_t)topk > cap / n;" "mutant 4"
check_mutated "$W/mut5" model.c "c->layer_types[src] == LZ_LT_FULL;" "mutant 5"
check_mutated "$W/mut6" model.c "if (c->csa2_source[i] != last_writer[r]) {" "mutant 6"

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

# The probe's own total ok() call count, unaffected by any mutation
# that does not crash it outright - matches mutant 2's own
# STAY_PASS+TURN_FAIL array lengths further down (40 entries combined),
# verified by direct count when those arrays were last written, not
# re-derived here since bash arrays are not yet in scope this early in
# the script. Mutants 3-6 run the probe to completion (never crash) and
# so must each account for exactly this many PASS+FAIL lines, same
# reasoning as mutant 2's own total-count assertion; mutant 1 crashes
# before check3b and declares its own, smaller total instead (its own
# comment has the count).
PROBE_TOTAL_LINES=40

echo "-- current tree (must PASS) --"
build_and_run "$W/fixed" "the current (fixed) tree"
FIXED_RC=$?
if [ "$FIXED_RC" -eq 2 ]; then
    exit 1
elif [ "$FIXED_RC" -ne 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL - the current tree did not pass its own probe"
    RC=1
fi

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
    # STAY_PASS/TURN_FAIL below are re-verified empirically against this
    # exact probe after the C2 fix (report Sec2.3.1 mode-dispatch task's
    # fifth review round), not carried over unread: that fix made a
    # REINDEX/REUSE layer's own n_vis a direct pos/cm computation (see
    # forward_csa2.c's own comment on that line), which does not live
    # inside any of the three `if (mode == ...)` conditions this mutant
    # patches - so "layer B sees only strictly earlier positions" and
    # its own k_eff-growth sequence are NOW STRUCTURALLY UNABLE to
    # detect this mutation (measured, not assumed: both stayed PASS
    # under the mutated dispatch). check2's own B/C-attn_out-changed
    # pair and check5's own batch-vs-unbatched line swap sides for the
    # same underlying reason - the corrupted shared ring state this
    # mutant produces interacts with the fixed n_vis formula
    # differently than it did with the old shared-counter read. What
    # still catches this mutant: the "untouched by C" publish-buffer
    # check (C now overwrites it, since its own step 3 guard is also
    # patched), check2/check3a's attn_out-changes assertions, and
    # check4b's kv_in-independence pair, plus the same check3b crash as
    # before.
    STAY_PASS=(
        "check1a: first-ever token's published k_eff is 0"
        "check1a: k_eff==0 token's attn_out is the exact zero vector"
        "check1b: batch-vs-unbatched - published k_eff matches token for token"
        "check1b: batch-vs-unbatched - published Top-K indices match token for token"
        "check1b: batch-vs-unbatched - last token's attn_out is bit-identical"
        "check_chain: all three layers share the same physical ci (cache_idx aliasing from the prior task still holds)"
        "check_chain: layer A's own published k_eff grows 0,1,2,2(=topk) across the chunk"
        "check_chain: layer B (reindex) sees only STRICTLY EARLIER positions' already-finalized blocks - k_eff grows 0,1,2,2 across its own chunk exactly like layer A's, not a constant leaked in from A's already-finished call"
        "check_chain: layer B's own Top-K selection is not identical across all four of its own tokens"
        "check2: layer A's own attn_out is untouched by a change to layer B's (unrelated) indexer weights"
        "check5: layer C's (reuse) o_proj output for EVERY token of a 3-token chunk processed as one nt=3 call per layer is bit-identical to the same 3 positions processed one at a time (nt=1, layer-major per position) - not merely the chunk's last token"
        "check_ci_independence: two independent FULL layers get MUTUALLY DISTINCT ci (no accidental aliasing between them)"
        "check_ci_independence: the reuse layer's ci is aliased to its own declared source (A), not to the unrelated FULL layer (D)"
        "check_ci_independence: layer E's (reuse, source A) attn_out is untouched by a change to layer D's (unrelated, independently-aliased) indexer weights - per-ci addressing genuinely isolates the two roots, not merely the one this fixture happens to read"
        "check4a: layer D1 (FULL, decoder-region CSA2) still reads kv_in in its own step 2"
        "check4b: ...and so is the published k_eff"
    )
    TURN_FAIL=(
        "check_chain: after layer C's (reuse) call, all four tokens' published Top-K selections are still exactly layer B's own"
        "check2: perturbing layer B's own indexer weights changes layer B's own attn_out"
        "check2: perturbing layer B's own indexer weights ALSO changes layer C's attn_out (C reads whatever B's own call left in their shared ci slot)"
        "check3a: perturbing layer A's compression weights changes layer B's attn_out"
        "check3a: perturbing layer A's compression weights also changes layer C's attn_out"
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
    # Total-count assertion (Minor-4, sixth review round), same idiom
    # mutants 3-6 use: STAY_PASS+TURN_FAIL here is not the probe's full
    # 40 lines - mutant 1 crashes before check3b ever prints, so its own
    # declared total (23 lines) is what a line vanishing outright (never
    # printed as either PASS or FAIL) would be invisible against.
    if [ "$(grep -c '^PASS\|^FAIL' "$W/mut1/run.log")" != "$((${#STAY_PASS[@]} + ${#TURN_FAIL[@]}))" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - mutant 1's own STAY_PASS+TURN_FAIL" \
             "($((${#STAY_PASS[@]} + ${#TURN_FAIL[@]})) lines) does not account for" \
             "every line the probe printed before crashing"
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
        "check_chain: layer B's own Top-K selection is not identical across all four of its own tokens"
        "check_chain: after layer C's (reuse) call, all four tokens' published Top-K selections are still exactly layer B's own"
        "check2: layer A's own attn_out is untouched by a change to layer B's (unrelated) indexer weights"
        "check2: perturbing layer B's own indexer weights changes layer B's own attn_out"
        "check2: perturbing layer B's own indexer weights ALSO changes layer C's attn_out (C reads whatever B's own call left in their shared ci slot)"
        "check3a: perturbing layer A's compression weights changes layer B's attn_out"
        "check3a: perturbing layer A's compression weights also changes layer C's attn_out"
        "check_ci_independence: two independent FULL layers get MUTUALLY DISTINCT ci (no accidental aliasing between them)"
        "check_ci_independence: the reuse layer's ci is aliased to its own declared source (A), not to the unrelated FULL layer (D)"
        "check_ci_independence: layer E's (reuse, source A) attn_out is untouched by a change to layer D's (unrelated, independently-aliased) indexer weights - per-ci addressing genuinely isolates the two roots, not merely the one this fixture happens to read"
        "check4a: layer D1 (FULL, decoder-region CSA2) still reads kv_in in its own step 2"
        "check4b: layer D2 (reindex, decoder-region CSA2) attn_out is byte-identical whether kv_in is NULL or a poisoned non-NULL"
        "check4b: ...and so is the published k_eff"
        "check4b: ...and so is the published Top-K index set"
        "check3b: layer C (reuse) completed its whole 3-token call without touching any of the seven csa2_w_* tensors it does not own, and its output is finite"
        "overflow guard: nf = cap/4+1, topk=4, elem_sz=1 - 4x nf exceeds the exact byte ceiling - correctly flagged"
        "overflow guard: the same nf at topk=3 stays under the ceiling - correctly NOT flagged"
        "overflow guard: this project's own worst-case numbers (nf=9, batch_max=8, topk=512, elem_sz=sizeof(int)) do not trip it - true on every size_t width, not host-dependent"
        "overflow guard: nf==0 does not divide by zero and answers 'no overflow' (xcalloc(0, ...) itself short-circuits before ever reaching n*sz)"
        "overflow guard: batch_max==0 likewise does not divide by zero"
        "bin validation: the well-formed chain fixture passes on every layer"
        "bin validation: csa2_mode outside {0,1,2} is refused"
        "bin validation: a non-FULL mode with the FULL-mode sentinel source (-1) is refused, not read as a negative index"
        "bin validation: a source naming itself is refused"
        "bin validation: a source naming a LATER layer is refused"
        "bin validation: a source naming a LINEAR (not full_attention) layer is refused"
        "bin validation: a source that is full_attention but never marked csa2_on is refused"
        "bin validation: an ordinary non-csa2_on layer (default FULL mode) is not refused"
        "bin validation (F9): the well-formed chain (layer 3's source is layer 2, the actual last writer of root 0's slot) passes"
        "bin validation (F9): the per-layer predicate ALONE still accepts source=0 for layer 3 - demonstrating the gap this addendum closes, not just asserting the fix works"
        "bin validation (F9): the whole-config predicate refuses it, naming layer 3 as the violation"
    )
    TURN_FAIL=(
        "check1b: batch-vs-unbatched - published k_eff matches token for token"
        "check1b: batch-vs-unbatched - published Top-K indices match token for token"
        "check_chain: layer A's own published k_eff grows 0,1,2,2(=topk) across the chunk"
        "check_chain: layer B (reindex) sees only STRICTLY EARLIER positions' already-finalized blocks - k_eff grows 0,1,2,2 across its own chunk exactly like layer A's, not a constant leaked in from A's already-finished call"
        "check5: layer C's (reuse) o_proj output for EVERY token of a 3-token chunk processed as one nt=3 call per layer is bit-identical to the same 3 positions processed one at a time (nt=1, layer-major per position) - not merely the chunk's last token"
    )
    missing=0
    check_lines "$W/mut2/run.log" "mutant 2" "PASS" "${STAY_PASS[@]}" || missing=1
    check_lines "$W/mut2/run.log" "mutant 2" "FAIL" "${TURN_FAIL[@]}" || missing=1
    # Total-count assertion, same idiom mutant 3/4/5/6 already use below -
    # STAY_PASS+TURN_FAIL above must be EVERY line the probe can print,
    # not just the ones this mutation was originally written against.
    # Missing this let mutant 2's own overflow-guard/bin-validation/F9
    # lines (16 of the probe's 40 total) go unlisted in EITHER array,
    # invisible to a FUTURE change that made any of them turn FAIL under
    # this mutation - they all currently stay PASS (this mutation never
    # touches model.c), which is why they are all in STAY_PASS now.
    TOTAL=$((${#STAY_PASS[@]} + ${#TURN_FAIL[@]}))
    if [ "$(grep -c '^PASS\|^FAIL' "$W/mut2/run.log")" != "$TOTAL" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - mutant 2's own STAY_PASS+TURN_FAIL" \
             "($TOTAL lines) does not account for every line the probe printed"
        missing=1
    fi
    # Cross-check against PROBE_TOTAL_LINES (mutants 3-6's own total-
    # count assertions, defined near the top of this script since bash
    # arrays are not yet in scope there) - this is the ONE place that
    # constant can drift out of sync with reality, since it is a
    # hardcoded copy of what these two arrays' lengths add up to here.
    if [ "$TOTAL" != "$PROBE_TOTAL_LINES" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - PROBE_TOTAL_LINES ($PROBE_TOTAL_LINES)" \
             "no longer matches mutant 2's own STAY_PASS+TURN_FAIL length ($TOTAL) -" \
             "update the constant near the top of this script"
        missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant 2 failed exactly the direct per-token publish-buffer reads, and no others --"
    fi
fi

echo ""
echo "-- mutant 3: ci term dropped from the publish base pointers --"
build_and_run "$W/mut3" "mutant 3 (ci term dropped)"
MUT3_RC=$?
if [ "$MUT3_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT3_RC" -eq 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 3 PASSED; the probe cannot see this defect"
    RC=1
else
    if grep -qF "FAIL check_ci_independence: layer E's (reuse, source A) attn_out is untouched by a change to layer D's (unrelated, independently-aliased) indexer weights - per-ci addressing genuinely isolates the two roots, not merely the one this fixture happens to read" "$W/mut3/run.log" \
       && [ "$(grep -c '^FAIL' "$W/mut3/run.log")" = "1" ] \
       && [ "$(grep -c '^PASS\|^FAIL' "$W/mut3/run.log")" = "$PROBE_TOTAL_LINES" ]; then
        echo "-- mutant 3 failed exactly the ci-independence check, and no others --"
    else
        echo "csa2_mode_dispatch_gate: FAIL - mutant 3 did not fail exactly one line" \
             "(check_ci_independence's own cross-root read), or did not print" \
             "all $PROBE_TOTAL_LINES lines the probe normally prints"
        RC=1
    fi
fi

echo ""
echo "-- mutant 4: lz_csa2_pub_overflows neutered (always answers 'no overflow') --"
build_and_run "$W/mut4" "mutant 4 (overflow guard neutered)"
MUT4_RC=$?
if [ "$MUT4_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT4_RC" -eq 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 4 PASSED; the probe cannot see this defect"
    RC=1
else
    if grep -qF "FAIL overflow guard: nf = cap/4+1, topk=4, elem_sz=1 - 4x nf exceeds the exact byte ceiling - correctly flagged" "$W/mut4/run.log" \
       && [ "$(grep -c '^FAIL' "$W/mut4/run.log")" = "1" ] \
       && [ "$(grep -c '^PASS\|^FAIL' "$W/mut4/run.log")" = "$PROBE_TOTAL_LINES" ]; then
        echo "-- mutant 4 failed exactly the must-overflow boundary check, and no others --"
    else
        echo "csa2_mode_dispatch_gate: FAIL - mutant 4 did not fail exactly one line" \
             "(the must-overflow boundary check), or did not print all" \
             "$PROBE_TOTAL_LINES lines the probe normally prints"
        RC=1
    fi
fi

echo ""
echo "-- mutant 5: lz_csa2_mode_source_valid's source half neutered --"
build_and_run "$W/mut5" "mutant 5 (mode_source_valid neutered)"
MUT5_RC=$?
if [ "$MUT5_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT5_RC" -eq 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 5 PASSED; the probe cannot see this defect"
    RC=1
else
    TURN_FAIL=(
        "bin validation: a non-FULL mode with the FULL-mode sentinel source (-1) is refused, not read as a negative index"
        "bin validation: a source naming itself is refused"
        "bin validation: a source naming a LATER layer is refused"
        "bin validation: a source naming a LINEAR (not full_attention) layer is refused"
        "bin validation: a source that is full_attention but never marked csa2_on is refused"
    )
    missing=0
    check_lines "$W/mut5/run.log" "mutant 5" "FAIL" "${TURN_FAIL[@]}" || missing=1
    if [ "$(grep -c '^FAIL' "$W/mut5/run.log")" != "${#TURN_FAIL[@]}" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - mutant 5 failed a different number of" \
             "lines than expected (${#TURN_FAIL[@]}); checks are not independent"
        missing=1
    fi
    if [ "$(grep -c '^PASS\|^FAIL' "$W/mut5/run.log")" != "$PROBE_TOTAL_LINES" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - mutant 5 did not print all" \
             "$PROBE_TOTAL_LINES lines the probe normally prints"
        missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant 5 failed exactly the five bad-source checks, and no others --"
    fi
fi

echo ""
echo "-- mutant 6: lz_csa2_last_writer_valid neutered --"
build_and_run "$W/mut6" "mutant 6 (last_writer_valid neutered)"
MUT6_RC=$?
if [ "$MUT6_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT6_RC" -eq 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 6 PASSED; the probe cannot see this defect"
    RC=1
else
    if grep -qF "FAIL bin validation (F9): the whole-config predicate refuses it, naming layer 3 as the violation" "$W/mut6/run.log" \
       && [ "$(grep -c '^FAIL' "$W/mut6/run.log")" = "1" ] \
       && [ "$(grep -c '^PASS\|^FAIL' "$W/mut6/run.log")" = "$PROBE_TOTAL_LINES" ]; then
        echo "-- mutant 6 failed exactly the F9 whole-config check, and no others --"
    else
        echo "csa2_mode_dispatch_gate: FAIL - mutant 6 did not fail exactly one line" \
             "(the F9 whole-config check), or did not print all $PROBE_TOTAL_LINES" \
             "lines the probe normally prints"
        RC=1
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL"
    exit 1
fi
echo "csa2_mode_dispatch_gate: PASS"
exit 0
