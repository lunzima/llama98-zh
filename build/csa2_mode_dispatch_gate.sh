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
# THE CONTROL is SEVEN INDEPENDENT MUTATIONS, not a git-history diff, same
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
# own. Mutation 7 below is a move rather than a substitution, and covers
# what none of the six can: WHERE the finalization step sits in
# forward_csa2.c's per-token loop, which is what makes n_vis a single
# mode-independent formula. This file's own history is the record of
# those rounds; it is not rewritten to look like it was designed in one
# pass, because it was not.
#
#   MUTATION 1, "dispatch reverted": the three csa2_mode guards in
#   forward_csa2.c - step 2's FULL-only `if (mode == LZ_CSA2_FULL)`,
#   step 3's finalize guard `if (off == cm - 1 && mode ==
#   LZ_CSA2_FULL)`, and step 4's REUSE-skip `if (mode !=
#   LZ_CSA2_REUSE)` - are each replaced with an always-true condition,
#   in a scratch copy. Every csa2_on layer then runs the unconditional
#   six-step body, regardless of what L->csa2_mode says. Expected
#   fallout, read off this probe's own filtered OUTPUT under the
#   mutation (build/csa2_mode_dispatch_probe.c's ACTUAL current check
#   list - the exact set below, not a shape assumed from the code
#   alone):
#     STAY PASS - every Full-only check (1a, 1b), check_chain's cache-
#       idx-shared, layer-A-own-growth, layer-B-k_eff-growth and
#       layer-B-selection-differs lines, check2's A-untouched line and
#       its B-changed line (B's attn_out still depends on B's OWN
#       indexer weights under any dispatch, so that line tests the
#       fixture's sensitivity rather than the dispatch, and rightly
#       stays green), all three check_ci_independence lines (A and D
#       are both Full, already unconditional either way), check5,
#       check4a, and check4b's own published-k_eff line.
#     TURN FAIL - check_chain's "after layer C's call" line, check2's
#       C-changed line, both of check3a's lines, and check4b's attn_out
#       and published-index-set lines - a REINDEX layer now also runs
#       steps 2/3 with WRONG-but-present tensors (build_layer_reindex's
#       own comment on why REINDEX is given a full tensor set),
#       corrupting the shared ring/pending state its aliased ci shares
#       with its source, and kv_in is read again for D2.
#       n_vis is NOT among the lines this reaches: it is a direct
#       per-position (pos + 1) / cm computation, which does not live
#       inside any of the three conditions patched here.
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
#   have sufficed". Expected fallout, read off this probe's own run
#   under the mutation rather than assumed:
#     - check_chain_soft's two DIRECT publish-buffer reads turn FAIL -
#       "layer A's own published k_eff grows 1,2,2,2(=topk) across the
#       chunk" (A's own four tokens, tk=0..3, all publish into the same
#       physical slot, so only the last one survives) and "layer B
#       (reindex) k_eff grows 1,2,2,2 across its own chunk" (same
#       collapse, one layer downstream).
#     - check1b's own two published-buffer match assertions ALSO turn
#       FAIL: they are not Reindex/Reuse-specific (check1b is a single
#       FULL-mode layer), but they specifically compare batched vs
#       unbatched PUBLISHED CONTENT token for token, which this
#       mutation breaks even for Full-only.
#     - check5 turns FAIL too: under per-ci addressing a REUSE layer
#       reads whichever token wrote that one slot LAST, so a 3-token
#       nt=3 call and three nt=1 calls no longer agree on every token.
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
#   fallout: ONLY check_overflow_guard's own "nf = cap/4+1, topk=4,
#   elem_sz=1 - 4x nf exceeds the exact byte ceiling - correctly
#   flagged" line turns FAIL (the one assertion that requires the
#   function to answer "yes, this overflows"); the "stays under the
#   ceiling" control, the production-worst-case check, and both
#   nf==0/batch_max==0 short-circuit checks (which never reach the
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
#   MUTATION 7, "finalization moved back to the end of the per-token
#   loop": forward_csa2.c's whole finalize region - the
#   `if (off == cm - 1 && mode == LZ_CSA2_FULL) { ... }` block together
#   with the comment that introduces it - is CUT from between step 2 and
#   the indexer and re-INSERTED after step 6's o_proj, inside the same
#   loop. A move, not a substitution: no `if (mode == ...)` guard is
#   touched, so mutation 1's own fallout list is untouched by it, and no
#   line's text changes, so mutation 1/2's substitution greps cannot see
#   it. What it breaks is the one property that makes
#   `n_vis = (pos + 1) / cm` usable for every mode at once - that
#   ring[blk]/ring_idx[blk] already hold the block a query is standing on
#   by the time step 5 reads them. Expected fallout, read off this
#   probe's own run under the mutation rather than assumed:
#     - TURN FAIL: exactly check1a's two attn_out assertions - cm=1
#       position 0, and cm=2 position 1, a block's own last raw token.
#       Both positions select a single block either way, so `k_eff 1,
#       selecting block 0` keeps printing PASS at both ratios (n_vis is
#       a direct per-position computation that no move can reach, and
#       the one-candidate selection is index 0 whatever the scores are).
#       What the move changes is the ring CONTENT step 5 reads: with the
#       finalization at the end of the loop, ring[0] is still the
#       allocation's zeros when step 5 reads it, a one-candidate softmax
#       over a zero entry weights it 1.0, and the output gate and o_proj
#       carry that exact zero vector out.
#     - STAY PASS: every other line the probe prints - 42 of the 44. The
#       plain ones are the 40 the probe carried before this series added
#       check1a's four assertions (this gate's own PROBE_TOTAL_LINES was
#       40 at 978144d and is 44 now; two of those four are the TURN FAIL
#       pair above, so 40 + these two + those two is the 44). The other
#       two are named here, check1a's cm=1 and cm=2 selection lines,
#       which are the same shape as each other and are blind for the
#       same reason. That is the whole point of this mutation: a check
#       that compares an attn_out against another attn_out, or a
#       published k_eff against another published k_eff, runs BOTH sides
#       under the same reordering and so agrees with itself whatever the
#       order is. Only an absolute value can see it, and before these
#       two assertions nothing in this probe looked at one.
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
mkdir -p "$W/fixed" "$W/mut1" "$W/mut2" "$W/mut3" "$W/mut4" "$W/mut5" "$W/mut6" "$W/mut7"
trap 'rm -rf "$W"' EXIT

for d in fixed mut1 mut2 mut3 mut4 mut5 mut6 mut7; do
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
    "forward_csa2.c:        lz_matmul_w(s->xb2 + (size_t)tk * hdim, s->attn_out, &L->o_proj," \
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

python3 - "$W/mut7/forward_csa2.c" <<'PYEOF' || exit 1
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    before = f.read().splitlines(True)

GUARD = "        if (off == cm - 1 && mode == LZ_CSA2_FULL) {\n"
OPROJ0 = "        lz_matmul_w(s->xb2 + (size_t)tk * hdim, s->attn_out, &L->o_proj,\n"
OPROJ1 = "                   qd, hdim, s->xq, s->xqs);\n"


def fail(msg):
    print("csa2_mode_dispatch_gate: FAIL - mutation 7 " + msg, file=sys.stderr)
    sys.exit(1)


def only(lines, needle, what):
    hits = [i for i, l in enumerate(lines) if l == needle]
    if len(hits) != 1:
        fail("expected exactly one %s line in the source, found %d"
             % (what, len(hits)))
    return hits[0]


guard_i = only(before, GUARD, "finalize guard")
op0 = only(before, OPROJ0, "step 6 o_proj call")
if op0 + 1 >= len(before) or before[op0 + 1] != OPROJ1:
    fail("the o_proj call is not the two-line shape the move anchors on")
if not guard_i < op0:
    fail("the finalize guard is not ahead of o_proj in the source to begin with")

# The contiguous block comment directly above the guard is part of the
# region that moves (a stray step-3 comment left behind would describe a
# step that no longer runs there).
cut_start = guard_i
if before[cut_start - 1].rstrip().endswith("*/"):
    j = cut_start - 1
    while not before[j].lstrip().startswith("/*"):
        j -= 1
    cut_start = j

# Cut end: the guard's matching close brace, counted on the code itself
# rather than on a comment or a fixed line offset.
depth = 0
k = guard_i
while True:
    if k >= len(before):
        fail("the finalize guard's braces never balance")
    depth += before[k].count("{") - before[k].count("}")
    k += 1
    if depth == 0:
        break
cut_end = k

block = before[cut_start:cut_end]
after = before[:cut_start] + before[cut_end:]
op0_after = only(after, OPROJ0, "step 6 o_proj call (after the cut)")
after = after[:op0_after + 2] + block + after[op0_after + 2:]

# The move must be a PURE reordering: same lines, same count, only the
# two regions swapped. Anything else and this arm is not testing the
# ordering it claims to.
if len(after) != len(before):
    fail("the move changed the line count (%d -> %d)" % (len(before), len(after)))
if sorted(after) != sorted(before):
    fail("the move altered line content - it has to be a pure reordering")
guard_after = only(after, GUARD, "finalize guard (after the move)")
if guard_after != op0_after + 2 + (guard_i - cut_start):
    fail("the finalize region did not land directly after the o_proj call")
if not guard_after > op0_after:
    fail("the finalize guard is still ahead of the o_proj call")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(after)
print("mutation 7 applied: %d lines cut at line %d, re-inserted after line %d"
      % (len(block), cut_start + 1, op0_after + 2))
PYEOF

# A needle that is gone means the mutation took. The three outcomes are
# kept apart rather than collapsed into `if grep -q`: grep exits 2 when
# it cannot read the file at all, and `if` reads that as "not found" -
# i.e. as a successful mutation. The arms below are still caught by the
# compile step (a missing source fails the link, and every arm treats
# rc 2 as fatal), so this is a self-check reporting a read error as a
# pass rather than a hole in the gate - but a self-check should fail
# closed on its own account too.
check_mutated() {
    local dir="$1" file="$2" needle="$3" label="$4"
    local rc
    grep -qF "$needle" "$dir/$file"
    rc=$?
    case $rc in
    0)  echo "csa2_mode_dispatch_gate: FAIL - could not mutate $label ($dir/$file still has $needle)"
        exit 1 ;;
    1)  : ;;
    *)  echo "csa2_mode_dispatch_gate: FAIL - could not read $dir/$file to check $label" \
             "(grep exited $rc)"
        exit 1 ;;
    esac
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
# Mutant 7's verification lives here rather than in check_mutated above:
# it is a MOVE, so its needle is never gone - only somewhere else - and
# "the needle is gone" is the one thing that would be wrong to assert.
# The python step already asserted purity (same lines, same count, same
# block) on the text it had in hand; this re-reads the file from disk and
# asserts the ordering the name promises, so a mutation that silently did
# nothing cannot pass for a mutation that worked.
#
# Both anchors are matched as WHOLE LINES (-x) and each is required to
# occur exactly once, so the two line numbers below are unambiguous.
#
# The uniqueness count is not this arm's main protection, though. The
# python step above already refuses any source whose o_proj call is not
# the exact two-line shape the move cuts around, and pins the guard's
# output line number by an equation that implies it lands after o_proj -
# a source forged to keep this shell test happy is caught there first.
# What this block adds is a check that reads the file BACK FROM DISK, so
# a mutation whose write silently did nothing is caught here instead of
# being reported as a mutation that worked.
G7_LINE='        if (off == cm - 1 && mode == LZ_CSA2_FULL) {'
O7_LINE='        lz_matmul_w(s->xb2 + (size_t)tk * hdim, s->attn_out, &L->o_proj,'
G7_N=$(grep -cxF "$G7_LINE" "$W/mut7/forward_csa2.c")
O7_N=$(grep -cxF "$O7_LINE" "$W/mut7/forward_csa2.c")
if [ "$G7_N" != "1" ] || [ "$O7_N" != "1" ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 7's own anchors are not each" \
         "unique in the mutated source (finalize guard ${G7_N}x, o_proj call" \
         "${O7_N}x) - this arm cannot tell where the block landed"
    exit 1
fi
G7=$(grep -nxF "$G7_LINE" "$W/mut7/forward_csa2.c" | cut -d: -f1)
O7=$(grep -nxF "$O7_LINE" "$W/mut7/forward_csa2.c" | cut -d: -f1)
# The two -z arms are NOT redundant with the count check above. Under
# `set -u` an empty G7 makes [ "" -le "5" ] print "integer expression
# expected" and return 2, which `if` reads as FALSE: the FAIL message
# below would be skipped and the run would carry on to build and run the
# probe, reporting a read error as a pass. Fail closed instead.
if [ -z "$G7" ] || [ -z "$O7" ] || [ "$G7" -le "$O7" ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 7 left the finalize guard at or" \
         "ahead of the o_proj call (guard line '${G7:-none}', o_proj line '${O7:-none}')"
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
# STAY_PASS+TURN_FAIL array lengths further down (44 entries combined),
# counted off the control run's own output rather than derived, since
# bash arrays are not yet in scope this early in the script. Mutants
# 3-7 run the probe to completion (never crash) and so must each
# account for exactly this many PASS+FAIL lines, same reasoning as
# mutant 2's own total-count assertion; mutant 1 crashes before check3b
# and declares its own, smaller total instead (its own comment has the
# count).
PROBE_TOTAL_LINES=44

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
    # STAY_PASS/TURN_FAIL are this mutation's ACTUAL PASS/FAIL split,
    # measured against the probe rather than assumed from the code.
    # The one asymmetry worth naming: the "check_chain: layer B
    # (reindex) k_eff grows ..." line CANNOT detect this mutation,
    # because n_vis is a per-position (pos + 1) / cm computation that
    # does not live inside any of the three `if (mode == ...)`
    # conditions patched here. What does catch it is the "untouched by
    # C" publish-buffer line (C overwrites it, since its own guard is
    # patched too), check2's C-changed line, check3a's pair, and
    # check4b's kv_in-independence pair - plus the check3b crash below.
    STAY_PASS=(
        "check1a: at cm=2, the first-ever token's published k_eff is 0"
        "check1a: that k_eff==0 token's attn_out is the exact zero vector"
        "check1a: at cm=1, the first-ever token sees exactly the one block it just closed - k_eff 1, selecting block 0 (the block holding its own position)"
        "check1a: at cm=1, that same token's attn_out is not the zero vector - the block it just closed was already in the ring when step 5 read it"
        "check1a: at cm=2, the token that closes block 0 (pos 1) selects it - k_eff 1, selecting block 0"
        "check1a: ...and its attn_out is not the zero vector - block 0 was already final in the ring when step 5 read it, in the same iteration that closed it"
        "check1b: batch-vs-unbatched - published k_eff matches token for token"
        "check1b: batch-vs-unbatched - published Top-K indices match token for token"
        "check1b: batch-vs-unbatched - last token's attn_out is bit-identical"
        "check_chain: all three layers share the same physical ci (cache_idx aliasing from the prior task still holds)"
        "check_chain: layer A's own published k_eff grows 1,2,2,2(=topk) across the chunk"
        "check_chain: layer B (reindex) k_eff grows 1,2,2,2 across its own chunk - the per-position (pos + 1)/cm bound, exactly like layer A's, not a constant leaked in from A's already-finished call"
        "check_chain: layer B's own Top-K selection is not identical across all four of its own tokens"
        "check2: layer A's own attn_out is untouched by a change to layer B's (unrelated) indexer weights"
        "check2: perturbing layer B's own indexer weights changes layer B's own attn_out"
        "check5: layer C's (reuse) o_proj output for EVERY token of a 3-token chunk processed as one nt=3 call per layer is bit-identical to the same 3 positions processed one at a time (nt=1, layer-major per position) - not merely the chunk's last token"
        "check_ci_independence: two independent FULL layers get MUTUALLY DISTINCT ci (no accidental aliasing between them)"
        "check_ci_independence: the reuse layer's ci is aliased to its own declared source (A), not to the unrelated FULL layer (D)"
        "check_ci_independence: layer E's (reuse, source A) attn_out is untouched by a change to layer D's (unrelated, independently-aliased) indexer weights - per-ci addressing genuinely isolates the two roots, not merely the one this fixture happens to read"
        "check4a: layer D1 (FULL, decoder-region CSA2) still reads kv_in in its own step 2"
        "check4b: ...and so is the published k_eff"
    )
    TURN_FAIL=(
        "check_chain: after layer C's (reuse) call, all four tokens' published Top-K selections are still exactly layer B's own"
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
    # Total-count assertion, same idiom mutants 3-7 use below. Mutant 1
    # crashes before check3b ever prints, so STAY_PASS+TURN_FAIL here is
    # not the probe's full line count - the two arrays' own combined
    # length (27) is the only thing that would notice a line vanishing
    # outright, i.e. never printed as either PASS or FAIL. It also
    # re-checks the split: 21 PASS + 6 FAIL is what the crash leaves.
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
        "check1a: at cm=2, the first-ever token's published k_eff is 0"
        "check1a: that k_eff==0 token's attn_out is the exact zero vector"
        "check1a: at cm=1, the first-ever token sees exactly the one block it just closed - k_eff 1, selecting block 0 (the block holding its own position)"
        "check1a: at cm=1, that same token's attn_out is not the zero vector - the block it just closed was already in the ring when step 5 read it"
        "check1a: at cm=2, the token that closes block 0 (pos 1) selects it - k_eff 1, selecting block 0"
        "check1a: ...and its attn_out is not the zero vector - block 0 was already final in the ring when step 5 read it, in the same iteration that closed it"
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
        "check_chain: layer A's own published k_eff grows 1,2,2,2(=topk) across the chunk"
        "check_chain: layer B (reindex) k_eff grows 1,2,2,2 across its own chunk - the per-position (pos + 1)/cm bound, exactly like layer A's, not a constant leaked in from A's already-finished call"
        "check5: layer C's (reuse) o_proj output for EVERY token of a 3-token chunk processed as one nt=3 call per layer is bit-identical to the same 3 positions processed one at a time (nt=1, layer-major per position) - not merely the chunk's last token"
    )
    missing=0
    check_lines "$W/mut2/run.log" "mutant 2" "PASS" "${STAY_PASS[@]}" || missing=1
    check_lines "$W/mut2/run.log" "mutant 2" "FAIL" "${TURN_FAIL[@]}" || missing=1
    # Total-count assertion, same idiom mutants 3 through 7 already use below -
    # STAY_PASS+TURN_FAIL above must be EVERY line the probe can print,
    # not just the ones this mutation is most obviously about. A line
    # that is in NEITHER array is invisible to a future change that made
    # it turn FAIL under this mutation, which is why the overflow-guard/
    # bin-validation/F9 lines are all in STAY_PASS too: this mutation
    # never touches model.c or forward_state.c, so they all stay PASS.
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
echo "-- mutant 7: finalization moved back to the end of the per-token loop --"
build_and_run "$W/mut7" "mutant 7 (finalization reordered)"
MUT7_RC=$?
if [ "$MUT7_RC" -eq 2 ]; then
    exit 1
elif [ "$MUT7_RC" -eq 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL - mutant 7 PASSED; the probe cannot see this defect"
    RC=1
else
    TURN_FAIL=(
        "check1a: at cm=1, that same token's attn_out is not the zero vector - the block it just closed was already in the ring when step 5 read it"
        "check1a: ...and its attn_out is not the zero vector - block 0 was already final in the ring when step 5 read it, in the same iteration that closed it"
    )
    # The selection lines stay PASS by construction - n_vis is a direct
    # per-position computation no reordering reaches, and a one-candidate
    # selection is index 0 whatever the scores are - so they are named
    # here rather than left to the count, which cannot tell a line that
    # stayed green from one that never printed.
    STAY_PASS=(
        "check1a: at cm=1, the first-ever token sees exactly the one block it just closed - k_eff 1, selecting block 0 (the block holding its own position)"
        "check1a: at cm=2, the token that closes block 0 (pos 1) selects it - k_eff 1, selecting block 0"
    )
    missing=0
    check_lines "$W/mut7/run.log" "mutant 7" "FAIL" "${TURN_FAIL[@]}" || missing=1
    check_lines "$W/mut7/run.log" "mutant 7" "PASS" "${STAY_PASS[@]}" || missing=1
    if [ "$(grep -c '^FAIL' "$W/mut7/run.log")" != "${#TURN_FAIL[@]}" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - mutant 7 failed a different number of" \
             "lines than expected (${#TURN_FAIL[@]}); a reordering that reaches a" \
             "check no other mutation reaches is exactly what this arm looks for"
        missing=1
    fi
    if [ "$(grep -c '^PASS\|^FAIL' "$W/mut7/run.log")" != "$PROBE_TOTAL_LINES" ]; then
        echo "csa2_mode_dispatch_gate: FAIL - mutant 7 did not print all" \
             "$PROBE_TOTAL_LINES lines the probe normally prints"
        missing=1
    fi
    if [ "$missing" -ne 0 ]; then
        RC=1
    else
        echo "-- mutant 7 failed exactly the two absolute-attention-value checks, and no others --"
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "csa2_mode_dispatch_gate: FAIL"
    exit 1
fi
echo "csa2_mode_dispatch_gate: PASS"
exit 0
