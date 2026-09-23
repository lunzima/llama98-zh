#!/bin/bash
# The s->wscr borrow rule.
#
# lz_t_f32(t, SCRATCH) BORROWS: it returns t->f for LZ_FMT_F32 and widens
# every other format INTO SCRATCH, handing SCRATCH back. A second borrow of
# the same scratch therefore REPLACES the first rather than joining it, and
# the earlier pointer starts reading the later tensor's bytes with nobody
# changing its value. Every shipped checkpoint is packed (no F32 tensor at
# all), so the scratch path is the one that always runs and an F32 export is
# what hides this - forward_kda's decay gate read dt_bias where it wanted
# A_log and saturated at both ends of (-5, 0).
#
# The rule and what it cannot see are stated in full in
# build/wscr_borrow_check.py's docstring. Two things about the shape of this
# gate are deliberate and worth keeping if it is ever rewritten:
#
#   * It counts the borrows it looked at, not just the violations it found.
#     A checker whose regex stops matching exits 0 and reads exactly like a
#     clean tree - that is the failure this project keeps re-learning, and
#     it is why "0 borrows scanned" is exit 2 here instead of exit 0.
#
#   * The exemption in the checker carries the LINE it was granted for.
#     Move that code and the gate goes red on a stale exemption rather than
#     green on a rule that no longer covers what it used to.
#
# THE SECOND RULE, and the reason this gate is not only about local live
# ranges. The rule above ends with the function; a borrow STORED INTO A
# `static` OBJECT ends with the process, and that is the shape a
# pointer-keyed cache is built out of - `if (slot[i].w == w) return
# slot[i].p;` is an unbounded identity claim about a pointer that a scratch
# reuse makes false. src/ops_norm.c's two norm-weight pools and
# src/ops_gdn.c's gdn_shadow are the three instances; each store is
# reported and each is exempted by LINE with its own argument, so the next
# one added goes red rather than waiting for a model comparison to notice.
#
# WHAT IT DOES NOT COVER: it is a text-level check of src/*.c. A borrow
# reached through a macro, or a scratch pointer recomputed at run time, is
# invisible to it. A stored pointer whose declaration carries no `const` on
# the pointee is not read as a borrow, and a borrowed pointer stored into a
# non-static object that nonetheless outlives the scratch - a heap struct
# living as long as the model - is not reported either; the checker's
# docstring lists both next to its other blind spots. It supplements reading
# the call site; it does not replace it. There is no runtime arm - the rule
# is about which pointer is live, which no single execution can observe.
#
# Needs no build, no model - runs in well under a second.
#
# Exit codes:
#   0  PASS - no borrow outlives a second borrow of the same scratch
#   1  FAIL - at least one does, or an exemption went stale
#   2  the check could not run (no python3, no src/, nothing scanned)
set -u
cd "$(dirname "$0")/.."

command -v python3 >/dev/null 2>&1 || {
    echo "wscr_borrow_gate: SKIP - no python3"; exit 2; }

python3 build/wscr_borrow_check.py .
rc=$?
if [ "$rc" -eq 0 ]; then
    echo "wscr_borrow_gate: PASS"
elif [ "$rc" -eq 1 ]; then
    echo "wscr_borrow_gate: FAIL"
fi
[ "$rc" -eq 0 ] || exit "$rc"

# THE MUTATIONS, and there are two because the storage-point rule makes two
# promises. A rule that only ever reads a clean tree is a rule nobody has
# seen work, so both arms plant the failure and require it reported:
#
#   a store of a borrowed `const float *` into a static object, with no
#   exemption - the shape every pointer-keyed cache is built out of; and
#
#   an exemption whose recorded line holds no store. Move a store and the
#   exemption must go stale and say so, rather than keep matching by name
#   at whatever line the store landed on - which is the property the whole
#   line-anchored form exists for.
#
# The trees are copies under mktemp; the real src/ is never written to.
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT

mkdir -p "$W/mut/src"
cp src/*.c "$W/mut/src/"
cat >> "$W/mut/src/ops_norm.c" <<'PLANTED'

/* Planted by build/wscr_borrow_gate.sh: a borrowed pointer cached in a
   static object with no content verification and no exemption. */
static const float *g_planted_src;
static void g_planted_store(const float *w) { g_planted_src = w; }
PLANTED

out=$(python3 build/wscr_borrow_check.py "$W/mut" 2>&1)
rc=$?
if [ "$rc" -ne 1 ] || ! printf '%s\n' "$out" | grep -q 'g_planted_src = w'; then
    echo "wscr_borrow_gate: FAIL - a planted store into a static object was"
    echo "  not reported (rc=$rc):"
    printf '%s\n' "$out" | tail -6
    exit 1
fi
echo "  mutation: planted static store reported"

sed 's/^STORE_EXEMPT = {/STORE_EXEMPT = {\n    ("ops_norm.c", 999999): ("bogus",),/' \
    build/wscr_borrow_check.py > "$W/stale_check.py"
out=$(python3 "$W/stale_check.py" . 2>&1)
rc=$?
if [ "$rc" -ne 1 ] || ! printf '%s\n' "$out" | grep -q 'STALE EXEMPTION'; then
    echo "wscr_borrow_gate: FAIL - an exemption recorded for a line that"
    echo "  holds no store was not reported (rc=$rc):"
    printf '%s\n' "$out" | tail -6
    exit 1
fi
echo "  mutation: stale exemption line reported"

echo "wscr_borrow_gate: PASS (both mutations red)"
exit 0
