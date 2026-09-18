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
# WHAT IT DOES NOT COVER: it is a text-level check of src/*.c. A borrow
# reached through a macro, or a scratch pointer recomputed at run time, is
# invisible to it. It supplements reading the call site; it does not replace
# it. There is no runtime arm - the rule is about which pointer is live,
# which no single execution can observe.
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
exit "$rc"
