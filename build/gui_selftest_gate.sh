#!/bin/bash
# Run the GUI's own selftest, and treat its verdict as a gate.
#
# WHY THIS EXISTS. gui/main.c carries 432 runtime checks - it builds the
# real windows, presses the real keys and reads the real controls back.
# Nothing ran it. It was reachable only by typing
# `kunkun98-gui --selftest <path>` by hand, so its result was whatever
# it had been the last time somebody remembered to look.
#
# WHAT THAT COST, concretely, because this is not a hypothetical: two of
# those checks were RED for as long as the beep checkbox had existed.
# The checkbox took mnemonic P, which Temperature already had - Windows
# answers a duplicated mnemonic by cycling focus between the two
# controls instead of activating either, so Alt+P looked like a dead
# key. And the checkbox was never added to the Tab chain, so Tab from
# the think box went somewhere the visual order did not predict. Both
# were found by a user pressing the key, not by the suite that already
# knew.
#
# WHAT IS ASSERTED
#   1. The binary runs the selftest to completion. A crash mid-run
#      leaves a truncated report with no FAIL line in it, which greps
#      exactly like a clean pass - so the trailer must be PRESENT, not
#      merely free of failures.
#   2. fails 0.
#   3. checks >= FLOOR. A selftest that quietly stops running most of
#      itself still reports "fails 0".
#   4. skips <= SKIP_CEIL. A check that turns itself off is a check
#      that stopped carrying anything.
#   5. The two predicates that caught the bugs above are present BY
#      NAME. Deleting one is then a deliberate act rather than a count
#      that nobody reads.
#   6. The parser can see a failure. Asserted against a synthetic
#      report, because every rule here is a grep and a grep that
#      matches nothing passes.
#
# WINDOWS ONLY, and that is not a limitation to work around: the thing
# under test is a Win32 GUI. Off Windows there is no binary and the
# gate skips.
#
# Exit codes:
#   0  PASS
#   1  FAIL
#   2  SKIP - no GUI binary
set -u
cd "$(dirname "$0")/.."

eval "$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="the GUI self-check" bash build/tree_lock.sh acquire --print-env)" || exit 2
trap 'bash build/tree_lock.sh release' EXIT

BIN=./kunkun98-gui.exe
[ -x "$BIN" ] || BIN=./kunkun98-gui
[ -x "$BIN" ] || {
    echo "gui_selftest_gate: SKIP - no kunkun98-gui; make gui first"
    exit 2
}

# The binary must not predate the sources it is supposed to be testing.
# Running last week's exe against this week's dialog is the failure this
# whole file exists to stop, one level up.
STALE=$(find gui common -name '*.c' -newer "$BIN" 2>/dev/null | head -3)
if [ -n "$STALE" ]; then
    echo "gui_selftest_gate: FAIL - $BIN is older than its own sources:"
    printf '  %s\n' $STALE
    echo "  Rebuild with 'make gui'. A stale binary passes checks that"
    echo "  the current code would fail."
    exit 1
fi

FLOOR=432          # ratchet: raise when checks are added, never lower
                   # 421 -> 432 when the sound window's read-back joined
                   # st_music_dialog (2026-09-14)
SKIP_CEIL=1        # the statusbar strip comparison needs a second run

REPORT=build/gui_selftest.txt
rm -f "$REPORT"
"$BIN" --selftest "$REPORT" >/dev/null 2>&1
RC=$?

if [ ! -s "$REPORT" ]; then
    echo "gui_selftest_gate: FAIL - the selftest wrote no report"
    echo "  (exit $RC). It did not run; it did not pass."
    exit 1
fi

# The trailer, first, because everything below it is a grep over a file
# that may have stopped early.
CHECKS=$(sed -n 's/^checks \([0-9]*\)$/\1/p' "$REPORT" | tail -1)
FAILS=$(sed -n  's/^fails \([0-9]*\)$/\1/p'  "$REPORT" | tail -1)
SKIPS=$(sed -n  's/^skips \([0-9]*\)$/\1/p'  "$REPORT" | tail -1)
if [ -z "$CHECKS" ] || [ -z "$FAILS" ] || [ -z "$SKIPS" ]; then
    echo "gui_selftest_gate: FAIL - no trailer in $REPORT."
    echo "  The run ended before its summary, so the absence of a FAIL"
    echo "  line means nothing. Last lines:"
    tail -5 "$REPORT" | sed 's/^/    /'
    exit 1
fi

BAD=0

if [ "$FAILS" -ne 0 ]; then
    echo "gui_selftest_gate: $FAILS failing check(s):"
    grep '^FAIL' "$REPORT" | sed 's/^/  /'
    BAD=1
fi

if [ "$CHECKS" -lt "$FLOOR" ]; then
    echo "gui_selftest_gate: only $CHECKS checks ran, floor is $FLOOR."
    echo "  Checks went missing. 'fails 0' over a shrunken suite is not"
    echo "  the same statement as 'fails 0'."
    BAD=1
fi

if [ "$SKIPS" -gt "$SKIP_CEIL" ]; then
    echo "gui_selftest_gate: $SKIPS skips, ceiling is $SKIP_CEIL."
    grep '^SKIP' "$REPORT" | sed 's/^/  /'
    BAD=1
fi

# The two that were red. Named, so removing one has to be argued for.
for want in \
    "settings: no two controls in the dialog share a mnemonic" \
    "settings: Tab visits every row in visual order"
do
    if ! grep -qF "$want" "$REPORT"; then
        echo "gui_selftest_gate: the check '$want' is gone."
        echo "  It caught a shipped bug. Deleting it needs a reason."
        BAD=1
    fi
done

if [ "$RC" -ne 0 ] && [ "$BAD" -eq 0 ]; then
    echo "gui_selftest_gate: FAIL - the report is clean but the binary"
    echo "  exited $RC. One of the two is wrong and this gate reads the"
    echo "  report, so it is the report that cannot be trusted."
    BAD=1
fi

# Control: the rules above are greps, and a grep that matches nothing
# reports success. Feed the same predicates a report that IS broken.
CTL=$(printf 'PASS a\nFAIL b\nchecks 2\nfails 1\nskips 0\n')
CF=$(printf '%s\n' "$CTL" | sed -n 's/^fails \([0-9]*\)$/\1/p' | tail -1)
CT=$(printf 'PASS a\nchecks 2\n' | sed -n 's/^skips \([0-9]*\)$/\1/p' | tail -1)
if [ "$CF" != "1" ] || [ -n "$CT" ]; then
    echo "gui_selftest_gate: FAIL - the parser reports fails=$CF on a"
    echo "  report with one failure, and trailer='$CT' on a truncated"
    echo "  one. It cannot see the states it exists to catch."
    BAD=1
fi

if [ "$BAD" -eq 0 ]; then
    echo "gui_selftest_gate: PASS ($CHECKS checks, 0 fails, $SKIPS skip;"
    echo "  floor $FLOOR, and the parser reddens on a planted failure)"
    exit 0
fi
echo "gui_selftest_gate: FAIL"
exit 1
