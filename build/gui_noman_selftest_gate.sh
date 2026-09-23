#!/bin/bash
# The GUI's own selftest, against the UNMANIFESTED build.
#
# WHY A SECOND BINARY, AND WHY IT IS NOT REDUNDANT WITH
# gui_selftest_gate.sh. The application manifest asks XP+ to theme the
# common controls, so on a modern host the manifest build draws the
# HOST's window through comctl32 v6. Stripping the manifest renders the
# same code through the unthemed comctl32 v5 painting - the version
# Windows 98 itself ships, and the one the deliverable actually gets.
#
# This is not a review aid. A status-bar-height bug (gui/layout.c's
# status_h parameter) sat red in this path while the manifest build
# stayed green the whole time, because only one of the two was ever run
# by hand. tests/test_gui_layout.py's docstring says the same thing about
# NOMAN_BIN's own comment: it "is the build that represents the target,
# and it is the one nobody thinks to run".
#
# HOW IT GOT HERE. This was test_noman_window_selftest, in the training
# tree, where it built ./kunkun98-noman out of THAT tree's src/ - which
# CLAUDE.md rule zero retired, so the build died on src/RETIRED.h and the
# test went red for a reason that had nothing to do with the GUI. The
# binary it means to test is built here now, from the live sources.
#
# THE BINARY IS BUILT, NOT ASSUMED, AND SO IS ITS .rc. The Makefile's
# kunkun98-noman target derives its .rc from gui/kunkun98.rc by deleting
# the manifest line, and FAILS if that strip stops matching - because a
# "noman" binary that still carries the manifest would run the manifest
# path under a different name and report success. This gate re-checks
# the same property on the ARTEFACT instead of trusting the name: the
# manifest's assemblyIdentity must be ABSENT from the binary. Without
# that check, a Makefile whose strip silently stopped matching would
# make this gate a duplicate of gui_selftest_gate.sh that reports PASS.
#
# WHAT IS ASSERTED - the same rules as gui_selftest_gate.sh, for the same
# reasons, plus the manifest check above:
#   1. the report has a TRAILER. A crash mid-run leaves a truncated file
#      with no FAIL line in it, which greps exactly like a clean pass.
#   2. fails 0.
#   3. checks >= FLOOR. A selftest that quietly stops running most of
#      itself still reports "fails 0".
#   4. skips <= SKIP_CEIL.
#   5. the two predicates that caught shipped bugs are present BY NAME,
#      so deleting one is a deliberate act rather than a count nobody
#      reads.
#   6. the parser can see a failure, asserted against a synthetic report -
#      every rule here is a grep, and a grep that matches nothing passes.
#
# SKIPPING IS WORSE THAN FAILING, SO IT IS NOT OFFERED. The first version
# of this gate's predecessor in the training tree asked the banner for a
# model it did not need, which made it SKIP on a machine where the
# original ran - a red silently became a green. Here, exit 2 means the
# PLATFORM cannot build the artefact (not Windows, no windres) and says
# so; a build that was attempted and failed is exit 1, never 2. A gate
# that skips when it could have run is strictly worse than one that
# reddens, because the red is the only thing that ever gets fixed.
#
# Exit codes:
#   0  PASS
#   1  FAIL
#   2  SKIP - not a Windows host, so no GUI binary can exist
set -u
cd "$(dirname "$0")/.."

if [ "${OS:-}" != "Windows_NT" ] && ! command -v windres >/dev/null 2>&1; then
    echo "gui_noman_selftest_gate: SKIP - no windres, so the GUI cannot"
    echo "  be built on this host. The thing under test is a Win32 GUI."
    exit 2
fi

LZ_ENV=$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="the unmanifested GUI self-check" bash build/tree_lock.sh acquire --print-env) || exit 2
eval "$LZ_ENV"
trap 'bash build/tree_lock.sh release' EXIT

BIN=./kunkun98-noman.exe
[ -x "$BIN" ] || BIN=./kunkun98-noman
[ -x "$BIN" ] || {
    echo "gui_noman_selftest_gate: FAIL - no kunkun98-noman binary."
    echo "  make check builds it via this gate's dependency; a binary"
    echo "  missing here means the BUILD failed, not that there is"
    echo "  nothing to test. Build it with 'make kunkun98-noman'."
    exit 1
}

# The manifest must really be gone. See the header: without this, a
# Makefile whose strip stopped matching would build the manifest binary
# and this gate would pass while testing the other one.
if grep -qa 'assemblyIdentity' "$BIN"; then
    echo "gui_noman_selftest_gate: FAIL - $BIN carries an application"
    echo "  manifest (assemblyIdentity is in the binary). This is the"
    echo "  MANIFEST build under another name: the comctl32 v5 path this"
    echo "  gate exists for was never exercised."
    exit 1
fi

# The binary must not predate the sources it tests, nor the .rc it was
# stripped from - a stale noman binary is the manifest build's age, not
# this one's.
STALE=$(find gui common -name '*.c' -newer "$BIN" 2>/dev/null | head -3)
if [ -z "$STALE" ] && [ gui/kunkun98.rc -nt "$BIN" ]; then
    STALE=gui/kunkun98.rc
fi
if [ -n "$STALE" ]; then
    echo "gui_noman_selftest_gate: FAIL - $BIN is older than its own"
    echo "  sources:"
    printf '  %s\n' $STALE
    echo "  Rebuild with 'make kunkun98-noman'."
    exit 1
fi

FLOOR=432          # ratchet: raise when checks are added, never lower.
                   # Same number as gui_selftest_gate.sh - one gui/main.c
                   # is behind both binaries, so a difference between the
                   # two counts is itself the finding.
SKIP_CEIL=1        # the statusbar strip comparison needs a second run

REPORT=build/gui_noman_selftest.txt
rm -f "$REPORT"
"$BIN" --selftest "$REPORT" >/dev/null 2>&1
RC=$?

if [ ! -s "$REPORT" ]; then
    echo "gui_noman_selftest_gate: FAIL - the selftest wrote no report"
    echo "  (exit $RC). It did not run; it did not pass."
    exit 1
fi

# The trailer first, because everything below it is a grep over a file
# that may have stopped early.
CHECKS=$(sed -n 's/^checks \([0-9]*\)$/\1/p' "$REPORT" | tail -1)
FAILS=$(sed -n  's/^fails \([0-9]*\)$/\1/p'  "$REPORT" | tail -1)
SKIPS=$(sed -n  's/^skips \([0-9]*\)$/\1/p'  "$REPORT" | tail -1)
if [ -z "$CHECKS" ] || [ -z "$FAILS" ] || [ -z "$SKIPS" ]; then
    echo "gui_noman_selftest_gate: FAIL - no trailer in $REPORT."
    echo "  The run ended before its summary, so the absence of a FAIL"
    echo "  line means nothing. Last lines:"
    tail -5 "$REPORT" | sed 's/^/    /'
    exit 1
fi

BAD=0

if [ "$FAILS" -ne 0 ]; then
    echo "gui_noman_selftest_gate: $FAILS failing check(s), in the"
    echo "  UNTHEMED path the target actually gets:"
    grep '^FAIL' "$REPORT" | sed 's/^/  /'
    BAD=1
fi

if [ "$CHECKS" -lt "$FLOOR" ]; then
    echo "gui_noman_selftest_gate: only $CHECKS checks ran, floor is"
    echo "  $FLOOR. Checks went missing. 'fails 0' over a shrunken suite"
    echo "  is not the same statement as 'fails 0'."
    BAD=1
fi

if [ "$SKIPS" -gt "$SKIP_CEIL" ]; then
    echo "gui_noman_selftest_gate: $SKIPS skips, ceiling is $SKIP_CEIL."
    grep '^SKIP' "$REPORT" | sed 's/^/  /'
    BAD=1
fi

# Every check name the selftest is expected to report - the same 205-name
# list gui_selftest_gate.sh reads, from the same file, because both
# binaries run the same gui/main.c. Named, so removing one has to be
# argued for; a count would let this arm lose a third of itself quietly.
if ! bash build/gui_selftest_named_checks.sh "$REPORT"; then
    echo "gui_noman_selftest_gate: names above are missing from the"
    echo "  unmanifested build's report."
    BAD=1
fi

if [ "$RC" -ne 0 ] && [ "$BAD" -eq 0 ]; then
    echo "gui_noman_selftest_gate: FAIL - the report is clean but the"
    echo "  binary exited $RC. One of the two is wrong and this gate"
    echo "  reads the report, so it is the report that cannot be trusted."
    BAD=1
fi

# Control: the rules above are greps, and a grep that matches nothing
# reports success. Feed the same predicates a report that IS broken.
CTL=$(printf 'PASS a\nFAIL b\nchecks 2\nfails 1\nskips 0\n')
CF=$(printf '%s\n' "$CTL" | sed -n 's/^fails \([0-9]*\)$/\1/p' | tail -1)
CT=$(printf 'PASS a\nchecks 2\n' | sed -n 's/^skips \([0-9]*\)$/\1/p' | tail -1)
if [ "$CF" != "1" ] || [ -n "$CT" ]; then
    echo "gui_noman_selftest_gate: FAIL - the parser reports fails=$CF"
    echo "  on a report with one failure, and trailer='$CT' on a"
    echo "  truncated one. It cannot see the states it exists to catch."
    BAD=1
fi

if [ "$BAD" -ne 0 ]; then
    echo "gui_noman_selftest_gate: FAIL"
    exit 1
fi

echo "gui_noman_selftest_gate: PASS"
echo "  $CHECKS checks, $FAILS fails, $SKIPS skips, in the unmanifested"
echo "  build (comctl32 v5, the painting Windows 98 gets)."
echo "  Manifest absent from the binary (assemblyIdentity not found)."
echo "  floor $FLOOR, and the parser reddens on a planted failure."
exit 0
