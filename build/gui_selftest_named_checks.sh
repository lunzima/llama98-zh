#!/bin/bash
# Every check name the GUI selftest is expected to report must appear in
# the report. Reads the names from gui_selftest_expected_checks.txt beside
# this file; takes the report path as $1.
#
# WHY THIS IS ITS OWN SCRIPT RATHER THAN A LOOP INSIDE A GATE. Two gates
# run the selftest - gui_selftest_gate.sh against the shipped build and
# gui_noman_selftest_gate.sh against the unmanifested one - and they are
# the same gui/main.c, so the expected names are one list, not two. A
# second copy is what goes stale the next time a check is added.
#
# WHY NAMES AT ALL, when both gates already assert a floor on the count.
# A report of 400 PASS lines that never looked at the button labels
# satisfies every count-based rule, and so does one that quietly stopped
# running a third of itself while keeping the count up. The names are
# what say the count is made of the right things. This list was 944 lines
# of tests/test_gui_layout.py in the training tree; it moved here with the
# gate that runs it.
#
# Exit: 0 all present, 1 with the missing names listed.
set -u

REPORT=${1:?usage: gui_selftest_named_checks.sh <report>}
NAMES="$(dirname "$0")/gui_selftest_expected_checks.txt"

if [ ! -f "$NAMES" ]; then
    echo "  $NAMES is missing; the named-check list is not where this"
    echo "  script is. A clone that lost it would report PASS on nothing."
    exit 1
fi
if [ ! -s "$REPORT" ]; then
    echo "  no report at $REPORT to check names against."
    exit 1
fi

BAD=0
N=0
while IFS= read -r want; do
    case "$want" in ''|\#*) continue ;; esac
    N=$((N + 1))
    if ! grep -qF "$want" "$REPORT"; then
        echo "  the check '$want' is gone."
        BAD=1
    fi
done < "$NAMES"

if [ "$N" -eq 0 ]; then
    echo "  the names file parsed as zero entries - every rule in it is a"
    echo "  grep, and a grep over nothing passes. Refusing to report PASS."
    exit 1
fi

exit $BAD
