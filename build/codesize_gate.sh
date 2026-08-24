#!/bin/bash
# Two size criteria, because one of them is not tight enough to warn.
#
#   A. The shipped GUI fits on a 1.44 MB floppy. This is the constraint
#      that matters: a program handed to someone on a disk is a
#      different object from one that is downloaded, and the moment it
#      does not fit the answer is a second disk and an installer, which
#      this project does not have.
#
#   B. The CLI's .text stays under a ceiling. Criterion A is at 68%
#      today, which means it would not redden until something had gone
#      very wrong; B is the early one, and it measures the thing the
#      target's 8 KB instruction cache actually cares about. Both
#      binaries link the same engine and, via
#      gcc_flag_parity_gate.sh, compile it with the same flags, so a
#      change that inflates one inflates both - B sees it first.
#
# A measures FILE bytes rounded up to the 512-byte cluster of a FAT12
# volume, because that is what the disk spends. B measures .text,
# because the import table, resources and page padding occupy the disk
# but never the instruction cache. They are different questions and take
# different numbers.
#
# NOT COUNTED under A, and it would matter if the answer were close: the
# volume's own overhead is subtracted (boot sector, two FATs, root
# directory - 33 of 2880 sectors), but a directory entry per file and
# the long-name entries a name over 8.3 needs are not.
#
# The flag that moved B most was -O3, and it is why the Makefile no
# longer passes it: 401,616 bytes of .text against -O2's 261,968, for
# byte-identical output.
#
# Exit codes:
#   0  PASS - both criteria met, margins printed
#   1  FAIL - either exceeded, or a measurement did not happen
#   2  SKIP - a binary has not been built
set -u
cd "$(dirname "$0")/.."

# A formatted 1.44 MB floppy: 2880 sectors of 512, less 33 for the boot
# sector, two 9-sector FATs and a 14-sector root directory.
CLUSTER=512
CAPACITY=$((2847 * 512))

# Everything that has to be on the disk for the GUI to run. The engine
# is linked in, not a DLL, so this is one file today; the loop exists
# because "it is one file" is a fact that can change quietly.
SHIP="kunkun98-gui.exe"

# 320 KB. Measured at 261,968 with -O2, so 22% of headroom - room for
# ordinary growth, well under the 401,616 that -O3 produced.
TEXT_CEIL=327680
CLI=llama98.exe
[ -f "$CLI" ] || CLI=llama98

FAIL=0

# ---- A: the floppy -------------------------------------------------
TOTAL=0
for f in $SHIP; do
    [ -f "$f" ] || { echo "codesize_gate: SKIP - $f not built; run make gui"; exit 2; }
    B=$(wc -c < "$f")
    C=$(( (B + CLUSTER - 1) / CLUSTER * CLUSTER ))
    printf '  %-20s %9d bytes -> %9d on disk\n' "$f" "$B" "$C"
    TOTAL=$((TOTAL + C))
done

# Positive control. A wc that returned nothing, or a SHIP list that lost
# its entries, totals zero and passes every ceiling. This program cannot
# be under 64 KB.
if [ "$TOTAL" -lt 65536 ]; then
    echo "codesize_gate: FAIL - the ship set totals $TOTAL bytes, which is"
    echo "  not a plausible size; the measurement is wrong and a PASS here"
    echo "  would mean nothing"
    exit 1
fi

APCT=$(awk -v t="$TOTAL" -v c="$CAPACITY" 'BEGIN{printf "%.1f", 100.0*t/c}')
if [ "$TOTAL" -le "$CAPACITY" ]; then
    echo "  A floppy:  $TOTAL of $CAPACITY (${APCT}%), $((CAPACITY - TOTAL)) free"
else
    echo "  A floppy:  FAIL - $TOTAL needs more than one 1.44MB disk"
    echo "             ($CAPACITY usable, over by $((TOTAL - CAPACITY)))"
    FAIL=1
fi

# ---- B: the instruction cache --------------------------------------
[ -f "$CLI" ] || { echo "codesize_gate: SKIP - no $CLI; run make native"; exit 2; }

text_of() { size -A "$1" 2>/dev/null | awk '$1 == ".text" { print $2; exit }'; }
T=$(text_of "$CLI")
if [ -z "${T:-}" ] && command -v wsl >/dev/null 2>&1; then
    ROOT=$(pwd)
    case "$ROOT" in /[a-z]/*) W="/mnt${ROOT}" ;; *) W="$ROOT" ;; esac
    T=$(wsl bash -lc "cd '$W' && size -A '$CLI' 2>/dev/null \
        | awk '\$1 == \".text\" { print \$2; exit }'")
fi
if [ -z "${T:-}" ]; then
    echo "codesize_gate: SKIP - no size(1) that can read $CLI"
    exit 2
fi
# Same control, for the same reason.
if [ "$T" -lt 65536 ]; then
    echo "codesize_gate: FAIL - .text read as $T bytes, not a plausible"
    echo "  size for this engine; the measurement is wrong"
    exit 1
fi

BPCT=$(awk -v t="$T" -v c="$TEXT_CEIL" 'BEGIN{printf "%.1f", 100.0*t/c}')
if [ "$T" -le "$TEXT_CEIL" ]; then
    echo "  B .text:   $T of $TEXT_CEIL (${BPCT}%)"
else
    echo "  B .text:   FAIL - $T over the $TEXT_CEIL ceiling (${BPCT}%)"
    echo "             If deliberate, raise TEXT_CEIL WITH the reason; the"
    echo "             number it protects is an 8 KB instruction cache."
    FAIL=1
fi

[ "$FAIL" -eq 0 ] && { echo "codesize_gate: PASS"; exit 0; }
echo "codesize_gate: FAIL"
exit 1
