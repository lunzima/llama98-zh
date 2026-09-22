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
# NEITHER MEASURES RUNTIME MEMORY, and that was checked rather than
# assumed to be a gap. The state is demand-sized, not ceiling-sized:
# 3.5 MB at --ctx 512 and the same 3.5 MB at --ctx 32768, with the KV
# slot count following the generation instead (64 slots at -n 1, 217 at
# -n 200, +0.1 MB). So there is no reservation-against-the-maximum and
# no shadow copy to find, and the buffer pairs in the base 3.5 MB
# (x/xb/xb2, hb/hb2, ktmp/vtmp) are the ordinary residual-plus-scratch
# shape. A gate would have nothing to hold down.
#
# NOT COUNTED under A, and it would matter if the answer were close: the
# volume's own overhead is subtracted (boot sector, two FATs, root
# directory - 33 of 2880 sectors), but a directory entry per file and
# the long-name entries a name over 8.3 needs are not.
#
# The flag that moved B most was -O3, and it is why the Makefile no
# longer passes it: 401,616 bytes of .text against -O2's 261,968, for
# byte-identical output. Both arms of that pair were one tree, measured
# together, under one optimisation level - it is a RATIO (-O3 is 1.53x
# -O2), and that is what the decision rests on. The absolute -O2 figure
# has since moved: the same uniform -O2 build measures 359,760 today
# (see the re-baseline note above TEXT_CEIL for the replay command).
#
# Exit codes:
#   0  PASS - both criteria met, margins printed
#   1  FAIL - either exceeded, or a measurement did not happen
#   2  SKIP - a binary has not been built
set -u
cd "$(dirname "$0")/.."

eval "$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="the size ceilings" bash build/tree_lock.sh acquire --print-env)" || exit 2
trap 'bash build/tree_lock.sh release' EXIT

# A formatted 1.44 MB floppy: 2880 sectors of 512, less 33 for the boot
# sector, two 9-sector FATs and a 14-sector root directory.
CLUSTER=512
CAPACITY=$((2847 * 512))

# Everything that has to be on the disk for the GUI to run. The engine
# is linked in, not a DLL, so this is one file today; the loop exists
# because "it is one file" is a fact that can change quietly.
#
# THIS IS THE mingw BUILD, NOT THE ONE THAT GOES ON THE FLOPPY. A Win98
# machine runs kunkun98-w.exe, the Watcom build, and the two are not the
# same size: 1,003,685 against 793,088, a 21% gap between the artifact
# measured and the artifact shipped. The floppy answer below is
# therefore conservative today - the real disk holds MORE headroom than
# it reports - but conservative by accident is still the wrong file, and
# the sign is not guaranteed: the two compilers have moved apart before.
# Same for section B, which ceilings llama98.exe (610,930) while the
# shipping CLI is llama98-w.exe (486,400).
#
# Not changed here, because switching the measured artifact needs the
# Watcom binaries present for the gate to run at all, and this gate is
# already a `make gui` prerequisite away from SKIPping. Recorded so the
# number stops reading as the shipped one.
SHIP="kunkun98-gui.exe"

# 340 KB. Re-measured 2026-09-22 at 330,432 on the artifact this gate
# actually reads - `make native`'s llama98.exe - so 17,728 bytes (5.4%) of
# headroom. The four things a number like this owes the next reader:
#
# WHICH ARTIFACT, UNDER WHICH FLAGS. llama98.exe as `make native` builds
# it, which is NOT one optimisation level: Makefile:174 compiles the cold
# sources at -Os (X86_64_CFLAGS) and Makefile:183 the hot TUs at -O2
# (X86_64_HOT_CFLAGS), a per-TU split for an 8 KB I-cache. The ceiling
# derives from THAT measurement, because that is the file $CLI points at.
#
# The ceiling it replaces (327,680 = 320 KB) was derived from a UNIFORM
# -O2 build - "Measured at 261,968 with -O2, so 22% of headroom" - and
# that is the defect this fixes: the old baseline and the old artifact
# were not the same configuration. Replaying both numbers today, same
# source tree (ca02bda):
#
#   uniform -O2   (make llama98 X86_64_CFLAGS="-O2 -march=x86-64 \
#                  -mtune=znver3 -std=c99 -Wall -Wextra -Isrc \
#                  -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 \
#                  -DLZ_AVX2_TU=1")                              359760
#   shipping      (make native)                                 330432
#     -> the per-TU -Os tier is worth 29,328 bytes (8.2%) today
#     -> against the old note's 261,968, the SOURCE has grown 97,792
#        bytes; the FLAG SPLIT is what keeps the shipping file 29 KB
#        below a uniform -O2 build of the same source. The old 320 KB
#        ceiling sat 32 KB BELOW today's uniform -O2 artifact, i.e. it
#        had long stopped describing any configuration this tree builds.
#
# WHY THIS MARGIN. The old 22% was chosen when nothing ran this gate;
# that cushion was standing in for an alarm. Since 1bad3c3 the gate is a
# prerequisite of `check`, so it IS the alarm and the margin only has to
# absorb ordinary growth between re-baselines: 5.4% is about ten times
# the largest single feature measured this cycle (per-layer compression
# ratio, +1,600 bytes) and still catches a flag-level mistake (the -O3
# experiment that produced 401,616).
#
# WHY THE OLD BREACH WENT UNSEEN. This gate sat in gate_wiring_gate.sh's
# ci.sh-only list behind "builds three toolchains to compare .text" - a
# reason false of the script (in code position it invokes make/wcc386/
# wlink zero times; it measures two already-built files). Nothing ran it,
# so .text crossed 327,680 with no line anywhere saying so. 1bad3c3 wired
# it into `check` and corrected that line; this re-baseline comes after
# that, not instead of it: the instrument was fixed first, then the
# number was read.
#
# REPLAYING EITHER CEILING: criterion B is `size -A "$CLI" | awk '$1 ==
# ".text" { print $2; exit }'` on a `make native` build (this gate does
# exactly that, with a WSL fallback when the host has no GNU size).
# Criterion A is a FILE-BYTES measurement against a physical constant -
# CAPACITY above is 2880 sectors of 512 less 33 for the boot sector, two
# FATs and the root directory - so it needs no re-baselining and has no
# configuration to drift: it asks whether `make gui`'s kunkun98-gui.exe
# still fits on one 1.44 MB disk.
TEXT_CEIL=348160
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

# Provenance, before the measurement. This gate reads a binary it did not
# build, and a number off the wrong binary is worse than no number: a
# .text of 401,744 was reported as over the ceiling in this project when
# what had happened was that a gate which builds at -O3 had left its last
# build behind. build/native holds the three ISA objects llama98 links
# and every rule that produces the binary produces them too, so a binary
# standing next to no object directory came from a build whose flags
# nothing here can name.
if [ ! -d build/native ]; then
    echo "codesize_gate: FAIL - $CLI exists but build/native/ does not, so"
    echo "  this binary is left over from a build that wiped its objects"
    echo "  and its flags are unknown. Measuring it would report a size"
    echo "  for a configuration that is not the shipping one."
    echo "  Run 'make llama98' and try again."
    exit 1
fi

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
