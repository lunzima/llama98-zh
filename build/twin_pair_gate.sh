#!/bin/bash
# Every x86 operator has BOTH a gcc version and a Watcom version.
#
# The MMX/SSE/SSE2 kernels are written twice - once as gcc intrinsics
# and once as wcc386 `#pragma aux` assembly - and the kernel matrix
# already records that as two separate cells per ISA: `<isa>I` for the
# intrinsic and `<isa>A` for the assembly. What nothing checked is that
# the two cells of a pair are BOTH present. An operator implemented on
# one side only builds and runs everywhere, because whichever compiler
# is missing its cell falls back to the C reference and stays
# bit-identical while doing it - the same silence that let a whole tier
# of hand-written assembly stay out of the Watcom build for months
# (CLAUDE.md, the -D__MMX__=1 note).
#
# THE PAIRING, not the cell: `mmxI` being gcc-only and `mmxA` being
# Watcom-only is the DESIGN, not a defect. A pair is satisfied when the
# intrinsic cell carries G and the assembly cell carries W, or when
# either is registered impossible (`i`) - a cell that cannot exist does
# not need a twin.
#
# A COUNT OF THE IMPLEMENTED POPULATION IS PRINTED AND CHECKED, because
# "0 unpaired" is also what a matrix of nothing-but-impossible cells
# reports. If the implemented pair count ever drops below the floor
# below, this gate has stopped measuring and says so instead of passing.
#
# Exit codes:
#   0  PASS - every x86 (unit, ISA) pair has both sides or is impossible
#   1  FAIL - a pair is one-sided, or the implemented population shrank
#   2  SKIP - no binary, or it cannot print the matrix
set -u
cd "$(dirname "$0")/.."

BIN=llama98.exe
[ -x "$BIN" ] || BIN=llama98
[ -x "$BIN" ] || { echo "twin_pair_gate: SKIP - no llama98 binary"; exit 2; }

MODELS_ROOT=${LZ_MODELS_ROOT:-/e/LLM/models}
MODEL=${LZ_NORM_INT_GATE_MODEL:-$MODELS_ROOT/kmr20}
[ -d "$MODEL" ] || { echo "twin_pair_gate: SKIP - no model at $MODEL"; exit 2; }

# Implemented pairs. A floor, not an equality: adding an operator raises
# it and does not redden the gate, losing one does. RATCHETED when a
# pair lands, so a delivered kernel cannot quietly disappear later - the
# gate was created at 21 and the number below is what the tree carries
# now.
FLOOR=37

# CELLS REGISTERED todo, which is what a cell looks like after the
# "partial relief obliges a kernel" standard reopens one and before
# anyone writes it. This gate counts (unit, ISA) PAIRS, not cells.
#
# A CEILING rather than a floor: the number owed may shrink freely and
# may not grow unnoticed. Raising it is the deliberate act of accepting
# a debt, which is the only way it should ever go up. Before this state
# existed the gate had two answers, implemented or impossible, and a
# reopened cell read as an unpaired kernel.
#
# It reached ZERO when expfx's last cell landed - every x86 cell either
# written on both sides or registered impossible with a reason - and
# went back up when two NEW operator units arrived: recur and epi. That
# is the mechanism working, not failing. A unit lands its gcc side
# first and owes the Watcom twin for as long as it takes to write, and
# this number is what stops "for as long as it takes" becoming
# permanent. It is expected to fall as those two are paired; it should
# not be raised again without a unit to point at.
TODO_CEIL=5

# CELLS REGISTERED declined (d) - the one exception to "every cell is
# written or impossible": a body was WRITTEN and MEASURED and it is
# slower than the scalar C loop it would replace, so shipping it is a
# regression, not relief. This is the 5800X negative-optimization veto,
# and it is the only reason a cell may stay empty on both sides without
# being impossible. A ceiling like todo's, for the same reason:
# declining is a deliberate act, and the number may not grow without
# someone pointing at a fresh measurement.
DEC_CEIL=2

"./$BIN" "$MODEL" --kernel-matrix 2>&1 | grep "^cell" | awk -v floor="$FLOOR" -v tceil="$TODO_CEIL" -v dceil="$DEC_CEIL" '
$3=="mmxI"||$3=="sse2I"||$3=="sseI" {
    isa=substr($3,1,length($3)-1); I[$2" "isa]=$5; SI[$2" "isa]=$4 }
$3=="mmxA"||$3=="sse2A"||$3=="sseA" {
    isa=substr($3,1,length($3)-1); A[$2" "isa]=$5; SA[$2" "isa]=$4 }
END {
    bad=0; impl=0; imp=0; dec=0; todo=0;
    for (k in I) {
        if (SI[k]=="i" || SA[k]=="i") { imp++; continue }
        if (SI[k]=="d" || SA[k]=="d") { dec++; continue }
        if (SI[k]=="t" || SA[k]=="t") { todo++; continue }
        impl++;
        g = (index(I[k],"G") > 0);
        w = (index(A[k],"W") > 0);
        if (!(g && w)) {
            printf "FAIL: %-18s intrinsic=%s assembly=%s - %s\n", k, I[k], A[k],
                   (g ? "no Watcom twin" : (w ? "no gcc twin" : "neither side"));
            bad++;
        }
    }
    printf "x86 pairs: %d implemented, %d registered impossible, %d declined, %d owed\n",
           impl, imp, dec, todo;
    if (todo > tceil) {
        printf "FAIL: %d cells registered todo, ceiling is %d - a cell the\n", todo, tceil;
        printf "      ISA can partly reach is owed, and the number owed may\n";
        printf "      not grow without someone deciding it should\n";
        bad++;
    }
    if (dec > dceil) {
        printf "FAIL: %d cells registered declined, ceiling is %d - a cell\n", dec, dceil;
        printf "      measured slower than C is a decision, and the number\n";
        printf "      declined may not grow without a fresh measurement\n";
        bad++;
    }
    if (impl < floor) {
        printf "FAIL: implemented population is %d, below the floor of %d -\n", impl, floor;
        printf "      this gate has stopped measuring rather than found nothing\n";
        bad++;
    }
    if (bad) { print "twin_pair_gate: FAIL"; exit 1 }
    print "twin_pair_gate: PASS";
}'
