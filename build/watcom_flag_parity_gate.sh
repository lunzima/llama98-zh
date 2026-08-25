#!/bin/bash
# The Watcom GUI and the Watcom CLI compile the engine with the same
# flags.
#
# gcc_flag_parity_gate.sh asks this of the gcc pair and its header has
# the reasoning: both targets link the same engine sources, and when the
# flags drift the shipped window runs a different engine from the
# shipped command line while both claim to be the same build. Nothing
# reports it, because comparing OUTPUT does not: -O2 and -O3 produced
# byte-identical logits on two models there.
#
# THIS IS THE PAIR THAT ACTUALLY SHIPS TO THE TARGET. The gcc build is
# the development one; llama98-w.exe and kunkun98-w.exe are what a Win98
# machine runs, and their flags come from two separate Make variables -
# CLI_FLAGS and GUI_FLAGS, CLI_FLAGS_HIGH and GUI_FLAGS_HIGH - written
# out one beside the other. Two definitions that must agree and nothing
# checking that they do is the same shape the gcc side was in.
#
# WHAT IS COMPARED: the CPU floor (-4r/-6r), the optimisation letters
# (-otexan), the model and language switches (-za99, -zp4, -we) and
# every -D. Not the whole command line: the GUI legitimately adds -bm
# for the multithreaded runtime that gui/worker.c's _beginthreadex
# needs, -Igui, and a resource compiler invocation with flags of its
# own. A gate demanding byte equality would fail on those and be
# deleted, so -bm is the one allowed difference and it is allowed BY
# NAME rather than by being filtered out of both sides.
#
# THE PER-FILE SPLIT IS PART OF IT. ENG_CPU_CASE gives ops.c and
# cpucheck.c the -6r/-D__MMX__=1 set and everything else the -4r floor,
# and the same case arm is reused by all four recipes - so both the
# ordinary and the HIGH set have to match, which is why the extraction
# takes the whole recipe rather than one variable. A GUI that compiled
# the MMX units at -4r would not link mmintrin.h at all.
#
# THE DOS TARGET IS OUT OF SCOPE ON PURPOSE. DOS_CPU_FLAGS is -3r
# against NT's -4r, a deliberate floor difference for a target with no
# Windows API, and there is no DOS GUI to pair it with.
#
# `make -Bn` rather than a real build: this asks what the recipe WOULD
# pass, which is the question, and it costs nothing. -B so the recipe
# prints even when the objects are up to date.
#
# Exit codes:
#   0  PASS - the two sets differ only by -bm
#   1  FAIL - they differ otherwise, with both printed
#   2  SKIP - make cannot produce a recipe for one of them
set -u
cd "$(dirname "$0")/.."

# -o and -z are Watcom's optimisation and model letters; -b is the
# target/runtime selector, which is where -bm lives; -[0-9]r is the CPU
# floor. Trailing quote and semicolon come off: the flags sit inside a
# shell `if` the recipe emits, so the last token of a branch carries the
# quoting with it.
flags_of() {
    make -Bn "$1" 2>/dev/null \
        | tr ' ' '\n' \
        | sed "s/[';]*$//" \
        | grep -E '^-[0-9]r$|^-o[a-z]+$|^-z[a-z0-9]+$|^-b[a-z]|^-we$|^-D' \
        | sort -u
}

CLI=$(flags_of watcom-cli)
GUI=$(flags_of watcom-gui)

if [ -z "$CLI" ] || [ -z "$GUI" ]; then
    echo "watcom_flag_parity_gate: SKIP - make produced no recipe for one target"
    exit 2
fi

# Positive control, the same one the gcc gate carries: the extraction
# must find something, and it must find the flags this gate is about. An
# empty-vs-empty comparison is equal and means nothing, and a regex that
# silently stopped matching -D would make every define look agreed.
N=$(printf '%s\n' "$CLI" | wc -l)
if [ "$N" -lt 8 ]; then
    echo "watcom_flag_parity_gate: FAIL - only $N flag(s) extracted from the CLI"
    echo "  recipe. The extraction is broken; equality below would be vacuous."
    exit 1
fi
for must in '-4r' '-6r' '-D__MMX__=1' '-otexan'; do
    case "
$CLI" in *"
$must"*) ;;
        *) echo "watcom_flag_parity_gate: FAIL - $must missing from the CLI set,"
           echo "  so the extraction is not seeing the recipe it is meant to."
           exit 1 ;;
    esac
done

# -bm is allowed on the GUI side and nowhere else. Dropped from the GUI
# set by name; if it ever appears on the CLI side that is a real
# difference and the diff below reports it.
GUI_CMP=$(printf '%s\n' "$GUI" | grep -v '^-bm$')

if [ "$CLI" = "$GUI_CMP" ]; then
    echo "watcom_flag_parity_gate: PASS ($N flags, identical apart from the"
    echo "  GUI's -bm; CPU floor, optimisation letters and every -D agree)"
    exit 0
fi

echo "watcom_flag_parity_gate: FAIL - the Watcom GUI and CLI engine flags differ"
echo "  only in the CLI:"
printf '%s\n' "$CLI" | grep -vxF -f <(printf '%s\n' "$GUI_CMP") | sed 's/^/    /'
echo "  only in the GUI:"
printf '%s\n' "$GUI_CMP" | grep -vxF -f <(printf '%s\n' "$CLI") | sed 's/^/    /'
echo "  CLI_FLAGS/GUI_FLAGS and CLI_FLAGS_HIGH/GUI_FLAGS_HIGH are written"
echo "  separately in the Makefile. Change one, change the other."
exit 1
