#!/bin/bash
# The GUI and the CLI compile the engine with the same flags.
#
# Both link the same $(ENG_SRCS). When they differ, the shipped window
# runs a different engine from the shipped command line while both claim
# to be the same build - and nothing says so, because the numbers agree:
# -O2 and -O3 produce byte-identical --dump-all-logits here, measured on
# two models before the flags were unified. A gate that only compared
# output would have passed throughout.
#
# WHAT IS COMPARED: the optimisation level, the -march, and every -D the
# recipe passes. Not the whole command line - the GUI legitimately adds
# -mwindows, -Igui, -Icommon and its resource object, and a gate that
# demanded byte equality would fail on those and be deleted.
#
# `make -n` rather than a real build: this asks what the recipe WOULD
# pass, which is the question, and it costs nothing.
#
# Exit codes:
#   0  PASS - the two sets are equal
#   1  FAIL - they differ, with both printed
#   2  SKIP - make cannot produce a recipe for one of them
set -u
cd "$(dirname "$0")/.."

flags_of() {
    # -B so the recipe prints even when the target is up to date.
    make -Bn "$1" 2>/dev/null \
        | tr ' ' '\n' \
        | grep -E '^-O[0-9s]$|^-march=|^-D' \
        | sort -u
}

CLI=$(flags_of llama98)
GUI=$(flags_of kunkun98-gui)

if [ -z "$CLI" ] || [ -z "$GUI" ]; then
    echo "gcc_flag_parity_gate: SKIP - make produced no recipe for one target"
    exit 2
fi

# Positive control: the extraction must actually find something. An
# empty-vs-empty comparison is equal and means nothing, which is the
# shape this project has hit three times.
N=$(printf '%s\n' "$CLI" | wc -l)
if [ "$N" -lt 3 ]; then
    echo "gcc_flag_parity_gate: FAIL - only $N flag(s) extracted from the CLI"
    echo "  recipe; the pattern has stopped matching and equality here"
    echo "  would mean nothing"
    exit 1
fi

if [ "$CLI" = "$GUI" ]; then
    echo "gcc_flag_parity_gate: PASS ($N flags, identical)"
    printf '  %s\n' $CLI
    exit 0
fi

echo "gcc_flag_parity_gate: FAIL - the GUI and CLI compile the engine differently"
echo "  CLI only:"
comm -23 <(printf '%s\n' "$CLI") <(printf '%s\n' "$GUI") | sed 's/^/    /'
echo "  GUI only:"
comm -13 <(printf '%s\n' "$CLI") <(printf '%s\n' "$GUI") | sed 's/^/    /'
exit 1
