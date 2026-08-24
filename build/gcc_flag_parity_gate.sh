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
# WHY FLAGS SETTLE THE RUNTIME TIER TOO, which is not obvious and was
# checked rather than assumed. The GUI never calls lz_kernel_select -
# there is no such call anywhere under gui/ - so g_kernel reaches its
# value through the lazy `if (!g_kernel) lz_kernel_select(AUTO)` guards
# scattered through the dispatch files. The hazard would be an operator
# entry point without one: it would dispatch with g_kernel still 0,
# which reads as LZ_KERNEL_REF, and the window would silently run the
# scalar reference while the command line ran SSE2.
# Counted: ops_matmul.c 6, ops_sched.c 3, ops.c, ops_gdn.c and
# ops_quant.c 1 each. ops_norm.c has none - and does not reference
# g_kernel at all, so it does not dispatch on the tier and needs none.
# Every file that dispatches on it is guarded.
# With identical flags and the same machine, the detected tier is
# therefore the same value on both. What the GUI still cannot do is SAY
# which tier it is on; the CLI prints it in its banner and the About
# dialog has no equivalent line.
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

# The release level, checked here because this is the one place that
# already knows what the recipes pass.
#
# -O3 is not a release level for this project. It inflated the shipped
# binaries 19% (llama98) and 14% (the gui) over -O2 while producing
# byte-equal --dump-all-logits, which is what makes it a size decision
# with nothing on the other side of the ledger. The target family's
# instruction caches are small enough that the size is itself a speed
# question.
#
# Equality above would not catch this: both targets could agree on -O3
# and the gate would say PASS. Two separate properties, two checks.
for f in $CLI $GUI; do
    case "$f" in
        -O3)
            echo "gcc_flag_parity_gate: FAIL - a release recipe passes -O3"
            echo "  Release builds are -O2. -O3 costs ~19% binary size here"
            echo "  and buys nothing numerically; see X86_64_CFLAGS."
            exit 1
            ;;
    esac
done

# ---- AND THE SAME OPERATORS, not just the same flags ------------------
#
# The requirement is that the GUI and the CLI use the same optimization
# configuration AND the same operator configuration. Equal flags settle
# the first half only: two binaries can compile every kernel identically
# and still select different ones at run time.
#
# THE INVARIANT IS THAT gui/ SELECTS NOTHING. The CLI parses --kernel,
# --fixed, --pair and --prefetch and calls the four selectors; the GUI
# exposes none of them and calls none, so it runs the engine defaults -
# which are the CLI's defaults. That is the whole guarantee, and it
# holds only as long as nothing under gui/ starts calling a selector.
#
# ASKED OF THE LINKER, not of the text. Two textual attempts failed
# first and both failed silently-looking: a plain grep counts the
# comment in gui/aboutdlg.c that describes what lz_kernel_tier() does
# internally, and running the source through -E to drop comments then
# counts the four PROTOTYPES that ops.h inlines into every translation
# unit - four hits in every gui file, none of them a call.
#
# An undefined symbol in the object file is a call and nothing else. It
# cannot be a comment, a declaration, or a macro that expanded to
# nothing.
SELECTORS='lz_kernel_select|lz_float_ref_select|lz_pair_select|lz_prefetch_select'
SEL_HITS=""
SW=$(mktemp -d)
for g in gui/*.c; do
    [ -f "$g" ] || continue
    gcc $GUI -Isrc -Icommon -Igui -c -o "$SW/probe.o" "$g" 2>/dev/null || continue
    n=$(nm -u "$SW/probe.o" 2>/dev/null | grep -cE "\b($SELECTORS)\b" || true)
    [ -n "$n" ] || n=0
    [ "$n" -gt 0 ] && SEL_HITS="$SEL_HITS $(basename $g)($n)"
done

# Positive control pointed the way this needs: the same search over the
# CLI MUST find calls. If the compile or nm stopped working, gui/ would
# read clean for that reason and this half would pass by measuring
# nothing.
CTL=0
if gcc $GUI -Isrc -Icommon -c -o "$SW/ctl.o" src/cli_main.c 2>/dev/null; then
    CTL=$(nm -u "$SW/ctl.o" 2>/dev/null | grep -cE "\b($SELECTORS)\b" || true)
fi
[ -n "$CTL" ] || CTL=0
rm -rf "$SW"
if [ "$CTL" -eq 0 ]; then
    echo "gcc_flag_parity_gate: FAIL - the selector search finds no call in"
    echo "  src/cli_main.c, which makes them; the search is broken and a"
    echo "  clean gui/ result would mean nothing"
    exit 1
fi

if [ -n "$SEL_HITS" ]; then
    echo "gcc_flag_parity_gate: FAIL - gui/ selects operators for itself:"
    for h in $SEL_HITS; do echo "    $h"; done
    echo "  The GUI has no operator options and must run the engine"
    echo "  defaults, so that it and the CLI cannot diverge at run time."
    exit 1
fi

if [ "$CLI" = "$GUI" ]; then
    echo "gcc_flag_parity_gate: PASS ($N flags, identical, release level"
    echo "  $(printf '%s\n' $CLI | grep -m1 -E '^-O');"
    echo "  gui/ calls 0 operator selectors against the CLI's $CTL)"
    printf '  %s\n' $CLI
    exit 0
fi

echo "gcc_flag_parity_gate: FAIL - the GUI and CLI compile the engine differently"
echo "  CLI only:"
comm -23 <(printf '%s\n' "$CLI") <(printf '%s\n' "$GUI") | sed 's/^/    /'
echo "  GUI only:"
comm -13 <(printf '%s\n' "$CLI") <(printf '%s\n' "$GUI") | sed 's/^/    /'
exit 1
