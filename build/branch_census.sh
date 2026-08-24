#!/bin/bash
# Which branches actually cost anything on a core with a bad predictor.
#
# THE TARGET FAMILY IS IN-ORDER WITH A WEAK PREDICTOR - a Pentium has a
# 256-entry BTB and no recovery worth the name, so a mispredict is the
# pipeline. That makes two properties matter together and neither alone:
#
#   EXECUTIONS   a branch in a per-element loop is a different object
#                from the same source line in a setup path
#   BALANCE      a branch taken 99.9% of the time is predicted and
#                costs nothing; one near 50/50 is a coin flip every
#                time it runs
#
# STATIC DENSITY ANSWERS NEITHER. Counting `if` in the source finds the
# guards a reader notices, not the ones the machine runs, and it cannot
# see which way they go. gcov -b -c reports both, so that is what this
# uses: taken counts per arm, per branch, from a real generation.
#
# WALL CLOCK ANSWERS NEITHER EITHER, which is why this is not a timing
# script. A mispredict on this host is a few cycles behind an
# out-of-order window; the number that transfers to the target is how
# often the branch runs and how unpredictable it is, and both are
# properties of the program rather than of the machine measuring it.
#
# WHAT A HIGH SCORE MEANS. Candidate, not defect. A 50/50 branch in the
# hot path may be structural - a format dispatch that genuinely has two
# equally likely cases - and removing it may cost more than it saves.
# The reading is "look here", and the look is manual.
#
# THE FIRST LOOK, so the next reader starts below it rather than at the
# top. The ranking's head on kmr20 at -n 8 is epi_shr_half's
# `if (rs <= 0) return s;` (src/ops.c) - 4,495,576 executions with the
# minority arm at 43%, which is close to a coin flip several million
# times a generation and the largest single mispredict source in the
# engine. It STAYS: the comment above it already says why, and the
# reason is correctness rather than taste. At rs <= 0 with s ==
# LLONG_MIN the magnitude is 2^63 and converting it back is
# implementation-defined; every rs >= 1 keeps the result under 2^63.
# A branchless select would evaluate the arm the guard exists to skip.
#
# So the head of this table is load-bearing, which is worth knowing
# before anyone spends a round on it. What follows it - the per-group
# accumulate loop in lz_epi_mac_i16 and the ones in ops_matmul.c, the
# UTF-8 decode in unicode.c - are LOOP-BACK edges, whose minority arm
# is the single exit and which no predictor gets wrong twice.
#
# NAMED, NOT NUMBERED, because the numbers rotted: this paragraph cited
# a line in ops.c for that accumulate loop, and the loop had moved
# twenty-odd lines down by the time anyone looked. A line number in a
# comment is a fact about a file that keeps being edited, and the next
# reader checks the line rather than the function - so no line numbers
# here, including for the correction. The head is still epi_shr_half at
# the same 4,495,576 executions.
#
# Usage: bash build/branch_census.sh [MODEL] [TOP]
set -u
cd "$(dirname "$0")/.."

command -v gcov >/dev/null 2>&1 || { echo "branch_census: SKIP - no gcov"; exit 2; }
command -v gcc  >/dev/null 2>&1 || { echo "branch_census: SKIP - no gcc"; exit 2; }

MODEL=${1:-/e/LLM/models/kmr20}
TOP=${2:-25}
[ -d "$MODEL" ] || { echo "branch_census: SKIP - no model: $MODEL"; exit 2; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

# The engine only. The front end's option parsing runs once and would
# fill the table with branches nobody executes twice.
ENG=$(make -s print-ENG) || { echo "branch_census: cannot read ENG"; exit 2; }
COM=$(make -s print-COM) || { echo "branch_census: cannot read COM"; exit 2; }
SRC=""
for f in $ENG; do SRC="$SRC src/$f.c"; done
for f in $COM; do SRC="$SRC common/$f.c"; done

# -O2 and the shipping defines: a coverage build at -O0 would count
# branches the optimiser deletes. --coverage keeps its own arcs even at
# -O2, and the counts are of the OPTIMISED control flow, which is the
# one that ships.
gcc -O2 -march=x86-64 -std=c99 -Isrc -Icommon -ffp-contract=off \
    -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 --coverage \
    -o "$W/cov.exe" $SRC src/cli_attr.c src/cli_main.c \
    build/native/ops_mmx.o build/native/ops_mmx_sse.o build/native/ops_sse2.o \
    -lm >"$W/build.log" 2>&1 || {
    echo "branch_census: SKIP - coverage build failed"; tail -3 "$W/build.log"; exit 2; }

# One short generation. Fixed seed so two runs of this script compare.
( cd "$W" && ./cov.exe "$MODEL" --prompt "hi" -n 8 --seed 1145141919 ) \
    >"$W/run.log" 2>&1 || { echo "branch_census: SKIP - the run failed"
                            tail -3 "$W/run.log"; exit 2; }

# gcov writes its .gcov into the CURRENT directory, not into -o, and it
# resolves source paths relative to the current directory too. So it has
# to run from the repo root with -o pointing at the .gcda, and the
# output lands here and is swept up afterwards. The first version ran it
# from the root but globbed $W for the results and printed an empty
# table - a census that prints nothing reads exactly like a census that
# found nothing.
# THE .gcda FILES, not the source names. gcov derives the notes-file
# name from whatever it is handed, so `gcov src/ops_gdn.c` looks for
# ops_gdn.gcno - and gcc named it cov-ops_gdn.gcno, after the output
# binary, because many sources compiled into one. It then reported
# "cannot open notes file" and "No executable lines" into the log this
# script was not reading, and the table came out empty.
gcov -b -c -o "$W" "$W"/*.gcda >"$W/gcov.log" 2>&1
mv -f ./*.gcov "$W/" 2>/dev/null

# Each branch prints as two "branch N taken M" lines under the line that
# holds it. Score = executions x minority share, so a hot balanced
# branch outranks both a hot predictable one and a balanced cold one.
awk '
    /^[ ]*[0-9]+:[ ]*[0-9]+:/ { split($0, p, ":"); line = p[2] + 0; next }
    /^branch  *[0-9]+ taken/ {
        n = $4 + 0
        if (arm == 0) { a = n; arm = 1; next }
        b = n; arm = 0
        tot = a + b
        if (tot < 1000) next
        lo = (a < b) ? a : b
        printf "%d\t%d\t%d\t%s\n", lo * 2, tot, line, FILENAME
    }
' "$W"/*.gcov 2>/dev/null \
  | sort -rn | head -"$TOP" \
  | awk -F'\t' 'BEGIN {
        printf "%-12s %-12s %-7s %s\n", "score", "executions", "line", "file" }
      { f = $4; sub(/.*[\/\\]/, "", f); sub(/\.gcov$/, "", f)
        printf "%-12d %-12d %-7d %s\n", $1, $2, $3, f }'
echo
echo "score = executions x minority-arm share x 2 (1.0 = a perfect coin flip)"
echo "branches under 1000 executions are dropped: a setup guard is free"
