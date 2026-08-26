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

MODEL=${1:-/e/LLM/models/kmr20}
TOP=${2:-25}

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

# build/cov_run.sh holds the build, the run and the gcov invocation, all
# three of which have edges worth meeting once rather than twice. It is
# shared with div_hotness.sh so both censuses report on the SAME run -
# a branch count and a divide count from different generations are not
# comparable and would look as if they were.
. build/cov_run.sh
cov_run "$W" "$MODEL" 8 || exit 2

# Each branch prints as two "branch N taken M" lines under the line that
# holds it. Score = executions x minority share, so a hot balanced
# branch outranks both a hot predictable one and a balanced cold one.
# A never-executed arm is gcov's spelling of "taken 0", and it must
# still consume the arm slot: skipping it (matching only "taken") pairs
# this branch's taken arm with the NEXT branch's, and two stable dispatch
# guards then read as a 50/50 coin flip that never ran. gcov 16 prints
# "branch N never executed (fallthrough)".
awk '
    /^[ ]*[0-9]+:[ ]*[0-9]+:/ { split($0, p, ":"); line = p[2] + 0; next }
    /^branch  *[0-9]+ (taken|never)/ {
        n = ($3 == "taken") ? $4 + 0 : 0
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
