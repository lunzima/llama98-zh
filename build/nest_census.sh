#!/bin/bash
# Which functions are actually tangled, ranked by nesting depth.
#
# WHY NOT LINE COUNT. A long function is not a spaghetti function: this
# tree has several that are long because they carry a table or a
# derivation in comments, and shortening those loses the reasoning
# without changing the control flow anyone has to hold in their head.
# What makes a function hard to change is how deep the arms go and how
# many of them there are - a body at depth 7 has 7 conditions live at
# once at its innermost statement, and every edit has to be correct
# under all of them.
#
# So depth is the sort key and length is a column beside it. Both are
# printed because the pair is the reading: deep and short is a dispatch
# ladder, deep and long is the thing to break up, and long and shallow
# is usually fine.
#
# WHAT DEPTH MEANS HERE: brace nesting inside the function body, with
# the body itself counting as 1. A `for` containing an `if` containing a
# `switch` arm with a block reads 4. Braceless single-statement arms
# count as nothing, which UNDERSTATES a ladder written without braces -
# build/ppbrace_gate.sh has its own opinion about those, and this census
# deliberately does not double as that gate.
#
# THIS IS A CENSUS, NOT A GATE. It prints; it does not judge. A depth-6
# format dispatch may be exactly right.
#
# Usage: bash build/nest_census.sh [TOP] [files...]
set -u
cd "$(dirname "$0")/.."

TOP=${1:-20}
shift 2>/dev/null || true
FILES=${*:-}
if [ -z "$FILES" ]; then
    FILES=$(ls src/*.c common/*.c gui/*.c 2>/dev/null)
fi

command -v awk >/dev/null 2>&1 || { echo "nest_census: SKIP - no awk"; exit 2; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
awk -f build/nest_census.awk $FILES > "$W/all.txt" || {
    echo "nest_census: the scan failed"; exit 2; }

N=$(wc -l < "$W/all.txt")
if [ "$N" -lt 100 ]; then
    # Positive control. The scan finding a handful of functions in a
    # tree this size means the brace machine lost sync - most likely on
    # a construct the comment stripper mishandles - and a short ranking
    # would read as "few functions are deep" rather than as "the
    # instrument stopped counting".
    echo "nest_census: only $N function(s) found across the tree."
    echo "  That is too few; the brace tracking has lost sync and the"
    echo "  ranking below would be of whatever it happened to keep."
    exit 2
fi

echo "functions by nesting depth (depth is the sort key, length is context)"
echo
printf '%-6s %-7s %-20s %s\n' depth lines file function
sort -t"$(printf '\t')" -k1,1rn -k2,2rn "$W/all.txt" | head -"$TOP" \
  | awk -F'\t' '{ printf "%-6d %-7d %-20s %s\n", $1, $2, $3, $4 }'

echo
awk -F'\t' '{ n++; s += $1; if ($1 > m) m = $1; if ($1 >= 6) d6++ }
     END { printf "%d functions scanned, deepest %d, mean %.2f, %d at depth 6 or more\n",
                  n, m, s / n, d6 + 0 }' "$W/all.txt"
