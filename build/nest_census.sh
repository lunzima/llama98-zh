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
WHOLE=0
if [ -z "$FILES" ]; then
    FILES=$(ls src/*.c common/*.c gui/*.c 2>/dev/null)
    WHOLE=1
fi
NFILE=$(printf '%s\n' $FILES | wc -l)

command -v awk >/dev/null 2>&1 || { echo "nest_census: SKIP - no awk"; exit 2; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
awk -f build/nest_census.awk $FILES > "$W/all.txt" || {
    echo "nest_census: the scan failed"; exit 2; }

# Sync check. The scan finding a handful of functions where there are
# many means the brace machine came adrift - most likely on a construct
# the comment stripper mishandles - and a short ranking would read as
# "few functions are deep" rather than as "the instrument stopped
# counting".
#
# THE THRESHOLD IS PER FILE, not a tree-wide constant. The first version
# refused any run returning under 100, which is right for the default
# set of ~50 files and wrong the moment anyone passes a file list: two
# operator files legitimately hold 87 functions between them and the
# check called that a loss of sync. A count that is correct being
# reported as instrument failure is the same defect as the reverse, and
# it arrived within an hour of the gate being written.
#
# THE EXPLICIT-LIST CHECK IS WEAK and saying so is the point: two files
# per function is a floor a badly desynced scan could still clear. The
# whole-tree run keeps the 100 and is the one to trust; a file list
# trades that for being able to ask about a subset at all.
N=$(wc -l < "$W/all.txt")
MIN=$((NFILE * 2))
[ "$WHOLE" = 1 ] && [ "$MIN" -lt 100 ] && MIN=100
if [ "$N" -lt "$MIN" ]; then
    echo "nest_census: only $N function(s) across $NFILE file(s), under the"
    echo "  $MIN this scan expects. The brace tracking has lost sync and the"
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
