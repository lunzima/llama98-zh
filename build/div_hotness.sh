#!/bin/bash
# How often each divide the shipping build emits actually executes.
#
# div_census.sh answers "where are the divides" and says in its own
# header that where each one sits is the whole reading - a divide in a
# function that runs once per model load is free, the same instruction
# in a per-element loop is not. It cannot answer that; it counts
# instructions in a .o and has no idea which ones run.
#
# The obvious way to close the gap is to work out from the disassembly
# which divides sit inside a loop. That was tried and it does not work:
# a backward branch spanning a divide tells you a jump goes backwards
# over it, not that the divide is in the loop body, and ranking by the
# span of that branch put a once-per-model-load allocator at the top.
# There is no loop membership in a linear instruction listing without
# reconstructing the CFG and its dominators.
#
# So do not infer it - measure it. gcov reports per-line EXECUTION
# COUNTS from a real generation, and a divide's source line is in the
# debug info. Join the two: sites from the shipping build, counts from
# the run.
#
# THE TWO SIDES ARE DIFFERENT BUILDS and that is deliberate. Sites come
# from 32-bit i486 tuned for an in-order core, which is what ships and
# therefore which divides survive. Counts come from an x86-64 coverage
# build, because that is what runs here. A line's execution count is a
# property of the program, not of the instruction set, so it carries;
# WHICH lines hold a divide does not carry, which is why the site list
# is taken from the shipping side and never from the coverage build.
#
# WHAT A HIGH SCORE MEANS. A divide worth a look, nothing more. Some are
# unavoidable - an overflow guard that only a divide computes correctly
# is not debt at any frequency. The reading is "look here".
#
# COLD AND UNATTRIBUTED ARE DIFFERENT ANSWERS and the table keeps them
# apart. gcov writes "#####" for a line it instrumented and nobody
# reached - that is a count, and the count is zero. It writes "-" for a
# line it holds no code for at all, which means the attribution failed,
# usually because the optimiser moved the expression. Folding the two
# together reports "96 sites unaccounted for" when 96 sites are simply
# cold, and hides however many really were unaccounted for inside that
# figure.
#
# THIS IS A CENSUS, NOT A GATE. It prints; it does not judge.
#
# Usage: bash build/div_hotness.sh [MODEL] [TOP]
set -u
cd "$(dirname "$0")/.."

command -v objdump >/dev/null 2>&1 || { echo "div_hotness: SKIP - no objdump"; exit 2; }

MODEL=${1:-/e/LLM/models/kmr20}
TOP=${2:-25}

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

. build/cov_run.sh
cov_run "$W" "$MODEL" 8 || exit 2

DIVRE='^(idiv|div|fdiv|fdivr|fdivp|fdivrp|divss|divsd|divps|divpd)'

ENG=$(make -s print-ENG) || { echo "div_hotness: cannot read ENG"; exit 2; }
COM=$(make -s print-COM) || { echo "div_hotness: cannot read COM"; exit 2; }
SRC=""
for f in $ENG; do SRC="$SRC src/$f.c"; done
for f in $COM; do SRC="$SRC common/$f.c"; done

: > "$W/sites.txt"
for f in $SRC; do
    [ -f "$f" ] || continue
    gcc -O2 -m32 -march=i486 -mtune=pentium -std=c99 -g -Isrc -Icommon \
        -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -c -o "$W/site.o" "$f" 2>/dev/null || {
        echo "div_hotness: no shipping-config build for $f"; continue; }
    # -l emits the source location every time it changes, so the file
    # and line have to be carried forward across the instructions that
    # follow. Drive letters make the path contain a colon; splitting on
    # the LAST colon is what makes "E:\...\foo.c:1848" parse.
    objdump -dl "$W/site.o" 2>/dev/null | awk -v re="$DIVRE" '
        /^[0-9a-f]+ </ { fn = $0; sub(/^[0-9a-f]+ </, "", fn)
                         sub(/>:.*/, "", fn); next }
        /^[ \t]/ {
            nf = split($0, p, "\t")
            if (nf >= 3 && file != "") {
                split(p[3], m, " ")
                if (m[1] ~ re) print file "\t" ln "\t" fn
            }
            next
        }
        /:[0-9]+$/ {
            loc = $0; gsub(/\\/, "/", loc)
            n = split(loc, q, ":"); ln = q[n] + 0
            file = loc; sub(/:[0-9]+$/, "", file); sub(/.*\//, "", file)
        }
    ' >> "$W/sites.txt"
done

# Per-line execution counts. gcov prints "<count>: <line>: <text>" with
# the count as "-" for a line holding no code and "#####" for one that
# holds code nobody reached.
awk -F: '
    FNR == 1 { g = FILENAME; sub(/.*[\/]/, "", g); sub(/\.gcov$/, "", g) }
    /^ *[0-9#-]+: *[0-9]+:/ {
        c = $1; gsub(/ /, "", c); l = $2 + 0
        if (l > 0) print g "\t" l "\t" c
    }
' "$W"/*.gcov > "$W/counts.txt"

awk -F'\t' '
    NR == FNR { cnt[$1 "\t" $2] = $3; next }
    {
        k = $1 "\t" $2
        sites[k "\t" $3]++
        seen[k "\t" $3] = 1
    }
    END {
        for (s in sites) {
            split(s, a, "\t")
            k = a[1] "\t" a[2]
            c = (k in cnt) ? cnt[k] : "-"
            n = sites[s]
            if (c == "-")           printf "U\t0\t%d\t%s\n", n, s
            else if (c == "#####")  printf "C\t0\t%d\t%s\n", n, s
            else                    printf "H\t%d\t%d\t%s\n", c * n, n, s
        }
    }
' "$W/counts.txt" "$W/sites.txt" > "$W/joined.txt"

echo "divides ranked by executions x divides-on-the-line"
echo
printf '%-14s %-4s %-22s %-7s %s\n' weighted div file line function
grep '^H' "$W/joined.txt" | sort -t"$(printf '\t')" -k2,2rn | head -"$TOP" \
  | awk -F'\t' '{ printf "%-14d %-4d %-22s %-7d %s\n", $2, $3, $4, $5, $6 }'

echo
awk -F'\t' '$1 == "H" { w += $2; s += $3; n++ }
     END { printf "hot sites: %d on %d lines, %d divide executions total\n",
                  s, n, w }' "$W/joined.txt"
awk -F'\t' '$1 == "C" { s += $3; n++ }
     END { printf "cold sites: %d on %d lines (reached zero times in this run)\n",
                  s + 0, n + 0 }' "$W/joined.txt"

grep '^U' "$W/joined.txt" > "$W/unattr.txt" || true
awk -F'\t' 'END { printf "unattributed sites: %d (gcov holds no code for the line objdump named)\n", NR + 0 }' "$W/unattr.txt"
awk -F'\t' '{ printf "  %-4d %-22s %-7d %s\n", $3, $4, $5, $6 }' "$W/unattr.txt" \
  | sort | head -20

echo
echo "counts are one generation: --prompt hi -n 8 --seed 1145141919"
echo "sites are the 32-bit i486 shipping configuration; counts are x86-64"
