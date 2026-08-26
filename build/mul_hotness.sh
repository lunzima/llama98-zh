#!/bin/bash
# How often each multiply the shipping build emits actually executes.
#
# Same join as div_hotness.sh: sites from the shipping config, counts
# from a coverage run. A multiply in a function that runs once per model
# load is free; the same instruction in a per-element loop is not. Only
# gcov's per-line execution counts say which.
#
# THE SITES UNDERSTATE WATCOM. The site build here is gcc -m32
# -march=i486 -mtune=pentium, which is the div_hotness.sh convention for
# "what ships". But the REAL shipping compiler is Watcom (wcl386), and
# its instruction selection is clumsier: where gcc -mtune=pentium
# synthesizes a constant multiply as lea/shift (and so drops out of this
# site list), Watcom emits imul. So this table answers "which lines'
# multiplies run hot", NOT "which imuls ship" - the Watcom-clumsy imuls
# are the ones this table is missing, and they are found by diffing
# Watcom's wdis against gcc -mtune=pentium, not by counting here.
#
# WHAT A HIGH SCORE MEANS. A multiply worth a look. Some are runtime
# strides (row*width) that no instruction selection can avoid; the
# reading is "look here".
#
# THIS IS A CENSUS, NOT A GATE.
#
# Usage: bash build/mul_hotness.sh [MODEL] [TOP]
set -u
cd "$(dirname "$0")/.."

command -v objdump >/dev/null 2>&1 || { echo "mul_hotness: SKIP - no objdump"; exit 2; }

MODEL=${1:-/e/LLM/models/kmr20}
TOP=${2:-25}

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

. build/cov_run.sh
cov_run "$W" "$MODEL" 8 || exit 2

MULRE='^(imul|mul)'

ENG=$(make -s print-ENG) || { echo "mul_hotness: cannot read ENG"; exit 2; }
COM=$(make -s print-COM) || { echo "mul_hotness: cannot read COM"; exit 2; }
SRC=""
for f in $ENG; do SRC="$SRC src/$f.c"; done
for f in $COM; do SRC="$SRC common/$f.c"; done

: > "$W/sites.txt"
for f in $SRC; do
    [ -f "$f" ] || continue
    gcc -O2 -m32 -march=i486 -mtune=pentium -std=c99 -g -Isrc -Icommon \
        -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -c -o "$W/site.o" "$f" 2>/dev/null || {
        echo "mul_hotness: no shipping-config build for $f"; continue; }
    objdump -dl "$W/site.o" 2>/dev/null | awk -v re="$MULRE" '
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

echo "multiplies ranked by executions x multiplies-on-the-line"
echo
printf '%-14s %-4s %-22s %-7s %s\n' weighted mul file line function
grep '^H' "$W/joined.txt" | sort -t"$(printf '\t')" -k2,2rn | head -"$TOP" \
  | awk -F'\t' '{ printf "%-14d %-4d %-22s %-7d %s\n", $2, $3, $4, $5, $6 }'

echo
awk -F'\t' '$1 == "H" { w += $2; s += $3; n++ }
     END { printf "hot sites: %d on %d lines, %d multiply executions total\n",
                  s, n, w }' "$W/joined.txt"
awk -F'\t' '$1 == "C" { s += $3; n++ }
     END { printf "cold sites: %d on %d lines (reached zero times in this run)\n",
                  s + 0, n + 0 }' "$W/joined.txt"

grep '^U' "$W/joined.txt" > "$W/unattr.txt" || true
awk -F'\t' 'END { printf "unattributed sites: %d (gcov holds no code for the line objdump named)\n", NR + 0 }' "$W/unattr.txt"

echo
echo "counts are one generation: --prompt hi -n 8 --seed 1145141919"
echo "sites are gcc -m32 -mtune=pentium (UNDERSTATES Watcom's imuls)"
