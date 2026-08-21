#!/bin/bash
# Prints the census-vs-measured table for one arm, ranked by measured
# work. Reads only.
#   $1 census stdout of the run, $2 measured.tsv for the same run,
#   $3 rawfix file, $4 token count
set -u
cd "$(dirname "$0")"
CEN=$1; MEAS=$2; RAWFIX=$3; TOK=$4

sed -n '/f32 operations per token/,/^  total/p' "$CEN" |
awk '{
    line = $0
    sub(/^  /, "", line)
    if (line ~ /^ /) next
    n = split(line, f, /  +/)
    if (n < 2) next
    name = f[1]; val = f[2]
    if (val ~ /^[0-9]+$/) print name "\t" val
}' > /tmp/_census_sites.tsv

awk -v tokens="$TOK" -v rawfix="$RAWFIX" -v sitemap=sitemap.tsv \
    -v linemap=linemap.tsv -f rollup.awk "$MEAS" |
awk '/^== leaf/{exit} NR>1 {print}' > /tmp/_measured_sites.tsv

awk -F'\t' '
FILENAME ~ /_census_sites/ {
    key = toupper($1); gsub(/ /, "_", key)
    if (key == "DEQUANT_EPILOGUE") key = "EPI"
    if (key == "ATTENTION_SCORING") key = "ATTN_SCORE"
    if (key == "ATTENTION_WSUM") key = "ATTN_WSUM"
    if (key == "CONV1D") key = "CONV"
    if (key == "NORMS") key = "NORM"
    if (key == "RECURRENCE_WRITE") key = "RECUR"
    if (key == "QUANTIZE") key = "QUANT"
    if (key == "TOTAL") key = "TOTAL"
    cen[key] = $2 + 0
    next
}
{
    split($0, a, / +/)
    s = a[1]; v = a[2] + 0
    if (s == "") next
    meas[s] += v
    if (s ~ /^NOSITE/) nosite += v
    mtot += v
}
END {
    printf "%-16s %14s %14s %12s %8s\n", "site", "census", "measured", "delta", "ratio"
    for (s in meas) {
        c = (s in cen) ? cen[s] : 0
        printf "%-16s %14.0f %14.0f %12.0f %8s\n", s, c, meas[s], meas[s] - c,
               (c > 0) ? sprintf("%.2fx", meas[s] / c) : "-"
    }
    printf "%-16s %14.0f %14.0f %12.0f %8.2fx\n", "TOTAL", cen["TOTAL"], mtot,
           mtot - cen["TOTAL"], mtot / cen["TOTAL"]
}
' /tmp/_census_sites.tsv /tmp/_measured_sites.tsv | sort -k3 -nr
