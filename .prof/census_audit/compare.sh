#!/bin/bash
# Prints the census-vs-measured table for one arm, ranked by measured
# work. Reads only.
#   $1 census stdout of the run, $2 measured.tsv for the same run,
#   $3 rawfix file, $4 token count
set -u
cd "$(dirname "$0")"
CEN=$1; MEAS=$2; RAWFIX=$3; TOK=$4

# linemap.tsv keys 36 rows to src/forward.c LINE NUMBERS, so any edit to
# that file silently moves work between sites - not a crash, a wrong
# table. It has already happened once: a commit between the measurement
# and the publication shifted forward.c and moved 6,240 ops/token out of
# the NOSITE rows into SIGMOID.
#
# Refuse instead of reporting. Regenerating the map is the fix; this
# only makes the staleness loud. Overridable for the "I know, I am
# regenerating" case.
#
# Hash the WORKING FILE, not HEAD. The measurement is taken from a
# binary, and the binary is built from what is on disk. An earlier
# version of this check asked git for HEAD's blob, which passes in a
# repo whose working tree carries someone else's uncommitted forward.c -
# exactly the case it exists for. `git hash-object` on the path is also
# what works in a `git archive` checkout, which has no .git at all.
#
# Refuse only on a POSITIVE mismatch: a check that also fires when it
# cannot tell would be red everywhere and therefore carry nothing.
LINEMAP_FORWARD_BLOB=ae54f175c17dbfd38a76cec81295a04d3f9c538f
if [ "${CENSUS_AUDIT_SKIP_BLOBCHECK:-0}" != "1" ]; then
    cur=$(git hash-object --path src/forward.c ../../src/forward.c 2>/dev/null) \
        || cur=
    if [ -z "$cur" ]; then
        echo "compare.sh: WARNING - cannot identify src/forward.c;" >&2
        echo "  linemap.tsv's 36 line keys are unverified for this tree." >&2
    elif [ "$cur" != "$LINEMAP_FORWARD_BLOB" ]; then
        echo "compare.sh: REFUSING - linemap.tsv was generated against" >&2
        echo "  src/forward.c blob $LINEMAP_FORWARD_BLOB" >&2
        echo "  this tree has       $cur" >&2
        echo "  Its 36 forward.c line keys are stale; the table would be" >&2
        echo "  wrong in a way that looks right. Regenerate linemap.tsv," >&2
        echo "  or set CENSUS_AUDIT_SKIP_BLOBCHECK=1 if you just did." >&2
        exit 2
    fi
fi

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
