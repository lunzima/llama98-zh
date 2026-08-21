# Per-TU join of float-instruction counts with gcov execution counts.
# Reads only.
#   -v insfile=<lineinsn table>   file<TAB>line<TAB>func<TAB>mn<TAB>n
#   -v tu=<name>
#   ARGV: <insfile> <gcov -a -t output for the same TU>
#
# gcov's printed per-line number is the SUM of the counts of every basic
# block that starts on that line, not the number of times the line ran:
# an -O0 line holding three inlined intrinsics prints 3x its true count.
# So the execution count used here comes from the `-a` block detail:
#   no block lines      -> the printed count (one containing block)
#   blocks, all equal   -> that common value (exact: every instruction on
#                          the line runs once per block execution)
#   blocks, unequal     -> UNEQUAL, flagged, resolved by other means
# Emits: tu file line func mn static exec product flag source
FILENAME == insfile {
    ins[$1 "\t" $2 "\t" $3 "\t" $4] = $5
    next
}
/^ *[^:]*: *0:Source:/ {
    f = $0
    sub(/^.*Source:/, "", f)
    gsub(/\\/, "/", f)
    sub(/^.*\//, "", f)
    curf = f
    next
}
/-block [0-9]+ *$/ {
    # "   48240: 1719-block 14" - one colon only, so it must be handled
    # before the two-colon source-line parse below.
    p1 = index($0, ":")
    if (p1 == 0) next
    cnt = substr($0, 1, p1 - 1)
    rest = substr($0, p1 + 1)
    ln = rest
    sub(/-block.*$/, "", ln)
    gsub(/[ \t]/, "", ln)
    gsub(/^[ \t]+/, "", cnt)
    gsub(/[ \t]+$/, "", cnt)
    sub(/\*$/, "", cnt)
    if (cnt == "#####" || cnt == "$$$$$") cnt = 0
    if (cnt == "-") next
    key = curf "\t" (ln + 0)
    nb[key]++
    if (nb[key] == 1) bmin[key] = bmax[key] = cnt + 0
    else {
        if (cnt + 0 < bmin[key]) bmin[key] = cnt + 0
        if (cnt + 0 > bmax[key]) bmax[key] = cnt + 0
    }
    next
}
{
    p1 = index($0, ":")
    if (p1 == 0) next
    cnt = substr($0, 1, p1 - 1)
    rest = substr($0, p1 + 1)
    p2 = index(rest, ":")
    if (p2 == 0) next
    ln = rest
    sub(/:.*$/, "", ln)
    gsub(/[ \t]/, "", ln)
    txt = substr(rest, p2 + 1)
    gsub(/^[ \t]+/, "", cnt)
    gsub(/[ \t]+$/, "", cnt)
    sub(/\*$/, "", cnt)
    if (cnt == "#####" || cnt == "$$$$$") cnt = 0
    key = curf "\t" (ln + 0)
    gc[key] = (cnt == "-") ? -1 : cnt + 0
    src[key] = txt
}
END {
    for (k in ins) {
        split(k, a, "\t")
        key = a[1] "\t" a[2]
        flag = "ok"
        if (!(key in gc)) { c = -2; flag = "nogcov" }
        else if (nb[key] > 0) {
            if (bmin[key] == bmax[key]) c = bmin[key]
            else { c = bmax[key]; flag = "UNEQUAL" }
        } else c = gc[key]
        prod = (c < 0) ? c : c * ins[k]
        lo = (nb[key] > 0) ? bmin[key] * ins[k] : prod
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
               tu, a[1], a[2], a[3], a[4], ins[k], c, prod, flag, lo, src[key]
    }
}
