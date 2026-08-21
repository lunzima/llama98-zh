# Drill one census site's measured f32 work down to functions and source
# lines. Same inputs as rollup.awk, one extra: -v want=<SITE>.
# Reads only.
#   awk -v tokens=603 -v rawfix=<f> -v sitemap=sitemap.tsv -v want=RECUR \
#       -f site_detail.awk measured.tsv
#
# Input columns (measure.sh's output):
#   1 tu  2 file  3 line  4 func  5 mnemonic|call_<callee>
#   6 static  7 exec  8 prod  9 flag  10 lo  11 src
#
# Leaf helpers are charged at the same measured per-call cost rollup.awk
# uses, so the per-line rows sum to that site's rollup total.
#
# NOTE: the sitemap decides which functions land on a site, and for
# ops_gdn.c it puts all eight recurrence functions on RECUR while only
# gdn_p2_fixed_rows:1679 bills there. A site total is therefore not the
# same question as "is the formula at that billing point right" - this
# script exists to tell the two apart.
function cls(mn) {
    if (mn ~ /^(comi|ucomi|cmp|min|max)(ss|ps)$/ ||
        mn ~ /^(and|andn|or|xor)ps$/) return "cmp"
    if (mn ~ /(sd|pd)$/ || mn ~ /^cvtsi2sd/ || mn ~ /^cvttsd2si/ ||
        mn ~ /^cvtsd2ss$/ || mn ~ /^cvtss2sd$/) return "f64"
    if (mn ~ /^(mul|add|sub|div|sqrt)(ss|ps)$/) return "f32"
    if (mn ~ /^cvt/) return "f32"
    return "other"
}
function wt(mn) {
    if (mn ~ /ps$/ || mn ~ /^cvtps2dq$/ || mn ~ /^cvtdq2ps$/) return 4
    if (mn ~ /pd$/) return 2
    return 1
}
function kind(mn) {
    if (mn ~ /^mul/) return "mul"
    if (mn ~ /^add/) return "add"
    if (mn ~ /^sub/) return "sub"
    if (mn ~ /^div/) return "div"
    if (mn ~ /^sqrt/) return "sqrt"
    if (mn ~ /^cvt/) return "cvt"
    return mn
}
BEGIN {
    FS = "\t"
    if (tokens == 0) tokens = 1
    while ((getline ln < rawfix) > 0) {
        split(ln, r, "\t")
        if (r[1] != "" && r[1] !~ /^#/) ovr[r[1] "\t" r[2] + 0] = r[3] + 0
    }
    while ((getline ln < sitemap) > 0) {
        if (ln ~ /^#/ || ln == "") continue
        split(ln, r, "\t")
        site[r[1]] = r[2]
    }
}
{
    rows[NR] = $0
    key = $2 "\t" $3 + 0
    exec = (key in ovr) ? ovr[key] : $7
    ex[NR] = exec
    if (exec < 0) next
    if ($5 ~ /^call_/) {
        callee = substr($5, 6)
        if (site[callee] == "LEAF") ncall[callee] += exec * $6
        next
    }
    if (cls($5) != "f32") next
    own[$4] += wt($5) * exec * $6
}
END {
    for (f in own) if ((f in site) && site[f] == "LEAF" && ncall[f] > 0)
        percall[f] = own[f] / ncall[f]

    for (i = 1; i <= NR; i++) {
        $0 = rows[i]
        exec = ex[i]
        if (exec < 0) continue
        f = $4
        if (!(f in site) || site[f] != want) continue
        lk = f "\t" $2 ":" $3
        if ($5 ~ /^call_/) {
            callee = substr($5, 6)
            if (site[callee] != "LEAF") continue
            v = exec * $6 * percall[callee]
            if (v == 0) continue
            line[lk] += v; byfn[f] += v; det[lk "\t" callee] += v
            src[lk] = $11; nn[lk "\t" callee] += exec * $6
            continue
        }
        if (cls($5) != "f32") continue
        v = wt($5) * exec * $6
        line[lk] += v; byfn[f] += v; det[lk "\t" kind($5)] += v
        src[lk] = $11; nn[lk "\t" kind($5)] += exec * $6
    }
    printf "== %s: measured f32 per token, by function\n", want
    tot = 0
    for (f in byfn) { printf "%-26s %12.1f\n", f, byfn[f] / tokens; tot += byfn[f] }
    printf "%-26s %12.1f\n", "TOTAL", tot / tokens
    printf "\n== by line (ops/token; n = executions of that operation)\n"
    n = 0
    for (k in line) { ord[++n] = k; val[k] = line[k] }
    for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++)
        if (val[ord[j]] > val[ord[i]]) { t = ord[i]; ord[i] = ord[j]; ord[j] = t }
    for (i = 1; i <= n; i++) {
        k = ord[i]
        if (line[k] == 0) continue
        split(k, a, "\t")
        printf "%12.1f  %-24s %-22s %s\n", line[k] / tokens, a[2], a[1],
               substr(src[k], 1, 78)
        for (d in det) {
            split(d, b, "\t")
            if (b[1] "\t" b[2] != k) continue
            printf "%12s    %-10s %10.1f   n=%.1f\n", "", b[3],
                   det[d] / tokens, nn[d] / tokens
        }
    }
}
