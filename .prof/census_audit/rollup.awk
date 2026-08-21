# Rolls the measured per-line f32 counts up into census sites.
# Reads only.  -v tokens= -v rawfix= -v sitemap= -v linemap=
#
# Two-level attribution:
#   own[fn]        f32 work whose instructions are in fn itself
#   calls[fn,leaf] how many times fn calls an unbilled leaf helper
# A leaf's per-call cost is measured, not assumed: own cost / call count,
# with the leaf's own leaf-calls folded in first (q8_round inside
# sigmoid_q15's neighbours, lz_i32f inside nothing). Then every function
# that is not a leaf lands on the site its billing point names, and the
# leaves land on whoever called them.
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
    while ((getline ln < linemap) > 0) {
        if (ln ~ /^#/ || ln == "") continue
        split(ln, r, "\t")
        lsite[r[1] "\t" r[2] + 0] = r[3]
    }
}
{
    key = $2 "\t" $3 + 0
    exec = (key in ovr) ? ovr[key] : $7
    if (exec < 0) next
    if ($5 ~ /^call_/) {
        callee = substr($5, 6)
        if (site[callee] == "LEAF") {
            calls[$4 SUBSEP callee] += exec * $6
            ncall[callee] += exec * $6
            csite[$4 SUBSEP callee] = (key in lsite) ? lsite[key] : ""
        }
        next
    }
    if (cls($5) != "f32") next
    own[$4] += wt($5) * exec * $6
    if (key in lsite) linew[$4 SUBSEP lsite[key]] += wt($5) * exec * $6
}
END {
    # Leaf per-call cost, own work only (no leaf calls nested two deep in
    # this tree: sigmoid_q15 and q8_round call nothing billable).
    # `f in site` first, every time: referencing site[f] would create the
    # key with an empty value and turn every unmapped function into a
    # nameless row instead of an UNMAPPED one.
    for (f in own) if ((f in site) && site[f] == "LEAF" && ncall[f] > 0)
        percall[f] = own[f] / ncall[f]

    for (f in own) {
        if ((f in site) && site[f] == "LEAF") continue
        s = (f in site) ? site[f] : "UNMAPPED:" f
        rest = own[f]
        for (k in linew) {
            split(k, a, SUBSEP)
            if (a[1] != f) continue
            tot[a[2]] += linew[k]
            rest -= linew[k]
        }
        tot[s] += rest
    }
    for (k in calls) {
        split(k, a, SUBSEP)
        if ((a[1] in site) && site[a[1]] == "LEAF") continue
        s = (csite[k] != "") ? csite[k] : ((a[1] in site) ? site[a[1]] : "UNMAPPED:" a[1])
        tot[s] += calls[k] * percall[a[2]]
        leafto[s "\t" a[2]] += calls[k] * percall[a[2]]
    }
    printf "== measured f32 per token, rolled up to census sites\n"
    for (s in tot) printf "%-22s %12.1f\n", s, tot[s] / tokens
    printf "== leaf helper cost charged to each site\n"
    for (k in leafto) {
        split(k, a, "\t")
        printf "   %-22s %-18s %12.1f\n", a[1], a[2], leafto[k] / tokens
    }
    printf "== measured leaf per-call cost (f32 ops)\n"
    for (f in percall) printf "   %-18s %8.3f  calls/token %10.1f\n",
                              f, percall[f], ncall[f] / tokens
}
