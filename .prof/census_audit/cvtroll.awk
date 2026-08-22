# Conversion-only view of the measured float table: which int<->float
# converts execute, at which census site, produced by which construct.
# Reads only.
#   -v tokens= -v sitemap= -v linemap= [-v rawfix=] [-v basis=hi|lo]
#
# Attribution is rollup.awk's, unchanged: linemap overrides sitemap per
# source line, and a LEAF helper's measured per-call cost is charged to
# whoever called it. What is added is that every bucket is split into
#   I2F  int -> float32   (cvtsi2ss*, cvtdq2ps)
#   F2I  float32 -> int   (cvtt?ss2si, cvtt?ps2dq)
#   F32  all billable f32 work, converts included
# so a site's conversion share is a filter on the same collection rather
# than a second instrument.
#
# int<->double is excluded with the rest of f64. That is not an
# assumption: every executed cvtsi2sd/cvttsd2si in this tree is either
# an LZ_FCX census-counter argument, which exists only in the counting
# build, or one-time table setup.
#
# basis=lo uses joinall.awk's minimum-block execution count instead of
# the maximum, so two runs bracket the lines gcov reports UNEQUAL.
#
# -v magic=<file> adds the conversions this tree performs WITHOUT a cvt
# instruction - q8_round's magic add and lz_exp's floor round a float to
# an integer with an addss and a mask, so an instruction count scores
# them 0 while a machine with no FPU pays a full soft-float add. The
# file declares them rather than the classifier guessing:
#   LEAF<TAB>function<TAB>converts per call
#   LINE<TAB>file<TAB>line<TAB>converts per execution
# They land in a separate MF2I column; F2I stays what the hardware does.

function cvtcls(mn) {
    if (mn ~ /^cvtsi2ss/ || mn ~ /^cvtdq2ps$/) return "I2F"
    if (mn ~ /^cvtt?ss2si/ || mn ~ /^cvtt?ps2dq$/) return "F2I"
    return ""
}
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
    if (mn ~ /ps$/ || mn ~ /^cvtps2dq$/ || mn ~ /^cvtdq2ps$/ ||
        mn ~ /^cvttps2dq$/) return 4
    if (mn ~ /pd$/) return 2
    return 1
}
BEGIN {
    FS = "\t"
    if (tokens == 0) tokens = 1
    if (basis == "") basis = "hi"
    nm = split("F32 I2F F2I MF2I", M, " ")
    if (magic != "")
        while ((getline ln < magic) > 0) {
            if (ln ~ /^#/ || ln == "") continue
            split(ln, r, "\t")
            if (r[1] == "LEAF") mleaf[r[2]] = r[3] + 0
            else if (r[1] == "LINE") mline[r[2] "\t" r[3] + 0] = r[4] + 0
        }
    if (rawfix != "")
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
    if ($7 < 0) next
    key = $2 "\t" $3 + 0
    p = (key in ovr) ? ovr[key] * $6 : ((basis == "lo") ? $10 : $8)
    isleaf = (($4 in site) && site[$4] == "LEAF")

    if (!isleaf && (key in mline) && !(key in mseen) && $6 > 0) {
        mseen[key] = 1
        ms = (key in lsite) ? lsite[key] : \
             (($4 in site) ? site[$4] : "UNMAPPED:" $4)
        mv = (p / $6) * mline[key]
        tot[ms SUBSEP "MF2I"] += mv
        sites[ms] = 1
        mcn = "magic " $2 ":" $3 " (" $4 ")"
        cons[ms SUBSEP mcn SUBSEP "MF2I"] += mv
        conskey[ms SUBSEP mcn] = 1
    }

    if ($5 ~ /^call_/) {
        callee = substr($5, 6)
        if (!((callee in site) && site[callee] == "LEAF")) next
        calls[$4 SUBSEP callee] += p
        ncall[callee] += p
        csite[$4 SUBSEP callee] = (key in lsite) ? lsite[key] : ""
        next
    }
    if (cls($5) != "f32") next
    w = wt($5) * p
    cc = cvtcls($5)
    if (isleaf) {
        own[$4 SUBSEP "F32"] += w
        if (cc != "") own[$4 SUBSEP cc] += w
        leaves[$4] = 1
        next
    }
    s = (key in lsite) ? lsite[key] : \
        (($4 in site) ? site[$4] : "UNMAPPED:" $4)
    tot[s SUBSEP "F32"] += w
    sites[s] = 1
    if (cc != "") {
        tot[s SUBSEP cc] += w
        cn = "own " $2 ":" $3 " (" $4 ")"
        cons[s SUBSEP cn SUBSEP cc] += w
        conskey[s SUBSEP cn] = 1
    }
}
END {
    for (f in leaves)
        if (ncall[f] > 0)
            for (i = 1; i <= nm; i++)
                percall[f SUBSEP M[i]] = own[f SUBSEP M[i]] / ncall[f]
    for (f in mleaf)
        if (ncall[f] > 0) { leaves[f] = 1; percall[f SUBSEP "MF2I"] = mleaf[f] }

    for (k in calls) {
        split(k, a, SUBSEP)
        if ((a[1] in site) && site[a[1]] == "LEAF") continue
        s = (csite[k] != "") ? csite[k] : \
            ((a[1] in site) ? site[a[1]] : "UNMAPPED:" a[1])
        sites[s] = 1
        for (i = 1; i <= nm; i++) {
            v = calls[k] * percall[a[2] SUBSEP M[i]]
            if (v == 0) continue
            tot[s SUBSEP M[i]] += v
            if (M[i] != "F32") {
                cons[s SUBSEP a[2] SUBSEP M[i]] += v
                conskey[s SUBSEP a[2]] = 1
            }
        }
    }

    printf "== SITE  converts/token (basis=%s, %d tokens)\n", basis, tokens
    printf "%5s %-17s %10s %10s %10s %10s %12s %7s\n",
           "", "site", "i2f", "f2i", "cvt", "magic", "f32_total", "cvt/f32"
    gi = gf = gm = gt = 0
    for (s in sites) {
        i2f = tot[s SUBSEP "I2F"]; f2i = tot[s SUBSEP "F2I"]
        mg = tot[s SUBSEP "MF2I"]; f32 = tot[s SUBSEP "F32"]
        c = i2f + f2i
        gi += i2f; gf += f2i; gm += mg; gt += f32
        if (f32 == 0 && c + mg == 0) continue
        printf "SITE %-17s %10.0f %10.0f %10.0f %10.0f %12.0f %6.1f%%\n", s,
               i2f / tokens, f2i / tokens, c / tokens, mg / tokens,
               f32 / tokens, (f32 > 0) ? 100 * c / f32 : 0
    }
    printf "SITE %-17s %10.0f %10.0f %10.0f %10.0f %12.0f %6.1f%%\n", "TOTAL",
           gi / tokens, gf / tokens, (gi + gf) / tokens, gm / tokens,
           gt / tokens, (gt > 0) ? 100 * (gi + gf) / gt : 0
    printf "SEMANTIC TOTAL (cvt + magic) %.0f/token\n", (gi + gf + gm) / tokens

    printf "\n== CONSTRUCT  converts/token by site and producer\n"
    for (k in conskey) {
        split(k, a, SUBSEP)
        i2f = cons[k SUBSEP "I2F"]; f2i = cons[k SUBSEP "F2I"]
        mg = cons[k SUBSEP "MF2I"]
        if (i2f + f2i + mg == 0) continue
        printf "CONS %-17s %-34s %10.0f %10.0f %10.0f %10.0f\n", a[1], a[2],
               i2f / tokens, f2i / tokens, mg / tokens,
               (i2f + f2i + mg) / tokens
    }

    printf "\n== LEAF  measured per-call cost\n"
    for (f in leaves)
        printf "LEAF %-20s calls/tok %10.1f  i2f %6.3f f2i %6.3f magic %6.3f f32 %7.3f\n",
               f, ncall[f] / tokens, percall[f SUBSEP "I2F"],
               percall[f SUBSEP "F2I"], percall[f SUBSEP "MF2I"],
               percall[f SUBSEP "F32"]
}
