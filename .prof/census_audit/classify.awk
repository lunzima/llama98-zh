# Classifies measured float instructions into the census's own convention.
# Reads only.  Input: measure.sh's table.  -v tokens=<n> for per-token.
# Classes:
#   f32   mul/add/sub/div on float, and int<->float converts  (BILLABLE)
#   cmp   float compares and sign masks                       (census counts 0)
#   f64   anything in double, plus float<->double converts    (not f32 work)
function cls(mn) {
    if (mn ~ /^(comi|ucomi|cmp|min|max)(ss|ps)$/ ||
        mn ~ /^(and|andn|or|xor)ps$/) return "cmp"
    if (mn ~ /(sd|pd)$/ || mn ~ /^cvtsi2sd/ || mn ~ /^cvttsd2si/ ||
        mn ~ /^cvtsd2ss$/ || mn ~ /^cvtss2sd$/ || mn ~ /^cvtpd/ ||
        mn ~ /^cvtdq2pd$/) return "f64"
    if (mn ~ /^(mul|add|sub|div|sqrt|rcp|rsqrt)(ss|ps)$/) return "f32"
    if (mn ~ /^cvt/) return "f32"
    return "other"
}
# A packed instruction is 4 (ps) or 2 (pd) element operations: the census
# counts elements, not instructions, which is what makes an intrinsic
# kernel comparable to the C body it replaces.
function wt(mn) {
    if (mn ~ /ps$/ || mn ~ /^cvtps2dq$/ || mn ~ /^cvtdq2ps$/ ||
        mn ~ /^cvttps2dq$/) return 4
    if (mn ~ /pd$/ || mn ~ /^cvtpd/ || mn ~ /^cvtdq2pd$/) return 2
    return 1
}
BEGIN { FS = "\t"; if (tokens == 0) tokens = 1
    # Execution counts measured by the raw counters, for the source lines
    # whose gcov number is a sum over several basic blocks. Keyed
    # file<TAB>line; the count is the whole run, not per token.
    if (rawfix != "")
        while ((getline ln < rawfix) > 0) {
            split(ln, r, "\t")
            if (r[1] != "") ovr[r[1] "\t" r[2] + 0] = r[3] + 0
        }
}
{
    c = cls($5)
    key = $2 "\t" $3 + 0
    exec = (key in ovr) ? ovr[key] : $7
    w = wt($5) * exec * $6
    tot[c] += w
    tu[$1 "\t" c] += w
    fn[$1 "\t" $4 "\t" c] += w
}
END {
    printf "== totals (per token, %d tokens)\n", tokens
    for (k in tot) printf "  %-6s %14.1f\n", k, tot[k] / tokens
    printf "== per TU (f32 billable class)\n"
    for (k in tu) {
        split(k, a, "\t")
        if (a[2] != "f32") continue
        printf "  %-14s %14.1f\n", a[1], tu[k] / tokens
    }
    printf "== per function (f32 billable class)\n"
    for (k in fn) {
        split(k, a, "\t")
        if (a[3] != "f32" || fn[k] == 0) continue
        printf "FN %-14s %-28s %14.1f\n", a[1], a[2], fn[k] / tokens
    }
}
