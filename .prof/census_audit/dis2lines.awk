# objdump -dl --inlines -> "file<TAB>line<TAB>function<TAB>mnemonic<TAB>n".
# Reads only.
#
# Attribution is to the OUTERMOST frame: an always_inline intrinsic body
# (xmmintrin.h etc.) is charged to the .c line that called it. Charging it
# to the header line is wrong twice over - gcov's count for that header
# line is the sum over every expansion in the TU, and several functions
# share it - which is how one mulps line got billed to two kernels at once.
/IMAGE_REL_AMD64_(REL32|ADDR64)|R_X86_64_(PLT32|PC32)/ {
    if (pend != "") {
        tgt = $NF
        sub(/[-+]0x[0-9a-f]+$/, "", tgt)
        h[pend "\tcall_" tgt]++
        pend = ""
    }
    next
}
/^[0-9a-f]+ <.*>:$/ {
    fn = $2
    gsub(/[<>:]/, "", fn)
    next
}
/^inlined by .*:[0-9]+ \(.*\)$/ {
    p = $0
    sub(/^inlined by /, "", p)
    sub(/ \(.*\)$/, "", p)
    n = p
    sub(/^.*:/, "", n)
    f = p
    sub(/:[0-9]+$/, "", f)
    gsub(/\\/, "/", f)
    sub(/^.*\//, "", f)
    outf = f
    outl = n + 0
    next
}
/^[^ \t].*:[0-9]+$/ {
    p = $0
    n = p
    sub(/^.*:/, "", n)
    f = p
    sub(/:[0-9]+$/, "", f)
    gsub(/\\/, "/", f)
    sub(/^.*\//, "", f)
    curf = f
    curl = n + 0
    outf = ""
    next
}
/^[ \t]*[0-9a-f]+:\t/ {
    line = $0
    sub(/^[^\t]*\t/, "", line)
    sub(/^[^\t]*\t/, "", line)
    sub(/^[ \t]+/, "", line)
    mn = line
    sub(/[ \t].*$/, "", mn)
    af = (outf != "") ? outf : curf
    al = (outf != "") ? outl : curl
    outf = ""
    if (mn == "") next
    # Call sites, so a leaf helper's cost can be charged to the site that
    # calls it: emitted as the pseudo-mnemonic "call_<target>".
    if (mn == "call") {
        if (line ~ /<[^>+]*>$/) {
            tgt = line
            sub(/^.*</, "", tgt)
            sub(/>$/, "", tgt)
            h[af "\t" al "\t" fn "\tcall_" tgt]++
        } else {
            # Cross-TU call in an unlinked object: the target is in the
            # relocation record objdump -r prints on the next line.
            pend = af "\t" al "\t" fn
        }
        next
    }
    if (mn ~ /^(add|sub|mul|div|sqrt|rsqrt|rcp|min|max|comi|ucomi|and|andn|or|xor|unpck|shuf|blend|round|cmp)(ss|sd|ps|pd)$/ ||
        mn ~ /^cvt/ || mn ~ /^f(add|sub|mul|div|ild|istp|sqrt|abs|chs|com|ucom)/) {
        h[af "\t" al "\t" fn "\t" mn]++
    }
    next
}
END {
    for (k in h) print k "\t" h[k]
}
