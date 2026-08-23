# Which functions perform a billable float operation and bill nothing.
#
#   awk -f .prof/unbilled_scan.awk src/*.c
#
# build/f32count_billing_gate.sh keys on the SIGNATURE (a float return
# with an integer-accumulator parameter) and its own header says so: an
# unbilled float loop inside a void function is invisible to it, and it
# names the rule that would catch both shapes - key on the OPERATION.
# This is that scan, run to size the exception list before deciding
# whether the rule can become a gate.
#
# The two operations are this tree's own primitives, not arbitrary float
# arithmetic: lz_i32f is the int->float convert (a billable `cvt` by
# definition) and q8_round is the float->int one. A function containing
# either performs a conversion; if it names neither LZ_FCX nor LZ_FC,
# nothing bills that conversion inside it.
#
# Top-level function definitions only, found the same way the gate finds
# them - this tree's style puts a definition's return type at column 0
# and always indents the body, so a brace-depth walk from the opening
# brace is unambiguous.
/^[A-Za-z_][A-Za-z0-9_ \t*]*[ \t*]([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(/ { }
{ L[NR] = $0; F[NR] = FILENAME }
END {
    for (i = 1; i <= NR; i++) {
        line = L[i]
        # a definition line: starts at column 0, has "name(" in it, and
        # is not a declaration (no trailing ';' on the signature block)
        if (line !~ /^[A-Za-z_]/) continue
        if (line !~ /\(/) continue
        if (line ~ /^#/) continue
        # walk forward to the opening brace, at most 12 lines of params
        j = i; brace = -1; k = 0
        while (k < 12 && j <= NR) {
            if (index(L[j], "{") > 0) { brace = j; break }
            if (index(L[j], ";") > 0) break
            k++; j++
        }
        if (brace < 0) continue
        # name of the function: last identifier before the first '('
        head = line
        sub(/\(.*$/, "", head)
        nm = head
        sub(/^.*[^A-Za-z0-9_]/, "", nm)
        if (nm == "" || nm == "if" || nm == "for" || nm == "while" ||
            nm == "switch" || nm == "return") continue
        # body by brace depth
        d = 0; ops = ""; bill = 0
        for (j = brace; j <= NR; j++) {
            s = L[j]
            for (c = 1; c <= length(s); c++) {
                ch = substr(s, c, 1)
                if (ch == "{") d++
                else if (ch == "}") d--
            }
            if (index(s, "lz_i32f(") > 0) ops = ops "i32f "
            if (index(s, "q8_round(") > 0) ops = ops "q8round "
            if (index(s, "LZ_FCX(") > 0 || index(s, "LZ_FC(") > 0) bill = 1
            if (d <= 0) break
        }
        if (ops != "" && !bill)
            printf "%s:%d  %s  [%s]\n", F[i], i, nm, ops
        i = j
    }
}
