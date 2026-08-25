# Maximum brace nesting per function, with the body's line count beside
# it. Driven by build/nest_census.sh; see that file's header for why the
# sort key is depth and not length.
#
# Comments and literals are stripped by a character state machine rather
# than by a regex, because this tree's comments are long prose that
# contains braces, quotes and apostrophes, and a line-based strip gets
# every one of those wrong. The machine carries /* */ across lines.
#
# Preprocessor lines are skipped whole. That is only sound because
# build/ppbrace_gate.sh already refuses a #if block whose braces do not
# balance - without it, one arm of a conditional could open a brace this
# counter never closes and every function after it would read as nested
# one level deeper.

function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }

function emit() {
    if (fname != "") printf "%d\t%d\t%s\t%s\n", maxd, NR - fstart + 1, fbase, fname
    fname = ""
}

FNR == 1 { fbase = FILENAME; sub(/.*[\/]/, "", fbase)
           depth = 0; incomment = 0; instr = 0; fname = ""; pending = "" }

{
    line = $0
    # A preprocessor line contributes no braces. Checked before the
    # comment machine so a trailing /* on a #define does not open one.
    if (!incomment && line ~ /^[ \t]*#/) next

    out = ""
    i = 1
    n = length(line)
    while (i <= n) {
        c = substr(line, i, 1)
        d = substr(line, i, 2)
        if (incomment) {
            if (d == "*/") { incomment = 0; i += 2 } else { i++ }
            continue
        }
        if (instr) {
            if (c == "\\") { i += 2; continue }
            if (c == q) instr = 0
            i++
            continue
        }
        if (d == "/*") { incomment = 1; i += 2; continue }
        if (d == "//") break
        if (c == "\"" || c == "'") { instr = 1; q = c; i++; continue }
        out = out c
        i++
    }

    n = length(out)
    for (i = 1; i <= n; i++) {
        c = substr(out, i, 1)
        if (c == "{") {
            if (depth == 0) {
                # A brace opening at file scope is a function body only
                # if what precedes it ends in a parameter list. Aggregate
                # initialisers and struct bodies end in an identifier or
                # an =, and are what this rejects.
                p = trim(pending)
                if (p ~ /\)$/) {
                    fname = p
                    sub(/[ \t]*\([^(]*\)$/, "", fname)
                    sub(/.*[^A-Za-z0-9_]/, "", fname)
                    if (fname == "") fname = p
                    fstart = NR
                    maxd = 1
                } else {
                    fname = ""
                }
                pending = ""
            }
            depth++
            if (fname != "" && depth > maxd) maxd = depth
        } else if (c == "}") {
            depth--
            if (depth <= 0) { depth = 0; emit(); pending = "" }
        } else if (depth == 0) {
            if (c == ";") pending = ""
            else pending = pending c
        }
    }
}

END { emit() }
