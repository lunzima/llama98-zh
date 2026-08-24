# Braces and statements that straddle a preprocessor boundary.
#
# Run by build/ppbrace_gate.sh, which holds the baseline and the
# mutation. This file only measures.
#
# Two shapes, both of which make the text mean something other than what
# its indentation says:
#
#   SPLIT PAIR   an #if region whose braces do not balance, so a `{`
#                opens under one switch setting and its `}` closes under
#                another - or two arms that leave different depths.
#                Neither arm can be read alone, and an edit near one end
#                silently changes the other.
#
#   SPLIT STMT   an assignment cut in half by the #if: `plg =` on one
#                side and `pl_row + gg * 32;` on the other. It reads as a
#                conditional expression and is not one, and a
#                declaration it carries lands after a statement, which
#                C89 does not have.
#
# WHAT IS NOT REPORTED, because a first version reported all of it and
# was 48 findings of noise against 3 real ones:
#
#   extern "C" {     under #ifdef __cplusplus. Universal, and the brace
#                    imbalance is the whole idiom.
#   conditional      a parameter list or argument list that gains
#   parameters       entries when a switch is on. The alternative is two
#   and arguments    whole signatures, which is worse.
#   table entries    a dispatch table with one #if per ISA slot. That is
#                    the clearest form available.
#   a label          `cleanup:` before an #if ends in a colon and
#                    continues nothing.
#
#   A DANGLING ELSE before #endif. `if (fast) {...} else` followed by
#   #endif and then a statement makes the whole fallback conditional,
#   every brace balances, and the alternative duplicates the fallback in
#   both arms.
#
# Braces inside comments and string literals do not count; the scanner
# tracks block comments across lines and skips quoted text.

function strip(s,   out, i, c, n) {
    out = ""
    n = length(s)
    for (i = 1; i <= n; i++) {
        c = substr(s, i, 1)
        if (incomment) {
            if (c == "*" && substr(s, i + 1, 1) == "/") { incomment = 0; i++ }
            continue
        }
        if (instr) {
            if (c == "\\") { i++; continue }
            if (c == instr) instr = 0
            continue
        }
        if (c == "/" && substr(s, i + 1, 1) == "*") { incomment = 1; i++; continue }
        if (c == "/" && substr(s, i + 1, 1) == "/") break
        if (c == "\"" || c == "'") { instr = c; continue }
        out = out c
    }
    return out
}

FNR == 1 { depth = 0; sp = 0; incomment = 0; instr = 0; prev = ""; cont = 0 }

{
    line = $0
    # LINE CONTINUATION, and it is not a detail: a multi-line macro puts
    # its opening brace on the #define line - a directive, skipped - and
    # its closing `} while (0)` on a continuation line, which is not.
    # Counting only the second half reported the whole file as one
    # straddle. The #pragma aux bodies never showed it because their
    # continuations are quoted asm and strip() takes the braces out.
    if (cont) {
        strip(line)
        cont = (line ~ /\\[ \t]*$/)
        next
    }
    if (line ~ /^[ \t]*#/ && line ~ /\\[ \t]*$/) {
        strip(line)
        cont = 1
        prev = ""
        next
    }
    if (line ~ /^[ \t]*#[ \t]*(if|ifdef|ifndef)/) {
        # A directive can still close a block comment opened above it, so
        # the comment state advances through every line, code or not.
        strip(line)
        sp++
        start[sp] = depth
        armdelta[sp] = -9999               # no arm has closed yet
        startline[sp] = FNR
        # extern "C" is the exemption, taken from the CONDITION rather
        # than from the brace, so a real imbalance under some other
        # __cplusplus guard would still be reported.
        cpp[sp] = (line ~ /__cplusplus/)
        if (prev ~ /[^=<>!+*\/%&|^-]=[ \t]*$/)
            printf "%s:%d: SPLIT STMT - an assignment continues into the #if\n", \
                   FILENAME, FNR
        prev = ""
        next
    }
    if (line ~ /^[ \t]*#[ \t]*(else|elif)/) {
        strip(line)
        if (sp > 0) {
            if (armdelta[sp] == -9999) armdelta[sp] = depth - start[sp]
            else if (depth - start[sp] != armdelta[sp] && !cpp[sp])
                printf "%s:%d: SPLIT PAIR - arms of the #if at line %d leave different brace depths\n", \
                       FILENAME, FNR, startline[sp]
            depth = start[sp]
        }
        prev = ""
        next
    }
    if (line ~ /^[ \t]*#[ \t]*endif/) {
        strip(line)
        if (sp > 0) {
            d = depth - start[sp]
            if (armdelta[sp] == -9999) {
                if (d != 0 && !cpp[sp])
                    printf "%s:%d: SPLIT PAIR - the #if at line %d leaves brace depth %+d\n", \
                           FILENAME, FNR, startline[sp], d
            } else if (d != armdelta[sp] && !cpp[sp]) {
                printf "%s:%d: SPLIT PAIR - arms of the #if at line %d leave different brace depths\n", \
                       FILENAME, FNR, startline[sp]
            }
            depth = start[sp] + (armdelta[sp] == -9999 ? 0 : armdelta[sp])
            sp--
        }
        prev = ""
        next
    }
    if (line ~ /^[ \t]*#/) { strip(line); prev = ""; next }

    stripped = strip(line)
    n = split(stripped, ch, "")
    for (i = 1; i <= n; i++) {
        if (ch[i] == "{") depth++
        else if (ch[i] == "}") depth--
    }
    if (stripped ~ /[^ \t]/) prev = stripped
}
