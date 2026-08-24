"""Rewrite non-ASCII bytes inside C string literals as \\xNN escapes.

WHY. Visual C++ 4.0 rejects raw UTF-8 in a string literal with
"error C2001: newline in constant": its lexer treats the lead bytes as
DBCS, consumes the following byte as a trail byte, and eventually eats
the closing quote. chat.c already carries the fix - every Chinese byte
hex-escaped, with a comment saying the source file stays ASCII - and
err.c and gui/localized_strings.c never got it.

THE ESCAPE-RUN TRAP, which is why this is a program and not a regex.
C's \\x consumes as many hex digits as follow it, so "\\xe4a" is one
character with value 0xe4a, not 0xe4 followed by 'a'. Any escape
immediately followed by [0-9A-Fa-f] must be terminated - this closes the
string and reopens it, which the translator concatenates back into one
literal. Missing that is how a "mechanical" conversion silently changes
the bytes.

ONLY INSIDE STRING LITERALS. Characters in comments and in identifiers
are left alone: the compiler's complaint was about literals, and
rewriting a comment would be a different change wearing the same name.

Usage:
    python build/utf8_escape.py --check FILE...   report, change nothing
    python build/utf8_escape.py --write FILE...   rewrite in place

--check is the round trip: it re-reads what it would write, extracts the
literals from both, and reports any whose bytes differ. A conversion
that cannot prove it preserved the bytes is not a conversion.
"""
import sys

HEX = set(b"0123456789abcdefABCDEF")


def convert_line(line):
    """Return the line with non-ASCII bytes inside literals escaped."""
    out = bytearray()
    i = 0
    n = len(line)
    in_str = False
    pending_escape = False   # last emitted token was \xNN
    while i < n:
        c = line[i]
        if not in_str:
            pending_escape = False
            if c == 0x22:                      # opening quote
                # A quote inside a // or /* comment is not a literal, but
                # this file's tables have no such case and guessing at
                # comment state would be the bigger risk. Callers run
                # --check.
                in_str = True
            out.append(c)
            i += 1
            continue
        # inside a literal
        if c == 0x5C:                          # backslash: copy the escape
            out.append(c)
            i += 1
            if i < n:
                out.append(line[i])
                i += 1
            pending_escape = False
            continue
        if c == 0x22:                          # closing quote
            in_str = False
            pending_escape = False
            out.append(c)
            i += 1
            continue
        if c >= 0x80:
            out.extend(b"\\x%02x" % c)
            pending_escape = True
            i += 1
            continue
        if pending_escape and c in HEX:
            # Terminate the run by escaping this character too: the
            # backslash ends the previous \\xNN, and one literal stays
            # one literal. Closing and reopening the string with `" "`
            # also works for the compiler - it concatenates - but it
            # changes the literal COUNT, which is what the round-trip
            # check compares, and the check caught exactly that.
            out.extend(b"\\x%02x" % c)
            i += 1
            continue
        pending_escape = False
        out.append(c)
        i += 1
    return bytes(out)


def literals(data):
    """Every string literal's BYTES, escapes resolved, in order."""
    res = []
    i = 0
    n = len(data)
    while i < n:
        if data[i] != 0x22:
            i += 1
            continue
        i += 1
        cur = bytearray()
        while i < n and data[i] != 0x22:
            if data[i] == 0x5C and i + 1 < n:
                nx = data[i + 1]
                if nx in b"x":
                    i += 2
                    h = bytearray()
                    while i < n and data[i] in HEX:
                        h.append(data[i])
                        i += 1
                    cur.append(int(h, 16) & 0xFF)
                    continue
                cur.append(0x5C)
                cur.append(nx)
                i += 2
                continue
            cur.append(data[i])
            i += 1
        i += 1
        res.append(bytes(cur))
    return res


def main(argv):
    if len(argv) < 3 or argv[1] not in ("--check", "--write"):
        print(__doc__)
        return 2
    mode = argv[1]
    bad = 0
    for path in argv[2:]:
        src = open(path, "rb").read()
        dst = b"".join(convert_line(l) for l in src.splitlines(keepends=True))
        a, b = literals(src), literals(dst)
        if a != b:
            print("%s: ROUND TRIP FAILED" % path)
            for x, y in zip(a, b):
                if x != y:
                    print("   was %r" % x[:40])
                    print("   now %r" % y[:40])
                    break
            if len(a) != len(b):
                print("   literal count %d -> %d" % (len(a), len(b)))
            bad = 1
            continue
        nonascii = sum(1 for ch in src if ch >= 0x80)
        left = sum(1 for ch in dst if ch >= 0x80)
        print("%s: %d non-ASCII byte(s) -> %d, %d literal(s) unchanged"
              % (path, nonascii, left, len(a)))
        if mode == "--write" and dst != src:
            open(path, "wb").write(dst)
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv))
