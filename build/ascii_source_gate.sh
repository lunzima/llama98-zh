#!/bin/bash
# Every byte of every source file is ASCII.
#
# WHY, and it is not tidiness. The supported compiler set includes
# Visual C++ 4.0 on Windows NT and Open Watcom, both of which read
# source in the HOST's codepage rather than as UTF-8. On a DBCS host -
# Chinese 936, Japanese 932 - a byte in 0x81..0xFE is a lead byte and
# the lexer consumes the byte AFTER it as part of the same character.
# A UTF-8 sequence whose last byte lands in that range therefore eats
# whatever follows it, and what follows a character in a comment is
# sometimes the `*` of `*/`.
#
# The failure is silent in the worst way: the file still compiles on
# the developer's machine, because that machine is not DBCS.
#
# AND THE PROJECT BANS CHINESE COMMENTS OUTRIGHT, which is the other
# half. src/unicode.h carried `Unicode White_Space（Rust regex \s）`
# with FULL-WIDTH parentheses - a Chinese comment convention, in a
# header, unnoticed. It went unnoticed because the obvious search for
# it does not work: `grep -P '[\x{4e00}-\x{9fff}]'` returned zero
# matches on files that demonstrably contain CJK, since the pattern
# depends on the locale being a UTF-8 one and the file being decoded
# as such. A BYTE test has no such dependency, which is why this gate
# is written at the byte level and not the character level.
#
# WHAT WAS THERE when this was written, all of it now ASCII: the micro
# sign in "uop", superscript two in "O(n^2)", and the typographic set
# +- x * - sum in, plus the full-width parens above. None of it said
# anything its ASCII spelling does not.
#
# WHAT IS NOT COVERED: string literals that need non-ASCII bytes spell
# them with \x escapes, which are ASCII source - gui/localized_strings.c
# is the whole Chinese UI and passes this gate unchanged. That is the
# right shape anyway: an escaped literal survives any host codepage.
#
# Exit codes:
#   0  PASS
#   1  FAIL - a non-ASCII byte, or the detector cannot see one
set -u
cd "$(dirname "$0")/.."

FILES=$(ls src/*.c src/*.h common/*.c common/*.h gui/*.c gui/*.h 2>/dev/null)
[ -n "$FILES" ] || { echo "ascii_source_gate: FAIL - no sources found"; exit 1; }

BAD=0
N=0
for f in $FILES; do
    N=$((N + 1))
    hits=$(LC_ALL=C grep -c $'[\x80-\xff]' "$f" 2>/dev/null)
    [ "$hits" = "0" ] && continue
    echo "FAIL: $f has $hits line(s) with a non-ASCII byte:"
    LC_ALL=C grep -n $'[\x80-\xff]' "$f" | head -4 | sed 's/^/    /'
    BAD=1
done

# Control, because this is one grep and a grep that matches nothing
# reports success. A planted high byte must be seen, and a file of
# ordinary ASCII must not be.
CTL=$(printf 'plain ascii line\n' | LC_ALL=C grep -c $'[\x80-\xff]')
MUT=$(printf 'a \xc2\xb5op b\n'   | LC_ALL=C grep -c $'[\x80-\xff]')
if [ "$CTL" != "0" ] || [ "$MUT" != "1" ]; then
    echo "FAIL: the detector reports ascii=$CTL planted=$MUT; it cannot"
    echo "  tell the two apart, so its silence carries nothing"
    BAD=1
fi

if [ "$BAD" -eq 0 ]; then
    echo "ascii_source_gate: PASS ($N files, every byte ASCII;"
    echo "  a planted 0xC2 0xB5 reddens the detector)"
    exit 0
fi
echo "ascii_source_gate: FAIL"
echo "  Spell it in ASCII, or as a \\x escape if a string literal really"
echo "  needs the byte. A comment never does."
exit 1
