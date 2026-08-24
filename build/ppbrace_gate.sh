#!/bin/bash
# A brace pair or an assignment may not straddle an #if.
#
# build/ppbrace_census.awk is the scanner and its header defines both
# shapes and lists what it deliberately does not report. This holds the
# baseline, and it exists because neither shape produces a diagnostic:
# every configuration compiles, and the text simply stops meaning what
# its indentation says.
#
# THE BASELINE IS THREE PAIRS, all in forward.c, all the same shape:
#
#     #if (LZ_ATTN_FIXED & 2)
#     if (...) { kernel } else {          <- opens here
#     #endif
#     <scalar fallback>
#     #if (LZ_ATTN_FIXED & 2)
#     }                                   <- closes in a DIFFERENT #if
#     #endif
#
# forward.c:2324/2362 and 2369/2395 are the attention weighted sum, high
# plane and low; 3937/3979 is the MoE SwiGLU int16 exit. The two scoring
# sites a few lines above take the other shape - a dangling `else`
# before #endif, every brace balanced - which is why they are not here
# and why the fix for these three is known: wrap the fallback in braces
# so the `else` is the only token that crosses.
#
# They are baselined rather than fixed because the fix moves code inside
# forward_attn, and a reassociated float sum is a different number: it
# needs the fixed-tier cross-toolchain comparison run against it, not a
# reviewer's confidence. Recorded so the count cannot grow while that
# waits.
#
# Exit codes:
#   0  PASS - no file exceeds its baseline
#   1  FAIL - a new straddle, or the scanner stopped detecting one
set -u
cd "$(dirname "$0")/.."

SCAN=build/ppbrace_census.awk
SRC=$(ls src/*.c src/*.h gui/*.c gui/*.h common/*.c common/*.h 2>/dev/null)

# file:count, counting REPORT LINES - a pair is two, one per end, and
# both have to be named or a half-fix reads as progress.
BASELINE="forward.c:6"

FAIL=0
OUT=$(awk -f "$SCAN" $SRC 2>&1)
if printf '%s\n' "$OUT" | grep -q "awk:"; then
    echo "FAIL: the scanner itself reported an error:"
    printf '%s\n' "$OUT" | grep "awk:" | head -3 | sed 's/^/      /'
    FAIL=1
fi

for f in $(printf '%s\n' "$OUT" | grep -oE "^[^:]+:" | sed 's/:$//' | sort -u); do
    b=$(basename "$f")
    n=$(printf '%s\n' "$OUT" | grep -c "^$f:")
    exp=0
    for e in $BASELINE; do
        [ "${e%%:*}" = "$b" ] && exp=${e##*:}
    done
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $b has $n straddle(s), baseline $exp"
        printf '%s\n' "$OUT" | grep "^$f:" | head -6 | sed 's/^/      /'
        FAIL=1
    elif [ "$n" -lt "$exp" ]; then
        echo "note: $b is at $n, below its baseline of $exp - tighten BASELINE"
    fi
done

# THE MUTATION, both shapes, because a scanner that detects one and has
# gone blind to the other passes on a baseline that only exercises the
# first. The tree's live surface is six PAIR lines and zero STMT lines,
# so the STMT rule has nothing holding it up - it is the one that would
# rot unnoticed, and it is also the narrower rule of the two: it was
# widened down from a version that reported conditional parameters,
# argument lists, table entries and labels, 48 findings of noise.
PROBE=/tmp/ppbmut.$$.c
printf 'void f(int c) {\n#if C\n    if (c) { g(); } else {\n#endif\n    h();\n#if C\n    }\n#endif\n}\n' > "$PROBE"
MP=$(awk -f "$SCAN" "$PROBE" 2>&1 | grep -c "SPLIT PAIR")
printf 'int *g(void) {\n    int *p =\n#if C\n        &a;\n#else\n        &b;\n#endif\n    return p;\n}\n' > "$PROBE"
MS=$(awk -f "$SCAN" "$PROBE" 2>&1 | grep -c "SPLIT STMT")
# And a NEGATIVE mutation: the exempted idioms must stay silent, or the
# gate is green today only because nobody has run it on a header.
printf '#ifdef __cplusplus\nextern "C" {\n#endif\nvoid f(int a,\n#if C\n               int b,\n#endif\n               int c);\n#ifdef __cplusplus\n}\n#endif\n' > "$PROBE"
MN=$(awk -f "$SCAN" "$PROBE" 2>&1 | grep -c "SPLIT")
rm -f "$PROBE"

if [ "$MP" -eq 0 ] || [ "$MS" -eq 0 ]; then
    echo "FAIL: planted straddles report pair=$MP stmt=$MS; a zero means"
    echo "      the count above is the scanner being blind, not the tree"
    echo "      being clean"
    FAIL=1
fi
if [ "$MN" -ne 0 ]; then
    echo "FAIL: the exempted idioms report $MN finding(s) - extern \"C\" and"
    echo "      a conditional parameter are not straddles, and a scanner"
    echo "      that says they are gets switched off"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    N=$(printf '%s\n' "$OUT" | grep -c "SPLIT")
    echo "ppbrace_gate: PASS ($N straddle(s), all baselined; mutation"
    echo "  pair=$MP stmt=$MS, exempted idioms silent)"
    exit 0
fi
echo "ppbrace_gate: FAIL"
exit 1
