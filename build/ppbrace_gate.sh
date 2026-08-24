#!/bin/bash
# A brace pair or an assignment may not straddle an #if.
#
# build/ppbrace_census.awk is the scanner and its header defines both
# shapes and lists what it deliberately does not report. This holds the
# baseline, and it exists because neither shape produces a diagnostic:
# every configuration compiles, and the text simply stops meaning what
# its indentation says.
#
# THE BASELINE IS EMPTY. It was three pairs, all in forward.c, all
#
#     #if (LZ_ATTN_FIXED & 2)
#     if (...) { kernel } else {          <- opens here
#     #endif
#     <scalar fallback>
#     #if (LZ_ATTN_FIXED & 2)
#     }                                   <- closes in a DIFFERENT #if
#     #endif
#
# two in the attention weighted sum, high plane and low, and one at the
# MoE SwiGLU int16 exit. All three now take the shape the two scoring
# sites beside them already had: the `else` is the last token of the
# guarded arm and the fallback is an unconditional block, so every brace
# opens and closes in one region and only the `else` crosses. A dangling
# `else` before #endif is not reported - see the scanner's header.
#
# The change is free, which was measured rather than argued: forward.c
# compiles to a BYTE-IDENTICAL object before and after under all of
# default, LZ_ATTN_FIXED 0/1/2, LZ_KV_2PLANE=1, LZ_SWIGLU_I16 0/1 and
# 2PLANE+ATTN_FIXED=0. That mattered because a reassociated float sum is
# a different number, and adding a block around a statement is exactly
# the kind of edit that can spill an x87 register. The control for the
# identity is that those same switches give each other DIFFERENT
# objects, so the regions are live and the identity is not dead code.
#
# Exit codes:
#   0  PASS - no straddle
#   1  FAIL - a new straddle, or the scanner stopped detecting one
set -u
cd "$(dirname "$0")/.."

SCAN=build/ppbrace_census.awk
SRC=$(ls src/*.c src/*.h gui/*.c gui/*.h common/*.c common/*.h 2>/dev/null)

# file:count, counting REPORT LINES - a pair is two, one per end, so a
# half-fix cannot read as progress.
BASELINE=""

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
# And a CONTINUED MACRO, which is the case this gate got wrong on its
# own author: `#define M do { \` is a directive and skipped, while the
# `} while (0)` on the continuation line is not, so counting only the
# second half made every file holding one read as a straddle.
printf '%s\n' 'void f(void) {' '#define M do { \' '    g(); \' '    } while (0)' '    M;' '}' > "$PROBE"
MC=$(awk -f "$SCAN" "$PROBE" 2>&1 | grep -c "SPLIT")
MN=$((MN + MC))
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
    echo "ppbrace_gate: PASS ($N straddle(s); planted mutation pair=$MP"
    echo "  stmt=$MS, exempted idioms silent)"
    exit 0
fi
echo "ppbrace_gate: FAIL"
exit 1
