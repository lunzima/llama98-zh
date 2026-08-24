#!/bin/bash
# The translation limits a 1995 compiler is entitled to enforce.
#
# c89_floor_gate compiles with gcc -std=c89 -pedantic and its own header
# says what that does NOT prove: that Visual C++ 4.0 accepts the result.
# gcc's limits are effectively unbounded, so it is silent about every
# construct that is legal C89 yet larger than the standard's guaranteed
# minimum - and a compiler from 1995 is within its rights at exactly
# those minima. This checks the ones that are both reachable in this
# codebase and silent when they break.
#
#   31 significant characters in an EXTERNAL identifier (C89 5.2.4.1).
#            The defect is not length, it is COLLISION: two names that
#            agree in the first 31 become one symbol at link time, and
#            the linker reports a duplicate or, worse, silently binds
#            every call to whichever it saw first. Length alone is
#            harmless and is only reported as context.
#   509 characters in a STRING LITERAL after concatenation.
#            Over the limit a conforming-but-minimal compiler may
#            truncate. This tree has message tables and a system prompt
#            assembled from hex escapes, which is exactly the shape that
#            grows past it without anyone noticing.
#   127 nesting levels are far above anything here and are not checked;
#   257 case labels likewise. Limits are checked when the codebase can
#   plausibly reach them, not because the standard lists them.
#
# CHECKED ELSEWHERE, AND CLEAN, so nobody re-derives them: no
# `#pragma once`, no __func__ or __FUNCTION__, no anonymous union or
# struct members, no trailing comma in an enumerator, and no bitfield of
# a type other than int - C89 allows only int, and a 1995 compiler is
# entitled to enforce that. All five are zero tree-wide.
#
# PLAIN `char` SIGNEDNESS is implementation-defined and differs by
# platform, which is the kind of thing that produces a wrong answer
# rather than a diagnostic. Measured rather than reasoned about:
# building with -funsigned-char and running the same prompt and seed
# gives byte-identical output to the default build. Nothing here depends
# on it. To re-check:
#     make clean && make EXTRA_DEFS=-funsigned-char
# then diff a generation against one from the ordinary build.
#
# WHAT THIS STILL DOES NOT PROVE. Nothing in this tree can run VC++ 4.0.
# Green means the code does not exceed the limits that compiler is
# allowed to have; it does not mean it compiles there. Reading this as
# "the MIPS target builds" is the same mistake c89_floor_gate warns
# about, one layer further out.
#
# Exit codes:
#   0  PASS
#   1  FAIL - a collision, or a literal over the limit
set -u
cd "$(dirname "$0")/.."

FAIL=0

# SCOPE: what the MIPS target compiles. The SIMD translation units are
# excluded because their long literals are #pragma aux bodies - Watcom
# inline assembly, x86 opcodes, inside build guards that a MIPS compiler
# never reaches. Counting them would put sixty entries in front of the
# two that matter, and a gate whose output is mostly noise gets skimmed.
# ops_arm_prim.h is excluded for the same reason on the other
# architecture.
SRC=$(ls src/*.c src/*.h gui/*.c gui/*.h common/*.c common/*.h 2>/dev/null \
      | grep -vE "ops_(mmx|mmx_sse|sse2|arm|arm_prim|t2_arm)\.(c|h)$")

# path:literals-over-limit. Only non-zero entries.
#
#   src/cli_main.c:2  The --help text, 972 and 7348 characters, each
#                     assembled from adjacent literals. Real, and known:
#                     a compiler holding to the 509 minimum may truncate
#                     them, which would silently shorten --help rather
#                     than fail the build. Recorded rather than split,
#                     because splitting it into fputs calls is a change
#                     to the one function whose whole purpose is to be
#                     read as a block, and nothing in this tree can yet
#                     confirm VC++ 4.0 actually truncates.
STR_BASELINE="src/cli_main.c:2"

# ---- 31-character external identifiers -------------------------------
#
# Identifiers with external linkage only: file-scope names that are not
# static. Approximated by taking every identifier of 32+ characters and
# checking the truncation for duplicates - the approximation errs toward
# reporting more, which is the safe direction for a gate.
grep -rhoE "\b[A-Za-z_][A-Za-z0-9_]{31,}" $SRC 2>/dev/null \
    | sort -u > /tmp/vc4_longids.$$
NLONG=$(wc -l < /tmp/vc4_longids.$$)
COLL=$(cut -c1-31 /tmp/vc4_longids.$$ | sort | uniq -d)
if [ -n "$COLL" ]; then
    echo "FAIL: identifiers that become one symbol when truncated to 31:"
    for p in $COLL; do
        echo "  prefix $p"
        grep "^$p" /tmp/vc4_longids.$$ | sed 's/^/    /'
    done
    FAIL=1
fi

# The gate's positive control. The truncation must actually be able to
# collide; a cut that produced unique strings for every input would make
# the check above vacuous.
CTL=$(printf '%s\n%s\n' \
      "lz_a_very_long_identifier_name_one" \
      "lz_a_very_long_identifier_name_two" | cut -c1-31 | sort | uniq -d | wc -l)
if [ "$CTL" -ne 1 ]; then
    echo "FAIL: the truncation check cannot detect a planted collision"
    FAIL=1
fi

# ---- 509-character string literals -----------------------------------
#
# awk, not a shell regex: the pattern needs a backslash class and every
# attempt to pass one through the shell in this project has had the
# escaping eaten. Adjacent literals are concatenated by the translator,
# so runs of them on consecutive lines are summed.
awk '
    function flush(  ) {
        if (run > 509) printf "%s: %d-char literal ending line %d\n", fn, run, prev
        run = 0
    }
    FNR == 1 { flush(); fn = FILENAME }
    {
        line = $0
        # Adjacent literals concatenate ONLY when nothing but whitespace
        # separates them. A comma does not concatenate - it separates
        # table entries, and summing across one turns err.c 138 message
        # strings into a single 4642-character literal that does not
        # exist. That was this scan version one, and it reported the
        # whole message table as a violation.
        while (match(line, /"([^"\\]|\\.)*"/)) {
            gap = substr(line, 1, RSTART - 1)
            if (gap !~ /^[ \t]*$/) flush()
            run += RLENGTH - 2
            prev = FNR
            line = substr(line, RSTART + RLENGTH)
        }
        if (line !~ /^[ \t]*(\\)?$/) flush()
    }
    END { flush() }
' $SRC 2>/dev/null > /tmp/vc4_longstr.$$
# Ratchet per file, like the other language gates: the existing surface
# is real and a gate that reddens on all of it gets switched off.
while read -r path; do
    n=$(grep -c "^$path:" /tmp/vc4_longstr.$$)
    exp=0
    for e in $STR_BASELINE; do
        [ "${e%%:*}" = "$path" ] && exp=${e##*:}
    done
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $path has $n literal(s) over 509 chars, baseline $exp"
        grep "^$path:" /tmp/vc4_longstr.$$ | sed 's/^/      /'
        FAIL=1
    fi
done < <(cut -d: -f1 /tmp/vc4_longstr.$$ | sort -u)

# Positive control for the literal scan, same reasoning.
LCTL=$(printf 'char *x = "%0.s-" ;\n' | awk '
    { line = $0; n = 0
      while (match(line, /"([^"\\]|\\.)*"/)) { n += RLENGTH - 2
          line = substr(line, RSTART + RLENGTH) }
      print n }')
if [ "$LCTL" != "1" ]; then
    echo "FAIL: the literal scan does not measure a one-character literal"
    echo "      (got '$LCTL'), so its silence means nothing"
    FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
    NSTR=$(wc -l < /tmp/vc4_longstr.$$ 2>/dev/null || echo 0)
    rm -f /tmp/vc4_longids.$$ /tmp/vc4_longstr.$$
    echo "vc4_limits_gate: PASS ($NLONG identifier(s) over 31 chars, none"
    echo "  colliding; $NSTR literal(s) over 509, all baselined)"
    exit 0
fi
rm -f /tmp/vc4_longids.$$ /tmp/vc4_longstr.$$
echo "vc4_limits_gate: FAIL"
exit 1
