#!/bin/bash
# The language floor, as a gate instead of a memory.
#
# The target family includes Windows NT on MIPS built with Visual C++
# 4.0 - a 1995 C89 compiler. gcc and Watcom both accept far more than
# that, so nothing in the normal build says a word when C99 creeps in.
# This runs gcc -std=c89 -pedantic over src/ and counts the three
# constructs that are both easy to add by accident and impossible for
# VC++ 4.0:
#
#   mixed declarations and code   - a declaration after a statement
#   //                            - already 0 in this tree, kept at 0
#   designated initialisers       - [IDX] = val, C99 only
#
# NOT `long long` and NOT <stdint.h>. Those were 213 and 23 hits of
# known, tracked debt, and the plan for them was one header per
# toolchain rather than a per-site fix - src/lz_int.h is that header. A
# gate that fails on the whole existing surface fails every run and
# gets switched off; this one has to redden only on NEW breakage, so it
# is a RATCHET: a baseline count per file, and any increase is a
# failure.
#
# The baseline is the state at the commit that added this gate, and it
# is written here rather than regenerated, because a baseline the gate
# recomputes is a baseline that ratchets the wrong way.
#
# Exit codes:
#   0  PASS - no file exceeds its baseline
#   1  FAIL - a file gained a C89 violation
#   2  SKIP - no gcc
set -u
cd "$(dirname "$0")/.."

command -v gcc >/dev/null 2>&1 || { echo "c89_floor_gate: SKIP - no gcc"; exit 2; }

# file:count. Only files with a non-zero baseline are listed; everything
# else must stay at zero.
#
#   ops.c:1          `volatile float tamf` in lz_quantize_q8_int64
#                    (1718f11). The volatile is load-bearing - it stops
#                    Watcom fusing the int64 convert into the multiply,
#                    which costs 1 ULP - so moving it is not free.
#   forward.c:16     48327cc, 75c8adc and neighbours, in forward_attn's
#                    per-head blocks.
#
# matmul_q41_impl's used to be a third: `int i, t;` sat after the
# kernel-select call. Fixed rather than baselined - the whole
# declaration block moved above the call and the call became the
# function's first statement. Safe because nothing in those
# initialisers reads the tier: q41_fixed_epi asks lz_epi_mode and
# lz_epi_ready, and lz_row_pick - which does read it - still runs
# after. Verified byte-identical on both fixtures rather than argued.
#
# The rest predate the language floor. They are recorded, not excused:
# this gate stops the number growing, and each is a one-line reorder
# for whoever next edits that function. The two violations THIS
# gate was written after finding - both in ops_gdn.c, both introduced
# after bc4f1ba landed mid-session - are fixed rather than baselined,
# which is the difference between debt and a new hole.
# DESIGNATED INITIALISERS are the third construct, added after the
# compile pass below went in and found nothing: gcc reports them as a
# WARNING, so an errors-only check walks straight past 266 of them in
# err.c and 186 in gui/localized_strings.c. Both are the message tables,
# where the whole point of the form is that an enum value and its string
# cannot drift apart - so they are baselined rather than converted.
# Positional initialisers would put 452 strings in an order nothing
# checks, which trades a compiler the tree cannot run for a class of bug
# it currently cannot have.
BASELINE="ops.c:1 forward.c:16 err.c:266"

FAIL=0
TOTAL=0
for f in src/*.c; do
    b=$(basename "$f")
    exp=0
    for e in $BASELINE; do
        [ "${e%%:*}" = "$b" ] && exp=${e##*:}
    done
    n=$(gcc -std=c89 -pedantic -Isrc -fsyntax-only "$f" 2>&1 \
        | grep -cE "mixed declarations|ISO C90 forbids .*//|forbids specifying subobject" )
    TOTAL=$((TOTAL + n))
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $b has $n C89 violation(s), baseline $exp"
        gcc -std=c89 -pedantic -Isrc -fsyntax-only "$f" 2>&1 \
            | grep -E "mixed declarations|ISO C90 forbids .*//|forbids specifying subobject" \
            | head -5 | sed 's/^/      /'
        FAIL=1
    elif [ "$n" -lt "$exp" ]; then
        echo "note: $b is at $n, below its baseline of $exp - tighten BASELINE"
    fi
done

# The gate's own positive control: if -pedantic stopped reporting the
# construct at all, every count would be 0 and this would pass while
# measuring nothing. ops.c's one known violation is the control.
if [ "$TOTAL" -eq 0 ]; then
    echo "FAIL: total is 0, including ops.c's known one - the compiler is not"
    echo "      reporting mixed declarations and this gate is blind"
    FAIL=1
fi

# DOES IT COMPILE, which the counting above never asked.
#
# Everything before this point greps for two constructs and tolerates a
# baseline of them. That is a WARNING-class check and it is blind to the
# error class: `restrict` is C99, twenty-six declarations used it, every
# translation unit that saw one failed to PARSE, and this gate reported
# PASS across all forty errors. Counting patterns is not compiling.
#
# ERRORS ONLY, no baseline. A warning here is a style question with a
# recorded number above; an error means the floor is not met at all, and
# there is no such thing as an acceptable number of those.
#
# WHAT THIS DOES NOT PROVE, and the distinction matters more here than
# in most gates: green means the code is standard C89. It does NOT mean
# Visual C++ 4.0 accepts it - that compiler's C89 is incomplete and has
# quirks of its own, and nothing in this tree can run it. Reading this
# gate as "the MIPS target builds" is the mistake it is written to
# prevent.
CFAIL=0
for f in src/*.c; do
    case "$(basename "$f")" in
        ops_mmx.c|ops_mmx_sse.c|ops_sse2.c) isa="-mmmx -msse -msse2" ;;
        *) isa="" ;;
    esac
    n=$(gcc -std=c89 -pedantic -fsyntax-only -Isrc -Icommon \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 $isa "$f" 2>&1 \
        | grep -c " error: ")
    if [ "$n" -gt 0 ]; then
        echo "FAIL: $(basename "$f") - $n C89 compile error(s)"
        gcc -std=c89 -pedantic -fsyntax-only -Isrc -Icommon \
            -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 $isa "$f" 2>&1 \
            | grep -m2 " error: " | sed 's/^/      /'
        CFAIL=$((CFAIL + n))
        FAIL=1
    fi
done
[ "$CFAIL" -eq 0 ] && echo "c89 compile: 0 errors across $(ls src/*.c | wc -l) file(s)"

if [ "$FAIL" -eq 0 ]; then
    echo "c89_floor_gate: PASS ($TOTAL known violation(s), none new)"
    exit 0
fi
echo "c89_floor_gate: FAIL"
exit 1
