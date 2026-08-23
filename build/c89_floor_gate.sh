#!/bin/bash
# Iron law 10's language floor, as a gate instead of a memory.
#
# The target family includes Windows NT on MIPS built with Visual C++
# 4.0 - a 1995 C89 compiler. gcc and Watcom both accept far more than
# that, so nothing in the normal build says a word when C99 creeps in.
# This runs gcc -std=c89 -pedantic over src/ and counts the two
# constructs that are BOTH easy to add by accident and impossible for
# VC++ 4.0:
#
#   mixed declarations and code   - a declaration after a statement
#   //                            - already 0 in this tree, kept at 0
#
# NOT `long long` and NOT <stdint.h>. Those are 213 and 23 hits of
# KNOWN, TRACKED debt (CLAUDE.md iron law 10 counts them), and the plan
# for them is one header per toolchain rather than a per-site fix. A
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
BASELINE="ops.c:1 forward.c:16"

FAIL=0
TOTAL=0
for f in src/*.c; do
    b=$(basename "$f")
    exp=0
    for e in $BASELINE; do
        [ "${e%%:*}" = "$b" ] && exp=${e##*:}
    done
    n=$(gcc -std=c89 -pedantic -Isrc -fsyntax-only "$f" 2>&1 \
        | grep -cE "mixed declarations|ISO C90 forbids .*//" )
    TOTAL=$((TOTAL + n))
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $b has $n C89 violation(s), baseline $exp"
        gcc -std=c89 -pedantic -Isrc -fsyntax-only "$f" 2>&1 \
            | grep -E "mixed declarations|ISO C90 forbids .*//" \
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

if [ "$FAIL" -eq 0 ]; then
    echo "c89_floor_gate: PASS ($TOTAL known violation(s), none new)"
    exit 0
fi
echo "c89_floor_gate: FAIL"
exit 1
