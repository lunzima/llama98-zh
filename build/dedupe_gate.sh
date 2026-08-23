#!/bin/bash
# Shared-helper dedupe gate (task #31).
#
# Asserts that each of the five names task #30's kernel extraction kept
# re-duplicating - lz_i32f, LZ_WSUM_CHUNK, LZ_SLOT_NEXT, gdn_tail_row,
# p2_shift_of - is DEFINED exactly once across the whole src/ tree, in
# src/ops_kernel_shared.h. A second definition anywhere else (under its
# real name or a renamed copy like lz_i32f_wsum/gdn_tail_row_mmx) means
# the next kernel-extraction piece re-introduced the exact defect this
# gate exists to catch: a project-wide invariant (lz_i32f) or a
# ring-buffer walk constant (LZ_WSUM_CHUNK/LZ_SLOT_NEXT) duplicated
# across a TU boundary, free to drift the moment one copy is edited
# and the others are not.
#
# Exit codes:
#   0  PASS - each name defined exactly once, in ops_kernel_shared.h
#   1  FAIL - a name is defined zero times, more than once, or the one
#             definition is not in ops_kernel_shared.h
#
# Needs no build, no disassembly - grep -c against the checked-out
# source is the whole check, so it runs in milliseconds and catches a
# recurrence before either compiler ever sees it.
set -u
cd "$(dirname "$0")/.."

FAIL=0
SHARED=src/ops_kernel_shared.h

check_one() {
    local pattern="$1" name="$2"
    local hits total
    hits=$(grep -rl -- "$pattern" src/*.h src/*.c 2>/dev/null)
    total=$(printf '%s\n' "$hits" | grep -c -- "$pattern" 2>/dev/null)
    # grep -c per file, summed: a name defined twice in ops.c would
    # still show as "one file" via grep -l, so count occurrences, not
    # files.
    total=0
    for f in src/*.h src/*.c; do
        [ -f "$f" ] || continue
        n=$(grep -c -- "$pattern" "$f")
        total=$((total + n))
        if [ "$n" -gt 0 ] && [ "$f" != "$SHARED" ]; then
            echo "FAIL: $name defined outside $SHARED (in $f, $n hit(s))"
            FAIL=1
        fi
    done
    if [ "$total" -eq 0 ]; then
        echo "FAIL: $name has zero definitions anywhere"
        FAIL=1
    elif [ "$total" -gt 1 ]; then
        echo "FAIL: $name defined $total times, expected 1"
        FAIL=1
    else
        echo "OK:   $name defined exactly once, in $SHARED"
    fi
}

check_one '^static float lz_i32f('       'lz_i32f'
check_one '^#define LZ_WSUM_CHUNK'       'LZ_WSUM_CHUNK'
check_one '^#define LZ_SLOT_NEXT'        'LZ_SLOT_NEXT'
check_one '^static void gdn_tail_row('   'gdn_tail_row'
check_one '^static int p2_shift_of('     'p2_shift_of'

# ---- the same defect, in macros, found by scanning rather than by
# ---- being reported ---------------------------------------------------
# Six more were carrying two to four copies each, all agreeing, none
# with anything to say if one were edited and the others not:
#
#   LZ_GDN_FIXED     4 (ops.c, ops_gdn.c, ops_matmul.c, ops_sched.c)
#   LZ_NORM_MAX_N    3 (ops_norm.c, ops_quant.c, ops_sched.c)
#   LZ_EPI_MAX_NG    2 (ops.c, ops_matmul.c)
#   LZ_MM_WIDEN_MAX  2 (ops.c, ops_matmul.c)
#   LZ_X87_FLOAT_CW  2 (ops.c, ops_quant.c)
#   LZ_ATTN_MAX_HD   2 (ops.h, ops_gdn.c)
#
# FOUR OF THE SIX BOUND A STATIC ARRAY, which is what makes this a
# memory-safety check and not a tidiness one: two TUs disagreeing on
# LZ_NORM_MAX_N is one side indexing a buffer the other side allocated
# shorter. ops.c's own note on LZ_MM_WIDEN_MAX read "Defined here as
# well as in ops_matmul.c, because lz_matmul_xq_nt checks it" - a
# description of the duplication offered as a reason for it.
#
# They live in ops.h now. HOME is a parameter because these belong with
# the tier flags rather than with the kernel helpers above.
check_one_in() {
    local pattern="$1" name="$2" home="$3"
    local total=0 n f
    for f in src/*.h src/*.c; do
        [ -f "$f" ] || continue
        n=$(grep -c -- "$pattern" "$f")
        total=$((total + n))
        if [ "$n" -gt 0 ] && [ "$f" != "$home" ]; then
            echo "FAIL: $name defined outside $home (in $f, $n hit(s))"
            FAIL=1
        fi
    done
    if [ "$total" -eq 1 ]; then
        echo "OK:   $name defined exactly once, in $home"
    else
        echo "FAIL: $name defined $total times, expected 1"
        FAIL=1
    fi
}

check_one_in '^#define LZ_GDN_FIXED '    'LZ_GDN_FIXED'    src/ops.h
check_one_in '^#define LZ_NORM_MAX_N '   'LZ_NORM_MAX_N'   src/ops.h
check_one_in '^#define LZ_EPI_MAX_NG '   'LZ_EPI_MAX_NG'   src/ops.h
check_one_in '^#define LZ_MM_WIDEN_MAX ' 'LZ_MM_WIDEN_MAX' src/ops.h
check_one_in '^#define LZ_X87_FLOAT_CW ' 'LZ_X87_FLOAT_CW' src/ops.h
check_one_in '^#define LZ_EPI_FIXED '    'LZ_EPI_FIXED'    src/ops.h
check_one_in '^#define LZ_ATTN_MAX_HD '  'LZ_ATTN_MAX_HD'  src/ops.h

# Renamed-copy check, separate from the count above: a name defined
# once under a DIFFERENT identifier (lz_i32f_wsum, gdn_tail_row_mmx)
# would pass every test above while still being exactly the defect
# task #31 reported - grep for the two renames that actually happened
# once, so a third one is caught the same way.
for renamed in lz_i32f_wsum gdn_tail_row_mmx; do
    # ops_kernel_shared.h's own header comment names both renames on
    # purpose, as the record of what task #31 fixed - excluded here,
    # not because the check is wrong but because that file is the one
    # place this text is supposed to still exist.
    n=$(grep -rc -- "$renamed" src/*.h src/*.c 2>/dev/null \
        | grep -v "^$SHARED:" | awk -F: '{s+=$2} END{print s+0}')
    if [ "$n" -gt 0 ]; then
        echo "FAIL: renamed duplicate '$renamed' still present outside $SHARED ($n hit(s))"
        FAIL=1
    fi
done

if [ "$FAIL" -eq 0 ]; then
    echo "dedupe_gate: PASS"
    exit 0
else
    echo "dedupe_gate: FAIL"
    exit 1
fi
