#!/bin/bash
# lz_fwht_i32's MMX stage kernel against the C body, plus the mutation
# that shows the comparison is live.
#
#   bash .prof/fwht_mmx_xcheck.sh
#
# Three builds from ONE snapshot of src/, differing by one line each:
#   on   as shipped
#   off  LZ_FWHT_MMX_EXTERN removed - the C body, which is the control
#   mut  the kernel's add and sub swapped - must differ from off
#
# THE MUTATION IS NOT OPTIONAL HERE and it earned that on the first run.
# The MMX branch was compiled out entirely: fwht.c's includes had landed
# inside a `#if defined(LZ_ARM_ASM_EXTERN)`, which is false on x86, so
# LZ_FWHT_MMX_EXTERN was never seen. "on" and "off" agreed perfectly -
# because they were the same code - and only the mutation showing NO
# difference either revealed it.
#
# -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 are not decoration: the Makefile passes
# them to every TU, and src/ops_mmx.h's whole body sits behind the
# first. A build without them is a build where this kernel does not
# exist, which is the same green-for-the-wrong-reason failure.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/fwhtmmx}
CC=${CC:-gcc}
D="-DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
F="-O2 -std=c99 -mmmx -msse -msse2"
SRC="fwht ops ops_kernel ops_epi ops_quant ops_sched ops_norm ops_gdn ops_matmul
     ops_t2_scalar ops_mmx ops_mmx_sse ops_sse2 cpucheck err"

rm -rf "$O"; mkdir -p "$O"
for T in on off mut sse2 mut2; do
    mkdir -p "$O/$T"; cp -r "$R/src" "$O/$T/src"
done
sed -i 's|^#define LZ_FWHT_MMX_EXTERN 1|/* disabled for the control arm */|' \
    "$O/off/src/ops_mmx.h"
sed -i 's|__m64 s = _mm_add_pi32(u, v);|__m64 s = _mm_sub_pi32(u, v);|' \
    "$O/mut/src/ops_mmx.c"
sed -i 's|_mm_add_epi32(u, v)|_mm_sub_epi32(u, v)|' "$O/mut2/src/ops_sse2.c"
grep -q 'LZ_FWHT_MMX_EXTERN 1' "$O/off/src/ops_mmx.h" && {
    echo "FAIL: the control arm still has the kernel enabled"; exit 1; }
grep -q '_mm_sub_pi32(u, v);' "$O/mut/src/ops_mmx.c" || {
    echo "FAIL: the mutation did not apply - the kernel body moved"; exit 1; }
grep -q '_mm_sub_epi32(u, v)' "$O/mut2/src/ops_sse2.c" || {
    echo "FAIL: the SSE2 mutation did not apply - that body moved"; exit 1; }

for T in on off mut sse2 mut2; do
    list=""
    for f in $SRC; do list="$list $O/$T/src/$f.c"; done
    if ! $CC $F $D -I"$O/$T/src" -o "$O/bin_$T" \
            "$R/.prof/fwht_mmx_xcheck.c" $list -lm 2>"$O/$T.log"; then
        echo "fwht_mmx_xcheck: SKIP - $T arm did not build"
        tail -5 "$O/$T.log"; exit 2
    fi
    /bin/true > "$O/$T.txt"
    for K in ref mmx sse2; do
        "$O/bin_$T" $K >> "$O/$T.txt" || exit 1
    done
done

FAILED=0
if cmp -s "$O/on.txt" "$O/off.txt"; then
    echo "-- MMX kernel is bit-identical to the C body --"
    grep '^FWHT' "$O/on.txt"
else
    echo "FAIL: the MMX kernel and the C body disagree"
    diff "$O/off.txt" "$O/on.txt" | head -10
    FAILED=1
fi
if cmp -s "$O/sse2.txt" "$O/off.txt"; then
    echo "-- SSE2 kernel is bit-identical to the C body --"
else
    echo "FAIL: the SSE2 kernel and the C body disagree"
    diff "$O/off.txt" "$O/sse2.txt" | head -10
    FAILED=1
fi
for M in mut mut2; do
    if cmp -s "$O/$M.txt" "$O/off.txt"; then
        echo "FAIL: mutating '$M' changed NOTHING - that branch is never"
        echo "      reached, so the comparison above ran the C body twice"
        FAILED=1
    else
        echo "-- mutation $M: $(diff "$O/off.txt" "$O/$M.txt" | grep -c '^<') lines differ, as required --"
    fi
done
[ $FAILED -eq 0 ] || exit 1
echo "fwht_mmx_xcheck: PASS"
exit 0
