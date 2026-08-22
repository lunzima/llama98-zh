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
SRC="fwht ops ops_quant ops_sched ops_norm ops_gdn ops_matmul
     ops_t2_scalar ops_mmx ops_mmx_sse ops_sse2 cpucheck err"

rm -rf "$O"; mkdir -p "$O"
for T in on off mut; do
    mkdir -p "$O/$T"; cp -r "$R/src" "$O/$T/src"
done
sed -i 's|^#define LZ_FWHT_MMX_EXTERN 1|/* disabled for the control arm */|' \
    "$O/off/src/ops_mmx.h"
sed -i 's|__m64 s = _mm_add_pi32(u, v);|__m64 s = _mm_sub_pi32(u, v);|' \
    "$O/mut/src/ops_mmx.c"
grep -q 'LZ_FWHT_MMX_EXTERN 1' "$O/off/src/ops_mmx.h" && {
    echo "FAIL: the control arm still has the kernel enabled"; exit 1; }
grep -q '_mm_sub_pi32(u, v);' "$O/mut/src/ops_mmx.c" || {
    echo "FAIL: the mutation did not apply - the kernel body moved"; exit 1; }

for T in on off mut; do
    list=""
    for f in $SRC; do list="$list $O/$T/src/$f.c"; done
    if ! $CC $F $D -I"$O/$T/src" -o "$O/bin_$T" \
            "$R/.prof/fwht_mmx_xcheck.c" $list -lm 2>"$O/$T.log"; then
        echo "fwht_mmx_xcheck: SKIP - $T arm did not build"
        tail -5 "$O/$T.log"; exit 2
    fi
    "$O/bin_$T" > "$O/$T.txt" || exit 1
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
if cmp -s "$O/mut.txt" "$O/off.txt"; then
    echo "FAIL: swapping the kernel's add and sub changed NOTHING - the"
    echo "      MMX branch is not reached, so the check above compared"
    echo "      the C body to itself"
    FAILED=1
else
    echo "-- mutation: $(diff "$O/off.txt" "$O/mut.txt" | grep -c '^<') lines differ, as required --"
fi
[ $FAILED -eq 0 ] || exit 1
echo "fwht_mmx_xcheck: PASS"
exit 0
