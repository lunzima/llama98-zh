#!/bin/bash
# lz_matmul's SSE1 F32-weight row against its C body.
#
#   bash .prof/matmul_f32_xcheck.sh
#
# ONE BINARY, THREE TIERS. src/ops.c dispatches on g_kernel, so ref
# reaches the C loop and mmx/sse2 the kernel; the probe takes the tier
# as argv. The mutation needs a second build - the second load's offset
# moved by one float, which is the indexing mistake a float kernel can
# make while still producing plausible numbers.
#
# WHY THIS ONE IS NOT OBVIOUSLY BIT-IDENTICAL, unlike the integer
# kernels: float addition does not reassociate, so a horizontal-add SIMD
# version would change the answer. This one does not, because the C
# already keeps EIGHT accumulators striding by 8 - lane k of the two xmm
# registers accumulates exactly what scalar a_k does, in the same order
# - and the final combine stays in C.
#
# The Watcom half has a precondition this driver cannot express and the
# probe therefore sets itself: PC=24. On x87 the C body's accumulators
# live in 80-bit registers, so without lz_fpu_float_begin the C is not
# computing in float32 at all and no float32 kernel could match it. The
# first run of this check reported the Watcom cell wrong for exactly
# that reason. Verified by hand there: ref/mmx/sse2 all 1ccc199f, the
# same value gcc produces, and a mutated kernel gives 45ee3c51 with ref
# unchanged.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/mmf32}
CC=${CC:-gcc}
D="-DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
F="-O2 -std=c99 -mmmx -msse -msse2"
SRC="ops ops_kernel ops_epi ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     ops_mmx ops_mmx_sse ops_sse2 fwht cpucheck err"

rm -rf "$O"; mkdir -p "$O/on" "$O/mut"
cp -r "$R/src" "$O/on/src"; cp -r "$R/src" "$O/mut/src"
sed -i 's|_mm_loadu_ps(row + j + 4),|_mm_loadu_ps(row + j + 3),|' \
    "$O/mut/src/ops_mmx_sse.c"
grep -q 'row + j + 3' "$O/mut/src/ops_mmx_sse.c" || {
    echo "FAIL: the mutation did not apply - the kernel body moved"; exit 1; }

for T in on mut; do
    list=""
    for f in $SRC; do list="$list $O/$T/src/$f.c"; done
    if ! $CC $F $D -I"$O/$T/src" -o "$O/bin_$T" \
            "$R/.prof/matmul_f32_xcheck.c" $list -lm 2>"$O/$T.log"; then
        echo "matmul_f32_xcheck: SKIP - $T arm did not build"
        tail -5 "$O/$T.log"; exit 2
    fi
done

get() { "$O/bin_$1" "$2" | grep '^MATMULF32'; }
REF=$(get on ref); MMX=$(get on mmx); SSE=$(get on sse2)
MREF=$(get mut ref); MSSE=$(get mut sse2)
echo "  shipped   ref=$REF  mmx=$MMX  sse2=$SSE"
echo "  mutated   ref=$MREF  sse2=$MSSE"

# ---- SOFT-FLOAT, the fourth regime -------------------------------------
# THE PROBE ALREADY TOOK 'a' AND 'c' AND THIS SCRIPT NEVER PASSED THEM.
# The .c grew the two ARM tiers when the cell landed (14c0a0d) and was
# checked by hand; a check run by hand is not a gate, and the arm-asm
# cell has been in the tree since with nothing asserting it here.
#
# WHY IT IS A SEPARATE REGIME and not one more tier: gcc/x86 evaluates in
# SSE registers, Watcom evaluates in x87's 80-bit ones, and the ARM cross
# build has no float unit at all - three different ways to round the same
# expression. mmx-against-sse2 cannot see any of that, because both are
# the same compiler on the same registers.
#
# Skipped, loudly, when the cross toolchain is absent - the x86 rows
# above still stand and the report says which regime went unchecked.
ARM_CC=$(command -v arm-linux-gnueabi-gcc-14 2>/dev/null \
      || command -v arm-linux-gnueabi-gcc 2>/dev/null || true)
ARM_ROWS=""
if [ -n "$ARM_CC" ] && command -v qemu-arm-static >/dev/null 2>&1; then
    AF="-O2 -std=c99 -static -march=armv5te -mtune=arm926ej-s"
    AF="$AF -mfloat-abi=soft -DLZ_NO_FPU=1 -ffp-contract=off"
    ASRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_arm
          ops_t2_scalar ops_arm fwht cpucheck err"
    alist=""
    for f in $ASRC; do alist="$alist $O/on/src/$f.c"; done
    if $ARM_CC $AF -I"$O/on/src" -o "$O/bin_arm" \
            "$R/.prof/matmul_f32_xcheck.c" $alist -lm 2>"$O/arm.log"; then
        AREF=$(qemu-arm-static "$O/bin_arm" c | grep '^MATMULF32')
        AASM=$(qemu-arm-static "$O/bin_arm" a | grep '^MATMULF32')
        echo "  soft-fp   arm-c=$AREF  arm-asm=$AASM"
        ARM_ROWS=1
    else
        echo "  soft-fp   SKIP - the ARM arm did not build"
        tail -3 "$O/arm.log"
    fi
else
    echo "  soft-fp   SKIP - no ARM cross toolchain or qemu; the x87 and"
    echo "            soft-float regimes go unchecked by this run"
fi

FAILED=0
[ "$SSE" = "$REF" ] || { echo "FAIL: the SSE kernel differs from the C body"; FAILED=1; }
[ "$MMX" = "$REF" ] || { echo "FAIL: the mmx tier differs from the C body"; FAILED=1; }
[ "$MREF" = "$REF" ] || { echo "FAIL: the mutation changed the C body too"; FAILED=1; }
if [ "$MSSE" = "$REF" ]; then
    echo "FAIL: mutating the kernel changed NOTHING - it is never reached,"
    echo "      so the check above ran the C body twice"
    FAILED=1
fi
if [ -n "$ARM_ROWS" ]; then
    # The value is compared ACROSS regimes, not just within: a soft-float
    # build that agreed with itself and disagreed with SSE would pass
    # every within-regime check and still be the bug this exists for.
    [ "$AREF" = "$REF" ] || {
        echo "FAIL: the soft-float C body differs from the x86 one"
        FAILED=1; }
    [ "$AASM" = "$REF" ] || {
        echo "FAIL: the ARM hand kernel differs from the x86 C body"
        FAILED=1; }
fi
[ $FAILED -eq 0 ] || exit 1
echo "matmul_f32_xcheck: PASS"
exit 0
