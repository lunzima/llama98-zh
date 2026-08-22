#!/bin/bash
# norm_ss_fixed's SSE2 element loop against its C body, plus the
# mutation that shows the comparison is live.
#
#   bash .prof/norm_ss_sse2_xcheck.sh
#
# ONE BINARY, THREE TIERS. src/ops_quant.c dispatches on g_kernel, so
# --kernel ref and mmx reach the C loop and sse2 reaches the kernel -
# the probe takes the tier as argv and this runs all three. That is the
# comparison; it needs no second build.
#
# The mutation DOES need one: packs_epi32's operand order swapped, which
# interleaves the two halves of every group of 8 and is exactly the kind
# of indexing mistake a bit-identity check exists for. ref must stay
# unchanged under it (the C body is untouched) and sse2 must move.
#
# -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 are what the Makefile passes to every TU
# and what src/ops_sse2.h's body sits behind; without them the kernel is
# not in the build and every arm agrees for the wrong reason.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/normsse2}
CC=${CC:-gcc}
D="-DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
F="-O2 -std=c99 -mmmx -msse -msse2"
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     ops_mmx ops_mmx_sse ops_sse2 fwht cpucheck err"

rm -rf "$O"; mkdir -p "$O/on" "$O/mut"
cp -r "$R/src" "$O/on/src"; cp -r "$R/src" "$O/mut/src"
sed -i 's|_mm_packs_epi32(a, b)|_mm_packs_epi32(b, a)|' "$O/mut/src/ops_sse2.c"
grep -q '_mm_packs_epi32(b, a)' "$O/mut/src/ops_sse2.c" || {
    echo "FAIL: the mutation did not apply - the kernel body moved"; exit 1; }

for T in on mut; do
    list=""
    for f in $SRC; do list="$list $O/$T/src/$f.c"; done
    if ! $CC $F $D -I"$O/$T/src" -o "$O/bin_$T" \
            "$R/.prof/norm_ss_sse2_xcheck.c" $list -lm 2>"$O/$T.log"; then
        echo "norm_ss_sse2_xcheck: SKIP - $T arm did not build"
        tail -5 "$O/$T.log"; exit 2
    fi
done

get() { "$O/bin_$1" "$2" | grep '^NORMSS'; }
REF=$(get on ref); MMX=$(get on mmx); SSE=$(get on sse2)
MREF=$(get mut ref); MSSE=$(get mut sse2)
echo "  shipped   ref=$REF  mmx=$MMX  sse2=$SSE"
echo "  mutated   ref=$MREF  sse2=$MSSE"

FAILED=0
[ "$SSE" = "$REF" ] || { echo "FAIL: the SSE2 kernel differs from the C body"; FAILED=1; }
[ "$MMX" = "$REF" ] || { echo "FAIL: --kernel mmx does not reach the C body"; FAILED=1; }
[ "$MREF" = "$REF" ] || { echo "FAIL: the mutation changed the C body too"; FAILED=1; }
if [ "$MSSE" = "$REF" ]; then
    echo "FAIL: mutating the kernel changed NOTHING - the SSE2 branch is"
    echo "      never reached, so the check above ran the C body twice"
    FAILED=1
fi
[ $FAILED -eq 0 ] || exit 1
echo "norm_ss_sse2_xcheck: PASS"
exit 0
