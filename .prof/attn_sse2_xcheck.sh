#!/bin/bash
# Attention's two fixed-point MACs: SSE2 twins against MMX against C.
#
#   bash .prof/attn_sse2_xcheck.sh
#
# The probe calls all three bodies directly, so ONE build is enough for
# the identity; the mutation needs a second. Read .prof/attn_sse2_xcheck.c
# for why lane order is the failure mode a value sweep has to be designed
# around here, and for what this probe deliberately leaves to
# build/kernel_isa_identity_gate.sh.
#
# THE OTHER TWO HALVES OF THE CLAIM, both measured 2026-08-23 on kmr20
# with build/norm_int_gate_tokens_zh.txt and neither testable from here:
#
#   TARGET COMPILER. `bash .prof/wbuild_xcheck.sh attn_sse2_xcheck` gives
#   ATTNSSE2 4f31504d, the same value gcc prints below - so the twins
#   agree with each other, not merely each with its own C. (That probe
#   drops the MMX pair arm: lz_wsum_pair_asm has no exported symbol on
#   Watcom. See the .c file.)
#
#   REACHED IN THE SHIPPING BINARY, four cells, four mutations, each one
#   moving --kernel sse2's --dump-all-logits while --kernel mmx stays at
#   07f14ea6:
#     wsump sse2I  lane swap in lz_wsum_pair_sse2  -> all 6 combinations
#                  of build/kernel_isa_identity_gate.sh go red
#     atdot sse2I  part32_x16(w + 1, x)            -> same, all 6 red
#     wsump sse2A  punpcklwd -> punpckhwd          -> sse2 a3d7a8c9
#     atdot sse2A  lz_dot32_x16_sse2(w + 1, x)     -> sse2 4dd37917
#   Unlike fwhtf, whose two x86 tiers share ONE body and are therefore
#   invisible to a tier comparison, these four are exactly what that
#   gate can see - which is why it is cited as the dispatch half here.
#
# THE MUTATION swaps the two halves of one interleave
# (unpacklo/unpackhi), which is the mistake this kernel can actually
# make: it keeps every product, every sign and every magnitude, and only
# writes them to the wrong lanes. A value mutation would be testing
# nothing - integer sums are exact under reassociation, so the arithmetic
# is identical by construction and only the addressing is at risk.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/attnsse2}
CC=${CC:-gcc}
D="-DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
F="-O2 -std=c99 -mmmx -msse -msse2"
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     ops_mmx ops_mmx_sse ops_sse2 fwht cpucheck err"

rm -rf "$O"; mkdir -p "$O/on" "$O/mut"
cp -r "$R/src" "$O/on/src"; cp -r "$R/src" "$O/mut/src"
sed -i 's|_mm_madd_epi16(_mm_unpacklo_epi16(al, bl), c)|_mm_madd_epi16(_mm_unpackhi_epi16(al, bl), c)|' \
    "$O/mut/src/ops_sse2.c"
grep -c '_mm_madd_epi16(_mm_unpackhi_epi16(al, bl), c)' \
    "$O/mut/src/ops_sse2.c" | grep -qx 2 || {
    echo "FAIL: the mutation did not apply - the kernel body moved"; exit 1; }

for T in on mut; do
    list=""
    for f in $SRC; do list="$list $O/$T/src/$f.c"; done
    if ! $CC $F $D -I"$O/$T/src" -o "$O/bin_$T" \
            "$R/.prof/attn_sse2_xcheck.c" $list -lm 2>"$O/$T.log"; then
        echo "attn_sse2_xcheck: SKIP - $T arm did not build"
        tail -5 "$O/$T.log"; exit 2
    fi
done

ON=$("$O/bin_on"); ON_RC=$?
MUT=$("$O/bin_mut" 2>&1); MUT_RC=$?
echo "  shipped  rc=$ON_RC  $ON"
echo "  mutated  rc=$MUT_RC  $(echo "$MUT" | head -1)"

FAILED=0
if [ "$ON_RC" != 0 ]; then
    echo "FAIL: the shipped kernels do not agree with the C reference"
    echo "$ON" | head -5
    FAILED=1
fi
if [ "$MUT_RC" = 0 ]; then
    echo "FAIL: mutating the SSE2 kernel changed NOTHING - either it is"
    echo "      never called or the sweep cannot see a lane swap, and"
    echo "      then the check above proved nothing"
    FAILED=1
fi
[ $FAILED -eq 0 ] || exit 1
echo "attn_sse2_xcheck: PASS"
exit 0
