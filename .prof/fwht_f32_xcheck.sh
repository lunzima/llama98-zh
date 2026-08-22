#!/bin/bash
# lz_fwht's SSE1 float butterfly against its C body.
#
#   bash .prof/fwht_f32_xcheck.sh
#
# ONE BINARY, THREE TIERS - src/ops.c dispatches on g_kernel, the probe
# takes the tier as argv. The mutation needs a second build.
#
# THE MUTATION IS CHOSEN FOR WHAT THIS KERNEL CAN GET WRONG. Its answer
# is bit-identical to the C by construction (each output is one add or
# one sub of the same pair the scalar loop forms - no constant, no
# reassociation), so a value mutation would be testing nothing. What is
# NOT structural is the INDEXING, and the boundary between the stages the
# kernel takes (len >= 4) and the ones the C keeps (len 1, 2). So the
# mutation moves the second half's pointer by one float: still a well
# formed butterfly, still finite numbers, wrong pairing.
#
# The third clause is the one that has actually caught something here:
# on the int32 sibling the mutation once changed NOTHING because the
# kernel's includes had landed inside an ARM-only #if, so "on" and "off"
# were the same code. A mutation that moves nothing is a failure, not a
# pass.
#
# PC=24 is set inside the probe, not here - on x87 the C body's temporaries
# live in 80-bit registers and no float32 kernel could match it. That is
# the Watcom arm's precondition; the same trap the F32 matmul cell hit.
#
# REACHED IN THE SHIPPING BINARY, not just in this probe - a separate
# claim, and one the probe cannot make. Measured 2026-08-23 on kmr20 with
# build/norm_int_gate_tokens_zh.txt, --kernel sse2, --dump-all-logits
# (78,905,344 bytes), three engine builds from one snapshot:
#   shipped   --kv-rot on  fdb4c596...   --kv-rot off  07f14ea6...
#   mutated   --kv-rot on  652a7ade...   --kv-rot off  07f14ea6...
#   SIMD arm forced off, --kv-rot on     fdb4c596...
# So: the kernel runs (mutating it moves the logits), the mutation is
# confined to it (rot off is untouched), and the SIMD and C arms agree on
# all 78 MB rather than only on the probe's sweep.
#
# THE WATCOM HALF IS NOT TESTED BY THIS DRIVER (it builds with $CC).
# `bash .prof/wbuild_xcheck.sh fwht_f32_xcheck` runs the same probe
# through wcc386/wlink; measured there 2026-08-23, ref/mmx/sse2 all
# 27fdf9c0 - the same value gcc prints below, so the two twins agree bit
# for bit and not merely each with its own C.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/fwhtf}
CC=${CC:-gcc}
D="-DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
F="-O2 -std=c99 -mmmx -msse -msse2"
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     ops_mmx ops_mmx_sse ops_sse2 fwht cpucheck err"

rm -rf "$O"; mkdir -p "$O/on" "$O/mut"
cp -r "$R/src" "$O/on/src"; cp -r "$R/src" "$O/mut/src"
sed -i 's|float \*b = y + i + len;|float *b = y + i + len + 1;|' \
    "$O/mut/src/ops_mmx_sse.c"
grep -q 'y + i + len + 1' "$O/mut/src/ops_mmx_sse.c" || {
    echo "FAIL: the mutation did not apply - the kernel body moved"; exit 1; }

for T in on mut; do
    list=""
    for f in $SRC; do list="$list $O/$T/src/$f.c"; done
    if ! $CC $F $D -I"$O/$T/src" -o "$O/bin_$T" \
            "$R/.prof/fwht_f32_xcheck.c" $list -lm 2>"$O/$T.log"; then
        echo "fwht_f32_xcheck: SKIP - $T arm did not build"
        tail -5 "$O/$T.log"; exit 2
    fi
done

get() { "$O/bin_$1" "$2" | grep '^FWHTF'; }
REF=$(get on ref); MMX=$(get on mmx); SSE=$(get on sse2)
MREF=$(get mut ref); MSSE=$(get mut sse2)
echo "  shipped   ref=$REF  mmx=$MMX  sse2=$SSE"
echo "  mutated   ref=$MREF  sse2=$MSSE"

FAILED=0
[ "$SSE" = "$REF" ] || { echo "FAIL: the SSE kernel differs from the C body"; FAILED=1; }
[ "$MMX" = "$REF" ] || { echo "FAIL: the mmx tier differs from the C body"; FAILED=1; }
[ "$MREF" = "$REF" ] || { echo "FAIL: the mutation changed the C body too"; FAILED=1; }
if [ "$MSSE" = "$REF" ]; then
    echo "FAIL: mutating the kernel changed NOTHING - it is never reached,"
    echo "      so the check above ran the C body twice"
    FAILED=1
fi
[ $FAILED -eq 0 ] || exit 1
echo "fwht_f32_xcheck: PASS"
exit 0
