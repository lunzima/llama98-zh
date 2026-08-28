#!/bin/bash
# Drive build/i32f_acc_probe.c on both toolchains and compare the two.
#
# THE KERNELS THIS COVERS were the last two in
# build/kernel_probe_census.sh's "reached on paper only" column:
# lz_i32f_acc32_sse2_asm and lz_i32f_acc4_sse_asm. gcc compiles no
# #pragma aux body, so a gcc-built probe reaches them and executes
# nothing - the column exists because lz_gdn1_x4_asm sat in the REACHED
# one while accumulating into an uninitialised mm7.
#
# WHAT EACH ARM ASSERTS. The probe's own assertion is TIER AGAINST TIER
# (sse against sse2), which contains no reference and therefore holds on
# either toolchain; under gcc it also asserts both against the scalar
# reference. It does NOT assert the reference under Watcom, where x87
# and _control87(PC_24) still leave about 3% of lanes differing in the
# last bit - and the direction is settled by hash rather than argued:
# both Watcom kernels, both gcc kernels and the gcc reference all give
# 6538995D, and only the Watcom-compiled reference differs.
#
# This driver adds the part neither arm can do alone: the two builds'
# kernel hashes must be equal.
#
# Exit: 0 pass, 1 fail, 2 skip (no Watcom - the gcc half still runs).
set -u
cd "$(dirname "$0")/.."

CC=${CC:-gcc}
O=${O:-/tmp/i32facc}
rm -rf "$O"; mkdir -p "$O"
ENG="ops ops_kernel ops_epi ops_quant ops_sched ops_norm ops_gdn ops_matmul
     ops_t2_scalar ops_mmx ops_mmx_sse ops_sse2 fwht cpucheck err"
SRCS=""
for f in $ENG; do SRCS="$SRCS src/$f.c"; done

echo "-- gcc: the intrinsics tiers --"
if ! $CC -O2 -std=c99 -mmmx -msse -msse2 -ffp-contract=off \
        -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -Isrc -o "$O/g.exe" \
        build/i32f_acc_probe.c $SRCS -lm > "$O/gbuild.log" 2>&1; then
    echo "i32f_acc_gate: FAIL - the gcc probe did not build"
    tail -8 "$O/gbuild.log"; exit 1
fi
"$O/g.exe" > "$O/g.log" 2>&1
grc=$?
tail -3 "$O/g.log"
[ $grc -eq 0 ] || { echo "i32f_acc_gate: FAIL (gcc)"; exit 1; }
GH=$(grep -o "sse hash [0-9A-F]*" "$O/g.log" | head -1 | awk '{print $3}')

echo "-- Open Watcom: the #pragma aux tiers --"
if [ ! -f .prof/wbuild_xcheck.sh ]; then
    echo "i32f_acc_gate: SKIP - .prof/wbuild_xcheck.sh absent, so the"
    echo "  assembly tiers were NOT run"
    exit 2
fi
if ! bash .prof/wbuild_xcheck.sh i32f_acc_probe > "$O/w.log" 2>&1 || \
   [ ! -x .prof/_wxc/probe.exe ]; then
    echo "i32f_acc_gate: SKIP - no Watcom build; the assembly tiers were"
    echo "  NOT run"
    tail -4 "$O/w.log"; exit 2
fi
.prof/_wxc/probe.exe > "$O/wrun.log" 2>&1
wrc=$?
tail -4 "$O/wrun.log"
[ $wrc -eq 0 ] || { echo "i32f_acc_gate: FAIL (Watcom)"; exit 1; }
WH=$(grep -o "sse hash [0-9A-F]*" "$O/wrun.log" | head -1 | awk '{print $3}')

if [ -z "$GH" ] || [ -z "$WH" ]; then
    echo "i32f_acc_gate: FAIL - a kernel hash is missing, so the two"
    echo "  toolchains were not compared even though both ran"
    exit 1
fi
if [ "$GH" != "$WH" ]; then
    echo "i32f_acc_gate: FAIL - gcc $GH, Watcom $WH"
    exit 1
fi
echo "i32f_acc_gate: PASS (both toolchains, kernel hash $GH)"
exit 0
