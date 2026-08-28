#!/bin/bash
# Drive build/vmax_vscale_probe.c on BOTH toolchains.
#
# WHY A DRIVER AND NOT A HAND RUN. The probe was written, built with
# wcc386 once by hand, and passed; build/kernel_probe_census.sh then
# still listed lz_vmax_sse_asm and lz_vscale_sse_asm as reached only by
# a gcc-built probe, and it was right to. gcc compiles no #pragma aux
# body, so the Watcom arm of a check exists only while something runs
# it - a hand run covers the kernel for as long as the shell scrollback
# lasts.
#
# That distinction has a body count. lz_gdn1_x4_asm accumulated into an
# uninitialised mm7 while two probes drove lz_gdn_step straight through
# it: both drivers build with $CC, neither has a Watcom arm, and the
# census counted the kernel as reached.
#
# Exit: 0 pass, 1 fail, 2 skip (no Watcom - the gcc half still runs and
# must pass, but it exercises the intrinsics twin, not the assembly).
set -u
cd "$(dirname "$0")/.."

CC=${CC:-gcc}
O=${O:-/tmp/normsse}
rm -rf "$O"; mkdir -p "$O"

echo "-- gcc: the intrinsics twins --"
if ! $CC -O2 -std=c99 -msse -msse2 -ffp-contract=off \
        -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -Isrc -o "$O/g.exe" \
        build/vmax_vscale_probe.c build/norm_sse_probe_stubs.c -lm \
        > "$O/gbuild.log" 2>&1; then
    echo "norm_sse_parity_gate: FAIL - the gcc probe did not build"
    tail -8 "$O/gbuild.log"; exit 1
fi
"$O/g.exe" > "$O/g.log" 2>&1
grc=$?
tail -3 "$O/g.log"
[ $grc -eq 0 ] || { echo "norm_sse_parity_gate: FAIL (gcc)"; exit 1; }
GH=$(grep -o "kernel hash [0-9A-F]*" "$O/g.log" | head -1 | awk '{print $3}')

echo "-- Open Watcom: the #pragma aux twins --"
if [ ! -f .prof/wbuild_xcheck.sh ]; then
    echo "norm_sse_parity_gate: SKIP - .prof/wbuild_xcheck.sh absent, so"
    echo "  the assembly twins were NOT compared"
    exit 2
fi
if ! bash .prof/wbuild_xcheck.sh vmax_vscale_probe > "$O/w.log" 2>&1 || \
   [ ! -x .prof/_wxc/probe.exe ]; then
    echo "norm_sse_parity_gate: SKIP - no Watcom build; the assembly"
    echo "  twins were NOT compared"
    tail -4 "$O/w.log"; exit 2
fi
.prof/_wxc/probe.exe > "$O/wrun.log" 2>&1
wrc=$?
tail -3 "$O/wrun.log"
[ $wrc -eq 0 ] || { echo "norm_sse_parity_gate: FAIL (Watcom)"; exit 1; }
WH=$(grep -o "kernel hash [0-9A-F]*" "$O/wrun.log" | head -1 | awk '{print $3}')

# THE CROSS-TOOLCHAIN COMPARISON, which neither arm makes on its own:
# each build checks its own kernel against its own reference, and only
# the hash relates the two halves.
if [ -z "$GH" ] || [ -z "$WH" ]; then
    echo "norm_sse_parity_gate: FAIL - a kernel hash is missing, so the"
    echo "  two toolchains were not compared even though both ran"
    exit 1
fi
if [ "$GH" != "$WH" ]; then
    echo "norm_sse_parity_gate: FAIL - gcc $GH, Watcom $WH"
    exit 1
fi
echo "norm_sse_parity_gate: PASS (both toolchains, kernel hash $GH)"
exit 0
