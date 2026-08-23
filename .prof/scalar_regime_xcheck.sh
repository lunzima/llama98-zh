#!/bin/bash
# The scalar transcendentals, all four float regimes, side by side.
#
#   bash .prof/scalar_regime_xcheck.sh
#
# WHAT THIS COVERS THAT THE OTHER PROBES DO NOT. fwht_f32_xcheck and
# matmul_f32_xcheck compare a SIMD body against the C beside it, inside
# ONE binary. .prof/arm_leafgate.c compares the ARM assembly against the
# ARM C, also inside one binary. Both are within-regime. This one holds
# four BUILDS' answers next to each other, which is the only way to see
# the difference this engine actually has: x87 evaluates in 80-bit
# registers where SSE and soft-float evaluate in 32.
#
#   SSE2 / MMX   gcc x86-64, both kernel tiers of one binary
#   x87          Open Watcom 32-bit, all three of its tiers
#   soft-float   arm-linux-gnueabi + qemu, no float unit at all
#
# PC=24 IS INSIDE THE PROBE AND HAS TO BE. Run without it the Watcom arm
# reports exp, rsqrt and softplus as three divergences, and all three are
# the probe rather than the engine - see the note at the top of
# scalar_regime_xcheck.c, which records the numbers so the next person
# recognises them rather than investigating them.
#
# Exit 0 all regimes agree, 1 a real divergence, 2 a toolchain absent
# (named, never silent - a regime that did not run is not a regime that
# agreed).
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/srx}
CC=${CC:-gcc}
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     fwht cpucheck err"
X86_EXTRA="ops_mmx ops_mmx_sse ops_sse2"
ARM_EXTRA="ops_t2_arm ops_arm"

rm -rf "$O"; mkdir -p "$O"
MISSING=""
FAILED=0

# CR IS STRIPPED FROM EVERY ARM, and this is not tidiness. Run from WSL
# the Watcom probe is a Windows PE emitting CRLF while the gcc and ARM
# binaries emit LF, so the three agreed on every digit and the string
# compare reported all three x87 tiers divergent - a FAIL on the host
# where the soft-float arm can actually run, and a PASS on the host where
# it cannot. Two hosts, two verdicts, one set of numbers.
norm() { tr -d '\r'; }

list() { local p="$1"; shift; local f o=""; for f in "$@"; do o="$o $p/$f.c"; done; echo "$o"; }

# ---- gcc / SSE2 + MMX ---------------------------------------------------
if ! $CC -O2 -std=c99 -I"$R/src" -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 \
        -o "$O/x86" "$R/.prof/scalar_regime_xcheck.c" \
        $(list "$R/src" $SRC $X86_EXTRA) -lm > "$O/x86.log" 2>&1; then
    echo "scalar_regime_xcheck: ERROR - the gcc arm did not build"
    tail -5 "$O/x86.log"; exit 2
fi
REF=$("$O/x86" | norm)
echo "-- gcc x86-64 (SSE2) --"; echo "$REF" | sed 's/^/  /'

# ---- Open Watcom / x87 --------------------------------------------------
# Reuses .prof/wbuild_xcheck.sh, which is the generic Watcom probe
# builder - it takes the probe's basename and builds the same source set
# with wcc386. Its own runner prints only the last line, so the three
# tiers are read from the executable it leaves behind.
if bash "$R/.prof/wbuild_xcheck.sh" scalar_regime_xcheck > "$O/wat.log" 2>&1 \
   && [ -x "$R/.prof/_wxc/probe.exe" ]; then
    echo "-- Open Watcom 32-bit (x87) --"
    for T in ref mmx sse2; do
        G=$("$R/.prof/_wxc/probe.exe" "$T" | norm)
        printf '  %-5s\n' "$T"; echo "$G" | sed 's/^/    /'
        [ "$G" = "$REF" ] || {
            echo "FAIL: the x87 build's $T tier differs from gcc/SSE2"
            FAILED=1; }
    done
else
    MISSING="$MISSING x87"
    echo "-- Open Watcom 32-bit (x87) -- SKIP, no C:\\WATCOM"
fi

# ---- ARM / soft-float ---------------------------------------------------
ACC=$(command -v arm-linux-gnueabi-gcc-14 2>/dev/null \
   || command -v arm-linux-gnueabi-gcc 2>/dev/null || true)
if [ -n "$ACC" ] && command -v qemu-arm-static >/dev/null 2>&1; then
    if $ACC -O2 -std=c99 -static -march=armv5te -mtune=arm926ej-s \
            -mfloat-abi=soft -DLZ_NO_FPU=1 -ffp-contract=off -I"$R/src" \
            -o "$O/arm" "$R/.prof/scalar_regime_xcheck.c" \
            $(list "$R/src" $SRC $ARM_EXTRA) -lm > "$O/arm.log" 2>&1; then
        G=$(qemu-arm-static "$O/arm" | norm)
        echo "-- ARM soft-float (no FPU) --"; echo "$G" | sed 's/^/  /'
        [ "$G" = "$REF" ] || {
            echo "FAIL: the soft-float build differs from gcc/SSE2"
            FAILED=1; }
    else
        MISSING="$MISSING soft-float"
        echo "-- ARM soft-float -- SKIP, the cross build failed"
        tail -5 "$O/arm.log"
    fi
else
    MISSING="$MISSING soft-float"
    echo "-- ARM soft-float -- SKIP, no cross toolchain or qemu"
fi

[ $FAILED -eq 0 ] || { echo "scalar_regime_xcheck: FAIL"; exit 1; }
if [ -n "$MISSING" ]; then
    echo "scalar_regime_xcheck: PASS (UNCHECKED REGIMES:$MISSING)"
    exit 2
fi
echo "scalar_regime_xcheck: PASS - all four regimes agree"
exit 0
