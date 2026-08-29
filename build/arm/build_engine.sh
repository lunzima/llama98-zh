#!/bin/bash
# Cross-build the llama98 engine for ARMv5TE (the kunkun-CE target).
#
# TARGET: Anyka AK7802, ARM926EJ-S, 248/266 MHz, no FPU, no SIMD, 64 MB
# DDR, running Linux. This is a THIRD toolchain alongside gcc/MinGW and
# Open Watcom, and unlike those two it is not a Windows target at all.
#
# WHY EVERY FLAG IS THERE:
#
#   -march=armv5te   NOT the toolchain default. arm-linux-gnueabi-gcc
#                    defaults to armv5t - __ARM_ARCH is 5 either way,
#                    but __ARM_FEATURE_DSP is ABSENT without the "te".
#                    The E is the entire reason this ISA is interesting
#                    for a matmul: SMLAxy (16x16+32, near single-cycle
#                    on ARM9E), LDRD/STRD, and PLD. Omitting it silently
#                    discards all three.
#   -mtune=arm926ej-s  scheduling for the actual core.
#   -mfloat-abi=soft   the AK7802 has no FPU. VERIFIED BY DISASSEMBLY,
#                    not by the macro: __VFP_FP__ is defined even under
#                    soft-float (it names the FP FORMAT, not the unit),
#                    while the emitted code contains zero VFP
#                    instructions and calls __aeabi_fmul.
#   -DLZ_NO_FPU=1     same fact, told to the engine: lz_soft_float_target()
#                    then resolves every --*-write auto to fixed without a
#                    runtime CPUID, so the slow __aeabi_* soft-float path is
#                    avoided on the operators that have a fixed tier.
#   -ffp-contract=off  no multiply-add contraction, or
#                    this build stops being bit-identical to the others.
#
# Run under emulation with:
#   qemu-arm-static -cpu arm926 -L /usr/arm-linux-gnueabi ./llama98-arm
#
# QEMU GIVES CORRECTNESS, NOT CYCLES. It translates instructions and
# models no cache, no bus and no memory latency, so a wall-clock number
# from it means nothing for the AK7802. What IS transferable is the
# static instruction count - the only quantity that carries across
# machines anyway - and qemu can produce that with a TCG plugin.
set -u

# Unsuffixed FIRST but not only: Debian ships arm-linux-gnueabi-gcc-14
# with no unsuffixed alias, and this script's failure mode there is 24
# compile FAILs in a row - which reads as "the tree does not build for
# ARM", not as "the compiler is called something else".
# kernel_matrix_gate.sh already probes both names; this is the same
# probe for anyone running the script directly.
if [ -z "${CC:-}" ]; then
    for c in arm-linux-gnueabi-gcc arm-linux-gnueabi-gcc-14; do
        command -v "$c" >/dev/null 2>&1 && CC=$c && break
    done
fi
CC=${CC:-arm-linux-gnueabi-gcc}
OUT=${OUT:-/tmp/armbuild}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

FLAGS="-O2 -std=c99 -Wall -Wextra"
FLAGS="$FLAGS -march=armv5te -mtune=arm926ej-s -mfloat-abi=soft -DLZ_NO_FPU=1"
# EXTRA_FLAGS, an env override for one-off builds: the fast-float parity
# arm passes -DLZ_SOFTFP_FAST=1 (a diagnostic build that must NOT be the
# default). Appended to every TU, which is what the RTZ tails in
# ops_arm_prim.h need - they are selected per-TU by the same macro.
FLAGS="$FLAGS ${EXTRA_FLAGS:-}"
# -fomit-frame-pointer: gcc's ARM default keeps a frame pointer, which is
# pure overhead in a release build and bloats the soft-float module -- the
# __aeabi_f* fast paths are compared against libgcc's naked (frame-free)
# asm, so they must be compiled frame-free too (docs/arm-asm-audit.md's
# static-instruction-count metric).
FLAGS="$FLAGS -ffp-contract=off -fomit-frame-pointer -I$ROOT/src"

# Asked for, not restated - see the Makefile's print-% comment. This
# copy happened to be correct, which is exactly why it is worth
# deleting: the identical comment sat above the x87 script's copy, which
# was nine files short, so the sentence proved nothing either way.
ENG=$(make -s -C "$ROOT" print-ENG)
if [ -z "$ENG" ]; then
    echo "arm: cannot read ENG from $ROOT/Makefile (make -s print-ENG)"
    exit 2
fi
# lz_softfp.c is the ARM-only soft-float replacement for libgcc's
# __aeabi_f* arithmetic, compare and convert routines. NOT the EABI
# divmod ones, which this line advertised for one change too long: they
# were removed after measurement (src/lz_softfp.c's closing comment has
# the numbers - libgcc's CLZ-normalised divide runs 5x-34x fewer
# instructions), and src/ now contains no __aeabi_*div* at all
# (its body is #if __arm__ && __GNUC__, so it would compile to an empty
# TU on the x86 gcc and Watcom builds - that is why it is not in the
# shared ENG list). The ARM cross-build links it explicitly, so its
# object satisfies the engine's __aeabi_* references before libgcc's
# archive members are pulled.
ENG="$ENG lz_softfp"

mkdir -p "$OUT"
rm -f "$OUT"/*.o "$OUT"/*.log

fail=0
warn=0
for f in $ENG; do
    if $CC $FLAGS -c "$ROOT/src/$f.c" -o "$OUT/$f.o" 2>"$OUT/$f.log"; then
        n=$(grep -c 'warning:' "$OUT/$f.log" || true)
        warn=$((warn + n))
        printf '  %-14s ok    %s warnings\n' "$f" "$n"
    else
        printf '  %-14s FAIL\n' "$f"
        grep -E 'error:' "$OUT/$f.log" | head -4 | sed 's/^/      /'
        fail=$((fail + 1))
    fi
done

echo
echo "objects: $(ls -1 "$OUT"/*.o 2>/dev/null | wc -l)  failed: $fail  warnings: $warn"
[ "$fail" -eq 0 ] || exit 1

# Link the CLI. cli_main.c's -i loop runs on the shared LZSession core
# in common/, a front-end helper deliberately outside $(ENG) - so this
# lists it explicitly with -Icommon, exactly as the Makefile's native
# target and the Watcom CLI/DOS builds do.
COM=$(make -s -C "$ROOT" print-CLI_COM)
if [ -f "$ROOT/src/cli_main.c" ]; then
    COMSRC=""
    for c in $COM; do
        [ -f "$ROOT/common/$c.c" ] && COMSRC="$COMSRC $ROOT/common/$c.c"
    done
    if $CC $FLAGS -I"$ROOT/common" -o "$OUT/llama98-arm" \
         "$ROOT/src/cli_main.c" "$ROOT/src/cli_argv.c" \
         "$ROOT/src/cli_attr.c" $COMSRC "$OUT"/*.o -lm \
         2>"$OUT/link.log"; then
        echo "linked: $OUT/llama98-arm"
        file "$OUT/llama98-arm" 2>/dev/null | sed 's/^/  /'
    else
        echo "link FAILED:"
        grep -E 'error|undefined' "$OUT/link.log" | head -8 | sed 's/^/  /'
        exit 1
    fi
fi
