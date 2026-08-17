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
#   -ffp-contract=off  iron law 2(d): no multiply-add contraction, or
#                    this build stops being bit-identical to the others.
#
# Run under emulation with:
#   qemu-arm-static -cpu arm926 -L /usr/arm-linux-gnueabi ./llama98-arm
#
# QEMU GIVES CORRECTNESS, NOT CYCLES. It translates instructions and
# models no cache, no bus and no memory latency, so a wall-clock number
# from it means nothing for the AK7802. What IS transferable is the
# static instruction count - which is what iron law 3 says to compare
# anyway - and qemu can produce that with a TCG plugin.
set -u

CC=${CC:-arm-linux-gnueabi-gcc}
OUT=${OUT:-/tmp/armbuild}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

FLAGS="-O2 -std=c99 -Wall -Wextra"
FLAGS="$FLAGS -march=armv5te -mtune=arm926ej-s -mfloat-abi=soft"
FLAGS="$FLAGS -ffp-contract=off -I$ROOT/src"

# Same list as the Makefile's ENG, and it has to stay the same list -
# see the Makefile note about source lists living in exactly one place.
ENG="err compat lfn json safetensors model ops forward sampler generate
     tokenizer unicode chat gbk cpucheck"

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
COM="stream settings command savechat mru chatfile session"
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
