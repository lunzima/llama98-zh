#!/bin/bash
# What a float CHAIN costs on ARMv5TE, against what its parts cost.
#
#   wsl -e bash -lc "bash /mnt/e/llama98-zh/.prof/arm_chain_count.sh"
#
# .prof/arm_prim_count.sh answers "what does operator X cost". This
# answers the one question docs/arm-asm-audit.md's G2 rows turn on and
# that one cannot: when two float operations share their operands, is
# the shared unpack+align worth a hand-written cell, or does libgcc's
# per-call fast path already beat what a chain would cost?
#
# M5 is why this is measured and not argued. A hand-written single float
# multiply was scoped at 41 instructions against __aeabi_fmul's 24 -
# libgcc folds normalisation into an sbc borrow and the rounding carry
# into the exponent add, and an estimate that has not read that code is
# not evidence.
#
# Modes (src in .prof/arm_prim_count.c):
#   c0   the loop with no float operation - the control
#   16   one bare multiply          17  one bare add
#   19   the butterfly: a+b and a-b from one pair (fwhtf armA)
#   20   the MAC: acc + w*x                       (f32mm armA)
#   21   mode 19 through lz_faddsub_arm_asm, fallback included - the
#        shipped cost of the fwhtf armA cell against 19, its control
#
# WHAT THE NUMBERS DECIDE. 19 minus c0 against 2*(17 - c0) says how much
# of the butterfly is already shared by the compiler; 20 minus c0
# against (16 - c0) + (17 - c0) says the same for the MAC. A hand-written
# cell has to beat the measured chain, not the sum of the parts.
set -e
R=${R:-/mnt/e/llama98-zh}
O=${O:-/tmp/chainc}
CC=${CC:-arm-linux-gnueabi-gcc-14}
Q=${Q:-qemu-arm-static}
F="-O2 -std=c99 -march=armv5te -mtune=arm926ej-s -mfloat-abi=soft"
F="$F -DLZ_NO_FPU=1 -ffp-contract=off -ffunction-sections -fdata-sections"

command -v "$CC" >/dev/null 2>&1 || { echo "arm_chain_count: SKIP - no $CC"; exit 2; }
command -v "$Q"  >/dev/null 2>&1 || { echo "arm_chain_count: SKIP - no $Q"; exit 2; }

mkdir -p "$O"; rm -f "$O"/*.o "$O"/*.elf
OBJ=""
for f in ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_arm \
         ops_t2_scalar ops_arm fwht cpucheck err; do
    $CC $F -I"$R/src" -c "$R/src/$f.c" -o "$O/$f.o"
    OBJ="$OBJ $O/$f.o"
done
for m in 0 16 17 19 20 21 22 23 24 25 26 27 28; do
    $CC $F -I"$R/src" -I"$R/.prof" -DPRIM_MODE=$m -DPRIM_TIER=1 \
        -DPRIM_INTARG=0 -static -Wl,--gc-sections \
        "$R/.prof/arm_prim_count.c" $OBJ -lm -o "$O/m$m.elf"
done

count() { $Q -one-insn-per-tb -d exec,nochain "$1" "$2" 2>&1 >/dev/null | wc -l; }

# Linearity is the judgement that makes a marginal number mean anything:
# X(512)-X(256) must equal X(256)-X(0), or the arm has no per-call cost
# and the difference is measuring startup.
for m in 0 16 17 19 20 21 22 23 24 25 26 27 28; do
    z=$(count "$O/m$m.elf" 0)
    a=$(count "$O/m$m.elf" 256)
    b=$(count "$O/m$m.elf" 512)
    d1=$((a - z)); d2=$((b - a))
    lin=OK; [ "$d1" = "$d2" ] || lin="NONLINEAR($d1 vs $d2)"
    printf 'mode %-3s n0=%-8s n256=%-8s n512=%-8s per-iter=%s  %s\n' \
        "$m" "$z" "$a" "$b" "$(echo "scale=2; $d2/256" | bc)" "$lin"
done
exit 0
