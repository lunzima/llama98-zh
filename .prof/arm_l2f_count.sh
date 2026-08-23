#!/bin/bash
# Count .prof/arm_l2f_count.c's five arms on ARMv5TE under QEMU.
#
#   wsl -e bash -lc "bash /mnt/e/llama98-zh/.prof/arm_l2f_count.sh"
#
# Same instrument and same three judgements as .prof/arm_prim_count.sh:
# executed instructions via -one-insn-per-tb, the marginal between n and
# 2n so no startup lands in the number, and a control arm the others are
# read against. Read the .c for what each arm is and why the inputs are
# the measured peaks rather than toy values.
set -e
R=${R:-/mnt/e/llama98-zh}
O=${O:-/tmp/l2fc}
CC=${CC:-arm-linux-gnueabi-gcc-14}
Q=${Q:-qemu-arm-static}
F="-O2 -std=c99 -march=armv5te -mtune=arm926ej-s -mfloat-abi=soft"
F="$F -ffp-contract=off"

command -v "$CC" >/dev/null 2>&1 || { echo "arm_l2f_count: SKIP - no $CC"; exit 2; }
command -v "$Q"  >/dev/null 2>&1 || { echo "arm_l2f_count: SKIP - no $Q"; exit 2; }

mkdir -p "$O"; rm -f "$O"/*.elf
for m in 0 1 2 3 4; do
    $CC $F -DL2F_MODE=$m -static "$R/.prof/arm_l2f_count.c" -o "$O/m$m.elf"
done

count() { $Q -one-insn-per-tb -d exec,nochain "$1" "$2" 2>&1 >/dev/null | wc -l; }

for m in 0 1 2 3 4; do
    a=$(count "$O/m$m.elf" 256)
    b=$(count "$O/m$m.elf" 512)
    printf 'mode %s  n256=%-8s n512=%-8s per-iter=%s\n' \
        "$m" "$a" "$b" "$(echo "scale=2; ($b-$a)/256" | bc)"
done
exit 0
