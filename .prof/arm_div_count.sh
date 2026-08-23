#!/bin/bash
# Count .prof/arm_div_count.c's five arms on ARMv5TE under QEMU.
#
#   wsl -e bash -lc "bash /mnt/e/llama98-zh/.prof/arm_div_count.sh"
#
# The f32 census counts a divide as ONE operation, exactly like a
# multiply. On a soft-float target that is not close to true, and every
# "reduce the divides" decision in this tree has been made without the
# number. Same instrument and judgements as .prof/arm_prim_count.sh:
# executed instructions, the marginal between n and 2n, a control arm.
#
# MEASURED (control 6.00 subtracted):
#   multiply                 30    __aeabi_fmul
#   divide                  117    __aeabi_fdiv          3.9x a multiply
#   hoisted reciprocal       30    one fdiv outside the loop, mul inside
#   divide by 256.0f         22    strength-reduced to a multiply, and
#                                  cheaper than a general one because a
#                                  power-of-two operand skips normalising
#
# So a divide costs what four multiplies do, and hoisting the reciprocal
# out of a loop turns 117 per element into 30. That is the whole reason
# to know this number before ranking anything by the census, which shows
# a divide and a multiply as the same 1.
#
# The /256.0f arm also answers a question about ops.h rule 6 item 1
# ("write the multiply explicitly, gcc leaves the divide"): it does NOT
# contradict it. That rule is about dividing by 32767, which is not a
# power of two and therefore cannot be strength-reduced exactly - gcc is
# right to leave it. A power-of-two divisor is reduced by everyone.
set -e
R=${R:-/mnt/e/llama98-zh}
O=${O:-/tmp/divc}
CC=${CC:-arm-linux-gnueabi-gcc-14}
Q=${Q:-qemu-arm-static}
F="-O2 -std=c99 -march=armv5te -mtune=arm926ej-s -mfloat-abi=soft"
F="$F -ffp-contract=off"

command -v "$CC" >/dev/null 2>&1 || { echo "arm_div_count: SKIP - no $CC"; exit 2; }
command -v "$Q"  >/dev/null 2>&1 || { echo "arm_div_count: SKIP - no $Q"; exit 2; }

mkdir -p "$O"; rm -f "$O"/*.elf
for m in 0 1 2 3 4; do
    $CC $F -DDIV_MODE=$m -static "$R/.prof/arm_div_count.c" -o "$O/m$m.elf"
done

count() { $Q -one-insn-per-tb -d exec,nochain "$1" "$2" 2>&1 >/dev/null | wc -l; }

for m in 0 1 2 3 4; do
    a=$(count "$O/m$m.elf" 256)
    b=$(count "$O/m$m.elf" 512)
    printf 'mode %s  n256=%-8s n512=%-8s per-iter=%s\n' \
        "$m" "$a" "$b" "$(echo "scale=2; ($b-$a)/256" | bc)"
done
exit 0
