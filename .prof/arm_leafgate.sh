#!/bin/bash
# Build and run the ARM operator-primitive bit-identity gate under QEMU.
# Run from WSL (the cross toolchain and qemu-arm-static live there).
#
#   bash .prof/arm_leafgate.sh [i32f|q8r|exp|all] [seed]
set -e
R=${R:-/mnt/e/llama98-zh}
O=${O:-/tmp/lgate}
CC=${CC:-arm-linux-gnueabi-gcc-14}
Q=${Q:-qemu-arm-static}

# WSL ONLY, said before anything runs. The ARM cross compiler and qemu
# are there and not in Git Bash, where $R does not exist either - and
# "command not found" partway through a comparison is a worse failure
# than refusing at the top, because the lines already printed look like
# results. Same check as arm_mutgate.sh, for the same reason.
if [ ! -d "$R/src" ]; then
    echo "arm_leafgate: FAIL - \$R=$R has no src/. Run under WSL:"
    echo "  wsl bash -lc \"cd /mnt/e/llama98-zh && bash .prof/arm_leafgate.sh\""
    exit 2
fi
F="-O2 -std=c99 -march=armv5te -mtune=arm926ej-s -mfloat-abi=soft"
F="$F -DLZ_NO_FPU=1 -ffp-contract=off -ffunction-sections -fdata-sections"

mkdir -p "$O"
OBJ=""
for f in ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_arm \
         ops_t2_scalar ops_arm fwht cpucheck err; do
    $CC $F -I"$R/src" -c "$R/src/$f.c" -o "$O/$f.o"
    OBJ="$OBJ $O/$f.o"
done
$CC $F -I"$R/src" -static -Wl,--gc-sections \
    "$R/.prof/arm_leafgate.c" $OBJ -lm -o "$O/leafgate.elf"

exec $Q -cpu arm926 "$O/leafgate.elf" "${1:-all}" "${2:-0x9E3779B9}" "${3:-1}"
