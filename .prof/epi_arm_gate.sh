#!/bin/bash
# Build and run the epi MAC oracle against the SHIPPED ARMv5TE kernel.
#
#   bash .prof/epi_arm_gate.sh [leaf] [seed] [divisor]
#
# The three arguments are arm_mutgate.sh's calling convention, kept so
# that script can drive this one the same way it drives arm_leafgate.sh.
# `leaf` is ignored (there is one leaf here); seed 0 means the fixed
# seed; the divisor shortens the random sweep for a mutation run.
#
# What this proves and what it does not: the oracle compares
# lz_epi_mac_i16_arm_asm against three independent scalar derivations,
# so a PASS is about the VALUE. It says nothing about whether the
# engine's dispatch ever reaches the kernel - that is what the
# [epi: N calls] counter on the --debug line is for, because the C
# fallback returns the same bytes and no value comparison can tell the
# two apart.
set -e
R=${R:-/mnt/e/llama98-zh}
O=${O:-/tmp/epiarm}
CC=${CC:-arm-linux-gnueabi-gcc-14}
Q=${Q:-qemu-arm-static}

# WSL ONLY, said before anything runs - arm_leafgate.sh's reason: the
# cross compiler and qemu are there and not in Git Bash, where $R does
# not exist either, and lines already printed look like results.
if [ ! -d "$R/src" ]; then
    echo "epi_arm_gate: FAIL - \$R=$R has no src/. Run under WSL:"
    echo "  wsl bash -lc \"cd /mnt/e/llama98-zh && bash .prof/epi_arm_gate.sh\""
    exit 2
fi

# The engine's own ARM flags (build/arm/build_engine.sh). -march=armv5te
# is the one that cannot be dropped: without the "te" the assembler
# rejects SMLAL's ARMv5TE encoding context and the kernel is not what
# ships.
F="-O2 -std=c99 -Wall -Wextra -march=armv5te -mtune=arm926ej-s"
F="$F -mfloat-abi=soft -DLZ_NO_FPU=1 -ffp-contract=off"

mkdir -p "$O"
rm -f "$O/epi_arm.elf"
$CC $F -I"$R/src" -DEPI_ARM -static \
    "$R/.prof/epi_mac_oracle.c" "$R/src/ops_arm.c" -o "$O/epi_arm.elf"

exec $Q -cpu arm926 "$O/epi_arm.elf" "${1:-epi}" "${2:-0}" "${3:-1}"
