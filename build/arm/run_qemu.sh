#!/bin/bash
# Run the ARMv5TE build under QEMU user-mode emulation.
#
#   run_qemu.sh <model-dir> [llama98 args...]
#
# -cpu arm926 is not decoration. QEMU defaults to a much later core, and
# under the default an ARMv6 or ARMv7 instruction that crept into the
# build would execute happily - the emulator would hide exactly the
# portability bug this environment exists to find. arm926 makes the
# emulator refuse anything the real AK7802 would refuse.
#
# *** WHAT THIS CANNOT TELL YOU ***
#
# TIMING. QEMU user-mode translates instructions and models no cache, no
# bus, no memory latency and no pipeline. A wall-clock number from it is
# a number about this x86 host, not about a 248 MHz ARM926 with 16-bit
# DDR. Iron law 3 already says host nanoseconds do not transfer; here
# they are not even host nanoseconds of the right work.
#
# What DOES transfer is the static instruction count, which is what iron
# law 3 asks to compare in the first place. `-d in_asm` or a TCG plugin
# gives it. The one number that matters most for kunkun-CE - the real
# cost of a ternary kernel on this core - needs either hardware or a
# cycle-accurate simulator, and neither is this.
set -u

MODEL=${1:-}
if [ -z "$MODEL" ]; then
    echo "usage: run_qemu.sh <model-dir> [args...]" >&2
    exit 2
fi
shift

BIN=${BIN:-/tmp/armbuild/llama98-arm}
SYSROOT=${SYSROOT:-/usr/arm-linux-gnueabi}

if [ ! -x "$BIN" ]; then
    echo "$BIN missing - run build/arm/build_engine.sh first" >&2
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "$MODEL is not a directory" >&2
    exit 1
fi

exec qemu-arm-static -cpu arm926 -L "$SYSROOT" "$BIN" "$MODEL" "$@"
