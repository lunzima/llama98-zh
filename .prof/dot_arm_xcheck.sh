#!/bin/bash
# lz_dot32_q16/q41/q61_arm_asm against their C tiers, on ARM under QEMU.
#
#   bash .prof/dot_arm_xcheck.sh
#
# WHY THESE THREE. build/kernel_probe_census.sh's ARM half reports them
# as reached by no ARM-cross-building driver. build/arm/parity_gate.sh
# does run --kernel arm-asm over a real model, so they are very probably
# executed there - but the census exists to replace "very probably", the
# Q8_0 sibling already has a direct check, and a kernel whose only
# coverage is an argument is the shape that let lz_gdn1_x4_asm
# accumulate into an uninitialised mm7.
#
# THE MUTATION IS A HALFWORD SELECTION, not a value. src/ops_arm.h says
# each asm tier owes its C tier bit-identity - the same integer products
# in the same lanes - so a wrong sum is not the failure mode these have.
# What is not structural is WHICH halfword of which operand each
# SMLA<x><y> multiplies, so the mutation turns one smlatt into smlatb:
# still a well-formed kernel over the same data, returning a plausible
# number.
#
# Exit: 0 pass, 1 fail, 2 skip (no cross-compiler or no QEMU).
set -u
cd "$(dirname "$0")/.."

FLAGS="-march=armv5te -mtune=arm926ej-s -mfloat-abi=soft -DLZ_NO_FPU=1"
FLAGS="$FLAGS -O2 -std=c99 -Isrc"

# Beside the repo, not in $TMPDIR: on a Windows host Git Bash's /tmp is
# %TEMP% and WSL's is its own, so a path from one is invisible to the
# other - and that surfaces as the mutated build not compiling, which
# reads as "the mutation changed nothing".
W=$(mktemp -d "$(pwd)/../.dotarm-XXXXXX"); trap 'rm -rf "$W"' EXIT

wslpath_of() {
    case "$1" in
        /[a-z]/*) printf '/mnt%s' "$1" ;;
        *)        printf '%s' "$1" ;;
    esac
}

if ! command -v wsl >/dev/null 2>&1 || \
   ! wsl -e bash -lc 'command -v arm-linux-gnueabi-gcc-14 >/dev/null' 2>/dev/null; then
    echo "dot-arm-xcheck: SKIP - no ARM cross-compiler or QEMU"
    exit 2
fi

build_and_run() {   # $1 = tree root holding src/ and .prof/, $2 = tag
    local d
    d=$(wslpath_of "$1")
    wsl -e bash -lc "cd '$d' && arm-linux-gnueabi-gcc-14 $FLAGS \
        -o /tmp/dotarm-$2 .prof/dot_arm_xcheck.c src/ops_arm.c 2>&1 && \
        qemu-arm-static -cpu arm926 -L /usr/arm-linux-gnueabi /tmp/dotarm-$2"
}

OUT=$(build_and_run "$(pwd)" clean 2>&1)
echo "$OUT"
case "$OUT" in
    *"dot-arm-xcheck: PASS"*) ;;
    *) echo "dot-arm-xcheck: FAIL - an asm tier disagrees with its C tier"
       exit 1 ;;
esac

# The mutation, in a copy: this tree is shared and a planted defect must
# not outlive one build.
cp -r src "$W/src" && cp -r .prof "$W/.prof" 2>/dev/null
PY=python; command -v python >/dev/null 2>&1 || PY=python3
"$PY" - "$W/src/ops_arm.c" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8", errors="replace").read()
old = '"smlatt r12,   r4, r8,  r12'
if s.count(old) != 1:
    print("dot-arm-xcheck: FAIL - the mutation site is gone, so the green"
          " above is unproven")
    sys.exit(2)
open(p, "w", encoding="utf-8", newline="").write(
    s.replace(old, '"smlatb r12,   r4, r8,  r12', 1))
PYEOF
[ $? -eq 0 ] || exit 1

MOUT=$(build_and_run "$W" mut 2>&1)
case "$MOUT" in
    *"dot-arm-xcheck: FAIL"*)
        echo "dot-arm-xcheck: mutation caught (one smlatt -> smlatb)"
        echo "dot-arm-xcheck: PASS"
        exit 0 ;;
    *"dot-arm-xcheck: PASS"*)
        echo "dot-arm-xcheck: FAIL - the mutation changed NOTHING, so the"
        echo "  green above is not evidence."
        exit 1 ;;
    *)
        echo "$MOUT"
        echo "dot-arm-xcheck: FAIL - the mutated build produced no verdict,"
        echo "  so the mutation is UNTESTED rather than absent."
        exit 1 ;;
esac
