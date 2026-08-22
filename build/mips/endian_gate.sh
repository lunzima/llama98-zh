#!/bin/bash
# Big-endian vs little-endian bit-identity gate (iron law 2).
#
#   bash build/mips/endian_gate.sh
#
# Run from WSL or Linux. Cross-builds the engine TWICE from ONE source
# tree - once big-endian, once little-endian - runs both under qemu-user
# on the same models with the same tokens, and compares --dump-logits
# byte for byte.
#
# WHY THIS IS THE ONLY GATE THAT CAN SEE A BYTE-ORDER BUG
#
# Every other target the engine builds for is little-endian: x86 (gcc,
# MinGW, Open Watcom), x86-64, and the ARMv5TE cross-build. A load path
# that assumes little-endian byte order produces the correct answer on
# all of them, so the entire existing gate suite - including the ARM
# bit-identity gate, which is otherwise the strictest thing here - stays
# green with the bug present. Two arms that differ ONLY in endianness are
# what tells "portable" from "little-endian and never tested elsewhere".
#
# WHAT IT ASSERTS
#
#   toolchain     both arms report the SAME gcc version. Debian builds
#                 gcc-14-mips-linux-gnu and gcc-14-mipsel-linux-gnu from
#                 one source package, so this normally holds; when it
#                 does not, a DIFFER cannot be attributed to byte order
#                 and the gate says so instead of blaming the engine.
#   both run      neither arm crashed or refused the model. A big-endian
#                 arm that dies in the loader and a big-endian arm that
#                 produces wrong logits are DIFFERENT failures and must
#                 not print alike - the first is reported as RUNFAIL with
#                 the engine's own message, never as a byte mismatch.
#   bit-identity  BE logits == LE logits, after the BE dump is byte-
#                 swapped into little-endian (see NORMALISATION).
#   liveness      the LE arm's logits are not all-zero and not constant.
#                 Two arms that both produce 4 KB of zeroes compare equal
#                 and prove nothing; this is the "identically declined"
#                 check that build/arm/parity_gate.sh's tier-agreement
#                 assertion plays for ARM.
#
# NORMALISATION: --dump-logits writes host-order f32 (cli_main.c does a
# plain fwrite of the logits array). The big-endian arm therefore emits
# big-endian floats even when it computed exactly the right numbers, so
# comparing the raw files would fail on every model for a reason that is
# not a bug. The BE dump is byte-swapped here, in the gate, by a python
# one-liner that touches nothing the engine ships - deliberately NOT by
# teaching the engine to write little-endian dumps, which would put the
# transformation under test inside the thing being tested.
#
# Exit codes:
#   0  PASS
#   1  FAIL - the arms disagree, or an arm did not run
#   2  SKIP - no MIPS cross toolchain; nothing was checked
set -u

cd "$(dirname "$0")/../.."
ROOT=$(pwd)

MODELS=${MODELS:-"/mnt/e/LLM/models/kunkun98-pilot-20260806 /mnt/e/LLM/models/kunmoe-v2 /mnt/e/LLM/models/_armgate2"}
TOK=${TOK:-1,2,3,4,5,6,7,8,9,10}
W=${W:-/tmp/mipsendian}
KERNELS=${KERNELS:-ref}

FAIL=0
bad() { echo "endian-gate: FAIL $*"; FAIL=1; }

# ---- toolchain ----------------------------------------------------------
# Absent is a SKIP naming what stops being checked, never a silent pass -
# build/arm/parity_gate.sh and build/kernel_matrix_gate.sh set the same
# convention. Unlike ARM this arm has no partial mode: without BOTH
# triples there is no comparison to make at all.
find_cc() {
    local triple=$1 c
    for c in "$triple-gcc-14" "$triple-gcc"; do
        if command -v "$c" >/dev/null 2>&1; then echo "$c"; return 0; fi
        if [ -n "${LZ_MIPS_XTC:-}" ] && [ -x "$LZ_MIPS_XTC/usr/bin/$c" ]; then
            echo "$c"; return 0
        fi
    done
    return 1
}
if [ -n "${LZ_MIPS_XTC:-}" ]; then
    PATH="$LZ_MIPS_XTC/usr/bin:$PATH"
    LD_LIBRARY_PATH="$LZ_MIPS_XTC/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    export PATH LD_LIBRARY_PATH
fi

CC_BE=$(find_cc mips-linux-gnu   || true)
CC_LE=$(find_cc mipsel-linux-gnu || true)
missing=""
[ -n "$CC_BE" ] || missing="$missing gcc-14-mips-linux-gnu(BE)"
[ -n "$CC_LE" ] || missing="$missing gcc-14-mipsel-linux-gnu(LE)"
command -v qemu-mips-static   >/dev/null 2>&1 || missing="$missing qemu-mips-static"
command -v qemu-mipsel-static >/dev/null 2>&1 || missing="$missing qemu-mipsel-static"
if [ -n "$missing" ]; then
    echo "endian-gate: SKIP - missing:$missing"
    echo "  Nothing was checked. Byte-order portability of the model"
    echo "  loader goes UNVERIFIED: no other target in this tree is"
    echo "  big-endian, so no other gate can see it."
    echo "  Install: apt-get install gcc-14-mips-linux-gnu \\"
    echo "             gcc-14-mipsel-linux-gnu qemu-user-static"
    echo "  Without root: apt-get download the packages and dpkg -x them"
    echo "  into a prefix, then set LZ_MIPS_XTC to it."
    exit 2
fi

VER_BE=$($CC_BE -dumpfullversion 2>/dev/null || $CC_BE -dumpversion)
VER_LE=$($CC_LE -dumpfullversion 2>/dev/null || $CC_LE -dumpversion)
echo "-- toolchain --"
echo "  BE $CC_BE  $VER_BE"
echo "  LE $CC_LE  $VER_LE"
if [ "$VER_BE" != "$VER_LE" ]; then
    bad "compiler versions differ ($VER_BE vs $VER_LE) - a byte mismatch"
    echo "     below could be the compilers, not the engine. Install a"
    echo "     matched pair before believing any DIFFER line."
fi

# ---- build both arms ----------------------------------------------------
rm -rf "$W"; mkdir -p "$W"
for a in be le; do
    echo "-- cross-building $a --"
    if ! ARCH=$a OUT="$W/$a" bash "$ROOT/build/mips/build_engine.sh" \
            >"$W/$a.build.log" 2>&1; then
        echo "endian-gate: ERROR $a build failed"
        tail -n 15 "$W/$a.build.log"
        exit 1
    fi
    grep -E '^(linked|objects):' "$W/$a.build.log" | sed 's/^/  /'
done

# Endianness of the produced ELF is checked, not assumed: a triple that
# silently produced two same-endian binaries would make every model
# compare equal and the gate would report PASS for a tree with no
# byte-order handling at all.
be_kind=$(file "$W/be/llama98-mips")
le_kind=$(file "$W/le/llama98-mips")
case "$be_kind" in *MSB*) ;; *) bad "the BE build is not MSB: $be_kind" ;; esac
case "$le_kind" in *LSB*) ;; *) bad "the LE build is not LSB: $le_kind" ;; esac
[ "$FAIL" -eq 0 ] || { echo "endian_gate: FAIL"; exit 1; }

# ---- run and compare ----------------------------------------------------
# swap4 IN OUT - reverse every 4-byte group. Independent of the engine on
# purpose; see NORMALISATION above.
swap4() {
    python3 -c 'import sys
d=open(sys.argv[1],"rb").read()
assert len(d)%4==0, "dump is not a whole number of f32"
open(sys.argv[2],"wb").write(b"".join(d[i:i+4][::-1] for i in range(0,len(d),4)))' "$1" "$2"
}

echo "-- models --"
ran_any=0
for m in $MODELS; do
    name=$(basename "$m")
    if [ ! -d "$m" ]; then
        echo "  SKIP $name (no such directory)"
        continue
    fi
    for k in $KERNELS; do
        rc_be=0; rc_le=0
        qemu-mips-static   "$W/be/llama98-mips" "$m" --kernel "$k" \
            --tokens "$TOK" --dump-logits "$W/$name.$k.be.raw" \
            >"$W/$name.$k.be.log" 2>&1 || rc_be=$?
        qemu-mipsel-static "$W/le/llama98-mips" "$m" --kernel "$k" \
            --tokens "$TOK" --dump-logits "$W/$name.$k.le.bin" \
            >"$W/$name.$k.le.log" 2>&1 || rc_le=$?

        # A crash and a wrong number are different failures. Report the
        # arm's own message rather than letting a missing dump file turn
        # into a byte mismatch.
        if [ "$rc_le" -ne 0 ] || [ ! -s "$W/$name.$k.le.bin" ]; then
            bad "RUNFAIL le $name/$k (rc=$rc_le)"
            tail -n 4 "$W/$name.$k.le.log" | sed 's/^/       /'
            continue
        fi
        if [ "$rc_be" -ne 0 ] || [ ! -s "$W/$name.$k.be.raw" ]; then
            bad "RUNFAIL be $name/$k (rc=$rc_be)"
            tail -n 4 "$W/$name.$k.be.log" | sed 's/^/       /'
            continue
        fi
        ran_any=1

        swap4 "$W/$name.$k.be.raw" "$W/$name.$k.be.bin"

        # Liveness: an all-constant dump makes any comparison vacuous.
        live=$(python3 -c 'import sys,struct
d=open(sys.argv[1],"rb").read()
v=struct.unpack("<%df"%(len(d)//4),d)
print("dead" if len(set(v))<=1 else "live")' "$W/$name.$k.le.bin")
        if [ "$live" != "live" ]; then
            bad "$name/$k: the LE dump is constant - comparing it to" \
                "anything proves nothing"
            continue
        fi

        if cmp -s "$W/$name.$k.be.bin" "$W/$name.$k.le.bin"; then
            printf '  IDENTICAL  %-26s %-6s %s bytes  md5=%s\n' "$name" "$k" \
                "$(wc -c < "$W/$name.$k.le.bin")" \
                "$(md5sum < "$W/$name.$k.le.bin" | cut -c1-8)"
        else
            n=$(cmp -l "$W/$name.$k.be.bin" "$W/$name.$k.le.bin" 2>/dev/null | wc -l)
            tot=$(wc -c < "$W/$name.$k.le.bin")
            printf '  DIFFER     %-26s %-6s %s/%s bytes differ\n' \
                "$name" "$k" "$n" "$tot"
            FAIL=1
        fi
    done
done

if [ "$ran_any" -eq 0 ]; then
    echo "endian-gate: SKIP - no model directory was found"
    echo "  Set MODELS=... to point at some. Nothing was checked."
    exit 2
fi

if [ "$FAIL" -ne 0 ]; then
    echo "endian_gate: FAIL"
    exit 1
fi
echo "endian_gate: PASS"
exit 0
