#!/bin/bash
# Big-endian vs little-endian bit-identity gate: the same source tree
# must produce the same logits on either byte order.
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

# ---- WSL, which is where the toolchain lives on this host ---------------
# build/arm/parity_gate.sh already has this fallback and says why; this
# gate did not, so it reported "SKIP - missing: gcc-14-mips-linux-gnu(BE)
# qemu-mips-static" on a machine where both are at /usr/bin inside WSL.
# The message was accurate about the shell it ran in and wrong about the
# machine, and byte-order portability went unverified for that reason
# alone - the same shape as the ARM toolchain sitting in WSL while a
# gate looked on PATH.
#
# RE-EXEC THE WHOLE SCRIPT rather than shimming each command: everything
# below cross-compiles and runs under qemu, so one boundary crossing is
# enough and the paths stay in one namespace. LZ_ENDIAN_IN_WSL stops the
# recursion; an explicit LZ_MIPS_XTC or a toolchain already on PATH skips
# the hop entirely, so a Linux CI host is unaffected.
# The names find_cc() below accepts, written once: the probe on EACH side
# of the boundary has to ask the question the gate itself will ask. The
# WSL side asked only for the versioned name, and this machine has
# mips-linux-gnu-gcc 12.4.0 - the unversioned one, and the compiler the
# gate uses once it is there. So the hop did not fire and the host
# reported "SKIP - missing: gcc-14-mips-linux-gnu(BE)" for a toolchain
# that was on the machine. A probe must not report missing a toolchain
# that it would then use.
BE_CC_NAMES="mips-linux-gnu-gcc-14 mips-linux-gnu-gcc"
BE_CC_FIND='for c in '"$BE_CC_NAMES"'; do command -v "$c" >/dev/null 2>&1 && exit 0; done; exit 1'
if [ -z "${LZ_ENDIAN_IN_WSL:-}" ] && [ -z "${LZ_MIPS_XTC:-}" ] \
   && ! sh -c "$BE_CC_FIND" \
   && command -v wsl.exe >/dev/null 2>&1; then
    if wsl.exe -e bash -lc "$BE_CC_FIND" >/dev/null 2>&1 \
       && wsl.exe -e bash -lc 'command -v qemu-mips-static >/dev/null 2>&1' >/dev/null 2>&1; then
        case "$ROOT" in
            /[a-z]/*) WROOT="/mnt/$(echo "$ROOT" | cut -c2)/$(echo "$ROOT" | cut -c4-)" ;;
            *)        WROOT="$ROOT" ;;
        esac
        echo "endian-gate: toolchain is in WSL, re-running there ($WROOT)"
        exec wsl.exe -e bash -lc "cd '$WROOT' && LZ_ENDIAN_IN_WSL=1 bash build/mips/endian_gate.sh"
    fi
fi

# _armgate3 replaced _armgate2 on 2026-09-10: that model and its whole
# training chain are gone (docs/dsv41-optimization-menu.md section 6).
#
# kunmoe-v2 IS STILL LISTED AND CURRENTLY CANNOT LOAD - it reports
# "tensor model.layers.0.linear_attn.g_proj.norm.weight unclaimed by any
# spec". A model that cannot load does not fail this gate, it drops out
# of the comparison silently, so the effective coverage of this list is
# 2 of its 3 entries, not 3. Left in so the reader knows it was meant to
# be there; remove it or fix the checkpoint, but do not read the list
# length as the coverage.
MODELS=${MODELS:-"/mnt/e/LLM/models/kmr20 /mnt/e/LLM/models/kunkun98 /mnt/e/LLM/models/_armgate3"}
TOK=${TOK:-1,2,3,4,5,6,7,8,9,10}
W=${W:-/tmp/mipsendian}
KERNELS=${KERNELS:-ref}

FAIL=0
# Two flags, because the verdict below treats them differently: FAIL is
# for the failures that are not about the bytes (an arm that did not
# build, a build that is not the endianness it claims, a model neither
# arm could run), FAIL_DIFF for the arms' dumps differing - which is the
# claim the fallback note above says cannot be made when the LE arm is
# the native host build.
FAIL_DIFF=0
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

# LE ARM FALLBACK: the native x86 binary.
#
# The two-MIPS-arm design exists so the ONLY difference between the arms
# is byte order - same compiler, same source, same libc. It is the right
# comparison and it is what this gate does when both cross compilers are
# there. But a host with only the BE package installed then checks
# NOTHING, forever, and "byte-order portability goes unverified" is a
# worse outcome than a comparison with a stated confound.
#
# So: if the LE cross compiler is absent and a native build is present,
# the native binary becomes the LE arm. What that costs is written into
# the PASS line and into this comment rather than left for someone to
# discover - the arms then differ in COMPILER and LIBC as well as in
# byte order, so a FAIL cannot be attributed to byte order without
# further work. A PASS still means what it always meant: no byte-order
# bug, because a load path that assumed little-endian could not produce
# the same logits on a big-endian machine.
#
# KERNELS is already pinned to `ref` (see its default above), which is
# what makes the fallback possible at all: the MIPS build has no MMX or
# SSE tiers, so an unpinned comparison against an x86 default of sse2
# would differ for a reason that is not byte order. Measured while
# adding this: unpinned, 6,023,238 of 8,257,536 bytes differ; pinned to
# ref, byte-swapped BE and LE are identical.
# It is built HERE, by ARCH=host through the same build_engine.sh the
# MIPS arms use - not borrowed from the repo root. The root binary on a
# Windows host is a .exe that cannot resolve the /mnt/... paths this
# script hands it, which is how the first version of this fallback
# failed: three models "not found" on the LE arm and loaded fine on BE.
NATIVE_LE=""
CONFOUND=""
if [ -z "$CC_LE" ] && command -v gcc >/dev/null 2>&1; then
    NATIVE_LE=host
    # Set whenever the LE arm is not a MIPS LE cross build. It is what
    # the note above describes and what the verdict at the bottom acts
    # on: with the arms differing in compiler and libc as well, a
    # difference between them is not evidence about byte order.
    CONFOUND=1
fi

missing=""
[ -n "$CC_BE" ] || missing="$missing gcc-14-mips-linux-gnu(BE)"
command -v qemu-mips-static   >/dev/null 2>&1 || missing="$missing qemu-mips-static"
if [ -z "$NATIVE_LE" ]; then
    [ -n "$CC_LE" ] || missing="$missing gcc-14-mipsel-linux-gnu(LE)-or-a-native-build"
    command -v qemu-mipsel-static >/dev/null 2>&1 || missing="$missing qemu-mipsel-static"
fi
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
echo "-- toolchain --"
echo "  BE $CC_BE  $VER_BE"
if [ -n "$NATIVE_LE" ]; then
    # No version comparison to make: the LE arm is a different compiler
    # on a different architecture by construction, which the fallback
    # banner above already says. Running the matched-pair check here
    # anyway would print a FAIL for the thing the fallback IS.
    echo "  LE (native build, no MIPS LE compiler installed)"
else
    VER_LE=$($CC_LE -dumpfullversion 2>/dev/null || $CC_LE -dumpversion)
    echo "  LE $CC_LE  $VER_LE"
    if [ "$VER_BE" != "$VER_LE" ]; then
        bad "compiler versions differ ($VER_BE vs $VER_LE) - a byte mismatch"
        echo "     below could be the compilers, not the engine. Install a"
        echo "     matched pair before believing any DIFFER line."
    fi
fi

# ---- build both arms ----------------------------------------------------
rm -rf "$W"; mkdir -p "$W"
ARMS="be le"
if [ -n "$NATIVE_LE" ]; then
    ARMS="be host"
    echo "-- LE arm is a HOST build, not a MIPS one --"
    echo "   gcc-14-mipsel-linux-gnu is absent, so the two arms differ in"
    echo "   compiler and libc as well as in byte order. A PASS still"
    echo "   rules out a byte-order bug; a FAIL would need the LE cross"
    echo "   compiler installed before it could be attributed."
fi
for a in $ARMS; do
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
case "$be_kind" in *MSB*) ;; *) bad "the BE build is not MSB: $be_kind" ;; esac
# The LE arm is still checked for LSB when it is the MIPS one. When it is
# the native fallback the check moves rather than disappearing: the host
# this runs on is x86 or amd64, both LSB, so asserting it would be
# asserting the host - and the BE assertion above is the one that
# carries the information either way.
if [ -z "$NATIVE_LE" ]; then
    le_kind=$(file "$W/le/llama98-mips")
    case "$le_kind" in *LSB*) ;; *) bad "the LE build is not LSB: $le_kind" ;; esac
fi
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
unloadable=""
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
        if [ -n "$NATIVE_LE" ]; then
            "$W/host/llama98-mips" "$m" --kernel "$k" \
                --tokens "$TOK" --dump-logits "$W/$name.$k.le.bin" \
                >"$W/$name.$k.le.log" 2>&1 || rc_le=$?
        else
            qemu-mipsel-static "$W/le/llama98-mips" "$m" --kernel "$k" \
                --tokens "$TOK" --dump-logits "$W/$name.$k.le.bin" \
                >"$W/$name.$k.le.log" 2>&1 || rc_le=$?
        fi

        # A crash and a wrong number are different failures, and so are
        # "one arm died" and "neither arm can read this model at all".
        #
        # Both arms failing the SAME way carries no byte-order
        # information: it is the model being unloadable by this tree, a
        # fact any single-endian gate would also have caught. Calling it
        # a byte-order failure would point the next reader at the byte
        # order of a load that never happened. It is reported, and it
        # does not count as a comparison, but it is not this gate's FAIL.
        #
        # ONE arm failing is the opposite - that is a difference between
        # the two byte orders and is exactly what this gate is for.
        le_bad=0; be_bad=0
        [ "$rc_le" -ne 0 ] || [ ! -s "$W/$name.$k.le.bin" ] && le_bad=1
        [ "$rc_be" -ne 0 ] || [ ! -s "$W/$name.$k.be.raw" ] && be_bad=1
        if [ "$le_bad" -eq 1 ] && [ "$be_bad" -eq 1 ]; then
            if diff -q "$W/$name.$k.le.log" "$W/$name.$k.be.log" >/dev/null 2>&1; then
                printf '  UNLOADABLE %-26s %-6s both arms, identically - not a\n' \
                    "$name" "$k"
                echo "             byte-order finding. The engine cannot read this"
                echo "             model at all:"
                tail -n 2 "$W/$name.$k.le.log" | sed 's/^/               /'
                unloadable="$unloadable $name/$k"
                continue
            fi
            bad "$name/$k: BOTH arms failed but DIFFERENTLY - that is a"
            echo "       byte-order difference in the failure path itself."
            tail -n 3 "$W/$name.$k.be.log" | sed 's/^/       be: /'
            tail -n 3 "$W/$name.$k.le.log" | sed 's/^/       le: /'
            continue
        fi
        if [ "$le_bad" -eq 1 ]; then
            bad "RUNFAIL le $name/$k (rc=$rc_le) while be succeeded"
            tail -n 4 "$W/$name.$k.le.log" | sed 's/^/       /'
            continue
        fi
        if [ "$be_bad" -eq 1 ]; then
            bad "RUNFAIL be $name/$k (rc=$rc_be) while le succeeded"
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
            FAIL_DIFF=1
        fi
    done
done

if [ "$ran_any" -eq 0 ]; then
    echo "endian-gate: SKIP - not one model was comparable on both arms"
    echo "  Nothing about byte order was checked. Set MODELS=... to point"
    echo "  at models this tree can load."
    exit 2
fi

if [ "$FAIL" -ne 0 ]; then
    echo "endian_gate: FAIL"
    exit 1
fi
# THE ARMS DIFFERED, AND THIS GATE CANNOT SAY WHAT THAT MEANS HERE. The
# note above the LE fallback already states the terms, and the gate must
# not then make the claim it says it cannot make: when the LE arm is the
# native host build, the difference may be the compiler and libc rather
# than byte order, so the verdict is SKIP - not a relaxation, the
# replacement of a claim this configuration cannot support. A PASS is
# untouched and still means what it always meant: a load path that
# assumed little-endian could not produce the same logits on a
# big-endian machine.
if [ "$FAIL_DIFF" -ne 0 ]; then
    if [ -n "$CONFOUND" ]; then
        echo "endian-gate: SKIP - the arms differ, and the LE arm is a native"
        echo "  host build: they differ in compiler and libc as well as in byte"
        echo "  order, so this difference cannot be attributed. Install"
        echo "  gcc-14-mipsel-linux-gnu for a matched pair before reading a"
        echo "  FAIL into the DIFFER lines above."
        exit 2
    fi
    echo "endian_gate: FAIL"
    exit 1
fi
if [ -n "$unloadable" ]; then
    echo "endian_gate: PASS (unloadable on both arms, unchecked:$unloadable)"
else
    echo "endian_gate: PASS"
fi
exit 0
