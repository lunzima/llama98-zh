#!/bin/bash
# src/lz_softfp.c against libgcc's own soft-float, operator by operator,
# under qemu-arm-static - AND a check that no operator escaped the test.
#
# WHY THE COVERAGE HALF MATTERS MORE THAN THE CASE COUNT. A comparison
# that covers part of the file can report a perfect case count: one in a
# scratch directory outside the tree exercised eight of the twenty-two
# exported symbols while reporting "every operator against libgcc's, 0 of
# 41,016,746 random and directed cases differing". __aeabi_f2lz was one of the
# fourteen it never called, and it was broken: declared to return `long`
# where the EABI name means a 64-bit integer, so it filled r0 and left
# the high half in r1 as whatever the caller had there. Every conversion
# came back as a correct low word beneath garbage.
#
# It stayed invisible because nothing reached it - src/json.c held its
# numbers as double, so safetensors' offsets converted through libgcc's
# __aeabi_d2lz, which this file does not define. The same commit narrowed
# that field to float. Every tensor offset then went through the broken
# routine and the loader started rejecting offsets of 0..2048 as
# negative. Forty-one million cases of the OTHER eight operators said
# nothing about it. So this script fails if a symbol lz_softfp.c exports
# is not named in softfp_probe.c, and that assertion is the point.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP (no ARM toolchain / qemu).
set -u
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)

PROBE=build/arm/softfp_probe.c
SRC=src/lz_softfp.c

# ---- toolchain: PATH first, WSL fallback, absent is SKIP -------------
# Same convention as build/arm/parity_gate.sh: a missing cross compiler
# says nothing about the engine, so it must not be able to fail this.
#
# TWO DRIVERS CAN COMPILE THIS FILE, and they are tried in this order:
#
#   1. arm-linux-gnueabi-gcc-14 / arm-linux-gnueabi-gcc - the
#      soft-float ABI compiler the gate was written for. Its flags,
#      its objects and the comparison are unchanged.
#   2. arm-linux-gnueabihf-gcc, with -mfloat-abi=soft (already in
#      $FLAGS below) and the armel sysroot named explicitly. A
#      hard-float driver is hard-float by CONFIGURATION, not by the
#      flags it is handed: -dumpmachine answers arm-linux-gnueabihf
#      and its default mode is THUMB, so it needs -marm and the armel
#      headers, start files and libgcc, none of which its own sysroot
#      has. What is being COMPARED does not change: lz_softfp.c is
#      still built for the soft-float ABI, and the reference is still
#      libgcc's own __aeabi_* out of the armel libgcc.a - the same
#      file arm-linux-gnueabi-gcc-14 links. Two measured facts hold
#      that up rather than an assertion here: cc1 refuses the
#      alternative for this target (at -march=armv5te, which has no
#      FPU, -mfloat-abi=hard stops at "selected architecture lacks an
#      FPU" instead of quietly emitting VFP code), and ld refuses to
#      merge a VFP-register-argument object with a non-VFP one, naming
#      it. The ABI cannot drift without something stopping the build.
#   3. neither: SKIP, naming what was wanted, what was found, and what
#      it would take to run.
#
# WHY THE PROBE HAD TO STOP ASKING FOR A NAME. It wanted the name
# `arm-linux-gnueabi-gcc-14`, and when that name was absent it printed
# "no ARM cross compiler and qemu-arm-static (neither on PATH nor in
# WSL)". On the machine where that line was read, WSL has
# arm-linux-gnueabihf-gcc, qemu-arm-static, the armel binutils, the
# armel sysroot (/usr/arm-linux-gnueabi: headers, libc.a, crt*.o) and
# the armel libgcc - everything except the armel DRIVER, one
# uninstalled package. The gate was dark for the name of a package, and
# the line it printed read as "this box has no ARM toolchain at all",
# which is how a machine with a usable toolchain kept a gate invisible.
SYSROOT=${SYSROOT:-/usr/arm-linux-gnueabi}
# $CC first when the caller set one, so the existing escape hatch still
# picks the compiler of record; then the preference order above.
CC_TRY="${CC:-} arm-linux-gnueabi-gcc-14 arm-linux-gnueabi-gcc \
        arm-linux-gnueabihf-gcc"

found_cc() {
    for c in $CC_TRY; do
        command -v "$c" >/dev/null 2>&1 && printf '%s ' "$c"
    done
}
LOCAL_FOUND=$(found_cc)
LOCAL_QEMU=no
command -v qemu-arm-static >/dev/null 2>&1 && LOCAL_QEMU=yes

WSL_FOUND=""
WSL_QEMU=no
if command -v wsl >/dev/null 2>&1; then
    WSL_FOUND=$(wsl -e bash -lc "for c in $CC_TRY; do command -v \$c >/dev/null 2>&1 && printf '%s ' \$c; done" 2>/dev/null)
    if [ -n "$WSL_FOUND" ] && \
       wsl -e bash -lc 'command -v qemu-arm-static >/dev/null 2>&1'; then
        WSL_QEMU=yes
    fi
fi

RUN=""
if [ -n "$LOCAL_FOUND" ] && [ "$LOCAL_QEMU" = yes ]; then
    RUN=local
elif [ -n "$WSL_FOUND" ] && [ "$WSL_QEMU" = yes ]; then
    RUN=wsl
fi
if [ -z "$RUN" ]; then
    echo "softfp_gate: SKIP - no ARM cross compiler this gate can use," \
         "so lz_softfp.c was NOT compared against libgcc" >&2
    echo "      wanted: arm-linux-gnueabi-gcc-14 or arm-linux-gnueabi-gcc" >&2
    echo "              (soft-float ABI), else arm-linux-gnueabihf-gcc" \
         "+ $SYSROOT + qemu-arm-static" >&2
    echo "      found on PATH: ${LOCAL_FOUND:-none}; qemu-arm-static: $LOCAL_QEMU" >&2
    echo "      found in WSL : ${WSL_FOUND:-none}; qemu-arm-static: $WSL_QEMU" >&2
    echo "      to run it: apt install gcc-14-arm-linux-gnueabi (the" \
         "soft-float driver), or" >&2
    echo "                 libc6-dev-armel-cross beside arm-linux-gnueabihf-gcc" >&2
    exit 2
fi

# ---- the coverage assertion, and it runs FIRST ----------------------
# Reading the exported symbols off the object rather than a list kept
# here on purpose: a list would go stale the moment someone adds a
# routine, which is the failure this whole gate exists to prevent.
# THE WORK DIRECTORY SITS BESIDE THE REPO, not in /tmp, for the reason
# parity_gate.sh gives: under the wsl backend the compiler writes from
# inside WSL while this script reads from the host, and WSL cannot see
# MSYS's /tmp. A drive-letter path is DrvFs and both can. Beside $ROOT
# rather than under it so nothing lands in the working tree.
W="${LZ_SCRATCH:-$(dirname "$ROOT")/tmp}/.softfp-gate-$$"
mkdir -p "$W" || { echo "softfp_gate: FAIL - cannot create $W"; exit 1; }
trap 'rm -rf "$W"' EXIT

to_wsl() {
    case "$1" in
        /[a-z]/*) printf '/mnt/%s/%s' "$(echo "$1" | cut -c2)" "$(echo "$1" | cut -c4-)" ;;
        *) printf '%s' "$1" ;;
    esac
}
if [ "$RUN" = wsl ]; then
    WROOT=$(to_wsl "$ROOT"); WW=$(to_wsl "$W")
    RUNNER() { wsl -e bash -lc "cd '$WROOT' && $1"; }
else
    WROOT=$ROOT; WW=$W
    RUNNER() { ( cd "$ROOT" && eval "$1" ); }
fi

FLAGS="-O2 -std=c99 -march=armv5te -mtune=arm926ej-s -mfloat-abi=soft"
FLAGS="$FLAGS -ffp-contract=off -fomit-frame-pointer -Isrc"
# STRICT (RNE): this gate compares the soft-float against libgcc, and
# LZ_SOFTFP_FAST now defaults to 1 on the ARM cross (RTZ is shipping).
# The RTZ soft-float is verified by parity_gate's fastfp row instead.
FLAGS="$FLAGS -DLZ_SOFTFP_FAST=0"

# ---- which driver, and whether it can really do the job -------------
# The first recipe whose CANARY builds wins. The canary is the same
# flags on a two-line translation unit that includes <stdint.h> the way
# src/lz_int.h does, and it is what makes "arm-linux-gnueabihf-gcc with
# no armel sysroot installed" a SKIP here instead of the "did not
# cross-compile" FAIL below, where it would read as a defect in
# lz_softfp.c rather than in the machine.
cat > "$W/canary.c" <<'EOS'
#include <stdint.h>
float lz_softfp_canary(float a, float b) { return a + b; }
EOS

if [ "$RUN" = wsl ]; then CAND="$WSL_FOUND"; else CAND="$LOCAL_FOUND"; fi
CC_SEL=""; CCEXT=""; CC_WHY=""
for c in $CAND; do
    TARGET=$(RUNNER "$c -dumpmachine" 2>/dev/null)
    case "$TARGET" in
      *gnueabihf)
        # The armel sysroot REPLACES this driver's own, it is not
        # re-rooted under it: the hard-float include and start-file
        # directories are absolute paths in its configuration, so
        # --sysroot does not reach them and -nostdinc has to take them
        # out of the way first. -B is what redirects crt*.o and
        # libgcc.a; without it ld stops with "crtbeginT.o uses VFP
        # register arguments, this does not", which is the same
        # mismatch this whole recipe exists to avoid, caught here.
        ARMEL_LIB=$(RUNNER 'ls -d /usr/lib/gcc-cross/arm-linux-gnueabi/*/ 2>/dev/null | sort -V | tail -n 1')
        CCEXT="-marm -nostdinc -isystem $SYSROOT/include"
        CCEXT="$CCEXT -isystem ${ARMEL_LIB}include"
        CCEXT="$CCEXT -B$SYSROOT/lib/ -B$ARMEL_LIB"
        CC_WHY="$c (hard-float driver, asked for the soft-float ABI)"
        ;;
      *) CC_WHY="$c (soft-float driver)" ;;
    esac
    OK=no
    RUNNER "$c $FLAGS $CCEXT -c '$WW/canary.c' -o '$WW/canary.o'" \
        >"$W/canary.log" 2>&1 && OK=yes
    if [ "$OK" = yes ]; then CC_SEL=$c; break; fi
    CCEXT=""; CC_WHY=""
done
if [ -z "$CC_SEL" ]; then
    echo "softfp_gate: SKIP - an ARM compiler was found but none could" \
         "build a soft-float object, so lz_softfp.c was NOT compared" \
         "against libgcc" >&2
    echo "      tried:${CAND:- none}" >&2
    echo "      the canary needs the soft-float ABI, <stdint.h>, the" \
         "armel sysroot ($SYSROOT) and" >&2
    echo "      the armel binutils (arm-linux-gnueabi-{nm,objcopy})" >&2
    echo "      to run it: apt install gcc-14-arm-linux-gnueabi, or" \
         "libc6-dev-armel-cross" >&2
    echo "                 beside arm-linux-gnueabihf-gcc" >&2
    tail -n 3 "$W/canary.log" 2>/dev/null | sed 's/^/      /' >&2
    exit 2
fi
echo "softfp_gate: compiler $CC_WHY"

# ONE runner call, and the result comes back through a FILE rather than
# through the pipe. A `nm | awk` written inline has to survive this
# script's quoting, bash -lc's, and WSL's; the first attempt lost its
# field selector on the way and reported "no exported symbols", which
# reads exactly like a broken source file.
if ! RUNNER "$CC_SEL $FLAGS $CCEXT -c $SRC -o '$WW/mine.o' && \
     arm-linux-gnueabi-nm -g --defined-only '$WW/mine.o' > '$WW/syms.raw'" \
        >"$W/cc.log" 2>&1; then
    echo "softfp_gate: FAIL - $SRC did not cross-compile"
    tail -n 5 "$W/cc.log" 2>/dev/null | sed 's/^/      /'
    exit 1
fi

SYMS=$(awk '{print $3}' "$W/syms.raw")
if [ -z "$SYMS" ]; then
    echo "softfp_gate: FAIL - could not read exported symbols from $SRC"
    exit 1
fi

# COMMENTS ARE STRIPPED BEFORE THE SEARCH, and finding that out took a
# planted mutation. The first version grepped the probe whole, so a
# symbol named only in the prose above satisfied it - and this file's
# prose names every symbol it discusses, __aeabi_f2lz included. Deleting
# the declaration still left three mentions and the check stayed green
# (the build then failed, so nothing unsafe shipped, but the gate was
# reporting the wrong thing). What must be true is that the symbol
# appears in CODE.
awk '
    { line = $0; out = ""; i = 1
      while (i <= length(line)) {
          two = substr(line, i, 2)
          if (!inc && two == "/*") { inc = 1; i += 2; continue }
          if ( inc && two == "*/") { inc = 0; i += 2; continue }
          if (!inc) out = out substr(line, i, 1)
          i++
      }
      print out }
' "$PROBE" > "$W/probe.code"

MISSING=""
NSYM=0
for s in $SYMS; do
    NSYM=$((NSYM + 1))
    grep -q -- "$s" "$W/probe.code" || MISSING="$MISSING $s"
done
if [ -n "$MISSING" ]; then
    echo "softfp_gate: FAIL - $SRC exports these with no test in $PROBE:"
    for s in $MISSING; do echo "      $s"; done
    echo "      Add a case naming the symbol. An operator nobody calls is"
    echo "      exactly how __aeabi_f2lz shipped returning half a result."
    exit 1
fi
echo "softfp_gate: $NSYM exported symbol(s), every one named in the probe"

# ---- the comparison -------------------------------------------------
# objcopy renames OUR copies so both implementations coexist in one
# binary; anything not under test keeps a mine_ prefix so it cannot
# satisfy a reference meant for libgcc.
REN=""
for s in $SYMS; do
    case $s in
      __aeabi_fadd|__aeabi_fsub|__aeabi_fmul|__aeabi_fdiv|__aeabi_i2f|\
__aeabi_ui2f|__aeabi_l2f|__aeabi_f2iz|__aeabi_f2lz|__aeabi_fcmpeq|\
__aeabi_fcmplt|__aeabi_fcmple|__aeabi_fcmpge|__aeabi_fcmpgt|\
__aeabi_uidiv|__aeabi_idiv|__aeabi_cfcmpeq|__aeabi_cfrcmple|\
__aeabi_uidivmod|__aeabi_idivmod|__aeabi_uldivmod)
          REN="$REN --redefine-sym $s=lzsf_${s#__aeabi_}" ;;
      __addsf3)
          REN="$REN --redefine-sym $s=lzsf_addsf3" ;;
      *)  REN="$REN --redefine-sym $s=mine_$s" ;;
    esac
done

if ! RUNNER "arm-linux-gnueabi-objcopy $REN '$WW/mine.o' '$WW/ren.o' && \
     $CC_SEL $FLAGS $CCEXT -w -static -o '$WW/probe' $PROBE '$WW/ren.o' -lm" \
        >"$W/link.log" 2>&1; then
    echo "softfp_gate: FAIL - probe did not build"
    tail -n 8 "$W/link.log" 2>/dev/null | sed 's/^/      /'
    exit 1
fi

OUT=$(RUNNER "qemu-arm-static '$WW/probe'" 2>&1)
RC=$?
printf '%s\n' "$OUT"

if [ "$RC" -ne 0 ]; then
    echo "softfp_gate: FAIL"
    exit 1
fi
echo "softfp_gate: PASS"
exit 0
