#!/bin/bash
# src/lz_softfp.c against libgcc's own soft-float, operator by operator,
# under qemu-arm-static - AND a check that no operator escaped the test.
#
# WHY THE COVERAGE HALF MATTERS MORE THAN THE CASE COUNT. This comparison
# used to live in a scratch directory outside the tree and to exercise
# eight of the file's twenty-two exported symbols, while the commit that
# added it reported "every operator against libgcc's, 0 of 41,016,746
# random and directed cases differing". __aeabi_f2lz was one of the
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
CC=${CC:-arm-linux-gnueabi-gcc}
RUN=""
if command -v "$CC" >/dev/null 2>&1 || command -v arm-linux-gnueabi-gcc-14 >/dev/null 2>&1; then
    command -v "$CC" >/dev/null 2>&1 || CC=arm-linux-gnueabi-gcc-14
    if command -v qemu-arm-static >/dev/null 2>&1; then RUN=local; fi
fi
if [ -z "$RUN" ] && command -v wsl >/dev/null 2>&1; then
    if wsl -e bash -lc "command -v arm-linux-gnueabi-gcc-14 >/dev/null 2>&1 && command -v qemu-arm-static >/dev/null 2>&1" >/dev/null 2>&1; then
        RUN=wsl
    fi
fi
if [ -z "$RUN" ]; then
    echo "softfp_gate: SKIP - no ARM cross compiler and qemu-arm-static" \
         "(neither on PATH nor in WSL); lz_softfp.c was NOT compared" \
         "against libgcc" >&2
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

# ONE runner call, and the result comes back through a FILE rather than
# through the pipe. A `nm | awk` written inline has to survive this
# script's quoting, bash -lc's, and WSL's; the first attempt lost its
# field selector on the way and reported "no exported symbols", which
# reads exactly like a broken source file.
if ! RUNNER "arm-linux-gnueabi-gcc-14 $FLAGS -c $SRC -o '$WW/mine.o' && \
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
     arm-linux-gnueabi-gcc-14 $FLAGS -w -static -o '$WW/probe' $PROBE '$WW/ren.o' -lm" \
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
