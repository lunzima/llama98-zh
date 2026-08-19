#!/bin/bash
# Build the llama98 engine for 32-bit x87 gcc (AOSC OS / Afterglow's
# packaging configuration for the Socket-7-class family: Pentium MMX,
# K6, Cyrix 6x86, WinChip - MMX present on all of them, no SSE, x87 the
# only float unit). A FOURTH toolchain shape alongside gcc/x86-64, Open
# Watcom and the ARM cross-build - same compiler family as the dev
# machine's own gcc, but a different ISA width and float unit, which is
# exactly what makes it able to reproduce a bug neither of the others
# can.
#
# THE FLOOR IS -march=i586, MATCHING THAT FAMILY. Socket 7 is P5;
# CMOV/FCOMI/FCMOV are i686 (Pentium Pro) additions, and none of Pentium
# MMX/K6/Cyrix 6x86/WinChip have them - #UD on every machine this build
# ships to. Measured before dropping the floor (tests/test_ops_isa_floor.py's
# informational i686 census, before this change: 197 cmov + 156 fcomi +
# 28 fcmov across $(ENG)): i586 compiles every file clean, zero cmov/
# fcomi/fcmov anywhere, and the linked binary's output is bit-identical
# to the i686 build's (--kernel ref and --kernel mmx both checked) - gcc
# falls back to branches on its own once the target does not support
# the instruction, no source change needed.
#
# WHY EVERY FLAG IS THERE:
#
#   -m32 -march=i586        see above.
#   -mno-mmx -mno-sse       the target has MMX (every machine this build
#   -mno-sse2               ships to does), but this flag set is for
#                            every file EXCEPT src/ops_mmx.c, which
#                            builds separately below with -mmmx -msse.
#                            That split (see src/ops_mmx.h) is the fix
#                            for src/ops.c:5788's rms=-nan bug: gcc and
#                            clang both use MMX for an 8-byte block
#                            store when -mmmx is on, with no emms before
#                            the x87 code that follows, corrupting the
#                            x87 tag word. The engine's MMX-intrinsic
#                            kernels still exist and are still linked in
#                            via ops_mmx.o - only the TU that compiles
#                            them keeps the flag.
#   -DLZ_MMX_TU=1            tells every file compiled with the negatives
#                            above that src/ops_mmx.c's kernels are part
#                            of this link, so their dispatch tables and
#                            tier reports stay correct despite this TU
#                            never seeing __MMX__ itself.
#   -mfpmath=387             NOT the default for -m32 on a modern gcc,
#                            which would pick SSE2 scalar float even at
#                            -m32 if the target supports it. This flag
#                            forces genuine x87 float, which is the
#                            entire point of the target: no SSE unit
#                            exists on the real hardware.
#   -ffp-contract=off        cross-compiler bit-identity, same reason
#                            as every other build here (see the ARM
#                            script's header).
#
# docs/x87-gcc-reg-stack-leak.md and docs/x87-underflow-selfcontained.md
# record an earlier investigation into this same rms=-nan symptom, from
# before the MMX/x87 register-file aliasing mechanism above was known;
# both carry explicit retraction notes for the claims the mmx split
# superseded. Read them for the evidence chain, not for the diagnosis.
#
# WALL-CLOCK NUMBERS FROM THIS SCRIPT MEAN NOTHING FOR REAL SOCKET-7/
# AOSC HARDWARE. This host is a 64-bit CPU running the 32-bit binary
# natively (no emulation), which is representative of correctness but
# not of cycles - same caveat the ARM script gives for QEMU, for the
# opposite reason (this one is too FAST relative to the target, not an
# emulator too slow).
set -u

CC=${CC:-gcc}
OUT=${OUT:-/tmp/x87build}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

# -mmmx is NOT on this line and that is the fix, not an oversight: gcc
# and clang both use MMX for an 8-byte block store when -mmmx is on,
# even with no MMX intrinsic anywhere nearby in the source, and neither
# issues emms before the x87 code that follows - which is exactly the
# "real indefinite" (0xFFC00000) this script exists to reproduce.
# -mno-mmx/-mno-sse/-mno-sse2 make the emission impossible at compile
# time rather than merely absent; src/ops_mmx.c (below) is the one
# translation unit that still gets -mmmx, and LZ_MMX_TU tells every
# other one that it exists and is linkable - see src/ops_mmx.h.
FLAGS="-O2 -std=c99 -Wall -Wextra"
FLAGS="$FLAGS -m32 -march=i586 -mno-mmx -mno-sse -mno-sse2 -mfpmath=387"
FLAGS="$FLAGS -ffp-contract=off -DLZ_MMX_TU=1 -I$ROOT/src"

# src/ops_mmx.c's own flags: the negatives above reversed, no
# -mfpmath=387 (that file has no ordinary scalar float arithmetic for it
# to affect, and clang rejects the combination with -msse). -march=i586,
# NOT i686, and no -msse here - measured: gcc treats -msse as license to
# assume CMOV is available for the WHOLE translation unit it is passed
# to, not just the SSE-touching functions in it (lz_amax32_mmx, which
# has nothing to do with SSE, picked up a cmov the moment -msse joined
# -march=i586 on this file's command line - #UD on every machine this
# script's ENG loop above targets). That is why the SSE1-only functions
# live in a separate file now - see src/ops_mmx_sse.c's header.
MMX_FLAGS="-O2 -std=c99 -Wall -Wextra"
MMX_FLAGS="$MMX_FLAGS -m32 -march=i586 -mmmx"
MMX_FLAGS="$MMX_FLAGS -ffp-contract=off -DLZ_MMX_TU=1 -I$ROOT/src"

# src/ops_mmx_sse.c: the two functions that need genuine SSE1. -march=i686
# here is honest, not a compromise - real SSE1 hardware is i686+ anyway
# (SSE starts at Pentium III; no Socket-7 part has it), so a machine
# that can even reach these two functions (gated behind a runtime SSE
# CPUID check, see src/ops.c's LZ_HAVE_P2_SSE/LZ_HAVE_Q8R_SSE dispatch)
# already has CMOV too.
SSE_FLAGS="-O2 -std=c99 -Wall -Wextra"
SSE_FLAGS="$SSE_FLAGS -m32 -march=i686 -mmmx -msse"
SSE_FLAGS="$SSE_FLAGS -ffp-contract=off -DLZ_MMX_TU=1 -I$ROOT/src"

# Same list as the Makefile's ENG, and it has to stay the same list -
# see the Makefile note about source lists living in exactly one place.
# ops_mmx/ops_mmx_sse are NOT in ENG (Watcom and the ARM cross-build
# must not see them - see the Makefile's own ENG_MMX_GCC variable) so
# they are built separately, right after this loop.
ENG="err compat lfn json safetensors model ops forward sampler generate
     tokenizer unicode chat gbk cpucheck"

# $OUT is scratch, wiped and rebuilt from nothing on every run - not a
# durable artifact. A stale directory from an old invocation (a
# pre-split source tree, a different script version, or unrelated
# debris that happens to share the path) must never coexist with a
# fresh one and be mistaken for current: that reads as a passing check
# on a binary that predates whatever change is being verified.
rm -rf "$OUT"
mkdir -p "$OUT"

fail=0
warn=0
for f in $ENG; do
    if $CC $FLAGS -c "$ROOT/src/$f.c" -o "$OUT/$f.o" 2>"$OUT/$f.log"; then
        n=$(grep -c 'warning:' "$OUT/$f.log" || true)
        warn=$((warn + n))
        printf '  %-14s ok    %s warnings\n' "$f" "$n"
    else
        printf '  %-14s FAIL\n' "$f"
        grep -E 'error:' "$OUT/$f.log" | head -4 | sed 's/^/      /'
        fail=$((fail + 1))
    fi
done

if $CC $MMX_FLAGS -c "$ROOT/src/ops_mmx.c" -o "$OUT/ops_mmx.o" 2>"$OUT/ops_mmx.log"; then
    n=$(grep -c 'warning:' "$OUT/ops_mmx.log" || true)
    warn=$((warn + n))
    printf '  %-14s ok    %s warnings\n' "ops_mmx" "$n"
else
    printf '  %-14s FAIL\n' "ops_mmx"
    grep -E 'error:' "$OUT/ops_mmx.log" | head -4 | sed 's/^/      /'
    fail=$((fail + 1))
fi

if $CC $SSE_FLAGS -c "$ROOT/src/ops_mmx_sse.c" -o "$OUT/ops_mmx_sse.o" 2>"$OUT/ops_mmx_sse.log"; then
    n=$(grep -c 'warning:' "$OUT/ops_mmx_sse.log" || true)
    warn=$((warn + n))
    printf '  %-14s ok    %s warnings\n' "ops_mmx_sse" "$n"
else
    printf '  %-14s FAIL\n' "ops_mmx_sse"
    grep -E 'error:' "$OUT/ops_mmx_sse.log" | head -4 | sed 's/^/      /'
    fail=$((fail + 1))
fi

echo
echo "objects: $(ls -1 "$OUT"/*.o 2>/dev/null | wc -l)  failed: $fail  warnings: $warn"
[ "$fail" -eq 0 ] || exit 1

# Link the CLI. cli_main.c's -i loop runs on the shared LZSession core
# in common/, a front-end helper deliberately outside $(ENG) - so this
# lists it explicitly with -Icommon, exactly as the Makefile's native
# target and the ARM/Watcom CLI/DOS builds do.
COM="stream settings command savechat mru chatfile session"
if [ -f "$ROOT/src/cli_main.c" ]; then
    COMSRC=""
    for c in $COM; do
        [ -f "$ROOT/common/$c.c" ] && COMSRC="$COMSRC $ROOT/common/$c.c"
    done
    if $CC $FLAGS -I"$ROOT/common" -o "$OUT/llama98-x87" \
         "$ROOT/src/cli_main.c" "$ROOT/src/cli_argv.c" \
         "$ROOT/src/cli_attr.c" $COMSRC "$OUT"/*.o -lm \
         2>"$OUT/link.log"; then
        echo "linked: $OUT/llama98-x87"
        file "$OUT/llama98-x87" 2>/dev/null | sed 's/^/  /'
    else
        echo "link FAILED:"
        grep -E 'error|undefined' "$OUT/link.log" | head -8 | sed 's/^/  /'
        exit 1
    fi
fi
