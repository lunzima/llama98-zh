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
# THIS BUILD, LIKE EVERY LOWERED-FLOOR TARGET IN THIS PROJECT (task #28:
# Watcom-Windows i486, MS-DOS DOS32A i386), ASSUMES A 387 IS PRESENT
# (386+387, 486DX, or - here - anything Socket-7-class, which always has
# one). No x86 target is a no-387 build. That is a decision, not an
# omission: the user deferred no-387 x86 support, so it is a RESERVATION
# - tracked as the `x87` ladder class of the training tree's
# tests/isa_floor.py, which stays a reservation rather than an enforced
# floor for exactly this reason. The project's software-float target is
# the ARM cross build (build/arm/: -mfloat-abi=soft -DLZ_NO_FPU=1, with
# src/lz_softfp.c), whose parity gate runs the fixed tiers on that
# FPU-less core. Do not read that as no-387 x86 coverage: the soft-float
# routines, their call sites and the codegen there are ARM's.
#
# THIS SCRIPT BUILDS AND LINKS. NOTHING RUNS WHAT IT PRODUCES, and the
# distinction matters because ci.sh's stage around it prints PASS and
# calls this "the fourth toolchain shape", which reads like the fourth
# member of the eight-way precision comparison the project requires. It
# is not: it is a link check. No logits from this binary have ever been
# compared with anything.
#
# WHAT IS COVERED, so the gap is the right size rather than the whole
# word "x87": Watcom's x87 arithmetic IS compared numerically -
# kernel_isa_identity_gate's xcc row runs "gcc ref == watcom ref (SSE
# scalar vs x87)" on both fixtures. So x87 as a FLOAT UNIT has a parity
# check. What has none is gcc's own x87 CODEGEN at -m32, which is a
# different instruction selection from either of those and the only
# reason this fourth shape exists.
#
# Closing it needs the binary run under a 32-bit loader and its
# --dump-all-logits compared against the x86-64 build's, which needs
# -m32 multilib in WSL; without it the stage SKIPs and so would the
# comparison. The SKIP is honest. The PASS is what overstates.
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
#                            every file EXCEPT src/ops_mmx.c (-mmmx),
#                            src/ops_mmx_sse.c (-mmmx -msse) and
#                            src/ops_sse2.c (-mmmx -msse -msse2), which
#                            each build separately below. The MMX/SSE1
#                            split (see src/ops_mmx.h) is the fix for
#                            the rms=-nan bug in lz_rmsnorm's row: gcc and
#                            clang both use MMX for an 8-byte block
#                            store when -mmmx is on, with no emms before
#                            the x87 code that follows, corrupting the
#                            x87 tag word. The SSE2 split (src/ops_sse2.h)
#                            is code structure, not a correctness fix -
#                            %xmm does not alias x87 - but gains the
#                            32-bit x87 target a real SSE2 tier for the
#                            first time (P4 Northwood, Pentium M), and
#                            gives the ISA-floor gate the same clean
#                            per-file boundary as the other two splits.
#                            Every kernel still exists and is still
#                            linked in via the three .o files - only the
#                            TU that compiles each keeps its own flags.
#   -DLZ_MMX_TU=1            tells every file compiled with the negatives
#   -DLZ_SSE2_TU=1           above that src/ops_mmx.c's and
#                            src/ops_sse2.c's kernels are part of this
#                            link, so their dispatch tables and tier
#                            reports stay correct despite this TU never
#                            seeing __MMX__/__SSE2__ itself.
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
# time rather than merely absent; src/ops_mmx.c and src/ops_sse2.c
# (below) are the two translation units that still get -mmmx/-msse2,
# and LZ_MMX_TU/LZ_SSE2_TU tell every other one that they exist and are
# linkable - see src/ops_mmx.h and src/ops_sse2.h.
FLAGS="-O2 -std=c99 -Wall -Wextra"
FLAGS="$FLAGS -m32 -march=i586 -mno-mmx -mno-sse -mno-sse2 -mfpmath=387"
FLAGS="$FLAGS -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -I$ROOT/src"

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

# src/ops_sse2.c: every gcc SSE2 kernel (task #22 - code structure, not a
# correctness fix; %xmm does not alias x87). -march=i686 for the same
# reason as SSE_FLAGS above: real SSE2 hardware in the target family (P4
# Northwood, Pentium M) is i686+ (SSE2 debuted with NetBurst, after
# Pentium Pro), so this floor costs nothing - checked (same empirical
# habit that caught -msse enabling CMOV in src/ops_mmx.c): this file
# does contain 4 cmov instructions (lz_rope_sse2's ordinary scalar loop
# bounds), but unlike src/ops_mmx.c's case that is not a leak onto code
# that needs to run on plain-i586 hardware - this whole file only runs
# behind a REAL CPUID check now (src/ops.c's kernel_detect(), which
# task #22 also had to fix - see its LZ_ROW_SSE2_EXTERN branch and
# src/cpucheck.c's gcc-side lz_cpuid1_edx()), so a machine that reaches
# any function in this file already has CMOV too. No LZ_MMX_TU: this
# file does not include src/ops_mmx.h.
SSE2_FLAGS="-O2 -std=c99 -Wall -Wextra"
SSE2_FLAGS="$SSE2_FLAGS -m32 -march=i686 -mmmx -msse -msse2"
SSE2_FLAGS="$SSE2_FLAGS -ffp-contract=off -DLZ_SSE2_TU=1 -I$ROOT/src"

# ASKED FOR, not restated. This was a hand-copy of the Makefile's ENG
# carrying a comment saying it had to stay in step, and it did not: nine
# files short, so the link below failed with eight undefined references
# for as long as this script's CI stage had been skipping for want of a
# -m32 compiler. A stage that never runs cannot notice its own list
# rotting; `make -s print-ENG` has nothing to rot.
#
# ops_mmx/ops_mmx_sse/ops_sse2 are NOT in ENG (Watcom and the ARM
# cross-build must not see them - see the Makefile's own ENG_MMX_GCC
# variable) so they are built separately, right after this loop.
ENG=$(make -s -C "$ROOT" print-ENG)
if [ -z "$ENG" ]; then
    echo "x87: cannot read ENG from $ROOT/Makefile (make -s print-ENG)"
    exit 2
fi

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

if $CC $SSE2_FLAGS -c "$ROOT/src/ops_sse2.c" -o "$OUT/ops_sse2.o" 2>"$OUT/ops_sse2.log"; then
    n=$(grep -c 'warning:' "$OUT/ops_sse2.log" || true)
    warn=$((warn + n))
    printf '  %-14s ok    %s warnings\n' "ops_sse2" "$n"
else
    printf '  %-14s FAIL\n' "ops_sse2"
    grep -E 'error:' "$OUT/ops_sse2.log" | head -4 | sed 's/^/      /'
    fail=$((fail + 1))
fi

echo
echo "objects: $(ls -1 "$OUT"/*.o 2>/dev/null | wc -l)  failed: $fail  warnings: $warn"
[ "$fail" -eq 0 ] || exit 1

# Link the CLI. cli_main.c's -i loop runs on the shared LZSession core
# in common/, a front-end helper deliberately outside $(ENG) - so this
# lists it explicitly with -Icommon, exactly as the Makefile's native
# target and the ARM/Watcom CLI/DOS builds do.
# $(CLI_COM), not $(COM): this links the command line, and the gui-only
# members of common/ are not in it. Asked for, for the reason the ENG
# query above gives.
COM=$(make -s -C "$ROOT" print-CLI_COM)
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
