#!/bin/bash
# x86 <-> ARMv5TE bit-identity parity gate.
#
#   parity_gate.sh <model-dir> [tokens]
#
# Cross-builds the ARM binary (build_engine.sh), runs the SAME model
# through it under QEMU and through an already-built native x86 binary,
# with EVERY AUTO-resolved tier PINNED EXPLICITLY on both sides (the
# list comes from --pin-tiers, see below), then cmp's the --dump-all-logits
# output byte for byte.
#
# Exit codes:
#   0  PASS - bit-identical
#   1  FAIL - a build/run step failed, or the two dumps differ
#   2  SKIP - ARM toolchain / QEMU / sysroot / x86 binary not available
#             (neither natively nor via a WSL fallback)
#
# WHY THE TWO OLDEST PINS BITE, as a worked example of the rest:
#
#   build_engine.sh does not define __MMX__ (that macro is x86-only, and
#   nothing here asks the ARM cross compiler to fake it), so the ARM
#   binary carries none of the x86 tiers. It has three of its own -
#   ref, arm-c and arm-asm, the last two being the ternary column's C
#   and hand-written cells - and `auto` now resolves to arm-asm, so an
#   UNPINNED run compares a hand-written ARM kernel against an x86 one
#   and calls a real divergence "expected". ref is the tier both sides
#   literally execute: it is always compiled on x86 (the pure C
#   reference every other tier is checked against) and it is a real
#   code path on ARM, not a name. The three ARM tiers owe each other
#   bit-identity UNDER THESE PINS, which is what the loop below checks
#   by running this binary three times with --kernel ref / arm-c /
#   arm-asm.
#
#   They do NOT owe it on an unpinned auto run, and that is not a
#   defect: ref routes Q8_0 to matmul_scalar_ref, which has no fixed
#   epilogue at all, structurally, regardless of any tier's value, while
#   arm-c/arm-asm reach matmul_q8_impl where the epilogue selects
#   epi_fixed - unconditionally now on every platform (the shipping
#   build is fixed-point by default everywhere; nothing asks a
#   per-platform hardware question - see
#   docs/fixed-tier-pipeline-breakpoints.md), so ref and arm-c/arm-asm
#   diverge on the default on x86 too, not only against an ARM target.
#   Measured on _armgate2: ref differs from arm-c by default and matches
#   once the epilogue is off. That used to be `--epi-write float`, one
#   of seven per-tier runtime flags; they collapsed into --fixed, and
#   the epilogue's own axis is now the LZ_EPI_FIXED compile switch (see
#   ops.h and build/kernel_isa_identity_gate.sh, which builds with it at
#   0 for exactly this row). The epilogue is Q8_0-only on every
#   platform, so a model without Q8_0 tensors shows no such split. See
#   docs/arm-asm-audit.md section 1 (written before the tiers'
#   auto-resolution changed; read its epilogue-routing claim, not its
#   per-platform-hardware framing).
#
#   Asking for a tier the build lacks clamps, and cli_main.c's --kernel
#   handler only prints "requested X, running Y" when the CLI arg and
#   the tier actually run disagree; on an unpinned "auto" run there is
#   no CLI arg to disagree with, so nothing gets printed.
#
#   The recurrence has no axis of its own any more: it resolves from
#   --fixed like every other arithmetic site (see lz_float_ref_select in
#   src/ops_sched.c and docs/fixed-tier-pipeline-breakpoints.md), fixed
#   by default, the same on x86 and on ARM - no per-platform divergence
#   to guard against on this axis by itself. lz_tier_pins() below still
#   pins the whole engine explicitly anyway, rather than trusting that
#   two builds' unpinned autos agree; --pin-tiers is the single source
#   of the whole list, so this comment does not need to track it value
#   by value.
#
# What this does NOT check: timing. See run_qemu.sh's own header - QEMU
# user-mode has no cache/bus/pipeline model, so a wall-clock number here
# is a number about the x86 host translating instructions, not about the
# AK7802. This gate is bit-identity only.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

MODEL=${1:-}
# TEN TOKENS WAS NOT ENOUGH AND THIS GATE'S SIBLING SAYS SO IN WRITING.
# build/fixed_tier_xcompiler_gate.sh's header: the tier divergences it
# hunts need "~150 tokens of specific text" before they surface, and "at
# 8 tokens even THAT one reads clean" - which is why that gate runs the
# 603/661-token real-text fixtures. This one ran 1..10, so the
# soft-float arm of the cross-regime comparison was being certified on a
# length the tree had already measured as too short. The two gates also
# therefore shared no fixture, so "gcc == Watcom" and "gcc == ARM" did
# not compose into a four-way claim.
#
# Default is now the same fixture, truncated past that threshold.
# PARITY_NTOK picks the length: 160 crosses the ~150 the sibling names,
# at roughly seven minutes of QEMU. The whole 603 passes too - run
# 2026-08-23, bit-identical on all three ARM tiers - and is not the
# default only because it takes about half an hour.
TOKFILE=${TOKFILE:-$ROOT/build/norm_int_gate_tokens_zh.txt}
NTOK=${PARITY_NTOK:-160}
if [ -z "${2:-}" ] && [ -f "$TOKFILE" ]; then
    TOKENS=$(tr -s ' \n' '\n' < "$TOKFILE" | grep -v '^$' | head -n "$NTOK" \
             | paste -sd, -)
    echo "-- fixture: first $NTOK of $(basename "$TOKFILE") --"
else
    TOKENS=${2:-1,2,3,4,5,6,7,8,9,10}
fi
if [ -z "$MODEL" ]; then
    echo "usage: parity_gate.sh <model-dir> [tokens]" >&2
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL" >&2
    exit 2
fi

# to_wsl_path P - /e/foo -> /mnt/e/foo (DrvFs). A no-op for anything not
# drive-letter shaped, so it is safe to call on a path that is already
# WSL-native.
to_wsl_path() {
    case "$1" in
        /[a-z]/*) printf '/mnt/%s/%s' "$(echo "$1" | cut -c2)" "$(echo "$1" | cut -c4-)" ;;
        *) printf '%s' "$1" ;;
    esac
}

CC=${CC:-arm-linux-gnueabi-gcc}
SYSROOT=${SYSROOT:-/usr/arm-linux-gnueabi}
ARM_OUT=${ARM_OUT:-/tmp/armbuild}
ARM_BIN=${ARM_BIN:-$ARM_OUT/llama98-arm}
X86_BIN=${X86_BIN:-}

# ---- ARM toolchain: PATH first, WSL fallback second - ALL of this is --
# ---- SKIP, not FAIL -------------------------------------------------
# A missing cross toolchain says nothing about the engine; only a build
# or a mismatch that actually ran should fail the gate.
#
# The toolchain lives in WSL on this host and on PATH on a Linux CI one.
# Try PATH as-is first (respects a $CC override); only fall back to WSL
# when that fails, following the same convention as
# build/kernel_matrix_gate.sh's arm_try_local()/wsl split. Absent from
# both is SKIP, and the message says which of the two was missing what.
LOCAL_OK=1
LOCAL_WHY=""
if ! command -v "$CC" >/dev/null 2>&1; then
    if command -v arm-linux-gnueabi-gcc-14 >/dev/null 2>&1; then
        CC=arm-linux-gnueabi-gcc-14
    else
        LOCAL_OK=0
        LOCAL_WHY="ARM cross compiler not found on PATH: $CC"
    fi
fi
if [ "$LOCAL_OK" = 1 ] && ! command -v qemu-arm-static >/dev/null 2>&1; then
    LOCAL_OK=0
    LOCAL_WHY="qemu-arm-static not found on PATH"
fi
if [ "$LOCAL_OK" = 1 ] && [ ! -d "$SYSROOT" ]; then
    LOCAL_OK=0
    LOCAL_WHY="ARM sysroot not found: $SYSROOT"
fi

ARM_BACKEND=""
if [ "$LOCAL_OK" = 1 ]; then
    ARM_BACKEND=local
elif command -v wsl >/dev/null 2>&1; then
    WSL_ROOT=$(to_wsl_path "$ROOT")
    if wsl -e bash -lc "command -v arm-linux-gnueabi-gcc-14 >/dev/null 2>&1 && command -v qemu-arm-static >/dev/null 2>&1 && [ -d /usr/arm-linux-gnueabi ]" >/dev/null 2>&1; then
        ARM_BACKEND=wsl
        CC=arm-linux-gnueabi-gcc-14
        SYSROOT=/usr/arm-linux-gnueabi
    fi
fi
if [ -z "$ARM_BACKEND" ]; then
    UNCHECKED="the x86<->ARM logits comparison (pinned-tier run and the arm-c/arm-asm tier checks) did not run"
    if command -v wsl >/dev/null 2>&1; then
        echo "SKIP: $LOCAL_WHY, and WSL has none either (need arm-linux-gnueabi-gcc-14 + qemu-arm-static + /usr/arm-linux-gnueabi). $UNCHECKED." >&2
    else
        echo "SKIP: $LOCAL_WHY, and there is no WSL to fall back to. $UNCHECKED." >&2
    fi
    exit 2
fi

if [ -z "$X86_BIN" ]; then
    # Host-native FIRST. Under WSL both names are executable - .exe runs
    # through interop - but a Windows binary handed a /mnt/e/... path
    # cannot resolve it, so preferring .exe there fails with "cannot find
    # config.json" on a directory that plainly exists. Picking a binary
    # this host cannot actually drive is worse than finding none.
    # NOT A PREFERENCE ON LINUX, AN EXCLUSION. This listed the .exe
    # second there, which still selects it whenever the ELF binary is
    # absent - and absent is the normal state on a Windows host, whose
    # `make` produces only the .exe. The comment above already says
    # picking a binary this host cannot drive is worse than finding
    # none; ranking it second does exactly that. Finding none is now the
    # answer, and the block below builds one.
    case "$(uname -s)" in
        Linux*)  order="$ROOT/llama98" ;;
        *)       order="$ROOT/llama98.exe $ROOT/llama98" ;;
    esac
    for cand in $order; do
        [ -x "$cand" ] && X86_BIN=$cand && break
    done
fi
# NOTHING USABLE FOUND: build one, out of tree. On a Windows host the
# repo's only native binary is llama98.exe, and the block above rejects
# it here for a good reason - it cannot resolve the /mnt/... paths qemu
# and this script use. That used to end in SKIP, so the gate reported
# nothing on the machine where the ARM toolchain actually lives.
#
# INTO ITS OWN DIRECTORY, never build/native/. That path is shared
# between a MinGW build and a Linux build of the same tree, and neither
# host's ld accepts the other's objects - native_obj_gate exists because
# of it. `make native` here would leave ELF objects where the Windows
# build expects COFF and break it far from the cause. So this compiles
# the same $(ENG) list the Makefile owns into $OUT, exactly as
# build/arm/build_engine.sh does for the ARM side.
if [ -z "$X86_BIN" ] || [ ! -x "$X86_BIN" ]; then
    case "$(uname -s)" in
        Linux*) ;;
        *)  echo "SKIP: no native x86 binary to compare against (build with 'make native' or set X86_BIN)" >&2
            exit 2 ;;
    esac
    XOUT=${XOUT:-/tmp/x86ref}
    XCC=${XCC:-gcc}
    command -v "$XCC" >/dev/null 2>&1 || {
        echo "SKIP: no x86 binary and no $XCC to build one" >&2; exit 2; }
    echo "-- no ELF x86 binary; building a reference into $XOUT --"
    XFLAGS="-O2 -std=c99 -Wall -Wextra -I$ROOT/src -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
    # BOTH LISTS. ENG is the portable engine; ENG_MMX is the three
    # SIMD translation units the Makefile compiles separately, and the
    # ARM builder does not need them because ARM has no MMX. Taking only
    # ENG here links with lz_fwht_stage_mmx and five others undefined -
    # which is what it did. Read from the Makefile rather than restated,
    # for the reason build/arm/build_engine.sh's header gives: the copy
    # that got restated is the copy that went stale.
    XENG=$(make -s -C "$ROOT" print-ENG) || {
        echo "SKIP: cannot read ENG from the Makefile" >&2; exit 2; }
    XENG="$XENG $(make -s -C "$ROOT" print-ENG_MMX)"
    mkdir -p "$XOUT"; rm -f "$XOUT"/*.o
    xfail=0
    for f in $XENG; do
        $XCC $XFLAGS -mmmx -msse -msse2 -c "$ROOT/src/$f.c" -o "$XOUT/$f.o" \
            2>"$XOUT/$f.log" || { xfail=1; echo "  $f FAIL"; \
            grep -E 'error:' "$XOUT/$f.log" | head -3 | sed 's/^/      /'; }
    done
    XCOM=$(make -s -C "$ROOT" print-CLI_COM)
    XCOMSRC=""
    for c in $XCOM; do
        [ -f "$ROOT/common/$c.c" ] && XCOMSRC="$XCOMSRC $ROOT/common/$c.c"
    done
    if [ "$xfail" -eq 0 ] && $XCC $XFLAGS -I"$ROOT/common" \
            -o "$XOUT/llama98-x86" "$ROOT/src/cli_main.c" \
            "$ROOT/src/cli_argv.c" "$ROOT/src/cli_attr.c" \
            $XCOMSRC "$XOUT"/*.o -lm 2>"$XOUT/link.log"; then
        X86_BIN="$XOUT/llama98-x86"
        echo "   built: $X86_BIN"
    else
        echo "SKIP: could not build an x86 reference" >&2
        grep -E 'error|undefined' "$XOUT/link.log" 2>/dev/null | head -6 \
            | sed 's/^/      /'
        exit 2
    fi
fi
if [ -z "$X86_BIN" ] || [ ! -x "$X86_BIN" ]; then
    echo "SKIP: no native x86 binary to compare against (build with 'make native' or set X86_BIN)" >&2
    exit 2
fi

# ---- work directories -----------------------------------------------
# $WORK holds x86-side and log files (always local to this host).
# $ARM_WORK holds the ARM build and its --dump-logits output; under the
# wsl backend those files are written from inside WSL, so $ARM_WORK has
# to live under a drive letter (DrvFs), not under mktemp's MSYS /tmp -
# WSL cannot see the latter. It is placed next to $ROOT rather than
# under $ROOT/build/arm/ because that directory is `!`-reincluded whole
# in .gitignore (untracked cruft there is not ignored); next to $ROOT it
# is outside the repo and the trap removes it regardless of backend.
WORK=$(mktemp -d)
if [ "$ARM_BACKEND" = wsl ]; then
    ARM_WORK="$(dirname "$ROOT")/.parity-gate-wsl-$$"
    mkdir -p "$ARM_WORK" || {
        echo "FAIL: could not create $ARM_WORK" >&2
        exit 1
    }
    ARM_OUT="$ARM_WORK/build"
    ARM_BIN="$ARM_OUT/llama98-arm"
    WSL_ARM_OUT=$(to_wsl_path "$ARM_OUT")
    WSL_ARM_BIN=$(to_wsl_path "$ARM_BIN")
    MODEL_ABS=$(cd "$MODEL" && pwd) || {
        echo "FAIL: could not resolve $MODEL to an absolute path" >&2
        exit 1
    }
    WSL_MODEL=$(to_wsl_path "$MODEL_ABS")
else
    ARM_WORK="$WORK"
fi
trap 'rm -rf "$WORK" "${ARM_WORK:-}"' EXIT

# run_arm_dump DUMPFILE ARGS... - run the ARM binary under QEMU, locally
# or via WSL per $ARM_BACKEND, writing --dump-all-logits to DUMPFILE.
# DUMPFILE must be under $ARM_WORK so both sides of a wsl run can see it.
#
# --dump-all-logits, not --dump-logits. The latter writes the LAST
# position and nothing else, so this gate - the only thing standing
# between a soft-float divergence and shipping - was asserting
# bit-identity on 1 of the fixture's 159 prediction vectors. A kernel
# that diverged at position 3 and reconverged by the end would have
# passed it. The run is what costs 35 minutes under QEMU; the dump is
# 20 MB instead of 128 KB, so this is 159x the assertion for free.
run_arm_dump() {
    local dump="$1"; shift
    if [ "$ARM_BACKEND" = wsl ]; then
        local wdump
        wdump=$(to_wsl_path "$dump")
        wsl -e bash -lc "cd '$WSL_ROOT' && BIN='$WSL_ARM_BIN' SYSROOT=/usr/arm-linux-gnueabi bash build/arm/run_qemu.sh '$WSL_MODEL' $* --dump-all-logits '$wdump'"
    else
        BIN="$ARM_BIN" SYSROOT="$SYSROOT" "$HERE/run_qemu.sh" "$MODEL" "$@" --dump-all-logits "$dump"
    fi
}

# ---- cross-build the ARM side -------------------------------------------
if [ "$ARM_BACKEND" = wsl ]; then
    echo "-- cross-building ARM under WSL (CC=$CC) --"
    BUILD_LOG="$ARM_WORK/build.log"
    if ! wsl -e bash -lc "cd '$WSL_ROOT' && CC='$CC' OUT='$WSL_ARM_OUT' bash build/arm/build_engine.sh" >"$BUILD_LOG" 2>&1; then
        echo "FAIL: ARM cross-build under WSL failed, see $BUILD_LOG" >&2
        tail -n 20 "$BUILD_LOG" >&2
        exit 1
    fi
    if [ ! -f "$ARM_BIN" ]; then
        echo "FAIL: ARM cross-build under WSL reported success but $ARM_BIN is missing" >&2
        exit 1
    fi
else
    echo "-- cross-building ARM (CC=$CC) --"
    BUILD_LOG="$ARM_OUT.build.log"
    mkdir -p "$(dirname "$ARM_OUT")" 2>/dev/null
    if ! CC="$CC" OUT="$ARM_OUT" "$HERE/build_engine.sh" >"$BUILD_LOG" 2>&1; then
        echo "FAIL: ARM cross-build failed, see $BUILD_LOG" >&2
        tail -n 20 "$BUILD_LOG" >&2
        exit 1
    fi
    if [ ! -x "$ARM_BIN" ]; then
        echo "FAIL: ARM cross-build script reported success but $ARM_BIN is missing" >&2
        exit 1
    fi
fi

# ---- run both sides with the tiers pinned --------------------------------
X86_DUMP="$WORK/x86.logits"
ARM_DUMP="$ARM_WORK/arm.logits"

# ASKED FOR, not transcribed. The configuration must be pinned: not
# because ARM and x86 disagree on what they resolve to by default (they
# do not - both are fixed everywhere, see
# docs/fixed-tier-pipeline-breakpoints.md), but because this gate
# deliberately compares the FLOAT arm (lz_tier_pins()'s own values,
# ops_sched.c) rather than the shipping default, so that a real
# x86<->ARM divergence in the float arithmetic is not masked by both
# sides already agreeing on integer results that cannot differ. An
# unpinned run would compare each side's default and tell this gate
# nothing about the float paths it exists to check.
#
# WHAT THIS GATE NO LONGER COVERS, now that the seven per-tier flags
# collapsed into --fixed: the pin string used to be a MIXED setting, so
# each run exercised some fixed kernels and some float ones on both
# targets at once. It is `--fixed off` now, a clean float arm. The fixed
# side is not lost - it is integer arithmetic, so x86 and ARM must be
# byte-identical on it and the shipping-default run in this gate's own
# rows below is that comparison - but the two are separate rows now
# rather than one blended one, and only the float row carries a
# tolerance.
#
# The list is NOT a literal here, because a literal goes stale silently:
# one that pinned only --kernel and the recurrence, against four more
# pinned knobs, made this gate report 94505 of 131072 bytes differing
# for that reason alone. lz_tier_pins() in src/ops_sched.c is the single
# home, so changing the pins updates this script with no edit here.
PINS=$("$X86_BIN" --pin-tiers) || {
    echo "FAIL: could not read --pin-tiers from $X86_BIN" >&2
    exit 1
}
if [ -z "$PINS" ]; then
    echo "FAIL: --pin-tiers returned nothing; refusing to compare with " \
         "unpinned AUTO tiers" >&2
    exit 1
fi

echo "-- x86 native: $X86_BIN --"
if ! "$X86_BIN" "$MODEL" $PINS \
        --tokens "$TOKENS" --dump-all-logits "$X86_DUMP" \
        >"$WORK/x86.log" 2>&1; then
    echo "FAIL: x86 run failed:" >&2
    cat "$WORK/x86.log" >&2
    exit 1
fi

echo "-- ARM under QEMU (arm926, backend=$ARM_BACKEND): $ARM_BIN --"
if ! run_arm_dump "$ARM_DUMP" $PINS \
        --tokens "$TOKENS" \
        >"$WORK/arm.log" 2>&1; then
    echo "FAIL: ARM/QEMU run failed:" >&2
    cat "$WORK/arm.log" >&2
    exit 1
fi

if [ ! -s "$X86_DUMP" ] || [ ! -s "$ARM_DUMP" ]; then
    echo "FAIL: one or both --dump-logits files are empty or missing" >&2
    echo "  x86: $([ -f "$X86_DUMP" ] && wc -c < "$X86_DUMP" || echo MISSING) bytes" >&2
    echo "  arm: $([ -f "$ARM_DUMP" ] && wc -c < "$ARM_DUMP" || echo MISSING) bytes" >&2
    exit 1
fi

X86_SZ=$(wc -c < "$X86_DUMP")
ARM_SZ=$(wc -c < "$ARM_DUMP")
if [ "$X86_SZ" != "$ARM_SZ" ]; then
    echo "FAIL: dump size mismatch - x86 $X86_SZ bytes, arm $ARM_SZ bytes" >&2
    exit 1
fi

if ! cmp -s "$X86_DUMP" "$ARM_DUMP"; then
    NDIFF=$(cmp -l "$X86_DUMP" "$ARM_DUMP" | wc -l)
    echo "FAIL: x86/ARM logits differ - $NDIFF of $X86_SZ bytes differ" >&2
    cmp "$X86_DUMP" "$ARM_DUMP" >&2
    exit 1
fi
echo "PASS: x86 and ARM --dump-all-logits are bit-identical ($X86_SZ bytes)"

# ---- the ARM column's own tiers ------------------------------------------
# The pins force --kernel ref because that is the one tier BOTH sides can
# run, which means the comparison above never executes an ARM ternary
# kernel at all. What has to hold is bit-identity ACROSS tiers, so the other
# two run here against the SAME binary's ref dump ($ARM_DUMP, produced
# with the pins as-is). Without this a hand-written kernel could rot
# untouched behind a passing gate.
#
# The clamp check is not decoration: lz_kernel_select silently falls back
# to a tier the build has, so a tier that vanished would come back as a
# green comparison of ref against ref. cli_main prints the "[kernel:
# requested" line exactly when that happened.
for K in arm-c arm-asm; do
    TDUMP="$ARM_WORK/arm.$K.logits"
    if ! run_arm_dump "$TDUMP" $PINS --kernel "$K" \
            --tokens "$TOKENS" \
            >"$WORK/arm.$K.log" 2>&1; then
        echo "FAIL: ARM --kernel $K run failed:" >&2
        cat "$WORK/arm.$K.log" >&2
        exit 1
    fi
    if grep -q '\[kernel: requested' "$WORK/arm.$K.log"; then
        echo "FAIL: --kernel $K was not honoured by $ARM_BIN:" >&2
        grep '\[kernel: requested' "$WORK/arm.$K.log" >&2
        exit 1
    fi
    if ! cmp -s "$ARM_DUMP" "$TDUMP"; then
        NDIFF=$(cmp -l "$ARM_DUMP" "$TDUMP" | wc -l)
        echo "FAIL: ARM tier $K differs from ref - $NDIFF of $ARM_SZ bytes" >&2
        exit 1
    fi
    echo "PASS: ARM tier $K is bit-identical to ref ($ARM_SZ bytes)"
done

# ---- the ARM column's FIXED tiers -------------------------------------
# The pins force --fixed off, so the loop above never executes a
# fixed-point ARM body: recur's lz_gdn1_x4_arm_asm, epi's, sigq's,
# nrmss's and fwhti's hand-written cells all sit behind the fixed tier,
# which the float arm turns off. They ARE the shipping default (-fixed
# all is what every build runs; the float arm exists only for
# cross-checking). A fixed kernel change therefore needs this round,
# not an ad-hoc probe - the recur kernel landed 0d10a89 only after a
# throwaway unit probe caught what the float-only parity gate could not
# see. The later --fixed all overrides the pins' --fixed off (the CLI
# applies flags in order).
#
# THE REFERENCE IS arm-c, NOT ref. --kernel ref routes every format to
# matmul_scalar_ref, whose Step-2 reduction is float regardless of
# --fixed - it has no fixed epilogue at all (see the header note above),
# so under --fixed all it still yields the FLOAT answer. The fixed
# tier's scales are quantized to int16 (epi_pack_act / epi_prep_plane),
# so fixed differs from float by design (the accepted pass-2 precision,
# docs/fixed-tier-pipeline-breakpoints.md) and a fixed-vs-ref comparison
# can never be bit-identical. arm-c is the fixed C cell, arm-asm the
# hand-written one; arm-asm must agree with arm-c bit for bit, which is
# the fixed analogue of the float loop's arm-c-vs-ref above.
ARM_FX_DUMP="$ARM_WORK/arm.fixed.arm-c.logits"
if ! run_arm_dump "$ARM_FX_DUMP" $PINS --fixed all --kernel arm-c \
        --tokens "$TOKENS" \
        >"$WORK/arm.fixed.arm-c.log" 2>&1; then
    echo "FAIL: ARM --fixed all --kernel arm-c run failed:" >&2
    cat "$WORK/arm.fixed.arm-c.log" >&2
    exit 1
fi
if grep -q '\[kernel: requested' "$WORK/arm.fixed.arm-c.log"; then
    echo "FAIL: --kernel arm-c was not honoured by $ARM_BIN:" >&2
    grep '\[kernel: requested' "$WORK/arm.fixed.arm-c.log" >&2
    exit 1
fi
FXDUMP="$ARM_WORK/arm.fixed.arm-asm.logits"
if ! run_arm_dump "$FXDUMP" $PINS --fixed all --kernel arm-asm \
        --tokens "$TOKENS" \
        >"$WORK/arm.fixed.arm-asm.log" 2>&1; then
    echo "FAIL: ARM --fixed all --kernel arm-asm run failed:" >&2
    cat "$WORK/arm.fixed.arm-asm.log" >&2
    exit 1
fi
if grep -q '\[kernel: requested' "$WORK/arm.fixed.arm-asm.log"; then
    echo "FAIL: --kernel arm-asm was not honoured by $ARM_BIN:" >&2
    grep '\[kernel: requested' "$WORK/arm.fixed.arm-asm.log" >&2
    exit 1
fi
if ! cmp -s "$ARM_FX_DUMP" "$FXDUMP"; then
    NDIFF=$(cmp -l "$ARM_FX_DUMP" "$FXDUMP" | wc -l)
    echo "FAIL: ARM fixed tier arm-asm differs from arm-c - $NDIFF bytes" >&2
    exit 1
fi
echo "PASS: ARM fixed tier arm-asm is bit-identical to arm-c (fixed, $ARM_SZ bytes)"
exit 0
