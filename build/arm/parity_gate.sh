#!/bin/bash
# x86 <-> ARMv5TE bit-identity parity gate.
#
#   parity_gate.sh <model-dir> [tokens]
#
# Cross-builds the ARM binary (build_engine.sh), runs the SAME model
# through it under QEMU and through an already-built native x86 binary,
# with EVERY AUTO-resolved tier PINNED EXPLICITLY on both sides (the
# list comes from --pin-tiers, see below), then cmp's the --dump-logits
# output byte for byte.
#
# Exit codes:
#   0  PASS - bit-identical
#   1  FAIL - a build/run step failed, or the two dumps differ
#   2  SKIP - ARM toolchain / QEMU / sysroot / x86 binary not available
#
# WHY THE TWO OLDEST PINS BITE, as a worked example of the rest:
#
#   build_engine.sh does not define __MMX__ (that macro is x86-only, and
#   nothing here asks the ARM cross compiler to fake it), so the ARM
#   binary compiles in exactly one kernel tier: ref. Asking it for
#   anything else silently clamps to ref - cli_main.c's --kernel handler
#   only prints the "requested X, running Y" warning when the CLI arg
#   and the tier actually run disagree, and on an unpinned "auto" run
#   there is no CLI arg to disagree with, so nothing gets printed. ref
#   is also always compiled on x86 (it is the pure C reference every
#   other tier is checked against), so it is the one tier BOTH sides can
#   literally execute - not merely "closest".
#
#   --recur-write only accepts auto|fixed on the CLI - there is no
#   explicit "float" string (see lz_gdn_p2_select in src/ops.c). "auto"
#   resolves from lz_cpu_has_sse(): true on this x86 host -> float,
#   false on the ARM target -> fixed. Left on auto, the two builds run
#   genuinely different arithmetic (measured: --kernel ref alone with
#   --recur-write left on auto already produces a different dump than
#   pinning both, on the SAME x86 binary), and a real cross-build
#   divergence would look exactly like this expected one. "fixed" is
#   the only value selectable on both sides.
#
# What this does NOT check: timing. See run_qemu.sh's own header - QEMU
# user-mode has no cache/bus/pipeline model, so a wall-clock number here
# is a number about the x86 host translating instructions, not about the
# AK7802. This gate is bit-identity only.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

MODEL=${1:-}
TOKENS=${2:-1,2,3,4,5,6,7,8,9,10}
if [ -z "$MODEL" ]; then
    echo "usage: parity_gate.sh <model-dir> [tokens]" >&2
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "SKIP: model dir not found: $MODEL" >&2
    exit 2
fi

CC=${CC:-arm-linux-gnueabi-gcc}
SYSROOT=${SYSROOT:-/usr/arm-linux-gnueabi}
ARM_OUT=${ARM_OUT:-/tmp/armbuild}
ARM_BIN=${ARM_BIN:-$ARM_OUT/llama98-arm}
X86_BIN=${X86_BIN:-}

# ---- toolchain / artifact preflight - ALL of these are SKIP, not FAIL --
# A missing cross toolchain says nothing about the engine; only a build
# or a mismatch that actually ran should fail the gate.
if ! command -v "$CC" >/dev/null 2>&1; then
    echo "SKIP: ARM cross compiler not found on PATH: $CC" >&2
    exit 2
fi
if ! command -v qemu-arm-static >/dev/null 2>&1; then
    echo "SKIP: qemu-arm-static not found on PATH" >&2
    exit 2
fi
if [ ! -d "$SYSROOT" ]; then
    echo "SKIP: ARM sysroot not found: $SYSROOT" >&2
    exit 2
fi
if [ -z "$X86_BIN" ]; then
    # Host-native FIRST. Under WSL both names are executable - .exe runs
    # through interop - but a Windows binary handed a /mnt/e/... path
    # cannot resolve it, so preferring .exe there fails with "cannot find
    # config.json" on a directory that plainly exists. Picking a binary
    # this host cannot actually drive is worse than finding none.
    case "$(uname -s)" in
        Linux*)  order="$ROOT/llama98 $ROOT/llama98.exe" ;;
        *)       order="$ROOT/llama98.exe $ROOT/llama98" ;;
    esac
    for cand in $order; do
        [ -x "$cand" ] && X86_BIN=$cand && break
    done
fi
if [ -z "$X86_BIN" ] || [ ! -x "$X86_BIN" ]; then
    echo "SKIP: no native x86 binary to compare against (build with 'make native' or set X86_BIN)" >&2
    exit 2
fi

# ---- cross-build the ARM side -------------------------------------------
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

# ---- run both sides with the tiers pinned --------------------------------
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

X86_DUMP="$WORK/x86.logits"
ARM_DUMP="$WORK/arm.logits"

# ASKED FOR, not transcribed. Every AUTO-resolved tier must be pinned:
# each resolves from hardware, the predicate answers differently on ARM
# than on x86, and an unpinned one makes this comparison compare nothing.
#
# The list is NOT a literal here, because a literal goes stale silently:
# one that pinned only --kernel and --recur-write, against four more AUTO
# knobs sharing the same predicate, made this gate report 94505 of 131072
# bytes differing for that reason alone. lz_tier_pins() in ops.c is the
# single home, so adding a knob updates this script with no edit here.
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
        --tokens "$TOKENS" --dump-logits "$X86_DUMP" \
        >"$WORK/x86.log" 2>&1; then
    echo "FAIL: x86 run failed:" >&2
    cat "$WORK/x86.log" >&2
    exit 1
fi

echo "-- ARM under QEMU (arm926): $ARM_BIN --"
if ! BIN="$ARM_BIN" SYSROOT="$SYSROOT" "$HERE/run_qemu.sh" "$MODEL" \
        $PINS \
        --tokens "$TOKENS" --dump-logits "$ARM_DUMP" \
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

if cmp -s "$X86_DUMP" "$ARM_DUMP"; then
    echo "PASS: x86 and ARM --dump-logits are bit-identical ($X86_SZ bytes)"
    exit 0
else
    NDIFF=$(cmp -l "$X86_DUMP" "$ARM_DUMP" | wc -l)
    echo "FAIL: x86/ARM logits differ - $NDIFF of $X86_SZ bytes differ" >&2
    cmp "$X86_DUMP" "$ARM_DUMP" >&2
    exit 1
fi
