#!/bin/bash
# Kernel coverage gate: a cell per operator per tier per build class,
# and an empty one carries a reason.
#
# The matrix used to be a thing you could PRINT. Nothing read it, so the
# one failure it exists to catch stayed invisible: on 2026-08-09 the
# Watcom build lost every hand-written #pragma aux kernel (no
# -D__MMX__=1) and every numeric gate stayed green, because a kernel
# that falls back to the scalar reference is bit-identical to one that
# does not. A row of dots is what that looks like, and a row of dots is
# also what "nothing to offer here" looks like. This gate is what tells
# them apart.
#
# WHAT IT ASSERTS
#
#   per binary   `--kernel-matrix` exits 0. The binary itself checks its
#                own columns: '!' = registered for this build class and
#                absent, '?' = empty with no registry entry, '+' =
#                present but unregistered. Any of the three is non-zero.
#   agreement    every consulted binary reports the SAME cells with the
#                SAME `have` mask. One registry, or the union is two
#                different tables added together.
#   union        every cell the registry says some class carries is 'x'
#                in that class's binary. A SINGLE binary cannot show
#                this: the assembly columns are empty on gcc and the
#                intrinsics columns are empty on Watcom, both correctly.
#                That is exactly why 2026-08-09 went unnoticed.
#   the rest     every cell no class carries is 'i' (impossible), 't'
#                (todo) or 'd' (written, measured, not better) in every
#                binary, never blank.
#
# gcc and Watcom are MANDATORY - a union of one is not a union. ARM is
# optional and skipping it is announced with the cells that go unchecked,
# never silently.
#
# Exit codes:
#   0  PASS  (possibly with the ARM arm skipped - it says so)
#   1  FAIL
#   2  ERROR - a mandatory toolchain or build is missing
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)

MAKE=${MAKE:-make}
FAIL=0
SKIPPED=""

say()  { echo "$@"; }
bad()  { echo "kernel-matrix-gate: FAIL $*"; FAIL=1; }

# ---- collect one binary's registry --------------------------------------
# Files the cells under $WORK/<class>.cells, one line per cell:
#   <fmt>/<tier> <status> <have>
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

CLASSES=""

# collect LABEL OUTFILE RC - vet one already-captured `--kernel-matrix`
# run and file its cells under the class letter it reports.
collect() {
    local label="$1" out="$2" rc="$3"
    local cls n
    if [ "$rc" -ne 0 ]; then
        bad "$label: --kernel-matrix exited $rc"
        grep '^kernel-matrix: FAIL' "$out" | sed 's/^/    /'
        return 1
    fi
    if ! grep -q '^cells ' "$out"; then
        bad "$label: no 'cells' summary line - is this binary current?"
        return 1
    fi
    n=$(awk '/^cells /{print $2}' "$out")
    cls=$(awk '/^cells /{print $4}' "$out")
    case "$cls" in
        G|W|A) ;;
        *) bad "$label: unknown build class '$cls'"; return 1 ;;
    esac
    # The five row-kernel units x ten tiers are PINNED - they are the
    # grid's shape and the training tree's parser depends on it. The
    # total is a floor, not an equality, because operator units join the
    # registry without touching this gate (src/ops.c's km_unit). Both
    # are here because a loop that stops early shrinks the total AND
    # drops pairs, and every other check would still pass.
    if [ "$n" -lt 50 ]; then
        bad "$label: reported $n cells, expected at least 50"
        return 1
    fi
    grep '^cell ' "$out" | awk '{printf "%s/%s %s %s %s\n", $2, $3, $4, $5, $6}' \
        > "$WORK/$cls.cells"
    for u in q8_0 q4_1 q6_1 q16_0 t2; do
        for t in ref mmxI sseI sse2I mmxA sseA sse2A armC armA g128; do
            if ! grep -q "^$u/$t " "$WORK/$cls.cells"; then
                bad "$label: no cell for $u/$t - the registry lost a row kernel"
                return 1
            fi
        done
        if ! grep -q "^$u/ref .* row\$" "$WORK/$cls.cells"; then
            bad "$label: unit $u is not kind 'row' - the grid shape moved"
            return 1
        fi
    done
    say "  $label: class $cls, $(wc -l < "$WORK/$cls.cells") cells, exit 0"
    CLASSES="$CLASSES $cls"
    return 0
}

# run_matrix LABEL CMD... - capture `CMD --kernel-matrix` and vet it.
run_matrix() {
    local label="$1"; shift
    "$@" --kernel-matrix >"$WORK/$label.out" 2>&1
    collect "$label" "$WORK/$label.out" $?
}

# ---- gcc (class G) ------------------------------------------------------
# `make llama98` is a real file target with real prerequisites, so this
# is free when it is up to date and a rebuild when it is not. Reusing
# whatever ./llama98 happens to be would be the "you ran the old binary"
# trap: a gate whose whole job is to notice a vanished kernel must not
# read a binary from before it vanished.
say "-- gcc build (class G) --"
if ! command -v gcc >/dev/null 2>&1; then
    echo "kernel-matrix-gate: ERROR gcc not found - the intrinsics columns"
    echo "  (mmxI, sse2I) can only be read from a gcc binary"
    exit 2
fi
if ! "$MAKE" llama98 >"$WORK/gcc.build.log" 2>&1; then
    echo "kernel-matrix-gate: ERROR 'make llama98' failed"
    tail -n 15 "$WORK/gcc.build.log"
    exit 2
fi
GCC_BIN="$ROOT/llama98.exe"
[ -x "$GCC_BIN" ] || GCC_BIN="$ROOT/llama98"
if [ ! -x "$GCC_BIN" ]; then
    echo "kernel-matrix-gate: ERROR 'make llama98' succeeded but no binary"
    exit 2
fi
run_matrix gcc "$GCC_BIN" || true

# ---- Open Watcom (class W) ----------------------------------------------
# Built here rather than relied on from `check`'s prerequisite list:
# POSIX make does not order .PHONY prerequisites, and mm_alias_gate.sh
# made the same call for the same reason. 3 seconds.
say "-- Watcom build (class W) --"
if ! "$MAKE" watcom-cli >"$WORK/wat.build.log" 2>&1; then
    echo "kernel-matrix-gate: ERROR 'make watcom-cli' failed"
    tail -n 15 "$WORK/wat.build.log"
    exit 2
fi
WAT_BIN="$ROOT/llama98-w.exe"
if [ ! -f "$WAT_BIN" ]; then
    echo "kernel-matrix-gate: ERROR watcom-cli built nothing at $WAT_BIN"
    exit 2
fi
run_matrix watcom "$WAT_BIN" || true

# ---- ARMv5TE (class A), optional ----------------------------------------
# The cross toolchain lives in WSL on this host and on PATH on a Linux
# one. Absent is a SKIP that names what stops being checked, never a
# silent pass - build/arm/parity_gate.sh sets the same convention.
# The slow arm: 23 cross-compiles plus a QEMU run, ~25s here. A knob
# rather than a default-off, because default-off is how a column stops
# being checked without anyone deciding to stop checking it.
say "-- ARM cross build (class A) --"
# ARM_RAN says the arm produced a verdict, PASS or FAIL. Tracked apart
# from the verdict itself so a failing ARM run is not also announced as
# "no toolchain" - a failure and an absence must not print alike.
ARM_RAN=0
if [ "${LZ_KM_SKIP_ARM:-0}" != "0" ]; then
    SKIPPED="arm (LZ_KM_SKIP_ARM=1)"
    say "  SKIP: LZ_KM_SKIP_ARM=1. The armC/armA columns go unchecked."
fi
arm_try_local() {
    command -v arm-linux-gnueabi-gcc-14 >/dev/null 2>&1 ||
    command -v arm-linux-gnueabi-gcc >/dev/null 2>&1
}
if [ -n "$SKIPPED" ]; then
    :
elif arm_try_local && command -v qemu-arm-static >/dev/null 2>&1; then
    ACC=$(command -v arm-linux-gnueabi-gcc-14 || command -v arm-linux-gnueabi-gcc)
    if CC="$ACC" OUT="$WORK/arm" bash build/arm/build_engine.sh \
           >"$WORK/arm.build.log" 2>&1 && [ -x "$WORK/arm/llama98-arm" ]; then
        ARM_RAN=1
        run_matrix arm qemu-arm-static -cpu arm926 \
                   -L /usr/arm-linux-gnueabi "$WORK/arm/llama98-arm" || true
    else
        ARM_RAN=1
        bad "ARM cross-build failed - see $WORK/arm.build.log"
        tail -n 10 "$WORK/arm.build.log" | sed 's/^/    /'
    fi
elif command -v wsl >/dev/null 2>&1; then
    # $ROOT is Git Bash shaped (/e/llama98-zh); WSL needs /mnt/e/...
    WSL_ROOT=$ROOT
    case "$ROOT" in
        /[a-z]/*) WSL_ROOT="/mnt/$(echo "$ROOT" | cut -c2)/$(echo "$ROOT" | cut -c4-)" ;;
    esac
    if wsl -e bash -lc "command -v arm-linux-gnueabi-gcc-14 >/dev/null && command -v qemu-arm-static >/dev/null" >/dev/null 2>&1; then
        if wsl -e bash -lc "cd '$WSL_ROOT' && CC=arm-linux-gnueabi-gcc-14 OUT=/tmp/kmgate-arm bash build/arm/build_engine.sh" \
               >"$WORK/arm.build.log" 2>&1; then
            wsl -e bash -lc "qemu-arm-static -cpu arm926 -L /usr/arm-linux-gnueabi /tmp/kmgate-arm/llama98-arm --kernel-matrix" \
                >"$WORK/arm.raw" 2>&1
            ARC=$?
            ARM_RAN=1
            collect arm "$WORK/arm.raw" "$ARC" || true
        else
            ARM_RAN=1
            bad "ARM cross-build under WSL failed - see $WORK/arm.build.log"
            tail -n 10 "$WORK/arm.build.log" | sed 's/^/    /'
        fi
    fi
fi
if [ "$ARM_RAN" -eq 0 ] && [ -z "$SKIPPED" ]; then
    SKIPPED="arm (no cross toolchain)"
    say "  SKIP: no ARM cross toolchain (arm-linux-gnueabi-gcc + qemu-arm-static,"
    say "        natively or in WSL). The armC/armA columns go unchecked."
fi

[ "$FAIL" -eq 0 ] || { echo "kernel_matrix_gate: FAIL"; exit 1; }

# ---- mandatory classes --------------------------------------------------
for need in G W; do
    case " $CLASSES " in
        *" $need "*) ;;
        *) bad "no class-$need binary was consulted - a union of one is not a union" ;;
    esac
done
[ "$FAIL" -eq 0 ] || { echo "kernel_matrix_gate: FAIL"; exit 1; }

# ---- agreement: one registry, not several -------------------------------
REF=""
for c in $CLASSES; do
    if [ -z "$REF" ]; then REF=$c; continue; fi
    # Compare cell name, have mask and unit kind (fields 1, 3, 4);
    # status is field 2, per build, and is supposed to differ.
    if ! diff <(awk '{print $1, $3, $4}' "$WORK/$REF.cells" | sort) \
              <(awk '{print $1, $3, $4}' "$WORK/$c.cells" | sort) >"$WORK/agree.diff"; then
        bad "class $REF and class $c report different registries:"
        head -n 8 "$WORK/agree.diff" | sed 's/^/    /'
    fi
done

# ---- union + reasons ----------------------------------------------------
# One pass over the reference registry. For each cell: if `have` names a
# class, that class's binary must say 'x'; if `have` names none, every
# binary must say 'i' or 't'.
COVERED=0
UNCHECKED=""
while read -r cell _status have _kind; do
    case "$have" in
        ---)
            for c in $CLASSES; do
                s=$(awk -v k="$cell" '$1 == k {print $2}' "$WORK/$c.cells")
                case "$s" in
                    i|t|d) ;;
                    *) bad "$cell is empty everywhere but class $c reports '$s'," \
                           "not a reason letter" ;;
                esac
            done
            ;;
        *)
            for letter in G W A; do
                case "$have" in *"$letter"*) ;; *) continue ;; esac
                case " $CLASSES " in
                    *" $letter "*)
                        s=$(awk -v k="$cell" '$1 == k {print $2}' "$WORK/$letter.cells")
                        if [ "$s" = "x" ]; then
                            COVERED=$((COVERED + 1))
                        else
                            bad "$cell is registered for class $letter but that" \
                                "binary reports '$s' - the path is gone"
                        fi
                        ;;
                    *) UNCHECKED="$UNCHECKED $cell/$letter" ;;
                esac
            done
            ;;
    esac
done < "$WORK/$REF.cells"

say "-- union --"
say "  covered: $COVERED (unit x tier x build class)"
say "  registered empty: $(awk '$2 == "i"' "$WORK/$REF.cells" | wc -l) impossible," \
    "$(awk '$2 == "t"' "$WORK/$REF.cells" | wc -l) todo," \
    "$(awk '$2 == "d"' "$WORK/$REF.cells" | wc -l) declined" \
    "(per binary; the reasons are in --kernel-matrix)"
if [ -n "$UNCHECKED" ]; then
    say "  UNCHECKED (no binary of that class):$UNCHECKED"
fi

if [ "$FAIL" -ne 0 ]; then
    echo "kernel_matrix_gate: FAIL"
    exit 1
fi
if [ -n "$SKIPPED" ]; then
    echo "kernel_matrix_gate: PASS (skipped: $SKIPPED)"
else
    echo "kernel_matrix_gate: PASS"
fi
exit 0
