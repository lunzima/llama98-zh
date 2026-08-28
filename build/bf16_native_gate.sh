#!/bin/bash
# bf16 cross-check against the compiler's own __bf16 type.
#
# WHY A SECOND OPINION IS NEEDED. src/lz_bf16.c ships with its own test,
# which compares the ARM assembly fast paths against a C core in the
# same file. That catches a slip in the assembly, but both sides were
# written here and share whatever the author misunderstood about the
# format. __bf16 did not: it is gcc's, lowered either to its own
# f32-and-round sequence or - on a machine with AVX512-BF16 or
# AVX-NE-CONVERT - to vcvtneps2bf16, i.e. to silicon. Agreement with it
# is evidence; agreement with ourselves is not.
#
# WHAT THE SHIPPING BUILD DOES, and it is the opposite of what a reader
# might expect from this gate's existence: EVERY target runs
# src/lz_bf16.c, x86 included. Not because x86 lacks the ability - this
# very gate proves the compiler has it - but because one arithmetic on
# all targets is what makes an ARM run and an x86 run byte-comparable.
# A hardware bf16 path on x86 would round the same way today and
# silently stop when someone builds on a machine without the extension.
# The compiler type is a WITNESS here, never a component.
#
# NaN PAYLOADS ARE COMPARED BY CLASS, not by bits: IEEE 754 leaves the
# payload of a generated NaN to the implementation, so two conforming
# ones may differ there and neither is wrong. Everything else is
# compared bit for bit.
#
# SKIPS (exit 2) rather than fails when the compiler has no __bf16 -
# it needs gcc 13 or newer. A skip here means the check did not run,
# not that it passed.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP.
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "bf16_native_gate: SKIP - no C compiler"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/bf16gate.$$)
mkdir -p "$W"
trap 'rm -rf "$W"' EXIT

# BOTH ARMS OF LZ_BF16_FAST, because the second one had no check at all.
# The knob swaps round-to-nearest-even for truncation toward zero, so the
# probe's equality test could never pass under it and the gate simply
# never built it - leaving a shipping switch whose arithmetic nothing
# verified. The probe now widens its comparison to the truncation band
# when the macro is set (see its same()), so this builds it twice and
# both runs are real assertions.
RC=0
for FAST in 0 1; do
    if [ "$FAST" = 1 ]; then
        echo "-- LZ_BF16_FAST=1 (truncating; checked against RNE +-1ulp toward zero) --"
    else
        echo "-- LZ_BF16_FAST=0 (round-to-nearest-even; checked bit for bit) --"
    fi
    if ! $CC -O2 -std=c99 -Isrc -DLZ_BF16_FAST=$FAST -o "$W/probe" \
            build/bf16_native_probe.c src/lz_bf16.c -lm 2>"$W/cc.log"; then
        echo "bf16_native_gate: SKIP - probe did not build (FAST=$FAST)"
        tail -n 3 "$W/cc.log" | sed 's/^/      /'
        exit 2
    fi
    OUT=$("$W/probe" 2>&1)
    R=$?
    printf '%s\n' "$OUT"
    if [ "$R" -eq 2 ]; then
        echo "bf16_native_gate: SKIP"
        exit 2
    fi
    [ "$R" -ne 0 ] && RC=1
done
if [ "$RC" -ne 0 ]; then
    echo "bf16_native_gate: FAIL"
    exit 1
fi
echo "bf16_native_gate: PASS (both LZ_BF16_FAST arms)"
exit 0
