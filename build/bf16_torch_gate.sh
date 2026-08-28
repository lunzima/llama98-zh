#!/bin/bash
# A THIRD opinion on src/lz_bf16.c, from outside this machine's toolchain.
#
# The file already has two: its own C core (its self-test) and the
# compiler's __bf16 (build/bf16_native_gate.sh). Both of those are gcc on
# this host, so a misreading of the FORMAT - rather than of the
# arithmetic - could satisfy them both and still be wrong. PyTorch's
# c10::BFloat16 was written by other people, ships to millions, and is
# the implementation the rest of the world compares against. Agreement
# with it is evidence of a different kind, not more of the same.
#
# It is also the cheap version of a plan that would have been expensive.
# The obvious third reference is QEMU's fpu/softfloat.c, which is the
# only complete and testable bf16 definition in existence - but its
# bfloat16_* functions sit on the whole FloatParts64 machinery, so using
# them means vendoring a large slice of QEMU. torch is already installed
# and is just as independent for this purpose.
#
# SKIPS (exit 2) rather than fails when torch is absent. A skip means
# the check did not run, not that it passed - which is why this is
# ci.sh-only and bf16_native_gate.sh, whose witness ships with the
# compiler, is the one in `make check`.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP.
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "bf16_torch_gate: SKIP - no C compiler" >&2; exit 2; }
PY=$(command -v python3 || command -v python) || {
    echo "bf16_torch_gate: SKIP - no python" >&2; exit 2; }
"$PY" -c "import torch" >/dev/null 2>&1 || {
    echo "bf16_torch_gate: SKIP - torch is not installed; lz_bf16.c was" \
         "NOT compared against a third implementation" >&2; exit 2; }

W=$(mktemp -d 2>/dev/null || echo /tmp/bf16torch.$$)
mkdir -p "$W"
trap 'rm -rf "$W"' EXIT

if ! $CC -O2 -std=c99 -Isrc -o "$W/probe" \
        build/bf16_torch_probe.c src/lz_bf16.c 2>"$W/cc.log"; then
    echo "bf16_torch_gate: SKIP - probe did not build" >&2
    tail -n 3 "$W/cc.log" | sed 's/^/      /'
    exit 2
fi
if ! "$W/probe" "$W/cases.bin"; then
    echo "bf16_torch_gate: FAIL - the probe did not emit its cases"
    exit 1
fi

OUT=$("$PY" build/bf16_torch_check.py "$W/cases.bin" 2>&1)
RC=$?
printf '%s\n' "$OUT"
if [ "$RC" -eq 2 ]; then echo "bf16_torch_gate: SKIP"; exit 2; fi
if [ "$RC" -ne 0 ]; then echo "bf16_torch_gate: FAIL"; exit 1; fi
echo "bf16_torch_gate: PASS"
exit 0
