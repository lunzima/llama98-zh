#!/bin/bash
# Zero-%mm gate on src/ops_sse2.c (task #30/#32).
#
# ops_sse2.c's own file contract (src/ops_sse2.h's header comment) is
# %xmm ONLY: SSE2 has full integer support on %xmm, so unlike
# ops_mmx_sse.c (where SSE1-on-%mm mixing is a real, expected thing -
# pmaxsw/pavgb/psadbw genuinely need the MMX register file), any %mm
# instruction reaching this file is a defect, not a contract nuance.
# Task #32 is the concrete way one could appear: LZ_SSE2_GROUP=0 routes
# a format's SSE2 row function through a cross-TU call into
# src/ops_mmx.c's real MMX kernel rather than inlining MMX instructions
# into this file directly - the wrapper pattern (lz_dot128_x16_mmx_w,
# lz_dot128_q41_mmx_w) exists specifically so that routing stays a call,
# never a %mm instruction physically landing here.
#
# Checked on BOTH toolchains, both A/B settings of LZ_SSE2_GROUP:
#   gcc:    objdump -d -M att   (isa_floor.py's own disassemble(), same
#                                 regex convention: aliases_x87 in that
#                                 module is exactly this fact - a %mm
#                                 register touch, independent of ISA
#                                 classification)
#   Watcom: wdis -a             (objdump cannot read Watcom's OMF
#                                 objects - this project's own established
#                                 tool for reading them, see Makefile's
#                                 -3r/-4r codegen-identity comment)
#
# Builds its own throwaway objects rather than depending on `make`
# build order: `make check` does not currently build a gcc object at
# all, and relying on watcom-cli's internal recipe order to have
# produced build/watcom/cli/ops_sse2.obj before this script runs would
# make the gate's correctness depend on Makefile prerequisite ordering,
# which POSIX make does not guarantee for a plain .PHONY prerequisite
# list.
#
# Exit codes:
#   0  PASS - zero %mm on both toolchains, both LZ_SSE2_GROUP settings
#   1  FAIL - a %mm instruction reached src/ops_sse2.c somewhere
#   2  ERROR - a toolchain or disassembler is missing
set -u
cd "$(dirname "$0")/.."

TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
FAIL=0

if [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
    WCC=/c/WATCOM/binnt64/wcc386.exe
    WDIS=/c/WATCOM/binnt64/wdis.exe
elif [ -x /mnt/c/WATCOM/binl64/wcc386 ]; then
    WCC=/mnt/c/WATCOM/binl64/wcc386
    WDIS=/mnt/c/WATCOM/binl64/wdis
else
    echo "ERROR: no Watcom toolchain found (checked binnt64 and binl64)"
    exit 2
fi
if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: gcc not found"
    exit 2
fi
if ! command -v objdump >/dev/null 2>&1; then
    echo "ERROR: objdump not found"
    exit 2
fi

# count_mm FILE LABEL - print the %mm register-operand count for one
# disassembly text file. \bmm[0-7]\b: Watcom's own syntax has no '%'
# prefix (wdis -a emits "movq mm0,mm1"); objdump -M att's does
# ("movq %mm0,%mm1") but \bmm[0-7]\b matches inside either form the
# same way, since '%' is not a word character.
count_mm() {
    local file="$1" label="$2" n
    n=$(grep -oE '\bmm[0-7]\b' "$file" | wc -l)
    echo "  $label: mm=$n"
    if [ "$n" -gt 0 ]; then
        FAIL=1
        echo "  FAIL: $label has $n %mm register operand(s) - showing first 5:"
        grep -nE '\bmm[0-7]\b' "$file" | head -5 | sed 's/^/    /'
    fi
}

for group in 1 0; do
    echo "=== LZ_SSE2_GROUP=$group ==="

    # gcc: portable x86-64, matches native's own flags (MMX_TU/SSE2_TU
    # defines only - never -mmmx/-msse/-msse2 on this file, exactly as
    # the Makefile documents for every plain $(ENG) file).
    gcc -O2 -std=c99 -Isrc -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 \
        -DLZ_SSE2_GROUP=$group -c -o "$TMPD/ops_sse2_gcc.o" src/ops_sse2.c \
        || { echo "ERROR: gcc failed to compile ops_sse2.c"; exit 2; }
    objdump -d -M att "$TMPD/ops_sse2_gcc.o" > "$TMPD/ops_sse2_gcc.dis" 2>&1
    count_mm "$TMPD/ops_sse2_gcc.dis" "gcc"

    # Watcom: -6r matches ops.c/cpucheck.c's HIGH_CPU_FLAGS tier, the
    # only tier ops_sse2.c's own .686-directive kernels need.
    "$WCC" -za99 -otexan -zp4 -D__MMX__=1 -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 \
        -DLZ_SSE2_GROUP=$group -6r -bt=nt -IC:/WATCOM/h -IC:/WATCOM/h/nt \
        -Isrc -fo="$TMPD/ops_sse2_wc.obj" src/ops_sse2.c \
        > "$TMPD/wcc.log" 2>&1
    if [ ! -f "$TMPD/ops_sse2_wc.obj" ]; then
        echo "ERROR: wcc386 failed to compile ops_sse2.c"
        cat "$TMPD/wcc.log"
        exit 2
    fi
    "$WDIS" -a "$TMPD/ops_sse2_wc.obj" > "$TMPD/ops_sse2_wc.dis" 2>&1
    count_mm "$TMPD/ops_sse2_wc.dis" "watcom"
done

if [ "$FAIL" -eq 0 ]; then
    echo "mm_alias_gate: PASS"
    exit 0
else
    echo "mm_alias_gate: FAIL"
    exit 1
fi
