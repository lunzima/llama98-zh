#!/bin/bash
# armv5telint: ARMv5TE optimization linter, first pass.
#
# WHAT THIS IS. A peephole census over the engine's ARM build, in the
# spirit of gaul/armlint and CSharperMantle/loonglint but at the source
# level of what actually matters here: the hand-written asm kernels are
# already optimal (the comments carry per-instruction counts), so the
# suboptimal sequences this finds are the ones gcc emits for the C tier
# - the code path that runs when a kernel's asm cell is not selected.
# The reading is "here a hand-written cell would win", not "gcc is
# wrong": gcc cannot emit ldm/smlatt from the C tier because int16_t
# promises only 2-byte alignment and gcc cannot prove the 4-byte
# contract the asm cells assert (see src/ops_arm.h).
#
# WHAT IT CHECKS, first pass, two patterns, both provable given the
# 4-byte contract:
#
#   ldrh RUN      four or more consecutive ldrh (16-bit loads) can be
#                 one ldm (4 words = 8 halfwords), a 4:1 load saving.
#                 The C tier lz_dot32_q16_arm emits 8 of them.
#   smlabb ALONE  a function full of smlabb with no smlatt/smlatb/
#                 smlabt wastes the top halfword each ldm already
#                 loaded; the asm cells pair BB with TT and halve the
#                 loads (src/ops_arm.c lz_dot32_q16_arm_asm).
#
# THIS IS A CENSUS, NOT A GATE. It prints; it does not judge. A finding
# means "look here", and the look is manual - the rewrite is only
# correct under the 4-byte alignment contract the asm cells assert.
#
# Usage: bash build/armv5telint.sh [SRC...]
set -u
cd "$(dirname "$0")/.."

command -v wsl.exe >/dev/null 2>&1 || { echo "armv5telint: SKIP - no wsl.exe"; exit 2; }

SRC=${*:-}
if [ -z "$SRC" ]; then
    # The C tier of the non-ternary kernels plus the ternary twin.
    SRC="src/ops_arm.c src/ops_t2_arm.c"
fi

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

for f in $SRC; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .c)
    # One WSL invocation per file: compile to /tmp, disassemble to stdout.
    # cd /mnt/e/llama98-zh first so the relative -I and $f resolve.
    wsl.exe -e sh -c "cd /mnt/e/llama98-zh && arm-linux-gnueabi-gcc-14 \
        -march=armv5te -O2 -std=c99 -Isrc -Icommon -c -o /tmp/$base.o $f \
        2>/dev/null && arm-linux-gnueabi-objdump -d /tmp/$base.o 2>/dev/null" \
        > "$W/$base.dis" 2>/dev/null || {
        echo "armv5telint: no build for $f"; continue; }
done

echo "armv5telint: ARMv5TE suboptimal sequences (census)"
echo
echo "  ldrh run >= 4  - candidates for one ldm (4:1 load saving)"
echo "  smlabb alone   - top halfword unused; smlatt pairing would halve loads"
echo

for d in "$W"/*.dis; do
    [ -f "$d" ] || continue
    f=$(basename "$d" .dis)
    awk -v file="$f" '
        /^[0-9a-f]+ <[^>]+>:/ {
            if (run >= 4) printf "  ldrh-run %-2d      %-28s %s\n", run, fn, file
            if (smlabb > 0 && smlatt == 0)
                printf "  smlabb-alone %-2d  %-28s %s\n", smlabb, fn, file
            fn = $0; sub(/^[0-9a-f]+ </, "", fn); sub(/>:.*/, "", fn)
            run = 0; smlabb = 0; smlatt = 0
            next
        }
        /ldrh/ { run++ ; next }
        { if (run >= 4) printf "  ldrh-run %-2d      %-28s %s\n", run, fn, file
          run = 0 }
        /smlabb|smlabt|smlatb|smlatt/ {
            if ($0 ~ /smlabb/) smlabb++
            else smlatt++
        }
        END {
            if (run >= 4) printf "  ldrh-run %-2d  %-28s %s\n", run, fn, file
            if (smlabb > 0 && smlatt == 0)
                printf "  smlabb-alone %-2d %-28s %s\n", smlabb, fn, file
        }
    ' "$d"
done

echo
echo "readings are per C-tier object; a hand-written asm cell at the same"
echo "contract (4-byte aligned) is the rewrite that removes the finding"
