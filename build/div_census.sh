#!/bin/bash
# Divisions the shipping build actually emits, counted as instructions.
#
# WHY NOT GREP THE SOURCE. A '/' in C is a divide, a comment opener, a
# path inside a string and part of '/='; a '%' is a modulo and a printf
# conversion. A regex over ops*.c returns four figures depending on how
# it is written and none of them is the number the CPU sees. More to the
# point, the compiler removes most of them: a divide by a compile-time
# power of two is a shift, a divide by a loop-invariant is hoisted and
# reciprocal-multiplied, and an integer divide by a constant becomes a
# multiply-high. Counting source sites measures a machine nobody ships.
#
# So: compile at the shipping settings and count div/idiv/fdiv/divss/
# divps in .text. What survives here is what the hardware runs.
#
# 32-bit, i486, tuned for an in-order core - the configuration the
# Makefile ships, for the reason in xcc_codesize.sh's header.
#
# THIS IS A CENSUS, NOT A GATE. It prints; it does not judge. A divide
# is not automatically debt: in a function that runs once per model load
# it is free, and the same instruction in a per-element loop is not.
# Where each one sits is the reading, and that needs the function name,
# which is why the per-function breakdown is the second table.
#
# ops.c, the largest count in the set, has been walked. Nothing there is
# a target and the reasons are worth keeping so the next reader does not
# repeat the walk:
#   lz_t_row_f32 6, lz_matmul_xq_nt 6, lz_moe_route 5, and a long tail -
#   none above six, so there is no concentration to attack.
#   The one divide inside a per-element loop is in lz_quantize_q8's
#   `else` arm, `((float)t[k] * deq) / sg`. Its `if` arm already hoists
#   127/tam and multiplies; the divide arm is the OVERFLOW GUARD, taken
#   only when amax is small enough that 127/amax exceeds FLT_MAX.
#   The rest are once per group (127.0f/tam, the Q8RCP reciprocal) or
#   once per call (the `n % gs` defensive checks).
#
# forward.c's 67 are similarly spread: 7 in kv_slot's inlined copies,
# the other ~52 across 48 lines at one to four each.
set -u
cd "$(dirname "$0")/.."
command -v gcc >/dev/null 2>&1 || { echo "SKIP: no gcc"; exit 2; }

DIVRE="^(idiv|div|fdiv|fdivr|fdivp|fdivrp|divss|divsd|divps|divpd)"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
FILES=${*:-"src/ops.c src/ops_gdn.c src/ops_matmul.c src/ops_norm.c
            src/ops_quant.c src/ops_sched.c src/forward.c src/sampler.c
            src/generate.c"}

TOTAL=0
printf '%-16s %6s\n' file divides
for f in $FILES; do
    [ -f "$f" ] || continue
    b=$(basename "$f" .c)
    gcc -O2 -m32 -march=i486 -mtune=pentium -std=c99 -Isrc -Icommon \
        -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -c -o "$W/$b.o" "$f" 2>/dev/null \
        || { printf '%-16s %6s\n' "$b" "(no build)"; continue; }
    objdump -d "$W/$b.o" 2>/dev/null \
        | awk -F'\t' -v f="$b" '
            /^[0-9a-f]+ </ { fn = $0; sub(/^[0-9a-f]+ </, "", fn); sub(/>.*/, "", fn) }
            NF >= 3 { split($3, m, " "); print f, fn, m[1] }' > "$W/$b.txt"
    n=$(awk -v re="$DIVRE" '$3 ~ re { c++ } END { print c + 0 }' "$W/$b.txt")
    TOTAL=$((TOTAL + n))
    printf '%-16s %6d\n' "$b" "$n"
    cat "$W/$b.txt" >> "$W/all.txt"
done
echo
echo "total: $TOTAL"

echo
echo "by function (where it sits is the whole reading):"
awk -v re="$DIVRE" '$3 ~ re { c[$1 "  " $2]++ }
     END { for (k in c) printf "  %-6d %s\n", c[k], k }' "$W/all.txt" \
    | sort -rn
