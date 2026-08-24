/* What the epi accumulation costs per element on the shipping x86
 * target, scalar against SSE2.
 *
 * The shipping build is 32-bit i486, which is where this has to be
 * counted: the SSE2 body needs three live accumulators plus two
 * constants, and eight XMM registers is not enough for that - an x86-64
 * count would hide the spills as register renames.
 *
 * Build and read:
 *   gcc -O2 -std=c99 -m32 -march=i486 -Isrc -c .prof/epi_mac_x86_probe.c \
 *       -o /tmp/epix86.o
 *   objdump -d /tmp/epix86.o
 *
 * The scalar loop is the one lz_epi_mac_i16 falls back to, and on
 * Watcom it is a #pragma aux (src/ops.c) whose body is nine
 * instructions an element: mov, movsx, imul, add, adc, two pointer
 * bumps, dec, jnz. One-operand imul is a signed 32x32 -> 64 with the
 * result already in edx:eax, so the scalar path needs no split at all -
 * which is the thing that makes this comparison worth doing.
 */
#include "lz_int.h"

lz_i64 epi_probe_x86_c(const int *a, const short *m, int n) {
    lz_i64 s = 0;
    int g;
    for (g = 0; g < n; g++) s += (lz_i64)a[g] * (lz_i64)m[g];
    return s;
}
