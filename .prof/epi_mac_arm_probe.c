/* What gcc emits for the epi accumulation on ARMv5TE.
 *
 * The shipping body is static in src/ops.c and inlines into its one
 * caller, so it has no symbol to disassemble. This is the same loop as
 * a standalone function: the count taken from here is the loop gcc
 * emits for this shape, which is what the hand-written kernel in
 * src/ops_arm.c is compared against.
 *
 * Build and read:
 *   arm-linux-gnueabi-gcc-14 -O2 -std=c99 -march=armv5te -mfloat-abi=soft \
 *     -Isrc -c .prof/epi_mac_arm_probe.c -o /tmp/epiprobe.o
 *   arm-linux-gnueabi-objdump -d /tmp/epiprobe.o
 */
#include "lz_int.h"

lz_i64 epi_probe_c(const int *a, const short *m, int n) {
    lz_i64 s = 0;
    int g;
    for (g = 0; g < n; g++) s += (lz_i64)a[g] * (lz_i64)m[g];
    return s;
}
