/* Does the Watcom F32 matmul row kernel agree with the scalar contract?
 *
 * The kernel computes a FRESH sum - it zeroes xmm0/xmm1, accumulates
 * row[i+d]*x[i+d] over i in steps of 8 up to in_dim & ~7, and overwrites
 * acc8[0..3]/acc8[4..7]. Each lane d is a separate scalar accumulation
 * (no cross-lane mixing), so the scalar reference below - same i order,
 * same per-d order - is the same number, compiled with -fp6 so the float
 * ops are 32-bit like the kernel's mulps/addps rather than x87's 80-bit.
 *
 * NOT an accumulate: the gcc twin _mm_setzero_ps()'s and _mm_storeu_ps's,
 * so the reference zeroes acc8 rather than adding to it. The tail beyond
 * in_dim & ~7 is the CALLER's job and neither arm touches it.
 */
#include <stdio.h>
#include "lz_int.h"

int lz_matmul_row_sse(const float *row, const float *x, int in_dim, float *acc8);

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) { rs = rs * 1103515245UL + 12345UL; return rs; }
static float nextf(void) {
    unsigned u = nextu();
    return (float)((int)(u & 0xFFFFFF) - 0x800000) / 4096.0f;
}

static void ref(const float *row, const float *x, int in_dim, float *acc8) {
    int k = in_dim & ~7, i, d;
    for (d = 0; d < 8; d++) acc8[d] = 0.0f;
    for (i = 0; i < k; i += 8)
        for (d = 0; d < 8; d++) {
            /* volatile forces the product to round to 32-bit, matching
               the kernel's mulps; without it the compiler may keep the
               product in 80-bit and round only at the final add. */
            volatile float prod = row[i + d] * x[i + d];
            acc8[d] += prod;
        }
}

int main(void) {
    float row[64], x[64], a1[8], a2[8];
    long bad = 0, n = 0, moved = 0;
    int pass, d;
    static const int dims[] = { 0, 1, 7, 8, 9, 16, 24, 31, 32, 40, 64 };

    for (pass = 0; pass < 20000; pass++) {
        int dim = dims[pass % 11];
        for (d = 0; d < 64; d++) { row[d] = nextf(); x[d] = nextf(); }
        for (d = 0; d < 8; d++) { a1[d] = 12345.0f; a2[d] = 12345.0f; }
        (void)lz_matmul_row_sse(row, x, dim, a1);
        ref(row, x, dim, a2);
        for (d = 0; d < 8; d++, n++) {
            if (a1[d] != a2[d]) {
                if (bad < 6)
                    printf("  pass %d dim %d d %d: kernel %g ref %g\n",
                           pass, dim, d, (double)a1[d], (double)a2[d]);
                bad++;
            }
            if (a1[d] != 12345.0f) moved++;
        }
    }
    /* Sensitivity: the operator overwrites acc8, so most lanes must move
       off the sentinel for the "0 mismatches" to mean anything. */
    printf("  %ld of %ld lane(s) moved off the sentinel\n", moved, n);
    if (moved < n / 2) {
        printf("  under half the lanes moved; the arms agreed mostly on"
               " the untouched sentinel\n");
        bad++;
    }
    printf("matmul_row_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
