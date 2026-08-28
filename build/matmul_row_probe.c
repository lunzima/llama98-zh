/* Does the Watcom F32 matmul row kernel agree with the scalar contract?
 *
 * The kernel computes a FRESH sum - it zeroes xmm0/xmm1, accumulates
 * row[i+d]*x[i+d] over i in steps of 8 up to in_dim & ~7, and overwrites
 * acc8[0..3]/acc8[4..7]. Each lane d is a separate scalar accumulation
 * (no cross-lane mixing), so the scalar reference below - same i order,
 * same per-d order - is the same number.
 *
 * THE X87 PRECISION-CONTROL FIELD IS WHAT MAKES THAT TRUE ON WATCOM,
 * and `volatile` is not enough. Volatile forces the 32-bit STORE; the
 * operation still runs at the x87 default of a 64-bit mantissa, so each
 * accumulate rounds twice - 64 bits then 24 - where the kernel's addps
 * rounds once. Without _control87(PC_24) the reference disagreed with
 * the kernel on 33,820 of 160,000 lanes: zero for in_dim 0/1/7/8/9,
 * which accumulate at most once, rising with the number of accumulate
 * steps (16 -> 3820, 64 -> 7394), even across all eight lanes, one or
 * two ULP. With PC_24 it is 0, and the header used to blame -fp6, which
 * is not the knob and changes nothing.
 *
 * The per-arm hashes stay, because a kernel-versus-reference count
 * cannot tell "the kernel differs" from "the reference differs" -
 * changing toolchain changes both arms at once, and reading "passes
 * under gcc, fails under Watcom" as a statement about the kernel is how
 * this was first mis-diagnosed. The kernel hash is a function of the
 * kernel alone and must be equal in both builds: B470E23B.
 *
 * NOT an accumulate: the gcc twin _mm_setzero_ps()'s and _mm_storeu_ps's,
 * so the reference zeroes acc8 rather than adding to it. The tail beyond
 * in_dim & ~7 is the CALLER's job and neither arm touches it.
 */
#include <stdio.h>
#include <string.h>
#ifdef __WATCOMC__
#include <float.h>      /* _control87 / PC_24 - see the header */
#endif
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
            /* volatile rounds the PRODUCT to 32-bit, matching mulps. It
               does nothing for the accumulate below - see the header:
               that one needs the x87 precision-control field. */
            volatile float prod = row[i + d] * x[i + d];
            acc8[d] += prod;
        }
}

int main(void) {
    /* ALIGNED VIEWS, not bare arrays: the kernel loads row and x with
       movaps, which faults on a misaligned address. gcc aligns these
       stack arrays to 16 and hides the requirement; Watcom aligns them
       to 4, so this probe died on entry and the F32 row kernel had no
       Watcom-side check at all. Same defect as dot128_x16/q41/q61. */
    static float buf[4 * (64 + 4)];
    float *row = (float *)(((size_t)buf + 15) & ~(size_t)15);
    float *x  = row + 68;
    float *a1 = x + 68;
    float *a2 = a1 + 68;
    long bad = 0, n = 0, moved = 0;
    /* Hashed per arm, not just compared: see the header. */
    unsigned long khash = 2166136261UL, rhash = 2166136261UL;
    int pass, d;
    static const int dims[] = { 0, 1, 7, 8, 9, 16, 24, 31, 32, 40, 64 };

#ifdef __WATCOMC__
    /* Round every x87 operation to a 24-bit mantissa, so the reference's
       accumulate rounds once like addps instead of twice. This is the
       whole reason the probe can be run here at all. */
    _control87(PC_24, MCW_PC);
#endif
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
            {   unsigned uk, ur;
                memcpy(&uk, &a1[d], 4); memcpy(&ur, &a2[d], 4);
                khash ^= uk; khash *= 16777619UL; khash &= 0xFFFFFFFFUL;
                rhash ^= ur; rhash *= 16777619UL; rhash &= 0xFFFFFFFFUL;
            }
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
    printf("  kernel hash %08lX, reference hash %08lX\n", khash, rhash);
    printf("matmul_row_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
