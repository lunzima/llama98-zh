/* Does the Watcom F32 matmul row kernel agree with the scalar contract?
 *
 * The kernel computes a FRESH sum - it zeroes xmm0/xmm1, accumulates
 * row[i+d]*x[i+d] over i in steps of 8 up to in_dim & ~7, and overwrites
 * acc8[0..3]/acc8[4..7]. Each lane d is a separate scalar accumulation
 * (no cross-lane mixing), so the scalar reference below - same i order,
 * same per-d order - is the same number.
 *
 * WATCOM DOES NOT GIVE THE REFERENCE 32-BIT ROUNDING, and this header
 * used to claim -fp6 did. Built by wcc386 the reference differs from the
 * kernel on 33,820 of 160,000 lanes: zero for in_dim 0/1/7/8/9, which
 * accumulate at most once, and rising with the number of accumulation
 * steps (16 -> 3820, 64 -> 7394), evenly across all eight lanes, at one
 * or two ULP. That is an accumulator carried wider than 32 bits, not a
 * lane or length defect.
 *
 * THE KERNEL IS NOT WHAT DIFFERS. Hashing each arm alone, the Watcom
 * kernel and the gcc build both produce B470E23B over all 160,000
 * lanes; only the Watcom-compiled reference stands apart. A
 * kernel-versus-reference count cannot tell those apart, because
 * changing toolchain changes both arms at once.
 *
 * So the verdict is toolchain-dependent and the probe says which it
 * gave: under gcc the count is the verdict, under Watcom the count is
 * reported and not failed and the kernel hash is, against the gcc
 * build's.
 *
 * NOT an accumulate: the gcc twin _mm_setzero_ps()'s and _mm_storeu_ps's,
 * so the reference zeroes acc8 rather than adding to it. The tail beyond
 * in_dim & ~7 is the CALLER's job and neither arm touches it.
 */
#include <stdio.h>
#include <string.h>
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
            /* volatile asks for the product to round to 32-bit, matching
               the kernel's mulps. gcc obliges; Watcom does not carry it
               through the accumulation, which is what the header is
               about - do not read this line as a guarantee. */
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
#ifdef __WATCOMC__
    /* The reference is x87 here, so a difference against it says nothing
       about the kernel. Report what was and was not checked instead of
       returning a verdict this build cannot support. */
    printf("matmul_row_probe: %ld value(s), %ld differ from an x87"
           " reference\n", n, bad);
    printf("  the kernel is UNCHECKED here except through the hash above:"
           " it must\n  equal the gcc build's kernel hash.\n");
    return moved * 2 < n;
#else
    printf("matmul_row_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
#endif
}
