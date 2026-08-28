/* accf[d] += (float)acc32[d], 32 elements, on every tier that has one.
 *
 * WHY THIS EXISTS. build/kernel_probe_census.sh counts a #pragma aux
 * kernel as reached only if some driver builds the probe that reaches it
 * with WATCOM, because gcc compiles no pragma body: a gcc-only probe
 * reaches such a kernel on paper and executes nothing. Two were left in
 * that column - lz_i32f_acc32_sse2_asm and lz_i32f_acc4_sse_asm - and
 * the column exists because lz_gdn1_x4_asm sat in the REACHED one while
 * accumulating into an uninitialised mm7.
 *
 * TWO TIERS, TWO SHAPES. lz_i32f_acc32_sse does four elements per call
 * of a 12-instruction SSE1 body and loops eight times; lz_i32f_acc32_simd
 * does all 32 in one SSE2 body. Same contract, and it is a contract
 * about ROUNDING as much as arithmetic: one int32->float32 conversion
 * and one float add per element, each rounded once.
 *
 * WHAT WOULD HIDE A DEFECT, and is therefore avoided:
 *   - accumulators starting at zero: `0 + x` is exact whatever the
 *     rounding, so a wrong add would still agree. accf starts non-zero
 *     and at a magnitude where the add rounds.
 *   - int32 values under 2^24: those convert exactly, and a conversion
 *     that rounded the wrong way would agree anyway. The sweep drives
 *     values that need rounding, INT32_MIN and INT32_MAX included.
 *   - a lane permutation: every element gets a distinct value, so
 *     swapping two is visible.
 *
 * THE TWO TIERS ARE THE PRIMARY COMPARISON, not the tiers against the
 * scalar reference, and that is deliberate. The reference is compiled by
 * whichever toolchain built the probe, and under Watcom it is x87:
 * _control87(PC_24) fixes that for build/matmul_row_probe.c and does NOT
 * fix it here, so the reference differs from both kernels in the last
 * bit on about 3% of values. Which arm is wrong is not left to argument
 * - each is hashed separately, and the Watcom kernels, the gcc kernels
 * and the gcc reference all produce 6538995D while only the
 * Watcom-compiled reference stands apart.
 *
 * So: sse against sse2 is asserted on both toolchains and is a
 * twin-against-twin comparison with no reference in it; the reference is
 * asserted where it is sound and reported where it is not; and the
 * kernel hash is what relates the two builds. build/norm_sse_parity_gate.sh
 * is the driver that does that last part.
 *
 * Seed fixed at 1145141919.
 */
#include <stdio.h>
#include <string.h>
#ifdef __WATCOMC__
#include <float.h>
#endif
#include "lz_int.h"

void lz_i32f_acc32_sse(float *accf, const int32_t *acc32);
void lz_i32f_acc32_simd(float *accf, const int32_t *acc32);

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) { rs = rs * 1103515245UL + 12345UL; return rs; }

static void ref32(float *accf, const int32_t *acc32) {
    int d;
    for (d = 0; d < 32; d++) {
        volatile float c = (float)acc32[d];
        volatile float s = accf[d] + c;
        accf[d] = s;
    }
}

int main(void) {
    static int32_t q[32];
    static float a0[32], a1[32], a2[32], ar[32];
    unsigned long h1 = 2166136261UL, h2 = 2166136261UL, hr = 2166136261UL;
    long n = 0, bad = 0, moved = 0, rounded = 0, refdiff = 0;
    int pass, d;

#ifdef __WATCOMC__
    _control87(PC_24, MCW_PC);
#endif
    for (pass = 0; pass < 20000; pass++) {
        for (d = 0; d < 32; d++) {
            unsigned long u = nextu();
            /* A quarter of the passes drive the extremes; the rest span
               the whole int32 range, most of which does not fit in a
               float32 mantissa and therefore rounds. */
            if ((pass & 3) == 0)
                q[d] = (d & 1) ? (int32_t)0x7FFFFFFFL : (int32_t)0x80000000L;
            else
                q[d] = (int32_t)(u ^ (u << 13));
            /* Non-zero, and at a magnitude where adding q[d] rounds. */
            a0[d] = (float)((int)(u % 65537u) - 32768) * 1024.0f + 0.5f;
            /* A DATA PROPERTY, not a compiler one. Counting inputs by
               `(int32_t)(float)q[d] != q[d]` measured how the toolchain
               folds a cast round-trip - it read 541,921 under gcc and 0
               under Watcom on the same numbers. |q| >= 2^24 cannot be
               held exactly by a float32 and is a fact about q alone. */
            if (q[d] >= 16777216L || q[d] <= -16777216L) rounded++;
        }
        memcpy(a1, a0, sizeof a0);
        memcpy(a2, a0, sizeof a0);
        memcpy(ar, a0, sizeof a0);

        lz_i32f_acc32_sse(a1, q);
        lz_i32f_acc32_simd(a2, q);
        ref32(ar, q);

        for (d = 0; d < 32; d++, n++) {
            unsigned u1, u2, ur;
            /* TIER AGAINST TIER: no reference in it, so it holds on
               either toolchain. This is the assertion. */
            if (a1[d] != a2[d]) {
                if (bad < 4)
                    printf("  TIERS DIFFER pass %d d %d: sse %g sse2 %g\n",
                           pass, d, (double)a1[d], (double)a2[d]);
                bad++;
            }
            if (a1[d] != ar[d]) refdiff++;
            if (a1[d] != a0[d]) moved++;
            memcpy(&u1, &a1[d], 4); memcpy(&u2, &a2[d], 4);
            memcpy(&ur, &ar[d], 4);
            h1 ^= u1; h1 *= 16777619UL; h1 &= 0xFFFFFFFFUL;
            h2 ^= u2; h2 *= 16777619UL; h2 &= 0xFFFFFFFFUL;
            hr ^= ur; hr *= 16777619UL; hr &= 0xFFFFFFFFUL;
        }
    }

    /* Sensitivity. A pair of do-nothing kernels agrees with a reference
       that also did nothing, and prints the same green. */
    printf("  %ld of %ld lane(s) changed, %ld input(s) needed rounding\n",
           moved, n, rounded);
    if (moved * 2 < n || rounded == 0) {
        printf("  the sweep did not exercise what it claims to\n");
        bad++;
    }
    printf("  sse hash %08lX, sse2 hash %08lX, reference hash %08lX\n",
           h1, h2, hr);
    if (h1 != h2) {
        printf("  the two TIERS disagree with each other\n");
        bad++;
    }
#ifdef __WATCOMC__
    /* The reference is x87 here and PC_24 does not bring it into line -
       see the header. Reported, not asserted; the kernel hash above is
       what the driver compares against the gcc build's. */
    if (refdiff)
        printf("  %ld lane(s) differ from the x87 reference - reported,"
               " not failed\n", refdiff);
#else
    if (refdiff) {
        printf("  %ld lane(s) differ from the scalar reference\n", refdiff);
        bad++;
    }
#endif
    printf("i32f_acc_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
