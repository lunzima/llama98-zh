/* Do the SSE1 vmax and vscale helpers agree with the scalar contract?
 *
 * WHY THESE TWO AND WHY NOW. build/kernel_probe_census.sh walks the call
 * graph from every probe down to every #pragma aux body and reports what
 * nothing reaches. Of 74 kernels it found seven, and five of those have
 * nothing to compare - two read CPUID, three are prefetches, which
 * produce no observable value at all. lz_vmax_sse_asm and
 * lz_vscale_sse_asm are the other two: they compute values, they have
 * gcc twins in src/ops_kernel_norm.h, and until this file nothing ran
 * either of them on either toolchain.
 *
 * THE TWO CONTRACTS, from src/ops_mmx_sse.c:
 *   vmax    *pmax = max(*pmax, x[0..n4*4)). It SEEDS from the incoming
 *           value rather than from -inf, so the caller can fold several
 *           spans - and a kernel that ignored the seed would agree with
 *           a reference that also started at -inf. The sweep therefore
 *           passes seeds both below and above the data.
 *   vscale  x[i] *= *pk, in place. n4 is 4-float blocks; the tail past
 *           n4*4 belongs to the caller and neither arm may touch it.
 *
 * WHAT WOULD HIDE A DEFECT, and is avoided:
 *   - a max that is always in the first block: the position is swept.
 *   - all-positive data: max over |x| and max over x differ only when
 *     something is negative, and a kernel that took the wrong one would
 *     otherwise agree.
 *   - a scale of 1.0: it makes every lane-permutation and every
 *     wrong-lane store invisible. pk is random per pass.
 *   - n4 = 0: both arms must leave the sentinel alone, which is checked
 *     rather than assumed.
 *
 * FLOAT ROUNDING. vscale is one multiply per element, single-rounded on
 * both sides; vmax is a comparison and rounds nothing. Neither
 * accumulates, so neither needs the x87 precision-control field that
 * build/matmul_row_probe.c sets - but it is set anyway, so that this
 * file does not become the one that has to rediscover why the others do.
 *
 * The seed is fixed at 1145141919: two runs compare the same numbers or
 * the comparison is not one.
 */
#include <stdio.h>
#include <string.h>
#ifdef __WATCOMC__
#include <float.h>
#endif
#include "lz_int.h"

void lz_vmax_sse_w(const float *x, int n4, float *pmax);
void lz_vscale_sse_w(float *x, int n4, const float *pk);

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) { rs = rs * 1103515245UL + 12345UL; return rs; }
static float nextf(void) {
    unsigned u = nextu();
    return (float)((int)(u & 0xFFFFFF) - 0x800000) / 4096.0f;
}

static void ref_vmax(const float *x, int n4, float *pmax) {
    float m = *pmax;
    int i, n = n4 * 4;
    for (i = 0; i < n; i++)
        if (x[i] > m) m = x[i];
    *pmax = m;
}

static void ref_vscale(float *x, int n4, const float *pk) {
    float k = *pk;
    int i, n = n4 * 4;
    for (i = 0; i < n; i++) {
        volatile float p = x[i] * k;
        x[i] = p;
    }
}

int main(void) {
    static float buf[4 * (64 + 4)];
    float *x  = (float *)(((size_t)buf + 15) & ~(size_t)15);
    float *a1 = x + 68;
    float *a2 = a1 + 68;
    long bad = 0, n = 0, moved = 0, seeded = 0;
    unsigned long khash = 2166136261UL, rhash = 2166136261UL;
    int pass, i;
    static const int n4s[] = { 0, 1, 2, 3, 5, 8, 16 };

#ifdef __WATCOMC__
    _control87(PC_24, MCW_PC);
#endif
    for (pass = 0; pass < 20000; pass++) {
        int n4 = n4s[pass % 7];
        float k = nextf();
        float m1, m2, seed;
        for (i = 0; i < 64; i++) x[i] = nextf();
        /* Seeds above and below the data in turn: a kernel that dropped
           the incoming value would pass a sweep that only ever seeded
           low, because the data's own max would win either way. */
        seed = (pass & 1) ? 1.0e9f : -1.0e9f;
        m1 = m2 = seed;
        lz_vmax_sse_w(x, n4, &m1);
        ref_vmax(x, n4, &m2);
        if (m1 != m2) {
            if (bad < 6)
                printf("  vmax pass %d n4 %d seed %g: kernel %g ref %g\n",
                       pass, n4, (double)seed, (double)m1, (double)m2);
            bad++;
        }
        if (m1 != seed) seeded++;
        n++;
        { unsigned u; memcpy(&u, &m1, 4);
          khash ^= u; khash *= 16777619UL; khash &= 0xFFFFFFFFUL;
          memcpy(&u, &m2, 4);
          rhash ^= u; rhash *= 16777619UL; rhash &= 0xFFFFFFFFUL; }

        for (i = 0; i < 64; i++) { a1[i] = x[i]; a2[i] = x[i]; }
        for (i = n4 * 4; i < 64; i++) { a1[i] = 12345.0f; a2[i] = 12345.0f; }
        lz_vscale_sse_w(a1, n4, &k);
        ref_vscale(a2, n4, &k);
        for (i = 0; i < 64; i++, n++) {
            if (a1[i] != a2[i]) {
                if (bad < 6)
                    printf("  vscale pass %d n4 %d i %d: kernel %g ref %g\n",
                           pass, n4, i, (double)a1[i], (double)a2[i]);
                bad++;
            }
            if (i < n4 * 4) {
                unsigned uk, ur;
                memcpy(&uk, &a1[i], 4); memcpy(&ur, &a2[i], 4);
                khash ^= uk; khash *= 16777619UL; khash &= 0xFFFFFFFFUL;
                rhash ^= ur; rhash *= 16777619UL; rhash &= 0xFFFFFFFFUL;
                if (a1[i] != x[i]) moved++;
            } else if (a1[i] != 12345.0f) {
                /* Past n4*4 is the caller's. Writing there is a defect
                   the value comparison above would MISS, because both
                   arms would have to write it wrong the same way. */
                printf("  vscale wrote past n4*4 at i %d (n4 %d)\n", i, n4);
                bad++;
            }
        }
    }

    /* Sensitivity, both halves. Zero moved lanes and zero seeds beaten
       are what a pair of do-nothing kernels looks like, and they read
       identically to agreement. */
    printf("  %ld scaled lane(s) changed, %ld pass(es) where the data"
           " beat the seed\n", moved, seeded);
    if (moved == 0 || seeded == 0) {
        printf("  one of the two kernels never did anything; the"
               " agreement above is empty\n");
        bad++;
    }
    printf("  kernel hash %08lX, reference hash %08lX\n", khash, rhash);
    printf("vmax_vscale_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
