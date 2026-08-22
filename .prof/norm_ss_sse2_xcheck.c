/* norm_ss_fixed's SSE2 element loop against its C body.
 *
 * TWO THINGS TO PROVE, and only one of them is about SIMD.
 *   cvtps2dq IS q8_round. Both round half to even - the magic add by
 *   construction, the instruction by the default MXCSR mode - so the
 *   kernel is not approximating the rounding, it is spelling it a
 *   second way. Every element in every row below exercises it.
 *   The SUM is the same sum. Integer addition is associative and
 *   exact, so pmaddwd's pairwise order cannot move the total; what a
 *   sweep can still catch is an indexing or widening mistake, and the
 *   pairwise result is 131069 short of INT32_MAX at |v| = 32767, which
 *   is close enough to be worth reaching.
 *
 * So the rows include one scaled so its amax lands exactly on the
 * exponent boundary (every |v| at the top of the Q15 range), one that
 * is all zeros, one with a single huge outlier (which sets the shared
 * scale and crushes everything else to 0), and lengths that are not
 * multiples of 8 so the C tail runs alongside the kernel.
 *
 * qout NULL and non-NULL are separate rows: the kernel branches on it
 * once outside its loop and a mistake there would only show on one.
 *
 * Prints a checksum of every (ss, exponent, Q15 image); the driver
 * compares tiers and builds. */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ops.h"
#include "ops_quant.h"

#define NMAX 4096

static float xs[NMAX];
static short q[NMAX];

static uint32_t rs = 0x9E3779B9u;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

int main(int argc, char **argv) {
    static const int NS[] = { 8, 9, 15, 16, 31, 64, 127, 512, 1000, 4096 };
    uint32_t h = 2166136261u;
    int ni, pat, i, useq;
    if (argc > 1) {
        int t = LZ_KERNEL_REF;
        if (argv[1][0] == 'm') t = LZ_KERNEL_MMX;
        else if (argv[1][0] == 's') t = LZ_KERNEL_SSE2;
        lz_kernel_select(t);
    }
    for (ni = 0; ni < (int)(sizeof(NS) / sizeof(NS[0])); ni++) {
        int n = NS[ni];
        for (pat = 0; pat < 5; pat++) {
            for (i = 0; i < n; i++) {
                switch (pat) {
                case 0: xs[i] = ((float)(int32_t)rnd() / 2147483648.0f) * 4.0f;
                        break;
                case 1: xs[i] = 0.0f; break;
                case 2: xs[i] = (i == n / 2) ? 1e18f : 1.0f; break;
                case 3: xs[i] = (i & 1) ? 1.0f : -1.0f; break;
                default: xs[i] = ((float)(int32_t)rnd() / 2147483648.0f)
                                 * 1e-20f; break;
                }
            }
            for (useq = 0; useq < 2; useq++) {
                union { float f; uint32_t u; } b;
                int e = -1;
                float ss;
                memset(q, 0, sizeof(q));
                ss = norm_ss_fixed(xs, n, useq ? q : (short *)0, &e);
                b.f = ss;
                h = (h ^ b.u) * 16777619u;
                h = (h ^ (uint32_t)e) * 16777619u;
                if (useq)
                    for (i = 0; i < n; i++)
                        h = (h ^ (uint32_t)(int32_t)q[i]) * 16777619u;
                printf("n=%4d pat=%d q=%d  ss=%.9g e=%d\n",
                       n, pat, useq, (double)ss, e);
            }
        }
    }
    printf("NORMSS %08lx\n", (unsigned long)h);
    return 0;
}
