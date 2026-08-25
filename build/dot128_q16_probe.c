/* Does the Q16_0 group-of-four kernel agree with four per-32 calls?
 *
 * Same shape and same reasoning as build/dot128_x16_probe.c: the group
 * form's claim is that it computes the SAME four sums and only walks
 * and finishes them differently, so the reference is the per-32 kernel
 * rather than a C model.
 *
 * WHAT THIS ONE HAS TO CATCH THAT THE OTHERS DID NOT. The group kernel
 * no longer walks its input linearly. It runs the four sub-blocks as
 * four concurrent streams - chunk i of every block before chunk i+1 of
 * any of them - so every weight and activation offset in the body was
 * rewritten. An offset that lands one stream on another's data is the
 * defect this exists for, and it is invisible unless the four
 * sub-blocks hold DIFFERENT values.
 *
 * So the sweep never gives two sub-blocks the same content: the
 * exhaustive arm varies the byte value by position as well as by pass,
 * and the seeded arm is random per element. The lane-order block at the
 * end asserts the four sums are distinct before it asserts they are
 * right.
 *
 * WEIGHTS ARE int16 HERE, not packed codes, so there is no shift level
 * to sweep - what matters is sign and magnitude. |w| and |x| both reach
 * 32767, one pmaddwd lane holds two products at 2^30 each, and 8 of
 * those accumulate per lane: 2^33, which OVERFLOWS int32. That is not a
 * flaw in the probe, it is the format's own bound, and paddd wraps the
 * same way in both kernels - the point of driving it there is that a
 * regrouped accumulation would still have to wrap identically.
 */
#include <stdio.h>
#include "lz_int.h"

void lz_dot128_q16_mmx_w(const int16_t *w, const int16_t *x, int32_t *out4);
int32_t dot32_q16_mmx(const int16_t *w, const int16_t *x);
void lz_dot128_q16_sse2_w(const int16_t *w, const int16_t *x, int32_t *out4);
int32_t dot32_q16_sse2a(const int16_t *w, const int16_t *x);

static unsigned long rs = 1145141919UL;
static int nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int)(rs >> 1);
}

int main(void) {
    int16_t w[128], x[128];
    int32_t got[4], want[4];
    long bad = 0, n = 0;
    int pass, k, b;

    for (pass = 0; pass < 4096; pass++) {
        for (k = 0; k < 128; k++) {
            if (pass < 256) {
                /* Position-dependent, so no two sub-blocks agree. A
                   pass that filled all 128 with one value would make
                   the four streams interchangeable and this arm blind
                   to the thing the kernel just changed. */
                w[k] = (int16_t)((pass << 7) + k * 259);
                x[k] = (int16_t)((k & 1) ? 32767 - k : -32767 + k * 3);
            } else if (pass < 512) {
                w[k] = (int16_t)(k - 64);
                x[k] = (int16_t)(nextv() % 65536 - 32768);
            } else {
                w[k] = (int16_t)(nextv() % 65536 - 32768);
                x[k] = (int16_t)(nextv() % 65536 - 32768);
            }
        }

        lz_dot128_q16_mmx_w(w, x, got);
        for (b = 0; b < 4; b++)
            want[b] = dot32_q16_mmx(w + b * 32, x + b * 32);

        for (b = 0; b < 4; b++, n++) {
            if (got[b] != want[b]) {
                if (bad < 6)
                    printf("  pass %d lane %d: want %ld got %ld\n",
                           pass, b, (long)want[b], (long)got[b]);
                bad++;
            }
        }

        /* THE SSE2 TIER, group and per-32. Q16_0 had no cross-tier
           check at all until the SSE2 group kernel was given an
           address - the MMX side had a wrapper and this one did not,
           so the requirement that the tiers agree bit for bit was
           unmeasured on this format. It is also the only arm here that
           does not depend on ops_mmx.c being right. */
        {
            int32_t s4[4];
            lz_dot128_q16_sse2_w(w, x, s4);
            for (b = 0; b < 4; b++, n += 2) {
                if (s4[b] != got[b]) {
                    if (bad < 6)
                        printf("  pass %d lane %d: mmx %ld sse2 %ld\n",
                               pass, b, (long)got[b], (long)s4[b]);
                    bad++;
                }
                if (dot32_q16_sse2a(w + b * 32, x + b * 32) != want[b]) {
                    if (bad < 6)
                        printf("  pass %d block %d: per-32 tiers differ\n",
                               pass, b);
                    bad++;
                }
            }
        }
    }

    /* THE LANES MUST NOT BE INTERCHANGEABLE, or a fold that transposed
       wrongly - or a stream that read its neighbour - would pass
       everything above. */
    {
        int distinct = 1;
        for (k = 0; k < 128; k++) {
            w[k] = (int16_t)(k * 37 - 500);
            x[k] = (int16_t)(k * 11 + 3);
        }
        lz_dot128_q16_mmx_w(w, x, got);
        for (b = 0; b < 4; b++)
            want[b] = dot32_q16_mmx(w + b * 32, x + b * 32);
        for (b = 1; b < 4; b++) if (want[b] == want[0]) distinct = 0;
        for (b = 0; b < 4; b++, n++) if (got[b] != want[b]) bad++;
        if (!distinct) {
            printf("  lane-order check is blind: the four sums are equal\n");
            bad++;
        } else {
            printf("  lane order: four distinct sums %ld %ld %ld %ld\n",
                   (long)want[0], (long)want[1], (long)want[2], (long)want[3]);
        }
    }

    printf("dot128_q16_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
