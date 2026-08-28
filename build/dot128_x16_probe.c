/* Does the Q8_0 group-of-four kernel agree with four per-32 calls?
 *
 * Same shape as build/dot128_t2_probe.c and for the same reason: the
 * group form's claim is that it computes the SAME four sums and only
 * finishes them differently, so the reference is the per-32 kernel
 * itself rather than a C model. A C model shares none of the
 * assembly's bugs but also none of its meaning - written from the same
 * wrong idea of the unpack, both would agree and both be wrong.
 *
 * WHAT VARIES. The weights here are plain int8, not packed codes, so
 * there is no shift-level structure to sweep - what matters instead is
 * the SIGN, because the kernel widens bytes with punpcklbw against a
 * zero register and then relies on pmaddwd's signed multiply. A weight
 * of -1 and a weight of 255 are the same byte and must not give the
 * same answer. The first pass block therefore drives every byte value
 * 0..255 through every one of the 32 weight positions.
 *
 * ACTIVATIONS AT THE EXTREMES ON PURPOSE: pmaddwd sums two products
 * into an int32 lane and the kernel accumulates 16 of those per lane
 * before folding. |w| <= 128 and |x| <= 32767 puts one product at
 * 4.19e6 and the running lane sum at 1.34e8, well inside int32 - the
 * sweep exists to show the fold adds the right lanes, not to find an
 * overflow the format cannot reach.
 *
 * WHICH ARM ANSWERS WHICH QUESTION, stated because the exhaustive one
 * does not answer both:
 *
 *   passes 0-255: the WIDENING. Every byte value through every
 *   position, which is where a sign error shows.
 *
 *   passes 256+, fixed seed: WHICH ACTIVATIONS EACH CHUNK MULTIPLIES.
 *   With all 32 weight bytes equal, a chunk collapses to w * (sum of
 *   its four x), so a defect that swaps a chunk's activations for its
 *   neighbour's is invisible whenever those sums agree. This is
 *   measured behaviour of the t2 probe, not a guess - see its comment.
 *   The seeded arm is what covers it.
 *
 *   the tail block: LANE ORDER, on inputs built to make the four sums
 *   distinct and checked to be distinct before being checked correct.
 *
 * AND THE SSE2 CROSS-CHECK, which is the only arm not rooted in the MMX
 * file. The group kernel and the per-32 kernel it is held against live
 * in the same TU with the same body shape, so a change made to both
 * would agree with itself and this probe would report nothing.
 * lz_dot32_x16_sse2 shares no code with either. Bit-identity across
 * tiers is a requirement in its own right, so this arm is not an extra
 * - it is what makes the others safe to trust.
 */
#include <stdio.h>
#include "lz_int.h"

void lz_dot128_x16_mmx_w(const int8_t *w, const int16_t *x, int32_t *out4);
int32_t dot32_x16_mmx(const int8_t *w, const int16_t *x);
int32_t lz_dot32_x16_sse2(const int8_t *w, const int16_t *x);
void dot32_x16_mmx_2(const int8_t *w, const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob);
void lz_dot128_x16_sse2_w(const int8_t *w, const int16_t *x, int32_t *out4);
void dot32_x16_sse2a_2(const int8_t *w, const int16_t *xa, const int16_t *xb,
                       int32_t *oa, int32_t *ob);

static unsigned long rs = 1145141919UL;
static int nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int)(rs >> 1);
}

int main(void) {
    /* THE WEIGHT BUFFER IS AN ALIGNED VIEW, not a bare array. The SSE2
       kernel reads it with movdqa, which faults on a misaligned address.
       gcc aligns a 128-byte stack array to 16 and hides that; Watcom
       aligns it to 4, so this probe died on entry to the SSE2 kernel and
       had never once compared the group form against four per-32 calls
       on the toolchain the assembly exists for. */
    static int8_t wbuf[128 + 16];
    int8_t *w = (int8_t *)(((size_t)wbuf + 15) & ~(size_t)15);
    int16_t x[128];
    int32_t got[4], want[4];
    long bad = 0, n = 0, same_pair = 0;
    int pass, k, b;

    for (pass = 0; pass < 4096; pass++) {
        for (k = 0; k < 128; k++) {
            if (pass < 256) w[k] = (int8_t)(unsigned char)pass;
            else            w[k] = (int8_t)(nextv() & 0xFF);
        }
        for (k = 0; k < 128; k++) {
            if (pass < 256)      x[k] = (int16_t)((k & 1) ? 32767 : -32767);
            else if (pass < 512) x[k] = (int16_t)(k - 64);
            else                 x[k] = (int16_t)(nextv() % 65536 - 32768);
        }

        lz_dot128_x16_mmx_w(w, x, got);
        for (b = 0; b < 4; b++)
            want[b] = dot32_x16_mmx(w + b * 32, x + b * 32);

        for (b = 0; b < 4; b++, n++) {
            if (got[b] != want[b]) {
                if (bad < 6)
                    printf("  pass %d lane %d: want %ld got %ld\n",
                           pass, b, (long)want[b], (long)got[b]);
                bad++;
            }
        }

        /* THE MMX PER-32 KERNEL AGAINST THE SSE2 ONE, which is the only
           check here that does not depend on the MMX file being right.
           Everything above holds the group kernel to the per-32 kernel,
           and the two live in the same TU with the same body shape - a
           change made to both would agree with itself. SSE2 shares no
           code with either, and the requirement is bit-identity across
           tiers anyway, so this is the oracle rather than a bonus. */
        for (b = 0; b < 4; b++, n++) {
            int32_t s2 = lz_dot32_x16_sse2(w + b * 32, x + b * 32);
            if (s2 != want[b]) {
                if (bad < 6)
                    printf("  pass %d block %d: mmx %ld sse2 %ld\n",
                           pass, b, (long)want[b], (long)s2);
                bad++;
            }
        }

        /* THE SSE2 GROUP KERNEL, against the MMX group kernel directly.
           The two tiers must agree bit for bit - that is a standing
           requirement, not something this probe invented - and holding
           group to group also covers the FOLD, which the per-32
           comparisons above cannot see at all: they never exercise a
           four-sub-block reduction. */
        {
            int32_t s4[4];
            lz_dot128_x16_sse2_w(w, x, s4);
            for (b = 0; b < 4; b++, n++) {
                if (s4[b] != got[b]) {
                    if (bad < 6)
                        printf("  pass %d lane %d: mmx %ld sse2 %ld\n",
                               pass, b, (long)got[b], (long)s4[b]);
                    bad++;
                }
            }
        }

        /* THE PAIRED KERNEL, held to SSE2 as well, and to SSE2 rather
           than to its own single-token twin for the same reason: they
           are neighbours in one file and share a body shape. Its claim
           is one weight unpack serving two DIFFERENT activation
           vectors, so the two must be different or the check cannot
           tell a kernel that reads xa twice from one that is right. */
        for (b = 0; b < 2; b++, n += 2) {
            /* ONE weight block, TWO different activation vectors -
               that is the kernel's whole claim, so xb is deliberately
               not adjacent to xa. */
            const int8_t  *wp = w + b * 32;
            const int16_t *xa = x + b * 32;
            const int16_t *xb = x + b * 32 + 64;
            int32_t pa = 0, pb = 0;
            int32_t wa = lz_dot32_x16_sse2(wp, xa);
            int32_t wb = lz_dot32_x16_sse2(wp, xb);
            dot32_x16_mmx_2(wp, xa, xb, &pa, &pb);
            if (pa != wa || pb != wb) {
                if (bad < 6)
                    printf("  pass %d pair %d: got %ld,%ld want %ld,%ld\n",
                           pass, b, (long)pa, (long)pb,
                           (long)wa, (long)wb);
                bad++;
            }
            /* The SSE2 twin of the same paired kernel, which had no
               address until this round either. */
            {
                int32_t s2a = 0, s2b = 0;
                dot32_x16_sse2a_2(wp, xa, xb, &s2a, &s2b);
                n += 2;
                if (s2a != pa || s2b != pb) {
                    if (bad < 6)
                        printf("  pass %d pair %d: mmx %ld,%ld sse2 %ld,%ld\n",
                               pass, b, (long)pa, (long)pb,
                               (long)s2a, (long)s2b);
                    bad++;
                }
            }
            /* If the two sums are equal the case cannot tell the two
               outputs apart, so count them: an arm made entirely of
               those would pass a kernel that wrote oa into both. */
            if (wa == wb) same_pair++;
        }
    }

    /* THE LANES MUST NOT BE INTERCHANGEABLE, or a fold that transposed
       wrongly would pass everything above. */
    {
        int distinct = 1;
        for (k = 0; k < 128; k++) w[k] = (int8_t)(k - 60);
        for (k = 0; k < 128; k++) x[k] = (int16_t)(k * 37 - 500);
        lz_dot128_x16_mmx_w(w, x, got);
        for (b = 0; b < 4; b++)
            want[b] = dot32_x16_mmx(w + b * 32, x + b * 32);
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

    /* The paired arm is only as good as the fraction of its cases where
       the two outputs actually differ. Reported rather than assumed. */
    printf("  paired arm: %ld of 8192 case(s) had oa == ob and could not"
           " tell the two outputs apart\n", same_pair);
    if (same_pair > 4096) {
        printf("  paired arm is mostly blind: over half its cases have"
               " equal outputs\n");
        bad++;
    }

    printf("dot128_x16_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
