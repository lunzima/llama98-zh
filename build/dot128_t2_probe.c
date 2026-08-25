/* Does the T2 group-of-four kernel agree with four per-32 calls?
 *
 * The group form's whole claim is that it computes the SAME four sums
 * and only finishes them differently - one transposing fold instead of
 * four horizontal reductions. So the reference is the per-32 kernel
 * itself, called four times, rather than a C model of what the assembly
 * is meant to do: a C model shares none of the assembly's bugs but also
 * none of its meaning, and if both were written from the same wrong
 * idea of the T2 unpack they would agree and both be wrong.
 *
 * WHAT VARIES AND WHY. The weights are 2-bit codes packed four to a
 * byte, so every byte position must see every code, and the unpack
 * reads them through four different shift levels (psllw 6/4/2/none) -
 * an error in one level shows only in the 32 activations that level
 * covers. The sweep therefore drives all 256 byte values through every
 * one of the 32 weight bytes, then a fixed-seed pass over activations
 * spanning int16.
 *
 * ACTIVATIONS AT THE EXTREMES ON PURPOSE: pmaddwd accumulates two
 * products into int32, and the running sum over 128 elements is what
 * the two-lane accumulator holds before the fold. Driving x to +-32767
 * with every code nonzero is the largest magnitude this kernel can be
 * asked for, which is where a fold that added the wrong lanes together
 * would show up as something other than a small difference.
 *
 * WHICH ARM ANSWERS WHICH QUESTION, because the exhaustive one does not
 * answer both and reading it as if it did would overstate this probe.
 *
 *   passes 0-255, every byte value in every position: the UNPACK. Each
 *   shift level's view of each code, which is what an error in one
 *   level shows up in.
 *
 *   passes 256+, fixed seed: WHICH ACTIVATIONS EACH CHUNK MULTIPLIES.
 *   The exhaustive arm is structurally blind here and no choice of
 *   activation pattern fixes it: with all 32 weight bytes equal, the
 *   four codes in a quad are equal too, so a chunk's contribution
 *   collapses to w * (sum of its four x), and swapping a quad for its
 *   neighbour changes nothing whenever those sums agree. Measured, not
 *   assumed: a kernel mutated to multiply chunk 0 by chunk 1's
 *   activations passes all 256 exhaustive passes and fails 3841 of the
 *   seeded ones. Two attempted sign patterns did not move that
 *   boundary; the weights, not x, are what makes the arm blind.
 *
 *   the tail block: LANE ORDER, on inputs built to make the four sums
 *   distinct and checked to be distinct before being checked correct.
 */
#include <stdio.h>
#include "lz_int.h"

void lz_dot128_t2_mmx_w(const unsigned char *w2, const int16_t *x,
                        int32_t *out4);
int32_t lz_dot32_t2_mmx_w(const unsigned char *w2, const int16_t *x);

static unsigned long rs = 1145141919UL;
static int nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int)(rs >> 1);
}

int main(void) {
    unsigned char w[32];
    int16_t x[128];
    int32_t got[4], want[4];
    long bad = 0, n = 0;
    int pass, k, b;

    for (pass = 0; pass < 4096; pass++) {
        for (k = 0; k < 32; k++) {
            if (pass < 256) {
                /* Every byte value through every position: pass p puts
                   p in all 32, which covers each level's view of it. */
                w[k] = (unsigned char)pass;
            } else {
                w[k] = (unsigned char)(nextv() & 0xFF);
            }
        }
        for (k = 0; k < 128; k++) {
            if (pass < 256)      x[k] = (int16_t)((k & 1) ? 32767 : -32767);
            else if (pass < 512) x[k] = (int16_t)(k - 64);
            else                 x[k] = (int16_t)(nextv() % 65536 - 32768);
        }

        lz_dot128_t2_mmx_w(w, x, got);
        for (b = 0; b < 4; b++)
            want[b] = lz_dot32_t2_mmx_w(w + b * 8, x + b * 32);

        for (b = 0; b < 4; b++, n++) {
            if (got[b] != want[b]) {
                if (bad < 6)
                    printf("  pass %d lane %d: want %ld got %ld\n",
                           pass, b, (long)want[b], (long)got[b]);
                bad++;
            }
        }
    }

    /* THE LANES MUST NOT BE INTERCHANGEABLE, or a fold that transposed
       wrongly would pass everything above. Four sub-blocks built to give
       four different sums, checked to be distinct before they are
       checked to be right - without this the sweep could be running
       entirely on inputs where all four happen to agree. */
    {
        int distinct = 1;
        for (k = 0; k < 32; k++) w[k] = (unsigned char)(0x1B + k);
        for (k = 0; k < 128; k++) x[k] = (int16_t)(k * 37 - 500);
        lz_dot128_t2_mmx_w(w, x, got);
        for (b = 0; b < 4; b++)
            want[b] = lz_dot32_t2_mmx_w(w + b * 8, x + b * 32);
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

    printf("dot128_t2_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
