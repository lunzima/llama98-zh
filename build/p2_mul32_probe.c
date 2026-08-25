/* Do the two tiers of the GDN pass-2 MUL agree, field for field?
 *
 * lz_p2_mul32_mmx_asm and lz_p2_mul32_sse2_asm are the last pair in
 * these files that had never been compared, and the reason is the same
 * as everywhere else: the SSE2 side had no address. Its MMX twin has
 * had lz_p2_mul32_mmx since the tier split.
 *
 * THIS ONE WRITES INTO THE BLOCK rather than returning anything - a[32]
 * and the amax accumulator - so the comparison is over struct fields.
 * That is also why it is last: every other probe here could diff a
 * returned buffer.
 *
 * WHAT IT MEASURES is agreement between two hand-written bodies over
 * different register files. There is no declared formula for this one
 * the way there is for wsum_pair, and no scalar reference on the
 * Watcom side, so agreement is the claim - stated rather than dressed
 * up as correctness.
 *
 * AMAX ACCUMULATES, so both tiers get the same non-zero starting value
 * on half the passes. A defect that ignored the incoming accumulator
 * would be invisible on a zeroed one.
 *
 * SIXTEEN-BYTE ALIGNED: the SSE2 body reads and writes the block with
 * movdqa. Alignment is a precondition, and a probe that ignores one
 * does not test a weaker version - it faults, or it reports the
 * precondition as a defect. Both have happened in this directory.
 */
#include <stdio.h>
#include "lz_int.h"
#include "ops_p2_blk.h"

void lz_p2_mul32_mmx(const int8_t *hi, const int8_t *lo,
                     const int16_t *dq, lz_p2_blk *blk);
void lz_p2_mul32_sse2_w(const int8_t *hi, const int8_t *lo,
                        const int16_t *dq, lz_p2_blk *blk);

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) {
    rs = rs * 1103515245UL + 12345UL;
    return rs;
}

static char arena[1024 + 15];
static void *aligned(int off) {
    unsigned base = (unsigned)((unsigned long)(void *)arena & 15u);
    return arena + ((16u - base) & 15u) + off;
}

int main(void) {
    int8_t *hi, *lo;
    int16_t *dq;
    lz_p2_blk *b1, *b2;
    long bad = 0, n = 0, moved = 0;
    int pass, k;

    hi = (int8_t *)aligned(0);        /*  32 */
    lo = (int8_t *)aligned(32);       /*  32 */
    dq = (int16_t *)aligned(64);      /*  64 */
    b1 = (lz_p2_blk *)aligned(128);   /* 240 */
    b2 = (lz_p2_blk *)aligned(384);   /* 240 */

    for (pass = 0; pass < 8192; pass++) {
        for (k = 0; k < 32; k++) {
            if (pass < 256) {
                hi[k] = (int8_t)(unsigned char)((pass + k * 3) & 0xFF);
                lo[k] = (int8_t)(unsigned char)((pass * 5 + k) & 0xFF);
                dq[k] = (int16_t)(pass * 127 + k * 61 - 16384);
            } else {
                hi[k] = (int8_t)(unsigned char)(nextu() & 0xFF);
                lo[k] = (int8_t)(unsigned char)(nextu() & 0xFF);
                dq[k] = (int16_t)((int)(nextu() % 65536) - 32768);
            }
        }
        /* mul is "m1,m2 repeated 4x", per ops_p2_blk.h. The two tiers
           read different amounts of it - MMX takes four int16 with
           movq, SSE2 takes all eight with movdqa - so eight
           independent values is not a harder test, it is a different
           operator, and they diverge on it correctly. Third probe in a
           row where the first run's "finding" was a broadcast operand
           I had filled as if it were free. */
        {
            int16_t m1 = (int16_t)((int)(nextu() % 65536) - 32768);
            int16_t m2 = (int16_t)((int)(nextu() % 65536) - 32768);
            for (k = 0; k < 8; k += 2) {
                b1->mul[k] = b2->mul[k] = m1;
                b1->mul[k + 1] = b2->mul[k + 1] = m2;
            }
        }
        for (k = 0; k < 4; k++) {
            b1->amax[k] = (pass & 1) ? 0 : (int32_t)nextu();
            b2->amax[k] = b1->amax[k];
        }
        for (k = 0; k < 32; k++) { b1->a[k] = 0; b2->a[k] = 0; }

        lz_p2_mul32_mmx(hi, lo, dq, b1);
        lz_p2_mul32_sse2_w(hi, lo, dq, b2);

        for (k = 0; k < 32; k++, n++) {
            if (b1->a[k] != b2->a[k]) {
                if (bad < 6)
                    printf("  pass %d a[%d]: mmx %ld sse2 %ld\n",
                           pass, k, (long)b1->a[k], (long)b2->a[k]);
                bad++;
            }
            if (b1->a[k]) moved++;
        }
        for (k = 0; k < 4; k++, n++) {
            if (b1->amax[k] != b2->amax[k]) {
                if (bad < 6)
                    printf("  pass %d amax[%d]: mmx %ld sse2 %ld\n",
                           pass, k, (long)b1->amax[k], (long)b2->amax[k]);
                bad++;
            }
        }
    }

    printf("  %ld of %ld a[] lane(s) came out nonzero\n", moved, n * 8 / 9);
    if (moved < n / 4) {
        printf("  too few lanes did anything; the tiers agreed mostly on"
               " zeros\n");
        bad++;
    }
    printf("p2_mul32_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
