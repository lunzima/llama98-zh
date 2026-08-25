/* Do the two tiers of the attention pair MAC agree, byte for byte?
 *
 * lz_wsum_pair_asm (MMX) and lz_wsum_pair_sse2_asm (SSE2) compute the
 * same thing and are required to agree bit for bit. The SSE2 side has
 * had an addressable wrapper since the tier split; the MMX side did
 * not, so the comparison could not be made. It can now.
 *
 * THE OPERATION ACCUMULATES. acc32 is read as well as written, so both
 * tiers get the same starting contents on every pass - a probe that
 * zeroed one and not the other would report a difference that is its
 * own.
 *
 * THERE IS A REAL ORACLE HERE, unlike the p2_split32 probe: the
 * contract is written down at the kernel's declaration -
 *
 *     acc32[d] += rowA[d]*coef[0] + rowB[d]*coef[1], d in [0,32)
 *     coef is [ckA,ckB,ckA,ckB]
 *
 * - so this checks both tiers against that formula in C as well as
 * against each other. A declared contract is an oracle; a shape
 * inferred from the output is not.
 *
 * COEF MUST BE DUPLICATED, and that is the precondition the first
 * version of this probe broke. It filled four independent values, and
 * the tiers then disagreed on half the lanes - correctly, because they
 * read different amounts of it: MMX takes all four with movq, SSE2
 * takes the first two with movd and broadcasts them with pshufd.
 * 131,072 "mismatches" that were a probe violating a documented
 * requirement it had not read.
 *
 * SIXTEEN-BYTE ALIGNED, because the SSE2 body uses movdqa twelve
 * times. Alignment is a precondition of the operator, not a nicety: a
 * misaligned buffer faults rather than answering wrongly.
 */
#include <stdio.h>
#include "lz_int.h"

void lz_wsum_pair_mmx_w(const int8_t *rowA, const int8_t *rowB,
                        const int16_t *coef, int32_t *acc32);
void lz_wsum_pair_sse2(const int8_t *rowA, const int8_t *rowB,
                       const int16_t *coef, int32_t *acc32);

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) {
    rs = rs * 1103515245UL + 12345UL;
    return rs;
}

static char arena[512 + 15];
static void *aligned(int off) {
    unsigned base = (unsigned)((unsigned long)(void *)arena & 15u);
    return arena + ((16u - base) & 15u) + off;
}

int main(void) {
    int8_t *ra, *rb;
    int16_t *cf;
    int32_t *a1, *a2, seed[32];
    long bad = 0, n = 0, moved = 0;
    int pass, k;

    ra = (int8_t *)aligned(0);        /*  32 bytes */
    rb = (int8_t *)aligned(32);       /*  32 bytes */
    cf = (int16_t *)aligned(64);      /*   8 bytes */
    a1 = (int32_t *)aligned(96);      /* 128 bytes */
    a2 = (int32_t *)aligned(224);     /* 128 bytes */

    for (pass = 0; pass < 8192; pass++) {
        for (k = 0; k < 32; k++) {
            if (pass < 256) {
                /* Both rows through every byte value, and the two rows
                   deliberately different: a kernel that read rowA twice
                   would pass on equal rows. */
                ra[k] = (int8_t)(unsigned char)((pass + k) & 0xFF);
                rb[k] = (int8_t)(unsigned char)((pass * 3 + k * 5) & 0xFF);
            } else {
                ra[k] = (int8_t)(unsigned char)(nextu() & 0xFF);
                rb[k] = (int8_t)(unsigned char)(nextu() & 0xFF);
            }
        }
        /* [ckA, ckB, ckA, ckB] - the layout the declaration requires.
           Four independent values is not a harder test, it is a
           different operator. */
        cf[0] = cf[2] = (int16_t)(pass < 256 ? (pass * 131 - 16384)
                                             : (int)(nextu() % 65536) - 32768);
        cf[1] = cf[3] = (int16_t)(pass < 256 ? (pass * 37 + 999 - 16384)
                                             : (int)(nextu() % 65536) - 32768);
        /* THE SAME STARTING ACCUMULATOR for both tiers, and not always
           zero: this operator accumulates, so a defect that ignored the
           incoming value would be invisible on a zeroed buffer. */
        for (k = 0; k < 32; k++) {
            seed[k] = (pass & 1) ? 0 : (int32_t)nextu();
            a1[k] = seed[k];
            a2[k] = seed[k];
        }

        lz_wsum_pair_mmx_w(ra, rb, cf, a1);
        lz_wsum_pair_sse2(ra, rb, cf, a2);

        for (k = 0; k < 32; k++, n += 2) {
            /* The declared contract, in C. int32 wraparound is what
               both kernels do on overflow, and it is what this does. */
            int32_t want = (int32_t)(seed[k]
                          + (int32_t)ra[k] * (int32_t)cf[0]
                          + (int32_t)rb[k] * (int32_t)cf[1]);
            if (a1[k] != a2[k]) {
                if (bad < 6)
                    printf("  pass %d lane %d: mmx %ld sse2 %ld\n",
                           pass, k, (long)a1[k], (long)a2[k]);
                bad++;
            }
            if (a1[k] != want) {
                if (bad < 6)
                    printf("  pass %d lane %d: mmx %ld, contract says %ld\n",
                           pass, k, (long)a1[k], (long)want);
                bad++;
            }
            if (a1[k] != seed[k]) moved++;
        }
    }

    /* Sensitivity: lanes the operator actually changed. Two tiers that
       both wrote nothing would agree perfectly. */
    printf("  %ld of %ld lane(s) differ from the seeded accumulator\n",
           moved, n);
    if (moved < n / 4) {
        printf("  under a quarter of the lanes moved; the tiers agreed"
               " mostly on untouched memory\n");
        bad++;
    }
    printf("wsum_pair_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
