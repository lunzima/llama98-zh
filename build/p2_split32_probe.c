/* Do the three tiers of the GDN pass-2 SPLIT agree, byte for byte?
 *
 * Until this probe none of them could be called. lz_p2_split32_mmx_asm,
 * lz_p2_split32_sse_asm and lz_p2_split32_sse2_asm are each named once,
 * inside the row function that builds the lz_p2_blk around them, so no
 * translation unit but their own could drive a split with chosen input.
 * Three implementations of one operation, required to agree bit for
 * bit, and never compared.
 *
 * WHAT IT MEASURES, precisely: that the three hand-written tiers
 * produce the same 64 bytes. They are not copies of each other - MMX
 * and SSE2 clamp with psubsw/paddsw against blk->kclp, SSE1 with a
 * single pmaxsw against blk->kmin - so agreement is a real statement,
 * unlike the per-32/group pairs elsewhere in these files which are one
 * body written twice.
 *
 * WHAT IT DOES NOT MEASURE: whether all three are right. There is no
 * scalar reference for this operator on the Watcom side; the gcc
 * intrinsics forms (lz_p2_split32_mmx and friends) are compiled only
 * into the other build.
 *
 * AND THE FIRST VERSION OF THIS PROBE ASSERTED ONE ANYWAY - that
 * 256*hi + lo reconstructs a - which is not the contract. `lo` is
 * clamped into Q8's range, so at a = 128 the tiers all return
 * hi = 1, lo = -127 and the reconstruction is 129: LOSSY BY DESIGN.
 * 391 "mismatches" were that invented model, none of them a
 * disagreement between tiers. An oracle guessed from the shape of the
 * output is not an oracle.
 *
 * THE SWEEP DRIVES a THROUGH ITS EDGES, not just its middle: both
 * clamp boundaries in both directions, the rounding tie at 128, zero,
 * the int32 extremes, and a fixed-seed pass over the full range.
 */
#include <stdio.h>
#include "lz_int.h"
#include "ops_p2_blk.h"

void lz_p2_split32_mmx_w(const void *blk, signed char *hi, signed char *lo);
void lz_p2_split32_sse_w(const void *blk, signed char *hi, signed char *lo);
void lz_p2_split32_sse2_w(const void *blk, signed char *hi, signed char *lo);

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) {
    rs = rs * 1103515245UL + 12345UL;
    return rs;
}

/* The edges: both clamp boundaries, the rounding tie, and zero. */
static const int32_t EDGE[] = {
    0, 1, -1, 127, 128, 129, -127, -128, -129,
    32512, 32513, 32640, 32641, 32642,
    -32512, -32513, -32640, -32641, -32642,
    2147483647L, -2147483647L - 1
};

/* SIXTEEN-BYTE ALIGNED, and not as a precaution: the SSE2 tier reads
   this block with movdqa sixteen times, which faults outright on a
   misaligned address. The shipping code allocates it exactly this way
   (p2_blk_sse2 in src/ops_sse2.c). A plain stack local is 4-aligned
   under -zp4 and the first version of this probe crashed on it - the
   alignment is a precondition of the operator, so a probe that ignores
   it does not test a weaker version, it tests nothing. */
static lz_p2_blk *aligned_blk(void) {
    static char raw[sizeof(lz_p2_blk) + 15];
    unsigned off = (unsigned)((unsigned long)(void *)raw & 15u);
    return (lz_p2_blk *)(void *)(raw + ((16u - off) & 15u));
}

int main(void) {
    lz_p2_blk *bp = aligned_blk();
#define blk (*bp)
    signed char h1[32], l1[32], h2[32], l2[32], h3[32], l3[32];
    long bad = 0, n = 0, recon = 0;
    int pass, k, j;
    int nedge = (int)(sizeof EDGE / sizeof EDGE[0]);

    for (j = 0; j < 8; j++) {
        blk.k128[j] = 128;
        blk.kclp[j] = 32641;
        blk.kmin[j] = -127;
        blk.mul[j] = 0;
    }
    for (j = 0; j < 4; j++) { blk.amax[j] = 0; blk.rnd[j] = 0; blk.cnt[j] = 0; }

    for (pass = 0; pass < 4096; pass++) {
        for (k = 0; k < 32; k++) {
            if (pass < 64) {
                /* Every edge value in every lane, and the lane index
                   shifts it so no two lanes hold the same number - a
                   uniform block cannot tell a kernel that processes
                   lane 0 four times from one that is right. */
                blk.a[k] = EDGE[(pass + k) % nedge];
            } else if (pass < 128) {
                blk.a[k] = (int32_t)((pass - 64) * 512 + k * 7 - 16384);
            } else {
                blk.a[k] = (int32_t)nextu();
            }
        }

        lz_p2_split32_mmx_w(&blk, h1, l1);
        lz_p2_split32_sse_w(&blk, h2, l2);
        lz_p2_split32_sse2_w(&blk, h3, l3);

        for (k = 0; k < 32; k++, n += 2) {
            if (h1[k] != h2[k] || l1[k] != l2[k]) {
                if (bad < 6)
                    printf("  pass %d lane %d: mmx %d,%d sse %d,%d\n",
                           pass, k, h1[k], l1[k], h2[k], l2[k]);
                bad++;
            }
            if (h1[k] != h3[k] || l1[k] != l3[k]) {
                if (bad < 6)
                    printf("  pass %d lane %d: mmx %d,%d sse2 %d,%d\n",
                           pass, k, h1[k], l1[k], h3[k], l3[k]);
                bad++;
            }
            /* SENSITIVITY, not correctness: count the lanes where the
               three tiers had a chance to differ at all. A block whose
               outputs are all zero would agree trivially. */
            if (h1[k] || l1[k]) recon++;
        }
    }

    printf("  %ld of %ld lane(s) produced a nonzero split\n", recon, n / 2);
    if (recon < n / 4) {
        printf("  under a quarter of the lanes did anything; the tiers"
               " agreed mostly on zeros\n");
        bad++;
    }
    printf("p2_split32_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
