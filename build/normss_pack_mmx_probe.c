/* Does the MMX pack-and-square half of norm_ss_fixed agree with the C?
 *
 * Same shape as build/q8pack_mmx_probe.c and for the same reason: the
 * cell is a pure function of one int32 array - it writes the clamped
 * int16 and returns the sum of their squares - so it can be answered
 * without a model, without the engine linking and without a tier
 * selector. The float half that feeds it is the caller's x87 and is
 * the C's own arithmetic, so it is not what needs checking here.
 *
 * THE BOUNDARIES, enumerated rather than sampled: packssdw saturates at
 * [-32768, 32767] and the C clamps to [-32767, 32767], so the whole
 * question is the one lane value the two disagree about and the
 * saturation edges either side of it. The sum is checked as well as the
 * lanes - a correct clamp with a wrong pmaddwd pairing would pass a
 * lane-only comparison.
 */
#include <stdio.h>
#include "lz_int.h"

lz_i64 lz_norm_ss_pack_mmx(const int32_t *q, short *qout);

static unsigned long rs = 1145141919UL;
static int32_t nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int32_t)(rs >> 1);
}

int main(void) {
    int32_t in[32];
    short got[32], want[32];
    long bad = 0, n = 0;
    int k, pass;

    static const int32_t edge[] = {
        0, 1, -1, 32766, 32767, 32768, 32769,
        -32766, -32767, -32768, -32769, 65535, -65536,
        2147483647L, -2147483647L - 1
    };
    const int ne = (int)(sizeof edge / sizeof edge[0]);

    for (pass = 0; pass < 200000 + ne; pass++) {
        lz_i64 wsum = 0, gsum;
        for (k = 0; k < 32; k++) {
            if (pass < ne)          in[k] = edge[(pass + k) % ne];
            else if (pass & 1)      in[k] = (nextv() % 70000) - 35000;
            else                    in[k] = nextv();
            {
                int v = (int)in[k];
                v = (v >  32767) ?  32767 : v;
                v = (v < -32767) ? -32767 : v;
                want[k] = (short)v;
                wsum += (lz_i64)(v * v);
            }
        }
        gsum = lz_norm_ss_pack_mmx(in, got);
        for (k = 0; k < 32; k++, n++)
            if (got[k] != want[k]) {
                if (bad < 6)
                    printf("  lane %2d in %11ld want %6d got %6d\n",
                           k, (long)in[k], (int)want[k], (int)got[k]);
                bad++;
            }
        if (gsum != wsum) {
            if (bad < 6)
                printf("  sum want %lld got %lld\n",
                       (long long)wsum, (long long)gsum);
            bad++;
        }
    }

    /* qout NULL must not write and must still sum. */
    for (k = 0; k < 32; k++) in[k] = (int32_t)(k * 1000 - 16000);
    if (lz_norm_ss_pack_mmx(in, (short *)0) == 0) {
        printf("  qout NULL returned 0 for a non-zero row\n");
        bad++;
    }

    printf("normss_pack_mmx_probe: %ld lane(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
