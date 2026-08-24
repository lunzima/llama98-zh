/* Does the MMX pack half of Q8 rounding agree with the C, exhaustively
 * where it can and densely where it cannot?
 *
 * The cell replaces only the clamp-and-pack: the caller hands it int32
 * that x87 already produced with q8_round(x * inv), and it writes the
 * int8 the C's two compares would have written. So the question is a
 * pure int32 -> int8 function on one input, and it can be answered
 * without a model, without the engine linking, and without a tier
 * selector - which is the point of probing it here rather than through
 * a generation.
 *
 * THE BOUNDARIES ARE WHERE THIS BREAKS, so they are enumerated rather
 * than sampled: packssdw saturates at +-32768 and packsswb at
 * [-128, 127], and the C wants [-127, 127]. Every value within two of
 * each of those edges is tested, plus the int32 extremes, plus a dense
 * pseudo-random sweep at the fixed seed.
 */
#include <stdio.h>
#include <stdlib.h>
#include "lz_int.h"

void lz_q8round32_pack_mmx(const int32_t *q, int8_t *out);

/* The C this must reproduce - lz_quantize_q8's fallback tail. */
static int8_t ref(int32_t qi) {
    int v = (int)qi;
    v = (v > 127) ? 127 : v;
    return (int8_t)((v < -127) ? -127 : v);
}

static unsigned long rs = 1145141919UL;
static int32_t nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int32_t)(rs >> 1);
}

int main(void) {
    int32_t in[32];
    int8_t got[32];
    long bad = 0, n = 0;
    int i, k, pass;

    static const int32_t edge[] = {
        0, 1, -1, 126, 127, 128, 129, -126, -127, -128, -129,
        32766, 32767, 32768, 32769, -32766, -32767, -32768, -32769,
        65535, 65536, -65536, 2147483647L, -2147483647L - 1
    };
    const int ne = (int)(sizeof edge / sizeof edge[0]);

    /* Every edge in every lane position: the pack instructions treat
       lanes independently, but a bug in the shuffle order shows only
       when the lanes differ. */
    for (i = 0; i < ne; i++) {
        for (k = 0; k < 32; k++) in[k] = edge[(i + k) % ne];
        lz_q8round32_pack_mmx(in, got);
        for (k = 0; k < 32; k++, n++)
            if (got[k] != ref(in[k])) {
                if (bad < 8)
                    printf("  lane %2d in %11ld want %4d got %4d\n",
                           k, (long)in[k], (int)ref(in[k]), (int)got[k]);
                bad++;
            }
    }

    /* And a dense sweep, fixed seed, mixing the near-zero band where
       most real values live with the full int32 range. */
    for (pass = 0; pass < 200000; pass++) {
        for (k = 0; k < 32; k++)
            in[k] = (pass & 1) ? (nextv() % 300) - 150 : nextv();
        lz_q8round32_pack_mmx(in, got);
        for (k = 0; k < 32; k++, n++)
            if (got[k] != ref(in[k])) {
                if (bad < 8)
                    printf("  sweep in %11ld want %4d got %4d\n",
                           (long)in[k], (int)ref(in[k]), (int)got[k]);
                bad++;
            }
    }

    printf("q8pack_mmx_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
