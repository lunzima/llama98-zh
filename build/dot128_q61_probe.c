/* Does the Q6_1 group-of-four kernel agree with four per-32 calls, and
 * with the SSE2 tier?
 *
 * Q6_1 WAS THE LAST DOT FORMAT WITH NO PROBE AT ALL. Not because
 * anyone judged it safe: its per-32 wrapper was `static` and its SSE2
 * group kernel had no wrapper, so there was nothing for a probe in
 * another translation unit to call. Both tiers of the most intricate
 * unpack in the engine were unverified for that reason alone.
 *
 * THE UNPACK IS WHY IT MATTERS. A Q6_1 value is six bits split across
 * two planes: a 4-bit nibble in one, a 2-bit code in the other, and
 * the kernels reassemble them with two masks, four shift levels and a
 * por. Every one of those steps is a place to be one bit wrong in a way
 * that shows only in the elements that level covers.
 *
 * SO THE EXHAUSTIVE ARM DRIVES BOTH PLANES INDEPENDENTLY and does not
 * fill either with a constant: a uniform plane makes the four
 * sub-blocks interchangeable, and then any defect crossing a sub-block
 * boundary is invisible. That is measured behaviour of the x16 and t2
 * probes, whose uniform arms missed their own mutations for all 256
 * exhaustive passes.
 *
 * THE SSE2 ARM IS THE ORACLE. The MMX group kernel and the MMX per-32
 * kernel are file-mates built from the same unpack, so they would agree
 * with each other after a shared mistake. lz_dot128_q61_sse2_w shares
 * no code with either.
 */
#include <stdio.h>
#include "lz_int.h"

void lz_dot128_q61_mmx_w(const unsigned char *w4, const unsigned char *w2,
                         const int16_t *x, int32_t *out4);
int32_t dot32_q61_mmx(const unsigned char *w4, const unsigned char *w2,
                      const int16_t *x);
void lz_dot128_q61_sse2_w(const unsigned char *w4, const unsigned char *w2,
                          const int16_t *x, int32_t *out4);
int32_t dot32_q61_sse2a(const unsigned char *w4, const unsigned char *w2,
                        const int16_t *x);
void dot32_q61_mmx_2(const unsigned char *w4, const unsigned char *w2,
                     const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob);
void dot32_q61_sse2a_2(const unsigned char *w4, const unsigned char *w2,
                       const int16_t *xa, const int16_t *xb,
                       int32_t *oa, int32_t *ob);

static unsigned long rs = 1145141919UL;
static int nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int)(rs >> 1);
}

int main(void) {
    /* ALIGNED VIEWS, not bare arrays: the SSE2 kernel reads both weight
       planes with movdqa and faults on a misaligned address. gcc aligns
       these stack arrays to 16 and hides it; Watcom aligns them to 4, so
       this probe crashed on entry to the SSE2 kernel and had never
       compared the group form on the toolchain the assembly is for. */
    static unsigned char w4buf[64 + 16], w2buf[32 + 16];
    unsigned char *w4 = (unsigned char *)(((size_t)w4buf + 15) & ~(size_t)15);
    unsigned char *w2 = (unsigned char *)(((size_t)w2buf + 15) & ~(size_t)15);
    int16_t x[128];
    int32_t got[4], want[4], s4[4];
    long bad = 0, n = 0;
    int pass, k, b;

    for (pass = 0; pass < 4096; pass++) {
        for (k = 0; k < 64; k++)
            w4[k] = (unsigned char)(pass < 256 ? ((pass + k * 7) & 0xFF)
                                               : (nextv() & 0xFF));
        for (k = 0; k < 32; k++)
            w2[k] = (unsigned char)(pass < 256 ? ((pass * 5 + k * 3) & 0xFF)
                                               : (nextv() & 0xFF));
        for (k = 0; k < 128; k++) {
            if (pass < 256)      x[k] = (int16_t)((k & 1) ? 32767 - k
                                                          : -32767 + k * 3);
            else if (pass < 512) x[k] = (int16_t)(k - 64);
            else                 x[k] = (int16_t)(nextv() % 65536 - 32768);
        }

        lz_dot128_q61_mmx_w(w4, w2, x, got);
        for (b = 0; b < 4; b++)
            want[b] = dot32_q61_mmx(w4 + b * 16, w2 + b * 8, x + b * 32);
        for (b = 0; b < 4; b++, n++) {
            if (got[b] != want[b]) {
                if (bad < 6)
                    printf("  pass %d lane %d: want %ld got %ld\n",
                           pass, b, (long)want[b], (long)got[b]);
                bad++;
            }
        }

        lz_dot128_q61_sse2_w(w4, w2, x, s4);
        for (b = 0; b < 4; b++, n += 2) {
            if (s4[b] != got[b]) {
                if (bad < 6)
                    printf("  pass %d lane %d: mmx %ld sse2 %ld\n",
                           pass, b, (long)got[b], (long)s4[b]);
                bad++;
            }
            if (dot32_q61_sse2a(w4 + b * 16, w2 + b * 8,
                                x + b * 32) != want[b]) {
                if (bad < 6)
                    printf("  pass %d block %d: per-32 tiers differ\n",
                           pass, b);
                bad++;
            }
        }

        /* THE PAIRED KERNELS, both tiers. One weight block against two
           deliberately non-adjacent activation vectors. */
        for (b = 0; b < 2; b++, n += 4) {
            const unsigned char *p4 = w4 + b * 16;
            const unsigned char *p2 = w2 + b * 8;
            const int16_t *xa = x + b * 32;
            const int16_t *xb = x + b * 32 + 64;
            int32_t wb = dot32_q61_mmx(p4, p2, xb);
            int32_t ma = 0, mb = 0, sa = 0, sb = 0;
            dot32_q61_mmx_2(p4, p2, xa, xb, &ma, &mb);
            dot32_q61_sse2a_2(p4, p2, xa, xb, &sa, &sb);
            if (ma != want[b] || mb != wb) {
                if (bad < 6)
                    printf("  pass %d pair %d mmx: got %ld,%ld want %ld,%ld\n",
                           pass, b, (long)ma, (long)mb,
                           (long)want[b], (long)wb);
                bad++;
            }
            if (sa != ma || sb != mb) {
                if (bad < 6)
                    printf("  pass %d pair %d: mmx %ld,%ld sse2 %ld,%ld\n",
                           pass, b, (long)ma, (long)mb, (long)sa, (long)sb);
                bad++;
            }
        }
    }

    /* THE LANES MUST NOT BE INTERCHANGEABLE. */
    {
        int distinct = 1;
        for (k = 0; k < 64; k++)  w4[k] = (unsigned char)(0x1B + k * 3);
        for (k = 0; k < 32; k++)  w2[k] = (unsigned char)(0x4E + k * 5);
        for (k = 0; k < 128; k++) x[k] = (int16_t)(k * 37 - 500);
        lz_dot128_q61_mmx_w(w4, w2, x, got);
        for (b = 0; b < 4; b++)
            want[b] = dot32_q61_mmx(w4 + b * 16, w2 + b * 8, x + b * 32);
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

    printf("dot128_q61_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
