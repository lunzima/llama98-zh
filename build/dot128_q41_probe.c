/* Does the Q4_1 group-of-four kernel agree with four per-32 calls, and
 * with the SSE2 tier?
 *
 * THE SSE2 ARM IS THE ORACLE, not a bonus. lz_dot32_q41_asm and
 * lz_dot128_q41_asm live in one file with the SAME loop body - the
 * group kernel is literally the per-32 one four times over with grown
 * offsets - so a change made to both would leave them agreeing with
 * each other and this probe reporting nothing. That is what happened to
 * the x16 pair when the same reordering was applied there; see
 * build/pair_census.sh. lz_dot128_q41_sse2_w shares no code with
 * either, and MMX/SSE2 bit-identity is a standing requirement anyway.
 *
 * WHAT VARIES. Q4_1 packs two 4-bit codes per byte and the kernel
 * splits them with pand/psrlw against one mask, so every byte value
 * must reach every weight position: an error in the high-nibble path
 * shows only in the 16 elements that path covers. The first arm drives
 * all 256 byte values through all 64 weight bytes.
 *
 * THE EXHAUSTIVE ARM IS DELIBERATELY NOT UNIFORM. Filling every weight
 * byte with one value makes the four sub-blocks interchangeable, and
 * then any defect that crosses sub-block boundaries is invisible -
 * measured on the x16 and t2 probes, where exactly that arm missed
 * their mutations for all 256 passes. Here the value varies by position
 * as well as by pass.
 *
 * ACTIVATIONS REACH +-32767 so one pmaddwd lane holds two products at
 * 15*32767; eight of those per lane stay inside int32, so the sweep
 * tests the fold rather than the format's overflow behaviour.
 */
#include <stdio.h>
#include "lz_int.h"

void lz_dot128_q41_mmx_w(const unsigned char *w, const int16_t *x,
                         int32_t *out4);
int32_t dot32_q41_mmx(const unsigned char *w, const int16_t *x);
void lz_dot128_q41_sse2_w(const unsigned char *w, const int16_t *x,
                          int32_t *out4);
void dot32_q41_mmx_2(const unsigned char *w, const int16_t *xa,
                     const int16_t *xb, int32_t *oa, int32_t *ob);
void dot32_q41_sse2a_2(const unsigned char *w, const int16_t *xa,
                       const int16_t *xb, int32_t *oa, int32_t *ob);

static unsigned long rs = 1145141919UL;
static int nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int)(rs >> 1);
}

int main(void) {
    /* An ALIGNED VIEW, not a bare array: the SSE2 kernel reads the
       weights with movdqa and faults on a misaligned address. gcc aligns
       a 64-byte stack array to 16 and hides it; Watcom aligns it to 4,
       so this probe crashed on entry to the SSE2 kernel and had never
       compared the group form on the toolchain the assembly is for. */
    static unsigned char wbuf[64 + 16];
    unsigned char *w = (unsigned char *)(((size_t)wbuf + 15) & ~(size_t)15);
    int16_t x[128];
    int32_t got[4], want[4], s4[4];
    long bad = 0, n = 0;
    int pass, k, b;

    for (pass = 0; pass < 4096; pass++) {
        for (k = 0; k < 64; k++) {
            if (pass < 256) w[k] = (unsigned char)((pass + k * 7) & 0xFF);
            else            w[k] = (unsigned char)(nextv() & 0xFF);
        }
        for (k = 0; k < 128; k++) {
            if (pass < 256)      x[k] = (int16_t)((k & 1) ? 32767 - k
                                                          : -32767 + k * 3);
            else if (pass < 512) x[k] = (int16_t)(k - 64);
            else                 x[k] = (int16_t)(nextv() % 65536 - 32768);
        }

        lz_dot128_q41_mmx_w(w, x, got);
        for (b = 0; b < 4; b++)
            want[b] = dot32_q41_mmx(w + b * 16, x + b * 32);
        for (b = 0; b < 4; b++, n++) {
            if (got[b] != want[b]) {
                if (bad < 6)
                    printf("  pass %d lane %d: want %ld got %ld\n",
                           pass, b, (long)want[b], (long)got[b]);
                bad++;
            }
        }

        lz_dot128_q41_sse2_w(w, x, s4);
        for (b = 0; b < 4; b++, n++) {
            if (s4[b] != got[b]) {
                if (bad < 6)
                    printf("  pass %d lane %d: mmx %ld sse2 %ld\n",
                           pass, b, (long)got[b], (long)s4[b]);
                bad++;
            }
        }

        /* THE PAIRED KERNELS, both tiers, first test either has had.
           One weight block, two deliberately different activation
           vectors: a kernel that read xa twice passes otherwise. */
        for (b = 0; b < 2; b++, n += 4) {
            const unsigned char *wp = w + b * 16;
            const int16_t *xa = x + b * 32;
            const int16_t *xb = x + b * 32 + 64;
            int32_t wb = dot32_q41_mmx(wp, xb);
            int32_t ma = 0, mb = 0, sa = 0, sb = 0;
            dot32_q41_mmx_2(wp, xa, xb, &ma, &mb);
            dot32_q41_sse2a_2(wp, xa, xb, &sa, &sb);
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
        for (k = 0; k < 64; k++)  w[k] = (unsigned char)(0x1B + k * 3);
        for (k = 0; k < 128; k++) x[k] = (int16_t)(k * 37 - 500);
        lz_dot128_q41_mmx_w(w, x, got);
        for (b = 0; b < 4; b++)
            want[b] = dot32_q41_mmx(w + b * 16, x + b * 32);
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

    printf("dot128_q41_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
