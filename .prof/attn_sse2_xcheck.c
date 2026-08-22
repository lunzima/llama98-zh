/* Attention's two fixed-point MACs: SSE2 tier against MMX against C.
 *
 * Both cells were empty in the kernel matrix with the reason "both call
 * sites reach dot32_x16_mmx / lz_wsum_pair_mmx unconditionally, so no
 * SSE2 twin was ever selected - unaudited, not impossible". This is the
 * gate for the twins that reason is now retired by.
 *
 * THREE ARMS, not two. The C reference matters here in a way it does not
 * for the float kernels: this is the one place where "bit-identical by
 * construction" is a real argument (integer sums are exact under
 * reassociation, so ANY grouping gives the same int32), which means a
 * wrong LANE ORDER is the failure mode left, and a wrong lane order
 * produces a plausible number. So the sweep is built to make lane order
 * observable: every element gets a distinct value, and the coefficients
 * differ between the two rows.
 *
 * WHAT WOULD HIDE A LANE BUG and is therefore avoided:
 *   - rowA == rowB, or ckA == ckB: the pair MAC becomes symmetric
 *   - constant rows: every lane holds the same product
 *   - coefficients of 0 or 1: the multiply stops carrying information
 * Each of those appears in the sweep too, but only alongside the
 * asymmetric cases - as boundary values, never as the whole test.
 *
 * The dot's x256 fold is checked implicitly: it is inside the kernel
 * (punpcklbw puts the weight in the HIGH byte) and outside the C
 * reference (which writes the *256 explicitly), so an arm that dropped
 * it would differ by a factor of 256 on every value.
 *
 * Saturation is NOT swept, because there is none to sweep: with |w|<=127,
 * |x|<=32767 the 32-element dot caps near 1.3e8 and the pair MAC near
 * 8.3e6 per element, both far inside int32. The engine's real bound is
 * the CHUNK length, which lz_attn_wsum_q8 owns, not these kernels.
 *
 * WHAT THIS FILE DOES NOT COVER, and where that is covered instead:
 * it calls the three bodies DIRECTLY, so it says nothing about whether
 * --kernel sse2 selects the SSE2 one. That is a claim about the shipped
 * dispatch and it belongs to a gate that runs the shipped dispatch -
 * build/kernel_isa_identity_gate.sh, which since these cells were
 * written no longer compares two names for one body here: mmx and sse2
 * now execute different attention MACs, and mutating the SSE2 body
 * moves that gate. Measured, not assumed; see this probe's driver.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ops.h"
#include "ops_mmx.h"
#include "ops_sse2.h"
#include "mmx_compat.h"

static uint32_t rs = 0x1234567u;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

/* The two C references, copied from src/ops_gdn.c's non-MMX branch -
 * deliberately copied rather than called, so a change to the shipped
 * reference cannot silently move both sides of this comparison. */
static int32_t dot_ref(const int8_t *w, const int16_t *x) {
    int32_t acc = 0;
    int k;
    for (k = 0; k < 32; k++) acc += (int32_t)w[k] * 256 * (int32_t)x[k];
    return acc;
}
static void wsum_ref(const int8_t *rowA, const int8_t *rowB,
                     const int16_t *coef, int32_t *acc32) {
    int32_t ckA = coef[0], ckB = coef[1];
    int d;
    for (d = 0; d < 32; d++)
        acc32[d] += (int32_t)rowA[d] * ckA + (int32_t)rowB[d] * ckB;
}

static int8_t  wA[32], wB[32];
static int16_t xv[32], coef[4];

static void fill(int pat) {
    int i;
    for (i = 0; i < 32; i++) {
        switch (pat) {
        case 0:                              /* every lane distinct */
            wA[i] = (int8_t)(i - 16);
            wB[i] = (int8_t)(31 - i - 16);
            xv[i] = (int16_t)((i + 1) * 101);
            break;
        case 1:                              /* extremes */
            wA[i] = (int8_t)((i & 1) ? 127 : -128);
            wB[i] = (int8_t)((i & 3) ? -127 : 127);
            xv[i] = (int16_t)((i & 1) ? 32767 : -32768);
            break;
        case 2:                              /* rowA == rowB (boundary) */
            wA[i] = wB[i] = (int8_t)(rnd() & 0xFF);
            xv[i] = (int16_t)(rnd() & 0xFFFF);
            break;
        case 3:                              /* sign-only */
            wA[i] = (int8_t)((i % 3) - 1);
            wB[i] = (int8_t)(1 - (i % 3));
            xv[i] = (int16_t)((i % 5) - 2);
            break;
        default:
            wA[i] = (int8_t)(rnd() & 0xFF);
            wB[i] = (int8_t)(rnd() & 0xFF);
            xv[i] = (int16_t)(rnd() & 0xFFFF);
            break;
        }
    }
}

int main(int argc, char **argv) {
    static const int16_t CK[][2] = {
        { 1, 1 }, { 0, 1 }, { 1, 0 }, { 0, 0 },
        { 32767, -32768 }, { -32768, 32767 }, { 7, -13 }, { -1, -1 }
    };
    uint32_t h = 2166136261u;
    int pat, ci, i, r, bad = 0;
    (void)argc; (void)argv;

    for (r = 0; r < 4; r++) {                 /* four passes, fresh rng */
        for (pat = 0; pat < 6; pat++) {
            fill(pat);
            for (ci = 0; ci < (int)(sizeof(CK) / sizeof(CK[0])); ci++) {
                int32_t ref[32], amx[32], as2[32];
                int32_t d_r, d_m, d_s;
                coef[0] = CK[ci][0]; coef[1] = CK[ci][1];
                coef[2] = CK[ci][0]; coef[3] = CK[ci][1];
                /* Non-zero starting accumulator: the kernels ADD in
                   place, and starting from zero would not catch a
                   kernel that stored instead of accumulating. */
                for (i = 0; i < 32; i++)
                    ref[i] = amx[i] = as2[i] = (i + 1) * 7 - 100;

                wsum_ref(wA, wB, coef, ref);
#if defined(__WATCOMC__)
                /* The MMX pair MAC has no exported symbol on this
                   toolchain: lz_wsum_pair_asm is a #pragma aux whose
                   only call site is src/ops_mmx.c's own group loop, so
                   there is nothing to call from here. The MMX arm is
                   dropped rather than faked - it stays covered by the
                   engine-level tier comparison. */
                memcpy(amx, ref, sizeof(ref));
#else
                lz_wsum_pair_mmx(wA, wB, coef, amx);
                _mm_empty();          /* the MMX body leaves %mm live */
#endif /* __WATCOMC__ */
                lz_wsum_pair_sse2(wA, wB, coef, as2);
                d_r = dot_ref(wA, xv);
                d_m = dot32_x16_mmx(wA, xv);
                _mm_empty();
                d_s = lz_dot32_x16_sse2(wA, xv);

                for (i = 0; i < 32; i++) {
                    h = (h ^ (uint32_t)as2[i]) * 16777619u;
                    if (as2[i] != ref[i] || amx[i] != ref[i]) {
                        printf("MISMATCH wsum pat=%d ck=%d lane=%d "
                               "sse2=%ld mmx=%ld ref=%ld\n", pat, ci, i,
                               (long)as2[i], (long)amx[i], (long)ref[i]);
                        bad = 1;
                    }
                }
                h = (h ^ (uint32_t)d_s) * 16777619u;
                if (d_s != d_r || d_m != d_r) {
                    printf("MISMATCH dot pat=%d ck=%d sse2=%ld mmx=%ld "
                           "ref=%ld\n", pat, ci,
                           (long)d_s, (long)d_m, (long)d_r);
                    bad = 1;
                }
                if (bad) return 1;
            }
        }
    }
    _mm_empty();
    printf("ATTNSSE2 %08lx\n", (unsigned long)h);
    return 0;
}
