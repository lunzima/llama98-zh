/* lz_fwht's SSE1 float butterfly against its C body.
 *
 * The float sibling of .prof/fwht_mmx_xcheck.c, and the claim is the
 * same shape but not the same argument. There the transform is integer
 * and exact by construction; here it is float, and what makes a vector
 * form EXACT rather than close is that no constant enters and each
 * output is one add or one sub of the same two inputs the scalar loop
 * pairs. It is not a reassociation - the pairing is identical, only the
 * order the pairs are visited in changes, and each pair is independent.
 *
 * So the sweep aims at the two things that could still be wrong: the
 * INDEXING (a stride off by one still produces a plausible transform)
 * and the boundary between the stages the kernel takes (len >= 4) and
 * the ones the C keeps (len 1 and 2). Block sizes run from 8 - the
 * smallest the kernel is offered - to 512, the cap subn_fwht_quant's
 * fixed-point headroom imposes.
 *
 * Values include ones where the difference cancels exactly (all equal),
 * where it does not (alternating large/small, which is where a wrong
 * lane shows up instead of hiding), and denormal and huge magnitudes.
 *
 * PC=24 on x87, for the reason .prof/matmul_f32_xcheck.c spells out at
 * length: outside it the C body is not computing in float32 and no
 * float32 kernel can match it.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ops.h"

#define NMAX 512

static float in[NMAX], w[NMAX];

static uint32_t rs = 0x9E3779B9u;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

int main(int argc, char **argv) {
    static const int NS[] = { 8, 16, 32, 64, 128, 256, 512 };
    uint32_t h = 2166136261u;
    unsigned fpu;
    int ni, pat, i;
    if (argc > 1) {
        int t = LZ_KERNEL_REF;
        if (argv[1][0] == 'm') t = LZ_KERNEL_MMX;
        else if (argv[1][0] == 's') t = LZ_KERNEL_SSE2;
        lz_kernel_select(t);
    }
    fpu = lz_fpu_float_begin();
    for (ni = 0; ni < (int)(sizeof(NS) / sizeof(NS[0])); ni++) {
        int n = NS[ni];
        for (pat = 0; pat < 6; pat++) {
            for (i = 0; i < n; i++) {
                switch (pat) {
                case 0: in[i] = (float)(int32_t)rnd() / 2147483648.0f; break;
                case 1: in[i] = 1.0f; break;              /* every diff 0 */
                case 2: in[i] = (i & 1) ? 1e20f : 1e-20f; break;
                case 3: in[i] = (i & 1) ? -1.0f : 1.0f; break;
                case 4: in[i] = (i == 0) ? 1.0f : 0.0f; break;  /* impulse */
                default: in[i] = 1.0f / (float)(i + 1); break;
                }
            }
            memcpy(w, in, sizeof(float) * (size_t)n);
            lz_fwht(w, n);
            for (i = 0; i < n; i++) {
                union { float f; uint32_t u; } b;
                b.f = w[i];
                h = (h ^ b.u) * 16777619u;
            }
            printf("n=%3d pat=%d  y[0]=%.9g y[%d]=%.9g\n",
                   n, pat, (double)w[0], n - 1, (double)w[n - 1]);
        }
    }
    lz_fpu_float_end(fpu);
    printf("FWHTF %08lx\n", (unsigned long)h);
    return 0;
}
