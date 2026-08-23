/* lz_matmul's SSE1 F32-weight row against its C body.
 *
 * BIT-IDENTITY IS THE WHOLE CLAIM, and it is not obvious the way the
 * integer kernels' is. Float addition does not reassociate, so a
 * horizontal-add SIMD version WOULD change the answer; this one does
 * not, because the C already keeps eight independent accumulators
 * striding by 8 (to break the latency chain, per its own comment) and
 * lane k of the two xmm registers accumulates exactly what scalar a_k
 * accumulates, in the same order. The final combine
 * ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)) stays in C.
 *
 * So the sweep is aimed at what could still break it: an in_dim that is
 * not a multiple of 8 (the C tail runs beside the kernel), a row short
 * enough that the kernel declines entirely, and values whose partial
 * sums differ by enough that a wrong lane assignment shows up rather
 * than cancelling - alternating magnitudes, one huge element, and
 * catastrophic-cancellation pairs.
 *
 * The tier is argv[1]: src/ops.c dispatches on g_kernel, so ref must
 * reach the C loop and sse2/mmx the kernel. A run of one tier proves
 * nothing on its own.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ops.h"

#define MAXIN  1024
#define MAXOUT 32

static float w[MAXIN * MAXOUT];
static float x[MAXIN];
static float o[MAXOUT];

static uint32_t rs = 0x9E3779B9u;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

int main(int argc, char **argv) {
    static const int INS[] = { 1, 7, 8, 9, 15, 16, 33, 64, 127, 512, 1024 };
    uint32_t h = 2166136261u;
    int ii, pat, i, od = 8;
    unsigned fpu;
    if (argc > 1) {
        int t = LZ_KERNEL_REF;
        if (argv[1][0] == 'm') t = LZ_KERNEL_MMX;
        else if (argv[1][0] == 's') t = LZ_KERNEL_SSE2;
#if defined(LZ_KERNEL_ARM_ASM)
        /* The ARM cell landed after this probe was written, and it is
           the same question in the same shape: `a` must reach
           lz_matmul_row_arm_asm and `c` the C loop beside it. */
        else if (argv[1][0] == 'a') t = LZ_KERNEL_ARM_ASM;
        else if (argv[1][0] == 'c') t = LZ_KERNEL_ARM;
#endif /* LZ_KERNEL_ARM_ASM */
        lz_kernel_select(t);
    }
    /* PC=24, WHICH IS NOT OPTIONAL HERE. On x87 the C body's eight
       accumulators live in 80-bit registers, so without this it is not
       computing in float32 at all and no float32 SIMD kernel could
       match it - the first run of this probe reported the Watcom SSE1
       cell "wrong" for exactly that reason, at the fourth decimal.
       lz_forward_chunk wraps the whole forward in this pair; the probe
       has to enter the same regime or it is measuring a different
       machine. A no-op wherever the FPU is not x87. */
    fpu = lz_fpu_float_begin();
    for (ii = 0; ii < (int)(sizeof(INS) / sizeof(INS[0])); ii++) {
        int n = INS[ii];
        for (pat = 0; pat < 5; pat++) {
            for (i = 0; i < n * od; i++) {
                switch (pat) {
                case 0: w[i] = ((float)(int32_t)rnd() / 2147483648.0f); break;
                case 1: w[i] = (i & 1) ? 1e6f : 1e-6f; break;
                case 2: w[i] = (i % 8 == 3) ? 1e9f : 1.0f; break;
                case 3: w[i] = (i & 1) ? 1.0f : -1.0f; break;
                default: w[i] = 1.0f / (float)(i + 1); break;
                }
            }
            for (i = 0; i < n; i++)
                x[i] = ((float)(int32_t)rnd() / 2147483648.0f) * 2.0f;
            memset(o, 0, sizeof(o));
            lz_matmul(o, x, w, n, od);
            for (i = 0; i < od; i++) {
                union { float f; uint32_t u; } b;
                b.f = o[i];
                h = (h ^ b.u) * 16777619u;
            }
            printf("in=%4d pat=%d  o[0]=%.9g o[%d]=%.9g\n",
                   n, pat, (double)o[0], od - 1, (double)o[od - 1]);
        }
    }
    lz_fpu_float_end(fpu);
    printf("MATMULF32 %08lx\n", (unsigned long)h);
    return 0;
}
