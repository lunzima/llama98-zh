/* What an int64->float convert costs on ARMv5TE, against int32->float.
 *
 * STANDALONE, not a mode in .prof/arm_prim_count.c, because that file is
 * being extended by another line of work and two sessions adding mode
 * numbers to one enum is how they collide. It carries its own control
 * arm for the same reason every arm there does: the loop with the
 * operation removed and nothing else changed.
 *
 * THE QUESTION. docs/int-pipeline-project.md 0.5 wants pass 1's epilogue
 * to sum the two state planes in the integer domain:
 *   (lz_i32f(au) + lz_i32f(aul) * 2^-8) * su    2 cvt + 2 mul + 1 add
 *   lz_i32f64((au << 8) + aul) * (su * 2^-8)    1 cvt + 1 mul
 * Measured peaks put |au| at 2^26.7, so the shift needs int64 and the
 * remaining convert is __aeabi_l2f rather than __aeabi_i2f. Whether the
 * trade is still a win depends on a number nobody in this tree has
 * counted, and M5's record is that an estimate ran 70% high.
 *
 * ARMS
 *   0  control: the loop, the loads, the sink, no conversion
 *   1  one int32 -> float                    (__aeabi_i2f)
 *   2  one int64 -> float                    (__aeabi_l2f)
 *   3  the SHIPPING chain: 2 i2f + fmul + fadd + fmul
 *   4  the CANDIDATE chain: shift, int64 add, l2f, fmul
 *
 * 3 and 4 are what the decision actually rests on; 1 and 2 are there so
 * a surprising 4 can be attributed rather than guessed at.
 *
 * The inputs are the measured magnitudes, not toy values: a soft-float
 * convert's cost depends on how far the operand has to be normalised,
 * so feeding it small numbers would flatter both arms unequally.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef L2F_MODE
#define L2F_MODE 0
#endif

#define N 64

/* Volatile so nothing folds: the compiler can see every value here. */
static volatile int32_t g_au[N], g_aul[N];
static volatile float   g_su = 3.0517578e-05f;   /* 1/32768, a real su */
volatile int32_t g_sink;

int main(int argc, char **argv) {
    long n = (argc > 1) ? atol(argv[1]) : 256;
    union { float f; int32_t i; } u;
    long i;
    int k;

    /* Seeded to the measured peaks (docs/int-pipeline-project.md 0.5):
       |au| up to 110,344,130 and |aul| up to 75,814,542. A convert's
       cost is a function of the operand, so the distribution matters. */
    for (k = 0; k < N; k++) {
        g_au[k]  = (int32_t)(110344130L - (long)k * 1500000L);
        g_aul[k] = (int32_t)(75814542L  - (long)k * 1000000L);
    }

    for (i = 0; i < n; i++) {
        int j = (int)(i & (N - 1));
#if L2F_MODE == 1
        u.f = (float)g_au[j];                       g_sink = u.i;
#elif L2F_MODE == 2
        /* SHIFTED, not merely widened. `(float)(int64_t)g_au[j]` looks
           like an int64 convert and is not: the value provably fits in
           int32, so gcc calls __aeabi_i2f and the arm measures 27.00 -
           the same number as mode 1, to the instruction. That agreement
           is what gave it away. The shift is what makes the operand
           genuinely 64-bit, and it is also the operand the candidate
           chain actually converts. */
        u.f = (float)((int64_t)g_au[j] << 8);       g_sink = u.i;
#elif L2F_MODE == 3
        u.f = ((float)g_au[j] +
               (float)g_aul[j] * (1.0f / 256.0f)) * g_su;
        g_sink = u.i;
#elif L2F_MODE == 4
        {
            int64_t t = ((int64_t)g_au[j] << 8) + (int64_t)g_aul[j];
            u.f = (float)t * g_su;                  g_sink = u.i;
        }
#elif L2F_MODE == 5
        /* THE SHIPPING CHAIN AS IT IS SINCE 7c8ff86, which mode 3 is no
           longer: the 2^-8 weighting is an exponent subtract now, not a
           __aeabi_fmul. Mode 3 is kept because the rejection recorded in
           docs/int-pipeline-project.md 0.5 was measured against it and a
           table whose baseline silently moved is worse than a stale one.
           The body below must stay a verbatim copy of f32_scale_pow2 in
           src/ops_kernel_shared.h - it is duplicated here on purpose, so
           the probe stays standalone, and if the two drift this arm
           measures nothing. */
        {
            union { float f; uint32_t u; } b;
            uint32_t bex;
            int ex;
            float lo;
            b.f = (float)g_aul[j];
            bex = (b.u >> 23) & 0xFFu;
            ex  = (int)bex - 8;
            if ((uint32_t)(bex - 1u) >= 0xFEu ||
                (uint32_t)(ex - 1) >= 254u) {
                lo = (float)g_aul[j] * (1.0f / 256.0f);
            } else {
                b.u = (b.u & 0x807FFFFFu) | ((uint32_t)ex << 23);
                lo = b.f;
            }
            u.f = ((float)g_au[j] + lo) * g_su;     g_sink = u.i;
        }
#else
        g_sink = g_au[j];
#endif
    }
    printf("%ld\n", (long)g_sink);
    return 0;
}
