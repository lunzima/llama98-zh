/* Standalone check for the conv1d fixed-tier sigmoid fold (ops.c:
 * lz_causal_conv1d_step_fixed / ops_quant.c: sigmoid_q15_t,
 * lz_sig_q15_fold). Reimplements both the old shape (v = acc*oscale,
 * then sigmoid_q15(v) through the ENGINE's own table math) and the new
 * one (sigmoid_q15_t fed a pre-scaled t = acc*sig_k1 + sig_k2,
 * sig_oscale2 folding the exit descale) side by side and compares bits.
 *
 * WHY A REIMPLEMENTATION, NOT THE SHIPPED FUNCTIONS: the claim under
 * test is an algebraic identity (exact power-of-two scaling commutes
 * with float rounding) that has to hold for arbitrary (acc, oscale),
 * not just the values one forward pass produces - a standalone sweep
 * covers the joint range .prof/wconv2.c-style end-to-end driving would
 * not reach in a reasonable number of tokens. The end-to-end check
 * (this fold IS what's shipped) is the --dump-logits gcc/Watcom/HEAD
 * comparison in the commit, not this file.
 *
 * Sweeps oscale over every exponent lz_conv_norm_pow2/LZ_CONV_ES can
 * produce (2^-39 .. 2^-9) and acc over a dense band plus a geometric
 * sweep out to the accumulator bound (+-2^30), including the +-1 offset
 * neighbours of each power of two to hit sigmoid_q15_t's t<=0/t>=N
 * clamp boundaries. 130820 cases, 0 mismatches.
 *
 * Build: gcc -O2 -std=c99 -o verify_conv_fold3 verify_conv_fold3.c -lm */
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define LZ_SIG_N    384
#define LZ_SIG_LO   (-12.0f)
#define LZ_SIG_STEP 16.0f

static int32_t g_sigtab[LZ_SIG_N + 1];

static void sig_build(void) {
    int i;
    for (i = 0; i <= LZ_SIG_N; i++) {
        float x = LZ_SIG_LO + (float)i / LZ_SIG_STEP;
        float z = expf(x >= 0.0f ? -x : x);
        float s = (x >= 0.0f) ? 1.0f / (1.0f + z) : z / (1.0f + z);
        g_sigtab[i] = (int32_t)(s * 32768.0f + 0.5f);
    }
}

int32_t sigmoid_q15(float x) {
    float t; int idx; int32_t frac, a, b;
    if (x <= LZ_SIG_LO) return 0;
    if (x >= -LZ_SIG_LO) return 32768;
    t = (x - LZ_SIG_LO) * LZ_SIG_STEP;
    idx = (int)t;
    if (idx >= LZ_SIG_N) return g_sigtab[LZ_SIG_N];
    frac = (int32_t)((t - (float)idx) * 32768.0f);
    a = g_sigtab[idx];
    b = g_sigtab[idx + 1];
    return a + (((b - a) * frac) >> 15);
}

int32_t sigmoid_q15_t(float t) {
    int idx; int32_t frac, a, b;
    if (t <= 0.0f) return 0;
    if (t >= (float)LZ_SIG_N) return g_sigtab[LZ_SIG_N];
    idx = (int)t;
    frac = (int32_t)((t - (float)idx) * 32768.0f);
    a = g_sigtab[idx];
    b = g_sigtab[idx + 1];
    return a + (((b - a) * frac) >> 15);
}

static float old_path(float acc_f, float oscale) {
    float v = acc_f * oscale;
    int32_t s15 = sigmoid_q15(v);
    float sig = (float)s15 * (1.0f / 32768.0f);
    return v * sig;
}

static float new_path(float acc_f, float oscale) {
    float k1 = oscale * LZ_SIG_STEP;
    float k2 = -LZ_SIG_LO * LZ_SIG_STEP;
    float oscale2 = oscale * (1.0f / 32768.0f);
    float t = acc_f * k1 + k2;
    int32_t s15 = sigmoid_q15_t(t);
    return (acc_f * oscale2) * (float)s15;
}

static long bad = 0, total = 0;

static void check(float acc_f, float oscale, int oi) {
    float o1 = old_path(acc_f, oscale);
    float o2 = new_path(acc_f, oscale);
    uint32_t b1, b2;
    memcpy(&b1, &o1, 4);
    memcpy(&b2, &o2, 4);
    total++;
    if (b1 != b2) {
        bad++;
        if (bad <= 20)
            printf("MISMATCH acc=%g oscale=2^%d old=%.9g(%08x) new=%.9g(%08x)\n",
                   acc_f, oi, o1, b1, o2, b2);
    }
}

int main(void) {
    sig_build();
    int oi, ai;
    for (oi = -39; oi <= -9; oi++) {
        float oscale = ldexpf(1.0f, oi);
        /* dense linear sweep near +-1e6 (covers table-relevant v range for
           these oscales), plus a geometric sweep out to the accumulator
           bound to hit huge |v| that must clamp. */
        for (ai = -2000000; ai <= 2000000; ai += 977)
            check((float)ai, oscale, oi);
        {
            long long g;
            for (g = 1; g <= 1073741824; g *= 2) {
                check((float)g, oscale, oi);
                check((float)-g, oscale, oi);
                check((float)(g + 1), oscale, oi);
                check((float)-(g + 1), oscale, oi);
            }
        }
        check(0.0f, oscale, oi);
    }
    printf("total=%ld bad=%ld\n", total, bad);
    return 0;
}
