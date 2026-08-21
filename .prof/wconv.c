/* Is a fixed-point causal conv1d worth writing?
 *
 * lz_causal_conv1d_step is the one operator in the engine with none of
 * iron law 8's seven implementations: no SIMD, no fixed tier, pure
 * scalar f32. On a target with no FPU every one of its multiply-adds is
 * a libgcc call, so it is an obvious candidate - but it is also the
 * operator where the candidate is most likely to LOSE, because each
 * channel does only k=4 multiply-adds and a fixed tier has to pay a
 * conversion in and a conversion out around them.
 *
 * This probe answers two questions before any kernel is written:
 *
 *   1. how many f32 operations does each version actually do, counted
 *      rather than estimated (cycles do not transfer - iron law 3 -
 *      but operation counts do);
 *   2. how far does the fixed version drift from the f32 one, over
 *      inputs shaped like the real thing.
 *
 * THE CANDIDATE follows gdn_p2_group_fixed's shape rather than
 * inventing one: normalize by POWERS OF TWO only, so the scale is exact
 * and the same on both compilers, and keep the recurrent state in fixed
 * point so the conversion is paid once per new sample instead of once
 * per tap. That second half is what made the pass-2 tier a win - its
 * own comment says the fixed tier does not accelerate the write-back
 * quantize, it deletes it.
 *
 * WHAT WOULD FOOL THIS:
 *   - counting only the multiply-adds. The conversions are the whole
 *     question, so they are counted too, and the count is printed per
 *     channel-step rather than as a total.
 *   - a benign input distribution. The conv history is a decayed signal;
 *     a uniform random state hides the case where one tap dominates and
 *     the shared exponent has to cover a wide range. Both are run.
 *   - reporting drift without a scale. The engine's Q8 activation noise
 *     floor is ~3.9e-3; a drift far below that is free, one near it is
 *     not. The floor is printed next to the number.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCH   512      /* CE and kunmoe both use 512-wide conv channels */
#define K     4
#define HIST  (K - 1)

/* ---- reference: exactly what src/ops.c does today -------------------- */
static void conv_f32(float *o, const float *x, const float *st_in,
                     float *st_out, const float *w, int n_ch, int k,
                     long *f32_ops) {
    int c, j, hist = k - 1;
    for (c = 0; c < n_ch; c++) {
        const float *si = st_in + (size_t)c * hist;
        float *so = st_out + (size_t)c * hist;
        const float *wr = w + (size_t)c * k;
        float sum = 0.0f;
        for (j = 0; j < hist; j++) { sum += wr[j] * si[j]; *f32_ops += 2; }
        sum += wr[hist] * x[c]; *f32_ops += 2;
        o[c] = sum;                       /* silu applied by the caller */
        for (j = 0; j + 1 < hist; j++) so[j] = si[j + 1];
        so[hist - 1] = x[c];
    }
}

/* ---- candidate: int16 taps, int16 history, power-of-two scales ------- */
typedef struct {
    short  mw[NCH][K];     /* weights, normalized per channel */
    signed char ew[NCH];   /* w[j] ~ mw[j] * 2^-ew */
    short  sh[NCH][HIST];  /* history, already fixed point */
    signed char es;        /* shared across channels: one exponent */
    int    bound;          /* accum_bound(K) */
    long   peak;           /* largest |acc| seen, so the margin is a
                              MEASURED number and not a claim */
} FixState;

/* The bound is only worth stating if something checks it. Tracks the
   real peak and refuses to continue past int32 - so loosening
   accum_bound turns this red instead of returning a plausible wrong
   answer, which is what full-range normalization did. */
static void ovf_check(int acc, FixState *s) {
    long a = acc < 0 ? -(long)acc : (long)acc;
    if (a > s->peak) s->peak = a;
}

/* THE ACCUMULATOR BOUND, derived rather than picked.
 *
 * k products of two values bounded by B accumulate to at most k*B^2, and
 * that has to stay inside int32 - int64 accumulation is not an option on
 * the target (ARMv5TE has SMLAL, but the x86 side would need a carry
 * chain per tap and MMX's pmaddwd produces int32 pairs that overflow the
 * same way).
 *
 *   k * B^2 <= 2^31 - 1   ->   B <= sqrt((2^31-1)/k)
 *   k = 4  ->  B <= 23170  ->  largest power of two is 2^14 = 16384
 *
 * At B = 2^14 the worst case is 4 * 2^28 = 2^30, one full bit of margin.
 * Costs one bit of mantissa on each side against a full-range int16.
 *
 * Full-range normalization (B = 32767) is what the first draft of this
 * probe used, and it reported max |diff| = 2.0 - saturation, not
 * rounding, because 4*32767^2 = 4.3e9 wraps int32. The neighbouring
 * failure is already on the record: p2_group_norm's own comment about a
 * value at 32767.5 rounding to 32768 and flipping sign as int16. */
static int accum_bound(int k) {
    int b = 1;
    while ((double)k * (double)(b * 2) * (double)(b * 2) <= 2147483647.0)
        b *= 2;
    return b;
}

/* Normalize v[0..n-1] into int16 by a power of two, clamped to +-bound.
   Power of two only: the scale is exact in binary, so this step adds no
   rounding of its own and cannot differ between x87 and SSE. */
static int norm_pow2(const float *v, int n, short *out, int bound) {
    float amax = 0.0f;
    int i, e;
    for (i = 0; i < n; i++) { float a = (float)fabs(v[i]); if (a > amax) amax = a; }
    if (amax == 0.0f) { for (i = 0; i < n; i++) out[i] = 0; return 0; }
    e = 0;
    while (amax * (float)(1 << (e + 1)) < (float)bound && e < 30) e++;
    for (i = 0; i < n; i++) {
        float t = v[i] * (float)(1 << e);
        int q = (int)(t < 0 ? t - 0.5f : t + 0.5f);
        if (q > bound) q = bound;
        if (q < -bound) q = -bound;
        out[i] = (short)q;
    }
    return e;
}

static void conv_fixed(float *o, const float *x, FixState *s,
                       int n_ch, int k, long *f32_ops) {
    int c, j, hist = k - 1;
    for (c = 0; c < n_ch; c++) {
        int q;
        /* int32, which is the point: accum_bound() picked B so that this
           cannot overflow. Widening it to int64 here would hide a wrong
           bound instead of catching it. */
        int acc = 0;
        float t = x[c] * (float)(1 << s->es);   /* 1 f32 mul */
        *f32_ops += 1;
        q = (int)(t < 0 ? t - 0.5f : t + 0.5f);
        if (q > s->bound) q = s->bound;
        if (q < -s->bound) q = -s->bound;
        for (j = 0; j < hist; j++) {
            acc += (int)s->mw[c][j] * (int)s->sh[c][j];
            ovf_check(acc, s);
        }
        acc += (int)s->mw[c][hist] * q;
        ovf_check(acc, s);
        /* one f32 multiply to leave fixed point */
        o[c] = (float)acc * (float)ldexp(1.0, -(s->ew[c] + s->es));
        *f32_ops += 2;                       /* int->float, and the mul */
        for (j = 0; j + 1 < hist; j++) s->sh[c][j] = s->sh[c][j + 1];
        s->sh[c][hist - 1] = (short)q;
    }
}

static unsigned long rs = 12345;
static float frnd(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (float)((rs >> 16) & 0x7FFF) / 16383.5f - 1.0f;
}

int main(void) {
    static float w[NCH * K], x[NCH], st[NCH * HIST], st2[NCH * HIST];
    static float o_ref[NCH], o_fix[NCH];
    static FixState fs;
    int c, j, step, mode;
    long ops_ref, ops_fix;

    printf("causal conv1d: f32 reference vs fixed-point candidate\n");
    printf("n_ch=%d k=%d  (Q8 activation noise floor ~3.9e-3)\n\n", NCH, K);

    fs.bound = accum_bound(K);
    printf("accumulator bound B = %d (k*B^2 = %.3e, int32 max 2.147e+09)\n\n",
           fs.bound, (double)K * (double)fs.bound * (double)fs.bound);

    for (mode = 0; mode < 2; mode++) {
        double worst = 0.0, sumsq = 0.0;
        long n = 0;
        /* mode 0: uniform. mode 1: one tap dominates per channel, which
           is what a decayed history looks like and what a shared
           exponent has the most trouble covering. */
        for (c = 0; c < NCH; c++) {
            for (j = 0; j < K; j++) w[c * K + j] = frnd() * 0.5f;
            for (j = 0; j < HIST; j++) {
                float v = frnd();
                if (mode == 1 && j != c % HIST) v *= 0.01f;
                st[c * HIST + j] = v;
            }
        }
        memcpy(st2, st, sizeof(st));
        for (c = 0; c < NCH; c++)
            fs.ew[c] = (signed char)norm_pow2(w + c * K, K, fs.mw[c],
                                              fs.bound);
        {   /* one shared exponent for the whole state, as a kernel would */
            static float all[NCH * HIST];
            static short tmp[NCH * HIST];
            memcpy(all, st, sizeof(all));
            fs.es = (signed char)norm_pow2(all, NCH * HIST, tmp, fs.bound);
            for (c = 0; c < NCH; c++)
                for (j = 0; j < HIST; j++)
                    fs.sh[c][j] = tmp[c * HIST + j];
        }
        ops_ref = ops_fix = 0;
        for (step = 0; step < 64; step++) {
            for (c = 0; c < NCH; c++) x[c] = frnd();
            conv_f32(o_ref, x, st, st2, w, NCH, K, &ops_ref);
            memcpy(st, st2, sizeof(st));
            conv_fixed(o_fix, x, &fs, NCH, K, &ops_fix);
            for (c = 0; c < NCH; c++) {
                double d = fabs((double)o_ref[c] - (double)o_fix[c]);
                if (d > worst) worst = d;
                sumsq += d * d; n++;
            }
        }
        printf("%s\n", mode ? "one dominant tap per channel:" : "uniform state:");
        printf("  max |diff| %.3e   rms %.3e\n", worst, sqrt(sumsq / (double)n));
        printf("  f32 ops per channel-step:  reference %.1f   fixed %.1f\n",
               (double)ops_ref / (64.0 * NCH), (double)ops_fix / (64.0 * NCH));
        printf("  peak |acc| %ld   headroom to int32 %.2fx\n\n",
               fs.peak, 2147483647.0 / (double)(fs.peak ? fs.peak : 1));
    }
    return 0;
}
