/* Fixed conv1d tier, compared against the ENGINE's own float path.
 *
 * wconv.c answered "is the idea worth building" with a candidate written
 * inside the probe. This one compiles src/ops.c and drives the two real
 * entry points, so what is measured is the shipped code rather than a
 * transcription of it - the distinction that made a local `current()`
 * copy in an earlier gate agree with itself while the real classifier
 * said something else.
 *
 * Build:  .prof/build.sh wconv2 [gcc|watcom]
 *
 * WHAT IT CHECKS
 *   1. drift of lz_causal_conv1d_step_fixed against lz_causal_conv1d_step
 *      over 64 steps of a live history, both input shapes;
 *   2. the accumulator never leaves int32 - asserted on the real bound,
 *      not assumed from the derivation;
 *   3. the fixed path is self-consistent: same inputs, same bytes out,
 *      run twice (a tier that is not deterministic cannot be gated).
 *
 * Four candidate variants run alongside, each answering whether the
 * shipped tier leaves precision on the table: a per-channel history
 * exponent, a two-plane history, a weights-only arm that locates where
 * the drift lives, and both asymmetric splits of the int32 budget.
 *
 * WHAT IT DOES NOT CHECK. Cross-compiler bit-identity of the FIXED tier
 * against the FLOAT one - they are different numerics by construction
 * (4.8e-5), which is why the tier is a knob and not a replacement. What
 * must be bit-identical is fixed-vs-fixed across builds, and that is a
 * two-build cmp of this probe's dump, not something one run can show.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ops.h"
#include "ops.c"

#define NCH   512
#define K     4
#define HIST  (K - 1)
#define STEPS 64

static unsigned long rs = 12345;
static float frnd(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (float)((rs >> 16) & 0x7FFF) / 16383.5f - 1.0f;
}

static float w[NCH * K], x[NCH];
static float stf[NCH * HIST], stf2[NCH * HIST];
static short mw[NCH * K], shq[NCH * HIST], shq2[NCH * HIST];
static signed char ew[NCH];
static float oscale[NCH], in_scale;
static float o_ref[NCH], o_fix[NCH], o_fix2[NCH];

/* A per-channel history exponent, which the shipped tier does not have.
 * The conv is DEPTHWISE - channel c's history only ever mixes with
 * channel c's taps - so nothing forces one shared exponent, and a shared
 * one has to cover the widest channel while every narrower channel pays
 * for it. Same op count in the hot loop: one loaded scale instead of one
 * constant.
 *
 * Measured here rather than argued, because "more precision per channel"
 * is only worth the second array if the channels actually differ. */
static short shp[NCH * HIST];
static signed char esc[NCH];
static float oscale_pc[NCH], in_scale_pc[NCH];
static float o_pc[NCH];

/* TWO-PLANE history: value ~ (hi + lo/L) * 2^-es, the shape
 * LZ_GDN_STATE_2PLANE uses for the recurrence state. The single plane
 * spends its range on the maximum (|x| up to 22.3 at es=9), so a typical
 * value lands on far fewer levels than int16 could hold; the low plane
 * carries that residue.
 *
 * The two accumulators stay separate and are combined in FLOAT, not in
 * integer: acc_hi*L + acc_lo would need B down at 1448 to fit int32,
 * which is coarser than the single plane's 16384 - the fix would undo
 * itself.
 *
 * f32 operations are counted, not assumed, because that is the whole
 * question: the low plane costs a subtract and a multiply going in, and
 * a second convert-and-multiply coming out. */
#define LOSCALE 256.0f
static short shh[NCH * HIST], shl[NCH * HIST];
static float o_2p[NCH], oscale_lo[NCH];
static long ops_2p;

static void conv_fixed_2p(float *o, const float *x, short *hi, short *lo,
                          const short *mw_, const float *osc,
                          const float *osc_lo, float insc, int bound,
                          int n_ch, int k) {
    int hist = k - 1, c, j;
    for (c = 0; c < n_ch; c++) {
        short *hp = hi + (size_t)c * hist, *lp = lo + (size_t)c * hist;
        const short *wr = mw_ + (size_t)c * k;
        int ah = 0, al = 0, qh, ql;
        float t = x[c] * insc;                      /* 1 */
        float r;
        qh = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
        if (qh > bound) qh = bound;
        if (qh < -bound) qh = -bound;
        r = (t - (float)qh) * LOSCALE;              /* 2, 3 (sub + mul) */
        ql = (int)(r < 0.0f ? r - 0.5f : r + 0.5f);
        if (ql > bound) ql = bound;
        if (ql < -bound) ql = -bound;
        ops_2p += 3;
        for (j = 0; j < hist; j++) {
            ah += (int)wr[j] * (int)hp[j];
            al += (int)wr[j] * (int)lp[j];
        }
        ah += (int)wr[hist] * qh;
        al += (int)wr[hist] * ql;
        o[c] = lz_silu((float)ah * osc[c] + (float)al * osc_lo[c]);
        ops_2p += 5;                     /* 2 converts, 2 muls, 1 add */
        for (j = 0; j + 1 < hist; j++) { hp[j] = hp[j+1]; lp[j] = lp[j+1]; }
        if (hist > 0) { hp[hist-1] = (short)qh; lp[hist-1] = (short)ql; }
    }
}

/* Two diagnostic arms, to locate the 3.1e-5 rather than assume it.
 *
 *   wonly - exact float history and input, quantized weights only. If the
 *           drift survives this, the history planes are not where it lives
 *           and no amount of state precision moves it.
 *   asym  - single plane, but the int32 budget split unevenly: weights get
 *           32767 and activations 8192 instead of 16384 each. k*Bw*Bx is
 *           unchanged at 1.074e9, so this costs nothing - it just moves one
 *           bit from the activation side to the weight side. */
static float stfw[NCH * HIST], o_wo[NCH];
static short mwa[NCH * K], sha[NCH * HIST], mwb[NCH * K], shb[NCH * HIST];
static signed char ewa[NCH], ewb[NCH];
static float oscale_a[NCH], in_scale_a, o_asym[NCH];
static float oscale_b[NCH], in_scale_b, o_asym2[NCH];
#define BW_ASYM 32767
#define BX_ASYM 8192

static void conv_wonly(float *o, const float *x, float *sh,
                       const short *mw_, const signed char *ew_,
                       int n_ch, int k) {
    int hist = k - 1, c, j;
    for (c = 0; c < n_ch; c++) {
        float *si = sh + (size_t)c * hist;
        const short *wr = mw_ + (size_t)c * k;
        float s = (float)ldexp(1.0, -(int)ew_[c]);
        float acc = 0.0f;
        for (j = 0; j < hist; j++) acc += (float)wr[j] * s * si[j];
        acc += (float)wr[hist] * s * x[c];
        o[c] = lz_silu(acc);
        for (j = 0; j + 1 < hist; j++) si[j] = si[j + 1];
        if (hist > 0) si[hist - 1] = x[c];
    }
}

static void conv_fixed_pc(float *o, const float *x, short *sh,
                          const short *mw_, const float *osc,
                          const float *insc, int bound, int n_ch, int k) {
    int hist = k - 1, c, j;
    for (c = 0; c < n_ch; c++) {
        short *si = sh + (size_t)c * hist;
        const short *wr = mw_ + (size_t)c * k;
        int acc = 0, q;
        float t = x[c] * insc[c];
        q = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
        if (q > bound) q = bound;
        if (q < -bound) q = -bound;
        for (j = 0; j < hist; j++) acc += (int)wr[j] * (int)si[j];
        acc += (int)wr[hist] * q;
        o[c] = lz_silu((float)acc * osc[c]);
        for (j = 0; j + 1 < hist; j++) si[j] = si[j + 1];
        if (hist > 0) si[hist - 1] = (short)q;
    }
}

/* Ratios against the shipped tier, both axes. %.3g rather than %.2f
   because the per-channel arm lands at 6.6e-5 and a fixed-point format
   shows that as 0.00. */
static void report(const char *name, double wmax, double ssq,
                   double base_max, double base_ssq, long n) {
    double rms = sqrt(ssq / (double)n), base_rms = sqrt(base_ssq / (double)n);
    printf("  %-18s max %.3e  rms %.3e   -> %.3gx max, %.3gx rms\n",
           name, wmax, rms,
           wmax > 0.0 ? base_max / wmax : 0.0,
           rms > 0.0 ? base_rms / rms : 0.0);
}

int main(void) {
    int c, j, step, mode, bound, es, fails = 0;
    bound = lz_conv_accum_bound(K);
    printf("conv1d fixed tier vs the engine's float path\n");
    printf("n_ch=%d k=%d bound=%d (k*B^2 = %.3e, int32 max 2.147e+09)\n",
           NCH, K, bound, (double)K * (double)bound * (double)bound);
    printf("Q8 activation noise floor ~3.9e-3\n\n");

    for (mode = 0; mode < 2; mode++) {
        double worst = 0.0, sumsq = 0.0, worst_pc = 0.0, sumsq_pc = 0.0;
        double worst_2p = 0.0, sumsq_2p = 0.0;
        double worst_wo = 0.0, sumsq_wo = 0.0, worst_a = 0.0, sumsq_a = 0.0;
        double worst_b = 0.0, sumsq_b = 0.0;
        long n = 0;
        for (c = 0; c < NCH; c++) {
            for (j = 0; j < K; j++) w[c * K + j] = frnd() * 0.5f;
            for (j = 0; j < HIST; j++) {
                float v = frnd();
                if (mode == 1 && j != c % HIST) v *= 0.01f;
                stf[c * HIST + j] = v;
            }
        }
        for (c = 0; c < NCH; c++)
            ew[c] = (signed char)lz_conv_norm_pow2(w + c * K, K,
                                                   mw + c * K, bound);
        es = lz_conv_norm_pow2(stf, NCH * HIST, shq, bound);
        lz_conv_build_scales(oscale, &in_scale, ew, es, NCH);
        memcpy(shq2, shq, sizeof(shq));
        /* two-plane: same es, history split hi/lo */
        {
            int i;
            for (i = 0; i < NCH * HIST; i++) {
                float t = stf[i] * in_scale;
                int h = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
                float r;
                if (h > bound) h = bound;
                if (h < -bound) h = -bound;
                r = (t - (float)h) * LOSCALE;
                shh[i] = (short)h;
                shl[i] = (short)(int)(r < 0.0f ? r - 0.5f : r + 0.5f);
            }
            for (c = 0; c < NCH; c++)
                oscale_lo[c] = oscale[c] / LOSCALE;
        }
        /* per-channel variant: its own exponent per channel */
        for (c = 0; c < NCH; c++) {
            esc[c] = (signed char)lz_conv_norm_pow2(stf + (size_t)c * HIST,
                                                    HIST, shp + (size_t)c * HIST,
                                                    bound);
            in_scale_pc[c] = (float)ldexp(1.0, esc[c]);
            oscale_pc[c] = (float)ldexp(1.0, -((int)ew[c] + (int)esc[c]));
        }
        memcpy(stfw, stf, sizeof(stfw));
        for (c = 0; c < NCH; c++)
            ewa[c] = (signed char)lz_conv_norm_pow2(w + c * K, K,
                                                    mwa + c * K, BW_ASYM);
        lz_conv_build_scales(oscale_a, &in_scale_a, ewa,
                             lz_conv_norm_pow2(stf, NCH * HIST, sha, BX_ASYM),
                             NCH);
        for (c = 0; c < NCH; c++)
            ewb[c] = (signed char)lz_conv_norm_pow2(w + c * K, K,
                                                    mwb + c * K, BX_ASYM);
        lz_conv_build_scales(oscale_b, &in_scale_b, ewb,
                             lz_conv_norm_pow2(stf, NCH * HIST, shb, BW_ASYM),
                             NCH);

        for (step = 0; step < STEPS; step++) {
            for (c = 0; c < NCH; c++) x[c] = frnd();
            lz_causal_conv1d_step(o_ref, x, stf, stf2, w, NCH, K);
            memcpy(stf, stf2, sizeof(stf));
            lz_causal_conv1d_step_fixed(o_fix, x, shq, shq, mw, oscale,
                                        in_scale, bound, NCH, K);
            /* determinism: a second identical call on a copy of the
               state must produce the same bytes */
            lz_causal_conv1d_step_fixed(o_fix2, x, shq2, shq2, mw, oscale,
                                        in_scale, bound, NCH, K);
            if (memcmp(o_fix, o_fix2, sizeof(o_fix)) != 0) {
                printf("  step %d: fixed tier is not deterministic\n", step);
                fails++;
            }
            conv_fixed_pc(o_pc, x, shp, mw, oscale_pc, in_scale_pc,
                          bound, NCH, K);
            conv_fixed_2p(o_2p, x, shh, shl, mw, oscale, oscale_lo,
                          in_scale, bound, NCH, K);
            conv_wonly(o_wo, x, stfw, mw, ew, NCH, K);
            lz_causal_conv1d_step_fixed(o_asym, x, sha, sha, mwa, oscale_a,
                                        in_scale_a, BX_ASYM, NCH, K);
            lz_causal_conv1d_step_fixed(o_asym2, x, shb, shb, mwb, oscale_b,
                                        in_scale_b, BW_ASYM, NCH, K);
            for (c = 0; c < NCH; c++) {
                double d = fabs((double)o_ref[c] - (double)o_fix[c]);
                double p = fabs((double)o_ref[c] - (double)o_pc[c]);
                if (d > worst) worst = d;
                if (p > worst_pc) worst_pc = p;
                { double q2 = fabs((double)o_ref[c] - (double)o_2p[c]);
                  double qw = fabs((double)o_ref[c] - (double)o_wo[c]);
                  double qa = fabs((double)o_ref[c] - (double)o_asym[c]);
                  if (q2 > worst_2p) worst_2p = q2; sumsq_2p += q2 * q2;
                  if (qw > worst_wo) worst_wo = qw; sumsq_wo += qw * qw;
                  if (qa > worst_a) worst_a = qa; sumsq_a += qa * qa;
                  qa = fabs((double)o_ref[c] - (double)o_asym2[c]);
                  if (qa > worst_b) worst_b = qa; sumsq_b += qa * qa; }
                sumsq += d * d; sumsq_pc += p * p; n++;
            }
        }
        printf("%s\n", mode ? "one dominant tap per channel:"
                            : "uniform state:");
        printf("  %-18s max %.3e  rms %.3e   (shipped)\n", "shared es",
               worst, sqrt(sumsq / (double)n));
        report("per-chan", worst_pc, sumsq_pc, worst, sumsq, n);
        report("two-plane", worst_2p, sumsq_2p, worst, sumsq, n);
        report("weights only", worst_wo, sumsq_wo, worst, sumsq, n);
        report("asym w32767/x8192", worst_a, sumsq_a, worst, sumsq, n);
        report("asym w8192/x32767", worst_b, sumsq_b, worst, sumsq, n);
        /* float = 2k: lz_causal_conv1d_step is k multiplies and k adds
           per channel (the first add is against the 0.0f initializer,
           which -ffp-contract=off without -ffast-math cannot remove).
           1-plane = 3 regardless of k: one multiply in, one int->float
           convert and one multiply out. silu sits on top of all three
           equally and is not counted here. */
        printf("  f32 ops per channel-step: float %d, 1-plane 3, 2-plane %.1f\n\n",
               2 * K, (double)ops_2p / (double)(STEPS * NCH));
        ops_2p = 0;
        if (worst > 3.9e-3) {
            printf("  ** drift reached the noise floor\n");
            fails++;
        }
    }

    /* The bound, checked rather than trusted. Worst case is every tap at
       +-bound with the same sign; if that fits, nothing real can fail. */
    {
        double worst_case = (double)K * (double)bound * (double)bound;
        printf("worst-case accumulator %.3e, int32 max 2.147e+09 -> %s\n",
               worst_case, worst_case <= 2147483647.0 ? "fits" : "OVERFLOWS");
        if (worst_case > 2147483647.0) fails++;
    }
    printf("\n%s\n", fails ? "FAILURES" : "all checks passed");
    return fails ? 1 : 0;
}
