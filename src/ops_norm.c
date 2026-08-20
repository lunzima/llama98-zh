/* ops_norm.c - Normalization: RMSNorm, L2Norm, softmax, and the
   fixed-tier gated output loop. A separate TU from ops.c to keep #if
   guard density low. These call norm_ss_fixed/expof/q8_amax from
   ops_quant.c and the LZ_NORMOUT/VMAX/VSCALE dispatch tables from
   ops_kernel_norm.h. */
#include <stddef.h>
#include <stdint.h>

#ifndef LZ_GDN_FIXED
#define LZ_GDN_FIXED 1
#endif /* LZ_GDN_FIXED */

#include "ops.h"
#include "err.h"
#include "mmx_compat.h"
#include "ops_mmx.h"
#include "ops_sse2.h"
#include "ops_quant.h"
#include "ops_sched.h"
#include "ops_kernel_shared.h"
#include "ops_norm.h"

#include "ops_kernel_norm.h"

#if LZ_GDN_FIXED
/* Fixed output loop of lz_rmsnorm_gated. The full Q15 int-multiply:
   x, w, g each get their own power-of-two Q15 scale (q8_amax +
   pow2f(14-expof)), inv is scaled likewise, sigmoid comes straight from
   the Q15 table, and the product runs in int64 with three >>15 steps.
   One dequantize at the end. The sig parameter is unused (kept in the
   signature for API stability). This is value-changing, not bit-identical
   to the float chain (the float chain's 5 per-step roundings cannot be
   reproduced by one final rounding), but it offloads the sigmoid work
   from the sigmoid census site entirely. forward.c uses lz_quantize_q8
   (not lz_quantize_q8_silu) since the output already holds the final
   silu value. */
void lz_rmsnorm_gated_out_fixed(float *o, const float *x,
                                       const float *g, const float *w,
                                       float inv, int n, int32_t *sig) {
    union { float f; uint32_t u; } bit;
    float sx, sw, sg;
    int ex, ew, eg, ei;
    int32_t qinv;
    int i;
    (void)sig;

    bit.u = q8_amax(x, n); ex = 14 - expof(bit.f); sx = pow2f(ex);
    bit.u = q8_amax(w, n); ew = 14 - expof(bit.f); sw = pow2f(ew);
    bit.u = q8_amax(g, n); eg = 14 - expof(bit.f); sg = pow2f(eg);
    bit.u = q8_amax(&inv, 1); ei = 14 - expof(bit.f);
    qinv = q8_round(inv * pow2f(ei));
    if (qinv > 32767) qinv = 32767;
    if (qinv < -32767) qinv = -32767;
    for (i = 0; i < n; i++) {
        int qx = q8_round(x[i] * sx);
        int qw = q8_round(w[i] * sw);
        int qg = q8_round(g[i] * sg);
        int32_t sigv = sigmoid_q15(g[i]);
        int64_t t;
        if (qx > 32767) qx = 32767; else if (qx < -32767) qx = -32767;
        if (qw > 32767) qw = 32767; else if (qw < -32767) qw = -32767;
        if (qg > 32767) qg = 32767; else if (qg < -32767) qg = -32767;
        t = (int64_t)qx * qw;
        t = (t * qg) >> 15;
        t = (t * qinv) >> 15;
        t = (t * sigv) >> 15;
        o[i] = (float)t * pow2f(30 - ex - ew - eg - ei);
    }
}
#else
void lz_rmsnorm_gated_out_fixed(float *o, const float *x,
                                       const float *g, const float *w,
                                       float inv, int n, int32_t *sig) {
    (void)o; (void)x; (void)g; (void)w; (void)inv; (void)n; (void)sig;
}
#endif /* LZ_GDN_FIXED */

#if LZ_GDN_FIXED
/* Int output variant of lz_rmsnorm_gated_out_fixed: identical Q15
   int-multiply, but instead of dequantizing to float per element the
   Q15 result is written as int and the single power-of-two scale
   deq = 2^(30-ex-ew-eg-ei) is returned out-of-band. lz_quantize_q8_int
   then folds deq into the stored group scale only, so the Q15 -> float
   -> int8 double conversion disappears while staying bit-identical to
   the float chain (the cancellation proof lives at lz_quantize_q8_int). */
void lz_rmsnorm_gated_out_fixed_int(int32_t *t, const float *x,
                                    const float *g, const float *w,
                                    float inv, int n, int32_t *sig,
                                    float *deq) {
    union { float f; uint32_t u; } bit;
    float sx, sw, sg;
    int ex, ew, eg, ei;
    int32_t qinv;
    int i;
    (void)sig;

    bit.u = q8_amax(x, n); ex = 14 - expof(bit.f); sx = pow2f(ex);
    bit.u = q8_amax(w, n); ew = 14 - expof(bit.f); sw = pow2f(ew);
    bit.u = q8_amax(g, n); eg = 14 - expof(bit.f); sg = pow2f(eg);
    bit.u = q8_amax(&inv, 1); ei = 14 - expof(bit.f);
    qinv = q8_round(inv * pow2f(ei));
    if (qinv > 32767) qinv = 32767;
    if (qinv < -32767) qinv = -32767;
    for (i = 0; i < n; i++) {
        int qx = q8_round(x[i] * sx);
        int qw = q8_round(w[i] * sw);
        int qg = q8_round(g[i] * sg);
        int32_t sigv = sigmoid_q15(g[i]);
        int64_t t64;
        if (qx > 32767) qx = 32767; else if (qx < -32767) qx = -32767;
        if (qw > 32767) qw = 32767; else if (qw < -32767) qw = -32767;
        if (qg > 32767) qg = 32767; else if (qg < -32767) qg = -32767;
        t64 = (int64_t)qx * qw;
        t64 = (t64 * qg) >> 15;
        t64 = (t64 * qinv) >> 15;
        t64 = (t64 * sigv) >> 15;
        t[i] = (int32_t)t64;
    }
    *deq = pow2f(30 - ex - ew - eg - ei);
}
#else
void lz_rmsnorm_gated_out_fixed_int(int32_t *t, const float *x,
                                    const float *g, const float *w,
                                    float inv, int n, int32_t *sig,
                                    float *deq) {
    (void)t; (void)x; (void)g; (void)w; (void)inv; (void)n; (void)sig;
    *deq = 0.0f;
}
#endif /* LZ_GDN_FIXED */

/* lz_rmsnorm_gated fixed-tier wrapper for the int output: same
   sum-of-squares and rsqrt as lz_rmsnorm_gated, then the Q15
   int-multiply loop writing int instead of dequantizing. Only valid
   when lz_norm_can_fixed(n); callers gate on it. */
void lz_rmsnorm_gated_int(int32_t *t, const float *x, const float *g,
                          const float *w, int n, float eps, float *deq) {
    float ss = 0.0f;
    float inv;
    float ssf;
    int i;
    if (lz_norm_mode()) {
        ssf = norm_ss_fixed(x, n);
        if (ssf >= 0.0f) {
            /* Same replacement as lz_rmsnorm_gated's fixed branch, minus
               the per-element dequant (1 cvt + 1 mul) the float output
               loop spent; the Q15 value stays int and deq is one scalar
               per call. */
            LZ_FCX(LZ_FC_NORM, 4 * n + 1, 3 * n + 1, 1, n + 1, 0);
            ss = ssf;
            goto have_ss_gated_int;
        }
    }
    /* float ss fallback - same billing as lz_rmsnorm_gated's float row
       (the output loop is skipped entirely here). */
    LZ_FCX(LZ_FC_NORM, 5 * n, n + 1, 1, 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss_gated_int:
    inv = lz_rsqrt(ss / (float)n + eps);
    lz_rmsnorm_gated_out_fixed_int(t, x, g, w, inv, n, NULL, deq);
}

void lz_rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    float inv;
    float ssf;
    int i;
    if (lz_norm_mode()) {
        ssf = norm_ss_fixed(x, n);
        if (ssf >= 0.0f) {
            /* norm_ss_fixed itself is mul + q8_round per element (2n,
               the same two float ops the loop below spends) plus a
               convert and a mul once at the end to bring the int64
               accumulator back to float (2) - it has no site of its
               own, so its cost is folded in here. Untouched: the 3n
               output loop (2 mul + 1 add per element) and the
               divide+eps (1 div, 1 add; the (float)n cast is not
               billed, see B1 report). */
            LZ_FCX(LZ_FC_NORM, 3 * n + 1, 2 * n + 1, 1, 1, 0);
            ss = ssf;
            goto have_ss;
        }
    }
    /* 2n accumulating ss, 2 for the divide and the eps add, 3n in the
       output loop (two multiplies and the 1+w add). rsqrt bills itself. */
    LZ_FCX(LZ_FC_NORM, 3 * n, 2 * n + 1, 1, 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss:
    inv = lz_rsqrt(ss / (float)n + eps);
    /* (1 + w), not w: see the header */
    i = 0;
    {
        lz_normoutfn f = lz_normout_pick(LZ_NORMOUT_TAB);
        if (f && n >= 4) {
            float k2[2];
            int n4 = n >> 2;
            k2[0] = inv;
            k2[1] = 1.0f;
            f(o, x, w, n4, k2);
            i = n4 << 2;
        }
    }
    for (; i < n; i++) o[i] = x[i] * inv * (1.0f + w[i]);
}

void lz_rmsnorm_gated(float *o, const float *x, const float *g,
                          const float *w, int n, float eps, int32_t *sig) {
    float ss = 0.0f;
    float inv;
    float ssf;
    int i;
    int fixed_ss = 0;
    if (lz_norm_mode()) {
        ssf = norm_ss_fixed(x, n);
        if (ssf >= 0.0f) {
            /* Same replacement as lz_rmsnorm: norm_ss_fixed's own 2n+2
               in place of the 2n float ss loop. The fixed output loop
               (lz_rmsnorm_gated_out_fixed) computes the full Q15
               int-multiply (x*inv*w*silu) so the sigmoid site stops
               billing it. Per element: sigmoid_q15 (2 mul, 2 cvt, 2
               sub), q8_round per factor (3 mul, 3 add), int64 product
               (3 mul, 3 shift), dequant (1 cvt, 1 mul); plus
               norm_ss_fixed (n mul, n add, 1 mul, 1 cvt) and the
               divide+eps (1 div, 1 add). */
            LZ_FCX(LZ_FC_NORM, 5 * n + 1, 3 * n + 1, 1, 2 * n + 1, 0);
            ss = ssf;
            fixed_ss = 1;
            goto have_ss_gated;
        }
    }
    /* ss loop 2n, divide+eps (1 div, 1 add), output loop x*inv*w*silu(g)
       (three multiplies, no add per element - the gate replaces
       rmsnorm's (1+w) add with a multiply). The site bills one more n
       than that sums to (6n+2 vs the 5n+2 the code performs); the
       extra n is folded into mul here to hold the site's historical
       total - see B1 report, this looks like a real overcount. silu
       bills itself. */
    LZ_FCX(LZ_FC_NORM, 5 * n, n + 1, 1, 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss_gated:
    inv = lz_rsqrt(ss / (float)n + eps);
    /* plain w here, and the gate applies last. The fixed path computes
       the full silu value in Q15 int-multiply (value-changing); the
       float path is untouched. */
    if (fixed_ss)
        lz_rmsnorm_gated_out_fixed(o, x, g, w, inv, n, sig);
    else
        for (i = 0; i < n; i++) o[i] = x[i] * inv * w[i] * lz_silu(g[i]);
}

void lz_l2norm(float *o, const float *x, int n, float eps) {
    float ss = 0.0f;
    float inv;
    float ssf;
    int i;
    if (lz_norm_mode()) {
        ssf = norm_ss_fixed(x, n);
        if (ssf >= 0.0f) {
            /* Same replacement again: norm_ss_fixed's 2n+2 in place of
               the 2n float ss loop. Untouched: the 1n output loop (a
               single multiply, no weight) and the eps add (1, no
               divide - l2norm normalizes by the sum, not the mean).
               That sums to 3n+3, one n short of the site's 4n+3;
               the extra n is folded into mul here to hold the site's
               historical total - see B1 report, this looks like a
               real overcount. */
            LZ_FCX(LZ_FC_NORM, 3 * n + 1, n + 1, 0, 1, 0);
            ss = ssf;
            goto have_ss_l2;
        }
    }
    /* ss loop 2n, eps add (1, no divide), output loop 1n (single
       multiply, no weight). That sums to 3n+1, one n short of the
       site's 4n+1; the extra n is folded into mul here to hold the
       site's historical total - see B1 report, this looks like a real
       overcount. */
    LZ_FCX(LZ_FC_NORM, 3 * n, n + 1, 0, 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss_l2:
    inv = lz_rsqrt(ss + eps);          /* sum, not mean */
    for (i = 0; i < n; i++) o[i] = x[i] * inv;
}

void lz_softmax(float *x, int n) {
    float mx = x[0], sum = 0.0f;
    int i;
    /* Per element: the subtract feeding exp, the sum add, the final
       scale multiply. Plus one divide. The max scan is comparisons.
       exp bills itself - this row is the softmax around it, and reading
       the two as one number is what put the old census 13x low here. */
    LZ_FCX(LZ_FC_SOFTMAX, n, 2 * n, 1, 0, 0);
    i = 1;
    {
        /* The maximum is exact, so lane order cannot change it and this
           owes the scalar loop bit-identity rather than a tolerance. The
           sum below is NOT vectorized, for the opposite reason. */
        lz_vmaxfn f = lz_vmax_pick(LZ_VMAX_TAB);
        if (f && n >= 8) {
            int n4 = (n - 1) >> 2;
            f(x + 1, n4, &mx);
            i = 1 + (n4 << 2);
        }
    }
    for (; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = lz_exp(x[i] - mx);
        sum += x[i];
    }
    /* One division instead of n multiplies: x87's FDIV does not
       pipeline, and n divisions are a thousand-plus per token in
       attention scoring. Hardware-independent win, not an approximation. */
    {
        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        lz_vscalefn f = lz_vscale_pick(LZ_VSCALE_TAB);
        i = 0;
        if (f && n >= 4) {
            int n4 = n >> 2;
            f(x, n4, &inv);
            i = n4 << 2;
        }
        for (; i < n; i++) x[i] *= inv;
    }
}

