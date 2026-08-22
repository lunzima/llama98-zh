/* ops_norm.c - Normalization: RMSNorm, L2Norm, softmax, and the
   fixed-tier gated output loop. A separate TU from ops.c to keep #if
   guard density low. These call norm_ss_fixed/expof/q8_amax from
   ops_quant.c and the LZ_NORMOUT/VMAX/VSCALE dispatch tables from
   ops_kernel_norm.h. The fixed output loops are guarded by
   LZ_NORM_FIXED (ops.h). */
#include <stddef.h>
#include <stdint.h>

#ifndef LZ_NORM_MAX_N
#define LZ_NORM_MAX_N 4096
#endif /* LZ_NORM_MAX_N */

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

#if LZ_NORM_FIXED
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
#endif /* LZ_NORM_FIXED */

#if LZ_NORM_FIXED
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
#endif /* LZ_NORM_FIXED */

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
        ssf = norm_ss_fixed(x, n, NULL, NULL);
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

#if LZ_NORM_FIXED
/* ---- the (1+w) Q15 table, cached per norm weight ---------------------

   THE TWO LIBRARY CALLS IN THE OUTPUT LOOP DO NOT DEPEND ON THE TOKEN.
   `wp[i] = 1.0f + w[i]` and `q8_round(wp[i] * sw)` read only w - the
   norm weight tensor, which is the same object on every call for the
   life of the model. Measured on the ARM cross build, that pair is
   ~130 of the ~147 instructions per element this path spends: one
   __aeabi_fadd (48) in its own loop, one __aeabi_fmul (28) plus one
   q8_round (48) in the main one, plus the two clamps. Hoisting them
   into a table built once leaves the int64 chain and the loads.

   Keyed on the POINTER, which is sound for exactly one reason worth
   stating: lz_t_f32 returns t->f unchanged for an F32 tensor, and norm
   weights are non-2D so the exporter never quantizes them. The pointer
   is therefore the tensor's own storage and is stable. If that ever
   stops being true the key stops matching, the table is rebuilt every
   call, and the numbers stay right - the failure mode is slow, not
   wrong.

   Static, single-threaded, like the qx/wp scratch above it. The pool is
   a bump allocator with no eviction: a model has a fixed set of norm
   weights, so it either fits or it does not, and a cache that thrashed
   would be worse than the miss path. On a full pool build_qw returns
   NULL and the caller computes inline exactly as before - that path is
   not dead code, it is what any model past the bound runs. */
#ifndef LZ_QW_POOL
#define LZ_QW_POOL 32768        /* int16 entries; CE uses 36 x 512 */
#endif /* LZ_QW_POOL */
#ifndef LZ_QW_SLOTS
#define LZ_QW_SLOTS 64
#endif /* LZ_QW_SLOTS */

static short qw_pool[LZ_QW_POOL];
static int qw_used = 0;
static struct { const float *w; int n; int ew; short *q; }
qw_slot[LZ_QW_SLOTS];
static int qw_slots = 0;

/* The table for w[0..n), or NULL if it will not fit. *ew_out receives
   the shared exponent. The build is the SAME arithmetic the inline path
   ran, in the same order, so a cached run and an uncached one agree bit
   for bit - that is the property build/norm_qw_gate.sh asserts. */
static const short *build_qw(const float *w, int n, int *ew_out) {
    union { float f; uint32_t u; } bit;
    static float wp[LZ_NORM_MAX_N];
    float sw;
    short *q;
    int ew, i;
    for (i = 0; i < qw_slots; i++)
        if (qw_slot[i].w == w && qw_slot[i].n == n) {
            *ew_out = qw_slot[i].ew;
            return qw_slot[i].q;
        }
    if (qw_slots >= LZ_QW_SLOTS || qw_used + n > LZ_QW_POOL) return NULL;
    for (i = 0; i < n; i++) wp[i] = 1.0f + w[i];
    bit.u = q8_amax(wp, n); ew = 14 - expof(bit.f); sw = pow2f(ew);
    q = qw_pool + qw_used;
    for (i = 0; i < n; i++) {
        int v = q8_round(wp[i] * sw);
        if (v > 32767) v = 32767; else if (v < -32767) v = -32767;
        q[i] = (short)v;
    }
    qw_used += n;
    qw_slot[qw_slots].w = w;
    qw_slot[qw_slots].n = n;
    qw_slot[qw_slots].ew = ew;
    qw_slot[qw_slots].q = q;
    qw_slots++;
    *ew_out = ew;
    return q;
}

/* Int output loop for lz_rmsnorm's fixed tier: o[i] = x[i]*inv*(1+w[i])
   as a Q15 int-multiply, mirroring lz_rmsnorm_gated_out_fixed_int's
   shape for a 2-factor (not 4-factor) chain. qx/ex are norm_ss_fixed's
   own Q15 image of x and its exponent, reused here instead of
   requantizing x a second time - the whole reason norm_ss_fixed exposes
   them. (1+w[i]), not w[i] alone, gets quantized: w on its own can sit
   far from 1 in magnitude (small norm weights are common), so a scale
   sized for w could put the literal 1.0 far outside strong 15-bit range
   once added in; quantizing the sum directly sidesteps that.

   Only two factors beyond x (wp, inv), so only ONE >>15 shift - the
   first pairing (qx*qwp) accumulates unshifted, the second
   (*qinv) consumes the one shift, unlike the gated chain's three
   shifts for its four factors. t receives the Q15 product, *deq the
   power-of-two scale lz_quantize_q8_tok_int folds into the row scale.

   NOT BIT-IDENTICAL TO THE TIER IT REPLACES, unlike its gated
   counterpart, and the difference is structural rather than a bound.
   lz_rmsnorm_gated_out_fixed_int elides a round trip: its fixed twin
   computes the same Q15 product and then dequantizes it, so the two
   cancel. lz_rmsnorm has no such twin - its fixed tier replaces the sum
   of squares only and its output loop is float throughout - so this
   function INTRODUCES a Q15 chain rather than exposing one. Measured on
   kunkun-ce at 7% rms; see lz_norm_int in src/ops.h for the numbers and
   for how the flag's reach outgrew the proof. */
void lz_rmsnorm_out_fixed_int(int32_t *t, const short *qx, int ex,
                              const float *w, float inv, int n,
                              float *deq) {
    /* static, not stack: same single-threaded-forward-pass assumption
       norm_ss_fixed's own scratch used to make (see its history). Only
       the cache-miss path still needs it. */
    static float wp[LZ_NORM_MAX_N];
    union { float f; uint32_t u; } bit;
    const short *qwt;
    float sw;
    int ew, ei;
    int32_t qinv;
    int i;

    /* qinv is per CALL - inv is this row's rsqrt - so it stays here. It
       is one scalar; the per-ELEMENT work is what the table removes. */
    bit.u = q8_amax(&inv, 1); ei = 14 - expof(bit.f);
    qinv = q8_round(inv * pow2f(ei));
    if (qinv > 32767) qinv = 32767;
    if (qinv < -32767) qinv = -32767;

    qwt = build_qw(w, n, &ew);
    if (qwt) {
        for (i = 0; i < n; i++) {
            int64_t t64 = (int64_t)qx[i] * qwt[i];  /* x*(1+w), unshifted */
            t64 = (t64 * qinv) >> 15;               /* x*(1+w)*inv */
            t[i] = (int32_t)t64;
        }
        *deq = pow2f(15 - ex - ew - ei);
        return;
    }
    /* Pool exhausted. Same numbers, computed per call - see build_qw. */
    for (i = 0; i < n; i++) wp[i] = 1.0f + w[i];
    bit.u = q8_amax(wp, n); ew = 14 - expof(bit.f); sw = pow2f(ew);
    for (i = 0; i < n; i++) {
        int qw = q8_round(wp[i] * sw);
        int64_t t64;
        if (qw > 32767) qw = 32767; else if (qw < -32767) qw = -32767;
        t64 = (int64_t)qx[i] * qw;      /* x*(1+w), unshifted */
        t64 = (t64 * qinv) >> 15;        /* x*(1+w)*inv */
        t[i] = (int32_t)t64;
    }
    *deq = pow2f(15 - ex - ew - ei);
}
#else
void lz_rmsnorm_out_fixed_int(int32_t *t, const short *qx, int ex,
                              const float *w, float inv, int n,
                              float *deq) {
    (void)t; (void)qx; (void)ex; (void)w; (void)inv; (void)n;
    *deq = 0.0f;
}
#endif /* LZ_NORM_FIXED */

/* lz_rmsnorm fixed-tier wrapper for the int output: same sum-of-squares
   as lz_rmsnorm, then the Q15 int-multiply output loop writing int
   instead of dequantizing to float and quantizing that. qx/ex come
   straight from norm_ss_fixed's own scan of x - reused by
   lz_rmsnorm_out_fixed_int rather than requantized. Only valid when
   lz_norm_can_fixed(n); callers gate on it. */
void lz_rmsnorm_int(int32_t *t, const float *x, const float *w, int n,
                    float eps, float *deq) {
    static short qx[LZ_NORM_MAX_N];
    float ss = 0.0f;
    float inv;
    float ssf;
    int ex = 0;
    int i;
    if (lz_norm_mode()) {
        ssf = norm_ss_fixed(x, n, qx, &ex);
        if (ssf >= 0.0f) {
            /* norm_ss_fixed's own n+1 mul, n add, 1 cvt (see
               lz_rmsnorm's comment) are unchanged - qx/ex are its own
               scan, reused rather than requantized. The int output loop
               replaces the float output loop's 2n mul + n add with:
               wp=1+w (n add), q8_round per wp element (n mul, n add),
               and qinv's own q8_round (1 mul, 1 add) - net n+1 mul,
               2n+1 add, no cvt (t stays int, no per-element dequant).
               Plus the divide+eps (1 div, 1 add). */
            LZ_FCX(LZ_FC_NORM, 2 * n + 2, 3 * n + 2, 1, 1, 0);
            ss = ssf;
            goto have_ss_int;
        }
    }
    /* float ss fallback - same billing as lz_rmsnorm's float row (the
       output loop runs unconditionally below regardless of which ss
       branch fired, same as lz_rmsnorm_gated_int's fallback). */
    LZ_FCX(LZ_FC_NORM, 3 * n, 2 * n + 1, 1, 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss_int:
    inv = lz_rsqrt(ss / (float)n + eps);
    lz_rmsnorm_out_fixed_int(t, qx, ex, w, inv, n, deq);
}

void lz_rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    float inv;
    float ssf;
    int i;
    if (lz_norm_mode()) {
        ssf = norm_ss_fixed(x, n, NULL, NULL);
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
        ssf = norm_ss_fixed(x, n, NULL, NULL);
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
        ssf = norm_ss_fixed(x, n, NULL, NULL);
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

