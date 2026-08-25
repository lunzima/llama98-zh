/* ops_norm.c - Normalization: RMSNorm, L2Norm, softmax, and the
   fixed-tier gated output loop. A separate TU from ops.c to keep #if
   guard density low. These call norm_ss_fixed/expof/q8_amax from
   ops_quant.c and the LZ_NORMOUT/VMAX/VSCALE dispatch tables from
   ops_kernel_norm.h. The fixed output loops are guarded by
   LZ_NORM_FIXED (ops.h). */
#include <stddef.h>
#include "lz_int.h"   /* <stdint.h> is not on the language floor */

/* LZ_NORM_MAX_N: ops.h, included below. */

#include "ops.h"
#include "err.h"
#include "mmx_compat.h"
#include "ops_mmx.h"
#include "ops_sse.h"     /* the three norm _w wrappers LZ_NORM_SLOTS pastes */
#include "ops_sse2.h"
#include "ops_quant.h"
#include "ops_sched.h"
#include "ops_kernel_shared.h"
#include "ops_norm.h"

#include "ops_kernel_norm.h"

/* The Q15 scale exponent for a norm, rounded UP: 14 - ceil(log2(x)).
   Mirrors gdn_build_table's "round the block UP" (ops_gdn.c) - the scale
   becomes a stable power-of-2 upper bound, so a 1-ULP cross-compiler
   difference in x (x87 vs SSE2) that straddles a power-of-2 boundary
   cannot flip the exponent and halve/double the whole Q15 chain. Costs
   one bit of Q15 range, which the gdn twin measured at +6e-4 NLL. */
static int norm_q15_exp(float x) {
    int e = expof(x);
    if (pow2f(e) < x) e++;
    return 14 - e;
}

/* ---- the (1 + w) table, cached per norm weight ------------------------

   The output loop's add does not depend on the token. lz_rmsnorm's
   `o[i] = x[i] * inv * (1.0f + w[i])` reads x and inv per call, but
   `1.0f + w[i]` reads only w - the norm weight tensor, the same object
   on every call for the life of the model. Measured on the ARM cross
   build (.prof/arm_prim_count.sh mode 14): the whole loop is 112
   instructions per element and a bare __aeabi_fadd is 43 of them, so
   hoisting the add leaves two multiplies and the loads.

   The path this serves is the one that runs. lz_rmsnorm_out_fixed_int
   below carries a Q15 version of the same hoist, but every call site
   that reaches it (forward.c, nine of them) is inside a SubLN branch;
   counted on .prof/x87test, 147,200 of 147,200 norm elements go through
   the float loop and none through the int one.

   Bit-identical by construction rather than by tolerance: the table
   holds exactly fl(1.0f + w[i]), which is what the inline expression
   computes, and float addition is deterministic. A miss runs the inline
   form, so both live in one build and .prof/norm_wp_xcheck.c compares
   them element by element.

   Keyed on the POINTER, sound for one reason worth stating: lz_t_f32
   returns t->f unchanged for an F32 tensor, and norm weights are non-2D,
   so the exporter never quantizes them and the pointer is the tensor's
   own storage. If that stops being true the key stops matching, the
   table is rebuilt every call, and the numbers stay right - the failure
   mode is slow, not wrong.

   Static and single-threaded, a bump allocator with no eviction, for the
   same reason the Q15 pool below is: a model has a fixed set of norm
   weights, so it either fits or it does not, and a cache that thrashed
   would be worse than the miss path. */
#ifndef LZ_WP_POOL
#define LZ_WP_POOL 32768        /* floats; CE uses 36 x 512 */
#endif /* LZ_WP_POOL */
#ifndef LZ_WP_SLOTS
#define LZ_WP_SLOTS 64
#endif /* LZ_WP_SLOTS */

static float wp_pool[LZ_WP_POOL];
static int wp_used = 0;
static struct { const float *w; int n; float *p; } wp_slot[LZ_WP_SLOTS];
static int wp_slots = 0;

/* Whether ss_over_n below takes its multiply arm. Also the census
 * predicate: a site that billed the divide flat would report one for
 * every call while the code performs one only for the lengths that are
 * not a power of two - and this tree has 512, 128 and 64 alongside the
 * MoE intermediate's 896, so both arms are live in the same run. */
/* MAYBE_UNUSED because its only callers are LZ_SSN_MUL and LZ_SSN_DIV
   just below, and both of those appear only inside LZ_FCX - which
   expands to nothing unless the build is a counting one. */
LZ_MAYBE_UNUSED static int ssn_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }
#define LZ_SSN_MUL(n) (ssn_pow2(n) ? 1 : 0)
#define LZ_SSN_DIV(n) (ssn_pow2(n) ? 0 : 1)

/* ss / n, without the divide when n is a power of two.
 *
 * A divide is not a multiply: 117 instructions against 30 on ARMv5TE
 * (.prof/arm_div_count.sh), while the f32 census counts both as 1. The
 * four RMSNorm entries below each spend one on `ss / (float)n`, and n is
 * a row length - 512, 128, 64 on every checkpoint here.
 *
 * EXACT, not close, and the guard is what makes it so rather than an
 * argument about which lengths occur: for a power-of-two n, (float)n is
 * exact and 1/n is exactly representable, so multiplying by 2^-log2(n)
 * only moves the exponent and lands on the same float the division
 * rounds to. Any other n takes the division, unchanged - so a model
 * with a length like 896 is not silently given a different number.
 *
 * expof((float)n) IS log2(n) here for the same reason: a power of two
 * has a zero mantissa, so its stored exponent is the answer. */
static float ss_over_n(float ss, int n) {
    if (n > 0 && (n & (n - 1)) == 0) return ss * pow2f(-expof((float)n));
    return ss / (float)n;
}

static const float *build_wp(const float *w, int n) {
    float *p;
    int i;
    for (i = 0; i < wp_slots; i++)
        if (wp_slot[i].w == w && wp_slot[i].n == n) return wp_slot[i].p;
    if (wp_slots >= LZ_WP_SLOTS || wp_used + n > LZ_WP_POOL) return NULL;
    p = wp_pool + wp_used;
    for (i = 0; i < n; i++) p[i] = 1.0f + w[i];
    wp_used += n;
    wp_slot[wp_slots].w = w;
    wp_slot[wp_slots].n = n;
    wp_slot[wp_slots].p = p;
    wp_slots++;
    return p;
}

#if LZ_NORM_FIXED
/* (a*b) >> 15 truncating, as int32, for the Q15 output chains. a and b
   are Q15 (|.| <= 32768) so a*b <= 2^30 and the shifted result always
   fits int32 - every intermediate in the four-step chains below stays in
   range, which is what lets the loop carry an int32 accumulator instead
   of an int64 one. gcc widens to lz_i64 and lets the hardware multiply;
   Watcom routes the natural `(lz_i64)a * b` through its __I8M library
   call and the shift through __I8RS, shuffling 64-bit register pairs
   for every step - the norm output loops measured 4 __I8M + 3 __I8RS
   per element against gcc's imul. The aux body is one imul and one
   shrd; the multiply is exact and the value fits int32 after the shift,
   so the two toolchains agree bit for bit. */
#if defined(__WATCOMC__)
static int32_t lz_i64mul_rsh15(int32_t a, int32_t b);
#pragma aux lz_i64mul_rsh15 = \
    "imul ebx" \
    "mov  ecx, 15" \
    "shrd eax, edx, cl" \
    "sar  edx, cl" \
    parm [eax] [ebx] \
    value [eax] \
    modify [ecx edx];
#else
#define lz_i64mul_rsh15(a, b) \
    (int32_t)((((lz_i64)(a) * (b)) >> 15))
#endif /* __WATCOMC__ */

/* Fixed output loop of lz_rmsnorm_gated. The full Q15 int-multiply:
   x, w, g each get their own power-of-two Q15 scale (q8_amax +
   pow2f(14-expof)), inv is scaled likewise, sigmoid comes straight from
   the Q15 table, and the product runs as an int32 accumulator through
   three fused >>15 multiplies. One dequantize at the end. The sig
   parameter is unused (kept in the signature for API stability). This
   is value-changing, not bit-identical to the float chain (the float
   chain's 5 per-step roundings cannot be reproduced by one final
   rounding), but it offloads the sigmoid work from the sigmoid census
   site entirely. forward.c uses lz_quantize_q8 (not lz_quantize_q8_silu)
   since the output already holds the final silu value. */
static void lz_rmsnorm_gated_out_fixed(float *o, const float *x,
                                       const float *g, const float *w,
                                       float inv, int n) {
    union { float f; uint32_t u; } bit;
    float sx, sw, sg;
    int ex, ew, eg, ei, oe;
    int32_t qinv;
    int i;

    bit.u = q8_amax(x, n); ex = norm_q15_exp(bit.f); sx = pow2f(ex);
    bit.u = q8_amax(w, n); ew = norm_q15_exp(bit.f); sw = pow2f(ew);
    bit.u = q8_amax(g, n); eg = norm_q15_exp(bit.f); sg = pow2f(eg);
    bit.u = q8_amax(&inv, 1); ei = norm_q15_exp(bit.f);
    oe = 30 - ex - ew - eg - ei;   /* the output exponent, loop-invariant */
    qinv = q8_round(inv * pow2f(ei));
    if (qinv > 32767) qinv = 32767;
    if (qinv < -32767) qinv = -32767;
    for (i = 0; i < n; i++) {
        int qx = q8_round(x[i] * sx);
        int qw = q8_round(w[i] * sw);
        int qg = q8_round(g[i] * sg);
        int32_t sigv = sigmoid_q15(g[i]);
        int32_t t;
        if (qx > 32767) qx = 32767; else if (qx < -32767) qx = -32767;
        if (qw > 32767) qw = 32767; else if (qw < -32767) qw = -32767;
        if (qg > 32767) qg = 32767; else if (qg < -32767) qg = -32767;
        t = qx * qw;
        t = lz_i64mul_rsh15(t, qg);
        t = lz_i64mul_rsh15(t, qinv);
        t = lz_i64mul_rsh15(t, sigv);
        {
            /* The exponent is loop-invariant and the scale is a power
               of two, so f32_scale_pow2 (ops_kernel_shared.h) does it
               as an exponent add - same number, no __aeabi_fmul, once
               per element. pow2f stays as the fallback for the zero
               and out-of-range cases it declines; t == 0 is ordinary
               here, so that arm is live. */
            float tf = (float)t, sc;
            o[i] = f32_scale_pow2(tf, oe, &sc) ? sc : tf * pow2f(oe);
        }
    }
}
#else
static void lz_rmsnorm_gated_out_fixed(float *o, const float *x,
                                       const float *g, const float *w,
                                       float inv, int n) {
    (void)o; (void)x; (void)g; (void)w; (void)inv; (void)n;
}
#endif /* LZ_NORM_FIXED */

/* Positive controls for the two switches on the gated int output loop,
   in ELEMENTS, and outside LZ_NORM_FIXED so a tier-off build still
   links. Both read 0 in their control build, which is the only thing
   that separates "the path ran" from "the path was compiled out": the
   gated chain's output is bit-identical across LZ_GATE_QW and close
   across LZ_GATE_SIG_I, so an output comparison cannot tell either apart
   on its own. eg_lo/eg_hi are the observed range of the gate's Q15
   exponent - sigmoid_q15_i's integer path has a band (LZ_SIG_EMAX) and
   falls back to the float coordinate outside it, so this is the
   measurement that says which one ran rather than an assumption that
   the domain holds. */
lz_i64 lz_debug_norm_sig_i = 0;
lz_i64 lz_debug_norm_qw = 0;
int lz_debug_norm_eg_lo = 32767;
int lz_debug_norm_eg_hi = -32768;

#if LZ_NORM_FIXED
/* ---- the norm weight's Q15 table, cached per weight pointer ----------

   The library calls that read only w do not depend on the token. Both
   int output loops below quantize a norm weight to Q15 - the gated one
   quantizes w, the plain one 1+w - and a norm weight tensor is the same
   object on every call for the life of the model. Measured on the ARM
   cross build, that pair is ~130 of the ~147 instructions per element
   the plain loop spends: one __aeabi_fadd (48) in its own loop, one
   __aeabi_fmul (28) plus one q8_round (48) in the main one, plus the two
   clamps. Hoisting them into a table built once leaves the int64 chain
   and the loads. The amax scan goes with them: it is compares and
   absolute values, outside the census's convention but not free.

   Keyed on the POINTER, PLUS a two-word content sentinel, and the
   sentinel is not belt-and-braces. lz_t_f32 returns t->f unchanged for
   an F32 tensor, which is what every norm weight in every checkpoint
   here is (.prof/dtype_by_role.c), so the pointer is the tensor's own
   storage and is stable. But for a QUANTIZED tensor lz_t_f32 dequantizes
   into the caller's scratch and returns THAT, so two different norm
   weights of the same length would arrive as the same pointer and a
   pointer-only key would hand the second one the first one's table -
   silently wrong, not slow. The sentinel (the bit patterns of w[0] and
   w[n-1]) catches that for any pair differing at either end; on a
   mismatch the lookup declines and the caller runs its inline arm, which
   is correct by construction and does not thrash the pool.

   Static, single-threaded, like the scratch below. The pool is a bump
   allocator with no eviction: a model has a fixed set of norm weights,
   so it either fits or it does not, and a cache that thrashed would be
   worse than the miss path. On a full pool build_qw returns NULL and the
   caller computes inline exactly as before - that path is not dead code,
   it is what any model past the bound runs. */
#ifndef LZ_QW_POOL
#define LZ_QW_POOL 32768        /* int16 entries; CE uses 36 x 512 */
#endif /* LZ_QW_POOL */
#ifndef LZ_QW_SLOTS
#define LZ_QW_SLOTS 64
#endif /* LZ_QW_SLOTS */

static short qw_pool[LZ_QW_POOL];
static int qw_used = 0;
static struct { const float *w; int n; int add1; int ew;
                uint32_t k0, k1; short *q; } qw_slot[LZ_QW_SLOTS];
static int qw_slots = 0;

/* The Q15 table for w[0..n), or NULL if the pool is full or the sentinel
   says this pointer is carrying different data than it did. add1 selects
   which table: 1 for the Q15 image of 1+w (lz_rmsnorm's chain), 0 for w
   itself (the gated chain). They are separate slots because they are
   different tables of the same tensor. *ew_out receives the shared
   exponent.

   The build is the SAME arithmetic the inline path ran, in the same
   order, so a cached run and an uncached one agree bit for bit.
   .prof/norm_wp_xcheck.c asserts that for both tables by driving more
   distinct weight pointers than the pool has slots, so the two arms run
   in one process and must agree; its mutation run reports the split.
   (An earlier version of this comment cited build/norm_qw_gate.sh, which
   was never written.) */
static const short *build_qw(const float *w, int n, int add1, int *ew_out) {
    union { float f; uint32_t u; } bit;
    static float wp[LZ_NORM_MAX_N];
    uint32_t k0, k1;
    float sw;
    short *q;
    int ew, i;

    bit.f = w[0];     k0 = bit.u;
    bit.f = w[n - 1]; k1 = bit.u;
    for (i = 0; i < qw_slots; i++)
        if (qw_slot[i].w == w && qw_slot[i].n == n &&
            qw_slot[i].add1 == add1) {
            if (qw_slot[i].k0 != k0 || qw_slot[i].k1 != k1) return NULL;
            *ew_out = qw_slot[i].ew;
            return qw_slot[i].q;
        }
    if (qw_slots >= LZ_QW_SLOTS || qw_used + n > LZ_QW_POOL) return NULL;
    q = qw_pool + qw_used;
    /* One fill per table rather than one fill over a `src` pointer: each
       branch is then textually the arithmetic ITS consumer's inline arm
       runs, which is what the bit-identity claim is about, and
       .prof/norm_wp_xcheck.sh has a mutation point per table. */
    if (add1) {
        for (i = 0; i < n; i++) wp[i] = 1.0f + w[i];
        bit.u = q8_amax(wp, n); ew = norm_q15_exp(bit.f); sw = pow2f(ew);
        for (i = 0; i < n; i++) {
            int v = q8_round(wp[i] * sw);
            if (v > 32767) v = 32767; else if (v < -32767) v = -32767;
            q[i] = (short)v;
        }
    } else {
        bit.u = q8_amax(w, n); ew = norm_q15_exp(bit.f); sw = pow2f(ew);
        for (i = 0; i < n; i++) {
            int v = q8_round(w[i] * sw);
            if (v > 32767) v = 32767; else if (v < -32767) v = -32767;
            q[i] = (short)v;
        }
    }
    qw_used += n;
    qw_slot[qw_slots].w = w;
    qw_slot[qw_slots].n = n;
    qw_slot[qw_slots].add1 = add1;
    qw_slot[qw_slots].ew = ew;
    qw_slot[qw_slots].k0 = k0;
    qw_slot[qw_slots].k1 = k1;
    qw_slot[qw_slots].q = q;
    qw_slots++;
    *ew_out = ew;
    return q;
}

/* Int output variant of lz_rmsnorm_gated_out_fixed: identical Q15
   int-multiply, but instead of dequantizing to float per element the
   Q15 result is written as int and the single power-of-two scale
   deq = 2^(30-ex-ew-eg-ei) is returned out-of-band. lz_quantize_q8_int
   then folds deq into the stored group scale only, so the Q15 -> float
   -> int8 double conversion disappears while staying bit-identical to
   the float chain (the cancellation proof lives at lz_quantize_q8_int).

   qw comes from the table above (LZ_GATE_QW); the local fill is the
   miss path, running the same arithmetic per call. The sigmoid reads the
   gate's Q15 image rather than the float g (LZ_GATE_SIG_I) - see
   src/ops.h for both switches and for why the second one is
   value-changing. */
static void lz_rmsnorm_gated_out_fixed_int(int32_t *t, const float *x,
                                    const float *g, const float *w,
                                    float inv, int n, float *deq) {
    static short qwl[LZ_NORM_MAX_N];
    union { float f; uint32_t u; } bit;
    const short *qwt = NULL;
    float sx, sw, sg;
    int ex, ew = 0, eg, ei;
    int32_t qinv;
    int i;

    bit.u = q8_amax(x, n); ex = norm_q15_exp(bit.f); sx = pow2f(ex);
    bit.u = q8_amax(g, n); eg = norm_q15_exp(bit.f); sg = pow2f(eg);
    bit.u = q8_amax(&inv, 1); ei = norm_q15_exp(bit.f);
    qinv = q8_round(inv * pow2f(ei));
    if (qinv > 32767) qinv = 32767;
    if (qinv < -32767) qinv = -32767;
    if (eg < lz_debug_norm_eg_lo) lz_debug_norm_eg_lo = eg;
    if (eg > lz_debug_norm_eg_hi) lz_debug_norm_eg_hi = eg;

#if LZ_GATE_QW
    qwt = build_qw(w, n, 0, &ew);
    if (qwt) lz_debug_norm_qw += n;
#endif /* LZ_GATE_QW */
    if (!qwt) {
        bit.u = q8_amax(w, n); ew = norm_q15_exp(bit.f); sw = pow2f(ew);
        for (i = 0; i < n; i++) {
            int v = q8_round(w[i] * sw);
            if (v > 32767) v = 32767; else if (v < -32767) v = -32767;
            qwl[i] = (short)v;
        }
        qwt = qwl;
    }
    for (i = 0; i < n; i++) {
        int qx = q8_round(x[i] * sx);
        int qw = qwt[i];
        int qg = q8_round(g[i] * sg);
        int32_t sigv;
        int32_t acc;
        if (qx > 32767) qx = 32767; else if (qx < -32767) qx = -32767;
        if (qg > 32767) qg = 32767; else if (qg < -32767) qg = -32767;
#if LZ_GATE_SIG_I
        sigv = sigmoid_q15_i(qg, eg);
#else
        sigv = sigmoid_q15(g[i]);
#endif /* LZ_GATE_SIG_I */
        acc = qx * qw;
        acc = lz_i64mul_rsh15(acc, qg);
        acc = lz_i64mul_rsh15(acc, qinv);
        acc = lz_i64mul_rsh15(acc, sigv);
        t[i] = acc;
    }
#if LZ_GATE_SIG_I
    lz_debug_norm_sig_i += n;
#endif /* LZ_GATE_SIG_I */
    *deq = pow2f(30 - ex - ew - eg - ei);
}
#else
static void lz_rmsnorm_gated_out_fixed_int(int32_t *t, const float *x,
                                    const float *g, const float *w,
                                    float inv, int n, float *deq) {
    (void)t; (void)x; (void)g; (void)w; (void)inv; (void)n;
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
               per call.

               The two switches subtract exactly what they remove, which
               is why they have to be 0 or 1 rather than merely nonzero:
               LZ_GATE_SIG_I drops the float coordinate the sigmoid was
               billed for (1 mul, 1 cvt per element) and LZ_GATE_QW drops
               w's q8_round (1 mul, 1 add). The amax scan either switch
               also removes is compares, outside this census's
               convention. */
            LZ_FCX(LZ_FC_NORM,
                   (4 - LZ_GATE_SIG_I - LZ_GATE_QW) * n + 1 + LZ_SSN_MUL(n),
                   (3 - LZ_GATE_QW) * n + 1, LZ_SSN_DIV(n),
                   (1 - LZ_GATE_SIG_I) * n + 1, 0);
            ss = ssf;
            goto have_ss_gated_int;
        }
    }
    /* float ss fallback. NOT RE-DERIVED, unlike the three other
       "historical total" sites corrected in this file and src/ops.c,
       and NOT for the reason that looks obvious. It is not that "the
       output loop is skipped entirely here": lz_rmsnorm_gated_out_fixed_int
       runs unconditionally eight lines down and has no site of its own,
       so this site has to cover it.
       Whether 5n is the right cover for an int output loop is an open
       question, not a settled one.
       Left alone deliberately: lz_norm_int() is -1 by default so
       nothing here runs in the shipping census, and replacing one
       unverified number with another unverified number is not a
       correction. */
    LZ_FCX(LZ_FC_NORM, 5 * n + LZ_SSN_MUL(n), n + 1, LZ_SSN_DIV(n), 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss_gated_int:
    inv = lz_rsqrt(ss_over_n(ss, n) + eps);
    lz_rmsnorm_gated_out_fixed_int(t, x, g, w, inv, n, deq);
}

#if LZ_NORM_FIXED
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

   Not bit-identical to the tier it replaces, unlike its gated
   counterpart, and the difference is structural rather than a bound.
   lz_rmsnorm_gated_out_fixed_int elides a round trip: its fixed twin
   computes the same Q15 product and then dequantizes it, so the two
   cancel. lz_rmsnorm has no such twin - its fixed tier replaces the sum
   of squares only and its output loop is float throughout - so this
   function INTRODUCES a Q15 chain rather than exposing one. Measured on
   kunkun-ce at 7% rms; see lz_norm_int in src/ops.h for the numbers and
   for how the flag's reach outgrew the proof. */
static void lz_rmsnorm_out_fixed_int(int32_t *t, const short *qx, int ex,
                              const float *w, float inv, int n,
                              float *deq) {
    /* static, not stack: the same single-threaded-forward-pass
       assumption norm_ss_fixed's own scratch rests on. Only the
       cache-miss path needs it. */
    static float wp[LZ_NORM_MAX_N];
    union { float f; uint32_t u; } bit;
    const short *qwt;
    float sw;
    int ew, ei;
    int32_t qinv;
    int i;

    /* qinv is per CALL - inv is this row's rsqrt - so it stays here. It
       is one scalar; the per-ELEMENT work is what the table removes. */
    bit.u = q8_amax(&inv, 1); ei = norm_q15_exp(bit.f);
    qinv = q8_round(inv * pow2f(ei));
    if (qinv > 32767) qinv = 32767;
    if (qinv < -32767) qinv = -32767;

    qwt = build_qw(w, n, 1, &ew);
    if (qwt) {
        for (i = 0; i < n; i++) {
            int32_t acc = qx[i] * qwt[i];           /* x*(1+w), unshifted */
            acc = lz_i64mul_rsh15(acc, qinv);       /* x*(1+w)*inv */
            t[i] = acc;
        }
        *deq = pow2f(15 - ex - ew - ei);
        return;
    }
    /* Pool exhausted. Same numbers, computed per call - see build_qw. */
    for (i = 0; i < n; i++) wp[i] = 1.0f + w[i];
    bit.u = q8_amax(wp, n); ew = norm_q15_exp(bit.f); sw = pow2f(ew);
    for (i = 0; i < n; i++) {
        int qw = q8_round(wp[i] * sw);
        int32_t acc;
        if (qw > 32767) qw = 32767; else if (qw < -32767) qw = -32767;
        acc = qx[i] * qw;                           /* x*(1+w), unshifted */
        acc = lz_i64mul_rsh15(acc, qinv);           /* x*(1+w)*inv */
        t[i] = acc;
    }
    *deq = pow2f(15 - ex - ew - ei);
}
#else
static void lz_rmsnorm_out_fixed_int(int32_t *t, const short *qx, int ex,
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
            LZ_FCX(LZ_FC_NORM, 2 * n + 2 + LZ_SSN_MUL(n), 3 * n + 2, LZ_SSN_DIV(n), 1, 0);
            ss = ssf;
            goto have_ss_int;
        }
    }
    /* float ss fallback - same billing as lz_rmsnorm's float row (the
       output loop runs unconditionally below regardless of which ss
       branch fired, same as lz_rmsnorm_gated_int's fallback). */
    LZ_FCX(LZ_FC_NORM, 3 * n + LZ_SSN_MUL(n), 2 * n + 1, LZ_SSN_DIV(n), 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss_int:
    inv = lz_rsqrt(ss_over_n(ss, n) + eps);
    lz_rmsnorm_out_fixed_int(t, qx, ex, w, inv, n, deq);
}

/* THE FLOAT ENTRY, and it is still where the norms census row lives.
 *
 * Measured on kmr20/zh, 603 tokens: 22,311 calls, 37 per token, over
 * five sites in forward.c - and the census puts `norms` at 106,772 float
 * operations per token, 17.8% of the engine's total, essentially all of
 * it here. lz_rmsnorm_int above is the same arithmetic writing int32
 * instead of dequantizing, so the gap is not a missing operator. It is
 * five consumers, each its own int-pipeline step, listed smallest first
 * so the next one to do is obvious:
 *
 *   final_norm            1 call/token. The NEXT STATEMENT after it is
 *                         lz_quantize_q8 on the value it just built -
 *                         a float constructed and taken apart again.
 *                         Blocked only on lz_matmul_xq's float argument,
 *                         which it reads when the weight is F32, and on
 *                         the fnorm tap.
 *   post_attention_norm   8 calls/token. Consumer is the MoE or dense
 *                         FFN entry, which quantizes internally.
 *   input_layernorm       8 calls/token. Consumer is one of THREE block
 *                         types (forward_attn, forward_kda, forward_ssm),
 *                         each quantizing on its own, plus a float tap -
 *                         so this one is three steps, not one.
 *   q_norm                16 calls/token, and k_norm 4. Consumer is
 *                         RoPE, which is float trigonometry; there is no
 *                         int entry to hand it to.
 *
 * Not started here because every one of them is an edit to forward.c.
 * The numbers are recorded so the next reader does not re-measure. */
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
            LZ_FCX(LZ_FC_NORM, 3 * n + 1 + LZ_SSN_MUL(n), n + 1, LZ_SSN_DIV(n), 1, 0);
            ss = ssf;
            goto have_ss;
        }
    }
    /* 2n accumulating ss, 2 for the divide and the eps add, 2n in the
       output loop's two multiplies. rsqrt bills itself. THE (1+w) ADD IS
       NOT HERE: it is billed where it is performed, below, because the
       cached table makes it a once-per-model cost rather than a
       per-token one and a census that billed it either way could not
       order this row against the rows it competes with. */
    LZ_FCX(LZ_FC_NORM, 3 * n + LZ_SSN_MUL(n), n + 1, LZ_SSN_DIV(n), 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss:
    inv = lz_rsqrt(ss_over_n(ss, n) + eps);
    /* (1 + w), not w: see the header */
    i = 0;
    {
        lz_normoutfn f = lz_normout_pick(LZ_NORMOUT_TAB);
        if (f && n >= 4) {
            float k2[2];
            int n4 = n >> 2;
            k2[0] = inv;
            k2[1] = 1.0f;
            LZ_FCX(LZ_FC_NORM, 0, n4 << 2, 0, 0, 0);   /* the kernel's +k2[1] */
            f(o, x, w, n4, k2);
            i = n4 << 2;
        } else {
            /* The table only pays when the scalar loop is the whole row.
               With a vector kernel above, what is left here is at most
               three tail elements, and a pool slot for three adds is a
               worse trade than the adds. */
            const float *wp = build_wp(w, n);
            if (wp) {
                for (; i < n; i++) o[i] = x[i] * inv * wp[i];
                return;
            }
        }
    }
    LZ_FCX(LZ_FC_NORM, 0, n - i, 0, 0, 0);
    for (; i < n; i++) o[i] = x[i] * inv * (1.0f + w[i]);
}

void lz_rmsnorm_gated(float *o, const float *x, const float *g,
                          const float *w, int n, float eps, int32_t *sig) {
    float ss = 0.0f;
    float inv;
    float ssf;
    int i;
    int fixed_ss = 0;
    (void)sig;   /* API stability: the fixed tier computes sigmoid inline */
    if (lz_norm_mode()) {
        ssf = norm_ss_fixed(x, n, NULL, NULL);
        if (ssf >= 0.0f) {
            /* Same replacement as lz_rmsnorm: norm_ss_fixed's own 2n+2
               in place of the 2n float ss loop. The fixed output loop
               (lz_rmsnorm_gated_out_fixed) computes the full Q15
               int-multiply (x*inv*w*silu) so the sigmoid site stops
               billing it. Per element: sigmoid_q15 (2 mul, 2 cvt, 2
               sub), q8_round per factor (3 mul, 3 add), int64 product
               (3 mul, 3 shift), dequant (1 cvt, 0 mul - the scale is a
               power of two and f32_scale_pow2 adds it to the exponent);
               plus norm_ss_fixed (n mul, n add, 1 mul, 1 cvt) and the
               divide+eps (1 div, 1 add). */
            LZ_FCX(LZ_FC_NORM, 4 * n + 1 + LZ_SSN_MUL(n), 3 * n + 1, LZ_SSN_DIV(n), 2 * n + 1, 0);
            ss = ssf;
            fixed_ss = 1;
            goto have_ss_gated;
        }
    }
    /* ss loop n mul + n add, divide+eps (1 div, 1 add), output loop
       x*inv*w*silu(g) - three multiplies, no add per element, since the
       gate replaces rmsnorm's (1+w) add with a multiply. silu bills
       itself. That is 4n mul, n+1 add, 1 div; the site billed 5n mul
       to hold an older total. Same correction as lz_l2norm's, and worth
       0 in the shipping census for the same reason - measured. */
    LZ_FCX(LZ_FC_NORM, 4 * n + LZ_SSN_MUL(n), n + 1, LZ_SSN_DIV(n), 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss_gated:
    inv = lz_rsqrt(ss_over_n(ss, n) + eps);
    /* plain w here, and the gate applies last. The fixed path computes
       the full silu value in Q15 int-multiply (value-changing); the
       float path is untouched. */
    if (fixed_ss)
        lz_rmsnorm_gated_out_fixed(o, x, g, w, inv, n);
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
            /* norm_ss_fixed: n mul (x[i]*sc) + n add (q8_round's magic
               add, and q8_round has no site of its own) + 1 mul + 1 cvt
               for (float)acc * pow2f(-2e). Then lz_rsqrt(ss + eps) is
               one add (rsqrt bills itself, and l2norm normalizes by the
               SUM, so there is no divide). Then the output loop is n
               mul, no weight.
                 mul  (n + 1) + n = 2n+1
                 add  n + 1
                 cvt  1        div 0

               2n+1, not 3n+1; that one n is the only correction here
               that mattered. Billing 3n+1 "to hold the site's
               historical total" overstates by n. Measured on kmr20 at
               603 tokens: the extra n is the difference between a NORMS
               class of 131,156 and 118,868 ops/token, 9.4% - because
               this function runs 192
               times a token (6 KDA layers x 16 heads x q and k, n=64)
               and 192*64 = 12,288 is exactly the delta. That was enough
               to make NORMS look like the largest class in the census
               (18.2%) when the dequant epilogue is (17.2% against
               16.8%). A ranking is what a census is FOR, so a formula
               kept for continuity with an older number was costing the
               thing the instrument exists to produce. */
            LZ_FCX(LZ_FC_NORM, 2 * n + 1, n + 1, 0, 1, 0);
            ss = ssf;
            goto have_ss_l2;
        }
    }
    /* ss loop n mul + n add, eps add (1, no divide - see the fixed
       branch), output loop n mul. Same correction as the fixed branch
       above and for the same reason; this one is worth 0 in the shipping
       census because lz_norm_mode() is on by default and this fallback
       does not run. Measured, not assumed: changing it alone moves the
       kmr20 census by zero. */
    LZ_FCX(LZ_FC_NORM, 2 * n, n + 1, 0, 0, 0);
    for (i = 0; i < n; i++) ss += x[i] * x[i];
have_ss_l2:
    inv = lz_rsqrt(ss + eps);          /* sum, not mean */
    for (i = 0; i < n; i++) o[i] = x[i] * inv;
}

/* lz_l2norm's int16 entry: x[i] * 2^-e is the value, and the caller
   producing it (lz_causal_conv1d_step_fixed's int16 exit) never built
   the float. Output stays FLOAT, and that is not an oversight - the
   consumer is gdn_build_table's `ss[kk*ng+gg] * q[kk]`, whose scale
   varies with BOTH indices, so there is no common factor to lift and an
   int16 handed over there costs a convert in that multiply. The chain
   ends here on purpose.

   WHAT IT SAVES, per element: the float entry's requantisation (a
   multiply and q8_round's magic add, inside norm_ss_fixed) and the
   producer's two converts. WHAT IT COSTS: one convert in the output
   loop, which the float entry does not pay because its x is already
   float. Net one convert and two operations an element - see
   docs/int-pipeline-project.md 9.4 for why the naive reading of this is
   twice as good as the real one.

   Eps is why this is not free. L2 normalisation is scale-invariant in
   mathematics, so "the exponent cancels" looks like an enabling fact.
   It is not: rsqrt is taken of ss + eps and eps is ABSOLUTE, so the
   integer sum of squares has to be brought to the true scale before it
   is added. That is the pow2f below and it is the whole reason this
   function needs e at all.

   The sum is EXACT: 32767^2 * n stays inside int64 for any n this
   engine uses (n = 64 gives 6.9e10, and even n = 4096 gives 4.4e12).
   So the only rounding before rsqrt is the single convert. */
void lz_l2norm_i16(float *o, const short *x, int n, int e, float eps) {
    lz_i64 acc = 0;
    float ss, inv, m;
    int i;
    /* n mul + n add for the squares are INTEGER and bill nothing, the
       same convention lz_swiglu_q15_i16's site uses. What bills is the
       one convert, the scale multiply, eps's add, and the output loop's
       convert-and-multiply per element. */
    LZ_FCX(LZ_FC_NORM, n + 2, 1, 0, n + 1, 0);
    /* (short)(short) fits int32 (32767^2 < 2^31), so widening after the
       multiply keeps Watcom on a 32-bit imul instead of __I8M for the
       same exact product - the same narrowing the output loops above
       rely on. */
    for (i = 0; i < n; i++) acc += (lz_i64)(x[i] * x[i]);
    /* (float)acc first, then the power of two: the product of two
       exponent-only factors is exact either way, but splitting them
       keeps the convert on a value whose magnitude the caller can bound
       from e alone. */
    ss = (float)acc * pow2f(-2 * e);
    inv = lz_rsqrt(ss + eps);
    m = inv * pow2f(-e);
    for (i = 0; i < n; i++) o[i] = (float)x[i] * m;
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
    /* The max scan runs in the integer domain. On a soft-float target
       `x[i] > mx` is a bl __aeabi_fcmpgt, ~30 instructions, once per
       element; the key below is three (lsr, orr, eor) and then a plain
       unsigned cmp.

       The mapping is the standard IEEE-754 order-preserving one: xor
       the sign-extended sign bit into every bit and force the top bit,
       so non-negatives land above 0x80000000 in increasing order and
       negatives below it in decreasing order. Comparing keys as
       UNSIGNED then reproduces `>` exactly - it is a permutation of the
       same total order, not an approximation, so this owes the float
       loop bit-identity and delivers it.

       NaN is the one input where it does not: float `>` is false for
       NaN so a plain loop never selects one, while NaN's key is the
       largest and this loop would. Same disposition ops.c's SSE2 q8
       round takes - NaN reaching attention scoring means the upstream
       is already destroyed, and no slow path is kept for it. */
    {
        union { float f; uint32_t u; } b;
        uint32_t best;
        b.f = mx;
        best = b.u ^ (((uint32_t)((int32_t)b.u >> 31)) | 0x80000000u);
        for (; i < n; i++) {
            uint32_t k;
            b.f = x[i];
            k = b.u ^ (((uint32_t)((int32_t)b.u >> 31)) | 0x80000000u);
            if (k > best) { best = k; mx = x[i]; }
        }
    }
    /* The subtract, then the exponentials, then the sum - three passes
       rather than one pass, because lz_exp_fixed's Taylor only has
       a kernel if it sees a block. The sum is a SEPARATE pass rather
       than folded into the exit loop so that its addition order is the
       one the single loop had; float addition is not associative and
       this owes that loop bit-identity, not a tolerance.
       LZ_SOFTMAX_RUN=0 keeps the per-element form as the control arm. */
#if LZ_SOFTMAX_RUN
    if (lz_scalar_mode()) {
        for (i = 0; i < n; i++) x[i] -= mx;
        lz_exp_fixed_run(x, x, n);
        for (i = 0; i < n; i++) sum += x[i];
    } else
#endif /* LZ_SOFTMAX_RUN */
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

