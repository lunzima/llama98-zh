#include <stddef.h>   /* size_t: used a lot here for pointer offsets; math.h does not guarantee it */
#include <float.h>    /* FLT_MAX: overflow guard for quant reciprocal */
#include <math.h>
#include <stdlib.h>   /* malloc/free: the epilogue's derived scale plane
                         is the only allocation in this file. Watcom
                         reports W131 without it and -we makes that an
                         error; gcc picked the prototypes up from
                         elsewhere and said nothing. */
#include <string.h>   /* memcpy: unaligned loads in the MMX/SSE2 kernels */

#include "ops.h"
#include "err.h"
#include "cpucheck.h"   /* LZ_CPUID1_EDX_AUX - shared CPUID leaf 1 EDX asm */
#include "mmx_compat.h" /* _mm_empty(): called from the architecture-neutral
                           dispatch paths too, so it must be declared on
                           targets that have no MMX at all - see the header */
#include "ops_mmx.h"    /* declarations for src/ops_mmx.c's functions - the
                           %mm-touching kernels this file no longer defines.
                           Unconditional: the macros and declarations inside
                           are themselves guarded on LZ_MMX_TU, a build flag
                           (-D, from the Makefile) rather than on __MMX__,
                           which this file must never be compiled with. */
#include "ops_sse2.h"   /* declarations for src/ops_sse2.c's functions - the
                           %xmm-touching kernels this file no longer defines.
                           Same LZ_SSE2_TU/__SSE2__ contract, one ISA tier up. */
#include "ops_kernel_shared.h" /* lz_i32f, LZ_WSUM_CHUNK, LZ_SLOT_NEXT,
                           gdn_tail_row, p2_shift_of - shared with the
                           suffixed TUs (task #31); included this early
                           because lz_i32f's first use is well before
                           any of the ops_kernel_*.h #includes below. */

/* GDN pass-1 tier default, hoisted here from where the kernels live
   (~line 3250) because lz_gdn_mode() sits far above them and `#if
   LZ_GDN_FIXED` on an UNDEFINED macro is silently 0 - which made the
   accessor return "float" always AND, through it, disabled the fixed
   path outright. The knob's two settings then produced identical
   logits, i.e. it looked numerics-neutral instead of broken. Caught by
   printing the tier in the banner; a knob whose effect is not
   observable is indistinguishable from a knob that does nothing. */
#ifndef LZ_GDN_FIXED
#define LZ_GDN_FIXED 1
#endif /* LZ_GDN_FIXED */

#include "ops_quant.h"   /* pow2f, q8_round, q8_amax, sigmoid_q15,
                           norm_ss_fixed, lz_exp_float_body,
                           lz_rsqrt_float_body, lz_exp_fixed,
                           lz_rsqrt_fixed, LZ_Q8_MIN_SCALE_F,
                           LZ_Q8_INV127, LZ_Q15_INV */
#include "ops_sched.h"   /* lz_q8r_tier, q8r_have_simd */
#include "ops_matmul.h"  /* matmul impls, LZ_ROW_* tables, epilogue fns */

/* Accumulator bound for the fixed conv tier, derived not chosen.
   k products of two values bounded by B reach k*B^2, which must stay
   inside int32: B <= sqrt((2^31-1)/k). For k=4 that is 23170, so the
   largest usable power of two is 2^14. Probe (.prof/wconv.c) measures a
   peak of 9.09e8 against the 1.07e9 bound - tight, never crossed, 2.36x
   inside int32.

   int64 accumulation is not the escape: ARMv5TE has SMLAL, but MMX's
   pmaddwd yields int32 pairs that wrap the same way, so the x86 side
   would need a carry chain per tap. One bound for both is cheaper. */
int lz_conv_accum_bound(int k) {
    int b = 1;
    while ((double)k * (double)(b * 2) * (double)(b * 2) <= 2147483647.0)
        b *= 2;
    return b;
}

/* Normalize n floats into int16 by a POWER OF TWO, clamped to +-bound.
   Returns e, so v[i] ~ out[i] * 2^-e. Power of two only: the scale is
   exact in binary, so this adds no rounding of its own and cannot
   differ between x87 and SSE. */
int lz_conv_norm_pow2(const float *v, int n, short *out, int bound) {
    float amax = 0.0f;
    int i, e = 0;
    for (i = 0; i < n; i++) {
        float a = v[i] < 0.0f ? -v[i] : v[i];
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) {
        for (i = 0; i < n; i++) out[i] = 0;
        return 0;
    }
    while (e < 30 && amax * (float)(1 << (e + 1)) < (float)bound) e++;
    for (i = 0; i < n; i++) {
        float t = v[i] * (float)(1 << e);
        int q = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
        if (q > bound) q = bound;
        if (q < -bound) q = -bound;
        out[i] = (short)q;
    }
    return e;
}

#if LZ_F32COUNT
double lz_f32_class[LZ_FC_N][LZ_FCC_N];
int lz_f32_reached[LZ_FC_N];
#endif /* LZ_F32COUNT */

static const char *const LZ_FC_NAMES[LZ_FC_N] = {
    "dequant epilogue", "attention scoring", "attention wsum",
    "softmax", "exp", "sigmoid", "conv1d", "norms", "rsqrt",
    "recurrence write"
};

static const char *const LZ_FCC_NAMES[LZ_FCC_N] = {
    "mul", "add", "div", "cvt", "cmp", "other"
};

const char *lz_f32_count_name(int site) {
    return (site >= 0 && site < LZ_FC_N) ? LZ_FC_NAMES[site] : "?";
}

double lz_f32_count_get(int site) {
#if LZ_F32COUNT
    int c;
    double tot;
    if (site < 0 || site >= LZ_FC_N) return 0.0;
    tot = 0.0;
    for (c = 0; c < LZ_FCC_N; c++) tot += lz_f32_class[site][c];
    return tot;
#else
    (void)site;
    return -1.0;   /* not "zero work": the build cannot answer */
#endif /* LZ_F32COUNT */
}

double lz_f32_class_get(int site, int cls) {
#if LZ_F32COUNT
    if (site < 0 || site >= LZ_FC_N || cls < 0 || cls >= LZ_FCC_N) return 0.0;
    return lz_f32_class[site][cls];
#else
    (void)site; (void)cls;
    return -1.0;   /* not "zero work": the build cannot answer */
#endif /* LZ_F32COUNT */
}

const char *lz_f32_class_name(int cls) {
    return (cls >= 0 && cls < LZ_FCC_N) ? LZ_FCC_NAMES[cls] : "?";
}

int lz_f32_count_reached(int site) {
#if LZ_F32COUNT
    return (site >= 0 && site < LZ_FC_N) ? lz_f32_reached[site] : 0;
#else
    (void)site;
    return 0;
#endif /* LZ_F32COUNT */
}

void lz_f32_count_reset(void) {
#if LZ_F32COUNT
    int i, c;
    for (i = 0; i < LZ_FC_N; i++) {
        for (c = 0; c < LZ_FCC_N; c++) lz_f32_class[i][c] = 0.0;
        lz_f32_reached[i] = 0;
    }
#endif /* LZ_F32COUNT */
}

/* g_kernel defined in ops_sched.c; extern in ops_kernel_shared.h. */

#include "ops_kernel_norm.h"

/* Same home and reason: the fixed output loop of lz_rmsnorm_gated, defined
   beside norm_ss_fixed where sigmoid_q15's table and the Q15 helpers
   live. Writes the final silu value into o (Q15 int-multiply, value-
   changing). sig is unused (API stability). */
static void lz_rmsnorm_gated_out_fixed(float *o, const float *x,
                                       const float *g, const float *w,
                                       float inv, int n, int32_t *sig);

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

/* g_sig and the sigmoid tier selectors are in ops_sched.c. */

float lz_sigmoid(float x) {
    /* Branching avoids exp overflow at large |x|. Both branches take
       args <= 0, landing exactly in lz_exp's approximation contract.
       This is the absolute hot spot for exp: silu goes through it,
       conv1d needs 55K per token. */
    if (lz_sig_mode()) {
        /* Two multiplies and a subtract to reach the table, one convert
           and one multiply to come back - against 13 in lz_exp plus
           these three. */
        LZ_FCX(LZ_FC_SIGMOID, 3, 1, 0, 1, 0);
        return (float)sigmoid_q15(x) * (1.0f / 32768.0f);
    }
    /* Billed flat for both branches: add (1.0f+z) and divide. The
       positive branch also negates before calling lz_exp, but a negate
       is a sign flip (bills zero), so both branches are really 2 ops;
       the extra unit here is folded into add to hold the site's
       historical total - see B1 report. */
    LZ_FCX(LZ_FC_SIGMOID, 0, 2, 1, 0, 0);
    if (x >= 0.0f) {
        float z = lz_exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = lz_exp(x);
        return z / (1.0f + z);
    }
}

float lz_silu(float x) {
    LZ_FCX(LZ_FC_SIGMOID, 1, 0, 0, 0, 0);  /* x * sigmoid(x): one multiply */
    return x * lz_sigmoid(x);
}

/* ---- x87 precision control (meaningful only on Watcom/Win98) ----

   x87 by default leaves intermediates in 80-bit registers, so the same
   C yields different floats on x87 vs SSE - measured end-to-end logits
   cosine of only 0.9996, making cross-compiler bitwise differential
   testing impossible. With the control word's PC field set to 24 bits,
   **every arithmetic instruction rounds as float at the hardware
   level**, isomorphic to SSE, at zero instruction cost (no store
   needed after each step).

   The cost: double is also compressed to 24 bits, so during PC=24 no
   libm double routines may be called and the engine must not use
   double magic numbers - lz_exp is all-float for this.
   The few places that genuinely need double (lz_softplus) restore
   precision temporarily inside the function.

   The scope is deliberately narrowed to one lz_forward pass: loading,
   RoPE table building, tokenizing and sampling all sit outside the
   region and are unaffected; two fldcw per token is negligible.

   LZ_X87_FLOAT_CW gates the gcc/clang half of this: 32-bit x86 built
   WITHOUT SSE scalar float (the build/x87/build_engine.sh target -
   Socket-7-class, x87 the only float unit) has the exact same 80-bit-
   intermediate problem Watcom does, and __SSE_MATH__/__SSE2_MATH__
   being undefined is precisely how gcc says "this TU's scalar float
   goes through x87, not SSE" - the compiler defines them only when
   -mfpmath=sse is in effect, independent of whether -msse/-msse2 merely
   enable the instruction set. x86-64 (always __SSE2_MATH__) and any
   32-bit build that opts into SSE scalar float correctly skip this and
   fall to the same "already has float semantics" no-op Watcom's
   comment describes for gcc in general. */
#if defined(__i386__) && !defined(__SSE_MATH__) && !defined(__SSE2_MATH__)
#define LZ_X87_FLOAT_CW 1
/* lz_x87_cw_get/_set defined in ops_quant.c. */
extern unsigned short lz_x87_cw_get(void);
extern void lz_x87_cw_set(unsigned short cw);
#endif /* __i386__ && !__SSE_MATH__ && !__SSE2_MATH__ */

unsigned lz_fpu_float_begin(void) {
#if defined(__WATCOMC__)
    unsigned save = _control87(0, 0);
    _control87(_PC_24, _MCW_PC);
    return save;
#elif defined(LZ_X87_FLOAT_CW)
    unsigned short save = lz_x87_cw_get();
    lz_x87_cw_set((unsigned short)(save & ~0x0300u));   /* PC=24 */
    return save;
#else
    return 0;   /* gcc/SSE already has float semantics */
#endif /* __WATCOMC__ */
}

void lz_fpu_float_end(unsigned save) {
#if defined(__WATCOMC__)
    _control87(save, _MCW_PC);
#elif defined(LZ_X87_FLOAT_CW)
    lz_x87_cw_set((unsigned short)save);
#else
    (void)save;
#endif /* __WATCOMC__ */
}

float lz_softplus(float x) {
    /* Stable form of log(1+exp(x)): for large x it degenerates to x,
       avoiding exp overflow. In this model a + dt_bias can take fairly
       large positive values; the naive form overflows to inf. */
    if (x > 20.0f) return x;
#if defined(__WATCOMC__)
    {
        /* libm's double routines return wrong values under PC=24;
           restore 64-bit precision here temporarily. Only nv x linear
           layers = 144 calls per token; two fldcw at ~50 cycles is
           ~14K cycles (48 us on PII), not worth writing our own log. */
        unsigned save = _control87(0, 0);
        float r;
        _control87(_PC_64, _MCW_PC);
        r = (float)log1p(exp((double)x));
        _control87(save, _MCW_PC);
        return r;
    }
#elif defined(LZ_X87_FLOAT_CW)
    {
        /* Same reason as the Watcom branch above: PC=24 is in effect
           for the whole lz_forward pass on this target too now, and
           glibc's double log1p/exp read wrong under it exactly like
           Watcom's do. */
        unsigned short save = lz_x87_cw_get();
        float r;
        lz_x87_cw_set((unsigned short)(save | 0x0300u));   /* PC=64 */
        r = (float)log1p(exp((double)x));
        lz_x87_cw_set(save);
        return r;
    }
#else
    return (float)log1p(exp((double)x));
#endif /* __WATCOMC__ */
}

/* q8_round, lz_rsqrt_float_body, lz_rsqrt_fixed and their tables
   are in ops_quant.c. */

float lz_rsqrt(float x) {
#if defined(LZ_EXACT_MATH)
    return 1.0f / (float)sqrt((double)x);
#else
    if (lz_scalar_mode())
        return lz_rsqrt_fixed(x);
    LZ_FCX(LZ_FC_RSQRT, 7, 2, 0, 0, 0);   /* the halving plus 2 x (3 mul, 1 sub) */
    return lz_rsqrt_float_body(x);
#endif /* LZ_EXACT_MATH */
}

/* lz_exp_float_body, lz_exp_fixed, LZ_EXP_TAB32, LZ_EXP_LOG2E32,
   LZ_EXP_LO and their tables are in ops_quant.c. */

float lz_exp(float x) {
#if !defined(LZ_EXACT_MATH)
    if (lz_scalar_mode())
        return lz_exp_fixed(x);
    return lz_exp_float_body(x);
#else
    return (float)exp((double)x);
#endif /* LZ_EXACT_MATH */
}

/* ---- Q8 quantization per-element rounding: SSE2 version (32 at a time) ----

   The scalar version is ~12 instructions per element, of which the
   `fstp [tmp]; mov eax,[tmp]` pair is the price of magic-number
   rounding dodging control-word rewrites - the control word is
   avoided, the store/load is not.

   SSE2's `cvtps2dq` does this directly, and **the default MXCSR is
   round-half-even, the same rule as the magic add** - so this is not
   "numerically close" but BIT-IDENTICAL: exhaustively checked every
   quarter-integer within ±127, 16 ULPs on each side of
   every half-integer, the zero neighborhood (incl. ±0 and subnormals),
   and 40K random groups - 1,297,000 elements, 0 mismatches, verified
   on both compilers.

   Measured (same method, min of 7 rounds):

     Open Watcom 32-bit x87   24.50 ms -> 0.89 ms   27.4x
     gcc -O2 (SSE)            3.06 ms -> 0.80 ms    3.8x

   The only behavioral difference is NaN input: the scalar gives 0,
   SSE2's cvtps2dq gives 0x80000000 for unrepresentable inputs,
   saturating to -127. Both are garbage; NaN reaching here in
   production means the upstream is already destroyed - no slow path
   kept for it.

   The clamping stays. On the fast path |grp[k]| <= amax and
   inv = 127/amax, so qi is always in [-127,127] and pminsw/pmaxsw are
   dead code; but packsswb's saturation boundary is -128, not -127 -
   keeping these 4 instructions is what makes this match the scalar
   reference on all non-NaN inputs.

   **No emms emitted**: xmm and x87/MMX do not share register files
   (rule 6.6 only covers MMX), so this kernel can be called straight
   from lz_gdn_step's x87 section.

   No PIII tier - `cvtps2dq` is SSE2;
   SSE1 only has 2-lane `cvtps2pi` (writes MMX registers, emms
   management needed), that is a separate build. */
/* Included HERE, not at the top: the position is the whole point. These
   kernels see exactly the declarations in scope at this point, so
   including them here is provably a no-op - the code section is
   byte-identical across all three builds. Hoisting the #include would
   change that. */
#include "ops_kernel_q8round.h"

/* lz_q8r_tier and q8r_have_simd: declared in ops_sched.h,
   defined in ops_sched.c. */
/* Cached CPUID MMX bit. Non-static: ops_quant.c's q8_amax needs it.
   Declared UNconditionally: _mm_empty() compiles on ARM too (as a no-op),
   and the call sites test lz_cpu_has_mmx() there, so the name must exist
   on every target - it just returns 0 where there is no MMX to have. */
int lz_cpu_has_mmx(void);

/* lz_i32f (from ops_kernel_shared.h), q8_amax, q8_group_scale,
   LZ_Q8_MIN_SCALE_F, LZ_Q8_INV127 and LZ_Q15_INV are in
   ops_quant.c. lz_quantize_q8 stays here (it uses the dispatch
   tables and lz_q8r_tier). */

void lz_quantize_q8(const float *x, int n, int gs, int8_t *q, float *s) {
    int g, k;
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
    /* The rounding tier is selected once for the whole call, not per group */
    int tier = ((gs & 31) == 0) ? lz_q8r_tier() : 0;
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */
    if (gs <= 0 || n < gs || (n % gs) != 0) return;   /* defense: f32 weights have gs=0 etc. */
    for (g = 0; g < n / gs; g++) {
        const float *grp = x + (size_t)g * gs;
        int8_t *out = q + (size_t)g * gs;
        float inv;
        int fast = q8_group_scale(grp, gs, &s[g], &inv);
        if (fast) {
            /* One division per group (the reciprocal); everything inside
               is multiplies. x87's FDIV is 39 cycles vs FMUL's 5 - this
               step is worth far more on a PII than on this machine. */
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
            if ((gs & 31) == 0 && tier) {
#if defined(__WATCOMC__)
                /* task #30 (4/N): one call per GROUP, not per 32-element
                   sub-chunk - the #pragma aux bodies used to expand
                   inline at these call sites, once per sub-chunk, with
                   zero real call overhead. lz_q8round_group_sse2/_sse do
                   the WHOLE gs/32-chunk loop for this group (including,
                   for the SSE1+MMX tier, the emms the comment below
                   used to place right here - moved with the loop, same
                   position relative to the next group's x87 work). */
#ifdef LZ_HAVE_Q8R_SIMD
                if (tier == 2)
                    lz_q8round_group_sse2(grp, out, gs, &inv);
                else
#endif /* LZ_HAVE_Q8R_SIMD */
                    lz_q8round_group_sse(grp, out, gs, &inv);
#else
#ifdef LZ_HAVE_Q8R_SIMD
                if (tier == 2) {
                    for (k = 0; k + 31 < gs; k += 32)
                        lz_q8round32_simd(grp + k, out + k, &inv);
                } else
#endif /* LZ_HAVE_Q8R_SIMD */
                {
                    for (k = 0; k + 31 < gs; k += 32)
                        lz_q8round32_sse(grp + k, out + k, &inv);
                    /* **This emms must be INSIDE the group loop**, not
                       moved to the function tail. Rule 6.6's "kernels
                       don't emit; the caller emits once after its loop"
                       assumes no x87 in the loop body; that does not
                       hold here - the next group's
                       `s[g] = amax*(1/127)` and `inv = 127/amax` are
                       both x87, while after cvtps2pi writes MMX the x87
                       tag word reads "all full", so the next fld
                       overflows the stack and reads garbage.

                       Measured cost: this bug is COMPLETELY INVISIBLE
                       on the gcc build (x86-64 floats go through SSE;
                       dirty MMX state has no effect); only Watcom/x87
                       end-to-end differential testing catches it. Rule
                       2 is not formalism. */
                    if (lz_cpu_has_mmx()) _mm_empty();
                }
#endif /* __WATCOMC__ */
            } else
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */
            for (k = 0; k < gs; k++) {
                int qi = q8_round(grp[k] * inv);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        } else {
            for (k = 0; k < gs; k++) {
                int qi = q8_round(grp[k] / s[g]);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        }
    }
}

/* Fused fixed-tier gated-RMSNorm quantize (see ops.h). The norm wrote
   the pre-silu product into p (ssm_out) and the Q15 sigmoid into sig;
   this reconstructs the silu value and quantizes it, so the Q15 ->
   float -> int round-trip the plain lz_quantize_q8 used to re-quantize
   is gone while the value stays bit-identical.

   Why the 2^-15 folds into the SCALE and not the element loop: amax
   runs on the pre-scale product u[i] = p[i]*(g[i]*(float)sig[i]), and
   amax*2^-15 == max|u*2^-15| exactly because 2^-15 is a power of two.
   The stored s[] and the round then match lz_quantize_q8's bit for bit
   (u*(127/amax) == (u*2^-15)*(127/(amax*2^-15)) - both round the same
   exact product once). o doubles as the amax-pass scratch and receives
   the reconstructed float. */
void lz_quantize_q8_silu(const float *p, const float *g, const int32_t *sig,
                         int n, int gs, int8_t *q, float *s, float *o) {
    int grp, k;
    LZ_FCX(LZ_FC_NORM, 3 * n, 0, 0, n, 0);
    if (gs <= 0 || n < gs || (n % gs) != 0) {
        /* f32-weight downstream: no quantization; reconstruct the silu
           value into o only (the f32 matmul fallback reads it). */
        if (o) {
            for (k = 0; k < n; k++)
                o[k] = (p[k] * (g[k] * (float)sig[k])) * (1.0f / 32768.0f);
        }
        return;
    }
    for (grp = 0; grp < n / gs; grp++) {
        const float *pg = p + (size_t)grp * gs;
        const float *gg = g + (size_t)grp * gs;
        const int32_t *sg = sig + (size_t)grp * gs;
        int8_t *out = q + (size_t)grp * gs;
        float *og = o + (size_t)grp * gs;   /* u[] scratch, then o[] */
        union { float f; uint32_t u; } bit;
        uint32_t am;
        float amax, amax_o, inv;
        int fast;
        /* amax pass: the pre-scale product, in-place over p (element i
           depends only on p[i], read before the write). The 2^-15 is
           not applied here - it folds into the scale below. */
        for (k = 0; k < gs; k++)
            og[k] = pg[k] * (gg[k] * (float)sg[k]);
        am = q8_amax(og, gs);
        bit.u = am;
        amax = bit.f;
        if (amax > 0.0f) {
            /* amax_o = amax*2^-15 exactly; s[] is the same stored scale
               lz_quantize_q8 would write for the float silu output. */
            amax_o = amax * (1.0f / 32768.0f);
            s[grp] = amax_o * LZ_Q8_INV127;
            /* Round-scale: u*inv == o*inv_o (see the header comment). */
            inv = 127.0f / amax;
            fast = (127.0f / amax_o <= FLT_MAX);
            if (s[grp] < LZ_Q8_MIN_SCALE_F) {
                s[grp] = 1.0f;
                inv = 0.0f;
                fast = 1;
            }
        } else {
            s[grp] = 1.0f;
            inv = 0.0f;
            fast = 1;
        }
        if (fast) {
            for (k = 0; k < gs; k++) {
                int qi = q8_round(og[k] * inv);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        } else {
            for (k = 0; k < gs; k++) {
                int qi = q8_round((og[k] * (1.0f / 32768.0f)) / s[grp]);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        }
        /* Reconstructed silu value, for the f32 matmul fallback and the
           sgdn tap. The 2^-15 multiply is exact, so this is the same
           float the norm's own output loop used to write. */
        for (k = 0; k < gs; k++)
            og[k] = og[k] * (1.0f / 32768.0f);
    }
}

#include "ops_kernel_dot.h"

/* ---------------------------------------------- runtime kernel dispatch

   One Win98 binary must run on both a PII (MMX only) and a T42/Pentium M
   (SSE2), so both asm kernels are linked in and selected once at startup
   via CPUID.

   **The dispatch point is at the matmul level**: one branch per matmul,
   after which the 32-element kernel inside the row loop is a DIRECT call.
   Dispatching deeper would pay one indirect call per 32 MACs — exactly
   what this kernel design exists to avoid. */
/* Defined near the top of the file, above q8_amax, which also reads it. */

#if defined(__WATCOMC__)
/* CPUID leaf 1 EDX: bit 23 = MMX, bit 25 = SSE, bit 26 = SSE2. One
   symbol, one implementation, in cpucheck.c - this file calls it through
   the cpucheck.h declaration. A real __asm-block function, not a
   duplicated #pragma aux (see cpucheck.h for why those two cannot share
   a body across files). */
#endif /* __WATCOMC__ */

/* g_pf, lz_cpuid_ext_edx, CPUID caching, g_has_sse, lz_cpu_has_sse:
   moved to ops_sched.c. */

/* CPUID leaf 1 EDX bit 23 = MMX. Cached for the same reason g_has_sse is:
   the _mm_empty() call sites sit in row/group loops and would turn into a
   serializing cpuid per row otherwise. Bit 23, NOT bit 25: Pentium MMX
   has MMX without SSE, and a machine that has MMX is exactly the machine
   where an unconditional emms is a legal (if redundant) instruction. On a
   machine WITHOUT MMX (486DX, early Pentium) this is false and every
   unconditional emms that followed a scalar row is skipped - the fix for
   "emms is an MMX instruction, so it is #UD on an MMX-less CPU". Defined
   on every target (not just x86): the call sites test it on ARM too, where
   _mm_empty() is a no-op and this returns 0 so the test compiles away. */
#if defined(__WATCOMC__) || defined(__i386__) || defined(__x86_64__)
static int g_has_mmx = -1;
int lz_cpu_has_mmx(void) {
    if (g_has_mmx < 0) g_has_mmx = (lz_cpuid1_edx() & (1u << 23)) != 0;
    return g_has_mmx;
}
#else
int lz_cpu_has_mmx(void) { return 0; }
#endif /* __WATCOMC__ || __i386__ || __x86_64__ */

/* pf_detect, lz_prefetch_mode, g_pair, lz_pair_mode: in ops_sched.c. */

/* All tier globals, selectors, CPU detection, kernel dispatch:
   moved to ops_sched.c. lz_epi_ready stays here. */

/* Derived bound for the fixed epilogue: per group the product is
   |acc| * |xw*ww| <= 2^27 * 2^30, so ng of them stay inside
   int64 while ng <= 64. */
#define LZ_EPI_MAX_NG 64

int lz_epi_ready(const LZTensor *w) { return w && w->sq && w->sexp; }

/* Only for the half-built case below; lz_t_free owns the normal path. */
static void epi_drop(LZTensor *w) {
    free(w->sq);   w->sq = NULL;
    free(w->sexp); w->sexp = NULL;
}

/* One shared exponent per ROW, not per tensor: .prof/wepirange.c
   measured the per-row spread of the weight scales at 99.6% within 8x,
   while across rows it is far wider. A per-tensor exponent would spend
   that whole range on every row. */
void lz_epi_prep(LZTensor *w, int in_dim) {
    int ngw, rows, row, g;
    if (!w || !w->scale || w->gs <= 0 || in_dim <= 0) return;
    if (w->sq) return;                       /* idempotent */
    if (in_dim % w->gs) return;
    ngw = in_dim / w->gs;
    if (in_dim / 32 > LZ_EPI_MAX_NG) return; /* outside the derived bound */
    rows = w->n / in_dim;
    if (rows <= 0) return;
    w->sq   = (int16_t *)malloc((size_t)ngw * (size_t)rows * sizeof(int16_t));
    w->sexp = (signed char *)malloc((size_t)rows);
    if (!w->sq || !w->sexp) { epi_drop(w); return; }
    for (row = 0; row < rows; row++) {
        const float *ws = w->scale + (size_t)row * ngw;
        int16_t *out = w->sq + (size_t)row * ngw;
        float amax = 0.0f;
        int e = 0;
        for (g = 0; g < ngw; g++) {
            float a = ws[g] < 0.0f ? -ws[g] : ws[g];
            if (a > amax) amax = a;
        }
        if (amax <= 0.0f) {
            for (g = 0; g < ngw; g++) out[g] = 0;
            w->sexp[row] = 0;
            continue;
        }
        while (e < 100 && amax * pow2f(e + 1) < 32767.0f) e++;
        for (g = 0; g < ngw; g++) {
            float t = ws[g] * pow2f(e);
            int q = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
            if (q >  32767) q =  32767;
            if (q < -32767) q = -32767;
            out[g] = (int16_t)q;
        }
        w->sexp[row] = (signed char)e;
    }
}

/* int32 x int32 -> int64, which C cannot ask for: `(long long)a *
   (long long)b` is 64x64 by the standard and Watcom emits `call __I8M`,
   2.75x the cost of the one `imul r/m32` x86 has had since the 386. */
#if defined(__WATCOMC__)
long long lz_epi_mac(const int32_t *a, const int32_t *m, int n);
/* ebp is pushed and popped rather than declared modified - it is the
   frame pointer, and the rule is that modify tells
   the truth. imul owns edx:eax, so the two accumulator halves are ebx
   and ebp. */
#pragma aux lz_epi_mac =                \
    "push ebp"                          \
    "xor ebx,ebx"                       \
    "xor ebp,ebp"                       \
    "L1:"                               \
    "mov eax,[esi]"                     \
    "mov edx,[edi]"                     \
    "imul edx"                          \
    "add ebx,eax"                       \
    "adc ebp,edx"                       \
    "add esi,4"                         \
    "add edi,4"                         \
    "dec ecx"                           \
    "jnz L1"                            \
    "mov eax,ebx"                       \
    "mov edx,ebp"                       \
    "pop ebp"                           \
    parm [esi] [edi] [ecx]              \
    value [edx eax]                     \
    modify [eax ebx ecx edx esi edi 8087];
#else
static long long lz_epi_mac(const int32_t *a, const int32_t *m, int n) {
    long long s = 0;
    int g;
    for (g = 0; g < n; g++) s += (long long)a[g] * (long long)m[g];
    return s;
}
#endif /* __WATCOMC__ */


/* Cap for the activation pre-expansion buffers (element count). Above
   it, fall back to the non-expanded path. 4096 covers the 0.8B's max
   in_dim (intermediate 3584) and s1v2's 2048; 8KB on the stack, only
   present in SSE2 builds (the MMX tier has its own cap). */
/* These pre-expansion buffers MUST be static, not on the stack.
   8KB xw[] plus 512B acc32[] pushes the stack pointer two pages down
   on function entry, and Open Watcom's -ox includes -s (no stack
   probes); skipping the guard page reads uncommitted garbage - measured
   end-to-end logits becoming nondeterministic garbage (rms
   5418/5415/5412 drifting run to run), while -ot or -od works. Win98's
   default stack is even tighter than NT's; relying on compiler flags
   is not safe. The engine is single-threaded and non-recursive, so
   static is safe. */
/* LZ_MM_WIDEN_MAX + scratch moved to ops_matmul.c. */

/* Cap for the activation pre-expansion buffers (element count).
   Defined here (moved with the scratch buffers to ops_matmul.c; kept
   here too because lz_matmul_xq_nt checks it). */
#define LZ_MM_WIDEN_MAX 4096

/* ---- prefetch (lever J) ------------------------------------------------
   LZ_PF_DIST, LZ_PFI_LOAD/NTA/AMD/NONE and the lz_pf_load/lz_pf_nta/
   lz_pf_amd pragmas moved to src/ops_kernel_shared.h (task #30, 5/N):
   every format's row function needs them, wherever it ends up living,
   the same reasoning LZ_WSUM_CHUNK/LZ_SLOT_NEXT already established.
   Full rationale (why three tiers, why the distance is a knob, why
   "no prefetch" is a real mode and not a fallback) lives there now. */

/* LZ_SSE2_GROUP (the -DLZ_SSE2_GROUP=0 A/B control for the 128-element
   SSE2 group kernels) moved to src/ops_kernel_shared.h (task #30,
   5/N): the row functions that test it are migrating out to
   src/ops_mmx.c/src/ops_sse2.c one format at a time, and this needs to
   be visible wherever each one ends up, same reasoning as LZ_PF_DIST,
   the LZ_PFI_ family and LZ_PF_NONE above. */

/* ---- row-kernel tier tables (stage 3) ---------------------------------

   ONE signature for every (format x tier). A row function fills
   acc32[tk*nb + s] for ONE weight row across nt tokens and touches no
   float at all - the float work is epi_q8 / epi_q41, which stage 4
   already reduced to one copy each.

   w4/w2 are the weight row's planes (w2 is the Q6_1 2-bit plane, NULL
   elsewhere); pf_end is the prefetch upper bound - PIII's prefetchnta
   is a harmless hint out of bounds but PII's dummy load is a REAL
   access and segfaults on an unmapped page.

   THE DISPATCH IS AT ROW LEVEL, never inside. At in_dim=1024 that is
   one indirect call per 32 kernel calls; one per 32 MACs would be the
   disaster the file's opening comment warns about. */
/* lz_rowfn typedef + fwd decl moved to ops_matmul.c. */

/* The seven variants, as table slots. A build fills at
   most three of them (gcc cannot compile #pragma aux; Watcom's _m_*
   intrinsics are 6x slower than its own x87 scalar), and REF is not
   here at all - lz_matmul_xq_nt intercepts that tier before reaching
   any impl, so that the scalar reference stays a real oracle.

   An unfilled slot is NULL and the pick falls back, loudly rather than
   silently: a tier that cannot be reached must not report itself as
   the tier that ran. That confusion is the whole reason this file is
   being restructured. */
/* The slot constants are defined near the top of the file, beside
   LZ_DEFINE_PICK, because the operator tables there are read first. */

/* ISA from the runtime tier, impl from the build - the same join
   lz_kernel_tier() reports, in one place instead of four #if mazes.

   THE SSE SLOTS ARE EMPTY BY INSTRUCTION SET, NOT BY BACKLOG, and that
   distinction is the whole reason gaps need a reason - checked against
   the actual kernels rather than reasoning about the ISA.
   A row kernel's operation sequence is pand / psrlw / punpcklbw(z,.) /
   punpckhbw(z,.) / pmaddwd / paddd. SSE1's additions to the MMX integer
   set are pshufw, pinsrw, pextrw, pmovmskb, pmulhuw, pavgb, pavgw,
   pmaxsw, pminsw, pmaxub, pminub, psadbw, maskmovq, movntq. None of
   them replaces a step or removes one:
     pshufw  only permutes existing 16-bit lanes; it cannot zero-extend
             bytes to int16, so it cannot stand in for punpck
     pextrw  is 16-bit; the horizontal reduction here is 32-bit
     pmaxsw  belongs to lz_quantize_q8, not to matmul
     movntq  would write acc32, which the float epilogue reads on the
             very next line - a non-temporal store is a pessimisation
             exactly where the data is reused immediately
   So a PIII runs the MMX kernel, and asking for SSE lands there - one
   readable line with the reasoning attached, rather than an accident of
   #if nesting. Prefetch - the one thing SSE really does add here - is
   orthogonal and lives in --prefetch. */
/* lz_row_pick moved to ops_matmul.c. */

/* THE Q8 FLOAT EPILOGUE - one copy, shared by every kernel tier.
   accb[g] holds the g-th 32-element sub-block's integer sum SCALED BY
   256 (what part32_x16 and the MMX/asm kernels all produce); this turns
   it into one output element.

   The SSE2-intrinsics branch fills acc32 rather than fusing integer and
   float work, so it can be expressed as a row function that "only fills
   acc32" - which is what lets this epilogue stay a single shared copy.

   EVERY LINE OF ASSOCIATION ORDER HERE IS LOAD-BEARING; the order is
   what makes two builds agree bit for bit:
     - `acc * (sx * sw)`, never `(acc * sx) * sw` - one last-bit apart;
     - the r>1 path accumulates within a weight group in s order and
       multiplies by ws ONCE, with no cross-group 4-way accumulator;
     - the r==1 path reduces ((a0+a2)+(a1+a3)), matching the SSE2
       horizontal fold, not the natural ((a0+a1)+(a2+a3)).
   A 1e-6 difference here is not cosmetic: SSM state requantization
   amplifies it token by token, and eight tokens are enough to fork the
   generated text - and note that |logit diff|
   saturates and cannot rank such changes.

   `post` cancels the kernel's own scaling: 1/256 for Q8_0 (part32_x16
   and the MMX/asm kernels all scale their products by 256) and 1.0 for
   Q16_0, whose kernels do not. A power of two either way, so the cancel
   is exact, and `x * 1.0f` is bit-identical to `x` - which is what lets
   Q16_0 share this instead of carrying a second copy differing in one
   multiply. Both call sites pass a literal, so it folds away. */
/* The fixed counterpart. Same signature plus the tensor's int16 plane
   and the row, because the shared exponent is per row.

   xq16/mprod are static rather than stack (rule 6 item 4) and sized by
   the same bound lz_epi_prep enforces, so a shape it refused cannot
   reach here. */
static int16_t g_epi_xq[LZ_BATCH_MAX][LZ_EPI_MAX_NG];
static int     g_epi_xe[LZ_BATCH_MAX];
static int32_t g_epi_m[LZ_EPI_MAX_NG];

/* All tokens packed up front, because the row loop is OUTER and the
   token loop inner - packing inside would run once per (row, token) and
   cost exactly what the epilogue it replaces costs. That mistake is
   what made the first measured candidate 1.88x instead of 32x. */
void epi_pack_act(const float *xqs, int nb, int nt) {
    int tk, g;
    /* One multiply per group plus the amax scan, per token. Counted so
       the census cannot report the fixed tier as free - the work moved,
       it did not vanish, and a site that bills nothing is how the old
       hand-made table lost the two heaviest ones. The amax scan and the
       packing loop's round-direction compare are both comparisons, not
       billed (same convention as the four scans B1 leaves alone). */
    LZ_FCX(LZ_FC_EPI, (long)nt * nb, 0, 0, 0, 0);
    for (tk = 0; tk < nt; tk++) {
        const float *xsb = xqs + (size_t)tk * nb;
        float amax = 0.0f;
        int e = 0;
        for (g = 0; g < nb; g++) {
            float a = xsb[g] < 0.0f ? -xsb[g] : xsb[g];
            if (a > amax) amax = a;
        }
        if (amax <= 0.0f) {
            for (g = 0; g < nb; g++) g_epi_xq[tk][g] = 0;
            g_epi_xe[tk] = 0;
            continue;
        }
        while (e < 100 && amax * pow2f(e + 1) < 32767.0f) e++;
        for (g = 0; g < nb; g++) {
            float t = xsb[g] * pow2f(e);
            int q = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
            if (q >  32767) q =  32767;
            if (q < -32767) q = -32767;
            g_epi_xq[tk][g] = (int16_t)q;
        }
        g_epi_xe[tk] = e;
    }
}

float epi_fixed(const int32_t *accb, const LZTensor *w, int row,
                       int tk, int ng, int r, float post) {
    const int16_t *ww = w->sq + (size_t)row * (ng / r);
    long long s;
    int g, e;
    for (g = 0; g < ng; g++)
        g_epi_m[g] = (int32_t)g_epi_xq[tk][g] * (int32_t)ww[g / r];
    s = lz_epi_mac(accb, g_epi_m, ng);
    /* The exponent is per ROW, so a libm call here would run once per
       output element and cost more than the epilogue it replaces. */
    e = -g_epi_xe[tk] - (int)w->sexp[row];      /* int arithmetic, no f32 work */
    LZ_FCX(LZ_FC_EPI, 2, 0, 0, 1, 0);          /* convert, two multiplies */
    return (float)s * pow2f(e) * post;
}

float epi_q8(const int32_t *accb, const float *xsb, const float *ws,
                    int ng, int r, float post) {
    int g;
    /* r>1: per group a convert, a multiply and an add, then per r-run one
       multiply and one add, then the post scale. r==1: convert plus two
       multiplies plus an add per group, and 4 to fold the accumulators.
       This is the row the old census put at 1,995,552 by a rule nobody
       could reconstruct. The convert (lz_i32f, once per group either
       way) is the same ng in both branches. */
    LZ_FCX(LZ_FC_EPI,
           r > 1 ? ng + (ng / r) + 1 : 2 * ng + 1,     /* mul */
           r > 1 ? ng + (ng / r)     : ng + 3,          /* add */
           0,                                            /* div */
           ng,                                           /* cvt */
           0);
    if (r > 1) {
        float sum = 0.0f;
        int g128;
        for (g128 = 0; g128 < ng / r; g128++) {
            float dot = 0.0f;
            for (g = 0; g < r; g++) {
                int idx = g128 * r + g;
                dot += lz_i32f(accb[idx]) * xsb[idx];
            }
            sum += dot * ws[g128];
        }
        return sum * post;
    } else {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (g = 0; g + 3 < ng; g += 4) {
            a0 += lz_i32f(accb[g])     * (xsb[g]     * ws[g]);
            a1 += lz_i32f(accb[g + 1]) * (xsb[g + 1] * ws[g + 1]);
            a2 += lz_i32f(accb[g + 2]) * (xsb[g + 2] * ws[g + 2]);
            a3 += lz_i32f(accb[g + 3]) * (xsb[g + 3] * ws[g + 3]);
        }
        for (; g < ng; g++)
            a0 += lz_i32f(accb[g]) * (xsb[g] * ws[g]);
        return ((a0 + a2) + (a1 + a3)) * post;
    }
}

/* ---- Q8_0 row kernels, one per (ISA x impl) ---------------------------
   Each lives next to its table entry under ONE guard, so a definition
   and its use cannot drift apart - the drift that caused link breaks. */

/* Q8 guard block moved to ops_matmul.c. */

/* LZ_ROW_Q8 + matmul_q8_impl moved to ops_matmul.c. */

/* Get the f32 view. In ops.c rather than model.c: this is a FORMAT
   matter, sharing the same layout knowledge as the matmul dispatch
   below - adding a format only touches this one file. */

/* Q6_1 single-element read: 4-bit plane b4 (16 bytes) + 2-bit plane
   b2 (8 bytes), both pointing at the start of the SAME 32-element
   sub-block. k in [0,32), returns [0,63].

   The two planes' groupings are deliberately aligned: on the 4-bit
   side k<16 takes the low nibble, k>=16 the high nibble; on the 2-bit
   side byte k&7 holds the (k>>3)-th bit pair. On the SIMD side, the
   four groups from `pand 0x03` + `psrlw 2/4/6` are exactly
   {0-7}{8-15}{16-23}{24-31}, corresponding group-for-group with the
   4-bit plane's two 8-byte loads; pre-expanded activation registers
   are reused as-is. */
static int lz_q61_get(const unsigned char *b4, const unsigned char *b2, int k) {
    int lo = (k < 16) ? (b4[k] & 15) : (b4[k - 16] >> 4);
    int hi = (b2[k & 7] >> (2 * (k >> 3))) & 3;
    return lo + (hi << 4);
}

float *lz_t_f32(const LZTensor *t, float *scratch) {
    int g, k;
    if (!t) return NULL;
    if (t->dtype == LZ_FMT_F32) return t->f;
    if (!scratch || !t->q || !t->scale) return NULL;
    if (t->dtype == LZ_FMT_Q4_1) {
        /* nibbles: each 32-element sub-block is 16 bytes; byte j's low
           nibble = element j, high nibble = j+16 */
        const unsigned char *p = (const unsigned char *)t->q;
        if (!t->zero) return NULL;
        for (g = 0; g < t->n / t->gs; g++) {
            float d = t->scale[g], m = t->zero[g];
            int base;
            for (base = g * t->gs; base < (g + 1) * t->gs; base += 32) {
                const unsigned char *b = p + base / 2;
                for (k = 0; k < 16; k++) {
                    scratch[base + k]      = (float)(b[k] & 15) * d + m;
                    scratch[base + k + 16] = (float)(b[k] >> 4) * d + m;
                }
            }
        }
        return scratch;
    }
    if (t->dtype == LZ_FMT_Q6_1) {
        const unsigned char *p4 = (const unsigned char *)t->q;
        const unsigned char *p2 = p4 + (size_t)t->n / 2;
        if (!t->zero) return NULL;
        for (g = 0; g < t->n / t->gs; g++) {
            float d = t->scale[g], m = t->zero[g];
            int base;
            for (base = g * t->gs; base < (g + 1) * t->gs; base += 32) {
                const unsigned char *b4 = p4 + base / 2;
                const unsigned char *b2 = p2 + base / 4;
                for (k = 0; k < 32; k++)
                    scratch[base + k] =
                        (float)(lz_q61_get(b4, b2, k)) * d + m;
            }
        }
        return scratch;
    }
    if (t->dtype == LZ_FMT_T2) {
        /* One 2-bit plane, Q6_1's addressing. code-1 in {-1,0,+1}; no
           min term, so a code of 1 dequantizes to exactly 0.0f for
           every group regardless of scale - which is the property the
           format exists for. */
        const unsigned char *p2 = (const unsigned char *)t->q;
        for (g = 0; g < t->n / t->gs; g++) {
            float d = t->scale[g];
            int base;
            for (base = g * t->gs; base < (g + 1) * t->gs; base += 32) {
                const unsigned char *b2 = p2 + base / 4;
                for (k = 0; k < 32; k++)
                    scratch[base + k] =
                        (float)(((b2[k & 7] >> (2 * (k >> 3))) & 3) - 1) * d;
            }
        }
        return scratch;
    }
    if (t->dtype == LZ_FMT_Q16_0) {
        const int16_t *p = (const int16_t *)(const void *)t->q;
        for (g = 0; g < t->n / t->gs; g++) {
            float s = t->scale[g];
            for (k = 0; k < t->gs; k++)
                scratch[g * t->gs + k] = (float)p[g * t->gs + k] * s;
        }
        return scratch;
    }
    for (g = 0; g < t->n / t->gs; g++) {
        float s = t->scale[g];
        const int8_t *q = t->q + g * t->gs;
        float *d = scratch + g * t->gs;
        for (k = 0; k < t->gs; k++) d[k] = q[k] * s;
    }
    return scratch;
}

void lz_t_row_f32(const LZTensor *t, int row, int dim, float *out) {
    size_t base;
    int g0, i;
    if (!t || !out || dim <= 0) return;
    if (t->dtype == LZ_FMT_F32) {
        if (t->f) memcpy(out, t->f + (size_t)row * dim, (size_t)dim * sizeof(float));
        return;
    }
    if (!t->q || !t->scale || t->gs <= 0) return;
    base = (size_t)row * dim;
    g0 = (int)(base / (size_t)t->gs);
    if (t->dtype == LZ_FMT_Q4_1) {
        const unsigned char *p = (const unsigned char *)t->q + base / 2;
        if (!t->zero) return;
        for (i = 0; i < dim; i += 32) {
            int g = g0 + i / t->gs;
            float d = t->scale[g], m = t->zero[g];
            const unsigned char *b = p + i / 2;
            int k;
            for (k = 0; k < 16; k++) {
                out[i + k]      = (float)(b[k] & 15) * d + m;
                out[i + k + 16] = (float)(b[k] >> 4) * d + m;
            }
        }
        return;
    }
    if (t->dtype == LZ_FMT_Q6_1) {
        const unsigned char *p4 = (const unsigned char *)t->q + base / 2;
        const unsigned char *p2 = (const unsigned char *)t->q +
                                  (size_t)t->n / 2 + base / 4;
        if (!t->zero) return;
        for (i = 0; i < dim; i += 32) {
            int g = g0 + i / t->gs;
            float d = t->scale[g], m = t->zero[g];
            const unsigned char *b4 = p4 + i / 2;
            const unsigned char *b2 = p2 + i / 4;
            int k;
            for (k = 0; k < 32; k++)
                out[i + k] = (float)lz_q61_get(b4, b2, k) * d + m;
        }
        return;
    }
    if (t->dtype == LZ_FMT_T2) {
        const unsigned char *p2 = (const unsigned char *)t->q + base / 4;
        for (i = 0; i < dim; i += 32) {
            int g = g0 + i / t->gs, k;
            float d = t->scale[g];
            const unsigned char *b2 = p2 + i / 4;
            for (k = 0; k < 32; k++)
                out[i + k] =
                    (float)(((b2[k & 7] >> (2 * (k >> 3))) & 3) - 1) * d;
        }
        return;
    }
    if (t->dtype == LZ_FMT_Q16_0) {
        const int16_t *p = (const int16_t *)(const void *)t->q + base;
        for (i = 0; i < dim; i++)
            out[i] = (float)p[i] * t->scale[g0 + i / t->gs];
        return;
    }
    {
        const int8_t *q = t->q + base;
        for (i = 0; i < dim; i++)
            out[i] = (float)q[i] * t->scale[g0 + i / t->gs];
    }
}

int lz_act_gs(const LZTensor *w, int in_dim) {
    if (!w || w->dtype == LZ_FMT_F32) return 0;
    if (w->gs >= 32 && (w->gs % 32) == 0 && (in_dim % 32) == 0) return 32;
    return w->gs;                   /* degenerate exported tier: keep old "activation follows weight" */
}

/* Scalar reference path: every format must be CORRECT here.

   This is not a fallback; it is the CONTRACT. SIMD kernels are written
   per (format x instruction set); any missing cell falls here
   automatically. Absence is the norm, not the exception: Watcom's
   MMX intrinsics measure 6x slower than its own x87 scalar
   (21.7 ms vs 3.40 ms); only hand-written #pragma aux asm works
   (0.82 ms). I.e. **each new format means hand-writing another asm
   file**, so formats should be few, and a new format must first run on
   this path, measure its quality, then decide whether an asm is worth
   it.

   Semantics (covering both Q8_0 and Q4_1; weight groups gs are a
   multiple of 32, activation groups always 32):

     Q8_0: o[i] = Σ_g  ws[g]·Σ_{sub-block s∈g} xqs[s]·dot32(w, x)
     Q4_1: o[i] = Σ_g ( d[g]·Σ_s xqs[s]·dot32(q, x) + m[g]·Σ_s xqs[s]·Σx )

   **The reference must reproduce the SIMD reduction ORDER bit for
   bit, not merely be "mathematically equivalent".** The contract is
   not the two summation formulas but HOW THE PARENTHESES GO - because
   SSM state is quantized, a 1e-6 association difference flips some
   ±1 LSB quantization decision and then amplifies per token - eight
   tokens fork generation. So this code is deliberately
   dumb: int32 sub-block sums first go into arrays (on the SIMD side
   MMX/SSE2 does this), then the exact same float reduction structure
   finishes. Three details must not change:

     - `acc·(sx·sw)` not `(acc·sx)·sw` (gs=32 tier)
     - 4-way accumulators paired as `(a0+a2)+(a1+a3)`, tail ALL into a0 (gs=32 tier)
     - Q4_1's dot and zero each get their own accumulator, merged once at the end (not interleaved per group)

   The x256 fold exists on the SIMD side but not here: 256 is a power
   of two, scaling changes no rounding at any step, so the two sides
   remain bit-identical. */
/* matmul_scalar_ref_one + matmul_scalar_ref moved to ops_matmul.c. */

/* Q4_1 matmul kernel.

   `o[i] = Σ_g [ d[i][g]·Σ_{sub-block s∈g} xqs[s]·dot32(q_w, q_x) + m[i][g]·xg[g] ]`

   Two savings:
   1. **Zero term hoisted out of the row loop**. `xg[g] = Σ_{s∈g}
      xqs[s]·Σ_k xq[k]` depends only on activations - computed once per
      matmul, shared by all output rows. So Q4_1's inner loop differs
      from Q4_0's by not a single instruction, while the asymmetric
      tier's quality measures 2.7-4.4x better.
   2. **Nibble expansion needs no sign extension** (unsigned 0..15),
      keeping the x256 fold from `punpcklbw(0,·)`, cancelled once at
      the row end.

   Reduction order deliberately matches the Q8 wide-group path
   (per-sub-block sequential accumulation; SSE2's batched 4-vector
   reduction `((q0+q1)+q2)+q3` is equivalent), so the gcc/SSE2 and MMX
   builds are bit-identical. */
/* THE Q4_1 / Q6_1 FLOAT EPILOGUE - one copy, shared by every tier.
   Same shape as epi_q8 plus the zero-point term: accb[sb] is the
   sub-block's integer sum scaled by 256, xgb[g] the activation-only
   zero term for weight group g (computed once per token, reused across
   output rows), wd/wm the group's scale and min.

   The association order is again load-bearing, again what makes the MMX
   and SSE2 paths agree bit for bit: dot accumulates
   in s order within a weight group, is multiplied by wd ONCE, and the
   256 cancel happens on dotsum, AFTER which zsum is added. The SSE2
   fold has the same shape - a 4-lane multiply and a
   ((q0+q1)+q2)+q3 fold; term for term that is this loop. */
/* `wm` is the per-group ZERO coefficient, and T2 shares this epilogue by
   passing a negated scale rather than a stored min - ternary carries no
   min, and Sum w*x = d*[Sum code*x - Sum x] makes the coefficient -d.
   One function rather than a near-copy on purpose: the reduction ORDER
   is what the bit-identity contract is about, and two copies of it drift
   the moment one is touched. Callers hand in whichever array is theirs;
   matmul_t2_impl builds the negated one once per row. */
float epi_q41(const int32_t *accb, const float *xsb, const float *xgb,
                     const float *wd, const float *wm, int ng, int r) {
    float dotsum = 0.0f, zsum = 0.0f;
    int g, s;
    for (g = 0; g < ng; g++) {
        float dot = 0.0f;
        for (s = 0; s < r; s++) {
            int sb = g * r + s;
            dot += lz_i32f(accb[sb]) * xsb[sb];
        }
        dotsum += dot * wd[g];
        zsum += xgb[g] * wm[g];
    }
    return dotsum * (1.0f / 256.0f) + zsum;
}

/* ---- Q4_1 row kernels, one per (ISA x impl) --------------------------- */

/* Q41 guard block moved to ops_matmul.c. */

/* LZ_ROW_Q41 + matmul_q41_impl moved to ops_matmul.c. */

/* ---- T2 row kernels, one per (ISA x impl) -----------------------------
   The single 2-bit plane arrives in `w4`; `w2v` and both prefetch bounds
   are unused - 8 bytes per 32 elements is a quarter of Q4_1's weight
   traffic, too few cache lines for a prefetch distance to mean anything
   unmeasured.

   No pairing yet. Not a decision - unmeasured. `--pair` does nothing for
   this format today. */

/* T2 guard block moved to ops_matmul.c. */

/* row_t2_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

/* row_t2_mmx_asm / row_t2_sse2_asm moved to src/ops_mmx.c /
   src/ops_sse2.c (task #30, 7/N), alongside their leaf kernels
   (lz_dot32_t2_asm / lz_dot32_t2_sse2_asm) which they call
   direct-by-name, same-TU. Declared in src/ops_mmx.h / src/ops_sse2.h. */

/* SSE1 slots NULL: T2's unpack is `pand 0x03` + three `psrlw`, and SSE1
   adds no integer instruction that replaces any of them. */
/* LZ_ROW_T2 moved to ops_matmul.c. */

/* ---- Q6_1 row kernels, one per (ISA x impl) ---------------------------
   The only format that uses BOTH plane pointers and BOTH prefetch
   bounds. The 2-bit plane needs its own: `pf_end` stops exactly at the
   plane boundary. At 22% of all weight bytes on the recipe of record,
   its miss frequency is half the 4-bit plane's, not 1/16 like a scale
   array: it IS a hot stream, just a narrower one. Both bounds are real,
   not belt-and-braces: LZ_PF_LOAD is a REAL dummy load on PII and
   faults past the end of the tensor. */

/* Q61 guard block moved to ops_matmul.c. */

/* row_q61_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

/* LZ_ROW_Q61 moved to ops_matmul.c. */

/* T2 matmul. Same shape as matmul_q61_impl minus the second plane, with
   one addition: the zero coefficient is -scale, so the negated array is
   built per row into g_t2neg and handed to epi_q41 as its `wm`. */
/* matmul_t2_impl + matmul_q61_impl moved to ops_matmul.c. */

/* Q16 block moved to ops_matmul.c. */

/* Matmul consuming already-quantized activations: the forward hot path.
   xq/xqs quantized by the caller at lz_act_gs()'s group (one activation
   may be reused by several matmuls). f32 weights fall back to f32 -
   one call site serves all weight representations. */
void lz_matmul_xq(float *o, const float *x, const int8_t *xq,
                  const float *xqs, const LZTensor *w,
                  int in_dim, int out_dim) {
    lz_matmul_xq_nt(o, x, xq, xqs, w, in_dim, out_dim, 1);
}

void lz_matmul_xq_nt(float *o, const float *x, const int8_t *xq,
                     const float *xqs, const LZTensor *w,
                     int in_dim, int out_dim, int nt) {
    int i;

    /* An out-of-range nt would read past g_xw/g_acc32. The caller
       (lz_forward_batch) already chunks by LZ_BATCH_MAX; reaching here
       is a bug - clamp, never silently overrun. */
    if (nt < 1) return;
    if (nt > LZ_BATCH_MAX) nt = LZ_BATCH_MAX;
    if (!w || w->dtype == LZ_FMT_F32) {
        for (i = 0; i < nt; i++)
            lz_matmul(o + (size_t)i * out_dim, x + (size_t)i * in_dim,
                      w ? w->f : NULL, in_dim, out_dim);
        return;
    }
    if (!xq || !xqs || w->gs <= 0 || (in_dim % w->gs) != 0) {
        for (i = 0; i < nt * out_dim; i++) o[i] = 0.0f;  /* defense: no silent miscalc */
        return;
    }
    /* The widen buffers are LZ_BATCH_MAX × LZ_MM_WIDEN_MAX; nt tokens'
       activations always fit: the g_xw-filling path carries its own
       `in_dim <= LZ_MM_WIDEN_MAX` check (Q8_0's !widened branch does
       not widen and goes per-token scalar), and Q4_1/Q6_1 dispatch
       already requires in_dim <= LZ_MM_WIDEN_MAX. */
    /* LZ_KERNEL_REF must genuinely reach the scalar reference, not just
       affect the name reported. Dispatch is by dtype, so without this
       check `--kernel ref` would actually run SIMD, unnoticed because
       all kernels are bit-identical. The scalar reference is the
       CONTRACT for new formats (rule 2); an oracle that cannot be
       selected is no oracle. */
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* The SIMD fast path currently covers only (Q8_0, gs=32). Other
       formats/groups go through the scalar reference - correct but
       slow.

       This is not a question of "whether asm is worth it" decided by
       measurement. The rule requires every operator to have C /
       MMX / SSE / SSE2 paths, because the bottleneck is a seesaw -
       relieve the bytes and it moves to the ALU, relieve the ALU and it
       moves back, so "measure which side binds today, then optimise
       only that side" chases a moving target and leaves both sides
       undone. Measurement says how fast and whether it is bit-identical,
       not whether to write the kernel. Every remaining
       scalar-reference fallthrough below is therefore a TODO with a
       name, not a decision. */
    /* Both SIMD paths pre-expand activations to int16; beyond the
       buffer cap they fall back to the scalar reference. Q8 has no
       non-expanding MMX intrinsics fallback - it would be 6x slower
       than x87 scalar on Watcom; keeping it would only make a real
       trigger worse. */
    if (w->dtype == LZ_FMT_Q8_0 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0) {
        matmul_q8_impl(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* Q4_1's SIMD path pre-expands activations to int16; beyond the
       buffer cap it falls back to the scalar reference. s1v3's max
       in_dim is 1024, the 0.8B's 3584 - both within 4096. */
    if (w->dtype == LZ_FMT_Q4_1 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && w->zero) {
        matmul_q41_impl(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    if (w->dtype == LZ_FMT_Q6_1 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && w->zero) {
        matmul_q61_impl(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* No `w->zero` in the guard: ternary carries no min, and a T2 tensor
       that HAS one was mis-loaded. */
    if (w->dtype == LZ_FMT_T2 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && !w->zero) {
        matmul_t2_impl(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)
    /* Q16_0 has no non-widened fallback: unlike Q8_0 it always needs the
       activation expanded to int16, so in_dim past the buffer cap goes to
       the scalar reference. The gcc/SSE2-only dev build has no Q16_0 SIMD
       kernel either and lands there too - that is deliberate, not an
       oversight. The differential oracle rule 2 asks for is a gcc build
       with LZ_MMX_TU set (src/ops_mmx.c linked in), which compiles the
       intrinsics twin and exercises this same path. */
    if (w->dtype == LZ_FMT_Q16_0 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX) {
        matmul_q16_impl(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */
    matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
}

/* o = W x: unified entry for weight tensors.
   dtype=0 goes f32; dtype=1 quantizes x (Q8_0, group = w->gs) into the
   caller-provided xq/xqs buffers, int8×int8→int32 accumulation,
   dequantized per group by (sx * sw). xq needs in_dim bytes, xqs
   in_dim/w->gs f32s. */
void lz_matmul_w(float *o, const float *x, const LZTensor *w,
                 int in_dim, int out_dim, int8_t *xq, float *xqs) {
    int i;

    if (!w || w->dtype == LZ_FMT_F32) {
        lz_matmul(o, x, w ? w->f : NULL, in_dim, out_dim);
        return;
    }
    if (w->gs <= 0 || (in_dim % w->gs) != 0 || !xq || !xqs) {
        /* Defense: normal export guarantees in_dim % gs == 0 (in-row
           grouping). Triggering means a bug - zero the output and
           return; no silent miscalc, no crash. */
        for (i = 0; i < out_dim; i++) o[i] = 0.0f;
        return;
    }
    /* Activation groups follow lz_act_gs, not w->gs - weights may be
       coarse to gs=128 while activations stay 32. */
    lz_quantize_q8(x, in_dim, lz_act_gs(w, in_dim), xq, xqs);
    lz_matmul_xq(o, x, xq, xqs, w, in_dim, out_dim);
}

void lz_matmul(float *o, const float *x, const float *w,
                   int in_dim, int out_dim) {    int i;
    /* The hottest loop in the whole forward, ~750M MACs per token.

       8 independent accumulators here are not for loop unrolling but
       to break the dependency chain: a single accumulator makes every
       iteration wait for the previous float add to retire (~4 cycles
       latency on Zen 3), pinning throughput at 2 flops/cycle.
       Measured: single accumulator 9.1 GFLOP/s, eight accumulators
       19.6 GFLOP/s, while L1-vs-memory for weights makes almost no
       difference - the bottleneck is the latency chain, not bandwidth.

       Note the summation order therefore changes, giving ~1e-7-level
       float differences. Differential tests use tolerances at the
       same order as the reference implementation; this change does not
       break them. */
    for (i = 0; i < out_dim; i++) {
        const float *row = w + (size_t)i * in_dim;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        float sum;
        int j = 0;
        for (; j + 7 < in_dim; j += 8) {
            a0 += row[j]     * x[j];
            a1 += row[j + 1] * x[j + 1];
            a2 += row[j + 2] * x[j + 2];
            a3 += row[j + 3] * x[j + 3];
            a4 += row[j + 4] * x[j + 4];
            a5 += row[j + 5] * x[j + 5];
            a6 += row[j + 6] * x[j + 6];
            a7 += row[j + 7] * x[j + 7];
        }
        /* pair-wise merge rather than sequential adds, reducing rounding error in these last steps */
        sum = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
        for (; j < in_dim; j++) sum += row[j] * x[j];
        o[i] = sum;
    }
}

/* lz_rope_sse2's body lives in src/ops_sse2.c: it writes %xmm
   registers, so it belongs in the translation unit built with -msse2.
   Declared in src/ops_sse2.h, guarded there by LZ_ROPE_SSE2_EXTERN. */



/* Lloyd-Max fixed point for N(0,1), 16 levels. Provenance and the
   re-derivation script are in ops.h's comment - do not "tidy" these
   digits, every cached row decodes against them. */
const float lz_kv4_cents[16] = {
      -2.732589571f,   -2.069017227f,   -1.618046386f,   -1.256231197f,
      -0.942340456f,   -0.656759119f,   -0.388048299f,   -0.128395030f,
       0.128395030f,    0.388048299f,    0.656759119f,    0.942340456f,
       1.256231197f,    1.618046386f,    2.069017227f,    2.732589571f
};

/* Decision boundaries: midpoints of adjacent centroids. Derived here
   rather than shipped so the two can never drift apart. */
static const float kv4_mid[15] = {
      -2.400803399f,   -1.843531806f,   -1.437138792f,   -1.099285826f,
      -0.799549788f,   -0.522403709f,   -0.258221664f,    0.000000000f,
       0.258221664f,    0.522403709f,    0.799549788f,    1.099285826f,
       1.437138792f,    1.843531806f,    2.400803399f
};

void lz_kv4_quantize(const float *x, int n, unsigned char *out, float *scale) {
    /* Not exactly representable in float32 - see LZ_EXP_LOG2E32's
       comment in lz_exp for why an inline literal here would risk
       gcc's x87 excess-precision folding (fldt). */
    static const float LZ_KV4_EPS = 1e-20f;
    float n2 = 0.0f, s, inv;
    int i;

    for (i = 0; i < n; i++) n2 += x[i] * x[i];
    n2 = (float)sqrt((double)n2);
    /* sqrt(n) is exact for the power-of-two head dims this path serves;
       the divide has a runtime divisor, so both compilers emit a real
       FDIV rather than one strength-reducing it. */
    s = n2 / (float)sqrt((double)n);
    if (scale) *scale = s;
    inv = (s > LZ_KV4_EPS) ? (1.0f / s) : 0.0f;

    for (i = 0; i < n; i += 2) {
        float z0 = x[i] * inv, z1 = x[i + 1] * inv;
        int c0 = 0, c1 = 0, k;
        /* Sorted boundaries, so a walk stops at the first one above z.
           15 compares worst case and no absolute value - cheaper than
           scanning the centroids themselves. */
        for (k = 0; k < 15; k++) { if (z0 < kv4_mid[k]) break; c0++; }
        for (k = 0; k < 15; k++) { if (z1 < kv4_mid[k]) break; c1++; }
        out[i / 2] = (unsigned char)(c0 | (c1 << 4));
    }
}

void lz_fwht(float *v, int n) {
    /* Sylvester recursion, in place. Every stage is a butterfly of one
       add and one sub, so the whole transform is exact in binary float
       up to the additions themselves - no constants enter, which is why
       this is bit-identical between wcc386 and gcc without any of the
       care the multiply-by-reciprocal paths demand. */
    int len, i, j;
    for (len = 1; len < n; len <<= 1) {
        for (i = 0; i < n; i += (len << 1)) {
            for (j = 0; j < len; j++) {
                float a = v[i + j];
                float b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
}

void lz_rope(float *v, int n_heads, int head_dim, int rotary_dim,
             int pos, const float *cs) {
    /* Bit-exact contract: every kernel below reproduces the scalar path
       per-element (each rotation is an independent IEEE mul/add). MMX is
       deliberately NOT used here - MMX has no floating point, so any MMX
       RoPE would have to quantize (x Q8 / cos-sin Q14), which breaks
       bit-exactness; RoPE is not a PII bottleneck (~2-3us/token), so
       correctness wins and MMX builds fall through to scalar. */
    /* LZ_ROPE_SSE2_EXTERN is defined (in src/ops_sse2.h) exactly when
       src/ops_sse2.c's lz_rope_sse2 is part of this link - the single
       source of truth for both this call site and the extern
       declaration, so the two cannot drift the way two independent
       `#if defined(__SSE2__) && !defined(__WATCOMC__)` copies once
       could (one compiling the definition out while the call stayed,
       failing the build with an implicit-declaration error). */
#if defined(LZ_ROPE_SSE2_EXTERN)
    lz_rope_sse2(v, n_heads, head_dim, rotary_dim, pos, cs);
#else
    int half = rotary_dim / 2;
    int h, i;
    const float *row = cs + (size_t)pos * half * 2;
    for (i = 0; i < half; i++) {
        float c = row[i * 2];
        float s = row[i * 2 + 1];
        for (h = 0; h < n_heads; h++) {
            float *base = v + (size_t)h * head_dim;
            float x0 = base[i];
            float x1 = base[i + half];
            base[i]        = x0 * c - x1 * s;
            base[i + half] = x1 * c + x0 * s;
        }
    }
    /* components beyond rotary_dim stay as-is; they do not rotate */
#endif /* LZ_ROPE_SSE2_EXTERN */
}

/* ---- fixed-point attention -------------------------------------------

   Two independent pieces, both replacing f32 x int8 inner loops in
   forward_attn():

   1. Scoring. q is quantized to Q8 once per head and widened to int16
      once, so q.k becomes int8(k) x int16(q) per 32-group - exactly the
      shape the production matmul already uses. No new kernel: it calls
      the same dot32_x16 matmul_q8_impl does, already validated. The x256
      the int8->int16 unpack introduces is cancelled once at the end,
      same as there.

   2. Weighted sum. out[d] = sum_t c_t[g] * vq_t[d], with
      c_t[g] = att[t] * vs[t][g], is isomorphic to the SSM recurrence's
      pass 1 (t plays kk, d plays vv), so it DOES need a new kernel.

   The weighted sum is the only place in the engine whose accumulator
   bound depends on context length: one term is at most 32767*127 =
   4,161,409, so T rows fit int32 only while T <= 516. Accumulation is
   chunked at 512 rows (512 * 4,161,409 = 2.13e9, ~0.8% under INT32_MAX)
   and each chunk goes through lz_i32f() before joining the running float
   total. The chunking is load-bearing, not decorative - test_overflow()
   drives a worst-case input through an unchunked variant and asserts
   the result IS corrupted.

   The default lives in ops.h (LZ_ATTN_FIXED), not here - a second
   #ifndef with a different default would be a lie. The error-magnitude
   argument that would normally settle this is NOT usable here: the
   logit-difference metric saturates, so "this error is 200x that one"
   says nothing about which output is better (section 6.1). */
#if LZ_ATTN_FIXED
/* Largest head_dim served by this path. 256 covers the 0.8B; anything
   larger falls back to the float loops rather than truncating. */
#define LZ_ATTN_MAX_HD 256
/* LZ_WSUM_CHUNK, LZ_SLOT_NEXT: src/ops_kernel_shared.h (task #31) -
   src/ops_mmx.c's lz_wsum_group_mmx walks the exact same ring buffer
   in the exact same chunks, and a constant duplicated across that TU
   boundary is the failure mode task #31 exists to close off. Outside
   both `& 1`/`& 2` bit guards because BOTH kernels need LZ_SLOT_NEXT -
   it sat under `& 1` first and the -DLZ_ATTN_FIXED=2 build would not
   compile. */

#if (LZ_ATTN_FIXED & 2)
#if defined(__WATCOMC__)
/* lz_wsum_pair_asm's #pragma aux body, and the per-pair loop that used
   to call it, both moved to src/ops_mmx.c's lz_wsum_group_mmx in task
   #30 (3/N) - see that function and ops.c's lz_attn_wsum_q8 (search
   "one call per GROUP") for why: T is context-length-dependent and
   unbounded, so a per-pair or even per-chunk call here would scale the
   wrong way. No LZ_WSUM_PAIR macro here anymore: it would alias a name
   no longer declared in this TU, and nothing under __WATCOMC__ needs
   it - lz_attn_wsum_q8's Watcom branch calls lz_wsum_group_mmx
   directly, the same way task #30 (2/N) retired the p2 aliases. */

#elif defined(LZ_WSUM_MMX_EXTERN)
/* lz_wsum_pair_mmx's body lives in src/ops_mmx.c now - it writes %mm
   registers. Declared in src/ops_mmx.h, already visible here. */
#define LZ_WSUM_PAIR(ra, rb, co, ac) lz_wsum_pair_mmx(ra, rb, co, ac)

#else
/* Non-MMX builds. Integer sums are exact under reassociation, so this is
   bit-identical to both kernels above - the same reason the SSM pass 1
   has a scalar twin. Without it, enabling this tier would make the output
   depend on which compiler built the binary, and rule 2's byte-compare
   gate would stop meaning anything. */
static void lz_wsum_pair_ref(const int8_t *rowA, const int8_t *rowB,
                             const int16_t *coef, int32_t *acc32) {
    int32_t ckA = coef[0], ckB = coef[1];
    int d;
    for (d = 0; d < 32; d++)
        acc32[d] += (int32_t)rowA[d] * ckA + (int32_t)rowB[d] * ckB;
}
#define LZ_WSUM_PAIR(ra, rb, co, ac) lz_wsum_pair_ref(ra, rb, co, ac)
#endif /* __WATCOMC__ */
#endif /* LZ_ATTN_FIXED & 2 */

#if (LZ_ATTN_FIXED & 1)
#if !defined(__WATCOMC__) && !defined(LZ_DOT_MMX_EXTERN)
/* Same story for the scoring dot. Mirrors dot32_x16_mmx INCLUDING its
   x256 product scaling, so the caller's single 1/256 is correct either
   way. */
static int32_t dot32_x16_attn_ref(const int8_t *w, const int16_t *x) {
    int32_t acc = 0;
    int k;
    for (k = 0; k < 32; k++) acc += (int32_t)w[k] * 256 * (int32_t)x[k];
    return acc;
}
#define LZ_ATTN_DOT32(w, x) dot32_x16_attn_ref(w, x)
#else
#define LZ_ATTN_DOT32(w, x) dot32_x16_mmx(w, x)
#endif /* !__WATCOMC__ && !LZ_DOT_MMX_EXTERN */

void lz_attn_score_q8(float *att, const float *qhh, int hd,
                      const int8_t *kc, const float *ks, int kvd,
                      int pos, float scale, int sink, int ring) {
    /* static, not stack (rule 6 item 4) */
    static int8_t  qq[LZ_ATTN_MAX_HD];
    static float   qs[LZ_ATTN_MAX_HD / 32];
    static int16_t qw[LZ_ATTN_MAX_HD];
    int ng = hd / 32, g, t, i, slot = 0;

    /* Once per head: the q quantization (a multiply per element, a
       reciprocal per group). Then per t: a convert, the qs*kts product
       and a multiply-add per group, plus 3 adds and 2 multiplies to
       fold. This is the term that makes the fixed tier LOSE below ~200
       tokens - it does not shrink with context, and the per-t term it
       replaces is 2*hd.
       Once-per-head: mul=hd (quantize), div=ng (reciprocal only - the
       amax*(1/127) mul q8_group_scale also does is not billed here).
       Per-t, per group: cvt=1, mul=2 (qs*kts, then the outer product),
       add=1 (accumulate) - times ng. Per-t fold: add=3, mul=2. */
    LZ_FCX(LZ_FC_ATTN_SCORE,
           hd + (long)(pos + 1) * (2 * ng + 2),
           (long)(pos + 1) * (ng + 3),
           ng,
           (long)(pos + 1) * ng,
           0);
    lz_quantize_q8(qhh, hd, 32, qq, qs);
    for (i = 0; i < hd; i++) qw[i] = (int16_t)qq[i];

    for (t = 0; t <= pos; t++, LZ_SLOT_NEXT(slot, sink, ring)) {
        const int8_t *kt = kc + (size_t)slot * kvd;
        const float *kts = ks + (size_t)slot * (kvd / 32);
        int32_t rr[LZ_ATTN_MAX_HD / 32];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

        /* Phase 1 is MMX only: collect every group's raw dot before any
           x87 runs. Interleaving the two leaves MMX registers dirty
           across an x87 fld/fmul - the bug rule 6 item 6 describes. It
           showed 0 mismatches in a short bit-identity test and -nan in a
           longer one, on the Watcom build only. */
        for (g = 0; g < ng; g++)
            rr[g] = LZ_ATTN_DOT32(kt + (size_t)g * 32, qw + g * 32);
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)
        if (lz_cpu_has_mmx()) _mm_empty();
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */
        /* Phase 2, x87 only. Raw magnitude reaches 256*127*127*32 =
           1.33e8 > 2^24, so lz_i32f() is mandatory here (rule 6 item 7),
           not a stylistic choice. */
        for (g = 0; g + 3 < ng; g += 4) {
            a0 += lz_i32f(rr[g + 0]) * (qs[g + 0] * kts[g + 0]);
            a1 += lz_i32f(rr[g + 1]) * (qs[g + 1] * kts[g + 1]);
            a2 += lz_i32f(rr[g + 2]) * (qs[g + 2] * kts[g + 2]);
            a3 += lz_i32f(rr[g + 3]) * (qs[g + 3] * kts[g + 3]);
        }
        for (; g < ng; g++)
            a0 += lz_i32f(rr[g]) * (qs[g] * kts[g]);
        att[t] = ((a0 + a2) + (a1 + a3)) * (1.0f / 256.0f) * scale;
    }
}
#endif /* LZ_ATTN_FIXED & 1 */

#if (LZ_ATTN_FIXED & 2)
/* cbuf/cq are caller-provided with seq_len entries each: the coefficient
   run is as long as the context, so unlike the per-head buffers above it
   cannot be a fixed static array. */
void lz_attn_wsum_q8(float *out, const float *att, int hd,
                     const int8_t *vc, const float *vs, int kvd,
                     int pos, float *cbuf, int16_t *cq, int sink, int ring) {
    int ng = hd / 32, g, T = pos + 1;
    /* Two independent walkers rather than one slot array: both passes
       below visit t in ascending order from 0, so each can carry its own
       running slot. An array would need a fixed cap, and a cap that a
       long context silently exceeds is the same class of defect this
       whole change exists to fix. */

    /* Per group: T to build the coefficients (mul, the cmax tracking
       compare is not billed), 1 divide plus 2T to quantize them (mul
       + q8_round add per t, common path), 64 per int32 chunk to drain
       the accumulator (32 convert + 32 add), and 33 to scale the 32
       outputs (all mul). The chunk term is why the count is not a
       clean multiple of T. */
    LZ_FCX(LZ_FC_ATTN_WSUM,
           (long)ng * (2L * T + 33L),
           (long)ng * ((long)T + 32L * ((T + LZ_WSUM_CHUNK - 1) / LZ_WSUM_CHUNK)),
           (long)ng,
           (long)ng * 32L * ((T + LZ_WSUM_CHUNK - 1) / LZ_WSUM_CHUNK),
           0);
    for (g = 0; g < ng; g++) {
        int32_t acc32[32];
        float accf[32];
        float cmax = 0.0f, inv;
        int t, d, t0;

        int slot = 0;
        for (t = 0; t < T; t++, LZ_SLOT_NEXT(slot, sink, ring)) {
            float c = att[t] * vs[(size_t)slot * (kvd / 32) + g];
            float ac = (c < 0.0f) ? -c : c;
            cbuf[t] = c;
            if (ac > cmax) cmax = ac;
        }
        if (cmax > 0.0f) {
            inv = 32767.0f / cmax;
            if (!(inv <= FLT_MAX)) {
                /* cmax subnormal: 32767/cmax overflows to inf and (int)inf
                   is undefined behavior. Same guard lz_quantize_q8 carries. */
                for (t = 0; t < T; t++) {
                    int qi = q8_round(cbuf[t] / (cmax * LZ_Q15_INV));
                    if (qi >  32767) qi =  32767;
                    if (qi < -32767) qi = -32767;
                    cq[t] = (int16_t)qi;
                }
            } else {
                for (t = 0; t < T; t++) {
                    int qi = q8_round(cbuf[t] * inv);
                    if (qi >  32767) qi =  32767;
                    if (qi < -32767) qi = -32767;
                    cq[t] = (int16_t)qi;
                }
            }
        } else {
            for (t = 0; t < T; t++) cq[t] = 0;
        }

#if defined(__WATCOMC__)
        /* Watcom (task #30, 3/N): one call per GROUP, not per pair. The
           #pragma aux body used to expand inline at LZ_WSUM_PAIR's call
           site, once per pair, with zero real call overhead; moving it
           to src/ops_mmx.c means the pair loop itself has to move too,
           or every pair would pay a cross-TU call - up to T/2 of them,
           T being context-length-dependent and unbounded (262144 on
           this model). lz_wsum_group_mmx does the ENTIRE multi-chunk
           second walk (all of LZ_WSUM_CHUNK's chunks, all pairs, the
           ring-buffer slot walk, the per-chunk emms and lz_i32f fold)
           for this one group, so ops.c's call count per group goes from
           zero to exactly one, not to T/2. */
        lz_wsum_group_mmx(vc, kvd, g, sink, ring, cq, T, accf);
#else
        for (d = 0; d < 32; d++) accf[d] = 0.0f;
        slot = 0;                       /* second walk, same t order */
        for (t0 = 0; t0 < T; t0 += LZ_WSUM_CHUNK) {
            int tn = T - t0;
            int tp;
            if (tn > LZ_WSUM_CHUNK) tn = LZ_WSUM_CHUNK;
            memset(acc32, 0, sizeof(acc32));
            for (tp = t0; tp + 1 < t0 + tn; tp += 2) {
                int16_t coef[4];
                const int8_t *rowA, *rowB;
                int slotA = slot;
                LZ_SLOT_NEXT(slot, sink, ring);
                rowA = vc + (size_t)slotA * kvd + (size_t)g * 32;
                rowB = vc + (size_t)slot  * kvd + (size_t)g * 32;
                LZ_SLOT_NEXT(slot, sink, ring);
                coef[0] = cq[tp];     coef[1] = cq[tp + 1];
                coef[2] = cq[tp];     coef[3] = cq[tp + 1];
                LZ_WSUM_PAIR(rowA, rowB, coef, acc32);
            }
            if (tp < t0 + tn) {            /* odd leftover row in this chunk */
                int16_t coef[4];
                const int8_t *rowA = vc + (size_t)slot * kvd + (size_t)g * 32;
                coef[0] = cq[tp]; coef[1] = 0;
                coef[2] = cq[tp]; coef[3] = 0;
                LZ_WSUM_PAIR(rowA, rowA, coef, acc32);
                LZ_SLOT_NEXT(slot, sink, ring);
            }
#if defined(LZ_WSUM_MMX_EXTERN)
            if (lz_cpu_has_mmx()) _mm_empty();      /* no x87 inside the row-pair loop above */
#endif /* LZ_WSUM_MMX_EXTERN */
            for (d = 0; d < 32; d++) accf[d] += lz_i32f(acc32[d]);
        }
#endif /* __WATCOMC__ */
        {
            float sscale = cmax * LZ_Q15_INV;
            for (d = 0; d < 32; d++) out[g * 32 + d] = accf[d] * sscale;
        }
    }
}
#endif /* LZ_ATTN_FIXED & 2 */
#endif /* LZ_ATTN_FIXED */

/* ---- SSM recurrence pass 1, fixed point --------------------------------

   Pass 1 is the two contractions u[vv] = sum_kk S[kk][vv]*ss[kk][gg]*k[kk]
   and the same with q. In float it is kd*vd multiply-adds over an int8
   state converted to float one element at a time - the state is ALREADY
   quantized, and the float math only carries the coefficients at full
   width. Quantizing the coefficients to int16 as well lets the whole
   contraction run as pmaddwd directly on the int8 state, with one dequant
   per 32 lanes instead of one float convert per element.

   The summation is exact - int32 accumulation of int8 x int16 products
   loses nothing. The ONLY new error source is quantizing the coefficients,
   which is why the precision is a dial rather than a fixed cost:

     LZ_GDN_FIXED=0  float pass 1, the original code.
     LZ_GDN_FIXED=1  one int16 coefficient plane. Coefficient error <= su/2,
                     measured single-step relative error 1.8e-05 on both
                     compilers (closed form: 169/(32767*337) ~= 1.5e-5).
     LZ_GDN_FIXED=2  two planes - the residual the first plane rounds away,
                     re-quantized 32767x finer and summed with the SAME
                     kernel against a second table. Coefficient error drops
                     to su/(2*32767) ~= 4.7e-10 relative, i.e. below float32
                     epsilon: the error this optimization introduces stops
                     being measurable. Costs 2x the integer work in pass 1
                     and needs no new assembly.

   Tier 1 is the default, and the reason is a measurement worth keeping:

   Tier 2 does exactly what it claims arithmetically - on s1v3 the logit
   difference against tier 0 is 2.4e-07 at ntok=2 and 2.1e-07 at ntok=3,
   i.e. 1.9e-05 of tier 1's, matching the predicted 1/32767. Then at ntok=4
   it jumps to 1.5e-02, the same band tier 1 sits in, and the ratio between
   the tiers becomes O(1) with random sign.

   The discontinuity is pass 2. It re-quantizes the state to int8 with
   lz_quantize_q8, and that is a DISCRETE decision: the moment any element
   lands on the other side of a rounding boundary the state differs by one
   LSB, which is ~8.5e-4 - orders of magnitude above either tier's
   coefficient error. There are millions of such decisions per token, so a
   boundary crossing is a matter of when, not if. Past that point the logit
   difference is set by LSB flips, not by coefficient precision.

   Which means the end-to-end ~1e-2 is NOT a precision defect of the fixed
   point pass. It is this engine's standing sensitivity to any perturbation
   of pass 1 - a legal reassociation would produce the same magnitude. All
   tier 2 buys is delaying the first divergence from token 2 to token 4, at
   2x the pass-1 cost. Not worth it for a chat workload; kept compiled-out
   because it becomes the right tier the day the state stops being int8.

   The place to spend on SSM fidelity is therefore pass 2's state
   quantization, not pass 1's coefficients.

   It applies to EVERY build, not only the MMX ones. A tier that is
   deliberately non-bit-identical must not be selected by which compiler
   you used, or rule 2's "all builds byte-compare equal" gate stops meaning
   anything. So the non-MMX path gets a scalar fixed-point summation that
   shares the coefficient table; integer sums are exact under
   reassociation, so the two are bit-identical to each other. */
#if LZ_GDN_FIXED
/* Table is int16_t[LZ_GDN_MAX_KD/2][8]; bump both together. A kd past this
   takes the float pass 1 rather than truncating silently. */
#define LZ_GDN_MAX_KD 256

#if defined(__WATCOMC__) || defined(LZ_GDN1_MMX_EXTERN)
#define LZ_GDN_HAVE_MMX 1
#endif /* __WATCOMC__ || LZ_GDN1_MMX_EXTERN */

#if defined(LZ_GDN_HAVE_MMX)
#if defined(__WATCOMC__)
/* lz_gdn1_x4_asm's Watcom body, and the whole of gdn_sum_gg's MMX
   branch that calls it 8 times per group, moved to src/ops_mmx.c as
   lz_gdn_sum_gg_mmx (task #30) - it writes %mm registers, and wcc386's
   -Nr/-Ns ceiling cannot be raised per-function the way a #pragma aux
   ".586" directive implies (task #28's finding), so it needs its own
   translation unit now that ops.c targets the lower floor. Declared in
   src/ops_mmx.h, already visible here. gdn_sum_gg's Watcom branch below
   calls it once per group - the same call count ops.c always had to
   this function, since it was already the real call boundary before
   this split (a static function, not inlined into its own caller). */
#else
/* lz_gdn1_x4_asm's gcc body lives in src/ops_mmx.c - it writes %mm
   registers. Declared in src/ops_mmx.h, already visible here. */
#endif /* __WATCOMC__ */
#endif /* LZ_GDN_HAVE_MMX */

/* Coefficients for one gg (32 vv lanes): kd/2 kk-pair entries of 8 int16,
   [ckA,ckB,ckA,ckB, cqA,cqB,cqA,cqB] - each broadcast to 4 lanes so
   pmaddwd can take it straight as a memory operand. An odd kd leaves one
   unpaired row, whose coefficients go to tail_cu/tail_cw and are summed in
   scalar code. Every current model has even kd, so that path is
   defensive only. */
typedef struct {
    int16_t t[LZ_GDN_MAX_KD / 2][8];
    int16_t tail_cu, tail_cw;
    float su, sw;         /* dequant scale: amax/32767, or 1/32767 for an all-zero group */
#if LZ_GDN_FIXED >= 2
    /* Second plane: the residual left by the first plane's rounding,
       re-quantized at 32767x finer scale. |residual| <= 0.5 by
       construction, so |t2| <= 16384 - inside int16 with a bit to spare,
       and the accumulator bound is 16384*127*256 = 5.3e8, still inside
       int32. Same kernel, second table: no new assembly, so the
       bit-identity already established for lz_gdn1_x4_asm carries over
       unchanged. */
    int16_t t2[LZ_GDN_MAX_KD / 2][8];
    int16_t tail2_cu, tail2_cw;
    float su2, sw2;       /* = su/32767, sw/32767 */
#endif /* LZ_GDN_FIXED >= 2 */
} LZGdnTable;

static void gdn_build_table(const float *ss, const float *q, const float *k,
                            int kd, int vd, int gg, LZGdnTable *tb) {
    int ng = vd / 32, kk;
    float amk = 0.0f, amq = 0.0f, ik, iq;

    for (kk = 0; kk < kd; kk++) {
        float a = ss[(size_t)kk * ng + gg] * k[kk];
        float b = ss[(size_t)kk * ng + gg] * q[kk];
        float t = (a < 0.0f) ? -a : a;
        if (t > amk) amk = t;
        t = (b < 0.0f) ? -b : b;
        if (t > amq) amq = t;
    }
    /* Rule 6 item 1: `amk / 32767.0f` gets strength-reduced to a multiply
       by Watcom and left as a divide by gcc - 1 ULP apart. Write the
       multiply explicitly. ik/iq are constant-over-variable, a real
       division on both sides, so they are fine as written. */
    tb->su = (amk > 0.0f) ? amk * LZ_Q15_INV : LZ_Q15_INV;
    tb->sw = (amq > 0.0f) ? amq * LZ_Q15_INV : LZ_Q15_INV;
    ik = (amk > 0.0f) ? 32767.0f / amk : 0.0f;
    iq = (amq > 0.0f) ? 32767.0f / amq : 0.0f;
    tb->tail_cu = 0; tb->tail_cw = 0;
#if LZ_GDN_FIXED >= 2
    tb->su2 = tb->su * LZ_Q15_INV;
    tb->sw2 = tb->sw * LZ_Q15_INV;
    tb->tail2_cu = 0; tb->tail2_cw = 0;
#endif /* LZ_GDN_FIXED >= 2 */

    for (kk = 0; kk < kd; kk++) {
        float fa = ss[(size_t)kk * ng + gg] * k[kk] * ik;
        float fb = ss[(size_t)kk * ng + gg] * q[kk] * iq;
        int ia = q8_round(fa);
        int ib = q8_round(fb);
        int kp = kk >> 1, half = kk & 1;
        int tail = (kk == kd - 1 && (kd & 1));
        if (ia >  32767) ia =  32767;
        if (ia < -32767) ia = -32767;
        if (ib >  32767) ib =  32767;
        if (ib < -32767) ib = -32767;
        if (tail) {
            tb->tail_cu = (int16_t)ia;
            tb->tail_cw = (int16_t)ib;
        } else {
            int16_t *slot = tb->t[kp];
            if (half == 0) {
                slot[0] = (int16_t)ia; slot[2] = (int16_t)ia;
                slot[4] = (int16_t)ib; slot[6] = (int16_t)ib;
            } else {
                slot[1] = (int16_t)ia; slot[3] = (int16_t)ia;
                slot[5] = (int16_t)ib; slot[7] = (int16_t)ib;
            }
        }
#if LZ_GDN_FIXED >= 2
        {
            /* Residual of the first plane, in units of su. Clamped
               defensively: |fa - ia| <= 0.5 holds whenever the clamp above
               did not fire, and when it did the residual is bounded by the
               clamp distance instead - which is why the range check is not
               an assert. */
            int i2a = q8_round((fa - (float)ia) * 32767.0f);
            int i2b = q8_round((fb - (float)ib) * 32767.0f);
            if (i2a >  32767) i2a =  32767;
            if (i2a < -32767) i2a = -32767;
            if (i2b >  32767) i2b =  32767;
            if (i2b < -32767) i2b = -32767;
            if (tail) {
                tb->tail2_cu = (int16_t)i2a;
                tb->tail2_cw = (int16_t)i2b;
            } else {
                int16_t *s2 = tb->t2[kp];
                if (half == 0) {
                    s2[0] = (int16_t)i2a; s2[2] = (int16_t)i2a;
                    s2[4] = (int16_t)i2b; s2[6] = (int16_t)i2b;
                } else {
                    s2[1] = (int16_t)i2a; s2[3] = (int16_t)i2a;
                    s2[5] = (int16_t)i2b; s2[7] = (int16_t)i2b;
                }
            }
        }
#endif /* LZ_GDN_FIXED >= 2 */
    }
}

/* gdn_tail_row: src/ops_kernel_shared.h (task #31) - src/ops_mmx.c's
   MMX branch needs the exact same odd-tail-row fixup. */

#if !defined(LZ_GDN_HAVE_MMX)
/* Non-MMX builds. Same table, same integer arithmetic, different summation
   order - and integer sums are exact under reassociation, so this is
   bit-identical to the MMX version. Verified over 200000+ random groups
   plus extremes. */
static void gdn_sum_gg(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg) {
    int kpairs = kd / 2, kp, j;
    for (j = 0; j < 32; j++) { au_gg[j] = 0; aw_gg[j] = 0; }
    for (kp = 0; kp < kpairs; kp++) {
        const int8_t *rowA = sq_gg + (size_t)(2 * kp) * vd;
        const int8_t *rowB = sq_gg + (size_t)(2 * kp + 1) * vd;
        int32_t ckA = tab[kp][0], ckB = tab[kp][1];
        int32_t cqA = tab[kp][4], cqB = tab[kp][5];
        for (j = 0; j < 32; j++) {
            au_gg[j] += (int32_t)rowA[j] * ckA + (int32_t)rowB[j] * ckB;
            aw_gg[j] += (int32_t)rowA[j] * cqA + (int32_t)rowB[j] * cqB;
        }
    }
    gdn_tail_row(sq_gg, kd, vd, tcu, tcw, au_gg, aw_gg);
}
#elif defined(__WATCOMC__)
/* Watcom: one call to src/ops_mmx.c's lz_gdn_sum_gg_mmx, which does the
   eight sub-window lz_gdn1_x4_asm calls internally with the pragma
   genuinely inline (task #30) - the same call count this function
   always had from ITS OWN caller, just crossing a TU boundary now. */
static void gdn_sum_gg(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg) {
    lz_gdn_sum_gg_mmx(sq_gg, kd, vd, tab, tcu, tcw, au_gg, aw_gg);
}
#else
/* Eight 4-lane sub-windows cover this gg's 32 vv lanes.

   The emms belongs HERE, once per gg - not per sub-window, and not after
   the whole loop. Rule 6 item 6 permits hoisting the emms out only when
   the loop body has no x87, and that precondition fails one level up: the
   next gg's gdn_build_table runs q8_round and 32767/amax on x87 right
   after this returns. Hoisting it would reopen the trap that broke the
   mmx tier. */
static void gdn_sum_gg(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg) {
    int kpairs = kd / 2, vb;
    for (vb = 0; vb < 8; vb++) {
        int v = vb * 4;
        int32_t out8[8];
        lz_gdn1_x4_asm(sq_gg + v, vd, &tab[0][0], kpairs, out8);
        au_gg[v + 0] = out8[0]; au_gg[v + 1] = out8[1];
        au_gg[v + 2] = out8[2]; au_gg[v + 3] = out8[3];
        aw_gg[v + 0] = out8[4]; aw_gg[v + 1] = out8[5];
        aw_gg[v + 2] = out8[6]; aw_gg[v + 3] = out8[7];
    }
    gdn_tail_row(sq_gg, kd, vd, tcu, tcw, au_gg, aw_gg);
    if (lz_cpu_has_mmx()) _mm_empty();
}
#endif /* LZ_GDN_HAVE_MMX */

/* au/aw come back as scaled-up int32 with su/sw the matching dequant
   scale. The accumulator bound is 127*32767*256 ~= 1.06e9, far past 2^24,
   so the caller must convert with lz_i32f() - rule 6 item 7. */
static void gdn_pass1_fixed(const int8_t *sq,
#if LZ_GDN_STATE_2PLANE
                            const int8_t *sq_lo, int32_t *aul, int32_t *awl,
#endif /* LZ_GDN_STATE_2PLANE */
                            const float *ss,
                            const float *q, const float *k, int kd, int vd,
                            int32_t *au, int32_t *aw, float *su, float *sw,
                            int32_t *au2, int32_t *aw2, float *su2, float *sw2) {
    int gg, ng = vd / 32;
    /* static, not stack: 2-4 KB of table on a Win98 stack, per rule 6 item 4. */
    static LZGdnTable tb;
#if LZ_GDN_FIXED < 2
    (void)au2; (void)aw2; (void)su2; (void)sw2;
#endif /* LZ_GDN_FIXED < 2 */
    for (gg = 0; gg < ng; gg++) {
        const int8_t *sg = sq + (size_t)gg * 32;
        gdn_build_table(ss, q, k, kd, vd, gg, &tb);
        gdn_sum_gg(sg, kd, vd, tb.t, tb.tail_cu, tb.tail_cw,
                   au + gg * 32, aw + gg * 32);
        su[gg] = tb.su; sw[gg] = tb.sw;
#if LZ_GDN_STATE_2PLANE
        /* Low state plane: same kernel, same coefficient table, different
           state plane. Its dequant weight is a constant 1/254 relative to
           the high plane, so no second scale array is needed. */
        gdn_sum_gg(sq_lo + (size_t)gg * 32, kd, vd, tb.t, tb.tail_cu, tb.tail_cw,
                   aul + gg * 32, awl + gg * 32);
#endif /* LZ_GDN_STATE_2PLANE */
#if LZ_GDN_FIXED >= 2
        /* Second plane, same kernel, second table. The MMX gdn_sum_gg emits
           its own emms, so this runs a redundant one - kept rather than
           hoisted, because rule 6 item 6's precondition ("no x87 in the
           loop body") fails here: gdn_build_table for the NEXT gg is x87.
           One extra tag-word write per gg is not worth reopening that trap. */
        gdn_sum_gg(sg, kd, vd, tb.t2, tb.tail2_cu, tb.tail2_cw,
                   au2 + gg * 32, aw2 + gg * 32);
        su2[gg] = tb.su2; sw2[gg] = tb.sw2;
#endif /* LZ_GDN_FIXED >= 2 */
    }
}
#endif /* LZ_GDN_FIXED */

/* ---- measurement scaffolding: exact float SSM state --------------------

   Never in a shipping build. `-DLZ_GDN_STATE_F32=1` makes the recurrence
   keep an exact float state instead of the int8 one, so the question
   "how far above an exact recurrence does this engine already sit, before
   pass 1's coefficients are quantized at all" is measurable rather than
   argued from amax/127.

   Keyed by the int8 state pointer the caller passes, which is 1:1 with
   (layer, head), so no signature change and no forward.c change - the
   scaffolding cannot perturb the shipping path by construction. The
   linear scan is O(layers*heads) per call and irrelevant at these
   sizes. */
#ifndef LZ_GDN_STATE_F32
#define LZ_GDN_STATE_F32 0
#endif /* LZ_GDN_STATE_F32 */

#if LZ_GDN_STATE_2PLANE
#if LZ_GDN_FIXED >= 2
#error "LZ_GDN_STATE_2PLANE with LZ_GDN_FIXED=2 is not implemented: it would be four kernel passes, and the coefficient plane was measured not to be the bottleneck. Use LZ_GDN_FIXED=0 or 1."
#endif /* LZ_GDN_FIXED >= 2 */

/* LZ_GDN_LO_SCALE now lives in ops.h - forward.c's KV path needs the same
   constant (LZ_KV_2PLANE), and two copies of a dequant weight is exactly
   the kind of duplicate that drifts. */

/* Two-plane quantize of one state row. Deliberately mirrors
   lz_quantize_q8: same integer-domain absmax, same amax*(1/127) scale
   written as a multiply (rule 6 item 1), same q8_round, and the SAME
   three rounding tiers. `hi` alone therefore matches what
   lz_quantize_q8 would have written, which makes
   -DLZ_GDN_STATE_2PLANE=0 vs 1 a clean A/B rather than two unrelated
   quantizers.

   Both planes go through the vectorized rounding kernel, the low one by
   feeding it the residual with a multiplier of 254 - the kernel computes
   round(x*inv) and does not care what x means. Rounding both planes in
   scalar code instead would cost +51.8% per token on s1v3: pass 2's
   quantize is the dominant term in lz_gdn_step, and the SSE2 rounding
   tier is ~25x the scalar magic-number path, so dropping to scalar
   there would swamp everything the two planes buy. What is left scalar
   is only the residual itself (one multiply, one int8->float, one
   subtract per element).

   `fc_site`: which census site to bill this call's f32 work to, or -1
   to bill nothing. Billed here, not at the call site: the per-element
   cost depends on q8_group_scale's `fast` flag and on lz_q8r_tier(),
   both internal to this loop, and duplicating that dispatch at a caller
   is a second copy of it that will drift. */
void lz_gdn_quantize_2p(const float *x, int n, int gs,
                        int8_t *hi, int8_t *lo, float *s, int fc_site) {
    int g, k, ng = n / gs, fast;
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
    int tier = ((gs & 31) == 0) ? lz_q8r_tier() : 0;
#ifdef LZ_Q8R_COUNT
    /* Which tier this call actually took, counted rather than inferred.
       Off unless a probe defines the macro.
       Exists because "the dispatch condition looks right" is not an
       answer to "which one ran" - the same distinction that let a whole
       tier of hand-written assembly stay out of the Watcom build for
       months. */
    extern long lz_q8r_hits[3];
    lz_q8r_hits[tier < 0 ? 0 : (tier > 2 ? 2 : tier)]++;
#endif /* LZ_Q8R_COUNT */
    static const float lo_mul = LZ_GDN_LO_SCALE;
    /* static, not stack (rule 6 item 4); one row's residual at most. */
    static float res[LZ_GDN_MAX_VD];
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */
    for (g = 0; g < ng; g++) {
        const float *grp = x + (size_t)g * gs;
        int8_t *ho = hi + (size_t)g * gs;
        int8_t *lw = lo + (size_t)g * gs;
        float inv;
        fast = q8_group_scale(grp, gs, &s[g], &inv);
        /* Paid once per group regardless of which branch below runs. */
        if ((unsigned)fc_site < (unsigned)LZ_FC_N) LZ_FCX(fc_site, 1, 0, 1, 0, 0);  /* amax*(1/127) mul, 127/amax div */
        if (!fast) {
            /* Same shape as the fast&&tier==0 branch below (that one
               multiplies instead of dividing), same count. */
            if ((unsigned)fc_site < (unsigned)LZ_FC_N) LZ_FCX(fc_site, gs, 3 * gs, gs, gs, 0);  /* div, q8_round(add), convert, sub(add), mul, q8_round(add) */
            for (k = 0; k < gs; k++) {
                float f = grp[k] / s[g];
                int h = q8_round(f);
                int l;
                if (h >  127) h =  127;
                if (h < -127) h = -127;
                l = q8_round((f - (float)h) * LZ_GDN_LO_SCALE);
                /* +-127, NOT +-128, even though int8 holds -128 and L =
                   256 can produce it. The SIMD rounding kernel clamps at
                   +-127 (pminsw/pmaxsw against 127/-127) and cannot be
                   told otherwise, so allowing -128 here would make the
                   scalar and SIMD tiers disagree - the quality probe
                   catches that as the FLOAT path getting 2.2x worse
                   when it should improve. */
                if (l >  127) l =  127;
                if (l < -127) l = -127;
                ho[k] = (int8_t)h;
                lw[k] = (int8_t)l;
            }
            continue;
        }
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
        if (tier) {
            /* The lz_q8round32_simd/_sse calls bracketing the visible
               residual line below are not free - reading only the C
               between them would miss 4 of the 7. */
            if ((unsigned)fc_site < (unsigned)LZ_FC_N) LZ_FCX(fc_site, 3 * gs, gs, 0, 3 * gs, 0);  /* residual mul+convert+sub=3, 2x kernel mul+convert=4 */
#if defined(__WATCOMC__)
            /* task #30 (4/N): one call per GROUP for the WHOLE two-plane
               round (both kernel loops and the residual arithmetic
               between them), not one per 32-element sub-chunk - same
               reasoning as lz_quantize_q8 above. lz_q8round_2p_group_
               sse2/_sse own the residual computation too (plain float
               arithmetic, safe in a Watcom #pragma-aux TU the way it is
               not in a gcc -mmmx one - see src/ops_mmx.c's header for
               why that split is toolchain-specific), so the hand-off is
               still one call, not "kernel, kernel, residual" three. */
#ifdef LZ_HAVE_Q8R_SIMD
            if (tier == 2)
                lz_q8round_2p_group_sse2(grp, ho, lw, gs, &inv, &lo_mul, res);
            else
#endif /* LZ_HAVE_Q8R_SIMD */
                lz_q8round_2p_group_sse(grp, ho, lw, gs, &inv, &lo_mul, res);
#else
#ifdef LZ_HAVE_Q8R_SIMD
            if (tier == 2) {
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_simd(grp + k, ho + k, &inv);
                for (k = 0; k < gs; k++)
                    res[k] = grp[k] * inv - (float)ho[k];
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_simd(res + k, lw + k, &lo_mul);
            } else
#endif /* LZ_HAVE_Q8R_SIMD */
            {
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_sse(grp + k, ho + k, &inv);
                /* emms before the x87 residual arithmetic below, and
                   again before the next group's amax/inv - same reason as
                   the identical placement in lz_quantize_q8 (rule 6 item
                   6's precondition does not hold here). */
                if (lz_cpu_has_mmx()) _mm_empty();
                for (k = 0; k < gs; k++)
                    res[k] = grp[k] * inv - (float)ho[k];
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_sse(res + k, lw + k, &lo_mul);
                if (lz_cpu_has_mmx()) _mm_empty();
            }
#endif /* __WATCOMC__ */
            continue;
        }
#endif /* LZ_HAVE_Q8R_SIMD || LZ_HAVE_Q8R_SSE */
        /* fast && tier==0: no SIMD rounding tier, e.g. --kernel ref. Same
           shape as the !fast branch above, multiply instead of divide. */
        if ((unsigned)fc_site < (unsigned)LZ_FC_N) LZ_FCX(fc_site, 2 * gs, 3 * gs, 0, gs, 0);  /* mul, q8_round(add), convert, sub(add), mul, q8_round(add) */
        for (k = 0; k < gs; k++) {
            float f = grp[k] * inv;
            int h = q8_round(f);
            int l;
            if (h >  127) h =  127;
            if (h < -127) h = -127;
            l = q8_round((f - (float)h) * LZ_GDN_LO_SCALE);
            if (l >  127) l =  127;
            if (l < -127) l = -127;
            ho[k] = (int8_t)h;
            lw[k] = (int8_t)l;
        }
    }
}
#endif /* LZ_GDN_STATE_2PLANE */

#if LZ_GDN_STATE_F32
#define LZ_GDN_SHADOW_MAX 1024
static const int8_t *g_shk[LZ_GDN_SHADOW_MAX];
static float *g_shv[LZ_GDN_SHADOW_MAX];
static int g_shn;
static float *gdn_shadow(const int8_t *sq, int n) {
    int i;
    for (i = 0; i < g_shn; i++) if (g_shk[i] == sq) return g_shv[i];
    if (g_shn >= LZ_GDN_SHADOW_MAX) return NULL;
    g_shv[g_shn] = (float *)calloc((size_t)n, sizeof(float));
    if (!g_shv[g_shn]) return NULL;
    g_shk[g_shn] = sq;
    return g_shv[g_shn++];
}
#endif /* LZ_GDN_STATE_F32 */

/* ---- SSM recurrence pass 2, fixed point --------------------------------

   The counterpart to gdn_pass1_fixed, and the reason it exists is a
   measurement rather than symmetry: on the all-scalar tier - which is
   the ONLY tier a Pentium II has, since lz_q8round32 structurally
   cannot have an MMX one - the write-back quantize is 83% of
   lz_gdn_step and pass 2's arithmetic a further 11% (Watcom). Both
   are stuck on x87 for one reason: pass 2 hands the
   quantizer a FLOAT. Producing int32 instead turns the quantize into
   int32 -> int8, which is packssdw/packsswb - pure MMX. The blocker is
   not the instruction set, it is the type at the boundary.

   TWO FACTS MAKE THIS SMALL, and both are worth stating because the
   first design draft treated the opposite as make-or-break:

   1. THE TWO-PLANE STATE IS ALREADY AN int16 STATE. With
      H = hi*256 + lo the value is exactly H*(s/256), and
      |H| <= 127*256+127 = 32639 < 32767. So there
      are no "two planes" in the integer path: pack on the way in, split
      on the way out, and the split is EXACT and division-free
      (hi = (H+128)>>8, lo = H - hi*256) where the float
      path rounds twice.

   2. THE SCALE ARITHMETIC STAYS IN FLOAT. The 83% is per ELEMENT
      (kd*vd); s_acc, the multipliers and the reciprocal are per GROUP
      (kd*ng, one thirty-second as many). Leaving those on x87 costs
      nothing measurable and removes the whole problem of aligning two
      fixed-point exponents. Same ratio argument that makes "integer
      contraction, float epilogue" right for the matmul kernels.

   Per element what remains is A = m1*H + m2*Dq, which is exactly what
   pmaddwd computes - one instruction for the whole accumulate.

   THE 32-BIT TRAP: A reaches 2.1e9, so the natural A*32258/amax rescale
   is 3.5e13 and
   overflows. `long` is 32-bit on BOTH Watcom and MinGW (LLP64), so this
   is not a host-only hazard. amax is shifted into 16 bits first, which
   keeps exactly the 15 bits H needs and is also the shape the MMX
   kernel wants (a per-group int16 reciprocal for pmulhw).

   AND BOTH SHIFTS ROUND. An arithmetic shift right truncates toward
   -inf, which is a BIAS; bias adds linearly where quantization noise
   adds in quadrature. Measured on the way here: 5.88e-05 with the two
   shifts truncating, 4.16e-05 with them rounded, against a float pass 2
   at 1.78e-05.

   ACCURACY, stated plainly: "no worse than the float path" is
   structurally unachievable - the float path's only error source is
   the final
   quantization, and this path has that PLUS coefficient quantization
   PLUS delta quantization PLUS the rescale. Measured added error is
   ~3.8e-05, about 2x the float path's own. For comparison the pass-1
   fixed tier that ships by default adds ~1.5e-05 to the same baseline.
   That is why this tier is OFF by default and gated on an end-to-end
   paired-NLL measurement, not on this operator-level number.

   Bit-identity across builds is free HERE and not by luck: everything
   per element is integer, and integer sums are exact under
   reassociation, so a future MMX twin agrees with this scalar path by
   construction - the same argument gdn_pass1_fixed's own comment makes.
   The float lines obey rule 6 item 1 (every constant divisor written as
   a multiply by its reciprocal; the one real division has a variable
   divisor, which both compilers emit as a true divide). */

/* expof and norm_ss_fixed are in ops_quant.c. */

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
static void lz_rmsnorm_gated_out_fixed(float *o, const float *x,
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
static void lz_rmsnorm_gated_out_fixed(float *o, const float *x,
                                       const float *g, const float *w,
                                       float inv, int n, int32_t *sig) {
    (void)o; (void)x; (void)g; (void)w; (void)inv; (void)n; (void)sig;
}
#endif /* LZ_GDN_FIXED */

#if LZ_GDN_FIXED
/* THE SPLIT IS A SHIFT, AND THE SHIFT IS 8 BECAUSE L IS 256. Every
   `>> 8`, `<< 8` and `* 256` in the fixed pass 2 is that one constant
   written three ways; ops.h's LZ_GDN_LO_SCALE is the same number a
   fourth time, in float. Building with -DLZ_GDN_LO_SCALE=254.0f - the
   A/B control ops.h advertises - would leave all of them behind and
   compute a wrong answer silently, so it is now a build error. */
typedef char lz_p2_needs_lo_scale_256[
    ((int)(LZ_GDN_LO_SCALE) == 256) ? 1 : -1];

#include "ops_kernel_p2.h"        /* the MMX pair; defines lz_p2_blk and,
                                     on Watcom, LZ_HAVE_P2_MMX_ASM */

/* p2_blk() moved to src/ops_mmx.c in task #30 (2/N), declared
   unconditionally in src/ops_mmx.h - both Watcom's lz_p2_rows_mmx/
   lz_p2_rows_sse (this task) and gcc's LZ_P2_MMX_EXTERN case (predates
   it) need the SAME 16-byte-aligned scratch buffer, and now cross a TU
   boundary to reach the code that fills it, so it has to be one real
   exported function rather than a static one this file's Watcom
   #pragma aux bodies used to reach for free. */

#if defined(LZ_HAVE_P2_MMX_ASM) || defined(LZ_P2_MMX_EXTERN)
#define LZ_HAVE_P2_MMX 1
/* No lz_p2_mul32_mmx/lz_p2_split32_mmx bare-name aliases here anymore
   (task #30, 2/N): those existed only so the #pragma aux bodies, once
   textually inline in THIS file, could be called by a name that also
   worked for gcc's real functions of the same name. Now that the
   bodies live in src/ops_mmx.c (declared in src/ops_mmx.h for both
   toolchains), Watcom's Phase 2 below calls lz_p2_rows_mmx/_sse/_sse2
   directly instead - the bare per-group names stay gcc's alone, used
   only by the gcc branch of gdn_p2_row_simd's Phase 2. */
#endif /* LZ_HAVE_P2_MMX_ASM || LZ_P2_MMX_EXTERN */

/* ---- the SSE1 tier: one instruction of content, and it is the split ---
   The A term and the absmax have no SSE1 form at all (ops_kernel_p2.h
   enumerates SSE1's whole addition to the MMX register file against what
   this kernel needs), so this tier is the MMX mul32 paired with a split
   whose low-plane clamp is a single pmaxsw. Named for the ISA a machine
   must HAVE to run it, which is what dispatch turns on. */
#if defined(LZ_HAVE_P2_SSE_ASM) || defined(LZ_P2_MMX_EXTERN)
#define LZ_HAVE_P2_SSE 1
/* No lz_p2_split32_sse bare-name alias here anymore (task #30, 2/N) -
   its body lives in src/ops_mmx_sse.c now for BOTH toolchains, named
   "sse" for the ISA a machine must HAVE to run it even though it still
   operates on %mm (SSE1's whole addition to the integer side of the
   MMX register file is used here exactly once: pmaxsw does the low
   plane's clamp to -127 directly). Watcom's Phase 2 reaches it only
   through lz_p2_rows_sse, declared in src/ops_mmx.h; gcc's per-group
   loop still calls it by its real name, also declared there. */
#endif /* LZ_HAVE_P2_SSE_ASM || LZ_P2_MMX_EXTERN */

/* ---- the 128-bit tier -------------------------------------------------
   Every instruction of the MMX pair exists on xmm in SSE2 - pmaddwd,
   psrad, packssdw, packsswb, the pcmpgtd/pand/pandn/por max tree, the
   saturating pair that clamps the low plane - so this is the same kernel
   twice as wide, not a different algorithm. It is a real tier rather
   than a rewrite of an existing one, and the machines it is for are the
   P4 Northwood and the Pentium M.

   NOT pmaxsd: that is SSE4.1. On SSE2 an int32 maximum still costs the
   four-instruction mask dance, so the absmax stays the expensive half
   here exactly as it is in MMX.

   Alignment is now load-bearing: movdqa and the m128 shift-count operand
   FAULT on a misaligned address, which is why p2_blk() aligns by hand.
   The caller's state planes get movdqu/movq, since their alignment is
   not ours to assume. */
#if defined(LZ_HAVE_P2_SSE2_ASM) || defined(LZ_P2_SSE2_EXTERN)
#define LZ_HAVE_P2_SSE2 1
/* One flag, two triggers: LZ_HAVE_P2_SSE2_ASM means Watcom's #pragma
   aux SSE2 pair is in the build, LZ_P2_SSE2_EXTERN means gcc's
   extern-linked intrinsics are - different code for the same operator
   on different compilers, both bodies in src/ops_sse2.c. No bare-name
   aliases here anymore (same reasoning as the MMX and SSE1 tiers above):
   Watcom's Phase 2 reaches this tier only through lz_p2_rows_sse2,
   declared in src/ops_sse2.h alongside the gcc-only per-group names,
   which the intrinsics' declarations share. */
#endif /* LZ_HAVE_P2_SSE2_ASM || LZ_P2_SSE2_EXTERN */

#if defined(LZ_HAVE_P2_SSE2) && !defined(LZ_HAVE_P2_MMX)
#error "SSE2 pass 2 without the MMX one: gdn_p2_row_simd is gated on MMX"
#endif /* LZ_HAVE_P2_SSE2 && !LZ_HAVE_P2_MMX */

#ifdef LZ_HAVE_P2_MMX
/* 0 scalar, 1 MMX, 2 SSE2. Simpler than lz_q8r_tier for
   one reason: this operator's tiers are pure integer SIMD, so they line
   up exactly with the kernel tier g_kernel already carries - unlike the
   q8 rounding, where a PIII has 64-bit integer SIMD but 128-bit float
   SSE and the two axes come apart.

   MMX is the floor of the whole target family (the machine table has a
   check in the MMX column on every row), so a machine that lands on
   scalar here is one where kernel_detect already said LZ_KERNEL_REF.
   No FORCE knob: unlike the SSE1 tiers, both of these are selected by
   default on some machine that runs the suite, so both are validated by
   every run rather than only by a probe. */
static int p2_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return 0;
#ifdef LZ_HAVE_P2_SSE2
    if (g_kernel == LZ_KERNEL_SSE2) return 3;
#endif /* LZ_HAVE_P2_SSE2 */
    /* SSE1 is read from CPUID rather than from g_kernel, the same way
       lz_q8r_tier does it and for the same reason: there is no
       LZ_KERNEL_SSE, and on a PIII g_kernel is LZ_KERNEL_MMX while the
       machine does have SSE. Deriving this from g_kernel would leave the
       tier unreachable on exactly the machines it exists for, and
       nothing would report it - the tiers are bit-identical.

       LZ_P2_FORCE_MMX is the mirror of LZ_Q8R_FORCE_SSE. Without it
       the PURE MMX split becomes unreachable on any machine with SSE1,
       which is every machine that can run this test suite, and a path
       that cannot be selected is a path that cannot be validated. */
#if defined(LZ_HAVE_P2_SSE) && !defined(LZ_P2_FORCE_MMX)
    /* REAL CPUID, not "the build carries the tier" - the same fix
       kernel_detect() needed (task #22): LZ_HAVE_P2_SSE being defined
       means src/ops_mmx_sse.c's lz_p2_split32_sse is linked in, which
       is true on every gcc build including the 32-bit x87 target, where
       it is not the same fact as "this machine has SSE1" (Pentium
       MMX/K6/Cyrix 6x86 in the Socket 7 family do not). True on x86-64
       unconditionally (SSE1 is ABI-guaranteed there), so nothing changes
       for that build. This function is the other hot-loop caller the
       big comment above lz_cpu_has_sse() names (once per state row), so
       it uses the cached function too, not a fresh cpuid per call. */
    if (lz_cpu_has_sse()) return 2;
#endif /* LZ_HAVE_P2_SSE && !LZ_P2_FORCE_MMX */
    return 1;
}
#endif /* LZ_HAVE_P2_MMX */

/* --- the per-group float and integer work, shared by both tiers -------

   These four exist as functions rather than as lines inside the scalar
   loop for one reason: the SIMD twin below needs every one of them, and
   a threshold computed in two places is a threshold that will end up
   with two values. What differs between the tiers is the 32-element
   integer body and nothing else. */

/* c1 weights the decayed old state, c2 the new delta term. ONE copy of
   these two expressions in the engine - the scalar and SIMD rows both
   call this - because they are the only float arithmetic left that is
   not exact, so two textually-equal copies could still differ by which
   intermediate the compiler chose to keep in an x87 register. */
static void p2_group_coef(float dec, float rs, float kv, float sd,
                          float *c1, float *c2) {
    *c1 = dec * rs * (1.0f / LZ_GDN_LO_SCALE);
    *c2 = kv * sd;
}

/* The float prologue, reduced to what the integer body can consume: two
   int16 multipliers and the power of two they were normalized against.
   Returns 0 for a group that flushes to zero.

   Why the normalization is by powers of two only, and why that is the
   whole reason this tier survives cross-compiler comparison, is written
   out at the top of gdn_p2_group_fixed's shift section below.

   THE +-32767 CLAMP IS NEW AND IT MOVES NUMBERS, so it is stated rather
   than buried. Normalization puts |t| in [2^14, 2^15), and q8_round of a
   t at or above 32767.5 returns 32768 - one past int16. The scalar path
   held that in an int32 and multiplied happily; pmaddwd cannot, and
   32768 truncated to int16 is -32768, a sign flip on the dominant
   coefficient. So BOTH paths clamp, and the fixed tier moves by one ULP
   of its own multiplier (3e-5 relative) on the ~1.5e-5 of groups that
   land in that last half-step. -32768 would not need clamping, but the
   bound is kept symmetric: a one-sided bound reads like an oversight and
   gets "fixed" in the wrong direction later. */
static int p2_group_norm(float c1, float c2, int16_t *mul4, float *sacc) {
    float cm1 = (c1 < 0.0f) ? -c1 : c1;
    float cm2 = (c2 < 0.0f) ? -c2 : c2;
    float cmax = (cm1 > cm2) ? cm1 : cm2;
    float p, t1, t2;
    int e, m1, m2;

    /* FLUSH BEFORE NORMALIZING, and this is a correctness guard rather
       than a shortcut. If cmax is small enough that the accumulator
       scale lands in the subnormal range, x87 and SSE stop agreeing
       about it - PC=24 fixes the mantissa at 24 bits but leaves the
       EXPONENT range extended, so a value held in an x87 register is
       normal where its float32 store is subnormal, and whether it gets
       stored is a register-allocation decision. Not theoretical: without
       this guard the two builds diverge at recurrence step 26, and
       compiling the tracing probe - whose only effect is to STORE these
       intermediates - makes the divergence disappear. A bug that hides
       when you look at it is this mechanism, every time.

       Same constant as lz_quantize_q8 (LZ_Q8_MIN_SCALE, ops.h): two
       operators disagreeing about where zero starts would disagree about
       results. A group with cmax below 1e-30 contributes |term| <=
       3e-26 next to activations of order 1. */
    if (!(cmax >= LZ_Q8_MIN_SCALE_F)) {
        mul4[0] = mul4[1] = mul4[2] = mul4[3] = 0;
        *sacc = 0.0f;
        return 0;
    }
    e  = expof(cmax);               /* cmax >= LZ_Q8_MIN_SCALE, so normal */
    p  = pow2f(14 - e);             /* exact */
    t1 = c1 * p;                    /* exact: multiply by a power of two */
    t2 = c2 * p;
    *sacc = pow2f(e - 14);          /* exact reciprocal of p */
    m1 = (t1 > -0.5f && t1 < 0.5f) ? 0 : q8_round(t1);
    m2 = (t2 > -0.5f && t2 < 0.5f) ? 0 : q8_round(t2);
    if (m1 >  32767) m1 =  32767;
    if (m1 < -32767) m1 = -32767;
    if (m2 >  32767) m2 =  32767;
    if (m2 < -32767) m2 = -32767;
    mul4[0] = (int16_t)m1;
    mul4[1] = (int16_t)m2;
    mul4[2] = (int16_t)m1;         /* the pair, twice: pmaddwd reads two */
    mul4[3] = (int16_t)m2;         /* dwords per register */
    return 1;
}

/* p2_shift_of: src/ops_kernel_shared.h (task #31) - the "shift not
   divide" rationale lives there now, alongside the two SIMD tiers
   (src/ops_mmx.c, src/ops_mmx_sse.c, src/ops_sse2.c) that need the
   exact same rescale shift. */

/* The float epilogue. s_new stays exact: sacc is already a power of two
   from the normalization, so multiplying by 2^sh and by L introduces no
   rounding of its own. Returns 0 when the group must be written as an
   exact zero - either the prologue flushed it (sacc == 0) or the new
   scale itself falls under the floor. */
static int p2_group_scale(float sacc, int sh, float *os) {
    float s_new;
    if (!(sacc > 0.0f)) { *os = 1.0f; return 0; }
    s_new = sacc * pow2f(sh) * LZ_GDN_LO_SCALE;
    if (!(s_new >= LZ_Q8_MIN_SCALE_F)) { *os = 1.0f; return 0; }
    *os = s_new;
    return 1;
}

/* The whole fixed pass 2 for one head, shared by lz_gdn_step and
   lz_kda_step. THIS WRAPPER IS THE UNIT THAT KEEPS GETTING FORGOTTEN,
   which is why it is a function rather than two copies.

   Omission is silent - the tier simply does nothing on the family that
   lacks it, both settings produce the same output, and it reads as
   "this tier is not worth much". KunMoe is 0 gdn / 6 kda / 2 full, so
   forgetting to wire pass 1 or pass 2 fixed into a new family is the
   only way it goes missing.

   `gvec` NULL means the scalar-decay family (GDN, one gt for the head);
   non-NULL means one decay per row (KDA). That is the ONLY difference
   between the two families here - everything else, including the
   per-token delta quantization, is character-for-character identical. */
static void gdn_p2_fixed_rows(const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                              const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                              const float *ss_in, float *ss_out,
                              const float *delta, const float *k,
                              const float *gvec, float gt,
                              int kd, int vd);

static void gdn_p2_group_fixed(const int8_t *ph,
#if LZ_GDN_STATE_2PLANE
                               const int8_t *pl,
#endif /* LZ_GDN_STATE_2PLANE */
                               float c1, const int16_t *dq, float c2,
                               int8_t *oh,
#if LZ_GDN_STATE_2PLANE
                               int8_t *ol,
#endif /* LZ_GDN_STATE_2PLANE */
                               float *os) {
    int32_t A[32];
    int16_t mul4[4];
    int32_t amax = 0;
    float sacc;
    int j, sh = 0;

    if (p2_group_norm(c1, c2, mul4, &sacc)) {
        int32_t m1 = mul4[0], m2 = mul4[1];
        for (j = 0; j < 32; j++) {
#if LZ_GDN_STATE_2PLANE
            int32_t H = ((int32_t)ph[j] << 8) + (int32_t)pl[j];
#else
            /* hi*L, and L is 256 - NOT 254. The caller's c1 carries 1/L
               in both representations, so a single-plane H of hi*254
               would shrink every decay term by 0.78% per step -
               invisible in the default build, which is two-plane; the
               A/B control build is the only one that would run it.
               Caught by writing the MMX twin, which has to state what H
               is. */
            int32_t H = (int32_t)ph[j] << 8;
#endif /* LZ_GDN_STATE_2PLANE */
            A[j] = m1 * H + m2 * (int32_t)dq[j];
            if (A[j] > amax) amax = A[j];
            if (-A[j] > amax) amax = -A[j];
        }
        sh = p2_shift_of(amax);

        for (j = 0; j < 32; j++) {
            /* Rounded shift, not truncating: an arithmetic shift right
               rounds toward -inf, which is a BIAS, and bias adds
               linearly where quantization noise adds in quadrature.
               Measured on this routine: two truncating shifts cost
               5.88e-05 against 4.16e-05 for two rounded ones. */
            int32_t hn = (sh > 0) ? ((A[j] + (1 << (sh - 1))) >> sh) : A[j];
            /* Split by SHIFT, which is what LZ_GDN_LO_SCALE = 256
               bought. (hn + 128) >> 8 is round-to-nearest for a
               power-of-two divisor, exact, and native on MMX (paddw +
               psraw); the /254 form needs a divide, which has no MMX
               equivalent (ops.h has the exhaustive search that closed
               that door). */
            int h = (int)((hn + 128) >> 8);
            if (h >  127) h =  127;
            if (h < -127) h = -127;
            oh[j] = (int8_t)h;
#if LZ_GDN_STATE_2PLANE
            {
                int32_t l = hn - ((int32_t)h << 8);
                /* Same +-127 as the float quantizer, for the same
                   reason: the two must agree about what a state code
                   word can be. Only -128 is reachable here, which is
                   what lets the MMX twin clamp with a saturating
                   subtract instead of a compare. */
                if (l >  127) l =  127;
                if (l < -127) l = -127;
                ol[j] = (int8_t)l;
            }
#endif /* LZ_GDN_STATE_2PLANE */
        }
    }
    if (!p2_group_scale(sacc, sh, os)) {
        for (j = 0; j < 32; j++) {
            oh[j] = 0;
#if LZ_GDN_STATE_2PLANE
            ol[j] = 0;
#endif /* LZ_GDN_STATE_2PLANE */
        }
    }
}

/* --- the SIMD twin, one whole ROW rather than one group ---------------

   The MMX cell for this operator. Bit-identical to the scalar
   above by construction, not by tolerance: every step is the same
   integer operation on the same integers, verified at 100k scale.

   WHY A ROW AND NOT A GROUP - it is entirely about rule 6 item 6. The
   emms may only be hoisted out of a loop whose body has no x87, and the
   per-group body has plenty: the coefficients, the normalization, the
   new scale. So the row is PHASE-SPLIT instead - all the float work of
   the row first, then all the MMX work, then one emms, then the float
   epilogue. One emms per row instead of one per 32 elements, which is
   the ratio that costs 16% of the matmul compute budget on
   a P6 and worse on a P4.

   The price is three loops over ng and three small static arrays, and
   the risk is that this sequencing drifts from the scalar one. That is
   why every float decision inside it - coefficients, multipliers,
   flush thresholds, new scale - is a call into the SAME helper the
   scalar path calls. What is duplicated here is the order of the
   phases, nothing else. */
#ifdef LZ_HAVE_P2_MMX
static void gdn_p2_row_simd(const int8_t *ph_row,
#if LZ_GDN_STATE_2PLANE
                            const int8_t *pl_row,
#endif /* LZ_GDN_STATE_2PLANE */
                            const float *rs_in, float dec,
                            const int16_t *dq, const float *s_dg, float kv,
                            int8_t *oh_row,
#if LZ_GDN_STATE_2PLANE
                            int8_t *ol_row,
#endif /* LZ_GDN_STATE_2PLANE */
                            float *rs_out, int ng, int tier) {
#if !defined(__WATCOMC__)
    lz_p2_blk *blk = p2_blk();     /* Watcom's row functions get their
                                       own via their own p2_blk() call -
                                       see the Watcom Phase 2 branch
                                       below (task #30, 2/N). */
#endif /* !__WATCOMC__ */
    static float sacc[LZ_GDN_MAX_VD / 32];
    static int   shv[LZ_GDN_MAX_VD / 32];
    static int16_t mul[LZ_GDN_MAX_VD / 32][4];
#if !LZ_GDN_STATE_2PLANE
    /* The single-plane build feeds the kernel a zero low plane and
       throws its low output away, rather than carrying a second pair of
       kernels that would be exercised by nothing. H = hi*256 + 0 is
       exactly the single-plane H. */
    static const int8_t zero_lo[32];
    static int8_t sink_lo[32];
#endif /* !LZ_GDN_STATE_2PLANE */
    int gg, j;

    /* Not billed here - gdn_p2_fixed_rows bills this row's cost once,
       above its call to this function. */

    /* PHASE 1 - float only. */
    for (gg = 0; gg < ng; gg++) {
        float c1, c2;
        p2_group_coef(dec, rs_in[gg], kv, s_dg[gg], &c1, &c2);
        p2_group_norm(c1, c2, mul[gg], &sacc[gg]);
    }

    /* PHASE 2 - integer and MMX only, NO x87 anywhere in this loop.
       Changing that is what would make the single emms below wrong. */
#if defined(__WATCOMC__)
    /* Watcom (task #30, 2/N): one call per row, not per group. Before
       this split every kernel below was a #pragma aux body expanded
       inline right here, so the per-group loop cost zero real calls;
       after it, the #pragma aux bodies are inline inside whichever row
       function `tier` selects (src/ops_mmx.c's lz_p2_rows_mmx,
       src/ops_mmx_sse.c's lz_p2_rows_sse, src/ops_sse2.c's
       lz_p2_rows_sse2), so the only new call this split adds is the one
       below - one per row, in place of the zero the old loop had, not
       one per group the way wrapping this loop's body unchanged would
       have cost. pl_stride/ol_stride carry the single-plane fallback
       (the same fixed zero/sink buffer at every group) across the TU
       boundary without teaching those files about LZ_GDN_STATE_2PLANE. */
    {
        const int8_t *plg0;
        int8_t *olg0;
        int pstride;
#if LZ_GDN_STATE_2PLANE
        plg0 = pl_row; olg0 = ol_row; pstride = 32;
#else
        plg0 = zero_lo; olg0 = sink_lo; pstride = 0;
#endif /* LZ_GDN_STATE_2PLANE */
#ifdef LZ_HAVE_P2_SSE2
        if (tier >= 3)
            lz_p2_rows_sse2(ph_row, plg0, pstride, dq, mul,
                            oh_row, olg0, pstride, shv, ng);
        else
#endif /* LZ_HAVE_P2_SSE2 */
#ifdef LZ_HAVE_P2_SSE
        if (tier >= 2)
            lz_p2_rows_sse(ph_row, plg0, pstride, dq, mul,
                           oh_row, olg0, pstride, shv, ng);
        else
#endif /* LZ_HAVE_P2_SSE */
        lz_p2_rows_mmx(ph_row, plg0, pstride, dq, mul,
                       oh_row, olg0, pstride, shv, ng);
    }
#else /* gcc: unchanged per-group loop, extern calls into ops_mmx.c/
         ops_mmx_sse.c/ops_sse2.c exactly as before this task. */
    for (gg = 0; gg < ng; gg++) {
        int32_t amax;
        int sh;
        /* Every constant replicated across the full 16 bytes. MMX reads
           the low 8 and SSE2 reads all 16, so one fill serves both -
           see the block's comment in ops_kernel_p2.h. */
        for (j = 0; j < 8; j += 2) {
            blk->mul[j]     = mul[gg][0];
            blk->mul[j + 1] = mul[gg][1];
        }
        const int8_t *plg =
#if LZ_GDN_STATE_2PLANE
            pl_row + gg * 32;
#else
            zero_lo;
#endif /* LZ_GDN_STATE_2PLANE */
#ifdef LZ_HAVE_P2_SSE2
        if (tier >= 3) lz_p2_mul32_sse2(ph_row + gg * 32, plg, dq + gg * 32, blk);
        else
#endif /* LZ_HAVE_P2_SSE2 */
        /* No SSE1 arm here on purpose: this half has no SSE1 content,
           so tier 2 runs the MMX mul32 unchanged. */
        lz_p2_mul32_mmx(ph_row + gg * 32, plg, dq + gg * 32, blk);
        amax = blk->amax[0];
        for (j = 1; j < 4; j++) if (blk->amax[j] > amax) amax = blk->amax[j];
        sh = p2_shift_of(amax);
        shv[gg] = sh;
        for (j = 0; j < 4; j++) {
            blk->rnd[j] = (sh > 0) ? (int32_t)(1 << (sh - 1)) : 0;
            blk->cnt[j] = (j == 0) ? (int32_t)sh : 0;
        }
        for (j = 0; j < 8; j++) {
            blk->k128[j] = 128;
            blk->kclp[j] = 32641;   /* MMX/SSE2: psubsw then paddsw */
            blk->kmin[j] = -127;    /* SSE1: one pmaxsw does the same */
        }
        {
            int8_t *olg =
#if LZ_GDN_STATE_2PLANE
                ol_row + gg * 32;
#else
                sink_lo;
#endif /* LZ_GDN_STATE_2PLANE */
#ifdef LZ_HAVE_P2_SSE2
            if (tier >= 3) lz_p2_split32_sse2(blk, oh_row + gg * 32, olg);
            else
#endif /* LZ_HAVE_P2_SSE2 */
#ifdef LZ_HAVE_P2_SSE
            if (tier >= 2) lz_p2_split32_sse(blk, oh_row + gg * 32, olg);
            else
#endif /* LZ_HAVE_P2_SSE */
            lz_p2_split32_mmx(blk, oh_row + gg * 32, olg);
        }
    }
#endif /* __WATCOMC__ */
    /* Only the 64-bit tiers leave MMX state behind, and the SSE1 one is
       a 64-bit tier: pmaxsw is an SSE1 instruction operating on an MMX
       REGISTER, so tier 2 needs the emms exactly as tier 1 does. Tier 3
       touches xmm and nothing else, so rule 6 item 6 has nothing to
       clear there, and skipping it is worth a few cycles a row on the
       P4 - the machine the SSE2 tier is for. */
    if (tier < 3) _mm_empty();

    /* PHASE 3 - float again. A group the prologue or the epilogue
       flushed is written as an exact zero here, on top of whatever the
       kernel put there; the kernel ran on a zero multiplier pair in that
       case and produced zeros anyway, but relying on that would be
       relying on an accident. */
    for (gg = 0; gg < ng; gg++)
        if (!p2_group_scale(sacc[gg], shv[gg], rs_out + gg))
            for (j = 0; j < 32; j++) {
                oh_row[gg * 32 + j] = 0;
#if LZ_GDN_STATE_2PLANE
                ol_row[gg * 32 + j] = 0;
#endif /* LZ_GDN_STATE_2PLANE */
            }
}
#endif /* LZ_HAVE_P2_MMX */
#endif /* LZ_GDN_FIXED */

/* WHICH pass-2 body ran - the same join lz_kernel_tier() reports for the
   matmul kernels, for the same reason. "The MMX pass 2 is selected" is
   otherwise unfalsifiable from outside: it is required to be bit
   identical to the scalar one, so no output changes when it silently
   is not compiled in, and that is not hypothetical - the Watcom build
   can carry none of the hand assembly and every bit-identity gate
   stays green.

   "-" means the question does not apply: the float pass 2 is selected,
   or this build has no fixed pass at all. */
const char *lz_gdn_p2_impl(void) {
#if LZ_GDN_FIXED
    if (!lz_gdn_p2_mode()) return "-";
#if defined(LZ_HAVE_P2_SSE2_ASM)
    if (p2_tier() >= 3) return "sse2-asm";
#elif defined(LZ_HAVE_P2_SSE2)
    if (p2_tier() >= 3) return "sse2-intrin";
#endif /* LZ_HAVE_P2_SSE2_ASM */
#if defined(LZ_HAVE_P2_SSE_ASM)
    if (p2_tier() >= 2) return "sse-asm";
#elif defined(LZ_HAVE_P2_SSE)
    if (p2_tier() >= 2) return "sse-intrin";
#endif /* LZ_HAVE_P2_SSE_ASM */
#if defined(LZ_HAVE_P2_MMX_ASM)
    if (p2_tier()) return "mmx-asm";
#elif defined(LZ_HAVE_P2_MMX)
    if (p2_tier()) return "mmx-intrin";
#endif /* LZ_HAVE_P2_MMX_ASM */
    return "ref";
#else
    return "-";
#endif /* LZ_GDN_FIXED */
}

#if LZ_GDN_FIXED
/* The fixed pass 1 for one head, shared by both families - the twin of
   gdn_p2_fixed_rows below, and just as easy to forget to wire into a
   new family.

   `gvec` NULL is the scalar-decay family: q and k go to the coefficient
   table as they are. Non-NULL is the per-channel one, where they are
   premultiplied first - which is the whole trick, since
   ss*gvec*k == ss*(gvec*k) means the SAME table builder and the SAME
   hand-written assembly serve both. Returns k.q, RAW and unweighted in
   both cases: it is not part of the contraction, the epilogue uses it
   against delta, and weighting it would be invisible at kd == 1 and
   wrong everywhere else. */
static float gdn_pass1_fixed_head(const int8_t *sq_in,
#if LZ_GDN_STATE_2PLANE
                                  const int8_t *sq2_in,
#endif /* LZ_GDN_STATE_2PLANE */
                                  const float *ss_in,
                                  const float *q, const float *k,
                                  const float *gvec, int kd, int vd,
                                  float *u, float *w) {
    static int32_t au[LZ_GDN_MAX_VD], aw[LZ_GDN_MAX_VD];
    static float   su[LZ_GDN_MAX_VD / 32], sw[LZ_GDN_MAX_VD / 32];
    static int32_t au2[LZ_GDN_MAX_VD], aw2[LZ_GDN_MAX_VD];
    static float   su2[LZ_GDN_MAX_VD / 32], sw2[LZ_GDN_MAX_VD / 32];
    static float   kg[LZ_GDN_MAX_KD], qg[LZ_GDN_MAX_KD];
#if LZ_GDN_STATE_2PLANE
    static int32_t aul[LZ_GDN_MAX_VD], awl[LZ_GDN_MAX_VD];
#endif /* LZ_GDN_STATE_2PLANE */
    const float *qq = q, *kk_ = k;
    float kq = 0.0f;
    int ng = vd / 32, kk, gg, j;

    if (gvec) {
        for (kk = 0; kk < kd; kk++) {
            float gc = gvec[kk];
            kg[kk] = k[kk] * gc;
            qg[kk] = q[kk] * gc;
            kq += k[kk] * q[kk];
        }
        qq = qg; kk_ = kg;
    } else {
        for (kk = 0; kk < kd; kk++) kq += k[kk] * q[kk];
    }

#if LZ_GDN_STATE_2PLANE
    gdn_pass1_fixed(sq_in, sq2_in, aul, awl, ss_in, qq, kk_, kd, vd,
                    au, aw, su, sw, au2, aw2, su2, sw2);
#else
    gdn_pass1_fixed(sq_in, ss_in, qq, kk_, kd, vd,
                    au, aw, su, sw, au2, aw2, su2, sw2);
#endif /* LZ_GDN_STATE_2PLANE */
    for (gg = 0; gg < ng; gg++)
        for (j = 0; j < 32; j++) {
            int vi = gg * 32 + j;
#if LZ_GDN_STATE_2PLANE
            u[vi] = (lz_i32f(au[vi]) +
                     lz_i32f(aul[vi]) * (1.0f / LZ_GDN_LO_SCALE)) * su[gg];
            w[vi] = (lz_i32f(aw[vi]) +
                     lz_i32f(awl[vi]) * (1.0f / LZ_GDN_LO_SCALE)) * sw[gg];
#elif LZ_GDN_FIXED >= 2
            u[vi] = lz_i32f(au[vi]) * su[gg] + lz_i32f(au2[vi]) * su2[gg];
            w[vi] = lz_i32f(aw[vi]) * sw[gg] + lz_i32f(aw2[vi]) * sw2[gg];
#else
            u[vi] = lz_i32f(au[vi]) * su[gg];
            w[vi] = lz_i32f(aw[vi]) * sw[gg];
#endif /* LZ_GDN_STATE_2PLANE */
        }
    return kq;
}

static void gdn_p2_fixed_rows(const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                              const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                              const float *ss_in, float *ss_out,
                              const float *delta, const float *k,
                              const float *gvec, float gt,
                              int kd, int vd) {
    static int16_t dq[LZ_GDN_MAX_VD];
    static float s_dg[LZ_GDN_MAX_VD / 32];
    int ng = vd / 32, kk, gg, vv;

    /* delta quantized PER 32-GROUP, not once for the whole vd vector.
       A single scale for the whole delta - the shape gdn_build_table
       uses for its coefficients - would be the thing most likely to
       break the error budget, given delta's dynamic range. Measured:
       one scale gives 4.34e-05 state error against an exact float
       recurrence; per group is measurably better. The cost is one extra
       absmax pass over
       32 elements per group and one more multiplier -
       both per group, nothing per element, so the 83% this whole tier
       exists to move is untouched.

       Group boundaries match the state's own quantization groups, which
       is what lets c2 stay a single scalar per (row, group) call. */
    for (gg = 0; gg < ng; gg++) {
        float amd = 0.0f, inv_d;
        for (vv = gg * 32; vv < gg * 32 + 32; vv++) {
            float a = (delta[vv] < 0.0f) ? -delta[vv] : delta[vv];
            if (a > amd) amd = a;
        }
        /* Same floor as everywhere else: a delta group below this is
           numerically zero, and letting its scale reach the subnormal
           range is how x87 and SSE stop agreeing (ops.h's
           LZ_Q8_MIN_SCALE has the mechanism). */
        if (!(amd * LZ_Q15_INV >= LZ_Q8_MIN_SCALE_F)) {
            s_dg[gg] = 1.0f;
            for (vv = gg * 32; vv < gg * 32 + 32; vv++) dq[vv] = 0;
            continue;
        }
        s_dg[gg] = amd * LZ_Q15_INV;
        inv_d = 32767.0f / amd;
        for (vv = gg * 32; vv < gg * 32 + 32; vv++) {
            int t = q8_round(delta[vv] * inv_d);
            if (t >  32767) t =  32767;
            if (t < -32767) t = -32767;
            dq[vv] = (int16_t)t;
        }
    }

    for (kk = 0; kk < kd; kk++) {
        const int8_t *row_in = sq_in + (size_t)kk * vd;
        int8_t *row_out = sq_out + (size_t)kk * vd;
        const float *rs_in = ss_in + (size_t)kk * ng;
        float *rs_out = ss_out + (size_t)kk * ng;
        float dec = gvec ? gvec[kk] : gt;    /* the one family difference */
#if LZ_GDN_STATE_2PLANE
        const int8_t *row_lo_in = sq2_in + (size_t)kk * vd;
        int8_t *row_lo_out = sq2_out + (size_t)kk * vd;
#endif /* LZ_GDN_STATE_2PLANE */
        /* Both SIMD (gdn_p2_row_simd) and scalar (gg loop below) row
           paths cost the same per group. Billed once, above the fork -
           billing only inside gdn_p2_row_simd would read zero on
           --kernel ref, where p2_tier()==0 takes the scalar path only. */
        LZ_FCX(LZ_FC_RECUR, 7 * ng, 2 * ng, 0, 0, 0);  /* coef 3 mul; norm 2 mul + 2 q8_round(add); scale 2 mul */
#ifdef LZ_HAVE_P2_MMX
        if (p2_tier()) {
            gdn_p2_row_simd(row_in,
#if LZ_GDN_STATE_2PLANE
                            row_lo_in,
#endif /* LZ_GDN_STATE_2PLANE */
                            rs_in, dec, dq, s_dg, k[kk], row_out,
#if LZ_GDN_STATE_2PLANE
                            row_lo_out,
#endif /* LZ_GDN_STATE_2PLANE */
                            rs_out, ng, p2_tier());
            continue;
        }
#endif /* LZ_HAVE_P2_MMX */
        for (gg = 0; gg < ng; gg++) {
            float c1, c2;
            p2_group_coef(dec, rs_in[gg], k[kk], s_dg[gg], &c1, &c2);
            gdn_p2_group_fixed(row_in + gg * 32,
#if LZ_GDN_STATE_2PLANE
                               row_lo_in + gg * 32,
#endif /* LZ_GDN_STATE_2PLANE */
                               c1, dq + gg * 32, c2,
                               row_out + gg * 32,
#if LZ_GDN_STATE_2PLANE
                               row_lo_out + gg * 32,
#endif /* LZ_GDN_STATE_2PLANE */
                               rs_out + gg);
        }
    }
}
/* Both of the above read the coefficient table, the pass-1 kernel and
   the per-group helpers, all of which live inside LZ_GDN_FIXED. Their
   only call sites are already guarded the same way, so guarding the
   definitions costs nothing and keeps the capacity-fallback build
   compilable. */
#endif /* LZ_GDN_FIXED */

/* ONE recurrence, two families. lz_gdn_step and lz_kda_step are now
   both thin wrappers over this.

   The identity that makes it exact - not approximate, exact:

       lz_kda_step(...) == gdn_step_impl(..., gvec, gt = 1.0f)

   because KDA's per-channel decay is already folded into pass 1 (its
   coefficients are gvec[kk]*k[kk]), so the only place GDN's scalar gt
   still appears is the epilogue's `gt*u` and `gt*w` and pass 2's
   `rs*gt`. Multiplying by 1.0f is exact for every finite value, so
   passing gt = 1.0f reproduces KDA's arithmetic bit for bit rather than
   merely closely.

   Why bother: a tier added to one family and forgotten in the other is
   SILENT - it just does nothing on that architecture, both settings
   agree, and it reads as "this tier is not worth much". KunMoe is
   0 gdn / 6 kda / 2 full, so the forgotten half is every time the only
   half that matters.

   `gvec` NULL selects the scalar-decay family. The gate is bit-exactness
   against a 10-hash baseline (2 models x --gdn x --gdn-p2); nothing may
   move. */
static void gdn_step_impl(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 const float *gvec, float gt, float beta, int kd, int vd) {
    float u[LZ_GDN_MAX_VD];      /* holds S^T k first, then reused in place as delta */
    float w[LZ_GDN_MAX_VD];      /* S^T q */
    float kq = 0.0f;
    int kk, gg, j, vv, ng;

    if (vd <= 0 || (vd % 32) != 0 || vd > LZ_GDN_MAX_VD || kd <= 0) {
        for (vv = 0; vv < vd; vv++) out[vv] = 0.0f;   /* defense: no silent miscalc */
        return;
    }
    ng = vd / 32;

#if LZ_GDN_STATE_F32
    /* NOTE: this shadow is keyed by the READ pointer's own address,
       1:1 with (layer, head) ONLY when every call for a given head
       always passes the SAME sq_in - true for ordinary decode (in==out,
       unchanged) but NOT during a speculative verify batch, where sq_in
       advances through the ring slot by slot. Enabling LZ_GDN_STATE_F32
       together with --spec K would silently look up the wrong shadow
       entry per ring step. Flagged rather than fixed, since fixing it
       means re-keying the shadow map by (layer,head) instead of by
       pointer, a change to a scaffold this feature does not otherwise
       touch. */
    {
        float *sf = gdn_shadow(sq_in, kd * vd);
        if (sf) {
            /* Same recurrence, exact float state: no quantization anywhere
               in the loop, so what is left is only float rounding. */
            for (vv = 0; vv < vd; vv++) { u[vv] = 0.0f; w[vv] = 0.0f; }
            for (kk = 0; kk < kd; kk++) {
                const float *row = sf + (size_t)kk * vd;
                float kc = k[kk], qc = q[kk];
                kq += kc * qc;
                for (vv = 0; vv < vd; vv++) {
                    u[vv] += row[vv] * kc;
                    w[vv] += row[vv] * qc;
                }
            }
            for (vv = 0; vv < vd; vv++) {
                u[vv] = (v[vv] - gt * u[vv]) * beta;
                out[vv] = gt * w[vv] + u[vv] * kq;
            }
            for (kk = 0; kk < kd; kk++) {
                float *row = sf + (size_t)kk * vd;
                float kc = k[kk];
                for (vv = 0; vv < vd; vv++)
                    row[vv] = row[vv] * gt + kc * u[vv];
            }
            return;
        }
    }
#endif /* LZ_GDN_STATE_F32 */

    /* Pass 1: state read once, gathering both contractions. Always reads
       sq_in/sq2_in/ss_in - the "_in" side, whether or not it aliases
       "_out" (see this function's own header comment on the ring). */
#if LZ_GDN_FIXED
    /* g_gdn_fixed is the runtime knob; the kd bound is a hard capacity
       limit (the coefficient table is int16_t[LZ_GDN_MAX_KD/2][8]).
       Either one falling through lands on the same float path below,
       which is exactly why that path is the baseline worth being able
       to ask for. */
    if (lz_gdn_mode() && kd <= LZ_GDN_MAX_KD) {
        kq = gdn_pass1_fixed_head(sq_in,
#if LZ_GDN_STATE_2PLANE
                                  sq2_in,
#endif /* LZ_GDN_STATE_2PLANE */
                                  ss_in, q, k, gvec, kd, vd, u, w);
    } else
#endif /* LZ_GDN_FIXED */
    {
    /* Float fallback: only 2 mul + 2 add per element - scale and k/q
       components are pre-multiplied. */
    for (vv = 0; vv < vd; vv++) { u[vv] = 0.0f; w[vv] = 0.0f; }
    for (kk = 0; kk < kd; kk++) {
        const int8_t *row = sq_in + (size_t)kk * vd;
        const float *rs = ss_in + (size_t)kk * ng;
        float kc = k[kk], qc = q[kk];
        /* The one family difference in this loop: KDA's contraction is
           against gvec-weighted k and q, GDN's against them raw. With
           gvec NULL the expressions below are character-for-character
           what the GDN-only version had. */
        float kcw = gvec ? kc * gvec[kk] : kc;
        float qcw = gvec ? qc * gvec[kk] : qc;
        kq += kc * qc;                     /* RAW dot, both families */
        for (gg = 0; gg < ng; gg++) {
            const int8_t *p = row + gg * 32;
            float *uu = u + gg * 32, *ww = w + gg * 32;
            float sck = rs[gg] * kcw, scq = rs[gg] * qcw;
#if LZ_GDN_STATE_2PLANE
            const int8_t *pl = sq2_in + (size_t)kk * vd + gg * 32;
#endif /* LZ_GDN_STATE_2PLANE */
            for (j = 0; j < 32; j++) {
#if LZ_GDN_STATE_2PLANE
                float e = (float)p[j] + (float)pl[j] * (1.0f / LZ_GDN_LO_SCALE);
#else
                float e = (float)p[j];
#endif /* LZ_GDN_STATE_2PLANE */
                uu[j] += e * sck;
                ww[j] += e * scq;
            }
        }
    }
    }

    /* Scalar epilogue: output needs no second state read.
       S_new = gt·S + k⊗δ  =>  S_new^T q = gt·(S^T q) + δ·(k·q) */
    for (vv = 0; vv < vd; vv++) {
        u[vv] = (v[vv] - gt * u[vv]) * beta;      /* u becomes delta */
        out[vv] = gt * w[vv] + u[vv] * kq;
    }

    /* Pass 2: read the OLD state from sq_in/ss_in (same as pass 1 - the
       decay/update formula below needs the pre-step value), write the
       NEW state to sq_out/ss_out. When in==out (ordinary decode) this
       is bit-identical to the single-pointer form: every read for row
       kk completes before that row's one write, so self-aliasing is
       safe and the split is just an option, not a requirement (see this
       function's own header comment). New values first land
       in one stack row buffer (vd entries, stays in L1), then the whole
       row is quantized back at once - reusing lz_quantize_q8, the same
       rounding semantics as everywhere else. Called per ROW, not per
       32-group: the 0.8B has 18 layers x 16 heads x 128 rows per
       token; per-group calls would be 147K function calls whose
       overhead alone eats the pass savings. */
#if LZ_GDN_FIXED
    if (lz_gdn_p2_mode()) {
        gdn_p2_fixed_rows(sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                          sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                          ss_in, ss_out, u, k, gvec, gt, kd, vd);
        /* docs/x87-gcc-reg-stack-leak.md: the row/group loop inside
           gdn_p2_fixed_rows is where the leak was bisected to. This is
           the finer of the two checkpoints; forward.c's per-layer one
           is the coarser one that first surfaced it. */
        LZ_X87_STACK_CHECK("gdn-p2-fixed-rows-return");
        return;
    }
#endif /* LZ_GDN_FIXED */

    for (kk = 0; kk < kd; kk++) {
        float buf[LZ_GDN_MAX_VD];
        const int8_t *row_in = sq_in + (size_t)kk * vd;
        int8_t *row_out = sq_out + (size_t)kk * vd;
        const float *rs_in = ss_in + (size_t)kk * ng;
        float *rs_out = ss_out + (size_t)kk * ng;
        float kc = k[kk];
        float dec = gvec ? gvec[kk] : gt;   /* the one family difference */
#if LZ_GDN_STATE_2PLANE
        const int8_t *row_lo_in = sq2_in + (size_t)kk * vd;
        int8_t *row_lo_out = sq2_out + (size_t)kk * vd;
#endif /* LZ_GDN_STATE_2PLANE */
        for (gg = 0; gg < ng; gg++) {
            const int8_t *p = row_in + gg * 32;
            const float *dd = u + gg * 32;
            float *bb = buf + gg * 32;
            float scg = rs_in[gg] * dec;
#if LZ_GDN_STATE_2PLANE
            const int8_t *pl = row_lo_in + gg * 32;
            for (j = 0; j < 32; j++)
                bb[j] = ((float)p[j] +
                         (float)pl[j] * (1.0f / LZ_GDN_LO_SCALE)) * scg +
                        kc * dd[j];
#else
            for (j = 0; j < 32; j++)
                bb[j] = (float)p[j] * scg + kc * dd[j];
#endif /* LZ_GDN_STATE_2PLANE */
        }
#if LZ_GDN_STATE_2PLANE
        /* Reads zero when --recur-write is fixed: the work moved to
           gdn_p2_fixed_rows, it did not vanish. */
        lz_gdn_quantize_2p(buf, vd, 32, row_out, row_lo_out, rs_out,
                           LZ_FC_RECUR);
#else
        /* Billed here, not inside lz_quantize_q8 (many other callers,
           out of scope for this task): its three branches - SIMD,
           scalar-multiply, scalar-divide - are all mul-or-div + q8_round,
           2/element, no residual plane to round a second time. Exact,
           not an approximation. The per-element op is billed as mul
           (the common "fast" branch); the rare divide branch is folded
           in here too, same flat-bill convention as the rest of this
           file. */
        LZ_FCX(LZ_FC_RECUR, vd + ng, vd, ng, 0, 0);  /* mul-or-div, q8_round(add); 2/group q8_group_scale (mul, div) */
        lz_quantize_q8(buf, vd, 32, row_out, rs_out);
#endif /* LZ_GDN_STATE_2PLANE */
    }
}

void lz_gdn_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 float gt, float beta, int kd, int vd) {
    gdn_step_impl(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                  sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                  ss_in, ss_out, q, k, v, NULL, gt, beta, kd, vd);
}

/* gt = 1.0f is not a placeholder. KDA's decay is already inside pass 1's
   coefficients, so the only surviving uses of the scalar gt are `gt*u`,
   `gt*w` and `rs*gt` - and multiplying by 1.0f is exact. This call
   therefore reproduces the hand-written KDA recurrence bit for bit, not
   approximately; the 10-hash baseline confirms it. */
void lz_kda_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 const float *gvec, float beta, int kd, int vd) {
    gdn_step_impl(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                  sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                  ss_in, ss_out, q, k, v, gvec, 1.0f, beta, kd, vd);
}

void lz_causal_conv1d_step(float *o, const float *x,
                               const float *state_in, float *state_out,
                               const float *w, int n_ch, int k) {
    int hist = k - 1;
    int c, j;
    /* k multiplies and k adds per channel - the first add is against the
       0.0f initializer, which -ffp-contract=off without -ffast-math
       cannot fold away. The old census counted the taps and not the
       adds. silu bills itself. */
    LZ_FCX(LZ_FC_CONV, (long)n_ch * k, (long)n_ch * k, 0, 0, 0);
    for (c = 0; c < n_ch; c++) {
        const float *st_in = state_in + (size_t)c * hist;
        float *st_out = state_out + (size_t)c * hist;
        const float *wr = w + (size_t)c * k;
        float sum = 0.0f;
        for (j = 0; j < hist; j++) sum += wr[j] * st_in[j];
        sum += wr[hist] * x[c];
        o[c] = lz_silu(sum);
        /* shift left by one (reading the OLD history from st_in),
           append the current input at the end (writing to st_out) -
           safe when st_in==st_out too, same forward-order argument as
           the in-place case: iteration j writes st_out[j] and reads
           st_in[j+1], and no earlier iteration ever wrote index j+1
           (only indices < j). */
        for (j = 0; j + 1 < hist; j++) st_out[j] = st_in[j + 1];
        if (hist > 0) st_out[hist - 1] = x[c];
    }
}

#if LZ_CONV_FIXED
/* Fixed tier. The float path above is untouched and stays the control
   arm; this is a separate body, selected by lz_conv_mode().
 *
 * Where the win comes from, and it is not the multiply-adds: keeping the
 * HISTORY in fixed point means one conversion per new sample instead of
 * one per tap. Probe (.prof/wconv.c): 8 f32 operations per channel-step
 * down to 3, drift 4.8e-5 against a 3.9e-3 noise floor. The same shape
 * pass-2's fixed tier uses - its own comment says it does not accelerate
 * the write-back quantize, it deletes it.
 *
 * `sh_in`/`sh_out` are int16 history at exponent `es`, `mw` the int16
 * taps at per-channel exponent `ew`. Both were normalized by
 * lz_conv_norm_pow2 against lz_conv_accum_bound(k), which is what keeps
 * `acc` inside int32 - see that function for the derivation.
 *
 * The x87/SSE agreement argument is the same as the float path's: the
 * only float arithmetic left is one multiply in and one multiply out,
 * both by an exact power of two, so neither rounds. */
void lz_causal_conv1d_step_fixed(float *o, const float *x,
                                 const short *sh_in, short *sh_out,
                                 const short *mw, const float *oscale,
                                 float in_scale, int bound,
                                 int n_ch, int k) {
    int hist = k - 1;
    int c, j;
    /* One multiply in, one convert and one multiply out, whatever k is.
       That independence from k is the whole point of the tier. The
       input-side round-to-int (add 0.5, convert) is not billed here,
       same convention as the other quantizers' rounding steps. */
    LZ_FCX(LZ_FC_CONV, (long)n_ch * 2, 0, 0, (long)n_ch, 0);
    for (c = 0; c < n_ch; c++) {
        const short *si = sh_in + (size_t)c * hist;
        short *so = sh_out + (size_t)c * hist;
        const short *wr = mw + (size_t)c * k;
        int acc = 0, q;
        float t = x[c] * in_scale;
        q = (int)(t < 0.0f ? t - 0.5f : t + 0.5f);
        if (q > bound) q = bound;
        if (q < -bound) q = -bound;
        for (j = 0; j < hist; j++) acc += (int)wr[j] * (int)si[j];
        acc += (int)wr[hist] * q;
        o[c] = lz_silu(lz_i32f(acc) * oscale[c]);
        for (j = 0; j + 1 < hist; j++) so[j] = si[j + 1];
        if (hist > 0) so[hist - 1] = (short)q;
    }
}

/* Build the per-channel output scale 2^-(ew[c]+es) and the input scale
   2^es, once, so the hot loop has no shift in it.
 *
 * NOT `1 << (ew + es)`: lz_conv_norm_pow2 caps its exponent at 30, so
 * the sum reaches 43 and that shift is undefined on a 32-bit int. The
 * first version of the kernel did exactly this and produced max drift
 * 4.2e8 on a uniform state while looking correct on a decayed one -
 * caught only because .prof/wconv2.c drives the shipped function rather
 * than a copy of it. ldexp walks the exponent field instead, and runs
 * once per channel at setup rather than per token. */
void lz_conv_build_scales(float *oscale, float *in_scale,
                          const signed char *ew, int es, int n_ch) {
    int c;
    *in_scale = pow2f(es);
    for (c = 0; c < n_ch; c++)
        oscale[c] = pow2f(-((int)ew[c] + es));
}
#endif /* LZ_CONV_FIXED */

void lz_moe_route(const float *logits, const float *bias, int n_experts,
                  int top_k, int sigmoid, int renormalize, float tau,
                  int *idx_out, float *w_out) {
    /* static, not stack: LZ_MAX_EXPERTS is 128, and rule six clause 4
       ("large buffers stay off the stack") is written for a Win98
       target with a small default stack - four 512-byte arrays per
       call is exactly the kind of thing that rule exists for. */
    static float score[LZ_MAX_EXPERTS];
    static float tscore[LZ_MAX_EXPERTS];
    static float key[LZ_MAX_EXPERTS];
    /* Sentinel for "already selected" in the top-k scan below - far
       enough from any real score/bias that excess precision could not
       change which elements it excludes, but not exactly representable
       in float32 either, so it still gets the same treatment as every
       other constant on this file's list (LZ_EXP_LOG2E32's comment in
       lz_exp has the mechanism). */
    static const float LZ_MOE_SENTINEL = -1e30f;
    int i, t;
    int tempered = (tau != 1.0f);
    float wsum = 0.0f;

    for (t = 0; t < top_k; t++) { idx_out[t] = -1; w_out[t] = 0.0f; }
    if (n_experts <= 0 || n_experts > LZ_MAX_EXPERTS || top_k <= 0) return;
    if (!(tau > 0.0f)) tempered = 0;   /* also catches NaN, which compares false */

    if (sigmoid) {
        for (i = 0; i < n_experts; i++) score[i] = lz_sigmoid(logits[i]);
    } else {
        /* softmax: max-subtracted for stability, sum via lz_exp for the
           same cross-platform determinism the rest of the engine uses -
           see lz_exp's contract in ops.h. */
        float m = logits[0], s = 0.0f;
        for (i = 1; i < n_experts; i++) if (logits[i] > m) m = logits[i];
        for (i = 0; i < n_experts; i++) {
            score[i] = lz_exp(logits[i] - m);
            s += score[i];
        }
        for (i = 0; i < n_experts; i++) score[i] /= s;   /* s is a runtime value, not a constant - rule six item 1 */
    }

    /* Router temperature. SELECTION stays on the untempered scores - see
       the header for why that split is the whole point of the knob - so
       this is a second array rather than a rescale of the first.

       Divided by tau rather than multiplied by a precomputed 1/tau: tau
       is a runtime value, so the division is a real FDIV on both
       compilers and rule six item 1 is satisfied either way, but the
       reference this is calibrated against divides, and matching it is
       worth more here than saving a few
       cycles in a per-layer, n_experts-long loop.

       NO FLUSH-TO-ZERO HERE, and that is a measurement rather than an
       omission. Dividing by a small tau drives l/tau deep negative,
       which is exactly the operand class rule two clause (c) forbids.
       No floor is needed: lz_exp already returns a hard
       0.0f below -87.3f, one ULP-ish above FLT_MIN, so lz_sigmoid over
       the whole negative range produces 0 subnormals in 200001 samples
       (smallest nonzero 0084c390, exponent field 1). The softmax branch
       cannot reach one either - its post-division scan is also 0 -
       because s is only large when many experts sit near the max, and
       that is the same condition that keeps their exponentials away
       from the floor.
       So the guard against subnormals is lz_exp's own range clamp, one
       call down. That is the dependency, and t_moe_tau pins it. */
    if (tempered) {
        if (sigmoid) {
            for (i = 0; i < n_experts; i++)
                tscore[i] = lz_sigmoid(logits[i] / tau);
        } else {
            float m = logits[0], s = 0.0f;
            for (i = 1; i < n_experts; i++) if (logits[i] > m) m = logits[i];
            for (i = 0; i < n_experts; i++) {
                tscore[i] = lz_exp((logits[i] - m) / tau);
                s += tscore[i];
            }
            /* s >= 1 always: the max element contributes exp(0). */
            for (i = 0; i < n_experts; i++) tscore[i] /= s;
        }
    }

    for (i = 0; i < n_experts; i++)
        key[i] = score[i] + (bias ? bias[i] : 0.0f);

    /* top-k by (score + bias); the GATHERED weight is the plain score
       (see ops.h - the bias enters selection only). O(top_k * n_experts)
       selection, fine at this scale (n_experts is a handful to a few
       dozen in every reference config, and this is not a hot loop next
       to the matmuls around it). */
    for (t = 0; t < top_k && t < n_experts; t++) {
        int best = -1;
        float bk = 0.0f;
        for (i = 0; i < n_experts; i++) {
            if (key[i] <= LZ_MOE_SENTINEL) continue;      /* already selected */
            if (best < 0 || key[i] > bk) { best = i; bk = key[i]; }
        }
        if (best < 0) break;
        idx_out[t] = best;
        w_out[t] = tempered ? tscore[best] : score[best];
        wsum += w_out[t];
        key[best] = LZ_MOE_SENTINEL;              /* remove from further rounds */
    }
    /* Every selected weight underflowed to zero - lz_exp's clamp, not a
       gradual loss of precision. Only reachable in the tempered sigmoid
       path (the softmax one normalizes by a sum that is at least 1), and
       only at small tau with logits far below zero, which is exactly the
       configuration this model has: its router logits sit around -5 and
       --moe-tau goes down to 0.1, so l/tau reaches -50 inside the CLI's
       own bounds and a slightly colder router reaches the clamp. A real
       case, not a defensive branch.
       One-hot on the top-1 selection is the tau -> 0 limit, i.e. the
       answer the arithmetic heads for as it underflows.
       Leaving the zeros instead would silently drop the whole routed
       branch for that token, which no caller checks for. */
    if (tempered && wsum <= 0.0f && idx_out[0] >= 0) {
        w_out[0] = 1.0f;
        wsum = 1.0f;
    }
    if (renormalize && wsum > 0.0f) {
        for (t = 0; t < top_k; t++)
            if (idx_out[t] >= 0) w_out[t] /= wsum; /* wsum is a runtime value */
    }
}

void lz_moe_hits_add(unsigned char *hits, int cap, const int *idx, int k) {
    int t;
    if (!hits || !idx) return;
    for (t = 0; t < k; t++) {
        /* Both skips are real cases, not defensive padding: lz_moe_route
           writes -1 into a slot it could not fill, and cap is the width
           of the array. See the header for why the second one is the
           caller's problem to report rather than this function's to
           hide. */
        if (idx[t] < 0 || idx[t] >= cap) continue;
        if (hits[idx[t]] < 255) hits[idx[t]]++;
    }
}

/* ---- kernel coverage matrix ------------------------------------------

   At the end of the file and outside every #if, because it must exist on
   EVERY target: cli_main.c calls it unconditionally, and the first
   version sat inside `#if defined(__MMX__)` and broke the ARM
   cross-build at link. That build is the one nobody runs, which is
   exactly why it found this.

   The 128-group column is DECLARED, not detected: whether a row kernel
   calls a group-of-four kernel is a compile-time fact inside a static
   function and cannot be read from outside. Both halves have to be
   edited together, and tests/test_kernel_matrix.py compares this against
   the source - the previous version of this map lived in a document and
   drifted for months. */
#if defined(__WATCOMC__)
#define LZ_G128_Q8  1
#define LZ_G128_Q41 1
#define LZ_G128_Q61 1
#define LZ_G128_Q16 1
#else
/* The gcc group kernels (dot128_*_mmx) exist and are wired into the row
   functions behind the LZ_G128_GCC knob (default 1, ops_kernel_shared.h)
   - see that knob's comment for the A/B rationale. This column tracks
   the wiring: LZ_G128_GCC=0 reverts it to '.' and the row functions to
   the per-32 path. The claim is "wired in", not "measured faster" -
   whether the group path wins on any machine is unmeasured. */
#define LZ_G128_Q8  LZ_G128_GCC
#define LZ_G128_Q41 LZ_G128_GCC
#define LZ_G128_Q61 LZ_G128_GCC
#define LZ_G128_Q16 LZ_G128_GCC
#endif /* __WATCOMC__ */
/* T2 on both compilers, and this one IS "nothing to offer". Derived by
   comparing against q6_1, which unpacks a 2-bit plane the same way and
   does have one: q6_1's group kernel hoists two mask constants (0x0F and
   0x03) plus a zero register across its four sub-blocks. T2's unpack is
   a zero register and three immediate shifts, and hoisting the zero
   turns `pxor mm2,mm2` into `movq mm2,mm7`, one instruction for one.
   Each sub-block carries its own 2-bit plane, its own accumulator and
   its own fold, so a T2 group kernel would emit the same instruction
   sequence as four separate calls. */
#define LZ_G128_T2  0

const char *lz_kernel_matrix(void) {
    static char buf[512];
    /* Q16's table is the one row inside the MMX block; the other four
       are unguarded and come out all-NULL on a target with no SIMD,
       which is the honest answer there. */
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN)
    const lz_rowfn *q16 = LZ_ROW_Q16;
#else
    const lz_rowfn *q16 = 0;
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */
    struct { const char *name; const lz_rowfn *tab; int g128; } rows[5];
    int i, j, p = 0;
    rows[0].name = "q8_0";  rows[0].tab = LZ_ROW_Q8;  rows[0].g128 = LZ_G128_Q8;
    rows[1].name = "q4_1";  rows[1].tab = LZ_ROW_Q41; rows[1].g128 = LZ_G128_Q41;
    rows[2].name = "q6_1";  rows[2].tab = LZ_ROW_Q61; rows[2].g128 = LZ_G128_Q61;
    rows[3].name = "q16_0"; rows[3].tab = q16;        rows[3].g128 = LZ_G128_Q16;
    rows[4].name = "t2";    rows[4].tab = LZ_ROW_T2;  rows[4].g128 = LZ_G128_T2;
    for (i = 0; i < 5; i++) {
        p += sprintf(buf + p, "%-6s", rows[i].name);
        for (j = 0; j < LZ_ROW_N; j++)
            buf[p++] = (rows[i].tab && rows[i].tab[j]) ? 'x' : '.';
        buf[p++] = ' ';
        buf[p++] = rows[i].g128 ? 'G' : '.';
        buf[p++] = '\n';
    }
    buf[p] = '\0';
    return buf;
}

/* lz_tier_pins and lz_tier_report are in ops_sched.c. */

/* ---- x87 stack-empty gate (debug only) --------------------------------

   See the header comment on LZ_X87_STACK_CHECK (ops.h) for what this is
   and its own limits. Record-only: the first dirty call site wins and
   stays latched for the run, same convention as lz_debug_attn_skip and
   friends - a counter that resets per call could not distinguish "never
   dirty" from "dirty once, then this call site happened to be clean".

   Real implementation only where fnstenv means what this file needs it
   to mean: gcc-family (not Watcom, whose inline asm is a different
   syntax entirely and whose 32-bit x87 output is already known correct
   - this gate exists for gcc's -O2 -mfpmath=387 bug specifically) on an
   x86 target (not the ARM cross-build, which has no x87 unit at all).
   Everywhere else the gate compiles to a stub that never reports dirty -
   stated here rather than left to be discovered, because a silent no-op
   gate is indistinguishable from a passing one otherwise. */
#if defined(LZ_X87_STACK_GATE)
#if !defined(__WATCOMC__) && (defined(__i386__) || defined(__x86_64__))
static const char *g_x87_dirty_where = NULL;
static unsigned g_x87_dirty_tag = 0;

void lz_x87_stack_check(const char *where) {
    unsigned short env[14];
    __asm__ volatile ("fnstenv %0" : "=m" (env));
    if (env[4] != 0xFFFFu && !g_x87_dirty_where) {
        g_x87_dirty_where = where;
        g_x87_dirty_tag = (unsigned)env[4];
    }
}

int lz_x87_stack_dirty(const char **where, unsigned *tag) {
    if (!g_x87_dirty_where) return 0;
    if (where) *where = g_x87_dirty_where;
    if (tag) *tag = g_x87_dirty_tag;
    return 1;
}
#else
/* Stub: LZ_X87_STACK_GATE was requested on a target this check does not
   cover. Compiles and links, always reports clean - NOT because the
   target is known clean, but because there is nothing here to ask. */
void lz_x87_stack_check(const char *where) { (void)where; }
int  lz_x87_stack_dirty(const char **where, unsigned *tag) {
    (void)where; (void)tag; return 0;
}
#endif /* !__WATCOMC__ && (__i386__ || __x86_64__) */
#endif /* LZ_X87_STACK_GATE */
