#include "lz_int.h"   /* lz_i64/lz_u64 and the width types:
                         Visual C++ 4.0 has neither <stdint.h> nor
                         `long long`, and 236 sites deciding that for
                         themselves is the shape one-authority forbids */
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
                           %mm-touching kernels are defined there. Unconditional:
                           the macros and declarations inside are themselves
                           guarded on LZ_MMX_TU, a build flag (-D, from the
                           Makefile) rather than on __MMX__, which this file
                           must never be compiled with. */
#include "ops_sse.h"    /* the SSE1-on-%xmm half of src/ops_mmx_sse.c. A
                           separate include from ops_mmx.h on purpose: those
                           kernels need an SSE1 machine, and an include line
                           saying "mmx" would not say so. */
#include "ops_sse2.h"   /* declarations for src/ops_sse2.c's functions - the
                           %xmm-touching kernels are defined there. Same
                           LZ_SSE2_TU/__SSE2__ contract, one ISA tier up. */
#include "ops_kernel_shared.h" /* lz_i32f, LZ_WSUM_CHUNK, LZ_SLOT_NEXT,
                           gdn_tail_row, p2_shift_of - shared with the
                           suffixed TUs; included this early because lz_i32f's
                           first use is well before any of the ops_kernel_*.h
                           #includes below. */

/* LZ_GDN_FIXED and its rationale are in ops.h. */

#include "ops_quant.h"   /* pow2f, q8_round, q8_amax, sigmoid_q15,
                           sigmoid_q15_t, lz_sig_q15_fold,
                           lz_sig_q15_t_offset, norm_ss_fixed,
                           lz_exp_float_body, lz_rsqrt_float_body,
                           lz_exp_fixed, lz_rsqrt_fixed,
                           LZ_Q8_MIN_SCALE_F, LZ_Q8_INV127, LZ_Q15_INV */
#include "ops_sched.h"   /* lz_q8r_tier, q8r_have_simd */
#include "ops_matmul.h"  /* matmul impls, LZ_ROW_* tables, epilogue fns */
#include "ops_t2_arm.h"  /* LZ_T2_ARM_ASM_EXTERN */
#include "ops_arm.h"     /* LZ_ARM_ASM_EXTERN */

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
    /* q8_amax, not a hand-written scan: two __aeabi_fcmpXX per ELEMENT
       otherwise, and this was the fifth copy of the same loop. See
       epi_pack_act for why the integer-domain form is exact. */
    union { float f; uint32_t u; } ab;
    float amax;
    int i, e = 0;
    ab.u = q8_amax(v, n);
    amax = ab.f;
    if (amax == 0.0f) {
        for (i = 0; i < n; i++) out[i] = 0;
        return 0;
    }
    while (e < 30 && amax * (float)(1 << (e + 1)) < (float)bound) e++;
    for (i = 0; i < n; i++) {
        float t = v[i] * (float)(1 << e);
        int q = q8_round(t);
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
    "softmax", "exp", "sigmoid", "softplus", "conv1d", "norms", "rsqrt",
    "recurrence write", "quantize"
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
       is a sign flip and bills zero, so both branches are 1 add + 1 div.
       The site billed 2 adds to hold an older total; corrected with the
       three in src/ops_norm.c, and like two of those it is worth 0 in
       the shipping census - lz_sig_mode() is on by default, so this
       branch does not run. Measured, not assumed. */
    LZ_FCX(LZ_FC_SIGMOID, 0, 1, 1, 0, 0);
    if (x >= 0.0f) {
        float z = lz_exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = lz_exp(x);
        return z / (1.0f + z);
    }
}

/* Integer-input lz_sigmoid: x == v * 2^-e (int-pipeline milestone 5).
   One convert and one multiply, against the three multiplies, add and
   convert the float entry needs to reach the same table - the producer
   has already delivered the coordinate in integers, so the derivation
   sigmoid_q15 does per call is not repeated, it is gone.

   The float tier is not this function's table at all (lz_exp-based, a
   different value), so it cannot be served in integers; callers gate on
   lz_sig_mode() and never reach here with the tier off. Reconstructing
   is what that unreachable case does anyway rather than returning a
   value from the wrong tier. */
float lz_sigmoid_i(int32_t v, int e) {
    if (!lz_sig_mode()) return lz_sigmoid((float)v * pow2f(-e));
    LZ_FCX(LZ_FC_SIGMOID, 1, 0, 0, 1, 0);
    return (float)sigmoid_q15_i(v, e) * (1.0f / 32768.0f);
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
#if defined(LZ_X87_FLOAT_CW)   /* the detect is in ops.h, once */
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

/* log(1+e^x). The body is lz_softplus_poly (ops_quant.c) - table+poly,
   all float, under lz_scalar_mode() like every other transcendental.

   Below is the arm it replaced and the control every number was
   measured against: two libm DOUBLE calls. On a target with an FPU
   those need the x87 control word widened back to 64 bits around them,
   because PC=24 is in effect for the whole lz_forward pass and libm's
   double routines read wrong under it; on a target with NO FPU there is
   no control word and the cost is the two soft-float double
   transcendentals themselves - 3603 guest instructions a call against
   the shipped body's 1389, measured under QEMU in
   .prof/softplus_weight.c, and invisible to the f32 census, which
   counts neither libm nor doubles.

   LZ_EXACT_MATH takes this arm too, for the reason lz_exp and lz_rsqrt
   do: that build exists to answer with the library. */
float lz_softplus(float x) {
#if LZ_SOFTPLUS_POLY && !defined(LZ_EXACT_MATH)
    return lz_softplus_poly(x);
#else
    /* Stable form of log(1+exp(x)): for large x it degenerates to x,
       avoiding exp overflow. In this model a + dt_bias can take fairly
       large positive values; the naive form overflows to inf. */
    if (x > 20.0f) return x;
#if defined(__WATCOMC__)
    {
        unsigned save = _control87(0, 0);
        float r;
        _control87(_PC_64, _MCW_PC);
        r = (float)log1p(exp((double)x));
        _control87(save, _MCW_PC);
        return r;
    }
#elif defined(LZ_X87_FLOAT_CW)
    {
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
#endif /* LZ_SOFTPLUS_POLY && !LZ_EXACT_MATH */
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
   quarter-integer within +/-127, 16 ULPs on each side of
   every half-integer, the zero neighborhood (incl. +/-0 and subnormals),
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
   (the no-emms convention only covers MMX), so this kernel can be called straight
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
/* lz_cpu_has_mmx is an inline read of g_lz_has_mmx, in ops_quant.h,
   which this file already includes. The one-time probe behind it is
   defined further down. */

/* lz_i32f (from ops_kernel_shared.h), q8_amax, q8_group_scale,
   LZ_Q8_MIN_SCALE_F, LZ_Q8_INV127 and LZ_Q15_INV are in
   ops_quant.c. lz_quantize_q8 stays here (it uses the dispatch
   tables and lz_q8r_tier). */

/* Above this amax, 127/amax provably cannot overflow, so the four
   INT-INPUT quantizers below do not have to perform the division to
   find that out.
   The true threshold is 127/FLT_MAX = 3.73e-37; this is eight orders
   above it, which is what makes the implication one-way and exact
   rather than a boundary the rounding of the division could sit on.
   Below it they still divide and still test, so behaviour is identical
   on every input - including under -DLZ_Q8_MIN_SCALE=0.0f, the probe
   build that is the only way the slow arm is reachable at all.
   A divide is not a multiply: 117 instructions against 30 on ARMv5TE
   (.prof/arm_div_count.sh), while the census counts both as 1. */
#define LZ_Q8_INV_SAFE 1.0e-30f

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
        /* Paid once per group regardless of which branch below runs -
           same convention as lz_gdn_quantize_2p's identical bill.
           amax*(1/127) mul, 127/amax div, and q8_group_scale's three
           float comparisons: amax > 0.0f, *inv <= FLT_MAX, *sg <
           LZ_Q8_MIN_SCALE_F. Billed here rather than inside it because
           its callers already bill everything else on its behalf, and
           a helper that bills half of itself is worse than one that
           bills none. Its amax SCAN is not among them: q8_amax compares
           bit patterns as uint32, one instruction each, not
           __aeabi_fcmp. */
        LZ_FCX(LZ_FC_QUANT, 1, 0, 1, 0, 3);
        if (fast) {
            /* One division per group (the reciprocal); everything inside
               is multiplies. x87's FDIV is 39 cycles vs FMUL's 5 - this
               step is worth far more on a PII than on this machine. */
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
            if ((gs & 31) == 0 && tier) {
                /* SIMD tiers round via a hardware convert (cvtps2pi or
                   equivalent inside the kernel calls below), not q8_round's
                   magic-number add - billed as mul+cvt, not mul+add, same
                   split lz_gdn_quantize_2p's kernel branches use. Same
                   count whichever of the two sub-tiers (SSE1+MMX vs SSE2)
                   or toolchain path below actually runs. */
                LZ_FCX(LZ_FC_QUANT, gs, 0, 0, gs, 0);
#if defined(__WATCOMC__)
                /* One call per GROUP, not per 32-element sub-chunk:
                   the #pragma aux bodies expand inline at the call
                   site, so a per-sub-chunk call would multiply that
                   inline code. lz_q8round_group_sse2/_sse do the WHOLE
                   gs/32-chunk loop for this group (including, for the
                   SSE1+MMX tier, the emms this function emits - same
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
                    /* This emms must be INSIDE the group loop, not at
                       the function tail. The usual convention - kernels
                       don't emit; the caller emits once after its loop -
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
            {
                LZ_FCX(LZ_FC_QUANT, gs, gs, 0, 0, 0);  /* mul, q8_round(add) */
                k = 0;
#if defined(LZ_HAVE_Q8R_PACK_MMX)
                /* The MMX tier: a machine with MMX and no SSE1 - the
                   floor of the target family - reaches this arm, where
                   before it ran the scalar loop below for every
                   element. Two passes over each 32: x87 does the
                   multiply and the round into a scratch, then
                   lz_q8round32_pack_mmx does the clamp and the packs
                   four wide and takes the emms with it. The scalar loop
                   below is not duplicated for the tail - `k` continues
                   into it, which is the same contract the SSE1 and SSE2
                   kernels have with their callers. */
                if (lz_cpu_has_mmx()) {
                    int32_t sc[32];
                    int j;
                    for (; k + 31 < gs; k += 32) {
                        for (j = 0; j < 32; j++)
                            sc[j] = q8_round(grp[k + j] * inv);
                        lz_q8round32_pack_mmx(sc, out + k);
                    }
                }
#endif /* LZ_HAVE_Q8R_PACK_MMX */
                for (; k < gs; k++) {
                    int qi = q8_round(grp[k] * inv);
                    qi = (qi > 127) ? 127 : qi;
                    out[k] = (int8_t)((qi < -127) ? -127 : qi);
                }
            }
        } else {
            LZ_FCX(LZ_FC_QUANT, 0, gs, gs, 0, 0);  /* div, q8_round(add) */
            for (k = 0; k < gs; k++) {
                int qi = q8_round(grp[k] / s[g]);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        }
    }
}

/* Per-token absmax INT8 quantize (see ops.h). One scale per n-wide row;
   the matmul reads xqs per 32 elements, so the per-token scale is
   replicated into every n/32 slot - the layout the caller's xqs stride
   (n/32) already provides. Uses q8_group_scale over the WHOLE row, so
   the fast/else decision, the MIN_SCALE floor and the stored scale all
   match lz_quantize_q8 bit for bit; only the grouping changes. */
void lz_quantize_q8_tok(const float *x, int n, int8_t *q, float *s) {
    int ns = n / 32;
    int k;
    float inv, sg;
    int fast;
    if (ns < 1) return;                 /* defense: n not a multiple of 32 */
    fast = q8_group_scale(x, n, &sg, &inv);
    LZ_FCX(LZ_FC_QUANT, 1, 0, 1, 0, 0);  /* amax*(1/127) mul, 127/amax div */
    for (k = 0; k < ns; k++) s[k] = sg;
    if (fast) {
        LZ_FCX(LZ_FC_QUANT, n, n, 0, 0, 0);  /* mul, q8_round(add) */
        for (k = 0; k < n; k++) {
            int qi = q8_round(x[k] * inv);
            qi = (qi > 127) ? 127 : qi;
            q[k] = (int8_t)((qi < -127) ? -127 : qi);
        }
    } else {
        LZ_FCX(LZ_FC_QUANT, 0, n, n, 0, 0);  /* div, q8_round(add) */
        for (k = 0; k < n; k++) {
            int qi = q8_round(x[k] / sg);
            qi = (qi > 127) ? 127 : qi;
            q[k] = (int8_t)((qi < -127) ? -127 : qi);
        }
    }
}

/* Int-domain twin of lz_quantize_q8_tok (see ops.h). t/deq are
   lz_rmsnorm_out_fixed_int's own output; the amax scan, the fast/else
   split and the element rounding all mirror lz_quantize_q8_int's fold
   (see its comment), just grouped by the whole row instead of gs. */
void lz_quantize_q8_tok_int(const int32_t *t, int n, float deq,
                            int8_t *q, float *s) {
    int ns = n / 32;
    int k;
    int32_t tam = 0;
    float amax, inv, sg;
    int fast;
    if (ns < 1) return;                 /* defense: n not a multiple of 32 */
    for (k = 0; k < n; k++) {
        int32_t a = t[k] < 0 ? (int32_t)(0u - (uint32_t)t[k]) : t[k];
        if (a > tam) tam = a;
    }
    amax = (float)tam * deq;
    /* deq mul, tam->float cvt, and the `amax > 0.0f` below. The element
       scan above is int32 - `a > tam` is one instruction, not a
       __aeabi_fcmp - so it is not billed as a comparison. */
    LZ_FCX(LZ_FC_QUANT, 1, 0, 0, 1, 1);
    if (amax > 0.0f) {
        sg = amax * LZ_Q8_INV127;
        /* amax*(1/127) mul, and the sg < LZ_Q8_MIN_SCALE_F below. */
        LZ_FCX(LZ_FC_QUANT, 1, 0, 0, 0, 1);
        inv = 1.0f;                          /* NON-ZERO FLAG ONLY */
        fast = 1;
        if (sg < LZ_Q8_MIN_SCALE_F) {
            sg = 1.0f;
            inv = 0.0f;
        } else if (amax < LZ_Q8_INV_SAFE) {
            /* Billed inside, for the reason the twin above gives. */
            LZ_FCX(LZ_FC_QUANT, 0, 0, 1, 0, 2);
            fast = ((127.0f / amax) <= FLT_MAX);
        }
    } else {
        sg = 1.0f;
        inv = 0.0f;
        fast = 1;
    }
    for (k = 0; k < ns; k++) s[k] = sg;
    if (fast) {
        float inv_elem = (inv == 0.0f) ? 0.0f : 127.0f / (float)tam;
        for (k = 0; k < n; k++) {
            int qi = q8_round((float)t[k] * inv_elem);
            qi = (qi > 127) ? 127 : qi;
            q[k] = (int8_t)((qi < -127) ? -127 : qi);
        }
    } else {
        for (k = 0; k < n; k++) {
            int qi = q8_round(((float)t[k] * deq) / sg);
            qi = (qi > 127) ? 127 : qi;
            q[k] = (int8_t)((qi < -127) ? -127 : qi);
        }
    }
}

/* Integer element loop for lz_quantize_q8_int (below). -DLZ_Q8INT_INT=0
   restores the float one and is the control arm every number about this
   loop was measured against. */
#ifndef LZ_Q8INT_INT
#define LZ_Q8INT_INT 1
#endif /* LZ_Q8INT_INT */

/* 127*m/tam, correctly rounded, without a per-element divide and without
   leaving int32. Shared by both integer quantizers below. Per group:
   rcp = ceil(127*2^SH/tam), the amax having first been normalized to
   32767 or less. Per element: q0 = (m*rcp + 2^(SH-1)) >> SH, then one
   unconditional correction.

   SH is 23 because the product has to fit int32 and its ceiling is
   127*2^SH + tam: at 23 that is 1,069,580,288 against 2,147,483,647,
   2.0x of headroom (observed maximum 1,069,628,752). SH cannot instead
   be raised until the correction is unnecessary - the product is always
   about q*2^SH while exactness needs 2^SH > 2*tam^2, and at tam = 32768
   that wants SH = 39. A single multiply-shift in int32 cannot round this
   quotient exactly; the correction is what makes it exact, not a
   fallback.

   Because rcp rounds UP, q0 is never below the true quotient and never
   more than one above it, so the correction only ever subtracts, and its
   test is the exact one: q0 is right iff 2*tam*q0 <= 254*m + tam.

   Half-away-from-zero, on the magnitude, matching epi_align_i16 rather
   than q8_round's half-to-even - the magic add is a float-domain trick
   with nothing to reuse here, and C integer division truncates toward
   zero, so the sign is taken out and put back explicitly. */
#define LZ_Q8RCP_SH   23
#define LZ_Q8RCP_HALF (1 << (LZ_Q8RCP_SH - 1))
#define LZ_Q8RCP_NUM  (127 << LZ_Q8RCP_SH)
#define LZ_Q8RCP_MAX  32767            /* amax range the reciprocal serves */

/* Positive control for lz_quantize_q8_int's integer element loop:
   elements rounded through the live reciprocal. Counted at the loop, not
   at the switch, and not in the zero-reciprocal branch either - a build
   that took the branch every time would perform no integer rounding and
   would still look enabled from the outside. Zero in the
   -DLZ_Q8INT_INT=0 control build by construction. */
lz_i64 lz_debug_q8int_int = 0;

/* Int-domain Q8 quantize for the fixed-tier gated norm. t holds the Q15
   result the norm wrote as int (no per-element dequant) and deq its
   power-of-two scale. The group's float amax would be t_amax*deq, so
   the stored scale matches lz_quantize_q8 on the float input
   (float)t[i]*deq bit for bit. deq cancels in the round:
     round((t*deq) * 127/(t_amax*deq)) == round(t * 127/t_amax),
   so the element loop uses 127/t_amax and deq enters the stored scale
   only (one multiply per group). inv (127/amax) keeps the fast/else
   decision and the MIN_SCALE zeroing identical to q8_group_scale; the
   else branch (unreachable with the default LZ_Q8_MIN_SCALE floor - a
   tiny amax hits MIN_SCALE first) keeps deq in the numerator to match
   the float path's own else exactly.

   What the width costs, and where it is paid. The int16 twin's element
   loop does not transplant here and the reason is arithmetic, not
   headroom: the shift needs 2^SH > tam for the reciprocal's error to
   stay under one level, so the product is about 127*tam, and past
   tam = 2^24 that is over INT32_MAX. This entry's per-group amax is
   measured at 461,538,651 (kmr20 zh; 431,706,623 en, 443,790,013 /
   440,078,597 on kunmoe-v2-t2-probe), about 2^28.8, and the twin's
   correction test `254*m + tam < 2*tam*qi` overflows int32 on its left
   side alone there. Two changes buy both back:

   - A PER-GROUP PRE-SHIFT normalizes tam into the range the reciprocal
     already serves, and rounds the element UP while the amax rounds
     down, so the approximation stays ONE-SIDED and the single subtract
     is still enough. The error it adds is under 254/16383 of a level.
     The shift is chosen per group AFTER the amax is known, which is why
     it costs nothing measurable: narrowing the producer instead would
     have to pick one shift for a whole row, and this entry's per-group
     amax spans 151,791 to 461,538,651 in a single run - a fixed >>15
     would leave the smallest group 4 levels of the 128 it needs.
   - THE CORRECTION TEST IS EVALUATED MOD 2^32. `254*m + tam < 2*tam*q`
     is `(127*m - tam*q) + (tam>>1) < 0` for integers at either parity,
     and THAT quantity is tam*(R - q) + floor(tam/2) with R - q in
     (-0.52, 0.5], so it fits int32 for every tam an int32 amax can
     hold, while the two sides it came from do not. In uint32 the wrap
     is exact and the sign is bit 31, so no int64 and no
     implementation-defined signed conversion.

   Result: the same exactly-rounded quotient the twin returns, and on
   the domain the two share (tam <= 32767) the two loops agree on all
   536,887,295 pairs - measured, not assumed, so the kda_lat call site
   that widens an int16 into this entry gets the int16 entry's answer.

   .prof/verify_q8int_int.c is the proof: 27,918,893,056 (tam, m) pairs
   against two independent references, covering sh == 0 and the whole
   sh == 1 band exhaustively and every other shift band at the corners
   of each reciprocal cell. */
void lz_quantize_q8_int(const int32_t *t, int n, int gs, float deq,
                        int8_t *q, float *s) {
    int g, k;
    if (gs <= 0 || n < gs || (n % gs) != 0) return;   /* defense: f32 weights have gs=0 etc. */
    for (g = 0; g < n / gs; g++) {
        const int32_t *grp = t + (size_t)g * gs;
        int8_t *out = q + (size_t)g * gs;
        int32_t tam = 0;
        float amax, inv;
        int fast;
        /* int-domain absmax over t: |t| is monotone in the float image,
           so this is the same maximum q8_amax would find over
           (float)t[i]*deq (deq >= 0). Two's-complement negation is
           UB-free; Q15 output never reaches INT32_MIN anyway. */
        for (k = 0; k < gs; k++) {
            int32_t a = grp[k] < 0 ? (int32_t)(0u - (uint32_t)grp[k])
                                   : grp[k];
            if (a > tam) tam = a;
        }
        amax = (float)tam * deq;
        /* deq mul, tam->float cvt, and the `amax > 0.0f` below. The
           element scan above is NOT billed as comparisons: `a > tam` is
           int32, one instruction, not a __aeabi_fcmp. */
        LZ_FCX(LZ_FC_QUANT, 1, 0, 0, 1, 1);
        if (amax > 0.0f) {
            s[g] = amax * LZ_Q8_INV127;
            /* amax*(1/127) mul, and the s[g] < LZ_Q8_MIN_SCALE_F below. */
            LZ_FCX(LZ_FC_QUANT, 1, 0, 0, 0, 1);
            /* Non-zero flag only. This quantizer's element loop scales
               by inv_elem = 127/(float)tam, never by inv, so inv's VALUE
               is dead here - it survives as the "amax was usable" flag
               inv_elem tests. That is exactly why the division can go:
               nothing reads what it computes, only whether it would have
               overflowed, and LZ_Q8_INV_SAFE answers that without
               dividing. */
            inv = 1.0f;
            fast = 1;
            if (s[g] < LZ_Q8_MIN_SCALE_F) {
                s[g] = 1.0f;
                inv = 0.0f;
            } else if (amax < LZ_Q8_INV_SAFE) {
                /* 127/amax div, the amax < LZ_Q8_INV_SAFE that got here,
                   and the <= FLT_MAX on the quotient. Billed INSIDE the
                   else-if, not above it: the INV_SAFE test only runs when
                   the MIN_SCALE one was false, and billing both at the
                   top would credit the rare branch to every group. */
                LZ_FCX(LZ_FC_QUANT, 0, 0, 1, 0, 2);
            }
        } else {
            s[g] = 1.0f;
            inv = 0.0f;
            fast = 1;
        }
        if (fast) {
#if LZ_Q8INT_INT
            if (inv == 0.0f) {
                /* MIN_SCALE or an empty group: the float path multiplies
                   by 0 and every q is 0. A BRANCH, where the int16 twin
                   folds the same case into its arithmetic - the fold does
                   not survive the width. With rcp == 0 the correction
                   term degenerates to 127*m + (tam>>1), which reaches bit
                   31 at m = 16,909,321 and would return -1. Live, not
                   defensive: the kda_lat call site takes it for 3,618 of
                   its 7,236 groups on the zh fixture, all with tam == 0. */
                for (k = 0; k < gs; k++) out[k] = 0;
            } else {
                /* Normalize the amax into the reciprocal's range, then
                   round each element with a multiply-shift and one
                   correction. Nothing here is billed: the loop performs
                   no float operation at all, and the reciprocal replaced
                   a float divide with an integer one. */
                int sh = 0;
                int32_t tamn, rcp, rnd;
                uint32_t tamu = (uint32_t)tam, tamh = tamu >> 1;
                while ((tam >> sh) > LZ_Q8RCP_MAX) sh++;
                tamn = tam >> sh;
                rnd  = (1 << sh) - 1;
                rcp  = (int32_t)((LZ_Q8RCP_NUM + tamn - 1) / tamn);
                for (k = 0; k < gs; k++) {
                    int32_t v = grp[k];
                    int neg = (v < 0);
                    /* two's-complement negation, UB-free, same
                       convention as the amax scan above */
                    uint32_t mu = neg ? 0u - (uint32_t)v : (uint32_t)v;
                    /* ceil, and unsigned so mu + rnd cannot overflow a
                       magnitude near INT32_MAX */
                    int32_t mn = (int32_t)((mu + (uint32_t)rnd) >> sh);
                    int32_t qi = (mn * rcp + LZ_Q8RCP_HALF) >> LZ_Q8RCP_SH;
                    /* `254*m + tam < 2*tam*qi` rewritten so it fits: see
                       this function's header. Bit 31 IS the sign, so no
                       signed conversion of a wrapped value. */
                    qi -= (int32_t)((127u * mu - tamu * (uint32_t)qi
                                     + tamh) >> 31);
                    /* Both clamps are defense and known to be: the proof
                       reports qi in [0, 127] over the whole domain,
                       because tam is this group's own amax. They stay for
                       an input the amax scan mis-negates - INT32_MIN,
                       whose magnitude the scan drops, leaving m > tam -
                       and cost nothing on the criterion this loop was
                       rewritten for (the census excludes compares). */
                    qi = (qi > 127) ? 127 : qi;
                    qi = (qi < 0) ? 0 : qi;
                    out[k] = (int8_t)(neg ? -qi : qi);
                }
                lz_debug_q8int_int += gs;
            }
#else
            /* The float path multiplies (t[i]*deq) by 127/(tam*deq); in
               the int domain the same round is t[i] * (127/tam). inv==0
               is the MIN_SCALE/zero-amax path, where the float path
               multiplies by 0 - mirror it (the stored scale is 1.0 and
               every q is 0). */
            float inv_elem = (inv == 0.0f) ? 0.0f : 127.0f / (float)tam;
            /* Common case (inv != 0): inv_elem's own div+cvt, once per
               group, plus per element a cvt (grp[k] to float), a mul
               and q8_round's add. The rare inv==0 path skips inv_elem's
               div+cvt - billed as if it always ran, same flat-bill
               convention the MIN_SCALE branch gets everywhere else. */
            LZ_FCX(LZ_FC_QUANT, gs, gs, 1, gs + 1, 0);
            for (k = 0; k < gs; k++) {
                /* lz_i32f, never a bare (float) cast: grp[k] passes 2^24
                   and inv_elem is not a power of two, so SSE rounds the
                   integer and then the product while x87 fuses fild+fmul
                   and rounds only the product - 1 ULP apart, and a .5 tie
                   turns that into a different int8, which quantization
                   chaos then spreads across every later token. lz_i32f's
                   split-and-add forces a float32 rounding neither can
                   fuse away. build/norm_int_xcompiler_gate.sh is the
                   gate; it needs long inputs from more than one corpus,
                   since the tie is data-specific. */
                int qi = q8_round(lz_i32f(grp[k]) * inv_elem);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
#endif /* LZ_Q8INT_INT */
        } else {
            /* Unreachable under the default LZ_Q8_MIN_SCALE floor (a tiny
               amax hits MIN_SCALE first, see this function's own comment)
               - fixed anyway rather than leaving a second copy of the
               lz_i32f hazard below. */
            LZ_FCX(LZ_FC_QUANT, gs, gs, gs, gs, 0);  /* cvt, mul, div, q8_round(add) */
            for (k = 0; k < gs; k++) {
                int qi = q8_round((lz_i32f(grp[k]) * deq) / s[g]);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        }
    }
}

/* Integer element loop for lz_quantize_q8_i16 (below). -DLZ_Q8I16_INT=0
   restores the float one and is the control arm every number about this
   loop was measured against. */
#ifndef LZ_Q8I16_INT
#define LZ_Q8I16_INT 1
#endif /* LZ_Q8I16_INT */

/* The reciprocal is LZ_Q8RCP_* above, defined once and shared with
   lz_quantize_q8_int - the two entries round the same quotient the same
   way and differ only in whether the amax needs normalizing first (here
   it never does: |t| <= 32768 already).

   Exhaustive, not sampled: .prof/verify_q8i16_int.c enumerates all
   536,920,065 (tam, m) pairs with 0 <= m <= tam <= 32768 against two
   independent references (integer floor division and long double) -
   0 mismatches, q always within [0, 127], and the correction fires on
   0.0627% of them, without which 336,591 cases are wrong. */

/* Positive control for the integer element loop: elements quantized
   through it. Counted at the loop, not at the switch - a build that
   defines LZ_Q8I16_INT and then never reaches the fast branch would
   still look enabled from the outside, and the int and float paths agree
   on all but a handful of elements, so the logits cannot separate them.
   Zero in the -DLZ_Q8I16_INT=0 control build by construction. */
lz_i64 lz_debug_q8i16_int = 0;

/* int16-input twin of lz_quantize_q8_int (see its comment for the deq
   folding, and ops.h for why the width gets its own entry rather than a
   widening copy into the int32 one). Structurally the same function
   with two differences, both consequences of |t| <= 32768:

     - the fast branch's element loop never leaves the integer domain.
       127*(t*deq)/(tam*deq) is 127*t/tam, and both operands are small
       enough for that quotient to be rounded exactly in int32 (see
       LZ_Q8RCP_SH above), so the producer's int16 exit is not paid for
       with a convert back to float on the very next statement. That is
       what makes the exits feeding this function removals rather than
       relocations, and it is bit-identical across toolchains by
       construction, with no float rounding for gcc/SSE and Watcom/x87 to
       disagree about. It agrees with the int32 entry again now that that
       one is integer too - measured on all 536,887,295 shared pairs, not
       assumed.
     - tam fits int16, so the amax scan needs no normalizing pre-shift.

   The stored scale s[g] stays float and stays exactly what it was:
   lz_matmul_xq reads xqs as float, and only the per-element path moved.

   deq is the producer's power-of-two exit scale (2^-target_e). */
void lz_quantize_q8_i16(const short *t, int n, int gs, float deq,
                        int8_t *q, float *s) {
    int g, k;
    if (gs <= 0 || n < gs || (n % gs) != 0) return;   /* defense: f32 weights have gs=0 etc. */
    for (g = 0; g < n / gs; g++) {
        const short *grp = t + (size_t)g * gs;
        int8_t *out = q + (size_t)g * gs;
        int32_t tam = 0;
        float amax, inv;
        int fast;
        for (k = 0; k < gs; k++) {
            int32_t a = grp[k] < 0 ? -(int32_t)grp[k] : (int32_t)grp[k];
            if (a > tam) tam = a;
        }
        amax = (float)tam * deq;
        /* deq mul, tam->float cvt, and the `amax > 0.0f` below. The
           element scan above is NOT billed as comparisons: `a > tam` is
           int32, one instruction, not a __aeabi_fcmp. */
        LZ_FCX(LZ_FC_QUANT, 1, 0, 0, 1, 1);
        if (amax > 0.0f) {
            s[g] = amax * LZ_Q8_INV127;
            /* amax*(1/127) mul, and the s[g] < LZ_Q8_MIN_SCALE_F below. */
            LZ_FCX(LZ_FC_QUANT, 1, 0, 0, 0, 1);
            /* Non-zero flag only. This quantizer's element loop scales
               by inv_elem = 127/(float)tam, never by inv, so inv's VALUE
               is dead here - it survives as the "amax was usable" flag
               inv_elem tests. That is exactly why the division can go:
               nothing reads what it computes, only whether it would have
               overflowed, and LZ_Q8_INV_SAFE answers that without
               dividing. */
            inv = 1.0f;
            fast = 1;
            if (s[g] < LZ_Q8_MIN_SCALE_F) {
                s[g] = 1.0f;
                inv = 0.0f;
            } else if (amax < LZ_Q8_INV_SAFE) {
                /* 127/amax div, the amax < LZ_Q8_INV_SAFE that got here,
                   and the <= FLT_MAX on the quotient. Billed INSIDE the
                   else-if, not above it: the INV_SAFE test only runs when
                   the MIN_SCALE one was false, and billing both at the
                   top would credit the rare branch to every group. */
                LZ_FCX(LZ_FC_QUANT, 0, 0, 1, 0, 2);
            }
        } else {
            s[g] = 1.0f;
            inv = 0.0f;
            fast = 1;
        }
        if (fast) {
#if LZ_Q8I16_INT
            /* rcp == 0 IS the float path's inv_elem == 0 (MIN_SCALE or a
               zero amax): the product vanishes, HALF alone shifts to 0,
               and the correction test 254*m+tam < 0 is false for every
               non-negative m - all-zero output, no branch of its own. It
               is also what keeps the divide below away from tam == 0,
               since inv != 0 implies tam >= 1. Nothing here is billed:
               the loop performs no float operation at all, and the
               reciprocal replaced a float divide with an integer one. */
            int32_t rcp = (inv == 0.0f)
                        ? 0 : (int32_t)((LZ_Q8RCP_NUM + tam - 1) / tam);
            for (k = 0; k < gs; k++) {
                int32_t v = grp[k];
                int neg = (v < 0);
                int32_t m = neg ? -v : v;
                int32_t qi = (m * rcp + LZ_Q8RCP_HALF) >> LZ_Q8RCP_SH;
                qi -= (254 * m + tam < 2 * tam * qi);
                /* Defense only, and known to be: the exhaustive proof
                   reports qi in [0, 127] over the whole domain, because
                   tam is this group's own amax ten lines up. It stays
                   for a future caller that passes a tam which is not,
                   and costs nothing on the criterion this loop was
                   rewritten for (the census excludes compares). The
                   lower clamp is gone: m, rcp and HALF are all
                   non-negative, and the correction can only fire when
                   2*tam*qi exceeds a non-negative number, which needs
                   qi >= 1 - so it never drives qi below zero. */
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)(neg ? -qi : qi);
            }
            lz_debug_q8i16_int += gs;
#else
            float inv_elem = (inv == 0.0f) ? 0.0f : 127.0f / (float)tam;
            LZ_FCX(LZ_FC_QUANT, gs, gs, 1, gs + 1, 0);
            for (k = 0; k < gs; k++) {
                int qi = q8_round((float)grp[k] * inv_elem);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
#endif /* LZ_Q8I16_INT */
        } else {
            /* Unreachable under the default LZ_Q8_MIN_SCALE floor, same
               as the int32 entry's else - carried so the two functions
               stay term for term the same. Unreachable here for a
               stronger reason than the floor, which is why it did NOT
               follow the fast branch into the integer domain: reaching
               it needs 127/amax to overflow float, i.e. amax < 4e-37,
               while amax is tam*deq with tam >= 1 and deq a power of two
               the producer's exponent keeps far above that. It is dead
               under -DLZ_Q8_MIN_SCALE=0.0f too. */
            LZ_FCX(LZ_FC_QUANT, gs, gs, gs, gs, 0);  /* cvt, mul, div, q8_round(add) */
            for (k = 0; k < gs; k++) {
                int qi = q8_round(((float)grp[k] * deq) / s[g]);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        }
    }
}

/* int64-input twin of lz_quantize_q8_int (see its comment for the deq
   folding). The attention weighted sum is exact only at int64 width, so
   this is what the fixed-tier attention int path consumes. The element
   loop's (float)grp[k] has the SAME fusion hazard tamf is already
   guarded against below - NOT measure-zero, that was the int32 twin's
   own comment before the RCA that found a real occurrence in 153
   tokens. A synthetic sweep over grp[k] near q8_round's .5-tie
   boundaries (.prof/probe_int64_fuse.c, T in {16777217, 50000000,
   102808064, 134217728, 2000000000}) found real gcc/Watcom mismatches
   without the volatile below - fixed the same way as tamf. */
void lz_quantize_q8_int64(const int64_t *t, int n, int gs, float deq,
                          int8_t *q, float *s) {
    int g, k;
    if (gs <= 0 || n < gs || (n % gs) != 0) return;   /* defense: f32 weights have gs=0 etc. */
    for (g = 0; g < n / gs; g++) {
        const int64_t *grp = t + (size_t)g * gs;
        int8_t *out = q + (size_t)g * gs;
        int64_t tam = 0;
        float amax, inv;
        int fast;
        /* volatile, and declared here with the others because a
           declaration after the scan below is C99 - see the assignment
           for what the qualifier is doing. */
        volatile float tamf;
        for (k = 0; k < gs; k++) {
            int64_t a = grp[k] < 0 ? (int64_t)(LZ_U64_C(0) - (uint64_t)grp[k])
                                   : grp[k];
            if (a > tam) tam = a;
        }
        /* Force the int64->float32 conversion out of the multiply: gcc
           rounds tam to float32 before the product, while an x87 build
           can fuse (float)tam * deq into one fild+fmul that rounds only
           the product - and deq here is NOT a power of two (unlike the
           int32 twin's deq), so the two roundings differ by 1 ULP.
           volatile forces the float32 store, breaking the fusion on both. */
        tamf = (float)tam;
        amax = tamf * deq;
        /* deq mul, tam->float cvt, and the `amax > 0.0f` below. The
           element scan above is NOT billed as comparisons: `a > tam` is
           int32, one instruction, not a __aeabi_fcmp. */
        LZ_FCX(LZ_FC_QUANT, 1, 0, 0, 1, 1);
        if (amax > 0.0f) {
            s[g] = amax * LZ_Q8_INV127;
            /* amax*(1/127) mul, and the s[g] < LZ_Q8_MIN_SCALE_F below. */
            LZ_FCX(LZ_FC_QUANT, 1, 0, 0, 0, 1);
            /* Non-zero flag only. This quantizer's element loop scales
               by inv_elem = 127/(float)tam, never by inv, so inv's VALUE
               is dead here - it survives as the "amax was usable" flag
               inv_elem tests. That is exactly why the division can go:
               nothing reads what it computes, only whether it would have
               overflowed, and LZ_Q8_INV_SAFE answers that without
               dividing. */
            inv = 1.0f;
            fast = 1;
            if (s[g] < LZ_Q8_MIN_SCALE_F) {
                s[g] = 1.0f;
                inv = 0.0f;
            } else if (amax < LZ_Q8_INV_SAFE) {
                /* 127/amax div, the amax < LZ_Q8_INV_SAFE that got here,
                   and the <= FLT_MAX on the quotient. Billed INSIDE the
                   else-if, not above it: the INV_SAFE test only runs when
                   the MIN_SCALE one was false, and billing both at the
                   top would credit the rare branch to every group. */
                LZ_FCX(LZ_FC_QUANT, 0, 0, 1, 0, 2);
            }
        } else {
            s[g] = 1.0f;
            inv = 0.0f;
            fast = 1;
        }
        if (fast) {
            float inv_elem = (inv == 0.0f) ? 0.0f : 127.0f / tamf;
            /* inv_elem reuses tamf (no extra cast, unlike the int32
               twin's repeated (float)tam); per element a cvt (grp[k],
               int64->float), a mul and q8_round's add. Same flat-bill
               convention for the rare inv==0 path as the int32 twin. */
            LZ_FCX(LZ_FC_QUANT, gs, gs, 1, gs, 0);
            for (k = 0; k < gs; k++) {
                /* volatile: same fusion hazard as tamf above, see this
                   function's own comment. */
                volatile float g32 = (float)grp[k];
                int qi = q8_round(g32 * inv_elem);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        } else {
            /* Unreachable under the default LZ_Q8_MIN_SCALE floor, same
               as the int32 twin's else branch - fixed anyway. */
            LZ_FCX(LZ_FC_QUANT, gs, gs, gs, gs, 0);  /* cvt, mul, div, q8_round(add) */
            for (k = 0; k < gs; k++) {
                volatile float g32 = (float)grp[k];
                int qi = q8_round((g32 * deq) / s[g]);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        }
    }
}

/* Fused fixed-tier gated-RMSNorm quantize (see ops.h). The norm wrote
   the pre-silu product into p (ssm_out) and the Q15 sigmoid into sig;
   this reconstructs the silu value and quantizes it, so the Q15 ->
   float -> int round-trip is avoided while the value stays bit-identical
   to a plain lz_quantize_q8.

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
    /* Moved from LZ_FC_NORM: this is lz_quantize_q8_silu's OWN
       arithmetic (the amax-pass reconstruction and the final
       2^-15 rescale below), not the norm function's - cross-compiler bit-identity requires a
       site counts its own arithmetic only, and this function is the
       quantize wrapper, not the norm itself. */
    LZ_FCX(LZ_FC_QUANT, 3 * n, 0, 0, n, 0);
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
            /* 2 mul (amax_o, s[grp]), 2 div (inv, and the fast check's
               own 127/amax_o - a genuinely separate division, not a
               reuse of inv: amax_o != amax). */
            LZ_FCX(LZ_FC_QUANT, 2, 0, 2, 0, 0);
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
            LZ_FCX(LZ_FC_QUANT, gs, gs, 0, 0, 0);  /* mul, q8_round(add) */
            for (k = 0; k < gs; k++) {
                int qi = q8_round(og[k] * inv);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        } else {
            LZ_FCX(LZ_FC_QUANT, gs, gs, gs, 0, 0);  /* mul, div, q8_round(add) */
            for (k = 0; k < gs; k++) {
                int qi = q8_round((og[k] * (1.0f / 32768.0f)) / s[grp]);
                qi = (qi > 127) ? 127 : qi;
                out[k] = (int8_t)((qi < -127) ? -127 : qi);
            }
        }
        /* Reconstructed silu value, for the f32 matmul fallback and the
           sgdn tap. The 2^-15 multiply is exact, so this is the same
           float the norm's own output loop writes. */
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
   Dispatching deeper would pay one indirect call per 32 MACs - exactly
   what this kernel design exists to avoid. */
/* Defined near the top of the file, above q8_amax, which also reads it. */

#if defined(__WATCOMC__)
/* CPUID leaf 1 EDX: bit 23 = MMX, bit 25 = SSE, bit 26 = SSE2. One
   symbol, one implementation, in cpucheck.c - this file calls it through
   the cpucheck.h declaration. A real __asm-block function, not a
   duplicated #pragma aux (see cpucheck.h for why those two cannot share
   a body across files). */
#endif /* __WATCOMC__ */

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
/* The cache is not enough on its own, and the reason is the call. This
   was one extern function reading a cached global, and the call sites
   are in row loops: 75,080,736 of them over 603 tokens on kmr20, which
   is 124,512 per token. gcc hoists most of them; Open Watcom hoists
   none, because an extern function may have side effects it cannot see
   - all five sites in ops_matmul.c survive as real calls there.

   So the FAST path is inline, in ops_quant.h, reading g_lz_has_mmx
   directly; this is only the one-time probe it falls back to. Same
   shape as q8_round, and for the same reason: the call and the
   dispatch cost more than the body does. */
int g_lz_has_mmx = -1;

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)
int lz_cpu_has_mmx_probe(void) {
    g_lz_has_mmx = (lz_cpuid1_edx() & (1u << 23)) != 0;
    return g_lz_has_mmx;
}
#else
int lz_cpu_has_mmx_probe(void) { g_lz_has_mmx = 0; return 0; }
#endif /* __i386__ || __x86_64__ || _M_IX86 */

/* pf_detect, lz_prefetch_mode, g_pair, lz_pair_mode: in ops_sched.c. */

/* Derived bound for the fixed epilogue: per group the product is
   |acc| * |xw*ww| <= 2^27 * 2^30, so ng of them stay inside
   int64 while ng <= 64. Value in ops.h; the derivation stays here. */

/* The narrowing half of the above, in the MAGNITUDE domain, factored out
   because epi_q41_fixed's exponent join needs the same shift with an
   int64 rather than a bounded int16 exit. Both of the corners the
   verification probe covers live here - the +half bias added in unsigned
   64-bit (umag <= 2^63, half <= 2^62, so the sum cannot wrap) and the
   closed form for rs >= 64, where `>>rs` and `1ULL<<(rs-1)` are
   themselves undefined: umag * 2^-rs <= 0.5 there, with equality (and so
   a half-away-from-zero result of 1) only at rs == 64, umag == 2^63. */
lz_u64 epi_shr_half_u(lz_u64 umag, int rs) {
    if (rs <= 0) return umag;
    if (rs >= 64)
        return (rs == 64 && umag == ((lz_u64)1 << 63)) ? 1 : 0;
    return (umag + ((lz_u64)1 << (rs - 1))) >> rs;
}

/* Signed wrapper. The magnitude is taken by converting to unsigned FIRST
   for the LLONG_MIN reason epi_align_i16's header gives; the result is
   below 2^63 for every rs >= 1 (widest is rs == 1, where
   (2^63 + 2^62) >> 1 < 2^63), so negating it back is representable.

   The sign work is branchless for the TARGET's sake: the two `neg ? :`
   ternaries this replaced were 144,472,859 evaluations each with the
   minority arm at 34.5%, and a P5 has no predictor worth the name.
   `mask` comes from the sign bit of the UNSIGNED value - `um >> 63` is
   defined where `s >> 63` on a signed would be implementation-defined -
   and (x ^ mask) - mask is two's-complement abs in and negate out.

   The shift costs a helper call on wcc386, and no portable spelling
   avoids it. Four were measured there, function bodies including
   prologue:
     (lz_u64)0 - (um >> 63)          13 insns, 1 __U8RS call, no branch
     (lz_u64)0 - (lz_u64)(s < 0)     15 insns, no call, ONE BRANCH
     (lz_u64)(s >> 63)                8 insns, 1 call, implementation
                                      -defined for negative s
     (uint32_t)(um >> 32) >> 31      13 insns, 1 __U8RS call
   wcc386 routes every 64-bit shift through __U8RS whether or not the
   count is a compile-time constant, so narrowing first buys nothing;
   and the compare form trades the call for exactly the branch this
   code exists to avoid. The form here is the best of the four.
   What WOULD do it in three instructions is a #pragma aux (mov the high
   dword, sar 31), which is a new Watcom primitive and owes a gcc twin
   and a kernel-matrix cell - gcc already emits the optimal two
   instructions from the C, so that twin would exist only to satisfy the
   pairing. Not done; the cost is recorded so it need not be re-derived.

   `s == 0` needs no special case, and a fast path for it would never
   fire: the general path gives umag = 0, and epi_shr_half_u(0, rs) is 0
   for every rs.

   `rs <= 0` STAYS, and not for its share: at rs <= 0 with s == LLONG_MIN
   the magnitude is 2^63 and converting it back with (lz_i64) is
   implementation-defined. Every rs >= 1 keeps the result under 2^63 (see
   above), which is what makes the general path safe and this case not.
   Removing it is a correctness bug no data from this model would show. */
/* Non-static: the conv1d fixed tier (ops_conv1d.c) uses the same
   half-away-from-zero narrowing. The u variant is also non-static - the
   epilogue (ops_epi.c) shares it with this TU. */
lz_i64 epi_shr_half(lz_i64 s, int rs) {
    lz_u64 um = (lz_u64)s;
    lz_u64 mask = (lz_u64)0 - (um >> 63);
    lz_u64 umag;
    if (rs <= 0) return s;
    umag = epi_shr_half_u((um ^ mask) - mask, rs);
    return (lz_i64)((umag ^ mask) - mask);
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

     Q8_0: o[i] = sum_g  ws[g]*sum_{sub-block s in g} xqs[s]*dot32(w, x)
     Q4_1: o[i] = sum_g ( d[g]*sum_s xqs[s]*dot32(q, x)
                          + m[g]*sum_s xqs[s]*sum(x) )

   **The reference must reproduce the SIMD reduction ORDER bit for
   bit, not merely be "mathematically equivalent".** The contract is
   not the two summation formulas but HOW THE PARENTHESES GO - because
   SSM state is quantized, a 1e-6 association difference flips some
   +/-1 LSB quantization decision and then amplifies per token - eight
   tokens fork generation. So this code is deliberately
   dumb: int32 sub-block sums first go into arrays (on the SIMD side
   MMX/SSE2 does this), then the exact same float reduction structure
   finishes. Three details must not change:

     - `acc*(sx*sw)` not `(acc*sx)*sw` (gs=32 tier)
     - 4-way accumulators paired as `(a0+a2)+(a1+a3)`, tail ALL into a0 (gs=32 tier)
     - Q4_1's dot and zero each get their own accumulator, merged once at the end (not interleaved per group)

   The x256 fold exists on the SIMD side but not here: 256 is a power
   of two, scaling changes no rounding at any step, so the two sides
   remain bit-identical. */

/* Q4_1 matmul kernel.

   `o[i] = sum_g [ d[i][g]*sum_{sub-block s in g} xqs[s]*dot32(q_w, q_x)
                   + m[i][g]*xg[g] ]`

   Two savings:
   1. **Zero term hoisted out of the row loop**. `xg[g] = sum_{s in g}
      xqs[s]*sum_k xq[k]` depends only on activations - computed once per
      matmul, shared by all output rows. So Q4_1's inner loop differs
      from Q4_0's by not a single instruction, while the asymmetric
      tier's quality measures 2.7-4.4x better.
   2. **Nibble expansion needs no sign extension** (unsigned 0..15),
      keeping the x256 fold from `punpcklbw(0,x)`, cancelled once at
      the row end.

   Reduction order deliberately matches the Q8 wide-group path
   (per-sub-block sequential accumulation; SSE2's batched 4-vector
   reduction `((q0+q1)+q2)+q3` is equivalent), so the gcc/SSE2 and MMX
   builds are bit-identical. */

/* ---- Q4_1 row kernels, one per (ISA x impl) --------------------------- */

/* ---- T2 row kernels, one per (ISA x impl) -----------------------------
   The single 2-bit plane arrives in `w4`; `w2v` and both prefetch bounds
   are unused - 8 bytes per 32 elements is a quarter of Q4_1's weight
   traffic, too few cache lines for a prefetch distance to mean anything
   unmeasured.

   No pairing yet. Not a decision - unmeasured. `--pair` does nothing for
   this format today. */

/* row_t2_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

/* row_t2_mmx_asm / row_t2_sse2_asm live in src/ops_mmx.c /
   src/ops_sse2.c, alongside their leaf kernels (lz_dot32_t2_asm /
   lz_dot32_t2_sse2_asm) which they call direct-by-name, same-TU.
   Declared in src/ops_mmx.h / src/ops_sse2.h. */

/* SSE1 slots NULL: T2's unpack is `pand 0x03` + three `psrlw`, and SSE1
   adds no integer instruction that replaces any of them. */

/* ---- Q6_1 row kernels, one per (ISA x impl) ---------------------------
   The only format that uses BOTH plane pointers and BOTH prefetch
   bounds. The 2-bit plane needs its own: `pf_end` stops exactly at the
   plane boundary. At 22% of all weight bytes on the recipe of record,
   its miss frequency is half the 4-bit plane's, not 1/16 like a scale
   array: it IS a hot stream, just a narrower one. Both bounds are real,
   not belt-and-braces: LZ_PF_LOAD is a REAL dummy load on PII and
   faults past the end of the tensor. */

/* row_q61_sse2_intrin's body lives in src/ops_sse2.c, same reason as
   row_q8_sse2_intrin above. Declared in src/ops_sse2.h. */

/* T2 matmul. Same shape as matmul_q61_impl minus the second plane, with
   one addition: the zero coefficient is -scale, so the negated array is
   built per row into g_t2neg and handed to epi_q41 as its `wm`. */

/* The float exit of the impls that take an LZMatOut, which is what every
   lz_matmul_xq_nt caller wants; the int16 one is reached through
   lz_matmul_xq_nt_i16 instead. */
static void mat_out_f32(LZMatOut *x, float *o) {
    memset(x, 0, sizeof(*x));
    x->o = o;
}

/* Matmul consuming already-quantized activations: the forward hot path.
   xq/xqs quantized by the caller at lz_act_gs()'s group (one activation
   may be reused by several matmuls). f32 weights fall back to f32 -
   one call site serves all weight representations. */
void lz_matmul_xq(float *o, const float *x, const int8_t *xq,
                  const float *xqs, const LZTensor *w,
                  int in_dim, int out_dim) {
    lz_matmul_xq_nt(o, x, xq, xqs, w, in_dim, out_dim, 1);
}

/* Which of lz_matmul_xq_nt's two inputs it will actually read.
   Exported because a caller that PREPARES those inputs differently -
   the SubLN Hadamard prepares the quantized row only - has to ask the
   same question, and two spellings of it drift. That drift already
   happened once: the rotation reached xq and never the float row, so
   an unquantized model computed W.x while a quantized one computed
   W.quant(Hx), silently, from one config. */
int lz_matmul_xq_reads_float_row(const LZTensor *w) {
    return !w || w->dtype == LZ_FMT_F32;
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
    if (lz_matmul_xq_reads_float_row(w)) {
        for (i = 0; i < nt; i++)
            lz_matmul(o + (size_t)i * out_dim, x + (size_t)i * in_dim,
                      w ? w->f : NULL, in_dim, out_dim);
        return;
    }
    if (!xq || !xqs || w->gs <= 0 || (in_dim % w->gs) != 0) {
        for (i = 0; i < nt * out_dim; i++) o[i] = 0.0f;  /* defense: no silent miscalc */
        return;
    }
    /* The widen buffers are LZ_BATCH_MAX * LZ_MM_WIDEN_MAX; nt tokens'
       activations always fit: the g_xw-filling path carries its own
       `in_dim <= LZ_MM_WIDEN_MAX` check (Q8_0's !widened branch does
       not widen and goes per-token scalar), and Q4_1/Q6_1 dispatch
       already requires in_dim <= LZ_MM_WIDEN_MAX. */
    /* LZ_KERNEL_REF must genuinely reach the scalar reference, not just
       affect the name reported. Dispatch is by dtype, so without this
       check `--kernel ref` would actually run SIMD, unnoticed because
       all kernels are bit-identical. The scalar reference is the
       CONTRACT for new formats; an oracle that cannot be
       selected is no oracle. */
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* No ARM interception here, and the reason it is absent rather than
       forgotten: all five formats have their armC/armA pair
       (src/ops_arm.c, plus ternary in src/ops_t2_arm.c), so the dispatch
       below is the whole story on ARM too. A format WITHOUT one needs
       the interception - route that dtype to matmul_scalar_ref, so the
       tiers differ by the kernel under test and not by which fallback
       they fell through. Such a format lands as a `t` in km_reg. */
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
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_q8_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* Q4_1's SIMD path pre-expands activations to int16; beyond the
       buffer cap it falls back to the scalar reference. s1v3's max
       in_dim is 1024, the 0.8B's 3584 - both within 4096. */
    if (w->dtype == LZ_FMT_Q4_1 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && w->zero) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_q41_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    if (w->dtype == LZ_FMT_Q6_1 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && w->zero) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_q61_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    /* No `w->zero` in the guard: ternary carries no min, and a T2 tensor
       that HAS one was mis-loaded. */
    if (w->dtype == LZ_FMT_T2 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX && !w->zero) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_t2_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN) || defined(__arm__)
    /* Q16_0 has no non-widened fallback: unlike Q8_0 it always needs the
       activation expanded to int16, so in_dim past the buffer cap goes to
       the scalar reference. The gcc/SSE2-only dev build has no Q16_0 SIMD
       kernel either and lands there too - that is deliberate, not an
       oversight. The differential oracle the bit-identity contract requires is a gcc build
       with LZ_MMX_TU set (src/ops_mmx.c linked in), which compiles the
       intrinsics twin and exercises this same path. */
    if (w->dtype == LZ_FMT_Q16_0 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX) {
        LZMatOut ex; mat_out_f32(&ex, o);
        matmul_q16_impl(&ex, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN */
    matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
}

/* o = W x: unified entry for weight tensors.
   dtype=0 goes f32; dtype=1 quantizes x (Q8_0, group = w->gs) into the
   caller-provided xq/xqs buffers, int8*int8 -> int32 accumulation,
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
#if defined(LZ_MATMUL_F32_SSE_EXTERN)
        /* The eight accumulators above ARE two SSE vectors - lane k of
           them accumulates exactly the sequence scalar a_k does, in the
           same order, so this is bit-identical rather than a
           re-association. Selected by tier, like every other kernel: a
           --kernel ref run must reach the loop below or the arm this is
           compared against would be this.

           SSE1, not SSE2: mulps and addps are SSE1's, and every SSE2
           machine has them - so one kernel serves both tiers and the
           SSE2 column holds the same body rather than a copy of it.
           No CPUID in the test: lz_kernel_select clamps both of these
           values to what CPUID found, so reaching either one is already
           the answer, and reading CPUID here would only ask it again. */
        if ((g_kernel == LZ_KERNEL_SSE2 || g_kernel == LZ_KERNEL_SSE) &&
            in_dim >= 8) {
            float acc8[8];
            j = lz_matmul_row_sse(row, x, in_dim, acc8);
            a0 = acc8[0]; a1 = acc8[1]; a2 = acc8[2]; a3 = acc8[3];
            a4 = acc8[4]; a5 = acc8[5]; a6 = acc8[6]; a7 = acc8[7];
        }
#endif /* LZ_MATMUL_F32_SSE_EXTERN */
#if defined(LZ_ARM_ASM_EXTERN)
        /* Same shape, same contract - the kernel INITIALISES acc8, so
           the eight scalars above are overwritten rather than added to,
           and lane k still accumulates the sequence scalar a_k does.
           On a target with no FPU each of these MACs is two library
           calls; the kernel fuses them and returns where it stopped.
           Selected by tier for the reason the SSE arm gives. */
        if (g_kernel == LZ_KERNEL_ARM_ASM && in_dim >= 8) {
            float acc8[8];
            j = lz_matmul_row_arm_asm(row, x, in_dim, acc8);
            a0 = acc8[0]; a1 = acc8[1]; a2 = acc8[2]; a3 = acc8[3];
            a4 = acc8[4]; a5 = acc8[5]; a6 = acc8[6]; a7 = acc8[7];
        }
#endif /* LZ_ARM_ASM_EXTERN */
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

/* lz_fwht, the float transform, is in src/fwht.c beside its int32 twin.
   It was here; one operator in two files meant changing the butterfly
   twice and made every per-file census see one operator as two. */

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

void lz_moe_route(const float *logits, const short *li, int li_e,
                  const float *bias, int n_experts,
                  int top_k, int sigmoid, int renormalize, float tau,
                  int *idx_out, float *w_out) {
    /* static, not stack: LZ_MAX_EXPERTS is 128, and rule six clause 4
       ("large buffers stay off the stack") is written for a Win98
       target with a small default stack - four 512-byte arrays per
       call is exactly the kind of thing that rule exists for. */
    static float score[LZ_MAX_EXPERTS];
    static float tscore[LZ_MAX_EXPERTS];
    static float key[LZ_MAX_EXPERTS];
    /* Where an int16 input is rebuilt as floats for the readers that
       have no integer entry - see the `li` half of this function's
       header. Static for the same reason the three above are. */
    static float flog[LZ_MAX_EXPERTS];
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

    /* The int input only reaches the sigmoid scorer as integers. Every
       other reader below wants a float vector, so build it once here -
       that is the producer's declined convert and multiply arriving
       after all, which is why it bills at the epilogue's site rather
       than at this function's. */
    if (li && (!sigmoid || tempered)) {
        float deq = pow2f(-li_e);
        for (i = 0; i < n_experts; i++) flog[i] = lz_i32f(li[i]) * deq;
        LZ_FCX(LZ_FC_EPI, n_experts, 0, 0, n_experts, 0);
        logits = flog;
    }

    if (sigmoid) {
        if (li) for (i = 0; i < n_experts; i++) score[i] = lz_sigmoid_i(li[i], li_e);
        else    for (i = 0; i < n_experts; i++) score[i] = lz_sigmoid(logits[i]);
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

       No flush-to-zero here; that is a measurement rather than an
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
   edited together; the TRAINING tree's tests/test_kernel_matrix.py
   (E:\LLM\llama98-zh, pointed at this tree by LZ_ENGINE - there is no
   tests/ here by policy) compares this against the source, and
   build/kernel_matrix_gate.sh checks the registry below out of the
   shipped binaries. The previous version of this map lived in a
   document and drifted for months. */
#if defined(__WATCOMC__)
#define LZ_G128_Q8  1
#define LZ_G128_Q41 1
#define LZ_G128_Q61 1
#define LZ_G128_Q16 1
#elif defined(LZ_DOT_MMX_EXTERN)
/* The gcc group kernels (dot128_*_mmx) exist and are wired into the row
   functions behind the LZ_G128_GCC knob (default 1, ops_kernel_shared.h)
   - see that knob's comment for the A/B rationale. This column tracks
   the wiring: LZ_G128_GCC=0 reverts it to 'o' and the row functions to
   the per-32 path. The claim is "wired in", not "measured faster" -
   whether the group path wins on any machine is unmeasured. */
#define LZ_G128_Q8  LZ_G128_GCC
#define LZ_G128_Q41 LZ_G128_GCC
#define LZ_G128_Q61 LZ_G128_GCC
#define LZ_G128_Q16 LZ_G128_GCC
#else
/* Neither compiler's group kernels are in this build: dot128_*_mmx sit
   behind LZ_DOT_MMX_EXTERN in src/ops_mmx.c, the #pragma aux twins
   behind __WATCOMC__. The ARM cross-build is the case - it has no row
   kernel for these four formats either, so there is nothing that could
   call a group kernel. Without this branch the ARM binary reports 'G'
   for all four - a column lit on a target whose row is empty, which is
   the exact shape of signal this map exists to refuse. */
#define LZ_G128_Q8  0
#define LZ_G128_Q41 0
#define LZ_G128_Q61 0
#define LZ_G128_Q16 0
#endif /* __WATCOMC__ */
/* T2's group form now exists on all three: dot128_t2_sse2
   (ops_kernel_dot_sse2.h), dot128_t2_mmx (ops_kernel_dot_mmx.h) and
   lz_dot128_t2_asm (ops_mmx.c, the Watcom #pragma aux).

   The argument that kept this at 0 examined the unpack. It said q6_1
   hoists two mask constants across four sub-blocks while T2's unpack has
   only a zero register, which trades pxor for movq one instruction for
   one. Both halves are true, and the group form does not pay there. It
   pays on the HORIZONTAL FOLD: per-32 every sub-block reduces its own
   lanes and extracts, and folding four together costs about half that
   and leaves the results in one register to store whole. The Watcom
   kernel also drops four emms to one, which the per-format twins above
   get for the same reason.

   THE THIRD BRANCH IS NOT OPTIONAL, and the four formats above carry
   the same one for the same reason. On the ARM cross-build neither
   __WATCOMC__ nor the gcc group kernels exist, so a plain #else lights
   this column on a target whose row is empty - kernel_matrix_gate
   reports it as "t2/g128 present but not registered for this build
   class", which is what it did. */
#if defined(__WATCOMC__)
#define LZ_G128_T2  1
#elif defined(LZ_ROW_SSE2_EXTERN)
#define LZ_G128_T2  LZ_G128_GCC
#else
#define LZ_G128_T2  0
#endif /* __WATCOMC__ */

/* ---- the reason registry ---------------------------------------------

   A missing cell must carry a reason, and "still empty"
   must be distinguishable from "impossible". Three classes, one letter
   each in the matrix (ops.h's LZKmCell documents the full alphabet):

     have != 0   some build class carries it. Absent HERE is 'o' (another
                 build has it) and absent THERE is '!' - a gate failure,
                 because that is what a path silently vanishing looks
                 like from inside the binary it vanished from.
     IMPOSSIBLE  no build will ever have it; the audit is below.
     TODO        nobody has written it. Not a defect - a registration.
                 An unregistered cell is '?', which IS a defect.
     DECLINED    it WAS written and measured, and the measurement said
                 the hand cell is not better. Settled, like IMPOSSIBLE,
                 but for a reason that a faster idea could overturn.

   Filling in IMPOSSIBLE where TODO belongs is the expensive mistake:
   it retires the cell permanently and the next person reads it as
   settled. TODO only waits.

   DECLINED exists because that pair could not express f32mm armA.
   IMPOSSIBLE was false - the code compiles and is bit-identical - and
   TODO invited the next reader to spend a day arriving at a number
   already measured. A class that cannot say "tried, worse" makes
   "tried, worse" indistinguishable from "nobody got to it".

   ---- IMPOSSIBLE, cell by cell -----------------------------------------

   sseI, sseA (all five formats). Audited instruction by instruction,
   not by impression:

   What a row kernel executes, every format (src/ops_kernel_dot_mmx.h,
   Watcom twins in src/ops_kernel_dot.h): movq loads; the unpack -
   punpcklbw/punpckhbw for q8_0 and q16_0, pand plus psrlw/psrld/psllw
   plus pxor/por for the 4-bit, 6-bit and 2-bit planes; pmaddwd; paddd;
   then psrlq + paddd + movd to fold the pair. Every one is MMX.

   What SSE1 adds on top of MMX, complete list: pavgb, pavgw, pextrw,
   pinsrw, pmaxsw, pmaxub, pminsw, pminub, pmovmskb, pmulhuw, psadbw,
   pshufw, maskmovq, movntq, prefetchnta/t0/t1/t2, sfence - plus the
   %xmm file, which under SSE1 holds packed float32 ONLY. Integer %xmm
   arrives with SSE2, and that is the sse2I/sse2A column.

   Against the sequence above: none of them is a signed 16x16->32
   multiply-add, so pmaddwd stands; pshufw permutes words and cannot
   zero-extend bytes, so the punpck pair stands; psadbw sums |a-b| over
   UNSIGNED bytes and produces no products; pmulhuw is unsigned and
   throws away the low half a dot product needs; pavg/pmax/pmin/pextrw/
   pinsrw/pmovmskb have no role in a dot product at all. The two that DO
   help are prefetchnta and movntq - and they are ALREADY in the MMX
   kernels (LZ_PFI_NTA, src/ops_kernel_shared.h, emitted as the raw
   bytes 0F 18 00 on Watcom). They are a memory hint, selected by
   --prefetch, which is orthogonal to --kernel by construction.

   Structural half, and it is independent of the ISA argument:
   lz_row_pick (src/ops_matmul.c) never indexes LZ_ROW_SSE_I or
   LZ_ROW_SSE_A, and ops.h has no LZ_KERNEL_SSE value, so `--kernel sse`
   cannot be spelled. A path the CLI cannot select cannot be
   compared, so a kernel written into those two slots would be dead on
   arrival.

   t2/g128. The argument is at LZ_G128_T2 above: q6_1 unpacks the same
   2-bit plane and does have a group kernel because it hoists two mask
   constants across four sub-blocks; t2's unpack is a zero register and
   three immediate shifts, so hoisting trades pxor for movq one for one
   and the group kernel would emit what four separate calls emit.

   ---- TODO -------------------------------------------------------------

   None. Every row unit carries both ARM cells: the C tier and the
   hand-written one, for all five formats (src/ops_arm.c, plus ternary
   in src/ops_t2_arm.c). The eleven remaining empty cells are the sseI
   and sseA columns and t2/g128, all IMPOSSIBLE and all audited above.

   A new format arrives with two empty ARM cells, and filling them is
   two edits, not one: the kernel, and the line in lz_matmul_xq_nt that
   lets its dtype past. That function carries no interception - the list
   of dtypes needing one is empty - so a new format reaches its impl
   with an all-NULL row table and falls back from inside it instead.
   Register the cells as TODO and add the
   interception for that one dtype until the pair exists - otherwise
   the two tiers differ by which fallback they fell through rather
   than by the kernel under test. */
#define LZ_KM_IMPOSSIBLE 1
#define LZ_KM_TODO       2
#define LZ_KM_DECLINED   3
/* NOT APPLICABLE, which is not the same empty as impossible and was
   spelled the same for as long as there were only three states.
   IMPOSSIBLE means the instruction set cannot express the work.
   NA means this COLUMN does not ask a question about this ROW - the
   g128 column is the matmul row kernel's 128-element group hoist, and
   an attention weighted sum whose group is the context length has no
   answer to give it, wrong or right.
   Keeping them apart matters because the headline count is read as a
   work list: eleven of the forty-nine "impossible" cells were category
   errors, so the ISA-blocked surface read 29% larger than it is, and
   nine of the eleven can never be implemented because there is nothing
   there to implement. */
#define LZ_KM_NA         4

/* g128's gcc expectation follows the knob rather than being hand-
   declared: -DLZ_G128_GCC=0 is the documented A/B control arm, the row
   functions really do revert to the per-32 path, and a control build
   that trips the gate is a control build nobody runs - an A/B arm
   with no gate rots. Watcom's group kernels
   have no knob and are always expected. */
#if defined(LZ_G128_GCC) && LZ_G128_GCC
#define LZ_KMB_G128 (LZ_KMB_GCC | LZ_KMB_WAT)
#else
#define LZ_KMB_G128 LZ_KMB_WAT
#endif /* LZ_G128_GCC */

/* A UNIT is one dispatchable kernel family - the row axis. Every unit
   today is a matmul row kernel and is named for its weight format, so
   the grid prints one line per unit and the two look like the same
   thing. They are not, and the ARM columns are why: `--kernel arm-c` /
   `arm-asm` has exactly ONE consumer (matmul_t2_impl's ternary row,
   src/ops_matmul.c). The moment a second operator gets an ARM tier -
   docs/arm-asm-audit.md ranks the candidates by calls per token - a
   single global enum value cannot say WHICH operator has the
   assembly -
   one LZ_KERNEL_MMX covering both Watcom assembly and gcc intrinsics
   would make lz_kernel_name unable to answer what ran. The registry therefore
   keys on the unit, not on the build.

   LZ_KM_ROW units print in the grid. LZ_KM_OP units are registry-only:
   the grid's six characters ARE the six LZ_ROW_* slots, and an operator
   whose dispatch table has a different shape does not belong in them.

   Adding a unit is one line in km_unit, one row in km_reg and one arm
   in km_present. Nothing in build/kernel_matrix_gate.sh changes. */
#define LZ_KM_ROW 0
#define LZ_KM_OP  1

#define LZ_KM_NUNIT 17
#define LZ_KM_NCOL 10
/* The one unit with ARM kernels, by index - km_tab and km_present both
   need it, and a named constant is what stops a reorder of km_unit from
   silently moving them onto another format. */
#define LZ_KM_UNIT_T2 4
/* The units with ARM row kernels, same reason: km_arm_row reads them,
   and a named constant is what stops a reorder of km_unit from moving
   an ARM cell onto a format that has none. */
#define LZ_KM_UNIT_Q16 3
#define LZ_KM_UNIT_Q8  0
#define LZ_KM_UNIT_Q41 1
#define LZ_KM_UNIT_Q61 2
/* The operator units. Registry-only (LZ_KM_OP), same reason as above:
   km_arm_asm reads them by name, so a reorder of km_unit cannot move a
   filled cell onto an operator that has no assembly. */
#define LZ_KM_UNIT_I32F  5
#define LZ_KM_UNIT_Q8RND 6
#define LZ_KM_UNIT_EXPFX 7
#define LZ_KM_UNIT_ATDOT 8
#define LZ_KM_UNIT_WSUMP 9
#define LZ_KM_UNIT_SIGQ  10
#define LZ_KM_UNIT_NRMSS 11
/* The two Hadamard transforms. TWO units, not one: they are separate
   bodies in separate files serving separate tiers (lz_fwht_i32 is the
   shipping path's, lz_fwht the float reference's), and one row would
   let a kernel appearing for either report the pair as covered. */
#define LZ_KM_UNIT_FWHTI 12
#define LZ_KM_UNIT_FWHTF 13
/* lz_matmul's F32-weight row. A unit of its own rather than a sixth
   format row: the five format rows are selected through LZ_ROW_*, and
   this one is reached by lz_matmul_xq_nt's f32 short-circuit before any
   dtype dispatch runs - a different mechanism, so a different row. */
#define LZ_KM_UNIT_F32MM 14
/* The two largest float census sites, which had no row until now.
   recur is the GDN recurrence (LZ_FC_RECUR, 26.7% of per-token float);
   epi is the matmul epilogue (LZ_FC_EPI, 10.0%). Their absence was not
   a statement that the operators lack kernels - recur has four x86
   bodies - it was that nothing in the table could say either way. */
#define LZ_KM_UNIT_RECUR 15
#define LZ_KM_UNIT_EPI   16
/* Column 0 is ref and columns 7..9 are armC/armA/g128; columns 1..6 are
   the six LZ_ROW_* slots in slot order, so column c reads tab[c - 1]. */
#define LZ_KM_COL_ROW0 1
#define LZ_KM_COL_ARMC 7
#define LZ_KM_COL_ARMA 8
#define LZ_KM_COL_G128 9

static const struct { const char *name; unsigned char kind; }
km_unit[LZ_KM_NUNIT] = {
    { "q8_0",  LZ_KM_ROW }, { "q4_1", LZ_KM_ROW }, { "q6_1", LZ_KM_ROW },
    { "q16_0", LZ_KM_ROW }, { "t2",   LZ_KM_ROW },
    { "i32f",  LZ_KM_OP  }, { "q8rnd", LZ_KM_OP },
    { "expfx", LZ_KM_OP  }, { "atdot", LZ_KM_OP }, { "wsump", LZ_KM_OP },
    { "sigq",  LZ_KM_OP  }, { "nrmss", LZ_KM_OP },
    { "fwhti", LZ_KM_OP  }, { "fwhtf", LZ_KM_OP },
    { "f32mm", LZ_KM_OP },
    { "recur", LZ_KM_OP  }, { "epi",   LZ_KM_OP }
};
static const char *const km_col[LZ_KM_NCOL] = {
    "ref", "mmxI", "sseI", "sse2I", "mmxA", "sseA", "sse2A",
    "armC", "armA", "g128"
};

/* Printed, so one line each; the audit above is the long form. */
/* The standard an empty ISA cell has to meet, because two weaker ones
   were used here before it and both let a writable kernel go unwritten:
   the tier must be unable to express ANY PART of the work. Partial
   relief obliges a kernel - a loop that goes half as fast in SIMD and
   finishes in scalar is still a cell, and "an SSE1 kernel would just be
   an MMX kernel with mulps in front" is a sentence about taxonomy that
   cost nrmss its two cells for as long as it stood.
   AND "THE COMPILER ALREADY DOES IT" IS NOT A REASON. gcc -O2 does
   vectorise the wsum chunk fold on x86-64, and that cell is written
   anyway: an auto-vectorisation has no name the matrix can register, no
   symbol a cross-toolchain comparison can reach, and it disappears when
   a compiler version or a flag moves. It is not a kernel, it is a
   compiler's current mood.
   So o3_probe's zero at -march=pentium3 is evidence in ONE direction
   only. It cannot license an empty cell; only the instruction list can.

   Confirmed by the compiler, not only by the instruction audit. gcc
   -O3 on the same C, build/o3_probe.sh: ALL FIFTEEN probed units emit
   zero packed ARITHMETIC at -m32 -march=pentium3, every one of them
   with a non-empty body, so the zero is the vectoriser declining and
   not the probe missing its symbol. The five weight formats these
   cells belong to are among them - q8_0 at 388 instructions, q4_1 at
   340 - as are lz_attn_score_q8 (393) and lz_attn_wsum_q8 (538), which
   emit 63 and 73 at -march=x86-64.

   THE FIGURE READ 151 AND 122 HERE for one commit and was wrong: the
   pattern counted movaps and movdqa, which is how gcc copies an xmm
   register in ordinary SCALAR float code. A packed MOVE is not
   evidence of vectorisation - separating the two dropped lz_exp from
   "22 packed" to zero arithmetic in 163 instructions.

   And the instruction list, which is the half that licenses the empty
   cell. What the MMX dot kernels execute is pmaddwd, punpcklbw, psllw,
   psrlw and pxor - measured out of src/ops_kernel_dot_mmx.h, not
   recalled. Against that, every integer instruction SSE1 adds:

     pmaxsw pminsw   a saturating clamp; these loops have none
     pavgb pavgw     rounded average; not a MAC
     pmulhuw         UNSIGNED high multiply; the operands are signed
     psadbw          unsigned byte SAD; not a signed MAC
     pshufw          a lane permute; the shifts here are bit-field
                     extraction inside a lane, not a permute
     pextrw pinsrw   scalar lane access; slower than the load it replaces
     pmovmskb        a mask extract with no consumer here
     maskmovq movntq stores; these kernels accumulate in a register

   None of them applies. The one SSE1 instruction that does is
   prefetchnta, and it is already emitted - 14 in the shipped binary,
   from the row loops in ops_matmul.c under --prefetch - so an SSE1
   cell would be the MMX kernel with nothing added.

   That is the test being met: not "gcc declined", which is only
   evidence, but "the tier has no instruction that touches this work". */
static const char km_why_sse1[] =
    "SSE1 adds no signed 16x16->32 MAC and no integer %xmm; its own MMX "
    "additions (pmaxsw/pshufw/psadbw) are real but are not a MAC";
/* The operator units' x86 and group columns. docs/arm-asm-audit.md is
   an ARM-only audit, so it cannot answer for an x86 tier here and this
   says so rather than guessing 'impossible'. */
/* AUDITED NOW, and the answer is the same for all four of the scalar
   leaves: a SIMD cell needs a caller that hands it more than one value,
   and these have none. lz_i32f, q8_round, lz_exp_fixed and sigmoid_q15
   are each called with one scalar from inside somebody else's loop, and
   the loops that ARE vectorizable (the quantize group, the norm element
   loop, the matmul rows) already carry their own kernels - a vector
   twin here would have to be called four values at a time by code that
   does not have four. That is a fact about the call sites, not about
   the ISA, so it does not change with the instruction set and the six
   columns get one reason.

   The two table leaves have a second, independent block: lz_exp_fixed
   and sigmoid_q15 index a table by a value computed per element, and
   neither MMX, SSE1 nor SSE2 has a gather - VPGATHERDD is AVX2. Four
   lanes would have to be scattered back to scalar loads, which is the
   scalar code with extra shuffling in front of it. */
/* This reason initially listed the quantize group, the norm
   element loop and the matmul rows as the only loops that could batch a
   scalar leaf, and concluded that none of them would because each
   carries its own kernel. It missed a fourth: lz_attn_wsum_q8's chunk
   fold, which hands over thirty-two at a time and sits inside a SIMD
   kernel's own epilogue, after the emms. i32f has four x86 cells now.

   What survives is the part that was never about batching. expfx and
   sigq index a table per element, and a per-element table index needs a
   gather - AVX2, which is off the target family entirely. That holds no
   matter who calls them. */
/* SPLIT IN TWO under the standard above. The gather argument closes the
   cells where the LOOKUP is the work, and does not close the ones where
   arithmetic happens around it. sigmoid_q15 ends in a linear
   interpolation - a + (((b - a) * frac) >> 15) - and its two table
   loads stay scalar while that does not have to. lz_exp_fixed has
   arithmetic around its lookup too, but not this arithmetic; see the
   next block, which is where saying otherwise here led.
   THE MULTIPLY IS int16 x int16, which is the fact that decides
   whether this is writable at all, and it was checked rather than
   assumed. g_sigtab holds (int32_t)(s * 32768 + 0.5) for s in [0, 1],
   so a and b are in [0, 32768] and their adjacent difference is in the
   hundreds - 384 steps across the table and a sigmoid slope that never
   exceeds 0.25. frac is (t - idx) * 32768 with t - idx in [0, 1), so
   [0, 32767]. Both operands fit int16, so the product is pmaddwd, which
   is MMX.
   HAD IT BEEN int32 x int32 these cells would be genuinely shut: the
   instruction for that is pmulld and pmulld is SSE4.1, off this target
   family entirely. Worth writing down because reaching for it is the
   obvious first move and it does not exist here.

   ONLY TWO OF THE THREE OPERANDS ARE int16, and mixing that up is the
   trap this note exists to spring. b - a and frac are; `a` ITSELF IS
   NOT - g_sigtab holds up to 32768 exactly (s = 1.0 rounds there, and
   both early exits return it), which is one past INT16_MAX. So the
   sequence keeps a in int32 and adds it with paddd, and only the
   difference and the fraction go through the 16-bit multiply:
     pmullw + pmulhw on (b-a, frac), punpck[lh]wd to rebuild two full
     int32 products, psrad 15, paddd a.
   Eight MMX instructions per four elements against the C's four
   scalar sub/mul/shift/add. Packing `a` into int16 to shorten that
   would silently clamp the one value the table's own endpoint is.

   The win is still small - two lanes on MMX, four on SSE2, against
   scalar loads that stay scalar. Small is not none, and none is the
   bar. */
/* expfx IS NOT THE SAME SHAPE as sigq. lz_exp_fixed does no linear
   interpolation: it is a Q20 Taylor series - s = rq >> 5, then
   cq = QSCALE + ((LN2*s + rnd) >> Q) + ((LN2SQ*s*s + rnd) >> 2Q), then
   prod = (g_exptab[idx] * cq + rnd) >> Q - whose multiplies are
   32 x 32 -> 64, not the 16 x 16 -> 32 that made sigq's tail a pmaddwd.
   That changes which instruction each tier needs:
     SSE2  pmuludq is exactly 32 x 32 -> 64, two lanes, and it needs NO
           sign handling here - every operand is non-negative. r is
           [0,1) by construction, so rq >= 0 and s = rq >> 5 >= 0;
           LN2 and LN2SQ are positive Q20 constants; g_exptab holds
           exp() times QSCALE; cq is QSCALE plus two non-negative
           terms. An earlier version of this note said the sign had to
           be folded in by hand, which would send the next reader down
           a path the arithmetic does not need.
     MMX   has nothing wider than pmaddwd. The product has to be built
           from 16-bit halves with shifts and adds - several
           instructions where SSE2 spends one. Owed, not impossible:
           expressible is the bar, not convenient.

   THE SECOND TAYLOR TERM IS NOT THE 64 x 32 IT LOOKS LIKE, and this
   note said twice that it was. C writes LN2SQ*s*s as
   ((int64)LN2SQ * s) * s, whose first product is 2^33 and whose second
   is therefore 64 x 32 - which pmuludq cannot do. But s = rq >> 5 with
   rq at most 2^20, so s <= 2^15 and s*s <= 2^30 fits int32. Grouped as
   LN2SQ * (s*s) both factors are under 2^32 and pmuludq takes it in
   one, and the two groupings are the SAME NUMBER: integer
   multiplication is associative and exact, and every intermediate here
   is at most 2^48, far inside int64.
   So all four multiplies in this operator are 32 x 32 -> 64:
   LN2*s, s*s, LN2SQ*(s*s), and g_exptab[idx]*cq. On SSE2 that is
   pmuludq four times (two lanes each); on MMX it is the 16-bit-halves
   construction, which is where the cost actually sits.

   The 32-entry table lookup stays scalar on both, same as sigq's. */
/* sigq's SSE1 cells, and this one CLOSES rather than opens. The
   interpolation is psubd, packssdw, pmullw, pmulhw, punpck[lh]wd,
   psrad and paddd - every one of them MMX, at the width MMX has. SSE1
   adds pmaxsw, pminsw, pshufw, pavgw, pmulhuw and psadbw to the MMX
   register file and not one of them appears in that list, so an SSE1
   cell would be the MMX kernel with nothing added. That is the test
   the standard sets, and it is the same instruction-list argument the
   fourteen dot-product cells pass. The wider form is SSE2's, and SSE2
   has its own cell. */
static const char km_why_sigq_sse1[] =
    "SSE1 adds nothing this sequence uses: psubd, packssdw, pmullw, "
    "pmulhw, punpck[lh]wd, psrad and paddd are all MMX, and SSE1's own "
    "additions (pmaxsw/pshufw/psadbw) appear nowhere in it. The wider "
    "form is SSE2's, which has its own cell";
/* km_why_expfx_x86 stood here and is gone with the last cell it
   explained. What it argued is now in the three bodies themselves
   (ops_sse2.c, ops_mmx.c, ops_mmx_sse.c), which is where an
   implemented kernel's reasoning belongs - a reason string is for a
   cell that does NOT exist. The 32-entry table lookup still stays
   scalar at every tier: a per-element index needs a gather and that is
   AVX2, which km_why_sse1's block above covers for both table units. */
/* expfx and sigq only, and the scope line is here because this string
   is wrong for any unit that HAS a group form. i32f's is the wsum
   chunk fold's 32 and q8rnd's is lz_q8round_group_sse's gs, so each of
   those carries a reason naming its own axis instead. For a per-element
   table lookup both clauses below are true: there is no group form, and the
   leaf really has nothing to hoist across. */
static const char km_why_op_g128[] =
    "no 128-element group form exists - a per-element table lookup has "
    "no group to hoist across";
static const char km_why_attn_g128[] =
    "this column is the 128-ELEMENT group; attention's group runs over "
    "context length, which is unbounded, and lz_wsum_group_mmx already "
    "covers it - a different axis, not this cell";
/* norm_ss_fixed walks a whole norm row (up to LZ_NORM_MAX_N), not a
   quantization group, so the 128-element column is not its axis. The
   one group-shaped thing it does - the amax scan - is q8_amax's, and
   q8_amax has its own kernels; billing them here would count them
   twice. */
/* nrmss is the one operator unit that is NOT a one-value leaf, so it
   does not get km_why_op_x86's reason: it is an element loop of the
   same shape lz_quantize_q8 already vectorizes, and whether an x86
   kernel belongs here is a real open question rather than a category
   error. The ARM-only audit cannot answer it either way. */
/* Unlike the scalar leaves this one IS an element loop - x[i]*sc, round
   to int16, accumulate v*v - so the question is which instruction set
   can do all three steps, and the answer splits the six columns rather
   than filling them with one reason.

   SSE2 can, and lz_norm_ss_sse2 is that cell: mulps for the scale,
   cvtps2dq for the round (round-to-nearest under the default MXCSR,
   which is q8_round's own rule), packssdw + pmaxsw for the clamp, and
   pmaddwd for the squares, which is exactly the accumulate this loop
   wants. Gate: .prof/norm_ss_sse2_xcheck.sh.

   SSE1 finishes it too, and lz_norm_ss_sse is that cell: the same five
   steps with cvtps2pi in place of cvtps2dq, so everything after the
   scale lands in %mm and the kernel owes an emms. Four elements a pass
   against eight. This row read "impossible" for as long as it did
   because the reason argued taxonomy - an SSE1/MMX hybrid is still an
   SSE1 cell, it needs SSE1 to run.

   MMX gets a cell too. "MMX alone cannot start it: the scale is a
   FLOAT multiply and MMX is integer-only" answers whether the loop runs
   in ONE pass, which is not what the cell asks. It does not have to.
   lz_norm_ss_pack_mmx takes the int32 an x87 pass
   produced and does the clamp and the pmaddwd sum of squares, which is
   the part MMX can reach; the emms is once per 32 elements rather than
   once per element, which is what separates it from a regression. */

static const char km_why_nrmss_g128[] =
    "this column is the 128-ELEMENT quantization group; norm_ss_fixed's "
    "loop runs over a whole norm row, and the group-shaped part of it "
    "is q8_amax, which carries its own kernels";
/* The Hadamards are the opposite of the scalar leaves above: a butterfly
   loop of pure add/sub over a power-of-two block, which is the shape
   SIMD is best at, and no multiply or constant enters.

   The two transforms take OPPOSITE answers per ISA because one is
   integer and the other is float - which is the whole reason they are
   two units. lz_fwht_i32 is paddd/psubd: MMX does two lanes, SSE2 does
   four, and SSE1 has no integer add at all. lz_fwht is addps/subps:
   SSE1 and SSE2 both do four lanes, and MMX is integer-only. So fwhti
   fills its four x86 cells with TWO bodies and fwhtf with one.

   Neither needs a shuffle. The butterfly at stride `len` pairs y[i]
   with y[i+len], and for every stage with len >= the vector width those
   are two CONTIGUOUS vectors - one load each, one add, one sub, two
   stores. The last stages, where len drops below the vector width, stay
   scalar: an unpack per butterfly would cost more than it saves, and
   they are the cheap ones anyway.

   Both are bit-identical to the C body rather than close, and that is a
   structural claim, not a measurement that happened to come out even:
   each output is one add or one sub of the same pair the scalar loop
   forms, so nothing is reassociated and no constant enters. The gates
   are .prof/fwht_mmx_xcheck.sh and .prof/fwht_f32_xcheck.sh, both with
   a mutation, and .prof/wbuild_xcheck.sh runs the same probes through
   the target compiler. */
static const char km_why_fwhti_sse1[] =
    "impossible: SSE1 has no integer add - paddd is MMX/SSE2. The int32 "
    "butterfly cannot be expressed in xmm under SSE1";
/* Audited with the kernel, which is how the row above it was found to
   be wrong. q8rnd had all six x86 cells marked impossible on the
   scalar-leaf reason, and four of them ship: lz_q8round32_sse
   (ops_mmx_sse.c) and lz_q8round32_simd (ops_sse2.c) on gcc,
   lz_q8round32_sse_asm and lz_q8round32_sse2_asm as #pragma aux on
   Watcom, selected at runtime by lz_q8r_tier. The function is named for
   the thirty-two values it takes at once. Only the MMX pair is
   genuinely out, and for a different reason than the one it carried. */
/* q8rnd DOES have a group form, which the shared operator reason denied
   twice over. lz_q8round_group_sse (src/ops_mmx.h) takes gs and runs
   the whole gs/32-chunk loop - hand it 128 and it runs four chunks. So
   neither "no group form exists" nor "a scalar leaf has no group" is
   true here; the second clause is the one this file already retired at
   km_why_i32f_g128.
   What is true is narrower, and it is what the column asks: this column
   is the matmul row kernel's relationship to a group-of-128 kernel
   (dot128_*), and q8round's group kernel is called by lz_quantize_q8
   per group, not hoisted across sub-blocks by a row. */
static const char km_why_q8rnd_g128[] =
    "this column is the ROW kernel's 128-element group hoist (dot128_*); "
    "q8round does have a group form - lz_q8round_group_sse takes gs and "
    "runs gs/32 chunks - but lz_quantize_q8 calls it per group rather "
    "than a row hoisting it across sub-blocks. A different axis";
static const char km_why_i32f_g128[] =
    "this column is the 128-ELEMENT quantization group; the fold's group "
    "is the wsum chunk's 32, which is where its cells are. Not \"a scalar "
    "leaf has no group\" - it has one, just not this one";
static const char km_why_i32f_mmx[] =
    "impossible: MMX has no integer-to-float conversion at all - the "
    "first one on this ladder is SSE1's cvtsi2ss. lz_i32f_acc32_sse is "
    "that cell";
static const char km_why_fwhtf_mmx[] =
    "impossible: MMX is integer-only and this transform is over floats. "
    "The int32 twin (fwhti) is the one MMX can take";
static const char km_why_fwht_g128[] =
    "this column is the 128-ELEMENT quantization group; the Hadamard's "
    "block is hadamard_o/hadamard_down (up to 512, capped by the "
    "fixed-point headroom) and is a different axis, not this cell";
/* Audited with the kernel. MMX is integer-only and this is a float
   multiply-add, so those two cells cannot exist. SSE1 is where mulps
   and addps arrive, and every SSE2 machine has them - so ONE body
   serves both tiers and the SSE2 cells hold it rather than a copy.
   The ARM cell is lz_matmul_row_arm_asm, a leaf rather than an inline
   block: the fusion's win is per GROUP (one scan, one decline, the
   first group needing no addend at all) and an inline per-element form
   gives it all back in wrapper. 69.8 an element against the C loop's
   78.1, .prof/arm_chain_count.sh mode 27 against 28. */
static const char km_why_f32mm_mmx[] =
    "impossible: MMX is integer-only and this is a float multiply-add. "
    "SSE1 is the first tier that can express it";
static const char km_why_f32mm_g128[] =
    "this column is the 128-ELEMENT quantization group; lz_matmul's "
    "unit of work is a whole weight ROW against a whole activation "
    "vector, a different axis";
/* The four 64-bit-lane cells. SSE1 shares this reason with MMX because
   for THIS operator it adds nothing: its integer additions are pmaxsw,
   pshufw, pextrw and movntq on %mm, none of which is a multiply and
   none of which is a 64-bit add.

   A BODY WAS WRITTEN AND MEASURED, which is why the cell is declined
   rather than impossible - and it is NOT IN THIS TREE. It was built in
   a scratch copy, run against the oracle, and discarded; grep for an
   MMX epi kernel here and there is none. "The body exists" is what this
   said, and a reader taking it at its word would go looking for a file
   to enable. The measurements below stand; the code has to be written
   again. The intrinsics version passes the oracle's
   20106 cases and costs 19 instructions an element on the 32-bit i486
   build against the scalar imul loop's 8: the int32 lane bound forces
   a spill every two elements and gcc stacks the accumulators on top of
   that. A Pentium MMX resolves auto to this tier, so shipping it would
   slow down the machine the tier exists for. The pack is {b,ah}
   against {m,m}, which makes one pmaddwd lane the whole high term;
   Open Watcom's mmintrin.h declares __m64 and none of the intrinsics,
   so the twin has to be a #pragma aux.

   That paragraph is a COMMENT and this is a RUNTIME STRING. The cell
   reason has to fit C89's 509-character literal, which VC++ 4.0 holds
   to - it does not get to carry the whole argument. */
static const char km_why_epi_x86[] =
    "declined: the products are pmaddwd's shape but the ACCUMULATOR is "
    "not. paddq is SSE2 even on %mm, so a 64-bit sum has to leave the "
    "vector unit: |ah*m| reaches 2^30 and two of them overflow an int32 "
    "lane, and a lane takes ONE madd result, not two: 2*2^30 lands exactly "
    "on 2^31. Written and measured, then not shipped: 19 instructions an "
    "element against the scalar loop's 8 - slower than pure C, which is "
    "the one exception (regression), not partial relief";
typedef struct { unsigned char have; unsigned char why; const char *text; } LZKmReg;
#define KM_H(mask) { (unsigned char)(mask), 0, (const char *)0 }
#define KM_I(text) { 0, LZ_KM_IMPOSSIBLE, (text) }
#define KM_T(text) { 0, LZ_KM_TODO, (text) }
#define KM_D(text) { 0, LZ_KM_DECLINED, (text) }
#define KM_N(text) { 0, LZ_KM_NA, (text) }
#define KM_ALL (LZ_KMB_GCC | LZ_KMB_WAT | LZ_KMB_ARM)

/* ref is the one DECLARED-present column with no per-format variation:
   lz_matmul_xq_nt routes LZ_KERNEL_REF to matmul_scalar_ref before any
   dtype dispatch, and matmul_scalar_ref_one branches on all five. It
   cannot go empty for one format only, so it is KM_ALL by construction
   and carries no signal of its own - it is here because the registry
   counts C as a tier and a column left out of the table is a column
   nobody can miss. */
/* FLAT, not [LZ_KM_NUNIT][LZ_KM_NCOL], and the reason is a compiler
   limit rather than taste: Open Watcom accepts fourteen rows of this
   shape and reports "E1175: Too many initializers" on the fifteenth,
   with NUNIT and NCOL both verified correct by #error probes at the
   declaration. One brace level fewer is what it takes. km_at() below is
   the only place the two-dimensional index lives. */
static const LZKmReg km_reg[LZ_KM_NUNIT * LZ_KM_NCOL] = {
/*          ref          mmxI              sseI                sse2I             mmxA              sseA                sse2A             armC                 armA                 g128            */
/* q8_0 */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* q4_1 */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* q6_1 */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* q16_0*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* t2   */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_ARM),    KM_H(LZ_KMB_G128),
/* operator units (LZ_KM_OP): registry only, no grid line. ref is the
   shared C body, which is also what --kernel arm-c runs - so armC is
   present on ARM for the same reason ref is present everywhere. */
/* i32f */ KM_H(KM_ALL), KM_I(km_why_i32f_mmx), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_I(km_why_i32f_mmx), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_i32f_g128),
/* q8rnd*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_q8rnd_g128),
/* expfx*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_op_g128),
/* The two attention MAC units DO have x86 kernels, and they are
   macro-selected rather than table-selected, so km_op_present answers
   for them. atdot's assembly is q8_0's leaf, reused rather than copied
   (docs/arm-asm-audit.md section 6); wsump's is its own. That holds at
   BOTH x86 tiers now - the SSE2 cells were empty not because SSE2 could
   not express these but because the two call sites named the MMX body
   with no tier test, so no SSE2 twin was selectable. Gate:
   .prof/attn_sse2_xcheck.sh (kernel against kernel against C) plus
   build/kernel_isa_identity_gate.sh (whether the dispatch reaches it). */
/* atdot*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_attn_g128),
/* wsump*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_attn_g128),
/* sigq */ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_sigq_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_sigq_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_op_g128),
/* nrmss*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_nrmss_g128),
/* Registering these two rows is what makes a gate able to tell "no
   assembly" from "no such operator": without them src/fwht.c could be
   deleted and not a single cell would change. */
/* fwhti*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_I(km_why_fwhti_sse1), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_I(km_why_fwhti_sse1), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM),       KM_N(km_why_fwht_g128),
/* fwhtf*/ KM_H(KM_ALL), KM_I(km_why_fwhtf_mmx), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_I(km_why_fwhtf_mmx), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM),          KM_N(km_why_fwht_g128),
/* f32mm*/ KM_H(KM_ALL), KM_I(km_why_f32mm_mmx), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_I(km_why_f32mm_mmx), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM),        KM_N(km_why_f32mm_g128),
/* recur's SSE1 cell is NOT km_why_sse1. That string fills the SSE1
   column for every weight-format row above and copying it down is the
   reflex; lz_p2_rows_sse (ops_mmx_sse.c) exists, and a false absent is
   worse than an empty cell because the gate then enforces it. The
   Watcom SSE columns stay with the gcc class: LZ_P2_SSE_EXTERN and
   LZ_P2_SSE2_EXTERN are what km_op_present can key off, and the SSE2
   one is declared under !__WATCOMC__. */
/* recur*/ KM_H(KM_ALL), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_GCC), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_op_g128),
/* epi  */ KM_H(KM_ALL), KM_D(km_why_epi_x86), KM_D(km_why_epi_x86), KM_H(LZ_KMB_GCC), KM_D(km_why_epi_x86), KM_D(km_why_epi_x86), KM_H(LZ_KMB_WAT), KM_H(LZ_KMB_ARM), KM_H(LZ_KMB_ARM), KM_N(km_why_op_g128)
};

#if defined(__WATCOMC__)
#define LZ_KMB_THIS LZ_KMB_WAT
#elif defined(__arm__)
#define LZ_KMB_THIS LZ_KMB_ARM
#else
#define LZ_KMB_THIS LZ_KMB_GCC
#endif /* __WATCOMC__ */

/* Q16's table is the one row inside the MMX block; the other four are
   unguarded and come out all-NULL on a target with no SIMD, which is
   the honest answer there. */
static const lz_rowfn *km_tab(int f) {
    switch (f) {
    case 0: return LZ_ROW_Q8;
    case 1: return LZ_ROW_Q41;
    case 2: return LZ_ROW_Q61;
#if defined(__WATCOMC__) || defined(LZ_DOT_MMX_EXTERN) || defined(__arm__)
    case 3: return LZ_ROW_Q16;
#else
    case 3: return (const lz_rowfn *)0;
#endif /* __WATCOMC__ || LZ_DOT_MMX_EXTERN || __arm__ */
    case LZ_KM_UNIT_T2: return LZ_ROW_T2;
    /* An operator unit has its own table with its own shape; it gets an
       arm here when it gets a row in km_reg, not a share of this one. */
    default: return (const lz_rowfn *)0;
    }
}

#if defined(__arm__)
/* Does THIS build carry an ARM row kernel for unit f? The list grows as
   cells get filled, in step with lz_matmul_xq_nt's dtype list - that
   line is what makes a filled cell actually run, and the two going out
   of step is a cell that reports present and never executes. */
static int km_arm_row(int f) {
    (void)f;
    /* Every unit, and deliberately without enumerating them: the C tier
       IS the shared body every build carries, so `--kernel arm-c` runs
       the same code `--kernel ref` runs. A list here would have to be
       extended with each new unit and would say nothing a reader could
       check - it was already out of date once. */
    return 1;
}

/* The asm column is answered per unit, not per build: each format's
   hand-written leaf sits behind its own file's EXTERN macro, so a
   toolchain that compiles one and not the other is a state this can
   report rather than one it averages away. */
static int km_arm_asm(int f) {
#if defined(LZ_T2_ARM_ASM_EXTERN)
    if (f == LZ_KM_UNIT_T2) return 1;
#endif /* LZ_T2_ARM_ASM_EXTERN */
#if defined(LZ_ARM_ASM_EXTERN)
    if (f == LZ_KM_UNIT_Q16 || f == LZ_KM_UNIT_Q8 ||
        f == LZ_KM_UNIT_Q41 || f == LZ_KM_UNIT_Q61) return 1;
#endif /* LZ_ARM_ASM_EXTERN */
#if defined(LZ_ARM_PRIM_ASM)
    /* sigq is both sigmoid_q15 entries: one table core, two coordinate
       front ends, one macro. They share a cell because they share the
       thing that makes the cell - splitting them would register two
       rows for one body. */
    if (f == LZ_KM_UNIT_I32F || f == LZ_KM_UNIT_Q8RND ||
        f == LZ_KM_UNIT_EXPFX || f == LZ_KM_UNIT_SIGQ) return 1;
    /* fwhtf's cell is lz_faddsub_arm_asm, an inline block like the four
       above rather than a loop leaf: the butterfly's operands come from
       two strides of the caller's array, so a leaf would pass them
       through memory and lose what the shared unpack bought. */
    if (f == LZ_KM_UNIT_FWHTF) return 1;
#endif /* LZ_ARM_PRIM_ASM */
#if defined(LZ_ARM_ASM_EXTERN)
    /* Both attention MACs, and atdot's cell IS q8_0's leaf - the reuse
       section 6 of the audit asked for, not a second copy. nrmss is a
       loop, so it is a leaf function in src/ops_arm.c rather than an
       inline block: its per-call setup amortizes over a whole row. */
    if (f == LZ_KM_UNIT_ATDOT || f == LZ_KM_UNIT_WSUMP ||
        f == LZ_KM_UNIT_NRMSS || f == LZ_KM_UNIT_F32MM) return 1;
    /* epi's is another loop leaf in src/ops_arm.c. It needs no split of
       its own: SMLAL is the whole operator, which is why this column
       filled before any of the six x86 ones. */
    if (f == LZ_KM_UNIT_EPI) return 1;
    /* fwhti's is a loop leaf in src/ops_arm.c; the float twin's is the
       inline block registered above, under a different guard. */
    if (f == LZ_KM_UNIT_FWHTI) return 1;
    /* recur's is lz_gdn1_x4_arm_asm, the pass-1 4-lane contraction leaf
       (docs/arm-asm-audit.md D2). */
    if (f == LZ_KM_UNIT_RECUR) return 1;
#endif /* LZ_ARM_ASM_EXTERN */
    (void)f;
    return 0;
}
#endif /* __arm__ */

static int km_g128(int f) {
    switch (f) {
    case 0:  return LZ_G128_Q8;
    case 1:  return LZ_G128_Q41;
    case 2:  return LZ_G128_Q61;
    case 3:  return LZ_G128_Q16;
    case LZ_KM_UNIT_T2: return LZ_G128_T2;
    /* Operator units have no group kernel; the registry says so and
       this must agree, or the cell reports present-and-unregistered. */
    default: return 0;
    }
}

/* The x86 tiers of the two attention MAC units. They have no dispatch
   table - both call sites pick their kernel with a #if, so the presence
   answer has to come from the same macros the #if reads, and it does. */
static int km_op_present(int f, int c) {
    /* The GDN recurrence. Four bodies, and until LZ_P2_SSE_EXTERN landed
       the SSE1 one had no macro to key a cell off - which is why an
       operator with x86 assembly showed as having none.
       Columns by build class, like FWHTI below: LZ_P2_MMX_EXTERN and
       LZ_P2_SSE_EXTERN are defined for both toolchains, so naming the
       gcc column unconditionally would make a Watcom build report the
       gcc cell present. */
#if defined(LZ_P2_MMX_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_P2_MMX_EXTERN */
#if defined(LZ_P2_SSE_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_P2_SSE_EXTERN */
#if defined(LZ_P2_SSE2_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_RECUR && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_P2_SSE2_EXTERN */
#if defined(LZ_EPI_SSE2_EXTERN)
    /* Both toolchains have an epi SSE2 body, so this one IS split by
       build class - the macro no longer is. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_EPI && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_EPI && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_EPI_SSE2_EXTERN */

#if defined(LZ_FWHT_MMX_EXTERN)
    /* The Hadamard's MMX stage kernel. LZ_FWHT_MMX_EXTERN is defined for
       BOTH toolchains, unlike the dot/wsum _EXTERNs below - so the COLUMN
       has to be chosen by build class here, or a gcc build reports the
       Watcom cell present and the matrix gate fails with "present but not
       registered for this build class", which is exactly what it did. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_FWHTI && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_FWHTI && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_FWHT_MMX_EXTERN */
#if defined(LZ_FWHT_SSE2_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_FWHTI && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_FWHTI && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_FWHT_SSE2_EXTERN */
#if defined(LZ_NORM_SS_SSE2_EXTERN)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_NORM_SS_SSE2_EXTERN */
#if defined(LZ_NORM_SS_SSE_EXTERN)
    /* The SSE1 cell, a body of its own rather than a second column on
       the SSE2 one: cvtps2pi puts the pack, the clamp and the
       multiply-add in %mm, four elements a pass against eight. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_NORM_SS_SSE_EXTERN */
#if defined(LZ_MATMUL_F32_SSE_EXTERN)
    /* FOUR cells from one body, unlike every other entry here: the
       kernel is SSE1 (mulps/addps), and an SSE2 machine runs the same
       instructions - so the SSE2 column holds this body rather than a
       second one, and lz_matmul selects it at either tier. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_F32MM &&
        (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A ||
         c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A)) return 1;
#else
    if (f == LZ_KM_UNIT_F32MM &&
        (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I ||
         c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I)) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_MATMUL_F32_SSE_EXTERN */
    /* The two MMX PACK HALVES, which are cells the "part of it counts"
       standard opened and which now exist. Both take the int32 the
       caller's x87 pass produced and do the part MMX can reach:
       q8round the clamp and the two packs, nrmss the clamp and the
       pmaddwd sum of squares. Registered from the same macros the call
       sites test, so a registration cannot outlive its kernel. */
#if defined(LZ_HAVE_Q8R_PACK_MMX)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_Q8R_PACK_MMX */
#if defined(LZ_HAVE_NORM_SS_PACK_MMX)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_NRMSS && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_NORM_SS_PACK_MMX */
    /* sigq's interpolation tail, and expfx's Q20 Taylor beneath it.
       SEPARATE MACROS even though the two units are neighbours in the
       registry: they are different kernels with different wiring, and
       registering both from one flag would let either cell outlive its
       own body. expfx's SSE2 pair is wired (lz_exp_fixed_run, reached
       from softmax); its MMX and SSE1 cells are still owed. */
#if defined(LZ_HAVE_LERP_Q15_MMX)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_SIGQ && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_SIGQ && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_LERP_Q15_MMX */
#if defined(LZ_HAVE_LERP_Q15_SIMD)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_SIGQ && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_SIGQ && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_LERP_Q15_SIMD */
#if defined(LZ_HAVE_EXP_Q20_SIMD)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_EXP_Q20_SIMD */
#if defined(LZ_HAVE_EXP_Q20_MMX)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#else
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_EXP_Q20_MMX */
#if defined(LZ_HAVE_EXP_Q20_SSE)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_EXPFX && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_EXP_Q20_SSE */

    /* i32f, the attention wsum chunk fold. Four cells from two bodies
       per toolchain: SSE1 converts one value at a time (cvtsi2ss,
       lz_i32f_acc32_sse) because the packed convert at that tier reads
       an %mm register the caller has just emms'd away from, and SSE2
       converts four (cvtdq2ps, lz_i32f_acc32_simd). */
#if defined(LZ_HAVE_I32FACC_SSE)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_I32F && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_I32F && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_I32FACC_SSE */
#if defined(LZ_HAVE_I32FACC_SSE2)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_I32F && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_I32F && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_I32FACC_SSE2 */

    /* q8rnd, whose four x86 cells were registered impossible on the
       scalar-leaf reason until they were audited against the kernels.
       TWO bodies here, not one shared across tiers like f32mm above:
       SSE1 rounds thirty-two values through two MMX registers
       (lz_q8round32_sse / lz_q8round32_sse_asm) and SSE2 does the same
       thirty-two in xmm (lz_q8round32_simd / lz_q8round32_sse2_asm), so
       each tier holds its own. lz_q8r_tier picks between them. */
#if defined(LZ_HAVE_Q8R_SSE)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A) return 1;
#else
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_Q8R_SSE */
#if defined(LZ_HAVE_Q8R_SIMD)
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (f == LZ_KM_UNIT_Q8RND && c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_HAVE_Q8R_SIMD */
#if defined(LZ_FWHT_F32_SSE_EXTERN)
    /* Four cells from one body, same as f32mm just above and for the
       same reason - addps/subps is SSE1, and lz_fwht offers it at both
       the mmx and sse2 tiers. Note the asymmetry with fwhti two entries
       up, which needs TWO bodies for its four cells because MMX's paddd
       is two lanes and SSE2's is four. */
#if defined(__WATCOMC__)
    if (f == LZ_KM_UNIT_FWHTF &&
        (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_A ||
         c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A)) return 1;
#else
    if (f == LZ_KM_UNIT_FWHTF &&
        (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE_I ||
         c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I)) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_FWHT_F32_SSE_EXTERN */
    if (f != LZ_KM_UNIT_ATDOT && f != LZ_KM_UNIT_WSUMP) return 0;
#if defined(LZ_ATTN_SSE2_EXTERN)
    /* Both attention MACs at the SSE2 tier. Defined for BOTH toolchains
       like LZ_FWHT_MMX_EXTERN above, so the column is chosen by build
       class rather than by the macro. atdot's SSE2 cell is a REUSE of
       the matmul's own Q8 leaf - the same relationship its ARM cell has
       to q8_0's - while wsump's is a new body; the matrix records
       PRESENCE, and a reused body is present. */
#if defined(__WATCOMC__)
    if (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_A) return 1;
#else
    if (c == LZ_KM_COL_ROW0 + LZ_ROW_SSE2_I) return 1;
#endif /* __WATCOMC__ */
#endif /* LZ_ATTN_SSE2_EXTERN */
#if defined(__WATCOMC__)
    if (c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_A) return 1;
#endif /* __WATCOMC__ */
#if defined(LZ_DOT_MMX_EXTERN)
    if (f == LZ_KM_UNIT_ATDOT && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* LZ_DOT_MMX_EXTERN */
#if defined(LZ_WSUM_MMX_EXTERN)
    if (f == LZ_KM_UNIT_WSUMP && c == LZ_KM_COL_ROW0 + LZ_ROW_MMX_I) return 1;
#endif /* LZ_WSUM_MMX_EXTERN */
    (void)c;
    return 0;
}

/* Is the cell in THIS binary? Columns 1..6 are read out of the real
   dispatch tables, so deleting a kernel is visible here; ref, armC,
   armA and g128 are declared, for the reasons noted at their columns. */
static int km_present(int f, int c) {
    const lz_rowfn *tab;
    if (c == 0) return 1;
    if (c >= LZ_KM_COL_ROW0 && c < LZ_KM_COL_ROW0 + LZ_ROW_N) {
        tab = km_tab(f);
        if (!tab) return km_op_present(f, c);
        return tab[c - LZ_KM_COL_ROW0] != 0;
    }
    if (c == LZ_KM_COL_G128) return km_g128(f);
#if defined(__arm__)
    /* The ARM row functions are picked inside each matmul_*_impl, not
       from the LZ_ROW_* tables - those six slots are x86 slots. One
       list, read by both ARM columns: a format whose C tier exists but
       whose asm tier does not is a real state (the asm needs
       __GNUC__), a format with the asm and no C tier is not. */
    if (c == LZ_KM_COL_ARMC) return km_arm_row(f);
    if (c == LZ_KM_COL_ARMA) return km_arm_asm(f);
#else
    (void)f;
#endif /* __arm__ */
    return 0;
}

/* The one place the two-dimensional index lives - see km_reg's own
   comment for why that table is flat. */
static const LZKmReg *km_at(int f, int c) {
    return &km_reg[(size_t)f * LZ_KM_NCOL + (size_t)c];
}

static char km_status(int f, int c) {
    unsigned have = km_at(f, c)->have;
    int here = (have & (unsigned)LZ_KMB_THIS) != 0;
    if (km_present(f, c)) return here ? 'x' : '+';
    if (here) return '!';
    if (have) return 'o';
    if (km_at(f, c)->why == LZ_KM_IMPOSSIBLE) return 'i';
    if (km_at(f, c)->why == LZ_KM_TODO) return 't';
    if (km_at(f, c)->why == LZ_KM_DECLINED) return 'd';
    if (km_at(f, c)->why == LZ_KM_NA) return 'n';
    return '?';
}

const char *lz_kernel_matrix(void) {
    static char buf[512];
    int i, j, p = 0;
    for (i = 0; i < LZ_KM_NUNIT; i++) {
        /* Row-kernel units only: the six characters below ARE the six
           LZ_ROW_* slots. An operator unit prints in the registry. */
        if (km_unit[i].kind != LZ_KM_ROW) continue;
        p += sprintf(buf + p, "%-6s", km_unit[i].name);
        for (j = 0; j < LZ_ROW_N; j++)
            buf[p++] = km_status(i, LZ_KM_COL_ROW0 + j);
#if defined(__arm__)
        /* Printed only on the build that has them: a matrix of six dots
           is what "this build has no ternary kernel" looks like, and on
           the ARM cross-build that would be false. */
        buf[p++] = ' ';
        buf[p++] = km_status(i, LZ_KM_COL_ARMC);
        buf[p++] = km_status(i, LZ_KM_COL_ARMA);
#endif /* __arm__ */
        buf[p++] = ' ';
        /* 'G' rather than 'x' when present: the column is a declaration,
           not a tier, and the two should not read alike. */
        buf[p] = km_status(i, LZ_KM_COL_G128);
        if (buf[p] == 'x') buf[p] = 'G';
        p++;
        buf[p++] = '\n';
    }
    buf[p] = '\0';
    return buf;
}

unsigned lz_kernel_build_class(void) { return (unsigned)LZ_KMB_THIS; }

int lz_kernel_cell(int i, LZKmCell *out) {
    int f, c;
    if (!out || i < 0 || i >= LZ_KM_NUNIT * LZ_KM_NCOL) return 0;
    f = i / LZ_KM_NCOL;
    c = i % LZ_KM_NCOL;
    out->unit   = km_unit[f].name;
    out->tier   = km_col[c];
    out->why    = km_at(f, c)->text;
    out->have   = km_at(f, c)->have;
    out->status = km_status(f, c);
    out->kind   = km_unit[f].kind;
    return 1;
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
