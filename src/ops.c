#include <stddef.h>   /* size_t: used a lot here for pointer offsets; math.h does not guarantee it */
#include <float.h>    /* FLT_MAX: overflow guard for quant reciprocal */
#include <math.h>
#include <string.h>   /* memcpy: unaligned loads in the MMX/SSE2 kernels */

#include "ops.h"
#include "err.h"
#include "cpucheck.h"   /* LZ_CPUID1_EDX_AUX - shared CPUID leaf 1 EDX asm */

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
#endif

void lz_rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    float ss = 0.0f;
    float inv;
    int i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    inv = lz_rsqrt(ss / (float)n + eps);
    /* (1 + w), not w: see the header */
    for (i = 0; i < n; i++) o[i] = x[i] * inv * (1.0f + w[i]);
}

void lz_rmsnorm_gated(float *o, const float *x, const float *g,
                          const float *w, int n, float eps) {
    float ss = 0.0f;
    float inv;
    int i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    inv = lz_rsqrt(ss / (float)n + eps);
    /* plain w here, and the gate applies last */
    for (i = 0; i < n; i++) o[i] = x[i] * inv * w[i] * lz_silu(g[i]);
}

void lz_l2norm(float *o, const float *x, int n, float eps) {
    float ss = 0.0f;
    float inv;
    int i;
    for (i = 0; i < n; i++) ss += x[i] * x[i];
    inv = lz_rsqrt(ss + eps);          /* sum, not mean */
    for (i = 0; i < n; i++) o[i] = x[i] * inv;
}

void lz_softmax(float *x, int n) {
    float mx = x[0], sum = 0.0f;
    int i;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) {
        x[i] = lz_exp(x[i] - mx);
        sum += x[i];
    }
    /* One division instead of n multiplies: x87's FDIV does not
       pipeline, and n divisions are a thousand-plus per token in
       attention scoring. Hardware-independent win, not an approximation. */
    {
        float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        for (i = 0; i < n; i++) x[i] *= inv;
    }
}

float lz_sigmoid(float x) {
    /* Branching avoids exp overflow at large |x|. Both branches take
       args <= 0, landing exactly in lz_exp's approximation contract.
       This is the absolute hot spot for exp: silu goes through it,
       conv1d needs 55K per token. */
    if (x >= 0.0f) {
        float z = lz_exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = lz_exp(x);
        return z / (1.0f + z);
    }
}

float lz_silu(float x) {
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
   region and are unaffected; two fldcw per token is negligible. */
unsigned lz_fpu_float_begin(void) {
#if defined(__WATCOMC__)
    unsigned save = _control87(0, 0);
    _control87(_PC_24, _MCW_PC);
    return save;
#else
    return 0;   /* gcc/SSE already has float semantics */
#endif
}

void lz_fpu_float_end(unsigned save) {
#if defined(__WATCOMC__)
    _control87(save, _MCW_PC);
#else
    (void)save;
#endif
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
#else
    return (float)log1p(exp((double)x));
#endif
}

/* Round-to-nearest (IEEE default round-half-to-even) via a magic-number
   add.

   Adding 1.5x2^23 forces the mantissa to align on integer bits; the
   integer part lands in the low 23 bits and is read straight out of the
   union. Valid for |qv| < 2^22 - quant inputs are scaled within ±127,
   lz_exp's exponent within ±127, all far inside.

   Why not `(int)`: C's float-to-int conversion truncates toward zero
   while x87 defaults to round-to-nearest, so the compiler must rewrite
   the FPU control word around every conversion - an operation that
   serializes the pipeline. Measured on Open Watcom 32-bit: a bare
   `(int)` cast costs 89.8 ns vs 4.0 ns for the magic add - 22x. On the
   dev box's gcc/SSE it is one cvttss2si instruction, invisible, but
   the target is x87, where this difference decides everything (~2.4M
   elements quantized per token).

   The rounding rule thus changes from half-away-from-zero to
   half-to-even. Only inputs exactly on a half-integer differ, and
   quantization noise is 254x larger; the exporter uses `np.rint`, in
   sync with this, unifying the project on IEEE defaults - no more two
   coexisting rules. */
static int q8_round(float qv) {
    union { float f; uint32_t u; } t;
    t.f = qv + 12582912.0f;                 /* 1.5 * 2^23 */
    return (int)(t.u & 0x007FFFFFu) - 0x00400000;
}

/* 1/sqrt(x). Magic-number seed + 2 Newton iterations.

   Uses Lomont's 0x5F375A86 rather than Quake III's 0x5F3759DF: both
   were swept over the full normal range - 1.7512e-3 vs 1.7521e-3 with
   one Newton, 4.6994e-6 vs 4.7012e-6 with two. Lomont's is genuinely
   better, though after two iterations float rounding dominates and the
   gap lands in the fourth digit. It costs nothing to switch, so
   switch.

   Why two iterations and not one: 1.75e-3 is already near Q8's
   quantization noise floor of 3.9e-3, and RMSNorm applies this
   coefficient to a WHOLE VECTOR - a systematic scaling error, not
   per-element independent noise, so it accumulates across layers. On
   Watcom, two Newtons cost only 0.2 ns more than one (4.0 vs 3.8); no
   reason to save.

   The speed gain is small (~470 calls per token, saving ~2 us); the
   real reason is unification: once the engine carries all
   transcendental functions, the gcc and Watcom builds are bit-identical. */
float lz_rsqrt(float x) {
#if defined(LZ_EXACT_MATH)
    return 1.0f / (float)sqrt((double)x);
#else
    union { float f; uint32_t u; } t;
    float h = 0.5f * x, y;
    t.f = x;
    t.u = 0x5f375a86u - (t.u >> 1);
    y = t.f;
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y;
#endif
}

float lz_exp(float x) {
#if !defined(LZ_EXACT_MATH)
    /* 2^(k/32), k=0..31. 128 bytes, no pressure on L1.

       Apache Commons Math's FastMath.exp is the same idea (table
       absorbs the high fractional bits, polynomial only the residue),
       but for double <0.5 ULP it uses ~40KB of tables (24KB int +
       16KB frac). On a PII with only 16KB L1 that would evict the
       weights being streamed, and our noise floor is Q8's 3.9e-3 - that
       precision is not needed.

       Growing the table to 128 entries squeezes the residual
       polynomial's fit error from 8.5e-8 to 1.3e-9, but the measured
       total error near zero only drops from 3.20e-7 to 2.38e-7 - the
       dominant terms are the table entries themselves and two float
       multiply roundings, not the polynomial. Going further would
       require storing the table as double, and the size is back to KBs. */
    static const float LZ_EXP_TAB32[32] = {
        1.000000000f, 1.021897149f, 1.044273782f, 1.067140401f,
        1.090507733f, 1.114386743f, 1.138788635f, 1.163724859f,
        1.189207115f, 1.215247360f, 1.241857812f, 1.269050957f,
        1.296839555f, 1.325236643f, 1.354255547f, 1.383909882f,
        1.414213562f, 1.445180807f, 1.476826146f, 1.509164428f,
        1.542210825f, 1.575980845f, 1.610490332f, 1.645755478f,
        1.681792831f, 1.718619298f, 1.756252160f, 1.794709075f,
        1.834008086f, 1.874167634f, 1.915206561f, 1.957144124f,
    };


    /* Schraudolph-style: add x·log2e·2^23 directly into the float's
       exponent field; one construction yields "2^n x linearly
       interpolated mantissa"; a 4th-order polynomial fixes the ratio
       of the interpolation to the true 2^f.

       Key point: the correction factor's argument needs NO recompute -
       the low 23 bits of the constructed integer ARE the fractional
       part; reassembling them into a float with exponent 127 gives
       u = 1+f in [1,2), all bit ops, skipping the "take int part ->
       back to float -> subtract" round trip.

       Measured on Open Watcom / x87 (300 rounds x 200K): this form
       8.8 ns/call, libm exp 11.5 ns, the alternative "f = y - (float)n
       + 5th-order Taylor" 13.6 ns. Raising the correction polynomial from
       2nd to 4th order costs nothing extra (7.3 -> 8.8 ns while error
       drops 0.644% -> 0.0258%), so take 4th order directly.

       The rounding must be DOWN (the low 23 bits only work as a
       positive fraction), hence subtract 0.5 before the magic number;
       the integer magnitude reaches 1e9, beyond the float magic's 2^22
       range, so use double's 1.5x2^52. Relies on little-endian reads
       of double's low 32 bits - the target platforms are x86 only. */
    union { float f; int32_t i; } u, p2n;
    float y32, r, w, c;
    int32_t t, n, idx;

    /* Overflow/underflow guard: beyond this range the constructed bit
       pattern is meaningless */
    if (x > 88.0f)   return FLT_MAX;
    if (x < -87.3f)  return 0.0f;

    /* ALL FLOAT, NO DOUBLE. A double magic (1.5x2^52) cannot be used:
       it would stuff x·log2e·2^23 in one shot, extracting exponent,
       table index and residue together - convenient but requiring a
       53-bit mantissa. On the Win98 side, the x87 precision control
       must be set to 24 bits for bitwise parity with SSE (see
       lz_fpu_float_begin); under that, double is also compressed to 24
       bits and the magic dies - measured: conv1d's silu collapsed
       entirely (rms 0.1091 -> 0.1257).

       The two-level float magic avoids that: round on a 1/32-step scale
       first, getting exponent and table index at once, residue by
       subtraction. |y·32| <= 4064, a thousand-fold margin below the
       float magic's 2^22 limit. The two extra int->float conversions
       are fild, not float->int; they do not trigger control-word
       rewrites. */
    y32 = x * (1.44269504f * 32.0f);          /* log2(e)·32 */
    u.f = y32 - 0.5f + 12582912.0f;           /* 1.5*2^23; the -0.5 turns round-half-away into floor */
    t = (u.i & 0x007FFFFF) - 0x00400000;      /* floor(y·32) */
    n = t >> 5;                               /* arithmetic shift = floor(t/32) */
    idx = t & 31;                             /* always 0..31 in two's complement */
    r = y32 - (float)t;                       /* [0,1) */
    w = 1.0f + r * 0.03125f;                  /* 1 + residual, in [1, 1+1/32) */
    p2n.i = (n + 127) << 23;
    /* 2^r on [1, 1+1/32): 2nd-order fit. The interval is only 1/32
       wide; 2nd order reaches 5e-6 - 57x better than a
       "4th-order over the whole [1,2)" fit at 2.85e-4, and 5% faster
       (Watcom measured 9.63 vs 10.15 ns). */
    c = (0.242844437f * w + 0.207425515f) * w + 0.549730133f;
    return p2n.f * (LZ_EXP_TAB32[idx] * c);
#else
    return (float)exp((double)x);
#endif
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

/* The dispatch section is below in this file; only forward decls here.
   Returns 0 = scalar / 1 = SSE1 (writes MMX, needs emms) / 2 = SSE2. */
static int lz_q8r_tier(void);
static int q8r_have_simd(void);

/* int32 -> float. **Every int32-to-float conversion in the engine goes
   through this**, not a bare `(float)v` - unified by construction rather
   than policed case by case. Definition and rationale further down.

   Why unify instead of gating: `lz_i32f(v)` is BIT-IDENTICAL to `(float)v`
   whenever |v| < 2^24 (the split is exact, x4096 is a power of two, and
   the sum is exactly representable), so there is nothing to lose. Deciding
   per call site "this accumulator stays under 2^24" is what left Q8_0
   using a bare cast with a 1.32e8 bound while its own scalar reference
   used lz_i32f - a cross-compiler divergence waiting for the right
   weights. Margin arguments rot; a uniform rule does not. */
static float lz_i32f(int32_t v);

/* The group scale, shared by lz_quantize_q8 and lz_gdn_quantize_2p.
   Writes *sg and *inv, returns the `fast` flag.

   THE POINT IS THAT THE FLOOR LIVES IN ONE PLACE. Both quantizers need
   this whole block - integer-domain absmax, amax*(1/127), the all-zero
   branch, the 127/amax overflow fallback - and LZ_Q8_MIN_SCALE (the
   subnormal floor that makes Watcom and gcc agree, ops.h) must be
   enforced in each. A constant whose whole job is to be the SAME
   everywhere is the worst thing to keep two copies of.

   Everything here is character-for-character what both copies had, in
   the same order, so this is a pure refactor and the bit-hash baseline
   must not move. */
static int q8_group_scale(const float *grp, int gs, float *sg, float *inv) {
    union { float f; uint32_t u; } bit;
    uint32_t am = 0;
    float amax;
    int k, fast;
    for (k = 0; k < gs; k++) {
        uint32_t a;
        bit.f = grp[k];
        a = bit.u & 0x7FFFFFFFu;
        if (a > am) am = a;
    }
    bit.u = am;
    amax = bit.f;
    if (amax > 0.0f) {
        *sg = amax * (1.0f / 127.0f);
        *inv = 127.0f / amax;
        fast = (*inv <= FLT_MAX);
        if (*sg < LZ_Q8_MIN_SCALE) {
            *sg = 1.0f;
            *inv = 0.0f;
            fast = 1;
        }
    } else {
        *sg = 1.0f;
        *inv = 0.0f;
        fast = 1;
    }
    return fast;
}

void lz_quantize_q8(const float *x, int n, int gs, int8_t *q, float *s) {
    int g, k;
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
    /* The rounding tier is selected once for the whole call, not per group */
    int tier = ((gs & 31) == 0) ? lz_q8r_tier() : 0;
#endif
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
#ifdef LZ_HAVE_Q8R_SIMD
                if (tier == 2) {
                    for (k = 0; k + 31 < gs; k += 32)
                        lz_q8round32_simd(grp + k, out + k, &inv);
                } else
#endif
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
                    _mm_empty();
                }
            } else
#endif
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

#include "ops_kernel_dot.h"

/* ---------------------------------------------- runtime kernel dispatch

   One Win98 binary must run on both a PII (MMX only) and a T42/Pentium M
   (SSE2), so both asm kernels are linked in and selected once at startup
   via CPUID.

   **The dispatch point is at the matmul level**: one branch per matmul,
   after which the 32-element kernel inside the row loop is a DIRECT call.
   Dispatching deeper would pay one indirect call per 32 MACs — exactly
   what this kernel design exists to avoid. */
static int g_kernel = 0;                    /* 0 = not yet selected */

#if defined(__WATCOMC__)
/* CPUID leaf 1 EDX: bit 23 = MMX, bit 25 = SSE, bit 26 = SSE2. One
   symbol, one implementation, in cpucheck.c - this file calls it through
   the cpucheck.h declaration. A real __asm-block function, not a
   duplicated #pragma aux (see cpucheck.h for why those two cannot share
   a body across files). */
#endif

/* ---- prefetch tier: ORTHOGONAL to the kernel tier -----------------------
   The kernel tier picks instruction-set width (MMX vs SSE2 integer);
   the prefetch tier picks the memory hint. They are not the same thing,
   and tying them into one enum would miss a whole family of machines:
   K6-2 / K7 have MMX only (no SSE) but DO have 3DNow!'s prefetch, which
   beats the PII's dummy load. Tying prefetch to "has SSE" would leave
   them with the dummy load. */
#define LZ_PF_NONE 0
#define LZ_PF_LOAD 1     /* PII: no prefetch instruction, dummy load only */
#define LZ_PF_NTA  2     /* PIII onward: prefetchnta (weights used once per token) */
#define LZ_PF_AMD  3     /* K6-2 / K7: 3DNow! prefetch */
static int g_pf = -1;

#if defined(__WATCOMC__)
/* CPUID extended leaf 0x80000001's EDX; bit 31 = 3DNow! (implies
   prefetch/prefetchw). Must first query 0x80000000's max extended leaf,
   else old CPUs return leaf 1's content. */
extern unsigned lz_cpuid_ext_edx(void);
#pragma aux lz_cpuid_ext_edx =      ".586"                          "push ebx"                      "mov  eax, 80000000h"           "cpuid"                         "cmp  eax, 80000001h"           "jb   short no_ext"             "mov  eax, 80000001h"           "cpuid"                         "mov  eax, edx"                 "jmp  short done_ext"           "no_ext:"                       "xor  eax, eax"                 "done_ext:"                     "pop  ebx"                      __value [__eax]                 __modify [__eax __ecx __edx]
#endif

/* CPUID ONCE, NOT PER CALL, and this is a fix rather than a tidy-up.
   lz_cpuid1_edx expands to a raw `cpuid`, which is a SERIALIZING
   instruction - the pipeline drains around it. Two tier selectors read
   bit 25 from it (lz_q8r_tier, p2_tier) and both are called from inner
   loops: lz_q8r_tier once per 32-element quantize group, p2_tier once
   per state row. On a machine that takes those branches the cost is a serializing
   instruction per group.

   MEASURED, and it is not small: the SSE1 pass-2 tier measured 3203 ms
   against the MMX tier's 500 ms - 6.4x - on two
   builds of the SAME binary that differed only in which tier p2_tier
   returned. The kernels were instruction-for-instruction identical at
   that point (the SSE1 one had its pmaxsw swapped back out to check
   exactly this), so the whole 6.4x was 5.12M cpuid instructions.

   IT LOOKED LIKE A SLOW KERNEL. That is the part worth remembering: a
   dispatch cost attributes itself to whatever it dispatches to, and the
   arm that pays it is whichever one the CPUID branch guards. The two
   detectors below run once at startup and are not the problem; the
   three tier selectors are.

   END TO END, not just in the operator microbenchmark, and the control
   is what makes it a mechanism rather than a correlation. Watcom, 48
   greedy tokens on d_ref, best of 3:

     --kernel mmx    0.065 -> 0.042 s/token   1.55x
     --kernel sse2   0.034 -> 0.034 s/token   unchanged

   The second row is the point. lz_q8r_tier returns at the SSE2 branch
   before it ever reaches the cpuid, so an SSE2 machine never paid this
   and the fix cannot help it - and it does not. The path that took the
   branch got 37% of its decode time back. A fix that helps exactly the
   arm that had the defect, and provably nothing else, is a mechanism.

   Only bit 25 (SSE) is cached, because that is the only bit read from a
   hot path. -1 as "not yet asked" rather than a separate flag: the value
   is a bool, so no valid answer collides with it.

   THE GUARD IS `__WATCOMC__` ALONE, MATCHING lz_cpuid1_edx's, and that
   is not a detail. It must NOT add `&& defined(__MMX__)` to silence a
   gcc -Wunused-function warning: every caller's gcc branch answers the
   question at compile time, so the narrower guard would not cover
   p2_tier's call site, which is `#if defined(__WATCOMC__)` with no MMX
   clause. A Watcom build without -D__MMX__ would then compile a call to
   an undeclared function: W131, no prototype, and Watcom's exit code is
   nonzero for warnings. This is the same failure this file's
   dot-kernel header opens by
   describing - one symbol, two guards, and the mismatch only shows in a
   configuration nobody happens to build. Matching the caller (as this
   does) also removes the gcc warning, since gcc never sees the function
   at all. */
#if defined(__WATCOMC__)
static int g_has_sse = -1;

static int lz_cpu_has_sse(void) {
#ifdef LZ_CPUID_NOCACHE
    /* THE CONTROL: reproduces the no-cache behaviour, one cpuid per
       call. It exists because a fix whose effect cannot be reproduced
       later is indistinguishable from a fix that did nothing - the
       symptom is 6.4x on a kernel that is innocent. Build with
       -DLZ_CPUID_NOCACHE=1 and the numbers in this comment come back. */
    return (lz_cpuid1_edx() & (1u << 25)) != 0;
#else
    if (g_has_sse < 0) g_has_sse = (lz_cpuid1_edx() & (1u << 25)) != 0;
    return g_has_sse;
#endif
}
#endif

static int pf_detect(void) {
#if defined(__WATCOMC__) && defined(__MMX__)
    unsigned edx = lz_cpuid1_edx();
    if (edx & (1u << 25)) return LZ_PF_NTA;        /* SSE onwards has prefetchnta */
    if (lz_cpuid_ext_edx() & (1u << 31)) return LZ_PF_AMD;   /* 3DNow! */
    if (edx & (1u << 23)) return LZ_PF_LOAD;       /* PII: dummy load only */
    return LZ_PF_NONE;
#else
    return LZ_PF_NTA;                              /* local gcc: __builtin_prefetch */
#endif
}

int lz_prefetch_mode(void) {
    if (g_pf < 0) g_pf = pf_detect();
    return g_pf;
}

/* Token pairing: share ONE weight unpack across two tokens, or don't.
   ORTHOGONAL to the kernel tier and to prefetch - it picks which
   32-element kernel a row loop calls, not which instruction set.

   This is a knob rather than a hardcoded choice because the two options
   optimize DIFFERENT things and nobody has measured which wins on the
   target: pairing removes the weight unpack for the second token (the
   memory side), while the 128-element group kernel amortizes fixed
   prologue overhead from once per 32 elements to once per 128 (the
   instruction side). Iron law 9 says a disputed choice becomes a
   switch, and iron law 3 says the answer is per machine anyway.

   Pairing needs nt >= 2, so `on` is a request, not a guarantee; at
   nt == 1 both settings run the same code.

   Like prefetch, this must not change a single output bit - the
   integer sums are identical, only the order weights are unpacked in
   differs. */
static int g_pair = 1;

int lz_pair_mode(void) { return g_pair; }

/* GDN pass-1 tier: fixed-point coefficient plane, or the original float
   pass. UNLIKE --kernel, --prefetch and --pair, this one CHANGES THE
   NUMBERS - that is its whole point, it trades a little precision for
   integer arithmetic (iron law 8's "a little precision may be spent").
   So its gate cannot be "bit identical across settings"; it is
   "bit identical WITHIN a setting across builds".

   The float path is still in every binary - kd > LZ_GDN_MAX_KD falls
   back to it, and a LZ_GDN_FIXED=0 control build runs it exclusively -
   so it is a CAPACITY FALLBACK, not a tier the CLI can select. The
   `float` arm of --gdn is gone because the pentium_mmx bench measured
   it dominated (+4.6%): the fixed default is the fast one, and the
   float arm is the baseline the fixed tier is judged against, not a
   deployment choice. Removing the SELECTOR does not remove the float
   body, which stays reachable as a capacity fallback, not dead code. */
static int g_gdn_fixed = 1;

int lz_gdn_mode(void) {
#if LZ_GDN_FIXED
    return g_gdn_fixed;
#else
    return 0;                    /* not compiled in; never claim otherwise */
#endif
}

/* Pass-2 tier, separate from the pass-1 one on purpose: they are two
   independent decisions with different evidence behind them. Pass 1's
   fixed tier adds ~1.5e-05 and ships by default; pass 2's adds ~3.8e-05
   (gdn_p2_group_fixed's comment has the measurement) and is OFF until an
   end-to-end paired NLL says that is invisible. Folding both into one
   flag would make the weaker-evidenced one ride in on the stronger. */
/* AUTO is the default, and it resolves by asking whether this machine has
   a SIMD tier for the write-back quantize - which is the entire reason
   the fixed pass 2 exists.

   The measurement behind that: the quantize costs 1187 ms scalar against
   172 ms at the SSE2 tier (Watcom, same isolated arm both
   times, so the 6.8x is a ratio the inlining bias cancels out of). A
   machine WITH that tier would be trading a fast SIMD float path for a
   scalar integer one - a regression. A machine without it (Pentium II:
   lz_q8round32 structurally cannot have an MMX tier, MMX has no
   float-to-int conversion) has nothing else to trade.

   So AUTO is not caution, it is iron law 3's per-machine rule applied to
   a knob whose sign flips inside the target family.

   Quality licence for turning it on where it helps: end-to-end paired
   NLL on d_ref, fixed vs float, t = -0.29 over 368 positions with the
   point estimate slightly FAVOURING fixed - indistinguishable, and the
   same bar the pass-1 fixed tier is cleared against (t = -0.24).

   AUTO asks q8r_have_simd(), i.e. the HARDWARE question, and
   deliberately not lz_q8r_tier(): the latter also answers 0 for
   `--kernel ref`, which would make a debugging override change the
   numbers. --kernel is documented numerics-neutral and
   test_reachable_tiers_are_bit_identical enforces that.

   Consequence, stated because it is a real limitation rather than a
   detail: on every machine that can run this suite, AUTO resolves to
   float, so its other branch is not reachable here at all. What IS
   testable locally is that AUTO resolves to float, that an explicit
   --gdn-p2 overrides it, and (via the fixed setting) that the fixed
   path itself is correct and cross-build identical. The AUTO decision
   on a PII rests on q8r_have_simd's CPUID check, which is the same
   check lz_q8r_tier uses to pick the scalar rounding tier there - so
   it is not new untested logic, just a second reader of it. */
#define LZ_P2_FLOAT 0
#define LZ_P2_FIXED 1
#define LZ_P2_AUTO  2
static int g_gdn_p2 = LZ_P2_AUTO;

int lz_gdn_p2_mode(void) {
#if LZ_GDN_FIXED
    if (g_gdn_p2 == LZ_P2_AUTO) return q8r_have_simd() == 0;
    return g_gdn_p2;
#else
    return 0;                    /* not compiled in; never claim otherwise */
#endif
}

int lz_gdn_p2_is_auto(void) { return g_gdn_p2 == LZ_P2_AUTO; }

const char *lz_gdn_p2_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "fixed") == 0) g_gdn_p2 = LZ_P2_FIXED;
    else if (strcmp(name, "auto") == 0) g_gdn_p2 = LZ_P2_AUTO;
    else return NULL;
    return lz_gdn_p2_mode() ? "fixed" : "float";
}

const char *lz_pair_select(const char *name) {
    if (!name) return NULL;
    if (strcmp(name, "on") == 0 || strcmp(name, "auto") == 0) g_pair = 1;
    else if (strcmp(name, "off") == 0) g_pair = 0;
    else return NULL;
    return g_pair ? "on" : "off";
}

const char *lz_prefetch_select(const char *name) {
    int have = pf_detect();
    int mode;
    if (!name)                        return NULL;
    else if (strcmp(name, "auto")  == 0) mode = -1;
    else if (strcmp(name, "none")  == 0) mode = LZ_PF_NONE;
    else if (strcmp(name, "load")  == 0) mode = LZ_PF_LOAD;
    else if (strcmp(name, "nta")   == 0) mode = LZ_PF_NTA;
    else if (strcmp(name, "3dnow") == 0) mode = LZ_PF_AMD;
    else                              return NULL;

    if (mode == -1) {
        g_pf = have;
    } else if (mode == LZ_PF_NONE || mode == LZ_PF_LOAD) {
        /* Both are available on every x86, so both are always honoured.
           NONE issues no instruction at all - it is what makes the
           latency tier measurable. LOAD is a plain `mov eax,[eax]`,
           valid on anything, which lets a PIII deliberately run the
           PII's prefetch tier; refusing it would block a comparison
           that costs nothing and answers "how much of J is the
           instruction and how much is the machine". */
        g_pf = mode;
    } else if (mode == have) {
        g_pf = mode;
    } else {
        /* The other two DO need CPU support: prefetchnta is SSE, 3DNow!
           prefetch is an AMD extension, and either one is an invalid
           opcode elsewhere. A benchmark that faults is not a slower
           benchmark. Downgrade rather than crash - the caller gets the
           effective mode back and is expected to print THAT, not what
           it asked for. */
        g_pf = have;
    }
    return lz_prefetch_name();
}

const char *lz_prefetch_name(void) {
    switch (lz_prefetch_mode()) {
    case LZ_PF_NTA:  return "nta";
    case LZ_PF_AMD:  return "3dnow";
    case LZ_PF_LOAD: return "load";
    default:         return "none";
    }
}

static int kernel_detect(void) {
#if defined(__WATCOMC__) && defined(__MMX__)
    unsigned edx = lz_cpuid1_edx();
    if (edx & (1u << 26)) return LZ_KERNEL_SSE2;      /* Pentium M / P4 */
    if (edx & (1u << 23)) return LZ_KERNEL_MMX;
    return LZ_KERNEL_REF;
#elif defined(__SSE2__)
    /* SSE2 BEFORE MMX, and the order is load-bearing. This build carries
       BOTH intrinsics tiers, so falling through to the MMX branch would
       silently demote every x86-64 run to 64-bit kernels. No CPUID here:
       __SSE2__ means the compiler was told the target has it. The MMX
       tier stays selectable through --kernel, which is the whole point
       of compiling both in. */
    return LZ_KERNEL_SSE2;
#elif defined(__MMX__)
    return LZ_KERNEL_MMX;
#else
    return LZ_KERNEL_REF;
#endif
}

int lz_kernel_select(int which) {
    int have = kernel_detect();
    if (which == LZ_KERNEL_AUTO) {
        g_kernel = have;
    } else {
#if defined(__WATCOMC__) && defined(__MMX__)
        /* Both asm tiers are compiled in, so any of them can be asked
           for - but not a tier the local CPU lacks (SSE2 needs CPUID
           bit 26; asking for it on a PII would be an invalid opcode,
           and a benchmark that faults is not a slower benchmark). */
        g_kernel = (which == LZ_KERNEL_SSE2 && have != LZ_KERNEL_SSE2)
                   ? have : which;
#elif defined(__MMX__) && defined(__SSE2__)
        /* ONE gcc build carries both intrinsics tiers, so both are
           honestly selectable - `--kernel mmx` on an x86-64 build really
           selects the MMX tier, and the tier report matches it. */
        g_kernel = which;
#else
        g_kernel = (which == LZ_KERNEL_REF) ? LZ_KERNEL_REF : have;
#endif
    }
    return g_kernel;
}

/* lz_quantize_q8's rounding tier. Here rather than above because
   g_kernel and the CPUID probes live in this section; only a forward
   decl sits above.

   0 = scalar magic rounding  1 = SSE1 (cvtps2pi, writes MMX)  2 = SSE2 (cvtps2dq)

   **Orthogonal to the kernel tier**, same reason as prefetch (see
   ops.h): the kernel tier only controls integer SIMD width, while
   PIII/K7's integer SIMD is 64-bit MMX yet float has 128-bit SSE.
   Tying rounding to the kernel tier would miss the whole PIII family -
   the exact lesson from deleting LZ_KERNEL_MMX_SSE.

   The REF tier must go scalar: it is the oracle and must not itself
   use SIMD.
   `LZ_Q8R_FORCE_SSE` forces the SSE1 tier on SSE2 dev machines for
   cross-compiler differential checks - otherwise that kernel is never
   selected locally and never validated. */
/* Does this BUILD on this MACHINE have any SIMD tier for the q8 rounding,
   ignoring the --kernel override? Split out from lz_q8r_tier because
   --gdn-p2 auto has to ask the hardware question and must NOT ask the
   override one.

   It must NOT call lz_q8r_tier: that tier answers the --kernel
   override, so `--kernel ref` would flip pass 2 from float to fixed -
   and --kernel is documented as numerics-neutral ("every tier
   bit-identical"), with test_reachable_tiers_are_bit_identical
   enforcing it. That test goes red immediately, and it is right to:
   making a debugging override change the numbers, in order to make
   another knob's branch reachable from the test suite, trades a real
   invariant for test convenience. */
static int q8r_have_simd(void) {
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
#if defined(__WATCOMC__) && defined(__MMX__)
    return lz_cpu_has_sse();       /* CPUID leaf 1 EDX 25 = SSE, cached */
#else
    return 1;                       /* gcc build: SSE guaranteed at compile time */
#endif
#else
    return 0;
#endif
}

static int lz_q8r_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return 0;
#if defined(LZ_HAVE_Q8R_SSE)
#ifdef LZ_Q8R_FORCE_SSE
    return 1;
#endif
#endif
#if defined(LZ_HAVE_Q8R_SIMD)
    if (g_kernel == LZ_KERNEL_SSE2) return 2;
#endif
#if defined(LZ_HAVE_Q8R_SSE)
#if defined(__WATCOMC__) && defined(__MMX__)
    /* CPUID leaf 1 EDX bit 25 = SSE, CACHED - this sits in an inner loop
       and cpuid serializes the pipeline. PII/K6-2 lack it; stay scalar. */
    if (lz_cpu_has_sse()) return 1;
#else
    return 1;                       /* local gcc build: SSE guaranteed at compile time */
#endif
#endif
    return 0;
}

const char *lz_kernel_name(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_SSE2) return "sse2";
    if (g_kernel == LZ_KERNEL_MMX) return "mmx";
    return "ref";
}

/* Compile-time path report - see lz_build_paths' comment in ops.h.
   The condition is copied from the four matmul_*_impl bodies verbatim
   and must stay copied: the moment it is "simplified" it stops being
   a report of what those functions did and becomes an opinion about
   what they should have done. */
const char *lz_build_paths(void) {
#if defined(__WATCOMC__) && defined(__MMX__)
    return "mmx-asm+sse2-asm";
#elif defined(__MMX__) && defined(__SSE2__)
    return "mmx-intrin+sse2-intrin";
#elif defined(__MMX__)
    return "mmx-intrin";
#elif defined(__SSE2__)
    return "sse2-intrin";
#else
    return "scalar";
#endif
}

/* WHICH ONE OF THE SEVEN OPERATOR VARIANTS ACTUALLY RAN.
   Iron law 8: scalar C, plus MMX/SSE/SSE2 as intrinsics, plus
   MMX/SSE/SSE2 as hand-written asm.

   Neither half alone identifies it. lz_kernel_name() answers only the
   ISA - it says "mmx" for the hand-written asm and for the intrinsics
   alike, because they share one enum value. lz_build_paths() answers
   only the compile-time impl, for the whole binary. It is the JOIN that
   names a variant, and the join is exact because no build carries both
   impls of one ISA: gcc cannot compile #pragma aux at all (iron law 2
   clause 1) and Watcom's _m_* intrinsics run 6x slower than its own x87
   scalar, so they are not compiled either (clause 4).

   This exists so a gate can assert the UNION of the tiers over every
   build. That union is the only assertion that catches a whole path
   silently vanishing: the documented Watcom build compiles none of the
   assembly and runs pure scalar, and the cross-compiler bit-identity
   check stays green throughout - every path is REQUIRED to agree bit
   for bit, so two builds running different paths look identical.
   Consistency gates cannot see this class of bug at all.

   The seven names are the CLI vocabulary too, so keep them literal and
   stable: ref, mmx-intrin, sse-intrin, sse2-intrin, mmx-asm, sse-asm,
   sse2-asm. Only five are reachable today; the two SSE tiers are task
   15 and land in stage 5 of the dispatch plan. */
const char *lz_kernel_tier(void) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    if (g_kernel == LZ_KERNEL_REF) return "ref";   /* scalar C: no impl axis */
#if defined(__MMX__)
#  ifdef __WATCOMC__
    return (g_kernel == LZ_KERNEL_SSE2) ? "sse2-asm" : "mmx-asm";
#  else
    return (g_kernel == LZ_KERNEL_SSE2) ? "sse2-intrin" : "mmx-intrin";
#  endif
#elif defined(__SSE2__)
    /* Only SSE2 and REF are selectable here (lz_kernel_select clamps a
       non-REF request to what kernel_detect found, and that is SSE2),
       and REF returned above. If that clamp ever changes, this line
       starts lying - which is what the gate is for. */
    return "sse2-intrin";
#else
    return "ref";                                  /* unreachable: detect gives REF */
#endif
}

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
#define LZ_MM_WIDEN_MAX 4096

/* Scratch shared by the three SIMD matmuls (batched version widens by
   LZ_BATCH_MAX). The three are never alive simultaneously
   (single-threaded, no nested calls), so one block serves all of them.
   matmul_scalar_ref's accref/xgref are deliberately NOT merged: on
   non-SIMD builds q41/q61 call scalar_ref after filling xw, and sharing
   would alias.

   token-major: g_xw[t*in_dim + k], g_acc32[t*nsb + s], g_xg[t*ng + g]. */
static int16_t g_xw[LZ_BATCH_MAX * LZ_MM_WIDEN_MAX];
static float   g_xg[LZ_BATCH_MAX * (LZ_MM_WIDEN_MAX / 32)];
/* Unguarded: the SSE2-intrinsics path writes here too now that it fills
   acc32 instead of fusing the float work into the integer loop. Costs
   4KB of BSS in every build. */
static int32_t g_acc32[LZ_BATCH_MAX * (LZ_MM_WIDEN_MAX / 32)];

/* ---- prefetch (lever J) ------------------------------------------------
   P6 has only 4 fill buffers and a 40-entry ROB. The matmul inner loop
   executes ~50 instructions per 32 weight bytes; to keep 4 misses in
   flight you must look ~128 bytes ahead = ~200 instructions - far
   beyond the ROB. So a PII most likely serializes per 32 bytes at one
   full memory latency: 32 B / ~180 ns ~ 178 MB/s, not the 300-400 MB/s
   the bus could deliver.

   Three tiers:
     PII        no PREFETCH instruction (that is SSE); only a dummy
                load - occupies one register and one ROB slot, but
                feeds the fill buffers.
     PIII/PM    `prefetchnta`. The nta variant fits this workload:
                weights are used once per token and must never enter L2
                (Coppermine's L2 is only 256KB and would be flushed by
                weights, evicting SSM state and activations along the
                way).
     ref        none.

   **Prefetch changes no numbers** - the gate is "all three tiers'
   logits are byte-identical to ref". If they differ, the prefetch is a
   real access or out of bounds.

   The distance is machine-dependent, kept as a compile-time constant
   to sweep 0/2/4/8 on target hardware. */
#ifndef LZ_PF_DIST
#define LZ_PF_DIST 4                    /* look ahead this many 32-byte cache lines */
#endif

/* Compile-time A/B control for the SSE2 group kernels (128 elements per
   call) that sit alongside the 32-element MMX ones.

   `-DLZ_SSE2_GROUP=0` makes an SSE2 CPU take the MMX group path instead,
   i.e. the MMX group kernels alone. Two binaries from one source tree,
   so the "did this change any number"
   question is answered by `cmp` on --dump-logits rather than by swapping
   source trees and hoping the rest matched. Same reason LZ_BATCH_MAX and
   LZ_Q8R_FORCE_SSE exist.

   Only the Watcom build has these kernels at all - the gcc translation
   unit does not contain them (they live inside `#if defined(__WATCOMC__)`),
   so the gcc side is unaffected by this switch by construction. */
#ifndef LZ_SSE2_GROUP
#define LZ_SSE2_GROUP 1
#endif

#if defined(__WATCOMC__)
/* Watcom side has two: dummy load (PII) and prefetchnta (PIII+).
   __modify declared truthfully - rule 2.3. */
extern void lz_pf_load(const void *p);
#pragma aux lz_pf_load =            \
    "mov eax, [eax]"                \
    __parm [__eax] __modify [__eax]

extern void lz_pf_nta(const void *p);
#pragma aux lz_pf_nta =             \
    ".686"                          \
    "prefetchnta [eax]"             \
    __parm [__eax] __modify []
extern void lz_pf_amd(const void *p);
#pragma aux lz_pf_amd =                 ".686"                              "db 0x0F, 0x0D, 0x00"               __parm [__eax] __modify []
#define LZ_PFI_LOAD(p) lz_pf_load(p)
#define LZ_PFI_NTA(p)  lz_pf_nta(p)
#define LZ_PFI_AMD(p)  lz_pf_amd(p)
#elif defined(__GNUC__)
#define LZ_PFI_LOAD(p) __builtin_prefetch((p), 0, 0)
#define LZ_PFI_NTA(p)  __builtin_prefetch((p), 0, 0)
#define LZ_PFI_AMD(p)  __builtin_prefetch((p), 0, 0)
#else
#define LZ_PFI_LOAD(p) ((void)0)
#define LZ_PFI_NTA(p)  ((void)0)
#define LZ_PFI_AMD(p)  ((void)0)
#endif

/* No prefetch at all. Not a platform fallback - a deliberately
   reachable mode, because the LATENCY TIER is a thing we need to
   measure and could not.

   The performance model has two tiers per machine: latency (no
   prefetch, assumed 178 MB/s on PIII) and bandwidth (prefetchnta,
   assumed 440). Every performance conclusion that matters - J's payoff,
   whether MTP has headroom to trade into - is a ratio between those two
   numbers. The dispatch otherwise has no way to reach "no prefetch":
   pf_detect() returns LOAD/NTA/AMD and the selection's else-branch
   falls to NTA, so a run on real hardware would only ever measure the
   bandwidth tier. This mode is the instrument for the latency half of
   every one of those questions. */
#define LZ_PFI_NONE(p) ((void)(p))

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
typedef void (*lz_rowfn)(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end,
                         const void *pf_end2);

/* Defined below; the tiered matmuls fall back to it when they have no
   row kernel for the selected tier, so it must be visible up here. */
static void matmul_scalar_ref(float *o, const int8_t *xq, const float *xqs,
                              const LZTensor *w, int in_dim, int out_dim,
                              int nt);

/* The seven variants of iron law 8, as table slots. A build fills at
   most three of them (gcc cannot compile #pragma aux; Watcom's _m_*
   intrinsics are 6x slower than its own x87 scalar), and REF is not
   here at all - lz_matmul_q8_nt intercepts that tier before reaching
   any impl, so that the scalar reference stays a real oracle.

   An unfilled slot is NULL and the pick falls back, loudly rather than
   silently: a tier that cannot be reached must not report itself as
   the tier that ran. That confusion is the whole reason this file is
   being restructured. */
#define LZ_ROW_MMX_I   0
#define LZ_ROW_SSE_I   1
#define LZ_ROW_SSE2_I  2
#define LZ_ROW_MMX_A   3
#define LZ_ROW_SSE_A   4
#define LZ_ROW_SSE2_A  5
#define LZ_ROW_N       6

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
static lz_rowfn lz_row_pick(const lz_rowfn *tab) {
    int want_sse2 = (g_kernel == LZ_KERNEL_SSE2);
#ifdef __WATCOMC__
    lz_rowfn f = tab[want_sse2 ? LZ_ROW_SSE2_A : LZ_ROW_MMX_A];
    return f ? f : tab[LZ_ROW_MMX_A];
#else
    lz_rowfn f = tab[want_sse2 ? LZ_ROW_SSE2_I : LZ_ROW_MMX_I];
    if (f) return f;
    return tab[LZ_ROW_SSE2_I] ? tab[LZ_ROW_SSE2_I] : tab[LZ_ROW_MMX_I];
#endif
}

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
   generated text. Iron law 2, and iron law 4's note that |logit diff|
   saturates and cannot rank such changes.

   `post` cancels the kernel's own scaling: 1/256 for Q8_0 (part32_x16
   and the MMX/asm kernels all scale their products by 256) and 1.0 for
   Q16_0, whose kernels do not. A power of two either way, so the cancel
   is exact, and `x * 1.0f` is bit-identical to `x` - which is what lets
   Q16_0 share this instead of carrying a second copy differing in one
   multiply. Both call sites pass a literal, so it folds away. */
static float epi_q8(const int32_t *accb, const float *xsb, const float *ws,
                    int ng, int r, float post) {
    int g;
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

#if defined(__MMX__)

/* Prefetch distance clamped by a pointer upper bound once: PIII's
   prefetchnta is harmless out of bounds (pure hint, no fault), but
   PII's dummy load is a REAL access - reading an unmapped page
   segfaults. One compare buys safety for the whole path; the branch is
   highly predictable. */
#define LZ_Q8_ACC(DOT32, PF)                                        \
    for (g = 0; g < nb; g++) {                                      \
        const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;     \
        if (pf_ < wend) PF(pf_);                                    \
        acc[g] = DOT32(wr + (size_t)g * 32, xwt + (size_t)g * 32);  \
    }

/* Kernel choice sits at the ROW level; the inner 32-element kernel
   stays a direct call. Prefetch tier is orthogonal and resolved here
   too, which is why this expands the loop body four times rather than
   testing per sub-block. */
#define LZ_Q8_PFSEL(DOT32)                                              \
    do {                                                                \
        int pf_m_ = lz_prefetch_mode();                                 \
        if (pf_m_ == LZ_PF_AMD)       { LZ_Q8_ACC(DOT32, LZ_PFI_AMD); }  \
        else if (pf_m_ == LZ_PF_LOAD) { LZ_Q8_ACC(DOT32, LZ_PFI_LOAD); } \
        else if (pf_m_ == LZ_PF_NONE) { LZ_Q8_ACC(DOT32, LZ_PFI_NONE); } \
        else                          { LZ_Q8_ACC(DOT32, LZ_PFI_NTA); }  \
    } while (0)

#ifdef __WATCOMC__
/* One group (128 elements) at a time when nb is a multiple of 4: the
   fixed overhead - mask construction, prologue/epilogue - is amortized
   from once per 32 elements to once per 128. Integer results are
   bit-identical to the per-32 version. All current tensors have in_dim
   a multiple of 128 (512/768/1024/3072). */
#define LZ_Q8_GROUP4(DOT128)                                            \
    do {                                                                \
        for (g = 0; g < nb; g += 4) {                                   \
            const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;     \
            if (pf_ < wend) {                                           \
                int pf_m_ = lz_prefetch_mode();                         \
                if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);          \
                else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);         \
                else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);         \
                else                          LZ_PFI_NTA(pf_);          \
            }                                                           \
            DOT128(wr + (size_t)g * 32, xwt + (size_t)g * 32, acc + g); \
        }                                                               \
    } while (0)

static void row_q8_mmx_asm(int32_t *acc32, const void *w4, const void *w2,
                           const int16_t *xw, int nb, int nt, int in_dim,
                           const void *pf_end,
                         const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    const int8_t *wend = (const int8_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Pairing beats the 128-element group kernel whenever there is
           a second token: the group kernel amortizes FIXED overhead
           (mask setup, prologue) from once per 32 elements to once per
           128, while pairing removes the weight unpack itself for the
           second token - and on the target the weight side is the
           memory side. Same choice the gcc twin makes, which also keeps
           the two impls' shapes comparable when they are diffed.
           At nt == 1 there is nothing to pair, so the group kernel
           keeps that path unchanged.

           NOT MEASURED ON TARGET. The op-count argument is 12 + 24*NT
           against 36*NT; whether it beats the group kernel's amortized
           prologue on a real PII is a wall-clock question, and the
           machine for it is not here. Iron law 3: this is a knob-shaped
           decision recorded as such, not a measured result. */
        if (nt - tk >= 2 && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            int g2;
            for (g2 = 0; g2 < nb; g2++) {
                const int8_t *pf_ = wr + (size_t)(g2 + LZ_PF_DIST) * 32;
                if (pf_ < wend) {
                    int pf_m_ = lz_prefetch_mode();
                    if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);
                    else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);
                    else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);
                    else                          LZ_PFI_NTA(pf_);
                }
                lz_dot32_x16_asm_2(wr + (size_t)g2 * 32,
                                   xwt + (size_t)g2 * 32,
                                   xw2 + (size_t)g2 * 32,
                                   acc + g2, accb + g2);
            }
            tk++;
            continue;
        }
        if ((nb & 3) == 0) LZ_Q8_GROUP4(lz_dot128_x16_asm);
        else               LZ_Q8_PFSEL(dot32_x16_mmx);
    }
}

static void row_q8_sse2_asm(int32_t *acc32, const void *w4, const void *w2,
                            const int16_t *xw, int nb, int nt, int in_dim,
                            const void *pf_end,
                         const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    const int8_t *wend = (const int8_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
#if LZ_SSE2_GROUP
        if ((nb & 3) == 0) LZ_Q8_GROUP4(lz_dot128_x16_sse2_asm);
#else
        /* LZ_SSE2_GROUP=0 keeps the 128-element step on the MMX kernel;
           the knob exists so an SSE2 CPU can be measured on either. */
        if ((nb & 3) == 0) LZ_Q8_GROUP4(lz_dot128_x16_asm);
#endif
        else               LZ_Q8_PFSEL(dot32_x16_sse2a);
    }
}

#else  /* gcc: intrinsics */

static void row_q8_mmx_intrin(int32_t *acc32, const void *w4, const void *w2,
                              const int16_t *xw, int nb, int nt, int in_dim,
                              const void *pf_end,
                         const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    const int8_t *wend = (const int8_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Pair this token with the next so ONE weight unpack serves
           both. The prefetch is issued once for the pair, not once per
           token: it is a hint about the WEIGHT row, and the second
           token reads the bytes the first just pulled in. An odd nt
           leaves the last token on the single-token kernel. */
        if (tk + 1 < nt && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            for (g = 0; g < nb; g++) {
                const int8_t *pf_ = wr + (size_t)(g + LZ_PF_DIST) * 32;
                if (pf_ < wend) LZ_PFI_NTA(pf_);
                dot32_x16_mmx_2(wr + (size_t)g * 32,
                                xwt + (size_t)g * 32,
                                xw2 + (size_t)g * 32,
                                acc + g, accb + g);
            }
            tk++;
        } else {
            LZ_Q8_ACC(dot32_x16_mmx, LZ_PFI_NTA);
        }
    }
}

#endif /* __WATCOMC__ */
#endif /* __MMX__ */

#if defined(__SSE2__) && !defined(__WATCOMC__)
static void row_q8_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                               const int16_t *xw, int nb, int nt, int in_dim,
                               const void *pf_end,
                         const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    int tk, g;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (g = 0; g + 3 < nb; g += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + g),
                fold4_sse2(
                part32_x16(wr + (size_t)(g + 0) * 32, xwt + (size_t)(g + 0) * 32),
                part32_x16(wr + (size_t)(g + 1) * 32, xwt + (size_t)(g + 1) * 32),
                part32_x16(wr + (size_t)(g + 2) * 32, xwt + (size_t)(g + 2) * 32),
                part32_x16(wr + (size_t)(g + 3) * 32, xwt + (size_t)(g + 3) * 32)));
        }
        for (; g < nb; g++) {           /* tail when nb % 4 != 0 */
            __m128i p = part32_x16(wr + (size_t)g * 32, xwt + (size_t)g * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[g] = _mm_cvtsi128_si32(p);
        }
    }
}
#endif

/* The table. A NULL slot is a tier this build does not carry; the two
   SSE slots stay NULL because no SSE row kernel exists for this format. */
static const lz_rowfn LZ_ROW_Q8[LZ_ROW_N] = {
#if defined(__MMX__) && !defined(__WATCOMC__)
    row_q8_mmx_intrin,
#else
    NULL,
#endif
    NULL,        /* sse-intrin: no SSE1 op applies, see lz_row_pick */
#if defined(__SSE2__) && !defined(__WATCOMC__)
    row_q8_sse2_intrin,
#else
    NULL,
#endif
#if defined(__MMX__) && defined(__WATCOMC__)
    row_q8_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_q8_sse2_asm
#else
    NULL, NULL, NULL
#endif
};

/* int8 matmul kernel for gs==32, dispatched by target platform
   (LZ_USE_MMX forces MMX for local verification; SSE2 is the x86-64
   default; ref is the fallback). */
static void matmul_q8_impl(float *o, const int8_t *xq, const float *xqs,
                           const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk, t;
    int r  = w->gs / 32;    /* 32-sub-blocks per weight group (gs=128 -> 4) */
    int nb = in_dim / 32;   /* 32-sub-block count */
    const int8_t *wend = w->q + (size_t)w->n;   /* prefetch upper bound */
    lz_rowfn row;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_Q8);

    /* Two ways to have no row kernel, and both must land on the scalar
       reference rather than on a second inline implementation:
       - this build carries none for the selected ISA (all-NULL table);
       - in_dim exceeds the pre-expansion buffer.
       Both land on the scalar reference rather than on a second inline
       implementation. Inline fallbacks (an MMX dot32_nowiden loop, an
       SSE2 part32_sse2 loop, the ref branch) would be three DIFFERENT
       code paths, none of them bit-identical to the others, all of them
       unreachable - the largest in_dim in any current model is 3584
       against a 4096 cap. Dead code in three association orders is
       worse than dead code in one, and matmul_scalar_ref is the
       contract (iron law 2). */
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }

    /* Expand the whole batch's activations to int16 once (in_dim <=
       4096, resident in L1); amortized over out_dim rows. */
    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];

    for (i = 0; i < out_dim; i++) {
        const int8_t *wr = w->q + (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)(i * in_dim) / w->gs;
        /* The weight row loads once here and serves all nt tokens; one
           EMMS per row, not per token - MMX and x87 share a register
           file and the epilogue below is x87 on the Watcom build. */
        row(g_acc32, wr, NULL, g_xw, nb, nt, in_dim, wend, NULL);
        _mm_empty();
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] =
                epi_q8(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                       ws, nb, r, 1.0f / 256.0f);
    }
}

/* Get the f32 view. In ops.c rather than model.c: this is a FORMAT
   matter, sharing the same layout knowledge as the matmul dispatch
   below - adding a format only touches this one file. */
/* int32 -> float, a form that is bit-identical across both compilers.

   `(float)v` is exact for |v| < 2^24 on both sides; BEYOND 2^24 there
   is rounding, and the rounding timing differs between compilers:
   x87's `fild` loads exactly into an 80-bit register and can `fmul`
   directly (rounding the product once), while SSE's `cvtsi2ss` must
   first round the integer to 24 bits, then multiply (rounding twice).
   PC=24 cannot govern this - precision control applies to arithmetic
   results, not to loads.

   Q8_0's sub-block accumulator bound is 127·127·32 = 5.2e5 and Q4_1's
   6.1e4, both within 2^24, so this road ran a long time untouched.
   **Q16_0 is the first format that can exceed it** (32767·127·32 =
   1.33e8); measured: some sub-block sum of 19834503 in out_proj line
   160 flipped 1 ULP, amplified by the reduction to 4 ULP, max
   end-to-end logits difference 1.4e-5.

   Split into two parts: |v>>12| < 2^19 and |v & 0xFFF| < 2^12, both
   exact as float; multiplying by 4096 (a power of two) is exact; only
   the final addition rounds - and addition is arithmetic, which PC=24
   governs. In two's complement (v>>12)*4096 + (v & 0xFFF) == v always
   holds (`>>` is arithmetic shift). */
static float lz_i32f(int32_t v) {
    return (float)(v >> 12) * 4096.0f + (float)(v & 0xFFF);
}

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
static void matmul_scalar_ref_one(float *o, const int8_t *xq, const float *xqs,
                                  const LZTensor *w, int in_dim, int out_dim) {
    int i, g, s, k;
    int gs = w->gs, r, ng, nsb;
    int q4 = (w->dtype == LZ_FMT_Q4_1 || w->dtype == LZ_FMT_Q6_1);
    int q6 = (w->dtype == LZ_FMT_Q6_1);
    int q16 = (w->dtype == LZ_FMT_Q16_0);
    /* static not stack: Win98's stack is tight; see the xw/acc32 note in this file. */
    static int32_t accref[LZ_MM_WIDEN_MAX / 32];
    static float   xgref[LZ_MM_WIDEN_MAX / 32];

    if (gs < 32 || (gs % 32) != 0 || (in_dim % gs) != 0) {
        /* degenerate tier (gs < 32, activations share weight groups): Q8_0 only */
        for (i = 0; i < out_dim; i++) {
            const int8_t *wr = w->q + (size_t)i * in_dim;
            const float *ws = w->scale + (size_t)(i * in_dim) / gs;
            float sum = 0.0f;
            for (g = 0; g < in_dim / gs; g++) {
                int32_t acc = 0;
                const int8_t *wq = wr + (size_t)g * gs;
                const int8_t *xr = xq + (size_t)g * gs;
                for (k = 0; k < gs; k++)
                    acc += (int32_t)wq[k] * (int32_t)xr[k];
                sum += lz_i32f(acc) * xqs[g] * ws[g];
            }
            o[i] = sum;
        }
        return;
    }
    r  = gs / 32;                   /* 32 sub-blocks per weight group */
    ng = in_dim / gs;
    nsb = in_dim / 32;              /* total number of 32-element sub-blocks */
    if (nsb > LZ_MM_WIDEN_MAX / 32) return;      /* defense: no silent miscalc */

    /* Q4_1's zero term xg[g] = Σ_{sub-block s∈g} xqs[s]·Σ_k xq[k] depends
       only on activations and is shared by all output rows - it MUST be
       summed in the same order as matmul_q41_impl, or the
       bit-identical contract breaks on this branch. */
    if (q4) {
        for (g = 0; g < ng; g++) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int sb = g * r + s;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xq[sb * 32 + k];
                a += lz_i32f(sx) * xqs[sb];
            }
            xgref[g] = a;
        }
    }

    for (i = 0; i < out_dim; i++) {
        const unsigned char *wn = (const unsigned char *)w->q +
                                  (size_t)i * in_dim / 2;
        const unsigned char *w2 = (const unsigned char *)w->q +
                                  (size_t)w->n / 2 + (size_t)i * in_dim / 4;
        const int8_t *wr = w->q + (size_t)i * in_dim;
        const int16_t *w16 = (const int16_t *)(const void *)w->q +
                             (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)i * ng;
        const float *wz = q4 ? w->zero + (size_t)i * ng : NULL;

        /* Step 1: the row's int32 sub-block sums. On the SIMD side
           this segment is the MMX/SSE2 kernel. */
        for (s = 0; s < nsb; s++) {
            int base = s * 32;
            int32_t acc = 0;
            if (q6) {
                /* dot(q,x) = dot(lo,x) + 16·dot(hi,x), exact in
                   integers. The SIMD side splits exactly this way: the
                   lo plane reuses the Q4_1 kernel, the hi plane goes
                   through the 2-bit sub-kernel, ending as
                   acc_lo + (acc_hi<<4). */
                const unsigned char *b4 = wn + (size_t)base / 2;
                const unsigned char *b2 = w2 + (size_t)base / 4;
                int32_t alo = 0, ahi = 0;
                for (k = 0; k < 16; k++) {
                    alo += (int32_t)(b4[k] & 15) * (int32_t)xq[base + k];
                    alo += (int32_t)(b4[k] >> 4) * (int32_t)xq[base + k + 16];
                }
                for (k = 0; k < 32; k++)
                    ahi += (int32_t)((b2[k & 7] >> (2 * (k >> 3))) & 3) *
                           (int32_t)xq[base + k];
                acc = alo + (ahi << 4);
            } else if (q4) {
                const unsigned char *p = wn + (size_t)base / 2;
                for (k = 0; k < 16; k++) {
                    acc += (int32_t)(p[k] & 15) * (int32_t)xq[base + k];
                    acc += (int32_t)(p[k] >> 4) * (int32_t)xq[base + k + 16];
                }
            } else if (q16) {
                for (k = 0; k < 32; k++)
                    acc += (int32_t)w16[base + k] * (int32_t)xq[base + k];
            } else {
                for (k = 0; k < 32; k++)
                    acc += (int32_t)wr[base + k] * (int32_t)xq[base + k];
            }
            accref[s] = acc;
        }

        /* Step 2: float reduction, byte-for-byte matching the SIMD side's epilogue. */
        if (q4) {
            float dotsum = 0.0f, zsum = 0.0f;
            for (g = 0; g < ng; g++) {
                float dot = 0.0f;
                for (s = 0; s < r; s++)
                    dot += lz_i32f(accref[g * r + s]) * xqs[g * r + s];
                dotsum += dot * ws[g];
                zsum += xgref[g] * wz[g];
            }
            o[i] = dotsum + zsum;
        } else if (r > 1) {
            float sum = 0.0f;
            for (g = 0; g < ng; g++) {
                float dot = 0.0f;
                for (s = 0; s < r; s++)
                    dot += lz_i32f(accref[g * r + s]) * xqs[g * r + s];
                sum += dot * ws[g];
            }
            o[i] = sum;
        } else {
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (g = 0; g + 3 < ng; g += 4) {
                a0 += lz_i32f(accref[g])     * (xqs[g]     * ws[g]);
                a1 += lz_i32f(accref[g + 1]) * (xqs[g + 1] * ws[g + 1]);
                a2 += lz_i32f(accref[g + 2]) * (xqs[g + 2] * ws[g + 2]);
                a3 += lz_i32f(accref[g + 3]) * (xqs[g + 3] * ws[g + 3]);
            }
            for (; g < ng; g++)                  /* tail all into a0, same as SIMD */
                a0 += lz_i32f(accref[g]) * (xqs[g] * ws[g]);
            o[i] = (a0 + a2) + (a1 + a3);
        }
    }
}

/* Scalar-reference batched wrapper. Here we DELIBERATELY do no weight
   reuse - the reference path's job is to be the oracle, not fast.
   Reruns per token as-is; batched and unbatched results must agree
   down to the rounding. */
static void matmul_scalar_ref(float *o, const int8_t *xq, const float *xqs,
                              const LZTensor *w, int in_dim, int out_dim, int nt) {
    int gsa = lz_act_gs(w, in_dim);
    int ss = (gsa > 0) ? in_dim / gsa : in_dim / 32;
    int t;
    for (t = 0; t < nt; t++)
        matmul_scalar_ref_one(o + (size_t)t * out_dim, xq + (size_t)t * in_dim,
                              xqs + (size_t)t * ss, w, in_dim, out_dim);
}

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
static float epi_q41(const int32_t *accb, const float *xsb, const float *xgb,
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

#if defined(__MMX__)

#define LZ_Q41_ACC(DOT32, PF)                                          \
    for (s = 0; s < nb; s++) {                                         \
        const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16; \
        if (pf_ < wend) PF(pf_);                                       \
        acc[s] = DOT32(wn + (size_t)s * 16, xwt + (size_t)s * 32);     \
    }

#define LZ_Q41_PFSEL(DOT32)                                              \
    do {                                                                 \
        int pf_m_ = lz_prefetch_mode();                                  \
        if (pf_m_ == LZ_PF_AMD)       { LZ_Q41_ACC(DOT32, LZ_PFI_AMD); }  \
        else if (pf_m_ == LZ_PF_LOAD) { LZ_Q41_ACC(DOT32, LZ_PFI_LOAD); } \
        else if (pf_m_ == LZ_PF_NONE) { LZ_Q41_ACC(DOT32, LZ_PFI_NONE); } \
        else                          { LZ_Q41_ACC(DOT32, LZ_PFI_NTA); }  \
    } while (0)

#ifdef __WATCOMC__
/* Same 128-element step as Q8_0: fixed overhead amortized from once per
   32 elements to once per 128, integer results bit-identical. */
#define LZ_Q41_GROUP4(DOT128)                                              \
    do {                                                                   \
        for (s = 0; s < nb; s += 4) {                                      \
            const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16; \
            if (pf_ < wend) {                                              \
                int pf_m_ = lz_prefetch_mode();                            \
                if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);             \
                else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);            \
                else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);            \
                else                          LZ_PFI_NTA(pf_);             \
            }                                                              \
            DOT128(wn + (size_t)s * 16, xwt + (size_t)s * 32, acc + s);    \
        }                                                                  \
    } while (0)

static void row_q41_mmx_asm(int32_t *acc32, const void *w4, const void *w2,
                            const int16_t *xw, int nb, int nt, int in_dim,
                            const void *pf_end,
                         const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *wend = (const unsigned char *)pf_end;
    int tk, s;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Pair whenever a second token exists - same trade as Q8 and
           worth more here (weight side 16 of 40 ops against Q8's 12 of
           36). Not measured on target; see row_q8_mmx_asm. */
        if (nt - tk >= 2 && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            int s2;
            for (s2 = 0; s2 < nb; s2++) {
                const unsigned char *pf_ = wn + (size_t)(s2 + LZ_PF_DIST) * 16;
                if (pf_ < wend) {
                    int pf_m_ = lz_prefetch_mode();
                    if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);
                    else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);
                    else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);
                    else                          LZ_PFI_NTA(pf_);
                }
                lz_dot32_q41_asm_2(wn + (size_t)s2 * 16,
                                   xwt + (size_t)s2 * 32,
                                   xw2 + (size_t)s2 * 32,
                                   acc + s2, accb + s2);
            }
            tk++;
            continue;
        }
        if ((nb & 3) == 0) LZ_Q41_GROUP4(lz_dot128_q41_asm);
        else               LZ_Q41_PFSEL(dot32_q41_mmx);
    }
}

static void row_q41_sse2_asm(int32_t *acc32, const void *w4, const void *w2,
                             const int16_t *xw, int nb, int nt, int in_dim,
                             const void *pf_end,
                         const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *wend = (const unsigned char *)pf_end;
    int tk, s;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
#if LZ_SSE2_GROUP
        if ((nb & 3) == 0) LZ_Q41_GROUP4(lz_dot128_q41_sse2_asm);
#else
        if ((nb & 3) == 0) LZ_Q41_GROUP4(lz_dot128_q41_asm);
#endif
        else               LZ_Q41_PFSEL(dot32_q41_sse2a);
    }
}

#else  /* gcc: intrinsics */

static void row_q41_mmx_intrin(int32_t *acc32, const void *w4, const void *w2,
                               const int16_t *xw, int nb, int nt, int in_dim,
                               const void *pf_end,
                         const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *wend = (const unsigned char *)pf_end;
    int tk, s;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Pair with the next token so ONE nibble unpack serves both. */
        if (tk + 1 < nt && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            for (s = 0; s < nb; s++) {
                const unsigned char *pf_ = wn + (size_t)(s + LZ_PF_DIST) * 16;
                if (pf_ < wend) LZ_PFI_NTA(pf_);
                dot32_q41_mmx_2(wn + (size_t)s * 16,
                                xwt + (size_t)s * 32,
                                xw2 + (size_t)s * 32,
                                acc + s, accb + s);
            }
            tk++;
        } else {
            LZ_Q41_ACC(dot32_q41_mmx, LZ_PFI_NTA);
        }
    }
}

#endif /* __WATCOMC__ */
#endif /* __MMX__ */

#if defined(__SSE2__) && !defined(__WATCOMC__)
static void row_q41_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                                const int16_t *xw, int nb, int nt, int in_dim,
                                const void *pf_end,
                         const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s + 3 < nb; s += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + s),
                fold4_sse2(
                part32_q41(wn + (size_t)(s + 0) * 16, xwt + (size_t)(s + 0) * 32),
                part32_q41(wn + (size_t)(s + 1) * 16, xwt + (size_t)(s + 1) * 32),
                part32_q41(wn + (size_t)(s + 2) * 16, xwt + (size_t)(s + 2) * 32),
                part32_q41(wn + (size_t)(s + 3) * 16, xwt + (size_t)(s + 3) * 32)));
        }
        for (; s < nb; s++) {
            __m128i p = part32_q41(wn + (size_t)s * 16, xwt + (size_t)s * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[s] = _mm_cvtsi128_si32(p);
        }
    }
}
#endif

static const lz_rowfn LZ_ROW_Q41[LZ_ROW_N] = {
#if defined(__MMX__) && !defined(__WATCOMC__)
    row_q41_mmx_intrin,
#else
    NULL,
#endif
    NULL,        /* sse-intrin: no SSE1 op applies, see lz_row_pick */
#if defined(__SSE2__) && !defined(__WATCOMC__)
    row_q41_sse2_intrin,
#else
    NULL,
#endif
#if defined(__MMX__) && defined(__WATCOMC__)
    row_q41_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_q41_sse2_asm
#else
    NULL, NULL, NULL
#endif
};

static void matmul_q41_impl(float *o, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    int i, g, s, t, tk;
    int r  = w->gs / 32;              /* 32 sub-blocks per weight group */
    int ng = in_dim / w->gs;          /* weight-group count */
    int nb = in_dim / 32;             /* 32-sub-block count */
    lz_rowfn row;

    row = lz_row_pick(LZ_ROW_Q41);
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    /* The zero term depends only on activations; computed per token,
       shared by all of that token's output rows */
    for (t = 0; t < nt; t++) {
        const int8_t *xqt = xq + (size_t)t * in_dim;
        const float *xst = xqs + (size_t)t * nb;
        float *xgt = g_xg + (size_t)t * ng;
        for (g = 0; g < ng; g++) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int sb = g * r + s, k;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqt[sb * 32 + k];
                a += lz_i32f(sx) * xst[sb];
            }
            xgt[g] = a;
        }
    }

    for (i = 0; i < out_dim; i++) {
        const unsigned char *wn =
            (const unsigned char *)w->q + (size_t)i * in_dim / 2;
        const unsigned char *wend =            /* prefetch upper bound */
            (const unsigned char *)w->q + (size_t)w->n / 2;
        const float *wd = w->scale + (size_t)i * ng;
        const float *wm = w->zero + (size_t)i * ng;
        /* Weight row loads once and serves all nt tokens; one EMMS per
           row, not per token - the epilogue below is x87 on Watcom. */
        row(g_acc32, wn, NULL, g_xw, nb, nt, in_dim, wend, NULL);
        _mm_empty();
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] =
                epi_q41(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                        g_xg + (size_t)tk * ng, wd, wm, ng, r);
    }
}

/* ---- Q6_1 row kernels, one per (ISA x impl) ---------------------------
   The only format that uses BOTH plane pointers and BOTH prefetch
   bounds. The 2-bit plane needs its own: `pf_end` stops exactly at the
   plane boundary. At 22% of all weight bytes on the recipe of record,
   its miss frequency is half the 4-bit plane's, not 1/16 like a scale
   array: it IS a hot stream, just a narrower one. Both bounds are real,
   not belt-and-braces: LZ_PF_LOAD is a REAL dummy load on PII and
   faults past the end of the tensor. */

#if defined(__MMX__)

/* Both planes' prefetch, one tier lookup. */
#define LZ_Q61_PF(S)                                                    \
    do {                                                                \
        const unsigned char *pf_ = wn + (size_t)((S) + LZ_PF_DIST) * 16; \
        const unsigned char *pf2_ = w2 + (size_t)((S) + LZ_PF_DIST) * 8; \
        int pf_m_ = lz_prefetch_mode();                                 \
        if (pf_ < wend) {                                               \
            if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);              \
            else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);             \
            else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);             \
            else                          LZ_PFI_NTA(pf_);              \
        }                                                               \
        if (pf2_ < wend2) {                                             \
            if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf2_);             \
            else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf2_);            \
            else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf2_);            \
            else                          LZ_PFI_NTA(pf2_);             \
        }                                                               \
    } while (0)

#define LZ_Q61_ACC(DOT32)                                               \
    for (s = 0; s < nb; s++) {                                          \
        LZ_Q61_PF(s);                                                   \
        acc[s] = DOT32(wn + (size_t)s * 16, w2 + (size_t)s * 8,         \
                       xwt + (size_t)s * 32);                           \
    }

#ifdef __WATCOMC__
#define LZ_Q61_GROUP4(DOT128)                                           \
    do {                                                                \
        for (s = 0; s < nb; s += 4) {                                   \
            LZ_Q61_PF(s);                                               \
            DOT128(wn + (size_t)s * 16, w2 + (size_t)s * 8,             \
                   xwt + (size_t)s * 32, acc + s);                      \
        }                                                               \
    } while (0)

static void row_q61_mmx_asm(int32_t *acc32, const void *w4, const void *w2v,
                            const int16_t *xw, int nb, int nt, int in_dim,
                            const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    const unsigned char *wend = (const unsigned char *)pf_end;
    const unsigned char *wend2 = (const unsigned char *)pf_end2;
    int tk, s;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Pair whenever a second token exists. Q6_1 gains the most of
           the three formats (weight side 61% of the kernel's ops);
           the gcc twin measured 1.271x on prefill. Not measured on
           target - see row_q8_mmx_asm. */
        if (nt - tk >= 2 && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            int s2;
            for (s2 = 0; s2 < nb; s2++) {
                LZ_Q61_PF(s2);
                lz_dot32_q61_asm_2(wn + (size_t)s2 * 16, w2 + (size_t)s2 * 8,
                                   xwt + (size_t)s2 * 32,
                                   xw2 + (size_t)s2 * 32,
                                   acc + s2, accb + s2);
            }
            tk++;
            continue;
        }
        if ((nb & 3) == 0) LZ_Q61_GROUP4(lz_dot128_q61_asm);
        else               LZ_Q61_ACC(dot32_q61_mmx);
    }
}

static void row_q61_sse2_asm(int32_t *acc32, const void *w4, const void *w2v,
                             const int16_t *xw, int nb, int nt, int in_dim,
                             const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    const unsigned char *wend = (const unsigned char *)pf_end;
    const unsigned char *wend2 = (const unsigned char *)pf_end2;
    int tk, s;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
#if LZ_SSE2_GROUP
        if ((nb & 3) == 0) LZ_Q61_GROUP4(lz_dot128_q61_sse2_asm);
#else
        if ((nb & 3) == 0) LZ_Q61_GROUP4(lz_dot128_q61_asm);
#endif
        /* MEASURED ON TARGET: on a real Pentium M the SSE2 kernel is
           20-30% faster at kernel level and no worse end-to-end. Uop
           counts compare only WITHIN one instruction set; a 128-bit uop
           and a 64-bit uop are not the same unit of work, so the
           uop-count argument does not apply across instruction sets. */
        else               LZ_Q61_ACC(dot32_q61_sse2a);
    }
}

#else  /* gcc: intrinsics */

static void row_q61_mmx_intrin(int32_t *acc32, const void *w4, const void *w2v,
                               const int16_t *xw, int nb, int nt, int in_dim,
                               const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    const unsigned char *wend = (const unsigned char *)pf_end;
    const unsigned char *wend2 = (const unsigned char *)pf_end2;
    int tk, s;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        /* Q6_1 gains the most of the three formats from pairing: the
           weight side is 61% of the kernel's ops against Q4_1's 40% and
           Q8's 33%, and on the recipe of record it also carries the
           most bytes. Both planes' prefetches are issued once for the
           pair - the second token reads bytes the first just pulled. */
        /* No LZ_Q61_FORCE_SSE2 macro here: a compile-time switch whose
           only job is to make the SSE2 kernel reachable on a machine
           where the dispatch would not pick it has nothing left to do,
           since one gcc build carries both tiers and `--kernel sse2`
           selects row_q61_sse2_intrin for real. A knob that duplicates
           a knob is worse than no knob - iron law 9 asks for a switch
           plus a gate proving it changes something, and this one would
           change nothing. */
        if (tk + 1 < nt && g_pair) {
            const int16_t *xw2 = xwt + in_dim;
            int32_t *accb = acc + nb;
            for (s = 0; s < nb; s++) {
                LZ_Q61_PF(s);
                dot32_q61_mmx_2(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                xwt + (size_t)s * 32, xw2 + (size_t)s * 32,
                                acc + s, accb + s);
            }
            tk++;
            continue;
        }
        LZ_Q61_ACC(dot32_q61_mmx);
    }
}

#endif /* __WATCOMC__ */
#endif /* __MMX__ */

#if defined(__SSE2__) && !defined(__WATCOMC__)
static void row_q61_sse2_intrin(int32_t *acc32, const void *w4, const void *w2v,
                                const int16_t *xw, int nb, int nt, int in_dim,
                                const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    int tk, s;
    (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++) {
            __m128i p = part32_q61(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                   xwt + (size_t)s * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[s] = _mm_cvtsi128_si32(p);
        }
    }
}
#endif

static const lz_rowfn LZ_ROW_Q61[LZ_ROW_N] = {
#if defined(__MMX__) && !defined(__WATCOMC__)
    row_q61_mmx_intrin,
#else
    NULL,
#endif
    NULL,        /* sse-intrin: no SSE1 op applies, see lz_row_pick */
#if defined(__SSE2__) && !defined(__WATCOMC__)
    row_q61_sse2_intrin,
#else
    NULL,
#endif
#if defined(__MMX__) && defined(__WATCOMC__)
    row_q61_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_q61_sse2_asm
#else
    NULL, NULL, NULL
#endif
};

static void matmul_q61_impl(float *o, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, g, s, t, tk;
    int gs = w->gs, r = gs / 32, ng = in_dim / gs;
    int nb = in_dim / 32;
    const unsigned char *p4b = (const unsigned char *)w->q;
    const unsigned char *p2b = p4b + (size_t)w->n / 2;
    lz_rowfn row;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_Q61);
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }

    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];
    for (t = 0; t < nt; t++) {
        const int8_t *xqt = xq + (size_t)t * in_dim;
        const float *xst = xqs + (size_t)t * nb;
        float *xgt = g_xg + (size_t)t * ng;
        for (g = 0; g < ng; g++) {
            float a = 0.0f;
            for (s = 0; s < r; s++) {
                int sb = g * r + s, k;
                int32_t sx = 0;
                for (k = 0; k < 32; k++) sx += xqt[sb * 32 + k];
                a += lz_i32f(sx) * xst[sb];
            }
            xgt[g] = a;
        }
    }

    for (i = 0; i < out_dim; i++) {
        const unsigned char *wn = p4b + (size_t)i * in_dim / 2;
        const unsigned char *w2 = p2b + (size_t)i * in_dim / 4;
        const float *wd = w->scale + (size_t)i * ng;
        const float *wm = w->zero + (size_t)i * ng;
        row(g_acc32, wn, w2, g_xw, nb, nt, in_dim,
            p2b,                        /* 4-bit plane ends where 2-bit starts */
            p2b + (size_t)w->n / 4);    /* 2-bit plane's own bound */
        _mm_empty();
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] =
                epi_q41(g_acc32 + (size_t)tk * nb, xqs + (size_t)tk * nb,
                        g_xg + (size_t)tk * ng, wd, wm, ng, r);
    }
}

#if defined(__MMX__)
/* Q16_0 SIMD matmul. Same shape as matmul_q8_impl - weight row loads
   once, nt tokens' int32 sub-block sums computed under MMX, ONE emms,
   then the x87 float epilogue - with two differences that both come
   from the weights already being int16:

     - no x256 product scaling, so no 1/256 to cancel at the end;
     - the weight row is int16, so a 32-element sub-block is 64 bytes,
       i.e. two cache lines rather than one. Prefetch distance is
       therefore expressed in the same units as the Q8 path (32-element
       sub-blocks) and lands twice as far ahead in bytes, which is what
       we want: the byte stream is twice as fast.

   The epilogue is character-for-character the non-q4 branch of
   matmul_scalar_ref_one, including the (a0+a2)+(a1+a3) reduction order
   and lz_i32f() on every accumulator. That is the contract (rule 2):
   this function is only allowed to be faster, never different. */
/* ---- Q16_0 row kernels -------------------------------------------------
   int16 weights, so no int8->int16 unpack and no x256 product scaling -
   the kernels return the raw int32 dot and the epilogue's `post` is 1.0.
   No sse2-intrin slot yet: part32_q16 does not exist. */

#define LZ_Q16_PF(G)                                                      \
    do {                                                                  \
        const int16_t *pf_ = wr + (size_t)((G) + LZ_PF_DIST) * 32;        \
        if (pf_ < wend) {                                                 \
            int pf_m_ = lz_prefetch_mode();                               \
            if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);                \
            else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);               \
            else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);               \
            else                          LZ_PFI_NTA(pf_);                \
        }                                                                 \
    } while (0)

#define LZ_Q16_ACC(DOT32)                                                 \
    for (g = 0; g < nb; g++) {                                            \
        LZ_Q16_PF(g);                                                     \
        acc[g] = DOT32(wr + (size_t)g * 32, xwt + (size_t)g * 32);        \
    }

#ifdef __WATCOMC__
#define LZ_Q16_GROUP4(DOT128)                                             \
    do {                                                                  \
        for (g = 0; g < nb; g += 4) {                                     \
            LZ_Q16_PF(g);                                                 \
            DOT128(wr + (size_t)g * 32, xwt + (size_t)g * 32, acc + g);   \
        }                                                                 \
    } while (0)

static void row_q16_mmx_asm(int32_t *acc32, const void *w4, const void *w2,
                            const int16_t *xw, int nb, int nt, int in_dim,
                            const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    const int16_t *wend = (const int16_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        if ((nb & 3) == 0) LZ_Q16_GROUP4(lz_dot128_q16_asm);
        else               LZ_Q16_ACC(dot32_q16_mmx);
    }
}

static void row_q16_sse2_asm(int32_t *acc32, const void *w4, const void *w2,
                             const int16_t *xw, int nb, int nt, int in_dim,
                             const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    const int16_t *wend = (const int16_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
#if LZ_SSE2_GROUP
        if ((nb & 3) == 0) LZ_Q16_GROUP4(lz_dot128_q16_sse2_asm);
#else
        if ((nb & 3) == 0) LZ_Q16_GROUP4(lz_dot128_q16_asm);
#endif
        else               LZ_Q16_ACC(dot32_q16_sse2a);
    }
}

#else  /* gcc: intrinsics */

static void row_q16_mmx_intrin(int32_t *acc32, const void *w4, const void *w2,
                               const int16_t *xw, int nb, int nt, int in_dim,
                               const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    const int16_t *wend = (const int16_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        LZ_Q16_ACC(dot32_q16_mmx);
    }
}

#endif /* __WATCOMC__ */

#if defined(__SSE2__) && !defined(__WATCOMC__)
static void row_q16_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                                const int16_t *xw, int nb, int nt, int in_dim,
                                const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    int tk, g;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (g = 0; g + 3 < nb; g += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + g),
                fold4_sse2(
                part32_q16(wr + (size_t)(g + 0) * 32, xwt + (size_t)(g + 0) * 32),
                part32_q16(wr + (size_t)(g + 1) * 32, xwt + (size_t)(g + 1) * 32),
                part32_q16(wr + (size_t)(g + 2) * 32, xwt + (size_t)(g + 2) * 32),
                part32_q16(wr + (size_t)(g + 3) * 32, xwt + (size_t)(g + 3) * 32)));
        }
        for (; g < nb; g++) {           /* tail when nb % 4 != 0 */
            __m128i p = part32_q16(wr + (size_t)g * 32, xwt + (size_t)g * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[g] = _mm_cvtsi128_si32(p);
        }
    }
}
#endif

static const lz_rowfn LZ_ROW_Q16[LZ_ROW_N] = {
#ifndef __WATCOMC__
    row_q16_mmx_intrin,
#else
    NULL,
#endif
    NULL,        /* sse-intrin: no SSE1 op applies, see lz_row_pick */
#if defined(__SSE2__) && !defined(__WATCOMC__)
    row_q16_sse2_intrin,
#else
    NULL,
#endif
#ifdef __WATCOMC__
    row_q16_mmx_asm,
    NULL,        /* sse-asm: ditto */
    row_q16_sse2_asm
#else
    NULL, NULL, NULL
#endif
};

static void matmul_q16_impl(float *o, const int8_t *xq, const float *xqs,
                            const LZTensor *w, int in_dim, int out_dim, int nt) {
    int i, tk;
    int r = w->gs / 32;
    int ng = in_dim / 32;
    const int16_t *wbase = (const int16_t *)(const void *)w->q;
    const int16_t *wend = wbase + (size_t)w->n;   /* prefetch upper bound */
    int t;
    lz_rowfn row;

    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    row = lz_row_pick(LZ_ROW_Q16);
    if (!row || in_dim > LZ_MM_WIDEN_MAX) {
        matmul_scalar_ref(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
    for (t = 0; t < nt * in_dim; t++) g_xw[t] = (int16_t)xq[t];

    for (i = 0; i < out_dim; i++) {
        const int16_t *wr = wbase + (size_t)i * in_dim;
        const float *ws = w->scale + (size_t)(i * in_dim) / w->gs;
        row(g_acc32, wr, NULL, g_xw, ng, nt, in_dim, wend, NULL);
        _mm_empty();                    /* exit MMX state before x87 */
        /* post = 1.0: the Q16_0 kernels do not scale by 256 the way the
           Q8_0 ones do. Otherwise term for term the same epilogue. */
        for (tk = 0; tk < nt; tk++)
            o[(size_t)tk * out_dim + i] =
                epi_q8(g_acc32 + (size_t)tk * ng, xqs + (size_t)tk * ng,
                       ws, ng, r, 1.0f);
    }
}

#endif /* Q16_0 SIMD matmul */

/* Matmul consuming already-quantized activations: the forward hot path.
   xq/xqs quantized by the caller at lz_act_gs()'s group (one activation
   may be reused by several matmuls). f32 weights fall back to f32 -
   one call site serves all weight representations. */
void lz_matmul_q8(float *o, const float *x, const int8_t *xq,
                  const float *xqs, const LZTensor *w,
                  int in_dim, int out_dim) {
    lz_matmul_q8_nt(o, x, xq, xqs, w, in_dim, out_dim, 1);
}

void lz_matmul_q8_nt(float *o, const float *x, const int8_t *xq,
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
       measurement. The iron law requires every operator to have C /
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
#if defined(__MMX__)
    /* Q16_0 has no non-widened fallback: unlike Q8_0 it always needs the
       activation expanded to int16, so in_dim past the buffer cap goes to
       the scalar reference. The gcc/SSE2-only dev build has no Q16_0 SIMD
       kernel either and lands there too - that is deliberate, not an
       oversight. The differential oracle rule 2 asks for is the gcc build
       WITH -DLZ_USE_MMX=1 -D__MMX__, which compiles the intrinsics twin
       and exercises this same path. */
    if (w->dtype == LZ_FMT_Q16_0 && w->gs >= 32 && (w->gs % 32) == 0 &&
        (in_dim % w->gs) == 0 && in_dim <= LZ_MM_WIDEN_MAX) {
        matmul_q16_impl(o, xq, xqs, w, in_dim, out_dim, nt);
        return;
    }
#endif
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
    lz_matmul_q8(o, x, xq, xqs, w, in_dim, out_dim);
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

#if defined(__SSE2__) && !defined(__WATCOMC__)
/* 4-wide float RoPE. Each (i, h) pair is an independent IEEE mul/add,
   so per-element results are bit-identical to the scalar path; only the
   loop order changes (head outer, i inner) to stream one head's two
   rotated halves contiguously. The cos/sin row is interleaved
   [c0,s0,c1,s1,...], so one 8-float load + two shuffles give the c and
   s vectors. half % 4 tail falls back to scalar. */
static void lz_rope_sse2(float *v, int n_heads, int head_dim,
                         int rotary_dim, int pos, const float *cs) {
    int half = rotary_dim / 2;
    int h, i;
    const float *row = cs + (size_t)pos * half * 2;
    for (h = 0; h < n_heads; h++) {
        float *base = v + (size_t)h * head_dim;
        for (i = 0; i + 4 <= half; i += 4) {
            __m128 lo = _mm_loadu_ps(row + i * 2);
            __m128 hi = _mm_loadu_ps(row + i * 2 + 4);
            __m128 c = _mm_shuffle_ps(lo, hi, 0x88); /* [c0,c1,c2,c3] */
            __m128 s = _mm_shuffle_ps(lo, hi, 0xDD); /* [s0,s1,s2,s3] */
            __m128 x0 = _mm_loadu_ps(base + i);
            __m128 x1 = _mm_loadu_ps(base + i + half);
            __m128 n0 = _mm_sub_ps(_mm_mul_ps(x0, c), _mm_mul_ps(x1, s));
            __m128 n1 = _mm_add_ps(_mm_mul_ps(x1, c), _mm_mul_ps(x0, s));
            _mm_storeu_ps(base + i, n0);
            _mm_storeu_ps(base + i + half, n1);
        }
        for (; i < half; i++) {
            float c0 = row[i * 2], s0 = row[i * 2 + 1];
            float x0 = base[i], x1 = base[i + half];
            base[i]        = x0 * c0 - x1 * s0;
            base[i + half] = x1 * c0 + x0 * s0;
        }
    }
}
#endif



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
    float n2 = 0.0f, s, inv;
    int i;

    for (i = 0; i < n; i++) n2 += x[i] * x[i];
    n2 = (float)sqrt((double)n2);
    /* sqrt(n) is exact for the power-of-two head dims this path serves;
       the divide has a runtime divisor, so both compilers emit a real
       FDIV rather than one strength-reducing it (iron law 6). */
    s = n2 / (float)sqrt((double)n);
    if (scale) *scale = s;
    inv = (s > 1e-20f) ? (1.0f / s) : 0.0f;

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
       care iron law 6 demands of the multiply-by-reciprocal paths. */
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
    /* The guard must match lz_rope_sse2's DEFINITION guard exactly: the
       definition is `__SSE2__ && !LZ_USE_MMX`. Guarding this call by
       `__SSE2__` alone breaks `make llama98-mmx` (-DLZ_USE_MMX=1): the
       definition compiles out, the call stays, and an implicit
       declaration fails the whole MMX target - the Makefile's PII main
       path. Two guards for one symbol is how that happens. */
#if defined(__SSE2__) && !defined(__WATCOMC__)
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
#endif
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

   -DLZ_ATTN_FIXED=1 enables it; default off pending the agreement
   measurement. Note the
   error-magnitude argument that would normally settle this is NOT usable
   here: the logit-difference metric saturates, so "this error is 200x
   that one" says nothing about which output is better (section 6.1). */
#ifndef LZ_ATTN_FIXED
#define LZ_ATTN_FIXED 0
#endif

#if LZ_ATTN_FIXED
/* Largest head_dim served by this path. 256 covers the 0.8B; anything
   larger falls back to the float loops rather than truncating. */
#define LZ_ATTN_MAX_HD 256
#define LZ_WSUM_CHUNK  512

#if (LZ_ATTN_FIXED & 2)
#if defined(__WATCOMC__)
/* Generated code; do not hand-edit.
   acc32[d] += rowA[d]*coef[0] + rowB[d]*coef[1] for d in [0,32).
   coef is [ckA,ckB,ckA,ckB] so one pmaddwd covers a 2-int32 lane pair.
   Reads [eax]/[edx]/[ecx], writes only [ebx] and MMX - every offset is a
   literal, no GP register is modified, so `__modify [8087]` is the truth
   and not an `__exact []`-style lie (rule 2 item 3).
   Emits no emms; the driver does one per chunk. */
extern void lz_wsum_pair_asm(const int8_t *rowA, const int8_t *rowB,
                const int16_t *coef, int32_t *acc32);
#pragma aux lz_wsum_pair_asm = \
    ".586" \
    "movq      mm7, [ecx]" \
    "movq      mm0, [eax]" \
    "movq      mm1, [edx]" \
    "movq      mm2, mm0" \
    "punpcklbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx]" \
    "paddd     mm5, mm4" \
    "movq      [ebx], mm5" \
    "movq      mm5, [ebx+8]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+8], mm5" \
    "movq      mm2, mm0" \
    "punpckhbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpckhbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+16]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+16], mm5" \
    "movq      mm5, [ebx+24]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+24], mm5" \
    "movq      mm0, [eax+8]" \
    "movq      mm1, [edx+8]" \
    "movq      mm2, mm0" \
    "punpcklbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+32]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+32], mm5" \
    "movq      mm5, [ebx+40]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+40], mm5" \
    "movq      mm2, mm0" \
    "punpckhbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpckhbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+48]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+48], mm5" \
    "movq      mm5, [ebx+56]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+56], mm5" \
    "movq      mm0, [eax+16]" \
    "movq      mm1, [edx+16]" \
    "movq      mm2, mm0" \
    "punpcklbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+64]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+64], mm5" \
    "movq      mm5, [ebx+72]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+72], mm5" \
    "movq      mm2, mm0" \
    "punpckhbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpckhbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+80]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+80], mm5" \
    "movq      mm5, [ebx+88]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+88], mm5" \
    "movq      mm0, [eax+24]" \
    "movq      mm1, [edx+24]" \
    "movq      mm2, mm0" \
    "punpcklbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpcklbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+96]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+96], mm5" \
    "movq      mm5, [ebx+104]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+104], mm5" \
    "movq      mm2, mm0" \
    "punpckhbw mm2, mm2" \
    "psraw     mm2, 8" \
    "movq      mm3, mm1" \
    "punpckhbw mm3, mm3" \
    "psraw     mm3, 8" \
    "movq      mm4, mm2" \
    "punpcklwd mm4, mm3" \
    "punpckhwd mm2, mm3" \
    "pmaddwd   mm4, mm7" \
    "pmaddwd   mm2, mm7" \
    "movq      mm5, [ebx+112]" \
    "paddd     mm5, mm4" \
    "movq      [ebx+112], mm5" \
    "movq      mm5, [ebx+120]" \
    "paddd     mm5, mm2" \
    "movq      [ebx+120], mm5" \
    __parm [__eax] [__edx] [__ecx] [__ebx] \
    __modify [8087]
#define LZ_WSUM_PAIR(ra, rb, co, ac) lz_wsum_pair_asm(ra, rb, co, ac)

#elif defined(__MMX__)
/* Dev-build twin (rule 2), step for step the same as the assembly above:
   sign-extend two int8 rows to int16, interleave so one pmaddwd covers a
   (rowA,rowB) pair against (ckA,ckB), accumulate into acc32 in place.
   Emits no emms; the driver does one per chunk, and that chunk's
   row-pair loop contains no x87 (rule 6 item 6). */
static void lz_wsum_pair_mmx(const int8_t *rowA, const int8_t *rowB,
                             const int16_t *coef, int32_t *acc32) {
    __m64 c, ra, rb, a0, b0, p, q, acc;
    int off;
    memcpy(&c, coef, 8);
    for (off = 0; off < 32; off += 8) {
        memcpy(&ra, rowA + off, 8);
        memcpy(&rb, rowB + off, 8);

        a0 = _mm_srai_pi16(_mm_unpacklo_pi8(ra, ra), 8);
        b0 = _mm_srai_pi16(_mm_unpacklo_pi8(rb, rb), 8);
        p = _mm_unpacklo_pi16(a0, b0);        /* [a0,b0,a1,b1] */
        q = _mm_unpackhi_pi16(a0, b0);        /* [a2,b2,a3,b3] */
        p = _mm_madd_pi16(p, c);
        q = _mm_madd_pi16(q, c);
        memcpy(&acc, acc32 + off, 8);
        acc = _mm_add_pi32(acc, p);
        memcpy(acc32 + off, &acc, 8);
        memcpy(&acc, acc32 + off + 2, 8);
        acc = _mm_add_pi32(acc, q);
        memcpy(acc32 + off + 2, &acc, 8);

        a0 = _mm_srai_pi16(_mm_unpackhi_pi8(ra, ra), 8);
        b0 = _mm_srai_pi16(_mm_unpackhi_pi8(rb, rb), 8);
        p = _mm_unpacklo_pi16(a0, b0);
        q = _mm_unpackhi_pi16(a0, b0);
        p = _mm_madd_pi16(p, c);
        q = _mm_madd_pi16(q, c);
        memcpy(&acc, acc32 + off + 4, 8);
        acc = _mm_add_pi32(acc, p);
        memcpy(acc32 + off + 4, &acc, 8);
        memcpy(&acc, acc32 + off + 6, 8);
        acc = _mm_add_pi32(acc, q);
        memcpy(acc32 + off + 6, &acc, 8);
    }
}
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
#endif
#endif /* LZ_ATTN_FIXED & 2 */

#if (LZ_ATTN_FIXED & 1)
#if !defined(__WATCOMC__) && !(defined(__MMX__))
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
#endif

void lz_attn_score_q8(float *att, const float *qhh, int hd,
                      const int8_t *kc, const float *ks, int kvd,
                      int pos, float scale) {
    /* static, not stack (rule 6 item 4) */
    static int8_t  qq[LZ_ATTN_MAX_HD];
    static float   qs[LZ_ATTN_MAX_HD / 32];
    static int16_t qw[LZ_ATTN_MAX_HD];
    int ng = hd / 32, g, t, i;

    lz_quantize_q8(qhh, hd, 32, qq, qs);
    for (i = 0; i < hd; i++) qw[i] = (int16_t)qq[i];

    for (t = 0; t <= pos; t++) {
        const int8_t *kt = kc + (size_t)t * kvd;
        const float *kts = ks + (size_t)t * (kvd / 32);
        int32_t rr[LZ_ATTN_MAX_HD / 32];
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

        /* Phase 1 is MMX only: collect every group's raw dot before any
           x87 runs. Interleaving the two leaves MMX registers dirty
           across an x87 fld/fmul - the bug rule 6 item 6 describes. It
           showed 0 mismatches in a short bit-identity test and -nan in a
           longer one, on the Watcom build only. */
        for (g = 0; g < ng; g++)
            rr[g] = LZ_ATTN_DOT32(kt + (size_t)g * 32, qw + g * 32);
#if defined(__MMX__)
        _mm_empty();
#endif
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
                     int pos, float *cbuf, int16_t *cq) {
    int ng = hd / 32, g, T = pos + 1;

    for (g = 0; g < ng; g++) {
        int32_t acc32[32];
        float accf[32];
        float cmax = 0.0f, inv;
        int t, d, t0;

        for (t = 0; t < T; t++) {
            float c = att[t] * vs[(size_t)t * (kvd / 32) + g];
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
                    int qi = q8_round(cbuf[t] / (cmax * (1.0f / 32767.0f)));
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

        for (d = 0; d < 32; d++) accf[d] = 0.0f;
        for (t0 = 0; t0 < T; t0 += LZ_WSUM_CHUNK) {
            int tn = T - t0;
            int tp;
            if (tn > LZ_WSUM_CHUNK) tn = LZ_WSUM_CHUNK;
            memset(acc32, 0, sizeof(acc32));
            for (tp = t0; tp + 1 < t0 + tn; tp += 2) {
                int16_t coef[4];
                const int8_t *rowA = vc + (size_t)tp * kvd + (size_t)g * 32;
                const int8_t *rowB = vc + (size_t)(tp + 1) * kvd + (size_t)g * 32;
                coef[0] = cq[tp];     coef[1] = cq[tp + 1];
                coef[2] = cq[tp];     coef[3] = cq[tp + 1];
                LZ_WSUM_PAIR(rowA, rowB, coef, acc32);
            }
            if (tp < t0 + tn) {            /* odd leftover row in this chunk */
                int16_t coef[4];
                const int8_t *rowA = vc + (size_t)tp * kvd + (size_t)g * 32;
                coef[0] = cq[tp]; coef[1] = 0;
                coef[2] = cq[tp]; coef[3] = 0;
                LZ_WSUM_PAIR(rowA, rowA, coef, acc32);
            }
#if defined(__MMX__)
            _mm_empty();      /* no x87 inside the row-pair loop above */
#endif
            for (d = 0; d < 32; d++) accf[d] += lz_i32f(acc32[d]);
        }
        {
            float sscale = cmax * (1.0f / 32767.0f);
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
#ifndef LZ_GDN_FIXED
#define LZ_GDN_FIXED 1
#endif

#if LZ_GDN_FIXED
/* Table is int16_t[LZ_GDN_MAX_KD/2][8]; bump both together. A kd past this
   takes the float pass 1 rather than truncating silently. */
#define LZ_GDN_MAX_KD 256

#if defined(__MMX__)
#define LZ_GDN_HAVE_MMX 1
#endif

#if defined(LZ_GDN_HAVE_MMX)
#if defined(__WATCOMC__)
/* Generated code; do not hand-edit.
   Verified over 200000 random + extreme groups (state +/-127,
   coefficients +/-32767, kd 2..66 even) against the scalar reference
   with zero mismatches; the kpairs=0 boundary was tested separately.
   Emits no emms - the caller does, after this gg's 8 sub-window calls,
   because building the next gg's table uses x87. */
extern void lz_gdn1_x4_asm(const int8_t *rowA, int32_t vd,
                           const int16_t *ctab, int32_t kpairs,
                           int32_t *out8);
#pragma aux lz_gdn1_x4_asm = \
    ".586" \
    "pxor      mm4, mm4" \
    "pxor      mm5, mm5" \
    "pxor      mm6, mm6" \
    "pxor      mm7, mm7" \
    "test      ecx, ecx" \
    "jz        LZ_GDN1_DONE" \
    "LZ_GDN1_LOOP:" \
    "movd      mm0, [eax]" \
    "punpcklbw mm0, mm0" \
    "psraw     mm0, 8" \
    "movd      mm1, [eax+edx]" \
    "punpcklbw mm1, mm1" \
    "psraw     mm1, 8" \
    "movq      mm2, mm0" \
    "punpcklwd mm2, mm1" \
    "punpckhwd mm0, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm3, [ebx]" \
    "paddd     mm4, mm3" \
    "pmaddwd   mm2, [ebx+8]" \
    "paddd     mm6, mm2" \
    "movq      mm3, mm0" \
    "pmaddwd   mm3, [ebx]" \
    "paddd     mm5, mm3" \
    "pmaddwd   mm0, [ebx+8]" \
    "paddd     mm7, mm0" \
    "lea       eax, [eax+edx*2]" \
    "add       ebx, 16" \
    "dec       ecx" \
    "jnz       LZ_GDN1_LOOP" \
    "LZ_GDN1_DONE:" \
    "movq      [esi], mm4" \
    "movq      [esi+8], mm5" \
    "movq      [esi+16], mm6" \
    "movq      [esi+24], mm7" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [__eax __ebx __ecx 8087]
#else
/* Dev-build twin (rule 2), step for step the same as the assembly above:
   sign-extend two int8 rows to int16, interleave so one pmaddwd consumes a
   (rowA,rowB) pair against a (cA,cB) coefficient pair, four accumulators
   for u/w x the low/high halves of the 4-lane window. */
static void lz_gdn1_x4_asm(const int8_t *rowA, int32_t vd,
                           const int16_t *ctab, int32_t kpairs,
                           int32_t *out8) {
    __m64 accU0 = _mm_setzero_si64(), accU1 = _mm_setzero_si64();
    __m64 accW0 = _mm_setzero_si64(), accW1 = _mm_setzero_si64();
    const int8_t *rA = rowA;
    int32_t kp;
    for (kp = 0; kp < kpairs; kp++) {
        __m64 a, b, pl, ph, t, cu, cw;
        int32_t a32, b32;
        memcpy(&a32, rA, 4);
        memcpy(&b32, rA + vd, 4);
        a = _mm_cvtsi32_si64(a32);
        b = _mm_cvtsi32_si64(b32);
        a = _mm_unpacklo_pi8(a, a);
        a = _mm_srai_pi16(a, 8);           /* 4 int16, sign-extended */
        b = _mm_unpacklo_pi8(b, b);
        b = _mm_srai_pi16(b, 8);
        pl = _mm_unpacklo_pi16(a, b);      /* [a0,b0,a1,b1] */
        ph = _mm_unpackhi_pi16(a, b);      /* [a2,b2,a3,b3] */
        memcpy(&cu, ctab + (size_t)kp * 8,     8);
        memcpy(&cw, ctab + (size_t)kp * 8 + 4, 8);
        t = _mm_madd_pi16(pl, cu); accU0 = _mm_add_pi32(accU0, t);
        t = _mm_madd_pi16(pl, cw); accW0 = _mm_add_pi32(accW0, t);
        t = _mm_madd_pi16(ph, cu); accU1 = _mm_add_pi32(accU1, t);
        t = _mm_madd_pi16(ph, cw); accW1 = _mm_add_pi32(accW1, t);
        rA += (size_t)vd * 2;
    }
    memcpy(out8 + 0, &accU0, 8);
    memcpy(out8 + 2, &accU1, 8);
    memcpy(out8 + 4, &accW0, 8);
    memcpy(out8 + 6, &accW1, 8);
    _mm_empty();
}
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
#endif
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
    tb->su = (amk > 0.0f) ? amk * (1.0f / 32767.0f) : (1.0f / 32767.0f);
    tb->sw = (amq > 0.0f) ? amq * (1.0f / 32767.0f) : (1.0f / 32767.0f);
    ik = (amk > 0.0f) ? 32767.0f / amk : 0.0f;
    iq = (amq > 0.0f) ? 32767.0f / amq : 0.0f;
    tb->tail_cu = 0; tb->tail_cw = 0;
#if LZ_GDN_FIXED >= 2
    tb->su2 = tb->su * (1.0f / 32767.0f);
    tb->sw2 = tb->sw * (1.0f / 32767.0f);
    tb->tail2_cu = 0; tb->tail2_cw = 0;
#endif

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
#endif
    }
}

/* Odd tail row: pure integer, touches neither x87 nor MMX, so it is safe
   on either side of an emms. */
static void gdn_tail_row(const int8_t *sq_gg, int kd, int vd,
                         int32_t cu, int32_t cw,
                         int32_t *au_gg, int32_t *aw_gg) {
    if (!(kd & 1)) return;
    {
        const int8_t *row = sq_gg + (size_t)(kd - 1) * vd;
        int j;
        for (j = 0; j < 32; j++) {
            au_gg[j] += (int32_t)row[j] * cu;
            aw_gg[j] += (int32_t)row[j] * cw;
        }
    }
}

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
    _mm_empty();
}
#endif /* LZ_GDN_HAVE_MMX */

/* au/aw come back as scaled-up int32 with su/sw the matching dequant
   scale. The accumulator bound is 127*32767*256 ~= 1.06e9, far past 2^24,
   so the caller must convert with lz_i32f() - rule 6 item 7. */
static void gdn_pass1_fixed(const int8_t *sq,
#if LZ_GDN_STATE_2PLANE
                            const int8_t *sq_lo, int32_t *aul, int32_t *awl,
#endif
                            const float *ss,
                            const float *q, const float *k, int kd, int vd,
                            int32_t *au, int32_t *aw, float *su, float *sw,
                            int32_t *au2, int32_t *aw2, float *su2, float *sw2) {
    int gg, ng = vd / 32;
    /* static, not stack: 2-4 KB of table on a Win98 stack, per rule 6 item 4. */
    static LZGdnTable tb;
#if LZ_GDN_FIXED < 2
    (void)au2; (void)aw2; (void)su2; (void)sw2;
#endif
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
#endif
#if LZ_GDN_FIXED >= 2
        /* Second plane, same kernel, second table. The MMX gdn_sum_gg emits
           its own emms, so this runs a redundant one - kept rather than
           hoisted, because rule 6 item 6's precondition ("no x87 in the
           loop body") fails here: gdn_build_table for the NEXT gg is x87.
           One extra tag-word write per gg is not worth reopening that trap. */
        gdn_sum_gg(sg, kd, vd, tb.t2, tb.tail2_cu, tb.tail2_cw,
                   au2 + gg * 32, aw2 + gg * 32);
        su2[gg] = tb.su2; sw2[gg] = tb.sw2;
#endif
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
#endif

#if LZ_GDN_STATE_2PLANE
#if LZ_GDN_FIXED >= 2
#error "LZ_GDN_STATE_2PLANE with LZ_GDN_FIXED=2 is not implemented: it would be four kernel passes, and the coefficient plane was measured not to be the bottleneck. Use LZ_GDN_FIXED=0 or 1."
#endif

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
   subtract per element). */
void lz_gdn_quantize_2p(const float *x, int n, int gs,
                        int8_t *hi, int8_t *lo, float *s) {
    int g, k, ng = n / gs, fast;
#if defined(LZ_HAVE_Q8R_SIMD) || defined(LZ_HAVE_Q8R_SSE)
    int tier = ((gs & 31) == 0) ? lz_q8r_tier() : 0;
#ifdef LZ_Q8R_COUNT
    /* Which tier this call actually took, counted rather than inferred.
       Off unless a probe defines the macro.
       Exists because "the dispatch condition looks right" is not an
       answer to "which one ran" - the same distinction that let a whole
       tier of hand-written assembly stay out of the Watcom build for
       months (iron law 8's closing note). */
    extern long lz_q8r_hits[3];
    lz_q8r_hits[tier < 0 ? 0 : (tier > 2 ? 2 : tier)]++;
#endif
    static const float lo_mul = LZ_GDN_LO_SCALE;
    /* static, not stack (rule 6 item 4); one row's residual at most. */
    static float res[LZ_GDN_MAX_VD];
#endif
    for (g = 0; g < ng; g++) {
        const float *grp = x + (size_t)g * gs;
        int8_t *ho = hi + (size_t)g * gs;
        int8_t *lw = lo + (size_t)g * gs;
        float inv;
        fast = q8_group_scale(grp, gs, &s[g], &inv);
        if (!fast) {
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
#ifdef LZ_HAVE_Q8R_SIMD
            if (tier == 2) {
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_simd(grp + k, ho + k, &inv);
                for (k = 0; k < gs; k++)
                    res[k] = grp[k] * inv - (float)ho[k];
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_simd(res + k, lw + k, &lo_mul);
            } else
#endif
            {
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_sse(grp + k, ho + k, &inv);
                /* emms before the x87 residual arithmetic below, and
                   again before the next group's amax/inv - same reason as
                   the identical placement in lz_quantize_q8 (rule 6 item
                   6's precondition does not hold here). */
                _mm_empty();
                for (k = 0; k < gs; k++)
                    res[k] = grp[k] * inv - (float)ho[k];
                for (k = 0; k + 31 < gs; k += 32)
                    lz_q8round32_sse(res + k, lw + k, &lo_mul);
                _mm_empty();
            }
            continue;
        }
#endif
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
#endif

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
#if LZ_GDN_FIXED
/* 2^e and floor(log2|x|), both by integer manipulation of the exponent
   field. Deliberately not ldexpf/frexpf: those are libm, and iron law 2
   wants every transcendental in the engine so the two builds cannot
   disagree via somebody else's rounding. These two do no rounding at
   all - they read and write an exponent - so they are identical on both
   sides by construction, which is the whole reason the normalization
   below uses them. `x` must be normal; the caller guarantees it with
   the LZ_Q8_MIN_SCALE guard. */
static float pow2f(int e) {
    union { float f; uint32_t u; } b;
    if (e < -126) return 0.0f;
    if (e >  127) e = 127;
    b.u = (uint32_t)(e + 127) << 23;
    return b.f;
}

static int expof(float x) {
    union { float f; uint32_t u; } b;
    b.f = x;
    return (int)((b.u >> 23) & 0xFFu) - 127;
}

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

/* 16-BYTE-ALIGNED STORAGE FOR THE BLOCK, by hand, because neither
   compiler can be asked for it here. Watcom is built with -zp4, which
   caps member alignment at 4 - a union with a double would not even
   guarantee 8 - and C89/C99 have no alignment specifier. The MMX
   kernels survived that (an unaligned movq is legal, merely slower);
   SSE2's movdqa and its m128 shift-count operand FAULT, so the
   alignment has to be real rather than hoped for.

   One buffer for the whole engine: the two kernels are called back to
   back inside one group and nothing re-enters between them. */
static lz_p2_blk *p2_blk(void) {
    static char raw[sizeof(lz_p2_blk) + 15];
    size_t off = (size_t)((uintptr_t)(void *)raw & 15u);
    return (lz_p2_blk *)(void *)(raw + ((16u - off) & 15u));
}

#if defined(LZ_HAVE_P2_MMX_ASM)
#define LZ_HAVE_P2_MMX 1
/* Plain aliases, not wrappers: a #pragma aux routine is expanded inline
   at its call site, and a static wrapper around it would put a real call
   back in front of that. The types line up exactly - int8_t is signed
   char and int16_t is short on both targets. */
#define lz_p2_mul32_mmx    lz_p2_mul32_mmx_asm
#define lz_p2_split32_mmx  lz_p2_split32_mmx_asm

#elif defined(__MMX__)
#define LZ_HAVE_P2_MMX 1
/* The gcc half of iron law 8's MMX cell. Instruction for instruction the
   same kernel as the assembly in ops_kernel_p2.h - that file carries the
   reasoning (why pmaddwd contracts H and dq in one go, why the absmax
   costs five instructions per two lanes, why the lo clamp is a
   saturating subtract). Only the notation differs, and the two are never
   in the same binary.

   memcpy for every load and store, not a cast to __m64*: ops.c takes
   that route everywhere its kernels touch caller memory (see the note on
   the string.h include at the top), because the state planes belong to
   the caller and nothing here may assume their alignment. */
static void lz_p2_mul32_mmx(const int8_t *hi, const int8_t *lo,
                            const int16_t *dq, lz_p2_blk *blk) {
    __m64 zero = _mm_setzero_si64();
    __m64 acc1 = _mm_setzero_si64();
    __m64 acc2 = _mm_setzero_si64();
    __m64 mul;
    int q;

    memcpy(&mul, blk->mul, 8);
    for (q = 0; q < 8; q++) {
        __m64 hw, lw, H, d, p0, p1, s0, s1, a0, a1, m;
        int32_t hv, lv;

        memcpy(&hv, hi + q * 4, 4);
        memcpy(&lv, lo + q * 4, 4);
        hw = _mm_unpacklo_pi8(zero, _mm_cvtsi32_si64(hv));   /* hi * 256 */
        lw = _mm_cvtsi32_si64(lv);
        lw = _mm_unpacklo_pi8(lw, lw);
        lw = _mm_srai_pi16(lw, 8);                           /* lo, signed */
        H  = _mm_add_pi16(hw, lw);
        memcpy(&d, dq + q * 4, 8);
        p0 = _mm_madd_pi16(_mm_unpacklo_pi16(H, d), mul);
        p1 = _mm_madd_pi16(_mm_unpackhi_pi16(H, d), mul);
        memcpy(blk->a + q * 4,     &p0, 8);
        memcpy(blk->a + q * 4 + 2, &p1, 8);
        s0 = _mm_srai_pi32(p0, 31);
        a0 = _mm_sub_pi32(_mm_xor_si64(p0, s0), s0);         /* |A| */
        s1 = _mm_srai_pi32(p1, 31);
        a1 = _mm_sub_pi32(_mm_xor_si64(p1, s1), s1);
        m    = _mm_cmpgt_pi32(acc1, a0);
        acc1 = _mm_or_si64(_mm_and_si64(acc1, m), _mm_andnot_si64(m, a0));
        m    = _mm_cmpgt_pi32(acc2, a1);
        acc2 = _mm_or_si64(_mm_and_si64(acc2, m), _mm_andnot_si64(m, a1));
    }
    memcpy(blk->amax,     &acc1, 8);
    memcpy(blk->amax + 2, &acc2, 8);
}

static void lz_p2_split32_mmx(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m64 rnd, cnt, k128, kclp;
    int o;

    memcpy(&rnd,  blk->rnd,  8);
    memcpy(&cnt,  blk->cnt,  8);
    memcpy(&k128, blk->k128, 8);
    memcpy(&kclp, blk->kclp, 8);
    for (o = 0; o < 4; o++) {
        __m64 a0, a1, a2, a3, hn0, hn1, h0, h1, l0, l1, t;

        memcpy(&a0, blk->a + o * 8,     8);
        memcpy(&a1, blk->a + o * 8 + 2, 8);
        memcpy(&a2, blk->a + o * 8 + 4, 8);
        memcpy(&a3, blk->a + o * 8 + 6, 8);
        a0 = _mm_sra_pi32(_mm_add_pi32(a0, rnd), cnt);
        a1 = _mm_sra_pi32(_mm_add_pi32(a1, rnd), cnt);
        a2 = _mm_sra_pi32(_mm_add_pi32(a2, rnd), cnt);
        a3 = _mm_sra_pi32(_mm_add_pi32(a3, rnd), cnt);
        hn0 = _mm_packs_pi32(a0, a1);
        hn1 = _mm_packs_pi32(a2, a3);
        h0 = _mm_srai_pi16(_mm_add_pi16(hn0, k128), 8);
        h1 = _mm_srai_pi16(_mm_add_pi16(hn1, k128), 8);
        l0 = _mm_sub_pi16(hn0, _mm_slli_pi16(h0, 8));
        l1 = _mm_sub_pi16(hn1, _mm_slli_pi16(h1, 8));
        l0 = _mm_adds_pi16(_mm_subs_pi16(l0, kclp), kclp);   /* max(l, -127) */
        l1 = _mm_adds_pi16(_mm_subs_pi16(l1, kclp), kclp);
        t = _mm_packs_pi16(h0, h1);
        memcpy(oh + o * 8, &t, 8);
        t = _mm_packs_pi16(l0, l1);
        memcpy(ol + o * 8, &t, 8);
    }
}
#endif

/* ---- the SSE1 tier: one instruction of content, and it is the split ---
   The A term and the absmax have no SSE1 form at all (ops_kernel_p2.h
   enumerates SSE1's whole addition to the MMX register file against what
   this kernel needs), so this tier is the MMX mul32 paired with a split
   whose low-plane clamp is a single pmaxsw. Named for the ISA a machine
   must HAVE to run it, which is what dispatch turns on. */
#if defined(LZ_HAVE_P2_SSE_ASM)
#define LZ_HAVE_P2_SSE 1
#define lz_p2_split32_sse  lz_p2_split32_sse_asm

#elif defined(__SSE__) && defined(__MMX__) && !defined(__WATCOMC__)
#define LZ_HAVE_P2_SSE 1
#include <xmmintrin.h>
static void lz_p2_split32_sse(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m64 rnd, cnt, k128, kmin;
    int o;

    memcpy(&rnd,  blk->rnd,  8);
    memcpy(&cnt,  blk->cnt,  8);
    memcpy(&k128, blk->k128, 8);
    memcpy(&kmin, blk->kmin, 8);
    for (o = 0; o < 4; o++) {
        __m64 a0, a1, a2, a3, hn0, hn1, h0, h1, l0, l1, t;

        memcpy(&a0, blk->a + o * 8,     8);
        memcpy(&a1, blk->a + o * 8 + 2, 8);
        memcpy(&a2, blk->a + o * 8 + 4, 8);
        memcpy(&a3, blk->a + o * 8 + 6, 8);
        a0 = _mm_sra_pi32(_mm_add_pi32(a0, rnd), cnt);
        a1 = _mm_sra_pi32(_mm_add_pi32(a1, rnd), cnt);
        a2 = _mm_sra_pi32(_mm_add_pi32(a2, rnd), cnt);
        a3 = _mm_sra_pi32(_mm_add_pi32(a3, rnd), cnt);
        hn0 = _mm_packs_pi32(a0, a1);
        hn1 = _mm_packs_pi32(a2, a3);
        h0 = _mm_srai_pi16(_mm_add_pi16(hn0, k128), 8);
        h1 = _mm_srai_pi16(_mm_add_pi16(hn1, k128), 8);
        l0 = _mm_sub_pi16(hn0, _mm_slli_pi16(h0, 8));
        l1 = _mm_sub_pi16(hn1, _mm_slli_pi16(h1, 8));
        l0 = _mm_max_pi16(l0, kmin);      /* the whole SSE1 tier, twice */
        l1 = _mm_max_pi16(l1, kmin);
        t = _mm_packs_pi16(h0, h1);
        memcpy(oh + o * 8, &t, 8);
        t = _mm_packs_pi16(l0, l1);
        memcpy(ol + o * 8, &t, 8);
    }
}
#endif

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
#if defined(LZ_HAVE_P2_SSE2_ASM)
#define LZ_HAVE_P2_SSE2 1
#define lz_p2_mul32_sse2    lz_p2_mul32_sse2_asm
#define lz_p2_split32_sse2  lz_p2_split32_sse2_asm

#elif defined(__SSE2__) && !defined(__WATCOMC__)
#define LZ_HAVE_P2_SSE2 1
static void lz_p2_mul32_sse2(const int8_t *hi, const int8_t *lo,
                             const int16_t *dq, lz_p2_blk *blk) {
    __m128i zero = _mm_setzero_si128();
    __m128i acc1 = zero, acc2 = zero;
    __m128i mul  = _mm_load_si128((const __m128i *)(const void *)blk->mul);
    __m128i m;
    int q;

    for (q = 0; q < 4; q++) {
        __m128i hw, lw, H, d, p0, p1, s0, s1, a0, a1;

        hw = _mm_loadl_epi64((const __m128i *)(const void *)(hi + q * 8));
        hw = _mm_unpacklo_epi8(zero, hw);                    /* hi * 256 */
        lw = _mm_loadl_epi64((const __m128i *)(const void *)(lo + q * 8));
        lw = _mm_unpacklo_epi8(lw, lw);
        lw = _mm_srai_epi16(lw, 8);                          /* lo, signed */
        H  = _mm_add_epi16(hw, lw);
        d  = _mm_loadu_si128((const __m128i *)(const void *)(dq + q * 8));
        p0 = _mm_madd_epi16(_mm_unpacklo_epi16(H, d), mul);
        p1 = _mm_madd_epi16(_mm_unpackhi_epi16(H, d), mul);
        _mm_store_si128((__m128i *)(void *)(blk->a + q * 8),     p0);
        _mm_store_si128((__m128i *)(void *)(blk->a + q * 8 + 4), p1);
        s0 = _mm_srai_epi32(p0, 31);
        a0 = _mm_sub_epi32(_mm_xor_si128(p0, s0), s0);       /* |A| */
        s1 = _mm_srai_epi32(p1, 31);
        a1 = _mm_sub_epi32(_mm_xor_si128(p1, s1), s1);
        m    = _mm_cmpgt_epi32(acc1, a0);
        acc1 = _mm_or_si128(_mm_and_si128(acc1, m), _mm_andnot_si128(m, a0));
        m    = _mm_cmpgt_epi32(acc2, a1);
        acc2 = _mm_or_si128(_mm_and_si128(acc2, m), _mm_andnot_si128(m, a1));
    }
    /* Fold to the four lanes the caller reads. MMX leaves two 2-lane
       accumulators there and SSE2 one 4-lane one; either way the caller
       folds four int32, so the two tiers hand back the same shape. */
    m    = _mm_cmpgt_epi32(acc1, acc2);
    acc1 = _mm_or_si128(_mm_and_si128(acc1, m), _mm_andnot_si128(m, acc2));
    _mm_store_si128((__m128i *)(void *)blk->amax, acc1);
}

static void lz_p2_split32_sse2(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m128i rnd  = _mm_load_si128((const __m128i *)(const void *)blk->rnd);
    __m128i cnt  = _mm_load_si128((const __m128i *)(const void *)blk->cnt);
    __m128i k128 = _mm_load_si128((const __m128i *)(const void *)blk->k128);
    __m128i kclp = _mm_load_si128((const __m128i *)(const void *)blk->kclp);
    int o;

    for (o = 0; o < 2; o++) {
        __m128i a0, a1, a2, a3, hn0, hn1, h0, h1, l0, l1, t;

        a0 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16));
        a1 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 4));
        a2 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 8));
        a3 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 12));
        a0 = _mm_sra_epi32(_mm_add_epi32(a0, rnd), cnt);
        a1 = _mm_sra_epi32(_mm_add_epi32(a1, rnd), cnt);
        a2 = _mm_sra_epi32(_mm_add_epi32(a2, rnd), cnt);
        a3 = _mm_sra_epi32(_mm_add_epi32(a3, rnd), cnt);
        hn0 = _mm_packs_epi32(a0, a1);
        hn1 = _mm_packs_epi32(a2, a3);
        h0 = _mm_srai_epi16(_mm_add_epi16(hn0, k128), 8);
        h1 = _mm_srai_epi16(_mm_add_epi16(hn1, k128), 8);
        l0 = _mm_sub_epi16(hn0, _mm_slli_epi16(h0, 8));
        l1 = _mm_sub_epi16(hn1, _mm_slli_epi16(h1, 8));
        l0 = _mm_adds_epi16(_mm_subs_epi16(l0, kclp), kclp); /* max(l, -127) */
        l1 = _mm_adds_epi16(_mm_subs_epi16(l1, kclp), kclp);
        t = _mm_packs_epi16(h0, h1);
        _mm_storeu_si128((__m128i *)(void *)(oh + o * 16), t);
        t = _mm_packs_epi16(l0, l1);
        _mm_storeu_si128((__m128i *)(void *)(ol + o * 16), t);
    }
}
#endif

#if defined(LZ_HAVE_P2_SSE2) && !defined(LZ_HAVE_P2_MMX)
#error "SSE2 pass 2 without the MMX one: gdn_p2_row_simd is gated on MMX"
#endif

#ifdef LZ_HAVE_P2_MMX
/* 0 scalar, 1 MMX, 2 SSE2. Simpler than lz_q8r_tier for
   one reason: this operator's tiers are pure integer SIMD, so they line
   up exactly with the kernel tier g_kernel already carries - unlike the
   q8 rounding, where a PIII has 64-bit integer SIMD but 128-bit float
   SSE and the two axes come apart.

   MMX is the floor of the whole target family (iron law 3's table has a
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
#endif
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
#if defined(__WATCOMC__)
    if (lz_cpu_has_sse()) return 2;   /* CPUID leaf 1 EDX 25, cached */
#else
    return 2;                    /* gcc build: SSE at compile time */
#endif
#endif
    return 1;
}
#endif

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
    if (!(cmax >= LZ_Q8_MIN_SCALE)) {
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

/* SHIFT, NOT DIVIDE. A divide would land amax exactly on 32512
   (hn = A*32512/amax) at one integer divide per element; MMX has no
   divide, so that alone keeps an MMX twin out of reach.

   Shifting instead maps amax somewhere into [16256, 32512] rather than
   onto 32512, which gives up to one bit of code range. PREDICTED to cost
   accuracy; MEASURED to improve it (4.27e-05 -> 2.98e-05)
   because the divide's own rounding was worth more than the lost bit.

   32512 rather than 32767 is what makes |hn| <= 32513 and therefore
   |hi| <= 127 after the split - the two saturating packs in the MMX twin
   depend on that bound and must never fire. */
static int p2_shift_of(int32_t amax) {
    int sh = 0;
    while ((amax >> sh) > 32512) sh++;
    return sh;
}

/* The float epilogue. s_new stays exact: sacc is already a power of two
   from the normalization, so multiplying by 2^sh and by L introduces no
   rounding of its own. Returns 0 when the group must be written as an
   exact zero - either the prologue flushed it (sacc == 0) or the new
   scale itself falls under the floor. */
static int p2_group_scale(float sacc, int sh, float *os) {
    float s_new;
    if (!(sacc > 0.0f)) { *os = 1.0f; return 0; }
    s_new = sacc * pow2f(sh) * LZ_GDN_LO_SCALE;
    if (!(s_new >= LZ_Q8_MIN_SCALE)) { *os = 1.0f; return 0; }
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
#endif
                              const float *ss_in, float *ss_out,
                              const float *delta, const float *k,
                              const float *gvec, float gt,
                              int kd, int vd);

static void gdn_p2_group_fixed(const int8_t *ph,
#if LZ_GDN_STATE_2PLANE
                               const int8_t *pl,
#endif
                               float c1, const int16_t *dq, float c2,
                               int8_t *oh,
#if LZ_GDN_STATE_2PLANE
                               int8_t *ol,
#endif
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
#endif
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
#endif
        }
    }
    if (!p2_group_scale(sacc, sh, os)) {
        for (j = 0; j < 32; j++) {
            oh[j] = 0;
#if LZ_GDN_STATE_2PLANE
            ol[j] = 0;
#endif
        }
    }
}

/* --- the SIMD twin, one whole ROW rather than one group ---------------

   Iron law 8's MMX cell for this operator. Bit-identical to the scalar
   above by construction, not by tolerance: every step is the same
   integer operation on the same integers, verified at 100k scale.

   WHY A ROW AND NOT A GROUP - it is entirely about rule 6 item 6. The
   emms may only be hoisted out of a loop whose body has no x87, and the
   per-group body has plenty: the coefficients, the normalization, the
   new scale. So the row is PHASE-SPLIT instead - all the float work of
   the row first, then all the MMX work, then one emms, then the float
   epilogue. One emms per row instead of one per 32 elements, which is
   the ratio iron law 6 costs out at 16% of the matmul compute budget on
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
#endif
                            const float *rs_in, float dec,
                            const int16_t *dq, const float *s_dg, float kv,
                            int8_t *oh_row,
#if LZ_GDN_STATE_2PLANE
                            int8_t *ol_row,
#endif
                            float *rs_out, int ng, int tier) {
    lz_p2_blk *blk = p2_blk();
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
#endif
    int gg, j;

    /* PHASE 1 - float only. */
    for (gg = 0; gg < ng; gg++) {
        float c1, c2;
        p2_group_coef(dec, rs_in[gg], kv, s_dg[gg], &c1, &c2);
        p2_group_norm(c1, c2, mul[gg], &sacc[gg]);
    }

    /* PHASE 2 - integer and MMX only, NO x87 anywhere in this loop.
       Changing that is what would make the single emms below wrong. */
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
#endif
#ifdef LZ_HAVE_P2_SSE2
        if (tier >= 3) lz_p2_mul32_sse2(ph_row + gg * 32, plg, dq + gg * 32, blk);
        else
#endif
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
#endif
#ifdef LZ_HAVE_P2_SSE2
            if (tier >= 3) lz_p2_split32_sse2(blk, oh_row + gg * 32, olg);
            else
#endif
#ifdef LZ_HAVE_P2_SSE
            if (tier >= 2) lz_p2_split32_sse(blk, oh_row + gg * 32, olg);
            else
#endif
            lz_p2_split32_mmx(blk, oh_row + gg * 32, olg);
        }
    }
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
#endif
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
#endif
#if defined(LZ_HAVE_P2_SSE_ASM)
    if (p2_tier() >= 2) return "sse-asm";
#elif defined(LZ_HAVE_P2_SSE)
    if (p2_tier() >= 2) return "sse-intrin";
#endif
#if defined(LZ_HAVE_P2_MMX_ASM)
    if (p2_tier()) return "mmx-asm";
#elif defined(LZ_HAVE_P2_MMX)
    if (p2_tier()) return "mmx-intrin";
#endif
    return "ref";
#else
    return "-";
#endif
}

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
#endif
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
#endif
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
#endif
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
#endif
        }
    return kq;
}

static void gdn_p2_fixed_rows(const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                              const int8_t *sq2_in, int8_t *sq2_out,
#endif
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
        if (!(amd * (1.0f / 32767.0f) >= LZ_Q8_MIN_SCALE)) {
            s_dg[gg] = 1.0f;
            for (vv = gg * 32; vv < gg * 32 + 32; vv++) dq[vv] = 0;
            continue;
        }
        s_dg[gg] = amd * (1.0f / 32767.0f);
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
#endif
#ifdef LZ_HAVE_P2_MMX
        if (p2_tier()) {
            gdn_p2_row_simd(row_in,
#if LZ_GDN_STATE_2PLANE
                            row_lo_in,
#endif
                            rs_in, dec, dq, s_dg, k[kk], row_out,
#if LZ_GDN_STATE_2PLANE
                            row_lo_out,
#endif
                            rs_out, ng, p2_tier());
            continue;
        }
#endif
        for (gg = 0; gg < ng; gg++) {
            float c1, c2;
            p2_group_coef(dec, rs_in[gg], k[kk], s_dg[gg], &c1, &c2);
            gdn_p2_group_fixed(row_in + gg * 32,
#if LZ_GDN_STATE_2PLANE
                               row_lo_in + gg * 32,
#endif
                               c1, dq + gg * 32, c2,
                               row_out + gg * 32,
#if LZ_GDN_STATE_2PLANE
                               row_lo_out + gg * 32,
#endif
                               rs_out + gg);
        }
    }
}

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
#endif
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
#endif

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
#endif
                                  ss_in, q, k, gvec, kd, vd, u, w);
    } else
#endif
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
#endif
            for (j = 0; j < 32; j++) {
#if LZ_GDN_STATE_2PLANE
                float e = (float)p[j] + (float)pl[j] * (1.0f / LZ_GDN_LO_SCALE);
#else
                float e = (float)p[j];
#endif
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
#endif
                          ss_in, ss_out, u, k, gvec, gt, kd, vd);
        return;
    }
#endif

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
#endif
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
#endif
        }
#if LZ_GDN_STATE_2PLANE
        lz_gdn_quantize_2p(buf, vd, 32, row_out, row_lo_out, rs_out);
#else
        lz_quantize_q8(buf, vd, 32, row_out, rs_out);
#endif
    }
}

void lz_gdn_step(float *out, const int8_t *sq_in, int8_t *sq_out,
#if LZ_GDN_STATE_2PLANE
                 const int8_t *sq2_in, int8_t *sq2_out,
#endif
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 float gt, float beta, int kd, int vd) {
    gdn_step_impl(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                  sq2_in, sq2_out,
#endif
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
#endif
                 const float *ss_in, float *ss_out,
                 const float *q, const float *k, const float *v,
                 const float *gvec, float beta, int kd, int vd) {
    gdn_step_impl(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                  sq2_in, sq2_out,
#endif
                  ss_in, ss_out, q, k, v, gvec, 1.0f, beta, kd, vd);
}

void lz_causal_conv1d_step(float *o, const float *x,
                               const float *state_in, float *state_out,
                               const float *w, int n_ch, int k) {
    int hist = k - 1;
    int c, j;
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
            if (key[i] <= -1e30f) continue;      /* already selected */
            if (best < 0 || key[i] > bk) { best = i; bk = key[i]; }
        }
        if (best < 0) break;
        idx_out[t] = best;
        w_out[t] = tempered ? tscore[best] : score[best];
        wsum += w_out[t];
        key[best] = -1e30f;                      /* remove from further rounds */
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
