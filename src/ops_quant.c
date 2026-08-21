/* ops_quant.c - Quantization primitives and fixed-point transcendental
   helpers. A separate TU from ops.c to keep its #if guard density low.

   Every function here is extern, declared in ops_quant.h. */
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ops.h"
#include "err.h"
#include "cpucheck.h"
#include "mmx_compat.h"
#include "ops_mmx.h"
#include "ops_sse2.h"
#include "ops_kernel_shared.h" /* lz_i32f, LZ_ROW_N, LZ_DEFINE_PICK, g_kernel */
#include "ops_quant.h"

#include "ops_kernel_amax.h"  /* dispatch table for q8_amax */

/* GDN fixed tier default - same as ops.c's block. */
#ifndef LZ_GDN_FIXED
#define LZ_GDN_FIXED 1
#endif /* LZ_GDN_FIXED */

#ifndef LZ_NORM_MAX_N
#define LZ_NORM_MAX_N 4096
#endif /* LZ_NORM_MAX_N */

/* ---- x87 precision control ------------------------------------------- */

#if defined(__i386__) && !defined(__SSE_MATH__) && !defined(__SSE2_MATH__)
#define LZ_X87_FLOAT_CW 1
/* fnstcw/fldcw directly, not a libc wrapper: this project's other 32-bit
   x86 targets (DOS, Win9x) have no libc guaranteed to carry one, and
   the instructions need no library at all. PC is control-word bits
   9:8 - 00 selects 24-bit (single) precision, 11 (the power-on default)
   selects 64-bit (extended). */
unsigned short lz_x87_cw_get(void) {
    unsigned short cw;
    __asm__ __volatile__("fnstcw %0" : "=m"(cw));
    return cw;
}
void lz_x87_cw_set(unsigned short cw) {
    __asm__ __volatile__("fldcw %0" : : "m"(cw));
}
#endif /* __i386__ && !__SSE_MATH__ && !__SSE2_MATH__ */

/* ---- pow2f ----------------------------------------------------------- */

/* 2^e by integer manipulation of the exponent field. Deliberately not
   ldexp/ldexpf: those are libm doubles, and every
   transcendental inside the engine so the two builds cannot disagree
   through somebody else's rounding - while the rules bar
   double from the PC=24 region outright, which is where every caller of
   this runs. It does no rounding at all, it writes an exponent, so the
   two compilers agree by construction rather than by measurement.

   One definition for the whole file: the fixed conv, epilogue and GDN
   paths all need it, and three ways to build a power of two is three
   chances for one of them to differ. */
float pow2f(int e) {
    union { float f; uint32_t u; } b;
    if (e < -126) return 0.0f;
    if (e >  127) e = 127;
    b.u = (uint32_t)(e + 127) << 23;
    return b.f;
}

/* ---- q8_round -------------------------------------------------------- */

/* Round-to-nearest (IEEE default round-half-to-even) via a magic-number
   add.

   Adding 1.5x2^23 forces the mantissa to align on integer bits; the
   integer part lands in the low 23 bits and is read straight out of the
   union. Valid for |qv| < 2^22 - quant inputs are scaled within +/-127,
   lz_exp's exponent within +/-127, all far inside.

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
   coexisting rules.

   ARM turns the trick upside down. There is no FPU and no control word
   to rewrite, so the magic add is a full __aeabi_fadd - 48 instructions
   measured for what is a shift and a rounding decision on the bit
   pattern. --kernel arm-asm takes that path (src/ops_arm_prim.h);
   outside |qv| < 2^22, the domain stated above, the primitive declines
   and the magic add below runs, so the two tiers agree everywhere and
   not just where the contract holds. */
int q8_round(float qv) {
    union { float f; uint32_t u; } t;
#if defined(LZ_ARM_PRIM_ASM)
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        int32_t r;
        t.f = qv;
        if (lz_q8_round_pow2_arm_asm(t.u, 0, &r)) return (int)r;
    }
#endif /* LZ_ARM_PRIM_ASM */
    t.f = qv + 12582912.0f;                 /* 1.5 * 2^23 */
    return (int)(t.u & 0x007FFFFFu) - 0x00400000;
}

/* ---- rsqrt (float body + fixed) -------------------------------------- */

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
   transcendental functions, the gcc and Watcom builds are bit-identical
   for THIS function - measured, 0/2592 across a swept exponent x
   mantissa-prefix grid. It holds here because every constant
   this function touches (0.5f, 1.5f, the hex integer 0x5f375a86u) is
   exactly representable in float32, so gcc's x87 excess-precision
   constant folding (see lz_exp's LZ_EXP_LOG2E32 below) has nothing to
   round differently in the first place. It does NOT hold for lz_exp:
   1.44269504f is not exactly representable, gcc keeps the extra bits,
   and the two builds' multiplies disagree - see lz_exp's own comment
   and tests/test_excess_precision.py.

   `static const float` (lz_exp's fix) is a strong default, not a
   guarantee at every call site: the same named constant has been
   measured loading via flds at one use and fldt at another in the
   same function, under a nearby cast's register pressure. The source
   idiom is a nudge; tests/test_excess_precision.py is the actual
   enforcement - do not trust the source shape alone. */
float lz_rsqrt_float_body(float x) {
    union { float f; uint32_t u; } t;
    float h = 0.5f * x, y;
    t.f = x;
    t.u = 0x5f375a86u - (t.u >> 1);
    y = t.f;
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y;
}

/* Fixed 1/sqrt(x): Q23 seed table + 2 integer Newton iterations.

   m = x's normalized mantissa in [1,2) is read straight from the float
   bits, so the core never enters the float domain; the exponent splits
   off as a power-of-two scale on the way out. Q23 is the mantissa's own
   scale, so the assembly is one fild and one multiply by an exact power
   of two - the result's mantissa is already in place.

   A 256-entry seed table holds Newton's error at ~0.1%, so two iterations
   land ~1e-12, under Q23's own 2^-23 quantum; the result is accordingly
   a correctly-rounded 1/sqrt, which differs from the float body's
   (~4.7e-6, its two float Newtons' residue) by design - this tier is
   documented value-changing.

   Every arithmetic step is integer and exact, so gcc and Watcom agree by
   construction. Subnormal/inf/nan x falls back to the float body. */
#define LZ_RSQRT_TAB_N 256
static int32_t g_rsqrttab[LZ_RSQRT_TAB_N];
static int g_rsqrttab_ready;

static void rsqrt_tab_build(void) {
    int i;
    for (i = 0; i < LZ_RSQRT_TAB_N; i++) {
        float m = 1.0f + ((float)i + 0.5f) / (float)LZ_RSQRT_TAB_N;
        g_rsqrttab[i] =
            (int32_t)(lz_rsqrt_float_body(m) * 8388608.0f + 0.5f); /* Q23 */
    }
    g_rsqrttab_ready = 1;
}

float lz_rsqrt_fixed(float x) {
    static const float LZ_RSQRT_SQRT2 = 0.7071067811865476f;
    union { float f; uint32_t u; } b;
    int32_t m23, y23;
    int64_t p, t, fac;
    int e, idx, odd;
    if (!g_rsqrttab_ready) rsqrt_tab_build();
    b.f = x;
    e = (int)((b.u >> 23) & 0xFFu) - 127;
    if (e < -126 || e > 126) return lz_rsqrt_float_body(x);
    m23 = (int32_t)((b.u & 0x007FFFFFu) | 0x00800000u);
    idx = (int)((b.u >> 15) & 0xFFu);
    y23 = g_rsqrttab[idx];
    /* Two Newton steps y = y*(3 - m*y^2)/2, all Q23. The shifts round
       half-up - truncation would bias a whole-vector scale. */
    p = (((int64_t)y23 * y23) + (1 << 22)) >> 23;
    t = (((int64_t)m23 * p) + (1 << 22)) >> 23;
    fac = (3 * (int64_t)(1 << 23) - t + 1) >> 1;
    y23 = (int32_t)((((int64_t)y23 * fac) + (1 << 22)) >> 23);
    p = (((int64_t)y23 * y23) + (1 << 22)) >> 23;
    t = (((int64_t)m23 * p) + (1 << 22)) >> 23;
    fac = (3 * (int64_t)(1 << 23) - t + 1) >> 1;
    y23 = (int32_t)((((int64_t)y23 * fac) + (1 << 22)) >> 23);
    odd = e & 1;
    /* fild + one/two pow2 multiplies: the odd-e case adds the SQRT2. */
    LZ_FCX(LZ_FC_RSQRT, odd ? 3 : 2, 0, 0, 1, 0);
    if (odd)
        return (float)y23 * pow2f(-(e >> 1) - 23) * LZ_RSQRT_SQRT2;
    return (float)y23 * pow2f(-(e >> 1) - 23);
}

/* ---- exp (float body + fixed) ---------------------------------------- */

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
   require storing the table as double, and the size is back to KBs.

   File scope: the fixed tier quantizes the same entries to Q20
   instead of carrying a second copy that could drift. */
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

/* `static const float`, not inline literals: under -std=c99 on x87,
   gcc folds an inline float-literal expression and keeps it at
   extended (~64-bit significand) precision in .rodata, loaded via
   fldt - carrying ~40 more significand bits than the `f` suffix
   promised. A `static const float` object's storage type forces
   gcc to round the initializer to true float32 (flds/fmuls, the
   same instruction shape Watcom already uses), because the object
   itself must be representable in 4 bytes. Measured: eliminates
   fldt for both constants below; tests/test_excess_precision.py
   is the gate. File scope for the same reason as LZ_EXP_TAB32: the
   fixed tier reads the same log2e scale and the same floor bound.

   -87.3f is not exactly representable in float32, so it needs the
   same treatment as the constants above; 88.0f is an exact integer
   and needs none (gcc emits a direct memory compare for it either
   way). */
static const float LZ_EXP_LOG2E32 = 1.44269504f * 32.0f;  /* log2(e)*32 */
static const float LZ_EXP_LO = -87.3f;

/* Schraudolph-style: add x*log2e*2^23 directly into the float's
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
float lz_exp_float_body(float x) {
    static const float LZ_EXP_C2 = 0.242844437f;
    static const float LZ_EXP_C1 = 0.207425515f;
    static const float LZ_EXP_C0 = 0.549730133f;
    union { float f; int32_t i; } u, p2n;
    float y32, r, w, c;
    int32_t t, n, idx;

    /* 1 scale, 2 for the magic add, 1 convert + 1 subtract for the
       residue, 2 for w, 4 for the quadratic, 2 to reassemble.
       mul: scale(1) + w(1) + quadratic(2) + reassemble(2) = 6.
       add: magic add(2) + residue's subtract(1) + w(1) + quadratic(2) = 6.
       cvt: residue's (float)t(1). */
    LZ_FCX(LZ_FC_EXP, 6, 6, 0, 1, 0);

    /* Overflow/underflow guard: beyond this range the constructed bit
       pattern is meaningless. */
    if (x > 88.0f)     return FLT_MAX;
    if (x < LZ_EXP_LO) return 0.0f;

    /* ALL FLOAT, NO DOUBLE. A double magic (1.5x2^52) cannot be used:
       it would stuff x*log2e*2^23 in one shot, extracting exponent,
       table index and residue together - convenient but requiring a
       53-bit mantissa. On the Win98 side, the x87 precision control
       must be set to 24 bits for bitwise parity with SSE (see
       lz_fpu_float_begin); under that, double is also compressed to 24
       bits and the magic dies - measured: conv1d's silu collapsed
       entirely (rms 0.1091 -> 0.1257).

       The two-level float magic avoids that: round on a 1/32-step scale
       first, getting exponent and table index at once, residue by
       subtraction. |y*32| <= 4064, a thousand-fold margin below the
       float magic's 2^22 limit. The two extra int->float conversions
       are fild, not float->int; they do not trigger control-word
       rewrites. */
    y32 = x * LZ_EXP_LOG2E32;
    u.f = y32 - 0.5f + 12582912.0f;           /* 1.5*2^23; the -0.5 turns round-half-away into floor */
    t = (u.i & 0x007FFFFF) - 0x00400000;      /* floor(y*32) */
    n = t >> 5;                               /* arithmetic shift = floor(t/32) */
    idx = t & 31;                             /* always 0..31 in two's complement */
    r = y32 - (float)t;                       /* [0,1) */
    w = 1.0f + r * 0.03125f;                  /* 1 + residual, in [1, 1+1/32) */
    p2n.i = (n + 127) << 23;
    /* 2^r on [1, 1+1/32): 2nd-order fit. The interval is only 1/32
       wide; 2nd order reaches 5e-6 - 57x better than a
       "4th-order over the whole [1,2)" fit at 2.85e-4, and 5% faster
       (Watcom measured 9.63 vs 10.15 ns). */
    c = (LZ_EXP_C2 * w + LZ_EXP_C1) * w + LZ_EXP_C0;
    return p2n.f * (LZ_EXP_TAB32[idx] * c);
}

/* Fixed exp, in Q20: the same floor and table index as the float body
   (the n/idx split is bit-identical - same y32, same magic), with the
   correction 2^(r/32) moved to the integer domain as a 2nd-order Taylor
   on s = r/32 in [0,1/32): 1 + ln2*s + (ln2)^2/2*s^2, bounded by
   (ln2)^3/6 * (1/32)^3 = 2.8e-7. The table is the float entries
   quantized to Q20 (2^-21), so the total drift from the float path is
   ~1e-6 relative - value-changing by design, and this tier is a knob.

   Underflow note: for x below -110*log2e the exponent n is so negative
   that pow2f clamps to zero where the float body emits a subnormal.
   The gap is below any quantization floor; documented rather than
   handled. */
#define LZ_EXP_Q 20
#define LZ_EXP_QSCALE (1 << LZ_EXP_Q)
static int32_t g_exptab[32];
static int g_exptab_ready;
static const int32_t LZ_EXP2_LN2 = 726817;   /* round(ln 2 * 2^20) */
static const int32_t LZ_EXP2_LN2SQ = 251906; /* round((ln 2)^2 / 2 * 2^20) */

static void exp_tab_build(void) {
    int i;
    for (i = 0; i < 32; i++)
        g_exptab[i] = (int32_t)(LZ_EXP_TAB32[i] * (float)LZ_EXP_QSCALE + 0.5f);
    g_exptab_ready = 1;
}

#if defined(LZ_ARM_PRIM_ASM)
/* Bit pattern of a compile-time float constant. gcc folds this away at
   -O2, so the guard windows below cost nothing to derive and cannot
   drift from the constants they are derived FROM - which hand-written
   hex would. */
static __inline__ uint32_t lz_f2u(float f) {
    union { float f; uint32_t u; } b;
    b.f = f;
    return b.u;
}
#endif /* LZ_ARM_PRIM_ASM */

/* The hand-written ARM tier (--kernel arm-asm) lives inline in the body
   below rather than in a twin function, because only four steps of it
   change and a twin would be a second copy of the Taylor core.
   docs/arm-asm-audit.md C2 named the content and section 7 measures it:
   the two guards become bit-pattern range tests, the floor and the
   residue become integer work on the mantissa (killing an fsub, an
   fadd, an i2f, a second fsub, an fmul and q8_round's own magic add),
   and the exit's power-of-two multiply becomes an exponent add.

   WHAT STAYS A LIBRARY CALL, and why. `x * LZ_EXP_LOG2E32` is a genuine
   float multiply by a constant that is not a power of two - inlining it
   means writing a soft-float multiplier and proving its rounding
   identical, which is a libgcc rewrite, not an ISA cell. The floor and
   residue steps DECLINE (and the C below runs) wherever their integer
   identity is not provable; see the two primitives' comments. */
float lz_exp_fixed(float x) {
    union { float f; int32_t i; } u;
    float y32, r;
    int32_t t, n, idx, rq, s, cq, prod;
#if defined(LZ_ARM_PRIM_ASM)
    const int asm_tier = (g_kernel == LZ_KERNEL_ARM_ASM);
    int32_t ms = 0, sh = 0;
    int have_floor = 0;
#endif /* LZ_ARM_PRIM_ASM */
    if (!g_exptab_ready) exp_tab_build();
    /* Same guards as the float body, so the two paths agree on where
       the approximation contract ends. The windows are closed at
       +-inf's bit pattern, which is what leaves NaN outside both and
       makes the integer test answer exactly what the float one does. */
#if defined(LZ_ARM_PRIM_ASM)
    if (asm_tier) {
        uint32_t ux = lz_f2u(x);
        if (lz_uwin_arm_asm(ux, lz_f2u(88.0f) + 1u,
                            0x7F800000u - lz_f2u(88.0f) - 1u))
            return FLT_MAX;
        if (lz_uwin_arm_asm(ux, lz_f2u(LZ_EXP_LO) + 1u,
                            0xFF800000u - lz_f2u(LZ_EXP_LO) - 1u))
            return 0.0f;
    } else
#endif /* LZ_ARM_PRIM_ASM */
    {
        if (x > 88.0f)     return FLT_MAX;
        if (x < LZ_EXP_LO) return 0.0f;
    }
    /* 1 scale, 1 magic add, 1 subtract + 1 scale + 1 magic add for the
       residual, fild + 1 pow2 multiply out: 3 mul, 3 add, 1 cvt. */
    LZ_FCX(LZ_FC_EXP, 3, 3, 0, 1, 0);
    y32 = x * LZ_EXP_LOG2E32;
#if defined(LZ_ARM_PRIM_ASM)
    if (asm_tier) {
        union { float f; uint32_t u; } b;
        b.f = y32;
        if (lz_exp_floor_arm_asm(b.u, &t, &ms, &sh)) {
            n = t >> 5;
            idx = t & 31;
            rq = lz_exp_resid_arm_asm(ms, t, sh);
            have_floor = 1;
        }
    }
    if (!have_floor)
#endif /* LZ_ARM_PRIM_ASM */
    {
        u.f = y32 - 0.5f + 12582912.0f;       /* same floor as the float body */
        t = (u.i & 0x007FFFFF) - 0x00400000;
        n = t >> 5;
        idx = t & 31;
        r = y32 - (float)t;                   /* [0,1) */
        rq = q8_round(r * (float)LZ_EXP_QSCALE);  /* residual, Q20 */
    }
    /* s = r/32 = rq/2^25, so the Q20 Taylor terms are (ln2_q*s)>>20 and
       (ln2sq_q*s*s)>>40; both shifts round half-up. */
    s = rq >> 5;
    cq = LZ_EXP_QSCALE
       + (int32_t)((((int64_t)LZ_EXP2_LN2 * s)
                    + (1 << (LZ_EXP_Q - 1))) >> LZ_EXP_Q)
       + (int32_t)((((int64_t)LZ_EXP2_LN2SQ * s * s)
                    + (1LL << (2 * LZ_EXP_Q - 1))) >> (2 * LZ_EXP_Q));
    prod = (int32_t)((((int64_t)g_exptab[idx] * cq)
                      + (1 << (LZ_EXP_Q - 1))) >> LZ_EXP_Q);
#if defined(LZ_ARM_PRIM_ASM)
    if (asm_tier) {
        int32_t k = n - LZ_EXP_Q;
        if (k >= -126 && k <= 127) {
            union { float f; uint32_t u; } pb;
            pb.f = lz_i32f_arm_asm(prod);     /* == (float)prod, exact here */
            if (lz_scale2_arm_asm(pb.u, k, &pb.u)) return pb.f;
        }
    }
#endif /* LZ_ARM_PRIM_ASM */
    return (float)prod * pow2f(n - LZ_EXP_Q);
}

/* ---- sigmoid Q15 table ----------------------------------------------- */

#define LZ_SIG_N    384
#define LZ_SIG_LO   (-12.0f)
#define LZ_SIG_STEP 16.0f          /* intervals per unit x */

static int32_t g_sigtab[LZ_SIG_N + 1];
static int     g_sigtab_ready;

static void sig_build(void) {
    int i;
    for (i = 0; i <= LZ_SIG_N; i++) {
        float x = LZ_SIG_LO + (float)i / LZ_SIG_STEP;
        /* The float body, not lz_exp: the sigmoid table must not move
           with the scalar tier's setting. */
        float z = lz_exp_float_body(x >= 0.0f ? -x : x);
        float s = (x >= 0.0f) ? 1.0f / (1.0f + z) : z / (1.0f + z);
        g_sigtab[i] = (int32_t)(s * 32768.0f + 0.5f);
    }
    g_sigtab_ready = 1;
}

/* Q15 sigmoid. Outside the table sigmoid is within 3.4e-4 of its limit,
   which is under the noise floor, so it clamps rather than widening the
   table for values the model does not reach. */
int32_t sigmoid_q15(float x) {
    float t;
    int idx;
    int32_t frac, a, b;
    if (!g_sigtab_ready) sig_build();
#if defined(LZ_ARM_PRIM_ASM)
    /* After sig_build, never before: the primitive reads the table. */
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        union { float f; uint32_t u; } bx;
        int32_t r;
        bx.f = x;
        if (lz_sig_q15_arm_asm(bx.u, g_sigtab, &r)) return r;
    }
#endif /* LZ_ARM_PRIM_ASM */
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

/* Table-only sigmoid_q15: takes the pre-scaled coordinate t = (x -
   LZ_SIG_LO) * LZ_SIG_STEP directly instead of computing it from x, for
   callers that fold LZ_SIG_STEP into their own per-call scale (see
   lz_sig_q15_fold/lz_sig_q15_t_offset). Clamp thresholds are t's own
   domain, derived from sigmoid_q15's (t is monotonic increasing in x):
   t <= 0 is x <= LZ_SIG_LO, t >= LZ_SIG_N is x >= -LZ_SIG_LO. Both
   clamps run before any cast, unlike a naive port of the x-domain
   checks, since a caller's folded t can be far outside int32 range for
   an out-of-scale accumulator. */
int32_t sigmoid_q15_t(float t) {
    int idx;
    int32_t frac, a, b;
    if (!g_sigtab_ready) sig_build();
#if defined(LZ_ARM_PRIM_ASM)
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        union { float f; uint32_t u; } bt;
        int32_t r;
        bt.f = t;
        if (lz_sig_q15_t_arm_asm(bt.u, g_sigtab, &r)) return r;
    }
#endif /* LZ_ARM_PRIM_ASM */
    if (t <= 0.0f) return 0;
    if (t >= (float)LZ_SIG_N) return g_sigtab[LZ_SIG_N];
    idx = (int)t;
    frac = (int32_t)((t - (float)idx) * 32768.0f);
    a = g_sigtab[idx];
    b = g_sigtab[idx + 1];
    return a + (((b - a) * frac) >> 15);
}

/* The additive half of sigmoid_q15_t's folded entry: t = acc*k1 + this,
   where k1 = oscale*LZ_SIG_STEP (lz_sig_q15_fold) reproduces (x -
   LZ_SIG_LO)*LZ_SIG_STEP for x = acc*oscale bit-for-bit, since both
   LZ_SIG_STEP and the offset below are exact powers of two applied to
   an already-computed value - multiplying by an exact power of two only
   shifts the exponent, so it introduces no rounding of its own and
   commutes exactly with the surrounding add/multiply chain. Verified
   over the accumulator's realistic range (.prof/verify_conv_fold3.c),
   0 mismatches against the unfolded x = acc*oscale path. A getter, not
   a header #define, so LZ_SIG_LO/STEP stay private to this file. */
float lz_sig_q15_t_offset(void) { return -LZ_SIG_LO * LZ_SIG_STEP; }

/* Per-channel factors for the fold above: k1 feeds sigmoid_q15_t's
   entry, oscale2 folds the exit's 1/32768 descale into the same
   dequant this channel already applies, so the caller never has to
   materialize x = acc*oscale as an intermediate - see
   lz_causal_conv1d_step_fixed for the shape this exists for. */
void lz_sig_q15_fold(float oscale, float *k1, float *oscale2) {
    *k1 = oscale * LZ_SIG_STEP;
    *oscale2 = oscale * (1.0f / 32768.0f);
}

/* ---- Q8 quantization helpers ----------------------------------------- */

/* The amax scan, dispatched. Bit-identical across every tier by
   construction rather than by tolerance - max is exact and associative,
   so lane order cannot change the answer. That is what makes this one
   of the few reductions here that may be vectorized at all.

   The tail is scalar: a group is 32 elements in every shipping recipe,
   so the tail is empty in practice, and a tail loop that never runs is
   cheaper than a kernel that has to handle every length. */
uint32_t q8_amax(const float *grp, int gs) {
    union { float f; uint32_t u; } bit;
    uint32_t am = 0;
    int k = 0;
    lz_amaxfn f;
    int lanes;

    /* One table, one pick, no #if at the call site - so `--kernel` moves
       this operator with all the others, and a build with no SIMD gets
       NULL and falls through to the scalar loop below. */
    if (!g_kernel) lz_kernel_select(LZ_KERNEL_AUTO);
    f = lz_amax_pick(LZ_AMAX_TAB);
    lanes = lz_amax_lanes(f);
    if (f && gs >= lanes) {
        am = f(grp, gs);
        k = gs & ~(lanes - 1);
        if (lz_amax_is_mmx(f)) {
            if (lz_cpu_has_mmx()) _mm_empty();
        }
    }
    for (; k < gs; k++) {
        uint32_t a;
        bit.f = grp[k];
        a = bit.u & 0x7FFFFFFFu;
        if (a > am) am = a;
    }
    return am;
}

/* Shared constants. `static const float` forces the round once, here,
   avoiding gcc's x87 excess-precision folding at each use site. */
const float LZ_Q8_MIN_SCALE_F = LZ_Q8_MIN_SCALE;
const float LZ_Q8_INV127 = 1.0f / 127.0f;
const float LZ_Q15_INV = 1.0f / 32767.0f;

int q8_group_scale(const float *grp, int gs, float *sg, float *inv) {
    union { float f; uint32_t u; } bit;
    uint32_t am = q8_amax(grp, gs);
    float amax;
    int fast;
    bit.u = am;
    amax = bit.f;
    if (amax > 0.0f) {
        *sg = amax * LZ_Q8_INV127;
        *inv = 127.0f / amax;
        fast = (*inv <= FLT_MAX);
        if (*sg < LZ_Q8_MIN_SCALE_F) {
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

/* ---- norm_ss_fixed (integer sum-of-squares) -------------------------- */

#if LZ_GDN_FIXED
int expof(float x) {
    union { float f; uint32_t u; } b;
    b.f = x;
    return (int)((b.u >> 23) & 0xFFu) - 127;
}

/* Sum of squares for the norms fixed tier, with the accumulation done
   as a per-element multiply-add in the INTEGER domain rather than
   float. Getting each element there still costs two float operations -
   the scale and q8_round's rounding add - which is exactly what the
   float loop this replaces spends on its own multiply-add, so this
   matches that cost rather than beating it. What it buys is an integer
   entry point: fused onto an epilogue that already emits integers, the
   two float operations disappear into work already being done there.

   The scale is a power of two, so it is exact in binary, adds no
   rounding of its own, and cannot differ between x87 and SSE.

   qout/eout expose the Q15 image and its exponent instead of
   discarding them: a caller building its own int output chain from x
   (lz_rmsnorm_out_fixed_int) can reuse this scan rather than quantizing
   x a second time. No static buffer of its own - a caller that does
   not need the image passes NULL and nothing is written beyond the
   accumulator, so a plain sum-of-squares call costs exactly what it did
   before this parameter existed.

   Declines by returning -1 rather than truncating, so an oversized n
   runs the float path instead of being silently wrong. */
float norm_ss_fixed(const float *x, int n, short *qout, int *eout) {
    union { float f; uint32_t u; } bit;
    long long acc = 0;
    float sc;
    int i, e;

    if (n > LZ_NORM_MAX_N || n <= 0) return -1.0f;
    bit.u = q8_amax(x, n);
    if (bit.u == 0) {                          /* all zero, exactly */
        if (qout) for (i = 0; i < n; i++) qout[i] = 0;
        if (eout) *eout = 0;
        return 0.0f;
    }
    e = 14 - expof(bit.f);
    sc = pow2f(e);
    for (i = 0; i < n; i++) {
        int v = q8_round(x[i] * sc);           /* one mul, one magic add */
        if (v >  32767) v =  32767;
        if (v < -32767) v = -32767;
        if (qout) qout[i] = (short)v;
        acc += (long long)(v * v);
    }
    if (eout) *eout = e;
    return (float)acc * pow2f(-2 * e);
}
#else
/* No fixed GDN in this build, so no expof to derive the scale from.
   Declining sends every caller down the float path, which is the
   correct behaviour for a build with no fixed arithmetic at all. */
float norm_ss_fixed(const float *x, int n, short *qout, int *eout) {
    (void)x; (void)n; (void)qout; (void)eout;
    return -1.0f;
}
#endif /* LZ_GDN_FIXED */
