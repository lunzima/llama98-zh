/* ops_quant.c - Quantization primitives and fixed-point transcendental
   helpers. A separate TU from ops.c to keep its #if guard density low.

   Every function here is extern, declared in ops_quant.h. */
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ops.h"
#include "ops_arm.h"     /* LZ_ARM_ASM_EXTERN, lz_norm_ss_fixed_arm_asm */
#include "err.h"
#include "cpucheck.h"
#include "mmx_compat.h"
#include "ops_mmx.h"
#include "ops_sse2.h"
#include "ops_kernel_shared.h" /* lz_i32f, LZ_ROW_N, LZ_DEFINE_PICK, g_kernel */
#include "ops_quant.h"

#include "ops_kernel_amax.h"  /* dispatch table for q8_amax */

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

/* q8_round now lives in ops_quant.h - it is called 234,713 times per
   token (counted, .prof/x87test) and the call plus the tier dispatch
   cost more than its body does. See the header for the measurement. */

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
   positive fraction), hence subtract 0.5 before the magic number.

   Scaling x by log2e*2^23 in ONE step would push the integer past the
   float magic's 2^22 range; the body scales by log2e*32 instead and
   recovers the exponent, the table index and the residue from that -
   see the two-level explanation there. Every bit operation goes
   through a `union { float f; int32_t i; }`, so the two members have
   the same width and this function does not depend on byte order. */
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

/* ---- folded-scale integer coordinates -------------------------------
 *
 * sigmoid_q15_i takes an integer whose scale is a POWER OF TWO (x ==
 * v * 2^-e). The entries here take one whose scale is not: x ==
 * v * m * 2^-s, with (m, s) a constant the caller folds once. That is
 * the shape the KDA decay gate needs, and the reason it still reached
 * both of its transcendentals through float coordinates after milestone
 * 5 - neither of its constants is a power of two:
 *
 *   pre  = gate_i16 + dt_bias_i16                           integer
 *   s15  = sigmoid(decay_base[h] * pre * 2^-LZ_KGATE_ES)    per head
 *   gt   = exp(s15 * lower_bound / 32768)                   per model
 *
 * decay_base[h] = exp(A_log[h]); measured on both checkpoints it runs
 * 0.00137..10.585 while lower_bound is a single -5.0 (.prof/m5range.c
 * reports both). So ONE SHARED SCALE DOES NOT SERVE BOTH: the exp's
 * fold is per model and the sigmoid's is per head. Per head rather than
 * per layer because it costs nothing to be - (m, s) is hoisted out of
 * the element loop either way, and a shared scale would quantize the
 * smallest head's constant against the largest head's exponent.
 *
 * WHY (m, s) AND NOT A PRE-MULTIPLIED COORDINATE, which is what
 * sigmoid_q15_t's float shape would suggest: the caller would need the
 * same 64-bit multiply, would add an intermediate rounding between it
 * and the table, and would put a signed shift in forward.c. Here the
 * product stays inside one exhaustively verified function and the table
 * index is derived from it with no rounding before it at all.
 *
 * WHY m CARRIES ONLY 24 BITS. The constants being folded ARE float32s,
 * so 24 bits is all the information in them; lz_fold_f32 splits one
 * exactly and the fold adds no error of its own. What it removes is the
 * pair of float32 roundings the old chain paid PER ELEMENT (`(float)s15
 * * lbq`, then `x * LZ_EXP_LOG2E32` inside lz_exp), so the integer
 * coordinate is the more accurate of the two and not the looser one.
 */

/* floor(p / 2^sh) for a signed 64-bit p, computed on a biased unsigned
   so that no negative value is ever right-shifted - that is only
   implementation-defined in C and none of this may depend on it.
   (uint64_t)p + 2^63 is exactly p + 2^63 as a value in [0, 2^64) for
   every p including INT64_MIN, and 2^63 is divisible by 2^sh, so
   (up >> sh) == floor(p/2^sh) + 2^(63-sh) and subtracting that constant
   back leaves the floor. Total: sh <= 0 is the identity, sh >= 64 is
   the sign, since |p| * 2^-64 < 1.

   This is the 64-bit form of the argument sig_t15_wide used to carry
   inline; that function calls this one now, so the file has ONE floor
   shift rather than two. It is not a second epi_shr_half either - that
   primitive rounds half-away-from-zero, and a rounded quotient has no
   remainder to pair with, which is exactly what a table coordinate is
   made of. */
static int64_t fold_floor_shr(int64_t p, int sh) {
    uint64_t up;
    if (sh <= 0) return p;
    if (sh >= 64) return (p < 0) ? -1 : 0;
    up = (uint64_t)p + ((uint64_t)1 << 63);
    return (int64_t)(up >> sh) - (int64_t)((uint64_t)1 << (63 - sh));
}

/* Split a * 2^p into m * 2^-s, EXACTLY.

   A finite normal float32 IS mant * 2^(e-150) with mant in [2^23, 2^24)
   and e the stored exponent field, so the split is a field read and an
   integer subtract: no rounding, no libm, and no double - which the
   PC=24 region would compress to 24 bits anyway.

   Returns 1 when the split is exact. 0 IS A REFUSAL, not a bound: on 0
   the entries below must not be given the pair, and m/s carry a
   saturated stand-in only so that a caller which ignores the return
   gets a defined value instead of an out-of-range shift. Refused are
   inf, NaN, subnormals, and any value whose exponent leaves the window
   [2^23 * 2^-LZ_FOLD_SMAX, 2^31). Zero splits exactly, as m = 0, s = 0.

   s is capped at LZ_FOLD_SMAX = 62 because that is the widest shift
   past which fold_floor_shr can only return a sign, and because both
   entries state their domain in terms of it. Negative exponents below
   -7 are refused rather than represented: |mant| < 2^24, so widening it
   by more than 7 places leaves int32. */
#define LZ_FOLD_SMAX 62

int lz_fold_f32(float a, int p, int32_t *m, int *s) {
    union { float f; uint32_t u; } b;
    int e, sc;
    int32_t mant;
    b.f = a;
    e = (int)((b.u >> 23) & 0xFFu);
    if (e == 0) {                          /* zero or subnormal */
        *m = 0;
        *s = 0;
        return (b.u & 0x7FFFFFFFu) == 0;
    }
    if (e == 0xFF) {                       /* inf or NaN */
        *m = 0;
        *s = 0;
        return 0;
    }
    mant = (int32_t)((b.u & 0x007FFFFFu) | 0x00800000u);
    if (b.u & 0x80000000u) mant = -mant;
    sc = 150 - e - p;                      /* a * 2^p == mant * 2^-sc */
    if (sc > LZ_FOLD_SMAX) {
        *m = 0;
        *s = 0;
        return 0;
    }
    if (sc < 0) {
        if (sc < -7) {
            *m = (mant < 0) ? -2147483647 : 2147483647;
            *s = 0;
            return 0;
        }
        *m = mant * (int32_t)(1 << -sc);   /* |mant| < 2^24, so < 2^31 */
        *s = 0;
        return 1;
    }
    *m = mant;
    *s = sc;
    return 1;
}

/* The exp entry's own fold: y32 == v * m * 2^-s for x == v * a * 2^p,
   i.e. m * 2^-s is a * 2^p * log2(e) * 32. LZ_EXP_LOG2E32 stays private
   to this file for the reason lz_sig_q15_t_offset keeps LZ_SIG_LO
   private, so the caller hands over its own constant and gets the table
   coordinate's fold back. The float32 multiply here is the ONLY
   rounding anywhere on the path, and it happens once per layer in place
   of two per element. */
int lz_exp_t_fold(float a, int p, int32_t *m, int *s) {
    return lz_fold_f32(a * LZ_EXP_LOG2E32, p, m, s);
}

/* The guard window in t's domain (t == floor(y32)), derived from the
   float bodies' own x-domain guards rather than written down: the two
   constants below are what `x > 88.0f` and `x < LZ_EXP_LO` are once
   multiplied by LZ_EXP_LOG2E32. One step of slack on each side, so the
   integer test never saturates for an x the float bodies would still
   evaluate; inside that step the ordinary path stays in range (t <=
   4064 keeps n <= 127, and pow2f already returns 0 below -126). */
static int32_t g_expt_hi, g_expt_lo;
static int g_expt_ready;

static void exp_t_bounds(void) {
    g_expt_hi = (int32_t)q8_round(88.0f * LZ_EXP_LOG2E32) + 1;
    g_expt_lo = (int32_t)q8_round(LZ_EXP_LO * LZ_EXP_LOG2E32) - 1;
    g_expt_ready = 1;
}

/* exp, entered through lz_exp's OWN table coordinate as an integer:
 * y32 == v * m * 2^-s, where y32 is what the float body computes as
 * `x * LZ_EXP_LOG2E32` and from which the floor, the table index and
 * the residual all follow. Build (m, s) with lz_exp_t_fold.
 *
 * THE ARITHMETIC, integer up to the last line:
 *   p    = v * m                   exact in int64, |p| <= 2^62
 *   t    = floor(p * 2^-s)         the same floor the magic add
 *                                  computes in lz_exp_fixed
 *   n    = floor(t/32),  idx = t & 31
 *   frac = p - t * 2^s             the low s bits of p, unsigned
 *   sq   = round(frac * 2^(15-s))  == round(r * 2^15), r == y32 - t
 * and sq enters lz_exp_fixed's Taylor unchanged: that body's own `s` is
 * `q8_round(r * 2^20) >> 5`, the same quantity reached through a magic
 * add and a shift. So this is lz_exp_fixed with the two i2f, the magic
 * floor and q8_round removed - not a second approximation of exp.
 *
 * ROUNDED, NOT FLOORED, AT sq. A floor there is a one-sided 2^-16
 * average bias on every element, and the factor it scales multiplies
 * the recurrence state once per step, so the bias would accumulate
 * along the sequence rather than cancel. Clamped at 32767 for the
 * r -> 1 boundary, where the rounded value names the next table entry's
 * residual instead of this one's.
 *
 * DOMAIN: v and m any int32, 0 <= s <= LZ_FOLD_SMAX. Outside that s
 * band the entry REFUSES and returns 0, which a caller must not pass on
 * as exp's value - lz_fold_f32 cannot produce such an s, and the call
 * site checks that fold's return before choosing this path at all.
 *
 * INTERMEDIATE RANGE OVER THE WHOLE ACCEPTED DOMAIN, not over what
 * today's caller presents (the narrow sigmoid_q15_i body overflowed at
 * e == 27 for exactly the other habit): |p| <= 2^62, attained at
 * v == m == INT32_MIN, which is why the product is int64 and not the
 * int32 the caller's own operands would suggest; |t| <= 2^62 before the
 * guards and 4064 after them; frac < 2^s <= 2^62; sq in [0, 32767];
 * cq < 2^21; g_exptab[idx] * cq < 2^42; prod < 2^22. Enumerated in
 * .prof/verify_exp_t.c.
 *
 * THE TIER: this is lz_exp_fixed's twin, so a caller asks
 * lz_scalar_mode() before choosing it, the same way lz_sigmoid_i's
 * callers ask lz_sig_mode(). An entry that ignored the tier would
 * answer from the wrong one. */
float lz_exp_t(int32_t v, int32_t m, int s) {
    int64_t p, t64;
    uint64_t frac;
    int32_t t, n, idx, sq, cq, prod;
    if (!g_exptab_ready) exp_tab_build();
    if (!g_expt_ready) exp_t_bounds();
    if (s < 0 || s > LZ_FOLD_SMAX) return 0.0f;
    p = (int64_t)v * (int64_t)m;
    t64 = fold_floor_shr(p, s);
    if (t64 > g_expt_hi) return FLT_MAX;
    if (t64 < g_expt_lo) return 0.0f;
    t = (int32_t)t64;
    n = (int32_t)fold_floor_shr((int64_t)t, 5);
    idx = t & 31;                          /* always 0..31 in two's complement */
    frac = (s == 0) ? (uint64_t)0
                    : ((uint64_t)p & ((((uint64_t)1) << s) - 1));
    if (s >= 16)
        sq = (int32_t)((frac + ((uint64_t)1 << (s - 16))) >> (s - 15));
    else if (s == 15)
        sq = (int32_t)frac;
    else
        sq = (int32_t)(frac << (15 - s));
    if (sq > 32767) sq = 32767;
    cq = LZ_EXP_QSCALE
       + (int32_t)((((int64_t)LZ_EXP2_LN2 * sq)
                    + (1 << (LZ_EXP_Q - 1))) >> LZ_EXP_Q)
       + (int32_t)((((int64_t)LZ_EXP2_LN2SQ * sq * sq)
                    + (1LL << (2 * LZ_EXP_Q - 1))) >> (2 * LZ_EXP_Q));
    prod = (int32_t)((((int64_t)g_exptab[idx] * cq)
                      + (1 << (LZ_EXP_Q - 1))) >> LZ_EXP_Q);
    /* One multiply and one convert, and that pair is the entire float
       cost of the entry - against the nine semantic conversions the
       float chain into lz_exp_fixed paid per element. */
    LZ_FCX(LZ_FC_EXP, 1, 0, 0, 1, 0);
    return (float)prod * pow2f(n - LZ_EXP_Q);
}

/* ---- softplus -------------------------------------------------------- */

/* log(1+u)/u on u in [0,1], degree 8, Remez (.prof/sp_fit.py). The
 * function this file needs is log1p, and this is the form that makes
 * log1p RELATIVELY accurate for free: the result is u * P(u), so as
 * u -> 0 the answer is u times something that rounds to exactly 1, and
 * no cancellation exists anywhere on the domain. A polynomial for
 * log1p(u) itself would lose every bit of the tail, and the shape the
 * rest of this file uses - strip the exponent, poly the mantissa -
 * loses it too, because 1+u rounds to 1.
 *
 * DEGREE 8 IS WHERE float32 STOPS PAYING, measured over every float32
 * in [2^-24, 1] per degree (.prof/sp_fit.py): 7 -> 3.99e-7, 8 ->
 * 2.03e-7, 9 -> 1.95e-7, 10 -> 2.07e-7. Past 8 the Horner's own
 * rounding grows as fast as the fit shrinks, so the extra multiply buys
 * 4% and then goes backwards. 8 also puts this step under
 * lz_exp_float_body's own error, which is what keeps the composition's
 * accuracy the exp's rather than the log's.
 *
 * `static const float` array rather than inline literals for the reason
 * LZ_EXP_TAB32 is one: an inline float-literal expression on x87 is
 * folded and kept at extended precision (fldt), carrying bits the `f`
 * suffix did not promise. c0 is exactly 1 and is written out anyway -
 * the Horner loop below reads all nine. */
static const float LZ_SP_LOG1P[9] = {
    1.000000000f,    -0.499995530f,    0.333203435f,
   -0.248529464f,     0.191451162f,   -0.137466222f,
    0.079210736f,    -0.030110965f,    0.005384062f,
};

/* Above this the correction log1p(exp(-x)) is smaller than half an ulp
   of x, so returning x is the correctly-rounded answer and not an
   approximation. DERIVED, not inherited: within a binade ulp(x) is
   constant and exp(-x) decreasing, so the binade's left edge is its
   worst point, and 2^4 is the lowest power of two whose edge clears -
   exp(-16) = 1.1e-7 against a half-ulp of 9.5e-7. Enumerated over every
   float32 above it in .prof/verify_softplus.c.

   The old libm form needed a shortcut here for a different reason (its
   exp overflowed); this body's exp argument is -|x| and can never
   overflow, so the shortcut is now only an economy. Measured argument
   range at the live GDN call site is -15.10..11.29 and the KDA gate's
   own pre runs -32.19..10.35, both instrumented over 603 tokens, so it
   fires on no checkpoint in the tree - it is a guard for a model that
   does not exist yet, and it is priced as one. */
#define LZ_SP_HI 16.0f

/* softplus(x) = log(1+e^x), in float, with no libm and no double.
 *
 *   softplus(x) = max(x,0) + log1p(exp(-|x|))
 *
 * is the standard stable rearrangement, and here it is also the cheap
 * one: the exp argument is <= 0, so u = exp(-|x|) lands in (0,1], which
 * is exactly the interval the polynomial above is fitted on, and no
 * second range reduction exists. For x > 0 the sum is of two
 * like-signed terms and cannot cancel.
 *
 * THE EXP IS THE TIER'S. lz_exp's own choice is spelled out here rather
 * than calling lz_exp: this file holds the two exp BODIES and ops.c
 * holds the chooser, so a call upward would make every probe that
 * #includes ops_quant.c link ops.c as well. Same predicate, same two
 * bodies. softplus was the one scalar transcendental outside the tier
 * before this, because it was libm.
 *
 * INTERMEDIATES over the WHOLE accepted domain, which is every float32:
 * u in [0,1] by construction (the abs, then the exp's own guards);
 * P(u) in [0.693, 1]; u*P(u) in [0, 0.694]. Nothing here is an integer,
 * so there is no width to overflow - the hazard this shape has instead
 * is u > 1, where the polynomial is outside its fit and the answer is
 * silently wrong. The abs is what prevents it, which is why
 * .prof/verify_softplus.c's compose row is aimed there.
 *
 * NaN and inf are outside the contract, as they are for lz_exp: `x >
 * LZ_SP_HI` is false for NaN and the exp bodies' own guards let NaN
 * through, so a NaN in gives an unspecified float out. The old libm
 * form propagated NaN; nothing in the engine feeds one. */
/* log1p on u in [0,1], the only new arithmetic here. Its own function
   rather than inline in the caller so that .prof/verify_softplus.c can
   sweep it over every float32 of its domain instead of only over the u
   the exp happens to produce - the same reason fold_floor_shr is
   separate. 8 Horner steps and one multiply. */
static float sp_log1p(float u) {
    float p = LZ_SP_LOG1P[8];
    int i;
    for (i = 7; i >= 0; i--) p = p * u + LZ_SP_LOG1P[i];
    LZ_FCX(LZ_FC_SOFTPLUS, 9, 8, 0, 0, 0);
    return u * p;
}

float lz_softplus_poly(float x) {
    float nx, u;
    if (x > LZ_SP_HI) return x;
    nx = (x < 0.0f) ? x : -x;              /* -|x|; a sign flip, no op */
    u = lz_scalar_mode() ? lz_exp_fixed(nx) : lz_exp_float_body(nx);
    if (x > 0.0f) {
        /* Billed in the branch that pays it rather than flat: the two
           branches really do differ by exactly one add. */
        LZ_FCX(LZ_FC_SOFTPLUS, 0, 1, 0, 0, 0);
        return x + sp_log1p(u);
    }
    return sp_log1p(u);
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

/* Integer-input entry, int-pipeline milestone 5: x == v * 2^-e.
 *
 * EXACT against sigmoid_q15((float)v * 2^-e), not an approximation of
 * it, and the argument is that every step of sigmoid_q15's coordinate
 * derivation is exact in float over this domain, so reproducing it in
 * integers cannot disagree:
 *
 *   t = (x - LZ_SIG_LO) * LZ_SIG_STEP, and both constants are exact
 *   powers of two in disguise: -LZ_SIG_LO * LZ_SIG_STEP == 192 exactly
 *   and LZ_SIG_STEP == 2^4. So with u = v + 12*2^e (an integer, and
 *   POSITIVE inside the clamps below), t == u * 2^(4-e) exactly.
 *   - the add x + 12 is exact while e + 5 <= 24, i.e. e <= 19: x has e
 *     fractional bits and x + 12 < 24 needs 5 integer ones.
 *   - the multiply by 2^4 only moves the exponent field.
 *   - t - (float)idx is exact: idx is t's integer part, and the
 *     difference needs only t's own fractional bits.
 *   - * 32768.0f is another exponent-field move, and the (int) cast
 *     truncates a non-negative value, which is floor.
 *   So frac == floor((t - idx) * 32768), and that is what the shifts
 *   below compute: for sh <= 15 exactly, and for sh > 15 the extra
 *   right shift is the same floor.
 * Beyond e = 19 the float path's `x + 12` starts rounding and this
 * entry is the more accurate of the two rather than the equal one.
 * Verified exhaustively over the full int16 input domain at the
 * exponents the engine uses, .prof/m5sig.c, 0 mismatches.
 *
 * u is formed before any shift so that the arithmetic runs on a
 * non-negative value: a right shift of a negative int is only
 * implementation-defined in C, and this entry must not depend on it.
 *
 * THE NARROW BODY STOPS AT e = 19 FOR A SECOND REASON, and it is not
 * accuracy: u == v + 12*2^e is only an int32 while 24*2^e is, so at
 * e = 27 every |v| above 2^31-1 - 12*2^27 overflows it. No int16
 * producer can reach that, which is why it stood; the conv accumulator
 * is an int32 and can, so the band above is the wide body's.
 *
 * No ARM primitive hook, unlike the two float entries above: those exist
 * to do the coordinate derivation the ARM way, and here there is no
 * float coordinate to derive. */

/* The wide band's coordinate, in units of 2^-15 of one table step:
 *
 *   t == (x - LZ_SIG_LO) * LZ_SIG_STEP and x == v * 2^-e, so
 *   t * 2^15 == 192 * 2^15 + v * 2^(19-e), and for e >= 19 that is
 *   192 * 2^15 + floor(v / 2^(e-19)) EXACTLY - an integer plus a
 *   floored shift, no float and no rounding anywhere. idx and frac are
 *   then this value's two halves, since 192*2^15 is an integer:
 *   floor(t*2^15) == idx*2^15 + floor((t-idx)*2^15).
 *
 * THE SHIFT RUNS ON A BIASED UNSIGNED, not on v itself, for the same
 * reason the narrow body forms u: `>>` on a negative int is only
 * implementation-defined. (uint32_t)v + 2^31 is exactly v + 2^31 as a
 * value in [0, 2^32) for every v including INT32_MIN, and 2^31 is
 * divisible by 2^s for s <= 31, so (uu >> s) == floor(v/2^s) + 2^(31-s)
 * and subtracting that constant back leaves the floor.
 *
 * NOTHING OVERFLOWS FOR s >= 1: |floor(v / 2^s)| <= 2^30, so q is
 * inside +-2^30 and the sum inside +-(2^30 + 192*2^15). s == 0 (e == 19)
 * is exactly where it would not be - v near INT32_MAX plus 192*2^15
 * leaves int32 - which is why the wide band starts at 20 and the
 * subtraction is written before the add.
 *
 * Verified exhaustively in .prof/verify_sig_i_wide.c against a reference
 * that does the floor in long long. */
#define LZ_SIG_T15  (((int32_t)(-(int)LZ_SIG_LO) * 16) << 15)  /* 192 << 15 */
#define LZ_SIG_EMAX 50          /* s == e - 19 <= 31, the widest shift */

static int32_t sig_t15_wide(int32_t v, int e) {
    return (int32_t)fold_floor_shr((int64_t)v, e - 19) + LZ_SIG_T15;
}

/* The table read and the lerp, from a coordinate in units of 2^-15 of
   one table step. Factored out because both integer entries reach the
   table through it, so the interpolation exists once. The caller has
   already established 0 < t15 < LZ_SIG_N << 15, which is what makes
   idx + 1 a legal index and the shift a shift of a positive value. */
static int32_t sig_from_t15(int32_t t15) {
    int idx = (int)(t15 >> 15);
    int32_t frac = t15 & 32767;
    int32_t a = g_sigtab[idx];
    int32_t b = g_sigtab[idx + 1];
    return a + (((b - a) * frac) >> 15);
}

int32_t sigmoid_q15_i(int32_t v, int e) {
    int32_t lim, u, frac, a, b;
    int idx, sh;
    if (!g_sigtab_ready) sig_build();
    if (e < 0 || e > LZ_SIG_EMAX) return sigmoid_q15((float)v * pow2f(-e));
    if (e >= 20) {
        int32_t t15 = sig_t15_wide(v, e);
        /* Live for e <= 27 and provably dead above it: |q| <= 2^(31-s)
           and 2^(31-s) < 192*2^15 once s >= 9, so no int32 v reaches
           either table edge there. Kept rather than branched away - the
           cost is two compares and the alternative is an out-of-range
           g_sigtab read if the table geometry ever moves. */
        if (t15 <= 0) return 0;
        if (t15 >= (int32_t)LZ_SIG_N << 15) return 32768;
        return sig_from_t15(t15);
    } else {
        lim = (int32_t)(-(int)LZ_SIG_LO) << e;  /* x == +-12, the table edge */
        if (v <= -lim) return 0;
        if (v >= lim) return 32768;
        u = v + lim;                            /* 0 < u < 24 * 2^e */
        sh = e - 4;                             /* t == u * 2^-sh */
        if (sh <= 0) { idx = (int)(u << -sh); frac = 0; }
        else {
            idx = (int)(u >> sh);
            frac = u & (((int32_t)1 << sh) - 1);
            frac = (sh <= 15) ? (frac << (15 - sh)) : (frac >> (sh - 15));
        }
    }
    a = g_sigtab[idx];
    b = g_sigtab[idx + 1];
    return a + (((b - a) * frac) >> 15);
}

/* Integer-input sigmoid at a FOLDED scale: x == v * m * 2^-s, with
 * (m, s) from lz_fold_f32. sigmoid_q15_i's scale has to be a power of
 * two and the KDA gate's is not (decay_base[h] * 2^-LZ_KGATE_ES), which
 * is why that site was still deriving its coordinate in float.
 *
 * EXACT, in the sense sigmoid_q15_i's wide band is: t * 2^15 ==
 * 192 * 2^15 + x * 2^19 because -LZ_SIG_LO * LZ_SIG_STEP is 192 exactly
 * and LZ_SIG_STEP is 2^4, and x * 2^19 == v * m * 2^(19-s). So the
 * coordinate is an integer product and a floored shift, with no float
 * and no rounding before the table index. sigmoid_q15_t rounds the same
 * coordinate twice (its multiply and its add, both float32) and then
 * converts three times to take it apart again.
 *
 * DOMAIN: v, m any int32; s any int. INTERMEDIATE RANGE over ALL of it:
 * |v * m| <= 2^62, attained at v == m == INT32_MIN, which is why the
 * product is int64 - the narrow sigmoid_q15_i body's int32 `u` is
 * exactly the intermediate that overflowed at e == 27 when a wider
 * producer arrived, and this one accepts a wider producer by
 * construction. The coordinate then stays int64 until the two clamps
 * have cut it to [0, LZ_SIG_N << 15], so nothing narrows before it is
 * bounded.
 *
 * s < 19 needs a LEFT shift, and that branch saturates on the table's
 * own edge rather than shifting: past the edge every input has the same
 * answer already, so the saturation adds no behaviour - the same
 * argument lz_sig_q15_domain makes for its producers. */
int32_t sigmoid_q15_ti(int32_t v, int32_t m, int s) {
    const int64_t sat = (int64_t)LZ_SIG_N << 15;
    int64_t p, t15;
    if (!g_sigtab_ready) sig_build();
    p = (int64_t)v * (int64_t)m;
    if (s >= 19) {
        t15 = fold_floor_shr(p, s - 19);
    } else if (p == 0) {
        t15 = 0;
    } else {
        int sl = 19 - s;                        /* >= 1 */
        if (sl >= 62 || p > (sat >> sl) || p < -(sat >> sl))
            t15 = (p > 0) ? sat : -sat;
        else
            t15 = p * ((int64_t)1 << sl);
    }
    t15 += LZ_SIG_T15;
    if (t15 <= 0) return 0;
    if (t15 >= sat) return 32768;
    return sig_from_t15((int32_t)t15);
}

/* Declines past e = 27 because 12*2^e leaves int32 there - which is the
   same fact as "no int32 input reaches the table edge at that exponent",
   so a producer at such an exponent needs no clamp rather than a wider
   one. 0 is a refusal, not a bound: a caller must not pass it on as one. */
int32_t lz_sig_q15_domain(int e) {
    if (e < 0 || e > 27) return 0;
    return (int32_t)(-(int)LZ_SIG_LO) << e;
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

/* ---- float divide by integer reciprocal ------------------------------
 *
 * A DIVIDE IS NOT A MULTIPLY on this target: 117 instructions against a
 * multiply's 30 (.prof/arm_chain_count.sh modes 29 and 16), while the
 * f32 census counts both as one. The engine issues 2,251 a token and
 * every one of them is `constant / amax` - q8_group_scale's 127/amax,
 * the recurrence's 32767/amd and gdn_build_table's two - so the COUNT is
 * already at its structural minimum, one reciprocal per quantization
 * group rather than one per element. What is left is the price.
 *
 * __aeabi_fdiv's core is a restoring loop, 4 quotient bits per 12
 * instructions, six times for 24. Generic and small, which libgcc has to
 * be and this does not: the quotient mantissa is 24 bits and ARMv5TE has
 * UMULL. Measured at 91.7 net against its 117, -21.6%.
 *
 * BIT-IDENTICAL, not close. The reciprocal is exact (floor(2^47/mb),
 * corrected), the quotient's remainder is exact, and the rounding is
 * half-to-even on that remainder - so this returns the same float a
 * correctly-rounded divide does, and everything it will not handle
 * (either operand zero/subnormal/inf/NaN, a result outside the normal
 * range) falls through to the divide itself. Swept in
 * .prof/fdiv_recip.c: 31.8M cases, 0 mismatches, including every one of
 * the 2^23 divisor significands against both constants the engine uses.
 *
 * THE TABLE IS BUILT, NOT WRITTEN. 4096 entries is the one part of this
 * that could be silently wrong - one bad entry is a wrong answer on one
 * divisor in 4096, with no symptom - so it comes from its own
 * definition on first use rather than from a literal array that would
 * need a producer script and a check that the two still agree.
 *
 * THE SEED MUST UNDER-ESTIMATE, and this is not a style point. Seeded
 * from each bucket's midpoint the estimate is too large for the
 * bucket's small half; then `2^47 - mb*r` underflows as unsigned, the
 * Newton step adds a garbage term, and the correction loop walks back
 * from there - to the RIGHT answer, every time. The value sweep passed.
 * It measured 655,457 instructions a call. Seeding from the bucket's
 * LARGEST mb makes the estimate one-sided and the correction bounded. */
#define LZ_FDR_BITS 12
#define LZ_FDR_N    (1 << LZ_FDR_BITS)
#define LZ_FDR_SH   (23 - LZ_FDR_BITS)

static uint32_t g_fdr_tab[LZ_FDR_N];
static int g_fdr_ready;

static void fdr_build(void) {
    int i;
    for (i = 0; i < LZ_FDR_N; i++) {
        uint64_t mb = 0x800000u + ((uint64_t)i << LZ_FDR_SH)
                    + ((1u << LZ_FDR_SH) - 1u);
        g_fdr_tab[i] = (uint32_t)(((uint64_t)1 << 47) / mb);
    }
    g_fdr_ready = 1;
}

/* floor(2^47 / mb) exactly, mb in [2^23, 2^24). */
static uint32_t fdr_recip47(uint32_t mb) {
    uint64_t N = (uint64_t)1 << 47;
    uint32_t r = g_fdr_tab[(mb >> LZ_FDR_SH) & (LZ_FDR_N - 1)];
    uint64_t e = N - (uint64_t)mb * r;      /* >= 0: the seed is a floor */
    r += (uint32_t)(((uint64_t)r * e) >> 47);
    while ((uint64_t)mb * (r + 1u) <= N) r++;
    while ((uint64_t)mb * r > N) r--;
    return r;
}

/* Positive control. The two arms of LZ_FDIV_RECIP are BIT-IDENTICAL, so
   a logits comparison cannot tell "the reciprocal ran" from "the switch
   compiled it out" - it reads rc=0 either way. This counter is the only
   thing that separates them, and `declined` is here for the same reason
   in reverse: a build where everything declines would count calls and
   still be running __aeabi_fdiv. */
long long lz_debug_fdiv_recip;
long long lz_debug_fdiv_decl;

float lz_fdiv_recip(float a, float b) {
    union { float f; uint32_t u; } ba, bb, br;
    uint32_t ea, eb, ma, mb, sg, R, Q0;
    uint64_t N, rem;
    int e;
    ba.f = a; bb.f = b;
    ea = (ba.u >> 23) & 255u;
    eb = (bb.u >> 23) & 255u;
    if (ea == 0u || ea == 255u || eb == 0u || eb == 255u) {
        lz_debug_fdiv_decl++;
        return a / b;
    }
    ma = (ba.u & 0x7FFFFFu) | 0x800000u;
    mb = (bb.u & 0x7FFFFFu) | 0x800000u;
    sg = (ba.u ^ bb.u) & 0x80000000u;
    e = (int)ea - (int)eb + 127;
    if (ma < mb) { ma <<= 1; e--; }
    if (e < 1 || e > 254) { lz_debug_fdiv_decl++; return a / b; }
    lz_debug_fdiv_recip++;
    if (!g_fdr_ready) fdr_build();
    R = fdr_recip47(mb);
    Q0 = (uint32_t)(((uint64_t)ma * R) >> 24);
    N = (uint64_t)ma << 23;
    rem = N - (uint64_t)Q0 * mb;
    while (rem >= mb) { Q0++; rem -= mb; }
    if (rem * 2u > mb || (rem * 2u == mb && (Q0 & 1u))) Q0++;
    /* Q0 can reach 2^24; adding it to (e-1)<<23 carries into the
       exponent field by itself, the trick lz_i32f_arm_asm also uses. */
    br.u = sg + ((uint32_t)(e - 1) << 23) + Q0;
    return br.f;
}

int q8_group_scale(const float *grp, int gs, float *sg, float *inv) {
    union { float f; uint32_t u; } bit;
    uint32_t am = q8_amax(grp, gs);
    float amax;
    int fast;
    bit.u = am;
    amax = bit.f;
    if (amax > 0.0f) {
        *sg = amax * LZ_Q8_INV127;
        /* 1,462 of the engine's 2,251 divides a token are this one. */
        *inv = LZ_FDIV(127.0f, amax);
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

/* ---- expof / norm_ss_fixed (integer sum-of-squares) ------------------ */

/* Under no tier guard: a pure read of a float's exponent field, used by
   the GDN coefficient scale (ops_gdn.c) and by the fixed norms
   (ops_norm.c, and norm_ss_fixed below). Guarding it on either one
   would make that subsystem's off switch compile the other one out. */
int expof(float x) {
    union { float f; uint32_t u; } b;
    b.f = x;
    return (int)((b.u >> 23) & 0xFFu) - 127;
}

#if LZ_NORM_FIXED
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
#if defined(LZ_ARM_ASM_EXTERN)
    /* The hand-written loop's preconditions, all three answered from
       the amax scan that just ran - see src/ops_arm.h. They are a
       property of the ROW, so testing them here costs O(1) where the
       equivalent per-element guards would cost two instructions each.
       Anything outside the band runs the C loop below. */
    if (g_kernel == LZ_KERNEL_ARM_ASM && bit.u < 0x7F800000u &&
        e >= -105 && e <= 125) {
        long long facc;
        lz_norm_ss_fixed_arm_asm(x, n, e, qout, &facc);
        if (eout) *eout = e;
        return (float)facc * pow2f(-2 * e);
    }
#endif /* LZ_ARM_ASM_EXTERN */
    sc = pow2f(e);
    i = 0;
#if defined(LZ_NORM_SS_SSE2_EXTERN)
    /* SELECTED BY TIER, like every other kernel here: --kernel ref and
       mmx must reach the C loop below or the arm this is compared
       against would be this. The kernel takes whole groups of 8 and
       hands back the count; the tail runs in C, so `i` continues rather
       than restarting. */
    if (g_kernel == LZ_KERNEL_SSE2 && n >= 8) {
        long long sacc = 0;
        i = lz_norm_ss_sse2(x, n, sc, qout, &sacc);
        acc = sacc;
    }
#endif /* LZ_NORM_SS_SSE2_EXTERN */
    for (; i < n; i++) {
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
/* The fixed norms tier is compiled out. Declining is the same answer
   the oversized-n case above gives, and reaches every caller through
   the one path they already handle, so each one runs its float body.
   Other fixed tiers in this build are unaffected. */
float norm_ss_fixed(const float *x, int n, short *qout, int *eout) {
    (void)x; (void)n; (void)qout; (void)eout;
    return -1.0f;
}
#endif /* LZ_NORM_FIXED */
