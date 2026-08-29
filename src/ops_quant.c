/* ops_quant.c - Quantization primitives and fixed-point transcendental
   helpers. A separate TU from ops.c to keep its #if guard density low.

   Every function here is extern, declared in ops_quant.h. */
#include "lz_int.h"   /* lz_i64: the 64-bit type, portably */
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

/* Watcom's C90 math.h has no sqrtf: route to double sqrt there. gcc
   keeps sqrtf so the ARM build pulls no libgcc double soft-float. */
#if defined(__WATCOMC__)
#define LZ_SQRTF(x) ((float)sqrt((double)(x)))
#else
#define LZ_SQRTF(x) sqrtf(x)
#endif /* __WATCOMC__ */
#include <string.h>

#include "ops.h"
#include "ops_arm.h"     /* LZ_ARM_ASM_EXTERN, lz_norm_ss_fixed_arm_asm */
#include "err.h"
#include "cpucheck.h"
#include "mmx_compat.h"
#include "ops_mmx.h"
#include "ops_sse.h"
#include "ops_sse2.h"
#include "ops_sched.h"   /* lz_cpu_has_sse - norm_ss_fixed's SSE1 arm */
#include "ops_kernel_shared.h" /* lz_i32f, LZ_ROW_N, LZ_DEFINE_PICK, g_kernel */
#include "ops_quant.h"

#include "ops_kernel_amax.h"  /* dispatch table for q8_amax */

/* LZ_NORM_MAX_N: ops.h. */

/* ---- x87 precision control ------------------------------------------- */

#if defined(LZ_X87_FLOAT_CW)   /* the detect is in ops.h, once */
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

/* (a*b + 2^22) >> 23 rounding half-up, as int32 - the Q23 Newton step
   shape of lz_rsqrt_fixed. Fused multiply-round-shift, the same trick
   as the exp helpers further down: Watcom's natural `(lz_i64)a*b` and
   `>> 23` would be __I8M + __I8RS per step, one imul + one shrd here.
   Valid for the rsqrt operands, whose mantissa-scale values keep the
   shifted result inside int32. */
#if defined(__WATCOMC__)
static int32_t lz_i64mul_rnd_rsh23(int32_t a, int32_t b);
#pragma aux lz_i64mul_rnd_rsh23 = \
    "imul ebx" \
    "add  eax, 0x00400000" \
    "adc  edx, 0" \
    "mov  ecx, 23" \
    "shrd eax, edx, cl" \
    "sar  edx, cl" \
    parm [eax] [ebx] \
    value [eax] \
    modify [ecx edx];
#else
#define lz_i64mul_rnd_rsh23(a, b) \
    (int32_t)((((lz_i64)(a) * (b)) + (1 << 22)) >> 23)
#endif /* __WATCOMC__ */

float lz_rsqrt_fixed(float x) {
    static const float LZ_RSQRT_SQRT2 = 0.7071067811865476f;
    union { float f; uint32_t u; } b;
    int32_t m23, y23;
    int32_t p, t, fac;
    int e, idx, odd;
    if (!g_rsqrttab_ready) rsqrt_tab_build();
    b.f = x;
    e = (int)((b.u >> 23) & 0xFFu) - 127;
    if (e < -126 || e > 126) return lz_rsqrt_float_body(x);
    m23 = (int32_t)((b.u & 0x007FFFFFu) | 0x00800000u);
    idx = (int)((b.u >> 15) & 0xFFu);
    y23 = g_rsqrttab[idx];
    /* Two Newton steps y = y*(3 - m*y^2)/2, all Q23. The shifts round
       half-up - truncation would bias a whole-vector scale. Every
       intermediate stays in int32 (the mantissa-scale operands make the
       >>23 results ~2^23 and the fac arithmetic positive), so each step
       is a fused multiply-round-shift. */
    p = lz_i64mul_rnd_rsh23(y23, y23);
    t = lz_i64mul_rnd_rsh23(m23, p);
    fac = (3 * (1 << 23) - t + 1) >> 1;
    y23 = lz_i64mul_rnd_rsh23(y23, fac);
    p = lz_i64mul_rnd_rsh23(y23, y23);
    t = lz_i64mul_rnd_rsh23(m23, p);
    fac = (3 * (1 << 23) - t + 1) >> 1;
    y23 = lz_i64mul_rnd_rsh23(y23, fac);
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

/* The two Q20 Taylor terms as fused multiply-round-shift leaves. gcc and
   the ARM cross-compiler widen to lz_i64 and let the hardware do the
   multiply; Watcom routes `(lz_i64)a * b` through its __I8M library
   call and the shifts through __I8RS, shuffling 64-bit register pairs
   for every operation - lz_exp_fixed measured 46% moves on Watcom
   against 22% on gcc (build/move_per_function.sh), and the shuffle is
   most of the gap. The #pragma aux bodies emit the same integer product
   and shift directly (imul/shrd), so the two toolchains agree bit for
   bit: the multiply is exact and shifting a fixed-point value by its
   own scale is a permutation of the same bits.

   PRECONDITION (both arms): a and b are non-negative. The two call
   sites satisfy it - s/sv[k] is the Q20 residue in [0, 2^15]
   (build/exp_q20_probe.c drives both ends) and the table/coefficient
   inputs are scaled by 2^20. q40's second multiply is MUL rather than
   IMUL because the low word of a*b must not be sign-extended into the
   64x32 product, and with b >= 0 the signed and unsigned results
   coincide. */
#if defined(__WATCOMC__)
static int32_t lz_exp_q20(int32_t a, int32_t b); /* (a*b + 2^19) >> 20 */
#pragma aux lz_exp_q20 = \
    "imul ebx" \
    "add  eax, 0x00080000" \
    "adc  edx, 0" \
    "mov  ecx, 20" \
    "shrd eax, edx, cl" \
    "sar  edx, cl" \
    parm [eax] [ebx] \
    value [eax] \
    modify [ecx edx];

static int32_t lz_exp_q40(int32_t a, int32_t b); /* (a*b*b + 2^39) >> 40 */
#pragma aux lz_exp_q40 = \
    "imul ebx" \
    "mov  ecx, edx" \
    "mul  ebx" \
    "imul ecx, ebx" \
    "add  edx, ecx" \
    "add  eax, 0" \
    "adc  edx, 0x80" \
    "mov  eax, edx" \
    "sar  eax, 8" \
    parm [eax] [ebx] \
    value [eax] \
    modify [ecx edx];
#else
#define lz_exp_q20(a, b) \
    (int32_t)((((lz_i64)(a) * (b)) + (1 << (LZ_EXP_Q - 1))) >> LZ_EXP_Q)
#define lz_exp_q40(a, b) \
    (int32_t)((((lz_i64)(a) * (b) * (b)) \
               + (LZ_I64_C(1) << (2 * LZ_EXP_Q - 1))) >> (2 * LZ_EXP_Q))
#endif /* __WATCOMC__ */

/* The bare 32x32 -> 64 multiply, for the products in the exp twins that
   have no fused shift (lz_exp_t's v*m). Watcom routes `(lz_i64)a * b`
   through __I8M unless the operands are simple memory loads it can
   narrow; a fused aux is one imul. gcc widens and multiplies, exact
   either way. */
#if defined(__WATCOMC__)
static lz_i64 lz_i64mul32(int32_t a, int32_t b);
#pragma aux lz_i64mul32 = \
    "imul ebx" \
    parm [eax] [ebx] \
    value [edx eax] \
    modify [edx];
#else
#define lz_i64mul32(a, b) ((lz_i64)(a) * (b))
#endif /* __WATCOMC__ */

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

/* Round-half-even on the bit pattern, UNCONDITIONALLY - the mode-free
   twin of lz_q8_round_bits, which follows the FPU mode under
   LZ_SOFTFP_FAST. The exp floor needs this because floor is a
   mathematical function: the decomposition t = round-half-even(y32-0.5)
   (= floor(y32) off the exact-integer boundaries) must not move with
   the FPU rounding mode. Under RTZ (--fastfp on / LZ_SOFTFP_FAST) the
   magic-add floor reads floor(y32-0.5), which for negative y32 with a
   small fractional y32-0.5 lands one BELOW the RNE the ARM asm floor
   makes (-127.5 -> -129 vs -128), and the ARM asm floor DECLINES past
   |N| > 2^24, falling to this C - so both targets need the mode-free
   floor in the fallback. Same body as lz_q8_round_bits' RNE arm. */
LZ_MAYBE_UNUSED static int lz_rne_round_bits(uint32_t u, int k, int32_t *out) {
    uint32_t m, t;
    int32_t r;
    int e = (int)((u >> 23) & 0xFFu) + k;
    if (e >= 149) return 0;                 /* |v| >= 2^22, or inf/nan */
    e = 150 - e;                            /* bits to drop, >= 2 */
    if (e >= 25) { *out = 0; return 1; }    /* |v| < 0.5 */
    m = (u & 0x00FFFFFFu) | 0x00800000u;
    r = (int32_t)(m >> e);
    t = m << (32 - e);
    if (t > 0x80000000u) r++;
    else if (t == 0x80000000u) r += (r & 1);  /* half to even */
    if (u & 0x80000000u) r = -r;
    if (r == 0x400000) return 0;
    *out = r;
    return 1;
}

float lz_exp_fixed(float x) {
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
        /* The magic-add floor needs round-to-nearest: under RTZ
           (--fastfp on / LZ_SOFTFP_FAST) the sum truncates and the
           extraction reads floor(y32-0.5), which for negative y32 lands
           one BELOW the round-half-even floor (-127.5 -> -129 vs -128) -
           the softmax's exp diverged. The floor is a MATHEMATICAL
           function (t = round-half-even(y32-0.5) = floor(y32) off the
           exact-integer boundaries); it must not move with the FPU
           rounding mode. The ARM asm floor agrees on RNE but DECLINES
           past |N| > 2^24 (a value like y32=-127.5 lands just past it)
           and falls to this C, so BOTH targets take the mode-free RNE
           path here under fastfp. */
        union { float f; uint32_t u; } du;
        du.f = y32 - 0.5f;
        if (lz_fastfp()) {
            if (lz_rne_round_bits(du.u, 0, &t)) {
                n = t >> 5;
                idx = t & 31;
                r = y32 - (float)t;
                rq = q8_round(r * (float)LZ_EXP_QSCALE);
            } else {
                du.f = y32 - 0.5f + 12582912.0f;
                t = (du.u & 0x007FFFFF) - 0x00400000;
                n = t >> 5;
                idx = t & 31;
                r = y32 - (float)t;
                rq = q8_round(r * (float)LZ_EXP_QSCALE);
            }
        } else {
            du.f = y32 - 0.5f + 12582912.0f;
            t = (du.u & 0x007FFFFF) - 0x00400000;
            n = t >> 5;
            idx = t & 31;
            r = y32 - (float)t;
            rq = q8_round(r * (float)LZ_EXP_QSCALE);
        }
    }
    /* s = r/32 = rq/2^25, so the Q20 Taylor terms are (ln2_q*s)>>20 and
       (ln2sq_q*s*s)>>40; both shifts round half-up. */
    s = rq >> 5;
    cq = LZ_EXP_QSCALE
       + lz_exp_q20(LZ_EXP2_LN2, s)
       + lz_exp_q40(LZ_EXP2_LN2SQ, s);
    prod = lz_exp_q20(g_exptab[idx], cq);
#if defined(LZ_ARM_PRIM_ASM)
    if (asm_tier) {
        int32_t k = n - LZ_EXP_Q;
        if (k >= -126 && k <= 127) {
            union { float f; uint32_t u; } pb;
            pb.f = lz_i32f_arm_asm(prod);     /* == (float)prod, exact here */
            if (lz_scale2_arm_asm(pb.u, k, &pb.u)) {
                return pb.f;
            }
        }
    }
#endif /* LZ_ARM_PRIM_ASM */
    {
        float exp_res = (float)prod * pow2f(n - LZ_EXP_Q);
        return exp_res;
    }
}

/* lz_exp_fixed over a RUN, for the same reason lz_sigmoid_q15_run
   exists: the Q20 Taylor is four widening multiplies an element and a
   kernel cannot see them one call at a time.
 *
 * THE SPLIT IS BY DOMAIN, not by branch, which is what makes it simpler
 * than the sigmoid run's. Only the two guard exits are per-element
 * decisions; everything between them is straight-line. So the scalar
 * front pass writes the table entry, the Q15 residue and the exit
 * exponent for each element and marks the guarded ones, the kernel runs
 * the Taylor over the whole block, and the float exit multiply walks
 * back over it. That exit stays scalar deliberately - pow2f is a
 * table-free exponent build and the multiply is one float op, so
 * vectorising it would need the exponents equal, which across a softmax
 * row they are not.
 *
 * The guarded slots get tv = LZ_EXP_QSCALE and sv = 0 rather than being
 * left uninitialised: the kernel reads every lane of the block whether
 * or not its answer survives, and a garbage operand is a garbage
 * operand even when its result is discarded. Their answers are written
 * last, after the exit loop, so nothing the vector part did can reach
 * the output.
 *
 * s IS NON-NEGATIVE AND AT MOST 2^15, which is what lets the kernel use
 * pmuludq: the magic floor rounds y32 - 0.5 to nearest, so t <= y32 and
 * r = y32 - t lands in [0, 1] (1 exactly when y32 is an integer and the
 * tie rounds down), giving rq in [0, 2^20] and s = rq >> 5 in [0, 2^15].
 * build/exp_q20_probe.c drives both ends of that range.
 *
 * Bit-identical to calling lz_exp_fixed per element, and the float-op
 * census is charged per element in the same place, so neither the
 * numbers nor the cost table move. The ARM asm tier returns to the
 * scalar entry outright: its floor and residue are inline primitives
 * that would have to be replayed here to stay identical, and it has no
 * x86 kernel to reach anyway. */
void lz_exp_fixed_run(const float *x, float *out, int n) {
    enum { LZ_EXPRUN = 64 };
    int32_t tv[LZ_EXPRUN], sv[LZ_EXPRUN], pv[LZ_EXPRUN], ev[LZ_EXPRUN];
    unsigned char sat[LZ_EXPRUN];
    int i, k, m;

#if defined(LZ_ARM_PRIM_ASM)
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        for (i = 0; i < n; i++) out[i] = lz_exp_fixed(x[i]);
        return;
    }
#endif /* LZ_ARM_PRIM_ASM */
    if (!g_exptab_ready) exp_tab_build();
    for (i = 0; i < n; i += LZ_EXPRUN) {
        m = n - i;
        if (m > LZ_EXPRUN) m = LZ_EXPRUN;
        for (k = 0; k < m; k++) {
            union { float f; int32_t i; } u;
            float xv = x[i + k], y32, r;
            int32_t t;
            sat[k] = 0;
            if (xv > 88.0f)          sat[k] = 1;
            else if (xv < LZ_EXP_LO) sat[k] = 2;
            if (sat[k]) { tv[k] = LZ_EXP_QSCALE; sv[k] = 0; ev[k] = 0; continue; }
            LZ_FCX(LZ_FC_EXP, 3, 3, 0, 1, 0);
            y32 = xv * LZ_EXP_LOG2E32;
            /* Same floor as the scalar entry's fastfp fix (see
               lz_exp_fixed): the magic add reads floor(y32-0.5) under
               RTZ, one below round-half-even for negative y32. The
               softmax reaches THIS entry (LZ_SOFTMAX_RUN), so the scalar
               fix alone left the softmax diverging under --fastfp. */
            if (lz_fastfp()) {
                union { float f; uint32_t u; } du;
                du.f = y32 - 0.5f;
                if (lz_rne_round_bits(du.u, 0, &t)) {
                    r = y32 - (float)t;
                    tv[k] = g_exptab[t & 31];
                    sv[k] = q8_round(r * (float)LZ_EXP_QSCALE) >> 5;
                    ev[k] = (t >> 5) - LZ_EXP_Q;
                    continue;
                }
            }
            u.f = y32 - 0.5f + 12582912.0f;
            t = (u.i & 0x007FFFFF) - 0x00400000;
            r = y32 - (float)t;
            tv[k] = g_exptab[t & 31];               /* t & 31 is [0,31] for
                                                       negative t too */
            sv[k] = q8_round(r * (float)LZ_EXP_QSCALE) >> 5;
            ev[k] = (t >> 5) - LZ_EXP_Q;
        }
#if defined(LZ_HAVE_EXP_Q20_SIMD)
        /* SSE2 first: four lanes against MMX's two, and it owes no
           emms. Below it the MMX cell, which a machine without SSE2
           still reaches - same order as the sigmoid run's. */
        if (g_kernel == LZ_KERNEL_SSE2)
            lz_exp_q20_simd(tv, sv, pv, m);
        else
#endif /* LZ_HAVE_EXP_Q20_SIMD */
#if defined(LZ_HAVE_EXP_Q20_SSE)
        /* Above the MMX cell and below SSE2: same width as MMX, fewer
           instructions, and a machine with SSE1 but not SSE2 (Pentium
           III, Athlon) is a real point on this target family. Selected
           by tier, not CPUID: reading CPUID here would shadow the MMX
           arm on every machine that runs the suite and need a separate
           control build to get underneath. */
        if (g_kernel == LZ_KERNEL_SSE)
            lz_exp_q20_sse(tv, sv, pv, m);
        else
#endif /* LZ_HAVE_EXP_Q20_SSE */
#if defined(LZ_HAVE_EXP_Q20_MMX)
        if (g_kernel != LZ_KERNEL_REF && lz_cpu_has_mmx())
            lz_exp_q20_mmx(tv, sv, pv, m);
        else
#endif /* LZ_HAVE_EXP_Q20_MMX */
        for (k = 0; k < m; k++) {
            int32_t cq = LZ_EXP_QSCALE
               + lz_exp_q20(LZ_EXP2_LN2, sv[k])
               + lz_exp_q40(LZ_EXP2_LN2SQ, sv[k]);
            pv[k] = lz_exp_q20(tv[k], cq);
        }
        for (k = 0; k < m; k++)
            out[i + k] = (float)pv[k] * pow2f(ev[k]);
        for (k = 0; k < m; k++) {
            if (!sat[k]) continue;
            out[i + k] = (sat[k] == 1) ? FLT_MAX : 0.0f;
        }
    }
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
 * pair of float32 roundings a naive chain pays PER ELEMENT (`(float)s15
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

   This is the 64-bit floor shift, factored out so the file has ONE of
   them rather than a copy inline in sig_t15_wide, which calls this
   instead. It is not a second epi_shr_half either - that
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
    p = lz_i64mul32(v, m);
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
       + lz_exp_q20(LZ_EXP2_LN2, sq)
       + lz_exp_q40(LZ_EXP2_LN2SQ, sq);
    prod = lz_exp_q20(g_exptab[idx], cq);
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
    /* After sig_build, never before: the primitive reads the table.
       NOT under fastfp: the asm rounds the coordinate half-to-even on
       the integer mantissa, while the C below reads (x - LZ_SIG_LO) *
       LZ_SIG_STEP with the soft-float add and then truncates - under
       RTZ (LZ_SOFTFP_FAST) those disagree for inputs whose coordinate
       lands on a half, and the SwiGLU's sigmoid diverged from x86
       (probe: ~0.24% of random x differ). The C's soft-float add is
       RTZ-consistent with x86, so the C path is the fastfp tier. */
    if (!LZ_SOFTFP_FAST && g_kernel == LZ_KERNEL_ARM_ASM) {
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

/* sigmoid_q15 over a RUN, which is what the MMX interpolation cell
   needs and what the scalar entry above cannot give it: a kernel that
   vectorises the arithmetic has to see more than one element.
 *
 * THE SPLIT IS BY BRANCH, not by element. Three of sigmoid_q15's exits
 * are saturations - below the table, above it, and past its last index
 * - and they are per-element decisions the vector part cannot make. So
 * this walks once in scalar to write idx, frac and the two table
 * entries for the elements that INTERPOLATE, marking the rest; runs
 * the kernel over the whole block; then writes the saturated answers
 * over their own slots. The saturating elements are rare in practice
 * (the table spans the range where sigmoid is not already 0 or 1) and
 * correctness does not depend on that - only the cost does.
 *
 * idx IS CLAMPED BEFORE THE TABLE READ even for elements whose answer
 * is about to be overwritten, because a[k] and b[k] are read for every
 * slot and an out-of-range index is an out-of-bounds load whether or
 * not the value is used. The scalar entry never had this problem: it
 * returns before it reads.
 *
 * The result is bit-identical to calling sigmoid_q15 per element - the
 * interpolating path is the same arithmetic and the saturating paths
 * are copied verbatim. build/lerp_q15_probe.c covers the kernel; this
 * wrapper's own risk is the branch split, which is why the SATURATED
 * slots are written last and unconditionally. */
void lz_sigmoid_q15_run(const float *x, int32_t *out, int n) {
    enum { LZ_SIGRUN = 64 };
    int32_t ta[LZ_SIGRUN], tb[LZ_SIGRUN], tf[LZ_SIGRUN];
    unsigned char sat[LZ_SIGRUN];
    int i, k, m;

    if (!g_sigtab_ready) sig_build();
    for (i = 0; i < n; i += LZ_SIGRUN) {
        m = n - i;
        if (m > LZ_SIGRUN) m = LZ_SIGRUN;
        for (k = 0; k < m; k++) {
            float xv = x[i + k], t;
            int idx;
            sat[k] = 0;
            if (xv <= LZ_SIG_LO)        { sat[k] = 1; out[i + k] = 0;     }
            else if (xv >= -LZ_SIG_LO)  { sat[k] = 1; out[i + k] = 32768; }
            t = (xv - LZ_SIG_LO) * LZ_SIG_STEP;
            idx = (int)t;
            if (idx >= LZ_SIG_N) {
                if (!sat[k]) { sat[k] = 1; out[i + k] = g_sigtab[LZ_SIG_N]; }
                idx = LZ_SIG_N - 1;
            } else if (idx < 0) {
                idx = 0;
            }
            ta[k] = g_sigtab[idx];
            tb[k] = g_sigtab[idx + 1];
            tf[k] = sat[k] ? 0 : (int32_t)((t - (float)idx) * 32768.0f);
        }
#if defined(LZ_HAVE_LERP_Q15_SIMD)
        /* SSE2 first: eight lanes against MMX's four, and it owes no
           emms. Below it the MMX cell, which a machine without SSE2
           still reaches. */
        if (g_kernel == LZ_KERNEL_SSE2)
            lz_lerp_q15_simd(ta, tb, tf, out + i, m);
        else
#endif /* LZ_HAVE_LERP_Q15_SIMD */
#if defined(LZ_HAVE_LERP_Q15_MMX)
        if (g_kernel != LZ_KERNEL_REF && lz_cpu_has_mmx())
            lz_lerp_q15_mmx(ta, tb, tf, out + i, m);
        else
#endif /* LZ_HAVE_LERP_Q15_MMX */
        for (k = 0; k < m; k++)
            out[i + k] = ta[k] + (((tb[k] - ta[k]) * tf[k]) >> 15);
        for (k = 0; k < m; k++) {
            if (!sat[k]) continue;
            if (x[i + k] <= LZ_SIG_LO)       out[i + k] = 0;
            else if (x[i + k] >= -LZ_SIG_LO) out[i + k] = 32768;
            else                             out[i + k] = g_sigtab[LZ_SIG_N];
        }
    }
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
 * that does the floor in lz_i64. */
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
    p = lz_i64mul32(v, m);
    if (s >= 19) {
        t15 = fold_floor_shr(p, s - 19);
    } else if (p == 0) {
        t15 = 0;
    } else {
        int sl = 19 - s;                        /* >= 1 */
        if (sl >= 62 || p > (sat >> sl) || p < -(sat >> sl))
            t15 = (p > 0) ? sat : -sat;
        else
            t15 = p << sl;                      /* p * 2^sl, exact under the guard */
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
lz_i64 lz_debug_fdiv_recip;
lz_i64 lz_debug_fdiv_decl;

/* Same shape, same reason, for LZ_Q8R_BITS: its two arms are also
   bit-identical, so nothing in a logits cmp separates "the integer
   rounding ran" from "the switch compiled it out". Defined
   unconditionally - a counter that only exists in the arm it counts
   cannot report zero, and zero is the answer that matters. */
lz_i64 lz_debug_q8r_bits;
lz_i64 lz_debug_q8r_decl;

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
#if LZ_SOFTFP_FAST
    /* Round toward zero: the correction loop above already made Q0 the
       exact floor, so the remainder is dropped rather than examined.
       x86's hardware division under MXCSR=truncate rounds the same way;
       the reciprocal's round-half-even here is what broke the fast-mode
       pairing (RNE made the two agree, RTZ exposed them). */
#else
    if (rem * 2u > mb || (rem * 2u == mb && (Q0 & 1u))) Q0++;
#endif /* LZ_SOFTFP_FAST */
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
        /* fast is ALWAYS 1 here, and every caller's div branch never
           runs. fast=0 would need 127/amax to overflow FLT_MAX, i.e.
           amax < 127/FLT_MAX ~ 3.7e-36; the floor below fires at amax <
           1.27e-28 (sg = amax/127 < 1e-30), eight orders of magnitude
           earlier, and forces fast=1. So amax is never in the range that
           makes *inv infinite. Verified over the full float domain
           (.prof/q8scale_probe.c, 4.2M calls covering every exponent
           field: zero fast=0). The div branches are kept as defensive
           fallbacks for a future change to LZ_Q8_MIN_SCALE, not as live
           alternatives. */
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
    lz_i64 acc = 0;
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
        lz_i64 facc;
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
        lz_i64 sacc = 0;
        i = lz_norm_ss_sse2(x, n, sc, qout, &sacc);
        acc = sacc;
    }
#endif /* LZ_NORM_SS_SSE2_EXTERN */
#if defined(LZ_NORM_SS_SSE_EXTERN)
    /* The SSE1 tier, BELOW the SSE2 one so a machine with both takes the
       wider kernel. Selected by tier, not CPUID: reading CPUID would make
       it fire for --kernel mmx too and leave it unreachable from any
       knob on a machine with SSE2, which is every machine in the
       suite. */
    if (i == 0 && g_kernel == LZ_KERNEL_SSE && n >= 4) {
        lz_i64 sacc = 0;
        i = lz_norm_ss_sse(x, n, sc, qout, &sacc);
        acc = sacc;
    }
#endif /* LZ_NORM_SS_SSE_EXTERN */
#if defined(LZ_HAVE_NORM_SS_PACK_MMX)
    /* The MMX tier, below both. A machine with MMX and no SSE1 - the
       floor of the target family - ran the scalar loop for every
       element and now runs it only for the tail. Two passes over each
       32: x87 does the scale and the round into a scratch, then the
       kernel does the clamp and the sum of squares four wide and takes
       the emms with it. The x87 pass IS the C's arithmetic, so what
       changes is only which instructions the clamp and the accumulate
       use.

       Reached by --kernel mmx, which is why there is no LZ_NORM_SS_FORCE_MMX
       / LZ_NORM_SS_FORCE_SSE pair to select the arms above it: LZ_KERNEL_SSE
       selects them at run time, so a compile-time escape is pointless.
       The three arms cascade on g_kernel with `i == 0` carrying the
       fallthrough. A compile-time
       escape beside a runtime tier is a second way to say the same
       thing, and two ways can disagree. */
    if (i == 0 && g_kernel != LZ_KERNEL_REF && lz_cpu_has_mmx() && n >= 32) {
        int32_t blk[32];
        int j;
        for (; i + 31 < n; i += 32) {
            for (j = 0; j < 32; j++) blk[j] = q8_round(x[i + j] * sc);
            acc += lz_norm_ss_pack_mmx(blk, qout ? qout + i : (short *)0);
        }
    }
#endif /* LZ_HAVE_NORM_SS_PACK_MMX */
    for (; i < n; i++) {
        int v = q8_round(x[i] * sc);           /* one mul, one magic add */
        if (v >  32767) v =  32767;
        if (v < -32767) v = -32767;
        if (qout) qout[i] = (short)v;
        acc += (lz_i64)(v * v);
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
    /* x87 has the rounding half of fast mode but no FTZ/DAZ - see
       lz_fastfp_g's comment for why that gap is measured-harmless. */
    if (lz_fastfp_g) _control87(_RC_CHOP, _MCW_RC);
    return save;
#elif defined(LZ_X87_FLOAT_CW)
    unsigned short save = lz_x87_cw_get();
    unsigned short cw = (unsigned short)(save & ~0x0300u);   /* PC=24 */
    if (lz_fastfp_g) cw = (unsigned short)(cw | 0x0C00u);    /* RC=chop */
    lz_x87_cw_set(cw);
    return save;
#elif defined(LZ_SSE_FLOAT_CSR)
    {
        unsigned csr, save;
        __asm__ __volatile__("stmxcsr %0" : "=m"(csr));
        save = csr;
        if (lz_fastfp_g) {
            /* bit15 FTZ, bit6 DAZ, bits14:13 RC=11 (toward zero). */
            csr |= 0x8040u;
            csr = (csr & ~0x6000u) | 0x6000u;
            __asm__ __volatile__("ldmxcsr %0" : : "m"(csr));
        }
        return save;
    }
#else
    return 0;   /* nothing to set: already float semantics, strict mode */
#endif /* __WATCOMC__ */
}

void lz_fpu_float_end(unsigned save) {
#if defined(__WATCOMC__)
    /* _MCW_RC too, not just _MCW_PC: fast mode set the rounding half and
       leaving it set would follow the caller out of the region. */
    _control87(save, _MCW_PC | _MCW_RC);
#elif defined(LZ_X87_FLOAT_CW)
    lz_x87_cw_set((unsigned short)save);
#elif defined(LZ_SSE_FLOAT_CSR)
    if (lz_fastfp_g) __asm__ __volatile__("ldmxcsr %0" : : "m"(save));
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
    return 1.0f / LZ_SQRTF(x);
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
    for (g = 0; g < (int)((unsigned)n / (unsigned)gs); g++) {
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
    int ns = (unsigned)n >> 5;
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
    int ns = (unsigned)n >> 5;
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
    for (g = 0; g < (int)((unsigned)n / (unsigned)gs); g++) {
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
                    qi -= (int32_t)(((mu << 7) - mu - tamu * (uint32_t)qi
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
    for (g = 0; g < (int)((unsigned)n / (unsigned)gs); g++) {
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
                qi -= ((m << 8) - (m << 1) + tam < 2 * tam * qi);
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
    for (g = 0; g < (int)((unsigned)n / (unsigned)gs); g++) {
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
    for (grp = 0; grp < (int)((unsigned)n / (unsigned)gs); grp++) {
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
    n2 = LZ_SQRTF(n2);
    /* sqrt(n) is exact for the power-of-two head dims this path serves;
       the divide has a runtime divisor, so both compilers emit a real
       FDIV rather than one strength-reducing it. */
    s = n2 / LZ_SQRTF((float)n);
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
