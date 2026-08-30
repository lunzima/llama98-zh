/* ops_quant.h - declarations for ops_quant.c's non-static functions.
   Quantization primitives: rounding, amax, group scale, fixed-point
   transcendental tables and helpers. */
#ifndef OPS_QUANT_H
#define OPS_QUANT_H

#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include "ops.h"
#include "ops_arm_prim.h"   /* LZ_ARM_PRIM_ASM, lz_q8_round_pow2_arm_asm */

/* The runtime-selected ISA tier, defined in ops_sched.c. Declared here
   rather than pulling in ops_kernel_shared.h, which would put lz_i32f
   and the dispatch macros into every TU that only wants to round. */
extern int g_kernel;

/* LZ_MAYBE_UNUSED is in lz_int.h, with the other portability spellings.
   Two of the TUs that include this one (ops_matmul.c, ops_sched.c) want
   the declarations and never round, and an unused-function warning per
   TU is not a price worth paying for an inline body. */

/* 2^e by integer manipulation of the exponent field. */
float pow2f(int e);

/* Positive controls for LZ_Q8R_BITS, defined in ops_quant.c. Both arms
   of that switch are bit-identical, so these are what separate them. */
extern lz_i64 lz_debug_q8r_bits;
extern lz_i64 lz_debug_q8r_decl;

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
   to rewrite, so the magic add is a full __aeabi_fadd - 43 instructions
   measured for what is a shift and a rounding decision on the bit
   pattern. --kernel arm-asm takes that path (src/ops_arm_prim.h);
   outside |qv| < 2^22, the domain stated above, the primitive declines
   and the magic add below runs, so the two tiers agree everywhere and
   not just where the contract holds.

   IN THE HEADER, AND THE REASON IS A COUNT. It is called 234,713 times
   per token - instrumented on .prof/x87test over five tokens, not
   derived from the census, which counts SOURCE-level invocations and
   reads 178,176 for a different model shape. Measured out of line it
   costs 36 instructions on the arm-asm tier while the primitive it
   dispatches to is about 20: the call, the argument and result moves,
   and the g_kernel load that cannot be hoisted out of a caller's loop
   are the rest. Same body, same numbers - `static` in a header is what
   this codebase already does for lz_i32f, and for the same reason. */
/* The same rounding as the magic add, done on the bit pattern.
 *
 * This is the C statement of what lz_q8_round_pow2_arm_asm does, and it
 * exists for two reasons. One: the assembly is reachable only on ARM and
 * only under --kernel arm-asm, while every target whose floats are
 * software pays the same __aeabi_fadd - a 386 with no 387, a MIPS with
 * no FPU. Two: a hand-written kernel whose rule is stated nowhere in C
 * is a rule with one witness.
 *
 * Cost, measured on armv5te soft-float: this reaches its return in
 * about 21 instructions with no call, against the magic arm's 7 plus a
 * branch into __addsf3, whose static size in that libgcc is 100
 * instructions (0x190 bytes). What __addsf3 actually executes for THIS
 * operand pattern - one large fixed addend, one small - is not measured
 * here and is shorter than its size; the number that is certain is that
 * the call goes away.
 *
 * DECLINES, and the decline is not a domain check. Outside |v| < 2^22
 * the magic add is unreadable and this returns 0, which is the stated
 * contract. It also declines when the result is exactly +2^22: the sum
 * reaches 2^24 there, its low 23 bits are zero, and the magic arm
 * answers -2^22. This arm has to MATCH that, not be right about it, so
 * it hands those two inputs (4194303.5 and 4194303.75) back.
 *
 * Verified against the magic arm over every one of the 2^32 bit
 * patterns: 2,499,805,182 agree, 1,795,162,114 decline, 0 differ. */
LZ_MAYBE_UNUSED static int lz_q8_round_bits(uint32_t u, int k, int32_t *out) {
    uint32_t m, t;
    int32_t r;
    int e = (int)((u >> 23) & 0xFFu) + k;
    if (e >= 149) return 0;                 /* |v| >= 2^22, or inf/nan */
    e = 150 - e;                            /* bits to drop, >= 2 */
    if (e >= 25) { *out = 0; return 1; }    /* |v| < 0.5 */
    m = (u & 0x00FFFFFFu) | 0x00800000u;    /* mantissa, implicit bit set */
    r = (int32_t)(m >> e);
#if LZ_SOFTFP_FAST
    /* Round toward zero: the dropped bits are dropped, no round-up and
       no tie-to-even. This is the LZ_Q8R_BITS path (ARM's q8_round), and
       it was missed when the fast-mode pairing landed - the magic-add
       path got the RTZ truncation but this one kept round-half-to-even,
       so the ARM quantizer diverged from x86's MXCSR-truncated magic add
       by one on values whose dropped bits exceeded half. */
#else
    t = m << (32 - e);                      /* the dropped bits, aligned */
    if (t > 0x80000000u) r++;
    else if (t == 0x80000000u) r += (r & 1);  /* half to even */
#endif /* LZ_SOFTFP_FAST */
    if (u & 0x80000000u) r = -r;
    if (r == 0x400000) return 0;
    *out = r;
    return 1;
}

/* Default on only where floats are software, the same disposition
   LZ_FDIV_RECIP takes and for the same reason: on a host with an FPU the
   magic add is three cycles and the integer sequence is not obviously a
   win, so it is available there and not chosen there. Bit-identical
   either arm, which is what makes the switch checkable by cmp.

   WHO ACTUALLY GETS IT. The ARM cross build is the one target that
   passes -DLZ_NO_FPU today, and there the assembly above takes every
   call: AUTO resolves to arm-asm, so lz_debug_q8r_bits reads 0 taken on
   that tier and 131,978,654 under --kernel arm-c. So this serves ARM's
   C and ref tiers, and any no-FPU target with no assembly of its own -
   a 386 with no 387, a MIPS without an FPU, the VC++ 4.0 targets. It
   also gives the rounding rule a second statement, in C, that the
   assembly can be checked against: the two are cmp-equal over the same
   fixture (78,905,344 bytes on _armgate2/zh). */
#ifndef LZ_Q8R_BITS
#if defined(LZ_NO_FPU)
#define LZ_Q8R_BITS 1
#else
#define LZ_Q8R_BITS 0
#endif
#endif /* LZ_Q8R_BITS */

LZ_MAYBE_UNUSED static int q8_round(float qv) {
    union { float f; uint32_t u; } t;
#if defined(LZ_ARM_PRIM_ASM)
    if (g_kernel == LZ_KERNEL_ARM_ASM) {
        int32_t r;
        t.f = qv;
        if (lz_q8_round_pow2_arm_asm(t.u, 0, &r)) return (int)r;
    }
#endif /* LZ_ARM_PRIM_ASM */
#if LZ_Q8R_BITS
    {
        int32_t r;
        t.f = qv;
        if (lz_q8_round_bits(t.u, 0, &r)) { lz_debug_q8r_bits++; return (int)r; }
        lz_debug_q8r_decl++;
    }
#endif /* LZ_Q8R_BITS */
#if !defined(LZ_ARM_PRIM_ASM) && !LZ_Q8R_BITS
    /* --fastfp on sets MXCSR to round-toward-zero, but the magic add does
       not round toward zero there: the sum 1.5*2^23 + qv truncates, and
       the low-23-bit extraction turns that into FLOOR for negative qv
       (-3.7 -> -4) while ARM's bit-pattern paths truncate toward zero
       (-3.7 -> -3). cvtps2dq / the SSE2 norm kernel truncate the same
       way ARM does, so this entry is the odd one out under fastfp - a
       hardware artifact of the trick, not the rounding it advertises.
       (int)qv is a real cvttss2si truncation, bit-identical to the ARM
       paths over the whole |qv| < 2^22 contract domain (their truncation
       never reaches the +2^22 decline). */
    if (lz_fastfp()) return (int)qv;
#endif /* !LZ_ARM_PRIM_ASM && !LZ_Q8R_BITS */
    t.f = qv + 12582912.0f;                 /* 1.5 * 2^23 */
    return (int)(t.u & 0x007FFFFFu) - 0x00400000;
}

/* Per-group absolute-maximum scan (dispatched via amax table). */
uint32_t q8_amax(const float *grp, int gs);

/* Per-group scale and reciprocal. Returns fast flag. */
int q8_group_scale(const float *grp, int gs, float *sg, float *inv);

/* a / b, by integer reciprocal instead of __aeabi_fdiv's restoring loop
   - 91.7 instructions against 117 on ARMv5TE, where the census counts a
   divide as one operation and the target charges four. BIT-IDENTICAL:
   correctly rounded, ties to even, and every case it declines falls
   through to the divide itself. See the body, and LZ_FDIV_RECIP in
   ops.h for when the callers use it. */
float lz_fdiv_recip(float a, float b);

/* Float-only exp/rsqrt bodies (scalar tier independent). */
float lz_exp_float_body(float x);
float lz_rsqrt_float_body(float x);

/* Fixed-point exp/rsqrt. */
float lz_exp_fixed(float x);
float lz_rsqrt_fixed(float x);

/* softplus without libm and without double: max(x,0) plus a degree-8
   minimax polynomial for log1p(u)/u composed with this file's own exp,
   so it follows lz_scalar_mode() like every other scalar transcendental.
   ops.c's lz_softplus is the entry; LZ_SOFTPLUS_POLY=0 keeps the libm
   pair as the control arm. See ops_quant.c for the domain and for why
   the result is relatively accurate all the way into the tail. */
float lz_softplus_poly(float x);

/* Q15 sigmoid (interpolated table). */
int32_t sigmoid_q15(float x);
/* sigmoid_q15 over a run, bit-identical to calling it per element.
   Exists so the MMX interpolation cell has a caller that hands it more
   than one value - see its definition for how the three saturating
   exits are split from the interpolating path. */
void lz_sigmoid_q15_run(const float *x, int32_t *out, int n);

/* lz_exp_fixed over a run, bit-identical to calling it per element and
   charging the float-op census identically. Exists so the Q20 Taylor
   cell has a caller that hands it a block; see its definition for the
   guard split and for why the exit multiply stays scalar. Only the
   fixed-point exp has one - the float body is already a straight line
   of hardware float ops with nothing for an integer kernel to take. */
void lz_exp_fixed_run(const float *x, float *out, int n);

/* Table-only sigmoid_q15 taking the pre-scaled coordinate directly, and
   the two helpers that build it from a caller's own dequant scale. */
int32_t sigmoid_q15_t(float t);
float lz_sig_q15_t_offset(void);
void lz_sig_q15_fold(float oscale, float *k1, float *oscale2);

/* Integer-input twin of sigmoid_q15: x == v * 2^-e, and the whole
   coordinate derivation is a shift and a mask instead of a subtract,
   two multiplies and two converts. Same table, same lerp, same Q15
   result - see ops_quant.c for why it is EXACT against sigmoid_q15 and
   not merely close. e is the number of fractional bits, 0 <= e <= 50;
   past 19 it is sigmoid_q15 that rounds and this entry that does not,
   so the two stop agreeing and this one is the correct of the pair. */
int32_t sigmoid_q15_i(int32_t v, int e);

/* Folded-scale twins of the two entries above: x == v * m * 2^-s, where
   (m, s) is a constant the caller folds ONCE with lz_fold_f32 /
   lz_exp_t_fold. sigmoid_q15_i needs a power-of-two scale and lz_exp
   needs a float argument; the KDA decay gate's two constants are
   neither, which is what kept it on a float coordinate. lz_exp_t takes
   lz_exp's own table coordinate y32 == x * log2(e) * 32, so its fold
   carries that factor and the caller never sees it.

   lz_exp_t is lz_exp_fixed's twin: ask lz_scalar_mode() before choosing
   it, the way lz_sigmoid_i's callers ask lz_sig_mode().

   The folds return 1 on an exact split and 0 as a REFUSAL - on 0 the
   pair must not be used. See ops_quant.c for each entry's domain and
   for the enumerated range of every intermediate. */
int32_t sigmoid_q15_ti(int32_t v, int32_t m, int s);
float lz_exp_t(int32_t v, int32_t m, int s);
int lz_fold_f32(float a, int p, int32_t *m, int *s);
int lz_exp_t_fold(float a, int p, int32_t *m, int *s);

/* The magnitude at which the table above saturates, in sigmoid_q15_i's
   input domain: |v| >= this is exactly where sigmoid_q15 clamps. A
   producer clamping its int16 exit here is therefore not adding a clamp,
   only moving the table's own one earlier. A getter rather than a
   header #define so LZ_SIG_LO stays private to ops_quant.c, the same
   reason lz_sig_q15_t_offset is one. */
int32_t lz_sig_q15_domain(int e);

/* Sum-of-squares in integer domain (fixed-tier norms). Returns -1 to
   decline (caller runs the float path). qout, if non-NULL, receives
   the Q15 image of x the scan already computed (n shorts); eout, if
   non-NULL, receives its exponent (sc = pow2f(*eout)). Both NULL is a
   pure sum-of-squares call, same cost as before - callers that only
   want ss (l2norm, the gated norms) pass NULL,NULL and nothing is
   written past the accumulator. On decline, qout/eout are untouched. */
float norm_ss_fixed(const float *x, int n, short *qout, int *eout);

/* x87 precision control (Watcom / 32-bit x87 only). */
#if defined(__WATCOMC__) || defined(LZ_X87_FLOAT_CW)
unsigned short lz_x87_cw_get(void);
void lz_x87_cw_set(unsigned short cw);
#endif /* __WATCOMC__ || LZ_X87_FLOAT_CW */

/* Exponent extraction (bit manipulation). Shared: the GDN coefficient
   scale and the fixed norms both derive their power-of-two scale from
   it, so it belongs to neither guard - a build that calls it from
   neither subsystem simply drops it at link time. */
int expof(float x);

/* Shared constants (defined once in ops_quant.c, extern here to avoid
   excess-precision folding in other TUs). */
extern const float LZ_Q8_MIN_SCALE_F;
extern const float LZ_Q8_INV127;
extern const float LZ_Q15_INV;

/* Cached CPUID MMX bit, read inline.
 *
 * Every _mm_empty() in this engine is guarded by this, and those sit in
 * row loops - 75,080,736 evaluations over 603 tokens on kmr20, 124,512
 * per token. As one extern function reading a cached global it was that
 * many CALLS on Open Watcom, which hoists none of them (an extern
 * function may have side effects it cannot see); gcc hoists most. The
 * body is a compare, so the call cost more than the answer.
 *
 * Defined on every target, not just x86: the sites test it on ARM too,
 * where _mm_empty() is a no-op and the probe returns 0, so the whole
 * test folds away. */
extern int g_lz_has_mmx;          /* -1 until probed, then 0 or 1 */
int lz_cpu_has_mmx_probe(void);   /* ops.c; runs once */

LZ_MAYBE_UNUSED static int lz_cpu_has_mmx(void) {
    return g_lz_has_mmx >= 0 ? g_lz_has_mmx : lz_cpu_has_mmx_probe();
}

#endif /* OPS_QUANT_H */
