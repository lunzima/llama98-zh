/* ops_quant.h - declarations for ops_quant.c's non-static functions.
   Quantization primitives: rounding, amax, group scale, fixed-point
   transcendental tables and helpers. */
#ifndef OPS_QUANT_H
#define OPS_QUANT_H

#include <stdint.h>
#include "ops.h"
#include "ops_arm_prim.h"   /* LZ_ARM_PRIM_ASM, lz_q8_round_pow2_arm_asm */

/* The runtime-selected ISA tier, defined in ops_sched.c. Declared here
   rather than pulling in ops_kernel_shared.h, which would put lz_i32f
   and the dispatch macros into every TU that only wants to round. */
extern int g_kernel;

/* For a `static` definition in a header: two of the TUs that include
   this one (ops_matmul.c, ops_sched.c) want the other declarations and
   never round, and an unused-function warning per TU is not a price
   worth paying for an inline body. Watcom has no equivalent spelling
   and does not warn about it, so the empty expansion is correct there
   rather than merely tolerated. */
#if defined(__GNUC__)
#define LZ_MAYBE_UNUSED __attribute__((unused))
#else
#define LZ_MAYBE_UNUSED
#endif /* __GNUC__ */

/* 2^e by integer manipulation of the exponent field. */
float pow2f(int e);

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
LZ_MAYBE_UNUSED static int q8_round(float qv) {
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

/* Cached CPUID MMX bit. Defined in ops.c. */
int lz_cpu_has_mmx(void);

#endif /* OPS_QUANT_H */
