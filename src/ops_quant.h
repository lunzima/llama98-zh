/* ops_quant.h - declarations for ops_quant.c's non-static functions.
   Quantization primitives: rounding, amax, group scale, fixed-point
   transcendental tables and helpers. */
#ifndef OPS_QUANT_H
#define OPS_QUANT_H

#include <stdint.h>
#include "ops.h"

/* 2^e by integer manipulation of the exponent field. */
float pow2f(int e);

/* Magic-number round-to-nearest (IEEE round-half-to-even). */
int q8_round(float qv);

/* Per-group absolute-maximum scan (dispatched via amax table). */
uint32_t q8_amax(const float *grp, int gs);

/* Per-group scale and reciprocal. Returns fast flag. */
int q8_group_scale(const float *grp, int gs, float *sg, float *inv);

/* Float-only exp/rsqrt bodies (scalar tier independent). */
float lz_exp_float_body(float x);
float lz_rsqrt_float_body(float x);

/* Fixed-point exp/rsqrt. */
float lz_exp_fixed(float x);
float lz_rsqrt_fixed(float x);

/* Q15 sigmoid (interpolated table). */
int32_t sigmoid_q15(float x);

/* Table-only sigmoid_q15 taking the pre-scaled coordinate directly, and
   the two helpers that build it from a caller's own dequant scale. */
int32_t sigmoid_q15_t(float t);
float lz_sig_q15_t_offset(void);
void lz_sig_q15_fold(float oscale, float *k1, float *oscale2);

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

/* Exponent extraction (bit manipulation). GDN_FIXED only. */
#if LZ_GDN_FIXED
int expof(float x);
#endif /* LZ_GDN_FIXED */

/* Shared constants (defined once in ops_quant.c, extern here to avoid
   excess-precision folding in other TUs). */
extern const float LZ_Q8_MIN_SCALE_F;
extern const float LZ_Q8_INV127;
extern const float LZ_Q15_INV;

/* Cached CPUID MMX bit. Defined in ops.c. */
int lz_cpu_has_mmx(void);

#endif /* OPS_QUANT_H */
