/* Row-kernel prototypes for the gcc-only AVX2 tier. Guarded the same
   way src/ops_sse2.h guards LZ_ROW_SSE2_EXTERN: LZ_AVX2_TU is a
   build-level macro (-D, set only when src/ops_avx2.c is part of this
   link), never derived from a predefined compiler macro. AVX2 is
   gcc-only by design (see the AVX2 tier design doc) - this file is
   never included under __WATCOMC__, and there is deliberately no
   *_avx2_asm counterpart to any of these. */
#ifndef LZ_OPS_AVX2_H
#define LZ_OPS_AVX2_H

/* "This link carries the AVX2 kernels": one authority, defined 0 or 1 in
   every build rather than only inside the guard below, because
   lz_kernel_select reads it at run time and an undefined name there
   would be a compile error instead of a working clamp.

   LZ_AVX2_TU is set for every gcc TU and src/ops_avx2.c is linked into
   every gcc target, but nothing forces the two to travel together -
   several gates build with -DLZ_MMX_TU -DLZ_SSE2_TU and no AVX2. The
   registry (a cell must not claim a class this binary does not
   implement) and the selector both need the answer. */
#if defined(LZ_AVX2_TU) && !defined(__WATCOMC__)
#define LZ_HAVE_AVX2_TU 1
#else
#define LZ_HAVE_AVX2_TU 0
#endif

#if LZ_HAVE_AVX2_TU
#define LZ_ROW_AVX2_EXTERN 1

#include "ops_matmul.h"
#include "ops_p2_blk.h"   /* lz_p2_blk - the pass-2 scratch block */

void row_q8_avx2_intrin(const lz_row_ctx *c);
void row_q41_avx2_intrin(const lz_row_ctx *c);
void row_q61_avx2_intrin(const lz_row_ctx *c);
void row_q16_avx2_intrin(const lz_row_ctx *c);
/* The ternary format. Its 2-bit decode is an x86-64 host's work like any
   other format's: the "it is kunkun-ce/ARM's format" argument in the
   registry was a statement about who USES it, not about what this ISA
   can express, and the matrix's own standard is the instruction list. */
void row_t2_avx2_intrin(const lz_row_ctx *c);

#define LZ_HAVE_AMAX_AVX2 1
unsigned lz_amax32_avx2(const float *x, int n);

/* norm_ss_fixed's element loop (km unit "nrmss"). Contract is
   lz_norm_ss_sse2's: consumes whole groups, writes the clamped int16
   quantisation to qout when it is non-NULL, accumulates the sum of their
   squares into *acc, and returns how many elements it took so the caller
   can run the tail. Integer and therefore exact - any grouping of the
   squares gives the same int64. */
#define LZ_HAVE_NORM_SS_AVX2 1
int lz_norm_ss_avx2(const float *x, int n, float sc, short *qout,
                    lz_i64 *acc);

#define LZ_HAVE_NORM_AVX2 1
void lz_rmsnorm_out_avx2(float *o, const float *x, const float *w,
                         int n4, const float *k2);
void lz_vmax_avx2(const float *x, int n4, float *pmax);
void lz_vscale_avx2(float *x, int n4, const float *pk);

#define LZ_EPI_AVX2_EXTERN 1
lz_i64 lz_epi_mac_i16_avx2(const int32_t *a, const int16_t *m, int n);

#define LZ_FWHT_AVX2_EXTERN 1
void lz_fwht_stage_avx2(int32_t *y, int n, int len);

/* The FLOAT Hadamard's stage kernel (km unit "fwhtf"). Butterfly only -
   one add and one sub per element pair, no reduction anywhere - so the
   8-wide body is bit-identical to the 4-wide one by construction, not
   by an argument about lane order. The caller runs len 1..4 in C: the
   inner loop steps 8, so len must be a multiple of 8 for this to cover
   a block exactly. */
#define LZ_FWHT_F32_AVX2_EXTERN 1
void lz_fwht_stage_f32_avx2(float *y, int n, int len);

#define LZ_ROPE_AVX2_EXTERN 1
void lz_rope_avx2(float *v, int n_heads, int head_dim, int rotary_dim,
                  int pos, const float *cs);

#define LZ_HAVE_P2_MUL32_AVX2 1
void lz_p2_mul32_avx2(const int8_t *hi, const int8_t *lo,
                      const int16_t *dq, lz_p2_blk *blk);

#define LZ_HAVE_P2_SPLIT32_AVX2 1
void lz_p2_split32_avx2(const lz_p2_blk *blk, int8_t *oh, int8_t *ol);

#define LZ_HAVE_Q8R_AVX2 1
void lz_q8round32_avx2(const float *x, int8_t *o, const float *pinv);
/* lz_q8round32_avx2_compiled_in_quant/_gdn are declared in
   src/ops_quant.h and src/ops_gdn.h, not here - this operator has TWO
   dispatch translation units and each proves its own #include of this
   header, so neither answer can live in the file that defines the macro
   unconditionally. */

/* Attention wsum's row-pair kernel (km unit "wsump"). Contract is
   lz_wsum_pair_sse2's, element for element: acc32[i] += rowA[i]*coef[0]
   + rowB[i]*coef[1] over the 32 lanes of one group. Integer, so exact
   under any lane grouping - the only thing this has to get right is
   WHICH lane lands where.

   The macro covers BOTH attention MACs, the same way the ARM side's
   LZ_ARM_ASM_EXTERN covers atdot and wsump together: atdot's AVX2 cell
   is q8_0's leaf reused (lz_dot32_x16_avx2 below), not a second body. */
#define LZ_ATTN_AVX2_EXTERN 1
void lz_wsum_pair_avx2(const int8_t *rowA, const int8_t *rowB,
                       const int16_t *coef, int32_t *acc32);
int32_t lz_dot32_x16_avx2(const int8_t *w, const int16_t *x);

/* The int32->float chunk fold (km unit "i32f"). Contract is
   lz_i32f_acc32_simd's: accf[d] += (float)acc32[d] for d in 0..31.
   cvtepi32_ps is the correctly rounded conversion lz_i32f's split form
   was measured equal to (ops_kernel_shared.h's LZ_I32F_ACC32 note), so
   widening the same conversion from 4 to 8 lanes changes no value. */
#define LZ_HAVE_I32FACC_AVX2 1
void lz_i32f_acc32_avx2(float *accf, const int32_t *acc32);

/* The Q15 table interpolation that ends sigmoid_q15 (km unit "sigq").
   Contract and precondition are lz_lerp_q15_mmx's: out[k] = a[k] +
   (((b[k]-a[k]) * frac[k]) >> 15) with |b-a| <= 32767 and frac in
   [0, 32767], so the 32-bit product here is the same value the SSE2
   twin reconstructs from its 16-bit mullo/mulhi pair - not a wider
   arithmetic, the same arithmetic spelled in one instruction.

   Outside the precondition they differ by 1: |b - a| above 32767 makes
   the SSE2 body's packs_epi32 saturate where this one multiplies the
   true value. No caller leaves it - frac is in [0, 32767] and a/b are
   adjacent table entries - but a probe driving that point sees this
   kernel match the contract rather than the tier it replaces. */
#define LZ_HAVE_LERP_Q15_AVX2 1
void lz_lerp_q15_avx2(const int32_t *a, const int32_t *b,
                      const int32_t *frac, int32_t *out, int n);

/* lz_exp_fixed's Q20 Taylor over a run (km unit "expfx"). Contract is
   lz_exp_q20_simd's: prod[k] = (tab[k]*cq + 2^19) >> 20 with
   cq = 2^20 + ((726817*s + 2^19) >> 20) + ((251906*s*s + 2^39+2^60) >> 40),
   all operands non-negative (ops_quant.c's own "s is non-negative and
   at most 2^15" note), which is what lets the widening multiplies be
   unsigned and the shifts logical on both tiers. */
#define LZ_HAVE_EXP_Q20_AVX2 1
void lz_exp_q20_avx2(const int32_t *tab, const int32_t *s,
                     int32_t *prod, int n);

/* lz_matmul's F32-weight row kernel (km unit "f32mm"). Contract is
   lz_matmul_row_sse's: acc8[k] accumulates row[j+k]*x[j+k] for
   j = 0, 8, 16, ... in ascending order, and the return value is how
   many elements were consumed (in_dim & ~7). One 256-bit accumulator
   IS the caller's eight scalars, lane for lane and step for step, so
   this is bit-identical rather than a re-association - which is also
   why it must stay mul+add: a fused multiply-add would fold one
   rounding where the scalar loop does two. */
#define LZ_MATMUL_F32_AVX2_EXTERN 1
int lz_matmul_row_avx2(const float *row, const float *x, int in_dim,
                       float *acc8);

/* lz_rope_avx2_compiled_in is declared in src/ops.h, not here: its
   definition is unconditional (in src/ops_rope.c) and must have a
   prototype in scope on every build, the way
   lz_epi_avx2_compiled_in/lz_fwht_avx2_compiled_in do. */

#endif /* LZ_HAVE_AVX2_TU */

#endif /* LZ_OPS_AVX2_H */
