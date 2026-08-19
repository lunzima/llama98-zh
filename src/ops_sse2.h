/* Declarations for the SSE2 kernels that live in src/ops_sse2.c - the
   translation unit built with -msse2 (and -march=i686, since real SSE2
   hardware in the target family, P4 Northwood and Pentium M, is i686+
   - see that file's header). For gcc this split is code structure, not
   a correctness fix the way src/ops_mmx.c's was: %xmm does not alias
   the x87 stack, so nothing here is implicated in the MMX/x87
   register-file aliasing bug. For Watcom (task #30, 2/N) it IS the same
   reason as ops_mmx.c/ops_mmx_sse.c: -Nr/-Ns is a hard ceiling its own
   .686 directives cannot exceed. See src/ops_mmx.h for the MMX/SSE1
   half.

   Independent of src/ops_mmx.h on purpose - neither includes the other.
   The one type they share, lz_p2_blk, lives in src/ops_p2_blk.h instead.

   Guarded by LZ_SSE2_TU, a build-level macro (-D, never derived from
   __SSE2__) meaning "src/ops_sse2.c is part of this link" - distinct
   from __SSE2__, which only answers whether THIS translation unit was
   told to emit SSE2. On the 32-bit x87 target, ops.c is built with
   -mno-sse2 and answers "no" while ops_sse2.o still exists and is still
   callable - exactly the LZ_MMX_TU precedent, one ISA tier up. */
#ifndef LZ_OPS_SSE2_H
#define LZ_OPS_SSE2_H

#include <stdint.h>

#include "ops_p2_blk.h"

#if defined(LZ_SSE2_TU)

#if !defined(__WATCOMC__)
#define LZ_ROW_SSE2_EXTERN  1
#define LZ_P2_SSE2_EXTERN   1
#define LZ_ROPE_SSE2_EXTERN 1
/* Kept under their existing names (predate this split, and are read by
   code far from here - q8r_have_simd/lz_q8r_tier, LZ_AMAX_TAB) rather
   than renamed to match the *_SSE2_EXTERN pattern above, same reason
   ops_mmx.h keeps LZ_HAVE_AMAX_MMX/LZ_HAVE_Q8R_SSE under their names. */
#define LZ_HAVE_AMAX_SSE2  1
#define LZ_HAVE_Q8R_SIMD   1
#endif

#if defined(__WATCOMC__)
/* task #30 (2/N): GDN/KDA fixed-point pass 2, SSE2 tier, row
   granularity. Self-contained unlike lz_p2_rows_sse (ops_mmx_sse.c):
   SSE2 has full content for BOTH mul32 and split32 (see
   ops_kernel_p2.h's ISA inventory), so this tier needs no cross-TU call
   at all - one call per row from ops.c's gdn_p2_row_simd, in place of
   the zero real calls the pre-split #pragma-aux-inline loop had, and
   nothing more. */
void lz_p2_rows_sse2(const int8_t *ph_row, const int8_t *pl_row,
                     int pl_stride, const int16_t *dq,
                     const int16_t (*mul)[4], int8_t *oh_row,
                     int8_t *ol_row, int ol_stride, int *shv, int ng);

/* task #30 (4/N): Q8 group-scale amax, SSE2 tier. lz_amax32_sse2_w is
   the thin wrapper LZ_AMAX_TAB (src/ops_kernel_amax.h) needs a real
   address for - a #pragma aux body cannot be taken by pointer. */
unsigned lz_amax32_sse2_w(const float *x, int n);

/* task #30 (4/N): Q8 activation rounding, SSE2 tier, GROUP granularity
   - same shape as src/ops_mmx.h's SSE1+MMX-tier twins, no emms needed
   here (pure xmm, no %mm register touched). */
void lz_q8round_group_sse2(const float *grp, int8_t *out, int gs,
                           const float *pinv);
void lz_q8round_2p_group_sse2(const float *grp, int8_t *ho, int8_t *lw,
                              int gs, const float *pinv,
                              const float *plo_mul, float *res);

/* task #30 (5/N): Q8_0 matmul, SSE2 tier. row_q8_sse2_asm is ALREADY
   the real call boundary (LZ_ROW_Q8, src/ops.c) - see that table's own
   comment. */
void row_q8_sse2_asm(int32_t *acc32, const void *w4, const void *w2,
                     const int16_t *xw, int nb, int nt, int in_dim,
                     const void *pf_end, const void *pf_end2);

/* task #30 (6/N): Q4_1 matmul, SSE2 tier. row_q41_sse2_asm is ALREADY
   the real call boundary (LZ_ROW_Q41, src/ops.c). */
void row_q41_sse2_asm(int32_t *acc32, const void *w4, const void *w2,
                      const int16_t *xw, int nb, int nt, int in_dim,
                      const void *pf_end, const void *pf_end2);

/* task #30 (7/N): T2 matmul, SSE2 tier. row_t2_sse2_asm is ALREADY
   the real call boundary (LZ_ROW_T2, src/ops.c). */
void row_t2_sse2_asm(int32_t *acc32, const void *w4, const void *w2v,
                     const int16_t *xw, int nb, int nt, int in_dim,
                     const void *pf_end, const void *pf_end2);

/* task #30 (8/N): Q6_1 matmul, SSE2 tier. row_q61_sse2_asm is ALREADY
   the real call boundary (LZ_ROW_Q61, src/ops.c). */
void row_q61_sse2_asm(int32_t *acc32, const void *w4, const void *w2v,
                      const int16_t *xw, int nb, int nt, int in_dim,
                      const void *pf_end, const void *pf_end2);

/* task #30 (9/N): Q16_0 matmul, SSE2 tier. row_q16_sse2_asm is ALREADY
   the real call boundary (LZ_ROW_Q16, src/ops.c). */
void row_q16_sse2_asm(int32_t *acc32, const void *w4, const void *w2,
                      const int16_t *xw, int nb, int nt, int in_dim,
                      const void *pf_end, const void *pf_end2);

#else /* gcc: intrinsics */

/* ---- matmul row kernels, one per quantized weight format -------------
   Each of these, not a leaf dot32_* helper, is what the split moves:
   unlike the MMX row functions (pure orchestration calling an extern
   dot32_*_mmx leaf), these fold four 32-element partial sums with
   fold4_sse2 and touch %xmm directly in the row loop itself. Bodies
   in src/ops_sse2.c; the part32_ family and fold4_sse2 itself are in
   src/ops_kernel_dot_sse2.h, included only by that file. */
void row_q8_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2);
void row_q41_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2);
void row_t2_sse2_intrin(int32_t *acc32, const void *w4, const void *w2v,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2);
void row_q61_sse2_intrin(int32_t *acc32, const void *w4, const void *w2v,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2);
void row_q16_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2);

/* ---- GDN/KDA fixed-point pass 2 --------------------------------------- */
void lz_p2_mul32_sse2(const int8_t *hi, const int8_t *lo,
                      const int16_t *dq, lz_p2_blk *blk);
void lz_p2_split32_sse2(const lz_p2_blk *blk, int8_t *oh, int8_t *ol);

/* ---- RoPE -------------------------------------------------------------- */
void lz_rope_sse2(float *v, int n_heads, int head_dim,
                  int rotary_dim, int pos, const float *cs);

/* ---- Q8 group-scale amax --------------------------------------------- */
unsigned lz_amax32_sse2(const float *x, int n);

/* ---- Q8 activation rounding, SSE2 tier -------------------------------- */
void lz_q8round32_simd(const float *x, int8_t *o, const float *pinv);

#endif /* __WATCOMC__ */

#endif /* LZ_SSE2_TU */

#endif /* LZ_OPS_SSE2_H */
