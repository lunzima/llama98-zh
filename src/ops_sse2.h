/* Declarations for the SSE2 kernels that live in src/ops_sse2.c - the
   translation unit built with -msse2 (and -march=i686, since real SSE2
   hardware in the target family, P4 Northwood and Pentium M, is i686+
   - see that file's header). For gcc this split is code structure, not
   a correctness fix the way src/ops_mmx.c's was: %xmm does not alias
   the x87 stack, so nothing here is implicated in the MMX/x87
   register-file aliasing bug. For Watcom it IS the same
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

#include "lz_int.h"   /* lz_i64 and the width types.
                         Replaces <stdint.h>, which Visual C++ 4.0
                         does not have. */

#include "ops_p2_blk.h"
#include "ops_matmul.h"   /* lz_row_ctx - the row kernels' parameter struct */

#if defined(LZ_SSE2_TU)

#if !defined(__WATCOMC__)
#define LZ_ROW_SSE2_EXTERN  1
#define LZ_ROPE_SSE2_EXTERN 1
#endif

/* Unconditional, like LZ_P2_MMX_EXTERN and LZ_P2_SSE_EXTERN: the SSE2
   recurrence pair is in the Watcom link too, and ops_gdn.c already
   reaches it through LZ_HAVE_P2_SSE2_ASM. Leaving it gcc-only left
   km_op_present with no macro for the Watcom cell, which is why an
   operator with a body on both sides registered one. Callers must now
   pick the column by build class - the macro is no longer the split. */
#define LZ_P2_SSE2_EXTERN   1

/* epi's int32 x int16 -> int64 accumulation. OUTSIDE the trio above,
   because both toolchains have a body: intrinsics on gcc, a #pragma aux
   with its own register map on Watcom, both in src/ops_sse2.c. Anything
   keyed off this macro must therefore pick its column by build class -
   naming the gcc one unconditionally would report it present in a
   Watcom build. */
#define LZ_EPI_SSE2_EXTERN  1
lz_i64 lz_epi_mac_i16_sse2(const int32_t *a, const int16_t *m, int n);

/* The two LZ_HAVE_* flags are defined for BOTH toolchains, not just gcc
   like the _EXTERN trio above: the amax SSE2 and q8round SSE2 bodies
   live in src/ops_sse2.c on Watcom too, so "this kernel is in the
   build" is the same LZ_SSE2_TU fact for either compiler. Kept under
   their existing names rather than renamed to
   match the *_SSE2_EXTERN pattern: code far from here reads them
   (q8r_have_simd/lz_q8r_tier, LZ_AMAX_TAB), same reason ops_mmx.h keeps
   LZ_HAVE_AMAX_MMX/LZ_HAVE_Q8R_SSE under their names. */
/* BOTH toolchains, like LZ_FWHT_MMX_EXTERN in src/ops_mmx.h and for the
   same reason: the Hadamard stage kernel has a real body on each side
   and src/fwht.c names one symbol. */
#define LZ_FWHT_SSE2_EXTERN 1

/* Declared outside the __WATCOMC__ split below, unlike everything else
   here: the wrapper has the same name and prototype in both branches
   (Watcom's wraps a #pragma aux, gcc's is the intrinsics body), so one
   declaration serves both. */
void lz_fwht_stage_sse2(int32_t *y, int n, int len);

/* norm_ss_fixed's element loop. Consumes whole groups of 8 and returns
   how many it took; the caller runs the remainder in C. qout may be
   NULL. See the body for why cvtps2dq is q8_round rather than an
   approximation of it. */
#define LZ_NORM_SS_SSE2_EXTERN 1
int lz_norm_ss_sse2(const float *x, int n, float sc, short *qout,
                    lz_i64 *acc);

/* Attention's two fixed-point MACs, SSE2 tier. Outside the __WATCOMC__
   split for the same reason lz_fwht_stage_sse2 is: one name and one
   prototype per kernel, a #pragma aux wrapper on one side and the
   intrinsics body on the other.

   coef points at FOUR int16 even though this kernel reads two: the MMX
   twin needs the pair duplicated to fill its 64-bit constant, and one
   signature for both tiers is worth more than two unread halfwords. */
#define LZ_ATTN_SSE2_EXTERN 1
int32_t lz_dot32_x16_sse2(const int8_t *w, const int16_t *x);
void lz_wsum_pair_sse2(const int8_t *rowA, const int8_t *rowB,
                       const int16_t *coef, int32_t *acc32);

#define LZ_HAVE_AMAX_SSE2  1
#define LZ_HAVE_Q8R_SIMD   1

/* Attention wsum's chunk fold - accf[d] += (float)acc32[d], 32 wide.
   BOTH toolchains, and the gcc half is written rather than left to the
   optimizer on purpose: gcc -O2 does vectorize the scalar loop today
   (19 cvtdq2ps in ops_gdn.o on x86-64), but an auto-vectorization is
   not a cell. It has no name for the kernel matrix to register, no
   symbol for a cross-toolchain comparison to reach, and it disappears
   silently on a compiler or flag change. Watcom emits none of it and
   runs lz_i32f's x87 split form instead, six operations an element. */
#define LZ_HAVE_I32FACC_SSE2 1

/* The Q15 table interpolation, SSE2 tier - eight elements a pass
   against the MMX cell's four, and no emms because every instruction is
   the %xmm form. Precondition |b - a| <= 32767, same as the MMX twin
   in src/ops_mmx.c, which carries the derivation. Declared for both
   arms under one name; the Watcom body is a #pragma aux behind a
   wrapper, because a pragma expands at its call site and has no
   address. */
/* lz_exp_fixed's Q20 Taylor, SSE2 tier. Four 32x32->64 multiplies -
   LN2*s, s*s, LN2SQ*(s*s) and tab*cq - every one a pmuludq, every
   operand non-negative. The regrouping of the second Taylor term is
   exact: s <= 2^15 so s*s fits int32, and integer multiplication is
   associative. Whole groups of 4; the caller runs the tail. */
#define LZ_HAVE_EXP_Q20_SIMD 1
void lz_exp_q20_simd(const int32_t *tab, const int32_t *s,
                     int32_t *prod, int n);

#define LZ_HAVE_LERP_Q15_SIMD 1
void lz_lerp_q15_simd(const int32_t *a, const int32_t *b,
                      const int32_t *frac, int32_t *out, int n);

/* Declared for both arms under one name. On Watcom the body is a
   #pragma aux, which expands at its CALL SITE and so cannot be reached
   from another translation unit - lz_i32f_acc32_simd is the real
   function in src/ops_sse2.c that wraps it, the same shape the _w
   wrappers have for ops_kernel_norm.h's tables. */
void lz_i32f_acc32_simd(float *accf, const int32_t *acc32);

#if defined(__WATCOMC__)

/* GDN/KDA fixed-point pass 2, SSE2 tier, row granularity.
   Self-contained unlike lz_p2_rows_sse (ops_mmx_sse.c): SSE2 has full
   content for BOTH mul32 and split32 (see ops_kernel_p2.h's ISA
   inventory), so this tier needs no cross-TU call at all - one call
   per row from ops.c's gdn_p2_row_simd, and nothing more. */
void lz_p2_rows_sse2(const int8_t *ph_row, const int8_t *pl_row,
                     int pl_stride, const int16_t *dq,
                     const int16_t (*mul)[4], int8_t *oh_row,
                     int8_t *ol_row, int ol_stride, int *shv, int ng);

/* Q8 group-scale amax, SSE2 tier. lz_amax32_sse2_w is the thin wrapper
   LZ_AMAX_TAB (src/ops_kernel_amax.h) needs a real address for - a
   #pragma aux body cannot be taken by pointer. */
unsigned lz_amax32_sse2_w(const float *x, int n);

/* Q8 activation rounding, SSE2 tier, GROUP granularity - same shape as
   src/ops_mmx.h's SSE1+MMX-tier twins, no emms needed here (pure xmm,
   no %mm register touched). */
void lz_q8round_group_sse2(const float *grp, int8_t *out, int gs,
                           const float *pinv);
void lz_q8round_2p_group_sse2(const float *grp, int8_t *ho, int8_t *lw,
                              int gs, const float *pinv,
                              const float *plo_mul, float *res);

/* Q8_0 matmul, SSE2 tier. row_q8_sse2_asm is the real call boundary
   (LZ_ROW_Q8, src/ops.c) - see that table's own comment. */
void row_q8_sse2_asm(const lz_row_ctx *c);

/* Q4_1 matmul, SSE2 tier. row_q41_sse2_asm is the real call boundary
   (LZ_ROW_Q41, src/ops.c). */
void row_q41_sse2_asm(const lz_row_ctx *c);

/* T2 matmul, SSE2 tier. row_t2_sse2_asm is the real call boundary
   (LZ_ROW_T2, src/ops.c). */
void row_t2_sse2_asm(const lz_row_ctx *c);

/* Q6_1 matmul, SSE2 tier. row_q61_sse2_asm is the real call boundary
   (LZ_ROW_Q61, src/ops.c). */
void row_q61_sse2_asm(const lz_row_ctx *c);

/* Q16_0 matmul, SSE2 tier. row_q16_sse2_asm is the real call boundary
   (LZ_ROW_Q16, src/ops.c). */
void row_q16_sse2_asm(const lz_row_ctx *c);

#else /* gcc: intrinsics */

/* ---- matmul row kernels, one per quantized weight format -------------
   Each of these - not a leaf dot32_* helper - is the row kernel this
   file owns: unlike the MMX row functions (pure orchestration calling
   an extern dot32_*_mmx leaf), these fold four 32-element partial sums
   with fold4_sse2 and touch %xmm directly in the row loop itself.
   Bodies in src/ops_sse2.c; the part32_ family and fold4_sse2 itself
   are in src/ops_kernel_dot_sse2.h, included only by that file. */
void row_q8_sse2_intrin(const lz_row_ctx *c);
void row_q41_sse2_intrin(const lz_row_ctx *c);
void row_t2_sse2_intrin(const lz_row_ctx *c);
void row_q61_sse2_intrin(const lz_row_ctx *c);
void row_q16_sse2_intrin(const lz_row_ctx *c);

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
