/* Declarations for the gcc/clang MMX-intrinsics kernels that live in
   src/ops_mmx.c - the only translation unit built with -mmmx (and -msse,
   for the SSE1-on-%mm cell). Every function declared here writes real
   %mm registers somewhere in its body, which is what put it here rather
   than in ops.c: %mm0-7 ARE the x87 stack registers, and a TU compiled
   with -mmmx can emit MMX of its own accord (a stray zeroing store, not
   tied to any explicit intrinsic in the source) anywhere -mmmx reaches,
   contaminating the x87 tag word for whatever runs after it. ops.c and
   its included headers must never carry that flag.

   Watcom is unaffected and untouched by this split: it compiles its own
   #pragma aux MMX/SSE1 kernels straight out of ops.c and the
   ops_kernel_*.h headers, exactly as before. This header's declarations
   are gcc-only and sit behind LZ_MMX_TU, a build-level macro (-D, never
   derived from __MMX__) meaning "src/ops_mmx.c is part of this link" -
   distinct from __MMX__, which only answers whether THIS translation
   unit was told to emit MMX, and on the 32-bit x87 target ops.c answers
   that "no" while ops_mmx.o still exists and is still callable.

   Every dispatch site that used to test __MMX__ to mean "is the MMX
   tier available in this build" now tests one of the macros below
   instead. Guards that decide something Watcom-specific (asm vs
   intrinsics) are untouched - this header only replaces the gcc half of
   those tests. */
#ifndef LZ_OPS_MMX_H
#define LZ_OPS_MMX_H

#include <stdint.h>
#include <stddef.h>

/* The pass-2 scratch block's type lives in src/ops_p2_blk.h, shared with
   src/ops_sse2.h and with ops.c (Watcom, needs it for
   ops_kernel_p2.h's #pragma aux declarations). UNCONDITIONAL - both
   compilers need the type regardless of LZ_MMX_TU; only the function
   declarations below are gcc-and-MMX-tier-specific. */
#include "ops_p2_blk.h"

#if defined(LZ_MMX_TU)

/* GDN1_MMX_EXTERN is unconditional (task #30 moved it first, verified
   safe on both compilers) - see ops.c's own `#if defined(__WATCOMC__)
   #elif defined(LZ_GDN1_MMX_EXTERN)` guard, checked in that order so
   Watcom's branch always wins regardless of this macro's value on that
   side. The other four stay gcc-only (!__WATCOMC__) until their own
   kernels move: ops_kernel_dot.h's dispatch table in particular does
   NOT have the same `&& !defined(__WATCOMC__)` defense
   ops_kernel_amax.h's does, so making LZ_DOT_MMX_EXTERN unconditional
   before that file's Watcom content actually moves makes Watcom's
   table silently reach for gcc-only row_*_mmx_intrin symbols - caught
   by a real compile error, not by luck, while wiring this. */
#define LZ_GDN1_MMX_EXTERN 1

/* task #30 (2/N): unconditional now too (both compilers) - Watcom's
   lz_p2_rows_mmx (this file's TU, ops_mmx.c) and lz_p2_rows_sse
   (ops_mmx_sse.c) need the SAME 16-byte-aligned scratch buffer this
   returns: the split step reads what the mul32 step wrote into it, and
   moving that hand-off across a TU boundary means passing the pointer
   explicitly rather than relying on file-local storage. Was gcc-only
   until ops_kernel_p2.h's Watcom content actually moved here - see the
   removed "collides with ops.c's own static p2_blk()" comment this
   replaced, in git history; ops.c's own copy is gone now (moved here). */
lz_p2_blk *p2_blk(void);

#if !defined(__WATCOMC__)
#define LZ_WSUM_MMX_EXTERN 1
#define LZ_P2_MMX_EXTERN   1
#define LZ_DOT_MMX_EXTERN  1
/* Kept under their existing names (predates this split) rather than
   renamed to match the *_MMX_EXTERN pattern above: renaming a macro
   already tested at several call sites risks missing one, for a
   cosmetic gain the rest of the file does not need. */
#define LZ_HAVE_AMAX_MMX   1
#define LZ_HAVE_Q8R_SSE    1
#endif

#if defined(__WATCOMC__)
/* Task #30: Watcom's own kernels now live in src/ops_mmx.c too (they
   did not before - this file used to be gcc-only). Grouped by their
   call-site shape, not by kernel, because that shape is the point:
   every one of these was called through a #pragma aux inline expansion
   before the split, so the function actually exported here is the
   widest-granularity real boundary that introduces no MORE call
   overhead than ops.c already had (see each site in ops.c for the
   specific "why this granularity" note). */
void lz_gdn_sum_gg_mmx(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg);

/* task #30 (2/N): GDN/KDA fixed-point pass 2, row granularity - one
   call per row from ops.c's gdn_p2_row_simd, in place of the zero real
   calls the pre-split #pragma-aux-inline loop had (see that function's
   Watcom branch for the accounting). pl_stride/ol_stride let the
   single-plane build pass the same fixed zero/sink buffer at every
   group (stride 0) without teaching this file about
   LZ_GDN_STATE_2PLANE - the 2-plane build passes stride 32 and a real
   per-group offset instead. */
void lz_p2_rows_mmx(const int8_t *ph_row, const int8_t *pl_row,
                    int pl_stride, const int16_t *dq,
                    const int16_t (*mul)[4], int8_t *oh_row,
                    int8_t *ol_row, int ol_stride, int *shv, int ng);

/* Real, addressable twin of the #pragma aux lz_p2_mul32_mmx_asm
   (ops_mmx.c) - needed cross-TU by ops_mmx_sse.c's lz_p2_rows_sse,
   which has no mul32 kernel of its own (SSE1 adds nothing to this
   half of the operator - see ops_kernel_p2.h's ISA inventory).
   lz_p2_rows_mmx above never calls through this: it inlines the
   pragma directly, so this wrapper adds no overhead to the tier that
   does not already need it. */
void lz_p2_mul32_mmx(const int8_t *hi, const int8_t *lo,
                     const int16_t *dq, lz_p2_blk *blk);

/* Body in src/ops_mmx_sse.c, declared here alongside its MMX-tier
   sibling rather than in a header of its own - this file already
   declares gcc's lz_p2_split32_sse below despite ITS body living in
   ops_mmx_sse.c too, so there is precedent for "the ops_mmx.c/
   ops_mmx_sse.c pair's declarations" rather than "ops_mmx.c's
   declarations" being what this header actually is. */
void lz_p2_rows_sse(const int8_t *ph_row, const int8_t *pl_row,
                    int pl_stride, const int16_t *dq,
                    const int16_t (*mul)[4], int8_t *oh_row,
                    int8_t *ol_row, int ol_stride, int *shv, int ng);

#else /* gcc: intrinsics, existing declarations */

/* ---- wsum: src/ops.c's LZ_WSUM_PAIR, attention scoring's dot ---------- */
void lz_wsum_pair_mmx(const int8_t *rowA, const int8_t *rowB,
                      const int16_t *coef, int32_t *acc32);

/* ---- GDN/KDA fixed-point pass 2 --------------------------------------- */
void lz_p2_mul32_mmx(const int8_t *hi, const int8_t *lo,
                     const int16_t *dq, lz_p2_blk *blk);
void lz_p2_split32_mmx(const lz_p2_blk *blk, int8_t *oh, int8_t *ol);
/* Named "sse" for the ISA a machine must HAVE to run it - it still
   operates on %mm (SSE1's integer additions reuse the MMX register
   file), so it belongs here exactly as much as the plain MMX kernels
   above. See ops.c's own note beside the Watcom twin. */
void lz_p2_split32_sse(const lz_p2_blk *blk, int8_t *oh, int8_t *ol);

/* ---- GDN/KDA pass 1, 4-lane table contraction -------------------------
   Same name as the Watcom #pragma aux declaration in ops.c: the two are
   never compiled into the same binary, so one external symbol serves
   both. */
void lz_gdn1_x4_asm(const int8_t *rowA, int32_t vd, const int16_t *ctab,
                    int32_t kpairs, int32_t *out8);

/* ---- 32-element dot products, one pair per quantized weight format ---
   Bodies in src/ops_kernel_dot_mmx.h (included only by ops_mmx.c),
   adjacent to their Watcom #pragma aux twins in src/ops_kernel_dot.h for
   comparison - see that file's per-format sections. */
int32_t dot32_x16_mmx(const int8_t *w, const int16_t *x);
void dot32_x16_mmx_2(const int8_t *w, const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob);
int32_t dot32_q41_mmx(const unsigned char *w, const int16_t *x);
void dot32_q41_mmx_2(const unsigned char *w, const int16_t *xa,
                     const int16_t *xb, int32_t *oa, int32_t *ob);
int32_t dot32_q61_mmx(const unsigned char *w4, const unsigned char *w2,
                      const int16_t *x);
void dot32_q61_mmx_2(const unsigned char *w4, const unsigned char *w2,
                     const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob);
int32_t dot32_t2_mmx(const unsigned char *w2, const int16_t *x);
int32_t dot32_q16_mmx(const int16_t *w, const int16_t *x);

/* ---- Q8 group-scale amax --------------------------------------------- */
unsigned lz_amax32_mmx(const float *x, int n);

/* ---- Q8 activation rounding, SSE1 tier (writes %mm via cvtps2pi) ------ */
void lz_q8round32_sse(const float *x, int8_t *o, const float *pinv);

#endif /* __WATCOMC__ */

#endif /* LZ_MMX_TU */

#endif /* LZ_OPS_MMX_H */
