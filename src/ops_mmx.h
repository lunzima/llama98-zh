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

/* The pass-2 scratch block. Defined here (not in ops_kernel_p2.h, which
   ops.c #includes exactly once with no include guards of its own) so
   that ops.c (Watcom, needs it for ops_kernel_p2.h's #pragma aux
   declarations) and src/ops_mmx.c (gcc intrinsics) can both see the
   type without including ops_kernel_p2.h twice in one translation unit.
   UNCONDITIONAL - both compilers need the type regardless of LZ_MMX_TU;
   only the function declarations below are gcc-and-MMX-tier-specific. */
typedef struct {
    int16_t mul[8];      /*   0  in   m1,m2 repeated 4x (pmaddwd operand) */
    int32_t amax[4];     /*  16  out  |A| accumulator lanes, caller folds */
    int32_t rnd[4];      /*  32  in   1 << (sh-1), or 0 when sh = 0       */
    int32_t cnt[4];      /*  48  in   sh; the shift count is the low 64   */
    int16_t k128[8];     /*  64  in   128, the split's rounding addend    */
    int16_t kclp[8];     /*  80  in   32641, the lo clamp for MMX/SSE2    */
    int32_t a[32];       /*  96  mid  A, written by one kernel, read by   */
    int16_t kmin[8];     /* 224  in   -127, the same clamp for SSE1       */
} lz_p2_blk;             /*           240 bytes total                     */

#if defined(LZ_MMX_TU) && !defined(__WATCOMC__)

#define LZ_WSUM_MMX_EXTERN 1
#define LZ_P2_MMX_EXTERN   1
#define LZ_DOT_MMX_EXTERN  1
#define LZ_GDN1_MMX_EXTERN 1
/* Kept under their existing names (predates this split) rather than
   renamed to match the *_MMX_EXTERN pattern above: renaming a macro
   already tested at several call sites risks missing one, for a
   cosmetic gain the rest of the file does not need. */
#define LZ_HAVE_AMAX_MMX   1
#define LZ_HAVE_Q8R_SSE    1

/* ---- wsum: src/ops.c's LZ_WSUM_PAIR, attention scoring's dot ---------- */
void lz_wsum_pair_mmx(const int8_t *rowA, const int8_t *rowB,
                      const int16_t *coef, int32_t *acc32);

/* ---- GDN/KDA fixed-point pass 2 --------------------------------------- */
lz_p2_blk *p2_blk(void);
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

#endif /* LZ_MMX_TU && !__WATCOMC__ */

#endif /* LZ_OPS_MMX_H */
