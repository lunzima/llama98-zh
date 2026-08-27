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

   Every dispatch site that tests "is the MMX tier available in this
   build" tests one of the macros below instead of __MMX__. Guards that
   decide something Watcom-specific (asm vs intrinsics) are untouched -
   this header only replaces the gcc half of those tests. */
#ifndef LZ_OPS_MMX_H
#define LZ_OPS_MMX_H

#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include <stddef.h>

/* The pass-2 scratch block's type lives in src/ops_p2_blk.h, shared with
   src/ops_sse2.h and with ops.c (Watcom, needs it for
   ops_kernel_p2.h's #pragma aux declarations). UNCONDITIONAL - both
   compilers need the type regardless of LZ_MMX_TU; only the function
   declarations below are gcc-and-MMX-tier-specific. */
#include "ops_p2_blk.h"
#include "ops_matmul.h"   /* lz_row_ctx - the row kernels' parameter struct */

#if defined(LZ_MMX_TU)

/* GDN1_MMX_EXTERN is unconditional, verified safe on both compilers -
   see ops.c's own `#if defined(__WATCOMC__) #elif defined(LZ_GDN1_MMX_EXTERN)`
   guard, checked in that order so Watcom's branch always wins
   regardless of this macro's value on that side. The other four stay
   gcc-only (!__WATCOMC__): ops_kernel_dot.h's dispatch table in
   particular does NOT have the same `&& !defined(__WATCOMC__)` defense
   ops_kernel_amax.h's does, so making LZ_DOT_MMX_EXTERN unconditional
   would make Watcom's table silently reach for gcc-only row_*_mmx_intrin
   symbols - caught by a real compile error, not by luck, while wiring
   this. */
#define LZ_GDN1_MMX_EXTERN 1

/* Same unconditional treatment, and for the same reason: the SSE1
   recurrence body (lz_p2_rows_sse, ops_mmx_sse.c) is compiled for both
   toolchains because ops_mmx_sse is in every Watcom link script. Until
   this existed, km_op_present had no macro to key that cell off, which
   is why an operator with four x86 bodies read as having none. */
#define LZ_P2_SSE_EXTERN 1

/* Unconditional (both compilers): Watcom's lz_p2_rows_mmx (this file's
   TU, ops_mmx.c) and lz_p2_rows_sse (ops_mmx_sse.c) need the SAME
   16-byte-aligned scratch buffer this returns: the split step reads
   what the mul32 step wrote into it, so the pointer is passed across
   the TU boundary explicitly rather than relying on file-local storage.
   ops.c's own static copy is gone. */
lz_p2_blk *p2_blk(void);

/* Unconditional, like LZ_P2_SSE_EXTERN above and for the same reason:
   the MMX recurrence bodies (lz_gdn1_x4_asm, lz_gdn_sum_gg_mmx) are in
   the Watcom link too, and ops_gdn.c already reaches them through
   LZ_HAVE_P2_MMX_ASM. Only km_op_present was left with no macro to key
   the Watcom cell off, so an operator with four x86 bodies registered
   three. Its probe already picks the column by build class. */
#define LZ_P2_MMX_EXTERN   1

#if !defined(__WATCOMC__)
#define LZ_WSUM_MMX_EXTERN 1
#define LZ_DOT_MMX_EXTERN  1
#endif

/* The two LZ_HAVE_* flags are defined for BOTH toolchains, not just gcc
   like the _EXTERN trio above: the amax MMX and q8round SSE1 bodies live
   in src/ops_mmx.c / src/ops_mmx_sse.c on Watcom too, so "this kernel
   is in the build" is the same LZ_MMX_TU fact for either compiler.
   Kept under their existing names rather than
   renamed to match the *_MMX_EXTERN pattern: code far from here reads
   them (q8r_have_simd/lz_q8r_tier, LZ_AMAX_TAB), and a rename risks
   missing one of those call sites. */
/* BOTH toolchains, not gcc-only like the _EXTERN trio above: the
   Hadamard stage kernel has a real body on each side (intrinsics and a
   #pragma aux), and src/ops_mmx.h maps the Watcom one onto the gcc
   name, so src/fwht.c can call ONE symbol without a compiler test of
   its own. */
#define LZ_FWHT_MMX_EXTERN 1

#define LZ_HAVE_AMAX_MMX   1
#define LZ_HAVE_Q8R_SSE    1

/* Q8 activation rounding, the PACK HALF only. The multiply and the
   round are float and MMX has neither; this takes the int32 they
   produce and does the clamp and the two packs, which are MMX. The
   caller runs the float half into a scratch with x87 and takes the
   emms this kernel emits at its exit - one per 32 elements, not one
   per element, which is what separates it from a regression.
   BOTH toolchains from one macro: the wrapper has the same name and
   prototype on each side. */
#define LZ_HAVE_Q8R_PACK_MMX 1
void lz_q8round32_pack_mmx(const int32_t *q, int8_t *out);

/* norm_ss_fixed's element loop, the PACK-AND-SQUARE HALF. Same split
   and same reason as the q8round pack above: the scale and the round
   are float, the clamp and the sum of squares are not. Takes 32 int32
   the caller's x87 pass produced, writes the clamped int16 when qout
   is wanted, returns the sum of their squares, and emits the emms.
   The SSE1 tier does the whole loop instead (lz_norm_ss_sse) because
   cvtps2pi converts into %mm; MMX has no convert at all. */
#define LZ_HAVE_NORM_SS_PACK_MMX 1
lz_i64 lz_norm_ss_pack_mmx(const int32_t *q, short *qout);

/* The Q15 table interpolation that ends sigmoid_q15 and lz_exp_fixed:
   out[k] = a[k] + (((b[k] - a[k]) * frac[k]) >> 15). The LOOKUP stays
   scalar - a per-element table index needs a gather, and that is AVX2 -
   and the arithmetic around it does not.
   PRECONDITION |b - a| <= 32767: the multiply is 16-bit and this
   saturates rather than wrapping if it is broken. The callers meet it
   because a and b are adjacent entries of a 384-step table. `a` itself
   does NOT fit int16 - it reaches 32768 - so it stays int32.
   Whole groups of 32; the caller runs the tail. */
#define LZ_HAVE_LERP_Q15_MMX 1
void lz_lerp_q15_mmx(const int32_t *a, const int32_t *b,
                     const int32_t *frac, int32_t *out, int n);

/* lz_exp_fixed's Q20 Taylor, MMX tier - the OTHER half of the same
   operator's arithmetic, and a different kernel from the one above
   rather than a second caller of it: exp's tail is a Taylor series, not
   an interpolation. Every product in it is 32 x 32 -> 64 and MMX's
   widest is pmaddwd, so the body rebuilds each one from limbs that fit
   int16. Two elements a pass against the SSE2 twin's four; the caller
   runs the tail. See the body for the four decompositions and for why
   each partial sum is proved to stay inside its int32 lane. */
#define LZ_HAVE_EXP_Q20_MMX 1
void lz_exp_q20_mmx(const int32_t *tab, const int32_t *s,
                    int32_t *prod, int n);

/* The same Taylor at the SSE1 tier - HERE and not in ops_sse.h for the
   reason that file's header gives: ops_sse.h holds the SSE1 kernels
   that work on %xmm, and this one works on %mm. Body in
   src/ops_mmx_sse.c, next to the other SSE1-over-MMX cells.
   A CELL OF ITS OWN, not a second caller of lz_exp_q20_mmx: pmulhuw
   supplies the unsigned 16x16 high half MMX lacks and pshufw packs two
   int32 to int16 without packssdw's saturation, which together retire
   the u = s>>1 split the MMX body needs because s reaches 2^15. Same
   two elements a pass; fewer instructions in them. */
#define LZ_HAVE_EXP_Q20_SSE 1
void lz_exp_q20_sse(const int32_t *tab, const int32_t *s,
                    int32_t *prod, int n);

/* norm_ss_fixed's element loop, SSE1 tier. HERE and not in ops_sse.h
   for the same reason as lz_p2_split32_sse below: SSE1's convert is
   cvtps2pi, so the pack, the clamp and the multiply-add all land in
   %mm and this cell owes an emms.
   OUTSIDE the __WATCOMC__ split, like LZ_FWHT_MMX_EXTERN above: the
   wrapper has the same name and prototype on each side. Declaring it
   in the gcc arm alone left the Watcom build registering a cell it
   could not see, which kernel_matrix_gate reported as
   "nrmss/sseA registered for this build and ABSENT".
   Groups of 4; the caller runs the tail. */
#define LZ_NORM_SS_SSE_EXTERN 1
int lz_norm_ss_sse(const float *x, int n, float sc, short *qout,
                   lz_i64 *acc);

/* lz_matmul_row_sse and lz_fwht_stage_f32_sse were declared here. They
   are SSE1 on %xmm and touch no MMX register, so they moved to
   src/ops_sse.h - see its header for the %mm/%xmm distinction that
   decides which kernels stay. */

#if defined(__WATCOMC__)
/* Watcom's kernels live in src/ops_mmx.c too. Grouped by their
   call-site shape, not by kernel, because that shape is the point:
   every one of these is called through a #pragma aux inline expansion,
   so the function actually exported here is the widest-granularity
   real boundary that introduces no MORE call overhead than ops.c
   already had (see each site in ops.c for the specific "why this
   granularity" note). */
void lz_gdn_sum_gg_mmx(const int8_t *sq_gg, int kd, int vd,
                       const int16_t (*tab)[8], int32_t tcu, int32_t tcw,
                       int32_t *au_gg, int32_t *aw_gg);

/* Attention scoring's weighted sum, GROUP granularity - see ops.c's
   lz_attn_wsum_q8 (search "one call per GROUP") for why this is the
   whole multi-chunk second walk for one group, not a per-pair or
   per-chunk wrapper: T (context length) is unbounded, so either of
   those would scale calls with T instead of with ng. */
void lz_wsum_group_mmx(const int8_t *vc, int kvd, int g, int sink, int ring,
                       const int16_t *cq, int T, float *accf);
void lz_wsum_group_mmx_int(const int8_t *vc, int kvd, int g, int sink, int ring,
                           const int16_t *cq, int T, int64_t *acc64);

/* GDN/KDA fixed-point pass 2, row granularity - one call per row from
   ops.c's gdn_p2_row_simd (see that function's Watcom branch for the
   accounting). pl_stride/ol_stride let the single-plane build pass the
   same fixed zero/sink buffer at every group (stride 0) without
   teaching this file about LZ_GDN_STATE_2PLANE - the 2-plane build
   passes stride 32 and a real per-group offset instead. */
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
   declarations" being what this header actually is.

   LZ_P2_SSE_EXTERN, which lets a cell for this kernel be probed, is
   NOT here - this point is inside the __WATCOMC__ arm that opens well
   above, so a macro defined here is invisible to gcc. It sits with
   LZ_GDN1_MMX_EXTERN instead, which is unconditional. */
void lz_p2_rows_sse(const int8_t *ph_row, const int8_t *pl_row,
                    int pl_stride, const int16_t *dq,
                    const int16_t (*mul)[4], int8_t *oh_row,
                    int8_t *ol_row, int ol_stride, int *shv, int ng);

/* Q8 group-scale amax, MMX tier. lz_amax32_mmx_w is the thin wrapper
   LZ_AMAX_TAB (src/ops_kernel_amax.h) needs a real address for - a
   #pragma aux body cannot be taken by pointer. */
unsigned lz_amax32_mmx_w(const float *x, int n);

/* The three norm _w wrappers were declared here. They are SSE1 on
   %xmm - no `8087` in any __modify list - so they moved to
   src/ops_sse.h with the rest of that tier. */

/* Q8 activation rounding, SSE1+MMX tier, GROUP granularity - the whole
   gs/32-chunk loop (lz_q8round_group_sse) or two-plane loop pair plus
   the residual arithmetic between them (lz_q8round_2p_group_sse), for
   ops.c's lz_quantize_q8 / lz_gdn_quantize_2p respectively. Bodies in
   src/ops_mmx_sse.c (cvtps2pi writes %mm registers, needs genuine SSE1
   for the float side - same file as lz_p2_split32_sse). res is
   caller-owned scratch (ops.c's static float res[LZ_GDN_MAX_VD]), sized
   once there. */
void lz_q8round_group_sse(const float *grp, int8_t *out, int gs,
                          const float *pinv);
void lz_q8round_2p_group_sse(const float *grp, int8_t *ho, int8_t *lw,
                             int gs, const float *pinv,
                             const float *plo_mul, float *res);

/* Q8_0 matmul, MMX tier. row_q8_mmx_asm is the real call boundary
   (LZ_ROW_Q8, src/ops.c) - see that table's own comment. dot32_x16_mmx
   also needs to be visible here: src/ops.c's LZ_ATTN_DOT32 macro
   (attention scoring, live by default - LZ_ATTN_FIXED is 3 in ops.h)
   calls it directly across the TU boundary. lz_dot128_x16_mmx_w exists
   only for src/ops_sse2.c's -DLZ_SSE2_GROUP=0 A/B control build - see
   that file's row_q8_sse2_asm. */
void row_q8_mmx_asm(const lz_row_ctx *c);
int32_t dot32_x16_mmx(const int8_t *w, const int16_t *x);
void lz_dot128_x16_mmx_w(const int8_t *w, const int16_t *x, int32_t *out4);

/* Q4_1 matmul, MMX tier. row_q41_mmx_asm is the real call boundary
   (LZ_ROW_Q41, src/ops.c). dot32_q41_mmx stays static (no
   LZ_ATTN_DOT32-style cross-TU caller for this format).
   lz_dot128_q41_mmx_w exists only for src/ops_sse2.c's
   -DLZ_SSE2_GROUP=0 A/B control build. */
void row_q41_mmx_asm(const lz_row_ctx *c);
void lz_dot128_q41_mmx_w(const unsigned char *w, const int16_t *x, int32_t *out4);

/* T2 matmul, MMX tier. row_t2_mmx_asm is the real call boundary
   (LZ_ROW_T2, src/ops.c). The 128-wide form is dot128_t2_mmx, declared
   above and used by the gcc row kernel; this Watcom path has no
   #pragma aux twin for it yet and runs the per-32 loop, so it needs no
   *_mmx_w wrapper - that wrapper shape exists only to let one TU take
   the address of another's #pragma aux body. */
void row_t2_mmx_asm(const lz_row_ctx *c);

/* Q6_1 matmul, MMX tier. row_q61_mmx_asm is the real call boundary
   (LZ_ROW_Q61, src/ops.c). lz_dot128_q61_mmx_w exists only for
   src/ops_sse2.c's -DLZ_SSE2_GROUP=0 A/B control build. */
void row_q61_mmx_asm(const lz_row_ctx *c);
void lz_dot128_q61_mmx_w(const unsigned char *w4, const unsigned char *w2,
                         const int16_t *x, int32_t *out4);

/* Q16_0 matmul, MMX tier. row_q16_mmx_asm is the real call boundary
   (LZ_ROW_Q16, src/ops.c). lz_dot128_q16_mmx_w exists only for
   src/ops_sse2.c's -DLZ_SSE2_GROUP=0 A/B control build. */
void row_q16_mmx_asm(const lz_row_ctx *c);
void lz_dot128_q16_mmx_w(const int16_t *w, const int16_t *x, int32_t *out4);

/* ---- Hadamard: one butterfly stage of lz_fwht_i32 --------------------
   The wrapper, not the pragma: src/fwht.c is the call site and is not
   this TU, so it cannot see a #pragma aux body. src/ops_mmx.c defines
   this as a real function around it, the same way lz_dot128_q16_mmx_w
   wraps lz_dot128_q16_asm. */
void lz_fwht_stage_mmx(int32_t *y, int n, int len);

#else /* gcc: intrinsics, existing declarations */

/* ---- Hadamard: one butterfly stage of lz_fwht_i32 -------------------- */
void lz_fwht_stage_mmx(int32_t *y, int n, int len);

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
   whose Watcom #pragma aux twins sit in ops_mmx.c itself - the two
   halves of each format are in the same translation unit, one behind
   __WATCOMC__ and one behind !__WATCOMC__, and never both. */
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
/* Two tokens, one ternary unpack - the t2 member of the g_pair family,
   gcc side only. Bit-identical to two dot32_t2_mmx calls; the Watcom twin
   is deliberately absent because its pmaddwd destroys the unpack register
   (see the body's header in src/ops_kernel_dot_mmx.h). */
void dot32_t2_mmx_2(const unsigned char *w2, const int16_t *xa, const int16_t *xb,
                    int32_t *oa, int32_t *ob);
int32_t dot32_q16_mmx(const int16_t *w, const int16_t *x);
/* Two tokens, one weight load - the q16_0 member of the g_pair family.
   Bit-identical to two dot32_q16_mmx calls; see the body for why the
   saving here is smaller than q8_0's. */
void dot32_q16_mmx_2(const int16_t *w, const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob);
/* T2 over a whole 128-element group: four sub-blocks, four sums, one
   pairwise fold. Bit-identical to four dot32_t2_mmx calls - see the body
   in src/ops_kernel_dot_mmx.h for why folding pairwise changes only
   which lanes meet first. */
void dot128_t2_mmx(const unsigned char *w2, const int16_t *x,
                   int32_t *out4);

/* ---- 128-element group kernels ----------------------------------------
   One call per 128 elements instead of four dot32 calls, with the mask
   constants (0x0F/0x03 and the zero register) hoisted across the four
   sub-blocks exactly as the Watcom #pragma aux twins in src/ops_mmx.c
   do. Bodies in src/ops_kernel_dot_mmx.h; bit-identical by construction
   (each sub-block's int32 dot folds and stores separately). Wired into
   the gcc row functions behind LZ_G128_GCC (ops_kernel_shared.h). */
void dot128_x16_mmx(const int8_t *w, const int16_t *x, int32_t *out4);
void dot128_q16_mmx(const int16_t *w, const int16_t *x, int32_t *out4);
void dot128_q41_mmx(const unsigned char *w, const int16_t *x, int32_t *out4);
void dot128_q61_mmx(const unsigned char *w4, const unsigned char *w2,
                    const int16_t *x, int32_t *out4);

/* ---- Q8 group-scale amax --------------------------------------------- */
unsigned lz_amax32_mmx(const float *x, int n);

/* ---- Q8 activation rounding, SSE1 tier (writes %mm via cvtps2pi) ------ */
void lz_q8round32_sse(const float *x, int8_t *o, const float *pinv);

#endif /* __WATCOMC__ */

#endif /* LZ_MMX_TU */

#endif /* LZ_OPS_MMX_H */
