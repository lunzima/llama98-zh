/* The functions in src/ops_mmx.c's family that need genuine SSE1, not
   just MMX. For gcc: split out because -march=i586 stops meaning "no
   CMOV" the moment -msse is also on the command line (measured: gcc
   treats -msse as license to assume at least Pentium III / i686, and
   will use cmov even in an unrelated plain-C comparison elsewhere in
   the same translation unit - lz_amax32_mmx in ops_mmx.c had one until
   this split, with nothing to do with SSE). Real SSE1 hardware is
   i686+ regardless (SSE starts at Pentium III; no Socket-7 part has
   it), so -march=i686 here is honest, not a compromise - it is
   ops_mmx.c that must stay at -march=i586 so its purely-MMX functions
   (reachable on Pentium MMX/K6/Cyrix 6x86/WinChip, none of which have
   CMOV) do not pick up a stray one.

   Watcom NOW compiles this file too (task #30, 2/N) - same reason as
   src/ops_mmx.c: -Nr/-Ns is a hard ceiling its own .586/.686 directives
   cannot exceed, so a #pragma aux body that needs .686 codegen (SSE1)
   cannot sit in a TU pinned at the new, lower i486/i386 floor. Watcom
   has no CMOV-leakage concern of gcc's kind (a #pragma aux body emits
   exactly the instructions it lists, nothing implied by a command-line
   flag), so this split serves a different purpose per toolchain, same
   as src/ops_mmx.c's own header explains for the register-file-alias
   vs. CPU-generation-ceiling split. */
#include <stdint.h>
#include <string.h>

#include "ops_mmx.h"

#if defined(LZ_MMX_TU)

#if defined(__WATCOMC__)
#include "mmx_compat.h"
#include "ops_p2_blk.h"

/* Duplicated from ops.c's own p2_shift_of / src/ops_mmx.c's copy, same
   precedent as src/ops_mmx.c's gdn_tail_row_mmx: five lines, pure
   integer, cheaper to keep in step by comment than to cost a real call
   for. */
static int p2_shift_of(int32_t amax) {
    int sh = 0;
    while ((amax >> sh) > 32512) sh++;
    return sh;
}

/* Real, addressable twin of src/ops_mmx.c's #pragma aux
   lz_p2_mul32_mmx_asm, needed cross-TU here because SSE1 adds nothing
   to this half of the operator (see ops_kernel_p2.h's ISA inventory) -
   this tier's mul32 step is the plain MMX one, unchanged. Declared in
   src/ops_mmx.h. */

/* The SSE1 split: the MMX one (src/ops_mmx.c) with the low-plane clamp
   done by pmaxsw instead of the psubsw/paddsw pair. Everything else is
   identical, because everything else has no SSE1 form (the enumeration
   is in src/ops_kernel_p2.h's header).

   pmaxsw is an SSE1 addition that operates on MMX REGISTERS, so this is
   still a 64-bit kernel and still leaves MMX state behind - the caller's
   emms rule is unchanged. What it needs is a machine with SSE1, which
   in the target family means a PIII, a K7 Palomino or an Athlon XP; a
   PII and a K6-2 have MMX only and take the plain MMX split instead.

   Naming: the tier is called "sse" because SSE1 is what a machine must
   HAVE to run it, which is the property that decides dispatch. Half of
   the pair it belongs to (mul32) is the MMX kernel unchanged, since
   that half has no SSE1 content at all.

   Two instructions saved per 8 elements, 8 per 32, ~2% of the operator.
   "Too small" is not a reason to leave a cell empty; this is what the
   cell costs and buys, stated so the next reader can see both. */
extern void lz_p2_split32_sse_asm(const void *blk, signed char *hi_out,
                                  signed char *lo_out);
#pragma aux lz_p2_split32_sse_asm = \
    ".686" \
    "movq      mm7, [eax+224]" \
    "movq      mm0, [eax+96]" \
    "movq      mm1, [eax+104]" \
    "movq      mm2, [eax+112]" \
    "movq      mm3, [eax+120]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "pmaxsw    mm0, mm7" \
    "pmaxsw    mm2, mm7" \
    "packsswb  mm1, mm3" \
    "movq      [edx], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx], mm0" \
    \
    "movq      mm0, [eax+128]" \
    "movq      mm1, [eax+136]" \
    "movq      mm2, [eax+144]" \
    "movq      mm3, [eax+152]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "pmaxsw    mm0, mm7" \
    "pmaxsw    mm2, mm7" \
    "packsswb  mm1, mm3" \
    "movq      [edx+8], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+8], mm0" \
    \
    "movq      mm0, [eax+160]" \
    "movq      mm1, [eax+168]" \
    "movq      mm2, [eax+176]" \
    "movq      mm3, [eax+184]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "pmaxsw    mm0, mm7" \
    "pmaxsw    mm2, mm7" \
    "packsswb  mm1, mm3" \
    "movq      [edx+16], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+16], mm0" \
    \
    "movq      mm0, [eax+192]" \
    "movq      mm1, [eax+200]" \
    "movq      mm2, [eax+208]" \
    "movq      mm3, [eax+216]" \
    "paddd     mm0, [eax+32]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "psrad     mm0, [eax+48]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "packssdw  mm0, mm1" \
    "packssdw  mm2, mm3" \
    "movq      mm1, mm0" \
    "paddw     mm1, [eax+64]" \
    "psraw     mm1, 8" \
    "movq      mm3, mm2" \
    "paddw     mm3, [eax+64]" \
    "psraw     mm3, 8" \
    "movq      mm4, mm1" \
    "psllw     mm4, 8" \
    "psubw     mm0, mm4" \
    "movq      mm5, mm3" \
    "psllw     mm5, 8" \
    "psubw     mm2, mm5" \
    "pmaxsw    mm0, mm7" \
    "pmaxsw    mm2, mm7" \
    "packsswb  mm1, mm3" \
    "movq      [edx+24], mm1" \
    "packsswb  mm0, mm2" \
    "movq      [ebx+24], mm0" \
    __parm [__eax] [__edx] [__ebx] \
    __modify [8087]

/* lz_p2_rows_sse(): all of a row's groups, tier 2. One call per row,
   same accounting as src/ops_mmx.c's lz_p2_rows_mmx - see that
   function's comment. The mul32 step is a real cross-TU call (SSE1 has
   no mul32 of its own, so it has to reach src/ops_mmx.c's
   lz_p2_mul32_mmx wrapper), unlike the split step, whose pragma is
   textually inline right here: one real call per group instead of the
   zero the pre-split code had, which is what "SSE1 adds nothing to the
   mul32 half" costs this tier that the MMX-only tier does not pay.
   p2_blk() is also a cross-TU call (src/ops_mmx.c) - once per row, not
   per group, since the split step reads the SAME buffer the mul32 step
   just wrote and both tiers must agree on which buffer that is. */
void lz_p2_rows_sse(const int8_t *ph_row, const int8_t *pl_row,
                    int pl_stride, const int16_t *dq,
                    const int16_t (*mul)[4], int8_t *oh_row,
                    int8_t *ol_row, int ol_stride, int *shv, int ng) {
    lz_p2_blk *blk = p2_blk();
    int gg, j;
    for (gg = 0; gg < ng; gg++) {
        int32_t amax;
        int sh;
        const int8_t *plg = pl_row + (size_t)gg * pl_stride;
        int8_t *olg = ol_row + (size_t)gg * ol_stride;
        for (j = 0; j < 8; j += 2) {
            blk->mul[j]     = mul[gg][0];
            blk->mul[j + 1] = mul[gg][1];
        }
        lz_p2_mul32_mmx(ph_row + gg * 32, plg, dq + gg * 32, blk);
        amax = blk->amax[0];
        for (j = 1; j < 4; j++) if (blk->amax[j] > amax) amax = blk->amax[j];
        sh = p2_shift_of(amax);
        shv[gg] = sh;
        for (j = 0; j < 4; j++) {
            blk->rnd[j] = (sh > 0) ? (int32_t)(1 << (sh - 1)) : 0;
            blk->cnt[j] = (j == 0) ? (int32_t)sh : 0;
        }
        for (j = 0; j < 8; j++) {
            blk->k128[j] = 128;
            blk->kclp[j] = 32641;   /* MMX/SSE2: psubsw then paddsw */
            blk->kmin[j] = -127;    /* SSE1: one pmaxsw does the same */
        }
        lz_p2_split32_sse_asm(blk, oh_row + gg * 32, olg);
    }
}

/* ---- RMSNorm's output loop / softmax's max scan and normalize loop,
   SSE1 tier - moved here from src/ops_kernel_norm.h (task #30, 4/N).
   All three are genuine SSE1-on-xmm with no MMX register touched (no
   `8087` in any __modify list below), the same reason lz_p2_split32_sse
   above lives in this file rather than src/ops_mmx.c. Zero call-shape
   change: each _w wrapper already sat between its pragma and its
   dispatch table (src/ops_kernel_norm.h's LZ_NORMOUT_TAB/LZ_VMAX_TAB/
   LZ_VSCALE_TAB) before the move. */

extern void lz_rmsnorm_out_sse_asm(float *o, const float *x, const float *w,
                                   int n4, const float *k2);
/* n4 is the number of 4-float blocks, not elements: the caller owns the
   tail. Unaligned loads throughout - o, x and w come from xcalloc and
   from tensor slices, and 16-byte alignment is not something this engine
   can promise for any of them.
 *
 * k2 is {inv, 1.0f}. The constant is PASSED rather than materialized in
 * register, because every register recipe for 1.0f (pcmpeqd then a shift
 * pair) needs an SSE2 integer instruction on XMM, and this routine is
 * meant to be SSE1. Two loads at entry is the price of the cell being
 * genuinely SSE1 rather than SSE2 wearing its name.
 *
 * esi, not ebp: ebp is the frame pointer and Watcom rejects it outright
 * with "Illegal register modified". */
#pragma aux lz_rmsnorm_out_sse_asm =    \
    ".686"                              \
    "movss    xmm7, [esi]"              \
    "shufps   xmm7, xmm7, 0"            \
    "movss    xmm6, [esi+4]"            \
    "shufps   xmm6, xmm6, 0"            \
    "test     ecx, ecx"                 \
    "jz       N2"                       \
    "N1:"                               \
    "movups   xmm0, [edx]"              \
    "mulps    xmm0, xmm7"               \
    "movups   xmm1, [ebx]"              \
    "addps    xmm1, xmm6"               \
    "mulps    xmm0, xmm1"               \
    "movups   [eax], xmm0"              \
    "add      eax, 16"                  \
    "add      edx, 16"                  \
    "add      ebx, 16"                  \
    "dec      ecx"                      \
    "jnz      N1"                       \
    "N2:"                               \
    parm [eax] [edx] [ebx] [ecx] [esi]  \
    modify [eax ebx ecx edx esi];

extern void lz_vmax_sse_asm(const float *x, int n4, float *pmax);
/* Softmax's maximum scan. maxps is exact and, with no NaN present, its
   order does not matter - so this produces the same float the scalar
   `if (x[i] > mx) mx = x[i]` loop would. NaN would break that (maxps
   returns its second operand when unordered, the scalar compare keeps
   the first); the engine's masked-out positions are -1e30f rather than
   NaN precisely so the arithmetic stays ordered.
 *
 * IN AND OUT THROUGH pmax, rather than a float return. `value [8087]`
 * means the result is on the x87 stack, and this routine's result is in
 * xmm0 - declaring one and producing the other returns whatever was on
 * the stack. It cost a cross-compiler DIFFER on --kernel sse2 and
 * nothing else, which is the only place that mistake is visible. */
#pragma aux lz_vmax_sse_asm =           \
    ".686"                              \
    "movss    xmm0, [ebx]"              \
    "shufps   xmm0, xmm0, 0"            \
    "test     edx, edx"                 \
    "jz       M2"                       \
    "M1:"                               \
    "movups   xmm1, [eax]"              \
    "maxps    xmm0, xmm1"               \
    "add      eax, 16"                  \
    "dec      edx"                      \
    "jnz      M1"                       \
    "M2:"                               \
    "movhlps  xmm1, xmm0"               \
    "maxps    xmm0, xmm1"               \
    "movaps   xmm1, xmm0"               \
    "shufps   xmm1, xmm1, 1"            \
    "maxps    xmm0, xmm1"               \
    "movss    [ebx], xmm0"              \
    parm [eax] [edx] [ebx]              \
    modify [eax edx];

extern void lz_vscale_sse_asm(float *x, int n4, const float *pk);
/* Softmax's normalize loop: x[i] *= inv, element-wise. */
#pragma aux lz_vscale_sse_asm =         \
    ".686"                              \
    "movss    xmm7, [ebx]"              \
    "shufps   xmm7, xmm7, 0"            \
    "test     edx, edx"                 \
    "jz       V2"                       \
    "V1:"                               \
    "movups   xmm0, [eax]"              \
    "mulps    xmm0, xmm7"               \
    "movups   [eax], xmm0"              \
    "add      eax, 16"                  \
    "dec      edx"                      \
    "jnz      V1"                       \
    "V2:"                               \
    parm [eax] [edx] [ebx]              \
    modify [eax edx];

/* Wrappers, for the same reason src/ops_mmx.c's amax ones give: a
   #pragma aux body is emitted at the call site and has no address to
   put in a table. */
void lz_rmsnorm_out_sse_w(float *o, const float *x, const float *w,
                          int n4, const float *k2) {
    lz_rmsnorm_out_sse_asm(o, x, w, n4, k2);
}
void lz_vmax_sse_w(const float *x, int n4, float *pmax) {
    lz_vmax_sse_asm(x, n4, pmax);
}
void lz_vscale_sse_w(float *x, int n4, const float *pk) {
    lz_vscale_sse_asm(x, n4, pk);
}

/* ---- Q8 activation rounding, SSE1+MMX tier - moved here from
   src/ops_kernel_q8round.h (task #30, 4/N). No cvtps2dq, only cvtps2pi
   - converting 2 floats at a time into MMX registers, which is why this
   cell needs genuine SSE1 (the multiply) AND touches %mm (the convert/
   pack/clamp), the same combination lz_p2_split32_sse_asm above has. */
extern void lz_q8round32_sse_asm(const float *x, int8_t *o, const float *inv);
#pragma aux lz_q8round32_sse_asm = \
    ".686" \
    "movss    xmm7, [ecx]" \
    "shufps   xmm7, xmm7, 0" \
    "pcmpeqw  mm5, mm5" \
    "psrlw    mm5, 9" \
    "pxor     mm6, mm6" \
    "psubw    mm6, mm5" \
    "movups   xmm0, [eax]" \
    "mulps    xmm0, xmm7" \
    "movups   xmm1, [eax+16]" \
    "mulps    xmm1, xmm7" \
    "cvtps2pi mm0, xmm0" \
    "movhlps  xmm2, xmm0" \
    "cvtps2pi mm1, xmm2" \
    "cvtps2pi mm2, xmm1" \
    "movhlps  xmm3, xmm1" \
    "cvtps2pi mm3, xmm3" \
    "packssdw mm0, mm1" \
    "packssdw mm2, mm3" \
    "pminsw   mm0, mm5" \
    "pmaxsw   mm0, mm6" \
    "pminsw   mm2, mm5" \
    "pmaxsw   mm2, mm6" \
    "packsswb mm0, mm2" \
    "movq     [edx], mm0" \
    "movups   xmm0, [eax+32]" \
    "mulps    xmm0, xmm7" \
    "movups   xmm1, [eax+48]" \
    "mulps    xmm1, xmm7" \
    "cvtps2pi mm0, xmm0" \
    "movhlps  xmm2, xmm0" \
    "cvtps2pi mm1, xmm2" \
    "cvtps2pi mm2, xmm1" \
    "movhlps  xmm3, xmm1" \
    "cvtps2pi mm3, xmm3" \
    "packssdw mm0, mm1" \
    "packssdw mm2, mm3" \
    "pminsw   mm0, mm5" \
    "pmaxsw   mm0, mm6" \
    "pminsw   mm2, mm5" \
    "pmaxsw   mm2, mm6" \
    "packsswb mm0, mm2" \
    "movq     [edx+8], mm0" \
    "movups   xmm0, [eax+64]" \
    "mulps    xmm0, xmm7" \
    "movups   xmm1, [eax+80]" \
    "mulps    xmm1, xmm7" \
    "cvtps2pi mm0, xmm0" \
    "movhlps  xmm2, xmm0" \
    "cvtps2pi mm1, xmm2" \
    "cvtps2pi mm2, xmm1" \
    "movhlps  xmm3, xmm1" \
    "cvtps2pi mm3, xmm3" \
    "packssdw mm0, mm1" \
    "packssdw mm2, mm3" \
    "pminsw   mm0, mm5" \
    "pmaxsw   mm0, mm6" \
    "pminsw   mm2, mm5" \
    "pmaxsw   mm2, mm6" \
    "packsswb mm0, mm2" \
    "movq     [edx+16], mm0" \
    "movups   xmm0, [eax+96]" \
    "mulps    xmm0, xmm7" \
    "movups   xmm1, [eax+112]" \
    "mulps    xmm1, xmm7" \
    "cvtps2pi mm0, xmm0" \
    "movhlps  xmm2, xmm0" \
    "cvtps2pi mm1, xmm2" \
    "cvtps2pi mm2, xmm1" \
    "movhlps  xmm3, xmm1" \
    "cvtps2pi mm3, xmm3" \
    "packssdw mm0, mm1" \
    "packssdw mm2, mm3" \
    "pminsw   mm0, mm5" \
    "pmaxsw   mm0, mm6" \
    "pminsw   mm2, mm5" \
    "pmaxsw   mm2, mm6" \
    "packsswb mm0, mm2" \
    "movq     [edx+24], mm0" \
    __parm [__eax] [__edx] [__ecx] \
    __modify [8087]

/* lz_q8round_group_sse(): the whole gs/32-chunk loop for one group of
   src/ops.c's lz_quantize_q8, byte-for-byte the loop body that used to
   sit at that call site, now behind ONE call per group instead of the
   gs/32 real calls a naive per-sub-chunk wrap would have cost. The
   emms is the SAME one lz_quantize_q8's own comment placed here -
   moved with the loop, not removed: rule 6.6's "kernels don't emit"
   still does not hold, the next group's x87 amax/inv still follows
   immediately in ops.c. */
void lz_q8round_group_sse(const float *grp, int8_t *out, int gs,
                          const float *pinv) {
    int k;
    for (k = 0; k + 31 < gs; k += 32)
        lz_q8round32_sse_asm(grp + k, out + k, pinv);
    _mm_empty();
}

/* lz_q8round_2p_group_sse(): the whole two-plane round for one group of
   src/ops.c's lz_gdn_quantize_2p - both kernel loops AND the residual
   float arithmetic between them, byte-for-byte what used to sit at
   that call site. The residual computation is plain C float
   arithmetic; putting it in this TU is safe for Watcom the way it
   would not be for gcc under -mmmx (src/ops_mmx.c's header explains
   the difference - Watcom never emits MMX except where a #pragma aux
   says so, so mixing float C and #pragma aux kernels in one TU here
   is exactly as safe as ops.c doing the same always was). */
void lz_q8round_2p_group_sse(const float *grp, int8_t *ho, int8_t *lw,
                             int gs, const float *pinv,
                             const float *plo_mul, float *res) {
    int k;
    for (k = 0; k + 31 < gs; k += 32)
        lz_q8round32_sse_asm(grp + k, ho + k, pinv);
    _mm_empty();
    for (k = 0; k < gs; k++)
        res[k] = grp[k] * (*pinv) - (float)ho[k];
    for (k = 0; k + 31 < gs; k += 32)
        lz_q8round32_sse_asm(res + k, lw + k, plo_mul);
    _mm_empty();
}

#else /* gcc: intrinsics */

#include "mmx_compat.h"
#include <mmintrin.h>
#include <xmmintrin.h>

/* GDN/KDA fixed-point pass 2, SSE1 cell. Watcom twin:
   lz_p2_split32_sse_asm, src/ops_kernel_p2.h. Named "sse" for the ISA a
   machine must HAVE to run it - it still operates on %mm (SSE1's whole
   addition to the integer side of the MMX register file is used here
   exactly once: pmaxsw does the low plane's clamp to -127 directly,
   replacing the psubsw/paddsw pair). */
void lz_p2_split32_sse(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m64 rnd, cnt, k128, kmin;
    int o;

    memcpy(&rnd,  blk->rnd,  8);
    memcpy(&cnt,  blk->cnt,  8);
    memcpy(&k128, blk->k128, 8);
    memcpy(&kmin, blk->kmin, 8);
    for (o = 0; o < 4; o++) {
        __m64 a0, a1, a2, a3, hn0, hn1, h0, h1, l0, l1, t;

        memcpy(&a0, blk->a + o * 8,     8);
        memcpy(&a1, blk->a + o * 8 + 2, 8);
        memcpy(&a2, blk->a + o * 8 + 4, 8);
        memcpy(&a3, blk->a + o * 8 + 6, 8);
        a0 = _mm_sra_pi32(_mm_add_pi32(a0, rnd), cnt);
        a1 = _mm_sra_pi32(_mm_add_pi32(a1, rnd), cnt);
        a2 = _mm_sra_pi32(_mm_add_pi32(a2, rnd), cnt);
        a3 = _mm_sra_pi32(_mm_add_pi32(a3, rnd), cnt);
        hn0 = _mm_packs_pi32(a0, a1);
        hn1 = _mm_packs_pi32(a2, a3);
        h0 = _mm_srai_pi16(_mm_add_pi16(hn0, k128), 8);
        h1 = _mm_srai_pi16(_mm_add_pi16(hn1, k128), 8);
        l0 = _mm_sub_pi16(hn0, _mm_slli_pi16(h0, 8));
        l1 = _mm_sub_pi16(hn1, _mm_slli_pi16(h1, 8));
        l0 = _mm_max_pi16(l0, kmin);      /* the whole SSE1 tier, twice */
        l1 = _mm_max_pi16(l1, kmin);
        t = _mm_packs_pi16(h0, h1);
        memcpy(oh + o * 8, &t, 8);
        t = _mm_packs_pi16(l0, l1);
        memcpy(ol + o * 8, &t, 8);
    }
}

/* Q8 activation rounding, SSE1 tier. Watcom twin: lz_q8round32_sse_asm,
   src/ops_kernel_q8round.h. No cvtps2dq, only cvtps2pi - converting 2
   floats at a time into MMX registers. Multiply in xmm, convert and
   pack in mm, using movhlps to slide the high 2 floats down. Writes
   MMX registers, so the caller must emit one emms after the group loop
   (rule 6.6). */
void lz_q8round32_sse(const float *x, int8_t *o, const float *pinv) {
    __m128 vi = _mm_set1_ps(*pinv);
    __m64 p127 = _mm_set1_pi16(127), m127 = _mm_set1_pi16(-127);
    int b;
    for (b = 0; b < 4; b++) {
        const float *sp = x + b * 8;
        __m128 f0 = _mm_mul_ps(_mm_loadu_ps(sp), vi);
        __m128 f1 = _mm_mul_ps(_mm_loadu_ps(sp + 4), vi);
        __m64 a0 = _mm_cvtps_pi32(f0);
        __m64 a1 = _mm_cvtps_pi32(_mm_movehl_ps(f0, f0));
        __m64 a2 = _mm_cvtps_pi32(f1);
        __m64 a3 = _mm_cvtps_pi32(_mm_movehl_ps(f1, f1));
        __m64 l = _mm_packs_pi32(a0, a1);
        __m64 h = _mm_packs_pi32(a2, a3);
        __m64 r;
        l = _mm_max_pi16(_mm_min_pi16(l, p127), m127);
        h = _mm_max_pi16(_mm_min_pi16(h, p127), m127);
        r = _mm_packs_pi16(l, h);
        memcpy(o + b * 8, &r, 8);
    }
}

#endif /* __WATCOMC__ */

#endif /* LZ_MMX_TU */
