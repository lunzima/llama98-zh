/* The functions in src/ops_mmx.c's family that need genuine SSE1, not
   just MMX. For gcc: split out because -march=i586 stops meaning "no
   CMOV" the moment -msse is also on the command line (measured: gcc
   treats -msse as license to assume at least Pentium III / i686, and
   will use cmov even in an unrelated plain-C comparison elsewhere in
   the same translation unit - lz_amax32_mmx in ops_mmx.c picked one up
   that way, with nothing to do with SSE). Real SSE1 hardware is
   i686+ regardless (SSE starts at Pentium III; no Socket-7 part has
   it), so -march=i686 here is honest, not a compromise - it is
   ops_mmx.c that must stay at -march=i586 so its purely-MMX functions
   (reachable on Pentium MMX/K6/Cyrix 6x86/WinChip, none of which have
   CMOV) do not pick up a stray one.

   Watcom compiles this file too - same reason as src/ops_mmx.c:
   -Nr/-Ns is a hard ceiling its own .586/.686 directives cannot
   exceed, so a #pragma aux body that needs .686 codegen (SSE1) cannot
   sit in a TU pinned at the lower i486/i386 floor. Watcom has no
   CMOV-leakage concern of gcc's kind (a #pragma aux body emits exactly
   the instructions it lists, nothing implied by a command-line flag),
   so this split serves a different purpose per toolchain, same as
   src/ops_mmx.c's own header explains for the register-file-alias vs.
   CPU-generation-ceiling split. */
#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include <string.h>

#include "ops_mmx.h"
#include "ops_sse.h"    /* this file's own %xmm half - see that header for
                           which kernels stayed with the MMX declarations */

#if defined(LZ_MMX_TU)

#if defined(__WATCOMC__)
#include "mmx_compat.h"
#include "ops_kernel_shared.h" /* p2_shift_of - shared with ops.c and
                           the other suffixed TUs. gcc's branch below
                           needs none of it. */
#include "ops_p2_blk.h"

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
    "movq      mm0, [eax+224]" \
    "movq      mm1, [eax+96]" \
    "movq      mm2, [eax+104]" \
    "movq      mm3, [eax+112]" \
    "movq      mm4, [eax+120]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "paddd     mm4, [eax+32]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "psrad     mm4, [eax+48]" \
    "packssdw  mm1, mm2" \
    "packssdw  mm3, mm4" \
    "movq      mm2, mm1" \
    "paddw     mm2, [eax+64]" \
    "psraw     mm2, 8" \
    "movq      mm4, mm3" \
    "paddw     mm4, [eax+64]" \
    "psraw     mm4, 8" \
    "movq      mm5, mm2" \
    "psllw     mm5, 8" \
    "psubw     mm1, mm5" \
    "movq      mm6, mm4" \
    "psllw     mm6, 8" \
    "psubw     mm3, mm6" \
    "pmaxsw    mm1, mm0" \
    "pmaxsw    mm3, mm0" \
    "packsswb  mm2, mm4" \
    "movq      [edx], mm2" \
    "packsswb  mm1, mm3" \
    "movq      [ebx], mm1" \
    \
    "movq      mm1, [eax+128]" \
    "movq      mm2, [eax+136]" \
    "movq      mm3, [eax+144]" \
    "movq      mm4, [eax+152]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "paddd     mm4, [eax+32]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "psrad     mm4, [eax+48]" \
    "packssdw  mm1, mm2" \
    "packssdw  mm3, mm4" \
    "movq      mm2, mm1" \
    "paddw     mm2, [eax+64]" \
    "psraw     mm2, 8" \
    "movq      mm4, mm3" \
    "paddw     mm4, [eax+64]" \
    "psraw     mm4, 8" \
    "movq      mm5, mm2" \
    "psllw     mm5, 8" \
    "psubw     mm1, mm5" \
    "movq      mm6, mm4" \
    "psllw     mm6, 8" \
    "psubw     mm3, mm6" \
    "pmaxsw    mm1, mm0" \
    "pmaxsw    mm3, mm0" \
    "packsswb  mm2, mm4" \
    "movq      [edx+8], mm2" \
    "packsswb  mm1, mm3" \
    "movq      [ebx+8], mm1" \
    \
    "movq      mm1, [eax+160]" \
    "movq      mm2, [eax+168]" \
    "movq      mm3, [eax+176]" \
    "movq      mm4, [eax+184]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "paddd     mm4, [eax+32]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "psrad     mm4, [eax+48]" \
    "packssdw  mm1, mm2" \
    "packssdw  mm3, mm4" \
    "movq      mm2, mm1" \
    "paddw     mm2, [eax+64]" \
    "psraw     mm2, 8" \
    "movq      mm4, mm3" \
    "paddw     mm4, [eax+64]" \
    "psraw     mm4, 8" \
    "movq      mm5, mm2" \
    "psllw     mm5, 8" \
    "psubw     mm1, mm5" \
    "movq      mm6, mm4" \
    "psllw     mm6, 8" \
    "psubw     mm3, mm6" \
    "pmaxsw    mm1, mm0" \
    "pmaxsw    mm3, mm0" \
    "packsswb  mm2, mm4" \
    "movq      [edx+16], mm2" \
    "packsswb  mm1, mm3" \
    "movq      [ebx+16], mm1" \
    \
    "movq      mm1, [eax+192]" \
    "movq      mm2, [eax+200]" \
    "movq      mm3, [eax+208]" \
    "movq      mm4, [eax+216]" \
    "paddd     mm1, [eax+32]" \
    "paddd     mm2, [eax+32]" \
    "paddd     mm3, [eax+32]" \
    "paddd     mm4, [eax+32]" \
    "psrad     mm1, [eax+48]" \
    "psrad     mm2, [eax+48]" \
    "psrad     mm3, [eax+48]" \
    "psrad     mm4, [eax+48]" \
    "packssdw  mm1, mm2" \
    "packssdw  mm3, mm4" \
    "movq      mm2, mm1" \
    "paddw     mm2, [eax+64]" \
    "psraw     mm2, 8" \
    "movq      mm4, mm3" \
    "paddw     mm4, [eax+64]" \
    "psraw     mm4, 8" \
    "movq      mm5, mm2" \
    "psllw     mm5, 8" \
    "psubw     mm1, mm5" \
    "movq      mm6, mm4" \
    "psllw     mm6, 8" \
    "psubw     mm3, mm6" \
    "pmaxsw    mm1, mm0" \
    "pmaxsw    mm3, mm0" \
    "packsswb  mm2, mm4" \
    "movq      [edx+24], mm2" \
    "packsswb  mm1, mm3" \
    "movq      [ebx+24], mm1" \
    __parm [__eax] [__edx] [__ebx] \
    __modify [8087]

/* The split kernel's address, so the three tiers can be compared. */
void lz_p2_split32_sse_w(const void *blk, signed char *hi_out,
                         signed char *lo_out) {
    lz_p2_split32_sse_asm(blk, hi_out, lo_out);
}

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
   SSE1 tier - in this TU; the dispatch tables are in
   src/ops_kernel_norm.h. All three are genuine SSE1-on-xmm with no MMX
   register touched (no `8087` in any __modify list below), the same
   reason lz_p2_split32_sse above lives in this file rather than
   src/ops_mmx.c. Zero call-shape change: each _w wrapper sits between
   its pragma and its dispatch table (src/ops_kernel_norm.h's
   LZ_NORMOUT_TAB/LZ_VMAX_TAB/LZ_VSCALE_TAB). */

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
#pragma aux lz_rmsnorm_out_sse_asm = \
    ".686"                              \
    "movss     xmm0, [esi]"              \
    "shufps    xmm0, xmm0, 0"            \
    "movss     xmm1, [esi+4]"            \
    "shufps    xmm1, xmm1, 0"            \
    "test      ecx, ecx"                 \
    "jz        N2"                       \
    "N1:"                               \
    "movups    xmm2, [edx]"              \
    "movups    xmm3, [ebx]"              \
    "mulps     xmm2, xmm0"               \
    "addps     xmm3, xmm1"               \
    "mulps     xmm2, xmm3"               \
    "movups    [eax], xmm2"              \
    "add       eax, 16"                  \
    "add       edx, 16"                  \
    "add       ebx, 16"                  \
    "dec       ecx"                      \
    "jnz       N1"                       \
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
#pragma aux lz_vmax_sse_asm = \
    ".686"                              \
    "movss     xmm0, [ebx]"              \
    "shufps    xmm0, xmm0, 0"            \
    "movaps    xmm1, xmm0"               \
    "test      edx, 1"                   \
    "jz        M0"                       \
    "movups    xmm2, [eax]"              \
    "maxps     xmm0, xmm2"               \
    "add       eax, 16"                  \
    "M0:"                               \
    "shr       edx, 1"                   \
    "jz        M2"                       \
    "M1:"                               \
    "movups    xmm2, [eax]"              \
    "maxps     xmm0, xmm2"               \
    "movups    xmm2, [eax+16]"           \
    "maxps     xmm1, xmm2"               \
    "add       eax, 32"                  \
    "dec       edx"                      \
    "jnz       M1"                       \
    "M2:"                               \
    "maxps     xmm0, xmm1"               \
    "movhlps   xmm2, xmm0"               \
    "maxps     xmm0, xmm2"               \
    "movaps    xmm2, xmm0"               \
    "shufps    xmm2, xmm2, 1"            \
    "maxps     xmm0, xmm2"               \
    "movss     [ebx], xmm0"              \
    parm [eax] [edx] [ebx]              \
    modify [eax edx];

extern void lz_vscale_sse_asm(float *x, int n4, const float *pk);
/* Softmax's normalize loop: x[i] *= inv, element-wise. */
#pragma aux lz_vscale_sse_asm = \
    ".686"                              \
    "movss     xmm0, [ebx]"              \
    "shufps    xmm0, xmm0, 0"            \
    "test      edx, edx"                 \
    "jz        V2"                       \
    "V1:"                               \
    "movups    xmm1, [eax]"              \
    "mulps     xmm1, xmm0"               \
    "movups    [eax], xmm1"              \
    "add       eax, 16"                  \
    "dec       edx"                      \
    "jnz       V1"                       \
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

/* ---- Q8 activation rounding, SSE1+MMX tier - in this TU; the
   algorithm documentation is in src/ops_kernel_q8round.h. No cvtps2dq,
   only cvtps2pi - converting 2 floats at a time into MMX registers,
   which is why this cell needs genuine SSE1 (the multiply) AND touches
   %mm (the convert/pack/clamp), the same combination
   lz_p2_split32_sse_asm above has. */
extern void lz_q8round32_sse_asm(const float *x, int8_t *o, const float *inv);
#pragma aux lz_q8round32_sse_asm = \
    ".686" \
    "movss     xmm0, [ecx]" \
    "shufps    xmm0, xmm0, 0" \
    "pcmpeqw   mm0, mm0" \
    "psrlw     mm0, 9" \
    "pxor      mm1, mm1" \
    "psubw     mm1, mm0" \
    "movups    xmm1, [eax]" \
    "mulps     xmm1, xmm0" \
    "movups    xmm2, [eax+16]" \
    "mulps     xmm2, xmm0" \
    "cvtps2pi  mm2, xmm1" \
    "movhlps   xmm3, xmm1" \
    "cvtps2pi  mm3, xmm3" \
    "cvtps2pi  mm4, xmm2" \
    "movhlps   xmm4, xmm2" \
    "cvtps2pi  mm5, xmm4" \
    "packssdw  mm2, mm3" \
    "packssdw  mm4, mm5" \
    "pminsw    mm2, mm0" \
    "pmaxsw    mm2, mm1" \
    "pminsw    mm4, mm0" \
    "pmaxsw    mm4, mm1" \
    "packsswb  mm2, mm4" \
    "movq      [edx], mm2" \
    "movups    xmm1, [eax+32]" \
    "mulps     xmm1, xmm0" \
    "movups    xmm2, [eax+48]" \
    "mulps     xmm2, xmm0" \
    "cvtps2pi  mm2, xmm1" \
    "movhlps   xmm3, xmm1" \
    "cvtps2pi  mm3, xmm3" \
    "cvtps2pi  mm4, xmm2" \
    "movhlps   xmm4, xmm2" \
    "cvtps2pi  mm5, xmm4" \
    "packssdw  mm2, mm3" \
    "packssdw  mm4, mm5" \
    "pminsw    mm2, mm0" \
    "pmaxsw    mm2, mm1" \
    "pminsw    mm4, mm0" \
    "pmaxsw    mm4, mm1" \
    "packsswb  mm2, mm4" \
    "movq      [edx+8], mm2" \
    "movups    xmm1, [eax+64]" \
    "mulps     xmm1, xmm0" \
    "movups    xmm2, [eax+80]" \
    "mulps     xmm2, xmm0" \
    "cvtps2pi  mm2, xmm1" \
    "movhlps   xmm3, xmm1" \
    "cvtps2pi  mm3, xmm3" \
    "cvtps2pi  mm4, xmm2" \
    "movhlps   xmm4, xmm2" \
    "cvtps2pi  mm5, xmm4" \
    "packssdw  mm2, mm3" \
    "packssdw  mm4, mm5" \
    "pminsw    mm2, mm0" \
    "pmaxsw    mm2, mm1" \
    "pminsw    mm4, mm0" \
    "pmaxsw    mm4, mm1" \
    "packsswb  mm2, mm4" \
    "movq      [edx+16], mm2" \
    "movups    xmm1, [eax+96]" \
    "mulps     xmm1, xmm0" \
    "movups    xmm2, [eax+112]" \
    "mulps     xmm2, xmm0" \
    "cvtps2pi  mm2, xmm1" \
    "movhlps   xmm3, xmm1" \
    "cvtps2pi  mm3, xmm3" \
    "cvtps2pi  mm4, xmm2" \
    "movhlps   xmm4, xmm2" \
    "cvtps2pi  mm5, xmm4" \
    "packssdw  mm2, mm3" \
    "packssdw  mm4, mm5" \
    "pminsw    mm2, mm0" \
    "pmaxsw    mm2, mm1" \
    "pminsw    mm4, mm0" \
    "pmaxsw    mm4, mm1" \
    "packsswb  mm2, mm4" \
    "movq      [edx+24], mm2" \
    __parm [__eax] [__edx] [__ecx] \
    __modify [8087]

/* lz_q8round_group_sse(): the whole gs/32-chunk loop for one group of
   src/ops.c's lz_quantize_q8 - one call per group instead of gs/32
   calls a per-sub-chunk wrap would cost. The emms is emitted here,
   inside the group function: the "kernels don't emit emms" convention does not
   hold for this loop, the next group's x87 amax/inv still follows
   immediately in ops.c. */
void lz_q8round_group_sse(const float *grp, int8_t *out, int gs,
                          const float *pinv) {
    int k;
    for (k = 0; k + 31 < gs; k += 32)
        lz_q8round32_sse_asm(grp + k, out + k, pinv);
    _mm_empty();
}

/* lz_q8round_2p_group_sse(): the whole two-plane round for one group
   of src/ops.c's lz_gdn_quantize_2p - both kernel loops AND the
   residual float arithmetic between them. The residual computation is
   plain C float arithmetic; putting it in this TU is safe for Watcom
   the way it would not be for gcc under -mmmx (src/ops_mmx.c's header
   explains the difference - Watcom never emits MMX except where a
   #pragma aux says so, so mixing float C and #pragma aux kernels in
   one TU here is exactly as safe as ops.c doing the same). */
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

/* ---- norm_ss_fixed's element loop, SSE1 tier, Watcom twin.
   gcc twin: lz_norm_ss_sse, the intrinsics body below. SSE2 twin:
   lz_norm_ss_sse2_asm in src/ops_sse2.c, which carries the derivation
   of every step - read that one first.

   FOUR ELEMENTS a pass against the SSE2 twin's eight, because SSE1's
   convert is cvtps2pi and its destination is %mm. Everything after the
   scale therefore happens in the MMX register file, and the wrapper
   below owes the emms.

   THE ACCUMULATE IS SCALAR, into the caller's int64 through [edi].
   paddq is SSE2 even on %mm, and an int32 accumulator cannot take a
   second pmaddwd result - each is at most 2147352578 against an
   INT32_MAX only 131069 above. So each pass drains its two lanes with
   an add/adc pair, which is also why ebx is free to reuse after the
   scale broadcast. */
extern int lz_norm_ss_sse_asm(const float *x, int n, const float *scp,
                              short *qout, lz_i64 *acc);
#pragma aux lz_norm_ss_sse_asm = \
    ".686"                                                      \
    "movss     xmm0, [ebx]"                                       \
    "shufps    xmm0, xmm0, 0"         /* sc broadcast */          \
    "pcmpeqw   mm0, mm0"                                          \
    "psrlw     mm0, 1"                /* 0x7FFF per lane */       \
    "pxor      mm1, mm1"                                          \
    "psubw     mm1, mm0"              /* -32767 per lane */       \
    "movq      mm0, mm1"              /* keep the low clamp */    \
    "xor       ecx, ecx"                                          \
    "mov       [edi], ecx"                                        \
    "mov       [edi+4], ecx"          /* acc = 0 */               \
    "and       edx, -4"                                           \
    "mov       ecx, edx"                                          \
    "test      ecx, ecx"                                          \
    "jz        nssdone"                                           \
    "nssloop:"                                                  \
    "movups    xmm1, [eax]"                                       \
    "mulps     xmm1, xmm0"                                        \
    "cvtps2pi  mm2, xmm1"            /* low two -> int32 */      \
    "movhlps   xmm2, xmm1"                                        \
    "cvtps2pi  mm3, xmm2"            /* high two -> int32 */     \
    "packssdw  mm2, mm3"             /* 4 int16, upper clamp */  \
    "pmaxsw    mm2, mm0"              /* low clamp to -32767 */   \
    "test      esi, esi"                                          \
    "jz        nssnostore"                                        \
    "movq      [esi], mm2"                                        \
    "add       esi, 8"                                            \
    "nssnostore:"                                               \
    "movq      mm4, mm2"                                          \
    "pmaddwd   mm4, mm2"              /* two pairwise squares */  \
    "movd      ebx, mm4"                                          \
    "movq      mm5, mm4"                                          \
    "psrlq     mm5, 32"                                           \
    "movd      ebp, mm5"                                          \
    "add       [edi], ebx"                                        \
    "adc       dword ptr [edi+4], 0"                              \
    "add       [edi], ebp"                                        \
    "adc       dword ptr [edi+4], 0"                              \
    "add       eax, 16"                                           \
    "sub       ecx, 4"                                            \
    "jnz       nssloop"                                           \
    "nssdone:"                                                  \
    "emms"                                                      \
    "mov       eax, edx"                                          \
    parm [eax] [edx] [ebx] [esi] [edi]                          \
    value [eax]                                                 \
    modify [eax ebx ecx edx esi ebp];

/* Thin wrapper: a #pragma aux body expands at its call site, and the
   call site is src/ops_quant.c. Same reason lz_norm_ss_sse2 exists. */
int lz_norm_ss_sse(const float *x, int n, float sc, short *qout,
                   lz_i64 *acc) {
    return lz_norm_ss_sse_asm(x, n, &sc, qout, acc);
}

/* ---- attention wsum's chunk fold, SSE1 tier, Watcom twin.
   accf[d] += (float)acc32[d]. cvtsi2ss converts ONE int32 at a time -
   SSE1's packed convert is cvtpi2ps, whose source is an %mm register,
   and the caller has just paid an emms to leave the row-pair loop and
   goes back to x87 immediately after. Reaching for %mm here would owe a
   second emms for two elements a conversion; the scalar convert owes
   nothing and still replaces lz_i32f's six-operation x87 split form.

   FOUR AT A TIME so the converts and the adds of neighbouring elements
   can pair - cvtsi2ss has a long enough latency that a single
   dependency chain would stall on every one. The 32-element loop is the
   wrapper's, like lz_q8round_group_sse's above. */
extern void lz_i32f_acc4_sse_asm(float *accf, const int32_t *acc32);
#pragma aux lz_i32f_acc4_sse_asm = \
    ".686" \
    "cvtsi2ss xmm0, dword ptr [edx]" \
    "cvtsi2ss xmm1, dword ptr [edx+4]" \
    "cvtsi2ss xmm2, dword ptr [edx+8]" \
    "cvtsi2ss xmm3, dword ptr [edx+12]" \
    "addss    xmm0, dword ptr [eax]" \
    "addss    xmm1, dword ptr [eax+4]" \
    "addss    xmm2, dword ptr [eax+8]" \
    "addss    xmm3, dword ptr [eax+12]" \
    "movss    dword ptr [eax], xmm0" \
    "movss    dword ptr [eax+4], xmm1" \
    "movss    dword ptr [eax+8], xmm2" \
    "movss    dword ptr [eax+12], xmm3" \
    __parm [__eax] [__edx] \
    __modify [8087]

void lz_i32f_acc32_sse(float *accf, const int32_t *acc32) {
    int k;
    for (k = 0; k < 32; k += 4) lz_i32f_acc4_sse_asm(accf + k, acc32 + k);
}

/* ---- Hadamard over FLOATS: one butterfly stage of lz_fwht, SSE1
   Watcom twin. gcc twin: lz_fwht_stage_f32_sse, the intrinsics body
   below - read that one for why this is exact rather than close, and
   why the float and int32 transforms take mirrored ISA answers.

   Byte offsets, like both of the int32 twins: `len` is scaled once
   instead of at every index. */
extern void lz_fwht_stage_f32_sse_asm(float *y, int n, int len);
/* No separate base register: after the inner loop a has reached base+len
   and b has reached base+2*len, which IS the next block's base. So the
   outer step is `a = b; b = a + len` - two moves, and ebx/ebp stay out
   of the clobber list entirely. */
#pragma aux lz_fwht_stage_f32_sse_asm = \
    ".686"                                                      \
    "shl    ecx, 2"                 /* len -> bytes */          \
    "lea    edx, [eax+edx*4]"       /* end = y + n */           \
    "lea    esi, [eax+ecx]"         /* b = y + len */           \
    "fwfouter:"                                                 \
    "mov    edi, ecx"                                           \
    "fwfinner:"                                                 \
    "movups xmm0, [eax]"                                        \
    "movups xmm1, [esi]"                                        \
    "movaps xmm2, xmm0"                                         \
    "addps  xmm0, xmm1"                                         \
    "subps  xmm2, xmm1"                                         \
    "movups [eax], xmm0"                                        \
    "movups [esi], xmm2"                                        \
    "add    eax, 16"                                            \
    "add    esi, 16"                                            \
    "sub    edi, 16"                                            \
    "jnz    fwfinner"                                           \
    "mov    eax, esi"                                           \
    "lea    esi, [esi+ecx]"                                     \
    "cmp    eax, edx"                                           \
    "jb     fwfouter"                                           \
    parm [eax] [edx] [ecx]                                      \
    modify [eax ecx edx esi edi];

void lz_fwht_stage_f32_sse(float *y, int n, int len) {
    lz_fwht_stage_f32_sse_asm(y, n, len);
}

/* ---- lz_matmul's F32-weight row, SSE1 Watcom twin.
   gcc twin: lz_matmul_row_sse, the intrinsics body below - read that
   one for why two xmm accumulators ARE the C's eight scalars rather
   than a re-association of them, and why the final combine stays with
   the caller.

   Touches no %mm register, so no emms is owed: mulps/addps are the SSE1
   additions to the FLOAT side, not the MMX register file that this
   file's other kernels share with x87. The row/x LOADS are movaps (both
   are 16-byte aligned - row is field->f from model.c's aligned_malloc,
   x is the forward activation from forward.c's xcalloc); the acc8 STORE
   stays movups because acc8 is the caller's stack array with no
   16-byte alignment promise. */
extern int lz_matmul_row_sse_asm(const float *row, const float *x,
                                 int in_dim, float *acc8);
#pragma aux lz_matmul_row_sse_asm = \
    ".686"                                                      \
    "xorps   xmm0, xmm0"                                        \
    "xorps   xmm1, xmm1"                                        \
    "and     ecx, -8"                                           \
    "mov     esi, ecx"                                          \
    "test    esi, esi"                                          \
    "jz      mmdone"                                            \
    "mmloop:"                                                   \
    "movaps  xmm2, [eax]"                                       \
    "movaps  xmm3, [edx]"                                       \
    "movaps  xmm4, [eax+16]"                                    \
    "movaps  xmm5, [edx+16]"                                    \
    "mulps   xmm2, xmm3"                                        \
    "mulps   xmm4, xmm5"                                        \
    "addps   xmm0, xmm2"                                        \
    "addps   xmm1, xmm4"                                        \
    "add     eax, 32"                                           \
    "add     edx, 32"                                           \
    "sub     esi, 8"                                            \
    "jnz     mmloop"                                            \
    "mmdone:"                                                   \
    "movups  [ebx], xmm0"                                       \
    "movups  [ebx+16], xmm1"                                    \
    "mov     eax, ecx"                                          \
    parm [eax] [edx] [ecx] [ebx]                                \
    value [eax]                                                 \
    modify [eax ecx edx esi];

/* Thin wrapper: a #pragma aux body expands at the call site, and the
   call site is src/ops.c. Same reason lz_dot128_q16_mmx_w exists. */
int lz_matmul_row_sse(const float *row, const float *x, int in_dim,
                      float *acc8) {
    return lz_matmul_row_sse_asm(row, x, in_dim, acc8);
}

/* ---- lz_exp_fixed's Q20 Taylor, SSE1 tier, Watcom twin.
   gcc twin: lz_exp_q20_sse, the intrinsics body in the other arm,
   which says what SSE1 adds over the MMX kernel and what it leaves
   unchanged. The constant block and the back half - the three-limb t2
   and the output multiply - are the MMX body's, verbatim; only the
   front half differs.

   THE CONSTANT BLOCK IS THE MMX ONE PLUS TWO WORD BROADCASTS, and it
   is a second array rather than a shared one because the two front
   halves need different things: pmaddwd wants 22 and 5921 in int32
   lanes at a 2^15 split, pmullw/pmulhuw want 11 and 5921 in all four
   int16 lanes at a 2^16 split. Sharing would mean one of the two
   reading the other's layout. */
/* One value broadcast across a 64-bit operand, as two int32 lanes or as
   four int16 ones. Written as expressions so the compiler produces the
   words - a hand-assembled hex constant here would be a table with no
   producer, and the two-power structure would stop being readable. */
#define LZ_ES2(v) ((((lz_i64)(v)) << 32) | (lz_i64)(unsigned)(v))
#define LZ_ES4(v) (((lz_i64)(unsigned)((v) & 0xFFFF)) \
                   * LZ_I64_C(0x0001000100010001))
static const lz_i64 exp_q20_sk[7] = {
    LZ_ES2(1023), LZ_ES2(2047), LZ_ES2(1 << 19),
    LZ_ES2(7), LZ_ES2(22530),
    LZ_ES4(11), LZ_ES4(5921)
};
#undef LZ_ES2
#undef LZ_ES4

/* np is the number of TWO-element passes. One emms at the exit: pshufw,
   pmullw, pmulhuw and pmaddwd all write %mm, so this owes it exactly as
   the MMX twin does - SSE1 moved the instructions, not the register
   file. */
extern void lz_exp_q20_2_sse_asm(const int32_t *tab, const int32_t *s,
                                 int32_t *prod, const void *k, int np);
#pragma aux lz_exp_q20_2_sse_asm = \
    ".686"                          \
    "expq20sloop:" \
    "pshufw    mm0, [edx], 8" \
    "movq      mm1, mm0" \
    "pmullw    mm1, [ecx+40]" \
    "movq      mm2, mm0" \
    "pmulhuw   mm2, [ecx+40]" \
    "movq      mm3, mm0" \
    "pmullw    mm3, [ecx+48]" \
    "movq      mm4, mm0" \
    "pmulhuw   mm4, [ecx+48]" \
    "punpcklwd mm1, mm2" \
    "movq      mm2, mm0" \
    "punpcklwd mm3, mm4" \
    "paddd     mm3, [ecx+16]" \
    "psrld     mm3, 16" \
    "paddd     mm1, mm3" \
    "psrld     mm1, 4" \
    "pmullw    mm2, mm0" \
    "pmulhuw   mm0, mm0" \
    "punpcklwd mm2, mm0" \
    "movq      mm4, mm2" \
    "pand      mm4, [ecx]" \
    "movq      mm5, mm2" \
    "psrld     mm5, 10" \
    "pand      mm5, [ecx]" \
    "psrld     mm2, 20" \
    "movq      mm6, mm4" \
    "pmaddwd   mm6, [ecx+32]" \
    "pmaddwd   mm4, [ecx+24]" \
    "movq      mm3, mm5" \
    "pmaddwd   mm3, [ecx+32]" \
    "pmaddwd   mm5, [ecx+24]" \
    "movq      mm7, mm2" \
    "pmaddwd   mm7, [ecx+32]" \
    "pmaddwd   mm2, [ecx+24]" \
    "pslld     mm4, 15" \
    "paddd     mm6, mm4" \
    "pslld     mm5, 15" \
    "paddd     mm3, mm5" \
    "pslld     mm2, 15" \
    "paddd     mm7, mm2" \
    "psrld     mm6, 10" \
    "paddd     mm3, mm6" \
    "psrld     mm3, 10" \
    "paddd     mm7, [ecx+16]" \
    "paddd     mm7, mm3" \
    "psrld     mm7, 20" \
    "paddd     mm1, mm7" \
    "movq      mm0, [eax]" \
    "movq      mm3, mm0" \
    "psrld     mm3, 11" \
    "movq      mm7, mm0" \
    "pand      mm7, [ecx+8]" \
    "pmaddwd   mm3, mm1" \
    "pmaddwd   mm7, mm1" \
    "paddd     mm7, [ecx+16]" \
    "psrld     mm7, 11" \
    "paddd     mm3, mm7" \
    "psrld     mm3, 9" \
    "paddd     mm0, mm3" \
    "movq      [ebx], mm0" \
    "add       eax, 8" \
    "add       edx, 8" \
    "add       ebx, 8" \
    "dec       esi" \
    "jnz       expq20sloop" \
    "emms" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [__eax __ebx __ecx __edx __esi 8087];

void lz_exp_q20_sse(const int32_t *tab, const int32_t *s,
                    int32_t *prod, int n) {
    int k = n & ~1;
    if (k) lz_exp_q20_2_sse_asm(tab, s, prod, exp_q20_sk, k >> 1);
    for (; k < n; k++) {
        int32_t sk = s[k], cq;
        cq = (1 << 20)
           + (int32_t)((((lz_i64)726817 * sk) + (1 << 19)) >> 20)
           + (int32_t)((((lz_i64)251906 * ((lz_i64)sk * sk))
                        + (LZ_I64_C(1) << 39)) >> 40);
        prod[k] = (int32_t)((((lz_i64)tab[k] * cq) + (1 << 19)) >> 20);
    }
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
   (the no-emms convention). */
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

/* ---- norm_ss_fixed's element loop, SSE1 gcc twin.
   Watcom twin: lz_norm_ss_sse_asm, the #pragma aux above. SSE2 twin:
   lz_norm_ss_sse2 in src/ops_sse2.c, which is where the derivation of
   every step lives - why cvtps2dq IS q8_round, why packssdw's
   saturation is the upper clamp for free, and why the pairwise square
   sum is safe and the next add is not.

   THE ONE DIFFERENCE, and it is what makes this an SSE1 cell rather
   than a copy: SSE1's convert is cvtps2pi, whose destination is the
   MMX register file. So the pack, the clamp and the multiply-add are
   all %mm here, four elements at a time against the SSE2 twin's eight,
   and this cell owes an emms - taken at the exit rather than per
   iteration, since nothing in the loop is x87.

   AND THE ACCUMULATE FALLS TO SCALAR, which is not a shortcut. paddq
   is SSE2 even on %mm, and an int32 accumulator cannot take a second
   pmaddwd result: each is at most 2147352578 and INT32_MAX is 131069
   above it. One movq and two int64 adds per four elements is the
   correct answer, not a concession.

   Groups of 4, and the caller runs the tail - same contract as the
   SSE2 twin, which takes groups of 8. */
int lz_norm_ss_sse(const float *x, int n, float sc, short *qout,
                   lz_i64 *acc) {
    const __m128 vsc = _mm_set1_ps(sc);
    const __m64 lo = _mm_set1_pi16((short)-32767);
    lz_i64 a = 0;
    int nb = n & ~3;
    int b;

    for (b = 0; b < nb; b += 4) {
        __m128 v = _mm_mul_ps(_mm_load_ps(x + b), vsc);
        __m64 i0 = _mm_cvtps_pi32(v);
        __m64 i1 = _mm_cvtps_pi32(_mm_movehl_ps(v, v));
        __m64 q = _mm_max_pi16(_mm_packs_pi32(i0, i1), lo);
        __m64 s = _mm_madd_pi16(q, q);
        union { __m64 m; int32_t d[2]; } o;
        if (qout) memcpy(qout + b, &q, 8);
        o.m = s;
        a += (lz_i64)o.d[0] + (lz_i64)o.d[1];
    }
    _mm_empty();
    *acc = a;
    return nb;
}

/* ---- lz_matmul's F32-weight row, SSE1 gcc twin.
   Watcom twin: lz_matmul_row_sse_asm, the #pragma aux above.

   THE ONE HOT LOOP THE MATRIX HAD NO UNIT FOR. Every quantized format
   carries row kernels; an F32 weight went through a plain C loop on
   every platform, and that loop's own comment calls itself "the hottest
   loop in the whole forward, ~750M MACs per token".

   BIT-IDENTICAL BECAUSE THE C ALREADY HAS EIGHT ACCUMULATORS. It strides
   by 8 and keeps a0..a7 apart to break the latency chain - so lane k of
   these two vectors accumulates EXACTLY the sequence of products
   scalar a_k does, in the same order. Two __m128 are those eight
   accumulators, not a re-association of them.

   THE FINAL COMBINE STAYS IN C, deliberately. Reducing eight lanes to
   one is 7 adds against in_dim MACs, so nothing is saved by doing it
   here - and the C's order is ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7)),
   which a shuffle-based horizontal add would have to reproduce exactly
   rather than approximately. Returning the eight partials and letting
   the caller combine them makes that impossible to get wrong.

   Returns how many elements it consumed (a multiple of 8); the tail is
   the caller's, in C, exactly as before. */
/* ---- Hadamard over FLOATS: one butterfly stage of lz_fwht, SSE1 gcc
   twin. Watcom twin: lz_fwht_stage_f32_sse_asm, the #pragma aux above.

   The float sibling of src/ops_mmx.c's int32 cell, and the ISA answers
   are mirrored: this one needs addps/subps, which arrive with SSE1 and
   which MMX (integer only) cannot express; the int32 one needs
   paddd/psubd, which MMX has and SSE1 does not. That is why the two
   transforms are two units in the coverage matrix rather than one.

   len >= 4 is the precondition: below it the halves of the butterfly
   are not whole 16-byte vectors, and lz_fwht keeps those stages.

   EXACT, not close. The transform is one add and one sub per butterfly
   with no constant entering - lz_fwht's own comment already says that
   is why it is bit-identical between wcc386 and gcc - so a vector form
   that pairs the same operands under the same operators produces the
   same floats. It is not a reassociation: y[i+j] and y[i+j+len] are
   combined pairwise exactly as the scalar loop combines them.

   No emms: SSE1's float side touches no %mm register. */
/* Attention wsum's chunk fold, SSE1 tier, gcc twin of
   lz_i32f_acc4_sse_asm above - read that one for why the convert is
   scalar (cvtsi2ss) rather than SSE1's packed cvtpi2ps, which would
   reach for %mm at a site that has just paid an emms.

   Written rather than left to gcc's vectorizer for the reason
   src/ops_sse2.c's SSE2 twin gives: an auto-vectorization has no name
   the kernel matrix can register and no symbol a cross-toolchain
   comparison can reach. */
void lz_i32f_acc32_sse(float *accf, const int32_t *acc32) {
    int k;
    for (k = 0; k < 32; k++)
        _mm_store_ss(accf + k,
                     _mm_add_ss(_mm_load_ss(accf + k),
                                _mm_cvtsi32_ss(_mm_setzero_ps(), acc32[k])));
}

void lz_fwht_stage_f32_sse(float *y, int n, int len) {
    int i, j;
    for (i = 0; i < n; i += (len << 1)) {
        float *a = y + i;
        float *b = y + i + len;
        for (j = 0; j < len; j += 4) {
            __m128 u = _mm_loadu_ps(a + j);
            __m128 v = _mm_loadu_ps(b + j);
            _mm_storeu_ps(a + j, _mm_add_ps(u, v));
            _mm_storeu_ps(b + j, _mm_sub_ps(u, v));
        }
    }
}

int lz_matmul_row_sse(const float *row, const float *x, int in_dim,
                      float *acc8) {
    __m128 a = _mm_setzero_ps();
    __m128 b = _mm_setzero_ps();
    int j = 0, ng = in_dim & ~7;
    for (; j < ng; j += 8) {
        a = _mm_add_ps(a, _mm_mul_ps(_mm_loadu_ps(row + j),
                                     _mm_loadu_ps(x + j)));
        b = _mm_add_ps(b, _mm_mul_ps(_mm_loadu_ps(row + j + 4),
                                     _mm_loadu_ps(x + j + 4)));
    }
    _mm_storeu_ps(acc8, a);
    _mm_storeu_ps(acc8 + 4, b);
    return ng;
}

/* ---- lz_exp_fixed's Q20 Taylor, SSE1 gcc twin.
   Watcom twin: lz_exp_q20_2_sse_asm in the other arm. MMX twin:
   lz_exp_q20_mmx (src/ops_mmx.c), SSE2 twin: lz_exp_q20_simd
   (src/ops_sse2.c) - the SSE2 one carries the arithmetic derivation
   and the MMX one carries the limb decomposition this shares.

   WHAT SSE1 ADDS, and it is exactly two instructions. The MMX body
   cannot multiply by s directly: s reaches 2^15, one past signed
   int16, and pmaddwd is a SIGNED 16 x 16. So it carries s as
   u = s >> 1 plus a masked odd bit, and pays for that in both the
   first Taylor term and the square. Here:

     pshufw   packs two int32 down to two int16 with no saturation.
              packssdw would clamp 32768 to 32767 - silently, on the one
              input the split existed for.
     pmulhuw  gives the UNSIGNED high half, so pmullw+pmulhuw is a true
              16 x 16 -> 32 over the full unsigned range.

   With those, 11*s and 5921*s and s*s are each three instructions on
   the value itself and the u/b apparatus disappears - about six
   instructions a pass. The constant split moves from 2^15 to 2^16 for
   the same reason (LN2 = 11*2^16 + 5921 here, 22*2^15 + 5921 there).

   WHAT SSE1 DOES NOT CHANGE is t2 and the output, which are identical
   to the MMX body: LN2SQ is 18 bits and fits no 16-bit lane at any
   tier below SSE2, so its three-limb decomposition stays. Two elements
   a pass, same as MMX - the products are 32-bit and two of them fill
   an %mm register. A four-wide front half is possible (pmullw sees
   four words at once) at the cost of holding the back half in two
   registers; not taken, because the back half is the larger part and
   MMX's eight registers are already why this file's constants live in
   memory. */
void lz_exp_q20_sse(const int32_t *tab, const int32_t *s,
                    int32_t *prod, int n) {
    const __m64 k1023  = _mm_set1_pi32(1023);
    const __m64 k2047  = _mm_set1_pi32(2047);
    const __m64 kr19   = _mm_set1_pi32(1 << 19);
    const __m64 k7     = _mm_set1_pi32(7);
    const __m64 k22530 = _mm_set1_pi32(22530);
    const __m64 w11    = _mm_set1_pi16(11);
    const __m64 w5921  = _mm_set1_pi16(5921);
    int k;

    for (k = 0; k + 1 < n; k += 2) {
        __m64 sv, tv, sp, av, bv, t1, ss;
        __m64 l0, l1, l2, m0, m1, m2, inner, t2, d, a1, a0;

        memcpy(&sv, s + k, 8);
        memcpy(&tv, tab + k, 8);
        /* words 0 and 2 are the two elements' low halves; s < 2^16 so
           nothing above them is live. */
        sp = _mm_shuffle_pi16(sv, 0x08);

        av = _mm_unpacklo_pi16(_mm_mullo_pi16(sp, w11),
                               _mm_mulhi_pu16(sp, w11));      /* 11*s */
        bv = _mm_unpacklo_pi16(_mm_mullo_pi16(sp, w5921),
                               _mm_mulhi_pu16(sp, w5921));    /* 5921*s */
        bv = _mm_srli_pi32(_mm_add_pi32(bv, kr19), 16);
        t1 = _mm_srli_pi32(_mm_add_pi32(av, bv), 4);

        ss = _mm_unpacklo_pi16(_mm_mullo_pi16(sp, sp),
                               _mm_mulhi_pu16(sp, sp));       /* s*s */

        l0 = _mm_and_si64(ss, k1023);
        l1 = _mm_and_si64(_mm_srli_pi32(ss, 10), k1023);
        l2 = _mm_srli_pi32(ss, 20);
        m0 = _mm_add_pi32(_mm_madd_pi16(l0, k22530),
                          _mm_slli_pi32(_mm_madd_pi16(l0, k7), 15));
        m1 = _mm_add_pi32(_mm_madd_pi16(l1, k22530),
                          _mm_slli_pi32(_mm_madd_pi16(l1, k7), 15));
        m2 = _mm_add_pi32(_mm_madd_pi16(l2, k22530),
                          _mm_slli_pi32(_mm_madd_pi16(l2, k7), 15));
        inner = _mm_srli_pi32(_mm_add_pi32(m1, _mm_srli_pi32(m0, 10)), 10);
        t2 = _mm_srli_pi32(_mm_add_pi32(_mm_add_pi32(m2, kr19), inner), 20);

        d  = _mm_add_pi32(t1, t2);
        a1 = _mm_srli_pi32(tv, 11);
        a0 = _mm_and_si64(tv, k2047);
        {
            __m64 p1 = _mm_madd_pi16(a1, d);
            __m64 p0 = _mm_add_pi32(_mm_madd_pi16(a0, d), kr19);
            __m64 r  = _mm_srli_pi32(
                           _mm_add_pi32(p1, _mm_srli_pi32(p0, 11)), 9);
            r = _mm_add_pi32(tv, r);
            memcpy(prod + k, &r, 8);
        }
    }
    _mm_empty();
    for (; k < n; k++) {
        int32_t sk = s[k], cq;
        cq = (1 << 20)
           + (int32_t)((((lz_i64)726817 * sk) + (1 << 19)) >> 20)
           + (int32_t)((((lz_i64)251906 * ((lz_i64)sk * sk))
                        + (LZ_I64_C(1) << 39)) >> 40);
        prod[k] = (int32_t)((((lz_i64)tab[k] * cq) + (1 << 19)) >> 20);
    }
}

#endif /* __WATCOMC__ */

#endif /* LZ_MMX_TU */
