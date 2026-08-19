/* Every gcc/clang SSE2-intrinsics kernel, pulled out of src/ops.c and
   its included headers into the one translation unit built with
   -msse2. Code structure, not a correctness fix (task #22, per the
   user: "SSE2 next stage should also be separated for code clarity") -
   %xmm does not alias the x87 stack the way %mm does, so nothing here
   is implicated in the bug src/ops_mmx.c's split exists to prevent. It
   still gets its own TU: the ISA-floor gate needs a clean per-file
   boundary the same way src/ops_mmx.c/src/ops_mmx_sse.c do, and the
   32-bit x87 target gains a real SSE2 tier from it (see
   build/x87/build_engine.sh's header) - before this split, ops.c's own
   -mno-sse2 meant the SSE2 row kernels were only ever reachable on the
   x86-64 dev build, where ops.c happened to be compiled with -msse2
   itself.

   -march=i686, not i586: real SSE2 hardware in the target family (P4
   Northwood, Pentium M) is i686+ - SSE2 debuted with NetBurst, after
   Pentium Pro. Same honesty argument as src/ops_mmx_sse.c's header:
   this file's own -march floor costs nothing because no SSE2-capable
   machine is below it anyway. Measured (see ops_kernel_dot_sse2.h and
   this file's own build) that -march=i686 -mmmx -msse -msse2 together
   raise nothing beyond ordinary SSE2 codegen - no AVX, no FMA, nothing
   unrequested - the same empirical check that caught -msse quietly
   enabling CMOV in src/ops_mmx.c, done again for this flag set on its
   own object.

   Watcom NOW compiles this file too (task #30, 2/N) - same reason as
   src/ops_mmx.c: -Nr/-Ns is a hard ceiling its own .686 directives
   cannot exceed, so a #pragma aux body needing .686 codegen cannot sit
   in a TU pinned at the new, lower i486/i386 floor. */
#include <stdint.h>
#include <string.h>

#include "ops_sse2.h"

#if defined(LZ_SSE2_TU)

#if defined(__WATCOMC__)
#include "ops_kernel_shared.h" /* p2_shift_of - shared with ops.c and
                           the other suffixed TUs (task #31). gcc's
                           branch below needs none of it. */
#include "ops_mmx.h"       /* lz_dot128_x16_mmx_w - row_q8_sse2_asm's
                           -DLZ_SSE2_GROUP=0 A/B control build needs the
                           MMX-tier group kernel cross-TU (task #30, 5/N). */
#include "ops_p2_blk.h"

/* ---- GDN/KDA fixed-point pass 2, SSE2 tier - moved here from
   src/ops_kernel_p2.h (task #30, 2/N). Algorithm documentation stays
   there, unaffected by the code's move. Self-contained unlike the MMX/
   SSE1 tiers' cross-TU hand-off (src/ops_mmx.c <-> src/ops_mmx_sse.c):
   SSE2 has full content for both mul32 and split32, so this tier's own
   p2_blk() never needs to be seen outside this file. */

/* 16-byte-aligned scratch, by hand - same reasoning as src/ops_mmx.c's
   p2_blk(), a private copy rather than a shared one: nothing outside
   this file's own lz_p2_rows_sse2 ever needs this tier's buffer. NOT
   one of task #31's five shared names: unlike p2_shift_of, this one is
   genuinely tier-private by design (see the comment above), not a
   duplicate of the same logical thing. */
static lz_p2_blk *p2_blk_sse2(void) {
    static char raw[sizeof(lz_p2_blk) + 15];
    size_t off = (size_t)((uintptr_t)(void *)raw & 15u);
    return (lz_p2_blk *)(void *)(raw + ((16u - off) & 15u));
}

/* Instruction for instruction the MMX pair in src/ops_mmx.c, on xmm -
   see that pair's comments for the algorithm; this is the same
   operations twice as wide, not a different one. Alignment is
   load-bearing here (movdqa, and the shift-count operand FAULTS on a
   misaligned address) - p2_blk_sse2() above provides it by hand since
   Watcom's -zp4 caps what the type system can promise. */
extern void lz_p2_mul32_sse2_asm(const signed char *hi, const signed char *lo,
                                 const short *dq, void *blk);
#pragma aux lz_p2_mul32_sse2_asm = \
    ".686" \
    "pxor      xmm7, xmm7" \
    "pxor      xmm5, xmm5" \
    "pxor      xmm4, xmm4" \
    "movdqa    xmm6, [ecx]" \
    \
    "movq      xmm0, [eax]" \
    "movq      xmm1, [edx]" \
    "movdqa    xmm2, xmm7" \
    "punpcklbw xmm2, xmm0" \
    "punpcklbw xmm1, xmm1" \
    "psraw     xmm1, 8" \
    "paddw     xmm2, xmm1" \
    "movdqu    xmm3, [ebx]" \
    "movdqa    xmm0, xmm2" \
    "punpcklwd xmm0, xmm3" \
    "punpckhwd xmm2, xmm3" \
    "pmaddwd   xmm0, xmm6" \
    "pmaddwd   xmm2, xmm6" \
    "movdqa    [ecx+96], xmm0" \
    "movdqa    [ecx+112], xmm2" \
    "movdqa    xmm1, xmm0" \
    "psrad     xmm1, 31" \
    "pxor      xmm0, xmm1" \
    "psubd     xmm0, xmm1" \
    "movdqa    xmm3, xmm2" \
    "psrad     xmm3, 31" \
    "pxor      xmm2, xmm3" \
    "psubd     xmm2, xmm3" \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm0" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm0" \
    "por       xmm5, xmm1" \
    "movdqa    xmm3, xmm4" \
    "pcmpgtd   xmm3, xmm2" \
    "pand      xmm4, xmm3" \
    "pandn     xmm3, xmm2" \
    "por       xmm4, xmm3" \
    \
    "movq      xmm0, [eax+8]" \
    "movq      xmm1, [edx+8]" \
    "movdqa    xmm2, xmm7" \
    "punpcklbw xmm2, xmm0" \
    "punpcklbw xmm1, xmm1" \
    "psraw     xmm1, 8" \
    "paddw     xmm2, xmm1" \
    "movdqu    xmm3, [ebx+16]" \
    "movdqa    xmm0, xmm2" \
    "punpcklwd xmm0, xmm3" \
    "punpckhwd xmm2, xmm3" \
    "pmaddwd   xmm0, xmm6" \
    "pmaddwd   xmm2, xmm6" \
    "movdqa    [ecx+128], xmm0" \
    "movdqa    [ecx+144], xmm2" \
    "movdqa    xmm1, xmm0" \
    "psrad     xmm1, 31" \
    "pxor      xmm0, xmm1" \
    "psubd     xmm0, xmm1" \
    "movdqa    xmm3, xmm2" \
    "psrad     xmm3, 31" \
    "pxor      xmm2, xmm3" \
    "psubd     xmm2, xmm3" \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm0" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm0" \
    "por       xmm5, xmm1" \
    "movdqa    xmm3, xmm4" \
    "pcmpgtd   xmm3, xmm2" \
    "pand      xmm4, xmm3" \
    "pandn     xmm3, xmm2" \
    "por       xmm4, xmm3" \
    \
    "movq      xmm0, [eax+16]" \
    "movq      xmm1, [edx+16]" \
    "movdqa    xmm2, xmm7" \
    "punpcklbw xmm2, xmm0" \
    "punpcklbw xmm1, xmm1" \
    "psraw     xmm1, 8" \
    "paddw     xmm2, xmm1" \
    "movdqu    xmm3, [ebx+32]" \
    "movdqa    xmm0, xmm2" \
    "punpcklwd xmm0, xmm3" \
    "punpckhwd xmm2, xmm3" \
    "pmaddwd   xmm0, xmm6" \
    "pmaddwd   xmm2, xmm6" \
    "movdqa    [ecx+160], xmm0" \
    "movdqa    [ecx+176], xmm2" \
    "movdqa    xmm1, xmm0" \
    "psrad     xmm1, 31" \
    "pxor      xmm0, xmm1" \
    "psubd     xmm0, xmm1" \
    "movdqa    xmm3, xmm2" \
    "psrad     xmm3, 31" \
    "pxor      xmm2, xmm3" \
    "psubd     xmm2, xmm3" \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm0" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm0" \
    "por       xmm5, xmm1" \
    "movdqa    xmm3, xmm4" \
    "pcmpgtd   xmm3, xmm2" \
    "pand      xmm4, xmm3" \
    "pandn     xmm3, xmm2" \
    "por       xmm4, xmm3" \
    \
    "movq      xmm0, [eax+24]" \
    "movq      xmm1, [edx+24]" \
    "movdqa    xmm2, xmm7" \
    "punpcklbw xmm2, xmm0" \
    "punpcklbw xmm1, xmm1" \
    "psraw     xmm1, 8" \
    "paddw     xmm2, xmm1" \
    "movdqu    xmm3, [ebx+48]" \
    "movdqa    xmm0, xmm2" \
    "punpcklwd xmm0, xmm3" \
    "punpckhwd xmm2, xmm3" \
    "pmaddwd   xmm0, xmm6" \
    "pmaddwd   xmm2, xmm6" \
    "movdqa    [ecx+192], xmm0" \
    "movdqa    [ecx+208], xmm2" \
    "movdqa    xmm1, xmm0" \
    "psrad     xmm1, 31" \
    "pxor      xmm0, xmm1" \
    "psubd     xmm0, xmm1" \
    "movdqa    xmm3, xmm2" \
    "psrad     xmm3, 31" \
    "pxor      xmm2, xmm3" \
    "psubd     xmm2, xmm3" \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm0" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm0" \
    "por       xmm5, xmm1" \
    "movdqa    xmm3, xmm4" \
    "pcmpgtd   xmm3, xmm2" \
    "pand      xmm4, xmm3" \
    "pandn     xmm3, xmm2" \
    "por       xmm4, xmm3" \
    \
    "movdqa    xmm1, xmm5" \
    "pcmpgtd   xmm1, xmm4" \
    "pand      xmm5, xmm1" \
    "pandn     xmm1, xmm4" \
    "por       xmm5, xmm1" \
    "movdqa    [ecx+16], xmm5" \
    __parm [__eax] [__edx] [__ebx] [__ecx] \
    __modify [8087]

/* The 128-bit split. Every constant stays in memory - all four are
   16-byte aligned m128 operands, which frees all eight registers for the
   sixteen elements in flight and is why this needs no register budget
   note the way the MMX one does. */
extern void lz_p2_split32_sse2_asm(const void *blk, signed char *hi_out,
                                   signed char *lo_out);
#pragma aux lz_p2_split32_sse2_asm = \
    ".686" \
    "movdqa   xmm0, [eax+96]" \
    "movdqa   xmm1, [eax+112]" \
    "movdqa   xmm2, [eax+128]" \
    "movdqa   xmm3, [eax+144]" \
    "paddd    xmm0, [eax+32]" \
    "paddd    xmm1, [eax+32]" \
    "paddd    xmm2, [eax+32]" \
    "paddd    xmm3, [eax+32]" \
    "psrad    xmm0, [eax+48]" \
    "psrad    xmm1, [eax+48]" \
    "psrad    xmm2, [eax+48]" \
    "psrad    xmm3, [eax+48]" \
    "packssdw xmm0, xmm1" \
    "packssdw xmm2, xmm3" \
    "movdqa   xmm1, xmm0" \
    "paddw    xmm1, [eax+64]" \
    "psraw    xmm1, 8" \
    "movdqa   xmm3, xmm2" \
    "paddw    xmm3, [eax+64]" \
    "psraw    xmm3, 8" \
    "movdqa   xmm4, xmm1" \
    "psllw    xmm4, 8" \
    "psubw    xmm0, xmm4" \
    "movdqa   xmm5, xmm3" \
    "psllw    xmm5, 8" \
    "psubw    xmm2, xmm5" \
    "psubsw   xmm0, [eax+80]" \
    "paddsw   xmm0, [eax+80]" \
    "psubsw   xmm2, [eax+80]" \
    "paddsw   xmm2, [eax+80]" \
    "packsswb xmm1, xmm3" \
    "movdqu   [edx], xmm1" \
    "packsswb xmm0, xmm2" \
    "movdqu   [ebx], xmm0" \
    \
    "movdqa   xmm0, [eax+160]" \
    "movdqa   xmm1, [eax+176]" \
    "movdqa   xmm2, [eax+192]" \
    "movdqa   xmm3, [eax+208]" \
    "paddd    xmm0, [eax+32]" \
    "paddd    xmm1, [eax+32]" \
    "paddd    xmm2, [eax+32]" \
    "paddd    xmm3, [eax+32]" \
    "psrad    xmm0, [eax+48]" \
    "psrad    xmm1, [eax+48]" \
    "psrad    xmm2, [eax+48]" \
    "psrad    xmm3, [eax+48]" \
    "packssdw xmm0, xmm1" \
    "packssdw xmm2, xmm3" \
    "movdqa   xmm1, xmm0" \
    "paddw    xmm1, [eax+64]" \
    "psraw    xmm1, 8" \
    "movdqa   xmm3, xmm2" \
    "paddw    xmm3, [eax+64]" \
    "psraw    xmm3, 8" \
    "movdqa   xmm4, xmm1" \
    "psllw    xmm4, 8" \
    "psubw    xmm0, xmm4" \
    "movdqa   xmm5, xmm3" \
    "psllw    xmm5, 8" \
    "psubw    xmm2, xmm5" \
    "psubsw   xmm0, [eax+80]" \
    "paddsw   xmm0, [eax+80]" \
    "psubsw   xmm2, [eax+80]" \
    "paddsw   xmm2, [eax+80]" \
    "packsswb xmm1, xmm3" \
    "movdqu   [edx+16], xmm1" \
    "packsswb xmm0, xmm2" \
    "movdqu   [ebx+16], xmm0" \
    __parm [__eax] [__edx] [__ebx] \
    __modify [8087]

/* lz_p2_rows_sse2(): all of a row's groups, tier 3. One call per row
   from ops.c's gdn_p2_row_simd (task #30, 2/N) - fully self-contained,
   unlike lz_p2_rows_sse (src/ops_mmx_sse.c): SSE2 has real content for
   BOTH mul32 and split32, so neither kernel here needs a cross-TU call,
   only the row boundary itself does, in place of the zero real calls
   the pre-split #pragma-aux-inline loop had. */
void lz_p2_rows_sse2(const int8_t *ph_row, const int8_t *pl_row,
                     int pl_stride, const int16_t *dq,
                     const int16_t (*mul)[4], int8_t *oh_row,
                     int8_t *ol_row, int ol_stride, int *shv, int ng) {
    lz_p2_blk *blk = p2_blk_sse2();
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
        lz_p2_mul32_sse2_asm((const signed char *)(ph_row + gg * 32),
                             (const signed char *)plg,
                             (const short *)(dq + gg * 32), blk);
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
        lz_p2_split32_sse2_asm(blk, oh_row + gg * 32, olg);
    }
}

/* ---- Q8 group-scale amax, SSE2 tier - moved here from
   src/ops_kernel_amax.h (task #30, 4/N). Same zero-call-shape-change
   reasoning as src/ops_mmx.c's lz_amax32_mmx_w: the wrapper already sat
   between the pragma and LZ_AMAX_TAB before the move. */

/* Four lanes. The horizontal fold is two shuffle-and-compare rounds -
   pshufd is SSE2, so no MMX register is touched and no emms is owed. */
extern unsigned lz_amax32_sse2_asm(const float *x, int n);
#pragma aux lz_amax32_sse2_asm =        \
    ".686"                              \
    "pxor     xmm0, xmm0"               \
    "pcmpeqd  xmm3, xmm3"               \
    "psrld    xmm3, 1"                  \
    "shr      edx, 2"                   \
    "jz       S2"                       \
    "S1:"                               \
    "movups   xmm1, [eax]"              \
    "pand     xmm1, xmm3"               \
    "movdqa   xmm2, xmm1"               \
    "pcmpgtd  xmm2, xmm0"               \
    "pand     xmm1, xmm2"               \
    "pandn    xmm2, xmm0"               \
    "por      xmm1, xmm2"               \
    "movdqa   xmm0, xmm1"               \
    "add      eax, 16"                  \
    "dec      edx"                      \
    "jnz      S1"                       \
    "S2:"                               \
    "pshufd   xmm1, xmm0, 0x4E"         \
    "movdqa   xmm2, xmm0"               \
    "pcmpgtd  xmm2, xmm1"               \
    "pand     xmm0, xmm2"               \
    "pandn    xmm2, xmm1"               \
    "por      xmm0, xmm2"               \
    "pshufd   xmm1, xmm0, 0xB1"         \
    "movdqa   xmm2, xmm0"               \
    "pcmpgtd  xmm2, xmm1"               \
    "pand     xmm0, xmm2"               \
    "pandn    xmm2, xmm1"               \
    "por      xmm0, xmm2"               \
    "movd     eax, xmm0"                \
    parm [eax] [edx]                    \
    value [eax]                         \
    modify [eax edx];

/* Thin wrapper so LZ_AMAX_TAB can hold this. */
unsigned lz_amax32_sse2_w(const float *x, int n) {
    return lz_amax32_sse2_asm(x, n);
}

/* ---- Q8 activation rounding, SSE2 tier - moved here from
   src/ops_kernel_q8round.h (task #30, 4/N). cvtps2dq converts 4 floats
   at once - no MMX register touched, unlike the SSE1+MMX tier's
   cvtps2pi (src/ops_mmx_sse.c), so no emms is owed anywhere in this
   cell. */
extern void lz_q8round32_sse2_asm(const float *x, int8_t *o, const float *inv);
#pragma aux lz_q8round32_sse2_asm = \
    ".686" \
    "movss    xmm4, [ecx]" \
    "shufps   xmm4, xmm4, 0" \
    "pcmpeqw  xmm5, xmm5" \
    "psrlw    xmm5, 9" \
    "pxor     xmm6, xmm6" \
    "psubw    xmm6, xmm5" \
    "movups   xmm0, [eax]" \
    "movups   xmm1, [eax+16]" \
    "movups   xmm2, [eax+32]" \
    "movups   xmm3, [eax+48]" \
    "mulps    xmm0, xmm4" \
    "mulps    xmm1, xmm4" \
    "mulps    xmm2, xmm4" \
    "mulps    xmm3, xmm4" \
    "cvtps2dq xmm0, xmm0" \
    "cvtps2dq xmm1, xmm1" \
    "cvtps2dq xmm2, xmm2" \
    "cvtps2dq xmm3, xmm3" \
    "packssdw xmm0, xmm1" \
    "packssdw xmm2, xmm3" \
    "pminsw   xmm0, xmm5" \
    "pmaxsw   xmm0, xmm6" \
    "pminsw   xmm2, xmm5" \
    "pmaxsw   xmm2, xmm6" \
    "packsswb xmm0, xmm2" \
    "movdqu   [edx], xmm0" \
    "movups   xmm0, [eax+64]" \
    "movups   xmm1, [eax+80]" \
    "movups   xmm2, [eax+96]" \
    "movups   xmm3, [eax+112]" \
    "mulps    xmm0, xmm4" \
    "mulps    xmm1, xmm4" \
    "mulps    xmm2, xmm4" \
    "mulps    xmm3, xmm4" \
    "cvtps2dq xmm0, xmm0" \
    "cvtps2dq xmm1, xmm1" \
    "cvtps2dq xmm2, xmm2" \
    "cvtps2dq xmm3, xmm3" \
    "packssdw xmm0, xmm1" \
    "packssdw xmm2, xmm3" \
    "pminsw   xmm0, xmm5" \
    "pmaxsw   xmm0, xmm6" \
    "pminsw   xmm2, xmm5" \
    "pmaxsw   xmm2, xmm6" \
    "packsswb xmm0, xmm2" \
    "movdqu   [edx+16], xmm0" \
    __parm [__eax] [__edx] [__ecx] \
    __modify [8087]

/* lz_q8round_group_sse2(): the whole gs/32-chunk loop for one group -
   see src/ops_mmx_sse.c's lz_q8round_group_sse for the accounting.
   No emms: this tier touches no %mm register. */
void lz_q8round_group_sse2(const float *grp, int8_t *out, int gs,
                           const float *pinv) {
    int k;
    for (k = 0; k + 31 < gs; k += 32)
        lz_q8round32_sse2_asm(grp + k, out + k, pinv);
}

/* lz_q8round_2p_group_sse2(): the whole two-plane round for one group -
   see src/ops_mmx_sse.c's lz_q8round_2p_group_sse for the accounting.
   No emms anywhere in this tier's version either. */
void lz_q8round_2p_group_sse2(const float *grp, int8_t *ho, int8_t *lw,
                              int gs, const float *pinv,
                              const float *plo_mul, float *res) {
    int k;
    for (k = 0; k + 31 < gs; k += 32)
        lz_q8round32_sse2_asm(grp + k, ho + k, pinv);
    for (k = 0; k < gs; k++)
        res[k] = grp[k] * (*pinv) - (float)ho[k];
    for (k = 0; k + 31 < gs; k += 32)
        lz_q8round32_sse2_asm(res + k, lw + k, plo_mul);
}

/* ---- Q8_0 32/128-element dot products and row kernel, SSE2 tier -
   moved here from src/ops_kernel_dot.h / src/ops.c (task #30, 5/N).
   Algorithm documentation unchanged, just relocated with the code -
   see src/ops_mmx.c's MMX-tier twin for the shared background (weight
   widening, why the paired kernel exists) and this file's own
   pre-existing SSE2 kernels above for why no emms is owed anywhere
   here. */
#include "ops.h"           /* lz_prefetch_mode - row_q8_sse2_asm needs it,
                              real extern linkage, unaffected by this move. */
#include "ops_kernel_dot_shared.h" /* LZ_Q8_PFSEL, LZ_Q8_GROUP4 */

/* T42 (Pentium M) tier: hand-written SSE2. Pentium M has SSE2; integer
   SIMD goes from 64-bit to 128-bit, each pmaddwd eats 8 int16s instead
   of 4 - SIMD instruction count per 32 MACs drops from ~50 to ~26.
   Open Watcom has no emmintrin.h (only mmintrin.h), so this must also
   be hand-written; its inline asm accepts SSE2 mnemonics (probe-verified).
   One bonus: SSE2 does not share a register file with x87, so EMMS
   becomes unnecessary.
   Call structure identical to the MMX version (returns the x256 int32),
   so the same function name is reused and the upper matmul needs zero
   changes.
   Verified: bit-matches the scalar over 200K random + full-range
   groups. */
extern int32_t lz_dot32_x16_sse2_asm(const int8_t *w, const int16_t *x);
#pragma aux lz_dot32_x16_sse2_asm = \
    ".686"                          \
    "pxor      xmm7, xmm7"          \
    "movdqu    xmm0, [eax]"         \
    "movdqa    xmm1, xmm7"          \
    "punpcklbw xmm1, xmm0"          \
    "movdqu    xmm2, [edx]"         \
    "pmaddwd   xmm1, xmm2"          \
    "movdqa    xmm6, xmm1"          \
    "movdqa    xmm1, xmm7"          \
    "punpckhbw xmm1, xmm0"          \
    "movdqu    xmm2, [edx+16]"      \
    "pmaddwd   xmm1, xmm2"          \
    "paddd     xmm6, xmm1"          \
    "movdqu    xmm0, [eax+16]"      \
    "movdqa    xmm1, xmm7"          \
    "punpcklbw xmm1, xmm0"          \
    "movdqu    xmm2, [edx+32]"      \
    "pmaddwd   xmm1, xmm2"          \
    "paddd     xmm6, xmm1"          \
    "movdqa    xmm1, xmm7"          \
    "punpckhbw xmm1, xmm0"          \
    "movdqu    xmm2, [edx+48]"      \
    "pmaddwd   xmm1, xmm2"          \
    "paddd     xmm6, xmm1"          \
    "movdqa    xmm2, xmm6"          \
    "psrldq    xmm2, 8"             \
    "paddd     xmm6, xmm2"          \
    "movdqa    xmm2, xmm6"          \
    "psrldq    xmm2, 4"             \
    "paddd     xmm6, xmm2"          \
    "movd      eax, xmm6"           \
    __parm [__eax] [__edx]          \
    __value [__eax]                 \
    __modify [8087]

/* Compiled in alongside the MMX tier; the runtime picks by CPUID (see
   lz_kernel_select). One binary must run on both PII (MMX only) and
   T42 (SSE2). */
static int32_t dot32_x16_sse2a(const int8_t *w, const int16_t *x) {
    return lz_dot32_x16_sse2_asm(w, x);
}

/* SSE2 one-group-at-a-time kernel (128 elements). Verified
   bit-identical to the scalar reference over 800032 sub-blocks on both
   compilers. Wired in because a real Pentium M measured SSE2 at least
   20-30% faster than MMX at the kernel level. */
extern void lz_dot128_x16_sse2_asm(const int8_t *w,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_x16_sse2_asm = \
    ".686" \
    "pxor      xmm7, xmm7" \
    "pxor      xmm6, xmm6" \
    "movdqu    xmm0, [eax]" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+16]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqu    xmm0, [eax+16]" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+32]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+48]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 8" \
    "paddd     xmm6, xmm2" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 4" \
    "paddd     xmm6, xmm2" \
    "movd      [ebx], xmm6" \
    "pxor      xmm6, xmm6" \
    "movdqu    xmm0, [eax+32]" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+64]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+80]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqu    xmm0, [eax+48]" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+96]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+112]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 8" \
    "paddd     xmm6, xmm2" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 4" \
    "paddd     xmm6, xmm2" \
    "movd      [ebx+4], xmm6" \
    "pxor      xmm6, xmm6" \
    "movdqu    xmm0, [eax+64]" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+128]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+144]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqu    xmm0, [eax+80]" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+160]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+176]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 8" \
    "paddd     xmm6, xmm2" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 4" \
    "paddd     xmm6, xmm2" \
    "movd      [ebx+8], xmm6" \
    "pxor      xmm6, xmm6" \
    "movdqu    xmm0, [eax+96]" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+192]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+208]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqu    xmm0, [eax+112]" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+224]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+240]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 8" \
    "paddd     xmm6, xmm2" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 4" \
    "paddd     xmm6, xmm2" \
    "movd      [ebx+12], xmm6" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

/* row_q8_sse2_asm: ALREADY the real call boundary from its own caller
   (matmul_q8_impl, src/ops.c, dispatched once per matmul via
   lz_row_pick's LZ_ROW_Q8 table) - moving the whole function here,
   together with the leaf kernels it calls direct-by-name inside its
   own loop, adds NO new call sites (the gdn1_x4 argument, piece 1/N).
   No pairing logic here (unlike the MMX row function): SSE2's group
   kernel is already the fast path, and pairing was never wired in for
   this tier. */
void row_q8_sse2_asm(int32_t *acc32, const void *w4, const void *w2,
                     const int16_t *xw, int nb, int nt, int in_dim,
                     const void *pf_end, const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    const int8_t *wend = (const int8_t *)pf_end;
    int tk, g;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
#if LZ_SSE2_GROUP
        if ((nb & 3) == 0) LZ_Q8_GROUP4(lz_dot128_x16_sse2_asm);
#else
        /* LZ_SSE2_GROUP=0 keeps the 128-element step on the MMX kernel;
           the knob exists so an SSE2 CPU can be measured on either.
           lz_dot128_x16_mmx_w is src/ops_mmx.c's real, addressable
           twin of the #pragma aux lz_dot128_x16_asm - needed cross-TU
           here for exactly this A/B control build, since a #pragma aux
           body cannot have its address taken and this file does not
           carry the MMX pragma itself. */
        if ((nb & 3) == 0) LZ_Q8_GROUP4(lz_dot128_x16_mmx_w);
#endif
        else               LZ_Q8_PFSEL(dot32_x16_sse2a);
    }
}

/* ---- Q4_1 32/128-element dot products and row kernel, SSE2 tier -
   moved here from src/ops_kernel_dot.h / src/ops.c (task #30, 6/N). */

/* T42-tier Q4_1 (same: 128-bit, no EMMS needed, same name + signature
   reused). */
extern int32_t lz_dot32_q41_sse2_asm(const unsigned char *w, const int16_t *x);
#pragma aux lz_dot32_q41_sse2_asm = \
    ".686"                          \
    "pxor      xmm7, xmm7"          \
    "pcmpeqb   xmm3, xmm3"          \
    "psrlw     xmm3, 12"            \
    "packuswb  xmm3, xmm3"          \
    "movdqu    xmm0, [eax]"         \
    "movdqa    xmm4, xmm0"          \
    "pand      xmm0, xmm3"          \
    "psrlw     xmm4, 4"             \
    "pand      xmm4, xmm3"          \
    "movdqa    xmm1, xmm7"          \
    "punpcklbw xmm1, xmm0"          \
    "movdqu    xmm2, [edx]"         \
    "pmaddwd   xmm1, xmm2"          \
    "movdqa    xmm6, xmm1"          \
    "movdqa    xmm1, xmm7"          \
    "punpckhbw xmm1, xmm0"          \
    "movdqu    xmm2, [edx+16]"      \
    "pmaddwd   xmm1, xmm2"          \
    "paddd     xmm6, xmm1"          \
    "movdqa    xmm1, xmm7"          \
    "punpcklbw xmm1, xmm4"          \
    "movdqu    xmm2, [edx+32]"      \
    "pmaddwd   xmm1, xmm2"          \
    "paddd     xmm6, xmm1"          \
    "movdqa    xmm1, xmm7"          \
    "punpckhbw xmm1, xmm4"          \
    "movdqu    xmm2, [edx+48]"      \
    "pmaddwd   xmm1, xmm2"          \
    "paddd     xmm6, xmm1"          \
    "movdqa    xmm2, xmm6"          \
    "psrldq    xmm2, 8"             \
    "paddd     xmm6, xmm2"          \
    "movdqa    xmm2, xmm6"          \
    "psrldq    xmm2, 4"             \
    "paddd     xmm6, xmm2"          \
    "movd      eax, xmm6"           \
    __parm [__eax] [__edx]          \
    __value [__eax]                 \
    __modify [8087]

static int32_t dot32_q41_sse2a(const unsigned char *w, const int16_t *x) {
    return lz_dot32_q41_sse2_asm(w, x);
}

extern void lz_dot128_q41_sse2_asm(const unsigned char *w,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q41_sse2_asm = \
    ".686" \
    "pxor      xmm7, xmm7" \
    "pcmpeqb   xmm3, xmm3" \
    "psrlw     xmm3, 12" \
    "packuswb  xmm3, xmm3" \
    "movdqu    xmm0, [eax]" \
    "movdqa    xmm4, xmm0" \
    "pand      xmm0, xmm3" \
    "psrlw     xmm4, 4" \
    "pand      xmm4, xmm3" \
    "pxor      xmm6, xmm6" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+16]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm4" \
    "movdqu    xmm2, [ecx+32]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm4" \
    "movdqu    xmm2, [ecx+48]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 8" \
    "paddd     xmm6, xmm2" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 4" \
    "paddd     xmm6, xmm2" \
    "movd      [ebx], xmm6" \
    "movdqu    xmm0, [eax+16]" \
    "movdqa    xmm4, xmm0" \
    "pand      xmm0, xmm3" \
    "psrlw     xmm4, 4" \
    "pand      xmm4, xmm3" \
    "pxor      xmm6, xmm6" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+64]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+80]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm4" \
    "movdqu    xmm2, [ecx+96]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm4" \
    "movdqu    xmm2, [ecx+112]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 8" \
    "paddd     xmm6, xmm2" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 4" \
    "paddd     xmm6, xmm2" \
    "movd      [ebx+4], xmm6" \
    "movdqu    xmm0, [eax+32]" \
    "movdqa    xmm4, xmm0" \
    "pand      xmm0, xmm3" \
    "psrlw     xmm4, 4" \
    "pand      xmm4, xmm3" \
    "pxor      xmm6, xmm6" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+128]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+144]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm4" \
    "movdqu    xmm2, [ecx+160]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm4" \
    "movdqu    xmm2, [ecx+176]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 8" \
    "paddd     xmm6, xmm2" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 4" \
    "paddd     xmm6, xmm2" \
    "movd      [ebx+8], xmm6" \
    "movdqu    xmm0, [eax+48]" \
    "movdqa    xmm4, xmm0" \
    "pand      xmm0, xmm3" \
    "psrlw     xmm4, 4" \
    "pand      xmm4, xmm3" \
    "pxor      xmm6, xmm6" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+192]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm0" \
    "movdqu    xmm2, [ecx+208]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpcklbw xmm1, xmm4" \
    "movdqu    xmm2, [ecx+224]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm7" \
    "punpckhbw xmm1, xmm4" \
    "movdqu    xmm2, [ecx+240]" \
    "pmaddwd   xmm1, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 8" \
    "paddd     xmm6, xmm2" \
    "movdqa    xmm2, xmm6" \
    "psrldq    xmm2, 4" \
    "paddd     xmm6, xmm2" \
    "movd      [ebx+12], xmm6" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

/* row_q41_sse2_asm: ALREADY the real call boundary from its own caller
   (matmul_q41_impl, src/ops.c, dispatched once per matmul via
   LZ_ROW_Q41). No pairing logic, same as Q8_0's SSE2 row function. */
void row_q41_sse2_asm(int32_t *acc32, const void *w4, const void *w2,
                      const int16_t *xw, int nb, int nt, int in_dim,
                      const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *wend = (const unsigned char *)pf_end;
    int tk, s;
    (void)w2; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
#if LZ_SSE2_GROUP
        if ((nb & 3) == 0) LZ_Q41_GROUP4(lz_dot128_q41_sse2_asm);
#else
        /* lz_dot128_q41_mmx_w is src/ops_mmx.c's real wrapper around
           the MMX #pragma aux group kernel - task #32. */
        if ((nb & 3) == 0) LZ_Q41_GROUP4(lz_dot128_q41_mmx_w);
#endif
        else               LZ_Q41_PFSEL(dot32_q41_sse2a);
    }
}

#else /* gcc: intrinsics */

#include <emmintrin.h>

#include "ops_kernel_dot_sse2.h"

/* ---- Q8 row kernel ------------------------------------------------------
   Watcom twin: row_q8_sse2_asm, src/ops.c. */
void row_q8_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2) {
    const int8_t *wr = (const int8_t *)w4;
    int tk, g;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (g = 0; g + 3 < nb; g += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + g),
                fold4_sse2(
                part32_x16(wr + (size_t)(g + 0) * 32, xwt + (size_t)(g + 0) * 32),
                part32_x16(wr + (size_t)(g + 1) * 32, xwt + (size_t)(g + 1) * 32),
                part32_x16(wr + (size_t)(g + 2) * 32, xwt + (size_t)(g + 2) * 32),
                part32_x16(wr + (size_t)(g + 3) * 32, xwt + (size_t)(g + 3) * 32)));
        }
        for (; g < nb; g++) {           /* tail when nb % 4 != 0 */
            __m128i p = part32_x16(wr + (size_t)g * 32, xwt + (size_t)g * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[g] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- Q4_1 row kernel -----------------------------------------------------
   Watcom twin: row_q41_sse2_asm, src/ops.c. */
void row_q41_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    int tk, s;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s + 3 < nb; s += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + s),
                fold4_sse2(
                part32_q41(wn + (size_t)(s + 0) * 16, xwt + (size_t)(s + 0) * 32),
                part32_q41(wn + (size_t)(s + 1) * 16, xwt + (size_t)(s + 1) * 32),
                part32_q41(wn + (size_t)(s + 2) * 16, xwt + (size_t)(s + 2) * 32),
                part32_q41(wn + (size_t)(s + 3) * 16, xwt + (size_t)(s + 3) * 32)));
        }
        for (; s < nb; s++) {
            __m128i p = part32_q41(wn + (size_t)s * 16, xwt + (size_t)s * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[s] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- T2 row kernel --------------------------------------------------------
   Watcom twin: row_t2_sse2_asm, src/ops.c. No pairing (unmeasured for
   this format, same as the other tiers). */
void row_t2_sse2_intrin(int32_t *acc32, const void *w4, const void *w2v,
                        const int16_t *xw, int nb, int nt, int in_dim,
                        const void *pf_end, const void *pf_end2) {
    const unsigned char *p2 = (const unsigned char *)w4;
    int tk, s;
    (void)w2v; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++) {
            __m128i p = part32_t2(p2 + (size_t)s * 8, xwt + (size_t)s * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[s] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- Q6_1 row kernel -----------------------------------------------------
   Watcom twin: row_q61_sse2_asm, src/ops.c. */
void row_q61_sse2_intrin(int32_t *acc32, const void *w4, const void *w2v,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2) {
    const unsigned char *wn = (const unsigned char *)w4;
    const unsigned char *w2 = (const unsigned char *)w2v;
    int tk, s;
    (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (s = 0; s < nb; s++) {
            __m128i p = part32_q61(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                   xwt + (size_t)s * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[s] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- Q16_0 row kernel ----------------------------------------------------
   Watcom twin: row_q16_sse2_asm, src/ops.c. */
void row_q16_sse2_intrin(int32_t *acc32, const void *w4, const void *w2,
                         const int16_t *xw, int nb, int nt, int in_dim,
                         const void *pf_end, const void *pf_end2) {
    const int16_t *wr = (const int16_t *)w4;
    int tk, g;
    (void)w2; (void)pf_end; (void)pf_end2;
    for (tk = 0; tk < nt; tk++) {
        const int16_t *xwt = xw + (size_t)tk * in_dim;
        int32_t *acc = acc32 + (size_t)tk * nb;
        for (g = 0; g + 3 < nb; g += 4) {
            _mm_storeu_si128((__m128i *)(void *)(acc + g),
                fold4_sse2(
                part32_q16(wr + (size_t)(g + 0) * 32, xwt + (size_t)(g + 0) * 32),
                part32_q16(wr + (size_t)(g + 1) * 32, xwt + (size_t)(g + 1) * 32),
                part32_q16(wr + (size_t)(g + 2) * 32, xwt + (size_t)(g + 2) * 32),
                part32_q16(wr + (size_t)(g + 3) * 32, xwt + (size_t)(g + 3) * 32)));
        }
        for (; g < nb; g++) {           /* tail when nb % 4 != 0 */
            __m128i p = part32_q16(wr + (size_t)g * 32, xwt + (size_t)g * 32);
            p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
            p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
            acc[g] = _mm_cvtsi128_si32(p);
        }
    }
}

/* ---- GDN/KDA fixed-point pass 2, SSE2 tier -------------------------------
   Watcom twin: lz_p2_mul32_sse2_asm / lz_p2_split32_sse2_asm,
   src/ops_kernel_p2.h. Instruction for instruction the MMX pair in
   src/ops_mmx.c, on xmm - see that pair's comments for the algorithm;
   this is the same operations twice as wide, not a different one.
   Alignment is load-bearing here (movdqa, and the shift-count operand
   FAULTS on a misaligned address) - ops.c's p2_blk() provides it by
   hand since Watcom's -zp4 caps what the type system can promise. */
void lz_p2_mul32_sse2(const int8_t *hi, const int8_t *lo,
                      const int16_t *dq, lz_p2_blk *blk) {
    __m128i zero = _mm_setzero_si128();
    __m128i acc1 = zero, acc2 = zero;
    __m128i mul  = _mm_load_si128((const __m128i *)(const void *)blk->mul);
    __m128i m;
    int q;

    for (q = 0; q < 4; q++) {
        __m128i hw, lw, H, d, p0, p1, s0, s1, a0, a1;

        hw = _mm_loadl_epi64((const __m128i *)(const void *)(hi + q * 8));
        hw = _mm_unpacklo_epi8(zero, hw);                    /* hi * 256 */
        lw = _mm_loadl_epi64((const __m128i *)(const void *)(lo + q * 8));
        lw = _mm_unpacklo_epi8(lw, lw);
        lw = _mm_srai_epi16(lw, 8);                          /* lo, signed */
        H  = _mm_add_epi16(hw, lw);
        d  = _mm_loadu_si128((const __m128i *)(const void *)(dq + q * 8));
        p0 = _mm_madd_epi16(_mm_unpacklo_epi16(H, d), mul);
        p1 = _mm_madd_epi16(_mm_unpackhi_epi16(H, d), mul);
        _mm_store_si128((__m128i *)(void *)(blk->a + q * 8),     p0);
        _mm_store_si128((__m128i *)(void *)(blk->a + q * 8 + 4), p1);
        s0 = _mm_srai_epi32(p0, 31);
        a0 = _mm_sub_epi32(_mm_xor_si128(p0, s0), s0);       /* |A| */
        s1 = _mm_srai_epi32(p1, 31);
        a1 = _mm_sub_epi32(_mm_xor_si128(p1, s1), s1);
        m    = _mm_cmpgt_epi32(acc1, a0);
        acc1 = _mm_or_si128(_mm_and_si128(acc1, m), _mm_andnot_si128(m, a0));
        m    = _mm_cmpgt_epi32(acc2, a1);
        acc2 = _mm_or_si128(_mm_and_si128(acc2, m), _mm_andnot_si128(m, a1));
    }
    /* Fold to the four lanes the caller reads. MMX leaves two 2-lane
       accumulators there and SSE2 one 4-lane one; either way the caller
       folds four int32, so the two tiers hand back the same shape. */
    m    = _mm_cmpgt_epi32(acc1, acc2);
    acc1 = _mm_or_si128(_mm_and_si128(acc1, m), _mm_andnot_si128(m, acc2));
    _mm_store_si128((__m128i *)(void *)blk->amax, acc1);
}

void lz_p2_split32_sse2(const lz_p2_blk *blk, int8_t *oh, int8_t *ol) {
    __m128i rnd  = _mm_load_si128((const __m128i *)(const void *)blk->rnd);
    __m128i cnt  = _mm_load_si128((const __m128i *)(const void *)blk->cnt);
    __m128i k128 = _mm_load_si128((const __m128i *)(const void *)blk->k128);
    __m128i kclp = _mm_load_si128((const __m128i *)(const void *)blk->kclp);
    int o;

    for (o = 0; o < 2; o++) {
        __m128i a0, a1, a2, a3, hn0, hn1, h0, h1, l0, l1, t;

        a0 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16));
        a1 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 4));
        a2 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 8));
        a3 = _mm_load_si128((const __m128i *)(const void *)(blk->a + o * 16 + 12));
        a0 = _mm_sra_epi32(_mm_add_epi32(a0, rnd), cnt);
        a1 = _mm_sra_epi32(_mm_add_epi32(a1, rnd), cnt);
        a2 = _mm_sra_epi32(_mm_add_epi32(a2, rnd), cnt);
        a3 = _mm_sra_epi32(_mm_add_epi32(a3, rnd), cnt);
        hn0 = _mm_packs_epi32(a0, a1);
        hn1 = _mm_packs_epi32(a2, a3);
        h0 = _mm_srai_epi16(_mm_add_epi16(hn0, k128), 8);
        h1 = _mm_srai_epi16(_mm_add_epi16(hn1, k128), 8);
        l0 = _mm_sub_epi16(hn0, _mm_slli_epi16(h0, 8));
        l1 = _mm_sub_epi16(hn1, _mm_slli_epi16(h1, 8));
        l0 = _mm_adds_epi16(_mm_subs_epi16(l0, kclp), kclp); /* max(l, -127) */
        l1 = _mm_adds_epi16(_mm_subs_epi16(l1, kclp), kclp);
        t = _mm_packs_epi16(h0, h1);
        _mm_storeu_si128((__m128i *)(void *)(oh + o * 16), t);
        t = _mm_packs_epi16(l0, l1);
        _mm_storeu_si128((__m128i *)(void *)(ol + o * 16), t);
    }
}

/* ---- RoPE -----------------------------------------------------------------
   4-wide float RoPE. Each (i, h) pair is an independent IEEE mul/add,
   so per-element results are bit-identical to the scalar path in
   src/ops.c's lz_rope; only the loop order changes (head outer, i
   inner) to stream one head's two rotated halves contiguously. The
   cos/sin row is interleaved [c0,s0,c1,s1,...], so one 8-float load +
   two shuffles give the c and s vectors. half % 4 tail falls back to
   scalar. */
void lz_rope_sse2(float *v, int n_heads, int head_dim,
                  int rotary_dim, int pos, const float *cs) {
    int half = rotary_dim / 2;
    int h, i;
    const float *row = cs + (size_t)pos * half * 2;
    for (h = 0; h < n_heads; h++) {
        float *base = v + (size_t)h * head_dim;
        for (i = 0; i + 4 <= half; i += 4) {
            __m128 lo = _mm_loadu_ps(row + i * 2);
            __m128 hi = _mm_loadu_ps(row + i * 2 + 4);
            __m128 c = _mm_shuffle_ps(lo, hi, 0x88); /* [c0,c1,c2,c3] */
            __m128 s = _mm_shuffle_ps(lo, hi, 0xDD); /* [s0,s1,s2,s3] */
            __m128 x0 = _mm_loadu_ps(base + i);
            __m128 x1 = _mm_loadu_ps(base + i + half);
            __m128 n0 = _mm_sub_ps(_mm_mul_ps(x0, c), _mm_mul_ps(x1, s));
            __m128 n1 = _mm_add_ps(_mm_mul_ps(x1, c), _mm_mul_ps(x0, s));
            _mm_storeu_ps(base + i, n0);
            _mm_storeu_ps(base + i + half, n1);
        }
        for (; i < half; i++) {
            float c0 = row[i * 2], s0 = row[i * 2 + 1];
            float x0 = base[i], x1 = base[i + half];
            base[i]        = x0 * c0 - x1 * s0;
            base[i + half] = x1 * c0 + x0 * s0;
        }
    }
}

/* ---- Q8 group-scale amax --------------------------------------------------
   Watcom twin: lz_amax32_sse2_asm, src/ops_kernel_amax.h. Four lanes,
   no MMX register touched (pshufd is SSE2), so no emms owed. See that
   file's header for what is being vectorized and why the compare-and-
   blend below is exact, not approximate. */
unsigned lz_amax32_sse2(const float *x, int n) {
    __m128i acc = _mm_setzero_si128();
    __m128i msk = _mm_set1_epi32(0x7FFFFFFF);
    int k;
    for (k = 0; k + 3 < n; k += 4) {
        __m128i v = _mm_and_si128(
            _mm_loadu_si128((const __m128i *)(const void *)(x + k)), msk);
        __m128i gt = _mm_cmpgt_epi32(v, acc);
        acc = _mm_or_si128(_mm_and_si128(v, gt), _mm_andnot_si128(gt, acc));
    }
    {
        union { __m128i m; unsigned u[4]; } o;
        unsigned a, b;
        o.m = acc;
        a = o.u[0] > o.u[1] ? o.u[0] : o.u[1];
        b = o.u[2] > o.u[3] ? o.u[2] : o.u[3];
        return a > b ? a : b;
    }
}

/* ---- Q8 activation rounding, SSE2 tier -----------------------------------
   Watcom twin: lz_q8round32_sse2_asm, src/ops_kernel_q8round.h.
   cvtps2dq converts 4 floats at once - no MMX register touched, unlike
   the SSE1 tier's cvtps2pi (src/ops_mmx_sse.c). */
void lz_q8round32_simd(const float *x, int8_t *o, const float *pinv) {
    __m128 vi = _mm_set1_ps(*pinv);
    __m128i p127 = _mm_set1_epi16(127), m127 = _mm_set1_epi16(-127);
    int b;
    for (b = 0; b < 2; b++) {
        const float *s = x + b * 16;
        __m128i a0 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(s), vi));
        __m128i a1 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(s + 4), vi));
        __m128i a2 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(s + 8), vi));
        __m128i a3 = _mm_cvtps_epi32(_mm_mul_ps(_mm_loadu_ps(s + 12), vi));
        __m128i l = _mm_packs_epi32(a0, a1);
        __m128i h = _mm_packs_epi32(a2, a3);
        l = _mm_max_epi16(_mm_min_epi16(l, p127), m127);
        h = _mm_max_epi16(_mm_min_epi16(h, p127), m127);
        _mm_storeu_si128((__m128i *)(void *)(o + b * 16), _mm_packs_epi16(l, h));
    }
}

#endif /* __WATCOMC__ */

#endif /* LZ_SSE2_TU */
