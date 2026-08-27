/* Every gcc/clang SSE2-intrinsics kernel, in the one translation unit
   built with -msse2 - separated from ops.c and its included headers
   for code clarity. %xmm does not alias the x87 stack the way %mm
   does, so nothing here is implicated in the bug src/ops_mmx.c's split
   exists to prevent. It still gets its own TU: the ISA-floor gate
   needs a clean per-file boundary the same way src/ops_mmx.c/
   src/ops_mmx_sse.c do, and the 32-bit x87 target gains a real SSE2
   tier from it (see build/x87/build_engine.sh's header) - with ops.c
   compiled with -mno-sse2, the SSE2 row kernels are only reachable on
   the x86-64 dev build, where ops.c is compiled with -msse2 itself.

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

   Watcom compiles this file too - same reason as src/ops_mmx.c:
   -Nr/-Ns is a hard ceiling its own .686 directives cannot exceed, so
   a #pragma aux body needing .686 codegen cannot sit in a TU pinned
   at the lower i486/i386 floor. */
#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#include <string.h>

#include "ops_sse2.h"

#if defined(LZ_SSE2_TU)

#if defined(__WATCOMC__)
#include "ops_kernel_shared.h" /* p2_shift_of - shared with ops.c and
                           the other suffixed TUs. gcc's branch below
                           needs none of it. */
#include "ops_mmx.h"       /* lz_dot128_x16_mmx_w - row_q8_sse2_asm's
                           -DLZ_SSE2_GROUP=0 A/B control build needs the
                           MMX-tier group kernel cross-TU. */
#include "ops_p2_blk.h"

/* ---- GDN/KDA fixed-point pass 2, SSE2 tier - in this TU; algorithm
   documentation is in src/ops_kernel_p2.h, unaffected by the code's
   location. Self-contained unlike the MMX/SSE1 tiers' cross-TU hand-off
   (src/ops_mmx.c <-> src/ops_mmx_sse.c): SSE2 has full content for both
   mul32 and split32, so this tier's own p2_blk() never needs to be seen
   outside this file. */

/* 16-byte-aligned scratch, by hand - same reasoning as src/ops_mmx.c's
   p2_blk(), a private copy rather than a shared one: nothing outside
   this file's own lz_p2_rows_sse2 ever needs this tier's buffer. Not
   one of the five names shared via ops_kernel_shared.h: unlike
   p2_shift_of, this one is genuinely tier-private by design (see the
   comment above), not a duplicate of the same logical thing. */
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
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "pxor      xmm2, xmm2" \
    "movdqu    xmm3, [ecx]" \
    \
    "movq      xmm4, [eax]" \
    "movq      xmm5, [edx]" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm6, xmm4" \
    "punpcklbw xmm5, xmm5" \
    "psraw     xmm5, 8" \
    "paddw     xmm6, xmm5" \
    "movdqu    xmm7, [ebx]" \
    "movdqa    xmm4, xmm6" \
    "punpcklwd xmm4, xmm7" \
    "punpckhwd xmm6, xmm7" \
    "pmaddwd   xmm4, xmm3" \
    "pmaddwd   xmm6, xmm3" \
    "movdqa    [ecx+96], xmm4" \
    "movdqa    [ecx+112], xmm6" \
    "movdqa    xmm5, xmm4" \
    "psrad     xmm5, 31" \
    "pxor      xmm4, xmm5" \
    "psubd     xmm4, xmm5" \
    "movdqa    xmm7, xmm6" \
    "psrad     xmm7, 31" \
    "pxor      xmm6, xmm7" \
    "psubd     xmm6, xmm7" \
    "movdqa    xmm5, xmm1" \
    "pcmpgtd   xmm5, xmm4" \
    "pand      xmm1, xmm5" \
    "pandn     xmm5, xmm4" \
    "por       xmm1, xmm5" \
    "movdqa    xmm7, xmm2" \
    "pcmpgtd   xmm7, xmm6" \
    "pand      xmm2, xmm7" \
    "pandn     xmm7, xmm6" \
    "por       xmm2, xmm7" \
    \
    "movq      xmm4, [eax+8]" \
    "movq      xmm5, [edx+8]" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm6, xmm4" \
    "punpcklbw xmm5, xmm5" \
    "psraw     xmm5, 8" \
    "paddw     xmm6, xmm5" \
    "movdqu    xmm7, [ebx+16]" \
    "movdqa    xmm4, xmm6" \
    "punpcklwd xmm4, xmm7" \
    "punpckhwd xmm6, xmm7" \
    "pmaddwd   xmm4, xmm3" \
    "pmaddwd   xmm6, xmm3" \
    "movdqa    [ecx+128], xmm4" \
    "movdqa    [ecx+144], xmm6" \
    "movdqa    xmm5, xmm4" \
    "psrad     xmm5, 31" \
    "pxor      xmm4, xmm5" \
    "psubd     xmm4, xmm5" \
    "movdqa    xmm7, xmm6" \
    "psrad     xmm7, 31" \
    "pxor      xmm6, xmm7" \
    "psubd     xmm6, xmm7" \
    "movdqa    xmm5, xmm1" \
    "pcmpgtd   xmm5, xmm4" \
    "pand      xmm1, xmm5" \
    "pandn     xmm5, xmm4" \
    "por       xmm1, xmm5" \
    "movdqa    xmm7, xmm2" \
    "pcmpgtd   xmm7, xmm6" \
    "pand      xmm2, xmm7" \
    "pandn     xmm7, xmm6" \
    "por       xmm2, xmm7" \
    \
    "movq      xmm4, [eax+16]" \
    "movq      xmm5, [edx+16]" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm6, xmm4" \
    "punpcklbw xmm5, xmm5" \
    "psraw     xmm5, 8" \
    "paddw     xmm6, xmm5" \
    "movdqu    xmm7, [ebx+32]" \
    "movdqa    xmm4, xmm6" \
    "punpcklwd xmm4, xmm7" \
    "punpckhwd xmm6, xmm7" \
    "pmaddwd   xmm4, xmm3" \
    "pmaddwd   xmm6, xmm3" \
    "movdqa    [ecx+160], xmm4" \
    "movdqa    [ecx+176], xmm6" \
    "movdqa    xmm5, xmm4" \
    "psrad     xmm5, 31" \
    "pxor      xmm4, xmm5" \
    "psubd     xmm4, xmm5" \
    "movdqa    xmm7, xmm6" \
    "psrad     xmm7, 31" \
    "pxor      xmm6, xmm7" \
    "psubd     xmm6, xmm7" \
    "movdqa    xmm5, xmm1" \
    "pcmpgtd   xmm5, xmm4" \
    "pand      xmm1, xmm5" \
    "pandn     xmm5, xmm4" \
    "por       xmm1, xmm5" \
    "movdqa    xmm7, xmm2" \
    "pcmpgtd   xmm7, xmm6" \
    "pand      xmm2, xmm7" \
    "pandn     xmm7, xmm6" \
    "por       xmm2, xmm7" \
    \
    "movq      xmm4, [eax+24]" \
    "movq      xmm5, [edx+24]" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm6, xmm4" \
    "punpcklbw xmm5, xmm5" \
    "psraw     xmm5, 8" \
    "paddw     xmm6, xmm5" \
    "movdqu    xmm7, [ebx+48]" \
    "movdqa    xmm4, xmm6" \
    "punpcklwd xmm4, xmm7" \
    "punpckhwd xmm6, xmm7" \
    "pmaddwd   xmm4, xmm3" \
    "pmaddwd   xmm6, xmm3" \
    "movdqa    [ecx+192], xmm4" \
    "movdqa    [ecx+208], xmm6" \
    "movdqa    xmm5, xmm4" \
    "psrad     xmm5, 31" \
    "pxor      xmm4, xmm5" \
    "psubd     xmm4, xmm5" \
    "movdqa    xmm7, xmm6" \
    "psrad     xmm7, 31" \
    "pxor      xmm6, xmm7" \
    "psubd     xmm6, xmm7" \
    "movdqa    xmm5, xmm1" \
    "pcmpgtd   xmm5, xmm4" \
    "pand      xmm1, xmm5" \
    "pandn     xmm5, xmm4" \
    "por       xmm1, xmm5" \
    "movdqa    xmm7, xmm2" \
    "pcmpgtd   xmm7, xmm6" \
    "pand      xmm2, xmm7" \
    "pandn     xmm7, xmm6" \
    "por       xmm2, xmm7" \
    \
    "movdqa    xmm5, xmm1" \
    "pcmpgtd   xmm5, xmm2" \
    "pand      xmm1, xmm5" \
    "pandn     xmm5, xmm2" \
    "por       xmm1, xmm5" \
    "movdqa    [ecx+16], xmm1" \
    __parm [__eax] [__edx] [__ebx] [__ecx] \
    __modify [8087]

/* The SSE2 mul32's address. Its MMX twin has had lz_p2_mul32_mmx since
   the tier split, so this was the last kernel in these three files
   that nothing outside its own could call. This one writes INTO the
   block - amax and a[] - rather than returning anything, so
   build/p2_mul32_probe.c compares those fields. */
void lz_p2_mul32_sse2_w(const int8_t *hi, const int8_t *lo,
                        const int16_t *dq, lz_p2_blk *blk) {
    lz_p2_mul32_sse2_asm((const signed char *)hi, (const signed char *)lo,
                         (const short *)dq, blk);
}

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

/* The split kernel's address, so the three tiers can be compared. */
void lz_p2_split32_sse2_w(const void *blk, signed char *hi_out,
                          signed char *lo_out) {
    lz_p2_split32_sse2_asm(blk, hi_out, lo_out);
}

/* lz_p2_rows_sse2(): all of a row's groups, tier 3. One call per row
   from ops.c's gdn_p2_row_simd - fully self-contained, unlike
   lz_p2_rows_sse (src/ops_mmx_sse.c): SSE2 has real content for BOTH
   mul32 and split32, so neither kernel here needs a cross-TU call,
   only the row boundary itself does. */
void lz_p2_rows_sse2(const int8_t *ph_row, const int8_t *pl_row,
                     int pl_stride, const int16_t *dq,
                     const int16_t (*mul)[4], int8_t *oh_row,
                     int8_t *ol_row, int ol_stride, int *shv, int ng) {
    lz_p2_blk *blk = p2_blk_sse2();
    int gg, j;
    const int8_t *plg = pl_row;
    int8_t *olg = ol_row;
    for (gg = 0; gg < ng; gg++, plg += pl_stride, olg += ol_stride) {
        int32_t amax;
        int sh;
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

/* ---- Q8 group-scale amax, SSE2 tier - in this TU. Same
   zero-call-shape-change reasoning as src/ops_mmx.c's lz_amax32_mmx_w:
   the wrapper sits between the pragma and LZ_AMAX_TAB. */

/* Four lanes. The horizontal fold is two shuffle-and-compare rounds -
   pshufd is SSE2, so no MMX register is touched and no emms is owed. */
extern unsigned lz_amax32_sse2_asm(const float *x, int n);
#pragma aux lz_amax32_sse2_asm = \
    ".686"                              \
    "pxor      xmm0, xmm0"               \
    "pcmpeqd   xmm1, xmm1"               \
    "psrld     xmm1, 1"                  \
    "shr       edx, 2"                   \
    "jz        S2"                       \
    "S1:"                               \
    "movups    xmm2, [eax]"              \
    "pand      xmm2, xmm1"               \
    "movdqa    xmm3, xmm2"               \
    "pcmpgtd   xmm3, xmm0"               \
    "pand      xmm2, xmm3"               \
    "pandn     xmm3, xmm0"               \
    "por       xmm2, xmm3"               \
    "movdqa    xmm0, xmm2"               \
    "add       eax, 16"                  \
    "dec       edx"                      \
    "jnz       S1"                       \
    "S2:"                               \
    "pshufd    xmm2, xmm0, 0x4E"         \
    "movdqa    xmm3, xmm0"               \
    "pcmpgtd   xmm3, xmm2"               \
    "pand      xmm0, xmm3"               \
    "pandn     xmm3, xmm2"               \
    "por       xmm0, xmm3"               \
    "pshufd    xmm2, xmm0, 0xB1"         \
    "movdqa    xmm3, xmm0"               \
    "pcmpgtd   xmm3, xmm2"               \
    "pand      xmm0, xmm3"               \
    "pandn     xmm3, xmm2"               \
    "por       xmm0, xmm3"               \
    "movd      eax, xmm0"                \
    parm [eax] [edx]                    \
    value [eax]                         \
    modify [eax edx];

/* Thin wrapper so LZ_AMAX_TAB can hold this. */
unsigned lz_amax32_sse2_w(const float *x, int n) {
    return lz_amax32_sse2_asm(x, n);
}

/* ---- norm_ss_fixed's element loop, SSE2 Watcom twin.
   gcc twin: lz_norm_ss_sse2, the intrinsics body below - read that one
   for why cvtps2dq IS q8_round, why packssdw is the upper clamp, and
   why the pairwise square sum cannot overflow but the next one can.

   `qout` NULL is handled by branching once, outside the loop, into two
   otherwise identical bodies rather than by storing to a scratch: three
   of norm_ss_fixed's four call sites pass NULL, the same reason the ARM
   twin gives (src/ops_arm.h). */
extern int lz_norm_ss_sse2_asm(const float *x, int n, const float *scp,
                               short *qout, lz_i64 *acc);
#pragma aux lz_norm_ss_sse2_asm = \
    ".686"                                                      \
    "movss     xmm0, [ebx]"                                       \
    "shufps    xmm0, xmm0, 0"         /* sc broadcast */          \
    "pcmpeqw   xmm1, xmm1"                                        \
    "psrlw     xmm1, 1"               /* 0x7FFF per lane */       \
    "pxor      xmm2, xmm2"                                        \
    "psubw     xmm2, xmm1"            /* -32767 per lane */       \
    "movdqa    xmm1, xmm2"            /* keep the low clamp */     \
    "pxor      xmm3, xmm3"            /* running int64 acc */     \
    "pxor      xmm4, xmm4"            /* zero for the widen */    \
    "and       edx, -8"                                           \
    "mov       ecx, edx"                                          \
    "test      ecx, ecx"                                          \
    "jz        nsdone"                                            \
    "nsloop:"                                                   \
    "movups    xmm5, [eax]"                                       \
    "movups    xmm6, [eax+16]"                                    \
    "mulps     xmm5, xmm0"                                        \
    "mulps     xmm6, xmm0"                                        \
    "cvtps2dq  xmm5, xmm5"                                       \
    "cvtps2dq  xmm6, xmm6"                                       \
    "packssdw  xmm5, xmm6"                                       \
    "pmaxsw    xmm5, xmm1"                                        \
    "test      esi, esi"                                          \
    "jz        nsnostore"                                         \
    "movdqu    [esi], xmm5"                                       \
    "add       esi, 16"                                           \
    "nsnostore:"                                                \
    "movdqa    xmm7, xmm5"                                        \
    "pmaddwd   xmm7, xmm5"                                        \
    "movdqa    xmm6, xmm7"                                        \
    "punpckldq xmm7, xmm4"                                      \
    "punpckhdq xmm6, xmm4"                                      \
    "paddq     xmm3, xmm7"                                        \
    "paddq     xmm3, xmm6"                                        \
    "add       eax, 32"                                           \
    "sub       ecx, 8"                                            \
    "jnz       nsloop"                                            \
    "nsdone:"                                                   \
    "movdqa    xmm5, xmm3"                                        \
    "psrldq    xmm5, 8"                                           \
    "paddq     xmm3, xmm5"                                        \
    "movq      [edi], xmm3"                                       \
    "mov       eax, edx"                                          \
    parm [eax] [edx] [ebx] [esi] [edi]                          \
    value [eax]                                                 \
    modify [eax ecx edx esi];

/* Thin wrapper: the #pragma aux body expands at the call site, and the
   call site is src/ops_quant.c. Same reason lz_amax32_sse2_w exists. */
int lz_norm_ss_sse2(const float *x, int n, float sc, short *qout,
                    lz_i64 *acc) {
    return lz_norm_ss_sse2_asm(x, n, &sc, qout, acc);
}

/* ---- Hadamard: one butterfly stage of lz_fwht_i32, SSE2 Watcom twin.
   gcc twin: lz_fwht_stage_sse2, the intrinsics body below. Four int32
   per movdqu against the MMX cell's two; see that one (src/ops_mmx.c)
   for why the halves are contiguous and need no shuffle.

   Byte offsets for the same reason the MMX twin uses them: `len` is
   scaled once instead of at every index. edi walks the low half, esi
   the high, both by 16 - the four int32 a movdqu moves. */
extern void lz_fwht_stage_sse2_asm(int32_t *y, int n, int len);
#pragma aux lz_fwht_stage_sse2_asm = \
    ".686"                                                      \
    "shl    ecx, 2"                 /* len -> bytes */          \
    "shl    edx, 2"                 /* n   -> bytes */          \
    "lea    ebx, [ecx+ecx]"         /* stride = 2*len bytes */  \
    "add    edx, eax"               /* end of y */              \
    "outer2:"                                                   \
    "mov    edi, eax"                                           \
    "lea    esi, [eax+ecx]"                                     \
    "mov    ebp, ecx"                                           \
    "inner2:"                                                   \
    "movdqu xmm0, [edi]"                                        \
    "movdqu xmm1, [esi]"                                        \
    "movdqa xmm2, xmm0"                                         \
    "paddd  xmm0, xmm1"                                         \
    "psubd  xmm2, xmm1"                                         \
    "movdqu [edi], xmm0"                                        \
    "movdqu [esi], xmm2"                                        \
    "add    edi, 16"                                            \
    "add    esi, 16"                                            \
    "sub    ebp, 16"                                            \
    "jnz    inner2"                                             \
    "add    eax, ebx"                                           \
    "cmp    eax, edx"                                           \
    "jb     outer2"                                             \
    parm [eax] [edx] [ecx]                                      \
    modify [eax ebx ecx edx esi edi ebp];

/* Thin wrapper: a #pragma aux body expands at the CALL SITE, and the
   call site is src/fwht.c - which is not this TU. Same reason
   lz_amax32_sse2_w above exists. */
void lz_fwht_stage_sse2(int32_t *y, int n, int len) {
    lz_fwht_stage_sse2_asm(y, n, len);
}

/* ---- attention wsum's chunk fold: accf[d] += (float)acc32[d], 32 wide.
   cvtdq2ps converts four int32 at once and addps accumulates four
   floats. No MMX register is touched, which matters at this particular
   site: the caller has just paid an emms to leave the row-pair loop,
   and a cell that reached for cvtpi2ps would owe another.

   Why this cell is Watcom-only. gcc compiles the scalar loop to exactly
   these instructions - 19 cvtdq2ps in ops_gdn.o at -O2 on x86-64 - so
   an intrinsics twin would duplicate what the compiler already emits.
   Watcom emits zero, and runs lz_i32f's x87 split form instead: 32 fild
   and 78 fmul in the same object.

   The same value as the scalar form, not a close one.
   ops_kernel_shared.h's
   lz_i32f records 87,108,878 values compared against the correctly
   rounded conversion with 0 mismatches - the whole +-2^25 band, where
   every rounding decision lives. cvtdq2ps IS that conversion.

   Unaligned moves throughout: acc32 is a caller's automatic array and
   accf is a caller-owned buffer, neither aligned to 16 by anything.
   Two lanes in flight per block so the convert and the add of adjacent
   groups can pair. */
void lz_i32f_acc32_sse2_asm(float *accf, const int32_t *acc32);
#pragma aux lz_i32f_acc32_sse2_asm = \
    ".686" \
    "movdqu    xmm0, [edx]" \
    "movdqu    xmm1, [edx+16]" \
    "cvtdq2ps  xmm0, xmm0" \
    "cvtdq2ps  xmm1, xmm1" \
    "movups    xmm2, [eax]" \
    "movups    xmm3, [eax+16]" \
    "addps     xmm2, xmm0" \
    "addps     xmm3, xmm1" \
    "movups    [eax], xmm2" \
    "movups    [eax+16], xmm3" \
    "movdqu    xmm0, [edx+32]" \
    "movdqu    xmm1, [edx+48]" \
    "cvtdq2ps  xmm0, xmm0" \
    "cvtdq2ps  xmm1, xmm1" \
    "movups    xmm2, [eax+32]" \
    "movups    xmm3, [eax+48]" \
    "addps     xmm2, xmm0" \
    "addps     xmm3, xmm1" \
    "movups    [eax+32], xmm2" \
    "movups    [eax+48], xmm3" \
    "movdqu    xmm0, [edx+64]" \
    "movdqu    xmm1, [edx+80]" \
    "cvtdq2ps  xmm0, xmm0" \
    "cvtdq2ps  xmm1, xmm1" \
    "movups    xmm2, [eax+64]" \
    "movups    xmm3, [eax+80]" \
    "addps     xmm2, xmm0" \
    "addps     xmm3, xmm1" \
    "movups    [eax+64], xmm2" \
    "movups    [eax+80], xmm3" \
    "movdqu    xmm0, [edx+96]" \
    "movdqu    xmm1, [edx+112]" \
    "cvtdq2ps  xmm0, xmm0" \
    "cvtdq2ps  xmm1, xmm1" \
    "movups    xmm2, [eax+96]" \
    "movups    xmm3, [eax+112]" \
    "addps     xmm2, xmm0" \
    "addps     xmm3, xmm1" \
    "movups    [eax+96], xmm2" \
    "movups    [eax+112], xmm3" \
    __parm [__eax] [__edx] \
    __modify [8087]

/* The real function the other translation units call. A #pragma aux
   body expands at its call site and has no address of its own, so the
   wrapper is what crosses the TU boundary - same shape as
   ops_kernel_norm.h's _w wrappers, and same reason. */
void lz_i32f_acc32_simd(float *accf, const int32_t *acc32) {
    lz_i32f_acc32_sse2_asm(accf, acc32);
}

/* ---- the Q15 table interpolation, SSE2 tier, Watcom twin.
   gcc twin: lz_lerp_q15_simd, the intrinsics body in the #else arm.
   MMX twin: lz_lerp_q15_mmx (src/ops_mmx.c), which carries the
   precondition |b - a| <= 32767 and why `a` stays int32.

   32 ELEMENTS a call, four passes of eight. No emms: every instruction
   is the %xmm form and nothing here aliases the x87 stack, which is
   the one way this differs from the MMX twin beyond the width. */
extern void lz_lerp_q15_32_sse2_asm(const int32_t *a, const int32_t *b,
                                    const int32_t *frac, int32_t *out);
#pragma aux lz_lerp_q15_32_sse2_asm = \
    ".686"                              \
    "push    esi"                       \
    "mov     esi, 4"                    \
    "lerp2loop:"                        \
    "movdqu  xmm0, [eax]"               \
    "movdqu  xmm1, [eax+16]"            \
    "movdqu  xmm2, [edx]"               \
    "movdqu  xmm3, [edx+16]"            \
    "psubd   xmm2, xmm0"                \
    "psubd   xmm3, xmm1"                \
    "packssdw xmm2, xmm3"               \
    "movdqu  xmm4, [ebx]"               \
    "movdqu  xmm5, [ebx+16]"            \
    "packssdw xmm4, xmm5"               \
    "movdqa  xmm5, xmm2"                \
    "pmullw  xmm5, xmm4"                \
    "pmulhw  xmm2, xmm4"                \
    "movdqa  xmm6, xmm5"                \
    "punpcklwd xmm6, xmm2"              \
    "punpckhwd xmm5, xmm2"              \
    "psrad   xmm6, 15"                  \
    "psrad   xmm5, 15"                  \
    "paddd   xmm6, xmm0"                \
    "paddd   xmm5, xmm1"                \
    "movdqu  [ecx], xmm6"               \
    "movdqu  [ecx+16], xmm5"            \
    "add     eax, 32"                   \
    "add     edx, 32"                   \
    "add     ebx, 32"                   \
    "add     ecx, 32"                   \
    "dec     esi"                       \
    "jnz     lerp2loop"                 \
    "pop     esi"                       \
    __parm [__eax] [__edx] [__ebx] [__ecx] \
    __modify [__eax __ebx __ecx __edx];

void lz_lerp_q15_simd(const int32_t *a, const int32_t *b,
                      const int32_t *frac, int32_t *out, int n) {
    int k = 0;
    for (; k + 31 < n; k += 32)
        lz_lerp_q15_32_sse2_asm(a + k, b + k, frac + k, out + k);
    for (; k < n; k++)
        out[k] = a[k] + (((b[k] - a[k]) * frac[k]) >> 15);
}

/* ---- lz_exp_fixed's Q20 Taylor, SSE2 tier, Watcom arm.
   gcc twin: lz_exp_q20_simd, the intrinsics body in the #else arm,
   which carries the derivation - why all four multiplies are 32x32->64,
   why every operand is non-negative, why pslldq+por recombines the two
   pmuludq halves, and why cq's 2^20 rides in the second term's rounding
   constant instead of a fifth broadcast.

   The four constants live in %xmm4-7 across the whole call, which is
   the whole reason the folding above was worth doing: 32-bit mode has
   eight %xmm registers, the body needs four working ones, and a fifth
   constant would have meant reloading one of them every pass.

   The 64-bit halves are written as SHIFT EXPRESSIONS, not as the hex
   words they assemble to. A hand-computed 0x10000080 would be a table
   with no producer - the compiler is the producer here, and it also
   makes the two-power structure of the constant legible. */
static const struct {
    lz_i64  q[4];         /* first: 8-byte alignment without padding */
    int32_t d[8];
} exp_q20_k = {
    { LZ_I64_C(1) << 19, LZ_I64_C(1) << 19,
      (LZ_I64_C(1) << 39) + (LZ_I64_C(1) << 60),
      (LZ_I64_C(1) << 39) + (LZ_I64_C(1) << 60) },
    { 726817, 726817, 726817, 726817,
      251906, 251906, 251906, 251906 }
};

/* np is the number of FOUR-element passes, not elements, so this arm
   consumes exactly what the gcc one does - a fixed 32-per-call form
   would leave rows shorter than 32 entirely to the scalar tail on one
   compiler and not the other. Same numbers either way; different code
   reached, which is what a cross-toolchain identity gate is for. */
extern void lz_exp_q20_sse2_asm(const int32_t *tab, const int32_t *s,
                                int32_t *prod, const void *k, int np);
#pragma aux lz_exp_q20_sse2_asm = \
    ".686"                          \
    "movdqu    xmm0, [ecx]" \
    "movdqu    xmm1, [ecx+16]" \
    "movdqu    xmm2, [ecx+32]" \
    "movdqu    xmm3, [ecx+48]" \
    "expq20loop:" \
    "movdqu    xmm4, [edx]" \
    "movdqa    xmm5, xmm4" \
    "movdqa    xmm6, xmm4" \
    "pmuludq   xmm5, xmm2" \
    "psrldq    xmm6, 4" \
    "pmuludq   xmm6, xmm2" \
    "paddq     xmm5, xmm0" \
    "psrlq     xmm5, 20" \
    "paddq     xmm6, xmm0" \
    "psrlq     xmm6, 20" \
    "pslldq    xmm6, 4" \
    "por       xmm5, xmm6" \
    "movdqa    xmm7, xmm5" \
    "movdqa    xmm6, xmm4" \
    "psrldq    xmm6, 4" \
    "pmuludq   xmm6, xmm6" \
    "pmuludq   xmm4, xmm4" \
    "pslldq    xmm6, 4" \
    "por       xmm4, xmm6" \
    "movdqa    xmm5, xmm4" \
    "movdqa    xmm6, xmm4" \
    "pmuludq   xmm5, xmm3" \
    "psrldq    xmm6, 4" \
    "pmuludq   xmm6, xmm3" \
    "paddq     xmm5, xmm1" \
    "psrlq     xmm5, 40" \
    "paddq     xmm6, xmm1" \
    "psrlq     xmm6, 40" \
    "pslldq    xmm6, 4" \
    "por       xmm5, xmm6" \
    "paddd     xmm7, xmm5" \
    "movdqu    xmm4, [eax]" \
    "movdqa    xmm5, xmm4" \
    "pmuludq   xmm5, xmm7" \
    "psrldq    xmm4, 4" \
    "psrldq    xmm7, 4" \
    "paddq     xmm5, xmm0" \
    "psrlq     xmm5, 20" \
    "pmuludq   xmm4, xmm7" \
    "paddq     xmm4, xmm0" \
    "psrlq     xmm4, 20" \
    "pslldq    xmm4, 4" \
    "por       xmm5, xmm4" \
    "movdqu    [ebx], xmm5" \
    "add       eax, 16" \
    "add       edx, 16" \
    "add       ebx, 16" \
    "dec       esi" \
    "jnz       expq20loop" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [__eax __ebx __ecx __edx __esi];

void lz_exp_q20_simd(const int32_t *tab, const int32_t *s,
                     int32_t *prod, int n) {
    int k = n & ~3;
    if (k) lz_exp_q20_sse2_asm(tab, s, prod, &exp_q20_k, k >> 2);
    for (; k < n; k++) {
        int32_t sk = s[k], cq;
        cq = (1 << 20)
           + (int32_t)((((lz_i64)726817 * sk) + (1 << 19)) >> 20)
           + (int32_t)((((lz_i64)251906 * ((lz_i64)sk * sk))
                        + (LZ_I64_C(1) << 39)) >> 40);
        prod[k] = (int32_t)((((lz_i64)tab[k] * cq) + (1 << 19)) >> 20);
    }
}

/* ---- Q8 activation rounding, SSE2 tier - in this TU; the algorithm
   documentation is in src/ops_kernel_q8round.h. cvtps2dq converts 4
   floats at once - no MMX register touched, unlike the SSE1+MMX tier's
   cvtps2pi (src/ops_mmx_sse.c), so no emms is owed anywhere in this
   cell. */
extern void lz_q8round32_sse2_asm(const float *x, int8_t *o, const float *inv);
#pragma aux lz_q8round32_sse2_asm = \
    ".686" \
    "movss     xmm0, [ecx]" \
    "shufps    xmm0, xmm0, 0" \
    "pcmpeqw   xmm1, xmm1" \
    "psrlw     xmm1, 9" \
    "pxor      xmm2, xmm2" \
    "psubw     xmm2, xmm1" \
    "movups    xmm3, [eax]" \
    "movups    xmm4, [eax+16]" \
    "movups    xmm5, [eax+32]" \
    "movups    xmm6, [eax+48]" \
    "mulps     xmm3, xmm0" \
    "mulps     xmm4, xmm0" \
    "mulps     xmm5, xmm0" \
    "mulps     xmm6, xmm0" \
    "cvtps2dq  xmm3, xmm3" \
    "cvtps2dq  xmm4, xmm4" \
    "cvtps2dq  xmm5, xmm5" \
    "cvtps2dq  xmm6, xmm6" \
    "packssdw  xmm3, xmm4" \
    "packssdw  xmm5, xmm6" \
    "pminsw    xmm3, xmm1" \
    "pmaxsw    xmm3, xmm2" \
    "pminsw    xmm5, xmm1" \
    "pmaxsw    xmm5, xmm2" \
    "packsswb  xmm3, xmm5" \
    "movdqu    [edx], xmm3" \
    "movups    xmm3, [eax+64]" \
    "movups    xmm4, [eax+80]" \
    "movups    xmm5, [eax+96]" \
    "movups    xmm6, [eax+112]" \
    "mulps     xmm3, xmm0" \
    "mulps     xmm4, xmm0" \
    "mulps     xmm5, xmm0" \
    "mulps     xmm6, xmm0" \
    "cvtps2dq  xmm3, xmm3" \
    "cvtps2dq  xmm4, xmm4" \
    "cvtps2dq  xmm5, xmm5" \
    "cvtps2dq  xmm6, xmm6" \
    "packssdw  xmm3, xmm4" \
    "packssdw  xmm5, xmm6" \
    "pminsw    xmm3, xmm1" \
    "pmaxsw    xmm3, xmm2" \
    "pminsw    xmm5, xmm1" \
    "pmaxsw    xmm5, xmm2" \
    "packsswb  xmm3, xmm5" \
    "movdqu    [edx+16], xmm3" \
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
   in this TU. Algorithm documentation is in src/ops_kernel_dot.h -
   see src/ops_mmx.c's MMX-tier twin for the shared background (weight
   widening, why the paired kernel exists) and this file's own SSE2
   kernels above for why no emms is owed anywhere here. */
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
    ".686" \
    "pxor      xmm0, xmm0" \
    "movdqa    xmm1, [eax]" \
    "movdqa    xmm2, xmm0" \
    "movdqa    xmm3, xmm0" \
    "punpcklbw xmm2, xmm1" \
    "punpckhbw xmm3, xmm1" \
    "movdqu    xmm4, [edx]" \
    "pmaddwd   xmm2, xmm4" \
    "movdqu    xmm5, [edx+16]" \
    "pmaddwd   xmm3, xmm5" \
    "movdqa    xmm6, xmm2" \
    "movdqa    xmm1, [eax+16]" \
    "paddd     xmm6, xmm3" \
    "movdqa    xmm2, xmm0" \
    "movdqa    xmm3, xmm0" \
    "punpcklbw xmm2, xmm1" \
    "punpckhbw xmm3, xmm1" \
    "movdqu    xmm4, [edx+32]" \
    "pmaddwd   xmm2, xmm4" \
    "movdqu    xmm5, [edx+48]" \
    "pmaddwd   xmm3, xmm5" \
    "paddd     xmm6, xmm2" \
    "paddd     xmm6, xmm3" \
    "movdqa    xmm4, xmm6" \
    "psrldq    xmm4, 8" \
    "paddd     xmm6, xmm4" \
    "movdqa    xmm4, xmm6" \
    "psrldq    xmm4, 4" \
    "paddd     xmm6, xmm4" \
    "movd      eax, xmm6" \
    __parm [__eax] [__edx]          \
    __value [__eax]                 \
    __modify [8087]

/* Two tokens, one weight side - the Watcom twin of part32_x16_2
   (src/ops_kernel_dot_sse2.h), and the SSE2 member of the g_pair family.
 *
 * The four unpacked weight halves stay live across both tokens, in
 * xmm1-xmm4, which is the whole saving: two movdqu and four punpck paid
 * once where two single-token calls pay them twice. Only the activation
 * loads and the pmaddwd double.
 *
 * REGISTER BUDGET, and it is exactly full. xmm1-4 the weight halves,
 * xmm5/xmm6 the two accumulators, xmm0 the weight-copy temp that
 * pmaddwd destroys, and xmm7 the zero - which is dead after the last
 * punpck, so it is reused as the activation temp. Nothing spills.
 *
 * Activations go through movdqu, never a pmaddwd memory operand: the
 * single-token body above does the same, because SSE2's integer ops
 * fault on a 16-byte-unaligned memory operand and xw carries no such
 * guarantee. That is also why the weight copy is register-register.
 *
 * Bit-identical to two lz_dot32_x16_sse2_asm calls: each token
 * accumulates in its own register in the same order, and the two folds
 * are the same pair of psrldq/paddd the single form ends with. */
extern void lz_dot32_x16_sse2_asm_2(const int8_t *w, const int16_t *xa,
                                    const int16_t *xb,
                                    int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_x16_sse2_asm_2 = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "movdqa    xmm1, [eax]" \
    "movdqa    xmm2, xmm0" \
    "punpcklbw xmm2, xmm1" \
    "movdqa    xmm3, xmm0" \
    "punpckhbw xmm3, xmm1" \
    "movdqa    xmm1, [eax+16]" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm1" \
    "movdqa    xmm5, xmm0" \
    "punpckhbw xmm5, xmm1" \
    "movdqu    xmm0, [edx]" \
    "movdqa    xmm6, xmm2" \
    "pmaddwd   xmm6, xmm0" \
    "movdqu    xmm0, [edx+16]" \
    "movdqa    xmm1, xmm3" \
    "pmaddwd   xmm1, xmm0" \
    "movdqu    xmm0, [edx+32]" \
    "movdqa    xmm7, xmm2" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm4" \
    "pmaddwd   xmm1, xmm0" \
    "movdqu    xmm0, [edx+48]" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm5" \
    "pmaddwd   xmm1, xmm0" \
    "movdqu    xmm0, [ebx]" \
    "pmaddwd   xmm7, xmm0" \
    "paddd     xmm6, xmm1" \
    "movdqu    xmm0, [ebx+16]" \
    "movdqa    xmm1, xmm3" \
    "pmaddwd   xmm1, xmm0" \
    "movdqu    xmm0, [ebx+32]" \
    "paddd     xmm7, xmm1" \
    "movdqa    xmm1, xmm4" \
    "pmaddwd   xmm1, xmm0" \
    "movdqu    xmm0, [ebx+48]" \
    "paddd     xmm7, xmm1" \
    "movdqa    xmm1, xmm5" \
    "pmaddwd   xmm1, xmm0" \
    "paddd     xmm7, xmm1" \
    "movdqa    xmm1, xmm6" \
    "psrldq    xmm1, 8" \
    "paddd     xmm6, xmm1" \
    "movdqa    xmm1, xmm6" \
    "psrldq    xmm1, 4" \
    "paddd     xmm6, xmm1" \
    "movd      [ecx], xmm6" \
    "movdqa    xmm1, xmm7" \
    "psrldq    xmm1, 8" \
    "paddd     xmm7, xmm1" \
    "movdqa    xmm1, xmm7" \
    "psrldq    xmm1, 4" \
    "paddd     xmm7, xmm1" \
    "movd      [esi], xmm7" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [8087]

/* The paired kernel's address, for the reason its MMX twin has one. */
void dot32_x16_sse2a_2(const int8_t *w, const int16_t *xa,
                       const int16_t *xb, int32_t *oa, int32_t *ob) {
    lz_dot32_x16_sse2_asm_2(w, xa, xb, oa, ob);
}

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
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "pxor      xmm2, xmm2" \
    "movdqa    xmm3, [eax]" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm3" \
    "movdqu    xmm5, [ecx]" \
    "pmaddwd   xmm4, xmm5" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+16]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm1, xmm4" \
    "movdqa    xmm3, [eax+16]" \
    "paddd     xmm1, xmm6" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm3" \
    "movdqu    xmm5, [ecx+32]" \
    "pmaddwd   xmm4, xmm5" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+48]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm1, xmm4" \
    "movdqa    xmm3, [eax+32]" \
    "paddd     xmm1, xmm6" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm3" \
    "movdqu    xmm5, [ecx+64]" \
    "pmaddwd   xmm4, xmm5" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+80]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm2, xmm4" \
    "movdqa    xmm3, [eax+48]" \
    "paddd     xmm2, xmm6" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm3" \
    "movdqu    xmm5, [ecx+96]" \
    "pmaddwd   xmm4, xmm5" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+112]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm2, xmm4" \
    "movdqa    xmm3, [eax+64]" \
    "paddd     xmm2, xmm6" \
    "movdqa    xmm5, xmm1" \
    "punpckldq xmm1, xmm2" \
    "punpckhdq xmm5, xmm2" \
    "paddd     xmm1, xmm5" \
    "movdqa    xmm5, xmm1" \
    "psrldq    xmm5, 8" \
    "paddd     xmm1, xmm5" \
    "movq      [ebx], xmm1" \
    "pxor      xmm1, xmm1" \
    "pxor      xmm2, xmm2" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm3" \
    "movdqu    xmm5, [ecx+128]" \
    "pmaddwd   xmm4, xmm5" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+144]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm1, xmm4" \
    "movdqa    xmm3, [eax+80]" \
    "paddd     xmm1, xmm6" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm3" \
    "movdqu    xmm5, [ecx+160]" \
    "pmaddwd   xmm4, xmm5" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+176]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm1, xmm4" \
    "movdqa    xmm3, [eax+96]" \
    "paddd     xmm1, xmm6" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm3" \
    "movdqu    xmm5, [ecx+192]" \
    "pmaddwd   xmm4, xmm5" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+208]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm2, xmm4" \
    "movdqa    xmm3, [eax+112]" \
    "paddd     xmm2, xmm6" \
    "movdqa    xmm4, xmm0" \
    "punpcklbw xmm4, xmm3" \
    "movdqu    xmm5, [ecx+224]" \
    "pmaddwd   xmm4, xmm5" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+240]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm2, xmm4" \
    "movdqa    xmm5, xmm1" \
    "paddd     xmm2, xmm6" \
    "punpckldq xmm1, xmm2" \
    "punpckhdq xmm5, xmm2" \
    "paddd     xmm1, xmm5" \
    "movdqa    xmm5, xmm1" \
    "psrldq    xmm5, 8" \
    "paddd     xmm1, xmm5" \
    "movq      [ebx+8], xmm1" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

/* Real, addressable twin of the above. A #pragma aux body has no
   address, and row_q8_sse2_asm below reaches the kernel by name inside
   its own loop, so nothing outside this file could call it - including
   build/dot128_x16_probe.c, which holds the MMX group kernel against
   the MMX and SSE2 per-32 ones and had no way to ask the same
   questions of this tier. Same reason lz_dot128_x16_mmx_w exists. */
void lz_dot128_x16_sse2_w(const int8_t *w, const int16_t *x,
                          int32_t *out4) {
    lz_dot128_x16_sse2_asm(w, x, out4);
}

/* row_q8_sse2_asm: the real call boundary from its own caller
   (matmul_q8_impl, src/ops.c, dispatched once per matmul via
   lz_row_pick's LZ_ROW_Q8 table). The whole function, together with
   the leaf kernels it calls direct-by-name inside its own loop, adds
   no new call sites. No pairing logic here (unlike the MMX row
   function): SSE2's group kernel is already the fast path, and pairing
   is not wired in for this tier. */
void row_q8_sse2_asm(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    const int8_t *wend = (const int8_t *)c->pf_end;
    int tk, g;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        /* Strength-reduced by hand: xwt/acc are kept as running pointers
           (xwt = xw + tk*in_dim, acc = acc32 + tk*nb) instead of re-derived
           from tk each iteration. Watcom's own strength reduction of the
           derived form emits `imul reg,reg,0` for the tk=0 initial value
           (0*nb), an 11-15-insn multiply instead of a mov - see
           build/imul0_probe.c. The running form drops it and stays
           bit-identical: same addresses, same order. */
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair before grouping, matching the gcc twin above and the MMX
               row kernels: the pair keeps the four unpacked weight halves
               live across two tokens, the group only amortizes the per-32
               call. An odd nt leaves its last token on the paths below. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                int gp;
                for (gp = 0; gp < c->nb; gp++) {
                    const int8_t *pf_ = wr + (size_t)(gp + LZ_PF_DIST) * 32;
                    if (pf_ < wend) {
                        int pf_m_ = lz_prefetch_mode();
                        if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);
                        else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);
                        else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);
                        else                          LZ_PFI_NTA(pf_);
                    }
                    lz_dot32_x16_sse2_asm_2(wr + (size_t)gp * 32,
                                            xwt + (size_t)gp * 32,
                                            xw2 + (size_t)gp * 32,
                                            acc + gp, accb + gp);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
#if LZ_SSE2_GROUP
            if ((c->nb & 3) == 0) LZ_Q8_GROUP4(lz_dot128_x16_sse2_asm);
#else
            /* LZ_SSE2_GROUP=0 keeps the 128-element step on the MMX kernel;
               the knob exists so an SSE2 CPU can be measured on either.
               lz_dot128_x16_mmx_w is src/ops_mmx.c's real, addressable
               twin of the #pragma aux lz_dot128_x16_asm - needed cross-TU
               here for exactly this A/B control build, since a #pragma aux
               body cannot have its address taken and this file does not
               carry the MMX pragma itself. */
            if ((c->nb & 3) == 0) LZ_Q8_GROUP4(lz_dot128_x16_mmx_w);
#endif /* LZ_SSE2_GROUP */
            else               LZ_Q8_PFSEL(dot32_x16_sse2a);
        }
    }
}

/* ---- Q4_1 32/128-element dot products and row kernel, SSE2 tier -
   in this TU. */

/* T42-tier Q4_1 (same: 128-bit, no EMMS needed, same name + signature
   reused). */
extern int32_t lz_dot32_q41_sse2_asm(const unsigned char *w, const int16_t *x);
#pragma aux lz_dot32_q41_sse2_asm = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pcmpeqb   xmm1, xmm1" \
    "psrlw     xmm1, 12" \
    "packuswb  xmm1, xmm1" \
    "movdqa    xmm2, [eax]" \
    "movdqa    xmm3, xmm2" \
    "pand      xmm2, xmm1" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm1" \
    "movdqa    xmm4, xmm0" \
    "movdqa    xmm5, xmm0" \
    "punpcklbw xmm4, xmm2" \
    "punpcklbw xmm5, xmm3" \
    "movdqu    xmm6, [edx]" \
    "pmaddwd   xmm4, xmm6" \
    "movdqu    xmm6, [edx+32]" \
    "pmaddwd   xmm5, xmm6" \
    "movdqa    xmm7, xmm4" \
    "movdqa    xmm4, xmm0" \
    "paddd     xmm7, xmm5" \
    "movdqa    xmm5, xmm0" \
    "punpckhbw xmm4, xmm2" \
    "punpckhbw xmm5, xmm3" \
    "movdqu    xmm6, [edx+16]" \
    "pmaddwd   xmm4, xmm6" \
    "movdqu    xmm6, [edx+48]" \
    "pmaddwd   xmm5, xmm6" \
    "paddd     xmm7, xmm4" \
    "paddd     xmm7, xmm5" \
    "movdqa    xmm6, xmm7" \
    "psrldq    xmm6, 8" \
    "paddd     xmm7, xmm6" \
    "movdqa    xmm6, xmm7" \
    "psrldq    xmm6, 4" \
    "paddd     xmm7, xmm6" \
    "movd      eax, xmm7" \
    __parm [__eax] [__edx]          \
    __value [__eax]                 \
    __modify [8087]

/* Not static: already the thin wrapper a probe needs. */
int32_t dot32_q41_sse2a(const unsigned char *w, const int16_t *x) {
    return lz_dot32_q41_sse2_asm(w, x);
}

/* Two tokens, one weight side - the Watcom twin of part32_q41_2.
 *
 * The register strategy differs from the q8_0 pair, forced by this
 * format. There the four unpacked halves stay live in xmm1-4 across
 * both tokens. Here the nibble mask (xmm3) and the zero (xmm7) are both
 * needed through the unpacks, and lo/hi occupy two more, so four live
 * halves would not fit. Instead each quarter is unpacked ONCE into xmm1
 * and used against both tokens before the next - which shares exactly
 * what matters: the load, the two masks, the shift and the four
 * punpck. Only the activation loads and the pmaddwd double.
 *
 * xmm3 is reused as the multiply temp after the two pand, since the
 * mask is dead by then and pmaddwd destroys its first operand.
 *
 * movdqu for activations, never a pmaddwd memory operand - same
 * alignment reason as the single-token body above. */
extern void lz_dot32_q41_sse2_asm_2(const unsigned char *w,
                                    const int16_t *xa, const int16_t *xb,
                                    int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_q41_sse2_asm_2 = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "pxor      xmm2, xmm2" \
    "pcmpeqb   xmm3, xmm3" \
    "psrlw     xmm3, 12" \
    "packuswb  xmm3, xmm3" \
    "movdqa    xmm4, [eax]" \
    "movdqa    xmm5, xmm4" \
    "pand      xmm4, xmm3" \
    "psrlw     xmm5, 4" \
    "pand      xmm5, xmm3" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm6, xmm4" \
    "movdqu    xmm7, [edx]" \
    "movdqa    xmm3, xmm6" \
    "pmaddwd   xmm3, xmm7" \
    "movdqu    xmm7, [ebx]" \
    "paddd     xmm1, xmm3" \
    "movdqa    xmm3, xmm6" \
    "pmaddwd   xmm3, xmm7" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm4" \
    "paddd     xmm2, xmm3" \
    "movdqu    xmm7, [edx+16]" \
    "movdqa    xmm3, xmm6" \
    "pmaddwd   xmm3, xmm7" \
    "movdqu    xmm7, [ebx+16]" \
    "paddd     xmm1, xmm3" \
    "movdqa    xmm3, xmm6" \
    "pmaddwd   xmm3, xmm7" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm6, xmm5" \
    "paddd     xmm2, xmm3" \
    "movdqu    xmm7, [edx+32]" \
    "movdqa    xmm3, xmm6" \
    "pmaddwd   xmm3, xmm7" \
    "movdqu    xmm7, [ebx+32]" \
    "paddd     xmm1, xmm3" \
    "movdqa    xmm3, xmm6" \
    "pmaddwd   xmm3, xmm7" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm6, xmm5" \
    "paddd     xmm2, xmm3" \
    "movdqu    xmm7, [edx+48]" \
    "movdqa    xmm3, xmm6" \
    "pmaddwd   xmm3, xmm7" \
    "movdqu    xmm7, [ebx+48]" \
    "paddd     xmm1, xmm3" \
    "movdqa    xmm3, xmm6" \
    "pmaddwd   xmm3, xmm7" \
    "movdqa    xmm7, xmm1" \
    "psrldq    xmm7, 8" \
    "paddd     xmm2, xmm3" \
    "paddd     xmm1, xmm7" \
    "movdqa    xmm7, xmm1" \
    "psrldq    xmm7, 4" \
    "paddd     xmm1, xmm7" \
    "movd      [ecx], xmm1" \
    "movdqa    xmm7, xmm2" \
    "psrldq    xmm7, 8" \
    "paddd     xmm2, xmm7" \
    "movdqa    xmm7, xmm2" \
    "psrldq    xmm7, 4" \
    "paddd     xmm2, xmm7" \
    "movd      [esi], xmm2" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [8087]

/* The paired kernel's address, for the reason its MMX twin now has
   one. */
void dot32_q41_sse2a_2(const unsigned char *w, const int16_t *xa,
                       const int16_t *xb, int32_t *oa, int32_t *ob) {
    lz_dot32_q41_sse2_asm_2(w, xa, xb, oa, ob);
}

extern void lz_dot128_q41_sse2_asm(const unsigned char *w,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q41_sse2_asm = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pcmpeqb   xmm1, xmm1" \
    "psrlw     xmm1, 12" \
    "packuswb  xmm1, xmm1" \
    "movdqa    xmm2, [eax]" \
    "movdqa    xmm3, xmm2" \
    "pand      xmm2, xmm1" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm1" \
    "pxor      xmm4, xmm4" \
    "movdqa    xmm5, xmm0" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm5, xmm2" \
    "punpcklbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+32]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm4, xmm5" \
    "movdqa    xmm5, xmm0" \
    "paddd     xmm4, xmm6" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm5, xmm2" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+16]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+48]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm4, xmm5" \
    "paddd     xmm4, xmm6" \
    "movdqa    xmm7, xmm4" \
    "psrldq    xmm7, 8" \
    "paddd     xmm4, xmm7" \
    "movdqa    xmm7, xmm4" \
    "psrldq    xmm7, 4" \
    "paddd     xmm4, xmm7" \
    "movd      [ebx], xmm4" \
    "movdqa    xmm2, [eax+16]" \
    "movdqa    xmm3, xmm2" \
    "pand      xmm2, xmm1" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm1" \
    "pxor      xmm4, xmm4" \
    "movdqa    xmm5, xmm0" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm5, xmm2" \
    "punpcklbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+64]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+96]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm4, xmm5" \
    "movdqa    xmm5, xmm0" \
    "paddd     xmm4, xmm6" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm5, xmm2" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+80]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+112]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm4, xmm5" \
    "paddd     xmm4, xmm6" \
    "movdqa    xmm7, xmm4" \
    "psrldq    xmm7, 8" \
    "paddd     xmm4, xmm7" \
    "movdqa    xmm7, xmm4" \
    "psrldq    xmm7, 4" \
    "paddd     xmm4, xmm7" \
    "movd      [ebx+4], xmm4" \
    "movdqa    xmm2, [eax+32]" \
    "movdqa    xmm3, xmm2" \
    "pand      xmm2, xmm1" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm1" \
    "pxor      xmm4, xmm4" \
    "movdqa    xmm5, xmm0" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm5, xmm2" \
    "punpcklbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+128]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+160]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm4, xmm5" \
    "movdqa    xmm5, xmm0" \
    "paddd     xmm4, xmm6" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm5, xmm2" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+144]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+176]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm4, xmm5" \
    "paddd     xmm4, xmm6" \
    "movdqa    xmm7, xmm4" \
    "psrldq    xmm7, 8" \
    "paddd     xmm4, xmm7" \
    "movdqa    xmm7, xmm4" \
    "psrldq    xmm7, 4" \
    "paddd     xmm4, xmm7" \
    "movd      [ebx+8], xmm4" \
    "movdqa    xmm2, [eax+48]" \
    "movdqa    xmm3, xmm2" \
    "pand      xmm2, xmm1" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm1" \
    "pxor      xmm4, xmm4" \
    "movdqa    xmm5, xmm0" \
    "movdqa    xmm6, xmm0" \
    "punpcklbw xmm5, xmm2" \
    "punpcklbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+192]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+224]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm4, xmm5" \
    "movdqa    xmm5, xmm0" \
    "paddd     xmm4, xmm6" \
    "movdqa    xmm6, xmm0" \
    "punpckhbw xmm5, xmm2" \
    "punpckhbw xmm6, xmm3" \
    "movdqu    xmm7, [ecx+208]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+240]" \
    "pmaddwd   xmm6, xmm7" \
    "paddd     xmm4, xmm5" \
    "paddd     xmm4, xmm6" \
    "movdqa    xmm7, xmm4" \
    "psrldq    xmm7, 8" \
    "paddd     xmm4, xmm7" \
    "movdqa    xmm7, xmm4" \
    "psrldq    xmm7, 4" \
    "paddd     xmm4, xmm7" \
    "movd      [ebx+12], xmm4" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

/* Real, addressable twin, on the same argument as
   lz_dot128_x16_sse2_w: the pragma has no address and row_q41_sse2_asm
   reaches it by name, so a probe in another TU cannot ask this tier
   anything. build/dot128_q41_probe.c needs it as the oracle for the
   MMX group kernel - the MMX per-32 kernel shares that kernel's body
   and would agree with it after a shared mistake. */
void lz_dot128_q41_sse2_w(const unsigned char *w, const int16_t *x,
                          int32_t *out4) {
    lz_dot128_q41_sse2_asm(w, x, out4);
}

/* row_q41_sse2_asm: ALREADY the real call boundary from its own caller
   (matmul_q41_impl, src/ops.c, dispatched once per matmul via
   LZ_ROW_Q41). No pairing logic, same as Q8_0's SSE2 row function. */
void row_q41_sse2_asm(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *wend = (const unsigned char *)c->pf_end;
    int tk, s;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair before grouping, matching the gcc twin: the pair shares
               the load, both masks, the shift and the four punpck across
               two tokens; the group only amortizes the per-32 call. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                int sp;
                for (sp = 0; sp < c->nb; sp++) {
                    /* Prefetch spelled out rather than through LZ_Q41_PF:
                       that macro is not on this TU's include path, and the
                       q8_0 pair above does the same for the same reason. */
                    const unsigned char *pf_ = wn + (size_t)(sp + LZ_PF_DIST) * 16;
                    if (pf_ < wend) {
                        int pf_m_ = lz_prefetch_mode();
                        if (pf_m_ == LZ_PF_AMD)       LZ_PFI_AMD(pf_);
                        else if (pf_m_ == LZ_PF_LOAD) LZ_PFI_LOAD(pf_);
                        else if (pf_m_ == LZ_PF_NONE) LZ_PFI_NONE(pf_);
                        else                          LZ_PFI_NTA(pf_);
                    }
                    lz_dot32_q41_sse2_asm_2(wn + (size_t)sp * 16,
                                            xwt + (size_t)sp * 32,
                                            xw2 + (size_t)sp * 32,
                                            acc + sp, accb + sp);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
#if LZ_SSE2_GROUP
            if ((c->nb & 3) == 0) LZ_Q41_GROUP4(lz_dot128_q41_sse2_asm);
#else
            /* lz_dot128_q41_mmx_w is src/ops_mmx.c's real wrapper around
               the MMX #pragma aux group kernel. */
            if ((c->nb & 3) == 0) LZ_Q41_GROUP4(lz_dot128_q41_mmx_w);
#endif /* LZ_SSE2_GROUP */
            else               LZ_Q41_PFSEL(dot32_q41_sse2a);
        }
    }
}

/* ---- T2 (ternary) dot product and row kernel, SSE2 tier - in this
   TU. No 128-wide group kernel exists for this format, so no
   LZ_SSE2_GROUP=0 routing is needed here. */
extern int32_t lz_dot32_t2_sse2_asm(const unsigned char *w2,
                                    const int16_t *x);
#pragma aux lz_dot32_t2_sse2_asm = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "pcmpeqb   xmm2, xmm2" \
    "psrlw     xmm2, 14" \
    "packuswb  xmm2, xmm2" \
    "movq      xmm3, [edx]" \
    "movdqa    xmm4, xmm3" \
    "pand      xmm4, xmm2" \
    "movdqa    xmm5, xmm3" \
    "psrlw     xmm5, 2" \
    "pand      xmm5, xmm2" \
    "punpcklqdq xmm4, xmm5" \
    "movdqa    xmm5, xmm3" \
    "psrlw     xmm5, 4" \
    "pand      xmm5, xmm2" \
    "movdqa    xmm6, xmm3" \
    "psrlw     xmm6, 6" \
    "pand      xmm6, xmm2" \
    "punpcklqdq xmm5, xmm6" \
    "pxor      xmm2, xmm2" \
    "movdqa    xmm6, xmm0" \
    "movdqa    xmm3, xmm0" \
    "punpcklbw xmm6, xmm4" \
    "punpcklbw xmm3, xmm5" \
    "movdqu    xmm7, [ecx+0]" \
    "pmaddwd   xmm6, xmm7" \
    "movdqu    xmm7, [ecx+32]" \
    "pmaddwd   xmm3, xmm7" \
    "paddd     xmm1, xmm6" \
    "movdqa    xmm6, xmm0" \
    "paddd     xmm2, xmm3" \
    "movdqa    xmm3, xmm0" \
    "punpckhbw xmm6, xmm4" \
    "punpckhbw xmm3, xmm5" \
    "movdqu    xmm7, [ecx+16]" \
    "pmaddwd   xmm6, xmm7" \
    "movdqu    xmm7, [ecx+48]" \
    "pmaddwd   xmm3, xmm7" \
    "paddd     xmm1, xmm6" \
    "paddd     xmm2, xmm3" \
    "paddd     xmm1, xmm2" \
    "movdqa    xmm4, xmm1" \
    "psrldq    xmm4, 8" \
    "paddd     xmm1, xmm4" \
    "movdqa    xmm4, xmm1" \
    "psrldq    xmm4, 4" \
    "paddd     xmm1, xmm4" \
    "movd      eax, xmm1" \
    __parm [__edx] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* Two tokens, one ternary decode - the Watcom twin of part32_t2_2.

   This one needed no restructure, unlike q6_1's. The single-token body
   above already builds BOTH code vectors up front (xmm0 lo, xmm1 hi)
   before it touches an activation, so the 2-bit plane in xmm3 and the
   0x03 mask in xmm7 are both dead by then. xmm3 becomes the second
   accumulator and xmm7 the multiply temp; nothing else moves.

   Each quarter is unpacked ONCE into xmm2 and used against both tokens,
   so the pair shares the load, four masks, three shifts, two merges and
   four unpacks - fourteen instructions, more than q4_1's eight. That is
   this tier's decode; the MMX one is a zero register and three shifts,
   where a pair would be worth almost nothing.

   movdqu for activations, never a pmaddwd memory operand - the
   alignment reason the rest of this file carries. */
extern void lz_dot32_t2_sse2_asm_2(const unsigned char *w2,
                                   const int16_t *xa, const int16_t *xb,
                                   int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_t2_sse2_asm_2 = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "pxor      xmm2, xmm2" \
    "pcmpeqb   xmm3, xmm3" \
    "psrlw     xmm3, 14" \
    "packuswb  xmm3, xmm3" \
    "movq      xmm4, [edx]" \
    "movdqa    xmm5, xmm4" \
    "pand      xmm5, xmm3" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 2" \
    "pand      xmm6, xmm3" \
    "punpcklqdq xmm5, xmm6" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 4" \
    "pand      xmm6, xmm3" \
    "movdqa    xmm7, xmm4" \
    "psrlw     xmm7, 6" \
    "pand      xmm7, xmm3" \
    "punpcklqdq xmm6, xmm7" \
    "movdqa    xmm7, xmm0" \
    "punpcklbw xmm7, xmm5" \
    "movdqu    xmm4, [ecx]" \
    "movdqa    xmm3, xmm7" \
    "pmaddwd   xmm3, xmm4" \
    "movdqu    xmm4, [ebx]" \
    "paddd     xmm1, xmm3" \
    "movdqa    xmm3, xmm7" \
    "pmaddwd   xmm3, xmm4" \
    "movdqa    xmm7, xmm0" \
    "punpckhbw xmm7, xmm5" \
    "paddd     xmm2, xmm3" \
    "movdqu    xmm4, [ecx+16]" \
    "movdqa    xmm3, xmm7" \
    "pmaddwd   xmm3, xmm4" \
    "movdqu    xmm4, [ebx+16]" \
    "paddd     xmm1, xmm3" \
    "movdqa    xmm3, xmm7" \
    "pmaddwd   xmm3, xmm4" \
    "movdqa    xmm7, xmm0" \
    "punpcklbw xmm7, xmm6" \
    "paddd     xmm2, xmm3" \
    "movdqu    xmm4, [ecx+32]" \
    "movdqa    xmm3, xmm7" \
    "pmaddwd   xmm3, xmm4" \
    "movdqu    xmm4, [ebx+32]" \
    "paddd     xmm1, xmm3" \
    "movdqa    xmm3, xmm7" \
    "pmaddwd   xmm3, xmm4" \
    "movdqa    xmm7, xmm0" \
    "punpckhbw xmm7, xmm6" \
    "paddd     xmm2, xmm3" \
    "movdqu    xmm4, [ecx+48]" \
    "movdqa    xmm3, xmm7" \
    "pmaddwd   xmm3, xmm4" \
    "movdqu    xmm4, [ebx+48]" \
    "paddd     xmm1, xmm3" \
    "movdqa    xmm3, xmm7" \
    "pmaddwd   xmm3, xmm4" \
    "movdqa    xmm4, xmm1" \
    "psrldq    xmm4, 8" \
    "paddd     xmm2, xmm3" \
    "paddd     xmm1, xmm4" \
    "movdqa    xmm4, xmm1" \
    "psrldq    xmm4, 4" \
    "paddd     xmm1, xmm4" \
    "movd      [esi], xmm1" \
    "movdqa    xmm4, xmm2" \
    "psrldq    xmm4, 8" \
    "paddd     xmm2, xmm4" \
    "movdqa    xmm4, xmm2" \
    "psrldq    xmm4, 4" \
    "paddd     xmm2, xmm4" \
    "movd      [edi], xmm2" \
    __parm [__edx] [__ecx] [__ebx] [__esi] [__edi] \
    __modify [8087]

/* Addresses for the two T2 kernels above. Neither had one, and the
   consequence was the same as Q16_0's: T2 has an MMX tier and an SSE2
   tier that are required to agree bit for bit, and nothing had ever
   compared them - build/dot128_t2_probe.c held the MMX group kernel
   against the MMX per-32 kernel and stopped there, both sides of it
   inside ops_mmx.c.

   The paired form gets one too. Its claim - one weight unpack serving
   two activation vectors - had never been checked against anything
   outside its own file either. */
int32_t lz_dot32_t2_sse2_w(const unsigned char *w2, const int16_t *x) {
    return lz_dot32_t2_sse2_asm(w2, x);
}

void lz_dot32_t2_sse2_2_w(const unsigned char *w2, const int16_t *xa,
                          const int16_t *xb, int32_t *oa, int32_t *ob) {
    lz_dot32_t2_sse2_asm_2(w2, xa, xb, oa, ob);
}

/* row_t2_sse2_asm: ALREADY the real call boundary (matmul_t2_impl,
   src/ops.c, dispatched once per matmul via LZ_ROW_T2). Moving the
   whole function here adds NO new call sites. */
void row_t2_sse2_asm(const lz_row_ctx *c) {
    const unsigned char *p2 = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair, matching the gcc twin: one ternary decode for two
               tokens. Fourteen instructions of weight side at this tier,
               against three at the MMX one. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (s = 0; s < c->nb; s++)
                    lz_dot32_t2_sse2_asm_2(p2 + (size_t)s * 8,
                                           xwt + (size_t)s * 32,
                                           xw2 + (size_t)s * 32,
                                           acc + s, accb + s);
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            for (s = 0; s < c->nb; s++)
                acc[s] = lz_dot32_t2_sse2_asm(p2 + (size_t)s * 8,
                                              xwt + (size_t)s * 32);
        }
    }
}

/* Two tokens, one 6-bit reassembly - the Watcom twin of part32_q61_2,
   and the pair that saves most of the five formats.

   This is a restructure of the single-token body below, not a
   transcription of it.
   That one interleaves: build the lo byte-vector, use it, then reuse
   the planes and the two masks to build hi. All eight %xmm are live
   while it does so, which leaves nowhere for a second accumulator.

   Here BOTH byte-vectors are reassembled first, into xmm2 (lo) and
   xmm3 (hi). That retires the 4-bit plane, the 2-bit plane and both
   masks in one go, and the freed registers become the second
   accumulator and the temps. Same arithmetic, different order of the
   same steps - the reassembly is line for line what the single form
   does, which is what keeps the 6-bit value identical.

   After the split: xmm2/xmm3 the byte-vectors, xmm4/xmm5 the two
   accumulators, xmm0 the shared unpack, xmm1 the activation, xmm6 the
   multiply temp pmaddwd destroys.

   movdqu for activations, never a pmaddwd memory operand - the
   alignment reason the other kernels in this file carry. */
extern void lz_dot32_q61_sse2_asm_2(const unsigned char *w4,
                                    const unsigned char *w2,
                                    const int16_t *xa, const int16_t *xb,
                                    int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_q61_sse2_asm_2 = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "pcmpeqb   xmm2, xmm2" \
    "psrlw     xmm2, 12" \
    "packuswb  xmm2, xmm2" \
    "pcmpeqb   xmm3, xmm3" \
    "psrlw     xmm3, 14" \
    "packuswb  xmm3, xmm3" \
    "movdqa    xmm4, [eax]" \
    "movq      xmm5, [edx]" \
    "movdqa    xmm6, xmm5" \
    "pand      xmm6, xmm3" \
    "movdqa    xmm7, xmm5" \
    "psrlw     xmm7, 2" \
    "pand      xmm7, xmm3" \
    "punpcklqdq xmm6, xmm7" \
    "psllw     xmm6, 4" \
    "movdqa    xmm7, xmm4" \
    "pand      xmm7, xmm2" \
    "por       xmm6, xmm7" \
    "movdqa    xmm7, xmm5" \
    "psrlw     xmm7, 4" \
    "pand      xmm7, xmm3" \
    "psrlw     xmm5, 6" \
    "pand      xmm5, xmm3" \
    "punpcklqdq xmm7, xmm5" \
    "psllw     xmm7, 4" \
    "psrlw     xmm4, 4" \
    "pand      xmm4, xmm2" \
    "por       xmm7, xmm4" \
    "pxor      xmm4, xmm4" \
    "punpcklbw xmm4, xmm6" \
    "movdqu    xmm5, [ebx]" \
    "movdqa    xmm3, xmm4" \
    "pmaddwd   xmm3, xmm5" \
    "movdqu    xmm5, [ecx]" \
    "paddd     xmm0, xmm3" \
    "movdqa    xmm3, xmm4" \
    "pmaddwd   xmm3, xmm5" \
    "pxor      xmm4, xmm4" \
    "punpckhbw xmm4, xmm6" \
    "paddd     xmm1, xmm3" \
    "movdqu    xmm5, [ebx+16]" \
    "movdqa    xmm3, xmm4" \
    "pmaddwd   xmm3, xmm5" \
    "movdqu    xmm5, [ecx+16]" \
    "paddd     xmm0, xmm3" \
    "movdqa    xmm3, xmm4" \
    "pmaddwd   xmm3, xmm5" \
    "pxor      xmm4, xmm4" \
    "punpcklbw xmm4, xmm7" \
    "paddd     xmm1, xmm3" \
    "movdqu    xmm5, [ebx+32]" \
    "movdqa    xmm3, xmm4" \
    "pmaddwd   xmm3, xmm5" \
    "movdqu    xmm5, [ecx+32]" \
    "paddd     xmm0, xmm3" \
    "movdqa    xmm3, xmm4" \
    "pmaddwd   xmm3, xmm5" \
    "pxor      xmm4, xmm4" \
    "punpckhbw xmm4, xmm7" \
    "paddd     xmm1, xmm3" \
    "movdqu    xmm5, [ebx+48]" \
    "movdqa    xmm3, xmm4" \
    "pmaddwd   xmm3, xmm5" \
    "movdqu    xmm5, [ecx+48]" \
    "paddd     xmm0, xmm3" \
    "movdqa    xmm3, xmm4" \
    "pmaddwd   xmm3, xmm5" \
    "movdqa    xmm5, xmm0" \
    "psrldq    xmm5, 8" \
    "paddd     xmm1, xmm3" \
    "paddd     xmm0, xmm5" \
    "movdqa    xmm5, xmm0" \
    "psrldq    xmm5, 4" \
    "paddd     xmm0, xmm5" \
    "movd      [esi], xmm0" \
    "movdqa    xmm5, xmm1" \
    "psrldq    xmm5, 8" \
    "paddd     xmm1, xmm5" \
    "movdqa    xmm5, xmm1" \
    "psrldq    xmm5, 4" \
    "paddd     xmm1, xmm5" \
    "movd      [edi], xmm1" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] [__edi] \
    __modify [8087]

/* The paired kernel's address. */
void dot32_q61_sse2a_2(const unsigned char *w4, const unsigned char *w2,
                       const int16_t *xa, const int16_t *xb,
                       int32_t *oa, int32_t *ob) {
    lz_dot32_q61_sse2_asm_2(w4, w2, xa, xb, oa, ob);
}

/* ---- Q6_1 32/128-element dot products and row kernel, SSE2 tier -
   in this TU. Both this format's SSE2 kernels are genuine %xmm-only
   implementations, not an MMX reuse. */
extern int32_t lz_dot32_q61_sse2_asm(const unsigned char *w4, const unsigned char *w2, const int16_t *x);
#pragma aux lz_dot32_q61_sse2_asm = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pcmpeqb   xmm1, xmm1" \
    "psrlw     xmm1, 12" \
    "packuswb  xmm1, xmm1" \
    "pcmpeqb   xmm2, xmm2" \
    "psrlw     xmm2, 14" \
    "packuswb  xmm2, xmm2" \
    "movdqa    xmm3, [eax]" \
    "movq      xmm4, [edx]" \
    "movdqa    xmm5, xmm4" \
    "pand      xmm5, xmm2" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 2" \
    "pand      xmm6, xmm2" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "pand      xmm6, xmm1" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+16]" \
    "paddd     xmm0, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 6" \
    "paddd     xmm0, xmm5" \
    "movdqa    xmm5, xmm4" \
    "psrlw     xmm5, 4" \
    "pand      xmm5, xmm2" \
    "pand      xmm6, xmm2" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "psrlw     xmm6, 4" \
    "pand      xmm6, xmm1" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx+32]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+48]" \
    "paddd     xmm0, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "paddd     xmm0, xmm5" \
    "movdqa    xmm5, xmm0" \
    "psrldq    xmm5, 8" \
    "paddd     xmm0, xmm5" \
    "movdqa    xmm5, xmm0" \
    "psrldq    xmm5, 4" \
    "paddd     xmm0, xmm5" \
    "movd      eax, xmm0" \
    __parm [__eax] [__edx] [__ecx] \
    __value [__eax] \
    __modify [8087]

int32_t dot32_q61_sse2a(const unsigned char *w4, const unsigned char *w2,
                               const int16_t *x) {
    return lz_dot32_q61_sse2_asm(w4, w2, x);
}

extern void lz_dot128_q61_sse2_asm(const unsigned char *w4, const unsigned char *w2,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q61_sse2_asm = \
    ".686" \
    "pcmpeqb   xmm0, xmm0" \
    "psrlw     xmm0, 12" \
    "packuswb  xmm0, xmm0" \
    "pcmpeqb   xmm1, xmm1" \
    "psrlw     xmm1, 14" \
    "packuswb  xmm1, xmm1" \
    "pxor      xmm2, xmm2" \
    "movdqa    xmm3, [eax]" \
    "movq      xmm4, [edx]" \
    "movdqa    xmm5, xmm4" \
    "pand      xmm5, xmm1" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 2" \
    "pand      xmm6, xmm1" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "pand      xmm6, xmm0" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+16]" \
    "paddd     xmm2, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 6" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm4" \
    "psrlw     xmm5, 4" \
    "pand      xmm5, xmm1" \
    "pand      xmm6, xmm1" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "psrlw     xmm6, 4" \
    "pand      xmm6, xmm0" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx+32]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+48]" \
    "paddd     xmm2, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm2" \
    "psrldq    xmm5, 8" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm2" \
    "psrldq    xmm5, 4" \
    "paddd     xmm2, xmm5" \
    "movd      [ebx], xmm2" \
    "pxor      xmm2, xmm2" \
    "movdqa    xmm3, [eax+16]" \
    "movq      xmm4, [edx+8]" \
    "movdqa    xmm5, xmm4" \
    "pand      xmm5, xmm1" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 2" \
    "pand      xmm6, xmm1" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "pand      xmm6, xmm0" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx+64]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+80]" \
    "paddd     xmm2, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 6" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm4" \
    "psrlw     xmm5, 4" \
    "pand      xmm5, xmm1" \
    "pand      xmm6, xmm1" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "psrlw     xmm6, 4" \
    "pand      xmm6, xmm0" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx+96]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+112]" \
    "paddd     xmm2, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm2" \
    "psrldq    xmm5, 8" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm2" \
    "psrldq    xmm5, 4" \
    "paddd     xmm2, xmm5" \
    "movd      [ebx+4], xmm2" \
    "pxor      xmm2, xmm2" \
    "movdqa    xmm3, [eax+32]" \
    "movq      xmm4, [edx+16]" \
    "movdqa    xmm5, xmm4" \
    "pand      xmm5, xmm1" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 2" \
    "pand      xmm6, xmm1" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "pand      xmm6, xmm0" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx+128]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+144]" \
    "paddd     xmm2, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 6" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm4" \
    "psrlw     xmm5, 4" \
    "pand      xmm5, xmm1" \
    "pand      xmm6, xmm1" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "psrlw     xmm6, 4" \
    "pand      xmm6, xmm0" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx+160]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+176]" \
    "paddd     xmm2, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm2" \
    "psrldq    xmm5, 8" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm2" \
    "psrldq    xmm5, 4" \
    "paddd     xmm2, xmm5" \
    "movd      [ebx+8], xmm2" \
    "pxor      xmm2, xmm2" \
    "movdqa    xmm3, [eax+48]" \
    "movq      xmm4, [edx+24]" \
    "movdqa    xmm5, xmm4" \
    "pand      xmm5, xmm1" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 2" \
    "pand      xmm6, xmm1" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "pand      xmm6, xmm0" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx+192]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+208]" \
    "paddd     xmm2, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "movdqa    xmm6, xmm4" \
    "psrlw     xmm6, 6" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm4" \
    "psrlw     xmm5, 4" \
    "pand      xmm5, xmm1" \
    "pand      xmm6, xmm1" \
    "punpcklqdq xmm5, xmm6" \
    "psllw     xmm5, 4" \
    "movdqa    xmm6, xmm3" \
    "psrlw     xmm6, 4" \
    "pand      xmm6, xmm0" \
    "por       xmm6, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpcklbw xmm5, xmm6" \
    "movdqu    xmm7, [ecx+224]" \
    "pmaddwd   xmm5, xmm7" \
    "movdqu    xmm7, [ecx+240]" \
    "paddd     xmm2, xmm5" \
    "pxor      xmm5, xmm5" \
    "punpckhbw xmm5, xmm6" \
    "pmaddwd   xmm5, xmm7" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm2" \
    "psrldq    xmm5, 8" \
    "paddd     xmm2, xmm5" \
    "movdqa    xmm5, xmm2" \
    "psrldq    xmm5, 4" \
    "paddd     xmm2, xmm5" \
    "movd      [ebx+12], xmm2" \
    __parm [__eax] [__edx] [__ecx] [__ebx] \
    __modify [8087]

/* The SSE2 group kernel's address. Q6_1 was the last dot format with
   no probe of any kind, and this was the missing half of it - the MMX
   group kernel had lz_dot128_q61_mmx_w and this one had nothing, so
   the two tiers had never been compared on this format either. */
void lz_dot128_q61_sse2_w(const unsigned char *w4, const unsigned char *w2,
                          const int16_t *x, int32_t *out4) {
    lz_dot128_q61_sse2_asm(w4, w2, x, out4);
}

/* row_q61_sse2_asm: ALREADY the real call boundary from its own caller
   (matmul_q61_impl, src/ops.c, dispatched once per matmul via
   LZ_ROW_Q61). No pairing on this tier (unmeasured for SSE2, per the
   original comment at this function's old home in ops.c). */
void row_q61_sse2_asm(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    const unsigned char *wend = (const unsigned char *)c->pf_end;
    const unsigned char *wend2 = (const unsigned char *)c->pf_end2;
    int tk, s;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair before grouping, matching the gcc twin. This format has
               the heaviest weight side of the five - the 6-bit value is
               reassembled from two planes before it can be widened - so one
               pass of it serving two tokens is the largest saving the pair
               buys anywhere. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                int sp;
                for (sp = 0; sp < c->nb; sp++) {
                    LZ_Q61_PF(sp);
                    lz_dot32_q61_sse2_asm_2(wn + (size_t)sp * 16,
                                            w2 + (size_t)sp * 8,
                                            xwt + (size_t)sp * 32,
                                            xw2 + (size_t)sp * 32,
                                            acc + sp, accb + sp);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
#if LZ_SSE2_GROUP
            if ((c->nb & 3) == 0) LZ_Q61_GROUP4(lz_dot128_q61_sse2_asm);
#else
            /* lz_dot128_q61_mmx_w is src/ops_mmx.c's real wrapper around
               the MMX #pragma aux group kernel. */
            if ((c->nb & 3) == 0) LZ_Q61_GROUP4(lz_dot128_q61_mmx_w);
#endif /* LZ_SSE2_GROUP */
            /* Measured on target: on a real Pentium M the SSE2 kernel is
               20-30% faster at kernel level and no worse end-to-end. Uop
               counts compare only WITHIN one instruction set; a 128-bit uop
               and a 64-bit uop are not the same unit of work, so the
               uop-count argument does not apply across instruction sets. */
            else               LZ_Q61_ACC(dot32_q61_sse2a);
        }
    }
}

/* ---- Q16_0 32/128-element dot products and row kernel, SSE2 tier -
   in this TU. */
extern int32_t lz_dot32_q16_sse2_asm(const int16_t *w, const int16_t *x);
#pragma aux lz_dot32_q16_sse2_asm = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "movdqa    xmm2, [eax]" \
    "movdqu    xmm3, [ecx]" \
    "pmaddwd   xmm2, xmm3" \
    "movdqa    xmm4, [eax+32]" \
    "movdqu    xmm5, [ecx+32]" \
    "pmaddwd   xmm4, xmm5" \
    "paddd     xmm0, xmm2" \
    "movdqa    xmm2, [eax+16]" \
    "paddd     xmm1, xmm4" \
    "movdqu    xmm3, [ecx+16]" \
    "pmaddwd   xmm2, xmm3" \
    "movdqa    xmm4, [eax+48]" \
    "movdqu    xmm5, [ecx+48]" \
    "pmaddwd   xmm4, xmm5" \
    "paddd     xmm0, xmm2" \
    "paddd     xmm1, xmm4" \
    "paddd     xmm0, xmm1" \
    "movdqa    xmm6, xmm0" \
    "psrldq    xmm6, 8" \
    "paddd     xmm0, xmm6" \
    "movdqa    xmm6, xmm0" \
    "psrldq    xmm6, 4" \
    "paddd     xmm0, xmm6" \
    "movd      eax, xmm0" \
    __parm [__eax] [__ecx] \
    __value [__eax] \
    __modify [8087]

int32_t dot32_q16_sse2a(const int16_t *w, const int16_t *x) {
    return lz_dot32_q16_sse2_asm(w, x);
}

/* Two tokens, one weight load - the Watcom twin of part32_q16_2, and
   the smallest pair of the five formats.

   int16 weights need no unpack, so what the pair shares is four
   movdqu and nothing else. xmm0-xmm3 hold the weight quarters for the
   duration, xmm4/xmm5 are the two accumulators, xmm6 the activation and
   xmm7 the multiply temp pmaddwd destroys - all eight in use, none
   spilled, which is what makes holding the weights the whole trick.

   Bit-identical to two lz_dot32_q16_sse2_asm calls: same four pmaddwd
   per token into that token's own accumulator, same fold order. */
extern void lz_dot32_q16_sse2_asm_2(const int16_t *w, const int16_t *xa,
                                    const int16_t *xb,
                                    int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_q16_sse2_asm_2 = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "movdqa    xmm2, [eax]" \
    "movdqa    xmm3, [eax+16]" \
    "movdqa    xmm4, [eax+32]" \
    "movdqa    xmm5, [eax+48]" \
    "movdqu    xmm6, [edx]" \
    "movdqa    xmm7, xmm2" \
    "pmaddwd   xmm7, xmm6" \
    "movdqu    xmm6, [edx+16]" \
    "paddd     xmm0, xmm7" \
    "movdqa    xmm7, xmm3" \
    "pmaddwd   xmm7, xmm6" \
    "movdqu    xmm6, [edx+32]" \
    "paddd     xmm0, xmm7" \
    "movdqa    xmm7, xmm4" \
    "pmaddwd   xmm7, xmm6" \
    "movdqu    xmm6, [edx+48]" \
    "paddd     xmm0, xmm7" \
    "movdqa    xmm7, xmm5" \
    "pmaddwd   xmm7, xmm6" \
    "movdqu    xmm6, [ebx]" \
    "paddd     xmm0, xmm7" \
    "movdqa    xmm7, xmm2" \
    "pmaddwd   xmm7, xmm6" \
    "movdqu    xmm6, [ebx+16]" \
    "paddd     xmm1, xmm7" \
    "movdqa    xmm7, xmm3" \
    "pmaddwd   xmm7, xmm6" \
    "movdqu    xmm6, [ebx+32]" \
    "paddd     xmm1, xmm7" \
    "movdqa    xmm7, xmm4" \
    "pmaddwd   xmm7, xmm6" \
    "movdqu    xmm6, [ebx+48]" \
    "paddd     xmm1, xmm7" \
    "movdqa    xmm7, xmm5" \
    "pmaddwd   xmm7, xmm6" \
    "movdqa    xmm6, xmm0" \
    "psrldq    xmm6, 8" \
    "paddd     xmm1, xmm7" \
    "paddd     xmm0, xmm6" \
    "movdqa    xmm6, xmm0" \
    "psrldq    xmm6, 4" \
    "paddd     xmm0, xmm6" \
    "movd      [ecx], xmm0" \
    "movdqa    xmm6, xmm1" \
    "psrldq    xmm6, 8" \
    "paddd     xmm1, xmm6" \
    "movdqa    xmm6, xmm1" \
    "psrldq    xmm6, 4" \
    "paddd     xmm1, xmm6" \
    "movd      [esi], xmm1" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [8087]

/* The paired kernel's address, for the reason its MMX twin now has
   one. */
void dot32_q16_sse2a_2(const int16_t *w, const int16_t *xa,
                       const int16_t *xb, int32_t *oa, int32_t *ob) {
    lz_dot32_q16_sse2_asm_2(w, xa, xb, oa, ob);
}

extern void lz_dot128_q16_sse2_asm(const int16_t *w, const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q16_sse2_asm = \
    ".686" \
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "pxor      xmm2, xmm2" \
    "pxor      xmm3, xmm3" \
    "movdqa    xmm4, [eax]" \
    "movdqa    xmm5, [eax+64]" \
    "movdqa    xmm6, [eax+128]" \
    "movdqa    xmm7, [eax+192]" \
    "pmaddwd   xmm4, [ecx]" \
    "pmaddwd   xmm5, [ecx+64]" \
    "pmaddwd   xmm6, [ecx+128]" \
    "pmaddwd   xmm7, [ecx+192]" \
    "paddd     xmm0, xmm4" \
    "paddd     xmm1, xmm5" \
    "paddd     xmm2, xmm6" \
    "paddd     xmm3, xmm7" \
    "movdqa    xmm4, [eax+16]" \
    "movdqa    xmm5, [eax+80]" \
    "movdqa    xmm6, [eax+144]" \
    "movdqa    xmm7, [eax+208]" \
    "pmaddwd   xmm4, [ecx+16]" \
    "pmaddwd   xmm5, [ecx+80]" \
    "pmaddwd   xmm6, [ecx+144]" \
    "pmaddwd   xmm7, [ecx+208]" \
    "paddd     xmm0, xmm4" \
    "paddd     xmm1, xmm5" \
    "paddd     xmm2, xmm6" \
    "paddd     xmm3, xmm7" \
    "movdqa    xmm4, [eax+32]" \
    "movdqa    xmm5, [eax+96]" \
    "movdqa    xmm6, [eax+160]" \
    "movdqa    xmm7, [eax+224]" \
    "pmaddwd   xmm4, [ecx+32]" \
    "pmaddwd   xmm5, [ecx+96]" \
    "pmaddwd   xmm6, [ecx+160]" \
    "pmaddwd   xmm7, [ecx+224]" \
    "paddd     xmm0, xmm4" \
    "paddd     xmm1, xmm5" \
    "paddd     xmm2, xmm6" \
    "paddd     xmm3, xmm7" \
    "movdqa    xmm4, [eax+48]" \
    "movdqa    xmm5, [eax+112]" \
    "movdqa    xmm6, [eax+176]" \
    "movdqa    xmm7, [eax+240]" \
    "pmaddwd   xmm4, [ecx+48]" \
    "pmaddwd   xmm5, [ecx+112]" \
    "pmaddwd   xmm6, [ecx+176]" \
    "pmaddwd   xmm7, [ecx+240]" \
    "paddd     xmm0, xmm4" \
    "paddd     xmm1, xmm5" \
    "paddd     xmm2, xmm6" \
    "paddd     xmm3, xmm7" \
    "movdqa    xmm4, xmm0" \
    "psrldq    xmm4, 8" \
    "paddd     xmm0, xmm4" \
    "movdqa    xmm4, xmm0" \
    "psrldq    xmm4, 4" \
    "paddd     xmm0, xmm4" \
    "movd      [ebx], xmm0" \
    "movdqa    xmm4, xmm1" \
    "psrldq    xmm4, 8" \
    "paddd     xmm1, xmm4" \
    "movdqa    xmm4, xmm1" \
    "psrldq    xmm4, 4" \
    "paddd     xmm1, xmm4" \
    "movd      [ebx+4], xmm1" \
    "movdqa    xmm4, xmm2" \
    "psrldq    xmm4, 8" \
    "paddd     xmm2, xmm4" \
    "movdqa    xmm4, xmm2" \
    "psrldq    xmm4, 4" \
    "paddd     xmm2, xmm4" \
    "movd      [ebx+8], xmm2" \
    "movdqa    xmm4, xmm3" \
    "psrldq    xmm4, 8" \
    "paddd     xmm3, xmm4" \
    "movdqa    xmm4, xmm3" \
    "psrldq    xmm4, 4" \
    "paddd     xmm3, xmm4" \
    "movd      [ebx+12], xmm3" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

/* Real, addressable twin. Q16_0 had NO cross-tier check of any kind
   before this: the MMX group kernel had a wrapper and the SSE2 one did
   not, so the two tiers were never compared on this format even though
   bit-identity across tiers is a standing requirement.
   build/dot128_q16_probe.c uses it. */
void lz_dot128_q16_sse2_w(const int16_t *w, const int16_t *x,
                          int32_t *out4) {
    lz_dot128_q16_sse2_asm(w, x, out4);
}

/* row_q16_sse2_asm: ALREADY the real call boundary (LZ_ROW_Q16,
   src/ops.c). */
void row_q16_sse2_asm(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    const int16_t *wend = (const int16_t *)c->pf_end;
    int tk, g;
    (void)c->w2; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair before grouping, as every other row kernel here does.
               Smallest pair of the five - int16 needs no unpack, so four
               shared loads is all of it. */
            if (c->nt - tk >= 2 && lz_pair_mode()) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                int gp;
                for (gp = 0; gp < c->nb; gp++) {
                    LZ_Q16_PF(gp);
                    lz_dot32_q16_sse2_asm_2(wr + (size_t)gp * 32,
                                            xwt + (size_t)gp * 32,
                                            xw2 + (size_t)gp * 32,
                                            acc + gp, accb + gp);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
#if LZ_SSE2_GROUP
            if ((c->nb & 3) == 0) LZ_Q16_GROUP4(lz_dot128_q16_sse2_asm);
#else
            /* lz_dot128_q16_mmx_w is src/ops_mmx.c's real wrapper around
               the MMX #pragma aux group kernel. */
            if ((c->nb & 3) == 0) LZ_Q16_GROUP4(lz_dot128_q16_mmx_w);
#endif /* LZ_SSE2_GROUP */
            else               LZ_Q16_ACC(dot32_q16_sse2a);
        }
    }
}

/* ---- attention's two fixed-point MACs, SSE2 tier ----------------------
   Both had an MMX cell and no SSE2 one, not because SSE2 could not
   express them but because the two call sites bound to the MMX name
   with no tier test at all - so no SSE2 twin was ever selectable. The
   matrix called that "unaudited, not impossible"; this is the audit.

   1. The SCORING DOT is a REUSE, not a new kernel. lz_dot32_x16_sse2_asm
      above already computes 256*sum(w*x) for int8 weights against
      int16 activations - the punpcklbw(0, w) puts each weight in the
      HIGH byte, which IS the x256 the attention path folds out with its
      single 1/256. Exactly the reuse ARM makes of q8_0's leaf
      (docs/arm-asm-audit.md section 6); a second body would be a copy.

   2. The WEIGHTED SUM is a real new body, and it is the MMX one
      widened: sign-extend two int8 rows to int16, interleave so one
      pmaddwd covers a (rowA,rowB) pair against (ckA,ckB), accumulate in
      place. SSE2 does 16 elements per pass where MMX does 8, so 32
      elements take two passes instead of four, and the 128-bit constant
      comes from ONE dword load broadcast with pshufd rather than the
      caller's duplicated coef[4] (which stays, unread past coef[1], so
      the two kernels keep one call signature).

   Bit-identical by construction, not by measurement: every operation
   here is integer, and integer sums are exact under reassociation. The
   gate still runs, because "by construction" is an argument about the
   algorithm and not about what got typed.

   No emms owed by either: pure %xmm, no %mm register touched. The
   group loop's existing _mm_empty stays where it is - it is the
   caller's duty and it is free on an already-clean state. */
int32_t lz_dot32_x16_sse2(const int8_t *w, const int16_t *x) {
    return lz_dot32_x16_sse2_asm(w, x);
}

/* parm order is rowA/rowB/coef/acc32 -> eax/edx/EBX/ECX, which is the
   last two SWAPPED relative to lz_wsum_pair_asm's eax/edx/ecx/ebx. Not
   an oversight and not worth unifying: both are reached only through
   the C wrapper below (or, for the MMX one, its own call site in this
   file's twin), so the mapping is internal to each pragma. Written this
   way because ecx addresses the accumulator sixteen times here and the
   register a Watcom pragma must not renumber is the one in the parm
   list, not the one in the body. */
extern void lz_wsum_pair_sse2_asm(const int8_t *rowA, const int8_t *rowB,
                                  const int16_t *coef, int32_t *acc32);
#pragma aux lz_wsum_pair_sse2_asm = \
    ".686" \
    "movd      xmm0, [ebx]" \
    "pshufd    xmm0, xmm0, 0" \
    "movdqa    xmm1, [eax]" \
    "movdqa    xmm2, [edx]" \
    "movdqa    xmm3, xmm1" \
    "movdqa    xmm4, xmm2" \
    "punpcklbw xmm3, xmm3" \
    "punpcklbw xmm4, xmm4" \
    "psraw     xmm3, 8" \
    "psraw     xmm4, 8" \
    "movdqa    xmm5, xmm3" \
    "punpcklwd xmm5, xmm4" \
    "punpckhwd xmm3, xmm4" \
    "pmaddwd   xmm5, xmm0" \
    "pmaddwd   xmm3, xmm0" \
    "movdqu    xmm6, [ecx]" \
    "movdqu    xmm7, [ecx+16]" \
    "paddd     xmm6, xmm5" \
    "paddd     xmm7, xmm3" \
    "movdqu    [ecx], xmm6" \
    "movdqu    [ecx+16], xmm7" \
    "movdqa    xmm3, xmm1" \
    "movdqa    xmm4, xmm2" \
    "punpckhbw xmm3, xmm3" \
    "punpckhbw xmm4, xmm4" \
    "psraw     xmm3, 8" \
    "psraw     xmm4, 8" \
    "movdqa    xmm5, xmm3" \
    "punpcklwd xmm5, xmm4" \
    "punpckhwd xmm3, xmm4" \
    "pmaddwd   xmm5, xmm0" \
    "pmaddwd   xmm3, xmm0" \
    "movdqu    xmm6, [ecx+32]" \
    "movdqu    xmm7, [ecx+48]" \
    "paddd     xmm6, xmm5" \
    "paddd     xmm7, xmm3" \
    "movdqu    [ecx+32], xmm6" \
    "movdqu    [ecx+48], xmm7" \
    "movdqa    xmm1, [eax+16]" \
    "movdqa    xmm2, [edx+16]" \
    "movdqa    xmm3, xmm1" \
    "movdqa    xmm4, xmm2" \
    "punpcklbw xmm3, xmm3" \
    "punpcklbw xmm4, xmm4" \
    "psraw     xmm3, 8" \
    "psraw     xmm4, 8" \
    "movdqa    xmm5, xmm3" \
    "punpcklwd xmm5, xmm4" \
    "punpckhwd xmm3, xmm4" \
    "pmaddwd   xmm5, xmm0" \
    "pmaddwd   xmm3, xmm0" \
    "movdqu    xmm6, [ecx+64]" \
    "movdqu    xmm7, [ecx+80]" \
    "paddd     xmm6, xmm5" \
    "paddd     xmm7, xmm3" \
    "movdqu    [ecx+64], xmm6" \
    "movdqu    [ecx+80], xmm7" \
    "movdqa    xmm3, xmm1" \
    "movdqa    xmm4, xmm2" \
    "punpckhbw xmm3, xmm3" \
    "punpckhbw xmm4, xmm4" \
    "psraw     xmm3, 8" \
    "psraw     xmm4, 8" \
    "movdqa    xmm5, xmm3" \
    "punpcklwd xmm5, xmm4" \
    "punpckhwd xmm3, xmm4" \
    "pmaddwd   xmm5, xmm0" \
    "pmaddwd   xmm3, xmm0" \
    "movdqu    xmm6, [ecx+96]" \
    "movdqu    xmm7, [ecx+112]" \
    "paddd     xmm6, xmm5" \
    "paddd     xmm7, xmm3" \
    "movdqu    [ecx+96], xmm6" \
    "movdqu    [ecx+112], xmm7" \
    __parm [__eax] [__edx] [__ebx] [__ecx] \
    __modify [8087];

/* Thin wrapper, and it is not optional: a #pragma aux body expands at
   the CALL SITE, and the call site is src/ops_mmx.c's group loop.
   Same shape as lz_fwht_stage_mmx and lz_dot128_q16_mmx_w. */
void lz_wsum_pair_sse2(const int8_t *rowA, const int8_t *rowB,
                       const int16_t *coef, int32_t *acc32) {
    lz_wsum_pair_sse2_asm(rowA, rowB, coef, acc32);
}

/* ---- epi's int32 x int16 -> int64 accumulation, SSE2 tier ------------ */

/* Twin of the intrinsics body in the gcc branch below, which carries the
   derivation: three madds on a carry-free 16-bit split, the borrow
   joining ah in int32 rather than in a lane it would overflow.

   WHAT IS HAND-WRITTEN HERE THAT gcc COULD NOT DO. Both constants stay
   in registers. gcc rebuilds zero and 0xFFFF inside the loop - pxor,
   then pcmpeqd plus psrld - because two accumulator pairs and four live
   temporaries already fill eight XMM registers and it has nowhere to
   hoist them to. The register map below reserves xmm2 and xmm3 for them
   and spends xmm4-xmm7 on the group, which fits exactly:

     xmm0  s_hi   accumulator, 2 x int64      xmm4  a, then als
     xmm1  s_als  accumulator, 2 x int64      xmm5  m widened, then a copy
     xmm2  zero                               xmm6  the ah+b product
     xmm3  0x0000FFFF per lane                xmm7  b, then a sign mask

   31 instructions per four elements - 7.75 an element, against gcc's
   8.75 for the same algorithm and 8.00 for the scalar loop. The four
   saved against gcc are the two constants it rebuilds every group; the
   rest is the same body. It is a small win on count and a large one on
   branches, which is the whole argument for a vector body here: one
   back edge per four elements instead of four.

   NO 16-BYTE ALIGNMENT IS ASSUMED. movdqu on a, movq on m - the caller
   is epi_q41_join, whose zero plane starts at an arbitrary row offset.

   b WITHOUT A CONSTANT: pslld 16 then psrld 31 leaves bit 15 of a in
   bit 0 and nothing else set, which is the same three instructions as
   copy/shift/mask and one fewer live register. That is what pays for
   0xFFFF staying resident.

   The epilogue folds each accumulator's two lanes with pshufd 0x4E
   rather than psrldq, matching lz_amax32_sse2_asm, and does the
   `(s_hi << 16) + s_als` join in the vector unit so only one 64-bit
   value crosses to edx:eax.

   Whole groups only. The wrapper's tail is the plain product, because
   ((ah + b) << 16) + als IS a. */
extern lz_i64 lz_epi_mac_i16_sse2_asm(const int32_t *a, const int16_t *m,
                                      int nb);
#pragma aux lz_epi_mac_i16_sse2_asm =   \
    ".686"                              \
    "pxor      xmm0, xmm0" \
    "pxor      xmm1, xmm1" \
    "pxor      xmm2, xmm2" \
    "pcmpeqd   xmm3, xmm3" \
    "psrld     xmm3, 16" \
    "EPIG:" \
    "movdqu    xmm4, [eax]" \
    "movq      xmm5, [edx]" \
    "punpcklwd xmm5, xmm2" \
    "movdqa    xmm6, xmm4" \
    "psrld     xmm6, 16" \
    "pmaddwd   xmm6, xmm5" \
    "movdqa    xmm7, xmm4" \
    "pslld     xmm7, 16" \
    "psrld     xmm7, 31" \
    "pmaddwd   xmm7, xmm5" \
    "pand      xmm4, xmm3" \
    "pmaddwd   xmm4, xmm5" \
    "paddd     xmm6, xmm7" \
    "movdqa    xmm7, xmm6" \
    "psrad     xmm7, 31" \
    "movdqa    xmm5, xmm6" \
    "punpckldq xmm6, xmm7" \
    "paddq     xmm0, xmm6" \
    "punpckhdq xmm5, xmm7" \
    "paddq     xmm0, xmm5" \
    "movdqa    xmm7, xmm4" \
    "psrad     xmm7, 31" \
    "movdqa    xmm5, xmm4" \
    "punpckldq xmm4, xmm7" \
    "paddq     xmm1, xmm4" \
    "punpckhdq xmm5, xmm7" \
    "paddq     xmm1, xmm5" \
    "add       eax, 16" \
    "add       edx, 8" \
    "dec       ecx" \
    "jnz       EPIG" \
    "pshufd    xmm2, xmm0, 0x4E" \
    "paddq     xmm0, xmm2" \
    "pshufd    xmm2, xmm1, 0x4E" \
    "paddq     xmm1, xmm2" \
    "psllq     xmm0, 16" \
    "paddq     xmm0, xmm1" \
    "movd      eax, xmm0" \
    "psrlq     xmm0, 32" \
    "movd      edx, xmm0" \
    __parm [__eax] [__edx] [__ecx]      \
    __value [__edx __eax]               \
    __modify [__eax __edx __ecx 8087];

/* Wrapper, for the reason lz_wsum_pair_sse2 gives, plus one of its own:
   the pragma runs whole groups and cannot branch back into C, so the
   remainder is handled here.

   nb is clamped rather than early-returned so a non-positive n falls
   through both the call and the loop - the C body in ops.c answers 0
   there, and a 4-wide loop entered with a negative count walks
   backwards through memory. */
lz_i64 lz_epi_mac_i16_sse2(const int32_t *a, const int16_t *m, int n) {
    int nb = (n > 0) ? (n >> 2) : 0;
    int g  = nb << 2;
    lz_i64 s = nb ? lz_epi_mac_i16_sse2_asm(a, m, nb) : 0;
    for (; g < n; g++) s += (lz_i64)a[g] * (lz_i64)m[g];
    return s;
}

#else /* gcc: intrinsics */

#include <emmintrin.h>

#include "ops_kernel_dot_sse2.h"
/* g_pair, for the row kernels' two-token path. The Watcom half of this
   file reaches the same knob through ops.h's lz_pair_mode(), included
   in its own branch; the gcc half had no need of it until the SSE2
   tier gained pairing. */
#include "ops_sched.h"

/* ---- Q8 row kernel ------------------------------------------------------
   Watcom twin: row_q8_sse2_asm, src/ops.c. */
void row_q8_sse2_intrin(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    int tk, g;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair and group TOGETHER, which the MMX tiers cannot do: there
               the pair replaces the group because a token pair cannot be
               expressed once the group loop owns the sub-block index. Here
               fold4_sse2 takes four partials per token, so one weight pass
               feeds two folds - both axes at once, four sub-blocks by two
               tokens.
               The weight side is what the pair saves: two loads and four
               unpacks against zero, paid once instead of twice. Only the
               activation loads and the pmaddwd double. An odd nt leaves its
               last token on the single-token path below. */
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (g = 0; g + 3 < c->nb; g += 4) {
                    __m128i pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3;
                    part32_x16_2(wr + (size_t)(g + 0) * 32,
                                 xwt + (size_t)(g + 0) * 32,
                                 xw2 + (size_t)(g + 0) * 32, &pa0, &pb0);
                    part32_x16_2(wr + (size_t)(g + 1) * 32,
                                 xwt + (size_t)(g + 1) * 32,
                                 xw2 + (size_t)(g + 1) * 32, &pa1, &pb1);
                    part32_x16_2(wr + (size_t)(g + 2) * 32,
                                 xwt + (size_t)(g + 2) * 32,
                                 xw2 + (size_t)(g + 2) * 32, &pa2, &pb2);
                    part32_x16_2(wr + (size_t)(g + 3) * 32,
                                 xwt + (size_t)(g + 3) * 32,
                                 xw2 + (size_t)(g + 3) * 32, &pa3, &pb3);
                    _mm_storeu_si128((__m128i *)(void *)(acc + g),
                                     fold4_sse2(pa0, pa1, pa2, pa3));
                    _mm_storeu_si128((__m128i *)(void *)(accb + g),
                                     fold4_sse2(pb0, pb1, pb2, pb3));
                }
                for (; g < c->nb; g++) {
                    __m128i pa, pb;
                    part32_x16_2(wr + (size_t)g * 32, xwt + (size_t)g * 32,
                                 xw2 + (size_t)g * 32, &pa, &pb);
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 8));
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 4));
                    acc[g] = _mm_cvtsi128_si32(pa);
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 8));
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 4));
                    accb[g] = _mm_cvtsi128_si32(pb);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            for (g = 0; g + 3 < c->nb; g += 4) {
                _mm_storeu_si128((__m128i *)(void *)(acc + g),
                    fold4_sse2(
                    part32_x16(wr + (size_t)(g + 0) * 32, xwt + (size_t)(g + 0) * 32),
                    part32_x16(wr + (size_t)(g + 1) * 32, xwt + (size_t)(g + 1) * 32),
                    part32_x16(wr + (size_t)(g + 2) * 32, xwt + (size_t)(g + 2) * 32),
                    part32_x16(wr + (size_t)(g + 3) * 32, xwt + (size_t)(g + 3) * 32)));
            }
            for (; g < c->nb; g++) {           /* tail when nb % 4 != 0 */
                __m128i p = part32_x16(wr + (size_t)g * 32, xwt + (size_t)g * 32);
                p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
                p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
                acc[g] = _mm_cvtsi128_si32(p);
            }
        }
    }
}

/* ---- Q4_1 row kernel -----------------------------------------------------
   Watcom twin: row_q41_sse2_asm, src/ops.c. */
void row_q41_sse2_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair and group together, the shape q8_0 established at this
               tier: fold4_sse2 already takes four partials per token, so one
               weight pass feeds two folds. q4_1 shares more than q8_0 did -
               the nibble mask, the shift and the second mask on top of the
               four unpacks. */
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (s = 0; s + 3 < c->nb; s += 4) {
                    __m128i pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3;
                    part32_q41_2(wn + (size_t)(s + 0) * 16,
                                 xwt + (size_t)(s + 0) * 32,
                                 xw2 + (size_t)(s + 0) * 32, &pa0, &pb0);
                    part32_q41_2(wn + (size_t)(s + 1) * 16,
                                 xwt + (size_t)(s + 1) * 32,
                                 xw2 + (size_t)(s + 1) * 32, &pa1, &pb1);
                    part32_q41_2(wn + (size_t)(s + 2) * 16,
                                 xwt + (size_t)(s + 2) * 32,
                                 xw2 + (size_t)(s + 2) * 32, &pa2, &pb2);
                    part32_q41_2(wn + (size_t)(s + 3) * 16,
                                 xwt + (size_t)(s + 3) * 32,
                                 xw2 + (size_t)(s + 3) * 32, &pa3, &pb3);
                    _mm_storeu_si128((__m128i *)(void *)(acc + s),
                                     fold4_sse2(pa0, pa1, pa2, pa3));
                    _mm_storeu_si128((__m128i *)(void *)(accb + s),
                                     fold4_sse2(pb0, pb1, pb2, pb3));
                }
                for (; s < c->nb; s++) {
                    __m128i pa, pb;
                    part32_q41_2(wn + (size_t)s * 16, xwt + (size_t)s * 32,
                                 xw2 + (size_t)s * 32, &pa, &pb);
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 8));
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 4));
                    acc[s] = _mm_cvtsi128_si32(pa);
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 8));
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 4));
                    accb[s] = _mm_cvtsi128_si32(pb);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            for (s = 0; s + 3 < c->nb; s += 4) {
                _mm_storeu_si128((__m128i *)(void *)(acc + s),
                    fold4_sse2(
                    part32_q41(wn + (size_t)(s + 0) * 16, xwt + (size_t)(s + 0) * 32),
                    part32_q41(wn + (size_t)(s + 1) * 16, xwt + (size_t)(s + 1) * 32),
                    part32_q41(wn + (size_t)(s + 2) * 16, xwt + (size_t)(s + 2) * 32),
                    part32_q41(wn + (size_t)(s + 3) * 16, xwt + (size_t)(s + 3) * 32)));
            }
            for (; s < c->nb; s++) {
                __m128i p = part32_q41(wn + (size_t)s * 16, xwt + (size_t)s * 32);
                p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
                p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
                acc[s] = _mm_cvtsi128_si32(p);
            }
        }
    }
}

/* ---- T2 row kernel --------------------------------------------------------
   Watcom twin: row_t2_sse2_asm, src/ops.c. No pairing (unmeasured for
   this format, same as the other tiers). */
void row_t2_sse2_intrin(const lz_row_ctx *c) {
    const unsigned char *p2 = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair ahead of the group form, the order every row kernel here
               uses. T2's decode at THIS tier is fourteen instructions - four
               masks, three shifts, two merges, four unpacks - so the pair
               shares more than q4_1's, unlike at the MMX tier where the same
               operator's unpack is a zero register and three shifts. */
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (s = 0; s < c->nb; s++) {
                    __m128i pa, pb;
                    part32_t2_2(p2 + (size_t)s * 8, xwt + (size_t)s * 32,
                                xw2 + (size_t)s * 32, &pa, &pb);
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 8));
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 4));
                    acc[s] = _mm_cvtsi128_si32(pa);
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 8));
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 4));
                    accb[s] = _mm_cvtsi128_si32(pb);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            s = 0;
            /* The knob is honoured but not redefined here. LZ_G128_GCC lives
               in ops_kernel_shared.h, which this file includes only under
               __WATCOMC__ (line 34) - so in a gcc build the name is not
               visible and a plain `#if LZ_G128_GCC` is `#if 0`, which is
               what it was: this loop compiled to nothing and -Wunused-function
               on dot128_t2_sse2 was the only thing that said so.
               Including that header here would also define LZ_SSE2_GROUP and
               flip the branch above onto the Watcom asm path. Defining the
               default a second time would be a sibling free to drift. So:
               on unless the command line turns it off, which is exactly how
               the A/B is run (-DLZ_G128_GCC=0). */
#if !defined(LZ_G128_GCC) || LZ_G128_GCC
            /* Groups of four while a whole one is left; the tail below runs
               the per-32 fold for nb % 4. Both write the same values - see
               dot128_t2_sse2 on why the transposed fold is bit-identical -
               so a row is not split between two answers, only between two
               instruction counts. */
            for (; s + 4 <= c->nb; s += 4)
                dot128_t2_sse2(p2 + (size_t)s * 8, xwt + (size_t)s * 32, acc + s);
#endif /* !defined(LZ_G128_GCC) || LZ_G128_GCC */
            for (; s < c->nb; s++) {
                __m128i p = part32_t2(p2 + (size_t)s * 8, xwt + (size_t)s * 32);
                p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
                p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
                acc[s] = _mm_cvtsi128_si32(p);
            }
        }
    }
}

/* ---- Q6_1 row kernel -----------------------------------------------------
   Watcom twin: row_q61_sse2_asm, src/ops.c. */
void row_q61_sse2_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    int tk, s;
    (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* The pair is worth most on this format: the 6-bit reassembly
               is ~19 instructions on the weight side against q4_1's 8 and
               q8_0's 6, and one pass of it now serves two tokens. No group
               form here to combine with - this row runs per-32 either way. */
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (s = 0; s < c->nb; s++) {
                    __m128i pa, pb;
                    part32_q61_2(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                 xwt + (size_t)s * 32, xw2 + (size_t)s * 32,
                                 &pa, &pb);
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 8));
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 4));
                    acc[s] = _mm_cvtsi128_si32(pa);
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 8));
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 4));
                    accb[s] = _mm_cvtsi128_si32(pb);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            for (s = 0; s < c->nb; s++) {
                __m128i p = part32_q61(wn + (size_t)s * 16, w2 + (size_t)s * 8,
                                       xwt + (size_t)s * 32);
                p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
                p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
                acc[s] = _mm_cvtsi128_si32(p);
            }
        }
    }
}

/* ---- Q16_0 row kernel ----------------------------------------------------
   Watcom twin: row_q16_sse2_asm, src/ops.c. */
void row_q16_sse2_intrin(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    int tk, g;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            /* Pair and group together, as q8_0 and q4_1 do at this tier.
               The smallest pair of the five: int16 weights need no unpack,
               so four shared loads is the whole saving. */
            if (tk + 1 < c->nt && g_pair) {
                const int16_t *xw2 = xwt + c->in_dim;
                int32_t *accb = acc + c->nb;
                for (g = 0; g + 3 < c->nb; g += 4) {
                    __m128i pa0, pa1, pa2, pa3, pb0, pb1, pb2, pb3;
                    part32_q16_2(wr + (size_t)(g + 0) * 32,
                                 xwt + (size_t)(g + 0) * 32,
                                 xw2 + (size_t)(g + 0) * 32, &pa0, &pb0);
                    part32_q16_2(wr + (size_t)(g + 1) * 32,
                                 xwt + (size_t)(g + 1) * 32,
                                 xw2 + (size_t)(g + 1) * 32, &pa1, &pb1);
                    part32_q16_2(wr + (size_t)(g + 2) * 32,
                                 xwt + (size_t)(g + 2) * 32,
                                 xw2 + (size_t)(g + 2) * 32, &pa2, &pb2);
                    part32_q16_2(wr + (size_t)(g + 3) * 32,
                                 xwt + (size_t)(g + 3) * 32,
                                 xw2 + (size_t)(g + 3) * 32, &pa3, &pb3);
                    _mm_storeu_si128((__m128i *)(void *)(acc + g),
                                     fold4_sse2(pa0, pa1, pa2, pa3));
                    _mm_storeu_si128((__m128i *)(void *)(accb + g),
                                     fold4_sse2(pb0, pb1, pb2, pb3));
                }
                for (; g < c->nb; g++) {
                    __m128i pa, pb;
                    part32_q16_2(wr + (size_t)g * 32, xwt + (size_t)g * 32,
                                 xw2 + (size_t)g * 32, &pa, &pb);
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 8));
                    pa = _mm_add_epi32(pa, _mm_srli_si128(pa, 4));
                    acc[g] = _mm_cvtsi128_si32(pa);
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 8));
                    pb = _mm_add_epi32(pb, _mm_srli_si128(pb, 4));
                    accb[g] = _mm_cvtsi128_si32(pb);
                }
                tk++;
                xwt += c->in_dim;   /* skip tk+1's pointers too */
                acc += c->nb;
                continue;
            }
            for (g = 0; g + 3 < c->nb; g += 4) {
                _mm_storeu_si128((__m128i *)(void *)(acc + g),
                    fold4_sse2(
                    part32_q16(wr + (size_t)(g + 0) * 32, xwt + (size_t)(g + 0) * 32),
                    part32_q16(wr + (size_t)(g + 1) * 32, xwt + (size_t)(g + 1) * 32),
                    part32_q16(wr + (size_t)(g + 2) * 32, xwt + (size_t)(g + 2) * 32),
                    part32_q16(wr + (size_t)(g + 3) * 32, xwt + (size_t)(g + 3) * 32)));
            }
            for (; g < c->nb; g++) {           /* tail when nb % 4 != 0 */
                __m128i p = part32_q16(wr + (size_t)g * 32, xwt + (size_t)g * 32);
                p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
                p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
                acc[g] = _mm_cvtsi128_si32(p);
            }
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

/* ---- attention wsum's chunk fold, SSE2 tier ------------------------------
   Watcom twin: lz_i32f_acc32_sse2_asm above. accf[d] += (float)acc32[d]
   for 32 elements - cvtdq2ps four int32 at a time, addps to accumulate.
   No MMX register, so no emms is owed at a site that has just paid one.

   Written rather than left to the optimizer, which is the point of the
   cell. gcc -O2 does vectorize the scalar loop this replaces, but an
   auto-vectorization has no name the kernel matrix can register, no
   symbol a cross-toolchain comparison can reach, and it goes away
   silently when a flag or a compiler version changes. The Watcom side
   never had it at all.

   Bit-identical to the scalar form by ops_kernel_shared.h's own
   evidence: lz_i32f's split and the correctly rounded conversion agree
   on 87,108,878 compared values, the whole +-2^25 band included.
   cvtdq2ps IS that conversion. */
void lz_i32f_acc32_simd(float *accf, const int32_t *acc32) {
    int b;
    for (b = 0; b < 8; b++) {
        __m128i vi = _mm_loadu_si128((const __m128i *)(const void *)
                                     (acc32 + b * 4));
        _mm_storeu_ps(accf + b * 4,
                      _mm_add_ps(_mm_loadu_ps(accf + b * 4),
                                 _mm_cvtepi32_ps(vi)));
    }
}

/* ---- the Q15 table interpolation, SSE2 tier.
   Watcom twin: lz_lerp_q15_32_sse2_asm, the #pragma aux in the other
   arm. MMX twin: lz_lerp_q15_mmx (src/ops_mmx.c), which carries the
   derivation and the precondition.

   Eight elements a pass against the MMX cell's four, and no emms: every
   instruction here is the %xmm form, so nothing aliases the x87 stack.
   That is the whole difference - psubd, packssdw, pmullw, pmulhw,
   punpck[lh]wd, psrad and paddd all exist at both widths and the
   sequence is identical otherwise.

   PRECONDITION |b - a| <= 32767, same as the MMX cell: the multiply is
   16-bit and packs_epi32 saturates rather than wrapping. The callers
   meet it because a and b are adjacent entries of a 384-step table.
   `a` itself reaches 32768 and stays int32. */
void lz_lerp_q15_simd(const int32_t *a, const int32_t *b,
                      const int32_t *frac, int32_t *out, int n) {
    int k;

    for (k = 0; k + 7 < n; k += 8) {
        __m128i a0 = _mm_loadu_si128((const __m128i *)(const void *)(a + k));
        __m128i a1 = _mm_loadu_si128((const __m128i *)(const void *)(a + k + 4));
        __m128i b0 = _mm_loadu_si128((const __m128i *)(const void *)(b + k));
        __m128i b1 = _mm_loadu_si128((const __m128i *)(const void *)(b + k + 4));
        __m128i f0 = _mm_loadu_si128((const __m128i *)(const void *)(frac + k));
        __m128i f1 = _mm_loadu_si128((const __m128i *)(const void *)(frac + k + 4));
        __m128i d  = _mm_packs_epi32(_mm_sub_epi32(b0, a0),
                                     _mm_sub_epi32(b1, a1));
        __m128i f  = _mm_packs_epi32(f0, f1);
        __m128i lo = _mm_mullo_epi16(d, f);
        __m128i hi = _mm_mulhi_epi16(d, f);
        __m128i p0 = _mm_srai_epi32(_mm_unpacklo_epi16(lo, hi), 15);
        __m128i p1 = _mm_srai_epi32(_mm_unpackhi_epi16(lo, hi), 15);
        _mm_storeu_si128((__m128i *)(void *)(out + k),
                         _mm_add_epi32(p0, a0));
        _mm_storeu_si128((__m128i *)(void *)(out + k + 4),
                         _mm_add_epi32(p1, a1));
    }
    for (; k < n; k++)
        out[k] = a[k] + (((b[k] - a[k]) * frac[k]) >> 15);
}

/* ---- lz_exp_fixed's Q20 Taylor, SSE2 tier.
   Watcom twin: lz_exp_q20_sse2_asm, the #pragma aux in the other arm.

   prod = (tab * cq + 2^19) >> 20, where
   cq = 2^20 + ((LN2*s + 2^19) >> 20) + ((LN2SQ*(s*s) + 2^39) >> 40).

   FOUR 32 x 32 -> 64 MULTIPLIES AND NOTHING WIDER, which is the fact
   that makes this cell exist. The C writes the second Taylor term as
   ((int64)LN2SQ * s) * s, whose first product is 2^33 and whose second
   would therefore be 64 x 32 - but s = rq >> 5 with rq <= 2^20, so
   s <= 2^15 and s*s <= 2^30 fits int32. Grouped as LN2SQ * (s*s) both
   factors are under 2^32, pmuludq takes it, and the two groupings are
   the same number: integer multiplication is associative and exact and
   every intermediate here is at most 2^48.

   Every operand is non-negative, so pmuludq's unsignedness costs
   nothing. r is [0,1) by construction so rq >= 0; LN2 and LN2SQ are
   positive constants; g_exptab holds exp() times 2^20; cq is 2^20 plus
   two non-negative terms.

   pmuludq works on the EVEN 32-bit lanes only, so each multiply runs
   twice - once on the vector and once on it shifted down by one lane.

   The two halves come back together with pslldq+por, not with a pair of
   unpacks: every result here fits int32, so the high half of each
   64-bit lane is zero, and the odd half shifted up by one lane lands
   exactly in the gaps the even half left. Two instructions instead of
   three, and - the reason it is written this way rather than as a
   micro-optimization - one fewer live register, which is what lets the
   Watcom twin hold all four of its constants in %xmm4-7 at once.

   THE 2^20 OF cq IS FOLDED INTO THE SECOND TAYLOR TERM'S ROUNDING
   CONSTANT for the same register reason: adding 2^60 before a >>40 adds
   exactly 2^20 after it and cannot disturb the low bits, so the
   separate broadcast constant disappears. The largest intermediate this
   creates is 2^60 + LN2SQ*(s*s) < 2^61, well inside the unsigned 64-bit
   lane psrlq works on. */
static __m128i exp_mul_shr(__m128i v, __m128i w, __m128i rnd, int sh) {
    /* BOTH operands take the lane shift, not just the first: pmuludq
       reads lanes 0 and 2, so the odd pass has to bring the odd lanes
       of each side down together. A broadcast constant survives the
       shift unchanged in those two lanes, which is why this one helper
       serves the constant multiplies and the per-element ones alike. */
    __m128i e = _mm_srli_epi64(_mm_add_epi64(_mm_mul_epu32(v, w), rnd), sh);
    __m128i o = _mm_srli_epi64(
        _mm_add_epi64(_mm_mul_epu32(_mm_srli_si128(v, 4),
                                    _mm_srli_si128(w, 4)), rnd), sh);
    return _mm_or_si128(e, _mm_slli_si128(o, 4));
}

void lz_exp_q20_simd(const int32_t *tab, const int32_t *s,
                     int32_t *prod, int n) {
    const __m128i kln2   = _mm_set1_epi32(726817);
    const __m128i kln2sq = _mm_set1_epi32(251906);
    const __m128i r20    = _mm_set1_epi64x((long long)1 << 19);
    const __m128i r40q   = _mm_set1_epi64x(((long long)1 << 39)
                                         + ((long long)1 << 60));
    const __m128i zero   = _mm_setzero_si128();
    int k;

    for (k = 0; k + 3 < n; k += 4) {
        __m128i sv = _mm_loadu_si128((const __m128i *)(const void *)(s + k));
        __m128i tv = _mm_loadu_si128((const __m128i *)(const void *)(tab + k));
        __m128i ss = exp_mul_shr(sv, sv, zero, 0);   /* s*s, exact in int32 */
        __m128i t1 = exp_mul_shr(sv, kln2,   r20,  20);
        __m128i t2 = exp_mul_shr(ss, kln2sq, r40q, 40);  /* carries the 2^20 */
        __m128i cq = _mm_add_epi32(t1, t2);
        _mm_storeu_si128((__m128i *)(void *)(prod + k),
                         exp_mul_shr(tv, cq, r20, 20));
    }
    for (; k < n; k++) {
        int32_t sk = s[k], cq;
        cq = (1 << 20)
           + (int32_t)((((lz_i64)726817 * sk) + (1 << 19)) >> 20)
           + (int32_t)((((lz_i64)251906 * ((lz_i64)sk * sk))
                        + (LZ_I64_C(1) << 39)) >> 40);
        prod[k] = (int32_t)((((lz_i64)tab[k] * cq) + (1 << 19)) >> 20);
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

/* ---- norm_ss_fixed's element loop, SSE2 gcc twin.
   Watcom twin: lz_norm_ss_sse2_asm, the #pragma aux in the branch
   above. Consumes whole groups of 8 and returns how many it took; the
   caller runs the remainder in C.

   Three steps, one instruction each, which is why this cell is worth
   filling and the scalar leaves next to it are not:
     x[i] * sc          mulps
     q8_round           cvtps2dq - round-to-nearest-even is the DEFAULT
                        MXCSR mode, and that is q8_round's own rule, so
                        the magic add is not being approximated here,
                        it is being spelled a second way
     v * v accumulated  pmaddwd, which is exactly a pairwise sum of
                        products and therefore exactly this loop's
                        accumulate

   The upper clamp is free. packssdw saturates int32 to [-32768, 32767],
   so `if (v > 32767) v = 32767` is the pack itself; only the low end
   needs fixing, from -32768 to the C's -32767, and that is one pmaxsw.

   No overflow in the pairwise sum, and it is close enough to be worth
   the arithmetic: |v| <= 32767 gives v*v <= 1073676289, and pmaddwd
   adds two of them - 2147352578, which is 131069 under INT32_MAX. So
   the pair is safe and the NEXT add is not, which is why each pmaddwd
   result is widened to int64 (punpck against zero, the squares being
   non-negative) before it joins the running sum.

   Bit-identical to the C by the same argument the MMX Hadamard uses for
   its half: integer addition is associative and exact, so the order the
   squares are summed in cannot change the total. */
int lz_norm_ss_sse2(const float *x, int n, float sc, short *qout,
                    lz_i64 *acc) {
    __m128i vacc = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();
    const __m128i lo = _mm_set1_epi16((short)-32767);
    const __m128 vsc = _mm_set1_ps(sc);
    union { __m128i v; lz_i64 q[2]; } out;
    int i, ng = n & ~7;
    for (i = 0; i < ng; i += 8) {
        __m128i a = _mm_cvtps_epi32(_mm_mul_ps(
            _mm_loadu_ps(x + i), vsc));
        __m128i b = _mm_cvtps_epi32(_mm_mul_ps(
            _mm_loadu_ps(x + i + 4), vsc));
        __m128i w = _mm_max_epi16(_mm_packs_epi32(a, b), lo);
        __m128i p;
        if (qout) _mm_storeu_si128((__m128i *)(void *)(qout + i), w);
        p = _mm_madd_epi16(w, w);
        vacc = _mm_add_epi64(vacc, _mm_unpacklo_epi32(p, zero));
        vacc = _mm_add_epi64(vacc, _mm_unpackhi_epi32(p, zero));
    }
    out.v = vacc;
    *acc = out.q[0] + out.q[1];
    return ng;
}

/* ---- Hadamard: one butterfly stage of lz_fwht_i32, SSE2 gcc twin.
   Watcom twin: lz_fwht_stage_sse2_asm, the #pragma aux in the branch
   above. Four int32 per xmm where the MMX cell (src/ops_mmx.c) does
   two - same butterfly, same no-shuffle argument, and the same
   by-construction bit-identity: integer add and subtract, wrapping,
   with no rounding and no reassociation.

   len >= 4 is the precondition, not a preference: below it the two
   halves of the butterfly are not whole 16-byte vectors, and src/fwht.c
   hands those stages to the MMX kernel rather than growing a tail here.

   No emms: SSE2 touches no MMX register (that is the whole difference
   from the SSE1+MMX tier), so this cell owes none. The caller's one
   emms is for the len == 2 stage, which IS on MMX. */
void lz_fwht_stage_sse2(int32_t *y, int n, int len) {
    int i, j;
    for (i = 0; i < n; i += (len << 1)) {
        int32_t *a = y + i;
        int32_t *b = y + i + len;
        for (j = 0; j < len; j += 4) {
            __m128i u = _mm_loadu_si128((const __m128i *)(const void *)(a + j));
            __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(b + j));
            _mm_storeu_si128((__m128i *)(void *)(a + j), _mm_add_epi32(u, v));
            _mm_storeu_si128((__m128i *)(void *)(b + j), _mm_sub_epi32(u, v));
        }
    }
}

/* ---- attention's two fixed-point MACs, SSE2 tier ----------------------
   Watcom twins: lz_dot32_x16_sse2 (a wrapper on the existing
   lz_dot32_x16_sse2_asm) and lz_wsum_pair_sse2_asm, both above in this
   file. Read the long comment there for why the dot is a REUSE and the
   weighted sum a real body, and for the bit-identity argument. */

/* part32_x16 leaves four int32 partials; the matmul row loop folds four
   of those together, which this single-group caller has nothing to
   pair with, so it reduces in place. The x256 is part32_x16's own -
   see its comment - and the attention path cancels it with the same
   single 1/256 the matmul row end uses. */
int32_t lz_dot32_x16_sse2(const int8_t *w, const int16_t *x) {
    __m128i p = part32_x16(w, x);
    p = _mm_add_epi32(p, _mm_srli_si128(p, 8));
    p = _mm_add_epi32(p, _mm_srli_si128(p, 4));
    return _mm_cvtsi128_si32(p);
}

void lz_wsum_pair_sse2(const int8_t *rowA, const int8_t *rowB,
                       const int16_t *coef, int32_t *acc32) {
    int32_t cw;
    __m128i c;
    int off;
    memcpy(&cw, coef, 4);           /* (ckA, ckB), one dword */
    c = _mm_set1_epi32(cw);
    for (off = 0; off < 32; off += 16) {
        __m128i ra = _mm_load_si128((const __m128i *)(const void *)(rowA + off));
        __m128i rb = _mm_load_si128((const __m128i *)(const void *)(rowB + off));
        __m128i al = _mm_srai_epi16(_mm_unpacklo_epi8(ra, ra), 8);
        __m128i bl = _mm_srai_epi16(_mm_unpacklo_epi8(rb, rb), 8);
        __m128i ah = _mm_srai_epi16(_mm_unpackhi_epi8(ra, ra), 8);
        __m128i bh = _mm_srai_epi16(_mm_unpackhi_epi8(rb, rb), 8);
        int32_t *acc = acc32 + off;
        _mm_store_si128((__m128i *)(void *)(acc + 0),
            _mm_add_epi32(_mm_load_si128((const __m128i *)(const void *)(acc + 0)),
                          _mm_madd_epi16(_mm_unpacklo_epi16(al, bl), c)));
        _mm_store_si128((__m128i *)(void *)(acc + 4),
            _mm_add_epi32(_mm_load_si128((const __m128i *)(const void *)(acc + 4)),
                          _mm_madd_epi16(_mm_unpackhi_epi16(al, bl), c)));
        _mm_store_si128((__m128i *)(void *)(acc + 8),
            _mm_add_epi32(_mm_load_si128((const __m128i *)(const void *)(acc + 8)),
                          _mm_madd_epi16(_mm_unpacklo_epi16(ah, bh), c)));
        _mm_store_si128((__m128i *)(void *)(acc + 12),
            _mm_add_epi32(_mm_load_si128((const __m128i *)(const void *)(acc + 12)),
                          _mm_madd_epi16(_mm_unpackhi_epi16(ah, bh), c)));
    }
}

/* epi's int32 x int16 -> int64 accumulation, the shape lz_epi_mac_i16
   has. Three madds per group of four and no pmuludq: each operand is
   built with the value in the low 16 bits and zero in the high, so
   _mm_madd_epi16 - which reads both halves as signed int16 and adds
   them - returns v*m per lane, the high half contributing nothing while
   the low keeps the correct two's complement pattern for negatives.

   The split is carry-free on purpose. Folding the borrow into ah is the
   textbook form and it takes ah to 32768 for a near INT_MAX, which is
   not int16 and wraps silently; carrying the borrow as its own term
   keeps every lane in range. S_ah and S_b meet only in 64 bits.

   Products widen every iteration rather than accumulating in int32:
   |ah * m| reaches 2^30, so two of them overflow a lane.

   WHAT THIS COSTS, because it is closer than it looks and nobody
   should assume a vector body wins here. Counted on the shipping 32-bit
   i486 build (.prof/epi_mac_x86_probe.c for the scalar side, objdump
   for both): 35 instructions per four elements against the scalar
   loop's 8 per element - 8.75 against 8.00. The scalar path is not slow
   here, because one-operand imul IS a signed 32x32 -> 64 and needs no
   split at all; what this buys is a quarter of the branches, not fewer
   instructions. Three of the 35 are gcc rebuilding zero and 0xFFFF
   inside the loop, which eight XMM registers leave it no room to hoist
   - the Watcom twin above keeps both resident and lands at 31.

   Predicate: .prof/epi_mac_oracle.c, which checks this against three
   independent scalar derivations over 20106 cases and carries a
   mutation (-DEPI_MUT_SIGNED_LOW) proving it can go red. */
lz_i64 lz_epi_mac_i16_sse2(const int32_t *a, const int16_t *m, int n) {
    const __m128i lo16 = _mm_set1_epi32(0x0000FFFF);
    const __m128i one  = _mm_set1_epi32(1);
    const __m128i zero = _mm_setzero_si128();
    __m128i acc_hi = zero, acc_als = zero;
    lz_i64 t[2], s_hi, s_als;
    int g = 0;

    for (; g + 3 < n; g += 4) {
        __m128i av  = _mm_loadu_si128((const __m128i *)(const void *)(a + g));
        __m128i mv  = _mm_loadl_epi64((const __m128i *)(const void *)(m + g));
        /* No mask on either of these two, and none is needed:
           punpcklwd against zero already leaves the high half clear, and
           a LOGICAL shift down by 16 is what "keep the top halfword,
           zero above it" is - psrad followed by an AND is the same two
           bits of work spelled in two instructions. Only als needs a
           real mask, because its high half is live data. */
        __m128i m32 = _mm_unpacklo_epi16(mv, zero);
        __m128i ah  = _mm_srli_epi32(av, 16);
        __m128i als = _mm_and_si128(av, lo16);
        __m128i bb  = _mm_and_si128(_mm_srli_epi32(av, 15), one);
        /* The borrow joins ah HERE, one add per group, after both have
           left the 16-bit lane. That is the whole reason the carry-free
           form needed a third accumulator, and it does not: what
           overflows int16 is ah+b, not (ah*m)+(b*m). |ah*m| <= 2^30 and
           |b*m| <= 2^15, so the int32 lane has room and one 64-bit
           accumulator is enough, not the two a third-accumulator form
           would need. */
        __m128i pa  = _mm_add_epi32(_mm_madd_epi16(ah, m32),
                                    _mm_madd_epi16(bb, m32));
        __m128i pl  = _mm_madd_epi16(als, m32);
        acc_hi  = _mm_add_epi64(acc_hi,
                  _mm_unpacklo_epi32(pa, _mm_srai_epi32(pa, 31)));
        acc_hi  = _mm_add_epi64(acc_hi,
                  _mm_unpackhi_epi32(pa, _mm_srai_epi32(pa, 31)));
        acc_als = _mm_add_epi64(acc_als,
                  _mm_unpacklo_epi32(pl, _mm_srai_epi32(pl, 31)));
        acc_als = _mm_add_epi64(acc_als,
                  _mm_unpackhi_epi32(pl, _mm_srai_epi32(pl, 31)));
    }
    _mm_storeu_si128((__m128i *)(void *)t, acc_hi);  s_hi  = t[0] + t[1];
    _mm_storeu_si128((__m128i *)(void *)t, acc_als); s_als = t[0] + t[1];
    /* The tail is the plain product, not a fourth copy of the split.
       ((ah + b) << 16) + als IS a, by construction, so an element that
       never enters a lane has no reason to be taken apart - and the
       three-term version that was here computed the same number in
       eleven operations instead of one. */
    s_hi = (s_hi << 16) + s_als;
    for (; g < n; g++) s_hi += (lz_i64)a[g] * (lz_i64)m[g];
    return s_hi;
}

#endif /* __WATCOMC__ */

#endif /* LZ_SSE2_TU */
