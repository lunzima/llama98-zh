/* Every 32-element dot-product kernel, all ISAs and both impls. Included
   once by ops.c; see the header comment of ops_kernel_q8round.h for why
   this is a header rather than a translation unit, and why it has no
   include guards.

   Contents, in file order (grouped by FORMAT with the ISA branches
   nested inside, which is what makes the guards hard to follow):

     part32_* / fold4_sse2          sse2-intrin  (gcc, !LZ_USE_MMX)
     x16 / q16 / q41 / q61          each: #ifdef __WATCOMC__ asm
     nowiden                              #else  mmx-intrin
*/

/* THE GUARD MUST MATCH THE USE SITE. A symbol whose definition is
   compiled under a wider predicate than its caller is either an
   undefined reference in one configuration or dead code in another, and
   both only show up in a build nobody happens to run.

   The scalar reference for a 32-element dot product is
   matmul_scalar_ref in ops.c, not a private copy here: it is the path
   every SIMD tier is judged bit-identical against. Three symbols in
   this file and ops.c had drifted wider than their callers, and the
   ARMv5TE cross-build - where neither __SSE2__ nor __MMX__ is defined -
   is the configuration that reports them. */

/* fold4_sse2 and the five part32_* kernels (x16/q41/q61/t2/q16) moved to
   src/ops_kernel_dot_sse2.h, included only by src/ops_sse2.c - they
   write %xmm registers. Declared (as the row_*_sse2_intrin functions
   that call them) in src/ops_sse2.h, visible in ops.c. */

/* MMX dot product with activations pre-expanded to int16, same
   algorithm as SSE2's part32_x16: weights via punpcklbw(0, w) (value
   x256, skips psraw), activations loaded directly as int16. SIMD
   instruction count per 32 elements drops from ~57 to ~35. Products
   scale 256x, cancelled once at the row end by the caller.

   dot32_x16_mmx / dot32_x16_mmx_2's gcc bodies live in
   src/ops_kernel_dot_mmx.h now (included only by src/ops_mmx.c - both
   write %mm registers); this #ifdef's Watcom half is unaffected and
   unchanged. */
#ifdef __WATCOMC__
#include "mmx_compat.h"
/* __modify must be truthful; never write __exact [].
   `__exact []` means "this code touches no register" - a lie: the
   asm writes eax (the return value), MMX/XMM, and MMX also moves the
   x87 stack. -oe (inline expansion) or -ol (loop optimization) alone
   are fine; together (-otel, -ox, -otexan) it blows up: after inlining,
   the loop optimizer keeps loop variables in eax across the asm
   because it believes nothing changed.
   The symptom is deeply misleading - end-to-end logits become
   nondeterministic garbage (three runs of the same exe give rms
   5418/5415/5412, drifting each time), while kernel-level differential
   tests of 200K groups all pass, because that call shape is simple and
   the optimizer has nothing to move.
   Writing [8087] makes Watcom treat it conservatively as "modified FP
   stack + default volatile registers". */
/* Watcom hand-written MMX dot product: 32 elements (int8 weights x
   int16 activations), pure asm pipeline. Same algorithm as the gcc
   version below (w expanded via punpcklbw(zero,w) = w<<8, cancelled by
   the caller's 1/256 at the row end). Written as a single pragma aux
   function to avoid Watcom's __m64 struct-return stack round trip
   (measured: the intrinsics version was 6x slower than x87; this one
   40x faster than it). Correctness: bit-matches the scalar over 100K
   random/extreme groups. eax=w, edx=x, returns eax;
   contains an emms (the caller's row-level EMMS is idempotent, no
   harm). */
/* Originally generated, hand-edited since - if you regenerate, diff
   the generator's output against what is here first.
   Kernels do NOT emit emms: the loop body is all integer and MMX; x87
   is only used in the whole-row reduction, and the caller already does
   one _mm_empty() after the loop. One emms per 32 elements would be
   pure redundancy - ~1.8M sub-block calls per token, at 6 cycles on a
   P6 that is 36 ms / 16% of the compute budget, worse on P4.
   Changing this REQUIRES proving the caller's loop body has no x87.
   Differential-checked over 200K groups (random + extremes,
   bit-identical to the scalar reference).
   Counts (µop, P6 accounting): Q8_0 52, Q4_1 63, Q6_1 86. */
extern int32_t lz_dot32_x16_asm(const int8_t *w, const int16_t *x);
#pragma aux lz_dot32_x16_asm = \
    ".586" \
    "pxor      mm7, mm7" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+8]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+8]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+16]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+24]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+16]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+32]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+40]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+24]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+48]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+56]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      eax, mm6" \
    __parm [__eax] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* PAIRED twin of the above: two tokens, ONE weight unpack.
   The gcc side is dot32_x16_mmx_2; this is the Watcom half, i.e. the
   half that runs on the target. Q8's weight side is 12 of 36 ops per
   32 MACs, so sharing it across NT tokens costs 12 + 24*NT instead of
   36*NT - ceiling 1.50x, 1.20x at NT=2.

   Register budget is why NT is 2 and not more: mm7 zero, mm6/mm5 the
   two accumulators, mm1 the raw weight octet, mm2 the widened half,
   mm3 its copy for the second token - six of eight, with none to spare
   for a third token's accumulator plus its activation.

   No emms here, by rule 6.6: the caller emits one after the row loop.

   VERIFIED BIT-IDENTICAL: 200000 random plus 8 extreme trials against
   the scalar reference, 0 mismatches - and the
   differential's own sensitivity checked by mutating one operand offset
   ([ebx+8] -> [ebx+16]), which turned it into 200000 mismatches. */
extern void lz_dot32_x16_asm_2(const int8_t *w, const int16_t *xa,
                               const int16_t *xb, int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_x16_asm_2 = \
    ".586" \
    "pxor      mm7, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm5, mm5" \
    "movq      mm1, [eax]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+8]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+8]" \
    "paddd     mm5, mm3" \
    "movq      mm1, [eax+8]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+16]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+16]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+24]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+24]" \
    "paddd     mm5, mm3" \
    "movq      mm1, [eax+16]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+32]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+32]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+40]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+40]" \
    "paddd     mm5, mm3" \
    "movq      mm1, [eax+24]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+48]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+48]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+56]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+56]" \
    "paddd     mm5, mm3" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ecx], mm6" \
    "movq      mm1, mm5" \
    "psrlq     mm1, 32" \
    "paddd     mm5, mm1" \
    "movd      [esi], mm5" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [8087]

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

/* Both are compiled in; the runtime picks by CPUID (see
   lz_kernel_select). One binary must run on both PII (MMX only) and
   T42 (SSE2). */
static int32_t dot32_x16_sse2a(const int8_t *w, const int16_t *x) {
    return lz_dot32_x16_sse2_asm(w, x);
}
static int32_t dot32_x16_mmx(const int8_t *w, const int16_t *x) {
    return lz_dot32_x16_asm(w, x);
}
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_x16_mmx (single token) and dot32_x16_mmx_2 (two tokens, ONE
   weight unpack - 36/24 ceiling, NT=2 register budget) live in
   src/ops_kernel_dot_mmx.h, included only by src/ops_mmx.c. Declared in
   src/ops_mmx.h, already visible in this translation unit. */
#endif

/* ---- Q16_0 32/128-element kernels -------------------------------------
   Q16_0 stores int16 weights, so unlike Q8_0 there is no int8->int16
   unpack step and therefore no x256 product scaling: these return the
   raw int32 dot, and the float epilogue has no 1/256 to cancel.

   Accumulator bound is 32767*127*32 = 1.33e8, well past 2^24, so every
   int32->float conversion downstream MUST go through lz_i32f(). This is
   not a precaution: on 4096 real out_proj sub-blocks with |acc| > 2^24,
   a bare (float) cast disagrees between the two compilers on 1210 of
   them (29.5%, always exactly 1 ULP), while lz_i32f() disagrees on 0.
   See rule 7 for why the divergence is in the load, not the arithmetic,
   and why PC=24 does not cover it. */
#if defined(__WATCOMC__)
/* 32-element version: same name/parameter shape as lz_dot32_x16_asm, called directly from the matmul inner loop. */
extern int32_t lz_dot32_q16_asm(const int16_t *w, const int16_t *x);
#pragma aux lz_dot32_q16_asm = \
    ".586" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax]" \
    "pmaddwd   mm1, [ecx]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+8]" \
    "pmaddwd   mm1, [ecx+8]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+16]" \
    "pmaddwd   mm1, [ecx+16]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+24]" \
    "pmaddwd   mm1, [ecx+24]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+32]" \
    "pmaddwd   mm1, [ecx+32]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+40]" \
    "pmaddwd   mm1, [ecx+40]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+48]" \
    "pmaddwd   mm1, [ecx+48]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+56]" \
    "pmaddwd   mm1, [ecx+56]" \
    "paddd     mm6, mm1" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      eax, mm6" \
    __parm [__eax] [__ecx] \
    __value [__eax] \
    __modify [8087]

extern int32_t lz_dot32_q16_sse2_asm(const int16_t *w, const int16_t *x);
#pragma aux lz_dot32_q16_sse2_asm = \
    ".686" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax]" \
    "movdqu    xmm1, [ecx]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+16]" \
    "movdqu    xmm1, [ecx+16]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+32]" \
    "movdqu    xmm1, [ecx+32]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+48]" \
    "movdqu    xmm1, [ecx+48]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      eax, xmm4" \
    __parm [__eax] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* Group-at-once (128 elements, 4 subblocks) version: same shape as
   lz_dot128_x16_asm. The 4 subblock sums are written out separately
   (each has its own xqs, so the float reduction cannot add them up
   inside the kernel). */
extern void lz_dot128_q16_asm(const int16_t *w, const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q16_asm = \
    ".586" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax]" \
    "pmaddwd   mm1, [ecx]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+8]" \
    "pmaddwd   mm1, [ecx+8]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+16]" \
    "pmaddwd   mm1, [ecx+16]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+24]" \
    "pmaddwd   mm1, [ecx+24]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+32]" \
    "pmaddwd   mm1, [ecx+32]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+40]" \
    "pmaddwd   mm1, [ecx+40]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+48]" \
    "pmaddwd   mm1, [ecx+48]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+56]" \
    "pmaddwd   mm1, [ecx+56]" \
    "paddd     mm6, mm1" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax+64]" \
    "pmaddwd   mm1, [ecx+64]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+72]" \
    "pmaddwd   mm1, [ecx+72]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+80]" \
    "pmaddwd   mm1, [ecx+80]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+88]" \
    "pmaddwd   mm1, [ecx+88]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+96]" \
    "pmaddwd   mm1, [ecx+96]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+104]" \
    "pmaddwd   mm1, [ecx+104]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+112]" \
    "pmaddwd   mm1, [ecx+112]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+120]" \
    "pmaddwd   mm1, [ecx+120]" \
    "paddd     mm6, mm1" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+4], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax+128]" \
    "pmaddwd   mm1, [ecx+128]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+136]" \
    "pmaddwd   mm1, [ecx+136]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+144]" \
    "pmaddwd   mm1, [ecx+144]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+152]" \
    "pmaddwd   mm1, [ecx+152]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+160]" \
    "pmaddwd   mm1, [ecx+160]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+168]" \
    "pmaddwd   mm1, [ecx+168]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+176]" \
    "pmaddwd   mm1, [ecx+176]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+184]" \
    "pmaddwd   mm1, [ecx+184]" \
    "paddd     mm6, mm1" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+8], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax+192]" \
    "pmaddwd   mm1, [ecx+192]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+200]" \
    "pmaddwd   mm1, [ecx+200]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+208]" \
    "pmaddwd   mm1, [ecx+208]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+216]" \
    "pmaddwd   mm1, [ecx+216]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+224]" \
    "pmaddwd   mm1, [ecx+224]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+232]" \
    "pmaddwd   mm1, [ecx+232]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+240]" \
    "pmaddwd   mm1, [ecx+240]" \
    "paddd     mm6, mm1" \
    "movq      mm1, [eax+248]" \
    "pmaddwd   mm1, [ecx+248]" \
    "paddd     mm6, mm1" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+12], mm6" \
    "emms" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

extern void lz_dot128_q16_sse2_asm(const int16_t *w, const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q16_sse2_asm = \
    ".686" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax]" \
    "movdqu    xmm1, [ecx]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+16]" \
    "movdqu    xmm1, [ecx+16]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+32]" \
    "movdqu    xmm1, [ecx+32]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+48]" \
    "movdqu    xmm1, [ecx+48]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      [ebx], xmm4" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax+64]" \
    "movdqu    xmm1, [ecx+64]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+80]" \
    "movdqu    xmm1, [ecx+80]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+96]" \
    "movdqu    xmm1, [ecx+96]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+112]" \
    "movdqu    xmm1, [ecx+112]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      [ebx+4], xmm4" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax+128]" \
    "movdqu    xmm1, [ecx+128]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+144]" \
    "movdqu    xmm1, [ecx+144]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+160]" \
    "movdqu    xmm1, [ecx+160]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+176]" \
    "movdqu    xmm1, [ecx+176]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      [ebx+8], xmm4" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax+192]" \
    "movdqu    xmm1, [ecx+192]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+208]" \
    "movdqu    xmm1, [ecx+208]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+224]" \
    "movdqu    xmm1, [ecx+224]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqu    xmm0, [eax+240]" \
    "movdqu    xmm1, [ecx+240]" \
    "pmaddwd   xmm0, xmm1" \
    "paddd     xmm4, xmm0" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      [ebx+12], xmm4" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

static int32_t dot32_q16_mmx(const int16_t *w, const int16_t *x) {
    return lz_dot32_q16_asm(w, x);
}
static int32_t dot32_q16_sse2a(const int16_t *w, const int16_t *x) {
    return lz_dot32_q16_sse2_asm(w, x);
}
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_q16_mmx's body lives in src/ops_kernel_dot_mmx.h, included only
   by src/ops_mmx.c - it writes %mm registers. Declared in
   src/ops_mmx.h, already visible in this translation unit. */
#endif /* Q16_0 kernels */

/* Q4_1 32-element dot product (MMX). Same algorithm as SSE2's
   part32_q41: pand(b,0x0F) grabs elements 0..15,
   pand(psrlw(b,4),0x0F) elements 16..31, then punpcklbw(0,·) expands
   to x256 int16 for pmaddwd. Returns the 32-element dot scaled x256;
   the caller cancels once at the row end. */
#ifdef __WATCOMC__
/* Watcom hand-written version, same rationale as lz_dot32_x16_asm
   (_m_* intrinsics are 6x slower than its own x87 scalar).
   Mask built on the fly instead of a memory load: all-ones -> psrlw 12
   gives 0x000F x4 -> packuswb saturating pack to 0x0F x8 - three
   instructions replacing one memory operand.
   Registers: mm0=low nibbles mm4=high nibbles mm1=activations
   mm2=expand temp mm3=mask mm6=accumulator mm7=zero mm5=reduce temp -
   exactly 8.
   Verified: bit-matches the scalar over 200K random + all-15 x
   all-±127 extreme groups; measured 3.75 ns/32MAC vs x87
   scalar 23.50 ns (6.3x). */
/* Originally generated, hand-edited since - if you regenerate, diff
   the generator's output against what is here first.
   Differential-checked over 200K groups (random + extremes,
   bit-identical to the scalar reference).
   Counts (µop, P6 accounting): Q8_0 52, Q4_1 63, Q6_1 86. */
extern int32_t lz_dot32_q41_asm(const unsigned char *w, const int16_t *x);
#pragma aux lz_dot32_q41_asm = \
    ".586" \
    "pxor      mm7, mm7" \
    "pxor      mm6, mm6" \
    "pcmpeqb   mm5, mm5" \
    "psrlw     mm5, 12" \
    "packuswb  mm5, mm5" \
    "movq      mm3, [eax]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+8]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+32]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+40]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+8]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+16]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+24]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+48]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+56]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      eax, mm6" \
    __parm [__eax] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* T42-tier Q4_1 (same: 128-bit, no EMMS needed, same name + signature reused) */
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

/* PAIRED twin: two tokens, ONE nibble unpack. Q4_1's weight side is 16
   of the 40 ops per 32 MACs (2 loads + 6 mask/shift + 8 punpck) - a
   nibble has to be split before it can be widened - so sharing it costs
   16 + 24*NT instead of 40*NT: ceiling 1.67x, 1.25x at NT=2. The gcc
   twin is dot32_q41_mmx_2; this is the half that runs on the target.

   mm0 holds 0x0F x8, built without touching a general register:
   pcmpeqb gives all ones, psrlw 12 leaves 0x000F per word, packuswb
   saturates those to 0x0F per byte. Register budget: mm0 mask, mm7
   zero, mm6/mm5 accumulators, mm1 lo nibbles, mm4 hi nibbles, mm2
   widened half, mm3 its copy for the second token - all eight.

   No emms here (rule 6.6: the caller emits one after the row loop).

   VERIFIED BIT-IDENTICAL: 200000 random plus 8 extreme trials against
   a scalar reference, 0 mismatches; the differential's
   own sensitivity checked by mutating one operand offset ([edx+32] ->
   [edx+48]), which turned it into 199998 mismatches. */
extern void lz_dot32_q41_asm_2(const unsigned char *w, const int16_t *xa,
                               const int16_t *xb, int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_q41_asm_2 = \
    ".586" \
    "pcmpeqb   mm0, mm0" \
    "psrlw     mm0, 12" \
    "packuswb  mm0, mm0" \
    "pxor      mm7, mm7" \
    "pxor      mm6, mm6" \
    "pxor      mm5, mm5" \
    "movq      mm1, [eax]" \
    "movq      mm4, mm1" \
    "pand      mm1, mm0" \
    "psrlw     mm4, 4" \
    "pand      mm4, mm0" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+8]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+8]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+32]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+32]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+40]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+40]" \
    "paddd     mm5, mm3" \
    "movq      mm1, [eax+8]" \
    "movq      mm4, mm1" \
    "pand      mm1, mm0" \
    "psrlw     mm4, 4" \
    "pand      mm4, mm0" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+16]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+16]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+24]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+24]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+48]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+48]" \
    "paddd     mm5, mm3" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm4" \
    "movq      mm3, mm2" \
    "pmaddwd   mm2, [edx+56]" \
    "paddd     mm6, mm2" \
    "pmaddwd   mm3, [ebx+56]" \
    "paddd     mm5, mm3" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ecx], mm6" \
    "movq      mm1, mm5" \
    "psrlq     mm1, 32" \
    "paddd     mm5, mm1" \
    "movd      [esi], mm5" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] \
    __modify [8087]

static int32_t dot32_q41_sse2a(const unsigned char *w, const int16_t *x) {
    return lz_dot32_q41_sse2_asm(w, x);
}
static int32_t dot32_q41_mmx(const unsigned char *w, const int16_t *x) {
    return lz_dot32_q41_asm(w, x);
}
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_q41_mmx (single token) and dot32_q41_mmx_2 (two tokens, ONE
   nibble unpack - the kernel that matters most, see the Watcom twin's
   comment) live in src/ops_kernel_dot_mmx.h, included only by
   src/ops_mmx.c. Declared in src/ops_mmx.h, already visible here. */
#endif /* Q4_1 __WATCOMC__ branch end */

/* ---- T2 (ternary, one 2-bit plane) ------------------------------------ */

/* Originally emitted by tools/kernelgen/gen_t2.py IN THE TRAINING TREE -
   the engine repository carries no tools, so the generator lives there
   and only its output is pasted here. The generator is scaffolding, not
   the source of truth: hand-editing is fine, but if you regenerate,
   diff against what is here first - anything hand-tuned since will be
   silently reverted otherwise.

   MMX 70 instructions / 9 loads; SSE2 47 / 5. For scale, Q6_1 is 86 uop
   / 11 loads and 59 / 6 - T2 is the same kernel with the 4-bit plane
   deleted, which is exactly why it was given Q6_1'''s 2-bit layout.

   VERIFIED BIT-IDENTICAL BEFORE BEING MOVED HERE:
   tools/kernelgen/wt2.c compares both against the scalar reference over
   a 32-position sweep (one nonzero weight at a time, both signs - a
   uniform block hides a wrong shift), six extreme patterns, and 200,000
   random trials. All three agree exactly; int32 accumulation is exact
   so there is no tolerance to argue about. */

#if defined(__WATCOMC__)
extern int32_t lz_dot32_t2_asm(const unsigned char *w2,
                               const int16_t *x);
#pragma aux lz_dot32_t2_asm = \
    ".586" \
    "pxor      mm6, mm6" \
    "movq      mm4, [edx]" \
    "movq      mm0, [ecx]" \
    "pxor      mm2, mm2" \
    "punpcklbw mm2, mm4" \
    "psllw     mm2, 6" \
    "psrlw     mm2, 14" \
    "psllw     mm2, 8" \
    "pmaddwd   mm2, mm0" \
    "paddd     mm6, mm2" \
    "movq      mm0, [ecx+8]" \
    "pxor      mm2, mm2" \
    "punpckhbw mm2, mm4" \
    "psllw     mm2, 6" \
    "psrlw     mm2, 14" \
    "psllw     mm2, 8" \
    "pmaddwd   mm2, mm0" \
    "paddd     mm6, mm2" \
    "movq      mm0, [ecx+16]" \
    "pxor      mm2, mm2" \
    "punpcklbw mm2, mm4" \
    "psllw     mm2, 4" \
    "psrlw     mm2, 14" \
    "psllw     mm2, 8" \
    "pmaddwd   mm2, mm0" \
    "paddd     mm6, mm2" \
    "movq      mm0, [ecx+24]" \
    "pxor      mm2, mm2" \
    "punpckhbw mm2, mm4" \
    "psllw     mm2, 4" \
    "psrlw     mm2, 14" \
    "psllw     mm2, 8" \
    "pmaddwd   mm2, mm0" \
    "paddd     mm6, mm2" \
    "movq      mm0, [ecx+32]" \
    "pxor      mm2, mm2" \
    "punpcklbw mm2, mm4" \
    "psllw     mm2, 2" \
    "psrlw     mm2, 14" \
    "psllw     mm2, 8" \
    "pmaddwd   mm2, mm0" \
    "paddd     mm6, mm2" \
    "movq      mm0, [ecx+40]" \
    "pxor      mm2, mm2" \
    "punpckhbw mm2, mm4" \
    "psllw     mm2, 2" \
    "psrlw     mm2, 14" \
    "psllw     mm2, 8" \
    "pmaddwd   mm2, mm0" \
    "paddd     mm6, mm2" \
    "movq      mm0, [ecx+48]" \
    "pxor      mm2, mm2" \
    "punpcklbw mm2, mm4" \
    "psrlw     mm2, 14" \
    "psllw     mm2, 8" \
    "pmaddwd   mm2, mm0" \
    "paddd     mm6, mm2" \
    "movq      mm0, [ecx+56]" \
    "pxor      mm2, mm2" \
    "punpckhbw mm2, mm4" \
    "psrlw     mm2, 14" \
    "psllw     mm2, 8" \
    "pmaddwd   mm2, mm0" \
    "paddd     mm6, mm2" \
    "movq      mm0, mm6" \
    "psrlq     mm0, 32" \
    "paddd     mm6, mm0" \
    "movd      eax, mm6" \
    "emms" \
    __parm [__edx] [__ecx] \
    __value [__eax] \
    __modify [8087]

extern int32_t lz_dot32_t2_sse2_asm(const unsigned char *w2,
                                    const int16_t *x);
#pragma aux lz_dot32_t2_sse2_asm = \
    ".686" \
    "pxor      xmm4, xmm4" \
    "pxor      xmm5, xmm5" \
    "pcmpeqb   xmm7, xmm7" \
    "psrlw     xmm7, 14" \
    "packuswb  xmm7, xmm7" \
    "movq      xmm3, [edx]" \
    "movdqa    xmm0, xmm3" \
    "pand      xmm0, xmm7" \
    "movdqa    xmm1, xmm3" \
    "psrlw     xmm1, 2" \
    "pand      xmm1, xmm7" \
    "punpcklqdq xmm0, xmm1" \
    "movdqa    xmm1, xmm3" \
    "psrlw     xmm1, 4" \
    "pand      xmm1, xmm7" \
    "movdqa    xmm2, xmm3" \
    "psrlw     xmm2, 6" \
    "pand      xmm2, xmm7" \
    "punpcklqdq xmm1, xmm2" \
    "movdqa    xmm2, xmm4" \
    "punpcklbw xmm2, xmm0" \
    "movdqu    xmm6, [ecx+0]" \
    "pmaddwd   xmm2, xmm6" \
    "paddd     xmm5, xmm2" \
    "movdqa    xmm2, xmm4" \
    "punpckhbw xmm2, xmm0" \
    "movdqu    xmm6, [ecx+16]" \
    "pmaddwd   xmm2, xmm6" \
    "paddd     xmm5, xmm2" \
    "movdqa    xmm2, xmm4" \
    "punpcklbw xmm2, xmm1" \
    "movdqu    xmm6, [ecx+32]" \
    "pmaddwd   xmm2, xmm6" \
    "paddd     xmm5, xmm2" \
    "movdqa    xmm2, xmm4" \
    "punpckhbw xmm2, xmm1" \
    "movdqu    xmm6, [ecx+48]" \
    "pmaddwd   xmm2, xmm6" \
    "paddd     xmm5, xmm2" \
    "movdqa    xmm0, xmm5" \
    "psrldq    xmm0, 8" \
    "paddd     xmm5, xmm0" \
    "movdqa    xmm0, xmm5" \
    "psrldq    xmm0, 4" \
    "paddd     xmm5, xmm0" \
    "movd      eax, xmm5" \
    __parm [__edx] [__ecx] \
    __value [__eax] \
    __modify [8087]

#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_t2_mmx's body lives in src/ops_kernel_dot_mmx.h, included only
   by src/ops_mmx.c - it writes %mm registers. Declared in
   src/ops_mmx.h, already visible in this translation unit. */
#endif  /* __WATCOMC__ / __MMX__ */

/* ---- Q6_1 (6.5 bits: 4-bit + 2-bit dual planes) ------------------------ */
#if defined(__WATCOMC__)
/* Originally generated, hand-edited since - if you regenerate, diff
   the generator's output against what is here first. 86 µop / 11 loads
   (P6 accounting). Only 6 of the 8 MMX registers are used,
   but activations MUST go through pmaddwd's memory operand - once the
   2-bit plane joins, all 8 registers are occupied; none is free to
   hold activations.

   Pentium M (SSE2 tier) temporarily reuses this 64-bit MMX kernel:
   correct, just not using the 128-bit width. Q6_1 hand-written SSE2
   asm is F3's remaining work. */
extern int32_t lz_dot32_q61_asm(const unsigned char *w4, const unsigned char *w2, const int16_t *x);
#pragma aux lz_dot32_q61_asm = \
    ".586" \
    "pxor      mm7, mm7" \
    "pxor      mm6, mm6" \
    "pcmpeqb   mm5, mm5" \
    "psrlw     mm5, 12" \
    "packuswb  mm5, mm5" \
    "pcmpeqb   mm0, mm0" \
    "psrlw     mm0, 14" \
    "packuswb  mm0, mm0" \
    "movq      mm4, [edx]" \
    "movq      mm3, [eax]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+8]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+32]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+40]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+8]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 2" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+16]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+24]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 6" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+48]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+56]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      eax, mm6" \
    __parm [__eax] [__edx] [__ecx] \
    __value [__eax] \
    __modify [8087]

static int32_t dot32_q61_mmx(const unsigned char *w4, const unsigned char *w2,
                             const int16_t *x) {
    return lz_dot32_q61_asm(w4, w2, x);
}

/* Originally generated, hand-edited since - if you regenerate, diff
   the generator's output against what is here first. 59 instructions / 6 loads.

   **Which is faster than the MMX version cannot be decided statically.**
   SSE2 is 59 instructions / ~116 µop; MMX is 78 / 86 µop - +35% µop,
   -24% instructions, the two directions oppose. Pentium M's
   (Banias/Dothan) SSE units are 64-bit wide, splitting 128-bit
   instructions into two µops - but split µops are not necessarily
   slower: Zen 1's AVX-256 also splits into two 128-bit µops with the
   same µop total as equivalent AVX-128 code, yet still wins, on front-
   end bandwidth and paired scheduling. The extra µops are not madds
   (those are even); they are fixed overhead paid twice (masks 6->12,
   epilogue 5->14) plus punpcklqdq, the inherent tax of the 128-bit
   layout. **That fixed overhead is fixable** - processing a whole
   gs=128 group at once amortizes it, and SSE2 benefits twice as much
   as MMX. So the SSE2 tier temporarily still uses the MMX kernel - not
   judged slower, just no target machine to measure on.

   Two SSE2-specific constraints (know before writing):
   - `pmaddwd xmm, m128` requires 16-byte alignment, which activation
     buffers do not guarantee, so `movdqu` to a register then
     `pmaddwd reg,reg` is mandatory - the instructions the MMX version
     saves via memory operands cannot be saved here.
   - 32-bit mode has only 8 xmms; the zero register becomes an in-place
     `pxor xmm2,xmm2` to fit. */
extern int32_t lz_dot32_q61_sse2_asm(const unsigned char *w4, const unsigned char *w2, const int16_t *x);
#pragma aux lz_dot32_q61_sse2_asm = \
    ".686" \
    "pxor      xmm4, xmm4" \
    "pcmpeqb   xmm7, xmm7" \
    "psrlw     xmm7, 12" \
    "packuswb  xmm7, xmm7" \
    "pcmpeqb   xmm6, xmm6" \
    "psrlw     xmm6, 14" \
    "packuswb  xmm6, xmm6" \
    "movdqu    xmm0, [eax]" \
    "movq      xmm1, [edx]" \
    "movdqa    xmm2, xmm1" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 2" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+16]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm1" \
    "psrlw     xmm2, 4" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 6" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+32]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+48]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      eax, xmm4" \
    __parm [__eax] [__edx] [__ecx] \
    __value [__eax] \
    __modify [8087]

/* SSE2 one-group-at-a-time kernels (128 elements). Verified
   bit-identical to the scalar reference over 800032 sub-blocks on both
   compilers.

   Wired in because a real Pentium M measured SSE2 at least 20-30%
   faster than MMX at the kernel level. See the note next to the
   Q6_1 dispatch for how to read the uop counts - they are not the
   deciding criterion across instruction sets. */
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

/* PAIRED twin: two tokens, ONE two-plane merge. Q6_1 gains the most of
   the three formats - the weight side is 61% of the kernel's ops, and
   on the recipe of record it carries the most bytes - so the ceiling is
   2.54x (the gcc twin measured 1.271x on prefill).

   Six parameters, so all six of Watcom's register slots are in use, and
   mm7 is NOT a persistent zero the way it is in the other two twins:
   with m0f, m03, the 2-bit plane, two accumulators and the composed
   6-bit values already resident, keeping a zero would leave nothing for
   the second token's copy of the widened half. `pxor mm4, mm4` before
   each punpck costs one instruction and buys that register back.

   The 2-bit plane's shifts (0/2/4/6) are 16-bit shifts, which is
   byte-wise equivalent here: after `& 3` only each byte's own low two
   bits survive and no shift used reaches across a byte boundary.

   No emms here (rule 6.6: the caller emits one after the row loop).

   VERIFIED BIT-IDENTICAL: 200000 random plus 8 extreme trials against
   a scalar reference, 0 mismatches; sensitivity checked
   by mutating one 2-bit plane shift (psrlw 6 -> psrlw 5), which turned
   it into 199997 mismatches. */
extern void lz_dot32_q61_asm_2(const unsigned char *w4, const unsigned char *w2,
                               const int16_t *xa, const int16_t *xb,
                               int32_t *oa, int32_t *ob);
#pragma aux lz_dot32_q61_asm_2 = \
    ".586" \
    "pcmpeqb   mm0, mm0" \
    "psrlw     mm0, 12" \
    "packuswb  mm0, mm0" \
    "pcmpeqb   mm1, mm1" \
    "psrlw     mm1, 14" \
    "packuswb  mm1, mm1" \
    "movq      mm2, [edx]" \
    "pxor      mm6, mm6" \
    "pxor      mm5, mm5" \
    "movq      mm3, [eax]" \
    "pand      mm3, mm0" \
    "movq      mm4, mm2" \
    "pand      mm4, mm1" \
    "psllw     mm4, 4" \
    "por       mm3, mm4" \
    "pxor      mm4, mm4" \
    "punpcklbw mm4, mm3" \
    "movq      mm7, mm4" \
    "pmaddwd   mm4, [ebx]" \
    "paddd     mm6, mm4" \
    "pmaddwd   mm7, [ecx]" \
    "paddd     mm5, mm7" \
    "pxor      mm4, mm4" \
    "punpckhbw mm4, mm3" \
    "movq      mm7, mm4" \
    "pmaddwd   mm4, [ebx+8]" \
    "paddd     mm6, mm4" \
    "pmaddwd   mm7, [ecx+8]" \
    "paddd     mm5, mm7" \
    "movq      mm3, [eax]" \
    "psrlw     mm3, 4" \
    "pand      mm3, mm0" \
    "movq      mm4, mm2" \
    "psrlw     mm4, 4" \
    "pand      mm4, mm1" \
    "psllw     mm4, 4" \
    "por       mm3, mm4" \
    "pxor      mm4, mm4" \
    "punpcklbw mm4, mm3" \
    "movq      mm7, mm4" \
    "pmaddwd   mm4, [ebx+32]" \
    "paddd     mm6, mm4" \
    "pmaddwd   mm7, [ecx+32]" \
    "paddd     mm5, mm7" \
    "pxor      mm4, mm4" \
    "punpckhbw mm4, mm3" \
    "movq      mm7, mm4" \
    "pmaddwd   mm4, [ebx+40]" \
    "paddd     mm6, mm4" \
    "pmaddwd   mm7, [ecx+40]" \
    "paddd     mm5, mm7" \
    "movq      mm3, [eax+8]" \
    "pand      mm3, mm0" \
    "movq      mm4, mm2" \
    "psrlw     mm4, 2" \
    "pand      mm4, mm1" \
    "psllw     mm4, 4" \
    "por       mm3, mm4" \
    "pxor      mm4, mm4" \
    "punpcklbw mm4, mm3" \
    "movq      mm7, mm4" \
    "pmaddwd   mm4, [ebx+16]" \
    "paddd     mm6, mm4" \
    "pmaddwd   mm7, [ecx+16]" \
    "paddd     mm5, mm7" \
    "pxor      mm4, mm4" \
    "punpckhbw mm4, mm3" \
    "movq      mm7, mm4" \
    "pmaddwd   mm4, [ebx+24]" \
    "paddd     mm6, mm4" \
    "pmaddwd   mm7, [ecx+24]" \
    "paddd     mm5, mm7" \
    "movq      mm3, [eax+8]" \
    "psrlw     mm3, 4" \
    "pand      mm3, mm0" \
    "movq      mm4, mm2" \
    "psrlw     mm4, 6" \
    "pand      mm4, mm1" \
    "psllw     mm4, 4" \
    "por       mm3, mm4" \
    "pxor      mm4, mm4" \
    "punpcklbw mm4, mm3" \
    "movq      mm7, mm4" \
    "pmaddwd   mm4, [ebx+48]" \
    "paddd     mm6, mm4" \
    "pmaddwd   mm7, [ecx+48]" \
    "paddd     mm5, mm7" \
    "pxor      mm4, mm4" \
    "punpckhbw mm4, mm3" \
    "movq      mm7, mm4" \
    "pmaddwd   mm4, [ebx+56]" \
    "paddd     mm6, mm4" \
    "pmaddwd   mm7, [ecx+56]" \
    "paddd     mm5, mm7" \
    "movq      mm4, mm6" \
    "psrlq     mm4, 32" \
    "paddd     mm6, mm4" \
    "movd      [esi], mm6" \
    "movq      mm4, mm5" \
    "psrlq     mm4, 32" \
    "paddd     mm5, mm4" \
    "movd      [edi], mm5" \
    __parm [__eax] [__edx] [__ebx] [__ecx] [__esi] [__edi] \
    __modify [8087]

static int32_t dot32_q61_sse2a(const unsigned char *w4, const unsigned char *w2,
                               const int16_t *x) {
    return lz_dot32_q61_sse2_asm(w4, w2, x);
}

/* "one group at a time" version: computes 4 sub-block sums of 128
   elements in one pass.

   Motivation: amortize the FIXED overhead - mask construction (6
   instructions), prologue/epilogue, paid once per 128 elements instead
   of once per 32. SSE2 benefits twice as much as MMX (its
   share is paid at 2 µop). Measured per 128 elements:
     MMX  316 -> 284 instructions (-10%), 344 -> 316 µop (-8%)
     SSE2 236 -> 215 instructions (-9%), 464 -> 430 µop (-7%)

   **The 4 sub-blocks' int32s must be written out separately, not
   summed inside the kernel**: each sub-block has its own activation
   scale xqs[sb]; the float reduction must go per sub-block. Hence the
   signature writes out4[4] instead of returning a value.

   The 4th pointer arg uses ebx; declared in __parm, Watcom handles its
   preservation.

   Gate: 100K groups x 4 sub-blocks, MMX and SSE2 both bit-identical
   to the scalar reference. */
extern void lz_dot128_x16_asm(const int8_t *w,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_x16_asm = \
    ".586" \
    "pxor      mm7, mm7" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+8]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+8]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+16]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+24]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+16]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+32]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+40]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+24]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+48]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+56]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax+32]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+64]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+72]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+40]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+80]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+88]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+48]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+96]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+104]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+56]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+112]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+120]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+4], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax+64]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+128]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+136]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+72]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+144]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+152]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+80]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+160]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+168]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+88]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+176]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+184]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+8], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm1, [eax+96]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+192]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+200]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+104]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+208]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+216]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+112]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+224]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+232]" \
    "paddd     mm6, mm2" \
    "movq      mm1, [eax+120]" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+240]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+248]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+12], mm6" \
    "emms" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

extern void lz_dot128_q41_asm(const unsigned char *w,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q41_asm = \
    ".586" \
    "pxor      mm7, mm7" \
    "pcmpeqb   mm5, mm5" \
    "psrlw     mm5, 12" \
    "packuswb  mm5, mm5" \
    "pxor      mm6, mm6" \
    "movq      mm3, [eax]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+8]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+32]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+40]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+8]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+16]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+24]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+48]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+56]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm3, [eax+16]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+64]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+72]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+96]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+104]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+24]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+80]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+88]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+112]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+120]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+4], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm3, [eax+32]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+128]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+136]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+160]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+168]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+40]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+144]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+152]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+176]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+184]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+8], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm3, [eax+48]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+192]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+200]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+224]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+232]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+56]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+208]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+216]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+240]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+248]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+12], mm6" \
    "emms" \
    __parm [__eax] [__ecx] [__ebx] \
    __modify [8087]

extern void lz_dot128_q61_asm(const unsigned char *w4, const unsigned char *w2,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q61_asm = \
    ".586" \
    "pxor      mm7, mm7" \
    "pcmpeqb   mm5, mm5" \
    "psrlw     mm5, 12" \
    "packuswb  mm5, mm5" \
    "pcmpeqb   mm0, mm0" \
    "psrlw     mm0, 14" \
    "packuswb  mm0, mm0" \
    "pxor      mm6, mm6" \
    "movq      mm4, [edx]" \
    "movq      mm3, [eax]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+8]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+32]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+40]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+8]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 2" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+16]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+24]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 6" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+48]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+56]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm4, [edx+8]" \
    "movq      mm3, [eax+16]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+64]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+72]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+96]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+104]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+24]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 2" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+80]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+88]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 6" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+112]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+120]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+4], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm4, [edx+16]" \
    "movq      mm3, [eax+32]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+128]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+136]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+160]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+168]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+40]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 2" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+144]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+152]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 6" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+176]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+184]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+8], mm6" \
    "pxor      mm6, mm6" \
    "movq      mm4, [edx+24]" \
    "movq      mm3, [eax+48]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+192]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+200]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 4" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+224]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+232]" \
    "paddd     mm6, mm2" \
    "movq      mm3, [eax+56]" \
    "movq      mm1, mm3" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 2" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+208]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+216]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm3" \
    "psrlw     mm1, 4" \
    "pand      mm1, mm5" \
    "movq      mm2, mm4" \
    "psrlw     mm2, 6" \
    "pand      mm2, mm0" \
    "psllw     mm2, 4" \
    "por       mm1, mm2" \
    "movq      mm2, mm7" \
    "punpcklbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+240]" \
    "paddd     mm6, mm2" \
    "movq      mm2, mm7" \
    "punpckhbw mm2, mm1" \
    "pmaddwd   mm2, [ecx+248]" \
    "paddd     mm6, mm2" \
    "movq      mm1, mm6" \
    "psrlq     mm1, 32" \
    "paddd     mm6, mm1" \
    "movd      [ebx+12], mm6" \
    "emms" \
    __parm [__eax] [__edx] [__ecx] [__ebx] \
    __modify [8087]
extern void lz_dot128_q61_sse2_asm(const unsigned char *w4, const unsigned char *w2,
                const int16_t *x, int32_t *out4);
#pragma aux lz_dot128_q61_sse2_asm = \
    ".686" \
    "pcmpeqb   xmm7, xmm7" \
    "psrlw     xmm7, 12" \
    "packuswb  xmm7, xmm7" \
    "pcmpeqb   xmm6, xmm6" \
    "psrlw     xmm6, 14" \
    "packuswb  xmm6, xmm6" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax]" \
    "movq      xmm1, [edx]" \
    "movdqa    xmm2, xmm1" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 2" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+16]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm1" \
    "psrlw     xmm2, 4" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 6" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+32]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+48]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      [ebx], xmm4" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax+16]" \
    "movq      xmm1, [edx+8]" \
    "movdqa    xmm2, xmm1" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 2" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+64]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+80]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm1" \
    "psrlw     xmm2, 4" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 6" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+96]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+112]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      [ebx+4], xmm4" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax+32]" \
    "movq      xmm1, [edx+16]" \
    "movdqa    xmm2, xmm1" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 2" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+128]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+144]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm1" \
    "psrlw     xmm2, 4" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 6" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+160]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+176]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      [ebx+8], xmm4" \
    "pxor      xmm4, xmm4" \
    "movdqu    xmm0, [eax+48]" \
    "movq      xmm1, [edx+24]" \
    "movdqa    xmm2, xmm1" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 2" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+192]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+208]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm1" \
    "psrlw     xmm2, 4" \
    "pand      xmm2, xmm6" \
    "movdqa    xmm3, xmm1" \
    "psrlw     xmm3, 6" \
    "pand      xmm3, xmm6" \
    "punpcklqdq xmm2, xmm3" \
    "psllw     xmm2, 4" \
    "movdqa    xmm3, xmm0" \
    "psrlw     xmm3, 4" \
    "pand      xmm3, xmm7" \
    "por       xmm3, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpcklbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+224]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "pxor      xmm2, xmm2" \
    "punpckhbw xmm2, xmm3" \
    "movdqu    xmm5, [ecx+240]" \
    "pmaddwd   xmm2, xmm5" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 8" \
    "paddd     xmm4, xmm2" \
    "movdqa    xmm2, xmm4" \
    "psrldq    xmm2, 4" \
    "paddd     xmm4, xmm2" \
    "movd      [ebx+12], xmm4" \
    __parm [__eax] [__edx] [__ecx] [__ebx] \
    __modify [8087]
#elif defined(LZ_DOT_MMX_EXTERN)
/* dot32_q61_mmx (single token) and dot32_q61_mmx_2 (two tokens, ONE
   two-plane merge - the kernel that matters most on the recipe of
   record, see the Watcom twin's comment) live in
   src/ops_kernel_dot_mmx.h, included only by src/ops_mmx.c. Declared in
   src/ops_mmx.h, already visible here. */
#endif
