/* Q8 group-scale amax kernels: the scalar scan lives in ops.c, the SIMD
   tiers live here. NOT a standalone translation unit - ops.c #includes
   this exactly once, and it must stay that way, for the same reason
   ops_kernel_q8round.h gives: splitting a file is allowed to be a
   readability change and not a codegen change.

   WHAT IS BEING VECTORIZED. q8_group_scale takes the maximum of
   |x[k]| over a group, computed on the BIT PATTERNS - mask off the sign
   and compare as integers, which is exact for every finite float and
   orders NaN above everything. Two properties follow, and both matter:

   - max is associative and exact, so a lane-parallel reduction gives
     the SAME answer as the scalar loop, bit for bit. That is unusual
     here: a float SUM cannot be vectorized without changing the
     association order and breaking cross-compiler bit-identity, which
     is why rmsnorm's and softmax's reductions do NOT appear in this
     file. Getting the same number by a different route is the whole
     licence for these kernels to exist.
   - after masking, every value is in [0, 0x7FFFFFFF], i.e. non-negative
     as a SIGNED int32. Neither MMX nor SSE2 has an unsigned 32-bit max,
     but both have a signed 32-bit COMPARE, and on non-negative inputs
     signed and unsigned order agree. So the compare-and-blend below is
     correct, not approximate.

   THE SSE1 CELL IS STRUCTURALLY EMPTY, and the reason is required rather
   than assumed. SSE1's additions to the MMX integer set are pshufw,
   pinsrw, pextrw, pmovmskb, pmulhuw, pavgb, pavgw, pmaxsw, pminsw,
   pmaxub, pminub, psadbw, maskmovq and movntq. The only maxima there are
   16-bit signed (pmaxsw) and 8-bit unsigned (pmaxub); a 32-bit magnitude
   cannot be split across two 16-bit lanes without carrying the
   comparison between them, which none of these do. SSE1 therefore offers
   nothing over MMX for this operator and gets no tier. */

#if defined(__WATCOMC__)

extern unsigned lz_amax32_mmx_asm(const float *x, int n);
/* Two lanes of uint32 in one MMX register. The mask is built in-register
   (pcmpeqd then psrld 1) rather than loaded, so the routine touches no
   memory but its input. Leaves MMX state dirty - the caller emits one
   emms after the group loop, never the kernel (rule 6 item 6). */
#pragma aux lz_amax32_mmx_asm =         \
    "pxor     mm0, mm0"                 \
    "pcmpeqd  mm3, mm3"                 \
    "psrld    mm3, 1"                   \
    "shr      edx, 1"                   \
    "jz       L2"                       \
    "L1:"                               \
    "movq     mm1, [eax]"               \
    "pand     mm1, mm3"                 \
    "movq     mm2, mm1"                 \
    "pcmpgtd  mm2, mm0"                 \
    "pand     mm1, mm2"                 \
    "pandn    mm2, mm0"                 \
    "por      mm1, mm2"                 \
    "movq     mm0, mm1"                 \
    "add      eax, 8"                   \
    "dec      edx"                      \
    "jnz      L1"                       \
    "L2:"                               \
    "movq     mm1, mm0"                 \
    "psrlq    mm1, 32"                  \
    "movq     mm2, mm0"                 \
    "pcmpgtd  mm2, mm1"                 \
    "pand     mm0, mm2"                 \
    "pandn    mm2, mm1"                 \
    "por      mm0, mm2"                 \
    "movd     eax, mm0"                 \
    parm [eax] [edx]                    \
    value [eax]                         \
    modify [eax edx 8087];
#define LZ_HAVE_AMAX_MMX 1

extern unsigned lz_amax32_sse2_asm(const float *x, int n);
/* Four lanes. The horizontal fold is two shuffle-and-compare rounds -
   pshufd is SSE2, so no MMX register is touched and no emms is owed. */
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
#define LZ_HAVE_AMAX_SSE2 1

/* Thin wrappers so the dispatch table can hold these.
 *
 * A #pragma aux body is emitted at each CALL SITE; there is no function
 * to point at, and taking its address is an undefined reference at link
 * - which is exactly what happened the first time these went into a
 * table. Every row_*_asm in ops.c has the same shape for the same
 * reason. The wrapper is the only caller, so the asm is still inlined
 * once and the indirect call is one per operator invocation, not one per
 * element. */
static unsigned lz_amax32_mmx_w(const float *x, int n) {
    return lz_amax32_mmx_asm(x, n);
}
static unsigned lz_amax32_sse2_w(const float *x, int n) {
    return lz_amax32_sse2_asm(x, n);
}

#else   /* gcc: intrinsics */

/* lz_amax32_mmx's body and the LZ_HAVE_AMAX_MMX definition for this
   branch now live in src/ops_mmx.c / src/ops_mmx.h: it writes %mm
   registers, so it belongs in the one translation unit built with
   -mmmx. ops_mmx.h is included near the top of ops.c, so the
   declaration and the macro are already visible here.

   lz_amax32_sse2's body and the LZ_HAVE_AMAX_SSE2 definition for this
   branch now live in src/ops_sse2.c / src/ops_sse2.h, same reasoning
   one ISA tier up: it writes %xmm registers, so it belongs in the
   translation unit built with -msse2. ops_sse2.h is included near the
   top of ops.c. */

#endif  /* __WATCOMC__ */

/* The dispatch table. Six slots in the LZ_ROW_* order; NULL is a claim
   with a reason attached, not an omission.

   SSE1 IS EMPTY BY INSTRUCTION SET. Its additions to the MMX integer
   set are pshufw, pinsrw, pextrw, pmovmskb, pmulhuw, pavgb, pavgw,
   pmaxsw, pminsw, pmaxub, pminub, psadbw, maskmovq and movntq. The only
   maxima there are 16-bit signed and 8-bit unsigned; a 32-bit magnitude
   cannot be split across two 16-bit lanes without carrying the
   comparison between them, which none of those do. So a Pentium III
   runs the MMX kernel, and asking for SSE lands there through the pick
   rather than through an #if.

   Every slot is NULL on a target with no SIMD - the ARM cross-build -
   and the pick returns NULL, so the operator takes its scalar path with
   no conditional compilation at the call site. */
typedef unsigned (*lz_amaxfn)(const float *x, int n);
static const lz_amaxfn LZ_AMAX_TAB[LZ_ROW_N] = {
#if defined(LZ_HAVE_AMAX_MMX) && !defined(__WATCOMC__)
    lz_amax32_mmx,
#else
    NULL,
#endif
    NULL,                       /* sse-intrin: see the note above */
#if defined(LZ_HAVE_AMAX_SSE2) && !defined(__WATCOMC__)
    lz_amax32_sse2,
#else
    NULL,
#endif
#if defined(LZ_HAVE_AMAX_MMX) && defined(__WATCOMC__)
    lz_amax32_mmx_w,
#else
    NULL,
#endif
    NULL,                       /* sse-asm: same instruction-set reason */
#if defined(LZ_HAVE_AMAX_SSE2) && defined(__WATCOMC__)
    lz_amax32_sse2_w
#else
    NULL
#endif
};
LZ_DEFINE_PICK(lz_amax_pick, lz_amaxfn)

/* Whether the picked kernel writes MMX registers, which decides who owes
   an emms. Derived from the slot rather than from the tier name: those
   two agreed until the day a table had a hole in it. */
static int lz_amax_is_mmx(lz_amaxfn f) {
    return f != 0 && (f == LZ_AMAX_TAB[LZ_ROW_MMX_I] ||
                      f == LZ_AMAX_TAB[LZ_ROW_MMX_A]);
}

/* How many elements the picked kernel consumes per step: two lanes for
   MMX, four for SSE2. The operator needs it to know where its scalar
   tail begins, and asking the table beats a second copy of the tier
   test at the call site. */
static int lz_amax_lanes(lz_amaxfn f) {
    if (!f) return 0;
    return lz_amax_is_mmx(f) ? 2 : 4;
}
