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

/* lz_amax32_mmx_asm's #pragma aux body and its lz_amax32_mmx_w wrapper
   (task #30, 4/N: the mechanical low-risk sweep) moved to
   src/ops_mmx.c - same -Nr/-Ns ceiling reason as every kernel this
   task has already moved. Zero call-shape change either way: the
   wrapper already sat between the pragma and the dispatch table below
   (a #pragma aux body cannot have its address taken, so a table entry
   needs a real function to point at), and that indirect call happens
   once per operator invocation regardless of which TU the wrapper's
   body lives in - the reason this whole family was classified low-risk
   from the start. LZ_HAVE_AMAX_MMX stays defined here: it is Watcom's
   own "MMX exists in this build" flag, unconditional the same way
   -D__MMX__=1 always is.

   lz_amax32_sse2_asm / lz_amax32_sse2_w: same story, moved to
   src/ops_sse2.c. LZ_HAVE_AMAX_SSE2 stays defined here too. */
#if defined(__WATCOMC__)
#define LZ_HAVE_AMAX_MMX  1
#define LZ_HAVE_AMAX_SSE2 1
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
