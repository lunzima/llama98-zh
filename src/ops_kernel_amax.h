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

/* The amax kernel bodies - lz_amax32_mmx_asm / lz_amax32_sse2_asm's
   #pragma aux pairs and their _w wrappers on Watcom, lz_amax32_mmx /
   lz_amax32_sse2 intrinsics on gcc - live in src/ops_mmx.c /
   src/ops_sse2.c. The LZ_HAVE_AMAX_MMX / LZ_HAVE_AMAX_SSE2 flags that
   say "this kernel is in the build" are defined in src/ops_mmx.h /
   src/ops_sse2.h under LZ_MMX_TU / LZ_SSE2_TU, both included near the
   top of ops.c before this file, so they are already visible here.
   The _w wrappers exist because a #pragma aux body cannot have its
   address taken, so the dispatch table below needs a real function to
   point at. */

/* The dispatch table. LZ_ROW_N slots in the LZ_ROW_* order; NULL is a
   claim with a reason attached, not an omission. The trailing avx2
   slot holds lz_amax32_avx2 (src/ops_avx2.c) whenever this build's
   AVX2 tier exists: LZ_DEFINE_PICK (ops_kernel_shared.h) reads
   LZ_ROW_AVX2_I first, ahead of the MMX/SSE2 chain, when g_kernel ==
   LZ_KERNEL_AVX2.

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
    lz_amax32_sse2_w,
#else
    NULL,
#endif
#if defined(LZ_HAVE_AMAX_AVX2)
    lz_amax32_avx2
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

/* Whether the picked kernel is the AVX2 body specifically - pointer
   identity, not "produced the right number". A bit-identity comparison
   against a scalar/SSE2 reference cannot tell "ran AVX2" from "silently
   fell back to SSE2, which computes the same value by construction" -
   this predicate exists so a caller can ask the question a value
   comparison structurally cannot answer (see q8_amax_picked_avx2,
   ops_quant.c, the one non-static bridge that lets a test outside this
   TU call it, since LZ_AMAX_TAB/lz_amax_pick stay static here). */
static int lz_amax_is_avx2(lz_amaxfn f) {
    return f != 0 && f == LZ_AMAX_TAB[LZ_ROW_AVX2_I];
}

/* How many elements the picked kernel consumes per step: two lanes for
   MMX, four for SSE2, eight for AVX2. The operator needs it to know
   where its scalar tail begins, and asking the table beats a second
   copy of the tier test at the call site.

   The AVX2 case has to be checked explicitly, not folded into the
   "not MMX" default the other two shared: that default was only ever
   true because SSE2 and MMX+SSE2/MMX-asm all happened to consume
   exactly four elements per step, an invariant AVX2 breaks (it
   consumes eight, with no tail of its own - see lz_amax32_avx2's
   header comment). Left unfixed, q8_amax's caller-side tail
   computation (`gs & ~(lanes - 1)`) would believe an eight-wide AVX2
   call had covered a range it stopped four elements short of - for
   gs in 4 mod 8 including gs==4, believe it covered the range
   entirely, silently returning 0 for real data. gs==4 is not
   theoretical: src/forward.c calls q8_amax with n=c->conv_kernel==4
   for every conv1d row, so an inaccurate lanes count here would
   zero every depthwise conv1d weight under --kernel avx2. */
static int lz_amax_lanes(lz_amaxfn f) {
    if (!f) return 0;
    if (f == LZ_AMAX_TAB[LZ_ROW_AVX2_I]) return 8;
    return lz_amax_is_mmx(f) ? 2 : 4;
}
