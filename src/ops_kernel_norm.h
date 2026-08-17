/* RMSNorm's output loop, vectorized. Included once by ops.c, like the
   other ops_kernel_*.h files, for the same codegen reason.

   WHICH HALF THIS IS, AND WHY ONLY THIS HALF. lz_rmsnorm is a float sum
   followed by an element-wise scale:

       ss = SUM x[i]*x[i]                      <- reduction, NOT here
       o[i] = x[i] * inv * (1.0f + w[i])       <- element-wise, here

   Vectorizing the sum would change the association order, and the tiers
   owe each other bit-identity, so it stays scalar. The output loop owes
   nothing: each lane performs the same two multiplies and one add on the
   same values in the same order as the scalar loop, so four lanes at
   once is bit-identical by construction rather than by tolerance. The
   tail is scalar and takes the same expression, so lengths that are not
   a multiple of four are covered without a second numeric path.

   THE MMX CELL IS STRUCTURALLY ABSENT, not unwritten: MMX is an
   integer-only instruction set with no floating-point operations at all.
   That is the same reason lz_q8round32 has no MMX tier.

   SSE1 AND SSE2 ARE THE SAME CODE HERE. mulps and addps are SSE1; SSE2
   adds double and integer width, neither of which this loop uses. So one
   implementation fills both cells, and saying so is the point - an
   unexplained empty cell and a cell that cannot differ look identical
   from outside. */

#if defined(__WATCOMC__)

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
#define LZ_HAVE_NORM_SSE 1

#else   /* gcc: intrinsics */

#if defined(__SSE__) || defined(__SSE2__)
#include <xmmintrin.h>
static void lz_rmsnorm_out_sse(float *o, const float *x, const float *w,
                               int n4, float inv) {
    __m128 vi = _mm_set1_ps(inv);
    __m128 one = _mm_set1_ps(1.0f);
    int b;
    for (b = 0; b < n4; b++) {
        __m128 vx = _mm_loadu_ps(x + b * 4);
        __m128 vw = _mm_loadu_ps(w + b * 4);
        _mm_storeu_ps(o + b * 4,
                      _mm_mul_ps(_mm_mul_ps(vx, vi), _mm_add_ps(one, vw)));
    }
}
#define LZ_HAVE_NORM_SSE 1
#endif

#endif  /* __WATCOMC__ */
