/* The element-wise float helpers: RMSNorm's output loop, and softmax's
   maximum scan and normalize loop. Included once by ops.c, like the
   other ops_kernel_*.h files, for the same codegen reason.

   All three are here because they share one property - each output
   element depends only on its own input, or the reduction is exact - so
   four lanes give the same bits as one. The reductions that are NOT
   exact (RMSNorm's sum of squares, softmax's sum) stay scalar in ops.c
   and are not in this file.

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

static float lz_vmax_sse(const float *x, int n4, float seed) {
    __m128 acc = _mm_set1_ps(seed);
    union { __m128 m; float f[4]; } o;
    int b;
    for (b = 0; b < n4; b++)
        acc = _mm_max_ps(acc, _mm_loadu_ps(x + b * 4));
    o.m = acc;
    {
        float a = o.f[0] > o.f[1] ? o.f[0] : o.f[1];
        float c = o.f[2] > o.f[3] ? o.f[2] : o.f[3];
        return a > c ? a : c;
    }
}

static void lz_vscale_sse(float *x, int n4, float k) {
    __m128 vk = _mm_set1_ps(k);
    int b;
    for (b = 0; b < n4; b++)
        _mm_storeu_ps(x + b * 4, _mm_mul_ps(_mm_loadu_ps(x + b * 4), vk));
}
#define LZ_HAVE_NORM_SSE 1
#endif

#endif  /* __WATCOMC__ */
