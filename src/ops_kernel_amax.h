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

#else   /* gcc: intrinsics */

#if defined(__MMX__)
#include <mmintrin.h>
#include "mmx_compat.h"
/* Leaves MMX state dirty; the caller emits emms once after the loop. */
static unsigned lz_amax32_mmx(const float *x, int n) {
    __m64 acc = _mm_setzero_si64();
    __m64 msk = _mm_set1_pi32(0x7FFFFFFF);
    int k;
    for (k = 0; k + 1 < n; k += 2) {
        __m64 v = _mm_and_si64(*(const __m64 *)(const void *)(x + k), msk);
        __m64 gt = _mm_cmpgt_pi32(v, acc);
        acc = _mm_or_si64(_mm_and_si64(v, gt), _mm_andnot_si64(gt, acc));
    }
    {
        union { __m64 m; unsigned u[2]; } o;
        o.m = acc;
        return o.u[0] > o.u[1] ? o.u[0] : o.u[1];
    }
}
#define LZ_HAVE_AMAX_MMX 1
#endif

#if defined(__SSE2__)
#include <emmintrin.h>
static unsigned lz_amax32_sse2(const float *x, int n) {
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
#define LZ_HAVE_AMAX_SSE2 1
#endif

#endif  /* __WATCOMC__ */
