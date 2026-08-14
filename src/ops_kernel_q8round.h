/* Q8 activation-rounding kernels: scalar magic-add lives in ops.c, the
   SIMD tiers live here. NOT a standalone translation unit - ops.c
   #includes this, and it must stay that way.

   Why a header and not a .c: code LAYOUT alone moves decode 3-7% on the
   dev box (measured: -falign-functions=64 turned the same change from
   0.976x to 1.024x). Separate translation units would change
   inlining decisions and layout at once, turning a pure move into an
   unattributable performance change, and every `static` here would stop
   being static. Splitting the file is a readability change; it is not
   allowed to be a codegen change.

   The .h extension is a lie of convenience: there are no include guards
   because this is included exactly once, and including it twice would
   redefine every static. */
#if defined(__WATCOMC__)
#include "mmx_compat.h"          /* _mm_empty(): the SSE1 lane wrote MMX registers */
extern void lz_q8round32_sse_asm(const float *x, int8_t *o, const float *inv);
#pragma aux lz_q8round32_sse_asm = \
    ".686" \
    "movss    xmm7, [ecx]" \
    "shufps   xmm7, xmm7, 0" \
    "pcmpeqw  mm5, mm5" \
    "psrlw    mm5, 9" \
    "pxor     mm6, mm6" \
    "psubw    mm6, mm5" \
    "movups   xmm0, [eax]" \
    "mulps    xmm0, xmm7" \
    "movups   xmm1, [eax+16]" \
    "mulps    xmm1, xmm7" \
    "cvtps2pi mm0, xmm0" \
    "movhlps  xmm2, xmm0" \
    "cvtps2pi mm1, xmm2" \
    "cvtps2pi mm2, xmm1" \
    "movhlps  xmm3, xmm1" \
    "cvtps2pi mm3, xmm3" \
    "packssdw mm0, mm1" \
    "packssdw mm2, mm3" \
    "pminsw   mm0, mm5" \
    "pmaxsw   mm0, mm6" \
    "pminsw   mm2, mm5" \
    "pmaxsw   mm2, mm6" \
    "packsswb mm0, mm2" \
    "movq     [edx], mm0" \
    "movups   xmm0, [eax+32]" \
    "mulps    xmm0, xmm7" \
    "movups   xmm1, [eax+48]" \
    "mulps    xmm1, xmm7" \
    "cvtps2pi mm0, xmm0" \
    "movhlps  xmm2, xmm0" \
    "cvtps2pi mm1, xmm2" \
    "cvtps2pi mm2, xmm1" \
    "movhlps  xmm3, xmm1" \
    "cvtps2pi mm3, xmm3" \
    "packssdw mm0, mm1" \
    "packssdw mm2, mm3" \
    "pminsw   mm0, mm5" \
    "pmaxsw   mm0, mm6" \
    "pminsw   mm2, mm5" \
    "pmaxsw   mm2, mm6" \
    "packsswb mm0, mm2" \
    "movq     [edx+8], mm0" \
    "movups   xmm0, [eax+64]" \
    "mulps    xmm0, xmm7" \
    "movups   xmm1, [eax+80]" \
    "mulps    xmm1, xmm7" \
    "cvtps2pi mm0, xmm0" \
    "movhlps  xmm2, xmm0" \
    "cvtps2pi mm1, xmm2" \
    "cvtps2pi mm2, xmm1" \
    "movhlps  xmm3, xmm1" \
    "cvtps2pi mm3, xmm3" \
    "packssdw mm0, mm1" \
    "packssdw mm2, mm3" \
    "pminsw   mm0, mm5" \
    "pmaxsw   mm0, mm6" \
    "pminsw   mm2, mm5" \
    "pmaxsw   mm2, mm6" \
    "packsswb mm0, mm2" \
    "movq     [edx+16], mm0" \
    "movups   xmm0, [eax+96]" \
    "mulps    xmm0, xmm7" \
    "movups   xmm1, [eax+112]" \
    "mulps    xmm1, xmm7" \
    "cvtps2pi mm0, xmm0" \
    "movhlps  xmm2, xmm0" \
    "cvtps2pi mm1, xmm2" \
    "cvtps2pi mm2, xmm1" \
    "movhlps  xmm3, xmm1" \
    "cvtps2pi mm3, xmm3" \
    "packssdw mm0, mm1" \
    "packssdw mm2, mm3" \
    "pminsw   mm0, mm5" \
    "pmaxsw   mm0, mm6" \
    "pminsw   mm2, mm5" \
    "pmaxsw   mm2, mm6" \
    "packsswb mm0, mm2" \
    "movq     [edx+24], mm0" \
    __parm [__eax] [__edx] [__ecx] \
    __modify [8087]

#define LZ_HAVE_Q8R_SSE 1
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

#define LZ_HAVE_Q8R_SIMD 1
#define lz_q8round32_simd lz_q8round32_sse2_asm
#elif defined(__SSE__) || defined(__SSE2__)
#include <xmmintrin.h>
#include "mmx_compat.h"
#define LZ_HAVE_Q8R_SSE 1
/* SSE1 tier (PIII / K7 / Athlon XP): no cvtps2dq, only cvtps2pi -
   converting 2 floats at a time into MMX registers. Multiply in xmm,
   convert and pack in mm, using movhlps to slide the high 2 floats
   down. Rounding is likewise MXCSR round-half-even, consistent across
   cvtps2dq, cvtps2pi and the magic add.

   **Writes MMX registers**, so the caller must emit one emms after the
   group loop (rule 6.6: kernels don't emit, callers do once). */
static void lz_q8round32_sse(const float *x, int8_t *o, const float *pinv) {
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
#endif
#if defined(__SSE2__) && !defined(__WATCOMC__)
#include <emmintrin.h>
#define LZ_HAVE_Q8R_SIMD 1
static void lz_q8round32_simd(const float *x, int8_t *o, const float *pinv) {
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
#endif

#if defined(__WATCOMC__)
#define lz_q8round32_sse lz_q8round32_sse_asm
#endif
