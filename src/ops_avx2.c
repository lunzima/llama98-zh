/* AVX2 row kernels for the matmul/dot-product family. gcc-only: this
   file is never compiled by Watcom, and this translation unit is never
   passed -mfma - only -mavx2 (see Makefile). This tier is never a
   member of kernel_detect()'s AUTO cascade; reachable only via an
   explicit --kernel avx2.

   Every accumulation below is INTEGER (int8/int16 products summed in
   int32), never floating point, so there is no FMA-vs-two-step
   rounding question for this file the way rule two worries about for
   the float epilogue elsewhere: within the documented per-group bound
   (|w<<8| <= 32512, |x| <= 127, 32-element group caps ~1.3e8, far
   under INT32_MAX), int32 addition is exact and fully associative.
   Bit-identity with the SSE2 tier (tests/test_ops.c, training-side
   repo) holds because both compute the same exact integer sum - not
   because this file replicates SSE2's exact intermediate register
   grouping, which it deliberately does not.

   This first pass does not implement the SSE2 tier's g_pair
   (two-token, shared-weight-load) optimization from
   src/ops_kernel_dot_sse2.h - each token is processed independently.
   Documented, deferred follow-up; not a silent omission. */
#if defined(LZ_AVX2_TU) && !defined(__WATCOMC__)

#include <immintrin.h>
#include <stdlib.h>
#include "ops_avx2.h"

/* 32-element partial sum, one int32 result. w<<8 convention matches
   part32_x16 (src/ops_kernel_dot_sse2.h) exactly - the shared row-end
   epilogue elsewhere divides by the same 256 regardless of which
   kernel tier produced acc32[], so this file must reproduce that
   convention, not the register sequence that produces it.

   _mm256_cvtepu8_epi16 zero-extends a 128-bit (16-byte) load to a
   256-bit (16 x int16) result IN ELEMENT ORDER - unlike
   _mm256_unpacklo/hi_epi8, which interleave within each 128-bit lane
   independently and would silently pair the wrong weight byte with
   the wrong activation if used here alongside a plain contiguous
   activation load. */
static int32_t part32_x32_avx2(const int8_t *w, const int16_t *x) {
    __m128i wb0 = _mm_loadu_si128((const __m128i *)w);
    __m128i wb1 = _mm_loadu_si128((const __m128i *)(w + 16));
    __m256i w0 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb0), 8);
    __m256i w1 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb1), 8);
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(w0, x0),
                                  _mm256_madd_epi16(w1, x1));
    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    return _mm_cvtsi128_si32(sum);
}

void row_q8_avx2_intrin(const lz_row_ctx *c) {
    const int8_t *wr = (const int8_t *)c->w4;
    int tk, g;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (g = 0; g < c->nb; g++)
                acc[g] = part32_x32_avx2(wr + (size_t)g * 32,
                                          xwt + (size_t)g * 32);
        }
    }
}

/* Task 6 replaces these with real implementations. Trivial abort()
   bodies only so the dispatch-table wiring (which references all four
   row_*_avx2_intrin names unconditionally once LZ_ROW_AVX2_EXTERN is
   defined) links; --kernel avx2 must not be selected for Q4_1/Q6_1/
   Q16_0 workloads until Task 6 lands. This makes any accidental early
   exercise of these three formats crash loudly rather than silently
   return wrong numbers. */
void row_q41_avx2_intrin(const lz_row_ctx *c) {
    (void)c;
    abort();
}
void row_q61_avx2_intrin(const lz_row_ctx *c) {
    (void)c;
    abort();
}
void row_q16_avx2_intrin(const lz_row_ctx *c) {
    (void)c;
    abort();
}

#endif /* LZ_AVX2_TU && !__WATCOMC__ */
