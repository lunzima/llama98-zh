/* AVX2 row kernels for the matmul/dot-product family. gcc-only: this
   file is never compiled by Watcom. Task 7's Makefile rule will only
   ever pass -mavx2 to this translation unit, never -mfma - see the
   design doc's "core tension and solution" section on why FMA is
   excluded even though -march=x86-64-v3 nominally bundles it in. This
   tier is never a member of kernel_detect()'s AUTO cascade; reachable
   only via an explicit --kernel avx2.

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
#include "ops_avx2.h"

/* Reduce one __m256i of 8 x int32 partial sums to a single scalar.
   Shared tail for every part32_*_avx2 below - none of them do a
   horizontal reduction any other way, so factoring it once here means
   the reduction sequence can only drift in one place, not four. */
static int32_t fold8_to_scalar_avx2(__m256i s) {
    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    return _mm_cvtsi128_si32(sum);
}

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
   activation load. Concretely: with wb0 holding weight bytes 0-31
   across its two 128-bit lanes, _mm256_unpacklo_epi8(z, wb0) would
   zero-extend bytes {0-7, 16-23} into its 16 int16 lanes, in that
   order - lane 0's low half interleaved with lane 1's low half, NOT
   bytes 0-15 the way a caller reasoning "unpacklo takes the low half"
   would expect. Paired against a plain contiguous _mm256_loadu_si256
   of activations 0-15, that multiplies weight byte 20 against
   activation element 12 instead of element 20 - finite, plausible,
   wrong. Recovering the contiguous order from the naive unpack would
   need an extra cross-lane step (_mm256_permute4x64_epi64 on wb0
   before unpacking, or permutevar8x32 after) that _mm256_cvtepu8_epi16
   needs not, because it is never lane-restricted to begin with.

   Unaligned loads throughout (_mm_loadu_si128/_mm256_loadu_si256),
   unlike part32_x16's aligned _mm_load_si128 for both operands: this
   file has no established guarantee that either the weight row
   (c->w4, from the model blob) or a 32-element activation group
   offset lands on a 32-byte boundary - src/ops_matmul.c's g_xw
   alignment comment establishes 16-byte, not 32-byte, as what the
   SSE2 tier actually relies on. An unaligned load costs nothing extra
   when the pointer happens to be more strictly aligned, so loadu is
   the conservative choice here, not an unverified aligned claim. */
static int32_t part32_x32_avx2(const int8_t *w, const int16_t *x) {
    __m128i wb0 = _mm_loadu_si128((const __m128i *)w);
    __m128i wb1 = _mm_loadu_si128((const __m128i *)(w + 16));
    __m256i w0 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb0), 8);
    __m256i w1 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(wb1), 8);
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(w0, x0),
                                  _mm256_madd_epi16(w1, x1));
    return fold8_to_scalar_avx2(s);
}

/* ---- Q8 row kernel -------------------------------------------------------
   No Watcom twin - AVX2 has none, unlike every other tier in this
   codebase. No pairing either: g_pair (src/ops_sse2.c) is the SSE2
   tier's two-token, shared-weight-load trick, not implemented here -
   see this file's header comment. Each token is processed
   independently. */
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

/* Q4_1 32-element partial sum. Nibble layout (model.h, LZ_FMT_Q4_1): a
   32-element sub-block is 16 bytes; byte j's LOW nibble is element j,
   its HIGH nibble is element j+16 - same convention part32_q41
   (src/ops_kernel_dot_sse2.h) decodes, reproduced here bit for bit.
   There is no zero-point work in this function: LZTensor.zero (the
   per-group min) is an affine term applied later by the epilogue
   (ops_matmul.c's epi_q41 family via the hoisted xg[]/zq term), never
   inside the row kernel - this function's job is only the same raw
   unsigned-nibble dot product part32_q41 computes, nothing more.

   `lo`/`hi` each hold all 16 elements of their half in one 16-byte
   register, in order - unlike part32_x32_avx2's two full 32-byte
   weight loads, a nibble sub-block is only 16 bytes total, so one
   _mm256_cvtepu8_epi16 per half is enough; no second widen call and
   no unpacklo/hi_epi8 trap to avoid, since the mask/shift already
   produced one byte per element in the right order before the widen. */
static int32_t part32_q41_avx2(const unsigned char *w, const int16_t *x) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m128i b  = _mm_loadu_si128((const __m128i *)w);
    __m128i lo = _mm_and_si128(b, m0f);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
    __m256i lo16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8);
    __m256i hi16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8);
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(lo16, x0),
                                  _mm256_madd_epi16(hi16, x1));
    return fold8_to_scalar_avx2(s);
}

/* ---- Q4_1 row kernel ------------------------------------------------
   No Watcom twin, no pairing - see this file's header comment; each
   token is processed independently, same as row_q8_avx2_intrin. */
void row_q41_avx2_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    int tk, s;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = part32_q41_avx2(wn + (size_t)s * 16,
                                          xwt + (size_t)s * 32);
        }
    }
}

/* Q6_1 32-element partial sum: two-plane format, model.h's LZ_FMT_Q6_1.
   `w4` is the SAME 4-bit-nibble plane as Q4_1 (elements 0-15 in low
   nibbles, 16-31 in high nibbles); `w2` is a second, independent 2-bit
   plane, 8 bytes per 32-element sub-block, byte j's bits 0-1/2-3/4-5/6-7
   holding elements j/j+8/j+16/j+24 respectively. A 6-bit code is
   q = lo_nibble | (2bit_field << 4); this function reads BOTH c->w4 AND
   c->w2, unlike every other format in this file, which reads only
   c->w4 - see row_q61_sse2_intrin (src/ops_sse2.c) for the same
   two-pointer read on the reference tier.

   Reassembly (m0f/m03 masks, the two shifts-by-2/4/6, `_mm_or_si128`
   with the nibble) is copied instruction-for-instruction from
   part32_q61's 128-bit domain, unchanged - that step produces one byte
   per element, in order, before anything is widened, so it carries no
   AVX2-specific risk. Only the final byte-to-int16 widen swaps
   part32_q61's unpacklo/hi_epi8 pair for a single in-order
   _mm256_cvtepu8_epi16 per half, same as part32_q41_avx2 above. No
   zero-point work here either, for the same reason as Q4_1: `zero` is
   an epilogue-side affine term, not part of this raw dot product. */
static int32_t part32_q61_avx2(const unsigned char *w4,
                                const unsigned char *w2, const int16_t *x) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i m03 = _mm_set1_epi8(0x03);
    __m128i b  = _mm_loadu_si128((const __m128i *)w4);
    __m128i t  = _mm_loadl_epi64((const __m128i *)w2);
    __m128i g0 = _mm_and_si128(t, m03);
    __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03);
    __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03);
    __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03);
    __m128i lo = _mm_or_si128(_mm_and_si128(b, m0f),
                              _mm_slli_epi16(_mm_unpacklo_epi64(g0, g1), 4));
    __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b, 4), m0f),
                              _mm_slli_epi16(_mm_unpacklo_epi64(g2, g3), 4));
    __m256i lo16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(lo), 8);
    __m256i hi16 = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hi), 8);
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(lo16, x0),
                                  _mm256_madd_epi16(hi16, x1));
    return fold8_to_scalar_avx2(s);
}

/* ---- Q6_1 row kernel ------------------------------------------------
   No Watcom twin, no pairing - see this file's header comment; each
   token is processed independently, same as row_q8_avx2_intrin. Reads
   both c->w4 (4-bit plane) and c->w2 (2-bit plane), each advanced at
   its own stride (16 and 8 bytes per 32-element sub-block). */
void row_q61_avx2_intrin(const lz_row_ctx *c) {
    const unsigned char *wn = (const unsigned char *)c->w4;
    const unsigned char *w2 = (const unsigned char *)c->w2;
    int tk, s;
    (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (s = 0; s < c->nb; s++)
                acc[s] = part32_q61_avx2(wn + (size_t)s * 16,
                                          w2 + (size_t)s * 8,
                                          xwt + (size_t)s * 32);
        }
    }
}

/* Q16_0 32-element partial sum: weight is already int16 (model.h,
   LZ_FMT_Q16_0), so there is no unpack step at all - the simplest of
   the four formats in this file, same conclusion part32_q16's own
   comment (src/ops_kernel_dot_sse2.h) draws for the SSE2 tier. Two
   plain 256-bit loads per operand, two pmaddwd, one add, one fold. No
   x<<8 scale here either: Q16_0's row-end epilogue uses post=1.0, not
   the 1/256 the other three formats share (see part32_q16's comment). */
static int32_t part32_q16_avx2(const int16_t *w, const int16_t *x) {
    __m256i w0 = _mm256_loadu_si256((const __m256i *)w);
    __m256i w1 = _mm256_loadu_si256((const __m256i *)(w + 16));
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 16));
    __m256i s = _mm256_add_epi32(_mm256_madd_epi16(w0, x0),
                                  _mm256_madd_epi16(w1, x1));
    return fold8_to_scalar_avx2(s);
}

/* ---- Q16_0 row kernel -------------------------------------------------
   No Watcom twin, no pairing - see this file's header comment; each
   token is processed independently, same as row_q8_avx2_intrin. */
void row_q16_avx2_intrin(const lz_row_ctx *c) {
    const int16_t *wr = (const int16_t *)c->w4;
    int tk, g;
    (void)c->w2; (void)c->pf_end; (void)c->pf_end2;
    {
        const int16_t *xwt = c->xw;
        int32_t *acc = c->acc32;
        for (tk = 0; tk < c->nt; tk++, xwt += c->in_dim, acc += c->nb) {
            for (g = 0; g < c->nb; g++)
                acc[g] = part32_q16_avx2(wr + (size_t)g * 32,
                                          xwt + (size_t)g * 32);
        }
    }
}

#endif /* LZ_AVX2_TU && !__WATCOMC__ */
