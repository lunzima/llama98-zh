/* The SSE2 32-element partial-sum kernels, one per quantized weight
   format, plus the 4-way fold that combines them. Included only by
   src/ops_sse2.c - every function here writes %xmm registers. Gcc twin
   of the Watcom #pragma aux SSE2 bodies in that same file (compare
   part32_x16 against lz_dot32_x16_sse2_asm, etc.) - same algorithm,
   different toolchain.

   No include guards: included exactly once, same contract as
   src/ops_kernel_dot_mmx.h. */
#include <emmintrin.h>

static __m128i fold4_sse2(__m128i s0, __m128i s1, __m128i s2, __m128i s3) {
    __m128i t0 = _mm_unpacklo_epi32(s0, s1);
    __m128i t1 = _mm_unpackhi_epi32(s0, s1);
    __m128i t2 = _mm_unpacklo_epi32(s2, s3);
    __m128i t3 = _mm_unpackhi_epi32(s2, s3);
    __m128i u0 = _mm_add_epi32(t0, t1);
    __m128i u1 = _mm_add_epi32(t2, t3);
    return _mm_add_epi32(_mm_unpacklo_epi64(u0, u1), _mm_unpackhi_epi64(u0, u1));
}

/* 32-element partial sum with activations pre-expanded to int16.

   Two savings:
   1. Activations are expanded once per matmul (in_dim entries,
      resident in L1); the weight side needs no unpack+psraw for them -
      sign-extension work halved.
   2. Weights expand via punpcklbw(0, w): the result is w<<8, skipping
      psraw. Products are scaled 256x overall, cancelled by the caller
      with a 1/256 multiply at the row end (power of two, exact in
      float). Sign comes free and correct: byte 0xFF becomes int16
      0xFF00 = -256 = -1x256.

   No overflow within a group: |w<<8| <= 32512, |x| <= 127, a 32-group
   accumulation caps around 1.3e8. */
static __m128i part32_x16(const int8_t *w, const int16_t *x) {
    __m128i z = _mm_setzero_si128();
    __m128i w0 = _mm_load_si128((const __m128i *)w);
    __m128i w1 = _mm_load_si128((const __m128i *)(w + 16));
    __m128i w0l = _mm_unpacklo_epi8(z, w0);
    __m128i w0h = _mm_unpackhi_epi8(z, w0);
    __m128i w1l = _mm_unpacklo_epi8(z, w1);
    __m128i w1h = _mm_unpackhi_epi8(z, w1);
    __m128i x0 = _mm_load_si128((const __m128i *)x);
    __m128i x1 = _mm_load_si128((const __m128i *)(x + 8));
    __m128i x2 = _mm_load_si128((const __m128i *)(x + 16));
    __m128i x3 = _mm_load_si128((const __m128i *)(x + 24));
    __m128i s0 = _mm_add_epi32(_mm_madd_epi16(w0l, x0), _mm_madd_epi16(w0h, x1));
    __m128i s1 = _mm_add_epi32(_mm_madd_epi16(w1l, x2), _mm_madd_epi16(w1h, x3));
    return _mm_add_epi32(s0, s1);
}

/* The same sub-block against TWO tokens, sharing the weight side.
 *
 * More is shared here than at the MMX tier, which is the argument for
 * doing it: q16_0's pair shares only the load, because int16 weights
 * need no unpack. Here the weight costs two loads AND four unpacks
 * (w0l/w0h/w1l/w1h against zero), and a pair pays that once instead of
 * twice. Only the four activation loads and the eight pmaddwd double.
 *
 * Bit-identical to two part32_x16 calls: each token accumulates in its
 * own pair of registers in the same order, and nothing crosses between
 * them.
 *
 * SAME (z, w) OPERAND ORDER as the single form. The byte lands in the
 * HIGH half of each int16, so the product carries a factor of 256 the
 * caller's scale already accounts for - swapping the operands here
 * would be a silent 256x, which is the note the MMX twin carries too. */
static void part32_x16_2(const int8_t *w, const int16_t *xa,
                         const int16_t *xb, __m128i *pa, __m128i *pb) {
    __m128i z = _mm_setzero_si128();
    __m128i w0 = _mm_load_si128((const __m128i *)w);
    __m128i w1 = _mm_load_si128((const __m128i *)(w + 16));
    __m128i w0l = _mm_unpacklo_epi8(z, w0);
    __m128i w0h = _mm_unpackhi_epi8(z, w0);
    __m128i w1l = _mm_unpacklo_epi8(z, w1);
    __m128i w1h = _mm_unpackhi_epi8(z, w1);
    __m128i a0 = _mm_load_si128((const __m128i *)xa);
    __m128i a1 = _mm_load_si128((const __m128i *)(xa + 8));
    __m128i a2 = _mm_load_si128((const __m128i *)(xa + 16));
    __m128i a3 = _mm_load_si128((const __m128i *)(xa + 24));
    __m128i b0 = _mm_load_si128((const __m128i *)xb);
    __m128i b1 = _mm_load_si128((const __m128i *)(xb + 8));
    __m128i b2 = _mm_load_si128((const __m128i *)(xb + 16));
    __m128i b3 = _mm_load_si128((const __m128i *)(xb + 24));
    *pa = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(w0l, a0), _mm_madd_epi16(w0h, a1)),
              _mm_add_epi32(_mm_madd_epi16(w1l, a2), _mm_madd_epi16(w1h, a3)));
    *pb = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(w0l, b0), _mm_madd_epi16(w0h, b1)),
              _mm_add_epi32(_mm_madd_epi16(w1l, b2), _mm_madd_epi16(w1h, b3)));
}

/* Q4_1 32-element partial sum (activations pre-expanded to int16).

   Nibble layout (see model.h): 32 elements in 16 bytes; byte j's low
   nibble is element j, high nibble element j+16. Hence
       pand(b, 0x0F)            -> elements 0..15, order preserved
       pand(psrlw(b,4), 0x0F)   -> elements 16..31, order preserved
   **The activation side needs NO reshuffle** - this is exactly why
   this layout is chosen over "two adjacent elements per byte" (which
   would unpack to even/odd lanes and force interleaving xw too).

   `psrlw` shifts 16-bit words, but with the 0x0F mask each byte still
   grabs its own high nibble: byte pair (b0,b1) as a word shifted
   right 4 gives low byte ((b1&0xF)<<4)|(b0>>4); AND with 0x0F leaves
   only b0>>4.

   Expansion follows Q8's `punpcklbw(0, x)`: result x<<8, skipping
   sign extension (nibbles are unsigned 0..15 anyway); products scale
   256x, cancelled once at the row end by the caller. No overflow:
   |w<<8| <= 3840, |x| <= 127, one lane accumulating 8 items caps
   around 3.9e6. */
static __m128i part32_q41(const unsigned char *w, const int16_t *x) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m128i z  = _mm_setzero_si128();
    __m128i b  = _mm_load_si128((const __m128i *)w);
    __m128i lo = _mm_and_si128(b, m0f);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
    __m128i l0 = _mm_unpacklo_epi8(z, lo);
    __m128i l1 = _mm_unpackhi_epi8(z, lo);
    __m128i h0 = _mm_unpacklo_epi8(z, hi);
    __m128i h1 = _mm_unpackhi_epi8(z, hi);
    __m128i x0 = _mm_load_si128((const __m128i *)x);
    __m128i x1 = _mm_load_si128((const __m128i *)(x + 8));
    __m128i x2 = _mm_load_si128((const __m128i *)(x + 16));
    __m128i x3 = _mm_load_si128((const __m128i *)(x + 24));
    __m128i s0 = _mm_add_epi32(_mm_madd_epi16(l0, x0), _mm_madd_epi16(l1, x1));
    __m128i s1 = _mm_add_epi32(_mm_madd_epi16(h0, x2), _mm_madd_epi16(h1, x3));
    return _mm_add_epi32(s0, s1);
}

/* The same sub-block against two tokens, sharing the weight side.
 *
 * Worth more here than at q8_0, which is why this format is second into
 * the tier. q8_0's weight side is two loads and four unpacks; q4_1's is
 * one load, a mask, a shift, a second mask and four unpacks, plus the
 * 0x0F constant kept live. All of it is paid once for two tokens, and
 * only the four activation loads and the eight pmaddwd double.
 *
 * Bit-identical to two part32_q41 calls: each token accumulates in its
 * own registers in the same order and nothing crosses between them. The
 * (z, nibble) operand order is the single form's, so the value still
 * lands in the high half of each int16 and the caller's 256x scale is
 * unchanged. */
static void part32_q41_2(const unsigned char *w, const int16_t *xa,
                         const int16_t *xb, __m128i *pa, __m128i *pb) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m128i z  = _mm_setzero_si128();
    __m128i b  = _mm_load_si128((const __m128i *)w);
    __m128i lo = _mm_and_si128(b, m0f);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), m0f);
    __m128i l0 = _mm_unpacklo_epi8(z, lo);
    __m128i l1 = _mm_unpackhi_epi8(z, lo);
    __m128i h0 = _mm_unpacklo_epi8(z, hi);
    __m128i h1 = _mm_unpackhi_epi8(z, hi);
    __m128i a0 = _mm_load_si128((const __m128i *)xa);
    __m128i a1 = _mm_load_si128((const __m128i *)(xa + 8));
    __m128i a2 = _mm_load_si128((const __m128i *)(xa + 16));
    __m128i a3 = _mm_load_si128((const __m128i *)(xa + 24));
    __m128i b0 = _mm_load_si128((const __m128i *)xb);
    __m128i b1 = _mm_load_si128((const __m128i *)(xb + 8));
    __m128i b2 = _mm_load_si128((const __m128i *)(xb + 16));
    __m128i b3 = _mm_load_si128((const __m128i *)(xb + 24));
    *pa = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(l0, a0), _mm_madd_epi16(l1, a1)),
              _mm_add_epi32(_mm_madd_epi16(h0, a2), _mm_madd_epi16(h1, a3)));
    *pb = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(l0, b0), _mm_madd_epi16(l1, b1)),
              _mm_add_epi32(_mm_madd_epi16(h0, b2), _mm_madd_epi16(h1, b3)));
}

/* Q6_1 SSE2 version: isomorphic to the MMX version - rebuild the 6-bit
   value in the BYTE domain and unpack once, rather than computing two
   dot products and merging. q = lo | (hi<<4), lo<16 and hi<4 do not
   overlap.

   The 2-bit plane has 8 bytes per 32 elements; the four groups
   {0-7}{8-15}{16-23}{24-31} are (b>>0)&3, (b>>2)&3, (b>>4)&3,
   (b>>6)&3; unpacklo_epi64 joins two groups into 16 bytes, aligning
   byte-for-byte with the 4-bit plane's lo/hi nibbles.

   `_mm_slli_epi16(x, 4)` is safe for byte values 0..3: byte 0's bits
   0,1 move to 4,5 and byte 1's bits 8,9 to 12,13 - no cross-byte
   spill. */
static __m128i part32_q61(const unsigned char *w4, const unsigned char *w2,
                          const int16_t *x) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i m03 = _mm_set1_epi8(0x03);
    __m128i z  = _mm_setzero_si128();
    __m128i b  = _mm_load_si128((const __m128i *)w4);
    __m128i t  = _mm_loadl_epi64((const __m128i *)w2);
    __m128i g0 = _mm_and_si128(t, m03);
    __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03);
    __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03);
    __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03);
    __m128i lo = _mm_or_si128(_mm_and_si128(b, m0f),
                              _mm_slli_epi16(_mm_unpacklo_epi64(g0, g1), 4));
    __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b, 4), m0f),
                              _mm_slli_epi16(_mm_unpacklo_epi64(g2, g3), 4));
    __m128i l0 = _mm_unpacklo_epi8(z, lo);
    __m128i l1 = _mm_unpackhi_epi8(z, lo);
    __m128i h0 = _mm_unpacklo_epi8(z, hi);
    __m128i h1 = _mm_unpackhi_epi8(z, hi);
    __m128i x0 = _mm_load_si128((const __m128i *)x);
    __m128i x1 = _mm_load_si128((const __m128i *)(x + 8));
    __m128i x2 = _mm_load_si128((const __m128i *)(x + 16));
    __m128i x3 = _mm_load_si128((const __m128i *)(x + 24));
    __m128i s0 = _mm_add_epi32(_mm_madd_epi16(l0, x0), _mm_madd_epi16(l1, x1));
    __m128i s1 = _mm_add_epi32(_mm_madd_epi16(h0, x2), _mm_madd_epi16(h1, x3));
    return _mm_add_epi32(s0, s1);
}

/* The same sub-block against two tokens - and of the five formats this
 * is where the pair is worth most.
 *
 * Count the weight side: two loads, four masks, three shifts, two ORs,
 * two 64-bit merges and four unpacks, because a 6-bit value has to be
 * reassembled from a 4-bit plane and a 2-bit plane before it can be
 * widened. That is roughly nineteen instructions against q4_1's eight
 * and q8_0's six, and the pair pays it once. Only the four activation
 * loads and the eight pmaddwd double, exactly as in the other two.
 *
 * Bit-identical to two part32_q61 calls: the reassembly is the single
 * form's line for line, so the 6-bit value and the caller's scale are
 * unchanged, and each token accumulates in its own registers in the
 * same order with nothing crossing between them. */
static void part32_q61_2(const unsigned char *w4, const unsigned char *w2,
                         const int16_t *xa, const int16_t *xb,
                         __m128i *pa, __m128i *pb) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i m03 = _mm_set1_epi8(0x03);
    __m128i z  = _mm_setzero_si128();
    __m128i b  = _mm_load_si128((const __m128i *)w4);
    __m128i t  = _mm_loadl_epi64((const __m128i *)w2);
    __m128i g0 = _mm_and_si128(t, m03);
    __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03);
    __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03);
    __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03);
    __m128i lo = _mm_or_si128(_mm_and_si128(b, m0f),
                              _mm_slli_epi16(_mm_unpacklo_epi64(g0, g1), 4));
    __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(b, 4), m0f),
                              _mm_slli_epi16(_mm_unpacklo_epi64(g2, g3), 4));
    __m128i l0 = _mm_unpacklo_epi8(z, lo);
    __m128i l1 = _mm_unpackhi_epi8(z, lo);
    __m128i h0 = _mm_unpacklo_epi8(z, hi);
    __m128i h1 = _mm_unpackhi_epi8(z, hi);
    __m128i a0 = _mm_load_si128((const __m128i *)xa);
    __m128i a1 = _mm_load_si128((const __m128i *)(xa + 8));
    __m128i a2 = _mm_load_si128((const __m128i *)(xa + 16));
    __m128i a3 = _mm_load_si128((const __m128i *)(xa + 24));
    __m128i b0 = _mm_load_si128((const __m128i *)xb);
    __m128i b1 = _mm_load_si128((const __m128i *)(xb + 8));
    __m128i b2 = _mm_load_si128((const __m128i *)(xb + 16));
    __m128i b3 = _mm_load_si128((const __m128i *)(xb + 24));
    *pa = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(l0, a0), _mm_madd_epi16(l1, a1)),
              _mm_add_epi32(_mm_madd_epi16(h0, a2), _mm_madd_epi16(h1, a3)));
    *pb = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(l0, b0), _mm_madd_epi16(l1, b1)),
              _mm_add_epi32(_mm_madd_epi16(h0, b2), _mm_madd_epi16(h1, b3)));
}

/* T2 (ternary) SSE2: part32_q61 with the 4-bit plane deleted.
 *
 * That is the whole derivation, and it is why this format got a kernel
 * for the price of reading one. LZ_FMT_T2 was given Q6_1's 2-bit plane
 * layout precisely so that this would be true - 8 bytes per 32
 * elements, groups {0-7}{8-15}{16-23}{24-31} at shifts 0/2/4/6 - so the
 * unpack here is the same six instructions, minus the `m0f` masking,
 * the `slli 4` and the two `or`s that merged the planes.
 *
 * Codes are 0..2 and read UNSIGNED, so `unpacklo_epi8(z, code)` needs no
 * sign extension and lands the byte in the high half of each int16 -
 * the x256 fold, cancelled once by epi_q41's `1/256`, exactly as for
 * Q4_1. The -1 that turns a code into a value is NOT here: it is the
 * hoisted `- Sum x` term, computed per group from activations alone.
 *
 * Bit-identical to matmul_scalar_ref's T2 branch by construction: int32
 * accumulation is exact and order-independent, and 256 is a power of two
 * so the fold changes no rounding at any step. */
static __m128i part32_t2(const unsigned char *w2, const int16_t *x) {
    const __m128i m03 = _mm_set1_epi8(0x03);
    __m128i z  = _mm_setzero_si128();
    __m128i t  = _mm_loadl_epi64((const __m128i *)(const void *)w2);
    __m128i g0 = _mm_and_si128(t, m03);
    __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03);
    __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03);
    __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03);
    __m128i lo = _mm_unpacklo_epi64(g0, g1);     /* codes  0..15 */
    __m128i hi = _mm_unpacklo_epi64(g2, g3);     /* codes 16..31 */
    __m128i l0 = _mm_unpacklo_epi8(z, lo);
    __m128i l1 = _mm_unpackhi_epi8(z, lo);
    __m128i h0 = _mm_unpacklo_epi8(z, hi);
    __m128i h1 = _mm_unpackhi_epi8(z, hi);
    __m128i x0 = _mm_load_si128((const __m128i *)(const void *)x);
    __m128i x1 = _mm_load_si128((const __m128i *)(const void *)(x + 8));
    __m128i x2 = _mm_load_si128((const __m128i *)(const void *)(x + 16));
    __m128i x3 = _mm_load_si128((const __m128i *)(const void *)(x + 24));
    __m128i s0 = _mm_add_epi32(_mm_madd_epi16(l0, x0), _mm_madd_epi16(l1, x1));
    __m128i s1 = _mm_add_epi32(_mm_madd_epi16(h0, x2), _mm_madd_epi16(h1, x3));
    return _mm_add_epi32(s0, s1);
}

/* The same sub-block against two tokens, sharing the ternary decode.
 *
 * T2 IS NOT THE CHEAPEST FORMAT AT THIS TIER, which is worth stating
 * because it is at the MMX one. There the unpack is a zero register and
 * three immediate shifts, and a pair would share almost nothing. Here
 * the four 2-bit fields are extracted with four masks, three shifts and
 * two 64-bit merges before the four unpacks - fourteen instructions on
 * the weight side, second only to q6_1's nineteen and more than q4_1's
 * eight. Same operator, different decode, opposite conclusion.
 *
 * Bit-identical to two part32_t2 calls: the decode is the single form's
 * line for line, each token accumulates in its own registers in the
 * same order, and nothing crosses between them. */
static void part32_t2_2(const unsigned char *w2, const int16_t *xa,
                        const int16_t *xb, __m128i *pa, __m128i *pb) {
    const __m128i m03 = _mm_set1_epi8(0x03);
    __m128i z  = _mm_setzero_si128();
    __m128i t  = _mm_loadl_epi64((const __m128i *)(const void *)w2);
    __m128i g0 = _mm_and_si128(t, m03);
    __m128i g1 = _mm_and_si128(_mm_srli_epi16(t, 2), m03);
    __m128i g2 = _mm_and_si128(_mm_srli_epi16(t, 4), m03);
    __m128i g3 = _mm_and_si128(_mm_srli_epi16(t, 6), m03);
    __m128i lo = _mm_unpacklo_epi64(g0, g1);
    __m128i hi = _mm_unpacklo_epi64(g2, g3);
    __m128i l0 = _mm_unpacklo_epi8(z, lo);
    __m128i l1 = _mm_unpackhi_epi8(z, lo);
    __m128i h0 = _mm_unpacklo_epi8(z, hi);
    __m128i h1 = _mm_unpackhi_epi8(z, hi);
    __m128i a0 = _mm_load_si128((const __m128i *)(const void *)xa);
    __m128i a1 = _mm_load_si128((const __m128i *)(const void *)(xa + 8));
    __m128i a2 = _mm_load_si128((const __m128i *)(const void *)(xa + 16));
    __m128i a3 = _mm_load_si128((const __m128i *)(const void *)(xa + 24));
    __m128i b0 = _mm_load_si128((const __m128i *)(const void *)xb);
    __m128i b1 = _mm_load_si128((const __m128i *)(const void *)(xb + 8));
    __m128i b2 = _mm_load_si128((const __m128i *)(const void *)(xb + 16));
    __m128i b3 = _mm_load_si128((const __m128i *)(const void *)(xb + 24));
    *pa = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(l0, a0), _mm_madd_epi16(l1, a1)),
              _mm_add_epi32(_mm_madd_epi16(h0, a2), _mm_madd_epi16(h1, a3)));
    *pb = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(l0, b0), _mm_madd_epi16(l1, b1)),
              _mm_add_epi32(_mm_madd_epi16(h0, b2), _mm_madd_epi16(h1, b3)));
}

/* T2 over a 128-element group: four sub-blocks, four sums, one fold.
 *
 * WHAT THIS SAVES, because the registry's reason for leaving the t2/g128
 * cell empty examined the wrong half of the kernel. That argument was
 * about the UNPACK - q6_1's group form hoists two mask constants across
 * four sub-blocks, and T2's unpack has only a zero register to hoist,
 * which trades pxor for movq one instruction for one. Both true, and
 * neither is where a group form pays.
 *
 * It pays on the HORIZONTAL FOLD. Per-32, every sub-block reduces its
 * own four lanes and extracts: two srli_si128, two add and one cvtsi128
 * each, twenty instructions for four sub-blocks. fold4_sse2 transposes
 * the four partials and reduces them together - four unpack32, three
 * add, two unpack64 - and the four results leave in ONE register, so the
 * caller stores sixteen bytes once instead of extracting four int32.
 *
 * Bit-identical to four part32_t2 calls folded singly: int32 addition is
 * associative and exact, and the transpose only changes which lanes meet
 * first. Same argument fold4_sse2's existing callers rest on. */
static void dot128_t2_sse2(const unsigned char *w2, const int16_t *x,
                           int32_t *out4) {
    __m128i s0 = part32_t2(w2,      x);
    __m128i s1 = part32_t2(w2 +  8, x +  32);
    __m128i s2 = part32_t2(w2 + 16, x +  64);
    __m128i s3 = part32_t2(w2 + 24, x +  96);
    _mm_storeu_si128((__m128i *)(void *)out4, fold4_sse2(s0, s1, s2, s3));
}

/* Q16_0: 32 int16 weights against 32 int16 activations, 4-way partial
   sum, no horizontal reduction - same contract as part32_x16, so
   fold4_sse2 folds four of these the same way.

   The simplest of the four formats: int16 weights need no unpacking at
   all, so this is four pmaddwd and three paddd. This is the shape a
   dispatch table makes visible and an #if maze does not - a NULL slot
   is a hole you can see, an absent #elif branch is not.

   Bit-identical to the MMX and asm kernels by construction: int32
   accumulation is exact and order-independent, and the float epilogue
   (epi_q8 with post=1.0, since Q16_0 kernels do not scale by 256) is
   untouched. */
static __m128i part32_q16(const int16_t *w, const int16_t *x) {
    __m128i s0 = _mm_add_epi32(
        _mm_madd_epi16(_mm_load_si128((const __m128i *)(const void *)w),
                       _mm_load_si128((const __m128i *)(const void *)x)),
        _mm_madd_epi16(_mm_load_si128((const __m128i *)(const void *)(w + 8)),
                       _mm_load_si128((const __m128i *)(const void *)(x + 8))));
    __m128i s1 = _mm_add_epi32(
        _mm_madd_epi16(_mm_load_si128((const __m128i *)(const void *)(w + 16)),
                       _mm_load_si128((const __m128i *)(const void *)(x + 16))),
        _mm_madd_epi16(_mm_load_si128((const __m128i *)(const void *)(w + 24)),
                       _mm_load_si128((const __m128i *)(const void *)(x + 24))));
    return _mm_add_epi32(s0, s1);
}

/* The same sub-block against two tokens - and this is the SMALLEST pair
 * of the five, which is worth saying plainly rather than implying the
 * gain the other four get.
 *
 * int16 weights need no unpack at all: the weight side is four loads and
 * nothing else, so that is the entire saving. Against q6_1's nineteen
 * instructions, t2's fourteen, q4_1's eight and q8_0's six. It is here
 * because a partial ISA capability obliges a kernel, not because the
 * arithmetic argues for one, and whether four shared loads clear the
 * wider prologue is what the knob is for.
 *
 * Bit-identical to two part32_q16 calls: same four madd per token into
 * that token's own accumulator, same order, no reassociation. */
static void part32_q16_2(const int16_t *w, const int16_t *xa,
                         const int16_t *xb, __m128i *pa, __m128i *pb) {
    __m128i w0 = _mm_load_si128((const __m128i *)(const void *)w);
    __m128i w1 = _mm_load_si128((const __m128i *)(const void *)(w + 8));
    __m128i w2 = _mm_load_si128((const __m128i *)(const void *)(w + 16));
    __m128i w3 = _mm_load_si128((const __m128i *)(const void *)(w + 24));
    __m128i a0 = _mm_load_si128((const __m128i *)(const void *)xa);
    __m128i a1 = _mm_load_si128((const __m128i *)(const void *)(xa + 8));
    __m128i a2 = _mm_load_si128((const __m128i *)(const void *)(xa + 16));
    __m128i a3 = _mm_load_si128((const __m128i *)(const void *)(xa + 24));
    __m128i b0 = _mm_load_si128((const __m128i *)(const void *)xb);
    __m128i b1 = _mm_load_si128((const __m128i *)(const void *)(xb + 8));
    __m128i b2 = _mm_load_si128((const __m128i *)(const void *)(xb + 16));
    __m128i b3 = _mm_load_si128((const __m128i *)(const void *)(xb + 24));
    *pa = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(w0, a0), _mm_madd_epi16(w1, a1)),
              _mm_add_epi32(_mm_madd_epi16(w2, a2), _mm_madd_epi16(w3, a3)));
    *pb = _mm_add_epi32(
              _mm_add_epi32(_mm_madd_epi16(w0, b0), _mm_madd_epi16(w1, b1)),
              _mm_add_epi32(_mm_madd_epi16(w2, b2), _mm_madd_epi16(w3, b3)));
}
