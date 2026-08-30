/* The gcc-intrinsics half of every 32-element MMX dot product, one pair
   per quantized weight format. Included exactly once, by src/ops_mmx.c
   only - this is the -mmmx translation unit, so these are the only
   copies of these functions gcc ever compiles.

   Declared in src/ops_mmx.h; not a standalone translation unit itself
   (no include guard - same reasoning src/ops_kernel_dot.h gives for why
   splitting a file must not become a codegen change).

   EACH FUNCTION'S WATCOM #pragma aux TWIN LIVES IN src/ops_mmx.c, named
   per format below - Watcom compiles no MMX intrinsics and never
   carries -mmmx, so the two halves never meet in one build. The pair is
   step-for-step the same kernel; that comment lives with the Watcom
   twin, not duplicated here.

   The twins are referenced BY SYMBOL NAME and not by file:line. A line
   range goes stale two ways here, and both have happened: the bodies
   move between src/ops_kernel_dot.h and this file's Watcom counterpart,
   and rescheduling them shifts every line again. A name survives
   both. */

/* ---- Q8_0: dot32_x16_mmx / dot32_x16_mmx_2 -----------------------------
   Watcom twins: lz_dot32_x16_asm / lz_dot32_x16_asm_2, src/ops_mmx.c. */
int32_t dot32_x16_mmx(const int8_t *w, const int16_t *x) {
    __m64 z = _mm_setzero_si64();
    __m64 w0, w1, w2, w3, x0, s, t;
    memcpy(&w0, w, 8);      memcpy(&w1, w + 8, 8);
    memcpy(&w2, w + 16, 8); memcpy(&w3, w + 24, 8);
    s = _mm_setzero_si64();
#define MMX_MAC(WREG, XOFF)                                             \
    memcpy(&x0, x + (XOFF), 8);                                         \
    t = _mm_unpacklo_pi8(z, WREG);                                      \
    t = _mm_madd_pi16(t, x0);                                           \
    s = _mm_add_pi32(s, t);                                             \
    memcpy(&x0, x + (XOFF) + 4, 8);                                     \
    t = _mm_unpackhi_pi8(z, WREG);                                      \
    t = _mm_madd_pi16(t, x0);                                           \
    s = _mm_add_pi32(s, t)
    MMX_MAC(w0, 0);
    MMX_MAC(w1, 8);
    MMX_MAC(w2, 16);
    MMX_MAC(w3, 24);
#undef MMX_MAC
    t = _mm_srli_si64(s, 32);
    s = _mm_add_pi32(s, t);
    return (int32_t)_mm_cvtsi64_si32(s);
}

/* Two tokens, ONE weight unpack - see the Watcom twin's comment for the
   op-count accounting (36/24 ceiling, NT=2 register budget). BIT-
   IDENTICAL BY CONSTRUCTION: int32 accumulation is exact regardless of
   order, and the float epilogue is untouched. */
void dot32_x16_mmx_2(const int8_t *w, const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob) {
    __m64 z = _mm_setzero_si64();
    __m64 sa = _mm_setzero_si64(), sb = _mm_setzero_si64();
    __m64 wb, lo, hi, xv, t;
    int blk;
    for (blk = 0; blk < 32; blk += 8) {
        memcpy(&wb, w + blk, 8);
        /* Same (z, w) operand order as the single-token kernel: the byte
           lands in the HIGH half of each int16, i.e. the product carries
           a factor of 256 that the caller's scale already accounts for.
           Swapping the operands here would be a silent 256x. */
        lo = _mm_unpacklo_pi8(z, wb);
        hi = _mm_unpackhi_pi8(z, wb);
        memcpy(&xv, xa + blk, 8);
        t = _mm_madd_pi16(lo, xv);
        sa = _mm_add_pi32(sa, t);
        memcpy(&xv, xa + blk + 4, 8);
        t = _mm_madd_pi16(hi, xv);
        sa = _mm_add_pi32(sa, t);
        memcpy(&xv, xb + blk, 8);
        t = _mm_madd_pi16(lo, xv);
        sb = _mm_add_pi32(sb, t);
        memcpy(&xv, xb + blk + 4, 8);
        t = _mm_madd_pi16(hi, xv);
        sb = _mm_add_pi32(sb, t);
    }
    t = _mm_srli_si64(sa, 32);
    sa = _mm_add_pi32(sa, t);
    *oa = _mm_cvtsi64_si32(sa);
    t = _mm_srli_si64(sb, 32);
    sb = _mm_add_pi32(sb, t);
    *ob = _mm_cvtsi64_si32(sb);
}

/* One 128-element group, four 32-element sub-blocks, ONE zero-register
   setup instead of four - the Watcom twin (lz_dot128_x16_asm,
   src/ops_mmx.c) hoists its pxor mm7,mm7 across all four sub-blocks by
   hand, and this does the same with a single `z` whose liveness spans the
   whole function, so the compiler is free to keep it resident instead of
   reconstructing it per sub-block. Each sub-block accumulates into its
   own `s`, folds and stores separately: out4[] holds the same four ints
   four dot32_x16_mmx calls would write, in the same order - bit-identical
   by construction, and the per-sub-block pmaddwd sequence matches the
   32-element kernel step for step. */
void dot128_x16_mmx(const int8_t *w, const int16_t *x, int32_t *out4) {
    __m64 z = _mm_setzero_si64();
    int sb;
    for (sb = 0; sb < 4; sb++) {
        __m64 s = _mm_setzero_si64();
        __m64 w0, w1, w2, w3, x0, t;
        memcpy(&w0, w + (size_t)sb * 32, 8);
        memcpy(&w1, w + (size_t)sb * 32 + 8, 8);
        memcpy(&w2, w + (size_t)sb * 32 + 16, 8);
        memcpy(&w3, w + (size_t)sb * 32 + 24, 8);
#define MMX_MAC128(WREG, XOFF)                                          \
        memcpy(&x0, x + (size_t)sb * 32 + (XOFF), 8);                   \
        t = _mm_unpacklo_pi8(z, WREG);                                  \
        t = _mm_madd_pi16(t, x0);                                       \
        s = _mm_add_pi32(s, t);                                         \
        memcpy(&x0, x + (size_t)sb * 32 + (XOFF) + 4, 8);               \
        t = _mm_unpackhi_pi8(z, WREG);                                  \
        t = _mm_madd_pi16(t, x0);                                       \
        s = _mm_add_pi32(s, t)
        MMX_MAC128(w0, 0);
        MMX_MAC128(w1, 8);
        MMX_MAC128(w2, 16);
        MMX_MAC128(w3, 24);
#undef MMX_MAC128
        t = _mm_srli_si64(s, 32);
        s = _mm_add_pi32(s, t);
        out4[sb] = (int32_t)_mm_cvtsi64_si32(s);
    }
}

/* ---- Q16_0: dot32_q16_mmx ----------------------------------------------
   Watcom twin: lz_dot32_q16_asm, src/ops_mmx.c.
   Dev-build twin: mirrors the asm step for step, eight pmaddwd
   into one accumulator, then fold the high dword into the low one. No
   reassociation - the integer sums are exact, but keeping the shape
   identical is what makes "same numbers" mean "same kernel". */
int32_t dot32_q16_mmx(const int16_t *w, const int16_t *x) {
    __m64 s = _mm_setzero_si64(), w0, x0, t;
    int k;
    for (k = 0; k < 32; k += 4) {
        memcpy(&w0, w + k, 8);
        memcpy(&x0, x + k, 8);
        t = _mm_madd_pi16(w0, x0);
        s = _mm_add_pi32(s, t);
    }
    t = _mm_srli_si64(s, 32);
    s = _mm_add_pi32(s, t);
    return (int32_t)_mm_cvtsi64_si32(s);
}

/* Two tokens against one weight load - the same trick q8_0, q4_1 and
   q6_1 already use, which q16_0 and t2 did not have on either
   toolchain. Eight registers is what makes it worth doing: the weight
   vector is loaded once and stays live while both tokens' activations
   pass through it, so the pair costs one load where two single calls
   cost two.
 *
 * Less to save here than at q8_0, and it is worth being exact about
 * why: int16 weights need no unpack, so what is shared is the LOAD
 * alone, not a load plus two punpck. Whether that clears the wider
 * prologue is a wall-clock question on hardware this project does not
 * have - hence the knob, not a claim.
 *
 * Bit-identical to two dot32_q16_mmx calls by construction: same eight
 * pmaddwd per token into that token's own accumulator, same fold, no
 * reassociation across the pair. */
void dot32_q16_mmx_2(const int16_t *w, const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob) {
    __m64 sa = _mm_setzero_si64(), sb = _mm_setzero_si64();
    __m64 w0, x0, t;
    int k;
    for (k = 0; k < 32; k += 4) {
        memcpy(&w0, w + k, 8);
        memcpy(&x0, xa + k, 8);
        t = _mm_madd_pi16(w0, x0);
        sa = _mm_add_pi32(sa, t);
        memcpy(&x0, xb + k, 8);
        t = _mm_madd_pi16(w0, x0);
        sb = _mm_add_pi32(sb, t);
    }
    sa = _mm_add_pi32(sa, _mm_srli_si64(sa, 32));
    sb = _mm_add_pi32(sb, _mm_srli_si64(sb, 32));
    *oa = (int32_t)_mm_cvtsi64_si32(sa);
    *ob = (int32_t)_mm_cvtsi64_si32(sb);
}

/* The 128-element Q16 group kernel: int16 weights need no unpack, so
   there is no mask to hoist - what the Watcom twin (lz_dot128_q16_asm)
   amortizes is the per-32 call and its prologue, by processing four
   sub-blocks in one function. Same shape: eight pmaddwd into one
   accumulator per sub-block, fold, store separately. */
void dot128_q16_mmx(const int16_t *w, const int16_t *x, int32_t *out4) {
    int sb;
    for (sb = 0; sb < 4; sb++) {
        __m64 s = _mm_setzero_si64(), w0, x0, t;
        int k;
        for (k = 0; k < 32; k += 4) {
            memcpy(&w0, w + (size_t)sb * 32 + k, 8);
            memcpy(&x0, x + (size_t)sb * 32 + k, 8);
            t = _mm_madd_pi16(w0, x0);
            s = _mm_add_pi32(s, t);
        }
        t = _mm_srli_si64(s, 32);
        s = _mm_add_pi32(s, t);
        out4[sb] = (int32_t)_mm_cvtsi64_si32(s);
    }
}

/* ---- Q4_1: dot32_q41_mmx / dot32_q41_mmx_2 -----------------------------
   Watcom twins: lz_dot32_q41_asm / lz_dot32_q41_asm_2 (and
   lz_dot128_q41_asm), src/ops_mmx.c.
   The kernel that matters most: most of the model runs on this path (63
   Q4_1 tensors on the recipe of record). */
int32_t dot32_q41_mmx(const unsigned char *w, const int16_t *x) {
    const __m64 m0f = (__m64)0x0F0F0F0F0F0F0F0FULL;
    __m64 z = _mm_setzero_si64();
    __m64 b, lo, hi, x0, s, t;
    s = _mm_setzero_si64();
#define MMX_Q41(BOFF, XOFF)                                             \
    memcpy(&b, (const char *)w + (BOFF), 8);                            \
    lo = _mm_and_si64(b, m0f);                                          \
    hi = _mm_and_si64(_mm_srli_pi16(b, 4), m0f);                        \
    memcpy(&x0, x + (XOFF), 8);                                         \
    t = _mm_madd_pi16(_mm_unpacklo_pi8(z, lo), x0);                     \
    s = _mm_add_pi32(s, t);                                             \
    memcpy(&x0, x + (XOFF) + 4, 8);                                     \
    t = _mm_madd_pi16(_mm_unpackhi_pi8(z, lo), x0);                     \
    s = _mm_add_pi32(s, t);                                             \
    memcpy(&x0, x + (XOFF) + 16, 8);                                    \
    t = _mm_madd_pi16(_mm_unpacklo_pi8(z, hi), x0);                     \
    s = _mm_add_pi32(s, t);                                             \
    memcpy(&x0, x + (XOFF) + 20, 8);                                    \
    t = _mm_madd_pi16(_mm_unpackhi_pi8(z, hi), x0);                     \
    s = _mm_add_pi32(s, t)
    /* bytes 0..7  -> elements 0..7 and 16..23; bytes 8..15 -> elements 8..15 and 24..31 */
    MMX_Q41(0, 0);
    MMX_Q41(8, 8);
#undef MMX_Q41
    t = _mm_srli_si64(s, 32);
    s = _mm_add_pi32(s, t);
    return (int32_t)_mm_cvtsi64_si32(s);
}

/* Two tokens, ONE nibble unpack - see the Watcom twin's comment for the
   op-count accounting (40/24 ceiling). Bit-identical by construction. */
void dot32_q41_mmx_2(const unsigned char *w, const int16_t *xa,
                     const int16_t *xb, int32_t *oa, int32_t *ob) {
    const __m64 m0f = (__m64)0x0F0F0F0F0F0F0F0FULL;
    __m64 z = _mm_setzero_si64();
    __m64 sa = _mm_setzero_si64(), sb = _mm_setzero_si64();
    __m64 b, lo, hi, wv, xv;
    int blk;
    for (blk = 0; blk < 16; blk += 8) {
        int xo = blk;                    /* elements blk..blk+7 and blk+16.. */
        memcpy(&b, (const char *)w + blk, 8);
        lo = _mm_and_si64(b, m0f);
        hi = _mm_and_si64(_mm_srli_pi16(b, 4), m0f);
        /* Four widened halves per block, in the SAME order the
           single-token kernel emits them (lo-lo, lo-hi, hi-lo, hi-hi)
           against activation offsets +0, +4, +16, +20. */
#define Q41_2_HALF(WH, XOFF)                                            \
        wv = (WH);                                                      \
        memcpy(&xv, xa + xo + (XOFF), 8);                               \
        sa = _mm_add_pi32(sa, _mm_madd_pi16(wv, xv));                   \
        memcpy(&xv, xb + xo + (XOFF), 8);                               \
        sb = _mm_add_pi32(sb, _mm_madd_pi16(wv, xv))
        Q41_2_HALF(_mm_unpacklo_pi8(z, lo), 0);
        Q41_2_HALF(_mm_unpackhi_pi8(z, lo), 4);
        Q41_2_HALF(_mm_unpacklo_pi8(z, hi), 16);
        Q41_2_HALF(_mm_unpackhi_pi8(z, hi), 20);
#undef Q41_2_HALF
    }
    b = _mm_srli_si64(sa, 32);
    sa = _mm_add_pi32(sa, b);
    *oa = (int32_t)_mm_cvtsi64_si32(sa);
    b = _mm_srli_si64(sb, 32);
    sb = _mm_add_pi32(sb, b);
    *ob = (int32_t)_mm_cvtsi64_si32(sb);
}

/* One 128-element group, four 32-element sub-blocks, ONE nibble unpack
   setup - the Watcom twin (lz_dot128_q41_asm) hoists its zero register
   AND its 0x0F mask across all four sub-blocks; this hoists the same
   two (`z`, `m0f`), leaving them live for the whole function instead of
   reconstructing them per sub-block. Per sub-block the sequence matches
   dot32_q41_mmx: the low-nibble byte pair (bytes 0..7, 8..15) feeds
   activation offsets +0/+8 and the high-nibble pair feeds +16/+24, each
   unpacked low and high, pmaddwd, accumulate into that sub-block's own
   `s`, fold, store. out4[] is the same four ints four dot32_q41_mmx
   calls would write. */
void dot128_q41_mmx(const unsigned char *w, const int16_t *x, int32_t *out4) {
    const __m64 m0f = (__m64)0x0F0F0F0F0F0F0F0FULL;
    __m64 z = _mm_setzero_si64();
    int sb;
    for (sb = 0; sb < 4; sb++) {
        __m64 s = _mm_setzero_si64();
        __m64 b, lo, hi, x0, t;
#define MMX_Q41_128(BOFF, XOFF)                                         \
        memcpy(&b, w + (size_t)sb * 16 + (BOFF), 8);                    \
        lo = _mm_and_si64(b, m0f);                                      \
        hi = _mm_and_si64(_mm_srli_pi16(b, 4), m0f);                    \
        memcpy(&x0, x + (size_t)sb * 32 + (XOFF), 8);                   \
        t = _mm_madd_pi16(_mm_unpacklo_pi8(z, lo), x0);                 \
        s = _mm_add_pi32(s, t);                                         \
        memcpy(&x0, x + (size_t)sb * 32 + (XOFF) + 4, 8);               \
        t = _mm_madd_pi16(_mm_unpackhi_pi8(z, lo), x0);                 \
        s = _mm_add_pi32(s, t);                                         \
        memcpy(&x0, x + (size_t)sb * 32 + (XOFF) + 16, 8);              \
        t = _mm_madd_pi16(_mm_unpacklo_pi8(z, hi), x0);                 \
        s = _mm_add_pi32(s, t);                                         \
        memcpy(&x0, x + (size_t)sb * 32 + (XOFF) + 20, 8);              \
        t = _mm_madd_pi16(_mm_unpackhi_pi8(z, hi), x0);                 \
        s = _mm_add_pi32(s, t)
        MMX_Q41_128(0, 0);
        MMX_Q41_128(8, 8);
#undef MMX_Q41_128
        t = _mm_srli_si64(s, 32);
        s = _mm_add_pi32(s, t);
        out4[sb] = (int32_t)_mm_cvtsi64_si32(s);
    }
}

/* ---- T2: dot32_t2_mmx ---------------------------------------------------
   Watcom twin: lz_dot32_t2_asm, src/ops_mmx.c.
   One 8-byte load of the plane, eight groups of (punpck against zero,
   align the 2-bit field to the top of the lane, drop the rest, put it
   back with the x256 fold, pmaddwd, accumulate), then a horizontal fold
   of two int32 lanes. THE SHIFT ORDER IS THE PART THAT IS EASY TO GET
   WRONG: after punpcklbw(0, byte) a lane holds byte<<8, so the 2-bit
   field at bit `sh` sits at lane bit 8+sh, and pmaddwd consumes the
   entire 16-bit lane. Aligning it with one right shift leaves the low
   byte's contents in the lane; push to the top, clear below, restore:
   psllw (6-sh) ; psrlw 14 ; psllw 8. */
/* The eight groups, stopping before the horizontal fold: two int32 lanes
   still to be summed. Split out so the 128-element form below can fold
   four of these TOGETHER - the arithmetic above the fold is untouched,
   which is what keeps both entries bit-identical to the original
   function. */
static __m64 part32_t2_mmx(const unsigned char *w2, const int16_t *x) {
    __m64 acc = _mm_setzero_si64();
    __m64 pl  = *(const __m64 *)(const void *)w2;
    int gi;
    /* (activation offset in int16s, use the high half of the plane byte
       group, 2-bit right shift) - the same eight rows the generator
       emits, in the same order. */
    static const struct { int xoff, hi, sh; } G[8] = {
        {  0, 0, 0 }, {  4, 1, 0 },
        {  8, 0, 2 }, { 12, 1, 2 },
        { 16, 0, 4 }, { 20, 1, 4 },
        { 24, 0, 6 }, { 28, 1, 6 }
    };
    for (gi = 0; gi < 8; gi++) {
        __m64 xv = *(const __m64 *)(const void *)(x + G[gi].xoff);
        __m64 z  = _mm_setzero_si64();
        __m64 t  = G[gi].hi ? _mm_unpackhi_pi8(z, pl)
                            : _mm_unpacklo_pi8(z, pl);
        if (6 - G[gi].sh) t = _mm_slli_pi16(t, 6 - G[gi].sh);
        t = _mm_srli_pi16(t, 14);
        t = _mm_slli_pi16(t, 8);
        acc = _mm_add_pi32(acc, _mm_madd_pi16(t, xv));
    }
    return acc;
}

int32_t dot32_t2_mmx(const unsigned char *w2, const int16_t *x) {
    __m64 acc = part32_t2_mmx(w2, x);
    acc = _mm_add_pi32(acc, _mm_srli_si64(acc, 32));
    return _mm_cvtsi64_si32(acc);
}

/* Two tokens against one ternary unpack, gcc side only. The C intrinsic
   _mm_madd_pi16 takes its operands read-only, so the unpacked weight `t`
   stays live across both tokens' multiplies and needs no copy. Bit-
   identical to two dot32_t2_mmx calls by construction: the same eight
   groups per token into that token's own accumulator, no reassociation
   across the pair. The Watcom twin is NOT written: its pmaddwd destroys
   the unpack register, so sharing it there costs a movq per group plus
   two more accumulators than eight %mm hold. */
void dot32_t2_mmx_2(const unsigned char *w2, const int16_t *xa, const int16_t *xb,
                    int32_t *oa, int32_t *ob) {
    __m64 sa = _mm_setzero_si64(), sb = _mm_setzero_si64();
    __m64 pl  = *(const __m64 *)(const void *)w2;
    int gi;
    static const struct { int xoff, hi, sh; } G[8] = {
        {  0, 0, 0 }, {  4, 1, 0 },
        {  8, 0, 2 }, { 12, 1, 2 },
        { 16, 0, 4 }, { 20, 1, 4 },
        { 24, 0, 6 }, { 28, 1, 6 }
    };
    for (gi = 0; gi < 8; gi++) {
        __m64 z  = _mm_setzero_si64();
        __m64 t  = G[gi].hi ? _mm_unpackhi_pi8(z, pl)
                            : _mm_unpacklo_pi8(z, pl);
        if (6 - G[gi].sh) t = _mm_slli_pi16(t, 6 - G[gi].sh);
        t = _mm_srli_pi16(t, 14);
        t = _mm_slli_pi16(t, 8);
        sa = _mm_add_pi32(sa, _mm_madd_pi16(t, *(const __m64 *)(const void *)(xa + G[gi].xoff)));
        sb = _mm_add_pi32(sb, _mm_madd_pi16(t, *(const __m64 *)(const void *)(xb + G[gi].xoff)));
    }
    sa = _mm_add_pi32(sa, _mm_srli_si64(sa, 32));
    sb = _mm_add_pi32(sb, _mm_srli_si64(sb, 32));
    *oa = (int32_t)_mm_cvtsi64_si32(sa);
    *ob = (int32_t)_mm_cvtsi64_si32(sb);
}

/* T2 over a 128-element group, MMX tier - the %mm twin of
   dot128_t2_sse2, and it pays for the same reason: the horizontal fold,
   not the unpack.
 *
 * Per-32 each sub-block folds its own two lanes and extracts - srli_si64,
 * add, cvtsi64_si32, three instructions each, twelve for four sub-blocks.
 * Pairwise here: unpackldq and unpackhdq bring two sub-blocks' lanes
 * alongside each other so ONE add finishes both, three instructions per
 * pair and six for four. The two results are whole %mm registers, so
 * they store as two 8-byte writes instead of four int32 extracts.
 *
 * Bit-identical to four dot32_t2_mmx calls: same part32_t2_mmx above,
 * and int32 addition is associative and exact, so pairing changes only
 * which lanes meet first.
 *
 * NO EMMS HERE. This leaves the register file in MMX state exactly as
 * dot32_t2_mmx does, and the caller's row kernel owns the emms - the
 * rule this file's other kernels already follow. */
void dot128_t2_mmx(const unsigned char *w2, const int16_t *x,
                   int32_t *out4) {
    __m64 a0 = part32_t2_mmx(w2,      x);
    __m64 a1 = part32_t2_mmx(w2 +  8, x +  32);
    __m64 a2 = part32_t2_mmx(w2 + 16, x +  64);
    __m64 a3 = part32_t2_mmx(w2 + 24, x +  96);
    __m64 p0 = _mm_add_pi32(_mm_unpacklo_pi32(a0, a1),
                            _mm_unpackhi_pi32(a0, a1));
    __m64 p1 = _mm_add_pi32(_mm_unpacklo_pi32(a2, a3),
                            _mm_unpackhi_pi32(a2, a3));
    *(__m64 *)(void *)(out4)     = p0;
    *(__m64 *)(void *)(out4 + 2) = p1;
}

/* ---- Q6_1: dot32_q61_mmx / dot32_q61_mmx_2 ------------------------------
   Watcom twins: lz_dot32_q61_asm / lz_dot32_q61_asm_2 (and
   lz_dot128_q61_asm), src/ops_mmx.c.
   Q6_1 carries embed_tokens, in_proj_qkv, in_proj_z, down_proj and
   q/k/v_proj - the largest tensors on the recipe of record - and its
   weight side is 61% of the kernel's ops (against Q4_1's 40% and Q8's
   33%) because a 6-bit value has to be assembled from a 4-bit plane and
   a 2-bit plane before it can be widened. */
int32_t dot32_q61_mmx(const unsigned char *w4, const unsigned char *w2,
                      const int16_t *x) {
    const __m64 m0f = (__m64)0x0F0F0F0F0F0F0F0FULL;
    const __m64 m03 = (__m64)0x0303030303030303ULL;
    __m64 z = _mm_setzero_si64();
    __m64 blk, two, q, u, x0, s;
    int i;
    s = _mm_setzero_si64();
    memcpy(&two, w2, 8);
    /* sub-block order 0,2,1,3: A low / A high / B low / B high, matching the asm version */
    for (i = 0; i < 4; i++) {
        static const int SH[4] = { 0, 4, 2, 6 };
        static const int XO[4] = { 0, 16, 8, 24 };
        memcpy(&blk, w4 + ((i == 0 || i == 1) ? 0 : 8), 8);
        q = (i == 1 || i == 3) ? _mm_srli_pi16(blk, 4) : blk;
        q = _mm_and_si64(q, m0f);
        u = _mm_and_si64(_mm_srli_pi16(two, SH[i]), m03);
        q = _mm_or_si64(q, _mm_slli_pi16(u, 4));
        memcpy(&x0, x + XO[i], 8);
        s = _mm_add_pi32(s, _mm_madd_pi16(_mm_unpacklo_pi8(z, q), x0));
        memcpy(&x0, x + XO[i] + 4, 8);
        s = _mm_add_pi32(s, _mm_madd_pi16(_mm_unpackhi_pi8(z, q), x0));
    }
    s = _mm_add_pi32(s, _mm_srli_si64(s, 32));
    return (int32_t)_mm_cvtsi64_si32(s);
}

/* Two tokens, ONE two-plane merge - the best of the three shared-unpack
   kernels and, on the recipe of record, the one that matters most (see
   the single-token comment above). One widened half is built at a time
   and immediately consumed by both tokens (rather than building both
   halves first), which keeps exactly the eight MMX registers live;
   building both halves first needs nine and spills. */
void dot32_q61_mmx_2(const unsigned char *w4, const unsigned char *w2,
                     const int16_t *xa, const int16_t *xb,
                     int32_t *oa, int32_t *ob) {
    const __m64 m0f = (__m64)0x0F0F0F0F0F0F0F0FULL;
    const __m64 m03 = (__m64)0x0303030303030303ULL;
    __m64 z = _mm_setzero_si64();
    __m64 blk, two, q, u, xv, sa, sb;
    int i;
    sa = _mm_setzero_si64();
    sb = _mm_setzero_si64();
    memcpy(&two, w2, 8);
    for (i = 0; i < 4; i++) {
        static const int SH[4] = { 0, 4, 2, 6 };
        static const int XO[4] = { 0, 16, 8, 24 };
        memcpy(&blk, w4 + ((i == 0 || i == 1) ? 0 : 8), 8);
        q = (i == 1 || i == 3) ? _mm_srli_pi16(blk, 4) : blk;
        q = _mm_and_si64(q, m0f);
        u = _mm_and_si64(_mm_srli_pi16(two, SH[i]), m03);
        q = _mm_or_si64(q, _mm_slli_pi16(u, 4));
        /* low half, both tokens */
        u = _mm_unpacklo_pi8(z, q);
        memcpy(&xv, xa + XO[i], 8);
        sa = _mm_add_pi32(sa, _mm_madd_pi16(u, xv));
        memcpy(&xv, xb + XO[i], 8);
        sb = _mm_add_pi32(sb, _mm_madd_pi16(u, xv));
        /* high half, both tokens */
        u = _mm_unpackhi_pi8(z, q);
        memcpy(&xv, xa + XO[i] + 4, 8);
        sa = _mm_add_pi32(sa, _mm_madd_pi16(u, xv));
        memcpy(&xv, xb + XO[i] + 4, 8);
        sb = _mm_add_pi32(sb, _mm_madd_pi16(u, xv));
    }
    sa = _mm_add_pi32(sa, _mm_srli_si64(sa, 32));
    *oa = (int32_t)_mm_cvtsi64_si32(sa);
    sb = _mm_add_pi32(sb, _mm_srli_si64(sb, 32));
    *ob = (int32_t)_mm_cvtsi64_si32(sb);
}

/* One 128-element group, four 32-element sub-blocks, ONE two-plane
   merge setup - the Watcom twin (lz_dot128_q61_asm) hoists its zero
   register, its 0x0F mask and its 0x03 mask across all four sub-blocks;
   this hoists the same three (`z`, `m0f`, `m03`). Per sub-block the
   sequence matches dot32_q61_mmx: the four (4-bit plane half, 2-bit
   shift) pairs against activation offsets 0/16/8/24, each unpacked low
   and high, pmaddwd, accumulate into that sub-block's own `s`, fold,
   store. out4[] is the same four ints four dot32_q61_mmx calls write. */
void dot128_q61_mmx(const unsigned char *w4, const unsigned char *w2,
                    const int16_t *x, int32_t *out4) {
    const __m64 m0f = (__m64)0x0F0F0F0F0F0F0F0FULL;
    const __m64 m03 = (__m64)0x0303030303030303ULL;
    __m64 z = _mm_setzero_si64();
    int sb;
    for (sb = 0; sb < 4; sb++) {
        __m64 s = _mm_setzero_si64();
        __m64 two, blk, q, u, x0;
        int i;
        memcpy(&two, w2 + (size_t)sb * 8, 8);
        for (i = 0; i < 4; i++) {
            static const int SH[4] = { 0, 4, 2, 6 };
            static const int XO[4] = { 0, 16, 8, 24 };
            memcpy(&blk, w4 + (size_t)sb * 16 + ((i == 0 || i == 1) ? 0 : 8), 8);
            q = (i == 1 || i == 3) ? _mm_srli_pi16(blk, 4) : blk;
            q = _mm_and_si64(q, m0f);
            u = _mm_and_si64(_mm_srli_pi16(two, SH[i]), m03);
            q = _mm_or_si64(q, _mm_slli_pi16(u, 4));
            memcpy(&x0, x + (size_t)sb * 32 + XO[i], 8);
            s = _mm_add_pi32(s, _mm_madd_pi16(_mm_unpacklo_pi8(z, q), x0));
            memcpy(&x0, x + (size_t)sb * 32 + XO[i] + 4, 8);
            s = _mm_add_pi32(s, _mm_madd_pi16(_mm_unpackhi_pi8(z, q), x0));
        }
        s = _mm_add_pi32(s, _mm_srli_si64(s, 32));
        out4[sb] = (int32_t)_mm_cvtsi64_si32(s);
    }
}
