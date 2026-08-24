/* The oracle any lz_epi_mac_i16 kernel has to match.
 *
 * The operator accumulates int32 x int16 -> int64:
 *
 *     for (g = 0; g < n; g++) s += (lz_i64)a[g] * (lz_i64)m[g];
 *
 * Its six x86 cells are empty (km_reg, "epi" row). SSE2 reaches it with
 * pmuludq plus sign correction and MMX by splitting the int32 into
 * 16-bit halves - both of which get the sign wrong quietly rather than
 * loudly. A wrong correction returns a plausible number, so the kernel
 * needs a predicate written before it exists, not after.
 *
 * This is that predicate. It is deliberately not a comparison against
 * the shipping C body: that body IS one of the things under test once a
 * kernel lands beside it. The reference here recomputes the sum a
 * second way - 32x32 partial products in unsigned arithmetic with the
 * signs handled explicitly - so agreement means two independent
 * derivations agree, not that one function was called twice.
 *
 * Build: gcc -O2 -std=c99 -Wall -Wextra -Isrc .prof/epi_mac_oracle.c -o /tmp/epi_oracle
 *
 * A NOTE FOR WHOEVER WRITES THE SSE2 KERNEL. SSE2 has no 32x32->32
 * multiply (_mm_mullo_epi32 is SSE4.1), so the two usable primitives
 * are pmuludq and pmaddwd, and both push you toward a 16-bit split.
 * The textbook signed split is
 *
 *     ah = (a >> 16) + ((a >> 15) & 1);   sal = (short)(a & 0xFFFF);
 *     a  = ah * 65536 + sal
 *
 * which lets pmaddwd do both halves because ah and sal are both
 * int16-shaped. It has one failure: for a near INT_MAX the carry pushes
 * ah to 32768, which is not int16, and the lane silently wraps to
 * -32768. a = 2147483647 is in the edge list above precisely so that
 * this is caught the first time it is run rather than in a model months
 * later. If a kernel passes the random sweep and fails the edge block,
 * this is the first thing to check.
 *
 * The carry is avoidable. It only exists because the textbook form
 * folds the borrow into ah, and nothing forces that: (a >> 16) is
 * already int16-shaped for any int32, and the borrow can be carried as
 * its own term instead.
 *
 *     ah  = a >> 16                 int16, no carry, never overflows
 *     als = (short)(a & 0xFFFF)     pmaddwd's signed reading of the low
 *     b   = (a >> 15) & 1           1 exactly when als read low by 2^16
 *
 *     a * m = ((ah * m) << 16) + (als * m) + ((b * m) << 16)
 *
 * Three accumulators, summed into 64 bits at the very end:
 *
 *     total = ((S_ah + S_b) << 16) + S_als
 *
 * S_ah and S_b are only added to each other in 64-bit, so the 32768
 * that broke the folded form never gets formed in a 16-bit lane. All
 * three products are int16 x int16, which is exactly pmaddwd's shape,
 * so no pmuludq and no sign correction are needed at all.
 *
 * Watch the accumulator width, not just the lane width: each product is
 * int32 and n runs to the group size, so the running sums need widening
 * to 64 bits before they can overflow, not after.
 */

#include <stdio.h>
#include <stdlib.h>
#include "lz_int.h"

#define SEED 1145141919u

/* The shape under test, as C. A kernel replaces this body; the harness
   does not change. */
static lz_i64 mac_direct(const int lz_i32_dummy_unused, const int *a,
                         const short *m, int n) {
    lz_i64 s = 0;
    int g;
    (void)lz_i32_dummy_unused;
    for (g = 0; g < n; g++) s += (lz_i64)a[g] * (lz_i64)m[g];
    return s;
}

/* Second derivation. Splits each int32 into a signed high half and an
   unsigned low half - which is exactly the decomposition an MMX kernel
   has to make - and reassembles in 64 bits. If a kernel gets the
   unsigned-low/signed-high distinction wrong, this is what disagrees
   with it. */
static lz_i64 mac_split(const int *a, const short *m, int n) {
    lz_i64 s = 0;
    int g;
    for (g = 0; g < n; g++) {
        lz_i64 ah = (lz_i64)(a[g] >> 16);            /* signed */
#ifdef EPI_MUT_SIGNED_LOW
        /* The mutation, and the specific bug it stands for: reading the
           low half as signed. pmaddwd and pmullw are signed operations,
           so an MMX kernel that feeds them the low half untouched makes
           exactly this error - and only for a[g] with bit 15 set, which
           is why an edge-case list that stops at 32767 misses it. A
           predicate that cannot go red here is not testing the kernel
           it was written for. */
        lz_i64 al = (lz_i64)(short)((unsigned)a[g] & 0xFFFFu);
#else
        lz_i64 al = (lz_i64)((unsigned)a[g] & 0xFFFFu); /* unsigned */
#endif
        lz_i64 mm = (lz_i64)m[g];
        s += ((ah * mm) << 16) + (al * mm);
    }
    return s;
}

/* Third derivation: the carry-free split the header describes, in
   scalar C. The formula is what an SSE2 or MMX kernel will encode, so
   it gets checked here before anyone writes intrinsics against it - a
   wrong formula and a wrong kernel look identical from the outside.
   Every product below is int16 x int16, which is what makes this
   pmaddwd's shape; the three sums meet only in 64 bits. */
static lz_i64 mac_carry_free(const int *a, const short *m, int n) {
    lz_i64 s_ah = 0, s_als = 0, s_b = 0;
    int g;
    for (g = 0; g < n; g++) {
        int ah  = a[g] >> 16;                        /* int16 range */
        int als = (short)((unsigned)a[g] & 0xFFFFu); /* signed low */
        int b   = (a[g] >> 15) & 1;
        int mm  = m[g];
        s_ah  += (lz_i64)(ah  * mm);
        s_als += (lz_i64)(als * mm);
        s_b   += (lz_i64)(b   * mm);
    }
    return ((s_ah + s_b) << 16) + s_als;
}

/* The SSE2 body itself, written here rather than in ops_sse2.c so it is
   testable against the three scalar derivations before it moves into a
   shipping file.

   Every operand is built with the value in the low 16 bits and zero in
   the high 16, which is what makes _mm_madd_epi16 return v*m per lane:
   it reads both halves as signed int16 and adds them, so a zero high
   half contributes nothing and the low half carries the correct
   two's-complement pattern even for negatives.

   Products widen to 64 bits every iteration rather than accumulating in
   int32. |ah * m| reaches 2^30, so two of them can overflow an int32
   lane - a periodic flush would need a period of one, which is what
   this already is. */
#include <emmintrin.h>
static lz_i64 mac_sse2(const int *a, const short *m, int n) {
    const __m128i lo16 = _mm_set1_epi32(0x0000FFFF);
    const __m128i one  = _mm_set1_epi32(1);
    const __m128i zero = _mm_setzero_si128();
    __m128i acc_hi = zero, acc_lo = zero;   /* 2 x int64 each */
    lz_i64 s_ah, s_als, s_b, tail_ah = 0, tail_als = 0, tail_b = 0;
    __m128i acc_b = zero;
    int g = 0;

    for (; g + 3 < n; g += 4) {
        __m128i av  = _mm_loadu_si128((const __m128i *)(a + g));
        __m128i mv  = _mm_loadl_epi64((const __m128i *)(m + g));
        __m128i m32 = _mm_and_si128(_mm_unpacklo_epi16(mv, zero), lo16);
        __m128i ah  = _mm_and_si128(_mm_srai_epi32(av, 16), lo16);
        __m128i als = _mm_and_si128(av, lo16);
        __m128i bb  = _mm_and_si128(_mm_srli_epi32(av, 15), one);
        __m128i p_ah  = _mm_madd_epi16(ah,  m32);
        __m128i p_als = _mm_madd_epi16(als, m32);
        __m128i p_b   = _mm_madd_epi16(bb,  m32);
        /* Widen each int32 lane to int64 by sign-extension, then add. */
        acc_hi = _mm_add_epi64(acc_hi,
                 _mm_unpacklo_epi32(p_ah, _mm_srai_epi32(p_ah, 31)));
        acc_hi = _mm_add_epi64(acc_hi,
                 _mm_unpackhi_epi32(p_ah, _mm_srai_epi32(p_ah, 31)));
        acc_lo = _mm_add_epi64(acc_lo,
                 _mm_unpacklo_epi32(p_als, _mm_srai_epi32(p_als, 31)));
        acc_lo = _mm_add_epi64(acc_lo,
                 _mm_unpackhi_epi32(p_als, _mm_srai_epi32(p_als, 31)));
        acc_b  = _mm_add_epi64(acc_b,
                 _mm_unpacklo_epi32(p_b, _mm_srai_epi32(p_b, 31)));
        acc_b  = _mm_add_epi64(acc_b,
                 _mm_unpackhi_epi32(p_b, _mm_srai_epi32(p_b, 31)));
    }
    {
        lz_i64 t[2];
        _mm_storeu_si128((__m128i *)t, acc_hi);  s_ah  = t[0] + t[1];
        _mm_storeu_si128((__m128i *)t, acc_lo);  s_als = t[0] + t[1];
        _mm_storeu_si128((__m128i *)t, acc_b);   s_b   = t[0] + t[1];
    }
    for (; g < n; g++) {   /* scalar tail, same three terms */
        int ah  = a[g] >> 16;
        int als = (short)((unsigned)a[g] & 0xFFFFu);
        int b   = (a[g] >> 15) & 1;
        tail_ah  += (lz_i64)(ah  * m[g]);
        tail_als += (lz_i64)(als * m[g]);
        tail_b   += (lz_i64)(b   * m[g]);
    }
    return (((s_ah + tail_ah) + (s_b + tail_b)) << 16) + (s_als + tail_als);
}

static unsigned rng_state = SEED;
static unsigned rng_next(void) {
    /* xorshift32 - the values only have to be reproducible, not good. */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

int main(void) {
    static int a[512];
    static short m[512];
    int trial, g, bad = 0;
    long checked = 0;

    /* Edge cases first, since these are where a sign correction fails
       and a uniform random sweep is least likely to land. */
    {
        static const int ae[] = { 0, 1, -1, 32767, -32768, 65535, -65536,
                                  2147483647, -2147483647 - 1, 65536,
                                  -65535, 1 << 30, -(1 << 30) };
        static const short me[] = { 0, 1, -1, 32767, -32768, 256, -256 };
        int i, j;
        for (i = 0; i < (int)(sizeof ae / sizeof ae[0]); i++) {
            for (j = 0; j < (int)(sizeof me / sizeof me[0]); j++) {
                lz_i64 d, s, c, v;
                int k;
                /* Filled to 8, not 1: at n=1 the SSE2 body runs only its
                   scalar tail, so a vector-path bug on exactly these
                   values would go unseen - which is what the first
                   version of this block did. */
                for (k = 0; k < 8; k++) { a[k] = ae[i]; m[k] = me[j]; }
                d = mac_direct(0, a, m, 8);
                s = mac_split(a, m, 8);
                c = mac_carry_free(a, m, 8);
                v = mac_sse2(a, m, 8);
                checked++;
                if (d != s || d != c || d != v) {
                    printf("MISMATCH a=%d m=%d direct=%ld split=%ld "
                           "cfree=%ld sse2=%ld\n", ae[i], (int)me[j],
                           (long)d, (long)s, (long)c, (long)v);
                    bad++;
                }
            }
        }
    }

    for (trial = 0; trial < 20000; trial++) {
        int n = 1 + (int)(rng_next() % 512u);
        lz_i64 d, s, c, v;
        for (g = 0; g < n; g++) {
            a[g] = (int)rng_next();
            m[g] = (short)(rng_next() & 0xFFFFu);
        }
        d = mac_direct(0, a, m, n);
        s = mac_split(a, m, n);
        c = mac_carry_free(a, m, n);
        v = mac_sse2(a, m, n);
        checked++;
        if (d != s || d != c || d != v) {
            printf("MISMATCH trial=%d n=%d direct=%ld split=%ld cfree=%ld sse2=%ld\n",
                   trial, n, (long)d, (long)s, (long)c, (long)v);
            if (++bad > 8) break;
        }
    }

    printf("epi_mac_oracle: %ld cases, %d mismatch(es)\n", checked, bad);
    if (bad) { printf("epi_mac_oracle: FAIL\n"); return 1; }
    printf("epi_mac_oracle: PASS\n");
    return 0;
}
