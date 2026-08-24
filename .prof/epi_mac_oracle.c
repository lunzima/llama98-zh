/* The oracle any lz_epi_mac_i16 kernel has to match.
 *
 * The operator accumulates int32 x int16 -> int64:
 *
 *     for (g = 0; g < n; g++) s += (lz_i64)a[g] * (lz_i64)m[g];
 *
 * On x86 there is no signed 32x32 multiply below SSE4.1: SSE2 reaches
 * this with pmuludq plus sign correction and MMX by splitting the int32
 * into 16-bit halves, both of which get the sign wrong quietly rather
 * than loudly. A wrong correction returns a plausible number, so the
 * kernel needed a predicate written before it existed, not after.
 *
 * This is that predicate. It is deliberately not a comparison against
 * the shipping C body: that body IS one of the things under test once a
 * kernel lands beside it. The reference here recomputes the sum a
 * second way - 32x32 partial products in unsigned arithmetic with the
 * signs handled explicitly - so agreement means two independent
 * derivations agree, not that one function was called twice.
 *
 * Builds, all five verified to run:
 *
 *   gcc -O2 -std=c99 -Wall -Wextra -msse2 -Isrc \
 *       .prof/epi_mac_oracle.c -o /tmp/epi_oracle
 *   ... -DEPI_MUT_SIGNED_LOW ...             must FAIL (25 mismatches)
 *   ... -DEPI_LINK_SHIPPED -DLZ_SSE2_TU=1 .prof/epi_mac_oracle.c \
 *       src/ops_sse2.c src/ops_sched.c src/cpucheck.c src/err.c
 *   bash .prof/epi_watcom_gate.sh            the Watcom SSE2 pragma
 *   bash .prof/epi_arm_gate.sh               the ARM kernel, under QEMU
 *
 * The last three carry the SAME case count as the first, which is the
 * point: gcc intrinsics, a Watcom #pragma aux and ARMv5TE assembly are
 * held to one predicate rather than to three that resemble each other.
 *
 * The three extra TUs in the link modes are ops_sse2.c's own references
 * (g_pair, lz_cpuid1_edx, lz_err_fmt) reached from functions this
 * harness never calls. They are listed rather than stubbed because a
 * stub is a second definition of engine state, and mingw's .refptr
 * indirection defeats -Wl,--gc-sections for the data ones anyway.
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

/* The fourth derivation is the KERNEL UNDER TEST, and which one that is
   depends on how this file is built:

     (default)         the SSE2 body copied below, so the harness builds
                       and runs standalone on any x86 host
     -DEPI_LINK_SHIPPED  lz_epi_mac_i16_sse2, from src/ops_sse2.c
     -DEPI_ARM         lz_epi_mac_i16_arm_asm, from src/ops_arm.c,
                       cross-built and run under QEMU

   The copy is kept for the reason the link modes exist at all: a copy
   that passes proves the FORMULA, and only a link mode proves the
   shipped function. Verifying a copy and calling the artifact verified
   is a mistake this tree has made before. */
#if defined(EPI_ARM)
#include "ops_arm.h"
#define KERNEL_NAME "arm"
/* ARMv5TE needs none of the 16-bit splitting the header describes:
   SMLAL is signed 32x32 with a 64-bit accumulate, which is this
   operator's shape exactly. What is being checked here is therefore not
   a sign correction but the operand plumbing around it - the LDMIA
   stride, the LDRSH sign extension, and the hi:lo recombination, where
   an unsigned widening of the low word is the one that fails only for
   half the inputs.
   Build: .prof/epi_arm_gate.sh */
static lz_i64 mac_kernel(const int *a, const short *m, int n) {
    return lz_epi_mac_i16_arm_asm((const int32_t *)a, (const int16_t *)m, n);
}
#elif defined(EPI_LINK_SHIPPED)
#include "ops_sse2.h"
#define KERNEL_NAME "sse2"
/* Build: gcc -O2 -std=c99 -Wall -Wextra -msse2 -Isrc -DEPI_LINK_SHIPPED \
          -DLZ_SSE2_TU=1 .prof/epi_mac_oracle.c src/ops_sse2.c -o /tmp/x */
static lz_i64 mac_kernel(const int *a, const short *m, int n) {
    return lz_epi_mac_i16_sse2((const int32_t *)a, (const int16_t *)m, n);
}
#else
#define KERNEL_NAME "sse2"
/* The SSE2 body itself, written here rather than in ops_sse2.c so it was
   testable against the three scalar derivations before it moved into a
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
static lz_i64 mac_kernel(const int *a, const short *m, int n) {
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
#endif /* kernel selection */

/* NOT %ld. This harness cross-builds to a 32-bit target, where long is
   32 bits and every reported value would be the low half of the one
   actually compared - two sums that differ only above bit 31 then print
   identically, which is exactly the failure the carry terms exist for.
   The comparisons were always on lz_i64; only the report was lossy. */
#define P64 "%" LZ_PRI64 "d"

static unsigned rng_state = SEED;
static unsigned rng_next(void) {
    /* xorshift32 - the values only have to be reproducible, not good. */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

/* argv is arm_mutgate.sh's calling convention, not a feature of this
   harness: [leaf] [seed] [divisor]. The divisor shortens the random
   sweep for a mutation run, which only has to reach the first
   mismatch. Seed 0 means "the fixed one", so a caller that passes no
   seed and one that passes 0 sample the same inputs. */
int main(int argc, char **argv) {
    static int a[512];
    static short m[512];
    int g, bad = 0;
    long trial, checked = 0;
    long trials = 20000;

    if (argc > 2) rng_state = (unsigned)strtoul(argv[2], NULL, 0);
    if (!rng_state) rng_state = SEED;
    if (argc > 3) {
        long d = strtol(argv[3], NULL, 0);
        if (d > 1) trials = (trials / d) < 1 ? 1 : (trials / d);
    }

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
                v = mac_kernel(a, m, 8);
                checked++;
                if (d != s || d != c || d != v) {
                    printf("MISMATCH a=%d m=%d direct=" P64 " split=" P64
                           " cfree=" P64 " " KERNEL_NAME "=" P64 "\n",
                           ae[i], (int)me[j], d, s, c, v);
                    bad++;
                }
            }
        }
    }

    /* n around the loop's floor, swept rather than sampled. The random
       block below draws n from 1..512, so it reaches any particular
       small n about once in five hundred trials - and small n is where
       a vector body is nothing BUT its entry guard and its scalar tail.
       At or below zero the answer is 0 by the C body's empty loop, and
       a guard that lets a 4-wide loop start there walks backwards. */
    {
        static const int nz[] = { -7, -1, 0 };
        int i, nn;
        for (i = 0; i < (int)(sizeof nz / sizeof nz[0]); i++) {
            lz_i64 v = mac_kernel(a, m, nz[i]);
            checked++;
            if (v != 0) {
                printf("MISMATCH n=%d " KERNEL_NAME "=" P64
                       ", expected 0\n", nz[i], v);
                bad++;
            }
        }
        for (nn = 1; nn <= 12; nn++) {
            lz_i64 d, s, c, v;
            for (g = 0; g < nn; g++) {
                a[g] = (int)rng_next();
                m[g] = (short)(rng_next() & 0xFFFFu);
            }
            d = mac_direct(0, a, m, nn);
            s = mac_split(a, m, nn);
            c = mac_carry_free(a, m, nn);
            v = mac_kernel(a, m, nn);
            checked++;
            if (d != s || d != c || d != v) {
                printf("MISMATCH n=%d direct=" P64 " split=" P64
                       " cfree=" P64 " " KERNEL_NAME "=" P64 "\n",
                       nn, d, s, c, v);
                bad++;
            }
        }
    }

    for (trial = 0; trial < trials; trial++) {
        int n = 1 + (int)(rng_next() % 512u);
        lz_i64 d, s, c, v;
        for (g = 0; g < n; g++) {
            a[g] = (int)rng_next();
            m[g] = (short)(rng_next() & 0xFFFFu);
        }
        d = mac_direct(0, a, m, n);
        s = mac_split(a, m, n);
        c = mac_carry_free(a, m, n);
        v = mac_kernel(a, m, n);
        checked++;
        if (d != s || d != c || d != v) {
            printf("MISMATCH trial=%ld n=%d direct=" P64 " split=" P64
                   " cfree=" P64 " " KERNEL_NAME "=" P64 "\n",
                   trial, n, d, s, c, v);
            if (++bad > 8) break;
        }
    }

    printf("epi_mac_oracle: %ld cases, %d mismatch(es)\n", checked, bad);
    if (bad) { printf("epi_mac_oracle: FAIL\n"); return 1; }
    printf("epi_mac_oracle: PASS\n");
    return 0;
}
