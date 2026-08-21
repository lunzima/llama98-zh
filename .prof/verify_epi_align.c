/* Standalone check for epi_align_i16 (src/ops.c, int-pipeline project
 * batch 1 #1 - the epi_fixed exponent alignment primitive).
 *
 * Reimplements the function under test verbatim (same technique as
 * verify_conv_fold3.c) and an INDEPENDENT reference built on long
 * double: on x87 (and every mainstream x86 ABI's `long double`, 80-bit
 * extended, 64-bit mantissa) an int64 converts to long double exactly,
 * and multiplying by an exact power of two only moves the exponent
 * field, so `(long double)s * powl(2, shift)` has no rounding of its
 * own to disagree with epi_align_i16's integer arithmetic about. The
 * reference then rounds with roundl() (half-away-from-zero, matching
 * this file's chosen convention - see epi_align_i16's real home for
 * why that is NOT q8_round's half-to-even) and clamps the same way.
 *
 * Domain swept: mag over a dense band, an every-power-of-two geometric
 * sweep out to the full int64 range (both s>=0 and s<0), the +-1
 * neighbours of each power of two (tie-boundary and shift-boundary
 * cases), e over its real range [-200, 0] (lz_epi_prep/epi_pack_act's
 * own 100-cap on each of xe/sexp), target_e over 0..40 (LZ_CONV_ES is
 * 9; the wider range exercises the shift>=0 clamp this function's
 * caller does not currently reach with target_e=9 alone, since
 * shift = e - 8 + target_e <= target_e - 8 given e <= 0 always), and
 * bound over the values lz_conv_accum_bound can actually return.
 *
 * Build: gcc -O2 -std=c99 -o verify_epi_align verify_epi_align.c -lm */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ---- verbatim copy of the function under test (src/ops.c) ------------ */
static int32_t epi_align_i16(long long s, int e, int target_e, int bound) {
    unsigned long long umag;
    long long r;
    int neg, shift;
    if (s == 0) return 0;
    neg = (s < 0);
    /* Portable, UB-free magnitude as unsigned: converting a negative
       signed value to unsigned is well-defined (s mod 2^64, i.e. the
       two's-complement bit pattern reinterpreted), and negating THAT in
       unsigned arithmetic is also well-defined wraparound - together
       they give the exact magnitude even for s == LLONG_MIN, which
       plain `-s` cannot: -LLONG_MIN is not representable in long long
       and is undefined behaviour (caught by this probe: 248 mismatches
       before this fix, all at s == LLONG_MIN). */
    umag = neg ? (unsigned long long)0 - (unsigned long long)s
               : (unsigned long long)s;
    shift = e - 8 + target_e;
    if (shift > 32) shift = 32;    /* umag<<32 already dwarfs any real bound */
    if (shift <= -64) {
        /* value = umag * 2^shift <= 2^63 * 2^shift <= 0.5 (umag's own
           ceiling, LLONG_MIN's magnitude, is 2^63), with equality only
           at shift == -64 AND umag == 2^63 - the single input where
           round-half-away-from-zero rounds an exact 0.5 up to 1 instead
           of down to 0. rs = -shift >= 64 cannot go through the shift
           branch below at all: `>>rs`/`<<(rs-1)` on a 64-bit type is
           undefined once rs reaches 64, so this case is handled by the
           closed form above rather than by widening the integer type. */
        r = (shift == -64 && umag == ((unsigned long long)1 << 63)) ? 1 : 0;
    } else if (shift >= 0) {
        unsigned long long capped = (unsigned long long)bound >> shift; /* shift<=32: safe */
        if (umag > capped + 1)
            r = (long long)bound + 1;          /* clamps below regardless */
        else
            r = (long long)(umag << shift);    /* <= (capped+1)<<shift <=
                                                   bound+2^32: far inside
                                                   unsigned 64 bits */
    } else {
        int rs = -shift;                                     /* 1..63 */
        unsigned long long half = (unsigned long long)1 << (rs - 1);
        r = (long long)((umag + half) >> rs);  /* unsigned add: no UB, and
                                                   provably < 2^64 (umag
                                                   <= 2^63, half <= 2^62) */
    }
    if (r > bound) r = bound;
    return (int32_t)(neg ? -r : r);
}

/* ---- independent reference --------------------------------------------
 * Deliberately NOT sharing one line of arithmetic with the function
 * under test: a bug in a shared helper would pass both. */
static int32_t reference(long long s, int e, int target_e, int bound) {
    int shift = e - 8 + target_e;
    long double v = (long double)s * powl(2.0L, (long double)shift);
    long double rr = roundl(v);
    if (rr >  (long double)bound) rr =  (long double)bound;
    if (rr < -(long double)bound) rr = -(long double)bound;
    return (int32_t)rr;
}

static long bad = 0, total = 0;

static void check(long long s, int e, int target_e, int bound) {
    int32_t got = epi_align_i16(s, e, target_e, bound);
    int32_t want = reference(s, e, target_e, bound);
    total++;
    if (got != want) {
        bad++;
        if (bad <= 30)
            printf("MISMATCH s=%lld e=%d target_e=%d bound=%d got=%d want=%d\n",
                   s, e, target_e, bound, got, want);
    }
}

int main(void) {
    static const int bounds[] = { 4, 16, 64, 256, 1024, 4096, 16384, 32767 };
    int bi, ti, ei;

    /* s == 0 and the immediate neighbours, every (e, target_e, bound). */
    for (bi = 0; bi < 8; bi++)
        for (ei = -200; ei <= 0; ei += 4)
            for (ti = 0; ti <= 40; ti += 2) {
                check(0, ei, ti, bounds[bi]);
                check(1, ei, ti, bounds[bi]);
                check(-1, ei, ti, bounds[bi]);
            }

    /* Geometric sweep of |s| out to the full int64 range, both signs,
       +-1 neighbours (shift/tie boundaries), at a coarser (e, target_e,
       bound) grid so the case count stays reasonable. */
    for (bi = 0; bi < 8; bi++)
        for (ei = -200; ei <= 0; ei += 8)
            for (ti = 0; ti <= 40; ti += 4) {
                long long g;
                for (g = 1; g > 0 && g < (1LL << 62); g <<= 1) {
                    check(g, ei, ti, bounds[bi]);
                    check(-g, ei, ti, bounds[bi]);
                    check(g + 1, ei, ti, bounds[bi]);
                    check(-(g + 1), ei, ti, bounds[bi]);
                    check(g - 1, ei, ti, bounds[bi]);
                    check(-(g - 1), ei, ti, bounds[bi]);
                }
                check(0x7FFFFFFFFFFFFFFFLL, ei, ti, bounds[bi]);
                check(-0x7FFFFFFFFFFFFFFFLL - 1, ei, ti, bounds[bi]);
            }

    /* Dense band near the values a real conv input actually produces
       (LZ_CONV_ES=9, .prof/wconv.c's own measured peak 9.09e8, well
       inside the int32 accumulator conv itself uses - epi_align_i16's
       *input* is a much larger int64, so this band is deliberately
       scaled up, but it is the shape a live wire would hit most often). */
    for (bi = 0; bi < 8; bi++) {
        long long si;
        for (si = -2000000; si <= 2000000; si += 977)
            for (ei = -60; ei <= 0; ei += 3)
                check(si, ei, 9, bounds[bi]);
    }

    /* Focused sweep on the shift==-64 boundary itself (the one place
       the closed-form branch above replaces the shift arithmetic):
       every target_e/e combination that lands exactly on shift in
       -66..-62, and umag walking densely through the top of its range
       (2^62..2^63) where the 0.5 tie actually lives. */
    for (bi = 0; bi < 8; bi++)
        for (ti = 0; ti <= 64; ti++) {
            int want_shift;
            for (want_shift = -66; want_shift <= -62; want_shift++) {
                long long delta;
                ei = want_shift + 8 - ti;
                if (ei > 0 || ei < -400) continue;
                for (delta = -3; delta <= 3; delta++) {
                    long long base = (1LL << 62);
                    check(base + delta, ei, ti, bounds[bi]);
                    check(-(base + delta), ei, ti, bounds[bi]);
                    /* LLONG_MAX/LLONG_MIN's own -3..+3 neighbourhood,
                       built with unsigned arithmetic so the harness
                       itself never negates LLONG_MIN (the exact UB
                       epi_align_i16 had to be fixed for). */
                    {
                        unsigned long long umax = 0x7FFFFFFFFFFFFFFFULL;
                        unsigned long long near_max = umax - (unsigned long long)(3 - delta);
                        unsigned long long near_min = umax + 1ULL + (unsigned long long)(delta + 3);
                        check((long long)near_max, ei, ti, bounds[bi]);
                        check((long long)near_min, ei, ti, bounds[bi]);
                    }
                }
                check(0x7FFFFFFFFFFFFFFFLL, ei, ti, bounds[bi]);
                check(-0x7FFFFFFFFFFFFFFFLL - 1, ei, ti, bounds[bi]);
            }
        }

    printf("total=%ld bad=%ld\n", total, bad);
    return bad != 0;
}
