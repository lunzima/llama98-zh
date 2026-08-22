/* Exhaustive check for lz_swiglu_q15_i16's integer arithmetic
 * (src/ops.c, int-pipeline milestone 8).
 *
 *   sv = epi_shr_half(p * sig, 14)          stage 1, int32 product
 *   q  = epi_align_i16(sv * g, e, eo, 32767) stage 2 product, stage 3 exit
 *
 * The full (p, sig, g) cross product is 1.4e14 and cannot be enumerated,
 * but the chain FACTORS: each stage's output domain is exactly the next
 * stage's input domain, and each stage's own domain is small enough to
 * enumerate. So every stage is checked over ALL of its inputs, and the
 * composite follows by composition rather than by sampling.
 *
 *   A1  every (p, sig) pair: the int32 product p*sig is the exact
 *       product (no overflow), and |sv| <= 65534. A1 also requires that
 *       65534 is ATTAINED, so the bound cannot be quietly slack - and
 *       that half of it earned its keep immediately, rejecting the 65536
 *       this file and src/ops.c first carried (32767*32768 rounds to
 *       65534.5, not 65536.5). |sv| is twice int16's range either way,
 *       which is why stage 2 cannot be an int16 multiply.
 *       Mutation: shift 14 -> 13 in the stage-1 narrowing. |sv| reaches
 *       131068 and A1 reports the bound broken.
 *   A2  every product MAGNITUDE in [0, 2^30]: epi_shr_half(m, 14)
 *       equals an integer reference AND a long double reference. Two
 *       references sharing no arithmetic, for the reason
 *       verify_q8i16_int.c's A2 gives - an assertion against a reference
 *       that shares the defect is worth nothing.
 *       Mutation: drop the `+ (1 << (rs-1))` bias. 1,073,676,288 misses.
 *   A3  every (|sv|, |g|) pair with |sv| <= 65534, |g| <= 32767: the
 *       int32 product sv*g is the exact product. This is the assertion
 *       that keeps stage 2 out of int64, and it sits at the edge -
 *       65534 * 32767 = 2,147,352,578 against INT32_MAX 2,147,483,647,
 *       131,069 to spare. At a stage-1 shift of 13 the same product is
 *       4,294,705,156 and does not fit, which is why 14 is the largest
 *       usable narrowing rather than a taste.
 *       Mutation: enumerate to SVMAX + 8. 131,068 overflows.
 *       That mutation's first run reddened NOTHING, which is how this
 *       row's original form was found to be unobservable - see the
 *       comment on the loop.
 *   A4  every reachable stage-2 accumulator magnitude, i.e. all of
 *       [0, 2,147,418,112]: epi_align_i16 at the shipping shift equals
 *       an integer reference, and |q| <= 32767 always.
 *       Mutation: reference truncates instead of rounding half away.
 *   A5  sign symmetry of both shifts: shr(-x) == -shr(x) and the exit
 *       likewise. A C right shift of a negative int is only
 *       implementation-defined, and this chain must not depend on it.
 *       Mutation: replace the magnitude form with `x >> rs`.
 *   A6  end-to-end cross-check on a (p, sig, g) sweep: the two-stage
 *       result is within 3 output ulps of a single-rounding long double
 *       evaluation of p*sig*g*2^-(15+ep+eg) at eo. 3 is DERIVED, not
 *       fitted: stage 1 contributes at most 0.5 ulp at exponent ep+1,
 *       which is 0.5*|g|/2^rs <= 2.0 output ulps at the shipping
 *       exponents, and stage 3 contributes at most 0.5. A6 is a
 *       cross-check on the composition, not the proof - A2 and A4 are.
 *       Mutation: swap the two operands of stage 1 (sig*g first, then
 *       p), which is a legal-looking reassociation with a different
 *       error and blows the ulp budget.
 *
 * Build: gcc -O2 -std=c99 -o verify_swiglu_i16 verify_swiglu_i16.c -lm
 * Usage: verify_swiglu_i16 [ep] [eg] [eo]     (default 11 12 11)
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* ---- verbatim copies of the primitives under test (src/ops.c) -------- */

static unsigned long long epi_shr_half_u(unsigned long long umag, int rs) {
    if (rs <= 0) return umag;
    if (rs >= 64)
        return (rs == 64 && umag == ((unsigned long long)1 << 63)) ? 1 : 0;
    return (umag + ((unsigned long long)1 << (rs - 1))) >> rs;
}

static long long epi_shr_half(long long s, int rs) {
    unsigned long long umag;
    int neg;
    if (rs <= 0 || s == 0) return s;
    neg = (s < 0);
    umag = neg ? (unsigned long long)0 - (unsigned long long)s
               : (unsigned long long)s;
    umag = epi_shr_half_u(umag, rs);
    return neg ? -(long long)umag : (long long)umag;
}

static int32_t epi_align_i16(long long s, int e, int target_e, int bound) {
    unsigned long long umag;
    long long r;
    int neg, shift;
    if (s == 0) return 0;
    neg = (s < 0);
    umag = neg ? (unsigned long long)0 - (unsigned long long)s
               : (unsigned long long)s;
    shift = e - 8 + target_e;
    if (shift > 32) shift = 32;
    if (shift >= 0) {
        unsigned long long capped = (unsigned long long)bound >> shift;
        if (umag > capped + 1)
            r = (long long)bound + 1;
        else
            r = (long long)(umag << shift);
    } else {
        r = (long long)epi_shr_half_u(umag, -shift);
    }
    if (r > bound) r = bound;
    return (int32_t)(neg ? -r : r);
}

/* The function under test, one element of it. */
static int32_t swiglu_one(int32_t pv, int32_t sig, int32_t gv, int e, int eo) {
    int32_t sv = (int32_t)epi_shr_half((long long)(pv * sig), 14);
    return epi_align_i16((long long)(sv * gv), e, eo, 32767);
}

/* ---- references sharing no arithmetic with the above ----------------- */

/* Round half away from zero, on the magnitude, by integer division. */
static long long ref_shr_int(long long v, int rs) {
    long long m = v < 0 ? -v : v;
    long long d = (long long)1 << rs;
    long long q = (m + d / 2) / d;
    return v < 0 ? -q : q;
}

/* Same, in long double. A 64-bit mantissa holds every value here
   exactly, so scalbnl introduces no rounding of its own and roundl is
   C99 half-away-from-zero. */
static long long ref_shr_ld(long long v, int rs) {
    long double x = scalbnl((long double)v, -rs);
    return (long long)roundl(x);
}

#define PMAX   32767
#define SIGMAX 32768
/* (32767*32768 + 2^13) >> 14 == 65534, not 65536: the rounded quotient is
   65534.5, which floors to 65534. A1 asserts the bound is ATTAINED, which
   is what caught the looser figure this constant first carried. */
#define SVMAX  65534
#define GMAX   32767
#define PRODMAX ((long long)PMAX * SIGMAX)          /* 1,073,709,056 */
#define ACCMAX  ((long long)SVMAX * GMAX)           /* 2,147,352,578 */

int main(int argc, char **argv) {
    int ep = (argc > 1) ? atoi(argv[1]) : 11;
    int eg = (argc > 2) ? atoi(argv[2]) : 12;
    int eo = (argc > 3) ? atoi(argv[3]) : 11;
    int e  = 8 - (ep + 1 + eg);
    int rs = ep + 1 + eg - eo;
    long long i, j;
    long long bad, svmax_seen, svmax_hits, accmax_seen, qmax_seen;
    int fail = 0;

    printf("exponents ep=%d eg=%d eo=%d  ->  epi_align_i16 e=%d, net shift rs=%d\n",
           ep, eg, eo, e, rs);
    printf("derived bounds: |p*sig| <= %lld, |sv| <= %d, |sv*g| <= %lld"
           " (INT32_MAX %d)\n", PRODMAX, SVMAX, ACCMAX, INT32_MAX);

    /* ---- A1: every (p, sig) pair --------------------------------- */
    bad = 0; svmax_seen = 0; svmax_hits = 0;
    for (i = -PMAX; i <= PMAX; i++) {
        int32_t pv = (int32_t)i;
        for (j = 0; j <= SIGMAX; j++) {
            long long exact = (long long)pv * j;
            unsigned wrapped = (unsigned)pv * (unsigned)j;
            long long sv;
            /* Same observability rule as A3's loop below: assert on the
               EXACT product, never on an int32 multiply compared to it. */
            if (exact > (long long)INT32_MAX ||
                exact < (long long)INT32_MIN) { bad++; continue; }
            if ((long long)(int32_t)wrapped != exact) { bad++; continue; }
            sv = epi_shr_half(exact, 14);
            if (sv < 0) sv = -sv;
            if (sv > svmax_seen) { svmax_seen = sv; svmax_hits = 0; }
            if (sv == SVMAX) svmax_hits++;
            if (sv > SVMAX) bad++;
        }
    }
    /* Attainment is part of THIS row's verdict, not a note under it: a
       row that prints PASS while a line below it says FAIL is a row that
       reads green in any summary anyone actually greps. */
    printf("A1  (p,sig) pairs %lld   int32-exact + |sv| <= %d, attained : %s"
           "   (max |sv| %lld, attained %lld times)\n",
           (long long)(2 * PMAX + 1) * (SIGMAX + 1), SVMAX,
           (bad || svmax_seen != SVMAX) ? "FAIL" : "PASS", svmax_seen,
           svmax_hits);
    if (bad) { printf("    %lld violations\n", bad); fail = 1; }
    if (svmax_seen != SVMAX) {
        printf("    bound %d not attained - it is slack, not tight\n", SVMAX);
        fail = 1;
    }

    /* ---- A2: every product magnitude, two references -------------- */
    bad = 0;
    for (i = 0; i <= PRODMAX; i++) {
        long long a = epi_shr_half(i, 14);
        if (a != ref_shr_int(i, 14) || a != ref_shr_ld(i, 14)) bad++;
    }
    printf("A2  magnitudes 0..%lld   shr_half == int ref == long double ref : %s\n",
           PRODMAX, bad ? "FAIL" : "PASS");
    if (bad) { printf("    %lld mismatches\n", bad); fail = 1; }

    /* ---- A3: every (|sv|, |g|) pair ------------------------------- */
    bad = 0; accmax_seen = 0;
    /* The claim is that the EXACT product fits int32, tested on the
       exact product - not on an int32 multiply compared against it. That
       first form is unobservable: signed overflow is undefined, and gcc
       -O2 folds `(long long)(int32_t)(a*b) != a*b` to false, so the row
       stayed green under a mutation that overflowed by 131,067. The
       unsigned product is the well-defined companion check that the
       32-bit multiply the engine performs carries the same bits. */
    for (i = 0; i <= SVMAX; i++)
        for (j = 0; j <= GMAX; j++) {
            long long exact = i * j;
            unsigned wrapped = (unsigned)i * (unsigned)j;
            if (exact > (long long)INT32_MAX) bad++;
            else if ((long long)wrapped != exact) bad++;
            if (exact > accmax_seen) accmax_seen = exact;
        }
    printf("A3  (|sv|,|g|) pairs %lld   int32-exact : %s   (max %lld, headroom %lld)\n",
           (long long)(SVMAX + 1) * (GMAX + 1), bad ? "FAIL" : "PASS",
           accmax_seen, (long long)INT32_MAX - accmax_seen);
    if (bad) { printf("    %lld overflows\n", bad); fail = 1; }

    /* ---- A4: every reachable accumulator magnitude ---------------- */
    bad = 0; qmax_seen = 0;
    for (i = 0; i <= ACCMAX; i++) {
        long long r = ref_shr_int(i, rs);
        int32_t q;
        if (r > 32767) r = 32767;
        q = epi_align_i16(i, e, eo, 32767);
        if (q != (int32_t)r) bad++;
        if (q > qmax_seen) qmax_seen = q;
    }
    printf("A4  accumulators 0..%lld   epi_align_i16 == int ref, |q| <= 32767 : %s"
           "   (max q %lld)\n", ACCMAX, bad ? "FAIL" : "PASS", qmax_seen);
    if (bad) { printf("    %lld mismatches\n", bad); fail = 1; }

    /* ---- A5: sign symmetry of both shifts ------------------------- */
    bad = 0;
    for (i = 1; i <= PRODMAX; i += 7919) {
        if (epi_shr_half(-i, 14) != -epi_shr_half(i, 14)) bad++;
        if (epi_align_i16(-i, e, eo, 32767) != -epi_align_i16(i, e, eo, 32767))
            bad++;
    }
    for (i = 0; i <= 1000000; i++) {
        if (epi_shr_half(-i, 14) != -epi_shr_half(i, 14)) bad++;
        if (epi_align_i16(-i, e, eo, 32767) != -epi_align_i16(i, e, eo, 32767))
            bad++;
    }
    printf("A5  sign symmetry of shr_half and the exit : %s\n",
           bad ? "FAIL" : "PASS");
    if (bad) { printf("    %lld asymmetries\n", bad); fail = 1; }

    /* ---- A6: end-to-end ulp budget on a (p, sig, g) sweep --------- */
    {
        long long n = 0, worst = 0, nclamp = 0;
        double sumsq = 0.0;
        for (i = -PMAX; i <= PMAX; i += 37) {
            int32_t pv = (int32_t)i;
            for (j = 0; j <= SIGMAX; j += 521) {
                int32_t sig = (int32_t)j;
                int32_t gv;
                for (gv = -GMAX; gv <= GMAX; gv += 1013) {
                    long double exact =
                        scalbnl((long double)pv * sig * gv, -(15 + ep + eg) + eo);
                    long long ref = (long long)roundl(exact);
                    int32_t got = swiglu_one(pv, sig, gv, e, eo);
                    long long d;
                    if (ref > 32767 || ref < -32767) { nclamp++; continue; }
                    d = (long long)got - ref;
                    if (d < 0) d = -d;
                    if (d > worst) worst = d;
                    sumsq += (double)d * (double)d;
                    n++;
                }
            }
        }
        printf("A6  sweep n=%lld (+%lld clamped, excluded)   max |two-stage -"
               " single-rounding| = %lld ulp, rms %.4f : %s\n",
               n, nclamp, worst, (n > 0) ? sqrt(sumsq / (double)n) : 0.0,
               (worst <= 3) ? "PASS" : "FAIL");
        if (worst > 3) fail = 1;
    }

    printf("%s\n", fail ? "VERIFY: FAIL" : "VERIFY: PASS");
    return fail;
}
