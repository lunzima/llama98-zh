/* Exhaustive check for lz_quantize_q8_i16's integer element loop
 * (src/ops.c, LZ_Q8I16_INT).
 *
 * The loop replaces `q8_round((float)grp[k] * (127.0f/(float)tam))` with
 * a per-group reciprocal, a multiply-shift and one unconditional
 * correction, all in int32. What has to be proved is that the result is
 * EXACTLY the correctly-rounded quotient 127*|grp[k]|/tam - a
 * multiply-shift that is merely close would move an int8 level on some
 * input nobody sampled.
 *
 * The domain is small enough to enumerate, so it is enumerated rather
 * than sampled (same technique as .prof/verify_epi_align.c and
 * .prof/m5sig.c). tam is the group's own integer amax over `short`
 * inputs, so 0 <= tam <= 32768 - 32768 is reachable, |SHRT_MIN| is not
 * 32767 - and |grp[k]| <= tam by construction. Every (tam, m) pair in
 * that triangle is checked, twice: once with the live reciprocal and
 * once with the zero one.
 *
 * FOUR assertions, deliberately answering four different questions, each
 * with a one-line mutation that reddens IT and no other (counts below
 * are what those mutations actually produced):
 *   A1  the loop == the integer reference, over the whole domain.
 *       Mutation: drop `+ LZ_Q8I16_HALF`. 268,097,742 mismatches.
 *   A2  the integer reference == a long double reference. A1 alone is
 *       worth nothing if its reference shares the loop's mistake, and
 *       these two share no arithmetic: 127*m and tam are exact in a
 *       64-bit mantissa, an exact tie makes the quotient q-0.5 with
 *       q <= 127 which is itself exactly representable, and everywhere
 *       else long double's 1e-19 relative error is nowhere near the
 *       1/(2*32768) = 1.5e-5 minimum distance to a decision boundary.
 *       Mutation: ref_ld rounds with floorl(v). 268,427,329.
 *   A3  the correction is LOAD-BEARING. The same loop with it deleted is
 *       run alongside and must disagree with the reference somewhere; a
 *       correction that never changes an answer is decoration and A1
 *       would pass with it removed.
 *       Mutation: give q8i16_q_nocorr the correction back. 0, i.e. red.
 *   A4  the rcp == 0 selector - the MIN_SCALE / zero-amax path, which
 *       real data does not reach - zeroes every element. This is the
 *       one branch the runtime gates cannot see at all.
 *       Mutation: `(zero && tam == 0) ?`, the plausible defect of zeroing
 *       only the empty group. 534,790,014 non-zero results.
 *
 * One EQUIVALENT mutant, worth recording so the next reader does not
 * count it as a hole: LZ_Q8I16_SH 23 -> 22 leaves A1 green. That is
 * correct behaviour, not a blind spot - the correction repairs any
 * one-sided error below 1, and at SH 22 the error ceiling is tam/2^22 <=
 * 1/128. SH is chosen for the int32 headroom, not for exactness.
 *
 * Printed but NOT asserted, because A1 already implies them: the range
 * of q (the caller's clamp is defense, not arithmetic) and the observed
 * maxima of the three intermediates against their derived int32 bounds.
 *
 * Build: gcc -O2 -std=c99 -o verify_q8i16_int verify_q8i16_int.c -lm */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ---- verbatim copy of the arithmetic under test (src/ops.c) ---------- */

#define LZ_Q8I16_SH   23
#define LZ_Q8I16_HALF (1 << (LZ_Q8I16_SH - 1))
#define LZ_Q8I16_NUM  (127 << LZ_Q8I16_SH)   /* 1,065,353,216 */

/* per group; `zero` is the shipped `(inv == 0.0f) ?` selector */
static int32_t q8i16_rcp(int32_t tam, int zero) {
    return zero ? 0 : (int32_t)((LZ_Q8I16_NUM + tam - 1) / tam);
}

/* per element, magnitude domain */
static int32_t q8i16_q(int32_t m, int32_t tam, int32_t rcp) {
    int32_t q = (m * rcp + LZ_Q8I16_HALF) >> LZ_Q8I16_SH;
    q -= (254 * m + tam < 2 * tam * q);
    return q;
}

/* the same loop with the correction removed - A3's subject */
static int32_t q8i16_q_nocorr(int32_t m, int32_t tam, int32_t rcp) {
    (void)tam;
    return (m * rcp + LZ_Q8I16_HALF) >> LZ_Q8I16_SH;
}

/* ---- references ------------------------------------------------------ */

/* floor(127*m/tam + 1/2) in integers: half-away-from-zero on a
   non-negative value, both operands positive so C's division floors. */
static int32_t ref_int(int32_t m, int32_t tam) {
    return (254 * m + tam) / (2 * tam);
}

static int32_t ref_ld(int32_t m, int32_t tam) {
    long double v = (long double)127 * (long double)m / (long double)tam;
    return (int32_t)floorl(v + 0.5L);
}

int main(void) {
    long long total = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, printed = 0;
    long long corrections = 0;
    long long max_prod = 0, max_lhs = 0, max_rhs = 0;
    int32_t qmin = 127, qmax = 0;
    int32_t tam;

    /* tam == 0 (an all-zero group) reaches the zero selector only; the
       live one would divide by it, which is why inv != 0 implies
       tam >= 1 in the caller. */
    if (q8i16_q(0, 0, q8i16_rcp(0, 1)) != 0) a4++;
    total++;

    for (tam = 1; tam <= 32768; tam++) {
        int32_t rcp = q8i16_rcp(tam, 0);
        int32_t rcp0 = q8i16_rcp(tam, 1);
        int32_t m;
        for (m = 0; m <= tam; m++) {
            int32_t prod = m * rcp + LZ_Q8I16_HALF;
            int32_t q0 = prod >> LZ_Q8I16_SH;
            int32_t lhs = 254 * m + tam, rhs = 2 * tam * q0;
            int32_t got = q8i16_q(m, tam, rcp);
            int32_t wi = ref_int(m, tam);
            int32_t wl = ref_ld(m, tam);

            total++;
            if (prod > max_prod) max_prod = prod;
            if (lhs > max_lhs) max_lhs = lhs;
            if (rhs > max_rhs) max_rhs = rhs;
            if (got != q0) corrections++;
            if (got < qmin) qmin = got;
            if (got > qmax) qmax = got;

            if (got != wi) {
                a1++;
                if (printed++ < 20)
                    printf("A1 tam=%d m=%d got=%d want=%d\n",
                           (int)tam, (int)m, (int)got, (int)wi);
            }
            if (wi != wl) {
                a2++;
                if (printed++ < 20)
                    printf("A2 tam=%d m=%d int=%d ld=%d\n",
                           (int)tam, (int)m, (int)wi, (int)wl);
            }
            if (q8i16_q_nocorr(m, tam, rcp) != wi) a3++;
            if (q8i16_q(m, tam, rcp0) != 0) {
                a4++;
                if (printed++ < 20)
                    printf("A4 tam=%d m=%d got=%d want=0\n", (int)tam,
                           (int)m, (int)q8i16_q(m, tam, rcp0));
            }
        }
    }

    printf("cases                  %lld (x2: live and zero reciprocal)\n",
           total);
    printf("A1 loop vs int ref     %lld mismatches (want 0)\n", a1);
    printf("A2 int ref vs ld ref   %lld mismatches (want 0)\n", a2);
    printf("A3 correction removed  %lld mismatches (want > 0)\n", a3);
    printf("A4 zero reciprocal     %lld non-zero results (want 0)\n", a4);
    printf("-- measured, not asserted --\n");
    printf("q range                [%d, %d]  (caller clamps at 127)\n",
           (int)qmin, (int)qmax);
    printf("corrections fired      %lld  %.4f%%\n", corrections,
           100.0 * (double)corrections / (double)total);
    printf("max m*rcp+half         %lld  (bound %d, int32 max 2147483647)\n",
           max_prod, LZ_Q8I16_NUM + 32768 + LZ_Q8I16_HALF);
    printf("max 254*m+tam          %lld  (bound %d)\n", max_lhs,
           254 * 32768 + 32768);
    printf("max 2*tam*q            %lld  (bound %d)\n", max_rhs,
           2 * 32768 * 127);

    if (a1 || a2 || a4 || a3 == 0) {
        printf("verify_q8i16_int: FAIL\n");
        return 1;
    }
    printf("verify_q8i16_int: PASS\n");
    return 0;
}
