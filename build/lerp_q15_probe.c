/* Does the MMX Q15 interpolation agree with the C?
 *
 * out[k] = a[k] + (((b[k] - a[k]) * frac[k]) >> 15) - the tail of
 * sigmoid_q15 and lz_exp_fixed, with the table lookups already done by
 * the caller. A pure function of three int32 arrays, so it answers
 * without a model, without the engine linking and without a tier
 * selector.
 *
 * THE OPERAND WIDTHS ARE THE WHOLE RISK. b - a and frac go through a
 * 16-bit multiply while `a` does not, because a reaches 32768 exactly
 * and that is one past INT16_MAX. So the sweep drives `a` across that
 * edge deliberately rather than sampling it, and drives b - a negative
 * as well as positive - psrad is an arithmetic shift and a logical one
 * would pass every non-negative case.
 */
#include <stdio.h>
#include "lz_int.h"

void lz_lerp_q15_mmx(const int32_t *a, const int32_t *b,
                     const int32_t *frac, int32_t *out, int n);

static unsigned long rs = 1145141919UL;
static int32_t nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int32_t)(rs >> 1);
}

#define N 64

int main(void) {
    int32_t a[N], b[N], f[N], got[N];
    long bad = 0, n = 0;
    int k, pass;

    /* a at and around the table's endpoint, the delta either side of
       its limit, frac at both ends - every combination, in every lane.
       b is a DELTA from a here, not an absolute: the kernel contract is
       |b - a| <= 32767 and the callers meet it because a and b are
       adjacent entries of a 384-step table. Driving b independently
       breaks the precondition rather than testing the kernel - which
       the first version of this probe did, and the 260 mismatches it
       reported were all a = 0 with b = 32768. The boundary IS tested,
       at exactly +-32767. */
    static const int32_t ae[] = { 0, 1, 32766, 32767, 32768, 16384 };
    static const int32_t de[] = { 0, 1, -1, 32767, -32767, 341, -341 };
    static const int32_t fe[] = { 0, 1, 16384, 32766, 32767 };
    const int na = 6, nb = 7, nf = 5;

    for (pass = 0; pass < 4000 + na * nb * nf; pass++) {
        for (k = 0; k < N; k++) {
            if (pass < na * nb * nf) {
                a[k] = ae[((pass / (nb * nf)) + k) % na];
                b[k] = a[k] + de[((pass / nf) + k) % nb];
                f[k] = fe[(pass + k) % nf];
            } else {
                a[k] = nextv() % 32769;
                b[k] = a[k] + (nextv() % 65535) - 32767;
                f[k] = nextv() % 32768;
            }
        }
        lz_lerp_q15_mmx(a, b, f, got, N);
        for (k = 0; k < N; k++, n++) {
            int32_t want = a[k] + (((b[k] - a[k]) * f[k]) >> 15);
            if (got[k] != want) {
                if (bad < 6)
                    printf("  a=%6ld b=%6ld f=%6ld want %8ld got %8ld\n",
                           (long)a[k], (long)b[k], (long)f[k],
                           (long)want, (long)got[k]);
                bad++;
            }
        }
    }

    /* A tail that is not a multiple of four must still be right: the
       kernel runs the remainder in C and an off-by-one there writes
       past the vector part or leaves it stale. */
    /* Within the contract here too - the first version set b = 32768 - k
       against a = k, which is a delta of 32768 at lane 0 and reported
       the precondition rather than the tail. */
    for (k = 0; k < N; k++) { a[k] = k; b[k] = k + 300 - 2 * k; f[k] = 1000 + k; }
    for (k = 1; k <= 7; k++) {
        int j;
        int32_t t[N];
        for (j = 0; j < N; j++) t[j] = -1;
        lz_lerp_q15_mmx(a, b, f, t, k);
        for (j = 0; j < k; j++, n++) {
            int32_t want = a[j] + (((b[j] - a[j]) * f[j]) >> 15);
            if (t[j] != want) {
                if (bad < 6) printf("  tail n=%d lane %d want %ld got %ld\n",
                                    k, j, (long)want, (long)t[j]);
                bad++;
            }
        }
        if (t[k] != -1) { printf("  tail n=%d wrote past the end\n", k); bad++; }
    }

    printf("lerp_q15_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
