/* Is sigmoid_q15_i EXACTLY sigmoid_q15, or only close?
 *
 * The int-pipeline milestone 5 entry claims equality, not a tolerance:
 * sigmoid_q15_i(v, e) == sigmoid_q15((float)v * 2^-e) for every v in
 * the producer's domain and every e the engine uses. The claim rests on
 * every step of the float entry's coordinate derivation being exact
 * over that domain (ops_quant.c has the argument), so a single
 * mismatch falsifies it - this is an exhaustive sweep, not a sample.
 *
 * Swept exhaustively over the full int16 domain (both signs, plus the
 * clamp shoulders past +-12*2^e) at every e from 0 to 19, which covers
 * the exponents the four wired chains use and the whole band where the
 * exactness argument is claimed to hold. e = 20..27 is swept too and
 * reported SEPARATELY: there the float entry's `x + 12` starts rounding
 * and the two are expected to differ, with the integer one being the
 * correct of the pair - a difference reported as a mismatch here would
 * be reporting the float entry's error against itself.
 *
 * Build:
 *   gcc -O1 -std=gnu99 -Isrc -o .prof/m5sig.exe .prof/m5sig.c \
 *       src/ops_quant.c src/ops_sched.c src/err.c src/compat.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ops_quant.h"

static float p2(int e) { return (float)ldexp(1.0, e); }

int main(void) {
    int e;
    long total = 0, bad = 0, badhi = 0;
    int worst = 0, worst_e = 0, worst_v = 0;

    for (e = 0; e <= 27; e++) {
        /* Past the table edge on both sides, so the clamp shoulders are
           swept and not just the interpolated interior. */
        long lo = -40000, hi = 40000, v;
        long emis = 0;
        int emax = 0;
        for (v = lo; v <= hi; v++) {
            int32_t gi = sigmoid_q15_i((int32_t)v, e);
            int32_t gf = sigmoid_q15((float)v * p2(-e));
            int d = (int)(gi - gf);
            if (d < 0) d = -d;
            if (d) {
                emis++;
                if (d > emax) emax = d;
                if (e <= 19 && d > worst) { worst = d; worst_e = e; worst_v = (int)v; }
            }
            total++;
        }
        if (e <= 19) bad += emis; else badhi += emis;
        printf("e=%-2d  swept %ld  mismatches %ld  max |diff| %d%s\n",
               e, hi - lo + 1, emis, emax,
               e > 19 ? "   (past the exactness band)" : "");
    }
    printf("\nswept %ld cases total\n", total);
    printf("e in [0,19] (the claim): %ld mismatches", bad);
    if (bad) printf("  worst |diff| %d at e=%d v=%d", worst, worst_e, worst_v);
    printf("\ne in [20,27]: %ld mismatches (expected nonzero)\n", badhi);

    /* The domain getter must agree with where the float entry clamps. */
    for (e = 0; e <= 19; e++) {
        int32_t lim = lz_sig_q15_domain(e);
        if (sigmoid_q15_i(lim, e) != 32768 ||
            sigmoid_q15_i(-lim, e) != 0 ||
            sigmoid_q15((float)lim * p2(-e)) != 32768 ||
            sigmoid_q15((float)(-lim) * p2(-e)) != 0) {
            printf("DOMAIN MISMATCH at e=%d (lim=%d)\n", e, (int)lim);
            bad++;
        }
    }
    printf("domain getter: checked e in [0,19]\n");
    return bad ? 1 : 0;
}
