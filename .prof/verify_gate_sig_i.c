/* Exhaustive check for the gated norm's integer-coordinate sigmoid
 * (src/ops_norm.c, LZ_GATE_SIG_I).
 *
 * WHAT CHANGED. lz_rmsnorm_gated_out_fixed_int quantizes the gate to
 * Q15 for its product chain,
 *
 *     qg = clamp(q8_round(g * 2^eg), +-32767)
 *
 * and then used to reach the sigmoid through the FLOAT g a second time,
 * `sigmoid_q15(g[i])`. It now reads the integer coordinate instead,
 * `sigmoid_q15_i(qg, eg)`. So the gate's two appearances in
 * silu(g) = g * sigmoid(g) finally agree on which g they mean, and the
 * three converts sigmoid_q15 spends deriving its table coordinate go
 * away. This is VALUE-CHANGING and the change is what has to be bounded.
 *
 * sigmoid_q15_i ITSELF IS NOT RE-DERIVED HERE. That it equals
 * sigmoid_q15 at the SAME coordinate is milestone 5's claim and
 * .prof/m5sig.c's exhaustive sweep, and its wide band is
 * .prof/verify_sig_i_wide.c's. This file checks the SUBSTITUTION - a
 * different coordinate, not a different function - over the domain this
 * call site can present: |v| <= 32767 (the clamp the caller applies
 * before the call) and every e in sigmoid_q15_i's own band, EMAX below.
 * The engine measures eg 11..15 on both gate corpora and both gate
 * models, so the sweep is far wider than the call site reaches.
 *
 * Linked against the real src/ops_quant.c rather than carrying copies:
 * the object under test here is the interpolation table, which is
 * static to that file, and a copied table would be checking a copy.
 *
 *   A1 RANGE. Over all 28 x 65,535 presented (v, e), sigmoid_q15_i
 *      returns a value in [0, 32768], and BOTH ends are ATTAINED. This
 *      is what says the substitution cannot widen the int64 chain
 *      downstream: sigmoid_q15's own range is the same interval, so
 *      `(t64 * sigv) >> 15` keeps every bound it had. "Attained" is not
 *      decoration - a bound nothing reaches would pass while the entry
 *      quietly saturated one short.
 *      Mutation: sigmoid_q15_i's `return 32768` -> `return 32769`.
 *   A2 CELL. The interval A3 bounds over is the right one: for every
 *      (v, e), q8_round's preimage of v at scale 2^e is contained in
 *      [(v-0.5)*2^-e, (v+0.5)*2^-e]. Checked by driving the engine's own
 *      expression, q8_round(g * pow2f(e)) with the caller's clamp, at
 *      seven points per cell - five strictly inside, which must all give
 *      v, and the two endpoints, which are round-half-even ties and may
 *      land on either neighbour. A3 evaluating AT the endpoints is
 *      therefore conservative, not optimistic.
 *      Mutation: q8_round's magic add gains a `+ 0.5f`, which slides the
 *      rounding boundary half a step.
 *   A3 BOUND. For every (v, e), the substitution's error at both ends of
 *      that cell obeys a DERIVED bound:
 *          |sigmoid_q15_i(v,e) - sigmoid_q15(g)| <= ceil(4096 * 2^-e) + 2
 *      4096 is 32768 * (1/4) * (1/2): the Q15 scale, the sigmoid's
 *      Lipschitz constant, and the half-step the quantizer may move the
 *      coordinate. +2 is the two evaluations' independent `>> 15` floors
 *      in the lerp. The observed maximum is printed, because a bound
 *      that is never approached says nothing about the real magnitude.
 *      Mutation: sigmoid_q15_i's `sh = e - 4` -> `sh = e - 3`.
 *   A4 MONO. sigmoid_q15 is non-decreasing over the whole ordered sweep
 *      of coordinates A3 evaluates. This is what makes A3's endpoint
 *      evaluation a BOUND ON THE CELL rather than a two-point sample of
 *      it: a monotone function on an interval attains its extremes at
 *      the ends. Without this row A3 is exhaustive on the integer grid
 *      and blind between grid points.
 *      Mutation: sigmoid_q15's `if (x <= LZ_SIG_LO) return 0` -> 1.
 *
 * Build:
 *   gcc -O1 -std=gnu99 -Isrc -o .prof/verify_gate_sig_i.exe \
 *       .prof/verify_gate_sig_i.c src/ops_quant.c src/ops_sched.c \
 *       src/cpucheck.c src/err.c src/compat.c -lm
 * Exit 0 = every row green, 1 = not.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "ops_quant.h"

/* Tracks LZ_SIG_EMAX (src/ops_quant.c), which is private to that file -
   the sweep must cover the whole band the entry accepts, not the band it
   accepted when this was written. A2's cell arithmetic is exact for
   every e here: (v +- 0.5) needs 17 bits and 2^-e only moves the
   exponent. */
#define EMAX 50
#define VMAX 32767
/* The exponent band the engine was measured to present (the CLI's
   `gate sig int ... eg lo..hi` line, kmr20 and kunmoe-v2-t2-probe, both
   gate corpora). A3's worst case over the whole domain is dominated by
   e = 0, which nothing reaches, so the worst inside this band is
   reported separately - that is the number the shipping path pays. */
#define EBAND_LO 11
#define EBAND_HI 15

static float p2(int e) { return (float)ldexp(1.0, e); }

/* The caller's own quantize-and-clamp, so A2 drives the engine's
   expression rather than a restatement of it. */
static int qgate(float g, int e) {
    int v = q8_round(g * p2(e));
    if (v > 32767) v = 32767; else if (v < -32767) v = -32767;
    return v;
}

int main(void) {
    long a1_cases = 0, a1_bad = 0;
    long a2_cases = 0, a2_bad = 0;
    long a3_cases = 0, a3_bad = 0;
    long a4_cases = 0, a4_bad = 0;
    int lo_hit = 0, hi_hit = 0;
    int a3_worst = 0, a3_worst_e = 0, a3_worst_v = 0;
    int a3_band = 0, a3_band_e = 0;
    double a3_band_sum = 0.0;
    long a3_band_n = 0;
    int a2_first_e = -1, a2_first_v = 0;
    int a4_first_e = -1;
    int e, rc = 0;

    for (e = 0; e <= EMAX; e++) {
        float step = p2(-e);
        int32_t prev = -1;            /* A4's running monotone witness */
        int v;
        for (v = -VMAX; v <= VMAX; v++) {
            float glo = ((float)v - 0.5f) * step;
            float ghi = ((float)v + 0.5f) * step;
            int32_t si = sigmoid_q15_i(v, e);
            int32_t slo, shi;
            int bound, k, d;

            /* A1 */
            a1_cases++;
            if (si < 0 || si > 32768) a1_bad++;
            if (si == 0) lo_hit = 1;
            if (si == 32768) hi_hit = 1;

            /* A2: five strictly interior points must quantize back to v.
               The endpoints are ties and are deliberately not required
               to - which is exactly why A3 uses them. */
            for (k = 1; k <= 5; k++) {
                float gin = glo + (ghi - glo) * ((float)k / 6.0f);
                a2_cases++;
                if (qgate(gin, e) != v) {
                    if (a2_first_e < 0) { a2_first_e = e; a2_first_v = v; }
                    a2_bad++;
                }
            }

            /* A3 + A4 */
            slo = sigmoid_q15(glo);
            shi = sigmoid_q15(ghi);
            bound = (int)ceil(4096.0 * ldexp(1.0, -e)) + 2;
            a3_cases += 2;
            d = (int)(si - slo); if (d < 0) d = -d;
            if (d > bound) a3_bad++;
            if (d > a3_worst) { a3_worst = d; a3_worst_e = e; a3_worst_v = v; }
            if (e >= EBAND_LO && e <= EBAND_HI) {
                if (d > a3_band) { a3_band = d; a3_band_e = e; }
                a3_band_sum += d; a3_band_n++;
            }
            d = (int)(si - shi); if (d < 0) d = -d;
            if (d > bound) a3_bad++;
            if (d > a3_worst) { a3_worst = d; a3_worst_e = e; a3_worst_v = v; }
            if (e >= EBAND_LO && e <= EBAND_HI) {
                if (d > a3_band) { a3_band = d; a3_band_e = e; }
                a3_band_sum += d; a3_band_n++;
            }

            /* A4: glo <= ghi <= next cell's glo, so the sweep visits the
               coordinates in increasing order and one comparison per
               evaluation covers the whole chain. */
            a4_cases += 2;
            if (slo < prev) { a4_bad++; if (a4_first_e < 0) a4_first_e = e; }
            prev = slo;
            if (shi < prev) { a4_bad++; if (a4_first_e < 0) a4_first_e = e; }
            prev = shi;
        }
    }

    printf("A1 RANGE  %ld cases, %ld out of [0,32768]; 0 attained %d,"
           " 32768 attained %d\n", a1_cases, a1_bad, lo_hit, hi_hit);
    if (a1_bad || !lo_hit || !hi_hit) { printf("A1 FAIL\n"); rc = 1; }

    printf("A2 CELL   %ld interior points, %ld quantized outside their"
           " cell", a2_cases, a2_bad);
    if (a2_first_e >= 0) printf(" (first e=%d v=%d)", a2_first_e, a2_first_v);
    printf("\n");
    if (a2_bad) { printf("A2 FAIL\n"); rc = 1; }

    printf("A3 BOUND  %ld endpoint evaluations, %ld over the derived"
           " bound; worst |diff| %d at e=%d v=%d\n",
           a3_cases, a3_bad, a3_worst, a3_worst_e, a3_worst_v);
    printf("          within the measured band e=%d..%d: worst %d"
           " (at e=%d), mean %.4f of 32768 full scale\n",
           EBAND_LO, EBAND_HI, a3_band, a3_band_e,
           a3_band_n ? a3_band_sum / (double)a3_band_n : 0.0);
    if (a3_bad) { printf("A3 FAIL\n"); rc = 1; }

    printf("A4 MONO   %ld ordered coordinates, %ld decreases", a4_cases,
           a4_bad);
    if (a4_first_e >= 0) printf(" (first at e=%d)", a4_first_e);
    printf("\n");
    if (a4_bad) { printf("A4 FAIL\n"); rc = 1; }

    printf(rc ? "FAIL\n" : "PASS\n");
    return rc;
}
