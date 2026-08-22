/* Cross-entry gate for sigmoid_q15 against sigmoid_q15_t.
 *
 * NO LONGER THE CONV EPILOGUE'S GATE. It was written for it: the C tier
 * reached the sigmoid through sigmoid_q15_t at the folded coordinate
 * t = acc_f*sig_k1[c] + sig_k2 and the --kernel arm-asm arm through
 * sigmoid_q15 at x = acc_f*oscale, and this file was the proof that the
 * two entries are one function. Under LZ_CONV_SIG_I both tiers call
 * sigmoid_q15_i on the accumulator's bits instead and neither of the
 * two float entries is on that path at all; .prof/verify_sig_i_wide.c
 * is what covers it now. What is kept here is the identity itself,
 * which sigmoid_q15_t's remaining callers (lz_silu, lz_sigmoid, the
 * =0 control arm) still rest on - and the table fact below, which no
 * algebraic argument replaces.
 *
 * THE IDENTITY IS ALGEBRAIC, WITH ONE TABLE FACT IN IT.
 *   t == fl(16*(x+12)) == 16*fl(x+12), since scaling by a power of two
 *   is exact and commutes with round-to-nearest, and sigmoid_q15's own
 *   coordinate is exactly 16*fl(x+12). So the two entries index the
 *   table at the same place whenever they index it at all.
 *   The clamps then have to agree too, and one of them agrees only
 *   because of a VALUE IN THE TABLE: sigmoid_q15 returns a literal
 *   32768 for x >= 12 while sigmoid_q15_t returns g_sigtab[384]. That
 *   is a fact about sig_build's output, not about the algebra, which is
 *   why it is swept here rather than argued in a comment.
 *
 * DRIVES THE SHIPPED FUNCTIONS, not a second copy of them.
 * .prof/verify_conv_fold3.c checks the same identity on a
 * reimplementation - useful for the joint (acc, oscale) range, useless
 * for catching a change to the shipped table or to either entry's
 * clamps, which is what this file exists for.
 *
 * COVERAGE IS COUNTED, NOT ASSUMED. The three regions (below the table,
 * inside it, above it) are tallied separately and an empty one FAILS:
 * a sweep that never reaches x >= 12 would report the clamp agreement
 * green without ever having looked at it.
 *
 * Build (x86, no QEMU needed - the identity is a fact about the two C
 * entries, and the ARM arm is what consumes it):
 *   gcc -O2 -std=c99 -I../src -o conv_epi_xentry conv_epi_xentry.c \
 *       ../src/ops.c ../src/ops_quant.c ../src/ops_sched.c \
 *       ../src/ops_norm.c ../src/ops_gdn.c ../src/ops_matmul.c \
 *       ../src/ops_t2_scalar.c ../src/fwht.c ../src/cpucheck.c \
 *       ../src/err.c -lm
 * Exit 0 = identical everywhere and every region visited, 1 = not. */
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "ops.h"
#include "ops_quant.h"
#include "ops_kernel_shared.h"

/* LZ_CONV_ES, which forward.c keeps private. Only used to place the
   oscale sweep where the engine's own exponents land; the identity does
   not depend on the value. */
#define CV_ES 9

static long g_bad, g_n;
static long g_lo, g_mid, g_hi;     /* x <= -12, inside, x >= 12 */

static void one(int32_t acc, int e) {
    float oscale = (float)ldexp(1.0, -(e + CV_ES));
    float k1, o2, k2 = lz_sig_q15_t_offset();
    float acc_f = lz_i32f(acc);
    float x, t;
    int32_t a, b;
    lz_sig_q15_fold(oscale, &k1, &o2);
    x = acc_f * oscale;
    t = acc_f * k1 + k2;
    a = sigmoid_q15(x);
    b = sigmoid_q15_t(t);
    g_n++;
    if (x <= -12.0f)     g_lo++;
    else if (x >= 12.0f) g_hi++;
    else                 g_mid++;
    if (a != b) {
        if (g_bad < 10)
            printf("  MISMATCH acc=%ld e=%d x=%.9g t=%.9g  sigq=%ld sigq_t=%ld\n",
                   (long)acc, e, (double)x, (double)t, (long)a, (long)b);
        g_bad++;
    }
}

int main(void) {
    int e, i;
    int32_t bound = lz_conv_accum_bound(4);
    /* THE TABLE FACT, NAMED. The sweep below would catch this too, but
       only as one bucket among 259,863 - and a future sweep that stopped
       reaching x >= 12 would stop catching it silently. Here the two
       values are asked for by name: sigmoid_q15's literal upper clamp
       against the g_sigtab[384] that sigmoid_q15_t's returns. */
    {
        int32_t lit = sigmoid_q15(12.0f);          /* the literal 32768 */
        int32_t tab = sigmoid_q15_t(384.0f);       /* g_sigtab[LZ_SIG_N] */
        printf("upper clamp: sigmoid_q15(12)=%ld  sigmoid_q15_t(384)=%ld\n",
               (long)lit, (long)tab);
        if (lit != tab) {
            printf("FAIL: the two entries' upper clamps are different "
                   "values - the conv epilogue's arm-asm tier is wrong "
                   "for every saturated channel\n");
            return 1;
        }
    }
    /* Every exponent lz_conv_norm_pow2 can return (0..30), and both
       signs of acc. The dense band walks the table's own resolution;
       the geometric one carries acc out to the accumulator bound so
       both clamps are reached at the small exponents. */
    for (e = 0; e <= 30; e++) {
        for (i = -4096; i <= 4096; i++) one((int32_t)i, e);
        for (i = 0; i < 31; i++) {
            int32_t p = (int32_t)1 << i;
            one(p, e);      one(p - 1, e);      one(p + 1, e);
            one(-p, e);     one(-p + 1, e);     one(-p - 1, e);
        }
    }
    /* The exact clamp boundaries, built backwards from x rather than
       stumbled into: acc = +-12 * 2^(e+9) puts x on +-12 exactly. */
    for (e = 0; e <= 18; e++) {
        double s = ldexp(12.0, e + CV_ES);
        if (s > 2.0e9) continue;
        one((int32_t)s, e);      one((int32_t)s - 1, e);
        one((int32_t)s + 1, e);  one(-(int32_t)s, e);
        one(-(int32_t)s + 1, e); one(-(int32_t)s - 1, e);
    }
    printf("conv-epi x-entry: %ld cases, %ld mismatches\n", g_n, g_bad);
    printf("  regions: x<=-12 %ld | inside %ld | x>=12 %ld  (bound %ld)\n",
           g_lo, g_mid, g_hi, (long)bound);
    if (!g_lo || !g_mid || !g_hi) {
        printf("FAIL: a region was never reached - the clamp it guards "
               "was not tested\n");
        return 1;
    }
    if (g_bad) { printf("FAIL\n"); return 1; }
    printf("PASS\n");
    return 0;
}
