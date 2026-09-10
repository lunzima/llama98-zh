/* Standalone scalar check for lz_situ_glu / lz_swiglu_clamp against a
 * libm double "exact" oracle - no model, no forward pass, just the two
 * new activation primitives on a fixed set of test vectors plus a
 * random sweep.
 *
 * lz_sigmoid/lz_tanh dispatch on lz_sig_mode() (ops_sched.c), which
 * defaults to the FIXED Q15 table, not the lz_exp float approximation -
 * same as lz_silu inherits. This harness forces lz_float_ref_g = 1
 * directly (the global --float-ref off sets) so the comparison is
 * against the float path lz_exp documents a relative-error contract
 * for (<= 1e-5), not the coarser fixed table.
 *
 *   gcc -O2 -std=c99 -I../src situ_swiglu_check.c -o situ_swiglu_check_gcc.exe -lm
 *   wcc386 -za99 -bt=nt -otexan -I../src situ_swiglu_check.c
 *   wlink name situ_swiglu_check_w system nt_win op maxe=25 op q file situ_swiglu_check.obj
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ops.h"

/* ops_quant.c has other functions (lz_quantize_q8 and friends) that call
   lz_cpu_has_mmx() - none of which lz_situ_glu/lz_swiglu_clamp/lz_tanh
   reach, but the linker still wants the symbol since it is one
   translation unit. The real lz_cpu_has_mmx_probe (ops.c) does a CPUID
   probe this harness has no use for; always-0 is correct here (it only
   ever gates an MMX _mm_empty() pairing in code paths never called). */
int g_lz_has_mmx = 0;
int lz_cpu_has_mmx_probe(void) { return 0; }

#include "ops_quant.c"
#include "ops_sched.c"

static double d_sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
static double d_tanh(double x) { return tanh(x); }

static double d_situ_glu(double gate, double up, double beta,
                         double linear_beta, int has_linear_beta) {
    double situ_a = beta * d_tanh(gate / beta) * d_sigmoid(gate);
    if (has_linear_beta) up = linear_beta * d_tanh(up / linear_beta);
    return situ_a * up;
}

static double d_swiglu_clamp(double gate, double up, double limit,
                             double alpha) {
    double glu;
    if (gate > limit) gate = limit;
    if (up > limit) up = limit;
    if (up < -limit) up = -limit;
    glu = gate * d_sigmoid(gate * alpha);
    return (up + 1.0) * glu;
}

static unsigned g_rng = 1145141919u;
static float frand(float lo, float hi) {
    /* xorshift32, deterministic across gcc/Watcom - no libc rand()
       implementation to disagree about.

       FOUND BY THIS HARNESS'S FIRST DRAFT: dividing by the DOUBLE
       constant 4294967295.0 gave gcc and Watcom two DIFFERENT streams
       of "random" floats under lz_fpu_float_begin()'s PC=24, not just
       two different rounding of the same math - x87 compresses doubles
       to 24 bits too under PC=24 (ops_quant.c's lz_exp design comment
       makes the same point), so Watcom's division ran at 24-bit
       mantissa precision while gcc's (SSE, unaffected by the x87
       control word) ran at full double precision. The two builds were
       then comparing lz_situ_glu/lz_swiglu_clamp against DIFFERENT
       (gate, up) inputs, which is why their "worst error" figures
       against the double oracle disagreed even though neither is
       wrong on its own.

       Two separate fixes, not one: staying in float32 removes the
       double-under-PC=24 problem, but (float) of the FULL 32-bit
       g_rng is itself rule six's int32->float hazard (item 7) -
       x87's fild loads exactly and rounds once when fused into the
       following multiply, SSE's cvtsi2ss rounds the int to float32
       FIRST and then rounds the multiply again, and g_rng routinely
       exceeds 2^24 where those two paths can disagree. Masking to the
       low 24 bits before converting sidesteps that one too: every
       value in [0, 2^24) converts to float32 EXACTLY, on both x87 and
       SSE, so there is only the one rounding (the multiply) left to
       agree on - and multiplying by an exact power-of-two reciprocal
       is exact either way (rule six's carve-out for divisors that are
       powers of two). */
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return lo + (hi - lo) *
                ((float)(g_rng & 0x00FFFFFFu) * (1.0f / 16777216.0f));
}

/* FOUND BY THIS HARNESS'S FIRST DRAFT: a bare relative-error check with
   only a 1e-6 floor flagged 65/200000 situ_glu draws as "bad", worst
   12.3% relative error - all of them at points where `up` (and hence
   the whole product) is near zero, where the true value itself is
   1e-4..1e-3 and a ~1e-5 ABSOLUTE difference (exactly what chaining
   lz_exp's own documented 1e-5 relative-error contract through two
   sigmoid/tanh calls predicts) reads as a huge ratio. Requiring BOTH a
   non-negligible absolute gap is what CLAUDE.md's rule four calls "a
   metric that goes blind near its own zero" - relative error alone
   cannot tell "small and slightly wrong" from "small and exactly
   right's rounding noise" apart. */
static int is_bad(double got, double want) {
    double abs_err = fabs(got - want);
    double rel_err = abs_err / (fabs(want) > 1e-6 ? fabs(want) : 1.0);
    return abs_err > 1e-4 && rel_err > 2e-4;
}

static double g_worst_abs_situ, g_worst_abs_clamp;
static int g_n_bad_situ, g_n_bad_clamp, g_n_situ, g_n_clamp;

/* Raw float dump, cmp'd byte-for-byte between the gcc and Watcom builds
   (rule two) - the check against the libm double oracle above can pass
   identically on both builds while the two disagree with EACH OTHER
   (found by this harness's first draft: frand()'s bug made exactly
   that happen, both runs individually "close enough" to the double
   reference, worst_abs_err 5.96e-2 on gcc vs 2.50e-1 on Watcom - two
   different numbers that were each, on their own, unremarkable). No
   printf %f here (rule six item 3 - moot after lz_fpu_float_end, but
   the binary form is what actually gets cmp'd, not a reprintable text
   line). */
/* Confirmed identical byte-for-byte (200,006/200,007 floats each,
   inputs and outputs both) between gcc and Watcom, argv[1] pointing
   both builds at their own dump file and `cmp`-ing the two. The
   worst_abs_err PRINTED by the two builds still differs slightly
   (~6e-2 vs ~2.5e-1 for situ_glu on one run) - that is d_situ_glu's
   libm tanh/exp (MinGW vs Watcom's math387r), not lz_situ_glu itself;
   the oracle is a sanity check on magnitude, not the thing being
   shipped, and is not required to be cross-vendor bit-identical the
   way lz_exp is. */
static FILE *g_dump;
static void dumpf(float v) { if (g_dump) fwrite(&v, sizeof v, 1, g_dump); }

static void check_situ(float gate, float up, float beta, float lb, int has_lb) {
    float got = lz_situ_glu(gate, up, beta, lb, has_lb);
    double want = d_situ_glu(gate, up, beta, lb, has_lb);
    double abs_err = fabs((double)got - want);
    g_n_situ++;
    if (abs_err > g_worst_abs_situ) g_worst_abs_situ = abs_err;
    if (is_bad(got, want)) g_n_bad_situ++;
    dumpf(got);
}

static void check_clamp(float gate, float up, float limit, float alpha) {
    float got = lz_swiglu_clamp(gate, up, limit, alpha);
    double want = d_swiglu_clamp(gate, up, limit, alpha);
    double abs_err = fabs((double)got - want);
    g_n_clamp++;
    if (abs_err > g_worst_abs_clamp) g_worst_abs_clamp = abs_err;
    if (is_bad(got, want)) g_n_bad_clamp++;
    dumpf(got);
}

int main(int argc, char **argv) {
    int i;
    unsigned fpu;

    if (argc > 1) g_dump = fopen(argv[1], "wb");

    lz_float_ref_g = 1;   /* --float-ref off: lz_sigmoid takes the lz_exp branch */
    fpu = lz_fpu_float_begin();

    /* Fixed vectors: zero, the two betas as inputs to themselves,
       large magnitude (but under lz_exp's -87..20 contract), and the
       has_linear_beta=0 (unbounded up) branch. */
    check_situ(0.0f, 0.0f, 4.0f, 25.0f, 1);
    check_situ(4.0f, 25.0f, 4.0f, 25.0f, 1);
    check_situ(-4.0f, -25.0f, 4.0f, 25.0f, 1);
    check_situ(12.0f, 1e6f, 4.0f, 25.0f, 0);
    check_situ(-12.0f, -1e6f, 4.0f, 25.0f, 0);
    check_situ(1.0f, 1.0f, 4.0f, 25.0f, 1);

    check_clamp(0.0f, 0.0f, 10.0f, 1.702f);
    check_clamp(10.0f, 10.0f, 10.0f, 1.702f);
    check_clamp(-10.0f, -10.0f, 10.0f, 1.702f);
    check_clamp(1e6f, 1e6f, 10.0f, 1.702f);
    check_clamp(-50.0f, 1e6f, 10.0f, 1.702f);
    check_clamp(5.0f, 0.0f, 10.0f, 1.702f);
    check_clamp(5.0f, 0.0f, 7.0f, 1.0f);

    /* Random sweep, both functions, 200000 draws in [-30, 30]. gate/up
       themselves go into the dump too - a cmp of only the OUTPUTS could
       not tell "both builds compute the same function" apart from
       "both builds happened to draw the same wrong inputs and still
       agreed", which is exactly backwards from what this file is for. */
    for (i = 0; i < 200000; i++) {
        float gate = frand(-30.0f, 30.0f), up = frand(-30.0f, 30.0f);
        dumpf(gate); dumpf(up);
        check_situ(gate, up, 4.0f, 25.0f, 1);
        check_clamp(gate, up, 10.0f, 1.702f);
    }

    lz_fpu_float_end(fpu);
    if (g_dump) fclose(g_dump);

    printf("situ_glu:      n=%d worst_abs_err=%.3e bad(abs>1e-4 && rel>2e-4)=%d\n",
          g_n_situ, g_worst_abs_situ, g_n_bad_situ);
    printf("swiglu_clamp:  n=%d worst_abs_err=%.3e bad(abs>1e-4 && rel>2e-4)=%d\n",
          g_n_clamp, g_worst_abs_clamp, g_n_bad_clamp);
    return (g_n_bad_situ || g_n_bad_clamp) ? 1 : 0;
}
