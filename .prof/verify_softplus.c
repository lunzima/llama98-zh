/* The table+poly softplus, exhaustively: sp_log1p, lz_softplus_poly and
 * the LZ_SP_HI shortcut (src/ops_quant.c).
 *
 * src/ops_quant.c is INCLUDED, not linked, so sp_log1p - a static, and
 * the one piece of new arithmetic here - is tested as the shipped
 * function rather than as a copy.
 *
 * FOUR ROWS, FOUR THINGS, written apart so that one surgical mutation
 * of the implementation reddens exactly ONE of them. Each row names its
 * own mutation, and each of the four was run:
 *
 *   B1 POLY    sp_log1p(u) is within a DERIVED cap of log1p(u), for
 *              EVERY float32 u in [2^-24, 1] - the whole domain the
 *              polynomial is fitted on, not the subset the exp happens
 *              to produce. Reference: log1pl on long double, which
 *              shares no arithmetic with a degree-8 float32 Horner.
 *              The cap is 3.3574e-7 and is derived, not fitted: the
 *              minimax fit's own error with the coefficients rounded to
 *              float32 (5.944e-8), plus a forward running bound on the
 *              16 Horner roundings (2.167e-7), plus the final multiply
 *              (5.960e-8). .prof/sp_fit.py computes all three.
 *              Mutation: c5 -0.137466222f -> -0.137465f, in src AND in
 *              this file's verbatim copy, so B2's reference moves with
 *              it and only this row can see it. The nudge is 134 ulps
 *              and worth 1.4e-6 relative, four times the cap.
 *
 *   B2 COMPOSE lz_softplus_poly(x) is BIT-IDENTICAL to the composition
 *              it claims to be: the shortcut, then -|x| into the tier's
 *              exp, then this file's verbatim copy of the polynomial,
 *              then the x add on the positive branch only. Under BOTH
 *              scalar tiers, so a body that hard-wired one exp would
 *              show up here. The reference reads LZ_SP_HI and the same
 *              coefficients, so it is blind to those two on purpose -
 *              B3 and B1 own them.
 *              This row also reports the intermediates' attained
 *              extremes and requires u <= 1 everywhere, which is the
 *              hazard the abs exists to prevent.
 *              Mutation: the positive branch's test, `x > 0.0f` ->
 *              `x > 1.0f`. Invisible to B1 (u-space), to B3 (which
 *              never calls the implementation) and to B4 (x <= -18).
 *
 *   B3 SHORT   returning x above LZ_SP_HI is the CORRECTLY ROUNDED
 *              answer, not an approximation: for every float32 x above
 *              it, x + log1pl(expl(-x)) rounds back to x. Long double
 *              only - this row does not call the implementation at all,
 *              it judges the constant.
 *              Mutation: LZ_SP_HI 16.0f -> 8.0f. B2's reference reads
 *              the same macro and stays green, which is the point.
 *
 *   B4 TAIL    deep in the negative tail the polynomial is the
 *              identity, so softplus costs the exp and nothing else and
 *              loses none of a value that is already 1e-8: for every
 *              float32 x <= -18, lz_softplus_poly(x) is BIT-IDENTICAL
 *              to the tier's exp of x. Both tiers. This is the
 *              relative-accuracy claim made exact - a log1p written as
 *              a polynomial in log1p rather than in u would fail here
 *              at every input.
 *              Mutation: an underflow guard `if (u < 1e-30f) u = 0.0f;`
 *              before the polynomial. Invisible to B1 (u >= 2^-24), to
 *              B2 (x >= -18 keeps u >= 1.5e-8) and to B3.
 *
 * Reported, not asserted: the end-to-end relative error of
 * lz_softplus_poly against logl(1+expl(x)) over the measured argument
 * range. It is the composition of B1's polynomial with lz_exp's own
 * error, both already bounded, so a second assertion here would only be
 * a second view of the same two predicates.
 *
 * Build (INCLUDES ops_quant.c, so do not also link it):
 *   gcc -O1 -std=gnu99 -Isrc -o .prof/verify_softplus.exe \
 *       .prof/verify_softplus.c src/ops_sched.c src/err.c src/compat.c \
 *       src/cpucheck.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include "ops_quant.h"
#include "ops_quant.c"

static long fails[5];
static long cases[5];
#define CHK(row, cond) do { cases[row]++; if (!(cond)) fails[row]++; } while (0)

/* The derived cap; see the header and .prof/sp_fit.py. */
static const long double CAP = 3.3574e-7L;

/* This file's verbatim copy of the shipped coefficients. B2 compares
   against a Horner built from these, so a coefficient mutated in src
   alone would redden B2 as well as B1 - which is why B1's mutation is
   applied here too. */
static const float REF_C[9] = {
    1.000000000f,    -0.499995530f,    0.333203435f,
   -0.248529464f,     0.191451162f,   -0.137466222f,
    0.079210736f,    -0.030110965f,    0.005384062f,
};

static float ref_log1p(float u) {
    float p = REF_C[8];
    int i;
    for (i = 7; i >= 0; i--) p = p * u + REF_C[i];
    return u * p;
}

static float ref_exp_tier(float x) {
    return lz_scalar_mode() ? lz_exp_fixed(x) : lz_exp_float_body(x);
}

/* The composition lz_softplus_poly claims to be, spelled out
   independently of it. Reads LZ_SP_HI and REF_C on purpose. */
static float ref_softplus(float x) {
    float nx, u, l;
    if (x > LZ_SP_HI) return x;
    nx = (x < 0.0f) ? x : -x;
    u = ref_exp_tier(nx);
    l = ref_log1p(u);
    return (x > 0.0f) ? (x + l) : l;
}

static float bits_to_f(uint32_t b) {
    union { uint32_t u; float f; } t;
    t.u = b;
    return t.f;
}
static uint32_t f_to_bits(float f) {
    union { uint32_t u; float f; } t;
    t.f = f;
    return t.u;
}

/* ---- B1: the polynomial, every float32 in [2^-24, 1] ----------------- */
static long double b1_worst;
static float b1_worst_u;

static void b1(void) {
    uint32_t lo = f_to_bits(ldexpf(1.0f, -24));
    uint32_t hi = f_to_bits(1.0f);
    uint32_t b;
    for (b = lo; b <= hi; b++) {
        float u = bits_to_f(b);
        long double ref = log1pl((long double)u);
        long double got = (long double)sp_log1p(u);
        long double r = fabsl(got - ref) / ref;
        if (r > b1_worst) { b1_worst = r; b1_worst_u = u; }
        CHK(1, r <= CAP);
    }
}

/* ---- B2: the composition, bit for bit, both tiers -------------------- */
static float b2_umax = -1.0f, b2_umin = 2.0f;
static float b2_pmax = -1.0f, b2_pmin = 2.0f;
static int b2_u_gt1;

static void b2_one(float x) {
    float got = lz_softplus_poly(x);
    float want = ref_softplus(x);
    float u, p;
    CHK(2, f_to_bits(got) == f_to_bits(want));
    if (x > LZ_SP_HI) return;
    u = ref_exp_tier((x < 0.0f) ? x : -x);
    if (u > b2_umax) b2_umax = u;
    if (u < b2_umin) b2_umin = u;
    if (u > 1.0f) b2_u_gt1 = 1;
    p = ref_log1p(u);
    if (u > 0.0f) {
        float q = p / u;
        if (q > b2_pmax) b2_pmax = q;
        if (q < b2_pmin) b2_pmin = q;
    }
}

static void b2(void) {
    /* Every float32 with 2^-8 <= |x| <= 64, plus a stride sweep of the
       smaller magnitudes. Below 2^-8 every u lands in [0.99610, 1], a
       window B1 already covers float by float, so what the small
       binades add is the branch structure and not the arithmetic - the
       stride sweep crosses both zeros and both signs. */
    const uint32_t small = f_to_bits(ldexpf(1.0f, -8));
    const uint32_t big = f_to_bits(64.0f);
    uint32_t b;
    int tier;
    for (tier = 0; tier <= 1; tier++) {
        lz_scalar_g = tier;
        for (b = small; b <= big; b++) {
            b2_one(bits_to_f(b));
            b2_one(bits_to_f(b | 0x80000000u));
        }
        for (b = 0; b < small; b += 251) {
            b2_one(bits_to_f(b));
            b2_one(bits_to_f(b | 0x80000000u));
        }
        b2_one(0.0f);
        b2_one(-0.0f);
    }
    lz_scalar_g = -1;
}

/* ---- B3: the shortcut's constant ------------------------------------- */
static void b3(void) {
    /* Every float32 strictly above LZ_SP_HI, up to FLT_MAX. Monotone
       inside a binade (ulp constant, exp(-x) decreasing), so the left
       edges would suffice; enumerated anyway because it is cheap. */
    uint32_t b;
    for (b = f_to_bits(LZ_SP_HI) + 1u; b <= f_to_bits(FLT_MAX); b++) {
        long double x = (long double)bits_to_f(b);
        long double full = x + log1pl(expl(-x));
        CHK(3, (float)full == bits_to_f(b));
    }
}

/* ---- B4: the tail is the exp, bit for bit ---------------------------- */
static void b4(void) {
    uint32_t b;
    int tier;
    for (tier = 0; tier <= 1; tier++) {
        lz_scalar_g = tier;
        for (b = f_to_bits(-18.0f); b <= 0xFF7FFFFFu; b++) {
            float x = bits_to_f(b);
            CHK(4, f_to_bits(lz_softplus_poly(x)) ==
                   f_to_bits(ref_exp_tier(x)));
        }
    }
    lz_scalar_g = -1;
}

/* ---- reported: end-to-end accuracy over the measured range ----------- */
static void report_acc(void) {
    /* The live call site (GDN forward_ssm) presents -15.10..11.29 and
       the KDA gate's own pre runs -32.19..10.35 (.prof measurements in
       the commit message); [-40, 16] contains both with margin. */
    int tier;
    for (tier = 0; tier <= 1; tier++) {
        long double worst = 0.0L, wx = 0.0L;
        uint32_t b;
        lz_scalar_g = tier;
        for (b = f_to_bits(ldexpf(1.0f, -14)); b <= f_to_bits(40.0f); b++) {
            float xs[2];
            int k;
            xs[0] = bits_to_f(b);
            xs[1] = -xs[0];
            for (k = 0; k < 2; k++) {
                long double x = (long double)xs[k];
                long double ref, got, r;
                if (x > 16.0L) continue;
                ref = log1pl(expl(x));
                if (ref <= 0.0L) continue;
                got = (long double)lz_softplus_poly(xs[k]);
                r = fabsl(got - ref) / ref;
                if (r > worst) { worst = r; wx = x; }
            }
        }
        /* (double) on the way out: this host's printf has no long
           double conversion and silently prints garbage for %Le. */
        printf("   end-to-end rel err, tier %s, x in [-40,16]: %.4e @x=%.6f\n",
               tier ? "fixed" : "float", (double)worst, (double)wx);
        lz_scalar_g = -1;
    }
}

/* The control the end-to-end number has to be read against: softplus is
   exp in the tail, so its error there IS the tier's exp error, and the
   very next line at both call sites (gt = lz_exp(g)) already pays it. */
static void report_exp(void) {
    int tier;
    for (tier = 0; tier <= 1; tier++) {
        long double worst = 0.0L, wx = 0.0L;
        uint32_t b;
        lz_scalar_g = tier;
        for (b = f_to_bits(ldexpf(1.0f, -14)); b <= f_to_bits(40.0f); b += 7) {
            float xs[2];
            int k;
            xs[0] = bits_to_f(b);
            xs[1] = -xs[0];
            for (k = 0; k < 2; k++) {
                long double x = (long double)xs[k], ref, got, r;
                if (x > 16.0L) continue;
                ref = expl(x);
                if (ref <= 0.0L) continue;
                got = (long double)ref_exp_tier(xs[k]);
                r = fabsl(got - ref) / ref;
                if (r > worst) { worst = r; wx = x; }
            }
        }
        printf("   control: lz_exp itself, tier %s, same range: "
               "%.4e @x=%.6f\n",
               tier ? "fixed" : "float", (double)worst, (double)wx);
        lz_scalar_g = -1;
    }
}

/* ---- reported: the libm arm this replaces ---------------------------- */
static void report_old(void) {
    long double worst = 0.0L, wx = 0.0L;
    uint32_t b;
    for (b = f_to_bits(ldexpf(1.0f, -14)); b <= f_to_bits(40.0f); b += 7) {
        float xs[2];
        int k;
        xs[0] = bits_to_f(b);
        xs[1] = -xs[0];
        for (k = 0; k < 2; k++) {
            long double x = (long double)xs[k];
            long double ref, got, r;
            float old;
            if (x > 16.0L) continue;
            ref = log1pl(expl(x));
            if (ref <= 0.0L) continue;
            old = (xs[k] > 20.0f) ? xs[k] : (float)log1p(exp((double)xs[k]));
            got = (long double)old;
            r = fabsl(got - ref) / ref;
            if (r > worst) { worst = r; wx = x; }
        }
    }
    printf("   LZ_SOFTPLUS_POLY=0 (libm double) over the same range: "
           "%.4e @x=%.6f\n", (double)worst, (double)wx);
}

int main(void) {
    static const char *names[5] = { "", "B1 POLY", "B2 COMPOSE",
                                    "B3 SHORT", "B4 TAIL" };
    int bad = 0, i;

    printf("LZ_SP_HI %.1f   cap %.4e\n", (double)LZ_SP_HI, (double)CAP);
    b1();
    b2();
    b3();
    b4();

    printf("   B1 worst rel %.4e @u=%.9g (cap %.4e)\n",
           (double)b1_worst, (double)b1_worst_u, (double)CAP);
    printf("   B2 attained: u in [%.9g, %.9g] (u>1 seen %d), "
           "P(u) in [%.7f, %.7f]\n",
           (double)b2_umin, (double)b2_umax, b2_u_gt1,
           (double)b2_pmin, (double)b2_pmax);
    report_acc();
    report_exp();
    report_old();

    for (i = 1; i <= 4; i++) {
        printf("%-11s %12ld cases  %ld mismatches  %s\n",
               names[i], cases[i], fails[i], fails[i] ? "FAIL" : "ok");
        if (fails[i]) bad = 1;
    }
    printf("%s\n", bad ? "VERIFY: FAIL" : "VERIFY: PASS");
    return bad;
}
