/* The folded-coordinate integer entries, exhaustively:
 * lz_fold_f32, lz_exp_t_fold, sigmoid_q15_ti, lz_exp_t and the floor
 * shift under all four (src/ops_quant.c).
 *
 * These take x == v * m * 2^-s, a RATIONAL scale, where sigmoid_q15_i
 * takes only a power-of-two one. Every row below tests an equality, not
 * a tolerance, except A6 which is an accuracy bound and says so.
 *
 * src/ops_quant.c is INCLUDED, not linked, so that fold_floor_shr - a
 * static, and the one piece of new arithmetic every entry shares - is
 * tested as the shipped function instead of as a copy.
 *
 * SEVEN ROWS, SEVEN THINGS, written apart so that one surgical mutation
 * of the implementation reddens exactly ONE of them. Each row names
 * its own mutation, and each of the seven was run:
 *
 *   A1 FOLD    lz_fold_f32 splits a * 2^p into m * 2^-s EXACTLY, and
 *              accepts exactly the window it documents. Reference:
 *              ldexpl on long double, whose 64-bit significand holds
 *              both sides with room to spare, and which shares no
 *              arithmetic with a routine that reads the exponent field
 *              by hand. Exhaustive by factoring - the implementation's
 *              mantissa path and its exponent path are independent (the
 *              mantissa never reaches `sc`, and `sc` reaches the
 *              mantissa only through one bounded left shift), so all
 *              2^23 mantissas at the window's two edges and an interior
 *              exponent, crossed with all 256 exponents at four
 *              mantissas, covers every distinct behaviour of the 2^32
 *              inputs.
 *              Mutation: accept one exponent more (SMAX -> SMAX+1 in
 *              the refusal test only), which no other row's folds are
 *              anywhere near.
 *
 *   A2 FLOOR   fold_floor_shr is floor(p / 2^sh) for EVERY int64 p and
 *              every sh, including sh == 0, sh >= 64 and p ==
 *              INT64_MIN. Two references sharing nothing with the
 *              implementation or with each other: `/` with a truncation
 *              correction, and long double floorl.
 *              Mutation: the sh >= 64 arm returns 0 instead of the
 *              sign. That arm is reachable from this row alone - every
 *              other row's shifts are 62 or below.
 *
 *   A3 SIGT    sigmoid_q15_ti(v, m, s) equals the table value at the
 *              coordinate a long double computes. The reference reaches
 *              the table through sigmoid_q15_i's OWN wide band - a
 *              different, already exhaustively verified entry
 *              (.prof/verify_sig_i_wide.c) - by handing it the same
 *              coordinate at e = 20, so what is under test is the
 *              coordinate arithmetic and not the lerp the two share.
 *              Mutation: the coordinate's scale, s - 19 -> s - 20.
 *              NOT the seam (s >= 19 -> s > 19): that one is invisible
 *              here and correctly so - at s == 19 the shift arm and the
 *              saturating arm compute the same thing, which is a fact
 *              about the code rather than a hole in the row.
 *
 *   A4 EXPT    lz_exp_t(v, m, s) equals lz_exp_fixed's Taylor evaluated
 *              at the (n, idx, sq) a long double computes - over the
 *              reachable domain AND over the accepted domain's
 *              extremes, which are one predicate and therefore one row.
 *              The Taylor is NOT under test here: it is lz_exp_fixed's,
 *              shipped and unchanged, so the reference copies it
 *              verbatim and A6 is what covers it.
 *              This row also reports the intermediates' attained
 *              maxima and requires the documented ones to be reached,
 *              so that no bound in the header is quietly slack. The
 *              extremes are the point: sigmoid_q15_i's narrow body was
 *              verified over the domain its callers of the day
 *              presented and overflowed at e == 27 for a later one.
 *              Mutation: floor sq instead of rounding it.
 *
 *   A5 GUARD   lz_exp_t saturates exactly where its t-domain window
 *              says, the boundary value is finite and positive, and
 *              s outside [0, 62] is refused. No other row's t comes
 *              within three decades of the window.
 *              Mutation: `t64 > hi` -> `t64 >= hi`.
 *
 *   A6 ACC     end-to-end relative error against expl over the whole
 *              REACHABLE domain, against a bound DERIVED from the five
 *              roundings the value passes through - not fitted to the
 *              measurement. This is the row that covers the Taylor,
 *              which A4 declares out of scope, so its mutation is
 *              applied to the implementation AND to this file's
 *              declared verbatim copy of it: a mutation to the
 *              implementation alone would redden A4 and prove nothing
 *              about whether A6 can see anything.
 *              Mutation: LZ_EXP2_LN2 726817 -> 726000, in src and in
 *              the copy below.
 *
 *   A7 XLOG    lz_exp_t_fold really does carry log2(e)*32 into the
 *              coordinate: lz_exp_t at the fold of (a, p) equals
 *              lz_exp_FIXED at the float argument v * a * 2^p. Without
 *              this row every other one is blind to that factor - they
 *              all build their reference from the same (m, s) the fold
 *              returned, so a fold that dropped it would agree with
 *              itself everywhere. Tolerance 1e-3, five decades looser
 *              than the missing factor it hunts and deliberately blind
 *              to anything at the ulp level, which is A6's subject.
 *              Mutation: drop LZ_EXP_LOG2E32 from lz_exp_t_fold.
 *
 * Build (INCLUDES ops_quant.c, so do not also link it):
 *   gcc -O1 -std=gnu99 -Isrc -o .prof/verify_exp_t.exe \
 *       .prof/verify_exp_t.c src/ops_sched.c src/err.c src/compat.c \
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

#define SIG_N    384
#define SIG_T15  (192 << 15)
#define EXP_Q    20
#define EXP_QSC  (1 << EXP_Q)
#define LN2_Q    726817
#define LN2SQ_Q  251906
#define FOLD_SMAX 62

static long fails[8];
static long cases[8];
#define CHK(row, cond) do { cases[row]++; if (!(cond)) fails[row]++; } while (0)

/* ---- two independent references for the floor shift ------------------ */
static int64_t ref_div(int64_t p, int sh) {
    int64_t d, q;
    if (sh <= 0) return p;
    if (sh >= 63) return (p < 0) ? -1 : 0;   /* |p| < 2^63 */
    d = (int64_t)1 << sh;
    q = p / d;
    if (p % d != 0 && p < 0) q--;
    return q;
}
static int64_t ref_ld(int64_t p, int sh) {
    long double q;
    if (sh <= 0) return p;
    if (sh >= 64) return (p < 0) ? -1 : 0;
    q = floorl((long double)p / powl(2.0L, (long double)sh));
    return (int64_t)q;
}

/* ---- the exp Taylor, verbatim from lz_exp_fixed (NOT under test) -----
   Its own table rather than the engine's static one, so that the
   reference is a second construction of the same constants; A6 is what
   would catch the two diverging. */
static int32_t g_tab[32];
static void tab_build(void) {
    static const float t32[32] = {
        1.000000000f, 1.021897149f, 1.044273782f, 1.067140401f,
        1.090507733f, 1.114386743f, 1.138788635f, 1.163724859f,
        1.189207115f, 1.215247360f, 1.241857812f, 1.269050957f,
        1.296839555f, 1.325236643f, 1.354255547f, 1.383909882f,
        1.414213562f, 1.445180807f, 1.476826146f, 1.509164428f,
        1.542210825f, 1.575980845f, 1.610490332f, 1.645755478f,
        1.681792831f, 1.718619298f, 1.756252160f, 1.794709075f,
        1.834008086f, 1.874167634f, 1.915206561f, 1.957144124f,
    };
    int i;
    for (i = 0; i < 32; i++)
        g_tab[i] = (int32_t)(t32[i] * (float)EXP_QSC + 0.5f);
}

static int32_t taylor_prod(int idx, int32_t sq) {
    int32_t cq = EXP_QSC
        + (int32_t)((((int64_t)LN2_Q * sq) + (1 << (EXP_Q - 1))) >> EXP_Q)
        + (int32_t)((((int64_t)LN2SQ_Q * sq * sq)
                     + (1LL << (2 * EXP_Q - 1))) >> (2 * EXP_Q));
    return (int32_t)((((int64_t)g_tab[idx] * cq) + (1 << (EXP_Q - 1))) >> EXP_Q);
}

/* ---- long double coordinate reference -------------------------------- */
/* y32 == v*m*2^-s. v*m is at most 2^62 and long double's significand is
   64 bits, so the product and the power-of-two scaling are both exact
   and floorl cannot disagree with the true floor. */
static void coord_ref(int32_t v, int32_t m, int s,
                      long double *y32, int64_t *t, int32_t *n,
                      int *idx, int32_t *sq) {
    long double p = (long double)v * (long double)m;
    long double y = p * powl(2.0L, (long double)-s);
    long double tf = floorl(y);
    long double r = y - tf;
    long double nf = floorl(tf / 32.0L);
    int64_t sqr;
    *y32 = y;
    *t = (int64_t)tf;
    *n = (int32_t)nf;
    *idx = (int)(tf - 32.0L * nf);
    sqr = (int64_t)llroundl(r * 32768.0L);     /* half away from zero */
    if (sqr > 32767) sqr = 32767;
    if (sqr < 0) sqr = 0;
    *sq = (int32_t)sqr;
}

static float pow2ref(int e) {
    if (e < -126) return 0.0f;
    if (e > 127) e = 127;
    return (float)ldexp(1.0, e);
}

/* ---- A1: lz_fold_f32 -------------------------------------------------- */
static void a1_one(float a, int p) {
    int32_t m = 12345;
    int s = -1;
    int ok = lz_fold_f32(a, p, &m, &s);
    union { float f; uint32_t u; } b;
    int e;
    b.f = a;
    e = (int)((b.u >> 23) & 0xFFu);
    if (ok) {
        CHK(1, s >= 0 && s <= FOLD_SMAX);
        CHK(1, ldexpl((long double)a, p) == ldexpl((long double)m, -s));
    }
    if (e != 0 && e != 0xFF) {
        int sc = 150 - e - p;
        CHK(1, ok == ((sc <= FOLD_SMAX) && (sc >= -7)));
    } else {
        CHK(1, ok == (e == 0 && (b.u & 0x7FFFFFFFu) == 0));
    }
}

static void a1(void) {
    static const int ps[3] = { -15, -9, 0 };
    static const uint32_t mants[4] = { 0u, 1u, 0x400000u, 0x7FFFFFu };
    int pi, e, k, sgn;
    uint32_t mant;
    union { float f; uint32_t u; } b;
    for (pi = 0; pi < 3; pi++) {
        for (e = 0; e <= 255; e++)
            for (k = 0; k < 4; k++)
                for (sgn = 0; sgn < 2; sgn++) {
                    b.u = ((uint32_t)sgn << 31) | ((uint32_t)e << 23) | mants[k];
                    a1_one(b.f, ps[pi]);
                }
        for (k = 0; k < 3; k++) {
            int ee = (k == 0) ? (150 - ps[pi] - FOLD_SMAX)
                   : (k == 1) ? (157 - ps[pi]) : 119;
            if (ee < 1) ee = 1;
            if (ee > 254) ee = 254;
            for (mant = 0; mant < 0x800000u; mant++)
                for (sgn = 0; sgn < 2; sgn++) {
                    b.u = ((uint32_t)sgn << 31) | ((uint32_t)ee << 23) | mant;
                    a1_one(b.f, ps[pi]);
                }
        }
    }
}

/* ---- A2: fold_floor_shr (the shipped static, via the include) -------- */
static void a2_one(int64_t p, int sh) {
    int64_t got = fold_floor_shr(p, sh);
    CHK(2, got == ref_div(p, sh));
    if (sh < 60) CHK(2, got == ref_ld(p, sh));
}

static void a2(void) {
    static const int64_t edges[] = {
        0, 1, -1, 2, -2, 31, -31, 32, -32, 65534, -65534,
        1LL << 30, -(1LL << 30), 1LL << 31, -(1LL << 31),
        (1LL << 46) - 1, -((1LL << 46) - 1),
        1LL << 62, -(1LL << 62),
        (int64_t)0x7FFFFFFFFFFFFFFFLL, (int64_t)(-0x7FFFFFFFFFFFFFFFLL - 1)
    };
    int i, sh;
    int64_t p;
    for (sh = 0; sh <= 70; sh++)
        for (i = 0; i < (int)(sizeof edges / sizeof edges[0]); i++)
            a2_one(edges[i], sh);
    /* Dense: every p in [-2^20, 2^20] at every sh where a floor can go
       wrong at all, so every bucket boundary is present many times. */
    for (sh = 0; sh <= 24; sh++)
        for (p = -(1LL << 20); p <= (1LL << 20); p++)
            a2_one(p, sh);
}

/* ---- A3: sigmoid_q15_ti ---------------------------------------------- */
static int32_t sig_ref(int32_t v, int32_t m, int s) {
    long double p = (long double)v * (long double)m;
    long double t15f = floorl(p * powl(2.0L, (long double)(19 - s)))
                     + (long double)SIG_T15;
    if (t15f <= 0.0L) return 0;
    if (t15f >= (long double)((int64_t)SIG_N << 15)) return 32768;
    return sigmoid_q15_i((int32_t)(2.0L * (t15f - (long double)SIG_T15)), 20);
}

static void a3_one(int32_t v, int32_t m, int s) {
    CHK(3, sigmoid_q15_ti(v, m, s) == sig_ref(v, m, s));
}

static void a3(int32_t m_lo, int m_slo, int32_t m_hi, int m_shi) {
    static const int32_t ms[] = {
        0, 1, -1, 2, 1 << 23, (1 << 24) - 1, -((1 << 24) - 1),
        2147483647, (int32_t)(-2147483647 - 1)
    };
    static const int ss[] = { 0, 1, 18, 19, 20, 29, 31, 42, 50, 62 };
    static const int32_t vs[] = {
        0, 1, -1, 65534, -65534, 1 << 20, -(1 << 20),
        2147483647, (int32_t)(-2147483647 - 1)
    };
    int i, j, k;
    int32_t v;
    /* The whole producer domain, at both checkpoints' real folds. */
    for (v = -65534; v <= 65534; v++) {
        a3_one(v, m_lo, m_slo);
        a3_one(v, m_hi, m_shi);
    }
    /* The whole accepted domain's corners, crossed. */
    for (i = 0; i < (int)(sizeof ms / sizeof ms[0]); i++)
        for (j = 0; j < (int)(sizeof ss / sizeof ss[0]); j++)
            for (k = 0; k < (int)(sizeof vs / sizeof vs[0]); k++)
                a3_one(vs[k], ms[i], ss[j]);
    /* A power-of-two m must reproduce sigmoid_q15_i exactly: two shipped
       entries, two code paths, one answer. */
    for (v = -65534; v <= 65534; v += 7) {
        int e;
        for (e = 20; e <= 40; e++)
            CHK(3, sigmoid_q15_ti(v, 1 << 10, e) == sigmoid_q15_i(v, e - 10));
    }
}

/* ---- A4: lz_exp_t ----------------------------------------------------- */
static int64_t a4_pmax;
static int32_t a4_sqmax, a4_prodmax;
static int a4_sq32767, a4_idx31, a4_pext, a4_neg;

static void a4_one(int32_t v, int32_t m, int s) {
    long double y32, p;
    int64_t t;
    int32_t n, sq, prod;
    int idx;
    float got = lz_exp_t(v, m, s);
    float want;
    coord_ref(v, m, s, &y32, &t, &n, &idx, &sq);
    if (t > 4064) want = FLT_MAX;
    else if (t < -4031) want = 0.0f;
    else {
        prod = taylor_prod(idx, sq);
        want = (float)prod * pow2ref(n - EXP_Q);
        if (sq > a4_sqmax) a4_sqmax = sq;
        if (prod > a4_prodmax) a4_prodmax = prod;
        if (sq == 32767) a4_sq32767 = 1;
        if (idx == 31) a4_idx31 = 1;
        if (t < 0) a4_neg = 1;
        CHK(4, sq >= 0 && sq <= 32767);
        CHK(4, prod > 0 && prod < (1 << 22));
        CHK(4, idx >= 0 && idx < 32);
        CHK(4, n >= -126 && n <= 127);
    }
    CHK(4, got == want);
    p = (long double)v * (long double)m;
    if (p > (long double)a4_pmax) a4_pmax = (int64_t)p;
    if (p >= 4611686018427387904.0L || p <= -4611686018427387904.0L)
        a4_pext = 1;
    CHK(4, p <= 4611686018427387904.0L && p >= -4611686018427387904.0L);
}

static void a4(int32_t em, int es) {
    static const int32_t ms[] = { 0, 1, -1, 1 << 23, (1 << 24) - 1,
                                  -((1 << 24) - 1), 2147483647,
                                  (int32_t)(-2147483647 - 1) };
    static const int ss[] = { 0, 1, 14, 15, 16, 31, 40, 61, 62 };
    static const int32_t vs[] = { 0, 1, -1, 32768, -32768, 2147483647,
                                  (int32_t)(-2147483647 - 1) };
    int32_t v;
    int i, j, k;
    /* Reachable: s15 runs 0..32768 and nothing else calls this entry. */
    for (v = 0; v <= 32768; v++) a4_one(v, em, es);
    /* A dense walk of every table index and every corner of the
       residual, on both sides of zero. Stopped short of the guard
       window on purpose: the window is A5's row, and a sweep that
       crossed it would make A5's mutation redden this row too. */
    for (v = -4000 * 32; v <= 4000 * 32; v++) a4_one(v, 1 << 20, 25);
    /* The accepted domain's extremes - the products an int32
       accumulator could not hold, and both ends of the s band. */
    for (i = 0; i < (int)(sizeof ms / sizeof ms[0]); i++)
        for (j = 0; j < (int)(sizeof ss / sizeof ss[0]); j++)
            for (k = 0; k < (int)(sizeof vs / sizeof vs[0]); k++)
                a4_one(vs[k], ms[i], ss[j]);
    CHK(4, a4_pext == 1);
    CHK(4, a4_idx31 == 1);
    CHK(4, a4_sq32767 == 1);
    CHK(4, a4_neg == 1);
    CHK(4, a4_pmax == 4611686018427387904LL);
}

/* ---- A5: the guards --------------------------------------------------- */
static void a5(int32_t em) {
    int s = 20;
    int32_t mm = 1 << 20;
    CHK(5, lz_exp_t(4065, 1, 0) == FLT_MAX);
    CHK(5, lz_exp_t(4065, mm, s) == FLT_MAX);
    CHK(5, lz_exp_t(4064, mm, s) < FLT_MAX && lz_exp_t(4064, mm, s) > 0.0f);
    CHK(5, lz_exp_t(-4032, mm, s) == 0.0f);
    /* One step inside the low edge the value is still built rather than
       saturated - it is pow2f that flushes it, not the guard. */
    CHK(5, lz_exp_t(-4031, mm, s) == 0.0f);
    /* s outside the band is refused, and the refusal is the documented
       one rather than an out-of-range shift. */
    CHK(5, lz_exp_t(1, em, -1) == 0.0f);
    CHK(5, lz_exp_t(1, em, FOLD_SMAX + 1) == 0.0f);
    CHK(5, lz_exp_t(1, em, 1000) == 0.0f);
}

/* ---- A6: end-to-end accuracy ------------------------------------------ */
static void a6(int32_t em, int es) {
    /* Bound, DERIVED from the five roundings the value passes through,
       each at its worst case:
         Taylor truncation   a^3/6, a = ln2/32          1.69e-6
         g_exptab rounding   0.5 / 2^20                 4.77e-7
         cq's two shifts     2 x 0.5 / 2^20             9.54e-7
         sq rounding         ln2 * 0.5 * 2^-20          3.31e-7
         prod rounding       0.5 / 2^20                 4.77e-7
       Sum 3.93e-6; the cap is 4.5e-6, 15% of slack and not a fitted
       number, and the Taylor term alone is 43% of it. `(float)prod *
       pow2f` adds nothing: prod < 2^22 converts exactly and the scale
       is a power of two. lz_exp_fixed's OWN error on the same sweep is
       measured beside it, which is what says the budget belongs to the
       fixed exp tier and not to this entry. */
    const long double cap = 4.5e-6L;
    const long double log2e32 = 1.44269504088896340736L * 32.0L;
    int32_t v;
    long double worst = 0.0L, worst_fixed = 0.0L;
    for (v = 0; v <= 32768; v++) {
        long double y32 = (long double)v * (long double)em
                        * powl(2.0L, (long double)-es);
        long double want = expl(y32 / log2e32);
        long double rel, relf;
        if (want <= 0.0L) continue;
        rel = fabsl((long double)lz_exp_t(v, em, es) / want - 1.0L);
        if (rel > worst) worst = rel;
        CHK(6, rel <= cap);
        relf = fabsl((long double)lz_exp_fixed((float)(double)(y32 / log2e32))
                     / want - 1.0L);
        if (relf > worst_fixed) worst_fixed = relf;
    }
    /* Printed as double: this host's printf has no long double support
       and prints garbage for %Le. The comparison above is long double;
       only the display narrows. */
    printf("   A6 worst relative error %.4e (cap %.1e);"
           "  lz_exp_fixed on the same sweep %.4e\n",
           (double)worst, (double)cap, (double)worst_fixed);
}

/* ---- A7: the fold carries log2(e)*32 ---------------------------------- */
static void a7(void) {
    /* Loose ON PURPOSE. The two entries agree to within their combined
       A6 budgets, about 9e-6, but what this row is looking for is a
       coordinate MISSING the log2(e)*32 factor, which turns exp(-5)
       into exp(-0.108) - a factor of 134. A cap near the ulp level
       would only re-test A6 and would go red for any change to the
       shared Taylor, which is A6's subject and not this one's. */
    const long double tol = 1.0e-3L;
    static const float as[] = { -5.0f, -1.0f, -0.25f, 0.125f, 1.0f };
    static const int ps[] = { -15, -9, 0 };
    int i, j;
    int32_t v;
    for (i = 0; i < (int)(sizeof as / sizeof as[0]); i++)
        for (j = 0; j < (int)(sizeof ps / sizeof ps[0]); j++) {
            int32_t m = 0;
            int s = 0;
            if (!lz_exp_t_fold(as[i], ps[j], &m, &s)) continue;
            for (v = 0; v <= 32768; v += 13) {
                float x = (float)((double)v * (double)as[i]
                                  * ldexp(1.0, ps[j]));
                long double a = (long double)lz_exp_t(v, m, s);
                long double b = (long double)lz_exp_fixed(x);
                if (b <= 0.0L || a >= (long double)FLT_MAX) continue;
                CHK(7, fabsl(a / b - 1.0L) <= tol);
            }
        }
}

int main(void) {
    static const char *names[8] = { "", "A1 FOLD", "A2 FLOOR", "A3 SIGT",
                                    "A4 EXPT", "A5 GUARD", "A6 ACC",
                                    "A7 XLOG" };
    int32_t em = 0, sm_lo = 0, sm_hi = 0;
    int es = 0, s_lo = 0, s_hi = 0;
    int bad = 0, i;

    tab_build();

    /* The folds both checkpoints build. lower_bound is -5.0 on each and
       decay_base runs 0.001369..10.584328 (.prof/m5range.c); the two
       extremes are what bracket the per-head exponent. */
    if (!lz_exp_t_fold(-5.0f, -15, &em, &es)) {
        printf("FATAL: the shipping exp fold was refused\n");
        return 1;
    }
    if (!lz_fold_f32(0.001369f, -9, &sm_lo, &s_lo) ||
        !lz_fold_f32(10.584328f, -9, &sm_hi, &s_hi)) {
        printf("FATAL: a shipping sigmoid fold was refused\n");
        return 1;
    }
    printf("folds: exp m=%d s=%d   sig lo m=%d s=%d   sig hi m=%d s=%d\n",
           em, es, sm_lo, s_lo, sm_hi, s_hi);

    a1();
    a2();
    a3(sm_lo, s_lo, sm_hi, s_hi);
    a4(em, es);
    a5(em);
    a6(em, es);
    a7();

    printf("   A4 attained: |p| max %lld (2^62 seen %d), sq max %d"
           " (32767 seen %d), prod max %d, idx 31 seen %d, t<0 seen %d\n",
           (long long)a4_pmax, a4_pext, (int)a4_sqmax, a4_sq32767,
           (int)a4_prodmax, a4_idx31, a4_neg);

    for (i = 1; i <= 7; i++) {
        printf("%-9s %12ld cases  %ld mismatches  %s\n",
               names[i], cases[i], fails[i], fails[i] ? "FAIL" : "ok");
        if (fails[i]) bad = 1;
    }
    printf("%s\n", bad ? "VERIFY: FAIL" : "VERIFY: PASS");
    return bad;
}
