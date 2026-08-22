/* sigmoid_q15_i's wide band (e >= 20), exhaustively.
 *
 * The claim the fixed conv1d epilogue now rests on is that
 *
 *     sigmoid_q15_i(v, e) == table_lerp(floor(t * 2^15)),
 *     t * 2^15 == 192*2^15 + v * 2^(19-e)
 *
 * EXACTLY, for every int32 v and every e the conv exponents reach
 * (measured 22..39 on kmr20 and kunmoe-v2-t2-probe). Nothing here is a
 * tolerance: the right-hand side is an integer computed in long long,
 * and one mismatch falsifies it.
 *
 * FOUR ROWS, FOUR THINGS. Written apart so that one surgical mutation
 * of the implementation reddens ONE of them:
 *
 *   WIDE-NEG   the biased-unsigned floor for v < 0. This is the only
 *              novel piece - `>>` on a negative int is implementation-
 *              defined, so the shift runs on (uint32_t)v + 2^31 and
 *              subtracts 2^(31-s) back.
 *   WIDE-POS   the same for v >= 0, where the bias is inert and what is
 *              being tested is the coordinate constant and the shift.
 *   NARROW     e <= 19, the untouched pre-existing body, against the
 *              SAME reference. m5sig.c only ever compared it to the
 *              float entry, which is exact only up to e = 19 and is
 *              therefore no reference at all above it; this row is the
 *              first check that the narrow body computes the same
 *              function the wide one does, which is what makes the two
 *              a single function with a seam at 19/20.
 *   OUTSIDE    e outside [0, 50] falls back to the float entry
 *              unchanged.
 *
 * EXHAUSTIVE, and for the wide rows that word is used strictly. For a
 * given e the implementation reads v only through (uu >> s), so every
 * distinct input class is a bucket of 2^s consecutive v; both EDGES of
 * every bucket are tested, over the whole int32 range. At s = 1 that is
 * the full 2^32 domain; at s = 8 it is 2^25 of it, chosen so that the
 * bucket-boundary cases - the only places a floor can go wrong - are
 * all present twice. e >= 28 additionally runs against the slow
 * division-based reference rather than the shift-based one.
 *
 * TWO REFERENCES, because the fast one shares a primitive with nothing
 * in the implementation but does assume gcc's `>>` on a negative long
 * long is arithmetic: ref_shift uses that, ref_div uses `/` and `%` and
 * corrects the truncation, and A0 requires them to agree everywhere
 * ref_div is affordable. Neither uses the implementation's bias trick.
 *
 * Build:
 *   gcc -O2 -std=gnu99 -Isrc -o .prof/verify_sig_i_wide.exe \
 *       .prof/verify_sig_i_wide.c src/ops_quant.c src/ops_sched.c \
 *       src/err.c src/compat.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ops_quant.h"

#define SIG_N     384
#define SIG_T15   6291456L            /* 192 << 15 */
#define SIG_TOP   ((long long)SIG_N << 15)

static int32_t tab[SIG_N + 1];

/* The table, read out of the shipped entry rather than rebuilt: at a
   node x = -12 + i/16 (exact in float, four fractional bits) the
   coordinate t is exactly i and frac is 0, so sigmoid_q15 returns
   g_sigtab[i] itself. The two ends come back through clamps instead,
   so they are taken from just inside and checked against the clamp
   values - if sig_build ever stops making those equal, this fails here
   rather than silently giving the reference a different table. */
static int build_table(void) {
    int i, ok = 1;
    for (i = 1; i < SIG_N; i++)
        tab[i] = sigmoid_q15(-12.0f + (float)i / 16.0f);
    tab[0] = sigmoid_q15_t(1.0f / 65536.0f);   /* idx 0, frac 0 */
    tab[SIG_N] = sigmoid_q15_t((float)SIG_N);  /* returns g_sigtab[N] */
    if (tab[0] != sigmoid_q15(-12.0f)) {
        printf("table: g_sigtab[0] %d != lower clamp %d\n",
               (int)tab[0], (int)sigmoid_q15(-12.0f));
        ok = 0;
    }
    if (tab[SIG_N] != 32768) {
        printf("table: g_sigtab[%d] %d != upper clamp 32768\n",
               SIG_N, (int)tab[SIG_N]);
        ok = 0;
    }
    for (i = 0; i < SIG_N; i++)
        if (tab[i + 1] < tab[i]) { printf("table: not monotone at %d\n", i); ok = 0; }
    return ok;
}

/* floor(a / 2^s), s >= 0, by truncating division plus a correction.
   No shift of a negative value anywhere. */
static long long ref_div(long long a, int s) {
    long long d = (long long)1 << s;
    long long q = a / d;
    if (a % d != 0 && a < 0) q--;
    return q;
}

/* The same, by arithmetic shift. gcc's >> on a negative long long is
   arithmetic, which IS floor; A0 checks that against ref_div rather
   than trusting it. */
static long long ref_shift(long long a, int s) { return a >> s; }

static int32_t ref_from_t15(long long t15) {
    long long idx, frac;
    int32_t a, b;
    if (t15 <= 0) return 0;
    if (t15 >= SIG_TOP) return 32768;
    idx = t15 >> 15;
    frac = t15 & 32767;
    a = tab[idx];
    b = tab[idx + 1];
    return a + (int32_t)(((long long)(b - a) * frac) >> 15);
}

/* The reference value. e >= 19 floors, e < 19 scales up - both exact in
   long long (|v| * 2^19 < 2^50). */
static int32_t ref(int32_t v, int e, int use_div) {
    long long t15;
    if (e >= 19)
        t15 = SIG_T15 + (use_div ? ref_div((long long)v, e - 19)
                                 : ref_shift((long long)v, e - 19));
    else
        t15 = SIG_T15 + ((long long)v << (19 - e));
    return ref_from_t15(t15);
}

static int fails;
static void report(const char *row, long long swept, long long bad,
                   int32_t bv, int be, int32_t got, int32_t want) {
    printf("%-10s swept %-14lld mismatches %-12lld %s\n", row, swept, bad,
           bad ? "FAIL" : "ok");
    if (bad) {
        printf("           first: v=%ld e=%d  got %ld want %ld\n",
               (long)bv, be, (long)got, (long)want);
        fails++;
    }
}

int main(void) {
    int e;
    long long swept_neg = 0, swept_pos = 0, swept_nar = 0, swept_out = 0;
    long long bad_neg = 0, bad_pos = 0, bad_nar = 0, bad_out = 0;
    int32_t fv_neg = 0, fv_pos = 0, fv_nar = 0, fv_out = 0;
    int fe_neg = 0, fe_pos = 0, fe_nar = 0, fe_out = 0;
    int32_t fg_neg = 0, fw_neg = 0, fg_pos = 0, fw_pos = 0;
    int32_t fg_nar = 0, fw_nar = 0, fg_out = 0, fw_out = 0;
    int clamp_lo = 0, clamp_hi = 0, clamp_lo_hi_e = 0;

    if (!build_table()) { printf("verify_sig_i_wide: FAIL (table)\n"); return 1; }

    /* A0: the two references agree. Not a claim about the engine - a
       claim about this file, and the reason the fast one is allowed to
       carry the 8.6e9-case rows below. */
    {
        long i;
        long long n = 0, bad = 0;
        for (e = 19; e <= 50; e++) {
            for (i = -4000000; i <= 4000000; i += 7) {
                long long a = (long long)i;
                if (ref_div(a, e - 19) != ref_shift(a, e - 19)) bad++;
                n++;
            }
            /* the extremes, where a correction is most likely to differ */
            {
                long long ex[6];
                int j;
                ex[0] = -2147483648LL; ex[1] = 2147483647LL; ex[2] = -1;
                ex[3] = 0; ex[4] = 1; ex[5] = -2147483647LL;
                for (j = 0; j < 6; j++) {
                    if (ref_div(ex[j], e - 19) != ref_shift(ex[j], e - 19)) bad++;
                    n++;
                }
            }
        }
        printf("%-10s swept %-14lld mismatches %-12lld %s\n",
               "A0-REFS", n, bad, bad ? "FAIL" : "ok");
        if (bad) fails++;
    }

    /* WIDE-NEG / WIDE-POS: both edges of every bucket, every e in the
       wide band. e >= 28 uses the division reference. */
    for (e = 20; e <= 50; e++) {
        int s = e - 19;
        long long nb = (long long)1 << (32 - s);
        long long qb;
        int use_div = (e >= 28);
        int saw_lo = 0, saw_hi = 0;
        for (qb = 0; qb < nb; qb++) {
            long long base = (qb << s) - 2147483648LL;
            int which;
            for (which = 0; which < 2; which++) {
                int32_t v = (int32_t)(which ? base + (((long long)1 << s) - 1)
                                            : base);
                int32_t got = sigmoid_q15_i(v, e);
                int32_t want = ref(v, e, use_div);
                if (v < 0) {
                    swept_neg++;
                    if (got != want && !bad_neg++) {
                        fv_neg = v; fe_neg = e; fg_neg = got; fw_neg = want;
                    }
                } else {
                    swept_pos++;
                    if (got != want && !bad_pos++) {
                        fv_pos = v; fe_pos = e; fg_pos = got; fw_pos = want;
                    }
                }
                if (got == 0) saw_lo = 1;
                if (got == 32768) saw_hi = 1;
            }
        }
        if (saw_lo) clamp_lo++;
        if (saw_hi) clamp_hi++;
        if (e >= 28 && (saw_lo || saw_hi)) clamp_lo_hi_e++;
    }
    report("WIDE-NEG", swept_neg, bad_neg, fv_neg, fe_neg, fg_neg, fw_neg);
    report("WIDE-POS", swept_pos, bad_pos, fv_pos, fe_pos, fg_pos, fw_pos);

    /* NARROW: e <= 19, the whole clamped domain plus a shoulder on each
       side, plus the int32 extremes. */
    for (e = 0; e <= 19; e++) {
        long long lim = (long long)13 << e;      /* 12<<e is the edge */
        long long v;
        for (v = -lim - 2; v <= lim + 2; v++) {
            int32_t got = sigmoid_q15_i((int32_t)v, e);
            int32_t want = ref((int32_t)v, e, 1);
            swept_nar++;
            if (got != want && !bad_nar++) {
                fv_nar = (int32_t)v; fe_nar = e; fg_nar = got; fw_nar = want;
            }
        }
        {
            int32_t ex[4];
            int j;
            ex[0] = -2147483647 - 1; ex[1] = 2147483647;
            ex[2] = -1073741824; ex[3] = 1073741823;
            for (j = 0; j < 4; j++) {
                int32_t got = sigmoid_q15_i(ex[j], e);
                int32_t want = ref(ex[j], e, 1);
                swept_nar++;
                if (got != want && !bad_nar++) {
                    fv_nar = ex[j]; fe_nar = e; fg_nar = got; fw_nar = want;
                }
            }
        }
    }
    report("NARROW", swept_nar, bad_nar, fv_nar, fe_nar, fg_nar, fw_nar);

    /* OUTSIDE: past the domain the entry reconstructs a float and calls
       sigmoid_q15. ldexp, not pow2f, so the expectation does not come
       from the same helper the implementation uses. */
    for (e = -4; e <= 64; e++) {
        long i;
        if (e >= 0 && e <= 50) continue;
        for (i = -3000000; i <= 3000000; i += 11) {
            int32_t v = (int32_t)i;
            int32_t got = sigmoid_q15_i(v, e);
            int32_t want = sigmoid_q15((float)v * (float)ldexp(1.0, -e));
            swept_out++;
            if (got != want && !bad_out++) {
                fv_out = v; fe_out = e; fg_out = got; fw_out = want;
            }
        }
    }
    report("OUTSIDE", swept_out, bad_out, fv_out, fe_out, fg_out, fw_out);

    /* Not a row, a statement of fact the wide branch's comment makes and
       this run either confirms or does not: the two clamps are LIVE for
       20 <= e <= 27 and DEAD for e >= 28. */
    printf("clamps     e with a 0 result: %ld   with a 32768 result: %ld"
           "   any at e>=28: %ld\n", clamp_lo, clamp_hi, clamp_lo_hi_e);

    printf("\nverify_sig_i_wide: %s (%ld row%s red)\n",
           fails ? "FAIL" : "PASS", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
