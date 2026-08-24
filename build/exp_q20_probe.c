/* Does the SSE2 Q20 Taylor agree with lz_exp_fixed's own arithmetic?
 *
 * prod = (tab * cq + 2^19) >> 20 with
 * cq = 2^20 + ((LN2*s + 2^19) >> 20) + ((LN2SQ*s*s + 2^39) >> 40).
 *
 * A pure function of two int32 arrays, so it answers without a model
 * and without the engine linking. The float front end that produces s
 * and the table index is the caller's and is not what is under test.
 *
 * THE REFERENCE IS WRITTEN AS THE C WRITES IT - ((int64)LN2SQ * s) * s -
 * and the kernel groups it as LN2SQ * (s*s). Those are the same number
 * only because every intermediate stays inside int64, which is exactly
 * the claim the kernel rests on, so the probe compares against the
 * ORIGINAL association rather than the regrouped one. Agreeing with a
 * regrouped reference would prove nothing about the regrouping.
 *
 * s's REAL RANGE IS [0, 2^15] - it is rq >> 5 with rq at most 2^20 -
 * and that bound is what makes s*s fit int32. The sweep drives the top
 * of it deliberately: at s = 2^15 the second term is at its largest and
 * a regrouping that overflowed would show there first.
 */
#include <stdio.h>
#include "lz_int.h"

void lz_exp_q20_simd(const int32_t *tab, const int32_t *s,
                     int32_t *prod, int n);

static unsigned long rs = 1145141919UL;
static int32_t nextv(void) {
    rs = rs * 1103515245UL + 12345UL;
    return (int32_t)(rs >> 1);
}

/* lz_exp_fixed's own lines, verbatim in their original association. */
static int32_t ref(int32_t tab, int32_t s) {
    int32_t cq = (1 << 20)
        + (int32_t)((((lz_i64)726817 * s) + (1 << 19)) >> 20)
        + (int32_t)((((lz_i64)251906 * s * s) + (LZ_I64_C(1) << 39)) >> 40);
    return (int32_t)((((lz_i64)tab * cq) + (1 << 19)) >> 20);
}

#define N 64

int main(void) {
    int32_t tab[N], s[N], got[N];
    long bad = 0, n = 0;
    int k, pass;

    static const int32_t se[] = { 0, 1, 2, 1023, 1024, 32766, 32767, 32768 };
    /* g_exptab spans exp() over [0,1) times 2^20, so [2^20, 2^21). */
    static const int32_t te[] = { 1048576, 1048577, 1500000, 2097151, 2097152 };
    const int ns = 8, nt = 5;

    for (pass = 0; pass < 20000 + ns * nt; pass++) {
        for (k = 0; k < N; k++) {
            if (pass < ns * nt) {
                s[k]   = se[((pass / nt) + k) % ns];
                tab[k] = te[(pass + k) % nt];
            } else {
                s[k]   = nextv() % 32769;
                tab[k] = 1048576 + (nextv() % 1048577);
            }
        }
        lz_exp_q20_simd(tab, s, got, N);
        for (k = 0; k < N; k++, n++) {
            int32_t want = ref(tab[k], s[k]);
            if (got[k] != want) {
                if (bad < 6)
                    printf("  tab=%8ld s=%6ld want %10ld got %10ld\n",
                           (long)tab[k], (long)s[k],
                           (long)want, (long)got[k]);
                bad++;
            }
        }
    }

    /* The tail: n not a multiple of four runs in C and must neither
       differ nor write past its end. */
    for (k = 0; k < N; k++) { s[k] = k * 4096; tab[k] = 1048576 + k * 1000; }
    for (k = 1; k <= 3; k++) {
        int j;
        int32_t t[N];
        for (j = 0; j < N; j++) t[j] = -1;
        lz_exp_q20_simd(tab, s, t, k);
        for (j = 0; j < k; j++, n++)
            if (t[j] != ref(tab[j], s[j])) {
                if (bad < 6) printf("  tail n=%d lane %d\n", k, j);
                bad++;
            }
        if (t[k] != -1) { printf("  tail n=%d wrote past the end\n", k); bad++; }
    }

    printf("exp_q20_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
}
