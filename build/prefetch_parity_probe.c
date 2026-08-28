/* Does the prefetch knob change any RESULT?
 *
 * --prefetch picks between four arms - none, load, nta, 3dnow - and each
 * arm runs the same accumulation loop with a different prefetch
 * instruction (LZ_Q41_PFSEL and its siblings in
 * src/ops_kernel_dot_shared.h). So "the mode changes timing and not
 * values" is structurally near-certain, and it was never checked. What
 * would break it is not the prefetch instruction, which has no
 * architectural effect: it is one arm having a different loop bound, a
 * different DOT32, or a pf_ guard that reads past `wend` - each of which
 * lives in the arm rather than in the shared body, and each of which
 * only one mode would show.
 *
 * build/kernel_probe_census.sh is why this exists. It reports the three
 * lz_pf_* pragmas as reached by no probe, and they are the one gap of
 * its five that a probe can close at all - the other two read CPUID.
 * They compute no value, so they cannot be cross-checked against a twin
 * the way a dot kernel is; what CAN be checked is the property the
 * engine relies on when it offers the knob, which is this one.
 *
 * THE MODES ARE NOT ALL AVAILABLE. lz_prefetch_select refuses a mode the
 * CPU cannot do (3dnow off an AMD, nta without SSE) and falls back,
 * returning the name it actually set. So the probe records the name it
 * GOT rather than the one it asked for, compares only distinct settings,
 * and fails if fewer than two were distinct - one mode agreeing with
 * itself is not a parity check, and on a machine that offers only one
 * this file must say so rather than print a green.
 *
 * The seed is fixed at 1145141919.
 */
#include <stdio.h>
#include <string.h>
#include "lz_int.h"
#include "ops.h"
#include "ops_quant.h"
#include "ops_kernel_shared.h"
#include "ops_matmul.h"

#define MAXDIM 256
#define MAXNT  2

static unsigned long rng_state;
static void rng_seed(unsigned long s) { rng_state = s; }
static unsigned long rng_next(void) {
    rng_state = (rng_state * 48271u) % 2147483647u;
    return rng_state;
}

/* Every populated slot of every format, hashed. The knob is read inside
   the row kernels, so the dispatch tables are what has to be driven -
   a single dot kernel never sees it. */
static unsigned long sweep(void) {
    static unsigned char w4buf[MAXDIM * MAXNT + 16];
    unsigned char *w4 = (unsigned char *)(((size_t)w4buf + 15) & ~(size_t)15);
    static unsigned char w2[MAXDIM * MAXNT];
    /* ALIGNED, like w4: the 128-wide group kernels load the ACTIVATIONS
       with movdqa too, not just the weights. gcc aligns a static array
       of this size to 16 and hides it; Watcom aligns to 8, and the probe
       faulted at nb%%4==0 - the only shape that reaches those kernels. */
    static int16_t xwbuf[MAXDIM * MAXNT + 8];
    int16_t *xw = (int16_t *)(((size_t)xwbuf + 15) & ~(size_t)15);
    static int32_t acc[MAXDIM * MAXNT];
    static const int DIMS[] = { 32, 64, 96, 128, 160 };
    static const struct { const char *n; const lz_rowfn *t; } FMT[] = {
        { "q8_0",  LZ_ROW_Q8  }, { "q4_1",  LZ_ROW_Q41 },
        { "q6_1",  LZ_ROW_Q61 }, { "q16_0", LZ_ROW_Q16 },
        { "t2",    LZ_ROW_T2  }
    };
    unsigned long h = 2166136261UL;
    int f, d, nt, s, i;

    rng_seed(1145141919UL);
    for (i = 0; i < MAXDIM * MAXNT; i++) {
        unsigned long r = rng_next();
        w4[i] = (unsigned char)((r & 0x0Fu) | (((r >> 5) & 0x0Fu) << 4));
        if ((w4[i] & 0x0Fu) == (w4[i] >> 4)) w4[i] ^= 0x10u;
        w2[i] = (unsigned char)(r >> 9);
        xw[i] = (int16_t)((int)(r % 511u) - 255);
    }
    for (f = 0; f < (int)(sizeof FMT / sizeof FMT[0]); f++)
        for (d = 0; d < (int)(sizeof DIMS / sizeof DIMS[0]); d++)
            for (nt = 1; nt <= MAXNT; nt++)
                for (s = 0; s < LZ_ROW_N; s++) {
                    lz_row_ctx rctx;
                    int nb = DIMS[d] / 32;
                    if (!FMT[f].t[s]) continue;
                    memset(acc, 0, sizeof acc);
                    rctx.acc32 = acc; rctx.w4 = w4; rctx.w2 = w2;
                    rctx.xw = xw; rctx.nb = nb; rctx.nt = nt;
                    rctx.in_dim = DIMS[d];
                    /* THE END OF THE WEIGHT BUFFER, which is what
                       ops_matmul.c passes (rctx.pf_end = wend, the end
                       of the same wr the row reads). Pointing it at a
                       different array makes `pf_ < wend` a comparison
                       between unrelated objects, and the group kernels
                       then prefetch from wherever that lands: this
                       probe crashed under Watcom at nb%4==0 until the
                       bound was right. .prof/rowdot_xcheck.c has the
                       same mistake and has never met it, because none
                       of its four sizes is a multiple of 128. */
                    rctx.pf_end = w4 + MAXDIM * MAXNT;
                    rctx.pf_end2 = w2 + MAXDIM * MAXNT;
                    FMT[f].t[s](&rctx);
                    for (i = 0; i < nb * nt; i++) {
                        unsigned u;
                        memcpy(&u, &acc[i], 4);
                        h ^= u; h *= 16777619UL; h &= 0xFFFFFFFFUL;
                    }
                }
    return h;
}

int main(void) {
    static const char *ASK[] = { "none", "load", "nta", "3dnow" };
    const char *got[4];
    unsigned long h[4];
    int i, j, distinct = 0, bad = 0;

    for (i = 0; i < 4; i++) {
        const char *g = lz_prefetch_select(ASK[i]);
        got[i] = g ? g : "?";
        h[i] = sweep();
        printf("  asked %-6s got %-6s hash %08lX\n", ASK[i], got[i], h[i]);
    }

    /* Distinct SETTINGS, not distinct requests: a machine without 3DNow
       answers "nta" to both, and comparing that pair proves nothing. */
    for (i = 0; i < 4; i++) {
        int seen = 0;
        for (j = 0; j < i; j++)
            if (strcmp(got[i], got[j]) == 0) seen = 1;
        if (!seen) distinct++;
    }
    for (i = 1; i < 4; i++)
        if (h[i] != h[0]) {
            printf("  MISMATCH mode %s differs from %s: %08lX vs %08lX\n",
                   got[i], got[0], h[i], h[0]);
            bad++;
        }

    if (distinct < 2) {
        printf("prefetch_parity: FAIL - only %d distinct mode(s) available"
               " here, so nothing\n  was compared. This is not a pass:"
               " one mode agrees with itself.\n", distinct);
        return 1;
    }
    if (bad) {
        printf("prefetch_parity: FAIL - the knob changed %d result(s)\n", bad);
        return 1;
    }
    printf("prefetch_parity: PASS (%d distinct mode(s), identical results)\n",
           distinct);
    return 0;
}
