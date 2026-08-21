/* Can the dequant epilogue be fixed point? The question before the work.
 *
 * With --attn-write fixed it is 66.4% of the f32 operations per token
 * (.prof census, ctx 1440), which makes it the next item - and the most
 * expensive one to build, since it means changing how scales propagate.
 * So measure the thing that decides feasibility BEFORE building, the way
 * .prof/wconv.c did for the conv tier.
 *
 * WHAT DECIDES IT. epi_q8 computes, per output element,
 *
 *     o = post * SUM_g  acc32[g] * xs[g] * ws[g]
 *
 * A fixed-point form needs one shared exponent for the whole sum, the
 * same trick the conv tier uses. That works only if the per-group
 * products xs[g]*ws[g] have a NARROW dynamic range within one row: the
 * shared exponent has to cover the largest, and every smaller group
 * loses bits proportionally. So the number to get is
 *
 *     max_g (xs[g]*ws[g]) / min_g (xs[g]*ws[g])
 *
 * per (tensor, output row), over real activations and real weights.
 * A ratio near 1 means a shared exponent is nearly free; 2^15 means the
 * smallest group is quantized to nothing and the idea is dead as stated.
 *
 * xs is per (token, input group) and ws is per (row, weight group), so
 * the range has both an activation part and a weight part. They are
 * reported separately: if the weight side alone is already wide, no
 * amount of per-token work fixes it.
 *
 * Build:  .prof/build.sh wepirange
 * Usage:  wepirange <model-dir> <tokens-file> [max-tensors]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ops.h"
#include "model.h"
#include "forward.h"

static void tap_act(const char *tag, int li, const float *p, int n);
#define LZ_TAP(tag, li, ptr, n) tap_act((tag), (li), (ptr), (n))

#include "ops.c"
#include "err.c"
#include "compat.c"
#include "json.c"
#include "safetensors.c"
#include "model.c"
#include "forward.c"

/* Histogram of log2 of the ratio, because the interesting answer is an
   exponent count, not a mean: "how many bits does the shared exponent
   cost the worst group in this row". */
#define NBUCK 24
static long hist_w[NBUCK], hist_x[NBUCK], hist_both[NBUCK];

static void bump(long *h, double ratio) {
    int b = 0;
    if (!(ratio > 1.0)) { h[0]++; return; }
    b = (int)(log(ratio) / log(2.0));
    if (b < 0) b = 0;
    if (b >= NBUCK) b = NBUCK - 1;
    h[b]++;
}

/* "an" is the normed hidden state fed to the projections - the vector
   whose per-32 group scales become epi_q8's xs[]. No engine change
   needed: forward.c already taps it. */
static void tap_act(const char *tag, int li, const float *p, int n) {
    int g, k, ng = n / 32;
    double lo = 1e30, hi = 0.0;
    (void)li;
    if (strcmp(tag, "an") != 0 || ng < 2) return;
    for (g = 0; g < ng; g++) {
        double amax = 0.0;
        for (k = 0; k < 32; k++) {
            double a = fabs((double)p[g * 32 + k]);
            if (a > amax) amax = a;
        }
        if (amax <= 0.0) continue;
        if (amax < lo) lo = amax;
        if (amax > hi) hi = amax;
    }
    if (hi > 0.0 && lo < 1e30) bump(hist_x, hi / lo);
}

static void report(const char *name, const long *h) {
    long tot = 0, cum = 0;
    int b;
    for (b = 0; b < NBUCK; b++) tot += h[b];
    if (!tot) { printf("  %-22s (no samples)\n", name); return; }
    printf("  %-22s", name);
    for (b = 0; b < NBUCK; b++) {
        cum += h[b];
        if (h[b] && (b < 6 || cum < tot)) printf(" 2^%d:%.1f%%", b,
                                                 100.0 * (double)h[b] / (double)tot);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    LZModel m;
    LZRunState st;
    char err[256];
    int *toks = NULL, nt = 0, cap = 0, i, li, maxt;
    FILE *f;

    if (argc < 3) { printf("usage: wepirange <dir> <tokens-file> [max]\n"); return 2; }
    maxt = (argc > 3) ? atoi(argv[3]) : 4;

    f = fopen(argv[2], "r");
    if (!f) { printf("cannot read %s\n", argv[2]); return 2; }
    for (;;) {
        int v;
        if (fscanf(f, "%d", &v) != 1) break;
        if (nt == cap) { cap = cap ? cap * 2 : 1024;
                         toks = (int *)realloc(toks, (size_t)cap * sizeof(int)); }
        toks[nt++] = v;
        fgetc(f);
    }
    fclose(f);
    if (nt > 16) nt = 16;          /* activations, not context length */

    if (lz_open(&m, argv[1], err, sizeof(err)) != 0) { printf("open: %s\n", err); return 1; }
    if (lz_read_weights(&m, err, sizeof(err)) != 0) { printf("weights: %s\n", err); return 1; }
    if (lz_state_alloc(&st, &m, nt + 8, 0, err, sizeof(err)) != 0) { printf("state: %s\n", err); return 1; }

    for (i = 0; i < nt; i++) lz_forward(&m, &st, toks[i], i);

    printf("epilogue scale range, %s\n", argv[1]);
    printf("ratio = max_g(xs*ws) / min_g(xs*ws) within one output row\n");
    printf("buckets are log2 of that ratio: 2^0 means a shared exponent is free\n\n");

    /* Weight side first, straight from the tensors: it needs no forward
       pass and it alone can kill the idea. */
    for (li = 0; li < m.config.n_layers && li < maxt; li++) {
        const LZLayer *L = &m.layers[li];
        const LZTensor *tn[4];
        const char *nm[4];
        int ind[4];
        int k, ti = 0;
        /* in_dim is the caller's knowledge, not the tensor's - LZTensor
           carries only an element count. Taking it from the config is
           what forward.c does at every call site. */
        tn[ti] = &L->q_proj;  ind[ti] = m.config.hidden_size;  nm[ti++] = "q_proj";
        tn[ti] = &L->o_proj;  ind[ti] = m.config.attn_q_dim;   nm[ti++] = "o_proj";
        for (k = 0; k < ti; k++) {
            const LZTensor *w = tn[k];
            int gs, ngw, row, rows, in_dim = ind[k];
            if (!w->q || !w->scale || w->gs <= 0 || in_dim <= 0) continue;
            gs = w->gs;
            ngw = in_dim / gs;
            if (ngw < 2) continue;
            rows = w->n / in_dim;
            for (row = 0; row < rows; row++) {
                const float *ws = w->scale + (size_t)row * ngw;
                double lo = 1e30, hi = 0.0;
                int g;
                for (g = 0; g < ngw; g++) {
                    double v = fabs((double)ws[g]);
                    if (v <= 0.0) continue;
                    if (v < lo) lo = v;
                    if (v > hi) hi = v;
                }
                if (hi > 0.0 && lo < 1e30) bump(hist_w, hi / lo);
            }
            if (li == 0) printf("  (%s: gs %d, %d groups, %d rows)\n",
                                nm[k], gs, ngw, rows);
        }
    }
    report("weight scales alone", hist_w);
    (void)hist_both;
    report("activation scales", hist_x);
    lz_state_free(&st);
    return 0;
}
