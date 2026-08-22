/* int-pipeline milestone 8: what dynamic range do the SwiGLU pair's
 * inputs and output actually have, and can their producers take the
 * int16 exit at all?
 *
 * The three SwiGLU sites in forward.c are
 *   routed expert   w1 -> moe_h1 (p),  w3 -> moe_h3 (g)
 *   shared expert   moe_shared_gate (p), moe_shared_up (g)
 *   dense FFN       gate_proj (p), up_proj (g)
 * and all three compute silu(p) * g elementwise.
 *
 * An int16 exit needs a CONSTANT target exponent measured from the real
 * distribution (see LZ_CONV_ES / LZ_MLAT_ES in forward.c for the same
 * argument). This measures, per site and per role:
 *   - max |x| and a power-of-two histogram
 *   - for each candidate ES, how many elements would clamp at +-32767
 *     under epi_align_i16's half-away-from-zero rounding
 *   - for the OUTPUT only, the spread of per-group(32) amax against the
 *     whole-corpus max, which is what decides whether ONE fixed output
 *     exponent can feed lz_quantize_q8_i16 without costing the small
 *     groups their Q8 resolution
 *   - the fraction of |p| past the sigmoid table's edge (|x| = 12),
 *     where sigmoid_q15 already returns a constant
 *
 * Hooked by macro-replacing lz_matmul_xq / lz_matmul_xq_nt for the
 * translation of forward.c ONLY (the macros are defined after ops.c is
 * spliced in, so the real definitions are untouched). Each call is
 * classified by comparing the weight pointer against every layer's
 * tensors, so no tap has to be added to forward.c to run this.
 *
 * Build (one TU, like .prof/m5range.c):
 *   gcc -O1 -std=gnu99 -Isrc -o .prof/m8range.exe .prof/m8range.c \
 *       src/cpucheck.c -lm
 * Usage:  m8range <model-dir> <tokens-file>
 *
 * FIDELITY: no LZ_MMX_TU / LZ_SSE2_TU here, so the quantized matmuls
 * reach matmul_scalar_ref rather than the SIMD impls + fixed epilogue.
 * The last bits of the VALUES differ from the shipping build; the
 * RANGES this program reports do not.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCH   9                /* 3 sites x {p, g, out} */
#define NES   25               /* candidate ES: 0 .. NES-1 */
#define NB    48               /* histogram bucket = floor(log2|x|) + 32 */
#define NGR   40               /* group-amax ratio buckets, in powers of 2 */

static const char *chname[NCH] = {
    "routed p (w1)",       "routed g (w3)",        "routed out",
    "shared p (gate)",     "shared g (up)",        "shared out",
    "dense p (gate_proj)", "dense g (up_proj)",    "dense out"
};

static double gmax[NCH];
static long   nseen[NCH];
static long   hist[NCH][NB];
static long   clampn[NCH][NES];
static long   nsat[NCH];        /* |x| >= 12: past the sigmoid table edge */
static long   ngrp[NCH];
static double grmax[NCH][NGR];  /* per-group amax, bucketed later */
static double *gamax[NCH];      /* growable list of per-group amax */
static long    gacap[NCH], gan[NCH];

static void acc(int c, double v) {
    double a = fabs(v);
    int e, j;
    if (a > gmax[c]) gmax[c] = a;
    nseen[c]++;
    if (a >= 12.0) nsat[c]++;
    if (a == 0.0) e = 0;
    else {
        e = (int)floor(log(a) / log(2.0)) + 32;
        if (e < 0) e = 0;
        if (e >= NB) e = NB - 1;
    }
    hist[c][e]++;
    for (j = 0; j < NES; j++)
        if (a * pow(2.0, (double)j) + 0.5 >= 32768.0) clampn[c][j]++;
}

/* Per-group(32) amax of an output vector, kept for the spread report. */
static void acc_groups(int c, const float *o, int n) {
    int i, k;
    for (i = 0; i + 32 <= n; i += 32) {
        double m = 0.0;
        for (k = 0; k < 32; k++) {
            double a = fabs((double)o[i + k]);
            if (a > m) m = a;
        }
        if (gan[c] == gacap[c]) {
            gacap[c] = gacap[c] ? gacap[c] * 2 : 4096;
            gamax[c] = (double *)realloc(gamax[c], (size_t)gacap[c] * sizeof(double));
        }
        gamax[c][gan[c]++] = m;
        ngrp[c]++;
    }
}

#include "ops.h"
#include "model.h"
#include "forward.h"
#include "ops.c"
#include "ops_quant.c"
#include "ops_sched.c"
#include "ops_matmul.c"
#include "ops_t2_scalar.c"
#include "ops_norm.c"
#include "ops_gdn.c"
#include "fwht.c"
#include "err.c"
#include "compat.c"
#include "lfn.c"
#include "json.c"
#include "safetensors.c"
#include "model.c"

static LZModel *g_m;
static float *g_out;            /* scratch for the reconstructed product */
static int    g_outcap;

/* Which of the six roles this weight tensor is, or -1. Scanned rather
   than passed, so forward.c needs no instrumentation of its own. */
static int classify(const LZTensor *w) {
    int li, e;
    if (!g_m) return -1;
    for (li = 0; li < g_m->config.n_layers; li++) {
        LZLayer *L = &g_m->layers[li];
        if (L->moe_expert_w1)
            for (e = 0; e < g_m->config.num_experts; e++) {
                if (w == &L->moe_expert_w1[e]) return 0;
                if (w == &L->moe_expert_w3[e]) return 1;
            }
        if (w == &L->moe_shared_gate) return 3;
        if (w == &L->moe_shared_up)   return 4;
        if (w == &L->gate_proj)       return 6;
        if (w == &L->up_proj)         return 7;
    }
    return -1;
}

/* p arrives first at every site; cache it so the g call can rebuild the
   product the SwiGLU loop is about to compute. */
static const float *pbuf[9];
static int          pn[9];

static void m8_note(const float *o, const LZTensor *w, int out_dim, int nt) {
    int role = classify(w);
    int i, n = out_dim * nt;
    if (role < 0) return;
    for (i = 0; i < n; i++) acc(role, (double)o[i]);
    if (role == 0 || role == 3 || role == 6) { pbuf[role] = o; pn[role] = n; }
    else {
        int pr = role - 1, oc = role + 1;
        if (!pbuf[pr] || pn[pr] != n) return;
        if (n > g_outcap) {
            g_outcap = n * 2;
            g_out = (float *)realloc(g_out, (size_t)g_outcap * sizeof(float));
        }
        for (i = 0; i < n; i++) {
            float v = lz_silu(pbuf[pr][i]) * o[i];
            g_out[i] = v;
            acc(oc, (double)v);
        }
        acc_groups(oc, g_out, n);
        pbuf[pr] = NULL;
    }
}

static void m8_mm_nt(float *o, const float *x, const int8_t *xq,
                     const float *xqs, const LZTensor *w,
                     int in_dim, int out_dim, int nt) {
    lz_matmul_xq_nt(o, x, xq, xqs, w, in_dim, out_dim, nt);
    m8_note(o, w, out_dim, nt);
}
static void m8_mm(float *o, const float *x, const int8_t *xq,
                  const float *xqs, const LZTensor *w,
                  int in_dim, int out_dim) {
    lz_matmul_xq(o, x, xq, xqs, w, in_dim, out_dim);
    m8_note(o, w, out_dim, 1);
}

#define lz_matmul_xq_nt(o, x, q, s, w, i, od, nt) m8_mm_nt(o, x, q, s, w, i, od, nt)
#define lz_matmul_xq(o, x, q, s, w, i, od)        m8_mm(o, x, q, s, w, i, od)
#include "forward.c"
#undef lz_matmul_xq_nt
#undef lz_matmul_xq

static int *read_tokens(const char *path, int *n_out) {
    FILE *f = fopen(path, "rb");
    int cap = 1024, n = 0, v;
    int *t;
    if (!f) return NULL;
    t = (int *)malloc((size_t)cap * sizeof(int));
    while (fscanf(f, "%d", &v) == 1) {
        if (n == cap) { cap *= 2; t = (int *)realloc(t, (size_t)cap * sizeof(int)); }
        t[n++] = v;
    }
    fclose(f);
    *n_out = n;
    return t;
}

static void report_ok(LZModel *m) {
    LZLayer *L = &m->layers[0];
    int li, found = 0;
    printf("\n== int16 exit availability (lz_matmul_xq_i16_ok)\n");
    for (li = 0; li < m->config.n_layers && !found; li++) {
        L = &m->layers[li];
        if (L->moe_expert_w1 && L->moe_expert_w1[0].n) {
            printf("  layer %d moe_expert_w1[0] in=%d  ok=%d\n", li,
                   m->config.moe_latent_dim,
                   lz_matmul_xq_i16_ok(&L->moe_expert_w1[0],
                                       m->config.moe_latent_dim, 1));
            printf("  layer %d moe_expert_w3[0] in=%d  ok=%d\n", li,
                   m->config.moe_latent_dim,
                   lz_matmul_xq_i16_ok(&L->moe_expert_w3[0],
                                       m->config.moe_latent_dim, 1));
            found = 1;
        }
        if (L->moe_shared_gate.n) {
            printf("  layer %d moe_shared_gate in=%d  ok(nt=1)=%d ok(nt=8)=%d\n",
                   li, m->config.hidden_size,
                   lz_matmul_xq_i16_ok(&L->moe_shared_gate, m->config.hidden_size, 1),
                   lz_matmul_xq_i16_ok(&L->moe_shared_gate, m->config.hidden_size, 8));
            printf("  layer %d moe_shared_up   in=%d  ok(nt=1)=%d ok(nt=8)=%d\n",
                   li, m->config.hidden_size,
                   lz_matmul_xq_i16_ok(&L->moe_shared_up, m->config.hidden_size, 1),
                   lz_matmul_xq_i16_ok(&L->moe_shared_up, m->config.hidden_size, 8));
            found = 1;
        }
        if (L->gate_proj.n) {
            printf("  layer %d gate_proj in=%d  ok(nt=1)=%d ok(nt=8)=%d\n",
                   li, m->config.hidden_size,
                   lz_matmul_xq_i16_ok(&L->gate_proj, m->config.hidden_size, 1),
                   lz_matmul_xq_i16_ok(&L->gate_proj, m->config.hidden_size, 8));
            printf("  layer %d up_proj   in=%d  ok(nt=1)=%d ok(nt=8)=%d\n",
                   li, m->config.hidden_size,
                   lz_matmul_xq_i16_ok(&L->up_proj, m->config.hidden_size, 1),
                   lz_matmul_xq_i16_ok(&L->up_proj, m->config.hidden_size, 8));
            found = 1;
        }
    }
}

int main(int argc, char **argv) {
    static LZModel m;
    static LZRunState st;
    char err[512];
    int i, j, c, ntok = 0;
    int *toks;

    if (argc < 3) { printf("usage: m8range <model-dir> <tokens-file>\n"); return 2; }
    if (lz_open(&m, argv[1], err, sizeof(err)) != 0) {
        printf("open failed: %s\n", err); return 1;
    }
    if (lz_read_weights(&m, err, sizeof(err)) != 0) {
        printf("weights failed: %s\n", err); return 1;
    }
    toks = read_tokens(argv[2], &ntok);
    if (!toks || ntok < 1) { printf("token file empty\n"); return 1; }
    if (lz_state_alloc(&st, &m, ntok + 8, 0, err, sizeof(err)) != 0) {
        printf("state failed: %s\n", err); return 1;
    }
    g_m = &m;
    report_ok(&m);

    for (i = 0; i < ntok; i++) lz_forward(&m, &st, toks[i], i);

    printf("\n== %s  %s  (%d tokens)\n", argv[1], argv[2], ntok);
    for (c = 0; c < NCH; c++) {
        if (nseen[c] == 0) continue;
        printf("\n-- %s: n=%ld  max|x|=%.6g  |x|>=12: %ld (%.4f%%)\n",
               chname[c], nseen[c], gmax[c], nsat[c],
               100.0 * (double)nsat[c] / (double)nseen[c]);
        printf("   ES   clamped     %%\n");
        for (j = 0; j < NES; j++) {
            if (clampn[c][j] == 0 && (j == 0 || clampn[c][j - 1] == 0)) {
                if (j > 0 && clampn[c][j] == 0) continue;
            }
            printf("   %2d  %8ld  %8.5f%%\n", j, clampn[c][j],
                   100.0 * (double)clampn[c][j] / (double)nseen[c]);
        }
        /* largest ES that clamps nothing */
        {
            int best = -1;
            for (j = 0; j < NES; j++) if (clampn[c][j] == 0) best = j;
            printf("   largest ES with 0 clamps: %d   (bound 32767 / max = %.2fx headroom)\n",
                   best, (gmax[c] > 0) ? 32767.0 / (gmax[c] * pow(2.0, (double)best)) : 0.0);
        }
        if (gan[c] > 0) {
            /* group-amax spread: how many powers of two below the corpus
               max each group's own amax sits. A group more than 8 below
               loses Q8 resolution to a fixed int16 exponent. */
            long b[NGR];
            long tot = 0;
            memset(b, 0, sizeof(b));
            for (j = 0; j < gan[c]; j++) {
                double r = gamax[c][j] / gmax[c];
                int k = (r > 0) ? (int)floor(-log(r) / log(2.0)) : NGR - 1;
                if (k < 0) k = 0;
                if (k >= NGR) k = NGR - 1;
                b[k]++;
            }
            printf("   per-group(32) amax, powers of two below corpus max:\n");
            for (j = 0; j < NGR; j++) {
                if (b[j] == 0) continue;
                tot += b[j];
                printf("     2^-%-2d  %8ld  %6.2f%%  cum %6.2f%%\n", j, b[j],
                       100.0 * (double)b[j] / (double)gan[c],
                       100.0 * (double)tot / (double)gan[c]);
            }
        }
    }
    return 0;
}
