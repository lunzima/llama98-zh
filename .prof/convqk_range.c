/* What target exponent an int16 exit from lz_causal_conv1d_step_fixed
 * needs, for the q and k chains (docs/int-pipeline-project.md 9.4).
 *
 * The exit's value is acc * s15 * sig_oscale2[c], all three of which are
 * already integers or powers of two, so the exit itself has no float
 * content - only a constant target exponent is missing, and a constant
 * is the one thing the consumer cannot derive per token. Same shape as
 * .prof/m5range.c, which chose LZ_KGATE_ES and the rest.
 *
 * TWO CHAINS, MEASURED APART. q and k come off the same conv with the
 * same weights but different projections, and the exit is per-CHANNEL
 * (sig_oscale2 varies with c) while l2norm consumes a whole VECTOR - so
 * a single exponent has to cover both the channel spread and the token
 * spread. Reported per chain: max |x|, a power-of-two histogram, and
 * the clamp count against +-32767 at each candidate exponent.
 *
 * PICK FROM THE CLAMP COLUMN, NOT THE MAX. The max is one sample; the
 * clamp count is the whole tail. m5range's header says the same and it
 * is the reason LZ_KGATE_ES is what it is.
 *
 * AND WATCH THE OTHER END. Unlike the gate chains, this one is consumed
 * by an L2 NORMALISATION, which divides the vector by its own length -
 * so a too-SMALL exponent costs relative precision on the smallest
 * channels rather than clipping the largest. The per-vector minimum is
 * therefore reported too: min over vectors of (max|x| in that vector),
 * which is what sets how many int16 levels the quietest head gets.
 *
 * -DLZ_CONVO_I16=0 IS NOT OPTIONAL. With the exit on, the conv writes
 * kda_*c_i16 and leaves the float buffers these taps read untouched, so
 * every sample is zero and the report is a table of nothing. That is
 * also the distribution the exponent has to be chosen on: the values
 * BEFORE quantization, which only the =0 build produces.
 *
 * Build (one TU, the engine spliced in, as m5range):
 *   gcc -O1 -std=gnu99 -Isrc -DLZ_CONVO_I16=0 \
 *       -o .prof/convqk_range.exe .prof/convqk_range.c src/cpucheck.c -lm
 * Usage:  convqk_range <model-dir> <tokens-file>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCH 3
#define NB  40                 /* bucket = floor(log2|x|) + 24 */
#define EMIN (-24)
#define EMAX 0

static const char *chname[NCH] = { "kqc (conv q)", "kkc (conv k)",
                                   "kvc (conv v)" };
static long hist[NCH][NB];
static long nseen[NCH];
static double gmax[NCH];
static long clampn[NCH][NB];
/* Per-VECTOR peak, tracked for the quietest one: l2norm divides by the
   vector's own length, so this is what decides the level count the
   smallest head is left with. */
static double vecmin[NCH];
static long nvec[NCH];

static void acc(int c, const float *p, int n) {
    int i, j;
    double vpk = 0.0;
    for (i = 0; i < n; i++) {
        double a = fabs((double)p[i]);
        int e;
        if (a > gmax[c]) gmax[c] = a;
        if (a > vpk) vpk = a;
        nseen[c]++;
        if (a == 0.0) e = 0;
        else {
            e = (int)floor(log(a) / log(2.0)) + 24;
            if (e < 0) e = 0;
            if (e >= NB) e = NB - 1;
        }
        hist[c][e]++;
        for (j = 0; j < NB; j++) {
            int te = EMIN + j;
            if (te > EMAX) break;
            /* round-half-away, the convention epi_align_i16 uses */
            if (a * pow(2.0, -te) + 0.5 >= 32768.0) clampn[c][j]++;
        }
    }
    nvec[c]++;
    if (vpk > 0.0 && (vecmin[c] == 0.0 || vpk < vecmin[c])) vecmin[c] = vpk;
}

static void tap_cq(const char *tag, int li, const float *p, int n);
#define LZ_TAP(tag, li, ptr, n) tap_cq((tag), (li), (ptr), (n))

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
#include "forward.c"

static void tap_cq(const char *tag, int li, const float *p, int n) {
    (void)li;
    if (strcmp(tag, "kqc") == 0) { acc(0, p, n); return; }
    if (strcmp(tag, "kkc") == 0) { acc(1, p, n); return; }
    /* v was added after the first landing, and adding it is the point:
       LZ_CONVO_ES was picked from q and k and then USED for v without
       anyone measuring v. That is choosing a constant on a distribution
       it was not measured on, which is the mistake this whole file
       exists to prevent. */
    if (strcmp(tag, "kvc") == 0) { acc(2, p, n); return; }
}

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

int main(int argc, char **argv) {
    static LZModel m;
    static LZRunState st;
    char err[512];
    int i, j, c, ntok = 0;
    int *toks;

    if (argc < 3) {
        printf("usage: convqk_range <model-dir> <tokens-file>\n");
        return 2;
    }
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
    printf("config: lin_k_head_dim=%d lin_key_dim=%d lin_value_dim=%d "
           "conv_kernel=%d n_layers=%d\n",
           m.config.lin_k_head_dim, m.config.lin_key_dim,
           m.config.lin_value_dim, m.config.conv_kernel, m.config.n_layers);
    printf("tiers: conv=%s recur=%s\n",
           lz_conv_mode() ? "fixed" : "float",
           lz_gdn_mode() ? "fixed" : "float");

    for (i = 0; i < ntok; i++) lz_forward(&m, &st, toks[i], i);

    printf("\n%d tokens from %s\n", ntok, argv[2]);
    for (c = 0; c < NCH; c++) {
        printf("\n== %s ==  samples %ld  vectors %ld  max|x| %.6g\n",
               chname[c], nseen[c], nvec[c], gmax[c]);
        printf("   quietest vector's peak %.6g  (ratio to global max %.1f)\n",
               vecmin[c], vecmin[c] > 0.0 ? gmax[c] / vecmin[c] : 0.0);
        printf("   log2 buckets (nonzero):");
        for (j = 0; j < NB; j++)
            if (hist[c][j]) printf(" 2^%d:%ld", j - 24, hist[c][j]);
        printf("\n");
        printf("   target_e -> clamps at 32767:");
        for (j = 0; j < NB; j++) {
            int te = EMIN + j;
            if (te > EMAX) break;
            if (te < -20) continue;
            printf(" %d:%ld", te, clampn[c][j]);
        }
        printf("\n");
    }
    lz_state_free(&st);
    lz_free(&m);
    return 0;
}
