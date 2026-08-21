/* int-pipeline milestone 5: what dynamic range do the four q16_0
 * outputs actually have, and what does this checkpoint's config say
 * about their consumers?
 *
 * An int16 exit needs a CONSTANT target exponent, measured once from
 * the real distribution (the same thing LZ_CONV_ES is, and the same
 * reason: the consumer has nothing to derive one from per token). This
 * measures the four:
 *
 *   kb    s->bvec            (kda_b_proj)    -> lz_sigmoid
 *   mrt   s->moe_router_logits (moe_gate_w)  -> lz_moe_route
 *   kfa   s->kda_gate_lat    (kda_f_a_proj)  -> lz_matmul_w's quantize
 *   kfb   s->kda_gate        (kda_f_b_proj)  -> the decay gate
 *
 * The first three come off taps forward.c already has. The fourth has
 * no tap before the decay loop overwrites it in place, so it is
 * RECONSTRUCTED here by running the same lz_matmul_w the engine runs,
 * on the same kda_gate_lat the "kfa" tap just delivered - not a
 * re-derivation, the shipped function on the shipped input.
 *
 * Reported per chain: max |x|, a power-of-two histogram, and for each
 * candidate target exponent the clamp count against a +-32767 int16
 * bound. Pick the exponent from the clamp column, not from the max:
 * the max is one sample and the clamp count is the whole tail.
 *
 * Also reports what kda_gate's own consumer saturates at, because "the
 * clamp is behind the consumer's own clamp" would have been the strong
 * argument for a thin bound - and on kmr20 it is NOT available:
 * exp(A_log) runs down to 0.00141, so a large gate value can still land
 * far inside the sigmoid's linear region. That is why LZ_KGATE_ES is
 * chosen on headroom rather than on saturation.
 *
 * Build (one TU: the engine is spliced in, so no link set applies and
 * .prof/build.sh's escalation cannot help; cpucheck.c is the one file
 * that is not spliced because ops.c only declares it):
 *   gcc -O1 -std=gnu99 -Isrc -o .prof/m5range.exe .prof/m5range.c \
 *       src/cpucheck.c -lm
 * Usage:  m5range <model-dir> <tokens-file>
 *
 * NOTE ON FIDELITY: this build defines neither LZ_MMX_TU nor
 * LZ_SSE2_TU, so q16_0 reaches matmul_scalar_ref rather than
 * matmul_q16_impl + epi_fixed. The VALUES therefore differ from the
 * shipping build in the last bits; the RANGES this program exists to
 * report do not, and the shipping build's own pre-clamp counter agreed
 * with them to within a few tenths of a percent.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NCH 4
#define NB  40                 /* bucket = floor(log2|x|) + 24 */
#define EMIN (-24)
#define EMAX 0

static const char *chname[NCH] = { "kb (bvec)", "mrt (router)",
                                   "kfa (gate_lat)", "kfb (kda_gate)" };
static long hist[NCH][NB];
static long nseen[NCH];
static double gmax[NCH];
/* clamp[c][j]: how many samples would exceed +-32767 at target_e = EMIN+j */
static long clampn[NCH][NB];

static void acc(int c, const float *p, int n) {
    int i, j;
    for (i = 0; i < n; i++) {
        double a = fabs((double)p[i]);
        int e;
        if (a > gmax[c]) gmax[c] = a;
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
}

static void tap_m5(const char *tag, int li, const float *p, int n);
#define LZ_TAP(tag, li, ptr, n) tap_m5((tag), (li), (ptr), (n))

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

static LZModel *g_m;
static int8_t *g_xq;
static float  *g_xqs, *g_fb, *g_wscr, *g_wscr2;
static double dbmin = 1e30, dbmax = 0.0, dtbmax = 0.0;
static long nsat, nsat_tot, nbig, nbig_sat, nbig16, nbig16_sat;

static void tap_m5(const char *tag, int li, const float *p, int n) {
    if (strcmp(tag, "kb") == 0)  { acc(0, p, n); return; }
    if (strcmp(tag, "mrt") == 0) { acc(1, p, n); return; }
    if (strcmp(tag, "kfa") == 0) {
        acc(2, p, n);
        if (g_m && li >= 0 && li < g_m->config.n_layers) {
            const LZLayer *L = &g_m->layers[li];
            const LZModelConfig *c = &g_m->config;
            int nvk = c->lin_n_v_heads * c->lin_k_head_dim;
            int kd = c->lin_k_head_dim, h, i;
            const float *alog, *dtb;
            if (L->kda_f_b_proj.n == 0) return;
            lz_matmul_w(g_fb, p, (LZTensor *)&L->kda_f_b_proj,
                        n, nvk, g_xq, g_xqs);
            acc(3, g_fb, nvk);
            /* The gate's own consumer: g = lb * sigmoid(exp(A_log[h]) *
               (gt + dt_bias)). sigmoid_q15 saturates at |arg| >= 12, so
               an int16 clamp on gt is a non-event wherever the argument
               is already past that - count both. */
            alog = lz_t_f32(&L->kda_A_log, g_wscr);
            dtb  = lz_t_f32(&L->kda_dt_bias, g_wscr2);
            for (h = 0; h < c->lin_n_v_heads; h++) {
                double db = (double)lz_exp(alog[h]);
                if (db < dbmin) dbmin = db;
                if (db > dbmax) dbmax = db;
                for (i = 0; i < kd; i++) {
                    int idx = h * kd + i;
                    double pre = (double)g_fb[idx] + (double)dtb[idx];
                    double arg = db * pre;
                    if (fabs((double)dtb[idx]) > dtbmax) dtbmax = fabs((double)dtb[idx]);
                    nsat_tot++;
                    if (fabs(arg) >= 12.0) nsat++;
                    if (fabs((double)g_fb[idx]) > 32.0) {
                        nbig++;
                        if (fabs(arg) >= 12.0) nbig_sat++;
                    }
                    if (fabs((double)g_fb[idx]) > 16.0) {
                        nbig16++;
                        if (fabs(arg) >= 12.0) nbig16_sat++;
                    }
                }
            }
        }
        return;
    }
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

    if (argc < 3) { printf("usage: m5range <model-dir> <tokens-file>\n"); return 2; }
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
    {
        int nvk = m.config.lin_n_v_heads * m.config.lin_k_head_dim;
        g_xq  = (int8_t *)calloc((size_t)m.config.hidden_size + 4096, 1);
        g_xqs = (float *)calloc((size_t)m.config.hidden_size + 4096, sizeof(float));
        g_fb  = (float *)calloc((size_t)nvk + 64, sizeof(float));
        g_wscr  = (float *)calloc((size_t)nvk + 64, sizeof(float));
        g_wscr2 = (float *)calloc((size_t)nvk + 64, sizeof(float));
        printf("config: hidden=%d kda_gate_rank=%d nvk=%d lin_n_v_heads=%d\n",
               m.config.hidden_size, m.config.kda_gate_rank, nvk,
               m.config.lin_n_v_heads);
        printf("        lin_k_head_dim=%d lin_key_dim=%d lin_value_dim=%d\n",
               m.config.lin_k_head_dim, m.config.lin_key_dim,
               m.config.lin_value_dim);
        printf("        kda_has_gate_lower_bound=%d kda_gate_lower_bound=%.6f\n",
               m.config.kda_has_gate_lower_bound, m.config.kda_gate_lower_bound);
        printf("        num_experts=%d per_token=%d moe_router_sigmoid=%d "
               "moe_renormalize=%d\n",
               m.config.num_experts, m.config.num_experts_per_token,
               m.config.moe_router_sigmoid, m.config.moe_renormalize);
        printf("        use_subn=%d n_layers=%d\n",
               m.config.use_subn, m.config.n_layers);
    }
    for (i = 0; i < ntok; i++) lz_forward(&m, &st, toks[i], i);

    printf("\n%d tokens from %s\n", ntok, argv[2]);
    for (c = 0; c < NCH; c++) {
        printf("\n== %s ==  samples %ld  max|x| %.6f\n",
               chname[c], nseen[c], gmax[c]);
        printf("   log2 buckets (nonzero):");
        for (j = 0; j < NB; j++)
            if (hist[c][j]) printf(" 2^%d:%ld", j - 24, hist[c][j]);
        printf("\n");
        printf("   target_e -> clamps at bound 32767:");
        for (j = 0; j < NB; j++) {
            int te = EMIN + j;
            if (te > EMAX) break;
            if (te < -20) continue;
            printf(" %d:%ld", te, clampn[c][j]);
        }
        printf("\n");
    }
    printf("\nkda_gate consumer saturation (sigmoid clamps at |arg| >= 12):\n");
    printf("  exp(A_log) in [%.6f, %.6f], max |dt_bias| %.6f\n",
           dbmin, dbmax, dtbmax);
    printf("  saturated %ld / %ld (%.2f%%)\n", nsat, nsat_tot,
           nsat_tot ? 100.0 * (double)nsat / (double)nsat_tot : 0.0);
    printf("  |gate| > 32: %ld, of which already saturated %ld\n", nbig, nbig_sat);
    printf("  |gate| > 16: %ld, of which already saturated %ld\n", nbig16, nbig16_sat);
    lz_state_free(&st);
    lz_free(&m);
    return 0;
}
