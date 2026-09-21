/* A-3: the inverse RoPE on CSA2's attention OUTPUT.

   Step 5 of forward_csa2.c scores and then accumulates the SAME ring
   pointer (one shared K=V entry, Eq 19), and step 3 rotated every ring
   entry - so the attention output is a weighted sum of ROTATED values.
   The reference un-rotates it at the QUERY's own position
   (.prof/dsv41_ref/model.py:781, `apply_rotary_emb(o[..., -rd:],
   freqs_cis, True)`), and this side now does the same by handing lz_rope
   the conjugate table (`rope_cs_inv`).

   TWO CHECKS, and they answer different questions.

   1. TABLE. Is rope_cs_inv actually the inverse of rope_cs? This one has
      a real oracle, independent of the reference: R(a)^-1 R(a) = I, so
      rotating by cs and then by cs_inv must return the input - to
      rounding, since two rotations round twice. The control arm is a
      NON-conjugated copy of the same table, which must miss by orders of
      magnitude; without that control the check could be passing because
      it cannot see anything.
      This is the arm that would catch a table that dropped the negation,
      or negated the cosines instead.

   2. WIRING. Is rope_cs_inv actually reached from forward_csa2? There is
      NO independent oracle for this one on this machine: the report gives
      no formula for the output-side rotation (only the fused-kernel
      mention at .prof/dsv41_report.txt:969-970), and the reference file
      is the whole of the available specification. So this probe does not
      assert a value - it PRINTS the attention output, and
      build/csa2_out_rope_gate.sh compares the fixed build's line against
      a build with the step's lz_rope call deleted. A difference is the
      assertion; "it equals the reference" is not checked here and cannot
      be. Read the gate's own header before treating this as more than it
      is.

   The fixture is the same shape build/csa2_mode_dispatch_probe.c uses -
   a hand-built LZModel/LZLayer, no config.json, no safetensors - with
   csa2_compress_m = 1, so every token closes its own block and its own
   ring entry is what it attends to (a query always has at least one
   visible block, so no row is degenerate).

   Usage: probe    Exit: 0 both checks passed, 1 otherwise. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "forward.h"
#include "ops.h"

#define G_DIM     64
#define G_NH       2
#define G_HD      32
#define G_QD      64
#define G_QGD    128
#define G_SEQLEN  32
#define G_IDXH     2
#define G_IDXHD   16
#define G_IDXR    16
#define G_CM       1
#define G_TOPK     2
#define G_NT       4

/* The same fixed synthetic-input seed every probe in this tree uses. */
#define G_SEED 1145141919u

static int g_fail;

static void ok(int cond, const char *msg) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", msg);
    if (!cond) g_fail = 1;
}

static unsigned g_st;
static unsigned lcg_next(void) { g_st = g_st * 1103515245u + 12345u; return g_st; }
/* [-0.2, 0.2) - same shape as the other probes' generators. */
static float lcg_f(void) {
    return (float)((int)((lcg_next() >> 8) % 2000u) - 1000) * 0.0002f;
}

static void fill_f(LZTensor *t, int n) {
    int i;
    memset(t, 0, sizeof(*t));
    t->dtype = LZ_FMT_F32;
    t->n = n;
    t->gs = 0;
    t->f = (float *)malloc((size_t)n * sizeof(float));
    for (i = 0; i < n; i++) t->f[i] = lcg_f();
}

static void free_layer(LZLayer *L) {
    free(L->q_proj.f); free(L->o_proj.f); free(L->q_norm.f); free(L->k_norm.f);
    free(L->csa2_w_kv.f); free(L->csa2_w_gate.f); free(L->csa2_w_kv_idx.f);
    free(L->csa2_k_norm_idx.f); free(L->csa2_w_dq_idx.f); free(L->csa2_w_uq_idx.f);
    free(L->csa2_w_weights.f);
    memset(L, 0, sizeof(*L));
}

static void init_config(LZModelConfig *c) {
    int i;
    memset(c, 0, sizeof(*c));
    c->vocab_size = 4;
    c->hidden_size = G_DIM;
    c->n_layers = 1;
    c->seq_len = G_SEQLEN;
    c->n_heads = G_NH;
    c->n_kv_heads = G_NH;
    c->head_dim = G_HD;
    c->intermediate_size = 64;
    /* The rotation width, and it must not exceed csa2_indexer_head_dim:
       lz_rope writes base[i] and base[i + rotary_dim/2] inside a row, and
       the indexer ring's rows are G_IDXHD wide. */
    c->rotary_dim = G_IDXHD;
    c->rope_theta = 10000.0f;
    c->rms_norm_eps = 1e-5f;
    c->attn_output_gate = 1;
    c->attn_q_dim = G_QD;
    c->attn_qgate_dim = G_QGD;
    c->attn_kv_dim = 64;
    c->use_subn = 0;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->layer_types[i] = LZ_LT_FULL;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->csa2_source[i] = -1;
    c->n_full_layers = 1;
    c->csa2_compress_m = G_CM;
    c->csa2_topk = G_TOPK;
    c->csa2_indexer_heads = G_IDXH;
    c->csa2_indexer_head_dim = G_IDXHD;
    c->csa2_indexer_q_lora_rank = G_IDXR;
    c->csa2_on[0] = 1;
    c->csa2_mode[0] = LZ_CSA2_FULL;
}

static void build_layer_full(LZLayer *L) {
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL;
    L->csa2_on = 1;
    L->csa2_mode = LZ_CSA2_FULL;
    fill_f(&L->q_proj, G_QGD * G_DIM);
    fill_f(&L->o_proj, G_DIM * G_QD);
    fill_f(&L->q_norm, G_HD);
    fill_f(&L->k_norm, G_HD);
    fill_f(&L->csa2_w_kv, G_HD * G_DIM);
    fill_f(&L->csa2_w_gate, G_HD * G_DIM);
    fill_f(&L->csa2_w_kv_idx, G_IDXHD * G_HD);
    fill_f(&L->csa2_k_norm_idx, G_IDXHD);
    fill_f(&L->csa2_w_dq_idx, G_IDXR * G_DIM);
    fill_f(&L->csa2_w_uq_idx, (G_IDXH * G_IDXHD) * G_IDXR);
    fill_f(&L->csa2_w_weights, G_IDXH * G_DIM);
}

/* ---------------------------------------------------------------- 1. table */

static float maxdiff(const float *a, const float *b, int n) {
    float m = 0.0f; int i;
    for (i = 0; i < n; i++) { float d = fabsf(a[i] - b[i]); if (d > m) m = d; }
    return m;
}

/* Worst per-element deviation after rotating by `a` and then by `b`,
   over every position in the table. A true inverse lands at rounding
   level; anything else does not. */
static float roundtrip_worst(const LZModelConfig *c, const float *cs,
                             const float *cs2, unsigned seed) {
    float x[G_NH * G_HD], orig[G_NH * G_HD];
    float worst = 0.0f;
    int pos, i;
    g_st = seed;
    for (pos = 0; pos < G_SEQLEN; pos++) {
        for (i = 0; i < G_NH * G_HD; i++) { x[i] = lcg_f(); orig[i] = x[i]; }
        lz_rope(x, G_NH, G_HD, c->rotary_dim, pos, cs);
        lz_rope(x, G_NH, G_HD, c->rotary_dim, pos, cs2);
        { float d = maxdiff(orig, x, G_NH * G_HD); if (d > worst) worst = d; }
    }
    return worst;
}

static void check_table(const LZModelConfig *c) {
    int n = G_SEQLEN * c->rotary_dim;
    float *cs = (float *)malloc((size_t)n * sizeof(float));
    float *cs_inv = (float *)malloc((size_t)n * sizeof(float));
    float *cs_ctrl = (float *)malloc((size_t)n * sizeof(float));
    float inv_err, ctrl_err;
    int i;

    lz_rope_build_table(cs, G_SEQLEN, c);
    lz_rope_conjugate_table(cs_inv, cs, (size_t)n);
    /* Control: the same table WITHOUT the conjugation - what a dropped
       negation produces. It must miss by far more than rounding, or this
       check cannot tell the two apart and proves nothing. */
    for (i = 0; i < n; i++) cs_ctrl[i] = cs[i];

    inv_err = roundtrip_worst(c, cs, cs_inv, G_SEED + 11);
    ctrl_err = roundtrip_worst(c, cs, cs_ctrl, G_SEED + 11);

    {
        char buf[256];
        snprintf(buf, sizeof buf,
                 "table: rotating by rope_cs then rope_cs_inv returns the "
                 "input (worst %.3g, rounding level)", (double)inv_err);
        ok(inv_err < 1e-5f, buf);
    }
    {
        char buf[256];
        snprintf(buf, sizeof buf,
                 "table: the NON-conjugated control does NOT round-trip "
                 "(worst %.3g) - so the check above can see the difference",
                 (double)ctrl_err);
        ok(ctrl_err > 1e-3f, buf);
    }
    free(cs); free(cs_inv); free(cs_ctrl);
}

/* -------------------------------------------------------------- 2. wiring */

static void check_wiring(void) {
    LZModel m;
    LZRunState s;
    LZLayer *L;
    char errbuf[256];
    int i;

    memset(&m, 0, sizeof m);
    init_config(&m.config);
    m.layers = (LZLayer *)calloc(1, sizeof(LZLayer));
    L = &m.layers[0];
    build_layer_full(L);

    if (lz_state_alloc(&s, &m, G_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL wiring: lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free_layer(L); free(m.layers);
        return;
    }
    g_st = G_SEED + 21;
    for (i = 0; i < G_NT * G_DIM; i++) s.xb[i] = lcg_f();
    forward_csa2(&m, &s, L, 0, 0, G_NT, NULL);

    {
        int finite = 1, d;
        for (d = 0; d < G_QD; d++) if (!isfinite(s.attn_out[d])) finite = 0;
        ok(finite, "wiring: the layer ran and its attention output is finite");
    }
    /* The comparison value. Printed, not asserted: this probe has no
       oracle for what the number SHOULD be, only for whether it moves
       when the un-rotation does. build/csa2_out_rope_gate.sh diffs this
       block against a build with the step's lz_rope call deleted. */
    printf("OUT-BEGIN\n");
    for (i = 0; i < G_QD; i++)
        printf("OUT %d %.9g\n", i, (double)s.attn_out[i]);
    printf("OUT-END\n");

    lz_state_free(&s);
    free_layer(L);
    free(m.layers);
}

int main(void) {
    LZModelConfig c;
    init_config(&c);
    check_table(&c);
    check_wiring();
    printf("csa2_out_rope_probe: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
