/* A-2: CED's substitution reaches BOTH of the compressor's inputs.

   Report Eq(1) (SS2.2, PDF p.9): `C_l = H_{L/2} W^{KV}_l , Z_l = H_{L/2}
   W^{Z}_l , l > L/2` - a decoder layer projects both its KV entries and
   their compression weights from the encoder's final hidden. The
   reference implementation says the same thing structurally:
   .prof/dsv41_ref/model.py's Compressor.forward does
   `kv, score = self.wkv(x), self.wgate(x)` off ONE `x`. And the training
   side here already does it (engine/csa2.py calls
   `csa2_compress(hs_kv, csa2_w_kv, csa2_w_gate, m)`).

   The engine substituted only the C half: `s->csa2_c_j` read `kvh` while
   `s->csa2_z_j` read the layer's own `h`. This probe exists because that
   divergence is INVISIBLE at csa2_compress_m == 1 - the softmax over a
   one-position block is 1 whatever the gate logits are, so Z drops out -
   and every other fixture in this tree's gates runs at m == 1. Hence m=2
   here, and the size of the fixture is the point rather than a detail.

   WHAT IT ASSERTS. With kv_in non-NULL (a "decoder region" layer, the
   only place the substitution is read at all):

     1. The finalized compressed entry does NOT move when the layer's own
        input moves and kv_in is held fixed. That is the property Eq(1)
        states: both inputs come from H_{L/2}, so nothing about this
        position's own hidden may reach them.
     2. CONTROL: the same entry DOES move when kv_in moves. Without this,
        arm 1 would also pass for a layer whose compression read nothing
        at all - the shape this project keeps having to guard against.

   Bit-exact comparison, not a tolerance: the two arms feed the same
   weights and the same kv_in, so any difference at all is the input
   leaking through, and a tolerance would only make that harder to see.

   Usage: probe    Exit: 0 both arms held, 1 otherwise. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "forward.h"
#include "ops.h"

#define Z_DIM     64
#define Z_NH       2
#define Z_HD      32
#define Z_QD      64
#define Z_QGD    128
#define Z_SEQLEN  32
#define Z_IDXH     2
#define Z_IDXHD   16
#define Z_IDXR    16
#define Z_CM       2      /* see the header: m=1 makes this whole probe blind */
#define Z_TOPK     2
#define Z_NT       2      /* exactly one block, closed at the second position */

#define Z_SEED 1145141919u

static int g_fail;
static void ok(int cond, const char *msg) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", msg);
    if (!cond) g_fail = 1;
}

static unsigned g_st;
static unsigned lcg_next(void) { g_st = g_st * 1103515245u + 12345u; return g_st; }
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
    c->hidden_size = Z_DIM;
    c->n_layers = 1;
    c->seq_len = Z_SEQLEN;
    c->n_heads = Z_NH;
    c->n_kv_heads = Z_NH;
    c->head_dim = Z_HD;
    c->intermediate_size = 64;
    c->rotary_dim = Z_IDXHD;      /* <= csa2_indexer_head_dim, see the others */
    c->rope_theta = 10000.0f;
    c->rms_norm_eps = 1e-5f;
    c->attn_output_gate = 1;
    c->attn_q_dim = Z_QD;
    c->attn_qgate_dim = Z_QGD;
    c->attn_kv_dim = 64;
    c->use_subn = 0;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->layer_types[i] = LZ_LT_FULL;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->csa2_source[i] = -1;
    c->n_full_layers = 1;
    c->csa2_compress_m = Z_CM;
    c->csa2_topk = Z_TOPK;
    c->csa2_indexer_heads = Z_IDXH;
    c->csa2_indexer_head_dim = Z_IDXHD;
    c->csa2_indexer_q_lora_rank = Z_IDXR;
    c->csa2_on[0] = 1;
    c->csa2_mode[0] = LZ_CSA2_FULL;
}

static void build_layer_full(LZLayer *L) {
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL;
    L->csa2_on = 1;
    L->csa2_mode = LZ_CSA2_FULL;
    fill_f(&L->q_proj, Z_QGD * Z_DIM);
    fill_f(&L->o_proj, Z_DIM * Z_QD);
    fill_f(&L->q_norm, Z_HD);
    fill_f(&L->k_norm, Z_HD);
    fill_f(&L->csa2_w_kv, Z_HD * Z_DIM);
    fill_f(&L->csa2_w_gate, Z_HD * Z_DIM);
    fill_f(&L->csa2_w_kv_idx, Z_IDXHD * Z_HD);
    fill_f(&L->csa2_k_norm_idx, Z_IDXHD);
    fill_f(&L->csa2_w_dq_idx, Z_IDXR * Z_DIM);
    fill_f(&L->csa2_w_uq_idx, (Z_IDXH * Z_IDXHD) * Z_IDXR);
    fill_f(&L->csa2_w_weights, Z_IDXH * Z_DIM);
}

/* Run the layer once over Z_NT positions with a fixed-hidden draw `xseed`
   and a fixed kv_in, and hand back the FINALIZED compressed entry for
   block 0 - the ring row, which is what the compressor produces. The
   query path also reads the position's own hidden; that is deliberate and
   is exactly what the first arm is asking about, since the ring must not
   move with it. */
static int run_entry(unsigned xseed, const float *kv, float *out, int n,
                     char *errbuf, int errlen) {
    LZModel m;
    LZRunState s;
    LZLayer *L;
    int i;

    memset(&m, 0, sizeof m);
    init_config(&m.config);
    m.layers = (LZLayer *)calloc(1, sizeof(LZLayer));
    L = &m.layers[0];
    /* The weights come from a FIXED seed in every run - only the
       position's own hidden varies between arms. Without this reset the
       layer would be rebuilt from wherever the shared stream happened to
       be left, so each run would have different weights and the
       comparison would differ for a reason that has nothing to do with
       what it is asking. */
    g_st = Z_SEED;
    build_layer_full(L);

    if (lz_state_alloc(&s, &m, Z_SEQLEN, 0, errbuf, errlen) != 0) {
        free_layer(L); free(m.layers);
        return 1;
    }
    /* Only now the variable under test. */
    g_st = xseed;
    for (i = 0; i < Z_NT * Z_DIM; i++) s.xb[i] = lcg_f();
    forward_csa2(&m, &s, L, 0, 0, Z_NT, kv);

    {
        int ci = s.cache_idx[0];
        const float *ring = s.csa2_comp_ring
            + lz_csa2_ring_base(&m.config, ci, s.seq_len) * Z_HD;
        memcpy(out, ring, (size_t)n * sizeof(float));
    }
    lz_state_free(&s);
    free_layer(L);
    free(m.layers);
    return 0;
}

int main(void) {
    float poison_a[Z_NT * Z_DIM], poison_b[Z_NT * Z_DIM];
    float e_x1_ka[Z_HD], e_x2_ka[Z_HD], e_x1_kb[Z_HD];
    char errbuf[256];
    int i, rc;

    /* Two kv_in buffers that share no values, and both far outside
       lcg_f's [-0.2, 0.2) range, so a read of the wrong one cannot be
       mistaken for a rounding difference. */
    for (i = 0; i < Z_NT * Z_DIM; i++) {
        poison_a[i] = 500.0f + (float)i;
        poison_b[i] = -700.0f - (float)i;
    }

    rc = run_entry(Z_SEED + 1, poison_a, e_x1_ka, Z_HD, errbuf, sizeof errbuf)
       | run_entry(Z_SEED + 2, poison_a, e_x2_ka, Z_HD, errbuf, sizeof errbuf)
       | run_entry(Z_SEED + 1, poison_b, e_x1_kb, Z_HD, errbuf, sizeof errbuf);
    if (rc != 0) {
        printf("FAIL setup: %s\n", errbuf);
        printf("csa2_z_subst_probe: FAIL\n");
        return 1;
    }

    {
        int nonzero = 0;
        for (i = 0; i < Z_HD; i++) if (e_x1_ka[i] != 0.0f) nonzero = 1;
        ok(nonzero, "setup: the compressed entry is not the zero vector "
                    "(a block really did close)");
    }

    ok(memcmp(e_x1_ka, e_x2_ka, sizeof e_x1_ka) == 0,
       "a2: the compressed entry does NOT move when this position's own "
       "hidden moves and kv_in is held fixed (Eq(1): both C and Z come "
       "from H_{L/2})");

    ok(memcmp(e_x1_ka, e_x1_kb, sizeof e_x1_ka) != 0,
       "a2 control: ...and it DOES move when kv_in moves - so the arm "
       "above is not passing because the compression reads nothing");

    printf("csa2_z_subst_probe: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
