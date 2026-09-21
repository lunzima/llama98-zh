/* Which dims the indexer key's RoPE rotates: the engine's own
 * forward_csa2() product, not a re-run of the idiom it uses.
 *
 * The V4.1 reference rotates the LAST rd dims of an indexer key
 * (`.prof/dsv41_ref/model.py`'s `k[..., -rd:]`), while src/ops_rope.c's
 * lz_rope rotates the FIRST rotary_dim dims of whatever row it is handed
 * (base[i] paired with base[i + rotary_dim/2]). src/forward_csa2.c
 * reconciles the two by advancing the pointer past the leading
 * indexer_head_dim - rotary_dim dims before the call, so the trailing
 * ones are the ones that rotate. This probe reads the two rows the
 * engine actually wrote into the indexer ring and asserts that
 * partition; a probe that called lz_rope itself would only re-test the
 * offset it hardcoded.
 *
 * The pre-rotation row is taken from the SAME model at position 0, where
 * the frequency table's angle is 0 (cos 1, sin 0) and lz_rope is the
 * identity under either slice - so the row at position 1 is compared
 * against the identical latent compressed at a different position, not
 * against a second implementation of the compressor. Both arms feed the
 * same hidden vector through lz_state_alloc's xb, and cm is 1, so the
 * block a position writes is exactly that one token's latent.
 *
 * CHECK LIST (each line is one ok() call; the gate script's mutant arms
 * name the fallout per line):
 *   idx/check1  leading idx_hd - rotary_dim dims bit-identical to the
 *               position-0 row (they are the dims the reference leaves
 *               alone). Reddens when the slice is the leading one.
 *   idx/check2  trailing rotary_dim dims equal the frequency table's
 *               rotation of the position-0 row's trailing dims.
 *   idx/check3  control: the trailing dims DID move, so a rotation that
 *               is silently absent cannot satisfy check1+check2.
 *   idx/check4  identity arm: at rotary_dim == indexer_head_dim the
 *               offset is 0 and the WHOLE row rotates - the slice change
 *               has nothing to move when the two widths coincide.
 *   idx/check5  control on the other call site: the main KV entry still
 *               rotates its LEADING rotary_dim dims only, so the indexer
 *               change did not follow the pointer arithmetic onto it.
 *
 * Exit: 0 all pass, 1 a check failed. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "forward.h"

static int g_fail = 0;
static void ok(int c, const char *w) {
    printf("%s %s\n", c ? "PASS" : "FAIL", w);
    fflush(stdout);
    if (!c) g_fail = 1;
}

#define F_DIM   64
#define F_NH     2
#define F_HD    32
#define F_QD    64
#define F_QGD  128
#define F_SEQ    8
#define F_IDXH   2
#define F_IDXHD 32   /* indexer key rows are this wide              */
#define F_IDXR  16
#define F_RD    16   /* rotary_dim; < F_IDXHD so the two slices are
                        different dims, == F_IDXHD in the identity arm */
#define F_CM     1   /* one raw position per block: the block a
                        position writes is that token's own latent   */
#define F_TOPK   2
#define SEED0 1145141919u

static unsigned lcg(unsigned *s) { *s = *s * 1103515245u + 12345u; return *s; }
static float lcgf(unsigned *s) {
    return (float)((int)((lcg(s) >> 8) % 2000u) - 1000) * 0.0002f;
}
static void fill(LZTensor *t, int n, unsigned *s) {
    int i;
    memset(t, 0, sizeof(*t));
    t->dtype = LZ_FMT_F32; t->n = n; t->gs = 0;
    t->f = (float *)malloc((size_t)n * sizeof(float));
    for (i = 0; i < n; i++) t->f[i] = lcgf(s);
}
static void layer(LZLayer *L, unsigned q, unsigned kv, unsigned ix) {
    unsigned a = q, b = kv, c = ix;
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL; L->csa2_on = 1; L->csa2_mode = LZ_CSA2_FULL;
    fill(&L->q_proj, F_QGD * F_DIM, &a);
    fill(&L->o_proj, F_DIM * F_QD, &a);
    fill(&L->q_norm, F_HD, &a);
    fill(&L->k_norm, F_HD, &b);
    fill(&L->csa2_w_kv, F_HD * F_DIM, &b);
    fill(&L->csa2_w_gate, F_HD * F_DIM, &b);
    fill(&L->csa2_w_kv_idx, F_IDXHD * F_HD, &b);
    fill(&L->csa2_k_norm_idx, F_IDXHD, &b);
    fill(&L->csa2_w_dq_idx, F_IDXR * F_DIM, &c);
    fill(&L->csa2_w_uq_idx, (F_IDXH * F_IDXHD) * F_IDXR, &c);
    fill(&L->csa2_w_weights, F_IDXH * F_DIM, &c);
}

/* One FULL CSA2 layer at the given rotary width. csa2_indexer_head_dim
   is F_IDXHD; rotary == F_IDXHD is the arm where the two slices are the
   same dims. */
static void build(LZModel *m, int rotary) {
    LZModelConfig *c = &m->config;
    int i;
    memset(m, 0, sizeof(*m));
    memset(c, 0, sizeof(*c));
    c->vocab_size = 4; c->hidden_size = F_DIM; c->n_layers = 1; c->seq_len = F_SEQ;
    c->n_heads = F_NH; c->n_kv_heads = F_NH; c->head_dim = F_HD;
    c->intermediate_size = F_DIM;
    c->rotary_dim = rotary; c->rope_theta = 10000.0f; c->rms_norm_eps = 1e-5f;
    c->attn_output_gate = 1;
    c->attn_q_dim = F_QD; c->attn_qgate_dim = F_QGD; c->attn_kv_dim = F_DIM;
    c->use_subn = 0;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->csa2_source[i] = -1;
    c->layer_types[0] = LZ_LT_FULL;
    c->n_full_layers = 1;
    c->csa2_on[0] = 1;
    c->csa2_mode[0] = LZ_CSA2_FULL;
    c->csa2_compress_m = F_CM; c->csa2_topk = F_TOPK;
    c->csa2_indexer_heads = F_IDXH;
    c->csa2_indexer_head_dim = F_IDXHD;
    c->csa2_indexer_q_lora_rank = F_IDXR;
    m->layers = (LZLayer *)calloc(1, sizeof(LZLayer));
    layer(&m->layers[0], SEED0 + 1, SEED0 + 2, SEED0 + 3);
}

static void fill_xb(LZRunState *s, unsigned *st) {
    int i;
    for (i = 0; i < F_DIM; i++) s->xb[i] = lcgf(st);
}

/* The row the engine writes compressed block `blk` into, in each of the
   two rings. The fixture has ONE csa2 layer, so cache_idx[0] is 0 and
   layer 0's ring sits at the head of the arena under every layout
   forward_state.c allocates: slot 0's own size - `ci * max_blocks` under
   one compression ratio for the whole model, a prefix sum over the slots
   before it under per-layer ones - is 0 either way at ci == 0.
   two_positions() refuses a nonzero ci rather than reading whatever
   happens to be there, so this cannot go quiet by coincidence. */
static const float *idx_row(const LZRunState *s, int blk) {
    return s->csa2_idx_comp_ring + (size_t)blk * F_IDXHD;
}
static const float *kv_row(const LZRunState *s, int blk) {
    return s->csa2_comp_ring + (size_t)blk * F_HD;
}

static int same(const float *a, const float *b, int n) {
    int i;
    for (i = 0; i < n; i++)
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) return 0;
    return 1;
}

/* lz_rope's own arithmetic, applied to a slice named by the caller: the
   trailing `rd` dims when last != 0, the leading ones otherwise. Same
   expression order as src/ops_rope.c (x0*c - x1*s, x1*c + x0*s), which
   the tiers are contractually bit-identical to. `cs_row` is the table
   row for the position, half*2 floats wide. */
static void rot(const float *in, const float *cs_row, int width, int rd,
                int last, float *out) {
    int half = rd / 2, i, base = last ? width - rd : 0;
    memcpy(out, in, (size_t)width * sizeof(float));
    for (i = 0; i < half; i++) {
        float c = cs_row[i * 2], sn = cs_row[i * 2 + 1];
        float x0 = in[base + i], x1 = in[base + half + i];
        out[base + i]        = x0 * c - x1 * sn;
        out[base + half + i] = x1 * c + x0 * sn;
    }
}

/* Position 0 and position 1 of one model, with the SAME hidden vector fed
   at both: rows[0] is the unrotated latent (angle 0), rows[1] is that
   latent rotated at position 1. */
static int two_positions(LZModel *m, float *idx0, float *idx1, float *kv0,
                         float *kv1, float *cs1, int *rotary_out) {
    LZRunState s;
    char eb[256];
    unsigned st = SEED0 + 40;
    int ci;
    if (lz_state_alloc(&s, m, F_SEQ, 0, eb, sizeof eb) != 0) {
        printf("FAIL idx/setup: lz_state_alloc: %s\n", eb);
        g_fail = 1;
        return 0;
    }
    ci = s.cache_idx[0];
    if (ci != 0) {
        printf("FAIL idx/setup: this fixture's one csa2 layer landed on cache "
               "slot %d, not 0 - every ring address below assumes slot 0\n", ci);
        g_fail = 1;
        return 0;
    }
    st = SEED0 + 40;
    fill_xb(&s, &st);
    forward_csa2(m, &s, &m->layers[0], 0, 0, 1, NULL);
    memcpy(idx0, idx_row(&s, 0), (size_t)F_IDXHD * sizeof(float));
    memcpy(kv0, kv_row(&s, 0), (size_t)F_HD * sizeof(float));

    st = SEED0 + 40;                  /* the SAME hidden vector, so block
                                         1's latent is block 0's */
    fill_xb(&s, &st);
    forward_csa2(m, &s, &m->layers[0], 0, 1, 1, NULL);
    memcpy(idx1, idx_row(&s, 1), (size_t)F_IDXHD * sizeof(float));
    memcpy(kv1, kv_row(&s, 1), (size_t)F_HD * sizeof(float));

    memcpy(cs1, s.rope_cs + (size_t)m->config.rotary_dim, /* pos 1 row */
           (size_t)m->config.rotary_dim * sizeof(float));
    *rotary_out = m->config.rotary_dim;
    return 1;
}

int main(void) {
    LZModel m;
    float idx0[F_IDXHD], idx1[F_IDXHD], want[F_IDXHD];
    float kv0[F_HD], kv1[F_HD], kwwant[F_HD], cs1[F_IDXHD];
    int rotary;

    /* ---- arm A: rotary_dim 16 against indexer_head_dim 32, the pair
       whose two slices are different dims ---- */
    build(&m, F_RD);
    if (!two_positions(&m, idx0, idx1, kv0, kv1, cs1, &rotary)) return 1;

    rot(idx0, cs1, F_IDXHD, rotary, 1, want);
    ok(same(idx0, idx1, F_IDXHD - rotary),
       "idx/check1: the indexer key row's leading idx_hd - rotary_dim dims "
       "are bit-identical to the position-0 row's (the slice the reference "
       "does not rotate)");
    ok(same(want + (F_IDXHD - rotary), idx1 + (F_IDXHD - rotary), rotary),
       "idx/check2: the trailing rotary_dim dims are the frequency table's "
       "rotation of the position-0 row's trailing dims");
    ok(!same(idx0 + (F_IDXHD - rotary), idx1 + (F_IDXHD - rotary), rotary),
       "idx/check3: control - the trailing dims did move between the two "
       "positions, so a rotation that is absent entirely cannot satisfy the "
       "two checks above");

    rot(kv0, cs1, F_HD, rotary, 0, kwwant);
    ok(same(kwwant, kv1, F_HD),
       "idx/check5: control - the main KV entry still rotates its LEADING "
       "rotary_dim dims only (its own call site did not take the offset)");

    /* ---- arm B: rotary_dim == indexer_head_dim, where the offset is 0
       and the whole row rotates whichever slice is named ---- */
    build(&m, F_IDXHD);
    if (!two_positions(&m, idx0, idx1, kv0, kv1, cs1, &rotary)) return 1;
    rot(idx0, cs1, F_IDXHD, rotary, 1, want);
    ok(same(want, idx1, F_IDXHD),
       "idx/check4: identity arm - at rotary_dim == indexer_head_dim the "
       "whole indexer key row rotates (the two slices are the same dims)");

    if (g_fail) {
        printf("csa2_indexer_rope_slice_probe: FAIL\n");
        return 1;
    }
    printf("csa2_indexer_rope_slice_probe: PASS\n");
    return 0;
}
