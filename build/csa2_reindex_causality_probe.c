/* C2 (report Sec2.3.1 mode-dispatch task, fifth review round): a
 * REINDEX layer's n_vis was not causal. (REUSE's own n_vis is dead code
 * - forward_csa2.c's step 3 skips REUSE entirely, so this bug affected
 * a REINDEX layer only; a downstream REUSE layer's OUTPUT still moved,
 * as check 2 below shows, but only because it reads whatever the
 * REINDEX layer's own, wrong, selection published - REUSE never
 * computed anything of its own.)
 *
 * forward_csa2.c used to read `n_vis = s->csa2_n_blocks[ci]` for every
 * mode. For a FULL layer that is correct: its own step 6 is the only
 * writer and has not run for this position yet, so n_vis == floor(pos/cm)
 * - exactly engine/csa2.py's csa2_causal_block_mask ("block s is visible
 * iff s < t // m"). For a REINDEX layer the ci is ALIASED to a source
 * whose ENTIRE call already finished, so the counter was already at
 * floor(chunk_end/cm):
 *   nt=4, cm=2, positions 0..3 -> FULL sees 0,0,1,1 (right)
 *                                 REINDEX saw 2,2,2,2 (wrong: position 0
 *                                 attended to block 1 = positions 2, 3)
 *   nt=1                       -> REINDEX saw 0,1,1,2 (wrong: at pos 1 it
 *                                 read block 0, the block it is inside)
 * Fixed by computing a REINDEX layer's own n_vis as pos/cm directly
 * (forward_csa2.c's own comment on that line has the full argument) -
 * the same quantity a FULL layer's incrementally-updated counter always
 * equalled, computed instead of read through state that only FULL's own
 * step 6 keeps current.
 *
 * WHY THIS PROBE READS s->xb2 AND NOT s->attn_out: attn_out is a single
 * per-call scratch, overwritten by every token, so only the LAST token of
 * a call is ever inspectable through it - which is why build/csa2_mode_
 * dispatch_probe.c's own check5 is structurally blind to this bug, and
 * why a chunk's last position agrees under every batch width regardless.
 * forward_csa2.c's own step 5 writes o_proj's result to
 * `s->xb2 + tk*hdim`, ONE SLOT PER TOKEN, and nothing in the call
 * overwrites it. That is the observable a batch-invariance assertion has
 * to use.
 *
 * build/csa2_mode_dispatch_probe.c's own check_chain_soft was updated in
 * the SAME commit as the fix this probe guards - it used to assert the
 * defect as correct ("n_vis is already 3, constant across B's own
 * tokens"), which would have read as "the fix is wrong" once it turned
 * red on a correct fix.
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
#define F_SEQ   32
#define F_IDXH   2
#define F_IDXHD 16
#define F_IDXR  16
#define F_CM     2   /* > 1 so "sees its own block" is distinguishable */
#define F_TOPK   2
#define F_NT     4   /* > F_CM so a whole FUTURE block is visible      */
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
static void fill_xb(LZRunState *s, int nt, unsigned *st) {
    int t, i;
    for (t = 0; t < nt; t++)
        for (i = 0; i < F_DIM; i++)
            s->xb[(size_t)t * F_DIM + i] = lcgf(st);
}
static void layer(LZLayer *L, int mode, unsigned q, unsigned kv, unsigned ix) {
    unsigned a = q, b = kv, c = ix;
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL; L->csa2_on = 1; L->csa2_mode = mode;
    fill(&L->q_proj, F_QGD * F_DIM, &a);
    fill(&L->o_proj, F_DIM * F_QD, &a);
    fill(&L->q_norm, F_HD, &a);
    fill(&L->k_norm, F_HD, &b);
    fill(&L->csa2_w_kv, F_HD * F_DIM, &b);
    fill(&L->csa2_w_gate, F_HD * F_DIM, &b);
    fill(&L->csa2_w_kv_idx, F_IDXHD * F_DIM, &b);
    fill(&L->csa2_w_gate_idx, F_IDXHD * F_DIM, &b);
    fill(&L->csa2_w_dq_idx, F_IDXR * F_DIM, &c);
    fill(&L->csa2_w_uq_idx, (F_IDXH * F_IDXHD) * F_IDXR, &c);
    fill(&L->csa2_w_weights, F_IDXH * F_DIM, &c);
}

/* A(FULL) -> B(REINDEX, source A) -> C(REUSE, source B) */
static void build(LZModel *m) {
    LZModelConfig *c = &m->config;
    int i;
    memset(m, 0, sizeof(*m));
    memset(c, 0, sizeof(*c));
    c->vocab_size = 4; c->hidden_size = F_DIM; c->n_layers = 3; c->seq_len = F_SEQ;
    c->n_heads = F_NH; c->n_kv_heads = F_NH; c->head_dim = F_HD;
    c->intermediate_size = F_DIM;
    c->rotary_dim = 32; c->rope_theta = 10000.0f; c->rms_norm_eps = 1e-5f;
    c->attn_output_gate = 1;
    c->attn_q_dim = F_QD; c->attn_qgate_dim = F_QGD; c->attn_kv_dim = F_DIM;
    c->use_subn = 0;
    for (i = 0; i < 3; i++) c->layer_types[i] = LZ_LT_FULL;
    c->n_full_layers = 3;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->csa2_source[i] = -1;
    c->csa2_on[0] = 1; c->csa2_on[1] = 1; c->csa2_on[2] = 1;
    c->csa2_mode[1] = LZ_CSA2_REINDEX; c->csa2_source[1] = 0;
    c->csa2_mode[2] = LZ_CSA2_REUSE;   c->csa2_source[2] = 1;
    c->csa2_compress_m = F_CM; c->csa2_topk = F_TOPK;
    c->csa2_indexer_heads = F_IDXH;
    c->csa2_indexer_head_dim = F_IDXHD;
    c->csa2_indexer_q_lora_rank = F_IDXR;
    m->layers = (LZLayer *)calloc(3, sizeof(LZLayer));
    layer(&m->layers[0], LZ_CSA2_FULL,    SEED0 + 1, SEED0 + 2, SEED0 + 3);
    layer(&m->layers[1], LZ_CSA2_REINDEX, SEED0 + 4, SEED0 + 5, SEED0 + 6);
    layer(&m->layers[2], LZ_CSA2_REUSE,   SEED0 + 7, SEED0 + 8, SEED0 + 9);
}

#define XA (SEED0 + 20)
#define XB (SEED0 + 21)
#define XC (SEED0 + 22)

int main(void) {
    LZModel m;
    LZRunState s;
    char eb[256];
    unsigned st;
    int tk, ci, ok_causal = 1;
    float xb2_b[3][F_NT][F_DIM];   /* [layer][token] under nt=F_NT */
    float xb2_u[3][F_NT][F_DIM];   /* [layer][token] under nt=1    */
    int n_b[F_NT];

    /* run 1: one nt=F_NT call per layer (what prefill does) */
    build(&m);
    if (lz_state_alloc(&s, &m, F_SEQ, 0, eb, sizeof eb) != 0) {
        printf("FAIL c2: lz_state_alloc: %s\n", eb);
        return 1;
    }
    ci = s.cache_idx[0];
    st = XA; fill_xb(&s, F_NT, &st);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, F_NT, NULL);
    for (tk = 0; tk < F_NT; tk++)
        memcpy(xb2_b[0][tk], s.xb2 + (size_t)tk * F_DIM, F_DIM * sizeof(float));
    st = XB; fill_xb(&s, F_NT, &st);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, F_NT, NULL);
    for (tk = 0; tk < F_NT; tk++) {
        memcpy(xb2_b[1][tk], s.xb2 + (size_t)tk * F_DIM, F_DIM * sizeof(float));
        n_b[tk] = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
    }
    st = XC; fill_xb(&s, F_NT, &st);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, F_NT, NULL);
    for (tk = 0; tk < F_NT; tk++)
        memcpy(xb2_b[2][tk], s.xb2 + (size_t)tk * F_DIM, F_DIM * sizeof(float));
    lz_state_free(&s);
    free(m.layers);

    /* CHECK 1 - CAUSALITY. engine/csa2.py's csa2_causal_block_mask: block
       s is visible to position t iff s < t/cm, so k_eff == min(topk,
       pos/cm). Asserted on the REINDEX layer, where the shared counter
       (already at floor(chunk_end/cm)) and the causal answer differ. */
    for (tk = 0; tk < F_NT; tk++) {
        int want = tk / F_CM;
        if (want > F_TOPK) want = F_TOPK;
        if (n_b[tk] != want) {
            printf("   token %d: published k_eff=%d, causal answer "
                   "min(topk, %d/%d)=%d\n", tk, n_b[tk], tk, F_CM, want);
            ok_causal = 0;
        }
    }
    ok(ok_causal,
       "c2/check1: the REINDEX layer's own k_eff is min(topk, pos/cm) for "
       "every token - it sees only blocks finalized by STRICTLY EARLIER "
       "positions, not every block its (already-finished) source built");

    /* run 2: F_NT separate nt=1 calls, layer-major per position - exactly
       forward.c's own dispatch order at chunk width 1 */
    build(&m);
    if (lz_state_alloc(&s, &m, F_SEQ, 0, eb, sizeof eb) != 0) {
        printf("FAIL c2: lz_state_alloc (unbatched): %s\n", eb);
        return 1;
    }
    {
        unsigned sa = XA, sb = XB, sc = XC;
        int pos;
        for (pos = 0; pos < F_NT; pos++) {
            fill_xb(&s, 1, &sa);
            forward_csa2(&m, &s, &m.layers[0], 0, pos, 1, NULL);
            memcpy(xb2_u[0][pos], s.xb2, F_DIM * sizeof(float));
            fill_xb(&s, 1, &sb);
            forward_csa2(&m, &s, &m.layers[1], 1, pos, 1, NULL);
            memcpy(xb2_u[1][pos], s.xb2, F_DIM * sizeof(float));
            fill_xb(&s, 1, &sc);
            forward_csa2(&m, &s, &m.layers[2], 2, pos, 1, NULL);
            memcpy(xb2_u[2][pos], s.xb2, F_DIM * sizeof(float));
        }
    }
    lz_state_free(&s);
    free(m.layers);

    /* CHECK 2 - BATCH INVARIANCE, the CLI's own documented contract
       ("--batch N ... Bit-identical at every width"). EVERY token, not
       just the last: a chunk's last position agrees under every width by
       construction, which is exactly the blindness that let this ship. */
    {
        int li, bad = 0;
        for (li = 0; li < 3; li++) {
            for (tk = 0; tk < F_NT; tk++) {
                if (memcmp(xb2_b[li][tk], xb2_u[li][tk],
                           F_DIM * sizeof(float)) != 0) {
                    printf("   layer %d token %d: nt=%d gives %.9g, "
                           "nt=1 gives %.9g\n", li, tk, F_NT,
                           (double)xb2_b[li][tk][0], (double)xb2_u[li][tk][0]);
                    bad = 1;
                }
            }
        }
        ok(!bad,
           "c2/check2: every token's o_proj output is bit-identical between "
           "one nt=4 call per layer and four nt=1 layer-major calls, for all "
           "three layers - not merely for the chunk's last token");
    }

    printf("\nc2_probe: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
