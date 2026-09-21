/* forward_csa2.c's causal block mask: WHICH compressed blocks a query
 * at position pos may score, for each of the three CSA2 modes.
 *
 * THE BOUNDARY. A block becomes visible once the query has passed its
 * last raw token, so for 0-indexed pos the count is (pos + 1) / cm -
 * the reference implementation's own compress_lens ((arange(1, seqlen+1)
 * // ratio) while prefilling, end_pos // ratio while decoding;
 * .prof/dsv41_ref/model.py 564 and 567 in the training tree). The
 * published k_eff is min(topk, n_vis), which is what this probe reads:
 * forward_csa2.c has no n_vis accessor, and k_eff is the same quantity
 * through a clamp this probe's own fixtures keep out of the way where
 * it matters.
 *
 * TWO WAYS TO GET THIS WRONG, and this probe separates them:
 *   - ONE BLOCK TIGHT, n_vis = pos / cm. Conservative (never a future
 *     block) but not free: at cm=2 every odd position waits a step
 *     longer than the reference to see its own two most recent raw
 *     positions, and at cm=1 - the report's own named ratio-1 special
 *     case (SS2.3), which is what the reference's whole decoder half
 *     runs - a query can never see the block holding its own position
 *     at all. check2/check4 below are the ones that name this.
 *   - A COUNT CARRIED IN SHARED STATE rather than derived from pos.
 *     A REINDEX layer's ci is ALIASED (forward_state.c's cache_idx
 *     fill) onto its source's, and the source's call for the whole
 *     chunk finishes before the REINDEX layer's own call starts
 *     (forward.c dispatches layers in increasing order), so any count
 *     living in that shared state is already at the CHUNK'S END by the
 *     time this layer reads it - every block the source finalized on
 *     LATER tokens, leaked into an earlier token's query. check1 and
 *     check3 are the ones that name this.
 *
 * WHY CHECK 3 READS s->xb2 AND NOT s->attn_out: attn_out is a single
 * per-call scratch, overwritten by every token, so only the LAST token
 * of a call is ever inspectable through it - which is why build/csa2_
 * mode_dispatch_probe.c's own check5 is structurally blind to a
 * per-chunk count, and why a chunk's last position agrees under every
 * batch width regardless. forward_csa2.c's own step 6 writes o_proj's
 * result to `s->xb2 + tk*hdim`, ONE SLOT PER TOKEN, and nothing in the
 * call overwrites it. That is the observable a batch-invariance
 * assertion has to use.
 *
 * Built and run by build/csa2_reindex_causality_gate.sh against the
 * real src/forward_csa2.c and against two mutants of its n_vis line -
 * one per failure mode above. See that script for the substitutions and
 * the expected fallout per mutant.
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
#define F_CM     2   /* > 1 so "the block a position is inside" is a
                        distinct block from "the one before it"       */
#define F_TOPK   2
#define F_NT     4   /* > F_CM so a whole FUTURE block exists to leak  */
/* The ratio-1 arm. topk is raised to F_NT so min(topk, n_vis) never
   clamps anywhere in that arm: the published k_eff IS n_vis there, and
   the published index set is every visible block, which is what lets
   check4 assert "block pos itself is among them" directly. */
#define F_CM1    1
#define F_TOPK1  F_NT
#define F_MAXTK  F_TOPK1   /* widest published slot any arm here reads */
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
    fill(&L->csa2_w_kv_idx, F_IDXHD * F_HD, &b);
    fill(&L->csa2_k_norm_idx, F_IDXHD, &b);
    fill(&L->csa2_w_dq_idx, F_IDXR * F_DIM, &c);
    fill(&L->csa2_w_uq_idx, (F_IDXH * F_IDXHD) * F_IDXR, &c);
    fill(&L->csa2_w_weights, F_IDXH * F_DIM, &c);
}

/* A(FULL) -> B(REINDEX, source A) -> C(REUSE, source B), all three
   aliased onto one physical ci. cm/topk are parameters because the
   ratio-1 arm needs its own pair - see F_CM1/F_TOPK1 above. */
static void build(LZModel *m, int cm, int topk) {
    LZModelConfig *c = &m->config;
    int i;
    memset(m, 0, sizeof(*m));
    memset(c, 0, sizeof(*c));
    c->vocab_size = 4; c->hidden_size = F_DIM; c->n_layers = 3; c->seq_len = F_SEQ;
    c->n_heads = F_NH; c->n_kv_heads = F_NH; c->head_dim = F_HD;
    c->intermediate_size = F_DIM;
    /* Rotation width must not exceed csa2_indexer_head_dim (set below) -
       the indexer ring's rows are indexer_head_dim wide and lz_rope
       writes base[i + rotary_dim/2], so a wider rotary_dim runs off the
       row. Same constraint as csa2_mode_dispatch_probe.c's own fixture. */
    c->rotary_dim = F_IDXHD; c->rope_theta = 10000.0f; c->rms_norm_eps = 1e-5f;
    c->attn_output_gate = 1;
    c->attn_q_dim = F_QD; c->attn_qgate_dim = F_QGD; c->attn_kv_dim = F_DIM;
    c->use_subn = 0;
    for (i = 0; i < 3; i++) c->layer_types[i] = LZ_LT_FULL;
    c->n_full_layers = 3;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->csa2_source[i] = -1;
    c->csa2_on[0] = 1; c->csa2_on[1] = 1; c->csa2_on[2] = 1;
    c->csa2_mode[1] = LZ_CSA2_REINDEX; c->csa2_source[1] = 0;
    c->csa2_mode[2] = LZ_CSA2_REUSE;   c->csa2_source[2] = 1;
    c->csa2_compress_m = cm; c->csa2_topk = topk;
    c->csa2_indexer_heads = F_IDXH;
    c->csa2_indexer_head_dim = F_IDXHD;
    c->csa2_indexer_q_lora_rank = F_IDXR;
    m->layers = (LZLayer *)calloc(3, sizeof(LZLayer));
    layer(&m->layers[0], LZ_CSA2_FULL,    SEED0 + 1, SEED0 + 2, SEED0 + 3);
    layer(&m->layers[1], LZ_CSA2_REINDEX, SEED0 + 4, SEED0 + 5, SEED0 + 6);
    layer(&m->layers[2], LZ_CSA2_REUSE,   SEED0 + 7, SEED0 + 8, SEED0 + 9);
}

/* Read this ci's published (k_eff, index set) for token tk. The three
   layers share one ci, so a layer's own publication has to be collected
   before the next layer's call overwrites it. */
static void read_pub(const LZRunState *s, int ci, int topk, int tk,
                     int *n_out, int *idx_out) {
    int k;
    *n_out = s->csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
    for (k = 0; k < topk; k++)
        idx_out[k] = s->csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * topk + k];
}

#define XA (SEED0 + 20)
#define XB (SEED0 + 21)
#define XC (SEED0 + 22)

int main(void) {
    LZModel m;
    LZRunState s;
    char eb[256];
    unsigned st;
    int tk, ci, k;
    float xb2_b[3][F_NT][F_DIM];   /* [layer][token] under nt=F_NT */
    float xb2_u[3][F_NT][F_DIM];   /* [layer][token] under nt=1    */
    int n_a[F_NT], n_b[F_NT];
    int dummy_idx[F_MAXTK];

    /* run 1: one nt=F_NT call per layer (what prefill does) */
    build(&m, F_CM, F_TOPK);
    if (lz_state_alloc(&s, &m, F_SEQ, 0, eb, sizeof eb) != 0) {
        printf("FAIL c2: lz_state_alloc: %s\n", eb);
        return 1;
    }
    ci = s.cache_idx[0];
    st = XA; fill_xb(&s, F_NT, &st);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, F_NT, NULL);
    for (tk = 0; tk < F_NT; tk++) {
        memcpy(xb2_b[0][tk], s.xb2 + (size_t)tk * F_DIM, F_DIM * sizeof(float));
        read_pub(&s, ci, F_TOPK, tk, &n_a[tk], dummy_idx);
    }
    st = XB; fill_xb(&s, F_NT, &st);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, F_NT, NULL);
    for (tk = 0; tk < F_NT; tk++) {
        memcpy(xb2_b[1][tk], s.xb2 + (size_t)tk * F_DIM, F_DIM * sizeof(float));
        read_pub(&s, ci, F_TOPK, tk, &n_b[tk], dummy_idx);
    }
    st = XC; fill_xb(&s, F_NT, &st);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, F_NT, NULL);
    for (tk = 0; tk < F_NT; tk++)
        memcpy(xb2_b[2][tk], s.xb2 + (size_t)tk * F_DIM, F_DIM * sizeof(float));
    lz_state_free(&s);
    free(m.layers);

    /* CHECK 1 - the REINDEX layer's own count is PER POSITION. Its ci
       is aliased onto a source whose whole call already finished, so
       every block of this chunk is already sitting in the shared ring;
       nothing but this count keeps a token out of the later ones. */
    {
        int bad = 0;
        for (tk = 0; tk < F_NT; tk++) {
            int want = (tk + 1) / F_CM;
            if (want > F_TOPK) want = F_TOPK;
            if (n_b[tk] != want) {
                printf("   token %d: published k_eff=%d, want min(topk, "
                       "(%d+1)/%d)=%d\n", tk, n_b[tk], tk, F_CM, want);
                bad = 1;
            }
        }
        ok(!bad,
           "c2/check1: the REINDEX layer's own k_eff is min(topk, (pos+1)/cm) "
           "for every token - derived from that token's own position, not "
           "from a count its (already-finished) source left in their shared "
           "state");
    }

    /* CHECK 2 - THE BLOCK BOUNDARY, on a FULL layer, named at the exact
       position where the two candidate formulas disagree: pos = j*cm +
       cm - 1, the last raw token of block j, where the reference counts
       block j as visible (n_vis = j+1) and a pos/cm engine does not
       (j). With F_CM=2/F_NT=4 those positions are 1 and 3, and this
       check asserts the whole four-token sequence so the two positions
       in between are pinned as well. */
    {
        int bad = 0;
        for (tk = 0; tk < F_NT; tk++) {
            int want = (tk + 1) / F_CM;
            if (want > F_TOPK) want = F_TOPK;
            if (n_a[tk] != want) {
                printf("   token %d (off=%d): published k_eff=%d, want "
                       "min(topk, (%d+1)/%d)=%d\n", tk, tk % F_CM, n_a[tk],
                       tk, F_CM, want);
                bad = 1;
            }
        }
        ok(!bad,
           "c2/check2: the FULL layer's own k_eff is min(topk, (pos+1)/cm) - "
           "on the LAST token of a block (pos%cm == cm-1) the block that "
           "token just closed is already visible to it, matching the "
           "reference's own compress_lens");
    }

    /* run 2: F_NT separate nt=1 calls, layer-major per position - exactly
       forward.c's own dispatch order at chunk width 1 */
    build(&m, F_CM, F_TOPK);
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

    /* CHECK 3 - BATCH INVARIANCE, the CLI's own documented contract
       ("--batch N ... Bit-identical at every width"). EVERY token, not
       just the last: a chunk's last position agrees under every width by
       construction, which is exactly the blindness that let this ship.
       A per-position count satisfies this for free; a count read out of
       chunk-wide shared state cannot. */
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
           "c2/check3: every token's o_proj output is bit-identical between "
           "one nt=4 call per layer and four nt=1 layer-major calls, for all "
           "three layers - not merely for the chunk's last token");
    }

    /* run 3: the ratio-1 arm. Every position closes its own block, so
       the block holding pos IS block pos and "can a query see the block
       it is inside" becomes a question with a directly readable answer:
       n_vis == pos + 1, and (F_TOPK1 being wide enough that nothing
       clamps) the published index set is every block up to and
       including pos. FULL and REINDEX both, since they reach that count
       by different routes - one finalizes the block itself, the other
       reads a ring its source filled. */
    {
        int n_pos[F_NT], idx_pos[F_NT][F_MAXTK];
        int li;
        static const char *const WHO[2] = { "FULL", "REINDEX" };

        build(&m, F_CM1, F_TOPK1);
        if (lz_state_alloc(&s, &m, F_SEQ, 0, eb, sizeof eb) != 0) {
            printf("FAIL c2: lz_state_alloc (cm=1): %s\n", eb);
            return 1;
        }
        ci = s.cache_idx[0];
        for (li = 0; li < 2; li++) {
            int bad_n = 0, bad_self = 0;
            st = XA + (unsigned)li; fill_xb(&s, F_NT, &st);
            forward_csa2(&m, &s, &m.layers[li], li, 0, F_NT, NULL);
            for (tk = 0; tk < F_NT; tk++)
                read_pub(&s, ci, F_TOPK1, tk, &n_pos[tk], idx_pos[tk]);
            for (tk = 0; tk < F_NT; tk++) {
                int saw_self = 0;
                if (n_pos[tk] != tk + 1) {
                    printf("   %s token %d: published k_eff=%d, want %d\n",
                           WHO[li], tk, n_pos[tk], tk + 1);
                    bad_n = 1;
                }
                for (k = 0; k < n_pos[tk] && k < F_TOPK1; k++)
                    if (idx_pos[tk][k] == tk) saw_self = 1;
                if (!saw_self) {
                    printf("   %s token %d: block %d (the one holding this "
                           "position) is not in the published selection\n",
                           WHO[li], tk, tk);
                    bad_self = 1;
                }
            }
            if (li == 0) {
                ok(!bad_n && !bad_self,
                   "c2/check4: at cm==1 (the report's own uncompressed-main-KV "
                   "special case, and the whole decoder half of the reference) "
                   "a FULL layer's query at pos sees exactly pos+1 blocks, the "
                   "block holding pos itself among them");
            } else {
                ok(!bad_n && !bad_self,
                   "c2/check5: the same at cm==1 for a REINDEX layer, which "
                   "reaches that count through its source's ring rather than "
                   "by finalizing the block itself");
            }
        }
        lz_state_free(&s);
        free(m.layers);
    }

    printf("\nc2_probe: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail;
}
