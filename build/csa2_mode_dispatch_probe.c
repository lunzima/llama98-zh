/* Full/Reindex/Reuse mode dispatch in forward_csa2.c (report Sec2.3.1,
   Part 5 Task 5.4 of the same plan build/csa2_modes_config_probe.c and
   build/csa2_cache_idx_alias_probe.c landed the schema/aliasing halves
   of). Those two tasks deliberately left forward_csa2.c reading nothing
   of L->csa2_mode; this is the task that makes it actually dispatch.

   Built and run by build/csa2_mode_dispatch_gate.sh against the real
   src/forward_csa2.c/forward_state.c/forward.h (not a transcription),
   and against two independent mutant copies:
     - "dispatch reverted": every mode always runs the pre-this-task,
       unconditional six-step body (steps 2/3/6 lose their csa2_mode
       guards) - must redden the Reindex/Reuse-specific checks below
       while leaving check1's Full-only checks green.
     - "publish buffer collapsed to per-ci": the (ci, tk) indexing on
       csa2_topk_idx_pub/csa2_topk_n_pub loses its "+ tk" term, back to
       a single (topk)-wide slot per ci - must redden specifically the
       per-token publish check inside check_chain_soft (item 5), since
       that is the one thing a per-ci-only buffer cannot get right for
       nt > 1.
   See that gate script's own header for the exact substitutions and
   why each is a text mutation rather than a git-ref diff.

   No config.json, no safetensors: every LZModelConfig/LZLayer field is
   set by hand, the same shape build/csa2_cache_idx_alias_probe.c and
   build/ced_proj_arm_probe.c's PART 2 both use - unlike either of
   those, this probe calls the real forward_csa2() DIRECTLY (it is a
   non-static, externally-declared function, forward.h's own
   declaration) rather than routing through the whole per-layer
   lz_forward_batch stack: this task's change is entirely internal to
   forward_csa2.c's own per-layer, per-token bookkeeping, and calling it
   directly lets each fixture drive one layer's own multi-token chunk in
   full, explicit control of pos0/nt/kv_in, and inspect the persistent
   publish buffers afterward - exactly the granularity the design this
   task adds needs.

   Fixture weight tensors are grouped into three independent LCG
   streams per layer - "q" (q_proj/o_proj/q_norm, step 1+4+5, every
   mode owns these), "kv" (k_norm + the four main/indexer COMPRESSION
   tensors, step 2+6, FULL only) and "idx" (the three indexer QUERY
   tensors, step 3, FULL+REINDEX) - so a check can perturb exactly one
   group and attribute an observed change to exactly the step that
   reads it. A REINDEX layer is given a full "kv" group too (matching
   what a real checkpoint carries today - model.c's tensor loader does
   not yet skip csa2_w_kv et al. for a non-FULL layer, a fact this
   file's own comment on build_layer_reindex records) even though the
   correct dispatch never reads it, precisely so the "dispatch
   reverted" mutation above computes something (wrong, and caught) in
   that branch instead of dereferencing a genuinely absent tensor and
   crashing before this probe can print anything. build_layer_reuse_
   strict is the one deliberate exception - see its own comment and
   check3b's.

   csa2_compress_m=1 (every token finalizes its own block immediately)
   and csa2_topk=2 are chosen so a 3-token chunk exercises n_vis=0,1,2
   for whichever layer is doing the finalizing, and a real (not
   trivial-"select everything") Top-K choice as soon as n_vis exceeds
   topk.

   Two code-review addenda, added after the mode-dispatch checks above
   were first landed and reviewed:
     - check_overflow_guard: lz_csa2_pub_overflows (forward_state.c)
       guards the NEW nf * LZ_BATCH_MAX * csa2_topk element count this
       task's publish buffers introduced - a three-factor product where
       the old, single-factor csa2_topk-only allocation needed a topk
       past roughly one billion to wrap a 32-bit size_t, this one wraps
       past roughly 4.19 million, well inside a plausible (if unusual)
       config value. Tested with synthetic divisor values derived from
       size_t's own maximum, not from LZ_MAX_LAYERS/LZ_BATCH_MAX,
       because the real production dimensions cannot combine with an
       int-range csa2_topk to overflow a 64-bit size_t at all - see the
       function's own header comment.
     - check_bin_validation: lz_csa2_mode_source_valid (model.c) closes
       the gap left when build/csa2_cache_idx_alias_gate.sh's own task
       landed cache_idx aliasing without a bin-format consumer to
       validate against - lz_open_bin's own header parsing checked only
       that csa2_mode[]/csa2_source[] were untouched PAST n_layers, never
       that an IN-RANGE entry was internally consistent, so a corrupted
       or hand-crafted model.bin could alias forward_state.c's cache_idx
       fill into a negative index or the wrong numbering space. Tested
       directly against the predicate (a hand-built LZModelConfig, same
       shape as PART 1 of build/ced_proj_arm_probe.c for lz_ced_kv_target/
       lz_kv_share_target) rather than through a synthetic model.bin
       file: lz_open_bin calls the same predicate, so there is no
       separate "real" code path this could diverge from, and hand-
       encoding the ~1300-byte v5 header byte for byte would be a
       disproportionate amount of format plumbing for one predicate.

   Exit: 0 all checks pass, 1 a check failed (printed). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "forward.h"
#include "err.h"

static int g_fail = 0;

static void ok(int cond, const char *what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    /* Flushed immediately, not left to exit-time buffering: check3b is
       deliberately last and deliberately crash-prone under this gate's
       own "dispatch reverted" mutation (its own comment) - a SIGSEGV
       there must not erase every earlier check's already-printed line
       along with it, and stdio is fully buffered (not line-buffered)
       once stdout is redirected to a file, which is exactly what the
       gate script does. */
    fflush(stdout);
    if (!cond) g_fail = 1;
}

/* ---------------------------------------------------------- fixture sizing */

/* head_dim a multiple of 32 - build/ced_proj_arm_probe.c's own comment
   on this exact trap (the KV cache's Q8 quantization group size), still
   allocated for every full-attention layer regardless of csa2_on. */
#define F_DIM     64
#define F_NH       2
#define F_HD      32
#define F_KVD     64
#define F_QD      64
#define F_QGD    128
#define F_INTER   64
#define F_VOCAB    4
#define F_ROT     32
#define F_SEQLEN  32
#define F_IDXH     2
#define F_IDXHD   16
#define F_IDXR    16
#define F_CM       1
#define F_TOPK     2

#define SEED0 1145141919u   /* this project's own fixed synthetic-input seed */

static unsigned lcg_next(unsigned *st) {
    *st = *st * 1103515245u + 12345u;
    return *st;
}
/* [-0.2, 0.2) - small, non-zero, decorrelated-enough; same shape as
   build/ced_proj_arm_probe.c's own p2_next. */
static float lcg_f(unsigned *st) {
    return (float)((int)((lcg_next(st) >> 8) % 2000u) - 1000) * 0.0002f;
}

static void fill_f(LZTensor *t, int n, unsigned *st) {
    int i;
    memset(t, 0, sizeof(*t));
    t->dtype = LZ_FMT_F32;
    t->n = n;
    t->gs = 0;
    t->f = (float *)malloc((size_t)n * sizeof(float));
    for (i = 0; i < n; i++) t->f[i] = lcg_f(st);
}

static void fill_xb(LZRunState *s, int nt, int dim, unsigned *st) {
    int tk, i;
    for (tk = 0; tk < nt; tk++)
        for (i = 0; i < dim; i++)
            s->xb[(size_t)tk * dim + i] = lcg_f(st);
}

/* ------------------------------------------------------------ layer fixtures */

/* q/o/q_norm (step 1, 4, 5 - every mode) + k_norm and the four main/
   indexer COMPRESSION tensors (step 2, 6 - FULL only) + the three
   indexer QUERY tensors (step 3 - FULL and REINDEX). A REINDEX layer
   gets the SAME full set - see this file's own header comment on why:
   the correct dispatch never reads the "kv" group for REINDEX, but a
   real checkpoint carries it regardless (model.c loads all seven
   csa2_w_* tensors for every csa2_on layer, mode notwithstanding - a
   pre-existing, unrelated fact about the loader, not this task's own
   scope), and giving REINDEX the same tensors here means a dispatch
   bug that wrongly ran FULL's steps for a REINDEX layer computes a
   wrong-but-finite answer instead of crashing before it can be caught
   by value comparison. */
static void fill_layer_common(LZLayer *L, unsigned seed_q, unsigned seed_kv, unsigned seed_idx) {
    unsigned stq = seed_q, stkv = seed_kv, stix = seed_idx;
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL;
    L->csa2_on = 1;
    fill_f(&L->q_proj, F_QGD * F_DIM, &stq);
    fill_f(&L->o_proj, F_DIM * F_QD, &stq);
    fill_f(&L->q_norm, F_HD, &stq);
    fill_f(&L->k_norm, F_HD, &stkv);
    fill_f(&L->csa2_w_kv, F_HD * F_DIM, &stkv);
    fill_f(&L->csa2_w_gate, F_HD * F_DIM, &stkv);
    fill_f(&L->csa2_w_kv_idx, F_IDXHD * F_DIM, &stkv);
    fill_f(&L->csa2_w_gate_idx, F_IDXHD * F_DIM, &stkv);
    fill_f(&L->csa2_w_dq_idx, F_IDXR * F_DIM, &stix);
    fill_f(&L->csa2_w_uq_idx, (F_IDXH * F_IDXHD) * F_IDXR, &stix);
    fill_f(&L->csa2_w_weights, F_IDXH * F_DIM, &stix);
}

static void build_layer_full(LZLayer *L, unsigned sq, unsigned skv, unsigned six) {
    fill_layer_common(L, sq, skv, six);
    L->csa2_mode = LZ_CSA2_FULL;
}

static void build_layer_reindex(LZLayer *L, unsigned sq, unsigned skv, unsigned six) {
    fill_layer_common(L, sq, skv, six);
    L->csa2_mode = LZ_CSA2_REINDEX;
}

/* Same full tensor set as build_layer_full/build_layer_reindex, mode
   REUSE - used only to give the "dispatch reverted" mutation something
   finite to compute for this layer too (this file's own header
   comment). The CORRECT dispatch never reads any of these seven for a
   REUSE layer; build_layer_reuse_strict below is the fixture that
   actually proves that. */
static void build_layer_reuse_soft(LZLayer *L, unsigned sq, unsigned skv, unsigned six) {
    fill_layer_common(L, sq, skv, six);
    L->csa2_mode = LZ_CSA2_REUSE;
}

/* The genuine article for check3b: q_proj/o_proj/q_norm only. k_norm
   and all seven csa2_w_* tensors are left at memset's zero - dtype
   LZ_FMT_F32 (0), n 0, f NULL - because a REUSE layer owns none of
   them (model.h's csa2_mode[] comment; this file's own header comment
   on the seven-tensor grouping). If the dispatch ever mistakenly read
   one, lz_matmul_w would dereference a NULL LZTensor.f and this
   process would crash immediately rather than silently computing over
   garbage - see check3b_reuse_owns_nothing's own comment for why that
   crash, if it ever happens, is the assertion. */
static void build_layer_reuse_strict(LZLayer *L, unsigned sq) {
    unsigned stq = sq;
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL;
    L->csa2_on = 1;
    L->csa2_mode = LZ_CSA2_REUSE;
    fill_f(&L->q_proj, F_QGD * F_DIM, &stq);
    fill_f(&L->o_proj, F_DIM * F_QD, &stq);
    fill_f(&L->q_norm, F_HD, &stq);
}

/* ------------------------------------------------------------ model fixtures */

static void init_config_common(LZModelConfig *c, int n_layers) {
    int i;
    memset(c, 0, sizeof(*c));
    c->vocab_size = F_VOCAB;
    c->hidden_size = F_DIM;
    c->n_layers = n_layers;
    c->seq_len = F_SEQLEN;
    c->n_heads = F_NH;
    c->n_kv_heads = F_NH;
    c->head_dim = F_HD;
    c->intermediate_size = F_INTER;
    c->rotary_dim = F_ROT;
    c->rope_theta = 10000.0f;
    c->rms_norm_eps = 1e-5f;
    c->attn_output_gate = 1;
    c->attn_q_dim = F_QD;
    c->attn_qgate_dim = F_QGD;
    c->attn_kv_dim = F_KVD;
    c->use_subn = 0;
    for (i = 0; i < n_layers; i++) c->layer_types[i] = LZ_LT_FULL;
    c->n_full_layers = n_layers;
    c->n_linear_layers = 0;
    for (i = 0; i < LZ_MAX_LAYERS; i++) c->csa2_source[i] = -1;
    c->csa2_compress_m = F_CM;
    c->csa2_topk = F_TOPK;
    c->csa2_indexer_heads = F_IDXH;
    c->csa2_indexer_head_dim = F_IDXHD;
    c->csa2_indexer_q_lora_rank = F_IDXR;
}

/* 1 layer, FULL, csa2_on - check1's Full-only regression fixture. */
static void build_model_single(LZModel *m, unsigned aq, unsigned akv, unsigned aix) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 1);
    m->config.csa2_on[0] = 1;
    m->layers = (LZLayer *)calloc(1, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
}

/* 3-layer transitive chain A(FULL) -> B(REINDEX, source A) -> C(REUSE,
   source B) - the exact shape build/csa2_cache_idx_alias_probe.c's own
   case A already proved resolves cache_idx aliasing without an
   explicit walk. "Soft" C: see build_layer_reuse_soft's own comment. */
static void build_model_chain_soft(LZModel *m,
                                   unsigned aq, unsigned akv, unsigned aix,
                                   unsigned bq, unsigned bkv, unsigned bix,
                                   unsigned cq, unsigned ckv, unsigned cix) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 3);
    m->config.csa2_on[0] = 1;
    m->config.csa2_on[1] = 1;
    m->config.csa2_on[2] = 1;
    m->config.csa2_source[1] = 0;
    m->config.csa2_source[2] = 1;
    m->layers = (LZLayer *)calloc(3, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
    build_layer_reindex(&m->layers[1], bq, bkv, bix);
    build_layer_reuse_soft(&m->layers[2], cq, ckv, cix);
}

/* Same chain shape, "strict" C - build_layer_reuse_strict's own comment. */
static void build_model_chain_strict(LZModel *m,
                                     unsigned aq, unsigned akv, unsigned aix,
                                     unsigned bq, unsigned bkv, unsigned bix,
                                     unsigned cq) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 3);
    m->config.csa2_on[0] = 1;
    m->config.csa2_on[1] = 1;
    m->config.csa2_on[2] = 1;
    m->config.csa2_source[1] = 0;
    m->config.csa2_source[2] = 1;
    m->layers = (LZLayer *)calloc(3, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
    build_layer_reindex(&m->layers[1], bq, bkv, bix);
    build_layer_reuse_strict(&m->layers[2], cq);
}

/* 2-layer CED fixture: D1(FULL) in the decoder region, D2(REINDEX,
   source D1). No forward.c/lz_forward involved - kv_in is passed to
   forward_csa2 directly by this file, the same parameter forward.c's
   own kv_in resolution would have computed for a decoder-region CSA2
   layer under ced_kv=proj (this file's own header comment on why a
   direct call is the right granularity for this task). */
static void build_model_ced(LZModel *m,
                            unsigned aq, unsigned akv, unsigned aix,
                            unsigned bq, unsigned bkv, unsigned bix) {
    memset(m, 0, sizeof(*m));
    init_config_common(&m->config, 2);
    m->config.csa2_on[0] = 1;
    m->config.csa2_on[1] = 1;
    m->config.csa2_source[1] = 0;
    m->layers = (LZLayer *)calloc(2, sizeof(LZLayer));
    build_layer_full(&m->layers[0], aq, akv, aix);
    build_layer_reindex(&m->layers[1], bq, bkv, bix);
}

/* ------------------------------------------------------------------ checks */

/* Item 1: Full-mode-only regression. Every currently-shipping checkpoint
   uses Full mode exclusively, so this task's dispatch must be a
   complete no-op for it. */
static void check1_full_only(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    unsigned stx;
    int ci;

    /* 1a: the very first token this ci ever sees has n_vis==0 - k_eff
       must be exactly 0 and attn_out the exact zero vector (the
       existing, already-shipped "no legitimate contribution" case,
       true regardless of this task's own change). */
    build_model_single(&m, SEED0 + 1, SEED0 + 2, SEED0 + 3);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check1a: lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    ci = s.cache_idx[0];
    stx = SEED0 + 10;
    fill_xb(&s, 1, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 1, NULL);
    ok(s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + 0] == 0,
       "check1a: first-ever token's published k_eff is 0");
    {
        int d, allz = 1;
        for (d = 0; d < F_QD; d++) if (s.attn_out[d] != 0.0f) allz = 0;
        ok(allz, "check1a: k_eff==0 token's attn_out is the exact zero vector");
    }
    lz_state_free(&s);
    free(m.layers);

    /* 1b: batch-vs-unbatched bit-identity, this project's own
       established idiom for "the per-token ring bookkeeping does not
       depend on how many tokens one call covers" (build/csa2_gate.sh's
       own batch4-vs-batch1 arm), applied to the mode-dispatch code
       this task adds: three tokens through ONE forward_csa2 call must
       match three tokens through three separate single-token calls,
       token for token, including the published Top-K buffer this task
       reshapes. */
    {
        LZModel m3;
        LZRunState s3;
        int pub_n3[3], pub_idx3[3][F_TOPK];
        float attn_last3[F_QD];
        int tk, k;

        build_model_single(&m3, SEED0 + 1, SEED0 + 2, SEED0 + 3);
        if (lz_state_alloc(&s3, &m3, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check1b: lz_state_alloc (batched): %s\n", errbuf);
            g_fail = 1;
            free(m3.layers);
            return;
        }
        ci = s3.cache_idx[0];
        stx = SEED0 + 20;
        fill_xb(&s3, 3, F_DIM, &stx);
        forward_csa2(&m3, &s3, &m3.layers[0], 0, 0, 3, NULL);
        for (tk = 0; tk < 3; tk++) {
            pub_n3[tk] = s3.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
            for (k = 0; k < F_TOPK; k++)
                pub_idx3[tk][k] = s3.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k];
        }
        memcpy(attn_last3, s3.attn_out, sizeof attn_last3);
        lz_state_free(&s3);
        free(m3.layers);

        {
            LZModel m1;
            LZRunState s1;
            unsigned stx1 = SEED0 + 20;
            int match_n = 1, match_idx = 1;

            build_model_single(&m1, SEED0 + 1, SEED0 + 2, SEED0 + 3);
            if (lz_state_alloc(&s1, &m1, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
                printf("FAIL check1b: lz_state_alloc (unbatched): %s\n", errbuf);
                g_fail = 1;
                free(m1.layers);
                return;
            }
            ci = s1.cache_idx[0];
            for (tk = 0; tk < 3; tk++) {
                fill_xb(&s1, 1, F_DIM, &stx1);
                forward_csa2(&m1, &s1, &m1.layers[0], 0, tk, 1, NULL);
                if (s1.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + 0] != pub_n3[tk])
                    match_n = 0;
                for (k = 0; k < F_TOPK; k++)
                    if (s1.csa2_topk_idx_pub[(size_t)ci * LZ_BATCH_MAX * F_TOPK + k] != pub_idx3[tk][k])
                        match_idx = 0;
            }
            ok(match_n,
               "check1b: batch-vs-unbatched - published k_eff matches token for token");
            ok(match_idx,
               "check1b: batch-vs-unbatched - published Top-K indices match token for token");
            ok(memcmp(s1.attn_out, attn_last3, sizeof attn_last3) == 0,
               "check1b: batch-vs-unbatched - last token's attn_out is bit-identical");
            lz_state_free(&s1);
            free(m1.layers);
        }
    }
}

/* Items 2, 3a, 5: the 3-layer transitive chain, real (non-degenerate)
   weights throughout. */
static void check_chain_soft(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    int tk, k, ci;
    int a_n[3], b_n[3];
    int a_idx[3][F_TOPK], b_idx[3][F_TOPK];
    float attn_a[F_QD], attn_b[F_QD], attn_c[F_QD];
    unsigned stx;
    const unsigned AQ = SEED0 + 100, AKV = SEED0 + 101, AIX = SEED0 + 102;
    const unsigned BQ = SEED0 + 103, BKV = SEED0 + 104, BIX = SEED0 + 105;
    const unsigned CQ = SEED0 + 106, CKV = SEED0 + 107, CIX = SEED0 + 108;
    const unsigned XBA = SEED0 + 109, XBB = SEED0 + 110, XBC = SEED0 + 111;

    build_model_chain_soft(&m, AQ, AKV, AIX, BQ, BKV, BIX, CQ, CKV, CIX);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check_chain: lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    ci = s.cache_idx[0];
    ok(s.cache_idx[1] == ci && s.cache_idx[2] == ci,
       "check_chain: all three layers share the same physical ci "
       "(cache_idx aliasing from the prior task still holds)");

    stx = XBA;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 3, NULL);
    memcpy(attn_a, s.attn_out, sizeof attn_a);
    for (tk = 0; tk < 3; tk++) {
        a_n[tk] = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
        for (k = 0; k < F_TOPK; k++)
            a_idx[tk][k] = s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k];
    }
    (void)a_idx;
    ok(a_n[0] == 0 && a_n[1] == 1 && a_n[2] == 2,
       "check_chain: layer A's own published k_eff grows 0,1,2 across the chunk");

    stx = XBB;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, 3, NULL);
    memcpy(attn_b, s.attn_out, sizeof attn_b);
    for (tk = 0; tk < 3; tk++) {
        b_n[tk] = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk];
        for (k = 0; k < F_TOPK; k++)
            b_idx[tk][k] = s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k];
    }
    /* B's own published k_eff is min(topk, n_vis), NOT n_vis itself -
       n_vis is checked separately, indirectly, by the very fact that a
       REAL (non-trivial) 2-of-3 Top-K selection happens at all: with
       topk=2 and n_vis=0 or 1, k_eff would be 0 or 1, never 2. */
    ok(b_n[0] == F_TOPK && b_n[1] == F_TOPK && b_n[2] == F_TOPK,
       "check_chain: layer B (reindex) sees enough of A's already-finalized "
       "blocks (n_vis >= topk) for every token of its own loop to publish a "
       "full, non-trivial Top-K (A's whole call already finished before "
       "B's began, so n_vis is already 3, constant across B's own tokens)");
    ok(!(memcmp(b_idx[0], b_idx[1], sizeof(int) * F_TOPK) == 0 &&
         memcmp(b_idx[1], b_idx[2], sizeof(int) * F_TOPK) == 0),
       "check_chain: layer B's own Top-K selection is not identical across "
       "all three of its own tokens");

    stx = XBC;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, 3, NULL);
    memcpy(attn_c, s.attn_out, sizeof attn_c);
    (void)attn_c;

    /* Item 5: the shared ci's published buffer must be untouched by C's
       own call (Reuse writes nothing) - each of the three tokens' worth
       of B's own distinct selection, captured above before C ever ran,
       must survive intact. */
    {
        int all_match = 1;
        for (tk = 0; tk < 3; tk++) {
            if (s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + tk] != b_n[tk])
                all_match = 0;
            for (k = 0; k < F_TOPK; k++)
                if (s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + tk) * F_TOPK + k] != b_idx[tk][k])
                    all_match = 0;
        }
        ok(all_match,
           "check_chain: after layer C's (reuse) call, all three tokens' "
           "published Top-K selections are still exactly layer B's own");
    }
    lz_state_free(&s);
    free(m.layers);

    /* Item 2: perturbing B's own indexer weights changes B's own
       attn_out; A's is untouched. */
    {
        LZModel m2;
        LZRunState s2;
        float attn_a2[F_QD], attn_b2[F_QD];

        /* +3, not an arbitrary round number: with n_vis=3 and topk=2 there
           are only 3 possible unordered pairs, and a perturbed indexer
           can legitimately land back on the SAME pair in a different
           order (attn_out is then bit-identical - summing two floats is
           commutative) without that meaning the dispatch is broken.
           Verified empirically against this exact fixture (base picks
           {0,1}, +3 picks {1,2} - a genuinely different set, not a
           reordering) rather than assumed. */
        build_model_chain_soft(&m2, AQ, AKV, AIX, BQ, BKV, BIX + 3u, CQ, CKV, CIX);
        if (lz_state_alloc(&s2, &m2, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check2: lz_state_alloc: %s\n", errbuf);
            g_fail = 1;
        } else {
            stx = XBA;
            fill_xb(&s2, 3, F_DIM, &stx);
            forward_csa2(&m2, &s2, &m2.layers[0], 0, 0, 3, NULL);
            memcpy(attn_a2, s2.attn_out, sizeof attn_a2);
            stx = XBB;
            fill_xb(&s2, 3, F_DIM, &stx);
            forward_csa2(&m2, &s2, &m2.layers[1], 1, 0, 3, NULL);
            memcpy(attn_b2, s2.attn_out, sizeof attn_b2);

            ok(memcmp(attn_a2, attn_a, sizeof attn_a) == 0,
               "check2: layer A's own attn_out is untouched by a change to "
               "layer B's (unrelated) indexer weights");
            ok(memcmp(attn_b2, attn_b, sizeof attn_b) != 0,
               "check2: perturbing layer B's own indexer weights changes "
               "layer B's own attn_out");
            lz_state_free(&s2);
        }
        free(m2.layers);
    }

    /* Item 3a: perturbing A's compression weights changes BOTH B's and
       C's attn_out (both ultimately read A's physical ring). */
    {
        LZModel m3;
        LZRunState s3;
        float attn_b3[F_QD], attn_c3[F_QD];

        build_model_chain_soft(&m3, AQ, AKV + 7919u, AIX, BQ, BKV, BIX, CQ, CKV, CIX);
        if (lz_state_alloc(&s3, &m3, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check3a: lz_state_alloc: %s\n", errbuf);
            g_fail = 1;
        } else {
            stx = XBA;
            fill_xb(&s3, 3, F_DIM, &stx);
            forward_csa2(&m3, &s3, &m3.layers[0], 0, 0, 3, NULL);
            stx = XBB;
            fill_xb(&s3, 3, F_DIM, &stx);
            forward_csa2(&m3, &s3, &m3.layers[1], 1, 0, 3, NULL);
            memcpy(attn_b3, s3.attn_out, sizeof attn_b3);
            stx = XBC;
            fill_xb(&s3, 3, F_DIM, &stx);
            forward_csa2(&m3, &s3, &m3.layers[2], 2, 0, 3, NULL);
            memcpy(attn_c3, s3.attn_out, sizeof attn_c3);

            ok(memcmp(attn_b3, attn_b, sizeof attn_b) != 0,
               "check3a: perturbing layer A's compression weights changes "
               "layer B's attn_out");
            ok(memcmp(attn_c3, attn_c, sizeof attn_c) != 0,
               "check3a: perturbing layer A's compression weights also "
               "changes layer C's attn_out");
            lz_state_free(&s3);
        }
        free(m3.layers);
    }
}

/* Item 5, the direct proof: batch-vs-unbatched bit-identity (check1b's
   own idiom, generalized from one Full-mode layer to the full 3-layer
   chain). With nt==1, tk is always 0 - a per-ci-only publish buffer and
   a per-(ci, tk) one are indistinguishable, so the unbatched, one-
   position-at-a-time run is a correct oracle regardless of this task's
   own change. If the chunked (nt=3, one call per layer) run's layer C
   disagrees with it, the persistent per-(ci, tk) addressing is the one
   thing responsible - nothing else differs between the two runs. This
   is the check that actually distinguishes a per-(ci, tk) buffer from
   a per-ci-only one; check_chain_soft's own "buffer untouched by C"
   check above is necessary but not sufficient for that (it only proves
   C never WRITES the shared slot, not that each of its own tokens
   READS the right one). */
static void check5_batch_vs_unbatched_chain(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    float attn_c_batched[F_QD];
    const unsigned AQ = SEED0 + 400, AKV = SEED0 + 401, AIX = SEED0 + 402;
    const unsigned BQ = SEED0 + 403, BKV = SEED0 + 404, BIX = SEED0 + 405;
    const unsigned CQ = SEED0 + 406, CKV = SEED0 + 407, CIX = SEED0 + 408;
    const unsigned XBA = SEED0 + 409, XBB = SEED0 + 410, XBC = SEED0 + 411;
    unsigned stx;

    build_model_chain_soft(&m, AQ, AKV, AIX, BQ, BKV, BIX, CQ, CKV, CIX);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check5: lz_state_alloc (batched): %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    stx = XBA; fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 3, NULL);
    stx = XBB; fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, 3, NULL);
    stx = XBC; fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, 3, NULL);
    memcpy(attn_c_batched, s.attn_out, sizeof attn_c_batched);
    lz_state_free(&s);
    free(m.layers);

    {
        LZModel m2;
        LZRunState s2;
        float attn_c_unbatched[F_QD];
        unsigned stxa = XBA, stxb = XBB, stxc = XBC;
        int pos;

        build_model_chain_soft(&m2, AQ, AKV, AIX, BQ, BKV, BIX, CQ, CKV, CIX);
        if (lz_state_alloc(&s2, &m2, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check5: lz_state_alloc (unbatched): %s\n", errbuf);
            g_fail = 1;
            free(m2.layers);
            return;
        }
        /* Layer-major per position - exactly forward.c's own dispatch
           order, chunk width 1 (ordinary decode). */
        for (pos = 0; pos < 3; pos++) {
            fill_xb(&s2, 1, F_DIM, &stxa);
            forward_csa2(&m2, &s2, &m2.layers[0], 0, pos, 1, NULL);
            fill_xb(&s2, 1, F_DIM, &stxb);
            forward_csa2(&m2, &s2, &m2.layers[1], 1, pos, 1, NULL);
            fill_xb(&s2, 1, F_DIM, &stxc);
            forward_csa2(&m2, &s2, &m2.layers[2], 2, pos, 1, NULL);
        }
        memcpy(attn_c_unbatched, s2.attn_out, sizeof attn_c_unbatched);
        lz_state_free(&s2);
        free(m2.layers);

        ok(memcmp(attn_c_batched, attn_c_unbatched, sizeof attn_c_batched) == 0,
           "check5: layer C's (reuse) final attn_out over a 3-token chunk "
           "processed as one nt=3 call per layer is bit-identical to the "
           "same 3 positions processed one at a time (nt=1, layer-major "
           "per position)");
    }
}

/* Item 4: the narrowed CED kv_in gate. */
static void check4_ced_gate(void) {
    float poison[3 * F_DIM];
    float attn_d1_null[F_QD], attn_d1_poison[F_QD];
    int i;
    char errbuf[256];
    unsigned stx;
    const unsigned AQ = SEED0 + 200, AKV = SEED0 + 201, AIX = SEED0 + 202;
    const unsigned BQ = SEED0 + 203, BKV = SEED0 + 204, BIX = SEED0 + 205;

    /* Obviously distinct from any lcg_f() draw (which stays within
       [-0.2, 0.2)) - if a bug read this where it should not, the
       divergence could not be missed. */
    for (i = 0; i < 3 * F_DIM; i++) poison[i] = 500.0f + (float)i;

    /* 4a: FULL mode still reads kv_in (Task 4's own behaviour, narrowed
       but not removed by this task). Two independent builds, kv_in
       NULL vs poisoned, same weights, same xb draw. */
    {
        LZModel ma;
        LZRunState sa;

        build_model_ced(&ma, AQ, AKV, AIX, BQ, BKV, BIX);
        if (lz_state_alloc(&sa, &ma, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check4a: lz_state_alloc (kv_in=NULL): %s\n", errbuf);
            g_fail = 1;
        } else {
            stx = SEED0 + 210;
            fill_xb(&sa, 3, F_DIM, &stx);
            forward_csa2(&ma, &sa, &ma.layers[0], 0, 0, 3, NULL);
            memcpy(attn_d1_null, sa.attn_out, sizeof attn_d1_null);
            lz_state_free(&sa);
        }
        free(ma.layers);
    }
    {
        LZModel mb;
        LZRunState sb;

        build_model_ced(&mb, AQ, AKV, AIX, BQ, BKV, BIX);
        if (lz_state_alloc(&sb, &mb, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check4a: lz_state_alloc (kv_in=poison): %s\n", errbuf);
            g_fail = 1;
        } else {
            stx = SEED0 + 210;
            fill_xb(&sb, 3, F_DIM, &stx);
            forward_csa2(&mb, &sb, &mb.layers[0], 0, 0, 3, poison);
            memcpy(attn_d1_poison, sb.attn_out, sizeof attn_d1_poison);
            lz_state_free(&sb);
        }
        free(mb.layers);
    }
    ok(memcmp(attn_d1_null, attn_d1_poison, sizeof attn_d1_null) != 0,
       "check4a: layer D1 (FULL, decoder-region CSA2) still reads kv_in in "
       "its own step 2");

    /* 4b: REINDEX never reads kv_in - D2's own call, given the SAME
       upstream ring state (D1 always run with kv_in=NULL to build it),
       must be byte-identical whether D2 itself is handed NULL or a
       poisoned non-NULL kv_in. One state, D2 called twice in a row:
       REINDEX writes nothing back into the ring/pending state (only
       FULL's steps 2/6 do), so replaying it is safe. */
    {
        LZModel m;
        LZRunState s;
        float attn_d2_null[F_QD], attn_d2_poison[F_QD];
        int idx_null[F_TOPK], idx_poison[F_TOPK];
        int n_d2_null, n_d2_poison, ci, k;
        unsigned stx_d2;

        build_model_ced(&m, AQ, AKV, AIX, BQ, BKV, BIX);
        if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
            printf("FAIL check4b: lz_state_alloc: %s\n", errbuf);
            g_fail = 1;
            free(m.layers);
            return;
        }
        ci = s.cache_idx[0];
        stx = SEED0 + 210;
        fill_xb(&s, 3, F_DIM, &stx);
        forward_csa2(&m, &s, &m.layers[0], 0, 0, 3, NULL);

        stx_d2 = SEED0 + 220;
        fill_xb(&s, 3, F_DIM, &stx_d2);
        forward_csa2(&m, &s, &m.layers[1], 1, 0, 3, NULL);
        memcpy(attn_d2_null, s.attn_out, sizeof attn_d2_null);
        n_d2_null = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + 2];
        for (k = 0; k < F_TOPK; k++)
            idx_null[k] = s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + 2) * F_TOPK + k];

        stx_d2 = SEED0 + 220;
        fill_xb(&s, 3, F_DIM, &stx_d2);   /* replay D2's own input exactly */
        forward_csa2(&m, &s, &m.layers[1], 1, 0, 3, poison);
        memcpy(attn_d2_poison, s.attn_out, sizeof attn_d2_poison);
        n_d2_poison = s.csa2_topk_n_pub[(size_t)ci * LZ_BATCH_MAX + 2];
        for (k = 0; k < F_TOPK; k++)
            idx_poison[k] = s.csa2_topk_idx_pub[((size_t)ci * LZ_BATCH_MAX + 2) * F_TOPK + k];

        ok(memcmp(attn_d2_null, attn_d2_poison, sizeof attn_d2_null) == 0,
           "check4b: layer D2 (reindex, decoder-region CSA2) attn_out is "
           "byte-identical whether kv_in is NULL or a poisoned non-NULL");
        ok(n_d2_null == n_d2_poison,
           "check4b: ...and so is the published k_eff");
        ok(memcmp(idx_null, idx_poison, sizeof idx_null) == 0,
           "check4b: ...and so is the published Top-K index set");

        lz_state_free(&s);
        free(m.layers);
    }
}

/* Item 3b: a REUSE layer owns none of the seven csa2_w_* tensors -
   ordered LAST in main(): the "dispatch reverted" mutation makes this
   fixture's own layer C dereference a NULL LZTensor.f and crash, and
   that crash must not swallow any earlier check's already-printed
   PASS/FAIL lines (build/csa2_mode_dispatch_gate.sh's own header
   comment has the exact reasoning). */
static void check3b_reuse_owns_nothing(void) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    unsigned stx;
    const unsigned AQ = SEED0 + 300, AKV = SEED0 + 301, AIX = SEED0 + 302;
    const unsigned BQ = SEED0 + 303, BKV = SEED0 + 304, BIX = SEED0 + 305;
    const unsigned CQ = SEED0 + 306;

    build_model_chain_strict(&m, AQ, AKV, AIX, BQ, BKV, BIX, CQ);
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL check3b: lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return;
    }
    stx = SEED0 + 310;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[0], 0, 0, 3, NULL);
    stx = SEED0 + 311;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[1], 1, 0, 3, NULL);
    stx = SEED0 + 312;
    fill_xb(&s, 3, F_DIM, &stx);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, 3, NULL);

    /* Reaching this line at all is most of the assertion: layer C
       (reuse) owns none of the seven csa2_w_* tensors in this fixture
       (all left at memset's zero - NULL data pointers). A dispatch bug
       that wrongly read even one of them would have dereferenced NULL
       a few lines up, and this process would already be dead. */
    {
        int d, finite = 1;
        for (d = 0; d < F_QD; d++)
            if (s.attn_out[d] != s.attn_out[d]) finite = 0;   /* NaN check */
        ok(finite,
           "check3b: layer C (reuse) completed its whole 3-token call "
           "without touching any of the seven csa2_w_* tensors it does "
           "not own, and its output is finite");
    }
    lz_state_free(&s);
    free(m.layers);
}

/* Code-review addendum 1: lz_csa2_pub_overflows (forward_state.c),
   guarding alloc_csa2_ring's nf * LZ_BATCH_MAX * csa2_topk element
   count against wrapping a 32-bit size_t. This project's own
   csa2_topk=512/LZ_BATCH_MAX=8/nf-in-single-digits worst case (this
   file's own header comment) is nowhere near either a 32-bit or a
   64-bit wrap; the real production dimensions (c->n_full_layers <=
   LZ_MAX_LAYERS, LZ_BATCH_MAX, and c->csa2_topk as a 32-bit int) can
   never combine to overflow a 64-bit size_t either, which is exactly
   why this development host cannot reproduce the bug end to end
   through lz_state_alloc - only the function's own arithmetic can be
   driven directly, with synthetic divisor values size_t's own maximum
   makes possible on ANY width. */
static void check_overflow_guard(void) {
    /* D = (size_t)-1 / 4 + 1 is just over one quarter of size_t's own
       range, on whatever width size_t actually is here - so 4*D wraps
       past size_t's maximum by construction (mod 2^N == 0, whatever N
       is: (size_t)-1 is 2^N - 1, D = (2^N-1)/4 + 1 = 2^(N-2) exactly
       for N even, and 4*D = 2^N wraps to 0) while 3*D does not. This is
       the tightest possible boundary, not just an obviously-safe or
       obviously-wrapping value. */
    size_t d = ((size_t)-1) / 4 + 1;
    ok(lz_csa2_pub_overflows(d, 1, 4),
       "overflow guard: 4 * (size_t's own max/4 + 1) wraps past size_t's "
       "maximum - correctly flagged");
    ok(!lz_csa2_pub_overflows(d, 1, 3),
       "overflow guard: 3 * (size_t's own max/4 + 1) stays under size_t's "
       "maximum - correctly NOT flagged (the tightest boundary, one less "
       "than the wrapping value above)");
    ok(!lz_csa2_pub_overflows(9, (size_t)LZ_BATCH_MAX, 512),
       "overflow guard: this project's own worst-case numbers (nf=9, "
       "batch_max=8, topk=512) do not trip it");
    ok(!lz_csa2_pub_overflows((size_t)LZ_MAX_LAYERS, (size_t)LZ_BATCH_MAX, 0x7fffffff),
       "overflow guard: even nf=LZ_MAX_LAYERS with the largest possible "
       "int topk does not trip it on this host's size_t width - the real "
       "production dimensions cannot reach the wrap this function guards "
       "against here, which is exactly why the D-based checks above (not "
       "this one) are what actually exercise the boundary");
}

/* Code-review addendum 2: lz_csa2_mode_source_valid (model.c), the
   predicate lz_open_bin's own header validation now applies to every
   in-range layer's csa2_mode/csa2_source pair - previously only the
   padding past n_layers was checked at all. Tested directly with a
   hand-built LZModelConfig, the same shape ced_proj_arm_probe.c's PART
   1 already uses for lz_ced_kv_target/lz_kv_share_target: replicating
   the full ~1300-byte v5 model.bin header byte for byte to exercise
   this through lz_open would be a disproportionate amount of format
   plumbing for one predicate, and the predicate IS what lz_open_bin's
   own loop calls - there is no separate "real" behavior it could
   diverge from. */
static void check_bin_validation(void) {
    LZModelConfig c;
    int i;

    /* Fixture: 4 layers [FULL, LINEAR, FULL, FULL], layer 2 REINDEX
       sourced from 0, layer 3 REUSE sourced from 2 - the same shape
       build/csa2_cache_idx_alias_probe.c's own case A uses. */
#define VALIDATE_SETUP() do { \
        memset(&c, 0, sizeof(c)); \
        c.n_layers = 4; \
        c.layer_types[0] = LZ_LT_FULL; \
        c.layer_types[1] = LZ_LT_LINEAR; \
        c.layer_types[2] = LZ_LT_FULL; \
        c.layer_types[3] = LZ_LT_FULL; \
        for (i = 0; i < LZ_MAX_LAYERS; i++) c.csa2_source[i] = -1; \
        c.csa2_on[0] = 1; c.csa2_on[2] = 1; c.csa2_on[3] = 1; \
        c.csa2_mode[2] = LZ_CSA2_REINDEX; c.csa2_source[2] = 0; \
        c.csa2_mode[3] = LZ_CSA2_REUSE;   c.csa2_source[3] = 2; \
    } while (0)

    VALIDATE_SETUP();
    ok(lz_csa2_mode_source_valid(&c, 0) && lz_csa2_mode_source_valid(&c, 1) &&
       lz_csa2_mode_source_valid(&c, 2) && lz_csa2_mode_source_valid(&c, 3),
       "bin validation: the well-formed chain fixture passes on every layer");

    /* Corruption 1: csa2_mode[3] is a garbage word, not one of the
       three legal codes - the review's own named scenario (a raw
       header field read as garbage). */
    VALIDATE_SETUP();
    c.csa2_mode[3] = 47;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: csa2_mode outside {0,1,2} is refused");

    /* Corruption 2: csa2_source[3] is the untouched FULL-mode default
       (-1) despite csa2_mode[3] naming a non-FULL mode - the review's
       own "cache_idx[-1]" scenario. */
    VALIDATE_SETUP();
    c.csa2_source[3] = -1;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: a non-FULL mode with the FULL-mode sentinel "
       "source (-1) is refused, not read as a negative index");

    /* Corruption 3: csa2_source[3] names a layer at or after itself
       (not strictly earlier). */
    VALIDATE_SETUP();
    c.csa2_source[3] = 3;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: a source naming itself is refused");
    VALIDATE_SETUP();
    c.csa2_source[2] = 3;   /* forward reference: 3 has not been set up yet either, but 2 < 3 is what matters here */
    ok(!lz_csa2_mode_source_valid(&c, 2),
       "bin validation: a source naming a LATER layer is refused");

    /* Corruption 4: csa2_source[3] names layer 1, which is LINEAR, not
       full_attention - the review's own "wrong numbering space (nl,
       not nf)" scenario, reachable on the bin path because csa2_on[]
       there is a raw array with no cross-check against layer_types of
       its own. */
    VALIDATE_SETUP();
    c.csa2_source[3] = 1;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: a source naming a LINEAR (not full_attention) "
       "layer is refused");

    /* Corruption 5: csa2_source[3] names layer 1's index reinterpreted
       as csa2_on despite never being marked so - a source must itself
       be csa2_on, not merely full_attention. */
    VALIDATE_SETUP();
    c.csa2_on[1] = 0;             /* already 0 - restated for clarity */
    c.layer_types[1] = LZ_LT_FULL; /* layer 1 is FULL now, but never csa2_on */
    c.csa2_source[3] = 1;
    ok(!lz_csa2_mode_source_valid(&c, 3),
       "bin validation: a source that is full_attention but never "
       "marked csa2_on is refused");

    /* Control: a layer that is not csa2_on at all, with csa2_mode left
       at its default (FULL, 0) - the ordinary, harmless case every
       non-CSA2 layer in a real checkpoint is in. Must NOT be refused -
       this predicate is not "must be csa2_on", it is "must be
       internally consistent". */
    VALIDATE_SETUP();
    ok(lz_csa2_mode_source_valid(&c, 1),
       "bin validation: an ordinary non-csa2_on layer (default FULL "
       "mode) is not refused");
#undef VALIDATE_SETUP
}

int main(void) {
    check1_full_only();
    check_chain_soft();
    check5_batch_vs_unbatched_chain();
    check4_ced_gate();
    check3b_reuse_owns_nothing();
    check_overflow_guard();
    check_bin_validation();

    if (g_fail) {
        printf("\ncsa2_mode_dispatch_probe: FAIL\n");
        return 1;
    }
    printf("\ncsa2_mode_dispatch_probe: PASS\n");
    return 0;
}
