/* Per-layer csa2_compress_m: config.json's "csa2_compress_m_layers"
   object, keyed by layer index like csa2_kv_mode/csa2_kv_source
   (model.h's own csa2_compress_m comment on LZModelConfig).

   WHAT THIS PROVES, AND WHAT IT DOES NOT. The claim is that the engine
   can EXPRESS two different compression ratios on two layers at once,
   each layer running at its own. It is not a claim that either ratio is
   better - model-selection-constraints.md's own section on what this
   machine cannot measure says the quality question needs training, and
   neither shipping checkpoint has a trained CSA2 layer to compare
   against.

   The observable is s->csa2_topk_n_pub, the k_eff forward_csa2.c's step
   4 publishes per (ci, token-of-chunk): k_eff is min(csa2_topk, n_vis)
   and n_vis is (pos + 1) / cm, an ABSOLUTE function of that layer's own
   ratio and the position. With csa2_topk above every n_vis this
   fixture reaches, the published sequence IS the layer's per-token
   visibility count, i.e. a direct readout of its cm. It is read at a
   fixed (ci, tk) offset that no part of the per-layer-cm change moves,
   so the probe is the same program under both the criterion's failing
   and passing arms - the failure is a VALUE difference, not a missing
   symbol.

   Two control arms sit beside the criterion and are the reason a red
   run is attributable. check2 raises the SCALAR from 2 to 4 and
   requires both layers' published sequences to move: that is the
   fixture's own sensitivity proof - the observable tracks cm at all.
   check1 leaves the scalar at 2 and requires both layers to publish the
   cm=2 sequence: the per-layer key's absence has to mean "the scalar",
   bit for bit, which is the same claim the existing gates pin end to
   end. check3 then names layer 2 alone and requires THAT layer to move
   while layer 0 does not. A run where check3 fails while check1/check2
   pass is the missing surface; a run where check2 fails too is a broken
   instrument, and is reported as such.

   No config.json fixture on disk, no safetensors: the JSON is written
   into a temp file and parsed by the REAL lz_load_config, and the model
   the forward runs on is built from that parsed config by hand (same
   shape build/csa2_mode_dispatch_probe.c uses), because this probe's
   subject is the config surface plus the value it resolves to.

   Built and run by build/csa2_compress_m_layers_gate.sh, against the
   real src/ plus a mutant copy whose per-layer lookup is replaced by the
   scalar - see that script's own header for the substitution and the
   exact lines expected to flip.

   Exit: 0 all cases pass, 1 a case failed (printed). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "forward.h"

static int g_fail = 0;

static void ok(int cond, const char *what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    /* Flushed per line, not left to exit-time buffering: the gate script
       redirects stdout to a file, where stdio is fully buffered and a
       later crash would erase every line already printed. */
    fflush(stdout);
    if (!cond) g_fail = 1;
}

/* ---------------------------------------------------------------- fixture */

/* Sizing mirrors build/csa2_mode_dispatch_probe.c's own headings:
   head_dim a multiple of 32 (the KV cache's Q8 group size).

   The three indexer widths are lz_load_config's own DEFAULTS
   (4/32/32 - model.c), not free choices: this probe's config comes from
   the JSON parser, so a tensor built at a narrower width than the
   parser's answer is read past its end by the same forward. The
   rotary/indexer-width pairing is the one relation the parser DOES
   check (LZ_ERR_CFG_CSA2_IDX_ROTARY: rotary_dim must not exceed
   csa2_indexer_head_dim) - head_dim 32 with the default
   partial_rotary_factor 1.0 gives rotary_dim 32, which fits the default
   indexer width exactly. */
#define F_DIM     64
#define F_NH       2
#define F_HD      32
#define F_KVD     64
#define F_QD      64
#define F_QGD    128
#define F_INTER   64
#define F_VOCAB    4
#define F_SEQLEN  64
#define F_IDXH     4
#define F_IDXHD   32
#define F_IDXR    32
#define F_TOPK     8
#define F_NT       8
/* Two ratios, both of which this fixture's 8-token chunk reaches: at cm
   2 the last token sees 4 blocks, at cm 4 it sees 2, and both are below
   F_TOPK so k_eff is n_vis itself rather than the top-k clamp. */
#define F_CM_SCALAR_LO 2
#define F_CM_SCALAR_HI 4
#define F_CM_LAYER2    4

#define SEED0 1145141919u

static unsigned lcg_next(unsigned *st) {
    *st = *st * 1103515245u + 12345u;
    return *st;
}
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

static const char *TMP_PATH = "csa2_compress_m_layers_probe.tmp.json";

/* 4 layers, all full_attention, csa2_layers=[0,2] - a LATER csa2_on
   layer (2) beside an earlier one (0), so "the scalar reaches every
   layer" and "the per-layer entry reaches one of them" are different
   observations. The two csa2 layers are both FULL mode, so each owns a
   physical ci of its own: nothing here aliases them together, and an
   aliased pair could not hold two ratios at all (which is why the
   engine refuses that shape - check4's last case). */
static int load_cfg(const char *mode_obj, const char *source_obj, int scalar_m,
                    const char *layer_obj, LZModelConfig *c,
                    char *errbuf, int errlen) {
    FILE *fp = fopen(TMP_PATH, "wb");
    int rc;
    if (!fp) {
        printf("FAIL - could not write %s\n", TMP_PATH);
        exit(2);
    }
    fprintf(fp,
        "{"
        "\"vocab_size\":%d,"
        "\"hidden_size\":%d,"
        "\"num_hidden_layers\":4,"
        "\"num_attention_heads\":%d,"
        "\"num_key_value_heads\":%d,"
        "\"head_dim\":%d,"
        "\"intermediate_size\":%d,"
        "\"attn_output_gate\":true,"
        "\"max_position_embeddings\":%d,"
        "\"layer_types\":[\"full_attention\",\"full_attention\","
                         "\"full_attention\",\"full_attention\"],"
        "\"csa2_layers\":[0,2],"
        "\"csa2_compress_m\":%d,"
        "\"csa2_topk\":%d,"
        "\"csa2_kv_mode\":%s,"
        "\"csa2_kv_source\":%s,"
        "\"csa2_compress_m_layers\":%s"
        "}",
        F_VOCAB, F_DIM, F_NH, F_NH, F_HD, F_INTER, F_SEQLEN,
        scalar_m, F_TOPK, mode_obj, source_obj, layer_obj);
    fclose(fp);
    memset(c, 0, sizeof(*c));
    errbuf[0] = '\0';
    rc = lz_load_config(c, TMP_PATH, errbuf, errlen);
    remove(TMP_PATH);
    return rc;
}

static void build_layer(LZLayer *L, unsigned seed) {
    unsigned st = seed;
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL;
    L->csa2_mode = LZ_CSA2_FULL;
    fill_f(&L->q_proj, F_QGD * F_DIM, &st);
    fill_f(&L->o_proj, F_DIM * F_QD, &st);
    fill_f(&L->q_norm, F_HD, &st);
    fill_f(&L->k_norm, F_HD, &st);
    fill_f(&L->csa2_w_kv, F_HD * F_DIM, &st);
    fill_f(&L->csa2_w_gate, F_HD * F_DIM, &st);
    fill_f(&L->csa2_w_kv_idx, F_IDXHD * F_HD, &st);
    fill_f(&L->csa2_k_norm_idx, F_IDXHD, &st);
    fill_f(&L->csa2_w_dq_idx, F_IDXR * F_DIM, &st);
    fill_f(&L->csa2_w_uq_idx, (F_IDXH * F_IDXHD) * F_IDXR, &st);
    fill_f(&L->csa2_w_weights, F_IDXH * F_DIM, &st);
}

/* Run one chunk of F_NT tokens through layer 0 and then layer 2, and
   read each layer's own published k_eff per token. Returns 0 on
   success; the published arrays land in n0/n2. */
static int run_arm(const LZModelConfig *cfg, int *n0, int *n2) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    int i, tk, ci0, ci2;

    memset(&m, 0, sizeof(m));
    m.config = *cfg;
    m.layers = (LZLayer *)calloc((size_t)cfg->n_layers, sizeof(LZLayer));
    for (i = 0; i < cfg->n_layers; i++) {
        build_layer(&m.layers[i], SEED0 + (unsigned)i * 7919u);
        /* csa2_on is the allocator's own scan predicate and
           forward.c's dispatch predicate; the fixture reproduces the
           config's answer for both. */
        m.layers[i].csa2_on = cfg->csa2_on[i];
        m.layers[i].csa2_mode = cfg->csa2_mode[i];
    }
    memset(&s, 0, sizeof(s));
    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL - lz_state_alloc: %s\n", errbuf);
        g_fail = 1;
        free(m.layers);
        return 1;
    }
    {
        unsigned stx = SEED0;
        for (tk = 0; tk < F_NT; tk++)
            for (i = 0; i < F_DIM; i++)
                s.xb[(size_t)tk * F_DIM + i] = lcg_f(&stx);
    }
    forward_csa2(&m, &s, &m.layers[0], 0, 0, F_NT, NULL);
    forward_csa2(&m, &s, &m.layers[2], 2, 0, F_NT, NULL);

    ci0 = s.cache_idx[0];
    ci2 = s.cache_idx[2];
    /* The readout's own precondition: two independently owned ci slots.
       If a change ever aliased these two together, the second call would
       overwrite the first's selection and every sequence assertion below
       would compare a layer against itself. */
    if (ci0 == ci2) {
        printf("FAIL - fixture: layers 0 and 2 share one ci (%d), so the two "
               "readouts are not independent\n", ci0);
        g_fail = 1;
        lz_state_free(&s);
        free(m.layers);
        return 1;
    }
    for (tk = 0; tk < F_NT; tk++) {
        n0[tk] = s.csa2_topk_n_pub[ci0 * LZ_BATCH_MAX + tk];
        n2[tk] = s.csa2_topk_n_pub[ci2 * LZ_BATCH_MAX + tk];
    }
    lz_state_free(&s);
    free(m.layers);
    return 0;
}

static int seq_is(const int *v, const int *want, int n) {
    int i;
    for (i = 0; i < n; i++)
        if (v[i] != want[i]) return 0;
    return 1;
}

static void show(const char *tag, const int *v, int n) {
    int i;
    printf("  %s =", tag);
    for (i = 0; i < n; i++) printf(" %d", v[i]);
    printf("\n");
}

int main(void) {
    LZModelConfig c;
    char errbuf[256];
    int rc, n0[F_NT], n2[F_NT];
    /* n_vis = (pos + 1) / cm for pos 0..7, one entry per layer ratio. */
    static const int cm2_seq[F_NT] = {0, 1, 1, 2, 2, 3, 3, 4};
    static const int cm4_seq[F_NT] = {0, 0, 0, 1, 1, 1, 1, 2};

    /* check1: the scalar alone, the shape every checkpoint in this tree
       has today. No csa2_compress_m_layers key at all. */
    rc = load_cfg("{}", "{}", F_CM_SCALAR_LO, "{}", &c, errbuf, sizeof errbuf);
    ok(rc == 0, "check1: a config with no csa2_compress_m_layers loads");
    if (rc == 0) {
        /* Fixture self-check, and a load-bearing one: this probe gets its
           dims from the JSON parser while build_layer() sizes the tensors
           from the F_* macros above, and the loader is not what binds the
           two. A tensor narrower than the config says is read past its end
           by lz_matmul_w inside the same forward - measured here, once,
           as a heap overrun that surfaced as an intermittent segfault in a
           later arm and vanished under a debugger. */
        ok(c.csa2_indexer_heads == F_IDXH &&
           c.csa2_indexer_head_dim == F_IDXHD &&
           c.csa2_indexer_q_lora_rank == F_IDXR &&
           c.rotary_dim <= c.csa2_indexer_head_dim,
           "check1: fixture - the parsed indexer widths are the ones this "
           "probe's hand-built tensors are sized for");
        if (run_arm(&c, n0, n2) == 0) {
            show("cm=2 layer 0", n0, F_NT);
            show("cm=2 layer 2", n2, F_NT);
            ok(seq_is(n0, cm2_seq, F_NT),
               "check1: control - layer 0 publishes 0,1,1,2,2,3,3,4");
            ok(seq_is(n2, cm2_seq, F_NT),
               "check1: control - layer 2 publishes the same, from the scalar");
        }
    }

    /* check2: the same shape at the other ratio. This is the fixture's
       sensitivity proof - the readout moves with cm, so a failure in
       check3 cannot be "the instrument reads nothing". */
    rc = load_cfg("{}", "{}", F_CM_SCALAR_HI, "{}", &c, errbuf, sizeof errbuf);
    ok(rc == 0, "check2: a config with scalar cm=4 loads");
    if (rc == 0) {
        if (run_arm(&c, n0, n2) == 0) {
            show("cm=4 layer 0", n0, F_NT);
            show("cm=4 layer 2", n2, F_NT);
            ok(seq_is(n0, cm4_seq, F_NT),
               "check2: sensitivity control - raising the scalar to 4 moves "
               "layer 0 to 0,0,0,1,1,1,1,2");
            ok(seq_is(n2, cm4_seq, F_NT),
               "check2: sensitivity control - and moves layer 2 with it");
        }
    }

    /* check3: THE CRITERION. One scalar, one per-layer entry naming only
       layer 2, two different ratios running at once. */
    rc = load_cfg("{}", "{}", F_CM_SCALAR_LO,
                  "{\"2\":4}", &c, errbuf, sizeof errbuf);
    ok(rc == 0, "check3: a config naming one layer in "
                "csa2_compress_m_layers loads");
    if (rc == 0) {
        if (run_arm(&c, n0, n2) == 0) {
            int diff = 0, i;
            show("per-layer layer 0", n0, F_NT);
            show("per-layer layer 2", n2, F_NT);
            ok(seq_is(n0, cm2_seq, F_NT),
               "check3: layer 0 (not named) still runs at the scalar, "
               "0,1,1,2,2,3,3,4");
            ok(seq_is(n2, cm4_seq, F_NT),
               "check3: layer 2 (named 4) runs at its own ratio, "
               "0,0,0,1,1,1,1,2");
            for (i = 0; i < F_NT; i++) if (n0[i] != n2[i]) diff = 1;
            ok(diff, "check3: the two layers' published sequences differ "
                     "within one run");
        }
    }

    /* check4: the load-time refusals. Each one is a shape that cannot
       mean anything, and the alternative to refusing is the one this
       project already rejects elsewhere (a garbage value behaving like a
       legal one). The error CODE is not named here - only the config key
       the message has to point at - so this probe compiles and runs
       against a tree that has no such enumeration yet. */
    rc = load_cfg("{}", "{}", F_CM_SCALAR_LO, "{\"1\":4}",
                  &c, errbuf, sizeof errbuf);
    ok(rc != 0,
       "check4: a per-layer entry for a layer that is not csa2_on is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_compress_m_layers") != NULL,
       "check4: that refusal names csa2_compress_m_layers");

    rc = load_cfg("{}", "{}", F_CM_SCALAR_LO, "{\"4\":4}",
                  &c, errbuf, sizeof errbuf);
    ok(rc != 0,
       "check4: a per-layer entry for a layer index past n_layers is refused");

    rc = load_cfg("{}", "{}", F_CM_SCALAR_LO, "{\"2\":0}",
                  &c, errbuf, sizeof errbuf);
    ok(rc != 0, "check4: a per-layer value of 0 is refused");

    rc = load_cfg("{}", "{}", F_CM_SCALAR_LO, "{\"2\":-3}",
                  &c, errbuf, sizeof errbuf);
    ok(rc != 0, "check4: a negative per-layer value is refused");

    rc = load_cfg("{}", "{}", F_CM_SCALAR_LO, "{\"2\":99}",
                  &c, errbuf, sizeof errbuf);
    ok(rc != 0,
       "check4: a per-layer value above seq_len is refused, as the bin "
       "header's own cm bound already refuses it");

    rc = load_cfg("{}", "{}", F_CM_SCALAR_LO, "{\"2\":\"4\"}",
                  &c, errbuf, sizeof errbuf);
    ok(rc != 0, "check4: a non-numeric per-layer value is refused");

    /* An aliased layer has no ring geometry of its own: it reads its
       source's blocks. Two ratios on one shared slot is not a
       configuration the engine can run, so it is refused rather than
       silently resolved to whichever of the two happens to be read
       first. */
    rc = load_cfg("{\"2\":\"reindex\"}", "{\"2\":0}", F_CM_SCALAR_LO,
                  "{\"2\":4}", &c, errbuf, sizeof errbuf);
    ok(rc != 0,
       "check4: a per-layer value that disagrees with the ratio of the "
       "source an aliased layer shares its ring with is refused");

    /* The same aliasing shape with the two ratios AGREEING must load:
       the refusal above is about the disagreement, not about naming an
       aliased layer at all. */
    rc = load_cfg("{\"2\":\"reindex\"}", "{\"2\":0}", F_CM_SCALAR_LO,
                  "{\"2\":2}", &c, errbuf, sizeof errbuf);
    ok(rc == 0,
       "check4: the same aliased pair with an agreeing per-layer value loads");

    return g_fail;
}
