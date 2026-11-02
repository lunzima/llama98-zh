/* Dense KV planes are numbered per APPLICATION, not per physical layer.
 *
 * A looping tower runs one physical layer in more than one pass of the
 * layer loop. The recurrent (KDA/GDN) state got its own slot per
 * application first; this probe covers the KV half. The engine's own
 * lz_state_alloc is driven here - built whole out of src/, not
 * transcribed - and the resulting mapping is inspected directly:
 *
 *   - every dense-KV application owns a plane, and no two of them share
 *     one (in particular the two applications of a repeated layer);
 *   - the plane indexes are a bijection onto 0..n_dense_kv-1, so the
 *     allocation and the addressing cannot silently disagree;
 *   - an identity schedule (exec_len == n_layers) keeps the legacy
 *     per-layer order, which is the mapping a non-looping model had
 *     before this change;
 *   - a CSA2 layer inside the loop block is still refused (its rings are
 *     cache_idx-addressed, which is per physical layer), and a CSA2
 *     layer applied once is not.
 *
 * No config.json and no weights: every width lz_state_alloc reads comes
 * from the config, and every LINEAR layer's scratch width is left at
 * zero so conv_f32_build's coverage loop iterates zero times (the same
 * fixture shape build/csa2_cache_idx_alias_probe.c documents).
 *
 * Built and run by build/loop_kv_plane_gate.sh against the current tree
 * and against mutations, each of which must turn its own check red.
 *
 * Exit: 0 all checks pass, 1 a check failed (printed). */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "model.h"
#include "forward.h"
#include "err.h"

static int g_fail = 0;

static void ok(int cond, const char *what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) g_fail = 1;
}

#define P_DIM     64
#define P_NH       2
#define P_HD      32   /* multiple of 32: the KV Q8 group size */
#define P_KVD     64   /* n_kv_heads(2) * head_dim(32) */
#define P_QD      64
#define P_QGD    128   /* attn_output_gate: 2 * attn_q_dim */
#define P_INTER   64
#define P_VOCAB    4
#define P_ROT     32
#define P_SEQLEN   8
#define P_NLAYERS  4

/* The application table model.c derives from loop_repeats /
 * loop_block_start / loop_block_size / loop_coda_size. Mirrored here on
 * purpose: the probe must build arbitrary schedules without a
 * checkpoint, and the derivation itself is gated elsewhere
 * (tests/test_loop_schedule_agrees.py). */
static void build_schedule(LZModelConfig *c, int repeats, int start,
                           int size, int coda) {
    int a = 0, i, p, k;
    if (repeats <= 1) {
        for (i = 0; i < c->n_layers; i++) {
            c->app_layer[i] = i;
            c->app_loop[i] = 0;
        }
        c->exec_len = c->n_layers;
        c->ced_split = 0;
        return;
    }
    c->ced_split = start;
    for (i = 0; i < start; i++, a++) {
        c->app_layer[a] = i;
        c->app_loop[a] = 0;
    }
    for (p = 0; p < repeats; p++)
        for (k = 0; k < size; k++, a++) {
            c->app_layer[a] = start + k;
            c->app_loop[a] = p + 1;
        }
    for (i = start + size; i < c->n_layers; i++, a++) {
        c->app_layer[a] = i;
        c->app_loop[a] = repeats + 1;
    }
    c->exec_len = a;
    (void)coda;
}

/* 4 layers, [FULL, LINEAR, FULL, LINEAR]. csa2_layer >= 0 marks that one
 * layer csa2_on (it then owns a compressed-ring slot and no dense
 * plane). */
static void build_model(LZModel *m, int repeats, int start, int size,
                        int coda, int csa2_layer) {
    int i;
    memset(m, 0, sizeof(*m));
    m->config.vocab_size = P_VOCAB;
    m->config.hidden_size = P_DIM;
    m->config.n_layers = P_NLAYERS;
    m->config.seq_len = P_SEQLEN;
    m->config.n_heads = P_NH;
    m->config.n_kv_heads = P_NH;
    m->config.head_dim = P_HD;
    m->config.intermediate_size = P_INTER;
    m->config.rotary_dim = P_ROT;
    m->config.rope_theta = 10000.0f;
    m->config.rms_norm_eps = 1e-5f;
    m->config.attn_output_gate = 1;
    m->config.attn_q_dim = P_QD;
    m->config.attn_qgate_dim = P_QGD;
    m->config.attn_kv_dim = P_KVD;
    m->config.csa2_compress_m = 2;

    m->config.layer_types[0] = LZ_LT_FULL;
    m->config.layer_types[1] = LZ_LT_LINEAR;
    m->config.layer_types[2] = LZ_LT_FULL;
    m->config.layer_types[3] = LZ_LT_LINEAR;
    m->config.n_full_layers = 2;
    m->config.n_linear_layers = 2;

    for (i = 0; i < LZ_MAX_LAYERS; i++) m->config.csa2_source[i] = -1;
    if (csa2_layer >= 0) m->config.csa2_on[csa2_layer] = 1;

    m->layers = (LZLayer *)calloc((size_t)P_NLAYERS, sizeof(LZLayer));
    for (i = 0; i < P_NLAYERS; i++) {
        m->layers[i].type = m->config.layer_types[i];
        m->layers[i].csa2_on = (i == csa2_layer);
    }

    build_schedule(&m->config, repeats, start, size, coda);
}

static LZRunState *g_s;

/* Allocates and keeps the state for the checks below. Returns 0 on
 * success; on refusal, the message is left in errbuf for the caller's
 * own check. */
static int alloc_case(LZModel *m, LZRunState *s, char *errbuf, int errlen) {
    return lz_state_alloc(s, m, P_SEQLEN, 0, errbuf, errlen);
}

/* The plane of the application that runs physical layer l in pass p. */
static int plane_of(const LZModelConfig *c, int l, int loop) {
    int a;
    for (a = 0; a < c->exec_len; a++)
        if (c->app_layer[a] == l && c->app_loop[a] == loop)
            return g_s->app_dense_idx[a];
    return -2;   /* no such application: a probe defect, never PASS */
}

/* True when the non-negative app_dense_idx entries are exactly
 * 0..n_dense_kv-1, each once. */
static int planes_are_a_bijection(const LZRunState *s, int exec_len) {
    const int n = s->n_dense_kv;
    int seen[LZ_MAX_LAYERS];
    int a, i;
    if (n < 0 || n > LZ_MAX_LAYERS) return 0;
    for (i = 0; i < n; i++) seen[i] = 0;
    for (a = 0; a < exec_len; a++) {
        int v = s->app_dense_idx[a];
        if (v < 0) continue;
        if (v >= n || seen[v]) return 0;
        seen[v] = 1;
    }
    for (i = 0; i < n; i++)
        if (!seen[i]) return 0;
    return 1;
}

int main(void) {
    LZModel m;
    LZRunState s;
    char errbuf[512];
    int rc;

    /* ---- Case A: dense KV inside the loop block, twice ---- */
    build_model(&m, 2, 2, 1, 1, -1);
    g_s = &s;
    memset(&s, 0, sizeof(s));
    errbuf[0] = 0;
    rc = alloc_case(&m, &s, errbuf, sizeof errbuf);
    printf("  case A schedule = [%d %d %d %d %d] (exec_len %d)\n",
           m.config.app_layer[0], m.config.app_layer[1], m.config.app_layer[2],
           m.config.app_layer[3], m.config.app_layer[4], m.config.exec_len);
    ok(rc == 0, "case A: a dense-KV layer applied twice is accepted");
    if (rc == 0) {
        int p1 = plane_of(&m.config, 2, 1);
        int p2 = plane_of(&m.config, 2, 2);
        int p0 = plane_of(&m.config, 0, 0);
        printf("  case A planes: layer0 pass1=%d, layer2 pass1=%d, "
               "layer2 pass2=%d (n_dense_kv %d)\n", p0, p1, p2,
               s.n_dense_kv);
        ok(s.n_dense_kv == 3,
           "case A: three dense planes are allocated (layer 0 once, "
           "layer 2 twice)");
        ok(planes_are_a_bijection(&s, m.config.exec_len),
           "case A: the application planes are a bijection onto "
           "0..n_dense_kv-1 (allocation and addressing agree)");
        ok(p1 >= 0 && p2 >= 0 && p1 != p2,
           "case A: the two applications of layer 2 land in DIFFERENT "
           "planes (a shared slot would make them equal)");
        ok(p1 != p0 && p2 != p0,
           "case A: neither pass of layer 2 reuses the prelude layer's "
           "plane");
        ok(s.dense_slot[1] == -1 && s.dense_slot[3] == -1,
           "case A: a LINEAR layer's current plane starts at -1 (it "
           "owns no dense plane)");
        lz_state_free(&s);
    }

    /* ---- Case A0: the identity schedule keeps the legacy order ---- */
    build_model(&m, 1, 0, 0, 0, -1);
    g_s = &s;
    memset(&s, 0, sizeof(s));
    errbuf[0] = 0;
    rc = alloc_case(&m, &s, errbuf, sizeof errbuf);
    ok(rc == 0, "case A0: a non-looping tower allocates");
    if (rc == 0) {
        /* Legacy per-layer numbering: layer 0 -> 0, layer 2 -> 1,
         * linear layers -> -1. */
        ok(s.app_dense_idx[0] == 0 && s.app_dense_idx[1] == -1 &&
           s.app_dense_idx[2] == 1 && s.app_dense_idx[3] == -1,
           "case A0: the identity schedule keeps the legacy per-layer "
           "plane order [0, -1, 1, -1]");
        ok(s.n_dense_kv == 2, "case A0: two dense planes, one per layer");
        lz_state_free(&s);
    }

    /* ---- Case A2: a third pass gets a third plane ---- */
    build_model(&m, 3, 2, 1, 1, -1);
    g_s = &s;
    memset(&s, 0, sizeof(s));
    errbuf[0] = 0;
    rc = alloc_case(&m, &s, errbuf, sizeof errbuf);
    ok(rc == 0, "case A2: a three-pass block allocates");
    if (rc == 0) {
        int p1 = plane_of(&m.config, 2, 1);
        int p2 = plane_of(&m.config, 2, 2);
        int p3 = plane_of(&m.config, 2, 3);
        ok(s.n_dense_kv == 4,
           "case A2: one plane per application (layer 0 + layer 2 x3)");
        ok(p1 != p2 && p2 != p3 && p1 != p3,
           "case A2: all three passes of layer 2 are distinct planes");
        lz_state_free(&s);
    }

    /* ---- Case A3: the state's own accounting grows by a plane per
       application. The four-layer fixture repeats the dense layer
       alone, so no recurrent slot is added and the only other buffer
       that changes is the application table itself (one int per
       application). ---- */
    {
        LZModel m1m, m2m, m3m;
        LZRunState s1, s2, s3;
        lz_i64 b1 = 0, b2 = 0, b3 = 0;
        int a1, a2, a3;
        build_model(&m1m, 1, 0, 0, 0, -1);
        build_model(&m2m, 2, 2, 1, 1, -1);
        build_model(&m3m, 3, 2, 1, 1, -1);
        memset(&s1, 0, sizeof(s1));
        memset(&s2, 0, sizeof(s2));
        memset(&s3, 0, sizeof(s3));
        errbuf[0] = 0;
        a1 = alloc_case(&m1m, &s1, errbuf, sizeof errbuf);
        a2 = alloc_case(&m2m, &s2, errbuf, sizeof errbuf);
        a3 = alloc_case(&m3m, &s3, errbuf, sizeof errbuf);
        ok(a1 == 0 && a2 == 0 && a3 == 0,
           "case A3: one, two and three passes of the dense layer all "
           "allocate");
        if (a1 == 0 && a2 == 0 && a3 == 0) {
            b1 = s1.bytes_alloc;
            b2 = s2.bytes_alloc;
            b3 = s3.bytes_alloc;
            printf("  case A3 state bytes: %ld / %ld / %ld\n",
                   (long)b1, (long)b2, (long)b3);
            ok(b2 - b1 == b3 - b2,
               "case A3: each extra pass costs the same number of bytes "
               "(one plane plus one application-table entry)");
            ok(b2 - b1 > (lz_i64)sizeof(int),
               "case A3: that cost is larger than the application-table "
               "entry alone - the plane is really allocated");
        }
        if (a1 == 0) lz_state_free(&s1);
        if (a2 == 0) lz_state_free(&s2);
        if (a3 == 0) lz_state_free(&s3);
        free(m1m.layers);
        free(m2m.layers);
        free(m3m.layers);
    }

    /* ---- Case B: a CSA2 layer inside the loop block is refused ---- */
    build_model(&m, 2, 2, 1, 1, 2);
    memset(&s, 0, sizeof(s));
    errbuf[0] = 0;
    rc = alloc_case(&m, &s, errbuf, sizeof errbuf);
    ok(rc != 0,
       "case B: a CSA2 layer applied twice is refused (its compressed "
       "rings are per physical layer)");
    ok(strstr(errbuf, "CSA2 layer 2 is applied 2 times") != NULL,
       "case B: the refusal names the CSA2 layer and its application count");
    if (rc == 0) lz_state_free(&s);

    /* ---- Case C: the same CSA2 layer applied once is accepted ---- */
    build_model(&m, 2, 3, 1, 0, 2);
    memset(&s, 0, sizeof(s));
    errbuf[0] = 0;
    rc = alloc_case(&m, &s, errbuf, sizeof errbuf);
    ok(rc == 0,
       "case C: a looping tower whose CSA2 layer runs once allocates "
       "(the refusal is about the repeated layer, not about csa2)");
    if (rc == 0) lz_state_free(&s);

    free(m.layers);
    if (g_fail) {
        printf("\nloop_kv_plane_probe: FAIL\n");
        return 1;
    }
    printf("\nloop_kv_plane_probe: PASS\n");
    return 0;
}
