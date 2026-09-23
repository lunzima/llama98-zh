/* Is alloc_csa2_ring sizing the four rings for the model, or nf times
   over? - asked of lz_state_alloc itself, on hand-built fixtures.

   csa2_ring_nf_gate.sh answers the same question on real checkpoints, and
   it SKIPs when n_full_layers < 2 with the reason that the defect is
   arithmetically absent there (nf * total == total at nf == 1). That
   skip states the premise; this probe MEASURES it, on two fixtures that
   differ in nothing but the number of full-attention layers, so the
   gate's arms can be told apart from "a machine that happens to be
   immune". The nf=1 fixture is the CONTROL: there, a build that
   multiplies by nf and one that does not must allocate the SAME number
   of bytes.

   The expected key total comes from the engine's own two size functions
   - lz_csa2_pend_base and lz_csa2_ring_base, called with
   c->n_full_layers, which is how every consumer computes its base - and
   not from a table retyped here.

   Usage: csa2_ring_nf_probe nf1|nf2
   Prints "bytes=<n> ring=<m> nf=<k>" and exits 0, or a FAIL line and 1.
   Built and run by build/csa2_ring_nf_gate.sh against the real
   forward_state.c and against a mutant with the nf factor restored. */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "model.h"
#include "forward.h"
#include "err.h"

#define F_DIM     64
#define F_NH       2
#define F_HD      32   /* a multiple of 32 - the Q8 group size */
#define F_KVD     64   /* n_kv_heads(2) * head_dim(32) */
#define F_QD      64   /* n_heads * head_dim */
#define F_QGD    128   /* 2 * attn_q_dim - attn_output_gate always on */
#define F_INTER   64
#define F_VOCAB    4
#define F_ROT     32
#define F_SEQLEN   8
#define F_IDXHD   16
#define F_CM       1
#define F_MAXL     3

/* [FULL, LINEAR] or [FULL, LINEAR, FULL]: the only difference between the
   two fixtures is how many full-attention layers the model has, which is
   the quantity the defect is proportional to. Every LZTensor stays at its
   calloc default and every linear width at 0, as csa2_cache_idx_alias_
   probe.c explains: lz_state_alloc reads no tensor and both fixtures are
   only ever allocated, never forwarded. */
static void build_model(LZModel *m, int n_full) {
    int i;
    memset(m, 0, sizeof(*m));
    m->config.vocab_size = F_VOCAB;
    m->config.hidden_size = F_DIM;
    m->config.n_layers = F_MAXL;
    m->config.seq_len = F_SEQLEN;
    m->config.n_heads = F_NH;
    m->config.n_kv_heads = F_NH;
    m->config.head_dim = F_HD;
    m->config.intermediate_size = F_INTER;
    m->config.rotary_dim = F_ROT;
    m->config.rope_theta = 10000.0f;
    m->config.rms_norm_eps = 1e-5f;
    m->config.attn_output_gate = 1;
    m->config.attn_q_dim = F_QD;
    m->config.attn_qgate_dim = F_QGD;
    m->config.attn_kv_dim = F_KVD;
    m->config.csa2_compress_m = F_CM;
    m->config.csa2_indexer_head_dim = F_IDXHD;
    m->config.n_full_layers = n_full;
    m->config.n_linear_layers = F_MAXL - n_full;

    m->config.layer_types[0] = LZ_LT_FULL;
    m->config.layer_types[1] = LZ_LT_LINEAR;
    m->config.layer_types[2] = (n_full >= 2) ? LZ_LT_FULL : LZ_LT_LINEAR;

    for (i = 0; i < LZ_MAX_LAYERS; i++) m->config.csa2_source[i] = -1;
    for (i = 0; i < F_MAXL; i++)
        if (m->config.layer_types[i] == LZ_LT_FULL) m->config.csa2_on[i] = 1;

    m->layers = (LZLayer *)calloc((size_t)F_MAXL, sizeof(LZLayer));
    if (!m->layers) return;
    for (i = 0; i < F_MAXL; i++) {
        m->layers[i].type = m->config.layer_types[i];
        if (m->config.layer_types[i] == LZ_LT_FULL) m->layers[i].csa2_on = 1;
    }
}

int main(int argc, char **argv) {
    LZModel m;
    LZRunState s;
    char errbuf[256];
    int n_full;
    size_t pdepth, nblocks, ring;
    lz_i64 bytes;

    if (argc != 2 || (strcmp(argv[1], "nf1") != 0 && strcmp(argv[1], "nf2") != 0)) {
        printf("usage: csa2_ring_nf_probe nf1|nf2\n");
        return 1;
    }
    n_full = (strcmp(argv[1], "nf1") == 0) ? 1 : 2;
    build_model(&m, n_full);
    if (!m.layers) { printf("FAIL: fixture allocation\n"); return 1; }
    memset(&s, 0, sizeof(s));

    if (lz_state_alloc(&s, &m, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL: lz_state_alloc: %s\n", errbuf);
        return 1;
    }
    /* The same two calls every consumer makes its base with, at the same
       argument - the total, not one slot's. */
    pdepth = lz_csa2_pend_base(&m.config, m.config.n_full_layers, s.ssm_ring_depth);
    nblocks = lz_csa2_ring_base(&m.config, m.config.n_full_layers, F_SEQLEN);
    ring = 2 * pdepth * F_HD * sizeof(float)
         + nblocks * F_HD * sizeof(float)
         + nblocks * F_IDXHD * sizeof(float);
    bytes = s.bytes_alloc;
    printf("bytes=%lld ring=%lld nf=%d pdepth=%lld nblocks=%lld\n",
           (long long)bytes, (long long)ring, n_full,
           (long long)pdepth, (long long)nblocks);
    lz_state_free(&s);
    free(m.layers);
    return 0;
}
