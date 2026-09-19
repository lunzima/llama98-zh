/* cache_idx aliasing for CSA2 REINDEX/REUSE-mode layers - the engine
   side of DeepSeek-V4.1 report Sec2.3.1's Full/Reindex/Reuse cross-
   layer KV reuse axis that build/csa2_modes_config_probe.c's own
   header explicitly deferred ("forward_csa2.c does not read either
   array yet - a later, separate task's cache_idx aliasing and mode
   dispatch"). This is that task's first half: cache_idx alone, not the
   dispatch (a later task still wires forward_csa2.c to read csa2_mode
   at all).

   forward_state.c's cache_idx loop numbers every full-attention layer
   with its own fresh nf++ slot today, REGARDLESS of csa2_mode - which
   is fine while nothing reads csa2_mode, but stops being fine the
   moment forward_csa2.c's own ring lookups (csa2_comp_ring et al.,
   indexed by cache_idx[layer]) are supposed to make a REINDEX/REUSE
   layer land on its SOURCE's slot rather than a slot of its own. This
   probe drives the real lz_state_alloc (src/forward_state.c) - built
   whole out of src/, not transcribed - and inspects the resulting
   s->cache_idx[] array directly.

   Fixture: 4 layers, [FULL, LINEAR, FULL, FULL]. Layers 0/2/3 are
   csa2_on; layer 2 is REINDEX off layer 0, layer 3 is REUSE off layer
   2 - a REAL two-deep transitive chain (valid per model.c's own
   cross-check: a REUSE source must be FULL or REINDEX, never REUSE
   itself - layer 2 is REINDEX, so layer 3 reusing off it is legal).
   This is the exact shape that proves "no explicit transitive walk
   needed": by the time layer 3's own loop iteration runs, layer 2's
   cache_idx entry has ALREADY been resolved to layer 0's value, so a
   single direct read is correct by construction.

   No config.json, no lz_load_config - every LZModelConfig/LZLayer
   field is set by hand, the same shape build/ced_proj_arm_probe.c's
   PART 2 uses to reach lz_state_alloc directly. Unlike that probe, no
   real weight tensors are needed: every non-FULL-layer scratch width
   this fixture touches (lin_conv_dim, lin_key_dim, lin_value_dim,
   lin_n_v_heads, lin_k_head_dim, kda_gate_rank) is left at zero, which
   makes conv_f32_build's per-channel loop iterate zero times for the
   one LINEAR layer (cnt[]==0, so its coverage check 0==lin_conv_dim(0)
   passes without ever calling lz_t_row_f32) - lz_state_alloc's own
   buffer allocations never read a weight tensor at all, only config
   widths, so a fixture with every LZTensor left at its calloc zero is
   sufficient. Only lz_state_alloc/lz_state_free are called - no
   lz_forward - since cache_idx is filled before any forward pass and
   this task's change is entirely inside that fill.

   Built and run by build/csa2_cache_idx_alias_gate.sh against the real
   src/forward_state.c, and against a mutant copy with the aliasing
   reverted to the old unconditional nf++ for every full-attention
   layer, to confirm the fixture is actually sensitive to the defect it
   names (only the aliasing assertions, case A's checks 3/4/4b, may
   turn FAIL).

   Exit: 0 all checks pass, 1 a check failed (printed). */
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

#define F_DIM     64
#define F_NH       2
#define F_HD      32   /* must be a multiple of 32 - the KV Q8 group size,
                           see build/ced_proj_arm_probe.c's own comment on
                           this exact fixture trap */
#define F_KVD     64   /* n_kv_heads(2) * head_dim(32), no GQA */
#define F_QD      64   /* n_heads * head_dim */
#define F_QGD    128   /* 2 * attn_q_dim - attn_output_gate always on */
#define F_INTER   64
#define F_VOCAB    4
#define F_ROT     32
#define F_SEQLEN   4
#define F_NLAYERS  4

/* Builds the shared 4-layer [FULL, LINEAR, FULL, FULL] fixture, with
   csa2_on/csa2_mode/csa2_source set as this case wants. alias_on
   selects whether layers 2/3 carry REINDEX/REUSE (the case under test)
   or stay at the untouched default LZ_CSA2_FULL (the control). Every
   LZTensor is left at its p2_zero-style calloc default (dtype/n/f all
   0) - see this file's header comment on why lz_state_alloc never
   needs to read one. */
static void build_model(LZModel *m, int alias_on) {
    int i;
    memset(m, 0, sizeof(*m));
    m->config.vocab_size = F_VOCAB;
    m->config.hidden_size = F_DIM;
    m->config.n_layers = F_NLAYERS;
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

    m->config.layer_types[0] = LZ_LT_FULL;
    m->config.layer_types[1] = LZ_LT_LINEAR;
    m->config.layer_types[2] = LZ_LT_FULL;
    m->config.layer_types[3] = LZ_LT_FULL;
    m->config.n_full_layers = 3;
    m->config.n_linear_layers = 1;
    /* Every lin_ width and kda_gate_rank field left at 0 - the one
       LINEAR layer's conv scratch is then zero-width by construction
       (this file's header comment). */

    for (i = 0; i < LZ_MAX_LAYERS; i++) m->config.csa2_source[i] = -1;

    m->config.csa2_on[0] = 1;
    m->config.csa2_on[2] = 1;
    m->config.csa2_on[3] = 1;

    m->layers = (LZLayer *)calloc((size_t)F_NLAYERS, sizeof(LZLayer));
    for (i = 0; i < F_NLAYERS; i++) m->layers[i].type = m->config.layer_types[i];
    m->layers[0].csa2_on = 1;
    m->layers[2].csa2_on = 1;
    m->layers[3].csa2_on = 1;

    if (alias_on) {
        m->layers[2].csa2_mode = LZ_CSA2_REINDEX;
        m->config.csa2_source[2] = 0;
        m->layers[3].csa2_mode = LZ_CSA2_REUSE;
        m->config.csa2_source[3] = 2;
    }
    /* alias_on == 0: csa2_mode stays at every layer's calloc default,
       LZ_CSA2_FULL (0) - csa2_on layers with no aliasing at all, the
       control case. csa2_source stays -1 for both, matching model.h's
       own "csa2_source[i] is -1 where csa2_mode[i] is FULL" invariant. */
}

int main(void) {
    LZModel m_a, m_b;
    LZRunState s_a, s_b;
    char errbuf[256];

    /* ---- Case A: aliasing on (REINDEX at 2, REUSE at 3, a 2-deep
       transitive chain 3 -> 2 -> 0) ---- */
    build_model(&m_a, 1);
    if (lz_state_alloc(&s_a, &m_a, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL case A: lz_state_alloc: %s\n", errbuf);
        return 1;
    }
    ok(s_a.cache_idx != NULL, "case A: cache_idx is allocated");
    if (s_a.cache_idx) {
        printf("  case A cache_idx = [%d, %d, %d, %d]\n",
               s_a.cache_idx[0], s_a.cache_idx[1], s_a.cache_idx[2], s_a.cache_idx[3]);
        ok(1, "case A: layer 0's own cache_idx exists (any value - it is the "
              "first full-attention layer, nothing to alias against)");
        ok(s_a.cache_idx[2] == s_a.cache_idx[0],
           "case A: layer 2 (reindex, source 0) aliases layer 0's cache_idx");
        ok(s_a.cache_idx[3] == s_a.cache_idx[2],
           "case A: layer 3 (reuse, source 2) aliases layer 2's cache_idx");
        ok(s_a.cache_idx[3] == s_a.cache_idx[0],
           "case A: transitively, layer 3 lands on layer 0's SAME physical "
           "slot (2-deep chain resolves without an explicit walk)");
        ok(s_a.cache_idx[1] == 0,
           "case A: layer 1 (LINEAR) gets its own independent nl-numbering, "
           "untouched by any full-attention aliasing");
    }
    lz_state_free(&s_a);

    /* ---- Case B: control - same shape, csa2_mode left at the default
       (every csa2_on layer is FULL) - no aliasing at all ---- */
    build_model(&m_b, 0);
    if (lz_state_alloc(&s_b, &m_b, F_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        printf("FAIL case B: lz_state_alloc: %s\n", errbuf);
        return 1;
    }
    if (s_b.cache_idx) {
        printf("  case B cache_idx = [%d, %d, %d, %d]\n",
               s_b.cache_idx[0], s_b.cache_idx[1], s_b.cache_idx[2], s_b.cache_idx[3]);
        ok(s_b.cache_idx[0] != s_b.cache_idx[2] &&
           s_b.cache_idx[2] != s_b.cache_idx[3] &&
           s_b.cache_idx[0] != s_b.cache_idx[3],
           "case B: default mode (no aliasing) - layers 0/2/3 get MUTUALLY "
           "DISTINCT cache_idx values, proving case A's aliasing is genuinely "
           "conditional on csa2_mode, not unconditional for every "
           "full-attention layer");
        ok(s_b.cache_idx[1] == 0,
           "case B: layer 1 (LINEAR) is unaffected by csa2_mode either way "
           "(same value as case A)");
    }
    lz_state_free(&s_b);

    if (g_fail) {
        printf("\ncsa2_cache_idx_alias_probe: FAIL\n");
        return 1;
    }
    printf("\ncsa2_cache_idx_alias_probe: PASS\n");
    return 0;
}
