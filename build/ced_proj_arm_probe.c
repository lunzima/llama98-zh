/* CED's engine-side "proj" arm (report Sec2.2 Eq(1)): ced_kv/s->ced_hs.
   Built and run by build/ced_proj_arm_gate.sh against the real
   src/model.c/forward.c/forward_csa2.c (not a transcription - see this
   project's own note on wfwd.c/wconv2.c: measuring the engine, not a
   copy of it), and against a mutant copy with lz_ced_kv_target forced
   to always return 0, to confirm the fixture is actually sensitive to
   the defect it names.

   PART 1 mirrors build/kv_share_ced_boundary_probe.c's own shape: a
   hand-built LZModelConfig and direct calls to the real
   lz_ced_kv_target/lz_kv_share_target, no model or forward pass needed.
   It covers the field's off/on/degenerate-ced_split domain and the
   non-overlap with attn_kv_share's own decoder-region exclusion
   (9722cf6) - the two predicates must never both claim the same layer,
   which is why forward.c's kv_in resolution has no arbitration to do
   between them, only an order.

   PART 2 is a real, hand-built 3-layer full-attention LZModel (no
   config.json, no safetensors - every tensor is a small deterministic
   F32 array built in this file) run through the real lz_forward, with
   forward.c's LZ_TAP macro redefined (see the gate script's -include)
   to capture s->x right after layer 0 finishes - the same "encoder
   exit hidden" quantity forward.c's own capture copies into s->ced_hs
   a moment later, at layer 1 (== ced_split)'s own entry. It checks:
     - ced_kv=off: s->ced_hs is not even allocated (NULL) - the control.
     - ced_kv=proj: s->ced_hs is byte-identical to the tapped layer-0
       output, and turning ced_kv on changes the model's own logits
       (same weights, same token, only the switch differs) - proof the
       substitution actually reaches the K/V projections and is not
       merely computed and discarded.
   Q's own input path (attn_impl's `p[0].in = s->xb`) is not exercised
   here as a separate numeric check - it is unconditional and this
   change does not touch forward_attn.c at all (0 lines), which is the
   cheaper and equally valid way to confirm it per the task's own
   framing.

   PART 3 is a refusal check against a REAL config.json - kunkun98's
   own, which already carries `"ced_kv": "proj"` (this checkpoint's
   training-side config expects the substitution this task adds) - with
   that value text-substituted to "reuse" in a scratch copy. Optional:
   SKIPPED, not failed, when the file is not found or no longer carries
   the exact substring, since it is a fact about a file this tree does
   not own, not about the code under test (LZ_JSON_GATE_CONFIG env var,
   same default path and skip posture as build/json_offset_gate.sh's
   own config.json regression).

   Seed: the project's fixed synthetic-input seed, 1145141919, feeds a
   small LCG for the deterministic (non-zero, decorrelated-enough)
   projection weights PART 2 needs - not a statistical sweep, so no
   other seed is meaningful here.

   Exit: 0 all parts pass, 1 a check failed (printed). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "forward.h"
#include "err.h"

static int g_fail = 0;

static void ok(int cond, const char *what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) g_fail = 1;
}

/* ------------------------------------------------------------ PART 1 */

static void part1_pure_predicate(void) {
    LZModelConfig c;
    int li;

    memset(&c, 0, sizeof(c));
    c.n_layers = 4;
    c.layer_types[0] = LZ_LT_FULL;
    c.layer_types[1] = LZ_LT_LINEAR;
    c.layer_types[2] = LZ_LT_LINEAR;
    c.layer_types[3] = LZ_LT_FULL;

    /* Row 1: ced_kv off (the field's own default) - never a target,
       whatever ced_split says. */
    c.ced_kv = 0;
    c.ced_split = 1;
    ok(lz_ced_kv_target(&c, 0) == 0, "ced_kv=off: layer 0 is not a target");
    ok(lz_ced_kv_target(&c, 1) == 0,
       "ced_kv=off: layer 1 (== ced_split) is not a target");
    ok(lz_ced_kv_target(&c, 3) == 0, "ced_kv=off: layer 3 is not a target");

    /* Row 2: ced_kv=proj, ced_split=1 - every layer from ced_split on,
       INCLUDING ced_split itself, is a target; the encoder layer before
       it is not. */
    c.ced_kv = 1;
    c.ced_split = 1;
    ok(lz_ced_kv_target(&c, 0) == 0,
       "ced_kv=proj: layer 0 (encoder) is not a target");
    ok(lz_ced_kv_target(&c, 1) == 1,
       "ced_kv=proj: layer 1 (== ced_split) is a target");
    ok(lz_ced_kv_target(&c, 2) == 1,
       "ced_kv=proj: layer 2 (decoder) is a target");
    ok(lz_ced_kv_target(&c, 3) == 1,
       "ced_kv=proj: layer 3 (decoder) is a target");

    /* Row 3: ced_split==0 is CED off by this project's own convention
       (lz_kv_share_target/lz_gr_span use the same test) - not a decoder
       spanning every layer. */
    c.ced_kv = 1;
    c.ced_split = 0;
    ok(lz_ced_kv_target(&c, 0) == 0,
       "ced_split==0: not a decoder boundary, layer 0 is not a target");
    ok(lz_ced_kv_target(&c, 3) == 0,
       "ced_split==0: not a decoder boundary, layer 3 is not a target");

    /* Row 4: ced_split==n_layers is likewise CED off, not a decoder
       covering every layer. Without rows 3+4, "return c->ced_kv" alone
       (ignoring ced_split entirely) would also pass row 2. */
    c.ced_kv = 1;
    c.ced_split = 4;
    ok(lz_ced_kv_target(&c, 3) == 0,
       "ced_split==n_layers: not a decoder boundary, layer 3 is not a target");

    /* Row 5: both attn_kv_share and ced_kv=proj set. lz_kv_share_target
       already refuses any decoder-region layer (the 9722cf6 fix);
       lz_ced_kv_target claims that SAME layer. The two must never agree
       on the same layer - forward.c's kv_in resolution checks CED
       first only because CED is the more specific rule, not because
       there is a real conflict to break a tie on. */
    c.ced_kv = 1;
    c.ced_split = 1;
    c.attn_kv_share = 1;
    for (li = 0; li < c.n_layers; li++) {
        int ced = lz_ced_kv_target(&c, li);
        int shr = lz_kv_share_target(&c, li);
        char buf[96];
        sprintf(buf, "layer %d: ced_kv_target=%d kv_share_target=%d, never both 1",
                li, ced, shr);
        ok(!(ced && shr), buf);
    }
    ok(lz_ced_kv_target(&c, 3) == 1 && lz_kv_share_target(&c, 3) == 0,
       "layer 3 (decoder, full_attention): CED claims it, attn_kv_share does not");
}

/* ------------------------------------------------------------ PART 2 */

/* head_dim (and therefore hidden_size/attn_kv_dim) must be a multiple
   of 32: the KV cache's Q8 quantization group size is hardcoded at 32
   and "head_dim is a multiple of 32, so quantization groups never
   cross heads" (forward_attn.c's own cache-write comment) - a smaller
   head_dim leaves the group's scale array short, every dequantized
   score and value comes back zero, and attn_out is silently the zero
   vector no matter what the projection weights hold. Measured: this
   fixture at head_dim=4 produced an all-zero "ax" tap on every layer
   before this was widened to 32. */
#define P2_DIM   64
#define P2_NH     2
#define P2_HD    32
#define P2_KVD   64   /* n_kv_heads(2) * head_dim(32), no GQA */
#define P2_QD    64   /* n_heads * head_dim */
#define P2_QGD  128   /* 2 * attn_q_dim - attn_output_gate is always on */
#define P2_INTER 64
#define P2_VOCAB  4
#define P2_ROT   32
#define P2_SEQLEN 4
#define P2_NLAYERS 3
#define P2_CED_SPLIT 1
#define P2_TOKEN  1

/* The tap target: forward.c's "out" tag at this layer index is s->x
   right after layer 0's own residual updates finish - the exact
   quantity report Sec2.2 calls h_S, captured a moment later (at layer
   P2_CED_SPLIT's own entry) into s->ced_hs. build/ced_proj_arm_gate.sh
   redefines LZ_TAP (forward.h's own "a dump tool redefines it" - the
   same mechanism .prof's dump probes use) to call this. */
#define P2_TAP_LAYER 0
static float g_tap_buf[P2_DIM];
static int   g_tap_seen = 0;

void ced_probe_tap(const char *tag, int li, const float *ptr, int n);
void ced_probe_tap(const char *tag, int li, const float *ptr, int n) {
    if (li == P2_TAP_LAYER && strcmp(tag, "out") == 0 && n == P2_DIM) {
        memcpy(g_tap_buf, ptr, (size_t)n * sizeof(float));
        g_tap_seen = 1;
    }
}

/* Small deterministic LCG, seeded from the project's fixed constant -
   not statistics, just a non-zero, decorrelated-enough fill so the
   fixture is not a degenerate all-equal-weights case. */
static unsigned g_lcg = 1145141919u;
static float p2_next(void) {
    g_lcg = g_lcg * 1103515245u + 12345u;
    return (float)((int)((g_lcg >> 8) % 2000u) - 1000) * 0.0002f; /* [-0.2, 0.2) */
}

static void p2_fill(LZTensor *t, int n) {
    int i;
    memset(t, 0, sizeof(*t));
    t->dtype = LZ_FMT_F32;
    t->n = n;
    t->gs = 0;
    t->f = (float *)malloc((size_t)n * sizeof(float));
    for (i = 0; i < n; i++) t->f[i] = p2_next();
}

static void p2_zero(LZTensor *t, int n) {
    memset(t, 0, sizeof(*t));
    t->dtype = LZ_FMT_F32;
    t->n = n;
    t->gs = 0;
    t->f = (float *)calloc((size_t)n, sizeof(float));
}

/* One full_attention, dense-FFN, use_subn layer. gate_proj/up_proj are
   zeroed, which makes silu(gate)*up == 0 exactly regardless of input,
   so down_norm(0)==0 and down_proj(0)==0 regardless of ITS weight too -
   the FFN contributes nothing to the residual and every layer's own
   output is its attention alone. That is what lets a 3-layer stack's
   behavior be attributed entirely to the K/V substitution under test,
   without needing to predict an FFN's numeric output by hand. */
static void p2_build_layer(LZLayer *L) {
    memset(L, 0, sizeof(*L));
    L->type = LZ_LT_FULL;
    p2_fill(&L->q_proj, P2_QGD * P2_DIM);
    p2_fill(&L->k_proj, P2_KVD * P2_DIM);
    p2_fill(&L->v_proj, P2_KVD * P2_DIM);
    p2_fill(&L->o_proj, P2_DIM * P2_QD);
    p2_fill(&L->q_norm, P2_HD);
    p2_fill(&L->k_norm, P2_HD);
    p2_fill(&L->attn_q_subn_norm, P2_DIM);
    p2_fill(&L->attn_k_subn_norm, P2_DIM);
    p2_fill(&L->attn_v_subn_norm, P2_DIM);
    p2_fill(&L->attn_o_subn_norm, P2_QD);
    p2_zero(&L->gate_proj, P2_INTER * P2_DIM);
    p2_zero(&L->up_proj, P2_INTER * P2_DIM);
    p2_zero(&L->down_proj, P2_DIM * P2_INTER);
    p2_zero(&L->gate_norm, P2_DIM);
    p2_zero(&L->up_norm, P2_DIM);
    p2_zero(&L->down_norm, P2_INTER);
}

static void p2_build_model(LZModel *m, int ced_kv) {
    int i;
    memset(m, 0, sizeof(*m));
    m->config.vocab_size = P2_VOCAB;
    m->config.hidden_size = P2_DIM;
    m->config.n_layers = P2_NLAYERS;
    m->config.seq_len = P2_SEQLEN;
    m->config.n_heads = P2_NH;
    m->config.n_kv_heads = P2_NH;
    m->config.head_dim = P2_HD;
    m->config.intermediate_size = P2_INTER;
    m->config.rotary_dim = P2_ROT;
    m->config.rope_theta = 10000.0f;
    m->config.rms_norm_eps = 1e-5f;
    m->config.use_subn = 1;
    m->config.attn_output_gate = 1;
    m->config.attn_q_dim = P2_QD;
    m->config.attn_qgate_dim = P2_QGD;
    m->config.attn_kv_dim = P2_KVD;
    m->config.ced_split = P2_CED_SPLIT;
    m->config.ced_kv = ced_kv;
    for (i = 0; i < P2_NLAYERS; i++) {
        m->config.layer_types[i] = LZ_LT_FULL;
        m->config.n_full_layers++;
    }

    p2_fill(&m->embed_tokens, P2_VOCAB * P2_DIM);
    p2_zero(&m->final_norm, P2_DIM);

    m->layers = (LZLayer *)calloc((size_t)P2_NLAYERS, sizeof(LZLayer));
    for (i = 0; i < P2_NLAYERS; i++) p2_build_layer(&m->layers[i]);
}

static void part2_end_to_end(void) {
    LZModel m;
    LZRunState s_on, s_off;
    char errbuf[256];
    float logits_on[P2_VOCAB], logits_off[P2_VOCAB];
    const float *lg;
    int i, differ;

    p2_build_model(&m, 1 /* ced_kv=proj */);

    if (lz_state_alloc(&s_on, &m, P2_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        ok(0, errbuf);
        printf("part2_end_to_end: FAIL - lz_state_alloc (ced_kv=proj)\n");
        return;
    }
    ok(s_on.ced_hs != NULL, "ced_kv=proj: s->ced_hs is allocated");

    g_tap_seen = 0;
    lg = lz_forward(&m, &s_on, P2_TOKEN, 0);
    ok(lg != NULL, "ced_kv=proj: lz_forward returned logits");
    if (lg) memcpy(logits_on, lg, sizeof logits_on);

    ok(g_tap_seen, "layer-0 output tap fired (LZ_TAP wiring reached forward.c)");
    if (g_tap_seen && s_on.ced_hs) {
        int nz = 0;
        for (i = 0; i < P2_DIM; i++) if (g_tap_buf[i] != 0.0f) nz = 1;
        ok(nz, "sanity: layer-0 output is not the degenerate all-zero vector");
        ok(memcmp(g_tap_buf, s_on.ced_hs, P2_DIM * sizeof(float)) == 0,
           "s->ced_hs is byte-identical to layer 0's own residual output "
           "(h_S, captured at layer ced_split's entry)");
    }

    /* Flip only the switch - same LZModel, same weights, same token.
       A fresh state because s->ced_hs's very existence is an
       allocation-time decision (forward_state.c, conditional on
       c->ced_kv). */
    lz_state_free(&s_on);
    m.config.ced_kv = 0;
    if (lz_state_alloc(&s_off, &m, P2_SEQLEN, 0, errbuf, sizeof errbuf) != 0) {
        ok(0, errbuf);
        printf("part2_end_to_end: FAIL - lz_state_alloc (ced_kv=off)\n");
        return;
    }
    ok(s_off.ced_hs == NULL,
       "ced_kv=off (the default): s->ced_hs is not even allocated - the control");

    lg = lz_forward(&m, &s_off, P2_TOKEN, 0);
    ok(lg != NULL, "ced_kv=off: lz_forward returned logits");
    if (lg) memcpy(logits_off, lg, sizeof logits_off);
    lz_state_free(&s_off);

    differ = memcmp(logits_on, logits_off, sizeof logits_on) != 0;
    ok(differ,
       "ced_kv=proj vs off (same weights, same token): final logits differ - "
       "the substitution reaches the decoder layers' K/V and is not merely "
       "computed and discarded");
}

/* ------------------------------------------------------------ PART 3 */

/* config.json's "ced_kv": "reuse" must be refused at load, naming the
   bad value (LZ_ERR_CFG_CED_KV) - "reuse" has no engine-side
   implementation (model.h's own field comment) and must not be treated
   as an unrecognized-but-harmless string. */
static void part3_refusal(const char *real_cfg_path) {
    FILE *fp;
    long sz;
    char *buf, *hit, *out;
    const char *needle = "\"ced_kv\": \"proj\"";
    const char *repl = "\"ced_kv\": \"reuse\"";
    size_t prefix_len, needle_len, repl_len, suffix_len;
    LZModelConfig c;
    char errbuf[256];
    int rc;

    if (!real_cfg_path || !real_cfg_path[0]) {
        printf("PART3 SKIP - no real config.json path given "
               "(set LZ_JSON_GATE_CONFIG or pass one as argv[1])\n");
        return;
    }
    fp = fopen(real_cfg_path, "rb");
    if (!fp) {
        printf("PART3 SKIP - %s not found\n", real_cfg_path);
        return;
    }
    fseek(fp, 0, SEEK_END);
    sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0) { fclose(fp); printf("PART3 SKIP - empty file\n"); return; }
    buf = (char *)malloc((size_t)sz + 1);
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
        fclose(fp);
        printf("PART3 SKIP - short read on %s\n", real_cfg_path);
        free(buf);
        return;
    }
    fclose(fp);
    buf[sz] = '\0';

    hit = strstr(buf, needle);
    if (!hit) {
        printf("PART3 SKIP - %s no longer carries the exact substring %s "
               "(a fact about that file, not about ced_kv's parser)\n",
               real_cfg_path, needle);
        free(buf);
        return;
    }
    needle_len = strlen(needle);
    repl_len = strlen(repl);
    prefix_len = (size_t)(hit - buf);
    suffix_len = (size_t)sz - prefix_len - needle_len;
    out = (char *)malloc(prefix_len + repl_len + suffix_len + 1);
    memcpy(out, buf, prefix_len);
    memcpy(out + prefix_len, repl, repl_len);
    memcpy(out + prefix_len + repl_len, hit + needle_len, suffix_len);
    out[prefix_len + repl_len + suffix_len] = '\0';
    free(buf);

    {
        const char *tmp_path = "ced_proj_arm_probe_reuse.tmp.json";
        fp = fopen(tmp_path, "wb");
        if (!fp) {
            printf("PART3 FAIL - could not write %s\n", tmp_path);
            g_fail = 1;
            free(out);
            return;
        }
        fwrite(out, 1, strlen(out), fp);
        fclose(fp);
        free(out);

        memset(&c, 0, sizeof(c));
        errbuf[0] = '\0';
        rc = lz_load_config(&c, tmp_path, errbuf, sizeof errbuf);
        remove(tmp_path);
    }

    ok(rc != 0, "ced_kv=\"reuse\" is refused at load");
    ok(rc != 0 && strstr(errbuf, "ced_kv") != NULL &&
       strstr(errbuf, "reuse") != NULL,
       "the refusal names ced_kv and the bad value (reuse)");
    if (rc != 0) printf("  (message: %s)\n", errbuf);
}

int main(int argc, char **argv) {
    part1_pure_predicate();
    part2_end_to_end();
    part3_refusal(argc > 1 ? argv[1] : NULL);

    if (g_fail) {
        printf("\nced_proj_arm_probe: FAIL\n");
        return 1;
    }
    printf("\nced_proj_arm_probe: PASS\n");
    return 0;
}
