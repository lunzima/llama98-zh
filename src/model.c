#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "err.h"
#include "lfn.h"
#include "model.h"

static void qerr(char *errbuf, int errlen, LZErr code, ...) {
    va_list ap;
    if (!errbuf || errlen <= 0) return;
    va_start(ap, code);
    lz_err_fmt_v(errbuf, errlen, code, ap);
    va_end(ap);
}

const char *lz_layer_type_name(int t) {
    if (t == LZ_LT_FULL) return "full_attention";
    if (t == LZ_LT_KDA)  return "linear_attention (kda)";
    return "linear_attention";
}

/* ---------------------------------------------------------------- config */

int lz_load_config(LZModelConfig *c, const char *path,
                       char *errbuf, int errlen) {
    char *buf;
    size_t sz = 0;
    LZJson j;
    const LZJsonNode *root, *tc, *lt, *rp, *e;
    int i;
    int is_kunmoe;

    memset(c, 0, sizeof(*c));
    buf = (char *)lz_read_file(path, &sz, errbuf, errlen);
    if (!buf) return 1;

    if (lz_json_parse(&j, buf, sz, errbuf, errlen) != 0) {
        free(buf);
        return 1;
    }
    free(buf);

    root = lz_json_root(&j);
    if (!root || root->type != LZ_JSON_OBJ) {
        qerr(errbuf, errlen, LZ_ERR_CFG_ROOT);
        lz_json_free(&j);
        return 1;
    }
    /* The multimodal shell puts text params under text_config; plain-text checkpoints have them at top level */
    tc = lz_json_get(&j, root, "text_config");
    if (!tc) tc = root;

    /* KunMoe (KDA + latent MoE) detection.
       Its config.json layer_types entries stay the plain
       "linear_attention"/"full_attention" strings - see that file's
       docstring, it keeps the Qwen3.5 skeleton deliberately - so
       model_type is what tells the loader a "linear_attention" layer
       means LZ_LT_KDA rather than LZ_LT_LINEAR (GDN). */
    {
        const LZJsonNode *mt = lz_json_get(&j, tc, "model_type");
        if (!mt) mt = lz_json_get(&j, root, "model_type");
        is_kunmoe = lz_json_str_eq(mt, "kunmoe");
    }

    c->vocab_size        = lz_json_get_int(&j, tc, "vocab_size", 0);
    c->hidden_size       = lz_json_get_int(&j, tc, "hidden_size", 0);
    c->n_layers          = lz_json_get_int(&j, tc, "num_hidden_layers", 0);
    c->n_heads           = lz_json_get_int(&j, tc, "num_attention_heads", 0);
    c->n_kv_heads        = lz_json_get_int(&j, tc, "num_key_value_heads", 0);
    c->head_dim          = lz_json_get_int(&j, tc, "head_dim", 0);
    c->lin_n_k_heads     = lz_json_get_int(&j, tc, "linear_num_key_heads", 0);
    c->lin_n_v_heads     = lz_json_get_int(&j, tc, "linear_num_value_heads", 0);
    c->lin_k_head_dim    = lz_json_get_int(&j, tc, "linear_key_head_dim", 0);
    c->lin_v_head_dim    = lz_json_get_int(&j, tc, "linear_value_head_dim", 0);
    c->conv_kernel       = lz_json_get_int(&j, tc, "linear_conv_kernel_dim", 0);
    c->intermediate_size = lz_json_get_int(&j, tc, "intermediate_size", 0);
    c->mtp_n_layers      = lz_json_get_int(&j, tc, "mtp_num_hidden_layers", 0);
    /* Defaults to intermediate_size (stock upstream never writes this
       key, because there the two ARE equal) - see model.h's field
       comment for why a checkpoint whose body doesn't use
       intermediate_size (kunmoe-v2) needs this to be independent. */
    c->mtp_intermediate_size =
        lz_json_get_int(&j, tc, "mtp_intermediate_size", c->intermediate_size);
    c->seq_len = lz_json_get_int(&j, tc, "max_position_embeddings", 0);
    c->rms_norm_eps      = (float)lz_json_get_num(&j, tc, "rms_norm_eps", 1e-6);
    c->full_attention_interval =
        lz_json_get_int(&j, tc, "full_attention_interval", 0);

    {
        const LZJsonNode *n = lz_json_get(&j, tc, "tie_word_embeddings");
        if (!n) n = lz_json_get(&j, root, "tie_word_embeddings");
        c->tie_word_embeddings = (n && n->type == LZ_JSON_BOOL && n->num != 0.0);
    }
    {
        const LZJsonNode *n = lz_json_get(&j, tc, "attn_output_gate");
        c->attn_output_gate = (n && n->type == LZ_JSON_BOOL && n->num != 0.0);
    }

    /* KDA gate (only meaningful when is_kunmoe; harmless zero otherwise).
       kda_gate_rank mirrors KunMoeConfig's "field or head_k_dim" default -
       0/absent means "use lin_k_head_dim", matching
       `self.gate_rank = config.kda_gate_rank or self.head_k_dim`. */
    c->kda_gate_rank = lz_json_get_int(&j, tc, "kda_gate_rank", 0);
    if (is_kunmoe && c->kda_gate_rank <= 0) c->kda_gate_rank = c->lin_k_head_dim;
    {
        /* Present-and-non-null vs absent-or-null must be distinguished:
           None (the default, no clamp) round-trips through JSON as an
           explicit `null`, not a missing key - KunMoeConfig always
           writes the key. lz_json_get_int/num both fold "missing" and
           "null" into the same default, so read the node directly. */
        const LZJsonNode *n = lz_json_get(&j, tc, "kda_gate_lower_bound");
        c->kda_has_gate_lower_bound = (n && n->type == LZ_JSON_NUM);
        c->kda_gate_lower_bound = c->kda_has_gate_lower_bound ? (float)n->num : 0.0f;
    }
    if (is_kunmoe) {
        const char *act = lz_json_get_str(&j, tc, "kda_gate_activation", "silu");
        if (strcmp(act, "silu") != 0) {
            /* The engine's gated RMSNorm (lz_rmsnorm_gated, shared with
               GDN) hardcodes silu, same as Qwen3_5RMSNormGated. A
               sigmoid-gated checkpoint (released Kimi-Linear's own
               default) would silently compute the wrong gate with no
               error - refuse instead. */
            qerr(errbuf, errlen, LZ_ERR_CFG_KDA_ACT, act);
            lz_json_free(&j);
            return 1;
        }
    }

    /* latent MoE (KunMoEGate / LatentMoE). num_experts == 0 (the default
       here, NOT KunMoeConfig's Python default of 8) means "no MoE in
       this checkpoint" - every layer keeps the classic dense FFN. A real
       kunmoe config.json always writes num_experts explicitly (HF
       configs serialize every attribute), so the 0 fallback only bites
       on a hand-written or non-kunmoe config, where "no MoE" is
       correct. */
    c->num_experts           = lz_json_get_int(&j, tc, "num_experts", 0);
    c->num_experts_per_token = lz_json_get_int(&j, tc, "num_experts_per_token", 0);
    c->num_shared_experts    = lz_json_get_int(&j, tc, "num_shared_experts", 0);
    c->moe_intermediate_size = lz_json_get_int(&j, tc, "moe_intermediate_size", 0);
    c->moe_latent_dim        = lz_json_get_int(&j, tc, "routed_expert_hidden_size", 0);
    c->first_k_dense_replace = lz_json_get_int(&j, tc, "first_k_dense_replace", 0);
    {
        const LZJsonNode *n = lz_json_get(&j, tc, "moe_renormalize");
        c->moe_renormalize = !n || (n->type == LZ_JSON_BOOL && n->num != 0.0);
    }
    c->moe_router_sigmoid =
        strcmp(lz_json_get_str(&j, tc, "moe_router_activation_func", "sigmoid"),
              "sigmoid") == 0;
    {
        const LZJsonNode *n = lz_json_get(&j, tc, "latent_moe_use_norm");
        c->moe_latent_use_norm = (n && n->type == LZ_JSON_BOOL && n->num != 0.0);
    }
    /* moe_shared_width: mirrors KunMoeConfig's derivation exactly (see
       LatentMoE.__init__) so config.json and the bin header agree on one
       resolved number instead of each re-deriving it. shared_explicit
       uses -1 as "absent/null", same reasoning as kda_gate_lower_bound
       above, because 0 is a legal explicit width. */
    {
        const LZJsonNode *n = lz_json_get(&j, tc, "shared_expert_intermediate_size");
        int shared_explicit = (n && n->type == LZ_JSON_NUM) ? (int)n->num : -1;
        if (shared_explicit >= 0) {
            c->moe_shared_width = c->num_shared_experts ? shared_explicit : 0;
        } else {
            c->moe_shared_width = c->moe_intermediate_size *
                                  (c->num_shared_experts > 0 ? c->num_shared_experts : 0);
        }
    }
    if (c->num_experts < 0 || c->num_experts > LZ_MAX_EXPERTS ||
        c->num_experts_per_token < 0 ||
        c->num_experts_per_token > c->num_experts ||
        c->first_k_dense_replace < 0) {
        qerr(errbuf, errlen, LZ_ERR_CFG_MOE_FIELDS,
             c->num_experts, c->num_experts_per_token, c->first_k_dense_replace);
        lz_json_free(&j);
        return 1;
    }

    /* rope_theta and partial_rotary_factor live under the nested
       rope_parameters; top-level same-name keys may be absent. Check
       both. */
    c->rope_theta = 10000.0f;
    c->partial_rotary_factor = 1.0f;
    rp = lz_json_get(&j, tc, "rope_parameters");
    if (rp) {
        c->rope_theta = (float)lz_json_get_num(&j, rp, "rope_theta",
                                               c->rope_theta);
        c->partial_rotary_factor =
            (float)lz_json_get_num(&j, rp, "partial_rotary_factor",
                                   c->partial_rotary_factor);
    }
    c->rope_theta = (float)lz_json_get_num(&j, tc, "rope_theta", c->rope_theta);
    c->partial_rotary_factor =
        (float)lz_json_get_num(&j, tc, "partial_rotary_factor",
                               c->partial_rotary_factor);

    /* rope_parameters.rope_scaling: {type: linear|yarn|none,
       factor, original_max_position_embeddings, beta_fast, beta_slow}.
       Defaults: no scaling. rope_scaling_type stays 0 unless the key
       exists AND names a supported type - an unsupported type name is
       refused, not silently read as "no scaling". */
    c->rope_scaling_type = 0;
    c->rope_scaling_factor = 1.0f;
    c->rope_scaling_orig_max = 0.0f;
    c->rope_scaling_beta_fast = 32.0f;
    c->rope_scaling_beta_slow = 1.0f;
    if (rp) {
        const LZJsonNode *rs = lz_json_get(&j, rp, "rope_scaling");
        if (rs) {
            const LZJsonNode *ty = lz_json_get(&j, rs, "type");
            const char *s = ty && ty->type == LZ_JSON_STR ? ty->text : NULL;            int t = 0;
            if (s && strcmp(s, "linear") == 0) t = 1;
            else if (s && strcmp(s, "yarn") == 0) t = 2;
            else if (s && strcmp(s, "none") != 0 && strcmp(s, "null") != 0) {
                qerr(errbuf, errlen, LZ_ERR_CFG_ROPE_SCALING, s ? s : "?");
                lz_json_free(&j);
                return 1;
            }
            c->rope_scaling_type = t;
            c->rope_scaling_factor =
                (float)lz_json_get_num(&j, rs, "factor",
                                       c->rope_scaling_factor);
            c->rope_scaling_orig_max =
                (float)lz_json_get_num(&j, rs,
                                       "original_max_position_embeddings",
                                       c->rope_scaling_orig_max);
            c->rope_scaling_beta_fast =
                (float)lz_json_get_num(&j, rs, "beta_fast",
                                       c->rope_scaling_beta_fast);
            c->rope_scaling_beta_slow =
                (float)lz_json_get_num(&j, rs, "beta_slow",
                                       c->rope_scaling_beta_slow);
        }
    }

    /* layer_types decides per-layer SSM vs attention; without it loading is impossible */
    lt = lz_json_get(&j, tc, "layer_types");
    if (!lt || lt->type != LZ_JSON_ARR) {
        qerr(errbuf, errlen, LZ_ERR_CFG_LAYER_TYPES);
        lz_json_free(&j);
        return 1;
    }
    if (lt->n_children != c->n_layers) {
        qerr(errbuf, errlen, LZ_ERR_CFG_LAYER_LEN,
             lt->n_children, c->n_layers);
        lz_json_free(&j);
        return 1;
    }
    if (c->n_layers <= 0 || c->n_layers > LZ_MAX_LAYERS) {
        qerr(errbuf, errlen, LZ_ERR_CFG_LAYER_RANGE,
             c->n_layers, LZ_MAX_LAYERS);
        lz_json_free(&j);
        return 1;
    }
    i = 0;
    for (e = lz_json_first(&j, lt); e; e = lz_json_next(&j, e), i++) {
        if (lz_json_str_eq(e, "full_attention")) {
            c->layer_types[i] = LZ_LT_FULL;
            c->n_full_layers++;
        } else if (lz_json_str_eq(e, "linear_attention")) {
            /* is_kunmoe decides GDN vs KDA - see the detection comment
               above. Either way this is a state-bearing "linear" layer
               for n_linear_layers/cache_idx purposes (see forward.c and
               lz_state_alloc), which is why both map into the same
               counter unchanged. */
            c->layer_types[i] = is_kunmoe ? LZ_LT_KDA : LZ_LT_LINEAR;
            c->n_linear_layers++;
        } else {
            qerr(errbuf, errlen, LZ_ERR_CFG_LAYER_TYPE, i,
                 e->text ? e->text : "(not-a-string)");
            lz_json_free(&j);
            return 1;
        }
    }
    lz_json_free(&j);

    /* derived quantities computed once, here */
    c->attn_q_dim     = c->n_heads * c->head_dim;
    c->attn_qgate_dim = c->attn_output_gate ? c->attn_q_dim * 2 : c->attn_q_dim;
    c->attn_kv_dim    = c->n_kv_heads * c->head_dim;
    c->lin_key_dim    = c->lin_n_k_heads * c->lin_k_head_dim;
    c->lin_value_dim  = c->lin_n_v_heads * c->lin_v_head_dim;
    c->lin_conv_dim   = c->lin_key_dim * 2 + c->lin_value_dim;
    c->rotary_dim     = (int)((float)c->head_dim * c->partial_rotary_factor);

    if (c->vocab_size <= 0 || c->hidden_size <= 0 || c->head_dim <= 0 ||
        c->intermediate_size <= 0) {
        qerr(errbuf, errlen, LZ_ERR_CFG_FIELDS,
             c->vocab_size, c->hidden_size, c->head_dim, c->intermediate_size);
        return 1;
    }
    if (c->n_heads <= 0 || c->n_kv_heads <= 0 ||
        c->n_heads % c->n_kv_heads != 0) {
        qerr(errbuf, errlen, LZ_ERR_CFG_HEADS,
             c->n_heads, c->n_kv_heads);
        return 1;
    }
    if (c->rotary_dim <= 0 || c->rotary_dim > c->head_dim ||
        c->rotary_dim % 2 != 0) {
        qerr(errbuf, errlen, LZ_ERR_CFG_ROTARY,
             c->rotary_dim, c->head_dim, (double)c->partial_rotary_factor);
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------ tensor spec table */

/* Expected shapes use symbolic dims so an architecture change makes
   the error point at exactly which tensor/dim */
enum {
    DK_END = 0,
    DK_HIDDEN, DK_INTER, DK_VOCAB,
    DK_ATTN_QGATE, DK_ATTN_KV, DK_ATTN_Q, DK_HEAD_DIM,
    DK_CONV, DK_VALUE, DK_NVHEADS, DK_VHEADDIM,
    DK_ONE, DK_KERNEL,
    DK_HIDDEN2,     /* 2 * hidden: mtp.fc reads [hidden, embedding] joined */
    /* KDA (LZ_LT_KDA) */
    DK_KEY,         /* lin_key_dim: kda_q_proj/kda_k_proj output width */
    DK_GATE_RANK,   /* kda_gate_rank: f_a_proj output / f_b_proj input */
    DK_NVK,         /* lin_n_v_heads * lin_k_head_dim: f_b_proj output, dt_bias */
    /* latent MoE */
    DK_NUM_EXPERTS, /* num_experts: router weight rows / bias width */
    DK_LATENT,      /* moe_latent_dim: routed_expert_down/up_proj's latent side */
    DK_SHARED_W,    /* moe_shared_width: shared expert inner width */
    DK_MOE_INTER,   /* moe_intermediate_size: per-expert w1/w2/w3 inner width */
    /* MTP head only (model_walk's separate MTP_FFN_SPECS, not
       LAYER_SPECS - see that table's own comment) */
    DK_MTP_INTER    /* mtp_intermediate_size: MTP block's own dense FFN width */
};

/* present() lets a spec depend on more than the two axes below (moe
   shared expert / latent norm are OPTIONAL even on a MoE layer, gated by
   a global config flag rather than by layer_type or ffn_kind). NULL
   means "always present given layer_type/ffn_kind already matched". */
typedef int (*LZPresentFn)(const LZModelConfig *c);

typedef struct {
    const char *suffix;
    int layer_type;          /* -1 = any attention kind; else LZ_LT_* */
    int ffn_kind;             /* -1 = any; 0 = dense FFN only; 1 = MoE FFN only */
    size_t field;            /* offsetof(LZLayer, field) */
    int dims[3];
    LZPresentFn present;
} LZLayerSpec;

#define LOFF(f) offsetof(LZLayer, f)

static int present_moe_shared(const LZModelConfig *c) { return c->moe_shared_width > 0; }
static int present_moe_latent_norm(const LZModelConfig *c) { return c->moe_latent_use_norm; }

static const LZLayerSpec LAYER_SPECS[] = {
    /* shared by every layer kind and every FFN kind */
    { "input_layernorm.weight",          -1, -1, LOFF(input_layernorm),
      { DK_HIDDEN, DK_END, DK_END }, NULL },
    { "post_attention_layernorm.weight", -1, -1, LOFF(post_attention_layernorm),
      { DK_HIDDEN, DK_END, DK_END }, NULL },

    /* dense FFN only (ffn_kind 0 - i.e. NOT a latent-MoE layer) */
    { "mlp.gate_proj.weight",            -1, 0, LOFF(gate_proj),
      { DK_INTER, DK_HIDDEN, DK_END }, NULL },
    { "mlp.up_proj.weight",              -1, 0, LOFF(up_proj),
      { DK_INTER, DK_HIDDEN, DK_END }, NULL },
    { "mlp.down_proj.weight",            -1, 0, LOFF(down_proj),
      { DK_HIDDEN, DK_INTER, DK_END }, NULL },

    /* full_attention only. q_proj is 2x wide; the second half is the gate */
    { "self_attn.q_proj.weight", LZ_LT_FULL, -1, LOFF(q_proj),
      { DK_ATTN_QGATE, DK_HIDDEN, DK_END }, NULL },
    { "self_attn.k_proj.weight", LZ_LT_FULL, -1, LOFF(k_proj),
      { DK_ATTN_KV, DK_HIDDEN, DK_END }, NULL },
    { "self_attn.v_proj.weight", LZ_LT_FULL, -1, LOFF(v_proj),
      { DK_ATTN_KV, DK_HIDDEN, DK_END }, NULL },
    { "self_attn.o_proj.weight", LZ_LT_FULL, -1, LOFF(o_proj),
      { DK_HIDDEN, DK_ATTN_Q, DK_END }, NULL },
    { "self_attn.q_norm.weight", LZ_LT_FULL, -1, LOFF(q_norm),
      { DK_HEAD_DIM, DK_END, DK_END }, NULL },
    { "self_attn.k_norm.weight", LZ_LT_FULL, -1, LOFF(k_norm),
      { DK_HEAD_DIM, DK_END, DK_END }, NULL },

    /* linear_attention / GatedDeltaNet only */
    { "linear_attn.in_proj_qkv.weight", LZ_LT_LINEAR, -1, LOFF(in_proj_qkv),
      { DK_CONV, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.in_proj_z.weight",   LZ_LT_LINEAR, -1, LOFF(in_proj_z),
      { DK_VALUE, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.in_proj_a.weight",   LZ_LT_LINEAR, -1, LOFF(in_proj_a),
      { DK_NVHEADS, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.in_proj_b.weight",   LZ_LT_LINEAR, -1, LOFF(in_proj_b),
      { DK_NVHEADS, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.conv1d.weight",      LZ_LT_LINEAR, -1, LOFF(conv1d),
      { DK_CONV, DK_ONE, DK_KERNEL }, NULL },
    { "linear_attn.A_log",              LZ_LT_LINEAR, -1, LOFF(A_log),
      { DK_NVHEADS, DK_END, DK_END }, NULL },
    { "linear_attn.dt_bias",            LZ_LT_LINEAR, -1, LOFF(dt_bias),
      { DK_NVHEADS, DK_END, DK_END }, NULL },
    { "linear_attn.norm.weight",        LZ_LT_LINEAR, -1, LOFF(ssm_norm),
      { DK_VHEADDIM, DK_END, DK_END }, NULL },
    { "linear_attn.out_proj.weight",    LZ_LT_LINEAR, -1, LOFF(out_proj),
      { DK_HIDDEN, DK_VALUE, DK_END }, NULL },

    /* LZ_LT_KDA only - KdaAttention, tensor
       for tensor. NOT the same fields as GDN above (see model.h). */
    { "linear_attn.q_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_q_proj),
      { DK_KEY, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.k_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_k_proj),
      { DK_KEY, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.v_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_v_proj),
      { DK_VALUE, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.q_conv1d.weight", LZ_LT_KDA, -1, LOFF(kda_q_conv1d),
      { DK_KEY, DK_ONE, DK_KERNEL }, NULL },
    { "linear_attn.k_conv1d.weight", LZ_LT_KDA, -1, LOFF(kda_k_conv1d),
      { DK_KEY, DK_ONE, DK_KERNEL }, NULL },
    { "linear_attn.v_conv1d.weight", LZ_LT_KDA, -1, LOFF(kda_v_conv1d),
      { DK_VALUE, DK_ONE, DK_KERNEL }, NULL },
    { "linear_attn.A_log",           LZ_LT_KDA, -1, LOFF(kda_A_log),
      { DK_NVHEADS, DK_END, DK_END }, NULL },
    { "linear_attn.f_a_proj.weight", LZ_LT_KDA, -1, LOFF(kda_f_a_proj),
      { DK_GATE_RANK, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.f_b_proj.weight", LZ_LT_KDA, -1, LOFF(kda_f_b_proj),
      { DK_NVK, DK_GATE_RANK, DK_END }, NULL },
    { "linear_attn.dt_bias",         LZ_LT_KDA, -1, LOFF(kda_dt_bias),
      { DK_NVK, DK_END, DK_END }, NULL },
    { "linear_attn.b_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_b_proj),
      { DK_NVHEADS, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.g_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_g_proj),
      { DK_VALUE, DK_HIDDEN, DK_END }, NULL },
    { "linear_attn.o_norm.weight",   LZ_LT_KDA, -1, LOFF(kda_o_norm),
      { DK_VHEADDIM, DK_END, DK_END }, NULL },
    { "linear_attn.o_proj.weight",   LZ_LT_KDA, -1, LOFF(kda_o_proj),
      { DK_HIDDEN, DK_VALUE, DK_END }, NULL },

    /* latent MoE FFN only (ffn_kind 1). Per-expert w1/w2/w3 are walked
       separately in model_walk - their count is a runtime config value,
       not something this fixed table can express. */
    { "mlp.gate.weight",                       -1, 1, LOFF(moe_gate_w),
      { DK_NUM_EXPERTS, DK_HIDDEN, DK_END }, NULL },
    { "mlp.gate.e_score_correction_bias",      -1, 1, LOFF(moe_gate_bias),
      { DK_NUM_EXPERTS, DK_END, DK_END }, NULL },
    { "mlp.routed_expert_down_proj.weight",    -1, 1, LOFF(moe_down_proj),
      { DK_LATENT, DK_HIDDEN, DK_END }, NULL },
    { "mlp.routed_expert_up_proj.weight",      -1, 1, LOFF(moe_up_proj),
      { DK_HIDDEN, DK_LATENT, DK_END }, NULL },
    { "mlp.routed_expert_norm.weight",         -1, 1, LOFF(moe_latent_norm),
      { DK_LATENT, DK_END, DK_END }, present_moe_latent_norm },
    { "mlp.shared_experts.gate_proj.weight",   -1, 1, LOFF(moe_shared_gate),
      { DK_SHARED_W, DK_HIDDEN, DK_END }, present_moe_shared },
    { "mlp.shared_experts.up_proj.weight",     -1, 1, LOFF(moe_shared_up),
      { DK_SHARED_W, DK_HIDDEN, DK_END }, present_moe_shared },
    { "mlp.shared_experts.down_proj.weight",   -1, 1, LOFF(moe_shared_down),
      { DK_HIDDEN, DK_SHARED_W, DK_END }, present_moe_shared }
};
#define N_LAYER_SPECS ((int)(sizeof(LAYER_SPECS) / sizeof(LAYER_SPECS[0])))

static long long resolve_dim(const LZModelConfig *c, int kind) {
    switch (kind) {
    case DK_HIDDEN:     return c->hidden_size;
    case DK_INTER:      return c->intermediate_size;
    case DK_VOCAB:      return c->vocab_size;
    case DK_ATTN_QGATE: return c->attn_qgate_dim;
    case DK_ATTN_KV:    return c->attn_kv_dim;
    case DK_ATTN_Q:     return c->attn_q_dim;
    case DK_HEAD_DIM:   return c->head_dim;
    case DK_CONV:       return c->lin_conv_dim;
    case DK_VALUE:      return c->lin_value_dim;
    case DK_NVHEADS:    return c->lin_n_v_heads;
    case DK_VHEADDIM:   return c->lin_v_head_dim;
    case DK_ONE:        return 1;
    case DK_KERNEL:     return c->conv_kernel;
    case DK_HIDDEN2:    return (long long)c->hidden_size * 2;
    case DK_KEY:        return c->lin_key_dim;
    case DK_GATE_RANK:  return c->kda_gate_rank;
    case DK_NVK:        return (long long)c->lin_n_v_heads * c->lin_k_head_dim;
    case DK_NUM_EXPERTS: return c->num_experts;
    case DK_LATENT:     return c->moe_latent_dim;
    case DK_SHARED_W:   return c->moe_shared_width;
    case DK_MOE_INTER:  return c->moe_intermediate_size;
    case DK_MTP_INTER:  return c->mtp_intermediate_size;
    default:            return -1;
    }
}

static int check_shape(const LZModelConfig *c, const LZStTensor *t,
                       const int *dims, char *errbuf, int errlen) {
    int nd = 0, i;
    while (nd < 3 && dims[nd] != DK_END) nd++;

    if (t->n_dims != nd) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_DIMS,
             t->name, nd, t->n_dims);
        return 1;
    }
    for (i = 0; i < nd; i++) {
        long long want = resolve_dim(c, dims[i]);
        if (t->shape[i] != want) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_SHAPE,
                 t->name, i, want, t->shape[i]);
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------ walk skeleton */

/* Call back once per expected tensor. Shared by bind and read so the
   name assembly and shape validation are not written twice - that kind
   of duplication is exactly how read/write layouts quietly diverge. */
typedef int (*LZVisit)(LZModel *m, const LZStTensor *t, LZTensor *field,
                          void *ctx, char *errbuf, int errlen);

static int model_walk(LZModel *m, LZVisit visit, void *ctx,
                    char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    char name[256];
    const LZStTensor *t;
    LZStTensor desc;            /* tensor desc for the bin path (reused; avoids stack invalidation) */
    int li, si;

    /* top level: embedding and final norm */
    snprintf(name, sizeof(name), "%sembed_tokens.weight", m->prefix);
    if (m->bin_file) {
        memset(&desc, 0, sizeof(desc));
        desc.name = name;
        desc.n_dims = 2;
        desc.shape[0] = c->vocab_size;
        desc.shape[1] = c->hidden_size;
        desc.n_elem = (long long)c->vocab_size * c->hidden_size;
        t = &desc;
    } else {
        t = lz_st_find(&m->st, name);
        if (!t) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_MISSING, name);
            return 1;
        }
    }
    {
        static const int d[3] = { DK_VOCAB, DK_HIDDEN, DK_END };
        if (check_shape(c, t, d, errbuf, errlen) != 0) return 1;
    }
    if (visit(m, t, &m->embed_tokens, ctx, errbuf, errlen) != 0) return 1;

    snprintf(name, sizeof(name), "%snorm.weight", m->prefix);
    if (m->bin_file) {
        memset(&desc, 0, sizeof(desc));
        desc.name = name;
        desc.n_dims = 1;
        desc.shape[0] = c->hidden_size;
        desc.n_elem = c->hidden_size;
        t = &desc;
    } else {
        t = lz_st_find(&m->st, name);
        if (!t) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_MISSING, name);
            return 1;
        }
    }
    {
        static const int d[3] = { DK_HIDDEN, DK_END, DK_END };
        if (check_shape(c, t, d, errbuf, errlen) != 0) return 1;
    }
    if (visit(m, t, &m->final_norm, ctx, errbuf, errlen) != 0) return 1;

    /* per layer */
    for (li = 0; li < c->n_layers; li++) {
        LZLayer *L = &m->layers[li];
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            LZTensor *field;

            if (sp->layer_type >= 0 && sp->layer_type != L->type) continue;
            if (sp->ffn_kind >= 0 && sp->ffn_kind != L->ffn_moe) continue;
            if (sp->present && !sp->present(c)) continue;

            snprintf(name, sizeof(name), "%slayers.%d.%s",
                     m->prefix, li, sp->suffix);
            if (m->bin_file) {
                long long ne = 1;
                int di;
                memset(&desc, 0, sizeof(desc));
                desc.name = name;
                for (di = 0; di < 3 && sp->dims[di] != DK_END; di++) {
                    long long d = resolve_dim(c, sp->dims[di]);
                    desc.shape[desc.n_dims++] = d;
                    ne *= d;
                }
                desc.n_elem = ne;
                t = &desc;
            } else {
                t = lz_st_find(&m->st, name);
                if (!t) {
                    qerr(errbuf, errlen, LZ_ERR_TENSOR_MISSING, name);
                    return 1;
                }
            }
            if (check_shape(c, t, sp->dims, errbuf, errlen) != 0) return 1;

            field = (LZTensor *)((char *)L + sp->field);
            if (visit(m, t, field, ctx, errbuf, errlen) != 0) return 1;
        }

        /* Per-expert triplet. Not in LAYER_SPECS above: the count
           (config.num_experts) is a runtime value, not something a
           fixed offsetof table can index. L->moe_expert_w1/w2/w3 are
           allocated to num_experts entries by setup_layers() before
           model_walk runs (see lz_open / lz_open_bin). */
        if (L->ffn_moe) {
            static const struct { const char *suf; size_t field; int dims[3]; }
            EXPERT_SPECS[] = {
                { "w1.weight", offsetof(LZLayer, moe_expert_w1), { DK_MOE_INTER, DK_LATENT, DK_END } },
                { "w2.weight", offsetof(LZLayer, moe_expert_w2), { DK_LATENT, DK_MOE_INTER, DK_END } },
                { "w3.weight", offsetof(LZLayer, moe_expert_w3), { DK_MOE_INTER, DK_LATENT, DK_END } }
            };
            int e, k;
            for (e = 0; e < c->num_experts; e++) {
                for (k = 0; k < 3; k++) {
                    LZTensor *arr, *field;
                    snprintf(name, sizeof(name), "%slayers.%d.mlp.experts.%d.%s",
                             m->prefix, li, e, EXPERT_SPECS[k].suf);
                    if (m->bin_file) {
                        long long ne = 1;
                        int di;
                        memset(&desc, 0, sizeof(desc));
                        desc.name = name;
                        for (di = 0; di < 3 && EXPERT_SPECS[k].dims[di] != DK_END; di++) {
                            long long d = resolve_dim(c, EXPERT_SPECS[k].dims[di]);
                            desc.shape[desc.n_dims++] = d;
                            ne *= d;
                        }
                        desc.n_elem = ne;
                        t = &desc;
                    } else {
                        t = lz_st_find(&m->st, name);
                        if (!t) {
                            qerr(errbuf, errlen, LZ_ERR_TENSOR_MISSING, name);
                            return 1;
                        }
                    }
                    if (check_shape(c, t, EXPERT_SPECS[k].dims, errbuf, errlen) != 0)
                        return 1;
                    arr = *(LZTensor **)((char *)L + EXPERT_SPECS[k].field);
                    field = &arr[e];
                    if (visit(m, t, field, ctx, errbuf, errlen) != 0) return 1;
                }
            }
        }
    }

    /* MTP head. Walked last so a checkpoint without one costs nothing.

       NOTE the names carry NO m->prefix: upstream puts them at the top
       level (`mtp.fc.weight`) while the body sits under
       `model.language_model.`. That asymmetry is exactly what let 15
       tensors slip past the leak check unnoticed - see lz_open. */
    if (m->mtp) {
        static const struct { const char *suffix; size_t field; int dims[3]; }
        MTP_SPECS[] = {
            { "fc.weight",                     offsetof(LZMtp, fc),
              { DK_HIDDEN, DK_HIDDEN2, DK_END } },
            { "norm.weight",                   offsetof(LZMtp, norm),
              { DK_HIDDEN, DK_END, DK_END } },
            { "pre_fc_norm_hidden.weight",     offsetof(LZMtp, pre_fc_norm_hidden),
              { DK_HIDDEN, DK_END, DK_END } },
            { "pre_fc_norm_embedding.weight",  offsetof(LZMtp, pre_fc_norm_embedding),
              { DK_HIDDEN, DK_END, DK_END } }
        };
        /* The MTP block's own dense FFN, sized by mtp_intermediate_size
           - separate from LAYER_SPECS' body dense-FFN entries (DK_INTER),
           even though the suffixes are identical. kunmoe-v2's body
           intermediate_size is an unrelated placeholder
           (first_k_dense_replace=0 means no body layer is dense), and
           reusing that field for the MTP block's width would silently
           couple two things that must be free to differ - see
           model.h's mtp_intermediate_size comment for the full
           argument. Everything else about the block (11 tensors: 2
           norms + full_attention's 6 + this FFN's 3) still matches a
           body full_attention layer's shape one for one, which is why
           only these three specs need to leave the generic reuse
           loop below. */
        static const struct { const char *suffix; size_t field; int dims[3]; }
        MTP_FFN_SPECS[] = {
            { "mlp.gate_proj.weight", LOFF(gate_proj), { DK_MTP_INTER, DK_HIDDEN, DK_END } },
            { "mlp.up_proj.weight",   LOFF(up_proj),   { DK_MTP_INTER, DK_HIDDEN, DK_END } },
            { "mlp.down_proj.weight", LOFF(down_proj), { DK_HIDDEN, DK_MTP_INTER, DK_END } }
        };
        int k;

        /* The block reuses LAYER_SPECS for everything EXCEPT the dense
           FFN (handled by MTP_FFN_SPECS above - identified by
           referencing DK_INTER, the one dim kind that table's other
           entries never use). Not a shortcut: every one of the other 8
           tensors matches a body full_attention layer in shape, which
           is what makes "reuse the same pruning path" possible on the
           training side too. */
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            LZTensor *field;
            if (sp->layer_type >= 0 && sp->layer_type != LZ_LT_FULL) continue;
            /* The MTP head's FFN is always the classic dense triplet -
               upstream ships it as an ordinary full_attention layer, and
               LZMtp has no ffn_moe flag to route MoE specs through even
               if it did. Without this filter the loop below also tries
               to load mtp.layers.0.mlp.gate.weight (the MoE router) on
               any kunmoe checkpoint, which does not exist. */
            if (sp->ffn_kind >= 0 && sp->ffn_kind != 0) continue;
            if (sp->present && !sp->present(c)) continue;
            if (sp->dims[0] == DK_INTER || sp->dims[1] == DK_INTER ||
                sp->dims[2] == DK_INTER) continue;   /* MTP_FFN_SPECS handles these */
            snprintf(name, sizeof(name), "mtp.layers.0.%s", sp->suffix);
            /* bin v5: same desc-building shape as every other section
               above - no field tags in the bin format, so the MTP
               tensors have to be described the same way the body's do. */
            if (m->bin_file) {
                long long ne = 1;
                int di;
                memset(&desc, 0, sizeof(desc));
                desc.name = name;
                for (di = 0; di < 3 && sp->dims[di] != DK_END; di++) {
                    long long d = resolve_dim(c, sp->dims[di]);
                    desc.shape[desc.n_dims++] = d;
                    ne *= d;
                }
                desc.n_elem = ne;
                t = &desc;
            } else {
                t = lz_st_find(&m->st, name);
                if (!t) {
                    qerr(errbuf, errlen, LZ_ERR_TENSOR_MISSING, name);
                    return 1;
                }
            }
            if (check_shape(c, t, sp->dims, errbuf, errlen) != 0) return 1;
            field = (LZTensor *)((char *)&m->mtp->blk + sp->field);
            if (visit(m, t, field, ctx, errbuf, errlen) != 0) return 1;
        }
        for (k = 0; k < (int)(sizeof(MTP_FFN_SPECS) / sizeof(MTP_FFN_SPECS[0])); k++) {
            LZTensor *field;
            snprintf(name, sizeof(name), "mtp.layers.0.%s", MTP_FFN_SPECS[k].suffix);
            if (m->bin_file) {
                long long ne = 1;
                int di;
                memset(&desc, 0, sizeof(desc));
                desc.name = name;
                for (di = 0; di < 3 && MTP_FFN_SPECS[k].dims[di] != DK_END; di++) {
                    long long d = resolve_dim(c, MTP_FFN_SPECS[k].dims[di]);
                    desc.shape[desc.n_dims++] = d;
                    ne *= d;
                }
                desc.n_elem = ne;
                t = &desc;
            } else {
                t = lz_st_find(&m->st, name);
                if (!t) {
                    qerr(errbuf, errlen, LZ_ERR_TENSOR_MISSING, name);
                    return 1;
                }
            }
            if (check_shape(c, t, MTP_FFN_SPECS[k].dims, errbuf, errlen) != 0) return 1;
            field = (LZTensor *)((char *)&m->mtp->blk + MTP_FFN_SPECS[k].field);
            if (visit(m, t, field, ctx, errbuf, errlen) != 0) return 1;
        }
        for (k = 0; k < (int)(sizeof(MTP_SPECS) / sizeof(MTP_SPECS[0])); k++) {
            LZTensor *field;
            snprintf(name, sizeof(name), "mtp.%s", MTP_SPECS[k].suffix);
            if (m->bin_file) {
                long long ne = 1;
                int di;
                memset(&desc, 0, sizeof(desc));
                desc.name = name;
                for (di = 0; di < 3 && MTP_SPECS[k].dims[di] != DK_END; di++) {
                    long long d = resolve_dim(c, MTP_SPECS[k].dims[di]);
                    desc.shape[desc.n_dims++] = d;
                    ne *= d;
                }
                desc.n_elem = ne;
                t = &desc;
            } else {
                t = lz_st_find(&m->st, name);
                if (!t) {
                    qerr(errbuf, errlen, LZ_ERR_TENSOR_MISSING, name);
                    return 1;
                }
            }
            if (check_shape(c, t, MTP_SPECS[k].dims, errbuf, errlen) != 0) return 1;
            field = (LZTensor *)((char *)m->mtp + MTP_SPECS[k].field);
            if (visit(m, t, field, ctx, errbuf, errlen) != 0) return 1;
        }
    }
    return 0;
}

/* --------------------------------------------------------------- binding */

typedef struct {
    unsigned char *claimed;     /* whether each tensor was claimed; for leak detection */
    long long n_params;
} LZBindCtx;

static int visit_bind(LZModel *m, const LZStTensor *t, LZTensor *field,
                      void *ctx, char *errbuf, int errlen) {
    LZBindCtx *bc = (LZBindCtx *)ctx;
    long idx = (long)(t - m->st.tensors);
    (void)field;
    (void)errbuf;
    (void)errlen;
    bc->claimed[idx] = 1;
    bc->n_params += t->n_elem;
    return 0;
}

/* The tensor-name prefix differs by shell: the multimodal wrapper uses
   model.language_model., the plain-text checkpoint model. Probe rather
   than hardcode. */
static int detect_prefix(LZModel *m, char *errbuf, int errlen) {    static const char *CANDIDATES[] = {
        "model.language_model.", "model.", "language_model.", ""
    };
    char name[256];
    size_t i;

    for (i = 0; i < sizeof(CANDIDATES) / sizeof(CANDIDATES[0]); i++) {
        snprintf(name, sizeof(name), "%sembed_tokens.weight", CANDIDATES[i]);
        if (lz_st_find(&m->st, name)) {
            snprintf(m->prefix, sizeof(m->prefix), "%s", CANDIDATES[i]);
            return 0;
        }
    }
    qerr(errbuf, errlen, LZ_ERR_PREFIX);
    return 1;
}

/* Set each layer's attention type and FFN routing, and allocate the
   per-expert tensor arrays MoE layers need. Shared by lz_open (safetensors)
   and lz_open_bin so the two paths cannot disagree about which layers are
   MoE - that decision (config.num_experts > 0 && li >= first_k_dense_replace)
   is written once here. */
static int setup_layers(LZModel *m, char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    int i;
    for (i = 0; i < c->n_layers; i++) {
        LZLayer *L = &m->layers[i];
        L->type = c->layer_types[i];
        L->ffn_moe = (c->num_experts > 0 && i >= c->first_k_dense_replace) ? 1 : 0;
        if (L->ffn_moe) {
            L->moe_expert_w1 = (LZTensor *)calloc((size_t)c->num_experts, sizeof(LZTensor));
            L->moe_expert_w2 = (LZTensor *)calloc((size_t)c->num_experts, sizeof(LZTensor));
            L->moe_expert_w3 = (LZTensor *)calloc((size_t)c->num_experts, sizeof(LZTensor));
            if (!L->moe_expert_w1 || !L->moe_expert_w2 || !L->moe_expert_w3) {
                qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
                return 1;
            }
        }
    }
    return 0;
}

/* Bind the MTP head's storage when the config declares one. Shared by
   lz_open (safetensors) and lz_open_bin so the two loaders cannot
   disagree about when m->mtp exists - same reasoning as setup_layers
   above. Refused above 1: upstream ships exactly one block and nothing
   here is written for a stack of them, so a 2 would bind the first and
   silently ignore the rest - the failure this whole area exists to
   prevent. A no-op (m->mtp stays NULL) when mtp_n_layers is 0, which is
   every model this project has produced. */
static int alloc_mtp(LZModel *m, char *errbuf, int errlen) {
    if (m->config.mtp_n_layers <= 0) return 0;
    if (m->config.mtp_n_layers > 1) {
        qerr(errbuf, errlen, LZ_ERR_MTP_LAYERS, m->config.mtp_n_layers);
        return 1;
    }
    m->mtp = (LZMtp *)calloc(1, sizeof(LZMtp));
    if (!m->mtp) {
        qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
        return 1;
    }
    m->mtp->blk.type = LZ_LT_FULL;
    return 0;
}

/* Load config and all weights in one pass from model.bin */
static int lz_open_bin(LZModel *m, const char *dir, char *errbuf, int errlen);

int lz_open(LZModel *m, const char *dir, char *errbuf, int errlen) {
    char path[512];
    LZBindCtx bc;
    int i;

    memset(m, 0, sizeof(*m));
    memset(&bc, 0, sizeof(bc));
    if (errbuf && errlen > 0) errbuf[0] = '\0';

    /* mixed-precision bin preferred: if model.bin exists, load everything
       at once. Resolved rather than fopen'd directly (src/lfn.h): this
       probe decides between the bin and safetensors loaders, so a wrong
       "no" sends the whole load down the other path. */
    if (lz_lfn_exists(dir, "model.bin"))
        return lz_open_bin(m, dir, errbuf, errlen);

    if (lz_lfn_path(dir, "config.json", path, (int)sizeof path,
                    errbuf, errlen) != 0) return 1;
    if (lz_load_config(&m->config, path, errbuf, errlen) != 0) return 1;

    if (lz_lfn_path(dir, "model.safetensors", path, (int)sizeof path,
                    errbuf, errlen) != 0) return 1;
    if (lz_st_open(&m->st, path, errbuf, errlen) != 0) return 1;

    if (detect_prefix(m, errbuf, errlen) != 0) goto fail;

    m->layers = (LZLayer *)calloc((size_t)m->config.n_layers,
                                     sizeof(LZLayer));
    if (!m->layers) {
        qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
        goto fail;
    }
    if (setup_layers(m, errbuf, errlen) != 0) goto fail;

    if (alloc_mtp(m, errbuf, errlen) != 0) goto fail;

    bc.claimed = (unsigned char *)calloc((size_t)m->st.n_tensors, 1);
    if (!bc.claimed) {
        qerr(errbuf, errlen, LZ_ERR_CLAIM_ALLOC);
        goto fail;
    }
    if (model_walk(m, visit_bind, &bc, errbuf, errlen) != 0) {
        free(bc.claimed);
        goto fail;
    }

    /* Leak check: an unclaimed tensor means a gap in architecture
       understanding - must error rather than silently ignore, since
       silent ignoring makes "a whole module not computed" look
       perfectly fine.

       The scan covers names at EVERY level, not only names under
       m->prefix. Upstream Qwen3.5's MTP head is 15 tensors named
       `mtp.*` at TOP level, while the text prefix on that checkpoint
       is `model.language_model.`, so a prefix-scoped scan would skip
       every one of them without a word. That is the same silent drop
       transformers performs via _keys_to_ignore_on_load_unexpected,
       reached by a different route.

       So the rule is inverted: EVERY unclaimed tensor is an error unless
       its prefix is on the exemption list below. An exemption has to be
       written down; a name cannot be excused by falling outside a
       guard. This project's own exports have zero tensors outside
       m->prefix, so the strict rule costs nothing today. */
    {
        static const char *EXEMPT[] = {
            /* The multimodal wrapper's vision tower. This engine is text
               only and does not pretend otherwise; the tensors exist in
               upstream checkpoints and are deliberately not computed. */
            "model.visual.",
            NULL
        };
        for (i = 0; i < m->st.n_tensors; i++) {
            const LZStTensor *t = &m->st.tensors[i];
            int k, excused = 0;
            if (bc.claimed[i]) continue;
            for (k = 0; EXEMPT[k]; k++) {
                size_t el = strlen(EXEMPT[k]);
                if (strncmp(t->name, EXEMPT[k], el) == 0) { excused = 1; break; }
            }
            if (!excused) {
                qerr(errbuf, errlen, LZ_ERR_UNCLAIMED, t->name);
                free(bc.claimed);
                goto fail;
            }
            m->n_params_skipped += t->n_elem;   /* named exemptions only */
        }
    }
    free(bc.claimed);

    m->n_params = bc.n_params;
    m->weights_loaded = 0;
    return 0;

fail:
    lz_free(m);
    return 1;
}

/* --------------------------------------------------------------- reading */

static int visit_read(LZModel *m, const LZStTensor *t, LZTensor *field,
                      void *ctx, char *errbuf, int errlen) {
    (void)ctx;

    /* Expanding to f32 needs 4×elem bytes; bf16 raw data is read into
       the second half and expanded in place, so this buffer doubles as
       the read buffer - no extra scratch memory. */
    field->dtype = 0;
    field->n = (int)t->n_elem;
    field->gs = 0;
    field->q = NULL;
    field->scale = NULL;
    field->f = (float *)malloc((size_t)t->n_elem * sizeof(float));
    if (!field->f) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC_BYTES,
             t->name, t->n_elem * (long long)sizeof(float));
        return 1;
    }
    if (lz_st_read_f32(&m->st, t, field->f, t->n_elem, errbuf, errlen) != 0) {
        free(field->f);
        field->f = NULL;
        return 1;
    }
    m->bytes_alloc += t->n_elem * (long long)sizeof(float);
    return 0;
}

/* bin path: tensors are streamed from the file in model_walk order (dtype 0=f32 / 1=Q8_0) */
static int visit_read_bin(LZModel *m, const LZStTensor *t, LZTensor *field,
                          void *ctx, char *errbuf, int errlen) {
    FILE *f = (FILE *)ctx;
    uint8_t dtype;
    uint32_t n, gs;
    (void)m;

    if (fread(&dtype, 1, 1, f) != 1 ||
        fread(&n, 4, 1, f) != 1) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
        return 1;
    }
    if ((long long)n != t->n_elem) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_ELEMS,
             t->name, t->n_elem, n);
        return 1;
    }
    field->n = (int)n;
    field->zero = NULL;
    if (dtype == LZ_FMT_F32) {
        field->dtype = LZ_FMT_F32;
        field->gs = 0;
        field->q = NULL;
        field->scale = NULL;
        field->f = (float *)malloc((size_t)n * sizeof(float));
        if (!field->f) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            return 1;
        }
        if (fread(field->f, sizeof(float), n, f) != n) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            free(field->f);
            field->f = NULL;
            return 1;
        }
        m->bytes_alloc += (long long)n * sizeof(float);
    } else if (dtype == LZ_FMT_Q16_0) {
        /* int16: n*2 bytes of data + n/gs scales. Used for in_proj_a/b -
           they produce gt, the decay rate multiplied into the SSM state
           every token; a quantization error here is not a one-shot
           perturbation but a systematic decay-rate offset. Measured:
           Q4_1 gt relative error is 20.2x the
           activation quantization noise floor, Q8_0 1.65x, Q16_0 back
           to 1.01x. */
        field->dtype = LZ_FMT_Q16_0;
        field->f = NULL;
        if (fread(&gs, 4, 1, f) != 1 || gs == 0 || (gs % 32) != 0 ||
            (n % gs) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)malloc((size_t)n * 2);
        field->scale = (float *)malloc((size_t)(n / gs) * sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (fread(field->q, 1, (size_t)n * 2, f) != (size_t)n * 2 ||
            fread(field->scale, sizeof(float), n / gs, f) != (size_t)(n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        m->bytes_alloc += (long long)n * 2 +
                          (long long)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q6_1) {
        /* two planes: n/2 bytes 4bit + n/4 bytes 2bit + n/gs scales +
           n/gs mins. Both planes live contiguously in one malloc, 4bit
           first. */
        field->dtype = LZ_FMT_Q6_1;
        field->f = NULL;
        if (fread(&gs, 4, 1, f) != 1 || gs == 0 || (gs % 32) != 0 ||
            (n % gs) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)malloc((size_t)n / 2 + (size_t)n / 4);
        field->scale = (float *)malloc((size_t)(n / gs) * sizeof(float));
        field->zero = (float *)malloc((size_t)(n / gs) * sizeof(float));
        if (!field->q || !field->scale || !field->zero) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (fread(field->q, 1, (size_t)n / 2 + (size_t)n / 4, f)
                != (size_t)n / 2 + (size_t)n / 4 ||
            fread(field->scale, sizeof(float), n / gs, f) != (size_t)(n / gs) ||
            fread(field->zero, sizeof(float), n / gs, f) != (size_t)(n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        m->bytes_alloc += (long long)n / 2 + (long long)n / 4 +
                          2 * (long long)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_T2) {
        /* ternary: n/4 bytes of 2-bit codes + n/gs scales. NO mins -
           ternary is symmetric, and `zero` staying NULL is what the
           kernels branch on to use -scale as the hoisted coefficient
           instead of a stored min (see model.h). */
        field->dtype = LZ_FMT_T2;
        field->f = NULL;
        if (fread(&gs, 4, 1, f) != 1 || gs == 0 || (gs % 32) != 0 ||
            (n % gs) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)malloc((size_t)n / 4);
        field->scale = (float *)malloc((size_t)(n / gs) * sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (fread(field->q, 1, (size_t)n / 4, f) != (size_t)n / 4 ||
            fread(field->scale, sizeof(float), n / gs, f) != (size_t)(n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        m->bytes_alloc += (long long)n / 4 +
                          (long long)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q4_1) {
        /* nibbles: n/2 bytes of data + n/gs scales + n/gs mins.
           gs must be a multiple of 32 - nibble sub-blocks are fixed at
           32 elements (see model.h). */
        field->dtype = LZ_FMT_Q4_1;
        field->f = NULL;
        if (fread(&gs, 4, 1, f) != 1 || gs == 0 || (gs % 32) != 0 ||
            (n % gs) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)malloc((size_t)n / 2);
        field->scale = (float *)malloc((size_t)(n / gs) * sizeof(float));
        field->zero = (float *)malloc((size_t)(n / gs) * sizeof(float));
        if (!field->q || !field->scale || !field->zero) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (fread(field->q, 1, n / 2, f) != n / 2 ||
            fread(field->scale, sizeof(float), n / gs, f) != n / gs ||
            fread(field->zero, sizeof(float), n / gs, f) != n / gs) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        m->bytes_alloc += (long long)n / 2 +
                          2 * (long long)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q8_0) {
        field->dtype = LZ_FMT_Q8_0;
        field->f = NULL;
        if (fread(&gs, 4, 1, f) != 1 || gs == 0 || (n % gs) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)malloc((size_t)n);
        field->scale = (float *)malloc((size_t)(n / gs) * sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            free(field->q);
            free(field->scale);
            field->q = NULL;
            field->scale = NULL;
            return 1;
        }
        if (fread(field->q, 1, n, f) != n ||
            fread(field->scale, sizeof(float), n / gs, f) != n / gs) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            free(field->q);
            free(field->scale);
            field->q = NULL;
            field->scale = NULL;
            return 1;
        }
        m->bytes_alloc += (long long)n + (long long)(n / gs) * sizeof(float);
    } else {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_DTYPE, t->name, dtype);
        return 1;
    }
    return 0;
}

/* Load config and all weights in one pass from model.bin */
static int lz_open_bin(LZModel *m, const char *dir, char *errbuf, int errlen);

static int lz_open_bin(LZModel *m, const char *dir, char *errbuf, int errlen) {
    char path[512];
    char magic[4];
    uint32_t ver;
    LZModelConfig *c = &m->config;
    int i;
    long long n_params = 0;

    /* lz_open's probe already proved this name opens, so a failure here
       is a race and keeps LZ_ERR_BIN_OPEN's wording. */
    if (lz_lfn_path(dir, "model.bin", path, (int)sizeof path,
                    errbuf, errlen) != 0) return 1;
    m->bin_file = fopen(path, "rb");
    if (!m->bin_file) {
        qerr(errbuf, errlen, LZ_ERR_BIN_OPEN, path);
        return 1;
    }

#define RD(ptr, sz) do { \
    if (fread((ptr), (sz), 1, m->bin_file) != 1) { \
        qerr(errbuf, errlen, LZ_ERR_BIN_CFG_READ); \
        goto fail; \
    } } while (0)

    RD(magic, 4);
    if (memcmp(magic, "L98Z", 4) != 0) {
        qerr(errbuf, errlen, LZ_ERR_BIN_MAGIC);
        goto fail;
    }
    RD(&ver, 4);
    /* v1 = f32/Q8_0 only; v2 onward may carry Q4_1 etc; v4 adds the KDA
       + latent MoE header fields below; v5 adds one more field (the MTP
       head's mtp_n_layers) on top of v4's tail - a checkpoint with an
       MTP head is therefore always >= v5 even when it carries no
       KDA/MoE (the v4 tail is still present, just zeroed). The exporter
       only bumps the version when a feature that needs it is actually
       used, so a legacy pure-Q8 GDN/dense file stays v1, byte for byte -
       and, symmetrically, an older reader given a newer file must be
       refused rather than silently reading past a header it does not
       know the shape of (RD would then desync into tensor data and fail
       some unrelated way downstream, which is not a diagnosis). */
    if (ver != 1 && ver != 2 && ver != 3 && ver != 4 && ver != 5 &&
        ver != 6) {
        qerr(errbuf, errlen, LZ_ERR_BIN_VERSION, ver);
        goto fail;
    }
    RD(&c->vocab_size, 4); RD(&c->hidden_size, 4);
    RD(&c->n_layers, 4);   RD(&c->seq_len, 4);
    RD(&c->n_heads, 4);    RD(&c->n_kv_heads, 4);
    RD(&c->head_dim, 4);   RD(&c->intermediate_size, 4);
    RD(&c->lin_n_k_heads, 4); RD(&c->lin_n_v_heads, 4);
    RD(&c->lin_k_head_dim, 4); RD(&c->lin_v_head_dim, 4);
    RD(&c->conv_kernel, 4);  RD(&c->n_linear_layers, 4);
    RD(&c->n_full_layers, 4); RD(&c->full_attention_interval, 4);
    RD(&c->tie_word_embeddings, 4); RD(&c->attn_output_gate, 4);
    RD(&c->rms_norm_eps, 4);
    RD(&c->rope_theta, 4); RD(&c->partial_rotary_factor, 4);
    RD(&c->attn_q_dim, 4); RD(&c->attn_qgate_dim, 4); RD(&c->attn_kv_dim, 4);
    RD(&c->lin_key_dim, 4); RD(&c->lin_value_dim, 4); RD(&c->lin_conv_dim, 4);
    RD(&c->rotary_dim, 4);
    RD(c->layer_types, LZ_MAX_LAYERS * 4);
    for (i = 0; i < LZ_MAX_LAYERS; i++) {
        if (i >= c->n_layers && c->layer_types[i] != 0) {
            qerr(errbuf, errlen, LZ_ERR_BIN_PAD);
            goto fail;
        }
    }
    /* v4 header tail: KDA gate + latent MoE. Absent (ver < 4) leaves every
       one of these at 0/0.0f from lz_open's memset(m, 0, ...) - the same
       "no KDA, no MoE" defaults lz_load_config uses for a non-kunmoe
       config.json, so the two loading paths cannot disagree. */
    if (ver >= 4) {
        RD(&c->kda_gate_rank, 4);
        RD(&c->kda_has_gate_lower_bound, 4);
        RD(&c->kda_gate_lower_bound, 4);
        RD(&c->num_experts, 4);
        RD(&c->num_experts_per_token, 4);
        RD(&c->num_shared_experts, 4);
        RD(&c->moe_intermediate_size, 4);
        RD(&c->moe_latent_dim, 4);
        RD(&c->first_k_dense_replace, 4);
        RD(&c->moe_renormalize, 4);
        RD(&c->moe_router_sigmoid, 4);
        RD(&c->moe_latent_use_norm, 4);
        RD(&c->moe_shared_width, 4);
        if (c->num_experts < 0 || c->num_experts > LZ_MAX_EXPERTS ||
            c->num_experts_per_token < 0 ||
            c->num_experts_per_token > c->num_experts ||
            c->first_k_dense_replace < 0) {
            qerr(errbuf, errlen, LZ_ERR_BIN_CFG);
            goto fail;
        }
    }
    /* v5 header tail: the MTP head's layer count and its FFN width.
       Absent (ver < 5) leaves mtp_n_layers at 0 from lz_open's
       memset(m, 0, ...) before this function is reached - "no MTP
       head", the same default the safetensors path's lz_load_config
       uses for a config.json without mtp_num_hidden_layers.
       mtp_intermediate_size has no such "absent key" fallback here -
       the bin format has no field tagging, so export_q8.py always
       writes both when it writes either (see write_config); it is only
       ever read below when mtp_n_layers > 0 too. */
    if (ver >= 5) {
        RD(&c->mtp_n_layers, 4);
        RD(&c->mtp_intermediate_size, 4);
    }
    /* v6 header tail: rope_scaling. Absent (ver < 6) leaves the five
       fields at the memset defaults - which are exactly the
       "no scaling" defaults lz_load_config uses when rope_parameters
       has no rope_scaling key (type 0, factor 1, orig_max 0,
       beta_fast 32, beta_slow 1), so the two loading paths cannot
       disagree. */
    if (ver >= 6) {
        RD(&c->rope_scaling_type, 4);
        RD(&c->rope_scaling_factor, 4);
        RD(&c->rope_scaling_orig_max, 4);
        RD(&c->rope_scaling_beta_fast, 4);
        RD(&c->rope_scaling_beta_slow, 4);
        if (c->rope_scaling_type < 0 || c->rope_scaling_type > 2 ||
            !(c->rope_scaling_factor > 0.0f) ||
            c->rope_scaling_beta_fast <= 0.0f ||
            c->rope_scaling_beta_slow <= 0.0f) {
            qerr(errbuf, errlen, LZ_ERR_BIN_CFG);
            goto fail;
        }
    }
#undef RD

    /* Defense: derived quantities and key fields self-consistent (same checks as lz_load_config) */
    if (c->vocab_size <= 0 || c->hidden_size <= 0 || c->head_dim <= 0 ||
        c->intermediate_size <= 0 || c->n_layers <= 0 ||
        c->n_layers > LZ_MAX_LAYERS ||
        c->n_heads <= 0 || c->n_kv_heads <= 0 ||
        c->n_heads % c->n_kv_heads != 0 ||
        c->rotary_dim <= 0 || c->rotary_dim > c->head_dim ||
        c->rotary_dim % 2 != 0) {
        qerr(errbuf, errlen, LZ_ERR_BIN_CFG);
        goto fail;
    }

    m->layers = (LZLayer *)calloc((size_t)c->n_layers, sizeof(LZLayer));
    if (!m->layers) {
        qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
        goto fail;
    }
    if (setup_layers(m, errbuf, errlen) != 0) goto fail;

    if (alloc_mtp(m, errbuf, errlen) != 0) goto fail;

    if (model_walk(m, visit_read_bin, m->bin_file, errbuf, errlen) != 0) {
        goto fail;
    }
    for (i = 0; i < c->n_layers; i++) {
        int si;
        const LZLayer *L = &m->layers[i];
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            if (sp->layer_type >= 0 && sp->layer_type != L->type) continue;
            n_params += ((LZTensor *)((char *)L + sp->field))->n;
        }
        if (L->ffn_moe) {
            int e;
            for (e = 0; e < c->num_experts; e++)
                n_params += L->moe_expert_w1[e].n + L->moe_expert_w2[e].n +
                            L->moe_expert_w3[e].n;
        }
    }
    n_params += m->embed_tokens.n + m->final_norm.n;
    if (m->mtp) {
        int si;
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            if (sp->layer_type >= 0 && sp->layer_type != LZ_LT_FULL) continue;
            if (sp->ffn_kind >= 0 && sp->ffn_kind != 0) continue;
            if (sp->present && !sp->present(c)) continue;
            n_params += ((LZTensor *)((char *)&m->mtp->blk + sp->field))->n;
        }
        n_params += m->mtp->fc.n + m->mtp->norm.n +
                    m->mtp->pre_fc_norm_hidden.n + m->mtp->pre_fc_norm_embedding.n;
    }
    m->n_params = n_params;
    m->prefix[0] = '\0';
    m->weights_loaded = 1;
    fclose(m->bin_file);
    m->bin_file = NULL;
    return 0;

fail:
    fclose(m->bin_file);
    m->bin_file = NULL;
    lz_free(m);
    return 1;
}


void lz_t_free(LZTensor *t) {
    if (!t) return;
    free(t->f);
    free(t->q);
    free(t->scale);
    free(t->zero);
    t->f = NULL;
    t->q = NULL;
    t->scale = NULL;
    t->zero = NULL;
}

int lz_read_weights(LZModel *m, char *errbuf, int errlen) {
    if (!m || !m->layers) {
        qerr(errbuf, errlen, LZ_ERR_NOT_OPEN);
        return 1;
    }
    if (m->weights_loaded) return 0;
    if (model_walk(m, visit_read, NULL, errbuf, errlen) != 0) return 1;
    m->weights_loaded = 1;
    return 0;
}

void lz_free(LZModel *m) {
    int li, si;
    if (!m) return;

    lz_t_free(&m->embed_tokens);
    lz_t_free(&m->final_norm);
    if (m->layers) {
        for (li = 0; li < m->config.n_layers; li++) {
            LZLayer *L = &m->layers[li];
            for (si = 0; si < N_LAYER_SPECS; si++) {
                const LZLayerSpec *sp = &LAYER_SPECS[si];
                LZTensor *field;
                if (sp->layer_type >= 0 && sp->layer_type != L->type) continue;
                field = (LZTensor *)((char *)L + sp->field);
                lz_t_free(field);
            }
            /* Per-expert arrays: not in LAYER_SPECS (see model_walk), so
               not covered by the loop above. free(NULL) is a no-op, so
               the three free() calls below are safe even for a non-MoE
               layer whose arrays were never allocated by setup_layers.

               INDEXING them is a different question: setup_layers does
               three separate calloc calls and checks all three, so a
               failure on the second leaves w1 set and w2 NULL. Indexing
               with a single `if (L->moe_expert_w1)` guard would then
               evaluate &((LZTensor *)0)[e] and crash on the cleanup
               path of an out-of-memory error, which is the one path a
               128MB Win98 box is most likely to take. Each array must
               therefore be checked on its own before indexing -
               free(NULL) says nothing about NULL[e]. */
            {
                int e;
                for (e = 0; e < m->config.num_experts; e++) {
                    if (L->moe_expert_w1) lz_t_free(&L->moe_expert_w1[e]);
                    if (L->moe_expert_w2) lz_t_free(&L->moe_expert_w2[e]);
                    if (L->moe_expert_w3) lz_t_free(&L->moe_expert_w3[e]);
                }
            }
            free(L->moe_expert_w1);
            free(L->moe_expert_w2);
            free(L->moe_expert_w3);
        }
        free(m->layers);
    }
    if (m->mtp) {
        int si;
        for (si = 0; si < N_LAYER_SPECS; si++) {
            const LZLayerSpec *sp = &LAYER_SPECS[si];
            if (sp->layer_type >= 0 && sp->layer_type != LZ_LT_FULL) continue;
            /* Mirror the ffn_kind filter model_walk uses to bind this
               block (see the comment there): the MTP head's FFN is
               always the dense triplet, never latent MoE, so the MoE
               specs must be skipped here too. Currently a no-op either
               way - m->mtp is calloc'd and the load loop never binds
               those fields on a kunmoe checkpoint, so lz_t_free would
               just free(NULL) four times - but leaving the filter out
               here made this loop disagree with the one that fills the
               struct, which is exactly the kind of drift that let the
               load-side bug (looking for a nonexistent
               mtp.layers.0.mlp.gate.weight) go unnoticed. */
            if (sp->ffn_kind >= 0 && sp->ffn_kind != 0) continue;
            lz_t_free((LZTensor *)((char *)&m->mtp->blk + sp->field));
        }
        lz_t_free(&m->mtp->fc);
        lz_t_free(&m->mtp->norm);
        lz_t_free(&m->mtp->pre_fc_norm_hidden);
        lz_t_free(&m->mtp->pre_fc_norm_embedding);
        free(m->mtp);
    }
    if (m->bin_file) fclose(m->bin_file);
    lz_st_close(&m->st);
    memset(m, 0, sizeof(*m));
}
