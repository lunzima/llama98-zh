#include <stdarg.h>
#include <stddef.h>
#include "lz_int.h"   /* not <stdint.h>: VC++ 4.0 has no such header */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "err.h"
#include "lfn.h"
#include "lz_int.h"   /* lz_i64: the 64-bit type, portably */
#include "lz4d.h"
#include "model.h"
#include "ops.h"   /* lz_bf16_store_g, LZ_MM_WIDEN_MAX */

#if LZ_DBG_LAYER_HASH
/* Per-layer residual hash for the fast-mode (--fastfp) cross-target
   divergence hunt. PURE INTEGER over the buffer's bit patterns: no float
   op, so it cannot change the rounding of the arithmetic it measures.
   Lives here (a large, cold TU) rather than in forward.c so the compiler
   cannot inline it into the layer loop and reorder float work around it.
   -DLZ_DBG_LAYER_HASH=1 is a diagnostic build only; nothing in a normal
   build references it. */
void lz_dbg_layer_hash(const char *tag, int li, const void *p, size_t nf) {
    const unsigned char *b = (const unsigned char *)p;
    unsigned long long h = 1469598103934665603ull;   /* FNV-1a */
    size_t i;
    for (i = 0; i < nf * 4u; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    fprintf(stderr, "LHASH %s %d %llu\n", tag, li,
            (unsigned long long)h);
}
#endif /* LZ_DBG_LAYER_HASH */

/* 16-byte aligned malloc/free for the weight buffers (field->f and the
   quantized planes field->q/scale/zero). The SSE1/SSE2 row/dot kernels
   read these with movaps/movdqa after the alignment. field->q is aligned
   uniformly so the Q8/Q4_1/Q16 formats - whose 32-element group is 32,
   16 or 64 bytes, all 16-byte multiples - become aligned loads; the
   Q6_1 2-bit plane and Q2 have an 8-byte group stride, so their loads
   stay movdqu even over the aligned base (16-aligned is still only
   8-aligned every other group). Over-allocates by one pointer plus the
   pad and stores the ORIGINAL pointer just before the aligned one, the
   same shape as forward.c's xcalloc/xfree. */
static void *aligned_malloc(size_t n, size_t sz) {
    char *base = (char *)malloc(n * sz + 16 + sizeof(void *));
    char *aligned;
    if (!base) return NULL;
    aligned = base + sizeof(void *);
    aligned += (size_t)(16 - ((size_t)aligned & 15)) & 15;
    ((void **)(void *)aligned)[-1] = base;
    return aligned;
}

static void aligned_free(void *p) {
    if (p) free(((void **)p)[-1]);
}

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

/* Settle hadamard_o / hadamard_down into concrete values and refuse the
   illegal ones. Called by BOTH loaders (config.json and model.bin) once
   lin_value_dim / intermediate_size are known, so the forward pass reads
   one already-resolved number and never re-derives.

   -1 means the product predates the fields; the two derivations below
   reproduce the forward pass's own inline computation, so a legacy
   model keeps its exact behaviour.

   An explicit block size that is not a power of two, or does not divide
   the projection's input dim, is a refusal rather than a fallback to 0:
   quietly dropping the transform would let a mis-exported model produce
   numbers that look reasonable and are not. Block size 1 IS legal and
   means off - a 1x1 Hadamard is the identity, so the forward pass's
   `blk > 1` guard is exact, not an approximation. */
static int hadamard_ok(int blk, int dim) {
    if (blk == 0) return 1;                    /* off */
    if (blk < 0) return 0;                     /* -1 is handled by the caller */
    if (blk & (blk - 1)) return 0;             /* not a power of two */
    return dim > 0 && dim % blk == 0;
}

static int resolve_hadamard(LZModelConfig *c, char *errbuf, int errlen) {
    int vdim = c->lin_value_dim;
    int idim = c->intermediate_size;
    if (c->hadamard_o == -1) {
        int oblk = vdim;
        if (oblk < 2 || (oblk & (oblk - 1))) oblk = 0;
        c->hadamard_o = oblk;
    } else if (!hadamard_ok(c->hadamard_o, vdim)) {
        qerr(errbuf, errlen, LZ_ERR_CFG_HADAMARD,
             "hadamard_o", c->hadamard_o, "lin_value_dim", vdim);
        return 1;
    }
    if (c->hadamard_down == -1) {
        int dblk = 512;
        while (dblk > 1 && idim % dblk) dblk >>= 1;
        if (dblk == 1) dblk = 0;
        c->hadamard_down = dblk;
    } else if (!hadamard_ok(c->hadamard_down, idim)) {
        qerr(errbuf, errlen, LZ_ERR_CFG_HADAMARD,
             "hadamard_down", c->hadamard_down, "intermediate_size", idim);
        return 1;
    }
    /* One hadamard_down serves two consumers: the body's dense FFN and
       the MTP block's, whose width is its own field. The block MUST
       divide both - forward.c's block loop steps b_i += blk to n and
       would write past the row otherwise. Equal widths (every product so
       far, and the default when the key is absent) make this vacuous. */
    if (c->use_subn && c->mtp_n_layers > 0 && c->hadamard_down > 1 &&
        (c->mtp_intermediate_size <= 0 ||
         c->mtp_intermediate_size % c->hadamard_down)) {
        qerr(errbuf, errlen, LZ_ERR_CFG_HADAMARD,
             "hadamard_down", c->hadamard_down,
             "mtp_intermediate_size", c->mtp_intermediate_size);
        return 1;
    }
    return 0;
}

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
    c->use_subn = lz_json_get_bool(&j, tc, "use_subn", 0);
    /* Block-diagonal Hadamard block sizes. -1 = "derive" (resolved by
       resolve_hadamard below), which is what a config.json written
       before these keys existed gets. */
    c->hadamard_o    = lz_json_get_int(&j, tc, "hadamard_o", -1);
    c->hadamard_down = lz_json_get_int(&j, tc, "hadamard_down", -1);

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
             c->rotary_dim, c->head_dim,
             (int)(c->partial_rotary_factor * 100.0f + 0.5f));
        return 1;
    }
    /* After lin_value_dim, which the derivations read. */
    if (resolve_hadamard(c, errbuf, errlen) != 0) return 1;
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
static int present_subn(const LZModelConfig *c) { return c->use_subn; }
static int present_subn_shared(const LZModelConfig *c) {
    return c->use_subn && c->moe_shared_width > 0;
}

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

    /* SubLN (use_subn=1 only): dense-FFN per-projection input RMSNorms. */
    { "mlp.gate_proj.norm.weight",       -1, 0, LOFF(gate_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "mlp.up_proj.norm.weight",         -1, 0, LOFF(up_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "mlp.down_proj.norm.weight",       -1, 0, LOFF(down_norm),
      { DK_INTER, DK_END, DK_END }, present_subn },

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

    /* SubLN (use_subn=1 only): full attention's per-projection INPUT
       RMSNorms - not q_norm/k_norm above, which are the per-head QK
       norms over head_dim and are present regardless. */
    { "self_attn.q_proj.norm.weight", LZ_LT_FULL, -1, LOFF(attn_q_subn_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "self_attn.k_proj.norm.weight", LZ_LT_FULL, -1, LOFF(attn_k_subn_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "self_attn.v_proj.norm.weight", LZ_LT_FULL, -1, LOFF(attn_v_subn_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "self_attn.o_proj.norm.weight", LZ_LT_FULL, -1, LOFF(attn_o_subn_norm),
      { DK_ATTN_Q, DK_END, DK_END }, present_subn },

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

    /* SubLN (use_subn=1 only): KDA per-projection input RMSNorms.
       kda_o_subn_norm is o_proj's OWN input norm over lin_value_dim - NOT
       kda_o_norm above (the per-head gated-silu norm). */
    { "linear_attn.q_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_q_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "linear_attn.k_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_k_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "linear_attn.v_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_v_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "linear_attn.g_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_g_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "linear_attn.o_proj.norm.weight", LZ_LT_KDA, -1, LOFF(kda_o_subn_norm),
      { DK_VALUE, DK_END, DK_END }, present_subn },

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
      { DK_HIDDEN, DK_SHARED_W, DK_END }, present_moe_shared },

    /* SubLN (use_subn=1 only): the latent-MoE block's per-projection
       input RMSNorms. The router carries one because its logits feed a
       softmax - an un-normalized input is a changed temperature, not a
       skipped normalization. */
    { "mlp.gate.norm.weight",                  -1, 1, LOFF(moe_gate_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "mlp.routed_expert_down_proj.norm.weight", -1, 1, LOFF(moe_down_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn },
    { "mlp.routed_expert_up_proj.norm.weight", -1, 1, LOFF(moe_up_norm),
      { DK_LATENT, DK_END, DK_END }, present_subn },
    { "mlp.shared_experts.gate_proj.norm.weight", -1, 1,
      LOFF(moe_shared_gate_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn_shared },
    { "mlp.shared_experts.up_proj.norm.weight", -1, 1,
      LOFF(moe_shared_up_norm),
      { DK_HIDDEN, DK_END, DK_END }, present_subn_shared },
    { "mlp.shared_experts.down_proj.norm.weight", -1, 1,
      LOFF(moe_shared_down_norm),
      { DK_SHARED_W, DK_END, DK_END }, present_subn_shared }
};
#define N_LAYER_SPECS ((int)(sizeof(LAYER_SPECS) / sizeof(LAYER_SPECS[0])))

static lz_i64 resolve_dim(const LZModelConfig *c, int kind) {
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
    case DK_HIDDEN2:    return (lz_i64)c->hidden_size * 2;
    case DK_KEY:        return c->lin_key_dim;
    case DK_GATE_RANK:  return c->kda_gate_rank;
    case DK_NVK:        return (lz_i64)c->lin_n_v_heads * c->lin_k_head_dim;
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
        lz_i64 want = resolve_dim(c, dims[i]);
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

static int m_uses_lz4(const LZModel *m);
static size_t lz4_reader(void *ctx, void *dst, size_t sz);

/* One tensor of the walk: resolve `name` (via the bin desc or the
   safetensors table), check its shape against `dims`, then hand it to
   `visit`. Extracted from model_walk so the obtain/check/visit sequence
   is written once - the per-site parts (name and `field`) are the
   callers'. */
static int model_walk_one(LZModel *m, const char *name, const int *dims,
                          LZTensor *field, LZVisit visit, void *ctx,
                          char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    const LZStTensor *t;
    LZStTensor desc;            /* bin-path desc; lives for this call so t stays valid */
    /* bin-path check is "reader set", not "bin_file non-NULL": the
       compressed sibling (model.bin.lz4) streams through m->rd with
       bin_file left NULL, and must take the same desc path. */
    if (m->bin_file || m_uses_lz4(m)) {
        lz_i64 ne = 1;
        int di;
        memset(&desc, 0, sizeof(desc));
        desc.name = name;
        for (di = 0; di < 3 && dims[di] != DK_END; di++) {
            lz_i64 d = resolve_dim(c, dims[di]);
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
    if (check_shape(c, t, dims, errbuf, errlen) != 0) return 1;
    if (visit(m, t, field, ctx, errbuf, errlen) != 0) return 1;
    return 0;
}

static int model_walk(LZModel *m, LZVisit visit, void *ctx,
                    char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    static const int EMBED_DIMS[3] = { DK_VOCAB, DK_HIDDEN, DK_END };
    static const int NORM_DIMS[3] = { DK_HIDDEN, DK_END, DK_END };
    char name[256];
    int li, si;

    /* top level: embedding and final norm */
    snprintf(name, sizeof(name), "%sembed_tokens.weight", m->prefix);
    if (model_walk_one(m, name, EMBED_DIMS, &m->embed_tokens, visit, ctx,
                       errbuf, errlen) != 0) return 1;

    snprintf(name, sizeof(name), "%snorm.weight", m->prefix);
    if (model_walk_one(m, name, NORM_DIMS, &m->final_norm, visit, ctx,
                       errbuf, errlen) != 0) return 1;

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
            field = (LZTensor *)((char *)L + sp->field);
            if (model_walk_one(m, name, sp->dims, field, visit, ctx,
                               errbuf, errlen) != 0) return 1;
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
                    arr = *(LZTensor **)((char *)L + EXPERT_SPECS[k].field);
                    field = &arr[e];
                    if (model_walk_one(m, name, EXPERT_SPECS[k].dims, field,
                                       visit, ctx, errbuf, errlen) != 0) return 1;
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
            field = (LZTensor *)((char *)&m->mtp->blk + sp->field);
            if (model_walk_one(m, name, sp->dims, field, visit, ctx,
                               errbuf, errlen) != 0) return 1;
        }
        for (k = 0; k < (int)(sizeof(MTP_FFN_SPECS) / sizeof(MTP_FFN_SPECS[0])); k++) {
            LZTensor *field;
            snprintf(name, sizeof(name), "mtp.layers.0.%s", MTP_FFN_SPECS[k].suffix);
            field = (LZTensor *)((char *)&m->mtp->blk + MTP_FFN_SPECS[k].field);
            if (model_walk_one(m, name, MTP_FFN_SPECS[k].dims, field, visit,
                               ctx, errbuf, errlen) != 0) return 1;
        }
        for (k = 0; k < (int)(sizeof(MTP_SPECS) / sizeof(MTP_SPECS[0])); k++) {
            LZTensor *field;
            snprintf(name, sizeof(name), "mtp.%s", MTP_SPECS[k].suffix);
            field = (LZTensor *)((char *)m->mtp + MTP_SPECS[k].field);
            if (model_walk_one(m, name, MTP_SPECS[k].dims, field, visit, ctx,
                               errbuf, errlen) != 0) return 1;
        }
    }
    return 0;
}

/* --------------------------------------------------------------- binding */

typedef struct {
    unsigned char *claimed;     /* whether each tensor was claimed; for leak detection */
    lz_i64 n_params;
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

/* Where the weights are when they are not in model.safetensors.
 *
 * A HUGGING FACE EXPORT NAMES ITS SHARDS AND LISTS THEM IN
 * model.safetensors.index.json, whose weight_map sends every tensor to a
 * file. Without this the loader looks for exactly one name, and a
 * perfectly ordinary export fails with "cannot find model.safetensors"
 * in a directory that plainly holds the weights - which is what
 * Qwen3.5-0.8B does here, its single shard being called
 * model.safetensors-00001-of-00001.safetensors.
 *
 * ONE SHARD ONLY, and the limit is deliberate rather than provisional.
 * LZSafetensors owns a single FILE * and one tensor table; spanning
 * several would mean a shard index per tensor and a handle array, which
 * is a real feature and not a redirect. Every index this has been shown
 * maps to one file, so the redirect is the whole of what is needed
 * today - and a multi-shard index gets a message SAYING it is
 * unsupported rather than the missing-file one, because the difference
 * matters to whoever hits it.
 *
 * Returns 1 and fills `out` when the index named a single shard, -1 with
 * errbuf set for a MULTI-shard one, and 0 for everything else - no
 * index, unreadable, malformed, empty weight_map. Those all fall back
 * to the caller's ordinary "cannot find model.safetensors", which is
 * the right message when the index could not be used: the file the
 * reader is missing is still that one. Only the multi-shard case earns
 * a message of its own, because there the weights ARE all present and
 * the loader is the thing that cannot take them.
 */
static int st_path_from_index(const char *dir, char *out, int outlen,
                              char *errbuf, int errlen) {
    char ipath[512];
    void *txt;
    size_t len;
    LZJson j;
    const LZJsonNode *map, *n;
    const char *first = NULL;
    int rc = 0;

    if (!lz_lfn_exists(dir, "model.safetensors.index.json")) return 0;
    if (lz_lfn_path(dir, "model.safetensors.index.json", ipath,
                    (int)sizeof ipath, errbuf, errlen) != 0) return 0;
    txt = lz_read_file(ipath, &len, errbuf, errlen);
    if (!txt) return 0;
    if (lz_json_parse(&j, (const char *)txt, len, errbuf, errlen) != 0) {
        free(txt);
        return 0;
    }
    map = lz_json_get(&j, lz_json_root(&j), "weight_map");
    if (!map) goto done;
    /* Distinct VALUES, not the member count: the map has one entry per
       tensor and they nearly all name the same file. */
    for (n = lz_json_first(&j, map); n; n = lz_json_next(&j, n)) {
        if (n->type != LZ_JSON_STR || !n->text) continue;
        if (!first) { first = n->text; continue; }
        if (strcmp(first, n->text) != 0) {
            qerr(errbuf, errlen, LZ_ERR_ST_INDEX_SHARDS);
            rc = -1;
            goto done;
        }
    }
    if (!first) goto done;
    if (lz_lfn_path(dir, first, out, outlen, errbuf, errlen) == 0) rc = 1;
done:
    lz_json_free(&j);
    free(txt);
    return rc;
}

int lz_open(LZModel *m, const char *dir, char *errbuf, int errlen) {
    char path[512];
    LZBindCtx bc;
    int i;

    memset(m, 0, sizeof(*m));
    memset(&bc, 0, sizeof(bc));
    if (errbuf && errlen > 0) errbuf[0] = '\0';

    /* Both loaders below convert from the file's little-endian order
       using LZ_BIG_ENDIAN, so a build that guessed its host's byte order
       wrong reads every weight reversed and reports nothing. Checked here,
       at the single door both paths go through, and before any file is
       touched - the answer does not depend on the model. */
    if (!lz_endian_ok()) {
        qerr(errbuf, errlen, LZ_ERR_ENDIAN_MISMATCH);
        return 1;
    }

    /* mixed-precision bin preferred: if model.bin (or the compressed
       sibling model.bin.lz4, the stunt path) exists, load everything at
       once. Resolved rather than fopen'd directly (src/lfn.h): this probe
       decides between the bin and safetensors loaders, so a wrong "no"
       sends the whole load down the other path. lz_open_bin prefers the
       .lz4 file when both are present. */
    if (lz_lfn_exists(dir, "model.bin") || lz_lfn_exists(dir, "model.bin.lz4"))
        return lz_open_bin(m, dir, errbuf, errlen);

    if (lz_lfn_path(dir, "config.json", path, (int)sizeof path,
                    errbuf, errlen) != 0) return 1;
    if (lz_load_config(&m->config, path, errbuf, errlen) != 0) return 1;

    /* model.safetensors first, then the shard index. Tried in this order
       so an export carrying both keeps the behaviour it already had. */
    if (lz_lfn_exists(dir, "model.safetensors")) {
        if (lz_lfn_path(dir, "model.safetensors", path, (int)sizeof path,
                        errbuf, errlen) != 0) return 1;
    } else {
        int ir = st_path_from_index(dir, path, (int)sizeof path,
                                    errbuf, errlen);
        if (ir < 0) return 1;
        if (ir == 0) {
            /* No index either: report the file the user expected to
               have, which is still model.safetensors. */
            if (lz_lfn_path(dir, "model.safetensors", path,
                            (int)sizeof path, errbuf, errlen) != 0) return 1;
        }
    }
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

    field->dtype = 0;
    field->n = (int)t->n_elem;
    field->gs = 0;
    field->q = NULL;
    field->scale = NULL;
    field->f = NULL;

    /* A BF16 TENSOR STAYS BF16, which halves what it occupies and what
       reading it costs in DDR traffic. Expanding it to f32 while loading
       would turn a two-bytes-per-parameter file into four bytes in
       RAM - on a 64MB target that is the difference between a model
       fitting and not.
     *
     * Nothing is lost and nothing rounds: a bf16 IS the high half of an
     * f32, so the widening lz_t_f32 does at the point of use is the
     * identical `<<16` the loader performs here, only deferred. Every
     * value a kernel sees is the same value it saw before, which is why
     * this needs no switch and no accuracy argument.
     *
     * F16 is deliberately NOT included: its widening is a real format
     * conversion (5-bit exponent to 8, subnormal renormalisation), so it
     * would have to be undone on every read rather than being a shift,
     * and it still expands here. */
    /* TWO-DIMENSIONAL, OR ONE-DIMENSIONAL AND SMALL. The reason is a
       consumer's contract rather than anything about bf16.

       THE 1-D CASE IS THE ONE THE EXPORTER CARES ABOUT. tools/
       export_q8.py keeps a tensor at F32 when Q8_0 is not accurate
       enough for it, and its rule reads: "Non-2D tensors and those with
       n < 512 are always f32 ... conv1d (unquantizable: 4-wide rows),
       A_log and dt_bias (feed exp), norms (leverage)". Those are
       exactly the structures that had no middle option - Q8_0 too
       coarse, F32 more than they need - and bf16 is that middle. A
       first version of this guard was 2-D only and therefore served
       none of them.
     *
     * lz_t_f32() hands back a WHOLE-TENSOR f32 view. For an F32 tensor
     * that is free - it returns t->f and copies nothing - but a bf16
     * tensor has to be widened into the caller's scratch, and that
     * scratch is s->wscr, qcap floats. So a tensor read that way may be
     * narrow only if the WHOLE of it fits, which is what the 1-D clause
     * checks: qcap starts at hidden_size and only grows, so a norm
     * (hidden-wide) and A_log/dt_bias (per-head, smaller) all fit.
     *
     * 3-D IS STILL EXCLUDED, and conv1d is why. The checkpoint measured
     * here has it at [3072, 1, 4] - 12,288 elements against a qcap in
     * the hundreds - and conv_fixed_build reads it whole through
     * lz_t_f32. Storing it narrow overran s->wscr and crashed there;
     * found by running it, not by reading it. Serving it needs that
     * caller to move to lz_t_row_f32, which is a separate change.
     *
     * The 2-D tensors have no such problem: they are weight matrices,
     * reached through lz_matmul_w / lz_matmul_xq_nt, which widen a row
     * at a time into their own bounded buffer, or through
     * lz_t_row_f32(), whose output is a row by contract. They are also
     * where the memory is - 94 of this checkpoint's 161 tensors and
     * essentially all of its bytes, so excluding the other two shapes
     * costs almost nothing.
     *
     * The second clause is that bounded buffer's own limit, the same one
     * every quantized kernel in ops_matmul.c carries. Past it a tensor
     * stays f32, which costs memory and never correctness. */
    if (t->dtype == LZ_DT_BF16 && lz_bf16_store_g &&
        ((t->n_dims == 2 && t->shape[1] <= LZ_MM_WIDEN_MAX) ||
         (t->n_dims == 1 && t->n_elem <= LZ_MM_WIDEN_MAX))) {
        lz_i64 nb = t->n_elem * (lz_i64)2;
        field->q = (int8_t *)aligned_malloc((size_t)t->n_elem, 2);
        if (!field->q) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC_BYTES, t->name, nb);
            return 1;
        }
        if (lz_st_read_raw(&m->st, t, field->q, errbuf, errlen) != 0) {
            aligned_free(field->q);
            field->q = NULL;
            return 1;
        }
        /* safetensors is little-endian by specification; the widening in
           lz_t_f32 reads the two bytes by name, so no swap is needed
           here and none would be correct on a big-endian host either. */
        field->dtype = LZ_FMT_BF16;
        m->bytes_alloc += nb;
        return 0;
    }

    /* Expanding to f32 needs 4*elem bytes; f16 raw data is read into
       the second half and expanded in place, so this buffer doubles as
       the read buffer - no extra scratch memory. */
    field->f = (float *)aligned_malloc((size_t)t->n_elem, sizeof(float));
    if (!field->f) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC_BYTES,
             t->name, t->n_elem * (lz_i64)sizeof(float));
        return 1;
    }
    if (lz_st_read_f32(&m->st, t, field->f, t->n_elem, errbuf, errlen) != 0) {
        aligned_free(field->f);
        field->f = NULL;
        return 1;
    }
    m->bytes_alloc += t->n_elem * (lz_i64)sizeof(float);
    return 0;
}

/* Read one quantized tensor's group size and vet it. Non-zero on either
   a short read or a value this format cannot use; the caller reports
   both as LZ_ERR_TENSOR_GS, which is what it did when the read and the
   checks shared one `if`.

   They cannot share one any more: the value has to be converted from
   little-endian BETWEEN being read and being divided by, and on a
   big-endian host an unconverted group size is not merely wrong but
   usually enormous, so `n % gs` would silently pass the "divides evenly"
   test by never dividing at all.

   mul32 says whether this format also requires a multiple of 32 - the
   sub-byte packings fix their sub-blocks at 32 elements (see model.h),
   Q8_0 does not. */
static size_t lz4_reader(void *ctx, void *dst, size_t sz);

static int m_uses_lz4(const LZModel *m) {
    return m->rd == lz4_reader;
}

static int read_gs(LZModel *m, uint32_t n, uint32_t *gs, int mul32) {
    if (m->rd(m->rd_ctx, gs, 4) != 4) return 1;
    lz_le32(gs, 1);
    if (*gs == 0 || (n % *gs) != 0) return 1;
    if (mul32 && (*gs % 32) != 0) return 1;
    return 0;
}

/* Read one tensor's header (dtype byte + LE element count). Shared by the
   bin path's tensor reads; factored out because every branch repeats it. */
static int read_tensor_hdr(LZModel *m, uint8_t *dtype, uint32_t *n) {
    if (m->rd(m->rd_ctx, dtype, 1) != 1 ||
        m->rd(m->rd_ctx, n, 4) != 4) return 1;
    return 0;
}

/* bin path: tensors are streamed from the reader in model_walk order
   (dtype 0=f32 / 1=Q8_0). All reads go through m->rd, so the same code
   streams from model.bin (file_reader) or model.bin.lz4 (lz4d_read). */
static int visit_read_bin(LZModel *m, const LZStTensor *t, LZTensor *field,
                          void *ctx, char *errbuf, int errlen) {
    uint8_t dtype;
    uint32_t n, gs;
    (void)ctx;

    /* Every multi-byte field below - the element count, the group size,
       the f32 planes and the Q16_0 int16 plane - is little-endian in the
       file and is converted after reading. The Q8_0, Q4_1, Q6_1 and T2
       code planes are NOT, and must not be: those are byte and sub-byte
       packings whose unpackers index bits within a byte, so their bytes
       are already in the only order they have. dtype is one byte.
       lz_le32/lz_le16 are empty on a little-endian build. */
    if (read_tensor_hdr(m, &dtype, &n)) {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
        return 1;
    }
    lz_le32(&n, 1);
    if ((lz_i64)n != t->n_elem) {
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
        field->f = (float *)aligned_malloc((size_t)n, sizeof(float));
        if (!field->f) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->f, (size_t)n * sizeof(float))
                != (size_t)n * sizeof(float)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            aligned_free(field->f);
            field->f = NULL;
            return 1;
        }
        lz_le32(field->f, (size_t)n);
        m->bytes_alloc += (lz_i64)n * sizeof(float);
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
        if (read_gs(m, n, &gs, 1) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n * 2, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n * 2) != (size_t)n * 2 ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        /* The only quantized code plane that is not bytes: Q16_0 stores
           int16 and the kernels read it through an int16_t *, so its
           halves have to be put the right way round here. */
        lz_le16(field->q, (size_t)n);
        lz_le32(field->scale, (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n * 2 +
                          (lz_i64)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q6_1) {
        /* two planes: n/2 bytes 4bit + n/4 bytes 2bit + n/gs scales +
           n/gs mins. Both planes live contiguously in one malloc, 4bit
           first. */
        field->dtype = LZ_FMT_Q6_1;
        field->f = NULL;
        if (read_gs(m, n, &gs, 1) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n / 2 + (size_t)n / 4, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        field->zero = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale || !field->zero) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n / 2 + (size_t)n / 4)
                != (size_t)n / 2 + (size_t)n / 4 ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs) ||
            m->rd(m->rd_ctx, field->zero, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        lz_le32(field->scale, (size_t)(n / gs));
        lz_le32(field->zero,  (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n / 2 + (lz_i64)n / 4 +
                          2 * (lz_i64)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_T2) {
        /* ternary: n/4 bytes of 2-bit codes + n/gs scales. NO mins -
           ternary is symmetric, and `zero` staying NULL is what the
           kernels branch on to use -scale as the hoisted coefficient
           instead of a stored min (see model.h). */
        field->dtype = LZ_FMT_T2;
        field->f = NULL;
        if (read_gs(m, n, &gs, 1) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n / 4, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n / 4) != (size_t)n / 4 ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        lz_le32(field->scale, (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n / 4 +
                          (lz_i64)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q4_1) {
        /* nibbles: n/2 bytes of data + n/gs scales + n/gs mins.
           gs must be a multiple of 32 - nibble sub-blocks are fixed at
           32 elements (see model.h). */
        field->dtype = LZ_FMT_Q4_1;
        field->f = NULL;
        if (read_gs(m, n, &gs, 1) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n / 2, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        field->zero = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale || !field->zero) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            lz_t_free(field);
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n / 2) != (size_t)n / 2 ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs) ||
            m->rd(m->rd_ctx, field->zero, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            lz_t_free(field);
            return 1;
        }
        lz_le32(field->scale, (size_t)(n / gs));
        lz_le32(field->zero,  (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n / 2 +
                          2 * (lz_i64)(n / gs) * sizeof(float);
    } else if (dtype == LZ_FMT_Q8_0) {
        field->dtype = LZ_FMT_Q8_0;
        field->f = NULL;
        if (read_gs(m, n, &gs, 0) != 0) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_GS, t->name);
            return 1;
        }
        field->gs = (int)gs;
        field->q = (int8_t *)aligned_malloc((size_t)n, 1);
        field->scale = (float *)aligned_malloc((size_t)(n / gs), sizeof(float));
        if (!field->q || !field->scale) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_ALLOC, t->name);
            aligned_free(field->q);
            aligned_free(field->scale);
            field->q = NULL;
            field->scale = NULL;
            return 1;
        }
        if (m->rd(m->rd_ctx, field->q, (size_t)n) != (size_t)n ||
            m->rd(m->rd_ctx, field->scale, sizeof(float) * (n / gs))
                != sizeof(float) * (n / gs)) {
            qerr(errbuf, errlen, LZ_ERR_TENSOR_READ, t->name);
            aligned_free(field->q);
            aligned_free(field->scale);
            field->q = NULL;
            field->scale = NULL;
            return 1;
        }
        lz_le32(field->scale, (size_t)(n / gs));
        m->bytes_alloc += (lz_i64)n + (lz_i64)(n / gs) * sizeof(float);
    } else {
        qerr(errbuf, errlen, LZ_ERR_TENSOR_DTYPE, t->name, dtype);
        return 1;
    }
    return 0;
}

/* Load config and all weights in one pass from model.bin */
static int lz_open_bin(LZModel *m, const char *dir, char *errbuf, int errlen);

/* Two reader implementations behind LZModel.rd/rd_ctx:
   - file_reader over bin_file (plain model.bin),
   - lz4d_read over a LZ4Stream* (model.bin.lz4, the stunt path).
   Everything the bin path reads - header fields, group sizes, tensor
   planes - goes through rd, so whether the file is compressed is
   invisible below lz_open_bin. */
static size_t file_reader(void *ctx, void *dst, size_t sz) {
    return fread(dst, 1, sz, (FILE *)ctx);
}

/* Reader-shaped adapter for lz4d_read: the LZModel reader signature takes
   void *ctx; lz4d_read wants a LZ4Stream *.

   NAMED FOR THIS FILE, not for lz4d. It was lz4d_read_void, which put it
   in lz4d.c's family - and it is not lz4d's: it is model.c's half of the
   pair with file_reader above, it is static, and m_uses_lz4 identifies
   the reader by comparing against its ADDRESS, so it could not move even
   if the name were right. */
static size_t lz4_reader(void *ctx, void *dst, size_t sz) {
    return lz4d_read((LZ4Stream *)ctx, dst, sz);
}

static void close_reader(LZModel *m) {
    if (m_uses_lz4(m)) {
        lz4d_close((LZ4Stream *)m->rd_ctx);
    } else if (m->bin_file) {
        fclose(m->bin_file);
    }
    m->bin_file = NULL;
    m->rd = NULL;
    m->rd_ctx = NULL;
}

static int lz_open_bin(LZModel *m, const char *dir, char *errbuf, int errlen) {
    char path[512];
    char magic[4];
    uint32_t ver;
    LZModelConfig *c = &m->config;
    int i;
    lz_i64 n_params = 0;

    /* Decide BY MAGIC, not extension: read model.bin's first 4 bytes; an
       LZ4 frame's magic (04 22 4d 18 LE) means the weight file is itself
       LZ4-compressed - whatever its name - and is streamed through lz4d.
       Anything else is rewound and read plain with file_reader. The
       legacy model.bin.lz4 sibling is still accepted when model.bin is
       absent (lz_open's probe already proved one of the two opens, so a
       failure here is a race and keeps LZ_ERR_BIN_OPEN's wording). */
    {
        uint8_t probe[4];
        int is_lz4 = 0;
        int got = 0;
        if (lz_lfn_path(dir, "model.bin", path, (int)sizeof path,
                        errbuf, errlen) == 0) {
            m->bin_file = fopen(path, "rb");
            if (!m->bin_file) {
                qerr(errbuf, errlen, LZ_ERR_BIN_OPEN, path);
                return 1;
            }
            got = (int)fread(probe, 1, 4, m->bin_file) == 4;
            is_lz4 = got && (uint32_t)((uint32_t)probe[0]
                                       | ((uint32_t)probe[1] << 8)
                                       | ((uint32_t)probe[2] << 16)
                                       | ((uint32_t)probe[3] << 24))
                           == LZ4D_FRAME_MAGIC;
        } else if (lz_lfn_path(dir, "model.bin.lz4", path,
                               (int)sizeof path, errbuf, errlen) == 0) {
            is_lz4 = 1;          /* legacy sibling: must be a frame */
        } else {
            qerr(errbuf, errlen, LZ_ERR_BIN_OPEN, "model.bin(.lz4)");
            return 1;
        }
        if (is_lz4) {
            /* Declared before the statements, not beside its use: this
               tree's floor is a 1995 C89 compiler and
               build/c89_floor_gate.sh holds model.c at zero violations. */
            LZ4Stream *lz;
            if (m->bin_file) fclose(m->bin_file);
            m->bin_file = NULL;
            lz = lz4d_open(path, errbuf, errlen);
            if (!lz) return 1;
            m->rd = lz4_reader;
            m->rd_ctx = lz;
        } else {
            rewind(m->bin_file);     /* the 4-byte probe is the header's magic */
            m->rd = file_reader;
            m->rd_ctx = m->bin_file;
        }
    }

/* RDB reads BYTES and leaves them alone; RD reads 32-bit WORDS and
   converts them from the file's little-endian order.

   Every field in this header is 4 bytes wide - int32 or float32, and the
   conversion is the same for both - except the magic, which is four
   characters and is the one caller of RDB. Putting the conversion in the
   macro rather than at each of the sixty-odd call sites is not brevity:
   a field added later gets it automatically, whereas a list of explicit
   conversions is a list somebody has to remember to extend, and the
   symptom of forgetting is a plausible-looking wrong number rather than
   a failure. lz_le32 is empty on a little-endian build, so this is also
   byte-for-byte what it was on every shipping target. */
#define RDB(ptr, sz) do { \
    if (m->rd(m->rd_ctx, (ptr), (sz)) != (size_t)(sz)) { \
        qerr(errbuf, errlen, LZ_ERR_BIN_CFG_READ); \
        goto fail; \
    } } while (0)

#define RD(ptr, sz) do { \
    RDB((ptr), (sz)); \
    lz_le32((ptr), (size_t)(sz) / 4); \
    } while (0)

    RDB(magic, 4);
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
        ver != 6 && ver != 7 && ver != 8) {
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
    /* v7 header tail: use_subn (SubLN). Absent (ver < 7) leaves it at 0
       from lz_open's memset(m, 0, ...) before this function is reached -
       the same "ordinary pre-layer norms" default lz_load_config uses for
       a config.json without a use_subn key, so the two loading paths
       cannot disagree. The exporter bumps to v7 only when a checkpoint
       actually sets use_subn, so legacy bins stay v6 byte for byte. */
    if (ver >= 7) {
        RD(&c->use_subn, 4);
    }
    /* v8 header tail: the two block-diagonal Hadamard block sizes. Unlike
       every tail above, absent (ver < 8) must NOT keep lz_open's
       memset(m, 0, ...) value: 0 is the legal, meaningful "no Hadamard"
       setting, and a v7 SubLN product DOES apply one. Set the sentinel
       explicitly so resolve_hadamard derives what those products were
       built with. */
    if (ver >= 8) {
        RD(&c->hadamard_o, 4);
        RD(&c->hadamard_down, 4);
    } else {
        c->hadamard_o = -1;
        c->hadamard_down = -1;
    }
#undef RD
#undef RDB

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
    if (resolve_hadamard(c, errbuf, errlen) != 0) goto fail;

    m->layers = (LZLayer *)calloc((size_t)c->n_layers, sizeof(LZLayer));
    if (!m->layers) {
        qerr(errbuf, errlen, LZ_ERR_LAYERS_ALLOC);
        goto fail;
    }
    if (setup_layers(m, errbuf, errlen) != 0) goto fail;

    if (alloc_mtp(m, errbuf, errlen) != 0) goto fail;

    if (model_walk(m, visit_read_bin, NULL, errbuf, errlen) != 0) {
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
    close_reader(m);
    return 0;

fail:
    close_reader(m);
    lz_free(m);
    return 1;
}


void lz_t_free(LZTensor *t) {
    if (!t) return;
    aligned_free(t->f);
    aligned_free(t->q);
    aligned_free(t->scale);
    aligned_free(t->zero);
    /* Freed here, not through ops.c: this file owns the struct's memory,
       and model.c does not include ops.h. */
    free(t->sq);
    free(t->sexp);
    free(t->zq);
    free(t->zexp);
    t->f = NULL;
    t->q = NULL;
    t->scale = NULL;
    t->zero = NULL;
    t->sq = NULL;
    t->sexp = NULL;
    t->zq = NULL;
    t->zexp = NULL;
    t->sq_row = 0;
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
    close_reader(m);
    lz_st_close(&m->st);
    memset(m, 0, sizeof(*m));
}
