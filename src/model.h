#ifndef LZ_MODEL_H
#define LZ_MODEL_H

#include <stdint.h>
#include <stdio.h>

#include "safetensors.h"

/* Loading for the Qwen3.5/3.6 family (qwen3_5_text) models, plus the
   KunMoe variant: KDA linear attention
   (per-channel decay) and/or latent MoE FFN in place of GatedDeltaNet
   and dense FFN. The engine supports mixed layers (GatedDeltaNet or KDA
   linear attention alternating with gated full attention) + SiLU-gated
   dense FFN or latent MoE + RMSNorm. Reads raw safetensors directly, no
   Python conversion in between.

   Architecture points (pruning and the C forward must both honor):
   - mixed layers: linear_attention(GatedDeltaNet or KDA) and
     full_attention alternate; the 0.8B is LLLA x6 = 24 layers;
   - full_attention has an output gate: q_proj output width is
     2*n_heads*head_dim, split PER HEAD as interleaved
     [q(head_dim), gate(head_dim)], NOT [all q][all gate] - this layout
     is very easy to get wrong;
   - QK-Norm: q/k each get one RMSNorm over head_dim;
   - tie_word_embeddings: no standalone lm_head; reuses embed_tokens;
   - KDA (LZ_LT_KDA): per-channel decay generalizing GDN's per-head
     scalar decay; separate q/k/v/g projections instead of one fused
     in_proj_qkv (see LZLayer's kda_* fields and ops.h's lz_kda_step);
   - latent MoE (LZLayer.ffn_moe): FFN layers from config.
     first_k_dense_replace on route through a compressed latent space
     with top-k routed experts plus one optional full-width shared
     expert (see LZLayer's moe_* fields). */

#define LZ_MAX_LAYERS 128
#define LZ_MAX_EXPERTS 128

#define LZ_LT_LINEAR 0      /* GatedDeltaNet */
#define LZ_LT_FULL   1      /* gated self-attention */
#define LZ_LT_KDA    2      /* Kimi Delta Attention: per-channel decay,
                                separate q/k/v/g projections. Selected when
                                config.json's top-level (or text_config's)
                                model_type is "kunmoe" - see
                                lz_load_config. The JSON layer_types string
                                stays "linear_attention" either way; this
                                code is what the string maps to. */

/* Weight format codes. Carried per tensor; mixed precision is simply
   different tensors having different codes. Adding a format needs: a
   code here, a dispatch branch in ops.c, and a name in export_q8.py's
   recipe table. */
#define LZ_FMT_F32   0      /* n f32s */
#define LZ_FMT_Q8_0  1      /* gs int8s + 1 f32 scale (symmetric; zero on the grid) */
#define LZ_FMT_Q4_1  2      /* gs 4-bit nibbles + 1 f32 scale + 1 f32 min (asymmetric) */
#define LZ_FMT_Q6_1  3      /* gs 6-bit (4-bit plane + 2-bit plane) + scale + min */
#define LZ_FMT_Q16_0 4      /* gs int16s + 1 f32 scale (symmetric) */

/* Weight tensor. Row-major (out, in); quantization groups are
   contiguous within a row.

   Nibble layout (LZ_FMT_Q4_1): a 32-element sub-block takes 16 bytes;
   byte j's LOW nibble is element j, its HIGH nibble is element j+16.
   Sub-blocks are fixed at 32 regardless of gs - the kernel's inner
   loop is always "32 at a time"; gs only decides how often scale/min
   change.

   gs is the WEIGHT group size, any multiple of 32; activation
   quantization groups are always 32, decoupled via lz_act_gs().
   Coarser weight groups save per-token bytes (gs=128 vs gs=32 saves
   8.3%); the activation side gives up no precision. */
typedef struct {
    int    dtype;           /* LZ_FMT_* */
    int    n;               /* element count */
    int    gs;              /* quantization group size (0 for LZ_FMT_F32) */
    float *f;               /* F32: data; quantized formats: NULL */
    int8_t *q;              /* Q8_0: n int8s; Q4_1: n/2 packed nibbles (read unsigned)
                               Q6_1: n/2 bytes 4-bit plane then n/4 bytes 2-bit plane
                               Q16_0: n int16s (reinterpret as int16_t) */
    float *scale;           /* quantized: one scale per group, n/gs */
    float *zero;            /* Q4_1/Q6_1: one min per group, n/gs; NULL otherwise */
} LZTensor;

/* Q6_1's 2-bit plane layout: every 32 elements take 8 bytes; byte j's
   bits 0-1 are element j, bits 2-3 are j+8, bits 4-5 are j+16, bits
   6-7 are j+24.

   This grouping is chosen, not arbitrary: `pand 0x03` and `psrlw 2/4/6`
   yield exactly {0-7}, {8-15}, {16-23}, {24-31}, aligned group-for-group
   with the two 8-byte loads of the 4-bit plane - the activation side's
   pre-expanded registers are reused as-is, no reshuffling needed.

   Value q = lo + (hi<<4) in [0,63], dequantized w = q·d + m. The dot
   product splits exactly: dot(q,x) = dot(lo,x) + 16·dot(hi,x), two
   int32 accumulators merged afterwards, no precision loss (the hi term
   is 3·127·32 = 12k, far from the int32 limit). */

/* Get an f32 view: LZ_FMT_F32 returns the internal pointer directly;
   quantized formats dequantize into a scratch buffer (returned; the
   caller provides scratch with enough capacity). */
float *lz_t_f32(const LZTensor *t, float *scratch);

/* Get row `row` (dim elements) as f32 into out. Used for embedding
   lookups.

   Exists for discipline: layout knowledge of the formats may live only
   in ops.c. Reading t->q directly elsewhere re-implements
   dequantization in a second place and silently drifts - a quantized
   row mis-read as plain ints still produces finite logits with wrong
   embeddings (measured: logits cosine 0.34). Any "I'll just read t->q
   directly" repeats that mistake. */
void lz_t_row_f32(const LZTensor *t, int row, int dim, float *out);

/* Free the tensor's internal buffers (f or q+scale); zero the fields */
void lz_t_free(LZTensor *t);

typedef struct {
    int vocab_size;
    int hidden_size;
    int n_layers;
    int seq_len;                /* max_position_embeddings, generation cap */
    int layer_types[LZ_MAX_LAYERS];
    int n_linear_layers;
    int n_full_layers;

    /* full_attention */
    int n_heads;
    int n_kv_heads;
    int head_dim;

    /* linear_attention (GatedDeltaNet) */
    int lin_n_k_heads;
    int lin_n_v_heads;
    int lin_k_head_dim;
    int lin_v_head_dim;
    int conv_kernel;

    /* KDA gate (LZ_LT_KDA layers only; zero/unused otherwise).
       decay[h,k] = -exp(A_log[h]) * softplus(f_b_proj(f_a_proj(x))[h,k]
                    + dt_bias[h,k]), optionally clamped from below. See
       KunMoeConfig.kda_gate_rank / kda_gate_lower_bound
       and ops.h's lz_kda_step. */
    int   kda_gate_rank;             /* f_a_proj output width; 0 = no KDA layers */
    int   kda_has_gate_lower_bound;  /* config field present (not None) */
    float kda_gate_lower_bound;      /* K3 production value: -5.0 */

    /* latent MoE FFN (KunMoeConfig / LatentMoE). num_experts == 0 means
       no layer in this model uses MoE - every *_moe field below is then
       unused and every layer reads the classic dense gate/up/down_proj
       triplet. first_k_dense_replace decides PER LAYER: layer li is MoE
       iff num_experts > 0 && li >= first_k_dense_replace - this is
       resolved once into LZLayer.ffn_moe at load time (see model.c),
       not recomputed in the forward pass. */
    int num_experts;
    int num_experts_per_token;
    int num_shared_experts;          /* 0 or 1 in every reference; kept general */
    int moe_intermediate_size;       /* per-expert w1/w2/w3 inner width */
    int moe_latent_dim;              /* routed_expert_hidden_size: hidden<->latent width */
    int first_k_dense_replace;
    int moe_renormalize;             /* divide selected weights by their sum */
    int moe_router_sigmoid;          /* 1 = sigmoid (KunMoEGate default), 0 = softmax */
    int moe_latent_use_norm;         /* routed_expert_norm applied before routed_expert_up_proj */
    /* Resolved shared-expert width in HIDDEN space (shared expert does not
       go through the latent). 0 = no shared expert. Mirrors KunMoeConfig's
       derivation (shared_expert_intermediate_size, or else
       moe_intermediate_size * num_shared_experts) so both the config.json
       path (lz_load_config) and the bin path agree on one resolved number
       rather than each re-deriving it. */
    int moe_shared_width;

    int intermediate_size;
    float rms_norm_eps;
    int tie_word_embeddings;
    int attn_output_gate;           /* whether q_proj carries the gate (true in this model) */
    int full_attention_interval;    /* every Nth layer is a full_attention */
    /* config.json "mtp_num_hidden_layers"; 0 when the key is absent,
       which is every model this project has produced except upstream
       Qwen/Qwen3.5-0.8B itself (1, 15 mtp.* tensors, 20,452,864
       parameters). The number and the weights are not independent: a
       checkpoint carrying mtp.* tensors with this at 0, or the
       reverse, is a mismatch the loader can now see. */
    int mtp_n_layers;
    /* config.json "mtp_intermediate_size"; defaults to intermediate_size
       (lz_load_config) when the key is absent - stock upstream never
       writes it, because there the MTP block's dense FFN and the body's
       are the same width (both 3584 on the 0.8B). A DEDICATED field
       exists because that equality is NOT general: kunmoe-v2's body is
       first_k_dense_replace=0 (every layer routed, MoE), so its own
       intermediate_size is unused by the body and is repurposed
       to a small placeholder (512) - reusing THAT field
       for the MTP block's FFN width would silently couple two unrelated
       things, and the next person to touch either one would silently
       break the other. carve_manifest.json (mtp-carve's output) records
       dst_intermediate=1792 for kunmoe-v2's carved head; that number is
       NOT hardcoded anywhere in this engine - it flows in through this
       field (config.json / bin v5 header), same "sweepable knob, not a
       constant" discipline as LZ_SPEC_K (ops.h) and LZ_BATCH_MAX. Only
       model_walk's MTP-block FFN specs (model.c) read this; every other
       DK_INTER site stays body-only. */
    int mtp_intermediate_size;

    /* RoPE. Note rope_theta hides inside the nested rope_parameters;
       the top level does not expose it. */
    float rope_theta;
    float partial_rotary_factor;    /* 0.25 in this model */

    /* rope_parameters.rope_scaling (Qwen3.5 config field; absent = 0).
       type 0 none / 1 linear / 2 yarn (NTK-aware YaRN interpolation).
       factor is the interpolation factor (e.g. 8 for 4096 -> 32768);
       orig_max is original_max_position_embeddings (yarn only);
       beta_fast/beta_slow are yarn's correction-range bounds
       (HF defaults 32.0 / 1.0). The per-position cos/sin table in
       forward.c applies these when it builds rope_cs. */
    int   rope_scaling_type;
    float rope_scaling_factor;
    float rope_scaling_orig_max;
    float rope_scaling_beta_fast;
    float rope_scaling_beta_slow;

    /* Derived quantities: stored explicitly instead of recomputed
       everywhere. Recomputed-on-the-fly code
       silently propagates a missed change into weight binding. */
    int attn_q_dim;         /* n_heads * head_dim, attention output width */
    int attn_qgate_dim;     /* attn_q_dim * 2, q_proj output width */
    int attn_kv_dim;        /* n_kv_heads * head_dim */
    int lin_key_dim;        /* lin_n_k_heads * lin_k_head_dim */
    int lin_value_dim;      /* lin_n_v_heads * lin_v_head_dim */
    int lin_conv_dim;       /* lin_key_dim * 2 + lin_value_dim */
    int rotary_dim;         /* head_dim * partial_rotary_factor; only these leading dims rotate */
} LZModelConfig;

/* Per-layer weight pointers. Only half the fields are valid per layer
   type; the other half are NULL. */
typedef struct {
    int type;                       /* LZ_LT_LINEAR / LZ_LT_FULL */

    LZTensor input_layernorm;        /* (hidden) f32 */
    LZTensor post_attention_layernorm;

    LZTensor gate_proj;              /* (inter, hidden) Q8 */
    LZTensor up_proj;                /* (inter, hidden) Q8 */
    LZTensor down_proj;              /* (hidden, inter) Q8 */

    /* full_attention only */
    LZTensor q_proj;                 /* (attn_qgate_dim, hidden) Q8 */
    LZTensor k_proj;                 /* (attn_kv_dim, hidden) Q8 */
    LZTensor v_proj;                 /* (attn_kv_dim, hidden) Q8 */
    LZTensor o_proj;                 /* (hidden, attn_q_dim) Q8 */
    LZTensor q_norm;                 /* (head_dim) f32 */
    LZTensor k_norm;                 /* (head_dim) f32 */

    /* linear_attention only */
    LZTensor in_proj_qkv;            /* (lin_conv_dim, hidden) Q8 */
    LZTensor in_proj_z;              /* (lin_value_dim, hidden) Q8 */
    LZTensor in_proj_a;              /* (lin_n_v_heads, hidden) Q8 */
    LZTensor in_proj_b;              /* (lin_n_v_heads, hidden) Q8 */
    LZTensor conv1d;                 /* (lin_conv_dim, 1, conv_kernel) f32 */
    LZTensor A_log;                  /* (lin_n_v_heads) f32 */
    LZTensor dt_bias;                /* (lin_n_v_heads) f32 */
    LZTensor ssm_norm;               /* (lin_v_head_dim) f32 */
    LZTensor out_proj;               /* (hidden, lin_value_dim) Q8 */

    /* LZ_LT_KDA only. Tensor-for-tensor against KdaAttention -
       deliberately NOT reusing the GDN field names above,
       because the shapes differ (three independent q/k/v projections and
       convs vs one fused in_proj_qkv, a per-channel dt_bias vs per-head,
       a low-rank f_a/f_b decay path GDN has no slot for). The recurrent
       state shape (lin_n_v_heads, lin_k_head_dim, lin_v_head_dim) and its
       Q8/2-plane quantization are unchanged from GDN - see LZRunState. */
    LZTensor kda_q_proj;             /* (lin_key_dim, hidden) Q8 */
    LZTensor kda_k_proj;             /* (lin_key_dim, hidden) Q8 */
    LZTensor kda_v_proj;             /* (lin_value_dim, hidden) Q8 */
    LZTensor kda_q_conv1d;           /* (lin_key_dim, 1, conv_kernel) f32 */
    LZTensor kda_k_conv1d;           /* (lin_key_dim, 1, conv_kernel) f32 */
    LZTensor kda_v_conv1d;           /* (lin_value_dim, 1, conv_kernel) f32 */
    LZTensor kda_A_log;              /* (lin_n_v_heads) f32, PER HEAD (not widened) */
    LZTensor kda_f_a_proj;           /* (kda_gate_rank, hidden) Q8 */
    LZTensor kda_f_b_proj;           /* (lin_n_v_heads*lin_k_head_dim, kda_gate_rank) Q8 */
    LZTensor kda_dt_bias;            /* (lin_n_v_heads*lin_k_head_dim) f32, PER CHANNEL */
    LZTensor kda_b_proj;             /* (lin_n_v_heads, hidden) Q8 - same role as in_proj_b */
    LZTensor kda_g_proj;             /* (lin_value_dim, hidden) Q8 - same role as in_proj_z */
    LZTensor kda_o_norm;             /* (lin_v_head_dim) f32 - same role as ssm_norm, silu gate */
    LZTensor kda_o_proj;             /* (hidden, lin_value_dim) Q8 - same role as out_proj */

    /* FFN routing. 0 (default) = classic dense gate/up/down_proj above;
       1 = latent MoE below. Resolved once at load time from
       config.num_experts / config.first_k_dense_replace, not recomputed
       per token. */
    int ffn_moe;

    /* latent MoE (LatentMoE / KunMoEGate).
       Only bound when ffn_moe is set. Names match the reference:
       "confusingly named" per that file's own comment -
       routed_expert_down_proj maps hidden->latent,
       routed_expert_up_proj maps latent->hidden. */
    LZTensor moe_gate_w;             /* (num_experts, hidden) Q8 or f32 - router logits */
    LZTensor moe_gate_bias;          /* (num_experts) f32 - e_score_correction_bias, SELECTION only */
    LZTensor moe_down_proj;          /* (moe_latent_dim, hidden) Q8 - routed_expert_down_proj */
    LZTensor moe_up_proj;            /* (hidden, moe_latent_dim) Q8 - routed_expert_up_proj */
    LZTensor moe_latent_norm;        /* (moe_latent_dim) f32, only when moe_latent_use_norm */
    LZTensor moe_shared_gate;        /* (moe_shared_width, hidden) Q8, only when moe_shared_width > 0 */
    LZTensor moe_shared_up;          /* (moe_shared_width, hidden) Q8 */
    LZTensor moe_shared_down;        /* (hidden, moe_shared_width) Q8 */
    /* Per-expert triplet, config.num_experts entries each, allocated at
       load time (model.c) - NOT part of the fixed LZLayerSpec table
       because the count is a runtime config value, not a compile-time
       constant. Names: mlp.experts.<e>.w1/w2/w3.weight. w1 = gate,
       w2 = down, w3 = up (KdaExpert.forward: w2(act(w1(x)) * w3(x))). */
    LZTensor *moe_expert_w1;         /* [num_experts] (moe_intermediate_size, moe_latent_dim) */
    LZTensor *moe_expert_w2;         /* [num_experts] (moe_latent_dim, moe_intermediate_size) */
    LZTensor *moe_expert_w3;         /* [num_experts] (moe_intermediate_size, moe_latent_dim) */
} LZLayer;

/* The MTP / "next-N" head. One full_attention block plus four tensors.
   Names below are upstream's (Qwen/Qwen3.5-0.8B); llama.cpp calls the
   same four eh_proj / shared_head_norm / hnorm / enorm. */
typedef struct {
    LZLayer  blk;                    /* mtp.layers.0.* - a full_attention layer */
    LZTensor fc;                     /* mtp.fc (hidden, 2*hidden) */
    LZTensor norm;                   /* mtp.norm (hidden) - before the shared head */
    LZTensor pre_fc_norm_hidden;     /* (hidden) - on the body's hidden state */
    LZTensor pre_fc_norm_embedding;  /* (hidden) - on the next token's embedding */
} LZMtp;

typedef struct {
    LZModelConfig config;
    LZSafetensors st;
    char prefix[64];                /* tensor-name prefix, auto-detected */
    FILE *bin_file;                 /* non-NULL: stream weights from model.bin */

    LZTensor embed_tokens;          /* (vocab, hidden) Q8; doubles as lm_head when tied */
    LZTensor final_norm;            /* (hidden) f32 */
    LZLayer *layers;

    /* Multi-token-prediction head, or NULL when config.mtp_n_layers is 0
       (every model this project has produced so far - real trained MTP
       weights do not exist yet either).

       Inference: forward.c's
       lz_mtp_draft_step runs one draft step, lz_forward_verify batches
       the target model's own check of k drafted tokens, and
       generate.c's lz_spec_round ties draft -> verify -> accept/reject
       -> rollback into one round, wired into lz_generate_resume behind
       LZGenOpts.spec_k (0 = off - see that struct's own comment).
       Binding the head is what keeps the loader from silently ignoring
       20.45 MB of weights, the exact failure transformers performs on
       this family.

       Shape: the block is an ordinary full_attention layer. That is not
       an assumption, it is how upstream ships it (all 11 tensors match
       layers.<n> of the body one for one) and how llama.cpp models it -
       its GGUF exposes the head as blk.<n_layers>, i.e. one more entry
       in the same block namespace, plus the four extras below.

       The forward it computes:
           x = fc( concat( pre_fc_norm_embedding(emb[i+1]),
                           pre_fc_norm_hidden(h_body) ) )
           x = block(x)
           logits = lm_head( norm(x) )
       lm_head is the shared embedding: upstream's config says
       mtp_use_dedicated_embeddings=false, which is also why a draft step
       costs a full vocabulary projection and that cost cannot be
       designed away.

       CONCAT ORDER: embedding occupies the LOW half of fc's input,
       hidden the HIGH half - `torch.cat([inputs_embeds, hidden_states],
       dim=-1)`.
         vLLM   qwen3_next_mtp.py Qwen3NextMultiTokenPredictor.forward,
                `torch.cat([inputs_embeds, hidden_states], dim=-1)` then
                `self.fc(hidden_states)`.
         sglang qwen3_next_mtp.py Qwen3NextForCausalLMMTP.forward,
                `self.fc(torch.cat((input_embeds, hidden_states), dim=-1))`.
       Both agree with fold_mtp_fc's docstring
       (embed low / hidden high), which is why that function's rotation
       math is correct.

       CAVEAT, unresolved: transformers implements no MTP forward for
       this family at all - both modeling_qwen3_5.py and
       modeling_qwen3_5_moe.py only carry the
       _keys_to_ignore_on_load_unexpected drop list, nothing that computes
       the module (checked in both files). Tensor shapes matching
       upstream's names one for one is checked in this repo; the rest of
       the forward (block = one ordinary full_attention decoder layer,
       norm = plain RMSNorm before the shared lm_head) is inferred from
       the same two files and has not been run against a live reference
       output - only against the vLLM/sglang source text above, and
       against ITSELF
       (checks that the bin and safetensors loaders
       agree, and that draft-verify-rollback reproduces a manual
       sequential replay - both are self-consistency checks on RANDOM
       weights, not correctness checks against a trained model's real
       output; iron law four's distinction between the two). Reads as a
       weakness list, not a to-do list yet: no real MTP weights exist to
       validate against. */
    LZMtp *mtp;

    int weights_loaded;             /* 0 = metadata bound only, weights not read */
    long long n_params;             /* text-tower parameter count */
    long long n_params_skipped;     /* dropped parts (vision tower) */
    long long bytes_alloc;          /* bytes actually allocated */
} LZModel;

/* Parse config.json. Supports both the multimodal shell (fields under
   text_config) and plain-text layouts. */
int lz_load_config(LZModelConfig *c, const char *path,
                       char *errbuf, int errlen);

/* Open a model directory: read config.json, open safetensors, verify
   every expected tensor exists with the right shape. Reads no weight
   data, so time and memory are minimal. */
int lz_open(LZModel *m, const char *dir, char *errbuf, int errlen);

/* Read all text-tower weights as f32. Needs ~4x parameter-count bytes
   of memory. */
int lz_read_weights(LZModel *m, char *errbuf, int errlen);

void lz_free(LZModel *m);

const char *lz_layer_type_name(int t);

#endif
