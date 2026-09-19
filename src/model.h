#ifndef LZ_MODEL_H
#define LZ_MODEL_H

#include "lz_int.h"   /* <stdint.h> is not on the language floor */
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
                                separate q/k/v/g projections. Selected from
                                the checkpoint's own linear-layer tensor
                                names, not from any string in config.json -
                                see arch_probe in model.c. The JSON
                                layer_types string stays "linear_attention"
                                either way; this code is what it maps to. */

/* ffn_activation codes (config.json's "ffn_activation" string, text_config
   or top level). NONE is every model this engine has ever loaded; the
   other two are the training side's engine/kunmoe_modeling.py
   situ_glu/swiglu_clamp, added here with no shipping checkpoint yet -
   see lz_ffn_gate_up (forward.h) and model.h's ffn_activation field. */
#define LZ_FFN_ACT_NONE         0
#define LZ_FFN_ACT_SITU         1   /* Kimi K3 SiTU-GLU, section 2.3.2 Eq. 12 */
#define LZ_FFN_ACT_GPTOSS_CLAMP 2   /* GPT-OSS/DeepSeek-V4.1 clamped SwiGLU */

/* csa2_mode[] codes (config.json's "csa2_kv_mode" object, keyed by string
   layer index - see LZModelConfig's own csa2_mode/csa2_source comment).
   Report Sec2.3.1's Full/Reindex/Reuse cross-layer KV axis, mirroring
   the training side's KunMoeConfig.csa2_kv_mode string values exactly.
   FULL is 0 so a checkpoint whose config.json never carried either key
   - the whole domain before this field existed - zero-initializes into
   the right default without a separate fill loop. */
#define LZ_CSA2_FULL    0
#define LZ_CSA2_REINDEX 1
#define LZ_CSA2_REUSE   2

/* Weight format codes. Carried per tensor; mixed precision is simply
   different tensors having different codes. Adding a format needs: a
   code here, a dispatch branch in ops.c, and a name in export_q8.py's
   recipe table. */
#define LZ_FMT_F32   0      /* n f32s */
#define LZ_FMT_Q8_0  1      /* gs int8s + 1 f32 scale (symmetric; zero on the grid) */
#define LZ_FMT_Q4_1  2      /* gs 4-bit nibbles + 1 f32 scale + 1 f32 min (asymmetric) */
#define LZ_FMT_Q6_1  3      /* gs 6-bit (4-bit plane + 2-bit plane) + scale + min */
#define LZ_FMT_Q16_0 4      /* gs int16s + 1 f32 scale (symmetric) */
#define LZ_FMT_T2    5      /* ternary {-1,0,+1} as 2-bit codes + 1 f32 scale */
#define LZ_FMT_BF16  6      /* n bf16s in q (read as uint16); no scale, no groups */

/* LZ_FMT_BF16 IS STORAGE, NOT QUANTIZATION, and it is the only format
   here that loses nothing. A bf16 is the high half of an f32, so
   widening is `bits << 16` and the result is the very number the file
   holds - there is no scale to apply, no group to belong to, and no
   rounding either way.
 *
 * WHY IT EXISTS. Expanding a BF16 tensor to f32 while loading (the
 * safetensors.c read path) would turn a file storing two bytes per
 * parameter into four in RAM. On a 64MB target that is the difference
 * between a model fitting and not: the checkpoint this was measured
 * against reports "0.23 GB at 4 bytes/param". Keeping it narrow halves
 * both the footprint and the DDR traffic of reading it, and DDR is the
 * bottleneck on a 248MHz part.
 *
 * BIT-IDENTICAL TO EXPANDING AT LOAD TIME, which is what makes it safe
 * to have on by default. The widening is the same `<<16` the loader
 * performed, only deferred to the point of use; every value the kernels
 * see is unchanged. That is a different claim from the quantized formats
 * above, which trade accuracy for size on purpose - this one trades
 * nothing. Narrowing an F32 file INTO bf16 would be lossy and is a
 * separate decision, not this format.
 *
 * gs is 0 and scale/zero are NULL, so any code that reaches for a scale
 * must test the dtype first - lz_t_f32 does, ahead of its scale guard.
 *
 * F32 TENSORS THAT CANNOT BE GROUPED (non-2D, row < MIN_GS=32) are
 * now narrowed to BF16 at export time (export_q8 plan()'s FMT_BF16
 * branch). This covers all 1D norms and 3D conv1d tensors.
 * The loss is sub-0.1% relative on norm weights near 1.0 and conv
 * weights near 0.02; unmeasured on PPL but structurally negligible.
 * lz_t_f32/lz_t_row_f32 widen bf16->f32 at the point of use.
 *
 * Q8_0 is 1.8x smaller than bf16 AND has hand-written integer kernels,
 * where bf16 widens back to f32 to compute. For tensors that CAN be
 * grouped, Q8_0 dominates. BF16 fills the gap Q8_0 cannot reach. */

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
    /* Derived plane for the fixed-point dequant epilogue: `scale`
       requantized to int16 with one shared exponent per output row.
       Built once by lz_epi_prep(), NULL when the float epilogue runs.
       It has to be load-time work - quantizing per row inside the
       forward costs as much float arithmetic as the epilogue it
       replaces, which is what the first candidate measured. */
    int16_t     *sq;        /* n/gs int16 */
    signed char *sexp;      /* one exponent per row */
    /* Row stride of sq and zq, in int16: in_dim/gs, the group count of
       one output row. Stored because the epilogue indexes those planes
       once per output element and the alternative is deriving it there
       from the kernel's (sub-block count, sub-blocks per group) pair,
       which is an integer divide a million times a generation. It is
       not a cached copy of that quotient - it is the stride the planes
       were actually allocated with, so it is the primary value and the
       quotient was the derivation. */
    int          sq_row;
    /* The same decomposition of the `zero` plane, built by
       lz_epi_prep_zero(). Q4_1/Q6_1 only: T2 has no stored min - its
       zero coefficient is -scale, which `sq`/`sexp` above already
       carry - and Q8_0/Q16_0 have no zero term at all, so both leave
       these NULL. */
    int16_t     *zq;        /* n/gs int16 */
    signed char *zexp;      /* one exponent per row */
} LZTensor;

/* Q6_1's 2-bit plane layout: every 32 elements take 8 bytes; byte j's
   bits 0-1 are element j, bits 2-3 are j+8, bits 4-5 are j+16, bits
   6-7 are j+24.

   This grouping is chosen, not arbitrary: `pand 0x03` and `psrlw 2/4/6`
   yield exactly {0-7}, {8-15}, {16-23}, {24-31}, aligned group-for-group
   with the two 8-byte loads of the 4-bit plane - the activation side's
   pre-expanded registers are reused as-is, no reshuffling needed.

   Value q = lo + (hi<<4) in [0,63], dequantized w = q*d + m. The dot
   product splits exactly: dot(q,x) = dot(lo,x) + 16*dot(hi,x), two
   int32 accumulators merged afterwards, no precision loss (the hi term
   is 3*127*32 = 12k, far from the int32 limit). */

/* LZ_FMT_T2: ternary {-1,0,+1}, BIT-LAYOUT IDENTICAL TO Q6_1's 2-bit
   plane above - every 32 elements take 8 bytes, byte j's bits 0-1 are
   element j, bits 2-3 are j+8, bits 4-5 are j+16, bits 6-7 are j+24.
   Reusing that layout is the point: `pand 0x03` / `psrlw 2/4/6` and the
   32-element sub-block framing are already written and already
   bit-identical across the kernels and both compilers.

   Encoding follows llama.cpp's TQ2_0 (ggml-common.h block_tq2_0,
   quantize_row_tq2_0_ref): CODE = VALUE + 1, so {-1,0,+1} store as
   {0,1,2} and 3 never occurs. Read unsigned; the kernel needs no sign
   extension, exactly as for Q4_1's nibbles.

   `scale` holds one f32 per group; `zero` is NULL. Ternary is
   symmetric, so there is no min to carry - but the dot product still
   splits into two terms the way Q4_1's does, and for the same reason:

       sum(w*x) = d*sum((code-1)*x) = d*[ sum(code*x) - sum(x) ]

   The second term depends only on activations, so it is hoisted out of
   the row loop into xgref[] and shared by every output row. That is the
   same structure llama.cpp's AVX2 TQ2_0 kernel uses (arch/x86/quants.c:
   `_mm256_sub_epi16(sumi0, ysum)` against the pre-computed y->bsums),
   arrived at independently on both sides.

   What is not copied, and why: llama.cpp also ships TQ1_0 at 1.6875
   bpw, packing 5 elements per byte in base 3 (3^5 = 243 < 256). It is
   the denser format and it is unusable on the ARM926EJ-S target -
   unpacking needs division by 3 and that core has NO hardware divide.
   TQ2_0's 2 bits per element cost 0.375 bpw more and unpack with a
   shift and a mask.

   ALSO NOT COPIED: llama.cpp packs 128 elements per 32 bytes (stride
   32); this is stride 8, per the Q6_1 layout. The kernel STRUCTURE
   transfers between the two, the byte offsets do not. */

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
    /* ffn_activation (KunMoeConfig): replaces every FFN's act(gate)*up -
       dense CE (dense_ffn_step) and MoE (forward_moe's routed and shared
       experts) alike - with a two-tensor gated activation. LZ_FFN_ACT_NONE
       leaves lz_silu(gate)*up exactly as it is today, which is every
       model this engine has ever loaded; the other two values are ISA-gap
       work with no shipping checkpoint yet (mirrors the training side's
       situ_glu/swiglu_clamp, neither of which any trained checkpoint uses
       as of this field's addition). See lz_ffn_gate_up (forward.h). */
    int ffn_activation;                 /* LZ_FFN_ACT_* */
    float activation_situ_beta;         /* Kimi K3 SiTU-GLU beta1 (4.0) */
    int   ffn_has_situ_linear_beta;     /* config field present (not None) */
    float activation_situ_linear_beta;  /* Kimi K3 SiTU-GLU beta2 (25.0) */
    float swiglu_clamp_limit;           /* GPT-OSS/V4.1 clamp threshold (10.0) */
    float swiglu_clamp_alpha;           /* sigmoid-GELU approx constant (1.702) */
    int tie_word_embeddings;
    int attn_output_gate;           /* whether q_proj carries the gate (true in this model) */
    /* Carried and printed, NOT consulted. layer_types decides which
       layers are full attention, one entry per layer (model.c reads it
       and refuses to load without it). This field is the upstream
       generator's stride and stops describing the model once layers are
       pruned: kunkun-ce is 5 layers with layer_types
       [linear, linear, full, linear, full] while this still reads 4,
       which would place the single full layer at index 3. */
    int full_attention_interval;
    /* SubLN (BitNet b1.58): delete the two pre-layer norms and give each
       ternary projection its own input RMSNorm (see the kda_*_norm /
       gate_norm / up_norm / down_norm fields on LZLayer). Mirrors the
       training side's config.use_subn. 0 = ordinary pre-layer norms. */
    int use_subn;
    /* Block-diagonal Hadamard (online, applied to the SubLN activation
       just before quantization - never folded into a weight, so nothing
       in the tensor data records whether it was done). hadamard_o is
       kda_o_proj's block, hadamard_down is the dense FFN down_proj's.
       0 = no Hadamard; a positive value is the block size (a power of
       two dividing that projection's input dim). -1 on the wire means
       "derive" and exists only for products that predate these fields
       (bin < v8, config.json without the keys); load time resolves it
       into a concrete 0-or-block-size, so the forward pass only ever
       sees the two settled values. */
    int hadamard_o;
    int hadamard_down;
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
    /* config.json "mtp_n_draft"; 1 (the only shape upstream or any
       checkpoint before this field existed has) when the key is
       absent. >1 selects
       LZMtp's five-tensor DSpark shape (main_proj/depth_embed/
       mask_embed/markov_embed/markov_proj) in place of `fc` - see that
       struct's own comment. mtp_markov_rank is 0 (meaningless) at
       mtp_n_draft==1, and at >1 is read directly off markov_embed's own
       shape at load time rather than duplicated here as a second
       source of truth for the same number. */
    int mtp_n_draft;
    /* config.json "mtp_markov_rank"; meaningless (0) at mtp_n_draft<=1.
       A real, independently-declared config field rather than something
       inferred from markov_embed's own tensor shape at load time, same
       convention every other DK_* dimension in this file follows
       (declared in config.json, then checked against the tensor that
       ships) - so a mismatch between what the checkpoint's config
       claims and what its own weights actually are is a load error,
       not a silent reshape. engine/mtp_head.py's own __init__ derives
       this same number from vocab_size when not given explicitly
       (DSpark's r=256, scaled) - the derivation lives on the training
       side; this field is just where its RESULT is recorded for the
       engine to check against. */
    int mtp_markov_rank;

    /* CED (Causal Encoder-Decoder), DeepSeek-V4.1 report SS2.2/SS3.2.2.
       config.json "ced_split"/"ced_replay_w" - 0/0 (default) is off and
       must reproduce a model that never carried these keys bit for bit
       (same "off means off" contract mtp_n_draft/engram/use_subn all
       already have).

       ced_split: the encoder/decoder boundary layer index S (layers
       0..S-1 are the causal encoder, S..n_layers-1 the decoder). This
       engine's OWN full-attention half of CED (the decoder's global K/V
       projected from layer S-1's exit hidden, never windowed - Eq. 1)
       is a SEPARATE, already-shipped mechanism (`--ced`, forward.c's
       `s->ced`/`ced_skip_last`) that predates this field and does not
       read it; ced_split here is consumed ONLY by the KDA bounded-
       replay reset below. The two mechanisms are independent and may
       be on or off in any combination - see docs/dsv41-optimization-
       menu.md's E3 section for why one field cannot serve both (the
       full-attention half never gets windowed at all, only the KDA
       half does).

       ced_replay_w: the KDA "SWA Bounded Replay" analog (Kunkun has no
       SWA - see the same doc section for why a KDA layer plays that
       role here). For KDA layers at index >= ced_split,
       lz_forward_batch resets the recurrent/conv state to empty at an
       offset within THAT CALL's own token span (see its own comment
       for why it is call-relative, not a global prompt position),
       matching the training-side simulation already shipped
       (engine/kunmoe_modeling.py's _kda_packing_hook/ced_replay_w - a
       real, already-trained-against mechanism, not a new invention).
       .prof/ced_replay_w_probe.py drives that same training-side
       mechanism directly and isolates the reset's own cost from CED's
       own (separate, already-accepted) full-attention approximation
       cost - see docs/dsv41-optimization-menu.md's E3 section for the
       measured numbers and why the isolation matters. */
    int ced_split;
    int ced_replay_w;

    /* ced_kv: the OTHER half of report Sec2.2 Eq(1) - whether a
       decoder-region full-attention layer's K/V projections read h_S
       (the encoder's own exit hidden state, captured once per chunk at
       layer ced_split) instead of that layer's own current s->xb.
       config.json "ced_kv": "off" (default, 0) reproduces a
       config.json that never carried the key; "proj" (1) turns the
       substitution on while every decoder layer keeps its own,
       independently-trained k_proj/v_proj (forward.c's kv_in
       resolution, forward.h's s->ced_hs - the same "capture one
       layer's s->xb, feed it to other layers' K/V input" shape
       attn_kv_share already uses, with a different source rule).

       "reuse" (DeepSeek-V4.1's OTHER CED arm - decoder layers sharing
       the SOURCE layer's k_proj/v_proj Parameter objects outright,
       the same aliasing attn_kv_share already does, rather than
       sharing only the projection's input) has no engine-side
       implementation and is refused at load (LZ_ERR_CFG_CED_KV)
       rather than silently treated as off - a checkpoint that says
       "reuse" and gets "off" would run and produce confidently wrong
       numbers with no error. */
    int ced_kv;

    /* S1, attn_kv_share (DeepSeek-V4.1 report SS2.3.1's "Reuse" mode).
       config.json "attn_kv_share" (a JSON true/false) and the v10
       model.bin header tail. 0 - the default, and the only value a bin
       file written before v10 can carry - is off and reproduces a model
       that never set the key bit for bit.

       THE SOURCE is the FIRST full_attention layer of the stack, the
       TARGETS are every full_attention layer after it, and fewer than
       two full-attention layers makes the key a no-op (config.json's own
       rule, engine/kunmoe_modeling.py's _wire_attn_kv_share).

       The training side hands each target the source attention's
       k_proj/v_proj/k_norm MODULES, and the input RMSNorms SubLN carries
       inside them, so save_pretrained emits those five tensors for the
       source layer only - a target layer's copy is absent from the
       checkpoint rather than equal to the source's. model_walk therefore
       skips the five specs on a target layer and apply_kv_share gives it
       the source layer's LZTensors, which makes them SHARED STORAGE on a
       target layer: lz_free frees them once, from the source.

       WHAT IS SHARED IS THE WEIGHTS AND THE INPUT. The training side also
       swaps the INPUT (`_kv_swap_input` feeds the source attention's
       captured input to the shared projections for a target layer's call
       too), so a trained target layer's K and V ARE the source layer's
       own rather than its own hidden state re-projected. forward.c
       captures the source's attention input into LZRunState.kvshare_x
       per chunk and hands it to every target layer's k/v projections
       (forward.h's own comment on that buffer); Q still reads the target
       layer's own input, because only the two K/V projections are fed
       the source's in the trained model. */
    int attn_kv_share;

    /* S4, MLA-style low-rank Q/O (DeepSeek-V4.1 report SS2.1). config.json
       "mla_q_lora_rank"/"mla_o_lora_rank" - 0/0 (default, matching
       engine/kunmoe_modeling.py's LowRankLinear/_wire_mla's own None/None
       default) is off and reproduces a model that never carried these
       keys bit for bit.

       Global, not per-layer: _wire_mla replaces q_proj/o_proj on EVERY
       full-attention BODY layer uniformly when set (never the MTP
       head's own attention, which _wire_mla's loop over
       self.model.layers never reaches). LZLayer's own mla_q_on/mla_o_on
       are the per-layer flags setup_layers derives from these two
       fields for exactly that reason: the MTP block's LZLayer
       (m->mtp->blk) is calloc'd separately and setup_layers never
       touches it, so its mla_q_on/mla_o_on stay 0 - reading these two
       config fields directly in forward_attn.c's attn_impl would get
       that wrong the one time attn_impl runs against the MTP block
       (lz_debug_mtp_attn_t1_slow's slow-path call into forward_attn).

       Mutually exclusive with use_subn, same reason kunmoe_modeling.py's
       _wire_mla asserts against it: LowRankLinear does not compose with
       either BitLinear or SubLN's per-projection input norm. Refused at
       load time (LZ_ERR_CFG_MLA_SUBN) rather than silently picking one,
       same policy as the rope_theta top-level-vs-nested collision
       check. */
    int mla_q_lora_rank;
    int mla_o_lora_rank;

    /* S2, DeepSeek-V4.1 report SS2.3 (CSA2 - Compressed Sparse
       Attention). config.json "csa2_layers" (a JSON array of layer
       indices, each of which must be full_attention) - absent/empty is
       off and reproduces a model that never carried the key bit for
       bit. Formulas cited and verified against two primary sources
       (arXiv:2512.02556 indexer, arXiv:2606.19348 compression) - see
       engine/csa2.py's own module docstring (training-side reference)
       and docs/dsv41-optimization-menu.md's CSA2 section for the
       citation trail and the deliberate simplifications this engine
       side does not reproduce (no cross-layer sharing - S1's
       attn_kv_share already covers a simpler version of that axis; no
       layered indexer; no sliding-window branch; no attention-sink
       integration; no grouped output projection; no composition with
       S4's shared low-rank latent - unlike the training-side reference,
       which does support that composition, this engine side always
       expects its OWN dedicated csa2_w_dq_idx tensor, so a checkpoint
       trained with S4+CSA2 sharing the latent on the SAME layer is
       refused by a missing-tensor error here, not silently
       misinterpreted).

       PER-LAYER, not global like mla_q_lora_rank/mla_o_lora_rank above:
       csa2_layers may name a SUBSET of the full-attention layers (S1's
       attn_kv_share, the other CSA2-derived mechanism, is also
       per-layer for the same reason). csa2_on[] holds which layer
       indices were named (parsed once at config-load time); LZLayer's
       own csa2_on flag (setup_layers-derived, same pattern as
       mla_q_on/mla_o_on) is what forward.c and model_walk actually
       read - see LZLayer's own comment for why the per-layer flag,
       not a lookup into this array, is the thing every consumer reads.

       The rest are this project's own sizing (32 indexer heads/head_dim
       128/top-k 512 are V4.1's numbers for a 40-layer, hidden=5120
       model) - same posture mla_q_lora_rank's own comment documents,
       and must match engine/csa2.py's identically-named config fields
       exactly since a checkpoint's tensor shapes are fixed at export
       time by whatever those fields were during training. */
    int csa2_on[LZ_MAX_LAYERS];
    int csa2_compress_m;
    int csa2_topk;
    int csa2_indexer_heads;
    int csa2_indexer_head_dim;
    int csa2_indexer_q_lora_rank;

    /* Report Sec2.3.1's Full/Reindex/Reuse cross-layer KV reuse axis -
       the engine-side schema half. forward_state.c's cache_idx fill
       aliases a REINDEX/REUSE layer's own physical slot onto its
       csa2_source's (a different task from this pair's own parsing),
       and forward_csa2.c's own mode dispatch is what actually gives
       FULL/REINDEX/REUSE their three distinct behaviors - see that
       file's own header comment for the six-step table. Mirrors the
       training side's KunMoeConfig.csa2_kv_mode / csa2_kv_source (a
       different repository - engine/csa2.py's three-mode forward
       landed there first) field for field: config.json objects keyed
       by STRING layer index (JSON has no integer keys), e.g.
       csa2_kv_mode: {"2": "reindex"}, csa2_kv_source: {"2": 0}.

       csa2_mode[i] is one of the LZ_CSA2_* codes above, csa2_on[i]
       whether or not - FULL (the default, need not be named) needs no
       source; REINDEX/REUSE both name a STRICTLY EARLIER member of
       csa2_layers in csa2_source[i]. That per-layer shape is necessary
       but not sufficient: the actual requirement is that csa2_source[i]
       be the CURRENT LAST WRITER of the physical slot it aliases, not
       merely any strictly-earlier csa2_on layer - lz_csa2_last_writer_
       valid (model.c) is the whole-config check this needs (its own
       comment has the concrete counter-example: a REINDEX layer chained
       off a FULL layer, then a REUSE layer naming that same FULL layer
       as ITS source too, skipping over the REINDEX layer that actually
       overwrote the shared slot last). "A REUSE layer's source must
       itself be FULL or REINDEX, never REUSE" - this project's OLDER,
       narrower rule - is a special case that check subsumes (a REUSE
       layer never becomes anyone's last writer, so naming one as a
       source always fails the same way). csa2_source[i] is -1 where
       csa2_mode[i] is FULL, which is every layer's default - see
       model.c's lz_load_config for the exact validation, which mirrors
       the training side's own KunMoeConfig.__post_init__ check for this
       pair. Also refused there (and by lz_open_bin's own mirror check):
       csa2_on[i] together with mla_q_lora_rank > 0 - forward_csa2.c's
       step 1 always reads q_proj directly and never S4's low-rank
       q_proj_down/up pair, so the two compose only on paper. */
    int csa2_mode[LZ_MAX_LAYERS];
    int csa2_source[LZ_MAX_LAYERS];

    /* Learned per-head attention sink (GptOss-style; the training
       side's _wire_attn_sink/_sink_eager_attention_forward). config.json
       "attn_sink" is a JSON true/false, read as a BOOL like
       attn_kv_share below - lz_json_get_int answers its default for a
       LZ_JSON_BOOL node and the switch would never fire. 0 is off and
       reproduces a model that never carried the key bit for bit.

       Named "_learned" because LZRunState already has an `attn_sink`:
       StreamingLLM's keep-the-first-N-prefix window (forward.h), an
       unrelated mechanism that shares neither the field nor the
       arithmetic.

       Only full_attention layers get one, and _wire_attn_sink skips the
       csa2_layers (CSA2 has no sink integration), so LZLayer derives its
       own per-layer flag rather than reading this - same pattern as
       csa2_on below. */
    int attn_sink_learned;

    /* config.json "attn_sink_init": the constant the training side fills
       the sinks parameter with at construction (_wire_attn_sink). The
       forward never reads it - what the model computes with is the
       trained tensor, which arrives in the checkpoint. Carried so the
       engine's copy of the config is the checkpoint's, not a subset of
       it, and so a diagnostic can say what "untrained" looks like for
       the switch: kunkun98's sinks are still exactly this value on every
       head, which is why its sink column carries no softmax mass.

       NOT in the .bin header: Task 1's v5 tail carries attn_sink_learned
       only, so the bin path leaves this field at lz_open's memset zero
       while the config.json path reads the checkpoint's value. Nothing
       reads the field, which is the only reason that divergence is
       invisible; whoever makes it load-bearing has to close it first,
       either by growing the header or by dropping the field. */
    float attn_sink_init;

    /* GR (gated residual), the branch count nr. 0 is off and reproduces
       a model that never carried the key bit for bit. 1 is not a
       configuration: GatedResidual4 raises below nr=2, because the
       training side has no single-stream GR - the control arm is the
       switch off, not a degenerate nr=1. Both loaders therefore read
       this as "0, or >= 2" - lz_load_config from config.json
       "gated_residual", lz_open_bin from the v5 tail (range-checked
       there, LZ_ERR_BIN_FIELD_RANGE) - and setup_layers derives the
       per-layer LZLayer.gr_on flag from the field, never a second copy
       of the rule.

       Its span endpoint is ced_split above: GR covers layers
       [0, ced_split) and nothing else, because the CED boundary has to
       see an already-collapsed [B,S,D] stream. */
    int gated_residual;

    /* RoPE. Note rope_theta hides inside the nested rope_parameters;
       the top level does not expose it. */
    float rope_theta;
    float partial_rotary_factor;    /* 0.25 in this model */

    /* rope_parameters.rope_type (Qwen3.5/transformers config field,
       flat - a sibling of rope_theta inside rope_parameters, not
       nested under a "rope_scaling" sub-object; absent = "default" = 0).
       type 0 default / 1 linear / 2 yarn (NTK-aware YaRN interpolation).
       factor is the interpolation factor (e.g. 8 for 4096 -> 32768);
       orig_max is original_max_position_embeddings (yarn only);
       beta_fast/beta_slow are yarn's correction-range bounds
       (HF defaults 32.0 / 1.0). attn_factor is yarn's post-scale on
       cos/sin (HF's `attention_factor`); 0 means absent, in which case
       forward_rope.c derives the standard single-mscale default rather than
       reading a DeepSeek-style mscale/mscale_all_dim pair, which this
       project does not parse. The per-position cos/sin table in
       forward_rope.c applies these when it builds rope_cs. */
    int   rope_scaling_type;
    float rope_scaling_factor;
    float rope_scaling_orig_max;
    float rope_scaling_beta_fast;
    float rope_scaling_beta_slow;
    float rope_scaling_attn_factor;

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

    /* Engram (conditional n-gram memory). 0 = no engram module.
       The fields mirror KunMoeConfig's engram_* keys; n_orders,
       n_spaces, head_space, per_order, conv_dim, total_rows and
       primes/offsets are derived at load time. */
    int engram;             /* 0 off / 1 on */
    int engram_ngram;       /* max n-gram order (default 3 -> orders {2,3}) */
    int engram_n_head;      /* hash heads per order */
    int engram_dim;         /* embedding dimension per head */
    int engram_size;        /* hash_budget: total table rows (engram_size in config) */
    /* Derived at load time (model.c), never stored in checkpoint: */
    int engram_n_orders;    /* ngram - 1 */
    int engram_n_spaces;    /* n_orders * n_head */
    int engram_head_space;  /* max(256, hash_budget // (n_spaces * engram_dim)) */
    int engram_per_order;   /* n_head * engram_dim: key/value proj input width */
    int engram_conv_dim;    /* hidden_size: depthwise conv channels */
    int engram_total_rows;  /* sum(primes[0..n_spaces-1]) */
    int engram_conv_kernel; /* conv kernel size (default 4) */
    int engram_conv_dilation; /* ngram: conv dilation */
    /* format_ids as a bitmap, vocab_size bits, one bit per token id.
       kunmoe_modeling.py's EngramMemory._content_mask does
       `torch.isin(input_ids, format_ids)`; this is the same
       membership test, O(1) per token instead of an O(|format_ids|)
       scan. NULL when engram_format_ids is empty or absent in
       config.json (same meaning as kunmoe_modeling.py's
       `if not self._format_ids: return None` - no format gating). */
    unsigned char *engram_format_bitmap;   /* [(vocab_size+7)/8] bytes */
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

    /* SubLN (use_subn=1 only; bound when config.use_subn is set):
       per-projection input RMSNorm for the classic dense FFN's ternary
       projections. Each norm is applied to that projection's OWN input in
       the forward (gate/up_proj: the hidden state; down_proj: the
       intermediate activation), replacing the two pre-layer norms. */
    LZTensor gate_norm;              /* (hidden) f32 - gate_proj input RMSNorm */
    LZTensor up_norm;                /* (hidden) f32 - up_proj input RMSNorm */
    LZTensor down_norm;              /* (intermediate) f32 - down_proj input RMSNorm */

    /* full_attention only */
    LZTensor q_proj;                 /* (attn_qgate_dim, hidden) Q8 - present iff !mla_q_on */
    LZTensor k_proj;                 /* (attn_kv_dim, hidden) Q8 */
    LZTensor v_proj;                 /* (attn_kv_dim, hidden) Q8 */
    LZTensor o_proj;                 /* (hidden, attn_q_dim) Q8 - present iff !mla_o_on */
    LZTensor q_norm;                 /* (head_dim) f32 */
    LZTensor k_norm;                 /* (head_dim) f32 */

    /* S4, MLA low-rank Q/O (model.h's own mla_q_lora_rank/mla_o_lora_rank
       comment on LZModelConfig). Present iff mla_q_on/mla_o_on - the
       PER-LAYER flags setup_layers derives from those two global config
       fields, not the fields themselves (see that comment for why the
       distinction matters for the MTP block's own LZLayer). q_proj_down
       (mla_q_lora_rank, hidden) feeds q_proj_up (attn_qgate_dim,
       mla_q_lora_rank) in place of q_proj; o_proj_down
       (mla_o_lora_rank, attn_q_dim) feeds o_proj_up (hidden,
       mla_o_lora_rank) in place of o_proj. Named to match
       engine/kunmoe_modeling.py's LowRankLinear.down/.up exactly, one
       level up (q_proj.down/q_proj.up as state-dict keys, q_proj_down/
       q_proj_up as C fields - a "." is not a legal C identifier
       character). */
    LZTensor q_proj_down, q_proj_up;
    LZTensor o_proj_down, o_proj_up;
    int mla_q_on, mla_o_on;          /* setup_layers-derived, see above */

    /* S2, CSA2 (model.h's own csa2_on[]/csa2_compress_m comment on
       LZModelConfig). Present iff csa2_on (setup_layers-derived, same
       per-layer-flag pattern as mla_q_on/mla_o_on) - replaces this
       layer's ENTIRE attention forward (forward_csa2.c), not just
       q_proj/o_proj, so q_proj/k_proj/v_proj/o_proj/q_norm/k_norm above
       stay bound and ARE still used (forward_csa2 reads the main
       query through q_proj/q_norm exactly like ordinary full attention;
       only K/V are replaced by the compressed entries below).
         csa2_w_kv, csa2_w_gate       (head_dim, hidden) each - main
                                       compression (Eq 9-12 simplified)
         csa2_w_kv_idx, csa2_w_gate_idx (csa2_indexer_head_dim, hidden)
                                       each - indexer key compression,
                                       same algorithm, own weights
         csa2_w_weights               (csa2_indexer_heads, hidden) -
                                       Eq 1's per-head w_{t,h}
         csa2_w_dq_idx                (csa2_indexer_q_lora_rank, hidden)
                                       - indexer query down-projection.
                                       ALWAYS present on this engine side
                                       (unlike the training-side
                                       reference, no S4-shared-latent
                                       composition - see
                                       LZModelConfig's own comment)
         csa2_w_uq_idx                (csa2_indexer_heads *
                                       csa2_indexer_head_dim,
                                       csa2_indexer_q_lora_rank) -
                                       indexer query up-projection */
    LZTensor csa2_w_kv, csa2_w_gate;
    LZTensor csa2_w_kv_idx, csa2_w_gate_idx;
    LZTensor csa2_w_weights;
    LZTensor csa2_w_dq_idx, csa2_w_uq_idx;
    int csa2_on;                     /* setup_layers-derived, see above */
    int csa2_mode;                   /* setup_layers-derived from
                                         LZModelConfig.csa2_mode[] above,
                                         same pattern as csa2_on -
                                         forward_csa2.c's own mode
                                         dispatch reads this field to
                                         pick FULL/REINDEX/REUSE's three
                                         distinct behaviors */

    /* SubLN (use_subn=1 only): full attention's per-projection INPUT
       RMSNorms. Distinct from q_norm/k_norm above, which are Qwen3.5's
       per-head QK norms over head_dim and are present at every use_subn
       setting. These normalize what the projection reads: the residual
       for q/k/v, the attention value output for o_proj. */
    LZTensor attn_q_subn_norm;       /* (hidden) f32 - q_proj input */
    LZTensor attn_k_subn_norm;       /* (hidden) f32 - k_proj input */
    LZTensor attn_v_subn_norm;       /* (hidden) f32 - v_proj input */
    LZTensor attn_o_subn_norm;       /* (attn_q_dim) f32 - o_proj input */

    /* Learned per-head attention sink (LZModelConfig's own
       attn_sink_learned comment has the mechanism and the training-side
       reference). (n_heads) - one logit per head, broadcast over every
       query position, concatenated to the logits BEFORE softmax and
       dropped AFTER it.

       attn_sink_on is setup_layers-derived and is the flag the forward
       reads, never c->attn_sink_learned - same per-layer pattern as
       csa2_on below, and for the same reason: _wire_attn_sink skips the
       csa2_layers, so the two full-attention layers of a shipping stack
       need not agree. It also gates whether model_walk claims this
       tensor at all, so a layer with the flag clear must not carry one -
       the leak check rejects a checkpoint whose file does.
       (m->mtp->blk never passes through setup_layers, so the MTP block's
       own copy of this field stays at its calloc 0.) */
    LZTensor attn_sinks;
    int attn_sink_on;

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

    /* SubLN (use_subn=1 only; bound when config.use_subn is set):
       per-projection input RMSNorm for the ternary KDA projections. Each
       norm is applied to that projection's OWN input in the forward
       (q/k/v/g: the hidden state; o_proj: the value output, lin_value_dim
       wide). kda_o_subn_norm is DISTINCT from kda_o_norm above - that one
       is the gated-silu norm over one value head, this one is o_proj's
       plain input RMSNorm over the whole lin_value_dim span. */
    LZTensor kda_q_norm;             /* (hidden) f32 - q_proj input RMSNorm */
    LZTensor kda_k_norm;             /* (hidden) f32 - k_proj input RMSNorm */
    LZTensor kda_v_norm;             /* (hidden) f32 - v_proj input RMSNorm */
    LZTensor kda_g_norm;             /* (hidden) f32 - g_proj input RMSNorm */
    LZTensor kda_o_subn_norm;        /* (lin_value_dim) f32 - o_proj input RMSNorm */

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

    /* SubLN (use_subn=1 only): the latent-MoE block's per-projection
       input RMSNorms. Under SubLN the block reads the RAW residual - the
       pre-layer norm is gone - so each projection normalizes its own
       input. The ROUTER is included on purpose: its logits go through a
       softmax, so the input magnitude is a temperature, and leaving it
       bare changes every routing decision rather than merely skipping a
       normalization. */
    LZTensor moe_gate_norm;          /* (hidden) f32 - router input */
    LZTensor moe_down_norm;          /* (hidden) f32 - routed_expert_down_proj input */
    LZTensor moe_up_norm;            /* (moe_latent_dim) f32 - routed_expert_up_proj input */
    LZTensor moe_shared_gate_norm;   /* (hidden) f32, only when moe_shared_width > 0 */
    LZTensor moe_shared_up_norm;     /* (hidden) f32 */
    LZTensor moe_shared_down_norm;   /* (moe_shared_width) f32 */
    /* Per-expert triplet, config.num_experts entries each, allocated at
       load time (model.c) - NOT part of the fixed LZLayerSpec table
       because the count is a runtime config value, not a compile-time
       constant. Names: mlp.experts.<e>.w1/w2/w3.weight. w1 = gate,
       w2 = down, w3 = up (KdaExpert.forward: w2(act(w1(x)) * w3(x))). */
    LZTensor *moe_expert_w1;         /* [num_experts] (moe_intermediate_size, moe_latent_dim) */
    LZTensor *moe_expert_w2;         /* [num_experts] (moe_latent_dim, moe_intermediate_size) */
    LZTensor *moe_expert_w3;         /* [num_experts] (moe_intermediate_size, moe_latent_dim) */

    /* GR (gated residual, LZModelConfig's own gated_residual comment has
       the mechanism and the span rule). One block per SUBLAYER - the
       attention and the MLP each get their own read/write gates, which is
       what the training side's two `GatedResidual4` attributes on a
       GrDecoderLayer are.

       names, shapes (nr = gated_residual, r = max(1, hidden/8),
       flat = nr * hidden):
         gr_attn.norms.<b>.weight  (hidden)     - branch b's read RMSNorm
         gr_attn.wd.weight         (r, flat)    - Eq. 31 W_d
         gr_attn.wu.weight         (flat, r)    - Eq. 31 W_u
         gr_attn.ww.weight         (nr, flat)   - Eq. 33 W_w
       and the same four again under gr_mlp. The `.weight` suffixes and the
       grouping match the checkpoint's own state-dict keys exactly, because
       they ARE those keys: the training side registers wd/wu/ww as
       nn.Linear and norms as an nn.ModuleList, so save_pretrained emits
       `<half>.norms.<b>.weight` then `<half>.wd/wu/ww.weight` in that
       order.

       gr_attn_norms/gr_mlp_norms are LZTensor ARRAYS allocated by
       setup_layers, the same posture as moe_expert_w1 above: the count nr
       is a runtime config value that no fixed offsetof table can index.
       Both are NULL on a layer with gr_on clear.

       gr_on is setup_layers-derived and is the flag the forward and
       model_walk read, never c->gated_residual - same per-layer pattern as
       csa2_on above, and for the same reason: GR covers only the encoder
       half [0, ced_split), so a stack's layers do not agree.
       (m->mtp->blk never passes through setup_layers, so the MTP block's
       own copies of these three flags stay at their calloc 0.) */
    LZTensor gr_attn_wd, gr_attn_wu, gr_attn_ww;
    LZTensor gr_mlp_wd, gr_mlp_wu, gr_mlp_ww;
    LZTensor *gr_attn_norms;         /* [gated_residual] (hidden) f32 */
    LZTensor *gr_mlp_norms;          /* [gated_residual] (hidden) f32 */
    int gr_on;                       /* layer carries GR branches at all */
    int gr_first;                    /* expands [B,S,D] into nr branches */
    int gr_last;                     /* collapses back to [B,S,D] (mean) */
} LZLayer;

/* The MTP / "next-N" head. One full_attention block plus four tensors.
   Names below are upstream's (Qwen/Qwen3.5-0.8B); llama.cpp calls the
   same four eh_proj / shared_head_norm / hnorm / enorm.

   fc (config.mtp_n_draft == 1, the only shape upstream ever ships) and
   the five DSpark tensors below (config.mtp_n_draft > 1) are MUTUALLY
   EXCLUSIVE, matching engine/mtp_head.py's MtpHead.__init__ exactly -
   never both bound on the same checkpoint. See engine/mtp_head.py's own
   __init__ docstring for the DSpark derivation (arXiv:2607.05147) this
   five-tensor shape implements: main_proj/depth_embed/mask_embed give
   the K-slot parallel base stage (lz_mtp_draft_batch); markov_embed/
   markov_proj are the cheap one-token-lookback logit correction
   (lz_mtp_markov_correct) - two SEPARATE functions because the base
   stage has no cross-slot dependency and the Markov correction does
   (needs the PREVIOUS slot's already-decided draft token), so they
   cannot be one call the way fc's single-slot path is. */
typedef struct {
    LZLayer  blk;                    /* mtp.layers.0.* - a full_attention layer */
    LZTensor fc;                     /* mtp.fc (hidden, 2*hidden) - mtp_n_draft==1 only */
    LZTensor norm;                   /* mtp.norm (hidden) - before the shared head */
    LZTensor pre_fc_norm_hidden;     /* (hidden) - on the body's hidden state */
    LZTensor pre_fc_norm_embedding;  /* (hidden) - on the next token's embedding */

    /* mtp_n_draft > 1 only (DSpark, arXiv:2607.05147). */
    LZTensor main_proj;              /* mtp.main_proj (hidden, 3*hidden) */
    LZTensor depth_embed;            /* mtp.depth_embed (mtp_n_draft, hidden) */
    LZTensor mask_embed;             /* mtp.mask_embed (hidden) - one shared row,
                                        not a table: every slot i>=1 reads the
                                        SAME vector, see mtp_head.py's forward */
    LZTensor markov_embed;           /* mtp.markov_embed (vocab_size, markov_rank) */
    LZTensor markov_proj;            /* mtp.markov_proj (vocab_size, markov_rank) */
} LZMtp;

/* Engram: conditional n-gram memory. Global module (not per-layer),
   injected after embed_tokens, before the layer loop. Tensors are
   allocated at load time by model.c from config.engram_* fields.

   Tensors and their PyTorch names (kunmoe_modeling.py):
     table      enggram_memory.table        (total_rows, engram_dim) Q8
     key_proj   enggram_memory.key_proj[o]   (hidden, per_order) Q8, x n_orders
     value_proj enggram_memory.value_proj[o] (hidden, per_order) Q8, x n_orders
     conv       enggram_memory.conv         (hidden, 1, kernel) f32, depthwise
     norm       enggram_memory.norm         (hidden) f32

   Derived arrays (primes, offsets) live on the struct, not in the
   checkpoint: they are functions of config, same as KDA's primes. */
typedef struct {
    LZTensor table;          /* (total_rows, engram_dim) Q8 */
    LZTensor *key_proj;      /* [n_orders] (hidden, per_order) Q8 */
    LZTensor *value_proj;    /* [n_orders] (hidden, per_order) Q8 */
    LZTensor conv;           /* (hidden, 1, conv_kernel) f32, depthwise */
    LZTensor norm;           /* (hidden) f32 */
    /* Derived, allocated at load time: */
    int *primes;             /* [n_spaces] */
    int *offsets;            /* [n_spaces] cumulative row offsets */
    /* Per-order rolling hash bases: base[o] = 2*((o+1)*7+1)+1 */
    lz_i64 *bases;           /* [n_orders] */
} LZEngram;

typedef struct {
    LZModelConfig config;
    LZSafetensors st;
    char prefix[64];                /* tensor-name prefix, auto-detected */
    FILE *bin_file;                 /* non-NULL: stream weights from model.bin */

    /* Streaming reader abstraction: every model.bin field/tensor read goes
       through rd(rd_ctx, dst, sz). Two implementations: file_reader over
       bin_file, and lz4d_read over a model.bin.lz4 stream (stunt). Returns
       bytes actually read; the callers treat anything short of `sz` as a
       read error. */
    size_t (*rd)(void *ctx, void *dst, size_t sz);
    void *rd_ctx;

    LZTensor embed_tokens;          /* (vocab, hidden) Q8; doubles as lm_head when tied */
    LZTensor final_norm;            /* (hidden) f32 */
    LZLayer *layers;

    /* Multi-token-prediction head, or NULL when config.mtp_n_layers is 0
       (every model this project has produced so far - real trained MTP
       weights do not exist yet either).

       Inference: forward_mtp.c's
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
       output; a consistency check and a correctness check are not the
       same claim). Reads as a
       weakness list, not a to-do list yet: no real MTP weights exist to
       validate against. */
    LZMtp *mtp;

    /* Engram conditional n-gram memory, or NULL when config.engram is 0.
       Global module: injected after embed_tokens, before the layer loop. */
    LZEngram *engram;

    int weights_loaded;             /* 0 = metadata bound only, weights not read */
    /* lz_i64, not long long. On every toolchain that
       exists today this IS long long; on Visual C++ 4.0 it is __int64,
       which is the point. safetensors.h supplies it (included above)
       and n_elem beside these already uses it. */
    lz_i64 n_params;                /* text-tower parameter count */
    lz_i64 n_params_skipped;        /* dropped parts (vision tower) */
    lz_i64 bytes_alloc;             /* bytes actually allocated */
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

/* S1 attn_kv_share's layer rule, from the config alone (the field's own
   comment above): lz_kv_share_src is the index of the layer whose K/V the
   others reuse, or -1 when nothing is shared; lz_kv_share_target says
   whether layer li takes them. Both loaders and forward.c read these, so
   the rule is written once - the loader for which tensors to expect, the
   forward pass for whose input reaches a target layer's K/V projections
   (forward.c's kvshare_x). */
int lz_kv_share_src(const LZModelConfig *c);
int lz_kv_share_target(const LZModelConfig *c, int li);

/* ced_kv's layer rule (model.h's own field comment above): 1 when
   layer li is in CED's decoder region and its K/V projections should
   read h_S (forward.c's s->ced_hs) instead of its own s->xb. True for
   ced_split itself (a no-op there, model.c's comment on the caller
   side has the argument) through n_layers-1. False whenever ced_kv is
   off or ced_split does not describe a real boundary (0 or n_layers -
   the same domain lz_kv_share_target and lz_gr_span already use), so
   forward.c's `li == c->ced_split` capture test can reuse this same
   function rather than repeating the "is CED active" condition. */
int lz_ced_kv_target(const LZModelConfig *c, int li);

/* Whether layer li's csa2_mode/csa2_source are internally consistent -
   csa2_mode[li] one of the three LZ_CSA2_* codes, and, for a non-FULL
   mode, csa2_source[li] a strictly earlier layer that is itself
   csa2_on AND full_attention. The same rule lz_load_config's own
   csa2_kv_mode/csa2_kv_source cross-checks already apply per JSON
   entry - extracted here (same shape as lz_kv_share_target/
   lz_ced_kv_target above: a pure predicate, no errbuf) so lz_open_bin's
   header validation can apply the identical rule to a raw field
   instead of trusting it, and so a probe can drive it directly without
   constructing a whole model.bin.

   li must be < c->n_layers - the padding region past it (still inside
   the fixed LZ_MAX_LAYERS arrays) is a different check (must be
   all-default, not "internally consistent"), which lz_open_bin's own
   padding loop already applies and this function does not repeat. */
int lz_csa2_mode_source_valid(const LZModelConfig *c, int li);

/* Whether every csa2_on layer's declared csa2_source is exactly its
   alias root's CURRENT LAST WRITER, not merely "a strictly earlier,
   csa2_on, full_attention layer" (lz_csa2_mode_source_valid's own,
   weaker, per-layer question). Cross-layer - unlike every other
   predicate on this page, it walks the whole config, in increasing
   layer order, tracking who most recently wrote each shared physical
   ci slot - because aliasing runs by PHYSICAL SLOT (forward_state.c's
   cache_idx fill), and two layers can share one without either being
   the other's own declared source (a FULL layer and every REINDEX
   layer chained off it, directly or transitively, all alias the same
   root). model.c's own comment on this function has the full argument
   and the concrete example (source=0 valid per-layer, still wrong once
   an intervening REINDEX layer at the same root has overwritten the
   slot layer 0 originally wrote).

   REQUIRES every layer to have already passed lz_csa2_mode_source_valid
   - both callers (lz_load_config, lz_open_bin) apply that check first,
   over the whole layer range, before this one; this function's own
   root-finding walk is only guaranteed to terminate under that
   precondition.

   Returns 1 if the whole config is consistent, 0 with *bad_layer set
   to the first violating layer otherwise (bad_layer may be NULL). */
int lz_csa2_last_writer_valid(const LZModelConfig *c, int *bad_layer);

#endif
