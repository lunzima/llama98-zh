#ifndef LZ_FORWARD_H
#define LZ_FORWARD_H

#include "model.h"
/* For LZ_GDN_STATE_2PLANE, which gates a field of LZRunState below and the
   arity of lz_gdn_step. This include is load-bearing, not tidiness: when
   the macro was only ever set with -D on the command line the ordering
   happened to work, and moving its default into ops.h broke the build
   here - forward.h was reached first, the guard read undefined-as-0, and
   the field vanished while ops.c still expected it. */
#include "ops.h"
/* For LZInspect, which LZRunState carries a pointer to. A pointer alone
   would only need a forward declaration, but the struct is small, has no
   dependencies of its own, and forward.c dereferences it - so including
   it here keeps every consumer from having to remember to. */
#include "inspect.h"

/* KV cache formats, selectable independently for keys and values
   (--kv / --kv-k / --kv-v). See LZRunState's kfmt field. */
#define LZ_KVF_Q8    0   /* int8, one absmax scale per 32 elements */
#define LZ_KVF_Q4    1   /* 4-bit Gaussian codebook, one norm per row */
/* 2 is deliberately unused (it was LZ_KVF_Q4R2; forward.c carries the
   measurement). The gap is deliberate: nothing iterates this range, and
   renumbering F32 would silently change what an old --kv value means to
   any caller that still passes an integer. */
#define LZ_KVF_F32   3   /* unquantized; a measuring arm, not a mode */

/* Default attention sink when a window is requested without one. 16 was
   measurably better than 4 (+5.351%% vs +6.095%% PPL at the same total
   span) and both are an order of magnitude better than none (+43.764%%). */
#define LZ_ATTN_SINK_DEFAULT 16

/* Floor for the AUTO attention window (lz_attn_window < 0, the default).
   1024 because window 1008 at ctx 2048 measured -0.142%% PPL - free, and
   marginally better than attending to everything - while window 496 cost
   +2.630%% and window 240 cost +5.428%%. Absolute rather than a fraction
   of ctx because the same window 240 cost +5.351%% at ctx 1024 and
   +5.428%% at ctx 2048: the harm follows the window's size, not its
   ratio to the context. Raise it and eviction does less; lower it and
   you leave the measured-free region. */
#define LZ_ATTN_WIN_FLOOR 1024

/* --profile phases. Coarse on purpose: the question a profile has to
   answer first is "which of the four big blocks", not "which line".

   TWO TIERS, and mixing them is a measurement error, not a formatting
   one. The first LZ_PROF_TOP entries partition the forward: their
   timers do not overlap, so they may be summed. REC and ACT are timed
   INSIDE those spans - REC around lz_gdn_step/lz_kda_step within LIN's
   timer, ACT around the SwiGLU loop within FFN's - so summing all seven
   counts that time twice, and a total larger than the run's own wall
   clock is the tell. */
#define LZ_PROF_ATTN   0
#define LZ_PROF_LIN    1
#define LZ_PROF_FFN    2
#define LZ_PROF_HEAD   3
#define LZ_PROF_NORM   4
#define LZ_PROF_TOP    5   /* [0,LZ_PROF_TOP) partition; sum only these */
#define LZ_PROF_REC    5   /* within LIN  */
#define LZ_PROF_ACT    6   /* within FFN  */
#define LZ_PROF_N      7

/* Single-step forward and runtime state for Qwen3.5 (M4.5).

   The hybrid architecture has two independent kinds of state:
   - full_attention layers use a KV cache that grows linearly with pos;
   - linear_attention layers use a fixed-size SSM recurrent state plus a
     convolution history, independent of pos.
   Each kind is indexed only within its own layer class, so we need a map
   from layer index to the index within that class. */

typedef struct {
    int seq_len;                /* max context length */
    int nt_cap;                 /* how many tokens the activation buffers hold (= LZ_BATCH_MAX) */

    /* Buffers marked (dim) are actually allocated nt_cap x dim, token-major:
       token t's slice lives at buf + t*dim. Decoding uses only the t=0
       slice, identical to the pre-batch behavior. */
    float *x, *xb, *xb2;        /* (hidden) */
    float *hb, *hb2;            /* (intermediate) */
    float *logits;              /* (vocab) only the LAST token's, not scaled by nt */

    /* full_attention scratch */
    float *qg;                  /* (attn_qgate_dim) q and gate interleaved per head */
    float *qh;                  /* (attn_q_dim)  normalized and rotated q */
    float *att;                 /* (seq_len)     per-head scores (reused per t, not scaled by nt) */
#if (LZ_ATTN_FIXED & 2)
    float   *wsum_cbuf;         /* (seq_len)     weighted-sum coefficients before quantization */
    int16_t *wsum_cq;           /* (seq_len)     the same, quantized to int16 */
#endif
    float *attn_out;            /* (attn_q_dim) */
    float *ktmp;                /* (attn_kv_dim) k before it enters the cache (post-rotation, quantized) */
    float *vtmp;                /* (attn_kv_dim) same for v */
    /* KV cache Q8: kq8/vq8 hold the rotated k/v (group 32), dequantized at score time.
       (K group 16 measured no gain.)
       Depth is n_full_layers PLUS ONE when m->mtp != NULL: the MTP
       block's own full_attention layer gets the last slot (index
       n_full_layers), reusing this same cache rather than a bespoke
       buffer - see the MTP scratch block below. */
    int8_t *kq8;                /* (n_full [+1 if m->mtp], seq, attn_kv_dim) */
    int8_t *vq8;
    float  *ksq;                /* (n_full, seq, attn_kv_dim/32) */
    float  *vsq;                /* (n_full, seq, attn_kv_dim/32) */
#if LZ_KV_2PLANE
    /* Low planes; same shape as kq8/vq8, sharing their scales. Value is
       (hi + lo/LZ_GDN_LO_SCALE) * scale. See LZ_KV_2PLANE in ops.h -
       this is OFF by default and exists to MEASURE whether the KV cache
       wants the extra precision, not because it was shown to. */
    int8_t *kq8_lo;
    int8_t *vq8_lo;
    float  *att_lo;             /* (seq_len) low-plane scores, before the 1/254 fold */
    float  *wsum_lo;            /* (head_dim) low-plane weighted sum, likewise */
#endif

    /* Unquantized KV cache (--kv f32). A MEASURING INSTRUMENT, not a
       deployment mode: it costs 4x the Q8 planes and no target machine
       has that memory.

       It exists because every claim about a KV format is a ratio, and
       until this arm existed the denominator could not be measured. The
       Hadamard rotation was evaluated as "on vs off" and came out 0.08 to
       0.30%% worse - a number that cannot be interpreted without knowing
       how much Q8 was losing in the first place: if Q8 already costs
       nothing, no rotation can win, and the comparison was never about
       the rotation. This repo has made exactly that mistake before with a
       bandwidth ratio whose denominator was unmeasurable (iron law 3).

       NULL unless --kv f32; the Q8 path is untouched and stays the
       default, bit for bit. */
    float  *kf32;               /* (n_full [+1 if m->mtp], seq, attn_kv_dim) */
    float  *vf32;
    /* Cache format, chosen PER SIDE. Not one knob for both, because the
       two sides are not worth the same: decomposed on cci3-hq with the
       f32 arm as reference, 4-bit costs +0.379%% PPL on the key side and
       +1.271%% on the value side (they sum to 1.650 against a joint
       1.701, so the split is essentially additive). Spending the same
       bit width on both means overpaying for keys.

       WHICH SIDE IS EXPENSIVE DEPENDS ON THE LANGUAGE. Decomposed with
       the f32 arm as reference, 1024 tokens, 4-bit on one side at a time:

                          K side    V side    joint
         TinyStories-en   +0.574%   +0.133%   +0.623%
         cci3-hq (zh)     +0.379%   +1.271%   +1.701%

       The key side costs about the same either way. The VALUE side
       differs by a factor of nine, and that single fact explains two
       tables that otherwise contradict each other (f32 reference, State
       at ctx 2048 / 2018 slots):

         arm            en         zh        State
         K q8   V q8    -0.126%   -0.225%    45.1 MB
         K q4   V q4    +0.623%   +1.701%    36.3 MB
         K q4   V q8    +0.660%   -0.072%    40.7 MB

       English is key-dominated, so correcting the key residual nearly
       erases its gap and upgrading values buys nothing. Chinese is
       value-dominated, so the reverse. There is no single asymmetric
       allocation that is right for both, and picking one from English
       measurements would be picking the wrong one for this project -
       kunkun98 is a Chinese model, which puts K q4 / V q8 on the table
       and the residual correction off it.

       *** RE-MEASURED ON THE ACTUAL TARGET MODEL; THE NUMBERS ABOVE
       *** SHOULD NOT BE QUOTED.
       models/kunkun98-recover-r20, all 32 val documents, 20,885
       predicted positions, --dump-nll against the --kv f32 arm, paired
       per position:

         arm            d bits/tok   rel      t vs f32   worse positions
           q8   q8       -0.000355   -0.007%     -0.7        50.3%
           K q4  V q8    +0.008360   +0.163%      4.9        52.7%
           K q8  V q4    +0.021561   +0.419%     11.4        55.3%
           q4   q4       +0.028454   +0.553%     11.6        55.4%

       The DIRECTION replicates: on Chinese the value side is the
       expensive one, here by 2.6x (paired K-vs-V contrast +0.0132
       bits/tok, t = +5.3). The MAGNITUDES do not, and neither does the
       sign of K q4 / V q8 - the table above has it at -0.072% (better
       than f32) while this measures +0.163% with t = 4.9 (worse, and
       significantly so).

       *** THE TABLE ABOVE WAS TAKEN AT 1024 TOKENS AND THAT IS TOO FEW.
       Measured here on one 1024-token document first, the same paired
       test gave t = +0.31 / -1.24 / +0.71 for q8 / q4 / q4r2 - nothing
       resolved - and the share of positions that got WORSE was 50/50 in
       every arm. KV quantization error is near-symmetric per position:
       it helps as many tokens as it hurts and leaves a small mean
       shift, so a thousand positions cannot see effects this size. Even
       at 20,885 the worse-position share only reaches 55%. Any future
       KV precision claim needs 20k positions, not 1k.

       q8 is FREE on this model: -0.000355 bits/token at t = -0.7, i.e.
       not distinguishable from the f32 cache in either direction, for a
       4x saving. It is the default and the default is right.

       The zero control for all of the above: --kernel ref against the
       auto (sse2) tier is +0.000000 bits/token with 0.0% of positions
       differing, which is what says this harness can tell "no
       difference" from "small difference" at all. */
    int     kfmt, vfmt;

    /* 4-bit KV cache (--kv q4). Half the bytes of the Q8 planes, which is
       the only reason to touch the KV format at all on this target: the
       f32 arm above measures Q8's quality cost at -0.13 to -0.23%% PPL,
       i.e. there is no quality headroom to chase, only bandwidth.

       One scale per (layer, position, kv head) rather than per 32
       elements - see ops.h's lz_kv4_quantize for why that choice and the
       Hadamard rotation only make sense together. */
    unsigned char *k4;          /* (n_full [+1], seq, attn_kv_dim/2) bytes */
    unsigned char *v4;
    float  *ks4;                /* (n_full [+1], seq, n_kv_heads) */
    float  *vs4;

    /* The QJL key sketch and its q4r2 residual form are deliberately
       absent; forward.c's own comment above lz_kv_rot_enable carries the
       measurement that retired them and the reason iron law nine does
       not protect them.

       ONE OPEN QUESTION SURVIVES THE FORMAT: whether the residual
       Chinese gap lived in the VALUE reconstruction rather than in key
       scores. Answered on kunkun98-recover-r20, 20,885 positions, f32
       arm as reference:

         K q4 / V q8   +0.163%      K q8 / V q4   +0.419%

       Confirmed, by a factor of 2.6, paired t = +5.3, at IDENTICAL state
       (7.4 MB both). So a key-side correction was always going to be
       working on the cheaper half - which is also why the residual
       sketch could not earn its bytes. If one side is to be quantized,
       it is the key side. */


    /* linear_attention scratch */
    float *qkv;                 /* (lin_conv_dim) */
    float *qkv_c;               /* (lin_conv_dim) after the causal conv */
    float *zbuf;                /* (lin_value_dim) - also KDA's g_proj output */
    float *avec, *bvec;         /* (lin_n_v_heads) - bvec also KDA's b_proj output */
    float *ssm_out;             /* (lin_value_dim) - shared by GDN and KDA */
    float *qn, *kn;             /* (nt_cap, lin_k_head_dim) L2-norm scratch;
                                   the serial recurrence uses row 0 only */

    /* LZ_LT_KDA scratch. Separate q/k/v buffers rather than GDN's one
       fused in_proj_qkv: lz_matmul_q8_nt writes o[t*out_dim+i] with a
       FIXED stride, so three independently-shaped projections cannot
       share one nt-major buffer at different offsets without breaking
       that contract (see forward.c's forward_kda). conv_state IS still
       shared with GDN there: its total size only depends on
       lin_conv_dim, which is the same whether that width is one fused
       conv or three independent ones sliced out of the same buffer. */
    float *kda_q, *kda_k;       /* (nt_cap * lin_key_dim) pre-conv */
    float *kda_v;                /* (nt_cap * lin_value_dim) pre-conv */
    float *kda_qc, *kda_kc;     /* (nt_cap * lin_key_dim) post-conv, post-activation */
    float *kda_vc;                /* (nt_cap * lin_value_dim) post-conv, post-activation */
    float *kda_gate_lat;         /* (nt_cap * kda_gate_rank) f_a_proj output */
    float *kda_gate;              /* (nt_cap * lin_n_v_heads * lin_k_head_dim) decay,
                                      ALREADY EXPONENTIATED (gt = exp(g), same
                                      convention lz_gdn_step's scalar gt uses) */

    /* latent MoE scratch (LZLayer.ffn_moe layers only). TWO WIDTHS, and
       the split is the routing boundary: what one weight matrix serves
       for every token in the chunk is nt_cap-wide and runs batched,
       what depends on WHICH expert a token picked is one token's worth.
       Only the routed experts are in the second group -
       lz_matmul_q8_nt's one-weight-load-serves-nt-tokens batching has
       nothing to hold onto when two tokens in a chunk want different
       experts. See forward_moe's docstring in forward.c, which also
       records why grouping the tokens by expert was measured and
       rejected. */
    float *moe_router_logits;    /* (nt_cap * num_experts) */
    int   *moe_sel_idx;           /* (num_experts_per_token) */
    float *moe_sel_w;              /* (num_experts_per_token) */
    float *moe_lat_x;              /* (nt_cap * moe_latent_dim) routed_expert_down_proj output */
    float *moe_lat_y;              /* (nt_cap * moe_latent_dim) routed experts' weighted sum */
    float *moe_h1, *moe_h3;       /* (nt_cap * max(moe_intermediate_size, moe_shared_width)) */
    float *moe_h2;                 /* (moe_latent_dim) one expert's w2 output before scaling */
    float *moe_shared_out;        /* (nt_cap * hidden_size) shared expert's down_proj output */

    /* MTP draft head scratch (LZ_SPEC_K > 0 at runtime; allocated only
       when m->mtp != NULL - see lz_state_alloc). The KV cache the MTP
       block's own attention uses is NOT a separate buffer: it is one
       more slot appended to kq8/vq8/ksq/vsq above (index n_full_layers,
       via forward_attn's cache-layer sentinel in forward.c) - it needs
       no rollback of its own (see lz_spec_round's docstring in
       generate.c: dangling rows from a rejected draft are simply
       overwritten by the next round at higher positions, same
       "KV rollback is just not advancing the position pointer" argument
       forward.h already makes for the body's cache above).

       mtp_chain is both read AND overwritten by lz_mtp_draft_step: it
       carries the "hidden" register across chained draft steps (seeded
       by lz_forward_capture with the body's POST-final-norm hidden - see
       lz_forward_capture's own comment below - for the first step, then
       replaced each step with that step's own post-norm output, matching
       upstream's own recursion - see lz_mtp_draft_step's comment in
       forward.c).

       mtp_x/mtp_concat/mtp_emb_raw are sized nt_cap wide (not just one
       token), even though lz_mtp_draft_step's chained per-step usage
       only ever touches slot 0 - lz_mtp_prefill (below) reuses the same
       buffers batched across nt_cap prompt positions at
       once, and a second, differently-sized set of scratch would just
       be a driftable duplicate of this one. */
    float *mtp_x;              /* (nt_cap*hidden) MTP block's own residual stream */
    float *mtp_concat;         /* (nt_cap*2*hidden) fc input: [pre_fc_norm_embedding | pre_fc_norm_hidden] */
    float *mtp_emb_raw;        /* (nt_cap*hidden) raw embedding lookup scratch, pre-norm */
    float *mtp_chain;          /* (hidden) chained "hidden" register, see above - single-token only, prefill never touches it */
    float *mtp_draft_logits;   /* (vocab) one draft step's logits */
    /* The MTP block's OWN position counter, separate from the body's
       absolute token position - NOT the same number, because
       lz_mtp_prefill (below) only ever covers n_prompt-1 positions
       (0..n_prompt-2, the same span the body's own prefill forwards -
       see lz_generate_resume), never the full absolute range a
       multi-turn conversation's body position can reach. Indexing the
       MTP's KV cache slot (this struct's kq8 note) by the body's
       absolute position would make its self-attention scan phantom
       all-zero rows for any turn after the first. Also: a rejected
       draft step's row must never be attended over by a LATER round
       either (same rollback argument as the body's SSM/conv checkpoint,
       just satisfied by not advancing this counter past accepted steps,
       not by copying anything - see lz_spec_round). Owned entirely by
       the caller (generate.c): forward.c only reads it as a plain
       position argument to lz_mtp_draft_step/lz_mtp_prefill, never
       advances it itself. Reset to 0 by lz_state_reset, like every
       other recurrent index.

       Prefill (filling a gap against llama.cpp's own speculative.cpp):
       a fresh generation would otherwise start every MTP round blind -
       s->mtp_pos at 0 with an empty KV cache, no prompt history at all,
       even though the body's own KV cache is full of it. lz_mtp_prefill
       runs the MTP block over the body's
       already-computed prefill hidden states purely to populate this
       KV cache before the first round; see its own comment
       (forward.c) and lz_generate_resume's prefill call site. */
    int mtp_pos;
    /* Verify-pass logits: EVERY position of the (k+1)-token verify batch,
       not just the last - lz_forward_batch skips intermediate logits on
       purpose (forward.h's own note above it), but speculative verify
       needs to check draft[i] against the target model's OWN prediction
       at each position (see lz_spec_accept, llama_zh.h). Sized off
       LZ_SPEC_K_MAX (ops.h), the RUNTIME ceiling, not LZ_SPEC_K's
       compile-time default - same relationship LZ_BATCH_MAX has to
       nt_cap. */
    float *mtp_logits;         /* (LZ_SPEC_K_MAX+1) * vocab */

    /* Verify-pass POST-final-norm hidden states, one row per
       verify-batch position, same shape/offsetting as mtp_logits above
       (matching llama.cpp's own speculative.cpp: its accept() takes row
       `n_accepted` of exactly this snapshot as the next round's draft
       seed, instead of a dedicated extra forward - see lz_spec_round's
       own comment in generate.c for why row n_accept is causally
       unaffected by whatever happens at LATER, possibly-rejected-and-
       rolled-back rows). Post-norm, NOT raw: the draft head's
       pre_fc_norm_hidden consumes post-final-norm hiddens (verified
       element-wise against the HF reference forward; the batch's own
       LAST row arrives pre-normed from forward_chunk's in-place
       final_norm and is re-normed - a small, accepted deviation on that
       one row). */
    float *mtp_verify_hidden;  /* (LZ_SPEC_K_MAX+1) * hidden */

    /* temp>0 speculative decoding phase 2 (generate.c's
       lz_spec_round_temp): one row per DRAFTED token holding lz_sample_
       temp_q's own full post-temperature distribution q (sampler.c) -
       LZ_SPEC_K_MAX rows, not +1, since the verify batch's bonus/final
       row is never drafted, only corrected/emitted, and so has no q of
       its own to accept/reject against (lz_spec_accept_temp's own
       contract, llama_zh.h, only ever tests DRAFTED positions).
       Persisted across the whole draft loop (unlike mtp_draft_logits
       above, a single-row scratch overwritten every step) because
       verify runs only after ALL k draft steps, and each row's own q
       is needed again then - see lz_spec_round_temp's own comment for
       why recomputing it instead of storing it was rejected (the draft
       head's hidden state has already moved on by the time verify
       returns, so a stored row cannot be reconstructed after the
       fact). */
    float *mtp_draft_q;        /* LZ_SPEC_K_MAX * vocab */
    /* Single-row scratch for lz_target_dist's own out_p (sampler.c) -
       one verify row's target distribution p, reused across the
       k_eff+1 rows lz_spec_round_temp processes one at a time (each
       row's accept/reject decision is finalized before the next row's
       p is built, so unlike mtp_draft_q above this never needs more
       than one row alive at once). Cannot alias mtp_logits' own rows:
       lz_target_dist's out_p is a DIFFERENT array from the logits it
       reads (memset to 0 then scattered candidate values - aliasing
       them would erase the softmax it is still reading from). */
    float *mtp_target_p;       /* vocab */

    /* SSM/conv recurrent state, widened into a (ring_depth, ...) ring
       (user-directed: speculative-decode rollback without a
       checkpoint-restore-and-replay - see generate.c's lz_spec_round,
       and s->ssm_slot below for the mechanism). ring_depth is
       LZ_SPEC_K_MAX+1 when m->mtp != NULL (lz_state_alloc), 1
       otherwise - a model that can never run --spec never pays for a
       ring it cannot use. Per-slot shape is the same as the single-slot
       form:

         ssm_state_q8(_lo)  (n_linear, n_v_heads, k_head_dim, v_head_dim)
         ssm_state_s        same shape /32 (one f32 scale per Q8 group)
         conv_state         (n_linear, lin_conv_dim, conv_kernel-1)

       so slot i's byte range is exactly `i * (that per-slot size)`, the
       same size lz_ckpt_alloc's ckpt_sizes() computes and always has
       (unrelated LZStateCkpt, see its own comment below, keeps working
       against ONE slot's worth unchanged).

       WHY THIS COSTS ZERO EXTRA BANDWIDTH, ONLY MEMORY (the argument
       this whole design depends on - see lz_gdn_step's header comment
       in ops.h for the byte-count proof): every recurrent step here
       ALREADY reads the entire per-head state once and rewrites the
       entire per-head state once, unconditionally, every single token
       (lz_gdn_step/lz_kda_step's own two-pass shape; lz_causal_conv1d_
       step's own full-history read+roll). Widening to a ring and
       reading from slot i while writing slot i+1 touches the exact
       same number of bytes as reading and writing slot i in place -
       only the ADDRESSES differ, not the byte count. If the recurrence
       ever becomes incremental (only some rows touched per step), this
       stops being free and the ring's cost has to be re-derived from
       scratch - the argument's precondition is worth restating exactly
       because it is exactly the kind of thing a future change could
       quietly invalidate. */
    int8_t *ssm_state_q8;
#if LZ_GDN_STATE_2PLANE
    int8_t *ssm_state_q8_lo;    /* low plane, same shape as ssm_state_q8 */
#endif
    float  *ssm_state_s;
    float *conv_state;          /* (n_linear, lin_conv_dim, conv_kernel-1) */
    /* Which ring slot currently holds the CONFIRMED recurrent state.
       Ordinary decode and prefill NEVER advance this (forward_ssm/
       forward_kda pass the same slot as both read and write source,
       which is the single-slot in-place update - forward.c's own
       comment at the call site). Only a speculative verify batch
       (lz_forward_verify, nt = k_eff+1 tokens) advances it internally,
       one slot per token forwarded, WITHOUT touching s->ssm_slot itself
       until the round is over - see lz_spec_round's own comment for
       the exact indexing contract (which slot holds "state after
       n_accept+1 tokens", the value this field gets set to once
       accept/reject is decided; a rejection or a mid-round stop is
       therefore a plain index assignment, zero copy, zero forward).
       Reset to 0 by lz_state_reset, like every other recurrent index
       (mtp_pos's own comment makes the same point). */
    int     ssm_slot;
    /* How many ring slots ssm_state_q8(_lo)/ssm_state_s/conv_state
       actually have - computed ONCE in lz_state_alloc (m->mtp ?
       LZ_SPEC_K_MAX+1 : 1) and stored here rather than recomputed at
       every site that needs it (lz_state_reset, forward_ssm,
       forward_kda): the "must agree with lz_state_alloc's own formula"
       duplication that pattern would create is exactly the kind of
       thing that silently drifts after a config field changes - this
       project has hit that shape of bug before (see ckpt_sizes's own
       comment in forward.c, added for the identical reason on
       LZStateCkpt's shapes). Never 0. */
    int     ssm_ring_depth;

    /* Q8 activation quantization buffers (capacity = max matmul input dim) */
    int8_t *xq;                 /* quantized activations */
    float  *xqs;                /* per-group scales */
    float  *wscr;               /* weight dequantization scratch (norm/embed rows) */
    int     qcap;               /* xq capacity */

    /* Precomputed RoPE table: (seq_len x rotary_dim/2) {cos, sin} pairs.
       Shared by all layers/heads under the same theta, avoiding a per-token
       per-layer pow/cos/sin (expensive on PII's x87). */
    float  *rope_cs;

    /* 1/sqrt(dim) for attention and SSM: config-only, computed once at
       alloc instead of per layer per token (FSQRT+FDIV about 110 cycles
       on x87). */
    float   attn_scale;         /* 1/sqrt(head_dim) */

    /* Hadamard rotation of the attention activations before the KV cache
       is quantized (llama.cpp PR 21038's method: rotate Q, K,
       V, cache the rotated K/V, and rotate the attention output back).
       Rotation preserves dot products, so scores are unchanged in exact
       arithmetic; what changes is that the rotated vectors have no
       outlier coordinates left for a per-group absmax scale to waste its
       range on. PR 21038 measures this end to end on Qwen3 0.6B: Q8 KV
       goes from PPL 13.9115 to 13.6713 against an f16 cache baseline of
       13.6711 - i.e. the rotation makes Q8 KV free.

       Sizes follow that PR: K uses the largest usable power of two (the
       whole head here), V uses 64 ("using smaller rotation matrices for V
       seems beneficial"). 0 means "no rotation", which every model whose
       head_dim is not a multiple of 64 gets, and which --kv-rot off
       forces - that path must stay bit-identical to the pre-rotation
       engine.

       These are sizes, not booleans, because the rotation block width is
       exactly the kind of knob iron law 3 says must be sweepable rather
       than baked in: it trades mixing quality against nothing on this
       machine (lz_fwht is n*log2(n) adds either way), but that balance is
       a property of the target, not of the algorithm.

       DO NOT TRY TO FOLD THESE INTO THE PROJECTION WEIGHTS. QuaRot-style
       fusion was considered and is mostly impossible here, and the part
       that is possible is not worth having.

       Per token per full-attention layer, head_dim 256, 32 query heads,
       2 KV heads:
         q rotation      32 x FWHT(256)     65,536 adds   NOT foldable
         output unrotate 32 x 4 x FWHT(64)  49,152 adds   NOT foldable
         k rotation       2 x FWHT(256)      4,096 adds   NOT foldable
         v rotation       2 x 4 x FWHT(64)   3,072 adds   foldable
       against 16,384*L mul-adds for the attention itself: 11%% of that at
       L=64, 1.5%% at L=512, 0.4%% at L=2048. Wall clock at 120 tokens
       could not separate rot on from rot off at all (8.2 s vs 8.4 s).

       Why the two big ones cannot move. RoPE, written on the complex
       pairs z_i = x_i + i*x_{i+32}, is multiplication by e^(i phi_i) per
       component. A real linear map commutes with ALL such diagonal phase
       multiplications only if it is diagonal in that complex basis - a
       per-pair rotation, mixing nothing across pairs. Mixing across pairs
       is the entire function of the Hadamard, so no foldable H exists
       over the rotary dimensions. And the output un-rotation is separated
       from o_proj by the output gate, which is elementwise against qgt in
       the ORIGINAL basis and is not diagonal in the rotated one.

       On a P6 the case is weaker still, not stronger: FADD is one per
       cycle and FMUL one per two, so a pure-add transform is relatively
       cheaper there than here (5.8%% of attention at L=64, 0.7%% at
       L=512). */
    int     kv_rot_k;           /* Hadamard width for K and Q, 0 = off */
    int     kv_rot_v;           /* Hadamard width for V and the output */

    /* StreamingLLM: keep the first attn_sink positions and the most
       recent attn_win, mask the middle. 0/0 = ordinary full attention.
       See forward.c for why this lands as a mask before the eviction. */
    int     attn_sink, attn_win;
    int     kv_ring;            /* ring modulus (window + spec guard), 0 = off */
    int     kv_slots;           /* allocated cache depth: seq_len, or sink+ring */
    float   ssm_scale;          /* 1/sqrt(lin_k_head_dim) */

    int *cache_idx;             /* layer index -> index within its class */
    long long bytes_alloc;

    /* Bumped by lz_state_alloc and lz_state_reset. A checkpoint records
       it and lz_ckpt_restore refuses a mismatch.

       This exists because a checkpoint deliberately does NOT copy the KV
       cache - see LZStateCkpt - so restoring is only sound while the KV
       entries below the checkpoint's position still hold the same
       prefix. A reset zeroes them. Without the counter, restoring across
       a reset would put a correct recurrent state on top of a zeroed KV
       and generate confidently wrong text, with nothing to see: no
       crash, no error, just a different answer. */
    unsigned epoch;

    /* Where lz_forward writes what it just did, or NULL for "do not
       bother". It rides on the run state rather than on lz_forward's
       parameter list because every caller already threads this struct
       through and almost none of them want an inspector: adding a
       parameter would have touched the CLI, the endpoint, the MTP path
       and every test, all to pass NULL.
       Only the fields lz_forward owns are written here (the expert
       bitmap); the sampler fills its own half through lz_sample_ex,
       which is a different object's business and takes it as an
       argument. */
    LZInspect *ins;
} LZRunState;

/* Snapshot of the position-carrying RECURRENT state, for reusing a
   conversation prefix across turns instead of re-forwarding it.

   Measured on a 10-turn chat: the tokens that actually need
   re-forwarding are a constant 53 per turn, while turn 10's prompt is
   481 tokens - 428 of them re-derive a state that was already computed.

   What is here, and what is deliberately not:

     ssm_state_q8 (+ _lo), ssm_state_s, conv_state   copied - 1.51 MB
     kq8 / vq8 / ksq / vsq (the KV cache)            NOT copied

   The KV cache is append-only and indexed by absolute position, so
   rewinding to position P and forwarding different tokens simply
   overwrites entries at >= P while entries below P stay valid. Copying
   it would cost 6 MB at seq_len 2048 and buy nothing. The price of that
   choice is the `epoch` field above: the KV must not have been reset in
   between, and that has to be CHECKED, not assumed.

   NOT the speculative-decode rollback ring (s->ssm_slot above) - the
   two are deliberately separate mechanisms with separate owners, on
   the user's own explicit instruction not to mix them. This type
   exists for cross-TURN prefix reuse (LZPrefixCache, several seconds
   and hundreds of tokens apart) and still does a real save/restore
   copy - a turn boundary is not on any hot path, so there is nothing
   to make free here the way the ring makes a spec round's rollback
   free. lz_spec_round does not take an LZStateCkpt parameter for its
   own per-round rollback (that use is served by s->ssm_slot); this
   struct's only caller is LZPrefixCache. */
typedef struct {
    int8_t *ssm_q8;
#if LZ_GDN_STATE_2PLANE
    int8_t *ssm_q8_lo;
#endif
    float  *ssm_s;
    float  *conv;
    size_t  n_ssm_q8, n_ssm_s, n_conv;   /* element counts, for a shape check */
    int     pos;                         /* tokens forwarded when taken; -1 = empty */
    unsigned epoch;                      /* the state's epoch at save time */
} LZStateCkpt;

/* Allocate a checkpoint sized for this model. Non-zero + errbuf on
   failure. Safe to call lz_ckpt_free on a zeroed struct. */
int  lz_ckpt_alloc(LZStateCkpt *ck, const LZModel *m, char *errbuf, int errlen);
void lz_ckpt_free(LZStateCkpt *ck);

/* Capture s's recurrent state; `pos` is how many tokens have been
   forwarded into s. Non-zero + errbuf on failure. */
int  lz_ckpt_save(LZStateCkpt *ck, const LZRunState *s, const LZModel *m,
                  int pos, char *errbuf, int errlen);

/* Put the captured state back and report its position through *out_pos.
   Refuses (non-zero) an empty checkpoint, a shape mismatch, or a state
   that has been reset or reallocated since the save - the last one is
   the whole reason epoch exists. The caller then continues with
   lz_generate_resume(start_pos = *out_pos). */
int  lz_ckpt_restore(const LZStateCkpt *ck, LZRunState *s, const LZModel *m,
                     int *out_pos, char *errbuf, int errlen);

/* spec_k_max (the SSM/conv rollback ring - s->ssm_slot's own comment):
   the LARGEST --spec K this state will ever be asked to
   run, or 0/negative if it will never run speculative decoding at all.
   Sizes s->ssm_ring_depth to spec_k_max+1 (clamped to LZ_SPEC_K_MAX)
   instead of always the compile-time ceiling - a caller that knows it
   only ever wants k<=2 (say) should not pay for slots 3..6. Silently
   clamped up to 1 (never 0) when m->mtp is NULL, since a model with no
   MTP head can never run --spec regardless of what is passed here.

   THIS IS A SIZING HINT, NOT A CONTRACT lz_generate_resume enforces on
   its own: opts->spec_k must not exceed s->ssm_ring_depth - 1, checked
   there (LZ_ERR_SPEC_K_RANGE) precisely because a k too large for what
   was allocated here would silently alias ring slots via `% ring_depth`
   - a much worse failure than a clean refusal. Callers that do not know
   or do not care what k a caller will eventually request (most tests,
   tools, and the HTTP server, which does not wire --spec at all yet)
   should pass LZ_SPEC_K_MAX, the compile-time ceiling, so the state
   accepts any --spec request - passing 0 there would build a state
   that then REFUSES any --spec request at all, not an efficiency-only
   choice. */
int  lz_state_alloc(LZRunState *s, const LZModel *m, int seq_len,
                        int spec_k_max, char *errbuf, int errlen);
void lz_state_free(LZRunState *s);
/* Clear KV cache, SSM recurrent state and conv history; call before a new generation */
void lz_state_reset(LZRunState *s, const LZModel *m);

/* Single-step forward; returns a pointer into s->logits (do not free).
   pos starts at 0; the caller advances it. */
float *lz_forward(const LZModel *m, LZRunState *s, int token, int pos);

/* Batched forward (prefill): tokens[0..n) occupy positions pos0..pos0+n-1;
   returns the LAST token's logits (intermediate logits are not computed —
   lm_head is the single largest matmul, and nobody needs its intermediate
   results during prefill).

   n > LZ_BATCH_MAX is chunked internally; callers need not care about the
   batch width.

   **Bit-identical to calling lz_forward token by token**, including KV
   cache, SSM recurrent state and conv history. Batching only reuses weight
   loads; it changes no rounding anywhere.

   Verification compares all five buffers named above, not just the
   logits: a state that has diverged but whose argmax has not yet
   noticed is the worst shape this defect can take, since every later
   token is computed from it. */
float *lz_forward_batch(const LZModel *m, LZRunState *s,
                        const int *tokens, int n, int pos0);

/* ---------------------------------------------------- MTP speculative decoding */

/* Like lz_forward, but also seeds s->mtp_chain with the POST-final-norm
   residual-stream hidden state that produced this token's logits -
   exactly the vector the trunk's own lm_head consumes - the MTP draft
   head's "h_body" input for the first step of a round (model.h's LZMtp
   forward comment; the POST-final-norm stage is deliberate, see
   forward.c's own comment on capture_hidden for the three-way
   citation). A no-op
   beyond the ordinary forward when m->mtp is NULL (mtp_chain is then
   unallocated and untouched). Returns a pointer into s->logits, like
   lz_forward; NULL on the same failures lz_forward can have. */
float *lz_forward_capture(const LZModel *m, LZRunState *s, int token, int pos);

/* One MTP draft step. Reads s->mtp_chain as the "hidden" input (seeded by
   lz_forward_capture for the first step of a round; OVERWRITTEN by this
   call with this step's own post-norm output, which is exactly upstream's
   own chaining rule - see forward.c). next_token's embedding is the
   step's "next token" input (the just-produced anchor/draft token).
   pos is this step's position in the MTP block's OWN numbering (see
   s->mtp_pos above) - NOT the body's absolute token position - used for
   the MTP block's own RoPE and its reserved KV cache slot (forward.h's
   kq8 note). The caller (generate.c's lz_spec_round) owns advancing
   s->mtp_pos; this function only consumes whatever position it is given.
   Returns a pointer to s->mtp_draft_logits (vocab entries), or NULL if
   m->mtp is NULL or pos is out of range. */
float *lz_mtp_draft_step(const LZModel *m, LZRunState *s, int next_token, int pos);

/* Speculative-decode VERIFY pass: like lz_forward_batch (same chunking,
   same bit-identical-to-token-by-token guarantee for the hidden states),
   but computes EVERY position's logits into s->mtp_logits, not just the
   last - lz_spec_accept (llama_zh.h) needs the target model's own
   prediction at each drafted position, not only the final one.
   n is normally k+1 (the accepted anchor plus k draft tokens - see
   generate.c's lz_spec_round); values beyond LZ_SPEC_K_MAX are not
   expected to occur since the CLI/opts path caps spec_k there, but this
   function itself only requires n >= 1.
   Returns 0 on success, LZ_ERR_FORWARD-worthy failure (NULL from an
   internal forward_chunk call) as nonzero - errbuf is the caller's job,
   same convention lz_forward/lz_forward_batch use (no errbuf here). */
int lz_forward_verify(const LZModel *m, LZRunState *s,
                      const int *tokens, int n, int pos0);

/* Like lz_forward_batch, but also writes the POST-final-norm
   residual-stream hidden state of EVERY one of the n positions into
   hidden_out (n*hidden_size floats, token-major) - lz_forward_capture's
   own capture (see its comment above), for EVERY position, not just the
   batch's last - needed by lz_mtp_prefill below (it needs position i's
   body hidden to run the MTP block AT position i, not just the batch's
   last one).
   Chunked exactly like lz_forward_batch (same LZ_BATCH_MAX-bounded
   loop); each chunk writes its own slice of hidden_out at the right
   offset, so the caller does not need to know the chunk width either.
   Returns the last chunk's logits pointer (same convention as
   lz_forward_batch - callers here only want hidden_out, not the
   logits, but returning NULL on failure still lets the caller check it
   the same way). */
float *lz_forward_batch_capture(const LZModel *m, LZRunState *s,
                                const int *tokens, int n, int pos0,
                                float *hidden_out);

/* Run the MTP block over n prompt positions purely to populate its own
   KV cache before the first speculative round - see s->mtp_pos's
   comment above and this function's own comment in forward.c for why a
   fresh generation would otherwise start every round blind. h_body_all[i] is
   the body's post-final-norm hidden at position pos0+i
   (lz_forward_batch_capture's own output); next_tokens[i] is the
   actual prompt token
   that followed position pos0+i (the MTP's ground-truth "next token"
   input, same role lz_mtp_draft_step's next_token argument plays for
   one chained step). Output logits are never computed - see forward.c;
   this call's only observable effect is the KV cache rows forward_attn
   writes as a side effect. Returns 0 on success, nonzero (no errbuf -
   same convention lz_forward_batch/lz_forward_verify use) on a NULL/
   range argument or a missing MTP head. */
int lz_mtp_prefill(const LZModel *m, LZRunState *s, const float *h_body_all,
                   const int *next_tokens, int n, int pos0);

/* Catch-up decode (independent audit finding + llama.cpp's own
   speculative.cpp precedent - see generate.c's lz_spec_round for the
   full derivation). Thin wrapper
   around lz_mtp_prefill, identical arguments and identical effect -
   exists ONLY so this specific call site (re-decoding an accepted
   draft span's MTP rows with the TARGET model's own verified hidden
   states, replacing what the draft loop wrote using chained/estimated
   hidden for every step after the first) can be timed/counted
   separately from lz_mtp_prefill's OTHER call site (the one-time prompt
   prefill in lz_generate_resume, before any round exists) - conflating
   the two would hide whether catch-up is running at all behind the
   prompt prefill's own much larger, one-time cost. See lz_debug_us_
   catchup / lz_debug_n_catchup below (forward.c) for the counters this
   produces. */
int lz_mtp_catchup(const LZModel *m, LZRunState *s, const float *h_body_all,
                   const int *next_tokens, int n, int pos0);

#endif
