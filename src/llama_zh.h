#ifndef LLAMA_ZH_H
#define LLAMA_ZH_H

/* llama98 engine public API.
   Model construction supports two families. The Qwen3.5/3.6 one
   (qwen3_5_text): GatedDeltaNet linear attention alternating with gated
   full attention + SiLU-gated FFN + RMSNorm. And the KunMoe variant
   (model_type "kunmoe"): KDA linear attention with per-channel decay,
   and/or a latent MoE FFN - see model.h, which has carried the fields
   for it since bin v4.
   KunMoe support is verified rather than assumed: the export tool
   takes model_type=kunmoe through its is_kunmoe branch, the resulting
   v4 bin passes `llama98 --check`, and greedy decoding produces fluent
   text.
   Forward, sampling and generation are shape-neutral and depend only on
   model.h's config. */

#include "model.h"
#include "forward.h"
#include "tokenizer.h"
#include "sampler.h"

/* Output callback: emits BYTES, not characters (encoding-neutral).
   Generation hands the decoded byte stream to the sink in order; the
   frontend is responsible for writing/display. */
typedef void (*LZTokenSink)(const char *bytes, int len, void *ctx);
/* return 0 to request stop (used by the GUI's stop button) */
typedef int  (*LZShouldContinue)(void *ctx);
/* Prefill progress. Called between prefill slices with how many prompt
   tokens have been forwarded and how many there are in total.
   `ctx` is the SAME pointer the sink and cont are given - one context
   for all three, which is the convention gui/worker.h already records
   for the first two; a fourth lifetime to keep straight buys nothing.
   Fires on whichever thread generation runs on, so a GUI handler
   records the numbers and lets its own timer draw them - it must not
   touch a control, the same rule the token path follows. */
typedef void (*LZProgress)(int done, int total, void *ctx);

/* Defined in forward.h: LZRunState, lz_forward.
   Defined in model.h: LZModel, LZModelConfig.
   Defined in tokenizer.h: LZTokenizer, lz_encode, lz_decode. */

#define LZ_MAX_STOP 8           /* max stop strings */
#define LZ_MAX_EOS  4           /* max termination tokens */
#define LZ_STOP_TAIL 64         /* tail bytes kept for stop-string matching */

/* Why a generation ended. The names map onto OpenAI's finish_reason:
   EOS and STOP both report "stop", LENGTH reports "length", CANCELLED is
   a client disconnect and has no OpenAI equivalent (the response is
   simply not completed). */
typedef enum {
    LZ_FINISH_EOS = 0,      /* a termination token was sampled */
    LZ_FINISH_STOP,         /* a stop string matched */
    LZ_FINISH_LENGTH,       /* max_new_tokens or the KV cache ran out */
    LZ_FINISH_CANCELLED     /* the cont() callback asked to stop */
} LZFinish;

typedef struct {
    LZSampleParams sample;      /* sampling & penalties; defaults in sampler.h */
    unsigned long long rng_seed;
    int   max_new_tokens;       /* <=0 means use the model's seq_len */

    /* Termination tokens. The real turn-ender in Qwen's vocab is
       <|im_end|>; look it up via lz_gen_opts_set_eos and fill in. */
    int   eos_ids[LZ_MAX_EOS];
    int   n_eos;

    /* Stop strings. Matched against generated output BYTE-wise; on hit,
       generation stops without emitting the string. A chat model needs
       this: it learns to write <|im_end|>, but may also emit
       "<|im_start|>user" and start answering itself.

       BORROWED, and borrowed for the WHOLE call - not just its start.
       Each entry is dereferenced again on every generated token (strlen
       + memcmp against the output tail), so a caller that frees or
       rewrites the buffer after handing it over reads freed memory on
       the very next token. That matters for an HTTP front end because
       stop[] comes out of the request body: parsing the JSON into a
       scratch buffer, filling stop[], then reusing that buffer for the
       response is the natural shape and it is wrong.
       Nothing here is retained after the call returns. */
    const char *stop[LZ_MAX_STOP];
    int   n_stop;

    /* Optional generated-token echo (resume prefix verification). When
       non-NULL, up to out_tokens_cap sampled token ids
       are appended here (EOS/stop hits are not recorded). The bytes the
       model emitted may re-encode to a DIFFERENT greedy-BPE token list;
       lz_generate_resume requires the KV prefix to match the new
       render's prefix TOKEN-wise, so a caller doing KV reuse should
       compare these against the next render's encoded prefix and fall
       back to full generation on mismatch. */
    int   *out_tokens;
    int    out_tokens_cap;
    /* How many entries were actually written. Needed as its own count:
       *out_n_tokens is the GENERATED count, which includes a terminating
       EOS and a stop-triggering token, and neither of those is recorded.
       Reading out_tokens[0 .. *out_n_tokens-1] therefore reads one stale
       slot past the end. Written on every call, 0 when out_tokens is
       NULL. */
    int    out_tokens_n;

    /* Why generation stopped, and how many tokens the prompt cost.
       Both are written on every successful call.

       An OpenAI-shaped response needs them and neither can be derived:
       finish_reason cannot be inferred from the token count (EOS on the
       very last allowed step gives the same count as running out of
       budget), and prompt_tokens is computed inside lz_generate and
       discarded, so without this field a caller would have to re-run the
       whole BPE pass just to count - which on a Pentium-class target is
       real work, and can disagree with what generation actually used.

       Written through a const pointer, like out_tokens above. That is
       safe here because the engine is single-request by design (one
       LZRunState, one LZGenOpts, serialized callers); sharing one
       LZGenOpts across concurrent generations was never supported. */
    int    out_finish;          /* LZFinish */
    int    out_prompt_tokens;
    /* Which stop[] entry matched, or -1. Only meaningful when
       out_finish == LZ_FINISH_STOP.

       vLLM reports this as `stop_reason` on the choice and clients use
       it to tell WHICH of several stop strings fired - finish_reason is
       "stop" for all of them, and for EOS too. It costs nothing here:
       the matcher already knows the index at the moment it decides. It
       is also the only way the stop path is observable from outside,
       which matters because the defect it exists for (a missed cross-
       token match) was silent from the wire. */
    int    out_stop;

    /* Prefill progress, and the only way to interrupt a prefill.
       NULL (the default) is off, and off MUST be bit-identical to this
       same struct that never carried the field - the same contract
       spec_k = 0 and look_width <= 1 already have.

       It exists because prefill was one batched lz_forward_batch call
       with no callback of any kind reachable from inside it: cont() is
       only reached from the per-generated-token path, so a long prompt
       on a Pentium could report nothing AND could not be stopped. The
       hook that reports progress is the hook that makes cancelling
       possible, which is why this is not merely an indicator. */
    LZProgress on_prefill;

    /* Speculative decoding (MTP draft head, generate.c's lz_spec_round).
       0 (default, lz_gen_opts_defaults) = off, and OFF MUST BE BIT-
       IDENTICAL to this same struct with spec_k left at 0 - that
       equivalence is what makes the feature safe to turn on. 1..
       LZ_SPEC_K_MAX (ops.h) is the draft depth; anything else is
       refused (LZ_ERR_SPEC_K_RANGE). Requires m->mtp bound
       (LZ_ERR_SPEC_NO_HEAD). Temperature is not refused: temp<=
       LZ_TEMP_FLOOR runs lz_spec_round (greedy phase 1), temp>LZ_TEMP_
       FLOOR runs lz_spec_round_temp (generate.c). Penalties are NOT
       required to be disabled
       (there is no LZ_ERR_SPEC_PENALTY refusal) - verify applies them
       to each row via an assumed penalty window instead of refusing;
       see lz_spec_round's own comment and generate.c's
       build_assumed_window.

       DEFAULT INTENTIONALLY DIFFERS FROM llama.cpp's own reference
       (its common_params_speculative::n_max = 3, i.e. speculative
       decoding ON by default there). This project's own rule from the
       start: MTP defaults OFF, and spec_k == 0 must be bit-identical
       to a build that never linked this feature at all. Not an
       oversight - llama.cpp is a general serving tool where speculative
       decoding is close to always a win on its target hardware; this
       engine's own cost accounting is the opposite of settled for the
       Win98 target family, so defaulting ON would ship an unverified
       performance claim as a default. A caller who wants speculative
       decoding turns it on explicitly, same as --spec K on the CLI. */
    int    spec_k;

    /* Depth-limited lookahead (generate.c lz_look_pick). A TEST KNOB
       added on the owner's instruction, with the same contract spec_k
       carries: 0 or 1 = OFF, and off MUST BE BIT-IDENTICAL to a struct
       that never set it.

       *** NOT BEAM SEARCH - see ops.h LZ_LOOK_W_MAX for the full
       distinction. In one line: a beam carries W hypotheses forward
       across the whole generation and can return W ranked sequences;
       this restarts at every token, rolls each candidate out GREEDILY,
       and returns one. Deliberately NOT named --beam: a name that
       overclaims gets read as a status.

       *** WHAT THIS IS FOR, AND WHY THE ANSWER IS NOT ASSUMED ***
       The hypothesis under test is that likelihood-maximizing decoding
       makes THIS model's repetition worse, not better. Measured on
       models/kunkun98-recover-r20 at a real looping position, the top
       token's per-token log-probability was -0.085 against -1.466 at an
       ordinary position: the loop is seventeen times "better" by exactly
       the quantity a greedy rollout maximizes. Measured end to end, five
       prompts: 0/5 loops without it, 5/5 with look_raw at 2:2 AND at
       4:2 (Fisher p = 0.0079). Searching harder walks into the loop more
       faithfully, and widening does not rescue it.

       Sequential, not parallel: W simultaneous branches would need W KV
       caches (3.3 MB each at this model's 149 slots), which the Win98
       target cannot spend. Each candidate is instead walked forward on
       the one cache and backed out through an LZStateCkpt. NOT through
       the SSM rollback ring - forward.h's own contract says the single
       token path passes "the same slot as both read and write source",
       so a rollout built from lz_forward_capture destroys the state it
       would roll back to, and the ring is depth 1 anyway without an MTP
       head. Cost: one checkpoint (1.51 MB), one save and W restores per
       emitted token, on top of W*D forwards.

       look_raw: apply NO penalties during the rollout. Off by default
       because this engine's decoder has penalties and a test that
       silently drops them is testing a different decoder; on, it
       reproduces the OBJECTIVE of vLLM's BeamSearchParams, which
       carries no repetition/presence/frequency fields at all (verified
       against refsrc/vllm/vllm/sampling_params.py) - the objective, not
       the algorithm, which is the comparison that was actually wanted.

       look_lp: the length-penalty exponent from HF/vLLM's
       cum_logprob / len^lp. At fixed depth every branch has the same
       length and this is the identity; it only bites when a branch hits
       EOS early, which is precisely the case it exists for. */
    int    look_width;
    int    look_depth;
    int    look_raw;
    float  look_lp;

    /* Early-stop / discard thresholds, matching llama.cpp's own knobs
       (common_params_speculative::p_min / n_min) - see
       lz_spec_round's own comment for exactly what each one does and
       why neither can change what the engine emits, only how much
       speculative work it attempts first. Both default to 0 (matching
       llama.cpp's own defaults exactly, unlike spec_k above): p_min=0.0
       never stops a round early, n_min=0 never discards one - so these
       two knobs, unlike spec_k itself, do not need a "this project
       defaults differently" justification. */
    float  p_min;
    int    n_min;
    /* Debug-only: deliberately skip restoring conv_state on a rejected
       round, reproducing the exact failure mode where "missing
       conv_state fails the gate and looks like a rollback bug". NEVER
       set outside a test that expects the resulting corruption - this
       is the KNOWN-BAD control the three-part gate requires, not a
       performance knob. */
    int    spec_debug_break_rollback;
    /* Debug-only: skip lz_generate_resume's MTP prompt prefill call
       entirely (generate.c) - the draft head starts every round blind.
       Built for ONE purpose: a same-binary, same-weights A/B control
       for whether prefill is doing anything measurable, mirroring
       spec_debug_break_rollback's own shape above (a real code path
       this flag disables on purpose, not a hypothetical). This is
       condition "A" of the three-way A/B/C control below (s->mtp_pos
       stays 0, KV cache stays empty). Measured result: prefill ON does
       raise alpha at both a short and a long -n, consistently - the
       ORIGINAL prediction that a long run's own accumulated history
       would make prefill's head start matter less did NOT hold (the
       long-run gap was comparable to or larger than the short-run one).
       NEVER set outside a test that expects this comparison. */
    int    spec_debug_skip_prefill;
    /* Debug-only: condition "B" of the same three-way control -
       advance s->mtp_pos to EXACTLY the value a full prefill would
       leave it at (anchor_pos's own +n_prompt-1), but WITHOUT calling
       lz_mtp_prefill at all, so the MTP's KV cache rows for those
       positions are never written (they stay at lz_state_reset's zero
       fill). Isolates whether prefill's alpha benefit is POSITIONAL
       (lz_mtp_draft_step's attention RoPE-rotates by s->mtp_pos, so a
       shifted starting counter alone could explain the whole effect) or
       CONTENT-driven (the KV rows the skipped lz_mtp_prefill call would
       have written actually carry signal).

       Measured result: B matches C EXACTLY - not just close, bit-
       identical per-prompt alpha in every short and long comparison
       measured. The benefit IS positional: lz_mtp_prefill's actual
       forward pass over the prompt (the part whose cost grows with
       prompt length) was not earning its keep in this measurement -
       the single assignment this flag performs reproduced its whole
       alpha benefit. The exact WHY (candidate: this model's large
       rope_theta=10,000,000 stretches RoPE's low-frequency
       disambiguation-poor region well past small position counts,
       weakening self-attention when the counter starts near 0
       independent of content) is a hypothesis, not a second measured
       fact - not independently verified.

       Mutually exclusive with spec_debug_skip_prefill in effect - if
       both are set, skip_prefill wins (see lz_generate_resume). NEVER
       set outside such a test. */
    int    spec_debug_prefill_pos_only;
    /* Debug-only: overrides spec_debug_prefill_pos_only's counter jump
       from the fixed n_prompt-1 to this value, when > 0 (0, the
       default, keeps the original n_prompt-1 behavior unchanged).
       lz_debug_mtp_attn_window can only NARROW the window down to
       whatever has actually accumulated (it clamps back to
       "no restriction" once the
       requested width exceeds pos+1) - it can choose how much of the
       real history to use, but cannot MANUFACTURE more than
       n_prompt-1 positions' worth of head start. Only this field can:
       it is the one lever that can test starting positions LARGER than
       n_prompt-1, still at zero extra bytes/weight-loads (the KV cache
       stays zero-filled either way - see spec_debug_prefill_pos_only's
       own comment) but NOT zero ALU (every later round's attention
       still scores however many rows are in its window - see
       lz_debug_mtp_attn_rows, forward.c, for the honest per-run cost
       count this asks for alongside alpha). Only takes effect when
       spec_debug_prefill_pos_only is also set. NEVER set outside that
       one investigation. */
    int    spec_debug_prefill_pos_value;
    /* Debug-only: when > 0 and < n_prompt-1, condition C's own
       lz_mtp_prefill call only computes REAL content for the LAST this-
       many prompt positions - the head positions stay zero-filled, but
       s->mtp_pos still advances through the FULL n_prompt-1 span (see
       lz_generate_resume's own comment on this field). 0 (default)
       covers the whole prompt. Built after the
       spec_debug_prefill_pos_only result reversed which condition
       (B or C) the real benefit comes from: this asks the ACTUAL
       product question directly - does prefilling only the prompt's
       tail reach full coverage's alpha at a fraction of the cost -
       rather than the mechanism question the other debug flags
       answer. NEVER set outside that investigation. */
    int    spec_debug_prefill_coverage;
    /* Debug-only: skip lz_spec_round's catch-up decode (generate.c).
       When skipped, an accepted span's non-first MTP rows keep the
       draft loop's own chained/estimated hidden forever. Same shape as
       spec_debug_skip_prefill above - a same-binary, same-weights A/B
       control, not a performance knob - built because the original
       row-offset math was wrong in a way that DECREASED alpha (row 0
       does not need catching up, and overwriting it anyway discarded a
       verified-correct row - see lz_spec_round's own comment) and only
       a real on/off toggle at test time could have caught that
       reliably; a one-off hand-edited measurement did, but is not
       repeatable. Measured result (6 falsify prompts, -n 150, k in
       {2,3,4,5}): alpha is flat to positive with catch-up on in every
       case (+0.0010 to +0.0097), never negative. NEVER set outside a
       test that expects this comparison. */
    int    spec_debug_skip_catchup;

    /* Acceptance-rate accounting, written on every exit when spec_k > 0
       (all three stay 0 when spec_k == 0, like the rest of this
       struct's out_* fields). alpha = out_spec_accepted /
       out_spec_draft_tokens - this is the FIRST place in the engine
       that number is measurable ON-POLICY (drafts checked against the
       SAME model's own real generation, not corpus text - corpus-based
       measurement is off-policy and invalid). Anchor
       tokens (always kept regardless of the draft head) are not
       counted as draft tokens; only the k proposed-by-the-head tokens
       per round are. */
    int    out_spec_rounds;         /* how many speculative rounds ran */
    int    out_spec_draft_tokens;   /* sum of k over every round */
    int    out_spec_accepted;       /* sum of n_accept over every round */
} LZGenOpts;

/* Fill in defaults. **Callers must call this BEFORE changing any
   fields** - do not memset to zero yourself, repetition_penalty would
   be 0.0 and dividing a positive logit by 0 yields inf.
   Note defaults contain no eos_ids: those depend on the concrete vocab
   and are filled by lz_gen_opts_set_eos. */
void lz_gen_opts_defaults(LZGenOpts *o);

/* Look up termination tokens like <|im_end|> in the tokenizer and fill
   them in. Missing ones are ignored. */
void lz_gen_opts_set_eos(LZGenOpts *o, LZTokenizer *t);

/* ------------------------------------------------- speculative decoding */

/* Greedy speculative acceptance. The draft head that would feed it does
   not exist yet (this model has no mtp.* weights), so this is the
   reserved interface plus the one piece of it that can already be judged
   right or wrong.

   draft[i]  the draft head's i-th proposed token
   targ[i]   the VERIFY pass's argmax at the same position
   bonus     the verify pass's argmax one position past the last draft -
             it comes out of the same forward, so it costs nothing
   out       receives the accepted prefix; needs n_draft + 1 slots

   Returns how many DRAFT tokens were accepted, 0..n_draft. The number
   EMITTED is always accepted + 1, i.e. between 1 and n_draft + 1 - a
   verify pass never produces nothing, because the corrected token at the
   first mismatch is itself output.

   Three properties, all taken from vLLM's rejection_greedy_sample_kernel
   rather than derived:

     1. out[] is ALWAYS the verify pass's argmax, never the draft token.
        On accepted positions the two are equal by definition, so this
        looks like a free choice and is not: writing the draft token
        instead would make the output depend on the draft head, and the
        whole reason greedy speculation is safe is that it CANNOT.
        This is what makes "spec on == spec off" a bit-identity claim.
     2. Rejection LATCHES. The first mismatch takes the target's token
        and everything after it is discarded, even if a later draft
        token would have matched - the model never saw that prefix.
     3. All accepted -> append the bonus token. That is where the
        1 + a + ... + a^k throughput formula comes from. */
int lz_spec_accept(const int *draft, const int *targ, int n_draft,
                   int bonus, int *out);

/* Coupled (textbook) accept/residual-resample test - the temp>0 analog
   of lz_spec_accept above (phase 2 of temp>0 speculative decoding).
   Full contract, including why the return value uses -1
   rather than 0 as its error sentinel (0 is a legitimate token id
   here, unlike lz_spec_accept's own COUNT return), is at the
   implementation in generate.c, next to lz_spec_accept itself - same
   "reserved interface, exercisable standalone via ctypes" positioning,
   deliberately model-independent (p, q, u, residual_rng are all
   caller-supplied arrays/values, not read from an LZRunState). */
int lz_spec_accept_temp(const float *p, const float *q, int x, int vocab,
                        float u, unsigned long long *residual_rng,
                        int *out_accepted);

/* One speculative round: draft k tokens with the MTP head and verify
   them all in one batched forward, greedy only (temperature 0 - see
   LZGenOpts.spec_k). Penalties, if the caller's sampler has any active,
   are applied to each verify row via an ASSUMED window (samp parameter
   below) rather than being refused. Contract and the state-rollback
   argument (why the SSM/conv ring - forward.h's s->ssm_slot - is the
   rollback mechanism rather than a checkpoint restore, and why the
   MTP's own KV cache and position counter roll back by simply not
   advancing past the accepted count, never by copying anything) are in
   generate.c, next to the implementation - this is the reserved
   interface plus the piece that can be exercised standalone via ctypes,
   same shape lz_spec_accept above already has.

   Requires s->mtp_chain to already hold the draft head's seed hidden
   state for `anchor_pos`: the caller does NOT need to forward
   anchor_pos's token via lz_forward_capture on every call - matching
   llama.cpp's own speculative.cpp, this function's OWN previous
   invocation seeds it for the caller from that round's verify batch,
   row n_accept, at this same function's own tail; see the comment
   there. lz_forward_capture (forward.h) is only needed before this
   function's very FIRST call in a generation, or after a discarded
   round - lz_generate_resume's own `have_seed` tracks which case
   applies).

     anchor       the model's own, non-speculative prediction for
                  position anchor_pos+1 (from that lz_forward_capture
                  call) - NOT YET forwarded into s. Always kept
                  regardless of what the draft head proposes next.
     anchor_pos   absolute position of the last token already forwarded
                  into s (anchor's logits came from forwarding there).
     k            draft depth UPPER BOUND, 1..LZ_SPEC_K_MAX (ops.h) - the
                  ACTUAL depth drafted this round may be less, see p_min.
     p_min        stop drafting the moment the head's own top-1
                  confidence drops below this (0.0 default = never stop
                  early - see LZGenOpts.p_min).
     n_min        if fewer than n_min tokens got drafted (after p_min),
                  discard the WHOLE draft rather than pay for a verify
                  batch (0 default = never discard - see LZGenOpts.n_min).
                  A discarded round returns 0 (see below), not 1..k+1.
     samp         the caller's own LZSampler (read-only here - this
                  function never calls lz_sampler_observe, that stays
                  the caller's job once acceptance is known), used to
                  build each verify row's ASSUMED penalty window
                  (generate.c's build_assumed_window/verify_row_argmax)
                  when samp->p has any non-identity penalty active.
                  NULL disables this entirely - no window is built,
                  verify's argmax runs on raw logits (every existing
                  ctypes-direct caller passes NULL and is unaffected).
     debug_break_rollback  test-only, see LZGenOpts's own field comment.
     debug_skip_catchup    test-only, see LZGenOpts.spec_debug_skip_
                  catchup's own field comment.
     out          receives the round's newly accepted/corrected tokens,
                  NOT including anchor (the caller emits/records that
                  one itself, identically to the non-speculative path -
                  see lz_generate_resume). Needs k+1 slots. On a
                  discarded round (return 0), out[0] is set to `anchor`
                  purely so `out[*out_n_accept]` stays a valid "next
                  pending token" expression for the caller - it must
                  NOT be emitted again, anchor already was.
     out_n_accept how many of the k_eff draft tokens were accepted,
                  0..k_eff (0 on a discarded round too).
     out_new_pos  absolute position of the last token this call actually
                  forwarded into s (anchor_pos + 1 + *out_n_accept;
                  unchanged at anchor_pos on a discarded round, since
                  nothing new was forwarded).
     out_k_eff    how many draft steps were actually taken this round,
                  0..k (== k unless p_min or a forward failure cut it
                  short - a forward failure returns before this is set,
                  see below). This is the right denominator for an
                  acceptance-rate count: *out_n_accept / *out_k_eff, NOT
                  the requested k, which p_min can make an overcount.

   Returns the count written to out[] (1..k+1) on an ordinary round, 0 if
   n_min discarded the draft (see above - not an error), or a NEGATIVE
   LZErr code on failure (m->mtp missing, k out of range, or a forward
   failure - out_k_eff is NOT written on this path) - errbuf is filled
   the same way every other LZErr-returning function in this header
   fills it.

   No LZStateCkpt parameter - per-round rollback goes through the
   SSM/conv ring instead of caller-owned checkpoint scratch;
   LZStateCkpt's only caller is LZPrefixCache. */
int lz_spec_round(const LZModel *m, LZRunState *s,
                  int anchor, int anchor_pos, int k,
                  float p_min, int n_min, const LZSampler *samp,
                  int debug_break_rollback, int debug_skip_catchup,
                  int *out, int *out_n_accept, int *out_new_pos,
                  int *out_k_eff, char *errbuf, int errlen);

/* temp>0 speculative round - the coupled-sampling analog of lz_spec_
   round above (phase 2 of temp>0 speculative decoding). DELIBERATELY a
   separate function, not a branch inside lz_spec_round itself - see the
   implementation's own comment in generate.c for why (the short
   version: it keeps lz_spec_round's own source, and therefore greedy
   speculation's hard bit-identity gate, structurally untouched by
   anything written for the temp>0 path). samp is REQUIRED here (NULL
   is not a valid "greedy
   fallback" the way it is for lz_spec_round - see that parameter's own
   comment at the implementation) and NOT const: every row's accept
   test and residual resample draw from and advance samp->rng_state.
   Parameter/return/error conventions otherwise match lz_spec_round
   exactly; full contract at the implementation, next to it. */
int lz_spec_round_temp(const LZModel *m, LZRunState *s,
                       int anchor, int anchor_pos, int k,
                       float p_min, int n_min, LZSampler *samp,
                       int debug_break_rollback, int debug_skip_catchup,
                       int *out, int *out_n_accept, int *out_new_pos,
                       int *out_k_eff, char *errbuf, int errlen);

/* Full generation. prompt is given in bytes (UTF-8); it is encoded to
   tokens and forwarded token by token. Only the post-prompt generation
   is emitted; stops on EOS / stop string / cont. Errors go to errbuf,
   never printf, never exit.

   RETURN: 0 (LZ_ERR_OK) on success, otherwise the LZErr code (err.h).
   Not a bare 1 - the code is what lets a caller act on the failure
   without parsing errbuf, whose text is bilingual and switches at
   runtime via lz_set_error_lang. An OpenAI-shaped endpoint maps:

     LZ_ERR_PROMPT_LONG    400  request exceeds the context window
     LZ_ERR_STOP_LONG      400  a stop string is over LZ_STOP_TAIL bytes
     LZ_ERR_PROMPT_ENCODE  400  input the tokenizer cannot segment (an
                                unbroken letter run over LZ_TK_MAX_WORD
                                byte-level codepoints - about 1365 CJK
                                characters with no punctuation)
     everything else       500  ours: allocation, sampler init, forward

   The first three are all "the client must change the request", but they
   need DIFFERENT changes - trimming history fixes the first and does
   nothing for the other two - which is why they stay distinct codes
   rather than one "bad request".

   prompt_bytes is BORROWED and read only during the initial encode; it
   is not retained after the call. opts->stop[] is borrowed for longer -
   see its own note above. */
int lz_generate(const LZModel *m, LZTokenizer *t, LZRunState *s,
                const char *prompt_bytes, int prompt_len,
                const LZGenOpts *opts,
                LZTokenSink sink, LZShouldContinue cont, void *ctx,
                int *out_n_tokens, double *out_elapsed_ms,
                char *errbuf, int errlen);

/* Multi-turn resume. start_pos=0 behaves exactly like
   lz_generate (it is a thin wrapper around this). start_pos>0: the KV
   cache / SSM / conv state in s is KEPT, and the new prompt is
   forwarded starting at absolute position start_pos - the caller
   guarantees tokens [0, start_pos) in s already match the new prompt's
   prefix (byte-identical history, e.g. via lz_chat_norm_history).
   Caller bookkeeping: start_pos += n_prompt + generated each turn, and
   start_pos + n_prompt <= s->seq_len (else LZ_ERR_PROMPT_LONG).
   Parity gate: resume output must equal full re-generation bit for
   bit (fixed-seed multi-turn dialogue). */
int lz_generate_resume(const LZModel *m, LZTokenizer *t, LZRunState *s,
                       int start_pos,
                       const char *prompt_bytes, int prompt_len,
                       const LZGenOpts *opts,
                       LZTokenSink sink, LZShouldContinue cont, void *ctx,
                       int *out_n_tokens, double *out_elapsed_ms,
                       char *errbuf, int errlen);

/* Same generation loop, ONE more argument: `ins`, filled with what the
 * sampler's own shortlist looked like for the LAST TOKEN SAMPLED so
 * far - overwritten every time a new one is (never on a step that
 * digests a prompt token instead of sampling one; see lz_sample_ex's
 * own comment in sampler.h for what "the shortlist" means at each of
 * ITS exit points) - or left completely untouched when NULL.
 *
 * lz_generate_resume IS lz_generate_resume_ex(..., NULL) - a thin
 * wrapper, not a parallel implementation. Instrumented at BOTH of this
 * function's own lz_sample call sites - the ordinary per-token path
 * and speculative decoding's own anchor-token path - so a caller
 * watching `ins` sees every token this function actually samples, not
 * only the ones the non-speculative path produces.
 *
 * ins == NULL costs nothing beyond lz_sample_ex's own NULL branch at
 * each of those two call sites - no extra allocation, no extra pass.
 * The CLI, the HTTP endpoint, and every existing lz_generate_resume/
 * lz_generate caller keep calling those two names unchanged and never
 * pay for a caller that might want to watch; only gui/session.c's own
 * lz_gui_session_job calls this one directly, with a caller-owned
 * LZInspect embedded in LZGuiSession. */
int lz_generate_resume_ex(const LZModel *m, LZTokenizer *t, LZRunState *s,
                          int start_pos,
                          const char *prompt_bytes, int prompt_len,
                          const LZGenOpts *opts,
                          LZTokenSink sink, LZShouldContinue cont, void *ctx,
                          int *out_n_tokens, double *out_elapsed_ms,
                          char *errbuf, int errlen, LZInspect *ins);

/* ---------------------------------------------------- prefix reuse */

/* One conversation's reusable prefix: a recurrent-state checkpoint plus
   the token sequence it covers.

   Why this lives in the engine and not in each frontend: deciding
   whether a checkpoint may be reused is not a one-liner. It has to
   establish that splitting the render re-encodes token-exactly (BPE is
   context-sensitive across segment boundaries, so
   encode(head)+encode(tail) is not generally encode(head+tail)), that
   the checkpoint's tokens really are a prefix of this render's tokens,
   and that the run state has not been reset underneath it. Each of
   those failing silently produces a plausible-looking wrong answer, not
   an error. The CLI, the GUI and an HTTP front end would each have
   written their own version - and this project already learned that
   lesson once, which is why lz_chat_norm_history exists instead of the
   GUI re-implementing the template's normalization.

   Keyed on the TOKEN SEQUENCE, not on a position. That distinction is
   what makes reuse possible at all here: position-based resume is ruled
   out because the generation-prompt segment sits in turn N's forwarded
   stream and is absent from turn N+1's render, but the tokens BEFORE it
   match, and that is what gets matched. */
typedef struct {
    LZStateCkpt ck;
    int  *tok;          /* the covered token prefix */
    int   cap;          /* allocated entries in tok / scratch */
    int   n;            /* tokens covered; 0 = nothing cached */
    int   bytes;        /* bytes of the render those tokens came from */
    int  *cur, *pre, *tail;   /* per-call scratch, grown with cap */
    int   have;         /* a checkpoint has been saved */

    /* Accounting, so "should there be more than one cache?" can be
       answered with traffic instead of intuition.

       The design is deliberate: a ring of CHECKPOINTS gets zero hits by
       construction, because a non-matching request calls lz_state_reset,
       which bumps epoch, which voids every checkpoint at once. A working
       slot has to be a whole LZRunState (the KV lives there) at 3.86 MB
       on a machine with about 22 MB to spare. Whether that is worth
       paying depends on whether conversations actually interleave,
       which is what these count. */
    long  n_calls;      /* prepare() invocations */
    long  n_hits;       /* reused a cached prefix */
    long  n_tokens_saved;   /* tokens not re-forwarded because of a hit */
    long  n_mismatch;   /* had a cache, it did not match this render */
    long  n_unsplittable;   /* the split was not token-exact; full path */
} LZPrefixCache;

/* Snapshot of the counters above. Zeroed fields are legitimate values,
   not "unknown" - a server that served nothing reports zeros. */
void lz_prefix_stats(const LZPrefixCache *pc, long *calls, long *hits,
                     long *tokens_saved, long *mismatch, long *unsplittable);

int  lz_prefix_init(LZPrefixCache *pc, const LZModel *m,
                    char *errbuf, int errlen);
void lz_prefix_free(LZPrefixCache *pc);
/* Drop the cached prefix (e.g. the conversation was cleared). Keeps the
   allocation. */
void lz_prefix_reset(LZPrefixCache *pc);

/* Bring `s` to the furthest state that `render` can reuse, and report
   how the caller should generate from there.
   `split` is the byte offset where the reusable part of `render` ends -
   for the Qwen chat template that is
   `render_len - strlen(lz_chat_gen_prompt_tail(enable_thinking))`.

   On return the caller ALWAYS does exactly this, reuse or not:

       lz_generate_resume(m, t, s, *out_start_pos,
                          render + *out_suffix_off,
                          render_len - *out_suffix_off, ...)

   because the fallback path returns start_pos = 0 and suffix_off = 0,
   and lz_generate_resume(0, whole render) is lz_generate. There is no
   second code path in the caller to get wrong.

   *out_reused receives how many tokens were skipped (0 = none).
   Returns 0 on success; a non-zero LZErr means the state is untouched
   and the caller should fall back to lz_generate itself. */
int  lz_prefix_prepare(LZPrefixCache *pc, const LZModel *m, LZTokenizer *t,
                       LZRunState *s, const char *render, int render_len,
                       int split, int *out_start_pos, int *out_suffix_off,
                       int *out_reused, char *errbuf, int errlen);

/* How many tokens of `pre[0..n_pre)` this cache would reuse. Read-only:
   touches neither the cache nor any state.

   This IS lz_prefix_prepare's check 2, factored out rather than copied -
   prepare calls it too. A session pool has to rank slots before it
   commits to one, and a second copy of the matching rule would be free
   to drift from the one that decides. */
int  lz_prefix_match(const LZPrefixCache *pc, const int *pre, int n_pre);

/* ---------------------------------------------------------- session pool */

/* N independent (LZRunState, LZPrefixCache) pairs, so two conversations
   can interleave without evicting each other.

   A slot is a WHOLE LZRunState, not a checkpoint. That is settled, and
   the reason is in LZPrefixCache's comment above: a ring of checkpoints
   scores zero hits by construction, because any non-matching request
   calls lz_state_reset and the epoch bump voids every checkpoint at
   once. The KV cache lives in the run state, so the run state is the
   unit. At seq 2048 that is 3.86 MB per slot - the whole reason this is
   a runtime knob defaulting to 1 rather than a constant.

   Sizing is the caller's call because it is a memory decision, not an
   engine one: the endpoint runs concurrency 1 and still benefits (two
   conversations alternating turns hit different slots), while a 64 MB
   Win98 box may not have 3.86 MB to spare at all. Use lz_pool_stats to
   decide with traffic rather than intuition - n_evict_useful counts the
   evictions that a bigger pool would have avoided. */
typedef struct LZSlot {
    LZRunState    st;
    LZPrefixCache pc;
    long          used;         /* LRU stamp; 0 = never used */
} LZSlot;

typedef struct {
    LZSlot *slot;
    int     n;                  /* slot count */
    long    clock;              /* monotonic LRU stamp source */
    int    *pre;                /* scratch for the ranking encode */
    int     pre_cap;
    /* Accounting, same discipline as LZPrefixCache's. */
    long    n_calls;
    long    n_hits;             /* a slot matched something */
    long    n_cold;             /* no slot matched; took a free/LRU one */
    long    n_evict_useful;     /* evicted a slot that HAD a live prefix -
                                   i.e. a bigger pool would have kept it */
} LZSessionPool;

int  lz_pool_init(LZSessionPool *p, const LZModel *m, int n_slots,
                  int seq_len, char *errbuf, int errlen);
void lz_pool_free(LZSessionPool *p);

/* Pick the best slot for `render` and prepare it. Same contract as
   lz_prefix_prepare, plus *out_state - the caller then does exactly

       lz_generate_resume(m, t, *out_state, *out_start_pos,
                          render + *out_suffix_off,
                          render_len - *out_suffix_off, ...)

   which is the identical shape to the single-state path, so there is no
   second call site to get wrong. */
int  lz_pool_prepare(LZSessionPool *p, const LZModel *m, LZTokenizer *t,
                     const char *render, int render_len, int split,
                     LZRunState **out_state, int *out_start_pos,
                     int *out_suffix_off, int *out_reused,
                     char *errbuf, int errlen);

void lz_pool_stats(const LZSessionPool *p, long *calls, long *hits,
                   long *cold, long *evict_useful, long *bytes);

#endif
