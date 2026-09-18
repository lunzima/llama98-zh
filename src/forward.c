#include <math.h>
#include "lz_mathf.h"   /* lz_powf/lz_sinf/lz_cosf: float transcendentals, no libm, bit-identical x86/ARM */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"     /* includes compat.h for lz_time_ms (LZ_PROF* macros) */
#include "lz_int.h"    /* lz_i64: the 64-bit type, portably */
#include "err.h"
#include "ops.h"
#include "ops_quant.h"   /* q8_round/q8_amax/pow2f/LZ_Q8_*, lz_rsqrt_float_body */
#include "fwht.h"        /* lz_fwht_i32, the SubLN Hadamard kernel */



/* Per-layer intermediate tap hook for differential testing. The hook (and
   its LZ_TAP_OFF companion) now live in forward.h so the attn path's own TU
   sees the same definition; see there for the two-consumer rationale. */
/* Debug-only investigative probe for the dilution-vs-position
   question on the MTP prefill A/B/C study. Scales the MTP block's own
   attention-branch output before the residual add
   in lz_mtp_draft_step, on EVERY draft step. 1.0f (default) skips the
   multiply entirely - an exact no-op on the hot path, same "zero cost
   unless opted in" shape LZ_TAP already has in this file. Not part of
   the public API: no LZGenOpts field, no llama98.def/.map entry -
   reached only via an `extern` declaration in cli_main.c's own
   --spec-debug-attn-scale parsing. NEVER set outside that one
   investigation. */
float lz_debug_mtp_attn_scale = 1.0f;

/* Debug-only investigative probe, part 2 (the "decoupled"
   experiment - same report as lz_debug_mtp_attn_scale above): when > 0
   and forward_attn's own `layer` argument is the MTP
   sentinel (LZ_MTP_CACHE_LAYER, negative), restrict the MTP's own
   scoring/softmax/weighted-sum to just the last N positions
   (t in [pos-N+1, pos]) while `pos` ITSELF - and therefore RoPE's
   rotation angle - is left completely unchanged. This is the one lever
   the shipped code has no way to pull: `pos` alone controls BOTH the
   RoPE angle AND the attention window width (window == pos+1, always -
   see forward.c's own score loop), so distinguishing "the counter
   value matters" from "the window width matters" needs decoupling
   them, not just comparing conditions A/B/C which move both together.
   0 (default) is an exact no-op - never taken, zero cost. NEVER set
   outside this one investigation; body layers (layer >= 0) never read
   this at all, so a normal (non-MTP) forward pass is unaffected
   regardless of this value. */
int lz_debug_mtp_attn_window = 0;

/* Debug-only investigative probe, part 3 (reminder: a wider MTP attention window is zero extra
   BYTES/weight loads, but NOT zero ALU - the score and weighted-sum
   loops both scale with however many rows get attended to, same
   "count bytes but not compute" trap that has reversed a conclusion
   here before). Running total of MTP-only score-loop rows
   actually scanned (summed as `pos - win_t0 + 1`, once per head, every
   time forward_attn runs with `layer < 0`); body layers never touch
   this. Read directly rather than estimated, so an alpha gain from a
   wider starting position can be weighed against its real ALU cost,
   not just its byte cost (which is always zero here - the whole point
   of skipping lz_mtp_prefill's own forward pass). Zero unless this
   specific investigation reads/resets it; body forward passes are
   provably unaffected since they never increment it. */
lz_i64 lz_debug_mtp_attn_rows = 0;

/* Debug-only investigative probe, part 4. Microseconds spent inside
   each of a speculative round's three pieces.

   Why a timer instead of more arithmetic: the round's cost was derived
   six different ways from byte counts and CLI deltas, and six
   hypotheses about where it goes were each refuted by the next
   measurement (the draft head, lz_matmul_w's "slow path", unquantized
   mtp.* tensors, a redundant capture forward, an all_logits per-token
   lm_head, the per-round checkpoint). What IS measured: on a Ryzen
   5800X with Qwen3.5-0.8B, a round costs 3.78x one plain forward, of
   which the draft step is 0.32x - exactly what its bytes predict - and
   Verifying two tokens costs 2.46x, where the same accounting says
   ~1.5x. Prefill batches the same trunk at 0.54x per token, so the
   amortization exists and this path is not getting it. That gap is the
   only thing left worth chasing, and it will not be found by dividing
   more totals.

   Integer microseconds, not float seconds: %f is barred inside the
   PC=24 region, and these are read by cli_main.c which
   may print them; an lz_i64 crosses that boundary safely. Zero cost
   when nothing reads them, and body-only forwards never touch the two
   MTP ones. */
lz_i64 lz_debug_us_verify = 0;
lz_i64 lz_debug_us_draft = 0;
lz_i64 lz_debug_us_capture = 0;

/* Call-count companion to lz_debug_us_capture, for the "capture skipped,
   not just cheap" gate: a removed-but-still-called
   lz_forward_capture and a genuinely removed one look identical on any
   pure timing/output comparison if the removed call happened to be fast
   that run - this counter is what actually distinguishes them, same
   role lz_debug_mtp_attn_rows plays for the attention-window change. */
lz_i64 lz_debug_n_capture = 0;

/* Positive controls for four more int-pipeline switches. The v_proj /
   q8_i16 / q8_int / swiglu / conv-sig set already has them and the
   reason is stated at their print site: the int and float arms are
   bit-comparable, so an output comparison can never separate "the exit
   ran" from "the exit was refused". LZ_BVEC_I16, LZ_KLAT_I16,
   LZ_MLAT_I16 and LZ_MLAT_REUSE were registered in
   switch_matrix_gate and int_pipeline_arm_gate without one, so both
   gates proved their arms BUILD and RUN and neither could prove either
   arm did anything. mlat_quant is the pair for reuse specifically:
   with it on that number is below mlat's own, and with it off they are
   equal, which is what a compiled-out guard looks like from outside. */
lz_i64 lz_debug_bvec_i16 = 0;
lz_i64 lz_debug_klat_i16 = 0;
lz_i64 lz_debug_mlat_i16 = 0;
lz_i64 lz_debug_mlat_quant = 0;

/* Timer + call-count pair for lz_mtp_catchup, same shape as the capture
   pair above and same reason: the explicit requirement is a POSITIVE
   control proving catch-up decode actually
   ran, not just a plausible-looking diff - "removed" and "still there
   but never called" are indistinguishable without a counter (the
   pos_only dead-code lesson, see lz_debug_mtp_attn_rows's own history).
   lz_debug_us_catchup is expected to be small - lz_mtp_catchup skips
   both the FFN half and lm_head (lz_mtp_prefill's own comment), so it
   is cheaper per token than even a draft step, let alone a full
   forward. */
lz_i64 lz_debug_us_catchup = 0;
lz_i64 lz_debug_n_catchup = 0;

/* --kv-rot on|off. Default OFF, and that default is a measurement, not a
   guess: with the current Q8_0 group-32 KV cache the rotation costs PPL
   on every corpus tried (Qwen3.5-0.8B, temp 0 so the
   comparison is exact rather than sampled):

     TinyStories-en 1024 tok   5.7915 -> 5.7986   +0.123%
     fineweb-edu    2048 tok  14.8680 -> 14.8798  +0.080%
     cci3-hq        1024 tok  48.3017 -> 48.4461  +0.299%

   Why it can lose here even though llama.cpp PR 21038 measures a gain:
   Q8_0 already carries ONE ABSMAX SCALE PER 32 ELEMENTS, so it is
   already adapting to local variance. The raw K/V rows are sparse - a
   few large coordinates, many near-zero - and the near-zero groups get a
   tiny scale and excellent resolution. Spreading the energy evenly gives
   every group the same middling scale, which helps the outlier group and
   hurts all the others. The rotation pays when the quantizer CANNOT
   adapt locally, or when the bit width is low enough that outliers
   dominate - which is the 4-bit case, not this one.

   THE 4-BIT CASE CONFIRMS THAT READING. Same harness, f32 KV as the
   reference arm, PPL relative to it:

     corpus            q4 rot-off   q4 rot-on   rotation removes
     TinyStories-en      +6.720%     +0.623%        90.7%
     cci3-hq (zh)       +21.354%     +1.701%        92.0%

   So the rotation is not a wash that happened to lose - it is worth
   roughly a factor of eleven on the format it was designed for, and
   nothing on the one that was already adapting locally. --kv q4 turns it
   on for exactly this reason (cli_main.c).

   Default OFF still, because the default cache is Q8. Off must reproduce
   the pre-rotation engine bit for bit. */
int lz_kv_rot_enable = 0;

/* Positive control for the rotation. "Rotation preserves dot products"
   makes on-vs-off look identical for the wrong reason as easily as the
   right one, so the gate cannot be an output comparison alone - this
   counts head-rotations actually performed. */
lz_i64 lz_debug_n_kv_rot = 0;

/* Positive control for --batch, same reason as the one above. Batching
   is defined to be bit-identical across widths, so an output comparison
   can never tell "the knob works and is exact" from "the knob was never
   read" - and the latter is what a compile-time-only LZ_BATCH_MAX
   looked like from the CLI. Counts forward_chunk calls: prefilling n
   tokens at width T must take ceil(n/T) of them, which is a number the
   width cannot fake. */
lz_i64 lz_debug_n_chunks = 0;

/* Positive control for skip_logits. Prefilling n tokens at width T must
   run lm_head ONCE, not ceil(n/T) times - and since the skipped chunks'
   logits were never read, output comparison cannot see the difference
   any more than it can see the batch width. */
lz_i64 lz_debug_n_lmhead = 0;

/* Positive control for the sink+window row skip. Counted in the SKIP
   branch, not in the scoring loop: on the default path (no window) the
   branch is never taken, so this costs a predictable compare and no
   increment, and a broken skip reports 0 rather than reporting the
   count it was supposed to achieve. A counter derived from pos and
   attn_win would have been the fake kind - right by construction even
   if the loops still walked every row. */
lz_i64 lz_debug_attn_skip = 0;

/* Positive control for --ced. Counted at the SKIP ITSELF, in
   forward_chunk's layer loop, because the whole claim of that knob is
   that its output is unchanged: identical logits are exactly what a
   --ced that never fired also produces, so the comparison on its own is
   a signal that is always on. This counter is the other half - a run of
   --ced on whose logits match AND whose count is zero did not skip
   anything, and the gate fails it there rather than recording a
   speedup nothing performed. */
lz_i64 lz_debug_ced_skipped = 0;

/* Positive control for CED bounded replay's reset (lz_ced_replay_reset,
   below), same shape as lz_debug_ced_skipped above and for the same
   reason: a reset that fires zero times leaves the state untouched,
   which reads identically to "the knob is off" from the logits alone.
   Counted once per LAYER the loop actually resets (not once per call),
   so it also catches a ced_split boundary that resolves to zero
   qualifying layers. */
lz_i64 lz_debug_ced_replay_layers = 0;

/* Positive control for the V projection's int16 exit (LZ_VPROJ_I16).
   Counted at the CONSUMER - inside the branch that hands vtmp_i16 to
   lz_quantize_q8_i16 - not where the projection wrote it: an exit that
   was taken and then read from the wrong buffer would still increment a
   producer-side counter, and the int16 path is bit-comparable to the
   float one, so the logits cannot tell the two apart on their own.
   Elements, not calls, so it is the number the conversion arithmetic
   uses directly. */
lz_i64 lz_debug_vproj_i16 = 0;

/* Positive control for the decay gate's integer chain (LZ_KGATE_EXP_I),
   same role and same reason as the counter above: the integer and float
   chains compute the same function to within the fixed tier's own
   drift, so no output comparison can tell "the chain ran" from "both
   folds refused and it silently kept the float one". Elements, so the
   number is the one the conversion arithmetic multiplies.

   _exp_s and _sig_slo/_sig_shi are the two folds' exponents as actually
   built from this checkpoint's constants - the derivation says they
   should be 31 and 29..42, and a value outside that says the fold met a
   constant the measurement never saw. _fold_no counts refusals: it must
   be 0 whenever the chain was asked for, and a nonzero there is the one
   failure mode that otherwise looks exactly like success. */
lz_i64 lz_debug_kgate_exp_i = 0;
lz_i64 lz_debug_kgate_fold_no = 0;
int lz_debug_kgate_exp_s = -1;
int lz_debug_kgate_sig_slo = 9999;
int lz_debug_kgate_sig_shi = -9999;

/* --kv f32: keep the KV cache unquantized. Reference arm only, see
   forward.h's kf32 comment for why it has to exist. */
int lz_kv_kfmt = LZ_KVF_Q8;
int lz_kv_vfmt = LZ_KVF_Q8;

/* --kv-ref k|v: keep ONE side unquantized while the other stays on the
   selected format. Decomposition instrument, not a mode.

   It exists because the aggregate number cannot say where the damage is.
   The 4-bit cache costs +0.623%% PPL on English and +1.701%% on Chinese,
   and the stage-2 residual correction - which touches KEYS only - fixes
   a quarter of the English gap and none of the Chinese one. Either the
   Chinese loss lives in the VALUE reconstruction, in which case every
   further hour spent on key sketches is spent on the half that is not
   broken, or it does not, in which case the correction has a different
   problem. Those two look identical from the aggregate. */

/* --kv-rot-v N: Hadamard width for the VALUE side, 0 = the default 64.
   Split out from the key width because the default 64 comes from
   llama.cpp's own tuning ("using smaller rotation matrices for V seems
   beneficial"), measured against ITS value quantizer - per-32 absmax on
   a q4_0 grid. Ours is a per-row norm against a Gaussian codebook,
   which wants isotropy far more badly, so their tuning does not
   necessarily transfer. Swept on this engine (cci3-hq 1024 tok, f32
   reference):
     vrot=64   +1.701%      vrot=128  +2.234%      vrot=256  +1.890%
   and 64 wins here too. Wider mixing helps K and hurts V: V's
   contribution is a weighted SUM, so its reconstruction error adds up
   coherently across rows, and a wider rotation smears each row's error
   over every output coordinate instead of leaving it local. K only ever
   produces a scalar score, where spreading is pure benefit.

   The knob stays because that asymmetry is worth being able to re-check
   on a different model, not because 64 is in doubt. */
int lz_kv_rot_v_dim = 0;

/* --rope-dynamic-ntk on|off (cli_main.c). Off (default) leaves the RoPE
   table build below on its original, unconditional theta - bit-
   identical to the engine before this knob existed. On rescales theta
   ONCE per table build (not once per position - the effective base is
   a property of how far this table's positions reach, not of any one
   position inside it) by the NTK-aware ratio bloc97 described
   (https://old.reddit.com/r/LocalLLaMA/comments/14lz7j5/), applied only
   when the caller's requested seq_len exceeds lz_rope_orig_max:

     theta_eff = rope_theta * (seq_len / orig_max) ^ (rotary_dim / (rotary_dim - 2))

   docs/length-extrapolation-research.md S5.5 is why this formula and
   not YaRN/PI: the other two need the extrapolation target fixed in
   advance (a microtuned scale factor, or a training-time schedule for
   a KNOWN longer length), and this project's actual scenario is
   "config's max_position_embeddings got hacked past what was trained,
   target unknowable ahead of time" - exactly the case dynamic NTK
   recomputation was designed for. Below lz_rope_orig_max this is
   the identity by construction (the multiplier only differs from 1
   once seq_len/orig_max > 1), so raising --ctx within the trained
   range changes nothing even with the flag on. */
int lz_rope_dynamic_ntk = 0;

/* Training length the ratio above scales against. NOT the same number
   as either shipping model's declared max_position_embeddings (8192 /
   32768) - confirmed via engine/train_data.py's LONG_SEQ constant and
   docs/kunkun-ce-params.md S4 ("8192 declared, 1536+5120 actually
   trained"): both of today's checkpoints were trained with their
   longest packed stream capped at 5120 tokens, a fact about the
   CURRENT training pipeline (one shared LONG_SEQ constant), not a
   value either checkpoint's config.json records. A knob rather than a
   #define for that reason - a future retrain can change its own
   long-stream length without a recompile of this engine. */
float lz_rope_orig_max = 5120.0f;

/* Positive control for --rope-dynamic-ntk, same role as
   lz_debug_n_kv_rot above: "off" and "on but seq_len <= orig_max" are
   BOTH bit-identical to the unscaled table (the second one by the
   formula's own construction, not by the flag being ignored), so an
   output comparison alone cannot tell "the branch ran and correctly
   did nothing" from "the branch never ran". Counts frequency pairs
   (half the rotary width) whose theta was actually rescaled; 0 on any
   run that never crosses lz_rope_orig_max, on or off. */
lz_i64 lz_debug_n_rope_ntk = 0;

/* StreamingLLM attention sink + recent window (arXiv:2309.17453,
   github.com/mit-han-lab/streaming-llm). 0 = off, which is the default
   and must stay bit-identical to full attention.

   MEASURED (cci3-hq, full attention as the reference).

   1024 tokens - the sink itself:
     window 256, no sink   +43.764%
     sink 4  + window 252   +6.095%
     sink 16 + window 240   +5.351%
     sink 4  + window 124   +8.275%
   Four positions recover 86%% of what pure window attention destroys,
   which is the paper's central claim reproducing on a hybrid model.

   2048 tokens - how much span is enough:
     sink 16 + window  240  (span 1/8 of ctx)   +5.428%
     sink 16 + window  496  (span 1/4)          +2.630%
     sink 16 + window 1008  (span 1/2)          -0.142%
   At half the context the cost is NEGATIVE - marginally better than
   attending to everything. So the conservative operating point is
   window ~= ctx/2 with a sink of 16: it halves the attention span for
   free, and paying more than that buys the memory back at a real price.
   Stacked with K q4 / V q8 at ctx 2048 the state goes 40.7 -> 34.9 MB.

   Aggressive eviction is available and measured, but it is a choice to
   be made deliberately, not a default: 1/4 span costs 2.6%% and 1/8
   costs 5.4%%.

   The paper's position shift is deliberately not done here, and that
   choice is load-bearing.
   The reference caches PRE-RoPE keys and re-applies RoPE to the entire
   cache every step, renumbering positions by cache slot
   (pos_shift/modify_llama.py). Our cache holds POST-RoPE, QUANTIZED
   keys, so copying that would mean dequantizing the whole K cache,
   rotating, and requantizing once per token - ruinous, and it would
   compound quantization error every step. The numbers above keep the
   ORIGINAL positions and still show the effect.

   That works here for a reason that has a boundary: position shift
   matters once total length approaches what the model saw in training,
   and 1024 tokens against a 262144 max_position_embeddings is nowhere
   near it. For this project's actual shape - multi-turn chat at ctx
   2048 on a Pentium - original positions are fine. For a genuine
   streaming deployment past the training length, this measurement says
   nothing and the conflict above comes back. */
int lz_attn_sink = 0;
/* -1 = AUTO (the default), 0 = off, >0 = explicit.
   AUTO resolves against seq_len in lz_state_alloc; see the comment
   there and LZ_ATTN_WIN_FLOOR. `--attn-window 0` still turns eviction
   off completely, which the measuring arms depend on. */
int lz_attn_window = -1;

/* --ced: causal encoder/decoder split. 0 = off.
   ON, the layer that ends the stack does not run over the prefix in
   lz_forward_batch: its K/V for every position is written by
   forward_attn_kv_only (its own k_proj/v_proj applied to the encoder's
   exit hidden, which is also that layer's own input, so no new tensors
   and no approximation), while its queries, attention, output
   projection and FFN run for the final chunk only. The last position's
   logits and the whole KV cache therefore come out BYTE-IDENTICAL to
   the unmodified path - which is the gate, and the reason this is a
   scheduling knob rather than a precision trade.

   lz_forward_batch is the only entry that may run it. lz_forward_verify
   and lz_forward_batch_capture want every position's output, so they
   pass 0 and say so at their call sites.

   Only the final chunk's logits are ever observed by lz_forward_batch
   (see skip_logits there), which is what makes the saving free. */
int lz_ced = 0;

/* --dump-prefill: write the logits lz_forward_batch returns, once per
   call, so the LAST write of a run is the post-prefill row.

   --dump-logits cannot serve this. It belongs to the --tokens path,
   which forwards one token at a time through lz_forward - and that path
   has no prefix to skip, so CED cannot apply to it at all: every
   position's output is the thing being asked for. The two observables
   were therefore mutually exclusive until this existed, which left the
   batch path with no exact output to compare. */
const char *lz_dump_prefill_path = NULL;
lz_i64 lz_debug_prefill_dumps = 0;

/* --moe-topk: route to N experts instead of the exported
   num_experts_per_token. See forward_moe for why this is a knob and
   what it costs. 0 = use the model's own value. */
int lz_moe_topk = 0;

/* --moe-tau: router temperature applied to the mixing weights only.
   1.0 = off, and off is a branch that touches nothing (see
   lz_moe_route). It is the companion of lz_moe_topk rather than an
   independent knob - ops.h carries the measurement showing that the
   winning arm needs both, and that --moe-topk on its own can only
   select a losing one. */
float lz_moe_tau = 1.0f;

/* Absolute position -> cache slot. Identity unless the ring is on.

   The ring modulus is kv_ring = window + LZ_SPEC_K_MAX + 1, but only the
   most recent `window` positions are ever READ. Those spare slots are
   not slack, they are what keeps speculative decoding correct: a verify
   batch writes k_eff+1 positions past the confirmed end, and if the
   draft is rejected those writes must not have destroyed anything still
   live. With the guard, the slots a rejected draft clobbers hold
   t = pos+1-(window+guard) and below, which the read window already
   excludes - so `--spec K` output stays bit-identical to `--spec 0`,
   which is a gate this engine already has and would otherwise break. */
/* --profile: microseconds per phase of forward_chunk. Accumulators only;
   cli_main prints them; nothing below it writes to the console.

   Off by default because lz_time_ms() is a syscall-grade clock read and
   the inner phases here run 24 times per token - the measurement would
   otherwise cost more than some of the things it measures. */
int lz_prof_enable = 0;
float lz_prof_us[LZ_PROF_N];



/* kv_slot for t+1 given kv_slot for t, without the divide.
 *
 * The scoring and accumulation loops walk t contiguously, so the cache
 * row they read walks with it and wraps at the end of the ring. Calling
 * kv_slot per iteration spends an integer modulo by a runtime value on
 * every one - around forty cycles on the in-order cores in the target
 * family, and not pipelined. This spends a compare and an increment,
 * and the compare is taken once per ring period, so it is as
 * predictable as a branch gets.
 *
 * One rule for all four cases, checked against kv_slot's three branches:
 *   ring == 0        slot == t, and sink + ring == sink, which slot + 1
 *                    equals exactly once (at t + 1 == sink) - where the
 *                    answer is sink either way
 *   t + 1 <  sink    below the sink, slot == t, increments
 *   t + 1 == sink    kv_slot gives sink + 0; slot was sink - 1
 *   t + 1 >  sink    increments until slot + 1 reaches sink + ring,
 *                    then restarts at sink
 * Callers must seed with kv_slot(s, first_t) and advance in the loop's
 * INCREMENT clause, not its body - the row advances on iterations the
 * body skips with continue.
 *
 * Which path this is on, measured rather than assumed. The loops below
 * are the FALLBACK. lz_attn_score_q8 and its wsum twin take attn_sink
 * and kv_ring and do the row arithmetic themselves, and they are
 * selected whenever LZ_ATTN_NORING holds - which is win_t0 == 0, which
 * is everything except the MTP's own attention under an investigative
 * override. Counting kv_slot's modulo over a 120-token generation with
 * the ring on gives 363 both before and after this change, and 363 is
 * one per token per attention layer: the KV WRITE. The scoring loops
 * contributed nothing because they did not run.
 *
 * So this helps the arms that do run them - the float reference, and
 * any head_dim the fixed kernel declines - and is inert on the shipping
 * path. That is worth having and is not worth describing as a speedup. */
/* One row of the Q8 weighted sum: dst += a * dequantised(vt).
 *
 * The two accumulation arms in forward_attn - the high plane into out,
 * and the low plane into wsum_lo - had this loop written out
 * identically. They differ in base pointer, destination and start
 * position, not in the arithmetic.
 *
 * FIVE PARAMETERS, and that is the point. The whole loop does not get
 * extracted the same way: it would need base, scale, kvd, kvh, hd,
 * scale factor, att, win_t0, pos and slot - ten or more, on a target
 * with eight registers, where every parameter is a stack slot and every
 * use is a load. lz_generate_resume_ex is 58% data movement for exactly
 * that reason. This takes the piece whose parameter count stays small
 * and whose cost amortises over hd elements. */


/* One row of the Q8 score: q . dequantised(kt), grouped by 32.
 *
 * The scoring twin of wsum_q8_row, and the same four-parameter reason.
 * The high and low planes ran this letter for letter; the accumulation
 * ORDER is what has to survive, since a float sum reassociated is a
 * different number, and that is why the group loop comes along rather
 * than being left at the call sites. */






/* Stored scale groups for a row of N elements at group width GS. A GS of
   0 is "one scale for the whole row", which stores none - callers that
   pass a width straight from lz_act_gs rely on that, since it returns 0
   for a width this path cannot honour. Written once because ten sites
   had it inline and a reader had to check each one for the guard. */


/* The slot after SLOT in a ring of DEPTH. SLOT is already reduced, so
   the successor is a compare; recomputing (base + tk + 1) % depth spends
   a second runtime modulo to reach the same answer. Distinct from
   kv_slot_next, which walks the KV ring and its attention sink. */




/* One Q8 KV plane's weighted sum: dst = sum_t att[t] * dequant(v[t]).
 *
 * WRITTEN ONCE BECAUSE IT WAS WRITTEN TWICE. forward_attn had this body
 * for the high plane and again, under LZ_KV_2PLANE, for the low one -
 * the fixed fast path and the scalar walk both duplicated, differing
 * only in which buffer they write and which plane they read. The second
 * copy is compiled out of every shipping build, so it was a body nobody
 * built and nobody ran, kept in step with this one by hand. Now the
 * plane is an argument and the two callers reach the same code.
 *
 * noring IS SEPARATE FROM walk_t0 and the low plane is why. The fixed
 * kernel cannot express win_t0 - it walks from 0 - so the caller guards
 * it with LZ_ATTN_NORING. The low plane's scalar walk, though, starts at
 * a literal 0 rather than at win_t0, so its guard and its start were
 * already two different values in the original. Preserved rather than
 * unified: they may well be the same thing, but the path that would
 * prove it is not in any build that runs here.
 *
 * dead0/dead1 are parameters for the same reason the skip is spelled out
 * instead of using LZ_ATTN_SKIP_DEAD - that macro reads them from the
 * caller's scope and is #undef'd at the end of forward_attn, so it does
 * not exist out here. Same behaviour, including the debug counter. */


/* --kv q4: 4-bit KV rows, one scale per row per head. Mutually exclusive
   with the f32 arm; cli_main enforces that, and lz_state_alloc makes f32
   win if both are somehow set. */

/* --kv q4r2 (4-bit stage 1 plus a QJL sign-sketch of the key residual)
   is deliberately absent, along with --qjl-dim and --kv-rot-v - not
   for lack of a measurement, but because of one. Measured on
   models/kunkun98-recover-r20, 20,885 predicted positions against the
   --kv f32 arm, with state taken at 4017 slots:

     arm            bits/tok vs f32    state
       q8   q8        -0.007%          8.6 MB
       K q4r2 V q8    +0.055%          8.5 MB
       K q4  V q8     +0.163%          7.4 MB
       q4   q4        +0.553%          6.1 MB

   q4r2 buys 0.1 MB for a significant quality loss (paired t = 3.5
   against q8): DOMINATED on both axes, not a trade-off with a regime
   where it wins. A contested choice stays a knob because the target
   MACHINE family flips it; a KV format's quality-per-byte is a
   property of the MODEL and does not flip between a Pentium II and a
   K6-2, so that rule does not reach this case. A dominated option is
   not a knob, it is a trap for whoever reads the list next. */

/* Block-diagonal Hadamard over one head: hd/n independent n-wide
   transforms. n divides hd by construction (lz_state_alloc picks it). */



void lz_state_free(LZRunState *s) {
    if (!s) return;
    xfree(s->x); xfree(s->xb); xfree(s->xb2); xfree(s->kvshare_x); xfree(s->hb); xfree(s->hb2);
    xfree(s->logits);
    xfree(s->qg); xfree(s->qh); xfree(s->att); xfree(s->attn_out);
    xfree(s->attn_acc); xfree(s->attn_ss);
#if (LZ_ATTN_FIXED & 2)
    xfree(s->wsum_cbuf); xfree(s->wsum_cq);
#endif /* LZ_ATTN_FIXED & 2 */
    xfree(s->ktmp); xfree(s->vtmp);
    xfree(s->kq8); xfree(s->vq8); xfree(s->ksq); xfree(s->vsq);
    xfree(s->kf32); xfree(s->vf32);
    xfree(s->k4); xfree(s->v4); xfree(s->ks4); xfree(s->vs4);
#if LZ_KV_2PLANE
    xfree(s->kq8_lo); xfree(s->vq8_lo); xfree(s->att_lo); xfree(s->wsum_lo);
#endif /* LZ_KV_2PLANE */
    xfree(s->qkv); xfree(s->qkv_c); xfree(s->zbuf); xfree(s->avec); xfree(s->bvec);
    xfree(s->ssm_out); xfree(s->ssm_sig); xfree(s->qn); xfree(s->kn);
    xfree(s->kda_q); xfree(s->kda_k); xfree(s->kda_v);
    xfree(s->kda_qc); xfree(s->kda_kc); xfree(s->kda_vc);
    xfree(s->kda_gate_lat); xfree(s->kda_gate);
    xfree(s->moe_router_logits); xfree(s->moe_sel_idx); xfree(s->moe_sel_w);
    xfree(s->moe_lat_x); xfree(s->moe_lat_y);
    xfree(s->moe_h1); xfree(s->moe_h3); xfree(s->moe_h2); xfree(s->moe_shared_out);
    xfree(s->moe_router_bias);
    xfree(s->mtp_x); xfree(s->mtp_concat); xfree(s->mtp_emb_raw);
    xfree(s->mtp_chain); xfree(s->mtp_draft_logits); xfree(s->mtp_logits);
    xfree(s->mtp_verify_hidden);
    xfree(s->mtp_draft_q);
    xfree(s->mtp_target_p);
    xfree(s->ssm_state_q8); xfree(s->ssm_state_s);
#if LZ_GDN_STATE_2PLANE
    xfree(s->ssm_state_q8_lo);
#endif /* LZ_GDN_STATE_2PLANE */
    xfree(s->conv_state); xfree(s->conv_w32);
    xfree(s->csa2_pending_c); xfree(s->csa2_pending_z);
    xfree(s->csa2_pending_c_idx); xfree(s->csa2_pending_z_idx);
    xfree(s->csa2_comp_ring); xfree(s->csa2_idx_comp_ring);
    xfree(s->csa2_n_blocks);
    xfree(s->csa2_scores); xfree(s->csa2_topk_idx); xfree(s->csa2_topk_val);
    xfree(s->csa2_c_j); xfree(s->csa2_z_j);
    xfree(s->csa2_c_j_idx); xfree(s->csa2_z_j_idx);
    xfree(s->csa2_qidx); xfree(s->csa2_widx); xfree(s->csa2_cq_idx);
    xfree(s->csa2_attn_probs);
    xfree(s->conv_state_q); xfree(s->conv_mw);
    xfree(s->kda_q_i16); xfree(s->kda_k_i16); xfree(s->kda_v_i16);
    xfree(s->kda_qc_i16); xfree(s->kda_kc_i16); xfree(s->kda_vc_i16);
    xfree(s->bvec_i16); xfree(s->moe_logits_i16);
    xfree(s->kda_lat_i16); xfree(s->kda_lat_i32);
    xfree(s->kda_gate_i16); xfree(s->kda_dtb_i16); xfree(s->kda_decay);
    xfree(s->moe_lat_i16); xfree(s->moe_lat_q); xfree(s->moe_lat_qs);
    xfree(s->vtmp_i16);
    xfree(s->moe_h1_i16); xfree(s->moe_h3_i16);
    xfree(s->conv_sig_k1); xfree(s->conv_sig_oscale2); xfree(s->conv_sig_e);
    xfree(s->xq); xfree(s->xqs); xfree(s->wscr); xfree(s->fwht_scratch);
    xfree(s->subn_norm_int);
    xfree(s->engram_scratch);
    xfree(s->engram_conv_hist);
    xfree(s->gr_scratch);
    xfree(s->rope_cs);
    xfree(s->cache_idx);
    memset(s, 0, sizeof(*s));
}

void lz_state_reset(LZRunState *s, const LZModel *m) {
    const LZModelConfig *c = &m->config;
    /* +1 layer slot when an MTP head is bound - see lz_state_alloc's
       matching comment; must agree or this zeroes only part of the
       buffer it allocated. */
    int kv_layers = c->n_full_layers + (m->mtp ? 1 : 0);
    if (!s) return;
    /* Every one of these is NULL in some cache mode now (q4 and qjl do
       not allocate the Q8 planes; only the f32 arm allocates kf32). An
       unconditional memset here is a segfault that hides from --tokens,
       which never resets - it only shows in generation. Guard on the
       POINTER, not on the mode flags, so a future mode cannot forget to
       be added to a condition. */
    if (s->kq8) memset(s->kq8, 0, (size_t)kv_layers * s->kv_slots *
                       c->attn_kv_dim);
    if (s->vq8) memset(s->vq8, 0, (size_t)kv_layers * s->kv_slots *
                       c->attn_kv_dim);
    if (s->ksq) memset(s->ksq, 0, (size_t)kv_layers * s->kv_slots *
                       c->attn_kv_dim / 32 * sizeof(float));
    if (s->vsq) memset(s->vsq, 0, (size_t)kv_layers * s->kv_slots *
                       c->attn_kv_dim / 32 * sizeof(float));
#if LZ_KV_2PLANE
    if (s->kq8_lo) memset(s->kq8_lo, 0, (size_t)kv_layers * s->kv_slots *
                          c->attn_kv_dim);
    if (s->vq8_lo) memset(s->vq8_lo, 0, (size_t)kv_layers * s->kv_slots *
                          c->attn_kv_dim);
#endif /* LZ_KV_2PLANE */
    if (s->kf32) memset(s->kf32, 0, (size_t)kv_layers * s->kv_slots *
                        c->attn_kv_dim * sizeof(float));
    if (s->vf32) memset(s->vf32, 0, (size_t)kv_layers * s->kv_slots *
                        c->attn_kv_dim * sizeof(float));
    if (s->k4) memset(s->k4, 0, (size_t)kv_layers * s->kv_slots *
                      c->attn_kv_dim / 2);
    if (s->v4) memset(s->v4, 0, (size_t)kv_layers * s->kv_slots *
                      c->attn_kv_dim / 2);
    if (s->ks4) memset(s->ks4, 0, (size_t)kv_layers * s->kv_slots *
                       c->n_kv_heads * sizeof(float));
    if (s->vs4) memset(s->vs4, 0, (size_t)kv_layers * s->kv_slots *
                       c->n_kv_heads * sizeof(float));
    /* Zero the FULL ring (s->ssm_ring_depth slots, set once by
       lz_state_alloc and never recomputed here - see that field's own
       comment for why) or this zeroes only part (or past the end) of
       the buffer that was actually allocated. */
    memset(s->ssm_state_q8, 0, (size_t)s->ssm_ring_depth * c->n_linear_layers *
           c->lin_n_v_heads * c->lin_k_head_dim * c->lin_v_head_dim);
#if LZ_GDN_STATE_2PLANE
    memset(s->ssm_state_q8_lo, 0, (size_t)s->ssm_ring_depth * c->n_linear_layers *
           c->lin_n_v_heads * c->lin_k_head_dim * c->lin_v_head_dim);
#endif /* LZ_GDN_STATE_2PLANE */
    memset(s->ssm_state_s, 0, (size_t)s->ssm_ring_depth * c->n_linear_layers *
           c->lin_n_v_heads * c->lin_k_head_dim * c->lin_v_head_dim / 32 *
           sizeof(float));
#if LZ_CONV_FIXED
    if (s->conv_state_q)
        memset(s->conv_state_q, 0,
               (size_t)s->ssm_ring_depth * c->n_linear_layers *
               c->lin_conv_dim * (c->conv_kernel - 1) * sizeof(short));
#endif /* LZ_CONV_FIXED */
    memset(s->conv_state, 0, (size_t)s->ssm_ring_depth * c->n_linear_layers *
           c->lin_conv_dim * (c->conv_kernel - 1) * sizeof(float));
    /* S2 CSA2 - guarded on the pointer, not on any csa2_on flag, same
       reasoning kq8/vq8's own guard comment gives: allocation already
       decided whether this model has anything to zero. Finalized-block
       counts go back to 0 exactly like ssm_slot/mtp_pos below - a reset
       discards every compressed entry the same way it discards the KV
       cache and the recurrent state. */
    if (s->csa2_comp_ring) {
        int cm = c->csa2_compress_m > 0 ? c->csa2_compress_m : 1;
        int max_blocks = (s->seq_len + cm - 1) / cm;
        size_t nf = (size_t)c->n_full_layers;
        memset(s->csa2_pending_c, 0, nf * cm * c->head_dim * sizeof(float));
        memset(s->csa2_pending_z, 0, nf * cm * c->head_dim * sizeof(float));
        memset(s->csa2_pending_c_idx, 0,
               nf * cm * c->csa2_indexer_head_dim * sizeof(float));
        memset(s->csa2_pending_z_idx, 0,
               nf * cm * c->csa2_indexer_head_dim * sizeof(float));
        memset(s->csa2_comp_ring, 0,
               nf * (size_t)max_blocks * c->head_dim * sizeof(float));
        memset(s->csa2_idx_comp_ring, 0,
               nf * (size_t)max_blocks * c->csa2_indexer_head_dim * sizeof(float));
        memset(s->csa2_n_blocks, 0, nf * sizeof(int));
    }
    /* The ring's own "which slot is confirmed" index - see forward.h's
       s->ssm_slot comment. Every recurrent index resets to 0 here,
       same as mtp_pos right below. */
    s->ssm_slot = 0;
    /* The MTP block's own position counter - its KV cache slot (zeroed
       above) is indexed by this, not by the body's absolute position;
       see forward.h's s->mtp_pos comment. */
    s->mtp_pos = 0;
    /* engram's cross-chunk token history (forward.h's field comment):
       a reset means "nothing came before", so the leading-entry count
       goes back to 0, matching engram_hash's own padding rule for a
       fresh sequence. */
    s->engram_hist_n = 0;
    s->engram_conv_hist_n = 0;
    /* Every checkpoint taken before this point is now void: the KV cache
       it relies on (and deliberately does not copy) has just been
       zeroed. See LZStateCkpt in forward.h. */
    s->epoch++;
}

/* -------------------------------------------------- full_attention block */

/* Run nt tokens through this layer together. Batching reuses the
   WEIGHT LOADS: each output row's weights stream from memory once,
   serving nt tokens (arithmetic intensity 1 -> nt).

   Each token's own algorithm and ordering are untouched, so nt=1 is
   bit-identical to batch slice t - a hard gate, not a soft ask (see
   forward.h).

   `layer` is either a real index into m->layers[] or LZ_MTP_CACHE_LAYER
   (the MTP block, see above); everything else about this function is
   unchanged either way - the MTP block's q/k/v/o/q_norm/k_norm tensors
   are checked at load time (model.c's model_walk) to have exactly the
   body's attn_qgate_dim/attn_kv_dim/attn_q_dim/head_dim, so nothing here
   needs to special-case its shapes, only which cache slot it writes. */


/* Debug-only: force lz_mtp_draft_batch's per-slot attention through the
   ordinary forward_attn (full Q/K/RoPE/score/softmax) instead of
   mtp_attn_t1's shortcut below, for a same-binary, same-weights A/B
   comparison (.prof/mtp_kdraft_check.c) - the two must always agree bit
   for bit; this control exists to keep proving that continuously, not
   to offer a slower alternative. Same shape as lz_debug_mtp_attn_scale
   above. NEVER set outside such a test. */
int lz_debug_mtp_attn_t1_slow = 0;

/* ---------------------------------------------------------------- main body */

/* Run one batch (nt <= LZ_BATCH_MAX) through all layers; returns the
   last token's logits. nt=1 is exactly the original lz_forward, no more
   and no less.

   Two additive parameters, both NULL/0 from the ordinary callers
   (lz_forward, lz_forward_batch) - behavior is byte-identical when
   both are absent:

     capture_hidden   if non-NULL, receives EVERY token's raw
                       (pre-final-norm) residual-stream hidden state,
                       nt*hidden_size floats, token-major (every
                       position, so lz_forward_batch_capture can hand
                       lz_mtp_prefill every prefilled position's hidden,
                       not just the batch's last) - the MTP draft head's
                       "h_body"
                       input (lz_forward_capture, forward.h). nt=1
                       callers (lz_forward_capture itself) see no change:
                       "every token" and "the last token" are the same
                       one row.
     all_logits/       if all_logits is set, compute EVERY position's
     logits_out        logits into logits_out (nt*vocab floats) instead
                       of the usual last-token-only path into s->logits -
                       needed by lz_forward_verify's speculative-decode
                       check. Mutually exclusive with the ordinary path
                       (see below): when set, s->logits is NOT written,
                       avoiding normalizing the last token's hidden state
                       twice.

                       SECOND ROLE: also the signal forward_ssm/
                       forward_kda use to advance the SSM/
                       conv rollback ring one slot per token instead of
                       updating s->ssm_slot's fixed slot in place - see
                       forward.h's s->ssm_slot comment. Reused rather
                       than a dedicated parameter because it is ALREADY
                       exactly and only true for lz_forward_verify's own
                       call (the sole all_logits=1 caller today) - the
                       same distinction this parameter already draws
                       ("every position's logits, needed to verify each
                       drafted token") is also exactly "this position's
                       forward is provisional until accept/reject
                       decides it, needed by the ring". If a future
                       caller ever wants all_logits without wanting ring
                       advancement (or vice versa), this coupling must
                       be split into two real parameters - flagged here
                       so that split is not a surprise. */
/* POST-final-norm hidden of EVERY token in this chunk - the MTP
   draft head's "hidden" input, for its first step
   (lz_forward_capture), a prompt prefill (lz_forward_batch_capture
   / lz_mtp_prefill), or a verify batch's per-position seed
   (lz_forward_verify -> s->mtp_verify_hidden). All three MTP hidden
   sources reach the block through this one parameter, so this is
   the single place the stage is decided.

   The input is POST-FINAL-NORM (an RMSNorm of s->x), not a raw
   memcpy. Feeding the draft head an already-normed vector is not
   "double-norming": an RMSNorm applied to an already-normed vector
   is a learned rescale with its own weight, and that is what the
   architecture specifies:

     vLLM (training-side reference): Qwen3NextModel.forward applies
       self.norm and RETURNS the normed value
       (refsrc/vllm/vllm/model_executor/models/qwen3_next.py:709-712);
       Qwen3NextForCausalLM.forward passes it through (:824-836);
       the runner hands exactly that to the drafter
       (v1/worker/gpu_model_runner.py:5199). The per-model override
       hook that could change this - get_mtp_target_hidden_states -
       is implemented ONLY by deepseek_v4, not by this family, so
       Qwen3-Next takes the plain post-norm path (:5182).
     llama.cpp: h_nextn is captured AFTER model.output_norm and is
       the same tensor that feeds the real logits
       (refsrc/llama.cpp/src/models/qwen35.cpp:209-213).

   llama.cpp's own generic field comment says "hidden state before
   final output norm" (src/llama-graph.h:903) and CONTRADICTS its
   own Qwen3.5 graph. It is not wrong, it is about a different
   architecture - DeepSeek V4's MTP genuinely wants the pre-hc_head
   residual (gpu_model_runner.py:5176-5178). Which stage feeds the
   draft head is a PER-ARCHITECTURE choice; this family takes the
   post-norm path.

   Normalizes into capture_hidden rather than in place, because
   s->x still has to reach the two output paths below unnormalized:
   the all_logits path norms every position itself and the ordinary
   path norms only the last, and either would double-apply if this
   had already touched s->x. TrunkCapture must
   stay on the same stage - a train/serve split here is invisible to
   every bit-identity gate (MTP only affects DRAFT quality; the
   target model decides the output either way) and shows up only as
   a depressed acceptance rate. */
static void chunk_capture_hidden(const LZModel *m, LZRunState *s,
                                 const LZModelConfig *c, float *capture_hidden,
                                 int nt, int dim) {
    int tk;
    for (tk = 0; tk < nt; tk++)
        lz_rmsnorm(capture_hidden + (size_t)tk * dim,
                   s->x + (size_t)tk * dim,
                   lz_t_f32(&m->final_norm, s->wscr), dim,
                   c->rms_norm_eps);
}

/* EVERY position's logits (the speculative-verify shape), into the
   caller's buffer.

   SAME TIER AS THE SINGLE-POSITION HEAD (chunk_logits_last), and that is a
   correctness requirement, not symmetry for its own sake.
   lz_rmsnorm_int is a TIER: its own header says it is not
   bit-identical to the lz_rmsnorm + lz_quantize_q8 pair. A verify
   row that took the float pair while ordinary decode took the
   int one produced DIFFERENT logits for the same position, so
   --spec K stopped matching --spec 0 as soon as one of those
   differences crossed an argmax boundary - with penalties at
   their identity values, on a bare temperature-0 greedy run.
   The two guards are chunk_logits_last's, for its reasons: gse > 0
   because the matmul reads xq/xqs only when the weight is
   quantized (at gse == 0 it reads s->x, which the int path never
   normalizes), and LZ_TAP_OFF because a tap build must keep the
   float path. */
static void chunk_logits_all(const LZModel *m, LZRunState *s,
                             const LZModelConfig *c, float *logits_out,
                             int nt, int dim) {
    int gse = lz_act_gs(&m->embed_tokens, dim);
    int nse = scale_groups(dim, gse);
    int use_int = 0;
    int tk;
#if defined(LZ_TAP_OFF)
    use_int = lz_norm_int() && lz_norm_can_fixed(dim) &&
              gse > 0 && (dim % gse) == 0;
#endif /* LZ_TAP_OFF */
    for (tk = 0; tk < nt; tk++) {
        float *xt = s->x + (size_t)tk * dim;
        if (use_int) {
            float deq;
            lz_rmsnorm_int(s->subn_norm_int, xt,
                           lz_t_f32(&m->final_norm, s->wscr), dim,
                           c->rms_norm_eps, &deq);
            lz_quantize_q8_int(s->subn_norm_int, dim, gse, deq,
                               s->xq + (size_t)tk * dim,
                               s->xqs + (size_t)tk * nse);
        } else {
            lz_rmsnorm(xt, xt, lz_t_f32(&m->final_norm, s->wscr), dim,
                      c->rms_norm_eps);
            lz_quantize_q8(xt, dim, gse, s->xq + (size_t)tk * dim,
                          s->xqs + (size_t)tk * nse);
        }
    }
    lz_matmul_xq_nt(logits_out, s->x, s->xq, s->xqs, &m->embed_tokens,
                   dim, c->vocab_size, nt);
}

/* lm_head only computes the LAST token. It is the single largest
   matmul (32768 vocab rows), and prefill's intermediate logits are
   wanted by nobody - computing the whole batch would pay the
   forward's most expensive step nt-1 times for nothing. At nt=1,
   nt-1 == 0, identical to the original path. */
static void chunk_logits_last(const LZModel *m, LZRunState *s,
                              const LZModelConfig *c, int nt, int dim) {
    LZ_PROF_DECL(_th);
    float *xl = s->x + (size_t)(nt - 1) * dim;
    int gse = lz_act_gs(&m->embed_tokens, dim);
    /* The norms census's smallest consumer, and the shape the other
       four share: a float vector built by the norm and taken apart
       again by the very next statement. The int tier writes the Q15
       product into s->subn_norm_int with its scale out of band and
       lz_quantize_q8_int folds that into the group scale, so xl is
       never materialized.
       This is a TIER, not a refactor - lz_rmsnorm_int's own header
       says it is not bit-identical to the float pair, the plain
       norm having no Q15-chain-then-dequant sibling to match. It
       therefore rides --fixed like every other fixed tier.

       Two guards beyond the tier, both of them things that read a
       buffer this path stops writing:
         gse > 0    the weight is quantized, so lz_matmul_xq below
                    takes xq/xqs. At gse == 0 the weight is F32 and
                    the matmul reads xl itself, which would still
                    hold the UN-normalized residual.
         LZ_TAP_OFF nobody is tapping "fnorm". A tap build keeps the
                    float path rather than reporting the residual as
                    if it were the norm. */
    int use_int = 0;
#if defined(LZ_TAP_OFF)
    use_int = lz_norm_int() && lz_norm_can_fixed(dim) &&
              gse > 0 && (dim % gse) == 0;
#endif /* LZ_TAP_OFF */
    /* tie_word_embeddings: output projection reuses the embedding
       directly; there is no standalone lm_head */
    if (use_int) {
        float deq;
        lz_rmsnorm_int(s->subn_norm_int, xl,
                       lz_t_f32(&m->final_norm, s->wscr), dim,
                       c->rms_norm_eps, &deq);
        lz_quantize_q8_int(s->subn_norm_int, dim, gse, deq,
                           s->xq, s->xqs);
    } else {
        lz_rmsnorm(xl, xl, lz_t_f32(&m->final_norm, s->wscr), dim,
                   c->rms_norm_eps);
        LZ_TAP("fnorm", -1, xl, dim);
        lz_quantize_q8(xl, dim, gse, s->xq, s->xqs);
    }
    LZ_PROF_BEG(_th);
    lz_matmul_xq(s->logits, xl, s->xq, s->xqs, &m->embed_tokens, dim,
                 c->vocab_size);
    LZ_PROF_END(_th, LZ_PROF_HEAD);
}

/* ------------------------------------------------------- gated residual */

/* GR's branches ride inside s->x (forward.h's own comment on that buffer).
   Branch-MAJOR: within one token's slice, branch i occupies
   [i*dim, (i+1)*dim). That is the layout the training side has on both
   sides of its own `flatten`: `expand`+`flatten` on the way in and
   `unflatten(-1, (nr, D))` on the way out produce the same order, so this
   is not a choice this port made.

   Training-side reference for all four functions below:
   engine/kunmoe_modeling.py's GatedResidual4 and GrDecoderLayer.forward. */

/* gr_first: the embedding, dim wide, becomes nr IDENTICAL branches, each
   token's stream moving from [tk*dim, tk*dim+dim) to
   [tk*nr*dim, (tk+1)*nr*dim).

   In place, and token by token DESCENDING. A token's destination region is
   nr times its source region and starts at or after it, so an ascending
   pass would overwrite tokens 1..nr-1's input before expanding them. EVERY
   branch is copied, branch 0 included: only token 0's source is already at
   its destination, and assuming otherwise left the other tokens' branch 0
   holding the previous token's embedding - caught by this engine's own
   gr_gate.sh `batch` arm, which exists for exactly this shape. */
static void gr_expand(LZRunState *s, int nt, int nr, int dim) {
    int tk, b;
    for (tk = nt - 1; tk >= 0; tk--) {
        const float *src = s->x + (size_t)tk * dim;
        float *dst = s->x + (size_t)tk * nr * dim;
        /* memmove, not memcpy: token 0's destination IS its source, and
           memcpy's contract excludes overlap even when the two are equal. */
        for (b = nr - 1; b >= 0; b--)
            memmove(dst + (size_t)b * dim, src, (size_t)dim * sizeof(float));
    }
}

/* Sum nr rows of `dim` floats in place, pairwise: leaves combine two at a
   time and the survivors halve each pass.

   THE ORDER IS NOT A PREFERENCE. The training side's own neutralised arm
   (`g` and `s` forced to 1, read normalisation removed) reproduces its
   GR-off arm BIT FOR BIT - measured there, 0 of 32768 logits bytes differ.
   That can only hold if summing the nr IDENTICAL branches gr_first
   produced is exact, and a running total is not: with v the shared value,
   `2v + v` rounds generically, so three of the four branches already miss.
   Pairwise gives `(v+v) + (v+v)` = `4v`, which is exact.

   Measured here the same way: with a running total the same arm differed
   from GR-off by 1-cos 1.2e-7, i.e. above the exact-reproduction the
   training side achieves. */
static void gr_tree_sum(float *a, int nr, int dim) {
    int step, b, i;
    for (step = 1; step < nr; step *= 2)
        for (b = 0; b + step < nr; b += 2 * step)
            for (i = 0; i < dim; i++)
                a[(size_t)b * dim + i] += a[(size_t)(b + step) * dim + i];
}

/* gr_last: the branch mean, back into the FIRST dim elements of the same
   token's slice - the [B,S,D] the rest of the stack expects (the next
   layer, CED's capture hook, the final norm, lm_head). MEAN, not sum:
   GatedResidual4.collapse is `streams.mean(dim=-2)`.

   Leaves are combined with the same pairwise tree as the read sum, and for
   the same measured reason (gr_tree_sum's comment). It runs in place on
   s->x: only rows at or after the destination are touched, and every row
   it reads has already been read by the pass that wrote it. */
static void gr_collapse(LZRunState *s, int nt, int nr, int dim) {
    int tk, i;
    for (tk = 0; tk < nt; tk++) {
        float *dst = s->x + (size_t)tk * dim;
        float *base = s->x + (size_t)tk * nr * dim;
        gr_tree_sum(base, nr, dim);
        for (i = 0; i < dim; i++) dst[i] = base[i] / (float)nr;
    }
}

/* The read gate: fills s->xb (the sublayer's input) and stashes the two
   things the write gate needs - Rhat in s->gr_rhat and the write gate
   s in s->gr_s. Reused for the attention sublayer and then for the MLP
   one, which is why the stash lives in the run state rather than on this
   stack frame.

   THIS NORM RUNS UNCONDITIONALLY, INCLUDING ON A SubLN BODY, and that is
   the one thing about this port that is easy to get backwards. The
   training side's `_NoPreLN` forward is the IDENTITY - under SubLN the
   block has no pre-normalisation at all, the normalisation moved into
   each projection - so `use_subn` is not a reason to skip this: it is the
   reason there would otherwise be NO normalisation here at all. SubLN
   normalises what each projection READS; GR's read normalises the
   residual BEFORE the gate, and forward_attn/forward_kda keep applying
   their own per-projection norms underneath. Both, not one or the other.

   An all-zero `norms.<b>.weight` is unit gain, not a broken tensor:
   lz_rmsnorm's parameterisation is (1 + w). */
static void gr_read(const LZModel *m, LZRunState *s, const LZTensor *norms,
                    const LZTensor *wd, const LZTensor *wu,
                    const LZTensor *ww, int nt, int nr, int dim) {
    const LZModelConfig *c = &m->config;
    int flat = nr * dim;
    int r = dim / 8 > 0 ? dim / 8 : 1;   /* max(1, d/8): GatedResidual4.__init__ */
    float nrf = (float)nr;
    int tk, b, i;

    for (tk = 0; tk < nt; tk++) {
        const float *R  = s->x + (size_t)tk * flat;
        float *RH = s->gr_rhat + (size_t)tk * flat;
        float *G  = s->gr_g + (size_t)tk * flat;
        float *h  = s->gr_h + (size_t)tk * r;
        float *sg = s->gr_s + (size_t)tk * nr;
        float *xb = s->xb + (size_t)tk * dim;

        for (b = 0; b < nr; b++)
            lz_rmsnorm(RH + (size_t)b * dim, R + (size_t)b * dim,
                       lz_t_f32(&norms[b], s->wscr), dim, c->rms_norm_eps);

        /* G = unvec(sigmoid(W_u SiLU((1/n_r) W_d vec(Rhat)))) - Eq. 31/32. The
           divisor is a runtime float (a divisor that is not a compile-time
           constant is the same instruction on both compilers), which is
           also what the training side's `/ self.nr` rounds like. */
        lz_matmul_w(h, RH, wd, flat, r, s->xq, s->xqs);
        for (i = 0; i < r; i++) {
            float t = h[i] / nrf;
            h[i] = lz_silu(t);
        }
        lz_matmul_w(G, h, wu, r, flat, s->xq, s->xqs);
        for (i = 0; i < flat; i++) G[i] = lz_sigmoid(G[i]);

        /* s = 2*sigmoid((1/n_r) W_w vec(Rhat)) - Eq. 33. Computed HERE, from
           the Rhat this call just produced, and read back by gr_write: the
           training side's write reads the Rhat its read stored
           (self._rhat), it does not re-derive one from the already-updated
           streams. */
        lz_matmul_w(sg, RH, ww, flat, nr, s->xq, s->xqs);
        for (i = 0; i < nr; i++) sg[i] = 2.0f * lz_sigmoid(sg[i] / nrf);

        /* x = (1/n_r) * sum_i (G_i * Rhat_i). The branch sum is a pairwise
           TREE, not a running total - see gr_tree_sum's own comment for why
           the order is load-bearing rather than a preference. The product
           goes back into G, which nothing reads afterwards. */
        for (i = 0; i < flat; i++) G[i] *= RH[i];
        gr_tree_sum(G, nr, dim);
        for (i = 0; i < dim; i++) xb[i] = G[i] / nrf;
    }
}

/* The write gate: R'_i = R_i + s_i*y, one scalar per branch. s->xb2 is
   dim wide and s->x is nr*dim wide, so this is not an elementwise add -
   each branch gets the SAME sublayer output scaled by its own gate. */
static void gr_write(LZRunState *s, int nt, int nr, int dim) {
    int tk, b, i;
    for (tk = 0; tk < nt; tk++) {
        float *R = s->x + (size_t)tk * nr * dim;
        const float *y = s->xb2 + (size_t)tk * dim;
        const float *sg = s->gr_s + (size_t)tk * nr;
        for (b = 0; b < nr; b++) {
            float *Rb = R + (size_t)b * dim;
            float sb = sg[b];
            for (i = 0; i < dim; i++) Rb[i] += sb * y[i];
        }
    }
}

static float *forward_chunk(const LZModel *m, LZRunState *s,
                            const int *tokens, int nt, int pos0,
                            float *capture_hidden,
                            int all_logits, float *logits_out,
                            int skip_logits, int ced_skip_last) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size;
    int l, i, tk;
    /* GR's branch count, 0 when the switch is off. Read once here; every
       per-layer decision below goes through L->gr_on/gr_first/gr_last,
       which setup_layers derived from this field and ced_split. */
    int nr = c->gated_residual;
    /* S1 attn_kv_share: the layer whose attention input every target
       layer's K/V projections read, or -1 (model.h's own comment on the
       rule; lz_kv_share_target is the other half). ksrc is resolved once
       per chunk - the config is fixed - and kv_in is the per-layer
       pointer the layer loop hands to forward_attn. */
    int ksrc = lz_kv_share_src(c);
    const float *kv_in;
    int ring_base;               /* see its own comment below, before the layer loop */
    unsigned fpu;

    lz_debug_n_chunks++;         /* positive control for --batch, see its definition */

    /* Start this chunk's expert union from empty. Cleared HERE rather
       than by the caller so the mask can never carry a previous token's
       experts into this one - the panel would show the union of two
       tokens and there would be nothing on screen to say so.
       With nt > 1 (prefill) the union spans the chunk, which nobody
       displays: the panel only ever draws snapshots from generated
       tokens, and those come through nt == 1. */
    if (s->ins) {
        memset(s->ins->expert_hits, 0, sizeof s->ins->expert_hits);
        s->ins->n_experts = c->num_experts;
        s->ins->experts_truncated = c->num_experts > LZ_INSPECT_EXPERT_MAX;
    }

    /* Defense: an out-of-range token reads past embed_tokens. The
       encode/sample paths always produce valid ids, but direct token
       calls (e.g. --tokens debugging) must be caught here. */
    for (tk = 0; tk < nt; tk++) {
        if (tokens[tk] < 0 || tokens[tk] >= c->vocab_size) return NULL;
        if (pos0 + tk < 0 || pos0 + tk >= s->seq_len) return NULL;
    }
    /* The whole forward runs under float precision; restore on exit -
       see lz_fpu_float_begin */
    fpu = lz_fpu_float_begin();

    /* embedding dequantization goes through the unified entry - layout
       knowledge lives only in ops.c */
    for (tk = 0; tk < nt; tk++)
        lz_t_row_f32(&m->embed_tokens, tokens[tk], dim,
                     s->x + (size_t)tk * dim);
    LZ_TAP("emb", -1, s->x, dim);

    /* Engram: conditional n-gram memory, injected after embed_tokens
       (the hook in kunmoe_modeling.py registers on embed_tokens).
       Modifies s->x in-place. No-op when config.engram is 0. */
    if (m->engram) {
        forward_engram(m, s, tokens, nt, pos0, s->engram_hist,
                      s->engram_hist_n);
        LZ_TAP("eng", -1, s->x, dim);
        /* Roll this chunk's own tail into the cross-chunk history for
           the NEXT chunk's hash/mask windows (forward.h's field
           comment - both read up to engram_ngram-1 tokens before
           their chunk's first token). Only the trailing min(nt, 31)
           tokens matter; a chunk at least that wide overwrites the
           whole history and needs no merge with what came before. */
        {
            int keep = (int)(sizeof(s->engram_hist) /
                             sizeof(s->engram_hist[0]));
            if (nt >= keep) {
                memcpy(s->engram_hist, tokens + (nt - keep),
                      (size_t)keep * sizeof(int));
                s->engram_hist_n = keep;
            } else {
                int old_keep = keep - nt;
                if (old_keep > s->engram_hist_n) old_keep = s->engram_hist_n;
                if (old_keep > 0)
                    memmove(s->engram_hist, s->engram_hist + (s->engram_hist_n - old_keep),
                           (size_t)old_keep * sizeof(int));
                memcpy(s->engram_hist + old_keep, tokens,
                      (size_t)nt * sizeof(int));
                s->engram_hist_n = old_keep + nt;
            }
        }
    }

    /* Ring base for this CHUNK (the SSM/conv rollback ring - forward.h's
       s->ssm_slot). Read ONCE here, before the layer loop,
       and passed down explicitly rather than having forward_ssm/
       forward_kda read s->ssm_slot themselves: those two are called
       once PER LINEAR/KDA LAYER (as many as n_linear_layers times per
       chunk), and s->ssm_slot must only advance ONCE per chunk's worth
       of nt tokens, not once per layer - so the read-once-here /
       advance-once-after-the-loop split is load-bearing, not tidiness.
       It also makes multi-chunk verify batches (lz_forward_verify's own
       chunking loop - LZ_BATCH_MAX=4 < LZ_SPEC_K_MAX+1=7, so a wide
       verify round genuinely needs more than one forward_chunk call)
       chain correctly for free: chunk 2 simply reads whatever chunk 1
       left in s->ssm_slot. */
    ring_base = s->ssm_slot;

    for (l = 0; l < c->n_layers; l++) {
        const LZLayer *L = &m->layers[l];
        LZ_PROF_DECL(_tp);
        LZ_PROF_DECL(_tf);

        /* docs/x87-gcc-reg-stack-leak.md: this is the checkpoint that
           first showed the x87 stack going dirty and staying dirty
           across layer boundaries. No-op unless LZ_X87_STACK_GATE. */
        LZ_X87_STACK_CHECK("layer-entry");

        /* gr_first: the embedding (dim wide) becomes nr identical
           branches, in place - GrDecoderLayer.forward's own
           `gr_attn.expand(hidden_states)`. Done at the TOP of the layer so
           the read gate below, the SubLN copy and every other reader of
           s->x in this iteration already see the widened stream. */
        if (L->gr_first) gr_expand(s, nt, nr, dim);

        /* SubLN (config.use_subn): the two pre-layer RMSNorms are
           deleted and each ternary projection normalizes its own input
           instead (forward_kda / dense_ffn_step). Ordinary models keep
           the original norms, bit for bit. The SubLN copy is elementwise
           so downstream readers of s->xb keep working unchanged.

           GR's read REPLACES this step rather than stacking on it (the
           training side's GrDecoderLayer does not call
           input_layernorm either), and it does its own normalisation
           whatever use_subn says - gr_read's own comment has that
           argument. Note it is added to the per-projection norms inside
           forward_attn/forward_kda, not instead of them. */
        if (L->gr_on) {
            gr_read(m, s, L->gr_attn_norms, &L->gr_attn_wd, &L->gr_attn_wu,
                    &L->gr_attn_ww, nt, nr, dim);
        } else if (c->use_subn) {
            for (i = 0; i < nt * dim; i++) s->xb[i] = s->x[i];
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb + (size_t)tk * dim, s->x + (size_t)tk * dim,
                           lz_t_f32(&L->input_layernorm, s->wscr), dim,
                           c->rms_norm_eps);
        }
        LZ_TAP("an", l, s->xb, dim);
        /* S1 attn_kv_share. Captured HERE, at the layer's attention
           entry: this is exactly the tensor the training side's capture
           hook records off the source attention module, so the shared
           k_proj/v_proj reproduce the source's own K/V rather than a
           re-projection of the target's input (model.h's attn_kv_share
           comment). Done for the source whatever its attention kind is -
           a csa2_on source layer keeps no per-position K/V of its own,
           which is the case this exists for. One copy per chunk, and
           only on a model that has a source at all. */
        kv_in = NULL;
        if (L->type == LZ_LT_FULL) {
            if (l == ksrc) {
                memcpy(s->kvshare_x, s->xb,
                       (size_t)nt * dim * sizeof(float));
            } else if (lz_kv_share_target(c, l)) {
                kv_in = s->kvshare_x;
            }
        }
        LZ_PROF_BEG(_tp);
        if (L->type == LZ_LT_FULL) {
            /* S2 CSA2 takes over this layer's ENTIRE attention forward -
               see model.h's LZModelConfig.csa2_on[] comment. Checked
               first and unconditionally: --ced's kv_only optimization
               below is forward_attn_kv_only's own KV-cache-shaped
               trick, and CSA2 has no such cache (compressed blocks, not
               per-position K/V) - composing the two is out of scope for
               this pass (forward_csa2.c's own header comment), so a
               csa2_on layer never takes the ced_skip_last branch even
               when it happens to be the model's last layer. */
            if (L->csa2_on) {
                forward_csa2(m, s, L, l, pos0, nt);
            } else
            /* --ced, on a chunk whose outputs nobody reads (see
               ced_skip_last). The last layer's K/V over these positions
               is the one thing the run keeps; everything else it would
               have computed here is not observed by any caller, because
               lz_forward_batch returns only the final chunk's logits.

               The `continue` is the whole point and is not tidiness: it
               skips the residual add AND the FFN as well as the
               attention, so this layer contributes nothing to s->x at
               these positions - which is correct, since s->x for them is
               not read again. s->xb2 is left holding whatever the
               previous layer put there and no reader follows. */
            if (ced_skip_last && l == c->n_layers - 1) {
                lz_debug_ced_skipped += nt;
                forward_attn_kv_only(m, s, L, l, pos0, nt, kv_in);
                continue;
            } else {
                forward_attn(m, s, L, l, pos0, nt, kv_in);
            }
        }
        else if (L->type == LZ_LT_KDA)  forward_kda(m, s, L, l, nt, all_logits, ring_base);
        else                             forward_ssm(m, s, L, l, nt, all_logits, ring_base);
        LZ_PROF_END(_tp, L->type == LZ_LT_FULL ? LZ_PROF_ATTN : LZ_PROF_LIN);
        LZ_TAP("blk", l, s->xb2, dim);
        if (L->gr_on) {
            gr_write(s, nt, nr, dim);
        } else {
            for (i = 0; i < nt * dim; i++) s->x[i] += s->xb2[i];
        }
        LZ_TAP("res", l, s->x, L->gr_on ? nr * dim : dim);

        if (L->gr_on) {
            gr_read(m, s, L->gr_mlp_norms, &L->gr_mlp_wd, &L->gr_mlp_wu,
                    &L->gr_mlp_ww, nt, nr, dim);
        } else if (c->use_subn) {
            for (i = 0; i < nt * dim; i++) s->xb[i] = s->x[i];
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb + (size_t)tk * dim, s->x + (size_t)tk * dim,
                           lz_t_f32(&L->post_attention_layernorm, s->wscr), dim,
                           c->rms_norm_eps);
        }
        LZ_TAP("pn", l, s->xb, dim);
        LZ_PROF_BEG(_tf);
        if (L->ffn_moe) {
            forward_moe(m, s, L, l, nt);
        } else {
            dense_ffn_step(m, s, L, l, nt, c->intermediate_size);
        }
        LZ_PROF_END(_tf, LZ_PROF_FFN);
        LZ_TAP("ffn", l, s->xb2, dim);
        if (L->gr_on) {
            gr_write(s, nt, nr, dim);
        } else {
            for (i = 0; i < nt * dim; i++) s->x[i] += s->xb2[i];
        }
        LZ_TAP("out", l, s->x, L->gr_on ? nr * dim : dim);

        /* gr_last: collapse back to [B,S,D] BEFORE the next iteration
           touches s->x. The layer this fires on is the CED boundary on a
           split stack (lz_gr_span), so both the next layer's attention
           and CED's own capture hook - which reads a plain [B,S,D] off
           this layer - see the shape they expect. Note it fires at the
           END of this layer and not at the top of the next: the boundary
           layer's OWN sublayers run on the widened stream, which is why
           this cannot be a `gr_first`-style step in the loop head. */
        if (L->gr_last) gr_collapse(s, nt, nr, dim);
    }
    /* Advance the ring ONCE for this whole chunk, after every layer has
       used ring_base - ordinary decode/prefill (all_logits==0) never
       touches s->ssm_slot at all, so those paths stay bit-identical to
       a single-slot engine. See ring_base's own comment above for why
       this lives here and not inside forward_ssm/forward_kda. */
    if (all_logits)
        s->ssm_slot = (ring_base + nt) % s->ssm_ring_depth;

    if (capture_hidden) {
        chunk_capture_hidden(m, s, c, capture_hidden, nt, dim);
    }

    if (all_logits) {
        /* Speculative-decode verify path (lz_forward_verify): EVERY
           position's logits, not just the last - see this function's
           own comment above. Mutually exclusive with the block below:
           normalizing position nt-1 here AND there would run rmsnorm on
           an already-normalized vector a second time. */
        chunk_logits_all(m, s, c, logits_out, nt, dim);
        lz_fpu_float_end(fpu);
        return logits_out;
    }

    /* skip_logits extends the chunk-internal "prefill's intermediate
       logits are wanted by nobody" reasoning (below) ACROSS chunks:
       when set, even the last token's lm_head is skipped, since only
       the final chunk's logits are ever read. Measured on the 0.8B:
       lm_head is 27.4 ms/token of a 99 ms/token prefill at width 1 and
       3.60 of 52 at width 8 - i.e. the largest single line item either
       way, and all but the last chunk's is discarded.

       Safe by construction, not by convention: the only callers that
       set it (lz_forward_batch / _capture) overwrite `lg` on every
       iteration, so an intermediate chunk's logit CONTENTS are
       unobservable in any case. The returned pointer is unchanged -
       still s->logits, still non-NULL - because callers use the return
       purely as a success flag. A future caller that wants to read
       logits from a non-final chunk must pass skip_logits=0; there is
       no way for it to notice otherwise. */
    if (skip_logits) {
        lz_fpu_float_end(fpu);
        return s->logits;
    }
    lz_debug_n_lmhead++;         /* positive control, see its definition */

    chunk_logits_last(m, s, c, nt, dim);
    lz_fpu_float_end(fpu);
    return s->logits;
}

float *lz_forward(const LZModel *m, LZRunState *s, int token, int pos) {
    return forward_chunk(m, s, &token, 1, pos, NULL, 0, NULL, 0, 0);
}

float *lz_forward_capture(const LZModel *m, LZRunState *s, int token, int pos) {
    /* s->mtp_chain is unallocated (NULL) when m->mtp is NULL - guard
       rather than let forward_chunk memcpy into it, matching this
       function's own contract ("a no-op ... when m->mtp is NULL"). */
    float t_us0 = lz_time_ms() * LZ_US_SCALE;  /* debug probe, see part 4 */
    float *r = forward_chunk(m, s, &token, 1, pos,
                             m->mtp ? s->mtp_chain : NULL, 0, NULL, 0, 0);
    lz_debug_us_capture += (lz_i64)(lz_time_ms() * LZ_US_SCALE - t_us0);
    lz_debug_n_capture++;
    return r;
}

/* CED bounded replay's reset half (model.h's ced_split/ced_replay_w
   comment) - zeroes the KDA recurrent/conv state for every linear-
   attention layer at or above ced_split, at ring slot s->ssm_slot, so
   the chunk about to run sees a state indistinguishable from a fresh
   sequence start there. Matches the training-side simulation
   (engine/kunmoe_modeling.py's _kda_packing_hook) inserting a packing
   boundary at the same absolute offset: position ids keep counting up,
   only the recurrence's own carried state is discarded.

   Zeroes the raw ring-buffer bytes every KDA precision tier (float,
   fixed-int16, SubLN, 2-plane) reads its incoming state from, rather
   than threading a reset branch through forward_kda's per-tier
   dispatch - the same q8_slot_stride/conv_slot_stride/li_off_* address
   arithmetic forward_kda's own header comment derives, kept in sync by
   being the same formulas, not re-derived independently. */
static void lz_ced_replay_reset(const LZModel *m, LZRunState *s) {
    const LZModelConfig *c = &m->config;
    int l;
    int slot = s->ssm_slot;
    int nv = c->lin_n_v_heads, kd = c->lin_k_head_dim, vd = c->lin_v_head_dim;
    int hist = c->conv_kernel - 1;
    size_t q8_slot_stride = (size_t)c->n_linear_layers * nv * kd * vd;
    size_t s_slot_stride  = q8_slot_stride / 32;
    size_t conv_slot_stride = (size_t)c->n_linear_layers *
                              c->lin_conv_dim * hist;

    for (l = 0; l < c->n_layers; l++) {
        int li;
        size_t li_off_q8, li_off_s, li_off_conv;
        if (c->layer_types[l] != LZ_LT_KDA || l < c->ced_split) continue;
        lz_debug_ced_replay_layers++;
        li = s->cache_idx[l];
        li_off_q8 = (size_t)li * nv * kd * vd;
        li_off_s = li_off_q8 / 32;
        li_off_conv = (size_t)li * c->lin_conv_dim * hist;
        memset(s->ssm_state_q8 + (size_t)slot * q8_slot_stride + li_off_q8,
               0, (size_t)nv * kd * vd);
#if LZ_GDN_STATE_2PLANE
        memset(s->ssm_state_q8_lo + (size_t)slot * q8_slot_stride + li_off_q8,
               0, (size_t)nv * kd * vd);
#endif
        memset(s->ssm_state_s + (size_t)slot * s_slot_stride + li_off_s,
               0, (size_t)nv * kd * vd / 32 * sizeof(float));
#if LZ_CONV_FIXED
        if (s->conv_state_q)
            memset(s->conv_state_q + (size_t)slot * conv_slot_stride + li_off_conv,
                   0, (size_t)c->lin_conv_dim * hist * sizeof(short));
#endif
        memset(s->conv_state + (size_t)slot * conv_slot_stride + li_off_conv,
               0, (size_t)c->lin_conv_dim * hist * sizeof(float));
    }
}

float *lz_forward_batch(const LZModel *m, LZRunState *s,
                        const int *tokens, int n, int pos0) {
    const LZModelConfig *c = &m->config;
    float *lg = NULL;
    int cap, done = 0;
    /* Reset boundary, an offset within THIS call's own n tokens (not an
       absolute/cumulative position) - matching what the training-side
       probe actually measured (.prof/ced_replay_w_probe.py): one
       forward call over a T-token sequence, boundary at T-W within it.
       This is exactly the prompt-prefill case (pos0==0, n==the whole
       prompt). A pos0>0 call (new turns appended after cached context)
       has no training-side analog to validate against - there is no
       incremental-cache concept in the packing-boundary simulation at
       all - so it gets the same call-relative rule for lack of a
       measured alternative, not because it has been shown correct.
       -1 means off (window disabled or window >= this call's own
       length, matching the training hook's own T>W guard exactly - see
       ced_replay_w_probe.py's W=N control). */
    int replay_boundary = (c->ced_replay_w > 0 && n > c->ced_replay_w) ?
                          n - c->ced_replay_w : -1;

    if (!tokens || n < 1) return NULL;
    cap = s->nt_cap;
    if (cap < 1) cap = 1;
    if (cap > LZ_BATCH_MAX) cap = LZ_BATCH_MAX;
    while (done < n) {
        int k = n - done;
        if (k > cap) k = cap;
        /* Only the final chunk's logits can ever be observed - `lg` is
           overwritten every iteration and only the last one is returned
           - so every earlier chunk skips the vocabulary projection. See
           skip_logits in forward_chunk. */
        /* --ced's ONLY entry. lz_forward_batch observes the final chunk's
           logits and nothing else - that is what skip_logits already
           encodes - so the earlier chunks' outputs of the last layer are
           unobserved and the layer can write its K/V and stop. Every
           other forward_chunk caller wants more than that (verify wants
           every position's logits, capture wants every position's
           hidden), so they pass 0 and CED stays out of them.

           ced_replay_w is scoped to THIS entry point for the same
           reason: the report's own bounded-replay mechanism is a
           PROMPT PREFILL optimization ("only prefills the last win
           tokens of the prompt"), not a general per-call hook - verify/
           capture batches below process short, already-generated
           continuations, not prompt prefill, so they intentionally do
           not carry this check. */
        if (replay_boundary >= 0) {
            /* Force a chunk boundary exactly at the reset point rather
               than resetting mid-chunk: forward_kda has no per-token
               reset hook, and none is needed - splitting the chunk here
               reuses the ring's existing "read ring_base once per
               forward_chunk call" contract instead of adding one. */
            if (done < replay_boundary && done + k > replay_boundary)
                k = replay_boundary - done;
            else if (done == replay_boundary)
                lz_ced_replay_reset(m, s);
        }
        lg = forward_chunk(m, s, tokens + done, k, pos0 + done, NULL, 0, NULL,
                           done + k < n, s->ced && (done + k < n));
        if (!lg) return NULL;
        done += k;
    }
    /* Written after the whole loop, so a sliced prefill leaves the LAST
       slice's row - which is the post-prefill row, the only one any
       caller observes. */
    if (lg && lz_dump_prefill_path) {
        FILE *f = fopen(lz_dump_prefill_path, "wb");
        if (f) {
            fwrite(lg, sizeof(float), (size_t)m->config.vocab_size, f);
            fclose(f);
        }
        lz_debug_prefill_dumps++;
    }
    return lg;
}


int lz_forward_verify(const LZModel *m, LZRunState *s,
                      const int *tokens, int n, int pos0) {
    const LZModelConfig *c = &m->config;
    int cap, done = 0;
    float t_us0 = lz_time_ms() * LZ_US_SCALE;  /* debug probe, see part 4 */

    if (!tokens || n < 1 || !m->mtp) return 1;
    cap = s->nt_cap;
    if (cap < 1) cap = 1;
    if (cap > LZ_BATCH_MAX) cap = LZ_BATCH_MAX;
    /* Same chunking shape as lz_forward_batch - bit-identical to token-
       by-token regardless of chunk width (forward.h's hard gate on
       lz_forward_batch applies unchanged here; this function only adds
       a different lm_head step, the per-layer body is untouched). Each
       chunk's logits land at its own offset into s->mtp_logits, and
       each chunk's RAW pre-final-norm hidden states land at the same
       offset into s->mtp_verify_hidden - lz_spec_round reads row
       n_accept of this as the next round's draft seed, letting the
       main loop skip an entire redundant per-round forward pass
       (forward.h's own comment on this field). */
    while (done < n) {
        int k = n - done;
        if (k > cap) k = cap;
        /* skip_logits is 0 here and must stay 0: verify wants EVERY
           position's logits (all_logits=1 writes them into mtp_logits),
           which is the opposite of the prefill case. */
        if (!forward_chunk(m, s, tokens + done, k, pos0 + done,
                           s->mtp_verify_hidden + (size_t)done * c->hidden_size,
                           1, s->mtp_logits + (size_t)done * c->vocab_size, 0, 0))
            return 1;
        done += k;
    }
    lz_debug_us_verify += (lz_i64)(lz_time_ms() * LZ_US_SCALE - t_us0);
    return 0;
}

float *lz_forward_batch_capture(const LZModel *m, LZRunState *s,
                                const int *tokens, int n, int pos0,
                                float *hidden_out) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size;
    float *lg = NULL;
    int cap, done = 0;

    if (!tokens || n < 1 || !hidden_out) return NULL;
    cap = s->nt_cap;
    if (cap < 1) cap = 1;
    if (cap > LZ_BATCH_MAX) cap = LZ_BATCH_MAX;
    /* Same chunking shape as lz_forward_batch - see forward.h's own
       comment on this function: each chunk's capture_hidden write
       (forward_chunk, above) lands at its own offset into hidden_out,
       so the assembled buffer is exactly n*dim regardless of chunk
       width, same as lz_forward_verify's mtp_logits offsetting. */
    while (done < n) {
        int k = n - done;
        if (k > cap) k = cap;
        lg = forward_chunk(m, s, tokens + done, k, pos0 + done,
                           hidden_out + (size_t)done * dim, 0, NULL,
                           done + k < n, 0);
        if (!lg) return NULL;
        done += k;
    }
    return lg;
}

/* Batched verify, generalized - forward.h's own comment has the full
   contract and why lz_forward_verify above cannot serve this caller
   (its `!m->mtp` gate and its s->mtp_logits destination, allocated only
   when a head is bound). Shares forward_chunk with lz_forward_verify:
   same chunking, same all_logits=1 per-position logits, same ring
   advance - the only difference is the output buffer and the missing
   m->mtp requirement. */
int lz_forward_verify_into(const LZModel *m, LZRunState *s,
                           const int *tokens, int n, int pos0,
                           float *logits_out) {
    const LZModelConfig *c = &m->config;
    int cap, done = 0;

    if (!tokens || n < 1 || !logits_out) return 1;
    cap = s->nt_cap;
    if (cap < 1) cap = 1;
    if (cap > LZ_BATCH_MAX) cap = LZ_BATCH_MAX;
    while (done < n) {
        int k = n - done;
        if (k > cap) k = cap;
        /* capture_hidden NULL: PLD does not chain a draft head off a
           captured hidden state (lz_pld_round's draft comes from a
           prompt n-gram match, not a forward pass), so there is nothing
           to capture here, unlike lz_forward_verify's own
           s->mtp_verify_hidden write. skip_logits stays 0 - every
           position's logits are wanted, same as lz_forward_verify. */
        if (!forward_chunk(m, s, tokens + done, k, pos0 + done,
                           NULL, 1, logits_out + (size_t)done * c->vocab_size, 0, 0))
            return 1;
        done += k;
    }
    return 0;
}
