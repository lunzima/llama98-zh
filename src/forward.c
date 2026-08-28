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


/* ------------------------------------------------------------ state allocation */

/* 16-byte aligned calloc, so the SSE kernels that read these buffers can
   use movaps/movdqa (aligned) instead of movups/movdqu. Over-allocates
   by one pointer plus the pad and stores the ORIGINAL pointer just before
   the aligned one; xfree hands that back to free(). The SSE1/SSE2 loads
   in the operators that read these buffers then become aligned loads -
   the "unaligned loads throughout" note in ops_mmx_sse.c is what this
   removes. Not every engine buffer goes through here (the tensor slices
   from model.c do not), so not every _mm_loadu_* is convertible; only the
   ones whose pointer descends from xcalloc and whose offset is a multiple
   of 4 floats are. */
static void *xcalloc(size_t n, size_t sz, int *ok, lz_i64*acc) {
    char *base, *aligned;
    if (n == 0) return NULL;
    base = (char *)calloc(n * sz + 16 + sizeof(void *), 1);
    if (!base) { *ok = 0; return NULL; }
    aligned = base + sizeof(void *);
    aligned += (size_t)(16 - ((size_t)aligned & 15)) & 15;
    ((void **)(void *)aligned)[-1] = base;
    *acc += (lz_i64)n * (lz_i64)sz;
    return aligned;
}

static void xfree(void *p) {
    if (p) free(((void **)p)[-1]);
}





#if LZ_CONV_FIXED
/* Quantize the conv taps once, laying them out per channel in the same
   order forward_kda/forward_ssm index the history.
 *
 * The offsets are duplicated from those two functions, which is a drift
 * surface, so the loop asserts that the channels it covered add up to
 * lin_conv_dim. A layout change then fails here instead of quantizing
 * the wrong tensor into the right slot. */
static int conv_fixed_build(LZRunState *s, const LZModel *m) {
    const LZModelConfig *c = &m->config;
    int layer, t, covered_ok = 1, built = 0;
    for (layer = 0; layer < c->n_layers; layer++) {
        const LZLayer *L;
        int li;
        size_t base;
        if (c->layer_types[layer] == LZ_LT_FULL) continue;
        L = &m->layers[layer];
        li = s->cache_idx[layer];
        base = (size_t)li * c->lin_conv_dim;
        built++;
        {
        size_t ch = 0;
        const LZTensor *tv[3];
        int cnt[3], ntv;
        if (L->kda_q_conv1d.q || L->kda_q_conv1d.f) {
            tv[0] = &L->kda_q_conv1d; cnt[0] = c->lin_key_dim;
            tv[1] = &L->kda_k_conv1d; cnt[1] = c->lin_key_dim;
            tv[2] = &L->kda_v_conv1d; cnt[2] = c->lin_value_dim;
            ntv = 3;
        } else {
            tv[0] = &L->conv1d; cnt[0] = c->lin_conv_dim; ntv = 1;
        }
        for (t = 0; t < ntv; t++) {
            const float *w = lz_t_f32(tv[t], s->wscr);
            int i;
            if (!w) return -1;
            for (i = 0; i < cnt[t]; i++) {
                signed char e = (signed char)lz_conv_norm_pow2(
                    w + (size_t)i * c->conv_kernel, c->conv_kernel,
                    s->conv_mw + (base + ch + i) * c->conv_kernel,
                    s->conv_bound);
                float oscale = pow2f(-((int)e + LZ_CONV_ES));
                lz_sig_q15_fold(oscale, &s->conv_sig_k1[base + ch + i],
                                &s->conv_sig_oscale2[base + ch + i]);
#if LZ_CONV_SIG_I
                /* The same exponent, undivided. Measured 22..39 on both
                   checkpoints, so it fits signed char with room; a model
                   whose taps put it outside sigmoid_q15_i's 0..50 is not
                   refused here - that entry rebuilds a float for such a
                   channel and only the saving is lost. */
                s->conv_sig_e[base + ch + i] = (signed char)((int)e + LZ_CONV_ES);
#endif /* LZ_CONV_SIG_I */
            }
            ch += (size_t)cnt[t];
        }
        if (ch != (size_t)c->lin_conv_dim) covered_ok = 0;
        }
    }
    /* Two coverage checks, not one: `covered_ok` catches a layer whose
       tensors do not add up to lin_conv_dim, `built` catches the case
       where the loop matched no layer at all - which would leave every
       tap zero and report success. */
    if (!covered_ok || built != c->n_linear_layers) return -1;
    s->conv_in_scale = pow2f(LZ_CONV_ES);
    return 0;
}
#endif /* LZ_CONV_FIXED */

#if LZ_KGATE_I16
/* dt_bias in kda_gate's own integer domain, laid out per linear layer.
 *
 * This is what makes the decay gate's `gt[i] + dt_bias[i]` an INTEGER
 * add once kda_gate arrives as int16: with both terms at 2^-LZ_KGATE_ES
 * the sum is one too, and the only float work left per channel is the
 * one multiply and one add that reach sigmoid_q15_t's coordinate. Built
 * once here rather than per token for the same reason conv_mw is - it
 * is a function of the weights alone.
 *
 * REFUSES rather than clamps. A channel whose |dt_bias| * 2^ES does not
 * fit int16 gets the whole table rejected and every KDA layer keeps its
 * float exit; clamping instead would move the decay gate on a checkpoint
 * nobody measured, silently, which is exactly what LZ_CONV_ES's comment
 * warns about one tier over. MEASURED on kmr20: max |dt_bias| 13.062,
 * i.e. 6,688 of 32,767 at ES = 9, 4.9x of headroom.
 *
 * Returns 1 on success. A GDN linear layer has no kda_dt_bias at all, so
 * a model mixing the two refuses here as well - it has no kda_gate to
 * feed either. */
static int kda_gate_build(LZRunState *s, const LZModel *m) {
    const LZModelConfig *c = &m->config;
    int layer, built = 0;
    int nvk = c->lin_n_v_heads * c->lin_k_head_dim;
    float sc = pow2f(LZ_KGATE_ES);
    if (!s->kda_dtb_i16 || nvk <= 0) return 0;
    for (layer = 0; layer < c->n_layers; layer++) {
        const LZLayer *L = &m->layers[layer];
        const float *db;
        int li, i;
        if (c->layer_types[layer] == LZ_LT_FULL) continue;
        if (L->kda_dt_bias.n == 0) return 0;
        li = s->cache_idx[layer];
        db = lz_t_f32(&L->kda_dt_bias, s->wscr);
        if (!db) return 0;
        for (i = 0; i < nvk; i++) {
            /* Half-away-from-zero, the convention epi_align_i16 rounds
               the other addend with - one domain, one rounding rule. */
            float v = db[i] * sc;
            long q = (long)(v >= 0.0f ? v + 0.5f : v - 0.5f);
            if (q > 32767 || q < -32767) return 0;
            s->kda_dtb_i16[(size_t)li * nvk + i] = (short)q;
        }
        built++;
    }
    return built == c->n_linear_layers;
}
#endif /* LZ_KGATE_I16 */

int lz_state_alloc(LZRunState *s, const LZModel *m, int seq_len,
                       int spec_k_max, char *errbuf, int errlen) {
    const LZModelConfig *c = &m->config;
    int ok = 1, l, nf = 0, nl = 0;
    /* Activation buffers are allocated for LZ_BATCH_MAX slices. The
       generation phase only uses slice 0; the rest is wasted - but for
       s1v3 that is ~0.25 MB, well under 1% of the weights under any
       recipe still on the table (see LZ_BATCH_MAX in ops.h - the byte
       figure is not settled), and the alternative is a batch-width
       parameter that would infect every state_alloc caller.

       nt_cap starts at LZ_BATCH_DEFAULT, which is <= LZ_BATCH_MAX: the
       buffers are sized for the CAP so a caller may raise nt_cap up to
       it at any time (CLI --batch), but the width that actually runs is
       the DEFAULT. The two are separate because the optimal width is a
       property of the target's L1, and the only machine we can measure
       is not a target - see LZ_BATCH_MAX's comment for the sweep and
       for the mechanism by which the answer can reverse on a PII. */
    int nt = LZ_BATCH_MAX;

    memset(s, 0, sizeof(*s));
    if (seq_len <= 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_SEQ_LEN);
        return 1;
    }
    s->seq_len = seq_len;
    s->nt_cap = LZ_BATCH_DEFAULT;

    s->x       = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
    s->xb      = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
    s->xb2     = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
    /* Sized for the LARGER of the body's dense-FFN width and the MTP
       block's own (dense_ffn_step's shared scratch, forward.c) - the
       two are independent fields (model.h's mtp_intermediate_size
       comment) and neither is guaranteed to
       bound the other (kunmoe-v2: body intermediate_size 512 is an
       unrelated placeholder, a carved MTP head can be wider). */
    {
        int hb_dim = c->intermediate_size;
        if (m->mtp && c->mtp_intermediate_size > hb_dim) hb_dim = c->mtp_intermediate_size;
        s->hb  = (float *)xcalloc((size_t)nt * hb_dim, sizeof(float), &ok, &s->bytes_alloc);
        s->hb2 = (float *)xcalloc((size_t)nt * hb_dim, sizeof(float), &ok, &s->bytes_alloc);
    }
    s->logits  = (float *)xcalloc((size_t)c->vocab_size, sizeof(float), &ok, &s->bytes_alloc);

    s->qg       = (float *)xcalloc((size_t)nt * c->attn_qgate_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->qh       = (float *)xcalloc((size_t)nt * c->attn_q_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->att      = (float *)xcalloc((size_t)seq_len, sizeof(float), &ok, &s->bytes_alloc);
#if (LZ_ATTN_FIXED & 2)
    /* Allocated whenever the kernel is compiled, not when the tier is
       selected: lz_float_ref_select can run after this point, and 6 bytes per
       context slot (196 KB at ctx 32768, counted in bytes_alloc) is a
       cheaper bargain than a buffer that appears only on some paths. */
    s->wsum_cbuf = (float *)xcalloc((size_t)seq_len, sizeof(float), &ok, &s->bytes_alloc);
    s->wsum_cq   = (int16_t *)xcalloc((size_t)seq_len, sizeof(int16_t), &ok, &s->bytes_alloc);
#endif /* LZ_ATTN_FIXED & 2 */
    s->attn_out = (float *)xcalloc((size_t)nt * c->attn_q_dim, sizeof(float), &ok, &s->bytes_alloc);
    /* Fixed-tier attention int path (--attn-int): attn_acc holds the exact
       int64 weighted sum and post-gate values, attn_ss the per-32-group
       dequant scale. Sized even when the knob is off, same reasoning as
       wsum_cbuf/wsum_cq: the selector can run after this point. */
    s->attn_acc = (int64_t *)xcalloc((size_t)nt * c->attn_q_dim, sizeof(int64_t), &ok, &s->bytes_alloc);
    s->attn_ss  = (float *)xcalloc((size_t)nt * (c->attn_q_dim >> 5), sizeof(float), &ok, &s->bytes_alloc);
    s->ktmp     = (float *)xcalloc((size_t)nt * c->attn_kv_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->vtmp     = (float *)xcalloc((size_t)nt * c->attn_kv_dim, sizeof(float), &ok, &s->bytes_alloc);
    /* KV cache Q8 (group 32, contiguous within a row). +1 layer slot when
       an MTP head is bound: its own full_attention block reuses this
       cache (index n_full_layers, see forward_attn's sentinel handling)
       rather than a separate buffer - same reasoning as everywhere else
       in this engine that a second copy of a KV cache was avoided. */
    {
        int kv_layers = c->n_full_layers + (m->mtp ? 1 : 0);
        s->kfmt = lz_kv_kfmt;
        s->vfmt = lz_kv_vfmt;
        /* Ring sizing, before anything is allocated. kv_slots is what the
           cache planes are actually indexed by; kv_slot() maps absolute
           positions into it. */
        /* lz_attn_window < 0 means AUTO, which is the default. The value
           is derived here rather than baked into the global because it
           depends on seq_len, which the global cannot see.

           AUTO = max(LZ_ATTN_WIN_FLOOR, seq_len/2), and it is chosen to
           be >= the measured-free point in BOTH readings of the data
           above. Read the numbers again: window 240 costs +5.351% at
           ctx 1024 and +5.428% at ctx 2048 - practically the same. The
           harm tracks the window's ABSOLUTE size, not its ratio to the
           context, which is why the floor is absolute. window 1008 at
           ctx 2048 is -0.142%, i.e. free, so the floor sits just above
           it at 1024. The seq_len/2 term only ever RAISES the window
           beyond the floor, and a wider window is strictly closer to
           full attention - so no ctx can land on a configuration more
           aggressive than one that was measured free.

           Nothing is claimed about ctx far past 2048: the measurements
           stop there. AUTO stays conservative there by construction
           (window grows with ctx), not by evidence. */
        if (lz_attn_window < 0) {
            int w = seq_len >> 1;
            if (w < LZ_ATTN_WIN_FLOOR) w = LZ_ATTN_WIN_FLOOR;
            /* A window that cannot be reached is not a window. Leaving
               it at 0 keeps the whole eviction path - and the ring, and
               the row skip - out of the picture for short contexts,
               which is what makes short-prompt behaviour bit-identical
               to the pre-default engine. */
            s->attn_win = (w + LZ_ATTN_SINK_DEFAULT < seq_len) ? w : 0;
        } else {
            s->attn_win = lz_attn_window > 0 ? lz_attn_window : 0;
        }
        /* A window with NO sink is the catastrophic configuration, not a
           cheaper one: measured +43.764%% PPL against +5.351%% with a
           sink of 16, on the same window budget. Nobody should be able to
           reach it by forgetting a flag, so a window on its own gets a
           sink. Setting --attn-sink 0 explicitly still gets zero, because
           that arm has to stay reachable for the comparison. */
        s->attn_sink = lz_attn_sink > 0 ? lz_attn_sink
                     : (lz_attn_sink < 0 ? 0
                        : (s->attn_win > 0 ? LZ_ATTN_SINK_DEFAULT : 0));
        s->kv_ring   = 0;
        s->kv_slots  = seq_len;
        if (s->attn_win > 0) {
            /* +LZ_SPEC_K_MAX+1 guard slots, see kv_slot()'s comment: a
               rejected speculative draft must only be able to clobber
               positions the read window has already dropped. */
            int ring = s->attn_win + LZ_SPEC_K_MAX + 1;
            if (s->attn_sink + ring < seq_len) {
                s->kv_ring  = ring;
                s->kv_slots = s->attn_sink + ring;
            }
            /* else the "ring" would be bigger than the context: leave it
               off rather than pretend to evict. */
        }
        /* Allocate ONLY the plane each side actually uses. Allocating a
           format that is not selected is not free bookkeeping: keeping
           the Q8 planes alongside the 4-bit ones makes the 4-bit cache
           cost MORE than the 8-bit one (29.1 MB vs 28.8 at ctx 2048) -
           the whole point inverted. */
        {
            size_t nrow = (size_t)kv_layers * s->kv_slots * c->attn_kv_dim;
            size_t nsc  = (size_t)kv_layers * s->kv_slots * c->n_kv_heads;
            if (s->kfmt == LZ_KVF_Q8) {
                s->kq8 = (int8_t *)xcalloc(nrow, 1, &ok, &s->bytes_alloc);
                s->ksq = (float *)xcalloc(nrow / 32, sizeof(float),
                                          &ok, &s->bytes_alloc);
#if LZ_KV_2PLANE
                s->kq8_lo = (int8_t *)xcalloc(nrow, 1, &ok, &s->bytes_alloc);
#endif /* LZ_KV_2PLANE */
            } else if (s->kfmt == LZ_KVF_F32) {
                s->kf32 = (float *)xcalloc(nrow, sizeof(float),
                                           &ok, &s->bytes_alloc);
            } else {
                s->k4 = (unsigned char *)xcalloc(nrow / 2, 1, &ok,
                                                 &s->bytes_alloc);
                s->ks4 = (float *)xcalloc(nsc, sizeof(float),
                                          &ok, &s->bytes_alloc);
            }
            if (s->vfmt == LZ_KVF_Q8) {
                s->vq8 = (int8_t *)xcalloc(nrow, 1, &ok, &s->bytes_alloc);
                s->vsq = (float *)xcalloc(nrow / 32, sizeof(float),
                                          &ok, &s->bytes_alloc);
#if LZ_KV_2PLANE
                s->vq8_lo = (int8_t *)xcalloc(nrow, 1, &ok, &s->bytes_alloc);
#endif /* LZ_KV_2PLANE */
            } else if (s->vfmt == LZ_KVF_F32) {
                s->vf32 = (float *)xcalloc(nrow, sizeof(float),
                                           &ok, &s->bytes_alloc);
            } else {
                s->v4 = (unsigned char *)xcalloc(nrow / 2, 1, &ok,
                                                 &s->bytes_alloc);
                s->vs4 = (float *)xcalloc(nsc, sizeof(float),
                                          &ok, &s->bytes_alloc);
            }
        }
    }
#if LZ_KV_2PLANE
    s->att_lo  = (float *)xcalloc((size_t)seq_len, sizeof(float), &ok, &s->bytes_alloc);
    s->wsum_lo = (float *)xcalloc((size_t)c->head_dim, sizeof(float), &ok, &s->bytes_alloc);
#endif /* LZ_KV_2PLANE */

    s->qkv    = (float *)xcalloc((size_t)nt * c->lin_conv_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->qkv_c  = (float *)xcalloc((size_t)nt * c->lin_conv_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->zbuf   = (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->avec   = (float *)xcalloc((size_t)nt * c->lin_n_v_heads, sizeof(float), &ok, &s->bytes_alloc);
    s->bvec   = (float *)xcalloc((size_t)nt * c->lin_n_v_heads, sizeof(float), &ok, &s->bytes_alloc);
    s->ssm_out= (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->ssm_sig = (int32_t *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(int32_t), &ok, &s->bytes_alloc);
    /* nt-wide, not one head-dim wide - allocated to match the other
       batched buffers; the serial recurrence only ever touches row 0.
       `nt` here is LZ_BATCH_MAX (line 407), the cap, not the current
       width - same rule the other batched buffers follow. */
    s->qn     = (float *)xcalloc((size_t)nt * c->lin_k_head_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kn     = (float *)xcalloc((size_t)nt * c->lin_k_head_dim, sizeof(float), &ok, &s->bytes_alloc);

    /* LZ_LT_KDA scratch. Zero-sized (NULL) when the model has no KDA
       layers (kda_gate_rank == 0, lin_key_dim/lin_n_v_heads still fine
       to multiply by since they'd only be 0 too on a config with no
       linear-attention layers at all). */
    s->kda_q  = (float *)xcalloc((size_t)nt * c->lin_key_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_k  = (float *)xcalloc((size_t)nt * c->lin_key_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_v  = (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_qc = (float *)xcalloc((size_t)nt * c->lin_key_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_kc = (float *)xcalloc((size_t)nt * c->lin_key_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_vc = (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_gate_lat = (float *)xcalloc((size_t)nt * c->kda_gate_rank, sizeof(float), &ok, &s->bytes_alloc);
    s->kda_gate     = (float *)xcalloc((size_t)nt * c->lin_n_v_heads * c->lin_k_head_dim,
                                       sizeof(float), &ok, &s->bytes_alloc);

    /* latent MoE scratch - see forward.h. Zero-sized when the model has
       no MoE layers (num_experts == 0).

       TWO WIDTHS, split on the routing boundary. The router, the latent
       down/up projections and the shared expert read one weight stream
       for every token in the chunk, so they run batched and their
       buffers are nt-wide like every other block here. The ROUTED
       experts cannot: which expert a token wants is a property of that
       token, so moe_h2 and the per-expert selection stay one token's
       worth. See forward_moe. */
    {
        int moe_scratch_w = c->moe_intermediate_size;
        if (c->moe_shared_width > moe_scratch_w) moe_scratch_w = c->moe_shared_width;
        s->moe_router_logits = (float *)xcalloc((size_t)nt * c->num_experts, sizeof(float), &ok, &s->bytes_alloc);
        /* Sized to the EXPERT COUNT, not to num_experts_per_token, so
           --moe-topk can go up as well as down. At 16 experts that is
           128 bytes total - cheaper than the clamp it removes, and a
           clamp that silently only works downwards is exactly the kind
           of knob that produces a one-sided sweep. */
        s->moe_sel_idx = (int *)xcalloc((size_t)c->num_experts, sizeof(int), &ok, &s->bytes_alloc);
        s->moe_sel_w   = (float *)xcalloc((size_t)c->num_experts, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_lat_x = (float *)xcalloc((size_t)nt * c->moe_latent_dim, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_lat_y = (float *)xcalloc((size_t)nt * c->moe_latent_dim, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_h1 = (float *)xcalloc((size_t)nt * moe_scratch_w, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_h3 = (float *)xcalloc((size_t)nt * moe_scratch_w, sizeof(float), &ok, &s->bytes_alloc);
        /* One token's worth: this is the routed loop's own temporary,
           written and consumed inside one (token, expert) step. */
        s->moe_h2 = (float *)xcalloc((size_t)c->moe_latent_dim, sizeof(float), &ok, &s->bytes_alloc);
        s->moe_shared_out = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
    }
    /* MTP draft head scratch - only when a head is bound (see forward.h).
       mtp_x/mtp_concat/mtp_emb_raw are nt_cap-wide: the chained
       per-step draft path (lz_mtp_draft_step) only ever touches
       slot 0, but lz_mtp_prefill batches the SAME buffers across nt
       prompt positions at once (forward.h's own comment on why this is
       one set of scratch, not two). mtp_chain/mtp_draft_logits stay
       single-token - the draft chain register and one step's logits are
       inherently one-token concepts, prefill never touches either. */
    if (m->mtp) {
        s->mtp_x     = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_concat = (float *)xcalloc((size_t)nt * 2 * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_emb_raw = (float *)xcalloc((size_t)nt * c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_chain = (float *)xcalloc((size_t)c->hidden_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_draft_logits = (float *)xcalloc((size_t)c->vocab_size, sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_logits = (float *)xcalloc((size_t)(LZ_SPEC_K_MAX + 1) * c->vocab_size,
                                         sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_verify_hidden = (float *)xcalloc((size_t)(LZ_SPEC_K_MAX + 1) * c->hidden_size,
                                                sizeof(float), &ok, &s->bytes_alloc);
        /* temp>0 speculative decoding phase 2: one row per DRAFTED
           token (LZ_SPEC_K_MAX, not +1 - the verify batch's own
           bonus/final row has no draft counterpart to accept/reject
           against, see forward.h's own comment). */
        s->mtp_draft_q = (float *)xcalloc((size_t)LZ_SPEC_K_MAX * c->vocab_size,
                                          sizeof(float), &ok, &s->bytes_alloc);
        s->mtp_target_p = (float *)xcalloc((size_t)c->vocab_size,
                                           sizeof(float), &ok, &s->bytes_alloc);
    }

    /* SSM/conv state: one (kd*vd) block per (layer, head), grouped
       within rows, times ssm_ring_depth slots - see forward.h's own
       comment on s->ssm_slot/s->ssm_ring_depth for the ring's shape
       and the "zero extra bandwidth, only memory" argument it depends
       on. Depth is 1 (no ring, byte-identical to the single-slot
       version) when there is no MTP head to draft with at all - a
       model that can never run --spec never allocates slots it cannot
       use - OR when the CALLER declares it will never request --spec
       (spec_k_max <= 0, this function's own parameter comment,
       forward.h). Otherwise spec_k_max+1, clamped to LZ_SPEC_K_MAX+1 -
       a caller that only ever wants k<=2 should not pay for slots
       3..6. Stored on s rather than recomputed at every site that
       needs it - see s->ssm_ring_depth's own comment for why. */
    if (!m->mtp || spec_k_max <= 0) {
        s->ssm_ring_depth = 1;
    } else {
        int d = spec_k_max + 1;
        if (d > LZ_SPEC_K_MAX + 1) d = LZ_SPEC_K_MAX + 1;
        s->ssm_ring_depth = d;
    }
    s->ssm_state_q8 = (int8_t *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_n_v_heads *
        c->lin_k_head_dim * c->lin_v_head_dim, 1, &ok, &s->bytes_alloc);
#if LZ_GDN_STATE_2PLANE
    s->ssm_state_q8_lo = (int8_t *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_n_v_heads *
        c->lin_k_head_dim * c->lin_v_head_dim, 1, &ok, &s->bytes_alloc);
#endif /* LZ_GDN_STATE_2PLANE */
    s->ssm_state_s = (float *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_n_v_heads *
        c->lin_k_head_dim * c->lin_v_head_dim / 32,
        sizeof(float), &ok, &s->bytes_alloc);
    s->conv_state = (float *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_conv_dim *
        (c->conv_kernel - 1), sizeof(float), &ok, &s->bytes_alloc);

    /* Fixed conv tier. Decided ONCE, here, and recorded in s->conv_fixed
       rather than re-read per token: lz_conv_mode() can be moved by a
       flag, and a state whose buffers were not allocated must not start
       taking the fixed branch halfway through a run.
     *
     * The history exponent was the one parameter that kept this tier
     * unwired for a while, and the answer it eventually got is
     * LZ_CONV_ES: a CONSTANT power of two, measured once from the real
     * conv input distribution rather than derived per token, because the
     * history starts at zero and has nothing to derive one from. The
     * recurrence's answer (a scale that TRAVELS WITH THE STATE,
     * ssm_state_s / p2_group_scale) would cost a renormalization pass
     * over the history every time the range moved, which is the cost
     * this tier exists to avoid. See LZ_CONV_ES for the distribution it
     * was measured against and for the fact that a value past 32 clips
     * SILENTLY - re-run that probe on any model whose conv inputs might
     * be scaled differently. */
    s->conv_fixed = lz_conv_mode();
    if (s->conv_fixed) {
        size_t nch = (size_t)c->n_linear_layers * c->lin_conv_dim;
        s->conv_bound = lz_conv_accum_bound(c->conv_kernel);
        s->conv_state_q = (short *)xcalloc(
            (size_t)s->ssm_ring_depth * nch * (c->conv_kernel - 1),
            sizeof(short), &ok, &s->bytes_alloc);
        s->conv_mw = (short *)xcalloc(nch * c->conv_kernel,
                                      sizeof(short), &ok, &s->bytes_alloc);
        /* Sized like kda_q/kda_k/kda_v above, in int16. Zero-sized (and
           so NULL) on a model with no KDA layers, which forward_kda's
           own predicate then reads as "no int path". */
        s->kda_q_i16 = (short *)xcalloc((size_t)nt * c->lin_key_dim,
                                        sizeof(short), &ok, &s->bytes_alloc);
        s->kda_k_i16 = (short *)xcalloc((size_t)nt * c->lin_key_dim,
                                        sizeof(short), &ok, &s->bytes_alloc);
        s->kda_v_i16 = (short *)xcalloc((size_t)nt * c->lin_value_dim,
                                        sizeof(short), &ok, &s->bytes_alloc);
#if LZ_CONVO_I16
        /* The conv's own int16 exit - see forward.h. */
        s->kda_qc_i16 = (short *)xcalloc((size_t)nt * c->lin_key_dim,
                                         sizeof(short), &ok, &s->bytes_alloc);
        s->kda_kc_i16 = (short *)xcalloc((size_t)nt * c->lin_key_dim,
                                         sizeof(short), &ok, &s->bytes_alloc);
        s->kda_vc_i16 = (short *)xcalloc((size_t)nt * c->lin_value_dim,
                                         sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_CONVO_I16 */
        s->conv_sig_k1 = (float *)xcalloc(nch, sizeof(float), &ok,
                                          &s->bytes_alloc);
        s->conv_sig_oscale2 = (float *)xcalloc(nch, sizeof(float), &ok,
                                               &s->bytes_alloc);
#if LZ_CONV_SIG_I
        s->conv_sig_e = (signed char *)xcalloc(nch, sizeof(signed char), &ok,
                                               &s->bytes_alloc);
#endif /* LZ_CONV_SIG_I */
        s->conv_sig_k2 = lz_sig_q15_t_offset();
        if (!ok) s->conv_fixed = 0;
        /* conv_fixed_build runs LATER, after cache_idx is filled - it
           needs the layer -> linear-index map and that is populated
           further down. Building here would read an all-zero map and
           quantize every layer's taps into slot 0. */
    }
    /* s->ssm_slot starts at 0 from this function's own memset(s,0,...)
       above - explicit here only as documentation, matching mtp_pos's
       own "reset to 0" comment; lz_state_reset (below) is the one that
       actually has to set it back to 0 on every subsequent reuse. */

    s->cache_idx = (int *)xcalloc((size_t)c->n_layers, sizeof(int), &ok, &s->bytes_alloc);

    /* Q8 activation quantization buffers: capacity = max matmul input dim */
    {
        int qcap = c->hidden_size;
        int nvk = c->lin_n_v_heads * c->lin_k_head_dim;
        if (c->intermediate_size > qcap) qcap = c->intermediate_size;
        if (c->attn_q_dim > qcap) qcap = c->attn_q_dim;
        if (c->lin_conv_dim > qcap) qcap = c->lin_conv_dim;
        if (c->attn_qgate_dim > qcap) qcap = c->attn_qgate_dim;
        /* KDA's f_b_proj (kda_gate_rank -> nvk) and latent MoE's
           per-expert matmuls (moe_latent_dim <-> moe_intermediate_size)
           and shared expert (hidden <-> moe_shared_width) all quantize
           through lz_matmul_w with s->xq/s->xqs as scratch - every one
           of their in_dims must fit, not just the dense-path ones above. */
        if (c->kda_gate_rank > qcap) qcap = c->kda_gate_rank;
        if (nvk > qcap) qcap = nvk;
        if (c->moe_latent_dim > qcap) qcap = c->moe_latent_dim;
        if (c->moe_intermediate_size > qcap) qcap = c->moe_intermediate_size;
        if (c->moe_shared_width > qcap) qcap = c->moe_shared_width;
        /* MTP's fc reads 2*hidden_size (lz_mtp_draft_step quantizes
           s->mtp_concat through lz_matmul_w with this same scratch) -
           not covered by any of the dims above on a model whose
           attention/FFN widths are all below 2x hidden. Its dense FFN
           (dense_ffn_step's gate/up_proj quantization) reads
           mtp_intermediate_size, independent of intermediate_size -
           same reasoning as s->hb/s->hb2's sizing
           above, not assumed to be bounded by anything else here. */
        if (m->mtp && 2 * c->hidden_size > qcap) qcap = 2 * c->hidden_size;
        if (m->mtp && c->mtp_intermediate_size > qcap) qcap = c->mtp_intermediate_size;
        s->qcap = qcap;
        s->xq = (int8_t *)xcalloc((size_t)nt * qcap, 1, &ok, &s->bytes_alloc);
        /* xqs sized for the worst case (in-row gs falling back to 1) */
        s->xqs = (float *)xcalloc((size_t)nt * qcap, sizeof(float), &ok, &s->bytes_alloc);
        /* AT LEAST LZ_MM_WIDEN_MAX, so the loader's bf16 rule is sound
           by construction rather than by inspection. model.c accepts a
           1-D tensor as bf16 when it is no longer than that constant,
           and lz_t_f32 widens such a tensor WHOLE into this buffer -
           two bounds that have to agree, and did only because every
           1-D tensor in the checkpoints to hand happens to be a norm
           or a per-head bias and so is already counted in qcap above.
           A model with a wider one would have overrun this. The cost
           is at most 16 KB. */
        {   int wcap = qcap;
            if (wcap < LZ_MM_WIDEN_MAX) wcap = LZ_MM_WIDEN_MAX;
            s->wscr = (float *)xcalloc((size_t)wcap, sizeof(float), &ok,
                                       &s->bytes_alloc); }
        /* SubLN Hadamard scratch: two n-wide int32 halves, n <= qcap. */
        s->fwht_scratch = (int32_t *)xcalloc((size_t)2 * qcap, sizeof(int32_t),
                                             &ok, &s->bytes_alloc);
        /* SubLN plain-norm int output scratch: one n-wide int32 row,
           n <= qcap - see forward.h's field comment. */
        s->subn_norm_int = (int32_t *)xcalloc((size_t)qcap, sizeof(int32_t),
                                              &ok, &s->bytes_alloc);
        /* ---- q16_0's int16 exits (int-pipeline milestone 5) ----------
           One buffer per chain, each the int16 twin of the float one it
           replaces for a token - exactly one of the two is written per
           call, the shape milestone 3's kda_q_i16 established. All are
           zero-sized (so NULL) on a model without the producing tensor,
           and every predicate below reads NULL as "no int path". */
#if LZ_BVEC_I16
        s->bvec_i16 = (short *)xcalloc((size_t)nt * c->lin_n_v_heads,
                                       sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_BVEC_I16 */
#if LZ_MOEGATE_I16
        s->moe_logits_i16 = (short *)xcalloc((size_t)nt * c->num_experts,
                                             sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_MOEGATE_I16 */
#if LZ_KLAT_I16
        s->kda_lat_i16 = (short *)xcalloc((size_t)nt * c->kda_gate_rank,
                                          sizeof(short), &ok, &s->bytes_alloc);
        /* The widening scratch lz_quantize_q8_int's int32 input needs -
           one token's worth, since f_b_proj is a per-token matmul. An
           int-to-int copy, so it bills nothing and adds no rounding. */
        s->kda_lat_i32 = (int32_t *)xcalloc((size_t)c->kda_gate_rank,
                                            sizeof(int32_t), &ok, &s->bytes_alloc);
#endif /* LZ_KLAT_I16 */
#if LZ_MLAT_I16
        s->moe_lat_i16 = (short *)xcalloc((size_t)nt * c->moe_latent_dim,
                                          sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_MLAT_I16 */
#if LZ_VPROJ_I16
        /* Sized like s->vtmp, in int16. Zero-sized (and so NULL) on a
           model with no full-attention layers, which forward_attn's
           predicate reads as "no int path". */
        s->vtmp_i16 = (short *)xcalloc((size_t)nt * c->attn_kv_dim,
                                       sizeof(short), &ok, &s->bytes_alloc);
#endif /* LZ_VPROJ_I16 */
#if LZ_SWIGLU_I16
        /* Sized like moe_h1/moe_h3 - the wider of the routed experts'
           intermediate and the shared expert's width, since both sites
           write these. Zero-sized (and so NULL) on a dense model, which
           forward_moe's predicate reads as "no int path". */
        {
            int sw = c->moe_intermediate_size;
            if (c->moe_shared_width > sw) sw = c->moe_shared_width;
            s->moe_h1_i16 = (short *)xcalloc((size_t)nt * sw, sizeof(short),
                                             &ok, &s->bytes_alloc);
            s->moe_h3_i16 = (short *)xcalloc((size_t)nt * sw, sizeof(short),
                                             &ok, &s->bytes_alloc);
        }
#endif /* LZ_SWIGLU_I16 */
        /* Under neither LZ_MLAT_I16 nor LZ_MLAT_REUSE: the expert loop's
           hoisted activation quantize writes here on the float path, the
           int path, and the reuse control arm alike - what the switches
           select is whether the quantize is skipped, not where it lands.
           One token wide (the loop is per token) and the scale array is
           sized for the worst case, an in-row gs of 1, the same way
           s->xqs is. */
        s->moe_lat_q = (int8_t *)xcalloc((size_t)c->moe_latent_dim, 1,
                                         &ok, &s->bytes_alloc);
        s->moe_lat_qs = (float *)xcalloc((size_t)c->moe_latent_dim,
                                         sizeof(float), &ok, &s->bytes_alloc);
#if LZ_KGATE_I16
        s->kda_gate_i16 = (short *)xcalloc((size_t)nt * nvk, sizeof(short),
                                           &ok, &s->bytes_alloc);
        s->kda_dtb_i16 = (short *)xcalloc(
            (size_t)c->n_linear_layers * nvk, sizeof(short), &ok,
            &s->bytes_alloc);
#endif /* LZ_KGATE_I16 */
    }
    if (!ok) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
        lz_state_free(s);
        return 1;
    }
    /* RoPE precomputed table: pos * (rotary_dim/2) {cos, sin} pairs.
       Shared by all layers/heads under the same theta; the half-precision
       frequency formula matches the reference (denominator is
       rotary_dim, not head_dim). */
    /* RoPE/scale tables must be computed at PC=24 on the x87 targets too
       (Watcom): lz_mathf's float transcendentals are bit-identical across
       gcc and ARM only when every float operation rounds to 24 bits, and
       on a 387 that means the precision control word, not the default
       64-bit mantissa. gcc's SSE and the ARM soft-float ignore the
       control word (always 24-bit float), so begin/end is a no-op there. */
    {
        unsigned fpu = lz_fpu_float_begin();
        s->rope_cs = (float *)xcalloc((size_t)seq_len * c->rotary_dim,
                                      sizeof(float), &ok, &s->bytes_alloc);
        if (ok) {
            int half = c->rotary_dim >> 1;
            for (l = 0; l < seq_len; l++) {
                int i;
                for (i = 0; i < half; i++) {
                    float freq = lz_powf(c->rope_theta,
                                         -2.0f * (float)i / (float)c->rotary_dim);
                    float ang = (float)l * freq;
                    s->rope_cs[((size_t)l * half + i) * 2]     = lz_cosf(ang);
                    s->rope_cs[((size_t)l * half + i) * 2 + 1] = lz_sinf(ang);
                }
            }
        }
        if (!ok) {
            lz_fpu_float_end(fpu);
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
            lz_state_free(s);
            return 1;
        }

        s->attn_scale = lz_rsqrt((float)c->head_dim);
        s->ssm_scale  = lz_rsqrt((float)c->lin_k_head_dim);
        lz_fpu_float_end(fpu);
    }
    s->epoch      = 1;          /* 0 is reserved for "never allocated" */

    /* Hadamard widths (see forward.h's kv_rot_k comment). K takes the
       largest power of two that divides head_dim, V takes 64 - the sizes
       llama.cpp PR 21038 settled on. A head_dim that is not a multiple of
       64 gets NO rotation rather than a partial one. */
    s->kv_rot_k = s->kv_rot_v = 0;
    if (lz_kv_rot_enable && c->head_dim > 0 && (c->head_dim % 64) == 0) {
        int n = 64;
        while ((c->head_dim % (n * 2)) == 0) n *= 2;
        s->kv_rot_k = n;
        s->kv_rot_v = lz_kv_rot_v_dim > 0 ? lz_kv_rot_v_dim : 64;
        if (s->kv_rot_v > n) s->kv_rot_v = n;
        if ((c->head_dim % s->kv_rot_v) != 0) s->kv_rot_v = 64;
    }

    /* full and linear layers are numbered independently */
    for (l = 0; l < c->n_layers; l++)
        s->cache_idx[l] = (c->layer_types[l] == LZ_LT_FULL) ? nf++ : nl++;

#if LZ_CONV_FIXED
    /* After cache_idx, which conv_fixed_build reads. */
    if (s->conv_fixed && conv_fixed_build(s, m) != 0) s->conv_fixed = 0;
#endif /* LZ_CONV_FIXED */
#if LZ_KGATE_I16
    /* Same ordering reason as conv_fixed_build's: the table is indexed
       by linear-layer number. A refusal drops the buffer, and
       forward_kda's predicate reads that NULL as "no int path" - one
       place decides, one place asks. */
    if (!kda_gate_build(s, m)) {
        xfree(s->kda_dtb_i16);  s->kda_dtb_i16 = NULL;
        xfree(s->kda_gate_i16); s->kda_gate_i16 = NULL;
    }
#endif /* LZ_KGATE_I16 */

    return 0;
}

void lz_state_free(LZRunState *s) {
    if (!s) return;
    xfree(s->x); xfree(s->xb); xfree(s->xb2); xfree(s->hb); xfree(s->hb2);
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
    xfree(s->mtp_x); xfree(s->mtp_concat); xfree(s->mtp_emb_raw);
    xfree(s->mtp_chain); xfree(s->mtp_draft_logits); xfree(s->mtp_logits);
    xfree(s->mtp_verify_hidden);
    xfree(s->mtp_draft_q);
    xfree(s->mtp_target_p);
    xfree(s->ssm_state_q8); xfree(s->ssm_state_s);
#if LZ_GDN_STATE_2PLANE
    xfree(s->ssm_state_q8_lo);
#endif /* LZ_GDN_STATE_2PLANE */
    xfree(s->conv_state);
    xfree(s->conv_state_q); xfree(s->conv_mw);
    xfree(s->kda_q_i16); xfree(s->kda_k_i16); xfree(s->kda_v_i16);
    xfree(s->kda_qc_i16); xfree(s->kda_kc_i16); xfree(s->kda_vc_i16);
    xfree(s->bvec_i16); xfree(s->moe_logits_i16);
    xfree(s->kda_lat_i16); xfree(s->kda_lat_i32);
    xfree(s->kda_gate_i16); xfree(s->kda_dtb_i16);
    xfree(s->moe_lat_i16); xfree(s->moe_lat_q); xfree(s->moe_lat_qs);
    xfree(s->vtmp_i16);
    xfree(s->moe_h1_i16); xfree(s->moe_h3_i16);
    xfree(s->conv_sig_k1); xfree(s->conv_sig_oscale2); xfree(s->conv_sig_e);
    xfree(s->xq); xfree(s->xqs); xfree(s->wscr); xfree(s->fwht_scratch);
    xfree(s->subn_norm_int);
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
    /* The ring's own "which slot is confirmed" index - see forward.h's
       s->ssm_slot comment. Every recurrent index resets to 0 here,
       same as mtp_pos right below. */
    s->ssm_slot = 0;
    /* The MTP block's own position counter - its KV cache slot (zeroed
       above) is indexed by this, not by the body's absolute position;
       see forward.h's s->mtp_pos comment. */
    s->mtp_pos = 0;
    /* Every checkpoint taken before this point is now void: the KV cache
       it relies on (and deliberately does not copy) has just been
       zeroed. See LZStateCkpt in forward.h. */
    s->epoch++;
}

/* ------------------------------------------------ recurrent checkpoint */

/* Element counts of the three buffers a checkpoint copies. Derived here
   once so save, restore and alloc cannot disagree about the shapes -
   they did not have to, and that is exactly how one of them ends up
   copying a stale size after a config field changes. */
static void ckpt_sizes(const LZModelConfig *c, size_t *n_q8, size_t *n_s,
                       size_t *n_conv) {
    *n_q8   = (size_t)c->n_linear_layers * c->lin_n_v_heads *
              c->lin_k_head_dim * c->lin_v_head_dim;
    *n_s    = *n_q8 / 32;
    *n_conv = (size_t)c->n_linear_layers * c->lin_conv_dim *
              (c->conv_kernel - 1);
}

int lz_ckpt_alloc(LZStateCkpt *ck, const LZModel *m, char *errbuf, int errlen) {
    size_t n_q8, n_s, n_conv;
    if (!ck || !m) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    memset(ck, 0, sizeof(*ck));
    ckpt_sizes(&m->config, &n_q8, &n_s, &n_conv);
    ck->ssm_q8 = (int8_t *)malloc(n_q8);
#if LZ_GDN_STATE_2PLANE
    ck->ssm_q8_lo = (int8_t *)malloc(n_q8);
#endif /* LZ_GDN_STATE_2PLANE */
    ck->ssm_s = (float *)malloc(n_s * sizeof(float));
    ck->conv  = (float *)malloc(n_conv * sizeof(float));
#if LZ_CONV_FIXED
    /* Allocated unconditionally, not on lz_conv_mode(): a checkpoint
       outlives the flag that was set when it was made, and half the
       size of the float one is not worth a conditional that can be
       wrong. */
    ck->conv_q = (short *)malloc(n_conv * sizeof(short));
    if (!ck->conv_q) { lz_ckpt_free(ck); if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_OOM); return LZ_ERR_OOM; }
#endif /* LZ_CONV_FIXED */
    if (!ck->ssm_q8 || !ck->ssm_s || !ck->conv
#if LZ_GDN_STATE_2PLANE
        || !ck->ssm_q8_lo
#endif /* LZ_GDN_STATE_2PLANE */
        ) {
        lz_ckpt_free(ck);
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
        return LZ_ERR_STATE_ALLOC;
    }
    ck->n_ssm_q8 = n_q8;
    ck->n_ssm_s  = n_s;
    ck->n_conv   = n_conv;
    ck->pos      = -1;          /* empty until a save */
    ck->epoch    = 0;
    return LZ_ERR_OK;
}

void lz_ckpt_free(LZStateCkpt *ck) {
    if (!ck) return;
    free(ck->ssm_q8);
#if LZ_GDN_STATE_2PLANE
    free(ck->ssm_q8_lo);
#endif /* LZ_GDN_STATE_2PLANE */
    free(ck->ssm_s);
    free(ck->conv);
    free(ck->conv_q);
    memset(ck, 0, sizeof(*ck));
}

int lz_ckpt_save(LZStateCkpt *ck, const LZRunState *s, const LZModel *m,
                 int pos, char *errbuf, int errlen) {
    size_t n_q8, n_s, n_conv;
    size_t o_q8, o_s, o_conv;   /* byte offset of s's OWN active ring slot */
    if (!ck || !s || !m || pos < 0 || !ck->ssm_q8) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    ckpt_sizes(&m->config, &n_q8, &n_s, &n_conv);
    if (n_q8 != ck->n_ssm_q8 || n_s != ck->n_ssm_s || n_conv != ck->n_conv) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CKPT_SHAPE);
        return LZ_ERR_CKPT_SHAPE;
    }
    /* Read from s->ssm_slot's OWN slot, not always offset 0 (the ring -
       forward.h's s->ssm_slot comment). This checkpoint is a SEPARATE
       mechanism from the ring (LZPrefixCache's cross-turn reuse,
       unrelated to lz_spec_round's own per-round rollback, which does
       not use LZStateCkpt at all - see lz_spec_round's own module
       comment), but it shares the SAME LZRunState, and a prior turn's
       own speculative decoding can leave s->ssm_slot pointing anywhere
       in the ring by the time this runs - offset 0 would then be stale,
       unrelated data. ring_depth is 1 for a model with no MTP head
       (lz_state_alloc's own comment), so ssm_slot is always 0 there and
       this reduces to the single-slot behavior. */
    o_q8  = (size_t)s->ssm_slot * n_q8;
    o_s   = (size_t)s->ssm_slot * n_s;
    o_conv = (size_t)s->ssm_slot * n_conv;
    memcpy(ck->ssm_q8, s->ssm_state_q8 + o_q8, n_q8);
#if LZ_GDN_STATE_2PLANE
    memcpy(ck->ssm_q8_lo, s->ssm_state_q8_lo + o_q8, n_q8);
#endif /* LZ_GDN_STATE_2PLANE */
    memcpy(ck->ssm_s, s->ssm_state_s + o_s, n_s * sizeof(float));
#if LZ_CONV_FIXED
    /* Whichever history is live. Saving the float one while the fixed
       tier runs would restore a zero history and read as a subtle
       quality drift rather than as a broken checkpoint. */
    if (s->conv_fixed)
        memcpy(ck->conv_q, s->conv_state_q + o_conv, n_conv * sizeof(short));
    else
#endif /* LZ_CONV_FIXED */
    memcpy(ck->conv, s->conv_state + o_conv, n_conv * sizeof(float));
    ck->pos   = pos;
    ck->epoch = s->epoch;
    return LZ_ERR_OK;
}

int lz_ckpt_restore(const LZStateCkpt *ck, LZRunState *s, const LZModel *m,
                    int *out_pos, char *errbuf, int errlen) {
    size_t n_q8, n_s, n_conv;
    if (!ck || !s || !m || !ck->ssm_q8) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    if (ck->pos < 0) {          /* never saved into */
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CKPT_EMPTY);
        return LZ_ERR_CKPT_EMPTY;
    }
    /* The KV cache is not in the checkpoint, so it has to still hold the
       prefix this state was saved with. A reset (or a fresh alloc) means
       it does not. Refusing here is the whole point of the epoch field:
       the alternative is a correct recurrent state sitting on top of a
       zeroed KV, which produces confident nonsense and no diagnostic. */
    if (ck->epoch != s->epoch) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CKPT_STALE);
        return LZ_ERR_CKPT_STALE;
    }
    ckpt_sizes(&m->config, &n_q8, &n_s, &n_conv);
    if (n_q8 != ck->n_ssm_q8 || n_s != ck->n_ssm_s || n_conv != ck->n_conv ||
        ck->pos >= s->seq_len) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_CKPT_SHAPE);
        return LZ_ERR_CKPT_SHAPE;
    }
    /* Write to slot 0 and REPOINT s->ssm_slot there (the ring): a
       restore establishes a fresh baseline, and slot 0 is as good a
       place as any to put it - lz_state_reset uses the same convention
       (ssm_slot=0), and this is exactly what "restore" ought to mean:
       as if the state had been freshly built up to ck->pos and nothing
       else. Every OTHER ring slot becomes irrelevant garbage until some
       future speculative round overwrites it, same as after any reset.
       Without this, s->ssm_slot would keep pointing at whatever slot a
       PRIOR turn's own speculative decoding last left it on - stale
       data unrelated to what was just restored - and every forward call
       after this one would silently read the wrong slot. Covered by
       test_cli_ckpt_parity.py / test_resume_parity.py's cross-turn
       checks against a session that also uses --spec. */
    memcpy(s->ssm_state_q8, ck->ssm_q8, n_q8);
#if LZ_GDN_STATE_2PLANE
    memcpy(s->ssm_state_q8_lo, ck->ssm_q8_lo, n_q8);
#endif /* LZ_GDN_STATE_2PLANE */
    memcpy(s->ssm_state_s, ck->ssm_s, n_s * sizeof(float));
#if LZ_CONV_FIXED
    if (s->conv_fixed)
        memcpy(s->conv_state_q, ck->conv_q, n_conv * sizeof(short));
    else
#endif /* LZ_CONV_FIXED */
    memcpy(s->conv_state, ck->conv, n_conv * sizeof(float));
    s->ssm_slot = 0;
    if (out_pos) *out_pos = ck->pos;
    return LZ_ERR_OK;
}

/* -------------------------------------------------- full_attention block */

/* layer sentinel for the MTP block's own attention: it is not one of
   m->layers[], so it has no entry in s->cache_idx. Its reserved KV cache
   slot is the one past every real full_attention layer (index
   n_full_layers - see lz_state_alloc's "+1 layer slot" comment); the
   sentinel just tells forward_attn to use that slot directly instead of
   indexing cache_idx[layer]. */
#define LZ_MTP_CACHE_LAYER (-1)

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


/* --------------------------------------------------------- MTP draft step */

float *lz_mtp_draft_step(const LZModel *m, LZRunState *s, int next_token, int pos) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size, i;
    unsigned fpu;
    float t_us0;                              /* debug probe, see part 4 */

    if (!m || !m->mtp) return NULL;
    if (next_token < 0 || next_token >= c->vocab_size) return NULL;
    if (pos < 0 || pos >= s->seq_len) return NULL;

    t_us0 = lz_time_ms() * LZ_US_SCALE;
    fpu = lz_fpu_float_begin();

    /* fc input: concat(pre_fc_norm_embedding(next_token's embedding),
       pre_fc_norm_hidden(s->mtp_chain)) - embedding occupies the LOW
       half, hidden the HIGH half (model.h's LZMtp comment;
       torch.cat([inputs_embeds, hidden_states], dim=-1)). s->mtp_chain
       holds either the body's raw hidden (seeded by lz_forward_capture,
       first step of a round) or the PREVIOUS step's own post-norm
       output (chained steps) - read here before being overwritten
       below by this step's own result. */
    lz_t_row_f32(&m->embed_tokens, next_token, dim, s->mtp_emb_raw);
    lz_rmsnorm(s->mtp_concat, s->mtp_emb_raw,
              lz_t_f32(&m->mtp->pre_fc_norm_embedding, s->wscr), dim,
              c->rms_norm_eps);
    lz_rmsnorm(s->mtp_concat + dim, s->mtp_chain,
              lz_t_f32(&m->mtp->pre_fc_norm_hidden, s->wscr), dim,
              c->rms_norm_eps);
    lz_matmul_w(s->mtp_x, s->mtp_concat, &m->mtp->fc, 2 * dim, dim,
               s->xq, s->xqs);

    /* One ordinary decoder layer (model.h: "the block is an ordinary
       full_attention layer"), same pre-norm/residual shape forward_chunk
       uses for every body layer, applied to s->mtp_x instead of s->x and
       to the MTP's own reserved KV cache slot (LZ_MTP_CACHE_LAYER). */
    lz_rmsnorm(s->xb, s->mtp_x,
              lz_t_f32(&m->mtp->blk.input_layernorm, s->wscr), dim,
              c->rms_norm_eps);
    forward_attn(m, s, &m->mtp->blk, LZ_MTP_CACHE_LAYER, pos, 1);
    if (lz_debug_mtp_attn_scale != 1.0f) {
        /* Investigative probe only - see this file's own comment on
           lz_debug_mtp_attn_scale above. */
        for (i = 0; i < dim; i++) s->xb2[i] *= lz_debug_mtp_attn_scale;
    }
    for (i = 0; i < dim; i++) s->mtp_x[i] += s->xb2[i];

    lz_rmsnorm(s->xb, s->mtp_x,
              lz_t_f32(&m->mtp->blk.post_attention_layernorm, s->wscr), dim,
              c->rms_norm_eps);
    dense_ffn_step(m, s, &m->mtp->blk, LZ_MTP_CACHE_LAYER, 1, c->mtp_intermediate_size);
    for (i = 0; i < dim; i++) s->mtp_x[i] += s->xb2[i];

    /* Exit norm - this step's OWN post-norm output. Doubles as (a) the
       source for this step's draft logits below and (b) VERBATIM the
       next chained step's "hidden" input (s->mtp_chain's comment,
       forward.h): upstream returns exactly this value and feeds it
       straight back into the next call, no extra processing. */
    lz_rmsnorm(s->mtp_chain, s->mtp_x,
              lz_t_f32(&m->mtp->norm, s->wscr), dim, c->rms_norm_eps);

    /* lm_head is the shared embedding - same tie_word_embeddings path
       forward_chunk's own lm_head step uses. */
    lz_quantize_q8(s->mtp_chain, dim, lz_act_gs(&m->embed_tokens, dim),
                  s->xq, s->xqs);
    lz_matmul_xq(s->mtp_draft_logits, s->mtp_chain, s->xq, s->xqs,
                &m->embed_tokens, dim, c->vocab_size);

    lz_fpu_float_end(fpu);
    lz_debug_us_draft += (lz_i64)(lz_time_ms() * LZ_US_SCALE - t_us0);
    return s->mtp_draft_logits;
}

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
static float *forward_chunk(const LZModel *m, LZRunState *s,
                            const int *tokens, int nt, int pos0,
                            float *capture_hidden,
                            int all_logits, float *logits_out,
                            int skip_logits) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size;
    int l, i, tk;
    LZ_PROF_DECL(_th);
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

        /* SubLN (config.use_subn): the two pre-layer RMSNorms are
           deleted and each ternary projection normalizes its own input
           instead (forward_kda / dense_ffn_step). Ordinary models keep
           the original norms, bit for bit. The SubLN copy is elementwise
           so downstream readers of s->xb keep working unchanged. */
        if (c->use_subn) {
            for (i = 0; i < nt * dim; i++) s->xb[i] = s->x[i];
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb + (size_t)tk * dim, s->x + (size_t)tk * dim,
                           lz_t_f32(&L->input_layernorm, s->wscr), dim,
                           c->rms_norm_eps);
        }
        LZ_TAP("an", l, s->xb, dim);
        LZ_PROF_BEG(_tp);
        if (L->type == LZ_LT_FULL)      forward_attn(m, s, L, l, pos0, nt);
        else if (L->type == LZ_LT_KDA)  forward_kda(m, s, L, l, nt, all_logits, ring_base);
        else                             forward_ssm(m, s, L, l, nt, all_logits, ring_base);
        LZ_PROF_END(_tp, L->type == LZ_LT_FULL ? LZ_PROF_ATTN : LZ_PROF_LIN);
        LZ_TAP("blk", l, s->xb2, dim);
        for (i = 0; i < nt * dim; i++) s->x[i] += s->xb2[i];
        LZ_TAP("res", l, s->x, dim);

        if (c->use_subn) {
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
        for (i = 0; i < nt * dim; i++) s->x[i] += s->xb2[i];
        LZ_TAP("out", l, s->x, dim);
    }
    /* Advance the ring ONCE for this whole chunk, after every layer has
       used ring_base - ordinary decode/prefill (all_logits==0) never
       touches s->ssm_slot at all, so those paths stay bit-identical to
       a single-slot engine. See ring_base's own comment above for why
       this lives here and not inside forward_ssm/forward_kda. */
    if (all_logits)
        s->ssm_slot = (ring_base + nt) % s->ssm_ring_depth;

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
    if (capture_hidden) {
        for (tk = 0; tk < nt; tk++)
            lz_rmsnorm(capture_hidden + (size_t)tk * dim,
                       s->x + (size_t)tk * dim,
                       lz_t_f32(&m->final_norm, s->wscr), dim,
                       c->rms_norm_eps);
    }

    if (all_logits) {
        /* Speculative-decode verify path (lz_forward_verify): EVERY
           position's logits, not just the last - see this function's
           own comment above. Mutually exclusive with the block below:
           normalizing position nt-1 here AND there would run rmsnorm on
           an already-normalized vector a second time. */
        int gse = lz_act_gs(&m->embed_tokens, dim);
        int nse = scale_groups(dim, gse);
        for (tk = 0; tk < nt; tk++) {
            float *xt = s->x + (size_t)tk * dim;
            lz_rmsnorm(xt, xt, lz_t_f32(&m->final_norm, s->wscr), dim,
                      c->rms_norm_eps);
            lz_quantize_q8(xt, dim, gse, s->xq + (size_t)tk * dim,
                          s->xqs + (size_t)tk * nse);
        }
        lz_matmul_xq_nt(logits_out, s->x, s->xq, s->xqs, &m->embed_tokens,
                       dim, c->vocab_size, nt);
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

    /* lm_head only computes the LAST token. It is the single largest
       matmul (32768 vocab rows), and prefill's intermediate logits are
       wanted by nobody - computing the whole batch would pay the
       forward's most expensive step nt-1 times for nothing. At nt=1,
       nt-1 == 0, identical to the original path. */
    {
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
    lz_fpu_float_end(fpu);
    return s->logits;
}

float *lz_forward(const LZModel *m, LZRunState *s, int token, int pos) {
    return forward_chunk(m, s, &token, 1, pos, NULL, 0, NULL, 0);
}

float *lz_forward_capture(const LZModel *m, LZRunState *s, int token, int pos) {
    /* s->mtp_chain is unallocated (NULL) when m->mtp is NULL - guard
       rather than let forward_chunk memcpy into it, matching this
       function's own contract ("a no-op ... when m->mtp is NULL"). */
    float t_us0 = lz_time_ms() * LZ_US_SCALE;  /* debug probe, see part 4 */
    float *r = forward_chunk(m, s, &token, 1, pos,
                             m->mtp ? s->mtp_chain : NULL, 0, NULL, 0);
    lz_debug_us_capture += (lz_i64)(lz_time_ms() * LZ_US_SCALE - t_us0);
    lz_debug_n_capture++;
    return r;
}

float *lz_forward_batch(const LZModel *m, LZRunState *s,
                        const int *tokens, int n, int pos0) {
    float *lg = NULL;
    int cap, done = 0;

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
        lg = forward_chunk(m, s, tokens + done, k, pos0 + done, NULL, 0, NULL,
                           done + k < n);
        if (!lg) return NULL;
        done += k;
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
                           1, s->mtp_logits + (size_t)done * c->vocab_size, 0))
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
                           done + k < n);
        if (!lg) return NULL;
        done += k;
    }
    return lg;
}

/* Run the MTP block over n prompt positions purely to populate its own
   KV cache before the first speculative round (forward.h's s->mtp_pos
   comment). h_body_all[i]/next_tokens[i] are position pos0+i's body
   hidden and actual-next-token, exactly as lz_mtp_draft_step's own
   "hidden"/"next_token" inputs, just for every prefill position instead
   of one chained step.

   Only the ATTENTION half of the MTP block runs (fc -> input_layernorm
   -> forward_attn); the FFN half (dense_ffn_step) is skipped entirely.
   This is deliberate, not a shortcut: forward_attn's KV-cache write
   (this call's whole purpose) happens as a side effect of computing
   k/v for THIS position, independent of anything the FFN would add to
   the residual afterward - and the MTP block is exactly one layer, so
   there is no LATER position that ever reads this position's post-FFN
   mtp_x (unlike the body's multi-layer residual stream, where layer
   l+1 needs layer l's FFN output). The exit-norm/lm_head step is
   skipped for the same reason lz_spec_round never chains off a prefill
   position: s->mtp_chain is freshly reseeded by lz_forward_capture the
   moment a real round starts (generate.c), so nothing here needs to
   leave a valid chain register behind.

   fc's per-position concat + matmul stays a plain nt-loop (lz_matmul_w,
   one row at a time) - same shape forward_kda's f_b_proj loop uses for
   its own differently-shaped-than-hidden projection; the part that
   actually gets nt-batched is forward_attn, via its own nt parameter,
   exactly like every body layer's attention block. */
int lz_mtp_prefill(const LZModel *m, LZRunState *s, const float *h_body_all,
                   const int *next_tokens, int n, int pos0) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size;
    int cap, done = 0;
    unsigned fpu;

    if (!m || !m->mtp || !s || !h_body_all || !next_tokens || n < 1) return 1;

    cap = s->nt_cap;
    if (cap < 1) cap = 1;
    if (cap > LZ_BATCH_MAX) cap = LZ_BATCH_MAX;

    fpu = lz_fpu_float_begin();

    while (done < n) {
        int k = n - done, tk;
        if (k > cap) k = cap;

        for (tk = 0; tk < k; tk++) {
            const float *h_i = h_body_all + (size_t)(done + tk) * dim;
            float *concat_t = s->mtp_concat + (size_t)tk * 2 * dim;
            float *emb_t = s->mtp_emb_raw + (size_t)tk * dim;
            float *x_t = s->mtp_x + (size_t)tk * dim;

            lz_t_row_f32(&m->embed_tokens, next_tokens[done + tk], dim, emb_t);
            lz_rmsnorm(concat_t, emb_t,
                      lz_t_f32(&m->mtp->pre_fc_norm_embedding, s->wscr), dim,
                      c->rms_norm_eps);
            lz_rmsnorm(concat_t + dim, h_i,
                      lz_t_f32(&m->mtp->pre_fc_norm_hidden, s->wscr), dim,
                      c->rms_norm_eps);
            lz_matmul_w(x_t, concat_t, &m->mtp->fc, 2 * dim, dim,
                       s->xq, s->xqs);
        }

        for (tk = 0; tk < k; tk++)
            lz_rmsnorm(s->xb + (size_t)tk * dim, s->mtp_x + (size_t)tk * dim,
                      lz_t_f32(&m->mtp->blk.input_layernorm, s->wscr), dim,
                      c->rms_norm_eps);
        /* The only thing this call needs: forward_attn writes this
           chunk's k/v rows into the MTP's reserved KV cache slot as a
           side effect (LZ_MTP_CACHE_LAYER) - s->xb2, its attention
           OUTPUT, is never read below. See this function's own comment
           above for why the FFN half is skipped. */
        forward_attn(m, s, &m->mtp->blk, LZ_MTP_CACHE_LAYER, pos0 + done, k);

        done += k;
    }

    lz_fpu_float_end(fpu);
    return 0;
}

/* Catch-up decode - see forward.h's own comment on why this is a
   separate entry point rather than callers just calling lz_mtp_prefill
   directly. Identical body to lz_mtp_prefill plus timing/counting;
   deliberately NOT sharing a common static helper with it - the two
   call sites' contracts read differently (one is "populate empty KV
   before the first round", the other is "replace speculative rows with
   verified ones after every round") even though the mechanics
   coincide, and forcing them through one name would blur exactly the
   distinction lz_debug_n_catchup exists to preserve. */
int lz_mtp_catchup(const LZModel *m, LZRunState *s, const float *h_body_all,
                   const int *next_tokens, int n, int pos0) {
    float t_us0 = lz_time_ms() * LZ_US_SCALE;
    int r = lz_mtp_prefill(m, s, h_body_all, next_tokens, n, pos0);
    lz_debug_us_catchup += (lz_i64)(lz_time_ms() * LZ_US_SCALE - t_us0);
    lz_debug_n_catchup++;
    return r;
}
