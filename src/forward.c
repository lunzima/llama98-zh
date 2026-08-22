#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "compat.h"    /* lz_time_ms, for the debug probes below only */
#include "err.h"
#include "ops.h"
#include "ops_quant.h"   /* q8_round/q8_amax/pow2f/LZ_Q8_*, lz_rsqrt_float_body */
#include "fwht.h"        /* lz_fwht_i32, the SubLN Hadamard kernel */

/* The reference implementation hardcodes l2norm's eps to 1e-6, which
   is semantically different from config's rms_norm_eps - they only
   coincide numerically. Keep them separate so a future config change
   cannot silently drag this one along. */
#define LZ_L2NORM_EPS 1e-6f

/* Per-layer intermediate tap hook for differential testing. No-op by
   default, zero cost in production; a dump tool defines it before
   including this file to write intermediates to disk, bisecting which
   layer/which quantity first diverges.

   Two consumers, and they answer different questions:
     a dump tool   vs transformers' Qwen3_5 - "is this RIGHT"
     gcc vs Watcom builds                  - "do the two builds AGREE"
   Deleting a tap silently shrinks the first one; a REQUIRED_TAGS list
   makes that fail instead. */
#ifndef LZ_TAP
#define LZ_TAP(tag, li, ptr, n) ((void)0)
#endif /* LZ_TAP */

/* Debug-only investigative probe for the team-lead's dilution-vs-position
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

/* Debug-only investigative probe, part 2 (team-lead's "decoupled"
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

/* Debug-only investigative probe, part 3 (team-lead's iron-law-three
   reminder: a wider MTP attention window is zero extra
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
long long lz_debug_mtp_attn_rows = 0;

/* Debug-only investigative probe, part 4. Microseconds spent inside
   each of a speculative round's three pieces.

   WHY A TIMER AND NOT MORE ARITHMETIC: the round's cost was derived
   six different ways from byte counts and CLI deltas, and six
   hypotheses about where it goes were each refuted by the next
   measurement (the draft head, lz_matmul_w's "slow path", unquantized
   mtp.* tensors, a redundant capture forward, an all_logits per-token
   lm_head, the per-round checkpoint). What IS measured: on a Ryzen
   5800X with Qwen3.5-0.8B, a round costs 3.78x one plain forward, of
   which the draft step is 0.32x - exactly what its bytes predict - and
   VERIFYING TWO TOKENS costs 2.46x, where the same accounting says
   ~1.5x. Prefill batches the same trunk at 0.54x per token, so the
   amortization exists and this path is not getting it. That gap is the
   only thing left worth chasing, and it will not be found by dividing
   more totals.

   Integer microseconds, not float seconds: %f is barred inside the
   PC=24 region, and these are read by cli_main.c which
   may print them; a long long crosses that boundary safely. Zero cost
   when nothing reads them, and body-only forwards never touch the two
   MTP ones. */
long long lz_debug_us_verify = 0;
long long lz_debug_us_draft = 0;
long long lz_debug_us_capture = 0;

/* Call-count companion to lz_debug_us_capture, for the "capture skipped,
   not just cheap" gate: a removed-but-still-called
   lz_forward_capture and a genuinely removed one look identical on any
   pure timing/output comparison if the removed call happened to be fast
   that run - this counter is what actually distinguishes them, same
   role lz_debug_mtp_attn_rows plays for the attention-window change. */
long long lz_debug_n_capture = 0;

/* Timer + call-count pair for lz_mtp_catchup, same shape as the capture
   pair above and same reason: team-lead's explicit
   requirement is a POSITIVE control proving catch-up decode actually
   ran, not just a plausible-looking diff - "removed" and "still there
   but never called" are indistinguishable without a counter (the
   pos_only dead-code lesson, see lz_debug_mtp_attn_rows's own history).
   lz_debug_us_catchup is expected to be small - lz_mtp_catchup skips
   both the FFN half and lm_head (lz_mtp_prefill's own comment), so it
   is cheaper per token than even a draft step, let alone a full
   forward. */
long long lz_debug_us_catchup = 0;
long long lz_debug_n_catchup = 0;

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
long long lz_debug_n_kv_rot = 0;

/* Positive control for --batch, same reason as the one above. Batching
   is defined to be bit-identical across widths, so an output comparison
   can never tell "the knob works and is exact" from "the knob was never
   read" - and the latter is what a compile-time-only LZ_BATCH_MAX
   looked like from the CLI. Counts forward_chunk calls: prefilling n
   tokens at width T must take ceil(n/T) of them, which is a number the
   width cannot fake. */
long long lz_debug_n_chunks = 0;

/* Positive control for skip_logits. Prefilling n tokens at width T must
   run lm_head ONCE, not ceil(n/T) times - and since the skipped chunks'
   logits were never read, output comparison cannot see the difference
   any more than it can see the batch width. */
long long lz_debug_n_lmhead = 0;

/* Positive control for the sink+window row skip. Counted in the SKIP
   branch, not in the scoring loop: on the default path (no window) the
   branch is never taken, so this costs a predictable compare and no
   increment, and a broken skip reports 0 rather than reporting the
   count it was supposed to achieve. A counter derived from pos and
   attn_win would have been the fake kind - right by construction even
   if the loops still walked every row. */
long long lz_debug_attn_skip = 0;

/* Positive control for the V projection's int16 exit (LZ_VPROJ_I16).
   Counted at the CONSUMER - inside the branch that hands vtmp_i16 to
   lz_quantize_q8_i16 - not where the projection wrote it: an exit that
   was taken and then read from the wrong buffer would still increment a
   producer-side counter, and the int16 path is bit-comparable to the
   float one, so the logits cannot tell the two apart on their own.
   Elements, not calls, so it is the number the conversion arithmetic
   uses directly. */
long long lz_debug_vproj_i16 = 0;

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

   WE DO NOT DO THE PAPER'S POSITION SHIFT, AND THAT IS LOAD-BEARING.
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
double lz_prof_us[LZ_PROF_N];

/* 1000.0 (ms -> us) is exact in any float width, so this cannot change
   what these debug counters measure - but `static const double`, not an
   inline literal, keeps every site below on the same footing as
   ops.c's LZ_EXP_LOG2E32 (that comment has the mechanism) rather than
   leaving an unexplained exception in tests/test_excess_precision.py. */
static const double LZ_US_SCALE = 1000.0;

/* The start time is a LOCAL, not a file-static. A file-static would be
   shared across nested phases: the recurrence timer - which nests inside
   the linear one - would overwrite the outer start, so `linear` would
   report only the tail after the last inner call and the total would
   come out 25%% short. Phases that nest are reported INCLUSIVE, with
   the inner one also listed on its own. */
#define LZ_PROF_BEG(v) double v = lz_prof_enable ? lz_time_ms() : 0.0
#define LZ_PROF_END(v, slot) do { if (lz_prof_enable) \
    lz_prof_us[slot] += (lz_time_ms() - (v)) * LZ_US_SCALE; } while (0)

static int kv_slot(const LZRunState *s, int t) {
    if (!s->kv_ring) return t;
    if (t < s->attn_sink) return t;
    return s->attn_sink + ((t - s->attn_sink) % s->kv_ring);
}

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
static void rot_head(float *v, int hd, int n) {
    int o;
    for (o = 0; o < hd; o += n) lz_fwht(v + o, n);
    lz_debug_n_kv_rot++;
}

/* ------------------------------------------------------------ state allocation */

static void *xcalloc(size_t n, size_t sz, int *ok, long long *acc) {
    void *p;
    if (n == 0) return NULL;
    p = calloc(n, sz);
    if (!p) { *ok = 0; return NULL; }
    *acc += (long long)n * (long long)sz;
    return p;
}

/* History exponent for the fixed conv tier.
 *
 * MEASURED, not chosen (.prof/wconvrange.c, recover_r20, 64 tokens,
 * 1.18M samples of the real conv inputs): max |x| = 22.3, the top 0.1%
 * sits at 2^3, and the highest non-empty bucket is 2^4 - one bucket of
 * outlier headroom, so this is not a long tail that a constant exponent
 * has to waste range on. The first and second halves of the sequence
 * agree on the 0.1% bucket, which is the property that matters here:
 * the history carries values from earlier steps at the same scale, so a
 * drifting range would need the scale to travel with the state the way
 * ssm_state_s does for the recurrence.
 *
 * 22.3 * 2^9 = 11423, inside the 16384 bound with a bucket to spare.
 * A value past 32 clips; the probe saw none, and the clamp is silent, so
 * re-run it on any model whose conv inputs might be scaled differently. */
#ifndef LZ_CONV_ES
#define LZ_CONV_ES 9
#endif /* LZ_CONV_ES */

/* The per-channel exponent table's call-site accessor. A macro rather
   than a plain `s->conv_sig_e + off` because the =0 build never
   allocates the array, and `NULL + off` is undefined even where nothing
   dereferences it - the kernel's own parameter is unread in that arm. */
#if LZ_CONV_SIG_I
#define LZ_CONV_SIG_E(s, off) ((s)->conv_sig_e + (off))
#else
#define LZ_CONV_SIG_E(s, off) ((const signed char *)0)
#endif /* LZ_CONV_SIG_I */

/* ---- q16_0's four int16 exits (int-pipeline milestone 5) -------------
 *
 * Every q16_0 role on this checkpoint - kda_b_proj, moe_gate_w,
 * kda_f_a_proj, kda_f_b_proj - now leaves lz_matmul_xq_nt_i16 as an
 * int16 at a fixed exponent instead of a float the consumer immediately
 * takes apart again. Each pair below is one chain: _I16 turns it off for
 * a control build, _ES is the exponent, i.e. the number of FRACTIONAL
 * bits (value == q * 2^-ES), the convention epi_align_i16's target_e
 * uses.
 *
 * THE EXPONENTS ARE MEASURED, not chosen, exactly like LZ_CONV_ES above
 * (.prof/m5range.c on kmr20, both gate corpora, 603/661 tokens). Each is
 * the LARGEST that clamps nothing, since the clamp is silent:
 *
 * Measured max |q| at the shipping ES, from an instrumented build's
 * pre-clamp counter, 0 clamps everywhere on both corpora. It aggregates
 * by weight SHAPE, so bvec and the router share a row - they are both
 * 512x16 gs=512 - and the max in it is the router's (its |x| runs to
 * 10.2 against bvec's 6.7, .prof/m5range.c):
 *
 *   shape             ES     bound     max |q| zh / en    headroom
 *   512x16  (bvec,
 *            moe gate) 11    24,576     21,013 / 20,904     1.17x
 *   512x64  gate_lat   10    32,767     21,288 / 17,963     1.54x
 *   64x1024 kda_gate    9    32,767     15,624 / 14,295     2.10x
 *
 * bvec and the router are the two whose consumer is lz_sigmoid, and
 * their bound is NOT int16's edge but the sigmoid table's own
 * (lz_sig_q15_domain, |x| = 12): past it sigmoid_q15 already returns a
 * constant, so clamping there adds no behaviour rather than adding a
 * silent one. Their headroom figure is therefore informational; the
 * other two clamp at a plain int16 edge and theirs is not.
 *
 * kda_gate is at 9 rather than the 10 that would have been the largest
 * non-clamping value, and that is a measurement: ES = 10 leaves 1.05x
 * of headroom against a SILENT clamp, and paired NLL cannot tell the two
 * apart (zh t +0.417 at 10 vs -0.131 at 9, en +1.396 vs +0.763, both
 * arms against the same float control, n = 602/660). Given a tie on
 * quality, the exponent with twice the headroom is the one to ship.
 * Re-run .prof/m5range.c on any checkpoint whose gate projection might
 * be scaled differently. */
#ifndef LZ_BVEC_I16
#define LZ_BVEC_I16 1
#endif /* LZ_BVEC_I16 */
#ifndef LZ_BVEC_ES
#define LZ_BVEC_ES 11
#endif /* LZ_BVEC_ES */

#ifndef LZ_MOEGATE_I16
#define LZ_MOEGATE_I16 1
#endif /* LZ_MOEGATE_I16 */
#ifndef LZ_MOEGATE_ES
#define LZ_MOEGATE_ES 11
#endif /* LZ_MOEGATE_ES */

#ifndef LZ_KLAT_I16
#define LZ_KLAT_I16 1
#endif /* LZ_KLAT_I16 */
#ifndef LZ_KLAT_ES
#define LZ_KLAT_ES 10
#endif /* LZ_KLAT_ES */

#ifndef LZ_KGATE_I16
#define LZ_KGATE_I16 1
#endif /* LZ_KGATE_I16 */
#ifndef LZ_KGATE_ES
#define LZ_KGATE_ES 9
#endif /* LZ_KGATE_ES */

/* ---- the MoE latent's int16 exit (int-pipeline milestone 6) ----------
 *
 * moe_down_proj is the tree's one Q8_0 producer whose consumer is an
 * activation quantize, so it is the first caller of
 * epi_fixed_align_i16's Q8_0 half - delivered in batch 1, verified
 * standalone, and until now reached only by q16_0. Same _I16 / _ES pair
 * shape as the four chains above and the same conventions.
 *
 * MEASURED, both gate corpora on kmr20, instrumented shipping build
 * (1.23M / 1.35M latent elements): max |x| 9.270 / 10.126, so ES = 11
 * is the largest that clamps nothing - ES = 12 clamps 75 / 67.
 *
 * It ships at 10, one lower, for the reason LZ_KGATE_ES is: 11 leaves
 * 1.58x of headroom against a SILENT clamp and paired NLL cannot tell
 * the two apart - zh t -1.190 at 11 vs -0.804 at 10, en +1.552 vs
 * +1.077, both arms against the same pre-change control, n = 602/660,
 * with argmax and top-5 agreement swapping which one leads. Given a
 * tie on quality, the exponent with twice the headroom (3.16x) ships.
 *
 * Unlike the four q16_0 chains, this one also has a per-GROUP precision
 * question, because the consumer is a per-32 activation quantize and a
 * group whose amax lands near the bottom of int16 would reach int8 with
 * fewer than 127 levels available. Measured over the same runs: the
 * SMALLEST 32-element group amax is 0.385 / 0.549, i.e. 394 / 562 at
 * the shipping ES, 3.1x above the 127 the int8 output needs, and no
 * group on either corpus falls under 128. So the shared exponent costs
 * the consumer nothing here; re-measure it on a checkpoint whose latent
 * might be scaled differently. */
#ifndef LZ_MLAT_I16
#define LZ_MLAT_I16 1
#endif /* LZ_MLAT_I16 */

/* The expert loop's hoisted activation quantize: reuse the previous
   expert's result when the group size has not changed. A separate axis
   from LZ_MLAT_I16 because it is a separate saving that happens to have
   arrived in the same commit - it is worth more than the int16 exit and
   is value-neutral, while the exit alone is a small regression. Without
   its own switch neither half's contribution can be rebuilt, which is
   how "measured -4,224/token" becomes a number nobody can reproduce. */
#ifndef LZ_MLAT_REUSE
#define LZ_MLAT_REUSE 1
#endif /* LZ_MLAT_REUSE */
#ifndef LZ_MLAT_ES
#define LZ_MLAT_ES 10
#endif /* LZ_MLAT_ES */

/* ---- the V projection's int16 exit (int-pipeline milestone 7) --------
 *
 * v_proj is the only matmul left in this file whose output goes into an
 * activation quantize with no float operation in between: the KV cache
 * write IS its consumer, so lz_quantize_q8_i16 serves it the way it
 * already serves the MoE latent. Same _I16 / _ES pair, same conventions.
 * Every other remaining consumer either needs a float by construction (a
 * residual add, an RMSNorm, RoPE, the logits' softmax, the routed
 * mixture's float weight) or needs a fixed-point kernel this tree does
 * not have (the gated norm's gate; the SwiGLU pair got one in milestone
 * 8, below).
 *
 * The exit is refused whenever anything else reads the float row: the
 * Hadamard rotation (kv_rot_v), the f32 reference plane (--kv f32) and
 * the 4-bit plane (--kv q4) all take vtt apart element by element. That
 * is the same set --attn-int already refuses on, for the same reason.
 *
 * MEASURED, both gate corpora on kmr20 AND kunmoe-v2-t2-probe (v_proj is
 * q6_1 on the first and t2 on the second, so both epilogue families are
 * covered), instrumented build over 308,736 / 338,432 v elements:
 * max |v| 6.42 / 6.79 (kmr20) and 6.99 / 6.80 (t2-probe), so ES = 12 is
 * the largest that clamps nothing against the int16 edge.
 *
 * It ships at 11, one lower, for the reason LZ_MLAT_ES does: the clamp
 * is silent, 11 leaves 2.29x of headroom against 12's 1.14x, and paired
 * NLL cannot tell the two apart. The per-GROUP question the consumer
 * raises - a 32-element group whose amax lands near the bottom of int16
 * reaches int8 with fewer than 127 levels - is measured the same way:
 * the SMALLEST group amax over those runs is 0.4157, i.e. 851 at the
 * shipping ES, 6.7x above the 127 the int8 cache needs, and no group on
 * either corpus or either model falls under 128. */
#ifndef LZ_VPROJ_I16
#define LZ_VPROJ_I16 1
#endif /* LZ_VPROJ_I16 */
#ifndef LZ_VPROJ_ES
#define LZ_VPROJ_ES 11
#endif /* LZ_VPROJ_ES */

/* ---- the SwiGLU pair's int16 exits (int-pipeline milestone 8) --------
 *
 * silu(p)*g was the largest remaining block of conversions in the
 * engine: the two producing matmuls each materialize a float
 * (epi_q41_fixed's `(float)comb * pow2f(target)`, one cvt per element),
 * lz_sigmoid converts once more to leave sigmoid_q15's table and
 * sigmoid_q15 itself measures 2.975 converts per call deriving the table
 * coordinate from a float, and the product then goes back to int through
 * lz_quantize_q8's SIMD round. lz_swiglu_q15_i16 (src/ops.c) removes all
 * four: with p and g arriving as int16, sigmoid_q15_i reads p's bits and
 * the whole chain from the two matmuls to lz_quantize_q8_i16 stays
 * integer.
 *
 * ONE switch for the two MoE sites, not one per site: they are the same
 * mechanism on the same models, reached by the same helper, and neither
 * would ever be enabled without the other - what the switch selects is
 * the saving, and the saving is one. The dense FFN's site is NOT here;
 * see forward_ffn.
 *
 * THREE exponents, all measured (.prof/m8range.c, both gate corpora on
 * kmr20 AND kunmoe-v2-t2-probe - w1/w3 are q4_1+q6_1 on the first and
 * q8_0 on the second, so both epilogue families are covered). Largest ES
 * that clamps NOTHING, per run, worst case across all four:
 *
 * PER SITE, because the two sites do not agree and one constant per role
 * has to take the smaller of them:
 *
 *   role              kmr20 zh/en   t2probe zh/en   worst
 *   routed p  (w1)     12 / 12        12 / 11        11
 *   shared p  (gate)   12 / 11        12 / 12        11
 *   routed g  (w3)     14 / 13        13 / 13        13
 *   shared g  (up)     12 / 12        13 / 13        12
 *   routed out         13 / 12        11 / 11        11
 *   shared out         11 / 11        12 / 12        11
 *
 * so the shipping p 11 / g 12 / out 11 is each role's worst case exactly,
 * with none of the three dropped an extra octave the way LZ_KGATE_ES and
 * LZ_VPROJ_ES were. That is a deliberate difference: a step down costs a
 * bit on every element of a chain already three roundings deep, and what
 * replaces the extra octave here is a live probe instead of an argument.
 * lz_debug_swiglu_pmax/_gmax/_omax carry the post-clamp maxima and the
 * CLI prints them - a maximum strictly below 32767 is the proof that
 * nothing clamped, since the clamp inside epi_align_i16 is silent.
 * Measured over both corpora on both models: 17,377 / 22,582 / 26,691,
 * i.e. 1.89x / 1.45x / 1.23x of margin. Re-run .prof/m8range.c on any
 * checkpoint whose FFN might be scaled differently, and read those three
 * on any run that matters.
 *
 * The sigmoid table's own edge is a non-event here and that is measured
 * too: |p| never reaches 12 on either model or either corpus (0 of
 * 8,644,608 routed and 0 of 2,469,888 shared elements), so sigmoid_q15_i
 * runs its interpolation on every element rather than its clamp. Its
 * domain cap (e <= 27, past which it reconstructs a float) is not
 * reachable either: ep is a compile-time 11. */
#ifndef LZ_SWIGLU_I16
#define LZ_SWIGLU_I16 1
#endif /* LZ_SWIGLU_I16 */
#ifndef LZ_SWIGLU_ES_P
#define LZ_SWIGLU_ES_P 11
#endif /* LZ_SWIGLU_ES_P */
#ifndef LZ_SWIGLU_ES_G
#define LZ_SWIGLU_ES_G 12
#endif /* LZ_SWIGLU_ES_G */
#ifndef LZ_SWIGLU_ES_O
#define LZ_SWIGLU_ES_O 11
#endif /* LZ_SWIGLU_ES_O */

#if LZ_CONV_FIXED
/* Quantize the conv taps once, laying them out per channel in the same
   order forward_kda/forward_ssm index the history.
 *
 * THE OFFSETS ARE DUPLICATED from those two functions, which is a drift
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
                float oscale = (float)ldexp(1.0, -((int)e + LZ_CONV_ES));
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
    s->conv_in_scale = (float)ldexp(1.0, LZ_CONV_ES);
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
    float sc = (float)ldexp(1.0, LZ_KGATE_ES);
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
       selected: lz_attn_select can run after this point, and 6 bytes per
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
    s->attn_ss  = (float *)xcalloc((size_t)nt * (c->attn_q_dim / 32), sizeof(float), &ok, &s->bytes_alloc);
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
            int w = seq_len / 2;
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

    /* SSM/conv state: one (kd×vd) block per (layer, head), grouped
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
     * THE HISTORY EXPONENT was the one parameter that kept this tier
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
        s->wscr = (float *)xcalloc((size_t)qcap, sizeof(float), &ok, &s->bytes_alloc);
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
    /* RoPE precomputed table: pos × (rotary_dim/2) {cos, sin} pairs.
       Shared by all layers/heads under the same theta; the half-precision
       frequency formula matches the reference (denominator is
       rotary_dim, not head_dim). */
    s->rope_cs = (float *)xcalloc((size_t)seq_len * c->rotary_dim,
                                  sizeof(float), &ok, &s->bytes_alloc);
    if (ok) {
        int half = c->rotary_dim / 2;
        for (l = 0; l < seq_len; l++) {
            int i;
            for (i = 0; i < half; i++) {
                double freq = pow((double)c->rope_theta,
                                  -2.0 * (double)i / (double)c->rotary_dim);
                double ang = (double)l * freq;
                s->rope_cs[((size_t)l * half + i) * 2]     = (float)cos(ang);
                s->rope_cs[((size_t)l * half + i) * 2 + 1] = (float)sin(ang);
            }
        }
    }
    if (!ok) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
        lz_state_free(s);
        return 1;
    }

    s->attn_scale = 1.0f / (float)sqrt((double)c->head_dim);
    s->ssm_scale  = 1.0f / (float)sqrt((double)c->lin_k_head_dim);
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
        free(s->kda_dtb_i16);  s->kda_dtb_i16 = NULL;
        free(s->kda_gate_i16); s->kda_gate_i16 = NULL;
    }
#endif /* LZ_KGATE_I16 */

    return 0;
}

void lz_state_free(LZRunState *s) {
    if (!s) return;
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->logits);
    free(s->qg); free(s->qh); free(s->att); free(s->attn_out);
    free(s->attn_acc); free(s->attn_ss);
#if (LZ_ATTN_FIXED & 2)
    free(s->wsum_cbuf); free(s->wsum_cq);
#endif /* LZ_ATTN_FIXED & 2 */
    free(s->ktmp); free(s->vtmp);
    free(s->kq8); free(s->vq8); free(s->ksq); free(s->vsq);
    free(s->kf32); free(s->vf32);
    free(s->k4); free(s->v4); free(s->ks4); free(s->vs4);
#if LZ_KV_2PLANE
    free(s->kq8_lo); free(s->vq8_lo); free(s->att_lo); free(s->wsum_lo);
#endif /* LZ_KV_2PLANE */
    free(s->qkv); free(s->qkv_c); free(s->zbuf); free(s->avec); free(s->bvec);
    free(s->ssm_out); free(s->ssm_sig); free(s->qn); free(s->kn);
    free(s->kda_q); free(s->kda_k); free(s->kda_v);
    free(s->kda_qc); free(s->kda_kc); free(s->kda_vc);
    free(s->kda_gate_lat); free(s->kda_gate);
    free(s->moe_router_logits); free(s->moe_sel_idx); free(s->moe_sel_w);
    free(s->moe_lat_x); free(s->moe_lat_y);
    free(s->moe_h1); free(s->moe_h3); free(s->moe_h2); free(s->moe_shared_out);
    free(s->mtp_x); free(s->mtp_concat); free(s->mtp_emb_raw);
    free(s->mtp_chain); free(s->mtp_draft_logits); free(s->mtp_logits);
    free(s->mtp_verify_hidden);
    free(s->mtp_draft_q);
    free(s->mtp_target_p);
    free(s->ssm_state_q8); free(s->ssm_state_s);
#if LZ_GDN_STATE_2PLANE
    free(s->ssm_state_q8_lo);
#endif /* LZ_GDN_STATE_2PLANE */
    free(s->conv_state);
    free(s->conv_state_q); free(s->conv_mw);
    free(s->kda_q_i16); free(s->kda_k_i16); free(s->kda_v_i16);
    free(s->bvec_i16); free(s->moe_logits_i16);
    free(s->kda_lat_i16); free(s->kda_lat_i32);
    free(s->kda_gate_i16); free(s->kda_dtb_i16);
    free(s->moe_lat_i16); free(s->moe_lat_q); free(s->moe_lat_qs);
    free(s->vtmp_i16);
    free(s->moe_h1_i16); free(s->moe_h3_i16);
    free(s->conv_sig_k1); free(s->conv_sig_oscale2); free(s->conv_sig_e);
    free(s->xq); free(s->xqs); free(s->wscr); free(s->fwht_scratch);
    free(s->subn_norm_int);
    free(s->rope_cs);
    free(s->cache_idx);
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

/* Defined with the KDA projections it was written for, and declared here
   because forward_attn's v_proj is a third caller of the same shape.
   Moving the definition up instead would separate it from kda_conv_proj,
   whose header carries the flag-discipline argument both of them obey. */
static void kda_i16_proj(LZRunState *s, int may_int, int *wrote_int,
                         float *of, short *oi,
                         const float *x, const LZTensor *w,
                         int in_dim, int out_dim, int nt,
                         int target_e, int bound);

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
static void forward_attn(const LZModel *m, LZRunState *s,
                         const LZLayer *L, int layer, int pos0, int nt) {
    const LZModelConfig *c = &m->config;
    int hd = c->head_dim, nh = c->n_heads, nkv = c->n_kv_heads;
    int kv_mul = nh / nkv;
    int ci = (layer >= 0) ? s->cache_idx[layer] : c->n_full_layers;
    size_t loff = (size_t)ci * s->kv_slots * c->attn_kv_dim;
    size_t soff = loff / 32;
    int8_t *kc = s->kq8 ? s->kq8 + loff : NULL;
    int8_t *vc = s->vq8 ? s->vq8 + loff : NULL;
    /* Reference-arm slices; NULL unless --kv f32. Same (layer, pos,
       kv_dim) layout as kq8/vq8 so `loff` indexes both. */
    float *kf = s->kf32 ? s->kf32 + loff : NULL;
    float *vf = s->vf32 ? s->vf32 + loff : NULL;
    /* 4-bit slices: half the bytes, plus one scale per (pos, kv head). */
    unsigned char *k4 = s->k4 ? s->k4 + loff / 2 : NULL;
    unsigned char *v4 = s->v4 ? s->v4 + loff / 2 : NULL;
    float *ks4 = s->ks4 ? s->ks4 + (size_t)ci * s->kv_slots * c->n_kv_heads : NULL;
    float *vs4 = s->vs4 ? s->vs4 + (size_t)ci * s->kv_slots * c->n_kv_heads : NULL;
    float *ks = s->ksq ? s->ksq + soff : NULL;
    float *vs = s->vsq ? s->vsq + soff : NULL;
#if LZ_KV_2PLANE
    int8_t *kc_lo = s->kq8_lo ? s->kq8_lo + loff : NULL;
    int8_t *vc_lo = s->vq8_lo ? s->vq8_lo + loff : NULL;
#endif /* LZ_KV_2PLANE */
    /* (H q) . (H k) = kv_rot_k * (q . k) because lz_fwht is the
       unnormalized Hadamard (see its comment in ops.h). Fold that factor
       out here rather than normalizing the transform: kv_rot_k is a power
       of two, so this division is exact and identical on both compilers,
       whereas dividing by a float constant makes the two compilers
       disagree - one strength-reduces it to a reciprocal multiply. */
    float scale = s->attn_scale;
    int hdim = c->hidden_size, kvd = c->attn_kv_dim, qd = c->attn_q_dim;
    int qgd = c->attn_qgate_dim;
    int gsa = lz_act_gs(&L->q_proj, hdim);
    int nsa = (gsa > 0) ? hdim / gsa : 0;
    int gso = lz_act_gs(&L->o_proj, qd);
    int nso = (gso > 0) ? qd / gso : 0;
    /* Per-token flag for the fixed-tier attention int path (--attn-int).
       The quantize step after the tk loop must know which tokens took the
       int wsum; only the MTP window override can split a batch. */
    int attn_int[LZ_BATCH_MAX];
    /* int-pipeline milestone 7. v_may is the capability, v_int is "the
       int buffer holds it" and is raised only by kda_i16_proj - two
       objects, for the reason that function's header gives.

       The consumer conditions are part of the capability, not of the
       matmul: vc is the plane lz_quantize_q8_i16 would write, and
       kv_rot_v / vf / v4 each read the float row element by element, so
       any of them means the exit would have to rebuild what it declined
       to build. Same set --attn-int refuses on. */
    int v_int = 0, v_may = 0;
    int h, d, t, tk;

    if (s->kv_rot_k) scale /= (float)s->kv_rot_k;
#if LZ_VPROJ_I16
    v_may = s->vtmp_i16 && vc && vs && !vf && !v4 && s->kv_rot_v == 0 &&
            !LZ_KV_2PLANE && kvd >= 32 && (kvd % 32) == 0 &&
            lz_matmul_xq_i16_ok(&L->v_proj, hdim, nt);
#endif /* LZ_VPROJ_I16 */

    /* Quantize xb once; the q/k/v projections share the same xq/xqs
       (avoid rescanning activations).

       Under mixed precision q/k/v may have different formats and
       different weight gs, but sharing one quantization still holds -
       lz_act_gs only looks at whether in_dim is a multiple of 32, and
       all three share the same in_dim, so the answer is identical.
       Asking q_proj once suffices. */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsa,
                       s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
    lz_matmul_xq_nt(s->qg, s->xb, s->xq, s->xqs, &L->q_proj, hdim, qgd, nt);
    lz_matmul_xq_nt(s->ktmp, s->xb, s->xq, s->xqs, &L->k_proj, hdim, kvd, nt);
    kda_i16_proj(s, v_may, &v_int, s->vtmp, s->vtmp_i16,
                 s->xb, &L->v_proj, hdim, kvd, nt, LZ_VPROJ_ES, 32767);

    /* QK-Norm + RoPE + write KV cache. **All cache rows of the batch
       must be written before scoring**: token t in the batch must see
       rows 0..t; merging the two loops would read rows not yet written. */
    for (tk = 0; tk < nt; tk++) {
        float *qgt = s->qg + (size_t)tk * qgd;
        float *qht = s->qh + (size_t)tk * qd;
        float *ktt = s->ktmp + (size_t)tk * kvd;
        float *vtt = s->vtmp + (size_t)tk * kvd;
        int pos = pos0 + tk;
        /* q_proj output is laid out PER HEAD as [q(hd), gate(hd)], not
           [all q][all gate]. The reference does view(..., n_heads, hd*2)
           then chunks along the last dim; treating it as two contiguous
           halves feeds every head misaligned data. */
        for (h = 0; h < nh; h++)
            lz_rmsnorm(qht + (size_t)h * hd, qgt + (size_t)h * 2 * hd,
                       lz_t_f32(&L->q_norm, s->wscr), hd, c->rms_norm_eps);
        for (h = 0; h < nkv; h++)
            lz_rmsnorm(ktt + (size_t)h * hd, ktt + (size_t)h * hd,
                       lz_t_f32(&L->k_norm, s->wscr), hd, c->rms_norm_eps);
        /* QK-Norm first, then RoPE, only then quantize into the cache -
           the cache holds the rotated k. head_dim is a multiple of 32,
           so quantization groups never cross heads. */
        lz_rope(qht, nh, hd, c->rotary_dim, pos, s->rope_cs);
        lz_rope(ktt, nkv, hd, c->rotary_dim, pos, s->rope_cs);
        /* Hadamard AFTER RoPE, on q and k alike - the cache holds rotated
           k, and a query rotated by the same H still scores against it
           (rotations preserve dot products). Doing it before RoPE would
           not: RoPE mixes coordinate i with i+rotary_dim/2, which does
           not commute with H. V is rotated too, and the attention output
           is rotated back below. */
        if (s->kv_rot_k) {
            for (h = 0; h < nh;  h++) rot_head(qht + (size_t)h * hd, hd, s->kv_rot_k);
            for (h = 0; h < nkv; h++) rot_head(ktt + (size_t)h * hd, hd, s->kv_rot_k);
        }
        if (s->kv_rot_v)
            for (h = 0; h < nkv; h++) rot_head(vtt + (size_t)h * hd, hd, s->kv_rot_v);
        /* Q8 writes, per side. Guarded on the pointer rather than on the
           format so a side whose plane was not allocated can never be
           written through - that is the shape of the segfault --kv q4
           already caused once, in lz_state_reset. */
#if LZ_KV_2PLANE
        /* hi and the scale come out byte-identical to lz_quantize_q8's -
           that is lz_gdn_quantize_2p's documented contract - so enabling
           the low plane cannot perturb the high one. fc_site -1: this is
           the KV cache write, not the recurrence write-back, and has no
           census site of its own yet. */
        if (kc) lz_gdn_quantize_2p(ktt, kvd, 32, kc + (size_t)kv_slot(s, pos) * kvd,
                                   kc_lo + (size_t)kv_slot(s, pos) * kvd,
                                   ks + (size_t)kv_slot(s, pos) * kvd / 32, -1);
        if (vc) lz_gdn_quantize_2p(vtt, kvd, 32, vc + (size_t)kv_slot(s, pos) * kvd,
                                   vc_lo + (size_t)kv_slot(s, pos) * kvd,
                                   vs + (size_t)kv_slot(s, pos) * kvd / 32, -1);
#else
        if (kc) lz_quantize_q8(ktt, kvd, 32, kc + (size_t)kv_slot(s, pos) * kvd,
                               ks + (size_t)kv_slot(s, pos) * kvd / 32);
        /* v_int says the projection left this token in vtmp_i16 at
           LZ_VPROJ_ES and vtt holds nothing - the int16 twin of the
           quantize below, folding 2^-ES into the stored group scale. */
        if (v_int) {
            lz_quantize_q8_i16(s->vtmp_i16 + (size_t)tk * kvd, kvd, 32,
                               pow2f(-LZ_VPROJ_ES),
                               vc + (size_t)kv_slot(s, pos) * kvd,
                               vs + (size_t)kv_slot(s, pos) * kvd / 32);
            lz_debug_vproj_i16 += kvd;
        } else if (vc) {
            lz_quantize_q8(vtt, kvd, 32, vc + (size_t)kv_slot(s, pos) * kvd,
                           vs + (size_t)kv_slot(s, pos) * kvd / 32);
        }
#endif /* LZ_KV_2PLANE */
        /* Reference arm: the same post-QK-norm, post-RoPE (and, when
           enabled, post-Hadamard) vectors the Q8 path just quantized,
           kept exactly. Written alongside rather than instead of the Q8
           planes so the two differ ONLY in the cache format. */
        if (kf) memcpy(kf + (size_t)kv_slot(s, pos) * kvd, ktt, (size_t)kvd * sizeof(float));
        if (vf) memcpy(vf + (size_t)kv_slot(s, pos) * kvd, vtt, (size_t)kvd * sizeof(float));
        /* Per HEAD, not per row: the norm that gets split off has to be
           the norm of the vector the attention actually dots against,
           and that is one head's worth. */
        if (v4)
            for (h = 0; h < nkv; h++)
                lz_kv4_quantize(vtt + (size_t)h * hd, hd,
                                v4 + (size_t)kv_slot(s, pos) * kvd / 2 + (size_t)h * (hd / 2),
                                vs4 + (size_t)kv_slot(s, pos) * nkv + h);
        if (k4) {
            for (h = 0; h < nkv; h++) {
                {
                    unsigned char *kc4 = k4 + (size_t)kv_slot(s, pos) * kvd / 2 +
                                         (size_t)h * (hd / 2);
                    float *ksc = ks4 + (size_t)kv_slot(s, pos) * nkv + h;
                    lz_kv4_quantize(ktt + (size_t)h * hd, hd, kc4, ksc);
                }
            }
        }
    }
    /* Post QK-Norm, post RoPE, pre-cache-quantization. These two are the
       last taps on the attention path that are still exact floats, so
       they are where a rotary_dim / head interleave / start_pos error has
       to show up before the KV cache's Q8 noise can hide it. */
    LZ_TAP("aq", layer, s->qh, qd);
    LZ_TAP("ak", layer, s->ktmp, kvd);

    for (tk = 0; tk < nt; tk++) {
        const float *qht = s->qh + (size_t)tk * qd;
        const float *qgt = s->qg + (size_t)tk * qgd;
        float *aot = s->attn_out + (size_t)tk * qd;
        int pos = pos0 + tk;
        /* win_t0: lz_debug_mtp_attn_window's own comment above - 0
           (normal) unless this is the MTP's own attention AND the
           investigative window override is active. */
        int win_t0 = 0;
        int dead0, dead1;            /* see LZ_ATTN_DEAD below */
        if (layer < 0 && lz_debug_mtp_attn_window > 0 &&
            pos - lz_debug_mtp_attn_window + 1 > win_t0)
            win_t0 = pos - lz_debug_mtp_attn_window + 1;
        /* lz_debug_mtp_attn_rows's own comment above - real ALU-cost
           proxy for the MTP's own attention (score loop runs once per
           head, (pos-win_t0+1) rows each), not just a byte count. Body
           layers (layer >= 0) never increment this. */
        if (layer < 0) lz_debug_mtp_attn_rows += (long long)(pos - win_t0 + 1) * nh;

        /* Positions [dead0, dead1) fall outside sink+window. The mask
           block further down already sets their att[] to -1e30f, so
           SCORING them is dead work and their contribution to the value
           sum is exactly zero - exp(-1e30 - max) underflows to +0.0f and
           a*0 adds nothing. Skipping them is therefore bit-identical.

           This skip is the compute half of the eviction change. The
           memory half landed first (kunkun98-pilot at 2048 tokens:
           4.6 -> 3.0 MB at window 256); the loops kept walking the
           whole prefix, and the compute half measured nothing (2.173 vs
           2.123 ms/token of attention, window off vs on).

           dead1 stays 0 when no window is set, so LZ_ATTN_DEAD is false
           for every t on the default path and this costs one predictable
           compare per row. */
        {
            int d0 = win_t0 > s->attn_sink ? win_t0 : s->attn_sink;
            dead0 = d0;
            dead1 = 0;
            if (s->attn_win > 0) {
                dead1 = pos + 1 - s->attn_win;
                if (dead1 < s->attn_sink) dead1 = s->attn_sink;
            }
        }
#define LZ_ATTN_DEAD(t) ((t) >= dead0 && (t) < dead1)
/* The kernels take kv_slot()'s two parameters and walk the ring
   themselves; what they still cannot express is win_t0, the
   investigative override that leaves att[0..win_t0-1] unscored. Rows the
   eviction MASK drops need no guard: the mask runs after scoring and
   overwrites whatever the kernel put there, and after softmax their
   weight is exactly zero. */
#define LZ_ATTN_NORING (win_t0 == 0)

        /* --attn-int: the int wsum keeps the exact int64 accumulator for
           lz_quantize_q8_int64 instead of the float dequant + re-quantize.
           Active only where the float chain would otherwise be the wsum's
           fixed tier: o_proj must quantize at gs==32 so each Q8 group is
           one 32-wide wsum group with ONE sscale; kv_rot_v==0 keeps the
           float un-rotate out (the int path has no Hadamard); v4/vf and
           the 2-plane low term are float-only consumers of `out`. */
#if (LZ_ATTN_FIXED & 2)
        /* wsum_cbuf and LZ_ATTN_MAX_HD only exist when the fixed wsum is
           compiled in, and the int path rides on it - so the predicate
           is compiled out with it rather than merely evaluating false. */
        attn_int[tk] = lz_attn_int() && (lz_attn_mode() & 2) &&
                       s->wsum_cbuf && LZ_ATTN_NORING &&
                       hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD &&
                       s->kv_rot_v == 0 && gso == 32 && !v4 && !vf &&
                       !LZ_KV_2PLANE;
#else
        attn_int[tk] = 0;
#endif /* LZ_ATTN_FIXED & 2 */

        for (h = 0; h < nh; h++) {
            const float *qhh = qht + (size_t)h * hd;
            int kvh = h / kv_mul;
            float *out = aot + (size_t)h * hd;

            if (k4) {
                /* score = sum_d q[d] * (scale_t * cents[code]) - the row
                   scale factors straight out of the sum, so it costs one
                   multiply per (row, head), not per coordinate. The
                   codebook is 64 bytes and stays in L1; that is what
                   makes a table lookup per coordinate affordable here
                   and would not be with a per-coordinate table. */
                for (t = win_t0; t <= pos; t++) {
                    if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                    const unsigned char *kt = k4 + (size_t)kv_slot(s, t) * kvd / 2 +
                                              (size_t)kvh * (hd / 2);
                    float sum = 0.0f;
                    for (d = 0; d < hd; d += 2) {
                        unsigned char pk = kt[d / 2];
                        sum += qhh[d]     * lz_kv4_cents[pk & 0x0F];
                        sum += qhh[d + 1] * lz_kv4_cents[pk >> 4];
                    }
                    s->att[t] = sum * ks4[(size_t)kv_slot(s, t) * nkv + kvh] * scale;
                }
            } else
            if (kf) {
                /* Reference scoring: no quantization anywhere on this
                   path, so any gap between this and the Q8 arm IS the
                   cache format's cost - which is the whole point of
                   having the arm. */
                for (t = win_t0; t <= pos; t++) {
                    if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                    const float *kt = kf + (size_t)kv_slot(s, t) * kvd + (size_t)kvh * hd;
                    float sum = 0.0f;
                    for (d = 0; d < hd; d++) sum += qhh[d] * kt[d];
                    s->att[t] = sum * scale;
                }
            } else
            {
#if (LZ_ATTN_FIXED & 1)
            /* The fixed path needs head_dim to be a multiple of 32 and
               within its static per-head buffers; anything else keeps the
               float loops rather than being silently truncated. The
               shape test is ANDed with the runtime tier, so a model the
               kernel cannot take falls back without the selector having
               to know about head_dim.
               LZ_ATTN_NORING is the third condition and the expensive one
               to have learned: both fixed kernels index the cache by
               ABSOLUTE t, so they read the wrong row the moment kv_slot()
               stops being the identity, and they score positions the
               eviction mask has dropped. */
            if ((lz_attn_mode() & 1) && LZ_ATTN_NORING &&
                hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                lz_attn_score_q8(s->att, qhh, hd,
                                 kc + (size_t)kvh * hd,
                                 ks + (size_t)(kvh * hd) / 32,
                                 kvd, pos, scale, s->attn_sink, s->kv_ring);
            } else
#endif /* LZ_ATTN_FIXED & 1 */
            for (t = win_t0; t <= pos; t++) {
                if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                /* scoring: q (f32) × k (Q8). float accumulation within
                   a group, one scale multiply at the group tail
                   (head_dim is a multiple of 32; saves a per-element
                   scale multiply). win_t0 is 0 (t starts at 0, as
                   always) unless the investigative window override
                   above is active. */
                const int8_t *kt = kc + (size_t)kv_slot(s, t) * kvd + (size_t)kvh * hd;
                const float *kts = ks + (size_t)kv_slot(s, t) * kvd / 32 +
                                   (size_t)(kvh * hd) / 32;
                float sum = 0.0f;
                /* A convert, a multiply and an add per element; a
                   multiply and an add per group; one output scale. */
                LZ_FCX(LZ_FC_ATTN_SCORE, hd + (hd / 32) + 1, hd + (hd / 32), 0, hd, 0);
                for (d = 0; d < hd; d += 32) {
                    float acc = 0.0f;
                    int e;
                    for (e = 0; e < 32; e++) acc += qhh[d + e] * (float)kt[d + e];
                    sum += acc * kts[d / 32];
                }
                s->att[t] = sum * scale;
            }
#if LZ_KV_2PLANE
            /* Scoring is linear in k, so score(hi + lo/254) = score(hi)
               + score(lo)/254. This is a SECOND PASS over the same
               kernels - no new kernel and no second pair of assembly
               implementations to keep bit-identical.
               Folded BEFORE softmax: the low plane corrects the logit,
               not the probability. */
            {
                int tt;
#if (LZ_ATTN_FIXED & 1)
                if ((lz_attn_mode() & 1) && LZ_ATTN_NORING &&
                    hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                    lz_attn_score_q8(s->att_lo, qhh, hd,
                                     kc_lo + (size_t)kvh * hd,
                                     ks + (size_t)(kvh * hd) / 32,
                                     kvd, pos, scale,
                                     s->attn_sink, s->kv_ring);
                } else
#endif /* LZ_ATTN_FIXED & 1 */
                for (tt = 0; tt <= pos; tt++) {
                    const int8_t *kt = kc_lo + (size_t)kv_slot(s, tt) * kvd + (size_t)kvh * hd;
                    const float *kts = ks + (size_t)kv_slot(s, tt) * kvd / 32 +
                                       (size_t)(kvh * hd) / 32;
                    float sum = 0.0f;
                    /* Same shape as the scoring loop above: convert +
                       multiply + add per element, multiply + add per
                       group, one output scale. */
                    LZ_FCX(LZ_FC_ATTN_SCORE, hd + (hd / 32) + 1, hd + (hd / 32), 0, hd, 0);
                    for (d = 0; d < hd; d += 32) {
                        float acc = 0.0f;
                        int e;
                        for (e = 0; e < 32; e++)
                            acc += qhh[d + e] * (float)kt[d + e];
                        sum += acc * kts[d / 32];
                    }
                    s->att_lo[tt] = sum * scale;
                }
                for (tt = 0; tt <= pos; tt++)
                    s->att[tt] += s->att_lo[tt] * (1.0f / LZ_GDN_LO_SCALE);
            }
#endif /* LZ_KV_2PLANE */
            }
            /* win_t0>0 means s->att[0..win_t0-1] were never scored this
               call (investigative window override) - softmax only the
               slice actually filled, [win_t0..pos]. Ordinary case
               (win_t0==0) is unchanged: lz_softmax(s->att, pos+1). */
            /* StreamingLLM eviction, as a MASK rather than an eviction
               (--attn-sink / --attn-window). Keeping the first sink_n
               positions and the most recent window_w, and masking what
               falls between, reproduces the attention that a
               start+recent cache would compute - without touching the
               cache layout, the hot scoring loops, or any of the four
               format branches above.

               That ordering is deliberate. The memory saving is the
               point of the technique, but it is not the RISK: the risk
               is that a small hybrid model loses too much by forgetting
               the middle. Measure the quality with an instrument that
               cannot introduce bugs of its own, and only then pay for
               the eviction machinery.

               -1e30f rather than -INFINITY: the window always contains
               `pos`, so at least one entry survives, but a sentinel that
               cannot produce inf-minus-inf inside lz_softmax is worth
               more than the exactness of a true infinity. */
            if (s->attn_win > 0 || s->attn_sink > 0) {
                int lo = (s->attn_win > 0) ? pos + 1 - s->attn_win : 0;
                if (lo < s->attn_sink) lo = s->attn_sink;
                for (t = win_t0 > s->attn_sink ? win_t0 : s->attn_sink;
                     t < lo; t++)
                    s->att[t] = -1e30f;
            }
            lz_softmax(s->att + win_t0, pos + 1 - win_t0);

            if (v4) {
                memset(out, 0, (size_t)hd * sizeof(float));
                for (t = win_t0; t <= pos; t++) {
                    if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                    const unsigned char *vt = v4 + (size_t)kv_slot(s, t) * kvd / 2 +
                                              (size_t)kvh * (hd / 2);
                    /* Row scale folded into the attention weight, same
                       trick as the scoring pass. */
                    float a = s->att[t] * vs4[(size_t)kv_slot(s, t) * nkv + kvh];
                    for (d = 0; d < hd; d += 2) {
                        unsigned char pk = vt[d / 2];
                        out[d]     += a * lz_kv4_cents[pk & 0x0F];
                        out[d + 1] += a * lz_kv4_cents[pk >> 4];
                    }
                }
            } else
            if (vf) {
                memset(out, 0, (size_t)hd * sizeof(float));
                for (t = win_t0; t <= pos; t++) {
                    if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                    const float *vt = vf + (size_t)kv_slot(s, t) * kvd + (size_t)kvh * hd;
                    float a = s->att[t];
                    LZ_FCX(LZ_FC_ATTN_WSUM, hd, hd, 0, 0, 0);   /* f32 V: no convert */
                    for (d = 0; d < hd; d++) out[d] += a * vt[d];
                }
            } else
            {
#if (LZ_ATTN_FIXED & 2)
            /* s->wsum_cbuf is in the condition, not asserted: an embedder
               may call lz_attn_select after the state exists, and the
               allocation above cannot see that. Reading the buffer the
               kernel needs is the only test that cannot drift from it. */
            if (attn_int[tk]) {
                lz_attn_wsum_q8_int(s->attn_acc + (size_t)tk * qd + (size_t)h * hd,
                                    s->attn_ss + (size_t)tk * (qd / 32) + (size_t)h * (hd / 32),
                                    s->att, hd,
                                    vc + (size_t)kvh * hd,
                                    vs + (size_t)(kvh * hd) / 32,
                                    kvd, pos, s->wsum_cbuf, s->wsum_cq,
                                    s->attn_sink, s->kv_ring);
            } else
            if ((lz_attn_mode() & 2) && s->wsum_cbuf && LZ_ATTN_NORING &&
                hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                lz_attn_wsum_q8(out, s->att, hd,
                                vc + (size_t)kvh * hd,
                                vs + (size_t)(kvh * hd) / 32,
                                kvd, pos, s->wsum_cbuf, s->wsum_cq,
                                s->attn_sink, s->kv_ring);
            } else {
#endif /* LZ_ATTN_FIXED & 2 */
            memset(out, 0, (size_t)hd * sizeof(float));
            for (t = win_t0; t <= pos; t++) {
                if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                const int8_t *vt = vc + (size_t)kv_slot(s, t) * kvd + (size_t)kvh * hd;
                const float *vts = vs + (size_t)kv_slot(s, t) * kvd / 32 +
                                   (size_t)(kvh * hd) / 32;
                float a = s->att[t];
                /* Convert, two multiplies and an add per element. */
                LZ_FCX(LZ_FC_ATTN_WSUM, 2 * hd, hd, 0, hd, 0);
                for (d = 0; d < hd; d++) out[d] += a * (float)vt[d] * vts[d / 32];
            }
#if (LZ_ATTN_FIXED & 2)
            }
#endif /* LZ_ATTN_FIXED & 2 */
#if LZ_KV_2PLANE
            /* Same argument as the scoring pass: the weighted sum is
               linear in v. */
            {
#if (LZ_ATTN_FIXED & 2)
                if ((lz_attn_mode() & 2) && s->wsum_cbuf && LZ_ATTN_NORING &&
                    hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                    lz_attn_wsum_q8(s->wsum_lo, s->att, hd,
                                    vc_lo + (size_t)kvh * hd,
                                    vs + (size_t)(kvh * hd) / 32,
                                    kvd, pos, s->wsum_cbuf, s->wsum_cq,
                                    s->attn_sink, s->kv_ring);
                } else {
#endif /* LZ_ATTN_FIXED & 2 */
                memset(s->wsum_lo, 0, (size_t)hd * sizeof(float));
                for (t = 0; t <= pos; t++) {
                    if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                    const int8_t *vt = vc_lo + (size_t)kv_slot(s, t) * kvd + (size_t)kvh * hd;
                    const float *vts = vs + (size_t)kv_slot(s, t) * kvd / 32 +
                                       (size_t)(kvh * hd) / 32;
                    float a = s->att[t];
                    /* Same shape as the wsum loop above: convert, two
                       multiplies and an add per element. */
                    LZ_FCX(LZ_FC_ATTN_WSUM, 2 * hd, hd, 0, hd, 0);
                    for (d = 0; d < hd; d++)
                        s->wsum_lo[d] += a * (float)vt[d] * vts[d / 32];
                }
#if (LZ_ATTN_FIXED & 2)
                }
#endif /* LZ_ATTN_FIXED & 2 */
                for (d = 0; d < hd; d++)
                    out[d] += s->wsum_lo[d] * (1.0f / LZ_GDN_LO_SCALE);
            }
#endif /* LZ_KV_2PLANE */
            }
            /* Rotate the attention output back. It is a linear combination
               of rotated V rows, so H applied once more undoes the
               rotation up to the factor kv_rot_v that lz_fwht carries
               (H H = n I). This has to happen BEFORE the output gate
               below: the gate is elementwise against qgt, which lives in
               the original basis. */
            if (s->kv_rot_v) {
                float invn = 1.0f / (float)s->kv_rot_v;
                rot_head(out, hd, s->kv_rot_v);
                for (d = 0; d < hd; d++) out[d] *= invn;
            }
        }

        /* output gate: same source as q, take the second half of each head.
           The int path gates in the int domain - acc * Q15(sigmoid) >> 15
           keeps the same sscale. The Q15 sigmoid is the FLOAT path's own
           sigmoid (lz_sigmoid, whatever tier) rounded to 32768-scale, so
           the gate semantics match the float chain exactly; only the
           integer-domain accumulator and the >> 15 divide change.

           Fixed tier: lz_sigmoid(g) is (float)sigmoid_q15(g) * 2^-15, so
           `... * 32768.0f` below just undoes that scaling - both factors
           are exact powers of two (sigmoid_q15's range is [0, 32768], well
           inside float's 24-bit mantissa), so the round trip is exact and
           sigmoid_q15(g) can be called directly, dropping the cvt+2mul.
           Float tier keeps the original expression: lz_sigmoid there is
           exp-based, a different value from sigmoid_q15's table. */
        if (attn_int[tk]) {
            int sig_fixed = lz_sig_mode();
            for (h = 0; h < nh; h++) {
                const float *gate = qgt + (size_t)h * 2 * hd + hd;
                int64_t *ac = s->attn_acc + (size_t)tk * qd + (size_t)h * hd;
                for (d = 0; d < hd; d++) {
                    int32_t s15 = sig_fixed ? sigmoid_q15(gate[d])
                                 : (int32_t)(lz_sigmoid(gate[d]) * 32768.0f);
                    ac[d] = (ac[d] * (int64_t)s15) >> 15;
                }
            }
        } else {
            for (h = 0; h < nh; h++) {
                const float *gate = qgt + (size_t)h * 2 * hd + hd;
                float *out = aot + (size_t)h * hd;
                for (d = 0; d < hd; d++) out[d] *= lz_sigmoid(gate[d]);
            }
        }
    }
#undef LZ_ATTN_DEAD
    LZ_TAP("ax", layer, s->attn_out, qd);

    /* o_proj input is the attention output (attn_q_dim); quantize once more.
       The int path quantizes each 32-group from attn_acc with its own
       sscale: attn_int requires gso==32, so one Q8 group == one wsum group
       and the xq/xqs layout is identical to the float call below. */
    for (tk = 0; tk < nt; tk++) {
        if (attn_int[tk]) {
            int g;
            for (g = 0; g < qd / 32; g++)
                lz_quantize_q8_int64(s->attn_acc + (size_t)tk * qd + g * 32,
                                     32, 32,
                                     s->attn_ss[(size_t)tk * (qd / 32) + g],
                                     s->xq + (size_t)tk * qd + g * 32,
                                     s->xqs + (size_t)tk * nso + g);
        } else {
            lz_quantize_q8(s->attn_out + (size_t)tk * qd, qd, gso,
                           s->xq + (size_t)tk * qd, s->xqs + (size_t)tk * nso);
        }
    }
    lz_matmul_xq_nt(s->xb2, s->attn_out, s->xq, s->xqs, &L->o_proj, qd,
                    hdim, nt);
}

/* ------------------------------------------------ linear_attention block */

/* advance_ring/ring_base (the SSM/conv rollback ring - see forward.h's
   s->ssm_slot and forward_chunk's own ring_base comment):
   advance_ring is forward_chunk's all_logits, ring_base is s->ssm_slot
   as it stood BEFORE this chunk's layer loop started (read once by the
   caller, not here, so it does not drift as forward_ssm/forward_kda
   both get called once per LINEAR/KDA layer within the same chunk).
   advance_ring==0 (ordinary decode/prefill): every token reads and
   writes ring_base, bit-identical to the pre-ring single-slot version.
   advance_ring!=0 (a verify batch): token tk reads slot
   (ring_base+tk)%ring_depth and writes (ring_base+tk+1)%ring_depth -
   see lz_gdn_step's own header comment (ops.h) for why this costs zero
   extra bandwidth over the non-ring case, only memory. */
static void forward_ssm(const LZModel *m, LZRunState *s,
                        const LZLayer *L, int layer, int nt,
                        int advance_ring, int ring_base) {
    const LZModelConfig *c = &m->config;
    int nk = c->lin_n_k_heads, nv = c->lin_n_v_heads;
    int kd = c->lin_k_head_dim, vd = c->lin_v_head_dim;
    int li = s->cache_idx[layer];
    int ring_depth = s->ssm_ring_depth;
    size_t q8_slot_stride = (size_t)c->n_linear_layers * nv * kd * vd;
    size_t s_slot_stride  = q8_slot_stride / 32;
    size_t conv_slot_stride = (size_t)c->n_linear_layers *
                              c->lin_conv_dim * (c->conv_kernel - 1);
    size_t li_off_q8 = (size_t)li * nv * kd * vd;
    size_t li_off_conv = (size_t)li * c->lin_conv_dim * (c->conv_kernel - 1);
    float scale = s->ssm_scale;
    int hdim = c->hidden_size, cdim = c->lin_conv_dim, vdim = c->lin_value_dim;
    int gsa = lz_act_gs(&L->in_proj_qkv, hdim);
    int nsa = (gsa > 0) ? hdim / gsa : 0;
    int h, i, tk;

    /* quantize xb once; the four input projections share it (all in_dim = hidden) */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsa,
                       s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
    lz_matmul_xq_nt(s->qkv, s->xb, s->xq, s->xqs, &L->in_proj_qkv, hdim,
                    cdim, nt);
    lz_matmul_xq_nt(s->zbuf, s->xb, s->xq, s->xqs, &L->in_proj_z, hdim,
                    vdim, nt);
    lz_matmul_xq_nt(s->avec, s->xb, s->xq, s->xqs, &L->in_proj_a, hdim, nv, nt);
    lz_matmul_xq_nt(s->bvec, s->xb, s->xq, s->xqs, &L->in_proj_b, hdim, nv, nt);

    /* depthwise causal convolution, SiLU activation */
    LZ_TAP("sqkv", layer, s->qkv, cdim);
    LZ_TAP("sz", layer, s->zbuf, vdim);
    LZ_TAP("sa", layer, s->avec, nv);
    LZ_TAP("sb", layer, s->bvec, nv);
    /* The conv advances serially over t: it reads/writes the rolling
       conv_state history, which is inherently ordered. **Deliberately
       not batched** - conv weights are only lin_conv_dim×k floats, 0.2%
       of per-token bytes; the bandwidth saved is not worth the risk of
       changing reduction order. Ring-aware per this function's
       own header comment - slot_in==slot_out when !advance_ring. */
    for (tk = 0; tk < nt; tk++) {
        int slot_in  = advance_ring ? (ring_base + tk) % ring_depth : ring_base;
        int slot_out = advance_ring ? (ring_base + tk + 1) % ring_depth : ring_base;
#if LZ_CONV_FIXED
        if (s->conv_fixed) {
            /* One block here, unlike KDA's three: this path's conv
               covers lin_conv_dim in a single tensor. Wiring only KDA
               would leave a model with SSM layers half on each tier,
               which no banner line would show. */
            size_t cb = (size_t)li * c->lin_conv_dim;
            /* No int16 input here (int-pipeline milestone 3 wired only
               KDA's three): the producer is in_proj_qkv, and on every
               checkpoint in this tree that tensor is EITHER absent
               (kmr20/kunmoe-v2: n=0, so this whole function never runs)
               OR f32 (s1v3, kunkun98-pilot, kunmoe-v2-dense: n=1572864,
               dtype f32). f32 leaves lz_matmul_xq_nt at its first branch
               and never reaches the Q4_1/Q6_1/T2 epilogue that has the
               int16 exit, so there is nothing to wire and no gate could
               exercise it. Not a tier split of the kind the comment
               below warns about: this conv stays wholly on the float
               input, all lin_conv_dim channels of it. */
            lz_causal_conv1d_step_fixed(
                s->qkv_c + (size_t)tk * cdim, s->qkv + (size_t)tk * cdim, NULL,
                s->conv_state_q + (size_t)slot_in * conv_slot_stride + li_off_conv,
                s->conv_state_q + (size_t)slot_out * conv_slot_stride + li_off_conv,
                s->conv_mw + cb * c->conv_kernel,
                s->conv_sig_k1 + cb, s->conv_sig_oscale2 + cb, s->conv_sig_k2,
                LZ_CONV_SIG_E(s, cb),
                s->conv_in_scale, s->conv_bound, cdim, c->conv_kernel);
            continue;
        }
#endif /* LZ_CONV_FIXED */
        lz_causal_conv1d_step(s->qkv_c + (size_t)tk * cdim,
                              s->qkv + (size_t)tk * cdim,
                              s->conv_state + (size_t)slot_in * conv_slot_stride + li_off_conv,
                              s->conv_state + (size_t)slot_out * conv_slot_stride + li_off_conv,
                              lz_t_f32(&L->conv1d, s->wscr), cdim,
                              c->conv_kernel);
    }
    LZ_TAP("sconv", layer, s->qkv_c, cdim);

    /* Recurrence is organized head-major (fix a head, run T steps), not
       t-major. One head's state is kd×vd = 4 KB, resident in L1 for T
       steps - saving (T-1)/T of the per-token 1.33 MB state traffic.
       Heads are independent and each head is strictly serial in t, so
       this reordering changes no numbers. */
    for (h = 0; h < nv; h++) {
        /* In this model nv == nk, one-to-one; when nv > nk the reference
           uses repeat_interleave, equivalent to taking key head
           h * nk / nv. */
        int kh = (nk == nv) ? h : (h * nk / nv);
        size_t h_off_q8 = li_off_q8 + (size_t)h * kd * vd;
        size_t h_off_s  = h_off_q8 / 32;
        /* exp(A_log[h]) depends on the head only, not the token - hoist
           it out of the tk loop so it is computed once per head, not once
           per token per head (the same fix as forward_kda's decay_base). */
        float a_exp = lz_exp(lz_t_f32(&L->A_log, s->wscr)[h]);
        float dt_bias_h = lz_t_f32(&L->dt_bias, s->wscr)[h];

        for (tk = 0; tk < nt; tk++) {
            int slot_in  = advance_ring ? (ring_base + tk) % ring_depth : ring_base;
            int slot_out = advance_ring ? (ring_base + tk + 1) % ring_depth : ring_base;
            const int8_t *sq_in  = s->ssm_state_q8 + (size_t)slot_in  * q8_slot_stride + h_off_q8;
            int8_t       *sq_out = s->ssm_state_q8 + (size_t)slot_out * q8_slot_stride + h_off_q8;
#if LZ_GDN_STATE_2PLANE
            const int8_t *sq2_in  = s->ssm_state_q8_lo + (size_t)slot_in  * q8_slot_stride + h_off_q8;
            int8_t       *sq2_out = s->ssm_state_q8_lo + (size_t)slot_out * q8_slot_stride + h_off_q8;
#endif /* LZ_GDN_STATE_2PLANE */
            const float *ss_in  = s->ssm_state_s + (size_t)slot_in  * s_slot_stride + h_off_s;
            float       *ss_out = s->ssm_state_s + (size_t)slot_out * s_slot_stride + h_off_s;
            const float *qp = s->qkv_c + (size_t)tk * cdim;
            const float *kp = qp + c->lin_key_dim;
            const float *vp = kp + c->lin_key_dim;
            const float *v_t = vp + (size_t)h * vd;
            float *out = s->ssm_out + (size_t)tk * vdim + (size_t)h * vd;
            float beta, g, gt;

            lz_l2norm(s->qn, qp + (size_t)kh * kd, kd, LZ_L2NORM_EPS);
            lz_l2norm(s->kn, kp + (size_t)kh * kd, kd, LZ_L2NORM_EPS);
            for (i = 0; i < kd; i++) s->qn[i] *= scale;

            beta = lz_sigmoid(s->bvec[(size_t)tk * nv + h]);
            /* A = exp(A_log) > 0 and softplus > 0, so g < 0 and
               gt = exp(g) lands in (0,1) - a decay factor. If gt >= 1
               ever comes out, a sign is flipped. */
            g = -a_exp *
                lz_softplus(s->avec[(size_t)tk * nv + h] + dt_bias_h);
            gt = lz_exp(g);

            /* State is read/written in two passes; quantization error
               decays per step with gt<1 (unlike KV cache, it does not
               accumulate). Self-consistency is guaranteed by
               lz_gdn_step: the next token reads the quantized value.
               sq_in/sq_out (and sq2/ss) differ only during a
               speculative verify batch - see this function's own
               signature comment. */
            LZ_PROF_BEG(_tr);
            lz_gdn_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                        sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                        ss_in, ss_out, s->qn, s->kn, v_t, gt, beta, kd, vd);
            LZ_PROF_END(_tr, LZ_PROF_REC);
        }
    }

    /* This is the plain-weight gated RMSNorm, not the (1+w) kind used
       by the outer layers. The fixed tier computes the full silu value
       in Q15 int-multiply (value-changing) and writes it into o;
       lz_quantize_q8 quantizes it. With --norm-int the fixed tier
       writes the Q15 value as int into ssm_sig instead and hands the
       power-of-two scale out-of-band, so lz_quantize_q8_int folds it
       into the stored group scale and the dequant round-trip is gone
       (bit-identical - see lz_quantize_q8_int). The float tier keeps
       the plain norm + lz_quantize_q8 path. */
    {
        int gso = lz_act_gs(&L->out_proj, vdim);
        int nso = (gso > 0) ? vdim / gso : 0;
        if (lz_norm_int() && lz_norm_can_fixed(vd) &&
            gso > 0 && (vd % gso) == 0) {
            /* Per-head quantize, not one whole-vdim call: the norm runs
               per head and each head carries its own deq, so a Q8 group
               must not straddle two heads' different scales. gso divides
               vd, so the group boundaries and the s->xqs layout are
               identical to the whole-vdim call below. ssm_sig is
               otherwise unused in this path (the sig argument is NULL)
               and is allocated nt*vdim int32, the same span as ssm_out. */
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++) {
                    float deq;
                    lz_rmsnorm_gated_int(s->ssm_sig + (size_t)tk * vdim + (size_t)h * vd,
                                         s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                         s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                         lz_t_f32(&L->ssm_norm, s->wscr), vd,
                                         c->rms_norm_eps, &deq);
                    lz_quantize_q8_int(s->ssm_sig + (size_t)tk * vdim + (size_t)h * vd,
                                       vd, gso, deq,
                                       s->xq + (size_t)tk * vdim + (size_t)h * vd,
                                       s->xqs + (size_t)tk * nso + (size_t)h * (vd / gso));
                }
        } else if (lz_norm_can_fixed(vd)) {
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++)
                    lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                     lz_t_f32(&L->ssm_norm, s->wscr), vd,
                                     c->rms_norm_eps,
                                     NULL);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                               s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        } else {
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++)
                    lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                     lz_t_f32(&L->ssm_norm, s->wscr), vd,
                                     c->rms_norm_eps, NULL);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                               s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        }
        LZ_TAP("sgdn", layer, s->ssm_out, vdim);
        lz_matmul_xq_nt(s->xb2, s->ssm_out, s->xq, s->xqs, &L->out_proj, vdim,
                        hdim, nt);
    }
}

/* ------------------------------------------------------ KDA (LZ_LT_KDA) block */

/* SubLN block-diagonal Hadamard + per-token absmax INT8 quantize for the
   o_proj / down_proj input (BitNet v2 H-BitLinear, minus the INT4 stage).
   x is the post-RMSNorm float activation row, n its width, blk the
   Hadamard block (a power of two dividing n; the training side caps it at
   512 for down_proj and uses value_dim for o_proj), fscr 2*n int32
   scratch.

   The transform is the fixed-point FWHT (lz_fwht_i32): the row is moved
   to int32 at a per-row power-of-two scale (row max ~2^14, so the
   unnormalized output's worst case blk*2^14 stays inside 2^24 and every
   int32->float is exact), then each blk-wide block butterflies in pure
   add/sub. The unnormalized transform grows RMS by sqrt(blk); 1/sqrt(blk)
   is folded into the STORED scale (the H.264 pattern), so the kernels
   stay integer and the dequant matches the training side's orthonormal
   _block_hadamard within quantization tolerance.

   gw is the width of one stored scale: n for BitNet v2's per-token
   absmax, or the engine's own activation group (32) when --subn-scale
   group is on. It is NOT the Hadamard block - the transform runs
   identically either way, and the grouping applies to the ROTATED int32
   values, where it is exact: within a group the dequant q[i]*s[g] still
   reproduces fscr[i]*inv_S/sqrt(blk), because inv_S and sqrt(blk) are
   row-wide constants and only the 127/tam fold is per group. Whichever
   gw is used, one xqs slot is still written per 32 elements, which is
   what the matmul reads.

   q/s semantics match lz_quantize_q8 (round-half-even, clamp
   [-127,127], all-zero group scale 1.0). */
static void subn_fwht_quant(const float *x, int n, int blk, int gw,
                            int8_t *q, float *s, int32_t *fscr) {
    union { float f; uint32_t u; } b;
    float amax, S, inv_S, inv_q, scale, rs;
    int e, i, b_i, base;
    b.u = q8_amax(x, n);
    amax = b.f;
    if (amax <= 0.0f) {          /* all-zero row: q all 0, scale 1.0 */
        for (i = 0; i < n; i++) q[i] = 0;
        for (i = 0; i < n / 32; i++) s[i] = 1.0f;
        return;
    }
    /* Power-of-two fixed-point scale: row max lands in [2^14, 2^15), so
       the worst-case unnormalized FWHT output is blk*2^15 <= 2^24. */
    e = (int)((b.u >> 23) & 0xFFu) - 127;
    S = pow2f(14 - e);
    inv_S = 1.0f / S;            /* exact: S is a power of two */
    for (i = 0; i < n; i++) fscr[i] = q8_round(x[i] * S);
    for (b_i = 0; b_i < n; b_i += blk)          /* block-diagonal Sylvester */
        lz_fwht_i32(fscr + n + b_i, fscr + b_i, blk);
    if (gw <= 0 || (gw & 31) != 0 || (n % gw) != 0) gw = n;
    /* Hoisted, not folded into a single constant: the multiply chain
       below must stay (tam*inv_S)*rs*INV127 in that order, or the
       per-token arm stops being bit-identical to the engine that had no
       gw. lz_rsqrt_float_body is pure and blk is row-wide, so calling it
       once here is the same call the row made before. */
    rs = lz_rsqrt_float_body((float)blk);
    for (base = 0; base < n; base += gw) {
        int32_t tam = 0;
        for (i = base; i < base + gw; i++) {
            int32_t a = fscr[n + i] < 0 ? (int32_t)(0u - (uint32_t)fscr[n + i])
                                        : fscr[n + i];
            if (a > tam) tam = a;
        }
        /* Dequant scale = post-Hadamard amax / sqrt(blk) / 127. The
           element loop rounds t*127/tam, identical to
           lz_quantize_q8_int's fold. tam == 0 is the group's fixed-point
           collapse and takes the same scale-1.0 exit q8_group_scale
           gives a zero group. */
        scale = (float)tam * inv_S * rs * LZ_Q8_INV127;
        if (tam == 0 || scale < LZ_Q8_MIN_SCALE_F) {
            scale = 1.0f;
            inv_q = 0.0f;
        } else {
            inv_q = 127.0f / (float)tam;
        }
        for (i = base; i < base + gw; i++) {
            int qv = q8_round((float)fscr[n + i] * inv_q);
            qv = (qv > 127) ? 127 : qv;
            q[i] = (int8_t)((qv < -127) ? -127 : qv);
        }
        for (i = base / 32; i < (base + gw) / 32; i++) s[i] = scale;
    }
}

/* Width of one stored activation scale on the SubLN o_proj/down_proj
   input, resolved from --subn-scale. 0 means "the whole row" and is
   what subn_fwht_quant / lz_quantize_q8_tok already do; gs is the
   site's lz_act_gs, which is 0 when the width is not a multiple of 32.
   Returning 0 for a gs this path cannot honour keeps the fallback in
   ONE place instead of at each of the four call sites. */
static int subn_scale_gs(int gs, int n) {
    if (!lz_subn_scale_group()) return 0;
    if (gs <= 0 || (gs & 31) != 0 || n < gs || (n % gs) != 0) return 0;
    return gs;
}

/* The SAME block-diagonal Hadamard, applied to the float row.
   subn_fwht_quant writes q/s and never touches x, so on the one path
   where lz_matmul_xq_nt reads x - an F32 weight - the rotation would
   otherwise not happen at all, and the model would compute W.x while
   the quantized build of the same config computes W.quant(Hx). Those
   are different functions, which is what made the unquantized run
   useless as a reference for the quantized one.

   Float, not the fixed-point twin, because this row IS the reference:
   lz_fwht takes no constants, so it is bit-identical across
   wcc386 and gcc, and the orthonormal form matches the training side's
   _block_hadamard exactly rather than within quantization tolerance.
   The 1/sqrt(blk) uses lz_rsqrt_float_body, the same call
   subn_fwht_quant folds into its stored scale, so the two paths differ
   by the transform's arithmetic and NOT by the normalisation constant.

   In place: the caller has already quantized from this row, and the
   matmul is the only reader left. */
static void subn_rot_f32(float *x, int n, int blk) {
    float inv = lz_rsqrt_float_body((float)blk);
    int b_i, i;
    for (b_i = 0; b_i < n; b_i += blk) lz_fwht(x + b_i, blk);
    for (i = 0; i < n; i++) x[i] *= inv;
}

/* Kimi Delta Attention: same role as forward_ssm's GatedDeltaNet block,
   generalized to a per-channel decay gate and three independent q/k/v
   projections (KdaAttention). See ops.h's
   lz_kda_step for the recurrence derivation - this function does not
   re-derive it, only wires the projections/conv/gate around it.

   Batches across nt for every projection that reads hidden-space input
   directly (q/k/v/f_a/b/g projections all share one xb quantization,
   the same trick forward_ssm/forward_attn use for their own hidden-space
   projections); f_b_proj and the conv are NOT nt-batched - f_b_proj
   because its input (kda_gate_lat) is a different width than hidden and
   needs its own quantization per token, the conv because it is
   inherently serial in t (same reasoning as forward_ssm's conv). */
/* One q/k/v projection feeding the fixed conv, int-pipeline milestone 3.
   With `*use_int` the matmul exits as int16 already at LZ_CONV_ES and
   clamped to s->conv_bound, and the conv takes it as-is: the producer's
   `(float)comb * pow2f(target)` and the consumer's `x[c] * in_scale` +
   q8_round both stop existing rather than moving somewhere else.

   ALL THREE OR NONE, and the flag is what enforces it. A refusal here
   cannot be recovered from per-projection: under SubLN each projection
   re-quantizes s->xq/s->xqs from its own norm, so an earlier one's float
   buffer can no longer be recomputed by the time a later one refuses.
   It also cannot happen - lz_matmul_xq_i16_ok is asked for q, k and v
   together before the first norm runs, and it IS the predicate the entry
   point asks, whose answer depends only on (dtype, in_dim, nt), the same
   three for all of q/k/v. Clearing the flag on the way to the float
   matmul is therefore correct for the only reachable case (the FIRST
   projection refusing, which sends the whole triple to float) and is
   here so that a future divergence degrades instead of reading a stale
   buffer. Same argument the fixed conv tier's own GDN comment makes
   about not splitting one SSM block across two tiers. */
static void kda_conv_proj(LZRunState *s, int *use_int, float *of, short *oi,
                          const float *x, const LZTensor *w,
                          int in_dim, int out_dim, int nt) {
#if LZ_CONV_FIXED
    if (*use_int) {
        if (lz_matmul_xq_nt_i16(oi, s->xq, s->xqs, w, in_dim, out_dim, nt,
                                LZ_CONV_ES, s->conv_bound))
            return;
        *use_int = 0;
    }
#else
    (void)use_int; (void)oi;
#endif /* LZ_CONV_FIXED */
    lz_matmul_xq_nt(of, x, s->xq, s->xqs, w, in_dim, out_dim, nt);
}

/* The same shape for the projections whose consumer is not the conv
   (int-pipeline milestone 5): one projection, one int16 exit at the
   consumer's own exponent, and the SAME all-or-nothing flag discipline
   kda_conv_proj documents at length.

   The capability comes IN as a value (`may_int`, from
   lz_matmul_xq_i16_ok); `*wrote_int` goes OUT and means one thing only:
   this call put its result in `oi`. Two separate objects on purpose. A
   single in/out flag reads as the capability everywhere it is set and
   as "the int buffer holds the output" everywhere it is consumed, so a
   branch that writes `of` WITHOUT coming through here leaves a
   capability behind for the consumer to read as a promise, and the
   consumer takes a stale int buffer. Splitting them makes that
   unrepresentable: `*wrote_int` starts at 0, nothing but this function
   ever raises it, and adding such a branch is now a compile error at
   the call rather than a silent wrong read at the consumer. */
static void kda_i16_proj(LZRunState *s, int may_int, int *wrote_int,
                         float *of, short *oi,
                         const float *x, const LZTensor *w,
                         int in_dim, int out_dim, int nt,
                         int target_e, int bound) {
    if (may_int &&
        lz_matmul_xq_nt_i16(oi, s->xq, s->xqs, w, in_dim, out_dim, nt,
                            target_e, bound)) {
        *wrote_int = 1;
        return;
    }
    *wrote_int = 0;
    lz_matmul_xq_nt(of, x, s->xq, s->xqs, w, in_dim, out_dim, nt);
}

/* advance_ring/ring_base: same contract as forward_ssm's own (see its
   header comment above) - reused verbatim, not re-derived, since both
   functions share the SAME ring buffers (ssm_state_q8/_lo/_s,
   conv_state) and the SAME per-chunk base forward_chunk computes once. */
static void forward_kda(const LZModel *m, LZRunState *s,
                        const LZLayer *L, int layer, int nt,
                        int advance_ring, int ring_base) {
    const LZModelConfig *c = &m->config;
    int nk = c->lin_n_k_heads, nv = c->lin_n_v_heads;
    int kd = c->lin_k_head_dim, vd = c->lin_v_head_dim;
    int gate_rank = c->kda_gate_rank;
    int li = s->cache_idx[layer];
    int ring_depth = s->ssm_ring_depth;
    size_t q8_slot_stride = (size_t)c->n_linear_layers * nv * kd * vd;
    size_t s_slot_stride  = q8_slot_stride / 32;
    size_t conv_slot_stride = (size_t)c->n_linear_layers *
                              c->lin_conv_dim * (c->conv_kernel - 1);
    size_t li_off_q8 = (size_t)li * nv * kd * vd;
    /* Same per-layer conv_state allocation GDN uses (sized by
       lin_conv_dim regardless of how that width is split), sliced into
       three independent regions [q | k | v] - matching the concat order
       kunmoe_modeling.py's own decode path uses for the same reason:
       three depthwise convs over disjoint channels == one depthwise
       conv over their concatenation, so the state layout is identical
       either way. hist/li_off_conv/q_off/k_off/v_off are the pieces a
       ring-slot base gets added to below - kept as offsets rather than
       resolved pointers because which slot applies varies per token
       (advance_ring). */
    int hist = c->conv_kernel - 1;
    size_t li_off_conv = (size_t)li * c->lin_conv_dim * hist;
    size_t q_off = 0;
    size_t k_off = (size_t)c->lin_key_dim * hist;
    size_t v_off = (size_t)2 * c->lin_key_dim * hist;
    float scale = s->ssm_scale;
    int hdim = c->hidden_size, kdim = c->lin_key_dim, vdim = c->lin_value_dim;
    int nvk = nv * kd;
    int gsa = lz_act_gs(&L->kda_q_proj, hdim);
    int nsa = (gsa > 0) ? hdim / gsa : 0;
    int h, i, tk;
    /* int-pipeline milestone 3: does the whole q/k/v triple qualify for
       the int16 matmul exit the fixed conv can consume directly? Decided
       ONCE, here, before any projection runs - see kda_conv_proj. The
       fixed conv tier is the consumer, so its absence answers this on its
       own; the three tensor questions are asked in full (no short-circuit
       reasoning from "they are the same dtype") because a checkpoint may
       quantize them differently. */
    int conv_int = 0;
#if LZ_CONV_FIXED
    if (s->conv_fixed && s->kda_q_i16 && s->kda_k_i16 && s->kda_v_i16) {
        int okq = lz_matmul_xq_i16_ok(&L->kda_q_proj, hdim, nt);
        int okk = lz_matmul_xq_i16_ok(&L->kda_k_proj, hdim, nt);
        int okv = lz_matmul_xq_i16_ok(&L->kda_v_proj, hdim, nt);
        conv_int = okq && okk && okv;
    }
#endif /* LZ_CONV_FIXED */
    /* int-pipeline milestone 5, decided here for the same reason
       conv_int is: the producing matmul has to know before it runs, and
       the consumer reads the answer far below.

       lz_sig_mode() gates bvec and the gate because the FLOAT sigmoid
       tier is not this table at all (lz_exp-based, a different value),
       so there is no integer entry to hand an integer to - the same
       shape as "the fixed conv tier is the consumer, so its absence
       answers this on its own". --sig-write float therefore turns both
       off at runtime, which is what makes it a control. */
    /* *_may is the capability, *_int is "the int buffer holds it" and is
       raised only by kda_i16_proj - see its header for why they are two
       objects. */
    int bvec_int = 0, lat_int = 0, gate_int = 0;
    int bvec_may = 0, lat_may = 0;
#if LZ_BVEC_I16
    bvec_may = lz_sig_mode() && s->bvec_i16 &&
               lz_matmul_xq_i16_ok(&L->kda_b_proj, hdim, nt);
#endif /* LZ_BVEC_I16 */
#if LZ_KLAT_I16
    lat_may = s->kda_lat_i16 && s->kda_lat_i32 && gate_rank > 0 &&
              lz_matmul_xq_i16_ok(&L->kda_f_a_proj, hdim, nt);
#endif /* LZ_KLAT_I16 */
#if LZ_KGATE_I16
    /* nt == 1 is not a restriction of the exit, it is what f_b_proj's
       call site is: this projection streams one token at a time (its
       in_dim is not hidden, so it cannot share the batched
       quantization), so the exit is asked for one token too.

       kda_has_gate_lower_bound is a CONSUMER question, not a taste one.
       With the lower bound the gate's transcendental is the sigmoid,
       which has an integer entry; without it the gate is
       lz_softplus(pre), and this tree has no fixed-point softplus at
       all, so that branch would have to rebuild the float the epilogue
       just declined to build - the work moved, not removed. */
    gate_int = lz_sig_mode() && s->kda_gate_i16 && s->kda_dtb_i16 &&
               c->kda_has_gate_lower_bound &&
               lz_matmul_xq_i16_ok(&L->kda_f_b_proj, gate_rank, 1);
#endif /* LZ_KGATE_I16 */

    if (c->use_subn) {
        /* SubLN per-projection input RMSNorms for the four hidden-space
           ternary projections (q/k/v/g): each normalizes the
           UN-normalized residual with its own norm weight before its
           matmul (the two pre-layer norms are deleted - see the layer
           loop in forward_chunk). They share s->xb2 as scratch, free
           until the o_proj matmul below overwrites it: each norm result
           is consumed by its own matmul before the next norm runs, and
           for Q8 weights the matmul reads the quantized xq/xqs, not the
           float scratch. f_a/b_proj stay on raw s->xb - no SubLN norm
           field exists for the low-rank decay-gate projections. */
        /* --norm-write int: fixed-tier norm writes a Q15 int product
           straight into subn_norm_int (single-token scratch, not
           nt-wide - see forward.h), the immediate tok-int quantize
           reads it, and s->xb2 is never touched. Falls back to the
           float norm + lz_quantize_q8_tok pair when the tier is off or
           hdim is out of norm_ss_fixed's range. */
        if (lz_norm_int() && lz_norm_can_fixed(hdim) &&
            gsa > 0 && (hdim % gsa) == 0) {
            for (tk = 0; tk < nt; tk++) {
                float deq;
                lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * hdim,
                               lz_t_f32(&L->kda_q_norm, s->wscr), hdim,
                               c->rms_norm_eps, &deq);
                lz_quantize_q8_tok_int(s->subn_norm_int, hdim, deq,
                                       s->xq + (size_t)tk * hdim,
                                       s->xqs + (size_t)tk * nsa);
            }
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * hdim, s->xb + (size_t)tk * hdim,
                           lz_t_f32(&L->kda_q_norm, s->wscr), hdim, c->rms_norm_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8_tok(s->xb2 + (size_t)tk * hdim, hdim,
                                   s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
        }
        kda_conv_proj(s, &conv_int, s->kda_q, s->kda_q_i16, s->xb2,
                      &L->kda_q_proj, hdim, kdim, nt);

        if (lz_norm_int() && lz_norm_can_fixed(hdim) &&
            gsa > 0 && (hdim % gsa) == 0) {
            for (tk = 0; tk < nt; tk++) {
                float deq;
                lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * hdim,
                               lz_t_f32(&L->kda_k_norm, s->wscr), hdim,
                               c->rms_norm_eps, &deq);
                lz_quantize_q8_tok_int(s->subn_norm_int, hdim, deq,
                                       s->xq + (size_t)tk * hdim,
                                       s->xqs + (size_t)tk * nsa);
            }
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * hdim, s->xb + (size_t)tk * hdim,
                           lz_t_f32(&L->kda_k_norm, s->wscr), hdim, c->rms_norm_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8_tok(s->xb2 + (size_t)tk * hdim, hdim,
                                   s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
        }
        kda_conv_proj(s, &conv_int, s->kda_k, s->kda_k_i16, s->xb2,
                      &L->kda_k_proj, hdim, kdim, nt);

        if (lz_norm_int() && lz_norm_can_fixed(hdim) &&
            gsa > 0 && (hdim % gsa) == 0) {
            for (tk = 0; tk < nt; tk++) {
                float deq;
                lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * hdim,
                               lz_t_f32(&L->kda_v_norm, s->wscr), hdim,
                               c->rms_norm_eps, &deq);
                lz_quantize_q8_tok_int(s->subn_norm_int, hdim, deq,
                                       s->xq + (size_t)tk * hdim,
                                       s->xqs + (size_t)tk * nsa);
            }
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * hdim, s->xb + (size_t)tk * hdim,
                           lz_t_f32(&L->kda_v_norm, s->wscr), hdim, c->rms_norm_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8_tok(s->xb2 + (size_t)tk * hdim, hdim,
                                   s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
        }
        kda_conv_proj(s, &conv_int, s->kda_v, s->kda_v_i16, s->xb2,
                      &L->kda_v_proj, hdim, vdim, nt);

        if (lz_norm_int() && lz_norm_can_fixed(hdim) &&
            gsa > 0 && (hdim % gsa) == 0) {
            for (tk = 0; tk < nt; tk++) {
                float deq;
                lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * hdim,
                               lz_t_f32(&L->kda_g_norm, s->wscr), hdim,
                               c->rms_norm_eps, &deq);
                lz_quantize_q8_tok_int(s->subn_norm_int, hdim, deq,
                                       s->xq + (size_t)tk * hdim,
                                       s->xqs + (size_t)tk * nsa);
            }
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * hdim, s->xb + (size_t)tk * hdim,
                           lz_t_f32(&L->kda_g_norm, s->wscr), hdim, c->rms_norm_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8_tok(s->xb2 + (size_t)tk * hdim, hdim,
                                   s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
        }
        lz_matmul_xq_nt(s->zbuf, s->xb2, s->xq, s->xqs, &L->kda_g_proj,
                        hdim, vdim, nt);

        /* f_a/b_proj: no SubLN norm field - quantize raw s->xb (the
           shared-activation-quantization trick still applies to these
           two). */
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8_tok(s->xb + (size_t)tk * hdim, hdim,
                               s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
        kda_i16_proj(s, lat_may, &lat_int, s->kda_gate_lat, s->kda_lat_i16,
                     s->xb, &L->kda_f_a_proj, hdim, gate_rank, nt,
                     LZ_KLAT_ES, 32767);
        kda_i16_proj(s, bvec_may, &bvec_int, s->bvec, s->bvec_i16,
                     s->xb, &L->kda_b_proj, hdim, nv, nt,
                     LZ_BVEC_ES, lz_sig_q15_domain(LZ_BVEC_ES));
    } else {
    /* Six hidden-space projections share one activation quantization,
       the same trick forward_ssm uses for its four (all have
       in_dim == hidden_size, so lz_act_gs agrees regardless of format). */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsa,
                       s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
    kda_conv_proj(s, &conv_int, s->kda_q, s->kda_q_i16, s->xb,
                  &L->kda_q_proj, hdim, kdim, nt);
    kda_conv_proj(s, &conv_int, s->kda_k, s->kda_k_i16, s->xb,
                  &L->kda_k_proj, hdim, kdim, nt);
    kda_conv_proj(s, &conv_int, s->kda_v, s->kda_v_i16, s->xb,
                  &L->kda_v_proj, hdim, vdim, nt);
    kda_i16_proj(s, lat_may, &lat_int, s->kda_gate_lat, s->kda_lat_i16,
                 s->xb, &L->kda_f_a_proj, hdim, gate_rank, nt,
                 LZ_KLAT_ES, 32767);
    kda_i16_proj(s, bvec_may, &bvec_int, s->bvec, s->bvec_i16,
                 s->xb, &L->kda_b_proj, hdim, nv, nt,
                 LZ_BVEC_ES, lz_sig_q15_domain(LZ_BVEC_ES));
    lz_matmul_xq_nt(s->zbuf, s->xb, s->xq, s->xqs, &L->kda_g_proj, hdim, vdim, nt);
    }
    /* These read the FLOAT projection buffers, which hold nothing when
       the matching flag took the int16 exit - a tap build wanting
       q/k/v/f_a/b there has to dump kda_q_i16/kda_k_i16/kda_v_i16
       (conv_int), kda_lat_i16 (lat_int) or bvec_i16 (bvec_int). */
    LZ_TAP("kq", layer, s->kda_q, kdim);
    LZ_TAP("kk", layer, s->kda_k, kdim);
    LZ_TAP("kv", layer, s->kda_v, vdim);
    LZ_TAP("kfa", layer, s->kda_gate_lat, gate_rank);
    LZ_TAP("kb", layer, s->bvec, nv);
    LZ_TAP("kg", layer, s->zbuf, vdim);

    /* Depthwise causal convolution, SiLU activation - three independent
       calls over disjoint channel ranges of the same per-layer state
       buffer (q_off/k_off/v_off above), each serial in t like
       forward_ssm's single conv. Ring-aware per this function's own
       header comment - slot_in==slot_out when !advance_ring. */
    for (tk = 0; tk < nt; tk++) {
        int slot_in  = advance_ring ? (ring_base + tk) % ring_depth : ring_base;
        int slot_out = advance_ring ? (ring_base + tk + 1) % ring_depth : ring_base;
#if LZ_CONV_FIXED
        if (s->conv_fixed) {
            /* Same three disjoint ranges, int16 history. The channel
               base is the history offset divided by hist - the two index
               different things (elements vs channels) and mixing them
               silently reads a neighbouring channel's taps. */
            size_t cb = (size_t)li * c->lin_conv_dim;
            short *qi = s->conv_state_q + (size_t)slot_in  * conv_slot_stride + li_off_conv;
            short *qo = s->conv_state_q + (size_t)slot_out * conv_slot_stride + li_off_conv;
            size_t q_ch = 0, k_ch = (size_t)c->lin_key_dim;
            size_t v_ch = (size_t)2 * c->lin_key_dim;
            /* `conv_int` says the projections above exited as int16 at
               LZ_CONV_ES already, so these read that and the float
               buffers hold nothing for this token. NULL is the float
               input the tier was written with. */
            const short *qxi = conv_int ? s->kda_q_i16 + (size_t)tk * kdim : NULL;
            const short *kxi = conv_int ? s->kda_k_i16 + (size_t)tk * kdim : NULL;
            const short *vxi = conv_int ? s->kda_v_i16 + (size_t)tk * vdim : NULL;
            lz_causal_conv1d_step_fixed(
                s->kda_qc + (size_t)tk * kdim, s->kda_q + (size_t)tk * kdim,
                qxi, qi + q_off, qo + q_off,
                s->conv_mw + (cb + q_ch) * c->conv_kernel,
                s->conv_sig_k1 + cb + q_ch, s->conv_sig_oscale2 + cb + q_ch,
                s->conv_sig_k2, LZ_CONV_SIG_E(s, cb + q_ch),
                s->conv_in_scale, s->conv_bound, kdim, c->conv_kernel);
            lz_causal_conv1d_step_fixed(
                s->kda_kc + (size_t)tk * kdim, s->kda_k + (size_t)tk * kdim,
                kxi, qi + k_off, qo + k_off,
                s->conv_mw + (cb + k_ch) * c->conv_kernel,
                s->conv_sig_k1 + cb + k_ch, s->conv_sig_oscale2 + cb + k_ch,
                s->conv_sig_k2, LZ_CONV_SIG_E(s, cb + k_ch),
                s->conv_in_scale, s->conv_bound, kdim, c->conv_kernel);
            lz_causal_conv1d_step_fixed(
                s->kda_vc + (size_t)tk * vdim, s->kda_v + (size_t)tk * vdim,
                vxi, qi + v_off, qo + v_off,
                s->conv_mw + (cb + v_ch) * c->conv_kernel,
                s->conv_sig_k1 + cb + v_ch, s->conv_sig_oscale2 + cb + v_ch,
                s->conv_sig_k2, LZ_CONV_SIG_E(s, cb + v_ch),
                s->conv_in_scale, s->conv_bound, vdim, c->conv_kernel);
            continue;
        }
#endif /* LZ_CONV_FIXED */
        {
        float *base_in  = s->conv_state + (size_t)slot_in  * conv_slot_stride + li_off_conv;
        float *base_out = s->conv_state + (size_t)slot_out * conv_slot_stride + li_off_conv;
        lz_causal_conv1d_step(s->kda_qc + (size_t)tk * kdim,
                              s->kda_q + (size_t)tk * kdim,
                              base_in + q_off, base_out + q_off,
                              lz_t_f32(&L->kda_q_conv1d, s->wscr), kdim,
                              c->conv_kernel);
        lz_causal_conv1d_step(s->kda_kc + (size_t)tk * kdim,
                              s->kda_k + (size_t)tk * kdim,
                              base_in + k_off, base_out + k_off,
                              lz_t_f32(&L->kda_k_conv1d, s->wscr), kdim,
                              c->conv_kernel);
        lz_causal_conv1d_step(s->kda_vc + (size_t)tk * vdim,
                              s->kda_v + (size_t)tk * vdim,
                              base_in + v_off, base_out + v_off,
                              lz_t_f32(&L->kda_v_conv1d, s->wscr), vdim,
                              c->conv_kernel);
        }
    }
    LZ_TAP("kqc", layer, s->kda_qc, kdim);
    LZ_TAP("kkc", layer, s->kda_kc, kdim);
    LZ_TAP("kvc", layer, s->kda_vc, vdim);

    /* f_b_proj: kda_gate_rank -> nvk, a DIFFERENT in_dim than hidden, so
       it needs its own quantization - lz_matmul_w does that internally,
       one token at a time (same reasoning forward_moe's per-expert
       matmuls use).

       Both of this matmul's sides are int-pipeline milestone 5 exits and
       they are INDEPENDENT: `lat_int` is its input (kda_gate_lat, from
       f_a_proj) and `gate_int` is its output (kda_gate, read by the decay
       gate below). The float body is lz_matmul_w spelled out - that
       function IS lz_quantize_q8 followed by lz_matmul_xq, and
       lz_matmul_xq is lz_matmul_xq_nt at nt == 1 - so that either side
       can be replaced without the other. */
    {
        int gsb = lz_act_gs(&L->kda_f_b_proj, gate_rank);
        int retry = 1;
        while (retry) {
            retry = 0;
            for (tk = 0; tk < nt; tk++) {
                if (lat_int) {
                    /* Widening, not requantizing: lz_quantize_q8_int
                       already takes exactly this - an integer vector plus
                       the power-of-two scale that turns it back into the
                       float the group scale must match - and it is the
                       entry the norm-int tier already uses at eight other
                       call sites. Only its int32 input width differs from
                       the exit's int16, and an int-to-int copy costs no
                       float op and introduces no rounding. */
                    int k;
                    const short *li = s->kda_lat_i16 + (size_t)tk * gate_rank;
                    for (k = 0; k < gate_rank; k++) s->kda_lat_i32[k] = li[k];
                    lz_quantize_q8_int(s->kda_lat_i32, gate_rank, gsb,
                                       pow2f(-LZ_KLAT_ES), s->xq, s->xqs);
                } else {
                    lz_quantize_q8(s->kda_gate_lat + (size_t)tk * gate_rank,
                                   gate_rank, gsb, s->xq, s->xqs);
                }
                if (gate_int) {
                    if (lz_matmul_xq_nt_i16(
                            s->kda_gate_i16 + (size_t)tk * nvk,
                            s->xq, s->xqs, &L->kda_f_b_proj,
                            gate_rank, nvk, 1, LZ_KGATE_ES, 32767))
                        continue;
                    /* Not reachable - the predicate above IS this entry
                       point's own, on the same (w, in_dim, nt). Handled
                       anyway, and handled by starting the chunk OVER on
                       the float exit: falling back for this token alone
                       would strand the earlier tokens' int buffers, the
                       mid-triple hazard kda_conv_proj names. */
                    gate_int = 0;
                    retry = 1;
                    break;
                }
                lz_matmul_xq_nt(s->kda_gate + (size_t)tk * nvk,
                                s->kda_gate_lat + (size_t)tk * gate_rank,
                                s->xq, s->xqs, &L->kda_f_b_proj,
                                gate_rank, nvk, 1);
            }
        }
    }

    /* Turn the pre-activation into the actual per-channel decay factor
       gt = exp(g). Without a lower bound, g = -exp(A_log[h]) *
       softplus(pre[h,k] + dt_bias[h,k]). With one (K3's gate_lower_bound),
       the kernel REPLACES that formula rather than clamping it:
       g = lower_bound * sigmoid(exp(A_log[h]) * (pre[h,k] + dt_bias[h,k]))
       (fla.ops.kda.chunk_kda's own docstring / naive_kda_lowerbound_gate;
       a plain floor on the unbounded formula is a DIFFERENT, wrong
       function - see KdaAttention.decay).
       This is that function, not a
       re-derivation of it - A_log stays PER HEAD (h only), dt_bias is
       PER CHANNEL (h, k), matching that file's own comment about why
       widening A_log was tried and reverted. */
    /* decay_base[h] = exp(A_log[h]) depends on the head ONLY, not on the
       token. Computed once per head here, then reused across nt tokens.
       A_log/dt_bias are per-head/per-channel weights, so their
       dequantization is hoisted out of the tk loop too. decay_base goes
       into s->wscr: A_log/dt_bias are F32 (lz_t_f32 returns t->f, no
       write to wscr), so wscr is free here, and it is sized >= nvk >= nv. */
    const float *a_log_arr = lz_t_f32(&L->kda_A_log, s->wscr);
    const float *dt_bias_arr = lz_t_f32(&L->kda_dt_bias, s->wscr);
    {
        float *decay_base = s->wscr;
        for (h = 0; h < nv; h++) decay_base[h] = lz_exp(a_log_arr[h]);
        /* The int-pipeline form of the SAME function, not a second one
           (int-pipeline milestone 5). With kda_gate arriving as int16 at
           2^-LZ_KGATE_ES and dt_bias pre-quantized into that same domain
           (kda_gate_build), `pre` is an INTEGER add, and the sigmoid is
           reached through sigmoid_q15_t's folded coordinate the way the
           fixed conv reaches it:
             t = pre_i * (decay_base[h] * 2^(4-ES)) + 192
           reproduces (decay_base[h]*pre - LZ_SIG_LO) * LZ_SIG_STEP with
           one multiply and one add, because both LZ_SIG_STEP and the
           offset are exact powers of two applied to an already-computed
           value (lz_sig_q15_t_offset's own argument). The exit folds the
           lower bound and the Q15 descale into one constant.
           Per channel: 3 mul, 1 add, 2 cvt, against the float form's
           add + multiply here and lz_sigmoid's 3 mul, 1 add, 1 cvt plus
           the epilogue's convert and multiply. */
        if (gate_int) {
            float lbq = c->kda_gate_lower_bound * (1.0f / 32768.0f);
            float gk = pow2f(4 - LZ_KGATE_ES);
            float t_off = lz_sig_q15_t_offset();
            const short *dtb = s->kda_dtb_i16 + (size_t)li * nvk;
            for (tk = 0; tk < nt; tk++) {
                const short *gi = s->kda_gate_i16 + (size_t)tk * nvk;
                float *gt = s->kda_gate + (size_t)tk * nvk;
                for (h = 0; h < nv; h++) {
                    float k1 = decay_base[h] * gk;
                    LZ_FCX(LZ_FC_SIGMOID, 3 * (long)kd, (long)kd, 0,
                           2 * (long)kd, 0);
                    for (i = 0; i < kd; i++) {
                        int idx = h * kd + i;
                        /* A BARE cast, and the bound is why: both addends
                           are int16, so |pre| <= 65534, far inside the
                           2^24 where an int32->float conversion is exact
                           on x87 and SSE alike. lz_i32f exists for the
                           conversions that are NOT exact - it costs two
                           converts, a multiply and an add of its own, so
                           using it where the hazard cannot occur would
                           spend four float ops to avoid nothing. Past
                           the conversion both compilers round the
                           multiply once and the add once (PC=24). */
                        int32_t pre = (int32_t)gi[idx] + (int32_t)dtb[idx];
                        int32_t s15 = sigmoid_q15_t((float)pre * k1 + t_off);
                        gt[idx] = lz_exp((float)s15 * lbq);
                    }
                }
            }
        } else
        for (tk = 0; tk < nt; tk++) {
            float *gt = s->kda_gate + (size_t)tk * nvk;
            for (h = 0; h < nv; h++) {
                for (i = 0; i < kd; i++) {
                    int idx = h * kd + i;
                    float pre = gt[idx] + dt_bias_arr[idx];
                    float g;
                    if (c->kda_has_gate_lower_bound)
                        g = c->kda_gate_lower_bound * lz_sigmoid(decay_base[h] * pre);
                    else
                        g = -decay_base[h] * lz_softplus(pre);
                    gt[idx] = lz_exp(g);
                }
            }
        }
    }
    LZ_TAP("kgate", layer, s->kda_gate, nvk);

    /* Recurrence, head-major like forward_ssm's - same L1-residency
       argument applies (one head's state is kd*vd, resident for T
       steps; heads are independent, each strictly serial in t). */
    for (h = 0; h < nv; h++) {
        int kh = (nk == nv) ? h : (h * nk / nv);
        size_t h_off_q8 = li_off_q8 + (size_t)h * kd * vd;
        size_t h_off_s  = h_off_q8 / 32;

        for (tk = 0; tk < nt; tk++) {
            int slot_in  = advance_ring ? (ring_base + tk) % ring_depth : ring_base;
            int slot_out = advance_ring ? (ring_base + tk + 1) % ring_depth : ring_base;
            const int8_t *sq_in  = s->ssm_state_q8 + (size_t)slot_in  * q8_slot_stride + h_off_q8;
            int8_t       *sq_out = s->ssm_state_q8 + (size_t)slot_out * q8_slot_stride + h_off_q8;
#if LZ_GDN_STATE_2PLANE
            const int8_t *sq2_in  = s->ssm_state_q8_lo + (size_t)slot_in  * q8_slot_stride + h_off_q8;
            int8_t       *sq2_out = s->ssm_state_q8_lo + (size_t)slot_out * q8_slot_stride + h_off_q8;
#endif /* LZ_GDN_STATE_2PLANE */
            const float *ss_in  = s->ssm_state_s + (size_t)slot_in  * s_slot_stride + h_off_s;
            float       *ss_out = s->ssm_state_s + (size_t)slot_out * s_slot_stride + h_off_s;
            const float *qp = s->kda_qc + (size_t)tk * kdim + (size_t)kh * kd;
            const float *kp = s->kda_kc + (size_t)tk * kdim + (size_t)kh * kd;
            const float *vp = s->kda_vc + (size_t)tk * vdim + (size_t)h * vd;
            const float *gv = s->kda_gate + (size_t)tk * nvk + (size_t)h * kd;
            float *out = s->ssm_out + (size_t)tk * vdim + (size_t)h * vd;
            float beta;

            lz_l2norm(s->qn, qp, kd, LZ_L2NORM_EPS);
            lz_l2norm(s->kn, kp, kd, LZ_L2NORM_EPS);
            for (i = 0; i < kd; i++) s->qn[i] *= scale;

            /* bvec's only consumer, and with `bvec_int` it never became
               a float: the projection exited int16 at LZ_BVEC_ES and
               lz_sigmoid_i reaches the same table from there. */
            beta = bvec_int
                 ? lz_sigmoid_i(s->bvec_i16[(size_t)tk * nv + h], LZ_BVEC_ES)
                 : lz_sigmoid(s->bvec[(size_t)tk * nv + h]);
            LZ_PROF_BEG(_tr);
            lz_kda_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                        sq2_in, sq2_out,
#endif /* LZ_GDN_STATE_2PLANE */
                        ss_in, ss_out, s->qn, s->kn, vp, gv, beta, kd, vd);
            LZ_PROF_END(_tr, LZ_PROF_REC);
        }
    }

    /* Raw recurrence output, pre-o_norm - separates "the recurrence is
       wrong" from "the gating is wrong" the same way GDN's sgdn tap
       alone cannot; this is what caught a test-fixture bug (v_head_dim
       not a multiple of 32) that "sgdn" alone would have reported as
       "everything downstream of KDA is wrong" instead. */
    LZ_TAP("kraw", layer, s->ssm_out, vdim);
    /* Same gated RMSNorm as GDN (plain-weight, silu gate) - see
       kda_o_norm's field comment in model.h for why the activation must
       stay silu on this engine. Fixed tier computes the full silu value
       in Q15 int-multiply (value-changing); float tier keeps the plain
       norm + quantize path. --norm-int takes the same int-output route
       as forward_ssm's sgdn block above (bit-identical, same guard). */
    {
        int gso = lz_act_gs(&L->kda_o_proj, vdim);
        int nso = (gso > 0) ? vdim / gso : 0;
        if (c->use_subn) {
            /* SubLN: o_proj's OWN input RMSNorm (kda_o_subn_norm) over
               the whole lin_value_dim span, applied AFTER the per-head
               gated silu norm (kda_o_norm). Computed in float regardless
               of the fixed/int tiers: the Q15 int-output shortcut
               materializes the gated result only as an int scale that
               cannot feed a second per-element weight multiply. */
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++)
                    lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                     lz_t_f32(&L->kda_o_norm, s->wscr), vd,
                                     c->rms_norm_eps, NULL);
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->ssm_out + (size_t)tk * vdim,
                           s->ssm_out + (size_t)tk * vdim,
                           lz_t_f32(&L->kda_o_subn_norm, s->wscr), vdim,
                           c->rms_norm_eps);
            /* SubLN Hadamard: o_proj's block, settled at load time
               (model.c resolve_hadamard). 0 = no Hadamard. */
            {
                int oblk = lz_hadamard_on() ? c->hadamard_o : 0;
                int qgw = subn_scale_gs(gso, vdim);
                if (oblk > 1) {
                    for (tk = 0; tk < nt; tk++)
                        subn_fwht_quant(s->ssm_out + (size_t)tk * vdim, vdim,
                                        oblk, qgw, s->xq + (size_t)tk * vdim,
                                        s->xqs + (size_t)tk * nso, s->fwht_scratch);
                    /* Only when the matmul below will read the float row -
                       otherwise this is a second transform nobody reads,
                       on every token, on the target machine. */
                    if (lz_matmul_xq_reads_float_row(&L->kda_o_proj))
                        for (tk = 0; tk < nt; tk++)
                            subn_rot_f32(s->ssm_out + (size_t)tk * vdim,
                                         vdim, oblk);
                } else if (qgw > 0) {
                    for (tk = 0; tk < nt; tk++)
                        lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, qgw,
                                       s->xq + (size_t)tk * vdim,
                                       s->xqs + (size_t)tk * nso);
                } else {
                    for (tk = 0; tk < nt; tk++)
                        lz_quantize_q8_tok(s->ssm_out + (size_t)tk * vdim, vdim,
                                           s->xq + (size_t)tk * vdim,
                                           s->xqs + (size_t)tk * nso);
                }
            }
        } else if (lz_norm_int() && lz_norm_can_fixed(vd) &&
            gso > 0 && (vd % gso) == 0) {
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++) {
                    float deq;
                    lz_rmsnorm_gated_int(s->ssm_sig + (size_t)tk * vdim + (size_t)h * vd,
                                         s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                         s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                         lz_t_f32(&L->kda_o_norm, s->wscr), vd,
                                         c->rms_norm_eps, &deq);
                    lz_quantize_q8_int(s->ssm_sig + (size_t)tk * vdim + (size_t)h * vd,
                                       vd, gso, deq,
                                       s->xq + (size_t)tk * vdim + (size_t)h * vd,
                                       s->xqs + (size_t)tk * nso + (size_t)h * (vd / gso));
                }
        } else if (lz_norm_can_fixed(vd)) {
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++)
                    lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                     lz_t_f32(&L->kda_o_norm, s->wscr), vd,
                                     c->rms_norm_eps,
                                     NULL);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                               s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        } else {
            for (tk = 0; tk < nt; tk++)
                for (h = 0; h < nv; h++)
                    lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                                     s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                                     lz_t_f32(&L->kda_o_norm, s->wscr), vd,
                                     c->rms_norm_eps, NULL);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                               s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        }
        LZ_TAP("sgdn", layer, s->ssm_out, vdim);
        lz_matmul_xq_nt(s->xb2, s->ssm_out, s->xq, s->xqs, &L->kda_o_proj, vdim,
                        hdim, nt);
    }
}

/* ------------------------------------------------ latent MoE FFN block */

/* Latent MoE (LatentMoE / KunMoEGate),
   replacing the dense gate/up/down_proj FFN for layers >=
   config.first_k_dense_replace (LZLayer.ffn_moe).

   FOUR PHASES, split on the one thing that is per-token: expert
   SELECTION. The router, the latent down/up projections and the shared
   expert all read one weight matrix for every token in the chunk, so
   they batch through lz_matmul_xq_nt exactly like dense_ffn_step. Only
   the routed experts cannot - two tokens in the same chunk may pick
   different experts entirely, so there is no shared weight stream.

   WHY NOT GROUP THE TOKENS BY EXPERT, which is what llama.cpp's
   ggml_compute_forward_mul_mat_id (a per-expert row bucket, no padding)
   and vLLM/SGLang's moe_align_block_size do: their scale is not ours.
   SGLang's CPU int8 MoE pads each expert's rows to BLOCK_M = 32 and
   drops the blocked path below M = 5; llama.cpp's tuning assumes
   batches in the thousands. LZ_BATCH_MAX is 8, and at 8 tokens over
   top-2 of 16 experts the expected DISTINCT expert count is
   16*(1-(15/16)^16) ~ 10.3 of 16 pairs - about a third of the weight
   streams, ~9% of prefill, in exchange for reordering the float
   accumulation in the kk loop. llama.cpp's own measurement that
   prompt-phase routing is flat (decode's is skewed) makes prefill the
   worst case for it.

   THE SPLIT, measured rather than assumed: ffn(topk) is
   linear in topk, so sweeping --moe-topk 1/2/4 separates the halves.
   On an 832-token prompt it fit ffn = 856,505 + 787,201*topk us
   exactly - at the trained topk=2 the routed experts are 64.8% of the
   ffn phase and everything batched here is the other 35.2%.

   nt=1 (generation) is unchanged: lz_matmul_xq_nt's t-loop degrades to
   one iteration, which that function's header states as a hard gate.

   Bit-identity between batched and per-token MoE prefill:
   verified across E:\LLM\models\kunmoe-v2 (widths
   1/2/3/4/8, 8 lengths each, plus 6 continuation steps) as well as
   s1v3 - the batch-parity claim reaches this function directly, not
   only a dense model that has no MoE and no KDA. It passes: the loop
   below has no state that carries across tk (xq/xqs are fully
   overwritten each iteration), so there was nothing FOR batch width to
   break here - the gap was real for throughput, not for correctness. */
static void forward_moe(const LZModel *m, LZRunState *s,
                        const LZLayer *L, int layer, int nt) {
    const LZModelConfig *c = &m->config;
    int hdim = c->hidden_size, latent = c->moe_latent_dim;
    int inter = c->moe_intermediate_size, shared_w = c->moe_shared_width;
    int ne = c->num_experts, topk = c->num_experts_per_token;
    /* --moe-topk: route to a different number of experts than the model
       was exported with. Nothing in this function or lz_moe_route is
       written for a particular k - the router pads unfillable slots with
       idx -1 / weight 0, the loop skips them, and moe_renormalize keeps
       the weights summing to one so the output scale does not move with
       k. So this is a knob, not a rewrite.
       What it is NOT is free: the model was TRAINED at its config value,
       and running off that value is off-distribution. It exists so the
       question can be measured - a parameter whose best value differs by
       machine gets swept, not baked - and the scratch buffers are sized from the
       config, so the override is clamped to them. */
    if (lz_moe_topk > 0 && lz_moe_topk <= ne)
        topk = lz_moe_topk;
    int gsh = lz_act_gs(&L->moe_down_proj, hdim);
    int nsh = (gsh > 0) ? hdim / gsh : 0;
    int tk, kk, vv, i;
    /* int-pipeline milestone 5: the router logits' only consumer is
       lz_moe_route, and the exit is the same one bvec takes - int16 at
       LZ_MOEGATE_ES, bounded by the sigmoid table's own domain rather
       than int16's edge. Decided before the matmul, like every other
       exit in this file, and lz_sig_mode() answers it for the same
       reason.
       moe_router_sigmoid is a consumer question too: the softmax scorer
       has no integer entry (lz_exp has no int-in form), so it would
       rebuild the float the epilogue declined to build - the same
       operation count, for a value that has moved. Refuse instead. */
    int mrt_int = 0, mrt_may = 0;
#if LZ_MOEGATE_I16
    mrt_may = lz_sig_mode() && s->moe_logits_i16 && c->moe_router_sigmoid &&
              lz_matmul_xq_i16_ok(&L->moe_gate_w, hdim, nt);
#endif /* LZ_MOEGATE_I16 */
    /* int-pipeline milestone 6: the latent's only consumer is the
       routed experts' activation quantize, which lz_quantize_q8_i16
       serves directly from the int16 - so moe_down_proj's Q8_0
       epilogue can stop building a float that the next statement takes
       apart again. Decided before the matmul, like every other exit in
       this file. */
    int mlat_int = 0, mlat_may = 0;
#if LZ_MLAT_I16
    mlat_may = s->moe_lat_i16 && L->moe_expert_w1 && L->moe_expert_w3 &&
               lz_matmul_xq_i16_ok(&L->moe_down_proj, hdim, nt);
    /* lz_matmul_xq reads its FLOAT argument when - and only when - the
       weight is F32, so an F32 expert would read a moe_lat_x the int
       exit never wrote. Refused for the whole layer rather than per
       expert: which experts a token routes to is not known until after
       the matmul that would have written the float. ne dtype loads per
       call and no float work. */
    for (i = 0; mlat_may && i < ne; i++)
        if (L->moe_expert_w1[i].dtype == LZ_FMT_F32 ||
            L->moe_expert_w3[i].dtype == LZ_FMT_F32)
            mlat_may = 0;
#endif /* LZ_MLAT_I16 */
    /* int-pipeline milestone 8: the two SwiGLU sites in this function.
       Both are decided here, before any matmul, and both are
       all-or-nothing over their own triple (gate, up, down) for the
       reason kda_conv_proj's comment gives - one consumer fed half in
       int and half in float is worse than either.

       The routed predicate scans every expert rather than the two a
       token routes to, exactly as mlat_may above does and for the same
       reason: routing is not known until after the matmul that would
       have written the float. The down-projection is in the scan because
       lz_matmul_xq reads its FLOAT argument when the weight is F32, and
       on the int path that argument is a moe_h1 nothing wrote.

       use_subn refuses BOTH sites even though this function does not
       implement SubLN's per-projection norms at all today. The moment it
       does, moe_shared_down_norm lands between the SwiGLU and its
       quantize, and there is no int16-input norm for it to read - the
       float would have to be rebuilt, which is the swap this milestone
       exists to avoid. Refusing now costs nothing (no MoE checkpoint
       here sets the flag) and keeps the two changes from having to be
       correct together. */
    int rsw_may = 0, ssw_may = 0;
#if LZ_SWIGLU_I16
    rsw_may = s->moe_h1_i16 && s->moe_h3_i16 && !c->use_subn &&
              L->moe_expert_w1 && L->moe_expert_w3 && L->moe_expert_w2;
    for (i = 0; rsw_may && i < ne; i++)
        if (!lz_matmul_xq_i16_ok(&L->moe_expert_w1[i], latent, 1) ||
            !lz_matmul_xq_i16_ok(&L->moe_expert_w3[i], latent, 1) ||
            lz_matmul_xq_reads_float_row(&L->moe_expert_w2[i]))
            rsw_may = 0;
    ssw_may = s->moe_h1_i16 && s->moe_h3_i16 && !c->use_subn && shared_w > 0 &&
              lz_matmul_xq_i16_ok(&L->moe_shared_gate, hdim, nt) &&
              lz_matmul_xq_i16_ok(&L->moe_shared_up, hdim, nt) &&
              !lz_matmul_xq_reads_float_row(&L->moe_shared_down);
#endif /* LZ_SWIGLU_I16 */
    (void)layer;   /* only used inside LZ_TAP, which is a no-op in a normal build */

    /* PHASE 1 - everything that reads xt and does not depend on routing.
       The router and the latent down-projection are ordinary matmuls:
       one weight matrix serves every token in the chunk, so they run
       batched like every other block in this file, instead of streaming
       the same weights nt times.
       They also share one activation quantization, the same reasoning
       forward_attn's q/k/v_proj comment gives. */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsh,
                       s->xq + (size_t)tk * hdim,
                       s->xqs + (size_t)tk * nsh);
    kda_i16_proj(s, mrt_may, &mrt_int, s->moe_router_logits,
                 s->moe_logits_i16,
                 s->xb, &L->moe_gate_w, hdim, ne, nt,
                 LZ_MOEGATE_ES, lz_sig_q15_domain(LZ_MOEGATE_ES));
    kda_i16_proj(s, mlat_may, &mlat_int, s->moe_lat_x, s->moe_lat_i16,
                 s->xb, &L->moe_down_proj, hdim, latent, nt,
                 LZ_MLAT_ES, 32767);
    /* Slot 0 is token 0's, so these tap what the per-token version
       taped under `if (tk == 0)`. Both read a FLOAT buffer that holds
       nothing when its chain took the int16 exit - dump
       moe_logits_i16 / moe_lat_i16 instead, same as the KDA taps
       above. */
    LZ_TAP("mrt", layer, s->moe_router_logits, ne);
    LZ_TAP("mlat", layer, s->moe_lat_x, latent);

    /* PHASE 2 - the routed experts, and the ONE part that stays per
       token: which expert a token wants is a property of that token, so
       there is no shared weight stream to batch over. Grouping the
       tokens by expert instead was measured and rejected - see the
       block comment above this function. */
    for (tk = 0; tk < nt; tk++) {
        float *lat_x = s->moe_lat_x + (size_t)tk * latent;
        const short *lat_i = mlat_int ? s->moe_lat_i16 + (size_t)tk * latent
                                      : NULL;
        float *lat_y = s->moe_lat_y + (size_t)tk * latent;
        int gs_have = 0;        /* group size moe_lat_q currently holds */

        lz_moe_route(s->moe_router_logits + (size_t)tk * ne,
                    mrt_int ? s->moe_logits_i16 + (size_t)tk * ne : NULL,
                    LZ_MOEGATE_ES,
                    lz_t_f32(&L->moe_gate_bias, s->wscr),
                    ne, topk, c->moe_router_sigmoid, c->moe_renormalize,
                    lz_moe_tau, s->moe_sel_idx, s->moe_sel_w);

        /* One layer's contribution to the token's union. Guarded, so a
           caller that did not ask for an inspector pays one predictable
           branch per MoE layer and nothing else - no call, no write. */
        if (s->ins)
            lz_moe_hits_add(s->ins->expert_hits, LZ_INSPECT_EXPERT_MAX,
                            s->moe_sel_idx, topk);

        memset(lat_y, 0, (size_t)latent * sizeof(float));
        for (kk = 0; kk < topk; kk++) {
            int ei = s->moe_sel_idx[kk];
            float ww = s->moe_sel_w[kk];
            if (ei < 0) continue;               /* topk > n_experts padding */
            /* KdaExpert.forward: w2(act(w1(x)) * w3(x)) - w1 is the gate,
               w3 the up projection, w2 the down projection back to latent. */
            /* w1 and w3 both read the latent with the same in_dim, so
               quantize it once and share, instead of letting each
               lz_matmul_w re-quantize the identical vector. The
               quantization ALSO survives across experts, because the
               vector does not change and lz_act_gs is a function of
               (in_dim, whether the weight's gs is a sane multiple of
               32) - not of which expert. So it is redone only when the
               group size actually differs, which keeps the
               mixed-precision hazard forward_attn's q/k/v comment names
               (w1 and w3 must read the SAME quantization) while paying
               for it once per DISTINCT group size instead of once per
               expert. Measured invariant on both checkpoints: 128
               w1/w3 pairs each, lz_act_gs == 32 throughout, so the
               second expert's quantize is pure repetition.

               Its output is moe_lat_q/_qs and NOT s->xq/s->xqs: the w2
               matmul below quantizes moe_h1 into those, so a result
               left there would be gone before the next expert reads it.
               That, not the group size, is what used to force the
               quantize back inside this loop. */
            {
                int gsl = lz_act_gs(&L->moe_expert_w1[ei], latent);
                /* LZ_MLAT_REUSE=0 is this saving's control arm: quantize
                   on every expert, as the loop did before the hoist.
                   Folds at compile time; the destination is moe_lat_q/_qs
                   either way, so the arm isolates the SKIP and not the
                   buffer move, and reproduces the pre-hoist cost exactly. */
                if (!LZ_MLAT_REUSE || gsl != gs_have) {
                    if (mlat_int)
                        lz_quantize_q8_i16(lat_i, latent, gsl,
                                           pow2f(-LZ_MLAT_ES),
                                           s->moe_lat_q, s->moe_lat_qs);
                    else
                        lz_quantize_q8(lat_x, latent, gsl,
                                       s->moe_lat_q, s->moe_lat_qs);
                    gs_have = gsl;
                }
            }
#if LZ_SWIGLU_I16
            /* The int path, when rsw_may said the whole triple can take
               it. Both exits are asked here rather than through a
               kda_i16_proj-style helper because there is nothing to
               recover: rsw_may already ran lz_matmul_xq_i16_ok on both
               tensors, and that predicate IS the entry point's own, so a
               refusal below is unreachable. The `&&` is the
               all-or-nothing enforcement anyway - a w3 that somehow
               refused leaves moe_h1_i16 written and moe_h1 stale, and
               the float branch below would then read the stale float, so
               the second call's failure has to take the first one's
               result with it. It does: nothing between here and the
               requantize reads either buffer. */
            if (rsw_may &&
                lz_matmul_xq_nt_i16(s->moe_h1_i16, s->moe_lat_q, s->moe_lat_qs,
                                    &L->moe_expert_w1[ei], latent, inter, 1,
                                    LZ_SWIGLU_ES_P, 32767) &&
                lz_matmul_xq_nt_i16(s->moe_h3_i16, s->moe_lat_q, s->moe_lat_qs,
                                    &L->moe_expert_w3[ei], latent, inter, 1,
                                    LZ_SWIGLU_ES_G, 32767)) {
                lz_swiglu_q15_i16(s->moe_h1_i16, s->moe_h1_i16, s->moe_h3_i16,
                                  inter, LZ_SWIGLU_ES_P, LZ_SWIGLU_ES_G,
                                  LZ_SWIGLU_ES_O);
                /* lz_matmul_w's two halves, with the quantize taking the
                   int16 instead of a float. The float argument is NULL
                   because moe_h1 holds nothing on this path and the
                   weight is not F32 (rsw_may checked). */
                lz_quantize_q8_i16(s->moe_h1_i16, inter,
                                   lz_act_gs(&L->moe_expert_w2[ei], inter),
                                   pow2f(-LZ_SWIGLU_ES_O), s->xq, s->xqs);
                lz_matmul_xq(s->moe_h2, NULL, s->xq, s->xqs,
                             &L->moe_expert_w2[ei], inter, latent);
            } else {
#endif /* LZ_SWIGLU_I16 */
            lz_matmul_xq(s->moe_h1, lat_x, s->moe_lat_q, s->moe_lat_qs,
                        &L->moe_expert_w1[ei], latent, inter);
            lz_matmul_xq(s->moe_h3, lat_x, s->moe_lat_q, s->moe_lat_qs,
                        &L->moe_expert_w3[ei], latent, inter);
            for (vv = 0; vv < inter; vv++)
                s->moe_h1[vv] = lz_silu(s->moe_h1[vv]) * s->moe_h3[vv];
            lz_matmul_w(s->moe_h2, s->moe_h1, &L->moe_expert_w2[ei],
                       inter, latent, s->xq, s->xqs);
#if LZ_SWIGLU_I16
            }
#endif /* LZ_SWIGLU_I16 */
            for (vv = 0; vv < latent; vv++)
                lat_y[vv] += ww * s->moe_h2[vv];
        }
        if (c->moe_latent_use_norm)
            lz_rmsnorm(lat_y, lat_y,
                      lz_t_f32(&L->moe_latent_norm, s->wscr), latent,
                      c->rms_norm_eps);
    }
    LZ_TAP("mrou", layer, s->moe_lat_y, latent);

    /* PHASE 3 - back out of the latent space. Routing-independent
       again, so batched. */
    {
        int gsu = lz_act_gs(&L->moe_up_proj, latent);
        int nsu = (gsu > 0) ? latent / gsu : 0;
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->moe_lat_y + (size_t)tk * latent, latent, gsu,
                           s->xq + (size_t)tk * latent,
                           s->xqs + (size_t)tk * nsu);
        lz_matmul_xq_nt(s->xb2, s->moe_lat_y, s->xq, s->xqs,
                        &L->moe_up_proj, latent, hdim, nt);
    }

    /* PHASE 4 - the shared expert. _shared in kunmoe_modeling.py: same
       SwiGLU shape as the dense FFN this block replaces, just at
       shared_w width and reading hidden-space x directly (no latent).
       Every token goes through it, so all three of its matmuls batch -
       this is dense_ffn_step's shape, and it is written the same way. */
    if (shared_w > 0) {
        int gss = lz_act_gs(&L->moe_shared_gate, hdim);
        int nss = (gss > 0) ? hdim / gss : 0;
        int gsd = lz_act_gs(&L->moe_shared_down, shared_w);
        int nsd = (gsd > 0) ? shared_w / gsd : 0;
        int ssw_int = 0;    /* the pair landed in moe_h1_i16/moe_h3_i16 */

        /* gate and up both read xt with in_dim == hidden_size, so
           quantize ONCE and share - the same fix, and the same hazard,
           as the routed experts' w1/w3 above. Re-quantized rather than
           reusing phase 1's: the shared gate may carry a different
           activation group size than the down-projection did. */
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gss,
                           s->xq + (size_t)tk * hdim,
                           s->xqs + (size_t)tk * nss);
#if LZ_SWIGLU_I16
        /* Same all-or-nothing pair as the routed experts above, and the
           same unreachability argument for the second call refusing -
           ssw_may already asked lz_matmul_xq_i16_ok for both. `ssw_int`
           goes OUT and means one thing only: moe_h1_i16/moe_h3_i16 hold
           the two projections, so nothing may read moe_h1/moe_h3. */
        if (ssw_may &&
            lz_matmul_xq_nt_i16(s->moe_h1_i16, s->xq, s->xqs,
                                &L->moe_shared_gate, hdim, shared_w, nt,
                                LZ_SWIGLU_ES_P, 32767) &&
            lz_matmul_xq_nt_i16(s->moe_h3_i16, s->xq, s->xqs,
                                &L->moe_shared_up, hdim, shared_w, nt,
                                LZ_SWIGLU_ES_G, 32767))
            ssw_int = 1;
#endif /* LZ_SWIGLU_I16 */
        if (ssw_int) {
            /* Token-major with stride shared_w, so the whole batch is one
               contiguous run - the float pair below treats it the same
               way. */
            lz_swiglu_q15_i16(s->moe_h1_i16, s->moe_h1_i16, s->moe_h3_i16,
                              nt * shared_w, LZ_SWIGLU_ES_P, LZ_SWIGLU_ES_G,
                              LZ_SWIGLU_ES_O);
        } else {
        lz_matmul_xq_nt(s->moe_h1, s->xb, s->xq, s->xqs,
                        &L->moe_shared_gate, hdim, shared_w, nt);
        lz_matmul_xq_nt(s->moe_h3, s->xb, s->xq, s->xqs,
                        &L->moe_shared_up, hdim, shared_w, nt);
        for (i = 0; i < nt * shared_w; i++)
            s->moe_h1[i] = lz_silu(s->moe_h1[i]) * s->moe_h3[i];
        }
        for (tk = 0; tk < nt; tk++) {
            if (ssw_int)
                lz_quantize_q8_i16(s->moe_h1_i16 + (size_t)tk * shared_w,
                                   shared_w, gsd, pow2f(-LZ_SWIGLU_ES_O),
                                   s->xq + (size_t)tk * shared_w,
                                   s->xqs + (size_t)tk * nsd);
            else
                lz_quantize_q8(s->moe_h1 + (size_t)tk * shared_w, shared_w, gsd,
                               s->xq + (size_t)tk * shared_w,
                               s->xqs + (size_t)tk * nsd);
        }
        /* NULL, not moe_h1, on the int path: the down-projection is the
           one caller that could still read the float row, and ssw_may
           already established it will not (lz_matmul_xq_reads_float_row).
           Passing the stale buffer would leave that as a promise. */
        lz_matmul_xq_nt(s->moe_shared_out, ssw_int ? NULL : s->moe_h1,
                        s->xq, s->xqs,
                        &L->moe_shared_down, shared_w, hdim, nt);
        for (i = 0; i < nt * hdim; i++) s->xb2[i] += s->moe_shared_out[i];
    }
    /* No tap for the combined routed+shared output here: forward_chunk's
       unconditional LZ_TAP("ffn", l, s->xb2, dim) right after this
       function returns already captures it - same buffer, same point,
       for both this branch and the dense one, so a second tap here
       would only double the dump for zero extra signal. */
}

/* ------------------------------------------------------ dense FFN block */

/* Classic dense SwiGLU FFN (gate/up/down_proj): s->xb (nt x dim) ->
   s->xb2 (nt x dim), via s->hb/s->hb2 scratch. Extracted out of
   forward_chunk's per-layer loop so the MTP block's draft step
   (forward_mtp_draft_step) can share it exactly rather than carry a
   second, driftable copy - the MTP head's FFN is ALWAYS this dense
   form, never latent MoE, regardless of what the body does (model.h's
   LZMtp comment; model.c's model_walk filters MoE specs out of the MTP
   block for the same reason). `layer` is only for LZ_TAP (a no-op in
   production); the MTP call site passes LZ_MTP_CACHE_LAYER.

   `idim` is an explicit parameter, NOT read off c->intermediate_size
   internally: the MTP block's FFN width (c->mtp_intermediate_size) is
   an independent field (model.h's own comment on why
   reusing intermediate_size for both would silently couple two
   unrelated things on a checkpoint like kunmoe-v2). The body call site
   passes c->intermediate_size, the MTP call site passes
   c->mtp_intermediate_size - s->hb/s->hb2/s->xq/s->xqs are sized for
   the larger of the two by lz_state_alloc (see its own comment). */
static void dense_ffn_step(const LZModel *m, LZRunState *s,
                           const LZLayer *L, int layer, int nt, int idim) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size;
    int gsg = lz_act_gs(&L->gate_proj, dim);
    int nsg = (gsg > 0) ? dim / gsg : 0;
    int gsd = lz_act_gs(&L->down_proj, idim);
    int nsd = (gsd > 0) ? idim / gsd : 0;
    int i, tk;
    (void)layer;   /* only used inside LZ_TAP, which is a no-op in a normal build */

    if (c->use_subn) {
        /* SubLN per-projection input RMSNorms for the dense FFN: gate
           and up each normalize the UN-normalized residual with their own
           norm weight before their matmul (s->xb2 as scratch, free until
           the down_proj matmul below), down normalizes the intermediate
           activation with down_norm (s->hb2 as scratch, dead after the
           SwiGLU loop). */
        /* --norm-write int: see forward_kda's matching comment - same
           subn_norm_int scratch, same fallback conditions. */
        if (lz_norm_int() && lz_norm_can_fixed(dim) &&
            gsg > 0 && (dim % gsg) == 0) {
            for (tk = 0; tk < nt; tk++) {
                float deq;
                lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * dim,
                               lz_t_f32(&L->gate_norm, s->wscr), dim,
                               c->rms_norm_eps, &deq);
                lz_quantize_q8_tok_int(s->subn_norm_int, dim, deq,
                                       s->xq + (size_t)tk * dim,
                                       s->xqs + (size_t)tk * nsg);
            }
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * dim, s->xb + (size_t)tk * dim,
                           lz_t_f32(&L->gate_norm, s->wscr), dim, c->rms_norm_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8_tok(s->xb2 + (size_t)tk * dim, dim,
                                   s->xq + (size_t)tk * dim, s->xqs + (size_t)tk * nsg);
        }
        lz_matmul_xq_nt(s->hb, s->xb2, s->xq, s->xqs, &L->gate_proj, dim,
                        idim, nt);

        if (lz_norm_int() && lz_norm_can_fixed(dim) &&
            gsg > 0 && (dim % gsg) == 0) {
            for (tk = 0; tk < nt; tk++) {
                float deq;
                lz_rmsnorm_int(s->subn_norm_int, s->xb + (size_t)tk * dim,
                               lz_t_f32(&L->up_norm, s->wscr), dim,
                               c->rms_norm_eps, &deq);
                lz_quantize_q8_tok_int(s->subn_norm_int, dim, deq,
                                       s->xq + (size_t)tk * dim,
                                       s->xqs + (size_t)tk * nsg);
            }
        } else {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * dim, s->xb + (size_t)tk * dim,
                           lz_t_f32(&L->up_norm, s->wscr), dim, c->rms_norm_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8_tok(s->xb2 + (size_t)tk * dim, dim,
                                   s->xq + (size_t)tk * dim, s->xqs + (size_t)tk * nsg);
        }
        lz_matmul_xq_nt(s->hb2, s->xb2, s->xq, s->xqs, &L->up_proj, dim,
                        idim, nt);
    } else {
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * dim, dim, gsg,
                       s->xq + (size_t)tk * dim, s->xqs + (size_t)tk * nsg);
    lz_matmul_xq_nt(s->hb, s->xb, s->xq, s->xqs, &L->gate_proj, dim,
                    idim, nt);
    lz_matmul_xq_nt(s->hb2, s->xb, s->xq, s->xqs, &L->up_proj, dim,
                    idim, nt);
    }
    {
        /* SwiGLU: one lz_exp per element, scalar. NOT vectorized, and
           not worth vectorizing - measured at 0.7%% of decode
           (--profile), because the three matmuls around it dominate the
           ffn phase and those are already SIMD.

           lz_swiglu_q15_i16 is deliberately NOT wired here, and the
           reason is the checkpoints rather than the code: gate/up/down
           are f32 on every dense model in /e/LLM/models/ that loads
           (.prof/dtype_by_role.c on all 30), so there is no int16 exit
           for the producers to take. The one exception is the throwaway
           ARM fixture _armgate2 (q4_1/q4_1/q6_1), where
           lz_matmul_xq_i16_ok does answer 1 - so this is unreached, not
           unreachable, and wiring it means calibrating three exponents
           on a fixture no quality gate runs.

           Do not read "scalar" as "impossible here". This repo already
           replaces float with integer MMX where it pays and controls the
           error with a second plane (LZ_GDN_STATE_2PLANE, LZ_KV_2PLANE),
           and lz_exp is itself a table-plus-polynomial hack rather than
           libm. The reason this one stays scalar is its SIZE, not its
           shape. */
        LZ_PROF_BEG(_ta);
        for (i = 0; i < nt * idim; i++)
            s->hb[i] = lz_silu(s->hb[i]) * s->hb2[i];
        LZ_PROF_END(_ta, LZ_PROF_ACT);
    }
    LZ_TAP("fh", layer, s->hb, idim);
    if (c->use_subn) {
        /* down_proj input = RMSNorm_{down_norm}(SwiGLU output), then the
           SubLN block Hadamard, whose block is settled at load time
           (model.c resolve_hadamard). 0 = no Hadamard. */
        {
            int dblk = lz_hadamard_on() ? c->hadamard_down : 0;
            /* --norm-write int only applies when dblk<=1: the dblk>1
               branch feeds subn_fwht_quant, which needs the FLOAT row to
               run the Hadamard transform before it quantizes - there is
               no int-input twin of it, so that branch stays exactly as
               before regardless of the tier. */
            if (dblk <= 1 && lz_norm_int() && lz_norm_can_fixed(idim) &&
                gsd > 0 && (idim % gsd) == 0) {
                for (tk = 0; tk < nt; tk++) {
                    float deq;
                    lz_rmsnorm_int(s->subn_norm_int, s->hb + (size_t)tk * idim,
                                   lz_t_f32(&L->down_norm, s->wscr), idim,
                                   c->rms_norm_eps, &deq);
                    lz_quantize_q8_tok_int(s->subn_norm_int, idim, deq,
                                           s->xq + (size_t)tk * idim,
                                           s->xqs + (size_t)tk * nsd);
                }
            } else {
                int qgw = subn_scale_gs(gsd, idim);
                for (tk = 0; tk < nt; tk++)
                    lz_rmsnorm(s->hb2 + (size_t)tk * idim, s->hb + (size_t)tk * idim,
                               lz_t_f32(&L->down_norm, s->wscr), idim, c->rms_norm_eps);
                if (dblk > 1) {
                    for (tk = 0; tk < nt; tk++)
                        subn_fwht_quant(s->hb2 + (size_t)tk * idim, idim, dblk,
                                        qgw, s->xq + (size_t)tk * idim,
                                        s->xqs + (size_t)tk * nsd, s->fwht_scratch);
                    /* See forward_kda's twin: the float row is what an
                       F32 down_proj reads, and only then is this worth
                       running. */
                    if (lz_matmul_xq_reads_float_row(&L->down_proj))
                        for (tk = 0; tk < nt; tk++)
                            subn_rot_f32(s->hb2 + (size_t)tk * idim,
                                         idim, dblk);
                } else if (qgw > 0) {
                    for (tk = 0; tk < nt; tk++)
                        lz_quantize_q8(s->hb2 + (size_t)tk * idim, idim, qgw,
                                       s->xq + (size_t)tk * idim,
                                       s->xqs + (size_t)tk * nsd);
                } else {
                    for (tk = 0; tk < nt; tk++)
                        lz_quantize_q8_tok(s->hb2 + (size_t)tk * idim, idim,
                                           s->xq + (size_t)tk * idim,
                                           s->xqs + (size_t)tk * nsd);
                }
            }
        }
        lz_matmul_xq_nt(s->xb2, s->hb2, s->xq, s->xqs, &L->down_proj, idim,
                        dim, nt);
    } else {
    /* Quantization groups must match the target weight gs: when the
       intermediate dim is not a multiple of 32 (e.g. 1021) the weight's
       in-row gs falls back, and xqs tail entries must be filled with
       the same gs. */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->hb + (size_t)tk * idim, idim, gsd,
                       s->xq + (size_t)tk * idim, s->xqs + (size_t)tk * nsd);
    lz_matmul_xq_nt(s->xb2, s->hb, s->xq, s->xqs, &L->down_proj, idim,
                    dim, nt);
    }
}

/* --------------------------------------------------------- MTP draft step */

float *lz_mtp_draft_step(const LZModel *m, LZRunState *s, int next_token, int pos) {
    const LZModelConfig *c = &m->config;
    int dim = c->hidden_size, i;
    unsigned fpu;
    double t_us0;                              /* debug probe, see part 4 */

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
    lz_debug_us_draft += (long long)(lz_time_ms() * LZ_US_SCALE - t_us0);
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
        int nse = (gse > 0) ? dim / gse : 0;
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
        lz_rmsnorm(xl, xl, lz_t_f32(&m->final_norm, s->wscr), dim,
                   c->rms_norm_eps);
        LZ_TAP("fnorm", -1, xl, dim);
        /* tie_word_embeddings: output projection reuses the embedding
           directly; there is no standalone lm_head */
        lz_quantize_q8(xl, dim, lz_act_gs(&m->embed_tokens, dim),
                       s->xq, s->xqs);
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
    double t_us0 = lz_time_ms() * LZ_US_SCALE;  /* debug probe, see part 4 */
    float *r = forward_chunk(m, s, &token, 1, pos,
                             m->mtp ? s->mtp_chain : NULL, 0, NULL, 0);
    lz_debug_us_capture += (long long)(lz_time_ms() * LZ_US_SCALE - t_us0);
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
    double t_us0 = lz_time_ms() * LZ_US_SCALE;  /* debug probe, see part 4 */

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
    lz_debug_us_verify += (long long)(lz_time_ms() * LZ_US_SCALE - t_us0);
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
    double t_us0 = lz_time_ms() * LZ_US_SCALE;
    int r = lz_mtp_prefill(m, s, h_body_all, next_tokens, n, pos0);
    lz_debug_us_catchup += (long long)(lz_time_ms() * LZ_US_SCALE - t_us0);
    lz_debug_n_catchup++;
    return r;
}
