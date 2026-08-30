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
#include "compat.h"   /* lz_time_ms, for the LZ_PROF* macros */

/* The reference implementation hardcodes l2norm's eps to 1e-6, which
   is semantically different from config's rms_norm_eps - they only
   coincide numerically. Keep them separate so a future config change
   cannot silently drag this one along. */
#define LZ_L2NORM_EPS 1e-6f

/* Per-layer intermediate tap hook (in the header so the attn path, now in
   its own TU, sees it). No-op by default; a dump tool redefines it. */
#ifndef LZ_TAP
#define LZ_TAP(tag, li, ptr, n) ((void)0)
/* Set only when nobody supplied a tap, so a site can ask "is anyone
   reading intermediates here". An int-pipeline step that stops writing
   the float buffer a tap points at does not blank that tap - it leaves
   it reading the PREVIOUS contents, which is a plausible-looking wrong
   number rather than an obvious zero. Such a step guards on this and
   keeps the float path for a tap build. */
#define LZ_TAP_OFF 1
#endif /* LZ_TAP */

/* Per-layer residual hash (see model.c's lz_dbg_layer_hash). Integer-only,
   a no-op unless -DLZ_DBG_LAYER_HASH=1. Same shape as LZ_TAP so the same
   sites can carry it; the KDA recurrence is serial-in-t, so the hash must
   be read at the layer boundary, never inside the recurrence. */
#if LZ_DBG_LAYER_HASH
void lz_dbg_layer_hash(const char *tag, int li, const void *p, size_t nf);
#define LZ_DBGHASH(tag, li, ptr, n) lz_dbg_layer_hash(tag, li, ptr, n)
#else
#define LZ_DBGHASH(tag, li, ptr, n) ((void)0)
#endif /* LZ_DBG_LAYER_HASH */

/* ---- attn helpers (src/forward_attn.c), shared with the paths still in
   forward.c - the four-path split moved them out and left the signatures
   here. */

/* Walk the cache in time order from T0 to POS, carrying the row index
   alongside. Once the ring wraps, SLOT is no longer T, so advancing one
   without the other reads a different token's row - the two fixed
   kernels index by absolute t and that is exactly why they carry an
   LZ_ATTN_NORING guard. Every scoring and weighted-sum loop in
   forward_attn needs this same pair, so it is written once here rather
   than eight times there. */
#define LZ_KV_WALK(S, POS, T, SLOT, T0) \
    for ((T) = (T0), (SLOT) = kv_slot((S), (T0)); (T) <= (POS); \
         (T)++, (SLOT) = kv_slot_next((S), (SLOT)))



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

/* The conv's OUTPUT exponent for the q and k chains (int-pipeline 9.4),
 * and the knob that turns that exit on. Unlike LZ_CONV_ES this one is
 * not chosen on headroom alone: the consumer is an L2 normalisation,
 * which divides each vector by its own length, so a clip is the only
 * thing a too-large exponent costs and a too-small one costs relative
 * precision on the QUIETEST vector rather than on the largest.
 *
 * .prof/convqk_range.c, kmr20/zh, 603 tokens, 3.7M samples each chain:
 * max 26.3953 (q) and 13.2716 (k), zero clamps at 2^10 on both. What
 * that reading does NOT show, and what the probe reports for this
 * reason, is that the quietest of the 3,618 vectors peaks at 0.278 -
 * 94.8x below the global max - so it gets 285 int16 levels where the
 * loudest gets 27,029. That spread vanishes in the normalisation; the
 * 285 does not. Re-run the probe on any checkpoint before trusting
 * either number, and read lz_conv_o_clamped after, because the clamp
 * is silent everywhere else. */
#ifndef LZ_CONVO_ES
#define LZ_CONVO_ES 10
#endif /* LZ_CONVO_ES */
/* v GETS ITS OWN, and the first version of this did not - it reused
 * q and k's 10, which is choosing a constant on a distribution nobody
 * measured. v's max is 4.5376 against q's 26.3953, so 10 left it at
 * 4,645 of the 32,767 available and threw away 2.8 bits; 12 is the
 * largest with zero clamps (13 clips 16 samples). Its spread is also
 * far tighter - the quietest vector peaks 12.4x below the largest,
 * against q's 94.8x - which is why it can afford the full range. */
#ifndef LZ_CONVO_V_ES
#define LZ_CONVO_V_ES 12
#endif /* LZ_CONVO_V_ES */
#ifndef LZ_CONVO_I16
#define LZ_CONVO_I16 1
#endif /* LZ_CONVO_I16 */

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
 * The exponents are measured, not chosen, exactly like LZ_CONV_ES above
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

/* The decay gate's two transcendentals in the integer domain, on top of
   the int16 exit above. LZ_KGATE_I16 delivered `pre` as an integer and
   then spent nine semantic conversions per element getting from it to
   gt: one for the sigmoid's float coordinate, three inside
   sigmoid_q15_t taking that coordinate apart again, one for the Q15
   descale, and four in lz_exp_fixed (two i2f plus its magic floor and
   q8_round, neither of which emits a cvt but both of which are a full
   soft-float add on a machine with no FPU). sigmoid_q15_ti and lz_exp_t
   take the two folded integer coordinates instead and leave ONE - the
   int->float on lz_exp_t's own exit, which stands until gt itself stops
   being a float array.

   0 is the control arm: it keeps the float-coordinate chain, which is
   also the arm that runs whenever the scalar transcendental tier is off
   (lz_exp_t is lz_exp_fixed's twin, not lz_exp's) or a fold refuses. */
#ifndef LZ_KGATE_EXP_I
#define LZ_KGATE_EXP_I 1
#endif /* LZ_KGATE_EXP_I */

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
/* Debug counters defined in forward.c, read by cli_main.c and by the
   attn helpers in forward_attn.c. */
extern lz_i64 lz_debug_attn_skip;
extern lz_i64 lz_debug_n_kv_rot;
extern int   lz_debug_mtp_attn_window;
extern lz_i64 lz_debug_mtp_attn_rows;
extern lz_i64 lz_debug_vproj_i16;

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

extern int lz_prof_enable;
extern float lz_prof_us[LZ_PROF_N];

/* 1000.0 (ms -> us) is exact in any float width, so this cannot change
   what these debug counters measure - but `static const double`, not an
   inline literal, keeps every site below on the same footing as
   ops.c's LZ_EXP_LOG2E32 (that comment has the mechanism) rather than
   leaving an unexplained exception in tests/test_excess_precision.py. */
static const float LZ_US_SCALE = 1000.0f;

/* The start time is a LOCAL, not a file-static. A file-static would be
   shared across nested phases: the recurrence timer - which nests inside
   the linear one - would overwrite the outer start, so `linear` would
   report only the tail after the last inner call and the total would
   come out 25%% short. Phases that nest are reported INCLUSIVE, with
   the inner one also listed on its own. */
/* Declares AND assigns, so it can only appear where a declaration is
   legal. Under C89 that is the top of a block, and none of the six call
   sites is there - Visual C++ 4.0 turns each into an error and gcc only
   warns, which is why they survived. Splitting it into a declaration
   half and an assignment half is the fix, and it has to land in the
   same edit as the six declarations it needs at the call sites: the
   macro alone leaves every _t* undeclared. */
#define LZ_PROF_DECL(v) float v = 0.0f
#define LZ_PROF_BEG(v) v = lz_prof_enable ? lz_time_ms() : 0.0f
#define LZ_PROF_END(v, slot) do { if (lz_prof_enable) \
    lz_prof_us[slot] += (lz_time_ms() - (v)) * LZ_US_SCALE; } while (0)

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
    int64_t *attn_acc;          /* (attn_q_dim) int-domain weighted sum + output gate
                                   (fixed-tier attention int path, --attn-int) */
    float *attn_ss;             /* (attn_q_dim/32) per-32-group dequant scale for
                                   the same path (sscale = cmax * LZ_Q15_INV) */
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
       bandwidth ratio whose denominator was unmeasurable.

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

       Which side is expensive depends on the language. Decomposed with
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
       measurement that retired them, and the reason the keep-both-tiers
       rule does not protect them.

       One open question survives the format: whether the residual
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
    int32_t *ssm_sig;           /* (lin_value_dim) Q15 sigmoid of the gated
                                   norm, written beside the pre-silu product
                                   in the fixed tier (lz_quantize_q8_silu
                                   consumes it). int32, not int16: the Q15
                                   range reaches 32768 at the clamp. */
    float *qn, *kn;             /* (nt_cap, lin_k_head_dim) L2-norm scratch;
                                   the serial recurrence uses row 0 only */

    /* LZ_LT_KDA scratch. Separate q/k/v buffers rather than GDN's one
       fused in_proj_qkv: lz_matmul_xq_nt writes o[t*out_dim+i] with a
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
                                      already exponentiated (gt = exp(g), same
                                      convention lz_gdn_step's scalar gt uses) */

    /* latent MoE scratch (LZLayer.ffn_moe layers only). TWO WIDTHS, and
       the split is the routing boundary: what one weight matrix serves
       for every token in the chunk is nt_cap-wide and runs batched,
       what depends on WHICH expert a token picked is one token's worth.
       Only the routed experts are in the second group -
       lz_matmul_xq_nt's one-weight-load-serves-nt-tokens batching has
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
       (speculative-decode rollback without a
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

       Why this costs zero extra bandwidth, only memory (the argument
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

    /* Fixed conv tier (lz_conv_mode()). Allocated only when it is live,
       and `conv_fixed` says so - a NULL check alone would not, because
       the tier can be compiled out. Same shape and ring layout as
       conv_state, int16 instead of float.
         conv_mw          (n_linear, lin_conv_dim, conv_kernel) taps
         conv_sig_k1      (n_linear, lin_conv_dim) sigmoid_q15_t's folded
                           entry factor: oscale * LZ_SIG_STEP
         conv_sig_oscale2 (n_linear, lin_conv_dim) folded exit factor:
                           oscale / 32768 (oscale is 2^-(ew+es); neither
                           array stores oscale itself - it has no other
                           consumer, see lz_sig_q15_fold)
         conv_sig_e       (n_linear, lin_conv_dim) the same exponent as an
                           integer, which is what the epilogue's sigmoid
                           reads under LZ_CONV_SIG_I
       All are derived from the model and never change; they live here
       rather than on LZModel only to keep the loader untouched. */
    short *conv_state_q;
    short *conv_mw;
    /* KDA's q/k/v projections in int16 at LZ_CONV_ES, when the whole
       triple can take lz_matmul_xq_nt_i16's exit (int-pipeline
       milestone 3). Same shapes as kda_q/kda_k/kda_v, which stay
       allocated and carry the same vectors whenever it cannot -
       exactly one of the two is written per call. Allocated with the
       rest of the fixed conv tier and NULL without it. */
    short *kda_q_i16, *kda_k_i16;
    short *kda_v_i16;
    /* The CONV's own int16 exit, at LZ_CONVO_ES (int-pipeline 9.4).
       All three, and v is not the special case the first version of
       this comment claimed: its consumer does use it in a float
       expression - `(v - gt*u)*beta` - but so does q's and k's, and
       the answer is the same one, a convert at the consumer rather
       than two at the producer. kda_qc/kda_kc/kda_vc stay allocated
       and hold the same vectors whenever the int exit is off; exactly
       one of the two is written.

       "EXACTLY ONE IS WRITTEN" IS MEASURED NOW, not asserted. A counter
       on the float store in lz_causal_conv1d_step_fixed_o16 (the
       `if (!done)` arm) reads 0 on kmr20 against 11,114,496 int writes,
       and 16,671,744 against 0 on _armgate2 - one counter, both
       directions, so the zero means the store did not run rather than
       the counter not being reached.

       So the float trio is dead memory whenever the int exit fires, and
       it is kept anyway: 3 x nt x dim floats is 12,288 bytes on kmr20,
       0.33% of a 3.7 MB run state, while dropping it means allocating
       on a predicate that is not final until conv_fixed_build runs,
       several hundred lines after this allocation - and a NULL here is
       a crash on the fallback path, not a wrong number. Priced and
       declined, not overlooked.

       Two probes that cannot settle this, recorded so nobody repeats
       them: poisoning the buffers at allocation (everything here is
       written before it is read, so the poison never survives to a
       read - and the control, poisoning a live buffer, came back green
       too), and nulling the pointers (control also green). Both had no
       discriminating power. The write counter is the one that does. */
    short *kda_qc_i16, *kda_kc_i16, *kda_vc_i16;
    /* q16_0's four int16 exits (int-pipeline milestone 5), same
       one-of-the-two contract as the three above and the same NULL
       meaning. Each sits at its chain's LZ_*_ES exponent:
         bvec_i16        s->bvec        -> lz_sigmoid_i
         moe_logits_i16  s->moe_router_logits -> lz_moe_route
         kda_lat_i16     s->kda_gate_lat -> f_b_proj's activation quantize
         kda_gate_i16    s->kda_gate    -> the decay gate
       kda_lat_i32 is the widening scratch lz_quantize_q8_int's int32
       input needs (one token wide); kda_dtb_i16 is dt_bias in
       kda_gate_i16's own domain, built once by kda_gate_build, and is
       the field whose NULL says "the gate exit was refused". */
    short *bvec_i16;
    short *moe_logits_i16;
    short *kda_lat_i16;
    int32_t *kda_lat_i32;
    short *kda_gate_i16;
    short *kda_dtb_i16;
    /* The MoE latent's int16 exit (int-pipeline milestone 6), same
       one-of-the-two contract: moe_lat_i16 carries what moe_lat_x
       would have, at LZ_MLAT_ES.
         moe_lat_i16  s->moe_lat_x -> the routed experts' w1/w3 quantize
       moe_lat_q/_qs are that quantize's OUTPUT, and they are separate
       from s->xq/s->xqs because the w2 matmul in the same loop body
       quantizes moe_h1 into those - one token wide, since the expert
       loop is per token. They are allocated with (and used by) the
       float path too: the hoist out of the expert loop is what needs
       them, not the int exit. */
    short *moe_lat_i16;
    int8_t *moe_lat_q;
    float  *moe_lat_qs;
    /* The V projection's int16 exit (int-pipeline milestone 7), same
       one-of-the-two contract: vtmp_i16 carries what vtmp would have,
       at LZ_VPROJ_ES.
         vtmp_i16  s->vtmp -> the V cache's activation quantize
       Only the plain single-plane Q8 cache write can read it; the
       Hadamard rotation and the f32/q4 reference planes each take the
       float row apart element by element, so forward_attn refuses the
       exit when any of them is live. */
    short *vtmp_i16;
    /* The SwiGLU pair's int16 exits (int-pipeline milestone 8), same
       one-of-the-two contract: moe_h1_i16/moe_h3_i16 carry what
       moe_h1/moe_h3 would have, at LZ_SWIGLU_ES_P / LZ_SWIGLU_ES_G.
       Unlike the exits above these serve TWO call sites of different
       widths (the routed experts' w1/w3 at moe_intermediate_size and the
       shared expert's gate/up at moe_shared_width), so they are sized
       like moe_h1/moe_h3 themselves - the max of the two, nt_cap wide.
       lz_swiglu_q15_i16 writes its output back over moe_h1_i16, which is
       then the activation lz_quantize_q8_i16 reads. */
    short *moe_h1_i16, *moe_h3_i16;
    float *conv_sig_k1;
    float *conv_sig_oscale2;
    signed char *conv_sig_e;    /* (n_linear, lin_conv_dim) ew[c] + LZ_CONV_ES,
                                   the integer the two floats above encode */
    float  conv_sig_k2;         /* lz_sig_q15_t_offset(), same for every channel */
    float  conv_in_scale;
    int    conv_bound;
    int    conv_fixed;
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

    /* SubLN Hadamard fixed-point scratch (use_subn only): 2*qcap int32.
       Source half holds the row converted to fixed point, destination
       half the unnormalized lz_fwht_i32 output (restrict forbids one
       buffer for both). */
    int32_t *fwht_scratch;

    /* SubLN plain-norm int output scratch (the norms tier's int output,
       use_subn only): qcap int32, one token wide. lz_rmsnorm_int writes here,
       lz_quantize_q8_tok_int reads it immediately after - single-token
       like fwht_scratch, not nt-wide, since each site's norm+quantize
       pair for one tk completes before the next tk starts. */
    int32_t *subn_norm_int;

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
       exactly the kind of knob that has to stay sweepable rather than
       baked in: it trades mixing quality against nothing on this
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
    lz_i64 bytes_alloc;

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

/* Attn helper signatures (src/forward_attn.c), shared with the paths
   still in forward.c. Declared here rather than at the top because they
   take LZRunState, whose typedef sits above. */
void wsum_q8_row(float *dst, const int8_t *vt, const float *vts, float a, int hd);
float score_q8_row(const float *qhh, const int8_t *kt, const float *kts, int hd);
int kv_slot_next(const LZRunState *s, int slot);
int kv_slot(const LZRunState *s, int t);
int scale_groups(int n, int gs);
int ring_slot_next(int slot, int depth);
void attn_wsum_plane(LZRunState *s, float *dst, const int8_t *vplane,
                     const float *vs, int hd, int kvd, int kvh, int pos,
                     int walk_t0, int noring, int dead0, int dead1);
void rot_head(float *v, int hd, int n);
void forward_attn(const LZModel *m, LZRunState *s,
                  const LZLayer *L, int layer, int pos0, int nt);
/* One projection with an int16 exit at the consumer's own exponent.
   EIGHT call sites in three files, which is why it is not called
   kda_i16_proj any more: forward_kda.c has four, forward_moe.c three
   (the router logits and the latent down-projection) and forward_attn.c
   one - the v_proj inside the q/k/v triple, NOT the o-projection tail.
   It lives in forward_kda.c because that is
   where it was written; the name no longer says it belongs to KDA,
   because half its callers do not. */
void proj_i16(LZRunState *s, int may_int, int *wrote_int,
                  float *of, short *oi,
                  const float *x, const LZTensor *w,
                  int in_dim, int out_dim, int nt,
                  int target_e, int bound);
/* MoE path (src/forward_moe.c): forward_moe is called from lz_forward; the
   two router knobs and the latent counters are defined in forward.c. */
void forward_moe(const LZModel *m, LZRunState *s,
                 const LZLayer *L, int layer, int nt);
extern int   lz_moe_topk;
extern float lz_moe_tau;
extern lz_i64 lz_debug_mlat_i16;
extern lz_i64 lz_debug_mlat_quant;
/* KDA-path counters, read by cli_main.c and written by forward_kda.c. */
extern lz_i64 lz_debug_bvec_i16;
extern lz_i64 lz_debug_klat_i16;
extern lz_i64 lz_debug_kgate_exp_i;
extern lz_i64 lz_debug_kgate_fold_no;
extern int   lz_debug_kgate_exp_s;
extern int   lz_debug_kgate_sig_slo;
extern int   lz_debug_kgate_sig_shi;
/* SSM path (src/forward_ssm.c): forward_ssm is called from lz_forward. */
void forward_ssm(const LZModel *m, LZRunState *s,
                 const LZLayer *L, int layer, int nt,
                 int advance_ring, int ring_base);
/* KDA + dense-FFN paths (src/forward_kda.c), called from lz_forward /
   lz_mtp_draft_step. */
void forward_kda(const LZModel *m, LZRunState *s,
                 const LZLayer *L, int layer, int nt,
                 int advance_ring, int ring_base);
void dense_ffn_step(const LZModel *m, LZRunState *s,
                    const LZLayer *L, int layer, int nt, int idim);

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
    /* int16 twin of `conv`, filled instead of it when the fixed conv
       tier is live. Additive: LZStateCkpt never leaves memory, so there
       is no stored layout to stay compatible with. */
    short  *conv_q;
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

   This is a sizing hint, not a contract lz_generate_resume enforces on
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
   returns the LAST token's logits (intermediate logits are not computed -
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
