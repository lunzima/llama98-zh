#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "compat.h"    /* lz_time_ms, for the debug probes below only */
#include "err.h"
#include "ops.h"

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
#endif

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
   "count bytes but not compute" trap CLAUDE.md's own MTP section
   already reversed on once). Running total of MTP-only score-loop rows
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

   Integer microseconds, not float seconds: iron law six clause 3 bans
   %f inside the PC=24 region, and these are read by cli_main.c which
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
   cli_main prints them (iron law 1 forbids console output below it).

   Off by default because lz_time_ms() is a syscall-grade clock read and
   the inner phases here run 24 times per token - the measurement would
   otherwise cost more than some of the things it measures. */
int lz_prof_enable = 0;
double lz_prof_us[LZ_PROF_N];

/* The start time is a LOCAL, not a file-static. A file-static would be
   shared across nested phases: the recurrence timer - which nests inside
   the linear one - would overwrite the outer start, so `linear` would
   report only the tail after the last inner call and the total would
   come out 25%% short. Phases that nest are reported INCLUSIVE, with
   the inner one also listed on its own. */
#define LZ_PROF_BEG(v) double v = lz_prof_enable ? lz_time_ms() : 0.0
#define LZ_PROF_END(v, slot) do { if (lz_prof_enable) \
    lz_prof_us[slot] += (lz_time_ms() - (v)) * 1000.0; } while (0)

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
   where it wins. Iron law nine keeps contested knobs because the target
   MACHINE family flips them; a KV format's quality-per-byte is a
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
    s->wsum_cbuf = (float *)xcalloc((size_t)seq_len, sizeof(float), &ok, &s->bytes_alloc);
    s->wsum_cq   = (int16_t *)xcalloc((size_t)seq_len, sizeof(int16_t), &ok, &s->bytes_alloc);
#endif
    s->attn_out = (float *)xcalloc((size_t)nt * c->attn_q_dim, sizeof(float), &ok, &s->bytes_alloc);
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
#endif
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
#endif
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
#endif

    s->qkv    = (float *)xcalloc((size_t)nt * c->lin_conv_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->qkv_c  = (float *)xcalloc((size_t)nt * c->lin_conv_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->zbuf   = (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
    s->avec   = (float *)xcalloc((size_t)nt * c->lin_n_v_heads, sizeof(float), &ok, &s->bytes_alloc);
    s->bvec   = (float *)xcalloc((size_t)nt * c->lin_n_v_heads, sizeof(float), &ok, &s->bytes_alloc);
    s->ssm_out= (float *)xcalloc((size_t)nt * c->lin_value_dim, sizeof(float), &ok, &s->bytes_alloc);
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

       TWO WIDTHS, and the split is exactly the routing boundary. The
       router, the latent down/up projections and the shared expert read
       one weight stream for every token in the chunk, so they run
       batched and their buffers are nt-wide like every other block in
       this file. The ROUTED experts cannot: which expert a token wants
       is a property of that token, so moe_h2 and the per-expert
       selection stay one token's worth.
       At LZ_BATCH_MAX=8 and this model's shapes the widening costs
       ~78 KB of state - measured against the ~13% of prefill the
       per-token weight re-streaming was costing (see forward_moe). */
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
#endif
    s->ssm_state_s = (float *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_n_v_heads *
        c->lin_k_head_dim * c->lin_v_head_dim / 32,
        sizeof(float), &ok, &s->bytes_alloc);
    s->conv_state = (float *)xcalloc(
        (size_t)s->ssm_ring_depth * c->n_linear_layers * c->lin_conv_dim *
        (c->conv_kernel - 1), sizeof(float), &ok, &s->bytes_alloc);
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

    return 0;
}

void lz_state_free(LZRunState *s) {
    if (!s) return;
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->hb2);
    free(s->logits);
    free(s->qg); free(s->qh); free(s->att); free(s->attn_out);
#if (LZ_ATTN_FIXED & 2)
    free(s->wsum_cbuf); free(s->wsum_cq);
#endif
    free(s->ktmp); free(s->vtmp);
    free(s->kq8); free(s->vq8); free(s->ksq); free(s->vsq);
    free(s->kf32); free(s->vf32);
    free(s->k4); free(s->v4); free(s->ks4); free(s->vs4);
#if LZ_KV_2PLANE
    free(s->kq8_lo); free(s->vq8_lo); free(s->att_lo); free(s->wsum_lo);
#endif
    free(s->qkv); free(s->qkv_c); free(s->zbuf); free(s->avec); free(s->bvec);
    free(s->ssm_out); free(s->qn); free(s->kn);
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
#endif
    free(s->conv_state);
    free(s->xq); free(s->xqs); free(s->wscr);
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
#endif
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
#endif
    memset(s->ssm_state_s, 0, (size_t)s->ssm_ring_depth * c->n_linear_layers *
           c->lin_n_v_heads * c->lin_k_head_dim * c->lin_v_head_dim / 32 *
           sizeof(float));
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
#endif
    ck->ssm_s = (float *)malloc(n_s * sizeof(float));
    ck->conv  = (float *)malloc(n_conv * sizeof(float));
    if (!ck->ssm_q8 || !ck->ssm_s || !ck->conv
#if LZ_GDN_STATE_2PLANE
        || !ck->ssm_q8_lo
#endif
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
#endif
    free(ck->ssm_s);
    free(ck->conv);
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
#endif
    memcpy(ck->ssm_s, s->ssm_state_s + o_s, n_s * sizeof(float));
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
#endif
    memcpy(s->ssm_state_s, ck->ssm_s, n_s * sizeof(float));
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
#endif
    /* (H q) . (H k) = kv_rot_k * (q . k) because lz_fwht is the
       unnormalized Hadamard (see its comment in ops.h). Fold that factor
       out here rather than normalizing the transform: kv_rot_k is a power
       of two, so this division is exact and identical on both compilers,
       whereas a 1/sqrt(n) constant would trip iron law 6. */
    float scale = s->attn_scale;
    int hdim = c->hidden_size, kvd = c->attn_kv_dim, qd = c->attn_q_dim;
    int qgd = c->attn_qgate_dim;
    int gsa = lz_act_gs(&L->q_proj, hdim);
    int nsa = (gsa > 0) ? hdim / gsa : 0;
    int h, d, t, tk;

    if (s->kv_rot_k) scale /= (float)s->kv_rot_k;

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
    lz_matmul_q8_nt(s->qg, s->xb, s->xq, s->xqs, &L->q_proj, hdim, qgd, nt);
    lz_matmul_q8_nt(s->ktmp, s->xb, s->xq, s->xqs, &L->k_proj, hdim, kvd, nt);
    lz_matmul_q8_nt(s->vtmp, s->xb, s->xq, s->xqs, &L->v_proj, hdim, kvd, nt);

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
           the low plane cannot perturb the high one. */
        if (kc) lz_gdn_quantize_2p(ktt, kvd, 32, kc + (size_t)kv_slot(s, pos) * kvd,
                                   kc_lo + (size_t)kv_slot(s, pos) * kvd,
                                   ks + (size_t)kv_slot(s, pos) * kvd / 32);
        if (vc) lz_gdn_quantize_2p(vtt, kvd, 32, vc + (size_t)kv_slot(s, pos) * kvd,
                                   vc_lo + (size_t)kv_slot(s, pos) * kvd,
                                   vs + (size_t)kv_slot(s, pos) * kvd / 32);
#else
        if (kc) lz_quantize_q8(ktt, kvd, 32, kc + (size_t)kv_slot(s, pos) * kvd,
                               ks + (size_t)kv_slot(s, pos) * kvd / 32);
        if (vc) lz_quantize_q8(vtt, kvd, 32, vc + (size_t)kv_slot(s, pos) * kvd,
                               vs + (size_t)kv_slot(s, pos) * kvd / 32);
#endif
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
               float loops rather than being silently truncated. Note both
               halves take the SAME branch - scoring and weighted sum are
               independent kernels, but splitting them here would mean two
               tiers to verify instead of one. */
            if (hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                lz_attn_score_q8(s->att, qhh, hd,
                                 kc + (size_t)kvh * hd,
                                 ks + (size_t)(kvh * hd) / 32,
                                 kvd, pos, scale);
            } else
#endif
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
               implementations to keep bit-identical (iron law 2).
               Folded BEFORE softmax: the low plane corrects the logit,
               not the probability. */
            {
                int tt;
#if (LZ_ATTN_FIXED & 1)
                if (hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                    lz_attn_score_q8(s->att_lo, qhh, hd,
                                     kc_lo + (size_t)kvh * hd,
                                     ks + (size_t)(kvh * hd) / 32,
                                     kvd, pos, scale);
                } else
#endif
                for (tt = 0; tt <= pos; tt++) {
                    const int8_t *kt = kc_lo + (size_t)kv_slot(s, tt) * kvd + (size_t)kvh * hd;
                    const float *kts = ks + (size_t)kv_slot(s, tt) * kvd / 32 +
                                       (size_t)(kvh * hd) / 32;
                    float sum = 0.0f;
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
#endif
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
                    for (d = 0; d < hd; d++) out[d] += a * vt[d];
                }
            } else
            {
#if (LZ_ATTN_FIXED & 2)
            if (hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                lz_attn_wsum_q8(out, s->att, hd,
                                vc + (size_t)kvh * hd,
                                vs + (size_t)(kvh * hd) / 32,
                                kvd, pos, s->wsum_cbuf, s->wsum_cq);
            } else {
#endif
            memset(out, 0, (size_t)hd * sizeof(float));
            for (t = win_t0; t <= pos; t++) {
                if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                const int8_t *vt = vc + (size_t)kv_slot(s, t) * kvd + (size_t)kvh * hd;
                const float *vts = vs + (size_t)kv_slot(s, t) * kvd / 32 +
                                   (size_t)(kvh * hd) / 32;
                float a = s->att[t];
                for (d = 0; d < hd; d++) out[d] += a * (float)vt[d] * vts[d / 32];
            }
#if (LZ_ATTN_FIXED & 2)
            }
#endif
#if LZ_KV_2PLANE
            /* Same argument as the scoring pass: the weighted sum is
               linear in v. */
            {
#if (LZ_ATTN_FIXED & 2)
                if (hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                    lz_attn_wsum_q8(s->wsum_lo, s->att, hd,
                                    vc_lo + (size_t)kvh * hd,
                                    vs + (size_t)(kvh * hd) / 32,
                                    kvd, pos, s->wsum_cbuf, s->wsum_cq);
                } else {
#endif
                memset(s->wsum_lo, 0, (size_t)hd * sizeof(float));
                for (t = 0; t <= pos; t++) {
                    if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
                    const int8_t *vt = vc_lo + (size_t)kv_slot(s, t) * kvd + (size_t)kvh * hd;
                    const float *vts = vs + (size_t)kv_slot(s, t) * kvd / 32 +
                                       (size_t)(kvh * hd) / 32;
                    float a = s->att[t];
                    for (d = 0; d < hd; d++)
                        s->wsum_lo[d] += a * (float)vt[d] * vts[d / 32];
                }
#if (LZ_ATTN_FIXED & 2)
                }
#endif
                for (d = 0; d < hd; d++)
                    out[d] += s->wsum_lo[d] * (1.0f / LZ_GDN_LO_SCALE);
            }
#endif
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

        /* output gate: same source as q, take the second half of each head */
        for (h = 0; h < nh; h++) {
            const float *gate = qgt + (size_t)h * 2 * hd + hd;
            float *out = aot + (size_t)h * hd;
            for (d = 0; d < hd; d++) out[d] *= lz_sigmoid(gate[d]);
        }
    }
#undef LZ_ATTN_DEAD
    LZ_TAP("ax", layer, s->attn_out, qd);

    /* o_proj input is the attention output (attn_q_dim); quantize once more */
    {
        int gso = lz_act_gs(&L->o_proj, qd);
        int nso = (gso > 0) ? qd / gso : 0;
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->attn_out + (size_t)tk * qd, qd, gso,
                           s->xq + (size_t)tk * qd, s->xqs + (size_t)tk * nso);
        lz_matmul_q8_nt(s->xb2, s->attn_out, s->xq, s->xqs, &L->o_proj, qd,
                        hdim, nt);
    }
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
    lz_matmul_q8_nt(s->qkv, s->xb, s->xq, s->xqs, &L->in_proj_qkv, hdim,
                    cdim, nt);
    lz_matmul_q8_nt(s->zbuf, s->xb, s->xq, s->xqs, &L->in_proj_z, hdim,
                    vdim, nt);
    lz_matmul_q8_nt(s->avec, s->xb, s->xq, s->xqs, &L->in_proj_a, hdim, nv, nt);
    lz_matmul_q8_nt(s->bvec, s->xb, s->xq, s->xqs, &L->in_proj_b, hdim, nv, nt);

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
#endif
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
#endif
                        ss_in, ss_out, s->qn, s->kn, v_t, gt, beta, kd, vd);
            LZ_PROF_END(_tr, LZ_PROF_REC);
        }
    }

    /* This is the plain-weight gated RMSNorm, not the (1+w) kind used
       by the outer layers */
    for (tk = 0; tk < nt; tk++)
        for (h = 0; h < nv; h++)
            lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                             s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                             s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                             lz_t_f32(&L->ssm_norm, s->wscr), vd,
                             c->rms_norm_eps);

    LZ_TAP("sgdn", layer, s->ssm_out, vdim);
    {
        int gso = lz_act_gs(&L->out_proj, vdim);
        int nso = (gso > 0) ? vdim / gso : 0;
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                           s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        lz_matmul_q8_nt(s->xb2, s->ssm_out, s->xq, s->xqs, &L->out_proj, vdim,
                        hdim, nt);
    }
}

/* ------------------------------------------------------ KDA (LZ_LT_KDA) block */

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

    /* Six hidden-space projections share one activation quantization,
       the same trick forward_ssm uses for its four (all have
       in_dim == hidden_size, so lz_act_gs agrees regardless of format). */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsa,
                       s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
    lz_matmul_q8_nt(s->kda_q, s->xb, s->xq, s->xqs, &L->kda_q_proj, hdim, kdim, nt);
    lz_matmul_q8_nt(s->kda_k, s->xb, s->xq, s->xqs, &L->kda_k_proj, hdim, kdim, nt);
    lz_matmul_q8_nt(s->kda_v, s->xb, s->xq, s->xqs, &L->kda_v_proj, hdim, vdim, nt);
    lz_matmul_q8_nt(s->kda_gate_lat, s->xb, s->xq, s->xqs, &L->kda_f_a_proj,
                    hdim, gate_rank, nt);
    lz_matmul_q8_nt(s->bvec, s->xb, s->xq, s->xqs, &L->kda_b_proj, hdim, nv, nt);
    lz_matmul_q8_nt(s->zbuf, s->xb, s->xq, s->xqs, &L->kda_g_proj, hdim, vdim, nt);
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
    LZ_TAP("kqc", layer, s->kda_qc, kdim);
    LZ_TAP("kkc", layer, s->kda_kc, kdim);
    LZ_TAP("kvc", layer, s->kda_vc, vdim);

    /* f_b_proj: kda_gate_rank -> nvk, a DIFFERENT in_dim than hidden, so
       it needs its own quantization - lz_matmul_w does that internally,
       one token at a time (same reasoning forward_moe's per-expert
       matmuls use). */
    for (tk = 0; tk < nt; tk++)
        lz_matmul_w(s->kda_gate + (size_t)tk * nvk,
                   s->kda_gate_lat + (size_t)tk * gate_rank,
                   &L->kda_f_b_proj, gate_rank, nvk, s->xq, s->xqs);

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
#endif
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

            beta = lz_sigmoid(s->bvec[(size_t)tk * nv + h]);
            LZ_PROF_BEG(_tr);
            lz_kda_step(out, sq_in, sq_out,
#if LZ_GDN_STATE_2PLANE
                        sq2_in, sq2_out,
#endif
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
       stay silu on this engine. */
    for (tk = 0; tk < nt; tk++)
        for (h = 0; h < nv; h++)
            lz_rmsnorm_gated(s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                             s->ssm_out + (size_t)tk * vdim + (size_t)h * vd,
                             s->zbuf + (size_t)tk * vdim + (size_t)h * vd,
                             lz_t_f32(&L->kda_o_norm, s->wscr), vd,
                             c->rms_norm_eps);
    LZ_TAP("sgdn", layer, s->ssm_out, vdim);
    {
        int gso = lz_act_gs(&L->kda_o_proj, vdim);
        int nso = (gso > 0) ? vdim / gso : 0;
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->ssm_out + (size_t)tk * vdim, vdim, gso,
                           s->xq + (size_t)tk * vdim, s->xqs + (size_t)tk * nso);
        lz_matmul_q8_nt(s->xb2, s->ssm_out, s->xq, s->xqs, &L->kda_o_proj, vdim,
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
   they batch through lz_matmul_q8_nt exactly like dense_ffn_step. Only
   the routed experts cannot - two tokens in the same chunk may pick
   different experts entirely, so there is no shared weight stream.

   WHY NOT GROUP THE TOKENS BY EXPERT. That is what llama.cpp's
   ggml_compute_forward_mul_mat_id does (a plain per-expert row bucket,
   no padding) and what vLLM/SGLang's CPU kernels do with
   moe_align_block_size. Their scale is not ours: SGLang's CPU int8 MoE
   pads each expert's rows to BLOCK_M = 2 * TILE_M = 32 and drops the
   blocked path entirely below M = 5, and llama.cpp's tuning assumes
   batches in the thousands. LZ_BATCH_MAX is 8. At 8 tokens, top-2 of
   16 experts, the expected number of DISTINCT experts in a chunk is
   16*(1-(15/16)^16) ~ 10.3 of 16 routed pairs, so grouping would save
   about a third of the expert weight streams - and llama.cpp's own
   measurement is that prompt-phase routing is flat (decode's is
   skewed), which is the worst case for it. Measured against the
   --moe-topk decomposition below, the ceiling was ~9% of prefill, for
   a change that has to reorder the float accumulation in the kk loop
   to get it. The batching above was the larger half and costs no
   reordering at all.

   HOW THE SPLIT WAS MEASURED, since a ratio needs a denominator
   (iron law 3): ffn(topk) is linear in topk, so sweeping --moe-topk
   1/2/4 separates the two halves by extrapolation. On this machine and
   an 832-token prompt it fit ffn = 856,505 + 787,201*topk us exactly,
   i.e. at the trained topk=2 the routed experts are 64.8% of the ffn
   phase and everything batched here is the other 35.2%.

   nt=1 (the generation loop) computes exactly what it did before: the
   t-loop in lz_matmul_q8_nt degrades to one iteration, which that
   function's header states as a hard gate.

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
       question can be measured (iron law 3 wants this kind of parameter
       swept, not baked), and the scratch buffers are sized from the
       config, so the override is clamped to them. */
    if (lz_moe_topk > 0 && lz_moe_topk <= ne)
        topk = lz_moe_topk;
    int gsh = lz_act_gs(&L->moe_down_proj, hdim);
    int nsh = (gsh > 0) ? hdim / gsh : 0;
    int tk, kk, vv, i;
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
    lz_matmul_q8_nt(s->moe_router_logits, s->xb, s->xq, s->xqs,
                    &L->moe_gate_w, hdim, ne, nt);
    lz_matmul_q8_nt(s->moe_lat_x, s->xb, s->xq, s->xqs,
                    &L->moe_down_proj, hdim, latent, nt);
    /* Slot 0 is token 0's, so these tap what the per-token version
       taped under `if (tk == 0)`. */
    LZ_TAP("mrt", layer, s->moe_router_logits, ne);
    LZ_TAP("mlat", layer, s->moe_lat_x, latent);

    /* PHASE 2 - the routed experts, and the ONE part that stays per
       token: which expert a token wants is a property of that token, so
       there is no shared weight stream to batch over. Grouping the
       tokens by expert instead was measured and rejected - see the
       block comment above this function. */
    for (tk = 0; tk < nt; tk++) {
        float *lat_x = s->moe_lat_x + (size_t)tk * latent;
        float *lat_y = s->moe_lat_y + (size_t)tk * latent;

        lz_moe_route(s->moe_router_logits + (size_t)tk * ne,
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
            /* w1 and w3 both read lat_x with the same in_dim (latent),
               so quantize it ONCE and share, instead of letting each
               lz_matmul_w re-quantize the identical vector. This must be
               INSIDE the kk loop, not hoisted out: the w2 matmul just
               below quantizes moe_h1 into the same xq/xqs scratch, so a
               hoisted quantization would be overwritten before the next
               expert's w1/w3 read it (a real, silent corruption). The
               day w1/w3 carry different weight gs this also keeps them
               reading the SAME quantization, the mixed-precision hazard
               forward_attn's q/k/v comment names.
               Slot 0 of the (now nt-wide) scratch throughout: this whole
               loop is one token's work and phases 1 and 3 are done with
               their copies by the time it runs. */
            {
                int gsl = lz_act_gs(&L->moe_expert_w1[ei], latent);
                lz_quantize_q8(lat_x, latent, gsl, s->xq, s->xqs);
            }
            lz_matmul_q8(s->moe_h1, lat_x, s->xq, s->xqs,
                        &L->moe_expert_w1[ei], latent, inter);
            lz_matmul_q8(s->moe_h3, lat_x, s->xq, s->xqs,
                        &L->moe_expert_w3[ei], latent, inter);
            for (vv = 0; vv < inter; vv++)
                s->moe_h1[vv] = lz_silu(s->moe_h1[vv]) * s->moe_h3[vv];
            lz_matmul_w(s->moe_h2, s->moe_h1, &L->moe_expert_w2[ei],
                       inter, latent, s->xq, s->xqs);
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
        lz_matmul_q8_nt(s->xb2, s->moe_lat_y, s->xq, s->xqs,
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

        /* gate and up both read xt with in_dim == hidden_size, so
           quantize ONCE and share - the same fix, and the same hazard,
           as the routed experts' w1/w3 above. Re-quantized rather than
           reusing phase 1's: the shared gate may carry a different
           activation group size than the down-projection did. */
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gss,
                           s->xq + (size_t)tk * hdim,
                           s->xqs + (size_t)tk * nss);
        lz_matmul_q8_nt(s->moe_h1, s->xb, s->xq, s->xqs,
                        &L->moe_shared_gate, hdim, shared_w, nt);
        lz_matmul_q8_nt(s->moe_h3, s->xb, s->xq, s->xqs,
                        &L->moe_shared_up, hdim, shared_w, nt);
        for (i = 0; i < nt * shared_w; i++)
            s->moe_h1[i] = lz_silu(s->moe_h1[i]) * s->moe_h3[i];
        for (tk = 0; tk < nt; tk++)
            lz_quantize_q8(s->moe_h1 + (size_t)tk * shared_w, shared_w, gsd,
                           s->xq + (size_t)tk * shared_w,
                           s->xqs + (size_t)tk * nsd);
        lz_matmul_q8_nt(s->moe_shared_out, s->moe_h1, s->xq, s->xqs,
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

    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * dim, dim, gsg,
                       s->xq + (size_t)tk * dim, s->xqs + (size_t)tk * nsg);
    lz_matmul_q8_nt(s->hb, s->xb, s->xq, s->xqs, &L->gate_proj, dim,
                    idim, nt);
    lz_matmul_q8_nt(s->hb2, s->xb, s->xq, s->xqs, &L->up_proj, dim,
                    idim, nt);
    {
        /* SwiGLU: one lz_exp per element, scalar. NOT vectorized, and
           not worth vectorizing - measured at 0.7%% of decode
           (--profile), because the three matmuls around it dominate the
           ffn phase and those are already SIMD.

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
    /* Quantization groups must match the target weight gs: when the
       intermediate dim is not a multiple of 32 (e.g. 1021) the weight's
       in-row gs falls back, and xqs tail entries must be filled with
       the same gs. */
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->hb + (size_t)tk * idim, idim, gsd,
                       s->xq + (size_t)tk * idim, s->xqs + (size_t)tk * nsd);
    lz_matmul_q8_nt(s->xb2, s->hb, s->xq, s->xqs, &L->down_proj, idim,
                    dim, nt);
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

    t_us0 = lz_time_ms() * 1000.0;
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
    lz_matmul_q8(s->mtp_draft_logits, s->mtp_chain, s->xq, s->xqs,
                &m->embed_tokens, dim, c->vocab_size);

    lz_fpu_float_end(fpu);
    lz_debug_us_draft += (long long)(lz_time_ms() * 1000.0 - t_us0);
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

        for (tk = 0; tk < nt; tk++)
            lz_rmsnorm(s->xb + (size_t)tk * dim, s->x + (size_t)tk * dim,
                       lz_t_f32(&L->input_layernorm, s->wscr), dim,
                       c->rms_norm_eps);
        LZ_TAP("an", l, s->xb, dim);
        LZ_PROF_BEG(_tp);
        if (L->type == LZ_LT_FULL)      forward_attn(m, s, L, l, pos0, nt);
        else if (L->type == LZ_LT_KDA)  forward_kda(m, s, L, l, nt, all_logits, ring_base);
        else                             forward_ssm(m, s, L, l, nt, all_logits, ring_base);
        LZ_PROF_END(_tp, L->type == LZ_LT_FULL ? LZ_PROF_ATTN : LZ_PROF_LIN);
        LZ_TAP("blk", l, s->xb2, dim);
        for (i = 0; i < nt * dim; i++) s->x[i] += s->xb2[i];
        LZ_TAP("res", l, s->x, dim);

        for (tk = 0; tk < nt; tk++)
            lz_rmsnorm(s->xb + (size_t)tk * dim, s->x + (size_t)tk * dim,
                       lz_t_f32(&L->post_attention_layernorm, s->wscr), dim,
                       c->rms_norm_eps);
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
        lz_matmul_q8_nt(logits_out, s->x, s->xq, s->xqs, &m->embed_tokens,
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
        lz_matmul_q8(s->logits, xl, s->xq, s->xqs, &m->embed_tokens, dim,
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
    double t_us0 = lz_time_ms() * 1000.0;      /* debug probe, see part 4 */
    float *r = forward_chunk(m, s, &token, 1, pos,
                             m->mtp ? s->mtp_chain : NULL, 0, NULL, 0);
    lz_debug_us_capture += (long long)(lz_time_ms() * 1000.0 - t_us0);
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
    double t_us0 = lz_time_ms() * 1000.0;      /* debug probe, see part 4 */

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
    lz_debug_us_verify += (long long)(lz_time_ms() * 1000.0 - t_us0);
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
    double t_us0 = lz_time_ms() * 1000.0;
    int r = lz_mtp_prefill(m, s, h_body_all, next_tokens, n, pos0);
    lz_debug_us_catchup += (long long)(lz_time_ms() * 1000.0 - t_us0);
    lz_debug_n_catchup++;
    return r;
}
