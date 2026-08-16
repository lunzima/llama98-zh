#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "err.h"
#include "generate.h"
#include "llama_zh.h"
#include "sampler.h"

/* Emission that never hands out half a character.

   Token bytes are byte-level BPE and the stop-string holdback slices the
   tail by BYTE COUNT, so both can cut through the middle of a multi-byte
   sequence. Measured on s1v3 generating Chinese: with no stop strings 0
   of 64 chunks ended mid-codepoint, but with one stop string active
   32 of 55 did - 58%. A terminal reassembles the stream and never shows
   it; an SSE endpoint has to put each chunk inside JSON, where half a
   codepoint cannot be encoded at all.

   So an incomplete trailing sequence is held back and prepended to the
   next emission. At most 3 bytes are ever pending. */
typedef struct {
    LZTokenSink sink;
    void       *ctx;
    unsigned char pend[4];
    int         n_pend;
} lz_emit_t;

static int lz_u8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 6) return 2;
    if ((c >> 4) == 14) return 3;
    if ((c >> 3) == 30) return 4;
    return 0;                    /* continuation or invalid lead */
}

static void lz_emit(lz_emit_t *e, const char *bytes, int len) {
    unsigned char buf[LZ_STOP_TAIL + 8];
    int n = 0, cut, i;

    if (len <= 0) return;
    for (i = 0; i < e->n_pend; i++) buf[n++] = e->pend[i];
    for (i = 0; i < len && n < (int)sizeof(buf); i++)
        buf[n++] = (unsigned char)bytes[i];
    e->n_pend = 0;

    /* Walk back at most 3 bytes for a lead whose sequence is not yet
       complete; everything before it is safe to emit. */
    cut = n;
    for (i = 1; i <= 3 && i <= n; i++) {
        int need = lz_u8_len(buf[n - i]);
        if (need == 0) continue;             /* continuation byte */
        if (need > i) cut = n - i;           /* truncated - hold it back */
        break;
    }
    if (cut > 0) e->sink((const char *)buf, cut, e->ctx);
    for (i = cut; i < n; i++) e->pend[e->n_pend++] = buf[i];
}

/* Final flush: whatever is still pending is genuinely incomplete input,
   not a split we introduced, so it goes out rather than being dropped. */
static void lz_emit_flush(lz_emit_t *e) {
    if (e->n_pend > 0) {
        e->sink((const char *)e->pend, e->n_pend, e->ctx);
        e->n_pend = 0;
    }
}

/* ------------------------------------------------------ stop strings */

/* Byte-wise stop-string filter.

   A stop string may SPAN MULTIPLE TOKENS - "<|im_start|>" need not be a
   single BPE token - so matching cannot look only inside the current
   token's bytes. Bytes that could still begin a stop string are held back
   and only released once they can no longer participate.

   Extracted from lz_generate_resume so it can be exercised directly. */
typedef struct {
    const char *const *stop;
    int   slen[LZ_MAX_STOP];    /* strlen of each, computed once */
    int   n_stop;
    int   max_stop;             /* longest stop string; <=0 disables */
    int   matched;              /* index of the stop that fired, else -1 */
    /* Held-back bytes, never more than max_stop-1 of them. Bounded by
       LZ_STOP_TAIL because that is also the largest stop string the
       caller may pass; the incoming token is NOT copied in here, see
       lz_stopf_at. */
    char  tail[LZ_STOP_TAIL];
    int   tail_len;
} lz_stopf_t;

static void lz_stopf_init(lz_stopf_t *f, const char *const *stop, int n_stop) {
    int i;
    memset(f, 0, sizeof(*f));
    f->stop = stop;
    f->n_stop = n_stop;
    f->matched = -1;
    for (i = 0; i < n_stop && i < LZ_MAX_STOP; i++) {
        f->slen[i] = stop[i] ? (int)strlen(stop[i]) : 0;
        if (f->slen[i] > f->max_stop) f->max_stop = f->slen[i];
    }
}

/* Byte i of the virtual concatenation tail[0..nh) ++ b[...].

   The incoming token is scanned WHERE IT LIES rather than being copied
   into the tail, and that is the whole point of this indirection. The
   tail buffer holds max_stop-1 bytes; a token in this vocab decodes to
   as much as 45 bytes, so a buffer that had to hold both would need
   LZ_STOP_TAIL-1 + 45 = 108. Making room by FLUSHING the overflow is
   wrong: it emits bytes that could still begin a match, the match is
   then missed silently, and the leading bytes of the stop string appear
   in the reply. Reachable whenever max_stop + token_bytes > 65, i.e.
   from any stop string of 21 bytes up. */
static unsigned char lz_stopf_at(const char *h, int nh, const char *b, int i) {
    return (unsigned char)(i < nh ? h[i] : b[i - nh]);
}

/* Emit the first n bytes of that concatenation, in at most two calls.
   Not byte by byte: the sink is one SSE event per call. */
static void lz_stopf_emit(lz_emit_t *e, const char *h, int nh,
                          const char *b, int n) {
    if (n <= 0) return;
    if (n <= nh) { lz_emit(e, h, n); return; }
    if (nh > 0) lz_emit(e, h, nh);
    lz_emit(e, b, n - nh);
}

/* Feed one token's bytes. Returns 1 when a stop string matched, in which
   case everything before the match has been emitted and generation must
   end; 0 when it did not, in which case everything that can no longer
   participate has been emitted. */
static int lz_stopf_feed(lz_stopf_t *f, lz_emit_t *e,
                         const char *bytes, int blen) {
    int nh, tot, end, k, keep, from, i;

    if (blen <= 0) return 0;
    if (f->max_stop <= 0) {          /* no stop strings; emit directly */
        lz_emit(e, bytes, blen);
        return 0;
    }
    nh = f->tail_len;
    tot = nh + blen;

    /* Scan by the position a match ENDS at, over ends inside the new
       bytes only - anything ending earlier was scanned when those bytes
       arrived. Earliest end wins; within one end, the earlier stop
       string wins.

       That is vLLM's rule (check_stop_strings: "the stop string that
       completes earliest in the text is selected ... ties are broken by
       stop-list order"), and its docstring gives the reason to prefer
       it over leftmost-START: matching by end is the rule whose answer
       does not depend on
       how the bytes were split into tokens. Leftmost-start differs
       whenever one token delivers the bytes completing two stop
       strings - stop ["ABCD", "C"] on "ABCD" keeps "AB" here and
       emitted nothing before. */
    for (end = nh + 1; end <= tot; end++) {
        for (k = 0; k < f->n_stop; k++) {
            int sl = f->slen[k], s = end - sl, j;
            if (sl <= 0 || s < 0) continue;
            /* Last byte first: this runs once per generated byte per
               stop string, and the last byte rejects almost all of it. */
            if ((unsigned char)f->stop[k][sl - 1] !=
                lz_stopf_at(f->tail, nh, bytes, end - 1)) continue;
            for (j = 0; j < sl - 1; j++)
                if ((unsigned char)f->stop[k][j] !=
                    lz_stopf_at(f->tail, nh, bytes, s + j)) break;
            if (j == sl - 1) {
                lz_stopf_emit(e, f->tail, nh, bytes, s);
                f->tail_len = 0;
                f->matched = k;
                return 1;
            }
        }
    }

    /* Nothing matched: hold back the last max_stop-1 bytes, since a
       shorter suffix cannot begin any stop string, and release the rest.
       Same quantity as vLLM's SamplingParams.output_text_buffer_length. */
    keep = f->max_stop - 1;
    if (keep > tot) keep = tot;
    from = tot - keep;
    lz_stopf_emit(e, f->tail, nh, bytes, from);
    /* Copied forward: the destination index is never above the source
       index, so overlapping the tail's own front is safe. */
    for (i = 0; i < keep; i++)
        f->tail[i] = (char)lz_stopf_at(f->tail, nh, bytes, from + i);
    f->tail_len = keep;
    return 0;
}

/* On a normal ending the held-back tail is part of the reply. Without
   this a generation that never hits a stop string loses its last few
   dozen bytes - exactly where the final period sits. */
static void lz_stopf_flush(lz_stopf_t *f, lz_emit_t *e) {
    if (f->tail_len > 0) {
        lz_emit(e, f->tail, f->tail_len);
        f->tail_len = 0;
    }
}

/* ------------------------------------------------- speculative decoding */

/* Positive control for the ring-based rollback: increments once per
   round that actually needed one (n_accept < k_eff, inside lz_spec_round
   below). Defined here rather than in forward.c, unlike the engine's
   other lz_debug_* counters - every one of those is both incremented
   AND defined in forward.c, but this one tracks a generate.c-level fact
   (how many rounds took the ring-recovery path), not a forward-pass-
   level one, so it lives with the code that actually increments it.
   cli_main.c declares it extern to print it, the same relationship it
   already has with forward.c's own counters. There is deliberately no
   "n_replay" counter alongside it: rollback is a ring index assignment,
   not a restore-and-re-forward, so lz_spec_round has no lz_ckpt_restore
   or lz_forward_batch call to instrument. This counter is the positive
   proof that stands in its place: rounds with real rejections took SOME
   path, and bit-identity (a REAL gate for this specific mechanism,
   unlike catch-up/the mtp_chain seed - see lz_spec_round's own module
   comment) is what proves that path was correct. */
long long lz_debug_n_ring_rollback = 0;

/* Contract and its provenance: llama_zh.h. Deliberately takes arrays
   rather than reaching into LZRunState - it has no state of its own, so
   it can be exercised without a model, which is the only part of the
   speculative path that is testable before a draft head exists. */
int lz_spec_accept(const int *draft, const int *targ, int n_draft,
                   int bonus, int *out) {
    int i;
    if (!draft || !targ || !out || n_draft < 0) return 0;
    for (i = 0; i < n_draft; i++) {
        out[i] = targ[i];               /* the VERIFY pass decides, always */
        if (draft[i] != targ[i]) return i;
    }
    out[n_draft] = bonus;
    return n_draft;
}

/* Residual resample: normalize(max(0, p-q)) over the full vocab, then
   draw from it (temp>0 speculative decoding phase 2). Two passes rather
   than materializing a normalized copy - the same
   shape sampler.c's own select_candidates/sample_mult already use
   (accumulate a total, then walk the cumulative sum scaled by a coin
   drawn against that total, never dividing each entry).

   total<=0.0f (p and q coincide everywhere, or q dominates p
   everywhere) falls back to argmax of p, the least-arbitrary choice
   available - lz_spec_accept_temp's own caller only reaches this
   function after a REJECTION, which requires u >= min(1,p(x)/q(x)),
   and a genuine u==threshold coin flip at the exact boundary is the
   only way to get here with an empty residual; it is not reachable by
   construction otherwise (p(x)<q(x) whenever x was rejected implies
   SOME index has positive residual mass, since a full distribution
   cannot have p uniformly <= q while also both summing to 1 unless
   p==q exactly). */
static int sample_residual(const float *p, const float *q, int vocab,
                           unsigned long long *residual_rng) {
    float total = 0.0f, coin, cdf = 0.0f;
    int i, last_pos = -1;
    for (i = 0; i < vocab; i++) {
        float d = p[i] - q[i];
        if (d > 0.0f) { total += d; last_pos = i; }
    }
    if (total <= 0.0f) return lz_sample_argmax(p, vocab);
    coin = lz_random_f32(residual_rng) * total;
    for (i = 0; i < vocab; i++) {
        float d = p[i] - q[i];
        if (d > 0.0f) {
            cdf += d;
            if (coin < cdf) return i;
        }
    }
    /* fp rounding fallback - "return the last candidate", same
       convention sample_mult (sampler.c) itself falls back to. */
    return last_pos;
}

/* Coupled (textbook) accept/residual-resample test - the temp>0 analog
   of lz_spec_accept above, same "no state of its own, testable without
   a model" positioning (see that function's own comment): p, q, u are
   all supplied by the caller (not read from a model or drawn from an
   internal RNG), so a test can hit exact boundary values.

   p    target's FULL distribution (vocab entries, sums to ~1) - lz_
        target_dist's own output (sampler.c), already penalty- and
        target-filter-adjusted by the caller.
   q    draft's FULL distribution (vocab entries, sums to ~1) - lz_
        sample_temp_q's own output (sampler.c), temperature only.
   x    the draft's own sampled token (lz_sample_temp_q's return
        value) - the ONE token this test is actually deciding about.
   u    injected uniform draw in [0,1) for the ACCEPT test itself - NOT
        drawn internally, so a test can hit exact boundaries (u just
        below / at / just above p(x)/q(x)).
   residual_rng  caller-owned RNG state for the REJECT branch's own
        resample - a draw INDEPENDENT of u: whether x is accepted and,
        if not, which token replaces it are two separate random
        choices, and correlating them (e.g. reusing u) would bias the
        residual draw in a way the theorem's own proof does not permit.
   out_accepted  written 1 if x was accepted as-is, 0 if a residual
        token was substituted - non-NULL required for a test to tell
        the two branches apart without inferring it from the return
        value (the residual sample CAN legitimately equal x).

   Returns the token to actually emit: x if accepted, otherwise
   sample_residual's own draw. -1 on invalid input (null p/q/
   residual_rng, non-positive vocab, or x out of [0,vocab) - token ids
   in this project are always >= 0, so -1 is never a real token and is
   safe as an error sentinel, unlike lz_spec_accept's own 0 - see this
   function's own contract in llama_zh.h for why an int return alone
   could not carry that distinction: 0 is itself a legitimate token id
   here, but "no draft ever ends in a 0-token round" is not something
   lz_spec_accept had to worry about, since IT reports a COUNT, not a
   token). */
int lz_spec_accept_temp(const float *p, const float *q, int x, int vocab,
                        float u, unsigned long long *residual_rng,
                        int *out_accepted) {
    float qx, ratio;
    if (!p || !q || !residual_rng || vocab <= 0 || x < 0 || x >= vocab) {
        if (out_accepted) *out_accepted = 0;
        return -1;
    }
    qx = q[x];
    if (qx <= 0.0f) {
        /* The draft assigning its own sampled token zero probability
           cannot happen from a correctly-built q (lz_sample_temp_q
           only ever returns an index with logits[index] > 0 after a
           real softmax) - guarded anyway so a hand-built test q
           (this function's whole reason for existing) cannot divide
           by zero. Treated as an unconditional reject: p(x)/q(x) is
           undefined, not infinite, and "always accept" would be the
           wrong direction (an event q assigns literally zero mass to
           was never really sampled by q in the first place). */
        ratio = 0.0f;
    } else {
        ratio = p[x] / qx;
        if (ratio > 1.0f) ratio = 1.0f;   /* a probability ratio never exceeds 1 */
    }
    if (u < ratio) {
        if (out_accepted) *out_accepted = 1;
        return x;
    }
    if (out_accepted) *out_accepted = 0;
    return sample_residual(p, q, vocab, residual_rng);
}

/* One MTP speculative round. Contract: llama_zh.h.

   State rollback: the ring mechanism exists because a snapshot/restore
   approach could never turn --spec K net-positive on any machine where
   verify does not amortize:

     KV cache      Rolls back by NOT ADVANCING the position counter past
                   the accepted prefix. The verify batch below writes
                   rows for ALL k+1 positions unconditionally (draft
                   tokens that turn out rejected included), but a row is
                   only ever READ by a later score loop that scans
                   t <= pos - so once *out_new_pos stops short of the
                   rejected rows, they are simply never read again and
                   get overwritten whenever a future position lands on
                   the same index. No copy anywhere. UNCHANGED by this
                   round's work.
     SSM/conv       CANNOT roll back by "not advancing" the way the KV
                   cache does: forward_ssm/forward_kda's recurrence
                   folds every token it sees into ONE running state,
                   in place - there is no "unread rows" to just stop
                   reading, the contamination IS the state. The fix is
                   not to snapshot around it, it is to give the
                   recurrence a RING of states instead of one
                   (forward.h's s->ssm_slot):
                   every recurrent step here already reads its entire
                   state once and rewrites it once, every token
                   (lz_gdn_step/lz_kda_step/lz_causal_conv1d_step's own
                   two-pass shape - see ops.h's lz_gdn_step comment for
                   the byte-count proof), so writing token i's result to
                   the NEXT ring slot instead of back onto the same one
                   costs the exact same bytes, and turns "rejection" and
                   "mid-round stop" into a single index assignment
                   instead of a restore-and-re-forward. See below.
     MTP's own KV   Same shape as the body's KV: rolls back by not
     and position   advancing s->mtp_pos past n_accept (see forward.h's
                   comment on that field). The MTP block writes k rows
                   during drafting regardless of what verify later
                   decides; a rejected round's later rows are simply
                   never scored by a future round, which always starts
                   from s->mtp_pos as of the LAST accepted count.
                   UNCHANGED by this round's work.

   THE RING'S OWN CONTRACT: lz_forward_verify's k_eff+1-token batch
   advances s->ssm_slot INTERNALLY, one slot per token forwarded
   (forward_chunk's own ring_base comment) - by the time it returns,
   s->ssm_slot already equals (base_slot + k_eff + 1) % ring_depth,
   "as if every drafted token had been accepted". This function's own
   job is to point s->ssm_slot at the RIGHT slot once n_accept is
   actually known: (base_slot + n_accept + 1) % ring_depth, which
   already EXISTS - lz_forward_verify's own per-token advance computed
   and left it there while forwarding the FULL batch, whether or not
   the tail ends up rejected (a batched forward's row i never reads any
   row's state past its own - forward_attn's/the recurrence's own
   `t <= pos`-shaped loops - so a rejected tail's rows never
   contaminate an earlier, still-wanted slot). No restore, no replay,
   for EITHER a plain rejection (n_accept < k_eff, handled here) or a
   mid-round stop that needs to trim even further back (ei < n_accept,
   handled by the CALLER - lz_generate_resume's own trim branch reads
   the SAME base_slot this function captures, since both read
   s->ssm_slot at the same point before this function's own verify call
   ever runs, with nothing in between to change it).

   THE TRAP THE ROLLBACK DESIGN EXISTS TO AVOID: if the accept/reject
   path is never actually exercised by a test, a bug in the restore
   branch (starting with "forgot to restore conv_state") could pass
   every gate silently. LZGenOpts.spec_debug_break_rollback exists for
   the SAME reason, reimplemented against the ring below - see its own
   comment at the point it is checked for what "broken" means now that
   there is no separate per-buffer restore step left to omit.

   p_min / n_min (filling the gap against llama.cpp's own speculative
   decoding - its speculative.cpp was read start to finish and diffed
   against this function; the core mechanism - one draft head, chained
   k steps, self-fed hidden state within a chain, target-hidden resync
   across rounds, embed-then-hidden concat - matches on all four
   points):

     p_min   Stop drafting a round EARLY the first time the head's own
             top-1 confidence drops below this threshold
             (speculative.cpp:1601-1609) - the LOW-CONFIDENCE token
             itself is never added to the draft, matching that
             reference exactly (the check runs BEFORE the token would
             be recorded, not after). 0.0 (default) never stops early -
             lz_argmax_p1's return is a probability, always > 0.0 for
             any finite logits, so `p1 < 0.0` is never true.
     n_min   AFTER drafting (with whatever p_min already trimmed it to),
             if fewer than n_min tokens were drafted, throw the WHOLE
             draft away rather than verify a too-short one
             (speculative.cpp:1663-1668) - not a partial acceptance, a
             full discard. 0 (default) never discards (k_eff >= 0 is
             always true).

   Neither knob can change what the engine ultimately EMITS: verify is
   always what decides an accepted token's identity (lz_spec_accept's
   own contract), and a token this function never offers as a draft
   still gets produced correctly, just via the bonus/correction slot
   instead of the accepted-draft slot. They only change how much
   speculative work is attempted before verify's answer is authoritative
   - see lz_argmax_p1's own comment (sampler.h) for the same argument
   applied to its own FP precision. */

/* The assumed penalty window: verify computes k_eff+1 rows in
   ONE batched forward, before it is known which (if any) draft tokens
   get accepted, so none of them are in samp->ring yet - lz_sampler_
   observe only ever runs on tokens the CALLER already knows were kept
   (lz_generate_resume's own emission loop, after this function
   returns). This builds, for a given row, the window that WOULD be in
   samp->ring if this round's own draft tokens draft[0..nv-1] had each
   been individually generated and observed already, so that row's
   penalties can be computed with apply_penalties_assumed instead of
   skipped.

   nv is the count of THIS ROUND'S OWN draft tokens folded in -
   draft[0..nv-1] - NOT a count that includes anchor: anchor is already
   the newest entry in samp->ring by the time lz_spec_round ever runs
   (lz_generate_resume observes it before calling this function, on
   every path - the fresh-capture branch's own lz_sampler_observe call,
   or the previous round's own emission loop when have_seed carried it
   over). Folding anchor in here too would double-count it - the one
   correction this function makes to the naive derivation, which used
   "verify_tokens[0..i]" (anchor included) without accounting for
   anchor already sitting in the real ring by the time verify runs.

   nv=0 (this round's very first row, predicting what follows anchor)
   therefore reduces to "exactly the real window as it stands right
   now" - the same content, in the same oldest-first order, apply_
   penalties itself would read directly - which is what makes row 0
   bit-identical to the non-speculative path's own first prediction.

   window[] must have room for samp->cap entries (a static buffer at the
   call site - iron law six's "no large arrays on the Win98 stack";
   samp->cap can be LZ_PENALTY_MAX_WINDOW=4096 at repeat_last_n=-1).
   Returns the window length actually written, always <= samp->cap:
   when nv < cap, tail_count (below) is bounded by cap-nv, so tail_count
   + nv <= cap by construction; when nv >= cap, exactly cap draft tokens
   are taken and nothing from the real ring survives (they alone would
   already have evicted it, the same way samp->ring itself would if fed
   one at a time). */
static int build_assumed_window(const LZSampler *samp, const int *draft,
                                int nv, int *window) {
    int cap = samp->cap;
    int keep, tail_count, i;
    if (nv >= cap) {
        for (i = 0; i < cap; i++) window[i] = draft[nv - cap + i];
        return cap;
    }
    keep = cap - nv;
    tail_count = (keep < samp->n_win) ? keep : samp->n_win;
    for (i = 0; i < tail_count; i++) {
        window[i] = samp->ring[(samp->head + samp->n_win - tail_count + i)
                               % samp->cap];
    }
    for (i = 0; i < nv; i++) window[tail_count + i] = draft[i];
    return tail_count + nv;
}

/* Applies the assumed-window penalties to one verify row's logits IN
   PLACE, if the caller's own penalties are not at their identity
   values - samp must be non-NULL (callers check first; see verify_row_
   argmax below and lz_spec_round_temp). Factored out (temp>0 phase 2)
   from verify_row_argmax's own body - PURE extraction, same
   condition/window-build/apply sequence - so that lz_spec_round_temp's
   own target-distribution builder can apply the identical penalty step
   without a second, independently-drifting copy of "when is the window
   even worth building" (iron law two). verify_row_argmax's own visible
   behavior is unaffected by this extraction - the greedy bit-identity
   gate re-covers this exact code path and is the actual proof, not this
   comment. */
static void apply_row_penalties_assumed(const LZSampler *samp, const int *draft,
                                        int nv, float *row) {
    int has_rep = (samp->p.repetition_penalty != 1.0f);
    int has_add = (samp->p.presence_penalty != 0.0f ||
                   samp->p.frequency_penalty != 0.0f);
    if (samp->p.repeat_last_n != 0 && (has_rep || has_add)) {
        /* static: up to LZ_PENALTY_MAX_WINDOW (4096) ints - iron
           law six's "no large buffers on the Win98 stack". */
        static int window[LZ_PENALTY_MAX_WINDOW];
        int wl = build_assumed_window(samp, draft, nv, window);
        apply_penalties_assumed(&samp->p, row, window, wl);
    }
}

/* log(sum(exp(l))) over `n` logits, streaming - the same online
   algorithm lz_sample_argmax_p (sampler.c) uses, and for the same
   reason: the rollout needs a normalizer, not a materialized probability
   vector, and a 32K-wide float array per lookahead step is exactly the
   allocation that comment refuses. Runs OUTSIDE the PC=24 region
   (lz_fpu_float_end already ran in forward_chunk), so double is safe
   here - iron law six's third clause. */
static double lz_look_lse(const float *l, int n) {
    double mx = (double)l[0], sum = 0.0;
    int i;
    for (i = 1; i < n; i++) if ((double)l[i] > mx) mx = (double)l[i];
    for (i = 0; i < n; i++) sum += exp((double)l[i] - mx);
    return mx + log(sum);
}

/* Top-`w` logit indices, descending. Insertion sort into a w-wide
   array: w <= LZ_LOOK_W_MAX (4), so this is 4 comparisons per vocab
   entry against a partial sort's log factor - the same shape
   cli_main.c's print_topk uses, kept separate because that one is
   debug output and this one is on a decode path. */
static void lz_look_topw(const float *l, int n, int w, int *idx) {
    int cnt = 0, i, j;
    static float val[LZ_LOOK_W_MAX];
    for (i = 0; i < n; i++) {
        if (cnt < w || l[i] > val[cnt - 1]) {
            j = (cnt < w) ? cnt : w - 1;
            while (j > 0 && val[j - 1] < l[i]) {
                val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--;
            }
            val[j] = l[i]; idx[j] = i;
            if (cnt < w) cnt++;
        }
    }
    for (i = cnt; i < w; i++) idx[i] = idx[cnt ? cnt - 1 : 0];
}

static int lz_look_is_eos(const LZGenOpts *o, int tok) {
    int k;
    for (k = 0; k < o->n_eos; k++) if (o->eos_ids[k] == tok) return 1;
    return 0;
}

/* Bounded lookahead: score the top-W continuations D tokens deep and
   return the FIRST token of the best one.
   -1 with errbuf set on a checkpoint failure; the caller then falls
   back to nothing - it aborts, because a half-restored recurrent state
   is not something to keep generating on.

   *** WHY A CHECKPOINT AND NOT THE ROLLBACK RING ***
   The ring (forward.h s->ssm_slot) makes a speculative round's rollback
   free, but it cannot serve this function: only lz_forward_verify
   advances slots, while the single-token path this lookahead is built
   from passes "the same slot as both read and write source, reproducing
   the old in-place update exactly". A lookahead built out of
   lz_forward_capture calls therefore DESTROYS the state it would need
   to roll back to. The ring is also depth 1 when the model has no MTP
   head, which is most of them.
   So the cost is one LZStateCkpt (1.51 MB, the same object the prefix
   cache already carries) plus one save and W restores per emitted
   token. Against W*D forward passes - each of which moves 41.65 MB of
   weights - the copies are about a tenth of the added time, not the
   dominant term. Stated rather than implied, per iron law three: this
   is an arithmetic estimate from the measured per-token weight traffic,
   NOT a measurement on a target machine.

   The KV cache is deliberately not saved: it is append-only and indexed
   by absolute position, so every branch overwrites entries at >= pos+1
   and the entries below stay valid. That is LZStateCkpt's own contract,
   and its `epoch` field is what makes the assumption checked rather
   than assumed. */
static int lz_look_pick(const LZModel *m, LZRunState *s, LZSampler *samp,
                        const LZGenOpts *o, LZStateCkpt *ck, float *scratch,
                        const float *logits0, int pos, int vocab,
                        char *errbuf, int errlen) {
    int w = o->look_width, dmax = o->look_depth;
    int cand[LZ_LOOK_W_MAX];
    int draft[LZ_LOOK_D_MAX];
    double best = 0.0;
    int best_i = 0, i, d, at;
    int saved_pos = pos;

    memcpy(scratch, logits0, (size_t)vocab * sizeof(float));
    if (samp && !o->look_raw) apply_row_penalties_assumed(samp, draft, 0, scratch);
    lz_look_topw(scratch, vocab, w, cand);

    if (lz_ckpt_save(ck, s, m, pos, errbuf, errlen) != 0) return -1;

    for (i = 0; i < w; i++) {
        double lse = lz_look_lse(scratch, vocab);
        double score = (double)scratch[cand[i]] - lse;
        int len = 1, stop = 0;
        if (i > 0) {
            if (lz_ckpt_restore(ck, s, m, &at, errbuf, errlen) != 0) return -1;
            memcpy(scratch, logits0, (size_t)vocab * sizeof(float));
            if (samp && !o->look_raw)
                apply_row_penalties_assumed(samp, draft, 0, scratch);
        }
        draft[0] = cand[i];
        for (d = 1; d < dmax && !stop; d++) {
            float *lg = lz_forward_capture(m, s, draft[d - 1], pos + d);
            int nxt;
            if (!lg) break;
            memcpy(scratch, lg, (size_t)vocab * sizeof(float));
            if (samp && !o->look_raw)
                apply_row_penalties_assumed(samp, draft, d, scratch);
            nxt = lz_sample_argmax(scratch, vocab);
            score += (double)scratch[nxt] - lz_look_lse(scratch, vocab);
            draft[d] = nxt;
            len++;
            if (lz_look_is_eos(o, nxt)) stop = 1;
        }
        /* cum_logprob / len^lp - HF's and vLLM's own scoring
           (refsrc/vllm/.../beam_search/utils.py get_beam_search_score).
           At a fixed depth every branch has the same len and this is the
           identity; it only separates branches that stopped early on
           EOS, which is the whole case it exists for. */
        if (o->look_lp != 0.0f && len > 1)
            score /= pow((double)len, (double)o->look_lp);
        if (i == 0 || score > best) { best = score; best_i = i; }
    }
    if (lz_ckpt_restore(ck, s, m, &at, errbuf, errlen) != 0) return -1;
    (void)saved_pos;
    return cand[best_i];
}

/* One verify row: apply the assumed-window penalties (if samp is
   non-NULL and the caller's own penalties are not at their identity
   values) to `row`'s logits IN PLACE, then take the same argmax rule
   every other greedy pick in this file uses. samp==NULL (every existing
   ctypes-direct caller of lz_spec_round) is the default: no penalty
   window, plain argmax on raw verify logits. */
static int verify_row_argmax(const LZSampler *samp, const int *draft, int nv,
                             float *row, int vocab) {
    if (samp) apply_row_penalties_assumed(samp, draft, nv, row);
    return lz_sample_argmax(row, vocab);
}

int lz_spec_round(const LZModel *m, LZRunState *s,
                  int anchor, int anchor_pos, int k,
                  float p_min, int n_min, const LZSampler *samp,
                  int debug_break_rollback, int debug_skip_catchup,
                  int *out, int *out_n_accept, int *out_new_pos,
                  int *out_k_eff, char *errbuf, int errlen) {
    int draft[LZ_SPEC_K_MAX];
    int verify_tokens[LZ_SPEC_K_MAX + 1];
    int targ[LZ_SPEC_K_MAX];
    int vocab, n_accept, i, rc, k_eff;
    int bonus;
    int base_slot;   /* s->ssm_slot before verify runs - see this
                         function's own module comment on the ring. */

    /* LZ_ERR_SET*(rc, errbuf, errlen, LZ_ERR_XXX) rather than a bare
       lz_err_fmt(errbuf, errlen, rc) call - iron law one requires the
       THIRD ARGUMENT to be a literal LZ_ERR_* enum at the call site,
       not a variable that merely happens to hold one; tools_check_
       err_full.py / test_err_localization.py enforce this so a future
       `snprintf(errbuf, len, some_int)`-shaped mistake cannot hide
       behind "it looked like the others". */
    if (!m || !s || !out || !out_n_accept || !out_new_pos || !out_k_eff) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        return -rc;
    }
    if (!m->mtp) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_SPEC_NO_HEAD);
        return -rc;
    }
    if (k < 1 || k > LZ_SPEC_K_MAX) {
        LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_SPEC_K_RANGE, k, LZ_SPEC_K_MAX);
        return -rc;
    }
    /* s (the SSM/conv rollback ring) may have been sized for FEWER than
       LZ_SPEC_K_MAX slots - lz_state_alloc's own spec_k_max parameter
       (forward.h). Checked HERE too, not only in
       lz_generate_resume's own copy of this check, because this
       function is documented as "the piece that can be exercised
       standalone via ctypes" (this function's own module comment) -
       callers that skip lz_generate_resume entirely (any future direct
       binding) must not be able
       to silently alias two different rollback targets onto the same
       physical ring slot by asking for a k this state was never sized
       for. */
    if (k > s->ssm_ring_depth - 1) {
        LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_SPEC_K_RANGE, k, s->ssm_ring_depth - 1);
        return -rc;
    }
    vocab = m->config.vocab_size;

    /* Draft chain: s->mtp_chain already holds the caller's
       lz_forward_capture hidden state for `anchor`; each step
       overwrites it with its own post-norm output for the next
       (lz_mtp_draft_step's own contract, forward.h). k_eff (<= k) is
       how many steps actually got recorded - p_min can cut it short. */
    k_eff = 0;
    for (i = 0; i < k; i++) {
        int in_tok = (i == 0) ? anchor : draft[i - 1];
        float p1;
        int idx;
        float *logits = lz_mtp_draft_step(m, s, in_tok, s->mtp_pos + i);
        if (!logits) {
            LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
            return -rc;
        }
        idx = lz_argmax_p1(logits, vocab, &p1);
        if (p1 < p_min) break;   /* low-confidence step: not recorded, matching speculative.cpp's own order */
        draft[i] = idx;
        k_eff = i + 1;
    }

    if (k_eff < n_min) {
        /* Discard the whole draft - not worth a verify batch
           (speculative.cpp:1663-1668). Degenerates this round to plain
           single-token decoding: nothing new is forwarded (anchor was
           never touched by drafting, which reads/writes only
           s->mtp_chain and the MTP's own reserved KV slot - see
           lz_mtp_draft_step, forward.h), so *out_new_pos stays at
           anchor_pos (the caller's "count forwarded" is unchanged) and
           out[0] echoes anchor purely so the caller's generic
           `token = out[*out_n_accept]` line does not need a special
           case for this path too - the CALLER still must not re-emit
           it (it was already emitted before this call), which is why
           the return value is the discard sentinel 0, distinct from
           every real round's 1..k+1. s->mtp_pos is not touched: k_eff
           draft steps were computed and their KV rows written, but
           nothing here ever advances the counter past them, so a later
           round starting from the OLD s->mtp_pos simply never scores
           those rows again - the same "don't advance past it" argument
           this function's own module comment makes for a rejected
           round, just triggered before verify instead of by it. */
        out[0] = anchor;
        *out_n_accept = 0;
        *out_new_pos = anchor_pos;
        *out_k_eff = k_eff;
        return 0;
    }

    /* base_slot: s->ssm_slot as it stands right now, BEFORE the verify
       batch below touches anything - this is what the ring's own
       per-token advance (forward_chunk, during lz_forward_verify) will
       count FROM. Captured here rather than trusted to still equal
       whatever the caller last saw, on the same discipline as anchor_
       pos itself being a parameter rather than re-derived. */
    base_slot = s->ssm_slot;

    verify_tokens[0] = anchor;
    for (i = 0; i < k_eff; i++) verify_tokens[i + 1] = draft[i];
    if (lz_forward_verify(m, s, verify_tokens, k_eff + 1, anchor_pos + 1) != 0) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
        return -rc;
    }
    for (i = 0; i < k_eff; i++) {
        targ[i] = verify_row_argmax(samp, draft, i,
                                    s->mtp_logits + (size_t)i * vocab, vocab);
    }
    bonus = verify_row_argmax(samp, draft, k_eff,
                              s->mtp_logits + (size_t)k_eff * vocab, vocab);

    n_accept = lz_spec_accept(draft, targ, k_eff, bonus, out);

    if (n_accept < k_eff) {
        /* Rejection - see this function's own module comment for the
           full ring derivation. lz_forward_verify's own per-token ring
           advance already computed and left, at slot (base_slot +
           n_accept + 1) % ring_depth, exactly "the state after
           forwarding n_accept+1 tokens" - the correct post-rollback
           state - as a side effect of forwarding the FULL k_eff+1-token
           batch (a later, rejected row never contaminates an earlier
           slot - forward_attn's/the recurrence's own `t <= pos`-shaped
           loops). Pointing s->ssm_slot there is the entire "rollback":
           no restore, no re-forward. */
        lz_debug_n_ring_rollback++;
        if (!debug_break_rollback) {
            s->ssm_slot = (base_slot + n_accept + 1) % s->ssm_ring_depth;
        }
        /* debug_break_rollback: KNOWN-BAD CONTROL, see this function's
           module comment and LZGenOpts.spec_debug_break_rollback -
           deliberately skip the assignment above, leaving s->ssm_slot
           at whatever lz_forward_verify's own internal advance left it,
           (base_slot + k_eff + 1) % ring_depth: "as if every drafted
           token had been accepted" even though n_accept < k_eff really
           were. This is a DIFFERENT wrong value than the correct one
           whenever n_accept < k_eff (which is exactly when this branch
           runs), so the body's own subsequent forwards read a
           genuinely wrong recurrent state - the same class of bug
           "restore ssm, skip conv" would have reproduced, in this
           mechanism's own terms (there is no way to correctly roll back
           "most of the state" and get this wrong only partially - one
           index governs all four buffers together, which is the point of
           the ring). */
    }
    /* n_accept == k_eff (nothing rejected): lz_forward_verify's own
       natural post-state, (base_slot + k_eff + 1) % ring_depth, IS
       ALREADY (base_slot + n_accept + 1) % ring_depth - no separate
       assignment needed, and debug_break_rollback has nothing to break
       here either (there is nothing to roll back FROM). */

    /* Seed s->mtp_chain for the CALLER's next round from the verify
       batch's own captured hidden state at row n_accept (matching
       llama.cpp's own speculative.cpp accept(): "i_h = min(n_accepted,
       n_rows-1)" reads exactly this row as the next round's draft seed)
       - this is "hidden at *out_new_pos" (the last
       position actually forwarded this round: verify_tokens[n_accept],
       an ACCEPTED token or anchor itself if n_accept==0, forwarded at
       anchor_pos+1+n_accept), paired later by the caller with the
       correction/bonus token at *out_new_pos+1 as MTP's own "next
       token" input - exactly the pairing the FIRST draft step of a
       round always needs (see lz_mtp_draft_step's own comment).

       Correct regardless of n_accept < k_eff (the ring-rollback branch
       above, which now only moves an index and never re-forwards
       anything): row n_accept's capture happened during the verify
       pass that ran unconditionally over the FULL k_eff+1-token batch,
       and that row's own computation is causally independent of every
       LATER (rejected) row - a batched forward's row i never reads any
       row's KV/state past position pos0+i (forward.c's own score/
       recurrence loops are strictly `t <= pos`) - so it stays correct
       whether or not the tail past it ends up rejected, without
       needing a fresh capture either way. This is what lets the caller
       skip an entire redundant per-round forward pass (measured:
       capture cost 1.00x a plain forward, on top of verify's own 1.92x
       - see the CLI's [spec us: ...] line). The caller
       (lz_generate_resume) still owns
       falling back to a fresh capture on its OWN first round and after
       any discarded round (lz_mtp_draft_step's own chaining
       overwrites s->mtp_chain even on a discarded draft, so a stale
       value there cannot be trusted).

       NO DIRECT GATE covers this argument: it is a proof, not a test.
       Bit-identity (--spec 0 vs --spec K) is a null judge for it, same
       as every other MTP-only-state claim in this file - a wrong seed
       can only degrade draft quality, never the emitted bytes, since
       the target model always decides those. The only thing that could
       actually catch a violation is alpha going down for no other
       explained reason; there is no dedicated counter or repro test the
       way lz_debug_n_capture (call count) or
       test_cli_mtp_pos_trim_rollback (an independently-derived expected
       value) cover their own claims. Recorded at the confidence level
       it actually has: believed correct, argued carefully, not
       independently gated.

       ALSO ON RECORD: this change measured ~13% LOWER throughput on
       this machine (spec1 0.104 -> 0.117 s/token), not higher - the
       capture call it removes was not pure waste, it forwarded the
       round's own anchor and that forward's logits produced a token as
       a side effect, so removing it also removes ~0.9 token/round of
       output on top of its 1.00x cost. The trade only nets positive
       once verify's own batch amortizes well, which this machine's
       un-SIMD'd verify path (iron law two) does not. This is a
       deliberate consistency-with-llama.cpp change, not a performance
       one, and iron law three means the ratio can (and on the Win98
       target family, likely does) flip. */
    {
        int dim = m->config.hidden_size;
        memcpy(s->mtp_chain, s->mtp_verify_hidden + (size_t)n_accept * dim,
              (size_t)dim * sizeof(float));
    }

    *out_n_accept = n_accept;
    *out_new_pos = anchor_pos + 1 + n_accept;
    *out_k_eff = k_eff;

    /* Catch-up decode (independent audit finding, confirmed against
       llama.cpp's own speculative.cpp, which runs it unconditionally
       whenever the draft model's KV is not literally shared with the
       target's): re-decode
       the ACCEPTED span's MTP rows using the TARGET model's own
       verified hidden states, replacing what the draft loop above just
       wrote using chained/estimated hidden for every step AFTER THE
       FIRST (lz_mtp_draft_step's own contract: step i>=1 reads
       s->mtp_chain as ITS OWN previous step's output, an estimate, not
       a real forward - only step i==0 used a genuine forward's hidden,
       s->mtp_chain as it stood at this round's own start). Without
       this, those estimated rows stay in the MTP's KV permanently -
       forward_attn's own `t <= pos` causal window means EVERY later
       round's draft chain attends over them as history, forever, and
       the ground truth that verify just computed for these exact
       positions (s->mtp_verify_hidden) never overwrites them (the
       "capture is redundant" optimization did not create this gap - it
       was already there; that change only stopped re-deriving row 0's
       own hidden a second time).

       ROW 0 (MTP index s->mtp_pos, from draft step i==0) is EXCLUDED,
       on purpose - including it MEASURED WORSE aggregate alpha (6
       falsify prompts, -n 60: k=2 0.7088 -> 0.6701, k=4 0.5143 ->
       0.4840, catch-up on vs off) - a real, reproducible regression,
       not noise. The error: row 0's
       INPUT was already real (hidden@anchor_pos from s->mtp_chain,
       token= anchor@anchor_pos+1) and its OWN prediction was
       independently VERIFIED correct (draft[0]==out[0], or there
       would be no accepted span to catch up at all) - overwriting it
       with (verify_hidden[0]=hidden@anchor_pos+1, out[0]@anchor_pos+2)
       does not add information, it DISCARDS a confirmed-correct row
       and replaces it with a different one, shifted one position
       later, that no longer covers the (hidden@anchor_pos, anchor)
       pairing at all - not a superset, a swap.

       So only rows s->mtp_pos+1 .. s->mtp_pos+n_accept-1 (draft steps
       i=1..n_accept-1, the ones that genuinely used an ESTIMATE) get
       replaced - n_accept-1 rows, not n_accept. Row s->mtp_pos+i needs
       (real hidden@anchor_pos+i, token@anchor_pos+i+1) - "verify row i
       is at body position anchor_pos+1+i" (s->mtp_verify_hidden's own
       comment, forward.h) means hidden@anchor_pos+i is verify_hidden
       [i-1], and the confirmed token that followed it is out[i-1] (out
       [j] is confirmed token at body position anchor_pos+2+j - see
       lz_spec_accept's own contract). Reindexed with j=i-1 (j=0..
       n_accept-2): row s->mtp_pos+1+j needs (verify_hidden[j], out[j])
       - exactly lz_mtp_prefill's own h_body_all[j]/next_tokens[j] shape
       with n=n_accept-1, pos0=s->mtp_pos+1, and s->mtp_verify_hidden/
       out passed AS-IS (no offset - the shift lives entirely in pos0
       and n, not in which slice of the arrays gets read).

       n_accept <= 1: at most row 0 exists in the accepted span, and
       row 0 never needs catch-up - skip entirely (also avoids calling
       lz_mtp_prefill with n=0, which its own contract disallows).
       Correct after the rollback+replay branch above too:
       s->mtp_verify_hidden[j] for j<n_accept-1 was captured during the
       ORIGINAL (pre-rollback) verify pass, and that row's computation
       is causally independent of every LATER (rejected, rolled-back)
       row - the same argument the mtp_chain reseed above already
       relies on. */
    if (n_accept > 1 && !debug_skip_catchup) {
        if (lz_mtp_catchup(m, s, s->mtp_verify_hidden, out, n_accept - 1,
                           s->mtp_pos + 1) != 0) {
            LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
            return -rc;
        }
    }

    /* The MTP's own position counter advances by n_accept, NOT by k_eff
       (and never by the original k when p_min shortened the draft) -
       see s->mtp_pos's comment (forward.h) and this function's module
       comment: a rejected draft step's row must never be attended over
       by a later round, and "don't advance past it" is how that
       rollback is satisfied - no data copy, same shape the body's KV
       cache already uses. */
    s->mtp_pos += n_accept;

    return n_accept + 1;
}

/* Sample one token from a FULL, already-normalized probability vector
   (temp>0 phase 2's own "bonus" token: the row past the
   last accepted draft has no draft counterpart to accept/reject
   against - speculative decoding still gets it for free out of the
   SAME verify batch, same as the greedy path's own bonus token, but
   here it is drawn from the target's distribution directly rather
   than taken as an argmax). Deliberately NOT sampler.c's own static
   sample_mult: that one assumes its caller already has a coin scaled
   to the array's own total mass (lz_sample's own step 7 convention,
   where the candidate set's mass is usually < 1); p here is already a
   full, RENORMALIZED distribution (lz_target_dist's own contract,
   sampler.h) that sums to ~1.0 on its own, so a coin in [0,1) needs no
   extra scaling - reusing sample_mult's contract here would require
   fabricating a "mass" argument that is always ~1.0 anyway. */
static int sample_from_dist(const float *p, int vocab, unsigned long long *rng) {
    float coin = lz_random_f32(rng);
    float cdf = 0.0f;
    int i;
    for (i = 0; i < vocab; i++) {
        cdf += p[i];
        if (coin < cdf) return i;
    }
    return vocab - 1;   /* fp rounding fallback, same convention as sample_mult/sample_residual */
}

/* temp>0 speculative round (phase 2 - the coupled-sampling analog of
   lz_spec_round above). DELIBERATELY A SEPARATE FUNCTION, not a
   temperature branch inside lz_spec_round itself - an explicit
   requirement, given in Chinese and translated here: lz_spec_round does
   not change one character for this phase,
   so phase 1's own hard bit-identity gate (K=1..6 vs K=0 at temp=0,
   under real penalties) stays covered by code that provably cannot
   have been touched by anything written here - not an argument that
   needs re-verifying each time this file changes, a structural fact.

   Consequence: the temperature-INDEPENDENT back half of a round - ring
   rollback, mtp_chain reseed from the verify batch's own captured
   hidden state, catch-up decode, mtp_pos advance - is DUPLICATED here
   rather than shared with lz_spec_round (which would require touching
   lz_spec_round's own body to call out to a shared helper). See that
   function's own module comment for the full derivation of each step;
   this copy must stay in sync with it by hand - a cost accepted
   deliberately in exchange for the structural guarantee above, not an
   oversight.

   Differences from lz_spec_round's own contract:

     samp        REQUIRED (LZ_ERR_NULL_ARG if NULL, unlike lz_spec_
                 round's own optional samp) - temperature, penalties,
                 filters, and the RNG stream this function draws from
                 all live in samp->p and samp->rng_state; there is no
                 meaningful "no sampler" mode once temperature > 0, the
                 way lz_spec_round's own argmax-only path has one.
                 NOT const (lz_spec_round's own samp is): every row's
                 accept test and, on rejection, its residual resample,
                 draw from and advance samp->rng_state - the SAME
                 stream lz_generate_resume's own plain (non-speculative)
                 path already draws from via lz_sample, so a caller
                 that toggles --spec on/off at the same seed still
                 consumes randomness from one continuous stream, not
                 two independently-seeded ones.
     p_min       compared against q(idx), the draft's OWN probability
                 for the token it actually sampled (lz_sample_temp_q's
                 own logits[idx] after the call) - the natural temp>0
                 replacement for lz_argmax_p1's top-1 confidence, which
                 stops meaning "the draft's own choice" once temperature
                 moves sampling off argmax (the same point that motivated
                 a new draft-side primitive at all instead of reusing
                 lz_argmax_p1).
     acceptance  each drafted row is tested with lz_spec_accept_temp
                 (generate.c) against that row's own target distribution
                 p (lz_target_dist, sampler.c - penalties via the SAME
                 assumed window mechanism phase 1 built, applied here
                 too) and draft distribution q (lz_sample_temp_q's own
                 stored output, s->mtp_draft_q) - REJECTION LATCHES,
                 same as lz_spec_accept's own greedy rule: the first
                 rejected row's own resampled token is emitted and
                 every later row (even one lz_spec_accept_temp would
                 have accepted, had it been tested) is discarded
                 unread, because the model never really saw that
                 prefix - a batched forward computed those rows'
                 raw logits regardless (verify runs unconditionally
                 over the whole k_eff+1 batch, same as lz_spec_round),
                 but this function stops CONSUMING them (and stops
                 drawing further random numbers) at the first reject,
                 both for efficiency and so a caller's random stream is
                 not perturbed by rows that were never really sampled.
     bonus       when every drafted row is accepted, the row PAST the
                 last one is not accept/reject tested (no draft
                 counterpart exists for it) - sampled directly from its
                 own target distribution instead (sample_from_dist
                 above), the same thing plain (non-speculative)
                 sampling would have done at that position.

   Parameters/return/error convention otherwise identical to lz_spec_
   round - see that function's own contract (llama_zh.h) for out[],
   out_n_accept, out_new_pos, out_k_eff, the discard-sentinel return of
   0, and the negative-LZErr-on-failure convention. */
int lz_spec_round_temp(const LZModel *m, LZRunState *s,
                       int anchor, int anchor_pos, int k,
                       float p_min, int n_min, LZSampler *samp,
                       int debug_break_rollback, int debug_skip_catchup,
                       int *out, int *out_n_accept, int *out_new_pos,
                       int *out_k_eff, char *errbuf, int errlen) {
    int draft[LZ_SPEC_K_MAX];
    int vocab, n_accept, i, rc, k_eff;
    int base_slot;

    if (!m || !s || !samp || !out || !out_n_accept || !out_new_pos || !out_k_eff) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        return -rc;
    }
    if (!m->mtp) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_SPEC_NO_HEAD);
        return -rc;
    }
    if (k < 1 || k > LZ_SPEC_K_MAX) {
        LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_SPEC_K_RANGE, k, LZ_SPEC_K_MAX);
        return -rc;
    }
    if (k > s->ssm_ring_depth - 1) {
        LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_SPEC_K_RANGE, k, s->ssm_ring_depth - 1);
        return -rc;
    }
    vocab = m->config.vocab_size;

    /* Draft chain: temperature-softmax-sample (lz_sample_temp_q) each
       step instead of argmax (lz_spec_round's own lz_argmax_p1) - see
       this function's own contract above for why p_min's comparison
       target changes to q(idx). Each accepted step's FULL distribution
       is preserved into s->mtp_draft_q (verify needs it again, and the
       draft head's own hidden state has moved on by then - see that
       field's own comment, forward.h). */
    k_eff = 0;
    for (i = 0; i < k; i++) {
        int in_tok = (i == 0) ? anchor : draft[i - 1];
        int idx;
        float *logits = lz_mtp_draft_step(m, s, in_tok, s->mtp_pos + i);
        if (!logits) {
            LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
            return -rc;
        }
        idx = lz_sample_temp_q(logits, vocab, samp->p.temperature, &samp->rng_state);
        if (logits[idx] < p_min) break;
        draft[i] = idx;
        memcpy(s->mtp_draft_q + (size_t)i * vocab, logits, (size_t)vocab * sizeof(float));
        k_eff = i + 1;
    }

    if (k_eff < n_min) {
        /* Discard the whole draft - identical shape to lz_spec_round's
           own n_min branch, see that function's own comment for the
           full argument (nothing here depends on greedy vs temp>0). */
        out[0] = anchor;
        *out_n_accept = 0;
        *out_new_pos = anchor_pos;
        *out_k_eff = k_eff;
        return 0;
    }

    base_slot = s->ssm_slot;

    {
        int verify_tokens[LZ_SPEC_K_MAX + 1];
        verify_tokens[0] = anchor;
        for (i = 0; i < k_eff; i++) verify_tokens[i + 1] = draft[i];
        if (lz_forward_verify(m, s, verify_tokens, k_eff + 1, anchor_pos + 1) != 0) {
            LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
            return -rc;
        }
    }

    /* Per-row coupled accept/residual-resample - see this function's
       own contract above for the rejection-latches / bonus rules. */
    n_accept = k_eff;
    for (i = 0; i < k_eff; i++) {
        float *row = s->mtp_logits + (size_t)i * vocab;
        float u;
        int accepted, tok;
        apply_row_penalties_assumed(samp, draft, i, row);
        lz_target_dist(&samp->p, row, vocab, samp->probindex, s->mtp_target_p);
        u = lz_random_f32(&samp->rng_state);
        tok = lz_spec_accept_temp(s->mtp_target_p, s->mtp_draft_q + (size_t)i * vocab,
                                  draft[i], vocab, u, &samp->rng_state, &accepted);
        out[i] = tok;
        if (!accepted) { n_accept = i; break; }
    }
    if (n_accept == k_eff) {
        /* Every drafted row accepted: the bonus row, sampled (not
           accept/reject tested - no draft counterpart exists for it). */
        float *row = s->mtp_logits + (size_t)k_eff * vocab;
        apply_row_penalties_assumed(samp, draft, k_eff, row);
        lz_target_dist(&samp->p, row, vocab, samp->probindex, s->mtp_target_p);
        out[k_eff] = sample_from_dist(s->mtp_target_p, vocab, &samp->rng_state);
    }

    if (n_accept < k_eff) {
        /* Ring rollback - identical mechanism to lz_spec_round's own,
           see that function's module comment for the full derivation
           (unchanged by temperature: this is purely about WHICH rows
           of the batched verify forward remain "read", never about how
           accept/reject was decided). */
        lz_debug_n_ring_rollback++;
        if (!debug_break_rollback) {
            s->ssm_slot = (base_slot + n_accept + 1) % s->ssm_ring_depth;
        }
    }

    /* mtp_chain reseed from the verify batch's own captured hidden
       state at row n_accept - identical to lz_spec_round's own tail,
       see that function's comment for the full causal-independence
       argument (unaffected by temperature: row n_accept's computation
       never depended on anything past it regardless of how later rows
       get judged). */
    {
        int dim = m->config.hidden_size;
        memcpy(s->mtp_chain, s->mtp_verify_hidden + (size_t)n_accept * dim,
              (size_t)dim * sizeof(float));
    }

    *out_n_accept = n_accept;
    *out_new_pos = anchor_pos + 1 + n_accept;
    *out_k_eff = k_eff;

    /* Catch-up decode - identical to lz_spec_round's own, see that
       function's module comment for the full derivation (row 0
       excluded on purpose, same reasoning, unaffected by temperature -
       catch-up is about which HIDDEN STATE the MTP's own KV holds for
       already-accepted positions, not about how those positions were
       decided to be accepted). */
    if (n_accept > 1 && !debug_skip_catchup) {
        if (lz_mtp_catchup(m, s, s->mtp_verify_hidden, out, n_accept - 1,
                           s->mtp_pos + 1) != 0) {
            LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
            return -rc;
        }
    }

    s->mtp_pos += n_accept;

    return n_accept + 1;
}

void lz_gen_opts_defaults(LZGenOpts *o) {
    if (!o) return;
    memset(o, 0, sizeof(*o));
    lz_sample_defaults(&o->sample);
    o->rng_seed = 0;              /* 0 -> sampler swaps in 1 internally */
    o->max_new_tokens = 0;        /* 0 -> model's seq_len, like num_predict -1 */
    o->n_eos = 0;                 /* filled per-vocab by lz_gen_opts_set_eos */
    o->n_stop = 0;
    /* NOT left at the memset's 0: zero is a valid stop[] index, so a
       caller reading it after a generation that ended on EOS would be
       told stop[0] fired. */
    o->out_stop = -1;
    /* Lookahead off. look_lp is the ONLY one that is not zero-is-off: 0.0
       would divide the score by len^0 == 1, which is a silent identity
       rather than "no length penalty applied", and the two read the
       same in a log. 1.0 is HF's and vLLM's own default. */
    o->look_width = 0;
    o->look_depth = 0;
    o->look_raw = 0;
    o->look_lp = 1.0f;
}

/* Find a token id by byte string in the vocab; -1 if absent.
   Vocab entries may contain arbitrary bytes, so compare by length, not
   strcmp. */
static int find_token(LZTokenizer *t, const char *s) {
    return lz_tokenizer_find(t, s, (int)strlen(s));
}

void lz_gen_opts_set_eos(LZGenOpts *o, LZTokenizer *t) {
    /* Order matters: <|im_end|> ends a chat turn (most common);
       <|endoftext|> ends a document (used in plain continuation).
       Collect both. */
    static const char *names[] = { "<|im_end|>", "<|endoftext|>" };
    int i, id;
    if (!o || !t) return;
    o->n_eos = 0;
    for (i = 0; i < (int)(sizeof(names) / sizeof(names[0])); i++) {
        if (o->n_eos >= LZ_MAX_EOS) break;
        id = find_token(t, names[i]);
        if (id >= 0) o->eos_ids[o->n_eos++] = id;
    }
}

/* --------------------------------------------- dynamic temperature */

/* The think-block tracker lives in generate.h - exported from this
   file so tests can pin it, the same
   arrangement sampler.h uses for its own load-bearing internals. The
   engine cannot link common/stream.c (the display layer, with its GBK
   conversion), so the tracker is a self-contained <think>/</think> tag
   scanner answering the same "inside a think block" question stream.c's
   lz_stream_in_think answers for the display side - both must use the
   same complete-tag rule, and each is pinned by its own tests (iron law
   two's drift warning applies to any second copy of "the same" tag
   logic). */
void lz_think_track_feed(LZThinkTrack *tr, const char *bytes, int len) {
    int i;
    if (!tr || !bytes) return;
    for (i = 0; i < len; i++) {
        tr->pend[tr->n_pend++] = bytes[i];
        /* A complete tag ending at this byte: the window holds the last
           n_pend bytes. The close tag (8) is checked before the open (7)
           so a close cannot be misread as the shorter open. */
        if (tr->n_pend >= LZ_THINK_PEND &&
            memcmp(tr->pend + tr->n_pend - LZ_THINK_PEND, "</think>",
                   LZ_THINK_PEND) == 0) {
            tr->in_think = 0;
        } else if (tr->n_pend >= 7 &&
                   memcmp(tr->pend + tr->n_pend - 7, "<think>", 7) == 0) {
            tr->in_think = 1;
        }
        /* Keep only the newest LZ_THINK_PEND-1 bytes for the next feed. */
        if (tr->n_pend > LZ_THINK_PEND - 1) {
            memmove(tr->pend, tr->pend + 1, (size_t)(LZ_THINK_PEND - 1));
            tr->n_pend = LZ_THINK_PEND - 1;
        }
    }
}

/* Post-process one just-decided token: EOS check, out_tokens recording,
   stop-string feed, cont() callback. Shared by lz_generate_resume's
   ordinary per-token path and its speculative-round path so the two
   cannot drift on what "one token" means for bookkeeping - exactly the
   property temp=0 spec-on/spec-off bit-identity depends on.

   `next_pos` is the position `next` WOULD occupy once forwarded - it
   may not be forwarded yet (matches lz_forward's own "pending token"
   convention: a token is decided, checked, and emitted before the NEXT
   iteration forwards it). `sampled` matches the original inline code's
   own flag: 0 while digesting the prompt (out_tokens must not record
   prompt echo), 1 for anything actually generated - by the model, or by
   the draft head and then accepted/corrected by verify.

   Returns 1 if generation must stop (*out_finish already set), 0 to
   continue. n_rec/out_finish are read-modify-write, like the values
   this replaces in the original inline code. */
static int gen_emit_token(int next, int sampled,
                          const LZGenOpts *opts, LZTokenizer *t,
                          lz_stopf_t *sf, lz_emit_t *em,
                          LZShouldContinue cont, void *ctx,
                          int next_pos, int start_pos, int n_prompt,
                          int *n_rec, int *out_finish,
                          LZThinkTrack *tr) {
    int k, hit = 0;

    for (k = 0; k < opts->n_eos; k++) {
        if (next == opts->eos_ids[k]) { hit = 1; break; }
    }
    if (hit) { *out_finish = LZ_FINISH_EOS; return 1; }

    if (sampled && opts->out_tokens && *n_rec < opts->out_tokens_cap)
        opts->out_tokens[(*n_rec)++] = next;

    if (next_pos >= start_pos + n_prompt) {
        int blen = 0;
        const char *bytes = lz_decode(t, next, &blen);
        /* Dynamic temperature: feed the think-tag scanner. Generated
           output is what moves the <think> state; prompt tokens were
           already scanned once, at the generation entry, so they are not
           re-fed here. */
        lz_think_track_feed(tr, bytes, blen);
        if (lz_stopf_feed(sf, em, bytes, blen)) {
            if (sampled && *n_rec > 0) (*n_rec)--;
            *out_finish = LZ_FINISH_STOP;
            return 1;
        }
    }

    if (cont && !cont(ctx)) { *out_finish = LZ_FINISH_CANCELLED; return 1; }
    return 0;
}

/* Prefill, in slices, so it can report where it is and be stopped.
 *
 * ONE implementation, taking the callback directly: lz_prefix_prepare
 * has no LZGenOpts to read one off, lz_generate_resume_ex does, and
 * that is the whole of the difference between them - see gen_prefill
 * below. Two copies of this loop drifted apart the moment the second
 * knob (cancellation) was added to only one of them.
 *
 * SLICES ARE A MULTIPLE OF s->nt_cap. lz_forward_batch chunks by that
 * width internally, so a slice that is not a multiple would make
 * ceil(n_i/T) sum to more than ceil(n/T) and change lz_debug_n_chunks -
 * the very counter that exists to prove the batching did what it said.
 * The RESULT is unaffected at any slicing, which is not an assumption
 * here but this project's standing "--batch 1..LZ_BATCH_MAX must agree
 * bit for bit" contract restated: where the prompt is cut cannot matter.
 * That contract, not the bypass below, is what makes the slicing safe.
 *
 * The both-NULL bypass is an optimisation, not the bit-identity
 * argument, and it is reachable from fewer places than it looks:
 * anything routed through common/session.c hands down a non-NULL
 * session_cont whether or not the front end set one, so only a caller
 * that passes NULL itself (openai.c's lz_prefix_prepare calls) takes
 * it.
 *
 * Returns 1 forwarded, 0 a forward failure, -1 cancelled. */
static int gen_prefill_raw(const LZModel *m, LZRunState *s,
                           const int *tok, int n, int start_pos,
                           const LZPrefillHooks *h) {
    /* Unpacked once, so the loop below is the same code it was when
       these were three parameters. */
    LZProgress       on_prefill = h ? h->on_prefill : NULL;
    LZShouldContinue cont       = h ? h->cont       : NULL;
    void            *ctx        = h ? h->ctx        : NULL;
    int width, done = 0;

    if (n <= 0) return 1;
    /* Nothing can observe the slicing, so do not pay for it. */
    if (!on_prefill && !cont)
        return lz_forward_batch(m, s, tok, n, start_pos) ? 1 : 0;

    /* Reported before the first slice as well as after each one: a
       prompt shorter than one slice would otherwise produce a single
       callback already saying done == total, and a caller showing an
       indicator while done < total would never show one. */
    if (on_prefill) on_prefill(0, n, ctx);

    width = s->nt_cap > 0 ? s->nt_cap : 1;
    /* Slice at a whole number of batches, and never fewer than one, or
       a large prompt would report progress a handful of tokens at a
       time and spend more on callbacks than on the forward. */
    while (width < 64) width *= 2;

    while (done < n) {
        int take = n - done;
        if (take > width) take = width;
        if (!lz_forward_batch(m, s, tok + done, take, start_pos + done))
            return 0;
        done += take;
        if (on_prefill) on_prefill(done, n, ctx);
        /* Checked AFTER the slice, so a stop always lands on a whole
           slice boundary and the run state is never left mid-batch. */
        if (done < n && cont && !cont(ctx)) return -1;
    }
    return 1;
}

/* The same thing for callers that hold an LZGenOpts. opts is non-NULL
   by the time any call site here runs - lz_generate_resume_ex rejects a
   NULL one at its entry. */
static int gen_prefill(const LZModel *m, LZRunState *s,
                       const int *tok, int n, int start_pos,
                       const LZGenOpts *opts, LZShouldContinue cont,
                       void *ctx) {
    LZPrefillHooks h;
    h.on_prefill = opts->on_prefill;
    h.cont       = cont;
    h.ctx        = ctx;
    return gen_prefill_raw(m, s, tok, n, start_pos, &h);
}

/* The same slicing for the speculative path's body pass, which captures
 * every position's hidden state as it goes for lz_mtp_prefill to
 * consume. It was the one prefill in this file that ran as a single
 * call: no way to report where it was, and the stop button dead for its
 * whole duration - on a long prompt that is most of the wait.
 *
 * lz_forward_batch_capture chunks exactly like lz_forward_batch and
 * writes each chunk's own slice of hidden_out (forward.h), so cutting
 * it at the same whole-batch boundaries gen_prefill_raw uses leaves the
 * chunk sequence, and the result, untouched.
 *
 * Returns 1 forwarded, 0 a forward failure, -1 cancelled. */
static int gen_prefill_capture(const LZModel *m, LZRunState *s,
                               const int *tok, int n, int start_pos,
                               float *h_all, int hidden,
                               const LZPrefillHooks *h) {
    LZProgress       on_prefill = h ? h->on_prefill : NULL;
    LZShouldContinue cont       = h ? h->cont       : NULL;
    void            *ctx        = h ? h->ctx        : NULL;
    int width, done = 0;

    if (n <= 0) return 1;
    if (!on_prefill && !cont)
        return lz_forward_batch_capture(m, s, tok, n, start_pos, h_all)
               ? 1 : 0;

    if (on_prefill) on_prefill(0, n, ctx);
    width = s->nt_cap > 0 ? s->nt_cap : 1;
    while (width < 64) width *= 2;

    while (done < n) {
        int take = n - done;
        if (take > width) take = width;
        if (!lz_forward_batch_capture(m, s, tok + done, take,
                                      start_pos + done,
                                      h_all + (size_t)done * (size_t)hidden))
            return 0;
        done += take;
        if (on_prefill) on_prefill(done, n, ctx);
        if (done < n && cont && !cont(ctx)) return -1;
    }
    return 1;
}

/* Failure exits below use LZ_ERR_SET* (err.h): return code and errbuf
   are set together so an HTTP caller's status can never contradict the
   message it ships alongside it.
   Client's fault (400): LZ_ERR_PROMPT_LONG, LZ_ERR_STOP_LONG,
   LZ_ERR_PROMPT_ENCODE. Ours (500): everything else here. */
int lz_generate_resume_ex(const LZModel *m, LZTokenizer *t, LZRunState *s,
                          int start_pos,
                          const char *prompt_bytes, int prompt_len,
                          const LZGenOpts *opts,
                          LZTokenSink sink, LZShouldContinue cont, void *ctx,
                          int *out_n_tokens, double *out_elapsed_ms,
                          char *errbuf, int errlen, LZInspect *ins) {
    LZSampler sampler;
    int *prompt_tokens = NULL;
    int n_prompt = 0;
    int token, next, pos, generated = 0;
    int n_rec = 0;               /* entries written to opts->out_tokens */
    int sampled;                 /* this step sampled, vs digested a prompt token */
    int max_new;
    int max_pos;                 /* total forward steps cap = prompt digestion + generation */
    double t_start, t_end;
    int rc = LZ_ERR_INTERNAL;
    /* Stop strings match against generated output BYTE-wise, so a tail
       must be kept. Sized for the longest stop string only; no need to
       hold the whole generation. */
    lz_stopf_t sf;
    lz_emit_t em;
    int finish = LZ_FINISH_LENGTH;   /* overwritten by whichever break wins */
    /* Kept out of the filter so the error gotos below - which run before
       the filter is initialised - still leave a defined value. */
    int stop_hit = -1;
    /* Speculative decoding (MTP). spec_active is opts->spec_k > 0, kept
       as its own flag rather than re-reading opts->spec_k everywhere so
       "off" reads as one boolean, not a comparison repeated at every
       call site.

       No LZStateCkpt here (see lz_spec_round's own module comment):
       per-round rollback is a plain s->ssm_slot index assignment, done
       inside lz_spec_round and (for a mid-round stop) in this
       function's own trim branch below - neither needs caller-owned
       scratch. LZPrefixCache's own checkpoint use is separate and
       unrelated. */
    int spec_active;
    /* Bounded lookahead. Zero-initialised so the `done:` cleanup is
       safe on every early goto above, including the ones that run
       BEFORE the allocation - lz_ckpt_free's own contract is that a
       zeroed struct is safe to free. */
    int look_active = 0;
    LZStateCkpt look_ck = {0};
    float *look_scr = NULL;
    /* Acceptance-rate accounting - see LZGenOpts's own comment. */
    int spec_rounds = 0, spec_draft_tokens = 0, spec_accepted = 0;
    /* Does s->mtp_chain already hold a valid "hidden at anchor_pos" seed
       from the PREVIOUS round's own verify capture (the llama.cpp-
       matching behavior - see lz_spec_round's own comment on
       s->mtp_verify_hidden)? Starts false (this call's own
       first round always needs a fresh capture); set true whenever a
       round completes WITHOUT being discarded (lz_spec_round wrote a
       fresh seed then); reset false after any discarded round
       (lz_mtp_draft_step's own chaining clobbers s->mtp_chain even on a
       discarded draft, so a stale leftover value cannot be trusted). */
    int have_seed = 0;
    /* Dynamic temperature: the think-block tracker, see its init below. */
    LZThinkTrack tr;

    if (!m || !t || !s || !opts || !sink) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        return rc;
    }

    /* Hand the inspector to the forward pass. It travels on the run
       state, not on lz_forward's parameter list - see LZRunState's own
       comment for why. Set unconditionally, including to NULL: this
       function can be called on a state a previous _ex call left an
       inspector in, and inheriting that would keep writing into a
       struct whose owner has moved on.
       Cleared again on every exit path below, for the same reason. */
    s->ins = ins;

    max_new = opts->max_new_tokens;
    if (max_new <= 0 || max_new > m->config.seq_len) {
        max_new = m->config.seq_len;
    }

    if (lz_sampler_init(&sampler, m->config.vocab_size, &opts->sample,
                        opts->rng_seed) != 0) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_SAMPLER_INIT);
        return rc;
    }
    /* Penalties reset to zero each generation. Without this, the
       previous turn's history keeps suppressing this one, and a long
       conversation runs out of usable words. */
    lz_sampler_reset(&sampler);

    /* Speculative decoding preconditions, checked once up front - a bad
       spec config fails before any work happens rather than partway
       through a generation. spec_k <= 0 (the default,
       lz_gen_opts_defaults) skips all of this; the loop below is then
       the plain non-speculative path - the "off" half of the temp=0
       spec-on/spec-off bit-identity gate. */
    /* Lookahead is OFF at width <= 1, and that is the definition, not a
       shortcut: with one candidate there is nothing to compare, so the
       honest thing is to fall through to the sampler rather than return
       a top-1 pick that would silently make --temp irrelevant. The
       bit-identity control is therefore `--lookahead 1:D == no flag`,
       and the CONTENT control is `--lookahead W:1 == --temp 0` (depth 1
       scores each candidate by its own logprob alone, so the best is
       the argmax) - two gates, because the first one alone would pass
       for a feature that had been deleted, which was verified by
       mutation rather than assumed. */
    look_active = opts->look_width > 1;
    if (look_active) {
        if (opts->look_width > LZ_LOOK_W_MAX) {
            LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_LOOK_W_RANGE,
                        opts->look_width, LZ_LOOK_W_MAX);
            goto done;
        }
        if (opts->look_depth < 1 || opts->look_depth > LZ_LOOK_D_MAX) {
            LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_LOOK_D_RANGE,
                        opts->look_depth, LZ_LOOK_D_MAX);
            goto done;
        }
        if (lz_ckpt_alloc(&look_ck, m, errbuf, errlen) != 0) {
            rc = LZ_ERR_ALLOC; goto done;
        }
        /* Heap, not a static array: vocab_size is a runtime value, and
           iron law six's "no large buffers on the stack" rules out the
           other option. Paid for only when the knob is on. */
        look_scr = (float *)malloc((size_t)m->config.vocab_size * sizeof(float));
        if (!look_scr) { LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_ALLOC); goto done; }
    }

    spec_active = opts->spec_k > 0;
    if (spec_active) {
        if (!m->mtp) {
            LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_SPEC_NO_HEAD);
            goto done;
        }
        if (opts->spec_k > LZ_SPEC_K_MAX) {
            LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_SPEC_K_RANGE,
                       opts->spec_k, LZ_SPEC_K_MAX);
            goto done;
        }
        /* s (the SSM/conv rollback ring) was sized for AT MOST
           s->ssm_ring_depth-1 - lz_state_alloc's own spec_k_max
           parameter, a sizing hint the CALLER chose, possibly smaller
           than LZ_SPEC_K_MAX to save memory (forward.h's own comment).
           A k that exceeds what was actually allocated would silently
           alias ring slots via `% s->ssm_ring_depth` inside lz_spec_
           round - two logically different rollback targets landing on
           the SAME physical slot - which is corruption, not merely a
           missed optimization, so this refuses instead of clamping or
           guessing. */
        if (opts->spec_k > s->ssm_ring_depth - 1) {
            LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_SPEC_K_RANGE,
                       opts->spec_k, s->ssm_ring_depth - 1);
            goto done;
        }
        /* The call site below DISPATCHES on temperature instead of
           refusing it: temp<=LZ_TEMP_FLOOR calls lz_spec_round
           (unchanged, phase 1's own bit-identity gate keeps covering it
           structurally), temp>LZ_TEMP_FLOOR calls lz_spec_round_temp
           (generate.c) instead. No precondition check is needed here
           for that alone: both functions accept any temperature value,
           they just differ in which one lz_generate_resume calls. */
        /* Penalties are NOT refused: verify applies them to each row
           via an assumed penalty window (lz_spec_round's own &sampler
           argument below, generate.c's build_assumed_window/apply_
           penalties_assumed) instead, so --spec K stays bit-identical
           to --spec 0 under the caller's own requested settings rather
           than needing them at their identity values. */
    }

    /* Count first, then allocate exactly: tokenizer.h's "tokens <= bytes"
       claim is false whenever NFC expands the input (see lz_encode's
       header comment), and this call is the one the chat endpoint
       reaches, so a fixed headroom would be a heap corruption a user can
       trigger by typing Devanagari. */
    n_prompt = lz_encode(t, prompt_bytes, prompt_len, 0, 0, NULL, 0);
    if (n_prompt < 1) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_PROMPT_ENCODE);
        goto done;
    }
    prompt_tokens = (int *)malloc(sizeof(int) * (size_t)n_prompt);
    if (!prompt_tokens) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_PROMPT_BUF);
        goto done;
    }
    /* Qwen's vocab has no BOS convention; encode without BOS and EOS */
    if (lz_encode(t, prompt_bytes, prompt_len, 0, 0,
                  prompt_tokens, n_prompt) != n_prompt) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_PROMPT_ENCODE);
        goto done;
    }
    if (start_pos < 0) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        goto done;
    }
    if (start_pos + n_prompt >= s->seq_len) {
        LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_PROMPT_LONG,
                    start_pos + n_prompt, s->seq_len - 1);
        goto done;
    }

    /* Length of the longest stop string decides how many bytes to hold
       back. A stop string longer than the tail buffer can never match -
       reject it outright rather than silently truncating. */
    /* Its own counter, deliberately not `rc` (the return code). The
       stop-string loop can jump to `done` from INSIDE the loop, and
       reusing rc as the index would leave rc == 0 on that path: the
       first stop string (index 0) would make lz_generate return SUCCESS
       while errbuf said "stop string exceeds 64-byte limit" and
       out_n_tokens was never written. Reachable straight from user
       input on an HTTP endpoint - stop[] comes from the request body. */
    {
        int si;
        for (si = 0; si < opts->n_stop; si++) {
            int sl = opts->stop[si] ? (int)strlen(opts->stop[si]) : 0;
            if (sl > LZ_STOP_TAIL) {
                LZ_ERR_SET1(rc, errbuf, errlen, LZ_ERR_STOP_LONG, LZ_STOP_TAIL);
                goto done;
            }
        }
    }

    em.sink = sink; em.ctx = ctx; em.n_pend = 0;
    lz_stopf_init(&sf, opts->stop, opts->n_stop);

    /* Dynamic temperature state, seeded once per generation:
       `tr` tracks whether generation is currently inside a <think> block.
       Seeded from the prompt's OWN think tags (a thinking turn's rendered
       prompt ends "<think>\n", so generation STARTS inside the block),
       then updated from each generated token's decoded bytes in
       gen_emit_token. */
    memset(&tr, 0, sizeof(tr));
    lz_think_track_feed(&tr, prompt_bytes, prompt_len);

    t_start = lz_time_ms();
    token = prompt_tokens[0];

    /* -n semantics: max_new counts NEWLY GENERATED tokens; the total
       steps add prompt digestion. Capped by the KV cache's seq_len. */
    max_pos = start_pos + n_prompt + max_new;
    if (max_pos > s->seq_len) max_pos = s->seq_len;
    if (max_pos < start_pos + n_prompt + 1) max_pos = start_pos + n_prompt + 1;

    /* prefill: the first n_prompt-1 prompt tokens are digested in a
       batch.

       Their logits are WANTED BY NOBODY - the loop below just takes
       prompt_tokens[pos+1] while pos < n_prompt-1, and sampling starts
       at step n_prompt-1. So the batch runs through and discards the
       intermediate logits, exactly saving lm_head's nt-1 passes.

       The last prompt token still goes through the single-token path;
       the loop structure is untouched: the batch merely replaces "the
       first n_prompt-1 steps" with a faster equivalent, bit-identical
       (forward.h). */
    /* start_pos == 0 means "a fresh sequence", so the run state has to
       BE fresh. Rewriting the KV cache from row 0 is not enough: the GDN
       recurrent state and the conv history are not indexed by position,
       they are accumulated, so re-running from 0 leaves the previous
       call's memory in them.
       Measured on s1v3, same prompt, same seed, temperature 0: the two
       runs diverge at the FIRST generated token and share no prefix
       (non-ASCII output elided - this file is engine layer, iron law 7):
       silently wrong output with no diagnostic. This call is what keeps
       the interactive CLI, which reuses one state across turns at
       start_pos 0, correct. */
    if (start_pos == 0) lz_state_reset(s, m);

    pos = start_pos;
    if (n_prompt > 1) {
        /* TWO branches, on the one question that changes what is
           forwarded: does the MTP head get a real prompt prefill?
           Only spec_active with neither debug knob set does. Every
           other case - spec off, or either knob on - forwards the body
           exactly the same way and differs only in what happens to the
           MTP counter afterwards, so it shares one call.

             spec_active && !skip && !pos_only -> C: lz_mtp_prefill
             everything else                   -> body-only prefill,
                                                  plus the counter jump
                                                  when pos_only asked
                                                  for one

           BOTH report progress and both can be stopped: C's body pass
           goes through gen_prefill_capture, which slices the same way
           gen_prefill_raw does. The MTP pass after it is left whole -
           one block against the body's every layer, and its
           coverage/rebase arithmetic is written against the whole
           range.

           SKIP STILL OUTRANKS POS_ONLY. With both knobs set the old
           chain took the skip branch and never jumped; the jump below
           carries that precedence explicitly, because it is the one
           thing an if/else-if chain expressed for free and a shared
           call does not. */
        if (spec_active && !opts->spec_debug_skip_prefill &&
            !opts->spec_debug_prefill_pos_only) {
            /* Condition C: MTP prompt prefill (filling the gap against
               llama.cpp's speculative.cpp - see forward.h's s->mtp_pos
               comment): run the MTP block over the same
               n_prompt-1 positions the body prefill below covers, using
               the body's OWN hidden state at each one, purely to
               populate the MTP's KV cache before the first round.
               Without this, every generation started the MTP block
               blind (empty KV cache, no prompt history) even though the
               body's own cache is full of it.
               h_body_all is heap, not stack (iron law six) - n_prompt
               is caller/prompt-controlled and can be arbitrarily large,
               unlike the fixed-size scratch in LZRunState. */
            float *h_body_all = (float *)malloc(sizeof(float) *
                (size_t)(n_prompt - 1) * (size_t)m->config.hidden_size);
            if (!h_body_all) {
                LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_PROMPT_BUF);
                goto done;
            }
            {
                LZPrefillHooks h;
                int pf;
                h.on_prefill = opts->on_prefill;
                h.cont       = cont;
                h.ctx        = ctx;
                pf = gen_prefill_capture(m, s, prompt_tokens, n_prompt - 1,
                                         start_pos, h_body_all,
                                         m->config.hidden_size, &h);
                if (pf <= 0) {
                    free(h_body_all);
                    if (pf < 0) {      /* stopped between slices */
                        finish = LZ_FINISH_CANCELLED;
                        rc = LZ_ERR_OK;
                        goto done;
                    }
                    LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
                    goto done;
                }
            }
            /* next_tokens[i] = the actual prompt token that followed
               body position start_pos+i, i.e. prompt_tokens[i+1] - the
               MTP's ground-truth "next token" input (lz_mtp_prefill's
               own comment, forward.c). pos0 = s->mtp_pos (NOT a
               hardcoded 0): the MTP's own position numbering
               accumulates across the whole session (mtp_pos's own
               comment), so a resumed multi-turn generation must
               continue it, not restart it - only a freshly reset state
               (start_pos == 0 above) happens to have mtp_pos == 0
               already.

               spec_debug_prefill_coverage's own comment (llama_zh.h):
               0 (default) covers all n_prompt-1 positions. A positive
               override < n_prompt-1
               computes REAL content via lz_mtp_prefill for only the
               LAST `coverage` prompt positions, but REBASES them to
               MTP positions 0..coverage-1 instead of leaving them at
               their true (large) absolute prompt offset - the earlier
               head positions are not just left zero, they are never
               represented in the MTP's own position numbering AT ALL,
               so no phantom zero rows ever exist for later attention to
               dilute against (a non-rebased version would leave real
               zero rows sitting at low indices, which an unrestricted
               later attention window would still read).
               Rebasing is safe: RoPE's dot product depends only on
               relative distance (q'_p . k'_j depends only on p-j), so
               the relative geometry among the covered tail tokens is
               identical whichever absolute numbering they get -
               statistically equivalent, not bit-identical (the rotated
               K's Q8 quantization rounds differently at a different
               absolute angle). Only correct for a fresh session
               (start_pos == 0, s->mtp_pos == 0 on entry) - multi-turn
               resumption of a rebased prefix is not this investigation's
               concern. */
            {
                int cov = opts->spec_debug_prefill_coverage;
                int partial = (cov > 0 && cov < n_prompt - 1);
                int tail_n = partial ? cov : n_prompt - 1;
                int skip = (n_prompt - 1) - tail_n;
                /* partial coverage rebases to 0..tail_n-1 (this field's
                   own comment above); the default/no-override path
                   keeps the ORIGINAL "continue from wherever s->mtp_pos
                   already was" contract unchanged - rebasing only the
                   test-only partial-coverage path, never the default,
                   so a resumed multi-turn session's ordinary (cov==0)
                   behavior is untouched. */
                int pos0 = partial ? 0 : s->mtp_pos + skip;
                if (lz_mtp_prefill(m, s,
                                   h_body_all + (size_t)skip * m->config.hidden_size,
                                   prompt_tokens + 1 + skip,
                                   tail_n, pos0) != 0) {
                    free(h_body_all);
                    LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
                    goto done;
                }
                s->mtp_pos = partial ? tail_n : s->mtp_pos + n_prompt - 1;
            }
            free(h_body_all);
        } else {
            int pf = gen_prefill(m, s, prompt_tokens, n_prompt - 1,
                                 start_pos, opts, cont, ctx);
            if (pf < 0) {              /* stopped between slices */
                finish = LZ_FINISH_CANCELLED;
                rc = LZ_ERR_OK;
                goto done;
            }
            if (!pf) {
                LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
                goto done;
            }
            /* spec_debug_prefill_pos_only: advance the MTP's own
               position counter to where a full prefill would leave it
               (or to spec_debug_prefill_pos_value's override) WITHOUT
               calling lz_mtp_prefill, so the counter moves but the
               MTP's KV rows for these positions stay at
               lz_state_reset's zero fill.
               spec_debug_prefill_pos_value's own comment (llama_zh.h):
               0 keeps the n_prompt-1 jump; a positive override tests a
               starting position LARGER than the prompt would ever
               produce, still against a zero-filled cache
               (lz_debug_mtp_attn_rows reports the ALU cost of that
               honestly). */
            if (spec_active && !opts->spec_debug_skip_prefill &&
                opts->spec_debug_prefill_pos_only) {
                int jump = (opts->spec_debug_prefill_pos_value > 0)
                          ? opts->spec_debug_prefill_pos_value
                          : n_prompt - 1;
                s->mtp_pos += jump;
            }
        }
        pos = start_pos + n_prompt - 1;
        token = prompt_tokens[n_prompt - 1];
    } else {
        token = prompt_tokens[0];
    }

    while (pos < max_pos - 1) {
        /* Speculative round: only past prompt digestion (the draft head
           itself never runs during prefill, only its KV cache gets
           populated there - see the lz_mtp_prefill call above and
           forward.h's s->mtp_pos comment), and only when there is
           budget for at least a 1-token draft within max_pos. Falls
           back to the ordinary path below otherwise, which is what
           makes the tail of a short generation degrade gracefully
           instead of erroring. */
        int round_k = 0;
        if (spec_active && pos >= start_pos + n_prompt - 1) {
            round_k = opts->spec_k;
            /* The ordinary per-token path below only ever forwards
               positions up to max_pos-2 (the loop condition stops
               issuing new forwards once pos >= max_pos-1). A round's
               last draft token, in the all-accepted case, lands at
               pos+1+round_k - capping it at max_pos-2 keeps the
               speculative path from ever forwarding further than plain
               decoding would have, which would silently generate one
               token more than max_new allows. */
            if (round_k > max_pos - 3 - pos) round_k = max_pos - 3 - pos;
        }

        if (round_k > 0) {
            int anchor_pos, anchor;
            int round_out[LZ_SPEC_K_MAX + 1];
            int n_accept, new_pos, n_emit, k_eff, ei, stopped = 0;
            /* s->ssm_slot as of the START of this round - set fresh
               right before lz_spec_round below (see that function's own
               module comment on the ring). Only the trim branch further
               down actually reads it. */
            int round_base_slot;

            if (have_seed) {
                /* The llama.cpp-matching behavior (measured: capture
                   cost 1.00x a plain forward, EVERY round - see the
                   CLI's own [spec us: ...] line): the PREVIOUS round's
                   own verify batch already forwarded
                   this exact token into the body and captured its raw
                   hidden state into s->mtp_chain (lz_spec_round's own
                   tail). `token` here is that SAME token - its EOS/
                   stop check, penalty-window observe, and generated++
                   were ALL already done by the previous round's own
                   emission loop below (the ei==n_accept iteration) -
                   repeating any of that here would double-count it.
                   anchor_pos is one less than `pos` here (not equal to
                   it, unlike the fresh-capture branch below): `pos` is
                   "decided count" (includes this pending token), while
                   lz_spec_round wants "the last position ACTUALLY
                   forwarded", which is one token earlier - see
                   lz_spec_round's own comment on s->mtp_verify_hidden
                   for why anchor forwarded at anchor_pos+1 lands
                   exactly back on `token`'s own true position. */
                anchor = token;
                anchor_pos = pos - 1;
            } else {
                float *logits = lz_forward_capture(m, s, token, pos);
                anchor_pos = pos;

                if (!logits) {
                    LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
                    goto done;
                }
                anchor = lz_sample_ex(&sampler, logits, ins, tr.in_think);
                lz_sampler_observe(&sampler, anchor);
                generated++;

                if (gen_emit_token(anchor, 1, opts, t, &sf, &em, cont, ctx,
                                   anchor_pos + 1, start_pos, n_prompt,
                                   &n_rec, &finish, &tr)) {
                    /* anchor itself ends generation. It was never
                       forwarded (lz_forward_capture only forwarded
                       `token`, at anchor_pos) - state already matches
                       exactly what was emitted, same as the ordinary
                       path's own "next causes EOS" case below. No
                       draft/verify/rollback needed. */
                    stop_hit = sf.matched;
                    break;
                }
            }

            /* Captured for THIS round's own trim branch below (ei <
               n_accept - 1), which needs the ring's base slot too but
               lives in this function, not lz_spec_round - same value
               either way, since nothing touches s->ssm_slot between
               this read and lz_spec_round's own identical read at its
               own entry (see that function's module comment). */
            round_base_slot = s->ssm_slot;
            /* Dispatch by temperature (phase 2 - see the spec_active
               precondition block's own comment above): lz_spec_round's
               own source is never reached by anything
               temperature does here, it is simply not called when
               temperature is active - the same shape lz_sample itself
               uses to dispatch "greedy" vs "everything else"
               (sampler.c's own `if (!(pr->temperature > LZ_TEMP_
               FLOOR))` branch), not a new pattern invented for this.

               Dynamic temperature: the dispatch AND the round use the
               EFFECTIVE temperature (think-block override). One
               speculative round spans up to k+1 draft/verify tokens
               which can cross a <think> boundary mid-round, so the round
               uses the think state at its START rather than per-token
               (the fine token-level cut is explicitly deferred). The
               override is applied by
               temporarily swapping sampler.p.temperature for the round -
               lz_spec_round_temp reads it via lz_sample_temp_q and
               lz_target_dist - then restoring it, so a round's draft and
               verify distributions agree with the per-token path on the
               same effective temperature. With dynamic temperature off,
               eff_temp IS sampler.p.temperature and the swap is a
               same-value no-op (iron law two). */
            {
                float eff_temp = lz_sample_eff_temp(&sampler.p,
                                                    tr.in_think);
                if (eff_temp > LZ_TEMP_FLOOR) {
                    float saved = sampler.p.temperature;
                    sampler.p.temperature = eff_temp;
                    n_emit = lz_spec_round_temp(m, s, anchor, anchor_pos, round_k,
                                                opts->p_min, opts->n_min, &sampler,
                                                opts->spec_debug_break_rollback,
                                                opts->spec_debug_skip_catchup,
                                                round_out, &n_accept, &new_pos, &k_eff,
                                                errbuf, errlen);
                    sampler.p.temperature = saved;
                } else {
                    n_emit = lz_spec_round(m, s, anchor, anchor_pos, round_k,
                                           opts->p_min, opts->n_min, &sampler,
                                           opts->spec_debug_break_rollback,
                                           opts->spec_debug_skip_catchup,
                                           round_out, &n_accept, &new_pos, &k_eff,
                                           errbuf, errlen);
                }
            }
            if (n_emit < 0) { rc = -n_emit; goto done; }
            spec_rounds++;
            spec_draft_tokens += k_eff;   /* NOT round_k - p_min can make k_eff < round_k, see out_k_eff's own comment */
            spec_accepted += n_accept;
            /* n_emit > 0: lz_spec_round wrote a fresh seed into
               s->mtp_chain from this round's own verify capture (its
               own comment) - safe to skip capture next time. n_emit
               == 0 (discard): verify never ran, and the draft chain's
               own (thrown-away) chaining already clobbered s->mtp_chain
               - force a fresh capture for the next attempt. */
            have_seed = (n_emit > 0);

            /* n_emit == 0: n_min discarded the whole draft (lz_spec_
               round's own comment). anchor was already emitted above;
               round_out[0] (== anchor, per that function's contract)
               must NOT be emitted a second time, so the accept/emit
               loop below is skipped entirely for this round. */
            if (n_emit > 0) {
            for (ei = 0; ei <= n_accept; ei++) {
                lz_sampler_observe(&sampler, round_out[ei]);
                generated++;
                if (gen_emit_token(round_out[ei], 1, opts, t, &sf, &em, cont, ctx,
                                   anchor_pos + 2 + ei, start_pos, n_prompt,
                                   &n_rec, &finish, &tr)) {
                    pos = anchor_pos + 2 + ei;
                    /* round_out[n_accept-1] (the LAST token this round
                       actually forwarded, when n_accept>0) and
                       round_out[n_accept] (the correction/bonus, NEVER
                       forwarded - see lz_spec_round) both already leave
                       s exactly at `pos`. Anything EARLIER than that
                       means s holds MORE tokens than were just emitted
                       (later accepted drafts, forwarded before we knew
                       this one would end generation) - trim: point the
                       SSM/conv ring index at the slot that already
                       holds "state after (anchor + round_out[0..ei])
                       tokens forwarded" - ei+2 tokens - so a later
                       resume cannot see phantom history past the true
                       end. No restore, no replay (the ring - see
                       lz_spec_round's own module comment for the full
                       mechanism): lz_spec_round's own verify batch
                       already forwarded the FULL k_eff+1-token draft
                       and left every intermediate state in the ring
                       (round_base_slot+1 .. round_base_slot+k_eff+1,
                       one slot per token count) as a side effect - this
                       trim only ever asks for a SMALLER count
                       (ei+2 <= n_accept < k_eff+1) than lz_spec_round's
                       own rollback already might have used, so the slot
                       it wants was always there to begin with, never
                       overwritten by anything this round did (a later,
                       now-irrelevant row never contaminates an earlier
                       one - forward_attn's/the recurrence's own
                       `t <= pos`-shaped loops). */
                    if (ei < n_accept - 1) {
                        s->ssm_slot = (round_base_slot + ei + 2) % s->ssm_ring_depth;
                        /* s->mtp_pos MUST shrink by the same shortfall
                           the body's own state just did (an independent
                           audit's finding: a trim that rolled back the
                           body but left s->mtp_pos untouched would
                           silently over-count forever after - LZRunState
                           persists across turns for both -i and the HTTP
                           endpoint, so the drift never self-heals).

                           THE INVARIANT: lz_spec_round's own tail already
                           advanced s->mtp_pos by the FULL n_accept,
                           unconditionally, before this trim branch ever
                           runs (it has no way to know a stop is coming).
                           That advance counts n_accept MTP rows as
                           "confirmed" - one row per round_out[0..
                           n_accept-1]. But only round_out[0..ei] (ei+1
                           tokens) actually survive being emitted; the
                           rest, round_out[ei+1..n_accept-1] - exactly
                           (n_accept - ei - 1) of them - never reached the
                           caller, same as the body tokens the restore+
                           replay above just discarded. Each of those
                           corresponds 1:1 to one of the MTP rows
                           s->mtp_pos's advance had already counted (draft
                           step i's row is what round_out[i] came from,
                           for i < n_accept), so the counter must give
                           back exactly that many.

                           Verified against the two cases that must NOT
                           trigger a correction, both already excluded by
                           this `if`: ei == n_accept-1 (every accepted
                           draft was emitted, 0 shortfall, and indeed
                           n_accept-ei-1 == 0) and ei == n_accept (the
                           bonus token itself, never forwarded into the
                           body either - see lz_spec_round's out_new_pos
                           comment - so it was never counted in
                           s->mtp_pos's advance to begin with, again 0
                           shortfall, and this whole `if` is false for
                           ei==n_accept since n_accept < n_accept-1 is
                           never true). */
                        s->mtp_pos -= (n_accept - ei - 1);
                    }
                    stopped = 1;
                    stop_hit = sf.matched;
                    break;
                }
            }
            if (stopped) break;
            }   /* end if (n_emit > 0) - discard case skips straight to
                   the generic pos/token bookkeeping below, which already
                   works unchanged for it (see lz_spec_round's discard-path
                   output contract: new_pos=anchor_pos, out[0]=anchor). */

            /* new_pos is FORWARDED count (anchor_pos + 1 + n_accept) -
               lz_spec_round's own contract, matching lz_ckpt_save's "pos
               = tokens forwarded". The outer loop's `pos`, like the
               ordinary path's below, tracks DECIDED count instead: every
               k=0 iteration's `pos++` fires for the just-sampled `next`
               even though it is left pending (not forwarded) when the
               loop exits right after - see that path's own comment.
               round_out[n_accept] (the correction/bonus) is exactly
               that "decided but not forwarded" token here, so the outer
               pos must count it too: new_pos + 1, not new_pos. Getting
               this wrong does not corrupt the KV/SSM state (lz_spec_
               round already left it correct) - it desyncs `pos` from
               `generated`, so max_new stops bounding how many tokens
               come out (caught empirically: -n 24 produced 46 tokens
               with --spec 1 because pos undercounted by exactly 1 per
               round and the loop kept running). */
            pos = new_pos + 1;
            token = round_out[n_accept];   /* pending, like the ordinary path's `next` */
            continue;
        }

        {
        float *logits = lz_forward(m, s, token, pos);
        if (!logits) {
            LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_FORWARD);
            goto done;
        }

        sampled = 0;
        if (pos < start_pos + n_prompt - 1) {
            next = prompt_tokens[pos + 1 - start_pos]; /* digesting the prompt */
        } else {
            if (look_active) {
                /* The lookahead picks; the sampler is bypassed for THIS
                   token, which is the point - a rollout that then
                   sampled from the full distribution would be measuring
                   the sampler, not the search. Penalties still apply,
                   inside lz_look_pick, through the same assumed-window
                   path the speculative verify uses. */
                next = lz_look_pick(m, s, &sampler, opts, &look_ck, look_scr,
                                    logits, pos, m->config.vocab_size,
                                    errbuf, errlen);
                if (next < 0) { rc = LZ_ERR_FORWARD; goto done; }
            } else
            next = lz_sample_ex(&sampler, logits, ins, tr.in_think);
            /* Record into penalty counts. Only GENERATED tokens - words
               in the prompt are often exactly what the answer needs;
               penalizing them backfires. Same semantics as the Python
               side; the two must agree. */
            lz_sampler_observe(&sampler, next);
            generated++;
            sampled = 1;
        }
        pos++;

        /* Termination tokens, stop strings, out_tokens recording and the
           cont() callback: gen_emit_token above (shared with the
           speculative round path so the two bookkeeping paths cannot
           drift - see its own comment). NOTE the last entry is sampled
           but never forwarded (the loop breaks before its lz_forward),
           so it is NOT in the KV either - that is why a resume has to
           re-send it; see llama_zh.h. */
        if (gen_emit_token(next, sampled, opts, t, &sf, &em, cont, ctx,
                           pos, start_pos, n_prompt, &n_rec, &finish, &tr)) {
            stop_hit = sf.matched;
            break;
        }
        token = next;
        }
    }

    lz_stopf_flush(&sf, &em);
    lz_emit_flush(&em);

    t_end = lz_time_ms();
    if (out_n_tokens) *out_n_tokens = generated;
    ((LZGenOpts *)opts)->out_finish = finish;
    ((LZGenOpts *)opts)->out_prompt_tokens = n_prompt;
    stop_hit = sf.matched;
    if (out_elapsed_ms) *out_elapsed_ms = t_end - t_start;
    rc = 0;

done:
    lz_ckpt_free(&look_ck);
    free(look_scr);
    /* Set on EVERY exit, including the error gotos above: a caller that
       reuses one LZGenOpts across turns would otherwise read the
       PREVIOUS call's count after a failure. opts is const by signature,
       but out_tokens_n is a caller output slot exactly like the
       out_tokens[] array it counts. */
    if (opts) {
        ((LZGenOpts *)opts)->out_tokens_n = opts->out_tokens ? n_rec : 0;
        ((LZGenOpts *)opts)->out_stop = stop_hit;
        ((LZGenOpts *)opts)->out_spec_rounds = spec_rounds;
        ((LZGenOpts *)opts)->out_spec_draft_tokens = spec_draft_tokens;
        ((LZGenOpts *)opts)->out_spec_accepted = spec_accepted;
    }
    free(prompt_tokens);
    lz_sampler_free(&sampler);
    /* The inspector belongs to the caller and is very often a local -
       gui/session.c passes &s->ins from a struct the worker owns. Leaving
       the pointer behind would let a later lz_forward on this same run
       state (a prefill, a checkpoint replay, anything) write into it long
       after the caller stopped reading, and nothing would say so. */
    s->ins = NULL;
    return rc;
}

/* Compatibility wrapper: lz_generate_resume_ex(..., NULL) - a thin
   forward, not a second copy of the loop above, so the two can never
   drift apart on the token sequence they produce. Every existing caller
   (the CLI, the HTTP endpoint, this file's own lz_generate below, every
   test) keeps calling this exact name and never pays for a caller that
   might want to watch. */
int lz_generate_resume(const LZModel *m, LZTokenizer *t, LZRunState *s,
                       int start_pos,
                       const char *prompt_bytes, int prompt_len,
                       const LZGenOpts *opts,
                       LZTokenSink sink, LZShouldContinue cont, void *ctx,
                       int *out_n_tokens, double *out_elapsed_ms,
                       char *errbuf, int errlen) {
    return lz_generate_resume_ex(m, t, s, start_pos, prompt_bytes,
                                 prompt_len, opts, sink, cont, ctx,
                                 out_n_tokens, out_elapsed_ms,
                                 errbuf, errlen, NULL);
}

/* Compatibility wrapper: start_pos=0 behaves exactly like the original
   single-shot generation. */
int lz_generate(const LZModel *m, LZTokenizer *t, LZRunState *s,
                const char *prompt_bytes, int prompt_len,
                const LZGenOpts *opts,
                LZTokenSink sink, LZShouldContinue cont, void *ctx,
                int *out_n_tokens, double *out_elapsed_ms,
                char *errbuf, int errlen) {
    return lz_generate_resume(m, t, s, 0, prompt_bytes, prompt_len,
                              opts, sink, cont, ctx,
                              out_n_tokens, out_elapsed_ms,
                              errbuf, errlen);
}

/* ---------------------------------------------------- prefix reuse */

int lz_prefix_init(LZPrefixCache *pc, const LZModel *m,
                   char *errbuf, int errlen) {
    int rc;
    if (!pc || !m) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    memset(pc, 0, sizeof(*pc));
    rc = lz_ckpt_alloc(&pc->ck, m, errbuf, errlen);
    if (rc != LZ_ERR_OK) return rc;
    return LZ_ERR_OK;
}

void lz_prefix_free(LZPrefixCache *pc) {
    if (!pc) return;
    lz_ckpt_free(&pc->ck);
    free(pc->tok);
    free(pc->cur);
    free(pc->pre);
    free(pc->tail);
    memset(pc, 0, sizeof(*pc));
}

void lz_prefix_reset(LZPrefixCache *pc) {
    if (!pc) return;
    pc->n = 0;
    pc->bytes = 0;
    pc->have = 0;
}

/* Grow the four parallel int buffers together. Together, because sizing
   them separately is how one of them ends up one call behind the others
   and the mismatch shows up as a buffer overrun rather than as an error. */
static int prefix_grow(LZPrefixCache *pc, int need) {
    int *a, *b, *c, *d;
    if (pc->cap >= need) return 1;
    a = (int *)realloc(pc->tok,  (size_t)need * sizeof(int));
    b = (int *)realloc(pc->cur,  (size_t)need * sizeof(int));
    c = (int *)realloc(pc->pre,  (size_t)need * sizeof(int));
    d = (int *)realloc(pc->tail, (size_t)need * sizeof(int));
    if (a) pc->tok  = a;
    if (b) pc->cur  = b;
    if (c) pc->pre  = c;
    if (d) pc->tail = d;
    if (!a || !b || !c || !d) return 0;
    pc->cap = need;
    return 1;
}

int lz_prefix_match(const LZPrefixCache *pc, const int *pre, int n_pre) {
    if (!pc || !pre || n_pre <= 0) return 0;
    if (!pc->have || pc->n <= 0 || pc->n > n_pre) return 0;
    if (memcmp(pre, pc->tok, (size_t)pc->n * sizeof(int)) != 0) return 0;
    return pc->n;
}

int lz_prefix_prepare(LZPrefixCache *pc, const LZModel *m, LZTokenizer *t,
                      LZRunState *s, const char *render, int render_len,
                      int split, int *out_start_pos, int *out_suffix_off,
                      int *out_reused, const LZPrefillHooks *hooks,
                      char *errbuf, int errlen) {
    int n_cur, n_pre, n_tail, base = 0;
    int tail_len;

    if (out_start_pos) *out_start_pos = 0;
    if (out_suffix_off) *out_suffix_off = 0;
    if (out_reused) *out_reused = 0;

    if (!pc || !m || !t || !s || !render || render_len < 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    tail_len = render_len - split;
    /* Nothing to reuse, or a split that is not inside the render: take
       the full path. Not an error - the caller's two-line shape stays
       the same either way. */
    pc->n_calls++;
    if (split <= 0 || tail_len <= 0) return LZ_ERR_OK;
    if (!prefix_grow(pc, render_len + 32)) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_PROMPT_BUF);
        return LZ_ERR_PROMPT_BUF;
    }

    /* snprintf retry: encode into what is allocated, and if any of the
       three reports MORE tokens than fit, grow to the reported size and
       redo. render_len + 32 is right for every input seen in practice but
       is not a bound - NFC expansion breaks it (lz_encode's header), and a
       silently truncated pre/tail here would not crash, it would compare
       equal on a prefix and reuse a state that belongs to different text. */
    n_cur  = lz_encode(t, render, render_len, 0, 0, pc->cur, pc->cap);
    n_pre  = lz_encode(t, render, split, 0, 0, pc->pre, pc->cap);
    n_tail = lz_encode(t, render + split, tail_len, 0, 0, pc->tail, pc->cap);
    if (n_cur > pc->cap || n_pre > pc->cap || n_tail > pc->cap) {
        int need = n_cur;
        if (n_pre > need) need = n_pre;
        if (n_tail > need) need = n_tail;
        if (!prefix_grow(pc, need + 32)) {
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_PROMPT_BUF);
            return LZ_ERR_PROMPT_BUF;
        }
        n_cur  = lz_encode(t, render, render_len, 0, 0, pc->cur, pc->cap);
        n_pre  = lz_encode(t, render, split, 0, 0, pc->pre, pc->cap);
        n_tail = lz_encode(t, render + split, tail_len, 0, 0, pc->tail, pc->cap);
    }

    /* Check 1: the split is token-exact. BPE merges across a segment
       boundary, so this cannot be assumed from the split landing after a
       special token - it has to be shown, per render. If it does not
       hold, forwarding head and tail separately would feed the model a
       different token stream than the full render, i.e. silently change
       the answer. */
    if (n_cur <= 0 || n_pre <= 0 || n_tail <= 0 ||
        n_pre + n_tail != n_cur ||
        memcmp(pc->cur, pc->pre, (size_t)n_pre * sizeof(int)) != 0 ||
        memcmp(pc->cur + n_pre, pc->tail, (size_t)n_tail * sizeof(int)) != 0) {
        pc->n_unsplittable++;
        lz_prefix_reset(pc);
        return LZ_ERR_OK;               /* full path */
    }

    /* Check 2: does the cached prefix match THIS render, token-wise?
       Check 3 (the state was not reset since) is enforced inside
       lz_ckpt_restore via the epoch counter. */
    if (lz_prefix_match(pc, pc->cur, n_pre) > 0) {
        int p = 0;
        if (lz_ckpt_restore(&pc->ck, s, m, &p, errbuf, errlen) == LZ_ERR_OK &&
            p == pc->n) {
            base = pc->n;
            pc->n_hits++;
            pc->n_tokens_saved += base;
        } else {
            pc->have = 0;               /* stale; rebuild from scratch */
        }
    } else if (pc->have) {
        /* Had a cached prefix and this render is not built on it. This
           is the counter that decides the multi-slot question: a
           conversation interleaved with another shows up here every
           time, while a single conversation appending turns never does. */
        pc->n_mismatch++;
    }
    if (base == 0) lz_state_reset(s, m);

    /* SLICED, and reported, like the prefill in lz_generate_resume_ex.
       On a cache miss base is 0 and this forwards the WHOLE reusable
       prefix - which in a prefix-reuse front end is nearly the whole
       prompt, and is where the wait actually is. Leaving it
       uninstrumented meant the indicator saw only the generation-prompt
       tail the resume path forwards afterwards: a handful of tokens,
       done instantly, so nothing was ever drawn. */
    if (n_pre > base) {
        int pf = gen_prefill_raw(m, s, pc->cur + base, n_pre - base, base,
                                 hooks);
        if (pf <= 0) {
            /* Either way the state is half-forwarded and the cache would
               describe a prefix that is not there, so both are torn
               down. The CODES differ because the caller's next move
               does: a forward failure falls back to the full path, a
               cancellation ends the turn. */
            lz_state_reset(s, m);
            lz_prefix_reset(pc);
            if (pf < 0) return LZ_ERR_CANCELLED;
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_FORWARD);
            return LZ_ERR_FORWARD;
        }
    }

    /* Cache this render's prefix for the next turn. A failed save is not
       fatal: the state is still correct for generating now, we just lose
       the reuse next time. */
    if (lz_ckpt_save(&pc->ck, s, m, n_pre, errbuf, errlen) == LZ_ERR_OK) {
        memcpy(pc->tok, pc->cur, (size_t)n_pre * sizeof(int));
        pc->n = n_pre;
        pc->bytes = split;
        pc->have = 1;
    } else {
        pc->have = 0;
    }

    if (out_start_pos) *out_start_pos = n_pre;
    if (out_suffix_off) *out_suffix_off = split;
    if (out_reused) *out_reused = base;
    return LZ_ERR_OK;
}

void lz_prefix_stats(const LZPrefixCache *pc, long *calls, long *hits,
                     long *tokens_saved, long *mismatch, long *unsplittable) {
    if (calls)        *calls        = pc ? pc->n_calls : 0;
    if (hits)         *hits         = pc ? pc->n_hits : 0;
    if (tokens_saved) *tokens_saved = pc ? pc->n_tokens_saved : 0;
    if (mismatch)     *mismatch     = pc ? pc->n_mismatch : 0;
    if (unsplittable) *unsplittable = pc ? pc->n_unsplittable : 0;
}

/* ---------------------------------------------------------- session pool */

int lz_pool_init(LZSessionPool *p, const LZModel *m, int n_slots,
                 int seq_len, char *errbuf, int errlen) {
    int i;

    if (!p || !m) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    memset(p, 0, sizeof(*p));
    if (n_slots < 1) n_slots = 1;
    p->slot = (LZSlot *)calloc((size_t)n_slots, sizeof(LZSlot));
    if (!p->slot) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
        return LZ_ERR_STATE_ALLOC;
    }
    /* Allocate every slot up front rather than lazily. On the target the
       failure mode that matters is running out of memory mid-conversation,
       and a pool that only reveals it cannot afford slot 3 on the third
       request has moved an allocation failure from startup into the
       middle of serving. Fail here or not at all. */
    for (i = 0; i < n_slots; i++) {
        /* 0: the HTTP path this pool serves does not wire --spec yet
           (server_main.c never sets LZGenOpts.spec_k) - sizing the ring
           for it here would be memory nobody can reach. Revisit if that
           changes (forward.h's spec_k_max comment). */
        if (lz_state_alloc(&p->slot[i].st, m, seq_len, 0, errbuf, errlen) != 0 ||
            lz_prefix_init(&p->slot[i].pc, m, errbuf, errlen) != LZ_ERR_OK) {
            p->n = i;               /* free what did come up */
            lz_pool_free(p);
            return LZ_ERR_STATE_ALLOC;
        }
        p->n = i + 1;
    }
    return LZ_ERR_OK;
}

void lz_pool_free(LZSessionPool *p) {
    int i;
    if (!p) return;
    for (i = 0; i < p->n; i++) {
        lz_prefix_free(&p->slot[i].pc);
        lz_state_free(&p->slot[i].st);
    }
    free(p->slot);
    free(p->pre);
    memset(p, 0, sizeof(*p));
}

/* Grow the ranking scratch. Separate from LZPrefixCache's own scratch:
   that one belongs to a slot, and ranking happens before a slot is
   chosen. */
static int pool_grow(LZSessionPool *p, int need) {
    int *n;
    if (p->pre_cap >= need) return 1;
    n = (int *)realloc(p->pre, (size_t)need * sizeof(int));
    if (!n) return 0;
    p->pre = n;
    p->pre_cap = need;
    return 1;
}

int lz_pool_prepare(LZSessionPool *p, const LZModel *m, LZTokenizer *t,
                    const char *render, int render_len, int split,
                    LZRunState **out_state, int *out_start_pos,
                    int *out_suffix_off, int *out_reused,
                    const LZPrefillHooks *hooks,
                    char *errbuf, int errlen) {
    int i, best = -1, best_score = 0, n_pre = 0;

    if (out_state) *out_state = NULL;
    if (!p || p->n < 1 || !m || !t || !render) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    p->n_calls++;

    /* Rank on the REUSABLE part only. lz_prefix_prepare compares its
       cached tokens against the full render's tokens, but under its own
       check 1 (the split is token-exact) the two agree on [0, n_pre).
       When check 1 fails prepare takes the full path anyway, so ranking
       here can only be right or conservative - never wrong in a way that
       changes an answer. That asymmetry is why one extra encode is an
       acceptable price for not duplicating prepare's logic. */
    if (split > 0 && split <= render_len && pool_grow(p, render_len + 32)) {
        n_pre = lz_encode(t, render, split, 0, 0, p->pre, p->pre_cap);
        if (n_pre > p->pre_cap) {       /* NFC expanded past the estimate */
            if (pool_grow(p, n_pre + 32))
                n_pre = lz_encode(t, render, split, 0, 0, p->pre, p->pre_cap);
            else
                n_pre = 0;              /* rank as a miss, never on garbage */
        }
    }

    if (n_pre > 0) {
        for (i = 0; i < p->n; i++) {
            int sc = lz_prefix_match(&p->slot[i].pc, p->pre, n_pre);
            /* Strictly greater: ties go to the earlier slot, which is
               arbitrary but deterministic. A tie means two slots hold
               the same prefix, which only happens if a conversation was
               forked - either is correct. */
            if (sc > best_score) { best_score = sc; best = i; }
        }
    }

    if (best >= 0) {
        p->n_hits++;
    } else {
        /* No slot can serve this. Take a never-used one if there is one,
           else the least recently used. */
        int lru = 0;
        for (i = 0; i < p->n; i++) {
            if (p->slot[i].used == 0) { lru = i; break; }
            if (p->slot[i].used < p->slot[lru].used) lru = i;
        }
        best = lru;
        p->n_cold++;
        /* Evicting a slot that still held a usable prefix is the signal
           that the pool is too small - count it, do not guess at it. */
        if (p->slot[best].pc.have && p->slot[best].pc.n > 0)
            p->n_evict_useful++;
    }

    p->slot[best].used = ++p->clock;
    if (out_state) *out_state = &p->slot[best].st;
    return lz_prefix_prepare(&p->slot[best].pc, m, t, &p->slot[best].st,
                             render, render_len, split, out_start_pos,
                             out_suffix_off, out_reused, hooks,
                             errbuf, errlen);
}

void lz_pool_stats(const LZSessionPool *p, long *calls, long *hits,
                   long *cold, long *evict_useful, long *bytes) {
    long b = 0;
    int i;
    if (p) for (i = 0; i < p->n; i++) b += (long)p->slot[i].st.bytes_alloc;
    if (calls)        *calls        = p ? p->n_calls : 0;
    if (hits)         *hits         = p ? p->n_hits : 0;
    if (cold)         *cold         = p ? p->n_cold : 0;
    if (evict_useful) *evict_useful = p ? p->n_evict_useful : 0;
    if (bytes)        *bytes        = b;
}
