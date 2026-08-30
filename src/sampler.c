#include <math.h>
#include "lz_mathf.h"   /* lz_powf: float, no libm, bit-identical x86/ARM */
#include <stdlib.h>
#include <string.h>

#include "ops.h"
#include "err.h"
#include "sampler.h"

void lz_sample_defaults(LZSampleParams *p) {
    /* Qwen-official suggestions for non-thinking (instruction) mode,
       with two deliberate deviations (temperature, repetition_penalty).
       Both are measured; the record is in the header. */
    p->temperature        = 0.6f;   /* DEVIATES from Qwen's 0.7 */
    /* Dynamic temperature: the value kept at the spec's suggested
       number, but the ENABLE FLAG is 0 - the feature is off unless a
       caller explicitly turns it on (CLI: --think-temp). This is what
       keeps a defaults-built struct on the plain temperature path, bit
       for bit. */
    p->temp_think         = 0.3f;
    p->think_temp_enabled = 0;
    p->topp               = 0.8f;
    /* DEVIATES from Qwen's 0.0, and the UNIT is llama.cpp's: 0.05 here
       means "keep tokens within 5% of the RAW peak", the number the
       owner has always used. At T=0.6 that converts to 0.0068 in this
       engine's post-temperature units - typing 0.05 natively would be
       3.3x stronger than intended. min_p is also what pays for the low
       temperature: without it, 0.7->0.5 costs +3 degenerate of 40; with
       it, +1. Header has the grid. */
    p->minp               = 0.05f;
    p->minp_llamacpp      = 1;
    p->topk               = 20;
    p->presence_penalty   = 1.5f;
    p->frequency_penalty  = 0.0f;
    /* DEVIATES from Qwen's 1.0, on a measurement: degenerate output
       57.5% -> 38.8% over 80 prompts (~2.4 sigma). The official set has
       no escalating anti-repetition term at all; this puts it back.
       Full record + re-measure trigger in the header. */
    p->repetition_penalty = 1.1f;
    p->repeat_last_n      = 64;     /* matches llama.cpp's same-named param */
}

void lz_sample_defaults_think(LZSampleParams *p) {
    /* Qwen splits thinking in two - general at temperature 1.0, precise
       code tasks at 0.6. This engine has ONE thinking preset and always
       will: a 57.6M model has no coding tier to speak of, so there is
       nothing for a second row to serve.

       temperature 0.6 MATCHES Qwen's coding row and this engine's own
       instruct preset. Qwen's general 1.0 works because a full-scale
       model's reasoning pass absorbs the sampling randomness afterwards
       - "brainstorm freely, then tidy up" - but at 0.8 this model's
       argmax is drowned by sampling noise on factual turns (it answered
       "the capital of the United States is Paris, France" at temp 0.8
       and correctly at temp 0), so 0.6 keeps the argmax. The thinking
       preset keeps its looseness from top_p 0.95, not from temperature.

       presence 1.5 MATCHES Qwen's coding row. The coding-row rationale
       says a chain of thought revisits the same entities on purpose, so
       a binary presence penalty should make it restate them in worse
       words; measured on the first thinking model (rl_r1), that fear
       did not materialize: with presence at 0 the degenerate-answer
       rate was 14-17/42 vs 3-5/42 with 1.5, and 1.5 introduced almost
       no new degeneracy (1/42). The failure mode this model actually
       has is running away when nothing pushes back - not being bruised
       by a fair re-mention penalty. */
    p->temperature        = 0.6f;
    /* Dynamic temperature: same OFF-by-default stance as the
       instruct preset. The thinking preset does NOT turn think-block low
       temperature on for you - that is a behavioural change the owner must
       opt into (--think-temp), not something a preset decides. */
    p->temp_think         = 0.3f;
    p->think_temp_enabled = 0;
    p->topp               = 0.95f;
    /* Same llama.cpp unit as the instruct preset. At T=1.0 the
       conversion is the identity, so the flag is documentation here
       rather than arithmetic - but it must still be set, or a caller
       lowering the temperature would silently switch unit systems. */
    p->minp               = 0.05f;
    p->minp_llamacpp      = 1;
    p->topk               = 20;
    p->presence_penalty   = 1.5f;
    p->frequency_penalty  = 0.0f;
    /* 1.1 like the instruct preset. Measured on the pilot model;
       carried into the thinking preset as the escalating anti-repetition
       term, now alongside presence 1.5 rather than as its sole
       replacement. It does mildly penalize the legitimate re-mention a
       chain of thought wants, but multiplicative 1.1 over a 64-token
       window is far gentler than additive 1.5. */
    p->repetition_penalty = 1.1f;
    p->repeat_last_n      = 64;
}

void lz_sample_apply_think_preset(LZSampleParams *dst, unsigned manual) {
    LZSampleParams t;
    lz_sample_defaults_think(&t);
    if (!(manual & LZ_MANUAL_TEMP))  dst->temperature        = t.temperature;
    if (!(manual & LZ_MANUAL_TOPP))  dst->topp               = t.topp;
    if (!(manual & LZ_MANUAL_MINP)) { dst->minp             = t.minp;
                                      dst->minp_llamacpp    = t.minp_llamacpp; }
    if (!(manual & LZ_MANUAL_TOPK))  dst->topk               = t.topk;
    if (!(manual & LZ_MANUAL_PRES))  dst->presence_penalty  = t.presence_penalty;
    if (!(manual & LZ_MANUAL_FREQ))  dst->frequency_penalty = t.frequency_penalty;
    if (!(manual & LZ_MANUAL_REP))   dst->repetition_penalty = t.repetition_penalty;
    if (!(manual & LZ_MANUAL_RLAST)) dst->repeat_last_n     = t.repeat_last_n;
    if (!(manual & LZ_MANUAL_THINK_TEMP)) { dst->temp_think = t.temp_think;
                                            dst->think_temp_enabled = t.think_temp_enabled; }
}

int lz_sampler_init(LZSampler *s, int vocab_size, const LZSampleParams *p,
                    lz_u64 seed) {
    int cap;
    memset(s, 0, sizeof(*s));
    s->vocab_size = vocab_size;
    if (p) {
        s->p = *p;
    } else {
        lz_sample_defaults(&s->p);
    }
    s->rng_state = seed ? seed : LZ_U64_C(1);
    s->probindex = (LZProbIndex *)malloc((size_t)vocab_size *
                                         sizeof(LZProbIndex));
    /* Penalty window.
       Do NOT clamp by vocab size - the ring buffer stores positions, not
       distinct tokens; 64 slots can hold the same token 64 times.
       Clamping by vocab truncates frequency counts on small vocabs
       (measured). */
    cap = s->p.repeat_last_n;
    if (cap < 0) cap = LZ_PENALTY_MAX_WINDOW;   /* -1: whole generation counts */
    if (cap < 1) cap = 1;                       /* keep 1 even when off, avoid div-by-zero */
    s->cap = cap;
    s->head = 0;
    s->n_win = 0;
    /* counts is a vocab-sized table; 64 KB at 32768 tokens. Acceptable
       on target hardware, but zeroing the whole table every token is
       not - so clearing only walks the ring buffer's few dozen slots. */
    s->counts = (unsigned short *)calloc((size_t)vocab_size,
                                         sizeof(unsigned short));
    s->ring = (int *)malloc((size_t)cap * sizeof(int));
    if (!s->probindex || !s->counts || !s->ring) {
        lz_sampler_free(s);
        return 1;
    }
    return 0;
}

void lz_sampler_free(LZSampler *s) {
    if (!s) return;
    free(s->probindex);
    free(s->counts);
    free(s->ring);
    s->probindex = NULL;
    s->counts = NULL;
    s->ring = NULL;
    s->n_win = 0;
}

void lz_sampler_reset(LZSampler *s) {
    int i, idx;
    if (!s || !s->counts || !s->ring) return;
    for (i = 0; i < s->n_win; i++) {
        idx = s->ring[(s->head + i) % s->cap];
        s->counts[idx] = 0;
    }
    s->head = 0;
    s->n_win = 0;
}

void lz_sampler_observe(LZSampler *s, int token) {
    int old;
    if (!s || !s->counts || !s->ring) return;
    if (token < 0 || token >= s->vocab_size) return;
    if (s->p.repeat_last_n == 0) return;      /* off: don't record */
    if (s->n_win == s->cap) {                 /* window full: evict oldest first */
        old = s->ring[s->head];
        if (s->counts[old] > 0) s->counts[old]--;
        s->head = (s->head + 1) % s->cap;
        s->n_win--;
    }
    s->ring[(s->head + s->n_win) % s->cap] = token;
    s->n_win++;
    if (s->counts[token] < 0xFFFF) s->counts[token]++;
}

static unsigned int random_u32(lz_u64 *state) {
    /* xorshift64* - matches llama2.c so the same seed reproduces */
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (unsigned int)((*state * LZ_U64_C(0x2545F4914F6CDD1D)) >> 32);
}

/* Not static: lz_sample_temp_q (below) and generate.c's
   lz_spec_accept_temp both need to draw from this SAME PRNG - a
   xorshift64* generator has no state to "share" other than the u64
   itself, so exporting the draw function is simpler and safer than
   duplicating xorshift64*'s five-line body a second time - two
   independent copies of "the same" arithmetic drift. random_u32 above
   stays static/private: nothing outside this file needs the raw 32-bit
   draw, only the [0,1) float lz_sample itself uses. */
float lz_random_f32(lz_u64 *state) {
    return (float)(random_u32(state) >> 8) / 16777216.0f;
}

/* THE ORDER-PRESERVING INTEGER KEY, the same one lz_softmax's max scan
   and lz_attn_wsum_q8's magnitude scan use. xor the sign-extended sign
   bit into every bit and force the top one: non-negatives land above
   0x80000000 in increasing order, negatives below it in decreasing
   order, and an UNSIGNED compare of the keys reproduces the float
   compare exactly - it is a permutation of the same total order, not an
   approximation.

   It is here because the two scans below run over the WHOLE VOCABULARY
   once per sampled token. On a soft-float target each `p[i] > max_p` is
   a bl __aeabi_fcmpgt, about 30 instructions; the key is three and the
   compare is one. At 32,768 entries that is the difference between 1.0 M
   instructions per token and 0.15 M.

   NaN is the one input where the two disagree, and it takes the same
   disposition the other two scans document: float comparison is false
   for NaN so a plain loop never selects one, while NaN's key is the
   largest and these would. A NaN logit means the forward pass is
   already destroyed. */
#define LZ_FKEY(u) ((u) ^ (((uint32_t)((int32_t)(u) >> 31)) | 0x80000000u))

static uint32_t fkey(float f) {
    union { float f; uint32_t u; } b;
    b.f = f;
    return LZ_FKEY(b.u);
}

static int sample_argmax(const float *p, int n) {
    int i, max_i = 0;
    uint32_t best = fkey(p[0]);
    for (i = 1; i < n; i++) {
        uint32_t k = fkey(p[i]);
        if (k > best) { max_i = i; best = k; }
    }
    return max_i;
}

int lz_sample_argmax(const float *logits, int n) {
    return sample_argmax(logits, n);
}

/* Argmax plus the top-1 softmax probability, ONE pass over logits - not
   argmax then a second full softmax pass, and not a materialized n-wide
   probability vector (lz_softmax, ops.c, does both of those; this is
   deliberately a different, narrower primitive). Built for the MTP
   speculative-decode p_min knob (generate.c's lz_spec_round): llama.cpp's
   reference (speculative.cpp:1601-1609)
   only ever needs the DRAFT head's own top-1 confidence to
   decide whether to keep drafting, never the full distribution.

   Standard online (streaming) softmax-denominator algorithm: `l` is
   sum_j exp(x[j] - running_max), rescaled by exp(old_max - new_max)
   whenever a new max is found so it stays relative to the CURRENT max
   without ever re-scanning what came before. p_top1 = exp(max - max) /
   l = 1/l exactly, because the top-1 term contributes exactly 1.0 to
   the sum by construction - no second lookup, no second exp call for
   the winning index.

   Same tie-break as sample_argmax above (strict >, first index wins) -
   load-bearing: if this and lz_sample_argmax ever ran on the same
   logits, they must agree, and a different tie rule would silently
   break that.

   Not wrapped in lz_fpu_float_begin/end: this runs on logits AFTER
   forward_chunk's own PC=24 region has already closed (same call site,
   same timing as lz_sample_argmax, sampler.c is never inside that
   region), and unlike the forward pass itself, imprecision here cannot
   change what token speculative decoding ultimately emits - it can
   only change how many draft steps get proposed before verify (which
   is always authoritative) checks them. A build-to-build ULP
   difference in p_top1 could shift the p_min stopping point by one
   step; it cannot change the final output token sequence, because
   lz_spec_accept always defers to verify's own argmax regardless of
   how many draft tokens were offered. */
int lz_argmax_p1(const float *logits, int n, float *out_p) {
    int i, best_i = 0;
    float best_v = logits[0];
    float l = 1.0f;   /* element 0's own contribution, exp(0) = 1 */
    for (i = 1; i < n; i++) {
        float v = logits[i];
        if (v > best_v) {
            l = l * lz_exp(best_v - v) + 1.0f;   /* rescale, then add this element's own (now exp(0)=1) term */
            best_v = v;
            best_i = i;
        } else {
            l += lz_exp(v - best_v);
        }
    }
    if (out_p) *out_p = (l > 0.0f) ? 1.0f / l : 1.0f;
    return best_i;
}

static int sample_mult(const float *p, int n, float coin) {
    float cdf = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        cdf += p[i];
        if (coin < cdf) return i;
    }
    return n - 1;
}

static int cmp_desc(const void *a, const void *b) {
    const LZProbIndex *x = (const LZProbIndex *)a;
    const LZProbIndex *y = (const LZProbIndex *)b;
    if (x->prob > y->prob) return -1;
    if (x->prob < y->prob) return 1;
    return 0;
}

/* Penalties. Only walks tokens generated WITHIN the window; never scans
   the whole vocab.
 *
 * The target is a Pentium II at ~0.7 s per token; scanning 32768 entries
 * for penalties would be pure waste. Default window is 64; the dedup
 * inner loop is at worst 64x64=4096, far less than a full scan.
 *
 * Dedup is mandatory: the same token can appear several times in the
 * window, but presence by definition penalizes only the fact of having
 * appeared, not its count - counting is frequency's job. */
static void apply_penalties(LZSampler *s, float *logits) {
    const LZSampleParams *p = &s->p;
    int i, j, t;
    int has_rep = (p->repetition_penalty != 1.0f);
    int has_add = (p->presence_penalty != 0.0f ||
                   p->frequency_penalty != 0.0f);
    if (p->repeat_last_n == 0) return;
    if (!has_rep && !has_add) return;
    for (i = 0; i < s->n_win; i++) {
        t = s->ring[(s->head + i) % s->cap];
        /* already handled this round; skip */
        for (j = 0; j < i; j++) {
            if (s->ring[(s->head + j) % s->cap] == t) break;
        }
        if (j < i) continue;
        if (has_rep) {
            /* CTRL paper & HF definition: positive logits divide by p,
               negative logits multiply by p. Not plain division - that
               would make negative logits larger. */
            if (logits[t] > 0.0f) logits[t] /= p->repetition_penalty;
            else                  logits[t] *= p->repetition_penalty;
        }
        if (has_add) {
            logits[t] -= p->presence_penalty +
                         p->frequency_penalty * (float)s->counts[t];
        }
    }
}

/* apply_penalties' own dedup+penalty logic, over a caller-built linear
   window instead of s->ring/s->counts. See sampler.h's own comment on
   this function for why it is a separate entry point (the assumed
   window has no persistent counts table backing it, and would
   overcount from s->counts if it tried to borrow one - a truncated
   window is not the same set of tokens the untruncated ring counted). */
void apply_penalties_assumed(const LZSampleParams *p, float *logits,
                             const int *window, int window_len) {
    int i, j, t, cnt;
    int has_rep = (p->repetition_penalty != 1.0f);
    int has_add = (p->presence_penalty != 0.0f ||
                   p->frequency_penalty != 0.0f);
    if (p->repeat_last_n == 0) return;
    if (!has_rep && !has_add) return;
    for (i = 0; i < window_len; i++) {
        t = window[i];
        /* already handled this round; skip - same dedup shape as
           apply_penalties' own loop. */
        for (j = 0; j < i; j++) {
            if (window[j] == t) break;
        }
        if (j < i) continue;
        if (has_rep) {
            if (logits[t] > 0.0f) logits[t] /= p->repetition_penalty;
            else                  logits[t] *= p->repetition_penalty;
        }
        if (has_add) {
            cnt = 0;
            if (p->frequency_penalty != 0.0f) {
                /* i is t's FIRST occurrence in window (the dedup loop
                   above found none earlier), so a forward scan from i
                   counts every occurrence, not just the ones after i -
                   there are none before it to miss. */
                for (j = i; j < window_len; j++)
                    if (window[j] == t) cnt++;
            }
            logits[t] -= p->presence_penalty +
                         p->frequency_penalty * (float)cnt;
        }
    }
}

/* Take top-k. Maintains a descending buffer of length k, one pass over
   the vocab.
 *
 * No qsort: at k=20 / vocab 32768 this is ~32768 comparisons plus a few
 * insertions, while qsort is 32768*log2(32768) ~ 490K comparisons - a
 * 15x difference. */
/* Same insertion selection, with every float comparison replaced by the
   integer key above - see fkey. The REJECT test is the one that runs
   32,768 times per sampled token (once the buffer is full, almost every
   entry loses it), so `cut` caches the incumbent's key rather than
   recomputing it per entry; it is refreshed on the only thing that can
   change it, an insertion. The values stored in buf[] are still the
   floats, so nothing downstream sees a key. */
static int select_topk(const float *p, int n, int k, LZProbIndex *buf) {
    int i, j, cnt = 0;
    uint32_t cut = 0;
    for (i = 0; i < n; i++) {
        float pi = p[i];
        uint32_t ki = fkey(pi);
        if (cnt == k && ki <= cut) continue;
        j = (cnt < k) ? cnt++ : k - 1;
        while (j > 0 && fkey(buf[j - 1].prob) < ki) {
            buf[j] = buf[j - 1];
            j--;
        }
        buf[j].prob = pi;
        buf[j].index = i;
        if (cnt == k) cut = fkey(buf[k - 1].prob);
    }
    return cnt;
}

/* top_k WITHOUT the whole-vocabulary softmax in front of it.
 *
 * THE NORMALIZER CANCELS IN EVERY COMPARISON DOWNSTREAM, which is what
 * makes this legal rather than merely cheaper. With top_k active, every
 * later step is a ratio inside the candidate set:
 *   min_p  buf[i].prob < m_eff * buf[0].prob
 *   top_p  cumulative >= topp * (the CANDIDATE SET's mass, not the
 *          vocabulary's - see select_candidates' own comment and the
 *          three-reference derivation it cites)
 *   draw   r = coin * cumulative, then a running cdf
 *   ins    buf[i].prob / mass
 *   target out_p[...] = buf[i].prob / mass
 * Scale all of buf[] by any positive constant and not one of those
 * decisions moves. So the 1/Z that lz_softmax divides by is computed
 * from 32,768 exponentials and then divided back out.
 *
 * AND THE SET IS THE SAME SET. exp is strictly increasing, so the k
 * largest post-softmax probabilities are the k largest logits, in the
 * same order, and select_topk's comparisons are all `<` between two
 * transformed values. What it does NOT preserve is a TIE that the
 * softmax manufactures: two distinct logits far below the max both
 * round to the same float probability, and there a naive
 * index-order tie-break picked between entries this one now orders
 * strictly. Those entries have equal probability by construction, so
 * the distribution sampled from is the same one - the token drawn from
 * it can differ, which is why this is a documented value-changing
 * change and not a bit-identical one.
 *
 * buf[] comes back holding UNNORMALIZED exp(l/T - lmax/T), and lmax is
 * the global maximum because the top-k set always contains the argmax.
 *
 * Measured: the full-vocabulary softmax is 32,768 elements per sampled
 * token at 322 ARM instructions each (.prof/arm_prim_count.sh mode 13),
 * about 10.5 M - larger than any single row in docs/arm-asm-audit.md,
 * and absent from that document entirely, which counts only the
 * attention softmax's 16*T.
 *
 * THE TEMPERATURE DIVIDE IS IN HERE FOR THE SAME REASON. Dividing every
 * logit by the temperature is another 32,768 __aeabi_fmuls per sampled
 * token, about 1.0 M ARM instructions, and 1/T is POSITIVE - so it is
 * monotone too and top_k picks the same k entries before it as after.
 * Scaling only the survivors is bit-identical rather than merely
 * equivalent: float multiply by a positive constant is monotone
 * non-decreasing, so max_j fl(l_j*inv) is fl(max_j l_j * inv), which is
 * the same `lmax` and therefore the same argument to every lz_exp. */
static int select_topk_exp(const float *l, int n, int k, float inv,
                           LZProbIndex *buf) {
    int n0 = select_topk(l, n, k, buf);
    float lmax;
    int i;
    if (n0 <= 0) return n0;
    for (i = 0; i < n0; i++) buf[i].prob *= inv;   /* temperature, 20 of them */
    lmax = buf[0].prob;                  /* still a logit at this point */
    for (i = 0; i < n0; i++) buf[i].prob = lz_exp(buf[i].prob - lmax);
    return n0;
}

/* minp^(1/T), memoised on the exact bits of both arguments.
 *
 * This is the only libm double call left in the per-token path, and its
 * two inputs are sampling settings: they change when a caller changes
 * one, which for a normal generation is never. Recomputing it per token
 * is the same answer at a double-precision pow's price, on a target
 * family whose slowest members have no hardware divide behind that pow
 * at all.
 *
 * Keyed by ==, not by a tolerance. A hit returns the value pow returned
 * for those bits, not a value near it, so memoising cannot move a
 * sampled token. The sentinels are negative and both real arguments are
 * guarded positive by the caller, so the first call always misses. A
 * NaN argument compares unequal to everything including itself and
 * simply recomputes, which is the behaviour that keeps a NaN from
 * pinning a stale value in the cache.
 *
 * No lock, and none needed: nothing in this tree creates a thread. The
 * server's slots are sequential. */
static float minp_pow(float minp, float temperature)
{
    static float k_minp = -1.0f, k_temp = -1.0f, cached = 0.0f;
    if (minp == k_minp && temperature == k_temp) return cached;
    cached = lz_powf(minp, 1.0f / temperature);
    k_minp = minp;
    k_temp = temperature;
    return cached;
}

/* Path when top_k is off: coarse threshold filter, then sort; matches llama2.c
 *
 * NOT THE SHIPPING PATH, and the qsort below is why that is worth
 * saying. An indirect call per comparison is poor on an in-order core
 * with weak prediction, which makes this look like an obvious target -
 * but lz_sample_defaults sets topk 20, select_candidates takes the
 * top_k branch whenever 0 < topk < n, and that branch uses select_topk:
 * an insertion sort into a k-sized array with an early-out that skips
 * most of the vocabulary, no function pointer anywhere. Confirmed by
 * running with and without --topk 0 and getting different output.
 * Optimising here would be spending on a branch the default never
 * takes. */
static int select_all(const float *p, int n, float topp, LZProbIndex *buf) {
    int n0 = 0, i;
    const float cutoff = (n > 1) ? (1.0f - topp) / (float)(n - 1) : 0.0f;
    for (i = 0; i < n; i++) {
        if (p[i] >= cutoff) {
            buf[n0].index = i;
            buf[n0].prob = p[i];
            n0++;
        }
    }
    if (n0 > 1) qsort(buf, (size_t)n0, sizeof(LZProbIndex), cmp_desc);
    return n0;
}

/* Steps 4-6 of lz_sample's own pipeline (candidate selection: top_k or
   coarse-filter, then min_p, then top_p) - factored out
   so the temp>0 speculative verify path's own lz_target_dist (below)
   can build the EXACT same candidate set without re-deriving llama.cpp/
   vLLM's own ordering and semantics a second time. That would be the
   real risk of NOT sharing this: a numeric divergence between two
   independently-written "the same filter" implementations catches
   nobody until it silently changes what --spec K's own target
   distribution means relative to plain (non-speculative) sampling -
   the same class of bug the assumed penalty window's own design
   avoided by calling this same rule "the same filter", not "a filter
   like it" (see apply_penalties_assumed's own comment).

   Precondition: logits[] already holds the post-temperature softmax
   (lz_sample's own step 2 / lz_target_dist's own equivalent, run by
   BOTH callers before this). Writes n0 candidates into buf[0..n0-1]
   (descending by probability, RAW POST-SOFTMAX values, NOT yet
   renormalized against each other - lz_sample's own step 7 divides by
   *out_mass to sample; lz_target_dist divides by it per-entry to
   materialize a full distribution instead). Returns the LAST buf
   index to keep (buf[0..return] survive) or -1 if nothing survived
   (this matches lz_sample's own n0<=0 fallback to argmax - *out_mass
   is left unwritten on this path). */
static int select_candidates(const LZSampleParams *pr, float *logits,
                             int n, LZProbIndex *buf, float *out_mass,
                             float temperature) {
    int n0, topk_cut = 0, i, last;
    float cumulative = 0.0f;

    /* `logits` arrives RAW - neither divided by the temperature nor
       softmaxed, and deliberately so: doing either in the caller spends
       it on the whole vocabulary. Each branch below does whichever it
       actually needs, and the top_k branch needs neither over more than
       k entries. See select_topk_exp. */
    if (pr->topk > 0 && pr->topk < n) {
        n0 = select_topk_exp(logits, n, pr->topk, 1.0f / temperature, buf);
        topk_cut = 1;
    } else {
        /* select_all's coarse cutoff is (1-topp)/(n-1), a threshold on
           NORMALIZED probability, and the `mass = 1.0f` below leans on
           the same thing - so this branch keeps the whole-vocabulary
           temperature divide and softmax it always had. */
        float inv = 1.0f / temperature;
        for (i = 0; i < n; i++) logits[i] *= inv;
        lz_softmax(logits, n);
        n0 = select_all(logits, n, pr->topp, buf);
    }
    if (n0 <= 0) return -1;

    /* min_p: floor as a fraction of the max probability, cutting the
       long tail. buf is descending, so buf[0].prob is the max. Scan
       starts at i=1 - the first entry is always kept, matching
       llama.cpp's `size_t i = 1; // first token always matches`.
       Without this, an illegal minp > 1 would empty the candidate set. */
    if (pr->minp > 0.0f && n0 > 1) {
        /* Unit conversion, not a different filter. buf holds
           POST-temperature probabilities, so a threshold expressed in
           llama.cpp units (fraction of the RAW peak) becomes
           minp^(1/T) here - see "min_p units" in sampler.h. Done per
           token rather than at init so it tracks a caller that
           changes temperature; memoised in minp_pow, so the steady
           state costs a float compare rather than a pow.
           Sampling runs OUTSIDE the PC=24 region (lz_fpu_float_end
           already ran in forward_chunk), so libm double is safe here -
           inside it, pow would return wrong values. */
        float m_eff = pr->minp;
        float thresh;
        int m = n0;
        /* `temperature` is the EFFECTIVE temperature for this sample (the
           base value when dynamic temperature is off) - the
           conversion must track the number that actually divided the
           logits, not pr->temperature: the two are the same float only
           when no override is active. */
        if (pr->minp_llamacpp && temperature > 0.0f)
            m_eff = minp_pow(pr->minp, temperature);
        thresh = m_eff * buf[0].prob;
        for (i = 1; i < n0; i++) {
            if (buf[i].prob < thresh) { m = i; break; }
        }
        n0 = m;
    }

    /* Default: keep the whole candidate set. Must be set AFTER the
       min_p truncation above (n0 may have shrunk) and BEFORE the
       top_p loop, which only assigns `last` on the iteration that
       crosses the threshold - when the threshold is never reached,
       this default is the answer. */
    last = n0 - 1;

    /* top_p: accumulate until the threshold is REACHED, including the
       entry that crosses it (>= rather than >, aligned with
       llama.cpp's own top_p comment: "the current iterate should be
       included"). The threshold is top_p of the CANDIDATE SET's mass,
       not of the whole vocabulary - per a three-reference derivation
       and a 48-real-logit-vector measurement. */
    if (pr->topp > 0.0f && pr->topp < 1.0f) {
        float mass = 1.0f, limit;
        if (topk_cut) {
            mass = 0.0f;
            for (i = 0; i < n0; i++) mass += buf[i].prob;
        }
        limit = pr->topp * mass;
        cumulative = 0.0f;
        for (i = 0; i < n0; i++) {
            cumulative += buf[i].prob;
            if (cumulative >= limit) { last = i; break; }
        }
    } else {
        for (i = 0; i < n0; i++) cumulative += buf[i].prob;
    }
    *out_mass = cumulative;
    return last;
}

/* The temp>0 speculative verify path's own target-distribution builder
   (generate.c's lz_spec_round_temp): penalties (assumed
   window - ALREADY applied to logits by the caller via apply_penalties_
   assumed before this runs, same step-1-first discipline lz_sample's
   own comment states) + temperature + softmax + target-only filters
   (top_k/top_p/min_p, via select_candidates above) - but instead of
   drawing ONE sample (lz_sample's own step 7), this writes the FULL
   renormalized distribution: lz_spec_accept_temp's own coupled test
   needs p(x) for the draft's token AND, on its residual branch,
   max(0, p(x')-q(x')) for potentially every x' in the vocab - a single
   scalar cannot answer that, only the whole vector can.

   out_p (vocab-length, caller-owned, n entries) receives it: entries
   the filters excluded are exactly 0.0 (not merely small), which is
   what makes it safe to feed straight into a residual scan without
   re-deriving which entries survived. logits[] is modified in place
   through temperature+softmax, the same convention as lz_sample. buf
   is scratch (same shape/size as lz_sample's own s->probindex) - this
   function takes it as a parameter rather than a full LZSampler
   because it has no ring/counts/rng of its own to go with one (same
   reasoning apply_penalties_assumed's own comment gives for why it
   does not take an LZSampler either).

   Filters here are the TARGET side's own settings - the draft side
   (lz_sample_temp_q, below) never applies top_k/top_p/min_p: only p
   needs to match what
   plain (non-speculative) sampling would have produced; q only needs
   to be a valid distribution the draft head can be scored against. */
void lz_target_dist(const LZSampleParams *pr, float *logits, int n,
                    LZProbIndex *buf, float *out_p) {
    int last, i;
    float mass;

    if ((pr->topk <= 0 || pr->topk >= n) &&
        (pr->topp <= 0.0f || pr->topp >= 1.0f) && pr->minp <= 0.0f) {
        float inv = 1.0f / pr->temperature;
        for (i = 0; i < n; i++) logits[i] *= inv;
        /* No filters: the softmax IS the final distribution already -
           same "nothing to restrict" case lz_sample's own step 3
           short-circuits, just returning the vector instead of a
           sample drawn from it. This is the one exit that needs every
           entry normalized, so it is the one that still pays for the
           whole-vocabulary softmax. */
        lz_softmax(logits, n);
        memcpy(out_p, logits, (size_t)n * sizeof(float));
        return;
    }

    memset(out_p, 0, (size_t)n * sizeof(float));
    last = select_candidates(pr, logits, n, buf, &mass, pr->temperature);
    if (last < 0) {
        /* Degenerate (matches lz_sample's own n0<=0 fallback: argmax -
           a one-hot distribution IS the correct renormalization of an
           empty candidate set's "whatever survives", same spirit as
           lz_sample falling back to a deterministic pick rather than
           sampling from nothing). */
        out_p[sample_argmax(logits, n)] = 1.0f;
        return;
    }
    for (i = 0; i <= last; i++) out_p[buf[i].index] = buf[i].prob / mass;
}

/* Softmax-then-sample at TEMPERATURE ONLY - no penalties, no top_k/
   top_p/min_p (the draft head is scored against its OWN full
   distribution, which filtering would falsify). The draft head's own
   sampling primitive for temp>0
   speculative decoding (generate.c's lz_spec_round_temp).

   logits modified in place through temperature+softmax, matching lz_
   sample/lz_target_dist's own convention - once this returns, logits[]
   itself IS the draft's full distribution q: q(x) for the returned
   token is logits[x], and the whole array is available (unchanged
   after this call) for lz_spec_accept_temp's own residual scan, at no
   extra cost - softmax already computed every entry regardless of
   which one gets sampled.

   Returns the sampled index. rng_state is caller-owned (the same
   LZSampler.rng_state the rest of a round's real sampling draws from,
   or a dedicated stream - this function does not care which, it only
   ever advances whatever pointer it is given, same convention as
   apply_penalties_assumed taking a caller-built window instead of
   reaching into a specific LZSampler). */
int lz_sample_temp_q(float *logits, int n, float temperature,
                     lz_u64 *rng_state) {
    float coin;
    int i;
    float inv = 1.0f / temperature;
    for (i = 0; i < n; i++) logits[i] *= inv;
    lz_softmax(logits, n);
    coin = lz_random_f32(rng_state);
    return sample_mult(logits, n, coin);
}

/* ins-fill helper for lz_sample_ex's two "one candidate, certainty 1.0"
   exit points (greedy at the temperature floor, and select_candidates'
   own "nothing survived" fallback) - both collapse to argmax, and both
   mean the same thing for a shortlist: there is exactly one entry. */
static void inspect_fill_single(LZInspect *ins, int idx) {
    ins->n_survived = 1;
    ins->n_cand = 1;
    ins->cand_id[0] = idx;
    ins->cand_p[0] = 1.0f;
}

/* ins-fill helper for lz_sample_ex's two "there is a real shortlist"
   exit points (the unfiltered full-softmax case and the ordinary
   top_k/top_p/min_p case) - both hand this a descending LZProbIndex
   buffer, how many of its entries actually survived (which may exceed
   LZ_INSPECT_CAND_MAX - only the first CAND_MAX are copied out, per
   this struct's own contract in src/inspect.h), and the probability
   mass to renormalize cand_p against: 1.0f for the unfiltered case
   (buf already holds the full, already-normalized softmax - see
   lz_sample_ex's own case 3), select_candidates' own *out_mass for the
   filtered case (buf holds RAW post-softmax values that only sum to
   the survived subset's own mass, not 1.0 - see select_candidates'
   own comment). */
static void inspect_fill_from_buf(LZInspect *ins, const LZProbIndex *buf,
                                  int n_survived, float mass) {
    int n_cand = n_survived < LZ_INSPECT_CAND_MAX ?
                 n_survived : LZ_INSPECT_CAND_MAX;
    int i;
    ins->n_survived = n_survived;
    ins->n_cand = n_cand;
    for (i = 0; i < n_cand; i++) {
        ins->cand_id[i] = buf[i].index;
        ins->cand_p[i] = (mass > 0.0f) ? buf[i].prob / mass : 0.0f;
    }
}

float lz_sample_eff_temp(const LZSampleParams *p, int in_think) {
    /* Pure number selection, no arithmetic - the "which float divides the
       logits" decision, shared by lz_sample_ex's per-token path and
       generate.c's round-level speculative override - one rule, not two
       copies. With the enable flag clear the function returns
       `temperature` itself, bit for bit, which is what makes dynamic
       temperature a no-op by default. */
    if (p->think_temp_enabled && in_think) return p->temp_think;
    return p->temperature;
}

int lz_sample_ex(LZSampler *s, float *logits, LZInspect *ins,
                 int in_think) {
    const LZSampleParams *pr = &s->p;
    int n = s->vocab_size;
    int i, last;
    float coin, cumulative = 0.0f, r, cdf = 0.0f;
    float temp;
    LZProbIndex *buf = s->probindex;

    /* 1. penalties on raw logits (same phase as HF's processors) */
    apply_penalties(s, logits);

    /* 2. temperature. The EFFECTIVE temperature - the base value, or the
       think-block override. With dynamic temperature off,
       `temp` is `pr->temperature` itself and every comparison below is
       bit-identical to before the feature existed. */
    temp = lz_sample_eff_temp(pr, in_think);
    /* 0 degenerates to greedy; top_k/top_p then moot.
     *
     * The test is `<=` against a FLOOR, not `== 0.0f`, and that is not
     * tidiness. temperature arrives from a JSON request body, so any
     * float is reachable:
     *
     *   1e-40   a legal JSON number and a denormal float. 1.0f/1e-40f
     *           overflows to +inf, and inf * a logit of exactly 0.0 is
     *           NaN - one NaN then poisons the whole softmax and the
     *           sampler picks from a table of NaNs.
     *   -1      inverts the distribution: softmax of negated logits
     *           makes the LEAST likely token the most likely. Silently,
     *           and it looks like a broken model rather than a bad
     *           parameter.
     *
     * vLLM guards both, and names the same failure: sampling_params.py
     * clamps 0 < temperature < 1e-2 up to 1e-2 with the warning "may
     * cause numerical errors nan or inf in tensors", and rejects
     * temperature < 0 outright.
     *
     * Below the floor the distribution is already argmax to well past
     * float precision, so collapsing to greedy is not an approximation -
     * exp((l_i - l_max) / 1e-3) underflows to 0 for any logit gap above
     * ~0.09. Negatives are clamped here too rather than trusted; the
     * endpoint rejects them with a 400, but the DLL and CLI are callers
     * as well and this is the one place all three pass through.
     */
    if (!(temp > LZ_TEMP_FLOOR_F)) {
        int idx = sample_argmax(logits, n);
        if (ins) inspect_fill_single(ins, idx);
        return idx;
    }
    /* NO temperature divide here, and that is the point: at this spot
       it would run over every vocab entry before anything had decided
       which entries matter. With a filter active it runs on the k
       survivors inside select_topk_exp instead, which is legal because
       1/T is positive and therefore monotone. The no-filter exit below
       does do the whole vector - it returns the whole vector. */
    coin = lz_random_f32(&s->rng_state);

    /* 3. none of top_k/top_p/min_p active: sample directly from the
       distribution. ins != NULL is the one place this costs anything
       extra: select_topk over the whole (already-softmaxed) vocab, the
       same one-pass insertion selection the top_k path itself uses,
       just to report the top LZ_INSPECT_CAND_MAX of a set that is
       really the WHOLE vocabulary (n_survived is n, not the count
       select_topk returns - it was capped at CAND_MAX by construction,
       that cap is not how many entries actually survived). mass is
       1.0f: logits[] IS the full normalized softmax at this point,
       nothing has renormalized a subset of it. */
    if ((pr->topk <= 0 || pr->topk >= n) &&
        (pr->topp <= 0.0f || pr->topp >= 1.0f) && pr->minp <= 0.0f) {
        int idx;
        float inv = 1.0f / temp;
        /* The one exit that samples from the WHOLE vector, so the one
           that still pays for the whole vector: the temperature divide
           and the softmax both run over every entry here, and nowhere
           else. Every other path below works inside the candidate set,
           where the normalizer cancels - see select_topk_exp. */
        for (i = 0; i < n; i++) logits[i] *= inv;
        lz_softmax(logits, n);
        idx = sample_mult(logits, n, coin);
        if (ins) {
            int k = (LZ_INSPECT_CAND_MAX < n) ? LZ_INSPECT_CAND_MAX : n;
            select_topk(logits, n, k, buf);
            inspect_fill_from_buf(ins, buf, n, 1.0f);
        }
        return idx;
    }

    /* 4-6. candidate set: top_k or coarse-filter, then min_p, then top_p
       - select_candidates above (shared with lz_target_dist's own
       temp>0 target-distribution builder, so the filter ordering and
       semantics live in exactly one place). */
    last = select_candidates(pr, logits, n, buf, &cumulative, temp);
    if (last < 0) {
        int idx = sample_argmax(logits, n);
        if (ins) inspect_fill_single(ins, idx);
        return idx;
    }

    if (ins) inspect_fill_from_buf(ins, buf, last + 1, cumulative);

    /* 7. renormalize over the candidate set by remaining mass, then sample */
    r = coin * cumulative;
    for (i = 0; i <= last; i++) {
        cdf += buf[i].prob;
        if (r < cdf) return buf[i].index;
    }
    return buf[last].index;
}

int lz_sample(LZSampler *s, float *logits) {
    return lz_sample_ex(s, logits, NULL, 0);
}
