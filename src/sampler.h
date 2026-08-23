#ifndef LZ_SAMPLER_H
#define LZ_SAMPLER_H

#include "lz_int.h"   /* lz_u64: the 64-bit type, portably */
#include "inspect.h"

/* Sampling parameters and penalties.
 *
 * Every number quoted below is measured, not recited.
 *
 * ## Defaults (lz_sample_defaults) and where they leave Qwen's table
 *
 *     temperature 0.6    <- Qwen 0.7. Owner's determinism preference;
 *                           measured loop cost inside 1 sigma.
 *     top_p       0.8       Qwen's.
 *     top_k       20        Qwen's. NOTE its rationale does not transfer:
 *                           20 is aggressive for a 151,936 vocabulary,
 *                           ours is 32,768 and frequency-pruned.
 *     min_p       0.05    <- Qwen 0.0, and in LLAMA.CPP UNITS - see below.
 *     presence    1.5       Qwen's value, but a different quantity here
 *                           (window, not whole output).
 *     frequency   0.0       Qwen's.
 *     repetition  1.1     <- Qwen 1.0. Measured: degenerate output
 *                           57.5% -> 38.8% over 80 prompts (~2.4 sigma).
 *                           The official set has NO escalating
 *                           anti-repetition term - repetition off,
 *                           frequency off, only the BINARY presence -
 *                           which is structurally weak against a loop.
 *     repeat_last_n 64      llama.cpp's parameter; vLLM has no equivalent.
 *
 * Every deviation is measured or attributed. None is "because tool X
 * says so" - the official set is benchmark-tuned, which is a stated
 * first-hand judgement, not a guess.
 *
 * repetition_penalty must stay EXPLICIT even at 1.0: implementations
 * disagree (HuggingFace 1.0, llama.cpp 1.1, vLLM/OpenAI 1.0) and this
 * engine descends from llama2.c, so an absent field means whoever
 * configures by another tool's habits silently gets a different penalty.
 *
 * lz_sample_defaults_think: temperature 0.6, top_p 0.95, presence 1.5,
 * repetition 1.1. Qwen splits thinking into general (1.0) and precise
 * code tasks (0.6); this engine has ONE preset and always will - a
 * 57.6M model has no coding tier for a second row to serve.
 *
 * One deviation from Qwen's Thinking (general) row:
 *
 *   repetition 1.1 vs 1.0 the escalating anti-repetition term that
 *                         guards a long reasoning run against looping.
 *
 * temperature and presence both MATCH Qwen's coding-row values rather
 * than the general row's, for reasons settled by measurement on the
 * first thinking model (rl_r1):
 *
 *   temperature 0.6 vs 1.0  the general 1.0 works upstream because a
 *                         full-scale model's reasoning pass absorbs the
 *                         sampling randomness afterwards; this one has
 *                         nothing to absorb with, and at 0.8 sampling
 *                         noise still drowns the argmax on factual
 *                         turns. 0.6 keeps the argmax; the thinking
 *                         looseness comes from top_p 0.95, not
 *                         temperature.
 *   presence 1.5 vs 1.5    the coding-row rationale says a chain of
 *                         thought revisits the same entities on purpose
 *                         and a binary penalty would make it restate
 *                         them in worse words; measurement did not bear
 *                         that out: degenerate-answer rate 14-17/42 at
 *                         0.0 vs 3-5/42 at 1.5, with 1.5 adding ~1/42
 *                         new degeneracy.
 *
 * ## Cross-ecosystem semantics: only top_k transfers as-is
 *
 * Root cause: llama.cpp applies temperature LAST, everyone else FIRST.
 * This engine follows transformers/vLLM/sglang.
 *
 *     top_k        no divergence - temperature is monotone, the ranking
 *                  and therefore the surviving set are identical.
 *     top_p        second-order ordering effect, no closed form. (The
 *                  first-order problem was ours and is fixed; see
 *                  sampler.c step 6.)
 *     min_p        closed form: m_engine = m_llamacpp^(1/T). At T=0.6 a
 *                  habitual 0.05 is 3.3x stronger here, and the value
 *                  reproducing that habit is 0.0068. Hence
 *                  minp_llamacpp, converted at sample time so it tracks
 *                  a caller that changes temperature.
 *     presence /   vLLM applies these over the WHOLE output; ours is a
 *     frequency    64-token window, so Qwen's 1.5 - tuned in vLLM's
 *                  regime - is a materially weaker filter here.
 *     repetition   vLLM also feeds prompt tokens into its penalty path;
 *                  we deliberately do not (below).
 *
 * The endpoint keeps NATIVE units for min_p: it is an OpenAI/vLLM wire
 * field, and reinterpreting it locally would make the same request mean
 * two things depending on whether the field was sent.
 *
 * ## Penalty shapes
 *
 *     repetition   multiplicative, sign-dependent: positive logits
 *                  divide by p, negative multiply (CTRL/HF definition,
 *                  not plain division). Its effect scales with the
 *                  logit's magnitude.
 *     presence     additive, once, identical for every token.
 *     frequency    additive, scaled by occurrence count.
 *
 * ## Scope: generated tokens only, recent window only
 *
 * Counts exclude the prompt - words in the prompt are often exactly what
 * the answer needs, so parameters
 * tuned in training behave the same when served.
 *
 * repeat_last_n defaults to 64 (llama.cpp's parameter). Not an optional
 * knob: accumulated over a whole generation, presence 1.5 would have
 * suppressed nearly every common word within a few hundred tokens and
 * forced the model to keep swapping vocabulary. Semantics:
 *
 *     0    penalties off
 *    -1    whole generation (capped at LZ_PENALTY_MAX_WINDOW)
 *    >0    last N generated tokens
 */

/* Ring buffer size when -1 (whole generation) is selected. This
   engine's seq_len is far below it, so to users it reads as
   "unlimited". */
#define LZ_PENALTY_MAX_WINDOW 4096

/* Temperature at or below this is treated as greedy. See lz_sample for
   why the comparison cannot be `== 0.0f`: a denormal temperature makes
   1/t overflow to inf, and inf * a zero logit is NaN. 1e-3 is two orders
   below the smallest temperature anyone configures and one below vLLM's
   own clamp (_MAX_TEMP = 1e-2, sampling_params.py). */
#define LZ_TEMP_FLOOR 1e-3f
/* 1e-3f is not exactly representable in float32, so an inline
   `> LZ_TEMP_FLOOR` comparison lets gcc's x87 excess-precision folding
   carry the constant at extended precision (fldt) independently in
   every TU that compares against it - see ops.c's LZ_EXP_LOG2E32
   comment for the mechanism. Each includer gets its own private
   (internal-linkage) `static const float` copy, which forces the round
   to true float32 once per TU instead. */
static const float LZ_TEMP_FLOOR_F = LZ_TEMP_FLOOR;

typedef struct {
    float prob;
    int   index;
} LZProbIndex;

typedef struct {
    float temperature;
    /* Dynamic temperature: per-region temperature override, OFF by
       default. A struct that never sets the enable flag keeps the
       sampler on plain `temperature`, byte-identical to the non-dynamic
       path - the flag is what a CLI switch turns on, and
       without it the sampler never looks past `temperature`.

         temp_think  used INSIDE a <think> block, when think_temp_enabled.

       The override only ever substitutes the NUMBER that divides logits
       at step 2 - it never changes how that division happens, which is
       what keeps the bit-identity contract intact. */
    float temp_think;
    int   think_temp_enabled;
    float topp;
    float minp;
    int   topk;                 /* <=0 or >=vocab disables it */
    float presence_penalty;
    float frequency_penalty;
    float repetition_penalty;   /* 1.0 = identity; must be explicit */
    int   repeat_last_n;        /* 0=off -1=all >0=window. Default 64 */
    /* Unit `minp` is expressed in. 0 = native (vLLM/OpenAI): the
       threshold is a fraction of the POST-temperature peak, which is
       what the wire protocol must mean. 1 = llama.cpp: a fraction of
       the RAW peak, converted at sample time by minp^(1/T).
       See "min_p units" in the notes above. The kernel stays on the
       vLLM/transformers/sglang order either way; only the number is
       translated. */
    int   minp_llamacpp;
} LZSampleParams;

typedef struct {
    int   vocab_size;
    LZSampleParams p;
    lz_u64 rng_state;
    LZProbIndex   *probindex;   /* sort buffer for top-k / top-p */
    unsigned short *counts;     /* occurrence count per token in the window */
    int           *ring;        /* last cap generated tokens */
    int            cap;         /* window capacity */
    int            head;        /* oldest slot in the ring */
    int            n_win;       /* entries actually in the window */
} LZSampler;

/* Instruction-mode (non-thinking) defaults */
void lz_sample_defaults(LZSampleParams *p);
/* Thinking-mode defaults: presence zeroed, temperature/top_p relaxed */
void lz_sample_defaults_think(LZSampleParams *p);

/* One bit per preset-following field, in the order the presets write
   them. A caller with a wider or narrower sampling surface still uses
   these bits - it simply sets only the ones its own surface exposes - so
   "which fields follow the think preset, and only while the user has not
   set them" is ONE rule here, not one per front end. cli_main.c and
   common/settings.c are the two callers; the GUI turns two of these bits
   into its per-field manual flags, the CLI carries the whole mask. */
#define LZ_MANUAL_TEMP       (1u << 0)
#define LZ_MANUAL_TOPP       (1u << 1)
#define LZ_MANUAL_MINP       (1u << 2)
#define LZ_MANUAL_TOPK       (1u << 3)
#define LZ_MANUAL_PRES       (1u << 4)
#define LZ_MANUAL_FREQ       (1u << 5)
#define LZ_MANUAL_REP        (1u << 6)
#define LZ_MANUAL_RLAST      (1u << 7)
#define LZ_MANUAL_THINK_TEMP (1u << 8)

/* Apply the thinking preset over `dst`, overwriting only the fields the
   caller has NOT marked as manually set. `manual` is a mask of the
   LZ_MANUAL_* bits above. This is the "a value the user typed survives
   the preset, whatever order the flags arrived in" rule, in one place
   rather than two drifting copies. */
void lz_sample_apply_think_preset(LZSampleParams *dst, unsigned manual);

int  lz_sampler_init(LZSampler *s, int vocab_size, const LZSampleParams *p,
                     lz_u64 seed);
void lz_sampler_free(LZSampler *s);

/* Start a new generation: clear penalty counts. Without this, the
   previous turn's history keeps suppressing this one. */
void lz_sampler_reset(LZSampler *s);

/* Record a just-generated token for penalty counting. Do NOT feed
   prompt tokens in. */
void lz_sampler_observe(LZSampler *s, int token);

/* Sample one token. logits are modified in place.
 *
 * Order: penalties -> temperature -> top_k -> top_p -> min_p -> sample.
 *
 * Penalties FIRST is the load-bearing part: after temperature, their
 * strength would vary with it (at temp 0.7 logits are scaled 1.43x, so
 * the same presence 1.5 would be that much weaker). Both references put
 * them first.
 *
 * Verified against the source, not recited:
 *
 *   transformers 5.5.0, generation/utils.py
 *       1089 RepetitionPenalty / 1195 custom processors merged /
 *       1214 Temperature / 1219 TopK / 1223 TopP / 1228 MinP
 *   llama.cpp master, common/common.h:260
 *       PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P,
 *       XTC, TEMPERATURE          <- temperature LAST
 *
 * This engine follows transformers, because the Python-side sampling drives HF
 * generate() and the two must agree. That agreement rests on line 1195
 * sitting BEFORE 1214 - our custom presence/frequency processor is
 * merged in before the warpers are appended. If a future transformers
 * moves the merge after the warpers, the Python side diverges SILENTLY.
 * Check that line before upgrading.
 *
 * min_p runs before top_p here, after it in both references. Equivalent:
 * both are prefix truncations of the same descending list, and min_p's
 * threshold comes from the maximum, which top_p never removes - two
 * prefixes intersect to the shorter one. Gated by t_sampler rather than
 * left as an argument.
 */
int  lz_sample(LZSampler *s, float *logits);

/* Same sampler, same pipeline, ONE more argument: `ins`, filled with
 * what the candidate set actually looked like this token, or left
 * entirely untouched when NULL.
 *
 * lz_sample IS lz_sample_ex(s, logits, NULL, 0) - a thin wrapper, not a
 * parallel implementation, so the two can never drift apart on the
 * token they return.
 *
 * `ins == NULL` costs nothing beyond the one branch that skips it at
 * each of the four return points below - no extra pass over logits, no
 * extra sort, nothing allocated. This is the whole argument for an
 * optional out-param over a callback: the hot path (CLI, the HTTP
 * endpoint, every speculative-decode draft/verify call) passes NULL
 * and never pays for a caller that might want to watch.
 *
 * What ins reports depends on which of lz_sample's own four exit
 * points fired, because "the candidate set" means something different
 * at each one:
 *
 *   temperature <= LZ_TEMP_FLOOR (greedy), or select_candidates left
 *   nothing (last < 0, its own argmax fallback): exactly one candidate,
 *   the argmax token, probability 1.0 - greedy decoding IS a shortlist
 *   of one.
 *
 *   no filters active (top_k/top_p/min_p all off): every vocabulary
 *   entry is nominally a "survivor" - n_survived is the full vocab
 *   size - but only the top LZ_INSPECT_CAND_MAX are reported, found
 *   with the SAME select_topk() the top_k path itself uses, over the
 *   already-softmaxed logits (an extra O(n) pass, paid only when ins
 *   is non-NULL and no filter already narrowed the field).
 *
 *   the ordinary top_k/top_p/min_p path: n_survived is select_
 *   candidates' own count (last+1), and cand_p is renormalized over
 *   that survived mass - the same renormalization step 7 divides by to
 *   sample, not the raw post-softmax probabilities from step 2.
 *
 * `in_think`: which region
 * this token is being sampled for - the pure "which NUMBER divides the
 * logits" selector; lz_sample_eff_temp below turns it into the effective
 * temperature, and the rest of the pipeline is unchanged. Pass 0 for "no
 * dynamic temperature": the effective temperature is then `temperature`
 * itself, so the call reduces to the plain temperature path. */
int  lz_sample_ex(LZSampler *s, float *logits, LZInspect *ins,
                  int in_think);

/* Effective temperature for one sample: the number
   lz_sample_ex's own step 2 divides logits by, after the think-block
   override (see LZSampleParams above). Exported because generate.c's
   speculative round needs the SAME rule for its round-level temperature
   as the per-token path uses - one rule, not a second copy. With the
   enable flag clear this returns `temperature` itself, bit for bit. */
float lz_sample_eff_temp(const LZSampleParams *p, int in_think);

/* Plain argmax itself, no penalties/temperature/top_k/top_p/min_p - the
   exact rule lz_sample falls back to at temperature <= LZ_TEMP_FLOOR
   (first index wins ties, strict >). Exposed for the speculative-decode
   draft/verify path (generate.c's lz_spec_round): both the draft head's
   own argmax and the verify pass's per-position argmax need exactly this
   rule, not a second implementation of it, to keep --spec 0 and --spec K
   bit-identical at temperature 0 - a different tie-breaking rule here
   would silently diverge from lz_sample's.

   Penalties are in scope for the verify path (the assumed penalty
   window - see apply_penalties_assumed below and lz_spec_round's own
   verify loop): a caller with non-identity penalties
   applies them to a row's logits FIRST, in place, the same way lz_sample
   itself does at step 1 of its own pipeline, and only THEN calls this
   function - argmax itself still never sees a penalty, it is applied
   the same "penalties, then greedy pick" order either path takes. */
int  lz_sample_argmax(const float *logits, int n);

/* Same penalty math as lz_sample's own internal apply_penalties, over a
   CALLER-BUILT linear window instead of the sampler's own ring - the
   speculative-decode verify path (generate.c's lz_spec_round) needs
   this because a verify row's penalty window is never a suffix of the
   REAL ring: it has to include this round's OWN not-yet-observed draft
   tokens, which do not exist in s->ring until (if) they are accepted.

   window[0..window_len-1] is oldest-first, exactly s->ring's own
   iteration order - see lz_spec_round's own comment for how it is
   assembled (real ring tail ++ this round's own draft prefix, with the
   same eviction the real ring would have applied had those tokens been
   observed one at a time).

   Frequency counts are recomputed FROM window itself (repetition_
   penalty and presence_penalty need no count, only "does t occur at
   all" - see the dedup loop), not read from any persistent table: the
   assumed window is frequently a TRUNCATED prefix of the real ring
   (repeat_last_n caps it), so s->counts, which reflects the UNTRUNCATED
   ring, would overcount relative to what this window actually holds.
   Same asymptotic cost as apply_penalties' own dedup scan (a second
   O(window_len) pass per first-occurrence position, so still
   O(window_len^2) overall), and only paid when frequency_penalty != 0
   (default 0.0, both factory presets - sampler.c) - the common case
   never runs it. */
void apply_penalties_assumed(const LZSampleParams *p, float *logits,
                             const int *window, int window_len);

/* lz_sample_argmax plus the top-1 softmax probability, in the SAME
   single pass (not argmax then a separate softmax scan). See sampler.c
   for the online-softmax-denominator derivation. Built for the MTP
   speculative-decode p_min knob (LZGenOpts.p_min, generate.c's
   lz_spec_round) - llama.cpp's reference only ever needs the draft
   head's own top-1 confidence, never the whole distribution
   (speculative.cpp:1601-1609). *out_p
   is written iff non-NULL; the return value is the argmax index, same
   tie-break as lz_sample_argmax (load-bearing - see that comment). */
int  lz_argmax_p1(const float *logits, int n, float *out_p);

/* xorshift64* draw, [0,1) - the SAME generator lz_sample itself draws
   from (sampler.c). Exported so generate.c's temp>0
   speculative path (lz_sample_temp_q below, and lz_spec_accept_temp's
   own residual resample) shares one PRNG implementation instead of a
   second hand-copied one - see lz_random_f32's own comment in
   sampler.c for why that specific duplication is the one this project
   least wants. state is caller-owned, advanced in
   place; typically an LZSampler's own rng_state, but any u64 works. */
float lz_random_f32(lz_u64 *state);

/* Target-side distribution builder for temp>0 speculative verify
   (generate.c's lz_spec_round_temp): penalties (assumed
   window - the CALLER applies apply_penalties_assumed to logits
   BEFORE calling this, same "penalties first" discipline lz_sample
   itself follows) + temperature + softmax + top_k/top_p/min_p (the
   SAME filter code lz_sample uses internally - select_candidates,
   sampler.c, shared not reimplemented), materialized as a FULL
   renormalized probability vector rather than one sample - lz_spec_
   accept_temp needs p(x) for the draft's token and, on its residual
   branch, potentially every p(x') in the vocab, not a single draw.

   logits (n entries, i.e. vocab_size) modified in place through
   temperature+softmax. buf is scratch, n entries, same shape as an
   LZSampler's own probindex (a caller with a real LZSampler can pass
   s->probindex directly). out_p (n entries, caller-owned) receives
   the distribution - filtered-out entries are exactly 0.0, not merely
   small, which is what makes it safe to feed straight into a residual
   scan. Filters are the TARGET side's own settings: the draft side
   (lz_sample_temp_q below) never
   applies them, only temperature - see that function's own comment
   for why filtering q would falsify the probability it is supposed to
   report. */
void lz_target_dist(const LZSampleParams *p, float *logits, int n,
                    LZProbIndex *buf, float *out_p);

/* Softmax-then-sample at TEMPERATURE ONLY, no penalties, no top_k/
   top_p/min_p - the draft head's own sampling primitive for temp>0
   speculative decoding (generate.c's lz_spec_round_temp). logits (n
   entries) modified in place through temperature+softmax; once this
   returns, logits[] itself IS the draft's full distribution q -
   logits[returned index] is q(x) for the sampled token, and the whole
   array (unchanged after this call) is what lz_spec_accept_temp's own
   residual branch needs, at no extra cost beyond the softmax this
   function was already going to compute. rng_state is caller-owned,
   advanced in place - not necessarily the same stream the target side
   samples from, this function does not care whose state pointer it
   receives (same "just advance whatever I am given" convention lz_
   random_f32 above documents). Returns the sampled index. */
int  lz_sample_temp_q(float *logits, int n, float temperature,
                      lz_u64 *rng_state);

#endif
