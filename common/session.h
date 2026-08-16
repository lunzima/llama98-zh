#ifndef LZ_SESSION_H
#define LZ_SESSION_H

/* Front-end-agnostic multi-turn conversation core.
 *
 * The conversation loop lives here: history, prompt render, prefix-cache
 * reuse, reply accumulation, and turn trimming. GUI-only concerns stay
 * out:
 *
 *   - the custom system prompt is a per-call `const char *system`
 *     parameter (NULL = engine built-in), never a field - see
 *     lz_session_begin / lz_session_regen / lz_session_job;
 *   - per-turn seeding is the caller's job: opts.rng_seed is used
 *     verbatim, and the GUI's fixed/random policy stays in gui/session.c;
 *   - the generation call takes an OPTIONAL LZInspect (lz_session_job's
 *     `ins` parameter, forwarded straight to lz_generate_resume_ex); the
 *     CLI passes NULL, the GUI passes &LZGuiSession.ins so gui/worker.c
 *     can copy the per-token snapshot out for the inspector panel;
 *   - the worker callbacks are forwarded per call (lz_session_job's
 *     sink/cont/cb_ctx parameters) instead of being stored on the struct.
 *
 * This header may include the engine headers under src/ but NOTHING from
 * gui/ (no modelload.h, no settings.h).
 *
 * Engine-facing: all comments are English, no console output, and the
 * session reports through LZErr codes + errbuf exactly like the engine. */

#include "chat.h"      /* LZChatHist, LZChatBuf, LZChatMsg */
#include "llama_zh.h"  /* LZModel, LZTokenizer, LZRunState, LZGenOpts,
                          LZPrefixCache, LZTokenSink, LZShouldContinue,
                          lz_prefix_* */

/* Which prefill strategy the next lz_session_job turn uses.
 *
 * LZ_PREFILL_FULL re-renders the whole conversation and starts the
 * KV/SSM state from scratch every turn; LZ_PREFILL_PREFIX asks the job to
 * try lz_prefix_prepare / lz_generate_resume instead, reusing the state a
 * previous turn already forwarded through - but only once pc_ready says
 * the cache actually exists for the CURRENTLY loaded model.
 *
 * memset'ed to LZ_PREFILL_FULL by lz_session_init (the enum's zero
 * value); a front end that wants PREFIX sets s->prefill explicitly - the
 * CLI's --ckpt flag and the GUI's ini read both do. FULL is the control
 * arm. */
typedef enum { LZ_PREFILL_FULL = 0, LZ_PREFILL_PREFIX } LZPrefillMode;

/* One conversation: history, the prompt for the current turn, the bytes
 * the model produced, and the last turn's generation statistics.
 *
 * The three engine pointers are BORROWED - the session neither loads nor
 * frees a model, tokenizer or run state; it only uses them.
 * lz_session_init stores them and lz_session_free deliberately does not
 * free them.
 *
 * Everything else is owned by the session: hist owns its message bytes,
 * prompt owns its rendered bytes, reply owns the
 * UTF-8 bytes the model generated (the authority for the assistant turn),
 * and pc owns the prefix-cache allocation.
 *
 * NOT named chat.h - the engine has one, and which `#include "chat.h"`
 * resolves to would depend on -I order. */
typedef struct {
    LZModel     *model;    /* borrowed */
    LZTokenizer *tok;      /* borrowed */
    LZRunState  *state;    /* borrowed */
    LZChatHist   hist;
    LZChatBuf    prompt;
    char        *reply;    /* UTF-8 as generated - the authority */
    int          reply_len, reply_cap;
    int          think;
    LZGenOpts    opts;
    int          trimmed;
    LZPrefillMode prefill;
    LZPrefixCache pc;
    int          pc_ready;
    /* Last turn's statistics, written by lz_session_job, read after.
     * NOT reset between turns; a new job overwrites them. */
    int n_out, n_prompt_tok, reused;
    double ms;
} LZSession;

void lz_session_init(LZSession *s, LZModel *model, LZTokenizer *tok,
                     LZRunState *state, int think);
void lz_session_free(LZSession *s);

/* Clear the conversation. The model stays loaded. */
void lz_session_reset(LZSession *s);

/* ---- the prefix cache's lifetime, in one place ----
 *
 * The rule:
 *   the MODEL changed  -> drop, then arm for the new one
 *   the HISTORY changed -> clear
 * Dropping when only the history moved throws away an allocation for
 * nothing; clearing when the model moved leaves a cache sized for weights
 * that are gone. */

/* The model is going away or has gone: free the cache and clear the ready
 * flag. Idempotent, and safe before any cache ever existed. */
void lz_session_prefix_drop(LZSession *s);

/* Build a cache for `model`'s weights. Returns 0 on success. FAILURE IS
 * NOT FATAL and callers must not treat it as a load failure: pc_ready
 * simply stays 0, and lz_session_job's prefix branch then never fires,
 * which is the same leniency its own prepare-failure fallback has one
 * level later.
 *
 * INVARIANT: `model` must be the same LZModel the session was initialized
 * with (s->model). The cache is sized for that model's weights; arming it
 * for any other model would leave a cache lz_session_job then matches
 * against the wrong weights. The GUI forward passes &mdl->model, which IS
 * s->model. */
int lz_session_prefix_arm(LZSession *s, LZModel *model,
                          char *errbuf, int errlen);

/* The history changed under a cache that is still valid for this model:
 * keep the allocation, forget what it held. Safe when there is no cache
 * (lz_prefix_reset only touches three ints on the struct itself, never
 * the pointers a real arm would have allocated). */
void lz_session_prefix_clear(LZSession *s);

/* Sampling defaults for the current think state, taken from the engine
 * (lz_sample_defaults / lz_sample_defaults_think) rather than written
 * here. */
void lz_session_set_think(LZSession *s, int think);

/* Push the user turn and render the whole conversation into prompt.
 * Full-history mode: correctness first, resume later. `system` is the
 * custom system prompt to prepend for this turn, or NULL for the engine's
 * built-in identity (see render_conv_core in session.c). Returns 0, or an
 * LZ_ERR_* code with errbuf set. */
int lz_session_begin(LZSession *s, const char *user_utf8, int len,
                     const char *system, char *errbuf, int errlen);

/* Another attempt at the user turn history ALREADY ends with - the
 * "regenerate" command. Everything lz_session_begin does except the push,
 * and begin is literally push-then-this, so the two cannot drift.
 *
 * Requires history to end with a user message; returns LZ_ERR_NO_USER if
 * it does not. The FRONT END is what makes that true, by popping the
 * previous reply before calling this.
 *
 * The core draws NO fresh seed here: a front end that wants the second
 * attempt to differ from the first sets s->opts.rng_seed before calling
 * in. Returns 0, or an LZ_ERR_* code with errbuf set. */
int lz_session_regen(LZSession *s, const char *system,
                     char *errbuf, int errlen);

/* The worker's generation job. `ud` is the session; `system` is the custom
 * system prompt the trim-and-retry re-render must keep using, or NULL for
 * the engine's built-in. BORROWED, and it must remain valid for the ENTIRE
 * job, not just this call - a trim-and-retry re-render can read it seconds
 * after the first render did.
 *
 * `ins` is an optional LZInspect the engine fills once per sampled token
 * (the GUI's inference inspector), or NULL for the plain lz_generate_resume
 * behaviour - lz_generate_resume IS lz_generate_resume_ex(..., NULL), so
 * passing NULL costs nothing.
 *
 * The custom `system` parameter exists because a trim-and-retry re-render
 * must produce the SAME prompt as the original render, and the core has
 * no stored system buffer - so it arrives with the job. The GUI passes
 * its own system buffer and &LZGuiSession.ins; the CLI passes NULL for
 * both. */
int lz_session_job(void *ud, const char *system, LZTokenSink sink,
                   LZShouldContinue cont, void *cb_ctx,
                   char *errbuf, int errlen, LZInspect *ins);

/* Normalise the accumulated reply and append it as the assistant turn.
 * Called after the job ends INCLUDING after a stop - a stopped reply is
 * still part of the conversation. */
int lz_session_end(LZSession *s, char *errbuf, int errlen);

/* Append generated bytes to the authority buffer. This is the body of the
 * job's sink, named so that the accumulate-then-backfill path can be
 * exercised without a generator. */
void lz_session_append_reply(LZSession *s, const char *bytes, int len);

/* Drop the OLDEST exchange, in pairs, keeping the newest user turn.
 * Returns 1 when something was dropped, 0 when there is nothing left to
 * drop. Not incremental: the whole prompt is re-rendered afterwards. */
int lz_session_trim(LZSession *s);

/* How many times the turn in flight had to be trimmed. Reset by
 * lz_session_begin; read by the front end after the job ends. */
int lz_session_trimmed(const LZSession *s);

const char *lz_session_reply(const LZSession *s, int *len);
int lz_session_turns(const LZSession *s);
const char *lz_session_prompt(const LZSession *s, int *len);

/* How many tokens the NEXT turn will occupy: the current history plus
 * the generation prompt every turn re-appends. That is the quantity the
 * engine compares against seq_len before it refuses a prompt as too
 * long, so a caller showing it beside that limit is showing the number
 * the limit actually applies to.
 * Exact, not an estimate: lz_encode runs the whole BPE pass and counts
 * with out=NULL. Returns < 0 when there is no model to tokenise with, or
 * on a render failure. `system` is the custom system prompt in force, or
 * NULL - it is part of the conversation for the purpose of the window
 * count (it occupies context every turn). */
int lz_session_token_count(LZSession *s, const char *system,
                           char *errbuf, int errlen);

/* ---- last turn's statistics (see the struct fields) ---- */
int   lz_session_last_reused(const LZSession *s);
int   lz_session_last_n_out(const LZSession *s);
int   lz_session_last_prompt_tok(const LZSession *s);
double lz_session_last_ms(const LZSession *s);

#endif
