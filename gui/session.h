#ifndef LZ_GUI_SESSION_H
#define LZ_GUI_SESSION_H

#include "../src/lz_int.h"   /* lz_u64: the 64-bit type, portably */
#include <stddef.h>     /* offsetof, for the layout asserts at the bottom */

#include "chat.h"
#include "llama_zh.h"
#include "modelload.h"
/* The shared conversation core. Included by a RELATIVE path, not
 * "session.h": this header is itself named session.h, so the plain quoted
 * spelling would find itself - already guarded, therefore a no-op - and
 * the common copy would never be pulled in. -I order cannot save it,
 * because the directory of the including file is searched before -I. */
#include "../common/session.h"
/* For LZ_COMMON_SEED_* and LZ_COMMON_SYSTEM_MAX only. settings.h is pure
 * logic with no Win32 and no dependency of its own, so this direction of
 * the include is the safe one; the reverse would drag the engine headers
 * into it. */
#include "settings.h"

/* One conversation: history, the prompt for the current turn, and the
 * bytes the model produced.
 *
 * NOT named chat.h - the engine has one, and which `#include "chat.h"`
 * resolves to would depend on -I order (the same trap gui/modelload.h
 * carries a note about).
 *
 * TWO DECISIONS ABOUT THE CONVERSATION STATE, and both are the kind
 * that look like implementation detail until they are wrong:
 *
 * 1. History lives in the ENGINE's LZChatHist, not in a front-end array.
 *    lz_chat_hist_* provides that history. A second history would
 *    diverge from lz_chat_norm_history's normalised form, and that
 *    form is the precondition for resuming a conversation rather than
 *    re-rendering it. chat.h's own comment records what a non-owning
 *    history costs: correct on turn one, silently wrong from turn two.
 *
 * 2. The backfill authority is THIS buffer, never the control. Reading
 *    the reply back out of RichEdit means reading GBK and converting up,
 *    and any character GBK cannot represent has already become '?' -
 *    which would then be written into history, diverge from the KV
 *    cache, and stay wrong for the rest of the conversation. The
 *    accumulation happens on the worker thread beside the sink, in the
 *    encoding the engine speaks.
 */

/* LZPrefillMode comes from common/session.h; this header keeps no copy
 * of its own, so a TU including both headers sees one typedef.
 * PREFIX IS THE DEFAULT; see gui/main.c's ini read for why and for the
 * measurement that justified it. FULL is the control arm: every turn
 * re-renders the whole conversation
 * and starts the KV/SSM state from scratch. PREFIX asks
 * lz_gui_session_job to try
 * lz_prefix_prepare/lz_generate_resume instead, reusing the state a
 * previous turn already forwarded through - but only once pc_ready
 * says the cache actually exists for the CURRENTLY loaded model (see
 * gui/main.c's prefix_teardown and finish_job's JOB_LOAD branch). */

typedef struct {
    /* ---- core fields, in LZSession's EXACT order (see common/session.h)
     *      and with its EXACT types, so (LZSession *)&LZGuiSession is a
     *      valid flat-inheritance view and the lz_session_* functions can
     *      operate on them. The offsetof asserts at the bottom of this
     *      header turn any future drift into a compile error. GUI-only
     *      fields follow at higher offsets; the core never sees them. ---- */
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
    int n_out, n_prompt_tok, reused;
    double ms;

    /* ---- GUI-only fields, below the core prefix so the core fields
     *      above stay contiguous. Types unchanged. ---- */
    LZGuiModel *mdl;           /* borrowed; must be ready before the job */

    /* The inference inspector's own snapshot:
     * lz_gui_session_job forwards &ins into lz_session_job's optional
     * inspector, which passes it to lz_generate_resume_ex - that
     * overwrites it once per token actually sampled (never on a step
     * that just digests a prompt token - see lz_sample_ex's own
     * comment in sampler.h). Embedded, not a pointer: this struct is
     * the session's own long-lived state, and lz_generate_resume_ex
     * needs somewhere to write into that outlives any one call the
     * way s->reply already does for the generated text itself.
     * gui/worker.c reads it back through the SAME pointer, given to
     * lz_worker_start alongside ud, once per token - see that file's
     * own comment for why it cannot get there through ud (LZWorkerJob
     * is deliberately opaque to worker.c, shared with the model-load
     * job, which has no LZInspect of its own to point at). Not reset
     * between turns: a new turn's own first WM_APP_INSPECT overwrites
     * whatever the previous turn left showing, and until it arrives
     * the panel simply keeps the last thing it was told - the same
     * "the panel can lag" contract already accepts for every frame,
     * not a special case for the first one. */
    LZInspect ins;

    /* Custom system prompt, if any. Applied at render time,
       never pushed into hist - see common/session.c's render_conv_core.
       Empty means "use the engine's built-in identity prompt" (Item 9b).
       A copy, set from LZGuiSettings by lz_gui_session_apply, so the
       render sites do not need to reach into the caller's struct. The
       core has no such field: every lz_session_* call that renders takes
       it as a per-call `system` parameter, and the forwards here pass
       this buffer. */
    char system[LZ_COMMON_SYSTEM_MAX + 1];

    /* Seed policy, set by lz_gui_session_set_seed and consumed at the
       top of every lz_gui_session_begin / lz_gui_session_regen (the
       core has no seed policy; opts.rng_seed is used verbatim). seed_turn
       only exists to break ties inside one millisecond - see the
       seed_next_turn comment in gui/session.c. */
    int                seed_mode;
    lz_u64 seed_fixed;
    lz_u64 seed_turn;

    /* The worker's own callbacks, not written by lz_gui_session_job
       (the core bundles the sink/cont/ctx per call instead) - kept so
       the struct layout and any out-of-tree reader keep their types. */
    LZTokenSink      ui_sink;
    LZShouldContinue ui_cont;
    void            *ui_ctx;
} LZGuiSession;

void lz_gui_session_init(LZGuiSession *s, LZGuiModel *mdl, int think);
void lz_gui_session_free(LZGuiSession *s);

/* Clear the conversation. The model stays loaded. */
void lz_gui_session_reset(LZGuiSession *s);

/* ---- the prefix cache's lifetime, in one place ----
 *
 * The three invalidation operations live here rather than at their
 * call sites, for two reasons, and the second is the one that
 * mattered:
 *
 *   - THE RULE is written down once, here. It is:
 *       the MODEL changed  -> drop, then arm for the new one
 *       the HISTORY changed -> clear
 *     Dropping when only the history moved throws away an allocation
 *     for nothing; clearing when the model moved leaves a cache sized
 *     for weights that are gone.
 *
 *   - The three invalidation paths can be reached without a window:
 *     gui/session.c can be driven headless and never calls anything in
 *     gui/main.c. A gate has to be able to call
 *     the product's own function, not a copy of its body.
 *
 * All three forward to the core's lz_session_prefix_*; the rule and
 * the gate reasoning above apply to the functions below. */

/* The model is going away or has gone: free the cache and clear the
 * ready flag. Idempotent, and safe before any cache ever existed. */
void lz_gui_session_prefix_drop(LZGuiSession *s);

/* Build a cache for `mdl`'s weights. Returns 0 on success. FAILURE IS
 * NOT FATAL and callers must not treat it as a load failure: pc_ready
 * simply stays 0, and lz_gui_session_job's prefix branch then never
 * fires, which is the same leniency its own prepare-failure fallback
 * has one level later. */
int lz_gui_session_prefix_arm(LZGuiSession *s, LZGuiModel *mdl,
                              char *errbuf, int errlen);

/* The history changed under a cache that is still valid for this model:
 * keep the allocation, forget what it held. Safe when there is no cache
 * (lz_prefix_reset only touches three ints on the struct itself, never
 * the pointers a real arm would have allocated). */
void lz_gui_session_prefix_clear(LZGuiSession *s);

/* Sampling defaults for the current think state, taken from the engine
 * (lz_sample_defaults / lz_sample_defaults_think) rather than written
 * here. Forwards to lz_session_set_think. A front end with its own
 * numbers gives different output from the CLI on the same model, and
 * the two drift apart the first time either is tuned. */
void lz_gui_session_set_think(LZGuiSession *s, int think);

/* Apply a settings snapshot. The one above sets a whole default block
 * from the mode; this one is what the front end calls after
 * gui/settings.c has decided, because "which mode" and "what the user
 * chose" stop agreeing the moment the user chooses anything.
 *
 * A WHOLE LZGuiSettings, not a parameter per setting. Five positional
 * arguments (think, temp, topp, rep, max_new) - three of them floats
 * in a row - would be a call nobody can read and a transposition the
 * compiler cannot catch. One struct also means the setting after this
 * one is a field, not a signature change at every call site. */
void lz_gui_session_apply(LZGuiSession *s, const LZGuiSettings *set);

/* Replace the custom system prompt in force. Empty restores the
 * engine's built-in identity. Split out of lz_gui_session_apply so a
 * caller changing ONLY the system prompt (the settings dialog after an
 * OK, or "restore defaults") does not have to re-feed every other
 * setting; the session needs to be told because the next render has to
 * use it. */
void lz_gui_session_set_system(LZGuiSession *s, const char *utf8);

/* The seed POLICY for every turn from here on. Stored, not
 * applied: lz_gui_session_begin / lz_gui_session_regen draw the actual
 * rng_seed at the top of each turn, which is what makes a re-run of the
 * same turn - the "regenerate" command this is a prerequisite for -
 * come back different.
 *
 * mode LZ_COMMON_SEED_RANDOM ignores `fixed` and takes a fresh value per
 * turn; LZ_COMMON_SEED_FIXED uses `fixed` verbatim, every turn, which is
 * the reproducibility case.
 *
 * The random value is the millisecond clock TIMES 1000 PLUS a per-turn
 * counter, the same shape cli_main.c uses, and the counter is not
 * decoration: two turns inside one millisecond would otherwise get the
 * same seed, and on this hardware "two turns in one millisecond" is
 * exactly what a scripted self-test does. */
void lz_gui_session_set_seed(LZGuiSession *s, int mode,
                             lz_u64 fixed);

/* Push the user turn and render the whole conversation into prompt.
 * Full-history mode: correctness first, resume later. Draws the
 * per-turn seed first, then forwards to
 * lz_session_begin with this session's system buffer. Returns 0, or an
 * LZ_ERR_* code with errbuf set. */
int lz_gui_session_begin(LZGuiSession *s, const char *user_utf8, int len,
                         char *errbuf, int errlen);

/* Another attempt at the user turn history ALREADY ends with - the
 * "regenerate" command. Everything lz_gui_session_begin
 * does except the push, and begin is literally push-then-this, so the
 * two cannot drift: a new per-turn reset added to one is added to both.
 *
 * Requires history to end with a user message; returns LZ_ERR_NO_USER
 * if it does not, rather than rendering a conversation whose last turn
 * is an assistant reply (lz_chat_render would happily append a
 * generation prompt to that, and the model would answer itself).
 * The FRONT END is what makes that true, by popping the previous reply
 * before calling this - see gui/main.c's rollback_last.
 *
 * A fresh seed is drawn here exactly as it is in begin, which is what
 * makes the second attempt differ from the first; with
 * LZ_COMMON_SEED_FIXED it deliberately does not, and a user who fixed the
 * seed asking for the same answer twice is asking for the same answer.
 * Returns 0, or an LZ_ERR_* code with errbuf set. */
int lz_gui_session_regen(LZGuiSession *s, char *errbuf, int errlen);

/* LZWorkerJob. `ud` is the session. Forwards to lz_session_job, keeping
 * this GUI function's EXACT LZWorkerJob shape (gui/worker.c calls it that
 * way); the core's extra system/inspector parameters are supplied here as
 * this session's own buffer and &ins. */
int lz_gui_session_job(void *ud, LZTokenSink sink, LZShouldContinue cont,
                       void *cb_ctx, char *errbuf, int errlen);

/* Normalise the accumulated reply and append it as the assistant turn.
 * Called after the job ends INCLUDING after a stop - a stopped reply is
 * still part of the conversation, and dropping it would make the next
 * turn's render disagree with what the user is looking at. */
int lz_gui_session_end(LZGuiSession *s, char *errbuf, int errlen);

/* Append generated bytes to the authority buffer. This is the body of
 * the job's sink, named so that the accumulate-then-backfill path can be
 * exercised without a generator: everything from here to the assistant
 * turn landing in history is front-end logic, and waiting for a trained
 * model to test it would mean testing it never. */
void lz_gui_session_append_reply(LZGuiSession *s, const char *bytes, int len);

/* Drop the OLDEST exchange, in pairs, keeping the newest user turn.
 *
 * The context-full path. Trimming in pairs is not tidiness: dropping
 * a lone user turn leaves the assistant reply that
 * answered it at the front of the conversation, and the render then
 * opens with a reply to a question nobody can see - which the model
 * reads as the conversation having started that way.
 *
 * Returns 1 when something was dropped, 0 when there is nothing left to
 * drop (one turn, and it does not fit - a single message longer than the
 * whole context is not a trimming problem).
 *
 * Not incremental, and it cannot be: the SSM state has no rollback, so
 * dropping history in place is impossible and the whole prompt is
 * re-rendered afterwards. Correctness over speed, deliberately. */
int lz_gui_session_trim(LZGuiSession *s);

/* How many times the turn in flight had to be trimmed. Reset by
 * lz_gui_session_begin; read by the front end after the job ends, so
 * the transcript can say so. The worker cannot say it itself - anything
 * it writes to the sink lands in history as part of the reply. */
int lz_gui_session_trimmed(const LZGuiSession *s);

const char *lz_gui_session_reply(const LZGuiSession *s, int *len);
int lz_gui_session_turns(const LZGuiSession *s);
const char *lz_gui_session_prompt(const LZGuiSession *s, int *len);

/* How many tokens the CURRENT history renders to, without the
 * generation prompt. Exact, not an estimate: lz_encode runs the whole
 * BPE pass and counts with out=NULL. Returns < 0 when there is no
 * model to tokenise with, or on a render failure.
 *
 * Called when idle, once per turn - never per token. The pass is the
 * same one lz_generate runs anyway, and on a Pentium II one extra pass
 * per turn is a cost the user can see the benefit of; one per token
 * would not be. */
int lz_gui_session_token_count(LZGuiSession *s, char *errbuf, int errlen);

/* ---- compile-time layout asserts ----
 *
 * LZGuiSession's core prefix MUST sit at the same offsets as LZSession's
 * fields, or casting s to (LZSession *) - which every forwarding function
 * in gui/session.c does - is undefined behaviour. C99 has no
 * _Static_assert, so each check is a typedef of an array sized -1 when
 * the offsets disagree: a negative array size is a hard compile error
 * naming the typedef, which names the field. Any future reorder of either
 * struct therefore fails the build here instead of silently corrupting
 * the conversation. */
typedef char lz_gui_layout_assert_model     [(offsetof(LZGuiSession, model)     == offsetof(LZSession, model))     ? 1 : -1];
typedef char lz_gui_layout_assert_tok       [(offsetof(LZGuiSession, tok)       == offsetof(LZSession, tok))       ? 1 : -1];
typedef char lz_gui_layout_assert_state     [(offsetof(LZGuiSession, state)     == offsetof(LZSession, state))     ? 1 : -1];
typedef char lz_gui_layout_assert_hist      [(offsetof(LZGuiSession, hist)      == offsetof(LZSession, hist))      ? 1 : -1];
typedef char lz_gui_layout_assert_prompt    [(offsetof(LZGuiSession, prompt)    == offsetof(LZSession, prompt))    ? 1 : -1];
typedef char lz_gui_layout_assert_reply     [(offsetof(LZGuiSession, reply)     == offsetof(LZSession, reply))     ? 1 : -1];
typedef char lz_gui_layout_assert_reply_len [(offsetof(LZGuiSession, reply_len) == offsetof(LZSession, reply_len)) ? 1 : -1];
typedef char lz_gui_layout_assert_reply_cap [(offsetof(LZGuiSession, reply_cap) == offsetof(LZSession, reply_cap)) ? 1 : -1];
typedef char lz_gui_layout_assert_think     [(offsetof(LZGuiSession, think)     == offsetof(LZSession, think))     ? 1 : -1];
typedef char lz_gui_layout_assert_opts      [(offsetof(LZGuiSession, opts)      == offsetof(LZSession, opts))      ? 1 : -1];
typedef char lz_gui_layout_assert_trimmed   [(offsetof(LZGuiSession, trimmed)   == offsetof(LZSession, trimmed))   ? 1 : -1];
typedef char lz_gui_layout_assert_prefill   [(offsetof(LZGuiSession, prefill)   == offsetof(LZSession, prefill))   ? 1 : -1];
typedef char lz_gui_layout_assert_pc        [(offsetof(LZGuiSession, pc)        == offsetof(LZSession, pc))        ? 1 : -1];
typedef char lz_gui_layout_assert_pc_ready  [(offsetof(LZGuiSession, pc_ready)  == offsetof(LZSession, pc_ready))  ? 1 : -1];
typedef char lz_gui_layout_assert_n_out     [(offsetof(LZGuiSession, n_out)     == offsetof(LZSession, n_out))     ? 1 : -1];
typedef char lz_gui_layout_assert_n_prompt_tok [(offsetof(LZGuiSession, n_prompt_tok) == offsetof(LZSession, n_prompt_tok)) ? 1 : -1];
typedef char lz_gui_layout_assert_reused    [(offsetof(LZGuiSession, reused)    == offsetof(LZSession, reused))    ? 1 : -1];
typedef char lz_gui_layout_assert_ms        [(offsetof(LZGuiSession, ms)        == offsetof(LZSession, ms))        ? 1 : -1];

#endif
