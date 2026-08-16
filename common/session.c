/* Front-end-agnostic conversation core. GUI-only concerns live in
 * gui/session.c, not here: the custom system prompt is a per-call
 * parameter, the per-turn seed policy, the EOS/stop-string policy and
 * the inference-inspector snapshot (LZInspect) are the caller's, and
 * the worker callbacks are forwarded per call instead of stored on the
 * struct. See common/session.h.
 *
 * Thread split: begin/end/reset run on the UI thread, the job runs on
 * the worker. They never overlap - the caller
 * must not call begin/end/reset while a job is running. The reply buffer
 * is written only by the job and read only after the job's thread has
 * been joined, so it needs no lock - and saying so here is cheaper than a
 * lock that hides the fact that the ordering is what makes this safe. */

#include <stdlib.h>
#include <string.h>

#include "err.h"
#include "sampler.h"
#include "chat.h"
#include "llama_zh.h"
#include "session.h"

void lz_session_init(LZSession *s, LZModel *model, LZTokenizer *tok,
                     LZRunState *state, int think) {
    if (!s) return;
    memset(s, 0, sizeof *s);
    lz_chat_hist_init(&s->hist);
    lz_chat_buf_init(&s->prompt);
    s->model = model;
    s->tok = tok;
    s->state = state;
    lz_gen_opts_defaults(&s->opts);
    /* The core holds no seed/EOS/stop policy: opts.rng_seed stays 0
       (the sampler substitutes 1) and n_eos/n_stop stay 0; each caller
       sets its own before the job - see the file header. */
    lz_session_set_think(s, think);
}

void lz_session_set_think(LZSession *s, int think) {
    if (!s) return;
    s->think = think ? 1 : 0;
    if (s->think) lz_sample_defaults_think(&s->opts.sample);
    else          lz_sample_defaults(&s->opts.sample);
}

/* Render the conversation through lz_chat_render, with a custom system
   prompt prepended when `system` is non-NULL and non-empty.
 *
 * WHY THE SYSTEM PROMPT IS A PARAMETER, NOT A FIELD: the core holds no
 * system buffer, so every render site takes it per call and
 * lz_session_begin / lz_session_regen / lz_session_job all thread it
 * through.
 *
 * WHY NOT STORED IN HIST: three reasons, and the first two are
 * correctness, the third is where the others would leak.
 *
 *   1. REPLACE, not append. lz_chat_render refuses any system message
 *      that is not at index 0 (LZ_ERR_SYSTEM_FIRST). If the system prompt
 *      lived in hist and the user then changed it, the history would hold
 *      an OLD system at index 0 and a NEW one could not be inserted; the
 *      only honest outcome would be wiping the history every time the
 *      setting changed. That is data loss with no technical cause.
 *   2. TRIM PROTECTION. lz_session_trim drops the oldest EXCHANGE in
 *      pairs. A system message stored in hist is not an exchange, and
 *      every trim would have to remember it is there - one dropped system
 *      message later, the render hits SYSTEM_FIRST and the conversation is
 *      dead. Keep the conversation and the identity separate and there is
 *      nothing to get wrong.
 *   3. PREFIX CACHE. The conversation prefix is keyed on the token stream
 *      of the RENDER. A system message that lives in the settings rather
 *      than the history still changes the render, so the caller must still
 *      lz_session_prefix_clear when the setting changes - but it does not
 *      have to rebuild the history to do it.
 *
 * EMPTY / NULL -> pass msgs through unchanged, and lz_chat_render's Item
 * 9b injects the trained built-in. */
static int render_conv_core(LZSession *s, const char *system, int add_gen,
                            LZChatBuf *out, char *errbuf, int errlen) {
    if (!system || system[0] == '\0')
        return lz_chat_render(s->hist.msgs, s->hist.n, add_gen, s->think,
                              out, errbuf, errlen);
    {
        LZChatMsg sys;
        LZChatMsg *msgs;
        int n = s->hist.n;
        int i;
        /* No heap: the conversation is at most LZ_CHAT_HIST_MAX messages,
           so a fixed local array covers every reachable case - see the
           constant in chat.h. The system string itself is borrowed from
           the caller for the duration of this call only. */
        LZChatMsg buf[LZ_CHAT_HIST_MAX + 1];

        sys.role = LZ_ROLE_SYSTEM;
        sys.content = system;
        sys.len = (int)strlen(system);
        buf[0] = sys;
        for (i = 0; i < n; i++) buf[i + 1] = s->hist.msgs[i];
        msgs = buf;
        return lz_chat_render(msgs, n + 1, add_gen, s->think,
                              out, errbuf, errlen);
    }
}

void lz_session_free(LZSession *s) {
    if (!s) return;
    lz_chat_hist_free(&s->hist);
    lz_chat_buf_free(&s->prompt);
    free(s->reply);
    s->reply = NULL;
    s->reply_len = s->reply_cap = 0;
    /* Symmetric with lz_session_init, which zeroes the whole struct
       including pc - freeing it here regardless of pc_ready is
       safe (lz_prefix_free tolerates a zeroed LZPrefixCache) and keeps
       this function a complete counterpart rather than one that only
       undoes some of init's work. */
    lz_prefix_free(&s->pc);
    s->pc_ready = 0;
    /* model / tok / state are BORROWED pointers, owned by the caller who
       loaded the model and allocated the run state. lz_session_free
       deliberately does NOT free them - only what lz_session_init's own
       side of this struct allocated. */
}

void lz_session_prefix_drop(LZSession *s) {
    if (!s) return;
    if (s->pc_ready) {
        lz_prefix_free(&s->pc);
        s->pc_ready = 0;
    }
}

int lz_session_prefix_arm(LZSession *s, LZModel *model,
                          char *errbuf, int errlen) {
    int rc;
    if (!s || !model) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        return rc;
    }
    rc = lz_prefix_init(&s->pc, model, errbuf, errlen);
    s->pc_ready = (rc == 0);
    return rc;
}

void lz_session_prefix_clear(LZSession *s) {
    if (!s) return;
    lz_prefix_reset(&s->pc);
}

void lz_session_reset(LZSession *s) {
    if (!s) return;
    lz_chat_hist_reset(&s->hist);
    s->reply_len = 0;
    /* The conversation just emptied, so whatever prefix was cached for
       it corresponds to a history that was just cleared - keeping it
       would let the next turn's lz_prefix_prepare match against that.
       The MODEL has not changed, so this is a clear and not a drop -
       see session.h's rule. */
    lz_session_prefix_clear(s);
}

static int reply_grow(LZSession *s, int need) {
    int cap = s->reply_cap ? s->reply_cap : 4096;
    char *p;
    while (cap < need) cap *= 2;
    p = (char *)realloc(s->reply, (size_t)cap);
    if (!p) return 0;
    s->reply = p;
    s->reply_cap = cap;
    return 1;
}

/* Runs on the worker thread, once per generated token. Accumulate FIRST,
 * then forward: if the UI side ever throws, the authority copy is already
 * complete. */
void lz_session_append_reply(LZSession *s, const char *bytes, int len) {
    if (!s || !bytes || len <= 0) return;
    if (s->reply_len + len > s->reply_cap && !reply_grow(s, s->reply_len + len))
        return;
    memcpy(s->reply + s->reply_len, bytes, (size_t)len);
    s->reply_len += len;
}

/* The worker's own callbacks, forwarded to lz_generate_resume. The engine
 * gives sink and cont ONE ctx between them, and that ctx is the session
 * (`ud`). The core's struct does not store the callbacks, so the job
 * bundles (session + callbacks) into one stack-local context for the
 * duration of the generate call. Safe because the job runs to completion
 * before this context goes out of scope. */
typedef struct {
    LZSession        *s;
    LZTokenSink       sink;
    LZShouldContinue  cont;
    void             *ctx;
} SessionCallbacks;

static void session_sink(const char *bytes, int len, void *ctx) {
    SessionCallbacks *cb = (SessionCallbacks *)ctx;
    lz_session_append_reply(cb->s, bytes, len);
    if (cb->sink) cb->sink(bytes, len, cb->ctx);
}

static int session_cont(void *ctx) {
    SessionCallbacks *cb = (SessionCallbacks *)ctx;
    if (!cb || !cb->cont) return 1;
    return cb->cont(cb->ctx);
}

int lz_session_trim(LZSession *s) {
    int drop, i;
    if (!s) return 0;
    /* Keep at least the newest user turn. With 1 or 2 messages there is no
       older exchange to give up. */
    if (s->hist.n < 3) return 0;
    /* Two if the front is a complete exchange, one if the history is
       somehow odd - which it should not be, but a trim that assumes an
       invariant it does not check is how a history ends up starting with
       an assistant turn. */
    drop = (s->hist.n >= 3 &&
            s->hist.msgs[0].role == LZ_ROLE_USER &&
            s->hist.msgs[1].role == LZ_ROLE_ASSISTANT) ? 2 : 1;
    if (s->hist.n - drop < 1) return 0;

    for (i = 0; i < drop; i++) {
        free(s->hist.owned[i]);
        s->hist.owned[i] = NULL;
    }
    for (i = drop; i < s->hist.n; i++) {
        s->hist.msgs[i - drop] = s->hist.msgs[i];
        s->hist.owned[i - drop] = s->hist.owned[i];
    }
    for (i = s->hist.n - drop; i < s->hist.n; i++) {
        s->hist.owned[i] = NULL;
        memset(&s->hist.msgs[i], 0, sizeof s->hist.msgs[i]);
    }
    s->hist.n -= drop;
    s->trimmed++;
    return 1;
}

int lz_session_trimmed(const LZSession *s) {
    return s ? s->trimmed : 0;
}

int lz_session_begin(LZSession *s, const char *user_utf8, int len,
                     const char *system, char *errbuf, int errlen) {
    int rc;
    if (!s || !user_utf8) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        return rc;
    }
    if (lz_chat_hist_push(&s->hist, LZ_ROLE_USER, user_utf8, len,
                          errbuf, errlen) != 0)
        /* Collapsed to ALLOC, not the push's real code - same contract
           as lz_session_end: the precise cause (LZ_ERR_HIST_FULL /
           LZ_ERR_HIST_ALLOC) is already in errbuf; rc is the coarse
           success/failure signal. A caller wanting the cause reads
           errbuf. */
        return LZ_ERR_ALLOC;
    /* The push IS the only difference from regenerating, so the rest is
       that function rather than a copy of it - see session.h. */
    return lz_session_regen(s, system, errbuf, errlen);
}

int lz_session_regen(LZSession *s, const char *system,
                     char *errbuf, int errlen) {
    int rc;
    if (!s) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        return rc;
    }
    /* Checked here and not only at the call site: begin comes through this
       function too, so the invariant "the render's last turn is the
       user's" is asserted on EVERY path that renders a generation prompt,
       not just on the new one. */
    if (s->hist.n <= 0 ||
        s->hist.msgs[s->hist.n - 1].role != LZ_ROLE_USER) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NO_USER);
        return rc;
    }
    s->reply_len = 0;
    s->trimmed = 0;
    /* No per-turn reseed here: the core has no seed policy. opts.rng_seed
       is used verbatim for this turn; a front end
       that wants a fresh seed per turn sets it before calling in. */
    return render_conv_core(s, system, 1, &s->prompt, errbuf, errlen);
}

int lz_session_job(void *ud, const char *system, LZTokenSink sink,
                   LZShouldContinue cont, void *cb_ctx,
                   char *errbuf, int errlen, LZInspect *ins) {
    LZSession *s = (LZSession *)ud;
    SessionCallbacks cb;
    /* rc defaults the way lz_generate_resume_ex's own does: every path
       below assigns it, and a default that is an error rather than
       garbage is what keeps a future one that forgets from returning
       success. `ran` says whether this pass has already generated. */
    int n_out = 0, rc = LZ_ERR_INTERNAL, ran;
    int start_pos = 0, reused = 0;
    double ms = 0.0;

    if (!s) { LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG); return rc; }
    if (!(s->model && s->tok && s->state)) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NOT_OPEN);
        return rc;
    }

    cb.s = s;
    cb.sink = sink;
    cb.cont = cont;
    cb.ctx = cb_ctx;
    s->reply_len = 0;

    /* No EOS/stop policy here - those are the CALLER's sampling
       configuration, set in s->opts before the job runs (the CLI's
       --no-eos / --stop, the GUI's EOS + STOPS in gui/session.c). */

    for (;;) {
        s->reply_len = 0;
        /* Hoisted to function scope so the success exit below can record
           them into the session's last-turn statistics. Reset here every
           pass: a trim-and-retry must not carry a previous pass's values
           into the statistics. */
        start_pos = 0;
        reused = 0;
        ran = 0;
        /* Two paths, re-decided every pass through this loop (not just
           once) - a trim-and-retry re-renders s->prompt, and the prefix
           path has to be offered that fresh render too, not forced onto
           whatever an earlier pass decided. */
        if (s->prefill == LZ_PREFILL_PREFIX && s->pc_ready) {
            /* The split is where the reusable part of the render ends -
               for this template, everything before the generation prompt
               tail. lz_prefix_prepare's own comment fixes this formula;
               it is not a guess. */
            int split = s->prompt.len -
                        (int)strlen(lz_chat_gen_prompt_tail(s->think));
            int suffix_off = 0;
            /* The same callbacks and ctx the generate call below gets:
               on a cache miss this forwards nearly the whole prompt, so
               it is where a front end's indicator has to be fed from. */
            LZPrefillHooks hooks;
            int prc;
            hooks.on_prefill = s->opts.on_prefill;
            hooks.cont       = session_cont;
            hooks.ctx        = &cb;
            prc = lz_prefix_prepare(&s->pc, s->model, s->tok, s->state,
                                    s->prompt.s, s->prompt.len, split,
                                    &start_pos, &suffix_off, &reused,
                                    &hooks, errbuf, errlen);
            if (prc == 0) {
                /* lz_generate_resume_ex with the job's OPTIONAL
                   inspector: NULL (CLI) IS lz_generate_resume, and the
                   GUI passes &LZGuiSession.ins for the inference
                   inspector - see llama_zh.h. */
                rc = lz_generate_resume_ex(s->model, s->tok, s->state, start_pos,
                                           s->prompt.s + suffix_off,
                                           s->prompt.len - suffix_off,
                                           &s->opts, session_sink, session_cont,
                                           &cb, &n_out, &ms, errbuf, errlen,
                                           ins);
                ran = 1;
            } else if (prc == LZ_ERR_CANCELLED) {
                /* Stopped during the prefix forward. Reported as a normal
                   empty turn, not an error: a stop pressed here and one
                   pressed after the first token are the same user action,
                   and a front end that had to tell them apart would be
                   deciding a thing the engine already knows. */
                s->n_out = 0;
                s->reused = reused;
                s->ms = 0;
                s->n_prompt_tok = 0;
                /* The SAME field lz_generate_resume_ex writes on its own
                   cancel path, and the one the GUI reads to decide
                   whether to say "stopped" - without this it keeps the
                   previous turn's value, so a stop pressed here would
                   silently produce no notice at all. Returning LZ_ERR_OK
                   without setting it is what makes the two stops look
                   alike to the caller; the reason has to travel with
                   it. */
                s->opts.out_finish = LZ_FINISH_CANCELLED;
                return LZ_ERR_OK;
            } else {
                /* prepare failed. The full path below resets the state
                   itself, so falling through is correct whatever prepare
                   left behind - and falling through rather than returning
                   is deliberate: a cache problem must not become a failed
                   turn. Persists for the REST of this
                   session (prefill stays FULL from here on, not just for
                   this one pass) - once the cache has proven unreliable
                   for this conversation, retrying it every turn would just
                   fail the same way again. */
                s->prefill = LZ_PREFILL_FULL;
            }
        }
        /* Anything the prefix branch did not already generate lands
           here: a FULL request, a PREFIX request whose cache never
           became ready (lz_prefix_init / lz_session_prefix_arm failed),
           and a prefix attempt that gave up and switched to FULL above.
           Asking "did this pass generate yet" rather than re-deriving it
           from (prefill, pc_ready) is what keeps that list from having
           to be right: the old test re-read state the branch above had
           just mutated, so its correctness depended on a side effect
           three lines up, and a future branch that exits the block
           without generating would have fallen through BOTH and returned
           an uninitialized rc. `ran` cannot fail that way - the fallback
           is re-forwarding, which is slow and correct. */
        if (!ran) {
            /* Full mode: the prompt already contains every turn, so the
               state must start empty or the prefix would be forwarded
               twice. */
            lz_state_reset(s->state, s->model);
            rc = lz_generate_resume_ex(s->model, s->tok, s->state, 0,
                                       s->prompt.s, s->prompt.len,
                                       &s->opts, session_sink, session_cont,
                                       &cb, &n_out, &ms, errbuf, errlen, ins);
        }
        if (rc != LZ_ERR_PROMPT_LONG) {
            /* Last turn's statistics, read by the front end after the job
               (lz_session_last_*). out_prompt_tokens is what the engine
               counted for the prompt it actually forwarded THIS pass;
               adding start_pos (the cached prefix it skipped) yields the
               whole render's token count, the same number either path
               produces. */
            s->n_out = n_out;
            s->reused = reused;
            s->ms = ms;
            s->n_prompt_tok = start_pos + s->opts.out_prompt_tokens;
            return rc;
        }

        /* Drop the oldest exchange and render the whole thing again. There
           is no in-place version of this - the SSM state cannot be rolled
           back, so "forget the first turn" and "start over without it"
           are the same operation.
           reply_len is cleared each pass: a prompt that was too long may
           still have emitted nothing, but a retry that kept whatever the
           previous attempt produced would splice two answers together. */
        if (!lz_session_trim(s)) return rc;
        if (render_conv_core(s, system, 1, &s->prompt, errbuf, errlen) != 0)
            return rc;
        /* Total 0: the prefill this pass reported is abandoned, start
           over - see LZProgress. The pass that just failed may already
           have forwarded most of the old render through
           lz_prefix_prepare and reported it, and without this a front
           end accumulating segments would add that work to the retry's
           and show a total for a prompt that no longer exists. */
        if (s->opts.on_prefill) s->opts.on_prefill(0, 0, &cb);
    }
}

int lz_session_end(LZSession *s, char *errbuf, int errlen) {
    char *norm;
    int n, rc;
    if (!s) { LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG); return rc; }
    /* Nothing generated - a stop before the first token, or a failure.
       Pushing an empty assistant turn would put a blank reply into the
       next render. */
    if (s->reply_len <= 0) return 0;
    /* Sized off the reply, not a fixed cap. The normaliser only ever
       DROPS bytes (reasoning, stray '\r') - see lz_chat_norm_history's
       own comment - so its output is always <= reply_len and this
       allocation can never truncate. Not static or stack: reply_len is
       unbounded here and iron law six rule 4 is about the target's
       small stack. */
    norm = (char *)malloc((size_t)s->reply_len + 1);
    if (!norm) return LZ_ERR_ALLOC;
    n = lz_chat_norm_history(s->reply, s->reply_len, norm, s->reply_len + 1,
                             errbuf, errlen);
    /* Both failure exits collapse to LZ_ERR_ALLOC on purpose: the callee
       has already written the REAL code into errbuf (LZ_ERR_NULL_ARG
       for the normaliser, LZ_ERR_HIST_FULL / LZ_ERR_HIST_ALLOC for the
       push), and returning a distinct code here would mean re-parsing
       that buffer. A caller that wants the precise cause reads errbuf,
       not rc. LZ_ERR_TRUNC cannot occur: the buffer is sized
       reply_len + 1 and the normaliser only shrinks. */
    if (n < 0) { free(norm); return LZ_ERR_ALLOC; }
    if (n == 0) { free(norm); return 0; }
    if (lz_chat_hist_push(&s->hist, LZ_ROLE_ASSISTANT, norm, n,
                          errbuf, errlen) != 0) {
        free(norm);
        return LZ_ERR_ALLOC;
    }
    free(norm);
    return 0;
}

const char *lz_session_reply(const LZSession *s, int *len) {
    if (len) *len = s ? s->reply_len : 0;
    return (s && s->reply) ? s->reply : "";
}

int lz_session_turns(const LZSession *s) { return s ? s->hist.n : 0; }

const char *lz_session_prompt(const LZSession *s, int *len) {
    if (len) *len = s ? s->prompt.len : 0;
    return (s && s->prompt.s) ? s->prompt.s : "";
}

int lz_session_token_count(LZSession *s, const char *system,
                           char *errbuf, int errlen) {
    LZChatBuf b;
    int n;
    if (!s || !(s->model && s->tok && s->state)) return -1;
    if (s->hist.n == 0) return 0;
    lz_chat_buf_init(&b);
    /* add_generation_prompt = 1, which is the SAME render lz_session_job
       hands to the engine - so this returns the quantity the engine
       compares against seq_len (start_pos + n_prompt, generate.c), not a
       number that merely resembles it.
       It was 0 once, on the reasoning that the generation prompt is not
       part of the conversation. It is not, but it is part of what every
       turn must fit: measured on this tokenizer the tail is 7 tokens
       with thinking on and 10 with it off, so a window shown as full at
       2048 dropped the oldest exchange at 2041. A headroom number that
       runs out before it reads empty is the shape iron law four names.
       Not a jump either - the tail is appended every turn, so it is a
       constant offset, not a fluctuation.
       render_conv_core, not lz_chat_render: a custom system prompt is
       part of the CONVERSATION for the purpose of the window count - it
       occupies context every turn. */
    if (render_conv_core(s, system, 1, &b, errbuf, errlen) != 0) {
        lz_chat_buf_free(&b);
        return -1;
    }
    /* out = NULL, out_cap = 0 is lz_encode's count-only form. The full BPE
       pass still runs, so this is exact and not an estimate - which
       matters, because the number is shown next to a hard limit that drops
       the user's oldest exchange when it is crossed. */
    n = lz_encode(s->tok, b.s, b.len, 0, 0, NULL, 0);
    lz_chat_buf_free(&b);
    return n;
}

/* ---- last turn's statistics ---- */

int lz_session_last_reused(const LZSession *s) {
    return s ? s->reused : 0;
}

int lz_session_last_n_out(const LZSession *s) {
    return s ? s->n_out : 0;
}

int lz_session_last_prompt_tok(const LZSession *s) {
    return s ? s->n_prompt_tok : 0;
}

double lz_session_last_ms(const LZSession *s) {
    return s ? s->ms : 0.0;
}
