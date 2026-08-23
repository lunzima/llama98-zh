/* Conversation state. See session.h for the two decisions that shape it.
 *
 * TASK 2: this file is now a THIN FORWARDING LAYER over the front-end-
 * agnostic core in common/session.c. LZGuiSession's leading fields are
 * LZSession's exact layout (flat inheritance - see gui/session.h's offsetof
 * asserts), so every function here just casts `s` to LZSession* and hands
 * the call to the core, adding only the GUI-only concerns the core
 * deliberately dropped: the stored custom system prompt (passed per call
 * as the `system` parameter), the per-turn seed policy, the model-ready
 * gate for the job and the token count, and the LZInspect snapshot.
 *
 * Thread split: begin/end/reset run on the UI thread, the job runs on the
 * worker. They never overlap - the caller must not call begin/end/reset
 * while a job is running. The reply buffer is written only by the job and
 * read only after the worker join, so it needs no lock - and saying so
 * here is cheaper than a lock that hides the fact that the ordering is
 * what makes this safe. */
#include <string.h>

#include "compat.h"     /* lz_time_ms, for the per-turn random seed */
#include "err.h"
#include "session.h"

/* Stop strings. A chat model learns to write <|im_end|>, but a small one
 * also writes "<|im_start|>user" and answers itself. BORROWED for the
 * whole lz_generate call (llama_zh.h says so), so these are static
 * storage rather than anything with a lifetime.
 * The GUI always wants EOS plus these stops; this is the GUI's policy,
 * applied here on the job (the core holds no stop policy of its own -
 * the CLI's --no-eos / --stop are set by src/cli_main.c). */
static const char *const STOPS[] = { "<|im_start|>", "<|endoftext|>" };
#define N_STOPS (int)(sizeof STOPS / sizeof STOPS[0])

void lz_gui_session_init(LZGuiSession *s, LZGuiModel *mdl, int think) {
    if (!s) return;
    memset(s, 0, sizeof *s);
    /* The core borrows mdl's three engine members; s->model/s->tok/
       s->state stay the embedded addresses of LZGuiModel for the session's
       whole life, which is exactly what lz_gui_session_job's ready gate
       keys off later. */
    lz_session_init((LZSession *)s, mdl ? &mdl->model : NULL,
                    mdl ? &mdl->tok : NULL, mdl ? &mdl->state : NULL, think);
    s->mdl = mdl;
    /* A fresh session is DETERMINISTIC until somebody says otherwise. The
       random-per-turn policy is the GUI's, applied by apply_settings, NOT
       the session's default - a memset default of RANDOM would silently
       make every headless caller non-reproducible. A parity gate needs a
       fixed seed; a chat window needs a fresh one; the layer that knows
       which is which is the one holding the settings. */
    s->seed_mode = LZ_COMMON_SEED_FIXED;
    s->seed_fixed = 0;
}

void lz_gui_session_free(LZGuiSession *s) {
    /* No GUI-only teardown: mdl is borrowed, system is an array, ins is
       embedded, the seed fields are scalars and the ui_* pointers are
       borrowed. Everything owned lives on the core prefix. */
    lz_session_free((LZSession *)s);
}

void lz_gui_session_reset(LZGuiSession *s) {
    lz_session_reset((LZSession *)s);
}

void lz_gui_session_prefix_drop(LZGuiSession *s) {
    lz_session_prefix_drop((LZSession *)s);
}

int lz_gui_session_prefix_arm(LZGuiSession *s, LZGuiModel *mdl,
                              char *errbuf, int errlen) {
    /* &mdl->model IS s->model (lz_gui_session_init stored it), which is
       exactly the invariant lz_session_prefix_arm documents - see
       common/session.h. */
    return lz_session_prefix_arm((LZSession *)s,
                                 mdl ? &mdl->model : NULL, errbuf, errlen);
}

void lz_gui_session_prefix_clear(LZGuiSession *s) {
    lz_session_prefix_clear((LZSession *)s);
}

void lz_gui_session_set_think(LZGuiSession *s, int think) {
    lz_session_set_think((LZSession *)s, think);
}

void lz_gui_session_apply(LZGuiSession *s, const LZGuiSettings *set) {
    if (!s || !set) return;
    /* Mode first: it rewrites the whole sampling block from the engine's
       defaults, so every explicit value has to be written after it or
       they would be the things that get overwritten. The sampling fields
       live on the CORE prefix (LZGenOpts opts); the GUI-only fields below
       never read them. */
    lz_gui_session_set_think(s, set->think);
    s->opts.sample.temperature = set->temp;
    s->opts.sample.topp        = set->topp;
    /* Think-block dynamic temperature (task #19). The GUI always enables
       it - the dialog has a value box but no on/off, and "manual flag 0
       but enabled" is the GUI's default (settings.h) - so the enabled
       flag is a constant here, unlike the CLI where --think-temp opts in.
       Written after set_think above, which rewrote the whole sampling
       block from the preset (both presets leave think_temp_enabled 0). */
    s->opts.sample.temp_think = set->think_temp;
    s->opts.sample.think_temp_enabled = 1;
    /* Sent even when it equals 1.0, which is the identity. sampler.h is
       explicit about this: HuggingFace defaults to 1.0, llama.cpp to
       1.1, vLLM to 1.0, so "leave it out and let the default apply"
       means three different things depending on who is reading. */
    s->opts.sample.repetition_penalty = set->rep;
    /* -1 goes down as -1. llama_zh.h reads "<=0 means use the model's
       seq_len", so this is the same request 0 would make - see
       LZ_COMMON_MAXNEW_UNLIMITED for why the spelling is worth keeping. */
    s->opts.max_new_tokens = set->max_new;
    lz_gui_session_set_system(s, set->system);
}

void lz_gui_session_set_system(LZGuiSession *s, const char *utf8) {
    if (!s) return;
    if (!utf8) { s->system[0] = '\0'; return; }
    {
        size_t n = strlen(utf8);
        if (n > LZ_COMMON_SYSTEM_MAX) n = LZ_COMMON_SYSTEM_MAX;
        memcpy(s->system, utf8, n);
        s->system[n] = '\0';
    }
}

void lz_gui_session_set_seed(LZGuiSession *s, int mode,
                             lz_u64 fixed) {
    if (!s) return;
    s->seed_mode = (mode == LZ_COMMON_SEED_FIXED) ? LZ_COMMON_SEED_FIXED
                                               : LZ_COMMON_SEED_RANDOM;
    s->seed_fixed = fixed;
}

/* Draw the seed for the turn that is about to start. GUI-only: the core
   has no seed policy and uses opts.rng_seed verbatim, so this writes the
   GUI's policy into the core field before the turn's begin/regen runs.
   Called from lz_gui_session_begin / lz_gui_session_regen ONLY - putting
   it there rather than in lz_gui_session_apply is what makes a re-run of
   one turn differ from the first run, which is the whole point (settings
   do not change between a reply and its regeneration). */
static void seed_next_turn(LZGuiSession *s) {
    if (s->seed_mode == LZ_COMMON_SEED_FIXED) {
        s->opts.rng_seed = s->seed_fixed;
        return;
    }
    s->seed_turn++;
    /* seed_turn breaks ties WITHIN one coarse clock tick; lz_seed_mix
       spreads the clock's weak entropy across the whole word instead of
       leaving the low bits at zero on DOS/Win9x, where lz_time_ms
       quantises to the 54.9 ms tick. */
    s->opts.rng_seed = lz_seed_mix((lz_u64)lz_time_ms())
                     + s->seed_turn;
}

int lz_gui_session_begin(LZGuiSession *s, const char *user_utf8, int len,
                         char *errbuf, int errlen) {
    int rc;
    if (!s) { LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG); return rc; }
    seed_next_turn(s);
    return lz_session_begin((LZSession *)s, user_utf8, len, s->system,
                            errbuf, errlen);
}

int lz_gui_session_regen(LZGuiSession *s, char *errbuf, int errlen) {
    int rc;
    if (!s) { LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG); return rc; }
    seed_next_turn(s);
    return lz_session_regen((LZSession *)s, s->system, errbuf, errlen);
}

int lz_gui_session_job(void *ud, LZTokenSink sink, LZShouldContinue cont,
                       void *cb_ctx, char *errbuf, int errlen) {
    LZGuiSession *s = (LZGuiSession *)ud;
    int rc, i;
    if (!s) { LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG); return rc; }
    /* The model-ready gate lives HERE, not in the core: the core checks
       `s->model && s->tok && s->state`, but for the GUI those are the
       always-non-NULL embedded addresses of LZGuiModel's members, so the
       core's check would pass on an unloaded model. have_* is the real
       gate. */
    if (!lz_gui_model_ready(s->mdl)) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NOT_OPEN);
        return rc;
    }
    /* EOS + stop strings are the CALLER's sampling configuration - the
       core sets none. The GUI always wants EOS and its own STOPS, so
       this is applied here on every job. */
    lz_gen_opts_set_eos(&s->opts, s->tok);
    for (i = 0; i < N_STOPS && i < LZ_MAX_STOP; i++) s->opts.stop[i] = STOPS[i];
    s->opts.n_stop = (N_STOPS < LZ_MAX_STOP) ? N_STOPS : LZ_MAX_STOP;
    /* The worker's callbacks travel per call (the core bundles its own
       session_sink/session_cont), so nothing is stashed on the struct's
       ui_* fields here any more. s->system rides along as the job's
       custom system prompt and &s->ins as its optional inspector - the
       core passes both through to the engine, and gui/worker.c copies
       *ins out once per token for the inspector panel. */
    return lz_session_job(ud, s->system, sink, cont, cb_ctx,
                          errbuf, errlen, &s->ins);
}

int lz_gui_session_end(LZGuiSession *s, char *errbuf, int errlen) {
    return lz_session_end((LZSession *)s, errbuf, errlen);
}

void lz_gui_session_append_reply(LZGuiSession *s, const char *bytes,
                                 int len) {
    lz_session_append_reply((LZSession *)s, bytes, len);
}

int lz_gui_session_trim(LZGuiSession *s) {
    return lz_session_trim((LZSession *)s);
}

int lz_gui_session_trimmed(const LZGuiSession *s) {
    return lz_session_trimmed((const LZSession *)s);
}

const char *lz_gui_session_reply(const LZGuiSession *s, int *len) {
    return lz_session_reply((const LZSession *)s, len);
}

int lz_gui_session_turns(const LZGuiSession *s) {
    return lz_session_turns((const LZSession *)s);
}

const char *lz_gui_session_prompt(const LZGuiSession *s, int *len) {
    return lz_session_prompt((const LZSession *)s, len);
}

int lz_gui_session_token_count(LZGuiSession *s, char *errbuf, int errlen) {
    /* Same ready gate as the job, for the same reason: the core's pointer
       check cannot tell "model loaded" from "struct zeroed" here. */
    if (!s || !lz_gui_model_ready(s->mdl)) return -1;
    return lz_session_token_count((LZSession *)s, s->system, errbuf, errlen);
}
