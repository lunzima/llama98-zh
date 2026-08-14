#ifndef LZ_GUI_MODELLOAD_H
#define LZ_GUI_MODELLOAD_H

#include "forward.h"
#include "llama_zh.h"
#include "model.h"
#include "settings.h"
#include "tokenizer.h"

/* NOT named model.h. The engine already has one, both directories are on
 * the include path, and which `#include "model.h"` resolves to would
 * then depend on -I order - silently, and differently for the two
 * toolchains.
 *
 * Loading a model, as a worker job.
 *
 * It runs on the worker thread for one reason: on a Pentium II this
 * takes seconds, and a UI thread inside lz_read_weights is a window
 * that does not repaint.
 *
 * The load is four calls that each allocate, so the interesting part is
 * the FAILURE path: a tokenizer that fails to load after the weights
 * succeeded must not leave the weights behind. lz_gui_model_unload is
 * the single teardown, it is idempotent, and the job calls it on every
 * failure - one place to get right instead of three.
 */

/* THE SIZE IS ONE SIZE FOR THE WHOLE SESSION.
 *
 * The rule is "one fixed size, not cli_main.c's single-turn figure":
 * with a PER-TURN size, turn two's full re-render exceeds what turn
 * one allocated, the turn-dropping fallback throws away turn one, and
 * the window reads as a model that forgot the first message. It is
 * enforced by seq_want being set once from the settings and only ever
 * changing when the USER changes it, which never happens mid-turn
 * (the settings dialog is modal and lz_gui_model_resize joins the
 * worker first).
 *
 * Do not read "configurable" as "per-turn". The rule is the rule;
 * only the number is the user's now. 2048 lives in gui/settings.h as
 * LZ_COMMON_CTX_DEFAULT.
 */

typedef struct {
    LZModel     model;
    LZTokenizer tok;
    LZRunState  state;
    int         have_model, have_tok, have_state;
    char        dir[512];        /* filled by the caller before the job */
    /* Context window, in and out.
     *
     * seq_want is an INPUT: the caller sets it before the load job the
     * same way it sets dir[]. seq_len is an OUTPUT: what the run state
     * was actually allocated with, after lz_common_ctx_clamp had its say,
     * and 0 whenever have_state is 0.
     *
     * Two fields rather than one because the clamp makes them
     * legitimately different, and a single field would leave every
     * reader guessing which of the two it was holding. The status bar's
     * denominator wants seq_len; the settings dialog wants
     * LZGuiSettings.ctx; nobody wants a field that is sometimes one and
     * sometimes the other. */
    int         seq_want;
    int         seq_len;
} LZGuiModel;

/* Does this directory hold a model this build can open?
 *
 * Only model.bin counts. src/model.c probes for exactly that name and
 * otherwise falls back to f32 safetensors, which on a 128 MB machine
 * is an out-of-memory failure several seconds in rather than a refusal
 * up front. Returns 1 when it is there. */
int lz_gui_model_dir_ok(const char *dir);

/* LZWorkerJob. `ud` is an LZGuiModel whose dir[] is set. sink and cont
 * are unused - loading emits nothing and is not interruptible; the shape
 * is the worker's, so this can be started the same way a generation is. */
int lz_gui_model_load_job(void *ud, LZTokenSink sink, LZShouldContinue cont,
                          void *cb_ctx, char *errbuf, int errlen);

/* Idempotent. Safe on a zeroed struct and on a half-built one. */
void lz_gui_model_unload(LZGuiModel *m);

int lz_gui_model_ready(const LZGuiModel *m);

/* Last path component of dir[], for the status line. Never NULL. */
const char *lz_gui_model_name(const LZGuiModel *m);

/* The loaded model's own context ceiling (cfg->seq_len), or 0 when
 * nothing is loaded - which is exactly the "no cap" value
 * lz_common_ctx_clamp expects, so the two compose without a special
 * case at the call site. */
int lz_gui_model_seq_cap(const LZGuiModel *m);

/* Re-allocate the run state at a new context size.
 *
 * ALLOCATES THE NEW STATE BEFORE FREEING THE OLD ONE, deliberately, and
 * this is the whole design of the function. The alternative - free,
 * then allocate, then try to re-allocate the old size if that failed -
 * has a lower peak (max instead of sum) and one fatal property: the
 * recovery allocation can itself fail, and then the program has no run
 * state at all and a model that looks loaded. Growing 2048 -> 4096 here
 * costs roughly old + new at the moment of the swap, which on a Win98
 * box is precisely when this fails; that failure is the SUPPORTED
 * outcome (the caller shows an error box and rolls the setting back),
 * whereas losing the working state is not an outcome anyone designed.
 *
 * On failure *m is byte-for-byte untouched - state, seq_len and all -
 * and errbuf holds the engine's own message. On success the old state
 * is freed, seq_len records what was allocated, and any prefix
 * checkpoint the caller holds is stale: the new state carries a new
 * epoch, so lz_ckpt_restore would refuse it, but the caller must still
 * lz_gui_session_prefix_clear rather than rely on that refusal.
 *
 * `want` goes through lz_common_ctx_clamp against this model's own cap. */
int lz_gui_model_resize(LZGuiModel *m, int want, char *errbuf, int errlen);

/* ---- allocation-failure injection, and the record of what was asked ----
 *
 * lz_gui_state_alloc_fail: when non-zero it is DECREMENTED and the next
 * run-state allocation fails with LZ_ERR_STATE_ALLOC without calling
 * lz_state_alloc at all. It exists because the failure path above is
 * the only way a settings dialog can break the program, and asking a
 * developer machine for a genuinely impossible allocation does not
 * work - 2 GB succeeds here, so a gate written that way is green
 * forever and proves nothing.
 *
 * Because it is a counter and not a flag, a gate can assert it came
 * back to 0 and thereby prove the injection point was REACHED. A flag
 * that is merely set and never observed to be consumed cannot tell
 * "the failure path ran" from "the failure path was never compiled in".
 *
 * lz_gui_state_alloc_last: the seq_len handed to the allocator on the
 * most recent attempt, injected or not. Written at the single point
 * both callers funnel through, so it cannot drift from the value that
 * really goes to lz_state_alloc - which is the thing worth asserting,
 * since the clamp is upstream of it. -1 when nothing has been tried. */
extern int lz_gui_state_alloc_fail;
extern int lz_gui_state_alloc_last;

#endif
