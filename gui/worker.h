#ifndef LZ_GUI_WORKER_H
#define LZ_GUI_WORKER_H

#include <windows.h>

#include "compat40.h"   /* WM_APP is 4.0-era; see there */
#include "llama_zh.h"   /* LZTokenSink, LZShouldContinue, LZInspect */

/* Background generation, and the rules that keep it from writing into
 * freed memory.
 *
 * ONE SEQUENCE, always: request stop, join, DRAIN, then touch anything
 * the worker can see. Every operation that resets
 * state, unloads a model, or exits goes through lz_worker_join_drain()
 * first. There is no "check if it is running" variant, because the
 * answer can change between the check and the next line.
 *
 * The drain step exists because lz_worker_join() only stops the THREAD.
 * thread_main posts WM_APP_GEN_DONE (and, before it, one WM_APP_TOKENS
 * per chunk already produced) unconditionally, even when it was told to
 * stop - so the moment lz_worker_join() returns, those messages are
 * already sitting in the queue, undispatched. A caller that touches
 * shared state (history, the transcript) right after lz_worker_join()
 * and before the queue is pumped is racing its own worker's leftover
 * mail: WM_APP_GEN_DONE eventually arrives anyway, on top of whatever
 * the caller just did, and finish_job() reads state that has since
 * changed under it. lz_worker_join_drain() closes that gap by pumping
 * the worker's own messages synchronously, in place, before returning -
 * see lz_worker_drain() below for exactly what "pumping" means here.
 *
 * OWNERSHIP: the worker allocates each token buffer, and each inspect
 * snapshot, and the UI frees it with lz_worker_free_tokens(), exactly
 * once, when WM_APP_TOKENS or WM_APP_INSPECT is handled - drained or
 * not, it is the same handler either way (see lz_worker_drain()). The
 * counters below exist so a test can assert the two numbers match
 * rather than trusting the code to be careful.
 */

#define WM_APP_TOKENS   (WM_APP + 1)   /* wParam = bytes, lParam = buffer */
#define WM_APP_GEN_DONE (WM_APP + 2)   /* wParam = LZ_ERR_* code */
/* wParam unused (0); lParam = a heap-allocated LZInspect* - one
 * snapshot, not a stream of bytes like WM_APP_TOKENS, so there is
 * nothing for wParam to carry a length of. Freed the same way as a
 * WM_APP_TOKENS payload - see lz_worker_free_tokens's own comment,
 * which is not token-specific. */
#define WM_APP_INSPECT  (WM_APP + 3)
/* Prefill advanced. No payload and nothing to free - it only asks the UI
 * thread to redraw the indicator from the counters the engine's callback
 * has already written. It exists because the refresh tick is 400 ms and
 * a prefill can finish inside one of them: waiting for the tick means
 * the indicator is never drawn at all on a fast machine, which is the
 * opposite of what it is for. */
#define WM_APP_PREFILL  (WM_APP + 4)

/* The job the worker runs.
 *
 * Shaped like the callback half of lz_generate on purpose: the
 * production wrapper is a one-line forward, and a scripted producer is
 * interchangeable with it, so the threading contract can be tested with
 * no model in the room. `sink` and `cont` share `cb_ctx`, as they do in
 * the engine.
 *
 * MUST check cont() before emitting, not after: the design budgets a
 * stop latency of about one token, and a job that checks
 * afterwards emits one more than it was asked to. */
typedef int (*LZWorkerJob)(void *ud, LZTokenSink sink,
                           LZShouldContinue cont, void *cb_ctx,
                           char *errbuf, int errlen);

/* Start. Returns 0 on success, non-zero if a job is already running or
 * the thread could not be created. Refusing rather than queueing is
 * deliberate - two generations sharing one LZRunState is exactly what
 * llama_zh.h says was never supported.
 *
 * ins: where the job writes its per-token candidate snapshots, or
 * NULL for a job that has none (lz_gui_model_load_job - every OTHER
 * LZWorkerJob so far). Not owned
 * by this file - the caller's job function is what actually fills it
 * (lz_gui_session_job passes &LZGuiSession.ins straight through to
 * lz_generate_resume_ex) - worker.c only ever READS *ins, once per
 * sink() call, to decide whether 500ms have passed since the last
 * WM_APP_INSPECT and, if so, post a COPY of it. That split (the job
 * writes, worker.c reads-and-throttles-and-copies) is what keeps this
 * file generic: it does not need to know what an LZGuiSession is, only
 * that ins, when given one, points at something an LZInspect can be
 * copied out of after every sink() call. */
int lz_worker_start(HWND notify, LZWorkerJob job, void *ud,
                    const LZInspect *ins);

/* Ask the running job to stop at its next check. Safe to call when
 * nothing is running. */
void lz_worker_request_stop(void);

/* Request stop and wait for the thread to end. Idempotent. After it
 * returns, no further WM_APP_TOKENS will be POSTED - but ones already
 * posted, and the WM_APP_GEN_DONE that follows them, are still in the
 * queue and still need dispatching. On its own this function is not the
 * sequence a UI-side caller wants; see lz_worker_join_drain() below,
 * which is. Kept exposed because finish_job() - the WM_APP_GEN_DONE
 * handler itself - needs the plain join, not a recursive drain. */
void lz_worker_join(void);

/* Pump exactly the worker's own messages (WM_APP_TOKENS, WM_APP_INSPECT,
 * WM_APP_GEN_DONE) out of `notify`'s queue, synchronously, until a
 * WM_APP_GEN_DONE has been dispatched or the queue holds none of them.
 *
 * Filtered to that message range and that window only, so it is not a
 * general message pump: anything else already queued (paint, a timer,
 * the next WM_COMMAND) is left exactly where it was. Each removed
 * message is handed to DispatchMessage, which is what makes this safe
 * to add without duplicating logic - WM_APP_TOKENS already frees its
 * buffer in the window procedure, and WM_APP_GEN_DONE already runs
 * finish_job() there, so draining through the real wndproc gets both
 * for free instead of re-implementing them here.
 *
 * "Until GEN_DONE" is a COMPLETE drain, not a hopeful one, because
 * thread_main posts DONE last and PostMessage-to-one-window preserves
 * order (see the comment in worker.c) - once this function has seen
 * GEN_DONE go through, the job that owned it cannot have any token
 * message still waiting behind it.
 *
 * Safe to call with nothing pending - worker never started, or a
 * previous drain already emptied the queue: PeekMessage returns FALSE
 * immediately and the loop body never runs.
 *
 * Assumes the worker thread has already exited (call lz_worker_join(),
 * or use lz_worker_join_drain() below, first) - draining while the
 * thread might still be posting would risk stopping short of DONE.
 *
 * Returns 1 if a GEN_DONE was drained (there was a job's tail to reap),
 * 0 if the queue held none of the worker's messages. */
int lz_worker_drain(HWND notify);

/* lz_worker_join() followed by lz_worker_drain(notify): the sequence
 * every UI-side caller that is about to touch g.sess, the transcript, or
 * anything else the worker can see must use INSTEAD of lz_worker_join()
 * alone - one call, so there is no second step left to forget. */
void lz_worker_join_drain(HWND notify);

int lz_worker_busy(void);

/* The job's errbuf, valid until the next start. */
const char *lz_worker_error(void);

/* Release a WM_APP_TOKENS or WM_APP_INSPECT payload - the only correct
 * way to free either. Not token-specific despite the name:
 * WM_APP_INSPECT reuses this same free path and the same sent/freed/
 * dropped counters below, rather than a second set scoped to itself -
 * the invariant those counters exist to prove, "everything posted
 * eventually gets freed exactly once", holds the same way regardless
 * of which message carried the buffer). Kept under this name rather
 * than renamed: every existing call site already spells it this way,
 * and a rename here would be a second, larger diff for no behaviour
 * change. */
void lz_worker_free_tokens(void *p);

/* Ownership counters, shared across WM_APP_TOKENS and WM_APP_INSPECT -
 * both are "a heap buffer this file posted", and the invariant is the
 * same for either. sent and freed must be equal once the queue has
 * been drained; dropped must be zero. */
long lz_worker_posts_sent(void);
long lz_worker_posts_freed(void);
long lz_worker_posts_dropped(void);

/* How many tokens the last job emitted, counted at the sink
 * (one sink() call per sampled token; a stop string splitting one
 * token's bytes into two emits is a rare rounding error). Live only
 * after the job ends - the worker writes it concurrently while running,
 * so this is for lz_worker_join/_drain callers, which is everyone the
 * UI side ever is. */
long lz_worker_tokens_sent(void);

#endif
