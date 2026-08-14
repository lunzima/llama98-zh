/* Background generation thread. See worker.h for the two rules.
 *
 * _beginthreadex, not CreateThread: both C runtimes here keep per-thread
 * state, and a thread that enters the CRT without it is the kind of bug
 * that shows up as corruption somewhere else entirely. Watcom declares
 * it with the same signature under __NT__, so one spelling covers both
 * toolchains.
 */
#include <process.h>
#include <stdlib.h>
#include <string.h>

#include "worker.h"

static struct {
    HANDLE thread;
    HWND notify;
    LZWorkerJob job;
    void *ud;
    /* Where the running job writes its per-token candidate snapshot,
       or NULL for a job with none - see lz_worker_start's own comment.
       Read-only from this file's point of view: the job function is
       what fills *ins (through a completely different route - see
       gui/session.c's own lz_generate_resume_ex call), this file only
       ever copies it out. */
    const LZInspect *ins;
    /* GetTickCount() at the last WM_APP_INSPECT actually posted, for
       the 500ms send-side throttle. 0 at lz_worker_start (see there
       for why that makes the FIRST sink() call after a start always
       send, not wait out a full 500ms with nothing on screen).
       Wraparound-safe by construction: `now - last_inspect_ms` is
       unsigned subtraction, correct modulo 2^32 regardless of which
       side of a GetTickCount() wrap either value falls on, as long as
       the true elapsed time never reaches 2^31 ms (24.8 days) - not a
       concern for one generation's worth of running time. */
    DWORD last_inspect_ms;
    volatile LONG stop;
    volatile LONG busy;
    long sent, freed, dropped;
    /* How many tokens the running job has emitted, counted on
       the SEND side exactly like the counters below. This is a TOKEN
       count, not a byte count - the sink hands worker.c decoded UTF-8
       byte blocks (WM_APP_TOKENS carries their LENGTH for the
       transcript), and the engine emits exactly one sink() call per
       sampled token except when a stop string splits one token's bytes
       into two emits (lz_stopf_emit), which is a rare and acceptable
       rounding error for a speed display. */
    long tokens;
    char err[1024];
} w;

/* Read through a volatile so the compiler cannot hoist it out of the
 * job's loop. The flag is written with InterlockedExchange from the UI
 * thread; a plain store would be fine on x86, but the read is the half
 * that gets optimised away. */
static int stop_requested(void) {
    return w.stop != 0;
}

static int worker_cont(void *ctx) {
    (void)ctx;
    return stop_requested() ? 0 : 1;
}

/* Copy *ins onto the heap and post it, exactly the same ownership
   shape worker_sink's own WM_APP_TOKENS post already uses - a COPY,
   not the pointer itself, because *w.ins keeps changing (the job
   overwrites it every sink() call) and the GUI thread reads its own
   message queue on its own schedule, arbitrarily later. Shared by the
   throttled call in worker_sink below and the unconditional finishing
   call in thread_main - one heap-and-post routine, not two copies of
   it that could drift on the ownership handshake. */
static void post_inspect(const LZInspect *ins) {
    LZInspect *buf;
    if (!w.notify) return;
    buf = (LZInspect *)malloc(sizeof(LZInspect));
    if (!buf) { w.dropped++; return; }
    *buf = *ins;
    if (!PostMessage(w.notify, WM_APP_INSPECT, 0, (LPARAM)buf)) {
        free(buf);
        w.dropped++;
        return;
    }
    w.sent++;
}

static void worker_sink(const char *bytes, int len, void *ctx) {
    char *buf;
    (void)ctx;
    if (len <= 0 || !w.notify) return;
    w.tokens++;               /* one sink() per emitted token */
    buf = (char *)malloc((size_t)len);
    if (!buf) { w.dropped++; return; }
    memcpy(buf, bytes, (size_t)len);
    if (!PostMessage(w.notify, WM_APP_TOKENS, (WPARAM)len, (LPARAM)buf)) {
        /* The queue refused it. Free it here - nobody else will, because
           the message that would have carried the ownership is the one
           that did not arrive. */
        free(buf);
        w.dropped++;
        return;
    }
    w.sent++;

    /* Inspector snapshot, throttled on the SEND side - distinct from
       the text just posted above, which is never throttled: the
       transcript has to show every byte the model produced, but the
       candidate panel only needs to be current within ~500ms, and a
       message per token would be pure waste on the target for
       something the panel repaints at 2 Hz regardless.
       w.last_inspect_ms starts at 0 (lz_worker_start), so the FIRST
       sink() call after a start always passes this check - the panel
       shows something as soon as there is anything to show, rather
       than sitting empty for up to 500ms after generation begins. */
    if (w.ins) {
        DWORD now = GetTickCount();
        if (now - w.last_inspect_ms >= 500) {
            post_inspect(w.ins);
            w.last_inspect_ms = now;
        }
    }
}

static unsigned __stdcall thread_main(void *arg) {
    int rc;
    (void)arg;
    w.err[0] = '\0';
    rc = w.job(w.ud, worker_sink, worker_cont, NULL, w.err, (int)sizeof w.err);
    /* Unconditional finishing snapshot - the throttle above only
       guarantees "at least every 500ms while the job runs"; on its
       own that leaves the panel stopped mid-frame whenever the very
       last token happened to land inside that window, which is the
       common case, not a rare one (a short reply can finish in under
       500ms of wall time entirely). Normal completion, a user Stop,
       or an error all reach here the same way, and all three count as
       "generation ended" for the requirement that the panel stop on
       the last token once generation ends - this line is what
       implements that, not merely documents it. Posted BEFORE
       WM_APP_GEN_DONE, the same ordering guarantee WM_APP_TOKENS
       already relies on (PostMessage to one window preserves order),
       so the panel's own handler sees the final frame before
       finish_job() runs its own cleanup. */
    if (w.ins) post_inspect(w.ins);
    PostMessage(w.notify, WM_APP_GEN_DONE, (WPARAM)rc, 0);
    return 0;
}

int lz_worker_start(HWND notify, LZWorkerJob job, void *ud,
                    const LZInspect *ins) {
    unsigned tid;
    if (!notify || !job) return 1;
    if (w.busy) return 1;
    w.notify = notify;
    w.job = job;
    w.ud = ud;
    w.ins = ins;
    w.last_inspect_ms = 0;   /* see this field's own comment */
    w.stop = 0;
    w.sent = w.freed = w.dropped = 0;
    w.tokens = 0;            /* a fresh count per job */
    w.err[0] = '\0';
    w.busy = 1;
    w.thread = (HANDLE)_beginthreadex(NULL, 256 * 1024, thread_main,
                                      NULL, 0, &tid);
    if (!w.thread) { w.busy = 0; return 1; }
    return 0;
}

void lz_worker_request_stop(void) {
    InterlockedExchange((LONG *)&w.stop, 1);
}

void lz_worker_join(void) {
    HANDLE t = w.thread;
    lz_worker_request_stop();
    if (t) {
        WaitForSingleObject(t, INFINITE);
        CloseHandle(t);
        w.thread = NULL;
    }
    w.busy = 0;
}

int lz_worker_drain(HWND notify) {
    MSG msg;
    int drained_done = 0;
    if (!notify) return 0;
    /* Range-filtered PeekMessage, not GetMessage: this must return
       immediately when there is nothing of ours queued, and it must
       never remove a message that belongs to the rest of the UI.
       Upper bound is WM_APP_INSPECT, not WM_APP_GEN_DONE: the three
       IDs are contiguous (WM_APP+1..+3), and leaving
       WM_APP_INSPECT out of this range would mean a caller that
       synchronously drains before touching shared state - which is the
       entire reason this function exists - could still leave a stray
       WM_APP_INSPECT sitting in the queue, undispatched, for the
       ordinary message loop to deliver later against state the caller
       has since changed. The loop below still only ever treats
       WM_APP_GEN_DONE as the terminal message; see thread_main's own
       comment for why a WM_APP_INSPECT can never be posted after it. */
    while (PeekMessage(&msg, notify, WM_APP_TOKENS, WM_APP_INSPECT,
                       PM_REMOVE)) {
        DispatchMessage(&msg);
        if (msg.message == WM_APP_GEN_DONE) { drained_done = 1; break; }
    }
    return drained_done;
}

void lz_worker_join_drain(HWND notify) {
    lz_worker_join();
    lz_worker_drain(notify);
}

int lz_worker_busy(void) { return w.busy != 0; }

const char *lz_worker_error(void) { return w.err; }

void lz_worker_free_tokens(void *p) {
    if (!p) return;
    w.freed++;
    free(p);
}

long lz_worker_posts_sent(void)    { return w.sent; }
long lz_worker_posts_freed(void)   { return w.freed; }
long lz_worker_posts_dropped(void) { return w.dropped; }

/* How many tokens the last job's sink saw. A live count is not
   safe to read while a job is running (the worker thread is writing it
   concurrently), so this is documented as "after the job ends" - which
   is exactly when the UI reads it. */
long lz_worker_tokens_sent(void) { return w.tokens; }
