/* Model loading, as a worker job. See model.h for why it is one.
 *
 * Everything here runs on the worker thread. It touches no window and no
 * control: the result travels back as WM_APP_GEN_DONE's return code plus
 * the errbuf the worker already carries, which is the same path a failed
 * generation uses. A second notification mechanism for loading would be
 * a second thing to get the teardown order wrong in.
 */
#include <stdio.h>
#include <string.h>

#include "err.h"
#include "lfn.h"
#include "modelload.h"

int lz_gui_model_dir_ok(const char *dir) {
    if (!dir || !dir[0]) return 0;
    /* The same resolver lz_open's own probe uses, so the pre-check and
       the loader agree by construction. Two identical sprintf calls do
       not: they still disagree the moment the volume cannot store the
       long name, since "identical" was only ever about the format
       string. A pre-check that passes where the load fails is the one
       failure this function exists to prevent. */
    return lz_lfn_exists(dir, "model.bin");
}

int lz_gui_state_alloc_fail = 0;
int lz_gui_state_alloc_last = -1;

/* The ONE place a run state is allocated in this front end - both the
   load job and the resize go through it, so the injection cannot be
   bypassed and neither can the record of what the allocator was asked
   for. See modelload.h for why both exist. */
static int state_alloc(LZRunState *st, const LZModel *model, int seq_len,
                       char *errbuf, int errlen) {
    lz_gui_state_alloc_last = seq_len;
    if (lz_gui_state_alloc_fail > 0) {
        lz_gui_state_alloc_fail--;
        /* The same code the real thing reports, so the injected path
           and the real path put identical text in front of the user -
           a gate that reads the message is then reading the message
           the user would see. */
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_STATE_ALLOC);
        return LZ_ERR_STATE_ALLOC;
    }
    /* spec_k_max 0: this front end wires no speculative decoding, and
       lz_state_alloc's own comment says a caller that does not should
       pass 0 rather than LZ_SPEC_K_MAX. */
    return lz_state_alloc(st, model, seq_len, 0, errbuf, errlen);
}

void lz_gui_model_unload(LZGuiModel *m) {
    if (!m) return;
    /* Reverse of the build order, and each guarded by its own flag: the
       failure path calls this with any prefix of the four steps done. */
    if (m->have_state) { lz_state_free(&m->state); m->have_state = 0; }
    if (m->have_tok)   { lz_tokenizer_free(&m->tok); m->have_tok = 0; }
    if (m->have_model) { lz_free(&m->model); m->have_model = 0; }
    m->seq_len = 0;
}

int lz_gui_model_seq_cap(const LZGuiModel *m) {
    if (!m || !m->have_model) return 0;
    return m->model.config.seq_len;
}

int lz_gui_model_resize(LZGuiModel *m, int want, char *errbuf, int errlen) {
    LZRunState ns;
    int rc, seq;

    if (!m) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        return rc;
    }
    if (!m->have_model || !m->have_state) {
        /* Nothing is allocated, so there is nothing to re-allocate and
           no failure to report: the next load will pick the size up
           from seq_want. Recording it here keeps that true even if the
           caller only ever calls this one. */
        m->seq_want = want;
        return 0;
    }
    seq = lz_common_ctx_clamp(want, lz_gui_model_seq_cap(m));
    /* Already that size AFTER the clamp, not before it: two different
       requests can clamp to the same allocation, and re-allocating an
       identical state would throw the KV cache away for nothing. */
    if (m->seq_len == seq) {
        m->seq_want = want;
        return 0;
    }
    rc = state_alloc(&ns, &m->model, seq, errbuf, errlen);
    if (rc != 0) return rc;      /* m->state untouched - see modelload.h */
    lz_state_free(&m->state);
    /* Struct assignment, not a memcpy of the pointers one by one: an
       LZRunState is pointers and scalars, nothing in it points back
       into itself, and nothing outside holds a pointer INTO it (the
       ins field is set per call by lz_generate_resume_ex, and iso
       points at the model, which did not change). */
    m->state = ns;
    m->seq_len = seq;
    m->seq_want = want;
    return 0;
}

int lz_gui_model_ready(const LZGuiModel *m) {
    return m && m->have_model && m->have_tok && m->have_state;
}

const char *lz_gui_model_name(const LZGuiModel *m) {
    const char *p, *last;
    if (!m || !m->dir[0]) return "";
    last = m->dir;
    for (p = m->dir; *p; p++) {
        if (*p == '/' || *p == '\\') last = p + 1;
    }
    return *last ? last : m->dir;
}

int lz_gui_model_load_job(void *ud, LZTokenSink sink, LZShouldContinue cont,
                          void *cb_ctx, char *errbuf, int errlen) {
    LZGuiModel *m = (LZGuiModel *)ud;
    char tokpath[600];
    int rc;

    (void)sink; (void)cont; (void)cb_ctx;
    if (!m) return LZ_ERR_NULL_ARG;

    /* Whatever was open is gone before anything new is opened. Loading
       on top of a live model would leak it and leave the window pointing
       at whichever half succeeded. */
    lz_gui_model_unload(m);

    rc = lz_open(&m->model, m->dir, errbuf, errlen);
    if (rc != 0) return rc;
    m->have_model = 1;

    rc = lz_read_weights(&m->model, errbuf, errlen);
    if (rc != 0) { lz_gui_model_unload(m); return rc; }

    /* Resolved, not appended - see src/lfn.h. */
    rc = lz_lfn_path(m->dir, "tokenizer.json", tokpath, (int)sizeof tokpath,
                     errbuf, errlen);
    if (rc != 0) { lz_gui_model_unload(m); return rc; }
    rc = lz_tokenizer_load(&m->tok, tokpath, errbuf, errlen);
    if (rc != 0) { lz_gui_model_unload(m); return rc; }
    m->have_tok = 1;

    /* Clamped against the model that just came up, which is the only
       moment its cfg->seq_len is knowable - the settings dialog can be
       opened with nothing loaded, so the clamp cannot live there. */
    {
        int seq = lz_common_ctx_clamp(m->seq_want, lz_gui_model_seq_cap(m));
        rc = state_alloc(&m->state, &m->model, seq, errbuf, errlen);
        if (rc != 0) { lz_gui_model_unload(m); return rc; }
        m->have_state = 1;
        m->seq_len = seq;
    }

    return 0;
}
