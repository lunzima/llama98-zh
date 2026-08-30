#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chat.h"
#include "err.h"

#define OPEN_TAG  "<think>"
#define CLOSE_TAG "</think>"
#define OPEN_LEN  7
#define CLOSE_LEN 8

/* The canonical identity prompt, and the SINGLE place it is written.
 *
 * Trained as a recognizable CONSTANT (3 of
 * every 6 ChatML samples), not as prose. Paraphrasing it anywhere ships an
 * UNTRAINED prompt - the exact failure this project measured (a 57.6M
 * model left with no identity agrees it is Qwen, DeepSeek, Claude and
 * Doubao in turn).
 *
 * UTF-8 with the Chinese hex-escaped, so the source file stays ASCII.
 * Byte-for-byte: render injects it raw, and the jinja template's
 * `{%- else %}` branch carries the same bytes.
 *
 * Exported through lz_chat_default_system (chat.h) so a front end can
 * SHOW the default without owning a second copy. Two copies of this text
 * is two authorities, and they would drift; the getter is the way out.
 * "Restore defaults" on the GUI is exactly this - it needs the text to
 * display, and it must not re-type it. */
static const char LZ_SYS_DEFAULT[] =
    "\xe4\xbd\xa0\xe6\x98\xaf\xe6\x98\x86\xe6\x98\x86"
    "98\xef\xbc\x8c\xe7\x94\xb1 Lunzima "
    "\xe5\x88\x9b\xe9\x80\xa0\xe3\x80\x82"
    "\xe4\xbd\xa0\xe6\x98\xaf\xe4\xb8\x80\xe4\xb8\xaa"
    "\xe4\xb9\x90\xe4\xba\x8e\xe5\x8a\xa9\xe4\xba\xba"
    "\xe7\x9a\x84\xe5\x8a\xa9\xe6\x89\x8b\xe3\x80\x82";

const char *lz_chat_default_system(void) {
    return LZ_SYS_DEFAULT;
}

void lz_chat_buf_init(LZChatBuf *b) {
    if (!b) return;
    b->s = NULL;
    b->len = 0;
    b->cap = 0;
}

void lz_chat_buf_free(LZChatBuf *b) {
    if (!b) return;
    free(b->s);
    b->s = NULL;
    b->len = 0;
    b->cap = 0;
}

/* Append n bytes. Returns 0 on success, non-zero on allocation failure.
   One extra byte is reserved for NUL so s can be passed directly as a C
   string to lz_generate. */
static int put(LZChatBuf *b, const char *s, int n) {
    if (n <= 0) return 0;
    if (b->len + n + 1 > b->cap) {
        int cap = b->cap ? b->cap : 256;
        char *p;
        while (cap < b->len + n + 1) {
            if (cap > (1 << 28)) return 1;     /* overflow guard, also caps runaway growth */
            cap *= 2;
        }
        p = (char *)realloc(b->s, (size_t)cap);
        if (!p) return 1;
        b->s = p;
        b->cap = cap;
    }
    memcpy(b->s + b->len, s, (size_t)n);
    b->len += n;
    b->s[b->len] = '\0';
    return 0;
}

static int puts_(LZChatBuf *b, const char *s) {
    return put(b, s, (int)strlen(s));
}

/* puts_ for a string LITERAL, whose length the compiler already knows.
 *
 * gcc folds strlen of a literal and Watcom does not: it emits the
 * classic inline sequence, `not ecx` then `repne scasb`, and counts the
 * bytes again on every call. Seventeen of those in this translation
 * unit, for twelve constant strings - plus the register save and
 * restore around each, which is most of why Watcom emits 85 pushes here
 * against gcc's 14.
 *
 * The empty-literal concatenation is not decoration. `sizeof(x) - 1` on
 * a `const char *` yields three or seven and the caller silently gets a
 * truncated string; writing it as "" lit "" means only a literal
 * compiles at all, so the misuse is a build error rather than a wrong
 * answer. An array like LZ_SYS_DEFAULT is not a literal and does not go
 * through here - its one call site spells its sizeof out. */
#define PUTLIT(b, lit) put((b), (lit), (int)(sizeof("" lit "") - 1))

static int is_space(char c) {
    /* matches Python str.strip()'s default set - jinja's |trim uses it */
    return c == ' ' || c == '\t' || c == '\n' || c == '\r'
        || c == '\v' || c == '\f';
}

/* jinja's |trim: strip whitespace at both ends */
static void trim(const char *s, int len, const char **o, int *olen) {
    int a = 0, b = len;
    while (a < b && is_space(s[a])) a++;
    while (b > a && is_space(s[b - 1])) b--;
    *o = s + a;
    *olen = b - a;
}

/* Find substring in [s, s+len); return offset, or -1 if not found.
   strstr won't do: content is length-delimited, not NUL-terminated. */
static int find(const char *s, int len, const char *pat, int plen, int from) {
    int i;
    if (plen <= 0 || len < plen) return -1;
    for (i = from < 0 ? 0 : from; i + plen <= len; i++) {
        if (memcmp(s + i, pat, (size_t)plen) == 0) return i;
    }
    return -1;
}

static int rfind(const char *s, int len, const char *pat, int plen) {
    int i;
    if (plen <= 0 || len < plen) return -1;
    for (i = len - plen; i >= 0; i--) {
        if (memcmp(s + i, pat, (size_t)plen) == 0) return i;
    }
    return -1;
}

/* Split into reasoning and content segments, replicating these two lines
   of the template:

     reasoning = content.split('</think>')[0].rstrip('
')
                        .split('<think>')[-1].lstrip('
')
     content   = content.split('</think>')[-1].lstrip('
')

   The two indices are NOT the same: reasoning takes everything before the
   FIRST </think>, content takes everything after the LAST one. With nested
   think blocks the difference shows; following intuition and using one
   index for both is wrong.
   reasoning then goes through |trim once more; content only
   lstrip('
'). */
static void split_think(const char *s, int len,
                        const char **r, int *rlen,
                        const char **c, int *clen) {
    int fc = find(s, len, CLOSE_TAG, CLOSE_LEN, 0);
    int lc, lo, a, b;
    if (fc < 0) {                       /* no </think>: everything is content */
        *r = s; *rlen = 0;
        *c = s; *clen = len;
        return;
    }
    /* reasoning segment */
    a = 0; b = fc;
    while (b > a && s[b - 1] == '\n') b--;          /* rstrip('\n') */
    lo = rfind(s + a, b - a, OPEN_TAG, OPEN_LEN);   /* split('<think>')[-1] */
    if (lo >= 0) a += lo + OPEN_LEN;
    while (a < b && s[a] == '\n') a++;              /* lstrip('\n') */
    trim(s + a, b - a, r, rlen);                    /* one more pass through |trim */

    /* content segment */
    lc = rfind(s, len, CLOSE_TAG, CLOSE_LEN);
    a = lc + CLOSE_LEN; b = len;
    while (a < b && s[a] == '\n') a++;              /* lstrip('\n') */
    *c = s + a; *clen = b - a;
}

static int msg_len(const LZChatMsg *m) {
    if (!m->content) return 0;
    return m->len < 0 ? (int)strlen(m->content) : m->len;
}

const char *lz_chat_gen_prompt_tail(int enable_thinking) {
    /* enable_thinking only takes effect here. On: leave an empty
       <think>\n for the model to fill in; off: complete the empty block
       so the model starts with content.

       Exported rather than inlined at the one use site because a caller
       doing cross-turn prefix reuse needs the SAME bytes to find where
       the reusable part of a render ends - this segment is exactly what
       turn N has and turn N+1's render does not. A second copy in the
       frontend would drift from the template silently: nothing errors,
       the split just stops being token-exact and prefix reuse quietly
       turns itself off (or worse, if the caller skipped the check). */
    return enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
}

int lz_chat_gen_prompt_starts_in_think(int enable_thinking) {
    /* Same reasoning as lz_chat_gen_prompt_tail's own comment, one
       level up: a GUI that re-decided "does the prompt end inside a
       <think> block" from enable_thinking directly would be a SECOND
       copy of that decision, free to drift the moment either literal
       string above changes shape. This derives the answer by counting
       the actual tags in the actual bytes lz_chat_gen_prompt_tail
       returns, so there is exactly one place that decision is made.

       gui/main.c seeds gui/stream.c's parser state with this at the
       start of every reply - without it, a thinking-enabled turn's
       prompt already ends "<think>\n" (the model resumes INSIDE the
       block and only ever emits the closing tag), and a parser that
       starts assuming "not in think" never finds an opening tag to
       match against, so the entire reasoning span renders as ordinary
       black text. */
    const char *tail = lz_chat_gen_prompt_tail(enable_thinking);
    int in_think = 0;
    int i = 0, n = (int)strlen(tail);
    while (i < n) {
        if (i + OPEN_LEN <= n && memcmp(tail + i, OPEN_TAG, OPEN_LEN) == 0) {
            in_think = 1;
            i += OPEN_LEN;
        } else if (i + CLOSE_LEN <= n &&
                  memcmp(tail + i, CLOSE_TAG, CLOSE_LEN) == 0) {
            in_think = 0;
            i += CLOSE_LEN;
        } else {
            i++;
        }
    }
    return in_think;
}

int lz_chat_render(const LZChatMsg *msgs, int n_msgs,
                   int add_generation_prompt, int enable_thinking,
                   LZChatBuf *out, char *errbuf, int errlen) {
    int i, last_query = -1;
    const char *ct;
    int ctlen;
    /* Failure exits return the LZErr code, not a bare 1, so an HTTP front
       end can map them: NO_MESSAGES / NO_USER / SYSTEM_FIRST / ROLE are
       all malformed request bodies (400), RENDER_ALLOC is ours (500).
       Initialised to INTERNAL so a path that forgets to set it says so
       instead of inheriting code 1 ("cannot open file"). */
    int rc = LZ_ERR_INTERNAL;

    if (!out) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_OUT_BUF);
        return rc;
    }
    lz_chat_buf_init(out);
    if (!msgs || n_msgs < 1) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NO_MESSAGES);
        return rc;
    }

    /* Index of the last user message. The template scans backwards and
       stops at the first user it meets - i.e. the last user message.
       It decides which assistant messages count as history: only those
       with index GREATER than it carry think blocks. Note this is not
       "the last message" - several assistant turns stacked at the end
       all carry think blocks. */
    for (i = n_msgs - 1; i >= 0; i--) {
        if (msgs[i].role == LZ_ROLE_USER) { last_query = i; break; }
    }
    if (last_query < 0) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NO_USER);
        return rc;
    }

    /* When the caller supplies no system message, inject the canonical
       identity prompt. The jinja template does the same in its
       `{%- else %}` branch and this literal must stay byte-identical to
       the one there.

       Why at all: upstream Qwen3.5 leaves the default empty and this
       project needs the Qwen2.5 behaviour. A generic transformers or
       OpenAI-style client that sends only a user turn would otherwise
       reach a 57.6M model with nothing telling it who it is - this
       project measured what that produces (agreeing it was Qwen,
       DeepSeek, Claude and Doubao in turn).

       THE STRING ITSELF IS NOT HERE. It lives in LZ_SYS_DEFAULT above,
       whose own comment says why - a render path and a front end both
       needing the text must not each carry a copy. */
    if (msgs[0].role != LZ_ROLE_SYSTEM) {
        if (PUTLIT(out, "<|im_start|>system\n")
            || put(out, LZ_SYS_DEFAULT, (int)(sizeof LZ_SYS_DEFAULT - 1))
            || PUTLIT(out, "<|im_end|>\n")) goto oom;
    }

    for (i = 0; i < n_msgs; i++) {
        const LZChatMsg *m = &msgs[i];
        trim(m->content ? m->content : "", msg_len(m), &ct, &ctlen);

        if (m->role == LZ_ROLE_SYSTEM) {
            if (i != 0) {
                LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_SYSTEM_FIRST);
                goto fail;
            }
            if (PUTLIT(out, "<|im_start|>system\n")
                || put(out, ct, ctlen)
                || PUTLIT(out, "<|im_end|>\n")) goto oom;

        } else if (m->role == LZ_ROLE_USER) {
            if (PUTLIT(out, "<|im_start|>user\n")
                || put(out, ct, ctlen)
                || PUTLIT(out, "<|im_end|>\n")) goto oom;

        } else if (m->role == LZ_ROLE_ASSISTANT) {
            const char *r, *c;
            int rlen, clen;
            split_think(ct, ctlen, &r, &rlen, &c, &clen);
            if (PUTLIT(out, "<|im_start|>assistant\n")) goto oom;
            if (i > last_query) {
                /* the final segment (after the last user) keeps reasoning */
                if (PUTLIT(out, "<think>\n")
                    || put(out, r, rlen)
                    || PUTLIT(out, "\n</think>\n\n")) goto oom;
            }
            /* History turns: reasoning is dropped ENTIRELY, only
               content survives. This is the template's behavior, not a
               simplification - the training side must assemble it the
               same way. */
            if (put(out, c, clen)
                || PUTLIT(out, "<|im_end|>\n")) goto oom;

        } else {
            LZ_ERR_SET1(rc, errbuf, errlen, LZ_ERR_ROLE, (int)m->role);
            goto fail;
        }
    }

    if (add_generation_prompt) {
        if (PUTLIT(out, "<|im_start|>assistant\n")) goto oom;
        if (puts_(out, lz_chat_gen_prompt_tail(enable_thinking))) goto oom;
    }
    return 0;

oom:
    LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_RENDER_ALLOC);
fail:
    lz_chat_buf_free(out);
    return rc;
}

int lz_chat_norm_history(const char *raw, int raw_len,
                         char *out, int outcap,
                         char *errbuf, int errlen) {
    const char *r, *c;
    int rlen, clen;

    if (!raw || !out || outcap <= 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return -1;
    }
    if (raw_len < 0) raw_len = (int)strlen(raw);
    /* History turns keep ONLY the content segment: reasoning is
       dropped entirely (template behavior, see lz_chat_render). */
    split_think(raw, raw_len, &r, &rlen, &c, &clen);
    if (clen >= outcap) {
        if (errbuf)
            lz_err_fmt(errbuf, errlen, LZ_ERR_TRUNC, clen + 1);
        return -1;
    }
    /* '\r' is dropped, not turned into '\n' - same choice gui/chatfile.c's
       decode makes for a CR that turns up mid-line, and for the same
       reason: this byte is noise, not a line break the source intended,
       so inventing one would change how many lines the message has. It
       reaches here through the byte-fallback path a small model's
       tokenizer can take even though the training corpus essentially
       never contains this byte (gui/savechat.h has the fuller account,
       on lz_gui_lf - this is the entry into LZChatMsg.content that
       function's fix does not cover, since it runs on typed input, not
       generated output). The filtered copy can only be shorter than
       `clen`, which the capacity check above already bounds, so no
       second bounds check is needed here. */
    {
        int i, w = 0;
        for (i = 0; i < clen; i++) {
            if (c[i] == '\r') continue;
            out[w++] = c[i];
        }
        out[w] = 0;
        return w;
    }
}

void lz_chat_hist_init(LZChatHist *h) {
    if (!h) return;
    memset(h, 0, sizeof(*h));
}

void lz_chat_hist_reset(LZChatHist *h) {
    int i;
    if (!h) return;
    for (i = 0; i < h->n; i++) {
        free(h->owned[i]);
        h->owned[i] = NULL;
    }
    h->n = 0;
}

void lz_chat_hist_free(LZChatHist *h) {
    lz_chat_hist_reset(h);
}

int lz_chat_hist_push(LZChatHist *h, LZRole role, const char *content,
                      int len, char *errbuf, int errlen) {
    char *copy;

    if (!h || !content) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return -1;
    }
    if (h->n >= LZ_CHAT_HIST_MAX) {
        if (errbuf)
            lz_err_fmt(errbuf, errlen, LZ_ERR_HIST_FULL, LZ_CHAT_HIST_MAX);
        return -1;
    }
    if (len < 0) len = (int)strlen(content);
    /* The copy is the whole point: LZChatMsg holds a view, and the caller
       that fills a chat history has neither a stable stack buffer nor a
       heap block it can keep - see the note in chat.h. */
    copy = (char *)malloc((size_t)len + 1);
    if (!copy) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_HIST_ALLOC);
        return -1;
    }
    memcpy(copy, content, (size_t)len);
    copy[len] = 0;
    h->owned[h->n] = copy;
    h->msgs[h->n].role = role;
    h->msgs[h->n].content = copy;
    h->msgs[h->n].len = len;
    h->n++;
    return 0;
}

int lz_chat_hist_pop(LZChatHist *h) {
    if (!h || h->n <= 0) return -1;
    h->n--;
    /* Both halves, not just the count: owned[] is one allocation per
       message (see push above), so leaving it behind leaks once per
       pop - and the front end's regenerate command pops on every use.
       Clearing msgs[] too keeps the slot from holding a pointer to
       freed memory, which is what the next push would overwrite but
       lz_chat_hist_free would not. */
    free(h->owned[h->n]);
    h->owned[h->n] = NULL;
    h->msgs[h->n].content = NULL;
    h->msgs[h->n].len = 0;
    h->msgs[h->n].role = LZ_ROLE_SYSTEM;
    return 0;
}
