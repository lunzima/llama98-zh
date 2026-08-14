/* See chatfile.h. */
#include <string.h>

#include "chatfile.h"
#include "err.h"

static const struct { LZRole role; const char *tag; } TAGS[] = {
    { LZ_ROLE_SYSTEM,    "[system]"    },
    { LZ_ROLE_USER,      "[user]"      },
    { LZ_ROLE_ASSISTANT, "[assistant]" }
};
#define N_TAGS (int)(sizeof TAGS / sizeof TAGS[0])

/* Append n bytes, counting past the end rather than writing past it -
   the snprintf-semantics contract lz_chatfile_encode promises. `out`
   may be NULL (the out=NULL,cap=0 counting form); the bound below never
   holds when cap<=0, so out is never dereferenced in that case. */
static void put(char *out, int cap, int *len, const char *s, int n) {
    int i;
    for (i = 0; i < n; i++, (*len)++)
        if (*len < cap - 1) out[*len] = s[i];
}

/* Is line [a,b) exactly one of the markers, allowing the doubled form
   that lz_chatfile_encode produces when message content itself is a
   marker line? Returns the number of leading '[' consumed (>=1) when it
   is such a line, 0 when it is not. A return of 1 means a REAL marker
   (role switch); >1 means escaped content (one '[' must be dropped on
   the way back in). */
static int is_marker_line(const char *s, int a, int b) {
    int i, lead = 0;
    while (a + lead < b && s[a + lead] == '[') lead++;
    if (lead == 0) return 0;
    for (i = 0; i < N_TAGS; i++) {
        int n = (int)strlen(TAGS[i].tag);
        /* One '[' is the tag itself; lead>1 is escaped content. */
        if (b - a == n + lead - 1 &&
            memcmp(s + a + lead - 1, TAGS[i].tag, (size_t)n) == 0)
            return lead;
    }
    return 0;
}

int lz_chatfile_encode(const LZChatMsg *msgs, int n, char *out, int cap) {
    int len = 0, i;
    if (out && cap > 0) out[0] = '\0';
    put(out, cap, &len, LZ_CHATFILE_MAGIC, (int)strlen(LZ_CHATFILE_MAGIC));
    put(out, cap, &len, "\n", 1);
    for (i = 0; i < n && msgs; i++) {
        int t, a = 0, blen;
        const char *body = msgs[i].content;
        for (t = 0; t < N_TAGS; t++) if (TAGS[t].role == msgs[i].role) break;
        if (t == N_TAGS) continue;          /* unknown role: not written */
        put(out, cap, &len, TAGS[t].tag, (int)strlen(TAGS[t].tag));
        put(out, cap, &len, "\n", 1);
        blen = msgs[i].len < 0 ? (int)strlen(body) : msgs[i].len;
        while (a <= blen) {
            int b = a;
            while (b < blen && body[b] != '\n') b++;
            /* A content line that would read back as a marker gets one
               more '[' - and so does an already-escaped one, or the
               un-escaping on the way in would not be reversible. */
            if (is_marker_line(body, a, b)) put(out, cap, &len, "[", 1);
            put(out, cap, &len, body + a, b - a);
            put(out, cap, &len, "\n", 1);
            if (b >= blen) break;
            a = b + 1;
        }
    }
    if (out && cap > 0) out[len < cap ? len : cap - 1] = '\0';
    return len;
}

/* Push one turn, translating lz_chat_hist_push's bare -1 into the
   LZ_ERR_* code that matches the message it already wrote into errbuf.
   Needed because lz_chat_hist_push itself only ever returns -1: without
   this, a caller mapping the return code onto behaviour (or onto an
   HTTP status, per err.h) would see a code that disagrees with the text
   sitting next to it in errbuf, which is exactly the failure mode
   err.h's own comment on LZ_ERR_SET warns about. The two ways it can
   fail here (h and content are never NULL at these call sites) are
   history-full and allocation failure, and which one just happened is
   decidable from h->n before the call. */
static int push_turn(LZChatHist *h, LZRole role, const char *content, int n,
                     char *errbuf, int errlen) {
    int full = (h->n >= LZ_CHAT_HIST_MAX);
    if (lz_chat_hist_push(h, role, content, n, errbuf, errlen) != 0)
        return full ? LZ_ERR_HIST_FULL : LZ_ERR_HIST_ALLOC;
    return 0;
}

int lz_chatfile_decode(const char *bytes, int len, LZChatHist *h,
                       const char *path, char *errbuf, int errlen) {
    LZChatHist tmp;
    int i, rc = 0, cur = -1, mlen = (int)strlen(LZ_CHATFILE_MAGIC);
    static char body[LZ_CHATFILE_BODY];    /* iron law six: not on the stack */
    int blen = 0, k;

    if (!bytes || !h) { LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
                        return rc; }
    if (len < mlen || memcmp(bytes, LZ_CHATFILE_MAGIC, (size_t)mlen) != 0) {
        /* No LZ_ERR_BAD_FORMAT in err.h; reusing LZ_ERR_EMPTY_FILE ("empty
           or invalid file: %s") is exactly this shape and already carries
           a %s for the path, so it does not need a new engine error code
           (which would also mean touching src/err.c's bilingual table and
           tools_check_err_full.py - out of scope here). */
        LZ_ERR_SET1(rc, errbuf, errlen, LZ_ERR_EMPTY_FILE, path ? path : "?");
        return rc;
    }
    /* Built into a SECOND history and swapped in at the end, so a
       failure halfway leaves the caller's untouched. */
    lz_chat_hist_init(&tmp);
    i = mlen;
    while (i < len && bytes[i] != '\n') i++;
    if (i < len) i++;

    /* i < len, not i <= len: the file this module writes always ends
       every content line with '\n' (see encode above), including the
       last one. Splitting on '\n' the naive way (the way str.split
       does) yields one trailing EMPTY line once i reaches len - that
       line is an artifact of where the last real line ended, not data,
       and must not become an extra blank line appended to whichever
       turn happens to be last in the file. i<len drops exactly that
       phantom iteration while still processing a genuinely unterminated
       final line (b reaches len from inside the loop below, which hits
       its own `if (b >= len) break;` before this condition is ever
       re-tested). */
    while (i < len) {
        int a = i, b = i, lead;
        while (b < len && bytes[b] != '\n') b++;
        /* CRLF: the trailing CR belongs to the separator, not the text. */
        {
            int e = b;
            if (e > a && bytes[e - 1] == '\r') e--;
            lead = is_marker_line(bytes, a, e);
            if (lead == 1) {
                if (cur >= 0) {
                    int prc = push_turn(&tmp, TAGS[cur].role, body, blen,
                                        errbuf, errlen);
                    if (prc != 0) { lz_chat_hist_free(&tmp); return prc; }
                }
                for (k = 0; k < N_TAGS; k++) {
                    int tn = (int)strlen(TAGS[k].tag);
                    if (e - a == tn &&
                        memcmp(bytes + a, TAGS[k].tag, (size_t)tn) == 0) {
                        cur = k; break;
                    }
                }
                blen = 0;
            } else if (cur >= 0) {
                int from = a + (lead > 1 ? 1 : 0);   /* un-escape */
                if (blen && blen < LZ_CHATFILE_BODY) body[blen++] = '\n';
                for (k = from; k < e && blen < LZ_CHATFILE_BODY; k++) {
                    /* Every CR is dropped, not just the one the `e--`
                       above already took off the end of the line. That
                       one covers a CRLF file; this covers a CR that is
                       not a line ending at all - an external editor's
                       leftover, an old-Mac line ending, a stray
                       keystroke in a .txt people open in Notepad.
                       Without it chatfile.h's "content must not contain
                       '\r'" is false on this path. Dropping rather than
                       translating to '\n': a CR mid-line is damage, and
                       inventing a line break where the file does not
                       have one would change how many lines the message
                       has. */
                    if (bytes[k] == '\r') continue;
                    body[blen++] = bytes[k];
                }
            }
        }
        if (b >= len) break;
        i = b + 1;
    }
    if (cur >= 0) {
        int prc = push_turn(&tmp, TAGS[cur].role, body, blen, errbuf, errlen);
        if (prc != 0) { lz_chat_hist_free(&tmp); return prc; }
    }
    lz_chat_hist_free(h);
    *h = tmp;                  /* owns tmp's allocations from here */
    return 0;
}
