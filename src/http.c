#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "http.h"
#include "err.h"

/* ------------------------------------------------------------ helpers */

static int ci_eq(const char *a, const char *b) {
    while (*a && *b) {
        int ca = (*a >= 'A' && *a <= 'Z') ? *a + 32 : *a;
        int cb = (*b >= 'A' && *b <= 'Z') ? *b + 32 : *b;
        if (ca != cb) return 0;
        a++; b++;
    }
    return *a == 0 && *b == 0;
}

const char *lz_http_header(const LZHttpReq *r, const char *name) {
    int i;
    if (!r || !name) return NULL;
    for (i = 0; i < r->n_headers; i++)
        if (ci_eq(r->hdr_name[i], name)) return r->hdr_value[i];
    return NULL;
}

static const char *status_text(int s) {
    switch (s) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 408: return "Request Timeout";
    case 413: return "Payload Too Large";
    case 431: return "Request Header Fields Too Large";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 503: return "Service Unavailable";
    default:  return "Error";
    }
}

/* ----------------------------------------------------------- response */

static int put(LZHttpResp *rs, const char *s, int n) {
    if (rs->failed) return -1;
    if (lz_net_send_all(rs->sock, s, n) != 0) {
        rs->failed = 1;
        return -1;
    }
    return 0;
}

static int puts_(LZHttpResp *rs, const char *s) {
    return put(rs, s, (int)strlen(s));
}

int lz_http_reply(LZHttpResp *rs, int status, const char *ctype,
                  const char *body, int len) {
    char head[512];
    if (!rs || rs->started) return -1;
    rs->started = 1;
    rs->finished = 1;
    if (len < 0) len = 0;
    snprintf(head, sizeof(head),
             "HTTP/1.1 %d %s\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %d\r\n"
             "Connection: close\r\n"
             "\r\n",
             status, status_text(status),
             ctype ? ctype : "text/plain; charset=utf-8", len);
    if (puts_(rs, head) != 0) return -1;
    if (len > 0 && body) return put(rs, body, len);
    return 0;
}

int lz_http_stream_begin(LZHttpResp *rs, int status, const char *ctype) {
    char head[512];
    if (!rs || rs->started) return -1;
    rs->started = 1;
    rs->chunked = 1;
    snprintf(head, sizeof(head),
             "HTTP/1.1 %d %s\r\n"
             "Content-Type: %s\r\n"
             "Transfer-Encoding: chunked\r\n"
             /* Proxies and browsers buffer event streams by default,
                which turns token-at-a-time output into one lump at the
                end - the streaming is still happening, it is just
                invisible, and that reads as "the server is slow". */
             "Cache-Control: no-cache\r\n"
             "X-Accel-Buffering: no\r\n"
             "Connection: close\r\n"
             "\r\n",
             status, status_text(status),
             ctype ? ctype : "text/event-stream");
    return puts_(rs, head);
}

/* Three sends per chunk (size line, body, trailer) rather than one
   assembled buffer. Measured on the wire: each SSE event does leave as
   three segments. Left alone deliberately - coalescing would save two
   syscalls per token, about 600 per response, which on a Pentium II is
   roughly a millisecond against a 240-494 ms per-token budget. Copying
   the body into a scratch buffer to save that would cost more in the
   memcpy than it saves in the syscall. */
int lz_http_stream_write(LZHttpResp *rs, const char *bytes, int n) {
    char sz[32];
    if (!rs || !rs->chunked || rs->finished) return -1;
    if (n <= 0) return 0;               /* a 0-length chunk would END it */
    snprintf(sz, sizeof(sz), "%x\r\n", (unsigned)n);
    if (puts_(rs, sz) != 0) return -1;
    if (put(rs, bytes, n) != 0) return -1;
    return puts_(rs, "\r\n");
}

int lz_http_stream_end(LZHttpResp *rs) {
    if (!rs || !rs->chunked || rs->finished) return -1;
    rs->finished = 1;
    return puts_(rs, "0\r\n\r\n");
}

/* ------------------------------------------------------------ request */

/* Split "Name: value" in place. Returns the value, or NULL if the line
   is not a valid header. */
static char *split_header(char *line) {
    char *colon = strchr(line, ':');
    char *v;
    if (!colon || colon == line) return NULL;
    /* RFC 9110: no whitespace between the field name and the colon. A
       space there is how request smuggling gets started, so reject
       rather than trim. */
    if (colon[-1] == ' ' || colon[-1] == '\t') return NULL;
    *colon = 0;
    v = colon + 1;
    while (*v == ' ' || *v == '\t') v++;
    return v;
}

static LZHttpMethod parse_method(const char *s) {
    if (strcmp(s, "GET") == 0) return LZ_HTTP_GET;
    if (strcmp(s, "POST") == 0) return LZ_HTTP_POST;
    if (strcmp(s, "HEAD") == 0) return LZ_HTTP_HEAD;
    if (strcmp(s, "OPTIONS") == 0) return LZ_HTTP_OPTIONS;
    return LZ_HTTP_OTHER;
}

/* Parse the head (request line + headers) that occupies buf[0..head_len).
   The buffer is modified in place: NULs are written over delimiters and
   the req points into it. Returns 0, or an HTTP status to answer with. */
static int parse_head(char *buf, int head_len, LZHttpReq *req) {
    char *p = buf, *end = buf + head_len;
    char *line, *sp1, *sp2, *q;
    int hdr_bytes = 0;

    memset(req, 0, sizeof(*req));

    /* request line */
    line = p;
    while (p < end && *p != '\n') p++;
    if (p >= end) return 400;
    if (p > line && p[-1] == '\r') p[-1] = 0;
    *p++ = 0;

    sp1 = strchr(line, ' ');
    if (!sp1) return 400;
    *sp1++ = 0;
    sp2 = strchr(sp1, ' ');
    if (!sp2) return 400;               /* HTTP/0.9 is not served */
    *sp2++ = 0;

    req->method = parse_method(line);
    if (req->method == LZ_HTTP_OTHER) return 501;
    req->http_11 = (strcmp(sp2, "HTTP/1.1") == 0);

    q = strchr(sp1, '?');
    if (q) {
        *q++ = 0;
        if ((int)strlen(q) >= LZ_HTTP_MAX_PATH) return 414;
        strcpy(req->query, q);
    }
    if ((int)strlen(sp1) >= LZ_HTTP_MAX_PATH) return 414;
    strcpy(req->path, sp1);

    /* headers */
    while (p < end) {
        char *name, *val;
        line = p;
        while (p < end && *p != '\n') p++;
        if (p >= end) break;
        if (p > line && p[-1] == '\r') p[-1] = 0;
        *p++ = 0;
        if (!*line) break;              /* blank line ends the head */

        /* uhttpd's cap, and for its reason: without it a client can hold
           the server on one connection while it feeds header lines, and
           each one is retained for the duration of the request. */
        hdr_bytes += (int)strlen(line) + 2;
        if (req->n_headers >= LZ_HTTP_MAX_HEADERS ||
            hdr_bytes > LZ_HTTP_MAX_HDRBYTES)
            return 431;

        val = split_header(line);
        if (!val) return 400;
        name = line;
        req->hdr_name[req->n_headers] = name;
        req->hdr_value[req->n_headers] = val;
        req->n_headers++;
    }
    return 0;
}

/* -------------------------------------------------------------- server */

int lz_http_open(LZHttpServer *sv, int port, int max_request,
                 char *errbuf, int errlen) {
    if (!sv) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return LZ_ERR_NULL_ARG;
    }
    memset(sv, 0, sizeof(*sv));
    if (max_request < 8192) max_request = 8192;
    sv->cap = max_request;
    sv->buf = (char *)malloc((size_t)max_request + 1);
    if (!sv->buf) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_ALLOC,
                               (long)max_request, "http request buffer");
        return LZ_ERR_ALLOC;
    }
    sv->listener = lz_net_listen(port, 8, errbuf, errlen);
    if (sv->listener == LZ_BAD_SOCK) {
        free(sv->buf);
        sv->buf = NULL;
        return LZ_ERR_NET_LISTEN;
    }
    return LZ_ERR_OK;
}

void lz_http_close(LZHttpServer *sv) {
    if (!sv) return;
    lz_net_close(sv->listener);
    sv->listener = LZ_BAD_SOCK;
    free(sv->buf);
    sv->buf = NULL;
}

/* Answer without invoking the handler. Used for anything rejected during
   parsing; the body is deliberately plain text, not JSON, because at
   this point we do not yet know the request was even meant for a JSON
   endpoint. */
static void fail(LZHttpResp *rs, int status) {
    const char *t = status_text(status);
    lz_http_reply(rs, status, "text/plain; charset=utf-8", t, (int)strlen(t));
}

int lz_http_serve_one(LZHttpServer *sv, LZHttpHandler h, void *ctx,
                      int timeout_ms, char *errbuf, int errlen) {
    lz_sock cs;
    LZHttpReq req;
    LZHttpResp rs;
    int n = 0, head_len = -1, want = -1, status, r;
    const char *cl;

    if (!sv || !sv->buf || !h) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NULL_ARG);
        return -1;
    }
    r = lz_net_wait_accept(sv->listener, timeout_ms);
    if (r <= 0) return r;               /* 0 = timeout, <0 = error */

    cs = lz_net_accept(sv->listener, errbuf, errlen);
    if (cs == LZ_BAD_SOCK) return -1;

    memset(&rs, 0, sizeof(rs));
    rs.sock = cs;

    /* Read until the head is complete, then until the body is. One
       buffer, one pass: the cap is the whole request, so there is no
       state to carry between reads beyond how much has arrived. */
    for (;;) {
        int got;
        if (head_len >= 0 && want >= 0 && n >= head_len + want) break;
        if (n >= sv->cap) {
            fail(&rs, head_len < 0 ? 431 : 413);
            goto done;
        }
        got = lz_net_recv(cs, sv->buf + n, sv->cap - n);
        if (got < 0) goto done;         /* connection broke; nothing to say */
        if (got == 0) {
            /* Orderly shutdown. Fine if the request was complete, a
               truncated request otherwise. */
            if (head_len >= 0 && want >= 0 && n >= head_len + want) break;
            if (head_len < 0) { fail(&rs, 400); goto done; }
            break;
        }
        n += got;
        sv->buf[n] = 0;

        if (head_len < 0) {
            char *e = strstr(sv->buf, "\r\n\r\n");
            char *e2 = strstr(sv->buf, "\n\n");
            /* Tolerate bare LF: some proxies and most hand-typed test
               clients send it, and rejecting it turns a debugging
               session into a hunt for invisible carriage returns. */
            if (e && (!e2 || e < e2)) head_len = (int)(e - sv->buf) + 4;
            else if (e2)              head_len = (int)(e2 - sv->buf) + 2;
            if (head_len < 0 && n > LZ_HTTP_MAX_HDRBYTES + LZ_HTTP_MAX_LINE) {
                fail(&rs, 431);
                goto done;
            }
            if (head_len >= 0) {
                char saved = sv->buf[head_len];
                sv->buf[head_len] = 0;
                status = parse_head(sv->buf, head_len, &req);
                sv->buf[head_len] = saved;
                if (status != 0) { fail(&rs, status); goto done; }
                cl = lz_http_header(&req, "Content-Length");
                want = cl ? atoi(cl) : 0;
                if (want < 0 || want > sv->cap - head_len) {
                    fail(&rs, 413);
                    goto done;
                }
                /* Transfer-Encoding on the REQUEST is not supported. A
                   reverse proxy sends Content-Length; accepting a
                   chunked request while ignoring the framing would read
                   the chunk headers as body bytes, which is silent
                   corruption rather than an error. */
                if (lz_http_header(&req, "Transfer-Encoding")) {
                    fail(&rs, 501);
                    goto done;
                }
            }
        }
    }

    req.body = want > 0 ? sv->buf + head_len : NULL;
    req.body_len = want;
    h(&req, &rs, ctx);

    /* A handler that answered nothing is a bug in the handler, and the
       client must not be left waiting on a socket that will never
       produce bytes. */
    if (!rs.started) fail(&rs, 500);
    else if (rs.chunked && !rs.finished) lz_http_stream_end(&rs);

done:
    lz_net_close(cs);
    return 1;
}
