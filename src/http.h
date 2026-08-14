#ifndef LZ_HTTP_H
#define LZ_HTTP_H

#include "net.h"

/* A minimal HTTP/1.1 server core: one connection at a time, no threads,
   no TLS, no file serving, no CGI.

   Structure and limits follow uhttpd (openwrt/uhttpd), which is the
   reference for "small HTTP server that has already met real traffic".
   What was taken: its client state machine (INIT -> HEADER -> DATA ->
   DONE), the shape of request-line and header parsing, and - the part
   that is easy to omit when writing from scratch and expensive to
   discover later - its habit of capping header COUNT and cumulative
   header BYTES and answering 431 rather than letting one connection pin
   unbounded memory.

   What was NOT taken, and why: uhttpd's I/O is libubox - uloop (epoll)
   and ustream. None of that exists on Win9x, and no shim makes epoll
   appear, so the event loop here is select() over at most two sockets.
   That is not a compromise at concurrency 1; it is the whole
   requirement. TLS (ustream-ssl) is dropped per deployment: this
   listens on loopback behind a reverse proxy that terminates TLS.

   DELIBERATELY NOT keep-alive. Every response closes the connection.
   With one request in flight at a time and a proxy in front that pools
   its own upstream connections, keep-alive buys a TCP handshake per
   request and costs a connection-reuse state machine - and getting that
   state machine subtly wrong (a stale body, a half-consumed request)
   produces responses attributed to the wrong request, which is the
   worst failure mode available here. */

/* Caps. Chosen to be survivable on a 128 MB Win98 box where the model
   already holds ~106 MB: the entire request must fit in memory at once
   because there is nowhere to spill it. */
#define LZ_HTTP_MAX_LINE     4096    /* request line */
#define LZ_HTTP_MAX_HEADERS  64      /* header count */
#define LZ_HTTP_MAX_HDRBYTES 8192    /* cumulative header bytes */
#define LZ_HTTP_MAX_PATH     1024

typedef enum {
    LZ_HTTP_GET = 0,
    LZ_HTTP_POST,
    LZ_HTTP_HEAD,
    LZ_HTTP_OPTIONS,
    LZ_HTTP_OTHER
} LZHttpMethod;

typedef struct {
    LZHttpMethod method;
    char  path[LZ_HTTP_MAX_PATH];   /* path only; query stripped into query */
    char  query[LZ_HTTP_MAX_PATH];
    const char *body;               /* not NUL-terminated; may be NULL */
    int   body_len;
    int   http_11;                  /* 1 = HTTP/1.1, 0 = 1.0 or older */
    /* Header lookup is a linear scan over what was parsed. With a cap of
       64 that is cheaper than any index, and it keeps the header block a
       plain buffer rather than another allocation. */
    const char *hdr_name[LZ_HTTP_MAX_HEADERS];
    const char *hdr_value[LZ_HTTP_MAX_HEADERS];
    int   n_headers;
} LZHttpReq;

/* Case-insensitive header lookup; NULL if absent. */
const char *lz_http_header(const LZHttpReq *r, const char *name);

/* Response handle. A handler either calls lz_http_reply once, or opens a
   stream with lz_http_stream_begin and then writes chunks. Doing
   neither produces a 500 from the core, so a handler that forgets to
   answer fails loudly instead of hanging the client. */
typedef struct {
    lz_sock sock;
    int     started;    /* headers written */
    int     chunked;    /* streaming */
    int     finished;
    int     failed;     /* the peer went away mid-response */
} LZHttpResp;

/* Buffered response with Content-Length. body may be NULL when len is 0. */
int lz_http_reply(LZHttpResp *rs, int status, const char *ctype,
                  const char *body, int len);

/* Streaming response (chunked). ctype is e.g. "text/event-stream".
   lz_http_stream_write returns non-zero once the peer is gone; a caller
   generating tokens should stop rather than spend a Pentium's minutes
   producing bytes nobody will read. */
int lz_http_stream_begin(LZHttpResp *rs, int status, const char *ctype);
int lz_http_stream_write(LZHttpResp *rs, const char *bytes, int n);
int lz_http_stream_end(LZHttpResp *rs);

/* Handler: fill the response. Return value is ignored - the response is
   whatever the handler wrote. */
typedef void (*LZHttpHandler)(const LZHttpReq *req, LZHttpResp *rs,
                              void *ctx);

typedef struct {
    lz_sock listener;
    char   *buf;            /* request assembly buffer */
    int     cap;            /* its size = max request bytes */
    int     stop;           /* set to break the serve loop */
} LZHttpServer;

/* max_request is the cap on request line + headers + body together. */
int  lz_http_open(LZHttpServer *sv, int port, int max_request,
                  char *errbuf, int errlen);
void lz_http_close(LZHttpServer *sv);

/* Accept and serve at most one connection, waiting up to timeout_ms for
   one to arrive. Returns 1 if a request was served, 0 on timeout,
   negative on a listener error. Malformed requests are answered with an
   error status and still count as served - a client's bad syntax is not
   the server's failure. */
int  lz_http_serve_one(LZHttpServer *sv, LZHttpHandler h, void *ctx,
                       int timeout_ms, char *errbuf, int errlen);

#endif
