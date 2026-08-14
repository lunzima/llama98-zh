#include <string.h>

#include "net.h"
#include "err.h"

#ifdef _WIN32
#  define LZ_ERRNO()      WSAGetLastError()
#  define LZ_EWOULDBLOCK  WSAEWOULDBLOCK
#  define LZ_EINTR        WSAEINTR
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <signal.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/select.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define LZ_ERRNO()      errno
#  define LZ_EWOULDBLOCK  EAGAIN
#  define LZ_EINTR        EINTR
#  define closesocket     close
#endif

int lz_net_init(char *errbuf, int errlen) {
#ifdef _WIN32
    WSADATA wsa;
    /* 1.1 on purpose - see net.h. WSAStartup negotiates DOWN to the
       requested version, so asking for 1.1 works on every stack from
       Win95 to Windows 11. */
    if (WSAStartup(MAKEWORD(1, 1), &wsa) != 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NET_INIT);
        return LZ_ERR_NET_INIT;
    }
#else
    /* Writing to a socket whose peer has gone raises SIGPIPE, whose
       default action is to terminate the process. For this server that
       means a client pressing Ctrl-C mid-stream kills the model - after
       it has spent a minute loading weights. Windows has no SIGPIPE at
       all, so this is one of the places the platforms differ by
       omission rather than by spelling. */
    signal(SIGPIPE, SIG_IGN);
    (void)errbuf; (void)errlen;
#endif
    return LZ_ERR_OK;
}

void lz_net_shutdown(void) {
#ifdef _WIN32
    WSACleanup();
#endif
}

lz_sock lz_net_listen(int port, int backlog, char *errbuf, int errlen) {
    lz_sock s;
    struct sockaddr_in addr;

    if (port <= 0 || port > 65535) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NET_PORT, port);
        return LZ_BAD_SOCK;
    }
    s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == LZ_BAD_SOCK) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NET_SOCKET, LZ_ERRNO());
        return LZ_BAD_SOCK;
    }
#ifndef _WIN32
    /* POSIX only. On Windows SO_REUSEADDR lets a second process bind a
       port another process is already listening on, silently stealing
       connections - the opposite of what it means on POSIX, where it
       only skips the TIME_WAIT delay. The Windows-correct option
       (SO_EXCLUSIVEADDRUSE) does not exist on Win9x. Not setting it is
       right on both. */
    {
        int one = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&one,
                   sizeof(one));
    }
#endif
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)port);
    /* Loopback only; see net.h. */
    addr.sin_addr.s_addr = htonl(0x7F000001UL);

    if (bind(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NET_BIND, port,
                               LZ_ERRNO());
        closesocket(s);
        return LZ_BAD_SOCK;
    }
    if (listen(s, backlog > 0 ? backlog : 4) != 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NET_LISTEN, LZ_ERRNO());
        closesocket(s);
        return LZ_BAD_SOCK;
    }
    return s;
}

int lz_net_wait_accept(lz_sock ls, int timeout_ms) {
    fd_set rd;
    struct timeval tv;
    int r;

    FD_ZERO(&rd);
    FD_SET(ls, &rd);
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    /* The first argument is ignored by Winsock (it keeps an array, not a
       bitmap) but required on POSIX. Passing ls+1 is correct on POSIX
       and harmless on Windows. */
    r = select((int)ls + 1, &rd, NULL, NULL, timeout_ms < 0 ? NULL : &tv);
    if (r < 0) return LZ_ERRNO() == LZ_EINTR ? 0 : -1;
    return r > 0 ? 1 : 0;
}

lz_sock lz_net_accept(lz_sock ls, char *errbuf, int errlen) {
    lz_sock c = accept(ls, NULL, NULL);
    if (c == LZ_BAD_SOCK) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_NET_ACCEPT, LZ_ERRNO());
        return LZ_BAD_SOCK;
    }
    {   /* Streaming responses are written in small pieces; without this
           Nagle holds each one for up to 200 ms waiting for company, and
           a token-at-a-time stream turns into bursts. */
        int one = 1;
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, (const char *)&one,
                   sizeof(one));
    }
    return c;
}

void lz_net_close(lz_sock s) {
    if (s != LZ_BAD_SOCK) closesocket(s);
}

int lz_net_recv(lz_sock s, char *buf, int cap) {
    int n;
    for (;;) {
        n = recv(s, buf, cap, 0);
        if (n < 0 && LZ_ERRNO() == LZ_EINTR) continue;
        return n;
    }
}

int lz_net_send_all(lz_sock s, const char *buf, int n) {
    int off = 0;
    while (off < n) {
        int w = send(s, buf + off, n - off, 0);
        /* NOT gate-covered, and measured to be so rather than assumed:
           on a blocking socket send() blocks until the stack has taken
           everything, so a 1 MB response goes out in one call.
           It stays because POSIX send() can return short when
           a signal interrupts it mid-transfer, or with SO_SNDTIMEO -
           neither reachable here. */
        if (w > 0) { off += w; continue; }
        if (w < 0 && LZ_ERRNO() == LZ_EINTR) continue;
        if (w < 0 && LZ_ERRNO() == LZ_EWOULDBLOCK) continue;
        return -1;
    }
    return 0;
}

int lz_net_peer_gone(lz_sock s) {
    fd_set rd;
    struct timeval tv;
    char probe;
    int r;

    FD_ZERO(&rd);
    FD_SET(s, &rd);
    tv.tv_sec = 0;
    tv.tv_usec = 0;
    r = select((int)s + 1, &rd, NULL, NULL, &tv);
    if (r <= 0) return 0;               /* nothing to read = still there */
    /* Readable means either the peer sent something or it closed. MSG_PEEK
       so a pipelined request is not consumed here; recv returning 0 is
       the orderly-shutdown signal. */
    r = recv(s, &probe, 1, MSG_PEEK);
    if (r == 0) return 1;
    if (r < 0 && LZ_ERRNO() != LZ_EWOULDBLOCK) return 1;
    return 0;
}
