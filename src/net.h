#ifndef LZ_NET_H
#define LZ_NET_H

/* Sockets, thin. Everything here exists because Winsock and BSD sockets
   disagree in ways that compile silently and fail at runtime.

   WINSOCK 1.1, deliberately, not 2. Nothing here needs anything Winsock
   2 added - WSAPoll is Vista and later, and at concurrency 1 select()
   over two descriptors is the whole requirement. 1.1 (wsock32.lib) is
   present on every Win9x from Win95 OSR1 onward; Winsock 2 needs an
   update on Win95 and buys us nothing. Open Watcom ships both headers
   and both import libraries, so this is a free choice - make it the one
   that runs on more machines.

   The five differences that matter, each of which is a runtime failure
   rather than a compile error if you get it wrong:

     1. A socket is SOCKET (unsigned) on Windows, int on POSIX, and the
        failure value is INVALID_SOCKET (~0) rather than -1. `if (s < 0)`
        is always false on Windows. That is why LZ_BAD_SOCK exists.
     2. close() does not close a socket on Windows - closesocket() does.
        close() on a socket handle either fails or closes an unrelated
        CRT file descriptor.
     3. Errors do not go to errno; they go to WSAGetLastError(), and the
        codes are different (WSAEWOULDBLOCK, not EAGAIN).
     4. Winsock must be started (WSAStartup) and stopped. Skipping it
        makes every call fail with WSANOTINITIALISED.
     5. POSIX raises SIGPIPE when writing to a socket whose peer has
        gone, and the default action is to kill the process. Windows has
        no such signal. A generation streaming to a client that hung up
        must end as a cancelled request, not as a dead server. */

#ifdef _WIN32
#  include <winsock.h>
typedef SOCKET lz_sock;
#  define LZ_BAD_SOCK  INVALID_SOCKET
#else
typedef int lz_sock;
#  define LZ_BAD_SOCK  (-1)
#endif

/* Start/stop the socket stack. No-op on POSIX except for the SIGPIPE
   disposition. Returns 0 on success. */
int  lz_net_init(char *errbuf, int errlen);
void lz_net_shutdown(void);

/* Listen on 127.0.0.1:port. Loopback only and not configurable: this
   process speaks plain HTTP with no authentication and is meant to sit
   behind a reverse proxy (spec section 10.1). Binding it to 0.0.0.0
   would put an unauthenticated model endpoint on the network, and
   "the operator can be careful" is not a control.
   backlog is passed to listen(). Returns LZ_BAD_SOCK on failure. */
lz_sock lz_net_listen(int port, int backlog, char *errbuf, int errlen);

/* Wait until `ls` has a connection pending or `timeout_ms` elapses.
   Returns 1 = ready, 0 = timed out, -1 = error. timeout_ms < 0 blocks.
   select(), not poll(): poll does not exist in Winsock at all, and
   WSAPoll arrived in Vista. */
int  lz_net_wait_accept(lz_sock ls, int timeout_ms);

lz_sock lz_net_accept(lz_sock ls, char *errbuf, int errlen);
void lz_net_close(lz_sock s);

/* Read up to cap bytes. Returns >0 bytes read, 0 on orderly shutdown,
   -1 on error. */
int  lz_net_recv(lz_sock s, char *buf, int cap);

/* Send all n bytes, looping over partial writes. Returns 0 on success,
   -1 if the peer is gone or the connection broke. A short send() is
   normal, not an error - treating it as one truncates responses under
   exactly the conditions (slow client, large body) that are hardest to
   reproduce. */
int  lz_net_send_all(lz_sock s, const char *buf, int n);

/* True when the peer has closed or sent data unexpectedly. Called to
   stop a generation whose client has gone away rather than spending
   minutes of a Pentium's time producing bytes nobody will read. */
int  lz_net_peer_gone(lz_sock s);

#endif
