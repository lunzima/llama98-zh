/* The POSIX branches use fseeko/off_t and
   clock_gettime/CLOCK_MONOTONIC. These are POSIX symbols that glibc does
   not expose by default under strict -std=c99 (non-gnu99), so the feature
   macro must be declared here explicitly - and it must take effect before
   any #include. The Windows branch (incl. Open Watcom) defines no macro
   beyond _WIN32; no impact.

   __DOS__ IS EXCLUDED HERE TOO (the DOS extender target).
   It is neither Windows nor POSIX, and asking for _POSIX_C_SOURCE on a
   target that has no POSIX at all does not add the symbols - it only
   makes the failure arrive later and read stranger. The #else branch is
   POSIX, so without this exclusion lz_time_ms reaches for clock_gettime
   and wcc386 reports an undeclared CLOCK_MONOTONIC rather than "there
   is no such clock". */
#if !defined(_WIN32) && !defined(__DOS__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif /* !_WIN32 && !__DOS__ && !_POSIX_C_SOURCE */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "compat.h"
#include "err.h"

/* DOS AND WINDOWS SHARE THE TEXT-MODE PROBLEM, not the API surface.
   Both translate 0x0D 0x0A on a stream opened in text mode, and both
   fix it with setmode(O_BINARY) out of the same two headers; only
   Windows has QueryPerformanceCounter and the rest of windows.h. Two
   macros rather than one, so that a future branch has to say which of
   the two properties it means. */
#if defined(_WIN32) || defined(__DOS__)
#define LZ_HAVE_SETMODE 1
#endif /* _WIN32 || __DOS__ */

#ifdef _WIN32
#include <windows.h>
#endif /* _WIN32 */
#ifdef LZ_HAVE_SETMODE
#include <fcntl.h>          /* O_BINARY */
#include <io.h>             /* setmode, fileno */
#endif /* LZ_HAVE_SETMODE */
#if !defined(_WIN32)
#include <time.h>
#endif /* !_WIN32 */

void *lz_read_file(const char *path, size_t *out_size,
                   char *errbuf, int errlen) {
    FILE *f;
    long size;
    void *buf;
    size_t got;

    f = fopen(path, "rb");
    if (!f) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_OPEN_FILE, path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_FSEEK, path);
        fclose(f);
        return NULL;
    }
    size = ftell(f);
    if (size <= 0) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_EMPTY_FILE, path);
        fclose(f);
        return NULL;
    }
    rewind(f);

    buf = malloc((size_t)size);
    if (!buf) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_ALLOC, size, path);
        fclose(f);
        return NULL;
    }
    got = fread(buf, 1, (size_t)size, f);
    fclose(f);
    if (got != (size_t)size) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_READ_SHORT,
                             (unsigned long)got, size);
        free(buf);
        return NULL;
    }
    if (out_size) *out_size = (size_t)size;
    return buf;
}

int lz_fseek64(FILE *f, lz_u64 off) {
#if defined(_WIN32) && !defined(__WATCOMC__)
    /* MinGW/MSVC both provide _fseeki64. Watcom's C library lacks it; fall back below. */
    return _fseeki64(f, (__int64)off, SEEK_SET) == 0 ? 0 : 1;
#elif defined(_LARGEFILE_SOURCE) || defined(__USE_LARGEFILE64) || \
      (defined(__unix__) && !defined(__WATCOMC__))
    return fseeko(f, (off_t)off, SEEK_SET) == 0 ? 0 : 1;
#else
    /* Fallback: only long is available. Out-of-range must error out
       explicitly, never silently truncate - truncation reads a completely
       wrong tensor, and the symptom is extremely hard to locate. */
    if (off > LZ_U64_C(0x7FFFFFFF)) return 1;
    return fseek(f, (long)off, SEEK_SET) == 0 ? 0 : 1;
#endif /* _WIN32 && !__WATCOMC__ || _LARGEFILE_SOURCE || __USE_LARGEFILE64 || (__unix__ && !__WATCOMC__) */
}

lz_i64 lz_fsize(const char *path) {
#if defined(_WIN32) && !defined(__WATCOMC__)
    FILE *f = fopen(path, "rb");
    __int64 n;
    if (!f) return -1;
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    n = _ftelli64(f);
    fclose(f);
    return (lz_i64)n;
#else
    FILE *f = fopen(path, "rb");
    long n;
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    n = ftell(f);
    fclose(f);
    return (lz_i64)n;
#endif /* _WIN32 && !__WATCOMC__ */
}

float lz_time_ms(void) {
#ifdef _WIN32
    static int inited = 0;
    static int has_qpc = 0;
    static LARGE_INTEGER freq;
    LARGE_INTEGER now;

    if (!inited) {
        has_qpc = QueryPerformanceFrequency(&freq) && freq.QuadPart > 0;
        inited = 1;
    }
    if (has_qpc && QueryPerformanceCounter(&now)) {
        return (float)now.QuadPart * 1000.0f / (float)freq.QuadPart;
    }
    /* Fallback: ~55ms resolution on Win9x; benchmarking needs enough tokens to average out */
    return (float)GetTickCount();
#elif defined(__DOS__)
    /* clock(), because DOS has neither of the other two. It counts from
       process start rather than from an epoch, which is all any caller
       here needs - every use is a difference.

       THE RESOLUTION IS THE 18.2 Hz BIOS TICK, about 54.9 ms, whatever
       CLOCKS_PER_SEC says: the macro is the UNIT the value is reported
       in, not the rate the counter advances at. Anyone timing on this
       target has to know this trap - the same one bit the Windows
       build, where clock()'s quantum is the
       15.6 ms scheduler tick and a 100 ms benchmark round reported
       2.02x and 3.41x for the same binary. Rounds have to be seconds
       long here, and "all the readings are integer multiples of one
       number" is how you recognise that they were not.

       Dividing by a float constant is barred, and this is the
       exemption that rule already carries: it protects the forward
       pass's cross-compiler bit-exactness, and a wall clock is not in
       the forward pass and is never compared bit for bit. Same
       exemption gui/main.c's temp_milli conversion claims. */
    return (float)clock() * 1000.0f / (float)CLOCKS_PER_SEC;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (float)ts.tv_sec * 1000.0f + (float)ts.tv_nsec / 1e6f;
#endif /* _WIN32 || __DOS__ */
}

/* time() rather than a platform branch: it is ANSI C, Open Watcom has it,
   and its epoch is the one OpenAI's `created` is defined against. */
long lz_time_epoch(void) {
    time_t t = time(NULL);
    if (t == (time_t)-1) return 0;
    return (long)t;
}

void lz_init_stdout(void) {
#ifdef LZ_HAVE_SETMODE
    /* Switch to binary mode first, then disable buffering. Order is irrelevant, but both are required. */
    /* ...EXCEPT ON A DOS CONSOLE (reported as "CLI layout garbled, as
       if systematically missing the other half of CRLF").

       Binary mode turns OFF the runtime's \n -> \r\n translation. That
       is what we want when stdout is a file or a pipe: the engine
       emits UTF-8 whose 0x0A bytes are data, and --dump-logits and the
       token dumps have to come out byte for byte. It is wrong when
       stdout is the DOS console: DOS moves the cursor down on 0x0A and
       returns the carriage only on 0x0D, so every line starts where
       the previous one ended and the output walks off the right of the
       screen.

       Windows does not show this because its console treats a bare
       0x0A as a newline. Same call, same flag, different terminal -
       which is why the guard is on the DESTINATION and not on the
       platform.

       isatty, not "always text on DOS": a redirected run must stay
       byte-exact, and that is the case the gates read. */
#ifdef __DOS__
    if (!isatty(fileno(stdout)))
        setmode(fileno(stdout), O_BINARY);
#else
    setmode(fileno(stdout), O_BINARY);
#endif /* __DOS__ */
#endif /* LZ_HAVE_SETMODE */
    setvbuf(stdout, NULL, _IONBF, 0);
}

void lz_init_stdin(void) {
#ifdef LZ_HAVE_SETMODE
    /* Without binary mode, 0x0D 0x0A in the input would be folded into
       0x0A, yet in UTF-8 text 0x0D can be a legitimate part of content.
       DOS needs this as much as Windows does - the translation is the
       C runtime's, not the operating system's, and Open Watcom's DOS
       runtime does it too. Guarded on LZ_HAVE_SETMODE rather than
       _WIN32 for exactly that reason. */
    setmode(fileno(stdin), O_BINARY);
#endif /* LZ_HAVE_SETMODE */
}

lz_u64 lz_seed_mix(lz_u64 x) {
    /* One round of xorshift64* (the same generator sampler.c's
       random_u32 uses), applied as a bijection over the seed. The
       multiplication constant is the same LZ_U64_C(0x2545F4914F6CDD1D) so a
       seed that happens to equal a sampler state stays in the same
       family - not a correctness requirement, just the one already
       proven on every target.
       x is first XORed with a non-zero constant: 0 is a FIXED POINT of
       xorshift (mixing 0 yields 0), and 0 is exactly what lz_time_ms
       returns on DOS within the first clock tick of the process - so a
       fast "start and generate" would otherwise land every such run on
       seed 1 after the sampler's `seed ? seed : 1`. The constant is
       arbitrary but non-zero and odd. See compat.h for why this must
       stay pure. */
    x ^= LZ_U64_C(0x9E3779B97F4A7C15);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * LZ_U64_C(0x2545F4914F6CDD1D);
}

/* ---- byte order (see compat.h for the contract) --------------------- */

#if LZ_BIG_ENDIAN

/* Byte-reversal through an unsigned char * rather than through an
   integer type: the buffers these are handed hold float as often as
   int32_t, and reading a float's bytes as an integer to shift them is
   the aliasing violation the rest of this tree avoids with unions. Bytes
   are bytes under any type. */
void lz_le32(void *p, size_t n) {
    unsigned char *b = (unsigned char *)p;
    size_t i;
    for (i = 0; i < n; i++, b += 4) {
        unsigned char t;
        t = b[0]; b[0] = b[3]; b[3] = t;
        t = b[1]; b[1] = b[2]; b[2] = t;
    }
}

void lz_le16(void *p, size_t n) {
    unsigned char *b = (unsigned char *)p;
    size_t i;
    for (i = 0; i < n; i++, b += 2) {
        unsigned char t = b[0]; b[0] = b[1]; b[1] = t;
    }
}

#else /* !LZ_BIG_ENDIAN */

/* Empty on purpose. A little-endian host needs no conversion, and an
   empty body is the strongest available statement that these builds are
   byte-for-byte what they were before byte-order support existed. */
void lz_le32(void *p, size_t n) { (void)p; (void)n; }
void lz_le16(void *p, size_t n) { (void)p; (void)n; }

#endif /* LZ_BIG_ENDIAN */

int lz_endian_ok(void) {
    /* Not a compile-time trick: the whole point is to catch the case
       where the compile-time answer is WRONG, so the probe has to be
       something the running machine decides. `volatile` keeps a
       constant-folding compiler from answering it from the same macro
       list that may have got it wrong. */
    volatile unsigned long probe = 1UL;
    int host_big = (*(const volatile unsigned char *)&probe) == 0;
#if LZ_BIG_ENDIAN
    return host_big == 1;
#else /* !LZ_BIG_ENDIAN */
    return host_big == 0;
#endif /* LZ_BIG_ENDIAN */
}
