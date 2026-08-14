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
#endif

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
#endif

#ifdef _WIN32
#include <windows.h>
#endif
#ifdef LZ_HAVE_SETMODE
#include <fcntl.h>          /* O_BINARY */
#include <io.h>             /* setmode, fileno */
#endif
#if !defined(_WIN32)
#include <time.h>
#endif

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

int lz_fseek64(FILE *f, unsigned long long off) {
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
    if (off > 0x7FFFFFFFULL) return 1;
    return fseek(f, (long)off, SEEK_SET) == 0 ? 0 : 1;
#endif
}

long long lz_fsize(const char *path) {
#if defined(_WIN32) && !defined(__WATCOMC__)
    FILE *f = fopen(path, "rb");
    __int64 n;
    if (!f) return -1;
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    n = _ftelli64(f);
    fclose(f);
    return (long long)n;
#else
    FILE *f = fopen(path, "rb");
    long n;
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    n = ftell(f);
    fclose(f);
    return (long long)n;
#endif
}

double lz_time_ms(void) {
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
        return (double)now.QuadPart * 1000.0 / (double)freq.QuadPart;
    }
    /* Fallback: ~55ms resolution on Win9x; benchmarking needs enough tokens to average out */
    return (double)GetTickCount();
#elif defined(__DOS__)
    /* clock(), because DOS has neither of the other two. It counts from
       process start rather than from an epoch, which is all any caller
       here needs - every use is a difference.

       THE RESOLUTION IS THE 18.2 Hz BIOS TICK, about 54.9 ms, whatever
       CLOCKS_PER_SEC says: the macro is the UNIT the value is reported
       in, not the rate the counter advances at. Anyone timing on this
       target must read iron law three's note on exactly this trap - the
       same one bit the Windows build, where clock()'s quantum is the
       15.6 ms scheduler tick and a 100 ms benchmark round reported
       2.02x and 3.41x for the same binary. Rounds have to be seconds
       long here, and "all the readings are integer multiples of one
       number" is how you recognise that they were not.

       Dividing by a constant is iron law six's rule 1, and this is the
       exemption that rule already carries: it protects the forward
       pass's cross-compiler bit-exactness, and a wall clock is not in
       the forward pass and is never compared bit for bit. Same
       exemption gui/main.c's temp_milli conversion claims. */
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
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
#endif
#endif
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
#endif
}

unsigned long long lz_seed_mix(unsigned long long x) {
    /* One round of xorshift64* (the same generator sampler.c's
       random_u32 uses), applied as a bijection over the seed. The
       multiplication constant is the same 0x2545F4914F6CDD1DULL so a
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
    x ^= 0x9E3779B97F4A7C15ULL;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
}
