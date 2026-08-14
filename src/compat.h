#ifndef LZ_COMPAT_H
#define LZ_COMPAT_H

#include <stddef.h>
#include <stdio.h>

/* Read an entire file into memory. On success returns the buffer and
   writes the size; on failure returns NULL and fills errbuf.
   Win98 has no mmap, so this path is used everywhere. */
void *lz_read_file(const char *path, size_t *out_size,
                   char *errbuf, int errlen);

/* 64-bit file positioning. Unpruned models are 1.7GB, so offsets exceed
   long (2GB on 32-bit platforms), while ANSI fseek only takes long -
   hence this platform shim (M4.5). Returns 0 on success; on 32-bit
   platforms an offset beyond long returns non-zero rather than silently
   truncating. */
int lz_fseek64(FILE *f, unsigned long long off);

/* Get file size, -1 on failure. Also needs 64 bits to cover unpruned models. */
long long lz_fsize(const char *path);

/* Monotonic millisecond timer. GetTickCount on Win98 has only 55ms
   resolution, so QueryPerformanceCounter is preferred. */
double lz_time_ms(void);

/* Derive a full-entropy seed from a weak, possibly-sparse input.
 *
 * The input is typically a wall-clock reading whose resolution is the
 * 18.2 Hz BIOS tick on DOS (~54.9 ms) and 55 ms on Win9x, so its low
 * bits are often zero and adjacent values differ only in the high few
 * bits. One round of the same xorshift64* the sampler draws from (see
 * random_u32 in sampler.c) avalanches that into the whole word, so
 * adjacent inputs land far apart.
 *
 * This is a pure function of its input and adds NO entropy - two calls
 * with the same input give the same seed, and on a machine whose clock
 * is coarser than the gap between two runs, those runs can still
 * collide. It exists to make the WEAK entropy the clock does provide
 * spread across the seed instead of concentrating in a few bits, and
 * to give every seed site one shared algorithm instead of the three
 * ad-hoc `lz_time_ms() * 1000` spellings that had drifted apart.
 *
 * Pure integer arithmetic, no float, no OS entropy API: identical on
 * DOS, NT 3.51 and Win95, and not part of the forward pass so iron law
 * six's bit-exactness rules do not bind it. */
unsigned long long lz_seed_mix(unsigned long long x);

/* Seconds since the Unix epoch, for OpenAI's `created` field.

   Separate from lz_time_ms because they answer different questions and
   neither substitutes for the other: lz_time_ms is MONOTONIC and has no
   epoch (QueryPerformanceCounter counts from an arbitrary origin), while
   this one is wall clock and may jump. Using the monotonic one here
   would put a number in `created` that no client can interpret.

   0 if the clock is unavailable - a Win98 box with a dead RTC is a real
   thing, and 0 is a fallback clients already tolerate. */
long lz_time_epoch(void);

/* Initialize stdout: binary mode + unbuffered.

   Binary mode is required, not an optimization: Windows text mode rewrites
   0x0A to 0x0D 0x0A in the output stream. The vocabulary may contain a
   standalone newline token; once the model generates it, the output byte
   stream gets corrupted, diverging from the Python reference and breaking
   the byte-for-byte reproducibility requirement.
   Verified that wcc386 and gcc behave identically under this call. */
void lz_init_stdout(void);
void lz_init_stdin(void);

#endif
