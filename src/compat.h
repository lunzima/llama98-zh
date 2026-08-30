#ifndef LZ_COMPAT_H
#define LZ_COMPAT_H

#include "lz_int.h"   /* lz_i64/lz_u64: the 64-bit type, portably */
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
int lz_fseek64(FILE *f, lz_u64 off);

/* Get file size, -1 on failure. Also needs 64 bits to cover unpruned models. */
lz_i64 lz_fsize(const char *path);

/* Monotonic millisecond timer. GetTickCount on Win98 has only 55ms
   resolution, so QueryPerformanceCounter is preferred. Returns float:
   the engine is all-float (no double anywhere), and millisecond timing
   needs far less than float's 24-bit mantissa. */
float lz_time_ms(void);

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
 * to give every seed site one shared algorithm rather than a private
 * `lz_time_ms() * 1000` spelling each.
 *
 * Pure integer arithmetic, no float, no OS entropy API: identical on
 * DOS, NT 3.51 and Win95, and not part of the forward pass, so the
 * cross-compiler bit-exactness rules do not bind it. */
lz_u64 lz_seed_mix(lz_u64 x);

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

/* ---- byte order -----------------------------------------------------
 *
 * On-disk model data is LITTLE-ENDIAN, always, on every host. That is
 * not a preference, it is what already exists: export_q8.py writes every
 * scalar with struct.pack("<i"/"<f"), and safetensors is little-endian
 * by specification. A reader is therefore converting from a fixed
 * external order to host order, never "writing whatever this machine
 * uses" - so there is no second convention to negotiate and no byte-order
 * flag in the file.
 *
 * LZ_BIG_ENDIAN is 1 on a big-endian host, 0 otherwise. Every shipping
 * target is little-endian - x86 (gcc/MinGW/Open Watcom), x86-64, ARMv5TE,
 * and MIPS under Windows NT, which runs little-endian on a Loongson 2F
 * just as it did on an R4000 - so the value is 0 everywhere the engine is
 * actually built, and the conversion helpers below compile to nothing.
 * That is deliberate: it is what makes "byte order support" cost zero
 * instructions and zero bits of output on the machines that ship.
 *
 * Detection is compile-time with a conservative macro list and a
 * -DLZ_BIG_ENDIAN=1 override, because there is no portable compile-time
 * predicate and VC++ 4.0 on MIPS predates every standard one. A host the
 * list does not recognise is assumed little-endian and then CAUGHT at
 * runtime by lz_endian_ok() - see there for why that is the safe way
 * round.
 */
#if !defined(LZ_BIG_ENDIAN)
# if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
#  if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#   define LZ_BIG_ENDIAN 1
#  else /* __BYTE_ORDER__ != __ORDER_BIG_ENDIAN__ */
#   define LZ_BIG_ENDIAN 0
#  endif /* __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ */
# elif defined(__BIG_ENDIAN__) || defined(_BIG_ENDIAN) || \
       defined(__MIPSEB__) || defined(_MIPSEB) || defined(MIPSEB) || \
       defined(__ARMEB__) || defined(__THUMBEB__) || defined(__AARCH64EB__) || \
       defined(__sparc) || defined(__sparc__) || defined(__hppa__) || \
       defined(__s390__) || defined(__s390x__) || defined(__m68k__)
#  define LZ_BIG_ENDIAN 1
# else /* no big-endian macro matched */
#  define LZ_BIG_ENDIAN 0
# endif /* byte-order macro probe */
#endif /* !LZ_BIG_ENDIAN */

/* Convert n words in place between little-endian and host order. Both
 * directions are the same operation, so one function serves load and
 * store; the names say the EXTERNAL order, which is the one that is
 * fixed.
 *
 * On a little-endian host these are empty functions - not merely cheap
 * but incapable of changing a byte, which is what makes the little-endian
 * builds bit-identical to the versions of this engine that had no
 * byte-order handling at all.
 *
 * These take void * because every caller has a differently-typed buffer
 * (float *, int32_t *, a struct field) and casting at the call site is
 * how the alignment question gets answered: the pointer always comes from
 * malloc or from an object of the right type, never from a byte offset
 * into a packed buffer. */
void lz_le32(void *p, size_t n);
void lz_le16(void *p, size_t n);

/* Does LZ_BIG_ENDIAN match the machine actually running? 1 yes, 0 no.
 *
 * The macro list above cannot know every compiler, and the failure it
 * has when it guesses wrong is the worst kind: a big-endian host that
 * took the little-endian default loads every weight byte-reversed and
 * produces confident nonsense, with no crash and no message. This is
 * checked once per model open so that case becomes a refusal instead.
 *
 * The reverse mistake - LZ_BIG_ENDIAN=1 on a little-endian host - is
 * caught by the same comparison. */
int lz_endian_ok(void);

#endif
