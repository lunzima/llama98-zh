#ifndef LZ_INT_H
#define LZ_INT_H
/* One authority for the integer types. The target family includes a
 * 1995 compiler, and it has neither of the two things this tree reaches
 * for by reflex.
 *
 * WHAT VISUAL C++ 4.0 DOES NOT HAVE, and the count of what that reached
 * when the conversion started (grep over src/):
 *
 *   <stdint.h>      absent - arrived in Visual C++ 2010.   23 files
 *   long long       absent - the 64-bit type is __int64.  213 uses
 *
 * The alternative to this header is 236 sites each deciding for
 * themselves, which is the shape "one authority per fact" exists to
 * forbid: two copies drift, and the second one is always the wrong one.
 *
 * WHY THE 64-BIT TYPE GETS A NEW NAME AND THE WIDTH TYPES DO NOT.
 * int32_t and friends are spelled the same everywhere once someone
 * supplies them, so this header supplies them where the platform does
 * not and includes <stdint.h> where it does. `long long` cannot be
 * spelled portably at all - it is a keyword the old compiler rejects -
 * so it becomes lz_i64/lz_u64 and the call sites change. On every
 * toolchain that exists today those typedefs ARE `long long`, so the
 * conversion is a rename with no arithmetic in it, and bit-identity is
 * the gate that says so.
 *
 * NOT A PORT. Nothing here has been compiled by Visual C++ 4.0 - there
 * is no such compiler on this machine. What this header buys is that
 * the port becomes one file's problem instead of 236 files' problem;
 * the claim that it WORKS there is not made and must not be inferred
 * from the fact that it builds under gcc and Watcom.
 */

/* _MSC_VER 1600 is Visual C++ 2010, the first with <stdint.h>. VC++ 4.0
 * is 1000. Anything else that lacks the header can join this arm by
 * name rather than by a version comparison it does not satisfy. */
#if defined(_MSC_VER) && _MSC_VER < 1600

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef short              int16_t;
typedef unsigned short     uint16_t;
typedef int                int32_t;
typedef unsigned int       uint32_t;
typedef __int64            lz_i64;
typedef unsigned __int64   lz_u64;
/* int64_t as well, and it was missing until a real VC++ 4.0 tried to
 * compile the tree. The `long long` sweep converted 213 sites to lz_i64
 * and stopped there; int64_t in a signature is the same type by another
 * spelling, and gcc has <stdint.h> so nothing here could see it. Four
 * uses in headers - ops.h twice, forward.h, ops_mmx.h - and 75 in .c,
 * every one of which failed with "syntax error : missing ')' before '*'"
 * because the type simply did not exist. */
typedef __int64            int64_t;
typedef unsigned __int64   uint64_t;

/* The printf length modifier for lz_i64. MSVC before 2013 does not
 * understand "ll"; it spells the same thing "I64". Call sites that
 * print a 64-bit counter use LZ_PRI64 rather than embedding either. */
#define LZ_PRI64 "I64"

/* A 64-bit unsigned constant. `1ULL` is a syntax error here - MSVC
 * before 2003 spells the suffix i64 and rejects ULL outright with
 * "bad suffix on number", which takes the whole expression with it.
 * Same shape as LZ_PRI64: one spelling at the call site, the toolchain
 * difference in this header. */
#define LZ_U64_C(x) x##ui64
#define LZ_I64_C(x) x##i64

#else

#include <stdint.h>
typedef long long          lz_i64;
typedef unsigned long long lz_u64;
#define LZ_PRI64 "ll"
#define LZ_U64_C(x) x##ULL
#define LZ_I64_C(x) x##LL

#endif /* _MSC_VER < 1600 */

/* `restrict` is C99 and 26 declarations across twelve files use it, so
 * a compiler without it does not merely lose the hint - it fails to
 * parse the declaration. OUTSIDE the branch above because the question
 * is the LANGUAGE version, not the vendor: gcc invoked as -std=c89 has
 * the same problem as a 1995 compiler, and gcc is the only one here
 * that can be asked.
 *
 * gcc keeps the hint through __restrict; anyone else loses it, which
 * costs nothing but the hint - restrict is an optimiser annotation, not
 * part of any interface's meaning. */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
/* C99 or later: restrict is a keyword. */
#elif defined(__GNUC__)
#define restrict __restrict
#else
#define restrict
#endif

/* For a `static` function defined in a header. Several translation
 * units include a header for one of its declarations and never call the
 * inline bodies beside it, and gcc warns once per function per TU: the
 * big-endian MIPS cross build alone reported 34 of them, which is a
 * build whose warning count carries no information. Watcom has no
 * equivalent spelling and does not warn, so the empty expansion there is
 * correct rather than merely tolerated.
 *
 * Here rather than in one of the headers that uses it: ops_quant.h and
 * ops_kernel_shared.h both need it, ops.c includes the second one
 * first, and two definitions of the same portability spelling is the
 * duplication this file exists to prevent. */
#if defined(__GNUC__)
#define LZ_MAYBE_UNUSED __attribute__((unused))
#else
#define LZ_MAYBE_UNUSED
#endif

#endif /* LZ_INT_H */
