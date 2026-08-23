#ifndef LZ_INT_H
#define LZ_INT_H
/* One authority for the integer types, because iron law 10 put a 1995
 * compiler in the target family and it has neither of the two things
 * this tree reaches for by reflex.
 *
 * WHAT VISUAL C++ 4.0 DOES NOT HAVE, and the count of what that reaches
 * (grep over src/ on 2026-08-24, the day the law landed):
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

/* The printf length modifier for lz_i64. MSVC before 2013 does not
 * understand "ll"; it spells the same thing "I64". Call sites that
 * print a 64-bit counter use LZ_PRI64 rather than embedding either. */
#define LZ_PRI64 "I64"

#else

#include <stdint.h>
typedef long long          lz_i64;
typedef unsigned long long lz_u64;
#define LZ_PRI64 "ll"

#endif /* _MSC_VER < 1600 */

#endif /* LZ_INT_H */
