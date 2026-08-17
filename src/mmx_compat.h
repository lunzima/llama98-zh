/* MMX intrinsics compatibility layer: dual style for gcc (_mm_*) and
   Open Watcom (_m_*). Watcom's _m_* functions take __m64* (the headers
   provide by-value macro wrappers), and __m64's arithmetic/type
   semantics differ from gcc - kernel code must use the non-nested
   "compute to a temp, then next step" style; this file only maps
   names. Under gcc this file is a no-op (_mm_* as-is). */
#ifndef LZ_MMX_COMPAT_H
#define LZ_MMX_COMPAT_H

#ifdef __WATCOMC__
#include <mmintrin.h>
#define _mm_unpacklo_pi8(a, b) _m_punpcklbw(a, b)
#define _mm_unpackhi_pi8(a, b) _m_punpckhbw(a, b)
#define _mm_madd_pi16(a, b)    _m_pmaddwd(a, b)
#define _mm_add_pi32(a, b)     _m_paddd(a, b)
#define _mm_srai_pi16(a, n)    _m_psrawi(a, n)
#define _mm_srli_si64(a, n)    _m_psrlqi(a, n)
#define _mm_setzero_si64()     _m_from_int(0)
#define _mm_cvtsi64_si32(s)    _m_to_int(s)
#define _mm_empty()            _m_empty()
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)
#include <mmintrin.h>
#else
/* Not an x86 target at all - ARMv5TE is the first one (the kunkun-CE
   device). There is no <mmintrin.h> to include, and nothing here needs
   one: every actual MMX kernel already sits behind #if defined(__MMX__)
   and does not exist on this build.
 *
 * _mm_empty() is the exception, and it is the reason this branch has to
 * exist rather than the header simply not being included. NINE of its
 * twelve call sites in ops.c are OUTSIDE any __MMX__ guard - they sit
 * in the architecture-neutral dispatch code, because on x86 the call is
 * free whether or not the MMX path ran, and <mmintrin.h> always
 * provided it. Cross-compiling for ARM turned them into implicit
 * declarations: a warning, not an error, and an undefined symbol only
 * at link time.
 *
 * Leaving MMX state is meaningless where there is no MMX state, so it
 * expands to nothing. Deliberately NOT a function - it has to vanish,
 * not become a call. */
#define _mm_empty()            ((void)0)
#endif

#endif /* LZ_MMX_COMPAT_H */
