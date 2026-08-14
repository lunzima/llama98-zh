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
#else
#include <mmintrin.h>
#endif

#endif /* LZ_MMX_COMPAT_H */
