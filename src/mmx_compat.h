/* MMX intrinsics compatibility layer: dual style for gcc (_mm_*) and
   Open Watcom (_m_*). Watcom's _m_* functions take __m64* (the headers
   provide by-value macro wrappers), and __m64's arithmetic/type
   semantics differ from gcc - kernel code must use the non-nested
   "compute to a temp, then next step" style; this file only maps
   names. Under gcc this file is a no-op (_mm_* as-is). */
#ifndef LZ_MMX_COMPAT_H
#define LZ_MMX_COMPAT_H

#ifdef __WATCOMC__
#if defined(__MMX__)
/* Kernel TU: -D__MMX__ is given only to the ISA-suffixed units, so this
   branch pulls the full intrinsics set. */
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
/* Non-kernel TU, low ISA floor: <mmintrin.h> will not compile here (its
   _m_* intrinsics are .586-class), but this unit still owes a real emms
   after calling into the suffixed TUs' MMX kernels. Emit the opcode as
   raw bytes - emms is 0F 77 - so the assembler never sees the mnemonic
   and the -4r/-3r ceiling cannot reject it. Same trick lz_pf_amd uses
   for prefetchw. No .586 directive: db needs none. */
extern void lz_mmx_empty_local(void);
#pragma aux lz_mmx_empty_local = "db 0x0F, 0x77" __parm [] __modify __exact []
#define _mm_empty()            lz_mmx_empty_local()
#endif
/* Gated on __MMX__, not on the architecture. A translation unit built
   without -mmmx must not pull this header in: <mmintrin.h> refuses to
   compile there, and more importantly a unit that never sees an MMX
   intrinsic is one the compiler will not hand MMX instructions of its
   own. On a 32-bit x87 target gcc and clang both use MMX for 8-byte
   block stores when -mmmx is on, and neither issues emms before the
   x87 code that follows - which fills the tag word and makes the next
   fld overflow to real indefinite. Only the MMX kernels' own unit gets
   -mmmx, and it owes the emms. */
#elif defined(__MMX__) && (defined(__i386__) || defined(__x86_64__) \
      || defined(_M_IX86))
#include <mmintrin.h>
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86)
/* x86, but THIS translation unit was not built with -mmmx - see
   src/ops_mmx.h. <mmintrin.h> refuses to compile without -mmmx, so
   there is no _mm_empty() to pull in here, yet this TU still needs a
   real emms: it calls into src/ops_mmx.c's kernels, which leave the MMX
   register file dirty (kernels don't emit emms, callers do),
   and this caller's -mmmx-free compilation is exactly why it must not
   rely on the intrinsic. GAS accepts the mnemonic regardless of the
   compiler's own -mmmx/-msse target flags - inline asm text is opaque
   to gcc's codegen choices, only real intrinsics are gated by them. */
#if defined(_MSC_VER)
/* NOTHING, and for the same reason the no-x86 arm below is nothing:
   this build has no MMX state to leave.
   The GNU form above is a syntax error here - "'__asm__' : undeclared
   identifier" and four more as the parser unwinds - and the MSVC
   spelling does not help, because Visual C++ 4.0 is a 1996 compiler and
   its inline assembler has no emms mnemonic at all: MMX shipped the
   following year. `__asm { emms }` gives "inline assembler syntax error
   in 'opcode'; found 'newline'" even on its own line, outside any
   macro.
   That is not a gap to work around. This arm is reached only when the
   translation unit was built without MMX, and the requirement for this
   compiler is the no-assembly build - there are no MMX kernels linked
   in to have dirtied the register file. The MIPS edition never reaches
   here at all: _M_IX86 is undefined, so it takes the no-x86 arm. */
#define _mm_empty()            ((void)0)
#else
#define _mm_empty()            __asm__ __volatile__("emms" ::: "memory")
#endif /* _MSC_VER */
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
