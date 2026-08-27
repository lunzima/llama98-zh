/* float transcendentals (lz_sinf/lz_cosf/lz_powf/lz_logf/lz_expf).
   powf/logf/expf ported verbatim from musl libm (MIT / SunPro / FreeBSD
   msun); sinf/cosf ported verbatim from glibc float s_sinf/s_cosf
   (__sincosf_table + reduce_fast/reduce_large). The musl/glibc bodies
   use double_t/double internals; THIS FILE CONVERTS ALL OF THEM TO
   FLOAT, so the object pulls no double soft-float on the ARMv5TE build
   and x86-64 (SSE) and ARM (soft-float) execute the same 24-bit float
   operation sequence -> bit-identical output by construction.

   The engine's "no double" law governs the inference DATA path; this
   transcendental layer is the counterpart of libm/libgcc, which the
   engine already links for rope table generation. Embedding it here
   makes the SAME code compile on x86 and ARM, so rope tables are
   bit-identical by construction rather than by two libms coinciding.

   Accuracy (float-only, ARM soft-float vs glibc libm, qemu-arm):
   lz_logf max 2 ulp; lz_expf max ~60 ulp; lz_powf max ~230 ulp
   (~2^-18 to 2^-15 relative); lz_sinf/lz_cosf see the reduce_fast
   float caveat in the .c. musl's double-internal versions were ~1 ulp;
   the float conversion trades that headroom for a double-free build. */
#ifndef LZ_MATHF_H
#define LZ_MATHF_H

#include "lz_int.h"

#if defined(__GNUC__) || defined(__WATCOMC__)

float lz_sinf(float x);
float lz_cosf(float x);
float lz_powf(float x, float y);
float lz_logf(float x);
float lz_expf(float x);

#endif /* __GNUC__ || __WATCOMC__ */
#endif /* LZ_MATHF_H */
