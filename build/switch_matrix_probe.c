/* Control-switch matrix probe (build/ci.sh stage 5).

   Compiled by build/switch_matrix_gate.sh with EVERY engine control
   switch at its non-default value. wcc386 has no macro-dump option the
   way gcc has -dM, so this probe is the Watcom-side proof that the
   -D flags reached the preprocessor: each #error below fires unless the
   flag is spelled right AND not overridden by a header default, and a
   #error is a nonzero wcc386 exit. A flag that is misspelled or ignored
   compiles clean and proves nothing - this file exists to make that
   case fail.

   Deliberately self-contained: no #include, so it has no dependency on
   any engine header being correct at this moment - it asserts only what
   the build flags are supposed to be. The expected values mirror
   build/switch_matrix_gate.sh's SWITCHES table; keep the two in sync.

   The LZ_BATCH_MAX entry is the documented "batching off" PAIR
   (-DLZ_BATCH_MAX=1 -DLZ_BATCH_DEFAULT=1): ops.h's guard rejects the
   lone flag, and the pair is what the gate builds. */
#ifndef LZ_GDN_FIXED
#error "LZ_GDN_FIXED not defined (expected 0)"
#endif
#if LZ_GDN_FIXED != 0
#error "LZ_GDN_FIXED != 0"
#endif

#ifndef LZ_GDN_STATE_2PLANE
#error "LZ_GDN_STATE_2PLANE not defined (expected 0)"
#endif
#if LZ_GDN_STATE_2PLANE != 0
#error "LZ_GDN_STATE_2PLANE != 0"
#endif

#ifndef LZ_ATTN_FIXED
#error "LZ_ATTN_FIXED not defined (expected 0)"
#endif
#if LZ_ATTN_FIXED != 0
#error "LZ_ATTN_FIXED != 0"
#endif

#ifndef LZ_CONV_FIXED
#error "LZ_CONV_FIXED not defined (expected 0)"
#endif
#if LZ_CONV_FIXED != 0
#error "LZ_CONV_FIXED != 0"
#endif

#ifndef LZ_NORM_FIXED
#error "LZ_NORM_FIXED not defined (expected 0)"
#endif
#if LZ_NORM_FIXED != 0
#error "LZ_NORM_FIXED != 0"
#endif

#ifndef LZ_SSE2_GROUP
#error "LZ_SSE2_GROUP not defined (expected 0)"
#endif
#if LZ_SSE2_GROUP != 0
#error "LZ_SSE2_GROUP != 0"
#endif

#ifndef LZ_Q8_MIN_SCALE
#error "LZ_Q8_MIN_SCALE not defined (expected 0.0f)"
#endif
/* #if cannot compare floats (C99: no floating constant in a preprocessor
   expression - both gcc and Watcom reject it), so assert the value in a
   C constant expression instead: array bound goes negative unless the
   macro is exactly 0.0f. */
typedef char lz_q8min_check[1 - 2 * (int)!(LZ_Q8_MIN_SCALE == 0.0f)];

#ifndef LZ_EXACT_MATH
#error "LZ_EXACT_MATH not defined (expected 1)"
#endif
#if LZ_EXACT_MATH != 1
#error "LZ_EXACT_MATH != 1"
#endif

#ifndef LZ_CPUID_NOCACHE
#error "LZ_CPUID_NOCACHE not defined (expected 1)"
#endif
#if LZ_CPUID_NOCACHE != 1
#error "LZ_CPUID_NOCACHE != 1"
#endif

#ifndef LZ_NO_FPU
#error "LZ_NO_FPU not defined (expected 1)"
#endif
#if LZ_NO_FPU != 1
#error "LZ_NO_FPU != 1"
#endif

#ifndef LZ_KV_2PLANE
#error "LZ_KV_2PLANE not defined (expected 1)"
#endif
#if LZ_KV_2PLANE != 1
#error "LZ_KV_2PLANE != 1"
#endif

#ifndef LZ_GDN_STATE_F32
#error "LZ_GDN_STATE_F32 not defined (expected 1)"
#endif
#if LZ_GDN_STATE_F32 != 1
#error "LZ_GDN_STATE_F32 != 1"
#endif

#ifndef LZ_BATCH_MAX
#error "LZ_BATCH_MAX not defined (expected 1)"
#endif
#if LZ_BATCH_MAX != 1
#error "LZ_BATCH_MAX != 1"
#endif

#ifndef LZ_BATCH_DEFAULT
#error "LZ_BATCH_DEFAULT not defined (expected 1)"
#endif
#if LZ_BATCH_DEFAULT != 1
#error "LZ_BATCH_DEFAULT != 1"
#endif

#ifndef LZ_SPEC_K
#error "LZ_SPEC_K not defined (expected 1)"
#endif
#if LZ_SPEC_K != 1
#error "LZ_SPEC_K != 1"
#endif

#ifndef LZ_PF_DIST
#error "LZ_PF_DIST not defined (expected 0)"
#endif
#if LZ_PF_DIST != 0
#error "LZ_PF_DIST != 0"
#endif

#ifndef LZ_Q8R_FORCE_SSE
#error "LZ_Q8R_FORCE_SSE not defined (expected 1)"
#endif
#if LZ_Q8R_FORCE_SSE != 1
#error "LZ_Q8R_FORCE_SSE != 1"
#endif

#ifndef LZ_P2_FORCE_MMX
#error "LZ_P2_FORCE_MMX not defined (expected 1)"
#endif
#if LZ_P2_FORCE_MMX != 1
#error "LZ_P2_FORCE_MMX != 1"
#endif

/* The five int-pipeline int16 exits. The per-entry Watcom probe
   check_one generates already asserts each one alone; these are the
   canonical set's cross-check, which they were missing. */
#ifndef LZ_BVEC_I16
#error "LZ_BVEC_I16 not defined (expected 0)"
#endif
#if LZ_BVEC_I16 != 0
#error "LZ_BVEC_I16 != 0"
#endif

#ifndef LZ_MOEGATE_I16
#error "LZ_MOEGATE_I16 not defined (expected 0)"
#endif
#if LZ_MOEGATE_I16 != 0
#error "LZ_MOEGATE_I16 != 0"
#endif

#ifndef LZ_KLAT_I16
#error "LZ_KLAT_I16 not defined (expected 0)"
#endif
#if LZ_KLAT_I16 != 0
#error "LZ_KLAT_I16 != 0"
#endif

#ifndef LZ_KGATE_I16
#error "LZ_KGATE_I16 not defined (expected 0)"
#endif
#if LZ_KGATE_I16 != 0
#error "LZ_KGATE_I16 != 0"
#endif

#ifndef LZ_MLAT_I16
#error "LZ_MLAT_I16 not defined (expected 0)"
#endif
#if LZ_MLAT_I16 != 0
#error "LZ_MLAT_I16 != 0"
#endif

#ifndef LZ_MLAT_REUSE
#error "LZ_MLAT_REUSE not defined (expected 0)"
#endif
#if LZ_MLAT_REUSE != 0
#error "LZ_MLAT_REUSE != 0"
#endif
