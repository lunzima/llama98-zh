/* ops_sched.h - declarations for ops_sched.c's functions that are
   called from other TUs (ops.c, ops_quant.c, forward.c).

   The tier selectors (lz_gdn_mode, lz_conv_mode, etc.) are already
   declared in ops.h. This header covers functions that were static
   in ops.c and need cross-TU visibility after the split. */
#ifndef OPS_SCHED_H
#define OPS_SCHED_H

/* Q8 rounding tier selector (0=scalar, 1=SSE, 2=SSE2).
   Called by lz_quantize_q8 (ops.c) and lz_gdn_quantize_2p (ops_gdn.c). */
int lz_q8r_tier(void);

/* Whether this build+machine has SIMD rounding (CPUID SSE).
   Called by lz_gdn_p2_mode, lz_epi_mode, etc. (all in ops_sched.c)
   and by lz_gdn_quantize_2p (ops_gdn.c, module 5). */
int q8r_have_simd(void);

/* Cached CPUID SSE bit. Non-static: p2_tier (ops.c) reads it. */
int lz_cpu_has_sse(void);

/* Token pairing flag. Non-static: the row kernels in ops.c read it. */
extern int g_pair;

#endif /* OPS_SCHED_H */
