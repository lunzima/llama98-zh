/* ops_sched.h - declarations for ops_sched.c's functions that are
   called from other TUs (ops.c, ops_quant.c, forward.c).

   The tier selectors (lz_gdn_mode, lz_conv_mode, etc.) are already
   declared in ops.h. This header covers functions that need cross-TU
   visibility. */
#ifndef OPS_SCHED_H
#define OPS_SCHED_H

/* Q8 rounding tier selector (0=scalar, 1=SSE, 2=SSE2).
   Called by lz_quantize_q8 (ops.c) and lz_gdn_quantize_2p (ops_gdn.c). */
int lz_q8r_tier(void);

/* Attention wsum's chunk-fold tier (0=scalar, 1=SSE, 2=SSE2).
   Called by lz_wsum_group_mmx (ops_mmx.c) and lz_attn_wsum_q8's own
   walk (ops_gdn.c) - the two arms of the LZ_WSUM_MMX_EXTERN split. */
int lz_i32facc_tier(void);

/* Whether this build+machine has SIMD rounding (CPUID SSE).
   Called by lz_gdn_p2_mode, lz_epi_mode, etc. (all in ops_sched.c)
   and by lz_gdn_quantize_2p (ops_gdn.c, module 5). */
int q8r_have_simd(void);

/* Cached CPUID SSE bit. Non-static: p2_tier (ops.c) reads it. */
int lz_cpu_has_sse(void);

/* Token pairing flag. Non-static: the row kernels in ops.c read it. */
extern int g_pair;

/* Accessor for g_kernel (the selected ISA tier). lz_row_pick reads the
   tier through this rather than the raw global. */
int lz_kernel_sel(void);

#endif /* OPS_SCHED_H */
