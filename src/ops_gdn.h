/* ops_gdn.h - declarations for ops_gdn.c functions.
   lz_gdn_step, lz_kda_step, lz_attn_score_q8, lz_attn_wsum_q8,
   lz_gdn_quantize_2p, lz_gdn_p2_impl are all public (declared in
   ops.h). The one declaration this header adds is the test hook below. */
#ifndef OPS_GDN_H
#define OPS_GDN_H

#include "ops.h"

/* Test hook: true iff THIS translation unit saw LZ_HAVE_Q8R_AVX2 when it
   compiled, i.e. iff ops_gdn.c's #include "ops_avx2.h" reached it.
   lz_gdn_quantize_2p's tier==3 arm is compiled out silently when it does
   not, and the tiers owe each other bit-identity, so a value comparison
   cannot tell the AVX2 kernel from the SSE2 fallback. Separate from
   lz_q8round32_avx2_compiled_in_quant (ops_quant.h) because the two
   dispatch TUs include the header independently and either can go
   missing alone. */
int lz_q8round32_avx2_compiled_in_gdn(void);

#endif /* OPS_GDN_H */
