/* ops_norm.h - declarations for ops_norm.c functions. */
#ifndef OPS_NORM_H
#define OPS_NORM_H

#include "ops.h"

/* Test hooks: true iff the named helper's dispatch, under the currently
   selected kernel, resolves to the AVX2 body specifically - pointer
   identity, not a value comparison. See their definitions (ops_norm.c)
   for why. */
int lz_rmsnorm_out_picked_avx2(void);
int lz_vmax_picked_avx2(void);
int lz_vscale_picked_avx2(void);

#endif /* OPS_NORM_H */
