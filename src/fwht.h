#ifndef LZ_FWHT_H
#define LZ_FWHT_H
#include "lz_int.h"   /* <stdint.h> is not on the language floor */
#ifdef __cplusplus
extern "C" {
#endif
/* Unnormalized Fast Walsh-Hadamard, n must be a power of 2.
   Output RMS is sqrt(n) x input RMS; the caller folds 1/sqrt(n) into the
   downstream activation-quantization scale so this stays pure add/sub.
   int32 accumulation: worst-case |y| = n*32767 ~= 1.68e7, inside int32. */
void lz_fwht_i32(int32_t *restrict y, const int32_t *restrict x, int n);

/* Wiring-proof for lz_fwht_i32's direct g_kernel==AVX2 branch (fwht.c):
   returns 1 iff LZ_FWHT_AVX2_EXTERN was defined when fwht.c compiled, 0
   otherwise. Same reasoning as ops_matmul.h's lz_epi_avx2_compiled_in -
   a value-only AVX2-vs-SSE2/ref comparison cannot distinguish a real
   AVX2 kernel from a silently-correct fallback, so callers must check
   this returns 1 before trusting any such comparison. */
int lz_fwht_avx2_compiled_in(void);
#ifdef __cplusplus
}
#endif
#endif
