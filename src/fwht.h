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
#ifdef __cplusplus
}
#endif
#endif
