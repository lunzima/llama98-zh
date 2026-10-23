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

/* ROTATED VOCABULARY (spec 1.16, 3.17 item 17). The two halves of
   engine/bitlinear.py's ternary_embedding / ternary_head, in that order,
   so that a matrix stored as M = W R^T still computes the primal model:

     lookup  lz_embed_row_back   e <- S (H e)      (H first, then the
                                                    +-1 diagonal)
     head    lz_embed_rot_in     x <- H (S x)      (the diagonal, then H)

   R = H S / sqrt(block) is ORTHONORMAL, so the block transform carries
   1/sqrt(block): H H = I only with it, and the reconstruction
   M[id] R R^T = W[id] collapses only if it is there. The factor is the
   one thing this file cannot get from lz_fwht, which is unnormalized by
   design (ops.h's own reason: the orthonormal form needs a 1/sqrt(n)
   that is not a power of two for odd log2(n)). Both halves therefore do
   "lz_fwht, then scale each element", the scale being
   2^(-log2(n)/2) - an exact power of two when log2(n) is even, and that
   power of two times 1/sqrt(2) when it is odd.

   CONTRACT, checked by the loader (model.c): block is 0 or a power of
   two that divides dim, and a non-zero block or seed means the
   checkpoint's latent is already in the rotated basis (ce_carve). With
   block == 0 only the diagonal applies; with seed == 0 only the block
   applies; both zero means this is the plain vocabulary and neither
   function is called. */
void lz_embed_row_back(float *v, int dim, int block, int signs_seed);
void lz_embed_rot_in(float *v, int dim, int block, int signs_seed);

/* Is there a rotation at all? One spelling for every call site so the
   nine of them cannot drift into testing only one of the two words. */
#define LZ_EMBED_ROT_ON(block, seed) ((block) != 0 || (seed) != 0)
#ifdef __cplusplus
}
#endif
#endif
