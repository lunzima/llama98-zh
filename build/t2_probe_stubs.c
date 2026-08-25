/* Link stubs for the standalone T2 probes.
 *
 * build/dot128_t2_probe.c drives two kernels out of src/ops_mmx.c, and
 * that TU also holds row kernels which reach the scheduler and the
 * int32->float epilogue. Linking those in would drag most of the engine
 * behind them, so the probe defines them here instead.
 *
 * NONE OF THESE IS ON THE PATH UNDER TEST. lz_dot128_t2_asm and
 * lz_dot32_t2_asm call none of them - they are referenced by other
 * functions in the same object file, which the linker resolves whether
 * or not anything calls them. Each therefore aborts rather than
 * returning something plausible: a stub that quietly answered would let
 * the probe measure a path it did not mean to.
 */
#include <stdio.h>
#include <stdlib.h>
#include "lz_int.h"

static void unreachable(const char *who) {
    fprintf(stderr, "t2 probe: %s was called; the probe does not "
                    "exercise it and its stub has no answer\n", who);
    exit(2);
}

/* g_kernel is DATA, so it cannot abort on use - it gets a value that is
   no tier. wsum_pair_tier compares against it; if the probe ever
   reached that code the comparison would match nothing and fall to the
   scalar arm, which is visible as a wrong answer rather than as a
   plausible one. */
int g_kernel = -1;

void lz_wsum_pair_sse2(const int8_t *rowA, const int8_t *rowB,
                       const int16_t *coef, int32_t *acc32);
void lz_wsum_pair_sse2(const int8_t *rowA, const int8_t *rowB,
                       const int16_t *coef, int32_t *acc32) {
    (void)rowA; (void)rowB; (void)coef; (void)acc32;
    unreachable("lz_wsum_pair_sse2");
}

int  lz_prefetch_mode(void)  { unreachable("lz_prefetch_mode");  return 0; }
int  lz_pair_mode(void)      { unreachable("lz_pair_mode");      return 0; }
int  lz_i32facc_tier(void)   { unreachable("lz_i32facc_tier");   return 0; }

void lz_i32f_acc32_simd(float *accf, const int32_t *acc32) {
    (void)accf; (void)acc32; unreachable("lz_i32f_acc32_simd");
}
void lz_i32f_acc32_sse(float *accf, const int32_t *acc32) {
    (void)accf; (void)acc32; unreachable("lz_i32f_acc32_sse");
}
