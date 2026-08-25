/* Link stubs for the standalone wsum_pair probe.
 *
 * build/wsum_pair_probe.c drives lz_wsum_pair_mmx_w (src/ops_mmx.c) and
 * lz_wsum_pair_sse2 (src/ops_sse2.c). Both TUs also hold row kernels and
 * dispatch that reach the scheduler and the int32->float epilogue; these
 * stubs keep the link self-contained. lz_wsum_pair_sse2 and
 * lz_i32f_acc32_simd are NOT here - ops_sse2.c defines both, and a stub
 * would collide.
 *
 * NONE OF THESE IS ON THE PATH UNDER TEST, so each aborts rather than
 * returning something plausible.
 */
#include <stdio.h>
#include <stdlib.h>
#include "lz_int.h"

static void unreachable(const char *who) {
    fprintf(stderr, "wsum probe: %s was called; the probe does not "
                    "exercise it and its stub has no answer\n", who);
    exit(2);
}

int g_kernel = -1;

int  lz_prefetch_mode(void) { unreachable("lz_prefetch_mode"); return 0; }
int  lz_pair_mode(void)     { unreachable("lz_pair_mode");     return 0; }
int  lz_i32facc_tier(void)  { unreachable("lz_i32facc_tier");  return 0; }

void lz_i32f_acc32_sse(float *accf, const int32_t *acc32) {
    (void)accf; (void)acc32; unreachable("lz_i32f_acc32_sse");
}
