/* Link stubs for the standalone exp_q20 probe, SSE2 tier.
 *
 * build/exp_q20_probe.c drives lz_exp_q20_simd out of src/ops_sse2.c, and
 * that TU also holds row kernels which reach lz_prefetch_mode/lz_pair_mode.
 * They are not on the path under test - the Q20 Taylor calls neither - so
 * each aborts rather than returning something plausible.
 */
#include <stdio.h>
#include <stdlib.h>

static void unreachable(const char *who) {
    fprintf(stderr, "exp probe: %s was called; the probe does not "
                    "exercise it and its stub has no answer\n", who);
    exit(2);
}

int lz_prefetch_mode(void) { unreachable("lz_prefetch_mode"); return 0; }
int lz_pair_mode(void)     { unreachable("lz_pair_mode");     return 0; }
