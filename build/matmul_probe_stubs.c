/* Link stubs for the standalone matmul_row probe.
 *
 * build/matmul_row_probe.c drives lz_matmul_row_sse out of
 * src/ops_mmx_sse.c, and that TU also holds p2 row kernels which reach
 * p2_blk() and lz_p2_mul32_mmx() (both in src/ops_mmx.c). They are not on
 * the matmul_row path, so each aborts rather than returning something
 * plausible.
 */
#include <stdio.h>
#include <stdlib.h>

static void unreachable(const char *who) {
    fprintf(stderr, "matmul probe: %s was called; the probe does not "
                    "exercise it and its stub has no answer\n", who);
    exit(2);
}

void *p2_blk(void)                { unreachable("p2_blk"); return 0; }
void  lz_p2_mul32_mmx(void)       { unreachable("lz_p2_mul32_mmx"); }
