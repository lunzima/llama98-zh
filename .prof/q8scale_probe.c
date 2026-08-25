/* Is q8_group_scale's `fast` flag ever false?
 *
 * fast = (*inv <= FLT_MAX), where *inv = 127/amax. After the MIN_SCALE
 * guard forces fast=1 for sg < 1e-30 (amax < 1.27e-28), the remaining
 * amax range has 127/amax <= 1e30 << FLT_MAX, so fast looks always 1.
 * This probe sweeps every float exponent and a spread of mantissas to
 * find a counterexample before the dead-path claim is made.
 */
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "ops.h"
#include "ops_quant.h"

int main(void) {
    long slow = 0, flush = 0, total = 0;
    uint32_t u;
    float worst_sg = 1e30f, worst_inv = 0.0f;
    /* all 256 exponent fields; mantissa spread per binade */
    for (int e = 0; e < 256; e++) {
        for (int m = 0; m < 4096; m++) {
            u = ((uint32_t)e << 23) | ((uint32_t)m << 11); /* 12 mantissa bits */
            for (int k = 0; k < 4; k++) {
                float x[32];
                union { float f; uint32_t u; } b;
                b.u = u + k;
                for (int j = 0; j < 32; j++) x[j] = b.f;
                float sg, inv;
                int fast = q8_group_scale(x, 32, &sg, &inv);
                total++;
                if (!fast) slow++;
                if (sg < 1.0e-30f) flush++;
                if (sg < worst_sg) worst_sg = sg;
                if (inv > worst_inv) worst_inv = inv;
            }
        }
    }
    printf("q8_group_scale: %ld calls, fast=0: %ld, flushed (sg<1e-30): %ld\n",
           total, slow, flush);
    printf("  worst_sg=%.3e worst_inv=%.3e\n", worst_sg, worst_inv);
    return slow ? 1 : 0;
}
