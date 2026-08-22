#include "fwht.h"
#include "ops_arm.h"
#if defined(LZ_ARM_ASM_EXTERN)
#include "ops.h"                  /* LZ_KERNEL_ARM_ASM */
#include "ops_kernel_shared.h"    /* g_kernel */
#endif /* LZ_ARM_ASM_EXTERN */

/* Sylvester recursion, in place, mirroring lz_fwht (float) in ops.c.
   Every stage is a butterfly of one add and one sub - no multiply,
   sqrt or divide in the transform body, so it is exact in int32 up to
   the additions themselves. The caller folds the 1/sqrt(n) energy
   factor into the downstream activation-quantization scale, keeping
   this a pure integer kernel for the ARMv5TE target (no FPU, no SIMD). */
void lz_fwht_i32(int32_t *restrict y, const int32_t *restrict x, int n) {
    int len, i, j;
    int32_t *tmp = y;          /* caller provides output; butterflies in place */
#if defined(LZ_ARM_ASM_EXTERN)
    /* Selected, not automatic: `--kernel arm-c` and `ref` must reach the
       C body below, or the tier this assembly is COMPARED AGAINST would
       be the assembly itself and build/arm/parity_gate.sh would be
       comparing a thing to itself. Same shape as lz_wsum_pair_arm and
       norm_ss_fixed's guard.

       n >= 8 is the assembly's precondition, not a preference: every
       loop there moves eight words and has no epilogue. Smaller n means
       blk 2 or 4, which no shipping config produces, so the C body stays
       the whole answer for those rather than growing a tail. */
    if (g_kernel == LZ_KERNEL_ARM_ASM && n >= 8) {
        lz_fwht_i32_arm_asm(y, x, n);
        return;
    }
#endif /* LZ_ARM_ASM_EXTERN */
    for (i = 0; i < n; i++) tmp[i] = x[i];
    for (len = 1; len < n; len <<= 1) {
        for (i = 0; i < n; i += (len << 1)) {
            for (j = 0; j < len; j++) {
                int32_t u = tmp[i + j];
                int32_t v = tmp[i + j + len];
                tmp[i + j]       = u + v;
                tmp[i + j + len] = u - v;
            }
        }
    }
}
