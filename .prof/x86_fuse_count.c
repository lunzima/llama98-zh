/* The pass-1 fusion's x86 side: static instruction counts, both shapes.
 *
 * docs/int-pipeline-project.md 0.5 left one prerequisite unmeasured and
 * called it a wall-clock question. It is not, or not entirely: iron law
 * three says STATIC INSTRUCTION COUNTS TRANSFER across the target family
 * and QEMU wall clock does not - so the loop body's instruction count is
 * a legitimate second caliber here, the same one the whole ARM line
 * rests on. What cannot be answered from this machine is cycles and
 * cache behaviour, not instructions.
 *
 * The question is whether the `volatile float` intermediate - which the
 * fusion NEEDS, because without it gcc and a 32-bit x87 build round the
 * int64->float differently by 1 ULP (src/ops.c's `volatile float tamf`
 * carries the same trap and the same fix) - costs more on x86 than the
 * fusion saves there. x86 only saves 3 operations an element; ARM saves
 * 80 instructions, so the answer only matters on this side.
 *
 * SHAPES
 *   0  shipping:  (i2f(au) + i2f(aul) * 2^-8) * su
 *   1  fused:     (float)(((int64)au << 8) + aul) * sus     (no volatile)
 *   2  fused + volatile float intermediate                  (shippable)
 *
 * Shape 1 is not shippable and is here only so the volatile's cost can
 * be separated from the fusion's gain rather than reported as one
 * number.
 *
 * Not a runnable benchmark: build it, disassemble the loop, count. The
 * driver does that.
 */
#include <stdint.h>

#ifndef FUSE_SHAPE
#define FUSE_SHAPE 0
#endif

#define LO 256.0f

void fuse_loop(float *u, const int32_t *au, const int32_t *aul,
               float su, int n) {
#if FUSE_SHAPE == 2
    float sus = su * (1.0f / LO);
#elif FUSE_SHAPE == 1
    float sus = su * (1.0f / LO);
#endif
    int i;
    for (i = 0; i < n; i++) {
#if FUSE_SHAPE == 0
        u[i] = ((float)au[i] + (float)aul[i] * (1.0f / LO)) * su;
#elif FUSE_SHAPE == 1
        u[i] = (float)(((int64_t)au[i] << 8) + (int64_t)aul[i]) * sus;
#else
        {
            volatile float uf =
                (float)(((int64_t)au[i] << 8) + (int64_t)aul[i]);
            u[i] = uf * sus;
        }
#endif
    }
}
