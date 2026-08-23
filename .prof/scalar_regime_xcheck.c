/* Do the scalar transcendentals give the same bits in all four float
 * regimes? SSE, MMX (same registers, different kernel tier), x87 and
 * soft-float.
 *
 * WHY THESE AND WHY NOW. The per-kernel probes (fwht_f32, matmul_f32)
 * compare a SIMD body against the C beside it, in one binary - a
 * within-binary check. .prof/arm_leafgate.c compares the ARM assembly
 * against the ARM C, also within one binary. Nothing compared a scalar
 * transcendental's value across BUILDS, and that is where the one
 * regime difference this engine actually has lives: x87 evaluates in
 * 80-bit registers, so `a*b + c` can carry more precision than float32
 * has until it is stored. The tree already knows this - LZ_X87_FLOAT_CW
 * sets PC=24 for exactly this reason and ops.h's own note on
 * `static const float` is about the same hazard - but knowing is not
 * measuring, and no gate held the four builds' answers side by side.
 *
 * The answer is one FNV hash per function over a fixed sweep, printed
 * as text, so the comparison is between builds rather than inside one.
 * Same sweep on every regime: the driver takes no arguments and its
 * inputs are literals, so a difference in the hash is a difference in
 * the arithmetic and nothing else.
 *
 * SWEEP SHAPE, and it is not uniform on purpose: these functions are
 * table-plus-polynomial, so the interesting inputs are the table edges,
 * the ties between entries, and the boundaries of every declared band -
 * a uniform sample spends itself in the middle of buckets where any two
 * implementations agree trivially. Each row therefore walks its
 * function's own domain in units of that function's own step.
 *
 * Build, three regimes:
 *   gcc -O2 -std=c99 -Isrc -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 \
 *       -o /tmp/srx .prof/scalar_regime_xcheck.c src/ops.c src/ops_quant.c \
 *       src/ops_sched.c src/ops_norm.c src/ops_gdn.c src/ops_matmul.c \
 *       src/ops_t2_scalar.c src/ops_mmx.c src/ops_mmx_sse.c src/ops_sse2.c \
 *       src/fwht.c src/cpucheck.c src/err.c -lm
 *   arm-linux-gnueabi-gcc-14 -O2 -std=c99 -static -march=armv5te \
 *       -mfloat-abi=soft -DLZ_NO_FPU=1 -ffp-contract=off -Isrc ... (ARM
 *       source set, see .prof/scalar_regime_xcheck.sh)
 *   wcc386 - see the script; the Watcom arm is the x87 one and the whole
 *       reason this file exists.
 */
#include <stdio.h>
#include <stdint.h>

#include "ops.h"
#include "ops_quant.h"

static uint32_t h;
static void mix(float v) {
    union { float f; uint32_t u; } b;
    int i;
    b.f = v;
    for (i = 0; i < 4; i++) {
        h ^= (b.u >> (i * 8)) & 0xFFu;
        h *= 16777619u;
    }
}
static void mixi(int32_t v) {
    int i;
    for (i = 0; i < 4; i++) {
        h ^= (uint32_t)(v >> (i * 8)) & 0xFFu;
        h *= 16777619u;
    }
}
static void row(const char *name) {
    printf("SRX %-10s %08lx\n", name, (unsigned long)h);
    h = 2166136261u;
}

int main(void) {
    int i;
    unsigned fpu;
    /* PC=24, AND THE FIRST RUN OF THIS FILE PROVED WHY. Without it the
       Watcom arm reported exp, rsqrt and softplus as three cross-regime
       divergences - d0387522 / d5d34ede / 24adf125 against SSE's
       337c3dff / 287bfe39 / 8df797ad. All three were the probe: on x87
       the intermediates live in 80-bit registers, so the body is not
       computing in float32 at all, and lz_forward_chunk wraps the whole
       forward in this pair precisely so that it is. A probe outside the
       pair is measuring a machine the engine never runs on.
       matmul_f32_xcheck.c's header records the identical mistake being
       made and caught once already; making it again is what says the
       note belongs in every probe of this shape and not just that one.
       A no-op wherever the FPU is not x87, so the other three regimes
       read the same either way. */
    fpu = lz_fpu_float_begin();
    h = 2166136261u;

    /* lz_exp: the decay gate's arguments run about -40..12 and the table
       step is what the two tiers disagree about if anything does. */
    for (i = -40000; i <= 12000; i++) mix(lz_exp((float)i * 0.001f));
    row("exp");

    /* lz_sigmoid: the Q15 table is 384 entries over [-12, 12], so walk
       it at a sixteenth of a step and past both clamps. */
    for (i = -14000; i <= 14000; i++) mix(lz_sigmoid((float)i * 0.001f));
    row("sigmoid");

    /* sigmoid_q15's integer entry, every table index and a spread of
       fractions inside each - this is the one that is pure integer
       once entered, so a difference here is a difference in the
       COORDINATE, which is float. */
    for (i = 0; i < 385 * 16; i++)
        mixi(sigmoid_q15((float)(i - 12 * 16 * 16) * (1.0f / 256.0f)));
    row("sigq15");

    /* lz_rsqrt over the range norms present: ss + eps, always positive,
       spanning many binades. Walk exponents rather than a linear grid. */
    for (i = -60; i <= 60; i++) {
        int j;
        for (j = 1; j < 64; j++)
            mix(lz_rsqrt((float)j * 0.015625f * pow2f(i)));
    }
    row("rsqrt");

    /* lz_softplus: -40..16 is its measured argument range plus its own
       shortcut edge. */
    for (i = -40000; i <= 16000; i++) mix(lz_softplus((float)i * 0.001f));
    row("softplus");

    /* q8_round through a public entry that uses it. lz_quantize_q8 on a
       row whose amax makes the scale awkward is what exercises the
       magic add's tie behaviour. */
    {
        static float x[64];
        static int8_t q[64];
        static float s[2];
        int k;
        for (k = 0; k < 64; k++) x[k] = (float)(k - 32) * 0.5312500f + 0.25f;
        lz_quantize_q8(x, 64, 32, q, s);
        for (k = 0; k < 64; k++) mixi((int32_t)q[k]);
        mix(s[0]); mix(s[1]);
    }
    row("quant_q8");

    lz_fpu_float_end(fpu);
    return 0;
}
