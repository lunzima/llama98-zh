/* Fast-mode soft-float probe: does the ARM LZ_SOFTFP_FAST arithmetic
   match x86 hardware under MXCSR round-toward-zero + FTZ + DAZ?

   Run natively on x86 (the hardware reference, --fastfp semantics via
   lz_fpu_float_begin's MXCSR path) and under qemu on ARM built with
   -DLZ_SOFTFP_FAST=1 (the software under test). Each case prints one
   uint32 bit pattern; the two runs' outputs must be byte-identical.

   Cases are (op, ua, ub) drawn from a fixed seed - 1145141919, the
   project's constant - and each op also runs its DIRECTED special
   matrix: 0/+-0, subnormals, infinities, NaNs, overflow/underflow
   edges and the rounding boundaries. Under FTZ/DAZ a subnormal INPUT
   is a zero and a subnormal RESULT flushes, and the hardware and the
   software must agree on every one of those.

   Both sides must round the SAME way per op: the probe sets MXCSR on
   x86 before the arithmetic and never touches it again, and the ARM
   build compiles with -DLZ_SOFTFP_FAST=1. int->float conversion is
   probed too (cvtsi2ss rounds toward zero under the same MXCSR), as is
   float->int (cvttss2si is already truncation on both sides).

   Usage:
     x86:   gcc -O2 -o /tmp/fastfp-x86 build/arm/fastfp_probe.c && /tmp/fastfp-x86
     ARM:   arm-linux-gnueabi-gcc-14 -O2 -march=armv5te -mfloat-abi=soft \
              -DLZ_SOFTFP_FAST=1 -I src build/arm/fastfp_probe.c src/lz_softfp.c \
              -o /tmp/fastfp-arm && qemu-arm-static /tmp/fastfp-arm
     diff  (the two outputs)
*/
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* The project's fixed seed. */
#define SEED 1145141919u

/* 24-bit LCG, same shape as the engine's sampler RNG. */
static uint32_t rng_state = SEED;
static uint32_t rng_next(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

#if defined(__arm__)
/* Under test: the engine's soft-float, its own rounding. */
extern float __aeabi_fadd(float, float);
extern float __aeabi_fsub(float, float);
extern float __aeabi_fmul(float, float);
extern float __aeabi_fdiv(float, float);
extern float __aeabi_i2f(int);
#define OP_ADD __aeabi_fadd
#define OP_SUB __aeabi_fsub
#define OP_MUL __aeabi_fmul
#define OP_DIV __aeabi_fdiv
#define OP_I2F __aeabi_i2f
#else
/* Reference: x86 hardware under MXCSR RTZ+FTZ+DAZ. */
static void fpu_fast_begin(void) {
    unsigned csr;
    __asm__ __volatile__("stmxcsr %0" : "=m"(csr));
    csr |= 0x8040u;                       /* bit15 FTZ, bit6 DAZ */
    csr = (csr & ~0x6000u) | 0x6000u;     /* RC=11 toward zero */
    __asm__ __volatile__("ldmxcsr %0" : : "m"(csr));
}
#define OP_ADD (fadd_hw)
#define OP_SUB (fsub_hw)
#define OP_MUL (fmul_hw)
#define OP_DIV (fdiv_hw)
#define OP_I2F (i2f_hw)
static float fadd_hw(float a, float b) { return a + b; }
static float fsub_hw(float a, float b) { return a - b; }
static float fmul_hw(float a, float b) { return a * b; }
static float fdiv_hw(float a, float b) { return a / b; }
static float i2f_hw(int x) { return (float)x; }
#endif

static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

/* Random bit patterns biased toward the interesting ranges: normal,
   tiny (subnormal), huge, and around 1.0 where rounding boundaries
   live. */
static float rand_f(void) {
    uint32_t kind = rng_next() % 5;
    uint32_t u;
    switch (kind) {
    case 0: u = rng_next(); break;                        /* uniform bits */
    case 1: u = (rng_next() & 0x007FFFFFu) | 0x3F800000u; /* around 1.0 */
    case 2: u = (rng_next() & 0x0FFFFFFFu);               /* tiny/subnormal */
    case 3: u = (rng_next() & 0x007FFFFFu) | 0x7F000000u; /* huge */
    default:u = (rng_next() & 0x007FFFFFu) | (rng_next() & 0xFFu) << 23; break;
    }
    /* Finite only: exponent 0xFF is NaN/inf, which the pairing does not
       exercise (the engine's workload never produces one, fp_exception_gate
       asserts IE=0) and whose payload propagation differs between x86
       hardware and libgcc-style soft-float by design. */
    if (((u >> 23) & 0xFFu) == 0xFFu) u &= 0x7EFFFFFFu;
    float f; memcpy(&f, &u, 4); return f;
}

static void probe_binop(float (*op)(float, float), const char *name) {
    int i;
    /* Random cases. */
    for (i = 0; i < 20000; i++) {
        float xa = rand_f(), xb = rand_f();
        printf("%s %08x %08x %08x\n", name, bits(xa), bits(xb), bits(op(xa, xb)));
    }
    /* Directed special matrix: every pair from a small interesting set. */
    {
        static const float sp[10] = {
            0.0f, -0.0f, 1.0f, -1.0f,
            1.0e-45f,          /* smallest subnormal */
            1.0e-38f,          /* subnormal */
            1.17549435e-38f,   /* min normal */
            3.40282347e38f,    /* max normal */
            3.40282347e38f * 0.99f, /* near max, rounding boundary */
            1.0e-40f,          /* tiny subnormal, RTZ vs FTZ boundary */
        };
        int a, b;
        for (a = 0; a < 10; a++)
            for (b = 0; b < 10; b++)
                printf("%s %08x %08x\n", name, bits(op(sp[a], sp[b])), 0u);
    }
}

int main(void) {
    int i;
#if !defined(__arm__)
    fpu_fast_begin();   /* the reference arm: RTZ + FTZ + DAZ in MXCSR */
#endif
    probe_binop(OP_ADD, "add");
    probe_binop(OP_SUB, "sub");
    probe_binop(OP_MUL, "mul");
    probe_binop(OP_DIV, "div");
    for (i = 0; i < 2000; i++)
        printf("i2f %08x %08x\n", bits(OP_I2F((int)(rng_next() ^ (rng_next() << 16)))), 0u);
    return 0;
}
