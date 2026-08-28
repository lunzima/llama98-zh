/* Emit src/lz_bf16.c's answers for a deterministic case set, for
   build/bf16_torch_gate.sh to check against PyTorch's.
 *
 * WHY A THIRD OPINION. lz_bf16.c is already compared against its own C
 * core (its self-test) and against the compiler's __bf16
 * (bf16_native_probe.c). Both of those live on this machine and share a
 * toolchain; PyTorch's c10::BFloat16 does not, and it is the
 * implementation the rest of the world actually runs. Where all three
 * agree, the format is not being misread in some way local to gcc.
 *
 * Writes a flat binary of records so the comparison needs no parsing:
 *   uint16 a, uint16 b, uint16 add, uint16 sub, uint16 mul, uint16 div
 * little-endian, which is what both sides of this pair run on. NaN
 * results are left as-is; the checker compares them by class, since
 * IEEE leaves a generated NaN's payload to the implementation and bf16
 * is not even an IEEE format. */
#include <stdio.h>
#include "lz_int.h"
#include "lz_bf16.h"

static uint32_t rnd(void) {
    static uint32_t s = 0x5eed1234u;
    s = s * 1664525u + 1013904223u;
    return s;
}

static void put16(FILE *f, uint32_t v) {
    fputc((int)(v & 0xFFu), f);
    fputc((int)((v >> 8) & 0xFFu), f);
}

int main(int argc, char **argv) {
    /* The values every bf16 implementation gets wrong first, then a
       random sweep. Same shape as bf16_native_probe.c's set so a
       disagreement can be lined up against that gate's result. */
    static const uint32_t sv[] = {
        0x0000u, 0x8000u, 0x3F80u, 0xBF80u, 0x4000u, 0x3F00u, 0x4049u,
        0x0001u, 0x8001u, 0x007Fu, 0x0080u, 0x8080u, 0x7F7Fu, 0xFF7Fu,
        0x7F80u, 0xFF80u, 0x7FC0u, 0xFFC0u, 0x4B80u, 0x3F81u, 0x3F7Fu
    };
    const int nsv = (int)(sizeof sv / sizeof sv[0]);
    FILE *f;
    int i, j;
    if (argc < 2) { fprintf(stderr, "usage: %s out.bin\n", argv[0]); return 2; }
    f = fopen(argv[1], "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", argv[1]); return 2; }

    for (i = 0; i < nsv; i++)
        for (j = 0; j < nsv; j++) {
            uint32_t a = sv[i], b = sv[j];
            put16(f, a); put16(f, b);
            put16(f, lz_bf16_add(a, b));
            put16(f, lz_bf16_sub(a, b));
            put16(f, lz_bf16_mul(a, b));
            put16(f, lz_bf16_div(a, b));
        }
    for (i = 0; i < 200000; i++) {
        uint32_t a = rnd() >> 16, b = rnd() >> 16;
        put16(f, a); put16(f, b);
        put16(f, lz_bf16_add(a, b));
        put16(f, lz_bf16_sub(a, b));
        put16(f, lz_bf16_mul(a, b));
        put16(f, lz_bf16_div(a, b));
    }
    fclose(f);
    return 0;
}
