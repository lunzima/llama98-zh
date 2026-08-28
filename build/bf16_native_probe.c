/* Cross-check src/lz_bf16.c against the compiler's own __bf16.
   Built and run by build/bf16_native_gate.sh; see that script for why
   this is a debug-only artifact and never ships.

   WHAT MAKES THIS AN INDEPENDENT CHECK. lz_bf16.c's own test compares
   it against a reference written in the same file's idiom, which
   catches slips but shares any misunderstanding. __bf16 is a different
   implementation by different people: gcc lowers its arithmetic to f32
   and rounds back (verified: 0x3F81 * 0x3F81 gives 0x3F82 either way),
   and on a machine with AVX512-BF16 or AVX-NE-CONVERT it lowers the
   conversion to vcvtneps2bf16 instead - hardware. Either way it did not
   read our code.

   This is also why the SHIPPING build never uses __bf16: every target,
   x86 included, runs lz_bf16.c so that all of them share one
   arithmetic and stay byte-comparable. The compiler type is a witness,
   not a component. */
#include <stdio.h>
#include <string.h>
#include "lz_int.h"
#include "lz_bf16.h"

#if !defined(__GNUC__) || (__GNUC__ < 13)
int main(void) {
    printf("bf16-native: SKIP - __bf16 needs gcc 13 or newer\n");
    return 2;
}
#else

typedef __bf16 nat;

static uint32_t nat_bits(nat x) {
    union { nat b; uint16_t u; } v;
    v.b = x;
    return (uint32_t)v.u;
}
static nat nat_of(uint32_t h) {
    union { uint16_t u; nat b; } v;
    v.u = (uint16_t)h;
    return v.b;
}

static uint32_t s = 0x13579bdfu;
static uint32_t rnd(void) { s = s * 1664525u + 1013904223u; return s; }

static long bad[7], n[7];
static const char *opname[7] = { "add", "sub", "mul", "div", "cmp", "cvt",
                                 "high" };

/* A NaN result is a NaN result: the two implementations may legitimately
   choose different payloads, and IEEE 754 does not fix which. Compare
   the CLASS in that case, the bits otherwise.

   UNDER LZ_BF16_FAST THE BITS CANNOT MATCH AND MUST NOT BE ASKED TO.
   That knob replaces round-to-nearest-even with truncation toward zero,
   so disagreeing with __bf16 in the last place IS the feature. Demanding
   equality there left the knob with no check at all - a build that lost
   the low significand bits outright, or shifted it to the wrong place,
   failed the same way a correct one did, so nobody could tell them
   apart. What truncation still owes is the two properties that make it
   a rounding mode: never more than one ulp from the correctly rounded
   result, and never farther from zero than it. Both ends of the format
   are covered by that band as encoded - an overflow RNE sends to inf
   (0x7F80) truncates to the largest finite (0x7F7F), one apart. */
static int same(uint32_t x, uint32_t y) {
    int xn = ((x & 0x7F80u) == 0x7F80u) && (x & 0x7Fu);
    int yn = ((y & 0x7F80u) == 0x7F80u) && (y & 0x7Fu);
    if (xn || yn) return xn && yn;
    x &= 0xFFFFu; y &= 0xFFFFu;
    if (x == y) return 1;
#if defined(LZ_BF16_FAST) && LZ_BF16_FAST
    if ((x & 0x8000u) != (y & 0x8000u)) return 0;
    { uint32_t xm = x & 0x7FFFu, ym = y & 0x7FFFu;   /* x ours, y RNE */
      return ym >= xm && ym - xm <= 1u; }
#else
    return 0;
#endif
}

static void ck(uint32_t a, uint32_t b) {
    nat na = nat_of(a), nb = nat_of(b);
    uint32_t got[4], want[4];
    int i;
    got[0] = lz_bf16_add(a, b);   want[0] = nat_bits((nat)(na + nb));
    got[1] = lz_bf16_sub(a, b);   want[1] = nat_bits((nat)(na - nb));
    got[2] = lz_bf16_mul(a, b);   want[2] = nat_bits((nat)(na * nb));
    got[3] = lz_bf16_div(a, b);   want[3] = nat_bits((nat)(na / nb));
    for (i = 0; i < 4; i++) {
        n[i]++;
        if (!same(got[i], want[i])) {
            bad[i]++;
            if (bad[i] <= 4)
                printf("  %s %04x,%04x: ours %04x  __bf16 %04x\n",
                       opname[i], a & 0xFFFFu, b & 0xFFFFu,
                       got[i] & 0xFFFFu, want[i] & 0xFFFFu);
        }
    }
    /* compare: ours returns -1/0/1, and 2 for unordered */
    {
        int mine = lz_bf16_cmp(a, b), theirs;
        if (na != na || nb != nb) theirs = 2;
        else if (na < nb) theirs = -1;
        else if (na > nb) theirs = 1;
        else theirs = 0;
        n[4]++;
        if (mine != theirs) {
            bad[4]++;
            if (bad[4] <= 4)
                printf("  cmp %04x,%04x: ours %d  __bf16 %d\n",
                       a & 0xFFFFu, b & 0xFFFFu, mine, theirs);
        }
    }
}

int main(void) {
    static const uint32_t v[] = {
        0x0000u, 0x8000u, 0x3F80u, 0xBF80u, 0x4000u, 0x3F00u, 0x4049u,
        0x0001u, 0x8001u, 0x007Fu, 0x0080u, 0x8080u, 0x7F7Fu, 0xFF7Fu,
        0x7F80u, 0xFF80u, 0x7FC0u, 0xFFC0u, 0x4B80u, 0x3F81u, 0x3F7Fu
    };
    unsigned nv = (unsigned)(sizeof(v) / sizeof(v[0]));
    unsigned i, j;
    int t, fail = 0;

    for (i = 0; i < nv; i++)
        for (j = 0; j < nv; j++) ck(v[i], v[j]);
    for (i = 0; i < 1000000u; i++) {
        uint32_t a = rnd() >> 16, b = rnd() >> 16;
        if ((a & 31u) == 0) a = v[(a >> 5) % nv];
        if ((b & 31u) == 0) b = v[(b >> 5) % nv];
        ck(a, b);
    }
    /* f32 -> bf16 conversion, the rounding most easily got wrong */
    for (i = 0; i < 1000000u; i++) {
        union { uint32_t u; float f; } f;
        f.u = rnd();
        {
            uint32_t mine = lz_f32_to_bf16(f.u);
            uint32_t theirs = nat_bits((nat)f.f);
            n[5]++;
            if (!same(mine, theirs)) {
                bad[5]++;
                if (bad[5] <= 4)
                    printf("  cvt %08x: ours %04x  __bf16 %04x\n",
                           f.u, mine & 0xFFFFu, theirs & 0xFFFFu);
            }
        }
    }

    /* THE HIGH HALF OF AN ARGUMENT MUST NOT MATTER, and this is checked
       rather than assumed because it was once false. A bf16 lives in the
       low 16 bits of a word; the C core drops the rest (lz_bf16_to_f32
       masks with 0xFFFF) but the ARM assembly read the word whole, so
       the same call gave one answer when the fast path took it and
       another when it declined - 12.0M of 26.3M cases, invisible to the
       caller and decided by the operand values. That is the decline
       contract broken, not a documentation gap, so it gets an assertion.
       Cheap here: the two runs must agree, whatever the answer is. */
    for (i = 0; i < 200000u; i++) {
        uint32_t a = rnd() >> 16, b = rnd() >> 16;
        uint32_t dirt = rnd() & 0xFFFF0000u;
        uint32_t A = a | dirt, B = b | dirt;
        n[6]++;
        if (lz_bf16_add(a, b) != lz_bf16_add(A, B) ||
            lz_bf16_sub(a, b) != lz_bf16_sub(A, B) ||
            lz_bf16_mul(a, b) != lz_bf16_mul(A, B) ||
            lz_bf16_div(a, b) != lz_bf16_div(A, B)) {
            bad[6]++;
            if (bad[6] <= 4)
                printf("  high %04x,%04x with %08x above: answer moved\n",
                       a & 0xFFFFu, b & 0xFFFFu, dirt);
        }
    }

    for (t = 0; t < 7; t++) {
        printf("  %-4s %8ld / %8ld differ\n", opname[t], bad[t], n[t]);
        if (bad[t]) fail = 1;
    }
    if (fail) { printf("bf16-native: FAIL\n"); return 1; }
    printf("bf16-native: PASS (lz_bf16.c agrees with the compiler's "
           "__bf16 on every case)\n");
    return 0;
}

#endif /* __GNUC__ >= 13 */
