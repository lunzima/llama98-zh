/* ARMv5TE hand-written primitives for the operators outside matmul.
   docs/arm-asm-audit.md is the per-operator audit these come from; the
   row kernels live in src/ops_arm.c and src/ops_t2_arm.c.

   WHY THESE ARE INLINE ASM AND NOT LEAF FUNCTIONS. lz_i32f is `static`
   in ops_kernel_shared.h and is inlined at every one of its call sites;
   q8_round's magic add is inlined into norm_ss_fixed. A leaf function
   here would add a call where there is none today, and the call alone
   costs more than some of these bodies. Each one is therefore an
   __asm__ block that expands where it is used, with numeric local
   labels so several expansions can coexist in one function.

   WHAT MAKES THEM REACHABLE ONLY FROM ASSEMBLY. The target has no FPU,
   so gcc turns every float operation into a `bl __aeabi_*`: 105
   instructions for lz_i32f's four calls, 48 for q8_round's one. The
   values these compute are integer facts about the bit pattern - a
   correctly-rounded int32->float32 conversion and a round-half-to-even
   to integer - which CLZ, the barrel shifter and conditional execution
   do in a dozen instructions with no call at all. gcc cannot get there
   from the C, because the C says "float add" and it has to honour that.

   EVERY ONE OWES ITS C TIER BIT-IDENTITY, not closeness. Where a case
   is not provably identical the primitive DECLINES (returns 0) and the
   caller runs the C, so the tier can only ever be slower, never
   different. The declined cases are named at each function. */
#ifndef LZ_OPS_ARM_PRIM_H
#define LZ_OPS_ARM_PRIM_H

#include <stdint.h>

#if defined(__arm__) && defined(__GNUC__)
#define LZ_ARM_PRIM_ASM 1

/* int32 -> float32, correctly rounded (round-half-to-even), which is
   exactly what ops_kernel_shared.h's lz_i32f computes: its two-part
   split exists to pin down x87-vs-SSE rounding TIMING, and on a machine
   with one conversion semantics the split is provably the same number
   (see that function's comment). Exact for every int32, INT32_MIN
   included, so there is no declined case.

   CLZ gives the normalizing shift in one instruction; the exponent is
   then an add, and the carry out of a rounded-up mantissa lands in the
   exponent field on its own because the implicit bit is added in rather
   than masked off. */
static __inline__ float lz_i32f_arm_asm(int32_t v) {
    union { float f; uint32_t u; } b;
    uint32_t res, a, n, sh;
    __asm__(
        "movs   %[a], %[v]\n\t"
        "moveq  %[r], #0\n\t"
        "beq    1f\n\t"
        "rsbmi  %[a], %[a], #0\n\t"      /* a = |v|, 2^31 stays 2^31 */
        "clz    %[n], %[a]\n\t"
        "rsb    %[r], %[n], #157\n\t"    /* biased exponent - 1 */
        "mov    %[r], %[r], lsl #23\n\t"
        "subs   %[s], %[n], #8\n\t"
        "addpl  %[r], %[r], %[a], lsl %[s]\n\t"
        "bpl    2f\n\t"
        "rsb    %[s], %[n], #8\n\t"      /* 1..8 bits to drop */
        "add    %[r], %[r], %[a], lsr %[s]\n\t"
        "rsb    %[n], %[s], #32\n\t"
        "mov    %[n], %[a], lsl %[n]\n\t"    /* dropped bits, left-justified */
        "cmp    %[n], #0x80000000\n\t"
        "addhi  %[r], %[r], #1\n\t"
        "andeq  %[n], %[r], #1\n\t"
        "addeq  %[r], %[r], %[n]\n\t"
        "2:\n\t"
        "cmp    %[v], #0\n\t"
        "orrlt  %[r], %[r], #0x80000000\n\t"
        "1:\n\t"
        : [r] "=&r"(res), [a] "=&r"(a), [n] "=&r"(n), [s] "=&r"(sh)
        : [v] "r"(v)
        : "cc");
    b.u = res;
    return b.f;
}

/* q8_round(qv * 2^k) without the magic add and without the multiply.
   Returns 0 when it declines, which is q8_round's own stated domain
   boundary |qv * 2^k| < 2^22 (inf and NaN land there too, since their
   biased exponent is 255) - plus one case the stated boundary misses.

   THE EXTRA DECLINE. The magic add is only readable while the sum
   stays inside [2^23, 2^24), and the sum reaches 2^24 exactly when the
   rounded result is +2^22: qv = 4194303.5 rounds half-to-even up to
   4194304, the sum lands on 16777216, the low 23 bits are zero and
   q8_round returns -4194304. That input satisfies |qv| < 2^22, so the
   C is wrong there and the assembly is right - which makes them differ,
   which is what matters here. Declining on result == +2^22 hands those
   two floats (4194303.5 and 4194303.75) back to the C. Found by
   .prof/arm_leafgate.c's half-integer sweep.

   Scaling by a power of two is exact in IEEE, so folding the multiply
   into the exponent read is not an approximation - it is the same
   number as `q8_round(x * 2^k)` computes, minus one __aeabi_fmul. k=0
   is the plain q8_round. */
static __inline__ int lz_q8_round_pow2_arm_asm(uint32_t u, int32_t k,
                                               int32_t *out) {
    uint32_t ok, e, m, t;
    int32_t r;
    __asm__(
        "mov    %[e], %[u], lsr #23\n\t"
        "and    %[e], %[e], #255\n\t"
        "add    %[e], %[e], %[k]\n\t"
        "mov    %[o], #0\n\t"
        "mov    %[r], #0\n\t"
        "cmp    %[e], #149\n\t"
        "bhs    1f\n\t"                  /* |value| >= 2^22, or inf/nan */
        "mov    %[o], #1\n\t"
        "rsb    %[e], %[e], #150\n\t"    /* bits to drop, >= 2 */
        "cmp    %[e], #25\n\t"
        "bhs    1f\n\t"                  /* |value| < 0.5 -> 0 */
        "bic    %[m], %[u], #0xFF000000\n\t"
        "orr    %[m], %[m], #0x800000\n\t"
        "mov    %[r], %[m], lsr %[e]\n\t"
        "rsb    %[t], %[e], #32\n\t"
        "mov    %[t], %[m], lsl %[t]\n\t"
        "cmp    %[t], #0x80000000\n\t"
        "addhi  %[r], %[r], #1\n\t"
        "andeq  %[t], %[r], #1\n\t"
        "addeq  %[r], %[r], %[t]\n\t"
        "cmp    %[u], #0\n\t"
        "rsbmi  %[r], %[r], #0\n\t"
        "cmp    %[r], #0x400000\n\t"
        "moveq  %[o], #0\n\t"            /* sum would reach 2^24 */
        "1:\n\t"
        : [r] "=&r"(r), [o] "=&r"(ok), [e] "=&r"(e), [m] "=&r"(m),
          [t] "=&r"(t)
        : [u] "r"(u), [k] "r"(k)
        : "cc");
    *out = r;
    return (int)ok;
}

#endif /* __arm__ && __GNUC__ */

#endif /* LZ_OPS_ARM_PRIM_H */
