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

/* Is `u` (a float's bit pattern) inside the unsigned window
   [lo, lo+span]? Floats compare in sign-magnitude order, so a range
   test on the bit pattern IS the float comparison - one subtract and
   one compare against the `bl __aeabi_fcmpXX` plus `bl __cmpsf2` gcc
   spends, about 30 instructions. The CALLER picks the window, and picks
   it so NaN (exponent 255, non-zero mantissa) falls outside: that is
   what keeps the answer identical rather than merely close. */
static __inline__ int lz_uwin_arm_asm(uint32_t u, uint32_t lo,
                                      uint32_t span) {
    uint32_t r;
    __asm__(
        "sub    %[r], %[u], %[lo]\n\t"
        "cmp    %[r], %[sp]\n\t"
        "movls  %[r], #1\n\t"
        "movhi  %[r], #0\n\t"
        : [r] "=&r"(r)
        : [u] "r"(u), [lo] "r"(lo), [sp] "r"(span)
        : "cc");
    return (int)r;
}

/* a * 2^k for a normal float a and a normal result: the exponent-field
   add __aeabi_fmul spends 28 instructions arriving at. Declines
   (returns 0) on a zero/subnormal/inf/nan operand and on any k that
   would leave the normal range - every case where the multiply is not
   simply an exponent add - so the fast path is exact by construction,
   multiplying by a power of two being exact in IEEE. */
static __inline__ int lz_scale2_arm_asm(uint32_t ua, int32_t k,
                                        uint32_t *out) {
    uint32_t ok, e, r;
    __asm__(
        "mov    %[o], #0\n\t"
        "mov    %[e], %[a], lsr #23\n\t"
        "ands   %[e], %[e], #255\n\t"
        "beq    1f\n\t"                  /* zero or subnormal */
        "cmp    %[e], #255\n\t"
        "beq    1f\n\t"                  /* inf or nan */
        "adds   %[e], %[e], %[k]\n\t"
        "ble    1f\n\t"                  /* result subnormal or zero */
        "cmp    %[e], #255\n\t"
        "bge    1f\n\t"                  /* result inf */
        "add    %[r], %[a], %[k], lsl #23\n\t"
        "mov    %[o], #1\n\t"
        "1:\n\t"
        : [o] "=&r"(ok), [e] "=&r"(e), [r] "=&r"(r)
        : [a] "r"(ua), [k] "r"(k)
        : "cc");
    *out = r;
    return (int)ok;
}

/* lz_exp_fixed's floor step. The C spends two soft-float calls on it -
   `y32 - 0.5f`, then the 1.5*2^23 magic add - to arrive at
   t = round-half-to-even(y32 - 0.5). Also hands back the signed
   mantissa and the shift, so the float is unpacked once for this and
   for the residue step below.

   THE DECLINE IS WHAT MAKES IT IDENTICAL. `y32 - 0.5f` is an exact
   float32 subtraction only while N = +-mantissa - 2^(sh-1) fits in 24
   bits; past that the subtraction rounds first and the magic add rounds
   again, and reproducing that double rounding is a float adder, not an
   ISA cell. The fast path is therefore the exponent range where sh is
   meaningful at all (|y32| in [1, 4096), i.e. |x| >= 0.0217) AND the
   explicit |N| <= 2^24 test; everything else, inf and NaN included,
   runs the C. */
static __inline__ int lz_exp_floor_arm_asm(uint32_t uy, int32_t *tout,
                                           int32_t *msout, int32_t *shout) {
    uint32_t ok, sh, t1, t2;
    int32_t ms, t;
    __asm__(
        "mov    %[t1], %[u], lsr #23\n\t"
        "and    %[t1], %[t1], #255\n\t"
        "sub    %[t1], %[t1], #127\n\t"
        "mov    %[o], #0\n\t"
        "cmp    %[t1], #11\n\t"
        "bhi    1f\n\t"                  /* |y32| < 1 or >= 4096, inf, nan */
        "rsb    %[sh], %[t1], #23\n\t"   /* 150 - biased exponent, 12..23 */
        "bic    %[m], %[u], #0xFF000000\n\t"
        "orr    %[m], %[m], #0x800000\n\t"
        "cmp    %[u], #0\n\t"
        "rsbmi  %[m], %[m], #0\n\t"      /* signed mantissa */
        "sub    %[t1], %[sh], #1\n\t"
        "mov    %[t2], #1\n\t"
        "mov    %[t2], %[t2], lsl %[t1]\n\t"        /* 2^(sh-1) */
        "sub    %[t], %[m], %[t2]\n\t"              /* N */
        "add    %[t1], %[t], #0x1000000\n\t"
        "cmp    %[t1], #0x2000000\n\t"
        "bhi    1f\n\t"                  /* |N| > 2^24: the subtract rounds */
        "mov    %[o], #1\n\t"
        "mov    %[t1], %[t], asr %[sh]\n\t"
        "sub    %[t], %[t], %[t1], lsl %[sh]\n\t"   /* remainder, >= 0 */
        "cmp    %[t], %[t2]\n\t"
        "addhi  %[t1], %[t1], #1\n\t"
        "andeq  %[t], %[t1], #1\n\t"
        "addeq  %[t1], %[t1], %[t]\n\t"
        "mov    %[t], %[t1]\n\t"
        "1:\n\t"
        : [o] "=&r"(ok), [sh] "=&r"(sh), [m] "=&r"(ms), [t] "=&r"(t),
          [t1] "=&r"(t1), [t2] "=&r"(t2)
        : [u] "r"(uy)
        : "cc");
    *tout = t;
    *msout = ms;
    *shout = (int32_t)sh;
    return (int)ok;
}

/* lz_exp_fixed's residue step: rq = q8_round((y32 - (float)t) * 2^20),
   which the C spends an __aeabi_i2f, an __aeabi_fsub, an __aeabi_fmul
   and q8_round's own magic add on - four calls for a quantity already
   sitting in the mantissa lz_exp_floor_arm_asm unpacked.

   y32 - (float)t is EXACT (both are multiples of 2^-sh and the result
   is at most 1 in magnitude), and so is the 2^20 that follows, so the
   only rounding left is the one to integer - done here on P, the
   residue in units of 2^-sh. P is formed modulo 2^32 because t << sh
   overflows int32 while P itself is in [0, 2^sh]. */
static __inline__ int32_t lz_exp_resid_arm_asm(int32_t ms, int32_t t,
                                               int32_t sh) {
    int32_t rq;
    uint32_t p, t1, t2;
    __asm__(
        "sub    %[p], %[m], %[t], lsl %[sh]\n\t"
        "rsbs   %[t1], %[sh], #20\n\t"
        "movpl  %[q], %[p], lsl %[t1]\n\t"
        "bpl    1f\n\t"
        "rsb    %[t1], %[t1], #0\n\t"    /* sh - 20, 1..3 */
        "mov    %[q], %[p], lsr %[t1]\n\t"
        "mov    %[t2], #1\n\t"
        "mov    %[t2], %[t2], lsl %[t1]\n\t"
        "sub    %[p], %[p], %[q], lsl %[t1]\n\t"    /* remainder */
        "cmp    %[t2], %[p], lsl #1\n\t"
        "addcc  %[q], %[q], #1\n\t"
        "andeq  %[p], %[q], #1\n\t"
        "addeq  %[q], %[q], %[p]\n\t"
        "1:\n\t"
        : [q] "=&r"(rq), [p] "=&r"(p), [t1] "=&r"(t1), [t2] "=&r"(t2)
        : [m] "r"(ms), [t] "r"(t), [sh] "r"(sh)
        : "cc");
    return rq;
}

#endif /* __arm__ && __GNUC__ */

#endif /* LZ_OPS_ARM_PRIM_H */
