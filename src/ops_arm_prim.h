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

/* ---- sigmoid_q15 / sigmoid_q15_t ------------------------------------- */

/* THE COORDINATE, ONCE. Both entry points end in the same three C
   lines - `idx = (int)t`, `frac = (int32_t)((t - (float)idx) *
   32768.0f)`, then a table pair and a linear step - and both are the
   same single integer: F = floor(t * 2^15), with idx = F >> 15 and
   frac = F & 32767.

   That identity is exact, not approximate. (float)idx is exact for
   idx <= 383; t - (float)idx is exact because t and idx are multiples
   of the same power of two and the difference is under 1, so it needs
   at most 23 significant bits; and the 32768.0f that follows is a
   power of two applied to a value that stays normal. Three float
   operations - two of them library calls on a target with no FPU -
   collapse into one shift.

   SMULBB, NOT MUL, for (b - a) * frac. sigmoid's derivative is at most
   1/4 and one table step is 1/16 of x, so |b - a| <= 32768/64 + 1 =
   513: both operands are int16 and the halfword multiply is the same
   product. That bound is a proof obligation about the TABLE, not
   something the code can test cheaply, so the gate sweeps every one of
   the 385 x 32768 (idx, frac) pairs against the C instead.

   NOT LDRD, deliberately, although g_sigtab[idx] and g_sigtab[idx+1]
   are adjacent int32s: LDRD needs an 8-byte-aligned address on
   ARMv5TE and idx is odd half the time. Declining half the domain to
   save one instruction is a bad trade, so this is two LDRs. */
#define LZ_SIG_ARM_LOOKUP                                     \
    "mov    %[e], %[F], lsr #15\n\t"                          \
    "sub    %[m], %[F], %[e], lsl #15\n\t"                    \
    "add    %[e], %[tb], %[e], lsl #2\n\t"                    \
    "ldr    %[r], [%[e]]\n\t"                                 \
    "ldr    %[e], [%[e], #4]\n\t"                             \
    "sub    %[e], %[e], %[r]\n\t"                             \
    "smulbb %[m], %[e], %[m]\n\t"                             \
    "add    %[r], %[r], %[m], asr #15\n\t"

/* sigmoid_q15_t(t): the table-only entry, so the coordinate arrives
   already scaled and there is no arithmetic in front of it at all.

   THE TWO CLAMPS BECOME ONE UNSIGNED COMPARE. Positive floats order
   the same way their bit patterns do, so `t <= 0.0f` and
   `t >= 384.0f` - two `bl __aeabi_fcmp*` calls, about 30 instructions
   each - are the single test u < 0x43C00000, which every negative
   value, every NaN and every t >= 384 fails because their bit patterns
   are larger. The out-of-window arm then answers both clamps from the
   sign bit, and declines only NaN.

   Declines on zero and subnormals: the implicit-bit reconstruction
   below is wrong for a zero exponent, and t = +0 is the C's `t <= 0`
   case rather than a table lookup. Both run the C. */
static __inline__ int lz_sig_q15_t_arm_asm(uint32_t ut, const int32_t *tab,
                                           int32_t *out) {
    uint32_t ok, e, m, F;
    int32_t r;
    __asm__(
        "mov    %[o], #0\n\t"
        "mov    %[r], #0\n\t"
        "cmp    %[u], %[c384]\n\t"
        "bhs    2f\n\t"
        "movs   %[e], %[u], lsr #23\n\t"
        "beq    1f\n\t"                  /* +0 or subnormal: declined */
        "mov    %[o], #1\n\t"
        "bic    %[m], %[u], #0xFF000000\n\t"
        "orr    %[m], %[m], #0x800000\n\t"
        "rsb    %[e], %[e], #135\n\t"    /* 0 .. 134 bits to drop */
        "mov    %[F], %[m], lsr %[e]\n\t"
        LZ_SIG_ARM_LOOKUP
        "b      1f\n\t"
        "2:\n\t"                         /* t >= 384, t <= 0, inf, nan */
        "mov    %[e], %[u], lsl #1\n\t"
        "cmp    %[e], #0xFF000000\n\t"
        "bhi    1f\n\t"                  /* nan: declined */
        "mov    %[o], #1\n\t"
        "cmp    %[u], #0\n\t"
        "ldrpl  %[r], [%[tb], #1536]\n\t"   /* t >= 384 -> tab[384] */
        "1:\n\t"
        : [o] "=&r"(ok), [r] "=&r"(r), [e] "=&r"(e), [m] "=&r"(m),
          [F] "=&r"(F)
        : [u] "r"(ut), [tb] "r"(tab), [c384] "r"(0x43C00000u)
        : "cc");
    *out = r;
    return (int)ok;
}

/* sigmoid_q15(x): the same table, reached through t = (x + 12) * 16.
   The multiply by 16 is an exact power of two and disappears into the
   coordinate; the ADD DOES NOT, and reproducing its rounding bit for
   bit is the whole content of this function.

   WHY IT IS AN INTEGER PROBLEM. What the table needs is
   F = floor(fl(x + 12) * 2^19), and fl(x + 12) is a correctly rounded
   float32 - i.e. the exact sum expressed in units of the RESULT's ulp
   and rounded half-to-even. The exact sum is 12 +- |x| with both terms
   dyadic, so in units of the result's ulp it is a 24-bit integer plus
   the bits of |x| that fall below that ulp, and the rounding decision
   is the standard "dropped bits, left-justified, against 2^31" the two
   primitives above already use. The four cases are the four places the
   result's binade can be:

     x >= 0, |x| >= 4      sum in [16,24), ulp 2^-19, 12 -> 0x600000
     x >= 0, |x| <  4      sum in [12,16), ulp 2^-20, 12 -> 0xC00000
     x <  0, |x| in [4,8)  sum in (4,8],   ulp 2^-21, EXACT
     x <  0, |x| >= 8      sum in (0,4],   ulp 2^-20, EXACT
     x <  0, |x| <  4      sum in (8,12],  ulp 2^-20, 12 -> 0xC00000

   The two exact rows are Sterbenz cancellation: no bits of x fall off,
   so no rounding happens and the binade of the result does not matter -
   only the scale the coordinate is read at does. The subtracting rows
   round the OTHER WAY round, because dropping bits off the subtrahend
   moves the result up: a tie there rounds to even by clearing the low
   bit rather than adding it.

   ZERO AND SUBNORMALS NEED NO CASE. Their exponent makes the shift
   count 130, so the quotient and the dropped bits are both zero on
   ARM's register-specified shifts, and the arm falls out at 12 exactly
   - which is what fl(x + 12) is for every |x| < 2^-126.

   THE ONE DECLINE IS NaN. Both clamps are answered from the sign bit
   once |x| >= 12, so infinities are handled too.

   idx CAN REACH 384 even though |x| < 12, which is why the lookup is
   guarded here and not in sigmoid_q15_t: x = 12 - 2^-20 makes the
   exact sum 24 - 2^-20, a tie at ulp 2^-19 that rounds up to 24, and
   the C's own `idx >= LZ_SIG_N` test is what catches it. */
static __inline__ int lz_sig_q15_arm_asm(uint32_t ux, const int32_t *tab,
                                         int32_t *out) {
    uint32_t ok, e, m, T, F;
    int32_t r;
    __asm__(
        "mov    %[o], #0\n\t"
        "mov    %[r], #0\n\t"
        "bic    %[e], %[u], #0x80000000\n\t"
        "cmp    %[e], %[c12]\n\t"
        "bhs    2f\n\t"                  /* |x| >= 12, inf, nan */
        "mov    %[o], #1\n\t"
        "mov    %[m], %[e], lsr #23\n\t"       /* biased exponent, 0..130 */
        "bic    %[T], %[e], #0xFF000000\n\t"
        "orr    %[T], %[T], #0x800000\n\t"     /* mantissa, implicit bit in */
        "cmp    %[u], #0\n\t"
        "bmi    3f\n\t"
        "cmp    %[m], #129\n\t"                /* x >= 0 */
        "blo    4f\n\t"
        "mov    %[F], #0x600000\n\t"           /* 12 in units of 2^-19 */
        "rsb    %[m], %[m], #131\n\t"
        "mov    %[e], #0\n\t"
        "b      5f\n\t"
        "4:\n\t"
        "mov    %[F], #0xC00000\n\t"           /* 12 in units of 2^-20 */
        "rsb    %[m], %[m], #130\n\t"
        "mov    %[e], #1\n\t"
        "5:\n\t"
        "add    %[F], %[F], %[T], lsr %[m]\n\t"
        "rsb    %[m], %[m], #32\n\t"
        "mov    %[T], %[T], lsl %[m]\n\t"      /* dropped bits, left-just. */
        "cmp    %[T], #0x80000000\n\t"
        "addhi  %[F], %[F], #1\n\t"
        "andeq  %[T], %[F], #1\n\t"
        "addeq  %[F], %[F], %[T]\n\t"
        "mov    %[F], %[F], lsr %[e]\n\t"
        "b      6f\n\t"
        "3:\n\t"                               /* x < 0 */
        "cmp    %[m], #129\n\t"
        "blo    7f\n\t"
        "bhi    8f\n\t"
        "rsb    %[F], %[T], #0x1800000\n\t"    /* |x| in [4,8): exact, 2^-21 */
        "mov    %[F], %[F], lsr #2\n\t"
        "b      6f\n\t"
        "8:\n\t"
        "rsb    %[F], %[T], #0xC00000\n\t"     /* |x| in [8,12): exact, 2^-20 */
        "mov    %[F], %[F], lsr #1\n\t"
        "b      6f\n\t"
        "7:\n\t"
        "rsb    %[m], %[m], #130\n\t"
        "mov    %[F], %[T], lsr %[m]\n\t"
        "rsb    %[F], %[F], #0xC00000\n\t"
        "rsb    %[m], %[m], #32\n\t"
        "mov    %[T], %[T], lsl %[m]\n\t"
        "cmp    %[T], #0x80000000\n\t"
        "subhi  %[F], %[F], #1\n\t"            /* dropped bits push it down */
        /* NO TIE INSTRUCTION HERE, and that is not an omission. Round-
           half-to-even on this branch picks between T-1 and T, and the
           `lsr #1` below discards exactly the bit they differ in:
           (T-1)>>1 == T>>1 for odd T, and T is the answer when T is
           even. The mutation harness found this by leaving a `bic ..,
           #1` green - it was dead. The ADD branch above is different,
           because there the tie moves T upward, which does survive the
           shift. */
        "mov    %[F], %[F], lsr #1\n\t"
        "6:\n\t"
        "cmp    %[F], #0xC00000\n\t"
        "bhs    9f\n\t"                        /* idx >= 384 */
        LZ_SIG_ARM_LOOKUP
        "b      1f\n\t"
        "9:\n\t"
        /* The C's own `idx >= LZ_SIG_N` return. It cannot change the
           VALUE - frac is zero whenever F reaches 0xC00000, so the
           interpolation would return tab[384] anyway - it is here so
           the lookup never reads tab[385], which is off the end of the
           array. A bounds guard, not an arithmetic one, so no mutation
           of it can be red. */
        "ldr    %[r], [%[tb], #1536]\n\t"
        "b      1f\n\t"
        "2:\n\t"
        "mov    %[m], %[u], lsl #1\n\t"
        "cmp    %[m], #0xFF000000\n\t"
        "bhi    1f\n\t"                        /* nan: declined */
        "mov    %[o], #1\n\t"
        "cmp    %[u], #0\n\t"
        "movpl  %[r], #0x8000\n\t"             /* x >= 12 -> 32768 */
        "1:\n\t"
        : [o] "=&r"(ok), [r] "=&r"(r), [e] "=&r"(e), [m] "=&r"(m),
          [T] "=&r"(T), [F] "=&r"(F)
        : [u] "r"(ux), [tb] "r"(tab), [c12] "r"(0x41400000u)
        : "cc");
    *out = r;
    return (int)ok;
}

#endif /* __arm__ && __GNUC__ */

#endif /* LZ_OPS_ARM_PRIM_H */
