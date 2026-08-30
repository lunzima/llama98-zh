/* ARMv5TE hand-written primitives for the operators outside matmul.
   docs/arm-asm-audit.md is the per-operator audit these come from; the
   row kernels live in src/ops_arm.c and src/ops_t2_arm.c.

   Why these are inline asm and not leaf functions. lz_i32f is `static`
   in ops_kernel_shared.h and is inlined at every one of its call sites;
   q8_round's magic add is inlined into norm_ss_fixed. A leaf function
   here would add a call where there is none today, and the call alone
   costs more than some of these bodies. Each one is therefore an
   __asm__ block that expands where it is used, with numeric local
   labels so several expansions can coexist in one function.

   What makes them reachable only from assembly. The target has no FPU,
   so gcc turns every float operation into a `bl __aeabi_*`: 105
   instructions for lz_i32f's four calls, 48 for q8_round's one. The
   values these compute are integer facts about the bit pattern - a
   correctly-rounded int32->float32 conversion and a round-half-to-even
   to integer - which CLZ, the barrel shifter and conditional execution
   do in a dozen instructions with no call at all. gcc cannot get there
   from the C, because the C says "float add" and it has to honour that.

   Every one owes its C tier bit-identity, not closeness. Where a case
   is not provably identical the primitive DECLINES (returns 0) and the
   caller runs the C, so the tier can only ever be slower, never
   different. The declined cases are named at each function. */
#ifndef LZ_OPS_ARM_PRIM_H
#define LZ_OPS_ARM_PRIM_H

#include "lz_int.h"   /* <stdint.h> is not on the language floor */

#if defined(__arm__) && defined(__GNUC__)
#define LZ_ARM_PRIM_ASM 1

/* LZ_SOFTFP_FAST is lz_softfp.c's round-toward-zero switch, but these
   macros also expand in fwht.c and ops_arm.c, so the default must be
   visible here for the RTZ tails below to read it. Same default as
   ops.h; reached before ops.h in some TUs, so it must not disagree. */
#ifndef LZ_SOFTFP_FAST
#if defined(LZ_NO_FPU) || defined(__arm__)
#define LZ_SOFTFP_FAST 1
#else
#define LZ_SOFTFP_FAST 0
#endif
#endif /* !LZ_SOFTFP_FAST */

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
#if LZ_SOFTFP_FAST
        /* Round toward zero: the dropped bits are dropped, no round-up
           and no tie-to-even - matching cvtsi2ss under x86's MXCSR
           RC=truncate. The exact branch above (|v| < 2^24) is unchanged:
           it has no rounding to do. */
#else
        "rsb    %[n], %[s], #32\n\t"
        "mov    %[n], %[a], lsl %[n]\n\t"    /* dropped bits, left-justified */
        "cmp    %[n], #0x80000000\n\t"
        "addhi  %[r], %[r], #1\n\t"
        "andeq  %[n], %[r], #1\n\t"
        "addeq  %[r], %[r], %[n]\n\t"
#endif /* LZ_SOFTFP_FAST */
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

   The extra decline. The magic add is only readable while the sum
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
#if LZ_SOFTFP_FAST
        /* Round toward zero: the dropped bits are dropped, no round-up
           and no tie-to-even - the LZ_Q8R_BITS/arm-asm rounding path
           was missed when the fast-mode pairing landed, so ARM's
           quantizer rounded half-to-even while x86's MXCSR-truncated
           magic add truncated. One ULP on a value whose dropped bits
           exceeded half, cascading through every downstream matmul. */
#else
        "rsb    %[t], %[e], #32\n\t"
        "mov    %[t], %[m], lsl %[t]\n\t"
        "cmp    %[t], #0x80000000\n\t"
        "addhi  %[r], %[r], #1\n\t"
        "andeq  %[t], %[r], #1\n\t"
        "addeq  %[r], %[r], %[t]\n\t"
#endif /* LZ_SOFTFP_FAST */
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

   The decline is what makes it identical. `y32 - 0.5f` is an exact
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
        /* |N| <= 2^24, or the subtract above rounded. UNREACHABLE on any
           input this primitive accepts: the entry test eight lines up
           bounds |y32| below 4096, so |N| stays far under 2^24. Kept as
           a guard on the arithmetic rather than as a live branch -
           .prof/arm_mutgate.sh's floor-exactness-test halves the window
           here and the sweep cannot tell, which is what "unreachable"
           looks like from the outside. Do not read that GREEN as the
           sweep being too narrow. */
        "add    %[t1], %[t], #0x1000000\n\t"
        "cmp    %[t1], #0x2000000\n\t"
        "bhi    1f\n\t"
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
#if LZ_SOFTFP_FAST
    /* Round toward zero: the residue p is non-negative, so a logical
       shift IS the truncation q8_round performs under fastfp (the C
       reference's q8_round now truncates; this RNE stayed behind, and
       the softmax's exp diverged from x86 on residues whose dropped bits
       exceeded half). */
    __asm__(
        "sub    %[p], %[m], %[t], lsl %[sh]\n\t"
        "rsbs   %[t1], %[sh], #20\n\t"
        "movpl  %[q], %[p], lsl %[t1]\n\t"
        "bpl    1f\n\t"
        "rsb    %[t1], %[t1], #0\n\t"    /* sh - 20, 1..3 */
        "mov    %[q], %[p], lsr %[t1]\n\t"
        "1:\n\t"
        : [q] "=&r"(rq), [p] "=&r"(p), [t1] "=&r"(t1)
        : [m] "r"(ms), [t] "r"(t), [sh] "r"(sh)
        : "cc");
#else
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
#endif /* LZ_SOFTFP_FAST */
    return rq;
}

/* ---- sigmoid_q15 / sigmoid_q15_t ------------------------------------- */

/* The coordinate, computed once. Both entry points end in the same three C
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

   The two clamps become one unsigned compare. Positive floats order
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

   Why it is an integer problem. What the table needs is
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

   Zero and subnormals need no case. Their exponent makes the shift
   count 130, so the quotient and the dropped bits are both zero on
   ARM's register-specified shifts, and the arm falls out at 12 exactly
   - which is what fl(x + 12) is for every |x| < 2^-126.

   The one decline is NaN. Both clamps are answered from the sign bit
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
        /* No tie instruction here; that is deliberate. Round-
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

/* ---- lz_fwht's butterfly: a+b and a-b from one unpack ---------------- */

/* gcc emits `bl __addsf3` and `bl __subsf3` here, 94.1 instructions
   measured (.prof/arm_chain_count.sh mode 19) against 86.1 for two
   isolated adds - libgcc shares nothing, so the same two operands are
   unpacked, aligned, normalised, rounded and packed twice.

   Alignment is the shared part, shared because it depends on
   the MAGNITUDES only: |a| and |b| line up the same way whichever sign
   the operation carries. What cannot be shared is normalise/round/pack,
   which is per result. This computes both magnitudes - P = |a| + |b|
   and M = ||a| - |b|| - off one unpack and one alignment.

   The two operations are one add and one subtract, always. a+b is an
   effective add exactly when the signs agree, and a-b is then the
   effective subtract; when they disagree the roles swap. So P and M are
   each computed once and the caller only chooses which is which.

   Signs fall out without a case analysis. The effective-add result is
   a+b when the signs agree and a-b when they do not, and both carry
   sign(a) - so P's sign is sign(a) unconditionally. The effective
   subtract is |a|-|b| taken with the sign of whichever operand is
   larger, under the operation that keeps a's sign when a wins: M's sign
   is sign(a) ^ (|a| < |b|), also unconditionally.

   DECLINES, and each is a case where identity would need a real IEEE
   adder rather than a proof:
     either operand zero, subnormal, inf or NaN   (exponent 0 or 255)
     P's exponent reaches 255 before rounding
     M underflows to exponent 0
   An exponent difference above 27 is NOT among them - see the note at
   the test itself, where it is answered rather than refused.

   P rounding UP into exponent 255 is not declined - inf is then the
   correctly rounded answer and the ADC produces it. Only an exponent
   already at 255 before the round is refused.
   The caller runs the C for those, so this tier can be slower than the
   C and never different.

   Why the sticky bit survives normalisation, the one step
   where a shift could invalidate it. The significands are held three
   bits above their final ulp, so an alignment shift of d <= 3 drops
   only the zeros those three bits added and the sticky is 0. For
   d >= 4 the aligned small operand is under 2^23 while the big one is
   at least 2^26, so their difference stays above 2^25 and CLZ asks for
   at most one bit of left shift - which moves the sticky from bit 0 to
   bit 1, still below the guard bit the tie test reads. A shift of two
   or more therefore only ever happens where the sticky is 0.
   .prof/verify_fwht_arm.c sweeps this rather than trusting it.

   One assembly block, not three, for a measured reason rather
   than stylistic. Split into unpack / add / subtract with C between
   them, gcc has to keep e, mA, mB and the two seeds live across the
   boundaries and spills eight words a butterfly - 100.0 instructions
   against libgcc's 106.7 on this operand distribution. Merged, the
   same arithmetic needs nine registers and no stack at all. The C left
   outside is the one thing that cannot go in: which of the two results
   is the sum, which the answer only knows from the operand signs. */
/* THE BODY IS THREE MACROS because there are two callers wanting two
   different things and only one copy of the arithmetic may exist.
   lz_faddsub_arm_asm (fwht's butterfly) needs both results. __aeabi_fadd
   and __aeabi_fsub need exactly ONE - and which one is decided by the
   sign test in the first two instructions, long before either result is
   computed, so the other block can be branched over instead of computed
   and discarded. That was about fifteen instructions thrown away on
   every scalar add in the engine. lz_fsum_arm_asm below is that caller;
   it shares HEAD/P/M with the two-result form rather than copying them,
   which is also what keeps build/dup_body_gate.sh quiet. */
/* HEAD leaves: %[ok] bit 0 = the signs differ, %[S] = P's sign, %[D] =
   M's sign, %[e] = the larger exponent, %[mA]/%[mB] = the aligned
   significands three bits above their final ulp. It jumps to 9f on the
   shapes this tier declines and to 8f when it has answered outright.

   d > 27 IS AN ANSWER, not a refusal. The small operand is then below
   half an ulp of the large one, so both results round to the large
   one's magnitude - and %[a] still holds it, shifted up by one. libgcc
   takes the same exit at d > 25 (ieee754-sf.S returns the larger
   operand outright). Without this exit the asm declines and pays a
   C-core call for a case an ORR answers. The sticky was being lost here anyway: LSR by a
   register uses only its low BYTE, so `rsb b, t, #32` goes negative
   past d=32 and the sticky test reads a shift-by-248, i.e. zero. It
   never showed up because the answer is the large operand either way. */
/* The d > 27 tail, chosen at the #define so the RTZ arm can decline.
   Selected as a separate macro for the same reason LZ_FADDSUB_ROUND
   is: a #if cannot live inside a macro body. */
#if LZ_SOFTFP_FAST
#define LZ_FADDSUB_LARGEDIFF \
        "b      9f\n\t"   /* RTZ: decline; the C core adds the tiny operand */
#else
#define LZ_FADDSUB_LARGEDIFF \
        "mov    %[a],  %[a], lsr #1\n\t"   /* |larger|, sign clear */    \
        "orr    %[S],  %[S], %[a]\n\t"                                   \
        "orr    %[D],  %[D], %[a]\n\t"                                   \
        "b      8f\n\t"
#endif /* LZ_SOFTFP_FAST */

#define LZ_FADDSUB_HEAD                                                  \
        "eor    %[ok], %[a], %[b]\n\t"                                   \
        "mov    %[ok], %[ok], lsr #31\n\t"  /* 1 if the signs differ */  \
        "and    %[S],  %[a], #0x80000000\n\t" /* P's sign is a's */      \
        "mov    %[a],  %[a], lsl #1\n\t"    /* |a|<<1: exp at 31:24 */   \
        "mov    %[b],  %[b], lsl #1\n\t"                                 \
        "sub    %[e],  %[a], #0x01000000\n\t"                            \
        "cmp    %[e],  #0xFE000000\n\t"                                  \
        "sublo  %[e],  %[b], #0x01000000\n\t"                            \
        "cmplo  %[e],  #0xFE000000\n\t"                                  \
        "bhs    9f\n\t"                     /* 0/subnormal/inf/nan */    \
        "cmp    %[a],  %[b]\n\t"            /* sign-magnitude order */   \
        "mov    %[D],  %[S]\n\t"                                         \
        "eorlo  %[D],  %[D], #0x80000000\n\t" /* M's sign */             \
        "movlo  %[e],  %[a]\n\t"                                         \
        "movlo  %[a],  %[b]\n\t"                                         \
        "movlo  %[b],  %[e]\n\t"                                         \
        "mov    %[e],  %[a], lsr #24\n\t"   /* larger exponent */        \
        "mov    %[t],  %[b], lsr #24\n\t"                                \
        "sub    %[t],  %[e], %[t]\n\t"      /* d */                      \
        "cmp    %[t],  #27\n\t"                                          \
        "bls    1f\n\t"                                                  \
        LZ_FADDSUB_LARGEDIFF                                             \
        "1:\n\t"                                                         \
        "mov    %[mB], %[b], lsl #8\n\t"                                 \
        "mov    %[mB], %[mB], lsr #6\n\t"                                \
        "orr    %[mB], %[mB], #0x04000000\n\t"                           \
        "rsb    %[b],  %[t], #32\n\t"                                    \
        "mov    %[b],  %[mB], lsl %[b]\n\t" /* d=0 shifts 32, giving 0 */\
        "mov    %[mB], %[mB], lsr %[t]\n\t"                              \
        "cmp    %[b],  #0\n\t"                                           \
        "orrne  %[mB], %[mB], #1\n\t"       /* sticky */                 \
        "mov    %[mA], %[a], lsl #8\n\t"                                 \
        "mov    %[mA], %[mA], lsr #6\n\t"                                \
        "orr    %[mA], %[mA], #0x04000000\n\t"

/* P = |a| + |b|: one carry out at most, folded with its sticky.

   Round and pack in four, the way __aeabi_fmul does it. The three extra
   bits go to the TOP of a register, so one compare against 2^31 puts
   the guard bit in C and the exact tie in Z; ADC then rounds up and
   adds the exponent together, BICEQ is round-half-to-even entire, and a
   significand that carries to 2^24 walks into the exponent field by
   itself because the implicit bit is added in rather than masked off.
   The six-instruction AND/CMP/ADDHI/ANDEQ/ADDEQ form this replaced
   needed three more to handle that carry.

   Clobbers %[a], %[b] and %[t]; %[e], %[mA] and %[mB] survive, which is
   what lets M run whether or not P ran before it. */
/* The round-and-pack tail, shared by P and M. REG is the destination
   operand name, "S" or "D". Selected at the #define, not inside the
   asm: a #if cannot live inside a macro body. */
#if LZ_SOFTFP_FAST
#define LZ_FADDSUB_ROUND(reg) \
        "add    %[" #reg "],  %[" #reg "], %[a], lsl #23\n\t"  /* RTZ */
#else
#define LZ_FADDSUB_ROUND(reg) \
        "cmp    %[b],  #0x80000000\n\t"                        \
        "adc    %[" #reg "],  %[" #reg "], %[a], lsl #23\n\t"  \
        "biceq  %[" #reg "],  %[" #reg "], #1\n\t"
#endif /* LZ_SOFTFP_FAST */

#define LZ_FADDSUB_P                                                     \
        "add    %[t],  %[mA], %[mB]\n\t"                                 \
        "sub    %[a],  %[e], #1\n\t"                                     \
        "cmp    %[t],  #0x08000000\n\t"                                  \
        "movhs  %[b],  %[t], lsr #1\n\t"                                 \
        "andhs  %[t],  %[t], #1\n\t"                                     \
        "orrhs  %[t],  %[b], %[t]\n\t"                                   \
        "addhs  %[a],  %[a], #1\n\t"                                     \
        "mov    %[b],  %[t], lsl #29\n\t"   /* dropped, left-justified */\
        "mov    %[t],  %[t], lsr #3\n\t"                                 \
        "cmp    %[a],  #254\n\t"                                         \
        "bhs    9f\n\t"                     /* P overflows to inf */     \
        "orr    %[S],  %[S], %[t]\n\t"                                   \
        LZ_FADDSUB_ROUND(S)

/* M = |a| - |b|. CLZ normalises; M cannot overflow, because rounding is
   monotone and M is below the larger operand, which is representable at
   this exponent. Reads only %[e], %[mA], %[mB] and %[D]'s sign, so it
   does not care whether P ran. */
#define LZ_FADDSUB_M                                                     \
        "subs   %[t],  %[mA], %[mB]\n\t"                                 \
        "moveq  %[D],  #0\n\t"              /* exact cancellation: +0 */ \
        "beq    8f\n\t"                                                  \
        "clz    %[a],  %[t]\n\t"                                         \
        "sub    %[a],  %[a], #5\n\t"                                     \
        "mov    %[t],  %[t], lsl %[a]\n\t"                               \
        "subs   %[a],  %[e], %[a]\n\t"                                   \
        "ble    9f\n\t"                     /* M is subnormal */         \
        "sub    %[a],  %[a], #1\n\t"                                     \
        "mov    %[b],  %[t], lsl #29\n\t"   /* same round-and-pack */    \
        "mov    %[t],  %[t], lsr #3\n\t"                                 \
        "orr    %[D],  %[D], %[t]\n\t"                                   \
        LZ_FADDSUB_ROUND(D)

static __inline__ int lz_faddsub_arm_asm(uint32_t ua, uint32_t ub,
                                         uint32_t *psum, uint32_t *pdif) {
    uint32_t ok, S, D, e, mA, mB, t, a = ua, b = ub;
    __asm__(
        LZ_FADDSUB_HEAD
        LZ_FADDSUB_P
        LZ_FADDSUB_M
        "8:\n\t"
        "orr    %[ok], %[ok], #2\n\t"
        "b      7f\n\t"
        "9:\n\t"
        "mov    %[ok], #0\n\t"
        "7:\n\t"
        : [ok] "=&r"(ok), [S] "=&r"(S), [D] "=&r"(D), [e] "=&r"(e),
          [mA] "=&r"(mA), [mB] "=&r"(mB), [t] "=&r"(t),
          [a] "+r"(a), [b] "+r"(b)
        :
        : "cc");
    if (!ok) return 0;
    if (ok & 1) { *psum = D; *pdif = S; }     /* signs differ */
    else        { *psum = S; *pdif = D; }
    return 1;
}

/* The sum alone, for __aeabi_fadd and __aeabi_fsub. Same contract and
   the same declines as lz_faddsub_arm_asm - it is the same three macros
   - but it computes ONE of the two results instead of both.

   The choice needs no extra work to discover. a+b is the effective add
   exactly when the signs agree, and %[ok] bit 0 already says so from
   HEAD's first two instructions, so a TST and a branch pick the block
   and the other one is never entered. That is about fifteen
   instructions off every scalar add and subtract in the engine.

   The d > 27 exit in HEAD still writes BOTH %[S] and %[D] rather than
   testing which is wanted: one wasted ORR is cheaper than the test, and
   keeping HEAD identical for both callers is what makes it shareable.

   ALWAYS_INLINE, and it is load-bearing rather than a hint. This body
   grew past gcc's inline-limit heuristic, which then outlined it and
   gave both callers a BL - a call and return, plus the register save
   and restore, wrapped around a leaf whose entire reason for existing
   is that __aeabi_fadd must not make a call. Measured, that handed back
   most of what skipping the unused block wins. lz_faddsub_arm_asm above
   does not need the attribute: fwht is its only caller and gcc inlines
   it unprompted. */
static __inline__ __attribute__((always_inline))
int lz_fsum_arm_asm(uint32_t ua, uint32_t ub,
                                      uint32_t *psum) {
    uint32_t ok, S, D, e, mA, mB, t, a = ua, b = ub;
    __asm__(
        LZ_FADDSUB_HEAD
        "tst    %[ok], #1\n\t"
        "bne    5f\n\t"                       /* signs differ: sum is M */
        LZ_FADDSUB_P
        "b      8f\n\t"
        "5:\n\t"
        LZ_FADDSUB_M
        "8:\n\t"
        "orr    %[ok], %[ok], #2\n\t"
        "b      7f\n\t"
        "9:\n\t"
        "mov    %[ok], #0\n\t"
        "7:\n\t"
        : [ok] "=&r"(ok), [S] "=&r"(S), [D] "=&r"(D), [e] "=&r"(e),
          [mA] "=&r"(mA), [mB] "=&r"(mB), [t] "=&r"(t),
          [a] "+r"(a), [b] "+r"(b)
        :
        : "cc");
    if (!ok) return 0;
    *psum = (ok & 1) ? D : S;
    return 1;
}

#endif /* __arm__ && __GNUC__ */

#endif /* LZ_OPS_ARM_PRIM_H */
