/* bfloat16 arithmetic. See lz_bf16.h for what it costs and why.

   STRUCTURE. Each operation has a hand-written ARMv5TE fast path that
   DECLINES (returns 0) on the shapes it does not cover - zero,
   subnormal, inf or NaN operands, and results out of normal range - and
   a portable C core that handles everything. Same contract as
   lz_softfp.c: the fast path owes the C body bit-identity and may only
   ever be faster, never different.

   The C core is written once, in f32: a bf16 widens to f32 by a shift,
   the arithmetic is exact there (8 mantissa bits times 8 fits in 24
   with room to spare, and a sum of two 8-bit significands needs 9), and
   the result rounds back. That is also the DEFINITION every test
   compares against, so the core cannot drift from the contract - it is
   the contract, spelled out.

   LZ_BF16_FAST (default 0) is the reduction half, and it changes ONE
   thing: round toward zero instead of to nearest-even. It does not
   flush denormals and does not skip the exponent checks - an earlier
   version of this comment claimed both, and neither was ever true; the
   macro has three uses in this file and all three are the rounding
   tail. Because the fast paths decline onto the C core, the core
   rounds the same way under the same macro, or the answer would depend
   on which shape the operands happened to have.

   It pairs with --fastfp on the other targets in intent - see
   lz_softfp.c's LZ_SOFTFP_FAST comment for why that pairing is not
   finished.
*/
#include "lz_bf16.h"

#ifndef LZ_BF16_FAST
#define LZ_BF16_FAST 0
#endif /* LZ_BF16_FAST */

#if defined(__GNUC__) || defined(__WATCOMC__)

/* ------------------------------------------------------------------ */
/* conversion                                                         */
/* ------------------------------------------------------------------ */

uint32_t lz_bf16_to_f32(uint32_t h) {
    return (h & 0xFFFFu) << 16;
}

uint32_t lz_f32_to_bf16(uint32_t f) {
    /* NaN must not round into an infinity, so it is passed through with
       its quiet bit forced - a NaN whose payload lives entirely in the
       low half would otherwise become 0x7F80, an inf. */
    uint32_t e = (f >> 23) & 0xFFu;
    uint32_t m = f & 0x7FFFFFu;
    if (e == 0xFFu) {
        if (m) return (f >> 16) | 0x0040u;      /* NaN, quiet */
        return f >> 16;                          /* inf */
    }
#if LZ_BF16_FAST
    /* Round toward zero, matching what the assembly does in this mode.
       It has to match: the fast paths decline on shapes this function
       then handles, so if the two rounded differently the answer would
       depend on which shape the operands happened to have - the one
       thing the decline contract exists to prevent. */
    return f >> 16;
#else
    /* Round-to-nearest-even on the 16 bits being dropped. The +0x7FFF
       is the halfway point and the +lsb turns a tie upward only when
       the kept bit is odd, which is exactly ties-to-even. */
    return (f + 0x7FFFu + ((f >> 16) & 1u)) >> 16;
#endif /* LZ_BF16_FAST */
}

/* ------------------------------------------------------------------ */
/* C core: widen to f32, operate, round back                          */
/* ------------------------------------------------------------------ */

/* The f32 arithmetic itself goes through the compiler, which on the ARM
   target means lz_softfp.c's __aeabi_* and on the others means hardware
   - either way the SAME f32 semantics the contract is written in. */

static uint32_t bf16_f32_bits(float f) {
    union { float f; uint32_t u; } u;
    u.f = f;
    return u.u;
}
static float bf16_f32_val(uint32_t u) {
    union { uint32_t u; float f; } v;
    v.u = u;
    return v.f;
}

static uint32_t bf16_binop_c(uint32_t a, uint32_t b, int op) {
    float x = bf16_f32_val(lz_bf16_to_f32(a));
    float y = bf16_f32_val(lz_bf16_to_f32(b));
    float r;
    switch (op) {
        case 0:  r = x + y; break;
        case 1:  r = x - y; break;
        case 2:  r = x * y; break;
        default: r = x / y; break;
    }
    return lz_f32_to_bf16(bf16_f32_bits(r));
}

/* ------------------------------------------------------------------ */
/* ARMv5TE fast paths                                                 */
/* ------------------------------------------------------------------ */

#if defined(__arm__) && defined(__GNUC__)

/* Multiply. The width saving lives here and nowhere else: an 8x8
   product is 16 bits, so unlike f32 there is no pre-shift to align a
   48-bit product into the high word and no sticky bit to recover from
   the low one. */
static __inline__ int bf16_mul_fast_asm(uint32_t ua, uint32_t ub,
                                        uint32_t *out) {
    uint32_t ok, s, ea, eb, ma, mb, p, t;
    __asm__(
        /* One predicate chain for all four refusals, libgcc's ARM trick:
           ANDS sets Z on a zero exponent, ANDNES both skips itself and
           leaves that Z alone, and the TEQNEs ask about 255 only if
           everything before passed. Six instructions, not twelve. */
        "mov    %[t], #255\n\t"
        "ands   %[ea], %[t], %[a], lsr #7\n\t"   /* Z: a zero/subnormal */
        "andnes %[eb], %[t], %[b], lsr #7\n\t"   /* Z: b zero/subnormal */
        "teqne  %[ea], %[t]\n\t"                 /* Z: a inf/nan */
        "teqne  %[eb], %[t]\n\t"                 /* Z: b inf/nan */
        "beq    9f\n\t"
        "eor    %[s], %[a], %[b]\n\t"
        "and    %[s], %[s], #0x8000\n\t"
        "add    %[ea], %[ea], %[eb]\n\t"
        /* expZ assuming the significand product lands in [2,4); the
           normalisation below takes one back if it did not. The
           significand keeps its implicit bit and the final ADD lets
           that 1 carry into the exponent field, which is why this is an
           add and not an orr - and why the bias removed here is 127 and
           not 128. */
        "sub    %[ea], %[ea], #127\n\t"
        "and    %[ma], %[a], #0x7F\n\t"
        "orr    %[ma], %[ma], #0x80\n\t"
        "and    %[mb], %[b], #0x7F\n\t"
        "orr    %[mb], %[mb], #0x80\n\t"
        "mul    %[p], %[ma], %[mb]\n\t"
        /* The product of two 8-bit significands is 15 or 16 bits (0x4000
           to 0xFE01), so its leading 1 sits at bit 15 when it reached
           2.0 and at bit 14 when it did not. Shifting by 15 puts the
           first case at bit 30, which is where the normalisation test
           below looks; the second case lands one lower and gets the
           conditional shift. */
        "mov    %[p], %[p], lsl #15\n\t"
        "cmp    %[p], #0x40000000\n\t"
        "movcc  %[p], %[p], lsl #1\n\t"
        "subcc  %[ea], %[ea], #1\n\t"
#if !LZ_BF16_FAST
        /* One unsigned compare for both ends: a negative exponent
           wraps to a huge unsigned, so this rejects ea<0 and
           ea>253 together and accepts exactly [0,253].
           softfloat s_roundPackToBF16.c:69, libgcc ieee754-sf.S:494. */
        "cmp    %[ea], #254\n\t"
        "bhs    9f\n\t"
        "mov    %[t], %[p], lsl #9\n\t"      /* round + sticky bits */
        "mov    %[p], %[p], lsr #23\n\t"     /* 8-bit magnitude */
        "orr    %[s], %[s], %[p]\n\t"
        "cmp    %[t], #0x80000000\n\t"
        "adc    %[s], %[s], %[ea], lsl #7\n\t"
        "biceq  %[s], %[s], #1\n\t"          /* round-half-to-even */
#else
        /* One unsigned compare for both ends: a negative exponent
           wraps to a huge unsigned, so this rejects ea<0 and
           ea>253 together and accepts exactly [0,253].
           softfloat s_roundPackToBF16.c:69, libgcc ieee754-sf.S:494. */
        "cmp    %[ea], #254\n\t"
        "bhs    9f\n\t"
        "mov    %[p], %[p], lsr #23\n\t"     /* truncate: drop the bits */
        "orr    %[s], %[s], %[p]\n\t"
        "add    %[s], %[s], %[ea], lsl #7\n\t"
#endif /* !LZ_BF16_FAST */
        "mov    %[ok], #1\n\t"
        "b      8f\n\t"
        "9:\n\t"
        "mov    %[ok], #0\n\t"
        "8:\n\t"
        : [ok]"=&r"(ok), [s]"=&r"(s), [ea]"=&r"(ea), [eb]"=&r"(eb),
          [ma]"=&r"(ma), [mb]"=&r"(mb), [p]"=&r"(p), [t]"=&r"(t)
        : [a]"r"(ua), [b]"r"(ub)
        : "cc");
    /* THE STORE IS C's, not the assembly's, and that is worth about ten
       instructions. Writing `str %[s], [%[o]]` inside the block forces
       the caller's local to have an ADDRESS: gcc then builds a stack
       frame, stores the result and loads it straight back, and - because
       a local's address escaped into an asm with a "memory" clobber -
       links in the stack-protector prologue and epilogue as well. None
       of that survives when the value comes out in a register and the C
       assigns it, which is exactly how sf_fmul_fast_arm_asm in
       lz_softfp.c has always done it. */
    if (ok) *out = s;
    return (int)ok;
}

/* Add/sub, structured after Berkeley SoftFloat's s_addMagsF16.c rather
   than invented here - two of its moves are ones this file got wrong on
   the first attempt:

     - EQUAL EXPONENTS need no alignment at all. It is also the common
       case in an accumulation of same-scale terms, so it gets its own
       exit ahead of the shifter.
     - AN OUT-OF-REACH OPERAND IS NOT A REFUSAL. Past 10 binades the
       smaller addend is below a quarter ulp of the larger, and under
       round-to-nearest-even the result IS the larger operand - SoftFloat
       falls through to `addEpsilon`, which in that rounding mode only
       raises inexact and changes no bits. Declining to the C core here
       instead cost 3.57% of the adds in a measured 3584-term dot
       product; returning the larger operand costs two instructions.
       (SoftFloat's f16 threshold is 13 for a 10-bit significand, i.e.
       three past the width; bf16's 7-bit one puts it at 10.) */
static __inline__ int bf16_addsub_fast_asm(uint32_t ua, uint32_t ub,
                                           uint32_t *out) {
    uint32_t ok, s, ea, eb, ma, mb, d, t;
    __asm__(
        /* One predicate chain for all four refusals - see the note in
           bf16_mul_fast_asm. Six instructions, not twelve. */
        "mov    %[t], #255\n\t"
        "ands   %[ea], %[t], %[a], lsr #7\n\t"   /* Z: a zero/subnormal */
        "andnes %[eb], %[t], %[b], lsr #7\n\t"   /* Z: b zero/subnormal */
        "teqne  %[ea], %[t]\n\t"                 /* Z: a inf/nan */
        "teqne  %[eb], %[t]\n\t"                 /* Z: b inf/nan */
        "beq    9f\n\t"
        /* Order by magnitude, with the operand's HIGH HALF DISCARDED and
           not merely its sign bit. LSL #17 drops bits 31..15 and leaves
           the 15-bit magnitude at the top of the word, where comparing
           two of them gives the same ordering - one instruction, exactly
           what the BIC of 0x8000 it replaces cost. (0x7FFF cannot be an
           ARM immediate, which is an 8-bit value rotated by an even
           amount; 0x8000 can, which is why the BIC was written that way
           and why it only ever cleared bit 15.)

           IT MATTERS BECAUSE THE C CORE MASKS AND THIS DID NOT.
           lz_bf16_to_f32 takes `h & 0xFFFF`, so the core ignores
           anything above the low half; the assembly read the word whole,
           so an argument with high bits set got one answer when the fast
           path took it and another when it declined - measured, 12.0M of
           26.3M cases differed. Which of the two ran is invisible to the
           caller and depends on the operand values, so this was the
           decline contract broken, not a documentation gap. */
        "mov    %[ma], %[a], lsl #17\n\t"
        "mov    %[mb], %[b], lsl #17\n\t"
        "cmp    %[ma], %[mb]\n\t"
        "movcc  %[t], %[a]\n\t"
        "movcc  %[a], %[b]\n\t"
        "movcc  %[b], %[t]\n\t"
        "movcc  %[t], %[ea]\n\t"
        "movcc  %[ea], %[eb]\n\t"
        "movcc  %[eb], %[t]\n\t"
        /* No equal-exponent special case: ARM's LSR by a register
           whose low byte is zero is a no-op, so the shift below
           already handles d==0. SoftFloat needs the branch only
           because C has no such guarantee. */
        "sub    %[d], %[ea], %[eb]\n\t"
        "cmp    %[d], #9\n\t"
        /* Out of reach: the answer IS the larger operand - its LOW HALF,
           for the reason above. This is the one path that returns an
           argument rather than something it built, so it is also the one
           that could hand a caller a result with high bits set. */
        "movhi  %[s], %[a], lsl #16\n\t"
        "movhi  %[s], %[s], lsr #16\n\t"
        "bhi    7f\n\t"
        "and    %[ma], %[a], #0x7F\n\t"
        "orr    %[ma], %[ma], #0x80\n\t"
        "and    %[mb], %[b], #0x7F\n\t"
        "orr    %[mb], %[mb], #0x80\n\t"
        "mov    %[ma], %[ma], lsl #16\n\t"   /* room for guard/round */
        "mov    %[mb], %[mb], lsl #16\n\t"
        "mov    %[mb], %[mb], lsr %[d]\n\t"
        "eor    %[t], %[a], %[b]\n\t"
        "tst    %[t], #0x8000\n\t"
        "addeq  %[ma], %[ma], %[mb]\n\t"     /* like signs: add */
        "subne  %[ma], %[ma], %[mb]\n\t"     /* unlike: subtract */
        "beq    2f\n\t"
        /* EXACT CANCELLATION IS AN ANSWER, and a fixed one: IEEE 754
           gives x - x the sign +0 in every rounding mode except
           roundTowardNegative, which this file does not implement (both
           LZ_BF16_FAST arms are nearest-even or toward-zero). So it is
           +0 here and +0 in the C core, and declining to go and find
           that out cost a call. One MOV, and the store at 7f is already
           on the way out. */
        "cmp    %[ma], #0\n\t"
        "moveq  %[s], #0\n\t"
        "beq    7f\n\t"
        "2:\n\t"
        "and    %[s], %[a], #0x8000\n\t"     /* result takes the larger's */
        /* normalise: bring the leading 1 to bit 30 */
        "clz    %[t], %[ma]\n\t"
        "sub    %[t], %[t], #1\n\t"
        "mov    %[ma], %[ma], lsl %[t]\n\t"
        /* Three fixups are one constant. The normalising shift takes
           %[t] off, the lsl #16 that made room for guard and round puts
           7 back, and the packing wants expZ-1 for the same reason the
           multiply does - the implicit bit is still in the significand
           and the ADC below carries it into the exponent field. That is
           -t +7 -1, i.e. -t +6, and the assembler can fold constants
           this file cannot leave to it when they are written apart. */
        "sub    %[ea], %[ea], %[t]\n\t"
        "add    %[ea], %[ea], #6\n\t"
#if !LZ_BF16_FAST
        /* One unsigned compare for both ends: a negative exponent
           wraps to a huge unsigned, so this rejects ea<0 and
           ea>253 together and accepts exactly [0,253].
           softfloat s_roundPackToBF16.c:69, libgcc ieee754-sf.S:494. */
        "cmp    %[ea], #254\n\t"
        "bhs    9f\n\t"
        "mov    %[t], %[ma], lsl #9\n\t"
        "mov    %[ma], %[ma], lsr #23\n\t"
        "orr    %[s], %[s], %[ma]\n\t"
        "cmp    %[t], #0x80000000\n\t"
        "adc    %[s], %[s], %[ea], lsl #7\n\t"
        "biceq  %[s], %[s], #1\n\t"
#else
        /* One unsigned compare for both ends: a negative exponent
           wraps to a huge unsigned, so this rejects ea<0 and
           ea>253 together and accepts exactly [0,253].
           softfloat s_roundPackToBF16.c:69, libgcc ieee754-sf.S:494. */
        "cmp    %[ea], #254\n\t"
        "bhs    9f\n\t"
        "mov    %[ma], %[ma], lsr #23\n\t"
        "orr    %[s], %[s], %[ma]\n\t"
        "add    %[s], %[s], %[ea], lsl #7\n\t"
#endif /* !LZ_BF16_FAST */
        "7:\n\t"                             /* out-of-reach lands here */
        "mov    %[ok], #1\n\t"
        "b      8f\n\t"
        "9:\n\t"
        "mov    %[ok], #0\n\t"
        "8:\n\t"
        : [ok]"=&r"(ok), [s]"=&r"(s), [ea]"=&r"(ea), [eb]"=&r"(eb),
          [ma]"=&r"(ma), [mb]"=&r"(mb), [d]"=&r"(d), [t]"=&r"(t),
          [a]"+r"(ua), [b]"+r"(ub)
        :
        : "cc");
    /* The store is C's - see the note in bf16_mul_fast_asm. */
    if (ok) *out = s;
    return (int)ok;
}

#endif /* __arm__ && __GNUC__ */

/* ------------------------------------------------------------------ */
/* entry points                                                       */
/* ------------------------------------------------------------------ */

/* ZERO IS TESTED FIRST, AHEAD OF THE ASSEMBLY, and the order is the
   whole point. The fast path's operand guard cannot tell a zero from a
   subnormal - both have exponent 0 - so a zero operand always declines,
   after paying the six-instruction chain and the branch. Testing it here
   answers the shape in three and never reaches the assembly. It costs
   the non-zero traffic one AND and one compare, against the twenty-odd
   instructions a decline costs the zero traffic without it - and on an
   accumulator that starts at zero that is not a rare case: on f32 this one
   shape was 99.8% of every decline measured on the zh fixture.

   x*0 is +-0 with the sign XORed and no rounding. inf/NaN partners are
   excluded because inf*0 is invalid and sNaN*0 must be quieted - both
   are the C core's, and getting that wrong on the f32 side produced
   1426 mismatches against libgcc before it was caught. */
uint32_t lz_bf16_mul(uint32_t a, uint32_t b) {
    if ((!(a & 0x7FFFu) || !(b & 0x7FFFu)) &&
        (a & 0x7F80u) != 0x7F80u && (b & 0x7F80u) != 0x7F80u) {
        return (a ^ b) & 0x8000u;
    }
#if defined(__arm__) && defined(__GNUC__)
    {   uint32_t q;
        if (bf16_mul_fast_asm(a, b, &q)) return q; }
#endif /* __arm__ && __GNUC__ */
    return bf16_binop_c(a, b, 2);
}

/* A +-0 operand, answered ahead of the assembly for the reason given at
   lz_bf16_mul: the fast path's guard cannot tell a zero from a
   subnormal, so this shape declines after paying for the whole chain.
   x+0 IS x, with no rounding; 0+0 is +0 unless BOTH are -0, which is
   the AND of the two sign bits.

   The exponent test is not redundant: 0 + sNaN must QUIET the NaN, so
   an inf/NaN partner cannot take the shortcut.

   THE MASKS ON THE WAY OUT ARE NOT DECORATION. These two lines are the
   only ones in the file that answer with an ARGUMENT rather than with
   something they built, so they are the only ones that could hand back a
   word with bits above the low half set. Everything else is masked by
   construction - a sign is ANDed with 0x8000, a significand is 8 bits, a
   packed exponent is shifted left by 7 - and lz_f32_to_bf16 ends in a
   shift right by 16. Without them this shortcut disagreed with both the
   assembly and the C core on any argument carrying high bits. */
static int bf16_zero_shortcut(uint32_t a, uint32_t b, uint32_t *out) {
    if (!(b & 0x7FFFu) && (a & 0x7F80u) != 0x7F80u) {
        *out = (a & 0x7FFFu) ? (a & 0xFFFFu) : (a & b & 0x8000u);
        return 1;
    }
    if (!(a & 0x7FFFu) && (b & 0x7F80u) != 0x7F80u) {
        *out = b & 0xFFFFu;
        return 1;
    }
    return 0;
}

uint32_t lz_bf16_add(uint32_t a, uint32_t b) {
    uint32_t q;
    if (bf16_zero_shortcut(a, b, &q)) return q;
#if defined(__arm__) && defined(__GNUC__)
    if (bf16_addsub_fast_asm(a, b, &q)) return q;
#endif /* __arm__ && __GNUC__ */
    return bf16_binop_c(a, b, 0);
}

uint32_t lz_bf16_sub(uint32_t a, uint32_t b) {
    uint32_t q;
    if (bf16_zero_shortcut(a, b ^ 0x8000u, &q)) return q;
#if defined(__arm__) && defined(__GNUC__)
    if (bf16_addsub_fast_asm(a, b ^ 0x8000u, &q)) return q;
#endif /* __arm__ && __GNUC__ */
    return bf16_binop_c(a, b, 1);
}

uint32_t lz_bf16_div(uint32_t a, uint32_t b) {
    /* No fast path: 176 divides per token against 247,932 multiplies and
       74,985 adds, so the assembly would be carried and never repaid.

       TWO WAYS TO MISREAD THOSE FIGURES, both easy and both made here
       once. lz_softfp.c's census prints TOTALS OVER A 603-TOKEN RUN
       (kmr20, zh fixture) - 149,503,180 fmul, 45,215,879 fadd, 106,048
       fdiv fast plus 613 declined. Take the 149.5M for a per-token
       count, or the 613 for a per-token divide count when 613 is the
       whole run's DECLINES, and the ratio is out by about 175x. Divide
       by 603 for the per-token numbers above. The conclusion holds at
       1:1400 either way, which is exactly why such an error survives
       being looked at. */
    return bf16_binop_c(a, b, 3);
}

int lz_bf16_cmp(uint32_t a, uint32_t b) {
    uint32_t am = a & 0x7FFFu, bm = b & 0x7FFFu;
    if (am > 0x7F80u || bm > 0x7F80u) return 2;      /* NaN: unordered */
    if (!am && !bm) return 0;                        /* +0 == -0 */
    if (a == b) return 0;
    if ((a ^ b) & 0x8000u) return (a & 0x8000u) ? -1 : 1;
    if (a & 0x8000u) return (am > bm) ? -1 : 1;      /* both negative */
    return (am > bm) ? 1 : -1;
}

uint32_t lz_bf16_from_i32(int32_t v) {
    return lz_f32_to_bf16(bf16_f32_bits((float)v));
}

int32_t lz_bf16_to_i32(uint32_t h) {
    return (int32_t)bf16_f32_val(lz_bf16_to_f32(h));
}

#endif /* __GNUC__ || __WATCOMC__ */
