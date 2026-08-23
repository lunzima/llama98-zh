/* The fused MAC core, shared by .prof/arm_prim_count.c mode 26 (what it
   costs) and .prof/verify_fmac_core.c (whether it is right). One copy,
   because a probe measuring one body while a gate checks another is the
   oldest way for both to be green at once. */
#ifndef SPW_FMAC_CORE_H
#define SPW_FMAC_CORE_H
#include <stdint.h>
#if defined(__arm__) && defined(__GNUC__)

/* THE CORE ALONE, no decline wrapper. Mode 25 measured the shippable
 * per-element primitive and lost by 12.5; disassembling it split that
 * into 8 of wrapper (three operand copies kept for a fallback that
 * never runs, four of ok-flag, an extra address reload) and 4.5 of
 * core. A LEAF kernel over a whole row pays libgcc's wrapper and not
 * that one - the decline becomes one branch out of the loop - so the
 * question the cell turns on is the CORE, and this measures it.
 *
 * The adder half is __addsf3's algorithm rather than my own: signed
 * significands so one ADDS with an inline ASR does both effective
 * operations with no branch, the shifted-out bits kept as a full low
 * word instead of compressed to a sticky, `cmp #25 / return the larger`
 * for the far-apart case my own version declined, and ADC/BICEQ to
 * round. What fusion removes from it is its scan of the product - this
 * code just built that operand and knows it is normal - and both
 * call/return pairs.
 *
 * NOT SHIPPABLE AS IS: it assumes normal operands, which the probe's
 * stream satisfies (mode 25 declined 0 times in 1024 calls) and a real
 * kernel could not. It is an upper bound on what the leaf can reach. */
static __inline__ uint32_t spw_fmac_core(uint32_t uacc, uint32_t uw,
                                         uint32_t ux) {
    uint32_t R, e, t, pl, ph, a = uw, b = ux, c = uacc;
    __asm__(
        /* ---- product, into R as a packed float ---- */
        "mov    %[t],  #255\n\t"
        "and    %[e],  %[t], %[a], lsr #23\n\t"
        "and    %[ph], %[t], %[b], lsr #23\n\t"
        "add    %[e],  %[e], %[ph]\n\t"
        "eor    %[R],  %[a], %[b]\n\t"
        "mov    %[ph], #0x8000000\n\t"
        "mov    %[a],  %[a], lsl #9\n\t"
        "orr    %[a],  %[ph], %[a], lsr #5\n\t"
        "mov    %[b],  %[b], lsl #9\n\t"
        "orr    %[b],  %[ph], %[b], lsr #5\n\t"
        "umull  %[pl], %[ph], %[a], %[b]\n\t"
        "and    %[R],  %[R], #0x80000000\n\t"
        "cmp    %[ph], #0x800000\n\t"
        "movcc  %[ph], %[ph], lsl #1\n\t"
        "orrcc  %[ph], %[ph], %[pl], lsr #31\n\t"
        "movcc  %[pl], %[pl], lsl #1\n\t"
        "orr    %[R],  %[R], %[ph]\n\t"
        "sbc    %[e],  %[e], #127\n\t"
        "cmp    %[pl], #0x80000000\n\t"
        "adc    %[R],  %[R], %[e], lsl #23\n\t"
        "biceq  %[R],  %[R], #1\n\t"
        /* ---- acc + product, __addsf3's shape minus its scan ---- */
        "mov    %[a],  %[c], lsl #1\n\t"
        "mov    %[e],  %[a], lsr #24\n\t"
        "mov    %[b],  %[R], lsl #1\n\t"
        /* EXACT CANCELLATION, which the normalising path below cannot
           answer: CLZ of zero is 32 and the shift that follows produces
           a garbage exponent. __addsf3 removes this case in its opening
           scan (`teqne r2, r3` catches equal magnitudes); the fused form
           has no such scan, so the test is here. Two instructions when
           the magnitudes differ, which is every case but this one. */
        "teq    %[a],  %[b]\n\t"
        "bne    5f\n\t"
        "eors   %[t],  %[c], %[R]\n\t"
        "movmi  %[R],  #0\n\t"                     /* opposite signs: +0 */
        "bmi    8f\n\t"
        "5:\n\t"
        "rsbs   %[ph], %[e], %[b], lsr #24\n\t"    /* d */
        "addgt  %[e],  %[e], %[ph]\n\t"
        "eorgt  %[R],  %[c], %[R]\n\t"             /* order the two words */
        "eorgt  %[c],  %[R], %[c]\n\t"
        "eorgt  %[R],  %[c], %[R]\n\t"
        "rsblt  %[ph], %[ph], #0\n\t"
        "cmp    %[ph], #25\n\t"
        "movhi  %[R],  %[c]\n\t"                   /* too far apart */
        "bhi    8f\n\t"
        "tst    %[c],  #0x80000000\n\t"
        "orr    %[c],  %[c], #0x800000\n\t"
        "bic    %[c],  %[c], #0xFF000000\n\t"
        "rsbne  %[c],  %[c], #0\n\t"               /* SIGNED significand */
        "tst    %[R],  #0x80000000\n\t"
        "orr    %[R],  %[R], #0x800000\n\t"
        "bic    %[R],  %[R], #0xFF000000\n\t"
        "rsbne  %[R],  %[R], #0\n\t"
        "sub    %[e],  %[e], #1\n\t"
        "adds   %[c],  %[c], %[R], asr %[ph]\n\t"  /* align AND add, one op */
        "rsb    %[ph], %[ph], #32\n\t"
        "mov    %[R],  %[R], lsl %[ph]\n\t"        /* the bits shifted out */
        "and    %[ph], %[c], #0x80000000\n\t"
        "bpl    1f\n\t"
        "rsbs   %[R],  %[R], #0\n\t"               /* negate the 64-bit pair */
        "rsc    %[c],  %[c], #0\n\t"
        "1:\n\t"
        "cmp    %[c],  #0x800000\n\t"
        "bcc    2f\n\t"
        "cmp    %[c],  #0x1000000\n\t"
        "bcc    3f\n\t"
        "movs   %[c],  %[c], lsr #1\n\t"
        "rrx    %[R],  %[R]\n\t"
        "add    %[e],  %[e], #1\n\t"
        "3:\n\t"
        "cmp    %[R],  #0x80000000\n\t"
        "adc    %[c],  %[c], %[e], lsl #23\n\t"
        "biceq  %[c],  %[c], #1\n\t"
        "orr    %[R],  %[c], %[ph]\n\t"
        "b      8f\n\t"
        "2:\n\t"                                   /* cancellation */
        "lsls   %[R],  %[R], #1\n\t"
        "adc    %[c],  %[c], %[c]\n\t"             /* 64-bit shift left one */
        "subs   %[e],  %[e], #1\n\t"
        "cmpcs  %[c],  #0x800000\n\t"
        "bcs    3b\n\t"
        "clz    %[t],  %[c]\n\t"
        "sub    %[t],  %[t], #8\n\t"
        "subs   %[e],  %[e], %[t]\n\t"
        "mov    %[c],  %[c], lsl %[t]\n\t"
        "add    %[c],  %[c], %[e], lsl #23\n\t"
        "orr    %[R],  %[c], %[ph]\n\t"
        "8:\n\t"
        : [R] "=&r"(R), [e] "=&r"(e), [t] "=&r"(t),
          [pl] "=&r"(pl), [ph] "=&r"(ph),
          [a] "+r"(a), [b] "+r"(b), [c] "+r"(c)
        :
        : "cc");
    return R;
}

#endif /* __arm__ && __GNUC__ */
#endif /* SPW_FMAC_CORE_H */
