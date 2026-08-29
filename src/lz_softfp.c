/* ARM EABI soft-float: single-precision (binary32) arithmetic,
   comparison, and int<->float conversion - bit-identical to libgcc's
   f32 soft-float routines, so the engine need not link libgcc's
   soft-float at all on ARMv5TE.

   Each __aeabi_* entry tries a hand-written ARMv5TE fast path first
   (CLZ + barrel shifter + conditional execution, the same cell library
   as ops_arm_prim.h - lz_i32f_arm_asm and lz_faddsub_arm_asm are reused
   directly) and DECLINES to the C core below on the cases the asm does
   not cover (subnormals, inf/NaN, denormal or overflow results). The
   decline-and-fallback rule is ops_arm_prim.h's: the tier owes the C
   bit-identity and can only ever be slower, never different.

   COST, COUNTED THE SAME WAY ON BOTH SIDES, which an earlier version of
   this line did not do: it put this file's inline-asm BLOCKS against
   libgcc's WHOLE functions ("~14-80 ... down from libgcc's 23-102"),
   which flatters the comparison by everything the wrapper costs and by
   every subnormal-and-NaN path libgcc carries and this file declines.
   The figures below are entry to return, fast path taken, counted off
   the disassembly of this file's own object:

       __aeabi_fadd   64   (was 76)
       __aeabi_fsub   63   (was 75)   one below fadd: the sign flip
                                      folds into the ADD that unpacks b
       __aeabi_fmul   33   (was 45)
       __aeabi_fdiv  122 worst case, 47 when the division is exact and
                     the quotient loop takes its early exit

   libgcc's corresponding fast paths are 28 for the multiply and about
   40 for the add - close, and reached the same way, because that is
   where the multiply's core came from (see sf_fmul_fast_arm_asm). The
   102 quoted before was its whole __aeabi_fmul. docs/arm-asm-audit.md
   keeps the per-cell table.

   EVERY FUNCTION is verified against libgcc (via qemu-arm-static on the
   arm-linux-gnueabi cross toolchain) on ~40M random bit patterns plus a
   directed special-value matrix: 0/+-0, subnormals, inf, NaN, overflow/
   underflow edges, and the rounding boundaries (round-to-nearest-even,
   the PC=24 precision the engine's x87 arm also enforces).

   LZ_SOFTFP_ASSUME_FINITE (default 0) drops __aeabi_fmul's overflow and
   underflow EXPONENT check - the branch that declines a result too big
   or too small to be a normal binary32. It does NOT touch the operand
   checks: those decline on a zero, subnormal, inf or NaN INPUT, and
   inputs are not what this switch makes a claim about.

   ONE SITE, NOT ALL THE FAST PATHS, which is narrower than this used to
   claim. __aeabi_fdiv's range check and the add's (in ops_arm_prim.h's
   LZ_FADDSUB_P) are unconditional and always have been, so the switch
   moves fmul alone. Measured whole-function: 50 -> 48 instructions with
   it on, fadd and fdiv unchanged at 108 and 51.

   THAT IS ALSO WHY IT STAYS OFF BY DEFAULT. Two instructions on one
   operator is the entire prize, and the price is that an overflowing
   product stops declining and starts packing a wrong exponent instead -
   silently, since there is no NaN to notice it by. The evidence for
   "it never overflows" is two fixtures, and a shipped binary sees
   neither of them. A decline that costs two instructions and converts
   a silent wrong answer into a correct slow one is not overhead worth
   removing; it is the contract.

   The claim it does make is narrow and measured: build/fp_exception_gate
   reads x86's sticky MXCSR flags after a real inference run and asserts
   OE=0 and UE=0, i.e. no result on that path ever overflowed to inf or
   fell below normal range. A branch that never runs costs nothing to
   remove and changes nothing when removed - which is the whole argument,
   and also why this must stay tied to the gate.

   WHAT IT IS NOT. It is not "assume no NaN": a NaN input PROPAGATES
   through a multiply without raising invalid, so the gate's IE=0 says
   nothing about NaN operands and the operand checks stay. It is not a
   rounding change either; results that do land in range are still
   round-to-nearest-even and still bit-identical to libgcc.

   If the gate ever fails, this switch is unsound for that model: an
   overflowing product would wrap its exponent instead of declining, and
   there is no NaN to notice it by.

   Rounding is IEEE 754 round-to-nearest-even, hardcoded. NaN payload is
   propagated libgcc-style (first NaN operand, quiet bit set); default
   NaN is 0x7FC00000 with the sign of the inf operand where libgcc does
   that (measured, see the *_impl functions). Integer conversions are
   truncation toward zero with saturation on overflow, NaN -> 0.
*/
#ifndef LZ_SOFTFP_ASSUME_FINITE
#define LZ_SOFTFP_ASSUME_FINITE 0
#endif /* !LZ_SOFTFP_ASSUME_FINITE */

/* LZ_SOFTFP_STATS (default 0, diagnostic only) counts, per operator, how
   often the fast path is TAKEN versus how often the C core is actually
   ENTERED - and for the latter, which operand shape sent it there.
   Self-contained via atexit so no caller changes; stderr, which is why
   it does not breach src/'s console ban for a build nobody ships.
      OUT=/tmp/x bash build/arm/build_engine.sh   # plus -DLZ_SOFTFP_STATS=1
      qemu-arm-static -cpu arm926 -L /usr/arm-linux-gnueabi \
          ./llama98-arm MODEL --tokens-file FIXTURE 2>&1 >/dev/null

   IT EXISTS BECAUSE COUNTING BEAT GUESSING, TWICE. Reading the code says
   every operator's fast path is worth optimising; the measurement said
   otherwise (kmr20, zh fixture).

   THESE ARE WHOLE-RUN TOTALS OVER 603 TOKENS, not per-token rates, and
   the per-token column is spelled out because the omission has already
   caused two wrong numbers elsewhere in this tree - a comment in
   lz_bf16.c argued from "150 million multiplies per token" and from
   reading the 613 as a per-token divide count when it is the run's
   declines. A reader skims the big number, not the header line.

     fmul  149,503,180 fast   1,298,514 declined   0.861%   247,932/tok
     fadd   45,215,879 fast  13,188,114 declined  22.581%    74,985/tok
     fdiv      106,048 fast         613 declined   0.575%       176/tok

   Effort had gone into shaving 4 instructions off fmul's fast path -
   which its 0.861% decline rate makes nearly free already - while fadd
   was paying a call into the C core on nearly a quarter of its uses. And
   the reason was not exotic: +-0 operands were 99.8% of fadd's declines
   and 99.9997% of fmul's, because an accumulator starts at zero and the
   fast path's combined 0/subnormal/inf/nan guard cannot tell those
   apart. Answering +-0 inline took C-core entries from 13,188,114 to
   22,730 for fadd and from 1,298,514 to 4 for fmul.

   So: measure the decline rate and its CAUSE before optimising a fast
   path. Count C-core entries, not primitive declines - the first version
   of this counter sat before the +-0 shortcut and reported no change at
   all after the shortcut was added. */
/* LZ_SOFTFP_FAST (default 0) makes the whole soft-float library round
   toward zero plus flush denormals, matching what x86's --fastfp MXCSR
   does (RC=truncate, FTZ, DAZ). It is the ARM half of the fast-mode
   pairing: a fast-mode ARM run must byte-compare with a fast-mode x86
   run the way the strict pair already does. Before this was completed
   the switch touched fmul alone, and the two sides disagreed on
   60,173,584 of 78,905,344 logit bytes - the coverage gap, not a
   correctness one (see build/arm/fastfp_probe.c for the verification).

   WHAT IT COVERS, named because the old fmul-only wording was the bug:
   fadd/fsub/fmul/fdiv, the C cores behind all three, the int<->float
   conversions (i2f/ui2f/l2f), and the faddsub cell in ops_arm_prim.h
   that fwht.c shares. FTZ flushes subnormal results, DAZ treats
   subnormal operands as +-0, and overflow CLAMPS to max_normal - the
   last is the round-toward-zero reading of a result past the top of the
   range, where round-to-nearest rounds up to inf. The operands keep
   their IEEE inf/NaN handling: MXCSR changes rounding and denormal
   handling only, so an ARM path that skipped them could never match.

   WHAT IT DOES NOT COVER, and cannot: NaN SIGN. x86 hardware produces
   a negative quiet NaN where this library, verified against libgcc,
   produces a positive one - 0xffc00000 vs 0x7fc00000 for 0/0. Strict
   mode REQUIRES the libgcc sign (that is the bit-identity contract),
   and the fast pair cannot ask for the x86 sign without breaking it.
   It is unreachable in practice: fp_exception_gate asserts IE=0 over a
   real run, so no NaN is ever produced, and the probe records the 40
   directed div-by-zero cases as the one surviving difference.

   Truncation against round-to-nearest-even saves, per result, the
   left-justify of the dropped bits, the compare against 2^31, and the
   BICEQ, and turns the ADC into an ADD. Against the soft-float census:

       fmul  247,932/tok x 33 = 8,181,756    -3 each =   743,796
       fadd   74,985/tok x 64 = 4,799,040    -3 each =   224,955
       fdiv      176/tok x122 =    21,472    -3 each =       528
       total                  13,002,268              =   969,279

   7.5% of the soft-float arithmetic. RTZ is the shipping semantics;
   the strict (RNE) soft-float is built with -DLZ_SOFTFP_FAST=0.
*/
/* LZ_SOFTFP_FAST's default lives in ops.h (single authority); this TU
   reads it as-is. */

#ifndef LZ_SOFTFP_STATS
#define LZ_SOFTFP_STATS 0
#endif /* !LZ_SOFTFP_STATS */
#include "lz_int.h"
#include <limits.h>   /* INT32_MIN/INT32_MAX */
#include "ops_arm_prim.h"   /* lz_i32f_arm_asm, lz_faddsub_arm_asm */

#if defined(__arm__) && defined(__GNUC__) && !defined(__SOFTFP_ONLY_HOST__)

#if LZ_SOFTFP_STATS
#include <stdio.h>
#include <stdlib.h>
enum { SF_MUL, SF_ADD, SF_DIV, SF_OPS };
static unsigned long sf_fast[SF_OPS], sf_slow[SF_OPS];
static const char *const sf_opname[SF_OPS] = { "fmul", "fadd", "fdiv" };
/* Why did faddsub decline? The rate alone does not say which guard
   fired, and the fixes differ by an order of magnitude: a +-0 operand
   needs three instructions inline, an expDiff > 27 needs none at all
   (the answer IS the larger operand), a real subnormal needs the C
   core. Counted separately so the cheap ones can be told apart. */
enum { SFD_ZERO, SFD_SUBNORM, SFD_INFNAN, SFD_EXPDIFF, SFD_RESULT, SFD_N };
static unsigned long sf_why[SF_OPS][SFD_N];
static const char *const sf_whyname[SFD_N] = {
    "+-0 operand", "subnormal operand", "inf/nan operand",
    "expDiff > 27", "overflow/subnormal result"
};
static void sf_bill_why(int op, uint32_t ua, uint32_t ub) {
    int ea = (int)((ua >> 23) & 0xFF), eb = (int)((ub >> 23) & 0xFF);
    int d;
    if ((!ea && !(ua & 0x7FFFFFu)) || (!eb && !(ub & 0x7FFFFFu)))
        sf_why[op][SFD_ZERO]++;
    else if (!ea || !eb)             sf_why[op][SFD_SUBNORM]++;
    else if (ea == 255 || eb == 255) sf_why[op][SFD_INFNAN]++;
    else {
        d = ea - eb; if (d < 0) d = -d;
        if (op == SF_ADD && d > 27) sf_why[op][SFD_EXPDIFF]++;
        else                        sf_why[op][SFD_RESULT]++;
    }
}
static void sf_stats_dump(void) {
    int i;
    for (i = 0; i < SF_OPS; i++) {
        unsigned long t = sf_fast[i] + sf_slow[i];
        if (!t) continue;
        fprintf(stderr, "softfp %s: fast=%lu decline=%lu (%.3f%% declined)\n",
                sf_opname[i], sf_fast[i], sf_slow[i],
                100.0 * (double)sf_slow[i] / (double)t);
    }
    for (i = 0; i < SF_OPS; i++) {
        unsigned long tw = 0; int j;
        for (j = 0; j < SFD_N; j++) tw += sf_why[i][j];
        if (!tw) continue;
        for (j = 0; j < SFD_N; j++) {
            if (!sf_why[i][j]) continue;
            fprintf(stderr, "  %s C-core: %-26s %10lu  %5.1f%%\n",
                    sf_opname[i], sf_whyname[j], sf_why[i][j],
                    100.0 * (double)sf_why[i][j] / (double)tw);
        }
    }
}
static int sf_stats_armed;
#define SF_BILL(op, took) do {                                   \
        if (!sf_stats_armed) { sf_stats_armed = 1;               \
                               atexit(sf_stats_dump); }          \
        if (took) sf_fast[op]++; else sf_slow[op]++;             \
    } while (0)

#else
#define SF_BILL(op, took) ((void)0)
#endif /* LZ_SOFTFP_STATS */

/* ------------------------------------------------------------------ */
/* shared helpers                                                     */
/* ------------------------------------------------------------------ */

static uint32_t sf_shift_right_jam32(uint32_t a, unsigned dist) {
    if (dist < 31) return a >> dist | ((uint32_t)(a << (-dist & 31)) != 0);
    return (a != 0);
}
static int sf_clz32(uint32_t a) { return a ? __builtin_clz(a) : 32; }

static uint32_t sf_round_pack_f32(int sign, int exp, uint32_t sig) {
#if LZ_SOFTFP_FAST
    /* Round toward zero plus flush-denormals, matching what x86's
       --fastfp MXCSR does (RC=truncate, FTZ, DAZ). Truncation: no
       roundIncrement and no tie-to-even mask - the dropped bits are
       dropped. FTZ: a subnormal result (exp reaches 0 with sig left)
       flushes to +-0 exactly as the hardware does under FTZ. And
       overflow CLAMPS to max_normal instead of rounding up to inf:
       round-toward-zero never rounds up, so an inf result can only
       come from an inf input, which the callers own. The callers
       normalise sig below 0x80000000, so the only overflow reachable
       here is exp == 0xFE (the 2^128 binade, above max_normal). */
    if (0xFD <= (unsigned int)exp) {
        if (exp < 0) {
            sig = sf_shift_right_jam32(sig, -exp);
            exp = 0;
        } else if (0xFD < exp) {
            return ((sign ? 0x80000000u : 0) | 0x7F7FFFFFu);   /* clamp */
        }
    }
    sig = sig >> 7;   /* truncate toward zero */
    if (sig >= 0x1000000u)
        return ((sign ? 0x80000000u : 0) | 0x7F7FFFFFu);      /* clamp */
    if (!sig || exp == 0) return ((uint32_t)sign << 31);   /* zero, or FTZ */
    return ((uint32_t)sign << 31) + ((uint32_t)exp << 23) + sig;
#else
    uint32_t roundIncrement = 0x40;
    uint32_t roundBits = sig & 0x7F;
    if (0xFD <= (unsigned int)exp) {
        if (exp < 0) {
            sig = sf_shift_right_jam32(sig, -exp);
            exp = 0;
            roundBits = sig & 0x7F;
        } else if ((0xFD < exp) || (0x80000000u <= sig + roundIncrement)) {
            return ((sign ? 0x80000000u : 0) | 0x7F800000u);
        }
    }
    sig = (sig + roundIncrement) >> 7;
    sig &= ~(uint32_t)((roundBits ^ 0x40) == 0);
    if (!sig) exp = 0;
    return ((uint32_t)sign << 31) + ((uint32_t)exp << 23) + sig;
#endif /* LZ_SOFTFP_FAST */
}

static uint32_t sf_norm_round_pack_f32(int sign, int exp, uint32_t sig) {
    int shiftDist = sf_clz32(sig) - 1;
    exp -= shiftDist;
    if ((7 <= shiftDist) && ((unsigned int)exp < 0xFD)) {
        return ((uint32_t)sign << 31) +
               ((uint32_t)(sig ? exp : 0) << 23) + (sig << (shiftDist - 7));
    } else {
        return sf_round_pack_f32(sign, exp, sig << shiftDist);
    }
}

/* ------------------------------------------------------------------ */
/* add / sub                                                          */
/* ------------------------------------------------------------------ */

static uint32_t sf_add_mags_f32(uint32_t uiA, uint32_t uiB) {
    int expA = (int)((uiA >> 23) & 0xFF), expB = (int)((uiB >> 23) & 0xFF);
    uint32_t sigA = uiA & 0x7FFFFF, sigB = uiB & 0x7FFFFF;
    int expDiff = expA - expB;
    int signZ = (int)(uiA >> 31);
    int expZ;
    uint32_t sigZ;
    if (!expDiff) {
        if (!expA) return uiA + sigB;
        if (expA == 0xFF) {
            if (sigA | sigB) return uiA | 0x00400000u;
            return uiA;
        }
        expZ = expA;
        sigZ = 0x01000000u + sigA + sigB;
        if (!(sigZ & 1) && (expZ < 0xFE)) {
            return ((uint32_t)signZ << 31) + ((uint32_t)expZ << 23) + (sigZ >> 1);
        }
        sigZ <<= 6;
    } else {
        signZ = (int)(uiA >> 31);
        sigA <<= 6; sigB <<= 6;
        if (expDiff < 0) {
            if (expB == 0xFF) {
                if (sigB) return uiB | 0x00400000u;
                return uiB;
            }
            expZ = expB;
            sigA += expA ? 0x20000000u : sigA;
            sigA = sf_shift_right_jam32(sigA, -expDiff);
        } else {
            if (expA == 0xFF) {
                if (sigA) return uiA | 0x00400000u;
                return uiA;
            }
            expZ = expA;
            sigB += expB ? 0x20000000u : sigB;
            sigB = sf_shift_right_jam32(sigB, expDiff);
        }
        sigZ = 0x20000000u + sigA + sigB;
        if (sigZ < 0x40000000u) { --expZ; sigZ <<= 1; }
    }
    return sf_round_pack_f32(signZ, expZ, sigZ);
}

static uint32_t sf_sub_mags_f32(uint32_t uiA, uint32_t uiB) {
    int expA = (int)((uiA >> 23) & 0xFF), expB = (int)((uiB >> 23) & 0xFF);
    uint32_t sigA = uiA & 0x7FFFFF, sigB = uiB & 0x7FFFFF;
    int expDiff = expA - expB;
    int signZ = (int)(uiA >> 31);
    int32_t sigDiff;
    int shiftDist, expZ;
    uint32_t sigX, sigY;

    if (!expDiff) {
        if (expA == 0xFF) {
            if (sigA | sigB) return uiA | 0x00400000u;
            return uiA | 0x7FC00000u;
        }
        sigDiff = (int32_t)sigA - (int32_t)sigB;
        if (!sigDiff) return 0;
        if (expA) --expA;
        signZ = (int)(uiA >> 31);
        if (sigDiff < 0) { signZ = !signZ; sigDiff = -sigDiff; }
        shiftDist = sf_clz32((uint32_t)sigDiff) - 8;
        expZ = expA - shiftDist;
#if LZ_SOFTFP_FAST
        /* FTZ: an exact difference that lands in the subnormal range
           (expZ <= 0 with sig nonzero) flushes to +-0, exactly as x86's
           FTZ flushes a subnormal result regardless of exactness. */
        if (expZ <= 0) return ((uint32_t)signZ << 31);
#else
        if (expZ < 0) { shiftDist = expA; expZ = 0; }
#endif /* LZ_SOFTFP_FAST */
        return ((uint32_t)signZ << 31) + ((uint32_t)expZ << 23)
               + (((uint32_t)sigDiff) << shiftDist);
    } else {
        signZ = (int)(uiA >> 31);
        sigA <<= 7; sigB <<= 7;
        if (expDiff < 0) {
            signZ = !signZ;
            if (expB == 0xFF) {
                if (sigB) return uiB | 0x00400000u;
                return ((uint32_t)signZ << 31) | 0x7F800000u;
            }
            expZ = expB - 1;
            sigX = sigB | 0x40000000u;
            sigY = sigA + (expA ? 0x40000000u : sigA);
            expDiff = -expDiff;
        } else {
            if (expA == 0xFF) {
                if (sigA) return uiA | 0x00400000u;
                return uiA;
            }
            expZ = expA - 1;
            sigX = sigA | 0x40000000u;
            sigY = sigB + (expB ? 0x40000000u : sigB);
        }
        return sf_norm_round_pack_f32(signZ, expZ,
                                      sigX - sf_shift_right_jam32(sigY, (unsigned)expDiff));
    }
}

/* ------------------------------------------------------------------ */
/* mul                                                                */
/* ------------------------------------------------------------------ */

static __attribute__((noinline)) uint32_t sf_mul_f32(uint32_t uiA, uint32_t uiB) {
    int signA = (int)(uiA >> 31), expA = (int)((uiA >> 23) & 0xFF);
    uint32_t sigA = uiA & 0x7FFFFF;
    int signB = (int)(uiB >> 31), expB = (int)((uiB >> 23) & 0xFF);
    uint32_t sigB = uiB & 0x7FFFFF;
    int signZ = signA ^ signB;
    uint32_t magA, magB;

    if (expA == 0xFF) {
        if (sigA) return uiA | 0x00400000u;
        if (expB == 0xFF) { if (sigB) return uiB | 0x00400000u; goto infZ; }
        if (!expB && !sigB) return uiA | 0x7FC00000u;  /* inf*0: NaN, sign from inf=a */
        goto infZ;
    }
    if (expB == 0xFF) {
        if (sigB) return uiB | 0x00400000u;
        if (!expA && !sigA) return uiB | 0x7FC00000u;  /* 0*inf: NaN, sign from inf=b */
        goto infZ;
    }
    if (!expA && !sigA) {
        if (expB == 0xFF) return uiB | 0x7FC00000u;
        return ((uint32_t)signZ << 31);
    }
    if (!expB && !sigB) {
        if (expA == 0xFF) return uiA | 0x7FC00000u;
        return ((uint32_t)signZ << 31);
    }
    magA = sigA; magB = sigB;
    if (!expA) { int n = sf_clz32(magA) - 8; expA = 1 - n; magA <<= n; }
    if (!expB) { int n = sf_clz32(magB) - 8; expB = 1 - n; magB <<= n; }
    {
        int expZ = expA + expB - 0x7F;
        uint64_t sig64;
        uint32_t sigZ;
        magA = (magA | 0x00800000u) << 7;
        magB = (magB | 0x00800000u) << 8;
        sig64 = (uint64_t)magA * magB;
        sigZ = (uint32_t)(sig64 >> 32 | ((uint32_t)(sig64 & 0xFFFFFFFFu) != 0));
        if (sigZ < 0x40000000u) { --expZ; sigZ <<= 1; }
        return sf_round_pack_f32(signZ, expZ, sigZ);
    }
 infZ:
    return ((uint32_t)signZ << 31) | 0x7F800000u;
}

/* ------------------------------------------------------------------ */
/* div (long division; ARMv5TE has no hardware divide unit)           */
/* ------------------------------------------------------------------ */

static uint32_t sf_udiv64_32(uint64_t n, uint32_t d, uint32_t *r) {
    uint32_t q = 0, rem = 0;
    int i;
    for (i = 31; i >= 0; i--) {
        rem = (rem << 1) | (uint32_t)((n >> (i + 32)) & 1u);
        if (rem >= d) { rem -= d; q |= (1u << i); }
    }
    for (i = 31; i >= 0; i--) {
        rem = (rem << 1) | (uint32_t)((n >> i) & 1u);
        if (rem >= d) { rem -= d; q |= (1u << i); }
    }
    if (r) *r = rem;
    return q;
}

static __attribute__((noinline)) uint32_t sf_div_f32(uint32_t uiA, uint32_t uiB) {
    int signA = (int)(uiA >> 31), expA = (int)((uiA >> 23) & 0xFF);
    uint32_t sigA = uiA & 0x7FFFFF;
    int signB = (int)(uiB >> 31), expB = (int)((uiB >> 23) & 0xFF);
    uint32_t sigB = uiB & 0x7FFFFF;
    int signZ = signA ^ signB;
    int expZ;
    uint32_t sigZ, rem;

    if (expA == 0xFF) {
        if (sigA) return uiA | 0x00400000u;
        if (expB == 0xFF) {
            if (sigB) return uiB | 0x00400000u;
            return uiB | 0x7FC00000u;
        }
        return ((uint32_t)signZ << 31) | 0x7F800000u;
    }
    if (expB == 0xFF) {
        if (sigB) return uiB | 0x00400000u;
        return ((uint32_t)signZ << 31);
    }
    if (!expB) {
        if (!sigB) {
            if (!(expA | sigA)) return uiA | 0x7FC00000u;
            return ((uint32_t)signZ << 31) | 0x7F800000u;
        }
        { int n = sf_clz32(sigB) - 8; expB = 1 - n; sigB <<= n; }
    }
    if (!expA) {
        if (!sigA) return ((uint32_t)signZ << 31);
        { int n = sf_clz32(sigA) - 8; expA = 1 - n; sigA <<= n; }
    }
    expZ = expA - expB + 0x7E;
    sigA |= 0x00800000u;
    sigB |= 0x00800000u;
    if (sigA < sigB) {
        --expZ;
        sigZ = sf_udiv64_32((uint64_t)sigA << 31, sigB, &rem);
    } else {
        sigZ = sf_udiv64_32((uint64_t)sigA << 30, sigB, &rem);
    }
    if (!(sigZ & 0x3F)) sigZ |= (rem != 0);
    return sf_round_pack_f32(signZ, expZ, sigZ);
}

/* ------------------------------------------------------------------ */
/* EABI arithmetic entry points                                       */
/* ------------------------------------------------------------------ */

float __aeabi_fadd(float a, float b) {
    union { float f; uint32_t u; } ua, ub, r;
    uint32_t S;
    ua.f = a; ub.f = b;
#if LZ_SOFTFP_FAST
    /* DAZ: a subnormal operand is treated as +-0 (sign preserved),
       matching x86's MXCSR DAZ. Done before the fast paths: their
       combined zero/subnormal guard cannot distinguish the two, and
       under DAZ the subnormal IS a zero, so the zero shortcut below is
       the right answer. */
    if ((ua.u & 0x7F800000u) == 0 && (ua.u & 0x007FFFFFu)) ua.u &= 0x80000000u;
    if ((ub.u & 0x7F800000u) == 0 && (ub.u & 0x007FFFFFu)) ub.u &= 0x80000000u;
#endif /* LZ_SOFTFP_FAST */
    /* fast path (normal operands, normal result): lz_fsum_arm_asm, the
       sum-only half of the engine's verified faddsub cell. The
       two-result form is fwht's; asking it for a sum computed the
       difference as well and threw it away, about fifteen instructions
       per call. Declines (returns 0) on subnormal/inf/nan operands and
       on subnormal results -- the C core below handles those. An
       exponent difference above 27 is NOT a decline any more; the cell
       answers it with the larger operand, as libgcc does. */
    { int took = lz_fsum_arm_asm(ua.u, ub.u, &S);
      SF_BILL(SF_ADD, took);
      if (took) { r.u = S; return r.f; } }
    /* A +-0 operand is 99.8% of every decline measured on the zh fixture
       (13,165,384 of 13,188,114): an accumulator starts at zero, and the
       fast path's combined 0/subnormal/inf/nan guard cannot tell a zero
       from a subnormal, so 0+x paid a call into the C core to compute x.
       IEEE has no rounding to do here - x+0 IS x - so it is answered in
       a handful of instructions instead. 0+0 is +0 unless BOTH are -0,
       which is the AND of the two sign bits. ub.u is the post-negation
       operand, so this is correct for fsub as written.

       The exponent test is NOT redundant: 0 + sNaN must QUIET the NaN
       (libgcc returns 0xfffb2aaf for 0 + 0xffbb2aaf, bit 22 set), so an
       inf/NaN partner cannot take this shortcut and goes to the C core.
       Caught by the libgcc comparison, not by reading the standard. */
    if (!(ub.u & 0x7FFFFFFFu) && (ua.u & 0x7F800000u) != 0x7F800000u) {
        if (ua.u & 0x7FFFFFFFu) return a;
        r.u = ua.u & ub.u & 0x80000000u;
        return r.f;
    }
    if (!(ua.u & 0x7FFFFFFFu) && (ub.u & 0x7F800000u) != 0x7F800000u) {
        r.u = ub.u;
        return r.f;
    }
#if LZ_SOFTFP_STATS
    /* Counted HERE, past the shortcut, so the tally is C-core entries -
       what the decline actually costs - not primitive declines. */
    sf_bill_why(SF_ADD, ua.u, ub.u);
#endif /* LZ_SOFTFP_STATS */
    if ((ua.u ^ ub.u) & 0x80000000u) r.u = sf_sub_mags_f32(ua.u, ub.u);
    else                             r.u = sf_add_mags_f32(ua.u, ub.u);
    return r.f;
}
float __aeabi_fsub(float a, float b) {
    /* a - b == a + (-b): flip b's sign, then the add dispatch's
       sign-difference test routes to add_mags/sub_mags correctly. */
    union { float f; uint32_t u; } ua, ub, r;
    uint32_t S;
    ua.f = a; ub.f = b;
    ub.u ^= 0x80000000u;
#if LZ_SOFTFP_FAST
    /* DAZ, as in __aeabi_fadd: subnormal operands are zeros. The sign
       flip above is preserved either way, so ordering is irrelevant. */
    if ((ua.u & 0x7F800000u) == 0 && (ua.u & 0x007FFFFFu)) ua.u &= 0x80000000u;
    if ((ub.u & 0x7F800000u) == 0 && (ub.u & 0x007FFFFFu)) ub.u &= 0x80000000u;
#endif /* LZ_SOFTFP_FAST */
    { int took = lz_fsum_arm_asm(ua.u, ub.u, &S);
      SF_BILL(SF_ADD, took);
      if (took) { r.u = S; return r.f; } }
    /* A +-0 operand is 99.8% of every decline measured on the zh fixture
       (13,165,384 of 13,188,114): an accumulator starts at zero, and the
       fast path's combined 0/subnormal/inf/nan guard cannot tell a zero
       from a subnormal, so 0+x paid a call into the C core to compute x.
       IEEE has no rounding to do here - x+0 IS x - so it is answered in
       a handful of instructions instead. 0+0 is +0 unless BOTH are -0,
       which is the AND of the two sign bits. ub.u is the post-negation
       operand, so this is correct for fsub as written.

       The exponent test is NOT redundant: 0 + sNaN must QUIET the NaN
       (libgcc returns 0xfffb2aaf for 0 + 0xffbb2aaf, bit 22 set), so an
       inf/NaN partner cannot take this shortcut and goes to the C core.
       Caught by the libgcc comparison, not by reading the standard. */
    if (!(ub.u & 0x7FFFFFFFu) && (ua.u & 0x7F800000u) != 0x7F800000u) {
        if (ua.u & 0x7FFFFFFFu) return a;
        r.u = ua.u & ub.u & 0x80000000u;
        return r.f;
    }
    if (!(ua.u & 0x7FFFFFFFu) && (ub.u & 0x7F800000u) != 0x7F800000u) {
        r.u = ub.u;
        return r.f;
    }
#if LZ_SOFTFP_STATS
    /* Counted HERE, past the shortcut, so the tally is C-core entries -
       what the decline actually costs - not primitive declines. */
    sf_bill_why(SF_ADD, ua.u, ub.u);
#endif /* LZ_SOFTFP_STATS */
    if ((ua.u ^ ub.u) & 0x80000000u) r.u = sf_sub_mags_f32(ua.u, ub.u);
    else                             r.u = sf_add_mags_f32(ua.u, ub.u);
    return r.f;
}
/* fmul fast path: umull + normalize + round, for normal operands with a
   normal result. 28 instructions; libgcc's ARM __aeabi_fmul spends 28 on
   the same shape (its 102 is the WHOLE function, subnormals and NaNs
   included - docs/arm-asm-audit.md records the fast paths separately, and
   quoting the 102 here once made a 40-instruction version look like a
   win). Declines (0) on zero/subnormal/inf/nan operands and on
   subnormal/inf results.

   THE SCALING IS THE WHOLE TRICK, and it is libgcc's (config/arm/
   ieee754-sf.S), reached here by way of the copy already in this tree at
   ops_arm.c's LZ_F32MM_PROD. Each significand is placed as a 28-bit
   quantity - implicit bit at 27, mantissa at 26..4 - so the 56-bit
   product sits with its leading 1 at bit 55 or 54. UMULL's HIGH word is
   then the 24-bit result significand ALREADY at bits 23..0, needing no
   final shift, and its LOW word is already exactly the round-and-sticky
   field, so `cmp lo, #0x80000000` IS the round-to-nearest-even test: C
   is the guard bit, Z is the exact tie. The earlier version put the
   leading 1 at bit 30, which bought nothing and cost a shift to extract
   the mantissa, a shift to extract round+sticky, and a separate
   teq/orrne to jam the sticky in.

   Three more foldings, none of them obvious:
     - The four-way operand guard is ONE predicate chain. ANDS sets Z on
       a zero exponent; ANDNES both skips itself and leaves Z set when
       that already happened; the two TEQNEs test for 255 only if
       everything before passed. Six instructions, against the twelve
       cmp/beq pairs the straightforward form needs.
     - SBC does the bias and the normalisation at once: the CMP above
       left C clear exactly when the product needed a left shift, and
       `sbc ea, ea, #127` subtracts 127+(1-C), i.e. 128 in that case.
     - The bias is 127 rather than 128 because the significand ORed into
       s still HAS its implicit bit, and ADC adding ea<<23 lets that bit
       carry into the exponent field. The same ADC folds in the rounding
       carry, which is why a mantissa that rounds up to 2^24 correctly
       becomes the next exponent - and 0x1.fffffep127 correctly becomes
       +inf, since the range test below runs before rounding. */
static __inline__ int sf_fmul_fast_arm_asm(uint32_t ua, uint32_t ub, uint32_t *out) {
    uint32_t ok, s, ea, ma, mb, hi, lo, t;
    __asm__(
        "mov    %[t], #255\n\t"
        "ands   %[ea], %[t], %[a], lsr #23\n\t"  /* Z: a zero/subnormal */
        "andnes %[hi], %[t], %[b], lsr #23\n\t"  /* Z: b zero/subnormal */
        "teqne  %[ea], %[t]\n\t"                 /* Z: a inf/nan */
        "teqne  %[hi], %[t]\n\t"                 /* Z: b inf/nan */
        "beq    9f\n\t"
        "add    %[ea], %[ea], %[hi]\n\t"
        "eor    %[s], %[a], %[b]\n\t"
        "mov    %[t], #0x8000000\n\t"      /* implicit bit, at 27 */
        "mov    %[ma], %[a], lsl #9\n\t"
        "orr    %[ma], %[t], %[ma], lsr #5\n\t"
        "mov    %[mb], %[b], lsl #9\n\t"
        "orr    %[mb], %[t], %[mb], lsr #5\n\t"
        "umull  %[lo], %[hi], %[ma], %[mb]\n\t"
        "and    %[s], %[s], #0x80000000\n\t"
        "cmp    %[hi], #0x800000\n\t"      /* C clear => needs one shift */
        "movcc  %[hi], %[hi], lsl #1\n\t"
        "orrcc  %[hi], %[hi], %[lo], lsr #31\n\t"
#if !LZ_SOFTFP_FAST
        "movcc  %[lo], %[lo], lsl #1\n\t"  /* only rounding reads lo */
#endif /* !LZ_SOFTFP_FAST */
        "orr    %[s], %[s], %[hi]\n\t"
        "sbc    %[ea], %[ea], #127\n\t"    /* bias and normalisation */
#if !LZ_SOFTFP_ASSUME_FINITE
        /* One unsigned compare for both ends: a subnormal result makes
           ea negative, which wraps to a huge unsigned. */
        "cmp    %[ea], #254\n\t"
        "bhs    9f\n\t"
#endif /* !LZ_SOFTFP_ASSUME_FINITE */
#if LZ_SOFTFP_FAST
        /* Round toward zero: drop the bits instead of examining them.
           ADD, not ADC - there is no carry to fold in, and the implicit
           bit still walks into the exponent field by itself. */
        "add    %[s], %[s], %[ea], lsl #23\n\t"
#else
        "cmp    %[lo], #0x80000000\n\t"    /* C = guard, Z = exact tie */
        "adc    %[s], %[s], %[ea], lsl #23\n\t"
        "biceq  %[s], %[s], #1\n\t"        /* round-half-to-even */
#endif /* LZ_SOFTFP_FAST */
        "mov    %[o], #1\n\t"
        "b      8f\n\t"
        "9:\n\t"
        "mov    %[o], #0\n\t"
        "8:\n\t"
        : [o] "=&r"(ok), [s] "=&r"(s), [ea] "=&r"(ea),
          [ma] "=&r"(ma), [mb] "=&r"(mb), [hi] "=&r"(hi), [lo] "=&r"(lo),
          [t] "=&r"(t)
        : [a] "r"(ua), [b] "r"(ub)
        : "cc");
    if (ok) *out = s;
    return (int)ok;
}
float __aeabi_fmul(float a, float b) {
    union { float f; uint32_t u; } ua, ub, r;
    uint32_t q;
    ua.f = a; ub.f = b;
#if LZ_SOFTFP_FAST
    /* DAZ, as in __aeabi_fadd: subnormal operands are zeros. The fmul
       fast path's operand guard declines on a subnormal, and under DAZ
       the subnormal is a zero, so the zero shortcut below answers it. */
    if ((ua.u & 0x7F800000u) == 0 && (ua.u & 0x007FFFFFu)) ua.u &= 0x80000000u;
    if ((ub.u & 0x7F800000u) == 0 && (ub.u & 0x007FFFFFu)) ub.u &= 0x80000000u;
#endif /* LZ_SOFTFP_FAST */
    { int took = sf_fmul_fast_arm_asm(ua.u, ub.u, &q);
      SF_BILL(SF_MUL, took);
      if (took) { r.u = q; return r.f; } }
    /* A +-0 operand is 1,298,510 of fmul's 1,298,514 declines on the zh
       fixture - the same accumulator-starts-at-zero shape fadd sees. The
       product IS +-0 with the sign XORed, no rounding involved, so it is
       answered here rather than in the C core. inf/NaN partners are
       excluded: inf*0 is invalid (NaN) and sNaN*0 must be quieted, both
       of which the C core owns. */
    if ((!(ua.u & 0x7FFFFFFFu) || !(ub.u & 0x7FFFFFFFu)) &&
        (ua.u & 0x7F800000u) != 0x7F800000u &&
        (ub.u & 0x7F800000u) != 0x7F800000u) {
        r.u = (ua.u ^ ub.u) & 0x80000000u;
        return r.f;
    }
#if LZ_SOFTFP_STATS
    sf_bill_why(SF_MUL, ua.u, ub.u);
#endif /* LZ_SOFTFP_STATS */
    r.u = sf_mul_f32(ua.u, ub.u);
    return r.f;
}

/* fdiv fast path: libgcc's radix-16 restoring division (4 quotient bits
   per iteration, 6 iterations). Normal operands with a NORMAL result
   only; declines (0) to sf_div_f32 on zero/subnormal/inf/nan operands
   and on subnormal/overflow results. The setup matches libgcc exactly:
   dividend/divisor = 0x10000000 | (mantissa << 9 >> 4), which the earlier
   bic/orr 24-bit-mantissa version got wrong (100% bit-wrong). */
static __inline__ int sf_fdiv_fast_arm_asm(uint32_t ua, uint32_t ub, uint32_t *out) {
    uint32_t ok, s, ea, eb, d, q, ip, t;
    __asm__(
        /* The same one predicate chain __aeabi_fmul uses: ANDS sets Z on
           a zero exponent, ANDNES both skips itself and leaves that Z
           alone, and the TEQNEs only ask about 255 if everything before
           passed. Six instructions rather than twelve. */
        "mov    %[t], #255\n\t"
        "ands   %[ea], %[t], %[a], lsr #23\n\t"  /* Z: a zero/subnormal */
        "andnes %[eb], %[t], %[b], lsr #23\n\t"  /* Z: b zero/subnormal */
        "teqne  %[ea], %[t]\n\t"                 /* Z: a inf/nan */
        "teqne  %[eb], %[t]\n\t"                 /* Z: b inf/nan */
        "beq    9f\n\t"
        "sub    %[ea], %[ea], %[eb]\n\t"   /* expA - expB */
        "eor    %[s], %[a], %[b]\n\t"      /* sign */
        "lsl    %[b], %[b], #9\n\t"
        "lsl    %[a], %[a], #9\n\t"        /* mantissas to bits 31..9 */
        "mov    %[t], #0x10000000\n\t"
        "orr    %[eb], %[t], %[b], lsr #4\n\t"  /* divisor: bit28 + mant>>4 */
        "orr    %[d], %[t], %[a], lsr #4\n\t"   /* dividend */
        "and    %[q], %[s], #0x80000000\n\t"    /* q accumulates sign */
        "cmp    %[d], %[eb]\n\t"
        "lslcc  %[d], %[d], #1\n\t"        /* align: dividend >= divisor */
        "adc    %[ea], %[ea], #125\n\t"    /* expZ = expA-expB+125+carry */
        /* One unsigned compare for both ends, as in __aeabi_fmul: a
           subnormal result makes expZ negative, which wraps high. */
        "cmp    %[ea], #254\n\t"
        "bhs    9f\n\t"
        "mov    %[ip], #0x800000\n\t"      /* quotient bit 23 */
        "1:\n\t"
        "cmp    %[d], %[eb]\n\t"
        "subcs  %[d], %[d], %[eb]\n\t"
        "orrcs  %[q], %[q], %[ip]\n\t"
        "cmp    %[d], %[eb], lsr #1\n\t"
        "subcs  %[d], %[d], %[eb], lsr #1\n\t"
        "orrcs  %[q], %[q], %[ip], lsr #1\n\t"
        "cmp    %[d], %[eb], lsr #2\n\t"
        "subcs  %[d], %[d], %[eb], lsr #2\n\t"
        "orrcs  %[q], %[q], %[ip], lsr #2\n\t"
        "cmp    %[d], %[eb], lsr #3\n\t"
        "subcs  %[d], %[d], %[eb], lsr #3\n\t"
        "orrcs  %[q], %[q], %[ip], lsr #3\n\t"
        /* TWO exits, which is libgcc's shape and the reason its loop is
           written with MOVNES rather than a second flag-setting shift.
           The first shift sets Z when the remainder has gone to zero -
           the division was exact and every quotient bit still to come is
           a zero already sitting in %[q]. Predicating the second shift
           on NE both skips it and LEAVES THAT Z ALONE, so the BNE below
           falls through. An unconditional `lsrs` here (what this used to
           be) overwrites the flag with the counter's own, and the loop
           then always runs its full six rounds - about seventy
           instructions spent on an answer already complete.
           The remainder cannot shift out of the register: the four
           conditional subtracts above leave it below %[eb]>>3, so
           `lsl #4` is zero only when the remainder itself was. */
        "movs   %[d], %[d], lsl #4\n\t"
        "movnes %[ip], %[ip], lsr #4\n\t"
        "bne    1b\n\t"
#if LZ_SOFTFP_FAST
        /* Round toward zero: drop the remainder entirely. ADD, not ADC -
           there is no round-up carry to fold in, matching fmul's RTZ
           branch. A subnormal result still declines via the range check
           above, so FTZ lives in the C core. */
        "add    %[q], %[q], %[ea], lsl #23\n\t"
#else
        "cmp    %[d], %[eb]\n\t"           /* remainder vs divisor: rounding */
        "adc    %[q], %[q], %[ea], lsl #23\n\t"
        "biceq  %[q], %[q], #1\n\t"        /* round-half-to-even */
#endif /* LZ_SOFTFP_FAST */
        "mov    %[o], #1\n\t"
        "b      8f\n\t"
        "9:\n\t"
        "mov    %[o], #0\n\t"
        "8:\n\t"
        : [o] "=&r"(ok), [s] "=&r"(s), [ea] "=&r"(ea), [eb] "=&r"(eb),
          [d] "=&r"(d), [q] "=&r"(q), [ip] "=&r"(ip), [t] "=&r"(t),
          [a] "+r"(ua), [b] "+r"(ub)
        :
        : "cc");
    if (ok) *out = q;
    return (int)ok;
}
float __aeabi_fdiv(float a, float b) {
    union { float f; uint32_t u; } ua, ub, r;
    uint32_t q;
    ua.f = a; ub.f = b;
#if LZ_SOFTFP_FAST
    /* DAZ, as in __aeabi_fadd: subnormal operands are zeros. The fdiv
       fast path declines on a subnormal divisor (its restoring division
       cannot take one), and under DAZ the subnormal is a zero, which
       the C core's divide-by-zero answer below owns. */
    if ((ua.u & 0x7F800000u) == 0 && (ua.u & 0x007FFFFFu)) ua.u &= 0x80000000u;
    if ((ub.u & 0x7F800000u) == 0 && (ub.u & 0x007FFFFFu)) ub.u &= 0x80000000u;
#endif /* LZ_SOFTFP_FAST */
    { int took = sf_fdiv_fast_arm_asm(ua.u, ub.u, &q);
      SF_BILL(SF_DIV, took);
      if (took) { r.u = q; return r.f; } }
#if LZ_SOFTFP_STATS
    sf_bill_why(SF_DIV, ua.u, ub.u);
#endif /* LZ_SOFTFP_STATS */
    r.u = sf_div_f32(ua.u, ub.u);
    return r.f;
}
/* gcc may emit __addsf3 (the internal name) rather than __aeabi_fadd
   for float addition on some targets; alias them. */
float __addsf3(float a, float b) { return __aeabi_fadd(a, b); }

/* ------------------------------------------------------------------ */
/* compare                                                            */
/* ------------------------------------------------------------------ */

static int sf_f32_is_nan(uint32_t u) {
    return ((u & 0x7F800000u) == 0x7F800000u) && (u & 0x7FFFFF);
}
static int sf_f32_cmp_lt(uint32_t a, uint32_t b) {
    uint32_t sa = a >> 31, sb = b >> 31;
    if (sf_f32_is_nan(a) || sf_f32_is_nan(b)) return 0;
    if (!(a & 0x7FFFFFFF) && !(b & 0x7FFFFFFF)) return 0;  /* +/-0.0: equal */
    if (sa != sb) return sa > sb;
    if (sa) return a > b;
    return a < b;
}

/* EABI C-language compare: -1/0/1 like __cmpsf2, with the same NaN
   convention (unordered -> 1). Used by __aeabi_cfcmpeq/cfrcmple, whose
   EABI contract is to set CPSR flags, not return the relation.

   NOT WORTH OPTIMISING, and the number rather than the impression says
   so. An audit proposed marking sf_f32_cmp_lt always_inline to save the
   BL and return below, at the price of duplicating its ~35-instruction
   body. Counting BLs in the linked ARM binary: __aeabi_cfcmpeq 0 sites,
   __aeabi_cfrcmple 0. Nothing in this engine reaches either, so nothing
   reaches here - gcc emits the value-returning fcmp* family instead
   (fcmpgt 95 sites, fcmplt 54, fcmpeq 36, fcmple 30, fcmpge 25). The
   two exist to satisfy the EABI, not because the compiler wants them.
   `used` is what keeps this one linked at all. */
static int sf_fcmp_rel(uint32_t a, uint32_t b) __attribute__((used));
static int sf_fcmp_rel(uint32_t a, uint32_t b) {
    if (sf_f32_is_nan(a) || sf_f32_is_nan(b)) return 1;  /* unordered */
    if (sf_f32_cmp_lt(a, b)) return -1;
    if (sf_f32_cmp_lt(b, a)) return 1;
    return 0;   /* equal, including +0 == -0 */
}

/* EABI cfcmpeq/cfrcmple: set Z/C/N flags like a CMP and return the
   (pushed) first-operand bit pattern, exactly libgcc's __aeabi_cfcmpeq.
   The compiler's flag tests (movlo/moveq/movhi) read them after the
   call. NaN is unordered: rel=1 -> cmp r0,#0 gives C=1,N=0,Z=0, so every
   ordered predicate sees false. */
__attribute__((naked)) int __aeabi_cfcmpeq(float a, float b) {
    __asm__(
        "push   {r0, r1, r2, r3, lr}\n\t"
        "bl     sf_fcmp_rel\n\t"
        "cmp    r0, #0\n\t"
        "cmnmi  r0, #0\n\t"
        "pop    {r0, r1, r2, r3, pc}\n\t"
    );
}
__attribute__((naked)) int __aeabi_cfrcmple(float a, float b) {
    __asm__(
        "mov    r2, r0\n\t"
        "mov    r0, r1\n\t"
        "mov    r1, r2\n\t"        /* swap first: r0=b, r1=a */
        "push   {r0, r1, r2, r3, lr}\n\t"  /* save the swapped r0 (=b) */
        "bl     sf_fcmp_rel\n\t"
        "cmp    r0, #0\n\t"
        "cmnmi  r0, #0\n\t"
        "pop    {r0, r1, r2, r3, pc}\n\t"
    );
}

/* EABI predicates: return 1/0, NaN unordered (all false). */
/* fcmpeq naked asm: equal bit patterns (r0==r1) or one +0.0 and one -0.0
   ((r0|r1)==0x80000000). NaN (either operand) -> 0. ~13 insns. Naked so
   gcc adds no frame-pointer spill around the asm. */
__attribute__((naked)) int __aeabi_fcmpeq(float a, float b) {
    __asm__(
        "mov    r2, r0, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    2f\n\t"              /* a is NaN */
        "mov    r2, r1, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    2f\n\t"              /* b is NaN */
        "cmp    r0, r1\n\t"
        "moveq  r0, #1\n\t"
        "bxeq   lr\n\t"              /* equal bit patterns */
        "orr    r2, r0, r1\n\t"
        "cmp    r2, #0x80000000\n\t"
        "moveq  r0, #1\n\t"          /* +0.0 vs -0.0 */
        "movne  r0, #0\n\t"
        "bx     lr\n\t"
        "2:\n\t"
        "mov    r0, #0\n\t"
        "bx     lr\n\t"
    );
}
/* fcmplt naked asm: sign-magnitude compare (libgcc cmpsf2's ordering).
   NaN or +/-0.0 -> 0; different signs -> negative < positive; same sign
   -> compare |a| vs |b| (reversed when both negative). ~20 insns. */
__attribute__((naked)) int __aeabi_fcmplt(float a, float b) {
    __asm__(
        "mov    r2, r0, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    5f\n\t"               /* a is NaN */
        "mov    r2, r1, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    5f\n\t"               /* b is NaN */
        "orr    r2, r0, r1\n\t"
        "bics   r2, r2, #0x80000000\n\t"
        "beq    5f\n\t"               /* both +/-0.0: not less */
        "mov    r2, r0, lsr #31\n\t"  /* sa */
        "mov    r3, r1, lsr #31\n\t"  /* sb */
        "teq    r2, r3\n\t"
        "bne    4f\n\t"               /* different signs */
        "bic    r2, r0, #0x80000000\n\t"  /* |a| */
        "bic    r3, r1, #0x80000000\n\t"  /* |b| */
        "cmp    r0, #0\n\t"
        "bmi    3f\n\t"               /* both negative */
        "cmp    r2, r3\n\t"
        "movlo  r0, #1\n\t"
        "movhs  r0, #0\n\t"
        "bx     lr\n\t"
        "3:\n\t"
        "cmp    r2, r3\n\t"
        "movhi  r0, #1\n\t"          /* |a| > |b| -> a < b (both neg) */
        "movls  r0, #0\n\t"
        "bx     lr\n\t"
        "4:\n\t"
        "cmp    r2, r3\n\t"          /* sa vs sb */
        "movhi  r0, #1\n\t"          /* a neg, b pos -> a < b */
        "movls  r0, #0\n\t"
        "bx     lr\n\t"
        "5:\n\t"
        "mov    r0, #0\n\t"
        "bx     lr\n\t"
    );
}
/* fcmple naked asm: a <= b = (a == b) || (a < b). NaN -> 0. ~28 insns. */
__attribute__((naked)) int __aeabi_fcmple(float a, float b) {
    __asm__(
        "mov    r2, r0, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    5f\n\t"               /* NaN -> 0 */
        "mov    r2, r1, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    5f\n\t"
        "cmp    r0, r1\n\t"
        "beq    3f\n\t"              /* equal bit patterns -> 1 */
        "orr    r2, r0, r1\n\t"
        "bics   r2, r2, #0x80000000\n\t"
        "beq    3f\n\t"               /* +0.0 vs -0.0: equal */
        "mov    r2, r0, lsr #31\n\t"  /* sa */
        "mov    r3, r1, lsr #31\n\t"  /* sb */
        "teq    r2, r3\n\t"
        "bne    4f\n\t"
        "bic    r2, r0, #0x80000000\n\t"  /* |a| */
        "bic    r3, r1, #0x80000000\n\t"  /* |b| */
        "cmp    r0, #0\n\t"
        "bmi    2f\n\t"
        "cmp    r2, r3\n\t"
        "movlo  r0, #1\n\t"          /* |a| < |b| -> a <= b */
        "movhs  r0, #0\n\t"
        "bx     lr\n\t"
        "2:\n\t"
        "cmp    r2, r3\n\t"
        "movhi  r0, #1\n\t"          /* |a| > |b| -> a < b (both neg) */
        "movls  r0, #0\n\t"
        "bx     lr\n\t"
        "4:\n\t"
        "cmp    r2, r3\n\t"
        "movhi  r0, #1\n\t"          /* a neg, b pos -> a < b */
        "movls  r0, #0\n\t"
        "bx     lr\n\t"
        "3:\n\t"
        "mov    r0, #1\n\t"          /* equal zeros */
        "bx     lr\n\t"
        "5:\n\t"
        "mov    r0, #0\n\t"
        "bx     lr\n\t"
    );
}
/* fcmpge naked asm: a >= b = (a == b) || (a > b). NaN -> 0. */
__attribute__((naked)) int __aeabi_fcmpge(float a, float b) {
    __asm__(
        "mov    r2, r0, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    5f\n\t"
        "mov    r2, r1, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    5f\n\t"
        "cmp    r0, r1\n\t"
        "beq    3f\n\t"
        "orr    r2, r0, r1\n\t"
        "bics   r2, r2, #0x80000000\n\t"
        "beq    3f\n\t"
        "mov    r2, r0, lsr #31\n\t"
        "mov    r3, r1, lsr #31\n\t"
        "teq    r2, r3\n\t"
        "bne    4f\n\t"
        "bic    r2, r0, #0x80000000\n\t"
        "bic    r3, r1, #0x80000000\n\t"
        "cmp    r0, #0\n\t"
        "bmi    2f\n\t"
        "cmp    r2, r3\n\t"
        "movhi  r0, #1\n\t"          /* |a| > |b| -> a >= b */
        "movls  r0, #0\n\t"
        "bx     lr\n\t"
        "2:\n\t"
        "cmp    r2, r3\n\t"
        "movlo  r0, #1\n\t"          /* |a| < |b| -> a > b (both neg) */
        "movhs  r0, #0\n\t"
        "bx     lr\n\t"
        "4:\n\t"
        "cmp    r2, r3\n\t"
        "movlo  r0, #1\n\t"          /* sa < sb (a pos, b neg) -> a > b */
        "movhs  r0, #0\n\t"
        "bx     lr\n\t"
        "3:\n\t"
        "mov    r0, #1\n\t"
        "bx     lr\n\t"
        "5:\n\t"
        "mov    r0, #0\n\t"
        "bx     lr\n\t"
    );
}
/* fcmpgt naked asm: a > b (mirror of fcmplt). */
__attribute__((naked)) int __aeabi_fcmpgt(float a, float b) {
    __asm__(
        "mov    r2, r0, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    5f\n\t"
        "mov    r2, r1, lsl #1\n\t"
        "cmp    r2, #0xFF000000\n\t"
        "bhi    5f\n\t"
        "orr    r2, r0, r1\n\t"
        "bics   r2, r2, #0x80000000\n\t"
        "beq    5f\n\t"
        "mov    r2, r0, lsr #31\n\t"
        "mov    r3, r1, lsr #31\n\t"
        "teq    r2, r3\n\t"
        "bne    4f\n\t"
        "bic    r2, r0, #0x80000000\n\t"
        "bic    r3, r1, #0x80000000\n\t"
        "cmp    r0, #0\n\t"
        "bmi    3f\n\t"
        "cmp    r2, r3\n\t"
        "movhi  r0, #1\n\t"          /* |a| > |b| -> a > b */
        "movls  r0, #0\n\t"
        "bx     lr\n\t"
        "3:\n\t"
        "cmp    r2, r3\n\t"
        "movlo  r0, #1\n\t"          /* |a| < |b| -> a > b (both neg) */
        "movhs  r0, #0\n\t"
        "bx     lr\n\t"
        "4:\n\t"
        "cmp    r2, r3\n\t"
        "movlo  r0, #1\n\t"          /* sa < sb (a pos, b neg) -> a > b */
        "movhs  r0, #0\n\t"
        "bx     lr\n\t"
        "5:\n\t"
        "mov    r0, #0\n\t"
        "bx     lr\n\t"
    );
}

/* ------------------------------------------------------------------ */
/* int <-> float                                                      */
/* ------------------------------------------------------------------ */

float __aeabi_i2f(int a) {
    /* the engine's verified lz_i32f_arm_asm: CLZ + barrel shifter + cond
       execution, vs the generic sf_norm_round_pack_f32 path it replaces.

       TWO BRANCHES, TWO COSTS. Quoting only the cheap one - "~14 insns
       for every int32" - is the mistake the header's "COST, COUNTED THE
       SAME WAY" note exists to prevent, and it is easy to make here
       because the cheap branch is the common one:

         |v| < 2^24  (clz >= 8)   14   exact, no rounding to do
         |v| >= 2^24 (clz <  8)   22   shifts right, then round-to-even

       libgcc's shared i2f/ui2f/l2f tail is 20 on BOTH branches (it feeds
       the 32-bit value through the 64-bit l2f path with the low word
       zeroed), so this wins the small-magnitude branch by 6 and loses
       the rounding branch by 2.

       WHICH DOMINATES IS NOW MEASURED, and it is not close: instrumenting
       this entry and running the zh fixture (kmr20, 8 tokens) counted
       647,458 conversions with |v| < 2^24 against 68 without - 99.99%
       take the cheap branch. The average is therefore 22 - 8*0.9999 =
       14.0 instructions against libgcc's flat 20.

       That reverses an audit's tentative recommendation to adopt
       libgcc's single-shape tail here: it would trade a 6-instruction
       win on essentially all traffic for a 2-instruction win on one call
       in ten thousand. The distribution is not an accident either -
       every call site is a dimension, an index, a loop bound or a
       quantised value, and none of those reach 16 million. */
    return lz_i32f_arm_asm(a);
}

/* i2f's C entry alias; gcc calls __aeabi_i2f directly. */
float __aeabi_ui2f(unsigned int v) {
    /* unsigned 32 -> float, RNE: lz_i32f_arm_asm's structure without the
       sign handling (bit 31 is a value bit here).

       Same two-branch shape as __aeabi_i2f above, and the same
       correction: this said "~14 insns vs libgcc's ~39 dynamic" while
       quoting only the exact branch, and libgcc's 39 was its stub plus
       the l2f tail counted whole. Measured the same way on both sides,
       11 on the exact branch (|v| < 2^24) and 19 when a right shift and
       a rounding step are needed, against libgcc's 20 either way. */
    union { float f; uint32_t u; } b;
    uint32_t res, a, n, sh;
    __asm__(
        "movs   %[a], %[v]\n\t"
        "moveq  %[r], #0\n\t"
        "beq    1f\n\t"
        "clz    %[n], %[a]\n\t"
        "rsb    %[r], %[n], #157\n\t"    /* biased exponent - 1 */
        "mov    %[r], %[r], lsl #23\n\t"
        "subs   %[s], %[n], #8\n\t"
        "addpl  %[r], %[r], %[a], lsl %[s]\n\t"
        "bpl    2f\n\t"
        "rsb    %[s], %[n], #8\n\t"      /* 1..8 bits to drop */
        "add    %[r], %[r], %[a], lsr %[s]\n\t"
#if LZ_SOFTFP_FAST
        /* Round toward zero, as in lz_i32f_arm_asm: the dropped bits are
           dropped, matching cvtsi2ss under x86's MXCSR RC=truncate. */
#else
        "rsb    %[n], %[s], #32\n\t"
        "mov    %[n], %[a], lsl %[n]\n\t"    /* dropped bits, left-justified */
        "cmp    %[n], #0x80000000\n\t"
        "addhi  %[r], %[r], #1\n\t"
        "andeq  %[n], %[r], #1\n\t"
        "addeq  %[r], %[r], %[n]\n\t"
#endif /* LZ_SOFTFP_FAST */
        "2:\n\t"
        "1:\n\t"
        : [r] "=&r"(res), [a] "=&r"(a), [n] "=&r"(n), [s] "=&r"(sh)
        : [v] "r"(v)
        : "cc");
    b.u = res;
    return b.f;
}

/* l2f COMPLETE asm, NAKED (asm is the whole function, no gcc wrapper):
   signed int64 -> float32, RNE. Implements the standard clz/shift/RNE
   algorithm (same constants as libgcc's __aeabi_l2f -- they are IEEE-
   derived). 31 insns, matching libgcc. The 64-bit arg arrives in r0:r1,
   the float result returns in r0, via bx lr. Every nonzero int64 maps to
   a normal float, so there is no decline; zero returns +0. */
__attribute__((naked)) float __aeabi_l2f(lz_i64 a) {
    __asm__(
        "orrs   r2, r0, r1\n\t"
        "bxeq   lr\n\t"                      /* zero: r0 = 0 */
        "ands   r3, r1, #0x80000000\n\t"      /* r3 = sign */
        "bpl    1f\n\t"
        "rsbs   r0, r0, #0\n\t"
        "rsc    r1, r1, #0\n\t"              /* negate to magnitude */
        "1:\n\t"
        "movs   ip, r1\n\t"
        "moveq  ip, r0\n\t"                  /* ip = word with the leading 1 */
        "moveq  r1, r0\n\t"
        "moveq  r0, #0\n\t"                  /* 32-bit: no rounding word */
        "orr    r3, r3, #0x5B000000\n\t"
        "subeq  r3, r3, #0x10000000\n\t"
        "sub    r3, r3, #0x800000\n\t"
        "clz    r2, ip\n\t"
        "subs   r2, r2, #8\n\t"
        "sub    r3, r3, r2, lsl #23\n\t"     /* for r2<0 this adds */
        "blt    2f\n\t"                      /* shift-right path */
#if LZ_SOFTFP_FAST
        /* Round toward zero, as in lz_i32f_arm_asm: the dropped bits are
           dropped, matching x86's cvtsi2ss under MXCSR RC=truncate. */
        "add    r3, r3, r1, lsl r2\n\t"
        "rsb    r2, r2, #32\n\t"
        "add    r0, r3, r0, lsr r2\n\t"
        "bx     lr\n\t"
        "2:\n\t"                             /* r2 < 0 */
        "add    r2, r2, #32\n\t"
        "rsb    r2, r2, #32\n\t"
        "add    r0, r3, r1, lsr r2\n\t"
        "bx     lr\n\t"
#else
        "add    r3, r3, r1, lsl r2\n\t"
        "lsl    ip, r0, r2\n\t"
        "rsb    r2, r2, #32\n\t"
        "cmp    ip, #0x80000000\n\t"
        "adc    r0, r3, r0, lsr r2\n\t"
        "biceq  r0, r0, #1\n\t"              /* round-half-to-even */
        "bx     lr\n\t"
        "2:\n\t"                             /* r2 < 0 */
        "add    r2, r2, #32\n\t"
        "lsl    ip, r1, r2\n\t"
        "rsb    r2, r2, #32\n\t"
        "orrs   r0, r0, ip, lsl #1\n\t"
        "adc    r0, r3, r1, lsr r2\n\t"
        "biceq  r0, r0, ip, lsr #31\n\t"
        "bx     lr\n\t"
#endif /* LZ_SOFTFP_FAST */
    );
}

/* f2iz COMPLETE asm (no decline): float -> int32, truncate toward zero,
   saturate on |a| >= 2^31, NaN -> 0. Ported structure of libgcc's 23-insn
   __aeabi_f2iz (the constants are IEEE-derived). FALLS THROUGH to the C
   return (no bx lr) so it is safe to inline. ~25 insns. */
/* NAKED, and in libgcc's own layout, which is worth three instructions
   on the taken path - 15 down to 12. All three came from the wrapper
   rather than the algorithm, which was already a faithful port:

     - gcc bound the inline-asm input to a fresh register and emitted a
       `mov r3, r0` to get there. A naked function reads r0 directly.
     - `movcc r, #0` ran on EVERY call, predicated false on all but the
       |a| < 1 case. libgcc puts the `mov r0, #0` at the branch target,
       where only that case pays for it.
     - the arms branched to one shared exit. Each returns where it is
       now, which a naked function can do and a managed one cannot -
       `bx lr` from inside a normal function would skip the epilogue.

   Argument in r0 by the soft-float ABI, result in r0. r1-r3 and ip are
   call-clobbered, so nothing needs saving. */
__attribute__((naked)) int __aeabi_f2iz(float a) {
    __asm__(
        "mov    r1, r0, lsl #1\n\t"
        "cmp    r1, #0x7F000000\n\t"
        "bcc    5f\n\t"
        "mov    r2, #158\n\t"
        "subs   r1, r2, r1, lsr #24\n\t"
        "bls    2f\n\t"
        "mov    r3, r0, lsl #8\n\t"
        "orr    r3, r3, #0x80000000\n\t"
        "tst    r0, #0x80000000\n\t"
        "lsr    r0, r3, r1\n\t"
        "rsbne  r0, r0, #0\n\t"
        "bx     lr\n\t"
        "2:\n\t"
        "cmn    r1, #97\n\t"
        "bne    3f\n\t"
        "lsls   r3, r0, #9\n\t"
        "bne    5f\n\t"
        "3:\n\t"
        "ands   r0, r0, #0x80000000\n\t"
        "mvneq  r0, #0x80000000\n\t"
        "bx     lr\n\t"
        "5:\n\t"
        "mov    r0, #0\n\t"
        "bx     lr\n\t"
    );
}

/* __aeabi_f2lz: float to signed 64-BIT integer. The `l` in the EABI's
   name is LONG LONG, not long, and getting that wrong is not a matter of
   range - it is a matter of REGISTERS. A 64-bit return comes back in
   r0:r1. Declare this `long` and forward it to __aeabi_f2iz and it
   writes r0 while leaving r1 holding whatever the caller's last
   computation put there - every result the right low word with a
   garbage high word:

       (lz_i64)2048.0f  ==  -9223372036854773760

   and the tempting defence - "the engine's f2lz sites are all in range"
   - answers the wrong question, since range is not what breaks. It went unnoticed because nothing reached it: json.c held its
   numbers as double, so safetensors' data_offsets conversion went to
   libgcc's __aeabi_d2lz, which this file does not define. Narrowing that
   field to float pointed every tensor offset straight at this function,
   and the loader started rejecting offsets of 0..2048 for being
   negative.

   Straight shift-and-place, no loop. Below 1.0 (exponent < 127, which
   also covers zero and the subnormals) the truncation is 0; at or past
   2^63 it saturates, except for -2^63 exactly, which IS representable.
   NaN gives 0, matching libgcc and __aeabi_f2iz above.

   OUT OF RANGE THIS DELIBERATELY DIFFERS FROM libgcc, the one place in
   this file that does, so it is worth being exact about. C leaves a
   float-to-integer conversion undefined once the value will not fit, and
   libgcc's __fixsfdi is C rather than the hand-written assembly its
   f2iz uses - so it simply WRAPS, and 2^64 comes back as -1. That is not
   a contract, it is an artifact. This saturates instead, which is what
   the rest of this file's integer conversions do and what a loader
   validating a length wants. Measured against libgcc: 0 of 3,171,248
   in-range cases differ, and 1,016,248 of 1,031,758 out-of-range ones
   do, by construction. IN RANGE IS THE PART THAT IS OWED. */
/* LZ_I64_MAX/MIN rather than a literal with an LL suffix. `LL` is a
   syntax error on the 1995 compiler this tree keeps a floor for - it
   spells the suffix i64 and rejects LL outright with "bad suffix on
   number", taking the whole expression with it - which is why
   src/lz_int.h supplies LZ_I64_C. MIN is built by negating MAX and
   stepping one further rather than written out, because the literal
   9223372036854775808 does not fit a signed 64-bit type and a compiler
   is entitled to widen it or complain. */
#define LZ_I64_MAX   LZ_I64_C(9223372036854775807)
#define LZ_I64_MIN   (-LZ_I64_MAX - 1)

lz_i64 __aeabi_f2lz(float a) {
    union { float f; uint32_t u; } ua;
    uint32_t e, sig;
    int sh;
    lz_i64 r;
    ua.f = a;
    e = (ua.u >> 23) & 0xFFu;
    if (e < 127u) return 0;                  /* |a| < 1, zero, subnormal */
    sig = (ua.u & 0x7FFFFFu) | 0x800000u;    /* 24-bit significand */
    sh = (int)e - 127 - 23;                  /* place its low bit */
    if (sh >= 40) {                          /* 2^63 and above, or inf/NaN */
        if (e == 0xFFu && (ua.u & 0x7FFFFFu)) return 0;          /* NaN */
        /* -2^63 is exactly representable and is not an overflow. */
        if ((ua.u & 0x80000000u) && sh == 40 && sig == 0x800000u)
            return LZ_I64_MIN;
        return (ua.u & 0x80000000u) ? LZ_I64_MIN : LZ_I64_MAX;
    }
    if (sh >= 0) r = (lz_i64)(lz_u64)sig << sh;
    else         r = (lz_i64)(sig >> (-sh));
    return (ua.u & 0x80000000u) ? -r : r;
}

/* INTEGER DIVISION IS libgcc's, DELIBERATELY, and this note is here
   because the obvious argument for taking it over is wrong.

   That argument is that "pulling one libgcc member drags its neighbours
   in", so taking __udivsi3 would undo the point of this file. It does
   not. An archive member is pulled only to satisfy an UNDEFINED symbol,
   and every soft-float symbol is already defined by this object, which
   the link line places first. Measured by building both ways: with the
   five divmod routines removed the binary still contains ZERO
   __aeabi_d* / __adddf3 / __muldf3 symbols, and __aeabi_fmul is still
   this file's (the ANDS/ANDNES/TEQNE guard chain is its signature).
   Only _udivsi3.o and _udivmoddi4.o come across, and they reference no
   float routine.

   Removing them is not neutral, it is a large win. Ours was a
   fixed-trip restoring loop - 32 iterations for 32/32, 64 for 64/64,
   regardless of operand size. libgcc's is CLZ-normalised and runs only
   as many steps as the quotient has bits. Executed instructions under
   qemu (-one-insn-per-tb, the instrument .prof/arm_div_count.sh uses,
   loop overhead subtracted):

       shape                      ours   libgcc
       small quotient (indices)    210       39     5.3x
       large quotient              180       82     2.2x
       n < d                       236        7      34x
       64/64                      1246      145     8.6x

   The engine has 129 integer-divide call sites, all of them index,
   stride and block arithmetic where the small-quotient row is the
   normal case. There is no rounding in an integer divide, so both
   compute the identical quotient and remainder; this is a pure speed
   change with nothing to verify beyond the link.

   IT COSTS 828 BYTES of .text (431,658 -> 432,486), because libgcc
   trades code for speed the other way round: its 32 steps are unrolled
   into a table entered partway by `addne pc, pc, curbit, lsl #2`, where
   ours was a loop. That is the right trade here - the data segment is
   416KB and the target has 64MB - but it is a trade and not a free
   lunch, so it is recorded rather than left for someone to rediscover.

   ONE BEHAVIOUR DIFFERS, in the right direction: a hand-written divide
   here returns 0/0 for a zero divisor while libgcc traps. x86 traps
   too, on the hardware instruction, so taking libgcc's is what makes
   the two targets agree instead of differing silently. */

#endif /* __arm__ && __GNUC__ */
