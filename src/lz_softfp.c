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
   bit-identity and can only ever be slower, never different. The fast
   paths run ~14-80 instructions per op, down from libgcc's 23-102.

   EVERY FUNCTION is verified against libgcc (via qemu-arm-static on the
   arm-linux-gnueabi cross toolchain) on ~40M random bit patterns plus a
   directed special-value matrix: 0/+-0, subnormals, inf, NaN, overflow/
   underflow edges, and the rounding boundaries (round-to-nearest-even,
   the PC=24 precision the engine's x87 arm also enforces).

   LZ_SOFTFP_ASSUME_FINITE (default 0) drops the fast path's overflow and
   underflow EXPONENT checks - the two branches that decline a result too
   big or too small to be a normal binary32. It does NOT touch the
   operand checks: those decline on a zero, subnormal, inf or NaN INPUT,
   and inputs are not what this switch makes a claim about.

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
   otherwise (kmr20, zh fixture, 603 tokens):

     fmul  149,503,180 fast   1,298,514 declined   0.861%
     fadd   45,215,879 fast  13,188,114 declined  22.581%
     fdiv      106,048 fast         613 declined   0.575%

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
/* LZ_SOFTFP_FAST (default 0) trades round-to-nearest-even for round
   toward zero in the fmul fast path. Measured saving: 5 instructions -
   sticky extraction (2), the guard-bit compare (1), the round-bit
   extract (1) and the tie-to-even mask (1). fmul goes 62 -> 57.

   IT DOES NOT YET PAIR WITH --fastfp, and the gap is measured, not
   suspected. The intent was that --fastfp (MXCSR round-toward-zero plus
   FTZ/DAZ on x86, _RC_CHOP on x87) and this switch would describe the
   same arithmetic, keeping a fast-mode ARM run byte-comparable with a
   fast-mode x86 run the way the strict pair already is. Run against the
   zh fixture they are NOT:

     ARM fast  vs ARM strict   57,737,364 of 78,905,344 bytes differ
     x86 fast  vs x86 strict   60,158,216
     ARM fast  vs x86 fast     60,173,584      <- should have been 0

   The cause is coverage, not correctness. MXCSR is GLOBAL: it retimes
   every SSE operation in the process, including fadd, fdiv and the SIMD
   kernels. This switch changes one operator. Closing it means carrying
   round-toward-zero through faddsub (a primitive fwht.c shares), fdiv,
   and the C cores behind all three, plus flushing denormals to match
   FTZ/DAZ - a much larger change than the 5 instructions it buys per
   multiply.

   So today this is an ARM-only knob whose output pairs with nothing.
   Leave it at 0 unless that work is finished; --fastfp on the other
   targets is likewise still waiting for it.

   It does NOT skip the inf/NaN operand checks, and that is not an
   oversight: MXCSR changes rounding and denormal handling only - the
   hardware still produces IEEE inf and NaN - so an ARM path that skipped
   them could never match the build it is meant to be compared against.
*/
#ifndef LZ_SOFTFP_FAST
#define LZ_SOFTFP_FAST 0
#endif /* !LZ_SOFTFP_FAST */

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
        if (expZ < 0) { shiftDist = expA; expZ = 0; }
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
    uint32_t S, D;
    ua.f = a; ub.f = b;
    /* fast path (normal operands, expDiff <= 27, normal result): the
       engine's verified lz_faddsub_arm_asm, ~60 insns vs libgcc's 100.
       Declines (returns 0) on subnormal/inf/nan operands, expDiff > 27
       and subnormal results -- the C core below handles those. S is the
       sum (lz_faddsub already routes the sign-difference case), D the
       difference; take S unconditionally. */
    { int took = lz_faddsub_arm_asm(ua.u, ub.u, &S, &D);
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
    uint32_t S, D;
    ua.f = a; ub.f = b;
    ub.u ^= 0x80000000u;
    { int took = lz_faddsub_arm_asm(ua.u, ub.u, &S, &D);
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
   normal result. ~30 insns vs libgcc's 102. Declines (0) on zero/
   subnormal/inf/nan operands and subnormal/inf results. */
static inline int sf_fmul_fast_arm_asm(uint32_t ua, uint32_t ub, uint32_t *out) {
    uint32_t ok, s, ea, eb, ma, mb, hi, lo, t;
    __asm__(
        "mov    %[ea], %[a], lsr #23\n\t"
        "and    %[ea], %[ea], #255\n\t"
        "mov    %[eb], %[b], lsr #23\n\t"
        "and    %[eb], %[eb], #255\n\t"
        "cmp    %[ea], #0\n\t"
        "beq    9f\n\t"                    /* a zero/subnormal */
        "cmp    %[eb], #0\n\t"
        "beq    9f\n\t"                    /* b zero/subnormal */
        "cmp    %[ea], #255\n\t"
        "beq    9f\n\t"                    /* a inf/nan */
        "cmp    %[eb], #255\n\t"
        "beq    9f\n\t"                    /* b inf/nan */
        "eor    %[s], %[a], %[b]\n\t"
        "and    %[s], %[s], #0x80000000\n\t"
        "add    %[ea], %[ea], %[eb]\n\t"
        "sub    %[ea], %[ea], #127\n\t"    /* expZ */
        "bic    %[ma], %[a], #0xFF000000\n\t"
        "orr    %[ma], %[ma], #0x800000\n\t"
        "bic    %[mb], %[b], #0xFF000000\n\t"
        "orr    %[mb], %[mb], #0x800000\n\t"
        "mov    %[ma], %[ma], lsl #7\n\t"
        "mov    %[mb], %[mb], lsl #8\n\t"
        "umull  %[lo], %[hi], %[ma], %[mb]\n\t"
#if !LZ_SOFTFP_FAST
        "teq    %[lo], #0\n\t"
        "orrne  %[hi], %[hi], #1\n\t"      /* sticky */
#endif /* !LZ_SOFTFP_FAST */
        "cmp    %[hi], #0x40000000\n\t"
        "movcc  %[hi], %[hi], lsl #1\n\t"
        "subcc  %[ea], %[ea], #1\n\t"      /* normalize: leading 1 at bit 30 */
#if !LZ_SOFTFP_ASSUME_FINITE
        "cmp    %[ea], #0\n\t"
        "blt    9f\n\t"                    /* subnormal result */
        "cmp    %[ea], #253\n\t"
        "bhi    9f\n\t"                    /* overflow to inf */
#endif /* !LZ_SOFTFP_ASSUME_FINITE */
#if LZ_SOFTFP_FAST
        /* Round toward zero: drop the bits instead of examining
           them. No sticky, no guard compare, no tie-to-even - five
           instructions less than the RNE tail below. ADD, not ADC:
           there is no carry to fold in, and the implicit bit still
           walks into the exponent field by itself. */
        "mov    %[hi], %[hi], lsr #7\n\t"  /* 24-bit mag at bits 23..0 */
        "orr    %[s], %[s], %[hi]\n\t"
        "add    %[s], %[s], %[ea], lsl #23\n\t"
#else
        "mov    %[t], %[hi], lsl #25\n\t"  /* round+sticky bits 31..25 */
        "mov    %[hi], %[hi], lsr #7\n\t"  /* 24-bit mag at bits 23..0 */
        "orr    %[s], %[s], %[hi]\n\t"
        "cmp    %[t], #0x80000000\n\t"
        "adc    %[s], %[s], %[ea], lsl #23\n\t" /* + expZ<<23, round-up in C */
        "biceq  %[s], %[s], #1\n\t"        /* round-half-to-even */
#endif /* LZ_SOFTFP_FAST */
        "mov    %[o], #1\n\t"
        "b      8f\n\t"
        "9:\n\t"
        "mov    %[o], #0\n\t"
        "8:\n\t"
        : [o] "=&r"(ok), [s] "=&r"(s), [ea] "=&r"(ea), [eb] "=&r"(eb),
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
static inline int sf_fdiv_fast_arm_asm(uint32_t ua, uint32_t ub, uint32_t *out) {
    uint32_t ok, s, ea, eb, d, q, ip, t;
    __asm__(
        "mov    %[t], #255\n\t"
        "and    %[ea], %[t], %[a], lsr #23\n\t"
        "and    %[eb], %[t], %[b], lsr #23\n\t"
        "cmp    %[ea], #0\n\t"
        "beq    9f\n\t"                    /* a zero/subnormal */
        "cmp    %[eb], #0\n\t"
        "beq    9f\n\t"                    /* b zero/subnormal */
        "cmp    %[ea], #255\n\t"
        "beq    9f\n\t"                    /* a inf/nan */
        "cmp    %[eb], #255\n\t"
        "beq    9f\n\t"                    /* b inf/nan */
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
        "cmp    %[ea], #0\n\t"
        "blt    9f\n\t"                    /* subnormal result: decline */
        "cmp    %[ea], #253\n\t"
        "bhi    9f\n\t"                    /* overflow to inf: decline */
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
        "lsls   %[d], %[d], #4\n\t"
        "lsrs   %[ip], %[ip], #4\n\t"
        "bne    1b\n\t"
        "cmp    %[d], %[eb]\n\t"           /* remainder vs divisor: rounding */
        "adc    %[q], %[q], %[ea], lsl #23\n\t"
        "biceq  %[q], %[q], #1\n\t"        /* round-half-to-even */
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
   EABI contract is to set CPSR flags, not return the relation. */
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
       execution, ~14 insns for every int32 (INT32_MIN included), vs the
       generic sf_norm_round_pack_f32 path it replaces. */
    return lz_i32f_arm_asm(a);
}

/* i2f's C entry alias; gcc calls __aeabi_i2f directly. */
float __aeabi_ui2f(unsigned int v) {
    /* unsigned 32 -> float, RNE: lz_i32f_arm_asm's structure without the
       sign handling (bit 31 is a value bit here). ~14 insns vs libgcc's
       ui2f stub + l2f tail (~39 dynamic). */
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
        "rsb    %[n], %[s], #32\n\t"
        "mov    %[n], %[a], lsl %[n]\n\t"    /* dropped bits, left-justified */
        "cmp    %[n], #0x80000000\n\t"
        "addhi  %[r], %[r], #1\n\t"
        "andeq  %[n], %[r], #1\n\t"
        "addeq  %[r], %[r], %[n]\n\t"
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
    );
}

/* f2iz COMPLETE asm (no decline): float -> int32, truncate toward zero,
   saturate on |a| >= 2^31, NaN -> 0. Ported structure of libgcc's 23-insn
   __aeabi_f2iz (the constants are IEEE-derived). FALLS THROUGH to the C
   return (no bx lr) so it is safe to inline. ~25 insns. */
static inline int32_t sf_f2iz_arm_asm(uint32_t u) {
    int32_t r;
    uint32_t e, m, t;
    __asm__(
        "mov    %[e], %[u], lsl #1\n\t"      /* |u|<<1: exp at 31:24 */
        "cmp    %[e], #0x7F000000\n\t"
        "movcc  %[r], #0\n\t"
        "bcc    9f\n\t"                      /* |a| < 1: 0 */
        "mov    %[t], #158\n\t"
        "subs   %[e], %[t], %[e], lsr #24\n\t" /* e = 158 - exp */
        "bls    2f\n\t"                      /* exp >= 158: overflow/inf/nan */
        "mov    %[m], %[u], lsl #8\n\t"
        "orr    %[m], %[m], #0x80000000\n\t" /* sig<<8 | implicit at 31 */
        "tst    %[u], #0x80000000\n\t"
        "lsr    %[r], %[m], %[e]\n\t"
        "rsbne  %[r], %[r], #0\n\t"          /* truncate, negate if negative */
        "b      9f\n\t"
        "2:\n\t"                             /* exp in [158, 255] */
        "cmn    %[e], #97\n\t"               /* e == -97 (exp 255)? */
        "bne    3f\n\t"
        "lsls   %[m], %[u], #9\n\t"          /* mantissa bit 31 */
        "bne    4f\n\t"                      /* NaN: 0 */
        "3:\n\t"                             /* inf or |a| >= 2^31: saturate */
        "ands   %[r], %[u], #0x80000000\n\t"
        "mvneq  %[r], #0x80000000\n\t"       /* INT_MIN / INT_MAX */
        "b      9f\n\t"
        "4:\n\t"
        "mov    %[r], #0\n\t"                /* NaN */
        "9:\n\t"
        : [r] "=&r"(r), [e] "=&r"(e), [m] "=&r"(m), [t] "=&r"(t)
        : [u] "r"(u)
        : "cc");
    return r;
}

int __aeabi_f2iz(float a) {
    union { float f; uint32_t u; } ua; ua.f = a;
    return sf_f2iz_arm_asm(ua.u);
}

/* __aeabi_f2lz: float to signed long. On ARM EABI "long" is 32-bit, so the
   in-range truncation behaviour is identical to f2iz; reuse it. The only
   divergence is out of int32 range, where libgcc wraps (mod 2^32, via a
   64-bit f2ulz) and this saturates - the engine's only f2lz sites (json
   shape ints, the kda int16 rounding) are in range, and x86's x87 already
   differs from libgcc there anyway. */
long __aeabi_f2lz(float a) {
    return __aeabi_f2iz(a);
}

/* ------------------------------------------------------------------ */
/* integer division (EABI divmod), no libgcc                          */
/* ------------------------------------------------------------------ */

/* __aeabi_uidivmod: unsigned 32/32, returns quotient in r0 and
   remainder in r1 (the EABI divmod convention). Restoring long division,
   32 iterations. d == 0 returns 0/0 (the engine never divides by zero;
   libgcc raises SIGFPE there, which we deliberately do not reproduce). */
__attribute__((naked)) unsigned __aeabi_uidivmod(unsigned n, unsigned d) {
    __asm__(
        "movs   r2, r1\n\t"        /* r2 = d, Z if d == 0 */
        "beq    2f\n\t"
        "cmp    r2, #1\n\t"
        "beq    3f\n\t"            /* d == 1: quot = n, rem = 0 */
        "mov    r1, #0\n\t"        /* rem */
        "mov    r3, r0\n\t"        /* r3 = n (shifted in) */
        "mov    r0, #0\n\t"        /* quot */
        "mov    ip, #31\n\t"
        "1:\n\t"
        "movs   r3, r3, lsl #1\n\t"  /* shift n, C = top bit */
        "adc    r1, r1, r1\n\t"      /* rem = rem<<1 | C */
        "cmp    r1, r2\n\t"
        "subcs  r1, r1, r2\n\t"
        "adc    r0, r0, r0\n\t"      /* quot = quot<<1 | (rem >= d) */
        "subs   ip, ip, #1\n\t"
        "bpl    1b\n\t"
        "bx     lr\n\t"
        "2:\n\t"                     /* d == 0 */
        "mov    r0, #0\n\t"
        "mov    r1, #0\n\t"
        "bx     lr\n\t"
        "3:\n\t"                     /* d == 1 */
        "mov    r1, #0\n\t"
        "bx     lr\n\t"
    );
}

/* __aeabi_uidiv: unsigned 32/32, quotient only. Thin wrapper - uidivmod
   already leaves the quotient in r0. */
__attribute__((naked)) unsigned __aeabi_uidiv(unsigned n, unsigned d) {
    __asm__(
        "push   {lr}\n\t"
        "bl     __aeabi_uidivmod\n\t"
        "pop    {lr}\n\t"
        "bx     lr\n\t"
    );
}

/* __aeabi_idivmod: signed 32/32 divmod. Takes magnitudes, calls uidivmod,
   then applies signs: quotient sign = n^d, remainder sign = n. */
__attribute__((naked)) int __aeabi_idivmod(int n, int d) {
    __asm__(
        "mov    r2, r0\n\t"          /* keep n for remainder sign */
        "eor    r3, r0, r1\n\t"      /* quotient sign bit */
        "cmp    r0, #0\n\t"
        "rsblt  r0, r0, #0\n\t"      /* |n| */
        "cmp    r1, #0\n\t"
        "rsblt  r1, r1, #0\n\t"      /* |d| */
        "push   {r2, r3, lr}\n\t"
        "bl     __aeabi_uidivmod\n\t"
        "pop    {r2, r3, lr}\n\t"
        "tst    r3, #0x80000000\n\t"
        "rsbne  r0, r0, #0\n\t"      /* negate quotient if signs differ */
        "cmp    r2, #0\n\t"
        "rsblt  r1, r1, #0\n\t"      /* remainder takes n's sign */
        "bx     lr\n\t"
    );
}

/* __aeabi_idiv: signed 32/32, quotient only. Wrapper over idivmod, which
   leaves the quotient in r0. */
__attribute__((naked)) int __aeabi_idiv(int n, int d) {
    __asm__(
        "push   {lr}\n\t"
        "bl     __aeabi_idivmod\n\t"
        "pop    {lr}\n\t"
        "bx     lr\n\t"
    );
}

/* __aeabi_uldivmod: unsigned 64/64 divmod. n in r0:r1, d in r2:r3;
   returns quotient in r0:r1 and remainder in r2:r3. 64-iteration
   restoring long division on (r4:r5 = rem, r6:r7 = quot). d == 0
   returns 0 (the engine never divides by zero). */
__attribute__((naked)) void __aeabi_uldivmod(unsigned long long n, unsigned long long d) {
    __asm__(
        "cmp    r3, #0\n\t"
        "cmpeq  r2, #0\n\t"
        "beq    4f\n\t"              /* d == 0 */
        /* r4-r8 are callee-saved: the caller's C code keeps live values
           in them across this call, so preserve them (libgcc's own
           __udivmoddi4 is C-compiled and does the same). */
        "push   {r4, r5, r6, r7, r8, lr}\n\t"
        "mov    r4, #0\n\t"          /* rem_hi */
        "mov    r5, #0\n\t"          /* rem_lo */
        "mov    r6, #0\n\t"          /* quot_hi */
        "mov    r7, #0\n\t"          /* quot_lo */
        "mov    ip, #63\n\t"
        "1:\n\t"
        /* Shift rem (r4:r5) left by 1, inserting n's bit 63 into the
           bottom.  Explicit register moves + orrcs, NOT an "adc r5,r5,r5;
           adc r4,r4,r4" carry chain: qemu does not propagate the carry
           flag through an adc whose destination is also a source (the
           64-bit chain then fills rem_hi with garbage).  Carry-in on a
           same-register adc is read correctly, carry-out is not, so the
           chain is unusable; orrcs reads a flag, never writes one. */
        "mov    r8, r1, lsr #31\n\t" /* r8 = n bit 63 (0/1) */
        "lsls   r5, r5, #1\n\t"      /* rem_lo <<= 1; C = old rem_lo bit 31 */
        "lsl    r4, r4, #1\n\t"      /* rem_hi <<= 1 */
        "orrcs  r4, r4, #1\n\t"      /* rem_hi |= old rem_lo bit 31 */
        "orr    r5, r5, r8\n\t"      /* rem_lo |= n bit 63 */
        "lsl    r1, r1, #1\n\t"
        "orr    r1, r1, r0, lsr #31\n\t" /* n_hi = (n_hi<<1)|(n_lo>>31) */
        "lsl    r0, r0, #1\n\t"      /* n_lo <<= 1 */
        "cmp    r4, r3\n\t"
        "cmpeq  r5, r2\n\t"          /* C = rem >= d (64-bit) */
        "movcs  r8, #1\n\t"
        "movcc  r8, #0\n\t"          /* r8 = (rem >= d) */
        "cmp    r8, #0\n\t"
        "beq    2f\n\t"
        "subs   r5, r5, r2\n\t"
        "sbc    r4, r4, r3\n\t"      /* rem -= d */
        "2:\n\t"
        "adds   r7, r7, r7\n\t"
        "adc    r6, r6, r6\n\t"      /* quot <<= 1 */
        "orr    r7, r7, r8\n\t"      /* quot_lo |= (rem >= d) */
        "subs   ip, ip, #1\n\t"
        "bpl    1b\n\t"
        "mov    r0, r7\n\t"          /* quot_lo */
        "mov    r1, r6\n\t"          /* quot_hi */
        "mov    r2, r5\n\t"          /* rem_lo */
        "mov    r3, r4\n\t"          /* rem_hi */
        "pop    {r4, r5, r6, r7, r8, lr}\n\t"
        "bx     lr\n\t"
        "4:\n\t"                     /* d == 0 */
        "mov    r0, #0\n\t"
        "mov    r1, #0\n\t"
        "mov    r2, #0\n\t"
        "mov    r3, #0\n\t"
        "bx     lr\n\t"
    );
}

#endif /* __arm__ && __GNUC__ */
