/* lz_mathf's float transcendentals, gcc against Watcom, bit for bit.
 *
 * WHY THIS FILE EXISTS. reduce_large ends `x = (float)(int64_t)res0;
 * return x * pi63;`. A 387 loads that 62-bit integer with `fild`, which
 * is exact and which the precision control word does NOT touch - so the
 * x87 build multiplied the exact integer by pi63 and rounded once, while
 * SSE and the ARM soft-float round the integer to float first and then
 * round the product. One rounding against two, differing by 1 ulp on
 * ~0.6% of arguments.
 *
 * That was not academic. forward.c builds the RoPE table from
 * lz_sinf/lz_cosf at EVERY position, once, at load time, so the first
 * argument where the two compilers disagreed made the whole float tier
 * (--fixed off) diverge from that token onward - on kmr20 at position
 * 125, on _armgate2 at 134. Both corpora of
 * build/fixed_tier_xcompiler_gate.sh diverged at the SAME token index
 * despite sharing no prefix, which is the signature that named the
 * cause: a per-position, input-independent, model-parameterised table.
 *
 * PC=24 IS NOT A SUBSTITUTE and that is the reusable part. The engine
 * brackets float work in lz_fpu_float_begin so the 387 rounds mantissas
 * where SSE rounds, and that closes the arithmetic gap - but it says
 * nothing about `fild`, which is exact at any precision control, nor
 * about the 15-bit exponent. Any int->float conversion feeding a
 * multiply is still a divergence site; ops_kernel_shared.h's lz_i32f
 * exists for the int32 half of exactly this.
 *
 * Sections 2-4 are not decoration: the fix is one line in one function,
 * and a probe that only reproduced the one known argument would pass
 * again the moment the same mistake appeared in lz_expf or lz_powf.
 *
 * The bracket is restated locally rather than linked - see
 * rope_fpu_begin. Built by build/mathf_xcompiler_gate.sh.
 */
#include <stdio.h>
#include "lz_mathf.h"
#if defined(__WATCOMC__)
#include <float.h>
#endif

/* THE PC=24 BRACKET, RESTATED RATHER THAN LINKED. ops_quant.c's
   lz_fpu_float_begin drags ops_sched, ops_kernel, the three SIMD TUs and
   their dispatch tables in behind it; none of that is on the path this
   probe measures. With lz_fastfp_g at 0 - the state under test, since
   the probe passes no --fastfp - the engine's function reduces to
   exactly these two branches: _control87(_PC_24, _MCW_PC) on Watcom,
   and on gcc/SSE a read of MXCSR with no write, i.e. nothing. Everything
   else in it is fast-mode-only. */
static unsigned rope_fpu_begin(void) {
#if defined(__WATCOMC__)
    unsigned save = _control87(0, 0);
    _control87(_PC_24, _MCW_PC);
    return save;
#else
    return 0u;
#endif
}
static void rope_fpu_end(unsigned save) {
#if defined(__WATCOMC__)
    _control87(save, _MCW_PC | _MCW_RC);
#else
    (void)save;
#endif
}

static unsigned long h_ = 5381UL;
static void feed(float f) {
    union { float f; unsigned u; } b;
    b.f = f;
    h_ = h_ * 33UL + b.u;
}

/* (theta, rotary_dim) for the two models the divergence was measured on.
   Positions run past both observed onsets. */
static const float THETA[2] = { 1000000.0f, 10000000.0f };
static const char *NAME[2]  = { "kmr20", "_armgate2" };
#define ROT  32
#define NPOS 200

int main(void) {
    unsigned fpu;
    int m, i;

    fpu = rope_fpu_begin();

    /* ---- 1. the RoPE table itself, one line per entry so a diff of the
       two runs names the exact (position, freq index) that moved rather
       than only saying the table differs. This is the regression. */
    for (m = 0; m < 2; m++) {
        int half = ROT >> 1;
        int l;
        for (l = 0; l < NPOS; l++) {
            for (i = 0; i < half; i++) {
                union { float f; unsigned u; } c, s;
                float freq = lz_powf(THETA[m],
                                     -2.0f * (float)i / (float)ROT);
                float ang = (float)l * freq;
                c.f = lz_cosf(ang);
                s.f = lz_sinf(ang);
                feed(c.f); feed(s.f);
                printf("rope %-9s %3d %2d %08x %08x\n",
                       NAME[m], l, i, c.u, s.u);
            }
        }
    }

    /* ---- 2. sin/cos past the reduce_fast/reduce_large boundary. The
       defect lived only in reduce_large, which |x| >= 120 selects, so a
       sweep that stops short of 120 proves nothing about it. Bucketed
       hashes rather than a line each: a diff still names the 256-wide
       bucket, without a million lines of output. */
    for (i = 0; i < 64; i++) {
        unsigned long b = 5381UL;
        int k;
        for (k = 0; k < 256; k++) {
            union { float f; unsigned u; } c, s;
            /* Integers first (the RoPE case), then a fractional offset
               so the sweep is not confined to exactly representable
               arguments. */
            float x = (float)(i * 256 + k) + (float)k * (1.0f / 256.0f);
            c.f = lz_cosf(x);
            s.f = lz_sinf(x);
            b = b * 33UL + c.u;
            b = b * 33UL + s.u;
            feed(c.f); feed(s.f);
        }
        printf("sincos bucket %2d [%6d,%6d) %010lu\n",
               i, i * 256, (i + 1) * 256, b);
    }

    /* ---- 3. exp/log over the range the engine actually uses (decay
       gates, softplus, sigmoid) plus enough tail to reach the branchy
       parts. */
    for (i = 0; i < 32; i++) {
        unsigned long b = 5381UL;
        int k;
        for (k = 0; k < 256; k++) {
            float x = -40.0f + (float)(i * 256 + k) * (80.0f / 8192.0f);
            union { float f; unsigned u; } e, g;
            e.f = lz_expf(x);
            g.f = lz_logf(x > 0.0f ? x : 1.0f + (float)k);
            b = b * 33UL + e.u; b = b * 33UL + g.u;
            feed(e.f); feed(g.f);
        }
        printf("explog bucket %2d %010lu\n", i, b);
    }

    /* ---- 4. pow, which is where the RoPE frequencies come from and
       which composes exp and log - a divergence in either shows up here
       even if the direct sweeps above happen to miss it. */
    for (i = 0; i < 16; i++) {
        unsigned long b = 5381UL;
        int k;
        for (k = 0; k < 256; k++) {
            union { float f; unsigned u; } p;
            float base = 1.0f + (float)(i * 256 + k) * (1.0f / 64.0f);
            float ex   = -2.0f + (float)k * (4.0f / 256.0f);
            p.f = lz_powf(base, ex);
            b = b * 33UL + p.u;
            feed(p.f);
        }
        printf("pow bucket %2d %010lu\n", i, b);
    }

    printf("lz_mathf hash : %010lu\n", h_);
    rope_fpu_end(fpu);
    return 0;
}
