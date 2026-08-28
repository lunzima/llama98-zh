/* Every operator src/lz_softfp.c exports, against libgcc's own.
   Built and run by build/arm/softfp_gate.sh under qemu-arm-static; that
   script also checks that this file MENTIONS every exported symbol, so
   adding a routine without a test here turns the gate red.

   WHY THE COVERAGE CHECK EXISTS. The first version of this comparison
   lived in a scratch directory and exercised eight of the twenty-two
   symbols the file defines, while the commit that introduced it said
   "every operator against libgcc's, 0 of 41,016,746 cases differing".
   __aeabi_f2lz was among the fourteen it did not touch, and it was
   wrong: declared `long` rather than the 64-bit type the EABI name
   means, so it returned a correct low word in r0 and left the high word
   in r1 untouched. Nothing reached it until src/json.c narrowed its
   number type to float, at which point every safetensors tensor offset
   went through it and the loader began rejecting offsets of 0..2048 for
   being negative. A test that had merely EXISTED for that symbol would
   have caught it on the first run; the count of cases was never the
   thing keeping the file honest.

   MINE vs THEIRS. The gate renames this file's symbols to lzsf_* with
   objcopy, so both implementations link into one binary and each case
   is computed twice.

   NAMING THE SYMBOL IS THE POINT. The extern declarations below use the
   real __aeabi_ names so the coverage check can find them by grep; the
   renamed copies are declared alongside. */
#include <stdio.h>
#include "lz_int.h"

/* libgcc's */
extern float __aeabi_fadd(float, float);
extern float __aeabi_fsub(float, float);
extern float __aeabi_fmul(float, float);
extern float __aeabi_fdiv(float, float);
extern float __addsf3(float, float);
extern float __aeabi_i2f(int);
extern float __aeabi_ui2f(unsigned);
extern float __aeabi_l2f(lz_i64);
extern int   __aeabi_f2iz(float);
extern lz_i64 __aeabi_f2lz(float);
extern int   __aeabi_fcmpeq(float, float);
extern int   __aeabi_fcmplt(float, float);
extern int   __aeabi_fcmple(float, float);
extern int   __aeabi_fcmpge(float, float);
extern int   __aeabi_fcmpgt(float, float);
extern unsigned __aeabi_uidiv(unsigned, unsigned);
extern int   __aeabi_idiv(int, int);
/* __aeabi_uidivmod, __aeabi_idivmod and __aeabi_uldivmod return a PAIR
   of results in registers and have no C prototype, so they are called
   from the inline assembly below. THEY CANNOT BE TESTED THROUGH `/` AND
   `%`, which is what this file tried first: the gate renames our copies
   so that both implementations can link together, which means an
   ordinary division in this file binds to LIBGCC's routine. Those three
   rows were comparing libgcc with itself and would have reported PASS
   with our versions deleted. __aeabi_cfcmpeq and __aeabi_cfrcmple have
   the same problem from the other direction - they answer in the ARM
   condition FLAGS rather than in a register - and are likewise called
   from assembly that captures CPSR. */

/* ours, after objcopy */
extern float lzsf_fadd(float, float);
extern float lzsf_fsub(float, float);
extern float lzsf_fmul(float, float);
extern float lzsf_fdiv(float, float);
extern float lzsf_addsf3(float, float);
extern float lzsf_i2f(int);
extern float lzsf_ui2f(unsigned);
extern float lzsf_l2f(lz_i64);
extern int   lzsf_f2iz(float);
extern lz_i64 lzsf_f2lz(float);
extern int   lzsf_fcmpeq(float, float);
extern int   lzsf_fcmplt(float, float);
extern int   lzsf_fcmple(float, float);
extern int   lzsf_fcmpge(float, float);
extern int   lzsf_fcmpgt(float, float);
extern unsigned lzsf_uidiv(unsigned, unsigned);
extern int   lzsf_idiv(int, int);

#define NOP 24
static long bad[NOP], n[NOP];
static const char *opname[NOP] = {
    "fadd", "fsub", "fmul", "fdiv", "addsf3",
    "i2f", "ui2f", "l2f", "f2iz", "f2lz",
    "fcmpeq", "fcmplt", "fcmple", "fcmpge", "fcmpgt",
    "uidiv", "idiv", "uidivmod", "idivmod", "uldivmod",
    "cfcmpeq", "cfrcmple", "f2lz-oor", "pad"
};
/* f2lz OUT OF RANGE is counted apart and never fails the gate. C leaves
   the conversion undefined once the value will not fit, and libgcc's
   __fixsfdi is C rather than assembly there, so it WRAPS - 2^64 comes
   back as -1. This file saturates, as its own f2iz does. The row is
   printed so the divergence stays visible and measured. */
#define OP_F2LZ_OOR 22

static uint32_t rnd(void) {
    static uint32_t s = 0x9e3779b9u;
    s = s * 1664525u + 1013904223u;
    return s;
}
static uint32_t fbits(float f) {
    union { float f; uint32_t u; } v; v.f = f; return v.u;
}
static float fval(uint32_t u) {
    union { uint32_t u; float f; } v; v.u = u; return v.f;
}
static void ckf(int op, float mine, float theirs) {
    n[op]++;
    if (fbits(mine) != fbits(theirs)) {
        bad[op]++;
        if (bad[op] <= 3)
            printf("  %s: mine %08x  libgcc %08x\n", opname[op],
                   (unsigned)fbits(mine), (unsigned)fbits(theirs));
    }
}
static void cki(int op, lz_i64 mine, lz_i64 theirs) {
    n[op]++;
    if (mine != theirs) {
        bad[op]++;
        if (bad[op] <= 3)
            printf("  %s: mine %ld  libgcc %ld\n", opname[op],
                   (long)mine, (long)theirs);
    }
}

/* f2lz's range. Above 2^64 (exponent > 190) and inside [2^63, 2^64)
   except for -2^63 exactly, the conversion is undefined in C. */
static int f2lz_in_range(uint32_t b) {
    uint32_t e = (b >> 23) & 0xFFu;
    if (e < 127u) return 1;
    if (e > 190u) return 0;
    if (e < 190u) return 1;
    return (b & 0x80000000u) && !(b & 0x7FFFFFu);
}

/* The flag-setting compares. Both take their operands in r0/r1 and
   answer in CPSR rather than in a register, so C cannot call them. */
#define CFCALL(fn, x, y, out)                                       \
    __asm__ __volatile__(                                           \
        "mov  r0, %[a]\n\t"                                         \
        "mov  r1, %[b]\n\t"                                         \
        "bl   " fn "\n\t"                                           \
        "mrs  %[o], cpsr\n\t"                                       \
        : [o] "=r"(out)                                             \
        : [a] "r"(x), [b] "r"(y)                                    \
        : "r0", "r1", "r2", "r3", "ip", "lr", "cc")

/* The divmods. 32-bit: n in r0, d in r1, quotient back in r0 and
   remainder in r1. 64-bit: n in r0:r1, d in r2:r3, quotient in r0:r1
   and remainder in r2:r3. Named register variables rather than
   hand-written MOVs so gcc places the operands itself and the "+r"
   constraints tell it all four are written. */
#define DIVMOD32(fn, N, D, Q, R)                                     \
    do {                                                             \
        register uint32_t a0 __asm__("r0") = (N);                    \
        register uint32_t a1 __asm__("r1") = (D);                    \
        __asm__ __volatile__("bl " fn                                \
            : "+r"(a0), "+r"(a1)                                     \
            : : "r2", "r3", "ip", "lr", "cc", "memory");             \
        (Q) = a0; (R) = a1;                                          \
    } while (0)

#define DIVMOD64(fn, N, D, Q, R)                                     \
    do {                                                             \
        register uint32_t b0 __asm__("r0") = (uint32_t)(N);          \
        register uint32_t b1 __asm__("r1") = (uint32_t)((N) >> 32);  \
        register uint32_t b2 __asm__("r2") = (uint32_t)(D);          \
        register uint32_t b3 __asm__("r3") = (uint32_t)((D) >> 32);  \
        __asm__ __volatile__("bl " fn                                \
            : "+r"(b0), "+r"(b1), "+r"(b2), "+r"(b3)                 \
            : : "ip", "lr", "cc", "memory");                         \
        (Q) = ((lz_u64)b1 << 32) | b0;                               \
        (R) = ((lz_u64)b3 << 32) | b2;                               \
    } while (0)

static void ck_flags(int op, uint32_t mine, uint32_t theirs) {
    n[op]++;
    /* N, Z, C and V only - the rest of CPSR is not this call's answer. */
    if ((mine & 0xF0000000u) != (theirs & 0xF0000000u)) {
        bad[op]++;
        if (bad[op] <= 3)
            printf("  %s: mine %08x  libgcc %08x\n", opname[op],
                   (unsigned)(mine & 0xF0000000u),
                   (unsigned)(theirs & 0xF0000000u));
    }
}

int main(void) {
    /* The special values every float routine gets wrong first. */
    static const uint32_t sv[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x807FFFFFu,
        0x00800000u, 0x3F800000u, 0xBF800000u, 0x3F7FFFFFu,
        0x3F800001u, 0x7F7FFFFFu, 0xFF7FFFFFu, 0x7F800000u,
        0xFF800000u, 0x7FC00000u, 0x7F800001u, 0xFF8A6556u,
        0x4B000000u, 0x4F000000u, 0x5F000000u, 0xDF000000u,
        0x00004000u, 0x40490FDBu, 0xC0490FDBu, 0x33800000u
    };
    const int nsv = (int)(sizeof sv / sizeof sv[0]);
    int i, j, t;
    int fail = 0;

    for (i = 0; i < nsv; i++)
        for (j = 0; j < nsv; j++) {
            float a = fval(sv[i]), b = fval(sv[j]);
            uint32_t f1, f2;
            ckf(0, lzsf_fadd(a, b),   __aeabi_fadd(a, b));
            ckf(1, lzsf_fsub(a, b),   __aeabi_fsub(a, b));
            ckf(2, lzsf_fmul(a, b),   __aeabi_fmul(a, b));
            ckf(3, lzsf_fdiv(a, b),   __aeabi_fdiv(a, b));
            ckf(4, lzsf_addsf3(a, b), __addsf3(a, b));
            cki(10, lzsf_fcmpeq(a, b), __aeabi_fcmpeq(a, b));
            cki(11, lzsf_fcmplt(a, b), __aeabi_fcmplt(a, b));
            cki(12, lzsf_fcmple(a, b), __aeabi_fcmple(a, b));
            cki(13, lzsf_fcmpge(a, b), __aeabi_fcmpge(a, b));
            cki(14, lzsf_fcmpgt(a, b), __aeabi_fcmpgt(a, b));
            CFCALL("lzsf_cfcmpeq",   sv[i], sv[j], f1);
            CFCALL("__aeabi_cfcmpeq", sv[i], sv[j], f2);
            ck_flags(20, f1, f2);
            CFCALL("lzsf_cfrcmple",   sv[i], sv[j], f1);
            CFCALL("__aeabi_cfrcmple", sv[i], sv[j], f2);
            ck_flags(21, f1, f2);
        }

    for (i = 0; i < 1500000; i++) {
        uint32_t ua = rnd(), ub = rnd();
        float a = fval(ua), b = fval(ub);
        uint32_t f1, f2;
        ckf(0, lzsf_fadd(a, b),   __aeabi_fadd(a, b));
        ckf(1, lzsf_fsub(a, b),   __aeabi_fsub(a, b));
        ckf(2, lzsf_fmul(a, b),   __aeabi_fmul(a, b));
        ckf(3, lzsf_fdiv(a, b),   __aeabi_fdiv(a, b));
        ckf(4, lzsf_addsf3(a, b), __addsf3(a, b));
        cki(10, lzsf_fcmpeq(a, b), __aeabi_fcmpeq(a, b));
        cki(11, lzsf_fcmplt(a, b), __aeabi_fcmplt(a, b));
        cki(12, lzsf_fcmple(a, b), __aeabi_fcmple(a, b));
        cki(13, lzsf_fcmpge(a, b), __aeabi_fcmpge(a, b));
        cki(14, lzsf_fcmpgt(a, b), __aeabi_fcmpgt(a, b));
        if ((i & 63) == 0) {
            CFCALL("lzsf_cfcmpeq",   ua, ub, f1);
            CFCALL("__aeabi_cfcmpeq", ua, ub, f2);
            ck_flags(20, f1, f2);
            CFCALL("lzsf_cfrcmple",   ua, ub, f1);
            CFCALL("__aeabi_cfrcmple", ua, ub, f2);
            ck_flags(21, f1, f2);
        }

        /* conversions */
        cki(8, lzsf_f2iz(a), __aeabi_f2iz(a));
        if (f2lz_in_range(ua)) cki(9, lzsf_f2lz(a), __aeabi_f2lz(a));
        else {
            n[OP_F2LZ_OOR]++;
            if (lzsf_f2lz(a) != __aeabi_f2lz(a)) bad[OP_F2LZ_OOR]++;
        }
        ckf(5, lzsf_i2f((int)ua),       __aeabi_i2f((int)ua));
        ckf(6, lzsf_ui2f(ua),           __aeabi_ui2f(ua));
        ckf(7, lzsf_l2f(((lz_i64)(int)ua << 32) | ub),
               __aeabi_l2f(((lz_i64)(int)ua << 32) | ub));

        /* integer division. ub == 0 is skipped: libgcc raises SIGFPE
           there and this file deliberately returns 0 instead, which is
           a divergence that cannot be compared by running both. */
        if (ub) {
            uint32_t q1, r1, q2, r2;
            cki(15, (lz_i64)lzsf_uidiv(ua, ub), (lz_i64)__aeabi_uidiv(ua, ub));
            DIVMOD32("lzsf_uidivmod",    ua, ub, q1, r1);
            DIVMOD32("__aeabi_uidivmod", ua, ub, q2, r2);
            n[17]++;
            if (q1 != q2 || r1 != r2) {
                bad[17]++;
                if (bad[17] <= 3)
                    printf("  uidivmod %u/%u: mine %u,%u  libgcc %u,%u\n",
                           (unsigned)ua, (unsigned)ub,
                           (unsigned)q1, (unsigned)r1,
                           (unsigned)q2, (unsigned)r2);
            }
        }
        /* INT32_MIN / -1 overflows; libgcc traps there and this file does
           not reproduce that, so it is not a case both can run. */
        if (ub && !(ua == 0x80000000u && ub == 0xFFFFFFFFu)) {
            int sa = (int)ua, sb = (int)ub;
            uint32_t q1, r1, q2, r2;
            cki(16, (lz_i64)lzsf_idiv(sa, sb), (lz_i64)__aeabi_idiv(sa, sb));
            DIVMOD32("lzsf_idivmod",    ua, ub, q1, r1);
            DIVMOD32("__aeabi_idivmod", ua, ub, q2, r2);
            n[18]++;
            if (q1 != q2 || r1 != r2) {
                bad[18]++;
                if (bad[18] <= 3)
                    printf("  idivmod %d/%d: mine %d,%d  libgcc %d,%d\n",
                           sa, sb, (int)q1, (int)r1, (int)q2, (int)r2);
            }
        }
        {
            lz_u64 nn = ((lz_u64)ua << 32) | ub;
            lz_u64 dd = (lz_u64)(ub | 1u) | ((lz_u64)(ua & 0xFFu) << 32);
            lz_u64 q1, r1, q2, r2;
            DIVMOD64("lzsf_uldivmod",    nn, dd, q1, r1);
            DIVMOD64("__aeabi_uldivmod", nn, dd, q2, r2);
            n[19]++;
            if (q1 != q2 || r1 != r2) {
                bad[19]++;
                if (bad[19] <= 3)
                    printf("  uldivmod: mine %lu,%lu  libgcc %lu,%lu\n",
                           (unsigned long)q1, (unsigned long)r1,
                           (unsigned long)q2, (unsigned long)r2);
            }
        }
    }

    for (t = 0; t < NOP; t++) {
        if (!n[t]) continue;
        printf("  %-9s %8ld / %8ld differ%s\n", opname[t], bad[t], n[t],
               t == OP_F2LZ_OOR ? "  (undefined in C; saturate vs wrap)"
                                : "");
        if (bad[t] && t != OP_F2LZ_OOR) fail = 1;
    }
    if (fail) { printf("softfp: FAIL\n"); return 1; }
    printf("softfp: PASS (every exported operator matches libgcc)\n");
    return 0;
}
