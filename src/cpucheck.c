/* CPU floor probe. See cpucheck.h for why the floor is what it is.
 *
 * Two implementations of the probe, one per compiler, because the
 * Watcom build is 32-bit and needs real assembly while the development
 * build is x86-64 where both answers are constants the architecture
 * guarantees. The DECISION is one implementation shared by both, which
 * is the half that can be wrong.
 */
#include <stdio.h>
#include <string.h>

#include "cpucheck.h"
#include "err.h"

#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>   /* __get_cpuid - the brand-string leaves */
#endif /* __i386__ || __x86_64__ */

/* CPUID leaf 1, EDX. Only the bits this file names. */
#define LZ_CPUID1_FPU  (1u << 0)
#define LZ_CPUID1_TSC  (1u << 4)
#define LZ_CPUID1_CMOV (1u << 15)
#define LZ_CPUID1_MMX  (1u << 23)
#define LZ_CPUID1_SSE  (1u << 25)
#define LZ_CPUID1_SSE2 (1u << 26)

int lz_cpu_check_bits(int have_cpuid, unsigned edx1,
                      char *errbuf, int errlen) {
    int rc;
    if (!have_cpuid) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_CPU_NO_CPUID);
        return rc;
    }
    if (!(edx1 & LZ_CPUID1_FPU)) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_CPU_NO_FPU);
        return rc;
    }
    /* Everything else - MMX, SSE, SSE2, 3DNow - is optional and picked
       by lz_kernel_select. A Socket 7 Pentium with none of them runs the
       scalar reference path, slowly and correctly. */
    return 0;
}

/* ---------------------------------------------------------- the probe */

#if defined(__WATCOMC__)

/* EFLAGS bit 21 (ID) is writable exactly when CPUID exists. This is the
   architecturally defined test and is safe on every 32-bit x86,
   including the ones that would fault on CPUID itself. */
extern unsigned lz_eflags_id_toggles(void);
#pragma aux lz_eflags_id_toggles = \
    "pushfd"                 \
    "pop  eax"               \
    "mov  ecx, eax"          \
    "xor  eax, 00200000h"    \
    "push eax"               \
    "popfd"                  \
    "pushfd"                 \
    "pop  eax"               \
    "xor  eax, ecx"          \
    "shr  eax, 21"           \
    "and  eax, 1"            \
    "push ecx"               \
    "popfd"                  \
    __value [__eax] __modify [__eax __ecx]

unsigned lz_cpuid1_edx(void) {
    /* A real function, not #pragma aux, so ops.c can call it across the
       module boundary (see cpucheck.h). The __asm block's prologue saves
       and restores ebx - CPUID clobbers it - so the body itself does not
       have to. edx is the caller-saved register carrying the return. */
    unsigned result;
    __asm {
        mov  eax, 1
        cpuid
        mov  result, edx
    }
    return result;
}

static int probe(unsigned *edx1) {
    if (!lz_eflags_id_toggles()) { *edx1 = 0; return 0; }
    *edx1 = lz_cpuid1_edx();
    return 1;
}

#elif defined(__i386__) || defined(__x86_64__)

/* EFLAGS bit 21 (ID) is writable exactly when CPUID exists - the same
   architecturally defined test as the Watcom branch's
   lz_eflags_id_toggles() above, ported to gcc inline asm. size_t, not
   unsigned long: it must match pushf/popf's operand width in whichever
   mode this TU is compiled for (4 bytes at -m32, 8 at -m64/x86-64) -
   unsigned long is only 4 bytes in both modes under this project's
   Windows/MinGW target (LLP64), which would push a 32-bit slot against
   a 64-bit pushf and corrupt the stack under -m64. */
static int lz_eflags_id_toggles(void) {
    size_t f0, f1;
    __asm__ __volatile__("pushf\n\tpop %0" : "=r"(f0) :: "cc");
    __asm__ __volatile__(
        "push %1\n\tpopf\n\tpushf\n\tpop %0"
        : "=r"(f1) : "r"(f0 ^ (size_t)0x200000) : "cc");
    __asm__ __volatile__("push %0\n\tpopf" :: "r"(f0) : "cc");
    return (int)(((f0 ^ f1) >> 21) & 1u);
}

unsigned lz_cpuid1_edx(void) {
    /* No EFLAGS.ID test here - by design, matching the Watcom half's
       shape (lz_cpuid1_edx_asm above is equally raw): the guard belongs
       to the one caller that decides whether it is safe to use, probe()
       below, not to this primitive. A second, independent caller that
       skipped the guard would be the actual bug; the gate exists to
       catch exactly that, not to duplicate the check here. */
    unsigned a, b, c, d;
    a = b = c = d = 0;
    __asm__ __volatile__("cpuid"
                         : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                         : "a"(1));
    (void)a; (void)b; (void)c;
    return d;
}

static int probe(unsigned *edx1) {
    /* Guarded: on i486/i386 there is no CPUID instruction and this
       would fault, and the x87 build targets real Socket-7 hardware.
       Harmless at today's i586 floor (every Socket 7 part has CPUID) -
       same guard the Watcom branch above already has. */
    if (!lz_eflags_id_toggles()) { *edx1 = 0; return 0; }
    *edx1 = lz_cpuid1_edx();
    return 1;
}

#else

static int probe(unsigned *edx1) {
    /* Not x86. Nothing here can run anyway, but reporting "no CPUID"
       would name the wrong problem, so claim the two required features
       and let the rest of the engine fail on its own terms. */
    *edx1 = LZ_CPUID1_FPU;
    return 1;
}

#endif /* __WATCOMC__ || __i386__ || __x86_64__ */

int lz_cpu_check(char *errbuf, int errlen) {
    unsigned edx1 = 0;
    int have = probe(&edx1);
    return lz_cpu_check_bits(have, edx1, errbuf, errlen);
}

const char *lz_cpu_features(void) {
    static char buf[64];
    unsigned edx1 = 0;
    int have;
    if (buf[0]) return buf;
    have = probe(&edx1);
    if (!have) { strcpy(buf, "none"); return buf; }
    strcpy(buf, "cpuid");
    if (edx1 & LZ_CPUID1_FPU)  strcat(buf, " fpu");
    if (edx1 & LZ_CPUID1_TSC)  strcat(buf, " tsc");
    if (edx1 & LZ_CPUID1_CMOV) strcat(buf, " cmov");
    if (edx1 & LZ_CPUID1_MMX)  strcat(buf, " mmx");
    if (edx1 & LZ_CPUID1_SSE)  strcat(buf, " sse");
    if (edx1 & LZ_CPUID1_SSE2) strcat(buf, " sse2");
    return buf;
}

/* --------------------------------------------------- CPU brand string
 *
 * The processor's model name: CPUID leaves 0x80000002..0x80000004, each
 * returning 16 bytes (four registers) of the 48-byte brand string. The
 * leaves are optional - a CPUID whose maximum extended leaf (the value
 * returned by leaf 0x80000000) is below 0x80000004 has no brand string,
 * and on a 386/486 there is no CPUID at all. Two helpers per compiler,
 * the same split probe() already uses: Watcom needs real asm, gcc uses
 * its __get_cpuid intrinsic. The one shared decision is the leaf-range
 * check, which is pure arithmetic and testable.
 */

/* Read CPUID leaf `leaf` into eax/ebx/ecx/edx. Returns 0 on failure
   (no CPUID, or the leaf is not supported). Used for leaves above 1,
   so the Watcom side cannot reuse lz_cpuid1_edx. */
#if defined(__WATCOMC__)

/* Leaf 0x80000000: the maximum extended leaf number. */
extern unsigned lz_cpucheck_maxext(void);
#pragma aux lz_cpucheck_maxext = \
    ".586"                     \
    "push ebx"                 \
    "mov  eax, 080000000h"     \
    "cpuid"                    \
    "pop  ebx"                 \
    __value [__eax] __modify [__eax __ecx __edx]

/* Leaf in eax, out pointer in edx - explicit register parameters, the
   same shape the engine's own multi-arg pragmas use (__parm with one
   bracketed register per parameter, and __modify naming the general
   registers this asm actually clobbers - the same named form
   lz_cpuid1_edx already uses, NOT __modify [8087], which is for
   x87/MMX and would let the compiler believe eax/ecx/edx/edi survive).
   edi is the scratch that holds `out` across the CPUID (CPUID clobbers
   ebx, ecx, edx with the brand data, so the pointer has to live
   somewhere the instruction does not touch). The brand bytes are 16
   bytes across eax:ebx:ecx:edx, written to out in that order (NOT the
   feature-leaf convention; getting it wrong leaves trailing garbage
   after the name). */
extern void lz_cpucheck_brand(unsigned leaf, char *out);
#pragma aux lz_cpucheck_brand = \
    ".586"                     \
    "mov  edi, edx"            \
    "push ebx"                 \
    "cpuid"                    \
    "mov  [edi+0],  eax"       \
    "mov  [edi+4],  ebx"       \
    "mov  [edi+8],  ecx"       \
    "mov  [edi+12], edx"       \
    "pop  ebx"                 \
    __parm [__eax] [__edx]     \
    __modify [__eax __ecx __edx __edi]

static int cpu_brand_raw(char *out, int cap) {
    unsigned maxext;
    if (!lz_eflags_id_toggles()) return 0;   /* no CPUID */
    maxext = lz_cpucheck_maxext();
    if (maxext < 0x80000004u) return 0;      /* no brand string */
    if (cap < 48) return 0;
    lz_cpucheck_brand(0x80000002, out);
    lz_cpucheck_brand(0x80000003, out + 16);
    lz_cpucheck_brand(0x80000004, out + 32);
    return 1;
}

#elif defined(__i386__) || defined(__x86_64__)

static int cpu_brand_raw(char *out, int cap) {
    unsigned a, b, c, d, maxext;
    a = b = c = d = 0;
    __get_cpuid(0x80000000, &a, &b, &c, &d);
    maxext = a;
    if (maxext < 0x80000004u) return 0;
    if (cap < 48) return 0;
    /* The brand leaves return their 16 bytes in EAX:EBX:ECX:EDX order,
       not the feature-leaf convention. Each register is one 4-byte
       little-endian word. */
    __get_cpuid(0x80000002, &a, &b, &c, &d);
    memcpy(out + 0,  &a, 4); memcpy(out + 4,  &b, 4);
    memcpy(out + 8,  &c, 4); memcpy(out + 12, &d, 4);
    __get_cpuid(0x80000003, &a, &b, &c, &d);
    memcpy(out + 16, &a, 4); memcpy(out + 20, &b, 4);
    memcpy(out + 24, &c, 4); memcpy(out + 28, &d, 4);
    __get_cpuid(0x80000004, &a, &b, &c, &d);
    memcpy(out + 32, &a, 4); memcpy(out + 36, &b, 4);
    memcpy(out + 40, &c, 4); memcpy(out + 44, &d, 4);
    return 1;
}

#else

static int cpu_brand_raw(char *out, int cap) {
    (void)out; (void)cap;
    return 0;
}

#endif /* __WATCOMC__ || __i386__ || __x86_64__ */

/* The brand string, trimmed. Word 95's System Info had no way to name a
   processor beyond 386/486/Pentium; a modern machine needs its real
   model name, which is exactly what these leaves carry. Returns the
   static buffer, or "" when the CPU has none. */
/* Printable ASCII, excluding the space. Used only to decide where the
   brand string's real text begins and ends, so a space counts as
   trimmable at the edges while the interior keeps its spaces - the name
   itself has them ("Intel(R) Core(TM) ..."). */
static int brand_printable(char c) {
    unsigned char u = (unsigned char)c;
    return u > ' ' && u < 0x7F;
}

const char *lz_cpu_brand(void) {
    static char buf[64];
    static int  done;
    int lo = 0, hi, n;

    if (done) return buf;
    done = 1;                    /* a CPU without a brand string caches
                                    the empty answer too, rather than
                                    re-running CPUID on every call */
    if (!cpu_brand_raw(buf, (int)sizeof buf)) { buf[0] = '\0'; return buf; }
    buf[48] = '\0';              /* cpu_brand_raw fills exactly 48 raw
                                    bytes and guarantees no terminator */

    /* BOTH ENDS, and every unprintable byte - not just trailing spaces.
       These 48 bytes come straight out of the CPUID registers with no
       sanitation of any kind: Intel LEFT-pads the name ("          Intel(R)
       Core(TM) ..."), so trimming only the right leaves the padding in,
       and a part that fills the leaves with something other than spaces
       puts control bytes at either end. Those draw as boxes in the About
       window, which is where this was noticed. */
    n = (int)strlen(buf);
    while (lo < n && !brand_printable(buf[lo])) lo++;
    hi = n;
    while (hi > lo && !brand_printable(buf[hi - 1])) hi--;
    if (lo > 0) memmove(buf, buf + lo, (size_t)(hi - lo));
    buf[hi - lo] = '\0';
    return buf;
}
