#ifndef LZ_CPUCHECK_H
#define LZ_CPUCHECK_H

/* Can this processor run the engine at all?
 *
 * THE FLOOR IS SOCKET 7, AND SOCKET 7 IS NOT "HAS MMX". The original
 * Pentium (P54C), the AMD K5 and the non-MX Cyrix 6x86 are all Socket 7
 * parts with no MMX at all. So MMX, SSE, SSE2 and 3DNow are optional
 * here and chosen by CPUID at run time (lz_kernel_select); the scalar
 * reference path is not a fallback of last resort, it is the path a
 * perfectly ordinary target CPU takes.
 *
 * That leaves exactly two hard requirements: the CPUID instruction, and
 * a floating-point unit.
 *
 * WHY CPUID IS CHECKED BEFORE IT IS USED. Executing CPUID on a 386 or an
 * early 486 is an invalid opcode - a crash, and the crash happens inside
 * the very code whose job is to find out whether the machine is
 * supported. The probe therefore starts by toggling EFLAGS bit 21, which
 * is the architecturally defined way to ask, and which is safe on every
 * 32-bit x86.
 *
 * The decision is split from the probe so it can be tested. On the
 * development machine every check passes and always will, so a
 * self-contained lz_cpu_check() would have an untestable failure path -
 * and a failure path nobody has ever seen run is not known to work.
 */

/* Decide from feature bits. Pure: no CPUID, no globals.
 *   have_cpuid  0 when the CPUID instruction is absent
 *   edx1        CPUID leaf 1's EDX, meaningful only when have_cpuid
 * Returns 0 when the processor is usable, else an LZ_ERR_CPU_* code
 * with errbuf filled. */
int lz_cpu_check_bits(int have_cpuid, unsigned edx1,
                      char *errbuf, int errlen);

/* Probe this processor and decide. Same return values. */
int lz_cpu_check(char *errbuf, int errlen);

/* CPUID leaf 1's EDX, as raw feature bits (bit 23 MMX, bit 25 SSE,
   bit 26 SSE2). Watcom only: on the gcc build the answer is a compile-time
   constant, so there is no implementation to call.

   A real FUNCTION with a __asm block, not #pragma aux. #pragma aux is the
   inline-code-generation syntax and emits no linkable symbol, so a leaf
   needed in two files could only be duplicated. This is a normal symbol
   ops.c calls, and the __asm block's own prologue saves/restores ebx
   (CPUID clobbers it), which a #pragma aux body has to do by hand. */
#if defined(__WATCOMC__)
unsigned lz_cpuid1_edx(void);
#endif

/* Space-separated feature list for a banner or an about box, e.g.
 * "cpuid fpu mmx sse". Never NULL, never localized - these are
 * instruction-set names, not prose. */
const char *lz_cpu_features(void);

/* The processor's model name, read from the CPUID brand string (leaves
 * 0x80000002..0x80000004), for the System Info dialog's CPU line. Empty
 * string when the CPU does not expose it (a 386/486, or a CPUID whose
 * max extended leaf is below 0x80000004). Not localized - it is the
 * chip's own name, e.g. "AMD Ryzen 7 5800X" or "GenuineIntel". */
const char *lz_cpu_brand(void);

#endif
