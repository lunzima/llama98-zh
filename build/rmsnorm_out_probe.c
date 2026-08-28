/* Does the Watcom RMS-norm output kernel agree with the scalar contract?
 *
 * The kernel computes o[i] = (x[i]*scale) * (w[i]+offset), element-wise,
 * in exactly that order: mulps x*xscale, addps w+offset, then mulps the
 * two. Each float op rounds to 32-bit, and the scalar reference asks for
 * the same with volatile intermediates.
 *
 * THAT REQUEST IS NOT GRANTED BY WATCOM, and the header used to claim
 * otherwise - "compiled -fp6 so nothing stays in x87's 80-bit". Built by
 * wcc386 the reference disagrees with the kernel on 109,897 of 1,280,000
 * values, and -fp6 changes nothing. The kernel is not what differs:
 * hashing each arm separately, the WATCOM KERNEL and the GCC-COMPILED
 * REFERENCE are the same 94C2D7F4 over every written lane, and only the
 * Watcom-compiled reference stands apart. A kernel-versus-reference
 * count cannot separate those two cases, because switching toolchain
 * switches both arms; a per-arm hash can.
 *
 * So the verdict differs by toolchain, and the probe says which it gave:
 *   gcc     - the reference is sound, the count is the verdict.
 *   Watcom  - the reference is x87, the count is REPORTED AND NOT
 *             FAILED, and the verdict is the kernel hash, which must
 *             equal the one the gcc build prints.
 */
#include <stdio.h>
#include <string.h>
#include "lz_int.h"

void lz_rmsnorm_out_sse_w(float *o, const float *x, const float *w,
                          int n4, const float *k2);

static unsigned long rs = 1145141919UL;
static unsigned long nextu(void) { rs = rs * 1103515245UL + 12345UL; return rs; }
static float nextf(void) {
    unsigned u = nextu();
    return (float)((int)(u & 0xFFFFFF) - 0x800000) / 4096.0f;
}

static void ref(float *o, const float *x, const float *w, int n4,
                const float *k2) {
    float scale = k2[0], offset = k2[1];
    int i, n = n4 * 4;
    for (i = 0; i < n; i++) {
        volatile float scaled = x[i] * scale;
        volatile float offw = w[i] + offset;
        o[i] = scaled * offw;
    }
}

int main(void) {
    float o1[64], o2[64], x[64], w[64], k2[2];
    long bad = 0, n = 0, moved = 0;
    /* Hashed per arm, not just compared: see the header. */
    unsigned long khash = 2166136261UL, rhash = 2166136261UL;
    int pass, i;
    static const int n4s[] = { 0, 1, 2, 3, 7, 8, 16 };

    for (pass = 0; pass < 20000; pass++) {
        int n4 = n4s[pass % 7];
        k2[0] = nextf();
        k2[1] = nextf();
        for (i = 0; i < 64; i++) { x[i] = nextf(); w[i] = nextf(); }
        for (i = 0; i < 64; i++) { o1[i] = 12345.0f; o2[i] = 12345.0f; }
        lz_rmsnorm_out_sse_w(o1, x, w, n4, k2);
        ref(o2, x, w, n4, k2);
        for (i = 0; i < 64; i++, n++) {
            if (o1[i] != o2[i]) {
                if (bad < 6)
                    printf("  pass %d n4 %d i %d: kernel %g ref %g\n",
                           pass, n4, i, (double)o1[i], (double)o2[i]);
                bad++;
            }
            if (o1[i] != 12345.0f) moved++;
            /* Written lanes only. The rest hold the sentinel in both
               arms and would hash equal whatever either arm computed. */
            if (i < n4 * 4) {
                unsigned uk, ur;
                memcpy(&uk, &o1[i], 4); memcpy(&ur, &o2[i], 4);
                khash ^= uk; khash *= 16777619UL; khash &= 0xFFFFFFFFUL;
                rhash ^= ur; rhash *= 16777619UL; rhash &= 0xFFFFFFFFUL;
            }
        }
    }
    printf("  kernel hash %08lX, reference hash %08lX\n", khash, rhash);
    /* Sensitivity: the kernel writes n4*4 lanes; the rest stay sentinel. */
    if (moved == 0) {
        printf("  no lane moved; the arms agreed on untouched memory\n");
        bad++;
    }
#ifdef __WATCOMC__
    /* The reference is x87 here, so a difference against it is not
       evidence about the kernel. Say what was and was not checked
       rather than returning a verdict this build cannot support. */
    printf("rmsnorm_out_probe: %ld value(s), %ld differ from an x87"
           " reference\n", n, bad);
    printf("  the kernel is UNCHECKED here except through the hash above:"
           " it must\n  equal the gcc build's kernel hash.\n");
    return moved == 0;
#else
    printf("rmsnorm_out_probe: %ld value(s), %ld mismatch(es)\n", n, bad);
    return bad != 0;
#endif
}
