/* Bit-identity check for lz_fwht_i32's MMX stage kernel.
 *
 * The transform is int32 add and subtract, wrapping, with no rounding,
 * no saturation and no reassociation - so unlike every float kernel in
 * this engine this one owes EXACT equality and there is no tolerance to
 * argue about. What the check is really for is the INDEXING: the MMX
 * body walks byte offsets and two words per iteration, and an off-by-one
 * in a stride would still produce a plausible-looking transform.
 *
 * Sweeps every block size a shipping config can produce (hadamard_o and
 * hadamard_down are powers of two, capped at 512 by the fixed-point
 * headroom - see subn_fwht_quant) plus the small ones the C body keeps
 * for itself, and inputs that reach the wrap: INT32_MIN/MAX, the
 * accumulator band, alternating signs, and one all-equal vector where
 * every butterfly's difference is exactly zero.
 *
 * The comparison is against the SAME function built without
 * LZ_FWHT_MMX_EXTERN - fwht_mmx_xcheck.sh builds both from one snapshot
 * of src/. Reimplementing the transform here would check my
 * transcription of it.
 *
 * Prints a checksum per (n, pattern); the script compares the two runs.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "fwht.h"
#include "ops.h"    /* lz_kernel_select - the tier is the point here */

#define NMAX 512

static int32_t in[NMAX], out[NMAX];

static uint32_t rs = 0x9E3779B9u;
static uint32_t rnd(void) {
    rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
    return rs;
}

int main(int argc, char **argv) {
    static const int NS[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    uint32_t h = 2166136261u;
    int ni, pat, i;
    /* The TIER is what selects the kernel (src/fwht.c dispatches on
       g_kernel, not on CPUID - a --kernel ref run must reach the C body
       or the control arm becomes the kernel). So the caller names it,
       and the driver runs this once per tier: ref, mmx, sse2. */
    if (argc > 1) {
        int t = LZ_KERNEL_REF;
        if (argv[1][0] == 'm') t = LZ_KERNEL_MMX;
        else if (argv[1][0] == 's') t = LZ_KERNEL_SSE2;
        lz_kernel_select(t);
    }
    for (ni = 0; ni < (int)(sizeof(NS) / sizeof(NS[0])); ni++) {
        int n = NS[ni];
        for (pat = 0; pat < 6; pat++) {
            for (i = 0; i < n; i++) {
                switch (pat) {
                case 0: in[i] = (int32_t)rnd(); break;
                case 1: in[i] = (int32_t)(rnd() % 65537u) - 32768; break;
                case 2: in[i] = (i & 1) ? 2147483647 : (-2147483647 - 1);
                        break;
                case 3: in[i] = 12345; break;          /* all differences 0 */
                case 4: in[i] = (int32_t)(rnd() & 0x3FFFFFFFu)
                                * ((i & 1) ? -1 : 1); break;
                default: in[i] = (i == 0) ? 1 : 0; break;  /* one impulse */
                }
            }
            lz_fwht_i32(out, in, n);
            for (i = 0; i < n; i++)
                h = (h ^ (uint32_t)out[i]) * 16777619u;
            printf("n=%3d pat=%d  out[0]=%11ld out[%d]=%11ld\n",
                   n, pat, (long)out[0], n - 1, (long)out[n - 1]);
        }
    }
    printf("FWHT %08lx\n", (unsigned long)h);
    return 0;
}
