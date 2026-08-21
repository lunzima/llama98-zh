#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

typedef union { float f; uint32_t u; } fu;

static int q8_round(float qv) {
    fu t;
    t.f = qv + 12582912.0f;
    return (int)(t.u & 0x007FFFFFu) - 0x00400000;
}

/* Mirrors lz_quantize_q8_int64's fast-branch body exactly (see
   src/ops.c), for one gs=32 group, WITHOUT the volatile fix - probing
   whether the loop shape (inv_elem computed once, read across a real
   loop) still exposes the same fusion hazard the int32 twin has. */
static void quant_group(const int64_t *grp, int gs, int8_t *out) {
    int64_t tam = 0;
    int k;
    for (k = 0; k < gs; k++) {
        int64_t a = grp[k] < 0 ? (int64_t)(0ull - (uint64_t)grp[k]) : grp[k];
        if (a > tam) tam = a;
    }
    {
        float tamf = (float)tam;
        float inv_elem = 127.0f / tamf;
        for (k = 0; k < gs; k++) {
            float g32 = (float)grp[k];
            int qi = q8_round(g32 * inv_elem);
            qi = (qi > 127) ? 127 : qi;
            out[k] = (int8_t)((qi < -127) ? -127 : qi);
        }
    }
}

int main(int argc, char **argv) {
    int64_t T;
    long L;
    int64_t grp[32];
    int8_t out[32];
    int mism = 0;

    if (argc < 2) { fprintf(stderr, "usage: %s T\n", argv[0]); return 2; }
    T = strtol(argv[1], NULL, 10);   /* runtime value, not foldable */

    for (L = 1; L < 127; L++) {
        int d;
        for (d = -2; d <= 2; d++) {
            /* tie boundary for output level L: V*127/T = L+0.5 */
            int64_t V = (int64_t)(((double)L + 0.5) * (double)T / 127.0) + d;
            if (V <= 0 || V > T) continue;
            memset(grp, 0, sizeof grp);
            grp[0] = T;
            grp[1] = -V;
            quant_group(grp, 32, out);
            printf("%lld %lld %d\n", (long long)T, (long long)V, (int)out[1]);
        }
    }
    (void)mism;
    return 0;
}
