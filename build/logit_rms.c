/* rms of the difference between two f32 --dump-logits files.
 *
 * Exists because build/kv_rot_gate.sh has to compare QUALITY, not
 * bytes, and a shell cannot do float arithmetic on binary files. Kept
 * next to the gate that uses it rather than in .prof/, since .prof is
 * gitignored and a gate whose comparator is not in a fresh clone is a
 * gate that does not run there.
 *
 *   logit_rms A.bin B.bin      prints "<rms> <n>" and exits 0
 *                              exits 1 if the files differ in size
 *
 * Prints with %.9e so a caller can compare two runs' output with a
 * string sort - which is what the gate does, to avoid needing float
 * arithmetic in the shell at all.
 *
 * Build: cc -O2 -o logit_rms build/logit_rms.c -lm  */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static long slurp(const char *p, float **out) {
    FILE *f = fopen(p, "rb");
    long n;
    float *b;
    if (!f) { fprintf(stderr, "logit_rms: cannot open %s\n", p); return -1; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    n = ftell(f);
    if (n <= 0 || (n % 4) != 0) { fclose(f); return -1; }
    rewind(f);
    b = (float *)malloc((size_t)n);
    if (!b) { fclose(f); return -1; }
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return -1; }
    fclose(f);
    *out = b;
    return n / 4;
}

int main(int argc, char **argv) {
    float *a = 0, *b = 0;
    long na, nb, i;
    double s = 0.0;
    if (argc != 3) { fprintf(stderr, "usage: logit_rms A.bin B.bin\n"); return 1; }
    na = slurp(argv[1], &a);
    nb = slurp(argv[2], &b);
    if (na < 0 || nb < 0) return 1;
    if (na != nb) {
        fprintf(stderr, "logit_rms: %ld vs %ld floats\n", na, nb);
        return 1;
    }
    for (i = 0; i < na; i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    printf("%.9e %ld\n", sqrt(s / (double)na), na);
    return 0;
}
