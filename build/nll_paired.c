/* Paired per-position NLL comparison between two --dump-nll files.
 *
 *   nll_paired A.nll B.nll     prints "<meanA> <meanB> <meanDiff> <se> <t> <n>"
 *
 * WHY PAIRED. Two arms of the same engine on the same tokens differ
 * position by position, and the two means alone cannot say whether a
 * gap of 0.006 out of 4.78 is systematic or is a handful of positions
 * moving. The per-position difference is the paired sample; its mean
 * over its own standard error is the statistic that answers it.
 *
 * B - A, so a NEGATIVE mean difference means B scores better (lower
 * NLL). |t| under about 2 is "these two arms are not distinguishable on
 * this sequence" - which is a result, not a failure to find one.
 *
 * Build: cc -O2 -o nll_paired build/nll_paired.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static long slurp(const char *p, float **out) {
    FILE *f = fopen(p, "rb");
    long n;
    float *b;
    if (!f) { fprintf(stderr, "nll_paired: cannot open %s\n", p); return -1; }
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
    double sa = 0.0, sb = 0.0, sd = 0.0, sdd = 0.0, md, var, se;
    if (argc != 3) { fprintf(stderr, "usage: nll_paired A.nll B.nll\n"); return 1; }
    na = slurp(argv[1], &a);
    nb = slurp(argv[2], &b);
    if (na < 0 || nb < 0) return 1;
    if (na != nb) { fprintf(stderr, "nll_paired: %ld vs %ld\n", na, nb); return 1; }
    for (i = 0; i < na; i++) {
        double d = (double)b[i] - (double)a[i];
        sa += a[i]; sb += b[i]; sd += d; sdd += d * d;
    }
    md  = sd / (double)na;
    var = (sdd - sd * sd / (double)na) / (double)(na > 1 ? na - 1 : 1);
    se  = sqrt(var / (double)na);
    printf("%.4f %.4f %+.5f %.5f %+.2f %ld\n",
           sa / na, sb / na, md, se, se > 0.0 ? md / se : 0.0, na);
    return 0;
}
