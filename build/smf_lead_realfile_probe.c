/* Prints the lead-in (in file ticks) lz_smf_open() computes for a real
 * MIDI file named on argv[1]. Used only by build/smf_lead_gate.sh to
 * compare the fixed tree's answer against the reverted-guard mutant's
 * on a file that actually exercises the defect - a synthetic fixture
 * proves the mechanism, this confirms the report.
 */
#include <stdio.h>
#include <stdlib.h>

#include "music.h"
#include "smf.h"

int main(int argc, char **argv) {
    LZSmfReader r;
    FILE *f;
    long n;
    unsigned char *img;

    if (argc < 2) { fprintf(stderr, "usage: realfile <path.mid>\n"); return 2; }
    f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "could not open %s\n", argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    img = (unsigned char *)malloc((size_t)n);
    if (!img || (long)fread(img, 1, (size_t)n, f) != n) {
        fprintf(stderr, "could not read %s\n", argv[1]);
        fclose(f);
        return 2;
    }
    fclose(f);

    if (lz_smf_open(&r, img, n, LZ_MUSIC_AS_WRITTEN, 100) != LZ_SMF_OK) {
        fprintf(stderr, "lz_smf_open refused %s\n", argv[1]);
        free(img);
        return 2;
    }
    printf("lead=%lu\n", r.lead);
    free(img);
    return 0;
}
