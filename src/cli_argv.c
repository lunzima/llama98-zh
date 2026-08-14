#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "cli_argv.h"
#include "gbk.h"

/* 936 is Windows' code page id for GBK (Simplified Chinese) - the only
   one lz_gbk_to_utf8 knows. Named rather than inlined so the one number
   this whole file's correctness hinges on is not buried in an if. */
#define LZ_CP_GBK 936

char **lz_gbk_argv_convert(int cp, int argc, char **argv) {
    char **out;
    int i, j;

    if (cp != LZ_CP_GBK || argc <= 0 || !argv) return NULL;

    out = (char **)calloc((size_t)argc, sizeof(char *));
    if (!out) return NULL;

    for (i = 0; i < argc; i++) {
        int len, need;
        char *buf;

        if (!argv[i]) { out[i] = NULL; continue; }
        len = (int)strlen(argv[i]);
        /* Query-then-allocate-exact, lz_gbk_to_utf8's own snprintf
           convention (src/gbk.h): size first, allocate exactly that
           many bytes plus the terminator, convert once into a buffer
           that cannot be too small. */
        need = lz_gbk_to_utf8(argv[i], len, NULL, 0, NULL);
        buf = (char *)malloc((size_t)need + 1);
        if (!buf) {
            for (j = 0; j < i; j++) free(out[j]);
            free(out);
            return NULL;
        }
        lz_gbk_to_utf8(argv[i], len, buf, need + 1, NULL);
        out[i] = buf;
    }
    return out;
}
