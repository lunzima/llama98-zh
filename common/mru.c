#include <string.h>

#include "mru.h"

/* GBK lead-byte range (src/gbk_tables.h: LZ_GBK_LEAD_LO/HI). Duplicated
   as two constants rather than pulled in via #include "gbk_tables.h" -
   that header is a ~4700-line generated decode/encode table built for
   gbk.c's transcoding, and this leaf module (see mru.h: "Pure: no
   Win32, no ini, no menu") only ever needs the lead-byte range, never
   a decode. */
#define GBK_LEAD_LO 0x81
#define GBK_LEAD_HI 0xFE

static int ci_eq(const char *a, const char *b) {
    /* Written rather than taken from the C library: the name is
       _stricmp on one toolchain and strcasecmp on the other, and this
       is four lines.
       Folding 'A'-'Z' byte-by-byte is only correct where every byte is
       either ASCII or the FIRST byte of a GBK pair - it is wrong at a
       GBK trail byte, because the trail byte is what selects the code
       point, not a letter case. The GBK trail range is 0x40-0xFE,
       which fully contains both 'A'-'Z' (0x41-0x5A) and 'a'-'z'
       (0x61-0x7A), so for any GBK lead byte L, "L A" and "L a" are two
       distinct characters (e.g. 0xB0 0x41 vs 0xB0 0x61) - not the same
       byte spelled two ways. Folding the trail byte's case would merge
       them, and lz_mru_push would then treat two different model
       directories as one, silently overwriting one path with the
       other.
       So: when a byte in [GBK_LEAD_LO, GBK_LEAD_HI] starts a pair on
       both sides, compare that byte and the one after it as an opaque
       2-byte unit, case untouched, and advance by 2. A lead byte with
       nothing after it (a string truncated mid-character) falls
       through to the single-byte path below instead - it is compared,
       and folded, as one ordinary byte, which never reads past the
       string's NUL. */
    while (*a && *b) {
        unsigned char x = (unsigned char)*a, y = (unsigned char)*b;
        if (x >= GBK_LEAD_LO && x <= GBK_LEAD_HI && a[1] &&
            y >= GBK_LEAD_LO && y <= GBK_LEAD_HI && b[1]) {
            if (x != y || (unsigned char)a[1] != (unsigned char)b[1])
                return 0;
            a += 2;
            b += 2;
            continue;
        }
        if (x >= 'A' && x <= 'Z') x = (unsigned char)(x - 'A' + 'a');
        if (y >= 'A' && y <= 'Z') y = (unsigned char)(y - 'A' + 'a');
        if (x != y) return 0;
        a++;
        b++;
    }
    return *a == *b;
}

void lz_mru_init(LZMru *m) { if (m) m->n = 0; }

void lz_mru_push(LZMru *m, const char *path) {
    int i, hit = -1;
    if (!m || !path || !path[0]) return;
    if ((int)strlen(path) >= LZ_MRU_LEN) return;
    for (i = 0; i < m->n; i++) if (ci_eq(m->item[i], path)) { hit = i; break; }
    if (hit < 0) {
        hit = (m->n < LZ_MRU_MAX) ? m->n++ : LZ_MRU_MAX - 1;
    }
    for (i = hit; i > 0; i--) strcpy(m->item[i], m->item[i - 1]);
    strcpy(m->item[0], path);
}

void lz_mru_remove(LZMru *m, const char *path) {
    int i, j;
    if (!m || !path) return;
    for (i = 0; i < m->n; i++) {
        if (!ci_eq(m->item[i], path)) continue;
        for (j = i; j + 1 < m->n; j++) strcpy(m->item[j], m->item[j + 1]);
        m->n--;
        return;
    }
}
