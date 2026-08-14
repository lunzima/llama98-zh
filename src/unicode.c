/* Unicode utility implementation: NFC normalization + char classification.
 *
 * NFC algorithm (UAX #15):
 *   1. canonical decomposition (look up LZ_UNI_DECOMP_*, tables already
 *      recursively expanded to full sequences)
 *   2. canonical ordering (stable sort by CCC within a segment; a segment
 *      runs from one starter to the next)
 *   3. canonical composition (scan backward for the first non-blocking
 *      char P; if (P, c) is in the composition table, combine; the result
 *      keeps participating in further composition)
 *
 * Byte-identical to Python unicodedata.normalize('NFC') (guaranteed by
 * a differential test).
 */
#include <stdlib.h>
#include <string.h>

#include "unicode.h"
#include "err.h"
#include "unicode_tables.h"

/* ------------------------------------------------------------ UTF-8 */

int lz_utf8_decode(const char *s, int len, uint32_t *cps, int maxcp) {
    const unsigned char *b = (const unsigned char *)s;
    int n = 0, i = 0;
    while (i < len && n < maxcp) {
        unsigned char c0 = b[i];
        uint32_t cp;
        int extra, k, ok = 1;
        if (c0 < 0x80) { cp = c0; extra = 0; }
        else if ((c0 & 0xE0) == 0xC0) { cp = c0 & 0x1F; extra = 1; }
        else if ((c0 & 0xF0) == 0xE0) { cp = c0 & 0x0F; extra = 2; }
        else if ((c0 & 0xF8) == 0xF0) { cp = c0 & 0x07; extra = 3; }
        else { cps[n++] = 0xFFFD; i++; continue; }
        if (i + extra >= len) { cps[n++] = 0xFFFD; i++; continue; }
        for (k = 1; k <= extra; k++) {
            if ((b[i + k] & 0xC0) != 0x80) { ok = 0; break; }
            cp = (cp << 6) | (b[i + k] & 0x3F);
        }
        if (!ok) { cps[n++] = 0xFFFD; i++; continue; }
        if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) ||
            (extra == 3 && cp < 0x10000) ||
            (cp >= 0xD800 && cp <= 0xDFFF)) {
            cps[n++] = 0xFFFD; i++; continue;
        }
        cps[n++] = cp;
        i += extra + 1;
    }
    return n;
}

int lz_utf8_encode(uint32_t cp, unsigned char *out) {
    if (cp < 0x80) { out[0] = (unsigned char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (unsigned char)(0xC0 | (cp >> 6));
        out[1] = (unsigned char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (unsigned char)(0xE0 | (cp >> 12));
        out[1] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (unsigned char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (unsigned char)(0xF0 | (cp >> 18));
    out[1] = (unsigned char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (unsigned char)(0x80 | (cp & 0x3F));
    return 4;
}

int lz_utf8_valid(const char *s, int n) {
    /* Strict yes/no validator. This is the "is this really UTF-8, not
       GBK that happens to fit" question cli_main.c's console sniff asks;
       it produces no codepoints and rejects anything the least bit off.
       The lead-byte ranges are the one place it is (and must be)
       stricter than lz_utf8_decode: that function's 4-byte lead test
       `(c0 & 0xF8) == 0xF0` admits 0xF5..0xF7 (codepoints past
       U+10FFFF), while this one pins the lead to 0xF0..0xF4 so no such
       value is ever assembled. The two only differ in what they DO with
       an invalid byte - reject vs U+FFFD - so an input this returns 1
       for decodes to exactly the same codepoints. */
    int i = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        int need;
        unsigned long cp;
        if (c < 0x80) { i++; continue; }
        if (c >= 0xC2 && c <= 0xDF)      { need = 1; cp = c & 0x1Fu; }
        else if (c >= 0xE0 && c <= 0xEF) { need = 2; cp = c & 0x0Fu; }
        else if (c >= 0xF0 && c <= 0xF4) { need = 3; cp = c & 0x07u; }
        else return 0;                   /* 0x80..0xC1, 0xF5..0xFF */
        if (i + need >= n) return 0;
        {
            int k;
            for (k = 1; k <= need; k++) {
                unsigned char t = (unsigned char)s[i + k];
                if (t < 0x80 || t > 0xBF) return 0;
                cp = (cp << 6) | (unsigned long)(t & 0x3Fu);
            }
        }
        if (need == 1 && cp < 0x80) return 0;
        if (need == 2 && cp < 0x800) return 0;
        if (need == 3 && cp < 0x10000) return 0;
        if (cp > 0x10FFFFul) return 0;
        if (cp >= 0xD800ul && cp <= 0xDFFFul) return 0;
        i += need + 1;
    }
    return 1;
}

/* ------------------------------------------------------------ table lookup */

static int range_has(const uint32_t *tab, int n, uint32_t cp) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        uint32_t a = tab[(size_t)mid * 2];
        uint32_t b = tab[(size_t)mid * 2 + 1];
        if (cp < a) hi = mid - 1;
        else if (cp > b) lo = mid + 1;
        else return 1;
    }
    return 0;
}

int lz_uni_is_letter(uint32_t cp) { return range_has(LZ_UNI_LETTER, LZ_UNI_LETTER_N, cp); }
int lz_uni_is_mark(uint32_t cp)   { return range_has(LZ_UNI_MARK, LZ_UNI_MARK_N, cp); }
int lz_uni_is_number(uint32_t cp) { return range_has(LZ_UNI_NUMBER, LZ_UNI_NUMBER_N, cp); }
int lz_uni_is_space(uint32_t cp)  { return range_has(LZ_UNI_SPACE, LZ_UNI_SPACE_N, cp); }

int lz_uni_ccc(uint32_t cp) {
    int lo = 0, hi = LZ_UNI_CCC_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        uint32_t c = LZ_UNI_CCC_CP[mid];
        if (cp == c) return LZ_UNI_CCC_VAL[mid];
        if (cp < c) hi = mid - 1;
        else lo = mid + 1;
    }
    return 0;
}

/* canonical decomposition: returns sequence pointer + length; NULL with
   len=0 when there is no decomposition */
static const uint32_t *uni_decomp(uint32_t cp, int *len) {
    /* Returned to the caller, so it must outlive this frame. Single
       threaded, and the caller consumes it before the next call. */
    static uint32_t buf[3];
    /* Hangul syllables (UAX#15 algorithmic decomposition; deliberately
       NOT in the generated data tables - 11172 entries would dwarf the
       rest of the table):
         SIndex = S - AC00
         L = 1100 + SIndex/588
         V = 1161 + (SIndex%588)/28
         T = 11A7 + SIndex%28   (only when SIndex%28 is non-zero) */
    if (cp >= 0xAC00 && cp <= 0xD7A3) {
        uint32_t si = cp - 0xAC00;
        uint32_t t = si % 28;
        buf[0] = 0x1100 + si / 588;
        buf[1] = 0x1161 + (si % 588) / 28;
        if (t) {
            buf[2] = 0x11A7 + t;
            *len = 3;
        } else {
            *len = 2;
        }
        return buf;
    }
    {
        int lo = 0, hi = LZ_UNI_DECOMP_N - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            uint32_t c = LZ_UNI_DECOMP_CP[mid];
            if (cp == c) {
                *len = LZ_UNI_DECOMP_LEN[mid];
                return LZ_UNI_DECOMP_SEQ + LZ_UNI_DECOMP_OFF[mid];
            }
            if (cp < c) hi = mid - 1;
            else lo = mid + 1;
        }
    }
    *len = 0;
    return NULL;
}

/* canonical composition: (first, second) -> result; -1 if not in table */
static int uni_compose(uint32_t first, uint32_t second) {
    /* Hangul algorithmic composition: L+V -> LV; LV+T -> LVT */
    if (first >= 0x1100 && first <= 0x1112 &&
        second >= 0x1161 && second <= 0x1175) {
        return (int)(0xAC00 +
                     ((first - 0x1100) * 21 + (second - 0x1161)) * 28);
    }
    if (first >= 0xAC00 && first <= 0xD7A3 &&
        (first - 0xAC00) % 28 == 0 &&
        second >= 0x11A8 && second <= 0x11C2) {
        return (int)(first + (second - 0x11A7));
    }
    {
        int lo = 0, hi = LZ_UNI_COMP_N - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            const uint32_t *e = LZ_UNI_COMP + (size_t)mid * 3;
            if (first == e[0]) {
                if (second == e[1]) return (int)e[2];
                if (second < e[1]) hi = mid - 1;
                else lo = mid + 1;
            } else if (first < e[0]) {
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
    }
    return -1;
}

/* ------------------------------------------------------------ NFC */

int lz_utf8_nfc(const char *in, int len, char *out, int outcap) {
    uint32_t *cps;
    uint32_t *work;          /* decomposed codepoint stream */
    uint32_t *composed;      /* composed codepoint stream */
    int ncp, nw, n2 = 0;
    int i, j;

    if (len <= 0) return 0;
    if (len > 0x7FFFFFF / 4) return -1;      /* overflow guard */

    cps = (uint32_t *)malloc((size_t)len * sizeof(uint32_t));
    work = (uint32_t *)malloc((size_t)len * 4 * sizeof(uint32_t));
    composed = (uint32_t *)malloc((size_t)len * 4 * sizeof(uint32_t));
    if (!cps || !work || !composed) {
        free(cps); free(work); free(composed);
        return -1;
    }

    ncp = lz_utf8_decode(in, len, cps, len);

    /* 1. canonical decomposition (tables pre-expanded; single lookup) */
    nw = 0;
    for (i = 0; i < ncp; i++) {
        int dlen;
        const uint32_t *d = uni_decomp(cps[i], &dlen);
        if (d) {
            for (j = 0; j < dlen; j++) work[nw++] = d[j];
        } else {
            work[nw++] = cps[i];
        }
    }

    /* 2. canonical ordering: stable insertion sort within each segment
          (starter through the char before the next starter) */
    for (i = 0; i < nw; ) {
        int seg_end = i + 1;
        while (seg_end < nw && lz_uni_ccc(work[seg_end]) > 0) seg_end++;
        for (j = i + 1; j < seg_end; j++) {
            uint32_t v = work[j];
            int cj = lz_uni_ccc(v);
            int k = j - 1;
            while (k >= i && lz_uni_ccc(work[k]) > cj) {
                work[k + 1] = work[k];
                k--;
            }
            work[k + 1] = v;
        }
        i = seg_end;
    }

    /* 3. canonical composition. A starter (ccc 0) only tries to combine
          with its immediate predecessor (no chars in between; Hangul L+V
          is such a case; plain A+B is not in the table and naturally
          fails). A combining char scans backward from last: a char with
          ccc >= its own blocks it (order is non-decreasing, so anything
          earlier is blocked by the same char); every non-blocking char
          before it is tried - a failed combination does not stop the
          scan, since an earlier char may be the correct partner
          (e.g. L+0301+031B+3099). */
    for (i = 0; i < nw; i++) {
        uint32_t c = work[i];
        int cc = lz_uni_ccc(c);
        if (cc == 0) {
            if (n2 > 0) {
                int r = uni_compose(composed[n2 - 1], c);
                if (r >= 0) {
                    composed[n2 - 1] = (uint32_t)r;
                    continue;       /* composed result may be a starter (e.g. Hangul syllable) */
                }
            }
            composed[n2++] = c;
            continue;
        }
        {
            int done = 0;
            for (j = n2 - 1; j >= 0; j--) {
                /* If a char B sits between composed[j] and c: B being a
                   starter or ccc(B) >= ccc(c) blocks composed[j], and
                   anything earlier is blocked by the same B - stop. */
                if (j < n2 - 1) {
                    uint32_t b = composed[j + 1];
                    int bc = lz_uni_ccc(b);
                    if (bc == 0 || bc >= cc) break;
                }
                {
                    int r = uni_compose(composed[j], c);
                    if (r >= 0) {
                        composed[j] = (uint32_t)r;
                        done = 1;
                        break;
                    }
                }
                /* combination failed: keep scanning backward (this char
                   does not block c; it may be a candidate or intermediary) */
            }
            if (!done) composed[n2++] = c;
        }
    }

    /* 4. encode back to UTF-8 */
    {
        int olen = 0;
        for (i = 0; i < n2; i++) {
            unsigned char ub[4];
            int ulen = lz_utf8_encode(composed[i], ub);
            if (olen + ulen > outcap) {
                free(cps); free(work); free(composed);
                return -1;
            }
            memcpy(out + olen, ub, (size_t)ulen);
            olen += ulen;
        }
        free(cps); free(work); free(composed);
        return olen;
    }
}
