/* GBK <-> UTF-8 transcoding. Table-driven, no OS calls, no allocation.
 *
 * Tables are generated, not hand-written (src/gbk_tables.h): a dense
 * [lead][trail] decode array and a paged encode array. Both directions
 * are O(1) per character, which matters because the streaming display
 * path converts every chunk the sampler produces.
 *
 * The UTF-8 side deliberately repeats lz_utf8_decode's validation rather
 * than calling it: that function decodes into a code point array and
 * reports how many CODE POINTS it produced, and the streaming caller
 * needs to know how many INPUT BYTES were consumed so it can carry a
 * split sequence into the next chunk. Going through an intermediate
 * array would also mean a bound on chunk size, a second buffer, and a
 * place for the two length units to be confused.
 *
 * Validation must stay identical to lz_utf8_decode's - overlong forms,
 * surrogates and bad continuation bytes all become U+FFFD, and recovery
 * advances exactly one byte.
 */
#include "gbk.h"
#include "unicode.h"
#include "gbk_tables.h"

#define LZ_GBK_FALLBACK '?'

/* Append n bytes, refusing to emit a partial character.
 *
 * `full` latches once anything has been refused. Without it a character
 * too big for the remaining room would be skipped while a SHORTER one
 * after it still fit, and the output would silently lose a character
 * from the middle rather than the end.
 *
 * `need` and `wrote` are separate counters and stop agreeing the moment
 * the buffer fills - `need` is the snprintf-style return value, `wrote`
 * is where the NUL goes. Collapsing them into one puts the terminator at
 * cap-1 with uninitialised bytes in front of it. */
static void emit(char *out, int cap, int *need, int *wrote, int *full,
                 const unsigned char *b, int n) {
    if (out && !*full) {
        if (*wrote + n < cap) {
            int k;
            for (k = 0; k < n; k++) out[*wrote + k] = (char)b[k];
            *wrote += n;
        } else {
            *full = 1;
        }
    }
    *need += n;
}

/* How many bytes follow the lead byte c0, or -1 if it cannot lead. */
static int utf8_extra(unsigned char c0) {
    if (c0 < 0x80) return 0;
    if ((c0 & 0xE0) == 0xC0) return 1;
    if ((c0 & 0xF0) == 0xE0) return 2;
    if ((c0 & 0xF8) == 0xF0) return 3;
    return -1;
}

int lz_gbk_from_utf8(const char *in, int len, char *out, int cap, int *used) {
    const unsigned char *b = (const unsigned char *)in;
    int i = 0, need = 0, wrote = 0, full = 0;

    if (!in || len < 0) len = 0;
    while (i < len) {
        unsigned char c0 = b[i];
        unsigned char buf[2];
        uint32_t cp;
        int extra = utf8_extra(c0);
        int adv = 1, k, ok = 1;

        if (extra < 0) {
            cp = 0xFFFD;
        } else if (i + extra >= len) {
            /* Not enough bytes left. Only a valid PREFIX may be held for
               the next chunk; "E4 41" at the tail is broken, not
               incomplete, and holding it would stall the stream forever
               waiting for a byte that will never make it valid. */
            int prefix = 1;
            for (k = i + 1; k < len; k++) {
                if ((b[k] & 0xC0) != 0x80) { prefix = 0; break; }
            }
            if (used && prefix) break;
            cp = 0xFFFD;
        } else {
            cp = (uint32_t)(extra == 0 ? c0 :
                            extra == 1 ? (c0 & 0x1F) :
                            extra == 2 ? (c0 & 0x0F) : (c0 & 0x07));
            for (k = 1; k <= extra; k++) {
                if ((b[i + k] & 0xC0) != 0x80) { ok = 0; break; }
                cp = (cp << 6) | (uint32_t)(b[i + k] & 0x3F);
            }
            if (!ok ||
                (extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) ||
                (extra == 3 && cp < 0x10000) ||
                (cp >= 0xD800 && cp <= 0xDFFF)) {
                cp = 0xFFFD;
            } else {
                adv = extra + 1;
            }
        }

        if (cp < 0x80) {
            buf[0] = (unsigned char)cp;
            emit(out, cap, &need, &wrote, &full, buf, 1);
        } else {
            uint16_t pair = 0;
            if (cp <= 0xFFFF) {
                uint8_t pg = LZ_GBK_ENC_PAGE[cp >> 8];
                if (pg != 0xFF) pair = LZ_GBK_ENC[pg][cp & 0xFF];
            }
            if (pair == 0) {
                buf[0] = LZ_GBK_FALLBACK;
                emit(out, cap, &need, &wrote, &full, buf, 1);
            } else {
                buf[0] = (unsigned char)(pair >> 8);
                buf[1] = (unsigned char)(pair & 0xFF);
                emit(out, cap, &need, &wrote, &full, buf, 2);
            }
        }
        i += adv;
    }

    if (used) *used = i;
    if (out && cap > 0) out[wrote] = '\0';
    return need;
}

int lz_gbk_to_utf8(const char *in, int len, char *out, int cap, int *used) {
    const unsigned char *b = (const unsigned char *)in;
    int i = 0, need = 0, wrote = 0, full = 0;

    if (!in || len < 0) len = 0;
    while (i < len) {
        unsigned char c0 = b[i];
        unsigned char buf[4];
        uint32_t cp = 0xFFFD;
        int adv = 1, n;

        if (c0 < 0x80) {
            cp = c0;
        } else if (c0 >= LZ_GBK_LEAD_LO && c0 <= LZ_GBK_LEAD_HI) {
            if (i + 1 >= len) {
                /* A lone lead byte at the tail is the split-character
                   case; a caller that is streaming holds it. */
                if (used) break;
            } else {
                unsigned char t = b[i + 1];
                uint16_t v = 0;
                if (t >= LZ_GBK_TRAIL_LO && t <= LZ_GBK_TRAIL_HI) {
                    v = LZ_GBK_DEC[c0 - LZ_GBK_LEAD_LO][t - LZ_GBK_TRAIL_LO];
                }
                if (v != 0) { cp = v; adv = 2; }
            }
        }

        n = lz_utf8_encode(cp, buf);
        emit(out, cap, &need, &wrote, &full, buf, n);
        i += adv;
    }

    if (used) *used = i;
    if (out && cap > 0) out[wrote] = '\0';
    return need;
}
