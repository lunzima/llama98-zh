/* lz4d.c - LZ4 frame decompression as a streaming reader (stunt).
 *
 * Two layers:
 *
 *   1. lz4_block_decompress - the LZ4 block format. Pure byte access,
 *      overlapping copy handled byte-by-byte, bounds-checked throughout.
 *      Algorithm follows lz4.c LZ4_decompress_generic (token's high 4
 *      bits = literal length, low 4 bits + 4 = match length, 15 is the
 *      extension escape; offset is 2 bytes LE).
 *
 *   2. LZ4Stream - parses the frame header, then serves decompressed
 *      bytes one block at a time into a small internal buffer, so a
 *      caller that only does `lz4d_read` sees a plain byte stream.
 *
 * BLOCK HEADER (from lz4frame.c LZ4F_decompress): the 4-byte LE value's
 * bit 31 (LZ4F_BLOCKUNCOMPRESSED_FLAG 0x80000000) marks an UNCOMPRESSED
 * block; bits 0-30 are the block size. There is NO last-block flag: the
 * frame ends when the content size is reached (this reader tracks it),
 * or when the next header cannot be read.
 *
 * BUFFER IS MCU-SIZED BY DESIGN: the decompression buffer is 64 KiB
 * (LZ4's smallest standard block), so streaming a model costs ~64 KiB of
 * extra memory no matter how large the model is. The producer must write
 * blocks of at most 64 KiB - BD block max size code 4, which is what a
 * standard LZ4 frame writer emits when asked for 64 KiB blocks.
 * A larger block is a hard error, not a silent truncation.
 *
 * Byte-level on purpose: no 32-bit word reads, so VC++ 4.0 on MIPS
 * (big-endian) decodes the same bytes the same way x86 does.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lz4d.h"
#include "lz_int.h"

#define LZ4D_MAGIC 0x184D2204u
#define LZ4D_BLOCKUNCOMPRESSED 0x80000000u
#define LZ4D_MAX_BLOCK 0x10000u   /* 64 KiB, MCU-sized */

/* ------------------------------------------------------------------ */
/* LZ4 block decompression                                            */
/* ------------------------------------------------------------------ */

/* DECLARATIONS FIRST, and `for (int i = ...)` nowhere: this file's own
   header says it is byte-level so that VC++ 4.0 on big-endian MIPS
   decodes the same bytes x86 does, and that compiler is C89. It was
   written with four C99 declarations, which build/c89_floor_gate.sh
   rejects and that compiler would not parse. Same code, hoisted. */
static int lz4_block_decompress(const uint8_t *src, int src_sz,
                                uint8_t *dst, int dst_cap) {
    const uint8_t *ip = src, *iend = src + src_sz;
    uint8_t *op = dst, *oend = dst + dst_cap;

    while (ip < iend) {
        uint32_t token = *ip++;
        int lit = (int)(token >> 4);
        uint32_t off;
        int mlen, i;
        if (lit == 15) {
            uint32_t b;
            do {
                if (ip >= iend) return -1;
                b = *ip++;
                lit += (int)b;
            } while (b == 255);
        }
        if (lit < 0 || ip + lit > iend || op + lit > oend) return -1;
        for (i = 0; i < lit; i++) *op++ = *ip++;
        if (ip >= iend) break;          /* last sequence is literals only */

        if (ip + 2 > iend) return -1;
        off = (uint32_t)ip[0] | ((uint32_t)ip[1] << 8);
        ip += 2;
        if (off == 0 || off > (uint32_t)(op - dst)) return -1;
        mlen = (int)(token & 15) + 4;
        if ((token & 15) == 15) {
            uint32_t b;
            do {
                if (ip >= iend) return -1;
                b = *ip++;
                mlen += (int)b;
            } while (b == 255);
        }
        if (mlen < 4 || op + mlen > oend) return -1;
        for (i = 0; i < mlen; i++) { *op = *(op - (int)off); op++; }
    }
    return (int)(op - dst);
}

/* ------------------------------------------------------------------ */
/* Frame stream                                                       */
/* ------------------------------------------------------------------ */

struct LZ4Stream {
    FILE *f;
    uint8_t *buf;               /* one block's decompressed bytes */
    size_t buf_cap;             /* 64 KiB, MCU-sized */
    int buf_pos, buf_len;       /* served region of buf */
    int eof, error;
    int has_content_len;        /* content size field present */
    lz_i64 content_len;         /* total uncompressed size */
    lz_i64 total;               /* bytes served so far */
};

static int rd_exact(FILE *f, void *dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

/* UNSIGNED, and the return type is the whole point. A block header
 * carries LZ4F_BLOCKUNCOMPRESSED_FLAG in bit 31, so the values this
 * reads routinely exceed INT_MAX. Returning them as `int` makes the
 * conversion implementation-defined, and `>> 31` on the negative
 * result after that is implementation-defined again - it happens to
 * give the right answer under gcc's arithmetic shift and would not
 * have to. This file exists to be byte-exact on a 1995 MIPS
 * compiler; it cannot lean on either. */
static uint32_t le32_get(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

LZ4Stream *lz4d_open(const char *path, char *errbuf, int errlen) {
    LZ4Stream *s;
    uint8_t hdr[4], flg, bd;
    int i;

    s = (LZ4Stream *)calloc(1, sizeof *s);
    if (!s) {
        if (errbuf && errlen > 0) {
            strncpy(errbuf, "lz4d: out of memory", (size_t)errlen);
            errbuf[errlen - 1] = 0;
        }
        return NULL;
    }
    s->f = fopen(path, "rb");
    if (!s->f) {
        if (errbuf && errlen > 0) {
            strncpy(errbuf, "lz4d: cannot open", (size_t)errlen);
            errbuf[errlen - 1] = 0;
        }
        free(s);
        return NULL;
    }
    s->buf = (uint8_t *)malloc(LZ4D_MAX_BLOCK);
    if (!s->buf) {
        fclose(s->f);
        free(s);
        if (errbuf && errlen > 0) {
            strncpy(errbuf, "lz4d: out of memory", (size_t)errlen);
            errbuf[errlen - 1] = 0;
        }
        return NULL;
    }
    s->buf_cap = LZ4D_MAX_BLOCK;

    if (!rd_exact(s->f, hdr, 4) || le32_get(hdr) != LZ4D_MAGIC) {
        s->error = 1;
        goto bad;
    }
    if (!rd_exact(s->f, &flg, 1) || !rd_exact(s->f, &bd, 1)) goto bad;
    if (((flg >> 6) & 3) != 1) goto bad;                 /* version 01 */
    if ((flg >> 4) & 1) goto bad;                        /* block checksum: unsupported */
    if (!((flg >> 5) & 1)) {                             /* block independence: REQUIRED */
        s->error = 1;
        goto bad;
    }
    if ((flg >> 3) & 1) {                                /* content size: 8 bytes LE */
        uint8_t cs[8];
        lz_i64 v = 0;
        if (!rd_exact(s->f, cs, 8)) goto bad;
        for (i = 7; i >= 0; i--) v = (v << 8) | (lz_i64)cs[i];
        s->has_content_len = 1;
        s->content_len = v;
    }
    if ((flg >> 0) & 1) {                                /* dict id: 4 bytes */
        uint8_t did[4];
        if (!rd_exact(s->f, did, 4)) goto bad;
    }
    if (!rd_exact(s->f, hdr, 1)) goto bad;               /* header checksum, ignored */

    /* BD block-size code is not used for sizing (lz4.frame writes codes
     * this reader cannot map); the buffer is fixed at 64 KiB. */
    (void)bd;
    return s;

bad:
    if (errbuf && errlen > 0) {
        strncpy(errbuf, "lz4d: not an LZ4 frame or bad header", (size_t)errlen);
        errbuf[errlen - 1] = 0;
    }
    fclose(s->f);
    free(s->buf);
    free(s);
    return NULL;
}

/* Refill buf with the next block's bytes. Returns 0 on EOF, 1 on
 * success, -1 on a stream error. */
static int next_block(LZ4Stream *s) {
    uint8_t hdr[4];
    int size, uncomp;

    if (s->error) return -1;
    if (s->has_content_len && s->total >= s->content_len) {
        s->eof = 1;
        return 0;
    }
    if (!rd_exact(s->f, hdr, 4)) {
        /* clean EOF at the end of content (or of a frame with no content
         * size); anything else is a truncated frame */
        if (!s->has_content_len || s->total >= s->content_len) {
            s->eof = 1;
            return 0;
        }
        s->error = 1;
        return -1;
    }
    size = (int)(le32_get(hdr) & 0x7fffffffu);
    uncomp = (le32_get(hdr) & LZ4D_BLOCKUNCOMPRESSED) != 0;
    if ((size_t)size > s->buf_cap) { s->error = 1; return -1; }
    if (uncomp) {                /* raw block: read verbatim */
        if (fread(s->buf, 1, (size_t)size, s->f) != (size_t)size) {
            s->error = 1;
            return -1;
        }
        s->buf_len = size;
    } else {
        uint8_t *c = (uint8_t *)malloc((size_t)size);
        int got;
        if (!c) { s->error = 1; return -1; }
        if (fread(c, 1, (size_t)size, s->f) != (size_t)size) {
            free(c);
            s->error = 1;
            return -1;
        }
        got = lz4_block_decompress(c, size, s->buf, (int)s->buf_cap);
        free(c);
        if (got < 0) { s->error = 1; return -1; }
        s->buf_len = got;
    }
    s->buf_pos = 0;
    if (s->has_content_len && s->total + (lz_i64)s->buf_len >= s->content_len) {
        s->eof = 1;
    }
    return 1;
}

size_t lz4d_read(LZ4Stream *s, void *dst, size_t sz) {
    uint8_t *out = (uint8_t *)dst;
    size_t done = 0;
    while (done < sz) {
        if (s->buf_pos >= s->buf_len) {
            int r = next_block(s);
            if (r <= 0) break;
        }
        {
            size_t avail = (size_t)(s->buf_len - s->buf_pos);
            size_t take = sz - done;
            if (take > avail) take = avail;
            memcpy(out + done, s->buf + s->buf_pos, take);
            s->buf_pos += (int)take;
            s->total += (lz_i64)take;
            done += take;
        }
    }
    return done;
}

int lz4d_eof(const LZ4Stream *s) { return s->eof && s->buf_pos >= s->buf_len; }
int lz4d_error(const LZ4Stream *s) { return s->error; }

void lz4d_close(LZ4Stream *s) {
    if (s->f) fclose(s->f);
    free(s->buf);
    free(s);
}

/* ------------------------------------------------------------------ */
/* Standalone round-trip test (LZ4D_TEST).                            */
/* ------------------------------------------------------------------ */
/* WHY A main() AND fprintf ARE ALLOWED HERE, when no other file in src/
   prints anything outside cli_main.c: the same reason lz_softfp.c's
   LZ_SOFTFP_STATS block gives - it is behind a macro no shipping build
   defines, so it does not breach src/'s console ban for a build nobody
   ships. Nothing links it: LZ4D_TEST is passed by hand.

       gcc -O2 -std=c99 -Isrc -DLZ4D_TEST -o lz4t src/lz4d.c
       ./lz4t MODEL/model.bin.lz4 out.bin && cmp MODEL/model.bin out.bin

   That round trip is the only direct check the decompressor has - the
   engine's own runs compare logits, which would hide a decompressor
   that was wrong in a way the arithmetic then smoothed over. Measured
   on _armgate2: 45,816,469 bytes, cmp clean.

   build/lz4_model_gate.sh runs exactly this, so the check has a home
   that is not a comment. */
#ifdef LZ4D_TEST
int main(int argc, char **argv) {
    LZ4Stream *s;
    static uint8_t out[1 << 20];
    FILE *f;
    long sz;
    size_t got;
    char eb[128];

    if (argc != 3) {
        fprintf(stderr, "usage: %s in.lz4 out.bin\n", argv[0]);
        return 2;
    }
    s = lz4d_open(argv[1], eb, sizeof eb);
    if (!s) { fprintf(stderr, "open failed: %s\n", eb); return 1; }
    f = fopen(argv[2], "wb");
    if (!f) return 1;
    for (;;) {
        got = lz4d_read(s, out, sizeof out);
        if (got == 0) break;
        fwrite(out, 1, got, f);
        if (got < sizeof out) break;
    }
    if (lz4d_error(s)) { fprintf(stderr, "stream error\n"); return 1; }
    sz = ftell(f);
    fprintf(stderr, "wrote %ld bytes, eof=%d\n", sz, lz4d_eof(s));
    fclose(f);
    lz4d_close(s);
    return 0;
}
#endif /* LZ4D_TEST */
