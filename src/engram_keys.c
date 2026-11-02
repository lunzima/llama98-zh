/* engram_keys.c - the static 2-gram key table: .npy loader and lookup.
 *
 * See engram_keys.h for what the table is and why it is a sidecar. The
 * file is numpy's own .npy format, so the parser here is deliberately
 * narrow: one dtype, one shape class, one order. Everything else is
 * refused with the reason, because a table read at the wrong stride
 * names a different n-gram at every position and no downstream check can
 * see it - the run stays finite and plausible.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "engram_keys.h"
#include "err.h"

/* numpy's magic: 0x93 followed by "NUMPY". */
static const unsigned char NPY_MAGIC[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };

/* A header longer than this is not a table this project writes (the
 * longest today is under 100 bytes); refusing keeps a corrupt length
 * from asking for an arbitrary allocation. */
#define NPY_MAX_HEADER 8192

/* Every refusal that is about the FILE's shape shares one message; the
   path and the reason are its two arguments. A non-constant code passed
   through a wrapper is what the error-localization gate forbids, so the
   code is fixed here rather than taken as a parameter. */
static int bad_format(char *errbuf, int errlen, const char *path,
                      const char *why) {
    lz_err_fmt(errbuf, errlen, LZ_ERR_ENGRAM_KEYS_FORMAT, path, why);
    return 1;
}

static unsigned long rd_le(const unsigned char *p, int n) {
    unsigned long v = 0;
    int i;
    for (i = n - 1; i >= 0; i--) v = (v << 8) | (unsigned long)p[i];
    return v;
}

int lz_keys_load(const char *path, lz_i64 **out, int *out_n,
                 char *errbuf, int errlen) {
    FILE *f;
    unsigned char pre[8];
    unsigned char *hdr = NULL;
    char *txt = NULL;
    unsigned char *raw = NULL;
    lz_i64 *keys = NULL;
    unsigned long hlen, n, got, i;
    int ver_major;

    *out = NULL;
    *out_n = 0;
    if (!path || !path[0]) {
        lz_err_fmt(errbuf, errlen, LZ_ERR_ENGRAM_KEYS_FILE, "", "empty path");
        return 1;
    }

    f = fopen(path, "rb");
    if (!f) {
        lz_err_fmt(errbuf, errlen, LZ_ERR_ENGRAM_KEYS_FILE, path,
                   "open failed");
        return 1;
    }
    if (fread(pre, 1, sizeof(pre), f) != sizeof(pre)) {
        fclose(f);
        return bad_format(errbuf, errlen, path, "shorter than the .npy preamble");
    }
    if (memcmp(pre, NPY_MAGIC, sizeof(NPY_MAGIC)) != 0) {
        fclose(f);
        return bad_format(errbuf, errlen, path, "no .npy magic");
    }
    ver_major = pre[6];
    if (ver_major != 1 && ver_major != 2 && ver_major != 3) {
        fclose(f);
        return bad_format(errbuf, errlen, path, "unsupported .npy version");
    }
    if (ver_major == 1) {
        unsigned char b[2];
        if (fread(b, 1, 2, f) != 2) {
            fclose(f);
            return bad_format(errbuf, errlen, path, "truncated header length");
        }
        hlen = rd_le(b, 2);
    } else {
        unsigned char b[4];
        if (fread(b, 1, 4, f) != 4) {
            fclose(f);
            return bad_format(errbuf, errlen, path, "truncated header length");
        }
        hlen = rd_le(b, 4);
    }
    if (hlen == 0 || hlen > NPY_MAX_HEADER) {
        fclose(f);
        return bad_format(errbuf, errlen, path, "header length out of range");
    }
    hdr = (unsigned char *)malloc(hlen + 1);
    if (!hdr) {
        fclose(f);
        lz_err_fmt(errbuf, errlen, LZ_ERR_ENGRAM_KEYS_FILE, path, "out of memory");
        return 1;
    }
    if (fread(hdr, 1, hlen, f) != hlen) {
        fclose(f);
        return bad_format(errbuf, errlen, path, "truncated header");
    }
    hdr[hlen] = '\0';
    txt = (char *)hdr;

    /* dtype: exactly the little-endian 64-bit signed integer numpy
       writes for an int64 array on every platform this project runs. */
    {
        const char *d = strstr(txt, "'descr':");
        if (!d || !strstr(d, "'<i8'")) {
            fclose(f);
            return bad_format(errbuf, errlen, path, "descr is not '<i8'");
        }
    }
    {
        const char *o = strstr(txt, "'fortran_order':");
        if (!o || !strstr(o, "False")) {
            fclose(f);
            return bad_format(errbuf, errlen, path, "fortran_order is not False");
        }
    }
    {
        const char *s = strstr(txt, "'shape':");
        const char *p;
        n = 0;
        if (!s) {
            fclose(f);
            return bad_format(errbuf, errlen, path, "no shape");
        }
        p = strchr(s, '(');
        if (!p) {
            fclose(f);
            return bad_format(errbuf, errlen, path, "malformed shape");
        }
        p++;
        while (*p == ' ') p++;
        if (*p < '0' || *p > '9') {
            fclose(f);
            return bad_format(errbuf, errlen, path, "shape is not one dimension");
        }
        while (*p >= '0' && *p <= '9') {
            n = n * 10 + (unsigned long)(*p - '0');
            if (n > 0x7fffffffUL) {
                fclose(f);
                return bad_format(errbuf, errlen, path, "row count does not fit an int");
            }
            p++;
        }
        while (*p == ' ') p++;
        /* A 1-D numpy shape is "(n,)"; anything else is refused. */
        if (*p != ',') {
            fclose(f);
            return bad_format(errbuf, errlen, path, "shape is not one dimension");
        }
    }
    if (n == 0) {
        fclose(f);
        return bad_format(errbuf, errlen, path, "table is empty");
    }

    raw = (unsigned char *)malloc((size_t)n * 8);
    keys = (lz_i64 *)malloc((size_t)n * sizeof(lz_i64));
    if (!raw || !keys) {
        fclose(f);
        free(raw); free(keys); free(hdr);
        lz_err_fmt(errbuf, errlen, LZ_ERR_ENGRAM_KEYS_FILE, path, "out of memory");
        return 1;
    }
    got = (unsigned long)fread(raw, 1, (size_t)n * 8, f);
    if (got != n * 8) {
        fclose(f);
        free(raw); free(keys); free(hdr);
        return bad_format(errbuf, errlen, path, "truncated payload");
    }
    /* No trailing bytes: a file with more is not the table its header
       describes, and reading only the prefix would look healthy. */
    if (fgetc(f) != EOF) {
        fclose(f);
        free(raw); free(keys); free(hdr);
        return bad_format(errbuf, errlen, path, "trailing bytes after the payload");
    }
    fclose(f);
    free(hdr);
    hdr = NULL;

    /* Byte assembly, not a cast: the payload is little-endian whatever
       the host is (VC++ 4.0 MIPS targets are big-endian). */
    for (i = 0; i < n; i++) {
        uint64_t v = 0;
        const unsigned char *p = raw + (size_t)i * 8;
        int b;
        for (b = 7; b >= 0; b--) v = (v << 8) | (uint64_t)p[b];
        keys[i] = (lz_i64)v;
        if (i > 0 && !(keys[i] > keys[i - 1])) {
            free(raw);
            free(keys);
            return bad_format(errbuf, errlen, path, "keys are not strictly increasing");
        }
    }
    free(raw);
    *out = keys;
    *out_n = (int)n;
    return 0;
}

int lz_keys_find(const lz_i64 *keys, int n, lz_i64 code) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (keys[mid] == code) return mid;
        if (keys[mid] < code) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
