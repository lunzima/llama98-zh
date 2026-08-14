#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "err.h"
#include "safetensors.h"

/* A header should never reach this size; rejects malformed files and
   avoids malloc'ing astronomical numbers based on them */
#define LZ_ST_MAX_HEADER  (64ULL * 1024ULL * 1024ULL)

static void sterr(char *errbuf, int errlen, LZErr code, ...) {
    va_list ap;
    if (!errbuf || errlen <= 0) return;
    va_start(ap, code);
    lz_err_fmt_v(errbuf, errlen, code, ap);
    va_end(ap);
}

static const struct { const char *name; int dt; int size; } DTYPES[] = {
    { "BF16", LZ_DT_BF16, 2 },
    { "F16",  LZ_DT_F16,  2 },
    { "F32",  LZ_DT_F32,  4 },
    { "F64",  LZ_DT_F64,  8 },
    { "I8",   LZ_DT_I8,   1 },
    { "U8",   LZ_DT_U8,   1 },
    { "I16",  LZ_DT_I16,  2 },
    { "I32",  LZ_DT_I32,  4 },
    { "I64",  LZ_DT_I64,  8 },
    { "BOOL", LZ_DT_BOOL, 1 }
};
#define N_DTYPES ((int)(sizeof(DTYPES) / sizeof(DTYPES[0])))

const char *lz_st_dtype_name(int dt) {
    int i;
    for (i = 0; i < N_DTYPES; i++) if (DTYPES[i].dt == dt) return DTYPES[i].name;
    return "?";
}

int lz_st_dtype_size(int dt) {
    int i;
    for (i = 0; i < N_DTYPES; i++) if (DTYPES[i].dt == dt) return DTYPES[i].size;
    return 0;
}

static int dtype_from_name(const char *s) {
    int i;
    if (!s) return LZ_DT_UNKNOWN;
    for (i = 0; i < N_DTYPES; i++)
        if (strcmp(DTYPES[i].name, s) == 0) return DTYPES[i].dt;
    return LZ_DT_UNKNOWN;
}

int lz_st_open(LZSafetensors *st, const char *path, char *errbuf, int errlen) {
    unsigned char lenbuf[8];
    unsigned long long hdr_len = 0;
    char *hdr = NULL;
    const LZJsonNode *root, *m;
    int i;
    unsigned long long max_end = 0;

    memset(st, 0, sizeof(*st));
    if (errbuf && errlen > 0) errbuf[0] = '\0';

    st->file_size = lz_fsize(path);
    if (st->file_size < 8) {
        sterr(errbuf, errlen, LZ_ERR_ST_NOT_FOUND, path);
        return 1;
    }

    st->fp = fopen(path, "rb");
    if (!st->fp) {
        sterr(errbuf, errlen, LZ_ERR_ST_OPEN, path);
        return 1;
    }
    if (fread(lenbuf, 1, 8, st->fp) != 8) {
        sterr(errbuf, errlen, LZ_ERR_ST_HEADER_LEN);
        goto fail;
    }
    /* assemble little-endian u64 byte by byte: no host-endianness or alignment assumptions */
    for (i = 7; i >= 0; i--) hdr_len = (hdr_len << 8) | (unsigned long long)lenbuf[i];

    if (hdr_len == 0 || hdr_len > LZ_ST_MAX_HEADER ||
        hdr_len + 8ULL > (unsigned long long)st->file_size) {
        sterr(errbuf, errlen, LZ_ERR_ST_HEADER_BAD,
              hdr_len, st->file_size);
        goto fail;
    }
    st->data_start = 8ULL + hdr_len;

    hdr = (char *)malloc((size_t)hdr_len);
    if (!hdr) {
        sterr(errbuf, errlen, LZ_ERR_ST_HEADER_ALLOC, hdr_len);
        goto fail;
    }
    if (fread(hdr, 1, (size_t)hdr_len, st->fp) != (size_t)hdr_len) {
        sterr(errbuf, errlen, LZ_ERR_ST_HEADER_READ);
        goto fail;
    }
    if (lz_json_parse(&st->json, hdr, (size_t)hdr_len, errbuf, errlen) != 0)
        goto fail;
    free(hdr);
    hdr = NULL;

    root = lz_json_root(&st->json);
    if (!root || root->type != LZ_JSON_OBJ) {
        sterr(errbuf, errlen, LZ_ERR_ST_HEADER_JSON);
        goto fail;
    }

    if (root->n_children == 0) {
        sterr(errbuf, errlen, LZ_ERR_ST_NO_TENSORS);
        goto fail;
    }
    st->tensors = (LZStTensor *)calloc((size_t)root->n_children,
                                       sizeof(LZStTensor));
    if (!st->tensors) {
        sterr(errbuf, errlen, LZ_ERR_ST_TENSOR_ALLOC);
        goto fail;
    }

    for (m = lz_json_first(&st->json, root); m;
         m = lz_json_next(&st->json, m)) {
        LZStTensor *t;
        const LZJsonNode *sh, *off, *e;
        const char *dts;
        long long s0, e0;

        /* __metadata__ is the format-mandated non-tensor member; skip */
        if (m->key && strcmp(m->key, "__metadata__") == 0) continue;
        if (m->type != LZ_JSON_OBJ) {
            sterr(errbuf, errlen, LZ_ERR_ST_DESC,
                  m->key ? m->key : "?");
            goto fail;
        }

        t = &st->tensors[st->n_tensors];
        t->name = m->key;

        dts = lz_json_get_str(&st->json, m, "dtype", NULL);
        t->dtype = dtype_from_name(dts);
        if (t->dtype == LZ_DT_UNKNOWN) {
            sterr(errbuf, errlen, LZ_ERR_ST_DTYPE,
                  t->name, dts ? dts : "(missing)");
            goto fail;
        }

        sh = lz_json_get(&st->json, m, "shape");
        if (!sh || sh->type != LZ_JSON_ARR) {
            sterr(errbuf, errlen, LZ_ERR_ST_SHAPE_MISSING, t->name);
            goto fail;
        }
        if (sh->n_children > LZ_ST_MAX_DIMS) {
            sterr(errbuf, errlen, LZ_ERR_ST_DIMS, t->name, sh->n_children);
            goto fail;
        }
        t->n_dims = sh->n_children;
        t->n_elem = 1;
        i = 0;
        for (e = lz_json_first(&st->json, sh); e;
             e = lz_json_next(&st->json, e), i++) {
            if (e->type != LZ_JSON_NUM || e->num < 0) {
                sterr(errbuf, errlen, LZ_ERR_ST_SHAPE, t->name);
                goto fail;
            }
            t->shape[i] = (long long)e->num;
            t->n_elem *= t->shape[i];
        }

        off = lz_json_get(&st->json, m, "data_offsets");
        if (!off || off->type != LZ_JSON_ARR || off->n_children != 2) {
            sterr(errbuf, errlen, LZ_ERR_ST_OFFSET, t->name);
            goto fail;
        }
        s0 = (long long)lz_json_at(&st->json, off, 0)->num;
        e0 = (long long)lz_json_at(&st->json, off, 1)->num;
        if (s0 < 0 || e0 < s0) {
            sterr(errbuf, errlen, LZ_ERR_ST_OFFSET_RANGE, t->name);
            goto fail;
        }
        t->off_begin = (unsigned long long)s0;
        t->nbytes = (unsigned long long)(e0 - s0);

        /* declared byte count must match shape×dtype, or later reads misalign silently */
        if (t->nbytes != (unsigned long long)t->n_elem *
                         (unsigned long long)lz_st_dtype_size(t->dtype)) {
            sterr(errbuf, errlen, LZ_ERR_ST_NBYTES,
                  t->name, t->nbytes,
                  t->n_elem * lz_st_dtype_size(t->dtype));
            goto fail;
        }
        if (st->data_start + (unsigned long long)e0 >
            (unsigned long long)st->file_size) {
            sterr(errbuf, errlen, LZ_ERR_ST_OVERFLOW, t->name);
            goto fail;
        }
        if ((unsigned long long)e0 > max_end) max_end = (unsigned long long)e0;

        st->n_tensors++;
    }

    if (st->n_tensors == 0) {
        sterr(errbuf, errlen, LZ_ERR_ST_NO_TENSORS);
        goto fail;
    }
    if (st->data_start + max_end != (unsigned long long)st->file_size) {
        sterr(errbuf, errlen, LZ_ERR_ST_DATA_SIZE,
              st->data_start, max_end, st->file_size);
        goto fail;
    }
    return 0;

fail:
    free(hdr);
    lz_st_close(st);
    return 1;
}

void lz_st_close(LZSafetensors *st) {
    if (!st) return;
    if (st->fp) fclose(st->fp);
    free(st->tensors);
    lz_json_free(&st->json);
    memset(st, 0, sizeof(*st));
}

const LZStTensor *lz_st_find(const LZSafetensors *st, const char *name) {
    int i;
    if (!st || !name) return NULL;
    for (i = 0; i < st->n_tensors; i++)
        if (strcmp(st->tensors[i].name, name) == 0) return &st->tensors[i];
    return NULL;
}

static int st_seek_read(LZSafetensors *st, const LZStTensor *t,
                        void *dst, unsigned long long nbytes,
                        char *errbuf, int errlen) {
    if (lz_fseek64(st->fp, st->data_start + t->off_begin) != 0) {
        sterr(errbuf, errlen, LZ_ERR_ST_SEEK,
              t->name, st->data_start + t->off_begin);
        return 1;
    }
    if (fread(dst, 1, (size_t)nbytes, st->fp) != (size_t)nbytes) {
        sterr(errbuf, errlen, LZ_ERR_ST_READ, t->name);
        return 1;
    }
    return 0;
}

int lz_st_read_raw(LZSafetensors *st, const LZStTensor *t,
                   void *dst, char *errbuf, int errlen) {
    if (!st || !t || !dst) {
        sterr(errbuf, errlen, LZ_ERR_ST_NULL2);
        return 1;
    }
    return st_seek_read(st, t, dst, t->nbytes, errbuf, errlen);
}

int lz_st_read_f32(LZSafetensors *st, const LZStTensor *t,
                   float *dst, long long n, char *errbuf, int errlen) {
    long long i;
    unsigned char *raw;

    if (!st || !t || !dst) {
        sterr(errbuf, errlen, LZ_ERR_ST_NULL);
        return 1;
    }
    if (n != t->n_elem) {
        sterr(errbuf, errlen, LZ_ERR_ST_ELEMS,
              t->name, n, t->n_elem);
        return 1;
    }

    if (t->dtype == LZ_DT_F32)
        return st_seek_read(st, t, dst, t->nbytes, errbuf, errlen);

    if (t->dtype != LZ_DT_BF16 && t->dtype != LZ_DT_F16) {
        sterr(errbuf, errlen, LZ_ERR_ST_DTYPE_F32,
              t->name, lz_st_dtype_name(t->dtype));
        return 1;
    }

    /* Read the 2-byte raw data into the second half of dst, then expand
       in place going forward. A 254M-element embedding then needs no
       extra 508MB scratch buffer.
       Safety: read position is 2n+2i bytes, write position 4i bytes; for
       i<n we always have 4i < 2n+2i, so writes never catch up with
       unread data. */
    raw = (unsigned char *)dst + (size_t)(n * 2);
    if (st_seek_read(st, t, raw, (unsigned long long)(n * 2), errbuf, errlen) != 0)
        return 1;

    if (t->dtype == LZ_DT_BF16) {
        /* bf16 is the high 16 bits of f32; a left shift suffices, no lookup table needed */
        for (i = 0; i < n; i++) {
            unsigned long bits = ((unsigned long)raw[i * 2 + 1] << 24) |
                                 ((unsigned long)raw[i * 2] << 16);
            unsigned int u = (unsigned int)bits;
            memcpy(&dst[i], &u, sizeof(float));
        }
    } else {
        for (i = 0; i < n; i++) {
            unsigned int h = ((unsigned int)raw[i * 2 + 1] << 8) |
                             (unsigned int)raw[i * 2];
            unsigned int sign = (h & 0x8000u) << 16;
            unsigned int exp  = (h >> 10) & 0x1Fu;
            unsigned int man  =  h & 0x3FFu;
            unsigned int u;
            if (exp == 0) {
                if (man == 0) {
                    u = sign;                       /* ±0 */
                } else {
                    /* subnormal: shift mantissa left until the top bit
                       overflows, adjusting the exponent in lockstep */
                    int shift = 0;
                    while (!(man & 0x400u)) { man <<= 1; shift++; }
                    man &= 0x3FFu;
                    u = sign | ((unsigned int)(127 - 15 - shift) << 23) | (man << 13);
                }
            } else if (exp == 0x1Fu) {
                u = sign | 0x7F800000u | (man << 13);   /* Inf / NaN */
            } else {
                u = sign | ((exp + 127u - 15u) << 23) | (man << 13);
            }
            memcpy(&dst[i], &u, sizeof(float));
        }
    }
    return 0;
}
