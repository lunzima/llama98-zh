#ifndef LZ_SAFETENSORS_H
#define LZ_SAFETENSORS_H

#include "lz_int.h"   /* lz_i64/lz_u64: the 64-bit type, portably */
#include <stdio.h>

#include "json.h"

/* safetensors reader (M4.5).
   Layout: [u64 LE header length N][N-byte JSON header][raw tensor data]
   Header looks like {"tensor_name":{"dtype":"BF16","shape":[..],"data_offsets":[s,e]}, ...}
   plus an optional "__metadata__" member, which is not a tensor.

   This module does not slurp the whole file: an unpruned model is 1.7GB,
   of which ~200MB is an unused vision tower. We keep the file handle and
   read individual tensors on demand. */

#define LZ_ST_MAX_DIMS 8

enum {
    LZ_DT_UNKNOWN = 0,
    LZ_DT_BF16,
    LZ_DT_F16,
    LZ_DT_F32,
    LZ_DT_F64,
    LZ_DT_I8,
    LZ_DT_U8,
    LZ_DT_I16,
    LZ_DT_I32,
    LZ_DT_I64,
    LZ_DT_BOOL
};

typedef struct {
    const char *name;               /* points into LZSafetensors::json's buffer */
    int dtype;
    int n_dims;
    lz_i64 shape[LZ_ST_MAX_DIMS];
    lz_i64 n_elem;
    lz_u64 off_begin;   /* offset relative to data area start */
    lz_u64 nbytes;
} LZStTensor;

typedef struct {
    FILE *fp;
    LZJson json;                    /* must outlive: tensor names point into its buffer */
    LZStTensor *tensors;
    int n_tensors;
    lz_u64 data_start;  /* 8 + header length */
    lz_i64 file_size;
} LZSafetensors;

/* return 0 on success, non-zero on failure and fill errbuf */
int  lz_st_open(LZSafetensors *st, const char *path, char *errbuf, int errlen);
void lz_st_close(LZSafetensors *st);

/* find by name; NULL if absent */
const LZStTensor *lz_st_find(const LZSafetensors *st, const char *name);

/* read the tensor and convert to f32. n must equal t->n_elem; dst must hold n floats.
   Supports BF16 / F16 / F32; other dtypes return an error. */
int lz_st_read_f32(LZSafetensors *st, const LZStTensor *t,
                   float *dst, lz_i64 n, char *errbuf, int errlen);

/* read t->nbytes bytes verbatim into dst */
int lz_st_read_raw(LZSafetensors *st, const LZStTensor *t,
                   void *dst, char *errbuf, int errlen);

const char *lz_st_dtype_name(int dt);
int lz_st_dtype_size(int dt);

#endif
