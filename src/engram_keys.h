#ifndef LZ_ENGRAM_KEYS_H
#define LZ_ENGRAM_KEYS_H
/* engram_keys.h - the static 2-gram key table's loader and lookup.
 *
 * Spec section 1.6: a model may replace the engram's rolling
 * multi-head hash with a STATIC table of selected 2-grams. The table is
 * a .npy of sorted, strictly increasing int64 codes beside the
 * checkpoint (the sidecar `engram_keys` in config.json), one row per
 * key; a code is (id0 << shift) | id1 with shift wide enough for the
 * vocabulary. A key that is not in the table contributes nothing - it
 * does not fall into a hash slot - which is the property the table
 * exists for.
 *
 * The file is written by tools/engram_keys.py / engine/engram_keys.py
 * and read by both the exporter and this engine; section 1.6 makes the
 * sidecar the one copy, so neither may derive a table of its own.
 *
 * No floating point and no endianness assumption: the payload is read
 * byte by byte into an lz_i64, so a big-endian target (VC++ 4.0 MIPS)
 * gets the same codes as the little-endian one.
 */

#include "lz_int.h"

/* Load the table at `path`. On success *out is a malloc'd array of *out_n
 * strictly increasing codes and the caller owns it. On failure errbuf is
 * set (LZ_ERR_ENGRAM_KEYS_*) and *out is NULL.
 *
 * Only the shape numpy writes for a 1-D '<i8' array is accepted:
 * fortran_order False, shape (N,), version 1.0/2.0/3.0 header. Anything
 * else is refused rather than reinterpreted - a table read at the wrong
 * stride rows into a different n-gram for every position and no
 * downstream check can see it. */
int lz_keys_load(const char *path, lz_i64 **out, int *out_n,
                 char *errbuf, int errlen);

/* Row of `code` in the sorted table, or -1 when it is absent. */
int lz_keys_find(const lz_i64 *keys, int n, lz_i64 code);

#endif
