#ifndef LZ4D_H
#define LZ4D_H
/* lz4d.h - LZ4 frame decompression as a streaming reader.
 *
 * Stunt project: the engine can load a model from `model.bin.lz4` (an
 * LZ4-frame-compressed model.bin) by streaming the frame into the same
 * per-field fread path. Pure byte access, no endianness assumptions, so
 * it runs identically on Watcom x86, gcc x86-64 and VC++ 4.0 MIPS
 * (big-endian): an LZ4 frame is a byte stream, and this reader never
 * casts bytes to a wider type to interpret them.
 *
 * Only the frame subset `lz4 -z`/lz4.frame produce by default is parsed:
 * version 01, block independence, no block checksum, optional content
 * size. A block with block-size == 0 is uncompressed and read verbatim.
 */

#include "lz_int.h"

typedef struct LZ4Stream LZ4Stream;

/* The first 4 bytes (little-endian) of every LZ4 frame. The loader
 * sniffs this to decide whether a weight file is compressed, regardless
 * of its extension. */
#define LZ4D_FRAME_MAGIC 0x184D2204u

/* Open `path` as an LZ4 frame. Returns NULL (errbuf set) if the file is
 * not a frame or the header is malformed. */
LZ4Stream *lz4d_open(const char *path, char *errbuf, int errlen);

/* Read up to `sz` bytes of decompressed data. Returns bytes read; 0 on
 * EOF. A stream error is sticky: after it, reads return 0. */
size_t lz4d_read(LZ4Stream *s, void *dst, size_t sz);

/* Non-zero once the frame is exhausted (or errored). */
int lz4d_eof(const LZ4Stream *s);

/* Non-zero if the stream hit a malformed frame. */
int lz4d_error(const LZ4Stream *s);

/* Close and free. */
void lz4d_close(LZ4Stream *s);

#endif
