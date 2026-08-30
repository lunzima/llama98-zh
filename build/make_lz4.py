#!/usr/bin/env python3
"""make_lz4.py - compress a model.bin into a standard LZ4 frame the
engine can stream (stunt project).

The engine's lz_open_bin prefers `model.bin.lz4` over `model.bin` when
both exist, and serves the decompressed bytes through the same reader
path, so this is a drop-in: place the produced file next to config.json
and tokenizer.json, and the engine reads it as if it were model.bin.

Frame layout (standard LZ4 frame, the subset src/lz4d.c accepts):
  magic 04 22 4d 18
  FLG   0x68  (version 01 | block independence | content size)
  BD    0x40  (block max size code 4 = 64 KiB)
  content size (8 bytes LE)
  header checksum (xxh32(FLG+BD+content) >> 8 & 0xFF)
  blocks of <= 64 KiB, each: 4-byte LE header whose bit 31 marks an
    UNCOMPRESSED block (raw data) and bits 0-30 the block size; the
    frame ends when the content size is reached (no last-block flag).

Block size is MCU-sized on purpose: the engine's decompression buffer is
64 KiB, so streaming a model costs ~64 KiB of extra memory regardless of
model size. A block the compressor produced larger than 64 KiB is a hard
error at load time, so this tool never emits one.

Usage:
  python make_lz4.py model.bin [model.bin.lz4]
  python make_lz4.py <model-dir>      # compress <dir>/model.bin in place
"""
import struct
import sys

import lz4.block
import xxhash

BLOCK = 64 * 1024
MAGIC = bytes([0x04, 0x22, 0x4D, 0x18])


def compress(data, block_size=BLOCK):
    out = bytearray(MAGIC)
    out += bytes([0x68, 0x40])                       # FLG, BD (64 KiB)
    out += struct.pack("<Q", len(data))
    out += bytes([(xxhash.xxh32(bytes(out[4:])).intdigest() >> 8) & 0xFF])
    pos = 0
    while pos < len(data):
        chunk = data[pos:pos + block_size]
        c = lz4.block.compress(chunk, store_size=False)
        if len(c) < len(chunk):
            out += struct.pack("<I", len(c)) + c
        else:
            out += struct.pack("<I", len(chunk) | 0x80000000) + chunk
        pos += block_size
    return bytes(out)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: make_lz4.py model.bin [model.bin.lz4]  |  make_lz4.py <dir>")
    src, dst = sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None
    if dst is None:
        dst = src + ".lz4"                       # model.bin -> model.bin.lz4
    with open(src, "rb") as f:
        data = f.read()
    out = compress(data)
    with open(dst, "wb") as f:
        f.write(out)
    print(f"{len(data)/1e6:.1f}MB -> {len(out)/1e6:.1f}MB "
          f"({len(out)/len(data):.3f}), {len(data)//BLOCK+1} x {BLOCK//1024}KiB "
          f"blocks -> {dst}")


if __name__ == "__main__":
    main()
