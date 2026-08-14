# DOS/32A — customized for llama98-zh

This directory does not vendor the DOS/32A source tree. It points at the
upstream release and records the one change llama98 makes to it.

## Upstream

- Project: DOS/32 Advanced DOS Extender, by Narech K.
- Source: https://github.com/amindlost/dos32a
- Pinned commit: `06d8e3a0b2397d872205d6c99a7de0d6b8628bcc` (v9.12 archive)
- License: BSD-style, 4 conditions — see `license` in this directory.

To obtain the source this patch applies to:

```
git clone https://github.com/amindlost/dos32a
cd dos32a
git checkout 06d8e3a0b2397d872205d6c99a7de0d6b8628bcc
git apply /path/to/llama98.patch
```

## The change

`llama98.patch` clears bit 3 of the extender's `_ID32` config byte
("show copyright") in `src/dos32a/dos32a.asm`. That makes the extender
start without printing its copyright banner; the license acknowledgment
moves into llama98's own `--help` output, which license condition 3
permits ("in the software itself").

## The shipped binary

`dos32a-custom.exe` is the official DOS/32A v9.12 binary (byte-for-byte
identical to the one Open Watcom 2.x ships in `binw/`) with that same
config bit patched: offset `0x75` changed `0x09` → `0x01`. A from-source
rebuild is not used because TASM 5.0's OMF object records do not round-trip
through Open Watcom's `wlink` (the entry-point segment is lost), so the
patched official binary is the source of truth for the shipped artifact.
