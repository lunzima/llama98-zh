# llama98

KunMoe LLM inference engine, and the 昆昆98 front end that wraps it. A C99
engine for the KunMoe architecture — KDA linear attention plus a latent
mixture of experts — built for a family of Pentium-class machines running
Windows 9x and DOS.

One Makefile drives two toolchains:

- **gcc** (the dev machine) — `native`, `gui`, `serve`
- **Open Watcom** (the target) — `watcom-cli`, `watcom-gui`, `watcom-dos`,
  `watcom-dll`

## Products

| Target | Output | What it is |
|---|---|---|
| `native` | `llama98` | engine CLI (dev machine) |
| `gui` | `kunkun98-gui` | 昆昆98 Win32 front end (dev machine) |
| `serve` | `kunkun98-serve` | OpenAI-compatible HTTP server |
| `watcom-cli` | `llama98-w.exe` | engine CLI (Windows 9x) |
| `watcom-gui` | `kunkun98-w.exe` | 昆昆98 front end (Windows 9x) |
| `watcom-dll` | `llama98.dll` | engine DLL, `llama98.def` is the export surface |
| `watcom-dos` | `llama98-dos.exe` | engine CLI for a 32-bit DOS extender |

`make check` builds the three Watcom Windows products. The build is the
verification: a clean link proves the DLL export surface matches the
engine, and `-D__MMX__=1` is baked in so the hand-written assembly cannot
silently vanish.

## Building

```sh
make native gui serve        # gcc, dev machine
make check                   # Watcom: cli + gui + dll
make watcom-dos              # DOS target (needs the extender stub)
```

The DOS build picks its extender via `LZ_DOS_EXTENDER`:

```sh
make watcom-dos LZ_DOS_EXTENDER=dos32a   # DOS/32A (default is dos4gw)
```

Both toolchains read the same source lists, defined once at the top of the
Makefile (`ENG`, `SRV`, `GUI`, `COM`). A source added to the engine is
added in exactly one place.

## Layout

```
Makefile          single build driver (gcc + Watcom)
llama98.def       DLL export surface, single source of truth
src/              the engine
gui/              the 昆昆98 front end
common/           shared front-end helpers (LZSession core, settings, ...)
assets/           finished images (icon, splash, lamps) referenced by gui/kunkun98.rc
contrib/dos32a/   customized DOS/32A extender + its source + license
build/watcom/     gen_exports.py (DLL build-time dependency)
```

## Requirements

- **gcc** (MinGW on the dev machine) for `native`/`gui`/`serve`
- **Open Watcom** (2.x) at `C:\WATCOM` for the `watcom-*` targets — the
  compiler, linker, and `wrc`. The DOS and Windows targets are cross-built
  from `binnt64`.
- **DOSBox-X** to run the DOS build, and to run the optional single-file
  packaging step (`make watcom-dos-single`, which documents the `SB.EXE`
  bind).

## Third-party acknowledgments

This product uses DOS/32 Advanced DOS Extender technology.

DOS/32A — Copyright (C) 1996-2006 by Narech K.

The customized extender in `contrib/dos32a/` suppresses the startup banner;
the acknowledgment above is printed by `llama98-dos --help` instead, which
satisfies condition 3 of the DOS/32A license (BSD-style, reproduced in
`contrib/dos32a/license`). See `contrib/dos32a/README.md` for what was
changed and why the shipped binary is patched rather than recompiled.
