#!/bin/bash
# Single-file build helper for .prof/ numeric-law probes.
#
# These are throwaway differentials and microbenchmarks, not part of the
# `make` graph (new assembly is validated HERE,
# stand-alone, before it is trusted enough to land in src/ops.c). There is
# deliberately no Makefile target for them - see README.md "why no
# Makefile integration".
#
# Usage:
#   .prof/build.sh <probe-name> [gcc|watcom] [extra compiler flags...]
#
#   probe-name       file stem under .prof/, e.g. "wdump" for wdump.c
#   gcc|watcom       which compiler; default gcc (fast host loop).
#                    watcom needs Open Watcom installed at C:\WATCOM
#                    (the target compiler).
#
# Run from the repo root (E:\llama98-zh) or from .prof/ itself; the
# script finds both by locating its own directory.
#
# WHAT THIS DOES NOT DO: decide which extra src/*.c a probe needs. Some
# probes splice the engine inline (`#include "ops.c"` etc. - a single
# translation unit, nothing else to link) and some declare their own
# `extern` prototypes and need src/ops.c (and usually src/err.c,
# src/compat.c) compiled and linked alongside. Guessing which is a
# maintenance trap the moment a probe's includes change, so this script
# ESCALATES instead of guessing: try the probe alone, and if the linker
# reports undefined references, retry with progressively larger stock
# sets of engine sources until one links or the sets run out. The set
# that worked is printed, so a probe's own comment can be corrected to
# say so next time it is touched.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/src"

PROBE="${1:-}"
COMPILER="${2:-gcc}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_FLAGS="$*"

if [ -z "$PROBE" ]; then
    echo "usage: $0 <probe-name-without-.c> [gcc|watcom] [extra flags...]" >&2
    exit 2
fi

SRC_FILE="$HERE/$PROBE.c"
if [ ! -f "$SRC_FILE" ]; then
    echo "no such probe: $SRC_FILE" >&2
    exit 2
fi

# Escalating link sets, cheapest first. Paths are relative to $SRC.
# lfn.c (Win9x/DOS long-filename compat) is new since these probes were
# written upstream of the tree split - src/model.c calls into it now, so
# even single-TU splice probes (#include "model.c" etc. inline) need it
# added, not substituted; every "with lfn.c" variant below exists for
# that reason (migration note, found compiling wdump.c 2026-08-17).
# cpucheck.c is the other new-since-split dependency: ops.c's Watcom-only
# arm (#if defined(__WATCOMC__)) calls lz_cpuid1_edx(), defined there.
# gcc never sees those call sites (same guard), so gcc builds never need
# it - only the watcom sets below carry it (migration note, found
# building wbench.c with wcc386 2026-08-17).
SETS=(
    ""
    "lfn.c"
    "cpucheck.c"
    "cpucheck.c lfn.c"
    "ops.c ops_conv1d.c err.c compat.c"
    "ops.c ops_conv1d.c err.c compat.c lfn.c"
    "ops.c ops_conv1d.c err.c compat.c cpucheck.c"
    "ops.c ops_conv1d.c err.c compat.c cpucheck.c lfn.c"
    "ops.c ops_conv1d.c err.c compat.c json.c safetensors.c model.c"
    "ops.c ops_conv1d.c err.c compat.c json.c safetensors.c model.c lfn.c"
    "ops.c ops_conv1d.c err.c compat.c json.c safetensors.c model.c cpucheck.c"
    "ops.c ops_conv1d.c err.c compat.c json.c safetensors.c model.c cpucheck.c lfn.c"
    "ops.c ops_conv1d.c err.c compat.c json.c safetensors.c model.c forward.c"
    "ops.c ops_conv1d.c err.c compat.c json.c safetensors.c model.c forward.c lfn.c"
    "ops.c ops_conv1d.c err.c compat.c json.c safetensors.c model.c forward.c cpucheck.c"
    "ops.c ops_conv1d.c err.c compat.c json.c safetensors.c model.c forward.c cpucheck.c lfn.c"
    "ops.c ops_conv1d.c err.c compat.c gbk.c unicode.c"
    "ops.c ops_conv1d.c err.c compat.c gbk.c unicode.c cpucheck.c"
)

build_gcc() {
    local objs="$1"
    local out="$HERE/${PROBE}.exe"
    local files="$SRC_FILE"
    for o in $objs; do files="$files $SRC/$o"; done
    # -Isrc: the probes #include "ops.h" etc. by bare name, matching the
    # sibling-of-src/ layout this .prof/ was placed at (same as the
    # deprecated tree it was migrated from).
    # timeout 300: probes that splice ops.c inline (#include "ops.c") hand
    # gcc -O2 several thousand lines of one translation unit, including
    # every hand-unrolled matmul kernel - measured up to ~2 minutes for
    # wengine.c on this host. Not a hang; just slow at -O2.
    timeout 300 gcc -O2 -std=gnu99 -Wall -I"$SRC" -o "$out" $files -lm $EXTRA_FLAGS 2>"$HERE/.build_err.$PROBE.txt"
}

build_watcom() {
    local objs="$1"
    if [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
        WCC=/c/WATCOM/binnt64/wcc386.exe
        WLINK=/c/WATCOM/binnt64/wlink.exe
        LIBROOT='C:\WATCOM\lib386'
        WROOT=/c/WATCOM
    else
        echo "watcom-not-found (expected C:\\WATCOM)" >&2
        return 127
    fi
    local objdir="$HERE/.wobj"
    mkdir -p "$objdir"
    local flags="-za99 -bt=nt -otexan -zp4 -DLZ_USE_MMX=1 -D__MMX__=1 $EXTRA_FLAGS"
    local inc="-I$WROOT/h -I$WROOT/h/nt -I$SRC"
    local objlist="$PROBE.obj"
    # wlink is a native Windows tool - it cannot parse git-bash's POSIX
    # paths (/e/llama98-zh/...). cd into objdir and use bare filenames
    # for everything that ends up in the .lnk script; only the compiler
    # invocations (which DO accept POSIX paths, same as gcc) keep them.
    "$WCC" $flags $inc -fo="$objdir/$PROBE.obj" "$SRC_FILE" >"$HERE/.build_err.$PROBE.txt" 2>&1 || return 1
    for o in $objs; do
        local stem="${o%.c}"
        "$WCC" $flags $inc -fo="$objdir/$stem.obj" "$SRC/$o" >>"$HERE/.build_err.$PROBE.txt" 2>&1 || return 1
        objlist="$objlist $stem.obj"
    done
    (
        cd "$objdir" || exit 1
        {
            echo "format windows nt"
            echo "runtime console"
            echo "option quiet"
            echo "name ${PROBE}.exe"
            echo "libpath $LIBROOT/nt"
            echo "libpath $LIBROOT"
            for f in $objlist; do echo "file $f"; done
            echo "library clib3r, math387r, kernel32, user32"
        } > "$PROBE.lnk"
        "$WLINK" @"$PROBE.lnk"
    ) >>"$HERE/.build_err.$PROBE.txt" 2>&1 || return 1
    cp "$objdir/${PROBE}.exe" "$HERE/${PROBE}.exe"
}

for objs in "${SETS[@]}"; do
    if [ "$COMPILER" = "watcom" ]; then
        build_watcom "$objs" && { echo "OK (watcom, extra objects: [${objs:-none}]) -> $PROBE.exe"; rm -f "$HERE/.build_err.$PROBE.txt"; exit 0; }
    else
        build_gcc "$objs" && { echo "OK (gcc, extra objects: [${objs:-none}]) -> $PROBE.exe"; rm -f "$HERE/.build_err.$PROBE.txt"; exit 0; }
    fi
done

echo "FAILED to build $PROBE with any known object set. Last error:" >&2
cat "$HERE/.build_err.$PROBE.txt" >&2 2>/dev/null
exit 1
