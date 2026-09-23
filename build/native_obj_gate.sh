#!/bin/bash
# build/native/*.o must match the OBJECT FORMAT OF THE HOST LINKING THEM.
#
# One relative path, build/native/, serves both hosts. A MinGW build and
# a Linux/WSL build of this tree write their objects to the same place,
# and neither `ld` rejects the other's: MinGW's links ELF into a
# malformed PE ("section below image base", not executable) and GNU ld
# links COFF into "dangerous relocation: R_AMD64_IMAGEBASE with
# __ImageBase undefined". Both failures land far from the build that
# caused them. This runs right before the link step so a mismatched
# object is a build failure instead.
#
# BOTH DIRECTIONS, and the second one was missing. The gate used to
# assert COFF unconditionally - the format of the host it was written
# on - so on Linux it rejected the correct objects and passed the
# wrong ones. It surfaced when build/arm/parity_gate.sh needed an x86
# Linux binary and could not get one: the tree held MinGW objects, the
# gate demanded MinGW objects, and the link failed anyway.
#
# Checks file TYPE, not provenance: it does not ask which build ran, it
# opens whatever is sitting in build/native/*.o right now.
#
# Exit codes:
#   0  PASS - no build/native/*.o present, or all match this host
#   1  FAIL - at least one is the other host's format
set -u
cd "$(dirname "$0")/.."

LZ_ENV=$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="the native object-format check" bash build/tree_lock.sh acquire --print-env) || exit 2
eval "$LZ_ENV"
trap 'bash build/tree_lock.sh release' EXIT

DIR=build/native
shopt -s nullglob
objs=("$DIR"/*.o)
shopt -u nullglob

[ ${#objs[@]} -eq 0 ] && exit 0

# uname, not $OSTYPE: this script runs under Git Bash on Windows (where
# OSTYPE is msys) and under WSL and native Linux, and uname_s tells the
# three apart the same way in all of them.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) WANT=COFF; WANT_NAME="COFF (MinGW)" ;;
    *)                    WANT=ELF;  WANT_NAME="ELF" ;;
esac

bad=()
for f in "${objs[@]}"; do
    t=$(file -b "$f")
    case "$t" in
        *"$WANT"*) ;;
        *) bad+=("$f: $t") ;;
    esac
done

[ ${#bad[@]} -eq 0 ] && exit 0

echo "FAIL: $DIR holds objects this host cannot link - expected $WANT_NAME" >&2
echo "on $(uname -s), so the other host's build wrote here and its \`ld\`" >&2
echo "would either produce a malformed binary or fail with a relocation" >&2
echo "error far from the cause." >&2
for b in "${bad[@]}"; do echo "  $b" >&2; done
echo "Fix: rm $DIR/*.o && make (or make native/serve) to rebuild them." >&2
exit 1
