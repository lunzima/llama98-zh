#!/bin/bash
# build/native/*.o must be COFF (MinGW gcc's native output), never ELF.
#
# A Linux/WSL gcc invocation that writes into this same relative path
# (build/native/) leaves ELF object files behind. MinGW's `ld` links
# them into the PE without a link error - the binary comes out
# malformed ("section below image base", not executable) and the
# failure is only visible when someone tries to run it, far from the
# build that produced it. This runs right before the link step, so a
# stray ELF object is a build failure, not a silent bad binary.
#
# Checks file TYPE, not provenance: it does not ask whether a Linux
# build ran, it opens whatever is sitting in build/native/*.o right now.
#
# Exit codes:
#   0  PASS - no build/native/*.o present, or all present ones are COFF
#   1  FAIL - at least one is not COFF (ELF, or anything else)
set -u
cd "$(dirname "$0")/.."

DIR=build/native
shopt -s nullglob
objs=("$DIR"/*.o)
shopt -u nullglob

[ ${#objs[@]} -eq 0 ] && exit 0

bad=()
for f in "${objs[@]}"; do
    t=$(file -b "$f")
    case "$t" in
        *COFF*) ;;
        *) bad+=("$f: $t") ;;
    esac
done

[ ${#bad[@]} -eq 0 ] && exit 0

echo "FAIL: $DIR has non-COFF object file(s) - a Linux/WSL build wrote" >&2
echo "here, and MinGW's ld would link these silently into a malformed," >&2
echo "non-executable PE." >&2
for b in "${bad[@]}"; do echo "  $b" >&2; done
echo "Fix: rm $DIR/*.o && make (or make native/serve) to rebuild them." >&2
exit 1
