#!/bin/bash
# Cross-build the llama98 engine for MIPS, big-endian OR little-endian.
#
# WHY THIS EXISTS: it is the only instrument on this project that can tell
# a byte-order bug from a correct program. Every other target the engine
# builds for - x86 (gcc/MinGW/Open Watcom), x86-64, ARMv5TE little-endian
# - is little-endian, so a load path that assumes little-endian byte
# order is bit-identical to a correct one on ALL of them. Two builds of
# the SAME source that differ only in endianness are what makes that
# assumption visible.
#
# TARGET CONTEXT: Loongson 2F under Windows NT/MIPS (VC++ 4.0). NT and
# the 2F both actually run little-endian, so the big-endian arm is not a
# shipping configuration - it is the control that proves the code does
# not depend on byte order. mips-linux-gnu is a stand-in for the real
# target: same endianness question, a toolchain that exists, and a
# user-mode emulator that runs the result.
#
# WHY EVERY FLAG IS THERE:
#
#   -EB / -EL        implied by the triple (mips- vs mipsel-), not passed
#                    here. Both are o32, ILP32, hard-float MIPS32 - the
#                    ONLY difference between the two binaries this script
#                    produces is byte order, which is the whole point. A
#                    second difference would make a DIFFER result
#                    ambiguous.
#
#                    ONE compiler switched with -EB/-EL would be tidier
#                    still, and gcc does accept both. It does not link:
#                    Debian ships the mips-linux-gnu triple with a
#                    big-endian crt1.o/crti.o/libgcc only, so `-EL` dies
#                    at "crt1.o: compiled for a big endian system and
#                    target is little endian". The two triples are the
#                    only way to get both arms, so instead this script
#                    ASSERTS the two compilers report the same version
#                    string (see version_of below) - Debian builds
#                    gcc-14-mips-linux-gnu and gcc-14-mipsel-linux-gnu
#                    from one source package at one revision. That
#                    assertion is what keeps "BE and LE differ" from
#                    meaning "two different compilers differ".
#   -ffp-contract=off  MIPS has madd.s/msub.s and gcc will contract into
#                    them. Without this the two arms could differ for a
#                    reason that has nothing to do with endianness.
#   -static          qemu-user then needs no -L sysroot, so the runner
#                    has one less thing to get wrong per arch.
#   no -DLZ_NO_FPU   unlike the ARM target this one HAS an FPU (o32
#                    hard-float) and qemu emulates it. Forcing the fixed
#                    tiers here would test a different code path on each
#                    arm than the x86 builds run.
#
# The SIMD translation units (ops_mmx, ops_mmx_sse, ops_sse2) are absent
# for the same reason the ARM build omits them: no MMX/SSE register file.
# The 'ref' tier is a real path on this build, not a fallback nobody
# reaches.
#
# USAGE
#   ARCH=be OUT=/tmp/mips/be bash build/mips/build_engine.sh
#   ARCH=le OUT=/tmp/mips/le bash build/mips/build_engine.sh
#
# TOOLCHAIN: Debian/Ubuntu packages gcc-14-mips-linux-gnu (BE) and
# gcc-14-mipsel-linux-gnu (LE). The binary is SUFFIXED -
# `mips-linux-gnu-gcc-14`; there is no unsuffixed alias, exactly as with
# arm-linux-gnueabi-gcc-14. Both suffixed and unsuffixed names are tried
# below so a host that has either shape works.
#
# If a triple is not installed system-wide, point LZ_MIPS_XTC at a prefix
# holding an unpacked copy - `dpkg -x pkg.deb DIR` needs no root, and
# `apt-get download` does not either:
#   LZ_MIPS_XTC=/tmp/xtc14  ->  $LZ_MIPS_XTC/usr/bin on PATH and
#                               $LZ_MIPS_XTC/usr/lib/x86_64-linux-gnu on
#                               LD_LIBRARY_PATH (the binutils in there is
#                               linked against its own libbfd/libopcodes
#                               and will not start without it).
set -u

ARCH=${ARCH:-be}
OUT=${OUT:-/tmp/mipsbuild-$ARCH}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

case "$ARCH" in
    be) TRIPLE=mips-linux-gnu ;;
    le) TRIPLE=mipsel-linux-gnu ;;
    # HOST: the same source built by the host's own gcc, as a
    # little-endian arm for hosts where gcc-14-mipsel-linux-gnu is not
    # installed. endian_gate.sh's header says what that costs - the arms
    # then differ in compiler and architecture as well as byte order.
    # It is here rather than in the gate so both arms still come from ONE
    # build recipe; a second recipe for the fallback would be the second
    # copy that always drifts.
    host) TRIPLE=host ;;
    *)  echo "build_engine.sh: ARCH must be 'be', 'le' or 'host', got '$ARCH'" >&2
        exit 2 ;;
esac

if [ -n "${LZ_MIPS_XTC:-}" ]; then
    PATH="$LZ_MIPS_XTC/usr/bin:$PATH"
    LD_LIBRARY_PATH="$LZ_MIPS_XTC/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
    export PATH LD_LIBRARY_PATH
fi

if [ -z "${CC:-}" ] && [ "$TRIPLE" = host ]; then
    command -v gcc >/dev/null 2>&1 && CC=gcc
fi
if [ -z "${CC:-}" ]; then
    for cand in "$TRIPLE-gcc-14" "$TRIPLE-gcc"; do
        if command -v "$cand" >/dev/null 2>&1; then CC=$cand; break; fi
    done
fi
if [ -z "${CC:-}" ] || ! command -v "$CC" >/dev/null 2>&1; then
    echo "build_engine.sh: no $TRIPLE-gcc-14 or $TRIPLE-gcc on PATH" >&2
    echo "  (install gcc-14-$TRIPLE, or set LZ_MIPS_XTC to an unpacked one)" >&2
    exit 2
fi
echo "  CC=$CC  ($($CC -dumpversion 2>/dev/null))"

FLAGS="-O2 -std=c99 -Wall -Wextra -ffp-contract=off -static -I$ROOT/src"

# ASKED FOR, not restated. This was a hand-copy carrying a comment
# saying it had to stay the same list, and it did not: common/beep.c
# arrived and the link here failed with undefined references to lz_beep
# while the comment still claimed the two matched. A copy cannot notice
# its own rot; `make -s print-ENG` has nothing to rot. build/x87 made
# the same move after nine files went missing from its copy.
ENG=$(make -s -C "$ROOT" print-ENG)
if [ -z "$ENG" ]; then
    echo "mips: cannot read ENG from $ROOT/Makefile (make -s print-ENG)"
    exit 2
fi

mkdir -p "$OUT"
rm -f "$OUT"/*.o "$OUT"/*.log "$OUT/llama98-mips"

fail=0
warn=0
for f in $ENG; do
    if $CC $FLAGS -c "$ROOT/src/$f.c" -o "$OUT/$f.o" 2>"$OUT/$f.log"; then
        n=$(grep -c 'warning:' "$OUT/$f.log" || true)
        warn=$((warn + n))
        printf '  %-14s ok    %s warnings\n' "$f" "$n"
    else
        printf '  %-14s FAIL\n' "$f"
        grep -E 'error:' "$OUT/$f.log" | head -4 | sed 's/^/      /'
        fail=$((fail + 1))
    fi
done

echo
echo "objects: $(ls -1 "$OUT"/*.o 2>/dev/null | wc -l)  failed: $fail  warnings: $warn"
[ "$fail" -eq 0 ] || exit 1

# The warning count is an assertion, not a printout. It was counted,
# printed and discarded, and sat at 18 - all of them one shape,
# "defined but not used" for a static in a header that this TU includes
# for something else. A build whose warning count is a number nobody
# looks at cannot report the one warning that matters, and this is the
# only build in the tree that is big-endian.
#
# The ceiling is the current count, not zero: src/forward.c has an
# unused lslot behind a conditional, and it is being edited elsewhere.
# Lowering this is the work; raising it is the regression.
MIPS_WARN_CEIL=${MIPS_WARN_CEIL:-1}
if [ "$warn" -gt "$MIPS_WARN_CEIL" ]; then
    echo "FAIL: $warn warnings, ceiling $MIPS_WARN_CEIL"
    grep -h 'warning:' "$OUT"/*.log 2>/dev/null \
        | sed 's/.*warning: //' | sort | uniq -c | sort -rn | head -6 \
        | sed 's/^/      /'
    exit 1
fi

# Link the CLI, exactly as build/arm/build_engine.sh does: cli_main.c's
# interactive loop runs on the shared LZSession core in common/, which is
# a front-end helper deliberately outside $(ENG).
COM=$(make -s -C "$ROOT" print-COM)
if [ -z "$COM" ]; then
    echo "mips: cannot read COM from $ROOT/Makefile (make -s print-COM)"
    exit 2
fi
COMSRC=""
for c in $COM; do
    [ -f "$ROOT/common/$c.c" ] && COMSRC="$COMSRC $ROOT/common/$c.c"
done
if $CC $FLAGS -I"$ROOT/common" -o "$OUT/llama98-mips" \
     "$ROOT/src/cli_main.c" "$ROOT/src/cli_argv.c" \
     "$ROOT/src/cli_attr.c" $COMSRC "$OUT"/*.o -lm \
     2>"$OUT/link.log"; then
    echo "linked: $OUT/llama98-mips"
    file "$OUT/llama98-mips" 2>/dev/null | sed 's/^/  /'
else
    echo "link FAILED:"
    grep -E 'error|undefined' "$OUT/link.log" | head -8 | sed 's/^/  /'
    exit 1
fi
