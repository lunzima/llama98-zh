#!/bin/bash
# Does Visual C++ 4.0 compile the no-assembly source set?
#
# c89_floor_gate compiles with gcc -std=c89 -pedantic and vc4_limits_gate
# counts translation limits, and both carry a header saying green does
# not mean VC++ 4.0 accepts the result. It did not, four times, for
# things neither could see:
#
#   int64_t     not a type there - lz_int.h typedefs int8_t through
#               uint32_t for _MSC_VER < 1600 and had missed this one
#   ULL         not a suffix before 2003; it is spelled i64, and ULL is
#               rejected outright, taking the expression with it
#   raw UTF-8   in a string literal the lexer reads the lead bytes as
#               DBCS, eats the following byte, and swallows the closing
#               quote - 138 of those in err.c
#   dirent.h    POSIX, absent; the walk in lfn.c now has an _findfirst
#               implementation beside the dirent one
#
# All four are fixed. This is what keeps them fixed, because the check
# that found them was a person typing a command, and a PASS someone
# typed looks exactly like a PASS a build produced.
#
# NO ASSEMBLY. ops_mmx, ops_mmx_sse, ops_sse2, ops_arm and ops_t2_arm
# are excluded: they are intrinsics and inline assembly for instruction
# sets this compiler's target does not have, and the owner's requirement
# is that the no-assembly build compiles.
#
# WHAT THIS STILL DOES NOT PROVE. This is the x86 edition. The MIPS
# edition shares the front end - both are compiler 10.00.x - so the
# language and library questions carry over, and the ones that do not
# are code generation and byte order, which have their own gates
# (kernel_isa_identity_gate, build/mips/endian_gate.sh). It also does
# not link, and does not run.
#
# Exit codes:
#   0  PASS
#   1  FAIL - a translation unit did not compile
#   2  SKIP - no VC++ 4.0 at the expected path
set -u
cd "$(dirname "$0")/.."

CL=${LZ_VC4_CL:-/c/MSDEVSTD/BIN/CL.EXE}
INC=${LZ_VC4_INC:-'C:\MSDEVSTD\INCLUDE'}
[ -x "$CL" ] || { echo "vc4_compile_gate: SKIP - no VC++ 4.0 at $CL"; exit 2; }

SKIPPED="ops_mmx ops_mmx_sse ops_sse2 ops_arm ops_t2_arm"

FAIL=0
N=0
CLEAN=0
for f in src/*.c; do
    b=$(basename "$f" .c)
    case " $SKIPPED " in *" $b "*) continue ;; esac
    N=$((N + 1))
    OUT=$(INCLUDE="$INC" "$CL" //c //nologo //Isrc //Icommon //Fo:NUL "$f" 2>&1)
    E=$(printf '%s\n' "$OUT" | grep -c "error C")
    if [ "$E" -gt 0 ]; then
        echo "FAIL: $b - $E error(s)"
        printf '%s\n' "$OUT" | grep -m3 "error C" | sed 's/^/      /'
        FAIL=1
    else
        CLEAN=$((CLEAN + 1))
    fi
done

# Positive control. A CL that refused to run at all, or a loop whose
# file list came back empty, produces zero errors and reads as success.
# Compiling something this compiler MUST reject proves the error channel
# works; a bare `long long` is the cheapest one, and it is also the
# defect class that was hardest to see coming.
CTL=$(printf 'long long x;\n' > /tmp/vc4ctl.$$.c
      INCLUDE="$INC" "$CL" //c //nologo //Fo:NUL /tmp/vc4ctl.$$.c 2>&1 \
      | grep -c "error C")
rm -f /tmp/vc4ctl.$$.c
if [ "${CTL:-0}" -eq 0 ]; then
    echo "vc4_compile_gate: FAIL - the compiler accepted \`long long\`, which"
    echo "  VC++ 4.0 cannot; either it is not VC++ 4.0 or the error channel"
    echo "  is not being read, and every PASS above means nothing"
    exit 1
fi

if [ "$FAIL" -eq 0 ]; then
    echo "vc4_compile_gate: PASS ($CLEAN of $N no-assembly unit(s))"
    exit 0
fi
echo "vc4_compile_gate: FAIL ($CLEAN of $N clean)"
exit 1
