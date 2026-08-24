#!/bin/bash
# Build and run the epi MAC oracle against the SHIPPED Watcom SSE2
# kernel - the #pragma aux twin of the intrinsics body, both in
# src/ops_sse2.c.
#
#   bash .prof/epi_watcom_gate.sh
#
# The gcc side of the same predicate is one gcc line (see the oracle's
# header); this one needs a linker response file, so it is a script.
#
# WHY THE THREE EXTRA TUs. ops_sse2.c reaches lz_pair_mode,
# lz_prefetch_mode, lz_cpuid1_edx and lz_err_fmt from functions this
# harness never calls. They are linked rather than stubbed: a stub is a
# second definition of engine state, and it would be the stub under
# test rather than the engine.
#
# Every path below is Windows-shaped because wlink reads the response
# file itself, unrolled through printf '%s\n' with one argument per
# line - a format string carrying the backslashes would have \n inside
# lib386\nt eaten as a newline, which is how the first version of this
# produced a response file with a bare "t" on line 6.
set -u
W=${WATCOM_ROOT:-C:/WATCOM}
WCC=${WCC:-/c/WATCOM/binnt64/wcc386.exe}
WLINK=${WLINK:-/c/WATCOM/binnt64/wlink.exe}
O=${O:-/tmp/epiwat}
OW=${OW:-C:\\Users\\Dick\\AppData\\Local\\Temp\\epiwat}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

if [ ! -x "$WCC" ]; then
    echo "epi_watcom_gate: SKIP - no $WCC"
    exit 2
fi

mkdir -p "$O"
rm -f "$O"/*.obj "$O"/epiw.exe

# -6r matches the CLI target (Makefile CLI_FLAGS); the pragma raises to
# .686 on its own, which is what lets the SSE2 mnemonics assemble.
FLAGS="-6r -oxi -zq -bt=nt -I$ROOT/src"
export INCLUDE='C:\WATCOM\h;C:\WATCOM\h\nt'

"$WCC" $FLAGS -DEPI_LINK_SHIPPED -DLZ_SSE2_TU=1 \
    "$ROOT/.prof/epi_mac_oracle.c" -fo="$O/epio.obj" || exit 1
"$WCC" $FLAGS -DLZ_SSE2_TU=1 "$ROOT/src/ops_sse2.c" \
    -fo="$O/ops_sse2.obj" || exit 1
for f in ops_sched cpucheck err; do
    "$WCC" $FLAGS "$ROOT/src/$f.c" -fo="$O/$f.obj" || exit 1
done

printf '%s\n' \
    "format windows nt" \
    "runtime console" \
    "option quiet" \
    "name $OW\\epiw.exe" \
    "libpath $W/lib386/nt" \
    "libpath $W/lib386" \
    "file $OW\\epio.obj" \
    "file $OW\\ops_sse2.obj" \
    "file $OW\\ops_sched.obj" \
    "file $OW\\cpucheck.obj" \
    "file $OW\\err.obj" \
    "library $W/lib386/nt/clib3r.lib, math387r, kernel32, user32, advapi32" \
    > "$O/epiw.lnk"

# The response file is the thing most likely to be silently wrong, so
# it is echoed before it is used rather than after a failure.
echo "-- response file --"
sed 's/^/   /' "$O/epiw.lnk"

"$WLINK" @"$OW\\epiw.lnk" || exit 1
[ -x "$O/epiw.exe" ] || { echo "epi_watcom_gate: FAIL - no exe"; exit 1; }
exec "$O/epiw.exe"
