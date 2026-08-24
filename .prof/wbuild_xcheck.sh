#!/bin/bash
# Build one of the .prof/*_xcheck.c kernel probes with the TARGET
# compiler and run it on all three tiers.
#
#   bash .prof/wbuild_xcheck.sh fwht_f32_xcheck
#
# Iron law two says a SIMD kernel is written twice and the two must agree
# bit for bit. The *_xcheck.sh drivers prove the gcc half against the C
# body and mutate it; they cannot prove the Watcom half, because they
# build with $CC. This does the other half: same probe source, same three
# tiers, wcc386 + wlink, so the hash printed here is directly comparable
# to the one the gcc driver prints.
#
# It is a HAND-RUN check, not a CI stage: it needs C:\WATCOM and a
# Windows host to execute the .exe, and it has no mutation arm (the gcc
# driver already establishes that the kernel is reached at all - what is
# open here is only whether the hand-written asm computes the same thing
# as the intrinsics).
#
# Flags mirror the Makefile's CLI target exactly, including the split
# that gives $(ENG_MMX) and cpucheck -6r -D__MMX__=1 while everything
# else stays at the -4r floor. Compiling the kernel TUs at the ordinary
# floor would silently drop every #pragma aux body (they carry .686) -
# see the Makefile's own comment on why that loss is invisible.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
P=${1:-}
# NOT under /tmp. wlink does not read its response file through the shell,
# so MSYS never rewrites the paths inside it and a POSIX /tmp/... line
# reaches a native wlink verbatim - "cannot open" on every object while
# the compiler reported none. Repo-relative keeps both halves honest.
O=${O:-.prof/_wxc}
[ -n "$P" ] || { echo "usage: $0 <probe-stem>"; exit 2; }
[ -f "$R/.prof/$P.c" ] || { echo "no such probe: .prof/$P.c"; exit 2; }
cd "$R" || exit 2

# LIBNT is spelled out rather than built as "$LIBROOT/nt": on Windows
# LIBROOT is a BACKSLASH path, and appending a forward slash produced
# 'C:\WATCOM\lib386/nt', which wlink does not resolve. The nt directory
# then goes unsearched, clib3r falls through to the target-neutral copy
# one level up, and every libc symbol comes back undefined - including
# __I8RS, which reads like a compiler problem rather than a path one.
if [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
    W=/c/WATCOM/binnt64; WROOT=/c/WATCOM
    LIBROOT='C:\WATCOM\lib386'; LIBNT='C:\WATCOM\lib386\nt'; LIBSEP='\'
elif [ -x /mnt/c/WATCOM/binl64/wcc386 ]; then
    W=/mnt/c/WATCOM/binl64; WROOT=/mnt/c/WATCOM
    LIBROOT=/mnt/c/WATCOM/lib386; LIBNT=/mnt/c/WATCOM/lib386/nt; LIBSEP=/
else
    echo "$P: SKIP - no C:\\WATCOM"; exit 2
fi
WCC="$W/wcc386"; WLINK="$W/wlink"
[ -x "$WCC" ] || WCC="$WCC.exe"
[ -x "$WLINK" ] || WLINK="$WLINK.exe"

BASE="-za99 -otexan -zp4 -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -bt=nt -we
      -DWIN32_LEAN_AND_MEAN -DWINVER=0x0351 -D_WIN32_WINNT=0x0351"
INC="-I$WROOT/h -I$WROOT/h/nt -I$R/src -I$R/common"
LOW="-4r"
HIGH="-6r -D__MMX__=1"
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     fwht err"
SRC_HIGH="ops_mmx ops_mmx_sse ops_sse2 cpucheck"

rm -rf "$O"; mkdir -p "$O"
for f in $SRC; do
    "$WCC" $BASE $LOW $INC -fo="$O/$f.obj" "$R/src/$f.c" > "$O/$f.log" 2>&1 \
        || { echo "$P: FAIL - $f.c"; tail -8 "$O/$f.log"; exit 1; }
done
for f in $SRC_HIGH; do
    "$WCC" $BASE $HIGH $INC -fo="$O/$f.obj" "$R/src/$f.c" > "$O/$f.log" 2>&1 \
        || { echo "$P: FAIL - $f.c"; tail -8 "$O/$f.log"; exit 1; }
done
"$WCC" $BASE $LOW $INC -fo="$O/probe.obj" "$R/.prof/$P.c" > "$O/probe.log" 2>&1 \
    || { echo "$P: FAIL - probe"; tail -8 "$O/probe.log"; exit 1; }

{
    echo "format windows nt"
    echo "runtime console"
    echo "option quiet"
    echo "name $O/probe.exe"
    # LIBSEP, not a literal '/': on Windows LIBROOT is a backslash path
    # ('C:\WATCOM\lib386') and appending "/nt" gave wlink
    # 'C:\WATCOM\lib386/nt', which it does not resolve. The nt directory
    # then never gets searched, clib3r falls through to the target-neutral
    # copy one level up, and every libc symbol comes back undefined -
    # including __I8RS, which reads like a compiler problem rather than a
    # path one.
    echo "libpath $LIBNT"
    echo "libpath $LIBROOT"
    for f in $SRC $SRC_HIGH; do echo "file $O/$f.obj"; done
    echo "file $O/probe.obj"
    # clib3r by FULL PATH, the way the Makefile's own link line does it.
    # By bare name it is resolved against libpath, and the copy that wins
    # there is the target-neutral one a level above nt/ - which does not
    # carry the NT C runtime, so every libc symbol comes back undefined.
    echo "library $LIBNT${LIBSEP}clib3r.lib, math387r, kernel32, user32, advapi32"
} > "$O/probe.lnk"
"$WLINK" @"$O/probe.lnk" > "$O/link.log" 2>&1 \
    || { echo "$P: FAIL - link"; tail -12 "$O/link.log"; exit 1; }

for T in ref mmx sse2; do
    printf '  watcom %-5s ' "$T"
    "$O/probe.exe" "$T" | tail -1
done
exit 0
