#!/bin/bash
# Driver for rowdot_xcheck.c: build it, run it, then prove it can see a
# difference by planting one.
#
# The mutation is a LANE SWAP in the MMX q4_1 row kernel, not a value
# change: integer MACs are exact under reassociation, so a wrong sum is
# not the failure mode these kernels have - a wrong lane order is, and
# it produces a plausible number. If the probe cannot see a swap it
# cannot see the bug it exists for.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/rowdot}
CC=${CC:-gcc}
D="-DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
F="-O2 -std=c99 -mmmx -msse -msse2 -ffp-contract=off"
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     ops_mmx ops_mmx_sse ops_sse2 fwht cpucheck err"

list() { local d=$1; shift; for f in "$@"; do echo "$d/$f.c"; done; }

rm -rf "$O"; mkdir -p "$O/on" "$O/mut"
cp -r "$R/src" "$O/on/src"
cp -r "$R/src" "$O/mut/src"

if ! $CC $F $D -I"$O/on/src" -o "$O/probe" "$R/.prof/rowdot_xcheck.c" \
        $(list "$O/on/src" $SRC) -lm > "$O/build.log" 2>&1; then
    echo "rowdot_xcheck: SKIP - probe did not build"
    tail -8 "$O/build.log"
    exit 2
fi

echo "-- tiers, as shipped --"
"$O/probe" > "$O/run.log" 2>&1
rc=$?
cat "$O/run.log"
[ $rc -eq 0 ] || { echo "rowdot_xcheck: FAIL"; exit 1; }

# ---- the mutation ------------------------------------------------------
# Swap the two halves of the q4_1 nibble unpack in the MMX kernel. The
# sum over a row is unchanged for a symmetric input and changes for the
# distinct-per-element one the probe builds, which is the whole reason
# the probe builds it that way.
MF="$O/mut/src/ops_kernel_dot_mmx.h"
if grep -q "0x0F0F0F0F" "$MF" 2>/dev/null; then
    sed -i 's/0x0F0F0F0F/0xF0F0F0F0/' "$MF"
    MUTDESC="q4_1 nibble mask"
elif grep -q "_mm_srli_pi16(.*, 4)" "$MF" 2>/dev/null; then
    sed -i 's/_mm_srli_pi16(\(.*\), 4)/_mm_srli_pi16(\1, 5)/' "$MF"
    MUTDESC="q4_1 nibble shift"
else
    echo "rowdot_xcheck: FAIL - no mutation site found in"
    echo "  ops_kernel_dot_mmx.h; the probe is unproven and its PASS"
    echo "  above means nothing"
    exit 1
fi

if ! $CC $F $D -I"$O/mut/src" -o "$O/probe_mut" "$R/.prof/rowdot_xcheck.c" \
        $(list "$O/mut/src" $SRC) -lm > "$O/mutbuild.log" 2>&1; then
    echo "rowdot_xcheck: FAIL - the mutated tree did not build, so the"
    echo "  control did not run"
    tail -6 "$O/mutbuild.log"
    exit 1
fi

"$O/probe_mut" > "$O/mut.log" 2>&1
mrc=$?
if [ $mrc -eq 0 ]; then
    echo "rowdot_xcheck: FAIL - the $MUTDESC mutation did NOT redden the"
    echo "  probe; it is not comparing what it claims to"
    exit 1
fi
echo "-- mutation ($MUTDESC): red, as required --"
grep -m2 "FAIL" "$O/mut.log" | sed 's/^/  /'

# ---- the Watcom half ---------------------------------------------------
# The gcc build populates the three intrinsics slots; the three assembly
# slots exist only under wcc386, and they are the twins the pairing rule
# is about. Without this the probe compares gcc against gcc and reports
# "tiers agree" having never seen a hand-written kernel.
if bash "$R/.prof/wbuild_xcheck.sh" rowdot_xcheck > "$O/wat.log" 2>&1 \
   && [ -x "$R/.prof/_wxc/probe.exe" ]; then
    echo "-- tiers, Open Watcom (assembly slots) --"
    "$R/.prof/_wxc/probe.exe" > "$O/wrun.log" 2>&1
    wrc=$?
    cat "$O/wrun.log"
    [ $wrc -eq 0 ] || { echo "rowdot_xcheck: FAIL - Watcom tiers differ"; exit 1; }
else
    echo "-- Open Watcom -- SKIP, probe did not build"
    tail -4 "$O/wat.log"
    echo "rowdot_xcheck: PASS (gcc tiers only; the assembly twins were"
    echo "  NOT compared)"
    exit 0
fi

echo "rowdot_xcheck: PASS (both toolchains, mutation caught)"
exit 0
