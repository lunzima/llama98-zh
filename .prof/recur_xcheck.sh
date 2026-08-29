#!/bin/bash
# Per-operator cross-check for the GDN recurrence arms that
# kda_gt1_xcheck does not cover.
#
# kda_gt1_xcheck mutates the gt==1.0f epilogue fast arm. The recurrence's
# other arms - the pass-1 tap (15.4% of LZ_FC_RECUR), the p2 group scale
# (23.1%), and the pass-2 delta quantize (part of the 38.4% pass-2 entry)
# - have no mutation that proves they run. Each can be silently wrong
# while kda_gt1 stays green.
#
# Each mutation below flips ONE computation. The probe drives the same
# lz_kda_step/lz_gdn_step sweep as kda_gt1, so a hash CHANGE proves that
# arm executes and feeds the output; a hash that does not move means the
# mutation is in dead code.
#
# Exit codes:
#   0  PASS - every mutation moved the hash it must
#   1  FAIL - a mutation did not move, or the gate build failed
#   2  SKIP - a mandatory tool is missing
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/recurxc}
CC=${CC:-gcc}
F="-O2 -std=c99"
SRC="ops ops_kernel ops_epi ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     fwht cpucheck err"

command -v gcc >/dev/null 2>&1 || { echo "recur_xcheck: SKIP - no gcc"; exit 2; }

mkdir -p "$O"
build() {   # build DIR TAG
    local d="$1" tag="$2" f list=""
    for f in $SRC; do list="$list $d/src/$f.c"; done
    $CC $F -I"$d/src" -o "$O/$tag" "$R/.prof/recur_xcheck.c" $list -lm \
        2>"$O/$tag.build.log"
}

if ! build "$R" gate; then
    echo "recur_xcheck: SKIP - probe did not build"
    tail -5 "$O/gate.build.log"
    exit 2
fi
echo "-- shipped --"
"$O/gate" | tee "$O/gate.out"
KREF=$(grep '^KDA' "$O/gate.out"); GREF=$(grep '^GDN' "$O/gate.out")
[ -n "$KREF" ] && [ -n "$GREF" ] || { echo "FAIL: no hashes"; exit 1; }

FAILED=0
# Three surgical mutations: each flips one arithmetic in one arm.
#   p2scale   the p2 group scale's fallback multiply (ops_gdn.c:1618)
#   delta     the pass-2 delta quantize's upper clamp (ops_gdn.c:2215)
#   pas1tap   the pass-1 tap's state*k product (ops_gdn.c:791)
for M in p2scale delta pas1tap; do
    rm -rf "$O/tree"; mkdir -p "$O/tree"; cp -r "$R/src" "$O/tree/src"
    case $M in
    p2scale) sed -i 's/else                       s_new = pow2f(ee);/else                       s_new = pow2f(ee + 1);/' \
                  "$O/tree/src/ops_gdn.c"; P='pow2f(ee + 1)' ;;
    delta)   sed -i 's/if (t >  32767) t =  32767;/if (t <  32767) t =  32767;/' \
                  "$O/tree/src/ops_gdn.c"; P='t <  32767' ;;
    pas1tap) sed -i 's/a = ssv \* k\[kk\];/a = ssv + k[kk];/' \
                  "$O/tree/src/ops_gdn.c"; P='ssv + k' ;;
    esac
    grep -q "$P" "$O/tree/src/ops_gdn.c" || {
        echo "FAIL: edit '$M' did not apply - its line moved"; FAILED=1; continue; }
    build "$O/tree" "$M" || { echo "FAIL: '$M' did not build"
                              tail -5 "$O/$M.build.log"; FAILED=1; continue; }
    K=$("$O/$M" | grep '^KDA'); G=$("$O/$M" | grep '^GDN')
    echo "-- $M --"
    if [ "$K" != "$KREF" ] && [ "$G" != "$GREF" ]; then
        echo "  both moved ($KREF -> $K, $GREF -> $G)"
    elif [ "$K" != "$KREF" ]; then
        echo "  KDA moved, GDN did not: the arm is KDA-only ($KREF -> $K, $GREF unchanged)"
    elif [ "$G" != "$GREF" ]; then
        echo "  GDN moved, KDA did not: the arm is GDN-only ($GREF -> $G, $KREF unchanged)"
    else
        echo "FAIL: '$M' did not move either hash - the arm is not reached"
        FAILED=1
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "recur_xcheck: FAIL"
    exit 1
fi
echo "recur_xcheck: PASS"
