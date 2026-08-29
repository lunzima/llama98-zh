#!/bin/bash
# Build and run the gt == 1.0f epilogue check against the OLD source,
# plus the mutation that shows the arm is reached.
#
#   bash .prof/kda_gt1_xcheck.sh
#
# control  the condition disabled (`gb.u == 0u`, never true) - the whole
#          epilogue back on the general loop, which is what the code did
#          before. Must produce IDENTICAL hashes.
# mutation the fast arm's subtract flipped to an add. Must produce a
#          DIFFERENT KDA hash and the SAME GDN one - different proves the
#          arm runs at all (no available model reaches it, see the .c),
#          same proves the two arms did not get crossed.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/kdagt1}
CC=${CC:-gcc}
F="-O2 -std=c99"
SRC="ops ops_kernel ops_epi ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     fwht cpucheck err"

mkdir -p "$O"
build() {   # build DIR TAG
    local d="$1" tag="$2" f list=""
    for f in $SRC; do list="$list $d/src/$f.c"; done
    $CC $F -I"$d/src" -o "$O/$tag" "$R/.prof/kda_gt1_xcheck.c" $list -lm \
        2>"$O/$tag.build.log"
}

if ! build "$R" gate; then
    echo "kda_gt1_xcheck: SKIP - probe did not build"
    tail -5 "$O/gate.build.log"
    exit 2
fi
echo "-- shipped --"
"$O/gate" | tee "$O/gate.out"
KREF=$(grep '^KDA' "$O/gate.out"); GREF=$(grep '^GDN' "$O/gate.out")
[ -n "$KREF" ] && [ -n "$GREF" ] || { echo "FAIL: no hashes"; exit 1; }

FAILED=0
for M in control mutation; do
    rm -rf "$O/tree"; mkdir -p "$O/tree"; cp -r "$R/src" "$O/tree/src"
    case $M in
    control)  sed -i 's/if (gb.u == 0x3F800000u) {/if (gb.u == 0u) {/' \
                  "$O/tree/src/ops_gdn.c"; P='gb.u == 0u' ;;
    mutation) sed -i '0,/u\[vv\] = (v\[vv\] - u\[vv\]) \* beta;/s//u[vv] = (v[vv] + u[vv]) * beta;/' \
                  "$O/tree/src/ops_gdn.c"; P='(v\[vv\] + u\[vv\])' ;;
    esac
    grep -q "$P" "$O/tree/src/ops_gdn.c" || {
        echo "FAIL: edit '$M' did not apply - its line moved"; FAILED=1; continue; }
    build "$O/tree" "$M" || { echo "FAIL: '$M' did not build"
                              tail -5 "$O/$M.build.log"; FAILED=1; continue; }
    K=$("$O/$M" | grep '^KDA'); G=$("$O/$M" | grep '^GDN')
    if [ "$M" = control ]; then
        if [ "$K" = "$KREF" ] && [ "$G" = "$GREF" ]; then
            echo "-- control: identical, as required --"
        else
            echo "FAIL: the fast arm is not the same function as the general"
            echo "      one (KDA $K vs $KREF, GDN $G vs $GREF)"
            FAILED=1
        fi
    else
        if [ "$K" = "$KREF" ]; then
            echo "FAIL: mutation left the KDA hash unchanged - the gt == 1"
            echo "      arm is never reached and the control proved nothing"
            FAILED=1
        elif [ "$G" != "$GREF" ]; then
            echo "FAIL: mutation changed the GDN hash too - the two arms are"
            echo "      crossed (GDN must not reach the gt == 1 loop)"
            FAILED=1
        else
            echo "-- mutation: KDA differs, GDN unchanged, as required --"
        fi
    fi
done
[ $FAILED -eq 0 ] || exit 1
echo "kda_gt1_xcheck: PASS"
exit 0
