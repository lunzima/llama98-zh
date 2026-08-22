#!/bin/bash
# Build and run the attention row-epilogue cross-check against the OLD
# source of each of the two changes, plus one mutation that must be red.
#
#   bash .prof/attn_fold_xcheck.sh
#
# Control trees (must produce the SAME hash as the shipped tree):
#   fold4   `ng >= 4 ?` widened to `ng >= 0 ?` - the four-accumulator
#           form on every width, which is what the code did before.
#   scale2  `* sc` restored to `* (1.0f / 256.0f) * scale` in the row.
# Mutation (must produce a DIFFERENT hash):
#   fold1   `ng >= 4 ?` narrowed to `ng >= 99 ?` - a0 alone on every
#           width, which is WRONG for ng >= 4 and is the exact error
#           the ng < 4 arm would be if its precondition were false.
#           Without it a sweep that never reached ng >= 4 would pass.
set -u
R=${R:-$(cd "$(dirname "$0")/.." && pwd)}
O=${O:-/tmp/attnfold}
CC=${CC:-gcc}
F="-O2 -std=c99"
SRC="ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_scalar
     fwht cpucheck err"

mkdir -p "$O"

build() {   # build DIR TAG
    local d="$1" tag="$2" f list=""
    for f in $SRC; do list="$list $d/src/$f.c"; done
    $CC $F -I"$d/src" -o "$O/$tag" "$R/.prof/attn_fold_xcheck.c" $list -lm \
        2>"$O/$tag.build.log"
}

if ! build "$R" gate; then
    echo "attn_fold_xcheck: SKIP - probe did not build"
    tail -5 "$O/gate.build.log"
    exit 2
fi
echo "-- shipped --"
"$O/gate" | tee "$O/gate.out"
REF=$(grep ATTFOLD "$O/gate.out")
[ -n "$REF" ] || { echo "FAIL: no hash from the shipped build"; exit 1; }

mk() {      # mk TAG SED-EXPR GREP-PROOF
    rm -rf "$O/tree"; mkdir -p "$O/tree"; cp -r "$R/src" "$O/tree/src"
    sed -i "$2" "$O/tree/src/ops_gdn.c"
    if ! grep -q "$3" "$O/tree/src/ops_gdn.c"; then
        echo "FAIL: edit '$1' did not apply - its line moved"
        return 2
    fi
    build "$O/tree" "$1" || { echo "FAIL: '$1' did not build"
                              tail -5 "$O/$1.build.log"; return 2; }
    return 0
}

FAILED=0
for M in fold4 scale2 fold1; do
    case $M in
    fold4)  E='s/ng >= 4 ? ((a0 + a2) + (a1 + a3)) : a0/ng >= 0 ? ((a0 + a2) + (a1 + a3)) : a0/'
            P='ng >= 0 ?' ; WANT=same ;;
    scale2) E='s/: a0) \* sc;/: a0) * (1.0f \/ 256.0f) * scale;/'
            P='(1.0f / 256.0f) \* scale;' ; WANT=same ;;
    fold1)  E='s/ng >= 4 ? ((a0 + a2) + (a1 + a3)) : a0/ng >= 99 ? ((a0 + a2) + (a1 + a3)) : a0/'
            P='ng >= 99 ?' ; WANT=diff ;;
    esac
    mk "$M" "$E" "$P" || { FAILED=1; continue; }
    GOT=$("$O/$M" | grep ATTFOLD)
    if [ "$WANT" = same ]; then
        if [ "$GOT" = "$REF" ]; then
            echo "-- control $M: identical, as required --"
        else
            echo "FAIL: control '$M' changed the output - the rewrite is not"
            echo "      the same function ($GOT vs $REF)"
            FAILED=1
        fi
    else
        if [ "$GOT" != "$REF" ]; then
            echo "-- mutation $M: differs, as required --"
        else
            echo "FAIL: mutation '$M' left the hash unchanged - the sweep"
            echo "      never reaches ng >= 4, so the control above proved"
            echo "      nothing about that arm"
            FAILED=1
        fi
    fi
done
[ $FAILED -eq 0 ] || exit 1
echo "attn_fold_xcheck: PASS (both controls identical, mutation red)"
exit 0
