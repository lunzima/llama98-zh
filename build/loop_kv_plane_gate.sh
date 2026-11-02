#!/bin/bash
# Dense KV planes are numbered per APPLICATION.
#
# A looping tower runs one physical layer in more than one pass. Dense
# KV is addressed per application, so every pass owns its own plane. The
# refusal that remains (LZ_ERR_CFG_LOOP_KV) is for CSA2 layers, whose
# compressed rings are cache_idx-addressed - still per physical layer.
# This gate drives the real lz_state_alloc out of src/ and inspects the
# mapping it builds:
#
#   - the two applications of a repeated layer own DIFFERENT planes;
#   - the plane indexes are a bijection onto 0..n_dense_kv-1;
#   - the identity schedule keeps the legacy per-layer order;
#   - a CSA2 layer (whose rings are cache_idx-addressed, per physical
#     layer) inside the block is still refused, and one applied once is
#     not.
#
# THE BEFORE ARM IS A MUTATION, not `git show HEAD:`: a git-pinned copy
# goes stale the day the change lands. Every arm below is the current
# source with one edit, and each must turn its own check red:
#
#   M1  count planes per physical layer  -> n_dense_kv is short by every
#       repeated layer, and the bijection check sees an index past the
#       allocation.
#   M2  number planes per physical layer -> the repeated layer's two
#       applications land on ONE plane (the bug this change removes).
#   M3  the pre-change refusal (dense-KV layer applied twice) -> the
#       block case is refused instead of running.
#   M4  drop the CSA2 refusal -> a csa2 layer applied twice is accepted
#       and its rings would be overwritten silently.
#   M5  reverse the plane order -> the identity schedule's legacy order
#       check goes red. Note what this arm can and cannot prove: the
#       logits of a non-looping model are a pure renaming under this
#       mutation (each layer reads only the plane it wrote), so a
#       bit-identity comparison CANNOT see it - the order is pinned by
#       the mapping check instead.
#
# The common objects are compiled once and shared by every arm; only
# forward_state.c differs between arms, so each mutation costs one small
# compile and one link.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "loop_kv_plane_gate: SKIP - no C compiler"
    exit 2
}
make -s -C "$ROOT" print-ENG >/dev/null 2>&1 || {
    echo "loop_kv_plane_gate: SKIP - make cannot print the engine list"
    exit 2
}
ENG=$(make -s -C "$ROOT" print-ENG)
ENG_MMX=$(make -s -C "$ROOT" print-ENG_MMX)
ENG_AVX2=$(make -s -C "$ROOT" print-ENG_AVX2)

W=$(mktemp -d 2>/dev/null || echo /tmp/loopkvplanegate.$$)
mkdir -p "$W"
trap 'rm -rf "$W"' EXIT

cp -r "$ROOT/src" "$W/src"
cp "$ROOT/build/loop_kv_plane_probe.c" "$W/probe.c"

CFLAGS="-O0 -std=c99 -Wall -Wextra -ffp-contract=off \
-DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_AVX2_TU=1"
OBJ="$W/obj"
mkdir -p "$OBJ"

echo "-- compiling the shared engine objects --"
for f in $ENG; do
    $CC $CFLAGS -I"$W/src" -c -o "$OBJ/$f.o" "$W/src/$f.c" 2>"$W/cc_$f.log" || {
        echo "  FAIL: $f.c did not build"; tail -5 "$W/cc_$f.log"; exit 1; }
done
for f in $ENG_MMX; do
    $CC $CFLAGS -I"$W/src" -mmmx -msse -msse2 -c -o "$OBJ/$f.o" \
        "$W/src/$f.c" 2>"$W/cc_$f.log" || {
        echo "  FAIL: $f.c did not build"; tail -5 "$W/cc_$f.log"; exit 1; }
done
for f in $ENG_AVX2; do
    $CC $CFLAGS -I"$W/src" -mavx2 -c -o "$OBJ/$f.o" \
        "$W/src/$f.c" 2>"$W/cc_$f.log" || {
        echo "  FAIL: $f.c did not build"; tail -5 "$W/cc_$f.log"; exit 1; }
done
mv "$OBJ/forward_state.o" "$W/base_forward_state.o"

# The exact source text each mutation rewrites. Kept out of the heredocs
# so there is one place that must agree with forward_state.c, not two.
COUNT_NOW='    for (a = 0; a < c->exec_len; a++) {
        l = c->app_layer[a];
        if (layer_owns_dense_kv(m, l)) n++;
    }'
COUNT_M1='    for (l = 0; l < c->n_layers; l++) {
        if (layer_owns_dense_kv(m, l)) n++;
    }'
FILL_NOW='    {
        int a;
        for (a = 0; a < c->exec_len; a++) {
            int al = c->app_layer[a];
            s->app_dense_idx[a] = layer_owns_dense_kv(m, al) ? nd++ : -1;
        }
    }'
FILL_M2='    {
        int a;
        for (a = 0; a < c->exec_len; a++) s->app_dense_idx[a] = -1;
        for (l = 0; l < c->n_layers; l++) {
            if (!layer_owns_dense_kv(m, l)) continue;
            for (a = 0; a < c->exec_len; a++)
                if (c->app_layer[a] == l) s->app_dense_idx[a] = nd;
            nd++;
        }
    }'
FILL_M5_OLD='            s->app_dense_idx[a] = layer_owns_dense_kv(m, al) ? nd++ : -1;'
FILL_M5_NEW='            s->app_dense_idx[a] = layer_owns_dense_kv(m, al) ? (s->n_dense_kv - 1 - nd++) : -1;'
GUARD_CSA2='            if (!m->layers[l].csa2_on) continue;'
GUARD_DENSE='            if (!layer_owns_dense_kv(m, l)) continue;'
GUARD_NONE='            continue;'

mutate() {   # <arm> <old> <new>
    local arm="$1" old="$2" new="$3"
    python3 - "$W/src/forward_state.c" <<PYEOF || return 1
import sys
old = '''$old'''
new = '''$new'''
p = sys.argv[1]
src = open(p, encoding="utf-8").read()
if src.count(old) != 1:
    sys.exit("mutation anchor found %d times" % src.count(old))
open(p, "w", encoding="utf-8").write(src.replace(old, new, 1))
PYEOF
}

run_arm() {   # <arm> -> writes $W/<arm>.log, returns probe's exit code
    local arm="$1"
    cp "$W/src/forward_state.c" "$W/$arm.c"
    $CC $CFLAGS -I"$W/src" -c -o "$W/$arm.o" "$W/$arm.c" \
        2>"$W/cc_$arm.log" || {
        echo "  FAIL: mutated forward_state.c did not build ($arm)"
        tail -5 "$W/cc_$arm.log"; return 2; }
    $CC -O0 -o "$W/$arm.exe" "$W/probe.c" "$W/$arm.o" \
        $(for f in $ENG $ENG_MMX $ENG_AVX2; do \
              [ "$f" = forward_state ] || echo "$OBJ/$f.o"; done) \
        -I"$W/src" -lm 2>"$W/link_$arm.log" || {
        echo "  FAIL: $arm did not link"; tail -5 "$W/link_$arm.log"; return 2; }
    ( cd "$W" && "./$arm.exe" > "$arm.log" 2>&1 )
    return $?
}

FAILED=0

echo "-- current tree (must PASS) --"
cp "$W/src/forward_state.c" "$W/now.c"
run_arm now
RC=$?
cat "$W/now.log"
if [ "$RC" -eq 2 ]; then exit 1; fi
if [ "$RC" -ne 0 ]; then
    echo "  FAIL: the current tree does not pass its own probe"
    FAILED=1
fi

# restore the pristine copy after the now arm (run_arm does not mutate)
cp "$W/src/forward_state.c" "$W/pristine.c"

check_arm() {   # <name> <expected-fail lines...> -- <must-stay-pass lines...>
    local name="$1"; shift
    local fails=() passes=()
    local sep=0
    for x in "$@"; do
        if [ "$x" = "--" ]; then sep=1; continue; fi
        if [ "$sep" -eq 0 ]; then fails+=("$x"); else passes+=("$x"); fi
    done
    echo ""
    echo "-- mutant $name --"
    run_arm "$name"
    local rc=$?
    if [ "$rc" -eq 2 ]; then FAILED=1; return; fi
    cat "$W/$name.log"
    local miss=0
    for line in "${fails[@]}"; do
        grep -qF "$line" "$W/$name.log" || {
            echo "  FAIL: mutant $name did not turn red on: $line"
            miss=1; }
    done
    for line in "${passes[@]}"; do
        grep -qF "$line" "$W/$name.log" || {
            echo "  FAIL: mutant $name lost an unrelated PASS: $line"
            miss=1; }
    done
    [ "$miss" -eq 0 ] && echo "  mutation $name: RED on the expected check(s), everything else unchanged"
    [ "$miss" -ne 0 ] && FAILED=1
}

A_PASS='PASS case A: a dense-KV layer applied twice is accepted'
A_PLANES='PASS case A: three dense planes are allocated'
A_BIJ='PASS case A: the application planes are a bijection'
A_DIFF='PASS case A: the two applications of layer 2 land in DIFFERENT'
A0_ORDER='PASS case A0: the identity schedule keeps the legacy per-layer'
A2_PLANES='PASS case A2: one plane per application'
A2_DISTINCT='PASS case A2: all three passes of layer 2 are distinct'
A3_SAME='PASS case A3: each extra pass costs the same number of bytes'
A3_PLANE='PASS case A3: that cost is larger than the application-table'
B_REFUSED='PASS case B: a CSA2 layer applied twice is refused'
B_NAMES='PASS case B: the refusal names the CSA2 layer'
C_ACCEPTED='PASS case C: a looping tower whose CSA2 layer runs once'

mutate_pristine() {   # <arm> <old> <new>
    cp "$W/pristine.c" "$W/src/forward_state.c"
    mutate "$1" "$2" "$3"
}

mutate_pristine m1 "$COUNT_NOW" "$COUNT_M1" || exit 1
check_arm m1 \
    'FAIL case A: three dense planes are allocated' \
    'FAIL case A: the application planes are a bijection' \
    'FAIL case A2: one plane per application' \
    'FAIL case A3: that cost is larger than the application-table' \
    -- "$A_DIFF" "$A0_ORDER" "$B_REFUSED" "$C_ACCEPTED"

mutate_pristine m2 "$FILL_NOW" "$FILL_M2" || exit 1
check_arm m2 \
    'FAIL case A: the application planes are a bijection' \
    'FAIL case A: the two applications of layer 2 land in DIFFERENT' \
    'FAIL case A2: all three passes of layer 2 are distinct' \
    -- "$A_PLANES" "$A3_SAME" "$A3_PLANE" "$A0_ORDER" "$B_REFUSED" "$C_ACCEPTED"

mutate_pristine m3 "$GUARD_CSA2" "$GUARD_DENSE" || exit 1
check_arm m3 \
    'FAIL case A: a dense-KV layer applied twice is accepted' \
    'FAIL case A2: a three-pass block allocates' \
    'FAIL case A3: one, two and three passes of the dense layer all allocate' \
    'FAIL case B: a CSA2 layer applied twice is refused' \
    -- "$A0_ORDER" "$C_ACCEPTED"

mutate_pristine m4 "$GUARD_CSA2" "$GUARD_NONE" || exit 1
check_arm m4 \
    'FAIL case B: a CSA2 layer applied twice is refused' \
    'FAIL case B: the refusal names the CSA2 layer' \
    -- "$A_PASS" "$A_DIFF" "$A3_SAME" "$A3_PLANE" "$A0_ORDER" "$C_ACCEPTED"

mutate_pristine m5 "$FILL_M5_OLD" "$FILL_M5_NEW" || exit 1
check_arm m5 \
    'FAIL case A0: the identity schedule keeps the legacy per-layer' \
    -- "$A_PLANES" "$A_BIJ" "$A_DIFF" "$A2_DISTINCT" "$A3_SAME" "$A3_PLANE" \
       "$B_REFUSED" "$C_ACCEPTED"

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "PASS loop_kv_plane_gate"
    exit 0
fi
echo ""
echo "FAIL loop_kv_plane_gate"
exit 1
