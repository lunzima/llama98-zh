#!/bin/bash
# A csa2_on layer owns no DENSE KV plane.
#
# forward.c hands a csa2_on layer's whole attention forward to
# forward_csa2.c, which stores compressed blocks in csa2_comp_ring/
# csa2_idx_comp_ring and never reads per-position K/V (that file contains
# no reference to kq8/kf32 at all). The planes were nevertheless sized off
# c->n_full_layers, which counts every full_attention layer whatever its
# csa2_on flag - so a csa2 layer was charged for a plane nothing wrote and
# nothing read. Measured on kunkun98 at its 64-slot working context, q8:
# 36,832 B of 3,659,828.
#
# THE BEFORE ARM IS A MUTATION, NOT `git show HEAD:`. It was a git ref in
# the first version of this gate, and that version went red the moment the
# fix was committed - HEAD caught up to the working tree, both arms became
# the same binary, and every difference the gate asserts collapsed to zero.
# A git-pinned before-copy has exactly one day of life and it is the day
# before the fix lands. The mutation (M1 below: the pre-fix predicate) is
# the same difference, is available on any tree including one with no git,
# and stays available forever.
#
# WHAT EACH ARM PROVES
#   bit-identity  M1 allocates the dead plane and this tree does not, and
#                 the two produce byte-identical logits. That is the whole
#                 claim: the plane was dead, so not allocating it cannot
#                 move a number.
#   accounting    M1's state exceeds this tree's by EXACTLY the planes that
#                 stopped being allocated - one per removed layer, per
#                 slot, at that --kv format's own per-slot cost. Run at
#                 q8, q4 and f32: the three formats have three different
#                 plane budgets (q8 keeps a scale per 32 elements, q4 one
#                 per kv head, f32 none), so a single q8 number would not
#                 tell a correct removal from one that deleted the wrong
#                 plane in the other two. The dense_idx array this tree
#                 adds cancels here - M1 carries it too.
#   control       a model with no csa2 layer: M1 and this tree agree
#                 byte for byte and logit for logit, so the change is
#                 scoped to the models that have one.
#
# MUTATIONS, and the two arms are not interchangeable - each catches what
# the other cannot:
#   M2 "drops a live layer's plane too" (count one less): bit-identity RED
#      (the last dense layer's plane becomes a zero-length allocation);
#      the accounting arm stays green, because the number it computes is
#      derived from the config, not from the binary.
#   M3 "frees one plane fewer than the model has" (the count off by one
#      in the other direction): the accounting arm's own formula no longer
#      matches, by exactly one slot's plane. M1 shows the arm reds when the
#      count is absent; M3 shows it reds when the count is wrong.
#
# Both arms are built in work dirs, never in the tree: this script may run
# while another process holds $ROOT/llama98.exe open.
#
#   csa2_dense_kv_gate.sh [csa2-model] [control-model]
#
# Exit: 0 pass, 1 fail, 2 skip (no python3, no runnable model).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-E:/LLM/models/kunkun98}
CONTROL=${2:-E:/LLM/models/kmr20}
W=$(mktemp -d)
FAILED=0
PROMPT="人工智能的发展历史可以追溯到二十世纪五十年代。"

if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: no python3 (needed to read the model's geometry)"
    exit 2
fi
if [ ! -d "$MODEL" ]; then
    echo "SKIP: no model at $MODEL"
    exit 2
fi

# build_arm <dir> <sed-expr|-> <marker|->, applied to src/forward_state.c>
build_arm() {
    local dir="$1" expr="$2" marker="$3"
    mkdir -p "$dir"
    cp -r "$ROOT/src" "$ROOT/common" "$dir/"
    cp "$ROOT/Makefile" "$dir/Makefile"
    if [ "$expr" != "-" ]; then
        sed -i "$expr" "$dir/src/forward_state.c"
        if ! grep -q "$marker" "$dir/src/forward_state.c"; then
            echo "  FAIL: mutation did not apply (marker not found in forward_state.c)"
            return 1
        fi
    fi
    ( cd "$dir" && make native ) > "$dir/build.log" 2>&1 || {
        echo "  FAIL: build failed in $dir"; tail -3 "$dir/build.log"; return 1; }
    return 0
}

# state <binary> <model> <kv> -> "<bytes> <slots>" from the exact count
# cli_main prints beside the rounded MB.
state() {
    "$1" "$2" --prompt "$PROMPT" -n 1 --kv "$3" 2>/dev/null \
        | sed -n 's/^State *[0-9.]* MB (\([0-9]*\) B, \([0-9]*\) slots.*/\1 \2/p' | head -1
}
bytes_of() { state "$1" "$2" "$3" | cut -d' ' -f1; }

# The model's own geometry, from its config: what one removed layer costs
# per slot at each --kv format, and how many layers were removed.
#   q8  : two planes of kvd one-byte elements, plus one f32 scale per 32
#         elements of each plane (alloc_kv_cache's `nrow / 32`).
#   q4  : two planes of kvd/2, plus one f32 scale per (slot, kv head) per
#         plane (`nsc`).
#   f32 : two planes of kvd four-byte elements, no scales.
geom() {
    python3 - "$MODEL" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
kvd = int(c["num_key_value_heads"]) * int(c["head_dim"])
nkv = int(c["num_key_value_heads"])
rem = 0
for l in (c.get("csa2_layers") or []):
    if str((c.get("csa2_kv_mode") or {}).get(str(l), "full")).lower() == "full":
        rem += 1
print(kvd, nkv, rem)
PY
}
G=$(geom) || { echo "SKIP: could not read the model's geometry"; exit 2; }
set -- $G
KVD=$1; NKV=$2; REMOVED=$3
if [ "$REMOVED" -eq 0 ]; then
    echo "SKIP: $MODEL has no csa2 layer in FULL mode, so no plane is freed"
    exit 2
fi
plane_cost() {
    case "$1" in
        q8)  echo $(( 2 * KVD + 2 * (KVD / 32) * 4 )) ;;
        q4)  echo $(( KVD + 2 * NKV * 4 )) ;;
        f32) echo $(( 8 * KVD )) ;;
        *)   echo 0 ;;
    esac
}
echo "== csa2_dense_kv_gate: $MODEL (kvd=$KVD kv_heads=$NKV removed=$REMOVED) =="

build_arm "$W/now" - - || exit 1
# M1: the pre-fix predicate - every full_attention layer owns a plane.
build_arm "$W/m1" \
    's/return m->config.layer_types\[l\] == LZ_LT_FULL \&\& !m->layers\[l\].csa2_on;/return m->config.layer_types[l] == LZ_LT_FULL;/' \
    '== LZ_LT_FULL;' || exit 1
NOW=$W/now/llama98.exe
M1=$W/m1/llama98.exe

# ---- bit-identity: the dead plane changes no number --------------------
"$NOW" "$MODEL" --prompt "$PROMPT" --dump-prompt-tokens "$W/toks" -n 1 >/dev/null 2>&1
if [ ! -s "$W/toks" ]; then
    echo "  FAIL: could not produce a token list"
    exit 1
fi
r=0
"$M1"  "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/m1.lg"  >/dev/null 2>&1 || r=1
"$NOW" "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/now.lg" >/dev/null 2>&1 || r=1
if [ "$r" -ne 0 ]; then
    echo "  FAIL: a bit-identity arm exited non-zero"
    FAILED=1
elif cmp -s "$W/m1.lg" "$W/now.lg"; then
    echo "  bit-identity (the pre-fix predicate allocates the dead plane here): identical, $(wc -c < "$W/now.lg") bytes"
else
    echo "  FAIL: allocating the dead plane changed the logits"
    FAILED=1
fi

# ---- accounting: M1's extra bytes ARE the removed planes --------------
for kv in q8 q4 f32; do
    mb=$(state "$M1"  "$MODEL" "$kv")
    nb=$(state "$NOW" "$MODEL" "$kv")
    if [ -z "$mb" ] || [ -z "$nb" ]; then
        echo "  FAIL: no state line for --kv $kv"
        FAILED=1
        continue
    fi
    ms=${mb##* }; ns=${nb##* }
    want=$(( REMOVED * ns * $(plane_cost "$kv") ))
    got=$(( ${mb%% *} - ${nb%% *} ))
    if [ "$ms" != "$ns" ]; then
        echo "  FAIL: --kv $kv slot count moved ($ms -> $ns)"
        FAILED=1
    elif [ "$got" -eq "$want" ]; then
        echo "  accounting --kv $kv: the pre-fix arm's extra $got B == $REMOVED x $ns slots x $(plane_cost "$kv")"
    else
        echo "  FAIL: --kv $kv the pre-fix arm carries $got B more, expected $want"
        FAILED=1
    fi
done

# ---- control: a model with no csa2 layer ------------------------------
if [ -d "$CONTROL" ]; then
    "$NOW" "$CONTROL" --prompt "$PROMPT" --dump-prompt-tokens "$W/ctoks" -n 1 >/dev/null 2>&1
    if [ -s "$W/ctoks" ]; then
        c=0
        "$M1"  "$CONTROL" --tokens-file "$W/ctoks" --dump-logits "$W/m1.clg"  >/dev/null 2>&1 || c=1
        "$NOW" "$CONTROL" --tokens-file "$W/ctoks" --dump-logits "$W/now.clg" >/dev/null 2>&1 || c=1
        if [ "$c" -ne 0 ]; then
            echo "  FAIL: the control arm exited non-zero"
            FAILED=1
        elif ! cmp -s "$W/m1.clg" "$W/now.clg"; then
            echo "  FAIL: the control model's logits moved"
            FAILED=1
        elif [ "$(bytes_of "$M1" "$CONTROL" q8)" = "$(bytes_of "$NOW" "$CONTROL" q8)" ]; then
            echo "  control (no csa2 layer): same bytes, identical logits - the change is scoped"
        else
            echo "  FAIL: the control model's state moved"
            FAILED=1
        fi
    else
        echo "  SKIP: control model produced no token list"
    fi
else
    echo "  SKIP: no control model at $CONTROL"
fi

# ---- M2: a live layer's plane dropped too -----------------------------
if build_arm "$W/m2" 's/^    int l, n = 0;$/    int l, n = -1;/' 'int l, n = -1;'; then
    if "$W/m2/llama98.exe" "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/m2.lg" >/dev/null 2>&1 \
       && cmp -s "$W/m2.lg" "$W/now.lg"; then
        echo "  FAIL: M2 (a live layer's plane dropped) left the logits identical"
        FAILED=1
    else
        echo "  mutation M2 (a live layer's plane dropped): bit-identity RED, as the header claims"
    fi
else
    FAILED=1
fi

# ---- M3: one layer too many ------------------------------------------
# The accounting arm again, with a WRONG fix in the "now" seat: it frees
# one plane fewer than the model has, so the number the arm demands no
# longer matches by exactly one slot's plane. That is the criterion being
# sensitive to a count that is off by one - not merely to a count that is
# absent, which M1 already covers.
if build_arm "$W/m3" 's/^    int l, n = 0;$/    int l, n = 1;/' 'int l, n = 1;'; then
    st=$(state "$W/m3/llama98.exe" "$MODEL" q8)
    mb=$(bytes_of "$M1" "$MODEL" q8)
    if [ -z "$st" ] || [ -z "$mb" ]; then
        echo "  FAIL: M3 produced no state line"
        FAILED=1
    elif [ "$(( mb - ${st%% *} ))" -eq "$(( (REMOVED - 1) * ${st##* } * $(plane_cost q8) ))" ]; then
        echo "  mutation M3 (one plane too many allocated): the arm demands $(( REMOVED * ${st##* } * $(plane_cost q8) )) B and this fix removes $(( mb - ${st%% *} )) B - RED by one slot's plane"
    else
        echo "  FAIL: M3 is off by $(( mb - ${st%% *} )) B, expected one slot's plane less than the full removal"
        FAILED=1
    fi
else
    FAILED=1
fi

rm -rf "$W"
if [ "$FAILED" -eq 0 ]; then
    echo "PASS csa2_dense_kv_gate"
    exit 0
fi
echo "FAIL csa2_dense_kv_gate"
exit 1
