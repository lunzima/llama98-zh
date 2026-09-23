#!/bin/bash
# A csa2_on layer owns no DENSE KV plane.
#
# forward.c hands a csa2_on layer's whole attention forward to
# forward_csa2.c, which stores compressed blocks in csa2_comp_ring/
# csa2_idx_comp_ring and never reads per-position K/V (that file contains
# no reference to kq8/kf32 at all). The planes were nevertheless sized off
# c->n_full_layers, which counts every full_attention layer whatever its
# csa2_on flag - so a csa2 layer was charged for a plane nothing wrote and
# nothing read. This gate holds the two consequences:
#
#   bit-identity    A model with a csa2 layer, run by the binary built
#                   from the previous source and by the one built from
#                   the current source, must produce byte-identical
#                   logits. That is the whole claim: the plane was dead,
#                   so removing it cannot move a number. --tokens-file is
#                   the observable both sides have, and it calls
#                   lz_forward one token at a time, so it reaches
#                   attn_impl and forward_csa2 for every position.
#   plane accounting  The state shrinks by EXACTLY the planes that stopped
#                   being allocated - one per removed layer, per slot, at
#                   that --kv format's own per-slot cost - and by nothing
#                   else. Run at q8, q4 and f32: the three formats have
#                   three different plane budgets (q8 keeps a scale per 32
#                   elements, q4 one per kv head, f32 none), so a single
#                   q8 number would not tell a correct removal from one
#                   that deleted the wrong plane in the other two.
#
# The byte assertion is the one the bit-identity arm cannot make: with the
# old count restored (mutation M1 below) every logit is still identical,
# because the plane really is dead - only the number of bytes moves. Both
# arms are therefore required; either alone passes a wrong change.
#
# MUTATIONS (each on a copy of src/, never by editing the tree)
#   M1 "counts every full layer again" - the pre-fix predicate. Expect the
#      byte arm RED and the bit-identity arm GREEN. This is the pairing
#      above, and it is also why this gate cannot be replaced by the
#      bit-identity comparison alone.
#   M2 "drops a live layer's plane too" - dense_kv_layers returns one less.
#      Expect the bit-identity arm RED (the last dense layer's plane is
#      then a zero-length allocation): it proves the identity arm can see
#      a deletion that DOES matter, not merely any diff.
#
# THE HEAD ARM'S SOURCE SET: HEAD's forward.h/forward_state.c/forward.c/
# forward_attn.c/forward_mtp.c plus the CURRENT cli_main.c. The byte arm
# needs the exact byte count, which HEAD's cli_main.c does not print
# ("%.1f MB" alone cannot resolve 36,832 B); cli_main.c takes no part in
# the plane accounting this gate is about. Every file that does is HEAD's.
#
# Both arms are built in work dirs, never in the tree: this script may run
# while another process holds $ROOT/llama98.exe open.
#
#   csa2_dense_kv_gate.sh [csa2-model] [control-model]
#
# Exit: 0 pass, 1 fail, 2 skip (no python3/safetensors, no runnable model).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-E:/LLM/models/kunkun98}
CONTROL=${2:-E:/LLM/models/kmr20}
W=$(mktemp -d)
FAILED=0
PROMPT="人工智能的发展历史可以追溯到二十世纪五十年代。"

# Every file the plane accounting runs through, from HEAD.
HEADFILES="src/forward.h src/forward_state.c src/forward.c src/forward_attn.c src/forward_mtp.c"

if ! command -v python3 >/dev/null 2>&1; then
    echo "SKIP: no python3 (needed to read the model's geometry)"
    exit 2
fi
if [ ! -d "$MODEL" ]; then
    echo "SKIP: no model at $MODEL"
    exit 2
fi

# build_arm <dir> <head|now> [<file> <sed-expr> <marker>]
# Copies src/common/Makefile, optionally replaces HEAD's five files,
# optionally applies one mutation to one file, and builds. $MUTANT is set
# to the file the mutation landed in so "it did not apply" is checked
# rather than inferred from a later failure.
build_arm() {
    local dir="$1" kind="$2" tfile="${3:-}" texpr="${4:-}" marker="${5:-}"
    mkdir -p "$dir"
    cp -r "$ROOT/src" "$ROOT/common" "$dir/"
    cp "$ROOT/Makefile" "$dir/Makefile"
    if [ "$kind" = head ]; then
        local f
        for f in $HEADFILES; do
            git -C "$ROOT" show "HEAD:$f" > "$dir/$f" || return 1
        done
    fi
    if [ -n "$tfile" ]; then
        sed -i "$texpr" "$dir/$tfile"
        if ! grep -q "$marker" "$dir/$tfile"; then
            echo "  FAIL: mutation did not apply to $tfile (marker not found)"
            return 1
        fi
    fi
    ( cd "$dir" && make native ) > "$dir/build.log" 2>&1 || {
        echo "  FAIL: build failed in $dir"; tail -3 "$dir/build.log"; return 1; }
    return 0
}

# state_bytes <binary> <model> <kv> -> "<bytes> <slots>" from the exact
# count cli_main prints beside the rounded MB.
state_bytes() {
    "$1" "$2" --prompt "$PROMPT" -n 1 --kv "$3" 2>/dev/null \
        | sed -n 's/^State *[0-9.]* MB (\([0-9]*\) B, \([0-9]*\) slots.*/\1 \2/p' | head -1
}

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
    mode = (c.get("csa2_kv_mode") or {}).get(str(l), "full")
    if str(mode).lower() == "full":
        rem += 1
print(kvd, nkv, rem, int(c["num_hidden_layers"]))
PY
}

G=$(geom) || { echo "SKIP: could not read the model's geometry"; exit 2; }
set -- $G
KVD=$1; NKV=$2; REMOVED=$3; NLAYERS=$4
if [ "$REMOVED" -eq 0 ]; then
    echo "SKIP: $MODEL has no csa2 layer in FULL mode, so no plane is freed"
    exit 2
fi
plane_cost() {   # <format>
    case "$1" in
        q8)  echo $(( 2 * KVD + 2 * (KVD / 32) * 4 )) ;;
        q4)  echo $(( KVD + 2 * NKV * 4 )) ;;
        f32) echo $(( 8 * KVD )) ;;
        *)   echo 0 ;;
    esac
}
# The fix also allocates dense_idx: one int per layer, in every model.
IDX_BYTES=$(( NLAYERS * 4 ))

echo "== csa2_dense_kv_gate: $MODEL (kvd=$KVD kv_heads=$NKV removed=$REMOVED idx=$IDX_BYTES) =="

if ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    echo "  FAIL: not a git tree, so the HEAD arm cannot be built"
    exit 1
fi

build_arm "$W/head" head || exit 1
build_arm "$W/now"  now  || exit 1
NOW=$W/now/llama98.exe
HEAD=$W/head/llama98.exe

# ---- arm 1: bit-identity ---------------------------------------------
"$HEAD" "$MODEL" --prompt "$PROMPT" --dump-prompt-tokens "$W/toks" -n 1 >/dev/null 2>&1
if [ ! -s "$W/toks" ]; then
    echo "  FAIL: could not produce a token list"
    exit 1
fi
r1=0
"$HEAD" "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/head.lg" >/dev/null 2>&1 || r1=1
"$NOW"  "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/now.lg"  >/dev/null 2>&1 || r1=1
if [ "$r1" -ne 0 ]; then
    echo "  FAIL: a bit-identity arm exited non-zero"
    FAILED=1
elif cmp -s "$W/head.lg" "$W/now.lg"; then
    echo "  bit-identity (csa2 model, $(wc -c < "$W/now.lg") bytes): identical"
else
    echo "  FAIL: removing the dead plane changed the logits"
    FAILED=1
fi

# ---- arm 2: plane accounting, one --kv format at a time ---------------
for kv in q8 q4 f32; do
    h=$(state_bytes "$HEAD" "$MODEL" "$kv")
    n=$(state_bytes "$NOW"  "$MODEL" "$kv")
    if [ -z "$h" ] || [ -z "$n" ]; then
        echo "  FAIL: no state line for --kv $kv"
        FAILED=1
        continue
    fi
    hb=${h%% *}; hs=${h##* }
    nb=${n%% *}; ns=${n##* }
    want=$(( REMOVED * hs * $(plane_cost "$kv") - IDX_BYTES ))
    got=$(( hb - nb ))
    if [ "$hs" != "$ns" ]; then
        echo "  FAIL: --kv $kv slot count moved ($hs -> $ns)"
        FAILED=1
    elif [ "$got" -eq "$want" ]; then
        echo "  plane accounting --kv $kv: freed $got B == $REMOVED x $hs slots x $(plane_cost "$kv") - $IDX_BYTES"
    else
        echo "  FAIL: --kv $kv freed $got B, expected $want"
        FAILED=1
    fi
done

# ---- arm 3: control, a model with no csa2 layer -----------------------
if [ -d "$CONTROL" ]; then
    "$HEAD" "$CONTROL" --prompt "$PROMPT" --dump-prompt-tokens "$W/ctoks" -n 1 >/dev/null 2>&1
    if [ -s "$W/ctoks" ]; then
        c=0
        "$HEAD" "$CONTROL" --tokens-file "$W/ctoks" --dump-logits "$W/head.clg" >/dev/null 2>&1 || c=1
        "$NOW"  "$CONTROL" --tokens-file "$W/ctoks" --dump-logits "$W/now.clg"  >/dev/null 2>&1 || c=1
        ch=$(state_bytes "$HEAD" "$CONTROL" q8)
        cn=$(state_bytes "$NOW"  "$CONTROL" q8)
        if [ "$c" -ne 0 ]; then
            echo "  FAIL: the control arm exited non-zero"
            FAILED=1
        elif ! cmp -s "$W/head.clg" "$W/now.clg"; then
            echo "  FAIL: the control model's logits moved"
            FAILED=1
        elif [ "${cn%% *}" = "$(( ${ch%% *} + IDX_BYTES ))" ]; then
            echo "  control (no csa2 layer): logits identical, no plane freed, +$IDX_BYTES B for the new index"
        else
            echo "  FAIL: control state moved by $(( ${cn%% *} - ${ch%% *} )) B, expected +$IDX_BYTES"
            FAILED=1
        fi
    else
        echo "  SKIP: control model produced no token list"
    fi
else
    echo "  SKIP: no control model at $CONTROL"
fi

# ---- mutations --------------------------------------------------------
# M1: the pre-fix predicate - every full_attention layer owns a plane.
if build_arm "$W/m1" now src/forward_state.c \
       's/return m->config.layer_types\[l\] == LZ_LT_FULL \&\& !m->layers\[l\].csa2_on;/return m->config.layer_types[l] == LZ_LT_FULL;/' \
       '== LZ_LT_FULL;'; then
    m=$("$W/m1/llama98.exe" "$MODEL" --prompt "$PROMPT" -n 1 --kv q8 2>/dev/null \
        | sed -n 's/^State *[0-9.]* MB (\([0-9]*\) B.*/\1/p' | head -1)
    n=$("$NOW" "$MODEL" --prompt "$PROMPT" -n 1 --kv q8 2>/dev/null \
        | sed -n 's/^State *[0-9.]* MB (\([0-9]*\) B.*/\1/p' | head -1)
    if [ -z "$m" ] || [ -z "$n" ]; then
        echo "  FAIL: M1 produced no state line"
        FAILED=1
    elif [ "$m" -gt "$n" ]; then
        "$W/m1/llama98.exe" "$MODEL" --tokens-file "$W/toks" --dump-all-logits "$W/m1.lg" >/dev/null 2>&1
        if cmp -s "$W/m1.lg" "$W/now.lg"; then
            echo "  mutation M1 (old predicate): byte arm RED, bit-identity still green - the pairing the header claims"
        else
            echo "  FAIL: M1 changed the logits, so the two arms are not the pair this gate assumes"
            FAILED=1
        fi
    else
        echo "  FAIL: M1 did not restore the extra plane ($m <= $n)"
        FAILED=1
    fi
else
    FAILED=1
fi

# M2: drop a live layer's plane as well - the identity arm must see it.
if build_arm "$W/m2" now src/forward_state.c \
       's/^    int l, n = 0;$/    int l, n = -1;/' 'int l, n = -1;'; then
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

rm -rf "$W"
if [ "$FAILED" -eq 0 ]; then
    echo "PASS csa2_dense_kv_gate"
    exit 0
fi
echo "FAIL csa2_dense_kv_gate"
exit 1
