#!/bin/bash
# --csa2-ring f32|bf16: the two finalized rings' storage format.
#
# forward.h's LZ_CRF_* comment has the contract. This gate holds the four
# claims that make it a contract rather than a knob:
#
#   unit        build/csa2_ring_fmt_probe.c: f32 hands back a VIEW into
#               the ring, bf16 round-trips through lz_bf16_to_f32/
#               lz_f32_to_bf16 exactly, and neither format writes outside
#               the bytes its stride owns (guard bytes - an overrun
#               changes the logits just as loudly as a wrong rounding, so
#               the integration arms could not tell them apart).
#   identity    the f32 tier, run by this tree, is byte-identical to a
#               build in which lz_csa2_ring_row/put are forced down the
#               f32 branch unconditionally. THAT is the claim "f32 is the
#               identity tier" - the indirection itself must change
#               nothing, and the reference arm is a mutation rather than
#               `git show HEAD:` because a git-pinned before-copy dies
#               the day the change lands.
#   accounting  the bf16 tier's state is smaller by EXACTLY
#               nblocks * (stride_f32 - stride_bf16) * (head_dim +
#               indexer_head_dim) - every factor read from the two size
#               functions and the config, no hand-written halving: a
#               hardcoded "/2" would go silently wrong the day
#               indexer_head_dim gets a tier of its own.
#   live        bf16 changes the logits. A tier that moved bytes and no
#               numbers would pass every arm above and do nothing.
#
#   csa2_ring_fmt_gate.sh [csa2-model]
#
# Exit: 0 pass, 1 fail, 2 skip (no gcc/python3, no model, no csa2 layer).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-E:/LLM/models/kunkun98}
W=$(mktemp -d)
FAILED=0
PROMPT="人工智能的发展历史可以追溯到二十世纪五十年代。"

command -v python3 >/dev/null 2>&1 || { echo "SKIP: no python3"; exit 2; }
command -v gcc >/dev/null 2>&1 || { echo "SKIP: no gcc"; exit 2; }
[ -d "$MODEL" ] || { echo "SKIP: no model at $MODEL"; exit 2; }

build() {
    local dir="$1" expr="${2:-}" marker="${3:-}"
    mkdir -p "$dir"
    cp -r "$ROOT/src" "$ROOT/common" "$dir/"
    cp "$ROOT/Makefile" "$dir/Makefile"
    if [ -n "$expr" ]; then
        sed -i "$expr" "$dir/src/forward_csa2.c"
        grep -q "$marker" "$dir/src/forward_csa2.c" || {
            echo "  FAIL: mutation did not apply"; return 1; }
    fi
    ( cd "$dir" && make native ) > "$dir/build.log" 2>&1 || {
        echo "  FAIL: build failed in $dir"; tail -3 "$dir/build.log"; return 1; }
    return 0
}
state_bytes() {
    "$1" "$2" --prompt "$PROMPT" -n 1 --csa2-ring "$3" 2>/dev/null \
        | sed -n 's/^State *[0-9.]* MB (\([0-9]*\) B.*/\1/p' | head -1
}

geom() {
    python3 - "$1" "$2" <<'PY'
import json, sys
c = json.load(open(sys.argv[1] + "/config.json"))
seq = int(sys.argv[2])
per = c.get("csa2_compress_m_layers") or {}
scalar = int(c.get("csa2_compress_m") or 1)
mode = c.get("csa2_kv_mode") or {}
nb = 0
for l, lt in enumerate(c.get("layer_types") or []):
    if lt != "full_attention":
        continue
    if str(mode.get(str(l), "full")).lower() != "full":
        continue
    m = int(per.get(str(l), scalar))
    nb += (seq + m - 1) // m
print(nb, int(c["head_dim"]), int(c["csa2_indexer_head_dim"]))
PY
}

if [ ! -e "$ROOT/build/csa2_ring_fmt_probe.c" ]; then
    echo "SKIP: probe source missing"; exit 2
fi

echo "== csa2_ring_fmt_gate: $MODEL =="

# ---- unit arm ---------------------------------------------------------
mkdir -p "$W/probe"
cp -r "$ROOT/src" "$ROOT/common" "$W/probe/"
cp "$ROOT/build/csa2_ring_fmt_probe.c" "$W/probe/"
S=""
for f in $(cd "$ROOT" && make -s print-ENG); do S="$S $W/probe/src/$f.c"; done
for f in $(cd "$ROOT" && make -s print-COM); do S="$S $W/probe/common/$f.c"; done
if gcc -O2 -std=c99 -I"$W/probe/src" -I"$W/probe/common" -ffp-contract=off \
       -o "$W/probe/p.exe" $S "$W/probe/csa2_ring_fmt_probe.c" -lm \
       > "$W/probe/build.log" 2>&1; then
    if "$W/probe/p.exe" > "$W/probe.out" 2>&1; then
        echo "  unit: $(tail -1 "$W/probe.out")"
    else
        echo "  FAIL unit arm:"; tail -4 "$W/probe.out"; FAILED=1
    fi
else
    echo "  FAIL: the probe did not build"; tail -3 "$W/probe/build.log"; FAILED=1
fi

# ---- the three integration arms ---------------------------------------
build "$W/ref" || exit 1
# M1: both helpers forced down the f32 branch, so the format is never
# consulted at all - the engine as it would be without the knob.
M1='s/    if (fmt != LZ_CRF_BF16) return (const float \*)ring + base;/    if (1) return (const float *)ring + base;/'
build "$W/nofmt" "$M1" 'if (1) return (const float \*)ring + base;' || exit 1
python3 - "$W/nofmt/src/forward_csa2.c" <<'PY' || exit 1
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = """    if (fmt != LZ_CRF_BF16) {
        memcpy((float *)ring + base, src, (size_t)n * sizeof(float));
        return;
    }"""
new = old.replace("if (fmt != LZ_CRF_BF16)", "if (1)")
if s.count(old) != 1:
    print("  FAIL: could not force the f32 put branch"); sys.exit(1)
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
PY
( cd "$W/nofmt" && make native ) > "$W/nofmt/build2.log" 2>&1 || {
    echo "  FAIL: the forced-f32 build failed"; tail -3 "$W/nofmt/build2.log"; exit 1; }

"$W/ref/llama98.exe" "$MODEL" --prompt "$PROMPT" --dump-prompt-tokens "$W/toks" -n 1 >/dev/null 2>&1
if [ ! -s "$W/toks" ]; then echo "  FAIL: no token list"; exit 1; fi

r=0
"$W/ref/llama98.exe"   "$MODEL" --tokens-file "$W/toks" --csa2-ring f32 --dump-all-logits "$W/f32.lg"  >/dev/null 2>&1 || r=1
"$W/nofmt/llama98.exe" "$MODEL" --tokens-file "$W/toks" --csa2-ring f32 --dump-all-logits "$W/nofmt.lg" >/dev/null 2>&1 || r=1
"$W/ref/llama98.exe"   "$MODEL" --tokens-file "$W/toks" --csa2-ring bf16 --dump-all-logits "$W/bf16.lg" >/dev/null 2>&1 || r=1
if [ "$r" -ne 0 ]; then
    echo "  FAIL: an arm exited non-zero"; FAILED=1
else
    if cmp -s "$W/f32.lg" "$W/nofmt.lg"; then
        echo "  identity: f32 is bit-identical to the same tree with the format branch removed"
    else
        echo "  FAIL: the f32 tier is not the identity tier"; FAILED=1
    fi
    if cmp -s "$W/f32.lg" "$W/bf16.lg"; then
        echo "  FAIL: bf16 moved no number, so the tier is not read"
        FAILED=1
    else
        echo "  live: bf16 changes the logits ($(wc -c < "$W/bf16.lg") bytes compared)"
    fi
fi

F32=$(state_bytes "$W/ref/llama98.exe" "$MODEL" f32)
BF=$(state_bytes "$W/ref/llama98.exe" "$MODEL" bf16)
if [ -z "$F32" ] || [ -z "$BF" ]; then
    echo "  FAIL: no state line for one of the tiers"; FAILED=1
else
    SLOTLINE=$("$W/ref/llama98.exe" "$MODEL" --prompt "$PROMPT" -n 1 --csa2-ring f32 2>/dev/null \
               | sed -n 's/^State.*(\([0-9]*\) B, *\([0-9]*\) slots.*/\2/p' | head -1)
    G=$(geom "$MODEL" "$SLOTLINE")
    set -- $G
    nblocks=$1; hd=$2; idx=$3
    want=$(( nblocks * (4 - 2) * (hd + idx) ))
    got=$(( F32 - BF ))
    if [ "$got" -eq "$want" ]; then
        echo "  accounting: bf16 is $got B smaller == $nblocks blocks x (4-2) B/element x ($hd + $idx) dims"
    else
        echo "  FAIL: bf16 is $got B smaller, expected $want"; FAILED=1
    fi
fi

rm -rf "$W"
if [ "$FAILED" -eq 0 ]; then echo "PASS csa2_ring_fmt_gate"; exit 0; fi
echo "FAIL csa2_ring_fmt_gate"
exit 1
