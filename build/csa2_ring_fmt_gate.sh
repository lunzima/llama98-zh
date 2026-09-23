#!/bin/bash
# --csa2-ring f32|bf16|q8|q16: the two finalized rings' storage format.
#
# forward.h's LZ_CRF_* comment has the contract. This gate holds the four
# claims that make it a contract rather than a knob:
#
#   unit        build/csa2_ring_fmt_probe.c: f32 hands back a VIEW into
#               the ring; each narrow tier round-trips as a fixed point
#               and within one quantum; a zero row reads back as zeros;
#               and no row writes outside the bytes
#               lz_csa2_ring_rowbytes gives it - guard bytes, because an
#               overrun changes the logits exactly as loudly as a wrong
#               rounding, and the integration arms below cannot tell
#               those two apart. The probe also PRINTS rowbytes for the
#               model's own two widths, which is where the accounting arm
#               below gets them.
#   identity    the f32 tier, run by this tree, is byte-identical to a
#               build whose tier selector always answers f32 - so nothing
#               downstream (allocation, rowbytes, row, put) ever sees a
#               narrow format. THAT is the claim "f32 is the identity
#               tier", and the reference arm is a mutation rather than
#               `git show HEAD:` because a git-pinned before-copy dies
#               the day the change lands. The mutant is asked for bf16 as
#               well and must ALSO answer f32: a mutation that did not
#               apply would otherwise read as a passing identity arm.
#   accounting  for each narrow tier, the state is smaller by EXACTLY
#               nblocks * (rowbytes(f32) - rowbytes(tier)) summed over the
#               two rings - both numbers printed by the probe at this
#               model's head_dim and csa2_indexer_head_dim, so no stride,
#               no element count and no halving factor is written down
#               here. A hand-written "/2" would go silently wrong the day
#               indexer_head_dim moves to a tier of its own.
#   live        each narrow tier changes the logits. A tier that moved
#               bytes and no numbers would pass every arm above and do
#               nothing.
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
    local dir="$1" expr="${2:-}" marker="${3:-}" mfile="${4:-src/forward_csa2.c}"
    mkdir -p "$dir"
    cp -r "$ROOT/src" "$ROOT/common" "$dir/"
    cp "$ROOT/Makefile" "$dir/Makefile"
    if [ -n "$expr" ]; then
        # The file the expression is aimed at is a parameter: the two
        # mutation arms here reach different TUs (the tier selector lives
        # in cli_main.c, not in the ring's own file).
        sed -i "$expr" "$dir/$mfile"
        grep -q "$marker" "$dir/$mfile" || {
            echo "  FAIL: mutation did not apply to $mfile"; return 1; }
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

# The model's two row widths, so the probe prints the row-byte cost of
# each tier AT THESE WIDTHS and the accounting arm below never retypes it.
GG=$(geom "$MODEL" 32768); set -- $GG
PHD=$2; PIDX=$3

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
    if "$W/probe/p.exe" "$PHD" "$PIDX" > "$W/probe.out" 2>&1; then
        echo "  unit: $(tail -1 "$W/probe.out")"
        rowbytes_of() { sed -n "s/^rowbytes $1 $2 //p" "$W/probe.out" | head -1; }
    else
        echo "  FAIL unit arm:"; tail -4 "$W/probe.out"; FAILED=1
    fi
else
    echo "  FAIL: the probe did not build"; tail -3 "$W/probe/build.log"; FAILED=1
fi

# ---- the three integration arms ---------------------------------------
build "$W/ref" || exit 1
# M1: the tier selector always answers f32, so nothing downstream - not
# the allocation, not rowbytes, not row/put - ever sees a narrow format.
# The engine as it would be without the knob, reached from the one place
# the format enters.
M1='s/    if (strcmp(m, "f32") == 0)  return LZ_CRF_F32;/    if (1) return LZ_CRF_F32;/'
build "$W/nofmt" "$M1" 'if (1) return LZ_CRF_F32;' src/cli_main.c || exit 1

"$W/ref/llama98.exe" "$MODEL" --prompt "$PROMPT" --dump-prompt-tokens "$W/toks" -n 1 >/dev/null 2>&1
if [ ! -s "$W/toks" ]; then echo "  FAIL: no token list"; exit 1; fi

r=0
"$W/ref/llama98.exe"   "$MODEL" --tokens-file "$W/toks" --csa2-ring f32 --dump-all-logits "$W/f32.lg"  >/dev/null 2>&1 || r=1
"$W/nofmt/llama98.exe" "$MODEL" --tokens-file "$W/toks" --csa2-ring f32 --dump-all-logits "$W/nofmt.lg" >/dev/null 2>&1 || r=1
# The mutant's own control: with the selector pinned, asking it for bf16
# must give the f32 answer. Without this line a mutation that failed to
# apply would read as a passing identity arm.
"$W/nofmt/llama98.exe" "$MODEL" --tokens-file "$W/toks" --csa2-ring bf16 --dump-all-logits "$W/nofmt16.lg" >/dev/null 2>&1 || r=1
if [ "$r" -ne 0 ]; then
    echo "  FAIL: an arm exited non-zero"; FAILED=1
else
    if cmp -s "$W/f32.lg" "$W/nofmt.lg" && cmp -s "$W/f32.lg" "$W/nofmt16.lg"; then
        echo "  identity: f32 is bit-identical to the same tree with the tier selector pinned to f32"
    else
        echo "  FAIL: the f32 tier is not the identity tier"; FAILED=1
    fi
fi

# ---- live + accounting, one narrow tier at a time ---------------------
F32=$(state_bytes "$W/ref/llama98.exe" "$MODEL" f32)
if [ -z "$F32" ]; then
    echo "  FAIL: no state line for f32"; FAILED=1
else
    SLOTLINE=$("$W/ref/llama98.exe" "$MODEL" --prompt "$PROMPT" -n 1 --csa2-ring f32 2>/dev/null \
               | sed -n 's/^State.*(\([0-9]*\) B, *\([0-9]*\) slots.*/\2/p' | head -1)
    G=$(geom "$MODEL" "$SLOTLINE")
    set -- $G
    nblocks=$1
    for tier in bf16 q8 q16; do
        N=$(state_bytes "$W/ref/llama98.exe" "$MODEL" "$tier")
        "$W/ref/llama98.exe" "$MODEL" --tokens-file "$W/toks" --csa2-ring "$tier" \
            --dump-all-logits "$W/$tier.lg" >/dev/null 2>&1
        if [ -z "$N" ]; then
            echo "  FAIL: no state line for --csa2-ring $tier"; FAILED=1
        elif cmp -s "$W/f32.lg" "$W/$tier.lg"; then
            echo "  FAIL: $tier moved no number, so the tier is not read"; FAILED=1
        else
            # What the tier saves, straight from the two row-byte numbers
            # the probe printed at THIS model's widths - no stride, no
            # element count and no halving factor is written down here.
            want=$(( nblocks * ( $(rowbytes_of f32 "$PHD") - $(rowbytes_of "$tier" "$PHD") \
                               + $(rowbytes_of f32 "$PIDX") - $(rowbytes_of "$tier" "$PIDX") ) ))
            got=$(( F32 - N ))
            if [ "$got" -eq "$want" ]; then
                echo "  $tier: $got B smaller == $nblocks blocks x (rowbytes f32 - rowbytes $tier) over both rings, and the logits moved"
            else
                echo "  FAIL: $tier is $got B smaller, expected $want"; FAILED=1
            fi
        fi
    done
fi

rm -rf "$W"
if [ "$FAILED" -eq 0 ]; then echo "PASS csa2_ring_fmt_gate"; exit 0; fi
echo "FAIL csa2_ring_fmt_gate"
exit 1
