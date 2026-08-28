#!/bin/bash
# A compressed model must load to the same logits as the uncompressed
# one, and the decompressor must reproduce the file byte for byte.
#
#   lz4_model_gate.sh <model-dir-with-model.bin> [tokens]
#
# TWO CHECKS, because they fail differently.
#
#   1. ROUND TRIP. Decompress with the LZ4D_TEST harness and cmp against
#      the original. This is the only DIRECT test the decompressor has.
#      The engine comparison below cannot replace it: a decompressor that
#      corrupts a few bytes of a quantised weight plane still produces
#      finite logits, and if those bytes land somewhere the arithmetic
#      smooths over, the run compares equal and the corruption ships.
#
#   2. THE ENGINE. Same model, compressed and not, same tokens, logits
#      compared bit for bit. This is what the feature promises.
#
# ONE COMPILER, TWO FILES - NOT two compilers. An earlier version of
# docs/lz4d.md claimed the Watcom build loading model.bin.lz4 matched the
# gcc build "byte for byte, cross-compiler and lz4 both". It never did:
# measured on _armgate2 the two differ (1536798181578518320 against
# 1708445672595413405) and always have, because x87 and SSE round
# differently - a difference this tree records elsewhere as known and
# accepted, and which has nothing to do with compression. Comparing
# across compilers here would either fail forever or, if someone
# "fixed" it with a tolerance, stop testing lz4 at all. The valid
# experiment holds the compiler fixed and varies only the file, which is
# what this does.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP (no binary, no model.bin, no python).
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
TOKENS=${2:-1,2,3,4,5,6,7,8}

BIN=${BIN:-}
if [ -z "$BIN" ]; then
    for c in "$ROOT/llama98.exe" "$ROOT/llama98"; do
        [ -x "$c" ] && BIN=$c && break
    done
fi
[ -z "$BIN" ] && { echo "lz4_model_gate: SKIP - no llama98 binary" >&2; exit 2; }
[ -z "$MODEL" ] || [ ! -f "$MODEL/model.bin" ] && {
    echo "lz4_model_gate: SKIP - no model.bin in '${MODEL:-<unset>}'" >&2; exit 2; }
PY=$(command -v python3 || command -v python) || {
    echo "lz4_model_gate: SKIP - no python for make_lz4.py" >&2; exit 2; }
CC=${CC:-gcc}
command -v "$CC" >/dev/null 2>&1 || {
    echo "lz4_model_gate: SKIP - no $CC to build the round-trip harness" >&2; exit 2; }

# Beside the repo, not /tmp: a Windows binary cannot resolve an MSYS
# path, and --dump-all-logits would write nowhere and be believed.
W="$(dirname "$ROOT")/.lz4-model-gate-$$"
mkdir -p "$W" || { echo "lz4_model_gate: FAIL - cannot create $W"; exit 1; }
trap 'rm -rf "$W"' EXIT

cp "$MODEL"/*.json "$W/" 2>/dev/null
cp "$MODEL/model.bin" "$W/model.bin" || {
    echo "lz4_model_gate: FAIL - cannot copy model.bin"; exit 1; }

if ! "$PY" "$ROOT/make_lz4.py" "$W/model.bin" >"$W/mk.log" 2>&1; then
    echo "lz4_model_gate: FAIL - make_lz4.py did not compress"
    tail -n 4 "$W/mk.log" | sed 's/^/      /'
    exit 1
fi
[ -s "$W/model.bin.lz4" ] || {
    echo "lz4_model_gate: FAIL - make_lz4.py produced no model.bin.lz4"; exit 1; }

# The compressed file must actually BE an LZ4 frame and must differ from
# the input; otherwise everything below compares a file with itself.
MAGIC=$(od -An -tx1 -N4 "$W/model.bin.lz4" | tr -d ' \n')
if [ "$MAGIC" != "04224d18" ]; then
    echo "lz4_model_gate: FAIL - model.bin.lz4 does not start with the LZ4" \
         "frame magic (got $MAGIC), so nothing below tests compression"
    exit 1
fi
if cmp -s "$W/model.bin" "$W/model.bin.lz4"; then
    echo "lz4_model_gate: FAIL - the 'compressed' file is byte-identical to" \
         "the input; the comparisons below would be vacuous"
    exit 1
fi

# ---- 1. round trip -------------------------------------------------
if ! $CC -O2 -std=c99 -I"$ROOT/src" -DLZ4D_TEST -o "$W/rt" \
        "$ROOT/src/lz4d.c" >"$W/cc.log" 2>&1; then
    echo "lz4_model_gate: FAIL - the LZ4D_TEST harness did not build"
    tail -n 4 "$W/cc.log" | sed 's/^/      /'
    exit 1
fi
if ! "$W/rt" "$W/model.bin.lz4" "$W/rt.bin" >"$W/rt.log" 2>&1; then
    echo "lz4_model_gate: FAIL - decompression reported an error"
    tail -n 3 "$W/rt.log" | sed 's/^/      /'
    exit 1
fi
if ! cmp -s "$W/model.bin" "$W/rt.bin"; then
    echo "lz4_model_gate: FAIL - the round trip is not byte-exact"
    cmp "$W/model.bin" "$W/rt.bin" 2>&1 | head -2 | sed 's/^/      /'
    exit 1
fi
RTSZ=$(wc -c < "$W/rt.bin")

# ---- 2. the engine, compressed vs not --------------------------------
# Two directories so the loader's own magic/sibling detection decides,
# rather than this script telling it which file to read.
mkdir -p "$W/plain" "$W/comp"
cp "$W"/*.json "$W/plain/" 2>/dev/null
cp "$W"/*.json "$W/comp/"  2>/dev/null
cp "$W/model.bin" "$W/plain/model.bin"
cp "$W/model.bin.lz4" "$W/comp/model.bin.lz4"

for arm in plain comp; do
    if ! ( cd "$W" && "$BIN" "$W/$arm" --tokens "$TOKENS" \
            --dump-all-logits "$arm.bin" ) >"$W/$arm.log" 2>&1; then
        echo "lz4_model_gate: FAIL - the $arm run did not finish"
        tail -n 6 "$W/$arm.log" | sed 's/^/      /'
        exit 1
    fi
    [ -s "$W/$arm.bin" ] || {
        echo "lz4_model_gate: FAIL - the $arm run wrote no logits"; exit 1; }
done

if ! cmp -s "$W/plain.bin" "$W/comp.bin"; then
    NDIFF=$(cmp -l "$W/plain.bin" "$W/comp.bin" | wc -l)
    echo "lz4_model_gate: FAIL - compressed and uncompressed disagree," \
         "$NDIFF of $(wc -c < "$W/plain.bin") bytes"
    echo "      The round trip above was byte-exact, so the decompressor is"
    echo "      not the suspect - look at the reader plumbing in model.c."
    exit 1
fi

echo "lz4_model_gate: PASS (round trip byte-exact over $RTSZ bytes;" \
     "compressed and uncompressed logits identical over" \
     "$(wc -c < "$W/plain.bin") bytes)"
exit 0
