#!/bin/bash
# --bf16-store on/off must produce IDENTICAL logits, and must actually
# have done something.
#
#   bf16_store_gate.sh <model-dir> [tokens]
#
# THE FIRST HALF IS THE CONTRACT. A BF16 tensor kept at two bytes per
# element and widened on read is the same number as one expanded to f32
# while loading - `bits << 16` either way - so the two arms owe each
# other bit-identity, not closeness. That is what makes the switch safe
# to default ON, and it is the only reason it does.
#
# THE SECOND HALF IS WHY THIS FILE IS NOT A TAUTOLOGY, and it is the
# lesson of three separate blind spots found in this tree:
# a gate that passes without having looked at anything is worse than no
# gate, because it reports the property as checked. If the model holds
# no BF16 tensor, or the switch is wired up wrong, or the loader stops
# selecting the narrow format, then both arms run the SAME code and
# compare equal for a reason that has nothing to do with the contract.
# So this also asserts that the two arms DISAGREE about memory: the
# whole point of the format is that `on` allocates less, and if it does
# not, the comparison above proved nothing and the gate says so.
#
# Exit codes: 0 PASS, 1 FAIL, 2 SKIP (no binary, no model, or a model
# with no BF16 tensors - a skip means the check did not run).
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
if [ -z "$BIN" ]; then
    echo "bf16_store_gate: SKIP - no llama98 binary (build with 'make')" >&2
    exit 2
fi
if [ -z "$MODEL" ] || [ ! -d "$MODEL" ]; then
    echo "bf16_store_gate: SKIP - model dir not given or not found: '$MODEL'" >&2
    exit 2
fi

# The work directory sits beside the repo rather than in /tmp for the
# reason build/arm/parity_gate.sh gives: a Windows binary cannot resolve
# an MSYS /tmp path, and --dump-all-logits would silently write nowhere.
W="$(dirname "$ROOT")/.bf16-store-gate-$$"
mkdir -p "$W" || { echo "bf16_store_gate: FAIL - cannot create $W"; exit 1; }
trap 'rm -rf "$W"' EXIT

run_arm() {
    # $1 = "on" | "off"
    ( cd "$W" && "$BIN" "$MODEL" --bf16-store "$1" --tokens "$TOKENS" \
        --dump-all-logits "$1.bin" ) >"$W/$1.log" 2>&1
}

for arm in on off; do
    if ! run_arm "$arm"; then
        echo "bf16_store_gate: FAIL - the --bf16-store $arm run did not finish"
        tail -n 6 "$W/$arm.log" | sed 's/^/      /'
        exit 1
    fi
    if [ ! -s "$W/$arm.bin" ]; then
        echo "bf16_store_gate: FAIL - --bf16-store $arm wrote no logits"
        exit 1
    fi
done

# Did the model carry any BF16 at all? The binary reports the count, so
# this reads it rather than parsing safetensors here - one place knows
# the file format and it is not this script.
NBF=$(sed -n 's/^bf16 storage .*(\([0-9]*\) BF16 tensor(s).*/\1/p' "$W/on.log" | head -1)
if [ -z "$NBF" ]; then
    echo "bf16_store_gate: FAIL - the binary printed no 'bf16 storage' line;" \
         "it is older than this gate, or the report was removed" >&2
    exit 1
fi
if [ "$NBF" = "0" ]; then
    echo "bf16_store_gate: SKIP - $(basename "$MODEL") holds no BF16 tensor," \
         "so both arms run the same code and comparing them shows nothing" >&2
    exit 2
fi

# ---- the contract ---------------------------------------------------
if ! cmp -s "$W/on.bin" "$W/off.bin"; then
    NDIFF=$(cmp -l "$W/on.bin" "$W/off.bin" | wc -l)
    SZ=$(wc -c < "$W/on.bin")
    echo "bf16_store_gate: FAIL - the two arms disagree, $NDIFF of $SZ bytes"
    echo "      Widening a bf16 is 'bits << 16' whichever end it happens at,"
    echo "      so any difference here is a defect in the storage path, not"
    echo "      a precision trade. --bf16-store off is the old behaviour."
    exit 1
fi

# ---- and proof that the comparison meant something ------------------
# "Done  N.NN GB" is the loader's own accounting of what it allocated.
ON_GB=$(sed -n 's/^Done  *\([0-9.]*\) GB.*/\1/p' "$W/on.log" | head -1)
OFF_GB=$(sed -n 's/^Done  *\([0-9.]*\) GB.*/\1/p' "$W/off.log" | head -1)
if [ -z "$ON_GB" ] || [ -z "$OFF_GB" ]; then
    echo "bf16_store_gate: FAIL - could not read the loader's allocation" \
         "figure from both runs, so the no-op check could not run" >&2
    exit 1
fi
if [ "$ON_GB" = "$OFF_GB" ]; then
    echo "bf16_store_gate: FAIL - identical logits, but both arms allocated" \
         "$ON_GB GB"
    echo "      $NBF BF16 tensor(s) are in the file and the switch changed"
    echo "      nothing, so the comparison above compared a build with"
    echo "      itself. Either the loader stopped selecting LZ_FMT_BF16 or"
    echo "      --bf16-store is not reaching it. A passing byte comparison"
    echo "      is not evidence when both sides ran the same code."
    exit 1
fi

echo "bf16_store_gate: PASS (logits bit-identical over $(wc -c < "$W/on.bin")" \
     "bytes; $NBF BF16 tensor(s), $OFF_GB GB expanded vs $ON_GB GB kept narrow)"
exit 0
