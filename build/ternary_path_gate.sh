#!/bin/bash
# A ternary tower loaded from safetensors is a DIFFERENT FUNCTION.
#
# WHY THIS EXISTS. A dense kunmoe tower's BitLinear weights are stored as
# an fp32 LATENT in a checkpoint. The training forward ternarizes them
# (engine/bitlinear.py weight_quant_ternary) and so does the exporter,
# which is why a model.bin holds T2 codes. THE ENGINE NEVER QUANTIZES A
# WEIGHT - it runs whatever the file holds - so the safetensors path of
# that tower runs the same architecture on weights that were never pushed
# onto the ternary grid.
#
# Nothing about it fails, and that is the whole problem. Measured on
# kunkun-ce dpo_r2, 8191 tokens, mean NLL: 4.9931 from safetensors against
# 3.1992 for the same weights through the exporter's bin. HF with
# weight_quant_ternary patched to identity reads 5.0044 on the same
# tokens, so the entire gap is the missing ternarization - and its size
# varies per model with how far that latent sits from the grid (1.755,
# 1.415 and 0.528 nats/token on three checkpoints here). A number that
# moves by model is exactly the shape that gets read as a verdict on the
# checkpoint, which is how it was read until the mechanism was isolated.
#
# The fix is the tree's own precedent for this class: when the CSA2
# indexer turned out never to have been trained, the answer was to say so
# out loud rather than to leave it to be inferred from the numbers.
# cli_main prints what the run is about to compute.
#
# WHAT THE PREDICATE IS, and why each clause is watched. The line fires
# on `have_st && cfg->num_experts == 0 && cfg->kda_gate_rank > 0`:
#   have_st          - the weights came from model.safetensors. A bin load
#                      runs the trained function and must stay quiet.
#   num_experts == 0 - the engine's existing name for the ternary tower
#                      (forward_kda.c's subn_scale_gs and resolve_hadamard
#                      use the same one). An MoE tower's projections are
#                      plain linears and were never ternarized.
#   kda_gate_rank>0  - scopes it to kunmoe. A GDN tower also has
#                      num_experts == 0 and has no BitLinear anywhere, so
#                      num_experts ALONE would tell it it was running
#                      untrained ternary weights.
# Each of the three gets its own mutation below, on a private copy of
# src/. A clause with no arm watching it is a clause that can be dropped
# in a later edit with every test still green.
#
# Exit codes: 0 pass, 1 fail, 2 skip.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODELS=${LZ_MODELS:-E:/LLM/models}
CKPT=${LZ_CKPT:-E:/LLM/ckpt}
W=$(mktemp -d)
FAILED=0
trap 'rm -rf "$W"' EXIT

LINE="ternary        OFF"

skip() { echo "ternary_path_gate: SKIP - $1"; exit 2; }

# ckpt/rl_r2 carries BOTH files, so the first two arms differ in exactly
# one thing: whether a model.bin sits beside the safetensors.
BOTH="$CKPT/rl_r2"
MOE="$MODELS/kunkun98"
GDN="$MODELS/Qwen3.5-0.8B"

[ -x "$ROOT/llama98.exe" ] || skip "no engine at $ROOT/llama98.exe"
[ -f "$BOTH/model.bin" ] || skip "no model.bin at $BOTH"
[ -f "$BOTH/model.safetensors" ] || skip "no model.safetensors at $BOTH"
[ -f "$MOE/model.safetensors" ] || skip "no MoE checkpoint at $MOE"
[ -f "$GDN/config.json" ] || skip "no GDN checkpoint at $GDN"

ST="$W/stonly"
mkdir -p "$ST"
cp "$BOTH/config.json" "$BOTH/model.safetensors" "$ST/"

# Runs --check and answers yes/no for the line. A load failure answers
# "no" and would PASS the negative arms for the wrong reason, so the
# caller checks the arm loaded at all - see the Structure OK probe below.
says() { "$ROOT/llama98.exe" "$1" --check 2>&1; }

report() {
    local label="$1" dir="$2" want="$3" out
    out=$(says "$dir")
    if ! printf '%s' "$out" | grep -q "Structure OK"; then
        echo "  FAIL: $label did not load - the arm's premise is gone"
        printf '%s\n' "$out" | tail -2 | sed 's/^/        /'
        FAILED=1
        return
    fi
    local got=no
    printf '%s' "$out" | grep -qF "$LINE" && got=yes
    if [ "$got" = "$want" ]; then
        echo "  ok  $label: says '$got'"
    else
        echo "  FAIL: $label says '$got', wanted '$want'"
        FAILED=1
    fi
}

echo "which arms announce the un-QAT model:"
report "safetensors, no bin      " "$ST"   yes
report "safetensors + model.bin  " "$BOTH" no
report "MoE tower (num_experts>0)" "$MOE"  no
report "GDN tower (not kunmoe)   " "$GDN"  no

# ---- mutations -------------------------------------------------------
# Each on a private copy of src/ (never the live tree), and each verified
# to have matched before it is trusted - a sed that silently found nothing
# would build the unmutated engine and this gate would report a green
# mutation.
PRED='have_st && cfg->num_experts == 0 && cfg->kda_gate_rank > 0'
# sed reads `&` in the REPLACEMENT as "the whole match", so every one in a
# mutant predicate has to be escaped. (In the pattern it is literal.)
esc() { printf '%s' "$1" | sed 's/&/\\&/g'; }

mutant() {
    local tag="$1" newpred="$2"
    local d="$W/$tag"
    MUTBIN=""
    mkdir -p "$d"
    cp -r "$ROOT/src" "$ROOT/common" "$d/"
    cp "$ROOT/Makefile" "$d/Makefile"
    sed -i "s|$PRED|$(esc "$newpred")|" "$d/src/cli_main.c"
    if ! grep -qF "$newpred" "$d/src/cli_main.c"; then
        echo "  FAIL: mutation '$tag' did not match cli_main.c - this gate no longer matches the source"
        return 1
    fi
    if ! ( cd "$d" && make native ) >"$d/build.log" 2>&1; then
        echo "  FAIL: mutant '$tag' failed to build"; tail -5 "$d/build.log"
        return 1
    fi
    MUTBIN="$d/llama98.exe"
    return 0
}

# Each mutation removes ONE clause. The arm it must redden is the one
# that clause exists for, and the bin arm must NOT move - a mutation that
# reddens everything proves the gate is watching the line, not the
# clause.
mut_arm() {
    local tag="$1" newpred="$2" arm="$3" label="$4"
    if ! mutant "$tag" "$newpred"; then
        FAILED=1
        return
    fi
    local out got=no
    out=$("$MUTBIN" "$arm" --check 2>&1)
    printf '%s' "$out" | grep -qF "$LINE" && got=yes
    if [ "$got" = yes ]; then
        echo "  mutation $tag ($label): red as it should be - that arm now announces it"
    else
        echo "  FAIL: mutation $tag ($label) was NOT caught - that clause has no arm watching it"
        FAILED=1
    fi
    if [ "$arm" != "$BOTH" ]; then
        out=$("$MUTBIN" "$BOTH" --check 2>&1)
        if printf '%s' "$out" | grep -qF "$LINE"; then
            echo "  FAIL: mutation $tag also moved the bin arm - it removed more than its clause"
            FAILED=1
        fi
    fi
}

mut_arm m1 'cfg->num_experts == 0 && cfg->kda_gate_rank > 0' \
        "$BOTH" "the bin arm stops being distinguished"
mut_arm m2 'have_st && cfg->kda_gate_rank > 0' \
        "$MOE" "an MoE tower is called a ternary one"
mut_arm m3 'have_st && cfg->num_experts == 0' \
        "$GDN" "a GDN tower is called a ternary one"

if [ "$FAILED" = 0 ]; then
    echo "ternary_path_gate: PASS - the line fires on the safetensors path of" \
         "a dense kunmoe tower, stays silent on a bin, on an MoE tower and on" \
         "a GDN tower, and each of the predicate's three clauses has a" \
         "mutation that reddens when it is dropped"
fi
exit "$FAILED"
