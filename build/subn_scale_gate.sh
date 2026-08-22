#!/bin/sh
# --subn-scale: the knob is alive, it defaults to tok, and it is inert
# where use_subn is off.
#
# WHAT EACH ASSERTION BUYS
#   default-is-tok  `--subn-scale tok` must be byte-identical to passing
#                   nothing. Without it a later default flip would ship
#                   silently, since nothing else in the tree names the
#                   default.
#   knob-is-alive   `--subn-scale group` must DIFFER from tok on a model
#                   that has use_subn and a Hadamard block. A flag whose
#                   two settings agree is not a conservative flag, it is
#                   a flag that is not wired - the failure this repo has
#                   hit more than once (g_gdn_fixed, --shard).
#   correctly-scoped  On a model WITHOUT use_subn the knob must change
#                   nothing. This is the assertion the other two cannot
#                   make: they would both stay green if subn_scale_gs
#                   were wired into every quantize site in the engine.
#
# WHY THE MODEL MUST BE QUANTIZED
#   lz_matmul_xq_nt sends an F32 weight to lz_matmul with the FLOAT row,
#   and subn_fwht_quant writes only xq/xqs - so on an unquantized model
#   the Hadamard, and with it this knob, produces values nobody reads.
#   A gate pointed at a safetensors checkpoint passes while asserting
#   nothing. SUBN_MODEL must therefore be a quantized bin.
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP (no model to run on - nothing was checked).

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENG="$ROOT/llama98.exe"
[ -x "$ENG" ] || ENG="$ROOT/llama98"
W="${TMPDIR:-/tmp}/subn_scale_gate.$$"

SUBN_MODEL="${SUBN_MODEL:-}"      # quantized bin, use_subn + hadamard_o > 1
PLAIN_MODEL="${PLAIN_MODEL:-}"    # any model without use_subn
TOKENS="${SUBN_TOKENS:-}"

miss=
[ -x "$ENG" ] || miss="$miss engine"
[ -n "$SUBN_MODEL" ] && [ -e "$SUBN_MODEL" ] || miss="$miss SUBN_MODEL"
[ -n "$TOKENS" ] && [ -f "$TOKENS" ] || miss="$miss SUBN_TOKENS"
if [ -n "$miss" ]; then
    echo "subn-scale-gate: SKIP - missing:$miss"
    echo "  SUBN_MODEL must be a QUANTIZED bin with use_subn and"
    echo "  hadamard_o > 1; an f32 checkpoint makes every assertion vacuous."
    exit 2
fi

mkdir -p "$W" || exit 1
FAIL=0
bad() { echo "subn-scale-gate: FAIL $*"; FAIL=1; }

run() {  # run MODEL OUT [extra args...]
    m=$1; out=$2; shift 2
    "$ENG" "$m" --tokens-file "$TOKENS" --dump-logits "$out" "$@" >"$out.log" 2>&1
}

echo "-- subn model: $SUBN_MODEL --"
run "$SUBN_MODEL" "$W/def.bin"                        || bad "default arm did not run"
run "$SUBN_MODEL" "$W/tok.bin" --subn-scale tok       || bad "tok arm did not run"
run "$SUBN_MODEL" "$W/grp.bin" --subn-scale group     || bad "group arm did not run"
[ "$FAIL" -eq 0 ] || { rm -rf "$W"; exit 1; }

# A constant or empty dump would make every cmp below meaningless.
sz=$(wc -c < "$W/def.bin")
if [ "$sz" -lt 64 ]; then
    bad "the default dump is $sz bytes - nothing was computed"
    rm -rf "$W"; exit 1
fi

if cmp -s "$W/def.bin" "$W/tok.bin"; then
    echo "  PASS  default-is-tok      ($sz bytes identical)"
else
    bad "default-is-tok: --subn-scale tok differs from passing nothing"
fi

if cmp -s "$W/tok.bin" "$W/grp.bin"; then
    bad "knob-is-alive: group and tok agree byte for byte. Either the" \
        "model has no use_subn/hadamard_o, or subn_scale_gs is returning" \
        "0 (lz_act_gs gives 0 for an F32 weight)."
else
    n=$(cmp -l "$W/tok.bin" "$W/grp.bin" | wc -l)
    echo "  PASS  knob-is-alive       ($n/$sz bytes differ)"
fi

if [ -n "$PLAIN_MODEL" ] && [ -e "$PLAIN_MODEL" ]; then
    echo "-- plain model: $PLAIN_MODEL --"
    run "$PLAIN_MODEL" "$W/p_def.bin"                    || bad "plain default did not run"
    run "$PLAIN_MODEL" "$W/p_grp.bin" --subn-scale group || bad "plain group did not run"
    if cmp -s "$W/p_def.bin" "$W/p_grp.bin"; then
        echo "  PASS  correctly-scoped    (no use_subn -> knob inert)"
    else
        bad "correctly-scoped: the knob changed a model that has no" \
            "use_subn - it has leaked outside the SubLN path"
    fi
else
    echo "  SKIP  correctly-scoped    (set PLAIN_MODEL to check the scope)"
fi

rm -rf "$W"
[ "$FAIL" -eq 0 ] || { echo "subn_scale_gate: FAIL"; exit 1; }
echo "subn_scale_gate: PASS"
exit 0
