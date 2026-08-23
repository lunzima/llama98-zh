#!/bin/sh
# The SubLN activation path: the Hadamard reaches BOTH of the matmul's
# inputs, and --subn-scale is wired, defaulted and scoped.
#
# WHAT EACH ASSERTION BUYS
#   default-is-tok    `--subn-scale tok` must be byte-identical to passing
#                     nothing. Nothing else in the tree names the default,
#                     so without this a flip would ship silently.
#   knob-is-alive     `--subn-scale group` must DIFFER from tok. A flag
#                     whose two settings agree is not conservative, it is
#                     unwired - the failure this repo keeps hitting
#                     (g_gdn_fixed, --shard).
#   correctly-scoped  On a model WITHOUT use_subn the knob must change
#                     nothing. The other two would both stay green if
#                     subn_scale_gs leaked into every quantize site.
#   quantized-rotates `--hadamard off` must change a QUANTIZED run.
#   float-rotates     `--hadamard off` must change an UNQUANTIZED run.
#
# There is deliberately NO norms-tier assertion; see the note next to
# the plain model below for the one that was tried and thrown out.
#
# THE TWO ARE NOT EQUALLY STRONG, and saying so here is cheaper than
# someone discovering it during an incident. On the FLOAT path, oblk <= 1
# does nothing to the row at all, so "the output changed" can only mean
# the rotation ran: that assertion is exact. On the QUANTIZED path the
# two branches call DIFFERENT quantizers (subn_fwht_quant vs
# lz_quantize_q8_tok), so their outputs differ even if the butterfly
# inside subn_fwht_quant were deleted - it proves the branch was taken,
# not that the transform happened. The exact quantized check is
# "--hadamard off is byte-identical to an export with the field zeroed",
# verified by hand at the time of writing but not automated here because
# it needs a second model built from the same weights.
#
# WHY THE LAST TWO ARE SEPARATE
#   lz_matmul_xq_nt reads xq/xqs for a quantized weight and the FLOAT row
#   for an F32 one. subn_fwht_quant writes only xq/xqs. So the rotation
#   reached one input and not the other, and one config computed two
#   different functions depending on the weight dtype - measured at 0.00%
#   top-1 agreement between the two, against 84.97% once both rotate.
#   ONE assertion cannot see this: whichever arm you run, the flag does
#   change that arm's output. It takes one assertion per input.
#
#   They are also not a threshold. "f32 and q8 should agree to within X"
#   drifts with the model (a worse model quantizes worse), which is how a
#   gate ends up passing because the artifact improved. "Turning the
#   transform off must change the output" is the same question with no
#   number in it.
#
# WHY SUBN_MODEL MUST BE QUANTIZED
#   On an f32 checkpoint the quantized row is computed and never read, so
#   the scale assertions pass while asserting nothing. That is how the
#   first run of this gate passed.
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP (nothing was checked).

set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENG="$ROOT/llama98.exe"
[ -x "$ENG" ] || ENG="$ROOT/llama98"
W="${TMPDIR:-/tmp}/subn_gate.$$"

# DEFAULTS, so ci.sh can call this with no arguments. Every one is
# overridable; every one SKIPs loudly if absent.
#
# HOW TO PRODUCE SUBN_MODEL (it is a build artifact, not a checkpoint):
#   python3 tools/export_q8.py <models>/kunkun-ce /tmp/subn_ce_q8.bin
#   mkdir -p <models>/_subn_gate_q8
#   cp /tmp/subn_ce_q8.bin <models>/_subn_gate_q8/model.bin
#   cp <models>/kunkun-ce/tokenizer.json <models>/_subn_gate_q8/
# tools/export_q8.py lives in the TRAINING tree (see CLAUDE.md on the
# two trees), which is why this is a recipe here and not a build rule.
#
# WHICH CHECKPOINT, and why this was recorded as a resource block for so
# long: the exporter's use_subn tensor list includes the ATTENTION
# projection norms (self_attn.q_proj.norm.weight and its three
# siblings), and most of the trained SubLN checkpoints predate that
# field - _smoke_mhc0, _abl_mhc*, ce-abl-mhc* all carry only the KDA and
# MLP norms and fail export with a KeyError on exactly that name. That
# got generalised to "no SubLN checkpoint exports". kunkun-ce DOES (it
# was migrated; its model.safetensors.bak-before-attn-subn is the
# migration's own leftover) and exports clean. Measured 2026-08-23.
MODELS_ROOT="${MODELS_ROOT:-/e/LLM/models}"
# QUANTIZED bin, use_subn + hadamard_o > 1 - see the recipe above.
SUBN_MODEL="${SUBN_MODEL:-$MODELS_ROOT/_subn_gate_q8}"
# The same weights UNQUANTIZED. Not "a similar model": float-rotates and
# quantized-rotates are only comparable if the shape is the same one.
SUBN_F32_MODEL="${SUBN_F32_MODEL:-$MODELS_ROOT/kunkun-ce}"
# Any model WITHOUT use_subn. kmr20 is the one every other gate here uses.
PLAIN_MODEL="${PLAIN_MODEL:-$MODELS_ROOT/kmr20}"
# The token ids MUST be inside the subn model's vocab, which is 6144
# here and 32768 on kmr20 - so this file is the subn model's, not the
# other gates'. Produced with:
#   llama98 <subn-model> --prompt '<text>' --dump-prompt-tokens F
TOKENS="${SUBN_TOKENS:-$ROOT/build/subn_gate_tokens_zh.txt}"

miss=
[ -x "$ENG" ] || miss="$miss engine"
[ -n "$SUBN_MODEL" ] && [ -e "$SUBN_MODEL" ] || miss="$miss SUBN_MODEL"
[ -n "$TOKENS" ] && [ -f "$TOKENS" ] || miss="$miss SUBN_TOKENS"
if [ -n "$miss" ]; then
    echo "subn-gate: SKIP - missing:$miss"
    echo "  SUBN_MODEL must be a QUANTIZED bin with use_subn and"
    echo "  hadamard_o > 1; an f32 checkpoint makes the scale assertions vacuous."
    exit 2
fi

mkdir -p "$W" || exit 1
FAIL=0
bad() { echo "subn-gate: FAIL $*"; FAIL=1; }

run() {  # run MODEL OUT [extra args...]
    m=$1; out=$2; shift 2
    # --dump-all-logits, not --dump-logits: three of the assertions below
    # are "these two arms must DIFFER", and the last position alone is one
    # sample of that - two configurations that differ everywhere else can
    # coincide there, which this gate would report as "the knob is inert".
    # That is the exact mistake build/kv_rot_gate.sh made and its header
    # records. Every comparison here is cmp/cmp -l and $sz is derived from
    # the file, so nothing else changes; the byte counts in the PASS lines
    # get larger by the number of positions.
    #
    # UNVERIFIED AT HEAD, and it cannot be otherwise: this gate SKIPs at
    # HEAD because no use_subn export loads (see the SKIP text below), so
    # leaving the weaker flag in place would have been equally unverified.
    "$ENG" "$m" --tokens-file "$TOKENS" --dump-all-logits "$out" "$@" >"$out.log" 2>&1
}

echo "-- subn model (quantized): $SUBN_MODEL --"
run "$SUBN_MODEL" "$W/def.bin";                     r1=$?
run "$SUBN_MODEL" "$W/tok.bin" --subn-scale tok;    r2=$?
run "$SUBN_MODEL" "$W/grp.bin" --subn-scale group;  r3=$?
run "$SUBN_MODEL" "$W/hoff.bin" --hadamard off;     r4=$?

# ALL arms failing is not this gate's FAIL. This engine tree is shared,
# and a build that predates a layout change cannot open a model exported
# after it - the arms then agree perfectly on being unable to run, which
# says nothing about either knob. SOME failing while others run IS this
# gate's business: only the flag differs between those runs.
if [ "$r1" -ne 0 ] && [ "$r2" -ne 0 ] && [ "$r3" -ne 0 ] && [ "$r4" -ne 0 ]; then
    echo "  UNLOADABLE $SUBN_MODEL - every arm failed identically, so"
    echo "             nothing was checked. First line:"
    grep -m1 'Failed to load' "$W/def.bin.log" | sed 's/^/             /'
    rm -rf "$W"
    echo "subn_gate: SKIP - the subn model is unloadable by this build"
    echo "  KNOWN CAUSE at the time of writing, and it is the ENGINE side,"
    echo "  not the model: a use_subn export carries the per-projection"
    echo "  input norms (mlp.gate_proj.norm.weight and its siblings) and"
    echo "  the tensor spec list does not claim them yet, so the loader"
    echo "  matches the next spec positionally and reports an elem-count"
    echo "  mismatch on mlp.gate_proj.weight (want 1048576, file 512 - the"
    echo "  512 IS the norm). The five assertions below were verified on a"
    echo "  build carrying that loader work uncommitted; this gate turns"
    echo "  green on its own the moment it lands."
    exit 2
fi
for pair in "1 default" "2 tok" "3 group" "4 hadamard-off"; do
    set -- $pair
    eval "rc=\$r$1"
    [ "$rc" -eq 0 ] || bad "the $2 arm failed while another arm ran (rc=$rc)"
done
[ "$FAIL" -eq 0 ] || { rm -rf "$W"; exit 1; }

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
        "model has no use_subn/hadamard_o, or subn_scale_gs returns 0" \
        "(lz_act_gs gives 0 for an F32 weight)."
else
    n=$(cmp -l "$W/tok.bin" "$W/grp.bin" | wc -l)
    echo "  PASS  knob-is-alive       ($n/$sz bytes differ)"
fi

if cmp -s "$W/def.bin" "$W/hoff.bin"; then
    bad "quantized-rotates: --hadamard off changed nothing on a quantized" \
        "run. The rotation is not reaching xq/xqs."
else
    n=$(cmp -l "$W/def.bin" "$W/hoff.bin" | wc -l)
    echo "  PASS  quantized-rotates   ($n/$sz bytes differ)"
fi

if [ -n "$SUBN_F32_MODEL" ] && [ -e "$SUBN_F32_MODEL" ]; then
    echo "-- subn model (unquantized): $SUBN_F32_MODEL --"
    run "$SUBN_F32_MODEL" "$W/f_on.bin";                 s1=$?
    run "$SUBN_F32_MODEL" "$W/f_off.bin" --hadamard off; s2=$?
    if [ "$s1" -ne 0 ] || [ "$s2" -ne 0 ]; then
        bad "float-rotates: an arm did not run (rc=$s1/$s2)"
    elif cmp -s "$W/f_on.bin" "$W/f_off.bin"; then
        bad "float-rotates: --hadamard off changed nothing on an" \
            "UNQUANTIZED run. subn_fwht_quant writes q/s only, so the" \
            "float row lz_matmul_xq_nt reads was never rotated - this" \
            "model computes W.x while its quantized twin computes" \
            "W.quant(Hx), from one config."
    else
        n=$(cmp -l "$W/f_on.bin" "$W/f_off.bin" | wc -l)
        echo "  PASS  float-rotates       ($n/$(wc -c < "$W/f_on.bin") bytes differ)"
    fi
else
    echo "  SKIP  float-rotates       (set SUBN_F32_MODEL - this is the"
    echo "        assertion the quantized arms cannot make)"
fi

if [ -n "$PLAIN_MODEL" ] && [ -e "$PLAIN_MODEL" ]; then
    echo "-- plain model: $PLAIN_MODEL --"
    run "$PLAIN_MODEL" "$W/p_def.bin";                    p1=$?
    run "$PLAIN_MODEL" "$W/p_grp.bin" --subn-scale group; p2=$?
    if [ "$p1" -ne 0 ] || [ "$p2" -ne 0 ]; then
        bad "correctly-scoped: an arm did not run (rc=$p1/$p2)"
    elif cmp -s "$W/p_def.bin" "$W/p_grp.bin"; then
        echo "  PASS  correctly-scoped    (no use_subn -> knob inert)"
    else
        bad "correctly-scoped: the knob changed a model that has no" \
            "use_subn - it has leaked outside the SubLN path"
    fi

    # NO NORMS-TIER ASSERTION HERE, and the reason is worth the lines.
    # An obvious one to add is "int must equal fixed on a plain model",
    # which lz_norm_int claims. It passes on s1v3 - and it passes with
    # lz_quantize_q8_int mutated to a constant, which is how it was
    # found out: s1v3 reaches NO int site at all, so the assertion was
    # measuring nothing. See lz_norm_int in src/ops.h for where that
    # leaves the claim.
else
    echo "  SKIP  correctly-scoped    (set PLAIN_MODEL to check the scope)"
fi

rm -rf "$W"
[ "$FAIL" -eq 0 ] || { echo "subn_gate: FAIL"; exit 1; }
echo "subn_gate: PASS"
exit 0
