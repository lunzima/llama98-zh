#!/bin/bash
# Learned per-head attention sink (config.json "attn_sink" /
# "attn_sink_init"). model.h's own LZModelConfig attn_sink_learned
# comment has the mechanism, LZLayer's attn_sinks comment has the
# per-layer rule, and the training side's
# engine/kunmoe_modeling.py::_sink_eager_attention_forward is the
# semantics this is a port of: one extra logit column per head,
# concatenated to the query-position logits BEFORE softmax and dropped
# AFTER it, with the surviving weights therefore summing to 1-p_sink
# and NOT renormalised.
#
#   attn_sink_learned_gate.sh <model-dir> [prompt]
#
# THE FIXTURE IS BUILT, NOT FOUND, same posture as kv_share_gate.sh.
# Two reasons, both measured:
#   - kunkun98/kunkun-ce are the only checkpoints here that set the key,
#     and they also carry gated_residual tensors this engine still has
#     no spec for, so they do not load at all. Blaming that on the sink
#     would make this gate red for another task's gap.
#   - The switch is a config key, so the SAME weights can be run with it
#     off, on, and perturbed - which is what the arms below need.
# GR is therefore stripped from every fixture (the keys and the
# gr_attn/gr_mlp tensors). That is out of this gate's scope and it is
# unconditional, so a fixture built before Task 3 and one built after it
# run the same arms.
#
# WHAT EACH ARM PROVES
#
#   off        The same checkpoint with the key absent and the sinks
#              tensor REMOVED - a checkpoint that never carried the
#              switch. (Leaving the tensor in place while clearing the
#              key is refused by the loader's leak check, and correctly
#              so: an unclaimed tensor means an unimplemented module.)
#              It must load, and it is what every other arm is measured
#              against.
#   on         The key present and the checkpoint's own sinks tensor.
#              Must load, and must carry exactly n_heads more parameters
#              than `off` - the count is how the gate knows the tensor is
#              claimed and not merely tolerated.
#
#   THE ON ARM ALONE PROVES ALMOST NOTHING, AND THAT IS MEASURED, NOT
#   ASSUMED. kunkun98's sinks are still exactly attn_sink_init (-30.0) on
#   every head, and -30 is far below the logit scale, so the sink column
#   contributes exp(-30 - max) to the softmax denominator - under the
#   float32 ULP of the sum. `on` and `off` therefore produce BITWISE
#   IDENTICAL prefill logits on this checkpoint, and would do so on an
#   engine that never read the tensor at all. The gate reports that
#   number and deliberately does NOT assert on it.
#
#   So the sensitivity arms carry the weight, and they perturb the TENSOR
#   rather than the config. Measured response of this checkpoint to a
#   uniform offset added to every sink (max |dlogit| at the prefill's
#   last position, and whether the first greedy token moves):
#
#       offset   sink becomes   max |dlogit|   first token moves
#        +1..+10   -29..-20        0.0        no (bitwise identical)
#        +20          -10        5.7e-06      no
#        +30            0        4.9e-03      no
#        +40          +10          2.84       YES (argmax 15183 -> 17785)
#
#   The dead band is a property of the CHECKPOINT's sink value, not of
#   the mechanism: an offset smaller than |attn_sink_init| leaves the
#   column sub-ULP, so no arm below may use one. Both sensitivity arms
#   are additive offsets large enough to clear whatever the file holds,
#   and they are two different instruments on purpose:
#
#   sens       sinks += 30. The BIT-LEVEL criterion: the prefill logits
#              must stop being byte-identical to `on`'s. This is the
#              tightest instrument the engine has - it fires at offset
#              +20, long before anything moves a token - and it is the
#              arm that catches an engine which loads the tensor and then
#              does not read it.
#   text       sinks += 40. The TEXT criterion: the greedy continuation
#              must change. Blunter (it needs the sink to reach the logit
#              scale before any argmax flips), and it is here because it
#              is the criterion a reader can eyeball.
#
#   misplaced  The same sinks tensor on the checkpoint's own layer AND on
#              a full_attention layer that csa2_layers names. That
#              layer's attn_sink_on is 0 by _wire_attn_sink's rule, so
#              the extra tensor must be refused as unclaimed. Without
#              this arm the per-layer rule is only a comment: `on` would
#              be equally green if the engine claimed a sink on every
#              full_attention layer, or on every layer.
#
#   bf16       The same tensor in the storage form the engine's own
#              exporter picks for it: its per-tensor rule is
#              `len(shape) != 2 or n < 512 -> FMT_BF16`, and (n_heads)
#              is 1-D. The engine decodes those two bytes in TWO places -
#              the loader's widen-at-load (visit_read, when --bf16-store
#              is off) and attn_sink_at, which decodes in place when the
#              tensor stays narrow - and those two must agree exactly.
#              The claim is an equality between the two decode sites, so
#              it holds for any value; the fixture uses +TEXT_DELTA
#              anyway, because bf16 0.0 is sixteen zero bits and is
#              therefore invariant under every byte-level mutation of the
#              decoder, which is exactly what M3 applies.
#
# MUTATIONS, all three of which must redden, each aimed at a different
# half of the wiring - because the halves fail independently:
#
#   M1  setup_layers' derivation pinned to 0. The load itself must fail:
#       the sinks tensor goes unclaimed. This is the mutation that
#       catches "the flag is never set".
#   M2  the flag is still set and the tensor is still claimed (so `on`
#       loads and the parameter count still matches), but the forward
#       passes NULL instead of the tensor. `on` stays GREEN under M2 -
#       only `sens` reddens. That is the whole reason the sensitivity
#       arms are not optional, and it is measured here rather than
#       argued.
#   M3  attn_sink_at's bf16 decode shifted one byte-position wrong. The
#       bf16 arm is an equality between two decode sites, which would
#       also read green on an engine where the in-place branch is never
#       taken at all; M3 is what separates "the two agree" from "only one
#       of them ran".
#
# Exit: 0 pass, 1 fail, 2 skip (no binary / no model dir / no sinks
# tensor to perturb / no python3 with numpy+safetensors).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
PROMPT=${2:-"人工智能的发展历史可以追溯到二十世纪五十年代，当时的科学家们开始思考机器是否能够模拟人类的思维过程。这是一段用于测试的更长提示词，包含更多汉字以确保提示长度超过窗口大小。"}
SENS_DELTA=30
TEXT_DELTA=40
NGEN=32

BIN=$ROOT/llama98.exe
[ -x "$BIN" ] || { echo "attn_sink_learned_gate: SKIP - no $BIN (run: make native)"; exit 2; }
[ -n "$MODEL" ] && [ -d "$MODEL" ] || {
    echo "attn_sink_learned_gate: SKIP - model dir not given or not found: '$MODEL'"; exit 2; }
[ -f "$MODEL/config.json" ] || {
    echo "attn_sink_learned_gate: SKIP - no config.json under '$MODEL'"; exit 2; }
python3 -c 'import numpy, safetensors' 2>/dev/null || {
    echo "attn_sink_learned_gate: SKIP - needs python3 with numpy and safetensors"; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
FAILED=0

# ---- fixtures --------------------------------------------------------
# One python pass writes all five, because the decisions they share
# (which layer is a sink layer, how many heads, which tensors are GR)
# have to be made once or they drift between arms.
python3 - "$MODEL" "$W" "$SENS_DELTA" "$TEXT_DELTA" <<'PY'
import json, os, shutil, sys

model_dir, w, sens_delta, text_delta = (sys.argv[1], sys.argv[2],
                                        float(sys.argv[3]), float(sys.argv[4]))

def bail(msg):
    print("SKIP: " + msg)
    sys.exit(2)

try:
    import numpy as np
    from safetensors.numpy import load_file, save_file
except Exception as exc:
    bail("python3 numpy/safetensors unavailable (%s)" % exc)

cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
tc = cfg.get("text_config", cfg)
lt = tc.get("layer_types") or []
n = tc.get("num_hidden_layers") or len(lt)
if not lt:
    bail("config.json has no layer_types")
csa2 = sorted(set(tc.get("csa2_layers") or ()))
n_heads = tc["num_attention_heads"]
init = float(tc.get("attn_sink_init", 0.0))

tensors = load_file(os.path.join(model_dir, "model.safetensors"))
prefix = "model." if ("model.layers.0.input_layernorm.weight" in tensors) else ""

def lname(i, suffix):
    return "%slayers.%d.%s" % (prefix, i, suffix)

# The training side's own rule (_wire_attn_sink), evaluated from the
# config so that the CHECKPOINT is the oracle for it: a sink tensor on
# any other layer means the two sides disagree about where a sink lives.
expect = [i for i, t in enumerate(lt) if t == "full_attention" and i not in csa2]
have = sorted(i for i in range(n) if lname(i, "self_attn.sinks") in tensors)
if not have:
    bail("'%s' carries no self_attn.sinks tensor - the switch is off here" % model_dir)
if have != expect:
    print("FAIL: sinks tensors sit on layers %s, but _wire_attn_sink's rule"
          " (full_attention minus csa2_layers %s) gives %s"
          % (have, csa2, expect))
    sys.exit(1)
if not csa2:
    bail("no csa2_layers, so the misplaced arm has no layer to move a"
         " sink onto")

sink_names = [lname(i, "self_attn.sinks") for i in have]
vals = [tensors[k] for k in sink_names]
inert = all(v.size and (v == init).all() for v in vals)
print("  fixture: sink layer(s) %s, %d head(s) each, attn_sink_init=%g"
      % (have, int(vals[0].size), init))
if inert:
    print("  fixture: every sink on disk is exactly attn_sink_init - the"
          " column is sub-ULP, so `on` is expected to match `off` bitwise")
else:
    print("  fixture: sinks on disk range %.6g..%.6g, so `on` is NOT inert"
          % (float(min(v.min() for v in vals)), float(max(v.max() for v in vals))))
print("  fixture: misplaced arm -> layer %d (csa2_layers)" % csa2[0])

GR_KEYS = ("gated_residual", "ced_split", "ced_replay_w")
SKIP_FILES = ("model.safetensors", "model.bin", "model.bin.lz4")
flat = np.concatenate([v.reshape(-1) for v in vals])

def emit(tag, *, sinks, switch, bf16_sinks, extra_layer=None):
    """One fixture dir. `sinks` is the sink value written to every layer
    the rule names (None = the tensor is absent), `switch` the config
    key, `extra_layer` additionally plants one on a layer the rule does
    NOT name (the misplaced arm)."""
    out = os.path.join(w, tag)
    os.makedirs(out, exist_ok=True)
    for name in os.listdir(model_dir):
        if name in SKIP_FILES or "safetensors" in name:
            continue
        src = os.path.join(model_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out, name))
    c = json.loads(json.dumps(cfg))
    ctc = c.get("text_config") or c
    for k in GR_KEYS:
        ctc.pop(k, None)
    for k in ("attn_sink", "attn_sink_init"):
        ctc.pop(k, None)
    if switch:
        ctc["attn_sink"] = True
        ctc["attn_sink_init"] = init
    json.dump(c, open(os.path.join(out, "config.json"), "w", encoding="utf-8"))

    tt = {}
    for k, v in tensors.items():
        if ".gr_attn." in k or ".gr_mlp." in k:
            continue
        if k in sink_names:
            continue
        tt[k] = v
    if sinks is not None:
        dst = list(sink_names)
        if extra_layer is not None:
            dst.append(lname(extra_layer, "self_attn.sinks"))
        for k in dst:
            tt[k] = np.array(sinks, dtype=np.float32)
    if bf16_sinks:
        write_bf16(out, tt, sink_names)
    else:
        save_file(tt, os.path.join(out, "model.safetensors"))

def write_bf16(out, tt, names):
    """The same tensor set with `names` stored as BF16.

    Hand-rolled because safetensors.numpy has no bf16 dtype, and the
    engine's own exporter writes this tensor as BF16: its rule is
    `len(shape) != 2 or n < 512 -> FMT_BF16`, and (n_heads) is 1-D. So
    this is the storage form the .bin path will carry, not a format
    invented here.

    BF16 is the top 16 bits of the f32 bit pattern - the same
    `bits << 16` the loader widens with - so the truncation is exact for
    the fixtures' values and the arm's assertion (below) does not depend
    on that: it compares the two storage modes against EACH OTHER, which
    is an equality claim whatever the value."""
    import struct
    header, blobs, off = {}, [], 0
    for name, arr in tt.items():
        a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
        if name in names:
            data = (a.view(np.uint32) >> 16).astype("<u2").tobytes()
            dt, shape = "BF16", list(arr.shape)
        else:
            data = a.tobytes()
            dt, shape = "F32", list(arr.shape)
        header[name] = {"dtype": dt, "shape": shape,
                        "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs.append(data)
    hj = json.dumps(header).encode("ascii")
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(os.path.join(out, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)

# off: never carried the switch.
emit("off", sinks=None, switch=False, bf16_sinks=False)
# on: the checkpoint's own tensor.
emit("on", sinks=flat, switch=True, bf16_sinks=False)
# sens/text: the same tensor, offset into the regime where the column
# stops being sub-ULP (see the header's measured table).
emit("sens", sinks=flat + sens_delta, switch=True, bf16_sinks=False)
emit("text", sinks=flat + text_delta, switch=True, bf16_sinks=False)
# bf16: the same tensor in the storage form the exporter's own 1-D rule
# picks for it. Read back two ways - widened at load, and widened inside
# attn_sink_at - and the two must agree exactly.
#
# THE VALUE IS +TEXT_DELTA, NOT 0, and that is load-bearing rather than
# arbitrary: bf16 0.0 is sixteen zero bits, which is invariant under
# every byte-level mutation of the decoder, so a zeroed arm would report
# M3 green without the branch being live. +10.0 is bf16-exact (the
# on-must-equal-off claim below does not depend on that - the two decode
# sites agree for any value), and its two bytes differ from each other,
# which is what makes a byte-level mutation observable.
emit("bf16", sinks=flat + text_delta, switch=True, bf16_sinks=True)
# misplaced: the extra tensor sits where _wire_attn_sink never puts one.
emit("misplaced", sinks=flat, switch=True, bf16_sinks=False, extra_layer=csa2[0])
PY
PYRC=$?
[ "$PYRC" -eq 2 ] && exit 2
[ "$PYRC" -ne 0 ] && { echo "attn_sink_learned_gate: FAIL - fixture build"; exit 1; }

NHEADS=$(python3 -c '
import json,sys
c=json.load(open(sys.argv[1],encoding="utf-8")); tc=c.get("text_config",c)
print(tc["num_attention_heads"])' "$MODEL/config.json")
CSA2FIRST=$(python3 -c '
import json,sys
c=json.load(open(sys.argv[1],encoding="utf-8")); tc=c.get("text_config",c)
print(sorted(set(tc.get("csa2_layers") or ()))[0])' "$MODEL/config.json")

echo "attn_sink_learned_gate: $MODEL (n_heads=$NHEADS, sens +$SENS_DELTA, text +$TEXT_DELTA)"

# ---- arms ------------------------------------------------------------
check() { "$BIN" "$W/$1" --check >"$W/$1.chk" 2>&1; }
params() { sed -n 's/^Params *\([0-9][0-9]*\).*/\1/p' "$1" | head -1; }
why() { sed -n 's/^Failed to load: \(.*\)/\1/p' "$1" | head -1; }

# run TAG - one prefill, then NGEN greedy tokens; writes .pf, .txt, .log
run() {
    local tag="$1"
    "$BIN" "$W/$tag" --prompt "$PROMPT" -n "$NGEN" --temp 0 --seed 1145141919 \
        --dump-prefill "$W/$tag.pf" >"$W/$tag.log" 2>&1 || {
        echo "  FAIL: '$tag' exited non-zero"; tail -5 "$W/$tag.log"; return 1; }
    [ -s "$W/$tag.pf" ] || { echo "  FAIL: '$tag' wrote no prefill logits"; return 1; }
    sed -n '/generation start/,/generation end/p' "$W/$tag.log" \
        | sed '1d;$d' >"$W/$tag.txt"
    [ -s "$W/$tag.txt" ] || { echo "  FAIL: '$tag' generated no text"; return 1; }
    return 0
}

check off || { echo "  FAIL: off arm does not load: $(why "$W/off.chk")"; exit 1; }
POFF=$(params "$W/off.chk")
run off || exit 1
echo "  off  loads (params $POFF)"

if check on; then
    PON=$(params "$W/on.chk")
    run on || exit 1
    echo "  on   loads (params $PON)"
    if [ "$((PON - POFF))" -eq "$NHEADS" ]; then
        echo "  on - off params = $((PON - POFF)) = n_heads (the sinks tensor is claimed, once per sink layer)"
    else
        echo "  FAIL: on - off params = $((PON - POFF)), expected n_heads = $NHEADS"
        FAILED=1
    fi
else
    echo "  FAIL: on arm does not load: $(why "$W/on.chk")"
    exit 1
fi

# The measured inertness, reported and deliberately not asserted - see
# the header. A trained checkpoint would legitimately land on the other
# branch, and this gate must not call that a failure.
if cmp -s "$W/off.pf" "$W/on.pf"; then
    echo "  on vs off: prefill logits BITWISE IDENTICAL (this checkpoint's sinks are at init - reported, not asserted)"
else
    echo "  on vs off: prefill logits differ (this checkpoint's sinks are not inert)"
fi

# misplaced arm - a sink on a csa2 layer must be refused
if check misplaced; then
    echo "  FAIL: a sinks tensor on csa2 layer $CSA2FIRST was accepted; _wire_attn_sink skips those layers, so nothing may claim it"
    FAILED=1
else
    echo "  misplaced (sink on csa2 layer $CSA2FIRST): refused ($(why "$W/misplaced.chk"))"
fi

# sensitivity arms
run sens || exit 1
run text || exit 1
if cmp -s "$W/on.pf" "$W/sens.pf"; then
    echo "  FAIL: sens (+$SENS_DELTA) prefill logits equal on's - the gate is not reading the sinks tensor"
    FAILED=1
else
    echo "  sens (+$SENS_DELTA): prefill logits differ from on (the tensor is read)"
fi
if cmp -s "$W/on.txt" "$W/text.txt"; then
    echo "  FAIL: text (+$TEXT_DELTA) generated the same $NGEN tokens as on - the sinks tensor does not reach the output"
    FAILED=1
else
    echo "  text (+$TEXT_DELTA): greedy continuation differs from on"
fi

# bf16 storage arm. The engine decodes the same two bytes in two places -
# the loader's widen-at-load (visit_read) and attn_sink_at, which decodes
# in place when --bf16-store keeps the tensor narrow. Those two must
# agree exactly, whatever the value is: this is an equality between the
# two decode sites, not a precision expectation.
"$BIN" "$W/bf16" --check --bf16-store on >"$W/bf16.chk" 2>&1 || {
    echo "  FAIL: bf16 arm does not load: $(why "$W/bf16.chk")"; exit 1; }
if grep -q "1 BF16 tensor(s) in the file" "$W/bf16.chk" &&
   grep -q "kept at 2 bytes/elem" "$W/bf16.chk"; then
    for s in on off; do
        "$BIN" "$W/bf16" --prompt "$PROMPT" -n "$NGEN" --temp 0 \
            --seed 1145141919 --bf16-store $s \
            --dump-prefill "$W/bf16_$s.pf" >/dev/null 2>&1 || {
            echo "  FAIL: bf16 arm exited non-zero at --bf16-store $s"; exit 1; }
    done
    if cmp -s "$W/bf16_on.pf" "$W/bf16_off.pf"; then
        echo "  bf16 arm: the loader's widen-at-load and attn_sink_at's own decode agree bit for bit"
    else
        echo "  FAIL: the bf16 sink decodes differently in the loader and in attn_sink_at"
        FAILED=1
    fi
    if cmp -s "$W/on.pf" "$W/bf16_on.pf"; then
        echo "  FAIL: the bf16 fixture's logits equal on's, but its sinks are +$SENS_DELTA - it is not the same tensor"
        FAILED=1
    fi
else
    echo "  FAIL: the bf16 fixture's sinks tensor did not stay BF16 at load - the arm's premise is gone ($(grep -m1 'bf16 storage' "$W/bf16.chk"))"
    FAILED=1
fi

# ---- mutations -------------------------------------------------------
# Each on a private copy of src/ (never the live tree), and each verified
# to have actually matched before it is trusted - a sed that silently
# found nothing would build the unmutated engine and this gate would
# report a green mutation.
MUTBIN=
mutant() {
    local tag="$1" file="$2" from="$3" to="$4" probe="$5"
    local d="$W/$tag"
    MUTBIN=""
    mkdir -p "$d"
    cp -r "$ROOT/src" "$ROOT/common" "$d/"
    cp "$ROOT/Makefile" "$d/Makefile"
    sed -i "s/$from/$to/" "$d/$file"
    if ! grep -qF "$probe" "$d/$file"; then
        echo "  FAIL: mutation '$tag' did not match $file - this gate no longer matches the source"
        return 1
    fi
    if ! ( cd "$d" && make native ) >"$d/build.log" 2>&1; then
        echo "  FAIL: mutant '$tag' failed to build"; tail -5 "$d/build.log"
        return 1
    fi
    MUTBIN="$d/llama98.exe"
    return 0
}

# M1: the derivation never fires, so the tensor is never claimed.
if mutant m1 src/model.c \
        'L->attn_sink_on = c->attn_sink_learned' \
        'L->attn_sink_on = 0 \&\& c->attn_sink_learned' \
        'L->attn_sink_on = 0 && c->attn_sink_learned'; then
    if "$MUTBIN" "$W/on" --check >"$W/m1.chk" 2>&1; then
        echo "  FAIL: mutation M1 (attn_sink_on pinned to 0) still accepted the on fixture"
        FAILED=1
    else
        echo "  mutation M1 (attn_sink_on pinned to 0): red as it should be ($(why "$W/m1.chk"))"
    fi
else
    FAILED=1
fi

# M2: the flag fires and the tensor is claimed, but the forward never
# reads it. `on` must stay GREEN here - that is the point of this arm.
if mutant m2 src/forward_attn.c \
        'layer, L->attn_sink_on ? &L->attn_sinks : NULL,' \
        'layer, NULL,' \
        'layer, NULL,'; then
    if ! "$MUTBIN" "$W/on" --check >"$W/m2.chk" 2>&1; then
        echo "  FAIL: mutation M2 did not even load the on fixture - it is not the mutation this gate means to apply"
        FAILED=1
    else
        "$MUTBIN" "$W/on" --prompt "$PROMPT" -n "$NGEN" --temp 0 \
            --seed 1145141919 --dump-prefill "$W/m2_on.pf" >/dev/null 2>&1
        "$MUTBIN" "$W/sens" --prompt "$PROMPT" -n "$NGEN" --temp 0 \
            --seed 1145141919 --dump-prefill "$W/m2_sens.pf" >/dev/null 2>&1
        if cmp -s "$W/m2_on.pf" "$W/m2_sens.pf"; then
            echo "  mutation M2 (forward stops reading the tensor): red as it should be - on stays green, sens collapses onto it"
        else
            echo "  FAIL: mutation M2 was not caught - the sensitivity arm is not watching the sinks tensor"
            FAILED=1
        fi
    fi
else
    FAILED=1
fi

# M3: attn_sink_at's bf16 decode, broken at the byte level. The bf16 arm
# above compares two decode sites and would read green on an engine where
# the in-place branch is never taken (or never written) - the arm's own
# value is only as good as a mutation that can redden it.
if mutant m3 src/forward_attn.c \
        'p\[i [*] 2 [+] 1\] << 24' \
        'p[i * 2 + 1] << 16' \
        'p[i * 2 + 1] << 16'; then
    "$MUTBIN" "$W/bf16" --prompt "$PROMPT" -n "$NGEN" --temp 0 \
        --seed 1145141919 --bf16-store on \
        --dump-prefill "$W/m3_bf16.pf" >/dev/null 2>&1
    if cmp -s "$W/bf16_on.pf" "$W/m3_bf16.pf"; then
        echo "  FAIL: mutation M3 (bf16 decode shifted wrong) was not caught - the bf16 arm is not reading that branch"
        FAILED=1
    else
        echo "  mutation M3 (bf16 decode shifted wrong): red as it should be - the in-place bf16 decode is live and the arm watches it"
    fi
else
    FAILED=1
fi

[ "$FAILED" -eq 0 ] && { echo "attn_sink_learned_gate: PASS"; exit 0; }
echo "attn_sink_learned_gate: FAIL"
exit 1
