#!/bin/bash
# S2 CSA2 (Compressed Sparse Attention) engine-side scalar reference
# gate. model.h's own LZModelConfig.csa2_on[] comment has the mechanism
# and citation trail; src/forward_csa2.c's own header comment has the
# scope (scalar only, no SIMD, no S4 composition, no --spec rollback).
#
#   csa2_gate.sh <model-dir> [prompt]
#
# No real CSA2-trained checkpoint exists (export_q8.py has no writer for
# this format yet - out of scope for this pass, see docs/dsv41-
# optimization-menu.md's CSA2 section), so this builds one: an existing
# use_subn=0 full-attention checkpoint gets a synthetic csa2_w_* tensor
# set bolted onto one of its full-attention layers and csa2_layers/
# csa2_compress_m/etc. added to its config.json.
#
# WHAT EACH ARM PROVES
#
#   batch4 vs batch1   The strongest correctness signal available
#                      without a trained reference: forward_csa2.c
#                      cannot batch across a chunk's nt tokens the way
#                      ordinary attention does (its own header comment
#                      explains why - block finalization state changes
#                      mid-chunk), so every token is processed one at a
#                      time regardless of --batch. If that per-token
#                      loop's ring-buffer bookkeeping (pending block
#                      accumulation, finalization, causal block count)
#                      were wrong in any way that depends on WHERE chunk
#                      boundaries fall, forcing every position through
#                      its own separate forward_csa2 call (--batch 1)
#                      would disagree with the default (--batch 4,
#                      several positions per call). Bitwise identity
#                      between the two is the assertion, not a
#                      tolerance - both arms compute the exact same
#                      sequence of state transitions in exact float
#                      arithmetic if the bookkeeping is position-
#                      correct, and a real bug (an off-by-one in which
#                      blocks are visible, e.g.) reliably desyncs
#                      exactly when a chunk boundary lands on a
#                      different token than the compression block
#                      boundary - --batch 4 against block size 2
#                      guarantees at least one such misalignment across
#                      an 8-token-plus prompt.
#   mutation           Corrupts csa2_w_weights (Eq 1's per-head scoring
#                      weights) on a COPY of the same fixture and
#                      requires the prefill logits to differ from the
#                      unmutated arm - the same "a knob wired to nothing
#                      also looks identical" concern every invariance
#                      gate in this tree carries (ced_gate.sh's own
#                      header), applied here to "does this tensor
#                      actually reach the output" instead.
#
# Exit: 0 pass, 1 fail, 2 skip (no binary / no model / no python3
# safetensors+numpy / model has no usable full_attention, use_subn=0
# layer).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
PROMPT=${2:-"人工智能的发展历史可以追溯到二十世纪五十年代，当时的科学家们开始思考机器是否能够模拟人类的思维过程。"}

BIN=$ROOT/llama98.exe
[ -x "$BIN" ] || { echo "csa2_gate: SKIP - no $BIN (run: make native)"; exit 2; }
[ -n "$MODEL" ] && [ -d "$MODEL" ] || {
    echo "csa2_gate: SKIP - model dir not given or not found: '$MODEL'"; exit 2; }
[ -f "$MODEL/config.json" ] && [ -f "$MODEL/model.safetensors" ] || {
    echo "csa2_gate: SKIP - no config.json/model.safetensors under '$MODEL'"; exit 2; }
python3 -c "import safetensors, numpy" >/dev/null 2>&1 || {
    echo "csa2_gate: SKIP - no python3 safetensors/numpy"; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
FAILED=0

BUILD_LOG="$W/build.log"
python3 - "$MODEL" "$W" > "$BUILD_LOG" 2>&1 <<'PY'
import json, os, sys
import numpy as np
from safetensors.numpy import load_file, save_file

model_dir, w = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
tc = cfg.get("text_config", cfg)
if tc.get("use_subn"):
    print("SKIP: use_subn=1 - this fixture cannot host the gate "
          "(no relation to CSA2 itself, just keeps the synthetic "
          "tensor set to the plain, un-normed shapes csa2 uses)")
    sys.exit(2)
lt = tc["layer_types"]
full_layers = [i for i, t in enumerate(lt) if t == "full_attention"]
if not full_layers:
    print("SKIP: no full_attention layer in this fixture")
    sys.exit(2)
layer = full_layers[0]
hidden = tc["hidden_size"]

tensors = load_file(os.path.join(model_dir, "model.safetensors"))
prefix = "model."
base = prefix + "layers.%d.self_attn." % layer
if (base + "q_proj.weight") not in tensors:
    prefix = ""
    base = prefix + "layers.%d.self_attn." % layer

# The indexer key's row is idx_hd wide and forward_csa2.c rotates it with
# the same lz_rope the main query uses, which writes base[i] and
# base[i + rotary_dim/2] inside one row. A row narrower than the host
# model's rotary width is therefore a pairing the loader refuses outright
# (LZ_ERR_CFG_CSA2_IDX_ROTARY) - and one this fixture must not build,
# because the synthetic run would be running off the end of every row.
# idx_hd is raised to the host's own rotary width instead of staying at a
# fixed 16; the rest of the fixture's shapes follow it.
_rp = tc.get("rope_parameters") or {}
rotary = int(tc["head_dim"] * tc.get("partial_rotary_factor",
                                     _rp.get("partial_rotary_factor", 1.0)))
idx_heads, idx_hd, idx_r, m, topk = 2, max(16, rotary), 16, 2, 4
rng = np.random.default_rng(1145141919)

def randw(out_dim, in_dim):
    return (rng.standard_normal((out_dim, in_dim)) * 0.02).astype(np.float32)

def build(tag, weights_fn):
    out_dir = os.path.join(w, tag)
    os.makedirs(out_dir, exist_ok=True)
    import shutil
    for name in os.listdir(model_dir):
        if name in ("model.safetensors", "model.bin", "model.bin.lz4"):
            continue
        src = os.path.join(model_dir, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_dir, name))
    out = dict(tensors)
    out[base + "csa2_w_kv.weight"] = weights_fn("kv", tc["head_dim"], hidden)
    out[base + "csa2_w_gate.weight"] = weights_fn("gate", tc["head_dim"], hidden)
    out[base + "csa2_w_kv_idx.weight"] = weights_fn("kv_idx", idx_hd, tc["head_dim"])
    # 1-D norm, drawn as (idx_hd, 1) so this arm consumes the same number
    # of draws from the shared RNG stream as every other tensor here -
    # the "wrong" arm's independence from "on" rests on that stream being
    # the same shape of draw for both.
    out[base + "csa2_k_norm_idx.weight"] = weights_fn("k_norm_idx", idx_hd, 1).reshape(-1)
    out[base + "csa2_w_weights.weight"] = weights_fn("weights", idx_heads, hidden)
    out[base + "csa2_w_dq_idx.weight"] = weights_fn("dq_idx", idx_r, hidden)
    out[base + "csa2_w_uq_idx.weight"] = weights_fn("uq_idx", idx_heads * idx_hd, idx_r)
    new_cfg = json.loads(json.dumps(cfg))
    new_tc = new_cfg.get("text_config", new_cfg)
    new_tc["csa2_layers"] = [layer]
    new_tc["csa2_compress_m"] = m
    new_tc["csa2_topk"] = topk
    new_tc["csa2_indexer_heads"] = idx_heads
    new_tc["csa2_indexer_head_dim"] = idx_hd
    new_tc["csa2_indexer_q_lora_rank"] = idx_r
    json.dump(new_cfg, open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"))
    save_file(out, os.path.join(out_dir, "model.safetensors"))

rng_state = rng.bit_generator.state
build("on", lambda *a: randw(a[1], a[2]))
# mutation: an INDEPENDENT random draw for csa2_w_weights only (the
# other six tensors are byte-identical to the "on" arm) - if the
# indexer's per-head weights never reached the output, this arm would
# be indistinguishable from "on".
rng.bit_generator.state = rng_state
def mutated(name, out_dim, in_dim):
    w_ = randw(out_dim, in_dim)
    if name == "weights":
        w_ = (np.random.default_rng(271828182).standard_normal((out_dim, in_dim)) * 0.02).astype(np.float32)
    return w_
build("wrong", mutated)
print("built on/wrong under %s, layer %d, m=%d topk=%d" % (w, layer, m, topk))
PY
BUILD_RC=$?
cat "$BUILD_LOG"
if [ "$BUILD_RC" -eq 2 ]; then exit 2; fi
if [ "$BUILD_RC" -ne 0 ]; then
    echo "csa2_gate: FAIL - fixture build failed"
    exit 1
fi

arm() {
    local tag="$1" batch="$2" out="$3"
    "$BIN" "$W/$tag" --prompt "$PROMPT" -n 1 --seed 1145141919 --batch "$batch" \
        --dump-prefill "$out" >"$W/$tag.b$batch.log" 2>&1 || {
        echo "  FAIL: '$tag' --batch $batch exited non-zero"
        tail -8 "$W/$tag.b$batch.log"; return 1; }
    [ -s "$out" ] || { echo "  FAIL: '$tag' --batch $batch wrote no prefill logits"; return 1; }
    return 0
}

arm on 4 "$W/on_b4.pf" || exit 1
arm on 1 "$W/on_b1.pf" || exit 1
arm wrong 4 "$W/wrong_b4.pf" || exit 1

if cmp -s "$W/on_b4.pf" "$W/on_b1.pf"; then
    echo "  batch4 vs batch1: bitwise identical (expected)"
else
    echo "  FAIL: --batch 4 and --batch 1 disagree on the same prompt - the"
    echo "        per-token ring-buffer state depends on where chunk"
    echo "        boundaries fall, which it must not"
    FAILED=1
fi

python3 - "$W/on_b4.pf" "$W/wrong_b4.pf" <<'PY' || FAILED=1
import sys
import numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32)
b = np.fromfile(sys.argv[2], dtype=np.float32)
if not (np.isfinite(a).all() and np.isfinite(b).all()):
    print("  FAIL: non-finite logits"); raise SystemExit(1)
if np.array_equal(a, b):
    print("  FAIL: mutation was not caught (wrong arm identical to on arm)")
    raise SystemExit(1)
print("  mutation (independent csa2_w_weights draw): red as it should be (logits differ)")
PY

[ "$FAILED" -eq 0 ] && { echo "csa2_gate: PASS"; exit 0; }
echo "csa2_gate: FAIL"
exit 1
