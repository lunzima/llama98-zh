#!/bin/bash
# S1 attn_kv_share engine-side gate. model.h's own LZModelConfig
# attn_kv_share comment has the mechanism (source = the first
# full_attention layer, targets = every full_attention layer after it,
# five tied tensors) and src/model.c's apply_kv_share the loader half.
#
#   kv_share_gate.sh <model-dir> [prompt]
#
# The fixture is BUILT, not found: no checkpoint in this tree that this
# engine can load carries the key (kunkun98/kunkun-ce set it and are the
# two shipping checkpoints, but they also carry gated-residual and
# learned-sink tensors this engine has no spec for, so they do not load
# here at all - the gate says so and skips rather than blaming this
# feature). What is built instead is the SAME checkpoint run through both
# arrangements of the same weights.
#
# WHAT EACH ARM PROVES
#
#   dup vs share      The two arrangements must NOT agree. The engine's
#                     claim is that a target layer's K and V ARE the
#                     source layer's - computed from the SOURCE
#                     attention's input (model.h's attn_kv_share comment)
#                     - not a re-projection of the target's own hidden
#                     state. With the key off the target projects its own
#                     input; with it on the K/V projections are handed
#                     the source's. Identical logits mean the swap is not
#                     wired and the engine is back to sharing only the
#                     weights, which is a different function.
#   adjacent pair     One layer between source and target, built by
#                     promoting the linear layer under the last
#                     full-attention layer and demoting the first one, so
#                     the target's own input IS the source's OUTPUT. A
#                     swap fed a tensor captured after the source layer
#                     instead of at its attention entry is then
#                     indistinguishable from no swap at all - which is
#                     the shape the training side's own note records as
#                     having gone unnoticed ("the layer-level input IS
#                     the widened stream there, not the [B,S,d] tensor
#                     this capture needs"). The plain pair is blind to
#                     it; this one is not (measured: capturing one layer
#                     late reddens this arm and leaves the plain pair
#                     green).
#   collapse pair     The same adjacent stack with the source layer made
#                     the IDENTITY (both sublayers' output projections
#                     zeroed), so the source's input and the target's are
#                     literally one tensor and the two arrangements must
#                     agree again. It is the arm that rules out a
#                     substituted tensor which is neither the source's
#                     input nor the target's own (measured: feeding the
#                     K/V projections s->xb2 - whatever the previous
#                     layer left - reddens this arm alone).
#   control           The same absent tensors with the key OFF must fail
#                     loudly, naming the target layer's k_proj - the
#                     demonstration that the arms above are not green
#                     because nothing was removed. This is the arm that
#                     makes the set a measurement of the SWITCH rather
#                     than of the fixture.
#   bin               model.bin's v10 tail is one more <i after v9's two,
#                     so a file whose header gains that word must parse
#                     its tensor stream exactly as before: an earlier
#                     version's file rebuilt as v10 with the word set to 0
#                     must produce identical logits to the file it came
#                     from. A reader that does not consume the word, or
#                     consumes it twice, desyncs here, and it desyncs on
#                     the FIRST tensor - which is why this arm is the one
#                     that pins the header parse. The same extension with
#                     the word set to 1 must FAIL: the stream still
#                     carries the tensors a sharing model does not store,
#                     so the walk and the file disagree - and the writer
#                     that sets the word owes it to drop them. On a file
#                     that is ALREADY v10 (an exported artifact) the pair
#                     turns around: it must load, and the same file with
#                     the word forced to 0 must be refused at the target
#                     layer. Both refusals are checked against that
#                     layer's name, not merely against a non-zero exit - a
#                     header-parse bug also exits non-zero, and the two
#                     are not the same finding.
#
# WHAT WOULD FOOL IT
#
#   - A gate that only asked "does the sharing fixture load" would be
#     green on an engine that skipped the five specs and left the target
#     layers' fields zeroed (attention over an all-zero K/V still
#     produces finite logits). The dup arm is what rules that out.
#   - The control arm asserts the ERROR NAMES the target's k_proj, not
#     merely that the load failed: any unrelated failure (a typo in a
#     generated path, a fixture this engine cannot read) also exits
#     non-zero, and "some error happened" would take that for success.
#   - A PARTIAL SWAP IS INVISIBLE HERE, and no arm below can fix that.
#     Feeding only k_proj (or only v_proj) the source's input while the
#     other keeps the target's, or swapping q_proj's input as well, are
#     all real deviations from the trained function - and every one of
#     them changes the logits, so the "must differ" arms stay green and
#     the collapse arm stays green too (there the two inputs are one
#     tensor). Logits are the only observable this engine exposes: it has
#     no K/V dump, and a self-contained oracle would need a fixture whose
#     two inputs are equal, which is exactly where a swap stops being
#     observable. Pinning those needs a reference outside this tree (one
#     training-side forward on the same fixture), not another arm here.
#
# Exit: 0 pass, 1 fail, 2 skip (no binary / no model dir / a model this
# engine cannot load at all / no python3 with safetensors+numpy for the
# safetensors arms / no root containing at least two full_attention
# layers).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
PROMPT=${2:-"人工智能的发展历史可以追溯到二十世纪五十年代，当时的科学家们开始思考机器是否能够模拟人类的思维过程。"}
SEED=1145141919

BIN=$ROOT/llama98.exe
[ -x "$BIN" ] || { echo "kv_share_gate: SKIP - no $BIN (run: make native)"; exit 2; }
[ -n "$MODEL" ] && [ -d "$MODEL" ] || {
    echo "kv_share_gate: SKIP - model dir not given or not found: '$MODEL'"; exit 2; }

HAVE_ST=0
[ -f "$MODEL/model.safetensors" ] && HAVE_ST=1
HAVE_BIN=0
[ -f "$MODEL/model.bin" ] && HAVE_BIN=1
if [ "$HAVE_ST" = 0 ] && [ "$HAVE_BIN" = 0 ]; then
    echo "kv_share_gate: SKIP - no model.safetensors or model.bin under '$MODEL'"
    exit 2
fi

# The fixture is the checkpoint as this engine sees it, so a checkpoint
# it cannot load leaves nothing to test. Reported rather than passed
# over: the message names what the model carries that this engine does
# not, and that is a fact about the tree, not about this switch.
if ! PRE=$("$BIN" "$MODEL" --check 2>&1); then
    echo "kv_share_gate: SKIP - the given checkpoint does not load in this engine:"
    echo "$PRE" | tail -2 | sed 's/^/  /'
    exit 2
fi

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
FAILED=0
RAN=0

# --------------------------------------------------------------- safetensors
if [ "$HAVE_ST" = 1 ]; then
    if ! python3 -c "import safetensors, numpy" >/dev/null 2>&1; then
        echo "  safetensors arms: SKIP - no python3 safetensors/numpy"
    else
        python3 - "$MODEL" "$W" >"$W/build.log" 2>&1 <<'PY'
import json, os, shutil, sys
import numpy as np
from safetensors.numpy import load_file, save_file

model_dir, w = sys.argv[1], sys.argv[2]
cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
tc = cfg.get("text_config", cfg)
lt = tc["layer_types"]
full = [i for i, t in enumerate(lt) if t == "full_attention"]
if len(full) < 2:
    print("SKIP: fewer than two full_attention layers - attn_kv_share is a no-op there")
    sys.exit(2)

src, tgts = full[0], full[1:]
tensors = load_file(os.path.join(model_dir, "model.safetensors"))
prefix = "model."
if ("model.layers.%d.self_attn.q_proj.weight" % src) not in tensors:
    prefix = ""

def name(layer, suffix):
    return "%slayers.%d.self_attn.%s" % (prefix, layer, suffix)

# The five tensors the training side ties - engine/kunmoe_modeling.py's
# _SHARED_KV_ATTRS. Only those the source layer actually carries: a
# model without SubLN has no k_proj.norm/v_proj.norm spec either, so the
# two lists stay the same one.
SHARED = [s for s in ("k_proj.weight", "v_proj.weight", "k_norm.weight",
                      "k_proj.norm.weight", "v_proj.norm.weight")
          if name(src, s) in tensors]
if not SHARED:
    print("SKIP: source full_attention layer %d carries none of the shared tensors" % src)
    sys.exit(2)

def emit(tag, tt, ltypes, flag):
    out = os.path.join(w, tag)
    os.makedirs(out, exist_ok=True)
    for n in os.listdir(model_dir):
        if n in ("model.safetensors", "model.bin", "model.bin.lz4"):
            continue
        p = os.path.join(model_dir, n)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(out, n))
    c = json.loads(json.dumps(cfg))
    ctc = c.get("text_config") or c
    ctc["layer_types"] = ltypes
    ctc["attn_kv_share"] = flag
    json.dump(c, open(os.path.join(out, "config.json"), "w", encoding="utf-8"))
    save_file(tt, os.path.join(out, "model.safetensors"))

def kv_arm(base, ltypes, dup, zero_src_outputs):
    """One arm's tensor set, read off `base`: the shared tensors duplicated
    onto (or absent from) each target layer, and optionally the source
    layer's own output projections zeroed so that layer is the identity."""
    f = [i for i, x in enumerate(ltypes) if x == "full_attention"]
    s0 = f[0]
    tt = dict(base)
    for tgt in f[1:]:
        for suf in SHARED:
            k = "%slayers.%d.self_attn.%s" % (prefix, tgt, suf)
            if dup:
                tt[k] = base["%slayers.%d.self_attn.%s" % (prefix, s0, suf)]
            elif k in tt:
                del tt[k]
    if zero_src_outputs:
        for k in zero_src_outputs:
            full_k = "%slayers.%d.%s" % (prefix, s0, k)
            tt[full_k] = np.zeros_like(tt[full_k])
    return tt

last_full = full[-1]
csa2_layers = tc.get("csa2_layers") or []

# (a) THE STACK AS IT IS: source and target far apart, other layers between
# them.
emit("dup",   kv_arm(tensors, lt, True, None), lt, False)
emit("share", kv_arm(tensors, lt, False, None), lt, True)
emit("ctrl",  kv_arm(tensors, lt, False, None), lt, False)
# The name the control arm has to be refused on: the walk reaches the
# target's k_proj before any other shared tensor (q_proj is present, and
# k_proj precedes v_proj/k_norm in LAYER_SPECS). Written to a file so the
# asserting shell below greps a literal, not a re-derivation.
open(os.path.join(w, "ctrl.expect"), "w").write(
    "layers.%d.self_attn.k_proj.weight" % tgts[0])

# (b) SOURCE AND TARGET ADJACENT, which is what separates "the target's K/V
# come from the source's own input" from "from the source's OUTPUT": with
# one layer between them, a tensor captured after the source layer instead
# of at its attention entry IS the target's own input, and the two readings
# become indistinguishable. Built by promoting the linear layer just below
# the last full-attention layer to full attention and demoting the first
# full-attention layer to linear, each layer's own attention tensors moving
# with it.
adj_note = "adjacent arms: built"
adj_arms = ""
prom = last_full - 1
if csa2_layers:
    adj_note = "adjacent arms: SKIP - the fixture names csa2_layers"
elif src == last_full or prom < 1 or lt[prom] != "linear_attention":
    adj_note = "adjacent arms: SKIP - no linear layer under the last full one"
elif ("%slayers.%d.self_attn.o_proj.weight" % (prefix, src)) not in tensors:
    adj_note = "adjacent arms: SKIP - no plain o_proj at the source layer"
else:
    adj_lt = list(lt)
    adj_lt[prom] = "full_attention"
    adj_lt[src] = "linear_attention"

    def move(t, a, b, kind):
        out = dict(t)
        pre = "%slayers.%d.%s." % (prefix, a, kind)
        for k in [k for k in t if k.startswith(pre)]:
            out["%slayers.%d.%s.%s" % (prefix, b, kind, k[len(pre):])] = t[k]
            del out[k]
        return out

    adj_t = move(tensors, src, prom, "self_attn")
    adj_t = move(adj_t, prom, src, "linear_attn")
    # The source layer made IDENTITY, so that its attention input and the
    # target's are the same tensor: with BOTH sublayers' outputs zero it
    # contributes nothing to the residual, and the target layer directly
    # above it receives exactly the source's own input. Every output
    # projection the FFN has has to be in this list - a routed path through
    # the latent, a shared expert, and a dense down_proj - or the layer is
    # not identity and the arm proves nothing. Measured: zeroing o_proj
    # alone leaves the FFN's contribution and the arm stays red.
    ZERO = [k for k in ("self_attn.o_proj.weight",
                        "mlp.routed_expert_up_proj.weight",
                        "mlp.shared_experts.down_proj.weight",
                        "mlp.down_proj.weight")
            if ("%slayers.%d.%s" % (prefix, prom, k)) in adj_t]
    emit("adj_dup", kv_arm(adj_t, adj_lt, True, None), adj_lt, False)
    emit("adj_share", kv_arm(adj_t, adj_lt, False, None), adj_lt, True)
    emit("adj_col_dup", kv_arm(adj_t, adj_lt, True, ZERO), adj_lt, False)
    emit("adj_col_share", kv_arm(adj_t, adj_lt, False, ZERO), adj_lt, True)
    adj_arms = " adj_dup adj_share adj_col_dup adj_col_share"
    print("adjacent source %d, target %d; source output projections zeroed: %s"
          % (prom, last_full, ZERO))

# The arms the shell below dumps and compares. ctrl is not one of them: it
# is the arm that must FAIL to load.
open(os.path.join(w, "arms"), "w").write("dup share" + adj_arms + "\n")

print("built dup/share/ctrl (+ adjacent) under %s - source layer %d, targets %s,"
      " shared %s; %s" % (w, src, tgts, SHARED, adj_note))
PY
        BUILD_RC=$?
        cat "$W/build.log"
        if [ "$BUILD_RC" -eq 2 ]; then
            echo "  safetensors arms: SKIP - fixture not usable"
        elif [ "$BUILD_RC" -ne 0 ]; then
            echo "kv_share_gate: FAIL - fixture build failed"
            exit 1
        else
            RAN=1

            dump() {   # dump <tag> ; logits -> $W/<tag>.pf, log -> $W/<tag>.log
                "$BIN" "$W/$1" --prompt "$PROMPT" -n 1 --seed "$SEED" \
                    --dump-prefill "$W/$1.pf" >"$W/$1.log" 2>&1
            }

            ARMS=$(cat "$W/arms")
            for tag in $ARMS; do
                dump "$tag" || {
                    echo "  FAIL: '$tag' exited non-zero"
                    tail -4 "$W/$tag.log" | sed 's/^/    /'
                    FAILED=1
                }
                [ -s "$W/$tag.pf" ] || {
                    echo "  FAIL: '$tag' wrote no prefill logits"
                    FAILED=1
                }
            done

            if [ "$FAILED" = 0 ]; then
                # differ: the two readings must part company. Both arms load
                # the same weights; the only difference is whether the
                # target layer's K/V come from the source's input or from
                # its own, so identical logits mean the swap is not wired.
                differ() {
                    if cmp -s "$W/$1.pf" "$W/$2.pf"; then
                        echo "  FAIL: $1 and $2 are bitwise identical -"
                        echo "        $3"
                        FAILED=1
                    else
                        echo "  $1 vs $2: differ, as the two readings must"
                    fi
                }
                # same: the two readings must AGREE here, because the arm
                # was built so their inputs are literally one tensor.
                same() {
                    if cmp -s "$W/$1.pf" "$W/$2.pf"; then
                        echo "  $1 vs $2: identical, as they must be"
                    else
                        echo "  FAIL: $1 and $2 disagree - $3"
                        FAILED=1
                    fi
                }
                differ dup share "the target layer's K/V are still its own input"
                case " $ARMS " in
                    *" adj_dup "*)
                        differ adj_dup adj_share \
                            "one layer apart, a tensor taken after the source layer is the target's own input"
                        same adj_col_dup adj_col_share \
                            "the source layer is the identity, so both readings feed the K/V projections the same tensor"
                        ;;
                esac
                python3 - "$W" $ARMS <<'PY' || FAILED=1
import os, struct, sys
w = sys.argv[1]
for tag in sys.argv[2:]:
    b = open(os.path.join(w, tag + ".pf"), "rb").read()
    v = struct.unpack("<%df" % (len(b) // 4), b[:len(b) // 4 * 4])
    bad = [x for x in v if x != x or x in (float("inf"), float("-inf"))]
    if bad:
        print("  FAIL: %s holds %d non-finite logit(s)" % (tag, len(bad)))
        raise SystemExit(1)
print("  logits finite on all %d arm(s)" % len(sys.argv[2:]))
PY
            fi

            # Control: the tensors the share arm lacks, with the switch
            # off. Must fail on the tensor that is actually missing, so
            # it is the message that is asserted, not the exit code -
            # any unrelated failure exits non-zero too.
            EXPECT=$(cat "$W/ctrl.expect")
            if "$BIN" "$W/ctrl" --prompt "$PROMPT" -n 1 --seed "$SEED" \
                   --dump-prefill "$W/ctrl.pf" >"$W/ctrl.log" 2>&1; then
                echo "  FAIL: control (tensors absent, attn_kv_share=0) loaded -"
                echo "        the share arm above proves nothing"
                FAILED=1
            elif ! grep -qF "$EXPECT" "$W/ctrl.log"; then
                echo "  FAIL: control failed, but not on the missing shared"
                echo "        tensor ($EXPECT):"
                tail -2 "$W/ctrl.log" | sed 's/^/    /'
                FAILED=1
            else
                echo "  control (tensors absent, attn_kv_share=0): refused on"
                echo "    $(tail -1 "$W/ctrl.log")"
            fi
        fi
    fi
fi

# ------------------------------------------------------------------ model.bin
# Two shapes, one pair of arms. A file written before this version gains the
# v10 tail here, at the value the reader defaults every earlier version's
# absent tail to - except the two the reader validates or derives from:
# rope_scaling (whose "off" values are type 0 with a positive factor and the
# stock YaRN bounds) and the Hadamard block sizes (-1 = derive, which 0 is
# not). A file that is ALREADY v10 carries the tail itself, so the arms
# become: load it as it is, and refuse it with the tail word forced to 0 -
# the same switch the exporter set, moved by hand.
#
#   bin_ok    loads, and its logits are identical to the untouched file's
#             (vacuous when the file is already v10 and is that file; what
#             the arm asserts there is that a shipped v10 artifact runs).
#   bin_bad   refused. Either the tail word was set on a stream that still
#             carries the tensors a sharing model does not store, or the
#             word was cleared on a stream that does not carry them.
if [ "$HAVE_BIN" = 1 ]; then
    python3 - "$MODEL" "$W" >"$W/binbuild.log" 2>&1 <<'PY'
import os, shutil, struct, sys

model_dir, w = sys.argv[1], sys.argv[2]
MAX_LAYERS = 128
SKIP_FILES = ("model.safetensors", "model.bin", "model.bin.lz4")

def header_len(ver):
    words = 28 + MAX_LAYERS          # fixed fields + layer_types
    if ver >= 4: words += 13         # KDA gate + latent MoE
    if ver >= 5: words += 2          # MTP head
    if ver >= 6: words += 5          # rope_scaling
    if ver >= 7: words += 1          # use_subn
    if ver >= 8: words += 2          # hadamard blocks
    if ver >= 9: words += 2          # MLA ranks
    if ver >= 10: words += 1         # attn_kv_share
    return 8 + 4 * words

def tails_below(ver):
    out = b""
    if ver < 4: out += struct.pack("<13i", *([0] * 13))
    if ver < 5: out += struct.pack("<2i", 0, 0)
    if ver < 6: out += struct.pack("<5i", 0, 0x3f800000, 0, 0x42000000, 0x3f800000)
    if ver < 7: out += struct.pack("<i", 0)
    if ver < 8: out += struct.pack("<2i", -1, -1)
    if ver < 9: out += struct.pack("<2i", 0, 0)
    return out

def emit(tag, blob):
    out = os.path.join(w, tag)
    os.makedirs(out, exist_ok=True)
    for n in os.listdir(model_dir):
        if n in SKIP_FILES: continue
        p = os.path.join(model_dir, n)
        if os.path.isfile(p):
            shutil.copyfile(p, os.path.join(out, n))
    open(os.path.join(out, "model.bin"), "wb").write(blob)

b = open(os.path.join(model_dir, "model.bin"), "rb").read()
ver = struct.unpack("<I", b[4:8])[0]
# layer_types sits right after the 28 fixed words; LZ_LT_FULL is 1.
lt = struct.unpack("<%di" % MAX_LAYERS, b[120:120 + 4 * MAX_LAYERS])
full = [i for i, t in enumerate(lt) if t == 1]

if ver >= 10:
    hl = header_len(ver)
    word = struct.unpack("<i", b[hl - 4:hl])[0]
    if word == 0:
        print("SKIP: this file is v%d with attn_kv_share=0 - nothing is shared" % ver)
        sys.exit(2)
    if len(full) < 2:
        print("SKIP: this file has %d full_attention layer(s) - nothing is shared"
              % len(full))
        sys.exit(2)
    emit("bin_ok", b)
    emit("bin_bad", b[:hl - 4] + struct.pack("<i", 0) + b[hl:])
    print("v%d file already carries the tail word 1 - control is the same file"
          " with 0" % ver)
else:
    if len(full) < 2:
        print("SKIP: this file has %d full_attention layer(s)" % len(full))
        sys.exit(2)
    hl = header_len(ver)
    emit("bin_ok", b[:4] + struct.pack("<I", 10) + b[8:hl]
                  + tails_below(ver) + struct.pack("<i", 0) + b[hl:])
    emit("bin_bad", b[:4] + struct.pack("<I", 10) + b[8:hl]
                   + tails_below(ver) + struct.pack("<i", 1) + b[hl:])
    print("rebuilt model.bin v%d -> v10 with the tail word 0 and 1" % ver)

# The refusal has to LAND ON THE TARGET LAYER, which is the whole
# observable: the walk skips that layer's shared specs, so the stream
# and the walk part company there and nowhere earlier. Which tensor name
# the message carries is mode-dependent and not asserted - cleared on a
# trimmed stream it is the k_proj the walk now expects, set on an
# untrimmed one it is whatever follows the skipped block (o_proj), and a
# header-level desync would name the first tensor instead of a layer.
open(os.path.join(w, "bin.expect"), "w").write(
    "layers.%d.self_attn." % full[1])
PY
    BINBUILD_RC=$?
    cat "$W/binbuild.log"
    if [ "$BINBUILD_RC" -eq 2 ]; then
        echo "  bin arms: SKIP - fixture not usable"
    elif [ "$BINBUILD_RC" -ne 0 ]; then
        echo "kv_share_gate: FAIL - bin fixture build failed"
        exit 1
    else
        RAN=1
        bdump() {   # bdump <dir> <out> <log>
            "$BIN" "$1" --prompt "$PROMPT" -n 1 --seed "$SEED" \
                --dump-prefill "$2" >"$3" 2>&1
        }
        bdump "$MODEL" "$W/bin_base.pf" "$W/bin_base.log" || {
            echo "  FAIL: the untouched model.bin exited non-zero"
            FAILED=1
        }
        bdump "$W/bin_ok" "$W/bin_ok.pf" "$W/bin_ok.log" || {
            echo "  FAIL: the loadable arm exited non-zero"
            tail -2 "$W/bin_ok.log" | sed 's/^/    /'
            FAILED=1
        }
        if [ "$FAILED" = 0 ] && [ -s "$W/bin_base.pf" ]; then
            if cmp -s "$W/bin_base.pf" "$W/bin_ok.pf"; then
                echo "  bin_ok: logits bitwise identical to the untouched file"
            else
                echo "  FAIL: the tail changed the tensor stream's parse"
                FAILED=1
            fi
        fi
        # The tail word off/on against the stream's actual contents: the
        # walk must refuse, and refuse at the target layer.
        EXPECT=$(cat "$W/bin.expect")
        if bdump "$W/bin_bad" "$W/bin_bad.pf" "$W/bin_bad.log"; then
            echo "  FAIL: bin_bad loaded - the walk and the tensor stream"
            echo "        disagree and the load did not say so"
            FAILED=1
        elif ! grep -qF "$EXPECT" "$W/bin_bad.log"; then
            echo "  FAIL: bin_bad was refused, but not at the target layer"
            echo "        ($EXPECT):"
            tail -2 "$W/bin_bad.log" | sed 's/^/    /'
            FAILED=1
        else
            echo "  bin_bad: refused at the target layer on"
            echo "    $(tail -1 "$W/bin_bad.log")"
        fi
    fi
fi

[ "$RAN" = 1 ] || { echo "kv_share_gate: SKIP - neither arm could be built"; exit 2; }
[ "$FAILED" -eq 0 ] && { echo "kv_share_gate: PASS"; exit 0; }
echo "kv_share_gate: FAIL"
exit 1
