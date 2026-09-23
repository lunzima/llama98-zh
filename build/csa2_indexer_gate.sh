#!/bin/bash
# The CSA2 indexer schedule, driven for real.
#
# THE DEFECT THIS EXISTS FOR. --csa2-warmup-steps / --csa2-sparse-steps
# shipped with no run ever passing a non-zero value, and every csa2_* tensor
# in both shipping checkpoints sits at its init fingerprint. Turning the
# schedule on was not sufficient on its own: the KL's reference
# distribution (`p`) was not detached, so the indexer's objective pushed
# gradient into the main attention and the run reached NaN inside four
# steps - measured, on the tree before the fix, as an all-NaN csa2 layer
# and `val nan`.
#
# WHAT EACH ARM PROVES
#   mutation   a copy of a real arm's checkpoint with a NaN injected into
#              one indexer tensor, and with another tensor reset to the
#              base's own values, is reported as NaN and as unchanged
#              respectively. Without it, "five tensors finite and moved"
#              and "the checker looks at nothing" print the same line.
#   open       the arm trained with the stage ON has all five indexer
#              tensors finite AND differing from init. On the tree before
#              the fix this arm is the RED one - it reported nan=5.
#   negative   the same arm with the stage OFF leaves all five bitwise at
#              init. That is what says the switch moves them, not the
#              optimizer: the control tensor moved in BOTH arms.
#
# WHAT IT DOES NOT COVER: whether the indexer is any GOOD. Four steps at
# one seed is an instrument for "does it move at all", which is the
# question the schedule's absence left open; it says nothing about quality.
#
# CI-ONLY (exit 2 when it cannot run): it needs a CUDA device, the base
# checkpoint, and the packed corpus, and it trains two short arms - about
# ten minutes. `check` is meant to be runnable before every commit.
set -u
TRAIN_ROOT="${TRAIN_ROOT:-E:/LLM/llama98-zh}"
PROBE="$TRAIN_ROOT/tools/csa2_indexer_probe.py"

command -v python3 >/dev/null 2>&1 || {
    echo "csa2_indexer_gate: SKIP - no python3"
    exit 2
}
[ -f "$PROBE" ] || {
    echo "csa2_indexer_gate: SKIP - $PROBE missing - set TRAIN_ROOT"
    exit 2
}
python3 -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 2)" \
    >/dev/null 2>&1 || {
    echo "csa2_indexer_gate: SKIP - no CUDA device (the arms train)"
    exit 2
}
[ -f "${LZ_MODELS:-E:/LLM/models}/kunkun-ce/model.safetensors" ] || {
    echo "csa2_indexer_gate: SKIP - no kunkun-ce base checkpoint under" \
         "${LZ_MODELS:-E:/LLM/models}"
    exit 2
}

export PYTHONPATH="$TRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
rc=0
python3 "$PROBE" --self-test || rc=$?
if [ "$rc" = 2 ]; then
    exit 2
fi
if [ "$rc" != 0 ]; then
    echo "csa2_indexer_gate: FAIL - the mutation arm did not fire; the" \
         "probe cannot report a NaN or a tensor that did not move, so its" \
         "verdict on a real arm means nothing"
    exit 1
fi

python3 "$PROBE" || rc=$?
if [ "$rc" = 2 ]; then
    exit 2
fi
if [ "$rc" != 0 ]; then
    echo "csa2_indexer_gate: FAIL - driving --csa2-warmup-steps does not" \
         "leave the indexer finite and moved (see the per-arm lines above;" \
         "the tree this gate was written on reported an all-NaN layer)"
    exit 1
fi
echo "csa2_indexer_gate: PASS - the stage moves all five indexer tensors," \
     "they are finite, the stage-off arm leaves them at init, and the" \
     "mutation arm fired"
