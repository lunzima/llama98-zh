#!/bin/bash
# A model directory must be loadable by the engine that produced it.
#
# THE DEFECT THIS EXISTS FOR. E:/LLM/models/kunkun-ce shipped a
# modeling_kunmoe.py three config fields and two methods behind
# engine/kunmoe_modeling.py. A model directory loads through
# trust_remote_code, so the KunMoeConfig that runs is THAT file's, while
# engine/csa2.py - the module that wires the CSA2 layers - is imported
# from the training tree whichever copy loads. The two are one pair and
# nothing kept them in step, so `_wire_csa2`'s `config.resolved_csa2_mode(i)`
# raised AttributeError before the first training step: the checkpoint
# could not be loaded by the code that wrote it, and the csa2 schedule
# behind that call has therefore never run.
#
# WHAT THE TWO ARMS PROVE
#   mutation   a copy of the artifact's own modeling_kunmoe.py with one
#              required name renamed away is reported MISSING. Without
#              this the gate could pass on a detector that never fires -
#              "every artifact ok" and "the checker looks at nothing"
#              print the same line.
#   real       every live artifact answers all of them. RED ON THE TREE
#              BEFORE THE FIX, on both artifacts, naming
#              resolved_csa2_mode / resolved_csa2_compress_m / csa2_kv_source.
#
# THE NAME LIST IS NOT IN THIS FILE. build-side would be a second copy of
# an interface that moves; tools/check_artifact_modeling.py parses it out
# of engine/csa2.py's own `config.<name>` accesses.
#
# WHAT THIS GATE DOES NOT COVER: whether the refreshed file is
# BEHAVIOURALLY right for the artifact's weights - only that the artifact
# can answer what the engine asks. The weights in these directories were
# migrated to the V4.1 indexer geometry (the .bak-pre-indexer-v41-* files
# date it), which is why refreshing the copy is the direction that
# matches what the directory already is.
#
# SKIPs (exit 2) when there is no artifact root or no training tree: the
# checker prints its own SKIP line, and a silent skip is what would be
# wrong rather than a reported one.
set -u
TRAIN_ROOT="${TRAIN_ROOT:-E:/LLM/llama98-zh}"
MODELS="${LZ_MODELS:-E:/LLM/models}"

command -v python3 >/dev/null 2>&1 || {
    echo "artifact_modeling_gate: SKIP - no python3"
    exit 2
}
CHECKER="$TRAIN_ROOT/tools/check_artifact_modeling.py"
[ -f "$CHECKER" ] || {
    echo "artifact_modeling_gate: SKIP - $CHECKER missing - set TRAIN_ROOT"
    exit 2
}

export PYTHONPATH="$TRAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"
rc=0
python3 "$CHECKER" --self-test --models "$MODELS" || rc=$?
if [ "$rc" = 2 ]; then
    exit 2
fi
if [ "$rc" != 0 ]; then
    echo "artifact_modeling_gate: FAIL - the mutation arm did not fire; the" \
         "checker cannot report a name it should miss, so its verdict on the" \
         "real artifacts means nothing"
    exit 1
fi

python3 "$CHECKER" --models "$MODELS" || rc=$?
if [ "$rc" = 2 ]; then
    exit 2
fi
if [ "$rc" != 0 ]; then
    echo "artifact_modeling_gate: FAIL - a model directory under" \
         "$MODELS carries a modeling_kunmoe.py that cannot answer what" \
         "engine/csa2.py reads off its config. Refresh it from" \
         "$TRAIN_ROOT/engine/kunmoe_modeling.py."
    exit 1
fi
echo "artifact_modeling_gate: PASS - every artifact answers engine/csa2.py's" \
     "config interface, and the mutation arm fired"
