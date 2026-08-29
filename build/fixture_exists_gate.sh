#!/bin/bash
# Every model fixture a script NAMES has to be on disk.
#
# A path in a script is a claim about the machine, and it rots exactly
# like a comment naming a function that got renamed - except that nothing
# reads it until the run that needed it, and by then the failure mode is
# a SKIP rather than an error.
#
# WHAT IT CAUGHT, and why the cost was not theoretical:
#
#   ci.sh's ARM_MODEL was $MODELS_ROOT/kunkun98-pilot-20260806, which is (historical)
#   not on disk. build/arm/parity_gate.sh SKIPs on a missing model dir,
#   ci.sh counts SKIP as not-a-failure, so the x86-against-ARM
#   bit-identity contract - the hardest promise this engine makes - was
#   not being checked by CI at all, and the run reported no failure.
#
#   The same name was in CHECK_MODELS, so the "three representative
#   checkpoints" sweep was running two.
#
#   fixed_tier_xcompiler_gate.sh's header told a reader to close its T2
#   coverage gap with LZ_NORM_INT_GATE_MODEL=/e/LLM/models/kunmoe-v2-t2- (historical)
#   probe. Also gone. The gap stayed open because the way out of it did
#   not resolve.
#
# WHY A SKIP IS THE DANGEROUS OUTCOME here rather than the safe one:
# a missing toolchain and a rotted fixture name produce the SAME skip,
# and only one of them is a legitimate reason not to run. This gate
# separates them, so "no ARM cross-compiler on this box" stays a skip
# and "the fixture moved" becomes a failure.
#
# SCOPE: paths under a models root, in build/ and src/. Comments count -
# a comment that tells the next person which model to use is an
# instruction, and a wrong one costs them the same time it cost here.
#
# HISTORICAL RECORDS ARE EXEMPT, by prefix on the same line: a note that
# says something WAS measured on a checkpoint is a fact about the past
# and stays true after the file is deleted. Mark those `(historical)`.
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP (no models root on this machine).
set -u
cd "$(dirname "$0")/.." || exit 1

MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
if [ ! -d "$MODELS_ROOT" ]; then
    echo "fixture_exists_gate: SKIP - no models root at $MODELS_ROOT," \
         "so no fixture name could be resolved either way"
    exit 2
fi

PY=python; command -v python >/dev/null 2>&1 || PY=python3
MODELS_ROOT="$MODELS_ROOT" "$PY" - <<'PYEOF'
import os, re, sys, glob

root = os.environ["MODELS_ROOT"]
pat = re.compile(r"(?:/e/LLM/models|\$MODELS_ROOT|\$\{MODELS_ROOT\})/([A-Za-z0-9._-]+)")

files = sorted(glob.glob("build/**/*.sh", recursive=True)
               + glob.glob("build/**/*.c", recursive=True)
               + glob.glob("src/*.c") + glob.glob("src/*.h"))

missing, checked, exempt = [], 0, 0
for f in files:
    for i, line in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
        for m in pat.finditer(line):
            name = m.group(1)
            # A shell variable spliced into the path is not a name this
            # gate can resolve; skip rather than invent a failure.
            if "$" in name:
                continue
            if "(historical)" in line:
                exempt += 1
                continue
            checked += 1
            if not os.path.isdir(os.path.join(root, name)):
                missing.append((f, i, name))

if missing:
    print("fixture_exists_gate: FAIL - named but not on disk:")
    for f, i, name in missing:
        print("    %s:%d  %s" % (f, i, name))
    print("  A missing fixture does not fail the gate that wants it - it")
    print("  SKIPs it, and a skip reads as 'not applicable here' rather")
    print("  than 'this check has stopped happening'. Point it at a")
    print("  checkpoint that exists, or mark the line (historical) if it")
    print("  is a record of a past measurement rather than an instruction.")
    sys.exit(1)

print("fixture_exists_gate: PASS (%d fixture reference(s) resolve, "
      "%d marked historical)" % (checked, exempt))
sys.exit(0)
PYEOF
