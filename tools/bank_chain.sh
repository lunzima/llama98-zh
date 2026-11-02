#!/bin/sh
# Rebuild banks/picked.ksb from the eleven source DLS banks, end to end.
#
# WHY THIS FILE EXISTS. pick_bank.sh records the SELECTION and says, in
# its own words, that "the calibration above it lives in the session's own
# tooling" - that tooling had no record anywhere, and re-running the
# selection with a different pass order produced 3468 regions where the
# shipped bank has 4621. A chain nobody can write down twice is not a
# chain, so this file is the rest of it: audit, passes, acceptance, each
# named with its inputs and its outputs.
#
# WHAT IT IS NOT. It does not reproduce banks/picked.ksb. That bank's
# pass count and order were never recorded; the shipped bank is treated as
# an artifact, not as a target. What this reproduces is ITSELF: run it
# twice on the same inputs and the five passes produce byte-identical
# output (sha1 63a91b0ab9afd3e8 on 2026-11-02).
#
# STEPS AND WHY THE ORDER
#   1. select   tools/pick_bank.sh's command, verbatim. Produces the
#               selection-stage bank; its own --calibrate settles levels
#               against gm.dls.
#   2. audit    corpus_report.py pointed at THAT bank. The next step reads
#               key_devs, which are per-bank; another bank's audit moved
#               18 regions where this one moved 1148.
#   3. passes   tools/bank_passes.py, in the order pick_bank.sh names.
#   4. check    structure and, on a run where the acceptance matters,
#               the per-song spread against a fixed baseline.
#
# USAGE
#   sh tools/bank_chain.sh [OUT.ksb] [ROWS.json] [BASELINE.txt]
#     OUT.ksb     default banks/picked.ksb  (back it up first)
#     ROWS.json   default build/tmp/corpus_rows.json (step 2 writes it)
#     BASELINE    optional; when given, step 4 compares against it
set -e
cd "$(dirname "$0")/.."
OUT=${1:-banks/picked.ksb}
ROWS=${2:-build/tmp/corpus_rows.json}
BASE=${3:-}
T=build/tmp
AUDIT="$T/audit_$(basename "$OUT" .ksb)"

echo "== 1. select   -> $T/selected.ksb"
sh tools/pick_bank.sh "$T/selected.ksb"

echo "== 2. audit    -> $ROWS"
mkdir -p "$AUDIT"
CORPUS_REPORT_BANK="$T/selected.ksb" CORPUS_REPORT_OUT="$AUDIT" \
  python3 build/tmp/corpus_report.py
cp "$AUDIT/corpus_rows.json" "$ROWS"

echo "== 3. passes   -> $OUT"
python3 tools/bank_passes.py "$T/selected.ksb" "$ROWS" "$OUT"

echo "== 4. structure"
python3 - "$OUT" <<'PY'
import struct, sys
d = open(sys.argv[1], 'rb').read()
n = struct.unpack_from('<H', d, 4)[0]
io, ro = 16, 16 + n * 7
regs = sum(struct.unpack_from('<BBBHH', d, io + i * 7)[3] for i in range(n))
print('   %d bytes, %d instruments, %d regions' % (len(d), n, regs))
PY

if [ -n "$BASE" ]; then
  echo "== 4b. acceptance against $BASE"
  python3 build/tmp/balance/corpus_levels.py "$OUT" \
    build/tmp/levels_new.txt
  python3 build/tmp/balance/level_ab.py "$BASE" build/tmp/levels_new.txt
fi
