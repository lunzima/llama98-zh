#!/bin/bash
# Produces the measured float-operation table for the whole C side of the
# engine from one run of llama98_covall.exe: per (TU, file, line, function,
# mnemonic), static instruction count x gcov execution count.
# Reads and measures only; writes nothing under src/.
set -e
cd "$(dirname "$0")"
OUT=${1:-/tmp/measured.tsv}
ENG="err compat lfn json safetensors model ops ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn fwht forward sampler generate tokenizer unicode chat gbk cpucheck ops_mmx ops_mmx_sse ops_sse2"
: > "$OUT"
for tu in $ENG; do
    [ -f "build/cov/$tu.gcda" ] || continue
    objdump -dlr --inlines "build/cov/$tu.o" > /tmp/_dis.$tu.txt 2>/dev/null
    awk -f dis2lines.awk /tmp/_dis.$tu.txt > /tmp/_ins.$tu.tsv
    gcov -a -t -o build/cov "src/$tu.c" > /tmp/_gcov.$tu.txt 2>/dev/null
    awk -v tu="$tu" -v insfile=/tmp/_ins.$tu.tsv -f joinall.awk \
        /tmp/_ins.$tu.tsv /tmp/_gcov.$tu.txt >> "$OUT"
done
echo "rows: $(wc -l < "$OUT")  -> $OUT"
