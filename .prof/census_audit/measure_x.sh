#!/bin/bash
# measure.sh against a covbuild_x.sh arm: same objdump x gcov product,
# reading build/cov_$TAG instead of build/cov. Reads and measures only.
#   TAG=vproj0 bash measure_x.sh /tmp/measured_vproj0.tsv
set -e
cd "$(dirname "$0")"
OUT=${1:-/tmp/measured.tsv}
TAG=${TAG:-x}
COV="build/cov_$TAG"
ENG="err compat lfn json safetensors model ops ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn fwht forward sampler generate tokenizer unicode chat gbk cpucheck ops_mmx ops_mmx_sse ops_sse2"
: > "$OUT"
for tu in $ENG; do
    [ -f "$COV/$tu.gcda" ] || continue
    objdump -dlr --inlines "$COV/$tu.o" > "/tmp/_dis.$TAG.$tu.txt" 2>/dev/null
    awk -f dis2lines.awk "/tmp/_dis.$TAG.$tu.txt" > "/tmp/_ins.$TAG.$tu.tsv"
    gcov -a -t -o "$COV" "src/$tu.c" > "/tmp/_gcov.$TAG.$tu.txt" 2>/dev/null
    awk -v tu="$tu" -v insfile="/tmp/_ins.$TAG.$tu.tsv" -f joinall.awk \
        "/tmp/_ins.$TAG.$tu.tsv" "/tmp/_gcov.$TAG.$tu.txt" >> "$OUT"
done
echo "rows: $(wc -l < "$OUT")  -> $OUT"
