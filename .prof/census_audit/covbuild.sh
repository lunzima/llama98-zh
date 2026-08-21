#!/bin/bash
# Builds llama98 with every engine TU compiled -O0 -g --coverage, so that
# gcov line counts x per-line float instruction counts (objdump -dl) give a
# mechanical count of executed f32 operations. The census flag stays on, so
# one run yields both the LZ_FCX census and the measured counts.
#
# The three SIMD TUs (ops_mmx, ops_mmx_sse, ops_sse2) keep their -O3
# intrinsics build: they are unreachable under --kernel ref, which is the
# configuration this instrument is meant to run in.
set -e
cd "$(dirname "$0")"

DEFS="-march=x86-64 -std=c99 -Wall -Wextra -Isrc -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_F32COUNT=1"
ENG="err compat lfn json safetensors model ops ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn fwht forward sampler generate tokenizer unicode chat gbk cpucheck"

mkdir -p build/cov
OBJS=""
for f in $ENG; do
    gcc -O0 -g --coverage $DEFS -c -o build/cov/$f.o src/$f.c
    OBJS="$OBJS build/cov/$f.o"
done
# The hand-written intrinsic kernels, same flags the Makefile gives them
# plus coverage: their packed ops carry 4 (ps) / 2 (pd) elements each and
# classify.awk weights them, so leaving them out would drop real f32 work.
for f in ops_mmx ops_mmx_sse ops_sse2; do
    gcc -O0 -g --coverage $DEFS -mmmx -msse -msse2 -c -o build/cov/$f.o src/$f.c
    OBJS="$OBJS build/cov/$f.o"
done

# Front end stays -O3: it is not part of the engine's f32 surface and its
# NLL path deliberately uses double.
gcc -O3 $DEFS -Icommon --coverage -o llama98_covall.exe \
    $OBJS common/session.c common/stream.c src/cli_attr.c src/cli_main.c -lm
echo "built llama98_covall.exe"
