#!/bin/bash
# covbuild.sh with extra -D flags, for measuring a control arm's
# conversions rather than only the shipping default's. $XDEFS is appended
# to the same DEFS covbuild.sh uses and the binary name carries $TAG, so
# two arms can sit side by side in one checkout. Reads and measures only.
#   XDEFS="-DLZ_VPROJ_I16=0" TAG=vproj0 bash covbuild_x.sh
set -e
cd "$(dirname "$0")"

XDEFS=${XDEFS:-}
TAG=${TAG:-x}
DEFS="-march=x86-64 -std=c99 -Wall -Wextra -Isrc -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -DLZ_F32COUNT=1 $XDEFS"
ENG="err compat lfn json safetensors model ops ops_quant ops_sched ops_matmul ops_t2_arm ops_arm ops_t2_scalar ops_norm ops_gdn fwht forward sampler generate tokenizer unicode chat gbk cpucheck"

rm -rf "build/cov_$TAG"
mkdir -p "build/cov_$TAG"
OBJS=""
for f in $ENG; do
    gcc -O0 -g --coverage $DEFS -c -o "build/cov_$TAG/$f.o" src/$f.c
    OBJS="$OBJS build/cov_$TAG/$f.o"
done
for f in ops_mmx ops_mmx_sse ops_sse2; do
    gcc -O0 -g --coverage $DEFS -mmmx -msse -msse2 -c -o "build/cov_$TAG/$f.o" src/$f.c
    OBJS="$OBJS build/cov_$TAG/$f.o"
done
gcc -O3 $DEFS -Icommon --coverage -o "llama98_covall_$TAG.exe" \
    $OBJS common/session.c common/stream.c src/cli_attr.c src/cli_main.c -lm
echo "built llama98_covall_$TAG.exe"
