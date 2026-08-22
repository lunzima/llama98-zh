#!/bin/bash
# Build and count every arm of .prof/arm_prim_count.c, both tiers.
#
# The count is executed instructions, not cycles: qemu-arm-static here
# has no TCG plugin support, so it comes from -one-insn-per-tb plus
# `-d exec,nochain` - one log line per instruction executed, program
# stdout discarded. Three judgements, none optional:
#   zero control  n=0 must agree line-for-line across every arm
#   linearity     X(512)-X(256) must equal X(256)-X(0) for the arm to
#                 have a per-call cost at all
#   real inputs   the argument sets come from arm_bias_dist.py's hooks
set -e
R=${R:-/mnt/e/llama98-zh}
O=${O:-/tmp/primc}
CC=${CC:-arm-linux-gnueabi-gcc-14}
Q=${Q:-qemu-arm-static}
F="-O2 -std=c99 -march=armv5te -mtune=arm926ej-s -mfloat-abi=soft"
F="$F -DLZ_NO_FPU=1 -ffp-contract=off -ffunction-sections -fdata-sections"

mkdir -p "$O"; rm -f "$O"/*.o "$O"/*.elf
OBJ=""
for f in ops ops_quant ops_sched ops_norm ops_gdn ops_matmul ops_t2_arm \
         ops_t2_scalar ops_arm fwht cpucheck err; do
    $CC $F -I"$R/src" -c "$R/src/$f.c" -o "$O/$f.o"
    OBJ="$OBJ $O/$f.o"
done

build() {   # build <mode> <tier> <intarg>
    $CC $F -I"$R/src" -I"$R/.prof" \
        -DPRIM_MODE=$1 -DPRIM_TIER=$2 -DPRIM_INTARG=$3 \
        -static -Wl,--gc-sections \
        "$R/.prof/arm_prim_count.c" $OBJ -lm -o "$O/m$1_t$2.elf"
}
for t in 0 1; do
    build 1 $t 0
    build 2 $t 0
    build 3 $t 0
    build 4 $t 0
    build 5 $t 0
    build 6 $t 0
    build 7 $t 0
    build 8 $t 0
    build 9 $t 0
    build 10 $t 0
    build 11 $t 0
    build 12 $t 0
    build 13 $t 0
    build 14 $t 0
    build 15 $t 0
    build 0 $t 0
    mv "$O/m0_t$t.elf" "$O/c0_t$t.elf"
    build 0 $t 1
    mv "$O/m0_t$t.elf" "$O/c1_t$t.elf"
done

count() { $Q -one-insn-per-tb -d exec,nochain "$1" "$2" 2>&1 >/dev/null | wc -l; }

# The attention arms run a whole 64-wide head over 256 rows per call -
# about 40k instructions each - so they are counted at 16/32 rather than
# 256/512. Same three judgements, smaller window.
#
# FOUR points, not three. The first window (0->N) carries every
# one-time table build - sig_build alone is 335,919 instructions - so
# gap1 = (N->2N) - (0->N) is NOT a linearity failure, it is that build
# leaving the measurement. What has to be flat is the SECOND pair:
# gap2 = (2N->3N) - (N->2N), and the per-call number below is taken
# from the N->2N window, whose steady state gap2 is what certifies.
printf '%-10s %10s %12s %12s %12s %10s %9s %9s\n' \
       arm n=0 n=N n=2N n=3N per-call gap1 gap2
for t in 0 1; do
    for e in m1 m2 m3 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15 c0 c1; do
        f="$O/${e}_t$t.elf"
        case $e in
        m5|m6) N=16 ;;
        m9|m10|m11|m12|m13|m14|m15) N=64 ;;
        *) N=256 ;;
        esac
        a=$(count "$f" 0); b=$(count "$f" "$N")
        c=$(count "$f" $((N * 2))); d=$(count "$f" $((N * 3)))
        printf '%-10s %10d %12d %12d %12d %10.2f %9d %9d\n' \
            "${e}_t$t" "$a" "$b" "$c" "$d" \
            "$(echo "scale=4; ($c - $b) / $N" | bc)" \
            "$(( (c-b) - (b-a) ))" "$(( (d-c) - (c-b) ))"
    done
done
