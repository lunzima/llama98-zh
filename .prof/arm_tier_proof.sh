#!/bin/bash
# "The tier is really reached" - at the PRODUCT, not at the leaf gate.
#
#   bash .prof/arm_tier_proof.sh <tree> <reference-dump-dir>
#
# Breaks ONE instruction in each new hand-written kernel, rebuilds the
# whole ARM engine, and dumps logits for the same model on all three ARM
# tiers. The claim being tested is not "the assembly is right" - the leaf
# gate answers that - it is "--kernel arm-asm executes this code and the
# other two tiers do not". So the pass condition is asymmetric:
#
#   arm-asm  MUST differ from the reference dump
#   arm-c    MUST be byte-identical
#   ref      MUST be byte-identical
#
# A leaf gate can be green because a tier branch never fires. This is the
# check that branch exists in the shipped binary.
#
# Every position, not the last one: --dump-logits writes a single vector,
# and a mutation whose effect on that one token cancels reads as "the
# tier was never reached". Must match arm_regress_dump.sh, which writes
# the reference dumps this compares against.
#
# Tiers are left on AUTO: on ARM that resolves sigmoid/norms to fixed,
# which is where these two kernels live. --pin-tiers would say float and
# turn them both off - see arm_regress_dump.sh's header.
set -u
SRC=${1:?tree to mutate}
REF=${2:?directory holding <model>.<kernel>.cur.bin}
Q=${Q:-qemu-arm-static}
SYSROOT=${SYSROOT:-/usr/arm-linux-gnueabi}
CC=${CC:-arm-linux-gnueabi-gcc-14}
MODEL=${MODEL:-/mnt/e/LLM/models/_armgate2}
TOK=${TOK:-1,2,3,4,5,6,7,8,9,10}
W=${W:-/tmp/armproof}
NAME=$(basename "$MODEL")

# THE sig ROW IS BAD BECAUSE THE MUTATION IS OFF THE PATH, ANSWERED.
# Breaking lz_sigmoid_q15's coordinate shift leaves _armgate2's logits
# byte-identical across all nine positions, so arm-asm reads "same"
# where it must read "differ". That is not a sampling artifact - the
# model has linear attention layers (9 of 12) and widening from one
# position to all nine did not move it - and it is not a broken gate.
#
# The two ARM sigmoid primitives are reached from exactly two call
# sites, both in src/ops_quant.c, and both hand them a FLOAT: `bx.f = x`
# into lz_sig_q15_arm_asm, `bt.f = t` into lz_sig_q15_t_arm_asm. They
# are the ARM tier of the float sigmoid.
#
# sigmoid_q15_i is the INTEGER entry and it calls neither. It reaches
# the float one only through its own out-of-range arm, `e < 0 ||
# e > LZ_SIG_EMAX`; inside that range it does its own table lookup and
# interpolation in C. The int pipeline runs it in range - the CLI
# banner's "conv sig int" and "gate sig int" channel-step counts are
# what say so - so the mutated instruction never executes and identical
# logits are the correct answer.
#
# WHAT THAT LEAVES. The row is measuring a primitive the shipping
# integer path does not call, which makes it a product-level proof of
# nothing rather than a failing one. Fixing it means mutating what
# sigmoid_q15_i actually executes, not this. The leaf gate still covers
# the primitive's own value.
MUT="sig|src/ops_arm_prim.h|s/rsb    %\[e\], %\[e\], #135/rsb    %[e], %[e], #134/
norm|src/ops_arm.c|s/andeq  r7, r9, #1/andeq  r7, r9, #0/
epi|src/ops_arm.c|s/smlal  %\[lo\], %\[hi\], r5, r9/smlal  %[lo], %[hi], r5, r8/"

rm -rf "$W"; mkdir -p "$W"
fail=0
while IFS='|' read -r tag file expr; do
    [ -n "$tag" ] || continue
    T="$W/$tag"
    rm -rf "$T"; mkdir -p "$T"
    # The whole tree, not just src/ and build/: cli_main.c includes
    # common/stream.h and the first version of this script found that
    # out as a link failure, which proves nothing at all.
    (cd "$SRC" && tar -cf - --exclude=.git --exclude=.prof .) | \
        (cd "$T" && tar -xf -)
    sed -i "$expr" "$T/$file" || { echo "$tag: sed failed"; fail=1; continue; }
    if cmp -s "$SRC/$file" "$T/$file"; then
        echo "$tag: sed matched nothing - the pattern is stale"; fail=1; continue
    fi
    if ! CC="$CC" OUT="$T/out" bash "$T/build/arm/build_engine.sh" \
            >"$T/build.log" 2>&1; then
        echo "$tag: ARM build failed"; tail -5 "$T/build.log"; fail=1; continue
    fi
    for k in ref arm-c arm-asm; do
        $Q -cpu arm926 -L "$SYSROOT" "$T/out/llama98-arm" "$MODEL" \
           --kernel "$k" --tokens "$TOK" --dump-all-logits "$T/$k.bin" \
           >"$T/$k.log" 2>&1 || { echo "$tag/$k: run failed"; fail=1; continue; }
        want=same; [ "$k" = arm-asm ] && want=differ
        if cmp -s "$REF/$NAME.$k.cur.bin" "$T/$k.bin"; then got=same; else got=differ; fi
        md5=$(md5sum < "$T/$k.bin" | cut -c1-8)
        if [ "$got" = "$want" ]; then
            printf 'OK    %-6s %-8s %-7s md5=%s\n' "$tag" "$k" "$got" "$md5"
        else
            printf 'BAD   %-6s %-8s %-7s md5=%s  (wanted %s)\n' \
                   "$tag" "$k" "$got" "$md5" "$want"
            fail=1
        fi
    done
done <<EOF
$MUT
EOF
exit $fail
