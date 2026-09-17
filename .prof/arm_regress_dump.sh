#!/bin/bash
# ARM-side logits regression: one tree against another, byte for byte,
# over every model and every ARM tier.
#
# --dump-all-logits, not --dump-logits. The latter writes ONE position,
# so a kernel whose effect on the last token happens to cancel reads as
# "no change" - which is what arm_tier_proof.sh's sig row was reporting.
# Every position costs 128 KB and buys a predicate that samples the
# whole sequence.
#
#   bash .prof/arm_regress_dump.sh <baseline-tree> <current-tree>
#
# Run from WSL. The two trees must differ ONLY in what is under test -
# building the baseline from `git archive HEAD` alone is not enough when
# other sessions have uncommitted files in the shared workspace, so the
# caller is expected to have overlaid those onto the baseline first.
#
# DO NOT PIN FROM --pin-tiers HERE. That option prints the tiers as they
# stand before a model is loaded, and on the ARM build it says
# "--sig-write float --norm-write float" while the same binary, left on
# auto, resolves BOTH to fixed once lz_soft_float_target() is consulted.
# Pinning from it therefore turns off exactly the two tiers this gate
# exists to compare - a gate that cannot run on the artifact. The runs
# below are left on auto and the resolved tier block is captured from
# each log as the evidence that fixed is what actually ran.
#
# (build/arm/parity_gate.sh pinning from the X86 binary is a different
# question and stays correct: it needs the SAME tier on both sides, and
# float is a tier both sides have.)
set -u
BASE=${1:?baseline tree}
CUR=${2:?current tree}
Q=${Q:-qemu-arm-static}
SYSROOT=${SYSROOT:-/usr/arm-linux-gnueabi}
CC=${CC:-arm-linux-gnueabi-gcc-14}
TOK=${TOK:-1,2,3,4,5,6,7,8,9,10}
# This list was three models and TWO OF THEM ARE GONE: _armgate2 (see
# docs/dsv41-optimization-menu.md section 6) and kunkun98-pilot-20260806,
# the checkpoint _armgate2 was exported from - the whole training chain
# that produced it is gone too. Both are replaced by _armgate3 where a
# replacement exists; kunkun98-pilot-20260806 has none, because nothing
# on disk is a 12-layer GDN at that geometry. So the list is shorter than
# it was and the gap is not filled by anything.
MODELS=${MODELS:-"/mnt/e/LLM/models/_armgate3 /mnt/e/LLM/models/kunkun98"}
W=${W:-/tmp/armreg}

rm -rf "$W"; mkdir -p "$W"
for t in base cur; do
    eval "src=\$$(echo "$t" | tr a-z A-Z)"
    echo "-- cross-building $t ($src) --"
    CC="$CC" OUT="$W/$t" bash "$src/build/arm/build_engine.sh" \
        >"$W/$t.build.log" 2>&1 || { tail -20 "$W/$t.build.log"; exit 1; }
done

fail=0
for m in $MODELS; do
    name=$(basename "$m")
    for k in ref arm-c arm-asm; do
        for t in base cur; do
            $Q -cpu arm926 -L "$SYSROOT" "$W/$t/llama98-arm" "$m" \
               --kernel "$k" --tokens "$TOK" \
               --dump-all-logits "$W/$name.$k.$t.bin" \
               >"$W/$name.$k.$t.log" 2>&1 || {
                echo "RUN FAIL $name $k $t"; tail -5 "$W/$name.$k.$t.log"
                fail=1; continue; }
        done
        if cmp -s "$W/$name.$k.base.bin" "$W/$name.$k.cur.bin"; then
            printf 'IDENTICAL  %-26s %-8s %s bytes  md5=%s\n' "$name" "$k" \
                "$(wc -c < "$W/$name.$k.cur.bin")" \
                "$(md5sum < "$W/$name.$k.cur.bin" | cut -c1-8)"
        else
            printf 'DIFFER     %-26s %-8s\n' "$name" "$k"
            fail=1
        fi
    done
    # The three tiers must also agree with EACH OTHER - that is the
    # bit-identity contract the assembly owes, and it is what makes
    # "identical to baseline" mean more than "identically declined".
    if cmp -s "$W/$name.ref.cur.bin" "$W/$name.arm-asm.cur.bin" &&
       cmp -s "$W/$name.ref.cur.bin" "$W/$name.arm-c.cur.bin"; then
        printf '  tiers agree: ref == arm-c == arm-asm\n'
    else
        printf '  TIERS DISAGREE on %s\n' "$name"
        fail=1
    fi
    printf '  tiers actually run: %s\n' \
        "$(sed -n '/^  sigmoid.silu/p;/^  norms /p' \
             "$W/$name.arm-asm.cur.log" | tr -s ' \n' ' ')"
done
exit $fail
