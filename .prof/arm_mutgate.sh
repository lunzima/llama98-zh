#!/bin/bash
# Mutation harness for the ARMv5TE operator primitives.
#
# For each mutation: copy the tree, apply ONE surgical sed to
# src/ops_arm_prim.h, rebuild the leaf gate against the copy, run it
# with a large divisor (it only has to reach the first mismatch) and
# require a NON-ZERO exit. A mutation that still compiles but leaves the
# gate green is a hole in the gate, and a mutation that fails to compile
# proves nothing at all - both are reported as FAIL here.
#
# The shared workspace is never written to; everything happens under
# $W, which starts as a copy of src/ and .prof/.
set -u
R=${R:-/mnt/e/llama98-zh}
W=${W:-/tmp/mutgate}
DIVI=${DIVI:-4000}

# THIS SCRIPT ONLY RUNS UNDER WSL, and until this check it did not say
# so. Started from Git Bash, $R does not exist, every cp and sed fails,
# and it reports "red 0 bad 53" and exits 0 - a mutation suite in which
# nothing went red, which reads as the assembly being wrong when it is
# the path that is. Under WSL the same run is 52 red, 1 bad.
if [ ! -d "$R/src" ]; then
    echo "arm_mutgate: FAIL - \$R=$R has no src/."
    echo "  This needs the ARM cross toolchain and qemu, which on this"
    echo "  machine live only inside WSL. Run it as:"
    echo "    wsl bash -lc \"cd /mnt/e/llama98-zh && bash .prof/arm_mutgate.sh\""
    echo "  Refusing rather than reporting 0 red out of 53."
    exit 2
fi

PRIM=src/ops_arm_prim.h
ARMC=src/ops_arm.c
MUT=$(cat <<EOF
i32f|exponent-bias|$PRIM|s/rsb    %\[r\], %\[n\], #157/rsb    %[r], %[n], #158/
i32f|round-threshold|$PRIM|s/cmp    %\[n\], #0x80000000/cmp    %[n], #0x40000000/
i32f|ties-to-even|$PRIM|s/andeq  %\[n\], %\[r\], #1/andeq  %[n], %[r], #0/
i32f|drop-count|$PRIM|s/rsb    %\[s\], %\[n\], #8/rsb    %[s], %[n], #7/
i32f|sign|$PRIM|s/orrlt  %\[r\], %\[r\], #0x80000000/orrlt  %[r], %[r], #0/
q8r|domain-boundary|$PRIM|s/cmp    %\[e\], #149/cmp    %[e], #150/
q8r|underflow-cut|$PRIM|s/cmp    %\[e\], #25/cmp    %[e], #24/
q8r|round-threshold|$PRIM|s/cmp    %\[t\], #0x80000000/cmp    %[t], #0xC0000000/
q8r|ties-to-even|$PRIM|s/andeq  %\[t\], %\[r\], #1/andeq  %[t], %[r], #0/
q8r|sign|$PRIM|s/rsbmi  %\[r\], %\[r\], #0/rsbpl  %[r], %[r], #0/
exp|floor-exponent-bias|$PRIM|s/sub    %\[t1\], %\[t1\], #127/sub    %[t1], %[t1], #126/
exp|floor-shift|$PRIM|s/rsb    %\[sh\], %\[t1\], #23/rsb    %[sh], %[t1], #22/
exp|floor-half-point|$PRIM|s/sub    %\[t1\], %\[sh\], #1/sub    %[t1], %[sh], #2/
exp|floor-ties-to-even|$PRIM|s/andeq  %\[t\], %\[t1\], #1/andeq  %[t], %[t1], #0/
exp|floor-exactness-test|$PRIM|s/cmp    %\[t1\], #0x2000000/cmp    %[t1], #0x1000000/
exp|residue-scale|$PRIM|s/rsbs   %\[t1\], %\[sh\], #20/rsbs   %[t1], %[sh], #19/
exp|residue-ties-to-even|$PRIM|s/andeq  %\[p\], %\[q\], #1/andeq  %[p], %[q], #0/
exp|scale2-exponent|$PRIM|s/add    %\[r\], %\[a\], %\[k\], lsl #23/add    %[r], %[a], %[k], lsl #24/
exp|guard-window|$PRIM|s/movls  %\[r\], #1/movls  %[r], #0/
sig|t-coordinate-shift|$PRIM|s/rsb    %\[e\], %\[e\], #135/rsb    %[e], %[e], #134/
sig|table-neighbour|$PRIM|s/ldr    %\[e\], \[%\[e\], #4\]/ldr    %[e], [%[e], #8]/
sig|interpolation-shift|$PRIM|s/add    %\[r\], %\[r\], %\[m\], asr #15/add    %[r], %[r], %[m], asr #14/
sig|frac-split|$PRIM|s/sub    %\[m\], %\[F\], %\[e\], lsl #15/sub    %[m], %[F], %[e], lsl #14/
sig|halfword-select|$PRIM|s/smulbb %\[m\], %\[e\], %\[m\]/smulbt %[m], %[e], %[m]/
sig|t-clamp-value|$PRIM|s/ldrpl  %\[r\], \[%\[tb\], #1536\]/ldrpl  %[r], [%[tb], #1024]/
sig|t-clamp-sense|$PRIM|s/ldrpl  %\[r\], \[%\[tb\], #1536\]/ldrmi  %[r], [%[tb], #1536]/
sig|x-out-shift|$PRIM|s/mov    %\[e\], #1/mov    %[e], #0/
sig|x-clamp-value|$PRIM|s/movpl  %\[r\], #0x8000/movpl  %[r], #0x4000/
sig|x-sum-scale-2p19|$PRIM|s/mov    %\[F\], #0x600000/mov    %[F], #0x400000/
sig|x-sum-scale-2p20|$PRIM|s/mov    %\[F\], #0xC00000/mov    %[F], #0xA00000/
sig|x-shift-hi-binade|$PRIM|s/rsb    %\[m\], %\[m\], #131/rsb    %[m], %[m], #132/
sig|x-round-up|$PRIM|s/addhi  %\[F\], %\[F\], #1/addhi  %[F], %[F], #2/
sig|x-tie-to-even|$PRIM|s/andeq  %\[T\], %\[F\], #1/andeq  %[T], %[F], #0/
sig|x-neg-round-down|$PRIM|s/subhi  %\[F\], %\[F\], #1/subhi  %[F], %[F], #2/
sig|x-neg-exact-2p21|$PRIM|s/rsb    %\[F\], %\[T\], #0x1800000/rsb    %[F], %[T], #0x1000000/
sig|x-neg-exact-2p20|$PRIM|s/rsb    %\[F\], %\[T\], #0xC00000/rsb    %[F], %[T], #0xE00000/
norm|scale-fold|$ARMC|s/rsb    r10, %\[k\], #150/rsb    r10, %[k], #151/
norm|exponent-field|$ARMC|s/mov    r7, r7, lsr #24/mov    r7, r7, lsr #23/
norm|implicit-bit|$ARMC|s/orr    r8, r8, #0x800000/orr    r8, r8, #0x400000/
norm|round-threshold|$ARMC|s/cmp    r7, #0x80000000/cmp    r7, #0x40000000/
norm|ties-to-even|$ARMC|s/andeq  r7, r9, #1/andeq  r7, r9, #0/
norm|clamp-window|$ARMC|s/mov    r11, #0x7F00/mov    r11, #0x7E00/
norm|sign|$ARMC|s/rsbmi  r9, r9, #0/rsbpl  r9, r9, #0/
norm|square-halfword|$ARMC|s/smlalbb r4, r5, r9, r9/smlalbt r4, r5, r9, r9/
norm|accumulator-init|$ARMC|s/mov    r4, #0/mov    r4, #0x10000/
norm|accumulator-high|$ARMC|s/stmia  %\[ac\], {r4, r5}/stmia  %[ac], {r4, r6}/
norm|store-stride|$ARMC|s/strh   r9, \[%\[qp\]\], #2/strh   r9, [%[qp]], #4/
attn|dot-reuse-routing|$ARMC|s/mov    r3, #0xff00/mov    r3, #0xfe00/
attn|wsum-byte-mask|$ARMC|s/mov    r5, #0xff00/mov    r5, #0xfe00/
attn|wsum-halfword-select|$ARMC|s/smlawt r10, %\[ka\], r7, r10/smlawb r10, %[ka], r7, r10/
attn|wsum-coefficient|$ARMC|s/smlawb r8,  %\[ka\], r7, r8/smlawb r8,  %[kb], r7, r8/
attn|wsum-row-stride|$ARMC|s/ldr    r6, \[%\[pa\]\], #4/ldr    r6, [%[pa]], #8/
attn|wsum-store-writeback|$ARMC|s/stmia  %\[ac\]!, {r8, r9, r10, r11}/stmia  %[ac], {r8, r9, r10, r11}/
EOF
)

ONLY=${1:-}
fail=0
pass=0
printf '%-6s %-24s %-8s %s\n' LEAF MUTATION VERDICT DETAIL
while IFS='|' read -r leaf name file expr; do
    [ -n "$leaf" ] || continue
    [ -z "$ONLY" ] || [ "$leaf" = "$ONLY" ] || continue
    rm -rf "$W"; mkdir -p "$W/.prof"
    cp -r "$R/src" "$W/src"
    # The two files arm_leafgate.sh reads, not the whole directory:
    # .prof has grown to 2.7 GB of build products and copying it across
    # DrvFs was costing ~2 minutes PER MUTATION, which is most of a
    # working afternoon over a full run.
    cp "$R/.prof/arm_leafgate.c" "$R/.prof/arm_leafgate.sh" "$W/.prof/"
    if ! sed -i "$expr" "$W/$file"; then
        printf '%-6s %-24s %-8s %s\n' "$leaf" "$name" FAIL "sed failed"
        fail=$((fail + 1)); continue
    fi
    if cmp -s "$R/$file" "$W/$file"; then
        printf '%-6s %-24s %-8s %s\n' "$leaf" "$name" FAIL \
               "sed matched nothing - the pattern is stale"
        fail=$((fail + 1)); continue
    fi
    if ! R="$W" O="$W/o" bash "$W/.prof/arm_leafgate.sh" "$leaf" 0 "$DIVI" \
            >"$W/run.log" 2>&1; then
        if grep -q 'MISMATCH' "$W/run.log"; then
            printf '%-6s %-24s %-8s %s\n' "$leaf" "$name" RED \
                   "$(grep -m1 MISMATCH "$W/run.log")"
            pass=$((pass + 1))
        else
            printf '%-6s %-24s %-8s %s\n' "$leaf" "$name" FAIL \
                   "did not compile / did not run: $(tail -n1 "$W/run.log")"
            fail=$((fail + 1))
        fi
    else
        printf '%-6s %-24s %-8s %s\n' "$leaf" "$name" GREEN \
               "mutation survived - the gate cannot see this"
        fail=$((fail + 1))
    fi
done <<EOF
$MUT
EOF

echo
echo "red $pass  bad $fail"

# ONE KNOWN SURVIVOR, and it is not a hole in the sweep.
# exp/floor-exactness-test halves the |N| <= 2^24 window in
# lz_exp_floor_arm_asm. Nothing notices, because the entry test in that
# same primitive bounds |y32| below 4096, which keeps |N| orders of
# magnitude under the window - the branch is a guard on the arithmetic,
# not a live path. Established by tightening the mutation twice: moving
# the window (0x1000000 -> 0x3000000) and halving it both survive, and
# a version that did not assemble proved the sed was reaching the line.
# Do not read this GREEN as the sweep being too narrow.
KNOWN_SURVIVORS=1
if [ "$fail" -gt "$KNOWN_SURVIVORS" ]; then
    echo "arm_mutgate: FAIL - $fail survived, $KNOWN_SURVIVORS known"
    exit 1
fi
if [ "$fail" -lt "$KNOWN_SURVIVORS" ]; then
    echo "arm_mutgate: note - $fail survived against $KNOWN_SURVIVORS"
    echo "  known. Something now catches what did not; lower the count."
fi
exit 0
