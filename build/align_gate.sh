#!/bin/bash
# Does the engine perform a misaligned load or store on real input?
#
# WHY THE MACHINE AND NOT A SURVEY. -Wcast-align=strict is blind on this
# tree: nearly every buffer reaches a kernel through void *, so the cast
# whose alignment it would check is not written where the compiler can
# see it. A survey of the pointer casts ends at one invariant - gs and
# in_dim are multiples of 32 - asserted at five sites and assumed
# everywhere else. UBSan checks the access itself.
#
# WHY IT MATTERS OFF x86. An unaligned load is legal on x86 and merely
# slower. On ARMv5TE it returns a ROTATED word: the right bytes in the
# wrong order, no fault, a wrong number. On the MIPS target it faults.
# Neither appears in any x86 run, which is every run this project makes
# by default.
#
# WHAT IT DOES NOT PROVE. This is the x86-64 Linux ABI's alignment
# requirements, checked on x86. A target with stricter ones, or with
# different struct packing - Watcom caps member alignment at 4 with -zp4,
# gcc does not - can still differ. It answers "does the C dereference a
# pointer that is not aligned for its type", which is the portable half
# of the question and the half no other gate here asks at all.
#
# EVERY TIER IS RUN, ref through sse2, out of one instrumented build.
# The tier decides which C executes, not only which kernel: on kmr20
# --kernel ref never enters the fixed epilogue, so a ref-only run leaves
# the int pipeline's prologues, tails and epilogues unmeasured and still
# prints zero. Four runs of one binary cost four forward passes and
# close that.
#
# What is still NOT measured is the SIMD intrinsics themselves, and that
# is a property of the instrumentation rather than a choice:
# _mm_loadu_si128 takes a pointer type declared __aligned__(1), so UBSan
# has nothing to check there and is right not to. What the tiers buy is
# the scalar C on either side of those loads.
#
# The SIMD units are always linked. Dropping them does not link, because
# LZ_MMX_TU=0 leaves the dispatch tables referring to dot32_q41_mmx and
# its siblings.
#
# Runs under WSL, like the MIPS and x87 stages: MinGW has no UBSan.
#
# Exit codes:
#   0  PASS - no misaligned access
#   1  FAIL - at least one, or a control failed
#   2  SKIP - no WSL, no gcc there, or no model fixture
set -u
cd "$(dirname "$0")/.."

MODEL=${1:-}
[ -n "$MODEL" ] || MODEL=/mnt/e/LLM/models/kmr20
TOK=build/norm_int_gate_tokens_zh.txt

command -v wsl >/dev/null 2>&1 || { echo "align_gate: SKIP - no wsl"; exit 2; }
ROOT=$(pwd)
case "$ROOT" in
    /[a-z]/*) WSL_ROOT="/mnt${ROOT}" ;;
    *)        WSL_ROOT="$ROOT" ;;
esac

OUT=/tmp/lz_align_gate
LOG=build/align-gate.log

# The source list comes from make, not from a glob: this is only worth
# anything if it compiles the program the Makefile compiles, and a glob
# picks up files that sit in src/ but are in no target. Computed HERE,
# on the Windows side - the Makefile does not run under WSL, and asking
# it there returns an empty list that would compile nothing and report a
# clean run.
SRCS=$(make -Bn llama98 2>/dev/null \
       | grep -oE 'src/[a-z0-9_]+[.]c|common/[a-z0-9_]+[.]c' \
       | sort -u | tr '\n' ' ')
if [ -z "$SRCS" ]; then
    echo "align_gate: FAIL - make named no sources for llama98"
    exit 1
fi

# The WSL side is a FILE, not a quoted string. An earlier draft inlined
# it into `wsl bash -lc "..."` and the nested quoting swallowed the
# command substitution that counts the instrumentation: SITES came back
# empty on a build that in fact had 5465 checks in it, and the gate
# reported its own positive control as a failure. Two layers of shell
# quoting is not something to get right by inspection.
# In the repo, not in /tmp: Git Bash's /tmp is a Windows path that WSL
# cannot see, so `wsl bash /tmp/x.sh` fails with "No such file". build/
# is visible from both sides, which is the whole reason this file is
# written rather than piped.
STEP=build/align-step.sh
{
    echo "set -u"
    echo "cd '$WSL_ROOT' || exit 2"
    echo "command -v gcc >/dev/null 2>&1 || { echo NOGCC; exit 2; }"
    echo "[ -d '$MODEL' ] || { echo NOMODEL; exit 2; }"
    echo "gcc -O1 -g -std=c99 -Isrc -Icommon -fsanitize=alignment \\"
    echo "  -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -mmmx -msse -msse2 \\"
    echo "  -o $OUT $SRCS -lm 2>$OUT.build.log || {"
    echo "      echo BUILDFAIL; tail -5 $OUT.build.log; exit 1; }"
    # The instrumentation count is the control that runs every time: a
    # build that silently lost -fsanitize=alignment reports zero
    # misaligned accesses and looks exactly like a clean one.
    echo "echo SITES=\$(objdump -d $OUT | grep -c __ubsan_handle_type_mismatch)"
    # EVERY TIER, not just ref. The instrumented build reaches the C
    # around the kernels, and which C that is depends on g_kernel: on
    # kmr20 --kernel ref does not enter the fixed epilogue at all, so a
    # ref-only run leaves the whole int pipeline's tails, prologues and
    # epilogues unmeasured while reporting zero.
    echo "for k in ref mmx sse sse2; do"
    echo "  $OUT '$MODEL' --kernel \$k --tokens-file '$TOK' \\"
    echo "    --dump-all-logits /dev/null >$OUT.\$k.log 2>&1"
    echo "  echo RC_\$k=\$?"
    echo "  echo HITS_\$k=\$(grep -c 'runtime error:.*misaligned' $OUT.\$k.log)"
    echo "  grep -q logits $OUT.\$k.log && echo FORWARDRAN_\$k"
    echo "  grep 'runtime error:.*misaligned' $OUT.\$k.log | sort -u \\"
    echo "    | head -5 | sed \"s/^/[\$k] /\""
    echo "done"
} > "$STEP"

# cd first, then a RELATIVE path. Git Bash rewrites an argument that
# starts with /mnt into a Windows path ("C:/Program Files/Git/mnt/..."),
# so the absolute form reaches WSL as a name that does not exist. The
# other WSL stages here take the same shape for the same reason.
wsl bash -lc "cd '$WSL_ROOT' && bash $STEP" >"$LOG" 2>&1
WRC=$?
rm -f "$STEP"

if grep -q "NOGCC" "$LOG"; then echo "align_gate: SKIP - no gcc in WSL"; exit 2; fi
if grep -q "NOMODEL" "$LOG"; then echo "align_gate: SKIP - no $MODEL"; exit 2; fi
if grep -q "BUILDFAIL" "$LOG"; then
    echo "align_gate: FAIL - the instrumented build did not link"
    grep -E "error" "$LOG" | head -5
    exit 1
fi
[ "$WRC" -eq 2 ] && { echo "align_gate: SKIP - wsl step exited 2"; exit 2; }

SITES=$(grep -m1 '^SITES=' "$LOG" | cut -d= -f2)
[ -n "${SITES:-}" ] || SITES=0
TIERS="ref mmx sse sse2"
HITS=0
for k in $TIERS; do
    h=$(grep -m1 "^HITS_$k=" "$LOG" | cut -d= -f2)
    [ -n "${h:-}" ] || h=0
    HITS=$((HITS + h))
done

# Three controls, because "no reports" is the same output as three
# different kinds of nothing happening.
if [ "$SITES" -lt 1000 ]; then
    echo "align_gate: FAIL - only $SITES alignment checks compiled in;"
    echo "  -fsanitize=alignment is not reaching the engine and a zero"
    echo "  here would mean nothing"
    exit 1
fi
for k in $TIERS; do
    if ! grep -q "FORWARDRAN_$k" "$LOG"; then
        echo "align_gate: FAIL - no logits line for --kernel $k: the forward"
        echo "  pass did not run, so that tier's code was never entered"
        exit 1
    fi
    rc=$(grep -m1 "^RC_$k=" "$LOG" | cut -d= -f2)
    if [ "${rc:-1}" -ne 0 ]; then
        echo "align_gate: FAIL - the instrumented run exited ${rc:-?} on --kernel $k"
        tail -5 "$LOG"
        exit 1
    fi
done

if [ "$HITS" -eq 0 ]; then
    echo "align_gate: PASS (0 misaligned across $(echo $TIERS | wc -w) tier(s),"
    echo "  $SITES checks compiled in)"
    exit 0
fi
echo "align_gate: FAIL - $HITS misaligned access(es)"
grep "runtime error" "$LOG" | head -8 | sed 's/^/    /'
exit 1
