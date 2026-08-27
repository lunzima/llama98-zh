#!/bin/bash
# IEEE special-case gate: assert that a real inference run produces no
# NaN and no overflow to Inf.
#
# WHY THIS EXISTS. The ARMv5TE soft-float multiply's fast path spends 12
# of its 44 instructions on cases that a language-level float has to
# handle but this engine's arithmetic never reaches:
#
#   4  inf/nan operand checks
#   4  overflow/underflow exponent checks
#   4  zero/subnormal operand checks
#
# Dropping them is worth 27% of every f32 multiply, and the ARM census
# counts 234k multiplies and 210k adds per token - so it is worth having.
# But it is only SOUND while the cases really never occur, and "never
# occurs" is a property of the model and the fixture, not of the code.
# A new checkpoint whose activations overflow would turn a fast-path
# multiply into a silently wrong one with no crash to notice.
#
# So the precondition gets measured on every run of `make check` rather
# than assumed from the one afternoon it was true.
#
# WHAT IT READS. x86's MXCSR keeps six sticky exception flags that the
# hardware sets for free. --fp-exceptions clears them at startup and
# prints them at exit, so the line describes that one run:
#
#   IE  invalid      - NaN produced (0/0, inf-inf, sqrt of negative)
#   OE  overflow     - a result went to Inf
#   DE  denormal     - a denormal operand was seen
#   ZE  div-by-zero, UE underflow, PE inexact
#
# ASSERTS IE=0 and OE=0. Those two are what the inf/nan and overflow
# checks guard.
#
# DOES NOT ASSERT DE=0, and that is deliberate: DE is 1 on the shipping
# fixture. Denormals DO occur. What was measured is that they do not
# change the answer - setting MXCSR's FTZ|DAZ (which flushes them) leaves
# every one of the 78,905,344 logit bytes identical - so the fast path
# may flush them, but it may not pretend they never arrive.
#
# PE is 1 on every run by construction (nearly every float operation is
# inexact) and says nothing.
#
# COVERAGE. One target, because the claim is about the ARITHMETIC, which
# is shared source. ARM soft-float has no hardware flags to read and x87
# reports the same six through fnstsw but is not wired up; a run there
# prints "not measured" rather than a missing line.
#
# Exit codes: 0 PASS, 1 FAIL, 2 ERROR/SKIP.
set -u
cd "$(dirname "$0")/.." || exit 1

MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
MODEL=${LZ_NORM_INT_GATE_MODEL:-$MODELS_ROOT/kmr20}
if [ ! -d "$MODEL" ]; then
    echo "fp_exception_gate: SKIP - model not found: $MODEL"
    exit 2
fi

BIN=llama98.exe
[ -x "$BIN" ] || BIN=llama98
if [ ! -x "$BIN" ]; then
    echo "fp_exception_gate: ERROR - no native binary; run make llama98"
    exit 2
fi

FAIL=0
PASS=0
for fx in zh en; do
    TOK="build/norm_int_gate_tokens_$fx.txt"
    if [ ! -f "$TOK" ]; then
        echo "  $fx: SKIP - fixture missing: $TOK"
        continue
    fi
    OUT=$("./$BIN" "$MODEL" --tokens-file "$TOK" --fp-exceptions 2>&1)
    LINE=$(printf '%s\n' "$OUT" | grep '^fp-exceptions:')
    if [ -z "$LINE" ]; then
        echo "FAIL: $fx - no fp-exceptions line (did the run get that far?)"
        printf '%s\n' "$OUT" | tail -n 3 | sed 's/^/      /'
        FAIL=1
        continue
    fi
    case "$LINE" in
        *"not measured"*)
            echo "  $fx: SKIP - $LINE"
            continue
            ;;
    esac
    IE=${LINE#*IE=}; IE=${IE%% *}
    OE=${LINE#*OE=}; OE=${OE%% *}
    if [ "$IE" = 0 ] && [ "$OE" = 0 ]; then
        echo "PASS: $fx - no NaN (IE=0), no overflow (OE=0)"
        echo "      $LINE"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $fx - IE=$IE OE=$OE"
        echo "      $LINE"
        echo "      A NaN or an Inf reached the arithmetic. The ARM"
        echo "      soft-float fast path drops the checks that would have"
        echo "      caught it, so that path is now unsound for this model."
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "fp_exception_gate: FAIL"
    exit 1
fi
if [ "$PASS" -eq 0 ]; then
    echo "fp_exception_gate: SKIP - nothing measured"
    exit 2
fi
echo "fp_exception_gate: PASS ($PASS fixture(s), IE=0 OE=0)"
exit 0
