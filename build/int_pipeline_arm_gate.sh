#!/bin/bash
# int-pipeline control-arm gate (build/ci.sh).
#
# switch_matrix_gate proves every control switch COMPILES at its
# non-default value and that the -D reached the preprocessor. It never
# links or runs one. That is the whole gap this gate closes.
#
# The int-pipeline switches are the ones where it matters. Each guards a
# path that replaced a float one, and each `=0` build keeps the replaced
# float path alive as the arm its milestone measured against. Nothing
# else executes those paths: the shipping default takes the int side and
# --fixed off, the one runtime float knob, turns off a different and
# much coarser thing. So a `=0` build is code that ran once, by hand, on the
# day it was written - exactly the "configuration nobody happens to
# build" this project keeps getting bitten by, one step further along
# (it builds; nobody checks it still WORKS).
#
# Per switch: link a full binary with -D<SWITCH>=0 and run it. Asserts
#   1. the link succeeds (the matrix only compiles TUs, so a missing
#      definition behind the guard shows up here and nowhere else),
#   2. the run exits 0,
#   3. --dump-all-logits produces the full expected size, not a
#      truncated or empty file. A crash after the banner still writes
#      something, so size is the assertion, not existence.
#
# Deliberately NOT asserted here: byte-identity against a baseline. The
# baseline moves with every commit that changes the shipping numerics,
# so a pinned expectation would be stale within a day and would then be
# red for a reason that has nothing to do with the commit under test -
# the failure mode build/fixed_tier_xcompiler_gate.sh spent weeks in
# when the norms tier's `int` value was folded away and its row stayed. Each
# milestone checks its own arm's byte-identity at the time it lands;
# this gate keeps the arm alive afterwards.
#
# DRIFT CHECK: the switch list below is reconciled against
# switch_matrix_gate.sh's SWITCHES. A new int-pipeline switch that is
# added there and not here makes this gate FAIL, so the two cannot get
# out of step silently. That is the same mistake in miniature that left
# the xcompiler gate listing a tier value the parser had dropped.
#
# LZ_SOFTPLUS_POLY IS INERT ON THE DEFAULT MODEL and this gate cannot
# say otherwise: lz_softplus runs only under forward_ssm, and kmr20 is
# KDA with a gate lower bound, so on it the =0 arm proves linkage and
# nothing else. Point the gate at a GDN checkpoint to exercise it -
# LZ_NORM_INT_GATE_MODEL=/e/LLM/models/_armgate3 - which is how the arm
# was checked, though on the PREVIOUS GDN fixture (_armgate2, gone; see
# docs/dsv41-optimization-menu.md section 6). _armgate3 covers the same
# path - it is GDN with a dense FFN - so the instruction still points
# somewhere real; whether the numbers reproduce on it has not been run.
# Recorded here rather than defaulted, because every other arm's numbers
# are on kmr20 and moving the default would silently re-baseline all
# twelve.
#
# gcc only. Watcom cross-compiler identity is fixed_tier_xcompiler_gate's
# subject and it exercises the shipping default, not these arms; adding
# a second toolchain here would double a cost that is already 9 links.
#
# Exit codes: 0 PASS, 1 FAIL, 2 ERROR (missing toolchain or model).
set -u
cd "$(dirname "$0")/.." || exit 1

ARMS="LZ_BVEC_I16 LZ_MOEGATE_I16 LZ_KLAT_I16 LZ_KGATE_I16 LZ_KGATE_EXP_I
LZ_MLAT_I16 LZ_MLAT_REUSE LZ_VPROJ_I16 LZ_Q8I16_INT LZ_Q8INT_INT
LZ_SWIGLU_I16 LZ_CONV_SIG_I LZ_CONVO_I16 LZ_SOFTPLUS_POLY"

if ! command -v gcc >/dev/null 2>&1; then
    echo "int_pipeline_arm_gate: ERROR - gcc not found"
    exit 2
fi

# --- drift check against the matrix's own table -------------------------
# Before the model check and before any build: a table out of step with
# the matrix is a setup error, and spending nine links to also report it
# would make the cheap failure cost as much as the expensive one.
MATRIX=$(grep -oE 'LZ_[A-Z0-9_]+=' build/switch_matrix_gate.sh | sed 's/=//' \
         | sort -u)
DRIFT=0
for m in $MATRIX; do
    case "$m" in
        *_I16|LZ_Q8I16_INT|LZ_Q8INT_INT|LZ_MLAT_REUSE|LZ_CONV_SIG_I|LZ_KGATE_EXP_I) ;;
        LZ_SOFTPLUS_POLY) ;;
        *) continue ;;
    esac
    echo "$ARMS" | tr ' ' '\n' | grep -qx "$m" || {
        echo "FAIL: $m is in switch_matrix_gate's SWITCHES but not in this"
        echo "      gate's ARMS - a control arm that compiles and is never run"
        DRIFT=1
    }
done
if [ "$DRIFT" -ne 0 ]; then
    echo "int_pipeline_arm_gate: FAIL (drift; no arm was built)"
    exit 1
fi

MODELS_ROOT=${MODELS_ROOT:-/e/LLM/models}
MODEL=${LZ_NORM_INT_GATE_MODEL:-$MODELS_ROOT/kmr20}
TOK=build/norm_int_gate_tokens_zh.txt
if [ ! -d "$MODEL" ] || [ ! -f "$TOK" ]; then
    echo "int_pipeline_arm_gate: SKIP - model or fixture missing ($MODEL, $TOK)"
    exit 2
fi

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT

FL_BASE="-O3 -march=x86-64 -std=c99 -Wall -Wextra -Isrc -ffp-contract=off"
FL_BASE="$FL_BASE -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"

# The expected dump size is taken from the shipping build rather than
# hard-coded: vocab x positions x 4, and both factors move with the
# model and the fixture. A pinned number here would be the stale-constant
# failure this file's header warns about.
rm -rf build/native
if ! make llama98 X86_64_CFLAGS="$FL_BASE" -j4 >"$TMPD/ref.log" 2>&1; then
    echo "int_pipeline_arm_gate: ERROR - reference build failed"
    tail -n 5 "$TMPD/ref.log"
    exit 2
fi
REF=llama98.exe
[ -f "$REF" ] || REF=llama98
if ! "./$REF" "$MODEL" --tokens-file "$TOK" \
        --dump-all-logits "$TMPD/ref.bin" >"$TMPD/refrun.log" 2>&1; then
    echo "int_pipeline_arm_gate: ERROR - reference run failed"
    tail -n 5 "$TMPD/refrun.log"
    exit 2
fi
WANT=$(wc -c <"$TMPD/ref.bin")
echo "reference dump $WANT bytes"

PASS=0
FAIL=0
for sw in $ARMS; do
    rm -rf build/native
    if ! make llama98 X86_64_CFLAGS="$FL_BASE -D$sw=0" -j4 \
            >"$TMPD/$sw.build.log" 2>&1; then
        echo "FAIL: $sw=0 does not link"
        grep -E 'error|undefined' "$TMPD/$sw.build.log" | head -3 | sed 's/^/      /'
        FAIL=1
        continue
    fi
    B=llama98.exe
    [ -f "$B" ] || B=llama98
    if [ ! -f "$B" ]; then
        echo "FAIL: $sw=0 produced no binary"
        FAIL=1
        continue
    fi
    if ! "./$B" "$MODEL" --tokens-file "$TOK" \
            --dump-all-logits "$TMPD/$sw.bin" >"$TMPD/$sw.run.log" 2>&1; then
        echo "FAIL: $sw=0 build does not run"
        tail -n 3 "$TMPD/$sw.run.log" | sed 's/^/      /'
        FAIL=1
        continue
    fi
    GOT=$(wc -c <"$TMPD/$sw.bin")
    if [ "$GOT" != "$WANT" ]; then
        echo "FAIL: $sw=0 dumped $GOT bytes, expected $WANT"
        FAIL=1
        continue
    fi
    echo "PASS: $sw=0 links, runs, $GOT bytes"
    PASS=$((PASS + 1))
done

# Put the tree back the way it was found. Every build above used
# FL_BASE, which is -O3 and has one control switch turned off, and the
# last one is what llama98.exe holds when this loop ends. Anything that
# reads the binary afterwards - codesize_gate measures its .text, and
# several later gates simply run it - would be reading an -O3 build of a
# crippled engine and would have no way to know. Wiping build/native is
# not enough on its own: it makes the NEXT `make llama98` rebuild, but a
# standalone run of this gate is followed by no make at all.
rm -rf build/native
if ! make llama98 -j4 >"$TMPD/restore.log" 2>&1; then
    echo "int_pipeline_arm_gate: FAIL - cannot restore the shipping build;"
    echo "  llama98.exe is left at $FL_BASE with one switch off"
    tail -n 5 "$TMPD/restore.log" | sed 's/^/      /'
    FAIL=1
fi

echo "int_pipeline_arm_gate: $PASS arm(s) exercised"
if [ "$FAIL" -eq 0 ]; then
    echo "int_pipeline_arm_gate: PASS"
    exit 0
fi
echo "int_pipeline_arm_gate: FAIL"
exit 1
