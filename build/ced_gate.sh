#!/bin/bash
# --ced gate.
#
#   ced_gate.sh <model-dir> [prompt]
#
# --ced's whole claim is that its output is UNCHANGED: the layer that
# ends the stack skips the prefix in prefill - its K/V is still written
# for every position, from the encoder's exit hidden - while its
# queries, attention, output projection and FFN run for the final chunk
# only. So the assertion is a bitwise comparison, not a tolerance.
#
# ---------------------------------------------------------------------
# IDENTICAL OUTPUT IS ALSO WHAT --ced WIRED TO NOTHING PRODUCES, which
# is why the comparison is only HALF of this gate and the skip counter
# is the other half. The on arm must report a non-zero skip and the off
# arm must report zero; a run that produces equal files with a zero
# count FAILS here rather than being recorded as a pass. That is the
# same shape --attn-window already uses, and it is needed more here:
# the window knob changes the numbers, so its own output moves even if
# the counter is lying, whereas this one cannot move at all.
#
# ---------------------------------------------------------------------
# WHY --dump-prefill AND NOT --dump-logits.
#
# They are mutually exclusive and the first version of this gate did not
# know it. --dump-logits belongs to the --tokens path, which forwards
# ONE TOKEN AT A TIME through lz_forward; that path has no prefix to
# skip - every position's output is what it was asked for - so --ced
# cannot apply to it and the run reported "0 prefill positions skipped"
# on BOTH arms while comparing two identical files. A green gate over a
# knob that never fired.
#
# --dump-prefill was added for this: it writes what lz_forward_batch
# returns, and lz_forward_batch is the only entry where --ced runs (the
# verify and capture entries want every position's output, so they pass
# 0 and stay out of it).
#
# ---------------------------------------------------------------------
# THE MUTATIONS, and what each one proves the gate is sensitive to.
#
#   M1 attn_impl returns immediately for kv_only - the skip still
#      fires, but the K/V rows for the skipped positions are never
#      written. Proves the comparison sees a broken SKIP.
#   M2 the caller's `done + k < n` is dropped, so the FINAL chunk is
#      skipped too and the observed logits come from a layer whose
#      output was never computed. Proves the comparison sees a skip in
#      the wrong PLACE, which no counter would catch (the count would
#      go UP, not to zero).
#
# A mutant that builds but passes is a FAIL: it means the gate is blind
# to that exact break, which is worse than not having the gate.
#
# Usage: ced_gate.sh <model-dir> [prompt]
# Exit:  0 pass, 1 fail, 2 skip (no binary / no model)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${1:-}
PROMPT=${2:-"人工智能的发展历史可以追溯到二十世纪五十年代，当时的科学家们开始思考机器是否能够模拟人类的思维过程。"}

BIN=$ROOT/llama98.exe
[ -x "$BIN" ] || { echo "ced_gate: SKIP - no $BIN (run: make native)"; exit 2; }
[ -n "$MODEL" ] && [ -d "$MODEL" ] || {
    echo "ced_gate: SKIP - model dir not given or not found: '$MODEL'"; exit 2; }

W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
FAILED=0

# Does <log> hold a ced line whose count is <cmp> zero?
count() { sed -n 's/.*\[ced: [a-z]*, \([0-9]*\) prefill positions skipped\].*/\1/p' "$1" | head -1; }

# arm TAG CEDFLAG [BINARY] - run one prefill and write $W/TAG.pf
arm() {
    local tag="$1" mode="$2" bin=${3:-$BIN}
    "$bin" "$MODEL" --prompt "$PROMPT" -n 1 --seed 1 \
        --dump-prefill "$W/$tag.pf" --ced "$mode" >"$W/$tag.log" 2>&1 || {
        echo "  FAIL: '$tag' exited non-zero"; tail -3 "$W/$tag.log"; return 1; }
    [ -s "$W/$tag.pf" ] || { echo "  FAIL: '$tag' wrote no prefill logits"; return 1; }
    return 0
}

echo "ced_gate: $MODEL"
arm off off || exit 1
arm on  on  || exit 1

NOFF=$(count "$W/off.log"); NON=$(count "$W/on.log")
[ -n "$NOFF" ] && [ -n "$NON" ] || {
    echo "  FAIL: no [ced: ...] line - the counter this gate leans on is gone"
    exit 1; }

if cmp -s "$W/off.pf" "$W/on.pf"; then PF=identical; else PF=differ; FAILED=1; fi
printf '  off %-14s on %-14s  logits %s\n' \
       "skipped $NOFF" "skipped $NON" "$PF"

[ "$PF" = identical ] || echo "  FAIL: --ced moved the prefill logits; it is not supposed to move anything"
# THE OTHER HALF. Zero skips on the on arm means the arm did not run,
# and equal files then prove nothing at all.
[ "$NON" -gt 0 ] || { echo "  FAIL: --ced on skipped nothing - the arms are the same run"
                      FAILED=1; }
[ "$NOFF" -eq 0 ] || { echo "  FAIL: --ced off skipped $NOFF positions"
                       FAILED=1; }

# ---------------------------------------------------------------------
# The off arm against HEAD. "off is the unmodified code path" is the
# claim every other --ced result is read against, and inspecting the
# guards cannot settle it: what has to hold is that the BINARY built
# from the current source, run with --ced off, produces what the binary
# built from the previous source produced. Built from `git show HEAD:`
# into a copy, so the working tree is never touched - this tree is
# shared and another build may be reading it.
#
# The observable has to be one HEAD ALSO HAS, and that rules out
# --dump-prefill: it is part of this change, so the HEAD binary refuses
# the option and the comparison would fail for the wrong reason. The
# --tokens-file path is the one that works, and it is the right one
# anyway: it calls lz_forward one token at a time, which reaches
# attn_impl with kv_only == 0 on every call - exactly the branch the
# guards were added to, and the only place a dropped guard could show up
# as a different number rather than as a different amount of work.
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    rm -rf "$W/head"; mkdir -p "$W/head"
    cp -r "$ROOT/src" "$ROOT/common" "$W/head/"
    cp "$ROOT/Makefile" "$W/head/Makefile"
    for f in src/err.h src/err.c src/forward.h src/forward_attn.c \
             src/forward.c src/cli_main.c; do
        git -C "$ROOT" show "HEAD:$f" > "$W/head/$f"
    done
    if ( cd "$W/head" && make native ) >"$W/head/build.log" 2>&1; then
        # The id list, from the current binary - both arms accept the
        # flag, and it is what the prompt tokenizes to on either.
        "$BIN" "$MODEL" --prompt "$PROMPT" --dump-prompt-tokens "$W/toks" \
               -n 1 >/dev/null 2>&1
        if [ -s "$W/toks" ]; then
            ok=1
            # No --ced flag at all on the HEAD arm: HEAD does not have
            # the option, which is the point - that IS the pre-change
            # engine.
            "$W/head/llama98.exe" "$MODEL" --tokens-file "$W/toks" \
                --dump-logits "$W/head.lg" >"$W/head.log" 2>&1 || ok=0
            "$BIN" "$MODEL" --tokens-file "$W/toks" --ced off \
                --dump-logits "$W/now.lg" >"$W/now.log" 2>&1 || ok=0
            if [ "$ok" -eq 0 ]; then
                echo "  FAIL: a run in the off-vs-HEAD pair exited non-zero"
                tail -3 "$W/head.log" "$W/now.log"; FAILED=1
            elif cmp -s "$W/head.lg" "$W/now.lg"; then
                echo "  off arm vs HEAD source: bitwise identical"
            else
                echo "  FAIL: --ced off does not match the pre-change engine"
                FAILED=1
            fi
        else
            echo "  FAIL: could not produce a token list for the off-vs-HEAD pair"
            FAILED=1
        fi
    else
        echo "  FAIL: the HEAD build failed, so the off arm was not checked against it"
        tail -3 "$W/head/build.log"; FAILED=1
    fi
else
    echo "  FAIL: not a git tree, so the off arm cannot be checked against HEAD"
    FAILED=1
fi

# ---------------------------------------------------------------------
# Mutations. Built from a COPY of src/, never by editing the tree: this
# script may be run while another build is in flight, and a mutant that
# overwrote src/forward_attn.c would be a live-file race with whatever
# else is reading it.
mutate() {
    local tag="$1" sedexpr="$2" target
    rm -rf "$W/$tag"; mkdir -p "$W/$tag"
    cp -r "$ROOT/src" "$ROOT/common" "$W/$tag/"
    cp "$ROOT/Makefile" "$W/$tag/Makefile"
    # Which file the expression is aimed at, so "the mutation did not
    # apply" can be checked rather than inferred from a build failure.
    case "$tag" in
        m1) target="$W/$tag/src/forward_attn.c" ;;
        *)  target="$W/$tag/src/forward.c" ;;
    esac
    sed -i "$sedexpr" "$target"
    if cmp -s "$target" "$ROOT/${target#$W/$tag/}"; then
        echo "  FAIL mutation $tag: the edit did not apply - the gate would be"
        echo "       green against a mutation that was never made"
        FAILED=1; return 1
    fi
    # A MUTATION THAT DOES NOT BUILD IS A FAIL, not a skip. It proves
    # nothing about the gate, and a skip here reads exactly like a pass
    # in the summary line - which is how the first version of this
    # script reported PASS while both mutants had silently been skipped
    # for a missing common/ in the copy.
    if ! ( cd "$W/$tag" && make native ) >"$W/$tag/build.log" 2>&1; then
        echo "  FAIL mutation $tag: it does not build, so it tested nothing"
        tail -3 "$W/$tag/build.log"; FAILED=1; return 1
    fi
    arm "$tag.on" on "$W/$tag/llama98.exe" || { FAILED=1; return 1; }
    if cmp -s "$W/$tag.on.pf" "$W/off.pf"; then
        echo "  FAIL mutation $tag: the mutant matched the off arm - this gate is blind to it"
        FAILED=1
    else
        echo "  mutation $tag: red as it should be"
    fi
}

# M1: kv_only computes nothing, so the prefix K/V rows are never written.
mutate m1 's|int kv_only) {|int kv_only) { if (kv_only) return;|'
# M2: the final chunk is skipped as well, so the logits are observed from
# a layer whose output nobody computed. Note the counter goes UP on this
# one - a count-only gate would call it healthy.
mutate m2 's|s->ced \&\& (done + k < n)|s->ced|'

[ "$FAILED" -eq 0 ] && echo "  ced_gate: PASS" || echo "  ced_gate: FAIL"
exit $FAILED
