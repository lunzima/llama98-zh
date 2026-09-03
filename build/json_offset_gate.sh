#!/bin/bash
# json.c number-precision gate: LZJsonNode.num must be a double parsed by
# strtod, not a float parsed by strtof.
#
# WHY THIS MATTERS. safetensors headers store data_offsets - byte offsets
# into the checkpoint - as plain JSON integers, and src/safetensors.c
# reads them straight off LZJsonNode.num (no lz_json_get_int in the way).
# float32's mantissa is 24 bits; any real checkpoint's offsets run well
# past 2^24, so a float field rounds some of them - SILENTLY, no parse
# error, the tensor's recorded start is just wrong. This is not
# hypothetical: a real load hit offset 536998048, which strtof rounds to
# 536998016. See src/json.h's LZJsonNode.num comment and
# build/json_offset_probe.c's header for the full account, including the
# case where the rounding error cancels out of the nbytes cross-check and
# lz_st_open() returns SUCCESS with a silently wrong tensor offset.
#
# THE CONTROL is a MUTATION, not a git-history diff: this script copies
# EVERY src/*.c and src/*.h into a scratch tree and applies the exact
# two-line surgical edit that turns the fix back into the bug (double
# num -> float num, strtod -> strtof) via sed on unique tokens. That
# keeps the control meaningful regardless of what git HEAD is doing
# elsewhere - a git-ref-pinned "before" copy would silently stop proving
# anything the day this fix gets committed and HEAD catches up to the
# working tree. If the sed finds nothing to replace (the field or the
# call got renamed, or the fix regressed on its own), that is reported
# as a gate FAILURE, not a silent no-op mutation.
#
# WHY EVERY FILE GETS COPIED, not just json.c/json.h: a quote-include
# (`#include "json.h"`) resolves against the DIRECTORY OF THE INCLUDING
# FILE before it ever consults -I. Building src/model.c with only
# -I<mutant-dir> added still pulls in src/json.h, because model.c's own
# directory (src/) is searched first and has a json.h right there. Two
# translation units then disagree about sizeof(LZJsonNode) and the link
# produces a Frankenstein binary that reads garbage - not a rebuild of
# the historical bug. Copying every file into one directory makes same-
# directory resolution do the right thing everywhere.
#
# Exit: 0 pass, 1 fail, 2 skip (no C compiler).
set -u
cd "$(dirname "$0")/.." || exit 1

CC=${CC:-cc}
command -v "$CC" >/dev/null 2>&1 || CC=gcc
command -v "$CC" >/dev/null 2>&1 || {
    echo "json_offset_gate: SKIP - no C compiler"
    exit 2
}

W=$(mktemp -d 2>/dev/null || echo /tmp/jsonoffgate.$$)
mkdir -p "$W/fixed" "$W/mutant"
trap 'rm -rf "$W"' EXIT

cp src/*.c src/*.h "$W/fixed/" 2>/dev/null
cp src/*.c src/*.h "$W/mutant/" 2>/dev/null

sed -i 's/double num;/float num;/' "$W/mutant/json.h"
if ! grep -q 'float num;' "$W/mutant/json.h" || grep -q 'double num;' "$W/mutant/json.h"; then
    echo "json_offset_gate: FAIL - could not mutate LZJsonNode.num in json.h" \
         "(the sed target text is gone - the gate no longer matches the source)"
    exit 1
fi
sed -i 's/double v;/float v;/; s/strtod(ps->p, \&endp);/strtof(ps->p, \&endp);/' "$W/mutant/json.c"
if ! grep -q 'float v;' "$W/mutant/json.c" || ! grep -q 'strtof(ps->p' "$W/mutant/json.c"; then
    echo "json_offset_gate: FAIL - could not mutate the number parser in json.c" \
         "(the sed target text is gone - the gate no longer matches the source)"
    exit 1
fi

RC=0

# ---------------------------------------------------------- TIER 1/2 probe
build_and_run_probe() {
    local dir="$1" label="$2" want_pass="$3"
    if ! $CC -O2 -std=c99 -Wall -Wextra -I"$dir" -o "$dir/probe" \
            build/json_offset_probe.c "$dir/json.c" "$dir/safetensors.c" \
            "$dir/err.c" "$dir/compat.c" 2>"$dir/cc.log"; then
        echo "json_offset_gate: FAIL - probe did not build against $label"
        tail -n 20 "$dir/cc.log"
        return 1
    fi
    "$dir/probe" "$dir/probe_tmp.safetensors" > "$dir/run.log" 2>&1
    local rc=$?
    cat "$dir/run.log"
    rm -f "$dir/probe_tmp.safetensors"
    if [ "$want_pass" = 1 ] && [ "$rc" -ne 0 ]; then
        echo "json_offset_gate: FAIL - $label did not pass the probe (exit $rc)"
        return 1
    fi
    if [ "$want_pass" = 0 ] && [ "$rc" -eq 0 ]; then
        echo "json_offset_gate: FAIL - $label PASSED the probe; it should have" \
             "failed (float/strtof cannot be distinguished from the fix)"
        return 1
    fi
    return 0
}

echo "-- current tree (must PASS) --"
build_and_run_probe "$W/fixed" "the current (fixed) tree" 1 || RC=1

echo ""
echo "-- mutated tree: double->float, strtod->strtof (must FAIL) --"
build_and_run_probe "$W/mutant" "the float/strtof mutant" 0 || RC=1
if [ "$RC" -eq 0 ]; then
    echo "-- mutant correctly failed - the probe is sensitive to this exact" \
         "defect class --"
fi

# --------------------------------------------- config.json regression
# LZJsonNode.num going from float to double must not change ANY
# config.json field lz_load_config reads - they are all small numbers
# (vocab_size, hidden_size, rope_theta, ...), nowhere near float32's
# 2^24 exactness boundary. Compares the fixed and mutant builds' full
# lz_load_config() output, via the REAL src/model.c, on a real
# config.json.
CFG="${LZ_JSON_GATE_CONFIG:-/e/LLM/models/kunmoe-v2/config.json}"
echo ""
if [ ! -f "$CFG" ]; then
    echo "json_offset_gate: config.json regression SKIPPED - $CFG not found" \
         "(set LZ_JSON_GATE_CONFIG)"
else
    echo "-- config.json regression: $CFG --"
    cp build/json_config_regress_probe.c "$W/fixed/harness.c"
    cp build/json_config_regress_probe.c "$W/mutant/harness.c"
    ENGX="err compat lfn json safetensors model ops_sched cpucheck lz4d harness"
    build_regress() {
        local dir="$1" label="$2"
        for f in $ENGX; do
            $CC -O2 -std=c99 -I"$dir" -c "$dir/$f.c" -o "$dir/$f.o" \
                2>>"$dir/cc2.log" || {
                echo "json_offset_gate: FAIL - $f.c did not build ($label)"
                tail -n 20 "$dir/cc2.log"; return 1; }
        done
        $CC "$dir"/{err,compat,lfn,json,safetensors,model,ops_sched,cpucheck,lz4d,harness}.o \
            -o "$dir/cfgregress" -lm 2>>"$dir/cc2.log" || {
            echo "json_offset_gate: FAIL - regression harness did not link ($label)"
            tail -n 20 "$dir/cc2.log"; return 1; }
        return 0
    }
    if build_regress "$W/fixed" "fixed" && build_regress "$W/mutant" "mutant"; then
        "$W/fixed/cfgregress" "$CFG" > "$W/fixed/cfg.log" 2>&1
        "$W/mutant/cfgregress" "$CFG" > "$W/mutant/cfg.log" 2>&1
        if ! diff -u "$W/fixed/cfg.log" "$W/mutant/cfg.log" > "$W/cfg.diff"; then
            echo "json_offset_gate: FAIL - config.json parsing changed between" \
                 "float and double LZJsonNode.num"
            cat "$W/cfg.diff"
            RC=1
        else
            echo "-- config.json: identical field values, fixed vs mutant" \
                 "($(wc -l < "$W/fixed/cfg.log") fields) --"
        fi
    else
        RC=1
    fi
fi

# --------------------------------------------------- Watcom compile-only
# Not a full link/run (DOS/NT extender plumbing is out of scope for a
# value gate) - just confirms json.c/safetensors.c still compile clean
# under the toolchain that ships. SKIP, not FAIL, when Watcom is absent.
echo ""
WCC=${WCC:-}
for cand in /c/WATCOM/binnt64/wcc386.exe /mnt/c/WATCOM/binl64/wcc386; do
    [ -z "$WCC" ] && [ -x "$cand" ] && WCC="$cand"
done
if [ -z "${WCC:-}" ] || [ ! -x "$WCC" ]; then
    echo "json_offset_gate: Watcom compile-check SKIPPED - no wcc386 found"
else
    WROOT=$(dirname "$(dirname "$WCC")")
    if "$WCC" -za99 -otexan -zp4 -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -bt=nt \
            -I"$WROOT/h" -I"$WROOT/h/nt" -Isrc \
            -fo="$W/fixed/json_w.obj" src/json.c > "$W/fixed/wcc_json.log" 2>&1 && \
       "$WCC" -za99 -otexan -zp4 -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -bt=nt \
            -I"$WROOT/h" -I"$WROOT/h/nt" -Isrc \
            -fo="$W/fixed/st_w.obj" src/safetensors.c > "$W/fixed/wcc_st.log" 2>&1; then
        echo "json_offset_gate: Watcom compile-check PASS (json.c, safetensors.c)"
    else
        echo "json_offset_gate: FAIL - Watcom could not compile json.c/safetensors.c"
        tail -n 10 "$W/fixed/wcc_json.log" "$W/fixed/wcc_st.log"
        RC=1
    fi
fi

echo ""
if [ "$RC" -ne 0 ]; then
    echo "json_offset_gate: FAIL"
    exit 1
fi
echo "json_offset_gate: PASS"
exit 0
