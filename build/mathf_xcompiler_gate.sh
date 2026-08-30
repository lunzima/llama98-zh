#!/bin/bash
# lz_mathf's float transcendentals must give the same bits under gcc and
# Watcom. Seconds, against build/fixed_tier_xcompiler_gate.sh's ~40
# minutes for the same class of defect.
#
# WHAT IT CAUGHT. reduce_large ended `x = (float)(int64_t)res0; return x
# * pi63;`. A 387 loads that 62-bit integer with `fild`, which is exact
# and which the precision control word does NOT touch, so the x87 build
# multiplied the exact integer by pi63 and rounded once while SSE and the
# ARM soft-float round the integer to float first and then round the
# product - one rounding against two, 1 ulp apart on ~0.6% of arguments.
# forward.c builds the RoPE table from lz_sinf/lz_cosf at every position
# at load time, so the first argument the two compilers disagreed on made
# the entire float tier (--fixed off) diverge from that token on: kmr20
# at position 125, _armgate2 at 134.
#
# WHY IT SURVIVED, and why this gate is short. The end-to-end gate
# (fixed_tier_xcompiler_gate.sh) DID see it - and then stopped, because
# the divergence was in its float control, whose failure is reported as
# "every tier result below would be uninterpretable, not running them".
# So the one gate that could see this defect answered by declining to
# run, and its two shipping rows (--fixed all, and the default) went
# untested for as long as the baseline was red. A cheap direct check
# means the expensive one is not the only asker.
#
# PC=24 IS NOT A SUBSTITUTE for the barrier, which is the part worth
# carrying forward: lz_fpu_float_begin makes the 387 round mantissas
# where SSE rounds and that closes the arithmetic gap, but it says
# nothing about `fild` (exact at any precision control) or about the
# 15-bit exponent. Every int->float conversion feeding a multiply is
# still a divergence site; ops_kernel_shared.h's lz_i32f is the int32
# half of the same problem.
#
# The control removes the barrier again and requires the probe to notice.
# A gate for a 1-ulp difference that is not mutation-tested is a gate
# nobody can tell from a no-op.
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP (a toolchain is missing).
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

PROBE=build/mathf_xcompiler_probe.c
[ -f "$PROBE" ] || { echo "mathf_xcompiler_gate: SKIP - $PROBE absent"; exit 2; }
command -v gcc >/dev/null 2>&1 || {
    echo "mathf_xcompiler_gate: SKIP - gcc not found"; exit 2; }

# Same two layouts the Makefile probes for, and the include root that
# goes with each: wcc386 finds neither <stdio.h> nor <float.h> without
# them, and the failure reads as a source error rather than a setup one.
if [ -x /c/WATCOM/binnt64/wcc386.exe ]; then
    WCC=/c/WATCOM/binnt64/wcc386.exe
    WLINK=/c/WATCOM/binnt64/wlink.exe
    WROOT=C:/WATCOM
    WBIN=/c/WATCOM/binnt64
elif [ -x /mnt/c/WATCOM/binl64/wcc386 ]; then
    WCC=/mnt/c/WATCOM/binl64/wcc386
    WLINK=/mnt/c/WATCOM/binl64/wlink
    WROOT=/mnt/c/WATCOM
    WBIN=/mnt/c/WATCOM/binl64
else
    echo "mathf_xcompiler_gate: SKIP - no wcc386 (checked binnt64/binl64);" \
         "gcc and Watcom were NOT compared"
    exit 2
fi

# Beside the repo, not /tmp: a Windows-hosted wlink resolves neither an
# MSYS nor a WSL path, and a build that silently wrote nowhere would be
# read as a pass.
W="${LZ_SCRATCH:-$(dirname "$ROOT")/tmp}/.mathf-xcc-$$"
mkdir -p "$W" || { echo "mathf_xcompiler_gate: FAIL - cannot create $W"; exit 1; }
trap 'rm -rf "$W"' EXIT

# $1 = source file for lz_mathf, $2 = output tag. Leaves $W/$2.txt.
build_and_run_watcom() {
    ( cd "$W" && "$WCC" -za99 -otexan -zp4 -4r -bt=nt \
        "-I$WROOT/h" "-I$WROOT/h/nt" "-I$ROOT/src" \
        -fo="$2_m.obj" "$1" ) > "$W/$2_cc.log" 2>&1 || {
        echo "mathf_xcompiler_gate: FAIL - wcc386 rejected $1"
        tail -n 5 "$W/$2_cc.log" | sed 's/^/      /'; return 1; }
    ( cd "$W" && WATCOM="$WROOT" PATH="$WBIN:$PATH" \
        "$WLINK" system nt name "$2.exe" \
        file "probe.obj,$2_m.obj" ) > "$W/$2_ln.log" 2>&1
    [ -x "$W/$2.exe" ] || {
        echo "mathf_xcompiler_gate: FAIL - wlink produced no $2.exe"
        tail -n 5 "$W/$2_ln.log" | sed 's/^/      /'; return 1; }
    "$W/$2.exe" > "$W/$2.txt" 2>&1
    [ -s "$W/$2.txt" ] || {
        echo "mathf_xcompiler_gate: FAIL - $2.exe printed nothing"; return 1; }
    return 0
}

# ---- gcc side ----------------------------------------------------------
if ! gcc -O2 -std=c99 -ffp-contract=off -Isrc -o "$W/g.exe" \
        "$PROBE" src/lz_mathf.c > "$W/g_cc.log" 2>&1; then
    echo "mathf_xcompiler_gate: FAIL - gcc rejected the probe"
    tail -n 5 "$W/g_cc.log" | sed 's/^/      /'; exit 1
fi
"$W/g.exe" > "$W/g.txt" 2>&1
[ -s "$W/g.txt" ] || { echo "mathf_xcompiler_gate: FAIL - gcc probe printed nothing"; exit 1; }

# ---- Watcom side, as shipped -------------------------------------------
( cd "$W" && "$WCC" -za99 -otexan -zp4 -4r -bt=nt \
    "-I$WROOT/h" "-I$WROOT/h/nt" "-I$ROOT/src" \
    -fo=probe.obj "$ROOT/$PROBE" ) > "$W/p_cc.log" 2>&1 || {
    echo "mathf_xcompiler_gate: FAIL - wcc386 rejected $PROBE"
    tail -n 5 "$W/p_cc.log" | sed 's/^/      /'; exit 1; }
build_and_run_watcom "$ROOT/src/lz_mathf.c" w || exit 1

if cmp -s "$W/g.txt" "$W/w.txt"; then
    echo "  as shipped : identical ($(wc -l < "$W/g.txt") entries)"
else
    echo "  as shipped : DIFFER, $(diff "$W/g.txt" "$W/w.txt" | grep -c '^<') of" \
         "$(wc -l < "$W/g.txt") entries"
    diff "$W/g.txt" "$W/w.txt" | head -n 6 | sed 's/^/      /'
    echo "mathf_xcompiler_gate: FAIL"
    echo "      A 1-ulp split here is not cosmetic: forward.c builds the"
    echo "      RoPE table from lz_sinf/lz_cosf at every position, so the"
    echo "      first argument that moves diverges the whole float tier"
    echo "      from that token onward. Look for an int->float conversion"
    echo "      feeding a multiply - fild is exact whatever PC=24 says."
    exit 1
fi

# ---- control: put the defect back and require the probe to see it ------
PY=python; command -v python >/dev/null 2>&1 || PY=python3
cp src/lz_mathf.c "$W/mut.c"
"$PY" - "$W/mut.c" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8", errors="replace").read()
old = "  x = fp_barrierf ((float)(int64_t)res0);"
new = "  x = (float)(int64_t)res0;"
if s.count(old) != 1:
    print("  the mutation site is gone; this gate is unproven")
    sys.exit(2)
open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new, 1))
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "mathf_xcompiler_gate: FAIL - could not plant the control" \
         "mutation, so the check above is unproven"
    exit 1
fi
build_and_run_watcom "$W/mut.c" mut || exit 1
if cmp -s "$W/g.txt" "$W/mut.txt"; then
    echo "  control    : IDENTICAL - the probe does not see the defect"
    echo "mathf_xcompiler_gate: FAIL - the control did not reproduce the"
    echo "      divergence, so the pass above proves nothing. Either the"
    echo "      barrier is no longer what forces the rounding, or the"
    echo "      probe stopped reaching reduce_large (it needs |x| >= 120)."
    exit 1
fi
echo "  control    : differs in" \
     "$(diff "$W/g.txt" "$W/mut.txt" | grep -c '^<') entries (barrier removed)"
echo "mathf_xcompiler_gate: PASS (gcc and Watcom identical, mutation caught)"
exit 0
