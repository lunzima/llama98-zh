#!/bin/bash
# Symmetric ISA-floor gate, Watcom side (task #30 closeout).
#
# The gcc side is tests/test_ops_isa_floor.py (training tree): it compiles
# every $(ENG) object and asserts zero MMX/SSE/SSE2 in the non-suffixed
# ones. This script checks the SAME invariant against Watcom's own objects,
# because objdump cannot read Watcom's OMF format - wdis is the established
# tool (see mm_alias_gate.sh). The invariant is "non-suffixed TU carries
# zero MMX/SSE/SSE2 register operands"; task #30 made it reachable on the
# Watcom side by moving ops.c's kernels to ops_mmx.c/ops_mmx_sse.c/
# ops_sse2.c and dropping ops.c to the -4r/-3r floor.
#
# Classified by REGISTER OPERAND, not mnemonic, matching isa_floor.py:
# \bmm[0-7]\b and \bxmm[0-7]\b. emms carries no register operand and is
# deliberately not counted - it is the fix (clears the MMX state the
# suffixed TUs left), not the bug, and its correctness on a machine without
# MMX is a runtime-dispatch question (task #29), not a static-floor one.
#
# Reads build/watcom/cli/*.obj rather than recompiling: make check's
# watcom-cli target already produced them, and recompiling 15 objects
# would double its runtime. The ordering dependency is real (run watcom-cli
# first) and is enforced below by failing on an empty object set rather
# than silently passing.
#
# Exit codes: 0 PASS, 1 FAIL, 2 ERROR (missing toolchain or objects).
set -u
cd "$(dirname "$0")/.."

eval "$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="the Watcom ISA floor" bash build/tree_lock.sh acquire --print-env)" || exit 2
trap 'bash build/tree_lock.sh release' EXIT

if [ -x /c/WATCOM/binnt64/wdis.exe ]; then
    WDIS=/c/WATCOM/binnt64/wdis.exe
elif [ -x /mnt/c/WATCOM/binl64/wdis ]; then
    WDIS=/mnt/c/WATCOM/binl64/wdis
else
    echo "isa_floor_watcom_gate: ERROR - wdis not found (checked binnt64/binl64)"
    exit 2
fi

OBJ=build/watcom/cli
FAIL=0
checked=0

for f in "$OBJ"/*.obj; do
    [ -e "$f" ] || continue
    b=$(basename "$f" .obj)
    # Suffixed TUs legitimately carry MMX/SSE; the common/ and _* objects
    # are not engine TUs and are compiled at the plain floor already.
    case "$b" in
        ops_mmx|ops_mmx_sse|ops_sse2|common_session|common_stream|_main|_argv|_attr)
            continue ;;
    esac
    checked=$((checked + 1))
    dis=$("$WDIS" -a "$f" 2>&1)
    if [ -z "$dis" ]; then
        echo "  ERROR: wdis produced no output for $b (object corrupt?)"
        FAIL=1
        continue
    fi
    mm=$(printf '%s\n' "$dis" | grep -cE '\bmm[0-7]\b')
    xm=$(printf '%s\n' "$dis" | grep -cE '\bxmm[0-7]\b')
    if [ "$mm" -ne 0 ] || [ "$xm" -ne 0 ]; then
        echo "  FAIL: $b mm=$mm xmm=$xm"
        FAIL=1
    fi
done

if [ "$checked" -eq 0 ]; then
    echo "isa_floor_watcom_gate: ERROR - no objects under $OBJ (run make watcom-cli first)"
    exit 2
fi
if [ "$FAIL" -eq 0 ]; then
    echo "isa_floor_watcom_gate: PASS ($checked non-suffixed objects, zero mm/xmm)"
    exit 0
fi
echo "isa_floor_watcom_gate: FAIL"
exit 1
