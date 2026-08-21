#!/bin/bash
# f32count-billing gate: catches a new float-producing-from-int-
# accumulator function that never calls LZ_FCX/LZ_FC - the defect class
# epi_q41 was (src/ops.c: unbilled since it was written, discovered only
# by hand-instrumenting a throwaway build and comparing call counts
# against the census; src/ops.h's own header comment on the census
# machinery names two earlier instances of the same class:
# matmul_q8_int_consume and lz_quantize_q8, the latter fixed by adding
# LZ_FC_QUANT).
#
# A grep for LZ_FCX's presence somewhere in a file proves nothing about
# COVERAGE: epi_q41 sat nine lines from epi_q8 in the same file, which
# DOES call LZ_FCX, for as long as the bug existed - src/ops.c has always
# had LZ_FCX matches, that was never the question.
#
# METHOD: find every function definition in src/*.c whose return type is
# `float` and whose parameter list contains an `int32_t *` or
# `int64_t *` - an integer accumulator pointer, the shape shared by
# every member of the dequant-epilogue family (epi_fixed, epi_q8,
# epi_q41) and anything structurally like it: something that converts an
# integer accumulation into a float result is exactly the operation the
# census exists to bill. For each match, walk its body by brace depth
# from the opening `{` to the matching close and require at least one
# `LZ_FCX(` or `LZ_FC(` call somewhere inside. A match with neither is a
# FAIL: it produces float from an int accumulator and bills nothing.
#
# Pure awk, no ripgrep dependency (an `rg` shell function exists in some
# interactive sessions here but is not a real binary on PATH - this
# script must run standalone under `make check`).
#
# Signature discovery relies on this tree's own consistent style: a
# top-level function definition's return type starts at column 0, and a
# function body is always indented - so `^float` is never ambiguous with
# a local `float x = ...;` declaration inside some OTHER function's body,
# and C has no nested function definitions to confuse the brace-depth
# walk either. Verified empirically before this gate was written: this
# pattern hits exactly epi_fixed/epi_q8/epi_q41 across the entire src/
# tree and nothing else - no false positives to maintain an exception
# list for. A function that legitimately needs no billing should not
# have this signature shape to begin with (return the raw accumulator
# instead, as epi_fixed_raw's `long long` and epi_align_i16's `int32_t`
# already do).
#
# LIMITATION - this gate does NOT cover the whole defect class, and a
# PASS must not be read as "every float path is billed". It keys on the
# RETURN TYPE, so an unbilled float loop inside a `void` function is
# invisible to it. One was found by hand after this gate was written and
# passing: the Q4_1/Q6_1/T2 zero term's activation half, three identical
# copies inlined in matmul_q41_impl/matmul_t2_impl/matmul_q61_impl,
# nt*ng*r converts+multiplies+adds each, unbilled since they were
# written (now epi_zero_act in src/ops_matmul.c, billed).
#
# The rule that WOULD have caught it keys on the operation instead of the
# signature: lz_i32f() is this tree's int->float convert and is a
# billable `cvt` by definition, so "a function that calls lz_i32f and
# neither LZ_FCX nor LZ_FC is unbilled" covers both shapes. Not
# implemented here because the 40 current call sites have not been
# audited for the legitimate exception (a leaf kernel whose CALLER bills
# on its behalf, e.g. src/ops_mmx.c), and a gate with an unaudited
# exception list is worse than a narrow one that says so.
#
# Exit codes: 0 PASS, 1 FAIL
set -u
cd "$(dirname "$0")/.." || exit 1

AWK_PROG='
{ L[NR] = $0 }
END {
    n = NR
    total = 0
    fail = 0
    for (i = 1; i <= n; i++) {
        if (L[i] !~ /^float [A-Za-z_][A-Za-z0-9_]*[ \t]*\(/) continue

        fname = L[i]
        sub(/^float[ \t]+/, "", fname)
        sub(/[ \t]*\(.*/, "", fname)

        # Gather signature lines until "{" (definition) or a bare ";"
        # first (a prototype, not a definition - skip it).
        sig = L[i]
        j = i
        while (sig !~ /\{/ && sig !~ /;/ && j < n) {
            j++
            sig = sig "\n" L[j]
        }
        if (sig !~ /\{/) continue

        is_cand = (sig ~ /int32_t[ \t]*\*/ || sig ~ /int64_t[ \t]*\*/)

        tmp = sig; o = gsub(/\{/, "{", tmp)
        tmp = sig; c = gsub(/\}/, "}", tmp)
        depth = o - c
        billed = (sig ~ /LZ_FCX\(/ || sig ~ /LZ_FC\(/) ? 1 : 0

        k = j
        while (depth > 0 && k < n) {
            k++
            tmp = L[k]; oo = gsub(/\{/, "{", tmp)
            tmp = L[k]; cc = gsub(/\}/, "}", tmp)
            depth += (oo - cc)
            if (!billed && (L[k] ~ /LZ_FCX\(/ || L[k] ~ /LZ_FC\(/)) billed = 1
        }

        if (is_cand) {
            total++
            if (!billed) {
                print FILENAME ":" i ": FAIL float " fname "(...) never calls LZ_FCX/LZ_FC"
                fail = 1
            }
        }
        i = k
    }
    print "CANDIDATES=" total > "/dev/stderr"
    exit fail
}
'

FAIL=0
CANDS=0
for f in src/*.c; do
    [ -e "$f" ] || continue
    out=$(awk "$AWK_PROG" "$f" 2>"$f.f32gate.stderr")
    st=$?
    c=$(sed -n 's/^CANDIDATES=//p' "$f.f32gate.stderr")
    rm -f "$f.f32gate.stderr"
    CANDS=$((CANDS + ${c:-0}))
    if [ -n "$out" ]; then
        echo "$out"
    fi
    if [ "$st" -ne 0 ]; then
        FAIL=1
    fi
done

echo "f32count_billing_gate: $CANDS candidate(s) checked"
if [ "$FAIL" -eq 0 ]; then
    echo "f32count_billing_gate: PASS"
    exit 0
fi
echo "f32count_billing_gate: FAIL"
exit 1
