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

# ---------------------------------------------------------------------
# THE SECOND RULE, keyed on the OPERATION rather than the signature.
#
# The LIMITATION note above named this rule and declined to implement it
# because "the current call sites have not been audited for the
# legitimate exception". They have been now, and the audit found THREE
# real gaps in one pass - the delta quantization in gdn_p2_fixed_rows,
# gdn_pass1_fixed_head's own prologue and epilogue, and the whole of
# gdn_build_table. Together they were 160,896 ops/token invisible, and
# billing them moved `recurrence write` from third place to first with
# more than twice the runner-up. The signature rule was passing that
# entire time: all three are void functions.
#
# THE RULE: a top-level function whose body calls lz_i32f() or
# q8_round() - this tree's int->float and float->int primitives - and
# calls neither LZ_FCX() nor LZ_FC() performs a conversion nothing
# bills. That is a FAIL unless it is in EXEMPT below.
#
# EXEMPT IS A TABLE OF REASONS, NOT A LIST OF NAMES. Each entry says
# which of the three legitimate shapes it is, because a name-only list
# lets a function change shape underneath its own exemption:
#
#   caller-bills   a leaf whose call sites bill on its behalf, and say so
#                  at the call site (norm_ss_fixed's callers each carry
#                  its n+1 mul / n add / 1 cvt in their own formula)
#   once-per-model setup that runs once and is cached or idempotent, so
#                  it carries no per-token cost for a per-token census to
#                  miss (lz_epi_prep is `if (w->sq) return;`,
#                  conv_fixed_build's own comment says "once", build_qw
#                  is a keyed slot cache, exp_t_bounds is
#                  `if (!g_expt_ready)`)
#   tier-or-model  reachable only on a path this census does not measure
#                  - --kernel ref, or a use_subn model
#
# ONE ENTRY IS AN ADMISSION AND IS MARKED AS ONE:
# lz_rmsnorm_gated_out_fixed_int's caller bills 5n with a comment that
# was wrong about why (it claimed the output loop is skipped; it is
# not). Its number has not been re-derived. It is off by default
# (lz_norm_int() is -1), so it costs the shipping census nothing, and
# replacing an unverified number with another is not a correction.
EXEMPT='
p2_group_norm                   caller-bills
norm_ss_fixed                   caller-bills
lz_wsum_group_mmx               caller-bills
lz_rmsnorm_gated_out_fixed      caller-bills
lz_rmsnorm_out_fixed_int        caller-bills
lz_rmsnorm_gated_out_fixed_int  caller-bills-UNVERIFIED
lz_conv_norm_pow2               once-per-model
epi_prep_plane                  once-per-model
build_qw                        once-per-model
exp_t_bounds                    once-per-model
matmul_scalar_ref_one           tier-or-model
subn_fwht_quant                 tier-or-model
lz_quantize_q8_tok_int          tier-or-model
'

OP_PROG='
{ L[NR] = $0 }
END {
    for (i = 1; i <= NR; i++) {
        line = L[i]
        if (line !~ /^[A-Za-z_]/) continue
        if (line ~ /^#/) continue
        if (index(line, "(") == 0) continue
        j = i; brace = -1; k = 0
        while (k < 12 && j <= NR) {
            if (index(L[j], "{") > 0) { brace = j; break }
            if (index(L[j], ";") > 0) break
            k++; j++
        }
        if (brace < 0) continue
        head = line
        sub(/\(.*$/, "", head)
        nm = head
        sub(/^.*[^A-Za-z0-9_]/, "", nm)
        if (nm == "" || nm == "if" || nm == "for" || nm == "while" ||
            nm == "switch" || nm == "return") continue
        d = 0; ops = 0; bill = 0
        for (j = brace; j <= NR; j++) {
            s = L[j]
            for (c = 1; c <= length(s); c++) {
                ch = substr(s, c, 1)
                if (ch == "{") d++
                else if (ch == "}") d--
            }
            if (index(s, "lz_i32f(") > 0 || index(s, "q8_round(") > 0) ops = 1
            if (index(s, "LZ_FCX(") > 0 || index(s, "LZ_FC(") > 0) bill = 1
            if (d <= 0) break
        }
        if (ops && !bill) {
            if (!(nm in ex))
                print FILENAME ":" i ": FAIL " nm "() converts and bills nothing"
            else
                seen[nm] = 1
            total++
        }
        i = j
    }
    for (e in ex) if (!(e in seen)) unused = unused " " e
    print "OPCANDS=" total > "/dev/stderr"
    print "UNUSED=" unused > "/dev/stderr"
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

echo "f32count_billing_gate: $CANDS candidate(s) checked (signature rule)"

# --- operation rule ----------------------------------------------------
EXFILE=$(mktemp)
echo "$EXEMPT" | awk 'NF { print $1 }' > "$EXFILE"
OPC=0
ALLSEEN=$(mktemp)
for f in src/*.c; do
    [ -e "$f" ] || continue
    out=$(awk -v exf="$EXFILE" '
        BEGIN { while ((getline e < exf) > 0) if (e != "") ex[e] = 1 }
        '"$OP_PROG" "$f" 2>"$f.op.stderr")
    c=$(sed -n 's/^OPCANDS=//p' "$f.op.stderr")
    OPC=$((OPC + ${c:-0}))
    rm -f "$f.op.stderr"
    if [ -n "$out" ]; then echo "$out"; FAIL=1; fi
done
rm -f "$EXFILE" "$ALLSEEN"

echo "f32count_billing_gate: $OPC converting function(s) checked (operation rule)"
if [ "$FAIL" -eq 0 ]; then
    echo "f32count_billing_gate: PASS"
    exit 0
fi
echo "f32count_billing_gate: FAIL"
exit 1
