#!/bin/bash
# Every x86 operator must exist in BOTH toolchains' hand-written form.
#
#   gcc_watcom_pair_gate.sh [binary]
#
# The SIMD kernels are written twice - once as gcc
# intrinsics, once as wcc386 #pragma aux - and bit-identity between the
# two is checked by build/kernel_isa_identity_gate.sh. What nothing
# checked is that the two SETS are the same set: a unit with an mmxI
# cell and an empty mmxA cell has no twin to be identical to, and the
# identity gate is silent about it because it can only compare kernels
# that exist.
#
# kernel_matrix_gate.sh is not this gate either. It asserts that every
# empty cell carries a REASON (impossible / todo / declined) and that
# the registered class matches the binary. A cell can be honestly
# registered `i` on one side and `x` on the other and pass it - the
# reason is per cell, and nothing looks across the pair.
#
# ASSERTS, for all three x86 ISA columns and every unit:
#     the gcc cell is filled  <=>  the Watcom cell is filled
# where "filled" means x (here) or o (in another build class) and
# "empty" means i / t / d. Both empty is fine and is the common case
# (SSE1 is `i` on 14 of 15 units, with the audit above km_reg as its
# reason); one filled and one empty is what this gate exists to catch.
#
# WHAT THE MUTATIONS SHOWED, INCLUDING THE PART THAT IS NOT NEW
# COVERAGE. Re-registering q4_1's mmxA cell as `i`, then as `t`, turns
# this gate red both times (21 paired -> 20, only that pair named). But
# kernel_matrix_gate.sh ALSO turns red on both, and for a better reason
# than mine - "q4_1/mmxA present but not registered for this build
# class", i.e. it noticed the registration stopped describing the
# binary. On a cell whose kernel exists, that gate already has this
# covered and this one is a restatement.
#
# The case this gate uniquely covers is the other one: a kernel that
# genuinely is NOT written on one side and is HONESTLY registered t or
# d there. kernel_matrix_gate passes that - the registration matches
# reality, which is all it asks - and this gate fails it, because the
# the requirement asks for more than honesty about the gap. That case does not
# exist in the tree today (0 todo, 0 declined), so the mutation proving
# it would have to DELETE a Watcom kernel rather than re-register one,
# and it has not been run. Said here rather than left implied: a gate
# whose teeth are only shown on the cases another gate already catches
# has not been shown to have teeth on the case it was written for.
#
# Exit: 0 PASS, 1 FAIL, 2 SKIP.
set -u
cd "$(dirname "$0")/.." || exit 1

BIN=${1:-}
if [ -z "$BIN" ]; then
    for c in ./llama98.exe ./llama98; do [ -x "$c" ] && BIN=$c && break; done
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || {
    echo "gcc_watcom_pair_gate: SKIP - no native binary (make native)"; exit 2; }

M=$("$BIN" --kernel-matrix 2>&1)
[ -n "$M" ] || { echo "gcc_watcom_pair_gate: SKIP - --kernel-matrix said nothing"; exit 2; }

# cell lines look like: "cell q8_0  mmxI  x G-- row -"
UNITS=$(echo "$M" | awk '/^cell /{print $2}' | sort -u)
[ -n "$UNITS" ] || { echo "gcc_watcom_pair_gate: SKIP - no cells parsed"; exit 2; }

cls() {   # cls UNIT COLUMN -> the one-letter class
    echo "$M" | awk -v u="$1" -v c="$2" '$1=="cell" && $2==u && $3==c {print $4; exit}'
}

FAIL=0
NPAIR=0
NBOTH=0
NOWED=0
for u in $UNITS; do
    for p in "mmxI:mmxA" "sseI:sseA" "sse2I:sse2A"; do
        gi=${p%%:*}; wa=${p#*:}
        g=$(cls "$u" "$gi"); w=$(cls "$u" "$wa")
        if [ -z "$g" ] || [ -z "$w" ]; then
            echo "FAIL: $u $gi/$wa - a column is missing from --kernel-matrix"
            FAIL=1; continue
        fi
        gf=0; wf=0
        case "$g" in x|o) gf=1 ;; esac
        case "$w" in x|o) wf=1 ;; esac
        if [ "$gf" != "$wf" ]; then
            # A DECLARED todo IS NOT THIS GATE'S FAILURE. `t` is an
            # explicit registry state carrying a reason string, and
            # twin_pair_gate counts every one of them against a ceiling
            # that only a deliberate edit raises. Failing here as well
            # judges the same fact twice, with the harsher of the two
            # verdicts winning - which makes that ceiling meaningless
            # and makes a unit landing its gcc side first redden the
            # build for everyone until its twin exists. It surfaced when
            # recur and epi arrived that way.
            #
            # Anything OTHER than `t` on the empty side still fails: `!`
            # is a kernel present but unregistered and `?` a
            # registration with no kernel, and both mean the registry is
            # describing a binary that does not exist. Those are this
            # gate's business and nobody else's.
            if [ "$g" = "t" ] || [ "$w" = "t" ]; then
                NOWED=$((NOWED + 1))
                continue
            fi
            echo "FAIL: $u - $gi is '$g' but $wa is '$w'; one toolchain has a"
            echo "      hand-written kernel the other does not, so there is no"
            echo "      twin for the identity gate to compare"
            FAIL=1
        elif [ "$gf" = 1 ]; then
            NPAIR=$((NPAIR + 1))
        else
            NBOTH=$((NBOTH + 1))
        fi
    done
done

# THE g128 COLUMN, which the loop above cannot reach. It is ONE column
# carrying both build classes in its have string rather than a
# gcc/Watcom pair of columns, so the test is on that string. Nothing
# else looks: kernel_matrix_gate only asks whether the registration
# describes the binary, and `x G--` describes it perfectly honestly
# while saying nothing about the Watcom kernel that is not there.
#
# t2 WAS THAT CASE and is not any more: lz_dot128_t2_asm exists, so all
# five formats have the 128-element group form on both toolchains and
# the list is empty. It is kept - not deleted - because an empty
# baseline is still the mechanism: the first gcc-only g128 column to
# appear reddens the gate and has to be argued for here rather than
# accumulating quietly.
G128_OWED=""
G128_SEEN=""
for u in $UNITS; do
    st=$(cls "$u" g128)
    case "$st" in x|o) ;; *) continue ;; esac
    h=$(echo "$M" | awk -v u="$u" '$1=="cell" && $2==u && $3=="g128" {print $5; exit}')
    case "$h" in
        *G*W*) NPAIR=$((NPAIR + 1)) ;;
        *)
            owed=0
            for o in $G128_OWED; do [ "$o" = "$u" ] && owed=1; done
            if [ "$owed" = 1 ]; then
                G128_SEEN="$G128_SEEN $u"
                echo "note: $u g128 is '$h' - the 128-element group form is"
                echo "      gcc-only and the Watcom twin is owed (baselined)"
            else
                echo "FAIL: $u g128 is '$h' - the 128-element group form exists"
                echo "      for one toolchain and not the other, and this one"
                echo "      is not in the baseline"
                FAIL=1
            fi
            ;;
    esac
done

# AND THE BASELINE MUST STILL BE OWED. The check above is one-directional:
# it catches a second gap arriving and says nothing about the first one
# being closed. A paid debt left in G128_OWED is worse than no baseline,
# because the name goes on excusing a cell that no longer needs excusing
# - and if that cell ever regressed to gcc-only again, the gate would
# print "baselined" instead of failing. So a name that is no longer
# unpaired is itself a failure, and clearing the debt costs deleting it.
for o in $G128_OWED; do
    seen=0
    for s in $G128_SEEN; do [ "$s" = "$o" ] && seen=1; done
    if [ "$seen" = 0 ]; then
        echo "FAIL: '$o' is in G128_OWED but its g128 column is not"
        echo "      gcc-only any more - either the Watcom twin landed, in"
        echo "      which case delete the name, or the unit stopped being"
        echo "      probed and the baseline is now excusing nothing"
        FAIL=1
    fi
done

echo "gcc_watcom_pair_gate: $NPAIR paired, $NBOTH empty on both sides,"
echo "  $NOWED declared todo (bounded by twin_pair_gate, not here)"
[ "$FAIL" -eq 0 ] || { echo "gcc_watcom_pair_gate: FAIL"; exit 1; }
echo "gcc_watcom_pair_gate: PASS"
exit 0
