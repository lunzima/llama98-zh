#!/bin/bash
# What does x86-64 gcc -O3 actually emit for the loops the kernel matrix
# calls impossible?
#
# WHY THIS EXISTS. A cell registered impossible carries a sentence, and
# a sentence is an argument. The compiler is the cheapest referee
# available: point it at the same C with every optimisation on and look
# at what comes out. If gcc vectorises a loop the matrix says cannot be
# vectorised, the matrix is wrong and the disassembly says so without
# anyone having to re-derive an ISA table. Three cells were already
# found wrong this way by hand - q8rnd, i32f and nrmss - and each had a
# reason that read plausibly.
#
# -O3 AND NOT -O2, deliberately, and it is not a contradiction with the
# release level. The release ships -O2 because -O3 costs .text for
# nothing on hand-written intrinsics (see codesize_gate). Here the
# question is not what to ship, it is what the compiler CAN see: -O3
# turns on the vectoriser that -O2 leaves off, so it is the upper bound
# on "a compiler could do this without a hand-written kernel".
#
# -march=x86-64 is SSE2 baseline. A v3 pass is run alongside it because
# some of the impossible reasons name AVX2 specifically ("a per-element
# table index needs a gather, and that is AVX2"), and the way to check
# that claim is to give gcc AVX2 and see whether it takes it.
#
# WHAT A COUNT MEANS AND DOES NOT. Packed instructions in a function are
# evidence the loop CAN be vectorised, not that the vectorisation is
# profitable or that it matches the target ISA - x86-64 has SSE2 that a
# Pentium does not. Read it as "the impossible needs re-examining
# here", never as "ship this".
#
# AND IT IS FUNCTION GRANULARITY, which is coarser than the question.
# expfx reads arith=127 at SSE2, but its loop lives in lz_moe_route
# beside a max scan, a sum and a normalise, and those three vectorise
# trivially. The number says nothing about the exp loop.
#
# Settling one takes a throwaway TU holding ONE loop and two controls -
# a loop gcc must vectorise, and the same shape minus the thing under
# suspicion. Done for expfx:
#
#     ctl_mul  o[i] = x[i]*y[i]        insns=73  arith=2   <- works
#     iso_exp  o[i] = lz_exp(x[i])     insns=28  arith=0
#     ctl_tab  o[i] = tab[idx[i] & 31] insns=22  arith=0
#
# Identical at -march=x86-64 and at -march=x86-64-v3. The control fires,
# so the measurement is live; the subject is zero; and ctl_tab - a bare
# per-element table index with no transcendental in it - is zero too,
# which is what isolates the CAUSE to the index rather than to the
# arithmetic around it. Offered AVX2, where vgatherdps exists, gcc still
# declines. expfx's reason is confirmed at the granularity it is stated.
#
# THAT FINDING IS WHY THE HAND-WRITTEN CELL HAS THE SHAPE IT HAS. The
# blocker it isolated is the per-element table INDEX, not the Q20 Taylor
# around it - so lz_exp_fixed_run walks the lookup in scalar and hands
# the kernel the two int32 arrays it produced. gcc still will not
# vectorise the original loop and that is still the right answer; the
# operator ships a kernel anyway, because the part gcc cannot reach and
# the part worth reaching are not the same part.
#
# So a nonzero here opens a question; it does not answer one.
#
# Usage: bash build/o3_probe.sh [unit ...]
#        no arguments walks every unit in MAP below.
#
# MAP IS HAND-MAINTAINED AND CANNOT BE DERIVED: the matrix knows which
# cells are impossible, but not which C function implements a unit, and
# picking that function is a judgement (see the three corrections above,
# each of which was a MAP entry pointing at a function the unit does not
# dominate). So the list is written out - and then CHECKED against the
# matrix at the end of the run, because a hand-maintained list of names
# is exactly the thing that goes stale silently. A unit that acquires an
# impossible cell and has no MAP entry would otherwise just not be
# probed, and the absence would look like a clean sheet.
#
# This text used to say the run covered "every unit the matrix currently
# calls impossible". It never did: MAP is every unit, impossible or not,
# and nothing compared the two lists.
set -u
cd "$(dirname "$0")/.."

command -v gcc >/dev/null 2>&1 || { echo "o3_probe: SKIP - no gcc"; exit 2; }
command -v objdump >/dev/null 2>&1 || { echo "o3_probe: SKIP - no objdump"; exit 2; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
D="-std=c99 -Isrc -Icommon -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"

# unit:file:symbol. The symbol is the LOOP, not the leaf: a function
# that returns one value has nothing to vectorise, and the matrix's
# operator cells are about the loops that call those leaves.
#
# TWO OF THESE POINTED AT THE WRONG FUNCTION and reported confidently.
# expfx was lz_exp, an eight-instruction dispatcher that tail-calls its
# two bodies - not a loop at all, and the rule above says so. Its real
# per-element loop is lz_moe_route's router softmax, `score[i] =
# lz_exp(logits[i] - m)`. i32f was norm_ss_fixed, which is nrmss's body;
# the two units printed identical numbers for that reason and one of
# them was measuring the other's operator.
#
# i32f AND wsump NOW SHARE A SYMBOL, and that is correct rather than a
# second copy of the same mistake: i32f's own reason says the fold "sits
# inside a SIMD kernel's own epilogue, after the emms", and that kernel
# is lz_attn_wsum_q8. Two units in one function print two identical
# rows; the alternative is pointing one of them somewhere it does not
# live.
#
# sigq was forward_attn, 684 lines of attention scoring, RoPE and
# softmax - a number off that function describes everything except the
# sigmoid. Its per-element loop is lz_rmsnorm_gated_out_fixed, whose
# body is `sigv = sigmoid_q15(g[i])` over the gate vector. Third entry
# of the ten found pointing somewhere the unit does not live; the
# check is to read the named function and confirm the operator's
# arithmetic is what dominates it.
MAP="
expfx:src/ops.c:lz_moe_route
sigq:src/ops_norm.c:lz_rmsnorm_gated_out_fixed
i32f:src/ops_gdn.c:lz_attn_wsum_q8
q8rnd:src/ops.c:lz_quantize_q8
nrmss:src/ops_quant.c:norm_ss_fixed
fwhti:src/fwht.c:lz_fwht_i32
fwhtf:src/ops.c:lz_fwht
f32mm:src/ops.c:lz_matmul_w
atdot:src/ops_gdn.c:lz_attn_score_q8
wsump:src/ops_gdn.c:lz_attn_wsum_q8
q8_0:src/ops_matmul.c:matmul_q8_impl
q4_1:src/ops_matmul.c:matmul_q41_impl
q6_1:src/ops_matmul.c:matmul_q61_impl
q16_0:src/ops_matmul.c:matmul_q16_impl
t2:src/ops_matmul.c:matmul_t2_impl
"

# THE FIVE WEIGHT-FORMAT UNITS WERE MISSING, and that is what the rule
# they serve makes serious: every cell the matrix calls impossible gets
# a look at what gcc -O3 emits. These five carry impossible cells - the
# sseI column on the "SSE1 adds no signed 16x16->32 MAC and no integer
# %xmm" reason, and g128 - and no row here at all, so the rule held for
# ten of fifteen units and nobody could see which five it did not.
#
# Comparing this list against the matrix is one command, and it is worth
# running whenever a unit gains an impossible cell:
#
#   llama98 --kernel-matrix | awk '$4=="i"{print $2}' | sort -u
#
# against the unit names on the left below.

# Packed-SIMD mnemonics, and nothing scalar: the ss/sd forms are what
# the compiler emits for ordinary float code and would make every
# function look vectorised.
# AN OPTIONAL v PREFIX ON EVERY ONE, which the first version left off:
# with -march=x86-64-v3 gcc emits vmulps and vpaddd, so an AVX2 column
# read ZERO everywhere while the SSE2 column read a hundred. The number
# was measuring the pattern, not the compiler.
PACKED='\bv?(mulps|addps|subps|divps|maxps|minps'
PACKED="$PACKED"'|mulpd|addpd|subpd|cvtps2dq|cvtdq2ps|cvttps2dq'
PACKED="$PACKED"'|paddb|paddw|paddd|paddq|psubb|psubw|psubd|psubq'
PACKED="$PACKED"'|pmaddwd|pmullw|pmulld|pmulhw|pmulhuw|psadbw'
PACKED="$PACKED"'|packssdw|packsswb|packuswb|pmaxsw|pminsw|pmaxsd|pminsd'
PACKED="$PACKED"'|punpck[lh][a-z]+|pshufd|pshufb|pshufw|pshuflw|pshufhw'
PACKED="$PACKED"'|pcmpeq[bwd]|pcmpgt[bwd]|psll[wdq]|psrl[wdq]|psra[wd]'
PACKED="$PACKED"'|permd|permq|pgatherdd|gatherdps|pbroadcast[bwdq]'
PACKED="$PACKED"'|fmadd[0-9]*ps|insertf128|extractf128|perm2f128'
# THE PATTERN WAS SSE2-SHAPED, so the AVX2 column undercounted the one
# thing it exists to referee. matmul_q41_impl at -march=x86-64-v3 holds
# 8 vpmovsxbw and 4 vpmovsxwd - the packed int8->int16->int32 widening
# that IS an int8 matmul's inner loop - and the count was 7, being
# vpaddd six times and vpsrlq once. Everything else went unseen, which
# is why every AVX2 row read LOWER than its SSE2 row and looked like
# wider registers doing the same work in fewer instructions.
#
# pmovsx/pmovzx are conversions, the same class as cvtdq2ps above, not
# register copies. The 128-lane halves have integer spellings
# (extracti128, inserti128) as well as the float ones already listed,
# and the variable shifts are AVX2-only. Lane manipulation counts here
# because pshufd and punpck already do.
PACKED="$PACKED"'|pmovsx[bwd][wdq]|pmovzx[bwd][wdq]|packusdw'
PACKED="$PACKED"'|inserti128|extracti128|psrldq|pslldq'
PACKED="$PACKED"'|psrlv[dq]|psllv[dq]|psrav[dq])\b'

# MOVES AND BITWISE ARE NOT EVIDENCE, and keeping them in the count above
# is how this script reported that gcc vectorises lz_exp. It does not:
# at -O3 -march=x86-64 that function is 163 instructions of which the
# "22 packed" were 20 movaps and 2 pshufd - zero packed ARITHMETIC.
# movaps is how gcc copies an xmm register in ordinary scalar float
# code, and xorps/andps are how it negates and takes absolute value, so
# a pattern holding them fires on exactly the scalar code this script
# exists to distinguish from vector code.
#
# They are still counted, and printed in their own column, because
# "gcc moved a lot of vectors around here" is worth seeing next to
# "and did no vector arithmetic". It is context, never the verdict.
MOVES='\bv?(movdqa|movdqu|movaps|movups|movntdq|movntps'
MOVES="$MOVES"'|andps|andnps|orps|xorps|andpd|xorpd)\b'

# THE WHOLE FUNCTION, not up to its first `ret`. Stopping there truncates
# any function with an early return - lz_exp_fixed's domain guards are
# two of them before a single loop iteration - and the loss is silent and
# one-directional. objdump separates functions with a blank line, so that
# is the terminator.
#
# REPORTS insns TOO, and that is not decoration. A symbol that was not
# found yields an empty body and "0 packed", which is indistinguishable
# from a function the compiler declined to vectorise. This script had ten
# rows of "0 packed" in its SSE1 column and no way to tell which kind
# they were. insns=0 means the probe missed; insns=163 arith=0 means it
# looked.
body_of() {
    objdump -d "$1" 2>/dev/null \
        | awk -v s="<$2>:" -v u="<_$2>:" \
              'index($0,s)||index($0,u) {on=1; next} on && /^$/ {exit} on'
}

report() {
    local unit=$1 tag=$2 body=$3
    local insns arith moves
    insns=$(printf '%s' "$body" | grep -c '')
    arith=$(printf '%s' "$body" | grep -coE "$PACKED")
    moves=$(printf '%s' "$body" | grep -coE "$MOVES")
    if [ "$insns" -eq 0 ]; then
        printf '  %-6s %-22s SYMBOL NOT FOUND\n' "$unit" "$tag"
        return
    fi
    printf '  %-6s %-22s insns=%-5s arith=%-4s moves=%s\n' \
           "$unit" "$tag" "$insns" "$arith" "$moves"
}

probe() {
    local unit=$1 file=$2 sym=$3 arch=$4 tag=$5
    gcc -O3 -march="$arch" $D -c -o "$W/o.o" "$file" 2>/dev/null || {
        printf '  %-6s %-22s BUILD FAIL\n' "$unit" "$tag"; return; }
    report "$unit" "$tag" "$(body_of "$W/o.o" "$sym")"
}

probe32() {
    local unit=$1 file=$2 sym=$3 arch=$4 tag=$5
    gcc -O3 -m32 -march="$arch" $D -c -o "$W/o32.o" "$file" 2>/dev/null || {
        printf '  %-6s %-22s BUILD FAIL\n' "$unit" "$tag"; return; }
    report "$unit" "$tag" "$(body_of "$W/o32.o" "$sym")"
}

SEL="${*:-}"
echo "gcc -O3, packed-SIMD instruction count inside each unit's loop"
echo "(evidence the loop CAN vectorise, not that it should - see header)"
for e in $MAP; do
    u=${e%%:*}; rest=${e#*:}; f=${rest%%:*}; s=${rest#*:}
    [ -z "$SEL" ] || case " $SEL " in *" $u "*) ;; *) continue ;; esac
    [ -f "$f" ] || { printf '  %-6s missing %s\n' "$u" "$f"; continue; }
    probe "$u" "$f" "$s" x86-64 "-march=x86-64 (SSE2)"
    probe "$u" "$f" "$s" x86-64-v3 "-march=x86-64-v3 (AVX2)"
    # AND AN SSE1-ONLY TARGET, because fourteen of the impossible cells
    # say SSE1 specifically ("SSE1 adds no signed 16x16->32 MAC and no
    # integer %xmm"). x86-64 is SSE2 baseline, so a hundred packed
    # instructions there says nothing about that claim - the referee for
    # an SSE1 cell has to be an SSE1 machine.
    probe32 "$u" "$f" "$s" pentium3 "-m32 -march=pentium3 (SSE1)"
done

# COMPLETENESS: every unit the matrix calls impossible must be in MAP.
# The binary is the source of truth for which those are - not this file,
# which is what went stale. Reported even when a unit subset was
# requested, because the gap is a property of MAP and not of the run.
#
# Silent on the other direction: MAP carrying a unit with no impossible
# cell left is not a defect. expfx sits there today with all three x86
# tiers written, and probing it is still how anyone checks whether the
# hand-written kernels are worth keeping.
BIN=./llama98.exe
[ -x "$BIN" ] || BIN=./llama98
if [ -x "$BIN" ]; then
    # MAP is newline-separated; unquoted expansion rejoins it with single
    # spaces so one pattern can carry the leading delimiter. Matching
    # bare "$u:" instead would be a substring test, and unit names are
    # prefixes of each other here (q8_0 / q8rnd, fwhti / fwhtf).
    MAPSP=" $(echo $MAP) "
    miss=
    for u in $("$BIN" --kernel-matrix 2>/dev/null \
               | awk '$1=="cell" && $4=="i" {print $2}' | sort -u); do
        case "$MAPSP" in
            *" $u:"*) ;;
            *) miss="$miss $u" ;;
        esac
    done
    if [ -n "$miss" ]; then
        echo
        echo "o3_probe: NOT PROBED -$miss"
        echo "  the matrix calls these impossible and MAP has no entry;"
        echo "  add one naming the function whose loop the unit dominates"
        exit 1
    fi
else
    echo
    echo "o3_probe: MAP completeness unchecked - no engine binary to ask"
fi
