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
# Usage: bash build/o3_probe.sh [unit ...]
#        no arguments prints every unit the matrix currently calls
#        impossible, with the function that implements it.
set -u
cd "$(dirname "$0")/.."

command -v gcc >/dev/null 2>&1 || { echo "o3_probe: SKIP - no gcc"; exit 2; }
command -v objdump >/dev/null 2>&1 || { echo "o3_probe: SKIP - no objdump"; exit 2; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
D="-std=c99 -Isrc -Icommon -ffp-contract=off -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"

# unit:file:symbol. The symbol is the LOOP, not the leaf: a function
# that returns one value has nothing to vectorise, and the matrix's
# operator cells are about the loops that call those leaves.
MAP="
expfx:src/ops.c:lz_exp
sigq:src/forward.c:forward_attn
i32f:src/ops_quant.c:norm_ss_fixed
q8rnd:src/ops.c:lz_quantize_q8
nrmss:src/ops_quant.c:norm_ss_fixed
fwhti:src/fwht.c:lz_fwht_i32
fwhtf:src/ops.c:lz_hadamard_f32
f32mm:src/ops_matmul.c:lz_matmul_w
atdot:src/ops_gdn.c:lz_attn_score_q8
wsump:src/ops_gdn.c:lz_attn_wsum_q8
"

# Packed-SIMD mnemonics, and nothing scalar: the ss/sd forms are what
# the compiler emits for ordinary float code and would make every
# function look vectorised.
# AN OPTIONAL v PREFIX ON EVERY ONE, which the first version left off:
# with -march=x86-64-v3 gcc emits vmulps and vpaddd, so an AVX2 column
# read ZERO everywhere while the SSE2 column read a hundred. The number
# was measuring the pattern, not the compiler.
PACKED='\bv?(mulps|addps|subps|divps|maxps|minps|andps|andnps|orps|xorps'
PACKED="$PACKED"'|mulpd|addpd|subpd|cvtps2dq|cvtdq2ps|cvttps2dq'
PACKED="$PACKED"'|paddb|paddw|paddd|paddq|psubb|psubw|psubd|psubq'
PACKED="$PACKED"'|pmaddwd|pmullw|pmulld|pmulhw|pmulhuw|psadbw'
PACKED="$PACKED"'|packssdw|packsswb|packuswb|pmaxsw|pminsw|pmaxsd|pminsd'
PACKED="$PACKED"'|punpck[lh][a-z]+|pshufd|pshufb|pshufw|pshuflw|pshufhw'
PACKED="$PACKED"'|pcmpeq[bwd]|pcmpgt[bwd]|psll[wdq]|psrl[wdq]|psra[wd]'
PACKED="$PACKED"'|movdqa|movdqu|movaps|movups|movntdq|movntps'
PACKED="$PACKED"'|permd|permq|pgatherdd|gatherdps|pbroadcast[bwdq]'
PACKED="$PACKED"'|fmadd[0-9]*ps|insertf128|extractf128|perm2f128)\b'

probe() {
    local unit=$1 file=$2 sym=$3 arch=$4 tag=$5
    gcc -O3 -march="$arch" $D -c -o "$W/o.o" "$file" 2>/dev/null || {
        printf '  %-6s %-22s BUILD FAIL\n' "$unit" "$tag"; return; }
    local n
    n=$(objdump -d "$W/o.o" 2>/dev/null \
        | awk -v s="<$sym>:" 'index($0, s) {on=1} on && /\tret/ {print; exit} on' \
        | grep -coE "$PACKED")
    printf '  %-6s %-22s %s packed\n' "$unit" "$tag" "$n"
}

probe32() {
    local unit=$1 file=$2 sym=$3 arch=$4 tag=$5
    gcc -O3 -m32 -march="$arch" $D -c -o "$W/o32.o" "$file" 2>/dev/null || {
        printf '  %-6s %-22s BUILD FAIL\n' "$unit" "$tag"; return; }
    local n
    n=$(objdump -d "$W/o32.o" 2>/dev/null \
        | awk -v s="<_$sym>:" -v t="<$sym>:" 'index($0,s)||index($0,t) {on=1} on && /\tret/ {print; exit} on' \
        | grep -coE "$PACKED")
    printf '  %-6s %-22s %s packed\n' "$unit" "$tag" "$n"
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
