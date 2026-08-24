#!/bin/bash
# How much of each toolchain's output moves data rather than computing.
#
# xcc_codesize.sh counts instructions and xcc_mnemonics.sh lists which
# ones. Neither answers the question this is for: of the instructions a
# unit emits, what FRACTION are shuffling values between registers and
# memory instead of doing the work? That fraction is what register
# pressure and a poor calling convention look like from outside, and it
# is the one number that stays comparable across units of different
# sizes.
#
# CLASSES, and where the lines are drawn:
#   move     mov push pop xchg movsx movzx movs stos lods cmov
#            - and nothing else. These produce no value that did not
#              already exist somewhere.
#   lea      SEPARATE, on purpose. It is an address computation that
#            compilers also use as a three-operand add-and-shift, so
#            filing it either way would be a guess. Reported so a reader
#            can decide; excluded from the move share.
#   compute  add sub adc sbb imul mul idiv div and or xor not neg
#            inc dec shl shr sar rol ror cmp test set*
#   control  jmp jcc call ret loop rep*
#   fp       everything starting f, plus the SSE families
#
# HOW TO READ IT. Not against zero: prologues, spills at a call
# boundary and struct copies are real work and a share around a third is
# ordinary for 32-bit x86. Compare the two COLUMNS. Where Watcom's share
# runs well above gcc's on the same source, the source is asking for
# more live values than eight registers hold, and that is a source
# question - fewer live pointers in a loop, a smaller struct passed by
# value, a helper that takes two arguments instead of six.
#
# AND DO NOT RANK BY push/pop, which is what this was written after.
# chat.c emits 14 pushes and 18 pops under gcc against 85 and 104 under
# Watcom - a 6x gap that reads like Watcom shuffling far more. Its
# movement SHARE is 44% against 45%. The two compilers spend the same
# fraction of their output moving data and disagree only about the
# instruction: gcc spills to stack slots with mov, Watcom pushes. A
# census over one mnemonic pair answers which idiom, not how much work.
#
# WHAT THIS CANNOT SEE: how often any of it executes. A move in a
# per-token loop and a move in a startup path count the same here. Rank
# with it, do not conclude from it.
#
# Both sides at the shipping settings, 32-bit, for the reason in
# xcc_codesize.sh's header.
set -u
cd "$(dirname "$0")/.."
WCC=/c/WATCOM/binnt64/wcc386.exe
WDIS=/c/WATCOM/binnt64/wdis.exe
command -v gcc >/dev/null 2>&1 || { echo "SKIP: no gcc"; exit 2; }
[ -x "$WCC" ] || { echo "SKIP: no wcc386"; exit 2; }

D="-Isrc -Icommon -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

MOVE='^(mov|movl|movb|movw|push|pushl|pop|popl|xchg|movsx|movzx|movsb|movsl|movsw|movsd|stos|stosb|stosl|lods|lodsb|cmov)'
LEA='^lea'
FP='^(f[a-z]|adds|subs|muls|divs|cvt|ucomis|comis|pxor|movap|movup|movq|movd)'

# One awk over a mnemonic list -> "total move lea fp".
classify() {
    awk -v m="$MOVE" -v l="$LEA" -v f="$FP" '
        { t++
          if ($0 ~ f)      { fp++ }
          else if ($0 ~ l) { lea++ }
          else if ($0 ~ m) { mv++ } }
        END { printf "%d %d %d %d\n", t, mv + 0, lea + 0, fp + 0 }'
}

printf '%-14s %s\n' "" "        gcc                    watcom"
printf '%-14s %8s %7s %6s   %8s %7s %6s\n' \
       file total move lea total move lea
for src in "$@"; do
    [ -f "$src" ] || continue
    b=$(basename "$src" .c)
    gcc -O2 -m32 -march=i486 -mtune=pentium -std=c99 $D \
        -c -o "$W/$b.o" "$src" 2>/dev/null || continue
    "$WCC" -q -za99 -otexan -zp4 -4r -bt=nt -I/c/WATCOM/h -I/c/WATCOM/h/nt \
        $D -fo="$W/$b.obj" "$src" >/dev/null 2>&1 || continue

    G=$(objdump -d "$W/$b.o" 2>/dev/null \
        | awk -F'\t' 'NF >= 3 && $1 ~ /^ *[0-9a-f]+:/ {
              split($3, m, " "); if (m[1] != "") print m[1] }' | classify)
    K=$("$WDIS" "$W/$b.obj" 2>/dev/null \
        | awk -F'\t' 'NF > 1 && $1 ~ /^[0-9A-F][0-9A-F][0-9A-F][0-9A-F]  / \
                      && $1 !~ /   [A-Z][A-Z]$/ {
              for (i = 2; i <= NF; i++)
                  if ($i != "") { split($i, m, " "); print tolower(m[1]); break } }' \
        | classify)

    set -- $G; gt=$1; gm=$2; gl=$3
    set -- $K; kt=$1; km=$2; kl=$3
    [ "${gt:-0}" -gt 0 ] && [ "${kt:-0}" -gt 0 ] || continue
    printf '%-14s %8d %6d%% %5d   %8d %6d%% %5d\n' \
        "$b" "$gt" $((gm * 100 / gt)) "$gl" "$kt" $((km * 100 / kt)) "$kl"
done
