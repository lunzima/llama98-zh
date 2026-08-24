#!/bin/bash
# Same source, two toolchains: how many instructions does each emit?
#
# The tree's other performance criteria answer other machines - static
# f32 counts and branch evaluations describe a P5, dev_timing.sh
# describes this host. None of them can say "Watcom generates far worse
# code than gcc for THIS function", which is what the non-operator
# optimisation question is.
#
# Instructions, not bytes: x86 encodings differ in length for reasons
# that have nothing to do with how well the code was compiled, and the
# .obj/.o container sizes are not comparable at all (OMF vs COFF, and
# gcc's carries debug sections Watcom's does not).
#
# BOTH SIDES 32-BIT. wcc386 emits 32-bit code with eight registers and
# stack-passed arguments beyond the first few; a gcc left at the host
# default emits x86-64, with sixteen registers and six register-passed
# arguments. Comparing those two measures the ISA, not the compiler -
# the first version of this script did exactly that, and the per-mnemonic
# view of the same objects showed Watcom emitting a hundred more pushes,
# which is what eight registers costs and says nothing about codegen.
# -march=i486 -mtune=pentium matches what the Makefile actually ships
# (-4r, tuned for an in-order multi-issue core).
#
# -O2, the release level, NOT -O3. Measuring an optimisation level the
# project does not ship describes a machine nobody runs; -O3 inflated
# the shipped binary 19% and was dropped for exactly that reason, so a
# ratio taken at -O3 would be answering about the discarded build.
#
# HOW TO READ THE RATIO. Not against 1.0. gcc still inlines more than
# Watcom at -O2, so it emits MORE instructions for the same source while
# executing fewer. Rank by distance from the SET'S OWN median, printed
# below the table, and treat the number as a screen that says where to
# go disassemble - never as a verdict, because instruction count is not
# cycle count and this repo has already measured one case where the
# shorter sequence was slower.
#
# THE RANKING MOVED THREE TIMES WHILE THIS SCRIPT WAS BEING CORRECTED,
# and every "worst file" named before the last correction came from a
# build nobody ships:
#   x86-64 gcc vs 32-bit Watcom   compat 1.22 was the top outlier
#   32-bit both, -O3              fwht 1.05 was
#   32-bit both, -O2 (shipping)   chat 1.17 was, and was real
# Only the third row is a reading. The first two measured the register
# file and an optimisation level the project dropped.
#
# WHAT THE SWEEP FOUND, once the instrument was right. One defect of the
# kind this dimension is for - a construct the new compiler folds and the
# old one does not - in chat.c: strlen of a string literal, emitted as
# repne scasb twelve times. Fixed, 1.17 -> 1.06. The remaining outliers
# are lfn 1.08 and net 1.04, both cold, and tokenizer 0.99, which is hot
# but shows nothing of that kind: no repne, no redundant library call,
# and its surplus is push/pop, which move_census.sh shows is idiom rather
# than volume - Watcom moves a SMALLER fraction there than gcc does.
# Absence of a finding is a result; recorded so the next sweep starts
# from it rather than repeating it.
set -u
WCC=/c/WATCOM/binnt64/wcc386.exe
[ -x "$WCC" ] || { echo "SKIP: no wcc386"; exit 2; }
D="-Isrc -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

# A DEFAULT FILE SET, because a census that has to be handed one cannot
# be run as a standing check and gets run as `bash build/xcc_codesize.sh`
# instead - which prints an empty table and a "0 files" line that reads
# like a clean result to anyone skimming.
#
# The set is the NON-OPERATOR code, which is the question this script
# exists for: the kernels are hand-written on both sides and a ratio
# over them measures two people's assembly, not two compilers'
# optimisers. src/ops*.c and the three SIMD units are therefore out;
# what is left is the front end, the loaders, the tokenizer and the
# session core, where gcc's advantage over a 1990s compiler is the
# thing the requirement is about.
FILES="$*"
if [ -z "$FILES" ]; then
    FILES=$(ls src/*.c common/*.c 2>/dev/null \
            | grep -vE "src/ops[a-z0-9_]*\.c$" \
            | grep -vE "src/(forward|fwht)\.c$")
fi

RATIOS=""
printf '%-20s %8s %8s %6s\n' file gcc wat ratio
for f in $FILES; do
    b=$(basename "$f" .c)
    gcc -O2 -m32 -march=i486 -mtune=pentium -std=c99 $D \
        -c -o "$W/$b.o" "$f" 2>/dev/null || continue
    g=$(objdump -d "$W/$b.o" 2>/dev/null | grep -cE "^ *[0-9a-f]+:")
    "$WCC" -q -za99 -otexan -zp4 -4r -bt=nt -I/c/WATCOM/h -I/c/WATCOM/h/nt \
        -Isrc -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -fo="$W/$b.obj" "$f" >/dev/null 2>&1 || continue
    # An INSTRUCTION line, not any addressed line. wdis prints const
    # tables with the same four-hex-digit address prefix, so counting
    # that prefix counts unicode_tables.h's data as code and reports a
    # blow-up that is entirely arrays. objdump -d on the gcc side only
    # ever walks .text, so the two columns would not be the same
    # quantity at all.
    #
    # Requiring a tab is NOT enough on its own. String bytes print with
    # their ASCII rendering after spaces and are excluded by NF > 1, but
    # a POINTER table prints as "DD<tab>offset L$6" - a tab, and the DD
    # sits at the END of the byte column padded with spaces, so it looks
    # like part of the bytes. err.c is 53 instructions and 266 of those
    # rows; uncorrected it read as 3.54x, the worst ratio in the tree
    # and pure artifact. Hence the second clause. Verified both ways
    # over all thirteen files: 404 rows classed directive, every one DD;
    # 31791 classed instruction, every mnemonic lower-case (the only
    # ones with digits are real x87 - fld1, f2xm1, fyl2x, fldl2e).
    #
    # awk on TAB, not grep -P: grep -P refuses a pattern outside a
    # UTF-8 locale and exits 2, which a naive `|| echo 0` reads as
    # "no instructions found".
    w=$(/c/WATCOM/binnt64/wdis.exe "$W/$b.obj" 2>/dev/null \
        | awk -F'\t' 'NF > 1 && $1 ~ /^[0-9A-F][0-9A-F][0-9A-F][0-9A-F]  / \
                      && $1 !~ /   [A-Z][A-Z]$/ { c++ } END { print c + 0 }')
    [ "$g" -gt 0 ] && [ "$w" -gt 0 ] || continue
    R=$(awk -v a="$g" -v b="$w" 'BEGIN{printf "%.2f", b/a}')
    RATIOS="$RATIOS $R"
    printf '%-20s %8s %8s %6s\n' "$b" "$g" "$w" "$R"
done

# The median is the baseline, for the reason in the header. Printed with
# the count so a one-file invocation cannot be mistaken for a set.
N=$(printf '%s\n' $RATIOS | wc -w)
if [ "$N" -ge 3 ]; then
    MED=$(printf '%s\n' $RATIOS | sort -n | awk '{a[NR]=$1} END{
        print (NR % 2) ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2 }')
    echo
    echo "median ratio over $N files: $MED  (rank by distance from THIS,"
    echo "not from 1.0 - gcc inlines more, so parity is already an outlier)"
else
    echo
    echo "($N file(s): too few for a median - the ratio has no baseline"
    echo " here and a single number is not a reading)"
fi
