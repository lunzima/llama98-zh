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
# HOW TO READ THE RATIO. Not against 1.0. gcc -O3 inlines and unrolls,
# so it emits MORE instructions than Watcom for the same source while
# executing fewer - measured across the thirteen non-operator TUs, the
# ratio centres near 0.75 and only two files exceed 1.0. A file at 1.0
# is therefore already 1.3x off the pack, not "even". Rank by distance
# from the SET'S OWN median, printed below the table, and treat the
# number as a screen that says where to go disassemble - never as a
# verdict, because instruction count is not cycle count and this repo
# has already measured one case where the shorter sequence was slower.
set -u
WCC=/c/WATCOM/binnt64/wcc386.exe
[ -x "$WCC" ] || { echo "SKIP: no wcc386"; exit 2; }
D="-Isrc -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

RATIOS=""
printf '%-20s %8s %8s %6s\n' file gcc wat ratio
for f in "$@"; do
    b=$(basename "$f" .c)
    gcc -O3 -march=x86-64 -std=c99 $D -c -o "$W/$b.o" "$f" 2>/dev/null || continue
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
    echo "not from 1.0 - gcc -O3 unrolls, so parity is already an outlier)"
else
    echo
    echo "($N file(s): too few for a median - the ratio has no baseline"
    echo " here and a single number is not a reading)"
fi
