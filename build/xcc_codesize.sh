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
# The RATIO is the reading, not either column. A file where Watcom emits
# 1.1x gcc is normal; one at 2x is a place where an idiom the newer
# compiler folds is being written out longhand.
set -u
WCC=/c/WATCOM/binnt64/wcc386.exe
[ -x "$WCC" ] || { echo "SKIP: no wcc386"; exit 2; }
D="-Isrc -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

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
    # 4x blow-up that is entirely arrays. objdump -d on the gcc side
    # only ever walks .text, so the two columns were not the same
    # quantity at all. The mnemonic is what makes a line an instruction.
    # awk on TAB, not grep -P: wdis separates the byte column from the
    # mnemonic with tabs, and `grep -P` refuses a pattern outside a
    # UTF-8 locale and exits 2 - which a naive `|| echo 0` reads as
    # "no instructions found".
    w=$(/c/WATCOM/binnt64/wdis.exe "$W/$b.obj" 2>/dev/null \
        | awk -F'\t' 'NF > 1 && $1 ~ /^[0-9A-F][0-9A-F][0-9A-F][0-9A-F]  / { c++ } END { print c + 0 }')
    [ "$g" -gt 0 ] && [ "$w" -gt 0 ] || continue
    printf '%-20s %8s %8s %6s\n' "$b" "$g" "$w" \
        "$(awk -v a="$g" -v b="$w" 'BEGIN{printf "%.2f", b/a}')"
done
