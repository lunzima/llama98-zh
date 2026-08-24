#!/bin/bash
# WHAT the old toolchain emits that the new one does not.
#
# xcc_codesize.sh ranks translation units by instruction-count ratio.
# That screen cannot distinguish "gcc unrolled this loop" from "Watcom
# compiled this badly", and it is blind to the case that matters most
# for the source: BOTH compilers emitting a library call where a
# different source form would need none. A count cannot see a call; a
# histogram can.
#
# Per file, the mnemonics Watcom emits more of, and the call targets
# each side emits. Read the call lists first - a helper call in the
# Watcom column that is absent from the gcc one is a construct gcc
# lowers inline and Watcom does not, which is exactly the class the
# non-operator optimisation question asks about.
#
# Instruction rows only, by the same discriminator xcc_codesize.sh uses:
# wdis prints constant tables with an address prefix indistinguishable
# from code unless the trailing directive column is excluded.
set -u
WCC=/c/WATCOM/binnt64/wcc386.exe
WDIS=/c/WATCOM/binnt64/wdis.exe
[ -x "$WCC" ] || { echo "SKIP: no wcc386"; exit 2; }
D="-Isrc -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

for f in "$@"; do
    b=$(basename "$f" .c)
    # 32-bit both sides, and tuned the way the Makefile ships. See
    # xcc_codesize.sh's header: an x86-64 gcc against a 32-bit Watcom
    # measures the register file, not the compiler.
    gcc -O2 -m32 -march=i486 -mtune=pentium -std=c99 $D \
        -c -o "$W/$b.o" "$f" 2>/dev/null || continue
    "$WCC" -q -za99 -otexan -zp4 -4r -bt=nt -I/c/WATCOM/h -I/c/WATCOM/h/nt \
        $D -fo="$W/$b.obj" "$f" >/dev/null 2>&1 || continue

    # Tab-separated, like the Watcom side. Scanning whitespace fields for
    # "the first thing shaped like an identifier" picks up the ENCODED
    # BYTES: c0, be, ad and da are all valid two-hex-digit bytes and all
    # match [a-z][a-z0-9]*, so any instruction with one of them in its
    # encoding was filed under that byte instead of its mnemonic. The
    # tell was xor and test reading 0 for gcc, which is not a thing an
    # x86 compiler does.
    objdump -d "$W/$b.o" 2>/dev/null \
        | awk -F'\t' 'NF >= 3 && $1 ~ /^ *[0-9a-f]+:/ {
              split($3, m, " "); if (m[1] != "") print m[1] }' \
        | sort | uniq -c | sort -rn > "$W/g.txt"
    # The mnemonic is the first NON-EMPTY tab field after the address and
    # byte column - wdis pads that column with a varying number of tabs,
    # so $2 is empty on most rows and taking it yields one histogram
    # bucket named "". xcc_codesize.sh shares this row filter and is
    # unaffected because it only counts rows.
    "$WDIS" "$W/$b.obj" 2>/dev/null \
        | awk -F'\t' 'NF > 1 && $1 ~ /^[0-9A-F][0-9A-F][0-9A-F][0-9A-F]  / \
                      && $1 !~ /   [A-Z][A-Z]$/ {
              for (i = 2; i <= NF; i++)
                  if ($i != "") { split($i, m, " "); print tolower(m[1]); break } }' \
        | sort | uniq -c | sort -rn > "$W/w.txt"

    echo "===== $b ====="
    # Call targets: the reading that matters, so it goes first.
    # -dr, not -d: in an UNLINKED object every external call still points
    # at its own relocation placeholder, which objdump renders as
    # "<containing_function+0x...>". Read that column and every call in
    # the file looks like a call to whatever function it sits inside.
    # The callee's name is in the relocation record on the next line.
    #
    # IMAGE_REL_AMD64_REL32, not R_X86_64_PLT32. This host's gcc emits
    # PE-COFF, and the ELF relocation names match nothing here - the
    # first version of this line silently printed an empty call list for
    # every file while the Watcom column beside it looked fine.
    echo "  calls, gcc:"
    objdump -dr "$W/$b.o" 2>/dev/null \
        | grep -A1 -E "^\s+[0-9a-f]+:.*\bcall\b" \
        | grep -oE "(IMAGE_REL_AMD64_REL32|R_X86_64_(PLT32|PC32))[[:space:]]+\S+" \
        | awk '{ sub(/[-+].*$/, "", $2); if ($2 !~ /^\./) print $2 }' \
        | sort | uniq -c | sort -rn | sed 's/^/    /'
    echo "  calls, watcom:"
    "$WDIS" "$W/$b.obj" 2>/dev/null | grep -oE "call[[:space:]]+[A-Za-z_][A-Za-z0-9_@$]*" \
        | awk '{ print $2 }' | sort | uniq -c | sort -rn | sed 's/^/    /'
    # Then the mnemonics Watcom emits at least 8 more of, which is where
    # a systematic difference shows and a one-off does not.
    # Two-file awk rather than join: join needs both inputs sorted on the
    # key in the collation order it expects, and getting that wrong does
    # not error - it silently collapses every mnemonic into one row, which
    # is what the first version of this printed.
    echo "  watcom surplus (>=8):"
    awk 'NR == FNR { g[$2] = $1; next }
         { d = $1 - (($2 in g) ? g[$2] : 0)
           if (d >= 8) printf "    %-10s gcc %5d  wat %5d  +%d\n",
                              $2, ($2 in g) ? g[$2] : 0, $1, d }' \
        "$W/g.txt" "$W/w.txt" | sort -k7 -t+ -rn
done
