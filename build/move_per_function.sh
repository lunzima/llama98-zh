#!/bin/bash
# Per-function move instruction breakdown for the three operator files
# where Watcom's move share exceeds gcc's by 5+ points.
#
# move_census.sh identifies the files; this script names the functions.
# The three files are:
#   ops_norm   gcc 35%  watcom 43%  gap +8
#   ops_gdn    gcc 37%  watcom 42%  gap +6
#   ops_quant  gcc 29%  watcom 37%  gap +6
#
# OUTPUT: per function, gcc total/move%, watcom total/move%, gap.
# Functions sorted by gcc instruction count (largest first). Only
# functions with 50+ instructions on either side are shown — smaller
# ones contribute nothing to the file-level gap.
#
# The gap is watcom_move% - gcc_move%. A positive gap means Watcom
# spends more of its instructions on data movement for that function.
# The file-level gap is the weighted average of these, so a few large
# functions with big gaps dominate.
#
# WHAT TO DO WITH THE RESULT. Same as move_census.sh: the gap is a
# property of the calling convention and register pressure, not a bug.
# The next step after this script names a function is to put a counter
# in it and generate one token — if it does not execute, the gap is
# irrelevant. See move_census.sh's generate.c analysis for the full
# reading guide.
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

# classify_per_function TOOLCHAIN MNEMONICS_FILE
# Reads mnemonics with function boundaries marked as "@@FUNC:name".
# Prints: name total move lea fp
classify_per_function() {
    awk -v m="$MOVE" -v l="$LEA" -v f="$FP" '
        /^@@FUNC:/ {
            if (name != "" && t > 0)
                printf "%s %d %d %d %d\n", name, t, mv+0, lea+0, fp+0
            name = substr($0, 8)
            t = 0; mv = 0; lea = 0; fp = 0
            next
        }
        {
            t++
            if ($0 ~ f)      { fp++ }
            else if ($0 ~ l) { lea++ }
            else if ($0 ~ m) { mv++ }
        }
        END {
            if (name != "" && t > 0)
                printf "%s %d %d %d %d\n", name, t, mv+0, lea+0, fp+0
        }'
}

# gcc_per_function SRC_FILE
# Compiles with gcc -m32, extracts mnemonics with function boundaries.
gcc_per_function() {
    local src="$1" b
    b=$(basename "$src" .c)
    gcc -O2 -m32 -march=i486 -mtune=pentium -std=c99 $D \
        -c -o "$W/$b.o" "$src" 2>/dev/null || return 1
    objdump -d "$W/$b.o" 2>/dev/null | awk -F'\t' '
        # Function label: "00000000 <_name>:"
        /^[0-9a-f]+ <.*>:/ {
            name = $0
            gsub(/^[0-9a-f]+ </, "", name)
            gsub(/>:.*$/, "", name)
            # Strip leading underscore (gcc convention)
            sub(/^_/, "", name)
            # Skip data symbols
            if (name ~ /^LZ_/ || name ~ /^[A-Z_]+$/) next
            print "@@FUNC:" name
            next
        }
        # Instruction line
        NF >= 3 && $1 ~ /^ *[0-9a-f]+:/ {
            split($3, m, " ")
            if (m[1] != "") print m[1]
        }' | classify_per_function
}

# wat_per_function SRC_FILE
# Compiles with Watcom, extracts mnemonics with function boundaries.
wat_per_function() {
    local src="$1" b
    b=$(basename "$src" .c)
    "$WCC" -q -za99 -otexan -zp4 -4r -bt=nt -I/c/WATCOM/h -I/c/WATCOM/h/nt \
        $D -fo="$W/$b.obj" "$src" >/dev/null 2>&1 || return 1
    "$WDIS" "$W/$b.obj" 2>/dev/null | awk -F'\t' '
        # Function label: "0000				name_:" (tabs, not spaces)
        /^[0-9A-F][0-9A-F][0-9A-F][0-9A-F]				[^ ]/ {
            name = $0
            sub(/^[0-9A-F][0-9A-F][0-9A-F][0-9A-F]				/, "", name)
            sub(/:$/, "", name)
            # Skip local labels (L$N)
            if (name ~ /^L\$[0-9]+$/) next
            # Skip data symbols (start with _ and uppercase)
            if (name ~ /^_[A-Z]/) next
            # Strip trailing underscore (Watcom convention)
            sub(/_$/, "", name)
            print "@@FUNC:" name
            next
        }
        # Instruction line: "0030  83 EC 04				sub		esp,0x4"
        /^[0-9A-F][0-9A-F][0-9A-F][0-9A-F]  [0-9A-F]/ {
            for (i = 2; i <= NF; i++) {
                if ($i != "") {
                    split($i, m, " ")
                    print tolower(m[1])
                    break
                }
            }
        }' | classify_per_function
}

# merge_and_report GCC_FILE WAT_FILE LABEL
# Merges per-function data from both toolchains, prints a table.
merge_and_report() {
    local gfile="$1" wfile="$2" label="$3"
    echo "=== $label ==="
    printf '%-35s %10s %8s   %10s %8s %6s\n' \
           "function" "gcc total" "move" "wat total" "move" "gap"
    # Merge on function name, sort by gcc total descending
    join -j 1 \
        <(sort -k1,1 "$gfile") \
        <(sort -k1,1 "$wfile") \
    | awk '{printf "%s %d %d %d %d\n", $1, $2, $3, $6, $7}' \
    | sort -k2,2 -rn \
    | while read name gt gm kt km; do
        [ "$gt" -lt 50 ] && [ "$kt" -lt 50 ] && continue
        gpct=$((gm * 100 / gt))
        kpct=$((km * 100 / kt))
        gap=$((kpct - gpct))
        printf '%-35s %10d %7d%%   %10d %7d%% %+5d\n' \
            "$name" "$gt" "$gpct" "$kt" "$kpct" "$gap"
    done
    echo
}

SRCS="src/ops_norm.c src/ops_gdn.c src/ops_quant.c"

for src in $SRCS; do
    [ -f "$src" ] || continue
    b=$(basename "$src" .c)
    echo "Compiling $src ..."

    gout="$W/${b}_gcc.txt"
    wout="$W/${b}_wat.txt"

    gcc_per_function "$src" > "$gout" 2>/dev/null
    wat_per_function "$src" > "$wout" 2>/dev/null

    merge_and_report "$gout" "$wout" "$b"
done
