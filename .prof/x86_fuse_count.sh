#!/bin/bash
# Static instruction counts for .prof/x86_fuse_count.c's three shapes.
#
#   wsl -e bash -lc "bash /mnt/e/llama98-zh/.prof/x86_fuse_count.sh"
#
# Iron law three: static instruction counts transfer across the target
# family, wall clock does not. So this counts, it does not time.
#
# THREE TARGETS, because the family has three float units and they give
# different answers to this question:
#   i586 x87    the Socket 7 / Win98 floor (build/x87/build_engine.sh's
#               own flags: -m32 -march=i586 -mno-mmx -mno-sse -mno-sse2
#               -mfpmath=387)
#   p4  sse2    the later 32-bit machines, float in xmm
#   x86-64      the development host, for continuity with the census
#
# The count is instructions in the LOOP BODY only - from the loop's top
# label to its backward branch - so the prologue, which runs once, does
# not dilute a per-element number.
set -e
R=${R:-/mnt/e/llama98-zh}
O=${O:-/tmp/x86fuse}
CC=${CC:-gcc}
mkdir -p "$O"; rm -f "$O"/*.o "$O"/*.txt

count_body() {   # count_body <objfile>
    # The loop body is the span from a backward branch's TARGET to that
    # branch, inclusive. A branch is backward when its target address has
    # already been seen as an instruction address - string equality on
    # the address text, no arithmetic. strtonum() was the first attempt
    # and it is a gawk extension: under mawk every arm printed "no-loop",
    # which is the failure mode a length-of-output check would not have
    # caught either, since all three printed the same wrong thing.
    objdump -d --no-show-raw-insn "$1" | awk '
        /<fuse_loop>:/ { inf = 1; next }
        inf && /^[0-9a-f]+ </ { inf = 0 }
        inf && /^[ \t]*[0-9a-f]+:/ {
            n++
            a = $1; sub(/:$/, "", a)
            addr[n] = a; seen[a] = n
            if ($2 ~ /^j/) jmp[n] = $3
        }
        END {
            best = 0; bestt = ""
            for (i = 1; i <= n; i++)
                if (jmp[i] != "" && (jmp[i] in seen) && seen[jmp[i]] < i) {
                    best = i; bestt = jmp[i]
                }
            if (!best) { print "no-loop"; exit }
            print best - seen[bestt] + 1
        }'
}

run_target() {   # run_target <label> <flags...>
    lbl="$1"; shift
    printf '%-12s' "$lbl"
    for s in 0 1 2; do
        if ! $CC "$@" -O2 -std=c99 -DFUSE_SHAPE=$s -c \
                "$R/.prof/x86_fuse_count.c" -o "$O/s$s.o" 2>"$O/err.txt"; then
            printf ' %-12s' "SKIP"
            continue
        fi
        printf ' %-12s' "$(count_body "$O/s$s.o")"
    done
    echo
}

printf '%-12s %-12s %-12s %-12s\n' target shipping fused fused+vol
run_target "i586-x87" -m32 -march=i586 -mno-mmx -mno-sse -mno-sse2 -mfpmath=387
run_target "p4-sse2"  -m32 -march=pentium4 -msse2 -mfpmath=sse
run_target "x86-64"   -march=x86-64
exit 0
