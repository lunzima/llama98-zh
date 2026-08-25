#!/bin/bash
# One coverage build, one generation, one gcov pass - shared by every
# census that needs execution counts rather than static counts.
#
# WHY IT IS ITS OWN FILE. Two censuses need the same three steps and the
# steps have sharp edges: gcov resolves source paths against the current
# directory, names its notes file after the OUTPUT binary rather than
# the source, and writes .gcov into the current directory instead of
# next to the .gcda. Each of those cost a round of an empty table when
# it was first met. Copying the recipe would mean the next reader has to
# meet them again in the copy, and the two copies would drift on the
# thing that matters most: WHICH RUN the counts come from. A count is
# only comparable to another count from the same prompt, the same token
# budget and the same seed.
#
# Sourced, not executed:
#     . build/cov_run.sh
#     cov_run "$W" "$MODEL" 8 "$EXTRA_CFLAGS"   || exit 2
# leaves $W/*.gcov, and prints its own SKIP line before returning
# non-zero. The fourth argument is for measuring a configuration the
# shipping one compiles out: LZ_FCX bills nothing without
# -DLZ_F32COUNT=1, so its call sites carry no code and gcov reports "-"
# for every one of them, which reads as eight dead sites rather than as
# a build that cannot answer.
#
# THE BUILD IS -O2, NOT -O0. A coverage build at -O0 counts control flow
# the optimiser deletes. --coverage keeps its arcs through -O2, so the
# counts describe the optimised program - the one that ships.
#
# THE ENGINE ONLY, plus the two CLI files needed to link. The front
# end's option parsing runs once and would fill any table built on this
# with lines nobody executes twice.

# The SIMD translation units are linked as prebuilt objects and carry no
# coverage arcs. Nothing here reports on them; a caller that needs them
# has to say so itself.
cov_run() {
    cov_w=$1
    cov_model=$2
    cov_n=${3:-8}
    cov_extra=${4:-}

    command -v gcov >/dev/null 2>&1 || { echo "cov_run: SKIP - no gcov"; return 2; }
    command -v gcc  >/dev/null 2>&1 || { echo "cov_run: SKIP - no gcc";  return 2; }
    [ -d "$cov_model" ] || { echo "cov_run: SKIP - no model: $cov_model"; return 2; }

    cov_eng=$(make -s print-ENG) || { echo "cov_run: cannot read ENG"; return 2; }
    cov_com=$(make -s print-COM) || { echo "cov_run: cannot read COM"; return 2; }
    cov_src=""
    for cov_f in $cov_eng; do cov_src="$cov_src src/$cov_f.c"; done
    for cov_f in $cov_com; do cov_src="$cov_src common/$cov_f.c"; done

    gcc -O2 -march=x86-64 -std=c99 -Isrc -Icommon -ffp-contract=off \
        -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 $cov_extra --coverage \
        -o "$cov_w/cov.exe" $cov_src src/cli_attr.c src/cli_main.c \
        build/native/ops_mmx.o build/native/ops_mmx_sse.o \
        build/native/ops_sse2.o \
        -lm >"$cov_w/build.log" 2>&1 || {
        echo "cov_run: SKIP - coverage build failed"
        tail -3 "$cov_w/build.log"; return 2; }

    # Fixed seed, so two runs of a census on the same tree compare.
    ( cd "$cov_w" && ./cov.exe "$cov_model" --prompt "hi" -n "$cov_n" \
        --seed 1145141919 ) >"$cov_w/run.log" 2>&1 || {
        echo "cov_run: SKIP - the run failed"
        tail -3 "$cov_w/run.log"; return 2; }

    # From the repo root, -o at the .gcda, and hand gcov the .gcda
    # rather than the source names: gcc named the notes files after the
    # output binary because many sources compiled into one, so
    # `gcov src/ops.c` looks for a file that does not exist and reports
    # it into a log nobody reads.
    gcov -b -c -o "$cov_w" "$cov_w"/*.gcda >"$cov_w/gcov.log" 2>&1
    mv -f ./*.gcov "$cov_w/" 2>/dev/null

    set -- "$cov_w"/*.gcov
    [ -f "$1" ] || { echo "cov_run: SKIP - gcov produced nothing"
                     tail -3 "$cov_w/gcov.log"; return 2; }
    return 0
}
