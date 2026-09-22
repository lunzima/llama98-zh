#!/bin/bash
# The language floor, as a gate instead of a memory.
#
# The target family includes Windows NT on MIPS built with Visual C++
# 4.0 - a 1995 C89 compiler. gcc and Watcom both accept far more than
# that, so nothing in the normal build says a word when C99 creeps in.
# This runs gcc -std=c89 -pedantic over src/ and counts the three
# constructs that are both easy to add by accident and impossible for
# VC++ 4.0:
#
#   mixed declarations and code   - a declaration after a statement
#   //                            - already 0 in this tree, kept at 0
#   designated initialisers       - [IDX] = val, C99 only
#
# NOT `long long` and NOT <stdint.h>. Those were 213 and 23 hits of
# known, tracked debt, and the plan for them was one header per
# toolchain rather than a per-site fix - src/lz_int.h is that header. A
# gate that fails on the whole existing surface fails every run and
# gets switched off; this one has to redden only on NEW breakage, so it
# is a RATCHET: a baseline count per file, and any increase is a
# failure.
#
# The baseline is the state at the commit that added this gate, and it
# is written here rather than regenerated, because a baseline the gate
# recomputes is a baseline that ratchets the wrong way.
#
# Exit codes:
#   0  PASS - no file exceeds its baseline
#   1  FAIL - a file gained a C89 violation
#   2  SKIP - no gcc
set -u
cd "$(dirname "$0")/.."

command -v gcc >/dev/null 2>&1 || { echo "c89_floor_gate: SKIP - no gcc"; exit 2; }

# file:count. Only files with a non-zero baseline are listed; everything
# else must stay at zero.
#
#   ops.c:1          `volatile float tamf` in lz_quantize_q8_int64
#                    (1718f11). The volatile is load-bearing - it stops
#                    Watcom fusing the int64 convert into the multiply,
#                    which costs 1 ULP - so moving it is not free.
#   forward.c:16     48327cc, 75c8adc and neighbours, in forward_attn's
#                    per-head blocks.
#
# matmul_q41_impl carries the same property and needs no baseline entry:
# `int i, t;` sits in the declaration block above the kernel-select call,
# so the call is the function's first statement. Safe because nothing in
# those initialisers reads the tier: q41_fixed_epi asks lz_epi_mode and
# lz_epi_ready, and lz_row_pick - which does read it - still runs
# after. Verified byte-identical on both fixtures rather than argued.
#
# The rest predate the language floor. They are recorded, not excused:
# this gate stops the number growing, and each is a one-line reorder
# for whoever next edits that function. The two violations THIS
# gate was written after finding - both in ops_gdn.c, both introduced
# after bc4f1ba landed mid-session - are fixed rather than baselined,
# which is the difference between debt and a new hole.
# DESIGNATED INITIALISERS are the third construct, and gcc reports them
# as a WARNING, so an errors-only check walks straight past them. They
# were 266 in err.c and 186 in gui/localized_strings.c - the two message
# tables, where the form exists so that a code and its string cannot
# drift apart.
#
# Both are converted. The objection to converting was that positional
# initialisers put the strings in an order nothing checks; that is now
# build/str_table_order_gate.sh, which compares each row's tag comment
# against its enum and is strictly stronger than the designated form -
# the designated form made order irrelevant, and so let one code sit two
# slots out of place in err.c for as long as it did.
#
#   Empty. It was 16 in forward.c and is 0 - Visual C++ 4.0 turns each
#   of these into an error rather than a warning, so they stopped being
#   tolerable debt the day the compiler arrived. The control below is a
#   planted probe now, for the reason its own comment gave in advance.
BASELINE=""

# ALL THREE DIRECTORIES. The front end and the shared front-end core
# are shipped products on the same compilers, and scanning one gui file
# out of fourteen was not a decision - all fourteen pass -fsyntax-only
# here, checked, and doing so found eight violations nothing was looking
# at. common/ was unscanned and clean; a directory that is clean and
# unwatched is one commit from not being.
CFILES="src/*.c gui/*.c common/*.c"

# THE INCLUDE PATH IS PER FILE, and matches what the Makefile passes.
# Widening it for everyone is not free: cli_main.c is built with
# -Icommon, and scanning it with -Isrc alone resolved a header
# differently and hid a real mixed declaration. Scanning it with the
# widened set that gui/localized_strings.c needs would be the same
# mistake pointed the other way - a gate whose compile line is not the
# build's compile line is answering about a program nobody ships.
inc_for() {
    case "$1" in
        src/cli_main.c)             echo "-Isrc -Icommon" ;;
        gui/*)                      echo "-Igui -Isrc -Icommon" ;;
        *)                          echo "-Isrc" ;;
    esac
}

# AND THE DEFINES, for the same reason and by the same rule. CFLAGS gives
# LZ_MMX_TU/LZ_SSE2_TU to every gcc TU and the three SIMD units get
# -mmmx -msse -msse2 on top. The warning scan below passed neither, so it
# read the #else arm of every LZ_HAVE_P2_* block - a different program
# from the one that ships, and one mixed declaration in ops_gdn.c lived
# in the gap between the two scans for as long as they disagreed.
# -D__arm__ FOR THE ARM-GUARDED FILES, or this gate reads 82 lines of a
# 1291-line file and calls it checked.
#
# src/lz_softfp.c wraps everything after its macro block in
# `#if defined(__arm__) && defined(__GNUC__) && ...`, opened at line 155
# and not closed until the last line, with no #else. On this host
# __arm__ is undefined, so the preprocessor threw the whole body away
# and the gate was parsing the typedefs lz_int.h contributes and nothing
# else - about 7% of the file, none of it code. Measured by counting
# preprocessor output both ways: 82 non-blank lines without the define,
# 1158 with.
#
# It was not clean under there. Opening the guard turned up three
# `static inline` (C99; the GNU spelling that works under -std=c89 is
# __inline__, which ops_arm_prim.h already uses throughout) - real C89
# errors that this gate is supposed to catch and could not see. The
# same blind spot hid two VC++ 4.0 violations found by hand the same
# day, an `LL` suffix and an `unsigned long long` parameter.
#
# gcc defines __GNUC__ itself, so -D__arm__ alone opens the guard. The
# inline asm inside is never CODEGEN'd here - -fsyntax-only stops before
# the assembler - so an x86 host can check ARM sources this way; only
# the parse has to succeed.
# FOUR FILES, not one. Found by preprocessing every src/*.c both ways and
# comparing the non-blank line counts - the ratio says at once how much
# of a file this gate was reading:
#
#     lz_softfp.c      82 -> 1158    93% was invisible
#     ops_arm.c       143 ->  560    74%
#     lz_bf16.c       171 ->  291    41%
#     ops_t2_arm.c     96 ->  125    23%
#
# Every other file gains only the few lines ops_arm_prim.h contributes
# through the same define, which is ambient rather than its own body
# being gated. The three besides lz_softfp.c were already clean when the
# guard was opened; they are listed anyway, because "clean today" and
# "checked from now on" are different properties and this gate exists to
# supply the second one.
arm_guarded() {
    case "$1" in
        src/lz_softfp.c|src/ops_arm.c|src/lz_bf16.c|src/ops_t2_arm.c)
            return 0 ;;
        *)  return 1 ;;
    esac
}

defs_for() {
    D="-DLZ_MMX_TU=1 -DLZ_SSE2_TU=1"
    arm_guarded "$1" && D="$D -D__arm__"
    case "$(basename "$1")" in
        ops_mmx.c|ops_mmx_sse.c|ops_sse2.c)
            echo "$D -mmmx -msse -msse2" ;;
        *)  echo "$D" ;;
    esac
}

# THE SWITCHES THAT ARE OFF BY DEFAULT. A scan of one configuration says
# nothing about the code the configuration excludes, and the excluded
# code is the code nobody reads. LZ_KV_2PLANE=1 alone reveals a mixed
# declaration in forward.c's low-plane weighted sum that every scan
# before this one called clean.
#
# One switch at a time, not the product: the point is to REACH the
# guarded text, and each of these guards a disjoint region. The
# combinations are switch_matrix_gate's job.
SWITCH_CFGS="
-DLZ_KV_2PLANE=1
-DLZ_GDN_STATE_2PLANE=0
-DLZ_ATTN_FIXED=0
-DLZ_ATTN_FIXED=1
-DLZ_ATTN_FIXED=2
-DLZ_GDN_FIXED=0
-DLZ_CONV_FIXED=0
-DLZ_NORM_FIXED=0
-DLZ_EPI_FIXED=0
"

PATTERN="mixed declarations|ISO C90 forbids .*//|forbids specifying subobject"

# One place that knows the compile line, so the passes below cannot drift
# apart the way the warning and error scans did.
scan_warn() {
    gcc -std=c89 -pedantic $(inc_for "$1") $(defs_for "$1") ${2:-} \
        -fsyntax-only "$1" 2>&1 | grep -E "$PATTERN"
}

FAIL=0
TOTAL=0
for f in $CFILES; do
    b=$(basename "$f")
    exp=0
    for e in $BASELINE; do
        [ "${e%%:*}" = "$b" ] && exp=${e##*:}
    done
    n=$(scan_warn "$f" | grep -c .)
    TOTAL=$((TOTAL + n))
    if [ "$n" -gt "$exp" ]; then
        echo "FAIL: $b has $n C89 violation(s), baseline $exp"
        scan_warn "$f" | head -5 | sed 's/^/      /'
        FAIL=1
    elif [ "$n" -lt "$exp" ]; then
        echo "note: $b is at $n, below its baseline of $exp - tighten BASELINE"
    fi
done

# The same scan through each off-by-default switch. Reported per
# configuration rather than summed into TOTAL: a violation that only
# exists under -DLZ_KV_2PLANE=1 is not a violation of the shipping
# build, and rolling it into the headline count would make the headline
# unfixable by anyone who builds the default.
SWFAIL=0
for cfg in $SWITCH_CFGS; do
    for f in $CFILES; do
        d=$(scan_warn "$f" "$cfg")
        [ -n "$d" ] || continue
        # Only what the DEFAULT configuration does not already report -
        # otherwise every baselined violation is re-reported nine times.
        base=$(scan_warn "$f" | grep -c .)
        n=$(printf '%s\n' "$d" | grep -c .)
        if [ "$n" -gt "$base" ]; then
            echo "FAIL: $cfg reveals $((n - base)) C89 violation(s) in $(basename "$f")"
            printf '%s\n' "$d" | head -3 | sed 's/^/      /'
            SWFAIL=$((SWFAIL + 1))
        fi
    done
done
[ "$SWFAIL" -eq 0 ] || FAIL=1
[ "$SWFAIL" -eq 0 ] && echo "c89 switches: 0 violations behind $(echo $SWITCH_CFGS | wc -w) off-by-default switch(es)"

# The gate's own positive control: if -pedantic stopped reporting the
# construct at all, every count would be 0 and this would pass while
# measuring nothing. ops.c's one known violation is the control.
# A count of zero over the whole tree would mean -pedantic stopped
# reporting the construct, not that the tree is clean: the baseline
# above names 23 known ones, so zero is the instrument failing. Compare
# against the baseline's own sum rather than against 0, so this keeps
# working as the baseline shrinks - and says so when it reaches zero,
# because then the control has to become something else.
EXPECT=0
for e in $BASELINE; do EXPECT=$((EXPECT + ${e##*:})); done
if [ "$EXPECT" -eq 0 ]; then
    # The baseline is empty, so there is no existing violation to prove
    # the scan on. Plant one instead - the control the old branch said
    # to write when this day came.
    # BOTH constructs, not just the mixed declaration. The scan counts
    # two patterns and a probe for one of them keeps passing while the
    # other goes undetected - and the designated initialiser is the one
    # that mattered here, 452 of this tree's violations against 23.
    PROBE=/tmp/c89ctl.$$.c
    printf 'void f(void) { int a = 1; a++; { int b = a; (void)b; } }\nvoid g(void) { int a = 1; a++; int b = a; (void)b; }\nstruct p { int x, y; };\nstruct p q = { .y = 2 };\n' > "$PROBE"
    OUT=$(gcc -std=c89 -pedantic -fsyntax-only "$PROBE" 2>&1)
    rm -f "$PROBE"
    NM=$(printf '%s\n' "$OUT" | grep -c "mixed declarations")
    ND=$(printf '%s\n' "$OUT" | grep -c "forbids specifying subobject")
    if [ "${NM:-0}" -eq 0 ] || [ "${ND:-0}" -eq 0 ]; then
        echo "FAIL: the planted probe reports mixed=$NM designated=$ND;"
        echo "      a zero in either means the zero above is the scan"
        echo "      being blind rather than the tree being clean"
        FAIL=1
    fi
elif [ "$TOTAL" -eq 0 ]; then
    echo "FAIL: total is 0 against a baseline of $EXPECT - the compiler is"
    echo "      not reporting mixed declarations and this gate is blind"
    FAIL=1
fi

# DOES IT COMPILE, which the counting above never asked.
#
# Everything before this point greps for two constructs and tolerates a
# baseline of them. That is a WARNING-class check and it is blind to the
# error class: `restrict` is C99, twenty-six declarations used it, every
# translation unit that saw one failed to PARSE, and this gate reported
# PASS across all forty errors. Counting patterns is not compiling.
#
# ERRORS ONLY, no baseline. A warning here is a style question with a
# recorded number above; an error means the floor is not met at all, and
# there is no such thing as an acceptable number of those.
#
# WHAT THIS DOES NOT PROVE, and the distinction matters more here than
# in most gates: green means the code is standard C89. It does NOT mean
# Visual C++ 4.0 accepts it - that compiler's C89 is incomplete and has
# quirks of its own, and nothing in this tree can run it. Reading this
# gate as "the MIPS target builds" is the mistake it is written to
# prevent.
# THROUGH defs_for, not a second copy of the flags. This loop used to
# restate -DLZ_MMX_TU/-DLZ_SSE2_TU and the -mmmx/-msse case analysis
# itself, so the two lists could disagree - and they did the moment
# defs_for learned -D__arm__ and this one did not, which would have left
# the compile half of the gate still parsing 82 lines of lz_softfp.c
# while the pattern half checked the whole thing.
CFAIL=0
NC=0
for f in $CFILES; do
    NC=$((NC + 1))
    n=$(gcc -std=c89 -pedantic -fsyntax-only -Isrc -Icommon -Igui \
            $(defs_for "$f") "$f" 2>&1 \
        | grep -c " error: ")
    if [ "$n" -gt 0 ]; then
        echo "FAIL: $(basename "$f") - $n C89 compile error(s)"
        gcc -std=c89 -pedantic -fsyntax-only -Isrc -Icommon -Igui \
            $(defs_for "$f") "$f" 2>&1 \
            | grep -m2 " error: " | sed 's/^/      /'
        CFAIL=$((CFAIL + n))
        FAIL=1
    fi
done
# The file count is printed because it is the thing that silently
# shrinks: this loop scanned src/ alone while the warning loop above had
# already been widened, so the two were answering about different
# programs.
[ "$CFAIL" -eq 0 ] && echo "c89 compile: 0 errors across $NC file(s)"

if [ "$FAIL" -eq 0 ]; then
    echo "c89_floor_gate: PASS ($TOTAL known violation(s), none new)"
    exit 0
fi
echo "c89_floor_gate: FAIL"
exit 1
