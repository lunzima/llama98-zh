#!/bin/bash
# ISA-floor gate, gcc AVX2 tier.
#
# WHAT IT ASSERTS. Every instruction in src/ops_avx2.c's object belongs to
# the AVX2 feature set - the set lz_cpu_has_avx2() probes (OSXSAVE, XCR0
# bits 1:2, CPUID leaf 7 EBX bit 5). Nothing in the object may need a
# CPUID bit that probe does not read.
#
# WHY IT IS NOT PEDANTRY. The tier is auto-selected now, so an instruction
# outside that set is not a slow path, it is #GP on a machine the selector
# called capable - and no numeric gate can see it, since all tiers are
# required to agree bit for bit anyway. -mavx2 is a licence, not a
# guarantee: the Makefile keeps the flag list at -mavx2 and deliberately
# no -mfma, and a later edit to that list, or a -mtune that reaches for a
# BMI shift, would put an instruction here that the probe does not cover.
#
# CLASSIFIED TWICE, by mnemonic family (each with the CPUID bit it would
# need instead) and by register operand: %zmm, %k0-7 and %ymm16-31 are
# EVEX-only, so no VEX encoding can name them. It also asserts the object
# CONTAINS AVX2, so a stale or empty one cannot pass.
#
# The floor for the NON-suffixed TUs is the training tree's
# tests/test_ops_isa_floor.py; this is the gcc-side twin of
# isa_floor_watcom_gate.sh, scoped to the one object deliberately above
# the baseline.
#
# Reads build/gui/ops_avx2.o (gui-selftest-gate builds it, and `make
# check` runs that first), falling back to build/native/. LZ_AVX2_OBJ
# points it at another object, which is how the planted mutation below is
# run: the same source built with a wider -march must redden this gate, or
# it is measuring nothing.
#
# Exit codes: 0 PASS, 1 FAIL, 2 ERROR (no object, no objdump, empty
# disassembly).
set -u
cd "$(dirname "$0")/.."

LZ_ENV=$(LZ_TREE_LOCK_NAME="$(basename "$0")" LZ_TREE_LOCK_WHAT="the AVX2 ISA floor" bash build/tree_lock.sh acquire --print-env) || exit 2
eval "$LZ_ENV"
trap 'bash build/tree_lock.sh release' EXIT

OBJ=${LZ_AVX2_OBJ:-}
if [ -z "$OBJ" ]; then
    for cand in build/gui/ops_avx2.o build/native/ops_avx2.o; do
        if [ -f "$cand" ]; then OBJ=$cand; break; fi
    done
fi
if [ -z "$OBJ" ]; then
    echo "isa_floor_avx2_gate: ERROR - no ops_avx2.o (run make gui, or make native)"
    exit 2
fi

OBJDUMP=${OBJDUMP:-objdump}
command -v "$OBJDUMP" >/dev/null 2>&1 || {
    echo "isa_floor_avx2_gate: ERROR - objdump not found"
    exit 2
}

DIS=$("$OBJDUMP" -d --no-show-raw-insn "$OBJ" 2>/dev/null)
if [ -z "$DIS" ]; then
    echo "isa_floor_avx2_gate: ERROR - objdump produced no output for $OBJ"
    exit 2
fi

# One mnemonic per line, taken from objdump's own instruction rows. The
# symbol lines ("0000 <lz_...>:") do not match: the address is followed
# by a space and a bracket, not by a colon.
MN=$(printf '%s\n' "$DIS" | sed -n 's/^[[:space:]]*[0-9a-f]*:[[:space:]]*\([a-z][a-z0-9.]*\).*/\1/p')

FAIL=0

# NOT IMPLIED BY AVX2, each with the CPUID bit it would need instead:
#   vfm*              FMA          leaf 1 ECX 12
#   andn/bextr/bls*/  BMI1         leaf 7 EBX 3
#   tzcnt             BMI1 (REP form)
#   bzhi/mulx/pdep/   BMI2         leaf 7 EBX 8
#   pext/rorx/sarx/shlx/shrx
#   popcnt            POPCNT       leaf 1 ECX 23
#   lzcnt             ABM          leaf 0x80000001 ECX 5
#   movbe             MOVBE        leaf 1 ECX 22
#   aes/vaes/pclmul/  AES/PCLMULQDQ/VNNI
#   sha1/sha256/gf2p8
#   vpgather/vpscatter AVX2 gather is in the set; the AVX-512 forms are not
#   vpternlog/vpermi2/vpermt2/vpcompress/vpexpand/vpconflict/vplzcnt/
#   vpopcnt/vrcp14/vrsqrt14/vpmovm2/vpmovdb/vpmovwb   AVX-512 subsets
#   adcx/adox         ADX          leaf 7 EBX 19
OUT=$(printf '%s\n' "$MN" | grep -E '^(vfm|popcnt|lzcnt|andn|bextr|blsi|blsmsk|blsr|bzhi|mulx|pdep|pext|rorx|sarx|shlx|shrx|adcx|adox|movbe|aes|vaes|pclmul|sha1|sha256|gf2p8|vpternlog|vpermi2|vpermt2|vpcompress|vpexpand|vpconflict|vplzcnt|vpopcnt|vrcp14|vrsqrt14|vpmovm2|vpmovdb|vpmovwb|kmov|kortest|ktest)' || true)
if [ -n "$OUT" ]; then
    echo "  FAIL: mnemonics outside the AVX2 feature set in $OBJ:"
    printf '%s\n' "$OUT" | sort | uniq -c | sed 's/^/    /'
    FAIL=1
fi

# EVEX-only register names. No VEX encoding - which is all AVX2 has - can
# name these, so their presence is AVX-512 whatever the mnemonic says.
EVEX=$(printf '%s\n' "$DIS" | grep -cE '%zmm[0-9]+|%k[0-7]|%ymm(1[6-9]|2[0-9]|3[01])')
if [ "$EVEX" -ne 0 ]; then
    echo "  FAIL: $EVEX AVX-512 register operand(s) in $OBJ"
    FAIL=1
fi

# ... AND THAT IT CARRIES AVX2 AT ALL.
YMM=$(printf '%s\n' "$DIS" | grep -cE '%ymm[0-9]+')
A2ONLY=$(printf '%s\n' "$MN" | grep -cE '^(vperm2i128|vpermq|vpermps|vpermd|vphaddd|vphaddw|vpbroadcastd|vpbroadcasti128|vpbroadcastq|vextracti128|vinserti128|vpmovzxbw|vpmulld|vpmaxsd|vpmaxud)' || true)
if [ "$YMM" -eq 0 ] || [ "$A2ONLY" -eq 0 ]; then
    echo "  FAIL: $OBJ carries no AVX2 (ymm operands $YMM, AVX2-only mnemonics $A2ONLY)"
    FAIL=1
fi

if [ "$FAIL" -ne 0 ]; then
    echo "isa_floor_avx2_gate: FAIL ($OBJ)"
    exit 1
fi
echo "isa_floor_avx2_gate: PASS ($OBJ: $YMM ymm operands, $A2ONLY AVX2-only mnemonics,"
echo "  0 outside the AVX2 feature set, 0 AVX-512 registers)"
exit 0
