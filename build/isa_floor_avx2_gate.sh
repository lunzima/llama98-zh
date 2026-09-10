#!/bin/bash
# ISA-floor gate, gcc AVX2 tier.
#
# WHAT IT ASSERTS. Every instruction in src/ops_avx2.c's object belongs to
# the AVX2 feature set - the exact set lz_cpu_has_avx2() probes
# (OSXSAVE, XCR0 bits 1:2, CPUID leaf 7 EBX bit 5). Nothing in the object
# may need a CPUID bit that probe does not read.
#
# WHY IT IS NOT PEDANTRY. The tier is auto-selected now: kernel_detect
# returns LZ_KERNEL_AVX2 on any machine that passes that probe. An
# instruction outside the set is therefore not a slow path, it is #GP on a
# machine the selector decided was capable - and no numeric gate can see
# it, because every tier is required to agree bit for bit anyway.
#
# -mavx2 IS A LICENCE, NOT A GUARANTEE. It lets gcc use the AVX2
# instruction set; it does not promise it will stop there. The Makefile
# keeps the flag list at -mavx2 and deliberately no -mfma (see the
# ENG_AVX2 rule's own comment), and a later edit to that list - or a gcc
# that starts contracting, or -mtune work that reaches for a BMI shift -
# would put an instruction in this object that the probe above does not
# cover. That is the failure this gate exists to catch, and it catches it
# statically, before any of it runs here.
#
# CLASSIFIED TWICE, by mnemonic family and by register operand, because
# the two catch different things: FMA/BMI/POPCNT/LZCNT are new mnemonics
# over the same registers, while AVX-512 shows up first as %zmm, %k0-7 or
# %ymm16-31, which no AVX2 encoding can name.
#
# Also asserts the object actually CONTAINS AVX2: a census that passes
# because the compiler emitted nothing, or because the object is stale or
# misnamed, would be the always-green probe this project keeps
# re-learning about.
#
# The gcc floor for the NON-suffixed TUs is the training tree's
# tests/test_ops_isa_floor.py (cited by isa_floor_watcom_gate.sh); this
# file covers the one object that is deliberately above the baseline, and
# is the gcc-side twin of that gate's Watcom check.
#
# Reads build/gui/ops_avx2.o: `make check` runs gui-selftest-gate first,
# which builds it. build/native/ is the fallback for a tree that ran
# `make native` only. LZ_AVX2_OBJ points it at another object, which is
# how the planted mutation below is run: the same source built with a
# wider -march must redden this gate, or it is not measuring anything.
#
# Exit codes: 0 PASS, 1 FAIL, 2 ERROR (no object, no objdump, empty
# disassembly).
set -u
cd "$(dirname "$0")/.."

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
