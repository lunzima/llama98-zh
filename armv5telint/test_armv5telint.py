#!/usr/bin/env python3
"""Unit tests for armv5telint: fabricated objdump fixtures, one per
assertion. The positive control is the point - a rule that cannot see
its own planted pattern is a rule that never fires (see the mutation
note in the repo's gate conventions)."""

import armv5telint


def _func(name, body):
    return "00000000 <%s>:\n%s\n" % (name, body)


def test_ldrh_run_fires_on_eight():
    # The C tier lz_dot32_q16_arm: 8 consecutive ldrh (could be 2 ldm).
    text = _func("lz_dot32_q16_arm",
                 "\n".join("  %x:\te1d3a0b0 \tldrh\tsl, [r3]" % i
                           for i in range(0, 0x20, 4)))
    found = armv5telint.lint(text)
    assert ("ldrh-run 8", "lz_dot32_q16_arm") in found, found


def test_ldrh_run_silent_below_four():
    text = _func("f", "\n".join("  %x:\te1d3a0b0 \tldrh\tr0, [r1]" % i
                                for i in (0, 4, 8)))
    found = armv5telint.lint(text)
    assert not any(f[0].startswith("ldrh-run") for f in found), found


def test_smlabb_alone_fires():
    text = _func("lz_dot32_q8_arm",
                 "   0:\te100088a \tsmlabb\tr0, r4, r8, r0\n"
                 "   4:\te1000985 \tsmlabb\tr0, r5, r9, r0")
    found = armv5telint.lint(text)
    assert ("smlabb-alone 2", "lz_dot32_q8_arm") in found, found


def test_mul_family_fires_with_gcc_ratios():
    # One planted mul, mla, smull, umull, smlal, umlal - each must be
    # flagged with the ratio read from aarch-cost-tables.h (2x/3x).
    body = ("   0:\te0000092 \tmul\tr0, r2, r0\n"
            "   4:\te0203192 \tmla\tr3, r2, r1, r3\n"
            "   8:\te0c20390 \tsmull\tr0, r2, r0, r3\n"
            "   c:\te0e21390 \tumull\tr1, r2, r3, r0\n"
            "  10:\te0e02391 \tsmlal\tr0, r2, r1, r3\n"
            "  14:\te0e0a391 \tumlal\tr0, r2, r1, r3")
    found = armv5telint.lint(_func("recur_step", body))
    got = {f[0].split()[1]: f[0] for f in found if f[0].startswith("mul-family")}
    assert got["mul"] == "mul-family mul 1 (2x vs armv7-a)", got
    assert got["mla"] == "mul-family mla 1 (3x vs armv7-a)", got
    assert got["smull"] == "mul-family smull 1 (2x vs armv7-a)", got
    assert got["umull"] == "mul-family umull 1 (2x vs armv7-a)", got
    assert got["smlal"] == "mul-family smlal 1 (3x vs armv7-a)", got
    assert got["umlal"] == "mul-family umlal 1 (3x vs armv7-a)", got


def test_muls_is_silent_equal_cost():
    # muls is 1 vs 1 in gcc's table (flag_setting), so it must NOT be
    # flagged - the rule's positive control for "equal cost stays silent".
    text = _func("f", "   0:\te010f092 \tmuls\tr0, r2, r0")
    found = armv5telint.lint(text)
    assert not any(f[0].startswith("mul-family") for f in found), found


def test_ldst_cost_fires():
    # Single ldr/str are 2x, ldrd is 3x (aarch-cost-tables.h ldst table).
    body = ("   0:\te5903000 \tldr\tr3, [r0]\n"
            "   4:\te5803000 \tstr\tr3, [r0]\n"
            "   8:\te1c030b0 \tldrd\tr3, [r0]")
    found = armv5telint.lint(_func("f", body))
    singles = [f for f in found if f[0].startswith("ldst-single")]
    doubles = [f for f in found if f[0].startswith("ldst-double")]
    assert singles and "2" in singles[0][0], found
    assert doubles and "3" in doubles[0][0], found


def test_ldrsh_typod_mnemonic_is_covered():
    # The LDST_2X tuple had 'ldrsch' (typo) which silently never matched.
    # The real mnemonic is ldrsh. Pin it so the typo cannot regress.
    body = "   0:\te1d030f0 \tldrsh\tr3, [r0]"
    found = armv5telint.lint(_func("f", body))
    assert any(f[0].startswith("ldst-single") for f in found), found


def test_optimal_asm_is_silent():
    # The hand-written cell: ldm + paired smlabb/smlatt. No finding.
    text = _func("lz_dot32_q16_arm_asm",
                 "   0:\te8b300f0 \tldm\tr3!, {r4, r5, r6, r7}\n"
                 "   4:\te1000884 \tsmlabb\tr0, r4, r8, r0\n"
                 "   8:\te10cc8e4 \tsmlatt\tip, r4, r8, ip")
    found = armv5telint.lint(text)
    assert found == [], found


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                failures += 1
                print("FAIL", name, "-", e)
    print("%d failure(s)" % failures)
    raise SystemExit(1 if failures else 0)
