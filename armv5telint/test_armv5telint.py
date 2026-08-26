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
