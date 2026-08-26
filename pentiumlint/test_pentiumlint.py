#!/usr/bin/env python3
"""Unit tests for pentiumlint: fabricated wdis fixtures. The positive
control is the point - a rule that cannot see its own planted pattern
never fires."""

import pentiumlint


def _dis(body):
    return "\n".join("%04x  %s\t\t%s\t\t%s" % (i, b, m, o)
                     for i, (b, m, o) in enumerate(body)) + "\n"


def test_inc_fires_on_p4_rule():
    text = _dis([("42", "inc", "edx"),
                 ("42", "inc", "ebx"),
                 ("4a", "dec", "edx")])
    found = pentiumlint.lint(text)
    assert len(found) == 1, found
    cat, mn, procs, reason, n, fname = found[0]
    assert cat == 1 and mn == "inc+dec" and procs == "p4" and n == 3, found


def test_movsx_cost_delta_on_p5():
    text = _dis([("0f bf c0", "movsx", "eax,al"),
                 ("0f bf c9", "movsx", "ecx,cl")])
    found = pentiumlint.lint(text)
    assert any(f[1] == "movsx" and "p5" in f[2] for f in found), found


def test_imul_cost_delta_on_p5():
    text = _dis([("f7 e9", "imul", "ecx"),
                 ("f7 e8", "imul", "eax")])
    found = pentiumlint.lint(text)
    assert any(f[1].startswith("imul") and "p5" in f[2] for f in found), found


def test_p5_k6_outrank_cyrix_p4_in_sort():
    # A fixture hitting both the p5 rule (imul) and the p4 rule (jcc).
    # p5/k6 findings must come first in the sorted output.
    text = _dis([("f7 e9", "imul", "ecx"),
                 ("74 00", "jz", "label")])
    found = pentiumlint.lint(text)
    ranks = [pentiumlint._procs_rank(f[2]) for f in found]
    assert ranks == sorted(ranks), (found, ranks)


def test_add_sub_is_silent():
    text = _dis([("83 e9 01", "sub", "ecx,0x00000001"),
                 ("83 c0 01", "add", "eax,0x00000001")])
    found = pentiumlint.lint(text)
    assert found == [], found


def test_inc_rule_is_tagged_p4_not_p5():
    # The correction from gcc's x86-tune.def: inc/dec is a p4 rule, and
    # the rule table must not name p5 (that was the wrong first pass).
    rule = pentiumlint.RULES[0]
    assert rule.procs == "p4", rule.procs
    assert "p5" not in rule.procs, rule.procs


def _rule_fires(mnemonic, operands, procs_needle, n=3):
    """Plant `n` copies of one mnemonic and assert a finding whose procs
    contains `procs_needle` is present. Returns the matching finding."""
    text = _dis([("00", mnemonic, operands)] * n)
    found = pentiumlint.lint(text)
    for f in found:
        if f[1].split("+")[0] == mnemonic and procs_needle in f[2]:
            return f
    assert False, "%s should fire a %s rule: %s" % (mnemonic, procs_needle, found)


def test_idiv_is_p4_only():
    f = _rule_fires("idiv", "ecx", "p4")
    assert "p5" not in f[2], ("idiv must not name p5 (1.47x < 1.5x): %s" % (f,))


def test_not_fires_on_p5():
    f = _rule_fires("not", "eax", "p5")
    assert f[0] == 1, f


def test_cdq_fires_on_k6():
    # USE_CLTD excludes both m_PENT and m_K6, so k6 is in scope.
    f = _rule_fires("cdq", "", "k6")
    assert f[0] == 1, f


def test_stc_fires_on_p4():
    f = _rule_fires("stc", "", "p4")
    assert f[0] == 1, f


def test_sahf_fires_on_p5():
    f = _rule_fires("sahf", "", "p5")
    assert f[0] == 1, f


def test_shl_fires_on_p4():
    f = _rule_fires("shl", "eax,0x2", "p4")
    assert f[0] == 2, f


def test_lea_fires_on_p4_k6():
    f = _rule_fires("lea", "eax,[eax+eax*4]", "p4")
    assert "k6" in f[2], f


def test_movsx_numbers_are_insns_not_x4_units():
    # The movsx reason must say "3 insns" (COSTS_N_INSNS(3)), not the
    # internal x4 unit "12". This catches the unit bug the audit found.
    rule = [r for r in pentiumlint.RULES if r.mnems == ("movsx",)][0]
    assert "3 insns" in rule.reason, rule.reason
    assert "12 " not in rule.reason, rule.reason


def test_movzx_numbers_are_insns_not_x4_units():
    rule = [r for r in pentiumlint.RULES if r.mnems == ("movzx",)][0]
    assert "2 insns" in rule.reason, rule.reason
    assert "8 " not in rule.reason, rule.reason


if __name__ == "__main__":
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
