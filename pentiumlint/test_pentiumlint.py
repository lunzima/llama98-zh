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
    cat, mn, procs, reason, n = found[0]
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
