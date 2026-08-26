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
    mn, procs, reason, n = found[0]
    assert mn == "inc+dec" and procs == "p4" and n == 3, found


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
