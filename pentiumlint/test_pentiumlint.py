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


def _op_fires(name, body, procs_needle):
    """Plant a body and assert an operand-rule finding named `name` with
    procs containing `procs_needle` is present. Returns the finding."""
    found = pentiumlint.lint(_dis(body))
    for f in found:
        if f[1] == name and procs_needle in f[2]:
            return f
    assert False, "%s should fire a %s rule: %s" % (name, procs_needle, found)


def _op_silent(name, body):
    """Plant a body and assert no finding named `name` is present. This is
    the negative control: the predicate must see the operands, not just
    the mnemonic."""
    found = pentiumlint.lint(_dis(body))
    assert not any(f[1] == name for f in found), (name, found)


def test_xor_reg_reg_fires_on_k6():
    f = _op_fires("xor-r,r", [("31 c0", "xor", "eax,eax")], "k6")
    assert f[0] == 2, f


def test_xor_reg_reg_silent_on_distinct_regs():
    _op_silent("xor-r,r", [("31 d8", "xor", "eax,ebx")])


def test_rmw_mem_fires_on_p5():
    f = _op_fires("rmw-mem", [("01 01", "add", "[ecx],eax")], "p5")
    assert f[0] == 1, f


def test_rmw_mem_silent_on_register_dest():
    _op_silent("rmw-mem", [("01 c1", "add", "eax,ecx")])


def test_push_mem_fires_on_p5():
    f = _op_fires("push-mem", [("ff 30", "push", "[eax]")], "p5")
    assert f[0] == 1, f


def test_push_mem_silent_on_register():
    _op_silent("push-mem", [("50", "push", "eax")])


def test_not_mem_fires_on_k6():
    f = _op_fires("not-mem", [("f7 10", "not", "[eax]")], "k6")
    assert f[0] == 1, f


def test_not_mem_silent_on_register():
    # not eax still fires the bare NOT_UNPAIRABLE rule (p5), but NOT the
    # K6 NOT_VECTORMODE operand rule.
    _op_silent("not-mem", [("f7 d0", "not", "eax")])


def test_rol_ror_imm8_fires_on_p4():
    f = _op_fires("rol/ror-8", [("c1 c0 08", "rol", "ax,0x8")], "p4")
    assert f[0] == 1, f


def test_rol_ror_imm8_silent_on_32bit_or_non8():
    _op_silent("rol/ror-8", [("c1 c0 08", "rol", "eax,0x8"),
                             ("c1 c0 03", "rol", "ax,0x3")])


def test_mov_reg_minus_one_fires_on_p5():
    f = _op_fires("mov-r,-1", [("b8 ff ff ff ff", "mov", "eax,0xffffffff")],
                  "p5")
    assert f[0] == 1, f


def test_mov_reg_minus_one_silent_on_zero():
    _op_silent("mov-r,-1", [("b8 00 00 00 00", "mov", "eax,0x0")])


def test_var_shift_fires_on_p5():
    f = _op_fires("var-shift", [("d3 e0", "shl", "eax,cl")], "p5")
    assert f[0] == 2, f


def test_var_shift_silent_on_constant():
    _op_silent("var-shift", [("c1 e0 02", "shl", "eax,0x2")])


def test_shift_rule_covers_all_shifts_and_rotates():
    # The P4 shift cost (shift_var/shift_const=4) covers the whole
    # shift/rotate family; a rule listing only shl/shr/sar would miss the
    # rest (the audit caught this). Pin that all 8 fire.
    for m in ("shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr"):
        text = _dis([("00", m, "eax,0x2")])
        found = pentiumlint.lint(text)
        assert any(f[1].startswith("shl+") and "p4" in f[2] for f in found), \
            (m, found)


def test_string_ops_is_cat1_single_stringop():
    # The movs/stos rule is a category-1 quirk (X86_TUNE_SINGLE_STRINGOP),
    # not a cost-delta, and it must name k6 too.
    rule = [r for r in pentiumlint.RULES if r.mnems == ("movs", "stos")][0]
    assert rule.category == 1, rule
    assert "SINGLE_STRINGOP" in rule.reason, rule.reason
    assert "k6" in rule.procs, rule.procs


def test_integration_on_real_gcc_intel_fixture():
    # A real gcc `objdump -M intel` fixture (not a synthetic _dis). This
    # exercises _GCC_INST_RE and _GCC_FUNC_RE, which the synthetic tests
    # never touch. Two functions, each with a distinct imul count.
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "fixtures", "forward_chunk_gcc.dis")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    found = pentiumlint.lint(text)

    # Per-function splitting: _forward_chunk has 1 imul, 1 movzx, 1 lea,
    # 1 cdq, 1 idiv; _lz_state_reset has 2 imul and nothing else.
    by_func = {}
    for cat, mn, procs, reason, n, fname in found:
        by_func.setdefault(fname, []).append((mn, n, procs))

    assert "_forward_chunk" in by_func, found
    assert "_lz_state_reset" in by_func, found
    chunk = dict((mn, n) for mn, n, _ in by_func["_forward_chunk"])
    assert chunk["imul+mul"] == 1, chunk
    assert chunk["movzx"] == 1, chunk
    assert chunk["lea"] == 1, chunk
    assert chunk["cdq+cltd"] == 1, chunk
    assert chunk["idiv"] == 1, chunk
    reset = dict((mn, n) for mn, n, _ in by_func["_lz_state_reset"])
    assert reset["imul+mul"] == 2, reset
    assert "movzx" not in reset, reset


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
