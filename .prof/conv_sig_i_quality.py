# -*- coding: utf-8 -*-
"""Paired quality: the conv epilogue's integer-coordinate sigmoid vs the
float one it replaces.

WHY NOT |logit diff|: it saturates (CLAUDE.md's own two measurements of
it, three orders of magnitude apart from the same arithmetic). The
criterion is downstream of the discrete decisions instead: per-position
NLL on real text, greedy argmax agreement, top-5 set agreement.

WHY PAIRED: both arms see the identical token sequence, so every
position is a matched pair and the between-position variance - orders of
magnitude larger than the effect - cancels. A marginal two-sample
comparison of the same numbers would answer a different question.
[[paired-not-marginal]]

TWO BINARIES, not one knob: LZ_CONV_SIG_I is a compile-time switch, so
the arms are two builds of the same tree.

    python3 .prof/conv_sig_i_quality.py <exeA> <exeB> <model> <tokfile>
"""
import os
import struct
import subprocess
import sys
import math


def run(exe, model, tokfile, tag, tmp):
    nll = os.path.join(tmp, tag + ".nll")
    t5 = os.path.join(tmp, tag + ".top5")
    cmd = [exe, model, "--tokens-file", tokfile,
           "--dump-nll", nll, "--dump-top5", t5]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.exit("run failed: %s\n%s" % (" ".join(cmd), r.stderr[-2000:]))
    with open(nll, "rb") as f:
        b = f.read()
    v = list(struct.unpack("<%df" % (len(b) // 4), b))
    with open(t5, "rb") as f:
        b = f.read()
    t = list(struct.unpack("<%di" % (len(b) // 4), b))
    top = [t[i * 5:(i + 1) * 5] for i in range(len(t) // 5)]
    return v, top


def main():
    exe_a, exe_b, model, tokfile = sys.argv[1:5]
    tmp = os.environ.get("TMPDIR", "/tmp")
    a_nll, a_top = run(exe_a, model, tokfile, "qa", tmp)
    b_nll, b_top = run(exe_b, model, tokfile, "qb", tmp)
    n = len(a_nll)
    if n != len(b_nll):
        sys.exit("length mismatch %d vs %d" % (n, len(b_nll)))

    d = [b_nll[i] - a_nll[i] for i in range(n)]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n else 0.0
    t = mean / se if se > 0 else 0.0
    # two-sided p from the normal approximation; n is 600+, so t and z
    # agree to three decimals and the difference cannot change a call.
    p = math.erfc(abs(t) / math.sqrt(2.0))
    nz = sum(1 for x in d if x != 0.0)
    worse = sum(1 for x in d if x > 0.0)
    better = sum(1 for x in d if x < 0.0)

    agree1 = sum(1 for i in range(n) if a_top[i][0] == b_top[i][0])
    agree5 = sum(1 for i in range(n) if set(a_top[i]) == set(b_top[i]))

    print("positions          %d" % n)
    print("mean NLL delta     %+.6e   (B - A; negative = B better)" % mean)
    print("sd of delta        %.6e" % sd)
    print("paired t           %+.3f    p = %.4f" % (t, p))
    print("mean NLL A / B     %.6f / %.6f" % (sum(a_nll) / n, sum(b_nll) / n))
    print("positions moved    %d  (%d worse, %d better)" % (nz, worse, better))
    print("argmax agreement   %d/%d  (%.4f%%)" % (agree1, n, 100.0 * agree1 / n))
    print("top-5 set agree    %d/%d  (%.4f%%)" % (agree5, n, 100.0 * agree5 / n))


main()
