"""Paired quality between two builds (int-pipeline milestone 5).

READS ONLY - it consumes --dump-nll / --dump-top5 files that were
produced by two already-built binaries. |logits diff| is deliberately
NOT reported: it saturates, which is why the tree's criterion is paired
NLL plus argmax and top-5 agreement (docs/int-pipeline-project.md
section 6).

The t is on the PAIRED per-position differences (after - before), which
is the whole point - the two arms see the same positions of the same
corpus, so the position-to-position variance that dwarfs the effect
cancels. A marginal (unpaired) comparison of the two means answers a
different and much weaker question.

usage: m5quality.py <after.nll> <before.nll> <after.top5> <before.top5>
"""
import sys
import array
import math


def f32(path):
    a = array.array('f')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    return a


def i32(path):
    a = array.array('i')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    return a


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    a, b = f32(sys.argv[1]), f32(sys.argv[2])
    if len(a) != len(b):
        print("nll length mismatch %d vs %d" % (len(a), len(b)))
        return 1
    n = len(a)
    d = [float(x) - float(y) for x, y in zip(a, b)]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if var > 0 else 0.0
    t = mean / se if se > 0 else 0.0
    print("paired NLL (after - before), n=%d" % n)
    print("  mean %+.6f nats   t = %+.3f" % (mean, t))
    print("  after mean %.6f   before mean %.6f" % (sum(a) / n, sum(b) / n))

    ta, tb = i32(sys.argv[3]), i32(sys.argv[4])
    if len(ta) != len(tb):
        print("top5 length mismatch %d vs %d" % (len(ta), len(tb)))
        return 1
    m = len(ta) // 5
    arg = sum(1 for i in range(m) if ta[i * 5] == tb[i * 5])
    setagree = sum(1 for i in range(m)
                   if set(ta[i * 5:i * 5 + 5]) == set(tb[i * 5:i * 5 + 5]))
    print("argmax agrees  %d/%d  %.1f%%" % (arg, m, 100.0 * arg / m))
    print("top-5 set      %d/%d  %.1f%%" % (setagree, m, 100.0 * setagree / m))
    return 0


if __name__ == '__main__':
    sys.exit(main())
