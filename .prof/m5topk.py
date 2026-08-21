"""Top-k flip rate between two routing dumps (int-pipeline milestone 5).

READS ONLY. Each dump is the concatenation of every lz_moe_route call's
idx_out array, in call order, written by the LZ_M5COUNT build's
lz_m5_topk (LZ_M5_TOPK=<path>). Two builds that differ only in whether
the router logits took the int16 exit produce dumps of the same length,
so a positional diff is a decision-by-decision comparison.

Reported three ways, because they answer different questions:
  set    - the two decisions chose the same SET of experts
  exact  - the same set in the same RANK order
  slot   - per (decision, slot) agreement, the finest of the three

usage: m5topk.py <a.bin> <b.bin> <top_k>
"""
import sys, array


def load(path):
    a = array.array('i')
    with open(path, 'rb') as f:
        a.frombytes(f.read())
    return a


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    a, b, k = load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3])
    if len(a) != len(b):
        print("length mismatch: %d vs %d ints" % (len(a), len(b)))
        return 1
    n = len(a) // k
    set_bad = exact_bad = slot_bad = 0
    for i in range(n):
        xa = a[i * k:(i + 1) * k]
        xb = b[i * k:(i + 1) * k]
        if list(xa) != list(xb):
            exact_bad += 1
        if set(xa) != set(xb):
            set_bad += 1
        slot_bad += sum(1 for u, v in zip(xa, xb) if u != v)
    print("decisions %d (top_k=%d)" % (n, k))
    print("  set differs   %7d  %.4f%%" % (set_bad, 100.0 * set_bad / n))
    print("  order differs %7d  %.4f%%" % (exact_bad, 100.0 * exact_bad / n))
    print("  slots differ  %7d  %.4f%%" % (slot_bad, 100.0 * slot_bad / (n * k)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
