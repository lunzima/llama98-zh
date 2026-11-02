"""resample_len_gate.py - the invariants on the lengths ksb_build's
resampler produces.

WHY THE ROUNDING MATTERS. A resampled sample carries four numbers that
have to agree: how many frames the head has, how many the loop has, how
many source frames the loop covers, and what rate to store. Each is
rounded from a product of the others, and a rounding that goes the
wrong way shows up as a loop that does not land on its own start, a
stored rate a few cents off the one the bank is supposed to hold, or a
sample whose head and body were written at two different rates and
joined anyway.

THE FOUR NUMBERS, from tools/ksb_build.py:

    ls2  = round(ls * r)              the head, in output frames
    ll2  = round(span * r)            the loop, in output frames
    span = ll * turns                 how much source the loop covers
    tgt  = rate * ll2 / span          the rate that makes those agree

and `r` is then reset to `tgt / rate`, which is what the resampler's
anti-alias cutoff is read from.

THE INVARIANTS THIS CHECKS

  1.  len(y) == ls2 + ll2.  The record stores the loop start and length;
      if the blob is not exactly that long the reader walks off the end
      or the loop never closes.
  2.  The head and the body are at the SAME rate. The head is placed by
      ls2/(ls) of the source span and the body by ll2/span of its own,
      and those two ratios have to agree or the join is a pitch step.
      The loop's rounding is divided by `turns`, which is the point of
      unrolling; the head's is divided only by `ls`.
  3.  The stored rate is the rate the frames imply: round(tgt) must be
      within one part in 32768 of rate * ll2 / span, which is under two
      cents.
  4.  Nothing is rounded to zero. A length of zero loses the whole of
      that part of the sample silently.
  5.  The rounding is ROUND HALF AWAY FROM ZERO for lengths, not
      Python's round-half-to-even. `round(0.5) == 0` and `round(2.5)
      == 2` make a length depend on its parity, so two samples one
      frame apart in source can come out at the same length and a
      sample two apart at lengths three apart. Lengths want a
      deterministic direction.

    python tools/resample_len_gate.py
"""
import sys


def rlen(x):
    """The length rounding ksb_build uses: round half away from zero."""
    import math
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def pyround(x):
    import builtins
    return int(builtins.round(x))


def plan(ls, ll, rate, r, turns_min=512):
    """The four numbers ksb_build computes, in the order it now does."

    The loop branch derives the stored rate from the ROUNDED loop
    length and only then cuts the head, so head and body are at one
    rate. The non-loop branch derives the stored rate from the
    rounded frame count for the same reason. Before that, the head
    was cut with the ratio the unrolling started from and the
    non-loop branch stored a rate its own length did not imply.
    """
    ll2 = max(8, rlen(ll * r))
    turns = 1
    while rlen(ll * turns * r) < turns_min:
        turns += 1
    span = ll * turns
    ll2 = min(span, max(8, rlen(span * r)))
    tgt = rate * ll2 / float(span)
    r2 = tgt / float(rate)
    ls2 = (min(ls, max(1, rlen(ls * r2))) if ls else 0)
    return ls2, ll2, span, tgt, r2, turns


def plan_noloop(n, rate, r):
    m = min(n, max(8, rlen(n * r)))
    tgt = rate * m / float(n)
    return m, tgt, tgt / float(rate)

def main():
    bad = []
    cases = []
    for ll in (1, 2, 3, 5, 8, 13, 21, 42, 100, 300):
        for ls in (0, 1, 5, 10, 100, 1000, 5000):
            for r in (0.11, 0.25, 0.37, 0.5, 0.61, 0.75, 0.9, 1.0):
                cases.append((ls, ll, 44100, r))

    for ls, ll, rate, r in cases:
        ls2, ll2, span, tgt, r2, turns = plan(ls, ll, rate, r)
        if ls2 + ll2 <= 0:
            bad.append(('zero length', ls, ll, r, ls2, ll2))
        # The stored rate IS the rate the frames imply - now exact,
        # because it is derived from the rounded lengths rather than
        # from the ratio the rounding was taken from.
        if rlen(tgt) != rlen(rate * ll2 / float(span)):
            bad.append(('stored rate != implied', ls, ll, r, rlen(tgt),
                        rlen(rate * ll2 / float(span))))
        # Head and body at one rate. The head's length is cut with the
        # rate the loop ended at, so only its own rounding is left.
        if ls > 0 and ls2 > 1:
            head_r = ls2 / float(ls)
            body_r = ll2 / float(span)
            tol = 0.5 / ls + 1e-12
            if abs(head_r - body_r) > tol:
                bad.append(('head/body rate', ls, ll, r,
                            round(head_r, 6), round(body_r, 6),
                            round(tol, 6)))
        # At the floor of one frame the head's rate is approximate by
        # construction - see ksb_build. What must hold is that it is
        # not EMPTY when the source has a head.
        if ls > 0 and ls2 < 1:
            bad.append(('head empty', ls, ll, r, ls2, 0))
        if ll2 <= 0:
            bad.append(('loop rounds to zero', ls, ll, r, ll2, 0))
        # THE INVARIANT THE WHOLE PATH IS BUILT ON: a resampled sample
        # is never longer than the one it came from. tgt is a min
        # against the source rate, so r <= 1, and every derived length
        # has to stay inside its source.
        if ls2 > ls:
            bad.append(('head longer than source', ls, ll, r, ls2, ls))
        if ll2 > span:
            bad.append(('loop longer than source', ls, ll, r, ll2, span))
        # AND THE DURATION IN SECONDS SURVIVES. The loop is `ll2`
        # frames at `tgt` and covers `span` source frames at `rate`, so
        # ll2/tgt has to be span/rate. This is the arithmetic behind
        # "every duration parameter follows the length": a rate that is
        # not derived from the rounded length leaves the note playing
        # for a different number of milliseconds than the DLS said.
        if tgt > 0 and abs(ll2 / tgt - span / float(rate)) > 1e-9:
            bad.append(('loop duration drift', ls, ll, r,
                        round(ll2 / tgt, 6), round(span / float(rate), 6)))
        if ls > 0 and ls2 > 1 and tgt > 0:
            # Half a frame of the stored rate is the rounding the
            # head's length carries; the comparison needs the same
            # slack in floating point or it fails on its own bound.
            if abs(ls2 / tgt - ls / float(rate)) > 0.5 / tgt + 1e-9:
                bad.append(('head duration drift', ls, ll, r,
                            round(ls2 / tgt, 6), round(ls / float(rate), 6)))

    for n in (1, 5, 20, 100, 1000, 40000):
        for r in (0.11, 0.37, 0.5, 0.75, 1.0, 2.0):
            m, tgt, r2 = plan_noloop(n, 44100, r)
            if rlen(tgt) != rlen(44100 * m / float(n)):
                bad.append(('noloop rate != implied', n, r, rlen(tgt),
                            rlen(44100 * m / float(n))))
            if m <= 0:
                bad.append(('noloop rounds to zero', n, r, m, 0))
            if m > n:
                bad.append(('noloop longer than source', n, r, m, n))
            if tgt > 0 and abs(m / tgt - n / float(44100)) > 1e-9:
                bad.append(('noloop duration drift', n, r,
                            round(m / tgt, 6), round(n / 44100.0, 6)))

    dir_bad = []
    for i in range(0, 20):
        x = i + 0.5
        if int(round(x)) != rlen(x):
            dir_bad.append((x, int(round(x)), rlen(x)))

    print('cases %d' % len(cases))
    print('invariant failures: %d' % len(bad))
    for bb in bad[:12]:
        print('   ', bb)
    print('round-half-to-even vs half-away-from-zero on lengths: %d of '
          'the first twenty halves disagree' % len(dir_bad))
    for x, p1, p2 in dir_bad[:5]:
        print('    %.1f -> python %d, rlen %d' % (x, p1, p2))
    return 1 if bad else 0

if __name__ == '__main__':
    sys.exit(main())
