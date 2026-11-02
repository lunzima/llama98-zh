#!/usr/bin/env python3
"""Measure one program's key range against gm, for keys the corpus never plays.

WHY THIS EXISTS. corpus_report measures the keys the corpus plays, and
that is the right criterion for "does this bank match gm where music
goes". But it leaves a hole: a program's corpus may cover 24 to 78 keys
and the bank has 128. p92's keys 120-127 read -2.7 down to -12.6 dB
under gm and p75's keys 118-127 read +14.5 dB at velocity 32 and -5 dB
at velocity 110 - none of which any corpus-driven number could ever
have shown, because those (key, velocity) pairs are never played.

A finding with no instrument behind it cannot be re-checked, and this
one took a hand-written probe to find. This is that probe.

WHAT IT ANSWERS, AND WHAT IT DOES NOT. Per note it prints the level
against gm, then the tone and length ratios against BOTH references.

THE TONE COLUMNS ANSWER TWO DIFFERENT QUESTIONS AND ONLY ONE OF THEM IS
THE STANDARD. This bank aims at ff8 and is merely bounded by gm - p111
reads 1.42 against gm and 1.016 against ff8, p55 reads 3.14 against gm
and 0.997 against ff8 - so "duller than gm" on its own is not a defect
and acted on as one re-tunes a bank that is already right. The gm
columns are still printed because "not too unlike gm" is the bound.

Level alone moving is a gain problem and a gain fix is right; the tone
ratios moving as well means the SAMPLE is wrong there and boosting it
would only make a wrong sound louder.

Usage:
  python3 tools/keyprobe.py 92 110 128 --vel 100
  python3 tools/keyprobe.py 40 30 44 --bank build/tmp/picked_ff8x2.ksb
  python3 tools/keyprobe.py 75 118 128 --vel 32,110
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, 'build', 'tmp'))
import corpus_report as cr  # noqa: E402


def main(argv):
    skip = set()
    for i, a in enumerate(argv[1:]):
        if a == '--bank' and i + 2 <= len(argv) - 1:
            skip.add(i + 2)
    args = [a for i, a in enumerate(argv[1:])
            if not a.startswith('--') and i + 1 not in skip]
    vels = [100]
    av = argv[1:]
    # EVERY VALUE-TAKING OPTION ACCEPTS BOTH SPELLINGS. `--bank X` was
    # silently ignored - the parser only knew `--bank=X` - so two runs
    # "compared" the same bank twice and reported identical output, which
    # read as "the knob does nothing". The same mistake had already been
    # made once with --vel in this file; both are handled the same way
    # now: collect the value, then drop it from the positional list.
    for i, a in enumerate(av):
        if a.startswith('--bank='):
            cr.BANK = os.path.abspath(a[7:])
        elif a == '--bank' and i + 1 < len(av):
            cr.BANK = os.path.abspath(av[i + 1])
    for i, a in enumerate(av):
        if a.startswith('--vel='):
            vels = [int(x) for x in a[6:].replace(',', ' ').split() if x]
        elif a == '--vel':
            # EVERY ARG AFTER IT THAT READS AS A NUMBER, so both
            # `--vel 32,110` and `--vel 32 110` work. The first version
            # of this read a single argv entry and silently measured one
            # velocity while the output looked like a two-velocity sweep.
            take = []
            for b in av[i + 1:]:
                try:
                    take.extend(int(x) for x in b.replace(',', ' ').split())
                except ValueError:
                    break
            if take:
                vels = take
    if len(args) < 3:
        print(__doc__)
        return 2
    prog, lo, hi = (int(x) for x in args[:3])
    notes = [(k, v) for v in vels for k in range(lo, hi)]
    mid = os.path.join(HERE, 'build', 'tmp', 'keyprobe.mid')
    wb = os.path.join(HERE, 'build', 'tmp', 'keyprobe_b.wav')
    wr = os.path.join(HERE, 'build', 'tmp', 'keyprobe_r.wav')
    wf = os.path.join(HERE, 'build', 'tmp', 'keyprobe_f.wav')
    cr.write_test(mid, prog, notes, None)
    ok = (cr.render(cr.BANK, mid, wb) and cr.render(cr.REF, mid, wr))
    if os.path.exists(cr.FF8):
        ok = ok and cr.render(cr.FF8, mid, wf)
    if not ok:
        print('render failed')
        return 1
    dt = cr.note_dt(prog)
    wins = cr.note_windows(wr, len(notes), dt)
    mb = cr.measure(wb, len(notes), dt, wins)
    mr = cr.measure(wr, len(notes), dt, wins)
    mf = cr.measure(wf, len(notes), dt, wins) if os.path.exists(wf) else []
    print('p%d   bank %s' % (prog, os.path.basename(cr.BANK)))
    print('%-5s %5s %9s | %7s %7s | %7s %7s | %7s'
          % ('key', 'vel', 'level dB', 'cen/gm', 'cen/ff8', 'hf/gm', 'hf/ff8', 'dur/gm'))
    for i, (k, v) in enumerate(notes):
        x, y = mb[i], mr[i]
        if not x or not y:
            print('%-5d %5d   no window' % (k, v))
            continue
        z = mf[i] if i < len(mf) else None
        lv = (20.0 * math.log10(x['rms'] / y['rms'])
              if x['rms'] > 0 and y['rms'] > 0 else 0.0)
        hf = (x['hf'] / y['hf']
              if y['hf'] >= cr.HF_FLOOR and x['hf'] > 0 else float('nan'))
        h8 = float('nan')
        if z and z['hf'] >= cr.HF_FLOOR and x['hf'] > 0:
            h8 = x['hf'] / z['hf']
        print('%-5d %5d %+9.2f | %7.3f %7.3f | %7.3f %7.3f | %7.2f'
              % (k, v, lv,
                 x['cen'] / y['cen'] if y['cen'] else 0.0,
                 (x['cen'] / z['cen']) if (z and z['cen']) else float('nan'),
                 hf, h8,
                 x['dur'] / y['dur'] if y['dur'] else 0.0))
    for f in (mid, wb, wr, wf):
        try:
            os.remove(f)
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
