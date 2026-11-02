"""Render a program's whole key list in ONE render per bank, and check it
against the per-note renders attack_probe already did.

WHY. attack_probe measures one note per render: 2083 samples is 2083
renders per bank, and each one pays for expanding the whole bank before
it synthesises a second of audio. Batching a program's keys into one
phrase is one render per (bank, program) - 129 instead of 2083 - and the
number it produces has to be the same number, or the batching is a
different measurement wearing the old one's name.

HOW THE IDENTITY IS KEPT. The measurement code is attack_probe's own
`measure()`, reached by returning a slice of the batched render where it
expects a path. Slicing from the note's onset is what the per-note render
does too - the essentia chain resets its filter state per array - so the
only thing that can differ is what follows the note, and that is exactly
what the comparison is for.

    python attack_batch.py TABLE.tsv [spacing_s]

TABLE.tsv is attack_probe's output; the rows with status OK are the work
list, so the comparison covers the same (program, key) pairs and nothing
else.
"""
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attack_probe as A  # noqa: E402

ROOT = A.ROOT
PLAYER = A.PLAYER
BANK = os.path.join(ROOT, 'banks', 'picked.ksb')
REF = os.path.join(ROOT, 'banks', 'gm.ksb')


def var(n):
    o = [n & 0x7F]
    n >>= 7
    while n:
        o.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(o))


def write_phrase(keys, path, gap_ticks, vel=A.VEL):
    """One note per key, `gap_ticks` apart, each held exactly as
    attack_probe's single-note phrase holds it.

    The gap has to be at least TOTAL_TICKS or a note's off would land
    after the next one's on and the deltas would go negative.
    """
    if gap_ticks < A.TOTAL_TICKS:
        raise SystemExit('gap %d is shorter than a note (%d ticks)'
                         % (gap_ticks, A.TOTAL_TICKS))
    trk = b'\x00\xff\x51\x03' + struct.pack('>I', A.TEMPO_US)[1:]
    prev = 0
    for i, k in enumerate(keys):
        on = i * gap_ticks
        trk += var(on - prev) + b'\x90' + bytes([k, vel])
        prev = on
        off = on + A.TOTAL_TICKS
        trk += var(off - prev) + b'\x80' + bytes([k, 0])
        prev = off
    trk += b'\x00\xff\x2f\x00'
    hdr = b'MThd' + struct.pack('>IHHH', 6, 0, 1, A.DIVISION)
    with open(path, 'wb') as f:
        f.write(hdr + b'MTrk' + struct.pack('>I', len(trk)) + trk)


def main(argv):
    table = argv[1]
    spacing_s = float(argv[2]) if len(argv) > 2 else 6.0
    check = int(argv[3]) if len(argv) > 3 else 24
    gap = int(round(spacing_s * A.DIVISION * 1e6 / A.TEMPO_US))
    work = {}
    hdr = open(table, encoding='utf-8').readline().rstrip('\n').split('\t')
    ix = {k: i for i, k in enumerate(hdr)}
    for line in open(table, encoding='utf-8'):
        r = line.rstrip('\n').split('\t')
        if len(r) < len(hdr) or r[ix['status']] != 'OK':
            continue
        work.setdefault(int(r[ix['program']]), []).append(int(r[ix['key']]))
    print('%d (program, key) pair(s) over %d program(s); spacing %.2f s'
          % (sum(len(v) for v in work.values()), len(work), spacing_s))

    def slice_measure(x, rate, i0, i1):
        seg = np.ascontiguousarray(x[i0:i1])
        orig = A.read_wav
        A.read_wav = lambda _p, s=seg, rr=rate: (s, rr)
        try:
            return A.measure('ignored')
        finally:
            A.read_wav = orig

    batch = {}
    nred = 0
    for p in sorted(work):
        keys = sorted(set(work[p]))
        midi = os.path.join(HERE, '_b%d.mid' % p)
        write_phrase(keys, midi, gap)
        for tag, bank in (('ours', BANK), ('ref', REF)):
            wav = os.path.join(HERE, '_b%d_%s.wav' % (p, tag))
            r = subprocess.run([PLAYER, '--bank', bank, '--prog', str(p),
                                '--wav', wav, midi],
                               capture_output=True, text=True)
            if not (r.returncode == 0 and os.path.exists(wav)):
                print('  p%d %s: render failed' % (p, tag))
                continue
            x, rate = A.read_wav(wav)
            nred += len(keys) - 1
            for i, k in enumerate(keys):
                i0 = int(round(i * spacing_s * rate))
                i1 = int(round((i + 1) * spacing_s * rate))
                batch[(p, k, tag)] = slice_measure(x, rate, i0, i1)
            os.remove(wav)
        os.remove(midi)

    # THE CONTROL ARM: the same pair rendered the way attack_probe renders
    # it - one note, one render - and measured by the same code. The old
    # table cannot serve as the control here because it was measured on a
    # different bank; a difference against it would say nothing about the
    # batching. This says everything about it.
    picks = []
    for p in sorted(work):
        keys = sorted(set(work[p]))
        step = max(1, len(keys) * 2 // max(1, check // max(1, len(work))))
        for k in keys[::step]:
            picks.append((p, k))
        if len(picks) >= check:
            break
    worst = {'ours': 0.0, 'ref': 0.0}
    bad = 0
    for p, k in picks:
        midi = os.path.join(HERE, '_c.mid')
        A.write_midi(k, midi)
        for tag, bank in (('ours', BANK), ('ref', REF)):
            wav = os.path.join(HERE, '_c_%s.wav' % tag)
            try:
                A.render(bank, p, k, midi, wav)
            except SystemExit:
                continue
            one = A.measure(wav)
            os.remove(wav)
            b = batch.get((p, k, tag))
            if not b or 'note' in one and one.get('note'):
                continue
            d = abs(float(one.get('lat_ess') or 0) - float(b.get('lat_ess') or 0))
            if d > worst[tag]:
                worst[tag] = d
            if d > 0.01:
                bad += 1
                print('  p%d key %d %s: batched %s vs per-note %s'
                      % (p, k, tag, one.get('lat_ess'), b.get('lat_ess')))
        os.remove(midi)

    print('%d pair(s) checked against the per-note render; worst '
          '|lat_ess| difference: ours %.6f, ref %.6f decades'
          % (len(picks), worst['ours'], worst['ref']))
    print('renders avoided by batching: %d' % nred)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
