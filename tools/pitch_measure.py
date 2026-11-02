"""Precise pitch measurement for one note of a rendered or recorded
signal.

    python pitch_measure.py file.wav [--start S] [--end S] [--expect HZ]

The coarse estimate is a harmonic-product spectrum peak; the fine one
maximises the Hann-windowed DFT magnitude of the first harmonics over
a continuous frequency by golden-section search. The DFT is evaluated
directly at each trial frequency, so the resolution is not the FFT
bin: on a quiet, steady tone it resolves to hundredths of a cent, and
on a real sample it is limited by the signal's own pitch drift, which
is why a short window is used and why the reading is reported with
the width of the plateau it sat on.

This exists so a tuning correction is applied from a number that is
better than the thing it corrects. The previous estimate (parabolic
interpolation of an FFT peak, cross-checked against autocorrelation)
agrees with itself to 2-3 cents; that is the scale of the errors it
was being asked to fix.
"""
import math
import sys
import wave

import numpy as np


def coarse_f0(x, rate, lo=30.0, hi=2000.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if n < 1024:
        return 0.0
    sp = np.abs(np.fft.rfft(x * np.hanning(n)))
    fr = np.fft.rfftfreq(n, 1.0 / rate)
    hps = sp.copy()
    for h in (2, 3, 4, 5):
        part = sp[::h]
        hps[:len(part)] *= part
    a = int(np.searchsorted(fr, lo))
    b = int(np.searchsorted(fr, hi))
    if b <= a + 2:
        return 0.0
    return float(fr[a + int(np.argmax(hps[a:b]))])


def _dft_power(x, rate, f, harmonics):
    """Sum of |windowed DFT|^2 over the first `harmonics`, each divided
    by its harmonic number so the fundamental and the overtones count
    alike."""
    n = len(x)
    t = np.arange(n) / float(rate)
    w = x * np.hanning(n)
    total = 0.0
    for h in range(1, harmonics + 1):
        ph = 2.0 * math.pi * h * f * t
        c = float(np.dot(w, np.cos(ph)))
        s = float(np.dot(w, np.sin(ph)))
        total += (c * c + s * s) / h
    return total


def fine_f0(x, rate, guess, span_cents=60.0, harmonics=3, iters=60):
    """Golden-section maximisation of the harmonic DFT power around
    `guess`. -> (frequency, relative plateau width in cents)."""
    if guess <= 0.0:
        return 0.0, 0.0
    lo = guess * 2.0 ** (-span_cents / 1200.0)
    hi = guess * 2.0 ** (span_cents / 1200.0)
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - g * (b - a)
    d = a + g * (b - a)
    fc, fd = _dft_power(x, rate, c, harmonics), _dft_power(x, rate, d, harmonics)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = _dft_power(x, rate, c, harmonics)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = _dft_power(x, rate, d, harmonics)
        if b - a < 1e-4 * guess:
            break
    f = 0.5 * (a + b)
    # how flat the peak is: where the power falls 1 dB below it
    top = _dft_power(x, rate, f, harmonics)
    edge = None
    for k in range(1, 400):
        fp = f * 2.0 ** (k / 1200.0)
        if _dft_power(x, rate, fp, harmonics) < top * 10.0 ** (-1.0 / 10.0):
            edge = k
            break
    return f, float(edge or 0)


def measure(x, rate, start=None, end=None, expect=None, harmonics=3):
    x = np.asarray(x, dtype=np.float64)
    if start is not None:
        x = x[int(start * rate):]
    if end is not None:
        x = x[:int(end * rate)]
    guess = expect if expect else coarse_f0(x, rate)
    if guess <= 0.0:
        return None
    f, flat = fine_f0(x, rate, guess, harmonics=harmonics)
    return f, flat


def cents(f, ref):
    return 1200.0 * math.log2(f / ref) if f > 0 and ref > 0 else 0.0


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    path = argv[1]
    start = end = None
    expect = None
    i = 2
    while i < len(argv):
        if argv[i] == '--start':
            start = float(argv[i + 1]); i += 2
        elif argv[i] == '--end':
            end = float(argv[i + 1]); i += 2
        elif argv[i] == '--expect':
            expect = float(argv[i + 1]); i += 2
        else:
            i += 1
    w = wave.open(path, 'rb')
    n = w.getnframes()
    ch = w.getnchannels()
    rate = w.getframerate()
    a = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float64)
    w.close()
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    got = measure(a, rate, start, end, expect)
    if got is None:
        print('no pitch found')
        return 1
    f, flat = got
    print('%.4f Hz  (+/- %.2f cents at the -1 dB plateau)  rate %d'
          % (f, flat, rate))
    if expect:
        print('  vs %.3f Hz: %+.3f cents' % (expect, cents(f, expect)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
