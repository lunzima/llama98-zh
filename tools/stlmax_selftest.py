"""The acceptance gate for tools/stlmax.py: seven checks, six mutations.

Run it after ANY change to the module. If it is green on the constants and
on the pooling, the instrument is still measuring what it says it does:

    python tools/stlmax_selftest.py                  # 7/7, exit 0
    python tools/stlmax_selftest.py --mutate attack-x10   # red, exit 1

Signals (known envelopes, 1 kHz tones - A-weighting is exactly 0 dB there):
  * fast-decay pulse: a 120 ms window, envelope exp(-t/1.5 ms), so the
    power falls as exp(-2t/1.5 ms);
  * steady pad: 5 s, 5 ms cosine fade-in.

WHY THE PULSE DECAYS THIS FAST. A 22 ms attack puts a hard ceiling on any
window of length L: a short sound's STLmax can beat its window average by
at most 10*log10(L/22ms). At L=120 ms that ceiling is 10*log10(0.12/0.022)
= 7.37 dB. A 30 ms exponential "marimba" envelope only reaches about 3.8 dB
and cannot clear the "> 6 dB" line; concentrating the energy in the first
few ms approaches the ceiling (1.5 ms measures 6.76 dB). This is the model's
ceiling at L=120 ms, not a threshold tuned to make a test pass.

The mutations, and what each one must break:
  attack-x10      ATTACK_S 0.22     -> attack cost, energy gap
  attack-fast     ATTACK_S 0.0022   -> attack cost
  alpha-unit      1 ms step read as 1 s -> attack cost, release
  release-x10     RELEASE_S 0.5     -> release
  release-2018    RELEASE_S 0.030   -> release (version switch, NOT an
                                      error: both TVL revisions are valid;
                                      it proves the check is tied to 2002)
  pool-mean       max replaced by mean -> energy gap reads 0.000
"""
import sys

import numpy as np

# On a cp936 console the piped output is written as UTF-8; an interactive
# terminal keeps the system encoding. Without this the Chinese is mojibake.
if not sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import stlmax as S

SR = 44100
PULSE_TAU = 0.0015
PULSE_DUR = 0.12
PAD_DUR = 5.0
TONE_HZ = 1000.0
CROSSFADE_S = 0.005
GAIN_DB = 6.0


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------
def make_pulse(tau=PULSE_TAU, dur=PULSE_DUR, f=TONE_HZ, amp=1.0):
    t = np.arange(int(round(dur * SR))) / float(SR)
    return amp * np.exp(-t / tau) * np.sin(2.0 * np.pi * f * t)


def make_pad(dur=PAD_DUR, f=TONE_HZ, amp=1.0):
    t = np.arange(int(round(dur * SR))) / float(SR)
    env = np.ones_like(t)
    k = int(round(CROSSFADE_S * SR))
    env[:k] = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / float(k))
    return amp * env * np.sin(2.0 * np.pi * f * t)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check(results, name, ok, detail):
    results.append((name, bool(ok), detail))


def run_checks():
    results = []
    pulse = make_pulse()
    pad = make_pad()

    # 0. The constants are the published ones: alpha = 1 - exp(-1 ms / tau),
    #    for both TVL revisions. The bracketed values are as published.
    a02 = S.alphas(**S.TVL_2002)
    a18 = S.alphas(**S.TVL_2018)
    want02 = {'attack': 0.045, 'release': 0.020,
              'lt_attack': 0.010, 'lt_release': 0.0005}
    want18 = {'attack': 0.045, 'release': 0.033,
              'lt_attack': 0.010, 'lt_release': 0.00133}
    bad = [(k, a02[k], want02[k]) for k in want02
           if abs(a02[k] - want02[k]) > 0.001]
    bad += [('2018.' + k, a18[k], want18[k]) for k in want18
            if abs(a18[k] - want18[k]) > 0.001]
    check(results, 'TVL constants: alpha match (2002/2018)', not bad,
          '2002 a=%.4f r=%.4f | 2018 r=%.4f ltr=%.5f%s'
          % (a02['attack'], a02['release'], a18['release'],
             a18['lt_release'], '' if not bad else '  dev:%r' % bad))

    # Calibration: scale the pulse so its STLmax equals the pad's. Gain is a
    # pure translation in this model, so one scaling suffices - and that is
    # also why the calibration is not a fit.
    a0 = S.analyse_note(pulse, SR, 0.0, PULSE_DUR)
    b = S.analyse_note(pad, SR, 0.0, PAD_DUR)
    gain_db = b['stlmax_db'] - a0['stlmax_db']
    pulse_c = pulse * 10.0 ** (gain_db / 20.0)
    a = S.analyse_note(pulse_c, SR, 0.0, PULSE_DUR)

    # 1. The calibration itself: equal STLmax.
    d_stl = a['stlmax_db'] - b['stlmax_db']
    check(results, 'STLmax equal after calibration', abs(d_stl) < 0.5,
          'pulse-pad = %+.3f dB (limit 0.5)' % d_stl)

    # 2. The same two signals must fall far apart under the energy average.
    d_en = a['energy_db'] - b['energy_db']
    check(results, 'energy average differs by > 6 dB', d_en < -6.0,
          'pulse-pad = %+.3f dB (limit -6; calibration gain %+.2f dB)'
          % (d_en, gain_db))

    # 3. STLmax never exceeds the window's instantaneous ceiling.
    ok = (a['stlmax_db'] <= a['inst_max_db'] + 1e-9
          and b['stlmax_db'] <= b['inst_max_db'] + 1e-9)
    check(results, 'STLmax <= instantaneous ceiling', ok,
          'pulse %.2f <= %.2f ; pad %.2f <= %.2f'
          % (a['stlmax_db'], a['inst_max_db'],
             b['stlmax_db'], b['inst_max_db']))

    # 4. Gain invariance: +-6 dB in -> +-6 dB out, within 0.01 dB.
    base = S.analyse_note(pad, SR, 0.0, PAD_DUR)['stlmax_db']
    worst = 0.0
    worst_g = 0.0
    for g in (-GAIN_DB, GAIN_DB):
        y = S.analyse_note(pad * 10.0 ** (g / 20.0), SR, 0.0, PAD_DUR)
        e = abs((y['stlmax_db'] - base) - g)
        if e > worst:
            worst, worst_g = e, g
    check(results, 'gain +-6 dB -> STLmax +-6 dB', worst < 0.01,
          'max error %.2e dB (at %+g dB)' % (worst, worst_g))

    # 5. The price of a 22 ms attack: the pulse's STLmax must sit clearly
    #    below its instantaneous peak. Measured -12.4 dB with the right
    #    constant; [-20,-8] catches attack x10 (-21.9), /10 (-4.4) and an
    #    alpha that ignores the step (0.0).
    cost = a['stlmax_db'] - a['inst_max_db']
    check(results, 'attack 22 ms costs [-20,-8] dB', -20.0 <= cost <= -8.0,
          'pulse STLmax - peak = %+.2f dB' % cost)

    # 6. Release constant: 50 ms after a steady tone stops, the short-term
    #    level must sit at e^-1. Frames are 1 ms; the last sounding one
    #    (0.5 s tone) has its centre at 0.4995 s.
    x = make_pad(dur=0.5)
    x = np.concatenate([x, np.zeros(int(round(0.2 * SR)))])
    t, lev, y = S.short_term_curve(x, SR)
    i0 = int(round(0.4995 / S.FRAME_S))
    i1 = i0 + 50
    drop = 10.0 * np.log10(y[i1] / y[i0])
    want = 10.0 * np.log10(np.exp(-1.0))
    check(results, 'release 50 ms: -4.34 dB after +50 ms',
          abs(drop - want) < 0.2,
          'measured %+.2f dB, expected %+.2f dB (t=%.4f..%.4f)'
          % (drop, want, t[i0], t[i1]))

    return results


# --------------------------------------------------------------------------
# mutations
# --------------------------------------------------------------------------
def _mut_alpha_unit(levels, dt=S.FRAME_S, attack_s=None, release_s=None):
    """Read the 1 ms step as 1 s: alpha ~ 1, the smoother follows instantly."""
    lev = np.asarray(levels, dtype=np.float64)
    if attack_s is None:
        attack_s = S.ATTACK_S
    if release_s is None:
        release_s = S.RELEASE_S
    aa = 1.0 - np.exp(-1.0 / attack_s)
    ar = 1.0 - np.exp(-1.0 / release_s)
    out = np.empty_like(lev)
    prev = 0.0
    for i in range(len(lev)):
        v = lev[i]
        a = aa if v > prev else ar
        prev = prev + a * (v - prev)
        out[i] = prev
    return out


def _mut_pool_mean():
    """Replace the window maximum with the window mean."""
    orig = S.analyse_note

    def wrapped(x, sr, t0=0.0, t1=None, **kw):
        d = orig(x, sr, t0, t1, **kw)
        d['stlmax_db'] = d['energy_db']
        return d
    S.analyse_note = wrapped


MUTATIONS = {
    'attack-x10': ('ATTACK_S = 0.22 (time constant x10)',
                   lambda: setattr(S, 'ATTACK_S', 0.22)),
    'attack-fast': ('ATTACK_S = 0.0022 (x0.1)',
                    lambda: setattr(S, 'ATTACK_S', 0.0022)),
    'alpha-unit': ('1 ms step read as 1 s (instantaneous smoother)',
                   lambda: setattr(S, 'smooth_levels', _mut_alpha_unit)),
    'release-x10': ('RELEASE_S = 0.5 (release x10)',
                    lambda: setattr(S, 'RELEASE_S', 0.5)),
    'release-2018': ('switch to the 2018 release (30 ms): the number must '
                     'move - both revisions are valid, this proves the check '
                     'is tied to 2002',
                     lambda: setattr(S, 'RELEASE_S', 0.030)),
    'pool-mean': ('STLmax uses the window mean instead of the maximum',
                  _mut_pool_mean),
}


def main(argv):
    mut = None
    args = list(argv[1:])
    if args and args[0] == '--mutate':
        if len(args) < 2 or args[1] not in MUTATIONS:
            sys.stderr.write('usage: stlmax_selftest.py [--mutate %s]\n'
                             % '|'.join(MUTATIONS))
            return 2
        mut = args[1]
    elif args:
        sys.stderr.write('usage: stlmax_selftest.py [--mutate %s]\n'
                         % '|'.join(MUTATIONS))
        return 2

    if mut:
        desc, fn = MUTATIONS[mut]
        fn()
        print('== mutation: %s : %s' % (mut, desc))
    else:
        print('== stlmax selftest (unmutated)')

    results = run_checks()
    bad = 0
    for name, ok, detail in results:
        print('  %-42s %s   %s' % (name, 'PASS' if ok else 'FAIL', detail))
        if not ok:
            bad += 1
    print('== %d/%d passed' % (len(results) - bad, len(results)))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
