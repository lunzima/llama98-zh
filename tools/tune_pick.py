"""Pick each program's source by RENDERING it, one note at a time, on
notes chosen from the program's own region boundaries.

    python tune_pick.py [--out map.json] [--only 1,2,3]

WHY NOT A FIXED SET OF TEST NOTES. Seven evenly spaced notes missed
the region boundary of SC-55's grand piano, which sits at key 64: the
program measured 2 cents at 62 and -34 at 64, and the selection never
saw it. A fixed set is a guess about where the zones divide; the
boundaries themselves are in the bank, so the test notes are taken
from there - every region's lowest key, highest key and midpoint -
plus a coarse grid for coverage.

WHY ONE RENDER PER NOTE. A seven-note render overlaps the tail of one
note with the onset of the next for anything with a long release, and
the measurement then lands between two pitches (a 0.4-cent instrument
read as 68). Each note is rendered alone, so nothing else is in the
window.
"""
import sys, os, json, struct, subprocess, wave, math, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pitch_measure as PM
from note_window import note_window
# THE ONSET CHAIN, FROM THE TOOL THAT BUILT IT. attack_probe carries the
# Essentia LogAttackTime copied literally from MTG/essentia; a second
# implementation here would drift from it and the two calibers would
# disagree without either being wrong.
from attack_probe import envelope_essentia, lat_essentia
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER = os.path.join(HERE, 'kunkun98-play.exe')
TMP = os.path.join(HERE, 'build', 'tmp')
# HOW MANY NOTES ARE RENDERED AT ONCE. Each one is a separate process
# that spends its time waiting on the player, so the ceiling is how many
# processes the disk and the scheduler want - not the GIL. Eight is well
# under the point where more workers stop being faster and start being
# thrash, and the run is 125k notes.
WORKERS = 8
# A NOTE RENDER TAKES UNDER A SECOND. One that has not returned in a
# minute is hung, and without a timeout the whole run waits on it
# forever - the stall this replaced.
RENDER_TIMEOUT = 60
# THE GM PERCUSSION RANGE, the keys a GM file can ask a kit for. The
# kit is measured on every key inside it.
KIT_KEY_LO, KIT_KEY_HI = 27, 87
SOURCES = [('gm', 'gm.dls'), ('FF8', 'FF8.dls'), ('Fury', 'Fury.dls'),
           ('GeneralUser_GS_144', 'GeneralUser_GS_144.dls'),
           ('FluidR3_GM', 'FluidR3_GM.dls'), ('Arachno', 'Arachno.dls'),
           ('SGM-V2.01', 'SGM-V2.01.dls'), ('SC-55', 'SC-55.dls'),
           ('Airfont_340', 'Airfont_340.dls'),
           ('Creative_28MB', 'Creative_28MB.dls'),
           ('Chorium', 'Chorium.dls')]


def bank_regions(path, prog):
    d = open(path, 'rb').read()
    n, nr, ns, ver, db = struct.unpack_from('<HHHHI', d, 4)
    io = 16
    ro = io + n * 7
    out = []
    for i in range(n):
        p, msb, lsb, cnt, first = struct.unpack_from('<BBBHH', d, io + i * 7)
        if p != prog:
            continue
        for j in range(cnt):
            o = ro + (first + j) * 104
            klo, khi = struct.unpack_from('<HH', d, o)
            out.append((klo, khi))
    return out


def test_notes(regions, perc=False):
    notes = set()
    for klo, khi in regions:
        notes.add(klo)
        notes.add(khi)
        notes.add((klo + khi) // 2)
    # A DRUM KIT IS SIXTY-ONE INSTRUMENTS, not one stretched across the
    # keyboard, so the grid below is the wrong instrument for it. That
    # grid is a melodic sampling of keys 21..108 at a stride of seven;
    # on the kit it landed on 42 and 49 of the ten cymbal keys, and the
    # twenty-note cap then dropped 49. The seven that were never
    # measured included key 44, the pedal hi-hat, which on one kit
    # reads a quarter of gm's brightness at two and a half times its
    # level - a defect no similarity reading could see, because the
    # reading never played the key. Every key the kit covers is a
    # different drum and is measured; the cap does not apply.
    if perc:
        for klo, khi in regions:
            notes.update(range(max(klo, KIT_KEY_LO),
                               min(khi, KIT_KEY_HI) + 1))
        return sorted(n for n in notes
                      if KIT_KEY_LO <= n <= KIT_KEY_HI)
    # THE GRID REACHES THE ENDS OF THE KEYBOARD. It used to stop at
    # 48..90, which left the top octave of several banks unmeasured -
    # FluidR3's grand piano had no test note above key 89, so its top
    # region could not be tuned or judged at all.
    #
    # NO CAP. The twenty-note subsample that used to follow dropped
    # 67 of the 88 keys in range - 76% - and the loss was not spread
    # evenly: it kept a stride-of-seven grid and threw away whatever
    # the region boundaries added between. A program's similarity and
    # tuning readings are then a claim about a quarter of its range.
    # The kit above is the same defect made visible; this is where it
    # was hiding for the melodic programs too.
    for n in range(21, 109, 7):
        notes.add(n)
    return sorted(n for n in notes if 21 <= n <= 108)


def write_note_midi(note, path, channel=0):
    def vlq(v):
        o = [v & 0x7F]
        v >>= 7
        while v:
            o.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(reversed(o))
    ev = (vlq(0) + bytes([0x90 | channel, note, 100]) + vlq(192)
          + bytes([0x80 | channel, note, 0]) + vlq(0)
          + bytes([0xFF, 0x2F, 0]))
    tempo = bytes([0, 0xFF, 0x51, 3, 7, 0xA1, 0x20, 0, 0xFF, 0x2F, 0])
    open(path, 'wb').write(b'MThd' + struct.pack('>IHHH', 6, 1, 2, 96) +
                           b'MTrk' + struct.pack('>I', len(tempo)) + tempo +
                           b'MTrk' + struct.pack('>I', len(ev)) + ev)


def midi_for(note, kit):
    """The fixture MIDI for one note, written once.

    One file per (note, kit) and shared by every bank and every program
    that wants that key, which is why the name carries neither. The
    workers inside one program ask for different notes - that is what
    makes them parallel - so two of them never want the same file."""
    path = os.path.join(TMP, 'n%d%s.mid' % (note, kit))
    if not os.path.exists(path):
        write_note_midi(note, path, 9 if kit else 0)
    return path


def logspec(seg, n=8192):
    """Log magnitude spectrum of a note's steady part, for comparing a
    candidate against gm's own rendering of the same note. Phase is
    irrelevant, the envelope is not: the correlation of the two logs is
    what the ear agrees on."""
    seg = np.asarray(seg, dtype=np.float64)
    seg = seg - seg.mean()
    if len(seg) < 1024:
        return None
    # ROUND DOWN TO A POWER OF TWO, AND NEVER PAST THE SEGMENT.
    # `1 << (len-1).bit_length()` rounds UP: 7938 samples became 8192,
    # and seg[:8192] is 7938 while np.hanning(8192) is 8192 - a shape
    # the multiply refuses. The fixed windows this used to see were all
    # longer than their rounded length, so it never showed; a window
    # that follows the note's own decay reaches lengths that are not.
    n = min(n, len(seg), 1 << ((len(seg)).bit_length() - 1))
    if n < 1024:
        return None
    sp = np.log1p(np.abs(np.fft.rfft(seg[:n] * np.hanning(n))))
    return sp


def sim(a, b):
    n = min(len(a), len(b))
    if n < 16:
        return 0.0
    aa, bb = a[:n], b[:n]
    aa = aa - aa.mean(); bb = bb - bb.mean()
    da = np.sqrt((aa * aa).sum()); db = np.sqrt((bb * bb).sum())
    if da <= 0 or db <= 0:
        return 0.0
    return float((aa * bb).sum() / (da * db))


def render_note(bank, prog, note, midi, window=None):
    """Render one note and measure it. Safe to call from many threads.

    THE WAV NAME CARRIES THE PROGRAM. It used to be the source's file
    name and the note alone, which is a unique key only while the caller
    is sequential: two programs measuring the same key on the same bank
    write the same file, and one reads the other's render.

    A TIMEOUT ON THE PLAYER. A single note takes under a second; one
    that has not returned in a minute is not going to, and without a
    timeout it stalls the whole run.
    """
    wv = os.path.join(TMP, 'tn_%d_%s_%d.wav'
                      % (prog, os.path.basename(bank), note))
    try:
        r = subprocess.run([PLAYER, '--bank', bank, '--prog', str(prog),
                            '--wav', wv, midi], capture_output=True,
                           timeout=RENDER_TIMEOUT)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or not os.path.exists(wv):
        return None
    try:
        w = wave.open(wv, 'rb')
        n, ch, rate = w.getnframes(), w.getnchannels(), w.getframerate()
        a = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float64)
        w.close()
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        # THE WINDOW IS THE NOTE'S OWN ACTIVE REGION, not a fixed slice.
        #
        # 0.35-0.95 s measures a pad's sustain and finds nothing left of
        # a woodblock; 0.02-0.25 s finds a woodblock's attack and hears
        # only the attack of a pad. Each is half the question, and
        # swapping between them by program number made it worse: p115's
        # Chorium returned no reading at all under the late window and
        # was left without a `balance` figure, so it won that program on
        # the criteria that did have readings, at 24 dB of internal
        # spread.
        #
        # What level and timbre mean is what the note delivers WHILE IT
        # SOUNDS, so the window runs from where it starts to where it
        # has fallen 25 dB - the rule _note_rms uses for the calibration
        # grid, and the rule corpus_report applies when it decides peak
        # or energy. Capped at two seconds so one pad with a twenty-
        # second tail cannot dominate its program; both banks meet the
        # same cap, so the ratio is still a ratio.
        perc = prog == 128
        # THE WINDOW COMES FROM THE REFERENCE. `balance` and `sim` are
        # bank-against-gm ratios, and a window derived from the bank
        # being measured makes a slow-decaying bank read longer and
        # higher - the ratio then describes two different spans. gm is
        # rendered first, its window is passed down, and every source
        # is measured over it. A woodblock still gets its whole decay
        # because gm's woodblock decays the same way.
        if window is not None:
            lo, hi = window
        else:
            got = note_window(a, rate)
            if got is None:
                return None
            lo, hi = got
        if hi - lo < 1024 or np.abs(a[lo:hi]).max() < 50:
            return None
        exp = 440.0 * 2.0 ** ((note - 69) / 12.0)
        if perc:
            c = None
        else:
            got = PM.fine_f0(a[lo:hi], rate, exp, span_cents=70, harmonics=3)
            # NO PITCH IS NOT NO NOTE. A woodblock, an agogo and a
            # tinkle bell have no fundamental to find, and a window as
            # short as their decay gives the estimator too few cycles
            # even when there is one. Dropping the row for want of a
            # pitch reading is what left p115 without a balance figure;
            # the spectrum and the seam are still the evidence.
            c = PM.cents(got[0], exp) if got[0] > 0 else None
        # THE STRONGEST LINE IN THE WINDOW IS THE SECOND OPINION. The
        # harmonic search can settle on a combination of partials when
        # a note is sparse - SC-55's top piano key read -15.7 cents
        # while its own strongest partial sat at +9.9, and a half
        # correction built on that moved the pitch the wrong way. When
        # the two disagree by more than 12 cents the note's pitch is
        # not evidence; the spectrum still is, so the note stays for
        # similarity and the seam, with no tuning reading.
        steady = a[lo:hi]
        sp = np.abs(np.fft.rfft(steady * np.hanning(len(steady))))
        fr = np.fft.rfftfreq(len(steady), 1.0 / rate)
        b0 = int(np.searchsorted(fr, exp * 2.0 ** (-70.0 / 1200.0)))
        b1 = int(np.searchsorted(fr, exp * 2.0 ** (70.0 / 1200.0)))
        if b1 > b0 + 1:
            k = b0 + int(np.argmax(sp[b0:b1]))
            # THE BIN CENTRE IS NOT THE LINE. A 0.6 s window is 1.7 Hz
            # wide, which is 105 cents at the bottom of the piano; the
            # raw bin read there disagreed with a correct harmonic
            # estimate by 30 cents and threw the reading away.
            # Log-magnitude parabolic interpolation is the standard
            # estimator for a Hann-windowed line and brings that to
            # some 10 cents.
            if 0 < k < len(sp) - 1:
                y0, y1, y2 = (math.log(sp[k - 1] + 1e-12),
                              math.log(sp[k] + 1e-12),
                              math.log(sp[k + 1] + 1e-12))
                den = y0 - 2.0 * y1 + y2
                if abs(den) > 1e-12:
                    k = k + 0.5 * (y0 - y2) / den
            pc = PM.cents(float(k * rate / len(steady)), exp)
            # 20 CENTS: wide enough for the estimator's own error at the
            # bottom of the keyboard, narrow enough to reject the
            # mislock that motivated the check (25.6 cents at SC-55's
            # top piano key).
            if c is not None and abs(pc - c) > 20.0:
                c = None
        # A READING BEYOND THE SEARCH WINDOW IS STILL EVIDENCE. It used
        # to be dropped, so a source a semitone out simply had fewer
        # notes and could win on the ones left; clamp the magnitude and
        # keep it.
        if c is not None and abs(c) > 69.0:
            c = 69.0 if c > 0.0 else -69.0
        # AFTER THE ATTACK, on whatever the note's own timescale is. A
        # second-difference peak in the first 100 ms IS the attack, so
        # measuring it there reads 0.7-1.9 for every kit and ranks
        # nothing; the decay is where a sample's own noise lives. The
        # split is at 20% into the active region, which is 12 ms into a
        # hi-hat and 0.4 s into a pad - each note's own attack, not a
        # constant that fits one of them.
        seg = a[min(hi, lo + (hi - lo) // 5):hi]
        seam = 0.0
        if len(seg) > 2048:
            d2 = np.abs(seg[2:] - 2 * seg[1:-1] + seg[:-2])
            # RELATIVE TO THE SIGNAL. The absolute second difference
            # ranks a loud source above a quiet one with the same seam;
            # the calibration removes that difference later, so the
            # selection must not score it.
            peak = max(1.0, float(np.abs(seg).max()))
            seam = float(np.percentile(d2, 99.9)) / peak
        steady = a[lo:hi]
        # THE LEVEL OF THE SAME WINDOW, for the internal-balance reading.
        # A program can sit at gm's median level and still be loud at one
        # end and quiet at the other; the ratio per key is what says so,
        # and it has to be measured per note because the ratio is the
        # reading - a level taken after any averaging is not.
        lvl = float(np.sqrt((steady * steady).mean())) if len(steady) else 0.0
        # THE RENDERED SIGNAL-TO-NOISE, peak against what is left of the
        # note after it has stopped. This is the noise a listener hears
        # on a sustained instrument, and NOTHING ELSE MEASURES IT: the
        # sample-level `noise` criterion reads hiss in the raw sample,
        # which is a different quantity. Measured on the shipped field,
        # the harp was chosen from the one source scoring 16.7 dB here
        # against gm's 40.1 - the worst of all eleven - and its noise
        # reading was the highest in the pool. The pad was chosen at
        # 7.0 dB against two sources at 49.5.
        ipk = int(np.argmax(a * a)) if len(a) else 0
        t0 = min(len(a), ipk + int(1.5 * rate))
        t1 = min(len(a), t0 + int(0.5 * rate))
        tail = a[t0:t1] if t1 - t0 > 512 else a[-max(512, len(a) // 10):]
        flr = float(np.sqrt((tail * tail).mean())) if len(tail) else 0.0
        pk = float(np.abs(a).max()) if len(a) else 0.0
        snr = (20.0 * math.log10(pk / flr) if (pk > 0 and flr > 0) else 0.0)
        # THE ATTACK, MEASURED ON THE WHOLE NOTE AND NOT ON THE WINDOW.
        # This is the one figure that sees the first tens of milliseconds:
        # a sample that takes 50 ms to build reads the same body as one
        # that takes 5. Measured on the bank, 41 of 129 programs are more
        # than 2 dB off gm at a 50 ms gate while every one of them is
        # inside 1.8 dB at the 0.8 s gate the audit calibrates on - so the
        # body measurements above cannot see this at all.
        #
        # THE -5 SENTINEL IS NOT A READING. Essentia returns log10(seconds)
        # and -5 when the attack cannot be resolved; two of those read a
        # difference of zero, which would be "identical onsets" when it is
        # really "neither one could be measured". They come back as missing
        # and are left out of the comparison rather than scored as perfect.
        try:
            lat, _ts, _tp, _n = lat_essentia(envelope_essentia(a, rate), rate)
        except Exception:
            lat = None
        if lat is not None and lat <= -4.5:
            lat = None
        return (None if c is None else round(c, 3), round(seam, 4),
                logspec(steady), round(lvl, 3), (lo, hi), round(snr, 3),
                None if lat is None else round(float(lat), 4))
    finally:
        try:
            os.remove(wv)
        except OSError:
            pass


def main(argv):
    t0 = time.time()
    outpath = os.path.join(TMP, 'tune_map_boundary.json')
    only = None
    i = 1
    while i < len(argv):
        if argv[i] == '--out':
            outpath = argv[i + 1]; i += 2
        elif argv[i] == '--only':
            only = set(int(x) for x in argv[i + 1].split(',')); i += 2
        else:
            i += 1
    banks = {}
    for name, fn in SOURCES:
        path = os.path.join(TMP, 'full_%s.ksb' % name)
        if not os.path.exists(path):
            print('missing bank', path, file=sys.stderr)
            return 1
        banks[name] = path
    table = {}
    if os.path.exists(outpath):
        table = json.load(open(outpath))
    for prog in range(129):
        if only is not None and prog not in only:
            continue
        # A KIT PLAYS ON THE DRUM CHANNEL. Channel 0 with a forced
        # program 128 resolves no zone at all - measured: the render
        # came back with the note unvoiced - so the kit's fixtures use
        # channel 9 and its own file names.
        kit = '_k' if prog == 128 else ''
        regions = []
        for name in banks:
            regions.extend(bank_regions(banks[name], prog))
        notes = test_notes(regions, perc=(prog == 128))
        row = table.get(str(prog), {})
        # THE REFERENCES ARE gm AND FF8, each rendered once per note
        # and shared by every candidate. gm is the tuning and level
        # reference; FF8 is the character the bank leans toward where
        # FF8 has the program at all. A program FF8 does not carry
        # produces no FF8 render and therefore no ff8sim reading.
        #
        # EVERY RENDER FOR THIS PROGRAM GOES OUT AT ONCE - the notes of
        # every bank it competes on, including the two references -
        # and the assembly below only reads what the pool produced.
        # Per-note renders are independent processes that mostly wait,
        # so running them one at a time was spending the run on
        # turn-taking rather than on work.
        # GM FIRST AND ALONE: its windows decide every other render's.
        cache = {}
        refs = {}
        for rname in ('gm', 'FF8'):
            if rname not in banks:
                continue
            r = {}
            for note in notes:
                midi = midi_for(note, kit)
                got = render_note(banks[rname], prog, note, midi)
                cache[(banks[rname], note)] = got
                if got is not None and got[2] is not None:
                    r[note] = (got[2], got[3], got[4])
            if r:
                refs[rname] = r
        want = {}
        for name in banks:
            if name in row or name == 'gm' or name == 'FF8':
                continue
            for note in notes:
                want[(banks[name], note)] = None
        if want:
            gmwin = {n: (refs.get('gm', {}).get(n) or (None,) * 3)[2]
                     for n in notes}
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                gmwin = {n: (refs.get('gm', {}).get(n) or (None,) * 3)[2]
                         for n in notes}
                futs = {ex.submit(render_note, b, prog, n, midi_for(n, kit),
                                  None if b == banks.get('gm') else
                                  gmwin.get(n)):
                        (b, n) for (b, n) in want}
                for f in as_completed(futs):
                    cache[futs[f]] = f.result()
        for name in banks:
            if name in row:
                continue
            vals = []
            sims = []
            fsims = []
            devs = []
            snrs = []
            lats = []
            seam = 0.0
            for note in notes:
                got = cache.get((banks[name], note))
                if got is None:
                    continue
                if got[0] is not None:
                    vals.append((note, got[0]))
                seam = max(seam, got[1])
                spec = got[2]
                base = refs.get('gm', {}).get(note)
                if base is not None and base[1] > 0 and got[3] > 0:
                    devs.append(20.0 * math.log10(got[3] / base[1]))
                if got[5]:
                    snrs.append(got[5])
                # THE ONSET DIFFERENCE AGAINST GM. Same shape as `devs`:
                # the two banks' own attack times on the same key. A note
                # where either side could not resolve an attack is left
                # out rather than counted as a difference of zero - see
                # render_note, the -5 sentinel is not a reading.
                gl = (cache.get((banks.get('gm'), note)) or (None,) * 7)[6]
                if got[6] is not None and gl is not None:
                    lats.append(abs(got[6] - gl))
                if spec is None:
                    continue
                for rname, acc in (('gm', sims), ('FF8', fsims)):
                    b = refs.get(rname, {}).get(note)
                    if b is not None:
                        acc.append(sim(spec, b[0]))
            if vals or seam > 0.0:
                row[name] = {'notes': vals, 'seam': seam,
                             'sim': round(float(np.median(sims)), 4) if sims else 0.0,
                             'ff8sim': (round(float(np.median(fsims)), 4)
                                        if fsims else None),
                             # THE INTERNAL BALANCE: how wide the level
                             # runs across the keys of the program,
                             # against gm on the same keys. A program at
                             # gm's median level can still be 18 dB down
                             # on one drum and right on the rest - a
                             # measured case - and one number per program
                             # cannot see it. Wider is worse; it is the
                             # spread, not the offset, because the offset
                             # is what the level pass corrects.
                             'balance': (round(max(devs) - min(devs), 3)
                                         if len(devs) >= 2 else None),
                             # THE RENDERED SIGNAL-TO-NOISE, median over
                             # the notes. See render_note: this is the
                             # noise on a sustained instrument and no
                             # other criterion measures it.
                             'snr': (round(float(np.median(snrs)), 3)
                                     if snrs else None),
                             # THE ONSET DIFFERENCE AGAINST GM, median over
                             # the keys where both sides resolved an
                             # attack. Lower is better, and it is the only
                             # criterion here that sees the first tens of
                             # milliseconds.
                             'latdiff': (round(float(np.median(lats)), 4)
                                         if lats else None)}
        if not row:
            continue
        table[str(prog)] = row
        def worst_note(kv):
            ns = kv[1]['notes']
            return (max((abs(v[1]) for v in ns), default=0.0), len(ns))
        best = min(row.items(), key=worst_note)
        wn, nn = worst_note(best)
        # FLUSH. Redirected stdout is block-buffered, so without it the
        # log stays empty for as long as the buffer holds - which reads
        # as a stall when the run is actually working. The line carries
        # a clock because "alive" and "stuck" look identical from the
        # outside unless the number moves.
        print('p%-3d best %-20s worst %+6.2f over %d note(s)  %.0fs' % (
            prog, best[0], wn, nn, time.time() - t0), flush=True)
        # AFTER EVERY PROGRAM. Saving on a stride of four threw away
        # four programs of work to a stall; a run that loses one is
        # recoverable by rerunning, and the table already loads what it
        # has.
        json.dump(table, open(outpath, 'w'), indent=0)
    json.dump(table, open(outpath, 'w'), indent=0)
    print('wrote', outpath)


if __name__ == '__main__':
    main(sys.argv)
