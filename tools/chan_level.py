"""Per-channel loudness of one piece, our bank against a reference bank.

THE QUESTION. corpus_report answers "is program P's level right", one note
at a time, over the corpus. A listener asks a different one: in THIS piece
some instruments come out too loud and some too quiet - which ones, and by
how much against the reference. Per-program single notes cannot see a mix:
a program can be right on its own and wrong where the piece puts it, and
the piece is what gets played.

HOW. Each channel is rendered on its own: every other channel's channel
events are dropped while all meta and sysex events are kept, so the tempo
map and the time axis are untouched and the isolated channel lands exactly
where it lands in the full render. Both banks are rendered from the SAME
isolated MIDI, and the two are compared over the note windows of that
channel - windows read out of the MIDI, not out of the audio, so the two
arms cannot drift onto different sample surfaces.

The number reported is the gated RMS in dB over the note windows, ours
against the reference, plus the two absolute levels so a loud reference and
a quiet one are not confused with each other.

    python tools/chan_level.py SONG.mid [--bank B] [--ref B]
                               [--chan 3,7] [--wav-dir DIR]

Defaults: --bank banks/picked.ksb --ref banks/gm.ksb --wav-dir build/tmp.
Renders are written to WAV_DIR and left there; each is the isolated channel,
so a 15-channel piece makes 30 files. Use --keep 0 to delete them after.
"""
import os
import re
import struct
import subprocess
import sys
import wave
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLAY = os.path.join(ROOT, 'kunkun98-play.exe')
DB = lambda x: 20.0 * np.log10(max(x, 1e-12))

# GM program names, 0-indexed. The reader of this table is asking "WHICH
# instrument is too loud", and a bare program number makes them look it up.
GM = (
 'Acoustic Grand','Bright Acoustic','Electric Grand','Honky-tonk','E.Piano 1','E.Piano 2',
 'Harpsichord','Clavinet','Celesta','Glockenspiel','Music Box','Vibraphone','Marimba',
 'Xylophone','Tubular Bells','Dulcimer','Drawbar Organ','Percussive Organ','Rock Organ',
 'Church Organ','Reed Organ','Accordion','Harmonica','Tango Accordion','Nylon Guitar',
 'Steel Guitar','Jazz Guitar','Clean Guitar','Muted Guitar','Overdriven','Distortion',
 'Guitar Harmonics','Acoustic Bass','Finger Bass','Pick Bass','Fretless Bass','Slap Bass 1',
 'Slap Bass 2','Synth Bass 1','Synth Bass 2','Violin','Viola','Cello','Contrabass',
 'Tremolo Strings','Pizzicato','Orchestral Harp','Timpani','Strings 1','Strings 2',
 'Synth Strings 1','Synth Strings 2','Choir Aahs','Voice Oohs','Synth Voice','Orchestra Hit',
 'Trumpet','Trombone','Tuba','Muted Trumpet','French Horn','Brass Section','Synth Brass 1',
 'Synth Brass 2','Soprano Sax','Alto Sax','Tenor Sax','Baritone Sax','Oboe','English Horn',
 'Bassoon','Clarinet','Piccolo','Flute','Recorder','Pan Flute','Blown Bottle','Shakuhachi',
 'Whistle','Ocarina','Square Lead','Saw Lead','Calliope','Chiff Lead','Charang','Voice Lead',
 'Fifths Lead','Bass & Lead','New Age Pad','Warm Pad','Polysynth','Choir Pad','Bowed Pad',
 'Metallic Pad','Halo Pad','Sweep Pad','Rain','Soundtrack','Crystal','Atmosphere','Brightness',
 'Goblins','Echoes','Sci-fi','Sitar','Banjo','Shamisen','Koto','Kalimba','Bag Pipe','Fiddle',
 'Shanai','Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom',
 'Synth Drum','Reverse Cymbal','Guitar Fret Noise','Breath Noise','Seashore','Bird Tweet',
 'Telephone Ring','Helicopter','Applause','Gunshot')


def gname(p):
    return GM[p] if p is not None and 0 <= p < len(GM) else '-'


# ---------------------------------------------------------------- MIDI
def vlq_read(b, p):
    v = 0
    while True:
        c = b[p]
        p += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, p


def vlq_put(v):
    out = bytearray([v & 0x7F])
    v >>= 7
    while v:
        out.insert(0, 0x80 | (v & 0x7F))
        v >>= 7
    return bytes(out)


def read_midi(path):
    b = open(path, 'rb').read()
    if b[:4] != b'MThd':
        raise ValueError('not a MIDI file: %s' % path)
    fmt, ntrk, div = struct.unpack('>HHH', b[8:14])
    tracks = []
    i = 14
    while i + 8 <= len(b):
        ln = struct.unpack('>I', b[i + 4:i + 8])[0]
        tracks.append(b[i + 8:i + 8 + ln])
        i += 8 + ln
    return fmt, div, tracks


def walk(trk):
    """Yield (abs_tick, raw_bytes, channel_or_None). Running status is
    resolved, so raw_bytes always carries its own status byte."""
    p = 0
    tick = 0
    st = None
    while p < len(trk):
        d, p = vlq_read(trk, p)
        tick += d
        b = trk[p]
        if b & 0x80:
            st = b
            p += 1
        if st == 0xFF:
            t = trk[p]
            p += 1
            ln, p = vlq_read(trk, p)
            data = trk[p:p + ln]
            p += ln
            yield tick, bytes([0xFF, t]) + vlq_put(ln) + data, None
        elif st in (0xF0, 0xF7):
            ln, p = vlq_read(trk, p)
            data = trk[p:p + ln]
            p += ln
            yield tick, bytes([st]) + vlq_put(ln) + data, None
        else:
            hi = st & 0xF0
            n = 1 if hi in (0xC0, 0xD0) else 2
            data = trk[p:p + n]
            p += n
            yield tick, bytes([st]) + data, (st & 0x0F, hi, data)


def isolate(path, keep):
    """Rewrite the file with only channel `keep`'s channel events.

    Every dropped event's delta time is CARRIED onto the next surviving
    event rather than discarded. Dropping the delta too would pull the whole
    channel earlier, and then an isolated render would disagree with the
    full one for a reason that has nothing to do with the bank. Meta and
    sysex events are all kept, so the tempo map and the time axis are the
    song's own.
    """
    fmt, div, tracks = read_midi(path)
    out_tracks = []
    for trk in tracks:
        buf = bytearray()
        carry = 0
        prev = 0
        for tick, raw, ch in walk(trk):
            d = tick - prev
            prev = tick
            if ch is not None and ch[0] != keep:
                carry += d
                continue
            buf += vlq_put(d + carry)
            carry = 0
            buf += raw
        out_tracks.append(bytes(buf))
    hdr = b'MThd' + struct.pack('>IHHH', 6, fmt, len(out_tracks), div)
    body = b''.join(b'MTrk' + struct.pack('>I', len(t)) + t for t in out_tracks)
    return hdr + body


def tempo_map(tracks):
    """(abs_tick, usec_per_quarter) sorted by tick. Divisions are assumed
    to be ticks-per-quarter, which is what every file we render uses."""
    ev = []
    for trk in tracks:
        for tick, raw, ch in walk(trk):
            if ch is None and raw[0] == 0xFF and raw[1] == 0x51:
                ln = raw[2]
                usec = int.from_bytes(raw[3:3 + ln], 'big')
                ev.append((tick, usec))
    ev.sort()
    return ev


def tick_to_sec(tick, div, tmap):
    sec = 0.0
    prev_tick = 0
    usec = 500000
    for t, u in tmap:
        if t >= tick:
            break
        sec += (t - prev_tick) * usec / 1e6 / div
        prev_tick = t
        usec = u
    sec += (tick - prev_tick) * usec / 1e6 / div
    return sec


def note_windows(tracks, div, keep):
    """Windows of one channel's notes, in RENDER time, plus the song-time
    span they came from.

    They are returned SHIFTED TO THE FIRST NOTE-ON because that is where the
    renderer puts t=0: lz_smf_next subtracts r->lead, the file's first
    note-on (common/smf.c), so a piece written with its first sound at
    song-time 5 s renders starting at 0.00 s - measured, not assumed, with
    a one-note file at tick 3840. Isolating a channel therefore slides that
    channel to zero, and windows taken in song time land outside the render
    and measure nothing at all.

    The unshifted span is returned too, so the caller can check the render
    is the length this shift implies before believing a number.
    """
    tmap = tempo_map(tracks)
    spans = []
    live = {}
    for trk in tracks:
        for tick, raw, ch in walk(trk):
            if ch is None or ch[0] != keep:
                continue
            hi, data = ch[1], ch[2]
            if hi == 0x90 and len(data) == 2 and data[1] > 0:
                live.setdefault(data[0], []).append(tick)
            elif hi in (0x80,) or (hi == 0x90 and len(data) == 2 and data[1] == 0):
                st = live.get(data[0])
                if st:
                    t0 = st.pop(0)
                    spans.append((tick_to_sec(t0, div, tmap),
                                  tick_to_sec(tick, div, tmap)))
    for rest in live.values():
        for t0 in rest:
            spans.append((tick_to_sec(t0, div, tmap),
                          tick_to_sec(t0 + div, div, tmap)))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    if not merged:
        return [], 0.0, 0.0
    zero = merged[0][0]
    return [(a - zero, b - zero) for a, b in merged], zero, merged[-1][1]


def load_mono(path):
    w = wave.open(path, 'rb')
    n, fr, c = w.getnframes(), w.getframerate(), w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float64)
    w.close()
    if c > 1:
        x = x.reshape(-1, c).mean(axis=1)
    return x, fr


def rms_over(x, fr, windows):
    """RMS of the frames inside the windows, counted once each."""
    tot = 0.0
    cnt = 0
    for a, b in windows:
        i0 = max(0, int(a * fr))
        i1 = min(len(x), int(b * fr))
        if i1 <= i0:
            continue
        seg = x[i0:i1]
        tot += float((seg * seg).sum())
        cnt += seg.size
    return (tot / cnt) ** 0.5 if cnt else 0.0


def render(bank, midi, dst):
    r = subprocess.run([PLAY, '--bank', bank, '--multi', '--wav', dst, midi],
                       capture_output=True, text=True, errors='replace')
    if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
        raise RuntimeError('render failed: %s\n%s' % (dst, r.stderr[-2000:]))


def display_name(path):
    """A name for this song that is safe to write into a filename or a
    table row.

    THE TEMP FILES ARE NAMED AFTER THE SONG, AND SOME SONG NAMES ARE NOT
    FILENAMES. Three pieces under corpus/lakh carry U+F03F - a macOS
    AppleDouble marker that leaked into the name - and the player could
    not create the render whose path contained it, so the piece measured
    as a render failure with nothing wrong with its audio. The same names
    arrive as surrogates through glob and cannot be written to a text
    table either. Both consumers need the same answer or the two arms of
    a comparison end up keyed on different strings and the rows silently
    fail to match, so there is one implementation of it.
    """
    raw = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r'[^A-Za-z0-9._-]', '_', raw)
    if stem != raw:
        stem = '%s_%04x' % (stem,
                            zlib.crc32(raw.encode('utf-8', 'replace')) & 0xffff)
    return stem


def main():
    # THIS TOOL PRINTS FILENAMES, AND SOME OF THEM ARE NOT ENCODABLE IN THE
    # CONSOLE CODE PAGE. Three corpus files carry U+F03F - a macOS
    # AppleDouble marker that leaked into the name - and a GBK encoder
    # raised on the header line after the render had already been measured.
    # An encoding failure is not a measurement failure and must not be
    # indistinguishable from one.
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    args = sys.argv[1:]
    song = None
    bank = os.path.join(ROOT, 'banks', 'picked.ksb')
    ref = os.path.join(ROOT, 'banks', 'gm.ksb')
    wavdir = os.path.join(ROOT, 'build', 'tmp')
    chans = None
    keep_files = True
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--bank':
            i += 1; bank = args[i]
        elif a == '--ref':
            i += 1; ref = args[i]
        elif a == '--wav-dir':
            i += 1; wavdir = args[i]
        elif a == '--chan':
            i += 1; chans = [int(v) for v in args[i].split(',')]
        elif a == '--keep':
            i += 1; keep_files = args[i] != '0'
        elif a.startswith('--'):
            raise SystemExit('unknown option ' + a)
        else:
            song = a
        i += 1
    if song is None:
        raise SystemExit('usage: chan_level.py SONG.mid [--bank B] [--ref B]')

    for p in (bank, ref, PLAY):
        if not os.path.exists(p):
            raise SystemExit('missing ' + p)
    fmt, div, tracks = read_midi(song)
    stem = display_name(song)
    want = sorted({ch[0] for trk in tracks for _, _, ch in walk(trk) if ch})
    if chans:
        want = [c for c in want if c in chans]
    if not want:
        raise SystemExit('no channels found')

    print('%s  (%d tracks, div %d, %d channels)'
          % (os.path.basename(song), len(tracks), div, len(want)))
    print('%-5s %-9s %9s %9s %9s   %s'
          % ('ch', 'program', 'ours dB', 'ref dB', 'diff', 'name'))
    rows = []
    for c in want:
        # First program change in TIME, not in track order: a part that
        # switches sound half way is named by the one it starts on.
        prog = None
        prog_tick = None
        for trk in tracks:
            for tick, _, ch in walk(trk):
                if ch and ch[0] == c and ch[1] == 0xC0:
                    if prog_tick is None or tick < prog_tick:
                        prog, prog_tick = ch[2][0], tick
        iso = isolate(song, c)
        tmp_mid = os.path.join(wavdir, '_chan_%s_c%02d.mid' % (stem, c))
        with open(tmp_mid, 'wb') as f:
            f.write(iso)
        _, idiv, itracks = read_midi(tmp_mid)
        win, zero, last = note_windows(itracks, idiv, c)
        if not win:
            print('%-5d %-9s   (no notes)  %s'
                  % (c, prog if prog is not None else '-', gname(prog)))
            os.remove(tmp_mid)
            continue
        outs = {}
        for tag, bp in (('ours', bank), ('ref', ref)):
            dst = os.path.join(wavdir, '%s_c%02d_%s.wav' % (stem, c, tag))
            render(bp, tmp_mid, dst)
            x, fr = load_mono(dst)
            got = len(x) / float(fr)
            # The shift is only trustworthy if the render is the note span
            # followed by a release tail - the loop stops at PLAY_TAIL_S
            # (5 s) of it. So the bound is ONE-SIDED: a render shorter than
            # the notes means the origin is somewhere else and the windows
            # are measuring silence. Longer is fine, that is the tail.
            span = last - zero
            if got < span - 0.5 or got > span + 7.0:
                print('  !! ch %d render %.2f s but notes span %.2f s '
                      '- origin is wrong, skipping' % (c, got, span))
                outs = None
                break
            outs[tag] = rms_over(x, fr, win)
            if not keep_files:
                os.remove(dst)
        if outs is None:
            os.remove(tmp_mid)
            continue
        os.remove(tmp_mid)
        if outs['ours'] <= 0 or outs['ref'] <= 0:
            dead = 'silent in both' if outs['ours'] <= 0 and outs['ref'] <= 0 else \
                   ('silent in ours' if outs['ours'] <= 0 else 'silent in ref')
            print('%-5d %-9s   (%s)  %s'
                  % (c, prog if prog is not None else '-', dead, gname(prog)))
            continue
        d = DB(outs['ours']) - DB(outs['ref'])
        rows.append((c, prog, DB(outs['ours']), DB(outs['ref']), d))
        print('%-5d %-9s %9.1f %9.1f %+9.2f   %s'
              % (c, prog if prog is not None else '-',
                 DB(outs['ours']), DB(outs['ref']), d, gname(prog)))
    print()
    ds = sorted(r[4] for r in rows)
    if not ds:
        # NO CHANNEL MEASURED IS A RESULT, NOT A STACK TRACE. Every channel
        # can be turned away by the render-length guard above - FURELISE's
        # last two notes are played at velocity 30 and the player ends the
        # render on its silence rule before their note-off - and indexing
        # an empty list here turned that into a traceback. The caller needs
        # to see "nothing measured" and say so, not guess from a crash.
        print('no channel measured (%d seen, all skipped)' % len(want))
        return 1
    mid = ds[len(ds) // 2]
    print('spread %.2f dB   median %+.2f   min %+.2f (ch %d)   max %+.2f (ch %d)'
          % (max(ds) - min(ds), mid,
             min(rows, key=lambda r: r[4])[4], min(rows, key=lambda r: r[4])[0],
             max(rows, key=lambda r: r[4])[4], max(rows, key=lambda r: r[4])[0]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
