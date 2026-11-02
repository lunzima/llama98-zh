/* Standard MIDI File reading. See smf.h, especially the tempo trick. */
#include "smf.h"
#include "music.h"

#include <string.h>

/* ---- reading the image ----
 *
 * A MIDI file is something a user dropped in a directory, so every read
 * is bounds-checked and a truncated one is ordinary input. */

static unsigned long be16(const unsigned char *p) {
    return ((unsigned long)p[0] << 8) | p[1];
}

static unsigned long be32(const unsigned char *p) {
    return ((unsigned long)p[0] << 24) | ((unsigned long)p[1] << 16) |
           ((unsigned long)p[2] << 8) | p[3];
}

/* A variable-length quantity. Advances *p; parks it at `end` and
 * returns 0 past the end.
 *
 * Capped at four bytes, the format's own limit: without it a run of
 * 0x80 - what a misaligned track looks like - scans the whole file. */
static unsigned long vlq(const unsigned char **p, const unsigned char *end) {
    unsigned long v = 0;
    int i;
    for (i = 0; i < 4; i++) {
        unsigned char c;
        if (*p >= end) { *p = end; return 0; }
        c = *(*p)++;
        v = (v << 7) | (unsigned long)(c & 0x7F);
        if (!(c & 0x80)) return v;
    }
    return v;
}

/* ---- the tempo map ---- */

unsigned long lz_smf_tempo_at(const LZSmfTempo *tm, int n,
                              unsigned long tick) {
    unsigned long v = 500000;   /* the format's own default: 120 bpm */
    int i;
    if (!tm) return v;
    for (i = 0; i < n; i++) {
        /* The step IN FORCE at `tick`, so a step starting exactly there
           counts: a tempo change and the note it was written for share
           a tick, and the note belongs to the new tempo. */
        if (tm[i].tick > tick) break;
        v = tm[i].us_qn;
    }
    return v;
}

/* The scaled position of a file tick - the tempo trick and the
 * division conversion, in one rounding. */
static unsigned long scale_tick(const LZSmfReader *r, unsigned long tick) {
    double acc = 0.0;
    unsigned long prev = 0, cur = 500000;
    int i;
    if (r->us_qn0 == 0 || r->division <= 0) return tick;
    for (i = 0; i < r->n_tempi && r->tm[i].tick < tick; i++) {
        acc += (double)(r->tm[i].tick - prev) * (double)cur;
        prev = r->tm[i].tick;
        cur = r->tm[i].us_qn;
    }
    acc += (double)(tick - prev) * (double)cur;
    acc = acc * (double)LZ_SMF_DIVISION /
          ((double)r->us_qn0 * (double)r->division);
    if (acc < 0.0) return 0;
    return (unsigned long)(acc + 0.5);
}

/* ---- one track's cursor ----
 *
 * Decodes forward to the next CHANNEL message, accumulating the deltas
 * of everything it steps over: a meta event still takes time even
 * though nothing is emitted for it. Leaves the event pending in the
 * cursor, because the merge has to compare ticks before it can choose.
 *
 * `tm`/`ntm` are filled on the first pass and NULL on the second - the
 * map has to be complete before anything is scaled, and a tempo change
 * in track 2 governs events track 1 already produced.
 *
 * RUNNING STATUS IS ONLY THE CHANNEL STATUS. A status byte 0x80..0xEF
 * replaces it; meta (0xFF) and sysex (0xF0/0xF7) are decoded as the
 * events they are and leave it standing, so a channel event after one
 * of them may still omit its status byte. The stored value is never
 * 0xFF/0xF0/0xF7, which is what keeps such a data byte from being read
 * as a two-parameter message under a status that has no parameters.
 *
 * Returns 1 when a tempo event did not fit in the map. The caller on
 * the tempo pass turns that into LZ_SMF_TOO_BIG: a map that lost a
 * step would move every later event in time, and the reader cannot
 * represent the file rather than merely choosing not to. */
static int track_advance(LZSmfTrack *t, LZSmfTempo *tm, int *ntm,
                         unsigned long *first_note) {
    while (t->live) {
        int c, hi, d1, d2 = 0, two, st;
        t->tick += vlq(&t->p, t->end);
        if (t->p >= t->end) { t->live = 0; return 0; }
        c = *t->p;
        if (c & 0x80) {
            t->p++;
            if (c == 0xFF) {
                int type;
                unsigned long n;
                if (t->p >= t->end) { t->live = 0; return 0; }
                type = *t->p++;
                n = vlq(&t->p, t->end);
                if (n > (unsigned long)(t->end - t->p))
                    n = (unsigned long)(t->end - t->p);
                if (type == 0x51 && n == 3 && tm && ntm) {
                    unsigned long us = ((unsigned long)t->p[0] << 16) |
                                       ((unsigned long)t->p[1] << 8) | t->p[2];
                    /* A zero would divide by zero in the scaling and is
                       not a tempo any file means. */
                    if (us == 0) us = 500000;
                    /* A RESTATED TEMPO IS NOT A STEP. Notation software
                       writes the value in force on every beat; storing
                       those spends the map's room on nothing, and the
                       xqwizard files that overflowed the old cap were
                       almost all this shape. */
                    if (!t->have_us || us != t->last_us) {
                        if (*ntm >= LZ_SMF_MAX_TEMPI) return 1;
                        tm[*ntm].tick = t->tick;
                        tm[*ntm].us_qn = us;
                        (*ntm)++;
                        t->last_us = us;
                        t->have_us = 1;
                    }
                }
                t->p += n;
                continue;
            }
            if (c == 0xF0 || c == 0xF7) {
                unsigned long n = vlq(&t->p, t->end);
                if (n > (unsigned long)(t->end - t->p))
                    n = (unsigned long)(t->end - t->p);
                t->p += n;
                continue;
            }
            if (c >= 0xF0) {
                /* System common and real-time bytes do not occur in a
                   track and none of them carries running status; skip
                   the status byte alone. */
                continue;
            }
            /* A channel status, and the only thing running status
               stores. */
            t->status = c;
            st = c;
        } else {
            /* Running status, and `status` still holds it - including
               across a meta or sysex event. */
            st = t->status;
        }
        hi = st & 0xF0;
        if (hi < 0x80) { t->p++; continue; }   /* no status yet */
        two = (hi != 0xC0 && hi != 0xD0);
        if (t->p >= t->end) { t->live = 0; return 0; }
        d1 = *t->p++;
        if (two) {
            if (t->p >= t->end) { t->live = 0; return 0; }
            d2 = *t->p++;
        }
        t->msg = (unsigned long)st |
                 ((unsigned long)(d1 & 0x7F) << 8) |
                 ((unsigned long)(d2 & 0x7F) << 16);
        /* A note-on with velocity ZERO is a note-off in every file that
           uses running status, which is most of them. Taking one as the
           first sound would trim the lead-in to a release.

           SMALLEST TICK SEEN, not merely the first one seen: the caller
           runs this across every track in turn, each to completion,
           rather than merging them by tick the way lz_smf_next does -
           so the first note-on this ever sees is whichever track sits
           at the lowest INDEX and has any note at all, not whichever
           note sounds earliest in the piece. A part written on a later
           track that leads the piece in - the shape of an intro carried
           on a channel other than the first - would win the update race
           against ITS OWN earlier note once some other track had
           already set *first_note, since the old `== -1` guard only let
           the value be written once. */
        if (first_note && hi == 0x90 && (d2 & 0x7F) != 0 &&
            (*first_note == (unsigned long)-1 || t->tick < *first_note))
            *first_note = t->tick;
        return 0;
    }
    return 0;
}

/* The map is collected track by track, and a tempo in track 2 governs
 * events track 1 produced, so it has to be in TICK order before
 * lz_smf_tempo_at or scale_tick can walk it. Insertion sort: stable,
 * which keeps equal ticks in the order they were collected - the lower
 * track index first - and needs no scratch memory. n is at most
 * LZ_SMF_MAX_TEMPI. */
static void tempo_sort(LZSmfTempo *tm, int n) {
    int i;
    for (i = 1; i < n; i++) {
        LZSmfTempo key = tm[i];
        int j = i - 1;
        while (j >= 0 && tm[j].tick > key.tick) {
            tm[j + 1] = tm[j];
            j--;
        }
        tm[j + 1] = key;
    }
}

/* The same rule across tracks as within one. The collection pass drops
   a restatement inside a single track; the sorted map then holds the
   union of every track's tempos, where a conductor track and a part
   that both state the tempo leave two entries saying one thing. An
   entry whose value equals the one already in force is a no-op and is
   removed. Equal ticks keep the order of collection, so the last value
   at a tick - the one lz_smf_tempo_at already used - survives. */
static int tempo_compact(LZSmfTempo *tm, int n) {
    int i, k = 0;
    for (i = 0; i < n; i++) {
        if (k > 0 && tm[i].us_qn == tm[k - 1].us_qn) continue;
        tm[k++] = tm[i];
    }
    return k;
}

static void tracks_seek(LZSmfReader *r) {
    int i;
    for (i = 0; i < r->ntrk; i++) {
        r->trk[i].p = r->trk[i].begin;
        r->trk[i].tick = 0;
        r->trk[i].msg = 0;
        r->trk[i].status = 0;
        r->trk[i].live = 1;
        r->trk[i].last_us = 0;
        r->trk[i].have_us = 0;
    }
}

int lz_smf_open(LZSmfReader *r, const unsigned char *img, long size,
                int instrument, int volume) {
    const unsigned char *p, *end;
    unsigned long first = (unsigned long)-1;
    int ntrk, div, i;

    if (!r || !img || size < 22) return LZ_SMF_BAD;
    if (memcmp(img, "MThd", 4) != 0) return LZ_SMF_BAD;
    if (be32(img + 4) < 6) return LZ_SMF_BAD;
    ntrk = (int)be16(img + 10);
    div = (int)be16(img + 12);
    /* Bit 15 means SMPTE: the unit is frames, not quarter notes, so the
       tempo trick has nothing to divide by. Refused, not misread. */
    if (div & 0x8000) return LZ_SMF_SMPTE;
    if (div <= 0) return LZ_SMF_BAD;
    if (ntrk > LZ_SMF_MAX_TRACKS) return LZ_SMF_TOO_BIG;

    memset(r, 0, sizeof *r);
    r->division = div;
    r->instrument = instrument;
    r->volume = volume < 0 ? 0
              : (volume > LZ_MUSIC_VOL_MAX ? LZ_MUSIC_VOL_MAX : volume);

    end = img + size;
    p = img + 8 + be32(img + 4);
    for (i = 0; i < ntrk && p + 8 <= end; i++) {
        unsigned long len;
        const unsigned char *tend;
        if (memcmp(p, "MTrk", 4) != 0) break;
        len = be32(p + 4);
        tend = p + 8 + len;
        if (tend > end) tend = end;   /* truncated: read what is there */
        r->trk[r->ntrk].begin = p + 8;
        r->trk[r->ntrk].end = tend;
        r->ntrk++;
        p = tend;
    }
    if (!r->ntrk) return LZ_SMF_BAD;

    /* PASS 1: the tempo map and the first sounding note, both of which
       every later position depends on. Cheap - it decodes, emits
       nothing, and keeps no events. */
    tracks_seek(r);
    for (i = 0; i < r->ntrk; i++) {
        LZSmfTrack *t = &r->trk[i];
        while (t->live)
            if (track_advance(t, r->tm, &r->n_tempi, &first))
                return LZ_SMF_TOO_BIG;
    }
    tempo_sort(r->tm, r->n_tempi);
    r->n_tempi = tempo_compact(r->tm, r->n_tempi);
    if (first == (unsigned long)-1) return LZ_SMF_EMPTY;

    /* T0 is the OPENING tempo. Any constant reproduces the map; this
       one also makes ratio 1.0 the speed the piece was written at. */
    r->us_qn0 = lz_smf_tempo_at(r->tm, r->n_tempi, 0);
    /* The lead-in comes off every position. The pieces carry up to six
       beats of it because a player opening a file cold can swallow a
       note at tick 0; a stream written into a device that is already
       open needs none of it. */
    r->lead = first;
    lz_smf_rewind(r);
    return LZ_SMF_OK;
}

void lz_smf_rewind(LZSmfReader *r) {
    int i;
    if (!r) return;
    tracks_seek(r);
    r->prefix = 0;
    /* The held event belonged to the pass that just ended, and the
       loop restarts from tick 0. */
    r->have_pend = 0;
    r->spent = 0;
    /* Every cursor holds its first event before the merge starts. The
       map is complete and sorted by now, so a tempo event is decoded
       but never collected and the overflow answer cannot arise. */
    for (i = 0; i < r->ntrk; i++)
        (void)track_advance(&r->trk[i], NULL, NULL, NULL);
}

/* The opening prefix: controller 7 on all sixteen channels, then the
 * forced program on the fifteen that are not the drum kit. At tick 0,
 * before anything else, so the first note is at the right volume and
 * on the right patch. */
static int prefix_event(LZSmfReader *r, LZSmfEvent *out) {
    int k = r->prefix;
    if (k < 16) {
        out->tick = 0;
        out->msg = (unsigned long)(0xB0 | k) | ((unsigned long)7 << 8) |
                   ((unsigned long)r->volume << 16);
        r->prefix++;
        return 1;
    }
    if (r->instrument == LZ_MUSIC_AS_WRITTEN) return 0;
    /* prefix is 16..31 here, so ch is 0..15 and needs no upper guard. */
    while (r->prefix < 32) {
        int ch = r->prefix - 16;
        r->prefix++;
        if (ch == 9) continue;          /* the drum kit keeps its own */
        out->tick = 0;
        out->msg = (unsigned long)(0xC0 | ch) |
                   ((unsigned long)(r->instrument & 0x7F) << 8);
        return 1;
    }
    return 0;
}

int lz_smf_next(LZSmfReader *r, LZSmfEvent *out) {
    int i, best = -1;
    LZSmfTrack *t;
    if (!r || !out) return 0;
    if (r->prefix < 32 && prefix_event(r, out)) return 1;

    /* The merge: the live track with the smallest tick, ties to the
       LOWER INDEX. That is what the stable sort this replaced did, and
       it is load-bearing - a note-off and the next note-on of the same
       pitch share a tick, and the off has to stay first. */
    for (i = 0; i < r->ntrk; i++) {
        if (!r->trk[i].live) continue;
        if (best < 0 || r->trk[i].tick < r->trk[best].tick) best = i;
    }
    if (best < 0) return 0;

    t = &r->trk[best];
    out->tick = scale_tick(r, t->tick);
    {
        unsigned long lead = scale_tick(r, r->lead);
        out->tick = out->tick > lead ? out->tick - lead : 0;
    }
    out->msg = t->msg;
    /* The file's own program changes are REWRITTEN, not merely preceded
       by the prefix: a piece that changes patch part way through would
       otherwise obey the menu for one bar and then leave it. */
    if (r->instrument != LZ_MUSIC_AS_WRITTEN &&
        (out->msg & 0xF0) == 0xC0 && (out->msg & 0x0F) != 9)
        out->msg = (out->msg & 0xFFUL) |
                   ((unsigned long)(r->instrument & 0x7F) << 8);
    (void)track_advance(t, NULL, NULL, NULL);
    return 1;
}

int lz_smf_chunk(LZSmfReader *r, unsigned long *out, int max,
                 unsigned long *prev) {
    LZSmfEvent e;
    unsigned long start;
    int n = 0;

    if (!r || !out || !prev || max <= 0) return 0;
    if (r->have_pend) {
        e = r->pend;
        r->have_pend = 0;
    } else if (!lz_smf_next(r, &e)) {
        r->spent = 1;
        return 0;
    }
    /* The chunk's own span is measured from its first event, so the
       bound is on how long the chunk lasts and not on where in the
       piece it happens to fall. */
    start = e.tick;
    for (;;) {
        out[n * 3 + 0] = e.tick - *prev;
        out[n * 3 + 1] = 0;   /* stream id, always 0 here */
        out[n * 3 + 2] = e.msg;
        *prev = e.tick;
        n++;
        if (n >= max) break;
        if (!lz_smf_next(r, &e)) {
            r->spent = 1;
            break;
        }
        if (r->span && e.tick - start > r->span) {
            r->pend = e;
            r->have_pend = 1;
            break;
        }
    }
    return n;
}

int lz_smf_opening_program(LZSmfReader *scratch, const unsigned char *img,
                           long size)
{
    LZSmfEvent ev;
    if (!scratch || !img)
        return 0;
    if (lz_smf_open(scratch, img, size, LZ_MUSIC_AS_WRITTEN, 0) != LZ_SMF_OK)
        return 0;
    while (lz_smf_next(scratch, &ev)) {
        if ((ev.msg & 0xF0UL) == 0xC0UL && (ev.msg & 0x0FUL) != 9)
            return (int)((ev.msg >> 8) & 0x7FUL);
    }
    return 0;
}
