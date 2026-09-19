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
 * in track 2 governs events track 1 already produced. */
static void track_advance(LZSmfTrack *t, LZSmfTempo *tm, int *ntm,
                          unsigned long *first_note) {
    while (t->live) {
        int c, hi, d1, d2 = 0, two;
        t->tick += vlq(&t->p, t->end);
        if (t->p >= t->end) { t->live = 0; return; }
        c = *t->p;
        if (c & 0x80) { t->status = c; t->p++; }
        /* else: running status, and `status` still holds it. */
        if (t->status == 0xFF) {
            int type;
            unsigned long n;
            if (t->p >= t->end) { t->live = 0; return; }
            type = *t->p++;
            n = vlq(&t->p, t->end);
            if (n > (unsigned long)(t->end - t->p))
                n = (unsigned long)(t->end - t->p);
            if (type == 0x51 && n == 3 && tm && ntm &&
                *ntm < LZ_SMF_MAX_TEMPI) {
                tm[*ntm].tick = t->tick;
                tm[*ntm].us_qn = ((unsigned long)t->p[0] << 16) |
                                 ((unsigned long)t->p[1] << 8) | t->p[2];
                /* A zero would divide by zero in the scaling and is not
                   a tempo any file means. */
                if (tm[*ntm].us_qn == 0) tm[*ntm].us_qn = 500000;
                (*ntm)++;
            }
            t->p += n;
            continue;
        }
        if (t->status == 0xF0 || t->status == 0xF7) {
            unsigned long n = vlq(&t->p, t->end);
            if (n > (unsigned long)(t->end - t->p))
                n = (unsigned long)(t->end - t->p);
            t->p += n;
            continue;
        }
        hi = t->status & 0xF0;
        if (hi < 0x80) { t->p++; continue; }   /* no status yet */
        two = (hi != 0xC0 && hi != 0xD0);
        if (t->p >= t->end) { t->live = 0; return; }
        d1 = *t->p++;
        if (two) {
            if (t->p >= t->end) { t->live = 0; return; }
            d2 = *t->p++;
        }
        t->msg = (unsigned long)t->status |
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
        return;
    }
}

static void tracks_seek(LZSmfReader *r) {
    int i;
    for (i = 0; i < r->ntrk; i++) {
        r->trk[i].p = r->trk[i].begin;
        r->trk[i].tick = 0;
        r->trk[i].msg = 0;
        r->trk[i].status = 0;
        r->trk[i].live = 1;
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
        while (t->live) track_advance(t, r->tm, &r->n_tempi, &first);
    }
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
    /* Every cursor holds its first event before the merge starts. */
    for (i = 0; i < r->ntrk; i++)
        track_advance(&r->trk[i], NULL, NULL, NULL);
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
    track_advance(t, NULL, NULL, NULL);
    return 1;
}

int lz_smf_chunk(LZSmfReader *r, unsigned long *out, int max,
                 unsigned long *prev) {
    LZSmfEvent e;
    int n = 0;
    if (!r || !out || !prev || max <= 0) return 0;
    while (n < max && lz_smf_next(r, &e)) {
        out[n * 3 + 0] = e.tick - *prev;
        out[n * 3 + 1] = 0;   /* stream id, always 0 here */
        out[n * 3 + 2] = e.msg;
        *prev = e.tick;
        n++;
    }
    return n;
}
