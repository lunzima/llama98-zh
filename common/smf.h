#ifndef LZ_COMMON_SMF_H
#define LZ_COMMON_SMF_H

/* Standard MIDI File -> short messages, one at a time, with no Win32
 * and no device in it. Kept: anything that fits in a three-byte channel
 * message, which is midiStreamOut's whole MEVT_SHORTMSG vocabulary.
 * Meta events are read for the tempo map and dropped; sysex is skipped.
 *
 * STREAMED, not parsed into an array. The array version had a ceiling -
 * so many events and no more - and raising it is just a bigger number:
 * 9% of the MIDI on this machine was over it, the largest by a factor
 * of five. A reader that holds ONE CURSOR PER TRACK has no ceiling at
 * all. Memory is O(tracks); surveyed, no file here has more than 13.
 *
 * The tracks are merged exactly the way the old stable sort merged
 * them: the live track with the smallest tick wins, and a tie goes to
 * the LOWER TRACK INDEX. That matters - a note-off and the next
 * note-on of the same pitch land on one tick, and the off has to stay
 * first or the note is silenced.
 *
 * THE TEMPO TRICK, which is why the file is shaped this way.
 *
 * The caller changes the tempo WHILE the piece plays, and the whole of
 * midiStreamOut's tempo vocabulary is MEVT_TEMPO - the event whose
 * parameter is microseconds per quarter note. The arrangements carry
 * tempo MAPS (six steps in the Fur Elise setting), and a map put into
 * the stream as MEVT_TEMPO events is the SAME event the caller's ratio
 * is: every step of the map would overwrite the ratio in force, and the
 * ratio would have to be re-emitted after each one, into a stream that
 * is being produced one event at a time.
 *
 * So the map is folded into the CLOCK. A delta of d ticks under tempo
 * T_i lasts d*T_i/division us; the same duration under one constant
 * tempo T0 needs d*T_i/T0 ticks. Scale every position that way and one
 * constant tempo reproduces the map exactly, which leaves the piece
 * with ONE tempo for the whole of it - and the caller's ratio is that
 * one number, emitted into the stream ahead of the chunk it governs.
 *
 * Positions are scaled ABSOLUTELY, so the rounding is one tick per
 * event rather than an error that accumulates over three minutes.
 *
 * The same scaling carries the file's DIVISION to this one. Surveyed
 * over 141 MIDI files: 72 are at 480, 36 at 192 - and those 36 are the
 * ones this project wrote - with 24, 96, 120, 240, 384, 960 and 1024
 * making up the rest. A player that took only its own division would
 * refuse almost every file a user dropped in, silently. Ticks out of
 * here are ALWAYS in LZ_SMF_DIVISION. */

/* The division every stream comes out in, so the device can be opened
 * once and keep it. */
#define LZ_SMF_DIVISION  192

/* ---- how much fits in one stream buffer ----
 *
 * COMPUTED from the two documented facts, not rounded off a
 * measurement.
 *
 * midiStreamOut's Remarks - the body, not the requirements table:
 * "For the current implementation of this function, the buffer must be
 * smaller than 64K." SMALLER THAN, so the largest legal length is
 * 65535 and not 65536.
 *
 * MIDIEVENT is dwDeltaTime, dwStreamID, dwEvent and a flexible
 * dwParms[], and that page says of dwParms: "If dwEvent specifies
 * MEVT_F_SHORT, do not use this member in the stream buffer."
 * Everything emitted here is MEVT_SHORTMSG, so an event is exactly
 * three DWORDs. Not sizeof(MIDIEVENT) - a compiler may pad the
 * flexible member, and the player cannot name the structure anyway.
 *
 * 65535/12 = 5461, using 65532 bytes; a 5462nd event would need 65544
 * and be refused. Measured against a real device for agreement: 5461
 * accepted, 6000 refused. The limit is enforced at PREPARE time, so
 * lz_ui_midi_send reports it rather than the stream failing later. */
#define LZ_SMF_BUF_MAX      65535
#define LZ_SMF_EVENT_BYTES     12

/* Overridable so a probe can shrink it and make the chunk joins happen
 * every second instead of every minute: a join that only occurs after
 * a minute of playing is one nobody ever exercises. */
#ifndef LZ_SMF_CHUNK_EVENTS
#define LZ_SMF_CHUNK_EVENTS (LZ_SMF_BUF_MAX / LZ_SMF_EVENT_BYTES)
#endif

/* The tempo map is one flat array and 512 steps is its ceiling. A file
 * with more is REFUSED with LZ_SMF_TOO_BIG rather than played with a
 * truncated map: a tempo step that is dropped moves every event after
 * it in time, which is a piece playing wrong rather than playing less.
 * The reader cannot represent such a file, so it says so.
 *
 * 64 WAS TOO LOW AND REFUSED REAL FILES. A survey of 13 MIDIs from the
 * xqwizard collection: five carried more tempo events than the old cap
 * - 79 (Fuer Elise), 156, 233, 501 - and none of them was corrupt. A
 * tempo event that restates the value already in force changes
 * nothing, so both collection and the sorted map drop those first
 * (MARIO1's 501 events carry 10 distinct values), and the cap applies
 * to what is left. */
#define LZ_SMF_MAX_TEMPI    512

/* Surveyed: 13 is the most any file here uses. 64 is room for four
 * times that; a file with more has its extra tracks REFUSED rather
 * than played without them, because a piece missing a part is a piece
 * playing wrong, not a piece playing less. */
#define LZ_SMF_MAX_TRACKS   64

typedef struct {
    unsigned long tick;   /* absolute, already scaled and trimmed */
    unsigned long msg;    /* status | data1<<8 | data2<<16 */
} LZSmfEvent;

typedef struct {
    unsigned long tick;   /* absolute, in the FILE's own ticks */
    unsigned long us_qn;  /* microseconds per quarter note */
} LZSmfTempo;

/* `tm[]` in LZSmfReader is SORTED BY TICK once lz_smf_open returns -
 * tempo events arrive track by track and lz_smf_tempo_at, scale_tick
 * both walk the array as a monotonic map. Steps that share a tick keep
 * the order they were collected in, which is the lower track index
 * first. */

/* One track's position. The pending event is DECODED and waiting: the
 * merge has to compare ticks before it knows which track to take
 * from, so each cursor always holds the next event it would give. */
typedef struct {
    const unsigned char *p, *end, *begin;
    unsigned long tick;   /* the FILE tick of the pending event */
    unsigned long msg;
    int status;           /* running status */
    int live;             /* 0 once the track is spent */
    /* The last tempo THIS TRACK stated, so a restatement of the value
       already in force is not stored a second time. A file that writes
       the tempo on every beat is common and says nothing new. */
    unsigned long last_us;
    int have_us;
} LZSmfTrack;

typedef struct {
    int  division;        /* the FILE's own; ticks come out in
                             LZ_SMF_DIVISION whatever this says */
    int  ntrk;
    int  n_tempi;
    unsigned long us_qn0; /* the reference tempo T0 the scaling uses */
    unsigned long lead;   /* file ticks before the first sounding note */
    int  instrument;      /* forced GM program, or LZ_MUSIC_AS_WRITTEN */
    int  volume;          /* 0..127, sent as controller 7 */
    int  prefix;          /* how much of the opening prefix has gone */
    /* The most ticks one chunk may cover, 0 for no bound at all. The
       caller sets it before each pull, from the tempo that chunk will
       play at; lz_smf_chunk's own comment says why the reader is where
       the bound has to be enforced. */
    unsigned long span;
    /* An event that was pulled for a chunk with no room left in it,
       waiting to be the first event of the next one. One event and not
       a buffer: the reader only goes forwards, so an event that came
       out to be measured has to be kept rather than put back. */
    LZSmfEvent pend;
    int  have_pend;
    /* 1 once lz_smf_next has answered 0 - the reader has no events
       left. Cleared by lz_smf_rewind. The caller cannot work this out
       from a chunk's own return: a chunk that stopped on `span` or on
       a full buffer says nothing about what was behind it. */
    int  spent;
    LZSmfTempo tm[LZ_SMF_MAX_TEMPI];
    LZSmfTrack trk[LZ_SMF_MAX_TRACKS];
} LZSmfReader;

#define LZ_SMF_OK        0
#define LZ_SMF_BAD       1   /* not an SMF, or a truncated one */
#define LZ_SMF_TOO_BIG   2   /* more tracks or tempi than there is room */
#define LZ_SMF_SMPTE     3   /* SMPTE time division, which this cannot use */
#define LZ_SMF_EMPTY     4   /* opened, read, and holds no notes */

/* Read the header and the tempo map, find the tracks, and leave the
 * reader ready to give its first event. `img` must outlive the reader:
 * nothing is copied and every cursor points into it.
 *
 *   instrument  a GM program forced onto every channel but 10 (drums,
 *               where a program change selects a KIT), or
 *               LZ_MUSIC_AS_WRITTEN to leave the file's own alone
 *   volume      0..127, sent as controller 7 on every channel before
 *               the first note - a volume that arrives after it is a
 *               first note at the wrong volume
 */
int lz_smf_open(LZSmfReader *r, const unsigned char *img, long size,
                int instrument, int volume);

/* Back to the beginning, for the loop. Cheap: it re-seeks the cursors,
 * it does not re-read anything. */
void lz_smf_rewind(LZSmfReader *r);

/* The next event, or 0 when the piece is over. */
int lz_smf_next(LZSmfReader *r, LZSmfEvent *out);

/* The program this file opens on: its first program change on a channel that
 * is not the drum kit, or GM program 0 when it never states one - which is the
 * format's own default.
 *
 * An instrument that is decided before the first note needs this, because
 * AS_WRITTEN is a decision to take the program from the file and the file
 * states it wherever it feels like, usually at tick 0 and sometimes not at
 * all. `scratch` is used and left open on `img`; the caller owns it. */
int lz_smf_opening_program(LZSmfReader *scratch, const unsigned char *img,
                           long size);

/* The tempo in force at a file tick. Exposed so a gate can ask it.
 * `tm` must be sorted by tick the way lz_smf_open leaves it; the walk
 * stops at the first step past `tick`, so an unsorted array would
 * answer with whichever step happened to follow. */
unsigned long lz_smf_tempo_at(const LZSmfTempo *tm, int n,
                              unsigned long tick);

/* Fill `out` with up to `max` events as MIDIEVENT triples - delta,
 * stream id, event - and return how many EVENTS were written.
 *
 * `*prev` is the tick the last chunk ended on and is updated, which is
 * what makes a join not a gap: the first delta of a chunk is measured
 * from the last event of the one before it, not from zero.
 * lz_smf_rewind and a *prev of 0 together are the loop point.
 *
 * `r->span` bounds a chunk by TIME as well as by size: the chunk ends
 * once the next event is more than `span` ticks past the chunk's own
 * first one. That bound has to be enforced HERE rather than by the
 * caller, because an event is only known to be past the bound after it
 * has been pulled - and there is no way to give it back. One that
 * crosses is therefore held in the reader and comes out as the first
 * event of the next chunk; nothing else is held.
 *
 * `r->spent` then reports whether there are events left at all. */
int lz_smf_chunk(LZSmfReader *r, unsigned long *out, int max,
                 unsigned long *prev);

#endif /* LZ_COMMON_SMF_H */
