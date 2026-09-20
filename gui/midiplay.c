/* One MIDI file, at a tempo that moves. See midiplay.h. */
#include "midiplay.h"

#include <stdio.h>
#include <stdlib.h>

#include "compat.h"     /* lz_time_ms - the stamps on the trace's lines */
#include "compat40.h"
#include "music.h"
#include "smf.h"

/* The image is allocated to fit the FILE, not to a fixed size - the
 * reader holds cursors into it, so it has to stay resident, and the
 * largest MIDI on this machine is 404 KB against a typical 100. The
 * cap is a refusal for something that is not a piece of music: a
 * megabyte is twice the largest real file seen. */
#define LZ_MIDI_MAX_FILE  (1024L * 1024L)

/* The shortest interval between two `probe` lines - the control reading
 * of the stream's position, taken where the ratio did not move.
 *
 * The tick that calls the follow runs at the stream_ms setting, 100 ms
 * by default, and a piece whose ratio is ramping carries a tempo event
 * in most of the chunks submitted near it. A bound a quarter of that
 * leaves several control samples between two tempo lines, which is what
 * makes the arms comparable, and keeps the file's growth tied to the
 * submission rate rather than to stream_ms when the user has set that
 * short. */
#define LZ_MIDI_PROBE_MIN_US  LZ_I64_C(25000)

static int   g_opened;          /* the device is open */
static int   g_tried;           /* and we have stopped asking */
static int   g_playing;
static int   g_loop;
static int   g_ratio = LZ_MUSIC_RATIO_ONE;
/* Buffers the device still owes a notification for - see
 * lz_music_own_done for the defect these exist to stop. */
static int   g_pending;
static int   g_stale;
static unsigned long g_us_qn0 = 500000;

/* The tempo a ratio asks for, from the piece's own opening tempo. In
 * common/music.c, not here: a gate cannot reach into a translation unit
 * that needs a MIDI device. */
#define tempo_for(r) lz_music_tempo_us(g_us_qn0, (r))

/* ---- the piece, in chunks ----
 *
 * One stream buffer holds 64 KB and a long piece does not fit, so the
 * events are sent a chunk at a time with two chunks in flight: the
 * device always has the next one before it needs the one it is on,
 * which is what makes the joins inaudible.
 *
 * The events are PULLED from the reader as each chunk is filled, never
 * held all at once - see common/smf.h. That is what leaves the piece
 * length unbounded. */
static LZSmfReader g_rd;
static int   g_more;            /* the reader still has events */
static int   g_slot_next;       /* the slot to send the next chunk in */
static unsigned long g_prev_tick;  /* deltas carry across a join */

/* On first use, not declared: a program with the music off should not
 * carry this in its image. Freed only at shutdown - the device reads
 * out of g_buf, and the reader points into g_img, so both have to
 * outlive every playback. */
static unsigned long *g_buf[LZ_UI_MIDI_SLOTS];   /* MIDIEVENT triples */
static unsigned char *g_img;
static long           g_img_cap;

/* ---- the trace ----
 *
 * One line per stream event, and only when the ini named a file for it -
 * see lz_midi_trace_to. OFF is the default and costs one pointer test
 * per event: no file, no formatting, and not even the device calls that
 * ask where the stream is or what the counter reads.
 *
 * The lines exist to answer one question about a piece that is playing
 * badly: what the stream was doing when the speed of it changed. A
 * tempo line therefore carries the stream's position either side of the
 * submission that carried the event, the interval between the two
 * samples, and the value the event was given; a chunk line carries the
 * same bracket around a submission with no event in it, which is the
 * control for the tempo arm; and every line carries the number of
 * buffers the device still owes. */

static FILE *g_trace;
static char  g_piece[MAX_PATH];  /* the piece the lines below are about */
static char  g_line[512];
static char  g_pos0[24];    /* the position before a submission */
static char  g_pos1[24];    /* and after it - two, because one line has both */
static char  g_dpos[24];    /* and the advance between that pair */
static char  g_us0[24];     /* the tempo before it, or `none` */
static int   g_sent;        /* chunks handed over since the piece began */
static int   g_done;        /* and the one the next notification is for */
/* The tempo the stream is running at: the value the MEVT_TEMPO at the
 * head of the last chunk handed over carried, and therefore what every
 * chunk from that one on is played at. ZERO MEANS NONE - no event has
 * been emitted into this stream - and it cannot be confused with a
 * tempo, because lz_music_tempo_us never answers one: it clamps at 1.
 * Cleared when the stream is closed, because the tempo is a property of
 * the stream and goes with it. */
static unsigned long g_us_emit;
/* When the last `probe` line was written, and whether one ever has. The
 * limit is a difference of two readings of the counter the line itself
 * reports, because lz_time_ms is quantised far more coarsely than a
 * bound this short - the trace header's `t IS COARSE` paragraph has the
 * measurement, and it is a reading taken before a bracket either way. */
static lz_i64        g_probe_last;
static int           g_probe_set;

#define tracing() (g_trace != NULL)

static void trace_line(const char *s) {
    /* Stamped with the monotone clock gui/main.c already uses, and
       flushed per line: a trace is read to explain something that has
       already gone wrong, and a buffered tail would be missing exactly
       when it is wanted. */
    fprintf(g_trace, "t=%lu %s\n", (unsigned long)lz_time_ms(), s);
    fflush(g_trace);
}

/* Where the stream is, into `buf`, or the word for "could not be
   asked". A zero in its place would read as a position the stream
   really had, and lz_ui_midi_position's 0 answer is exactly the case
   that means no such position was obtained. */
static void pos_text(char *buf, int cap, int got, unsigned long tick) {
    if (!got) snprintf(buf, (size_t)cap, "unavailable");
    else      snprintf(buf, (size_t)cap, "%lu", tick);
}

/* The three columns every arm of the trace reports, filled from one
   pair of samples: where the stream was before the window, where it was
   after it, and the advance between the two - or the word for an
   advance that could not be worked out, since a difference taken across
   a reading that failed is a number the stream never had. One function
   because every arm's advance has to have been computed the same way
   for the arms to be comparable. */
static void bracket_text(int got0, unsigned long p0,
                         int got1, unsigned long p1) {
    pos_text(g_pos0, (int)sizeof g_pos0, got0, p0);
    pos_text(g_pos1, (int)sizeof g_pos1, got1, p1);
    if (got0 && got1)
        snprintf(g_dpos, sizeof g_dpos, "%ld", (long)(p1 - p0));
    else
        snprintf(g_dpos, sizeof g_dpos, "unavailable");
}

void lz_midi_trace_to(const char *path) {
    long have;
    if (g_trace || !path || !*path) return;
    /* Appended, never truncated: a trace already collected has to
       survive a later start of the program. One run that opens the
       same path - a selftest, say - would otherwise leave the header
       and nothing else, which is indistinguishable from a run in which
       nothing happened. */
    g_trace = fopen(path, "ab");
    if (!g_trace) return;
    fseek(g_trace, 0, SEEK_END);
    have = ftell(g_trace);
    if (have > 0) {
        fputs("# --- new run ---\n", g_trace);
        return;
    }
    /* The header, because a reader who takes these ticks for the file's
       own draws the wrong conclusion from every line under it. Written
       a line at a time rather than formatted into one buffer: it is
       longer than any log line, and a truncated header is worse than
       none - the part that explains the ticks would be the part cut. */
    fputs("# kunkun98 MIDI stream trace - one line per stream event, t in "
          "monotone milliseconds.\n", g_trace);
    fprintf(g_trace,
            "# ticks are the stream's own, %d to a quarter note "
            "(common/smf.h's LZ_SMF_DIVISION).\n", LZ_SMF_DIVISION);
    fputs("# The piece's own tempo map is folded into the POSITIONS by "
          "common/smf.c, so a tick here is not a tick of the file\n"
          "# and its wall-clock length follows the tempo the file wrote. "
          "The speed ratio is a MEVT_TEMPO at the head of a chunk, so\n"
          "# the stream runs at one tempo at a time - the value of the last "
          "one emitted - and every chunk from there on is played at it.\n",
          g_trace);
    fputs("# pos is midiStreamPosition. Stopping does not rewind it, so "
          "compare pos with a chunk's tick range only within one\n"
          "# playback. n counts submissions since the piece began, in the "
          "order the device hands them back; a `chunk` line is a\n"
          "# submission with no tempo event in it and a `tempo` line is one "
          "with, so a chunk's number is either kind.\n", g_trace);
    fputs("# inflight is what the device still owes. pos=unavailable means "
          "a position that could not be asked for, not a stream at zero.\n",
          g_trace);
    /* Told here rather than left to be worked out from the lines,
       because it is the one thing about t that changes what a reader
       concludes from them, and it is not visible in the numbers. */
    fputs("# t IS COARSE AND ITS STEP DEPENDS ON UPTIME, so two lines "
          "sharing one t is not evidence that the two things they\n"
          "# report were simultaneous. lz_time_ms answers a float: with "
          "QueryPerformanceCounter at 10 MHz and the machine up about\n"
          "# ten days (t around 9.3e8) its count is already rounded to "
          "1048576 counts - 104.9 ms - before the divide, and the\n"
          "# reading then steps in 64 ms. That is most of the interval "
          "between two of these lines. A freshly booted machine resolves\n"
          "# far finer, so two captures cannot be compared on this column "
          "either.\n", g_trace);
    fprintf(g_trace,
            "# dt_us is the clock that resolves it: microseconds from "
            "%s, read as t0, pos0, the submission, pos1,\n"
            "# t1, so dt_us = t1 - t0 brackets both position samples. "
            "us0 is the tempo the stream was at before this chunk and us\n"
            "# the value this chunk's event carries - the music from the "
            "head of this chunk on is played at us and not at us0, so the\n"
            "# line has to carry both. us0=none means no event has been "
            "emitted into this stream yet, so the driver's own default is\n"
            "# what was in force; it is not 0.\n"
            "# A submission that moves the stream shows as pos1 > pos0 with "
            "dt_us far too small to have played the music between\n"
            "# them; pos1 == pos0 is one that moved nothing.\n",
            lz_ui_us_fine() ? "QueryPerformanceCounter, quantum 100 ns"
                            : "GetTickCount - every dt_us is a whole "
                              "millisecond, 15.6 ms or 55 ms apart");
    /* The control arm, and why it is the line that makes the others
       mean something. Nothing in a tempo line's own numbers separates a
       submission that moved the stream from a device that reports its
       position on some cadence of its own - both show as an advance
       across a window too short to have played the music. A window with
       no submission in it is what separates them, and a submission with
       no event in it separates the submission from the event. */
    fputs("# probe is the CONTROL for that: the same two position samples "
          "with NOTHING submitted between them, taken from the same call\n"
          "# the ratio moves are reported from and on the ticks where it "
          "did not move. pos, pos0 and dt_us mean what they mean on a\n"
          "# tempo line, so one expression reads all three arms: an advance "
          "that shows up on a probe line too is the device's own\n"
          "# reporting cadence, and one that shows up across every "
          "submission is the submission rather than anything in it.\n"
          "# chunk is the second control and the reason there are two: it "
          "is the same call on the same kind of buffer with no tempo event\n"
          "# in it, so an advance a tempo line shows and a chunk line does "
          "not belongs to the event. dpos is pos - pos0, the advance\n"
          "# itself, and every arm carries it.\n", g_trace);
    fputs("# A probe line is written ONLY while a piece is playing AND the "
          "device still owes a buffer (inflight > 0): with the stream\n"
          "# stopped, or with the last chunk played out, a position that "
          "stands still answers with the state rather than with the\n"
          "# submission. us is the tempo the last emitted event carried - a "
          "probe line has one, not two, so there is no us0 - and it\n"
          "# reads `none` until an event has been emitted, exactly as us0 "
          "does on a tempo line. The device applies an event when it\n"
          "# REACHES that chunk, so while one is still queued us is what "
          "the music will be played at and not yet what is sounding.\n"
          "# pos=unavailable, or dpos=unavailable, is a reading that could "
          "not be taken.\n", g_trace);
    fprintf(g_trace,
            "# At most one probe line per %lu ms, so several stand between "
            "two tempo lines while the file's growth stays tied to the\n"
            "# submission rate rather than to the tick rate.\n",
            (unsigned long)(LZ_MIDI_PROBE_MIN_US / LZ_I64_C(1000)));
    fflush(g_trace);
}

/* The tempo in force, into `buf`, or the word for "no event has been
   emitted into this stream yet". A zero in its place would read as a
   tempo the device was running at, and until the first event reaches it
   the stream is at the driver's own default of 500000 - a number this
   player did not set and, on a piece whose own opening tempo is
   anything else, is not the one the music is about to be played at. */
static void tempo_text(char *buf, int cap) {
    if (g_us_emit) snprintf(buf, (size_t)cap, "%lu", g_us_emit);
    else           snprintf(buf, (size_t)cap, "none");
}

/* Read the position twice with NOTHING between the samples - the
 * control arm for what a tempo line's bracket shows.
 *
 * The samples are the same two calls in the same order, so pos, pos0
 * and dt_us mean the same thing on every line: how far the stream moved
 * across one immediate pair of reads. That is what makes the pair
 * answerable at all. A submission that moves the stream and a device
 * that reports its position on some cadence of its own both show as an
 * advance over a window far too short to have played the music, and
 * only a window with no submission in it tells them apart - the reading
 * is taken from the same call site the submissions are made from, on
 * the ticks where the ratio did not move. A chunk line is the second
 * control for the same reason: a submission with no event in it
 * separates the submission from the event.
 *
 * Written ONLY while a piece is playing and the device still owes a
 * buffer: stopped, or with the last chunk played out, the stream has a
 * position that stands still for a reason that has nothing to do with
 * any submission, and a control sample taken there would answer with
 * the state instead.
 *
 * Rate-limited to one line per LZ_MIDI_PROBE_MIN_US, on the same
 * counter the bracket uses. The read that enforces the limit is the
 * bracket's own t0, so the bound costs no extra clock call. */
void lz_midi_probe(void) {
    unsigned long p0 = 0, p1 = 0;
    lz_i64 t0, t1;
    int got0, got1;

    if (!tracing()) return;
    if (!g_playing || g_pending <= 0) return;
    t0 = lz_ui_us_now();
    if (g_probe_set && t0 - g_probe_last < LZ_MIDI_PROBE_MIN_US) return;
    g_probe_last = t0;
    g_probe_set = 1;

    got0 = lz_ui_midi_position(&p0);
    got1 = lz_ui_midi_position(&p1);
    t1 = lz_ui_us_now();
    bracket_text(got0, p0, got1, p1);
    tempo_text(g_us0, (int)sizeof g_us0);
    snprintf(g_line, sizeof g_line,
             "probe pos=%s pos0=%s dpos=%s dt_us=%lu us=%s inflight=%d",
             g_pos1, g_pos0, g_dpos, (unsigned long)(t1 - t0), g_us0,
             g_pending);
    trace_line(g_line);
}

/* `inflight` is what the device will owe once the caller counts this
   one - and this one only when the device took it. */
static int inflight_after(int ok) { return g_pending + (ok ? 1 : 0); }

/* A submission that carried no tempo event: the control for the tempo
   arm, because it is the same call on the same kind of buffer with
   nothing in it that could re-clock the stream. */
static void trace_chunk(int slot, int events, unsigned long first,
                        unsigned long last, int ok, unsigned long dt_us) {
    snprintf(g_line, sizeof g_line,
             "chunk n=%d slot=%d ticks=%lu..%lu events=%d ok=%d pos=%s "
             "pos0=%s dpos=%s dt_us=%lu inflight=%d",
             g_sent, slot, first, last, events, ok, g_pos1, g_pos0, g_dpos,
             dt_us, inflight_after(ok));
    trace_line(g_line);
}

/* A submission that carried a MEVT_TEMPO at the head of it. Everything
   a chunk line carries is here as well, in the same order, so the two
   arms differ only in what they say the submission carried - a reader
   taking a piece's tick ranges off the `chunk` lines alone would be
   missing exactly the chunks that were re-clocked. us0 is the tempo the
   stream was at before this one, which reads `none` while no event has
   been emitted into this stream at all. */
static void trace_tempo(int opening, int slot, int events,
                        unsigned long first, unsigned long last,
                        unsigned long us, int ok, unsigned long dt_us) {
    snprintf(g_line, sizeof g_line,
             "tempo why=%s n=%d slot=%d ticks=%lu..%lu events=%d ratio=%d "
             "us=%lu ok=%d pos=%s pos0=%s dpos=%s dt_us=%lu us0=%s "
             "inflight=%d",
             opening ? "start" : "follow", g_sent, slot, first, last, events,
             g_ratio, us, ok, g_pos1, g_pos0, g_dpos, dt_us, g_us0,
             inflight_after(ok));
    trace_line(g_line);
}

static void trace_go(const char *why, int ok) {
    snprintf(g_line, sizeof g_line, "go why=%s ok=%d inflight=%d",
             why, ok, g_pending);
    trace_line(g_line);
}

/* `n` is the chunk that finished, or -1 when the notification was not
   one this playback was owed. */
static void trace_done(int n, const char *why) {
    if (why)
        snprintf(g_line, sizeof g_line, "done why=%s inflight=%d",
                 why, g_pending);
    else
        snprintf(g_line, sizeof g_line, "done n=%d inflight=%d",
                 n, g_pending);
    trace_line(g_line);
}

static int buffers(void) {
    int i, ok = 1;
    if (g_buf[0]) return 1;
    for (i = 0; i < LZ_UI_MIDI_SLOTS; i++) {
        g_buf[i] = (unsigned long *)malloc(sizeof **g_buf * 3 *
                                           LZ_SMF_CHUNK_EVENTS);
        if (!g_buf[i]) ok = 0;
    }
    if (ok) return 1;
    for (i = 0; i < LZ_UI_MIDI_SLOTS; i++) {
        free(g_buf[i]);
        g_buf[i] = NULL;
    }
    return 0;
}

/* The image, grown to hold this file. Kept between pieces so the usual
 * case is no allocation at all. */
static int image(long n) {
    unsigned char *p;
    if (n <= g_img_cap && g_img) return 1;
    p = (unsigned char *)malloc((size_t)n);
    if (!p) return 0;
    free(g_img);
    g_img = p;
    g_img_cap = n;
    return 1;
}

/* How long one chunk of music may last, in microseconds at the tempo it
 * will play at - which is also how quickly a change of speed can be
 * HEARD.
 *
 * The ratio travels as a MEVT_TEMPO at the head of a chunk, so it can
 * only take effect where a chunk begins. Between the head of a chunk
 * and the moment it is submitted there is nothing; the wait is on the
 * other side, in music already handed over - the chunk playing and the
 * one queued behind it - so the worst case is two chunks and the music
 * that is committed ahead of any change is one. That is what this
 * bounds.
 *
 * It has to be bounded at all because the 64 KB the device will take
 * holds a whole piece in one buffer, and this one was measured: the
 * piece the defect was reported against is 2073 events over 38164
 * ticks, one chunk for the lot. Fine for a tempo that never moves, and
 * useless for one that does.
 *
 * 200 ms, chosen between the two failures at the ends of it. Longer
 * chunks make a ramp arrive as a few steps rather than as the change of
 * speed it is - which is the shape of the complaint this comes from.
 * Shorter ones leave less music in front of the next submission than
 * the front end can spend drawing a token, and a device that runs out
 * between chunks is a gap in the music, which is the other shape of the
 * same complaint. At two chunks of reaction and one of cushion, 200 ms
 * samples a ramp - half a unit a second, so two seconds from 1x to 2x -
 * twenty times, and leaves 200 ms of music standing between the stream
 * and an empty queue.
 *
 * What it costs is the submissions: five a second at the very most,
 * each a prepare, a midiStreamOut and a notification, where a piece
 * used to cost one for the whole of it. The real rate is lower and the
 * paragraph below says why: MEASURED, a 40 s run of the piece this was
 * reported on sends 76 chunks over 10141 ticks, which is two a second.
 *
 * AND IT IS A BOUND ON A CHUNK, NOT ON WHEN A CHANGE IS HEARD. A chunk
 * also lasts as long as the gap between its own last event and the
 * first event of the next one, and a single delta cannot be cut in half
 * without putting an event into the stream that the piece does not
 * have. A piece with a two-second rest in it therefore has a two-second
 * chunk, and a change of speed that arrives during that rest waits for
 * the rest to end. The same measurement: 136 ticks of music per chunk
 * against the 49 the bound asks for, and 3 of the 74 gaps over 1.2
 * seconds. So the bound is a floor on how responsive this can be and not
 * a promise - the noise it makes is a change arriving late, where the
 * piece has no event of its own.
 *
 * THE CONVERSION is into the stream's own ticks, and those are a fixed
 * number of microseconds each - the fold left the piece at one tempo,
 * so a tick is the tempo in force divided by LZ_SMF_DIVISION. Taken
 * from the tempo the chunk will PLAY at rather than from the piece's
 * base tempo, so a chunk is this long at every ratio: the base-tempo
 * figure would bound how much music a chunk holds, which is not the
 * quantity anything here cares about.
 *
 * The product is 3.84e7 and a 32-bit unsigned long is the target's, so
 * it cannot overflow; the quotient is 2 or more for every tempo the
 * ratio can ask for, because lz_music_tempo_us holds the answer under
 * LZ_MUSIC_TEMPO_MAX. */
#define LZ_MIDI_CHUNK_US 200000UL

/* The bound above in the stream's own ticks. Never 0, and that guard is
 * not decoration: 0 is LZSmfReader.span's own word for "no bound", so a
 * quotient that came out 0 would quietly put the whole piece back into
 * one chunk - the mechanism off rather than the chunks short. */
static unsigned long chunk_span(unsigned long us) {
    unsigned long t = LZ_MIDI_CHUNK_US * LZ_SMF_DIVISION / us;
    return t < 1 ? 1 : t;
}

/* Pull the next chunk into slot `s` and hand it over. Deltas continue
 * from g_prev_tick, so a join is not a gap. Returns 0 when the reader
 * had nothing left.
 *
 * The chunk carries a MEVT_TEMPO at its head whenever the tempo in
 * force is not the one the stream is running at. That is the whole
 * mechanism: a piece whose ratio never moves carries no tempo event at
 * all, and one whose ratio moves carries exactly one at the head of the
 * chunk that follows each move - never a run of them, and never one in
 * the middle of a chunk, where it would re-clock the music on one side
 * of it differently from the other.
 *
 * The tempo goes at the HEAD rather than where the ratio moved because
 * the events after it have already been read into this buffer: the
 * chunk is what the stream has to be re-clocked for, so the first
 * instant it can be is the first instant of the chunk. What the ratio
 * was doing while the previous chunk played is not reproducible here
 * and was never in the music anyway - the fold took the file's tempo
 * out of the event stream and left it as one number. */
static int send_chunk(int s) {
    unsigned long us, p0 = 0, p1 = 0, from = 0, first = 0;
    lz_i64 t0 = 0, t1 = 0;
    int n, ok, on = tracing(), got0 = 0, got1 = 0, fresh, head;

    us = tempo_for(g_ratio);
    /* The first chunk of a piece always carries one, so a piece's
       opening tempo is WRITTEN rather than inherited. The driver's own
       default is 500000 us per quarter and the fold makes the stream's
       absolute tempo load-bearing - the piece this was reported on
       opens at 769231, so played at the default every note of it would
       be at the wrong speed - and leaving it out when the last piece
       happened to end at the same number would make a piece's opening
       depend on what was played before it. */
    fresh = (g_sent == 0) || us != g_us_emit;
    /* One of the buffer's events is the tempo event when there is one:
       LZ_SMF_CHUNK_EVENTS is a bound on what the device will take, and
       it counts triples of any kind. */
    head = fresh ? 3 : 0;
    g_rd.span = chunk_span(us);
    /* Read before the reader moves it: the chunk's first event is this
       many ticks plus its own opening delta. */
    if (on) from = g_prev_tick;
    n = lz_smf_chunk(&g_rd, g_buf[s] + head,
                     LZ_SMF_CHUNK_EVENTS - (fresh ? 1 : 0), &g_prev_tick);
    /* A chunk that stopped at its own bound says nothing about whether
       the piece has more; only the reader knows, and this is how it
       says so. */
    if (g_rd.spent) g_more = 0;
    if (n <= 0) return 0;
    if (fresh) {
        /* Three DWORDs like every other event in the buffer: no delta,
           because it belongs at the instant the chunk begins, and no
           stream id. compat40.h carries the event word. */
        g_buf[s][0] = 0;
        g_buf[s][1] = 0;
        g_buf[s][2] = (LZ_UI_MEVT_TEMPO << 24) | us;
    }
    if (on) {
        /* Read before the send, while g_us_emit is still the tempo the
           stream was at: the line reports both, and after this chunk
           they are the same number. */
        tempo_text(g_us0, (int)sizeof g_us0);
        /* Two position samples straddling the submission as closely as
           the code allows, so the interval between them is two adjacent
           calls and not the ~110 ms between two of these ticks. A jump
           inside the submission is what is being looked for; the
           ordinary passage of music is not it. dt_us is a counter
           difference for exactly that reason - lz_time_ms steps in 64 ms
           at this machine's uptime, which is most of the interval being
           judged. */
        t0 = lz_ui_us_now();
        got0 = lz_ui_midi_position(&p0);
    }
    ok = lz_ui_midi_send(s, g_buf[s],
                         (unsigned long)(n + (fresh ? 1 : 0)) * 3UL * 4UL);
    if (on) {
        first = from + g_buf[s][head];
        got1 = lz_ui_midi_position(&p1);
        t1 = lz_ui_us_now();
        bracket_text(got0, p0, got1, p1);
        if (fresh)
            trace_tempo(g_sent == 0, s, n, first, g_prev_tick, us, ok,
                        (unsigned long)(t1 - t0));
        else
            trace_chunk(s, n, first, g_prev_tick, ok,
                        (unsigned long)(t1 - t0));
    }
    /* Only an event the device took is one it is running at. */
    if (ok && fresh) g_us_emit = us;
    g_sent++;
    return ok;
}

int lz_midi_system_ok(HWND notify) {
    if (g_opened) return 1;
    /* Asked once: a machine without the entry points will not grow
       them, and a failed open on a busy device is slow enough that
       retrying per turn would be felt. lz_midi_shutdown re-arms it. */
    if (g_tried) return 0;
    g_tried = 1;
    /* One division for the device, for every piece: the parser
       converts whatever the file used into this one. */
    g_opened = lz_ui_midi_open(notify, LZ_SMF_DIVISION);
    return g_opened;
}

int lz_midi_wave_ok(void) {
    /* Not unwritten code: the player exists and the bank format
       carries a codec field. The compressed audio format is what is
       missing, so there is no bank to read. */
    return 0;
}

/* The common half: an image already in memory, whatever it came from. */
static int start_image(const unsigned char *img, long n, int loop,
                       int instrument, int volume, int ratio_milli,
                       const char *name) {
    int s;

    /* No division check: the reader converts the file's own to
       LZ_SMF_DIVISION, which is what the stream was opened with. A
       survey of the MIDI on this machine found 480 far commoner than
       192, so refusing anything else would refuse almost every file a
       user copied in. */
    if (lz_smf_open(&g_rd, img, n, instrument, volume) != LZ_SMF_OK)
        return 0;

    /* Stop FIRST, and before any of this piece's state is written:
       lz_midi_halt clears g_loop, so setting it above would be undone
       here and every piece would play once. It also accounts for what
       the device still owed the piece being replaced.

       No clock rewind. midiStreamStop does not reset the stream's
       position and there is no property that does, so closing and
       reopening looked like the fix - MEASURED, it is not one. A piece
       restarted after a stop finishes its first chunk 7625 ms in,
       exactly as a cold start does; the reopen changed nothing and
       cost 687 ms of silence. Deltas are relative, so a position that
       keeps running is not a position that is wrong. */
    lz_midi_halt();

    g_us_qn0 = g_rd.us_qn0;
    g_ratio = ratio_milli;
    g_loop = loop;
    if (tracing()) {
        snprintf(g_piece, sizeof g_piece, "%s", name ? name : "?");
        snprintf(g_line, sizeof g_line,
                 "start piece=%s ratio=%d us_qn0=%lu loop=%d",
                 g_piece, ratio_milli, g_us_qn0, loop);
        trace_line(g_line);
    }
    /* No tempo call before the notes any more: the piece's opening
       tempo travels as the MEVT_TEMPO at the head of the first chunk,
       which every piece's first chunk carries because g_sent is rewound
       to 0 below. It has to be written and cannot be inherited - the
       previous piece may have left the stream at another tempo, and the
       fold is only the piece's own speed at the tempo it was opened
       for. See send_chunk. */

    g_slot_next = 0;
    g_prev_tick = 0;
    g_pending = 0;
    g_sent = 0;
    g_done = 0;
    g_more = 1;
    /* Fill every slot before starting the clock. One chunk in the
       device and none behind it is a piece that stops at the first
       join while the notification walks the message queue. */
    for (s = 0; s < LZ_UI_MIDI_SLOTS && g_more; s++) {
        if (!send_chunk(g_slot_next)) break;
        g_slot_next = (g_slot_next + 1) % LZ_UI_MIDI_SLOTS;
        g_pending++;
    }
    if (!g_pending) return 0;
    g_playing = lz_ui_midi_go();
    if (tracing()) trace_go("start", g_playing);
    if (!g_playing) {
        lz_ui_midi_stop();
        g_pending = 0;
    }
    return g_playing;
}

int lz_midi_start(const char *path, int loop, int instrument,
                  int volume, int ratio_milli) {
    FILE *f;
    long n;

    if (!g_opened || !path || !*path) return 0;
    if (!buffers()) return 0;

    f = fopen(path, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    n = ftell(f);
    /* Sized from the file itself. The cap is a refusal for something
       that is not a piece of music, not a ceiling on how long a piece
       may be - the reader has none of those any more. */
    if (n <= 0 || n > LZ_MIDI_MAX_FILE) { fclose(f); return 0; }
    if (fseek(f, 0, SEEK_SET) != 0 || !image(n)) { fclose(f); return 0; }
    if ((long)fread(g_img, 1, (size_t)n, f) != n) { fclose(f); return 0; }
    fclose(f);

    return start_image(g_img, n, loop, instrument, volume, ratio_milli,
                       path);
}

int lz_midi_start_chime(int instrument, int volume) {
    if (!g_opened || !buffers()) return 0;
    return start_image(lz_music_chime, lz_music_chime_len, 0,
                       instrument, volume, LZ_MUSIC_RATIO_ONE,
                       "(the finish chime)");
}

void lz_midi_halt(void) {
    if (!g_opened) return;
    /* Read before the accounting below moves them: these are what the
       device still owed at the moment it was stopped, which is the
       number a reader wants - the total after the fold is not. */
    if (tracing()) {
        snprintf(g_line, sizeof g_line, "halt inflight=%d stale=%d",
                 g_pending, g_stale);
        trace_line(g_line);
    }
    g_playing = 0;
    g_loop = 0;
    lz_music_drop_pending(&g_pending, &g_stale);
    lz_ui_midi_stop();
}

/* The speed ratio moved. Nothing is sent from here: the value is
 * recorded and the next chunk carries it at its head, which is the only
 * place the stream can be re-clocked at all. How long that takes is the
 * chunk bound, twice over at the worst - see LZ_MIDI_CHUNK_US. */
void lz_midi_ratio(int ratio_milli) {
    if (!g_playing) return;
    g_ratio = ratio_milli;
}

/* Settings changed while a piece was playing.
 *
 * It STOPS, and the next turn starts with the new settings. Restarting
 * the piece here was the other candidate and it worked - measured, it
 * resumed from the top exactly as a cold start does - but from a
 * listener's seat a restart mid-phrase is the music jumping backwards,
 * which is what it was reported as. Stopping says one thing and says
 * it plainly.
 *
 * The arguments are gone with it: whatever the sound window changed is
 * read again by music_start on the next turn, from one place. */
void lz_midi_settings_changed(void) {
    lz_midi_halt();
}

int lz_midi_busy(void) {
    return g_playing;
}

void lz_midi_buffer_done(void) {
    int slot, ok;
    /* A leftover from a piece that was stopped. Ignoring it is the
       whole point: acted on, it stops the piece that replaced it. */
    if (!lz_music_own_done(&g_pending, &g_stale)) {
        if (tracing()) trace_done(-1, "stale");
        return;
    }
    if (!g_playing) {
        if (tracing()) trace_done(-1, "idle");
        return;
    }
    /* The device hands them back in the order they went out, so the one
       that just finished is the one this counter is on. */
    if (tracing()) trace_done(g_done, NULL);
    g_done++;

    /* Chunks come back in the order they went out, so the slot that
       just freed is the one furthest ahead of the next to fill. */
    slot = g_slot_next;

    if (!g_more) {
        /* The piece is fully queued. Loop from the top, or - for a
           one-shot, the chime - stop once the last chunk has played
           rather than at the first notification. */
        if (!g_loop) {
            if (g_pending > 0) return;   /* a chunk is still sounding */
            g_playing = 0;
            lz_ui_midi_stop();
            if (tracing()) trace_line("end why=played inflight=0");
            return;
        }
        /* Back to the start. The delta cursor resets with it, so the
           first event of the loop is at zero rather than a whole
           piece's worth of ticks in the future. */
        lz_smf_rewind(&g_rd);
        g_prev_tick = 0;
        g_more = 1;
        if (tracing()) {
            snprintf(g_line, sizeof g_line, "loop inflight=%d", g_pending);
            trace_line(g_line);
        }
    }
    if (send_chunk(slot)) {
        g_slot_next = (slot + 1) % LZ_UI_MIDI_SLOTS;
        g_pending++;
        ok = lz_ui_midi_go();
        if (tracing()) trace_go("resume", ok);
    } else if (g_pending <= 0) {
        g_playing = 0;
        lz_ui_midi_stop();
        if (tracing()) trace_line("end why=stalled inflight=0");
    }
}

void lz_midi_shutdown(void) {
    lz_midi_halt();
    if (g_opened) lz_ui_midi_close();
    g_opened = 0;
    g_tried = 0;
    g_pending = 0;
    g_stale = 0;
    /* The device goes with it, and the tempo is a property of the
       stream: a stream opened again is at the driver's default and not
       at whatever was last emitted into the one that closed. */
    g_us_emit = 0;
    free(g_img);
    g_img = NULL;
    g_img_cap = 0;
    {
        int i;
        for (i = 0; i < LZ_UI_MIDI_SLOTS; i++) {
            free(g_buf[i]);
            g_buf[i] = NULL;
        }
    }
    if (g_trace) {
        /* Nothing writes to it after this point, and a trace whose lines
           are still in a buffer when the process goes is a trace with
           the last of the story missing. */
        fclose(g_trace);
        g_trace = NULL;
    }
}
