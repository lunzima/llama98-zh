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
 * badly: what tempo writes landed, and what the stream was doing when
 * they did. A tempo line therefore carries the stream's position either
 * side of the write and the interval between the two samples, and every
 * line carries the number of buffers the device still owes. */

static FILE *g_trace;
static char  g_piece[MAX_PATH];  /* the piece the lines below are about */
static char  g_line[512];
static char  g_pos0[24];    /* the position before a tempo write */
static char  g_pos1[24];    /* and after it - two, because one line has both */
static char  g_us0[24];     /* the tempo the write replaced, or `none` */
static int   g_sent;        /* chunks handed over since the piece began */
static int   g_done;        /* and the one the next notification is for */
/* The tempo microseconds-per-quarter last written, which is the one the
 * ticks before the next write are worth, and whether any has been: a
 * zero in us0 would read as a tempo the device was running at, and the
 * driver's own default is what is in force until the first write. */
static unsigned long g_tempo_last;
static int           g_tempo_set;

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
          "# and its wall-clock length follows the tempo the file wrote.\n",
          g_trace);
    fputs("# pos is midiStreamPosition. Stopping does not rewind it, so "
          "compare pos with a chunk's tick range only within one\n"
          "# playback. chunk n counts submissions since the piece began, "
          "in the order the device hands them back.\n", g_trace);
    fputs("# inflight is what the device still owes. pos=unavailable means "
          "a position that could not be asked for, not a stream at zero.\n",
          g_trace);
    /* Told here rather than left to be worked out from the lines,
       because it is the one thing about t that changes what a reader
       concludes from them, and it is not visible in the numbers. */
    fputs("# t IS COARSE AND ITS STEP DEPENDS ON UPTIME, so a pair of tempo "
          "writes sharing one t is not evidence that they\n"
          "# were simultaneous. lz_time_ms answers a float: with "
          "QueryPerformanceCounter at 10 MHz and the machine up about\n"
          "# ten days (t around 9.3e8) its count is already rounded to "
          "1048576 counts - 104.9 ms - before the divide, and the\n"
          "# reading then steps in 64 ms. That is most of the ~110 ms "
          "between two writes. A freshly booted machine resolves far\n"
          "# finer, so two captures cannot be compared on this column "
          "either.\n", g_trace);
    fprintf(g_trace,
            "# dt_us is the clock that resolves it: microseconds from "
            "%s, read as t0, pos0, the write, pos1,\n"
            "# t1, so dt_us = t1 - t0 brackets both position samples. "
            "us0 is the tempo in force before the write and us the\n"
            "# one it writes - the tick-to-wall-clock conversion changes "
            "between the two samples, so the line has to carry both.\n"
            "# us0=none means no tempo has been written yet, so the "
            "driver's own default is what was in force; it is not 0.\n"
            "# A tempo write that moves the stream shows as pos1 > pos0 "
            "with dt_us far too small to have played the music\n"
            "# between them; pos1 == pos0 is a write that moved nothing.\n",
            lz_ui_us_fine() ? "QueryPerformanceCounter, quantum 100 ns"
                            : "GetTickCount - every dt_us is a whole "
                              "millisecond, 15.6 ms or 55 ms apart");
    fflush(g_trace);
}

/* Write the tempo, and record what the stream did across the write.
 *
 * The two position samples straddle the midiStreamProperty call as
 * closely as the code allows, and the interval between them is two
 * adjacent calls - not the ~110 ms between one tempo write and the
 * next. That is the whole point of the bracket: a jump inside the
 * write is what is being looked for, and the ordinary passage of music
 * between two writes is not it. dt_us is a counter difference for
 * exactly that reason; lz_time_ms steps in 64 ms at this uptime, which
 * is most of the interval being judged.
 *
 * The position is asked for AFTER the write as well as before, so the
 * line says where the stream had got to on each side. Whether the
 * property call succeeded changes which tempo the ticks are worth, so
 * us0 carries the one in force before and us the one written. */
static int tempo_write(const char *why, int ratio_milli, unsigned long us) {
    unsigned long p0 = 0, p1 = 0;
    lz_i64 t0 = 0, t1 = 0;
    int ok, on = tracing();
    int got0 = 0, got1 = 0;

    if (on) {
        t0 = lz_ui_us_now();
        got0 = lz_ui_midi_position(&p0);
    }
    ok = lz_ui_midi_tempo(us);
    if (!on) return ok;
    got1 = lz_ui_midi_position(&p1);
    t1 = lz_ui_us_now();
    pos_text(g_pos0, (int)sizeof g_pos0, got0, p0);
    pos_text(g_pos1, (int)sizeof g_pos1, got1, p1);
    if (g_tempo_set) snprintf(g_us0, sizeof g_us0, "%lu", g_tempo_last);
    else             snprintf(g_us0, sizeof g_us0, "none");
    /* `pos` keeps meaning what it meant before the bracket existed:
       where the stream was once the write had been made. */
    snprintf(g_line, sizeof g_line,
             "tempo why=%s ratio=%d us=%lu ok=%d pos=%s pos0=%s dt_us=%lu "
             "us0=%s inflight=%d",
             why, ratio_milli, us, ok, g_pos1, g_pos0,
             (unsigned long)(t1 - t0), g_us0, g_pending);
    trace_line(g_line);
    /* Only a write the device took changes what the ticks are worth. */
    if (ok) {
        g_tempo_last = us;
        g_tempo_set = 1;
    }
    return ok;
}

static void trace_chunk(int slot, int events, unsigned long first,
                        unsigned long last, int ok) {
    /* inflight is what it will be once the caller counts this one, and
       this one only when the device took it. */
    snprintf(g_line, sizeof g_line,
             "chunk n=%d slot=%d ticks=%lu..%lu events=%d ok=%d inflight=%d",
             g_sent, slot, first, last, events, ok,
             g_pending + (ok ? 1 : 0));
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

/* Pull the next chunk into slot `s` and hand it over. Deltas continue
 * from g_prev_tick, so a join is not a gap. Returns 0 when the reader
 * had nothing left. */
static int send_chunk(int s) {
    int n, ok, on = tracing();
    unsigned long from = 0;

    /* Read before the reader moves it: the chunk's first event is this
       many ticks plus its own opening delta. */
    if (on) from = g_prev_tick;
    n = lz_smf_chunk(&g_rd, g_buf[s], LZ_SMF_CHUNK_EVENTS, &g_prev_tick);
    if (n < LZ_SMF_CHUNK_EVENTS) g_more = 0;
    if (n <= 0) return 0;
    ok = lz_ui_midi_send(s, g_buf[s], (unsigned long)n * 3UL * 4UL);
    if (on) trace_chunk(s, n, from + g_buf[s][0], g_prev_tick, ok);
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

/* In common/music.c, not here: a gate cannot reach into a translation
 * unit that needs a MIDI device. */
#define tempo_for(r) lz_music_tempo_us(g_us_qn0, (r))

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
    /* The tempo BEFORE the notes. Setting it after would play the
       opening at whatever the last piece left behind, which on a turn
       that follows a fast one is a noticeably wrong first bar. */
    tempo_write("start", ratio_milli, tempo_for(ratio_milli));

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

void lz_midi_ratio(int ratio_milli) {
    if (!g_playing) return;
    if (ratio_milli == g_ratio) return;
    g_ratio = ratio_milli;
    tempo_write("follow", ratio_milli, tempo_for(ratio_milli));
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
    /* The device goes with it, so the tempo written to it goes too. */
    g_tempo_set = 0;
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
