/* One MIDI file, at a tempo that moves. See midiplay.h. */
#include "midiplay.h"

#include <stdio.h>
#include <stdlib.h>

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
    int n = lz_smf_chunk(&g_rd, g_buf[s], LZ_SMF_CHUNK_EVENTS,
                         &g_prev_tick);
    if (n < LZ_SMF_CHUNK_EVENTS) g_more = 0;
    if (n <= 0) return 0;
    return lz_ui_midi_send(s, g_buf[s], (unsigned long)n * 3UL * 4UL);
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
                       int instrument, int volume, int ratio_milli) {
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
    /* The tempo BEFORE the notes. Setting it after would play the
       opening at whatever the last piece left behind, which on a turn
       that follows a fast one is a noticeably wrong first bar. */
    lz_ui_midi_tempo(tempo_for(ratio_milli));

    g_slot_next = 0;
    g_prev_tick = 0;
    g_pending = 0;
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

    return start_image(g_img, n, loop, instrument, volume, ratio_milli);
}

int lz_midi_start_chime(int instrument, int volume) {
    if (!g_opened || !buffers()) return 0;
    return start_image(lz_music_chime, lz_music_chime_len, 0,
                       instrument, volume, LZ_MUSIC_RATIO_ONE);
}

void lz_midi_halt(void) {
    if (!g_opened) return;
    g_playing = 0;
    g_loop = 0;
    lz_music_drop_pending(&g_pending, &g_stale);
    lz_ui_midi_stop();
}

void lz_midi_ratio(int ratio_milli) {
    if (!g_playing) return;
    if (ratio_milli == g_ratio) return;
    g_ratio = ratio_milli;
    lz_ui_midi_tempo(tempo_for(ratio_milli));
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
    int slot;
    /* A leftover from a piece that was stopped. Ignoring it is the
       whole point: acted on, it stops the piece that replaced it. */
    if (!lz_music_own_done(&g_pending, &g_stale)) return;
    if (!g_playing) return;

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
            return;
        }
        /* Back to the start. The delta cursor resets with it, so the
           first event of the loop is at zero rather than a whole
           piece's worth of ticks in the future. */
        lz_smf_rewind(&g_rd);
        g_prev_tick = 0;
        g_more = 1;
    }
    if (send_chunk(slot)) {
        g_slot_next = (slot + 1) % LZ_UI_MIDI_SLOTS;
        g_pending++;
        lz_ui_midi_go();
    } else if (g_pending <= 0) {
        g_playing = 0;
        lz_ui_midi_stop();
    }
}

void lz_midi_shutdown(void) {
    lz_midi_halt();
    if (g_opened) lz_ui_midi_close();
    g_opened = 0;
    g_tried = 0;
    g_pending = 0;
    g_stale = 0;
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
}
