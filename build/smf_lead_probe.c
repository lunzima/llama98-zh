/* Does lz_smf_open() find the piece's true first sounding note across
 * every track, or only the first track (by index) that happens to have
 * one?
 *
 * See smf.c's own comment on track_advance's first_note parameter for
 * the full account. Short version: lz_smf_open's pass 1 calls
 * track_advance once per track, each run to completion before the next
 * track starts, sharing one `first` accumulator across all of them. A
 * note-on found while `first` still holds its sentinel updates it; a
 * note-on found afterward must still update it when it is EARLIER, or a
 * part that starts on a later-indexed track ahead of an earlier-indexed
 * one has its lead-in silently absorbed into the trim - every event in
 * the piece before the first track's own first note lands at output
 * tick 0, all at once, which sounds like the intro got skipped rather
 * than played.
 *
 * Two synthetic three-track files, built by hand at the byte level so
 * this probe has no dependency on any MIDI file on disk:
 *   ORDER_LATE_FIRST - track 1 (channel 0) sounds at tick 100, track 2
 *     (channel 1) sounds at tick 10. The true first note is track 2's,
 *     but track 1 is scanned first and has one of its own - exactly the
 *     shape that defeats an "only ever write once" accumulator.
 *   ORDER_EARLY_FIRST - the same two notes, ticks swapped between the
 *     tracks, so track order and tick order already agree. A control:
 *     if this ever regressed too, the fix would be over-firing rather
 *     than fixing the real defect.
 * Both carry a leading track 0 with no channel messages at all (a meta-
 * only track, the common real-file shape), and a velocity-0 note-on
 * placed before the real note in track 1 of both files, to confirm the
 * existing "note-on vel 0 is a note-off" rule survives the edit.
 *
 * Exit: 0 all checks pass, 1 any check fails.
 */
#include <stdio.h>
#include <string.h>

#include "music.h"
#include "smf.h"

static int g_fail = 0;

static void ok(int cond, const char *what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) g_fail = 1;
}

/* ---- byte-level SMF construction ---- */

static unsigned char buf[512];
static int len;

static void put_be16(int v) { buf[len++] = (unsigned char)(v >> 8); buf[len++] = (unsigned char)v; }
static void put_be32(long v) {
    buf[len++] = (unsigned char)(v >> 24); buf[len++] = (unsigned char)(v >> 16);
    buf[len++] = (unsigned char)(v >> 8);  buf[len++] = (unsigned char)v;
}
static void put_u8(int v) { buf[len++] = (unsigned char)v; }

/* Single-byte VLQ - every delta this probe uses is < 128. */
static void put_vlq(int v) { put_u8(v & 0x7F); }

/* One track's channel-message body: an optional velocity-0 note-on
 * (a note-off in disguise, must never be read as the first sound),
 * then the real note-on, then End Of Track. */
static void build_track(int delta, int channel, int note, int vel,
                        int lead_zero_vel_first) {
    if (lead_zero_vel_first) {
        put_vlq(0);
        put_u8(0x90 | channel);
        put_u8(note);
        put_u8(0);           /* velocity 0: a note-off in disguise */
    }
    put_vlq(delta);
    put_u8(0x90 | channel);
    put_u8(note);
    put_u8(vel);
    put_vlq(0);
    put_u8(0xFF); put_u8(0x2F); put_u8(0x00);   /* End Of Track */
}

static void begin_mtrk(int *hdr_at) {
    put_u8('M'); put_u8('T'); put_u8('r'); put_u8('k');
    *hdr_at = len;
    put_be32(0);   /* patched below */
}
static void end_mtrk(int hdr_at, int body_len) {
    buf[hdr_at + 0] = (unsigned char)(body_len >> 24);
    buf[hdr_at + 1] = (unsigned char)(body_len >> 16);
    buf[hdr_at + 2] = (unsigned char)(body_len >> 8);
    buf[hdr_at + 3] = (unsigned char)body_len;
}

/* track1_delta/track2_delta are each track's own first-event delta from
 * 0 (this probe never uses more than one channel event per track, so
 * delta == absolute tick). track1 is channel 0, track2 is channel 1;
 * both come after an empty meta-only track 0, matching a real file's
 * usual shape (track 0 carries tempo/name, never notes). */
static long build_file(int track1_delta, int track2_delta) {
    int hdr;
    len = 0;
    put_u8('M'); put_u8('T'); put_u8('h'); put_u8('d');
    put_be32(6);
    put_be16(1);     /* format 1 */
    put_be16(3);     /* 3 tracks */
    put_be16(192);   /* division, == LZ_SMF_DIVISION so lead reads in file ticks unscaled */

    /* track 0: meta only, no channel messages */
    begin_mtrk(&hdr);
    { int body_at = len;
      put_vlq(0); put_u8(0xFF); put_u8(0x2F); put_u8(0x00);
      end_mtrk(hdr, len - body_at); }

    /* track 1: channel 0 */
    begin_mtrk(&hdr);
    { int body_at = len;
      build_track(track1_delta, 0, 60, 100, 1);
      end_mtrk(hdr, len - body_at); }

    /* track 2: channel 1 */
    begin_mtrk(&hdr);
    { int body_at = len;
      build_track(track2_delta, 1, 64, 100, 0);
      end_mtrk(hdr, len - body_at); }

    return len;
}

int main(void) {
    LZSmfReader r;
    long n;

    /* ORDER_LATE_FIRST: track 1 (scanned first) sounds at 100, track 2
       (scanned second, channel 1) sounds at 10 - the true minimum, on
       the LATER track. */
    n = build_file(100, 10);
    ok(lz_smf_open(&r, buf, n, LZ_MUSIC_AS_WRITTEN, 100) == LZ_SMF_OK,
       "late-first file opens");
    ok(r.lead == 10,
       "late-first: lead is the TRUE minimum (10), not track 1's own first note (100)");

    /* ORDER_EARLY_FIRST control: same two notes, ticks swapped so track
       order already agrees with tick order. Must still read 10 - the
       fix must not change the answer when there was never a discrepancy
       to hide. */
    n = build_file(10, 100);
    ok(lz_smf_open(&r, buf, n, LZ_MUSIC_AS_WRITTEN, 100) == LZ_SMF_OK,
       "early-first file opens");
    ok(r.lead == 10,
       "early-first: lead is 10 when track order already matches tick order");

    /* The velocity-0 note-on placed before track 1's real note-on in
       BOTH files above must never have been read as the first sound -
       if it were, lead would come out 0 instead of 10 in the control
       case. Already covered by the assertion above; this just names it
       so a future refactor cannot silently start counting vel-0. */
    ok(r.lead != 0, "a velocity-0 note-on is never taken as the first sound");

    printf(g_fail ? "smf_lead_probe: FAIL\n" : "smf_lead_probe: PASS\n");
    return g_fail ? 1 : 0;
}
