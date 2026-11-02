#ifndef LZ_COMMON_MUSIC_H
#define LZ_COMMON_MUSIC_H

/* Background music for the generation phase: the rules, with no Win32
 * and no MIDI in them. The half that needs a device is gui/midiplay.c.
 *
 * Every value that crosses the ini is in THOUSANDTHS, not a float:
 * gui/inifile.h has the reason (the profile API has no float form, and
 * parsing one back runs through the locale's decimal point).
 */

/* Tempo range, in thousandths. */
#define LZ_MUSIC_RATIO_MIN   500
#define LZ_MUSIC_RATIO_MAX  2000
#define LZ_MUSIC_RATIO_ONE  1000

/* FIXED is not "off": the music plays, at the tempo it was written at. */
typedef enum {
    LZ_MUSIC_FOLLOW = 0,
    LZ_MUSIC_FIXED  = 1
} LZMusicSpeed;

/* BEEP stays on the list because it is the only one of the three a
 * machine with no MIDI device can still make. */
typedef enum {
    LZ_MUSIC_FINISH_NONE  = 0,
    LZ_MUSIC_FINISH_BEEP  = 1,   /* lz_beep() - MessageBeep(MB_OK) */
    LZ_MUSIC_FINISH_CHIME = 2
} LZMusicFinish;

/* SYSTEM hands the notes to the machine's MIDI mapper, which is why the
 * pieces are MIDI and not PCM: the patch is the machine's to choose.
 *
 * WAVE renders them here from a sample bank. The player exists and the
 * bank format carries a codec field; the compressed audio format it
 * needs is unfinished, so there is no bank to ship and
 * lz_music_engine_pick falls back to SYSTEM. That fall-back is SILENT,
 * so the dialog is what must mark the row unavailable. */
typedef enum {
    LZ_MUSIC_ENGINE_SYSTEM = 0,
    LZ_MUSIC_ENGINE_WAVE   = 1
} LZMusicEngine;

/* A guard against a degenerate sample, not a measured optimum: a
 * three-token reply is almost entirely first-token latency, and a
 * baseline taken from one pegs every later turn at 2x. */
#define LZ_MUSIC_CAL_MIN_TOKENS  16
#define LZ_MUSIC_CAL_MIN_MS    2000

/* Milli-tok/s. Stops a rate measured on the dev machine from being
 * written as a baseline no target machine can approach. */
#define LZ_MUSIC_TPS_MAX   1000000

typedef struct {
    int on;
    int speed;          /* LZMusicSpeed */
    int base_tps_milli; /* the rate that means 1.0x; 0 = not calibrated */
    int finish;         /* LZMusicFinish */
    int engine;         /* LZMusicEngine */
    int instrument;     /* a GM program, or LZ_MUSIC_AS_WRITTEN */
    int volume;         /* 0..127, straight into MIDI controller 7 */
} LZMusicSettings;

#define LZ_MUSIC_VOL_MAX      127
#define LZ_MUSIC_VOL_DEFAULT   80

/* ---- the instrument menu ----
 *
 * The ten GM programs banks/kk98.ksb carries, offered by BOTH engines
 * so one piece can be heard on either and compared. AS_WRITTEN is a
 * choice, not "none": the pieces carry their own program changes.
 *
 * ONE table, in music.c, read by the dialog and both players. */
#define LZ_MUSIC_INSTRUMENTS   10
#define LZ_MUSIC_AS_WRITTEN    (-1)

extern const unsigned char lz_music_instrument[LZ_MUSIC_INSTRUMENTS];

/* Its row, or -1. AS_WRITTEN is not in the table. */
int lz_music_instrument_index(int program);

void lz_music_defaults(LZMusicSettings *s);

/* An ini is a text file a user edits: clamp whatever it produced. */
void lz_music_clamp(LZMusicSettings *s);

/* 0 when there is nothing to measure - which is "no opinion", not
 * "zero tokens a second". */
int lz_music_tps_milli(int tokens, int elapsed_ms);

/* Answers RATIO_ONE, not the floor, whenever the question cannot be
 * answered: fixed mode, no baseline, nothing measured. "I do not know"
 * and "it is crawling" must not sound the same. */
int lz_music_ratio_milli(int tps_milli, int base_tps_milli, int speed);

/* No baseline yet AND a turn long enough to be a sample. */
int lz_music_should_calibrate(int base_tps_milli, int tokens,
                              int elapsed_ms);

/* ---- following, without lurching ----
 *
 * The rate is a CUMULATIVE average over the turn, which is the number
 * the status bar shows. Two things about it have to be handled or the
 * music jumps:
 *
 *   the first tokens      the denominator is a few tens of
 *                         milliseconds, so one token's latency comes
 *                         out as a rate several times the baseline and
 *                         pins the tempo at the ceiling
 *   any real change       applied in one write it is a jump cut, not
 *                         an accelerando
 */

/* Below this the average is one token's latency, not a rate. */
#define LZ_MUSIC_FOLLOW_MIN_TOKENS 8

/* Thousandths a second the ratio may travel. 500 puts a full 1x -> 2x
 * change at two seconds, which reads as the music speeding up rather
 * than as a different recording starting. A CHOICE, not a
 * measurement - it is here to be turned. */
#define LZ_MUSIC_SLEW_PER_SEC 500

/* Close enough: the tempo does not move for a target within this.
 *
 * A cumulative average has an INTEGER numerator and a denominator that
 * keeps growing, so between one token and the next the rate sags and
 * then recovers. Measured at a steady 8 tok/s: the target wobbled
 * 1.975x..2.000x for ever, which the slew could not damp because the
 * wobble was smaller than one tick's travel. Music that hunts by 2%
 * every second is not following anything, it is unsteady.
 *
 * 20 removes it entirely (measured: the ratio settles on one value and
 * never moves) at the cost of up to 2% of steady-state error, which is
 * below what anyone tracks in music playing behind a window. */
#define LZ_MUSIC_STEADY 20

/* Where the tempo should be heading, for a turn in progress. 1x until
 * there are enough tokens to divide by. */
int lz_music_target(int tokens, int elapsed_ms, int base_tps_milli,
                    int speed);

/* One step from `now` towards `target`, no further than the slew
 * allows in `dt_ms`. Lands exactly on the target rather than creeping
 * past it. */
int lz_music_ratio_step(int now, int target, int dt_ms);

/* It DIVIDES: a MIDI tempo is microseconds per beat, so twice the speed
 * is half the number. Here rather than beside the midiStream call
 * because a gate cannot reach into a file that needs a device. */
unsigned long lz_music_tempo_us(unsigned long base_us, int ratio_milli);

/* The tempo field is 24 bits; over it wraps, and a wrapped tempo plays
 * the whole piece in an instant. */
#define LZ_MUSIC_TEMPO_MAX  16777215UL

/* ---- the baseline, as text ----
 *
 * Two decimals, because the target manages single-digit tok/s. Hand
 * written rather than sprintf("%.2f")/strtod for the locale reason at
 * the top of this file: "4,20" and a reader that stops at the comma is
 * a setting that silently becomes 4. */
#define LZ_MUSIC_TPS_TEXT  16

void lz_music_format_tps(int milli, char *out, int cap);

/* 1 and fills *milli, or 0 and leaves it alone. Trailing rubbish is
 * refused rather than ignored. */
int lz_music_parse_tps(const char *s, int *milli);

/* ---- degrading, because the MIDI path is 4.0-era and the floor is
 *      3.51 (gui/compat40.h) ---- */

/* For an ini predating this setting. beep=0 stays SILENT: an upgrade
 * that starts making noise at someone who switched it off is the one
 * migration with no defence. */
int lz_music_finish_from_beep(int beep);

/* A chime with no engine becomes the beep, not silence. */
int lz_music_finish_effective(int finish, int engine_ok);

/* The wanted engine, the other one, or -1 for neither. */
int lz_music_engine_pick(int wanted, int system_ok, int wave_ok);

/* ---- the playlist ----
 *
 * The ini's list when it has one - a user who wrote a list wrote it to
 * leave something out - and the music directory otherwise. */
#define LZ_MUSIC_LIST_SEP ';'

/* Entry `idx` into `out`; 0 past the end. Spaces are trimmed and empty
 * entries skipped, so a hand-typed list indexes like a written one. */
int lz_music_list_at(const char *list, int idx, char *out, int cap);
int lz_music_list_count(const char *list);

/* Beside the executable, like the ini and for the same reason. */
#define LZ_MUSIC_DIR    "music"

/* The finish chime is BUILT IN and is not a file. Under 150 bytes, and
 * it is the only sound the program owes the user at a moment they may
 * not be watching - an installation where music/ was never copied
 * across still says the reply has landed. The bytes are in
 * common/chime.c and go through the same reader as any other piece. */
extern const unsigned char lz_music_chime[];
extern const int lz_music_chime_len;

/* A .mid, in whatever case the filesystem reports it. */
int lz_music_is_loop_file(const char *name);

/* Rotation, not a random pick: with eight pieces the same one three
 * turns running is ordinary chance and reads as a stuck player. */
int lz_music_next(int count, int *cursor);

/* ---- whose buffer just finished ----
 *
 * Stopping a stream makes the device hand the queued buffer back, and
 * the notification for it can arrive AFTER the next piece has started.
 * Taken at face value it stops the piece that replaced it - measured:
 * the chime was started, the music's leftover notification arrived, and
 * the chime was silenced before it sounded.
 *
 * So the two are counted. `pending` is what the device still owes the
 * current playback; `stale` is what it owes a playback that is over. */

/* A playback is being stopped or replaced: what it was owed is now a
 * leftover. */
void lz_music_drop_pending(int *pending, int *stale);

/* A buffer came back. 1 when it belongs to what is playing now, 0 when
 * it is a leftover and must be ignored. */
int lz_music_own_done(int *pending, int *stale);

#endif /* LZ_COMMON_MUSIC_H */
