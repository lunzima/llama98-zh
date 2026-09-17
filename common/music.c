/* The rules behind the generation-phase music. See music.h. */
#include "music.h"

#include <stdio.h>
#include <string.h>

/* The ten the wavetable bank carries, in menu order. Read off
 * banks/kk98.ksb itself rather than chosen here, which is why the
 * order is by program number and not by anything musical. */
const unsigned char lz_music_instrument[LZ_MUSIC_INSTRUMENTS] = {
    0,    /* acoustic grand */
    4,    /* electric piano */
    6,    /* harpsichord */
    7,    /* clavinet */
    12,   /* marimba */
    16,   /* drawbar organ */
    25,   /* steel guitar */
    46,   /* orchestral harp */
    80,   /* square lead */
    81    /* saw lead */
};

int lz_music_instrument_index(int program) {
    int i;
    if (program < 0) return -1;
    for (i = 0; i < LZ_MUSIC_INSTRUMENTS; i++)
        if ((int)lz_music_instrument[i] == program) return i;
    return -1;
}

void lz_music_defaults(LZMusicSettings *s) {
    if (!s) return;
    s->on = 1;
    s->speed = LZ_MUSIC_FOLLOW;
    /* No factory baseline: the number is a property of the machine,
       and the first long enough turn supplies it. */
    s->base_tps_milli = 0;
    s->finish = LZ_MUSIC_FINISH_CHIME;
    s->engine = LZ_MUSIC_ENGINE_SYSTEM;
    s->instrument = LZ_MUSIC_AS_WRITTEN;
    s->volume = LZ_MUSIC_VOL_DEFAULT;
}

void lz_music_clamp(LZMusicSettings *s) {
    if (!s) return;
    s->on = s->on ? 1 : 0;
    /* Out-of-range named values go to the DEFAULT, not to 0: finish's
       0 is NONE, and a typo that silences the program has no visible
       symptom. */
    if (s->finish != LZ_MUSIC_FINISH_NONE &&
        s->finish != LZ_MUSIC_FINISH_BEEP &&
        s->finish != LZ_MUSIC_FINISH_CHIME)
        s->finish = LZ_MUSIC_FINISH_CHIME;
    if (s->engine != LZ_MUSIC_ENGINE_WAVE)
        s->engine = LZ_MUSIC_ENGINE_SYSTEM;
    /* Back to the file's own, not to program 0: the bank has samples
       for exactly these ten, and an eleventh resolves to whichever
       region it falls back on. */
    if (s->instrument != LZ_MUSIC_AS_WRITTEN &&
        lz_music_instrument_index(s->instrument) < 0)
        s->instrument = LZ_MUSIC_AS_WRITTEN;
    if (s->speed != LZ_MUSIC_FIXED) s->speed = LZ_MUSIC_FOLLOW;
    if (s->base_tps_milli < 0) s->base_tps_milli = 0;
    if (s->base_tps_milli > LZ_MUSIC_TPS_MAX)
        s->base_tps_milli = LZ_MUSIC_TPS_MAX;
    if (s->volume < 0) s->volume = 0;
    if (s->volume > LZ_MUSIC_VOL_MAX) s->volume = LZ_MUSIC_VOL_MAX;
}

int lz_music_tps_milli(int tokens, int elapsed_ms) {
    double v;
    if (tokens <= 0 || elapsed_ms <= 0) return 0;
    /* Through a double: tokens * 1000000 overflows a 32-bit int at
       2148 tokens - an ordinary reply length - and the wrap is
       NEGATIVE, which clamps to the tempo floor. */
    v = (double)tokens * 1000000.0 / (double)elapsed_ms;
    if (v <= 0.0) return 0;
    if (v > (double)LZ_MUSIC_TPS_MAX) return LZ_MUSIC_TPS_MAX;
    return (int)v;
}

int lz_music_ratio_milli(int tps_milli, int base_tps_milli, int speed) {
    double r;
    if (speed == LZ_MUSIC_FIXED) return LZ_MUSIC_RATIO_ONE;
    if (base_tps_milli <= 0) return LZ_MUSIC_RATIO_ONE;
    if (tps_milli <= 0) return LZ_MUSIC_RATIO_ONE;
    r = (double)tps_milli * (double)LZ_MUSIC_RATIO_ONE
        / (double)base_tps_milli;
    if (r < (double)LZ_MUSIC_RATIO_MIN) return LZ_MUSIC_RATIO_MIN;
    if (r > (double)LZ_MUSIC_RATIO_MAX) return LZ_MUSIC_RATIO_MAX;
    return (int)r;
}

int lz_music_should_calibrate(int base_tps_milli, int tokens,
                              int elapsed_ms) {
    if (base_tps_milli > 0) return 0;
    if (tokens < LZ_MUSIC_CAL_MIN_TOKENS) return 0;
    if (elapsed_ms < LZ_MUSIC_CAL_MIN_MS) return 0;
    /* Both guards, not either: a fast machine passes the token count
       in well under two seconds. */
    return lz_music_tps_milli(tokens, elapsed_ms) > 0;
}

int lz_music_target(int tokens, int elapsed_ms, int base_tps_milli,
                    int speed) {
    /* Not "return the floor" and not "return whatever a two-token
       sample says": 1x is the answer to a question that cannot be
       asked yet, and it is where the piece already is. */
    if (tokens < LZ_MUSIC_FOLLOW_MIN_TOKENS) return LZ_MUSIC_RATIO_ONE;
    return lz_music_ratio_milli(lz_music_tps_milli(tokens, elapsed_ms),
                                base_tps_milli, speed);
}

int lz_music_ratio_step(int now, int target, int dt_ms) {
    long room;
    int diff;
    if (now < LZ_MUSIC_RATIO_MIN) now = LZ_MUSIC_RATIO_MIN;
    if (now > LZ_MUSIC_RATIO_MAX) now = LZ_MUSIC_RATIO_MAX;
    if (dt_ms <= 0) return now;
    /* Already close enough - see LZ_MUSIC_STEADY. Without this the
       tempo hunts by 2% for ever, because the wobble in a cumulative
       average is smaller than one tick's travel and the slew cannot
       damp what it can always reach. */
    diff = target > now ? target - now : now - target;
    if (diff <= LZ_MUSIC_STEADY) return now;
    /* A tick that arrived very late - the machine was busy, which on
       this hardware is the normal case - must not become one enormous
       step. Capped at a second's worth, so a stall spends a second
       catching up rather than jumping. */
    if (dt_ms > 1000) dt_ms = 1000;
    room = (long)LZ_MUSIC_SLEW_PER_SEC * dt_ms / 1000;
    /* Never zero: at a 100 ms tick the room is 50, but a shorter tick
       or a smaller slew would round to nothing and the tempo would
       never arrive at all. */
    if (room < 1) room = 1;
    if (target > now) return (target - now <= room) ? target
                                                   : now + (int)room;
    if (target < now) return (now - target <= room) ? target
                                                   : now - (int)room;
    return now;
}

unsigned long lz_music_tempo_us(unsigned long base_us, int ratio_milli) {
    double us;
    if (base_us == 0) return 500000;
    if (ratio_milli < LZ_MUSIC_RATIO_MIN) ratio_milli = LZ_MUSIC_RATIO_MIN;
    if (ratio_milli > LZ_MUSIC_RATIO_MAX) ratio_milli = LZ_MUSIC_RATIO_MAX;
    us = (double)base_us * (double)LZ_MUSIC_RATIO_ONE / (double)ratio_milli;
    if (us < 1.0) return 1;
    if (us > (double)LZ_MUSIC_TEMPO_MAX) return LZ_MUSIC_TEMPO_MAX;
    return (unsigned long)(us + 0.5);
}

/* ---- the baseline, as text ---- */

void lz_music_format_tps(int milli, char *out, int cap) {
    int whole, frac, n;
    char buf[LZ_MUSIC_TPS_TEXT];
    if (!out || cap <= 0) return;
    if (milli < 0) milli = 0;
    if (milli > LZ_MUSIC_TPS_MAX) milli = LZ_MUSIC_TPS_MAX;
    whole = milli / 1000;
    /* ROUNDED to the two digits shown, not truncated. A box that shows
       4.19 for 4.199 and then writes 4.19 back the moment the user
       presses OK loses a little of the value every time the window is
       opened - a setting that drifts because it was looked at. */
    frac = (milli % 1000 + 5) / 10;
    if (frac >= 100) { whole++; frac = 0; }
    /* sprintf with a fixed "%d.%02d" and no float in sight: the two
       parts are integers, so there is no decimal point for a locale to
       have an opinion about. */
    sprintf(buf, "%d.%02d", whole, frac);
    n = (int)strlen(buf);
    if (n >= cap) { out[0] = '\0'; return; }
    memcpy(out, buf, (size_t)n + 1);
}

int lz_music_parse_tps(const char *s, int *milli) {
    long whole = 0, frac = 0;
    int digits = 0, any = 0;
    const char *p = s;
    if (!s || !milli) return 0;
    while (*p == ' ' || *p == '\t') p++;
    while (*p >= '0' && *p <= '9') {
        /* Stop growing rather than wrap. A pasted twenty-digit number
           is not a rate, and the overflowed one it would become could
           land anywhere in range - including somewhere plausible. */
        if (whole < 1000000L) whole = whole * 10 + (*p - '0');
        p++;
        any = 1;
    }
    if (*p == '.') {
        p++;
        while (*p >= '0' && *p <= '9') {
            if (digits < 3) { frac = frac * 10 + (*p - '0'); digits++; }
            else if (digits < 9) digits++;   /* seen, ignored */
            p++;
            any = 1;
        }
    }
    while (*p == ' ' || *p == '\t') p++;
    /* Anything left is a refusal. Reading the leading part of "4.2x"
       would accept a typo silently, and a baseline is a number the
       user cannot check by ear. */
    if (*p != '\0' || !any) return 0;
    while (digits < 3) { frac *= 10; digits++; }
    whole = whole * 1000 + frac;
    if (whole > LZ_MUSIC_TPS_MAX) whole = LZ_MUSIC_TPS_MAX;
    *milli = (int)whole;
    return 1;
}

/* ---- degrading ---- */

int lz_music_finish_from_beep(int beep) {
    return beep ? LZ_MUSIC_FINISH_CHIME : LZ_MUSIC_FINISH_NONE;
}

int lz_music_finish_effective(int finish, int engine_ok) {
    if (finish == LZ_MUSIC_FINISH_CHIME && !engine_ok)
        return LZ_MUSIC_FINISH_BEEP;
    /* BEEP is NOT re-routed the other way. It needs no engine, and a
       user who picked it picked the sound they wanted rather than the
       one that was left. */
    return finish;
}

int lz_music_engine_pick(int wanted, int system_ok, int wave_ok) {
    if (wanted == LZ_MUSIC_ENGINE_WAVE) {
        if (wave_ok) return LZ_MUSIC_ENGINE_WAVE;
        if (system_ok) return LZ_MUSIC_ENGINE_SYSTEM;
        return -1;
    }
    if (system_ok) return LZ_MUSIC_ENGINE_SYSTEM;
    if (wave_ok) return LZ_MUSIC_ENGINE_WAVE;
    return -1;
}

/* ---- the playlist ---- */

/* One entry, as a [begin, end) pair inside `list`, skipping empties and
 * trimming spaces. Returns 0 when idx is past the end.
 *
 * Written as a scan rather than with strtok because strtok writes into
 * its argument, and the argument here is the ini value the caller still
 * needs whole. */
static int list_span(const char *list, int idx, const char **beg,
                     const char **end) {
    const char *p = list;
    int n = 0;
    if (!list) return 0;
    for (;;) {
        const char *s, *e;
        while (*p == ' ' || *p == '\t') p++;
        s = p;
        while (*p && *p != LZ_MUSIC_LIST_SEP) p++;
        e = p;
        while (e > s && (e[-1] == ' ' || e[-1] == '\t')) e--;
        if (e > s) {
            if (n == idx) { *beg = s; *end = e; return 1; }
            n++;
        }
        if (!*p) return 0;
        p++;   /* past the separator */
    }
}

int lz_music_list_at(const char *list, int idx, char *out, int cap) {
    const char *b, *e;
    int n;
    if (!out || cap <= 0) return 0;
    out[0] = '\0';
    if (idx < 0) return 0;
    if (!list_span(list, idx, &b, &e)) return 0;
    n = (int)(e - b);
    /* Refuse rather than truncate. A truncated path is a path to
       something else, and the something else is usually nothing - but
       "nothing" is exactly what a missing file looks like, so the
       caller would report the wrong fault. */
    if (n >= cap) return 0;
    memcpy(out, b, (size_t)n);
    out[n] = '\0';
    return 1;
}

int lz_music_list_count(const char *list) {
    const char *b, *e;
    int n = 0;
    while (list_span(list, n, &b, &e)) n++;
    return n;
}

/* Case-insensitive compare for ASCII, written out because tolower and
 * stricmp both take a locale's view of what a letter is and neither is
 * spelled the same way on both toolchains. */
static int same_ascii(const char *a, const char *b) {
    for (; *a && *b; a++, b++) {
        int x = *a, y = *b;
        if (x >= 'A' && x <= 'Z') x += 'a' - 'A';
        if (y >= 'A' && y <= 'Z') y += 'a' - 'A';
        if (x != y) return 0;
    }
    return *a == *b;
}

int lz_music_is_loop_file(const char *name) {
    int n;
    if (!name) return 0;
    n = (int)strlen(name);
    /* The extension, not a substring search: a file called
       "mid-recording.wav" is not a MIDI file, and a scan that took it
       would hand the player something it refuses on every rotation. */
    if (n < 5 || !same_ascii(name + n - 4, ".mid")) return 0;
    return 1;
}

void lz_music_drop_pending(int *pending, int *stale) {
    if (!pending || !stale) return;
    /* ADD. Assigning loses a leftover when a piece is replaced while an
       earlier one's notification is still outstanding, and the one that
       is lost then reads as the live playback's. */
    *stale += *pending;
    *pending = 0;
}

int lz_music_own_done(int *pending, int *stale) {
    if (!pending || !stale) return 0;
    /* Leftovers first: they were queued before whatever is playing
       now, so the device returns them first too. */
    if (*stale > 0) { (*stale)--; return 0; }
    if (*pending > 0) (*pending)--;
    return 1;
}

int lz_music_next(int count, int *cursor) {
    int i;
    if (count <= 0) return -1;
    if (!cursor) return 0;
    /* A cursor from the ini, or from a list that has since got shorter,
       can be anywhere. Bring it into range rather than trusting it:
       the alternative is an index one past the end of an array. */
    if (*cursor < 0 || *cursor >= count) *cursor = 0;
    i = *cursor;
    *cursor = (i + 1) % count;
    return i;
}
