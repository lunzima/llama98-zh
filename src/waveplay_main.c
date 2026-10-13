/* waveplay_main.c - play a MIDI file through the software sampler.
 *
 *   kunkun98-play [options] <file.mid> [max seconds]
 *     --bank F      the KSB bank (default banks\kk98.ksb)
 *     --rate N      output rate (default 22050)
 *     --prog N      force every channel onto GM program N
 *     --speed S     musical time per second of wall clock, in thousandths
 *     --wav F       write the rendered audio to F instead of a device
 *     --list        print what is in the bank and stop
 *
 * A second console front end, beside cli_main.c: the same reader and the
 * same sampler the window plays through, with the device's own blocking
 * write for a clock. Everything musical is in common/smf.c and common/
 * wave.c; what is left here is argument parsing, a clock loop and a way to
 * stop.
 *
 * THE DEVICE IS THE CLOCK, and that is why nothing here measures elapsed
 * time. Each pass renders a block and hands it over, and the write blocks
 * until the card has room, which is exactly how long that block takes to
 * play. No drift to correct, no sleep to be wrong about, and no difference
 * between the two platforms beyond which device got linked.
 *
 * --speed is the same quantity the window's tempo is, and moves the same
 * way: a higher value takes LESS wall time for the same music, at the pitch
 * the file was written at. It is applied where the window applies it - to
 * how many output frames a musical interval is worth - so a piece played
 * here and a piece played in the window at the same setting are the same
 * tempo.
 *
 * C99, and no floating point: --speed is a whole number of thousandths,
 * which is the window's own unit, so the one option that is not an integer
 * does not have to be one.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat40.h" /* lz_ui_wave_*: the device, resolved at run time */
#include "kac.h"
#include "ksb.h"
#include "lz_int.h"
#include "music.h"
#include "smf.h"
#include "wave.h"

/* How much is rendered and handed over at once, and it is the size of one
 * of the device's own buffers rather than something chosen for latency.
 *
 * A SMALL block is not a smaller buffer. lz_ui_wave_write waits for the
 * buffer it is about to fill to come free, so a caller that hands over
 * 64 frames at a time - 2.9 ms at 22050 - has all four of the device's
 * buffers covering 11.6 ms of audio, and every one of those 61 thousand
 * calls can wait on the 1 ms granularity of the wait itself. Playing
 * then takes LONGER than the music does, and a buffer that shallow
 * underruns on any hiccup, which is heard as crackle rather than as a
 * tempo error. One block per buffer is what the window's player does and
 * what this one does now. */
#define PLAY_SEQ_FRAMES 1024

/* A piece may run on past its last event looking for voices to die, and stops
 * at this. The bound is not stylistic: a backend that never freed a voice
 * would otherwise never stop, and a player that hangs reads as the music going
 * quiet rather than as a bug. Five seconds is longer than any release the
 * banks carry and short enough to notice. */
#define PLAY_TAIL_S 5

static unsigned char *g_bankimg; /* outlives the bank: every zone points in */
static unsigned char *g_midimg;  /* outlives the reader, same reason */

/* Non-NULL when the audio is going to a file rather than to a card. The
 * clock is the device's blocking write, and there is no device in this
 * mode - so a file run is not a way to LISTEN to a piece, it is a way to
 * find out what the player would have handed over, without waiting for
 * the piece to play. */
static FILE *g_wav;
static long  g_wavn;

static void put16(unsigned char *p, unsigned long v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
}

static void put32(unsigned char *p, unsigned long v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

/* A 16-bit mono PCM header for `frames` frames. Written twice: once as a
 * placeholder before the audio, and again at the end with the real count,
 * because how long a piece is is not known until it has been rendered. */
static void wav_head(FILE *f, long rate, long frames)
{
    unsigned char h[44];
    unsigned long data = (unsigned long)frames * 2UL;
    unsigned long r    = (unsigned long)rate;
    int           i;

    for (i = 0; i < 44; i++)
        h[i] = 0;
    memcpy(h, "RIFF", 4);
    put32(h + 4, data + 36UL);
    memcpy(h + 8, "WAVEfmt ", 8);
    put32(h + 16, 16UL);          /* PCM header size */
    put16(h + 20, 1UL);           /* WAVE_FORMAT_PCM */
    put16(h + 22, 1UL);           /* mono */
    put32(h + 24, r);
    put32(h + 28, r * 2UL);       /* bytes a second */
    put16(h + 32, 2UL);           /* block align */
    put16(h + 34, 16UL);          /* bits */
    memcpy(h + 36, "data", 4);
    put32(h + 40, data);
    fseek(f, 0L, SEEK_SET);
    fwrite(h, 1, 44, f);
}

/* Hand this block to whoever is listening: the card, or the file. */
static int dev_write(const short *pcm, long n)
{
    if (g_wav != NULL) {
        if (fwrite(pcm, sizeof(short), (size_t)n, g_wav) != (size_t)n)
            return 0;
        g_wavn += n;
        return 1;
    }
    return lz_ui_wave_write(pcm, n);
}

static unsigned char *slurp(const char *path, long *len)
{
    FILE    *f = fopen(path, "rb");
    long     n = 0;
    uint8_t *p = NULL;

    if (!f)
        return NULL;
    fseek(f, 0L, SEEK_END);
    n = ftell(f);
    fseek(f, 0L, SEEK_SET);
    if (n > 0) {
        p = (unsigned char *)malloc((size_t)n);
        if (p && fread(p, 1, (size_t)n, f) != (size_t)n) {
            free(p);
            p = NULL;
        }
    }
    fclose(f);
    *len = (p != NULL) ? n : 0;
    return p;
}

/* "1.75" -> 1750, and "2000" -> 2000: the argument is THOUSANDTHS, so a bare
 * number is already in the unit and only a decimal has to be scaled. Treating
 * both the same way is the one way to get this wrong that still looks
 * plausible -- "2000" coming back as two million plays the piece two thousand
 * times faster than asked, which is silence. */
static long parse_speed(const char *s)
{
    long whole = 0L, frac = 0L, scale = 1L;
    int  dot = 0;
    while (*s >= '0' && *s <= '9')
        whole = whole * 10L + (*s++ - '0');
    if (*s == '.') {
        dot = 1;
        s++;
        while (*s >= '0' && *s <= '9' && scale < 10000L) {
            frac = frac * 10L + (*s++ - '0');
            scale *= 10L;
        }
    }
    if (!dot)
        return whole;
    return whole * 1000L + (frac * 1000L) / scale;
}

/* What this interval of the piece is worth in output frames.
 *
 * frames = microseconds * rate / 1e6 * 1000 / speed, which reduces to
 * microseconds * rate / (1000 * speed): a thousandth of a multiplier, so a
 * LARGER number of thousandths gives FEWER frames for the same music, and
 * fewer frames is less wall time for it. Getting that backwards by a factor
 * of a thousand is not a small tempo error - every note lasts a thousand
 * times too long, a piece never reaches its end, and what is audible is one
 * note being dragged.
 *
 * The multiplication comes first and is one interval's worth: at the
 * extremes of this program's range that is 6e7 microseconds times 48000 times
 * 2000, comfortably inside a 64-bit signed, and the divisor is not folded in
 * until then. */
static lz_i64 interval_frames(const LZSmfReader *r, unsigned long ticks, long rate, long speed)
{
    lz_i64 us = (lz_i64)ticks * (lz_i64)r->us_qn0 / (lz_i64)LZ_SMF_DIVISION;
    return us * (lz_i64)rate / (LZ_I64_C(1000) * (lz_i64)speed);
}

int main(int argc, char **argv)
{
    const char    *path = NULL, *bankpath = "banks\\kk98.ksb", *wavpath = NULL;
    long           rate = 22050L, limit = 0L, banklen = 0L, imglen = 0;
    long           speed = 1000L; /* thousandths, 1000 is the written tempo */
    int            i, rc, prog = -1, list = 0;
    int            instrument = LZ_MUSIC_AS_WRITTEN, volume = 100;
    KsbBank        bank;
    LZSmfReader    rd;
    long           frames = 0, tail = 0;
    unsigned long  prev_tick = 0;
    int            over = 0;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bank") && i + 1 < argc) {
            bankpath = argv[++i];
        } else if (!strcmp(argv[i], "--rate") && i + 1 < argc) {
            rate = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--prog") && i + 1 < argc) {
            instrument = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--speed") && i + 1 < argc) {
            speed = parse_speed(argv[++i]);
        } else if (!strcmp(argv[i], "--wav") && i + 1 < argc) {
            wavpath = argv[++i];
        } else if (!strcmp(argv[i], "--list")) {
            list = 1;
        } else if (argv[i][0] == '-') {
            printf("unknown option %s\n", argv[i]);
            return 2;
        } else if (!path) {
            path = argv[i];
        } else {
            limit = atol(argv[i]);
        }
    }
    if (!path && !list) {
        printf("usage: kunkun98-play [--bank F] [--rate N] [--prog N] "
               "[--speed S] [--wav F] [--list] <file.mid> [seconds]\n");
        return 2;
    }
    if (rate < 8000L || rate > 48000L) {
        printf("--rate must be between 8000 and 48000\n");
        return 2;
    }
    if (speed <= 0L) {
        printf("--speed must be a positive number of thousandths\n");
        return 2;
    }

    g_bankimg = slurp(bankpath, &banklen);
    if (g_bankimg == NULL) {
        printf("cannot read %s\n", bankpath);
        return 1;
    }
    rc = ksb_open(&bank, g_bankimg, banklen);
    if (rc != KSB_OK) {
        printf("%s: %s\n", bankpath, ksb_errstr(rc));
        return 1;
    }
    if (list) {
        printf("%s: %ld bytes, %d instruments\n", bankpath, banklen,
               ksb_count(&bank));
        for (i = 0; i < ksb_count(&bank); i++) {
            int  p  = ksb_program(&bank, i);
            long fr = ksb_pcm_frames(&bank, p);
            printf("  program %-3d  %ld frames expand, %.0f KiB\n", p, fr,
                   fr * 2.0 / 1024.0);
        }
        return 0;
    }

    g_midimg = slurp(path, &imglen);
    if (g_midimg == NULL) {
        printf("cannot open %s\n", path);
        return 1;
    }
    /* AS WRITTEN is resolved BEFORE the reader opens, because the bank is
       expanded for one instrument and that instrument has to be known before
       the first note. The reader's own forcing then keeps the rest of the
       file on it, so a program change part way through cannot walk the piece
       onto samples that were never decoded. */
    if (instrument == LZ_MUSIC_AS_WRITTEN) {
        static LZSmfReader scratch;
        instrument = lz_smf_opening_program(&scratch, g_midimg, imglen);
    }
    if (lz_smf_open(&rd, g_midimg, imglen, instrument, volume) != LZ_SMF_OK) {
        printf("%s: not a MIDI file this player can read\n", path);
        return 1;
    }
    prog = instrument;

    {
        long   want = ksb_pcm_frames(&bank, prog);
        short *pcm  = NULL;
        if (want > 0) {
            pcm = (short *)malloc((size_t)want * sizeof(short));
            if (pcm == NULL) {
                printf("out of memory for %ld frames\n", want);
                return 1;
            }
            rc = ksb_unpack(&bank, pcm, prog);
            if (rc != KSB_OK) {
                printf("%s: %s\n", bankpath, ksb_errstr(rc));
                return 1;
            }
            printf("expanded %ld frames (%.0f KiB) for program %d\n", want,
                   want * 2.0 / 1024.0, prog);
        } else {
            printf("program %d is stored uncompressed, nothing to expand\n",
                   prog);
        }
    }
    if (wave_open(&bank, rate) != 0) {
        printf("wave_open failed\n");
        return 1;
    }
    if (wavpath != NULL) {
        g_wav = fopen(wavpath, "wb");
        if (g_wav == NULL) {
            printf("cannot write %s\n", wavpath);
            return 1;
        }
        wav_head(g_wav, rate, 0L);
        printf("%s -> %s at %ld Hz, %s, speed %ld/1000\n", path, wavpath,
               rate, bankpath, speed);
    } else {
        if (!lz_ui_wave_open(rate)) {
            printf("no sound device at %ld Hz\n", rate);
            return 1;
        }
        printf("%s -> waveOut at %ld Hz, %s, speed %ld/1000\n", path, rate,
               bankpath, speed);
    }

    for (;;) {
        LZSmfEvent ev;
        lz_i64     want;
        long       n;
        if (!over && lz_smf_next(&rd, &ev) == 0) {
            over = 1;
        }
        if (over) {
            /* The last event has gone out; what is left is the release. */
            static short blk[PLAY_SEQ_FRAMES];
            long         step = PLAY_SEQ_FRAMES;
            if (limit > 0L && frames + step > limit * rate)
                step = limit * rate - frames;
            if (step <= 0)
                break;
            wave_render(blk, step);
            if (!dev_write(blk, step))
                goto stop;
            frames += step;
            if (wave_busy() == 0)
                break;
            tail += step;
            if (tail >= (long)PLAY_TAIL_S * rate)
                break;
            continue;
        }
        want = interval_frames(&rd, ev.tick - prev_tick, rate, speed);
        prev_tick = ev.tick;
        while (want > 0) {
            static short blk[PLAY_SEQ_FRAMES];
            n = PLAY_SEQ_FRAMES;
            if (n > (long)want)
                n = (long)want;
            if (limit > 0L && frames + n > limit * rate)
                n = limit * rate - frames;
            if (n <= 0)
                break;
            wave_render(blk, n);
            if (!dev_write(blk, n))
                goto stop;
            frames += n;
            want -= n;
        }
        wave_message(ev.msg);
        if (limit > 0L && frames >= limit * rate)
            break;
    }
stop:
    if (g_wav != NULL) {
        wav_head(g_wav, rate, g_wavn);
        fclose(g_wav);
    } else {
        lz_ui_wave_close();
    }
    wave_close();
    printf("%ld frames, %ld.%02ld s; voices left %d, stolen %ld, unvoiced %ld\n",
           frames, frames / rate, (frames % rate) * 100L / rate, wave_busy(),
           wave_stolen(), wave_unvoiced());
    free(g_midimg);
    free(g_bankimg);
    return 0;
}
