/* wave.c - see wave.h. C89. */
#include "../src/lz_int.h"
#include "wave.h"
#include "fir_table.h"

#define DRUM_CH      9      /* GM channel 10, one-based */
#define GAIN_ONE  65536L

/* THE ENVELOPE IS LINEAR IN DECIBELS and a DLS decay time is the time
   to fall 96 dB - read out of OpenMPT's CDLSBank, which computes
   decayTime = attenuation * decay / 960 with 960 being 96 dB in
   tenths. Guessing 60 dB, which was the plan, would have run every
   decay 1.6 times fast.
 *
 * So the state is an ATTENUATION in tenths of a decibel that walks
 * upward at a constant rate, and the gain is a table lookup. That is
 * both what the format says and the cheapest thing to compute: no
 * logarithm, no per-sample multiply chain, and no drift.
 *
 * It advances every ENV_STEP frames rather than every frame, because
 * an envelope is a thing that moves over milliseconds - 32 frames is
 * 0.7 ms at 44100 - and doing it per frame would cost more than the
 * mixing. */
#define ENV_STEP    32
#define ENV_FLOOR  960L     /* 96 dB down is silence */

static const long db_whole[97] = {
     65536,  58409,  52057,  46396,  41350,  36854,  32846,  29274,
     26090,  23253,  20724,  18471,  16462,  14672,  13076,  11654,
     10387,   9257,   8250,   7353,   6554,   5841,   5206,   4640,
      4135,   3685,   3285,   2927,   2609,   2325,   2072,   1847,
      1646,   1467,   1308,   1165,   1039,    926,    825,    735,
       655,    584,    521,    464,    414,    369,    328,    293,
       261,    233,    207,    185,    165,    147,    131,    117,
       104,     93,     83,     74,     66,     58,     52,     46,
        41,     37,     33,     29,     26,     23,     21,     18,
        16,     15,     13,     12,     10,      9,      8,      7,
         7,      6,      5,      5,      4,      4,      3,      3,
         3,      2,      2,      2,      2,      1,      1,      1,
         1
};
static const long db_tenth[10] = {
     65536,  64786,  64044,  63311,  62586,  61870,  61162,  60462,
     59770,  59085
};

/* Attenuation in tenths of a decibel -> a 16.16 factor.
 *
 * ONE FACTOR IS NARROWED BEFORE THE MULTIPLY, and it has to be the
 * tenths. Multiplying the two tables straight is 58409 * 65536, which
 * is 3.8e9: fine in the 64-bit long Linux gives, an overflow to a
 * negative number in the 32-bit one Windows gives, and therefore a
 * garbage gain on every attenuation of a decibel or more - on one
 * platform only. Narrowing the tenths costs four hundredths of a
 * decibel because they never fall below 59085; narrowing the whole
 * decibels instead would take the quiet end of the table to zero. */
static long db_gain_tenths(long cb)
{
    if (cb <= 0L) return GAIN_ONE;
    if (cb >= ENV_FLOOR) return 0L;
    return (db_whole[cb / 10L] * (db_tenth[cb % 10L] >> 8)) >> 8;
}

/* Tenths of a dB, 16.16, added every ENV_STEP frames so that 96 dB
   takes `ms`. Grouped to stay inside a long: the rate is divided down
   to kHz first, which costs a quarter of a percent of the time. */
static long env_rate(long ms, long rate)
{
    long ticks;
    if (ms < 1L) ms = 1L;
    ticks = ms * (rate / 1000L) / ENV_STEP;
    if (ticks < 1L) ticks = 1L;
    return (ENV_FLOOR * 65536L) / ticks + 1L;
}

/* 2^(k/12) as 16.16, and the thirteenth entry so the interpolation
   below has a right-hand neighbour for the last semitone.
 *
 * NO FLOATING POINT ANYWHERE IN THE PLAYER. Not for speed - this runs
 * once a note - but because a machine without an FPU pays for a double
 * in a library call, and because integer arithmetic is the same on
 * every compiler and every host, which is what makes the Windows and
 * Linux builds produce identical samples rather than nearly identical
 * ones.
 *
 * ONE SEMITONE IS FINE ENOUGH TO INTERPOLATE ACROSS. A straight line
 * through an exponential over an interval h is out by at most h*h/8
 * times its second derivative, which here is 4.2e-4 - seven tenths of
 * a cent, an order below anything audible and two below the 30 cents
 * the bank's own gate allows. A hundred-entry cent table would buy
 * nothing but a hundred constants to get wrong. */
static const long semi[13] = {
     65536L,   69433L,   73562L,   77936L,
     82570L,   87480L,   92682L,   98193L,
    104032L,  110218L,  116772L,  123715L,
    131072L
};

typedef struct {
    /* The zone's frames: 16-bit, little-endian, assembled a byte at a time by
       fetch(). That keeps a big-endian host reading the file correctly and
       keeps ARMv5TE from an unaligned word load, which returns a rotated word
       there without faulting. */
    const unsigned char *pcm;
    long  end;              /* one past the last frame that may be read */
    long  loop_start, loop_len;
    long  pos;              /* whole frames */
    unsigned long frac;     /* 0..65535 of a frame */
    long  step;             /* whole frames an output frame */
    unsigned long sfrac;
    long  gain;             /* 16.16, folded from velocity and the bank */
    /* Envelope: an attenuation in tenths of a decibel as 16.16, and
       the rate it is climbing at. */
    long  atten, rate_dec, rate_rel, sustain;
    long  attack_left, attack_inc, attack_env;
    int   phase;            /* 0 attack, 1 decay/sustain, 2 release */
    int   tick;             /* frames until the next envelope step */
    int   ch, note;
    int   held;             /* still down, or held by the pedal */
    /* Voice-stealing order only, not a duration - a plain count, so it
       has to wrap the same way on every host: `pick()`'s sentinel is
       the top of THIS type, not of whatever `long` is. */
    uint32_t age;
} Voice;

typedef struct {
    int program;
    int bend;               /* cents */
    int volume, expression; /* 0..127 */
    int sustain;
} Chan;

static const KsbBank *g_bank;
static long g_rate;
static Voice g_v[WAVE_VOICES];
static Chan g_ch[16];
static uint32_t g_clock;
static long g_stolen, g_unvoiced;

int wave_open(const KsbBank *bank, long rate)
{
    int i;
    if (bank == 0 || rate < 4000) return 1;
    g_bank = bank;
    g_rate = rate;
    g_clock = 0UL;
    g_stolen = g_unvoiced = 0L;
    for (i = 0; i < WAVE_VOICES; i++) g_v[i].pcm = 0;
    for (i = 0; i < 16; i++) {
        g_ch[i].program = 0;
        g_ch[i].bend = 0;
        g_ch[i].volume = 100;
        g_ch[i].expression = 127;
        g_ch[i].sustain = 0;
    }
    return 0;
}

void wave_close(void)
{
    int i;
    for (i = 0; i < WAVE_VOICES; i++) g_v[i].pcm = 0;
    g_bank = 0;
}

int wave_busy(void)
{
    int i, n = 0;
    for (i = 0; i < WAVE_VOICES; i++) if (g_v[i].pcm) n++;
    return n;
}

long wave_stolen(void)   { return g_stolen; }
long wave_unvoiced(void) { return g_unvoiced; }

/* The quietest voice, and the oldest of those. Stealing the plain
   oldest takes the bass note a piece is leaning on while a decayed
   cymbal-tail of a voice sits there inaudible. */
static int pick(void)
{
    int i, best = 0;
    long soft = 0x7FFFFFFFL;
    uint32_t old = 0xFFFFFFFFUL;
    for (i = 0; i < WAVE_VOICES; i++) {
        if (!g_v[i].pcm) return i;
    }
    for (i = 0; i < WAVE_VOICES; i++) {
        if (g_v[i].gain < soft
            || (g_v[i].gain == soft && g_v[i].age < old)) {
            soft = g_v[i].gain;
            old = g_v[i].age;
            best = i;
        }
    }
    g_stolen++;
    return best;
}

/* 2^(cents/1200), split into a mantissa in [1, 2) as 16.16 and a count
   of octaves. Kept apart because the octaves are a shift, and doing
   them as part of the mantissa would either overflow going up or throw
   away the fraction going down. */
static long pitch_mantissa(long cents, int *octave)
{
    long oct, rem;
    int k, f;

    oct = cents / 1200L;
    rem = cents - oct * 1200L;
    if (rem < 0) { rem += 1200L; oct--; }
    k = (int)(rem / 100L);
    f = (int)(rem - (long)k * 100L);
    *octave = (int)oct;
    /* The widest gap between entries is 7357, so times 99 this stays
       four orders inside a long. */
    return semi[k] + (semi[k + 1] - semi[k]) * (long)f / 100L;
}

/* dB to a 16.16 factor, for the small negative trims a bank carries.
   -6 dB is a half, and the table walks down in whole decibels; the
   bank's values are all whole or half decibels of attenuation. */
static long db_gain(int db)
{
    static const long tab[13] = {
        65536, 58409, 52057, 46396, 41350, 36854, 32846,
        29278, 26090, 23253, 20724, 18471, 16462
    };
    if (db >= 0) return GAIN_ONE;
    if (db <= -12) return tab[12] >> ((-db - 12) / 6);
    return tab[-db];
}

static void start(int ch, int note, int vel)
{
    KsbZone z;
    Voice *v;
    long cents, base, step, mant;
    int slot, oct;

    if (ksb_zone(g_bank, g_ch[ch].program, note, &z) != 0) {
        g_unvoiced++;
        return;
    }
    slot = pick();
    v = &g_v[slot];

    v->pcm = z.pcm;
    v->loop_start = z.loop_start;
    v->loop_len = z.loop_len;
    /* WHERE PLAYBACK ENDS: the end of the loop when there is one, the
       end of the sample when there is not. The frame after the last is
       the loop's first, and for an unlooped sample there is none, so
       the last frame interpolates against itself. Reading pos+1
       unconditionally would step one byte past the sample - which lands
       inside the image, so it never faults and shows up only as a tick
       at the end of every note. */
    v->end = z.loop_len > 0 ? z.loop_start + z.loop_len : z.frames;
    if (v->end < 1) { v->pcm = 0; g_unvoiced++; return; }
    v->pos = 0;
    v->frac = 0UL;

    cents = (long)(note - z.unity) * 100L + z.fine + g_ch[ch].bend;
    mant = pitch_mantissa(cents, &oct);
    /* The stored rate over the output rate, as 16.16. Done unsigned
       because a bank may hold a rate up to 65535 and 65535<<16 is
       4294901760 - inside an unsigned long by sixty thousand, outside
       a signed one by half. */
    base = (long)(((unsigned long)z.rate << 16) / (unsigned long)g_rate);
    /* base * mant >> 16, split so neither half can overflow 32 bits.
       The two truncations cost under two parts in 16.16, a hundredth
       of a cent. */
    step = ((base >> 8) * mant >> 8) + (((base & 0xFFL) * mant) >> 16);
    while (oct > 0) {
        if (step >= (64L << 16)) break;     /* past this it is noise */
        step <<= 1;
        oct--;
    }
    while (oct < 0) { step >>= 1; oct++; }
    if (step > (64L << 16)) step = 64L << 16;
    if (step < 1L) step = 1L;
    v->step = step >> 16;
    v->sfrac = (unsigned long)(step & 0xFFFFL);

    /* Velocity squared: MIDI velocity is a loudness the player feels,
       not an amplitude, and a linear map makes everything quiet sound
       the same. */
    v->gain = (long)vel * (long)vel * GAIN_ONE / (127L * 127L);
    v->gain = (v->gain >> 8) * (db_gain(z.gain) >> 8);

    v->atten = 0L;
    v->sustain = (long)z.sustain_cb << 16;
    v->rate_dec = env_rate((long)z.decay_ms, g_rate);
    v->rate_rel = env_rate((long)z.release_ms, g_rate);
    v->tick = ENV_STEP;
    if (z.attack_ms > 0) {
        v->phase = 0;
        v->attack_env = 0L;
        /* Divided down first: a 65-second attack times 48000 is 3.1e9,
           which a 32-bit long does not hold. */
        v->attack_left = (long)z.attack_ms * (g_rate / 1000L);
        if (v->attack_left < 1L) v->attack_left = 1L;
        v->attack_inc = GAIN_ONE / (v->attack_left / ENV_STEP + 1L);
    } else {
        v->phase = 1;
        v->attack_env = GAIN_ONE;
    }
    v->ch = ch;
    v->note = note;
    v->held = 1;
    v->age = ++g_clock;
}

/* Release carries on from wherever the decay had got to, rather than
   restarting at full volume - a note released after its decay has
   almost died must not jump back up. */
static void release(Voice *v)
{
    v->held = 0;
    v->phase = 2;
    v->attack_env = GAIN_ONE;
}

static void op_note_off(int ch, int note)
{
    int i;
    for (i = 0; i < WAVE_VOICES; i++) {
        if (g_v[i].pcm && g_v[i].ch == ch && g_v[i].note == note
            && g_v[i].held) {
            if (g_ch[ch].sustain) g_v[i].held = 2;   /* the pedal has it */
            else release(&g_v[i]);
        }
    }
}

static void op_note_on(int ch, int note, int vel)
{
    if (vel <= 0) { op_note_off(ch, note); return; }
    /* THE DRUM CHANNEL IS SILENT, not mapped. This bank has no
       percussion, and answering a snare with whatever melodic zone
       covers MIDI 38 is worse than nothing - it is a wrong note in
       time with the music, which is harder to ignore than a gap. */
    if (ch == DRUM_CH) { g_unvoiced++; return; }
    start(ch, note, vel);
}

static void op_program(int ch, int pgm) { g_ch[ch].program = pgm & 0x7F; }

static void op_control(int ch, int cc, int val)
{
    int i;
    switch (cc) {
    case 7:  g_ch[ch].volume = val; break;
    case 11: g_ch[ch].expression = val; break;
    case 64:
        g_ch[ch].sustain = val >= 64;
        if (!g_ch[ch].sustain) {
            for (i = 0; i < WAVE_VOICES; i++) {
                if (g_v[i].pcm && g_v[i].ch == ch && g_v[i].held == 2) {
                    release(&g_v[i]);
                }
            }
        }
        break;
    case 120: case 123:
        for (i = 0; i < WAVE_VOICES; i++) {
            if (g_v[i].pcm && (ch < 0 || g_v[i].ch == ch)) release(&g_v[i]);
        }
        break;
    default: break;
    }
}

/* A bend of 8192 is centre and the range is the GM default of two
   semitones, which is what a file assumes when it never sends RPN 0. */
static void op_pitch(int ch, int bend)
{
    g_ch[ch].bend = (bend - 8192) * 200 / 8192;
}


/* One channel message, in the shape common/smf.h hands out: status in the low
   byte, then the two data bytes. Everything this synth has an answer for is
   here; a message it has none for is dropped, which is what a sampler with no
   expression control does with expression. */
int wave_message(unsigned long msg)
{
    int ch = (int)(msg & 0x0F);
    int d1 = (int)((msg >> 8) & 0x7F);
    int d2 = (int)((msg >> 16) & 0x7F);
    switch (msg & 0xF0UL) {
    case 0x80:
        op_note_off(ch, d1);
        break;
    case 0x90:
        /* A NOTE-ON AT ZERO VELOCITY IS A NOTE-OFF, which is how half the
           MIDI in the world spells one. */
        if (d2 > 0) op_note_on(ch, d1, d2);
        else op_note_off(ch, d1);
        break;
    case 0xB0:
        op_control(ch, d1, d2);
        break;
    case 0xC0:
        op_program(ch, d1);
        break;
    case 0xE0:
        op_pitch(ch, d1 | (d2 << 7));
        break;
    default:
        break;
    }
    return 0;
}

/* ONE ACCUMULATOR, CLIPPED ONCE. Summing voices straight into the
   16-bit output clips each one against what happened to be mixed
   before it, so the result depends on voice order and distorts even
   where the total would have fitted. */
#define WAVE_BLOCK 1024

/* One frame, whichever coding it is in. Assembled byte by byte for the
   16-bit case so a big-endian host reads the file the same way. */
static long fetch(const Voice *v, long i)
{
    return (long)(short)((unsigned int)v->pcm[i * 2]
                         | ((unsigned int)v->pcm[i * 2 + 1] << 8));
}

/* The same, for an index that may be outside the sample: folded into
   the loop when there is one, clamped when there is not. Only the few
   frames at either end of a sample need this, which is why the mixing
   loop tests once for the whole window rather than per tap. */
static long fetch_wrap(const Voice *v, long i)
{
    if (i < 0) return fetch(v, 0);
    if (i >= v->end) {
        if (v->loop_len > 0) {
            return fetch(v, v->loop_start + (i - v->end) % v->loop_len);
        }
        return fetch(v, v->end - 1);
    }
    return fetch(v, i);
}

/* Eight tap windowed sinc at `pos` + `frac`. See fir_table.h: three
   and a half decibels of the top octave, on most notes, come back. */
static long interp(const Voice *v, long pos, unsigned long frac)
{
    const short *h = fir_lut[frac * FIR_PHASES >> 16];
    long acc = 0;
    int k;
    if (pos >= 3 && pos + 4 < v->end) {
        for (k = 0; k < FIR_WIDTH; k++) {
            acc += fetch(v, pos - 3 + k) * (long)h[k];
        }
    } else {
        for (k = 0; k < FIR_WIDTH; k++) {
            acc += fetch_wrap(v, pos - 3 + k) * (long)h[k];
        }
    }
    /* The coefficients sum to 32768 and the widest sum of absolute
       values over all phases is 44456, so this is at most 1.46e9 for a
       full scale input - inside a 32-bit long. */
    acc >>= FIR_SHIFT;
    if (acc > 32767L) acc = 32767L;
    if (acc < -32768L) acc = -32768L;
    return acc;
}

static void mix(long *acc, long frames)
{
    long i, s, g, cg, env;
    int j;
    Voice *v;

    for (i = 0; i < frames; i++) acc[i] = 0L;
    for (j = 0; j < WAVE_VOICES; j++) {
        v = &g_v[j];
        if (!v->pcm) continue;
        /* Channel volume and expression fold in per block rather than
           at note-on, so a fader move is heard on notes already down. */
        cg = (long)g_ch[v->ch].volume * (long)g_ch[v->ch].expression
             * GAIN_ONE / (127L * 127L);
        cg >>= 8;
        env = (db_gain_tenths(v->atten >> 16) >> 8) * (v->attack_env >> 8);
        for (i = 0; i < frames; i++) {
            s = interp(v, v->pos, v->frac);
            /* Both factors are 16.16 and neither exceeds 1.0, so the
               halves multiply back to 16.16 and s*g stays inside a
               32-bit long: 32767 * 65536 is 2147418112. */
            g = (v->gain >> 8) * cg;
            g = (g >> 8) * (env >> 8);
            acc[i] += (s * g) >> 16;

            v->frac += v->sfrac;
            v->pos += v->step + (long)(v->frac >> 16);
            v->frac &= 0xFFFFUL;
            if (v->pos >= v->end) {
                if (v->loop_len <= 0) { v->pcm = 0; break; }
                do { v->pos -= v->loop_len; } while (v->pos >= v->end);
            }
            if (--v->tick <= 0) {
                v->tick = ENV_STEP;
                if (v->phase == 0) {
                    v->attack_env += v->attack_inc;
                    if (v->attack_env >= GAIN_ONE) {
                        v->attack_env = GAIN_ONE;
                        v->phase = 1;
                    }
                } else if (v->phase == 1) {
                    if (v->atten < v->sustain) {
                        v->atten += v->rate_dec;
                        if (v->atten > v->sustain) v->atten = v->sustain;
                    }
                } else {
                    v->atten += v->rate_rel;
                }
                if (v->atten >= (ENV_FLOOR << 16)) { v->pcm = 0; break; }
                env = (db_gain_tenths(v->atten >> 16) >> 8)
                      * (v->attack_env >> 8);
            }
        }
    }
}

void wave_render(short *out, long frames)
{
    static long acc[WAVE_BLOCK];
    long i, n;

    if (!g_bank) {
        for (i = 0; i < frames; i++) out[i] = 0;
        return;
    }
    while (frames > 0) {
        n = frames < WAVE_BLOCK ? frames : WAVE_BLOCK;
        mix(acc, n);
        for (i = 0; i < n; i++) {
            long s = acc[i];
            if (s > 32767L) s = 32767L;
            if (s < -32768L) s = -32768L;
            out[i] = (short)s;
        }
        out += n;
        frames -= n;
    }
}
