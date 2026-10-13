/* wave.c - see wave.h. C89. */
#include "../src/lz_int.h"
#include "wave.h"
#include "fir_table.h"
#include "pow2_table.h"

#define DRUM_CH      9      /* GM channel 10, one-based */
#define GAIN_ONE  65536L

/* An SF2 absolute-cents frequency, as 16.16. A DLS mod-LFO rate is
   passed straight to an SF2 generator and that generator counts cents
   from A440 at 6900, so 8.175799 * 2^(tc/1200) Hz - 8.175799, not 1.
   535809 is it in 16.16. Measured: gm.dls's -783 cents is 5.201 Hz. */
#define ABS_CENTS_TO_HZ  535809L

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
    /* THE MOD LFO, which is what moves the pitch while a note sounds.
       The phase is 32 bit and a frame adds lfo_inc to it, so the rate
       is the top byte of the phase per frame and does not depend on
       buffer size. lfo_pitch is the DEPTH in cents at full swing - 48
       on gm.dls's piano, 725 on its leads - and base_step is the step
       the note would have with no modulation at all, so a depth of
       zero leaves the pitch exactly where the static parameters put it. */
    unsigned long lfo_phase;
    long  lfo_inc;
    long  lfo_pitch;
    long  lfo_delay;        /* frames until the LFO starts at all */
    long  base_step;
    /* The two channel gains, as 16.16 and set once per note. They are
       GAIN_ONE when the region's pan is 0, which is every region in
       this bank - the ramp only runs as a note moves off centre. */
    long  gl, gr;
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
/* 2^(cents/1200) in 16.16, which is what a DLS pitch offset or a DLS
   time offset IS. The bank states them in cents - 1200 to the octave -
   and every modulation this file has to apply needs one of them: a
   vibrato depth is cents of pitch, a key's effect on decay is cents of
   time, and the mod LFO's own rate is time cents to a frequency.

   Octaves are peeled off by integer division and the remainder is one
   lookup, so it costs a division and an array read - not a logarithm,
   which is the thing this file does not have. A quarter of a percent
   worst case: measured against the exact value, 48 cents comes out
   1.027451 against 1.028114 and 725 comes out 1.517365 against 1.5201,
   both of which are under one cent. */
static long cents_gain(int cents)
{
    int oct, rem;
    long g;

    if (cents <= -12000) return 0L;
    if (cents >= 12000) return 65536L;
    oct = cents / 1200;
    rem = cents - oct * 1200;
    if (rem < 0) { oct--; rem += 1200; }
    g = pow2_frac[rem * 256 / 1200];
    if (oct > 0)
        g <<= oct;
    else if (oct < 0)
        g >>= -oct;
    return g;
}

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

/* Where a note sits between the channels: -500 hard left, 0 centre,
   +500 hard right, the unit being tenths of a percent.

   ONLY THE SIDE A NOTE MOVES TOWARD IS RAMPED. The usual form
   normalises the pair to sum to 1.0, but that leaves a centred note at
   0.5 and 0.5 - three decibels below the same note played down one
   channel - and every region gm.dls carries has a pan of 0, so that
   normalisation would quietly have taken three decibels off the whole
   bank in order to move a note the bank never moves. Centre stays full
   in both; the ramp starts only when a region asks for it. */
/* The mod LFO's own shape: 1 - 4*|phase - 0.5|, i.e. a triangle that
   runs -1 to +1 and back. It is what the reference player uses, and a
   triangle rather than a sine matters only in that the two disagree at
   every point except the peaks - a bank with a 5 Hz vibrato hears a
   difference a sine would round off at its turning points.

   `phase` is 32 bit for the whole cycle and the result runs
   -32768..32768, which the caller scales by the depth. */
static long lfo_tri(unsigned long phase)
{
    long p = (long)(phase >> 17);          /* one cycle as 0..32767 */
    long d = (p > 16384L) ? (p - 16384L) : (16384L - p);
    return 32768L - 4L * d;                /* -32768 at the ends, + at the top */
}

static void pan_gains(int pan, long *gl, long *gr)
{
    long full = GAIN_ONE;
    int  p = pan;

    if (p < -500) p = -500;
    if (p > 500) p = 500;
    if (p >= 0) {
        *gl = full * (500L - p) / 500L;
        *gr = full;
    } else {
        *gl = full;
        *gr = full * (500L + p) / 500L;
    }
}

static void start(int ch, int note, int vel)
{
    KsbZone z;
    Voice *v;
    long cents, base, step, mant, dec, atk;
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
    /* THE MOD LFO. Its rate is stated in the same cents the envelope is,
       so cents_gain turns it into a frequency in 16.16, and an increment
       is the phase one frame is worth - that frequency times 2^16 over
       the sample rate. The delay is read the same way into a frame
       count, and it counts down before the phase starts moving, so a
       0.33 s delay really is 0.33 s of a note without vibrato in it.
       A depth of zero leaves the step exactly where the static
       parameters put it, which is what the leads and the harp ask for
       by having no modLfo -> pitch connection at all. */
    v->base_step = step;
    v->lfo_pitch = z.lfo_pitch_tc;
    /* A QUARTER TURN, not zero. The reference player starts the LFO at
       phase 0.25 and a triangle reads 0 there, so the note opens in
       tune and the first thing it does is swing - starting at zero
       would open it at full depth and then fall. */
    v->lfo_phase = 0x40000000UL;
    v->lfo_inc = 0L;
    v->lfo_delay = 0L;
    if (z.lfo_pitch_tc != 0) {
        /* THE RATE IS AN ABSOLUTE PITCH, not a plain ratio. A DLS
           mod-LFO frequency is passed straight to an SF2 generator,
           and SF2's frequency generator is in absolute cents where
           6900 is A440 - so the Hz is 440 * 2^((tc-6900)/1200), which
           is 8.1758 * 2^(tc/1200) and NOT 2^(tc/1200). Getting that
           wrong is a factor of 8.18: gm.dls's -783 cents is a 5.2 Hz
           vibrato, not the 0.63 Hz a plain ratio gives, and that error
           is the harder kind to catch - counting a period from a pitch
           trace still returns a self-consistent number, and it is only
           checked against what the file says that it fails. */
        long freq = cents_gain(z.lfo_tc);
        freq = (long)(((lz_i64)freq * ABS_CENTS_TO_HZ) >> 16);
        if (freq > 0)
            v->lfo_inc = (long)(((lz_i64)freq << 16) / (lz_i64)g_rate);
        v->lfo_delay = (long)((lz_i64)cents_gain(z.lfo_delay_tc)
                              * (lz_i64)g_rate / 65536L);
        if (v->lfo_delay < 0L) v->lfo_delay = 0L;
    }
    pan_gains(z.pan, &v->gl, &v->gr);

    /* Velocity squared: MIDI velocity is a loudness the player feels,
       not an amplitude, and a linear map makes everything quiet sound
       the same. */
    v->gain = (long)vel * (long)vel * GAIN_ONE / (127L * 127L);
    v->gain = (v->gain >> 8) * (db_gain(z.gain) >> 8);

    v->atten = 0L;
    v->sustain = (long)z.sustain_cb << 16;
    /* THE ENVELOPE AS THE REGION MODULATES IT, not as the static
       parameters alone state it.

       keyNum -> decay appears 235 times on gm.dls and is the reason a
       low piano note rings while a high one dies: -3979 time cents
       across the key range is a factor of ten between the two ends of
       the keyboard. Taking the static decay alone plays every note at
       one length, which is not what the instrument does - the top of
       the keyboard reads a tenth of its own decay time.

       velocity -> attack is the same thing on the way in, 27 of them.

       The conversion needs no logarithm: adding cents to a time on one
       side is multiplying that time on the other, because
       2^(a+b) = 2^a * 2^b. So one multiply through cents_gain. The
       sources are note/127 and vel/127 - not bipolar, which is what
       usTransform says for all seven modulator pairs in this bank. */
    dec = (long)z.decay_ms;
    atk = (long)z.attack_ms;
    if (z.key_decay_tc) {
        int extra = (int)((long)z.key_decay_tc * note / 127L);
        dec = dec * cents_gain(extra) >> 16;
        if (dec < 1L) dec = 1L;
    }
    if (z.vel_attack_tc && atk > 0L) {
        int extra = (int)((long)z.vel_attack_tc * vel / 127L);
        atk = atk * cents_gain(extra) >> 16;
        if (atk < 1L) atk = 1L;
    }
    v->rate_dec = env_rate(dec, g_rate);
    v->rate_rel = env_rate((long)z.release_ms, g_rate);
    v->tick = ENV_STEP;
    if (atk > 0) {
        v->phase = 0;
        v->attack_env = 0L;
        /* Divided down first: a 65-second attack times 48000 is 3.1e9,
           which a 32-bit long does not hold. */
        v->attack_left = atk * (g_rate / 1000L);
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

static void mix(long *accL, long *accR, long frames)
{
    long i, s, g, cg, env, vg, eg;
    int j;
    Voice *v;

    for (i = 0; i < frames; i++) {
        accL[i] = 0L;
        accR[i] = 0L;
    }
    for (j = 0; j < WAVE_VOICES; j++) {
        v = &g_v[j];
        if (!v->pcm) continue;
        /* Channel volume and expression fold in per block rather than
           at note-on, so a fader move is heard on notes already down.
           BOTH ARE SQUARED, and that is the standard rather than a
           taste: the recommended curve for controller 7 is
           40*log10(cc/127) dB, which is (cc/127) SQUARED as an
           amplitude, and expression is on the same one. Velocity is
           squared at note-on for the same reason. A note's level is
           therefore the square of the product of the three, which is
           what a system synthesiser playing the same events arrives at
           - and the level is what has to agree before anything else
           about the two can be compared. */
        vg = (long)g_ch[v->ch].volume * (long)g_ch[v->ch].volume;
        eg = (long)g_ch[v->ch].expression * (long)g_ch[v->ch].expression;
        /* 16.16 down to the 8.8 that the mix below multiplies in, and
           then the two 8.8 factors back to one. */
        cg = (vg * GAIN_ONE / (127L * 127L)) >> 8;
        eg = (eg * GAIN_ONE / (127L * 127L)) >> 8;
        cg = (cg * eg) >> 8;
        /* AND NOT THE MASTER FACTOR. A reference implementation of this
           same bank - SpessaSynth - multiplies everything it renders by
           0.6, SPESSASYNTH_GAIN_FACTOR, so that it "does not stay too
           loud". It was copied here for a while and it was wrong:
           capturing the two players through the device's own loopback
           and comparing them shows our signal 21 dB below the system's
           while our noise floor is 7 dB LOWER, so every "hiss" reported
           was our signal sitting too close to a fixed floor rather than
           anything added. 0.6 is a level choice one synthesiser makes,
           not something in the format, and the thing being matched here
           is the machine's own synthesiser, which does not make it. */
        cg = (cg * 65536L) >> 16;
        env = (db_gain_tenths(v->atten >> 16) >> 8) * (v->attack_env >> 8);
        for (i = 0; i < frames; i++) {
            s = interp(v, v->pos, v->frac);
            /* Both factors are 16.16 and neither exceeds 1.0, so the
               halves multiply back to 16.16 and s*g stays inside a
               32-bit long: 32767 * 65536 is 2147418112. */
            g = (v->gain >> 8) * cg;
            g = (g >> 8) * (env >> 8);
            /* THE SAME VOICE INTO BOTH CHANNELS, each scaled by its own
               pan gain - which is GAIN_ONE on both sides for every
               region this bank carries, so a note that is not panned
               comes out of both channels at the level the single-channel
               mix gave it and the loudness does not move. Halving g
               gives 8.8 and so does the pan, and their product is the
               16.16 the old single accumulator held: 32767 * 65536 is
               2147418112, still inside a signed long. */
            accL[i] += (s * ((g >> 8) * (v->gl >> 8))) >> 16;
            accR[i] += (s * ((g >> 8) * (v->gr >> 8))) >> 16;

            v->frac += v->sfrac;
            v->pos += v->step + (long)(v->frac >> 16);
            v->frac &= 0xFFFFUL;
            /* THE LFO'S PHASE, once a frame. While there is a delay to
               count it stays at zero and sin_tbl[0] is 0, so the pitch
               comes out exactly as the static parameters said - which is
               the same as having no LFO at all. */
            if (v->lfo_delay > 0) {
                v->lfo_delay--;
            } else if (v->lfo_inc != 0) {
                v->lfo_phase += (unsigned long)v->lfo_inc;
            }
            if (v->pos >= v->end) {
                if (v->loop_len <= 0) { v->pcm = 0; break; }
                do { v->pos -= v->loop_len; } while (v->pos >= v->end);
            }
            if (--v->tick <= 0) {
                v->tick = ENV_STEP;
                /* THE PITCH THE LFO ASKS FOR, applied on the envelope's
                   own step rather than every frame: 32 frames is 1.45 ms
                   at 22050 and an LFO turning at 0.6 Hz does not travel
                   enough in that to be heard as a step. The depth is in
                   cents and the swing is the oscillator scaled to
                   +-32767, so the result is a cent count that
                   cents_gain turns into the same kind of multiplier the
                   envelope times use. The product is 32.32 and needs a
                   64-bit middle - 65536 for a step of 1.0 times a
                   2.0 gain is 8.6e9, which a 32-bit long is not. */
                if (v->lfo_pitch != 0 && v->lfo_inc != 0) {
                    int sc = (int)((long)v->lfo_pitch
                                   * lfo_tri(v->lfo_phase) >> 15);
                    long st = (long)(((lz_i64)v->base_step
                                      * cents_gain(sc)) >> 16);
                    v->step = st >> 16;
                    v->sfrac = (unsigned long)(st & 0xFFFFL);
                }
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
    static long accL[WAVE_BLOCK], accR[WAVE_BLOCK];
    long i, n;

    /* STEREO, interleaved left then right: `frames` FRAMES is 2*frames
       short samples. A DLS's reverb, chorus and pan are all stereo
       effects, so the player cannot be a single channel if it is ever
       to carry them - and it is not the mixer's job to decide where the
       channel count is, because the same wavetable drives the console
       player, the window and the file writer. */
    if (!g_bank) {
        for (i = 0; i < frames * 2; i++)
            out[i] = 0;
        return;
    }
    while (frames > 0) {
        n = frames < WAVE_BLOCK ? frames : WAVE_BLOCK;
        mix(accL, accR, n);
        for (i = 0; i < n; i++) {
            long s = accL[i];
            if (s > 32767L) s = 32767L;
            if (s < -32768L) s = -32768L;
            out[2 * i] = (short)s;
            s = accR[i];
            if (s > 32767L) s = 32767L;
            if (s < -32768L) s = -32768L;
            out[2 * i + 1] = (short)s;
        }
        out += 2 * n;
        frames -= n;
    }
}
