/* wave.c - see wave.h. C89. */
#include "../src/lz_int.h"
#include "wave.h"
#include "fir_table.h"
#include "pow2_table.h"
#include "pan_table.h"

#define DRUM_CH      9      /* GM channel 10, one-based */
#define DRUM_PROGRAM 128    /* the percussion kit's KSB program */
#define GAIN_ONE  65536L

/* An SF2 absolute-cents frequency, as 16.16. A DLS mod-LFO rate is
   passed straight to an SF2 generator and that generator counts cents
   from A440 at 6900, so 8.175799 * 2^(tc/1200) Hz - 8.175799, not 1.
   535809 is it in 16.16. Measured: gm.dls's -783 cents is 5.201 Hz. */
#define ABS_CENTS_TO_HZ  535809L

/* THE ENVELOPE IS LINEAR IN DECIBELS and a DLS decay time is the time
   to fall 96 dB - read out of OpenMPT's CDLSBank, which computes
   decayTime = attenuation * decay / 960 with 960 being 96 dB in
   tenths. Taking the decay as 60 dB would run every one 1.6 times
   fast.
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

/* A BOUNDED BOOST, +0.1 to +30 dB. A DLS gain is an attenuation and the
   reference player clamps a positive one to full scale; this engine uses
   the positive side so a bank's quiet program can be lifted to the level
   the arrangement expects. The tables are the reciprocal of the
   attenuation ones and the arithmetic is the same integer multiply, so
   the boost is bit-identical across compilers like every other gain.

   THE CEILING IS THE TABLE'S LENGTH, NOT AN ARITHMETIC LIMIT. It was
   12 dB, and p92's keys 120-127 sat 1.6 to 12.6 dB under gm there: one
   sample at root 81 is played as far as 46 semitones up, the level
   falls with the pitch distance, and the region gain that would have
   answered it asked for 24.6 dB and got 12. The tone and the envelope
   measured equal to gm's throughout (centroid 0.975-0.999, duration
   1.00) - a pure level difference, so gain is the right knob and the
   table was the only thing in the way.

   40 dB IS WHERE THE ARITHMETIC BECOMES THE LIMIT: the widest product
   this code makes is db_boost_whole[g/10] * (db_boost_tenth[g%10] >> 8),
   which at 40 dB is 6553600 * 256 = 1.68e9 against a signed long's
   2.1e9; one more decade of table and it overflows. A zone that needs
   more than this is not short of gain, it is short of a source sample
   nearer its pitch - see the root-note spread this bank was selected
   with. */
#define BOOST_MAX 400L
static const long db_boost_whole[41] = {
     65536,   73533,   82505,   92572,  103868,  116541,
     130762,  146717,  164619,  184706,  207243,  232531,
     260904,  292739,  328458,  368536,  413504,  463959,
     520571,  584090,  655360,  735326,  825049,  925721,
    1038676, 1165413, 1307615, 1467168, 1646190, 1847055,
    2072430, 2325305, 2609035, 2927386, 3284581, 3685360,
    4135042, 4639593, 5205710, 5840902, 6553600
};
static const long db_boost_tenth[10] = {
     65536,  66295,  67063,  67839,  68625,  69419,  70223,
     71036,  71859,  72691
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

/* The region's gain in tenths of a decibel, negative for attenuation and
   positive for boost, as a 16.16 factor. */
static long db_gain_of(long g)
{
    if (g < 0L) return db_gain_tenths(-g);
    if (g == 0L) return GAIN_ONE;
    if (g > BOOST_MAX) g = BOOST_MAX;
    return (db_boost_whole[g / 10L] * (db_boost_tenth[g % 10L] >> 8)) >> 8;
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

/* A stage that may be absent, as a count of envelope steps. Unlike
   env_rate this does not floor at one: a delay or hold of zero is a
   stage the bank never declared, not one that lasts a millisecond. */
static long ms_steps(long ms, long rate)
{
    if (ms <= 0L) return 0L;
    return ms * (rate / 1000L) / ENV_STEP;
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
       the rate it is climbing at. The stage is 0 delay, 1 attack,
       2 hold, 3 decay, 4 sustain and 5 release; a delay keeps
       attack_env at zero, which is silence. */
    long  atten, rate_dec, rate_rel, sustain;
    long  attack_left, attack_inc, attack_env;
    long  vdelay_left, vhold_left;  /* envelope steps left in each stage */
    int   phase;
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
    long  wheel_pitch;      /* a modLfo + CC1 depth, cents: the vibrato
                               LFO's CC1 depth when the bank declares no
                               vibratoLfo + CC1 amount of its own */
    long  lfo_delay;        /* frames until the LFO starts at all */
    /* A SECOND OSCILLATOR for the vibrato LFO, which is separate from
       the mod LFO above. Same triangle and same quarter-turn start, but
       its own rate and delay, and its depths are its own: a free depth,
       the CC1-scaled one the bank declares, then the two 50-cent
       defaults that CC1 and channel pressure carry. */
    unsigned long vib_phase;
    long  vib_inc;
    long  vib_free;
    long  vib_wheel;
    long  vib_delay;        /* frames until the vibrato LFO starts */
    long  base_step;
    /* THE LOW PASS FILTER, after the interpolator and before the gain
       (the order the format's spec states). The cutoff is smoothed
       towards the region's own at the envelope's update rate and the
       coefficients are recomputed only when the modulated cutoff moves
       by more than a cent, so a region with no modulation pays for one
       calculation per note. The coefficients are Q30 and the history Q15
       - see filt_coeffs. */
    long  filt_init;        /* the region's cutoff, absolute cents */
    long  filt_fc;          /* the smoothed cutoff, cents */
    long  filt_last;        /* the cutoff the coefficients are for */
    long  filt_q;           /* resonance, centibels */
    long  filt_env;         /* modEnv -> cutoff, cents at full level */
    long  filt_vel;         /* velocity -> cutoff, cents */
    long  filt_key;         /* keyNum -> cutoff, cents */
    long  filt_lfo;         /* modLFO -> cutoff, cents */
    long  filt_vib;         /* vibrato LFO -> cutoff, cents */
    long  f_b0, f_b1;       /* numerator, Q30; b0 == b2 */
    long  f_d1, f_d2;       /* feedback, Q30: -a1/a0 and a2/a0 */
    long  fx1, fx2;         /* input history, Q15 */
    long  fy1, fy2;         /* output history, Q15 */
    int   filt_on;          /* 1: the filter is applied this step */
    int   filt_any;         /* 0: nothing can ever move it, so no work */
    int   fir_idx;          /* which interpolation row this step needs */
    int   filt_first;       /* the first step opens at the region's own */
    int   vel;              /* note-on velocity, a filter source */
    /* THE MODULATION ENVELOPE, which drives pitch and is its own thing
       from the volume envelope beside it. Same four stages, same units
       it was read in: the level it holds is a 16.16 fraction of the
       depth, so 0 is the note at rest and 65536 is the depth applied
       in full. The bank gives it attack 0 and sustain 960 on every
       instrument that has one, which is "no modulation held". */
    long  env_pos;          /* 16.16, the level reached */
    long  env_sus;          /* the level it holds, 16.16 */
    long  env_att_inc;      /* frames of attack: how far one step rises */
    long  env_dec_inc;      /* frames of decay: how far one step falls */
    long  env_rel_inc;      /* frames of release: ditto */
    long  mdelay_left, mhold_left; /* envelope steps left in each stage */
    int   env_phase;        /* 0 delay, 1 attack, 2 hold, 3 decay,
                               4 sustain, 5 release */
    /* THE PITCH DEPTH of that envelope, in cents at full level. Only
       five regions of gm.dls carry a non-zero one, all at 327 - six
       semitones - and all on the sawtooth lead. */
    long  env_pitch;
    long  key_pitch;
    /* THE TREMOLO DEPTH, in centibels at full swing, from modLfo -> gain.
       gm.dls sets it on 7 regions at +1 cB and FF8 on 7 at the same, so
       it is a small colouring and not a gating effect - but the field is
       in the record either way, and a player that never reads it cannot
       play a bank that does set it. */
    long  lfo_gain;
    /* THE REGION'S PAN, kept rather than the two channel gains it turns
       into, because the channel's own pan is summed with it when those
       gains are worked out and that one can move under a held note.
       Centre is full in both, so a region at pan 0 does not move the
       level - see pan_gains. */
    long  region_pan;
    int   ch, note;
    int   held;             /* still down, or held by the pedal */
    /* WHICH NOTE-ON STARTED IT. One message may start several voices -
       a bank that layers regions over one key gives each layer its own -
       and they share this id so that the matching Note Off releases the
       whole layer together. */
    uint32_t note_id;
    int   key_group;        /* exclusive class; 0 is no group */
    int   region_id;        /* which region started it */
    /* Voice-stealing order only, not a duration - a plain count, so it
       has to wrap the same way on every host: `pick()`'s sentinel is
       the top of THIS type, not of whatever `long` is. */
    uint32_t age;
} Voice;
/* Which interpolation row a step needs - defined next to interp,
   declared here because the step is set long before it. */
static int fir_pick(unsigned long step);


typedef struct {
    int program;
    /* THE BANK SELECT, CC0 for the MSB and CC32 for the LSB, in the same
       split a DLS instrument header stores: its bank's bits 0-6 are the
       LSB and bits 8-14 the MSB. Both open at 0, which is the GM set, and
       both persist across program changes; CC121 does not reset them,
       because they are a patch selector rather than an expression
       control. */
    int bank_msb, bank_lsb;
    int bend;               /* cents */
    int volume, expression; /* 0..127 */
    int sustain;
    /* CC10 is the channel's own pan, added to the region's - the
       reference player sums the two - and CC1 is the modulation wheel,
       whose default modulator scales the vibrato depth. */
    int pan;                /* -500..500, 0 is centre */
    int wheel;              /* 0..127, 0 is no vibrato added */
    int pressure;           /* channel pressure, 0..127 */
    /* THE PITCH-BEND RANGE THE RPN ASKS FOR, in cents. The GM default is
       two semitones, and a file that never sends RPN 0 gets that. The
       RPN number and its 14-bit value are assembled from the two pairs
       of CCs that carry them. */
    int bend_range_cents;
    int rpn_msb, rpn_lsb;
    int data_msb, data_lsb;
} Chan;

static const KsbBank *g_bank;
static long g_rate;
/* How far one envelope step closes the gap between the smoothed and the
   region's own cutoff, in 16.16. The reference smooths once per 128-frame
   block at 0.03 and this runs once per ENV_STEP of 32, so a quarter of it
   keeps the same time constant; the rate term is the reference's own,
   because its smoothing is stated per block rather than per second. */
static long g_filt_smooth;
static Voice g_v[WAVE_VOICES];
static Chan g_ch[16];
static uint32_t g_clock;
/* The last note-event id handed out. It is compared only among the voices
   one channel+pitch still holds, so the 32-bit wrap is harmless: a note-on
   is separated from every other by at most the span of the pool, and the
   comparison is never a magnitude between distant ids. */
static uint32_t g_note_seq;
static long g_stolen, g_unvoiced;

int wave_open(const KsbBank *bank, long rate)
{
    int i;
    if (bank == 0 || rate < 4000) return 1;
    g_bank = bank;
    g_rate = rate;
    g_clock = 0UL;
    g_note_seq = 0UL;
    g_stolen = g_unvoiced = 0L;
    /* 0.03 * (44100 / rate) / 4, in 16.16, where the four is ENV_STEP
       against the reference's 128-frame block. */
    g_filt_smooth = (long)(((lz_i64)44100L * 3L * 65536L)
                           / ((lz_i64)rate * 400L));
    if (g_filt_smooth > 65536L) g_filt_smooth = 65536L;
    if (g_filt_smooth < 1L) g_filt_smooth = 1L;
    for (i = 0; i < WAVE_VOICES; i++) g_v[i].pcm = 0;
    for (i = 0; i < 16; i++) {
        g_ch[i].program = i == DRUM_CH ? DRUM_PROGRAM : 0;
        g_ch[i].bank_msb = 0;
        g_ch[i].bank_lsb = 0;
        g_ch[i].bend = 0;
        g_ch[i].volume = 100;
        g_ch[i].expression = 127;
        g_ch[i].sustain = 0;
        /* Centre pan, no wheel, and the GM default bend range: a file
           that never sends RPN 0 assumes two semitones. */
        g_ch[i].pan = 0;
        g_ch[i].wheel = 0;
        g_ch[i].pressure = 0;
        g_ch[i].bend_range_cents = 200;
        /* RPN null, which is what a channel is between selections. */
        g_ch[i].rpn_msb = 127;
        g_ch[i].rpn_lsb = 127;
        g_ch[i].data_msb = 0;
        g_ch[i].data_lsb = 0;
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

/* 2^(cents/1200) in 16.16, which is what a DLS pitch offset or a DLS
   time offset IS. The bank states them in cents - 1200 to the octave -
   and every modulation this file has to apply needs one of them: a
   vibrato depth is cents of pitch, a key's effect on decay is cents of
   time, and the mod LFO's own rate is time cents to a frequency.

   Octaves are peeled off by integer division and the remainder is one
   lookup, so it costs a division and an array read - not a logarithm,
   which is the thing this file does not have.

   THE TABLE IS INTERPOLATED. It holds 256 entries over an octave, so
   a nearest-entry read is 4.69 cents a step and 2.34 cents worst
   case - and this is not only a time: bend, keyNum -> pitch, the mod
   LFO's depth and the pitch envelope all come through here, so that
   error lands on the played pitch whenever any of them is active. The
   old note claimed under a cent and its own 725-cent example - 1.517365
   against 1.5201 - is 3.1 cents. A linear interpolation between the
   two entries costs one multiply and leaves under five thousandths. */
static long cents_gain(int cents)
{
    int oct, rem, idx;
    long g, pos, frac, a, b;

    if (cents <= -12000) return 0L;
    if (cents >= 12000) return 65536L;
    oct = cents / 1200;
    rem = cents - oct * 1200;
    if (rem < 0) { oct--; rem += 1200; }
    pos = (long)rem * 256L;
    idx = (int)(pos / 1200L);
    frac = pos - (long)idx * 1200L;
    a = pow2_frac[idx];
    b = (idx < 255) ? pow2_frac[idx + 1] : 131072L;
    g = a + (b - a) * frac / 1200L;
    if (oct > 0)
        g <<= oct;
    else if (oct < 0)
        g >>= -oct;
    return g;
}

/* A TIME SCALED BY A MODULATOR, its source a note or velocity running
   0..127 in time cents. Six call sites used to repeat this arithmetic
   with its 64-bit multiply and its floor at 1 ms; one helper means one
   place to get the overflow right (a 65-second time at full gain is
   4.3e9, outside a 32-bit long). */
static long scale_ms(long ms, int tc, int src)
{
    int extra;
    if (tc == 0 || ms <= 0L) return ms;
    extra = (int)((long)tc * src / 128L);
    ms = (long)(((lz_i64)ms * cents_gain(extra)) >> 16);
    return ms < 1L ? 1L : ms;
}

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

/* THE LOW PASS FILTER: two poles, the coefficients the reference player
 * computes, in integer arithmetic.
 *
 * The reference turns the cutoff into w = 2*pi*fc/fs and takes the RBJ
 * biquad with alpha = sin(w)/(2*resonanceGain), where the resonance gain
 * is 10^((Q - 3.01)/200) - the 3.01 is a peak offset in CENTIBELS, which
 * is what that player's code says even though its comment calls it dB.
 * qGain is 1/sqrt(10^(-Q/200)), the level the format asks the filter to
 * lose, and b0 = b1/2 = (1-cos(w))*qGain/2.
 *
 * EVERY COEFFICIENT IS Q30 AND THE HISTORY IS Q15. 16.16 is not enough
 * here: at the lowest cutoff a bank may state, 1500 cents, 1-cos(w) is
 * 3e-5, which is one or two steps of a 16.16 value - so the numerator
 * would be quantised to tens of percent. Q30 makes it 32700 steps and
 * the pole angle accurate to a hundredth of a cent. The cutoff is
 * converted through a locally interpolated copy of the same pow2 table
 * the rest of the file uses; the table alone steps in 4.7 cents, and a
 * cutoff has 1-cos(w) proportional to the frequency squared.
 *
 * 1-cos(w) IS COMPUTED AS 2 sin^2(w/2), not by subtraction. Both are the
 * same number, but the sine at half the angle is small and its linear
 * interpolation is therefore accurate while a subtraction from a
 * table-interpolated cosine near its peak would lose every bit that
 * matters at low cutoff.
 */
#define FILT_ONE      (1L << 30)
#define FILT_TURN     (1UL << 28)   /* one whole turn of phase */
#define FILT_QUARTER  (1UL << 26)

/* sin(k*pi/128) as Q30, k = 0..64: the quarter wave the phase lookup
   interpolates. */
static const unsigned long sin_q30[65] = {
             0L,   26350943L,   52686014L,   78989349L,  105245103L,  131437462L,  157550647L,  183568930L,
     209476638L,  235258165L,  260897982L,  286380643L,  311690799L,  336813204L,  361732726L,  386434353L,
     410903207L,  435124548L,  459083786L,  482766489L,  506158392L,  529245404L,  552013618L,  574449320L,
     596538995L,  618269338L,  639627258L,  660599890L,  681174602L,  701339000L,  721080937L,  740388522L,
     759250125L,  777654384L,  795590213L,  813046808L,  830013654L,  846480531L,  862437520L,  877875009L,
     892783698L,  907154608L,  920979082L,  934248793L,  946955747L,  959092290L,  970651112L,  981625251L,
     992008094L, 1001793390L, 1010975242L, 1019548121L, 1027506862L, 1034846671L, 1041563127L, 1047652185L,
    1053110176L, 1057933813L, 1062120190L, 1065666786L, 1068571464L, 1070832474L, 1072448455L, 1073418433L,
    1073741824L
};

/* sin of a phase stated in 1/2^28 of a turn, as Q30. The four quadrants
   are the quarter table read forwards, backwards and negated. */
static long sin_turns(unsigned long t)
{
    unsigned long q, rem, pos;
    long a, b;

    q = (t >> 26) & 3UL;
    rem = t & (FILT_QUARTER - 1UL);
    if (q & 1UL)
        rem = FILT_QUARTER - rem;
    if (rem >= FILT_QUARTER) {
        a = (long)sin_q30[64];
    } else {
        pos = rem >> 20;
        a = (long)sin_q30[pos];
        b = (long)sin_q30[pos + 1];
        a += (long)(((lz_i64)(b - a) * (long)(rem & 0xFFFFFUL)) >> 20);
    }
    if (q & 2UL)
        a = -a;
    return a;
}

/* 2^(cents/1200) in 16.16 with the table's entries interpolated, which
   is what the filter's cutoff and its two resonance gains are converted
   through. cents_gain truncates to the nearest table entry, and one
   entry is 1200/256 = 4.7 cents; that is under a twentieth of a
   semitone and fine for a pitch, but a cutoff near the bottom of the
   range has 1-cos(w) proportional to the frequency SQUARED, so the same
   truncation would move it by 5%. This costs one extra multiply on a
   path that runs once per note or per cutoff change. */
static long filt_hz(long cents)
{
    long oct, rem, a, f, g, next;

    oct = cents / 1200L;
    rem = cents - oct * 1200L;
    if (rem < 0) { oct--; rem += 1200L; }
    a = rem * 256L / 1200L;
    f = rem * 256L - a * 1200L;
    next = a < 255L ? pow2_frac[a + 1] : 131072L;
    g = pow2_frac[a] + (next - pow2_frac[a]) * f / 1200L;
    if (oct > 0)
        g <<= oct;
    else if (oct < 0)
        g >>= -oct;
    return g;
}

/* The biquad for `cents` and the voice's resonance. Clamped the way the
   reference clamps: the format's 1500..13500 cents, then 0.45 of the
   sample rate so the filter's peak stays below Nyquist. */
static void filt_coeffs(Voice *v, long cents)
{
    lz_i64 a1;
    long hz, maxhz, sinw, omc, alpha, a0, a2, b0, b1, half, cw, qq;
    unsigned long phase, sh, qg, invres;

    if (cents < 1500L) cents = 1500L;
    if (cents > 13500L) cents = 13500L;
    /* The cutoff stays in 16.16 hertz through the phase conversion: one
       hertz is 5% of the lowest cutoff the format states, and the
       numerator is proportional to the frequency squared there. */
    hz = filt_hz(cents - 6900L) * 440L;
    maxhz = (g_rate * 45L / 100L) << 16;
    if (hz > maxhz) hz = maxhz;
    if (hz < 1L) hz = 1L;
    phase = (unsigned long)(((lz_i64)hz << 12) / (lz_i64)g_rate);
    if (phase >= FILT_TURN)
        phase = FILT_TURN - 1UL;

    sinw = sin_turns(phase);
    half = (long)(phase >> 1);
    sh = (unsigned long)sin_turns((unsigned long)half);
    omc = (long)(2 * (((lz_i64)sh * (lz_i64)sh) >> 30));
    cw = FILT_ONE - omc;

    /* THE TWO GAINS THE RESONANCE ASKS FOR, from the power-of-two table.
       10^(-Q/400) is 2^(-Q*9.966/1200) and 10^(-(Q-3)/200) is
       2^(-(Q-3)*19.932/1200), the constants being 1200*log2(10)/400 and
       1200*log2(10)/200. The attenuation table the envelope uses is
       floored at 96 dB and also loses every value under 1/65536, which
       is exactly where the top of the resonance range lives - hence the
       direct conversion. The 3.01 the format names is a centibel offset
       in the reference's own arithmetic, not a decibel. */
    qq = v->filt_q;
    if (qq < 0L) qq = 0L;
    if (qq > 960L) qq = 960L;

    /* qGain = 10^(-Q/400), as Q30. */
    qg = (unsigned long)(filt_hz(-(qq * 9966L / 1000L)) << 14);
    /* 1/resonanceGain = 10^(-(Q-3.01)/200), as Q30. */
    invres = (unsigned long)(filt_hz(-((qq - 3L) * 19932L / 1000L)) << 14);

    alpha = (long)(((lz_i64)sinw * (lz_i64)invres) >> 31);
    /* A pole exactly on the unit circle would ring for ever and let
       rounding grow it; one Q30 step of damping keeps it inside. */
    if (alpha < 1L)
        alpha = 1L;
    b1 = (long)(((lz_i64)omc * (lz_i64)qg) >> 30);
    b0 = b1 >> 1;
    a0 = FILT_ONE + alpha;
    a1 = 2 * (lz_i64)cw;
    a2 = FILT_ONE - alpha;

    v->f_b0 = (long)(((lz_i64)b0 << 30) / a0);
    v->f_b1 = (long)(((lz_i64)b1 << 30) / a0);
    v->f_d1 = (long)((a1 << 30) / a0);
    v->f_d2 = (long)(((lz_i64)a2 << 30) / a0);
}

/* One frame through the biquad. The output is clamped to full scale in
   the state as well as at the output, so a resonant peak cannot overflow
   the 32-bit history; the reference leaves its float history unclamped
   and clips only the mixed result. */
static long filt_run(Voice *v, long x)
{
    lz_i64 y;
    long xq = x << 15;

    y = (lz_i64)v->f_b0 * xq
      + (lz_i64)v->f_b1 * v->fx1
      + (lz_i64)v->f_b0 * v->fx2
      + (lz_i64)v->f_d1 * v->fy1
      - (lz_i64)v->f_d2 * v->fy2;
    y >>= 30;
    v->fx2 = v->fx1;
    v->fx1 = xq;
    if (y > LZ_I64_C(0x40000000))
        y = LZ_I64_C(0x40000000);
    if (y < -LZ_I64_C(0x40000000))
        y = -LZ_I64_C(0x40000000);
    v->fy2 = v->fy1;
    v->fy1 = (long)y;
    x = (long)((y + LZ_I64_C(0x4000)) >> 15);
    if (x > 32767L) x = 32767L;
    if (x < -32768L) x = -32768L;
    return x;
}

/* The cutoff the voice's sources ask for this step, in cents, and the
   decision whether the filter runs at all. The reference opens the
   filter - applies the gain alone - when both the smoothed and the
   modulated cutoff are at the top of the range and the resonance is
   zero, which is what makes a bank that states no filter bit-identical
   to one played without one. */
static void filt_step(Voice *v)
{
    long ex = 0L, target;

    if (!v->filt_any) {
        v->filt_on = 0;
        return;
    }
    if (v->filt_first) {
        v->filt_fc = v->filt_init;
        v->filt_first = 0;
    } else {
        v->filt_fc += (long)(((lz_i64)(v->filt_init - v->filt_fc)
                              * g_filt_smooth) >> 16);
    }
    if (v->filt_env != 0L)
        ex += (long)(((lz_i64)v->filt_env * v->env_pos) >> 16);
    if (v->filt_vel != 0L)
        ex += v->filt_vel * (long)v->vel / 128L;
    if (v->filt_key != 0L)
        ex += v->filt_key * (long)v->note / 128L;
    if (v->filt_lfo != 0L && v->lfo_inc != 0L && v->lfo_delay <= 0L)
        ex += v->filt_lfo * lfo_tri(v->lfo_phase) >> 15;
    if (v->filt_vib != 0L && v->vib_inc != 0L && v->vib_delay <= 0L)
        ex += v->filt_vib * lfo_tri(v->vib_phase) >> 15;
    target = v->filt_fc + ex;

    if (v->filt_fc > 13499L && target > 13499L && v->filt_q == 0L) {
        v->filt_fc = 13500L;
        v->filt_on = 0;
        return;
    }
    v->filt_on = 1;
    if (target - v->filt_last > 1L || v->filt_last - target > 1L) {
        filt_coeffs(v, target);
        v->filt_last = target;
    }
}

/* Where a note sits between the channels: -500 hard left, 0 centre,
   +500 hard right, the unit being tenths of a percent.

   EQUAL POWER, not a straight ramp. The two halves are the cos and the
   sin of the same angle, so the power a note carries does not move with
   it - a centred note comes out at 0.707 of each channel and the sum of
   the two squares is the whole. A straight ramp would move a centred
   note's level by up to 6 dB depending only on where a bank happens to
   put its pan. */
static void pan_gains(int pan, long *gl, long *gr)
{
    int  p = pan;

    if (p < -500) p = -500;
    if (p > 500) p = 500;
    /* -500..500 out to 0..1024, and the table is 8.8, so it is doubled
       back into the 16.16 the mix multiplies in. */
    {
        int i = (p + 500) * (PAN_POINTS - 1) / 1000;
        *gl = (long)pan_left[i] << 1;
        *gr = (long)pan_right[i] << 1;
    }
}

static void exclusive_release(Voice *v);

/* ONE LFO'S RATE AND DELAY, the pair the bank states in absolute cents.
   The mod LFO and the vibrato LFO use the same arithmetic with their
   own fields; two copies of it were the easiest way to let them drift. */
static void lfo_setup(long *inc, long *delay, int rate_tc, int delay_tc)
{
    long freq = cents_gain(rate_tc);
    freq = (long)(((lz_i64)freq * ABS_CENTS_TO_HZ) >> 16);
    *inc = freq > 0L
        ? (long)(((lz_i64)freq << 16) / (lz_i64)g_rate) : 0L;
    *delay = (long)((lz_i64)cents_gain(delay_tc) * (lz_i64)g_rate / 65536L);
    if (*delay < 0L) *delay = 0L;
}

/* Start the one voice a zone describes. A note-on that several regions
   match calls this once per region; every voice it starts carries the
   same `note_id`, which is what the Note Off later releases. */
static void start_zone(int ch, int note, int vel, const KsbZone *z,
                       uint32_t note_id)
{
    Voice *v;
    long cents, base, step, mant, dec, atk;
    int slot, oct;

    /* EXCLUSIVE CLASS: a note in a choke group ends the group's previous
       notes on its own channel. A region flagged self-non-exclusive in
       its header keeps its own notes alive, which is how a kit lets
       repeated hits of one cymbal ring over each other. */
    if (z->key_group != 0) {
        int k;
        for (k = 0; k < WAVE_VOICES; k++)
            if (g_v[k].pcm && g_v[k].ch == ch && g_v[k].key_group == z->key_group
                && g_v[k].held) {
                if (z->self_non_exclusive && g_v[k].region_id == z->region_id)
                    continue;
                exclusive_release(&g_v[k]);
            }
    }
    slot = pick();
    v = &g_v[slot];

    v->pcm = z->pcm;
    v->loop_start = z->loop_start;
    v->loop_len = z->loop_len;
    /* WHERE PLAYBACK ENDS: the end of the loop when there is one, the
       end of the sample when there is not. The frame after the last is
       the loop's first, and for an unlooped sample there is none, so
       the last frame interpolates against itself. Reading pos+1
       unconditionally would step one byte past the sample - which lands
       inside the image, so it never faults and shows up only as a tick
       at the end of every note. */
    v->end = z->loop_len > 0 ? z->loop_start + z->loop_len : z->frames;
    if (v->end < 1) { v->pcm = 0; g_unvoiced++; return; }
    v->pos = 0;
    v->frac = 0UL;

    /* THE BEND IS NOT IN HERE. It is applied on every render step so a
       bend that moves under a held note follows it - what is stored in
       base_step is the pitch without one, and the vibrato and the pitch
       envelope are measured from that same figure. */
    cents = (long)(note - z->unity) * 100L + z->fine;
    mant = pitch_mantissa(cents, &oct);
    /* The stored rate over the output rate, as 16.16. Done unsigned
       because a bank may hold a rate up to 65535 and 65535<<16 is
       4294901760 - inside an unsigned long by sixty thousand, outside
       a signed one by half. */
    base = (long)(((unsigned long)z->rate << 16) / (unsigned long)g_rate);
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
    v->fir_idx = fir_pick((unsigned long)step);
    v->lfo_pitch = z->lfo_pitch_tc;
    v->wheel_pitch = z->wheel_pitch_tc;
    /* A QUARTER TURN, not zero. The reference player starts the LFO at
       phase 0.25 and a triangle reads 0 there, so the note opens in
       tune and the first thing it does is swing - starting at zero
       would open it at full depth and then fall. */
    v->lfo_phase = 0x40000000UL;
    v->lfo_inc = 0L;
    v->lfo_delay = 0L;
    /* THE VIBRATO LFO is a second oscillator with the same triangle and
       the same quarter-turn start. It carries the vibrato specifically:
       the bank's own depths, the two default 50-cent connections CC1
       and channel pressure make, and a rate and delay of its own. It
       starts at its default 8.1758 Hz when the bank declares no rate,
       which is what the record's zero means. */
    v->vib_phase = 0x40000000UL;
    v->vib_inc = 0L;
    v->vib_delay = 0L;
    v->vib_free = z->vib_pitch;
    v->vib_wheel = z->vib_wheel_pitch;
    /* The rate is an absolute pitch: see ABS_CENTS_TO_HZ. The mod LFO
       runs on every voice, and a zone whose depth is zero leaves the
       step where the static parameters put it. */
    lfo_setup(&v->lfo_inc, &v->lfo_delay, z->lfo_tc, z->lfo_delay_tc);
    lfo_setup(&v->vib_inc, &v->vib_delay, z->vibfreq_tc, z->vibdelay_tc);
    v->region_pan = z->pan;
    v->lfo_gain = z->lfo_gain;
    /* THE MODULATION ENVELOPE, its own state machine beside the volume
       one. Its output is a LEVEL - 0 is the note at rest and GAIN_ONE is
       the pitch depth applied in full - and the record holds the DLS
       EG2 sustain as the level it is stated to be, in tenths of a
       percent: 1000 is the depth held and 0 is none of it. The volume
       envelope's sustain is the opposite quantity, an attenuation, and
       reading this one the same way inverted every note that declares
       the depth at zero - which is most of gm.dls - into full
       application of it. */
    v->env_pos = 0L;
    v->env_att_inc = 0L;
    v->env_dec_inc = 0L;
    v->env_rel_inc = 0L;
    v->env_sus = (long)z->msustain_level * GAIN_ONE / 1000L;
    if (v->env_sus < 0L) v->env_sus = 0L;
    if (v->env_sus > GAIN_ONE) v->env_sus = GAIN_ONE;
    /* The modulation envelope's release, in the same shape the volume
       one uses: full scale over the stated time, and a note-off starts
       it wherever the envelope had got to. */
    {
        long ms = z->mrelease_ms > 0 ? (long)z->mrelease_ms : 1L;
        long steps = ms * (g_rate / 1000L) / ENV_STEP;
        if (steps < 1L) steps = 1L;
        v->env_rel_inc = GAIN_ONE / steps;
        if (v->env_rel_inc < 1L) v->env_rel_inc = 1L;
    }
    v->env_pitch = z->env_pitch_tc;
    v->key_pitch = z->key_pitch;
    /* THE DECAY STEP IS SET WHETHER OR NOT THERE IS AN ATTACK. The two
       branches only decide what the envelope does first; the decay rate
       must exist for both. Both envelope times carry their own source
       scaling, and these are the modulation envelope's: key for the
       decay, velocity for the attack. */
    if (z->mdecay_ms > 0) {
        long dec = (long)z->mdecay_ms;
        long steps;
        /* GeneralUser's electric piano below key 35 is the case that
           made the 64-bit multiply in scale_ms necessary: a 65-second
           decay at full gain wraps a 32-bit long negative and becomes
           one millisecond - a click and then silence. */
        dec = scale_ms(dec, z->mdecay_key_tc, note);
        steps = dec * (g_rate / 1000L) / ENV_STEP;
        if (steps < 1L) steps = 1L;
        v->env_dec_inc = (GAIN_ONE - v->env_sus) / steps;
        if (v->env_dec_inc < 1L) v->env_dec_inc = 1L;
    }
    /* The opening stages that may be absent. The hold carries the same
       key scaling the decay does, so a high note's hold is shorter by
       the bank's own time cents for every key above the source's
       reference. */
    v->mdelay_left = ms_steps((long)z->mdelay_ms, g_rate);
    {
        long hold = (long)z->mhold_ms;
        hold = scale_ms(hold, z->mhold_key_tc, note);
        v->mhold_left = ms_steps(hold, g_rate);
    }
    if (z->mattack_ms > 0) {
        long ms = (long)z->mattack_ms;
        long left;
        ms = scale_ms(ms, z->mattack_vel_tc, vel);
        left = ms * (g_rate / 1000L);
        if (left < 1L) left = 1L;
        v->env_att_inc = GAIN_ONE / (left / ENV_STEP + 1L);
    }
    /* THE ENVELOPE OPENS IN ITS FIRST DECLARED STAGE. A delay is no
       modulation at all, a hold is the peak, a decay starts from the
       peak towards the sustain, and a zone declaring none of them sits
       at the sustain for the whole note. */
    if (v->mdelay_left > 0L) {
        v->env_phase = 0;
        v->env_pos = 0L;
    } else if (v->env_att_inc > 0L) {
        v->env_phase = 1;
        v->env_pos = 0L;
    } else if (v->mhold_left > 0L) {
        v->env_phase = 2;
        v->env_pos = GAIN_ONE;
    } else if (v->env_dec_inc > 0L) {
        v->env_phase = 3;
        v->env_pos = GAIN_ONE;
    } else {
        v->env_phase = 4;
        v->env_pos = v->env_sus;
    }

    /* Velocity squared: MIDI velocity is a loudness the player feels,
       not an amplitude, and a linear map makes everything quiet sound
       the same. OUT OF 128 RATHER THAN 127, which is the scale the
       modulator's raw source is built on and the one the reference
       player normalises to; the two differ by 0.14 dB at full velocity,
       which is below anything audible but is not the same number. */
    v->gain = (long)vel * (long)vel * GAIN_ONE / (128L * 128L);
    /* THE REGION'S GAIN IS IN TENTHS OF A DECIBEL AND NEGATIVE, being an
       attenuation, and the table is indexed by an attenuation, so the
       sign comes off here. A positive value is a boost and the table
       holds no entry for one - full scale, which is also what the
       reference player arrives at, since it negates the same value by
       two and a half and clamps the negative result to zero. */
    v->gain = (long)(((lz_i64)v->gain
                      * db_gain_of(z->gain)) >> 16);
    /* VELOCITY -> GAIN IS NOT APPLIED. The record carries it because the
       source states it, but layering its curve on top of the velocity
       square below attenuates the same source twice: measured over
       onestop, the mix's one-second envelope swung by 8.3 dB against
       the DLS player's 5.6, and the piece pumped. The square alone
       tracks that player to 0.998 correlation. */

    v->atten = 0L;
    v->sustain = (long)z->sustain_cb << 16;
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
    dec = (long)z->decay_ms;
    atk = (long)z->attack_ms;
    dec = scale_ms(dec, z->key_decay_tc, note);
    atk = scale_ms(atk, z->vel_attack_tc, vel);
    v->rate_dec = env_rate(dec, g_rate);
    /* RELEASE FLOORS AT 30 MS, which is the reference's minimum note
       length. Below that a note off is a STEP rather than a ramp: one
       ENV_STEP is 1.45 ms at 22050, and a bank that asks for a shorter
       release was almost certainly asking for a shorter one than it
       meant. */
    {
        long rel = (long)z->release_ms;
        if (rel < 30L) rel = 30L;
        v->rate_rel = env_rate(rel, g_rate);
    }
    v->tick = ENV_STEP;
    /* The opening stages. A delay holds the envelope at zero, which is
       silence because the attack level is still zero; a hold keeps the
       peak it has just reached. Both are absent when the bank states 0
       milliseconds, and the hold carries keyNum -> hold the same way the
       decay carries its own key scaling. */
    v->vdelay_left = ms_steps((long)z->vdelay_ms, g_rate);
    {
        long hold = (long)z->vhold_ms;
        hold = scale_ms(hold, z->vhold_key_tc, note);
        v->vhold_left = ms_steps(hold, g_rate);
    }
    /* AN ATTACK OF ZERO IS A JUMP, and BOTH BANKS DECLARE ZERO on every
       region they carry - so every note in both opened with a step and
       that step is the click heard in the middle of a piece. The
       reference player's envelope generator floors an attack at -12000
       time cents, about a millisecond, and jumps a sub-buffer attack
       straight to hold; a millisecond here is under one ENV_STEP, which
       is 1.45 ms at 22050, so flooring it there would still be the same
       single step. Three milliseconds is two steps, which is the
       shortest ramp this envelope can actually draw. */
    if (atk < 3L) atk = 3L;
    {
        v->phase = v->vdelay_left > 0L ? 0 : 1;
        v->attack_env = 0L;
        /* Divided down first: a 65-second attack times 48000 is 3.1e9,
           which a 32-bit long does not hold. */
        v->attack_left = atk * (g_rate / 1000L);
        if (v->attack_left < 1L) v->attack_left = 1L;
        v->attack_inc = GAIN_ONE / (v->attack_left / ENV_STEP + 1L);
    }
    v->ch = ch;
    v->note = note;
    v->key_group = z->key_group;
    v->region_id = z->region_id;
    v->held = 1;
    v->note_id = note_id;
    v->age = ++g_clock;
    /* THE FILTER'S STATIC STATE AND ITS FIRST SET OF COEFFICIENTS. A
       region whose cutoff is open, whose resonance is zero and whose
       depths are all zero has nothing a filter could do, and is marked
       so the mixing loop does no filter work for it at all; every other
       region opens with the filter already at its own cutoff, which is
       the reference's own first-block behaviour. */
    v->vel = vel;
    v->filt_init = z->filt_fc;
    v->filt_q = z->filt_q;
    v->filt_env = z->filt_env;
    v->filt_vel = z->filt_vel;
    v->filt_key = z->filt_key;
    v->filt_lfo = z->filt_lfo;
    v->filt_vib = z->filt_vib;
    v->filt_fc = z->filt_fc;
    v->filt_last = -100000L;    /* no coefficients yet */
    v->filt_first = 1;
    v->fx1 = v->fx2 = v->fy1 = v->fy2 = 0L;
    v->filt_any = !(z->filt_fc >= 13500 && z->filt_q == 0
                    && z->filt_env == 0 && z->filt_vel == 0
                    && z->filt_key == 0 && z->filt_lfo == 0
                    && z->filt_vib == 0);
    v->filt_on = 0;
    if (v->filt_any)
        filt_step(v);
}

/* A note-on starts one voice for EVERY region the patch has over this key
   and velocity, which is how a bank's layered and velocity-stacked zones
   add up. The regions are enumerated in file order, so a note matching
   exactly one region starts exactly one voice.
   g_unvoiced counts a note once when no region matches at all, and a zone
   that turns out to have no frames to play counts itself once in
   start_zone. */
static void start(int ch, int note, int vel)
{
    KsbZone z;
    int i, n, msb, lsb;

    /* THE DRUM CHANNEL'S KIT IS BANK 0. A GS or XG file selects a drum
       bank of its own on channel 10 - 127 is the common one - and this
       bank carries the GM kit under bank 0 and no other, so the select is
       ignored rather than allowed to silence the channel. Melodic
       channels resolve the pair the file sent. */
    msb = ch == DRUM_CH ? 0 : g_ch[ch].bank_msb;
    lsb = ch == DRUM_CH ? 0 : g_ch[ch].bank_lsb;
    n = ksb_zone_count(g_bank, msb, lsb, g_ch[ch].program, note, vel);
    if (n <= 0) {
        g_unvoiced++;
        return;
    }
    g_note_seq++;
    for (i = 0; i < n; i++) {
        if (ksb_zone_at(g_bank, msb, lsb, g_ch[ch].program, note, vel,
                        i, &z) != 1) {
            g_unvoiced++;
            continue;
        }
        start_zone(ch, note, vel, &z, g_note_seq);
    }
}

/* Release carries on from wherever the envelope had got to. A note
   released during its attack keeps the attack level it had reached -
   forcing it to full scale would step the gain up at note-off. A note
   released during its delay is silent already, and the release ramp
   runs under an attack level of zero, so it stays silent and ends. */
static void release(Voice *v)
{
    v->held = 0;
    v->phase = 5;
    v->env_phase = 5;
}

/* A choke is a release short enough to read as a cut, but not so short
 * that it steps: the reference floors an exclusive cut at 70 ms. */
static void exclusive_release(Voice *v)
{
    long steps;
    release(v);
    steps = 70L * (g_rate / 1000L) / ENV_STEP;
    if (steps < 1L)
        steps = 1L;
    v->rate_rel = ((ENV_FLOOR << 16) - v->atten) / steps;
    if (v->rate_rel < 1L)
        v->pcm = 0;
}

static void op_note_off(int ch, int note)
{
    int      i;
    uint32_t id;
    Voice   *oldest = 0;

    /* PERCUSSION TAKES NOTE OFF LIKE ANY OTHER CHANNEL: the reference
       releases the hit's voice from wherever its envelope had got to.
       Program changes on the channel are still ignored - the kit has a
       fixed program and this bank carries one kit. */
    /* OVERLAPPING NOTES ARE FIFO: MIDI files do retrigger a pitch before its
       previous Note Off, and one Off must stop one note-on, not every one on
       that key. The oldest voice still sounding is the one this Off belongs
       to; the newer ones keep sounding until their own Offs arrive. Every
       zone that note-on started shares its id, so all of those voices go
       together, and a pedal-held voice is selected the same way it always
       was. */
    for (i = 0; i < WAVE_VOICES; i++) {
        if (g_v[i].pcm && g_v[i].ch == ch && g_v[i].note == note
            && g_v[i].held) {
            if (oldest == 0 || g_v[i].age < oldest->age)
                oldest = &g_v[i];
        }
    }
    if (oldest == 0)
        return;
    id = oldest->note_id;
    for (i = 0; i < WAVE_VOICES; i++) {
        if (!g_v[i].pcm || g_v[i].ch != ch || g_v[i].note != note
            || !g_v[i].held || g_v[i].note_id != id)
            continue;
        if (g_ch[ch].sustain)
            g_v[i].held = 2;    /* the pedal has it */
        else
            release(&g_v[i]);
    }
}

static void op_note_on(int ch, int note, int vel)
{
    if (vel <= 0) { op_note_off(ch, note); return; }
    start(ch, note, vel);
}

static void op_program(int ch, int pgm)
{
    if (ch == DRUM_CH)
        return;
    g_ch[ch].program = pgm & 0x7F;
}

/* Reset All Controllers, RP-15's subset: bend, pressure, CC1, CC11, the
 * pedal and the RPN state. The program, volume (CC7), pan (CC10), the
 * bend range and the bank select (CC0/CC32) are NOT controllers this
 * message resets: a sequence that resets controllers between movements
 * keeps the patch it chose. */
static void reset_channel(int ch)
{
    Chan *c = &g_ch[ch];
    int   i;

    c->bend = 0;
    c->expression = 127;
    c->sustain = 0;
    c->wheel = 0;
    c->pressure = 0;
    c->rpn_msb = c->rpn_lsb = 127;
    c->data_msb = c->data_lsb = 0;
    for (i = 0; i < WAVE_VOICES; i++)
        if (g_v[i].pcm && g_v[i].ch == ch && g_v[i].held == 2)
            release(&g_v[i]);
}

static void op_control(int ch, int cc, int val)
{
    int i;
    switch (cc) {
    /* BANK SELECT. The pair names the patch that the next note-on reads
       together with the program, and it stays in force until another one
       arrives; a program change does not clear it and neither does
       CC121. */
    case 0:  g_ch[ch].bank_msb = val & 0x7F; break;
    case 32: g_ch[ch].bank_lsb = val & 0x7F; break;
    case 1:  g_ch[ch].wheel = val; break;
    case 10:
        /* 0..127 out to -500..500, which is the unit the region's pan
           is in: 0 sits at centre and the two add. */
        g_ch[ch].pan = (val - 64) * 500 / 64;
        break;
    case 6:
        g_ch[ch].data_msb = val;
        if (g_ch[ch].rpn_msb == 0 && g_ch[ch].rpn_lsb == 0)
            g_ch[ch].bend_range_cents =
                ((g_ch[ch].data_msb << 7) | g_ch[ch].data_lsb) * 25 / 32;
        break;
    case 7:  g_ch[ch].volume = val; break;
    case 11: g_ch[ch].expression = val; break;
    case 38:
        /* THE DATA ENTRY IS 14 BITS: the MSB arrives on CC6 and the LSB
           on CC38, and RPN 0's value is in semitones with 128 units to
           the semitone. Both bytes update the range while RPN 0 is
           selected, so a file that sends only one of them still moves
           it - which is what the reference does. */
        g_ch[ch].data_lsb = val;
        if (g_ch[ch].rpn_msb == 0 && g_ch[ch].rpn_lsb == 0)
            g_ch[ch].bend_range_cents =
                ((g_ch[ch].data_msb << 7) | g_ch[ch].data_lsb) * 25 / 32;
        break;
    case 100:
        g_ch[ch].rpn_lsb = val;
        g_ch[ch].data_msb = g_ch[ch].data_lsb = 0;
        break;
    case 101:
        g_ch[ch].rpn_msb = val;
        g_ch[ch].data_msb = g_ch[ch].data_lsb = 0;
        break;
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
    case 120:
        /* All Sound Off: immediate. No release tail. */
        for (i = 0; i < WAVE_VOICES; i++)
            if (g_v[i].pcm && g_v[i].ch == ch) g_v[i].pcm = 0;
        break;
    case 121:
        reset_channel(ch);
        break;
    case 123: case 124: case 125: case 126: case 127:
        /* All Notes Off, and the omni/poly mode messages, which this
           sampler answers the same way: release everything on the channel. */
        for (i = 0; i < WAVE_VOICES; i++)
            if (g_v[i].pcm && g_v[i].ch == ch) release(&g_v[i]);
        break;
    default: break;
    }
}

/* A bend of 8192 is centre. THE RANGE COMES FROM THE RPN, not from the
   GM default: a file that sends RPN 0 gets the bend it asked for, and a
   file that never does gets two semitones because that is what the
   channel opens at. It is applied every render step rather than only at
   note-on, so a bend that moves under a held note follows. */
static void op_pitch(int ch, int bend)
{
    g_ch[ch].bend = (bend - 8192) * g_ch[ch].bend_range_cents / 8192;
}


/* One channel message, in the shape common/smf.h hands out: status in the low
   byte, then the two data bytes. Everything this synth has an answer for is
   here; a message it has none for is dropped. */
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
    case 0xA0:
        /* Polyphonic pressure has no default behaviour (SF2), and this
           sampler has no modulator matrix to give it one. */
        break;
    case 0xD0:
        g_ch[ch].pressure = (int)((msg >> 8) & 0x7FUL);
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

/* Which family row a step needs. The cutoff is min(1, 1/step) of the
   SAMPLE's own Nyquist: interpolating up, the reconstruction stops at
   the sample's rate and the cutoff is 1; reading faster than one, it
   stops at the output's, which is 1/step of the sample's. The WIDEST
   row belongs to the full cutoff, because that is where the images sit
   closest - a 22050 Hz sample played into 44100 puts one at
   15261 +- f, and the eight taps this used to be held it back by
   twenty decibels, which is the hiss that is not in the instrument's
   key. See tools/gen_fir.py for the measurement and the family. */
static int fir_pick(unsigned long step)
{
    lz_i64 fc;
    int i;

    if (step <= 65536UL)
        return 0;                       /* full cutoff: interpolating */
    fc = ((lz_i64)65536 << 16) / (lz_i64)step;
    /* fir_cut runs HIGH TO LOW - 0.70 first, 0.24 last - so the first
       entry that fits is the WIDEST that fits. Scanning from the far
       end finds the narrowest instead and quietly darkens every note;
       the table's order and the loop have to agree about which end is
       which. */
    for (i = 0; i < FIR_CUTS - 1; i++) {
        /* fir_cut[i] is the cutoff in 32768ths; 16.16 is twice that. */
        if ((lz_i64)fir_cut[i] * 2 <= fc)
            break;
    }
    return i;
}

/* A windowed sinc at `pos` + `frac`, at the cutoff this note needs.
   See fir_table.h: the row's width runs the other way from its
   cutoff, so the widest filter is the one at the sample's Nyquist. */
static long interp(const Voice *v, long pos, unsigned long frac)
{
    const short *h = FIR_ROW(v->fir_idx, frac);
    int w = FIR_TAPS(v->fir_idx);
    int half = (w - 1) / 2;
    long acc = 0;
    int k;
    /* NO SYMMETRY SHORTCUT. A windowed sinc is symmetric about the
       point where its argument is zero, and at every phase but two
       that point falls between two taps - so the slice of the table
       for a phase is not symmetric about the array's middle and
       folding it there turns a low pass into a double peaked thing
       that amplifies near the cutoff. That is what an earlier version
       of this did, and the top octave came back 27 dB hot because of
       it. Every tap is multiplied. */
    if (pos >= half && pos + half < v->end) {
        for (k = 0; k < w; k++) {
            acc += (long)h[k] * fetch(v, pos - half + k);
        }
    } else {
        for (k = 0; k < w; k++) {
            acc += (long)h[k] * fetch_wrap(v, pos - half + k);
        }
    }
    /* The widest row sums 27980 of absolute coefficients over all its
       phases, so a full scale input reaches 9.2e8 - inside a 32-bit
       long, which is what this accumulates in. */
    acc >>= FIR_SHIFT;
    if (acc > 32767L) acc = 32767L;
    if (acc < -32768L) acc = -32768L;
    return acc;
}

/* WHAT A VOICE IS ATTENUATED BY, in tenths of a decibel: the envelope's
   own climbing number, with the mod LFO folded on top of it.

   modLfo -> gain is stated in centibels at full swing and the
   oscillator swings both ways, so the net can come out below zero -
   and the table underneath has no entry for a boost. Clamping at zero
   is what "no attenuation" means there, which is the same answer the
   reference player reaches by clamping its own amount to 0..960.

   Measured on gm.dls, seven regions carry it at 1 cB and FF8's seven at
   1 cB as well: it is a small colouring of the tone rather than a
   gating effect, but a reader that skips it cannot play a bank that
   sets it. */
static long tremolo_cb(const Voice *v)
{
    long cb = v->atten >> 16;

    if (v->lfo_gain != 0 && v->lfo_inc != 0 && v->lfo_delay <= 0) {
        cb += (long)v->lfo_gain * lfo_tri(v->lfo_phase) >> 15;
        if (cb < 0L) cb = 0L;
    }
    return cb;
}

static void mix(long *accL, long *accR, long frames)
{
    long i, s, ge, cg, env, vg, eg, gl, gr;
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
        cg = (vg * GAIN_ONE / (128L * 128L)) >> 8;
        eg = (eg * GAIN_ONE / (128L * 128L)) >> 8;
        cg = (cg * eg) >> 8;
        /* AND THE PAN, region and channel together, on the same
           schedule the volume is: a fader move - and CC10 moves a
           held note's pan - has to be heard on notes already down. */
        pan_gains(v->region_pan + g_ch[v->ch].pan, &gl, &gr);
        /* AND NOT THE MASTER FACTOR. SpessaSynth multiplies everything
           it renders by 0.6 (SPESSASYNTH_GAIN_FACTOR) so that it "does
           not stay too loud". That is a level choice of one synthesiser,
           not something in the format, and the thing being matched here
           is the machine's own synthesiser, which does not make it: with
           0.6 applied, loopback capture put our signal 21 dB below the
           system's, close enough to a fixed floor to be heard as hiss. */
        /* THE ENVELOPE AND THE GAINS STAY AT 16.16 ALL THE WAY DOWN.
           Built through 8.8 intermediates - a shift by eight on each
           factor - the product reached zero at about -48 dB and the
           voice was cut there, which is what a piano's tail dying
           halfway through its decay was. 16.16 reaches the envelope's
           own -96 dB floor. */
        env = (long)(((lz_i64)db_gain_tenths(tremolo_cb(v))
                      * v->attack_env) >> 16);
        ge = (long)(((((lz_i64)v->gain * cg) >> 8) * env) >> 16);
        for (i = 0; i < frames; i++) {
            s = interp(v, v->pos, v->frac);
            /* THE FILTER RUNS ON THE INTERPOLATED FRAME, before the
               gain: that is the order the format's spec states, and it
               is where the reference applies it too. A voice whose
               filter is open skips this entirely. */
            if (v->filt_on)
                s = filt_run(v, s);
            /* NOT 32 BITS. ge is a 16.16 net gain and it can exceed 1.0:
               the velocity square, the channel volume and expression,
               the envelope and the region's gain all fold into it, and
               on a full velocity through a full fader it measured
               66568 (1.016) on onestop. -32768 * 66568 is below
               INT32_MIN by 3.3 million, so the product wrapped to a
               positive value: a Fury electric bass at 201.9 s jumped
               from -32312 to +32207 between two frames at the same
               sample position, which is the click. The pan below then
               costs one 8.8 shift and a multiply that cannot leave 32
               bits. */
            s = (long)(((lz_i64)s * ge) >> 16);
            accL[i] += (s * (gl >> 8)) >> 8;
            accR[i] += (s * (gr >> 8)) >> 8;

            v->frac += v->sfrac;
            v->pos += v->step + (long)(v->frac >> 16);
            v->frac &= 0xFFFFUL;
            /* THE LFOs' PHASES, once a frame. While there is a delay to
               count the phase stays at its quarter turn, where the
               triangle reads zero, so the pitch comes out exactly as the
               static parameters said - which is the same as having no
               LFO at all. The two oscillators run independently. */
            if (v->lfo_delay > 0) {
                v->lfo_delay--;
            } else if (v->lfo_inc != 0) {
                v->lfo_phase += (unsigned long)v->lfo_inc;
            }
            if (v->vib_delay > 0) {
                v->vib_delay--;
            } else if (v->vib_inc != 0) {
                v->vib_phase += (unsigned long)v->vib_inc;
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
                /* THE MODULATION ENVELOPE HAS NO COUNTER OF ITS OWN: it
                   steps here, on the volume envelope's schedule, once
                   every ENV_STEP frames. Its five stages are delay (no
                   modulation), attack, hold at the peak, decay toward
                   the sustain and then sustain; GAIN_ONE is the depth
                   applied in full. */
                {
                    if (v->env_phase == 0) {
                        if (--v->mdelay_left <= 0L) {
                            if (v->env_att_inc > 0L) {
                                v->env_phase = 1;
                            } else if (v->mhold_left > 0L) {
                                v->env_phase = 2;
                                v->env_pos = GAIN_ONE;
                            } else if (v->env_dec_inc > 0L) {
                                v->env_phase = 3;
                                v->env_pos = GAIN_ONE;
                            } else {
                                v->env_phase = 4;
                                v->env_pos = v->env_sus;
                            }
                        }
                    } else if (v->env_phase == 1) {
                        v->env_pos += v->env_att_inc;
                        if (v->env_pos >= GAIN_ONE) {
                            v->env_pos = GAIN_ONE;
                            if (v->mhold_left > 0L) {
                                v->env_phase = 2;
                            } else if (v->env_dec_inc > 0L) {
                                v->env_phase = 3;
                            } else {
                                v->env_phase = 4;
                                v->env_pos = v->env_sus;
                            }
                        }
                    } else if (v->env_phase == 2) {
                        if (--v->mhold_left <= 0L) {
                            if (v->env_dec_inc > 0L) {
                                v->env_phase = 3;
                            } else {
                                v->env_phase = 4;
                                v->env_pos = v->env_sus;
                            }
                        }
                    } else if (v->env_phase == 3) {
                        if (v->env_pos > v->env_sus) {
                            v->env_pos -= v->env_dec_inc;
                            if (v->env_pos <= v->env_sus) {
                                v->env_pos = v->env_sus;
                                v->env_phase = 4;
                            }
                        } else {
                            v->env_phase = 4;
                        }
                    } else if (v->env_phase == 4) {
                        /* sustain: the level does not move */
                    } else {
                        if (v->env_pos > 0L) {
                            v->env_pos -= v->env_rel_inc;
                            if (v->env_pos < 0L)
                                v->env_pos = 0L;
                        }
                    }
                }

                /* THE FILTER'S CUTOFF, on the same schedule: the
                   modulation envelope and the two LFOs have just moved,
                   so this is where the reference computes its cutoff
                   too. The smoothing and the coefficient update live in
                   filt_step; a voice marked filt_any off does nothing
                   here. */
                if (v->filt_any)
                    filt_step(v);

                /* THE PITCH THE LFO AND THE ENVELOPE ASK FOR, summed in
                   cents before the one lookup that converts them. A
                   region that declares neither leaves the step exactly
                   as the static parameters put it. */
                {
                    long cents = 0L;
                    if (v->lfo_inc != 0 && v->lfo_pitch != 0) {
                        /* The mod LFO's own depth. What CC1 scales is
                           the vibrato LFO's depth below: a DLS1 bank's
                           modLfo + CC1 connection is the vibrato
                           connection there. */
                        cents += v->lfo_pitch * lfo_tri(v->lfo_phase) >> 15;
                    }
                    if (v->vib_inc != 0) {
                        /* THE WHEEL'S DEPTH IS ONE CONNECTION, NOT TWO.
                           A bank's own vibratoLfo + CC1 amount is the
                           wheel's depth; failing that the amount a DLS1
                           modLfo + CC1 connection carries, which the
                           reference moves here; failing both, the
                           default 50 cents. Channel pressure's own
                           default 50 cents is added on top, and the
                           free depth is always there. */
                        long cc1depth = v->vib_wheel != 0L
                                      ? v->vib_wheel
                                      : (v->wheel_pitch != 0L
                                         ? v->wheel_pitch : 50L);
                        long depth = v->vib_free
                                     + cc1depth * g_ch[v->ch].wheel / 127L
                                     + 50L * g_ch[v->ch].pressure / 127L;
                        if (depth != 0)
                            cents += depth * lfo_tri(v->vib_phase) >> 15;
                    }
                    /* keyNum -> pitch: the source is the key over
                       the full 0..127 range, so the depth scales with
                       it. Static for the voice, but summed here with
                       the rest because the step is recomputed anyway. */
                    if (v->key_pitch != 0)
                        cents += v->key_pitch * (long)v->note / 128L;
                    if (v->env_pitch != 0 && v->env_pos != 0)
                        cents += (int)((long)v->env_pitch
                                       * v->env_pos >> 16);
                    /* AND THE BEND, which is why it was kept out of the
                       step the note opened with: a bend that moves under
                       a held note has to be heard to move. The range it
                       was scaled by is the RPN's, or two semitones when
                       the file never sends one. */
                    cents += g_ch[v->ch].bend;
                    if (cents != 0) {
                        long st = (long)(((lz_i64)v->base_step
                                          * cents_gain((int)cents)) >> 16);
                        v->step = st >> 16;
                        v->sfrac = (unsigned long)(st & 0xFFFFL);
                        /* The row follows the step: a bend that walks
                           the pitch across an octave walks the images
                           with it. */
                        v->fir_idx = fir_pick((unsigned long)st);
                    } else {
                        v->step = v->base_step >> 16;
                        v->sfrac = (unsigned long)(v->base_step & 0xFFFFL);
                        v->fir_idx = fir_pick((unsigned long)v->base_step);
                    }
                }
                /* THE VOLUME ENVELOPE'S FIVE STAGES, then release. A
                   delay leaves the attack level at zero, which is
                   silence; a hold keeps the peak the attack reached; a
                   decay climbs the attenuation toward its sustain and
                   then holds it. */
                if (v->phase == 0) {
                    if (--v->vdelay_left <= 0L)
                        v->phase = 1;
                } else if (v->phase == 1) {
                    v->attack_env += v->attack_inc;
                    if (v->attack_env >= GAIN_ONE) {
                        v->attack_env = GAIN_ONE;
                        v->phase = v->vhold_left > 0L ? 2 : 3;
                    }
                } else if (v->phase == 2) {
                    if (--v->vhold_left <= 0L)
                        v->phase = 3;
                } else if (v->phase == 3) {
                    if (v->atten < v->sustain) {
                        v->atten += v->rate_dec;
                        if (v->atten >= v->sustain) {
                            v->atten = v->sustain;
                            v->phase = 4;
                        }
                    } else {
                        v->phase = 4;
                    }
                } else if (v->phase == 4) {
                    /* sustain: the attenuation does not move */
                } else {
                    v->atten += v->rate_rel;
                }
                if (v->atten >= (ENV_FLOOR << 16)) { v->pcm = 0; break; }
                env = (long)(((lz_i64)db_gain_tenths(tremolo_cb(v))
                              * v->attack_env) >> 16);
                ge = (long)(((((lz_i64)v->gain * cg) >> 8) * env) >> 16);
            }
        }
    }
}

/* MASTER HEADROOM: -1 dB, applied before the clip so a peak that would
   have saturated is scaled instead of flattened. MEASURED, not a taste:
   on onestop the picked bank clipped 2439 samples and the gm reference
   919, and -0.5 dB took both to zero - the peaks overshoot full scale by
   about five percent, not by a musical amount. -1 dB keeps half a
   decibel in hand. It is a FIXED gain and deliberately not a compressor:
   a limiter's gain curve, let alone a tube's, would rewrite the level
   relationship the source criteria measure against gm, and the fault
   here is a rare transient, not pumping. The calibration window is a
   ratio against a reference rendered through this same path, so both
   sides lose the same decibel and every measured deviation is unchanged.

   Both accumulators are scaled here rather than per voice: one 64-bit
   multiply a frame (the sum can reach a few million, so 32 bits is not
   enough) costs less than folding the constant into every voice's gain,
   and it is the sum that has to be kept under full scale anyway. */
#define WAVE_HEADROOM_CB 10L

void wave_render(short *out, long frames)
{
    static long accL[WAVE_BLOCK], accR[WAVE_BLOCK];
    long i, n, hg;

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
    hg = db_gain_tenths(WAVE_HEADROOM_CB);
    while (frames > 0) {
        n = frames < WAVE_BLOCK ? frames : WAVE_BLOCK;
        mix(accL, accR, n);
        for (i = 0; i < n; i++) {
            long s = (long)(((lz_i64)accL[i] * hg) >> 16);
            if (s > 32767L) s = 32767L;
            if (s < -32768L) s = -32768L;
            out[2 * i] = (short)s;
            s = (long)(((lz_i64)accR[i] * hg) >> 16);
            if (s > 32767L) s = 32767L;
            if (s < -32768L) s = -32768L;
            out[2 * i + 1] = (short)s;
        }
        out += 2 * n;
        frames -= n;
    }
}
