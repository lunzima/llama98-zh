/* ksb.h - the KunKun98 sample bank, read-only.
 *
 * Instruments -> key zones -> samples. Not a DLS: no RIFF, no GUIDs, no
 * articulation chunks, no pool indirection, because this reader has to sit
 * next to an inference engine on a machine with no filesystem cache.
 * ksb_build.py makes one from a DLS.
 *
 * THE WHOLE FILE IS THE IMAGE. Everything below points into the buffer the
 * caller loaded; nothing is copied and nothing is allocated. The caller owns
 * the bytes and must keep them alive as long as the bank is open.
 *
 * EXPANSION IS PER PROGRAM OR WHOLE. A KAC-coded sample is not readable
 * where it lies - it has to be decoded. ksb_unpack decodes what one program
 * reaches (13 to 171 KiB, median 46, on the ten-instrument bank this ships
 * with); ksb_unpack_all decodes every KAC sample, which is what a file with
 * program changes needs and what costs memory (1.9 MiB for the full GM
 * bank). ksb_zone looks only in the instrument a program resolves to and
 * gives up rather than crossing into another one, so the one-program
 * expansion stays bounded.
 *
 * C89, and BIG-ENDIAN SAFE: every field is assembled byte by byte, so this
 * is the same on NT/MIPS as on x86. Casting into the image would be both
 * endian-wrong and, on ARMv5TE, an unaligned load that returns a rotated
 * word without faulting.
 */
#ifndef LZ_KSB_H
#define LZ_KSB_H

/* How the frames are stored. Two codings.
 *
 * KSB_PCM is the uncompressed one: signed 16-bit little-endian, two bytes a
 * frame, read straight out of the bank. KSB_KAC is one whole codec image per
 * sample; its own header carries the frame count, and it has to be decoded
 * into caller memory by ksb_unpack before anything plays it.
 *
 * The mixer reads plain 16-bit samples either way - a zone reports KSB_PCM
 * once it is in hand - and only the loader knows that some of them arrived
 * through the codec. */
#define KSB_PCM 0 /* signed 16-bit little-endian, two bytes a frame */
#define KSB_KAC 1 /* a KAC image; its own header carries the frame count */

/* The format version this header carries: the fourth H of KSB1's
   leading <HHHHI>, between n_sample and the data size. One value, and a bank that
   states another does not open - a version field that is written and never
   consulted states nothing. */
#define KSB_FORMAT_VERSION 1

/* One key zone. Everything the synth needs to start a note, resolved -
   there is no second lookup once this is in hand. */
/* THE FIELD ORDER FOLLOWS THE DLS RECORD. A region there is a header
   (`rgnh`: key and velocity ranges, key group), a wave link (`wlnk`: the
   table index), a wave sample (`wsmp`: unity, fine tune, gain) and then
   the articulation's connections, which this reader stores only for the
   destinations it plays. Keeping the same order here makes a field's
   meaning findable from either side.

   The sample itself comes between the link and the tuning because the
   mixer reads it on every frame: pointer, codec, length, loop, rate are
   the hot half of the zone, and everything from `unity` on is read once
   when the note starts. */
typedef struct {
    /* rgnh: which notes and how hard, and the exclusive class. A region
       split by velocity picks up a layer that shares its keys, so both
       ranges are filtered. `key_group` is the DLS key group - notes in
       one choke each other - and 0 is no group at all. */
    int   vel_min, vel_max;
    int   key_group;
    int   self_non_exclusive;   /* rgnh fusOptions bit 0 */
    int   region_id;            /* which region, for that flag */
    /* wlnk and wsmp: the sample and its tuning. */
    const unsigned char *pcm;
    int   codec;
    long  frames;
    long  loop_start;
    long  loop_len;             /* 0 when the sample does not loop */
    long  rate;                 /* frames a second, as stored */
    /* The note the sample is stored at, and the cents added to it:
       (note - unity) * 100 + fine. Fine is a SHORT because a DLS
       region's is: MT32.dls states -1100 on a French horn, and a byte
       held that by clamping, which played the note 972 cents sharp. */
    int   unity;
    int   fine;                 /* cents, added to unity */
    int   gain;                 /* TENTHS OF A DECIBEL, SIGNED: negative
                                   attenuates, positive boosts. Clamped
                                   by the mixer at +12 dB on the boost
                                   side and at the envelope floor on the
                                   other - see db_gain_of in wave.c,
                                   which is the authority.

                                   THIS IS AN ENGINE EXTENSION. A DLS
                                   attenuation is negative and the
                                   reference player clamps a positive one
                                   to full scale, so a bank that leans
                                   on the boost to reach the level an
                                   arrangement expects will play quiet
                                   there. It exists so a source 12 dB
                                   under the reference can still be
                                   brought to it; past that the program
                                   is unreachable and the selection
                                   refuses the source rather than
                                   promising a gain it cannot write.

                                   The field used to be documented as
                                   "at most 0" with no table entry for a
                                   positive value, which described the
                                   reference player and not this mixer,
                                   and sent the calibration to clamp at
                                   a limit the engine does not have. */
    /* THE STATIC ARTICULATORS, in destination-code order: pan, the mod
       LFO's rate and delay, EG1's times, then EG2's. */
    int   pan;                  /* tenths of a percent, -500..500 */
    int   lfo_tc;               /* 2^(tc/1200) Hz */
    int   lfo_delay_tc;         /* 2^(tc/1200) seconds */
    /* The volume envelope, WITHOUT WHICH EVERY NOTE IS AN ORGAN NOTE.
       The banks' pianos loop for ever and put their whole decay in here:
       sustain is nothing and the decay runs for tens of seconds. Times
       are milliseconds and mean the DLS thing - the time to fall 96 dB,
       the envelope being linear in decibels - and the sustain is an
       ATTENUATION in tenths of a decibel, 0 for full and 960 for
       silence, so the player needs no logarithm. */
    int   attack_ms;
    int   decay_ms;
    int   release_ms;
    int   sustain_cb;           /* tenths of a dB of attenuation */
    /* EG2: the same times, but targeting whatever the region modulates
       rather than the volume. Its sustain is a LEVEL in tenths of a
       percent - 1000 holds the modulator's depth and 0 holds none of it
       - not the attenuation the volume envelope carries. */
    int   mattack_ms;
    int   mdecay_ms;
    int   mrelease_ms;
    int   msustain_level;       /* tenths of a percent of the depth held */
    /* THE MODULATOR DEPTHS, grouped by destination. Units are the
       source's own: cents where the target is pitch, tenths of a
       decibel where it is gain, and time cents across the source's full
       range where it is a time. */
    int   lfo_pitch_tc;         /* modLfo -> pitch, cents */
    int   wheel_pitch_tc;       /* the same depth, scaled by CC1 */
    int   env_pitch_tc;         /* modEnv -> pitch, cents */
    int   key_pitch;            /* keyNum -> pitch, cents at key 128 */
    int   lfo_gain;             /* modLfo -> gain, centibels */
    /* Velocity's own attenuation, as the region states it: the value at
       velocity zero, through `vel_gain_curve` (0 linear, 1 concave).
       CARRIED, NOT APPLIED - the mixer uses velocity squared, which is
       what the reference measures (see wave.c). */
    int   vel_gain_cb;
    int   vel_gain_curve;
    /* The two envelopes' source scaling: EG1's decay by key and its
       attack by velocity, then EG2's. */
    int   key_decay_tc;
    int   vel_attack_tc;
    int   mdecay_key_tc;
    int   mattack_vel_tc;
    /* THE ENVELOPES' OPENING STAGES, in milliseconds. A delay holds the
       envelope at zero - silence for EG1, no modulation for EG2 - and a
       hold keeps it at its peak before the decay starts. 0 means the
       stage is absent, which is what a bank that never declares one
       plays. */
    int   vdelay_ms;
    int   vhold_ms;
    int   mdelay_ms;
    int   mhold_ms;
    /* keyNum -> hold, time cents per unit of source. The player scales
       the hold by 2^(tc * note / 128 / 1200), the same shape the two
       decay scalings use. */
    int   vhold_key_tc;
    int   mhold_key_tc;
    /* THE VIBRATO LFO, separate from the mod LFO. Its frequency and
       delay are absolute time cents - 0 is 8.1758 Hz - and the two
       depths are cents: the free one applied to every note, and the one
       CC1 scales, both as the DLS states them. */
    int   vibfreq_tc;
    int   vibdelay_tc;
    int   vib_pitch;
    int   vib_wheel_pitch;
    /* THE LOW PASS FILTER. The cutoff is in absolute cents, where 13500
       is open and 1500 the lowest the format states; the resonance is
       centibels; and the five depths are the cents each source moves the
       cutoff by, per unit of that source - the modulation envelope's
       level, velocity, key number, and the two LFO triangles. All zero
       depths with the cutoff at 13500 and no resonance is the state a
       bank that states no filter at all resolves to, and the player
       skips the filter for it. */
    int   filt_fc;              /* absolute cents, 1500..13500 */
    int   filt_q;               /* centibels, 0..960 */
    int   filt_env;             /* modEnv -> cutoff, cents */
    int   filt_vel;             /* velocity -> cutoff, cents */
    int   filt_key;             /* keyNum -> cutoff, cents */
    int   filt_lfo;             /* modLFO -> cutoff, cents */
    int   filt_vib;             /* vibLFO -> cutoff, cents */
} KsbZone;

/* The sample-table arrays below live in the handle, so the bound is a
   compile-time one. gm.dls alone is 412 samples; merging a second sound
   set over it is what pushed the limit - gm.dls with the whole FF8 bank
   laid over it is 605 - and FluidR3's full GM set is 1327, so 2048 is
   the first power of two that holds it. The handle grows by eight bytes
   per entry, and every handle in the tree is static except the console
   player's own. */
#define KSB_MAXSAMPLES 2048

typedef struct {
    const unsigned char *img;
    int   n_inst, n_region, n_sample;
    long  inst_off, region_off, sample_off, data_off;
    /* Filled by ksb_unpack; until then pcm is NULL and every pcm_at is -1.
       pcm_at[si] >= 0 means sample si is expanded and starts there. */
    const short *pcm;
    long  pcm_at[KSB_MAXSAMPLES];
    long  pcm_len[KSB_MAXSAMPLES];
} KsbBank;

/* 0 on success. Rejects anything it cannot read rather than reading it
   wrong: bad magic, a truncated table, a sample whose data runs past the end
   of the image, a codec this reader does not have. A bank that opens can be
   played without any further bounds testing in the mixing loop, which is the
   point. */
int ksb_open(KsbBank *b, const unsigned char *image, long size);

/* Frames `program`'s own zones expand to, and twice that in bytes. Cheap:
   headers only, no decoding. A program the bank does not carry resolves
   through the same fallback the zone lookup uses, so the figure is for
   the instrument that will actually sound. */
long ksb_pcm_frames(const KsbBank *b, int program);

/* The largest figure above, over every instrument the bank carries. Size a
   buffer with this and any program fits, without having to know which one
   will be asked for. */
long ksb_pcm_max_frames(const KsbBank *b);

/* Expand the samples `program` reaches into `pcm`, and point the bank's
   sample table there as KSB_PCM. Everything unreached is marked absent, so a
   later ksb_zone for it returns -1.

   Call once when the program is decided; call again with another program to
   change instrument, which is why it rewrites the whole table rather than
   adding to it. `pcm` must hold ksb_pcm_max_frames() shorts and outlive the
   bank. Returns 0 on success. */
int ksb_unpack(KsbBank *b, short *pcm, int program);

/* Frames every KAC-coded sample in the bank expands to, for a player that
 * wants more than one program at a time. Uncompressed samples read in place
 * and contribute nothing. A bank whose images do not open returns -1. */
long ksb_pcm_all_frames(const KsbBank *b);

/* Expand every KAC sample in the bank into `pcm`, which must hold
 * ksb_pcm_all_frames() shorts. Unlike ksb_unpack this keeps all programs
 * available at once, so program changes select real instruments. */
int ksb_unpack_all(KsbBank *b, short *pcm);

const char *ksb_errstr(int rc);

/* How many instruments, and the GM program of each. For listing. */
int ksb_count(const KsbBank *b);
int ksb_program(const KsbBank *b, int i);

/* Fill `z` for this patch and note. Returns 0 if something will sound.
 *
 * THE PATCH IS A (BANK MSB, BANK LSB, PROGRAM) TRIPLE, the pair being the
 * DLS bank split the way MIDI names it: bits 0-6 of the DLS bank are CC32
 * and bits 8-14 are CC0. A DLS file may hold several banks of one program
 * - FF8.dls holds thirty - and those instruments are reachable only
 * through this call.
 *
 * THE PROGRAM FALLS BACK, because a bank of ten instruments will be asked
 * for all 128. An exact triple wins; failing that the bank-0 record of the
 * same program; failing that the first instrument in the same GM family -
 * program/8, so any of the eight pianos answers for the others; and
 * failing that the lowest numbered instrument in bank 0. Silence would be
 * the wrong answer: a general MIDI file with no acoustic bass would lose
 * its whole bass line. A file that selects a bank this bank does not carry
 * lands on the GM set rather than on nothing.
 *
 * Program 128 is the percussion kit. MIDI programs stop at 127, so the
 * drum channel's key map takes a number no melodic program can name. It is
 * also the one program that does not fall back, and it is stored in bank
 * 0: a melodic instrument answering MIDI 36 would be a wrong note, which
 * is harder to miss than a gap.
 *
 * The NOTE does not fall back. A note outside every zone of the instrument
 * that the patch resolved to returns -1 - it does not reach into another
 * instrument, which is what keeps the expansion above bounded by one
 * instrument. */
/* The regions that sound for this patch, note AND velocity, in file
 * order. Both ranges have to match. A bank may layer several regions over
 * one key - a velocity split, a stacked instrument - and each one is a
 * separate voice in the player, so the reader enumerates them rather than
 * choosing one.
 *
 * ksb_zone_count returns how many regions match. ksb_zone_at fills the
 * `index`-th of them and returns 1, or 0 when `index` is out of range.
 * Only regions whose sample is present in the bank's sample table are
 * part of the sequence: one ksb_unpack has not expanded cannot sound, and
 * handing it to the mixer would read a buffer that is not there. A bank
 * where every match is absent therefore counts zero, the same answer a
 * note outside every key range gets.
 *
 * The patch falls back as described above, and the note does not. */
int ksb_zone_count(const KsbBank *b, int msb, int lsb, int program,
                   int note, int vel);
int ksb_zone_at(const KsbBank *b, int msb, int lsb, int program,
                int note, int vel, int index, KsbZone *z);

#define KSB_OK        0
#define KSB_EMAGIC    1
#define KSB_ESHORT    2
#define KSB_ERANGE    3
/* The header's format version is not this one: the whole bank refuses to
   open, before any table is walked. */
#define KSB_EFORMAT   4

#endif
