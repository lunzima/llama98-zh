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
 * ONE INSTRUMENT AT A TIME. A KAC-coded sample is not readable where it
 * lies - it has to be decoded - and decoding the whole bank costs 632 KiB on
 * the ten-instrument bank this ships with. But the front end forces a single
 * GM program on every channel bar the drum kit (common/smf.c), and a program
 * resolves to exactly one instrument: ksb_zone looks in that instrument's
 * zones and gives up rather than crossing into another one. So a piece
 * sounds exactly one instrument, and that is what gets expanded - 13 to
 * 171 KiB, median 46, on the same bank. ksb_unpack takes the program and
 * fills in only what that instrument reaches; every other sample reads as
 * absent, and ksb_zone says so by returning -1, which the synth already
 * counts (midi_wave.c's wave_unvoiced).
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

/* The format version this header carries: the third H of KSB1's leading
   <HHHHI>, between n_sample and the data size. One value, and a bank that
   states another does not open - a version field that is written and never
   consulted states nothing. */
#define KSB_FORMAT_VERSION 1

/* One key zone. Everything the synth needs to start a note, resolved -
   there is no second lookup once this is in hand. */
typedef struct {
    const unsigned char *pcm;
    int   codec;
    long  frames;
    long  loop_start;
    long  loop_len;             /* 0 when the sample does not loop */
    long  rate;                 /* frames a second, as stored */
    int   unity;                /* the MIDI note it was recorded at */
    int   fine;                 /* cents, added to unity */
    int   gain;                 /* dB, at most 0 */
    /* The volume envelope, WITHOUT WHICH EVERY NOTE IS AN ORGAN NOTE.
       The banks' pianos loop for ever and put their whole decay in here:
       sustain is nothing and the decay runs for tens of seconds. Times are
       milliseconds and mean the DLS thing - the time to fall 96 dB, the
       envelope being linear in decibels - and the sustain is an ATTENUATION
       in tenths of a decibel, 0 for full and 960 for silence, so the player
       needs no logarithm. */
    int   attack_ms;
    int   decay_ms;
    int   release_ms;
    int   sustain_cb;           /* tenths of a dB of attenuation */
} KsbZone;

#define KSB_MAXSAMPLES 512

typedef struct {
    const unsigned char *img;
    long  size;
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
   through the same fallback ksb_zone uses, so the figure is for the
   instrument that will actually sound. */
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

const char *ksb_errstr(int rc);

/* How many instruments, and the GM program of each. For listing. */
int ksb_count(const KsbBank *b);
int ksb_program(const KsbBank *b, int i);

/* Fill `z` for this program and note. Returns 0 if something will sound.
 *
 * THE PROGRAM FALLS BACK, because a bank of ten instruments will be asked
 * for all 128. An exact program wins; failing that the first instrument in
 * the same GM family - program/8, so any of the eight pianos answers for the
 * others - and failing that the lowest numbered instrument in the bank.
 * Silence would be the wrong answer: a general MIDI file with no acoustic
 * bass would lose its whole bass line.
 *
 * The NOTE does not fall back. A note outside every zone of the instrument
 * that program resolved to returns -1 - it does not reach into another
 * instrument, which is what keeps the expansion above bounded by one
 * instrument. */
int ksb_zone(const KsbBank *b, int program, int note, KsbZone *z);

#define KSB_OK        0
#define KSB_EMAGIC    1
#define KSB_ESHORT    2
#define KSB_ERANGE    3
/* The header's format version is not this one: the whole bank refuses to
   open, before any table is walked. */
#define KSB_EFORMAT   4

#endif
