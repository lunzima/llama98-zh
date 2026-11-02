/* ksb.c - see ksb.h. C89. */
#include "kac.h"
#include "ksb.h"

#define HDR      16
/* The instrument record: program, bank MSB, bank LSB, then region count
 * and first region.
 *
 * THE BANK IS TWO BYTES BECAUSE A DLS INSTRUMENT HAS ONE. Its header
 * states a 32-bit bank whose low seven bits are MIDI CC32 and whose next
 * seven are CC0, so a file that carries several banks of a program - and
 * FF8.dls carries thirty of them - names each instrument by that pair.
 * The reader looks a patch up on the triple.
 *
 * THE COUNT IS A SHORT: FluidR3 stacks 720 regions into one instrument
 * (a key and velocity lattice). Version 1 is this seven-byte record. */
#define INST_SZ   7
/* THE REGION RECORD, 104 BYTES, THE WHOLE LAYOUT AT ONCE. The fields
   the DLS states keep the DLS's own WIDTHS, and the converted
   articulators follow in the order of their destination codes. No
   field is packed around another: a wide value is stored wide, and the
   bytes nothing uses are reserved rather than closed by shifting the
   rest.

      0  key_lo u16      2  key_hi u16    4  vel_lo u16    6  vel_hi u16
      8  key_group u16  10  flags u16    12  sample u32   16  unity u16
     18  fine i16       20  gain i32     24  pan i16      26  lfo_tc i16
     28  lfo_delay i16  30  attack u16   32  decay u16    34  release u16
     36  sustain u16    38  mattack u16  40  mdecay u16   42  mrelease u16
     44  msustain u16   46  lfo_pitch i16 48 wheel_pitch i16
     50  env_pitch i16  52  key_pitch i16 54 lfo_gain i16 56 vel_gain i16
     58  vel_curve u8   59  flags2 u8    60  key_decay i16
     62  vel_attack i16 64  mdecay_key i16 66 mattack_vel i16
     68  vdelay u16     70  vhold u16    72  mdelay u16   74  mhold u16
     76  vhold_key i16  78  mhold_key i16 80 vibfreq i16  82 vibdelay i16
     84  vib_pitch i16  86  vib_wheel i16 88 filt_fc u16  90 filt_q u16
     92  filt_env i16   94  filt_vel i16 96  filt_key i16 98  filt_lfo i16
    100  filt_vib i16  102  reserved u16

   key/velocity ranges, key group, flags and unity are WORDs in the
   DLS; the wave-table index is a DWORD; fine is a SHORT; and the wsmp
   attenuation is a LONG. Storing any of them narrower lost a value the
   file states - MT32.dls's -1100-cent fine played 972 cents sharp in a
   byte. */
#define REG_SZ   104
#define SMP_SZ   20

/* Little-endian, assembled a byte at a time. See the header. */
static int rd16(const unsigned char *p)
{
    return (int)p[0] | ((int)p[1] << 8);
}

/* The same 16 bits as a signed value: pan and the LFO times are signed
   in the record, and a value like -32768 would otherwise read as an
   envelope time of 65536 ms. */
static int rd16s(const unsigned char *p)
{
    int v = rd16(p);
    return (v & 0x8000) ? v - 0x10000 : v;
}

/* Exactly 32 bits, signed. The `off < 0` and `n < 1` checks below only reject
   an oversized field because of this: on a `long` wider than 32 bits, the
   same bytes read back positive and huge instead of negative, and pass.
   int32_t is lz_int.h's, which is where the tree keeps the fixed-width
   names. */
static int32_t rd32(const unsigned char *p)
{
    return (int32_t)p[0] | ((int32_t)p[1] << 8)
         | ((int32_t)p[2] << 16) | ((int32_t)p[3] << 24);
}

const char *ksb_errstr(int rc)
{
    switch (rc) {
    case KSB_OK:     return "ok";
    case KSB_EMAGIC: return "not a KSB bank";
    case KSB_ESHORT: return "truncated";
    case KSB_ERANGE: return "a table entry points outside the file";
    case KSB_EFORMAT: return "this bank's format version is not this one";
    }
    return "?";
}

/* The instrument a patch resolves to. THE TRIPLE IS (MSB, LSB, PROGRAM),
   in that order of specificity:
     - an exact (msb, lsb, program) record;
     - for a melodic program, a bank-0 record of the same program;
     - a bank-0 record in the same GM family - program/8, so any of the
       eight pianos answers for the others;
     - the first instrument in bank 0.
   A bank the file carries nothing under therefore falls back to the GM
   set rather than to silence, which is what a file sending CC0/CC32 at a
   bank-0-only bank needs.

   Program 128 and above is the percussion kit and does not fall back -
   an eight-instrument melodic bank has nothing to answer a drum key with,
   and the first instrument in the bank playing MIDI 36 is a wrong note
   rather than a missing one. A drum patch is stored in bank 0; a caller
   that wants the kit for a drum channel passes 0, 0. Returns -1 when
   there is no instrument to answer with. */
static int pick_inst(const KsbBank *b, int msb, int lsb, int program)
{
    int i, fam = -1, any = -1;
    for (i = 0; i < b->n_inst; i++) {
        const unsigned char *n = b->img + b->inst_off + (long)i * INST_SZ;
        if (n[0] == program && n[1] == msb && n[2] == lsb) return i;
    }
    if (program >= 128) return -1;
    for (i = 0; i < b->n_inst; i++) {
        const unsigned char *n = b->img + b->inst_off + (long)i * INST_SZ;
        if (n[1] != 0 || n[2] != 0) continue;
        if (n[0] == program) return i;
        if (fam < 0 && (n[0] >> 3) == (program >> 3)) fam = i;
        if (any < 0) any = i;
    }
    return fam >= 0 ? fam : any;
}

/* Frames sample `si` EXPANDS to, which for the uncompressed coding is none: it
   is read straight out of the bank and takes no room in the caller's buffer
   however many frames it has. For KAC the count lives inside the image, since
   the sample table's length field is the coded byte count - reading it means
   opening the image, which is validation and the chunk walk and no decoding. */
static long sample_frames(const KsbBank *b, int si)
{
    const unsigned char *s = b->img + b->sample_off + (long)si * SMP_SZ;
    kacdec_t d;
    if (s[18] != KSB_KAC) {
        return 0;
    }
    kacdec_init(&d);
    if (kacdec_open(&d, b->img + b->data_off + rd32(s), (size_t)rd32(s + 4)) != 0)
        return -1;
    return (long)kacdec_total(&d);
}

long ksb_pcm_frames(const KsbBank *b, int program)
{
    int inst = pick_inst(b, 0, 0, program);
    const unsigned char *n;
    int count, first, i, j;
    long total = 0;
    if (inst < 0) return 0;
    n = b->img + b->inst_off + (long)inst * INST_SZ;
    count = rd16(n + 3);
    first = rd16(n + 5);
    for (i = 0; i < count; i++) {
        const unsigned char *r = b->img + b->region_off + (long)(first + i) * REG_SZ;
        int si = (int)rd32(r + 12);
        long f;
        /* Two zones of one instrument can share a sample, and it is expanded
           once, so count it once. */
        for (j = 0; j < i; j++) {
            const unsigned char *rj = b->img + b->region_off + (long)(first + j) * REG_SZ;
            if (rd16(rj + 8) == si) break;
        }
        if (j < i) continue;
        f = sample_frames(b, si);
        if (f > 0) total += f;
    }
    return total;
}

long ksb_pcm_max_frames(const KsbBank *b)
{
    long best = 0;
    int i;
    for (i = 0; i < b->n_inst; i++) {
        long f = ksb_pcm_frames(b, b->img[b->inst_off + (long)i * INST_SZ]);
        if (f > best) best = f;
    }
    return best;
}

int ksb_open(KsbBank *b, const unsigned char *image, long size)
{
    long need;
    int i;
    kacdec_t probe;

    if (size < HDR || image[0] != 'K' || image[1] != 'S'
        || image[2] != 'B' || image[3] != '1') {
        return KSB_EMAGIC;
    }
    b->img = image;
    b->n_inst   = rd16(image + 4);
    b->n_region = rd16(image + 6);
    b->n_sample = rd16(image + 8);
    if (rd16(image + 10) != KSB_FORMAT_VERSION) {
        return KSB_EFORMAT;
    }
    if (b->n_sample > KSB_MAXSAMPLES) {
        return KSB_ERANGE;
    }

    b->pcm = 0;
    b->inst_off   = HDR;
    b->region_off = b->inst_off   + (long)b->n_inst   * INST_SZ;
    b->sample_off = b->region_off + (long)b->n_region * REG_SZ;
    b->data_off   = b->sample_off + (long)b->n_sample * SMP_SZ;
    need = b->data_off + rd32(image + 12);
    if (need > size || b->data_off < HDR) {
        return KSB_ESHORT;
    }
    for (i = 0; i < b->n_sample; i++) {
        b->pcm_at[i] = -1;
        b->pcm_len[i] = 0;
    }

    /* EVERY REFERENCE CHECKED HERE, once, so the mixing loop can read without
       testing. A zone that got past this cannot walk off the end of the image
       no matter what the sequencer asks for.

       The length field is FRAMES for the uncompressed coding and BYTES for
       KAC, whose frames cannot be multiplied out from a byte count. Which
       one it is comes from the codec field, not from an assumption. */
    for (i = 0; i < b->n_sample; i++) {
        const unsigned char *s = image + b->sample_off + (long)i * SMP_SZ;
        long off = rd32(s), n = rd32(s + 4);
        long ls = rd32(s + 8), ll = rd32(s + 12);
        int codec = s[18];
        long bytes;
        if (codec != KSB_PCM && codec != KSB_KAC) {
            return KSB_ERANGE;
        }
        bytes = codec == KSB_PCM ? n * 2L : n;
        if (off < 0 || n < 1 || off + bytes > size - b->data_off) {
            return KSB_ERANGE;
        }
        if (codec == KSB_KAC) {
            /* The frame count is inside the image rather than in the record,
               so the loop cannot be checked against a field here - the record
               only says how many BYTES the image is. What is checked now is
               that the image opens, which is what makes the frame count
               reachable at all, and the loop bounds are checked against that
               count in ksb_unpack, where it exists. */
            kacdec_init(&probe);
            if (kacdec_open(&probe, image + b->data_off + off, (size_t)n) != 0) {
                return KSB_ERANGE;
            }
        } else if (ll > 0 && (ls < 0 || ls + ll > n)) {
            return KSB_ERANGE;
        }
        if (rd16(s + 16) < 1) {
            return KSB_ERANGE;
        }
    }
    for (i = 0; i < b->n_region; i++) {
        const unsigned char *r = image + b->region_off + (long)i * REG_SZ;
        if (rd32(r + 12) >= b->n_sample) {
            return KSB_ERANGE;
        }
    }
    for (i = 0; i < b->n_inst; i++) {
        const unsigned char *n = image + b->inst_off + (long)i * INST_SZ;
        if (n[1] > 127 || n[2] > 127) {
            return KSB_ERANGE;
        }
        if ((long)rd16(n + 5) + (long)rd16(n + 3) > b->n_region) {
            return KSB_ERANGE;
        }
    }
    return KSB_OK;
}

/* Decode sample `si` into `dst`. Returns 0 when it was decoded, 1 when it is
   not KAC and is read in place, and -1 when the image or its loop bounds are
   wrong. `frames` is the decoded length. */
static int decode_one(const KsbBank *b, int si, short *dst, long *frames)
{
    const unsigned char *s = b->img + b->sample_off + (long)si * SMP_SZ;
    long off = rd32(s), len = rd32(s + 4);
    long ls = rd32(s + 8), ll = rd32(s + 12);
    kacdec_t d;
    long total = 0;

    *frames = 0;
    if (s[18] != KSB_KAC) {
        return 1;
    }
    kacdec_init(&d);
    if (kacdec_open(&d, b->img + b->data_off + off, (size_t)len) != 0) {
        return -1;
    }
    if (kacdec_channels(&d) != 1) {
        return -1;
    }
    for (;;) {
        int32_t got = kacdec_read(&d, dst, 4096u);
        if (got < 0) {
            return -1;
        }
        if (got == 0) break;
        dst += got;
        total += got;
    }
    /* THE LOOP BOUNDS, against the frame count that actually exists. For
       the uncompressed coding ksb_open checks them against the record,
       because that record IS the frame count; for KAC the count is only
       known once the image has been opened, so this is where it lands.
       Left unchecked the mixer would fold a note past the end of its own
       sample. */
    if (ll > 0 && (ls < 0 || ls + ll > total)) {
        return -1;
    }
    *frames = total;
    return 0;
}

int ksb_unpack(KsbBank *b, short *pcm, int program)
{
    int inst = pick_inst(b, 0, 0, program);
    const unsigned char *n;
    int count, first, i;
    long at = 0;

    for (i = 0; i < b->n_sample; i++) {
        b->pcm_at[i] = -1;
        b->pcm_len[i] = 0;
    }
    if (inst < 0) {
        return KSB_ERANGE;
    }
    n = b->img + b->inst_off + (long)inst * INST_SZ;
    count = rd16(n + 3);
    first = rd16(n + 5);
    for (i = 0; i < count; i++) {
        const unsigned char *r = b->img + b->region_off + (long)(first + i) * REG_SZ;
        int si = (int)rd32(r + 12);
        long total = 0;
        int rc;
        if (b->pcm_at[si] >= 0) continue; /* already expanded by an earlier zone */
        rc = decode_one(b, si, pcm + at, &total);
        if (rc < 0) return KSB_ERANGE;
        if (rc > 0) continue;             /* uncompressed: read in place */
        b->pcm_at[si] = at;
        b->pcm_len[si] = total;
        at += total;
    }
    b->pcm = pcm;
    return KSB_OK;
}

long ksb_pcm_all_frames(const KsbBank *b)
{
    int  i;
    long total = 0;

    if (!b)
        return -1;
    for (i = 0; i < b->n_sample; i++) {
        const unsigned char *s = b->img + b->sample_off + (long)i * SMP_SZ;
        kacdec_t d;
        if (s[18] != KSB_KAC)
            continue;
        kacdec_init(&d);
        if (kacdec_open(&d, b->img + b->data_off + rd32(s), (size_t)rd32(s + 4)) != 0)
            return -1;
        total += (long)kacdec_total(&d);
    }
    return total;
}

int ksb_unpack_all(KsbBank *b, short *pcm)
{
    int  i;
    long at = 0;

    for (i = 0; i < b->n_sample; i++) {
        b->pcm_at[i] = -1;
        b->pcm_len[i] = 0;
    }
    for (i = 0; i < b->n_sample; i++) {
        long total = 0;
        int rc = decode_one(b, i, pcm + at, &total);
        if (rc < 0) return KSB_ERANGE;
        if (rc > 0) continue;             /* uncompressed: read in place */
        b->pcm_at[i] = at;
        b->pcm_len[i] = total;
        at += total;
    }
    b->pcm = pcm;
    return KSB_OK;
}

int ksb_count(const KsbBank *b) { return b->n_inst; }

int ksb_program(const KsbBank *b, int i)
{
    if (i < 0 || i >= b->n_inst) return -1;
    return b->img[b->inst_off + (long)i * INST_SZ];
}

/* Fill `z` from region record `ri`. Returns 0 when the region has data to
   play and -1 when it does not: a KAC image ksb_unpack has not expanded is
   absent from the sample table, and saying so is what keeps the mixer from
   reading a buffer that is not there. */
static int zone_fill(const KsbBank *b, int ri, KsbZone *z)
{
    const unsigned char *r, *s;
    int si;

    r = b->img + b->region_off + (long)ri * REG_SZ;
    si = (int)rd32(r + 12);
    s = b->img + b->sample_off + (long)si * SMP_SZ;
    if (s[18] == KSB_KAC) {
        if (!b->pcm || b->pcm_at[si] < 0) return -1;
        z->pcm    = (const unsigned char *)(b->pcm + b->pcm_at[si]);
        z->codec  = KSB_PCM;
        z->frames = b->pcm_len[si];
    } else {
        z->pcm    = b->img + b->data_off + rd32(s);
        z->codec  = s[18];
        z->frames = rd32(s + 4);
    }
    z->loop_start = rd32(s + 8);
    z->loop_len   = rd32(s + 12);
    z->rate       = rd16(s + 16);
    z->unity      = rd16(r + 16);
    z->fine       = rd16s(r + 18);
    /* A SHORT: FF8.dls asks for a -20.3 dB trim, which a signed byte
       cannot state. */
    z->gain       = (int)rd32(r + 20);
    /* The static articulators follow in DLS destination-code order:
       pan, the mod LFO's rate and delay, then EG1's times, then
       EG2's. */
    z->pan          = rd16s(r + 24);
    z->lfo_tc       = rd16s(r + 26);
    z->lfo_delay_tc = rd16s(r + 28);
    z->attack_ms  = rd16(r + 30);
    z->decay_ms   = rd16(r + 32);
    z->release_ms = rd16(r + 34);
    z->sustain_cb = rd16(r + 36);
    z->mattack_ms   = rd16(r + 38);
    z->mdecay_ms    = rd16(r + 40);
    z->mrelease_ms  = rd16(r + 42);
    z->msustain_level = rd16(r + 44);
    /* The depth of each modulator, kept as the source states it,
       grouped by destination: pitch (the LFO's own depth, the part
       CC1 scales, and the pitch envelope), gain (the LFO's tremolo
       and velocity's attenuation), then the two envelopes' source
       scaling. */
    z->lfo_pitch_tc   = rd16s(r + 46);
    z->wheel_pitch_tc = rd16s(r + 48);
    z->env_pitch_tc   = rd16s(r + 50);
    z->key_pitch      = rd16s(r + 52);
    z->lfo_gain       = rd16s(r + 54);
    z->vel_gain_cb    = rd16s(r + 56);
    z->vel_gain_curve = r[58];
    z->key_decay_tc   = rd16s(r + 60);
    z->vel_attack_tc  = rd16s(r + 62);
    z->mdecay_key_tc  = rd16s(r + 64);
    z->mattack_vel_tc = rd16s(r + 66);
    /* The envelopes' opening stages, then the vibrato LFO. The four
       times are milliseconds and the two scalings time cents; the
       vibrato's rate and delay are the absolute time cents a DLS
       states, and its two depths are cents. */
    z->vdelay_ms      = rd16(r + 68);
    z->vhold_ms       = rd16(r + 70);
    z->mdelay_ms      = rd16(r + 72);
    z->mhold_ms       = rd16(r + 74);
    z->vhold_key_tc   = rd16s(r + 76);
    z->mhold_key_tc   = rd16s(r + 78);
    z->vibfreq_tc     = rd16s(r + 80);
    z->vibdelay_tc    = rd16s(r + 82);
    z->vib_pitch      = rd16s(r + 84);
    z->vib_wheel_pitch = rd16s(r + 86);
    /* The low pass filter, appended to the record: the cutoff and
       the resonance as they are stated, then the five depths that
       move the cutoff. */
    z->filt_fc        = rd16(r + 88);
    z->filt_q         = rd16(r + 90);
    z->filt_env       = rd16s(r + 92);
    z->filt_vel       = rd16s(r + 94);
    z->filt_key       = rd16s(r + 96);
    z->filt_lfo       = rd16s(r + 98);
    z->filt_vib       = rd16s(r + 100);
    z->vel_min   = rd16(r + 4);
    z->vel_max   = rd16(r + 6);
    z->key_group = rd16(r + 8);
    z->self_non_exclusive = rd16(r + 10) & 1;
    z->region_id = ri;
    return 0;
}

/* Walk the resolved instrument's regions in file order.
 *
 * `want` is the index of the matching region to fill, and a negative
 * `want` counts the matching regions instead of filling one. Returns 1
 * when a zone was filled, 0 when there is no such region, and the count
 * when `want` is negative. A region whose sample is not in the table is
 * not part of the sequence: it cannot sound, so it is skipped rather than
 * handed to the mixer. */
static int zone_scan(const KsbBank *b, int msb, int lsb, int program,
                     int note, int vel, int want, KsbZone *z)
{
    const unsigned char *n, *r;
    int i, first, count, seen = 0;

    i = pick_inst(b, msb, lsb, program);
    if (i < 0) return 0;
    n = b->img + b->inst_off + (long)i * INST_SZ;
    count = rd16(n + 3);
    first = rd16(n + 5);

    for (i = 0; i < count; i++) {
        r = b->img + b->region_off + (long)(first + i) * REG_SZ;
        /* ON BOTH RANGES. A bank that splits a key across two layers by
           velocity sounds one of them; matching both is what chooses
           which. */
        if (note < rd16(r) || note > rd16(r + 2)) continue;
        if (vel < rd16(r + 4) || vel > rd16(r + 6)) continue;
        if (zone_fill(b, first + i, z) != 0) continue;
        if (want < 0) {
            seen++;
            continue;
        }
        if (seen == want) return 1;
        seen++;
    }
    return want < 0 ? seen : 0;
}

int ksb_zone_count(const KsbBank *b, int msb, int lsb, int program,
                   int note, int vel)
{
    KsbZone z;
    return zone_scan(b, msb, lsb, program, note, vel, -1, &z);
}

int ksb_zone_at(const KsbBank *b, int msb, int lsb, int program,
                int note, int vel, int index, KsbZone *z)
{
    if (index < 0) return 0;
    return zone_scan(b, msb, lsb, program, note, vel, index, z);
}
