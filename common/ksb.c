/* ksb.c - see ksb.h. C89. */
#include "kac.h"
#include "ksb.h"

#define HDR      16
#define INST_SZ   4
#define REG_SZ   16
#define SMP_SZ   20

/* Little-endian, assembled a byte at a time. See the header. */
static int rd16(const unsigned char *p)
{
    return (int)p[0] | ((int)p[1] << 8);
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

/* The instrument a program resolves to: an exact match, else the first in
   the same GM family, else the first in the bank. Returns -1 only for a bank
   with no instruments at all. */
static int pick_inst(const KsbBank *b, int program)
{
    int i, fam = -1, any = -1;
    for (i = 0; i < b->n_inst; i++) {
        int p = b->img[b->inst_off + (long)i * INST_SZ];
        if (p == program) return i;
        if (fam < 0 && (p >> 3) == (program >> 3)) fam = i;
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
    int inst = pick_inst(b, program);
    const unsigned char *n;
    int count, first, i, j;
    long total = 0;
    if (inst < 0) return 0;
    n = b->img + b->inst_off + (long)inst * INST_SZ;
    count = n[1];
    first = rd16(n + 2);
    for (i = 0; i < count; i++) {
        const unsigned char *r = b->img + b->region_off + (long)(first + i) * REG_SZ;
        int si = rd16(r + 2);
        long f;
        /* Two zones of one instrument can share a sample, and it is expanded
           once, so count it once. */
        for (j = 0; j < i; j++) {
            const unsigned char *rj = b->img + b->region_off + (long)(first + j) * REG_SZ;
            if (rd16(rj + 2) == si) break;
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
    b->size = size;
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
       KAC - the same split the format has always had for a coding whose
       frames cannot be multiplied out from a byte count. Which one it is
       therefore has to come from the codec field, not be assumed. */
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
        if (rd16(r + 2) >= b->n_sample) {
            return KSB_ERANGE;
        }
    }
    for (i = 0; i < b->n_inst; i++) {
        const unsigned char *n = image + b->inst_off + (long)i * INST_SZ;
        if ((long)rd16(n + 2) + n[1] > b->n_region) {
            return KSB_ERANGE;
        }
    }
    return KSB_OK;
}

int ksb_unpack(KsbBank *b, short *pcm, int program)
{
    int inst = pick_inst(b, program);
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
    count = n[1];
    first = rd16(n + 2);
    for (i = 0; i < count; i++) {
        const unsigned char *r = b->img + b->region_off + (long)(first + i) * REG_SZ;
        int si = rd16(r + 2);
        const unsigned char *s = b->img + b->sample_off + (long)si * SMP_SZ;
        long off = rd32(s), len = rd32(s + 4);
        long ls = rd32(s + 8), ll = rd32(s + 12);
        kacdec_t d;
        long total = 0;
        if (b->pcm_at[si] >= 0) continue; /* already expanded by an earlier zone */
        if (s[18] != KSB_KAC) continue;   /* uncompressed: read in place */
        kacdec_init(&d);
        if (kacdec_open(&d, b->img + b->data_off + off, (size_t)len) != 0) {
            return KSB_ERANGE;
        }
        if (kacdec_channels(&d) != 1) {
            return KSB_ERANGE;
        }
        for (;;) {
            int32_t got = kacdec_read(&d, pcm + at, 4096u);
            if (got < 0) {
                return KSB_ERANGE;
            }
            if (got == 0) break;
            at += got;
            total += got;
        }
        /* THE LOOP BOUNDS, against the frame count that actually exists. For
           the uncompressed coding ksb_open checks them against the record,
           because that record IS the frame count; for KAC the count is only
           known once the image has been opened, so this is where it lands.
           Left unchecked the mixer would fold a note past the end of its own
           sample. */
        if (ll > 0 && (ls < 0 || ls + ll > total)) {
            return KSB_ERANGE;
        }
        b->pcm_at[si] = at - total;
        b->pcm_len[si] = total;
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

int ksb_zone(const KsbBank *b, int program, int note, KsbZone *z)
{
    const unsigned char *n, *r, *s;
    int i, first, count, si;

    i = pick_inst(b, program);
    if (i < 0) return -1;
    n = b->img + b->inst_off + (long)i * INST_SZ;
    count = n[1];
    first = rd16(n + 2);

    for (i = 0; i < count; i++) {
        r = b->img + b->region_off + (long)(first + i) * REG_SZ;
        if (note < r[0] || note > r[1]) continue;
        si = rd16(r + 2);
        s = b->img + b->sample_off + (long)si * SMP_SZ;
        if (s[18] == KSB_KAC) {
            /* Not expanded means not this instrument, or ksb_unpack has not
               run yet. Either way the note has no data, and saying so is
               what keeps the mixer from reading a buffer that is not there. */
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
        z->unity      = r[4];
        z->fine       = (signed char)r[5];
        z->gain       = (signed char)r[6];
        z->attack_ms  = rd16(r + 8);
        z->decay_ms   = rd16(r + 10);
        z->release_ms = rd16(r + 12);
        z->sustain_cb = rd16(r + 14);
        return 0;
    }
    return -1;
}
