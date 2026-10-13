/* kac_dec.c -- the KAC decoder.
 *
 * The sample loop calls nothing, divides nothing and branches nowhere
 * except its own back edge; every ternary compiles to a conditional
 * move. Reloading a snapshot costs four kac_sat_w calls per channel.
 *
 * This file reads whole images of known length: it parses the header,
 * holds the file to the length it declares, and then decodes. It does
 * not encode anything and it does not take the data in pieces.
 */

#include "kac.h"

/* The slice data covering `samples` samples at a tier whose slice holds `n`:
 * the slice grid rounds up, and one slice is KAC_SLICE_SIZE bytes per channel.
 * A slice is that size at every tier whatever its N, so this one expression is
 * the whole of a chunk's byte extent -- the chunk walk, the closed form for a
 * whole-tier file, the block step and the index bounds all come out of here. */
static uint32_t kacdec_slice_bytes(uint32_t samples, unsigned n, unsigned channels)
{
    return ((samples + n - 1u) / n) * KAC_SLICE_SIZE * channels;
}

/* Switch the slice tables to one tier. */
static void kacdec_use_tier(kacdec_t *restrict d, unsigned tier)
{
    unsigned ti = tier - 1u;
    d->n        = kac_tiers[ti].n;
    d->w        = kac_tiers[ti].w;
    d->shift    = 64u - d->w;
    d->dqt      = kac_pick_dqt(tier);
}

/* Point the decoder at the block covering [start, start + len). `slice_off` is
 * the data-area offset of its first slice and `tier` the tier its slices are
 * written for -- the tier comes from the data, a chunked block's from its own
 * byte at slice_off - 1 and a whole-tier one's from the header; `start` and
 * `len` come from the block grid. Called at open, at a block boundary, at a
 * loop wrap and after a seek -- never inside the sample loop. */
static void kacdec_enter(kacdec_t *restrict d, uint32_t start, uint32_t slice_off, unsigned tier,
                         uint32_t len)
{
    kacdec_use_tier(d, tier);
    d->blk_start = start;
    d->blk_len   = len;
    d->blk_off   = slice_off;
}

/* One block of a chunked data area, from its tier byte: `start` and `slice_off`
 * as in kacdec_enter, and the block's length as the block grid allows it. The
 * tier byte is not judged here -- every caller has already put it through
 * kacdec_tier_byte, which is what keeps kac_tiers indexed inside its bounds. */
static void kacdec_set_block(kacdec_t *restrict d, uint32_t start, uint32_t slice_off)
{
    uint32_t left = d->total - start;
    kacdec_enter(d, start, slice_off, (unsigned)d->data[slice_off - 1u],
                 (left > KAC_CHUNK_SAMPLES) ? KAC_CHUNK_SAMPLES : left);
}

/* An open writes into the handle as it reads, so a candidate it goes on to
 * reject has already changed it. Clearing first is what makes a failed open
 * leave an inert handle rather than half of the previous use: no read
 * produces anything, so a caller can open again without re-initialising. */
static void kacdec_inert(kacdec_t *restrict d)
{
    d->data      = NULL;
    d->total     = 0u;
    d->pos       = 0u;
    d->loop_en   = 0;
    d->blk_start = 0u;
    d->blk_len   = 0u;
    d->blk_off   = 0u;
    d->init_snap = NULL;
    d->loop_snap = NULL;
    d->idx_off   = NULL;
    d->idx_snap  = NULL;
}

/* The state the decoder starts from: the initial state the preamble carries, or
 * the normative default when it carries none. Every weight of a loaded snapshot
 * passes through kac_sat_w, as the weights of any snapshot do. */
static void kacdec_start(kacdec_t *restrict d)
{
    if (d->init_snap) {
        kac_snap_load(&d->ch[0], d->init_snap);
        if (d->channels == 2)
            kac_snap_load(&d->ch[1], d->init_snap + KAC_SNAP_SIZE);
    } else {
        kac_ch_init(&d->ch[0]);
        kac_ch_init(&d->ch[1]);
    }
}

static int32_t kacdec_decode(kacdec_t *restrict d, int16_t *restrict out, uint32_t stride,
                             uint32_t cnt, int loop_en)
{
    int16_t        sink[KAC_MAX_CHANNELS];
    unsigned       w, shift, nch, n;
    const int16_t *dqt;
    const uint8_t *base;
    uint32_t       done = 0u;
    uint32_t       slice, seg, rem;
    unsigned       c, off;
    int            reload = 0;

    if (!d)
        return KAC_E_ARG;
    if (!out) {
        out    = sink;
        stride = 0u;
    }
    if (loop_en && !(d->flags & KAC_FLAG_LOOP))
        return KAC_E_NOLOOP;

    /* Slice invariant loads hoisted out of the sample loop. A CHUNKED block
     * boundary reloads them, which is outside that loop; a whole-tier file loads
     * them once per call. */
    w     = d->w;
    shift = d->shift;
    nch   = (unsigned)d->channels;
    n     = d->n;
    dqt   = d->dqt;
    base  = d->data + d->blk_off;

    slice = (d->pos - d->blk_start) / n;
    off   = (d->pos - d->blk_start) % n;

    while (done < cnt) {
        uint32_t stop = loop_en ? d->loop_end : d->total;

        if (d->pos >= stop) {
            if (!loop_en)
                return (int32_t)done; /* defensive: deliver what exists */
            kac_snap_load(&d->ch[0], d->loop_snap);
            if (nch == 2)
                kac_snap_load(&d->ch[1], d->loop_snap + KAC_SNAP_SIZE);
            d->pos = d->loop_start;
            if (d->chunked)
                kacdec_set_block(d, d->loop_blk_start, d->loop_blk_off);
            stop   = d->loop_end;
            reload = 1;
        } else if (d->chunked && d->pos >= d->blk_start + d->blk_len) {
            /* Step to the next block: its tier byte is the byte after this
             * block's slices. The chain was walked at open, so the only thing
             * this bound is for is a chain that runs past the data area. */
            uint32_t next = d->blk_off + kacdec_slice_bytes(d->blk_len, n, nch) + 1u;
            if (next >= d->data_bytes)
                return KAC_E_DATA;
            kacdec_set_block(d, d->blk_start + d->blk_len, next);
            reload = 1;
        }
        if (reload) {
            /* The prologue's set, less `nch`: the channel count cannot change
             * under a call, and there is nothing else in either list. */
            reload = 0;
            w      = d->w;
            shift  = d->shift;
            n      = d->n;
            dqt    = d->dqt;
            base   = d->data + d->blk_off;
            slice  = (d->pos - d->blk_start) / n;
            off    = (d->pos - d->blk_start) % n;
        }
        seg = n - off;
        rem = stop - d->pos;
        if (seg > rem)
            seg = rem;
        rem = d->blk_start + d->blk_len - d->pos; /* never cross a block: N changes there */
        if (seg > rem)
            seg = rem;
        rem = cnt - done;
        if (seg > rem)
            seg = rem;

        for (c = 0; c < nch; c++) {
            const uint8_t *restrict sp  = base + ((size_t)slice * nch + c) * KAC_SLICE_SIZE;
            uint64_t v                  = kac_get64(sp);
            const int16_t *restrict row = dqt + ((size_t)(unsigned)(v >> 60) << w);
            kac_ch *restrict ch         = &d->ch[c];
            int16_t *restrict op        = out + (size_t)done * stride + c;
            unsigned i;

            v <<= 4u + off * w;
            /* Two samples per pass; the tail below handles an odd seg. */
            for (i = 0; i + 1 < seg; i += 2) {
                unsigned qr0 = (unsigned)(v >> shift);
                unsigned qr1;
                v <<= w;
                qr1 = (unsigned)(v >> shift);
                v <<= w;
                op[0]      = (int16_t)kac_step(ch, row[qr0]);
                op[stride] = (int16_t)kac_step(ch, row[qr1]);
                op += 2 * stride;
            }
            if (i < seg) {
                unsigned qr = (unsigned)(v >> shift);
                v <<= w;
                *op = (int16_t)kac_step(ch, row[qr]);
                op += stride;
            }
        }
        d->pos += seg;
        done += seg;
        off += seg;
        if (off >= n) {
            off = 0u;
            slice++;
        }
    }
    return (int32_t)done;
}

void kacdec_init(kacdec_t *restrict d)
{
    uint8_t *p = (uint8_t *)d;
    size_t   i;
    for (i = 0; i < sizeof *d; i++)
        p[i] = 0;
}

void kacdec_close(kacdec_t *restrict d)
{
    kacdec_init(d);
}

uint32_t kacdec_sample_rate(const kacdec_t *d)
{
    return d->sample_rate;
}
uint32_t kacdec_total(const kacdec_t *d)
{
    return d->total;
}
unsigned kacdec_channels(const kacdec_t *d)
{
    return (unsigned)d->channels;
}
unsigned kacdec_tier(const kacdec_t *d)
{
    return (unsigned)d->tier;
}

void kacdec_set_loop(kacdec_t *restrict d, int enable)
{
    d->loop_en = enable ? 1 : 0;
}

/* Whether a tier number is one the format admits -- the range kac_int.h's tier
 * table spans. The header's Tier field and every chunk's tier byte are both
 * held to this. */
static int kacdec_tier_ok(unsigned tier)
{
    return tier >= (unsigned)KAC_TIER_MIN && tier <= (unsigned)KAC_TIER_MAX;
}

/* A chunk's tier byte, read and judged in one place: legal by kacdec_tier_ok,
 * and chunk 0's equal to the header's. `*n` comes back as that tier's slice
 * length, which is what every caller needs next -- and taking it from here is
 * what keeps the table indexed only after the range check. Who calls this
 * decides how the refusal is reported, since an open rejects the image and a
 * seek rejects the index entry.
 *
 * `is_chunk0` is what the callers know differently: the chunk walk reads it off
 * the byte's own position. A seek entry is not judged as chunk 0's -- the walk
 * validated the chain when the image was opened. */
static int kacdec_tier_byte(const kacdec_t *restrict d, unsigned b, int is_chunk0,
                            unsigned *restrict n)
{
    if (!kacdec_tier_ok(b) || (is_chunk0 && b != (unsigned)d->tier))
        return KAC_E_TIER;
    *n = kac_tiers[b - 1u].n;
    return KAC_OK;
}

/* The 36 byte header, decoded and judged: magic, Version, the reserved flag
 * bits, Channels, Tier and SampleRate, with *off coming back at the first byte
 * after them where the snapshots start. The loop fields and the four length
 * fields are decoded here too; what they have to satisfy is checked by the
 * caller, which is the one holding the rest of the image to check it against. */
static int kacdec_head(kacdec_t *restrict d, const uint8_t *restrict buf, size_t len,
                       size_t *restrict off)
{
    unsigned ti;

    if (!d || !buf)
        return KAC_E_ARG;
    if (len < KAC_HDR_SIZE)
        return KAC_E_SIZE;
    if (buf[0] != 'K' || buf[1] != 'A' || buf[2] != 'C' || buf[3] != 0)
        return KAC_E_MAGIC;

    d->version       = buf[4];
    d->flags         = buf[5];
    d->channels      = buf[6];
    d->tier          = buf[7];
    d->sample_rate   = kac_get32(buf + 8);
    d->total         = kac_get32(buf + 12);
    d->loop_start    = kac_get32(buf + 16);
    d->loop_end      = kac_get32(buf + 20);
    d->data_bytes    = kac_get32(buf + 24);
    d->index_stride  = kac_get32(buf + 28);
    d->index_entries = kac_get32(buf + 32);

    if (d->version != KAC_VERSION)
        return KAC_E_VERSION;
    if (d->flags & KAC_FLAG_RESERVED)
        return KAC_E_FLAGS;
    if (d->channels != 1 && d->channels != 2)
        return KAC_E_CHANNEL;
    if (!kacdec_tier_ok(d->tier))
        return KAC_E_TIER;

    ti         = (unsigned)d->tier - 1u;
    d->w       = kac_tiers[ti].w;
    d->n       = kac_tiers[ti].n;
    d->shift   = 64u - d->w;
    d->chunked = (d->flags & KAC_FLAG_CHUNKED) ? 1 : 0;
    d->dqt     = kac_pick_dqt(d->tier);
    *off       = KAC_HDR_SIZE;
    return KAC_OK;
}

/* Everything a file has to state, validated before anything decodes: a file
 * declares its own length and its own loop, and both are held to that. */
static int open_image(kacdec_t *restrict d, const uint8_t *restrict buf, size_t len)
{
    uint64_t need;
    size_t   off, snap_off;
    int      rc;

    rc        = kacdec_head(d, buf, len, &off);
    if (rc != KAC_OK)
        return rc;

    if (d->total == 0 || d->total > KAC_MAX_TOTAL)
        return KAC_E_TOTAL;

    if (d->flags & KAC_FLAG_LOOP) {
        if (d->loop_start >= d->loop_end)
            return KAC_E_LOOP;
        if (d->loop_start >= d->total)
            return KAC_E_LOOP;
        if (d->loop_end > d->total)
            return KAC_E_LOOP;
    }

    if (!d->chunked) {
        uint64_t expect = kacdec_slice_bytes(d->total, d->n, d->channels);
        if ((uint64_t)d->data_bytes != expect)
            return KAC_E_DATA;
    }

    if (d->flags & KAC_FLAG_SEEK) {
        if (d->index_stride == 0)
            return KAC_E_INDEX;
        /* Two rules, not one. CHUNKED puts every index point on a block
         * boundary; the whole-tier rule (>= N and a multiple of N) stands on
         * its own and is not replaced by this one -- applying the chunked rule
         * there would refuse legal IndexStride = 5120 files. */
        if (d->chunked) {
            if (d->index_stride % KAC_CHUNK_SAMPLES != 0u)
                return KAC_E_INDEX;
        } else if (d->index_stride % d->n != 0) {
            return KAC_E_INDEX;
        }
        if (d->index_entries != (d->total - 1u) / d->index_stride + 1u)
            return KAC_E_INDEX;
    } else if (d->index_stride != 0 || d->index_entries != 0) {
        return KAC_E_INDEX;
    }

    /* Offsets are accumulated as plain numbers and every pointer into the image
     * is formed only after the length check below: a file may declare a
     * DataBytes that reaches far past the end of buf, and forming a pointer
     * outside the object is undefined behaviour even when it is never read. */
    off += (d->flags & KAC_FLAG_INIT) ? (size_t)d->channels * KAC_SNAP_SIZE : 0u;
    snap_off = off;
    off += (d->flags & KAC_FLAG_LOOP) ? (size_t)d->channels * KAC_SNAP_SIZE : 0u;
    need = (uint64_t)off + d->data_bytes;
    if (d->flags & KAC_FLAG_SEEK)
        need += (uint64_t)d->index_entries * (4u + (uint64_t)d->channels * KAC_SNAP_SIZE);
    if ((uint64_t)len < need)
        return KAC_E_SIZE;

    d->init_snap = (d->flags & KAC_FLAG_INIT) ? buf + KAC_HDR_SIZE : NULL;
    d->loop_snap = (d->flags & KAC_FLAG_LOOP) ? buf + snap_off : NULL;
    d->data      = buf + off;
    off += d->data_bytes;
    d->idx_off = (d->flags & KAC_FLAG_SEEK) ? buf + off : NULL;
    off += (size_t)d->index_entries * 4u;
    d->idx_snap = (d->flags & KAC_FLAG_SEEK) ? buf + off : NULL;

    /* --- the walk -----------------------------------------------------------
     * A file states its own length, so the chain is held to it before anything
     * decodes: the walk below is that check, and block 0 is entered through the
     * same call the walk uses. */
    if (d->chunked) {
        /* With N varying per block there is no closed form, so the check walks
         * the chunks instead -- reading each tier byte, accumulating that
         * chunk's sample count and its slice bytes, and requiring the walk to
         * land exactly on TotalSamples and DataBytes. The length check above
         * already put the whole data area inside buf, and every read below is
         * guarded by `tb < data_bytes`, so the walk cannot leave the object; a
         * truncated file is rejected. */
        uint32_t cov   = 0u; /* samples covered */
        uint32_t tb    = 0u; /* offset of the current block's tier byte */
        int      first = 1;
        while (cov < d->total) {
            uint32_t blk = d->total - cov;
            uint32_t slice_off;
            unsigned nn;
            if (blk > KAC_CHUNK_SAMPLES)
                blk = KAC_CHUNK_SAMPLES;
            if (tb >= d->data_bytes)
                return KAC_E_DATA;
            if (kacdec_tier_byte(d, (unsigned)d->data[tb], first, &nn) != KAC_OK)
                return KAC_E_TIER; /* the header's Tier is chunk 0's tier */
            first     = 0;
            slice_off = tb + 1u;
            tb        = slice_off + kacdec_slice_bytes(blk, nn, d->channels);
            if (d->flags & KAC_FLAG_LOOP) {
                if (cov <= d->loop_start && d->loop_start < cov + blk) {
                    d->loop_blk_start = cov;
                    d->loop_blk_off   = slice_off;
                }
            }
            cov += blk;
        }
        if (cov != d->total || tb != d->data_bytes)
            return KAC_E_DATA;
        kacdec_set_block(d, 0u, 1u); /* block 0's tier byte is at data[0] */
    } else {
        kacdec_enter(d, 0u, 0u, (unsigned)d->tier, d->total);
    }

    kacdec_start(d);
    d->pos     = 0;
    d->loop_en = 0;
    return KAC_OK;
}

int kacdec_open(kacdec_t *restrict d, const uint8_t *restrict buf, size_t len)
{
    int rc;
    if (!d)
        return KAC_E_ARG;
    rc = open_image(d, buf, len);
    if (rc != KAC_OK)
        kacdec_inert(d);
    return rc;
}

int32_t kacdec_read(kacdec_t *restrict d, int16_t *restrict out, uint32_t frames)
{
    if (!d || !out)
        return KAC_E_ARG;
    if (d->loop_en)
        return kacdec_decode(d, out, d->channels, frames, 1);
    if (d->pos >= d->total)
        return 0; /* clean end of stream */
    if (frames > d->total - d->pos)
        frames = d->total - d->pos; /* clamp: the caller reads in fixed blocks */
    return kacdec_decode(d, out, d->channels, frames, 0);
}

int kacdec_seek(kacdec_t *restrict d, uint32_t sample)
{
    uint32_t k, sl, expect;

    if (!d)
        return KAC_E_ARG;
    if (!(d->flags & KAC_FLAG_SEEK) || d->index_stride == 0)
        return KAC_E_INDEX;
    if (sample >= d->total)
        return KAC_E_RANGE;
    k = sample / d->index_stride;
    if (k >= d->index_entries)
        return KAC_E_INDEX;
    if (d->chunked) {
        /* The offset table is necessary here, not redundant: the byte offset
         * depends on the tier sequence, so there is no closed form to
         * cross-check the entry against. What can still be checked is that the
         * entry points at a real block: it must carry a tier byte and the
         * block's slices must fit inside the data area. */
        uint32_t ent = kac_get32(d->idx_off + (size_t)k * 4u);
        uint32_t pos = k * d->index_stride;
        uint32_t len, nn;
        if (ent < 1u || ent >= d->data_bytes || pos % KAC_CHUNK_SAMPLES != 0u)
            return KAC_E_INDEX;
        /* The entry's byte is a chunk's tier byte, judged the same way. It
         * cannot fail today -- opening the image walked the whole chain through
         * this same check -- so what this is here for is the guard: it is what
         * keeps kac_tiers indexed inside its bounds, and kacdec_set_block below
         * takes the byte on trust. */
        if (kacdec_tier_byte(d, (unsigned)d->data[ent - 1u], 0, &nn) != KAC_OK)
            return KAC_E_INDEX;
        len = d->total - pos;
        if (len > KAC_CHUNK_SAMPLES)
            len = KAC_CHUNK_SAMPLES;
        if ((uint64_t)ent + (uint64_t)kacdec_slice_bytes(len, nn, d->channels) >
            (uint64_t)d->data_bytes)
            return KAC_E_INDEX;
        kacdec_set_block(d, pos, ent);
    } else {
        sl     = (k * d->index_stride) / d->n;
        expect = sl * KAC_SLICE_SIZE * d->channels;
        if (kac_get32(d->idx_off + (size_t)k * 4u) != expect)
            return KAC_E_INDEX;
    }
    kac_snap_load(&d->ch[0], d->idx_snap + (size_t)k * d->channels * KAC_SNAP_SIZE);
    if (d->channels == 2)
        kac_snap_load(&d->ch[1], d->idx_snap + ((size_t)k * d->channels + 1u) * KAC_SNAP_SIZE);
    d->pos = k * d->index_stride;
    if (sample > d->pos) {
        int32_t got = kacdec_decode(d, NULL, 0u, sample - d->pos, 0);
        if (got < 0)
            return (int)got;
    }
    return KAC_OK;
}
