/* kac_dec.c -- the KAC decoder. Files and streams share one engine; the
 * entry points only supply a view. kacdec_open() adds file-length
 * validation. The sample loop calls, divides and branches only at its back
 * edge. */
#include "kac.h"

/* Bytes of slice data for `samples` at slice length n. Slices are 8 bytes at
 * every tier, so this is the whole byte extent used by every walk and bound. */
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

/* Point the decoder at one block. Only the tier follows from the header; later
 * blocks carry it in their own byte, already validated by kacdec_tier_byte. */
static void kacdec_set_block(kacdec_t *restrict d, uint32_t start, uint32_t slice_off)
{
    uint32_t left = d->total - start;
    uint32_t blk  = kac_block_for_rate(d->sample_rate);
    unsigned b    = (unsigned)d->data[slice_off - 1u];

    kacdec_use_tier(d, b & KAC_TIER_MASK);
    d->ms        = (b & KAC_BLOCK_MS) ? 1 : 0;
    d->blk_start = start;
    d->blk_len   = (left > blk) ? blk : left;
    d->blk_off   = slice_off;
}

/* Leave a rejected handle inert, so a failed open cannot be half-used. */
static void kacdec_inert(kacdec_t *restrict d)
{
    d->stream    = 0;
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

/* Start from INIT_STATE, or the normative default; weights are saturated. */
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
    /* T7 leaves four non-payload bits. Reject any set bit, like a reserved
     * bit. Accumulate instead of testing here so the loop stays branch-free. */
    uint64_t pad = 0u;
    uint64_t padmask;

    if (!d)
        return KAC_E_ARG;
    if (!out) {
        out    = sink;
        stride = 0u;
    }
    if (loop_en && !(d->flags & KAC_FLAG_LOOP))
        return KAC_E_NOLOOP;

    /* Slice invariant loads hoisted out of the sample loop. A block boundary
     * reloads them, which is outside that loop. */
    w     = d->w;
    shift = d->shift;
    nch   = (unsigned)d->channels;
    n     = d->n;
    dqt   = d->dqt;
    base  = d->data + d->blk_off;

    slice   = (d->pos - d->blk_start) / n;
    off     = (d->pos - d->blk_start) % n;
    padmask = (((uint64_t)1u) << (60u - n * w)) - 1u;

    while (done < cnt) {
        uint32_t stop = loop_en ? d->loop_end : d->total;

        if (d->pos >= stop) {
            if (!loop_en)
                return pad ? KAC_E_DATA : (int32_t)done; /* defensive: deliver what exists */
            kac_snap_load(&d->ch[0], d->loop_snap);
            if (nch == 2)
                kac_snap_load(&d->ch[1], d->loop_snap + KAC_SNAP_SIZE);
            d->pos = d->loop_start;
            kacdec_set_block(d, d->loop_blk_start, d->loop_blk_off);
            stop   = d->loop_end;
            reload = 1;
        } else if (d->pos >= d->blk_start + d->blk_len) {
            /* Next block starts after this block's slices; its tier byte is
             * immediately before them. */
            uint32_t next = d->blk_off + kacdec_slice_bytes(d->blk_len, n, nch) + 1u;
            if (next >= d->data_bytes)
                return KAC_E_DATA;
            kacdec_set_block(d, d->blk_start + d->blk_len, next);
            reload = 1;
        }
        if (reload) {
            /* Reload the slice invariants after a boundary; keep this list and
             * the prologue's in sync. */
            reload  = 0;
            w       = d->w;
            shift   = d->shift;
            n       = d->n;
            dqt     = d->dqt;
            base    = d->data + d->blk_off;
            slice   = (d->pos - d->blk_start) / n;
            off     = (d->pos - d->blk_start) % n;
            padmask = (((uint64_t)1u) << (60u - n * w)) - 1u;
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

            /* T7 is the only tier whose slice has bits outside the payload; the
             * mask is zero for the rest, so this costs one AND and one OR. */
            pad |= v & padmask;
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
            if (d->ms && c + 1u == nch) {
                /* Merge once per segment, after both channels are written. The
                 * clamp is needed; a segment never crosses a block. */
                uint32_t j;
                for (j = 0; j < seg; j++) {
                    /* the two channels of one sample sit next to each other;
                     * the samples of one channel are `stride` apart */
                    int16_t *restrict p = out + (size_t)(done + j) * stride;
                    int32_t m = p[0], s = p[1];
                    p[0] = (int16_t)kac_clamp16(m + s);
                    p[1] = (int16_t)kac_clamp16(m - s);
                }
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
    return pad ? KAC_E_DATA : (int32_t)done;
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

/* Whether a tier number is one the format admits (Section 5.1). The header's
 * Tier field and every chunk's tier byte are both held to this. */
static int kacdec_tier_ok(unsigned tier)
{
    return tier >= (unsigned)KAC_TIER_MIN && tier <= (unsigned)KAC_TIER_MAX;
}

/* Whether the file may carry the coded form at all: two channels to make a pair
 * of, and a file that declares it (Section 3.1). The header's Tier field and
 * every chunk's tier byte share this predicate, so a byte that says coded in a
 * file that declares nothing, or with one channel, is refused. */
static int kacdec_form_ok(const kacdec_t *restrict d)
{
    return d->channels == 2 && (d->flags & KAC_FLAG_MS) != 0u;
}

/* Validate a chunk's tier byte and return its slice length. Callers decide how
 * to report failure. `is_chunk0` also requires agreement with the header. */
static int kacdec_tier_byte(const kacdec_t *restrict d, unsigned b, int is_chunk0,
                            unsigned *restrict n)
{
    if ((b & KAC_TIER_SPARE) != 0u || !kacdec_tier_ok(b & KAC_TIER_MASK))
        return KAC_E_TIER;
    if ((b & KAC_BLOCK_MS) != 0u && !kacdec_form_ok(d))
        return KAC_E_TIER;
    /* Chunk 0 and the header state the same thing twice -- the tier and the
     * channel form -- and the two halves are judged together, so a byte that
     * agrees on the tier but not on the form is still refused. */
    if (is_chunk0 &&
        ((b & KAC_TIER_MASK) != (unsigned)d->tier || ((b & KAC_BLOCK_MS) ? 1 : 0) != d->ms))
        return KAC_E_TIER;
    *n = kac_tiers[(b & KAC_TIER_MASK) - 1u].n;
    return KAC_OK;
}

/* Decode the common header fields. Length and loop validation belongs to the
 * caller because file and stream rules differ. */
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
    d->tier          = (unsigned)buf[7] & KAC_TIER_MASK;
    d->ms            = (buf[7] & KAC_BLOCK_MS) ? 1 : 0;
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
    if (d->channels < 1 || d->channels > KAC_MAX_CHANNELS)
        return KAC_E_CHANNEL;
    if ((buf[7] & KAC_TIER_SPARE) != 0u || !kacdec_tier_ok(d->tier))
        return KAC_E_TIER;
    /* The header's Tier field states chunk 0's form, and chunk 0's own byte is
     * held to the same predicate when the walk reaches it. */
    if (d->ms && !kacdec_form_ok(d))
        return KAC_E_FLAGS;

    ti       = (unsigned)d->tier - 1u;
    d->w     = kac_tiers[ti].w;
    d->n     = kac_tiers[ti].n;
    d->shift = 64u - d->w;
    d->dqt   = kac_pick_dqt(d->tier);
    *off     = KAC_HDR_SIZE;
    return KAC_OK;
}

/* Validate and open a complete file image. The block walk checks that the tier
 * bytes and slice lengths land exactly on TotalSamples and DataBytes. */
static int open_file(kacdec_t *restrict d, const uint8_t *restrict buf, size_t len)
{
    uint64_t need;
    size_t   off, snap_off;
    int      rc;

    d->stream = 0; /* a handle reused as a file must not still claim a feed */
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

    if (d->flags & KAC_FLAG_SEEK) {
        if (d->index_stride == 0)
            return KAC_E_INDEX;
        /* KAC_SPEC.md Section 3.1, the IndexStride row: an index point is a block
         * boundary, so the stride is a multiple of the block length. */
        if (d->index_stride % kac_block_for_rate(d->sample_rate) != 0u)
            return KAC_E_INDEX;
        if (d->index_entries != (d->total - 1u) / d->index_stride + 1u)
            return KAC_E_INDEX;
    } else if (d->index_stride != 0 || d->index_entries != 0) {
        return KAC_E_INDEX;
    }

    /* Compute offsets as numbers and form pointers only after the length check;
     * forming a pointer past the buffer is undefined even if never read. */
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

    /* Walk the block chain: the tier byte fixes each block's length. The walk
     * must land exactly on TotalSamples and DataBytes; this also rejects data
     * with no tier bytes, unless coincidental bytes make both counts agree. */
    {
        uint32_t cov   = 0u; /* samples covered */
        uint32_t tb    = 0u; /* offset of the current block's tier byte */
        int      first = 1;
        while (cov < d->total) {
            uint32_t blk = d->total - cov;
            uint32_t slice_off;
            unsigned nn;
            if (blk > kac_block_for_rate(d->sample_rate))
                blk = kac_block_for_rate(d->sample_rate);
            if (tb >= d->data_bytes)
                return KAC_E_DATA;
            if (kacdec_tier_byte(d, (unsigned)d->data[tb], first, &nn) != KAC_OK)
                return KAC_E_TIER; /* Section 3.1: Tier is block 0's tier */
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
        if (tb != d->data_bytes)
            return KAC_E_DATA;
        kacdec_set_block(d, 0u, 1u); /* block 0's tier byte is at data[0] */
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
    rc = open_file(d, buf, len);
    if (rc != KAC_OK)
        kacdec_inert(d);
    return rc;
}

/* Validate and open a stream preamble. */
static int open_stream(kacdec_t *restrict d, const uint8_t *restrict head, size_t len)
{
    size_t off;
    int    rc;

    rc = kacdec_head(d, head, len, &off);
    if (rc != KAC_OK)
        return rc;
    if (d->flags & KAC_FLAG_SEEK)
        return KAC_E_FLAGS; /* the index trails the data a stream never reaches */
    if (d->flags & KAC_FLAG_LOOP)
        return KAC_E_LOOP; /* wrapping needs a block the decoder has given back */
    /* A stream states its chunk's sample count in IndexStride and carries no
     * index entries: that count is what lets a receiver frame the bytes. */
    if (d->index_stride == 0u || d->index_stride > KAC_CHUNK_SAMPLES)
        return KAC_E_INDEX;
    if (d->index_entries != 0u)
        return KAC_E_INDEX;
    if ((d->flags & KAC_FLAG_INIT) != 0u &&
        (uint64_t)len < (uint64_t)off + (uint64_t)d->channels * KAC_SNAP_SIZE)
        return KAC_E_SIZE;

    /* Drop file length fields; IndexStride was validated as the chunk length.
     * TotalSamples becomes the running fed-sample count. */
    d->total         = 0u;
    d->data_bytes    = 0u;
    d->index_stride  = 0u;
    d->index_entries = 0u;
    d->loop_snap     = NULL;
    d->idx_off       = NULL;
    d->idx_snap      = NULL;
    d->init_snap     = ((d->flags & KAC_FLAG_INIT) != 0u) ? head + off : NULL;
    d->data          = NULL; /* the first feed supplies it */
    d->pos           = 0u;
    d->loop_en       = 0;
    d->stream        = 1;

    kacdec_start(d);
    /* The preamble is consumed here; INIT_STATE was loaded into the handle. */
    d->init_snap = NULL;
    d->blk_start = 0u;
    d->blk_len   = 0u;
    d->blk_off   = 0u;
    return KAC_OK;
}

int kacdec_stream_open(kacdec_t *restrict d, const uint8_t *restrict head, size_t len)
{
    int rc;
    if (!d)
        return KAC_E_ARG;
    rc = open_stream(d, head, len);
    if (rc != KAC_OK)
        kacdec_inert(d);
    return rc;
}

int kacdec_stream_feed(kacdec_t *restrict d, const uint8_t *restrict blk, size_t len)
{
    uint32_t start, samples, nbytes, limit;
    unsigned nch, n, nsl, slbytes;

    if (!d || !blk)
        return KAC_E_ARG;
    if (!d->stream)
        return KAC_E_ARG; /* a file is handed over whole, at open */
    if (d->pos != d->total)
        return KAC_E_ARG; /* the previous feed is not decoded yet */
    nch     = (unsigned)d->channels;
    slbytes = KAC_SLICE_SIZE * nch; /* one slice, every tier, every channel count */

    /* Bound before computing slice count so 32-bit arithmetic is safe. */
    limit = 1u + (KAC_CHUNK_SAMPLES / kac_narrowest_n()) * slbytes;
    if (len == 0u || len > (size_t)limit)
        return KAC_E_SIZE;
    nbytes = (uint32_t)len;
    start  = d->total;

    /* Section 3.1 holds here too: the header's Tier is block 0's tier. */
    if (kacdec_tier_byte(d, (unsigned)blk[0], start == 0u, &n) != KAC_OK)
        return KAC_E_TIER;
    if ((nbytes - 1u) % slbytes != 0u)
        return KAC_E_SIZE;
    nsl = (nbytes - 1u) / slbytes;
    /* Every tier's n divides KAC_CHUNK_SAMPLES, so the bound on the slice count
     * is also the bound on the block length. */
    if (nsl == 0u || nsl > KAC_CHUNK_SAMPLES / n)
        return KAC_E_SIZE;
    samples = nsl * n;
    if (samples > KAC_MAX_TOTAL - start)
        return KAC_E_TOTAL;
    d->data  = blk;
    d->total = start + samples;
    /* The same entry the file path takes into block 0: the tier byte at
     * slice_off - 1 is the one this feed was validated against. */
    kacdec_set_block(d, start, 1u);
    /* The block guard cannot fire here: a feed ends exactly at TotalSamples. */
    return KAC_OK;
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
    uint32_t k, ent, pos, len, nn;

    if (!d)
        return KAC_E_ARG;
    if (!(d->flags & KAC_FLAG_SEEK) || d->index_stride == 0)
        return KAC_E_INDEX;
    if (sample >= d->total)
        return KAC_E_RANGE;
    k = sample / d->index_stride;
    if (k >= d->index_entries)
        return KAC_E_INDEX;
    /* Offsets cannot be derived from sample position (the tier sequence decides
     * bytes), so validate that the entry points at a real block of matching size. */
    ent = kac_get32(d->idx_off + (size_t)k * 4u);
    pos = k * d->index_stride;
    if (ent < 1u || ent >= d->data_bytes)
        return KAC_E_INDEX;
    /* The entry's byte is a tier byte; this also keeps kac_tiers indexed inside
     * its bounds, since kacdec_set_block trusts the validated tier. */
    if (kacdec_tier_byte(d, (unsigned)d->data[ent - 1u], 0, &nn) != KAC_OK)
        return KAC_E_INDEX;
    len = d->total - pos;
    if (len > KAC_CHUNK_SAMPLES)
        len = KAC_CHUNK_SAMPLES;
    if ((uint64_t)ent + (uint64_t)kacdec_slice_bytes(len, nn, d->channels) >
        (uint64_t)d->data_bytes)
        return KAC_E_INDEX;
    kacdec_set_block(d, pos, ent);

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
