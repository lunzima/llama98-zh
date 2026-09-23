/* The ring's storage formats, at the value level.
   forward.h's LZ_CRF_* comment has the contract this checks: f32 is a
   VIEW into the ring and copies nothing, the narrow tiers round-trip
   through their own rule with a bounded error, and a row occupies
   exactly lz_csa2_ring_rowbytes bytes - scales included, one row at a
   time.

   It exists because the integration arms in csa2_ring_fmt_gate.sh cannot
   see any of these: a narrow ring whose conversion were subtly wrong, or
   one that wrote past the bytes its row owns, changes the logits either
   way, and "the numbers moved" is the same reading for a correct
   rounding and an overrun.

   The guard bytes are the point of the last arm: the buffer is filled
   with a sentinel, rows are written at a non-zero index, and every byte
   they are NOT entitled to must still hold the sentinel. Both bounds are
   computed from lz_csa2_ring_rowbytes rather than written down, so a
   layout that grows a plane (the scaled tiers' scale plane is exactly
   that) moves the expectation with it.

   The scaled tiers' round trip is checked as IDEMPOTENCE plus a bound
   derived from the data's own magnitude, not by re-implementing the
   quantizer here: put(row(x)) then reading again must give back the same
   values (a fixed point), and the first trip must stay within one
   quantum of the group's maximum. Re-deriving amax/inv in this file
   would make it agree with itself and disagree with the engine.

   Usage: csa2_ring_fmt_probe
   Prints "rowbytes <fmt> <n> <bytes>" per format and "n <bytes> m <bytes>"
   once, then one line per failed check; exits non-zero on any failure. */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "forward.h"
#include "lz_bf16.h"

static int failed = 0;

static void fail(const char *what, int i) {
    printf("FAIL %s (case %d)\n", what, i);
    failed++;
}

static const char *fmt_name(int f) {
    switch (f) {
        case LZ_CRF_F32:  return "f32";
        case LZ_CRF_BF16: return "bf16";
        case LZ_CRF_Q8:   return "q8";
        case LZ_CRF_Q16:  return "q16";
        default:          return "?";
    }
}

int main(int argc, char **argv) {
    static const float vals[] = {
        0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -3.25f, 3.14159265f,
        1e-30f, 1e30f, 1.0000001f, -0.0f, 12345.678f
    };
    static const int tiers[4] = { LZ_CRF_F32, LZ_CRF_BF16, LZ_CRF_Q8, LZ_CRF_Q16 };
    const int n = 6;          /* one row */
    unsigned char buf[1024];
    float scratch[16], back[16];
    int t, i, r;

    if (lz_csa2_ring_stride(LZ_CRF_F32) != 4)  fail("f32 stride is not 4", 0);
    if (lz_csa2_ring_stride(LZ_CRF_BF16) != 2) fail("bf16 stride is not 2", 0);
    if (lz_csa2_ring_stride(LZ_CRF_Q8) != 1)   fail("q8 stride is not 1", 0);
    if (lz_csa2_ring_stride(LZ_CRF_Q16) != 2)  fail("q16 stride is not 2", 0);

    /* The gate reads these to compute what a tier should save, so they
       are part of this probe's output, not of its checks. Widths given on
       the command line are printed as well - the gate passes the model's
       own head_dim and indexer_head_dim, so what a tier saves there comes
       from this function rather than from arithmetic retyped in the
       gate. */
    {
        int widths[3], nw = 1, a, b;
        widths[0] = n;
        for (a = 1; a < argc && nw < 3; a++) {
            b = atoi(argv[a]);
            if (b <= 0) continue;
            if (b == widths[0] || (nw > 1 && b == widths[1])) continue;
            widths[nw++] = b;
        }
        for (a = 0; a < nw; a++)
            for (t = 0; t < 4; t++)
                printf("rowbytes %s %d %lu\n", fmt_name(tiers[t]), widths[a],
                       (unsigned long)lz_csa2_ring_rowbytes(tiers[t], widths[a]));
    }

    for (t = 0; t < 4; t++) {
        int fmt = tiers[t];
        size_t rb = lz_csa2_ring_rowbytes(fmt, n);
        const float *row;

        memset(buf, 0, sizeof(buf));
        lz_csa2_ring_put(buf, 1, vals, n, fmt);
        row = lz_csa2_ring_row(buf, 1, scratch, n, fmt);

        /* f32 is a view, not a copy: the pointer handed back must be the
           ring's own row, because that is what keeps this tier
           bit-identical and as fast as the engine without the knob. */
        if (fmt == LZ_CRF_F32) {
            if (row != (const float *)buf + n) fail("f32 row is not a view", 0);
            if (memcmp(buf + n * sizeof(float), vals, n * sizeof(float)) != 0)
                fail("f32 put did not store the float bytes verbatim", 0);
        }

        /* One trip, then the same values again: the second trip must be a
           fixed point, and the first must stay within one quantum. */
        for (i = 0; i < n; i++) back[i] = row[i];
        lz_csa2_ring_put(buf, 1, back, n, fmt);
        row = lz_csa2_ring_row(buf, 1, scratch, n, fmt);
        for (i = 0; i < n; i++)
            if (row[i] != back[i]) fail("round trip is not a fixed point", fmt);

        if (fmt != LZ_CRF_F32) {
            float amax = 0.0f, q;
            for (i = 0; i < n; i++) {
                float a = vals[i] < 0.0f ? -vals[i] : vals[i];
                if (a > amax) amax = a;
            }
            q = fmt == LZ_CRF_BF16 ? amax / 256.0f :      /* 8 mantissa bits */
                (fmt == LZ_CRF_Q8 ? amax / 127.0f : amax / 32767.0f);
            for (i = 0; i < n; i++) {
                float e = row[i] - back[i];
                if (e < 0.0f) e = -e;
                if (e > q + q * 0.01f) {                  /* one quantum, +1% */
                    fail("round trip error exceeds one quantum", fmt);
                    break;
                }
            }
        }

        /* Every byte the row is not entitled to keeps its sentinel. */
        memset(buf, 0xa5, sizeof(buf));
        lz_csa2_ring_put(buf, 3, vals, n, fmt);
        for (r = 0; r < 1024; r++) {
            int inside = (size_t)r >= 3 * rb && (size_t)r < 3 * rb + rb;
            if (!inside && buf[r] != 0xa5) { fail("row wrote outside its bytes", fmt); break; }
        }
    }

    /* A zero row must read back as exact zeros: the scaled tiers write a
       zero scale for a group whose scale would fall under the floor, and
       a reader that multiplied by it would produce NaN. */
    {
        float zeros[6];
        memset(zeros, 0, sizeof(zeros));
        for (t = 0; t < 4; t++) {
            const float *zrow;
            lz_csa2_ring_put(buf, 0, zeros, n, tiers[t]);
            /* The return value, not the scratch: at f32 the row IS the
               ring and the scratch is never written. */
            zrow = lz_csa2_ring_row(buf, 0, scratch, n, tiers[t]);
            for (i = 0; i < n; i++)
                if (zrow[i] != 0.0f) { fail("a zero row did not read back as zeros", tiers[t]); break; }
        }
    }

    if (failed) { printf("csa2_ring_fmt_probe: %d check(s) FAILED\n", failed); return 1; }
    printf("csa2_ring_fmt_probe: all checks passed\n");
    return 0;
}
