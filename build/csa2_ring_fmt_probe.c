/* The ring's storage formats, at the value level.
   forward.h's LZ_CRF_* comment has the contract this checks: f32 is a
   VIEW into the ring and copies nothing, bf16 round-trips through
   lz_f32_to_bf16/lz_bf16_to_f32 and nothing else, and the stride is what
   every offset in the engine multiplies by.

   It exists because the integration arms in csa2_ring_fmt_gate.sh cannot
   see any of these: a bf16 ring whose conversion were subtly wrong, or
   one that wrote four bytes per element into a buffer sized for two,
   changes the logits either way, and "the numbers moved" is the same
   reading for a correct rounding and an overrun.

   The guard bytes are the point of the last arm: the buffer is filled
   with a sentinel, one element is written at a non-zero index, and every
   byte the element is NOT entitled to must still hold the sentinel. A
   format whose stride is written down in two places disagrees with
   itself exactly here.

   Prints one line per failed check and exits non-zero; the gate reads
   the count, not the text. */

#include <stdio.h>
#include <string.h>
#include "forward.h"
#include "lz_bf16.h"

static int failed = 0;

static void fail(const char *what) {
    printf("FAIL %s\n", what);
    failed++;
}

int main(void) {
    static const float vals[] = {
        0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -3.25f, 3.14159265f,
        1e-30f, 1e30f, 1.0000001f, -0.0f, 12345.678f
    };
    static const int n = (int)(sizeof(vals) / sizeof(vals[0]));
    unsigned char buf[256];
    float scratch[8];
    int i, r;

    if (lz_csa2_ring_stride(LZ_CRF_F32) != 4)
        fail("f32 stride is not 4");
    if (lz_csa2_ring_stride(LZ_CRF_BF16) != 2)
        fail("bf16 stride is not 2");

    /* f32 is a view, not a copy: the pointer handed back must be the ring
       itself, because that is what keeps this tier bit-identical and as
       fast as the engine without the knob. */
    memset(buf, 0, sizeof(buf));
    lz_csa2_ring_put(buf, 0, vals, 4, LZ_CRF_F32);
    if (lz_csa2_ring_row(buf, 0, scratch, 4, LZ_CRF_F32) != (const float *)buf)
        fail("f32 row is not a view into the ring");
    if (memcmp(buf, vals, 4 * sizeof(float)) != 0)
        fail("f32 put did not store the float bytes verbatim");
    if (lz_csa2_ring_row(buf, 2, scratch, 4, LZ_CRF_F32) !=
        ((const float *)buf) + 2)
        fail("f32 row does not offset by base elements");

    /* bf16: the stored bytes are lz_f32_to_bf16's, in low-byte-first
       order, and the read value is lz_bf16_to_f32's. */
    for (i = 0; i < n; i++) {
        uint32_t f, want, got;
        uint32_t h;
        const float *row;
        memcpy(&f, &vals[i], sizeof(float));
        want = lz_bf16_to_f32(lz_f32_to_bf16(f));
        memset(buf, 0, sizeof(buf));
        lz_csa2_ring_put(buf, 1, &vals[i], 1, LZ_CRF_BF16);
        h = lz_f32_to_bf16(f);
        if (buf[2] != (unsigned char)(h & 0xffu) ||
            buf[3] != (unsigned char)((h >> 8) & 0xffu))
            fail("bf16 put byte order");
        row = lz_csa2_ring_row(buf, 1, scratch, 1, LZ_CRF_BF16);
        memcpy(&got, &row[0], sizeof(float));
        if (got != want)
            fail("bf16 round trip is not lz_bf16_to_f32(lz_f32_to_bf16(x))");
    }

    /* Every byte the element is not entitled to keeps its sentinel. */
    memset(buf, 0xa5, sizeof(buf));
    lz_csa2_ring_put(buf, 3, vals, 2, LZ_CRF_BF16);
    for (r = 0; r < (int)sizeof(buf); r++) {
        int inside = (r >= 3 * 2 && r < (3 + 2) * 2);
        if (!inside && buf[r] != 0xa5) {
            fail("bf16 put wrote outside the two bytes it owns");
            break;
        }
    }
    memset(buf, 0xa5, sizeof(buf));
    lz_csa2_ring_put(buf, 3, vals, 2, LZ_CRF_F32);
    for (r = 0; r < (int)sizeof(buf); r++) {
        int inside = (r >= 3 * 4 && r < (3 + 2) * 4);
        if (!inside && buf[r] != 0xa5) {
            fail("f32 put wrote outside the four bytes it owns");
            break;
        }
    }

    if (failed) {
        printf("csa2_ring_fmt_probe: %d check(s) FAILED\n", failed);
        return 1;
    }
    printf("csa2_ring_fmt_probe: all checks passed\n");
    return 0;
}
