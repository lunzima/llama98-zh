/* Precision regression probe for src/json.c's number parsing.
   Built and run by build/json_offset_gate.sh; see that script for how
   it drives this SAME probe against both the current tree and the
   pre-fix (git HEAD) json.c/json.h to get a real before/after control,
   rather than just asserting "runs and returns 0".

   THE BUG. LZJsonNode.num used to be `float` (strtof). safetensors
   headers store data_offsets - byte offsets into the checkpoint - as
   plain JSON integers, and lz_st_open() reads them straight off ->num
   (src/safetensors.c, no lz_json_get_int in the way). float32's
   mantissa is 24 bits: any offset at or past 2^24 (16,777,216) that
   isn't a multiple of the local ULP loses precision SILENTLY - no
   parse error, just a wrong integer. A real load (kunmoe-scale
   checkpoint) hit this at offset 536998048, which strtof rounds to
   536998016 - the tensor's start moves by 32 bytes and nothing says so.

   TIER 1 tests the parser in isolation against an oracle that shares
   no code with json.c: every test value here fits in 31 bits, so a
   plain `long` literal typed next to the JSON text is exact regardless
   of what json.c does with it - it never goes through strtof/strtod at
   all.

   TIER 2 tests the real consumer chain (lz_st_open/lz_st_find) on an
   actual on-disk safetensors-shaped file, reproducing the exact way
   this stays silent in production: tensor A's data_offsets are chosen
   32 bytes into a float32 ULP=64 region, so BOTH endpoints round the
   same direction by the same amount under the old float path - nbytes
   (end - start) survives unchanged, so the shape/dtype-size
   cross-check that would normally catch a corrupt offset cannot see
   it. Tensor B's offsets are exactly representable and set the file's
   defining max-end, so the whole-file size check also stays green.
   Under the bug, lz_st_open returns 0 (success) with tensor A's
   off_begin silently wrong by 32 bytes - not an error path, a wrong
   answer with no signal.

   Seed: none needed - every case here is a fixed constant chosen for a
   specific rounding property, not a statistical sweep. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "json.h"
#include "safetensors.h"
#include "lz_int.h"

/* ---------------------------------------------------------- TIER 1 */

/* Values are typed once here and once again (independently) in the
   JSON text below; they must never be derived from one another. */
static const char *T1_JSON =
    "[4194304, 512, -1, 16777216, 16777217, "
    "536998048, 529658016, 534940607, 600998048, "
    "601098048, 601163584]";

static const long T1_EXPECT[] = {
    4194304, 512, -1, 16777216, 16777217,
    536998048, 529658016, 534940607, 600998048,
    601098048, 601163584
};
static const char *T1_LABEL[] = {
    "engram_size (config.json-scale, unaffected)",
    "hidden_size (config.json-scale, unaffected)",
    "hadamard_o default (small, negative)",
    "2^24 exact (float32 mantissa boundary)",
    "2^24+1 (needs rounding at float32)",
    "real repro s0 (loses 32 at float32)",
    "real repro e0 (exact at float32 - not every large value is lossy)",
    "534,940,607 near kunmoe-v2's ~534MB file size (needs rounding)",
    "masked tensor A e0 (loses 32, SAME amount as s0 above)",
    "tensor B s0 (exact at float32)",
    "tensor B e0 (exact at float32, defines file size in TIER 2)"
};
#define T1_N ((int)(sizeof(T1_EXPECT) / sizeof(T1_EXPECT[0])))
/* which of the above are NOT exactly representable as float32 - used
   only to sense-check that this fixture actually exercises the bug
   class, not to assert engine behaviour. A fixture where every value
   happens to survive float32 rounding would pass on the OLD code too,
   proving nothing. */
static const int T1_LOSSY_AT_F32[T1_N] = { 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0 };

static int tier1(void) {
    LZJson j;
    char errbuf[256];
    const LZJsonNode *root, *e;
    int bad = 0, n = 0, lossy_in_fixture = 0;

    if (lz_json_parse(&j, T1_JSON, strlen(T1_JSON), errbuf, sizeof errbuf) != 0) {
        printf("TIER1 FAIL: parse error: %s\n", errbuf);
        return 1;
    }
    root = lz_json_root(&j);
    for (e = lz_json_first(&j, root); e; e = lz_json_next(&j, e), n++) {
        /* the exact idiom src/safetensors.c uses on data_offsets/shape:
           (lz_i64) cast straight off ->num, no lz_json_get_int in the way */
        long got = (long)(lz_i64)e->num;
        long want = n < T1_N ? T1_EXPECT[n] : -1;
        int match = (got == want);
        printf("TIER1 %-60s parsed=%-12ld expect=%-12ld %s\n",
               n < T1_N ? T1_LABEL[n] : "?(fixture/JSON out of sync)",
               got, want, match ? "MATCH" : "MISMATCH");
        if (!match) bad++;
        if (n < T1_N) lossy_in_fixture += T1_LOSSY_AT_F32[n];
    }
    lz_json_free(&j);
    if (n != T1_N) {
        printf("TIER1 FAIL: fixture/JSON out of sync - %d value(s) typed, %d parsed\n",
               T1_N, n);
        bad++;
    }
    if (lossy_in_fixture == 0) {
        printf("TIER1 FAIL: none of the fixture values are lossy at float32 -"
               " this probe would pass even with the bug back in place\n");
        bad++;
    }
    printf("TIER1: %d value(s), %d mismatch(es), %d known lossy-at-float32\n",
           n, bad, lossy_in_fixture);
    return bad != 0;
}

/* ---------------------------------------------------------- TIER 2 */

/* Write [u64 LE header length][header][... hole ...][one byte], all
   through ONE handle - a second fopen(path,"wb") to "resize" the file
   would truncate what the first open just wrote. The seek-past-end +
   one byte gives a file of exactly total_size without an O(n) write of
   real content; lz_st_open never reads tensor bytes, only the header
   and the length arithmetic. */
static int t2_write_file(const char *path, const char *hdr, size_t hlen,
                         lz_u64 total_size) {
    unsigned char lenbuf[8];
    lz_u64 L;
    int i;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 1;
    L = (lz_u64)hlen;
    for (i = 0; i < 8; i++) { lenbuf[i] = (unsigned char)(L & 0xFF); L >>= 8; }
    if (fwrite(lenbuf, 1, 8, fp) != 8 || fwrite(hdr, 1, hlen, fp) != hlen) {
        fclose(fp);
        return 1;
    }
    if (total_size > 0) {
        if (fseek(fp, (long)(total_size - 1), SEEK_SET) != 0 ||
            fputc(0, fp) == EOF) {
            fclose(fp);
            return 1;
        }
    }
    return fclose(fp) != 0;
}

static int tier2(const char *path) {
    char hdr[512];
    int hlen;
    lz_u64 max_end = 601163584ULL; /* tensor B's e0 - drives the file size */
    LZSafetensors st;
    char errbuf[256];
    const LZStTensor *A, *B;
    int bad = 0;

    hlen = snprintf(hdr, sizeof hdr,
        "{\"A\":{\"dtype\":\"F32\",\"shape\":[16000000],"
          "\"data_offsets\":[536998048,600998048]},"
         "\"B\":{\"dtype\":\"F32\",\"shape\":[16384],"
          "\"data_offsets\":[601098048,601163584]}}");
    if (hlen <= 0 || (size_t)hlen >= sizeof hdr) {
        printf("TIER2 FAIL: header build\n");
        return 1;
    }

    if (t2_write_file(path, hdr, (size_t)hlen, 8ULL + (lz_u64)hlen + max_end) != 0) {
        printf("TIER2 FAIL: could not write/size the file\n");
        return 1;
    }

    if (lz_st_open(&st, path, errbuf, sizeof errbuf) != 0) {
        /* A control build CAN legitimately fail to open here (e.g. if
           its rounding also corrupts tensor B's offsets and the
           whole-file size check catches it) - the gate script treats a
           non-zero probe exit as "control tripped", open failure
           included. Reported, not silently swallowed. */
        printf("TIER2 open_rc=1 err=%s\n", errbuf);
        return 1;
    }
    A = lz_st_find(&st, "A");
    B = lz_st_find(&st, "B");
    if (!A || !B) {
        printf("TIER2 FAIL: tensor not found after a successful open\n");
        lz_st_close(&st);
        return 1;
    }
    printf("TIER2 open_rc=0\n");
    printf("TIER2 A.off_begin=%lu A.nbytes=%lu (true: 536998048 / 64000000)\n",
           (unsigned long)A->off_begin, (unsigned long)A->nbytes);
    printf("TIER2 B.off_begin=%lu B.nbytes=%lu (true: 601098048 / 65536)\n",
           (unsigned long)B->off_begin, (unsigned long)B->nbytes);
    if (A->off_begin != 536998048UL) { printf("TIER2 MISMATCH: A.off_begin wrong\n"); bad++; }
    if (A->nbytes    != 64000000UL)  { printf("TIER2 MISMATCH: A.nbytes wrong\n");    bad++; }
    if (B->off_begin != 601098048UL) { printf("TIER2 MISMATCH: B.off_begin wrong\n"); bad++; }
    if (B->nbytes    != 65536UL)     { printf("TIER2 MISMATCH: B.nbytes wrong\n");    bad++; }
    lz_st_close(&st);
    remove(path);
    return bad != 0;
}

int main(int argc, char **argv) {
    int bad = 0;
    const char *tmp_path = argc > 1 ? argv[1] : "json_offset_probe.tmp.safetensors";

    bad += tier1();
    bad += tier2(tmp_path);

    printf("json_offset_probe: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad != 0;
}
