/* config.json schema/validation for csa2_kv_mode/csa2_kv_source - the
   engine-side half of DeepSeek-V4.1 report Sec2.3.1's Full/Reindex/
   Reuse cross-layer KV reuse axis (model.h's own csa2_mode[]/
   csa2_source[] comment on LZModelConfig has the mechanism). The
   training side already landed the three-mode forward pass in a
   DIFFERENT repository (E:\LLM\llama98-zh, KunMoeConfig.csa2_kv_mode/
   csa2_kv_source, engine/csa2.py) - that tree shares no git history
   with this one, so its validation does not arrive here by any
   mechanism except being written again, which is what this probe
   checks against the real lz_load_config (src/model.c), not a
   transcription of it.

   forward_csa2.c does not read either array yet (a later, separate
   task's cache_idx aliasing and mode dispatch) - this is schema and
   validation only, driven through the ACTUAL JSON parser on hand-
   written config.json files, because the point is testing the parser,
   not the struct's field defaults.

   Fixture, shared by every case below: 6 layers, full_attention at
   0/2/4/5 and linear elsewhere, csa2_layers=[0,2,4,5]. Layer 0 has no
   strictly-earlier CSA2 layer at all (its own default-mode case), 2's
   only earlier CSA2 layer is 0, 4's are {0,2}, 5's are {0,2,4} -
   enough room to build a REINDEX/REUSE chain and every refusal shape
   below without a second fixture.

   Built and run by build/csa2_modes_config_gate.sh against the real
   src/model.c, and against a mutant copy with the REUSE-source-cannot-
   itself-be-REUSE cross-check deleted, to confirm the fixture is
   actually sensitive to the defect it names (case 6 only must flip).

   Exit: 0 all cases pass, 1 a case failed (printed). */
#include <stdio.h>
#include <string.h>
#include "model.h"

static int g_fail = 0;

static void ok(int cond, const char *what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) g_fail = 1;
}

static const char *TMP_PATH = "csa2_modes_config_probe.tmp.json";

/* Writes a config.json with the shared fixture plus the given
   csa2_kv_mode/csa2_kv_source object bodies verbatim (e.g. "{}" or
   "{\"2\":\"reindex\"}"), and loads it through the real lz_load_config. */
static int load_with(const char *mode_obj, const char *source_obj,
                     LZModelConfig *c, char *errbuf, int errlen) {
    FILE *fp = fopen(TMP_PATH, "wb");
    int rc;
    if (!fp) { printf("FAIL - could not write %s\n", TMP_PATH); return -1; }
    fprintf(fp,
        "{"
        "\"vocab_size\":32,"
        "\"hidden_size\":64,"
        "\"num_hidden_layers\":6,"
        "\"num_attention_heads\":2,"
        "\"num_key_value_heads\":2,"
        "\"head_dim\":32,"
        "\"intermediate_size\":64,"
        "\"attn_output_gate\":true,"
        "\"layer_types\":[\"full_attention\",\"linear_attention\","
                         "\"full_attention\",\"linear_attention\","
                         "\"full_attention\",\"full_attention\"],"
        "\"csa2_layers\":[0,2,4,5],"
        "\"csa2_kv_mode\":%s,"
        "\"csa2_kv_source\":%s"
        "}",
        mode_obj, source_obj);
    fclose(fp);
    memset(c, 0, sizeof(*c));
    errbuf[0] = '\0';
    rc = lz_load_config(c, TMP_PATH, errbuf, errlen);
    remove(TMP_PATH);
    return rc;
}

/* Same fixture as load_with, plus an explicit mla_q_lora_rank - I3
   (report Sec2.3.1 mode-dispatch task, fifth review round): S4+CSA2
   composition is refused explicitly (model.h's own csa2_w_dq_idx
   comment has the mechanism - forward_csa2.c's step 1 always reads
   L->q_proj directly, never S4's low-rank q_proj_down/up pair), not
   left to an accidental missing-tensor error. A separate function
   rather than an extra load_with() parameter every one of its dozen
   existing call sites would otherwise need updating for. */
static int load_with_mla(int mla_rank, const char *mode_obj,
                         const char *source_obj, LZModelConfig *c,
                         char *errbuf, int errlen) {
    FILE *fp = fopen(TMP_PATH, "wb");
    int rc;
    if (!fp) { printf("FAIL - could not write %s\n", TMP_PATH); return -1; }
    fprintf(fp,
        "{"
        "\"vocab_size\":32,"
        "\"hidden_size\":64,"
        "\"num_hidden_layers\":6,"
        "\"num_attention_heads\":2,"
        "\"num_key_value_heads\":2,"
        "\"head_dim\":32,"
        "\"intermediate_size\":64,"
        "\"attn_output_gate\":true,"
        "\"mla_q_lora_rank\":%d,"
        "\"layer_types\":[\"full_attention\",\"linear_attention\","
                         "\"full_attention\",\"linear_attention\","
                         "\"full_attention\",\"full_attention\"],"
        "\"csa2_layers\":[0,2,4,5],"
        "\"csa2_kv_mode\":%s,"
        "\"csa2_kv_source\":%s"
        "}",
        mla_rank, mode_obj, source_obj);
    fclose(fp);
    memset(c, 0, sizeof(*c));
    errbuf[0] = '\0';
    rc = lz_load_config(c, TMP_PATH, errbuf, errlen);
    remove(TMP_PATH);
    return rc;
}

int main(void) {
    LZModelConfig c;
    char errbuf[256];
    int rc;

    /* Case 1: default - a csa2_on layer named in neither object resolves
       to FULL with no source. Checked on two different csa2_on layers
       (0 and 2) so "always FULL/-1 regardless of layer" is not a
       single-sample coincidence. */
    rc = load_with("{}", "{}", &c, errbuf, sizeof errbuf);
    ok(rc == 0, "case 1: default config loads");
    if (rc == 0) {
        ok(c.csa2_mode[0] == LZ_CSA2_FULL && c.csa2_source[0] == -1,
           "case 1: layer 0 (csa2_on, unnamed) is FULL, source -1");
        ok(c.csa2_mode[2] == LZ_CSA2_FULL && c.csa2_source[2] == -1,
           "case 1: layer 2 (csa2_on, unnamed) is FULL, source -1");
    }

    /* Case 2: explicit REINDEX/REUSE with a valid strictly-earlier
       source each resolve correctly, and an unnamed layer (5) is
       unaffected by its csa2_on siblings being named. */
    rc = load_with("{\"2\":\"reindex\",\"4\":\"reuse\"}",
                   "{\"2\":0,\"4\":2}", &c, errbuf, sizeof errbuf);
    ok(rc == 0, "case 2: explicit reindex/reuse chain loads");
    if (rc == 0) {
        ok(c.csa2_mode[2] == LZ_CSA2_REINDEX && c.csa2_source[2] == 0,
           "case 2: layer 2 is reindex, source 0");
        ok(c.csa2_mode[4] == LZ_CSA2_REUSE && c.csa2_source[4] == 2,
           "case 2: layer 4 is reuse, source 2 (source is reindex, not reuse)");
        ok(c.csa2_mode[5] == LZ_CSA2_FULL && c.csa2_source[5] == -1,
           "case 2: layer 5 (csa2_on, unnamed) stays FULL, source -1");
    }

    /* Case 3: unrecognized mode string. */
    rc = load_with("{\"2\":\"bogus\"}", "{}", &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 3: unrecognized csa2_kv_mode string is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_kv_mode") != NULL &&
       strstr(errbuf, "bogus") != NULL,
       "case 3: refusal names csa2_kv_mode and the bad string");

    /* Case 4: source names a layer that is not csa2_on (layer 1 is
       linear_attention, never in csa2_layers). */
    rc = load_with("{\"2\":\"reindex\"}", "{\"2\":1}",
                   &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 4: source not in csa2_on is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_kv_source") != NULL,
       "case 4: refusal names csa2_kv_source");

    /* Case 5: source is a later (or equal) layer, not strictly earlier -
       5 is csa2_on, so this isolates "not earlier" from case 4's
       "not csa2_on at all". */
    rc = load_with("{\"4\":\"reindex\"}", "{\"4\":5}",
                   &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 5: source not strictly earlier is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_kv_source") != NULL,
       "case 5: refusal names csa2_kv_source");

    /* Case 6: a REUSE layer (5) whose source (4) is ITSELF reuse mode -
       forbidden even though 4 < 5 and 4 is csa2_on, because a REUSE
       chained off a REUSE has no single layer left holding real K/V. */
    rc = load_with("{\"2\":\"reindex\",\"4\":\"reuse\",\"5\":\"reuse\"}",
                   "{\"2\":0,\"4\":2,\"5\":4}", &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 6: reuse-of-reuse is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_kv_source") != NULL,
       "case 6: refusal names csa2_kv_source");

    /* Case 7a: an EXPLICIT full-mode layer with a source entry. */
    rc = load_with("{\"2\":\"full\"}", "{\"2\":0}", &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 7a: explicit full mode with a source entry is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_kv_source") != NULL,
       "case 7a: refusal names csa2_kv_source");

    /* Case 7b: same, but the layer's mode was never named at all (the
       DEFAULT full, not an explicit one) - must be refused identically. */
    rc = load_with("{}", "{\"2\":0}", &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 7b: default (unnamed) full mode with a source entry is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_kv_source") != NULL,
       "case 7b: refusal names csa2_kv_source");

    /* Case 8: a non-full mode with no source entry at all. */
    rc = load_with("{\"2\":\"reindex\"}", "{}", &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 8: non-full mode with no source entry is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_kv_source") != NULL,
       "case 8: refusal names csa2_kv_source");

    /* Case 9: csa2_kv_source's VALUE is not a number at all (a typo'd
       string). jnode_new zeroes every new node, so a string node's own
       ->num reads back 0.0 exactly like a legitimate "source is layer
       0" would - without a type check first this resolves to
       csa2_source[2]=0 and loads clean, silently accepting a config a
       human mistyped. */
    rc = load_with("{\"2\":\"reindex\"}", "{\"2\":\"typo\"}",
                   &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 9: non-numeric csa2_kv_source value is refused");
    ok(rc != 0 && strstr(errbuf, "csa2_kv_source") != NULL,
       "case 9: refusal names csa2_kv_source");

    /* Case 10: malformed OBJECT KEYS - leading whitespace, a leading
       sign, and a number too large for a layer index - must all be
       refused the same as any other out-of-range index, not silently
       parsed by strtol's own leading-whitespace/sign tolerance or
       truncated into a small in-range int by an overflowing cast. */
    rc = load_with("{\" 2\":\"reindex\"}", "{}", &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 10a: csa2_kv_mode key with leading whitespace is refused");
    rc = load_with("{\"+2\":\"reindex\"}", "{}", &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 10b: csa2_kv_mode key with a leading sign is refused");
    rc = load_with("{\"99999999999999999999\":\"reindex\"}", "{}",
                   &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 10c: csa2_kv_mode key too large for any layer index is refused");

    /* Case 11 (I3): mla_q_lora_rank > 0 together with ANY csa2_on layer
       is refused, even at csa2_kv_mode's default (FULL, unnamed) - the
       composition this engine does not support is layer-level, not
       mode-level (every mode's step 1 reads L->q_proj the same way). */
    rc = load_with_mla(16, "{}", "{}", &c, errbuf, sizeof errbuf);
    ok(rc != 0, "case 11: mla_q_lora_rank>0 with a csa2_on layer (default "
                "FULL mode) is refused");
    ok(rc != 0 && strstr(errbuf, "csa2") != NULL,
       "case 11: refusal names the csa2 layer");
    /* Control: the same mla_q_lora_rank with NO csa2_on layer at all
       must load clean - S4 alone, without CSA2, is unrelated to this
       refusal. csa2_layers must be empty too, or every layer it names
       becomes csa2_on regardless of csa2_kv_mode/csa2_kv_source. */
    {
        FILE *fp = fopen(TMP_PATH, "wb");
        if (!fp) {
            printf("FAIL - could not write %s\n", TMP_PATH);
        } else {
            fprintf(fp,
                "{"
                "\"vocab_size\":32,\"hidden_size\":64,"
                "\"num_hidden_layers\":6,\"num_attention_heads\":2,"
                "\"num_key_value_heads\":2,\"head_dim\":32,"
                "\"intermediate_size\":64,\"attn_output_gate\":true,"
                "\"mla_q_lora_rank\":16,"
                "\"layer_types\":[\"full_attention\",\"linear_attention\","
                                 "\"full_attention\",\"linear_attention\","
                                 "\"full_attention\",\"full_attention\"]"
                "}");
            fclose(fp);
            memset(&c, 0, sizeof(c));
            errbuf[0] = '\0';
            rc = lz_load_config(&c, TMP_PATH, errbuf, sizeof errbuf);
            remove(TMP_PATH);
            ok(rc == 0, "case 11: mla_q_lora_rank>0 with NO csa2_on layer "
                        "at all loads clean (S4 alone is unrelated)");
        }
    }

    if (g_fail) {
        printf("\ncsa2_modes_config_probe: FAIL\n");
        return 1;
    }
    printf("\ncsa2_modes_config_probe: PASS\n");
    return 0;
}
