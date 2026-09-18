/* lz_kv_share_target must not claim a layer inside CED's decoder
 * region (ced_split <= li < n_layers, when CED is on) as a valid
 * attn_kv_share target. Report Sec2.2 Eq(1) is unconditional there -
 * a decoder-region full-attention layer's K/V is always the CED
 * mechanism's own projection from the encoder's exit hidden - and
 * attn_kv_share (model.h's own comment: a project-invented
 * generalization of DeepSeek-V4.1 report Sec2.3.1's "Reuse" mode, not
 * an Eq(1) rule) has no basis to override it.
 *
 * Same collision the training side found and fixed first
 * (E:\LLM\llama98-zh, engine/kunmoe_modeling.py's _wire_attn_kv_share):
 * that tree and this one share no git history, so the fix does not
 * arrive here by any mechanism except being written again.
 *
 * Fixture: 4 layers, full_attention at 0 and 3, linear elsewhere,
 * attn_kv_share=1 - the same shape the training-side gate used. Three
 * rows:
 *   1. ced off (ced_split=0, the field's own default) - the pre-
 *      existing rule is unchanged, layer 3 is a target.
 *   2. ced on, split=1 - layer 0 is the whole encoder, layer 3 is in
 *      the decoder - layer 3 must stop being a target.
 *   3. ced_split == n_layers - CED OFF by this project's own
 *      convention (model.c's lz_gr_span: `ced_split > 0 && ced_split <
 *      n_layers` is the "CED on" test everywhere else in this file),
 *      not a decoder spanning every layer - layer 3 is a target again.
 *      Without this row, "always return 0 when ced_split != 0" would
 *      also pass row 2.
 *
 * Built and run by build/kv_share_ced_boundary_gate.sh against the
 * real src/model.c (not a transcription - CLAUDE.md's own note on
 * .prof/'s wfwd.c/wconv2.c applies here too), and against a mutant
 * copy with the fix's own if-block deleted, to confirm the fixture is
 * actually sensitive to the defect it names.
 *
 * Exit: 0 all rows pass, 1 a row failed (printed).
 */
#include <stdio.h>
#include <string.h>
#include "model.h"

static int g_fail = 0;

static void ok(int cond, const char *what) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", what);
    if (!cond) g_fail = 1;
}

int main(void) {
    LZModelConfig c;
    memset(&c, 0, sizeof(c));
    c.n_layers = 4;
    c.layer_types[0] = LZ_LT_FULL;
    c.layer_types[1] = LZ_LT_LINEAR;
    c.layer_types[2] = LZ_LT_LINEAR;
    c.layer_types[3] = LZ_LT_FULL;
    c.attn_kv_share = 1;

    /* Row 1: CED off, the pre-existing rule is unchanged. */
    c.ced_split = 0;
    ok(lz_kv_share_src(&c) == 0, "src is layer 0 (first full-attention)");
    ok(lz_kv_share_target(&c, 3) == 1,
       "ced off: layer 3 is still a valid attn_kv_share target");

    /* Row 2: CED on, split=1 - layer 3 is in the decoder region. */
    c.ced_split = 1;
    ok(lz_kv_share_target(&c, 3) == 0,
       "ced on (split=1): layer 3 is inside the decoder region, not "
       "a valid attn_kv_share target");

    /* Row 3: ced_split == n_layers is CED off, not a decoder covering
       every layer - layer 3 is a target again. */
    c.ced_split = 4;
    ok(lz_kv_share_target(&c, 3) == 1,
       "ced_split == n_layers is CED off, not a decoder covering "
       "every layer - layer 3 is a target again");

    if (g_fail) {
        printf("\nkv_share_ced_boundary_probe: FAIL\n");
        return 1;
    }
    printf("\nkv_share_ced_boundary_probe: PASS\n");
    return 0;
}
