/* Throwaway: check kda_q_proj/kda_k_proj/kda_v_proj dtypes and confirm
   in_proj_qkv (plain SSM) is unpopulated on the model given. Not part
   of the build. */
#include <stdio.h>
#include <string.h>
#include "model.h"

static const char *fmt(int d) {
    switch (d) {
        case LZ_FMT_F32:   return "f32";
        case LZ_FMT_Q8_0:  return "q8_0";
        case LZ_FMT_Q4_1:  return "q4_1";
        case LZ_FMT_Q6_1:  return "q6_1";
        case LZ_FMT_Q16_0: return "q16_0";
        case LZ_FMT_T2:    return "t2";
        default:           return "?";
    }
}

static void row(const char *name, const LZTensor *t) {
    printf("  %-24s dtype=%-6s gs=%-5d n=%llu\n",
           name, fmt(t->dtype), t->gs, (unsigned long long)t->n);
}

int main(int argc, char **argv) {
    LZModel m;
    char err[512];
    int li;
    const char *dir = (argc > 1) ? argv[1] : "/tmp/kmr20";

    memset(&m, 0, sizeof(m));
    if (lz_open(&m, dir, err, sizeof(err)) != 0) {
        printf("open failed: %s\n", err);
        return 1;
    }
    if (lz_read_weights(&m, err, sizeof(err)) != 0) {
        printf("read failed: %s\n", err);
        return 1;
    }
    for (li = 0; li < m.config.n_layers; li++) {
        LZLayer *L = &m.layers[li];
        printf("layer %d (type %s):\n", li, lz_layer_type_name(L->type));
        row("in_proj_qkv", &L->in_proj_qkv);
        row("kda_q_proj", &L->kda_q_proj);
        row("kda_k_proj", &L->kda_k_proj);
        row("kda_v_proj", &L->kda_v_proj);
        row("kda_g_proj", &L->kda_g_proj);
        row("kda_o_proj", &L->kda_o_proj);
        row("kda_f_a_proj", &L->kda_f_a_proj);
        row("kda_f_b_proj", &L->kda_f_b_proj);
        row("kda_b_proj", &L->kda_b_proj);
        row("gate_proj", &L->gate_proj);
        row("up_proj", &L->up_proj);
        row("down_proj", &L->down_proj);
        row("moe_gate_w", &L->moe_gate_w);
        row("moe_down_proj", &L->moe_down_proj);
        row("moe_up_proj", &L->moe_up_proj);
        row("moe_shared_gate", &L->moe_shared_gate);
        row("moe_shared_up", &L->moe_shared_up);
        row("moe_shared_down", &L->moe_shared_down);
        if (L->moe_expert_w1) {
            row("moe_expert_w1[0]", &L->moe_expert_w1[0]);
            row("moe_expert_w2[0]", &L->moe_expert_w2[0]);
            row("moe_expert_w3[0]", &L->moe_expert_w3[0]);
        }
    }
    lz_free(&m);
    return 0;
}
