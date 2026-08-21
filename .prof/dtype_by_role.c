/* Throwaway: which tensor ROLES carry which dtype, so the int-pipeline
   work can be aimed per format rather than per aggregate share.
   Not part of the build; run from the engine tree root. */
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

/* One row per (role, dtype) pair seen, accumulated across layers. */
#define MAXR 64
static char rname[MAXR][32];
static int  rd[MAXR];
static unsigned long long rn[MAXR];
static unsigned long long rc[MAXR];
static int nrole;

static void acc(const char *name, const LZTensor *t) {
    int i;
    if (!t || t->n == 0) return;
    for (i = 0; i < nrole; i++)
        if (rd[i] == t->dtype && strcmp(rname[i], name) == 0) break;
    if (i == nrole) {
        if (nrole >= MAXR) return;
        strncpy(rname[i], name, 31); rname[i][31] = 0;
        rd[i] = t->dtype; rn[i] = 0; rc[i] = 0;
        nrole++;
    }
    rn[i] += (unsigned long long)t->n;
    rc[i]++;
}

int main(int argc, char **argv) {
    LZModel m;
    char err[512];
    int li, e, i;
    const char *dir = (argc > 1) ? argv[1] : "/e/LLM/models/kmr20";

    memset(&m, 0, sizeof(m));
    if (lz_open(&m, dir, err, sizeof(err)) != 0) { printf("open: %s\n", err); return 1; }
    if (lz_read_weights(&m, err, sizeof(err)) != 0) { printf("read: %s\n", err); return 1; }

    for (li = 0; li < m.config.n_layers; li++) {
        LZLayer *L = &m.layers[li];
        acc("q_proj", &L->q_proj); acc("k_proj", &L->k_proj);
        acc("v_proj", &L->v_proj); acc("o_proj", &L->o_proj);
        acc("kda_q_proj", &L->kda_q_proj); acc("kda_k_proj", &L->kda_k_proj);
        acc("kda_v_proj", &L->kda_v_proj); acc("kda_o_proj", &L->kda_o_proj);
        acc("kda_g_proj", &L->kda_g_proj);
        acc("kda_f_a_proj", &L->kda_f_a_proj); acc("kda_f_b_proj", &L->kda_f_b_proj);
        acc("kda_b_proj", &L->kda_b_proj);
        acc("in_proj_qkv", &L->in_proj_qkv); acc("in_proj_z", &L->in_proj_z);
        acc("in_proj_a", &L->in_proj_a); acc("in_proj_b", &L->in_proj_b);
        acc("out_proj", &L->out_proj);
        acc("moe_down_proj", &L->moe_down_proj); acc("moe_up_proj", &L->moe_up_proj);
        acc("gate_proj", &L->gate_proj); acc("up_proj", &L->up_proj);
        acc("down_proj", &L->down_proj);
        if (L->moe_expert_w1) {
            for (e = 0; e < m.config.num_experts; e++) {
                acc("moe_expert_w1", &L->moe_expert_w1[e]);
                acc("moe_expert_w2", &L->moe_expert_w2[e]);
                acc("moe_expert_w3", &L->moe_expert_w3[e]);
            }
        }
        acc("moe_gate_w", &L->moe_gate_w);
        acc("moe_shared_gate", &L->moe_shared_gate);
        acc("moe_shared_up", &L->moe_shared_up);
        acc("moe_shared_down", &L->moe_shared_down);
    }
    acc("embed_tokens", &m.embed_tokens);

    printf("%-18s %-7s %14s %8s\n", "role", "dtype", "elements", "tensors");
    for (i = 0; i < nrole; i++)
        printf("%-18s %-7s %14llu %8llu\n", rname[i], fmt(rd[i]), rn[i], rc[i]);
    lz_free(&m);
    return 0;
}
