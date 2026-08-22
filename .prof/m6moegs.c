/* int-pipeline milestone 6: the two structural facts the MoE latent's
 * int16 exit rests on, measured per checkpoint rather than argued.
 *
 *   1. moe_down_proj's dtype/gs and its lz_epi_post_e entry - i.e.
 *      whether the producer reaches epi_fixed_align_i16 at all.
 *   2. whether lz_act_gs varies across the routed experts. That is the
 *      hazard the old comment above the expert loop's quantize named
 *      ("the day w1/w3 carry different weight gs"), and it is what
 *      decides whether one quantization can serve every expert of a
 *      token instead of one per expert.
 *
 * Not part of the build. Build (the engine's link set, minus the CLI):
 *   gcc -O3 -march=x86-64 -std=c99 -Isrc -ffp-contract=off \
 *       -DLZ_MMX_TU=1 -DLZ_SSE2_TU=1 -o .prof/m6moegs.exe .prof/m6moegs.c \
 *       src/err.c src/compat.c src/lfn.c src/json.c src/safetensors.c \
 *       src/model.c src/ops.c src/ops_quant.c src/ops_sched.c \
 *       src/ops_matmul.c src/ops_t2_arm.c src/ops_arm.c \
 *       src/ops_t2_scalar.c src/ops_norm.c src/ops_gdn.c src/fwht.c \
 *       src/cpucheck.c build/native/*.o -lm
 * Usage: m6moegs.exe <model-dir>
 */
#include <stdio.h>
#include <string.h>
#include "model.h"
#include "ops.h"
/* ops_matmul.h needs the row-kernel tables; one declaration is cheaper. */
int lz_epi_post_e(const LZTensor *w);

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

int main(int argc, char **argv) {
    LZModel m;
    char err[512];
    int li, ei;
    const char *dir = (argc > 1) ? argv[1] : "/e/LLM/models/kmr20";
    int latent, hidden, ne;
    long long dn = 0;
    int dt = 0;
    int gsmin = 1 << 30, gsmax = -1;
    int nvary = 0, nw13 = 0;

    memset(&m, 0, sizeof(m));
    if (lz_open(&m, dir, err, sizeof(err)) != 0) {
        printf("open failed: %s\n", err);
        return 1;
    }
    if (lz_read_weights(&m, err, sizeof(err)) != 0) {
        printf("read failed: %s\n", err);
        return 1;
    }
    latent = m.config.moe_latent_dim;
    hidden = m.config.hidden_size;
    ne     = m.config.num_experts;
    printf("hidden=%d latent=%d n_experts=%d topk=%d inter=%d shared_w=%d "
           "n_layers=%d use_subn=%d\n",
           hidden, latent, ne, m.config.num_experts_per_token,
           m.config.moe_intermediate_size, m.config.moe_shared_width,
           m.config.n_layers, m.config.use_subn);
    for (li = 0; li < m.config.n_layers; li++) {
        LZLayer *L = &m.layers[li];
        if (L->moe_down_proj.n) {
            printf("layer %d moe_down_proj dtype=%s gs=%d n=%llu "
                   "act_gs(hidden)=%d epi_post_e=%d\n",
                   li, fmt(L->moe_down_proj.dtype), L->moe_down_proj.gs,
                   (unsigned long long)L->moe_down_proj.n,
                   lz_act_gs(&L->moe_down_proj, hidden),
                   lz_epi_post_e(&L->moe_down_proj));
            dn += (long long)L->moe_down_proj.n;
            dt++;
        }
        if (!L->moe_expert_w1 || !L->moe_expert_w3) continue;
        for (ei = 0; ei < ne; ei++) {
            int g1 = lz_act_gs(&L->moe_expert_w1[ei], latent);
            int g3 = lz_act_gs(&L->moe_expert_w3[ei], latent);
            nw13++;
            if (g1 != g3) nvary++;
            if (g1 < gsmin) gsmin = g1;
            if (g1 > gsmax) gsmax = g1;
            if (g3 < gsmin) gsmin = g3;
            if (g3 > gsmax) gsmax = g3;
        }
    }
    printf("moe_down_proj: %d tensors, %lld elements\n", dt, dn);
    printf("expert w1/w3 pairs scanned=%d  act_gs min=%d max=%d  "
           "w1!=w3 pairs=%d\n", nw13, gsmin, gsmax, nvary);
    {
        int umin = 1 << 30, umax = -1;
        for (li = 0; li < m.config.n_layers; li++) {
            LZLayer *L = &m.layers[li];
            if (!L->moe_up_proj.n) continue;
            {
                int g = lz_act_gs(&L->moe_up_proj, latent);
                if (g < umin) umin = g;
                if (g > umax) umax = g;
            }
        }
        printf("moe_up_proj act_gs(latent) min=%d max=%d\n", umin, umax);
    }
    lz_free(&m);
    return 0;
}
