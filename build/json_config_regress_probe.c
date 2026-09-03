/* config.json regression harness for build/json_offset_gate.sh.
   Calls the REAL lz_load_config (src/model.c) - not a reimplementation
   of its field list - on a real config.json and prints every scalar it
   reads. The gate runs this once linked against the current tree and
   once against a mutant with LZJsonNode.num narrowed back to float, and
   diffs the two: every field here is small (vocab_size, hidden_size,
   rope_theta, ...), nowhere near float32's 2^24 exactness boundary, so
   the two outputs must be byte-for-byte identical. A diff here would
   mean the double/float change reached something it should not have. */
#include <stdio.h>
#include "model.h"

int main(int argc, char **argv) {
    LZModelConfig c;
    char errbuf[512];

    if (argc < 2) {
        fprintf(stderr, "usage: %s config.json\n", argv[0]);
        return 2;
    }
    if (lz_load_config(&c, argv[1], errbuf, sizeof errbuf) != 0) {
        printf("lz_load_config FAILED: %s\n", errbuf);
        return 1;
    }
    printf("vocab_size=%d\n", c.vocab_size);
    printf("hidden_size=%d\n", c.hidden_size);
    printf("n_layers=%d\n", c.n_layers);
    printf("n_heads=%d\n", c.n_heads);
    printf("n_kv_heads=%d\n", c.n_kv_heads);
    printf("head_dim=%d\n", c.head_dim);
    printf("lin_n_k_heads=%d\n", c.lin_n_k_heads);
    printf("lin_n_v_heads=%d\n", c.lin_n_v_heads);
    printf("lin_k_head_dim=%d\n", c.lin_k_head_dim);
    printf("lin_v_head_dim=%d\n", c.lin_v_head_dim);
    printf("conv_kernel=%d\n", c.conv_kernel);
    printf("intermediate_size=%d\n", c.intermediate_size);
    printf("mtp_n_layers=%d\n", c.mtp_n_layers);
    printf("mtp_intermediate_size=%d\n", c.mtp_intermediate_size);
    printf("engram=%d\n", c.engram);
    printf("engram_ngram=%d\n", c.engram_ngram);
    printf("engram_n_head=%d\n", c.engram_n_head);
    printf("engram_dim=%d\n", c.engram_dim);
    printf("engram_size=%d\n", c.engram_size);
    printf("seq_len=%d\n", c.seq_len);
    printf("rms_norm_eps=%.9g\n", (double)c.rms_norm_eps);
    printf("full_attention_interval=%d\n", c.full_attention_interval);
    printf("tie_word_embeddings=%d\n", c.tie_word_embeddings);
    printf("attn_output_gate=%d\n", c.attn_output_gate);
    printf("hadamard_o=%d\n", c.hadamard_o);
    printf("hadamard_down=%d\n", c.hadamard_down);
    printf("kda_gate_rank=%d\n", c.kda_gate_rank);
    printf("kda_gate_lower_bound=%.9g\n", (double)c.kda_gate_lower_bound);
    printf("rope_theta=%.9g\n", (double)c.rope_theta);
    printf("partial_rotary_factor=%.9g\n", (double)c.partial_rotary_factor);
    printf("num_experts=%d\n", c.num_experts);
    printf("num_experts_per_token=%d\n", c.num_experts_per_token);
    printf("num_shared_experts=%d\n", c.num_shared_experts);
    printf("moe_intermediate_size=%d\n", c.moe_intermediate_size);
    printf("moe_latent_dim=%d\n", c.moe_latent_dim);
    printf("first_k_dense_replace=%d\n", c.first_k_dense_replace);
    printf("moe_renormalize=%d\n", c.moe_renormalize);
    printf("moe_latent_use_norm=%d\n", c.moe_latent_use_norm);
    return 0;
}
