/* kunkun98-serve - OpenAI-compatible endpoint over the HTTP core.

   A frontend, not engine code: console output lives here the way it
   lives in cli_main.c, and everything below (openai.c, http.c, net.c)
   reports through errbuf.

   Usage: kunkun98-serve <model-dir> [--port N] [--ctx N] [--think]
                                     [--no-ckpt] [--name NAME]

   Listens on 127.0.0.1 only. Put a reverse proxy in front of it for
   anything else - it speaks plain HTTP and authenticates nobody. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "http.h"
#include "net.h"
#include "openai.h"

static char err[1024];
static volatile int g_stop = 0;

static void usage(void) {
    printf("Usage: kunkun98-serve <model-dir> [options]\n"
           "  --port N     listen port (default 8080)\n"
           "  --ctx N      context slots (default 2048)\n"
           "  --think      thinking-mode defaults\n"
           "  --no-ckpt    disable cross-request prefix reuse\n"
           "  --name NAME  model id reported by /v1/models\n"
           "  --requests N serve N requests then exit (testing)\n");
}

int main(int argc, char **argv) {
    LZModel m;
    LZTokenizer tok;
    LZRunState st;
    LZPrefixCache pc;
    LZSessionPool pool;
    LZOpenAICtx ctx;
    LZHttpServer sv;
    const char *dir = NULL, *name = "kunkun98";
    char tokpath[1024];
    int port = 8080, want_ctx = 2048, think = 0, ckpt = 1, max_req = -1;
    int slots = 1, have_pool = 0;
    int i, served = 0, have_pc = 0, rc = 1;

    lz_init_stdout();
    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (a[0] != '-' && !dir) dir = a;
        else if (strcmp(a, "--port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
        else if (strcmp(a, "--ctx") == 0 && i + 1 < argc) want_ctx = atoi(argv[++i]);
        else if (strcmp(a, "--think") == 0) think = 1;
        else if (strcmp(a, "--no-ckpt") == 0) ckpt = 0;
        else if (strcmp(a, "--slots") == 0 && i + 1 < argc) slots = atoi(argv[++i]);
        else if (strcmp(a, "--name") == 0 && i + 1 < argc) name = argv[++i];
        else if (strcmp(a, "--requests") == 0 && i + 1 < argc) max_req = atoi(argv[++i]);
        else { usage(); return 2; }
    }
    if (!dir) { usage(); return 2; }

    if (lz_open(&m, dir, err, sizeof(err)) != 0) {
        printf("open: %s\n", err); return 1;
    }
    if (lz_read_weights(&m, err, sizeof(err)) != 0) {
        printf("weights: %s\n", err); lz_free(&m); return 1;
    }
    snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", dir);
    if (lz_tokenizer_load(&tok, tokpath, err, sizeof(err)) != 0) {
        printf("tokenizer: %s\n", err); lz_free(&m); return 1;
    }
    if (want_ctx > m.config.seq_len) want_ctx = m.config.seq_len;
    if (want_ctx < 256) want_ctx = 256;
    /* 0: this server does not wire --spec yet - see generate.c's own
       lz_pool_alloc comment on the same point (forward.h's spec_k_max
       comment). */
    if (lz_state_alloc(&st, &m, want_ctx, 0, err, sizeof(err)) != 0) {
        printf("state: %s\n", err);
        lz_tokenizer_free(&tok); lz_free(&m); return 1;
    }
    /* One slot is the single-state path: allocating a pool of 1 would
       work, but it would spend a second LZRunState (st, above) for
       nothing. Above 1, the pool owns every state and `st` goes unused. */
    if (ckpt && slots > 1) {
        if (lz_pool_init(&pool, &m, slots, want_ctx, err, sizeof(err)) == 0) {
            have_pool = 1;
        } else {
            printf("slots: %s (falling back to 1)\n", err);
        }
    }
    if (ckpt && !have_pool && lz_prefix_init(&pc, &m, err, sizeof(err)) == 0)
        have_pc = 1;

    memset(&ctx, 0, sizeof(ctx));
    ctx.model = &m;
    ctx.tok = &tok;
    ctx.state = &st;
    ctx.prefix = have_pc ? &pc : NULL;
    ctx.pool = have_pool ? &pool : NULL;
    ctx.model_name = name;
    ctx.enable_thinking = think;
    ctx.max_new_default = 512;

    if (lz_net_init(err, sizeof(err)) != 0) {
        printf("net: %s\n", err); goto cleanup;
    }
    /* 256 KB request cap: a long conversation body fits, and it bounds
       what one client can make this process hold at once. */
    if (lz_http_open(&sv, port, 256 * 1024, err, sizeof(err)) != 0) {
        printf("listen: %s\n", err);
        lz_net_shutdown();
        goto cleanup;
    }
    printf("kunkun98-serve on http://127.0.0.1:%d  model=%s ctx=%d "
           "prefix-reuse=%s slots=%d\n", port, name, want_ctx,
           (have_pc || have_pool) ? "on" : "off", have_pool ? slots : 1);
    printf("ready\n");
    fflush(stdout);

    while (!g_stop && (max_req < 0 || served < max_req)) {
        int r = lz_http_serve_one(&sv, lz_openai_handle, &ctx, 1000,
                                  err, sizeof(err));
        if (r < 0) { printf("serve: %s\n", err); break; }
        if (r > 0) served++;
    }
    rc = 0;
    lz_http_close(&sv);
    lz_net_shutdown();

cleanup:
    if (have_pool) lz_pool_free(&pool);
    if (have_pc) lz_prefix_free(&pc);
    lz_state_free(&st);
    lz_tokenizer_free(&tok);
    lz_free(&m);
    return rc;
}
