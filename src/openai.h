#ifndef LZ_OPENAI_H
#define LZ_OPENAI_H

#include "http.h"
#include "llama_zh.h"
#include "chat.h"

/* OpenAI-compatible endpoint, on top of the HTTP core.

   Routes:
     POST /v1/chat/completions   the whole point; streams when asked
     GET  /v1/models             clients probe this to discover the name
     GET  /health                liveness for the reverse proxy

   Concurrency is 1 by design: one LZRunState, one prefix cache, one
   request at a time. That is not a limitation being worked around, it is
   the deployment - the machine has one core and tens of MB of weights to
   stream per token (the exact figure depends on a quantization recipe
   that is not settled; see LZ_BATCH_MAX in ops.h), so a second
   concurrent request would not be served faster, it would make both
   slower and double the memory.

   What is deliberately NOT implemented, so that a client asking for it
   gets an error rather than silence: tools/function calling, n > 1,
   logprobs (and top_logprobs), images, vLLM's stop_token_ids, and
   stream_options.continuous_usage_stats. A pruned 57.6M model has no
   tool-calling ability, and answering "here is your one choice" to n=5
   is worse than saying the parameter is unsupported.

   That list is a CONTRACT, not a description - every entry is a refusal
   in parse_body, so the header
   never promises a policy the server does not apply. Adding an entry
   here means adding both.

   ## Streaming shape

   Follows vLLM's chat stream generator
   (entrypoints/openai/chat_completion/serving.py):

     data: {... "choices":[{"delta":{"role":"assistant","content":""},
                            "logprobs":null,"finish_reason":null,
                            "stop_reason":null}]}
     data: {... "delta":{"content":"..."} ...}          (role NOT repeated)
     data: {... "delta":{}, "finish_reason":"stop"|"length"|"abort",
                "stop_reason":<the stop string>|null ...}
     data: {... "choices":[], "usage":{...}}     only if include_usage
     data: [DONE]

   The opening chunk is sent even when the reply turns out to be empty:
   it is how a client learns the turn started, which on a Pentium is tens
   of seconds before the first token.

   `usage` rides a trailing choices=[] event and ONLY when
   stream_options.include_usage asked for it. Sending it unasked breaks
   clients that assume every chunk has a choices[0].

   finish_reason maps LZFinish through openai.c's finish_name; note
   "abort" is vLLM's value for a client disconnect and is not in
   OpenAI's own enum.

   ## Token accounting

   prompt_tokens is the WHOLE rendered prompt, including any part served
   from the prefix cache - reuse is an internal optimization and must not
   be visible. completion_tokens includes the token that ended the
   generation (an EOS, or the one that completed a stop string), which is
   what vLLM counts: it appends that token to output_token_ids before
   stopping (v1/engine/detokenizer.py). LZGenOpts.out_tokens_n is a
   DIFFERENT number - it excludes exactly those - and using it here would
   under-report by one. */

typedef struct {
    const LZModel   *model;
    LZTokenizer     *tok;
    /* Exactly one of `pool` or (`state` + `prefix`) is used.

       `pool` is the multi-slot path: it owns its own states, and the
       state a request runs on is whichever slot best matches that
       request's prefix. Set it and `state`/`prefix` are ignored.

       Both shapes exist because the memory decision is the caller's:
       a slot is a whole LZRunState (3.86 MB at seq 2048), and a 64 MB
       Win98 box may not have a second one to give. */
    LZSessionPool   *pool;      /* NULL -> single-state path below */
    LZRunState      *state;
    LZPrefixCache   *prefix;    /* may be NULL to disable reuse */
    const char      *model_name;/* what /v1/models reports */
    int              enable_thinking;
    int              max_new_default;
    /* Set when a request is being served; the sink and the cancel
       callback need it. Not a parameter because LZTokenSink's ctx is
       already spoken for. */
    LZHttpResp      *resp;
} LZOpenAICtx;

/* Handler for lz_http_serve_one. ctx is an LZOpenAICtx *. */
void lz_openai_handle(const LZHttpReq *req, LZHttpResp *rs, void *ctx);

/* HTTP status for an engine error code. Exposed for testing: the
   mapping is the contract between the engine's LZErr and what a client
   sees, and getting it backwards (500 for a client's bad request) makes
   a well-behaved client retry forever. */
int lz_openai_status_for(int lz_err);

#endif
