#include <stdlib.h>
#include <string.h>

#include "openai.h"
#include "json.h"
#include "err.h"
#include "compat.h"         /* lz_time_epoch, for `created` */

#define LZ_OA_MAX_MSGS  LZ_CHAT_HIST_MAX

/* ------------------------------------------------------- status mapping */

int lz_openai_status_for(int e) {
    switch (e) {
    case LZ_ERR_OK:             return 200;
    /* The client must change the request - but each of these needs a
       DIFFERENT change, which is why they stay distinct rather than
       collapsing into one 400. Trimming history fixes the first and does
       nothing for the others. */
    case LZ_ERR_PROMPT_LONG:    return 400;
    case LZ_ERR_STOP_LONG:      return 400;
    case LZ_ERR_PROMPT_ENCODE:  return 400;
    case LZ_ERR_NO_MESSAGES:    return 400;
    case LZ_ERR_NO_USER:        return 400;
    case LZ_ERR_SYSTEM_FIRST:   return 400;
    case LZ_ERR_ROLE:           return 400;
    case LZ_ERR_TRUNC:          return 400;
    case LZ_ERR_HIST_FULL:      return 400;
    default:                    return 500;
    }
}

/* OpenAI's error.type, so a client can branch without parsing prose.
   Their own vocabulary, not ours: a client library matches on these. */
static const char *err_type(int status) {
    return status == 400 ? "invalid_request_error" : "server_error";
}

static void send_error(LZHttpResp *rs, int status, const char *msg) {
    LZJsonW w;
    lz_jw_init(&w);
    lz_jw_lit(&w, "{\"error\":{\"message\":");
    lz_jw_str(&w, msg, -1);
    lz_jw_lit(&w, ",\"type\":");
    lz_jw_str(&w, err_type(status), -1);
    lz_jw_lit(&w, ",\"param\":null,\"code\":null}}");
    if (!w.err && !rs->started)
        lz_http_reply(rs, status, "application/json", w.s, w.len);
    else if (!rs->started)
        lz_http_reply(rs, 500, "application/json",
                      "{\"error\":{\"message\":\"out of memory\"}}", 38);
    lz_jw_free(&w);
}

/* --------------------------------------------------------- request body */

typedef struct {
    LZChatMsg msgs[LZ_OA_MAX_MSGS];
    int       n_msgs;
    int       stream;
    int       include_usage;   /* stream_options.include_usage */
    int       max_tokens;
    int       has_temp;   float temp;
    int       has_topp;   float topp;
    int       has_seed;   float seed;
    /* The rest of what the sampler already implements. presence_penalty
       and frequency_penalty are OpenAI-STANDARD and must not be silently
       dropped: openai.h states the policy as "a client asking for
       something unsupported gets an error rather than silence", and `n`
       is refused exactly that way. This engine defaults presence_penalty
       to 1.5 (Qwen's published value), so a client sending
       presence_penalty: 0 to turn it off would otherwise get 1.5 and no
       hint that its request had been ignored.

       top_k and min_p are not OpenAI fields; they are the llama.cpp /
       vLLM extension spelling, and both are in Qwen's published defaults
       (top_k 20). Accepting them costs nothing and refusing them would
       be refusing a parameter we implement. */
    int       has_pres;   float pres;
    int       has_freq;   float freq;
    int       has_rep;    float rep;
    int       has_topk;   float topk;
    int       has_minp;   float minp;
    const char *stop[LZ_MAX_STOP];
    int       n_stop;
} LZOaReq;

/* Parse the body. Returns 0, or an LZErr whose status_for maps it. On
   failure *why gets a message meant for the client. */
static int parse_body(LZJson *j, const char *body, int len, LZOaReq *r,
                      const char **why) {
    const LZJsonNode *root, *msgs, *e;
    char jerr[256];
    int i;

    memset(r, 0, sizeof(*r));
    r->max_tokens = -1;
    *why = NULL;

    if (!body || len <= 0) { *why = "request body is empty"; return LZ_ERR_NO_MESSAGES; }
    if (lz_json_parse(j, body, (size_t)len, jerr, sizeof(jerr)) != 0) {
        *why = "request body is not valid JSON";
        return LZ_ERR_JSON_ROOT;
    }
    root = lz_json_root(j);
    if (!root || root->type != LZ_JSON_OBJ) {
        *why = "request body must be a JSON object";
        return LZ_ERR_JSON_ROOT;
    }
    msgs = lz_json_get(j, root, "messages");
    if (!msgs || msgs->type != LZ_JSON_ARR) {
        *why = "messages must be an array";
        return LZ_ERR_NO_MESSAGES;
    }
    for (e = lz_json_first(j, msgs); e; e = lz_json_next(j, e)) {
        const char *role, *content;
        if (r->n_msgs >= LZ_OA_MAX_MSGS) {
            *why = "too many messages";
            return LZ_ERR_HIST_FULL;
        }
        if (e->type != LZ_JSON_OBJ) {
            *why = "each message must be an object";
            return LZ_ERR_ROLE;
        }
        role = lz_json_get_str(j, e, "role", NULL);
        content = lz_json_get_str(j, e, "content", NULL);
        if (!role) { *why = "message is missing role"; return LZ_ERR_ROLE; }
        /* content: null is what tool-call messages carry. Treating it as
           an empty string would silently render a blank turn; refusing
           says what is actually unsupported. */
        if (!content) {
            *why = "message content must be a string "
                   "(tool calls and multi-part content are not supported)";
            return LZ_ERR_ROLE;
        }
        if (strcmp(role, "system") == 0)         r->msgs[r->n_msgs].role = LZ_ROLE_SYSTEM;
        else if (strcmp(role, "user") == 0)      r->msgs[r->n_msgs].role = LZ_ROLE_USER;
        else if (strcmp(role, "assistant") == 0) r->msgs[r->n_msgs].role = LZ_ROLE_ASSISTANT;
        else { *why = "unknown role (expected system/user/assistant)";
               return LZ_ERR_ROLE; }
        /* Borrowed from the JSON arena, which outlives the render. */
        r->msgs[r->n_msgs].content = content;
        r->msgs[r->n_msgs].len = -1;
        r->n_msgs++;
    }
    if (r->n_msgs == 0) { *why = "messages is empty"; return LZ_ERR_NO_MESSAGES; }

    {   /* "stream": true is a BOOLEAN, and lz_json_get_int only reads
           LZ_JSON_NUM - it returns the default for a bool, so reading it
           that way silently ignores the flag and answers every streaming
           request with a buffered body. The client waits for events that
           never come and then gets one lump; nothing errors. */
        const LZJsonNode *sn = lz_json_get(j, root, "stream");
        r->stream = sn && ((sn->type == LZ_JSON_BOOL && sn->num != 0.0) ||
                           (sn->type == LZ_JSON_NUM && sn->num != 0.0));
    }
    {   /* stream_options.include_usage: without it a streaming client
           has NO way to learn the token counts, because the usage block
           only exists on the non-streaming body. Ignoring it silently
           is the failure openai.h names by policy -
           and silently, here, means the client reads usage=None and
           reports 0 tokens for every streamed request it ever made.

           continuous_usage_stats (vLLM's other field) is refused rather
           than ignored: it changes the shape of EVERY chunk, so a client
           that asked for it and did not get it is parsing something
           different from what it expects. */
        const LZJsonNode *so = lz_json_get(j, root, "stream_options");
        if (so && so->type == LZ_JSON_OBJ) {
            const LZJsonNode *iu = lz_json_get(j, so, "include_usage");
            r->include_usage =
                iu && ((iu->type == LZ_JSON_BOOL && iu->num != 0.0) ||
                       (iu->type == LZ_JSON_NUM && iu->num != 0.0));
            if (lz_json_get(j, so, "continuous_usage_stats")) {
                *why = "stream_options.continuous_usage_stats is not supported"
                       " (include_usage is)";
                return LZ_ERR_ROLE;
            }
        }
    }

    r->max_tokens = lz_json_get_int(j, root, "max_tokens", -1);
    if (r->max_tokens < 0)
        r->max_tokens = lz_json_get_int(j, root, "max_completion_tokens", -1);

    {   /* A sentinel that cannot be a legal value, so "absent" and
           "explicitly 0" stay distinguishable - temperature 0 is greedy
           decoding, a thing users ask for on purpose. */
        float miss = -1e30f;
        float v = lz_json_get_num(j, root, "temperature", miss);
        if (v != miss) { r->has_temp = 1; r->temp = v; }
        v = lz_json_get_num(j, root, "top_p", miss);
        if (v != miss) { r->has_topp = 1; r->topp = v; }
        v = lz_json_get_num(j, root, "seed", miss);
        if (v != miss) { r->has_seed = 1; r->seed = v; }
        /* The sentinel matters most here: 0 is the OpenAI default for
           both penalties and a perfectly ordinary "turn it off", while
           this engine's default is 1.5. Reading them with a 0 default
           would make "absent" mean "off" and quietly override the
           configured behaviour in the other direction. */
        v = lz_json_get_num(j, root, "presence_penalty", miss);
        if (v != miss) { r->has_pres = 1; r->pres = v; }
        v = lz_json_get_num(j, root, "frequency_penalty", miss);
        if (v != miss) { r->has_freq = 1; r->freq = v; }
        v = lz_json_get_num(j, root, "repetition_penalty", miss);
        if (v != miss) { r->has_rep = 1; r->rep = v; }
        v = lz_json_get_num(j, root, "top_k", miss);
        if (v != miss) { r->has_topk = 1; r->topk = v; }
        v = lz_json_get_num(j, root, "min_p", miss);
        if (v != miss) { r->has_minp = 1; r->minp = v; }
    }

    {   /* stop may be a string or an array of them. */
        const LZJsonNode *st = lz_json_get(j, root, "stop");
        if (st && st->type == LZ_JSON_STR) {
            r->stop[0] = st->text;
            r->n_stop = 1;
        } else if (st && st->type == LZ_JSON_ARR) {
            for (e = lz_json_first(j, st); e; e = lz_json_next(j, e)) {
                if (r->n_stop >= LZ_MAX_STOP) break;
                if (e->type != LZ_JSON_STR) continue;
                r->stop[r->n_stop++] = e->text;
            }
        }
    }

    if (lz_json_get_int(j, root, "n", 1) != 1) {
        *why = "n must be 1 (multiple choices are not supported)";
        return LZ_ERR_ROLE;
    }
    if (lz_json_get(j, root, "tools") || lz_json_get(j, root, "functions")) {
        *why = "tools/function calling is not supported by this model";
        return LZ_ERR_ROLE;
    }
    /* vLLM's stop_token_ids. Refused, not ignored: a client using it is
       relying on generation ending somewhere, and ignoring it produces a
       reply that runs past that point with nothing to say so. */
    if (lz_json_get(j, root, "stop_token_ids")) {
        *why = "stop_token_ids is not supported; use stop strings";
        return LZ_ERR_ROLE;
    }
    {   /* openai.h lists logprobs among the things that get an ERROR
           rather than silence - a header promise this check makes true.

           Truthiness, not presence: `logprobs: false` is what a client
           that never wants them sends on every request, and refusing
           that would break clients over a parameter they turned off. */
        const LZJsonNode *lp = lz_json_get(j, root, "logprobs");
        const LZJsonNode *tl = lz_json_get(j, root, "top_logprobs");
        if ((lp && ((lp->type == LZ_JSON_BOOL && lp->num != 0.0) ||
                    (lp->type == LZ_JSON_NUM && lp->num != 0.0))) ||
            (tl && tl->type == LZ_JSON_NUM && tl->num != 0.0)) {
            *why = "logprobs are not supported by this endpoint";
            return LZ_ERR_ROLE;
        }
    }
    for (i = 0; i < r->n_stop; i++) {
        if ((int)strlen(r->stop[i]) > LZ_STOP_TAIL) {
            *why = "a stop string exceeds the 64-byte limit";
            return LZ_ERR_STOP_LONG;
        }
    }
    return LZ_ERR_OK;
}

/* -------------------------------------------------------------- output */

/* Collects non-streaming output; streams straight to the socket
   otherwise. */
typedef struct {
    LZJsonW      buf;         /* non-streaming accumulation */
    LZOpenAICtx *ctx;
    int          streaming;
    int          role_sent;   /* the first chunk carries the role, once */
    int          n_sent;
    unsigned long created;
    char         id[48];
} LZOaSink;

/* One SSE event.

   The delta shape follows vLLM's chat stream generator:

     first chunk    {"role":"assistant","content":""}   finish_reason null
     content chunk  {"content":"..."}                   finish_reason null
     final chunk    {}                                  finish_reason set

   The role belongs to the FIRST chunk only. Repeating it on every chunk
   is not what a client accumulating deltas expects, and the first chunk
   is also how a client learns the turn has started when the first token
   is still tens of seconds away on a Pentium. `logprobs` is always
   present as null: vLLM sets it explicitly on every choice, and a
   client that reads chunk.choices[0].logprobs without a guard sees a
   missing key rather than a null. */
typedef enum { SSE_FIRST, SSE_CONTENT, SSE_FINAL } LZSseKind;

static void sse_chunk(LZOaSink *s, LZSseKind kind, const char *content,
                      int len, const char *finish, const char *stop_reason) {
    LZJsonW w;
    lz_jw_init(&w);
    lz_jw_lit(&w, "data: {\"id\":");
    lz_jw_str(&w, s->id, -1);
    lz_jw_lit(&w, ",\"object\":\"chat.completion.chunk\",\"created\":");
    lz_jw_int(&w, (long)s->created);
    lz_jw_lit(&w, ",\"model\":");
    lz_jw_str(&w, s->ctx->model_name, -1);
    lz_jw_lit(&w, ",\"choices\":[{\"index\":0,\"delta\":");
    if (kind == SSE_FIRST) {
        lz_jw_lit(&w, "{\"role\":\"assistant\",\"content\":\"\"}");
    } else if (kind == SSE_CONTENT) {
        lz_jw_lit(&w, "{\"content\":");
        lz_jw_str(&w, content, len);
        lz_jw_lit(&w, "}");
    } else {
        lz_jw_lit(&w, "{}");
    }
    lz_jw_lit(&w, ",\"logprobs\":null,\"finish_reason\":");
    if (finish) lz_jw_str(&w, finish, -1); else lz_jw_lit(&w, "null");
    /* vLLM's extension: WHICH stop string fired. finish_reason is "stop"
       for a stop string and for EOS alike, so without this the two are
       indistinguishable on the wire. */
    lz_jw_lit(&w, ",\"stop_reason\":");
    if (stop_reason) lz_jw_str(&w, stop_reason, -1); else lz_jw_lit(&w, "null");
    lz_jw_lit(&w, "}]}\n\n");
    if (!w.err) lz_http_stream_write(s->ctx->resp, w.s, w.len);
    lz_jw_free(&w);
}

/* The trailing usage-only event, sent when stream_options.include_usage
   is set. choices is EMPTY here - that is what makes it distinguishable
   from a real chunk, and it is vLLM's shape. */
static void sse_usage(LZOaSink *s, long prompt_tokens, long completion_tokens) {
    LZJsonW w;
    lz_jw_init(&w);
    lz_jw_lit(&w, "data: {\"id\":");
    lz_jw_str(&w, s->id, -1);
    lz_jw_lit(&w, ",\"object\":\"chat.completion.chunk\",\"created\":");
    lz_jw_int(&w, (long)s->created);
    lz_jw_lit(&w, ",\"model\":");
    lz_jw_str(&w, s->ctx->model_name, -1);
    lz_jw_lit(&w, ",\"choices\":[],\"usage\":{\"prompt_tokens\":");
    lz_jw_int(&w, prompt_tokens);
    lz_jw_lit(&w, ",\"completion_tokens\":");
    lz_jw_int(&w, completion_tokens);
    lz_jw_lit(&w, ",\"total_tokens\":");
    lz_jw_int(&w, prompt_tokens + completion_tokens);
    lz_jw_lit(&w, "}}\n\n");
    if (!w.err) lz_http_stream_write(s->ctx->resp, w.s, w.len);
    lz_jw_free(&w);
}

static void sink_cb(const char *bytes, int len, void *vctx) {
    LZOaSink *s = (LZOaSink *)vctx;
    if (len <= 0) return;
    s->n_sent += len;
    if (s->streaming) {
        if (!s->role_sent) {
            sse_chunk(s, SSE_FIRST, NULL, 0, NULL, NULL);
            s->role_sent = 1;
        }
        sse_chunk(s, SSE_CONTENT, bytes, len, NULL, NULL);
    } else {
        lz_jw_raw(&s->buf, bytes, len);
    }
}

/* Stop generating once the client has gone. Without this a disconnected
   request keeps a Pentium busy for minutes producing bytes that go
   nowhere - and at concurrency 1 that also blocks every later request. */
static int cont_cb(void *vctx) {
    LZOaSink *s = (LZOaSink *)vctx;
    if (s->ctx->resp->failed) return 0;
    return lz_net_peer_gone(s->ctx->resp->sock) ? 0 : 1;
}

/* Verified against vllm/v1/engine/__init__.py FinishReason:
   stop / length / abort / error / repetition. OpenAI itself defines only
   stop, length, tool_calls and content_filter; "abort" is vLLM's, and
   its docstring is "aborted by client", which is exactly what
   LZ_FINISH_CANCELLED means here.

   Reporting a cancelled generation as "stop" would claim a clean end.
   It is usually unobservable - the peer is gone - but that is not a
   reason to say something untrue on the wire: cont_cb also fires on
   resp->failed, and a false positive from lz_net_peer_gone would hand a
   still-listening client a truncated answer labelled complete. */
static const char *finish_name(int f) {
    switch (f) {
    case LZ_FINISH_EOS:
    case LZ_FINISH_STOP:      return "stop";
    case LZ_FINISH_LENGTH:    return "length";
    case LZ_FINISH_CANCELLED: return "abort";
    default:                  return "stop";
    }
}

/* stop_reason: the stop string that fired, or NULL for everything else -
   including EOS, which finish_reason cannot distinguish from it. */
static const char *stop_reason_of(const LZGenOpts *g, const LZOaReq *r) {
    if (g->out_finish != LZ_FINISH_STOP) return NULL;
    if (g->out_stop < 0 || g->out_stop >= r->n_stop) return NULL;
    return r->stop[g->out_stop];
}

/* --------------------------------------------------------------- routes */

static void models(LZOpenAICtx *c, LZHttpResp *rs) {
    LZJsonW w;
    lz_jw_init(&w);
    lz_jw_lit(&w, "{\"object\":\"list\",\"data\":[{\"id\":");
    lz_jw_str(&w, c->model_name, -1);
    lz_jw_lit(&w, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"lunzima\"}]}");
    if (!w.err) lz_http_reply(rs, 200, "application/json", w.s, w.len);
    else send_error(rs, 500, "out of memory");
    lz_jw_free(&w);
}

static void completions(LZOpenAICtx *c, const LZHttpReq *req, LZHttpResp *rs) {
    LZJson j;
    /* On the heap, not the stack: LZOaReq carries msgs[LZ_CHAT_HIST_MAX]
       and that alone is 2 KB, which put this frame at 4288 bytes - over
       the stack budget, on a target whose
       stack is small and whose overflow is not catchable. One malloc per
       request is noise next to the generation it precedes. */
    LZOaReq *r = NULL;
    LZChatBuf cb;
    LZGenOpts g;
    LZOaSink sink;
    const char *why = NULL;
    char err[512];
    int rc, status, n_out = 0, start_pos = 0, suffix_off = 0, reused = 0;
    /* The state this request runs on. With a pool it is chosen per
       request; without one it is the single shared state. */
    LZRunState *st = c ? c->state : NULL;
    float ms = 0.0f;

    memset(&j, 0, sizeof(j));
    r = (LZOaReq *)malloc(sizeof(*r));
    if (!r) { send_error(rs, 500, "out of memory"); return; }
    rc = parse_body(&j, req->body, req->body_len, r, &why);
    if (rc != LZ_ERR_OK) {
        send_error(rs, lz_openai_status_for(rc), why ? why : "bad request");
        lz_json_free(&j);
        free(r);
        return;
    }

    rc = lz_chat_render(r->msgs, r->n_msgs, 1, c->enable_thinking, &cb,
                        err, sizeof(err));
    if (rc != LZ_ERR_OK) {
        /* lz_chat_render returns the LZErr itself, so the status comes
           from the same mapping every other failure uses rather than a
           hardcoded 400 - a render that failed on allocation is ours,
           not the client's. */
        send_error(rs, lz_openai_status_for(rc), err);
        lz_json_free(&j);
        free(r);
        return;
    }

    lz_gen_opts_defaults(&g);
    lz_gen_opts_set_eos(&g, c->tok);
    g.max_new_tokens = r->max_tokens > 0 ? r->max_tokens : c->max_new_default;
    if (r->has_temp) g.sample.temperature = (float)r->temp;
    if (r->has_topp) g.sample.topp = (float)r->topp;
    if (r->has_pres) g.sample.presence_penalty = (float)r->pres;
    if (r->has_freq) g.sample.frequency_penalty = (float)r->freq;
    if (r->has_rep)  g.sample.repetition_penalty = (float)r->rep;
    if (r->has_topk) g.sample.topk = (int)r->topk;
    if (r->has_minp) {
        g.sample.minp = (float)r->minp;
        /* NATIVE units on the wire, always. vLLM's min_p thresholds the
           post-temperature distribution, and a client sending min_p here
           is speaking that protocol - silently reinterpreting it in
           llama.cpp units because that is the local default would make
           the same request mean two different things depending on
           whether the field was sent. The converter is a CLI-side
           convenience; see sampler.h "min_p units". */
        g.sample.minp_llamacpp = 0;
    }
    /* Deterministic by default: a fixed seed makes a bug reproducible,
       and a client that wants variety sends one. The value is the
       project's one fixed seed, so a server run and a CLI run started
       without --seed are comparable. NOT a clock reading: that makes
       two such runs silently different. */
    g.rng_seed = r->has_seed ? (lz_u64)r->seed : LZ_U64_C(1145141919);
    {
        int i;
        for (i = 0; i < r->n_stop; i++) g.stop[i] = r->stop[i];
        g.n_stop = r->n_stop;
    }

    memset(&sink, 0, sizeof(sink));
    lz_jw_init(&sink.buf);
    sink.ctx = c;
    sink.streaming = r->stream;
    /* Real wall clock. Hardcoding 0 here would parse fine everywhere but
       stamp every response 1970-01-01, which breaks any client that
       sorts or ages by it. 0 remains the value when the box has no
       usable clock. */
    sink.created = (unsigned long)lz_time_epoch();
    c->resp = rs;
    /* Not random: this process has one request in flight, so a counter
       is unique for as long as anyone cares, and rand() would need
       seeding that has nothing to do with sampling. */
    {
        static unsigned long seq = 0;
        unsigned long v = ++seq;
        char t[24];
        int k = 0, m, n;
        strcpy(sink.id, "chatcmpl-lz");
        n = (int)strlen(sink.id);
        if (v == 0) t[k++] = '0';
        while (v) { t[k++] = (char)('0' + (int)(v % 10)); v /= 10; }
        for (m = k - 1; m >= 0 && n < (int)sizeof(sink.id) - 1; m--)
            sink.id[n++] = t[m];
        sink.id[n] = 0;
    }

    /* Prefix reuse across requests. The engine primitive does the
       token-exactness and staleness checks; here we only supply the
       split point, which is where the generation-prompt tail begins. */
    if (c->pool || c->prefix) {
        int split = cb.len - (int)strlen(lz_chat_gen_prompt_tail(c->enable_thinking));
        int rcp;
        if (c->pool) {
            /* The pool picks the state too, so `st` is reassigned before
               generation. Both paths leave st/start_pos/suffix_off in the
               same shape, which is why there is one lz_generate_resume
               call below and not two. */
            rcp = lz_pool_prepare(c->pool, c->model, c->tok, cb.s, cb.len,
                                  split, &st, &start_pos, &suffix_off,
                                  &reused, NULL, err, sizeof(err));
            if (!st) st = c->state;         /* defensive; pool always sets it */
        } else {
            rcp = lz_prefix_prepare(c->prefix, c->model, c->tok, st,
                                    cb.s, cb.len, split, &start_pos,
                                    &suffix_off, &reused, NULL,
                                    err, sizeof(err));
        }
        if (rcp != LZ_ERR_OK) {
            start_pos = 0;
            suffix_off = 0;
        }
    }

    if (r->stream) {
        if (lz_http_stream_begin(rs, 200, "text/event-stream") != 0) {
            lz_chat_buf_free(&cb);
            lz_json_free(&j);
            lz_jw_free(&sink.buf);
            free(r);
            return;
        }
    }

    rc = lz_generate_resume(c->model, c->tok, st, start_pos,
                            cb.s + suffix_off, cb.len - suffix_off, &g,
                            sink_cb, cont_cb, &sink, &n_out, &ms,
                            err, sizeof(err));

    if (rc != LZ_ERR_OK) {
        status = lz_openai_status_for(rc);
        if (r->stream) {
            /* Headers are already out; the only honest signal left is an
               error event followed by the terminator. A client that
               ignores it sees a short completion, which is why the
               non-streaming path is preferred for anything that must
               distinguish failure from a short answer. */
            LZJsonW w;
            lz_jw_init(&w);
            lz_jw_lit(&w, "data: {\"error\":{\"message\":");
            lz_jw_str(&w, err, -1);
            lz_jw_lit(&w, ",\"type\":");
            lz_jw_str(&w, err_type(status), -1);
            lz_jw_lit(&w, "}}\n\n");
            if (!w.err) lz_http_stream_write(rs, w.s, w.len);
            lz_jw_free(&w);
            lz_http_stream_write(rs, "data: [DONE]\n\n", 14);
            lz_http_stream_end(rs);
        } else {
            send_error(rs, status, err);
        }
        goto done;
    }

    if (r->stream) {
        const char *sr = stop_reason_of(&g, r);
        /* A generation that emitted nothing still owes the client the
           opening chunk: vLLM sends it before the first token exists,
           and a client that waits for a role delta would otherwise wait
           forever on an empty reply. */
        if (!sink.role_sent) {
            sse_chunk(&sink, SSE_FIRST, NULL, 0, NULL, NULL);
            sink.role_sent = 1;
        }
        sse_chunk(&sink, SSE_FINAL, NULL, 0, finish_name(g.out_finish), sr);
        if (r->include_usage)
            sse_usage(&sink, (long)(start_pos + g.out_prompt_tokens), (long)n_out);
        lz_http_stream_write(rs, "data: [DONE]\n\n", 14);
        lz_http_stream_end(rs);
    } else {
        LZJsonW w;
        const char *sr = stop_reason_of(&g, r);
        lz_jw_init(&w);
        lz_jw_lit(&w, "{\"id\":");
        lz_jw_str(&w, sink.id, -1);
        lz_jw_lit(&w, ",\"object\":\"chat.completion\",\"created\":");
        lz_jw_int(&w, (long)sink.created);
        lz_jw_lit(&w, ",\"model\":");
        lz_jw_str(&w, c->model_name, -1);
        lz_jw_lit(&w, ",\"choices\":[{\"index\":0,\"message\":{\"role\":"
                      "\"assistant\",\"content\":");
        lz_jw_str(&w, sink.buf.s ? sink.buf.s : "", sink.buf.len);
        lz_jw_lit(&w, "},\"logprobs\":null,\"finish_reason\":");
        lz_jw_str(&w, finish_name(g.out_finish), -1);
        lz_jw_lit(&w, ",\"stop_reason\":");
        if (sr) lz_jw_str(&w, sr, -1); else lz_jw_lit(&w, "null");
        lz_jw_lit(&w, "}],\"usage\":{\"prompt_tokens\":");
        lz_jw_int(&w, (long)(start_pos + g.out_prompt_tokens));
        lz_jw_lit(&w, ",\"completion_tokens\":");
        lz_jw_int(&w, (long)n_out);
        lz_jw_lit(&w, ",\"total_tokens\":");
        lz_jw_int(&w, (long)(start_pos + g.out_prompt_tokens + n_out));
        lz_jw_lit(&w, "}}");
        if (!w.err) lz_http_reply(rs, 200, "application/json", w.s, w.len);
        else send_error(rs, 500, "out of memory building the response");
        lz_jw_free(&w);
    }

done:
    lz_chat_buf_free(&cb);
    lz_json_free(&j);
    lz_jw_free(&sink.buf);
    free(r);
    c->resp = NULL;
}

void lz_openai_handle(const LZHttpReq *req, LZHttpResp *rs, void *vctx) {
    LZOpenAICtx *c = (LZOpenAICtx *)vctx;

    if (!c || !c->model || !c->tok || !c->state) {
        send_error(rs, 500, "server is not initialized");
        return;
    }
    if (strcmp(req->path, "/health") == 0) {
        /* Carries the prefix-reuse counters, not just liveness. They are
           here rather than in a log line because the question they exist
           to answer - do conversations interleave enough to justify more
           than one cache slot, at 3.86 MB each (spec 3.5.2-6) - needs
           REAL traffic, and reading a counter from the proxy is the only
           way to get that without a logging pipeline this box has no
           room for. */
        LZJsonW w;
        long calls = 0, hits = 0, saved = 0, miss = 0, unsplit = 0;
        if (c->prefix)
            lz_prefix_stats(c->prefix, &calls, &hits, &saved, &miss, &unsplit);
        lz_jw_init(&w);
        lz_jw_lit(&w, "{\"status\":\"ok\",\"prefix_cache\":");
        lz_jw_lit(&w, c->prefix ? "{\"enabled\":true," : "{\"enabled\":false,");
        lz_jw_lit(&w, "\"calls\":");        lz_jw_int(&w, calls);
        lz_jw_lit(&w, ",\"hits\":");        lz_jw_int(&w, hits);
        lz_jw_lit(&w, ",\"tokens_saved\":");lz_jw_int(&w, saved);
        lz_jw_lit(&w, ",\"mismatch\":");    lz_jw_int(&w, miss);
        lz_jw_lit(&w, ",\"unsplittable\":");lz_jw_int(&w, unsplit);
        lz_jw_lit(&w, "}}");
        if (!w.err) lz_http_reply(rs, 200, "application/json", w.s, w.len);
        else send_error(rs, 500, "out of memory");
        lz_jw_free(&w);
        return;
    }
    if (strcmp(req->path, "/v1/models") == 0 ||
        strcmp(req->path, "/models") == 0) {
        if (req->method != LZ_HTTP_GET) {
            send_error(rs, 405, "method not allowed");
            return;
        }
        models(c, rs);
        return;
    }
    if (strcmp(req->path, "/v1/chat/completions") == 0 ||
        strcmp(req->path, "/chat/completions") == 0) {
        if (req->method != LZ_HTTP_POST) {
            send_error(rs, 405, "method not allowed (use POST)");
            return;
        }
        completions(c, req, rs);
        return;
    }
    /* /v1/completions is the legacy text endpoint. Refusing by name is
       friendlier than a bare 404: it tells a client that guessed the old
       route which one to use. */
    if (strcmp(req->path, "/v1/completions") == 0) {
        send_error(rs, 404,
                   "legacy /v1/completions is not implemented; "
                   "use /v1/chat/completions");
        return;
    }
    send_error(rs, 404, "unknown route");
}
