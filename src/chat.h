#ifndef CHAT_H
#define CHAT_H

/* Qwen3.5 chat template rendering.
   Byte-for-byte replica of data/template/qwen3_5_unsloth.jinja; the training
   side assembles the same shape, and the
   two must agree - a mismatch never errors, it just makes the model behave
   oddly from the second turn on.

   tools / vision / tool_calls are unsupported: this project dropped the
   vision tower, and a model pruned to 64.5M has no tool-calling ability.
   Passing them in errors out rather than being silently ignored. */

typedef enum {
    LZ_ROLE_SYSTEM = 0,
    LZ_ROLE_USER,
    LZ_ROLE_ASSISTANT
} LZRole;

typedef struct {
    LZRole      role;
    const char *content;    /* UTF-8 bytes, not owned */
    int         len;        /* < 0 means length via NUL terminator */
} LZChatMsg;

/* Growable byte buffer. s is always NUL-terminated (handy as a C string),
   but len is authoritative - embedded NULs are not legal input here. */
typedef struct {
    char *s;
    int   len;
    int   cap;
} LZChatBuf;

void lz_chat_buf_init(LZChatBuf *b);
void lz_chat_buf_free(LZChatBuf *b);

/* The generation-prompt segment lz_chat_render appends after
   "<|im_start|>assistant\n". Single source of truth: the renderer itself
   calls this. A caller reusing a conversation prefix across turns needs
   the same bytes to locate where the reusable part ends - this segment
   is precisely what turn N's prompt has and turn N+1's render does not
   (measured: 4 tokens, and the reason turn 2 shares 36 of 40 rather than
   all 40). Returns a static string; do not free. */
const char *lz_chat_gen_prompt_tail(int enable_thinking);

/* The default system message lz_chat_render injects when a caller
   supplies no system message (Item 9b - see src/chat.c's own comment
   on why it exists, and what its
   absence costs).

   Returns a static string; do not free.

   WHY THIS GETTER EXISTS: a front end that wants to SHOW the default
   (the GUI's "restore defaults" needs to display what it will restore
   to) must not re-type the text. Two copies of a trained constant is
   two authorities, and they would drift - and a drifted copy is not a
   cosmetic difference, it is a paraphrased prompt reaching a model that
   was only ever trained on the exact bytes. This getter is the single
   source of truth; src/chat.c's own renderer and every front end read
   the same bytes from it. */
const char *lz_chat_default_system(void);

/* Whether the segment lz_chat_gen_prompt_tail returns leaves the model
   INSIDE a <think> block - derived by counting the actual tags in that
   string, not a second enable_thinking check (see lz_chat_gen_prompt_
   tail's own comment on why a second copy of that decision drifts
   silently). A caller seeding a display parser's starting state (a
   GUI's think/plain colouring at the start of a fresh reply) should
   call this instead of re-deciding from enable_thinking itself. */
int lz_chat_gen_prompt_starts_in_think(int enable_thinking);

/* Render a full conversation. Errors go to errbuf, never printf, never
   exit.

   RETURN: 0 (LZ_ERR_OK) on success, otherwise the LZErr code (err.h), so
   a caller can act on it without parsing the bilingual errbuf text.
   LZ_ERR_NO_MESSAGES / NO_USER / SYSTEM_FIRST / ROLE all mean the caller
   handed over a malformed conversation (HTTP 400); LZ_ERR_RENDER_ALLOC
   and LZ_ERR_OUT_BUF are ours (500).

   msgs[i].content is BORROWED for the duration of the call only - the
   bytes are copied into `out`, and nothing is retained afterwards. For a
   chat loop that is not enough; see LZChatHist below for why.

   add_generation_prompt  append the assistant start for generation.
   enable_thinking        affects only the generation prompt segment:
                            true  -> "<think>
"          (model writes reasoning)
                            false -> "<think>

</think>

" (empty block, answer directly)
                          No effect on ALREADY-COMPLETED turns - counter
                          to intuition, in the template enable_thinking
                          appears only in the generation-prompt branch.

   History-turn assistant messages carry no think block, and any reasoning
   they contain is stripped ENTIRELY; only assistant messages after the
   last user turn render a think block. */
int lz_chat_render(const LZChatMsg *msgs, int n_msgs,
                   int add_generation_prompt, int enable_thinking,
                   LZChatBuf *out, char *errbuf, int errlen);

/* Normalize a raw assistant reply (sink bytes) into history-turn form:
   reasoning dropped entirely (split_think rule), content lstrip'd.
   Returns bytes written (NUL not counted); -1 on error (errbuf set:
   LZ_ERR_NULL_ARG for bad args, LZ_ERR_TRUNC if outcap too small -
   needed size reported through errbuf).  raw_len < 0 => strlen. */
int lz_chat_norm_history(const char *raw, int raw_len,
                         char *out, int outcap,
                         char *errbuf, int errlen);

/* Conversation history that OWNS its message bytes.

   LZChatMsg deliberately does not own `content` - it is a view, which is
   right for a caller that renders a fixed array of literals. It is wrong
   for a chat loop, and getting it wrong is not obvious: the first
   version of the interactive CLI stored a pointer to the reusable input
   buffer for user turns, and for assistant turns stored a heap pointer
   it freed on the very next line. Both render correctly on turn one and
   silently wrong from turn two, because lz_chat_render only reads the
   array at the START of the following turn - by which time the stack
   buffer holds the newest input and the heap block is freed.

   So the loop keeps its history here instead, and every push copies. */

#define LZ_CHAT_HIST_MAX 128

typedef struct {
    LZChatMsg msgs[LZ_CHAT_HIST_MAX];
    char     *owned[LZ_CHAT_HIST_MAX];   /* one allocation per message */
    int       n;
} LZChatHist;

void lz_chat_hist_init(LZChatHist *h);
void lz_chat_hist_reset(LZChatHist *h);      /* frees, keeps the struct */
void lz_chat_hist_free(LZChatHist *h);

/* Append a copy of `content`. len < 0 means NUL-terminated. Returns 0 on
   success; -1 with errbuf set when the history is full or out of memory. */
int lz_chat_hist_push(LZChatHist *h, LZRole role, const char *content,
                      int len, char *errbuf, int errlen);

/* Drop the newest message, freeing its copy. Returns 0, or -1 when the
   history is already empty.

   NO errbuf, unlike push, and that is deliberate rather than an
   oversight: "there is nothing to pop" is a question the caller already
   knows the answer to (it can read h->n) and acts on rather than
   reports - the front end greys its Regenerate command instead of
   showing a message. push can fail for reasons the caller could not
   have predicted (out of memory, history full), which is what an
   errbuf is for.

   Exists for the front end's rollback commands - regenerate, edit the
   last turn, delete the last turn - which all need the same "take the
   conversation back one message" step. */
int lz_chat_hist_pop(LZChatHist *h);

#endif
