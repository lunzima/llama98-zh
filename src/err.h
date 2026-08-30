#ifndef LZ_ERR_H
#define LZ_ERR_H

#include <stdarg.h>

/* Engine error message localization.
 *
 * Every engine error fills the caller's errbuf via error code + arguments;
 * the message text comes from the bilingual EN/ZH tables in err.c and
 * switches at runtime:
 *
 *     lz_set_error_lang(1);   English (default)
 *     lz_set_error_lang(0);   Chinese
 *
 * Callers may also ignore the errbuf text entirely and localize by error
 * code themselves.
 */

typedef enum {
    LZ_ERR_OK = 0,

    /* compat / generic */
    LZ_ERR_OPEN_FILE,          /* cannot open file: %s */
    LZ_ERR_FSEEK,              /* fseek failed: %s */
    LZ_ERR_EMPTY_FILE,         /* empty or invalid file: %s */
    LZ_ERR_ALLOC,              /* out of memory (need %ld bytes: %s) */
    LZ_ERR_READ_SHORT,         /* short read: got %lu / %ld bytes */
    LZ_ERR_NULL_ARG,           /* null argument */
    LZ_ERR_STATE_ALLOC,        /* runtime state allocation failed */
    LZ_ERR_LAYERS_ALLOC,       /* layer array allocation failed */
    LZ_ERR_CLAIM_ALLOC,        /* claim table allocation failed */
    LZ_ERR_SEQ_LEN,            /* seq_len must be positive */

    /* The caller's cont() asked to stop DURING prefill. Its own code
       because the two things a caller does with it are opposite: a
       failure means fall back and try the other path, a cancellation
       means the turn is over. */
    LZ_ERR_CANCELLED,          /* stopped at the caller's request */

    /* long file names (lfn.h). Both carry the WANTED name: reporting
       "TOKENI~1.JSO was not found" to someone looking for
       tokenizer.json sends them after the wrong thing. */
    LZ_ERR_LFN_NOT_FOUND,      /* cannot find %s in %s (long or 8.3) */
    LZ_ERR_LFN_AMBIGUOUS,      /* %s: %d files share one 8.3 short name */

    /* generate */
    LZ_ERR_SAMPLER_INIT,       /* sampler init failed */
    LZ_ERR_PROMPT_BUF,         /* prompt buffer allocation failed */
    LZ_ERR_PROMPT_ENCODE,      /* prompt encode failed */
    LZ_ERR_PROMPT_LONG,        /* prompt too long: %d tokens, limit %d */
    LZ_ERR_STOP_LONG,          /* stop string exceeds %d-byte limit */
    LZ_ERR_FORWARD,            /* forward failed (token/position out of range) */

    /* recurrent-state checkpoint (forward.h LZStateCkpt) */
    LZ_ERR_CKPT_EMPTY,         /* checkpoint has never been saved into */
    LZ_ERR_CKPT_STALE,         /* run state was reset or reallocated since the save */
    LZ_ERR_CKPT_SHAPE,         /* checkpoint does not match this model's shapes */

    /* speculative decoding (MTP draft/verify, generate.c lz_spec_round) */
    LZ_ERR_SPEC_NO_HEAD,       /* spec_k > 0 but the model has no MTP head bound */
    LZ_ERR_SPEC_K_RANGE,       /* spec_k %d out of range 1..%d */
    /* depth-limited lookahead (generate.c lz_look_pick) */
    LZ_ERR_LOOK_W_RANGE,       /* lookahead width %d out of range 1..%d */
    LZ_ERR_LOOK_D_RANGE,       /* lookahead depth %d out of range 1..%d */
    /* No error codes for non-identity penalties or temperature>0 on the
       speculative path: verify applies penalties via an assumed window
       (generate.c), and lz_generate_resume dispatches temperature>0 to
       lz_spec_round_temp. */

    /* sockets (net.h) */
    LZ_ERR_NET_INIT,           /* socket stack init failed */
    LZ_ERR_NET_PORT,           /* invalid port %d */
    LZ_ERR_NET_SOCKET,         /* socket() failed (%d) */
    LZ_ERR_NET_BIND,           /* cannot bind 127.0.0.1:%d (%d) */
    LZ_ERR_NET_LISTEN,         /* listen() failed (%d) */
    LZ_ERR_NET_ACCEPT,         /* accept() failed (%d) */

    /* json */
    LZ_ERR_JSON_ALLOC,         /* JSON buffer allocation failed (%lu bytes) */
    LZ_ERR_JSON_ROOT,          /* JSON root parse failed */
    LZ_ERR_JSON_TRAIL,         /* trailing content after JSON root */

    /* model config */
    LZ_ERR_CFG_ROOT,           /* config.json root is not an object */
    LZ_ERR_CFG_LAYER_TYPES,    /* config missing layer_types */
    LZ_ERR_CFG_LAYER_LEN,      /* layer_types length %d != num_hidden_layers %d */
    LZ_ERR_CFG_LAYER_RANGE,    /* layer count %d out of range 1..%d */
    LZ_ERR_CFG_LAYER_TYPE,     /* unrecognized layer type at %d: %s */
    LZ_ERR_CFG_FIELDS,         /* config key fields missing or invalid */
    LZ_ERR_CFG_HEADS,          /* invalid attention heads: %d / %d */
    LZ_ERR_CFG_ROTARY,         /* invalid rotary_dim=%d (head_dim=%d, factor=%d%%) */
    LZ_ERR_CFG_ROPE_SCALING,   /* unsupported rope_scaling type: %s */
    LZ_ERR_CFG_KDA_ACT,        /* kda_gate_activation must be silu, got %s */
    LZ_ERR_CFG_MOE_FIELDS,     /* invalid MoE fields: num_experts=%d top_k=%d first_k_dense_replace=%d */
    LZ_ERR_CFG_HADAMARD,       /* %s=%d is not a power of two dividing %s=%d */

    /* model tensors */
    LZ_ERR_TENSOR_MISSING,     /* missing tensor %s */
    LZ_ERR_TENSOR_DIMS,        /* tensor %s dim mismatch: want %d, got %d */
    LZ_ERR_TENSOR_SHAPE,       /* tensor %s dim %d mismatch: want %lld, got %lld */
    LZ_ERR_PREFIX,             /* embed_tokens.weight not found (prefix?) */
    LZ_ERR_UNCLAIMED,          /* tensor %s unclaimed; arch gap? */
    LZ_ERR_MTP_LAYERS,         /* mtp_num_hidden_layers %d unsupported */
    LZ_ERR_TENSOR_ALLOC,       /* tensor %s allocation failed */
    LZ_ERR_TENSOR_ALLOC_BYTES, /* tensor %s allocation failed (need %lld bytes) */
    LZ_ERR_TENSOR_READ,        /* tensor %s read failed (truncated) */
    LZ_ERR_TENSOR_ELEMS,       /* tensor %s elem count mismatch: want %lld, file %u */
    LZ_ERR_TENSOR_GS,          /* tensor %s invalid group size */
    LZ_ERR_TENSOR_DTYPE,       /* tensor %s invalid dtype %u */
    LZ_ERR_NOT_OPEN,           /* model not open */

    /* model.bin (L98Z container) */
    LZ_ERR_BIN_OPEN,           /* cannot open %s */
    LZ_ERR_BIN_CFG_READ,       /* config read failed (truncated) */
    LZ_ERR_BIN_MAGIC,          /* not a model.bin (bad magic) */
    LZ_ERR_BIN_VERSION,        /* unsupported model.bin version %u */
    LZ_ERR_BIN_PAD,            /* nonzero layer_types padding */
    LZ_ERR_BIN_CFG,            /* model.bin config invalid */

    /* tokenizer */
    LZ_ERR_TK_ROOT,            /* tokenizer.json root is not an object */
    LZ_ERR_TK_MODEL,           /* missing model object */
    LZ_ERR_TK_TYPE,            /* unsupported tokenizer type (need BPE) */
    LZ_ERR_TK_VOCAB,           /* missing model.vocab */
    LZ_ERR_TK_MERGES,          /* missing model.merges */
    LZ_ERR_TK_ID_RANGE,        /* vocab id %d out of range */
    LZ_ERR_TK_ID_DUP,          /* duplicate vocab id %d */
    LZ_ERR_TK_WORD,            /* word %d cannot be restored to bytes */
    LZ_ERR_TK_ID_MISSING,      /* vocab missing id %d */
    LZ_ERR_TK_MERGE_PAIR,      /* merges[%d] is not a string pair */
    LZ_ERR_TK_MERGE_FULL,      /* merges hash table full */
    LZ_ERR_TK_MERGE_LONG,      /* merges[%d] side exceeds %d codepoints */

    /* safetensors */
    LZ_ERR_ST_NOT_FOUND,
    LZ_ERR_ST_OPEN,
    LZ_ERR_ST_HEADER_LEN,
    LZ_ERR_ST_HEADER_BAD,
    LZ_ERR_ST_HEADER_ALLOC,
    LZ_ERR_ST_HEADER_READ,
    LZ_ERR_ST_HEADER_JSON,
    LZ_ERR_ST_NO_TENSORS,
    LZ_ERR_ST_TENSOR_ALLOC,
    LZ_ERR_ST_DESC,
    LZ_ERR_ST_SHAPE,
    LZ_ERR_ST_SHAPE_MISSING,
    LZ_ERR_ST_DIMS,
    LZ_ERR_ST_DTYPE,
    LZ_ERR_ST_DTYPE_F32,
    LZ_ERR_ST_OFFSET,
    LZ_ERR_ST_OFFSET_RANGE,
    LZ_ERR_ST_SEEK,
    LZ_ERR_ST_OVERFLOW,
    LZ_ERR_ST_ELEMS,
    LZ_ERR_ST_NBYTES,         /* tensor %s byte count vs shape mismatch */
    LZ_ERR_ST_DATA_SIZE,      /* data section size does not match file */
    LZ_ERR_ST_READ,
    LZ_ERR_ST_INDEX_SHARDS,   /* index.json names more than one shard */
    LZ_ERR_ST_NULL,
    LZ_ERR_OOM,              /* out of memory */
    LZ_ERR_ST_NULL2,

    /* json internal parse errors */
    LZ_ERR_JSON_PARSE,         /* JSON parse failed at offset %ld: %s */
    LZ_ERR_JSON_BAD_ESC,       /* bad \\u escape */
    LZ_ERR_JSON_ESC_TRUNC,     /* string truncated after escape */
    LZ_ERR_JSON_STR_UNTERM,    /* unterminated string */
    LZ_ERR_JSON_COLON,         /* object member missing colon */
    LZ_ERR_JSON_OBJ_END,       /* object missing comma or right brace */
    LZ_ERR_JSON_DEPTH,         /* nesting too deep */
    LZ_ERR_JSON_NUM,           /* bad number format */
    LZ_ERR_JSON_ARR_END,       /* array missing comma or right bracket */
    LZ_ERR_JSON_VALUE,         /* unrecognized value */
    LZ_ERR_JSON_EXPECT_STR,    /* expected string */
    LZ_ERR_JSON_ESC,           /* unknown escape character */
    LZ_ERR_JSON_NODE_ALLOC,    /* node array allocation failed */
    LZ_ERR_JSON_EOF,           /* unexpected end of input */
    LZ_ERR_JSON_LITERAL,       /* invalid literal */

    /* chat template */
    LZ_ERR_OUT_BUF,            /* output buffer is null */
    LZ_ERR_NO_MESSAGES,        /* no messages */
    LZ_ERR_NO_USER,            /* no user turn in messages */
    LZ_ERR_SYSTEM_FIRST,       /* system message must come first */
    LZ_ERR_ROLE,               /* unknown message role: %d */
    LZ_ERR_RENDER_ALLOC,       /* render buffer allocation failed */
    LZ_ERR_TRUNC,              /* output buffer too small: need %d bytes */
    LZ_ERR_HIST_FULL,          /* conversation history full (%d turns) */
    LZ_ERR_HIST_ALLOC,         /* history turn allocation failed */

    /* CPU floor. The deliverable's real hardware floor is Socket 7, and
       Socket 7 is NOT "has MMX": the original Pentium P54C, the AMD K5
       and the non-MX Cyrix 6x86 are all Socket 7 and have none. So MMX,
       SSE, SSE2 and 3DNow are optional and CPUID-dispatched, and only
       two things are actually required - the CPUID instruction itself
       and a floating-point unit. These say which one is missing,
       because "unsupported CPU" tells the person holding the machine
       nothing they can act on. */
    LZ_ERR_CPU_NO_CPUID,       /* no CPUID instruction */
    LZ_ERR_CPU_NO_FPU,         /* no floating-point unit */

    /* The build was compiled for the wrong byte order for the machine it
       is running on - see lz_endian_ok in compat.h. Its own error rather
       than a generic one because the fix is a build flag
       (-DLZ_BIG_ENDIAN=0/1), not anything about the model file the user
       was pointing at when it fired. */
    LZ_ERR_ENDIAN_MISMATCH,    /* built for the wrong byte order */

    /* Generic failure. Two jobs, and it must stay LAST before
       LZ_ERR_COUNT for the second one:
       1. the initial value of a return code that every failure path is
          supposed to overwrite - so a path that forgets reports
          "internal error" instead of inheriting whatever code happens to
          sit at 1 ("cannot open file"), which would send a caller
          mapping codes onto HTTP statuses down entirely the wrong road;
       2. lz_err_fmt clamps an out-of-range code to LZ_ERR_COUNT - 1, so
          being last makes that clamp say "internal error" rather than
          the unrelated message of whichever entry ended up final. */
    LZ_ERR_INTERNAL,           /* internal error */

    LZ_ERR_COUNT
} LZErr;

/* Language: 1 = English (default), 0 = Chinese */
void lz_set_error_lang(int english);
int  lz_error_lang(void);

/* Format error code + args into buf in the current language. Internal
   use; callers doing their own localization may ignore the text and only
   check the return value / error code. */
/* Get the message template for an error code (current language). Used to
   embed a sub-error's text into an outer message, e.g. the JSON parser
   stuffing a concrete syntax error into LZ_ERR_JSON_PARSE's %s.
   Only meaningful for codes WITHOUT format placeholders. */
const char *lz_err_text(LZErr code);

int  lz_err_fmt(char *buf, int len, LZErr code, ...);
int  lz_err_fmt_v(char *buf, int len, LZErr code, va_list ap);

/* Set a return code and fill the caller's errbuf in ONE step.
 *
 * These exist because the two halves must not be able to drift apart.
 * An HTTP front end maps the RETURN CODE onto a status - LZ_ERR_PROMPT_LONG
 * is the client's fault (400), LZ_ERR_ALLOC is ours (500) - while errbuf
 * carries the human-readable half into the response body. A failure path
 * that set one and not the other would answer with a status contradicting
 * its own message, and review would not catch it: each half looks
 * plausible on its own.
 *
 * `code` is evaluated more than once, so do not pass an expression with
 * side effects. Everything is spelled out rather than captured implicitly
 * so these read the same in any function, in any file.
 *
 * Three arities instead of one variadic macro: there is no portable way
 * to spell an EMPTY __VA_ARGS__, and `, ##__VA_ARGS__` is a GNU extension
 * Watcom does not have, and the Watcom build is not optional.
 */
#define LZ_ERR_SET(rcvar, buf, len, code) \
    do { (rcvar) = (code); \
         if (buf) lz_err_fmt((buf), (len), (code)); } while (0)
#define LZ_ERR_SET1(rcvar, buf, len, code, a) \
    do { (rcvar) = (code); \
         if (buf) lz_err_fmt((buf), (len), (code), (a)); } while (0)
#define LZ_ERR_SET2(rcvar, buf, len, code, a, b) \
    do { (rcvar) = (code); \
         if (buf) lz_err_fmt((buf), (len), (code), (a), (b)); } while (0)

#endif
