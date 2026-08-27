#ifndef LZ_JSON_H
#define LZ_JSON_H

#include <stddef.h>

/* Minimal JSON DOM parser.
   Why it exists: safetensors headers and Qwen3.5 config.json are both JSON,
   and M4.5 requires zero Python at engine side, so we parse it in C (spec 13.2).

   Design constraints:
   - Nodes link by index, not pointer: the nodes array reallocs;
   - Strings are unescaped in place (with NUL terminator) on a writable copy of
     the source text. Unescaping only shrinks, so it always fits;
   - Recursion depth is capped so malformed input cannot blow the stack. */

typedef enum {
    LZ_JSON_NULL = 0,
    LZ_JSON_BOOL,
    LZ_JSON_NUM,
    LZ_JSON_STR,
    LZ_JSON_ARR,
    LZ_JSON_OBJ
} LZJsonType;

typedef struct {
    int type;                /* LZJsonType */
    const char *key;         /* member name; non-NULL only under OBJ parent (NUL-terminated) */
    int key_len;
    const char *text;        /* STR content (NUL-terminated) */
    int text_len;
    float num;               /* NUM value; BOOL uses 0/1 */
    int first_child;         /* index of first child of ARR/OBJ, -1 if none */
    int next_sibling;
    int n_children;
} LZJsonNode;

typedef struct {
    LZJsonNode *nodes;
    int n_nodes;
    int cap;
    char *buf;               /* writable copy of source text; strings unescaped in place */
} LZJson;

/* return 0 on success, non-zero on failure and fill errbuf. src is not modified. */
int  lz_json_parse(LZJson *j, const char *src, size_t len,
                   char *errbuf, int errlen);
void lz_json_free(LZJson *j);

const LZJsonNode *lz_json_root(const LZJson *j);
/* look up a member by name in OBJ; NULL if obj is NULL / not OBJ */
const LZJsonNode *lz_json_get(const LZJson *j, const LZJsonNode *obj,
                              const char *key);
/* element idx of ARR */
const LZJsonNode *lz_json_at(const LZJson *j, const LZJsonNode *arr, int idx);
/* iterate children of ARR/OBJ */
const LZJsonNode *lz_json_first(const LZJson *j, const LZJsonNode *n);
const LZJsonNode *lz_json_next(const LZJson *j, const LZJsonNode *n);

/* convenience getter: return def when the key is absent or the type mismatches.
   The caller must pass the default explicitly, so "got 0" and "not present" never conflate. */
int    lz_json_get_int(const LZJson *j, const LZJsonNode *obj,
                       const char *key, int def);
float lz_json_get_num(const LZJson *j, const LZJsonNode *obj,
                       const char *key, float def);
/* Accepts BOOL and NUM alike. Python writes config flags as `true`, so
   reading one with lz_json_get_int silently yields the default - the
   flag is off, the model loads, and only the tensors it gates go
   missing. */
int    lz_json_get_bool(const LZJson *j, const LZJsonNode *obj,
                        const char *key, int def);
/* return STR content if present, else def */
const char *lz_json_get_str(const LZJson *j, const LZJsonNode *obj,
                            const char *key, const char *def);
/* whether the node is a string whose content equals s */
int lz_json_str_eq(const LZJsonNode *n, const char *s);

/* ------------------------------------------------------------ writing

   A growable buffer plus the two things worth centralising: correct
   string escaping, and integers. Not a general serializer - the
   endpoint's JSON shapes are fixed, so hand-assembling them is less code
   than a document builder and easier to gate.

   NO FLOAT OUTPUT, on purpose. Nothing in an OpenAI chat completion
   needs one (`created` is an integer and logprobs are unsupported), and
   printf-ing a float is barred anywhere the x87 sits in its PC=24
   region. Not offering the function is cheaper than
   remembering where calling it would be safe. */
typedef struct {
    char *s;
    int   len, cap;
    int   err;      /* sticky: set on allocation failure, checked once at the end */
} LZJsonW;

void lz_jw_init(LZJsonW *w);
void lz_jw_free(LZJsonW *w);
/* Append bytes verbatim - structural characters, already-quoted keys. */
int  lz_jw_raw(LZJsonW *w, const char *s, int n);
int  lz_jw_lit(LZJsonW *w, const char *s);           /* strlen of the above */
/* Append a JSON string INCLUDING its surrounding quotes, escaping the
   quote, the backslash, and every control character below 0x20. UTF-8
   passes through unchanged: it is already valid JSON, and escaping it
   would triple the size of Chinese output for nothing. n < 0 = strlen.

   Centralised because generated text routinely contains the two
   characters that break hand-built JSON - a quote and a newline - and a
   response that fails to parse looks like a model problem, not an
   encoding one. */
int  lz_jw_str(LZJsonW *w, const char *s, int n);
int  lz_jw_int(LZJsonW *w, long v);

#endif
