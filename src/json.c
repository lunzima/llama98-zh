#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "err.h"

#define LZ_JSON_MAX_DEPTH 64

typedef struct {
    char *p;
    char *end;
    LZJson *j;
    char *errbuf;
    int errlen;
    int depth;
} LZJsonParser;

static void jerr(LZJsonParser *ps, LZErr code) {
    if (ps->errbuf && ps->errlen > 0 && ps->errbuf[0] == '\0') {
        /* record only the first error: later ones are usually knock-on
           effects, overwriting would mislead */
        long off = (long)(ps->p - ps->j->buf);
        /* LZ_ERR_JSON_PARSE's second placeholder is %s: it wants message
           text, not an error code. Passing code (an int) straight to
           vsnprintf dereferences it as a pointer - any malformed JSON
           would segfault. */
        lz_err_fmt(ps->errbuf, ps->errlen, LZ_ERR_JSON_PARSE, off,
                   lz_err_text(code));
    }
}

static int jnode_new(LZJsonParser *ps, int type) {
    LZJson *j = ps->j;
    if (j->n_nodes == j->cap) {
        int ncap = j->cap ? j->cap * 2 : 64;
        LZJsonNode *nn = (LZJsonNode *)realloc(
            j->nodes, (size_t)ncap * sizeof(LZJsonNode));
        if (!nn) {
            jerr(ps, LZ_ERR_JSON_NODE_ALLOC);
            return -1;
        }
        j->nodes = nn;
        j->cap = ncap;
    }
    {
        LZJsonNode *n = &j->nodes[j->n_nodes];
        memset(n, 0, sizeof(*n));
        n->type = type;
        n->first_child = -1;
        n->next_sibling = -1;
    }
    return j->n_nodes++;
}

static void jskip_ws(LZJsonParser *ps) {
    while (ps->p < ps->end) {
        char c = *ps->p;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ps->p++;
        else break;
    }
}

/* Write codepoint as UTF-8 into dst; returns bytes written */
static int jutf8_put(char *dst, unsigned long cp) {
    if (cp < 0x80UL) {
        dst[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800UL) {
        dst[0] = (char)(0xC0UL | (cp >> 6));
        dst[1] = (char)(0x80UL | (cp & 0x3FUL));
        return 2;
    }
    if (cp < 0x10000UL) {
        dst[0] = (char)(0xE0UL | (cp >> 12));
        dst[1] = (char)(0x80UL | ((cp >> 6) & 0x3FUL));
        dst[2] = (char)(0x80UL | (cp & 0x3FUL));
        return 3;
    }
    dst[0] = (char)(0xF0UL | (cp >> 18));
    dst[1] = (char)(0x80UL | ((cp >> 12) & 0x3FUL));
    dst[2] = (char)(0x80UL | ((cp >> 6) & 0x3FUL));
    dst[3] = (char)(0x80UL | (cp & 0x3FUL));
    return 4;
}

static int jhex4(const char *s, unsigned long *out) {
    int i;
    unsigned long v = 0;
    for (i = 0; i < 4; i++) {
        char c = s[i];
        v <<= 4;
        if (c >= '0' && c <= '9')      v |= (unsigned long)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned long)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned long)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

/* Parse a string literal, unescaping in place: read pointer p never
   falls behind write pointer w because every escape sequence's source
   length is >= its result length. On success *out_text points at the
   content start (NUL-terminated) and p stops after the closing quote. */
static int jparse_string(LZJsonParser *ps, const char **out_text, int *out_len) {
    char *w;
    char *start;

    if (ps->p >= ps->end || *ps->p != '"') {
        jerr(ps, LZ_ERR_JSON_EXPECT_STR);
        return -1;
    }
    ps->p++;
    start = ps->p;
    w = ps->p;

    while (ps->p < ps->end && *ps->p != '"') {
        if (*ps->p != '\\') {
            *w++ = *ps->p++;
            continue;
        }
        ps->p++;
        if (ps->p >= ps->end) {
            jerr(ps, LZ_ERR_JSON_ESC_TRUNC);
            return -1;
        }
        switch (*ps->p) {
        case '"':  *w++ = '"';  ps->p++; break;
        case '\\': *w++ = '\\'; ps->p++; break;
        case '/':  *w++ = '/';  ps->p++; break;
        case 'b':  *w++ = '\b'; ps->p++; break;
        case 'f':  *w++ = '\f'; ps->p++; break;
        case 'n':  *w++ = '\n'; ps->p++; break;
        case 'r':  *w++ = '\r'; ps->p++; break;
        case 't':  *w++ = '\t'; ps->p++; break;
        case 'u': {
            unsigned long cp;
            ps->p++;
            if (ps->end - ps->p < 4 || !jhex4(ps->p, &cp)) {
                jerr(ps, LZ_ERR_JSON_BAD_ESC);
                return -1;
            }
            ps->p += 4;
            /* high surrogate: needs a following low surrogate to recover the codepoint */
            if (cp >= 0xD800UL && cp <= 0xDBFFUL &&
                ps->end - ps->p >= 6 && ps->p[0] == '\\' && ps->p[1] == 'u') {
                unsigned long lo;
                if (jhex4(ps->p + 2, &lo) && lo >= 0xDC00UL && lo <= 0xDFFFUL) {
                    cp = 0x10000UL + ((cp - 0xD800UL) << 10) + (lo - 0xDC00UL);
                    ps->p += 6;
                }
            }
            w += jutf8_put(w, cp);
            break;
        }
        default:
            jerr(ps, LZ_ERR_JSON_ESC);
            return -1;
        }
    }
    if (ps->p >= ps->end) {
        jerr(ps, LZ_ERR_JSON_STR_UNTERM);
        return -1;
    }
    ps->p++;                    /* skip closing quote */
    *out_text = start;
    *out_len = (int)(w - start);
    *w = '\0';                  /* the closing quote's spot; always writable */
    return 0;
}

static int jparse_value(LZJsonParser *ps);

static int jparse_object(LZJsonParser *ps) {
    int self = jnode_new(ps, LZ_JSON_OBJ);
    int last = -1;
    if (self < 0) return -1;
    ps->p++;                    /* '{' */

    jskip_ws(ps);
    if (ps->p < ps->end && *ps->p == '}') {
        ps->p++;
        return self;
    }
    for (;;) {
        const char *k;
        int klen, child;

        jskip_ws(ps);
        if (jparse_string(ps, &k, &klen) != 0) return -1;
        jskip_ws(ps);
        if (ps->p >= ps->end || *ps->p != ':') {
            jerr(ps, LZ_ERR_JSON_COLON);
            return -1;
        }
        ps->p++;

        child = jparse_value(ps);
        if (child < 0) return -1;

        ps->j->nodes[child].key = k;
        ps->j->nodes[child].key_len = klen;
        if (last < 0) ps->j->nodes[self].first_child = child;
        else          ps->j->nodes[last].next_sibling = child;
        last = child;
        ps->j->nodes[self].n_children++;

        jskip_ws(ps);
        if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
        if (ps->p < ps->end && *ps->p == '}') { ps->p++; return self; }
        jerr(ps, LZ_ERR_JSON_OBJ_END);
        return -1;
    }
}

static int jparse_array(LZJsonParser *ps) {
    int self = jnode_new(ps, LZ_JSON_ARR);
    int last = -1;
    if (self < 0) return -1;
    ps->p++;                    /* '[' */

    jskip_ws(ps);
    if (ps->p < ps->end && *ps->p == ']') {
        ps->p++;
        return self;
    }
    for (;;) {
        int child = jparse_value(ps);
        if (child < 0) return -1;

        if (last < 0) ps->j->nodes[self].first_child = child;
        else          ps->j->nodes[last].next_sibling = child;
        last = child;
        ps->j->nodes[self].n_children++;

        jskip_ws(ps);
        if (ps->p < ps->end && *ps->p == ',') { ps->p++; continue; }
        if (ps->p < ps->end && *ps->p == ']') { ps->p++; return self; }
        jerr(ps, LZ_ERR_JSON_ARR_END);
        return -1;
    }
}

static int jparse_value(LZJsonParser *ps) {
    int r;

    if (ps->depth >= LZ_JSON_MAX_DEPTH) {
        jerr(ps, LZ_ERR_JSON_DEPTH);
        return -1;
    }
    jskip_ws(ps);
    if (ps->p >= ps->end) {
        jerr(ps, LZ_ERR_JSON_EOF);
        return -1;
    }

    ps->depth++;
    switch (*ps->p) {
    case '{':
        r = jparse_object(ps);
        break;
    case '[':
        r = jparse_array(ps);
        break;
    case '"': {
        const char *t;
        int tlen;
        r = jnode_new(ps, LZ_JSON_STR);
        if (r >= 0) {
            if (jparse_string(ps, &t, &tlen) != 0) r = -1;
            else {
                ps->j->nodes[r].text = t;
                ps->j->nodes[r].text_len = tlen;
            }
        }
        break;
    }
    case 't':
        if (ps->end - ps->p >= 4 && memcmp(ps->p, "true", 4) == 0) {
            r = jnode_new(ps, LZ_JSON_BOOL);
            if (r >= 0) ps->j->nodes[r].num = 1.0;
            ps->p += 4;
        } else { jerr(ps, LZ_ERR_JSON_LITERAL); r = -1; }
        break;
    case 'f':
        if (ps->end - ps->p >= 5 && memcmp(ps->p, "false", 5) == 0) {
            r = jnode_new(ps, LZ_JSON_BOOL);
            if (r >= 0) ps->j->nodes[r].num = 0.0;
            ps->p += 5;
        } else { jerr(ps, LZ_ERR_JSON_LITERAL); r = -1; }
        break;
    case 'n':
        if (ps->end - ps->p >= 4 && memcmp(ps->p, "null", 4) == 0) {
            r = jnode_new(ps, LZ_JSON_NULL);
            ps->p += 4;
        } else { jerr(ps, LZ_ERR_JSON_LITERAL); r = -1; }
        break;
    default: {
        char *endp = NULL;
        double v;
        if (*ps->p != '-' && (*ps->p < '0' || *ps->p > '9')) {
            jerr(ps, LZ_ERR_JSON_VALUE);
            ps->depth--;
            return -1;
        }
        /* buf is NUL-terminated overall; strtod cannot overrun */
        v = strtod(ps->p, &endp);
        if (endp == ps->p) {
            jerr(ps, LZ_ERR_JSON_NUM);
            ps->depth--;
            return -1;
        }
        r = jnode_new(ps, LZ_JSON_NUM);
        if (r >= 0) ps->j->nodes[r].num = v;
        ps->p = endp;
        break;
    }
    }
    ps->depth--;
    return r;
}

int lz_json_parse(LZJson *j, const char *src, size_t len,
                  char *errbuf, int errlen) {
    LZJsonParser ps;
    int root;

    memset(j, 0, sizeof(*j));
    if (errbuf && errlen > 0) errbuf[0] = '\0';

    j->buf = (char *)malloc(len + 1);
    if (!j->buf) {
        if (errbuf && errlen > 0)
            lz_err_fmt(errbuf, errlen, LZ_ERR_JSON_ALLOC,
                     (unsigned long)(len + 1));
        return 1;
    }
    memcpy(j->buf, src, len);
    j->buf[len] = '\0';

    ps.p = j->buf;
    ps.end = j->buf + len;
    ps.j = j;
    ps.errbuf = errbuf;
    ps.errlen = errlen;
    ps.depth = 0;

    root = jparse_value(&ps);
    if (root < 0) {
        lz_json_free(j);
        if (errbuf && errlen > 0 && errbuf[0] == '\0')
            lz_err_fmt(errbuf, errlen, LZ_ERR_JSON_ROOT);
        return 1;
    }
    jskip_ws(&ps);
    if (ps.p != ps.end) {
        if (errbuf && errlen > 0)
            lz_err_fmt(errbuf, errlen, LZ_ERR_JSON_TRAIL);
        lz_json_free(j);
        return 1;
    }
    return 0;
}

void lz_json_free(LZJson *j) {
    if (!j) return;
    free(j->nodes);
    free(j->buf);
    memset(j, 0, sizeof(*j));
}

const LZJsonNode *lz_json_root(const LZJson *j) {
    if (!j || j->n_nodes <= 0) return NULL;
    return &j->nodes[0];
}

const LZJsonNode *lz_json_first(const LZJson *j, const LZJsonNode *n) {
    if (!j || !n || n->first_child < 0) return NULL;
    return &j->nodes[n->first_child];
}

const LZJsonNode *lz_json_next(const LZJson *j, const LZJsonNode *n) {
    if (!j || !n || n->next_sibling < 0) return NULL;
    return &j->nodes[n->next_sibling];
}

const LZJsonNode *lz_json_get(const LZJson *j, const LZJsonNode *obj,
                              const char *key) {
    const LZJsonNode *c;
    if (!j || !obj || obj->type != LZ_JSON_OBJ || !key) return NULL;
    for (c = lz_json_first(j, obj); c; c = lz_json_next(j, c)) {
        if (c->key && strcmp(c->key, key) == 0) return c;
    }
    return NULL;
}

const LZJsonNode *lz_json_at(const LZJson *j, const LZJsonNode *arr, int idx) {
    const LZJsonNode *c;
    int i = 0;
    if (!j || !arr || arr->type != LZ_JSON_ARR || idx < 0) return NULL;
    for (c = lz_json_first(j, arr); c; c = lz_json_next(j, c), i++) {
        if (i == idx) return c;
    }
    return NULL;
}

int lz_json_get_int(const LZJson *j, const LZJsonNode *obj,
                    const char *key, int def) {
    const LZJsonNode *n = lz_json_get(j, obj, key);
    if (!n || n->type != LZ_JSON_NUM) return def;
    return (int)n->num;
}

double lz_json_get_num(const LZJson *j, const LZJsonNode *obj,
                       const char *key, double def) {
    const LZJsonNode *n = lz_json_get(j, obj, key);
    if (!n || n->type != LZ_JSON_NUM) return def;
    return n->num;
}

const char *lz_json_get_str(const LZJson *j, const LZJsonNode *obj,
                            const char *key, const char *def) {
    const LZJsonNode *n = lz_json_get(j, obj, key);
    if (!n || n->type != LZ_JSON_STR) return def;
    return n->text;
}

int lz_json_str_eq(const LZJsonNode *n, const char *s) {
    if (!n || n->type != LZ_JSON_STR || !n->text || !s) return 0;
    return strcmp(n->text, s) == 0;
}

/* ------------------------------------------------------------ writing */

void lz_jw_init(LZJsonW *w) {
    if (!w) return;
    w->s = NULL;
    w->len = 0;
    w->cap = 0;
    w->err = 0;
}

void lz_jw_free(LZJsonW *w) {
    if (!w) return;
    free(w->s);
    lz_jw_init(w);
}

static int jw_grow(LZJsonW *w, int need) {
    int cap;
    char *p;
    if (w->err) return 0;
    if (w->cap - w->len > need) return 1;
    cap = w->cap ? w->cap : 256;
    while (cap - w->len <= need) {
        if (cap > (1 << 28)) { w->err = 1; return 0; }
        cap *= 2;
    }
    p = (char *)realloc(w->s, (size_t)cap);
    if (!p) { w->err = 1; return 0; }
    w->s = p;
    w->cap = cap;
    return 1;
}

int lz_jw_raw(LZJsonW *w, const char *s, int n) {
    if (!w || !s || n <= 0) return w && !w->err ? 0 : -1;
    if (!jw_grow(w, n + 1)) return -1;
    memcpy(w->s + w->len, s, (size_t)n);
    w->len += n;
    w->s[w->len] = 0;
    return 0;
}

int lz_jw_lit(LZJsonW *w, const char *s) {
    return lz_jw_raw(w, s, s ? (int)strlen(s) : 0);
}

int lz_jw_str(LZJsonW *w, const char *s, int n) {
    int i;
    if (!w) return -1;
    if (!s) return lz_jw_lit(w, "\"\"");
    if (n < 0) n = (int)strlen(s);
    /* Worst case is six bytes out per byte in (\u00XX). Reserving it up
       front means the loop below never has to check again. */
    if (!jw_grow(w, n * 6 + 3)) return -1;
    w->s[w->len++] = '"';
    for (i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  w->s[w->len++] = '\\'; w->s[w->len++] = '"';  break;
        case '\\': w->s[w->len++] = '\\'; w->s[w->len++] = '\\'; break;
        case '\n': w->s[w->len++] = '\\'; w->s[w->len++] = 'n';  break;
        case '\r': w->s[w->len++] = '\\'; w->s[w->len++] = 'r';  break;
        case '\t': w->s[w->len++] = '\\'; w->s[w->len++] = 't';  break;
        case '\b': w->s[w->len++] = '\\'; w->s[w->len++] = 'b';  break;
        case '\f': w->s[w->len++] = '\\'; w->s[w->len++] = 'f';  break;
        default:
            if (c < 0x20) {
                static const char hex[] = "0123456789abcdef";
                w->s[w->len++] = '\\';
                w->s[w->len++] = 'u';
                w->s[w->len++] = '0';
                w->s[w->len++] = '0';
                w->s[w->len++] = hex[(c >> 4) & 0xF];
                w->s[w->len++] = hex[c & 0xF];
            } else {
                /* >= 0x80 falls here too: UTF-8 is valid JSON as-is. */
                w->s[w->len++] = (char)c;
            }
        }
    }
    w->s[w->len++] = '"';
    w->s[w->len] = 0;
    return 0;
}

int lz_jw_int(LZJsonW *w, long v) {
    char b[32];
    int n = 0, neg = 0, i;
    unsigned long u;
    if (!w) return -1;
    if (v < 0) { neg = 1; u = (unsigned long)(-(v + 1)) + 1UL; }
    else u = (unsigned long)v;
    if (u == 0) b[n++] = '0';
    while (u) { b[n++] = (char)('0' + (int)(u % 10)); u /= 10; }
    if (!jw_grow(w, n + 2)) return -1;
    if (neg) w->s[w->len++] = '-';
    for (i = n - 1; i >= 0; i--) w->s[w->len++] = b[i];
    w->s[w->len] = 0;
    return 0;
}
