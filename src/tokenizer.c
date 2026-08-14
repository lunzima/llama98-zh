/* Qwen3.5-family byte-level BPE tokenizer (C implementation).
 *
 * Data flow (aligned with the HF tokenizers library):
 *
 *   UTF-8 input
 *     -> special-token longest-match split
 *     -> pre-tokenize (GPT-2 regex approximation, at Unicode codepoint level)
 *     -> byte-level mapping (GPT-2 bytes_to_unicode, byte -> codepoint)
 *     -> greedy BPE merge (rank = order in the merges table)
 *     -> token id
 *
 * Decoding: a token's byte sequence is concatenated directly (a native
 * property of byte-level BPE; no ## or space markers like
 * SentencePiece).
 *
 * The merges hash table stores (pair -> rank): a pair's key is the left
 * and right symbols' codepoint sequences encoded at 2 bytes/codepoint.
 * Symbols keep merging and growing during encoding, so keys are
 * variable-length byte strings, not a fixed structure.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compat.h"
#include "err.h"
#include "json.h"
#include "tokenizer.h"
#include "unicode.h"

/* ------------------------------------------------------ byte mapping table */

static void build_byte_map(int *byte_to_cp, int *cp_to_byte) {
    int i, n = 0;
    for (i = 0; i < 256; i++) byte_to_cp[i] = -1;
    for (i = 0; i < 512; i++) cp_to_byte[i] = -1;
    /* printable range maps directly (identical to GPT-2) */
    for (i = '!'; i <= '~'; i++) byte_to_cp[i] = i;
    for (i = 0xA1; i <= 0xAC; i++) byte_to_cp[i] = i;
    for (i = 0xAE; i <= 0xFF; i++) byte_to_cp[i] = i;
    /* remaining bytes map to codepoints starting at 256 */
    for (i = 0; i < 256; i++) {
        if (byte_to_cp[i] < 0) byte_to_cp[i] = 256 + (n++);
    }
    for (i = 0; i < 256; i++) cp_to_byte[byte_to_cp[i]] = i;
}

/* --------------------------------------------------------- UTF-8 utilities */

/* Thin adapter: UTF-8 encode/decode and char classification live in unicode.c (table-driven, exact). */
static int utf8_to_cps(const unsigned char *s, int len, uint32_t *cps, int maxcp) {
    return lz_utf8_decode((const char *)s, len, cps, maxcp);
}
static int utf8_encode(uint32_t cp, unsigned char *out) {
    return lz_utf8_encode(cp, out);
}
static int is_letter(int c) { return lz_uni_is_letter((uint32_t)c); }
static int is_mark(int c)   { return lz_uni_is_mark((uint32_t)c); }
static int is_number(int c) { return lz_uni_is_number((uint32_t)c); }
static int is_space(int c)  { return lz_uni_is_space((uint32_t)c); }
static int is_newline(int c) { return c == '\n' || c == '\r'; }

/* ----------------------------------------------------- merges hash table */

static unsigned int merge_hash(const unsigned char *key, int len) {
    unsigned int h = 5381;
    int i;
    for (i = 0; i < len; i++) h = (h << 5) + h + key[i];
    return h;
}

/* Look up the rank of a pair (codepoint sequence of left+right symbols); -1 if absent */
static int merge_rank(const LZTokenizer *t, const uint32_t *left, int llen,
                      const uint32_t *right, int rlen) {
    unsigned char key[LZ_TK_MERGE_KEY_MAX];
    int key_len = 0, i, idx, step;
    /* Not a truncation: the loader rejects any merge whose key would
       exceed this buffer, so a pair too long to fit is a pair that
       cannot be in the table, and "absent" is the right answer. */
    if ((llen + rlen) * 2 > (int)sizeof(key)) return -1;
    for (i = 0; i < llen; i++) {
        key[key_len++] = (unsigned char)(left[i] & 0xFF);
        key[key_len++] = (unsigned char)((left[i] >> 8) & 0xFF);
    }
    for (i = 0; i < rlen; i++) {
        key[key_len++] = (unsigned char)(right[i] & 0xFF);
        key[key_len++] = (unsigned char)((right[i] >> 8) & 0xFF);
    }
    idx = (int)(merge_hash(key, key_len) & (unsigned int)(t->merge_cap - 1));
    step = 1;
    while (t->merge_tab[idx].used) {
        if (t->merge_tab[idx].key_len == key_len &&
            memcmp(t->merge_tab[idx].key, key, (size_t)key_len) == 0)
            return t->merge_tab[idx].rank;
        idx = (idx + step) & (t->merge_cap - 1);
        step++;
    }
    return -1;
}

static int merge_insert(LZTokenizer *t, const unsigned char *key, int key_len,
                        int rank) {
    int idx, step;
    if (t->n_merges >= t->merge_cap / 2) return -1;
    idx = (int)(merge_hash(key, key_len) & (unsigned int)(t->merge_cap - 1));
    step = 1;
    while (t->merge_tab[idx].used) {
        idx = (idx + step) & (t->merge_cap - 1);
        step++;
    }
    t->merge_tab[idx].key = (char *)malloc((size_t)key_len);
    if (!t->merge_tab[idx].key) return -1;
    memcpy(t->merge_tab[idx].key, key, (size_t)key_len);
    t->merge_tab[idx].key_len = key_len;
    t->merge_tab[idx].rank = rank;
    t->merge_tab[idx].used = 1;
    t->n_merges++;
    return 0;
}

/* ------------------------------------------------------ token lookup */

/* qsort with context (no glibc qsort_r dependency; works on MinGW/Watcom) */
static void sort_idx(int *idx, int lo, int hi,
                     int (*cmp)(const LZTokenizer *, int, int),
                     const LZTokenizer *t) {
    int i, j, pivot;
    if (lo >= hi) return;
    pivot = idx[(lo + hi) / 2];
    i = lo; j = hi;
    while (i <= j) {
        while (cmp(t, idx[i], pivot) < 0) i++;
        while (cmp(t, idx[j], pivot) > 0) j--;
        if (i <= j) {
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
            i++; j--;
        }
    }
    sort_idx(idx, lo, j, cmp, t);
    sort_idx(idx, i, hi, cmp, t);
}

static int word_cmp(const LZTokenizer *t, int a, int b) {
    if (t->word_lens[a] != t->word_lens[b])
        return t->word_lens[a] - t->word_lens[b];
    return memcmp(t->words[a], t->words[b], (size_t)t->word_lens[a]);
}

/* Look up a token id by byte string; -1 if absent */
static int vocab_find(const LZTokenizer *t, const unsigned char *bytes, int len) {
    int lo = 0, hi = t->total - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int w = t->sorted[mid];
        int c;
        if (t->word_lens[w] != len) c = t->word_lens[w] - len;
        else c = memcmp(t->words[w], bytes, (size_t)len);
        if (c == 0) return w;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* Convert a symbol (byte-level codepoint sequence) back to bytes, then look up the id */
static int symbol_to_id(const LZTokenizer *t, const uint32_t *cps, int n) {
    static unsigned char buf[LZ_TK_MAX_WORD];
    int i, m = 0;
    if (n > (int)sizeof(buf)) return -1;
    for (i = 0; i < n; i++) {
        int b = (cps[i] < 512) ? t->cp_to_byte[cps[i]] : -1;
        if (b < 0) return -1;
        buf[m++] = (unsigned char)b;
    }
    return vocab_find(t, buf, m);
}

/* ----------------------------------------------------- pre-tokenize */

typedef struct { int start, end; } LZSeg;

/* Approximate implementation of the GPT-2 pre-tokenizer regex,
   splitting on a Unicode codepoint array. The original regex:
     (?i:'s|'t|'re|'ve|'m|'ll|'d)|
     [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|
     \p{N}|
      ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|
     \s*[\r\n]+|
     \s+(?!\S)|
     \s+
   Returns the segment count; a segment is a [start, end) codepoint range. */
static int pretokenize(const uint32_t *cps, int n, LZSeg *segs, int maxsegs) {
    int nseg = 0, i = 0;

    while (i < n) {
        int c = cps[i];

        /* branch 1: English contraction suffixes 's 't 're 've 'm 'll 'd (case-insensitive) */
        if (c == '\'' && i + 1 < n) {
            static const char *suffs[] = { "s", "t", "re", "ve", "m", "ll", "d" };
            int si, matched = 0;
            for (si = 0; si < 7; si++) {
                int sl = (int)strlen(suffs[si]);
                int k;
                if (i + 1 + sl > n) continue;
                for (k = 0; k < sl; k++) {
                    int cc = cps[i + 1 + k];
                    int lc = (cc >= 'A' && cc <= 'Z') ? cc + 32 : cc;
                    if (lc != suffs[si][k]) break;
                }
                if (k == sl) { matched = 1; break; }
            }
            if (matched) {
                if (nseg < maxsegs) { segs[nseg].start = i; segs[nseg].end = i + 1 + (int)strlen(suffs[si]); }
                nseg++;
                i = i + 1 + (int)strlen(suffs[si]);
                continue;
            }
        }

        /* branch 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
           The prefix is any char that is not newline/letter/digit -
           including space. So " a" is one token (space+letters), not
           two. */
        if (is_letter(c) || is_mark(c)) {
            int start = i;
            i++;
            while (i < n && (is_letter(cps[i]) || is_mark(cps[i]))) i++;
            if (nseg < maxsegs) { segs[nseg].start = start; segs[nseg].end = i; }
            nseg++;
            continue;
        }
        if (!is_newline(c) && !is_number(c) && i + 1 < n &&
            (is_letter(cps[i + 1]) || is_mark(cps[i + 1]))) {
            int start = i;
            i++;
            while (i < n && (is_letter(cps[i]) || is_mark(cps[i]))) i++;
            if (nseg < maxsegs) { segs[nseg].start = start; segs[nseg].end = i; }
            nseg++;
            continue;
        }

        /* branch 3: a single digit */
        if (is_number(c)) {
            if (nseg < maxsegs) { segs[nseg].start = i; segs[nseg].end = i + 1; }
            nseg++;
            i++;
            continue;
        }

        /* branch 5: \s*[\r\n]+ newline run (checked before the punctuation/space branches) */
        if (is_newline(c)) {
            int start = i;
            while (i < n && (is_space(cps[i]) || is_newline(cps[i]))) i++;
            if (nseg < maxsegs) { segs[nseg].start = start; segs[nseg].end = i; }
            nseg++;
            continue;
        }

        /* branch 4: ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* punctuation run (may carry one leading space) */
        if (is_space(c) || !is_newline(c)) {
            int start = i;
            if (is_space(c)) i++;                     /* absorb leading space */
            while (i < n && !is_space(cps[i]) && !is_newline(cps[i]) &&
                   !is_letter(cps[i]) && !is_mark(cps[i]) && !is_number(cps[i]))
                i++;
            while (i < n && is_newline(cps[i])) i++;  /* trailing newlines */
            if (i > start) {
                if (nseg < maxsegs) { segs[nseg].start = start; segs[nseg].end = i; }
                nseg++;
                continue;
            }
        }

        /* branch 6/7: whitespace runs */
        {
            int start = i;
            while (i < n && is_space(cps[i])) i++;
            if (i == start) i++;                      /* safety: avoid infinite loop */
            if (nseg < maxsegs) { segs[nseg].start = start; segs[nseg].end = i; }
            nseg++;
        }
    }
    return nseg;
}

/* --------------------------------------------------------- BPE merging */

/* A symbol is a contiguous run of the word's codepoints. Symbols only
   ever merge with the neighbour on their right, so a symbol stays a
   contiguous range and `off` never has to move - which is why a merge is
   just `len += right->len` plus an unlink. Slots are never reused, so an
   index identifies the same symbol for the whole word; the heap below
   depends on that. */
typedef struct {
    int off;                /* start in cps[] */
    int len;                /* codepoints; 0 once merged away */
    int prev, next;         /* -1 at the ends */
} LZSym;

/* A candidate merge, as it looked when it was pushed. lsz/rsz are the
   two symbol lengths at push time and are what makes a stale entry
   detectable without re-deriving the pair's text. */
typedef struct {
    int rank;
    int left;
    int lsz, rsz;
} LZBigram;

/* Order: lowest rank first; ties to the LEFTMOST symbol.
   The tie rule is not a detail. `left` is the symbol's slot index, which
   is its leftmost codepoint's original position, so ordering by it
   reproduces exactly what the previous left-to-right rescan did with a
   strict `<`. HF's tokenizers and llama.cpp both break ties the same
   way; break them right instead and 3 of the 600 sweep strings change
   tokenization. */
static int bigram_less(const LZBigram *a, const LZBigram *b) {
    if (a->rank != b->rank) return a->rank < b->rank;
    return a->left < b->left;
}

static void heap_push(LZBigram *h, int *nh, const LZBigram *v) {
    int i = (*nh)++;
    h[i] = *v;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!bigram_less(&h[i], &h[p])) break;
        { LZBigram tmp = h[p]; h[p] = h[i]; h[i] = tmp; }
        i = p;
    }
}

static void heap_pop(LZBigram *h, int *nh, LZBigram *out) {
    int i = 0;
    *out = h[0];
    h[0] = h[--(*nh)];
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < *nh && bigram_less(&h[l], &h[m])) m = l;
        if (r < *nh && bigram_less(&h[r], &h[m])) m = r;
        if (m == i) break;
        { LZBigram tmp = h[m]; h[m] = h[i]; h[i] = tmp; }
        i = m;
    }
}

/* Push (a, a->next) if the vocabulary ranks it. */
static void bigram_try(const LZTokenizer *t, const uint32_t *cps,
                       const LZSym *sym, LZBigram *heap, int *nheap,
                       int a, int b) {
    LZBigram v;
    int r;
    if (a < 0 || b < 0) return;
    r = merge_rank(t, cps + sym[a].off, sym[a].len,
                   cps + sym[b].off, sym[b].len);
    if (r < 0) return;
    v.rank = r; v.left = a; v.lsz = sym[a].len; v.rsz = sym[b].len;
    heap_push(heap, nheap, &v);
}

/* Greedy BPE merge on a single token (byte-level codepoint sequence).
   Final symbols go to syms (concatenated codepoints); slens[i] is the
   codepoint count of symbol i. Returns symbol count; -1 on failure.

   O(n log n). The candidates live in a min-heap and a merge only
   invalidates its own two neighbours, so only those two get re-pushed.
   Rescanning every adjacent pair for the lowest rank each pass would be
   O(n^2) merge_rank calls - 7.45M at LZ_TK_MAX_WORD, and quadratic in a
   length the client controls. Same structure llama.cpp and Ollama both
   use.

   Stale entries are not removed from the heap; they are recognised on
   pop. An entry is valid iff both symbols are still exactly the size
   they were when it was pushed:
     - if `left` grew, it merged with something and its `next` moved too;
     - if `left` is unchanged then `next` is unchanged, because a symbol
       is only ever unlinked by a merge INTO its left neighbour, which
       would have changed that neighbour's length;
     - so re-deriving right = sym[left].next and checking its length is
       enough, and the pair's text never has to be rebuilt.
   Lengths only grow, so a stale entry can never become valid again. */
static int bpe_word(const LZTokenizer *t, const uint32_t *cps, int n,
                    uint32_t *syms, int *slens, int maxsym) {
    LZSym *sym = NULL;
    LZBigram *heap = NULL;
    int nheap = 0, i, pos, nsym = 0, rc = -1;

    if (n <= 0) return 0;
    if (n > maxsym) return -1;

    sym = (LZSym *)malloc((size_t)n * sizeof(LZSym));
    /* Push bound: n-1 seeds, plus at most 2 per merge and at most n-1
       merges, so 3n-3 entries can ever be pushed. Sizing for that means
       no growth check in the inner loop - and a heap overflow is not the
       kind of thing to leave resting on an argument, so the bound is
       verified by measurement: over 2028 strings (the 600-string sweep
       at 2000, the hand-written corpus, and a 1365-character CJK run at
       the cap) the peak simultaneous occupancy was 1365 against a bound
       of 12285. */
    heap = (LZBigram *)malloc((size_t)(3 * n) * sizeof(LZBigram));
    if (!sym || !heap) goto done;

    for (i = 0; i < n; i++) {
        sym[i].off = i;
        sym[i].len = 1;
        sym[i].prev = i - 1;
        sym[i].next = (i + 1 < n) ? i + 1 : -1;
    }
    for (i = 0; i + 1 < n; i++) bigram_try(t, cps, sym, heap, &nheap, i, i + 1);

    while (nheap > 0) {
        LZBigram b;
        int l, r;
        heap_pop(heap, &nheap, &b);
        l = b.left;
        if (sym[l].len != b.lsz) continue;          /* left moved on */
        r = sym[l].next;
        if (r < 0 || sym[r].len != b.rsz) continue; /* right moved on */

        sym[l].len += sym[r].len;
        sym[r].len = 0;
        sym[l].next = sym[r].next;
        if (sym[r].next >= 0) sym[sym[r].next].prev = l;

        bigram_try(t, cps, sym, heap, &nheap, sym[l].prev, l);
        bigram_try(t, cps, sym, heap, &nheap, l, sym[l].next);
    }

    /* Slot 0 is never unlinked - it has no left neighbour to be merged
       into - so the surviving chain always starts there. */
    pos = 0;
    for (i = 0; i >= 0; i = sym[i].next) {
        memcpy(syms + pos, cps + sym[i].off,
               (size_t)sym[i].len * sizeof(uint32_t));
        slens[nsym++] = sym[i].len;
        pos += sym[i].len;
    }
    rc = nsym;

done:
    free(sym);
    free(heap);
    return rc;
}

/* Byte length of a codepoint's UTF-8 form, without producing the bytes.
   Used only to size the per-word buffers below; the conversion itself
   still goes through utf8_encode, so the two cannot disagree about what
   gets written - this only has to agree about HOW MUCH. */
static int utf8_enc_len(uint32_t cp) {
    if (cp < 0x80)    return 1;
    if (cp < 0x800)   return 2;
    if (cp < 0x10000) return 3;
    return 4;
}

/* Encode plain text (no special tokens): pre-tokenize + BPE. Ids go to
   out (NULL = count only); returns the count; -1 on failure.

   The three per-word buffers are allocated ONCE per call, sized to the
   longest segment, rather than declared inside the segment loop with a
   fixed LZ_TK_MAX_WORD extent. As stack arrays they came to 49,376 bytes
   per call (gcc -fstack-usage), on the generation request path, against
   iron law six clause 4 - and a Win98 stack overflow is not catchable:
   the process dies at whatever call depth was deepest, which is rarely
   the frame at fault. The sizing pass is O(codepoints) and disappears
   next to bpe_word, which is O(n log n) in the segment length.

   LZ_TK_MAX_WORD still caps one segment, but it now bounds MEMORY
   rather than time - see its comment in tokenizer.h, which has been
   wrong about this twice. */
static int encode_segment(LZTokenizer *t, const unsigned char *bytes, int len,
                          int *out, int out_cap) {
    uint32_t *cps = NULL, *bcp = NULL, *syms = NULL;
    LZSeg *segs = NULL;
    int *slens = NULL;
    int ncp, nseg, si, i, nout = 0, cap = 0, rc = -1;

    if (len <= 0) return 0;
    /* Codepoints <= bytes; segments <= codepoints. Both are allocated
       by len, avoiding silent truncation of long texts (>4096
       codepoints) by fixed buffers. */
    cps = (uint32_t *)malloc((size_t)len * sizeof(uint32_t));
    segs = (LZSeg *)malloc((size_t)len * sizeof(LZSeg));
    if (!cps || !segs) goto done;
    ncp = utf8_to_cps(bytes, len, cps, len);
    nseg = pretokenize(cps, ncp, segs, len);

    /* Longest segment, in byte-level codepoints - which is its UTF-8
       byte count, since every byte maps to exactly one. */
    for (si = 0; si < nseg; si++) {
        int sb = 0;
        for (i = segs[si].start; i < segs[si].end; i++)
            sb += utf8_enc_len(cps[i]);
        if (sb > cap) cap = sb;
    }
    if (cap > LZ_TK_MAX_WORD) goto done;    /* cost guard; see above */
    if (cap == 0) { rc = 0; goto done; }    /* nothing encodable */

    /* Three allocations, not one carved block. The block would save two
       allocator calls out of the 141 a request costs (measured: a
       21-message chat render is 1698 bytes and runs encode_segment 47
       times, once per run of text between special tokens) - and an
       interleaved A/B said the block is 0.5% SLOWER here, not faster.
       At a wash, the version without the sizeof(int)==sizeof(uint32_t)
       assumption wins. */
    bcp   = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    syms  = (uint32_t *)malloc((size_t)cap * sizeof(uint32_t));
    slens = (int *)malloc((size_t)cap * sizeof(int));
    if (!bcp || !syms || !slens) goto done;

    for (si = 0; si < nseg; si++) {
        int nbcp = 0, nsym, pos;

        /* segment codepoints -> UTF-8 bytes -> byte-level codepoints */
        for (i = segs[si].start; i < segs[si].end; i++) {
            unsigned char ub[4];
            int ulen = utf8_encode(cps[i], ub);
            int k;
            for (k = 0; k < ulen; k++) {
                int cp = t->byte_to_cp[ub[k]];
                if (cp < 0) goto done;
                bcp[nbcp++] = cp;       /* cap is this segment's own size */
            }
        }
        if (nbcp == 0) continue;

        nsym = bpe_word(t, bcp, nbcp, syms, slens, cap);
        if (nsym < 0) goto done;
        pos = 0;
        for (i = 0; i < nsym; i++) {
            int id = symbol_to_id(t, syms + pos, slens[i]);
            if (id < 0) goto done;
            /* out == NULL is count-only; everything above still runs, so
               the count is the real BPE result and not an estimate.
               Past out_cap the counting continues but the writing stops -
               the caller gets the true total and can size a second pass,
               which is the only way it can learn the number it needed. */
            if (out && nout < out_cap) out[nout] = id;
            nout++;
            pos += slens[i];
        }
    }
    rc = nout;

done:
    free(cps);
    free(segs);
    free(bcp);
    free(syms);
    free(slens);
    return rc;
}

/* ------------------------------------------------------ special matching */

/* Which bytes can begin a special token. Built once per lz_encode call
   from the tokenizer's own specials - NOT assumed. This vocab's are all
   "<|...|>", but that is a property of this vocab, and a tokenizer whose
   specials start with something else must still tokenize correctly
   rather than quietly stop matching them. */
typedef unsigned char LZFirstByte[256];

static void first_byte_build(const LZTokenizer *t, LZFirstByte can) {
    int i;
    memset(can, 0, 256);
    for (i = 0; i < t->n_special; i++) {
        if (t->special_len[i] > 0)
            can[(unsigned char)t->special_text[i][0]] = 1;
    }
}

/* Match a special token at pos in bytes; returns the special index, or -1.

   `can` short-circuits the common case. The scan below runs at EVERY
   byte position of the input, over every special - 33 of them in this
   vocab - so without the filter a prompt costs len * n_special bounds
   checks and memcmp calls. Measured on a 3920-byte prompt: 0.631 ms
   without, 0.375 ms with. 41% of the whole encode was this loop failing
   to match, one byte at a time. */
static int match_special(const LZTokenizer *t, const LZFirstByte can,
                         const char *bytes, int len, int pos) {
    int i;
    if (!can[(unsigned char)bytes[pos]]) return -1;
    for (i = 0; i < t->n_special; i++) {
        int sl = t->special_len[i];
        if (pos + sl <= len &&
            memcmp(bytes + pos, t->special_text[i], (size_t)sl) == 0)
            return i;
    }
    return -1;
}

/* ------------------------------------------------------- public API */

int lz_tokenizer_vocab_size(const LZTokenizer *t) {
    return t ? t->total : 0;
}

int lz_tokenizer_n_special(const LZTokenizer *t) {
    return t ? t->n_special : 0;
}

int lz_tokenizer_n_merges(const LZTokenizer *t) {
    return t ? t->n_merges : 0;
}

int lz_encode(LZTokenizer *t, const char *bytes, int len,
              int bos, int eos, int *out, int out_cap) {
    int nout = 0, pos = 0, r = 0;
    char *norm = NULL;
    int nlen;
    /* On the stack, not in LZTokenizer: 256 bytes is small, building it
       costs a memset plus n_special stores once per call, and keeping it
       out of the struct leaves the ABI and lz_encode's thread safety
       unchanged. */
    LZFirstByte can;
    (void)bos; (void)eos;   /* Qwen has no BOS convention; EOS is handled by generation logic */
    if (!t || !bytes || len < 0) return -1;
    if (!out) out_cap = 0;      /* count-only; never trust a stale cap */
    if (out_cap < 0) return -1;
    first_byte_build(t, can);

    /* The normalizer is NFC (from tokenizer.json), aligned with HF
       tokenizers: normalize the whole input first, then special matching
       and BPE. added_tokens have normalized=false and are unaffected
       (this vocab's specials are all ASCII). */
    if (len > 0) {
        norm = (char *)malloc((size_t)len * 4 + 16);
        if (!norm) return -1;
        nlen = lz_utf8_nfc(bytes, len, norm, (size_t)len * 4 + 16);
        if (nlen < 0) { free(norm); return -1; }
        bytes = norm;
        len = nlen;
    }

    while (pos < len) {
        int sp = match_special(t, can, bytes, len, pos);
        if (sp >= 0) {
            if (out && nout < out_cap) out[nout] = t->special_id[sp];
            nout++;
            pos += t->special_len[sp];
        } else {
            int start = pos;
            int room;
            while (pos < len && match_special(t, can, bytes, len, pos) < 0) pos++;
            if (pos > start) {
                /* Remaining room, floored at 0: once the buffer is full the
                   segments still have to be encoded to keep counting, they
                   just stop writing. */
                room = (nout < out_cap) ? out_cap - nout : 0;
                r = encode_segment(t, (const unsigned char *)bytes + start,
                                   pos - start,
                                   (out && room > 0) ? out + nout : NULL, room);
                if (r < 0) { free(norm); return -1; }
                nout += r;
            }
        }
    }
    free(norm);
    return nout;
}

int lz_decode_into(LZTokenizer *t, int token, char *buf, int cap,
                    int *out_len) {
    if (!t || token < 0 || token >= t->total) {
        if (out_len) *out_len = 0;
        return 0;
    }
    {
        int n = t->word_lens[token];
        if (n > LZ_TK_DECODE_CAP) n = LZ_TK_DECODE_CAP;
        if (out_len) *out_len = n;
        if (buf && n > 0) {
            int m = (cap < n) ? cap : n;
            memcpy(buf, t->words[token], (size_t)m);
            return m;
        }
        return 0;
    }
}

const char *lz_decode(LZTokenizer *t, int token, int *out_len) {
    int n = 0;
    lz_decode_into(t, token, NULL, 0, &n);
    if (n <= 0) {
        if (out_len) *out_len = n;
        return "";
    }
    if (n > LZ_TK_DECODE_CAP) n = LZ_TK_DECODE_CAP;
    lz_decode_into(t, token, t->decode_buf, LZ_TK_DECODE_CAP, &n);
    if (out_len) *out_len = n;
    return t->decode_buf;
}

int lz_tokenizer_find(LZTokenizer *t, const char *bytes, int len) {
    if (!t || !bytes || len < 0) return -1;
    return vocab_find(t, (const unsigned char *)bytes, len);
}

/* Token text (JSON string UTF-8) -> byte sequence stored in
   words[id]. Token content is byte-level-mapped codepoints; restore
   each codepoint back to its original byte. */
/* Decode one UTF-8 codepoint at s[pos..len); advances pos. Returns the
   codepoint, or 0xFFFFFFFF if the byte is not a valid start/continuation
   (callers here treat that as "not a byte-level codepoint" and bail).

   This exists so word_from_json need not materialize the codepoint
   array; see the comment there. It deliberately mirrors lz_utf8_decode's
   acceptance, not a stricter one - a token this loop rejects but
   lz_utf8_decode accepts would make the vocabulary disagree with the
   encoder. */
static uint32_t utf8_next(const unsigned char *s, int len, int *pos) {
    uint32_t cp;
    int i = *pos, n, k;
    unsigned char c = s[i];

    if (c < 0x80)             { *pos = i + 1; return c; }
    else if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07; }
    else { *pos = i + 1; return 0xFFFFFFFFu; }

    if (i + n > len) { *pos = len; return 0xFFFFFFFFu; }
    for (k = 1; k < n; k++) {
        if ((s[i + k] & 0xC0) != 0x80) { *pos = i + k; return 0xFFFFFFFFu; }
        cp = (cp << 6) | (uint32_t)(s[i + k] & 0x3F);
    }
    *pos = i + n;
    return cp;
}

static int word_from_json(LZTokenizer *t, int id, const char *text, int len) {
    const unsigned char *s = (const unsigned char *)text;
    unsigned char *bytes;
    int pos = 0, m = 0;

    if (id < 0 || id >= t->total) return 0;

    /* Decoded IN PLACE rather than through a uint32_t cps[LZ_TK_MAX_WORD]
       staging array. Such an array would be 16,384 bytes of stack (iron
       law 6 clause 4: Win98 stacks are small) and pure staging - every
       codepoint consumed once, immediately, by the loop below. It would
       also cap a token at LZ_TK_MAX_WORD codepoints for no reason of its
       own; that cap belongs to bpe_word's cost guard, not to vocabulary
       loading. One byte-level codepoint is >= 1 byte of UTF-8, so len+1
       always suffices for the output. */
    bytes = (unsigned char *)malloc((size_t)len + 1);
    if (!bytes) return 0;
    while (pos < len) {
        uint32_t cp = utf8_next(s, len, &pos);
        int b = (cp < 512) ? t->cp_to_byte[cp] : -1;
        if (b < 0) { free(bytes); return 0; }
        bytes[m++] = (unsigned char)b;
    }
    if (t->words[id]) free(t->words[id]);
    t->words[id] = (char *)bytes;
    t->word_lens[id] = m;
    return 1;
}

int lz_tokenizer_load(LZTokenizer *t, const char *path,
                      char *errbuf, int errlen) {
    char *buf;
    size_t sz = 0;
    LZJson j;
    const LZJsonNode *root, *model, *vocab, *merges, *added, *e;
    int i;

    memset(t, 0, sizeof(*t));
    build_byte_map(t->byte_to_cp, t->cp_to_byte);

    buf = (char *)lz_read_file(path, &sz, errbuf, errlen);
    if (!buf) return 1;
    if (lz_json_parse(&j, buf, sz, errbuf, errlen) != 0) {
        free(buf);
        return 1;
    }
    free(buf);

    root = lz_json_root(&j);
    if (!root || root->type != LZ_JSON_OBJ) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_ROOT);
        goto fail;
    }
    model = lz_json_get(&j, root, "model");
    if (!model || model->type != LZ_JSON_OBJ) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_MODEL);
        goto fail;
    }
    e = lz_json_get(&j, model, "type");
    if (!lz_json_str_eq(e, "BPE")) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_TYPE);
        goto fail;
    }
    vocab = lz_json_get(&j, model, "vocab");
    merges = lz_json_get(&j, model, "merges");
    if (!vocab || vocab->type != LZ_JSON_OBJ) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_VOCAB);
        goto fail;
    }
    if (!merges || merges->type != LZ_JSON_ARR) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_MERGES);
        goto fail;
    }

    added = lz_json_get(&j, root, "added_tokens");
    t->vocab_size = vocab->n_children;
    t->n_added = (added && added->type == LZ_JSON_ARR) ? added->n_children : 0;
    t->total = t->vocab_size + t->n_added;

    t->words = (char **)calloc((size_t)t->total, sizeof(char *));
    t->word_lens = (int *)calloc((size_t)t->total, sizeof(int));
    t->sorted = (int *)malloc((size_t)t->total * sizeof(int));
    if (!t->words || !t->word_lens || !t->sorted) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_OOM);
        goto fail;
    }

    /* vocab: member names are tokens, values are ids */
    for (e = lz_json_first(&j, vocab); e; e = lz_json_next(&j, e)) {
        int id = (int)e->num;
        if (id < 0 || id >= t->vocab_size) {
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_ID_RANGE, id);
            goto fail;
        }
        if (t->words[id]) {
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_ID_DUP, id);
            goto fail;
        }
        if (!word_from_json(t, id, e->key, e->key_len)) {
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_WORD, id);
            goto fail;
        }
    }
    for (i = 0; i < t->vocab_size; i++) {
        if (!t->words[i]) {
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_ID_MISSING, i);
            goto fail;
        }
    }

    /* added_tokens: all treated as special (all 33 of Qwen's are) */
    t->n_special = 0;
    if (added && added->type == LZ_JSON_ARR) {
        for (e = lz_json_first(&j, added); e; e = lz_json_next(&j, e)) {
            const LZJsonNode *idn = lz_json_get(&j, e, "id");
            const LZJsonNode *cn = lz_json_get(&j, e, "content");
            int id;
            if (!idn || !cn || cn->type != LZ_JSON_STR) continue;
            id = (int)idn->num;
            if (id < 0 || id >= t->total) continue;
            if (t->n_special >= LZ_TK_MAX_SPECIAL) break;
            if (!word_from_json(t, id, cn->text, cn->text_len)) continue;
            t->special_id[t->n_special] = id;
            t->special_text[t->n_special] = t->words[id];
            t->special_len[t->n_special] = t->word_lens[id];
            t->n_special++;
        }
        /* sort by text length descending (longest match first) */
        for (i = 1; i < t->n_special; i++) {
            int k = i;
            while (k > 0 && t->special_len[k] > t->special_len[k - 1]) {
                int tmp_id = t->special_id[k];
                const char *tmp_t = t->special_text[k];
                int tmp_l = t->special_len[k];
                t->special_id[k] = t->special_id[k - 1];
                t->special_text[k] = t->special_text[k - 1];
                t->special_len[k] = t->special_len[k - 1];
                t->special_id[k - 1] = tmp_id;
                t->special_text[k - 1] = tmp_t;
                t->special_len[k - 1] = tmp_l;
                k--;
            }
        }
    }

    /* token index (for binary search) */
    for (i = 0; i < t->total; i++) t->sorted[i] = i;
    sort_idx(t->sorted, 0, t->total - 1, word_cmp, t);

    /* merges: pair's left/right tokens turned to bytes as the key; rank = array index (0-based) */
    t->merge_cap = 1;
    while (t->merge_cap < merges->n_children * 2) t->merge_cap <<= 1;
    t->merge_tab = (LZMergeSlot *)calloc((size_t)t->merge_cap,
                                         sizeof(LZMergeSlot));
    if (!t->merge_tab) {
        if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_OOM);
        goto fail;
    }
    i = 0;
    for (e = lz_json_first(&j, merges); e; e = lz_json_next(&j, e), i++) {
        /* Note: do not use lz_json_at(&j, merges, i) on merges - it
           scans linearly from the array head every time, degrading to
           O(n²) over 247K elements. Walk directly.

           Two on-disk shapes for one merge entry, both real: the older
           `tokenizers` releases (and every vocab this project has built
           itself) write ["left", "right"], a 2-element array; current
           upstream checkpoints - e.g. stock Qwen/Qwen3.5-0.8B's
           tokenizer.json - write a single string "left right",
           space-separated. Byte-level BPE pre-tokenization maps a real
           space to "\xc4\xa0" (U+0120) before this stage, so a literal
           ASCII 0x20 inside the string can only be the field separator,
           never token content - splitting on the first one is exact,
           not a heuristic. */
        const unsigned char *ltext, *rtext;
        int ltext_len, rtext_len;
        uint32_t lcps[LZ_TK_MERGE_SIDE_MAX], rcps[LZ_TK_MERGE_SIDE_MAX];
        unsigned char key[LZ_TK_MERGE_KEY_MAX];
        int llen, rlen, key_len = 0, k;
        if (e->type == LZ_JSON_STR) {
            const char *sp = (const char *)memchr(e->text, ' ', (size_t)e->text_len);
            if (!sp) {
                if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_MERGE_PAIR, i);
                goto fail;
            }
            ltext = (const unsigned char *)e->text;
            ltext_len = (int)(sp - e->text);
            rtext = (const unsigned char *)(sp + 1);
            rtext_len = e->text_len - ltext_len - 1;
        } else {
            const LZJsonNode *ln = lz_json_at(&j, e, 0);
            const LZJsonNode *rn = lz_json_at(&j, e, 1);
            if (!ln || !rn || ln->type != LZ_JSON_STR || rn->type != LZ_JSON_STR) {
                if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_MERGE_PAIR, i);
                goto fail;
            }
            ltext = (const unsigned char *)ln->text; ltext_len = ln->text_len;
            rtext = (const unsigned char *)rn->text; rtext_len = rn->text_len;
        }
        llen = utf8_to_cps(ltext, ltext_len, lcps, LZ_TK_MERGE_SIDE_MAX);
        rlen = utf8_to_cps(rtext, rtext_len, rcps, LZ_TK_MERGE_SIDE_MAX);
        /* lz_utf8_decode stops at maxcp and returns the SHORT count - it
           does not report that it truncated. Filling the buffer exactly
           is therefore indistinguishable from overflowing it, and an
           overflow would insert a key that is a prefix of the real one:
           the table would then hold an entry no lookup can match and a
           merge that should happen silently would not, changing
           tokenization with nothing to show for it. Refuse at load
           instead. Longest side measured: 15 codepoints in the pruned
           s1v3 vocab, 76 upstream - the limit sits at 128. */
        if (llen >= LZ_TK_MERGE_SIDE_MAX || rlen >= LZ_TK_MERGE_SIDE_MAX) {
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_MERGE_LONG, i,
                                   LZ_TK_MERGE_SIDE_MAX);
            goto fail;
        }
        for (k = 0; k < llen; k++) {
            key[key_len++] = (unsigned char)(lcps[k] & 0xFF);
            key[key_len++] = (unsigned char)((lcps[k] >> 8) & 0xFF);
        }
        for (k = 0; k < rlen; k++) {
            key[key_len++] = (unsigned char)(rcps[k] & 0xFF);
            key[key_len++] = (unsigned char)((rcps[k] >> 8) & 0xFF);
        }
        if (merge_insert(t, key, key_len, i) != 0) {
            if (errbuf) lz_err_fmt(errbuf, errlen, LZ_ERR_TK_MERGE_FULL);
            goto fail;
        }
    }

    lz_json_free(&j);
    return 0;

fail:
    lz_json_free(&j);
    lz_tokenizer_free(t);
    return 1;
}

void lz_tokenizer_free(LZTokenizer *t) {
    int i;
    if (!t) return;
    if (t->words) {
        for (i = 0; i < t->total; i++) free(t->words[i]);
        free(t->words);
    }
    free(t->word_lens);
    free(t->sorted);
    if (t->merge_tab) {
        for (i = 0; i < t->merge_cap; i++) free(t->merge_tab[i].key);
        free(t->merge_tab);
    }
    memset(t, 0, sizeof(*t));
}
