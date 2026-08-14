#ifndef LZ_TOKENIZER_H
#define LZ_TOKENIZER_H

#include <stddef.h>

/* Qwen3.5-family byte-level BPE tokenizer.
   Source: HF tokenizer.json (BPE + ByteLevel pre/post processor).
   Encoding aligns with the HF tokenizers library's GPT-2-style flow:
   NFC normalization (Unicode 15 tables, src/unicode.c) -> special
   longest-match -> pre-tokenize (GPT-2 regex, Unicode class tables)
   -> byte-level codepoint mapping -> greedy BPE merge -> id; decoding
   concatenates each token's bytes. */

/* Cap on ONE pre-tokenized segment, in byte-level codepoints
   (about 1365 CJK characters with no punctuation).

   What it is NOT:
     - not a buffer extent. encode_segment sizes its buffers to what the
       input actually needs.
     - not a TIME guard. bpe_word merges via a priority queue, so time
       scales near-linearly (measured exponent 0.91-1.06 at this value,
       not the 1.91-2.07 of a per-merge rescan).

   What it still is: a MEMORY guard on client-controlled input. bpe_word
   allocates 3n bigram slots (48n bytes) plus n symbols (16n), so 4096
   costs about 260 KB transient, and that scales linearly with whatever
   a request happens to contain. Raising it is cheap in time and linear
   in memory - but see word_from_json first, which sizes a STACK array
   by this constant and is 16,480 bytes. */
#define LZ_TK_MAX_WORD 4096
#define LZ_TK_MAX_SPECIAL 64        /* max special tokens */
#define LZ_TK_DECODE_CAP 4096       /* decode buffer bytes */

/* One side of a merge pair, in codepoints, and the resulting merge-table
   key in bytes (both sides, 2 bytes per codepoint).

   These bound BOTH ends on purpose: the loader refuses any merge longer
   than this, so no key in the table can exceed LZ_TK_MERGE_KEY_MAX,
   which is what lets merge_rank declare a 512-byte key buffer instead of
   one sized off LZ_TK_MAX_WORD*2 (8192) - a frame it would otherwise
   pay on every one of the millions of calls a single word costs. Written
   as one derived expression rather than two literals so the loader's
   buffer and the lookup's buffer cannot drift apart; if they did, the
   table would hold keys no lookup can ever match and the only symptom
   would be subtly different tokenization.

   MEASURED, on both vocabs this project loads:

     s1v3 (pruned)          32,278 merges   longest side  15 codepoints
     Qwen3.5-0.8B upstream 247,587 merges   longest side  76 codepoints

   The limit exists because the loader refuses any merge whose side
   exceeds it. A 64-codepoint limit would silently truncate upstream's
   longest sides: lz_utf8_decode stops at its limit and returns the SHORT
   count without reporting that it stopped, so a prefix key would go into
   the table and that merge would never fire - wrong tokenization, no
   error, no symptom to chase. 128 covers both vocabs' real maxima. */
#define LZ_TK_MERGE_SIDE_MAX 128
#define LZ_TK_MERGE_KEY_MAX  (LZ_TK_MERGE_SIDE_MAX * 2 * 2)

typedef struct {
    char *key;                      /* codepoint sequence, 2 bytes/codepoint */
    int   key_len;                  /* byte count */
    int   rank;
    int   used;
} LZMergeSlot;

typedef struct LZTokenizer {
    int vocab_size;                 /* ordinary token count */
    int n_added;                    /* added_tokens count */
    int total;                      /* vocab_size + n_added */

    char **words;                   /* [total] raw bytes per token (may contain any bytes) */
    int   *word_lens;
    int   *sorted;                  /* [total] token indices, ascending (len, bytes) */

    LZMergeSlot *merge_tab;         /* open-addressing hash table, power-of-2 capacity */
    int   merge_cap;
    int   n_merges;

    int   n_special;                /* special count */
    int   special_id[LZ_TK_MAX_SPECIAL];
    const char *special_text[LZ_TK_MAX_SPECIAL];  /* raw UTF-8 bytes */
    int   special_len[LZ_TK_MAX_SPECIAL];         /* descending by length */

    int   byte_to_cp[256];
    int   cp_to_byte[512];

    char  decode_buf[LZ_TK_DECODE_CAP];
} LZTokenizer;

/* Load from tokenizer.json. Non-zero return + errbuf on failure. */
int  lz_tokenizer_load(LZTokenizer *t, const char *path,
                       char *errbuf, int errlen);
void lz_tokenizer_free(LZTokenizer *t);

int  lz_tokenizer_vocab_size(const LZTokenizer *t);
int  lz_tokenizer_n_special(const LZTokenizer *t);
int  lz_tokenizer_n_merges(const LZTokenizer *t);

/* Encode. Returns the FULL token count the input produces, always -
   snprintf semantics. At most out_cap tokens are written to out; a
   return value greater than out_cap means out holds a truncated prefix
   and NOTHING was written past it. Negative on failure. bos/eos args are
   kept but ignored by this implementation - Qwen's vocab has no BOS
   convention, and EOS (<|im_end|>) is handled by the generation logic as
   a stop condition.

   out may be NULL (with out_cap 0) to COUNT ONLY - same convention as
   lz_decode_into's buf. The full BPE pass still runs, so the count is
   exact rather than an estimate. An HTTP front end needs this before it
   commits to a generation: rejecting an over-long request with 400, or
   trimming history to fit, otherwise means allocating a prompt-sized int
   array purely to discard it.

   THE out_cap ARGUMENT IS NOT DEFENSIVE PADDING: no fixed bound on the
   output is safe, because BPE runs on NFC-normalized text and NFC
   expands. Characters on Unicode's composition-exclusion list decompose
   and never recompose, so U+0958 (3 bytes) normalizes to U+0915 U+093C
   (6 bytes) and encodes to SIX tokens, not one. Ordinary Devanagari,
   Bengali, Gurmukhi or Oriya text therefore overruns any len+N buffer,
   as does any invalid UTF-8 (each bad byte becomes U+FFFD - 3 bytes,
   3 tokens). Both are reachable from the chat endpoint by a user simply
   typing - on Windows a Chinese argument arrives as GBK (MinGW converts
   main() argv to the ANSI code page), i.e. invalid UTF-8, i.e. 3x
   expansion. */
int  lz_encode(LZTokenizer *t, const char *bytes, int len,
               int bos, int eos, int *out, int out_cap);

/* Decode a single token. Returns a byte string + length pointing into
   an internal buffer that is overwritten by the next lz_decode call.
   NOT thread-safe for concurrent use of one tokenizer: the buffer lives
   in LZTokenizer (shared mutable state). Prefer lz_decode_into in any
   context where another thread may decode concurrently. */
const char *lz_decode(LZTokenizer *t, int token, int *out_len);

/* Thread-safe decode: caller-provided buffer. Returns bytes written.
   buf may be NULL to query the length (writes *out_len). */
int lz_decode_into(LZTokenizer *t, int token, char *buf, int cap,
                   int *out_len);

/* Look up a token id by byte string; -1 if absent (for the generation
   logic to find EOS/stop tokens). */
int lz_tokenizer_find(LZTokenizer *t, const char *bytes, int len);

#endif
