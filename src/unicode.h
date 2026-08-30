#ifndef LZ_UNICODE_H
#define LZ_UNICODE_H

#include "lz_int.h"   /* <stdint.h> is not on the language floor */

/* Unicode utilities: NFC normalization and character classification.
   Tables generated (Unicode 15.0);
   zero runtime dependencies, binary search. */

/* ---- UTF-8 ---- */

/* Decode to a codepoint array (invalid bytes become 0xFFFD).
   Returns codepoint count (<= maxcp). */
int lz_utf8_decode(const char *s, int len, uint32_t *cps, int maxcp);

/* Encode a single codepoint as UTF-8; returns 1..4 bytes */
int lz_utf8_encode(uint32_t cp, unsigned char *out);

/* Strict yes/no UTF-8 validity. 1 only when the whole range is a single
   run of well-formed UTF-8: no over-long forms, no surrogates, no
   codepoints past U+10FFFF, no truncated or bad-continuation sequences.
   This is the "is this really UTF-8, not GBK that happens to fit"
   question (cli_main.c's console sniff), NOT a decode: it produces no
   codepoints and tolerates nothing. The valid/invalid decision must stay
   identical to lz_utf8_decode's - the two only differ in what they DO
   with an invalid byte (reject vs U+FFFD). */
int lz_utf8_valid(const char *s, int n);

/* ---- NFC normalization ----
   Input UTF-8 (len bytes), write NFC result to out (outcap bytes).
   Returns output byte count; -1 on failure (outcap too small or decode
   error). Byte-identical to Python unicodedata.normalize('NFC', ...). */
int lz_utf8_nfc(const char *in, int len, char *out, int outcap);

/* ---- classification (binary search over range tables) ---- */
int lz_uni_is_letter(uint32_t cp);   /* \p{L} */
int lz_uni_is_mark(uint32_t cp);     /* \p{M} */
int lz_uni_is_number(uint32_t cp);   /* \p{N} */
int lz_uni_is_space(uint32_t cp);    /* Unicode White_Space (Rust regex \s) */

/* canonical combining class (0 = starter) */
int lz_uni_ccc(uint32_t cp);

#endif
