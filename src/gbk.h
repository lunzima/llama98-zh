#ifndef LZ_GBK_H
#define LZ_GBK_H

/* GBK <-> UTF-8 transcoding.
 *
 * Win9x has no CP_UTF8, so bytes handed to a window, a menu or a
 * RichEdit control are read with the ANSI code page - 936 on a Chinese
 * Win98. The engine speaks UTF-8 everywhere, so the front end needs a
 * transcoder that does not go through the OS. src/cli_main.c records the
 * same gap on the CLI side: a Chinese --prompt reaches the Watcom build
 * as GBK and the tokenizer turns every byte into U+FFFD.
 *
 * The character set is exactly the one the corpus is stripped down to, so
 * "the model can emit it" and "the front end can show it" are the same
 * predicate. It is a strict subset of Windows CP936 and
 * never contradicts it; the table generator carries the measurement
 * and the list of what is deliberately left out.
 *
 * Both functions have snprintf return semantics: the return value is the
 * number of bytes the FULL conversion needs, not counting the
 * terminating NUL, so `need = f(in, len, NULL, 0, NULL)` sizes a buffer
 * and `need + 1` always holds the result. At most `cap` bytes are
 * written and the output is NUL-terminated whenever cap > 0.
 *
 * They differ from snprintf in one way, on purpose: a character is
 * written whole or not at all. Truncation never leaves a lone GBK lead
 * byte or a half UTF-8 sequence at the end, because those do not render
 * as "a shortened string", they render as garbage.
 *
 * Neither can fail, so neither takes an errbuf.
 */

/* UTF-8 -> GBK.
 *
 * Code points GBK cannot represent become '?'. That is a DISPLAY-layer
 * fallback: never write the result back into conversation history, or
 * the loss becomes permanent and the next render diverges from the KV
 * cache. Keep the UTF-8 original as the authority.
 *
 * `used` (may be NULL) receives how many INPUT bytes were consumed. A
 * trailing incomplete UTF-8 sequence is left unconsumed so a streaming
 * caller can prepend it to the next chunk; with used == NULL that tail
 * is converted as U+FFFD -> '?' instead. Bytes that cannot begin any
 * valid sequence are always consumed, whether or not `used` is passed.
 */
int lz_gbk_from_utf8(const char *in, int len, char *out, int cap, int *used);

/* GBK -> UTF-8.
 *
 * Invalid bytes and unmapped pairs become U+FFFD, resynchronising one
 * byte at a time - the same convention as lz_utf8_decode.
 *
 * `used` (may be NULL) receives how many INPUT bytes were consumed; a
 * trailing lone lead byte is left unconsumed for the next chunk.
 */
int lz_gbk_to_utf8(const char *in, int len, char *out, int cap, int *used);

#endif
