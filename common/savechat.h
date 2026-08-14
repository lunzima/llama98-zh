#ifndef LZ_GUI_SAVECHAT_H
#define LZ_GUI_SAVECHAT_H

/* Line endings, on the way from a rich edit control into a .txt file.
 *
 * RichEdit separates lines with a bare CR internally, and WM_GETTEXT
 * hands them over that way. A file full of lone CRs opens in Notepad as
 * one enormous line - which is not a crash, not an error, and not
 * something anyone notices until they open the file they saved.
 *
 * Pure and byte-oriented, so it is testable without a window and safe on
 * GBK: a GBK trail byte can be any value from 0x40 to 0xFE, but never
 * 0x0D or 0x0A, so scanning for CR and LF cannot land inside a
 * character. That is a property of the encoding, not luck, and it is
 * why this can work on display bytes at all.
 */

/* CR, LF or CRLF in; CRLF out, exactly one per line break.
 *
 * snprintf semantics: returns the bytes the full result needs, writes at
 * most cap including the terminator. `len` < 0 means NUL-terminated. */
int lz_common_crlf(const char *in, int len, char *out, int cap);

/* The other direction, and the one the INBOUND path needs.
 *
 * A multi-line EDIT control hands GetWindowText its lines separated by
 * CRLF. This is the one place between that call and the chat history
 * that removes the CR; without it every multi-line message the user
 * typed would carry a '\r' into LZChatMsg.content - and from there into
 * the prompt the model sees, where it is a byte the training corpus
 * essentially never contains.
 *
 * Normalising here rather than in common/chatfile.c is deliberate:
 * chatfile splits content on '\n' and strips a trailing '\r' as part of
 * the separator, so it is one consumer of a byte every inbound path
 * should be rid of.
 *
 * SAFE ON UTF-8, FOR A DIFFERENT REASON THAN THE ONE ABOVE. This
 * function's only caller (gui/main.c's read_input_utf8) runs it AFTER
 * lz_gbk_to_utf8, not before - so what it scans for '\r'/'\n' is UTF-8,
 * not GBK, and the file-level comment's argument (no GBK trail byte is
 * ever 0x0D or 0x0A) does not apply to it. It is still safe: UTF-8
 * continuation bytes and multi-byte lead bytes are always >= 0x80, so a
 * byte with value 0x0D or 0x0A can only ever be a complete, standalone
 * ASCII character in valid UTF-8, never part of a longer sequence.
 * Scanning for CR/LF cannot land inside a character here either - the
 * property holds on both sides of the gbk_to_utf8 conversion, just not
 * for the same reason.
 *
 * OUTPUT IS NEVER LONGER THAN INPUT. Every loop iteration consumes at
 * least 1 input byte (1 for an ordinary byte or a lone CR, 2 for CRLF)
 * and produces exactly 1 output byte, so `need` (the snprintf-style
 * return value) never exceeds `len`. read_input_utf8 relies on this
 * without saying so: raw[49152] (the UTF-8 before this function runs)
 * and do_send's utf8[49152] (what this function writes into) are sized
 * equal, which is only safe because this function cannot grow its input.
 *
 * Same snprintf semantics as above. */
int lz_common_lf(const char *in, int len, char *out, int cap);

#endif
