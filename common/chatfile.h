#ifndef LZ_GUI_CHATFILE_H
#define LZ_GUI_CHATFILE_H

#include "chat.h"

/* The saved-conversation file, as pure bytes.
 *
 * WHAT IS AND IS NOT IN IT. The content comes from the HISTORY, not
 * from the transcript control, so the file holds the conversation and
 * not the window: no speaker labels, no "model loaded" lines, and no
 * think blocks - lz_chat_norm_history strips those out of history
 * already, and the file agreeing with history is what lets a loaded
 * conversation continue rather than merely be read.
 *
 * THE FILE IS LOSSY AND THAT IS NOT A BUG TO FIX HERE. The caller
 * writes it as GBK, because the target's Notepad reads ANSI; a
 * character GBK cannot represent becomes '?' on the way out. Writing
 * UTF-8 instead would make the file unreadable on the machine it is
 * for. The authority for what the model said stays in memory. This
 * module itself never touches GBK - it encodes/decodes UTF-8 bytes
 * only, the same bytes LZChatMsg.content already carries; the
 * GBK<->UTF-8 conversion and the CRLF conversion (common/savechat.h) both
 * happen on the caller's side, between this module and the file.
 *
 * AND THE DIRECTION OF THAT CRLF CONVERSION IS PART OF THE CONTRACT:
 * the FILE uses CRLF, memory uses bare LF. LZChatMsg.content must not
 * contain '\r'. This is not a preference - encode splits content on
 * '\n' and decode strips one trailing '\r' as part of the separator, so
 * a CR living in content comes back one byte short, silently, and only
 * on multi-line messages: a multi-line EDIT hands GetWindowText CRLF, so
 * lz_common_lf strips it on the way in (gui/main.c's read_input_utf8).
 *
 * DECODE ENFORCES ITS HALF OF THAT: no '\r' reaches the body, not only
 * the one that sits before a '\n'. A file is not necessarily one this
 * program wrote - it is a .txt the design expects people to open and
 * edit in Notepad - so a CR that is not a line ending is an input, not
 * an impossibility. The caller's lz_common_lf closes the input box; this
 * closes the file.
 *
 * THE MARKERS ARE AMBIGUOUS AND THE ESCAPE IS WHY THIS IS TESTED. A
 * line that is exactly "[user]" inside a message would otherwise split
 * the turn in two. On the way out such a line gets one more leading
 * '['; on the way in a run of leading '[' loses one. Round-tripping
 * that case is the reason this module exists as bytes rather than as
 * two sprintf calls in main.c. */

#define LZ_CHATFILE_MAGIC "# kunkun98 conversation v1"

/* Upper bound on ONE decoded message body. A message longer than this
 * is truncated on the way in, not overflowed - see lz_chatfile_decode.
 * 128 KiB is comfortably above anything LZ_CHAT_HIST_MAX turns of
 * ordinary chat produce; it exists so the decode buffer can be static
 * (iron law six: no large buffer on the stack) rather than unbounded. */
#define LZ_CHATFILE_BODY (1024 * 128)

/* snprintf semantics: returns the length the whole result needs, writes
 * at most `cap` bytes including the terminator. Never returns a bound
 * it did not compute - see iron law four, note 10, on why a function
 * that writes into the caller's buffer takes its size. */
int lz_chatfile_encode(const LZChatMsg *msgs, int n, char *out, int cap);

/* Decode into an already-initialised history. On success the history
 * holds exactly the turns in the file. On failure it is left EXACTLY as
 * it was - a half-loaded conversation renders a prompt that matches
 * nothing on screen, and the KV cache would then disagree with the
 * window for the rest of the session.
 *
 * `path` is used only to fill in the %s of the error message on a
 * malformed file; it is never opened here. Returns 0, or an LZ_ERR_*
 * code with errbuf set. */
int lz_chatfile_decode(const char *bytes, int len, LZChatHist *h,
                       const char *path, char *errbuf, int errlen);

#endif
