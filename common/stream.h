#ifndef LZ_GUI_STREAM_H
#define LZ_GUI_STREAM_H

/* Turns the arbitrary byte runs a generator produces into runs of
 * display bytes tagged with a STYLE MASK: think, and the three inline
 * Markdown switches bold/italic/code.
 *
 * Three boundary problems, and they are the reason this is a state
 * machine and not a call to lz_gbk_from_utf8:
 *
 *   1. A UTF-8 character can be split across chunks. Converting a
 *      partial sequence yields a replacement character that never
 *      un-becomes one.
 *   2. A <think>/</think> tag can be split across chunks. "<thi" at
 *      the end of one chunk and "nk>" at the start of the next is a
 *      tag; deciding when the first chunk arrives means deciding
 *      wrong.
 *   3. A Markdown marker can be split the same way - one asterisk at
 *      the end of a chunk, needing the next chunk's first byte to know
 *      whether it is "**" (bold) or a lone "*" (italic).
 *
 * All three are solved the same way - hold the tail until the next
 * chunk settles it - which is why they are one object rather than
 * separate layers. The longest thing that can need holding is in fact 7
 * ("</think" less its final character, still the longest of the four
 * markers), and LZ_STREAM_PEND is asserted against it.
 *
 * ALL FOUR MARKERS ARE CONSUMED, NOT DISPLAYED. A marker's own bytes
 * flip the relevant bit in `style` and are then dropped; only the
 * CONTENT on either side takes the style, the same way no renderer
 * echoes the asterisks in "**bold**".
 *
 * A <think>/</think> TAG'S OWN TRAILING NEWLINE IS PART OF THE TAG,
 * NOT PART OF THE CONTENT: a model that puts the
 * close tag on its own line has one line-ending that belongs to the
 * TAG (deleting the tag without it leaves an extra blank line the
 * model never asked for) and, usually, a second one right after that
 * IS the model's own intended paragraph break - dropping BOTH would
 * lose that break instead of fixing the bug. This is the SAME chunk-
 * boundary problem as #2 above, one call later: the newline can arrive
 * in a DIFFERENT chunk than the tag that makes it optional, so
 * `LZStream.after_tag` (below) carries the open question across calls
 * the same way `pend` carries an unsettled prefix.
 *
 * The four bits are independent toggles, not a nested grammar: no
 * pairing rules, no precedence beyond "**" being checked before a
 * lone "*" (so bold is not swallowed one asterisk at a time as two
 * italics), and they freely stack - <think> text can also be bold,
 * code can be inside a think block, etc.
 *
 * SCOPE - a USER decision, not a reading of CommonMark. IN: the
 * three inline styles above, plus the block-level "# ".."###### "
 * headings, "- "/"* "/"+ " bullets, "> " blockquotes, "---" rules,
 * "```" fences and "| a | b |" tables - all six built. Two things are
 * OUT on purpose rather than unfinished:
 *   - "1. " ordered lists are not touched AT ALL. A number and a dot
 *     already read as a list, and leaving them alone is also what keeps
 *     "2026. " from being eaten.
 *   - $latex$ is not rendered. Written down so it stays a decision
 *     rather than becoming a hole no one remembers.
 * Lists are not INDENTED either, only their marker changes: an indent
 * needs PARAFORMAT, and the table's tab stops are the only PARAFORMAT
 * this program sends (gui/main.c's append_run). The
 * scanner can still mis-toggle on an asterisk used as multiplication or
 * in maths - accepted, not fixed, per that scope.
 *
 * The sink receives GBK, because the only consumer is a control. The
 * UTF-8 original is the caller's to keep: conversation history has to
 * stay UTF-8, and a stream that also accumulated
 * it would be two authorities for one buffer.
 *
 * NOT thread safe, and not meant to be: it is driven from the UI thread
 * as WM_APP_TOKENS arrive. The conversion buffer is static.
 */

#define LZ_STREAM_PEND     8
#define LZ_STREAM_RUN_MAX  2048   /* display bytes per sink call */

/* How many U+2500 a "---" turns into. Here rather than buried in
 * stream.c: stream.c binds the
 * literal run to this with a compile-time assert, so the two cannot
 * drift apart silently. See stream.c's RULE_RUN for where 16 comes
 * from - it is the narrowest window this program allows, not a taste. */
#define LZ_STREAM_RULE_CHARS 16

/* Style bits, OR'd together in the `style` a run carries. Named after
 * what gui/main.c's append_run does with each: THINK recolours,
 * BOLD/ITALIC set a CHARFORMAT effect, CODE switches to a monospace
 * face. All four can be set on the same run. */
#define LZ_STYLE_THINK  0x01
#define LZ_STYLE_BOLD   0x02
#define LZ_STYLE_ITALIC 0x04
#define LZ_STYLE_CODE   0x08

/* Heading level, a 2-bit FIELD rather than a flag: 0 = not a heading,
 * 1/2/3 = h1/h2/h3. h4-h6 are CLAMPED to 3 - that is clamping, not
 * support; this checkpoint only ever writes "###".
 *
 * Two bits, not three: h4-h6 clamp rather than widening the field,
 * which is what keeps the whole mask inside one byte. */
#define LZ_STYLE_H_MASK  0x30
#define LZ_STYLE_H_SHIFT 4
#define LZ_STYLE_TABLE   0x40

/* This line is a bullet item. Unlike the three inline styles it is NOT
 * a character attribute - gui/main.c turns it into a PARAGRAPH hanging
 * indent, so a bullet that wraps has its continuation lined up under
 * the text instead of under the marker.
 *
 * Indenting is cheap here because append_run sends PARAFORMAT on every
 * run anyway - a bullet costs one more field in a message that is
 * already going out.
 *
 * 0x80 IS THE LAST BIT. It is usable only because the run-style arrays
 * are `unsigned char` - from a signed char
 * it would read back negative. The next flag needs a wider type in
 * BOTH the sink signature's callers and that array - it is not a free
 * bit, it is the last one. */
#define LZ_STYLE_BULLET  0x80

/* `gbk` holds `n` bytes and is NOT NUL-terminated past n. `style` is
 * the LZ_STYLE_* bits active for every byte in this run - a run never
 * spans a style change, so one mask describes the whole thing. */
typedef void (*LZStreamSink)(void *ud, const char *gbk, int n, int style);

typedef struct {
    char pend[LZ_STREAM_PEND];
    int  n_pend;
    int  style;
    /* Set the instant a <think>/</think> tag is fully consumed, cleared
     * the instant the (optional) newline question right after it is
     * resolved - see stream.c's own comment on the tag-consumption
     * branch for why this needs to be a field rather than local state:
     * the answer can depend on a chunk that has not arrived yet. */
    int  after_tag;
    /* Inside a ``` fenced block. A fence is the one place where the
     * single-byte reading of '`' breaks down: one backtick is decided
     * from a single byte, three are not - "``" can still become "```",
     * and an unclosed fence would leave LZ_STYLE_CODE ON and make
     * everything after the code block monospace, the info string
     * ("python") included. */
    int  fence;
    /* Discard everything up to and including the next newline. Its own
     * field because that newline can arrive in a later chunk. TWO users:
     * the info string after an opening fence ("```python"), and a
     * table's "|---|---|" separator row. Both are "the rest of this
     * line belongs to the marker, not to the content", and one flag
     * answering it once is better than two fields that would have to be
     * kept from being set at the same time. */
    int  skip_line;
    /* Whether the next byte starts a line. Every block-level construct
     * is only recognised at a line start, and this is that prerequisite,
     * carried across chunks because
     * byte 0's answer lives in the chunk before it. */
    int  at_bol;
    /* Inside the leading '>' run of a blockquote line. Eaten one byte at
     * a time through this field rather than counted with look-ahead,
     * because the number of '>'s has NO upper bound and the hold does:
     * LZ_STREAM_PEND is 8 and lz_stream_push truncates a longer tail
     * rather than failing, so a construct whose look-ahead grows with
     * the input would silently lose bytes. That is the same constraint
     * that makes "---" match exactly three hyphens; the difference is
     * only that a '>' run CAN be eaten incrementally, so
     * here it costs a field instead of a deliberately missed case. */
    int  quote;
} LZStream;

/* Emit UTF-8 instead of GBK, for a console or a redirected stream that
   is not GBK. Process-wide and set once at startup - see stream.c for
   why it is not per-stream. Off (GBK) unless set. */
void lz_stream_utf8_out(int on);

void lz_stream_init(LZStream *s);

/* Feed UTF-8. Emits zero or more runs through `sink`. Whatever cannot be
 * decided yet is held for the next call. */
void lz_stream_push(LZStream *s, const char *utf8, int len,
                    LZStreamSink sink, void *ud);

/* No more input is coming: flush the held tail as literal text, THEN
 * reset the whole style mask and after_tag to 0 - an unclosed
 * "**"/"*"/"`" at the end of one turn must not leak bold/italic/
 * monospace into the next. Call it when generation ends, or the last
 * partial tag of a reply is never shown at all - and because it is
 * also the reset, every turn MUST end with a call to this, live or
 * replayed, or the state genuinely does carry over. */
void lz_stream_end(LZStream *s, LZStreamSink sink, void *ud);

/* The THINK bit specifically - kept as its own bool-shaped getter/
 * setter rather than exposing the raw mask, because it is the one bit
 * a caller needs to seed BEFORE any content of a turn arrives (see
 * lz_stream_set_in_think below); the other three always start at 0
 * for a fresh reply and never need seeding. */
int lz_stream_in_think(const LZStream *s);

/* Seed the starting state for a FRESH reply - call once, right before
 * the first byte of a new turn's generation is pushed (after any UI
 * chrome like a speaker label, which must stay plain text regardless).
 * Needed because a thinking-enabled prompt already ends "<think>\n" -
 * the model resumes INSIDE the block and the reply stream itself never
 * contains an opening tag, only the closing one - so a parser that
 * always started at 0 would never find anything to match the close
 * against and the entire reasoning span would render as ordinary text.
 * The caller derives the right value with
 * lz_chat_gen_prompt_starts_in_think(enable_thinking) (src/chat.h) -
 * NOT a second enable_thinking check here, which is exactly the kind
 * of duplicated decision that file's own comments warn drifts
 * silently. Touches only the THINK bit, not the whole style mask and
 * not n_pend: lz_stream_end resets the whole style mask (THINK
 * included) to 0 before a new turn can start, and this call re-seeds
 * the THINK bit. */
void lz_stream_set_in_think(LZStream *s, int in_think);

#endif
