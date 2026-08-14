/* Incremental display assembly: chunked UTF-8 in, style-tagged GBK out.
 * See stream.h for why the three boundary problems live in one object.
 */
#include <string.h>

#include "stream.h"
#include "gbk.h"

static const char TAG_OPEN[]  = "<think>";
static const char TAG_CLOSE[] = "</think>";
#define TAG_OPEN_LEN  7
#define TAG_CLOSE_LEN 8

/* The hold must be able to carry the longest incomplete marker, which
 * is one byte short of the longest one (</think>, 8 bytes) - the
 * Markdown markers (at most 2 bytes, "**") never drive this number. A
 * UTF-8 sequence needs at most 3 held bytes, so the think tag is still
 * what sizes this. */
typedef char lz_stream_pend_is_big_enough[
    LZ_STREAM_PEND >= TAG_CLOSE_LEN - 1 ? 1 : -1];

/* Sized above the run cap on purpose. The cap is checked at character
 * boundaries, so a run overshoots it by up to one 4-byte character, and
 * the tag emit adds 8 more. Sizing this exactly at the cap would make
 * lz_gbk_from_utf8 truncate - safely, but with characters missing from
 * the display and nothing saying so. */
static char g_out[LZ_STREAM_RUN_MAX + 16];

/* What "---" turns into: a short run of box-drawing characters, "a
 * string that fits the width without wrapping and without overfilling
 * the line" (the user's own wording, in Chinese, translated here -
 * iron law seven).
 *
 * U+2500 BOX DRAWINGS LIGHT HORIZONTAL, written in UTF-8 and converted
 * by lz_gbk_from_utf8 like every other substitution here, so the
 * encoding table stays the single authority. Its GBK code (A9A4) was
 * measured against the engine's own converter with U+4E2D as a control
 * - a measured value, not a fresh guess.
 *
 * SIXTEEN of them, and the number comes from the narrowest window this
 * program allows rather than from how it looks on this desk. A GBK
 * full-width glyph at 9pt/96dpi is 12px, so this is 192px, inside
 * LZ_GUI_MIN_CW (360) with the side panel and a scrollbar still to pay
 * for - it cannot wrap even at the minimum size, which is the half of
 * the user's wording that CAN be checked. The other half - the
 * appropriate-width judgement - is appearance and has no automated
 * gate: if it reads short or long on screen, this constant is the
 * knob. */
#define RULE_CH "\xE2\x94\x80"
static const char RULE_RUN[] =
    RULE_CH RULE_CH RULE_CH RULE_CH RULE_CH RULE_CH RULE_CH RULE_CH
    RULE_CH RULE_CH RULE_CH RULE_CH RULE_CH RULE_CH RULE_CH RULE_CH;
/* The literal above and LZ_STREAM_RULE_CHARS have to agree, and a
 * comment saying so is not a gate. Three UTF-8 bytes per character. */
typedef char lz_rule_run_matches_the_count[
    (int)sizeof RULE_RUN - 1 == LZ_STREAM_RULE_CHARS * 3 ? 1 : -1];

/* Bytes in the UTF-8 sequence starting with c0. An invalid lead byte
 * counts as 1 so recovery advances, which is what lz_utf8_decode and
 * lz_gbk_from_utf8 both do - three places agreeing on resynchronisation
 * is what keeps a bad byte from shifting everything after it. */
static int seq_len(unsigned char c0) {
    if (c0 < 0x80) return 1;
    if ((c0 & 0xE0) == 0xC0) return 2;
    if ((c0 & 0xF0) == 0xE0) return 3;
    if ((c0 & 0xF8) == 0xF0) return 4;
    return 1;
}

static void emit(const char *utf8, int len, int style,
                 LZStreamSink sink, void *ud) {
    int n;
    if (len <= 0 || !sink) return;
    /* GBK is never longer than the UTF-8 it came from, and the caller
       below caps a run at LZ_STREAM_RUN_MAX input bytes, so the output
       always fits. */
    n = lz_gbk_from_utf8(utf8, len, g_out, (int)sizeof g_out, NULL);
    if (n > (int)sizeof g_out - 1) n = (int)sizeof g_out - 1;
    sink(ud, g_out, n, style);
}

/* Returns bytes consumed. Whatever is left is a tail the caller must
 * hold (or, with final != 0, emit literally). */
static int scan_bytes(LZStream *s, const char *b, int n, int final,
                      LZStreamSink sink, void *ud) {
    int i = 0, run = 0;

    /* Resolve a trailing-newline question left over from a tag consumed
     * in a PREVIOUS call - see the tag branch below for why this exists
     * at all. Checked first, before anything else in this
     * buffer, because the tag itself is already fully applied - what is
     * outstanding is only "does byte 0 of THIS buffer continue that
     * tag's own line ending". */
    if (s->after_tag) {
        s->after_tag = 0;
        if (n > 0 && b[0] == '\n') {
            i = 1;
        } else if (n > 0 && b[0] == '\r') {
            if (n > 1) {
                if (b[1] == '\n') i = 2;
                /* else: a bare \r is not a line ending - left alone,
                   falls into the ordinary scan below as content. */
            } else if (!final) {
                /* Still only the \r visible, even after a whole extra
                   call. Keep watching rather than guess wrong - the
                   SAME "wait for more" reasoning the tag match itself
                   uses just below. */
                s->after_tag = 1;
                return 0;
            }
            /* final && only the \r ever arrived: nothing can complete
               it, so it stands as ordinary content. */
        }
        run = i;
    }

    while (i < n) {
        /* Checked BEFORE advancing, so the run can overshoot the cap by
           at most one character rather than by a whole one. */
        if (i - run >= LZ_STREAM_RUN_MAX) {
            emit(b + run, i - run, s->style, sink, ud);
            run = i;
        }
        /* The rest of a line that belongs to a marker rather than to the
           content: an opening fence's info string ("```python"), or a
           table's "|---|---|" separator row. Eaten up to and including
           its newline, which may not arrive until a later chunk - hence
           the field. */
        if (s->skip_line) {
            emit(b + run, i - run, s->style, sink, ud);
            if (b[i] == '\n') s->skip_line = 0;
            i += 1;
            run = i;
            continue;
        }
        /* The rest of a blockquote's leading marker run: more '>', then
           at most ONE space, and the state dies on the first byte that
           is neither. See stream.h's `quote` for why this is a field and
           not a count - an unbounded look-ahead cannot be held.

           A newline lands here too, on a line like ">\n": it is not '>'
           and not ' ', so the state clears and the byte falls through to
           the ordinary path, keeping the blank line the model wrote. */
        if (s->quote) {
            if (b[i] == '>') {
                emit(b + run, i - run, s->style, sink, ud);
                i += 1;
                run = i;
                continue;
            }
            s->quote = 0;
            if (b[i] == ' ') {
                emit(b + run, i - run, s->style, sink, ud);
                i += 1;
                run = i;
                continue;
            }
            /* Neither: this byte is content, handled below as usual. */
        }
        /* Blockquote: every leading '>' on a line, plus one
           space after them, is consumed; NO style bit and no indent -
           the quote gets no formatting of its own at all.
           Only the first '>' is recognised here, at the line start; the
           block above carries the rest, including across a chunk
           boundary. */
        if (!s->fence && b[i] == '>' &&
            ((i == 0) ? s->at_bol : (b[i - 1] == '\n'))) {
            emit(b + run, i - run, s->style, sink, ud);
            s->quote = 1;
            i += 1;
            run = i;
            continue;
        }
        /* Heading: "# ".."###### " at a line start. The
           hashes and the ONE space after them are consumed; the level
           field stays on to the end of the line and INCLUDING that
           line's '\n'.

           Including the newline is not tidiness. A RichEdit line takes
           its height from the tallest character on it, the line-ending
           character included - leave the '\n' unstyled and a heading is
           a row of large text crammed into a normal line's leading.

           Up to 7 bytes of look-ahead ("###### "), well inside
           LZ_STREAM_PEND's 8. */
        if (!s->fence && b[i] == '#' &&
            ((i == 0) ? s->at_bol : (b[i - 1] == '\n'))) {
            int k = 0;
            while (i + k < n && k < 6 && b[i + k] == '#') k++;
            if (i + k >= n && k < 6 && !final) {
                emit(b + run, i - run, s->style, sink, ud);
                return i;                    /* "##" may still grow */
            }
            if (i + k < n && b[i + k] == ' ') {
                int lvl = k > 3 ? 3 : k;     /* clamped, see stream.h */
                emit(b + run, i - run, s->style, sink, ud);
                s->style = (s->style & ~LZ_STYLE_H_MASK) |
                           (lvl << LZ_STYLE_H_SHIFT);
                i += k + 1;
                run = i;
                continue;
            }
            if (i + k >= n && !final) {      /* space not here yet */
                emit(b + run, i - run, s->style, sink, ud);
                return i;
            }
            /* Not a heading ("#tag"): falls through as ordinary text. */
        }
        /* The LINE-SCOPED bits - the heading level and the table row -
           die with their own line ending. Checked after the byte is
           known to be a newline and before it is emitted, so the '\n'
           still carries them.

           One branch for both, because they need it for the same
           reason at two different layers: a heading's '\n' sets the
           line's HEIGHT (see above), and a table row's '\n' closes the
           PARAGRAPH its tab stops belong to. Emitting that newline
           without the bit would make gui/main.c set the plain
           paragraph format while the insertion point is still inside
           the table's own paragraph, wiping the tab stops it had just
           set. */
        if ((s->style & (LZ_STYLE_H_MASK | LZ_STYLE_TABLE |
                         LZ_STYLE_BULLET)) && b[i] == '\n') {
            emit(b + run, i - run + 1, s->style, sink, ud);
            s->style &= ~(LZ_STYLE_H_MASK | LZ_STYLE_TABLE |
                          LZ_STYLE_BULLET);
            i += 1;
            run = i;
            continue;
        }
        /* Inside a table row: an inner '|' becomes a tab, the row's own
           closing '|' is consumed. Which one it is takes exactly one
           byte of look-ahead - a '|' with a line ending or the end of
           the input after it is the closing one. */
        if ((s->style & LZ_STYLE_TABLE) && b[i] == '|') {
            int last;
            if (i + 1 >= n) {
                if (!final) {
                    emit(b + run, i - run, s->style, sink, ud);
                    return i;
                }
                last = 1;                /* nothing follows it at all */
            } else {
                last = b[i + 1] == '\n' || b[i + 1] == '\r';
            }
            emit(b + run, i - run, s->style, sink, ud);
            if (!last) emit("\t", 1, s->style, sink, ud);
            i += 1;
            run = i;
            continue;
        }
        /* Table row: a '|' at a line start. The outer pipes
           go, the inner ones become tabs, and the whole row carries the
           TABLE bit so gui/main.c can give that paragraph tab stops.

           The "|---|---|" separator row is eaten whole - and recognised
           from its FIRST cell only, within 6 bytes, rather than by
           scanning to the end of the line. Scanning to the line end is
           the obvious reading of "a row of dashes" and it is the same
           unbounded look-ahead the "---" rule above refuses, for the
           same reason. Three hyphens are required (after at most one
           space and one alignment colon) rather than one, so that a
           real first cell of "-5" is not mistaken for a separator; a
           cell that genuinely starts "---" is the accepted miss. */
        if (!s->fence && b[i] == '|' &&
            ((i == 0) ? s->at_bol : (b[i - 1] == '\n'))) {
            int j = i + 1;
            while (j < n && j - i <= 2 && (b[j] == ' ' || b[j] == ':')) j++;
            if (j + 3 > n && !final) {
                emit(b + run, i - run, s->style, sink, ud);
                return i;                /* not enough to tell yet */
            }
            emit(b + run, i - run, s->style, sink, ud);
            if (j + 3 <= n && b[j] == '-' && b[j + 1] == '-' &&
                b[j + 2] == '-') {
                s->skip_line = 1;        /* the rest of the row goes too */
            } else {
                s->style |= LZ_STYLE_TABLE;
            }
            i += 1;
            run = i;
            continue;
        }
        /* Horizontal rule: EXACTLY three hyphens, with a line ending or
           the end of the input right after them.

           The three hyphens are REPLACED by a short run of U+2500, they
           are not eaten - eating the whole line would throw away the
           paragraph break the model asked for, and leave nothing on
           screen at all. Only the hyphens are consumed here; the line's
           own '\n' flows through as ordinary content, which is what
           keeps the break.

           CommonMark says "three OR MORE", and that is deliberately not
           what this does. "Or more" means the look-ahead is as long as
           the run of hyphens, which has no upper bound, while the hold
           is 8 bytes and lz_stream_push truncates past it (stream.h's
           `quote` has the same note from the other side). Exactly-three
           pins the look-ahead at 5 bytes - the three, the byte that
           proves there is no fourth, and the '\n' of a "\r\n". "----"
           therefore prints literally: a missed case, chosen over an
           unbounded hold.

           Checked before the bullet branch below only for reading
           order; the two cannot both match, since a bullet needs a
           space where a rule needs a second hyphen. */
        if (!s->fence && b[i] == '-' &&
            ((i == 0) ? s->at_bol : (b[i - 1] == '\n'))) {
            int avail = n - i, k = 0, rule = 0;
            while (k < avail && k < 4 && b[i + k] == '-') k++;
            if (k < 3 && k == avail && !final) {
                emit(b + run, i - run, s->style, sink, ud);
                return i;                    /* "-" or "--" may still grow */
            }
            if (k == 3) {
                if (avail == 3) {
                    if (final) rule = 1;     /* end of input ends the line */
                    else {
                        emit(b + run, i - run, s->style, sink, ud);
                        return i;            /* byte 4 decides, not here yet */
                    }
                } else if (b[i + 3] == '\n') {
                    rule = 1;
                } else if (b[i + 3] == '\r') {
                    /* Only as part of "\r\n". A bare '\r' is not a
                       terminator: the close-tag branch consumes "\r\n"
                       but leaves a bare '\r'
                       (test_close_tag_bare_cr_is_not_eaten), and at_bol
                       likewise only ever starts a line after '\n'. A
                       "---\r" with nothing after it is therefore not a
                       rule and prints as it arrived. */
                    if (avail >= 5) rule = b[i + 4] == '\n';
                    else if (!final) {
                        emit(b + run, i - run, s->style, sink, ud);
                        return i;
                    }
                }
            }
            if (rule) {
                emit(b + run, i - run, s->style, sink, ud);
                emit(RULE_RUN, (int)sizeof RULE_RUN - 1, s->style, sink, ud);
                i += 3;                      /* the hyphens only */
                run = i;
                continue;
            }
            /* Four or more hyphens, or something other than a line
               ending after three: ordinary text, falls through. */
        }
        /* Bullet list: "- ", "* " or "+ " at a line start becomes "● ".
           A TEXT substitution, not a style bit and not an indent -
           indenting needs PARAFORMAT, and the table's tab stops are the
           only PARAFORMAT this program sends.

           U+25CF BLACK CIRCLE, written in UTF-8 (E2 97 8F) and
           converted by lz_gbk_from_utf8 like everything else, so the
           encoding table stays the one authority. Of the obvious
           alternatives, U+2022 "•" does not exist in GBK at all and
           comes out "?"; U+00B7 "·" is A1A4; U+25CF "●" is A1F1.

           Checked before the '*' branch, or "* item" would toggle italic
           instead. "**bold**" at a line start is unaffected: the second
           byte is '*', not a space. */
        if (!s->fence && (b[i] == '-' || b[i] == '*' || b[i] == '+') &&
            ((i == 0) ? s->at_bol : (b[i - 1] == '\n'))) {
            if (n - i < 2) {
                if (!final) {            /* "- " may still be completing */
                    emit(b + run, i - run, s->style, sink, ud);
                    return i;
                }
            } else if (b[i + 1] == ' ') {
                emit(b + run, i - run, s->style, sink, ud);
                /* The bit goes on BEFORE the marker is emitted, so the
                   marker itself carries it too: gui/main.c reads it off
                   whichever run reaches the paragraph first, and during
                   streaming that is this one. */
                s->style |= LZ_STYLE_BULLET;
                emit("\xE2\x97\x8F ", 4, s->style, sink, ud);
                i += 2;
                run = i;
                continue;
            }
        }
        if (b[i] == '<') {
            /* Only the tag that can actually occur here is considered.
               A stray "</think>" in ordinary text would otherwise turn
               the entire rest of the reply grey, and a model that emits
               one is exactly the model this front end is for. */
            int in_think = (s->style & LZ_STYLE_THINK) != 0;
            const char *want = in_think ? TAG_CLOSE : TAG_OPEN;
            int wlen = in_think ? TAG_CLOSE_LEN : TAG_OPEN_LEN;
            int avail = n - i;
            int cmp = avail < wlen ? avail : wlen;
            if (memcmp(b + i, want, (size_t)cmp) == 0) {
                if (avail >= wlen) {
                    /* Complete tag. The text before it belongs to the
                       old state and is emitted; the tag's own bytes
                       are consumed to flip the state and then dropped -
                       tags are not displayed at all. */
                    emit(b + run, i - run, s->style, sink, ud);
                    s->style ^= LZ_STYLE_THINK;
                    i += wlen;
                    /* The tag's OWN line ending, consumed too, not
                       just the tag: a model that writes the tag on its
                       own line - "reasoning\n</think>\n\nanswer" - has
                       one line-ending that belongs to the TAG, and
                       deleting only the tag would leave it behind as an
                       EXTRA blank line no one asked for. Only consumed
                       when it is genuinely there - "...</think>answer"
                       (tag not on its own line) leaves `answer`
                       untouched, byte for byte. An undecidable case
                       (only a bare \r visible, more input might still
                       be coming) is deferred to s->after_tag above
                       rather than guessed. */
                    if (i < n && b[i] == '\n') {
                        i += 1;
                    } else if (i < n && b[i] == '\r') {
                        if (i + 1 < n) {
                            if (b[i + 1] == '\n') i += 2;
                            /* else: bare \r, not a line ending - left
                               as ordinary content. */
                        } else if (!final) {
                            s->after_tag = 1;
                            run = i;
                            return i;   /* the \r itself stays held */
                        }
                        /* final && only the \r visible: stands as
                           ordinary content, nothing left to wait for. */
                    } else if (i == n && !final) {
                        /* Nothing at all follows the tag in this
                           buffer yet - the very next byte, whenever it
                           arrives, is what answers the question. */
                        s->after_tag = 1;
                        run = i;
                        return i;
                    }
                    run = i;
                    continue;
                }
                if (!final) {
                    /* A prefix that could still become a tag. Stop here
                       and hold from i. */
                    emit(b + run, i - run, s->style, sink, ud);
                    return i;
                }
                /* final: it will never complete, so it is literal text */
            }
            /* Not a tag: fall through and treat '<' as a normal byte. */
        } else if (b[i] == '*' && !s->fence) {   /* a fence is verbatim */
            /* "**" (bold) is checked before a lone "*" (italic) -
               otherwise "**x**" would read as two adjacent italic
               toggles rather than one bold span. Both are consumed,
               never displayed - no renderer echoes the asterisks in
               "**bold**". */
            int avail = n - i;
            if (avail >= 2) {
                /* Enough of the chunk is here to know for certain
                   which of the two this is - no ambiguity left. */
                emit(b + run, i - run, s->style, sink, ud);
                if (b[i + 1] == '*') { s->style ^= LZ_STYLE_BOLD; i += 2; }
                else                 { s->style ^= LZ_STYLE_ITALIC; i += 1; }
                run = i;
                continue;
            }
            /* Exactly one byte left in this call - it could still turn
               into "**" once more input arrives. */
            if (!final) {
                emit(b + run, i - run, s->style, sink, ud);
                return i;
            }
            /* final: never confirmed either way. The spec's own "an
               unclosed marker is plain text" - '*' is common in plain
               text and in math, so an asterisk nobody ever got to
               finish deciding about prints literally rather than
               silently toggling italic on with nothing left to show
               for it. Falls through to the
               ordinary-byte path below, exactly like an unfinished
               <think tag does just above. */
        } else if (b[i] == '`') {
            /* THREE backticks is a fence, one is inline code, and the
               difference cannot be told from the first byte - the same
               chunk-boundary problem '*' has, one byte deeper.

               Two available and not final is still undecided: "``" can
               become "```". LZ_STREAM_PEND is 8, so holding two is
               free. */
            /* A fence is only a fence at a line start. A backtick in
               the MIDDLE of a line is still decided from one byte with
               no look-ahead. */
            int bol = (i == 0) ? s->at_bol : (b[i - 1] == '\n');
            int avail = n - i;
            int run3 = bol && avail >= 3 &&
                       b[i + 1] == '`' && b[i + 2] == '`';
            if (!run3 && bol && avail < 3 && !final) {
                emit(b + run, i - run, s->style, sink, ud);
                return i;
            }
            emit(b + run, i - run, s->style, sink, ud);
            if (run3) {
                s->fence = !s->fence;
                /* BOTH directions eat the rest of their own line, the
                   closing fence included: a code block must not be
                   followed by a blank line nobody asked for, so the ```
                   and its line ending go together. The opening fence
                   eats its info string ("```python") the same way.

                   This is the same rule <think>/</think> follows
                   (stream.h: A TAG'S OWN TRAILING NEWLINE IS PART OF
                   THE TAG). A fence line belongs to the marker, so its
                   terminator does too - doing it on one side only is
                   not a policy, it is an asymmetry. */
                s->skip_line = 1;
                if (s->fence) s->style |= LZ_STYLE_CODE;
                else          s->style &= ~LZ_STYLE_CODE;
                i += 3;
            } else {
                /* Inline code, and inside a fence a lone backtick is
                   ordinary text - a fenced block is verbatim. */
                if (s->fence) { i += 1; run = i - 1; continue; }
                s->style ^= LZ_STYLE_CODE;
                i += 1;
            }
            run = i;
            continue;
        }
        {
            int L = seq_len((unsigned char)b[i]);
            if (i + L > n) {
                if (!final) {
                    emit(b + run, i - run, s->style, sink, ud);
                    return i;
                }
                /* final: hand the truncated sequence to the converter,
                   which turns it into one fallback character. */
                L = n - i;
            }
            i += L;
        }
    }
    emit(b + run, i - run, s->style, sink, ud);
    return i;
}

/* "Is the next byte at the start of a line" - the shared prerequisite
 * every block-level construct needs, paid once here rather than once
 * per construct.
 *
 * Carried in the state because the answer for byte 0 of a chunk depends
 * on the chunk before it. Within a chunk it is read off the buffer
 * (b[i-1] == '\n'), which is deliberately NOT the same as "the last byte
 * EMITTED was a newline": a consumed marker or a consumed </think>\n
 * still leaves its bytes in the buffer, and after a consumed newline the
 * next byte genuinely is at a line start. */
static int scan(LZStream *s, const char *b, int n, int final,
                LZStreamSink sink, void *ud) {
    int used = scan_bytes(s, b, n, final, sink, ud);
    if (used > 0) s->at_bol = (b[used - 1] == '\n');
    return used;
}

void lz_stream_init(LZStream *s) {
    if (!s) return;
    s->n_pend = 0;
    s->style = 0;
    s->after_tag = 0;
    s->fence = 0;
    s->skip_line = 0;
    s->quote = 0;
    s->at_bol = 1;      /* the first byte of a reply is at a line start */
    s->style &= ~(LZ_STYLE_H_MASK | LZ_STYLE_TABLE);
}

void lz_stream_push(LZStream *s, const char *utf8, int len,
                    LZStreamSink sink, void *ud) {
    if (!s || !utf8 || len <= 0) return;

    if (s->n_pend > 0) {
        /* Held bytes plus a short head of the new chunk: enough to
           settle any tag or character, and bounded so this stays a
           fixed-size local rather than a copy of the whole chunk. */
        char tmp[LZ_STREAM_PEND * 2];
        int take = len < LZ_STREAM_PEND ? len : LZ_STREAM_PEND;
        int have = s->n_pend + take, used;
        memcpy(tmp, s->pend, (size_t)s->n_pend);
        memcpy(tmp + s->n_pend, utf8, (size_t)take);
        used = scan(s, tmp, have, 0, sink, ud);
        if (used < s->n_pend) {
            /* Reachable only when the whole chunk went into tmp, i.e.
               take == len: given a full LZ_STREAM_PEND-byte head, every
               leading token is decidable, so scan cannot stop inside the
               held bytes. Nothing is dropped by returning here.
               `have - used` is at most one token short of complete, so
               it fits pend. */
            memmove(s->pend, tmp + used, (size_t)(have - used));
            s->n_pend = have - used;
            return;
        }
        utf8 += used - s->n_pend;
        len -= used - s->n_pend;
        s->n_pend = 0;
        if (len <= 0) return;
    }

    {
        int used = scan(s, utf8, len, 0, sink, ud);
        int left = len - used;
        if (left > 0) {
            /* scan only ever stops inside an unfinished tag or an
               unfinished character, both shorter than the buffer. */
            if (left > LZ_STREAM_PEND) left = LZ_STREAM_PEND;
            memcpy(s->pend, utf8 + len - left, (size_t)left);
            s->n_pend = left;
        }
    }
}

void lz_stream_end(LZStream *s, LZStreamSink sink, void *ud) {
    if (!s) return;
    if (s->n_pend > 0) {
        scan(s, s->pend, s->n_pend, 1, sink, ud);
        s->n_pend = 0;
    }
    /* ALL per-turn state resets here, not just n_pend: an unclosed
     * "**" or "`" in one turn must not leak bold/monospace into the
     * NEXT turn's entire reply. The whole mask is reset
     * unconditionally so no bit is left depending on a downstream
     * caller to happen to overwrite it. lz_stream_set_in_think still
     * runs afterward for a THINKING turn and is unaffected - it
     * OR/AND-masks a single bit, not a whole-mask assignment, so
     * seeding after this reset is exactly the same call it always was.
     *
     * after_tag reset too: it is the OTHER piece of state that
     * survives across lz_stream_push calls (stream.h's own comment on
     * the field), so leaving it set here would carry a "watching for
     * a newline" decision from THIS turn's last tag into bytes that
     * are about to belong to a completely different one.
     *
     * AFTER the flush above, not before: the scan() call just above
     * still has to see whatever style THIS turn was actually in while
     * it settles the held tail - resetting first would relabel that
     * tail's own content with the wrong style. */
    s->style = 0;
    s->after_tag = 0;
    /* An unclosed fence dies with the turn, for the same reason an
       unclosed <think> does: the next reply starts from a known state,
       not from whatever the last one forgot to close. */
    s->fence = 0;
    s->skip_line = 0;
    /* Half-eaten blockquote marker included: like `fence` and
       `after_tag`, it is a question this turn asked and this turn has
       to stop asking. */
    s->quote = 0;
    s->at_bol = 1;      /* the first byte of a reply is at a line start */
    s->style &= ~(LZ_STYLE_H_MASK | LZ_STYLE_TABLE);
}

int lz_stream_in_think(const LZStream *s) {
    return s ? (s->style & LZ_STYLE_THINK) != 0 : 0;
}

void lz_stream_set_in_think(LZStream *s, int in_think) {
    if (!s) return;
    if (in_think) s->style |= LZ_STYLE_THINK;
    else          s->style &= ~LZ_STYLE_THINK;
}
