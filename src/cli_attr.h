#ifndef LZ_CLI_ATTR_H
#define LZ_CLI_ATTR_H

/* Console text attributes for the CLI's Markdown styles.
 *
 * CLI-ONLY, outside $(ENG), the way src/cli_argv.c is - console I/O
 * stays inside CLI files and the engine library keeps none of it.
 *
 * Per platform, because there is no one mechanism that works on the
 * target family: ANSI escapes need ANSI.SYS on DOS and are not
 * interpreted at all by the Win9x console, so a single ANSI path would
 * fail on both of the machines this ships to, and fail by printing
 * escape bytes rather than by doing nothing.
 *
 *   DOS       _settextcolor/_outtext (graph.lib)
 *   Win32     SetConsoleTextAttribute
 *   other     nothing; the styles are dropped and bytes go out plain
 *
 * Attributes are set only when the style CHANGES. Style transitions are
 * far rarer than characters and the per-token path is the hot one. */

/* auto|on|off. `auto` means "on when stdout is a console", which is the
   only setting that can be right for both an interactive run and a
   redirected one. Returns the name actually in effect, or NULL if the
   string is unrecognised - the caller prints what it GOT, not what it
   asked for, the same discipline --prefetch already follows. */
const char *lz_attr_mode(const char *mode);

/* Write `n` bytes in `style` (an LZ_STYLE_* mask from
   common/stream.h). The attribute is set only when the style actually
   changes.
   THE WRITE lives here rather than at the call site because on DOS the
   two cannot be separated: _settextcolor colours what _outtext emits
   and has no effect on stdout, so a caller that set the attribute and
   then used fwrite would get correct-looking code that changes nothing
   on screen. Everywhere else this is fwrite with the attribute set
   first. */
void lz_attr_write(const char *bytes, int n, int style);

/* Back to the console's ordinary text. Called before the program exits
   and after each reply, or the shell inherits whatever colour the last
   run left behind - which on DOS persists until something else sets it. */
void lz_attr_reset(void);

#endif
