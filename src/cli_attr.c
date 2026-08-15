/* Console attributes for the CLI's Markdown styles. See cli_attr.h. */

#include <stdio.h>
#include <string.h>

#if defined(__DOS__)
#include <io.h>      /* isatty - see stdout_is_console */
#include <i86.h>   /* int386 - see dos_putc */

/* BIOS INT 10h, NOT graph.lib.
   lib386/dos/graph.lib does not link against this build at all: its
   objects pull in _Extender, _ExtenderRealModeSelector and _STACKLOW,
   a runtime this DOS-extender target does not have. (Its symbols are
   also undecorated where __watcall would want a trailing underscore,
   which is what the first attempt tripped over - but the runtime
   mismatch is the one that settles it.)
   int386 is in the ordinary C library and always links.
   The per-character interrupt was rejected on cost earlier and that was
   wrong for THIS workload: generation puts out a few tokens a second,
   so this path sees tens of characters a second, not a screen refresh.
   The cost is unmeasurable next to one forward pass. */
#elif defined(_WIN32)
#include <windows.h>
#endif

#include "cli_attr.h"
#include "stream.h"

/* The IBM text attributes this maps onto. 0x07 is the console's own
   ordinary light grey, which is why bold has to be 0x0F white rather
   than a "bright" bit applied to something already bright - on this
   palette plain text IS the light one. */
#define A_TEXT   0x07
#define A_BOLD   0x0F           /* white */
#define A_THINK  0x08           /* dark grey, the GUI's own think colour */
#define A_CODE   0x0B           /* light cyan */
#define A_ITALIC 0x0D           /* light magenta - text mode has no slant */

static int g_on;                /* attributes are being emitted */
#if defined(__DOS__)
static unsigned char g_dos_attr = 0x07;

/* One character through the BIOS, with the current attribute.
 *
 * AH=09h writes the character AND its attribute but does not move the
 * cursor; AH=0Eh moves the cursor and scrolls but in text mode ignores
 * the attribute in BL. Neither alone does the job, so this is 09h to
 * place the coloured cell followed by an explicit cursor advance -
 * which is also where the newline, the wrap and the scroll have to be
 * handled, since nothing else is doing it any more.
 *
 * Page 0 throughout: this program never switches video pages, and
 * asking BIOS for the active one per character would double the
 * interrupt count for an answer that cannot change. */
/* Screen geometry and the active page, asked ONCE.
 *
 * Neither changes while this program runs - it never sets a video mode -
 * and asking per character cost three interrupts for answers that could
 * not move. More importantly it was asking for only two of them and
 * assuming the rest.
 *
 * Rows come from AX=1130h, which returns DL = rows - 1 (checked against
 * the interrupt list, not remembered: the BIOS data byte at 0040:0084
 * it reads is itself stored as rows-1). That call is EGA and later, and
 * on CGA/MDA it returns nothing useful, so an implausible answer falls
 * back to 25 - which is what those adapters have anyway.
 *
 * The active page comes from AH=0Fh's BH. It was hardcoded to 0 before,
 * which is right until something else selects a page and then silently
 * writes every character to the wrong one. */
static int g_cols = 80, g_rows = 25, g_page;
static int g_probed;

static void dos_probe(void) {
    union REGS r;
    if (g_probed) return;
    g_probed = 1;

    r.h.ah = 0x0F;
    int386(0x10, &r, &r);
    g_cols = r.h.ah ? r.h.ah : 80;
    g_page = r.h.bh;

    /* AH=11h AL=30h, BH=0 (the INT 1Fh pointer - any specifier returns
       the same DL, and 0 is the one every adapter accepts). */
    r.h.ah = 0x11; r.h.al = 0x30; r.h.bh = 0;
    int386(0x10, &r, &r);
    g_rows = r.h.dl + 1;
    if (g_rows < 12 || g_rows > 60) g_rows = 25;
}

/* One character through the BIOS, with the current attribute.
 *
 * AH=09h writes the character AND its attribute but does not move the
 * cursor; AH=0Eh moves the cursor and scrolls but in text mode ignores
 * the attribute in BL. Neither alone does the job, so this is 09h to
 * place the coloured cell followed by an explicit cursor advance -
 * which is also where the newline, the wrap and the scroll have to be
 * handled, since nothing else is doing it any more. */
static void dos_putc(char ch) {
    union REGS r;
    int row, col;

    dos_probe();

    r.h.ah = 0x03; r.h.bh = (unsigned char)g_page;
    int386(0x10, &r, &r);
    row = r.h.dh; col = r.h.dl;

    if (ch == '\r') { col = 0; }
    else if (ch == '\n') { col = 0; row++; }
    else if (ch == '\b') { if (col > 0) col--; }
    else if (ch == '\t') { col = (col + 8) & ~7; }
    else {
        r.h.ah = 0x09; r.h.al = (unsigned char)ch;
        r.h.bh = (unsigned char)g_page; r.h.bl = g_dos_attr;
        r.x.ecx = 1;                     /* AH=09h requires CX >= 1 */
        int386(0x10, &r, &r);
        col++;
    }
    if (col >= g_cols) { col = 0; row++; }
    if (row >= g_rows) {
        /* Scroll one line, and the blanked line takes the CURRENT
           attribute - otherwise the bottom row comes back in whatever
           the previous scroll left and the colour breaks at the fold.
           Page 0 is not a choice here: the interrupt list records that
           this call scrolls page 0 exclusively on EGA/MCGA/VGA, so a
           program on another page must not rely on it. */
        r.h.ah = 0x06; r.h.al = 1; r.h.bh = g_dos_attr;
        r.h.ch = 0; r.h.cl = 0;
        r.h.dh = (unsigned char)(g_rows - 1);
        r.h.dl = (unsigned char)(g_cols - 1);
        int386(0x10, &r, &r);
        row = g_rows - 1;
    }
    r.h.ah = 0x02; r.h.bh = (unsigned char)g_page;
    r.h.dh = (unsigned char)row; r.h.dl = (unsigned char)col;
    int386(0x10, &r, &r);
}
#endif
static int g_cur = -1;          /* attribute currently set, -1 = unknown */

static int attr_for(int style) {
    /* Checked most specific first: a heading that is also bold is a
       heading, and think wins over everything because a whole reasoning
       block reading as body text is the confusion worth avoiding. */
    if (style & LZ_STYLE_THINK)  return A_THINK;
    if (style & LZ_STYLE_H_MASK) return A_BOLD;
    if (style & LZ_STYLE_CODE)   return A_CODE;
    if (style & LZ_STYLE_BOLD)   return A_BOLD;
    if (style & LZ_STYLE_ITALIC) return A_ITALIC;
    return A_TEXT;
}

static int stdout_is_console(void) {
#if defined(__DOS__)
    /* isatty, because on DOS the attribute path also takes over the
       WRITING (see lz_attr_write) and _outtext reaches the screen no
       matter where stdout was pointed. A redirected run therefore has
       to stay on fwrite entirely. */
    return isatty(fileno(stdout)) ? 1 : 0;
#elif defined(_WIN32)
    DWORD mode;
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE || h == NULL) return 0;
    return GetConsoleMode(h, &mode) ? 1 : 0;
#else
    return 0;
#endif
}

const char *lz_attr_mode(const char *mode) {
    if (!mode) return NULL;
    if (strcmp(mode, "off") == 0)       g_on = 0;
    else if (strcmp(mode, "on") == 0)   g_on = stdout_is_console();
    else if (strcmp(mode, "auto") == 0) g_on = stdout_is_console();
    else return NULL;
    g_cur = -1;
    /* "on" that could not be honoured reports itself as off. A run
       labelled with what it asked for rather than what it got is the
       shape iron law four keeps catching. */
    return g_on ? "on" : "off";
}

static void attr_apply(int style) {
    int a = attr_for(style);
    if (a == g_cur) return;
    g_cur = a;
#if defined(__DOS__)
    g_dos_attr = (unsigned char)(a & 0xFF);
#elif defined(_WIN32)
    {
        HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        if (h && h != INVALID_HANDLE_VALUE)
            SetConsoleTextAttribute(h, (WORD)a);
    }
#else
    (void)a;
#endif
}

void lz_attr_write(const char *bytes, int n, int style) {
    if (n <= 0) return;
    if (!g_on) { fwrite(bytes, 1, (size_t)n, stdout); return; }
    attr_apply(style);
#if defined(__DOS__)
    /* THE WRITE ITSELF moves here on DOS, and only on DOS.
       _settextcolor colours what _outtext emits; it has no effect on
       stdout, which goes through INT 21h and carries no attribute. An
       attribute path that left the writing to fwrite would set a colour
       nothing ever used - code that runs, links, and changes nothing on
       screen, which is worse than not having it at all.
       _outtext wants a NUL-terminated string and the caller's bytes are
       a slice, so they are copied out in bounded chunks rather than
       terminated in place. Static, not stack: iron law six, and this is
       the single-threaded CLI. */
    {
        int i;
        for (i = 0; i < n; i++) dos_putc(bytes[i]);
    }
#else
    fwrite(bytes, 1, (size_t)n, stdout);
#endif
}

void lz_attr_reset(void) {
    if (!g_on) return;
    attr_apply(0);
}
