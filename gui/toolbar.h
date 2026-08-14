#ifndef LZ_GUI_TOOLBAR_H
#define LZ_GUI_TOOLBAR_H

#include <windows.h>

/* comctl32's toolbar, declared by hand because commctrl.h cannot be
 * included at the API floor.
 *
 * Measured, not assumed: at -DWINVER=0x0351 the Open Watcom header dies
 * with five copies of
 *
 *     commctrl.h(4195): Error! E1022: Missing or misspelled data type
 *                       near 'NMHDR'
 *
 * because NMHDR itself is behind WINVER >= 0x0400 - the same guard that
 * makes -DWIN32_LEAN_AND_MEAN mandatory for this whole front end. So the
 * two structures and the handful of message numbers this needs are
 * spelled out below, the way gui/compat40.c already spells out
 * INITCOMMONCONTROLSEX.
 *
 * A STRUCT IS RISKIER THAN A CONSTANT, so this one has a gate.
 * TBBUTTON's layout differs between Win32 and Win64 - two reserved bytes
 * versus six, and pointer-sized trailing fields - and getting it wrong
 * does not fail to compile, it makes comctl32 read the button array at
 * the wrong stride and draw garbage. Without asserting size and every
 * offset against LZTbButton, this header is a guess that happens to
 * work on one machine.
 */

#if defined(_WIN64) || defined(__x86_64__) || defined(_M_X64)
typedef unsigned long long LZ_UPTR;
typedef long long          LZ_IPTR;
#define LZ_TB_RESERVED 6
#else
typedef unsigned long      LZ_UPTR;
typedef long               LZ_IPTR;
#define LZ_TB_RESERVED 2
#endif

typedef struct {
    int      iBitmap;
    int      idCommand;
    BYTE     fsState;
    BYTE     fsStyle;
    BYTE     bReserved[LZ_TB_RESERVED];
    LZ_UPTR  dwData;
    LZ_IPTR  iString;
} LZTbButton;

typedef struct {
    HINSTANCE hInst;
    LZ_UPTR   nID;
} LZTbAddBitmap;

/* The system's own toolbar glyphs, by name.
 *
 * comctl32 ships the strip every Win95 application drew from, and hands
 * it over on request: TB_ADDBITMAP with hInst = HINST_COMMCTRL and
 * nID = IDB_STD_SMALL_COLOR. Nothing is stored in this repository and
 * nothing is redistributed - on Windows 98 these ARE the period icons,
 * because they come out of the operating system.
 *
 * Named constants rather than raw indices on purpose. shell32 is the
 * other candidate and the wrong DLL twice over: it holds folder and
 * drive ICONS, not 16x16 toolbar glyphs, and its indices are ordinal
 * positions that have moved between Windows releases - an icon fetched
 * by number is a different picture on a different system. These names
 * are part of the documented API. */
#define LZ_STD_CUT         0
#define LZ_STD_COPY        1
#define LZ_STD_PASTE       2
#define LZ_STD_UNDO        3
#define LZ_STD_REDOW       4
#define LZ_STD_DELETE      5
#define LZ_STD_FILENEW     6
#define LZ_STD_FILEOPEN    7
#define LZ_STD_FILESAVE    8
#define LZ_STD_PRINTPRE    9
#define LZ_STD_PROPERTIES 10
#define LZ_STD_HELP       11
#define LZ_STD_FIND       12
#define LZ_STD_REPLACE    13
#define LZ_STD_PRINT      14

/* Create the tool bar. Returns NULL when comctl32 is absent, which is
 * the NT 3.51 case and is not an error: every command on the bar is also
 * on a menu, so the floor loses a convenience and nothing else. That is
 * why gui/main.c may not put a command here that is nowhere else.
 *
 * `cmds` are WM_COMMAND ids, `glyphs` the LZ_STD_* index each one shows,
 * `labels` the DISPLAY-form strings under the icons, `n` how many.
 */
HWND lz_gui_toolbar_create(HWND parent, HINSTANCE inst,
                           const int *cmds, const int *glyphs,
                           const char *const *labels,
                           int n, int ctl_id);

/* Grey a button. Safe with a NULL bar. */
void lz_gui_toolbar_enable(HWND tb, int cmd, int on);

/* Read that state back: 1 enabled, 0 greyed, -1 no such button or no
 * bar (which the selftest reads as "nothing to check here", the same
 * way it treats the whole NT 3.51 no-comctl32 case).
 *
 * Only the selftest needs it, for the same reason lz_gui_toolbar_text
 * exists: a command now wears up to three faces, and the way that
 * fails is one face disagreeing with the other two. A REBUILT strip
 * comes back with every button live, so any command whose grey state
 * follows the window's own state has to be re-applied afterwards -
 * three call sites now remember to, and nothing but reading the
 * control back can tell whether they did. */
int lz_gui_toolbar_enabled(HWND tb, int cmd);

/* Read a button's caption back out of the control.
 *
 * Only the selftest needs this, and that is the point: the captions go
 * in at creation time through TB_ADDSTRING and cannot be replaced, so a
 * language switch has to REBUILD the bar - and the way that fails is a
 * strip still wearing the old language while everything else changed.
 * Nothing but reading the control back can see it.
 *
 * Returns the length written, or -1 when there is no such button or the
 * caption does not fit. Safe with a NULL bar. */
int lz_gui_toolbar_text(HWND tb, int cmd, char *out, int cap);

/* The height one row of buttons actually needs, so the layout can be
 * checked against it rather than trusted. 0 when tb is NULL. */
int lz_gui_toolbar_needed_height(HWND tb);

#endif
