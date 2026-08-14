/* The tool bar. See toolbar.h for why commctrl.h is not included.
 *
 * The window's commands live on a ToolbarWindow32 with 16x16 icons,
 * not a row of standard text buttons under the menu bar - Win9x put
 * commands on a toolbar, and a strip of text buttons was a DIALOG's
 * shape.
 *
 * Labels UNDER the icons, which is the toolbar's default when buttons
 * carry strings. Icon-only with tooltips is the other period-correct
 * choice and is rejected here for two reasons: the tooltip text has to
 * come back through TTN_NEEDTEXT in the parent's WM_NOTIFY, which puts
 * comctl32 structures into main.c after all; and four two-character
 * Chinese labels are more use to the reader than four pictures.
 */
#include <string.h>

#include "toolbar.h"

/* WM_USER is 0x0400. These are the toolbar messages this file uses and
   no more; each is (WM_USER + n) with n as commctrl.h has it. */
#define LZ_TB_ENABLEBUTTON     (WM_USER + 1)
#define LZ_TB_GETSTATE         (WM_USER + 18)
#define LZ_TB_ADDBITMAP        (WM_USER + 19)
#define LZ_TB_ADDBUTTONSA      (WM_USER + 20)
#define LZ_TB_ADDSTRINGA       (WM_USER + 28)
#define LZ_TB_BUTTONSTRUCTSIZE (WM_USER + 30)
#define LZ_TB_SETBITMAPSIZE    (WM_USER + 32)
#define LZ_TB_AUTOSIZE         (WM_USER + 33)
#define LZ_TB_GETBUTTONTEXTA   (WM_USER + 45)
#define LZ_TB_GETBUTTONSIZE    (WM_USER + 58)
#define LZ_TB_GETMAXSIZE       (WM_USER + 83)
#define LZ_TB_SETBUTTONWIDTH   (WM_USER + 59)

#define LZ_TBSTATE_ENABLED  0x04
#define LZ_TBSTYLE_BUTTON   0x00

/* CCS_* keep the bar where the layout puts it. Without NORESIZE and
   NOPARENTALIGN a toolbar moves itself to the top of the client area and
   sizes itself to the full width on every WM_SIZE, which would make
   gui/layout.c's rectangle for it a polite suggestion.

   NODIVIDER IS DELIBERATELY NOT SET. A menu bar above is no reason to
   suppress it: Word 95's tool bar dock has an etched groove
   immediately under the menu. Measured on the real window (gui
   capture at x=700, a gap between buttons):

       y=46  240,240,240   COLOR_BTNFACE
       y=47  160,160,160   COLOR_3DSHADOW      groove
       y=48  255,255,255   COLOR_3DHILIGHT
       y=49  240,240,240   the bar's own background

   That pair is exactly what the divider draws, so the fix is to stop
   suppressing it rather than to paint one. */
#define LZ_CCS_NORESIZE      0x0004
#define LZ_CCS_NOPARENTALIGN 0x0008

static const char TOOLBAR_CLASS[] = "ToolbarWindow32";

/* HINST_COMMCTRL is ((HINSTANCE)-1); IDB_STD_LARGE_COLOR is 1 - the
   pair that asks comctl32 for its own 24x24 strip instead of loading a
   bitmap from this binary.
   LARGE, not SMALL. comctl32 ships both, and the era shipped both as a
   user choice - the "large icons" setting Word and Internet Explorer
   put in their toolbar options. At 16 the glyph is smaller than the two
   characters of its own label. */
#define LZ_HINST_COMMCTRL      ((HINSTANCE)(LZ_IPTR)-1)
#define LZ_IDB_STD_LARGE_COLOR 1
#define LZ_STD_GLYPH_COUNT     15
#define LZ_STD_GLYPH_PX        24

HWND lz_gui_toolbar_create(HWND parent, HINSTANCE inst,
                           const int *cmds, const int *glyphs,
                           const char *const *labels,
                           int n, int ctl_id) {
    HWND tb;
    LZTbAddBitmap ab;
    LZTbButton buttons[16];
    char strbuf[512];
    int i, len = 0, first_string;

    if (!parent || !cmds || !glyphs || !labels || n <= 0 || n > 16)
        return NULL;
    for (i = 0; i < n; i++)
        if (!cmds[i] || glyphs[i] < 0 || glyphs[i] >= LZ_STD_GLYPH_COUNT)
            return NULL;

    /* No dynamic loading in this file - that is compat40.c's job.
       lz_statusbar_class() has already brought comctl32 in and called
       InitCommonControlsEx with ICC_BAR_CLASSES, the same class group
       the toolbar belongs to. If that failed there is no toolbar class
       to find, CreateWindow returns NULL, and that is exactly the
       answer this function reports. */
    tb = CreateWindowExA(0, TOOLBAR_CLASS, NULL,
                         WS_CHILD | WS_VISIBLE | LZ_CCS_NORESIZE |
                         LZ_CCS_NOPARENTALIGN,
                         0, 0, 0, 0, parent, (HMENU)(LZ_IPTR)ctl_id,
                         inst, NULL);
    if (!tb) return NULL;

    /* Required first, and the reason is the ABI note in toolbar.h: this
       is how the control learns the stride of the array below. */
    SendMessage(tb, LZ_TB_BUTTONSTRUCTSIZE, (WPARAM)sizeof(LZTbButton), 0);
    SendMessage(tb, LZ_TB_SETBITMAPSIZE, 0,
                MAKELONG(LZ_STD_GLYPH_PX, LZ_STD_GLYPH_PX));

    /* The system's strip, not ours. wParam is the number of images the
       bitmap holds - the whole standard set, not the four we use, or
       comctl32 indexes into it short. */
    (void)inst;
    ab.hInst = LZ_HINST_COMMCTRL;
    ab.nID = (LZ_UPTR)LZ_IDB_STD_LARGE_COLOR;
    if (SendMessage(tb, LZ_TB_ADDBITMAP,
                    (WPARAM)LZ_STD_GLYPH_COUNT, (LPARAM)&ab) < 0) {
        DestroyWindow(tb);
        return NULL;
    }

    /* TB_ADDSTRING takes a run of NUL-terminated strings ended by a
       SECOND NUL, and returns the index of the first one.
       NEVER PUT AN EMPTY STRING IN THIS LIST. It is byte-for-byte the
       terminator, so the list ends there and every button after it gets
       no caption - icons with nothing under them, which reads as a font
       problem and is a string-table problem. This toolbar uses no
       separators (see gui/main.c for why), but an empty label would
       still end the list silently - and the trap is written down
       because the next empty label will not announce itself either. */
    for (i = 0; i < n; i++) {
        size_t l = strlen(labels[i]);
        if (!l) { DestroyWindow(tb); return NULL; }
        if (len + (int)l + 2 >= (int)sizeof strbuf) { DestroyWindow(tb); return NULL; }
        memcpy(strbuf + len, labels[i], l + 1);
        len += (int)l + 1;
    }
    strbuf[len] = '\0';
    first_string = (int)SendMessage(tb, LZ_TB_ADDSTRINGA, 0, (LPARAM)strbuf);
    if (first_string < 0) { DestroyWindow(tb); return NULL; }

    memset(buttons, 0, sizeof buttons);
    for (i = 0; i < n; i++) {
        buttons[i].iBitmap = glyphs[i];
        buttons[i].idCommand = cmds[i];
        buttons[i].fsState = LZ_TBSTATE_ENABLED;
        buttons[i].fsStyle = LZ_TBSTYLE_BUTTON;
        buttons[i].iString = (LZ_IPTR)(first_string + i);
    }
    if (!SendMessage(tb, LZ_TB_ADDBUTTONSA, (WPARAM)n, (LPARAM)buttons)) {
        DestroyWindow(tb);
        return NULL;
    }
    /* A floor on the button width. Left to itself the control sizes each
       button to the wider of its icon and its label, and a two-character
       Chinese label comes out exactly as wide as the text with no
       breathing room on either side - buttons that touch their own
       captions. 52 is the Win9x tool button with a label under it.
       TB_SETBUTTONWIDTH arrived in comctl32 4.70; on an older one it is
       an unanswered message and the buttons stay tight, which is a worse
       look and not a broken one. */
    /* lParam is MAKELONG(cxMin, cxMax). cxMax stays 0: raising it does
       not un-ellipsise the caption - the width ceiling is not what clips
       "Settings"; the fix lives in the caption instead (see
       LZ_STR_BTN_SETTINGS). */
    SendMessage(tb, LZ_TB_SETBUTTONWIDTH, 0, MAKELONG(52, 0));
    SendMessage(tb, LZ_TB_AUTOSIZE, 0, 0);
    return tb;
}

void lz_gui_toolbar_enable(HWND tb, int cmd, int on) {
    if (!tb) return;
    SendMessage(tb, LZ_TB_ENABLEBUTTON, (WPARAM)cmd, MAKELONG(on ? 1 : 0, 0));
}

int lz_gui_toolbar_enabled(HWND tb, int cmd) {
    LRESULT st;
    if (!tb) return -1;
    /* TB_GETSTATE returns -1 for an unknown command, which is why the
       cast to LRESULT is compared before the bit is tested: the state
       bits of -1 include LZ_TBSTATE_ENABLED, so masking first would
       report a button that does not exist as enabled. */
    st = SendMessage(tb, LZ_TB_GETSTATE, (WPARAM)cmd, 0);
    if (st == -1) return -1;
    return (st & LZ_TBSTATE_ENABLED) ? 1 : 0;
}

int lz_gui_toolbar_text(HWND tb, int cmd, char *out, int cap) {
    /* TB_GETBUTTONTEXT has no length parameter - it writes as much as
       the caption needs and returns the length, which is why the length
       is asked for FIRST with a NULL buffer. Handing it a short buffer
       and hoping is how a 512-byte caption lands in a 32-byte local.
       Returns -1 when the control has no such command, so a caller can
       tell "no caption" from "no button". */
    int n;
    if (out && cap > 0) out[0] = '\0';
    if (!tb || !out || cap <= 0) return -1;
    n = (int)SendMessage(tb, LZ_TB_GETBUTTONTEXTA, (WPARAM)cmd, 0);
    if (n < 0 || n >= cap) return -1;
    n = (int)SendMessage(tb, LZ_TB_GETBUTTONTEXTA, (WPARAM)cmd, (LPARAM)out);
    if (n < 0) { out[0] = '\0'; return -1; }
    out[n < cap ? n : cap - 1] = '\0';
    return n;
}

int lz_gui_toolbar_needed_height(HWND tb) {
    /* TB_GETMAXSIZE, not TB_GETBUTTONSIZE.
       TB_GETBUTTONSIZE is the button alone - it leaves out the margin
       the control keeps above and below the row. TB_GETMAXSIZE reports
       what the control needs for the whole row, which is the question
       being asked.
       It arrived in comctl32 4.71; when it is unanswered the button
       height plus the difference measured here is the fallback, and
       being a few pixels tall is the harmless direction. */
    SIZE sz;
    DWORD b;
    if (!tb) return 0;
    sz.cx = sz.cy = 0;
    /* +2: measured. The control reports 43 for a row it paints 45 tall
       (see the note on TOOLBAR_H in gui/layout.c). Adding the
       difference here keeps the selftest's comparison honest instead of
       leaving two pixels of blind spot in it. */
    if (SendMessage(tb, LZ_TB_GETMAXSIZE, 0, (LPARAM)&sz) && sz.cy > 0)
        return (int)sz.cy + 2;
    b = (DWORD)SendMessage(tb, LZ_TB_GETBUTTONSIZE, 0, 0);
    return (int)HIWORD(b) + 4;
}
