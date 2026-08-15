/* The About window - Help > About.
 *
 * Shape: plain popup + hand-run modal loop, exactly the settings dialog
 * (gui/settingsdlc.c). The one real difference is that nothing here has to
 * be READ BACK after the loop, so the handlers destroy the window directly
 * instead of setting a result for the caller to read controls through.
 *
 * Layout follows the reference (E:\LLM\refs\w95\word95-about-1x.png),
 * whose structure is: a TALL LOGO down the left
 * (the Word 95 app logo, about 68x247), the product text beside it split
 * into title / version / description lines, a full-width ETCHED DIVIDER
 * partway down, a small credits line below the divider, and OK bottom-right.
 * Measured from the reference, not guessed:
 *
 *   logo column    x 13..71   (~60 wide)
 *   text column    x 82..370
 *   divider        y 228..229 (two-row etched: light top, dark bottom)
 *   credits        y ~232..268 below the divider
 *   OK button      x 277..370, y 302..322  (~93x20, bottom-right)
 *
 * This dialog keeps the same ARRANGEMENT but uses the splash backdrop crop
 * (IDB_ABOUT_LOGO) in the tall logo column rather than the app icon - see
 * lz_gui_about_create. The logo column is a real rect the image sits in, so
 * the proportions read like the reference even before a final logo exists.
 *
 * The text and the icon are both references, never copies: the body is
 * LZ_STR_ABOUT_BODY - this file only SPLITS it by its own newlines for
 * display, never re-types a word - the caption is LZ_STR_DLG_ABOUT_TITLE,
 * and the icon is the same IDI_APP the main window loads. aboutdlg.h's
 * header comment says why.
 */
#include <stdio.h>       /* snprintf - split_body's line slices */
#include <string.h>

#include "aboutdlg.h"
#include "compat40.h"
#include "cpucheck.h"        /* lz_cpu_brand - the CPU line's model name */
#include "layout.h"          /* lz_gui_center_rect, LZ_GUI_DLG_* */
#include "localized_strings.h"
#include "resource.h"        /* IDI_APP */

#define ABT_CLASS "Kunkun98About"

/* WM_PRINTCLIENT is a 4.0 message and the 3.51 floor (WINVER=0x0351)
   hides it in winuser.h, exactly the gap main.c's status-bar bevels
   already hand-declare. Same spelling, so the two agree. */
#ifndef WM_PRINTCLIENT
#define WM_PRINTCLIENT 0x0318
#endif

/* PROCESSOR_INTEL_386/486/PENTIUM for the CPU fallback line. Watcom's
   winuser.h does not carry the PROCESSOR_INTEL_* names (gcc's does); the
   values are fixed by the SDK (386/486/586) - spelled out locally the
   way settingsdlg.c spells LZ_ES_NUMBER. */
#ifndef PROCESSOR_INTEL_386
#define PROCESSOR_INTEL_386 386
#define PROCESSOR_INTEL_486 486
#define PROCESSOR_INTEL_PENTIUM 586
#endif

/* Control ids. 3100+ so they cannot collide with the settings dialog's
 * 3000+ range or the main window's ID_* / IDM_* ids, and the selftest in
 * gui/main.c addresses them by their numeric value the way it does the
 * settings dialog's - renumbering an entry silently re-points those checks
 * at a different control. */
#define ID_ABT_OK     3101
#define ID_ABT_ICON   3102
#define ID_ABT_BODY   3103
#define ID_ABT_VER    3104   /* the version line */
#define ID_ABT_DESC   3105   /* the description block */
#define ID_ABT_CRED   3106   /* the credits line below the divider */
#define ID_ABT_DIV    3107   /* the etched divider */
#define ID_ABT_SYSINFO 3108  /* the System Info button */

/* Client area, from the reference's measured proportions
 * (E:\LLM\refs\w95\word95-about-1x.png):
 *
 *   logo column    x 2..71   (68 wide),  y 25..299  (275 tall) - a TALL
 *                  vertical region, not a square icon
 *   text column    x 82..370
 *   divider        y 229 (60% of the 383 height)
 *   credits        y ~262 below the divider
 *   OK button      y ~302..322, bottom-right
 *
 * The window is 360x300 client (383x383 minus the reference's borders),
 * which keeps the same proportions and still clears a 640x480 screen.
 *
 * The LOGO column is that 68-wide vertical region, kept at the
 * reference's ORIGINAL size. The splash backdrop crop (IDB_ABOUT_LOGO)
 * sits in it at the column's size, top-aligned, exactly as a real logo
 * of the region's size would. */
#define ABT_W 360
/* Height. Logo column to y202, divider 210, license AND the two buttons
   side by side from y222 (license left, buttons right). System Info
   bottom = 222+20+6+20 = 268, plus 13px margin = 281. */
#define ABT_H 281
#define ABT_ICON_PX 48
#define ABT_COL_W   68        /* the reference's logo column width */
/* The logo column HEIGHT, from the reference's real vertical extent
   (word95-about-1x.png: the logo graphic is y33..278 of 384 = 245px
   tall, and the divider sits 11px below it at y289). Scaled to this
   window's 300 height: 245/384*300 = 191, so the column ends at
   margin 11 + 191 = 202, above the divider at 225. The bad cyan-noise
   region below the real graphic is NOT part of the logo and must not
   be counted. */
#define ABT_COL_H  191        /* logo graphic height, scaled from 245/384 */

static int g_registered;
/* The divider's client-area y, set at create so WM_PAINT and the selftest
   agree on where it is. Etched via lz_draw_edge, not an SS_* style:
   SS_ETCHEDHORZ and SS_SUNKEN are Windows 95 additions and the 3.51 floor
   (WINVER=0x0351) hides both behind `#if (WINVER >= 0x0400)` in
   winuser.h - the same gap compat40.h already documents for DrawEdge
   itself. The front end draws its bevels with lz_draw_edge for exactly
   this reason (gui/main.c's status-bar bevels). */
static int g_div_y;
/* The title font, created by lz_gui_about_create and freed on WM_DESTROY.
   A font handed to a control with WM_SETFONT is BORROWED, not owned - the
   control keeps using the handle after the call returns, so the font must
   outlive the create call; a deleted handle leaves the title rendering in
   whatever FIXEDFONT degrades to. Dialog-lifetime ownership, exactly how
   the settings dialog keeps its fonts alive. */
static HFONT g_title_font;
static HFONT g_small_font;

static LRESULT CALLBACK abtproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_COMMAND:
        if (LOWORD(wp) == ID_ABT_OK || LOWORD(wp) == IDCANCEL) {
            /* Esc (via IsDialogMessage) and OK both land here; a modal
               box's OK and Esc mean the same thing - leave. Nothing to
               read back, so destroy now rather than deferring the way
               the settings dialog does. */
            DestroyWindow(h);
            return 0;
        }
        if (LOWORD(wp) == ID_ABT_SYSINFO) {
            /* Open the System Info dialog. The owner here
               is the ABOUT window, and a modal child of a modal child
               keeps the message flow correct: the about loop pumps, the
               sysinfo loop pumps, and on return the about window is
               still alive to keep pumping. The instance is the process's
               own - the same module both dialogs and the main window
               came from (a 64-bit-safe read of the window's would need
               GetWindowLongPtr, and there is nothing window-specific
               about the module handle to justify it). */
            lz_gui_sysinfo_dialog(h, GetModuleHandle(NULL));
            return 0;
        }
        break;
    case WM_CLOSE:
        DestroyWindow(h);
        return 0;
    case WM_DESTROY:
        /* The title font is dialog-lifetime (see g_title_font's own
           comment); free it here, after the controls that use it are
           gone. Same for the small license font. */
        if (g_title_font) {
            DeleteObject((HGDIOBJ)g_title_font);
            g_title_font = NULL;
        }
        if (g_small_font) {
            DeleteObject((HGDIOBJ)g_small_font);
            g_small_font = NULL;
        }
        return 0;
    case WM_PAINT:
    case WM_PRINTCLIENT: {
        /* The etched divider and the logo placeholder's frame, drawn
           with lz_draw_edge (the floor-safe bevel helper) rather than
           the SS_ETCHEDHORZ/SS_SUNKEN styles the 3.51 floor hides.
           WM_PRINTCLIENT alongside WM_PAINT, the same reason main.c's
           status-bar bevels handle both: a screenshot or any
           render-into-a-DC request would otherwise get the window
           without its divider. */
        PAINTSTRUCT ps;
        HDC dc;
        if (msg == WM_PRINTCLIENT) {
            dc = (HDC)wp;
        } else {
            dc = BeginPaint(h, &ps);
        }
        if (dc) {
            /* The divider: the reference's groove (word95-about-1x.png)
               is RAISED, not sunken - light grey (160) on the top row,
               white (255) below (y289..290 of the 384-tall reference).
               It spans the FULL client width, below the logo column.
               Drawn with two explicit pens; lz_draw_edge's etched edge
               does not render a visible line on this host. */
            if (g_div_y > 0) {
                /* The scheme's own shadow and highlight, not the two
                   values they happen to have in the default scheme -
                   160/160/160 and white are what COLOR_BTNSHADOW and
                   COLOR_BTNHIGHLIGHT return there, so this reads the
                   same and follows a changed scheme instead of
                   contradicting it. */
                HPEN top = CreatePen(PS_SOLID, 1,
                                     GetSysColor(COLOR_BTNSHADOW));
                HPEN bot = CreatePen(PS_SOLID, 1,
                                     GetSysColor(COLOR_BTNHIGHLIGHT));
                HPEN old = (HPEN)SelectObject(dc, top);
                int x0 = LZ_GUI_DLG_MARGIN;
                int x1 = ABT_W - LZ_GUI_DLG_MARGIN;
                MoveToEx(dc, x0, g_div_y, NULL);
                LineTo(dc, x1, g_div_y);
                SelectObject(dc, bot);
                MoveToEx(dc, x0, g_div_y + 1, NULL);
                LineTo(dc, x1, g_div_y + 1);
                SelectObject(dc, old);
                DeleteObject(top);
                DeleteObject(bot);
            }
            if (msg != WM_PRINTCLIENT) EndPaint(h, &ps);
        }
        if (msg == WM_PRINTCLIENT) return 0;
        break;
    }
    default:
        break;
    }
    return DefWindowProcA(h, msg, wp, lp);
}

static int register_class(HINSTANCE inst) {
    WNDCLASSA wc;
    if (g_registered) return 1;
    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc = abtproc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = ABT_CLASS;
    if (!RegisterClassA(&wc)) return 0;
    g_registered = 1;
    return 1;
}

/* One child, set to the UI font, mirroring settingsdlg.c's child_ex. */
static HWND child(HWND p, HINSTANCE inst, const char *cls, const char *text,
                  DWORD style, int x, int y, int w, int h, int id) {
    HWND c = CreateWindowExA(0, cls, text, WS_CHILD | WS_VISIBLE | style,
                             x, y, w, h, p, (HMENU)(LONG_PTR)id, inst, NULL);
    if (c) SendMessage(c, WM_SETFONT, (WPARAM)lz_ui_font(),
                       MAKELPARAM(TRUE, 0));
    return c;
}

/* The title line's font: the UI font, one size up (12pt vs 9pt). Built
 * by copying lz_ui_font()'s LOGFONT and raising the height, which keeps
 * whatever face the UI font has (SimSun in Chinese, the stock GUI font
 * in English) - the title reads as the same typeface, just bigger.
 * Sized from the device's LOGPIXELSY, not a hardcoded point-pixel
 * number, so a 120-DPI display gets a genuinely 12pt face (same
 * reasoning build_font() gives in gui/compat40.c).
 *
 * Returns an owned font; the caller must keep it alive for the
 * dialog's lifetime and DeleteObject it in WM_DESTROY. */
static HFONT title_font(void) {
    LOGFONTA lf;
    HDC dc;
    HFONT base = lz_ui_font();
    if (!base || !GetObjectA((HGDIOBJ)base, (int)sizeof lf, &lf)) return NULL;
    dc = GetDC(NULL);
    lf.lfHeight = -(dc ? MulDiv(12, GetDeviceCaps(dc, LOGPIXELSY), 72) : 16);
    if (dc) ReleaseDC(NULL, dc);
    return CreateFontIndirectA(&lf);
}

/* The license line's font: "Small Fonts" at 7pt, a bitmap face - NOT a
 * copy of the UI font. Word 95's own license/copyright line uses this
 * smaller bitmap face (measured from the reference: a ~9px non-
 * antialiased, proportional bitmap face, i.e. Small Fonts 7pt at 96 DPI,
 * which renders from the VGA-era fixed bitmap and has no smooth
 * outlines). Asking the system for the FACE by name rather than scaling
 * the UI font down is what actually lands on the bitmap: a scaled SimSun
 * would still render vector outlines, not the bitmap glyphs.
 *
 * Returns an owned font; the caller keeps it alive for the dialog's
 * lifetime and deletes it in WM_DESTROY (same contract as g_title_font). */
static HFONT small_font(void) {
    HDC dc = GetDC(NULL);
    int h = dc ? -MulDiv(7, GetDeviceCaps(dc, LOGPIXELSY), 72) : -9;
    HFONT f;
    if (dc) ReleaseDC(NULL, dc);
    f = CreateFontA(h, 0, 0, 0, FW_NORMAL, 0, 0, 0,
                    ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                    DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                    "Small Fonts");
    return f;
}

/* Split LZ_STR_ABOUT_BODY into its display roles. The body is five lines:
 *
 *   kunkun98                              -> title
 *   for Windows, version 0.1              -> version
 *   (blank)
 *   latent mixture of experts + ...       -> description block
 *   engine: llama98
 *
 * The license below the divider is NOT part of the body - it is its own
 * string (LZ_STR_ABOUT_LICENSE), rendered in Small Fonts 7pt. Splitting
 * is done here by line count against the LIVE string
 * table; the text itself is never copied or re-typed, so the string table
 * stays the single authority. A translation that changes how many lines
 * the body has changes what lands where, which is the correct reaction
 * rather than a bug. */
static void split_body(const char *body,
                       const char **out_title, const char **out_ver,
                       const char **out_desc) {
    static char title[64], ver[64], desc[256];
    const char *p = body, *nl;
    int line = 0;

    title[0] = ver[0] = desc[0] = '\0';
    while (p && *p) {
        nl = strchr(p, '\n');
        if (nl) {
            int n = (int)(nl - p);
            if (line == 0) { snprintf(title, sizeof title, "%.*s", n, p); }
            else if (line == 1) { snprintf(ver, sizeof ver, "%.*s", n, p); }
            else if (line >= 3) {
                /* Append ONLY this line's n bytes - a bare strncat(desc,
                   p, ...) would append the WHOLE rest of the body from p
                   onward (every later line twice), which the probe
                   caught. */
                int have = (int)strlen(desc);
                if (have && have < (int)sizeof desc - 1)
                    desc[have++] = '\n';
                if (have + n < (int)sizeof desc - 1) {
                    memcpy(desc + have, p, (size_t)n);
                    desc[have + n] = '\0';
                }
            }
            line++;
            p = nl + 1;
        } else {
            /* A final line with no trailing newline: still a desc line if
               past the blank. */
            int n = (int)strlen(p);
            if (line >= 3) {
                int have = (int)strlen(desc);
                if (have && have < (int)sizeof desc - 1)
                    desc[have++] = '\n';
                if (have + n < (int)sizeof desc - 1) {
                    memcpy(desc + have, p, (size_t)n);
                    desc[have + n] = '\0';
                }
            }
            break;
        }
    }
    if (out_title) *out_title = title;
    if (out_ver)   *out_ver   = ver;
    if (out_desc)  *out_desc  = desc;
}

HWND lz_gui_about_create(HWND owner, HINSTANCE inst) {
    HWND h;
    RECT rc, orc;
    int dx, dy, dw, dh;
    const char *title, *ver, *desc;
    int tx, tw, x, y;

    if (!register_class(inst)) return NULL;

    rc.left = 0; rc.top = 0; rc.right = ABT_W; rc.bottom = ABT_H;
    AdjustWindowRect(&rc, WS_CAPTION, FALSE);
    dw = rc.right - rc.left;
    dh = rc.bottom - rc.top;

    /* Centred on the owner - see settingsdlg.c's own comment for why
       CW_USEDEFAULT is wrong for a popup. */
    orc.left = orc.top = orc.right = orc.bottom = 0;
    if (owner) GetWindowRect(owner, &orc);
    lz_gui_center_rect(orc.left, orc.top, orc.right - orc.left,
                       orc.bottom - orc.top, dw, dh,
                       GetSystemMetrics(SM_CXSCREEN),
                       GetSystemMetrics(SM_CYSCREEN), &dx, &dy);

    h = CreateWindowExA(0, ABT_CLASS,
                        lz_str_display(LZ_STR_DLG_ABOUT_TITLE),
                        WS_POPUP | WS_CAPTION,
                        dx, dy, dw, dh, owner, NULL, inst, NULL);
    if (!h) return NULL;

    /* ---- the logo column, top-left, at the reference's ORIGINAL size
       (68 wide, 191 tall), holding the splash backdrop crop
       (IDB_ABOUT_LOGO) - the field + ghosted photo, no 3D rooster -
       dithered to the Win9x 256 palette so it BitBlts with no palette
       realisation, exactly like the splash. ---- */
    {
        HBITMAP hb = LoadBitmapA(inst, MAKEINTRESOURCEA(IDB_ABOUT_LOGO));
        HWND ctl = child(h, inst, "STATIC", NULL, SS_BITMAP,
                         LZ_GUI_DLG_MARGIN, LZ_GUI_DLG_MARGIN,
                         ABT_COL_W, ABT_COL_H, ID_ABT_ICON);
        if (ctl && hb)
            SendMessage(ctl, STM_SETIMAGE, IMAGE_BITMAP, (LPARAM)hb);
    }

    /* ---- text column, right of the logo, at the reference's x. ---- */
    tx = LZ_GUI_DLG_MARGIN + ABT_COL_W + LZ_GUI_DLG_MARGIN;
    tw = ABT_W - tx - LZ_GUI_DLG_MARGIN;
    x = tx;

    /* Title, large. Created into g_title_font, NOT a local deleted
       after SetFont - a font set with WM_SETFONT is borrowed by the
       control and must outlive the create call (see g_title_font's own
       comment; deleting it here is how the title falls back to
       FIXEDFONT). */
    split_body(lz_str_display(LZ_STR_ABOUT_BODY), &title, &ver, &desc);
    g_title_font = title_font();
    {
        HWND ctl = child(h, inst, "STATIC", title, SS_LEFT, x,
                         LZ_GUI_DLG_MARGIN, tw, 24, ID_ABT_BODY);
        if (ctl && g_title_font)
            SendMessage(ctl, WM_SETFONT, (WPARAM)g_title_font,
                        MAKELPARAM(TRUE, 0));
    }

    /* Version, then the description block, below the title. */
    y = LZ_GUI_DLG_MARGIN + 24 + 8;
    child(h, inst, "STATIC", ver, SS_LEFT, x, y, tw, 18, ID_ABT_VER);
    y += 18 + 12;
    child(h, inst, "STATIC", desc, SS_LEFT, x, y, tw, 60, ID_ABT_DESC);

    /* The etched divider. The reference (word95-about-1x.png) puts it at
       y289 of 384 = 75% of the height, full client width, BELOW the logo
       column (which ends around y284). Here it is a FIXED y just below
       the logo column (which ends at margin 11 + ABT_COL_H 191 = 202),
       not a fraction of the height - a fraction would drift the divider
       down as the license text below it grew the window. Recorded for
       WM_PAINT, which draws it with two explicit pens. */
    g_div_y = LZ_GUI_DLG_MARGIN + ABT_COL_H + 8;
    InvalidateRect(h, NULL, FALSE);

    /* License below the divider - Small Fonts 7pt. The reference puts its
       copyright/license line here (small, below the groove). Width is
       capped at the two buttons' LEFT edge (the buttons sit at x254) so
       the license column stays strictly to their left. Two lines at
       Small Fonts 7pt's 11px line pitch = 22px. */
    g_small_font = small_font();
    {
        int btn_left = ABT_W - 13 - 93;   /* the two buttons' x */
        HWND ctl = child(h, inst, "STATIC",
                         lz_str_display(LZ_STR_ABOUT_LICENSE), SS_LEFT,
                         LZ_GUI_DLG_MARGIN, g_div_y + 12,
                         btn_left - LZ_GUI_DLG_MARGIN, 22, ID_ABT_CRED);
        if (ctl && g_small_font)
            SendMessage(ctl, WM_SETFONT, (WPARAM)g_small_font,
                        MAKELPARAM(TRUE, 0));
    }

    /* The two buttons, stacked vertically at the bottom-right, at the
       SAME y as the license line - the license occupies the column LEFT
       of the buttons (x 11..243), the buttons the column RIGHT of it
       (x 254..347), both starting at g_div_y + 12 = 222. The reference
       puts OK (93x20) with its right edge 13px from the window's right
       edge and a 6px gap between OK and System Info below it. */
    child(h, inst, "BUTTON", lz_str_display(LZ_STR_BTN_OK),
          BS_PUSHBUTTON | WS_TABSTOP,
          ABT_W - 13 - 93,
          g_div_y + 12,
          93, 20, ID_ABT_OK);

    /* System Info, below OK. */
    child(h, inst, "BUTTON", lz_str_display(LZ_STR_BTN_SYSINFO),
          BS_PUSHBUTTON | WS_TABSTOP,
          ABT_W - 13 - 93,
          g_div_y + 12 + 20 + 6,
          93, 20, ID_ABT_SYSINFO);

    return h;
}

void lz_gui_about_dialog(HWND owner, HINSTANCE inst) {
    HWND h;
    MSG msg;

    h = lz_gui_about_create(owner, inst);
    if (!h) return;

    /* Modal by hand - same shape as settingsdlg. The enable MUST happen
       before the owner is activated, or focus goes elsewhere. */
    EnableWindow(owner, FALSE);
    ShowWindow(h, SW_SHOW);
    SetFocus(GetDlgItem(h, ID_ABT_OK));
    while (IsWindow(h) && GetMessage(&msg, NULL, 0, 0) > 0) {
        /* IsDialogMessage gives Tab order and Esc for free. */
        if (!IsDialogMessage(h, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
    if (IsWindow(owner)) EnableWindow(owner, TRUE);
    SetActiveWindow(owner);
}

/* ================================================ System Info dialog
 *
 * The About dialog's System Info... button. Mechanism
 * follows the decompiled Winword.exe (D:\ghidra\projects\W95Winword.gpr,
 * FUN_501e7c09): an enumerator over system facts, each fetched with the
 * matching Win32 call and formatted into a line:
 *
 *   case 0  OS version    GetVersionExA
 *   case 1  CPU           GetSystemInfo
 *   case 4  memory        GlobalMemoryStatus
 *   case 5  disk          GetDiskFreeSpaceA
 *
 * Word 95 loads its own wording for these (the 0x9a000x resources);
 * kunkun98 uses its own string-table labels but the calls are the same.
 * A dialog instead of the richer list Word 95's owned one: this is a
 * minimal client, and four facts answer "what machine am I on".
 */

#define SYS_CLASS "Kunkun98SysInfo"
#define ID_SYS_OK 3201
#define ID_SYS_OS  3202   /* the OS version line (first fact) */
#define ID_SYS_CPU 3203   /* the CPU line */

static int g_sys_registered;

static LRESULT CALLBACK sysproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_COMMAND:
        if (LOWORD(wp) == ID_SYS_OK || LOWORD(wp) == IDCANCEL) {
            DestroyWindow(h);
            return 0;
        }
        break;
    case WM_CLOSE:
        DestroyWindow(h);
        return 0;
    default:
        break;
    }
    return DefWindowProcA(h, msg, wp, lp);
}

static int sys_register(HINSTANCE inst) {
    WNDCLASSA wc;
    if (g_sys_registered) return 1;
    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc = sysproc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = SYS_CLASS;
    if (!RegisterClassA(&wc)) return 0;
    g_sys_registered = 1;
    return 1;
}

/* The OS version line. GetVersionExA - the same call case 0 of the
 * decompiled enumerator makes. Win9x reports major 4; NT5 reports 5.
 * The build number is the interesting part on the target (95=950,
 * 98=1998, NT4=1381). */
static void sys_os(char *out, int cap) {
    OSVERSIONINFOA vi;
    memset(&vi, 0, sizeof vi);
    vi.dwOSVersionInfoSize = sizeof vi;
    if (GetVersionExA(&vi))
        snprintf(out, cap, "Windows %lu.%lu (build %lu)",
                 vi.dwMajorVersion, vi.dwMinorVersion, vi.dwBuildNumber);
    else
        snprintf(out, cap, "?");
}

/* The CPU line. GetSystemInfo - case 1 of the decompiled enumerator.
   Word 95 switches on si.dwProcessorType for 386/486/Pentium and falls
   to an "other" line - which on a modern machine (dwProcessorType 8664)
   is all it can say. The MODERN processor detection reads the CPUID
   brand string instead (src/cpucheck.c's lz_cpu_brand), the
   chip's real model name; the 386/486/Pentium names are kept only as
   the fallback for CPUs that predate the brand leaves. The processor
   count still comes from GetSystemInfo, the call Word 95 made. */
static void sys_cpu(char *out, int cap) {
    SYSTEM_INFO si;
    const char *brand = lz_cpu_brand();
    GetSystemInfo(&si);
    if (brand[0]) {
        snprintf(out, cap, "%s, %lu processor%s", brand,
                 si.dwNumberOfProcessors,
                 si.dwNumberOfProcessors == 1 ? "" : "s");
        return;
    }
    /* No brand string (a 386/486, or a CPUID without the extended
       leaves). Word 95's own four-way naming is the honest fallback. */
    {
        const char *arch;
        switch (si.dwProcessorType) {
        case PROCESSOR_INTEL_386:      arch = "386";    break;
        case PROCESSOR_INTEL_486:      arch = "486";    break;
        case PROCESSOR_INTEL_PENTIUM:  arch = "Pentium"; break;
        default:                       arch = "Other"; break;
        }
        snprintf(out, cap, "%s, %lu processor%s", arch,
                 si.dwNumberOfProcessors,
                 si.dwNumberOfProcessors == 1 ? "" : "s");
    }
}

/* The memory line. GlobalMemoryStatus - case 4. dwTotalPhys is the
   installed RAM; dwAvailPhys is what is left. Both in bytes. */
static void sys_mem(char *out, int cap) {
    MEMORYSTATUS ms;
    memset(&ms, 0, sizeof ms);
    ms.dwLength = sizeof ms;
    GlobalMemoryStatus(&ms);
    snprintf(out, cap, "%lu MB total, %lu MB free",
             (unsigned long)(ms.dwTotalPhys / (1024 * 1024)),
             (unsigned long)(ms.dwAvailPhys / (1024 * 1024)));
}

/* The disk line. GetDiskFreeSpaceA - case 5. The decompile calls it with
   index 0, which here is the current working directory's drive. The
   current directory needs a full-size buffer - GetCurrentDirectoryA
   fails if the buffer is smaller than the path - then its first three
   characters ("C:\") are the root to ask about. */
static void sys_disk(char *out, int cap) {
    char dir[MAX_PATH];
    char root[4];
    DWORD spc, bps, freec, totc;
    if (!GetCurrentDirectoryA((DWORD)sizeof dir, dir)) {
        snprintf(out, cap, "?");
        return;
    }
    root[0] = dir[0];
    root[1] = ':';
    root[2] = '\\';
    root[3] = '\0';
    if (GetDiskFreeSpaceA(root, &spc, &bps, &freec, &totc)) {
        unsigned long long bytes = (unsigned long long)totc * spc * bps;
        snprintf(out, cap, "%s %llu MB total", root,
                 (unsigned long long)(bytes / (1024 * 1024)));
    } else {
        snprintf(out, cap, "%s ?", root);
    }
}

HWND lz_gui_sysinfo_create(HWND owner, HINSTANCE inst) {
    HWND h;
    RECT rc, orc;
    int dx, dy, dw, dh;
    int x, y, w;
    char os[160], cpu[160], mem[160], disk[160];
    char line[320];

    if (!sys_register(inst)) return NULL;

    /* Four facts, each label: value, one per line. The label string is
       the table's (e.g. "Windows: %s") formatted with the fetched value -
       the same shape the decompiled Winword.exe's enumerator produces,
       one line per system fact. */
    sys_os(os, (int)sizeof os);
    sys_cpu(cpu, (int)sizeof cpu);
    sys_mem(mem, (int)sizeof mem);
    sys_disk(disk, (int)sizeof disk);

    /* Window width is MEASURED, not fixed. The CPU brand string is the
       widest line (324px for "AMD Ryzen 7 5800X 8-Core Processor, 16
       processors" in the 9pt face); a fixed 320-wide window clips it by
       26px. The same measure-the-real-font shape settingsdlg.c uses for
       its rows: format all four lines, take the widest, add the two
       dialog margins. The value changes with the host's CPU and font, so
       the window must follow it. */
    {
        HDC mdc = GetDC(NULL);
        HGDIOBJ oldf = NULL;
        int maxw = 0, i;
        char *lines[4];
        lines[0] = os; lines[1] = cpu; lines[2] = mem; lines[3] = disk;
        if (mdc) oldf = SelectObject(mdc, lz_ui_font());
        for (i = 0; i < 4; i++) {
            snprintf(line, sizeof line, lz_str_display(LZ_STR_SYSINFO_OS + i),
                     lines[i]);
            if (mdc) {
                SIZE sz;
                if (GetTextExtentPoint32A(mdc, line, (int)strlen(line), &sz)
                    && sz.cx > maxw)
                    maxw = sz.cx;
            }
        }
        if (mdc) {
            if (oldf) SelectObject(mdc, oldf);
            ReleaseDC(NULL, mdc);
        }
        /* Floor 280 so the title still fits; the widest line + margins
           otherwise. The CLIENT width is maxw + two margins (that is
           what the STATIC lines get); dw/dh are the WINDOW size
           AdjustWindowRect turns that into. */
        if (maxw < 280) maxw = 280;
        rc.left = 0; rc.top = 0;
        rc.right = maxw + 2 * LZ_GUI_DLG_MARGIN;
        rc.bottom = 140;
        /* The client width, saved before AdjustWindowRect grows the
           rect into a window size - the STATIC lines live in client
           coordinates and want this, not the bordered dw. */
        w = rc.right;
        AdjustWindowRect(&rc, WS_CAPTION, FALSE);
        dw = rc.right - rc.left;
        dh = rc.bottom - rc.top;
    }

    orc.left = orc.top = orc.right = orc.bottom = 0;
    if (owner) GetWindowRect(owner, &orc);
    lz_gui_center_rect(orc.left, orc.top, orc.right - orc.left,
                       orc.bottom - orc.top, dw, dh,
                       GetSystemMetrics(SM_CXSCREEN),
                       GetSystemMetrics(SM_CYSCREEN), &dx, &dy);

    h = CreateWindowExA(0, SYS_CLASS,
                        lz_str_display(LZ_STR_DLG_SYSINFO_TITLE),
                        WS_POPUP | WS_CAPTION,
                        dx, dy, dw, dh, owner, NULL, inst, NULL);
    if (!h) return NULL;

    x = LZ_GUI_DLG_MARGIN;
    /* w is set in the measurement block above (the client width the
       STATIC lines get, i.e. the widest measured line + two margins). */
    y = LZ_GUI_DLG_MARGIN;

    snprintf(line, sizeof line, lz_str_display(LZ_STR_SYSINFO_OS), os);
    child(h, inst, "STATIC", line, SS_LEFT, x, y, w, 18, ID_SYS_OS);
    y += 18 + 6;

    snprintf(line, sizeof line, lz_str_display(LZ_STR_SYSINFO_CPU), cpu);
    child(h, inst, "STATIC", line, SS_LEFT, x, y, w, 18, ID_SYS_CPU);
    y += 18 + 6;

    snprintf(line, sizeof line, lz_str_display(LZ_STR_SYSINFO_MEM), mem);
    child(h, inst, "STATIC", line, SS_LEFT, x, y, w, 18, 0);
    y += 18 + 6;

    snprintf(line, sizeof line, lz_str_display(LZ_STR_SYSINFO_DISK), disk);
    child(h, inst, "STATIC", line, SS_LEFT, x, y, w, 18, 0);
    y += 18 + 6;

    /* OK, bottom-right. */
    child(h, inst, "BUTTON", lz_str_display(LZ_STR_BTN_OK),
          BS_PUSHBUTTON | WS_TABSTOP,
          dw - LZ_GUI_DLG_MARGIN - LZ_GUI_BTN_W,
          140 - LZ_GUI_DLG_MARGIN - LZ_GUI_BTN_H,
          LZ_GUI_BTN_W, LZ_GUI_BTN_H, ID_SYS_OK);

    return h;
}

void lz_gui_sysinfo_dialog(HWND owner, HINSTANCE inst) {
    HWND h;
    MSG msg;

    h = lz_gui_sysinfo_create(owner, inst);
    if (!h) return;

    EnableWindow(owner, FALSE);
    ShowWindow(h, SW_SHOW);
    SetFocus(GetDlgItem(h, ID_SYS_OK));
    while (IsWindow(h) && GetMessage(&msg, NULL, 0, 0) > 0) {
        if (!IsDialogMessage(h, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
    if (IsWindow(owner)) EnableWindow(owner, TRUE);
    SetActiveWindow(owner);
}

