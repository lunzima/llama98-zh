/* Run-time degradation for Windows NT 3.51. See compat40.h.
 *
 * Every entry point that might be missing is reached through
 * GetProcAddress rather than an import, because an import that the
 * running system cannot satisfy is not a failed call - it is a process
 * that never starts, with a dialog naming a DLL the user has never
 * heard of. That is also why the API-level gate cannot see any of this:
 * dynamic resolution is invisible to a linker.
 */
/* THIS FILE, AND ONLY THIS FILE, COMPILES AT THE 4.0 DECLARATION LEVEL.
 *
 * It has to. The 4.0 constants and structures it degrades away from -
 * WS_EX_CLIENTEDGE, DEFAULT_GUI_FONT, BROWSEINFOA - are hidden at the
 * 3.51 floor, and a file cannot fall back from something it cannot
 * name. Watcom's own headers are not even self-consistent down there:
 * commdlg.h drags in prsht.h, which uses NMHDR, which winuser.h
 * declares behind `WINVER >= 0x0400`. That is an SDK bug, not ours, and
 * it does not compile at 0x0351 no matter what the program does.
 *
 * The raise MUST happen before windows.h, not between two later
 * includes: winuser.h has an include guard, so lifting WINVER after
 * compat40.h has already pulled it in changes nothing.
 *
 * _WIN32_WINNT is NOT raised. WINVER is the 9x-line version and is what
 * guards the window furniture this file exists for; _WIN32_WINNT guards
 * the NT4-only entry points, and those stay forbidden even here -
 * "NT4 runs it" is exactly the assumption this project is trying not to
 * make.
 *
 * No other file carries a WINVER, a LoadLibrary or a GetProcAddress, so
 * this exemption stays one file wide. */
#undef WINVER
#define WINVER 0x0400

/* windows.h explicitly here, ahead of compat40.h: commdlg.h (next)
 * assumes windows.h's base types are already in scope, and it has to
 * come before compat40.h's own #include "compat40.h" so that
 * commdlg.h's REAL FR_DOWN/FR_MATCHCASE win the '#ifndef'
 * guard in compat40.h's own declaration of those two - the two
 * definitions are numerically identical but not the same TOKEN
 * SEQUENCE on either toolchain (0x1/0x4 here, 0x00000001L/0x00000004L
 * on Watcom, and gui/compat40.h's own hand-declared fallback matches
 * neither), so if compat40.h's guard ran first it would win instead
 * and commdlg.h's own unconditional #define right after it would
 * collide (Watcom's -we reports this as W140). Including windows.h a
 * second time via compat40.h right after is a no-op, guarded by
 * windows.h's own include guard. */
#include <windows.h>
#include <commdlg.h>
#include "compat40.h"    /* pulls windows.h; must come after the raise */
#include "resource.h"    /* IDI_APP for lz_ui_icon_16 */

#include <shlobj.h>
/* HDROP and DragAcceptFiles/DragQueryFileA/DragFinish live here, not in
 * shlobj.h above (that one is SHBrowseForFolder's header, a different
 * shell32 corner). windows.h does not pull this in on its own: the
 * whole front end builds with -DWIN32_LEAN_AND_MEAN
 * (build/watcom/winver.sh), which is precisely the flag that suppresses
 * shellapi.h's automatic include - so a file that wants Drag* has to
 * ask for it by name, same as commdlg.h and shlobj.h just above. */
#include <shellapi.h>
#include <string.h>

/* Resolved once. A GUI asks these questions on every repaint and every
 * appended token; GetProcAddress on each is a syscall for an answer
 * that cannot change while the process lives. */
static int   g_probed;
static int   g_major;
static HFONT g_font;
/* 1 when g_font came from CreateFont and must be deleted before it is
   replaced; 0 when it is a stock object, which must NOT be. */
static int g_font_owned;
/* -1 until the first build, so lz_ui_set_font_lang(0) right after probe
   is not mistaken for "already Chinese, nothing to do". */
static int g_font_english = -1;
static HMODULE g_riched;
static const char *g_rich_class;
static HMODULE g_imm;
static HMODULE g_shell;
static HMODULE g_comctl;
static const char *g_sbar_class;

typedef HIMC (WINAPI *ImmGetContextFn)(HWND);
typedef LONG (WINAPI *ImmGetCompStrFn)(HIMC, DWORD, LPVOID, DWORD);
typedef BOOL (WINAPI *ImmReleaseFn)(HWND, HIMC);
typedef BOOL (WINAPI *GetScrollInfoFn)(HWND, int, LPSCROLLINFO);
typedef HICON (WINAPI *LoadImageFn)(HINSTANCE, LPCSTR, UINT, int, int,
                                    UINT);
typedef LPITEMIDLIST (WINAPI *BrowseFn)(LPBROWSEINFOA);
typedef BOOL (WINAPI *PathFromIDLFn)(LPCITEMIDLIST, LPSTR);
typedef HRESULT (WINAPI *GetMallocFn)(LPMALLOC *);
/* void, not BOOL - DragAcceptFiles and DragFinish both return nothing,
 * unlike most of shell32's other BOOL-returning entry points. Getting
 * this typedef wrong would not fail to compile (a function pointer
 * cast is a cast), it would just read a garbage return value nothing
 * here uses - harmless today only because lz_drop_accept does not look
 * at what the call returned. */
typedef void (WINAPI *DragAcceptFn)(HWND, BOOL);
typedef UINT (WINAPI *DragQueryFileFn)(HDROP, UINT, LPSTR, UINT);
typedef void (WINAPI *DragFinishFn)(HDROP);
/* GetKeyboardLayout - a genuinely ancient user32 export (keyboard
 * layout management predates Win32 itself), but Watcom's winuser.h
 * hides its declaration behind `WINVER >= 0x0400` anyway (the guard
 * opens around line 6157 and does not close before it), so gui/main.c
 * cannot name it directly at the 3.51 floor - a constant is safe to
 * hand-declare (gui/compat40.h's own rule), a FUNCTION prototype is
 * not, so this is routed through here like everything else in this file. */
typedef HKL (WINAPI *GetKeyboardLayoutFn)(DWORD);
/* DrawEdge - a real Windows 95 addition, unlike GetKeyboardLayout
 * above: the export genuinely is absent on NT 3.51, so this one is a
 * capability question and not only a hidden declaration. */
typedef BOOL (WINAPI *DrawEdgeFn)(HDC, LPRECT, UINT, UINT);

typedef BOOL (WINAPI *SysParamFn)(UINT, UINT, PVOID, UINT);
typedef BOOL (WINAPI *DrawFrameCtlFn)(HDC, LPRECT, UINT, UINT);
typedef BOOL (WINAPI *DrawIconExFn)(HDC, int, int, HICON, int, int, UINT,
                                    HBRUSH, UINT);

/* GetProcAddress returns FARPROC, and casting that straight to a typed
 * function pointer is what -Wcast-function-type objects to. The cast
 * through `void (*)(void)` is the documented way to say "yes, I mean
 * it" - and it stays a function-pointer cast throughout, unlike the
 * common trick of going via void *, which is not defined for function
 * pointers at all. */
#define LZ_PROC(m, n) ((void (*)(void))GetProcAddress((m), (n)))

/* Pointer-sized integer, for the SetWindowLong cast. Spelled here for
   the same reason gui/toolbar.h spells its own: basetsd.h is not
   reachable at the API floor. */
#if defined(_WIN64) || defined(__x86_64__) || defined(_M_X64)
typedef long long LZ_IPTR_T;
#else
typedef long      LZ_IPTR_T;
#endif

static ImmGetContextFn  p_ImmGetContext;
static ImmGetCompStrFn  p_ImmGetCompositionString;
static ImmReleaseFn     p_ImmReleaseContext;
static GetScrollInfoFn  p_GetScrollInfo;
static LoadImageFn      p_LoadImageA;
static BrowseFn         p_SHBrowseForFolder;
static PathFromIDLFn    p_SHGetPathFromIDList;
static GetMallocFn      p_SHGetMalloc;
static DragAcceptFn     p_DragAcceptFiles;
static DragQueryFileFn  p_DragQueryFileA;
static DragFinishFn     p_DragFinish;
static GetKeyboardLayoutFn p_GetKeyboardLayout;
static DrawEdgeFn       p_DrawEdge;
static SysParamFn       p_SystemParametersInfo;
static DrawFrameCtlFn   p_DrawFrameControl;
static DrawIconExFn     p_DrawIconEx;
/* SimSun's face name in GBK - CB CE CC E5 - written as bytes so this
   file stays ASCII (iron law seven) and so the exact bytes are
   visible. It has to be the Chinese name: the "SimSun" alias is a
   later addition and is not what a Chinese Windows 98 keys its font
   table on. */
static const char FACE_SONGTI[] = "\xCB\xCE\xCC\xE5";

#ifndef GB2312_CHARSET
#define GB2312_CHARSET 134
#endif

/* Build the UI font for one language.
 *
 * WHY NOT JUST DEFAULT_GUI_FONT: the stock object is whatever the
 * running system decided, and the two systems disagree in a way that
 * matters here. On Chinese Windows 98 it
 * is SimSun 9pt - the thing this program is drawn for, and the metric
 * gui/layout.h's row heights were derived from (LZ_GUI_INPUT_MIN_H is
 * three lines of a 14-pixel pitch, which is SimSun 9pt at 96 DPI). On a
 * modern host it is MS Shell Dlg, which is not that, and RichEdit does
 * not take it at all - it keeps its own fixed-pitch default, which is
 * why the conversation pane came out monospaced while the menu bar did
 * not. Laying out for one font and asking the system for another is the
 * kind of mismatch nobody reports as a bug; it just looks wrong.
 *
 * English keeps the stock object: MS Sans Serif is what a Win9x English
 * system puts there, and asking for SimSun on a system that has no
 * Chinese font installed gets a substitution nobody chose.
 *
 * Height from LOGPIXELSY rather than a hardcoded -12, so a 120-DPI
 * display gets 9 points rather than 9/1.25 points. */
static void build_font(int english) {
    HFONT f = NULL;
    if (!english) {
        HDC dc = GetDC(NULL);
        int h = dc ? -MulDiv(9, GetDeviceCaps(dc, LOGPIXELSY), 72) : -12;
        if (dc) ReleaseDC(NULL, dc);
        f = CreateFontA(h, 0, 0, 0, FW_NORMAL, 0, 0, 0,
                        GB2312_CHARSET, OUT_DEFAULT_PRECIS,
                        CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
                        DEFAULT_PITCH | FF_DONTCARE, FACE_SONGTI);
    }
    if (g_font_owned && g_font) DeleteObject((HGDIOBJ)g_font);
    if (f) {
        g_font = f;
        g_font_owned = 1;
    } else {
        /* DEFAULT_GUI_FONT is stock object 17, added in 4.0. On 3.51
           GetStockObject returns NULL for it, so the fallback below is
           not a version test - it is the API's own answer. Stock
           objects are never deleted, hence g_font_owned = 0. */
        g_font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
        if (!g_font) g_font = (HFONT)GetStockObject(ANSI_VAR_FONT);
        if (!g_font) g_font = (HFONT)GetStockObject(SYSTEM_FONT);
        g_font_owned = 0;
    }
}

static void probe(void) {
    DWORD v;
    HMODULE user32;
    if (g_probed) return;
    g_probed = 1;

    v = GetVersion();
    g_major = (int)(LOBYTE(LOWORD(v)));

    /* Chinese by default, because lz_str_init's default is Chinese. A
       caller that switches language calls lz_ui_set_font_lang. */
    build_font(0);

    user32 = GetModuleHandleA("user32.dll");
    if (user32) {
        p_GetScrollInfo = (GetScrollInfoFn)LZ_PROC(user32,
                                                          "GetScrollInfo");
        p_LoadImageA = (LoadImageFn)LZ_PROC(user32, "LoadImageA");
        p_GetKeyboardLayout = (GetKeyboardLayoutFn)
            LZ_PROC(user32, "GetKeyboardLayout");
        p_DrawEdge = (DrawEdgeFn)LZ_PROC(user32, "DrawEdge");
        p_SystemParametersInfo = (SysParamFn)
            LZ_PROC(user32, "SystemParametersInfoA");
        p_DrawFrameControl = (DrawFrameCtlFn)
            LZ_PROC(user32, "DrawFrameControl");
        p_DrawIconEx = (DrawIconExFn)LZ_PROC(user32, "DrawIconEx");
    }

    g_imm = LoadLibraryA("imm32.dll");
    if (g_imm) {
        p_ImmGetContext = (ImmGetContextFn)
            LZ_PROC(g_imm, "ImmGetContext");
        p_ImmGetCompositionString = (ImmGetCompStrFn)
            LZ_PROC(g_imm, "ImmGetCompositionStringA");
        p_ImmReleaseContext = (ImmReleaseFn)
            LZ_PROC(g_imm, "ImmReleaseContext");
    }

    g_shell = LoadLibraryA("shell32.dll");
    if (g_shell) {
        p_SHBrowseForFolder = (BrowseFn)
            LZ_PROC(g_shell, "SHBrowseForFolderA");
        p_SHGetPathFromIDList = (PathFromIDLFn)
            LZ_PROC(g_shell, "SHGetPathFromIDListA");
        p_SHGetMalloc = (GetMallocFn)LZ_PROC(g_shell, "SHGetMalloc");
        p_DragAcceptFiles = (DragAcceptFn)
            LZ_PROC(g_shell, "DragAcceptFiles");
        p_DragQueryFileA = (DragQueryFileFn)
            LZ_PROC(g_shell, "DragQueryFileA");
        p_DragFinish = (DragFinishFn)LZ_PROC(g_shell, "DragFinish");
    }
}

int lz_os_major(void) { probe(); return g_major; }

/* NULL when user32 has no GetKeyboardLayout export at all (should not
 * happen on any real Windows - this is about the SDK header hiding a
 * declaration, not the DLL lacking the entry point - but nothing here
 * assumes that cannot happen). Callers that need "restore the original
 * layout" already have to tolerate a NULL return the same way they
 * tolerate LoadKeyboardLayoutA failing to find a layout to switch to. */
HKL lz_kbd_layout_get(void) {
    probe();
    return p_GetKeyboardLayout ? p_GetKeyboardLayout(0) : NULL;
}

/* The menu button's chicken icon. LoadImage is a Win95 addition; on
   3.51 there is no LoadImageA at all, so this returns NULL and the
   caller leaves the button as its plain labelled text - the same
   graceful degradation as every other 4.0 nicety in this file. */
HICON lz_ui_icon_16(HINSTANCE inst) {
    probe();
    if (!p_LoadImageA) return NULL;
    return p_LoadImageA(inst, MAKEINTRESOURCE(IDI_APP), IMAGE_ICON,
                        16, 16, 0);
}

const char *lz_statusbar_class(void) {
    typedef BOOL (WINAPI *IcexFn)(const void *);
    IcexFn icex;
    if (g_sbar_class) return g_sbar_class;
    if (g_comctl == NULL)
        g_comctl = LoadLibraryA("comctl32.dll");
    if (!g_comctl) return NULL;
    icex = (IcexFn)LZ_PROC(g_comctl, "InitCommonControlsEx");
    if (!icex) return NULL;
    /* INITCOMMONCONTROLSEX hand-rolled to avoid pulling commctrl.h
       into a file whose only other need for it is this one struct;
       ICC_BAR_CLASSES = 0x4 (Win95 comctl32 4.0 already knows it). */
    {
        DWORD icc[2];
        icc[0] = 8;                 /* dwSize */
        icc[1] = 0x4;               /* dwICC = ICC_BAR_CLASSES */
        if (icex(icc)) g_sbar_class = "msctls_statusbar32";
    }
    return g_sbar_class;
}

/* 0 on NT 3.51, where nothing is drawn: the dock groove and the status
   bar's panel bevels are 4.0 furniture, and their absence leaves flat
   surfaces rather than a broken window. Same shape as lz_ui_icon_16
   above. */
int lz_draw_edge(HDC dc, RECT *rc, unsigned edge, unsigned flags) {
    probe();
    if (!p_DrawEdge || !dc || !rc) return 0;
    return p_DrawEdge(dc, rc, (UINT)edge, (UINT)flags) ? 1 : 0;
}

/* ---- title-bar self-drawing's 4.0-era capabilities ---- */

/* 4.0 values, absent from the 3.51 headers. */
#define LZ_SPI_GETNONCLIENTMETRICS 0x0029
#define LZ_DFC_CAPTION             1
#define LZ_DFCS_CAPTIONCLOSE       0x0000
#define LZ_DFCS_CAPTIONMIN         0x0001
#define LZ_DFCS_CAPTIONMAX         0x0002
#define LZ_DFCS_INACTIVE           0x0100
#define LZ_DFCS_PUSHED             0x0200

/* The front of NONCLIENTMETRICSA, up to lfCaptionFont. The full struct is
   absent from the 3.51 headers, and we only need that one field - copying
   the whole thing would freeze a layout that changes with the SDK. cbSize
   must be 4.0's full 340 so the system can version-detect. */
#define LZ_NCM_SIZE_40 340
typedef struct {
    UINT     cbSize;
    int      iBorderWidth, iScrollWidth, iScrollHeight;
    int      iCaptionWidth, iCaptionHeight;
    LOGFONTA lfCaptionFont;
    /* iSmCaptionWidth.. follow, up to 340 bytes total. */
    char     pad[LZ_NCM_SIZE_40 - 24 - 60];
} LzNcm40;

static SysParamFn p_SystemParametersInfo;

int lz_caption_logfont(LOGFONTA *out)
{
    LzNcm40 ncm;
    if (!out) return 0;

    probe();
    if (!p_SystemParametersInfo) return 0;

    memset(&ncm, 0, sizeof ncm);
    ncm.cbSize = LZ_NCM_SIZE_40;
    if (sizeof ncm != LZ_NCM_SIZE_40) return 0;   /* layout wrong: do not ask */
    if (!p_SystemParametersInfo(LZ_SPI_GETNONCLIENTMETRICS, LZ_NCM_SIZE_40,
                                &ncm, 0))
        return 0;
    *out = ncm.lfCaptionFont;
    return 1;
}

static DrawFrameCtlFn p_DrawFrameControl;

int lz_caption_button(HDC dc, const RECT *r, int which,
                      int pushed, int inactive)
{
    RECT tmp = *r;
    UINT f;

    probe();
    if (!p_DrawFrameControl || !dc || !r) return 0;

    f = (which == 0) ? LZ_DFCS_CAPTIONCLOSE
      : (which == 1) ? LZ_DFCS_CAPTIONMAX : LZ_DFCS_CAPTIONMIN;
    if (pushed)   f |= LZ_DFCS_PUSHED;
    if (inactive) f |= LZ_DFCS_INACTIVE;
    return p_DrawFrameControl(dc, &tmp, LZ_DFC_CAPTION, f) ? 1 : 0;
}

int lz_caption_hicolor(void)
{
    HDC dc = GetDC(NULL);
    int bpp = dc ? GetDeviceCaps(dc, BITSPIXEL) * GetDeviceCaps(dc, PLANES) : 1;
    if (dc) ReleaseDC(NULL, dc);
    return bpp > 8;
}

/* WM_GETTITLEBARINFOEX = 0x033F, answered only from Vista on. The struct
   is absent from the 3.51 headers; the layout is fixed: cbSize / rcTitleBar
   / rgstate[6] / rgrect[6]. */
#define LZ_WM_GETTITLEBARINFOEX 0x033F
typedef struct {
    DWORD cbSize;
    RECT  rcTitleBar;
    DWORD rgstate[6];
    RECT  rgrect[6];
} LzTbInfoEx;

int lz_caption_button_width(HWND h)
{
    LzTbInfoEx ti;
    int i;

    memset(&ti, 0, sizeof ti);
    ti.cbSize = sizeof ti;
    SendMessageA(h, LZ_WM_GETTITLEBARINFOEX, 0, (LPARAM)&ti);
    /* rgrect indices: 2 minimize / 3 maximize / 4 help / 5 close. Take the
       first non-empty - all three buttons are the same width, so a missing
       one does not change the answer. */
    for (i = 2; i <= 5; i++)
        if (ti.rgrect[i].right > ti.rgrect[i].left)
            return (int)(ti.rgrect[i].right - ti.rgrect[i].left);
    return GetSystemMetrics(SM_CXSIZE);
}

int lz_caption_icon(HDC dc, HICON ic, int x, int y, int size)
{
    probe();
    if (!p_DrawIconEx || !dc || !ic) return 0;
    return p_DrawIconEx(dc, x, y, ic, size, size, 0, NULL, DI_NORMAL) ? 1 : 0;
}

int lz_os_has_40(void) { probe(); return g_major >= 4; }

HFONT lz_ui_font(void) { probe(); return g_font; }

void lz_ui_set_font_lang(int english) {
    probe();
    english = english ? 1 : 0;
    if (english == g_font_english) return;
    g_font_english = english;
    build_font(english);
}

DWORD lz_ex_style(DWORD want) {
    probe();
    if (g_major >= 4) return want;
    return want & ~(DWORD)(WS_EX_CLIENTEDGE | WS_EX_STATICEDGE);
}

/* RichEdit needs telling twice, and neither message is WM_SETFONT.
 *
 * WM_SETFONT sets a default that RichEdit is free to override per run of
 * text, and it does: IMF_AUTOFONT makes it pick a face per character
 * according to the script that character belongs to. The visible result
 * with a Chinese font selected is a conversation pane where the Chinese
 * is SimSun and every ASCII run - model names, numbers, punctuation -
 * comes out in RichEdit's own fixed-pitch fallback, on the same line.
 * Measured, not deduced: that is exactly what the transcript showed
 * after WM_SETFONT alone.
 *
 * So: turn the linking off (EM_SETLANGOPTIONS without IMF_AUTOFONT),
 * then state the format for ALL text (SCF_ALL), naming the face, the
 * charset and the size together - a CHARFORMAT that names a face but
 * leaves the charset unset gets the face re-resolved per script anyway.
 *
 * Message and flag numbers are written out rather than taken from
 * richedit.h: this file compiles at the 4.0 declaration level and
 * richedit.h is not part of that contract. WM_USER is 0x400.
 *   EM_SETCHARFORMAT   WM_USER + 68
 *   EM_SETLANGOPTIONS  WM_USER + 120
 *   EM_GETLANGOPTIONS  WM_USER + 121
 *   SCF_ALL            0x0004
 *   IMF_AUTOFONT       0x0002
 * The CHARFORMAT layout below is the 1.0 (CHARFORMATA) one, which
 * RichEdit 2.0 still accepts - cbSize is how it tells them apart, and
 * the floor may be riched32. */
void lz_richedit_use_font(HWND h, HFONT f) {
    struct {
        UINT  cbSize;
        DWORD dwMask;
        DWORD dwEffects;
        LONG  yHeight;
        LONG  yOffset;
        COLORREF crTextColor;
        BYTE  bCharSet;
        BYTE  bPitchAndFamily;
        char  szFaceName[32];
    } cf;
    LOGFONTA lf;
    LONG opt;
    HDC dc;
    int dpi;

    if (!h || !f) return;
    if (!GetObjectA((HGDIOBJ)f, (int)sizeof lf, &lf)) return;

    opt = (LONG)SendMessageA(h, WM_USER + 121, 0, 0);
    SendMessageA(h, WM_USER + 120, 0, (LPARAM)(opt & ~0x0002L));

    dc = GetDC(NULL);
    dpi = dc ? GetDeviceCaps(dc, LOGPIXELSY) : 96;
    if (dc) ReleaseDC(NULL, dc);

    memset(&cf, 0, sizeof cf);
    cf.cbSize = (UINT)sizeof cf;
    /* CFM_FACE 0x20000000 | CFM_CHARSET 0x08000000 | CFM_SIZE 0x80000000 */
    cf.dwMask = 0x20000000L | 0x08000000L | 0x80000000L;
    /* yHeight is in TWIPS - 1/20 of a point - while lfHeight is pixels
       and negative for "character height". Converting through the DC's
       real DPI keeps 9pt at 9pt on a 120-DPI display. */
    {
        long px = lf.lfHeight < 0 ? -lf.lfHeight : lf.lfHeight;
        cf.yHeight = (LONG)(px * 1440 / (dpi > 0 ? dpi : 96));
    }
    cf.bCharSet = lf.lfCharSet;
    cf.bPitchAndFamily = lf.lfPitchAndFamily;
    lstrcpynA(cf.szFaceName, lf.lfFaceName, (int)sizeof cf.szFaceName);
    SendMessageA(h, WM_USER + 68, 0x0004, (LPARAM)&cf);
}

const char *lz_richedit_class(void) {
    probe();
    if (g_rich_class) return g_rich_class;
    /* 2.0 first: it is what the deliverable targets, and 1.0 is the
       floor's consolation prize rather than a preference. */
    g_riched = LoadLibraryA("riched20.dll");
    if (g_riched) { g_rich_class = "RichEdit20A"; return g_rich_class; }
    g_riched = LoadLibraryA("riched32.dll");
    if (g_riched) { g_rich_class = "RichEdit"; return g_rich_class; }
    return NULL;
}

HBITMAP lz_mapped_bitmap(HINSTANCE inst, int id) {
    /* comctl32's CreateMappedBitmap, which is the era's answer to "this
       artwork has to sit on whatever colour the user picked for
       buttons": it swaps c0c0c0 for COLOR_BTNFACE, 808080 for
       COLOR_BTNSHADOW and ffffff for COLOR_BTNHIGHLIGHT as it loads.
       The same convention the toolbar bitmaps use.
       It matters more than it sounds: a Win9x button face IS c0c0c0, so
       a hardcoded background looks perfect on the target and shows as a
       grey square everywhere else - Windows 10 and 11 use f0f0f0.
       Falls back to LoadBitmap when comctl32 has no such export, which
       is the 3.51 case and the case where there is no status bar to put
       a lamp in either. */
    typedef HBITMAP (WINAPI *MappedBmpFn)(HINSTANCE, LZ_IPTR_T, UINT,
                                          void *, int);
    static MappedBmpFn p_CreateMappedBitmap;
    static int tried;
    if (!tried) {
        tried = 1;
        lz_statusbar_class();          /* ensures comctl32 is loaded */
        if (g_comctl)
            p_CreateMappedBitmap =
                (MappedBmpFn)LZ_PROC(g_comctl, "CreateMappedBitmap");
    }
    if (p_CreateMappedBitmap)
        return p_CreateMappedBitmap(inst, (LZ_IPTR_T)id, 0, NULL, 0);
    return LoadBitmapA(inst, MAKEINTRESOURCE(id));
}

int lz_ui_untheme(HWND h) {
    /* uxtheme's SetWindowTheme(h, L"", L"") turns theming off for one
       window, OF ANY CLASS - not an EDIT-specific call, which is why
       this function is named lz_ui_untheme. It exists from XP; on the
       target it is simply absent and this does nothing, which is
       correct - Windows 98 has no theme to remove.
       Why a text box needed it: this program ships a
       Microsoft.Windows.Common-Controls 6.0 manifest, so on XP and later
       uxtheme repaints user32's EDIT with a flat one-pixel border.
       RichEdit is NOT user32 and NOT comctl32 - riched20 draws its own
       classic sunken edge, and uxtheme cannot override THAT - so the two
       boxes in the main window ended up with different borders. Measured
       on the shipped build: the conversation's left edge runs 190/139/124
       (the 3D bevel), the input box's 237/243 (the themed line).
       Theming the RichEdit's BORDER is not an option (see this
       function's own header comment for the WM_NCPAINT attempt that did
       not work), so the edit is untheme'd down to it.
       For the RichEdit case specifically the call does not untheme the
       whole control: the RichEdit carries WS_VSCROLL, and that scrollbar
       is a standard NONCLIENT Windows scrollbar, not something riched20
       draws itself - uxtheme themes it the same generic way it themes
       any other window's, regardless of what client-area control owns
       the window. This same call is what turns THAT off too, once the
       transcript control passes through it (gui/main.c).
       The settings dialog (gui/settingsdlg.c) needs the identical call
       on a SCROLLBAR and on several BUTTONs (a checkbox, three
       pushbuttons) too - that dialog would otherwise look half-classic,
       half-themed next to the main window. A name that says "edit"
       invites exactly the "buttons don't need this" reasoning that is
       precisely wrong here. */
    typedef long (WINAPI *SetWindowThemeFn)(HWND, const unsigned short *,
                                            const unsigned short *);
    static HMODULE ux;
    static SetWindowThemeFn p_SetWindowTheme;
    static int tried;
    static const unsigned short EMPTY[1] = { 0 };

    if (!h) return 0;
    if (!tried) {
        tried = 1;
        ux = LoadLibraryA("uxtheme.dll");
        if (ux)
            p_SetWindowTheme =
                (SetWindowThemeFn)LZ_PROC(ux, "SetWindowTheme");
    }
    if (!p_SetWindowTheme) return 0;
    return p_SetWindowTheme(h, EMPTY, EMPTY) == 0;   /* S_OK */
}

void lz_edit_use_font_margins(HWND edit) {
    /* EM_SETMARGINS 0x00D3, EC_LEFTMARGIN|EC_RIGHTMARGIN = 3,
       EC_USEFONTINFO = 0xFFFF. Hand-written for the same reason as the
       rest of this file: the names are behind WINVER >= 0x0400 and the
       floor compiles below it. */
    if (!edit) return;
    SendMessage(edit, 0x00D3, 3, MAKELPARAM(0xFFFF, 0xFFFF));
}

int lz_ime_composing(HWND h) {
    HIMC imc;
    LONG n;
    probe();
    /* No imm32 means no IME, which means nothing is being composed.
       Answering "unknown" here would have to become "assume composing",
       and that would stop Enter from ever sending. */
    if (!p_ImmGetContext || !p_ImmGetCompositionString) return 0;
    imc = p_ImmGetContext(h);
    if (!imc) return 0;
    n = p_ImmGetCompositionString(imc, GCS_COMPSTR, NULL, 0);
    if (p_ImmReleaseContext) p_ImmReleaseContext(h, imc);
    return n > 0;
}

int lz_scroll_range_at_end(int max, unsigned page, int pos) {
    /* A page of zero is the bar reporting that it is disabled: the text
       fits, so there is nothing to have scrolled away from, so the view
       IS at the end. Reading that tuple as a position instead is why
       auto-scroll never starts. Measured on the running program
       (build/gate/shot.ps1 -Scroll 1001): an EMPTY riched20 transcript
       answers min=0 max=16 page=0 pos=0, so a bare pos + page >= max
       comparison is false before the first token of the first reply;
       append_run() asks only before appending and scrolls only on yes.

       An isolated riched20 does NOT reproduce this: there the empty
       control has no scroll bar at all and lz_scroll_at_end returns 1 on
       the GetScrollInfo failure above. Something the program does to its
       own transcript materialises the bar early - the reason the gate
       feeds the measured tuple rather than trying to build one.

       Split out from lz_scroll_at_end so it can be gated without a
       window, the same reason lz_drop_dir_of is exported. */
    if (page == 0) return 1;
    return pos + (int)page >= max;
}

int lz_scroll_at_end(HWND h) {
    probe();
    if (p_GetScrollInfo) {
        SCROLLINFO si;
        memset(&si, 0, sizeof si);
        si.cbSize = sizeof si;
        si.fMask = SIF_ALL;
        if (!p_GetScrollInfo(h, SB_VERT, &si)) return 1;
        return lz_scroll_range_at_end(si.nMax, si.nPage, si.nPos);
    }
    {
        /* Coarser: no page size, so "at the end" means "at the maximum
           position". For a control whose range is set the usual way -
           maximum position is the last scrollable line - that is the
           same question with one fewer term. */
        int lo = 0, hi = 0;
        GetScrollRange(h, SB_VERT, &lo, &hi);
        if (hi <= lo) return 1;                /* nothing to scroll */
        return GetScrollPos(h, SB_VERT) >= hi;
    }
}

/* Build commdlg's double-NUL-terminated filter pair. */
static int build_filter(char *buf, int cap, const char *desc,
                        const char *pattern) {
    int n = 0;
    if ((int)(strlen(desc) + strlen(pattern) + 3) > cap) return 0;
    n += (int)strlen(strcpy(buf + n, desc)) + 1;
    n += (int)strlen(strcpy(buf + n, pattern)) + 1;
    buf[n] = '\0';
    return 1;
}

int lz_pick_save_file(HWND owner, const char *title, const char *filter_desc,
                      const char *filter_pattern, const char *default_ext,
                      char *out, int cap) {
    OPENFILENAMEA ofn;
    char path[MAX_PATH];
    char filter[160];

    probe();
    if (!build_filter(filter, (int)sizeof filter, filter_desc,
                      filter_pattern))
        return 0;
    path[0] = '\0';
    memset(&ofn, 0, sizeof ofn);
    ofn.lStructSize = sizeof ofn;
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = path;
    ofn.nMaxFile = (DWORD)sizeof path;
    ofn.lpstrTitle = title;
    ofn.lpstrDefExt = default_ext;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
    if (!GetSaveFileNameA(&ofn)) return 0;
    strncpy(out, path, (size_t)cap - 1);
    out[cap - 1] = '\0';
    return 1;
}

int lz_pick_open_file(HWND owner, const char *title, const char *filter_desc,
                      const char *filter_pattern, char *out, int cap) {
    OPENFILENAMEA ofn;
    char path[MAX_PATH];
    char filter[160];

    probe();
    if (!build_filter(filter, (int)sizeof filter, filter_desc,
                      filter_pattern))
        return 0;
    path[0] = '\0';
    memset(&ofn, 0, sizeof ofn);
    ofn.lStructSize = sizeof ofn;
    ofn.hwndOwner = owner;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = path;
    ofn.nMaxFile = (DWORD)sizeof path;
    ofn.lpstrTitle = title;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
    if (!GetOpenFileNameA(&ofn)) return 0;
    strncpy(out, path, (size_t)cap - 1);
    out[cap - 1] = '\0';
    return 1;
}

int lz_pick_folder(HWND owner, const char *title, const char *filter_desc,
                   const char *filter_file, char *out, int cap) {
    probe();

    if (p_SHBrowseForFolder && p_SHGetPathFromIDList) {
        BROWSEINFOA bi;
        LPITEMIDLIST idl;
        char path[MAX_PATH];
        memset(&bi, 0, sizeof bi);
        bi.hwndOwner = owner;
        bi.pszDisplayName = path;
        bi.lpszTitle = title;
        bi.ulFlags = BIF_RETURNONLYFSDIRS;
        idl = p_SHBrowseForFolder(&bi);
        if (!idl) return 0;
        path[0] = '\0';
        if (!p_SHGetPathFromIDList(idl, path)) path[0] = '\0';
        if (p_SHGetMalloc) {
            LPMALLOC mal = NULL;
            if (p_SHGetMalloc(&mal) == 0 && mal) {
                mal->lpVtbl->Free(mal, idl);
                mal->lpVtbl->Release(mal);
            }
        }
        if (!path[0]) return 0;
        strncpy(out, path, (size_t)cap - 1);
        out[cap - 1] = '\0';
        return 1;
    }

    {
        /* No folder browser - NT 3.51's shell32 predates it. Ask for the
           model file instead and keep its directory. Arguably the better
           dialog anyway: the thing that makes a directory a model is the
           presence of that exact file, so picking it IS the pre-check. */
        OPENFILENAMEA ofn;
        char path[MAX_PATH];
        char filter[160];
        int i, cut = -1;

        if (!build_filter(filter, (int)sizeof filter, filter_desc,
                          filter_file))
            return 0;
        path[0] = '\0';
        memset(&ofn, 0, sizeof ofn);
        ofn.lStructSize = sizeof ofn;
        ofn.hwndOwner = owner;
        ofn.lpstrFilter = filter;
        ofn.lpstrFile = path;
        ofn.nMaxFile = (DWORD)sizeof path;
        ofn.lpstrTitle = title;
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
        if (!GetOpenFileNameA(&ofn)) return 0;

        for (i = 0; path[i]; i++)
            if (path[i] == '\\' || path[i] == '/') cut = i;
        if (cut <= 0) return 0;
        path[cut] = '\0';
        strncpy(out, path, (size_t)cap - 1);
        out[cap - 1] = '\0';
        return 1;
    }
}

int lz_drop_accept(HWND h, int on) {
    probe();
    if (!h || !p_DragAcceptFiles) return 0;
    p_DragAcceptFiles(h, on ? TRUE : FALSE);
    return 1;
}

int lz_drop_first_path(WPARAM hdrop, char *out, int cap) {
    UINT n;
    if (!out || cap <= 0) return 0;
    out[0] = '\0';
    probe();
    if (!hdrop || !p_DragQueryFileA) return 0;
    n = p_DragQueryFileA((HDROP)hdrop, 0, out, (UINT)cap);
    return n > 0;
}

void lz_drop_finish(WPARAM hdrop) {
    probe();
    if (hdrop && p_DragFinish) p_DragFinish((HDROP)hdrop);
}

int lz_drop_dir_of(const char *path, char *out, int cap) {
    DWORD a;
    int n, i;
    if (!path || !out || cap <= 0) return 0;
    n = (int)strlen(path);
    if (n <= 0 || n >= cap) return 0;
    strcpy(out, path);
    /* A trailing separator is not part of the name. "C:\" is the one
       place it is, so stop at 3. */
    while (n > 3 && (out[n - 1] == '\\' || out[n - 1] == '/')) out[--n] = '\0';
    a = GetFileAttributesA(out);
    if (a != 0xFFFFFFFF && (a & FILE_ATTRIBUTE_DIRECTORY)) return 1;
    for (i = n - 1; i > 0; i--)
        if (out[i] == '\\' || out[i] == '/') { out[i] = '\0'; return 1; }
    return 0;
}

/* Find in the conversation. See compat40.h for why both of
 * these exist: gui/main.c cannot include commdlg.h at the 3.51 floor,
 * so FINDREPLACEA is confined to this file entirely. */
HWND lz_find_open(HWND owner, char *needle_buf, int needle_cap) {
    /* Static, not a local: comdlg32 keeps writing into this struct
     * (and into needle_buf, the caller's own buffer) for as long as
     * the dialog stays open, well after this call returns. A stack
     * FINDREPLACEA here would have the dialog scribbling into a frame
     * that no longer exists the moment lz_find_open returns. */
    static FINDREPLACEA fr;
    if (!needle_buf || needle_cap <= 0) return NULL;
    memset(&fr, 0, sizeof fr);
    fr.lStructSize = sizeof fr;
    fr.hwndOwner = owner;
    fr.lpstrFindWhat = needle_buf;
    fr.wFindWhatLen = (WORD)needle_cap;
    fr.Flags = FR_DOWN;
    return FindTextA(&fr);
}

int lz_find_parse(LPARAM lp, int *down, int *match_case,
                  const char **needle) {
    LPFINDREPLACEA fr = (LPFINDREPLACEA)lp;
    if (!fr) return 0;
    if (fr->Flags & FR_DIALOGTERM) return 1;
    if (fr->Flags & FR_FINDNEXT) {
        if (down) *down = (fr->Flags & FR_DOWN) != 0;
        if (match_case) *match_case = (fr->Flags & FR_MATCHCASE) != 0;
        if (needle) *needle = fr->lpstrFindWhat;
        return 2;
    }
    return 0;
}
