/* Title bar self-drawing: the Win32 half. Sequence, truncation and compose
 * live in gui/caption.c; this file executes them and paints with a DC.
 * Reference implementation: E:/LLM/_capdemo/capdemo.c's w95_* group,
 * verified on Win11 + WIN98 compatibility layer. Field-checked against it,
 * not re-derived from memory (spec's risk #1).
 */
#include <windows.h>
#include <string.h>
#include "caption.h"
#include "compat40.h"

/* Constants hidden behind WINVER=0x0351. Values are Win9x-standard, verified
   against the dev box's winuser.h. SM_CXSMICON is a 4.0-era metric; the rest
   are the ordinary non-client/message constants the 3.51 headers just do not
   spell out. Hand-declared here for the same reason compat40.c spells its own
   4.0 values - the value is fixed by the ABI, the header is not. */
#ifndef SM_CXSMICON
#define SM_CXSMICON 49
#endif
#ifndef DI_NORMAL
#define DI_NORMAL 3
#endif
#ifndef HTCLOSE
#define HTCLOSE 20
#endif
#ifndef WM_SETTINGCHANGE
#define WM_SETTINGCHANGE 0x001A
#endif

/* Not in any SDK header - they are uxtheme's private traffic, and the only
   published description of them is other people's reverse engineering. The
   values are fixed by the ABI, same argument as the block above. */
#define LZ_WM_NCUAHDRAWCAPTION 0x00AE
#define LZ_WM_NCUAHDRAWFRAME   0x00AF

static HWND     g_hwnd  = NULL;
static WNDPROC  g_orig  = NULL;
static int      g_depth = 0;
static int      g_active = 1;   /* see sub() */
static LZCaption g_cap;
static LZCapWidths g_w;
static HFONT    g_f_brand = NULL, g_f_rest = NULL;

int lz_caption_can_paint(void)
{
    /* Always paint: the title bar self-draws on every platform, so a
       human can see it on the dev box too. */
    return 1;
}

/* ---- fonts ---- */

/* Brand segment: Arial Bold Italic, height from the system caption font.
   LZ_CAPTION_FALLBACK_FACE == 1 takes the pre-approved fallback branch -
   lfFaceName left empty so GDI picks by charset, giving slanted SimSun.
   Flip this one constant when the Win98 VM result is in. */
#ifndef LZ_CAPTION_FALLBACK_FACE
#define LZ_CAPTION_FALLBACK_FACE 0
#endif

static void build_fonts(void)
{
    LOGFONTA lf;

    if (g_f_brand) { DeleteObject(g_f_brand); g_f_brand = NULL; }
    if (g_f_rest)  { DeleteObject(g_f_rest);  g_f_rest  = NULL; }

    if (!lz_caption_logfont(&lf)) {
        memset(&lf, 0, sizeof lf);
        lf.lfHeight = -12;
    }
    g_f_rest = CreateFontIndirectA(&lf);

#if LZ_CAPTION_FALLBACK_FACE
    lf.lfFaceName[0] = '\0';
#else
    lstrcpynA(lf.lfFaceName, "Arial", LF_FACESIZE);
#endif
    lf.lfItalic         = 1;
    lf.lfWeight         = FW_BOLD;
    lf.lfCharSet        = ANSI_CHARSET;
    lf.lfPitchAndFamily = VARIABLE_PITCH;
    g_f_brand = CreateFontIndirectA(&lf);
    if (!g_f_brand) {              /* MSO95 has this same retry */
        lf.lfPitchAndFamily = FIXED_PITCH;
        g_f_brand = CreateFontIndirectA(&lf);
    }
}

/* ---- geometry: one copy, feeding both paint and hit-test ---- */

static void btn_rect(HWND h, int which, RECT *b)
{
    RECT wr;
    int bw, bh, right, top;
    GetWindowRect(h, &wr);
    /* Width asked of the frame: SM_CXSIZE is not the button width on modern
       frames (this box reports 36, the real buttons are 24), and drawing at
       36 puts the buttons where they do not hit. The other metrics stay
       MSO95's, which still mean what they always meant. */
    bw    = lz_caption_button_width(h);
    bh    = GetSystemMetrics(SM_CYSIZE) - 4;
    top   = GetSystemMetrics(SM_CYFRAME) + 2;
    right = (int)(wr.right - wr.left) - GetSystemMetrics(SM_CXFRAME) - 2;
    if (which == 1)      right -= bw + 2;      /* max */
    else if (which == 2) right -= bw * 2 + 2;  /* min */
    b->right = right; b->left = right - bw;
    b->top = top;     b->bottom = top + bh;
}

/* Paint all three, fixed order, pushed_which marks the depressed one (-1 =
   none). One paint entry point so full repaint and pressed repaint cannot
   draw two different pictures. */
static void paint_buttons(HDC dc, HWND h, int active, int pushed_which)
{
    RECT b;
    int i;
    for (i = 0; i <= 2; i++) {
        btn_rect(h, i, &b);
        lz_caption_button(dc, &b, i, i == pushed_which, !active);
    }
}

static void draw_buttons(HWND h, int pushed_which)
{
    HDC dc = GetWindowDC(h);
    paint_buttons(dc, h, g_active, pushed_which);
    ReleaseDC(h, dc);
}

/* A point carried by a mouse message, in window coords.
   Deliberately not GetCursorPos: this loop must be drivable from the
   message queue alone, or the gate would need the user's real mouse. It is
   also more faithful - the loop replays the input stream, and the physical
   cursor may have moved on by the time a message dequeues. */
static void msg_point(HWND h, UINT m, LPARAM l, POINT *p)
{
    RECT wr;
    p->x = (int)(short)LOWORD(l);
    p->y = (int)(short)HIWORD(l);
    if (m == WM_MOUSEMOVE || m == WM_LBUTTONUP)
        ClientToScreen(h, p);           /* those two are client coords */
    GetWindowRect(h, &wr);
    p->x -= wr.left;
    p->y -= wr.top;
}

/* The press also runs its own loop, per spec. DefWindowProc's is modal and
   draws the pressed and released states with the frame's own style - by the
   time the loop returns it has drawn twice, and "repaint afterwards" does
   not exist. */
static void track_button(HWND h, int which)
{
    RECT b;
    int  in = 1, was = 1;
    MSG  msg;

    btn_rect(h, which, &b);
    SetCapture(h);
    draw_buttons(h, which);

    for (;;) {
        if (GetCapture() != h) { in = 0; break; }
        if (!GetMessageA(&msg, NULL, 0, 0)) {
            PostQuitMessage((int)msg.wParam);
            in = 0;
            break;
        }
        if (msg.message == WM_MOUSEMOVE || msg.message == WM_NCMOUSEMOVE ||
            msg.message == WM_LBUTTONUP || msg.message == WM_NCLBUTTONUP) {
            POINT p;
            msg_point(h, msg.message, msg.lParam, &p);
            in = (p.x >= b.left && p.x < b.right &&
                  p.y >= b.top  && p.y < b.bottom);
            if (msg.message == WM_LBUTTONUP || msg.message == WM_NCLBUTTONUP)
                break;
            if (in != was) { draw_buttons(h, in ? which : -1); was = in; }
            continue;
        }
        if (msg.message == WM_KEYDOWN && msg.wParam == VK_ESCAPE) {
            in = 0;                     /* Win9x cancels this way */
            break;
        }
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }

    ReleaseCapture();
    draw_buttons(h, -1);

    if (in) {
        UINT sc = (which == 0) ? SC_CLOSE
                : (which == 1) ? (IsZoomed(h) ? SC_RESTORE : SC_MAXIMIZE)
                               : SC_MINIMIZE;
        /* Post, not Send: the loop above is still on the stack. */
        PostMessageA(h, WM_SYSCOMMAND, (WPARAM)sc, 0);
    }
}

static void bar_rect(HWND h, RECT *rc)
{
    RECT wr;
    GetWindowRect(h, &wr);
    rc->left   = GetSystemMetrics(SM_CXFRAME);
    rc->top    = GetSystemMetrics(SM_CYFRAME);
    rc->right  = (int)(wr.right - wr.left) - GetSystemMetrics(SM_CXFRAME);
    rc->bottom = rc->top + GetSystemMetrics(SM_CYCAPTION);
}

static void text_rect(HWND h, RECT *rc)
{
    RECT b;
    bar_rect(h, rc);
    rc->left += GetSystemMetrics(SM_CXSMICON) + 2;
    btn_rect(h, 2, &b);
    rc->right = (int)b.left - 2;
}

/* ---- measure ---- */

static void measure(void)
{
    HDC dc;
    HFONT oldf;
    SIZE s;

    if (!g_hwnd || !g_f_brand || !g_f_rest) return;
    dc   = GetWindowDC(g_hwnd);
    oldf = (HFONT)SelectObject(dc, g_f_brand);
    GetTextExtentPoint32A(dc, g_cap.brand, (int)strlen(g_cap.brand), &s);
    g_w.brand = s.cx;
    SelectObject(dc, g_f_rest);
    GetTextExtentPoint32A(dc, " - ", 3, &s);   g_w.sep_chat  = s.cx;
    GetTextExtentPoint32A(dc, " [", 2, &s);    g_w.sep_model = s.cx;
    GetTextExtentPoint32A(dc, "...", 3, &s);   g_w.ellipsis  = s.cx;
    GetTextExtentPoint32A(dc, g_cap.chat, (int)strlen(g_cap.chat), &s);
    g_w.chat = s.cx;
    GetTextExtentPoint32A(dc, g_cap.model, (int)strlen(g_cap.model), &s);
    g_w.model = s.cx + g_w.sep_model;   /* "]" folded in with " [" */
    SelectObject(dc, oldf);
    ReleaseDC(g_hwnd, dc);
}

/* ---- paint ---- */

static void paint(HWND h, int active)
{
    HDC dc;
    RECT bar, rc;
    HBRUSH bg;
    HFONT oldf;
    TEXTMETRICA tm;
    LZCapFit fit;
    HICON ic;
    COLORREF capfg;
    char buf[LZ_CAP_CHAT + 8];
    int x, y, iw, i;
    SIZE s;

    if (!g_f_brand || !g_f_rest) return;
    bar_rect(h, &bar);
    text_rect(h, &rc);
    if (rc.right <= rc.left) return;

    dc = GetWindowDC(h);

    /* The whole caption is ours - base, icon, text, buttons. Filling only
       the text band would leave the system's own base between the fill and
       the buttons, which is the half-way look. MSO95 does the same (draws
       everything, DrawFrameControl IAT 0x506e10ec). */
    {
        COLORREF c0 = RGB(0, 0, 0), c1, text;
        int barw = (int)(bar.right - bar.left);
        int hi = lz_caption_hicolor();
        int n = lz_caption_bands(barw, hi);
        int gcy = (int)(bar.bottom - bar.top) - 6;   /* inset, see spec */

        c1 = GetSysColor(active ? COLOR_ACTIVECAPTION
                                : COLOR_INACTIVECAPTION);

        if (!lz_caption_scheme_on(hi, active)) {
            /* low colour depth + inactive: flat system base + system text */
            c0 = c1;
            text = GetSysColor(COLOR_INACTIVECAPTIONTEXT);
            n = 0;                          /* fall into the flat fill below */
        } else {
            text = active ? RGB(255, 255, 255) : RGB(192, 192, 192);
            if (c1 == RGB(0, 0, 128)) c1 = RGB(0, 0, 226);
            if (lz_caption_invert(hi, active,
                    ((unsigned long)GetRValue(c1) << 16) |
                    ((unsigned long)GetGValue(c1) << 8) | GetBValue(c1))) {
                c0 = RGB(255, 255, 255);
                text = RGB(0, 0, 0);
            }
        }
        capfg = text;

        if (n == 0) {                       /* flat: low-colour inactive, or too narrow */
            bg = CreateSolidBrush(c1);
            FillRect(dc, &bar, bg);
            DeleteObject(bg);
        } else {
            for (i = 0; i < n; i++) {
                RECT bnd;
                int t = lz_caption_band_w(i, n, gcy, hi);
                bnd.left   = bar.left + (int)((long)barw * i / n);
                bnd.right  = bar.left + (int)((long)barw * (i + 1) / n);
                bnd.top    = bar.top;
                bnd.bottom = bar.bottom;
                bg = CreateSolidBrush(
                    RGB(GetRValue(c0) + (GetRValue(c1) - GetRValue(c0)) * t / 255,
                        GetGValue(c0) + (GetGValue(c1) - GetGValue(c0)) * t / 255,
                        GetBValue(c0) + (GetBValue(c1) - GetBValue(c0)) * t / 255));
                FillRect(dc, &bnd, bg);
                DeleteObject(bg);
            }
        }
    }

    iw = GetSystemMetrics(SM_CXSMICON);
    ic = lz_ui_icon_16((HINSTANCE)GetWindowLongPtrA(h, GWLP_HINSTANCE));
    if (ic)
        lz_caption_icon(dc, ic, (int)bar.left + 2,
                        (int)bar.top + ((int)(bar.bottom - bar.top) - iw) / 2,
                        iw);

    paint_buttons(dc, h, active, -1);

    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, capfg);        /* from the gradient calc, not GetSysColor */
    SetTextAlign(dc, TA_TOP | TA_LEFT);

    oldf = (HFONT)SelectObject(dc, g_f_rest);
    GetTextMetricsA(dc, &tm);
    y = (int)rc.top + ((int)(rc.bottom - rc.top) - tm.tmHeight) / 2;

    /* Truncation is computed at paint time: the available width changes with
       the window width, while segment widths are measured once at set time.
       Mixing the two is the easiest bug in this feature. */
    fit = lz_caption_fit(&g_cap, &g_w, (int)(rc.right - rc.left));

    x = (int)rc.left;
    SelectObject(dc, g_f_brand);
    TextOutA(dc, x + 1, y, g_cap.brand, (int)strlen(g_cap.brand));
    GetTextExtentPoint32A(dc, g_cap.brand, (int)strlen(g_cap.brand), &s);
    x += s.cx;

    SelectObject(dc, g_f_rest);
    buf[0] = '\0';
    if (fit.chat_bytes > 0) {
        lstrcatA(buf, " - ");
        memcpy(buf + 3, g_cap.chat, (size_t)fit.chat_bytes);
        buf[3 + fit.chat_bytes] = '\0';
        if (fit.chat_ellipsis) lstrcatA(buf, "...");
    }
    if (fit.show_model) {
        lstrcatA(buf, " [");
        lstrcatA(buf, g_cap.model);
        lstrcatA(buf, "]");
    }
    if (buf[0]) TextOutA(dc, x, y, buf, (int)strlen(buf));

    SelectObject(dc, oldf);
    ReleaseDC(h, dc);
}

/* ---- subclass procedure ---- */

static LRESULT CALLBACK sub(HWND h, UINT m, WPARAM w, LPARAM l)
{
    LZCapStep steps[8];
    int n, i, active;
    LRESULT r = 0;
    LONG style;

    if (m == WM_NCPAINT || m == WM_NCACTIVATE || m == WM_SETTEXT ||
        m == LZ_WM_NCUAHDRAWCAPTION || m == LZ_WM_NCUAHDRAWFRAME) {
        /* Activity is cached in g_active from WM_NCACTIVATE's wParam. Not
           GetActiveWindow: that is per-thread, so another process holding
           focus still reads "active" here, and any forced repaint renders
           an inactive window as active. MSO95 wrote exactly this (0x50601513). */
        if (m == WM_NCACTIVATE) g_active = (w != 0);
        active = g_active;
        style = GetWindowLongA(h, GWL_STYLE);
        n = lz_caption_plan((unsigned)m, lz_caption_can_paint(),
                            (style & WS_VISIBLE) != 0, g_depth, steps, 8);
        if (n < 0) return CallWindowProcA(g_orig, h, m, w, l);
        g_depth++;
        for (i = 0; i < n; i++) {
            switch (steps[i]) {
            case LZ_CAP_CLEAR_VISIBLE:
                style = GetWindowLongA(h, GWL_STYLE);
                SetWindowLongA(h, GWL_STYLE, style & ~WS_VISIBLE);
                break;
            case LZ_CAP_FORWARD:
                r = CallWindowProcA(g_orig, h, m, w, l);
                break;
            case LZ_CAP_RESTORE_VISIBLE:
                /* Touch only that one bit. Writing back a cached whole style
                   silently eats style bits changed during CallWindowProc
                   (per MSO95 0x506013c8). */
                SetWindowLongA(h, GWL_STYLE,
                               GetWindowLongA(h, GWL_STYLE) | WS_VISIBLE);
                break;
            case LZ_CAP_PAINT:
                paint(h, active);
                break;
            }
        }
        g_depth--;
        return r;
    }

    /* Somebody else changed the frame. SetMenu and DrawMenuBar both land
       here with SWP_FRAMECHANGED - gui/main.c's build_menu_bar rebuilds the
       bar on every language switch, model load and MRU update - and
       ShowWindow lands here with SWP_SHOWWINDOW. Both ERASE the non-client
       area and leave the repaint to be picked up later; if that repaint is
       still pending when the next thing happens, what is on screen is an
       unpainted menu strip under the platform's own caption. Forcing it
       here removes the dependency on when the queue next drains.

       The 1x1 rect is not a typo: RDW_FRAME redraws the whole non-client
       area as long as ANY part of the window is invalid, so this repaints
       the frame while invalidating exactly one client pixel. Passing NULL
       would repaint the entire client area on every menu rebuild. */
    if (m == WM_WINDOWPOSCHANGED) {
        const WINDOWPOS *wp = (const WINDOWPOS *)l;
        r = CallWindowProcA(g_orig, h, m, w, l);
        if (wp && lz_caption_frame_dirty((unsigned)wp->flags) &&
            lz_caption_can_paint() && g_depth == 0 &&
            (GetWindowLongA(h, GWL_STYLE) & WS_VISIBLE)) {
            RECT one;
            one.left = 0; one.top = 0; one.right = 1; one.bottom = 1;
            RedrawWindow(h, &one, NULL,
                         RDW_FRAME | RDW_INVALIDATE | RDW_UPDATENOW);
        }
        return r;
    }

    if (m == WM_NCHITTEST && lz_caption_can_paint()) {
        LRESULT hit = CallWindowProcA(g_orig, h, m, w, l);
        POINT p;
        RECT wr, b;
        if (hit == HTCAPTION || hit == HTCLOSE || hit == HTMAXBUTTON ||
            hit == HTMINBUTTON || hit == HTSYSMENU) {
            p.x = (short)LOWORD(l); p.y = (short)HIWORD(l);
            GetWindowRect(h, &wr);
            p.x -= (int)wr.left; p.y -= (int)wr.top;
            for (i = 0; i <= 2; i++) {
                btn_rect(h, i, &b);
                if (p.x >= b.left && p.x < b.right &&
                    p.y >= b.top  && p.y < b.bottom)
                    return (i == 0) ? HTCLOSE
                         : (i == 1) ? HTMAXBUTTON : HTMINBUTTON;
            }
            return HTCAPTION;
        }
        return hit;
    }

    /* Not forwarded: DefWindowProc would redraw the buttons with the frame's
       own style. See track_button. */
    if (m == WM_NCLBUTTONDOWN && lz_caption_can_paint() &&
        (w == HTCLOSE || w == HTMAXBUTTON || w == HTMINBUTTON)) {
        track_button(h, (w == HTCLOSE) ? 0 : (w == HTMAXBUTTON) ? 1 : 2);
        return 0;
    }

    if (m == WM_SETTINGCHANGE) {
        build_fonts();
        measure();
    }

    if (m == WM_DESTROY) {
        if (g_f_brand) { DeleteObject(g_f_brand); g_f_brand = NULL; }
        if (g_f_rest)  { DeleteObject(g_f_rest);  g_f_rest  = NULL; }
        if (g_orig) SetWindowLongPtrA(h, GWLP_WNDPROC, (LONG_PTR)g_orig);
    }

    return CallWindowProcA(g_orig, h, m, w, l);
}

int lz_caption_attach(HWND hwnd)
{
    if (g_hwnd == hwnd) return 1;          /* idempotent */
    g_hwnd = hwnd;
    build_fonts();
    /* SetWindowLongPtrA / GWLP_WNDPROC rather than the older spelling:
       Watcom defines the Ptr form as a macro straight onto SetWindowLongA
       (the 32-bit call Win98 owns), while MinGW's 64-bit headers no longer
       declare GWL_WNDPROC. Same as gui/main.c's input/transcript subclasses. */
    g_orig = (WNDPROC)SetWindowLongPtrA(hwnd, GWLP_WNDPROC, (LONG_PTR)sub);
    return g_orig != NULL;
}

void lz_caption_set(const LZCaption *c)
{
    char flat[LZ_CAP_BRAND + LZ_CAP_CHAT + LZ_CAP_MODEL + 8];

    g_cap = *c;
    /* The flat string fed to SetWindowTextA is NEVER truncated: the taskbar
       and Alt+Tab want the full text, truncation happens only on the paint
       side. The two paths diverge on purpose, not by drift. */
    lz_caption_compose(&g_cap, flat, (int)sizeof flat);
    if (g_hwnd) {
        /* measure BEFORE SetWindowTextA, not after: WM_SETTEXT now paints
           from inside the subclass, and paint reads the segment widths.
           Measuring afterwards would draw one push behind - the truncation
           of the OLD segments applied to the new text. */
        measure();
        SetWindowTextA(g_hwnd, flat);
        /* UPDATENOW, and a 1x1 rect. Without UPDATENOW this only queues a
           WM_NCPAINT, and a queued frame repaint is precisely what does not
           arrive when the thread is busy - startup being the busiest it
           ever is. WM_SETTEXT above already painted; this is the belt to
           that pair of braces, so it must not cost a full client repaint. */
        if (lz_caption_can_paint()) {
            RECT one;
            one.left = 0; one.top = 0; one.right = 1; one.bottom = 1;
            RedrawWindow(g_hwnd, &one, NULL,
                         RDW_FRAME | RDW_INVALIDATE | RDW_UPDATENOW);
        }
    }
}
