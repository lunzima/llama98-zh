/* Startup splash. See splash.h for why there is no palette code.
 */
#include <string.h>

#include "compat40.h"
#include "resource.h"
#include "splash.h"

#define SPLASH_CLASS   "Kunkun98Splash"
#define SPLASH_MIN_MS  800

static HBITMAP g_bmp;
static int g_w, g_h;
static DWORD g_shown_at;
static int g_registered;

static int load_bitmap(HINSTANCE inst) {
    BITMAP bm;
    if (g_bmp) return 1;
    /* LoadBitmapA, not LoadImage with LR_CREATEDIBSECTION: the latter is
       a 4.0 addition and would be hidden at the floor, and a DIB section
       would only be needed if the palette had to be realised - which it
       does not, see splash.h. */
    g_bmp = LoadBitmapA(inst, MAKEINTRESOURCEA(IDB_SPLASH));
    if (!g_bmp) return 0;
    if (!GetObjectA(g_bmp, sizeof bm, &bm)) {
        DeleteObject(g_bmp);
        g_bmp = NULL;
        return 0;
    }
    g_w = (int)bm.bmWidth;
    g_h = (int)bm.bmHeight;
    return 1;
}

int lz_gui_splash_size(HINSTANCE inst, int *w, int *h) {
    if (!load_bitmap(inst)) return 0;
    if (w) *w = g_w;
    if (h) *h = g_h;
    return 1;
}

static LRESULT CALLBACK splashproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        HDC mem = CreateCompatibleDC(dc);
        HGDIOBJ old = SelectObject(mem, g_bmp);
        BitBlt(dc, 0, 0, g_w, g_h, mem, 0, 0, SRCCOPY);
        SelectObject(mem, old);
        DeleteDC(mem);
        EndPaint(h, &ps);
        return 0;
    }
    return DefWindowProcA(h, msg, wp, lp);
}

HWND lz_gui_splash_show(HINSTANCE inst) {
    WNDCLASSA wc;
    HWND h;
    int x, y;

    if (!load_bitmap(inst)) return NULL;

    if (!g_registered) {
        memset(&wc, 0, sizeof wc);
        wc.lpfnWndProc = splashproc;
        wc.hInstance = inst;
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.lpszClassName = SPLASH_CLASS;
        if (!RegisterClassA(&wc)) return NULL;
        g_registered = 1;
    }

    /* Centred on the screen, no caption, no border - a splash with a
       title bar is a window, and this is a picture. */
    x = (GetSystemMetrics(SM_CXSCREEN) - g_w) / 2;
    y = (GetSystemMetrics(SM_CYSCREEN) - g_h) / 2;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    h = CreateWindowExA(0, SPLASH_CLASS, "", WS_POPUP, x, y, g_w, g_h,
                        NULL, NULL, inst, NULL);
    if (!h) return NULL;
    ShowWindow(h, SW_SHOW);
    /* Paint it NOW rather than waiting for the message loop: the caller
       is about to spend its time creating the main window, and a splash
       that only appears once that work is done has missed the point. */
    UpdateWindow(h);
    g_shown_at = GetTickCount();
    return h;
}

void lz_gui_splash_close(HWND h) {
    if (!h) return;
    for (;;) {
        /* Unsigned subtraction, so the 49-day GetTickCount wrap is
           handled by the arithmetic rather than by hoping. */
        DWORD elapsed = GetTickCount() - g_shown_at;
        if (elapsed >= SPLASH_MIN_MS) break;
        Sleep(SPLASH_MIN_MS - elapsed);
    }
    DestroyWindow(h);
    if (g_bmp) { DeleteObject(g_bmp); g_bmp = NULL; }
}
