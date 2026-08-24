/* Capture a window's caption strip to a BMP, and print the geometry the
 * system reports for it.
 *
 * PrintWindow, not BitBlt from the screen: a screen readback returns
 * whatever is stacked on top, which this repo has already been fooled
 * by once. PW_RENDERFULLCONTENT is asked for but not required - it does
 * not exist before Windows 8.1 and the call still works without it.
 *
 * Usage: capshot <window-title-substring> <out.bmp>
 *
 * The printed numbers are the reading; the bitmap is for looking at.
 * WHAT THE BITMAP DOES NOT SETTLE: whether a legacy window's caption is
 * drawn by the application or by the running Windows. Those look
 * different and only one of them is a reference for an imitation. Check
 * it before drawing any conclusion - Word 95 (class OpusApp) does draw
 * its own, which is what makes it usable as one.
 *
 * MEASURE THE CAPTURE, DO NOT LOOK AT IT. Scanning a column through a
 * caption button gives its top and bottom edges and the gaps either
 * side; scanning a row through the buttons gives their widths and the
 * gaps between them. Both are exact and neither depends on the eye,
 * which cannot tell 2 px from 3 px in a screenshot and has no opinion
 * at all about a version it has not seen side by side.
 */
#include <windows.h>
#include <stdio.h>
#include <string.h>

static char g_want[256];
static HWND g_hit;

static BOOL CALLBACK pick(HWND h, LPARAM lp)
{
    char t[512];
    (void)lp;
    if (!IsWindowVisible(h)) return TRUE;
    if (!GetWindowTextA(h, t, sizeof t)) return TRUE;
    if (strstr(t, g_want)) { g_hit = h; return FALSE; }
    return TRUE;
}

static int save_bmp(HBITMAP bm, int w, int h, const char *path)
{
    BITMAPFILEHEADER fh;
    BITMAPINFOHEADER ih;
    HDC dc = GetDC(NULL);
    int stride = ((w * 3 + 3) / 4) * 4;
    unsigned char *px = (unsigned char *)malloc((size_t)stride * h);
    FILE *f;
    int ok = 0;

    memset(&ih, 0, sizeof ih);
    ih.biSize = sizeof ih; ih.biWidth = w; ih.biHeight = h;
    ih.biPlanes = 1; ih.biBitCount = 24; ih.biCompression = BI_RGB;
    if (px && GetDIBits(dc, bm, 0, (UINT)h, px, (BITMAPINFO *)&ih, DIB_RGB_COLORS)) {
        memset(&fh, 0, sizeof fh);
        fh.bfType = 0x4D42;
        fh.bfOffBits = sizeof fh + sizeof ih;
        fh.bfSize = fh.bfOffBits + (DWORD)stride * h;
        f = fopen(path, "wb");
        if (f) {
            fwrite(&fh, sizeof fh, 1, f);
            fwrite(&ih, sizeof ih, 1, f);
            fwrite(px, (size_t)stride * h, 1, f);
            fclose(f);
            ok = 1;
        }
    }
    free(px);
    ReleaseDC(NULL, dc);
    return ok;
}

int main(int argc, char **argv)
{
    RECT wr, cr;
    POINT o;
    HDC sdc, mdc;
    HBITMAP bm, old;
    int w, h, capsz;
    char cls[256];

    if (argc < 3) { printf("usage: capshot <title-substr> <out.bmp>\n"); return 2; }
    strncpy(g_want, argv[1], sizeof g_want - 1);
    EnumWindows(pick, 0);
    if (!g_hit) { printf("no visible window whose title contains \"%s\"\n", argv[1]); return 1; }

    GetClassNameA(g_hit, cls, sizeof cls);
    GetWindowRect(g_hit, &wr);
    GetClientRect(g_hit, &cr);
    o.x = 0; o.y = 0;
    ClientToScreen(g_hit, &o);

    w = wr.right - wr.left;
    h = wr.bottom - wr.top;
    /* Distance from the top of the window to the top of the client area:
       the whole non-client top strip, frame plus caption. */
    capsz = o.y - wr.top;

    printf("class            %s\n", cls);
    printf("window rect      %ld,%ld %ldx%ld\n",
           (long)wr.left, (long)wr.top, (long)w, (long)h);
    printf("client origin    %ld,%ld\n", (long)o.x, (long)o.y);
    printf("nonclient top    %d px  (frame + caption)\n", capsz);
    printf("SM_CYCAPTION     %d\n", GetSystemMetrics(SM_CYCAPTION));
    printf("SM_CYFRAME       %d\n", GetSystemMetrics(SM_CYFRAME));
    printf("SM_CYSIZE        %d\n", GetSystemMetrics(SM_CYSIZE));
    printf("CYFRAME+CYCAPTION %d\n",
           GetSystemMetrics(SM_CYFRAME) + GetSystemMetrics(SM_CYCAPTION));

    if (h > 64) h = 64;          /* the caption strip is all that matters */
    sdc = GetDC(NULL);
    mdc = CreateCompatibleDC(sdc);
    bm  = CreateCompatibleBitmap(sdc, w, h);
    old = (HBITMAP)SelectObject(mdc, bm);
    if (!PrintWindow(g_hit, mdc, 0))
        printf("note: PrintWindow returned 0 - the image may be blank\n");
    SelectObject(mdc, old);
    if (!save_bmp(bm, w, h, argv[2])) { printf("could not write %s\n", argv[2]); return 1; }
    printf("wrote %s (%dx%d)\n", argv[2], w, h);
    DeleteObject(bm); DeleteDC(mdc); ReleaseDC(NULL, sdc);
    return 0;
}
