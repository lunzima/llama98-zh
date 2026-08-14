#ifndef LZ_GUI_SPLASH_H
#define LZ_GUI_SPLASH_H

#include <windows.h>

/* The startup splash (gui-draft section 11 step 8).
 *
 * assets/splash-256.bmp, 492x283, 8bpp, drawn with one BitBlt. No
 * palette work: make_splash.py quantised it to the Win9x standard
 * palette - 20 system colours plus the 6x6x6 halftone cube - so every
 * colour it needs is already realised on a 256-colour desktop. A splash
 * with its own optimal palette would need SelectPalette and
 * RealizePalette, and would still flash whenever another window took the
 * foreground.
 *
 * It has NO status line, by design, so "loading"
 * has nowhere to go on it and does not try.
 */

/* Show it, centred. NULL if the bitmap is missing - a splash that cannot
 * load is not a reason to refuse to start. */
HWND lz_gui_splash_show(HINSTANCE inst);

/* Take it down, but not before it has been up long enough to be seen.
 * On the target machine window creation alone takes longer than this;
 * on anything modern the splash would otherwise be a single frame of
 * flicker, which reads as a glitch rather than as a splash. */
void lz_gui_splash_close(HWND h);

/* Bitmap size, for tests and for the caller that wants to know whether
 * the resource is there at all. Returns 0 when it is not. */
int lz_gui_splash_size(HINSTANCE inst, int *w, int *h);

#endif
