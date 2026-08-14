#ifndef LZ_GUI_CAPTION_H
#define LZ_GUI_CAPTION_H

/* Title bar: the Word 95 italic brand segment.
 *
 * The top half is pure functions, no windows.h, so it compiles Win32-free
 * and really runs on the dev box.
 * The bottom half needs Win32 and lives in gui/captionwnd.c.
 *
 * The three segments are NOT reverse-parsed from the window text. MSO95 had
 * to parse window text because it is a shared DLL with no access to the
 * host's internals; kunkun98 IS the host. The chat name comes from the
 * chatfile filename, and '-' and '[' are normal in filenames - a real one
 * looks like "<subject> - <revision> [v2].txt", in Chinese and with those
 * exact separators - so parsing would silently split wrong on such
 * input, and compose/parse share the same separator assumptions so a
 * round-trip test would always pass (the round-trip-is-blind-to-shared-
 * assumptions shape).
 */

#define LZ_CAP_BRAND 64
#define LZ_CAP_CHAT  128
#define LZ_CAP_MODEL 64

typedef struct {
    char brand[LZ_CAP_BRAND];   /* GBK display form */
    char chat [LZ_CAP_CHAT];    /* GBK display form; empty = no chat name */
    char model[LZ_CAP_MODEL];   /* GBK display form; empty = no model */
} LZCaption;

/* Compose the flat string fed to SetWindowTextA. snprintf semantics: return
   the number of bytes the full result needs, write at most cap (incl NUL).
   Format: brand + (chat ? " - " chat : "") + (model ? " [" model "]" : "").
   A separator follows the segment it leads - dropping the chat name drops
   the " - " with it, leaving no dangling separator. */
int lz_caption_compose(const LZCaption *c, char *out, int cap);

/* Truncation rule. Measuring widths needs a DC, but "which segment to drop
   and where to cut" does not, so this eats measured pixel widths and the
   tests can feed tuples.
   The two separators are measured separately: " - " and " [" + "]" differ
   in width, and one shared sep width would compute the available space
   wrong when only one segment is dropped. */
typedef struct {
    int brand, sep_chat, chat, sep_model, model, ellipsis;
} LZCapWidths;

/* chat_bytes == 0 means the whole chat name is dropped (with its " - ").
   No separate show_chat field - two fields expressing one fact invite
   contradicting values. chat_ellipsis == 1 appends "..." at the cut. */
typedef struct {
    int chat_bytes;
    int chat_ellipsis;
    int show_model;
} LZCapFit;

/* Degradation order (owner-decided): truncate the chat name with "..."
   first, then drop the model name whole, brand always kept. */
LZCapFit lz_caption_fit(const LZCaption *c, const LZCapWidths *w, int avail);

/* The (a) clear-WS_VISIBLE sequence. The Win32 half just executes this, so
   the logic really runs and is gated on the dev box even though that box
   paints nothing. */
typedef enum {
    LZ_CAP_CLEAR_VISIBLE = 1,
    LZ_CAP_FORWARD,
    LZ_CAP_RESTORE_VISIBLE,
    LZ_CAP_PAINT
} LZCapStep;

/* msg is one of 0x0C / 0x85 / 0x86 / 0xAE / 0xAF (WM_SETTEXT, WM_NCPAINT,
   WM_NCACTIVATE, WM_NCUAHDRAWCAPTION, WM_NCUAHDRAWFRAME), numeric because
   this file does not include windows.h. Fills out, returns step count;
   -1 if cap is too small. ZERO is a valid answer and means "eat it":
   the two 0xAE/0xAF messages must never reach the original proc.

   visible is the window's current WS_VISIBLE. It is an input because the
   restore step ORs the bit back in unconditionally: run that on a window
   that has not been shown yet and the style claims visible while the
   window is not, after which ShowWindow(SW_SHOW) sees nothing to change
   and the window never appears. WM_SETTEXT reaches this from WM_CREATE
   (gui/main.c's push_caption), which is exactly that moment. */
int lz_caption_plan(unsigned msg, int can_paint, int visible, int depth,
                    LZCapStep *out, int cap);

/* Whether a WM_WINDOWPOSCHANGED with these SWP_* flags means the frame was
   just redrawn by somebody else and ours has to go back on top.

   Only SWP_FRAMECHANGED (0x0020) and SWP_SHOWWINDOW (0x0040) count, and
   the exclusion is the point: a plain move or size carries neither, and
   repainting the caption on those would double every frame paint of a
   window drag - on a Pentium II that is the whole reason the WS_VISIBLE
   trick exists. SetMenu and DrawMenuBar (gui/main.c rebuilds the bar on
   every language switch, model load and MRU update) both go through
   SWP_FRAMECHANGED, and ShowWindow through SWP_SHOWWINDOW. */
int lz_caption_frame_dirty(unsigned swp_flags);

/* Gradient band i sits this far between c0 and c1, as a 0..255 weight
   (0 = pure c0, 255 = pure c1). The caller interpolates per channel - this
   file does not touch COLORREF (Win32 byte order).

   cy is the gradient rectangle HEIGHT, not the full caption height: MSO95
   passes iStack_134 = iStack_10c - tStack_118.cy, inset from the top. From
   the Word 95 measured curve, a 22-row caption corresponds to cy~16, an
   inset of 6.

   hi is the colour-depth flag (MSO95's DAT_506d04bc, >8bpp). It makes bands
   4px instead of 8px; flat is an absolute band count, so it doubles under
   hi to keep the same pixel length - measured ~145px under both in Word 95.
   The doubling is fitted, not read from the code. */
int lz_caption_band_w(int i, int nsteps, int cy, int hi);

/* Band count. cx <= 100 returns 0: that title bar is too narrow to draw a
   gradient. hi -> 4px bands, else 8px. */
int lz_caption_bands(int cx, int hi);

/* Whether the Office colour scheme applies. Not on low colour depth +
   inactive - that is a flat system-color base + COLOR_INACTIVECAPTIONTEXT. */
int lz_caption_scheme_on(int hi, int active);

/* Whether the bright-flip (white bg / black text) triggers. Only possible
   at low colour depth + active. cap is (R<<16)|(G<<8)|B, not COLORREF -
   the pure-function layer does not touch Win32 byte order. */
int lz_caption_invert(int hi, int active, unsigned long cap);

#ifdef _WINDOWS_          /* bottom half: visible only after windows.h */
int  lz_caption_attach(HWND hwnd);
void lz_caption_set(const LZCaption *c);
int  lz_caption_can_paint(void);
#endif

#endif
