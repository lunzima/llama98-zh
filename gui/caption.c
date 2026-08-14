#include "caption.h"
#include <string.h>

/* Append to out, never overrunning; return the total bytes the full
   result WOULD need (snprintf semantics). */
static int app(char *out, int cap, int used, const char *s)
{
    int n = (int)strlen(s), i;
    for (i = 0; i < n; i++) {
        int at = used + i;
        if (cap > 0 && at < cap - 1) out[at] = s[i];
    }
    return used + n;
}

int lz_caption_compose(const LZCaption *c, char *out, int cap)
{
    int used = 0;

    if (cap > 0) out[0] = '\0';
    used = app(out, cap, used, c->brand);
    /* A separator follows the segment it leads: no segment, no separator. */
    if (c->chat[0]) {
        used = app(out, cap, used, " - ");
        used = app(out, cap, used, c->chat);
    }
    if (c->model[0]) {
        used = app(out, cap, used, " [");
        used = app(out, cap, used, c->model);
        used = app(out, cap, used, "]");
    }
    if (cap > 0) out[(used < cap - 1) ? used : cap - 1] = '\0';
    return used;
}

int lz_caption_plan(unsigned msg, int can_paint, int visible, int depth,
                    LZCapStep *out, int cap)
{
    LZCapStep tmp[4];
    int n = 0, i;

    if (msg == 0xAEu || msg == 0xAFu) {
        /* WM_NCUAHDRAWCAPTION / WM_NCUAHDRAWFRAME. Undocumented, and the
           reason a self-drawn caption works on an unthemed build and comes
           and goes on a themed one: with visual styles on (kunkun98 ships a
           Common-Controls 6.0.0.0 manifest, gui/kunkun98.manifest), uxtheme
           hooks DefWindowProc and sends these two to make the window redraw
           the PLATFORM's caption - outside WM_NCPAINT, so none of the
           sequence above ever sees it. They fire on activation changes and
           on caption-button hover, which is what "intermittent" looks like.

           NEVER forwarded: forwarding them IS the defect. Returning zero
           steps tells the caller to answer 0 and not call the original
           proc. Eating them is safe even when we cannot paint - the only
           thing lost is a caption we would have drawn over anyway. They do
           not exist before XP, so on the Win9x target this arm is dead. */
        if (can_paint && visible && depth == 0) tmp[n++] = LZ_CAP_PAINT;
    } else if (!can_paint || !visible || depth > 0) {
        tmp[n++] = LZ_CAP_FORWARD;
    } else if (msg == 0x86u || msg == 0x0Cu) {  /* WM_NCACTIVATE, WM_SETTEXT */
        /* Clear the bit on these two only. Per MSO95: its case 0x85 jumps
           straight to the restore-only path, never clearing; the clear
           lives on the WM_SETTEXT (0x50601349) and WM_NCACTIVATE
           (LAB_50601419) paths to suppress the extra caption repaint those
           two trigger.

           WM_SETTEXT clears the bit too: we are fed the three segments
           and never parse the window text - a true statement about the
           OTHER thing MSO95 does on that message, which is not why it
           clears the bit. MEASURED on the real window
           (_capdemo/probe_live.py, screen readback, not PrintWindow):
           SetWindowTextA repaints the platform's own caption over ours
           and it stays there. lz_caption_set calls exactly that. */
        tmp[n++] = LZ_CAP_CLEAR_VISIBLE;
        tmp[n++] = LZ_CAP_FORWARD;
        tmp[n++] = LZ_CAP_RESTORE_VISIBLE;
        tmp[n++] = LZ_CAP_PAINT;
    } else {                                /* WM_NCPAINT */
        tmp[n++] = LZ_CAP_FORWARD;
        tmp[n++] = LZ_CAP_PAINT;
    }

    if (cap < n) return -1;
    for (i = 0; i < n; i++) out[i] = tmp[i];
    return n;
}

int lz_caption_frame_dirty(unsigned swp_flags)
{
    /* SWP_FRAMECHANGED 0x0020, SWP_SHOWWINDOW 0x0040. Spelled numerically
       for the same reason as the message ids above. */
    return (swp_flags & (0x0020u | 0x0040u)) ? 1 : 0;
}

int lz_caption_bands(int cx, int hi)
{
    int n;
    if (cx <= 100) return 0;        /* FUN_5060167f's threshold */
    n = hi ? ((cx & ~3) >> 2) : (cx >> 3);
    return (n < 8) ? 8 : n;
}

int lz_caption_band_w(int i, int nsteps, int cy, int hi)
{
    int base, flat, ramp, t;

    if (nsteps < 1) return 0;
    base = ((cy * 2) & ~4) >> 2;
    flat = base + 10;
    /* flat is an absolute band count; under 4px bands the width halves, so
       flat doubles to preserve the same pixel length. Fitted, not read
       from the code, but it predicts the whole curve. */
    if (hi) flat *= 2;
    ramp = (base + 18 <= nsteps) ? nsteps - flat : 8;
    if (ramp < 1) ramp = 1;

    if (i < flat) return 0;
    t = i - flat;
    if (t > ramp) t = ramp;
    return 255 * t / ramp;
}

int lz_caption_scheme_on(int hi, int active)
{
    return (hi || active) ? 1 : 0;      /* DAT_506d04bc != 0 || param_3 != 0 */
}

int lz_caption_invert(int hi, int active, unsigned long cap)
{
    int r = (int)((cap >> 16) & 0xFF);
    int g = (int)((cap >> 8) & 0xFF);
    int b = (int)(cap & 0xFF);

    /* (DAT_506d04bc == 0) && (param_3 != 0) - without this, an inactive
       title bar at high colour depth flipped whole-bar white while the
       real one stayed near-black. */
    if (hi || !active) return 0;
    if (cap == 0xFF0000UL || cap == 0xFF00FFUL || cap == 0x0000FFUL)
        return 0;
    return (g == 0xFF || r > 0xEF || cap == 0xC0C0C0UL ||
            r + g + b > 600) ? 1 : 0;
}

/* GBK double-byte characters have a lead byte in 0x81..0xFE and a trail
   byte in 0x40..0xFE. Only a forward scan can tell whether byte n is a
   character boundary - guessing backwards from the middle is fooled by a
   trail byte that happens to land in the lead range. */
static int gbk_floor(const char *s, int want)
{
    int i = 0;
    while (i < want && s[i]) {
        unsigned char b = (unsigned char)s[i];
        int step = (b >= 0x81 && b <= 0xFE && s[i + 1]) ? 2 : 1;
        if (i + step > want) break;
        i += step;
    }
    return i;
}

LZCapFit lz_caption_fit(const LZCaption *c, const LZCapWidths *w, int avail)
{
    LZCapFit f;
    int rest;

    f.chat_bytes = 0;
    f.chat_ellipsis = 0;
    f.show_model = 0;

    rest = avail - w->brand;
    if (rest <= 0) return f;                 /* brand only */

    /* The model name is dropped last: reserve its space first. */
    if (c->model[0] && rest >= w->sep_model + w->model) {
        f.show_model = 1;
        rest -= w->sep_model + w->model;
    }

    if (!c->chat[0]) return f;

    if (rest >= w->sep_chat + w->chat) {     /* chat fits whole */
        f.chat_bytes = (int)strlen(c->chat);
        return f;
    }

    rest -= w->sep_chat + w->ellipsis;
    if (rest <= 0) return f;                 /* not even "..." fits: drop all */

    {
        int n = (int)strlen(c->chat);
        /* Estimate bytes proportionally to width, then back off to a GBK
           character boundary. The width is measured over the whole string,
           so this is an estimate; gbk_floor gives the real guarantee: never
           cut mid-character. */
        int want = (w->chat > 0) ? (int)((long)n * rest / w->chat) : 0;
        if (want > n) want = n;
        f.chat_bytes = gbk_floor(c->chat, want);
        f.chat_ellipsis = (f.chat_bytes < n);
        if (f.chat_bytes == 0) f.chat_ellipsis = 0;
    }
    return f;
}
