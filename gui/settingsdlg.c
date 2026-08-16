/* The inference settings window. See settingsdlg.h for why it is a plain
 * popup rather than a dialog template.
 */
#include <stddef.h>          /* offsetof - see SLIDER_ROWS */
#include <stdio.h>
#include <string.h>

#include "compat40.h"
#include "gbk.h"           /* GBK <-> UTF-8 for the system box */
#include "layout.h"          /* lz_gui_center_rect */
#include "localized_strings.h"
#include "settingsdlg.h"

#define DLG_CLASS "Kunkun98Settings"

/* Digits only in an EDIT. Windows 95 added it; Watcom's winuser.h does
 * not declare it at all, which MinGW's does - without this local
 * spelling the target build would fail on the undeclared identifier
 * while the dev build compiled clean. Spelled out locally for the same
 * reason gui/toolbar.c spells out its TB_* messages: the value is
 * fixed by the ABI, the header is not.
 *
 * A style bit an older USER32 does not know is IGNORED, not rejected,
 * so on the 3.51 floor this is a plain edit box - the reader validates
 * either way, and this only saves the user from being able to type a
 * letter in the first place. */
#define LZ_ES_NUMBER 0x2000L

enum {
    ID_THINK = 3001,
    ID_TEMP_LABEL,
    ID_TEMP,
    ID_TEMP_SCROLL,
    ID_OK,
    ID_CANCEL,
    ID_RESTORE,
    /* Appended, never inserted. gui/main.c's selftest addresses these
       controls by their numeric value because it cannot include this
       file's private enum, so renumbering an existing entry silently
       re-points every one of those checks at a different control -
       and a check pointed at the wrong control does not fail, it
       measures something else. The full map, since counting an enum
       by eye is how a miscount ends up writing text into the LABEL:

           3001 THINK   3002 TEMP_LABEL  3003 TEMP    3004 TEMP_SCROLL
           3005 OK      3006 CANCEL      3007 RESTORE
           3008 CTX_LABEL    3009 CTX    3010 CTX_SCROLL
           3011 TOPP_LABEL   3012 TOPP   3013 TOPP_SCROLL
           3014 REP_LABEL    3015 REP    3016 REP_SCROLL
           3017 MAXNEW_LABEL 3018 MAXNEW                              */
    ID_CTX_LABEL,
    ID_CTX,
    ID_CTX_SCROLL,   /* 3010 - placeholder: the ctx slider is gone; kept
                        so the ids after it do not shift (main.c's selftest
                        addresses controls by number). */
    /* 3011.. - appended again, same rule. */
    ID_TOPP_LABEL,
    ID_TOPP,
    ID_TOPP_SCROLL,
    ID_REP_LABEL,
    ID_REP,
    ID_REP_SCROLL,
    ID_MAXNEW_LABEL,
    ID_MAXNEW,
    ID_SYS_LABEL,
    ID_SYS,
    /* 3021.. - appended again, same rule. Think-block dynamic
       temperature, created under the base temperature row. */
    ID_THINK_TEMP_LABEL,
    ID_THINK_TEMP,
    ID_THINK_TEMP_SCROLL
};

/* Fixed geometry: four controls that never resize. The main window's
 * layout is a function because it stretches; this one would be a
 * function with one input and one output.
 *
 * The button and margin numbers come from gui/layout.h, where they are
 * written as the Windows 95 UI guide's dialog units: a Win9x command
 * button is 75x23 with a 6-pixel gap, and the dialog margin is 11.
 * Those four values are the most-looked-at metrics in the whole
 * system, which is why a deviation from the guide reads as wrong
 * without anyone being able to say why. */
/* Six settings need the room, and the width follows the height because
   the labels do too ("Repetition penalty" and "Max new tokens" are both
   wider than "Temperature", and label_w is one column for all four rows).
   The rows run off a `y` accumulator rather than fixed offsets, so the
   next setting is one value_row call and a number here, not a re-layout.
   The custom system prompt is a MULTILINE box, so it gets three
   row-heights instead of one; it is the LAST row so its box does not
   reflow whatever is under it.
   The think-block temperature row sits under the base temperature as
   one more value_row. 406 tall plus a caption still clears a 640x480
   screen, which is the floor gui/layout.c targets. */
#define DLG_W   330
#define DLG_H   406
#define PAD     LZ_GUI_DLG_MARGIN
#define ROW_H     22
#define BTN_W   LZ_GUI_BTN_W
#define BTN_H   LZ_GUI_BTN_H
#define BTN_GAP LZ_GUI_BTN_GAP

static int g_registered;
/* Set by the window procedure when a button closes the window. The
 * modal loop reads it after the loop ends rather than tracking state
 * across two functions. */
static int g_result;
/* lz_ui_untheme's own return for each of this dialog's controls - see
 * settingsdlg.h's own comment on LZGuiSettingsUntheme for why this is a
 * static rather than an out parameter threaded through
 * lz_gui_settings_dialog_create's existing signature. */
static LZGuiSettingsUntheme g_untheme;

void lz_gui_format_temp(float t, char *out, int cap) {
    /* Wide enough for the worst %.2f can produce from a float: it never
       switches to an exponent, so FLT_MAX expands its integer part in
       full - measured at 42 bytes, 43 with a sign. Formatting into this
       local buffer and copying out is what lets the function honour
       `cap`: sprintf straight into the caller's buffer with an
       `out[cap-1] = '\0'` after it would truncate a string that has
       already overrun. Both call sites pass char[32], so a temperature
       anywhere near FLT_MAX would smash ten bytes of their stack.
       Unreachable today - lz_common_settings_set_temp rejects an
       out-of-range value before either call site is reached - but a
       function that takes a capacity and does not use it is a trap
       primed for the next caller, and the guard lives in another file
       (iron law four, note 10). */
    char tmp[64];
    int n;
    if (!out || cap <= 0) return;
    /* Two decimals: the cap is 1.0 and the engine's defaults are one
       decimal, so two is one more than anyone needs and still short
       enough to read. sprintf of a float is fine here - the x87 control
       word is only narrowed inside the engine's forward pass, on the
       worker thread (iron law six, third point). */
    sprintf(tmp, "%.2f", (double)t);
    n = (int)strlen(tmp);
    if (n > cap - 1) n = cap - 1;
    memcpy(out, tmp, (size_t)n);
    out[n] = '\0';
}

/* Temperature <-> scrollbar position, 0..100. Two decimals of a value
   capped at LZ_COMMON_TEMP_MAX map onto SBS_HORZ's own range with nothing
   left over, which is why the span is 100 and not some other number - no
   remainder to round away between one control's units and the other's.

   Both directions clamp their own argument before doing the arithmetic,
   not after: an out-of-range float fed to a 0..100 scroll position would
   otherwise carry the out-of-range-ness through the multiply, and an
   out-of-range position is exactly what a stale WM_HSCROLL after a range
   change could deliver. */
int lz_gui_temp_to_scroll(float t) {
    int pos;
    if (t < 0.0f) t = 0.0f;
    if (t > LZ_COMMON_TEMP_MAX) t = LZ_COMMON_TEMP_MAX;
    pos = (int)(t / LZ_COMMON_TEMP_MAX * 100.0f + 0.5f);
    /* NOT redundant, and the reason is worth writing down because it
       reads redundant: after the two clamps above, t is in [0, MAX], so
       the arithmetic cannot leave [0, 100] - unless t is NaN. NaN fails
       BOTH comparisons above (every comparison with NaN is false), so it
       reaches the cast, where converting it to int is undefined and in
       practice yields something like INT_MIN. These two lines are what
       stop that from becoming a scroll position.
       Unreachable today: lz_common_settings_set_temp rejects NaN via its
       !(t >= 0.0f) test, so no caller can hand one over. Kept anyway for
       the same reason lz_gui_format_temp now honours its cap - the guard
       lives in another file, and a function should not depend on its
       callers having read it. */
    if (pos < 0) pos = 0;
    if (pos > 100) pos = 100;
    return pos;
}

float lz_gui_scroll_to_temp(int pos) {
    if (pos < 0) pos = 0;
    if (pos > 100) pos = 100;
    return (float)pos / 100.0f * LZ_COMMON_TEMP_MAX;
}

/* The four slider rows, as data.
 *
 * ONE TABLE, read by four things that would otherwise each have to know
 * all four rows: the WM_HSCROLL handler (thumb -> box), the EN_KILLFOCUS
 * handlers (box -> thumb), "restore defaults" and the think toggle
 * (settings -> both). The failure this prevents is the one that costs
 * nothing to introduce and never announces itself - a fifth row wired
 * into three of the four places.
 *
 * `is_int` picks which of the two conversion pairs applies; a row is
 * either a two-decimal float or a plain integer, and there is no third
 * kind. Both directions are function pointers because the mappings are
 * genuinely different arithmetic per row (a repetition penalty has
 * eleven useful positions, a temperature a hundred) and gui/settings.c
 * is where that arithmetic is tested without a window. */
typedef struct {
    int    edit_id, scroll_id;
    int    span_lo, span_hi;
    int    page;
    int    is_int;
    /* Where this row's value lives in an LZGuiSettings. An offset
       rather than four if-branches keyed on edit_id: with the offset
       the table IS the wiring, and a row added to it is a row every
       reader below already handles. */
    size_t field_off;
    float  fmin, fmax;                   /* float rows */
    int    (*f_to_scroll)(float);
    float  (*f_from_scroll)(int);
    int    imin, imax;                   /* integer rows */
    int    (*i_to_scroll)(int);
    int    (*i_from_scroll)(int);
} LZSliderRow;

static const LZSliderRow SLIDER_ROWS[] = {
    { ID_THINK_TEMP, ID_THINK_TEMP_SCROLL, 0, 100, 10, 0,
      offsetof(LZGuiSettings, think_temp),
      0.0f, LZ_COMMON_TEMP_MAX, lz_gui_temp_to_scroll, lz_gui_scroll_to_temp,
      0, 0, NULL, NULL },
    { ID_TEMP, ID_TEMP_SCROLL, 0, 100, 10, 0,
      offsetof(LZGuiSettings, temp),
      0.0f, LZ_COMMON_TEMP_MAX, lz_gui_temp_to_scroll, lz_gui_scroll_to_temp,
      0, 0, NULL, NULL },
    { ID_TOPP, ID_TOPP_SCROLL,
      LZ_COMMON_TOPP_SCROLL_MIN, LZ_COMMON_TOPP_SCROLL_MAX, 10, 0,
      offsetof(LZGuiSettings, topp),
      LZ_COMMON_TOPP_MIN, LZ_COMMON_TOPP_MAX,
      lz_common_topp_to_scroll, lz_common_scroll_to_topp,
      0, 0, NULL, NULL },
    { ID_REP, ID_REP_SCROLL, 0, LZ_COMMON_REP_SCROLL_MAX, 2, 0,
      offsetof(LZGuiSettings, rep),
      LZ_COMMON_REP_MIN, LZ_COMMON_REP_MAX,
      lz_common_rep_to_scroll, lz_common_scroll_to_rep,
      0, 0, NULL, NULL }
    /* ctx is not in the table: it is a plain box like max_new - see
       lz_gui_settings_dialog_create. A 512..32768 range would be 64
       slider positions, a coarse picker whose thumb is meaningless; the
       box is the whole control, and the two "length" settings (context
       + max new tokens) are a plain-box pair. */
};
#define SLIDER_ROW_COUNT ((int)(sizeof SLIDER_ROWS / sizeof SLIDER_ROWS[0]))

static int row_int(const LZGuiSettings *s, const LZSliderRow *r) {
    return *(const int *)(const void *)((const char *)s + r->field_off);
}
static float row_float(const LZGuiSettings *s, const LZSliderRow *r) {
    return *(const float *)(const void *)((const char *)s + r->field_off);
}

/* Write the row's value into its box, in whichever of the two forms it
   takes. `pos` is a scroll position, so this is the thumb -> box
   direction. */
static void row_show_scroll(HWND h, const LZSliderRow *r, int pos) {
    char buf[32];
    if (r->is_int) sprintf(buf, "%d", r->i_from_scroll(pos));
    else lz_gui_format_temp(r->f_from_scroll(pos), buf, (int)sizeof buf);
    SetWindowTextA(GetDlgItem(h, r->edit_id), buf);
}

/* Box -> thumb. A value the box itself would refuse leaves the
   scrollbar where it was; OK still does the real validation and can
   refuse the whole dialog. */
static void row_sync_from_edit(HWND h, const LZSliderRow *r) {
    HWND e = GetDlgItem(h, r->edit_id);
    HWND sb = GetDlgItem(h, r->scroll_id);
    char buf[64];
    buf[0] = '\0';
    if (!e || !sb) return;
    GetWindowTextA(e, buf, (int)sizeof buf);
    if (r->is_int) {
        int v;
        if (sscanf(buf, "%d", &v) == 1 && v >= r->imin && v <= r->imax)
            SetScrollPos(sb, SB_CTL, r->i_to_scroll(v), TRUE);
    } else {
        float v;
        if (sscanf(buf, "%f", &v) == 1 && v >= r->fmin && v <= r->fmax)
            SetScrollPos(sb, SB_CTL, r->f_to_scroll(v), TRUE);
    }
}

/* The system prompt into its box.
 *
 * The box is an ANSI control and speaks the DISPLAY code page;
 * LZGuiSettings.system is UTF-8. Writing the UTF-8 bytes straight in
 * makes the reader hand back the trailing byte of each character -
 * mojibake in the box and mojibake into the settings on OK.
 *
 * ONE function, because the box is written from two places - once at
 * creation and again by dlg_show, which "restore defaults" and the
 * think toggle both go through. Two copies of the conversion is how one
 * of them ends up not converting. */
static void sys_set_box(HWND box, const char *utf8)
{
    static char gbk[2 * LZ_COMMON_SYSTEM_MAX + 8];
    const char *shown = "";
    if (!box) return;
    if (utf8 && utf8[0]) {
        int n = lz_gbk_from_utf8(utf8, (int)strlen(utf8), gbk,
                                 (int)sizeof gbk, NULL);
        if (n > 0 && n < (int)sizeof gbk) shown = gbk;
    }
    SetWindowTextA(box, shown);
}

/* Every control, from a settings struct. The one place that knows how
   to paint the whole dialog, so "restore defaults" and the think toggle
   cannot each forget a different row. */
static void dlg_show(HWND h, const LZGuiSettings *s) {
    char buf[32];
    HWND c;
    int i;

    for (i = 0; i < SLIDER_ROW_COUNT; i++) {
        const LZSliderRow *r = &SLIDER_ROWS[i];
        int pos;
        if (r->is_int) {
            int v = row_int(s, r);
            sprintf(buf, "%d", v);
            pos = r->i_to_scroll(v);
        } else {
            float v = row_float(s, r);
            lz_gui_format_temp(v, buf, (int)sizeof buf);
            pos = r->f_to_scroll(v);
        }
        c = GetDlgItem(h, r->edit_id);
        if (c) SetWindowTextA(c, buf);
        c = GetDlgItem(h, r->scroll_id);
        if (c) SetScrollPos(c, SB_CTL, pos, TRUE);
    }
    /* The rows with no slider, hence not in the table. */
    sprintf(buf, "%d", s->max_new);
    c = GetDlgItem(h, ID_MAXNEW);
    if (c) SetWindowTextA(c, buf);
    /* The system prompt: the box carries the USER's text, and EMPTY is
       the state that means "use the built-in identity". Restore sets it
       empty, which is the box being cleared rather than showing the
       constant - settings.h says why the constant is never copied into
       the settings. */
    sys_set_box(GetDlgItem(h, ID_SYS), s->system);
    c = GetDlgItem(h, ID_THINK);
    if (c) SendMessage(c, BM_SETCHECK,
                       s->think ? BST_CHECKED : BST_UNCHECKED, 0);
}

static LRESULT CALLBACK dlgproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        /* Neither of these destroys the window. The caller has to READ
           the controls after the loop ends, and reading a window that
           the handler already tore down gets whatever GetWindowText
           returns for a dead HWND - which is "" and no error. */
        case ID_OK:
            g_result = 1;
            return 0;
        case IDCANCEL:            /* Esc, via IsDialogMessage */
        case ID_CANCEL:
            g_result = 0;
            return 0;
        case ID_RESTORE: {
            /* Restore writes the CONTROLS, not the caller's settings:
               nothing is committed until OK. A restore that took effect
               immediately would make Cancel a lie. */
            LZGuiSettings s;
            /* Fully initialised, not three fields of it - see
               lz_gui_settings_dialog_read's own note on what an
               uninitialised LZGuiSettings cost. */
            lz_common_settings_init(&s);
            s.think = (int)SendMessage(GetDlgItem(h, ID_THINK), BM_GETCHECK,
                                       0, 0) == BST_CHECKED;
            s.manual_temp = 0;
            s.manual_topp = 0;
            lz_common_settings_restore(&s);
            /* Every control, from the struct - not row by row here.
               The row-by-row version is how the context row gets added
               to the dialog and forgotten by Restore. */
            dlg_show(h, &s);
            return 0;
        }
        case ID_THINK: {
            /* Mirror the coupling live so the window shows what OK would
               do. Same call the command layer makes, so the two cannot
               disagree about the rule. */
            LZGuiSettings s, base;
            /* This handler has no access to the caller's struct and
               does not need one - lz_common_settings_set_think below
               rewrites everything it then displays. Spelled out because
               the `cur` parameter exists precisely to stop the next
               person passing something uninitialised.

               THE MANUAL FLAGS ARE THE EXCEPTION and they are why this
               is not simply dlg_show(h, &s): the reader cannot recover
               them from the controls (a box showing 0.80 looks the same
               whether the user typed it or the preset put it there), so
               they come from `base`, where init left them clear. That
               makes the live mirror show what a NEVER-TOUCHED value
               would do - which is right for the common case and
               deliberately conservative for the other: a user who typed
               a temperature and then toggles think sees it move here,
               and OK then commits what they see. */
            lz_common_settings_init(&base);
            if (lz_gui_settings_dialog_read(h, &base, &s) != 0) return 0;
            lz_common_settings_set_think(&s,
                (int)SendMessage(GetDlgItem(h, ID_THINK), BM_GETCHECK, 0, 0)
                == BST_CHECKED);
            dlg_show(h, &s);
            return 0;
        }
        default:
            /* Box -> thumb, for whichever slider row this is (if any).
               The other direction is WM_HSCROLL below.

               EN_KILLFOCUS rather than EN_CHANGE: EN_CHANGE fires on
               every keystroke, including the ones in the middle of
               typing "0.75" where the text is momentarily "0." or
               "0.7" - neither is a number sscanf can read, and moving
               the thumb on a rejected half-typed value would make the
               scrollbar flicker back to wherever it last was on every
               other key. Kill-focus is also when a Win9x user's
               attention has already left the box, so the thumb catching
               up then reads as the box confirming what they typed
               rather than fighting them while they type it.

               The thumb SNAPS to the nearest position while the box
               keeps the exact number typed - deliberate, and the reason
               the slider is called the coarse picker: 3000 is a
               perfectly good context, the slider simply has no position
               for it. */
            if (HIWORD(wp) == EN_KILLFOCUS) {
                int i;
                for (i = 0; i < SLIDER_ROW_COUNT; i++)
                    if (SLIDER_ROWS[i].edit_id == (int)LOWORD(wp)) {
                        row_sync_from_edit(h, &SLIDER_ROWS[i]);
                        break;
                    }
            }
            break;
        }
        break;

    case WM_HSCROLL: {
        /* Scrollbar -> edit, the other direction from ID_TEMP above.
           (HWND)lp is the control's own handle per WM_HSCROLL's contract
           when it is sent by a scrollbar control (as opposed to a window's
           own non-client scrollbars, where lp is 0) - checking it against
           the ones this dialog owns is what keeps this from reacting to
           some other control it might grow later.

           FOUR SCROLLBARS, ONE HANDLER, driven off SLIDER_ROWS: the
           span differs per row (a temperature has a hundred useful
           positions, a repetition penalty eleven) and so does the page
           step, which on an eleven-position bar cannot stay at the
           temperature row's 10 or one page would cross the whole
           thing. */
        HWND sb = (HWND)lp;
        const LZSliderRow *r = NULL;
        int i;
        for (i = 0; sb && i < SLIDER_ROW_COUNT; i++)
            if (sb == GetDlgItem(h, SLIDER_ROWS[i].scroll_id)) {
                r = &SLIDER_ROWS[i];
                break;
            }
        if (r) {
            int pos = GetScrollPos(sb, SB_CTL);
            switch (LOWORD(wp)) {
            case SB_LINELEFT:      pos -= 1;       break;
            case SB_LINERIGHT:     pos += 1;       break;
            case SB_PAGELEFT:      pos -= r->page; break;
            case SB_PAGERIGHT:     pos += r->page; break;
            /* HIWORD(wp) is the thumb's position DURING the drag; it is
               only valid for these two notifications, which is why it is
               not read for the line/page cases above. */
            case SB_THUMBTRACK:
            case SB_THUMBPOSITION:
                pos = (int)(short)HIWORD(wp);
                break;
            /* Home and End. A Win9x scrollbar answers them, so this one
               has to too - a scrollbar that does not is the kind of gap
               nobody files a bug about: the key simply does nothing and
               the user stops pressing it. SB_ENDSCROLL is deliberately
               NOT here: it arrives when a drag finishes, after
               SB_THUMBPOSITION has already delivered the final value, so
               handling it would re-read a position that is already
               current. */
            case SB_TOP:           pos = r->span_lo; break;
            case SB_BOTTOM:        pos = r->span_hi; break;
            default:
                return 0;
            }
            if (pos < r->span_lo) pos = r->span_lo;
            if (pos > r->span_hi) pos = r->span_hi;
            SetScrollPos(sb, SB_CTL, pos, TRUE);
            row_show_scroll(h, r, pos);
        }
        return 0;
    }

    case WM_CLOSE:
        g_result = 0;
        return 0;

    default:
        break;
    }
    return DefWindowProcA(h, msg, wp, lp);
}

static int register_class(HINSTANCE inst) {
    WNDCLASSA wc;
    if (g_registered) return 1;
    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc = dlgproc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = DLG_CLASS;
    if (!RegisterClassA(&wc)) return 0;
    g_registered = 1;
    return 1;
}

static HWND child_ex(HWND p, DWORD ex, const char *cls, const char *text,
                     DWORD style, int x, int y, int w, int hgt, int id,
                     HINSTANCE inst) {
    HWND c = CreateWindowExA(ex, cls, text, WS_CHILD | WS_VISIBLE | style,
                             x, y, w, hgt, p, (HMENU)(LONG_PTR)id, inst,
                             NULL);
    if (c) SendMessage(c, WM_SETFONT, (WPARAM)lz_ui_font(),
                       MAKELPARAM(TRUE, 0));
    return c;
}

static HWND child(HWND p, const char *cls, const char *text, DWORD style,
                  int x, int y, int w, int hgt, int id, HINSTANCE inst) {
    return child_ex(p, 0, cls, text, style, x, y, w, hgt, id, inst);
}

/* One value row: label, coarse slider, value box.
 *
 * FOUR settings have exactly this shape (temperature, top_p, the
 * repetition penalty, the context window) and a hand-written copy per
 * row is where they start disagreeing - one forgets lz_ui_untheme,
 * another creates the box before the slider and quietly reverses Tab
 * across that row, a third measures its own field width and breaks the
 * column. All four rows are calls to this one function so they cannot
 * diverge.
 *
 * SCROLLBAR (not msctls_trackbar32/the Trackbar common control):
 * SCROLLBAR is a user32 window class that has existed since Windows
 * 3.0 and is certainly present on the NT 3.51 floor; the Trackbar is a
 * comctl32 control introduced with Windows 95 and not guaranteed
 * there. "Use the older thing when the older thing still does the job."
 *
 * Slider first, box second, ALWAYS: IsDialogMessage's Tab chain follows
 * Z-order, which is creation order for sibling WS_TABSTOP children, and
 * Z-order does NOT follow the x-coordinates the controls end up painted
 * at. That is the whole reason this is one function.
 *
 * `edit_style` carries the per-row extras (LZ_ES_NUMBER for the ones
 * that only ever hold digits); everything else about the box is fixed
 * here so it cannot vary by row. */
static void value_row(HWND h, HINSTANCE inst, int y,
                      int label_id, int label_str, int label_w,
                      int sb_id, int sb_lo, int sb_hi, int sb_pos,
                      int edit_id, const char *edit_text, int edit_w,
                      DWORD edit_style,
                      int *out_sb_untheme, int *out_edit_untheme) {
    HWND ctl;
    int box_x = DLG_W - PAD - edit_w;
    int sb_x = PAD + label_w;
    int sb_w = box_x - BTN_GAP - sb_x;

    child(h, "STATIC", lz_str_display((LZStr)label_str), 0,
          PAD, y + 3, label_w, ROW_H, label_id, inst);

    /* Defensive floor, not a case this host's own fonts reach
       (measured: sb_w comes out well over 100 here) - only a guard
       against a future label so wide in some language that the two
       measured widths would otherwise collide. */
    if (sb_w < 20) sb_w = 20;

    if (sb_id) {
        ctl = child(h, "SCROLLBAR", "", WS_TABSTOP | SBS_HORZ,
                    sb_x, y, sb_w, LZ_GUI_EDIT_H, sb_id, inst);
        SetScrollRange(ctl, SB_CTL, sb_lo, sb_hi, FALSE);
        SetScrollPos(ctl, SB_CTL, sb_pos, TRUE);
        if (out_sb_untheme) *out_sb_untheme = lz_ui_untheme(ctl);
    }

    /* WS_EX_CLIENTEDGE, not WS_BORDER: a 4.0 text box is a sunken 3D
       well and WS_BORDER draws a flat line. lz_ex_style drops the bit
       on 3.51, where the plain border is what that system drew.
       lz_edit_use_font_margins because an edit made by CreateWindow has
       margins of zero while one made from a dialog template asks for
       EC_USEFONTINFO - that is why edits look right in programs that
       use a template and cramped in the ones that do not. */
    ctl = child_ex(h, lz_ex_style(WS_EX_CLIENTEDGE), "EDIT", edit_text,
                   WS_TABSTOP | ES_LEFT | ES_AUTOHSCROLL | edit_style,
                   box_x, y, edit_w, LZ_GUI_EDIT_H, edit_id, inst);
    lz_edit_use_font_margins(ctl);
    if (out_edit_untheme) *out_edit_untheme = lz_ui_untheme(ctl);
}

HWND lz_gui_settings_dialog_create(HWND owner, HINSTANCE inst,
                                   const LZGuiSettings *set) {
    HWND h, ctl;
    RECT rc, orc;
    char buf[32];
    int y = PAD, bx, dx, dy, dw, dh, label_w, edit_w;

    if (!set || !register_class(inst)) return NULL;

    rc.left = 0; rc.top = 0; rc.right = DLG_W; rc.bottom = DLG_H;
    AdjustWindowRect(&rc, WS_CAPTION, FALSE);
    dw = rc.right - rc.left;
    dh = rc.bottom - rc.top;

    /* Centred on the owner, not CW_USEDEFAULT. That constant is honoured
       only for overlapped windows; for a WS_POPUP the system substitutes
       zero, and this dialog opened at the screen's top-left corner every
       time - measured at 0,0 with the owner at 60,60. The arithmetic is
       in gui/layout.c so it can be swept without a display. */
    orc.left = orc.top = orc.right = orc.bottom = 0;
    if (owner) GetWindowRect(owner, &orc);
    lz_gui_center_rect(orc.left, orc.top, orc.right - orc.left,
                       orc.bottom - orc.top, dw, dh,
                       GetSystemMetrics(SM_CXSCREEN),
                       GetSystemMetrics(SM_CYSCREEN), &dx, &dy);

    h = CreateWindowExA(0, DLG_CLASS,
                        lz_str_display(LZ_STR_DLG_SETTINGS_TITLE),
                        WS_POPUP | WS_CAPTION,
                        dx, dy, dw, dh,
                        owner, NULL, inst, NULL);
    if (!h) return NULL;
    memset(&g_untheme, 0, sizeof g_untheme);

    /* Temperature row geometry, MEASURED rather than guessed: the
       label and the value box each get exactly the width their own
       text needs in the REAL font this dialog actually uses -
       lz_ui_font(), the identical call child_ex's own WM_SETFONT makes
       below, not a font this measurement invents. "0.00" stands in for
       the value box's content: LZ_COMMON_TEMP_MAX is 1.0, so
       lz_gui_format_temp's "%.2f" never produces more than one digit,
       a dot, and two digits - four characters, always, whatever the
       value - and every real font gives its ten digit glyphs the same
       advance width for exactly this reason (so numbers line up in a
       column), so "0.00" is not a guess at the widest case, it IS the
       only case. Whatever space is left over goes to the scrollbar -
       the whole point, since a label needing less room in the LIVE
       language (this dialog is rebuilt fresh per open, so it is
       already showing whichever language is current) gives that room
       back. */
    {
        /* ONE label width and ONE field width for ALL the rows, each
           the widest of its own measurements. The rows are the same
           shape - label, slider, value box - so anything that differs
           between them by a few pixels reads as a mistake rather than
           as a measurement, and it puts the sliders' right edges at
           different x, which is the one thing a Win9x property sheet
           never does. (Not hypothetical: a context row that measures its
           own field width comes out four pixels wider than the
           temperature row.)

           The value samples are "0.00" and "8888" - both exactly four
           characters, but a digit is wider than a period in every
           proportional face, and every real font gives its ten digits
           the same advance so that numbers line up in a column. So
           "8888" is not a guess at the widest case, it IS the widest
           case for a four-character number. */
        static const int LABELS[] = {
            LZ_STR_DLG_THINK_TEMP, LZ_STR_DLG_TEMPERATURE, LZ_STR_DLG_TOPP,
            LZ_STR_DLG_REP, LZ_STR_DLG_CONTEXT, LZ_STR_DLG_MAXNEW
        };
        HDC dc = GetDC(h);
        SIZE sz;
        TEXTMETRICA tm;
        int pad = 6, lw = 0, vw = 0, i;

        if (dc) {
            HGDIOBJ old = SelectObject(dc, lz_ui_font());
            if (GetTextMetricsA(dc, &tm) && tm.tmAveCharWidth > 0)
                pad = tm.tmAveCharWidth;
            for (i = 0; i < (int)(sizeof LABELS / sizeof LABELS[0]); i++) {
                const char *lbl = lz_str_display((LZStr)LABELS[i]);
                if (GetTextExtentPoint32A(dc, lbl, (int)strlen(lbl), &sz)
                    && sz.cx > lw)
                    lw = sz.cx;
            }
            if (GetTextExtentPoint32A(dc, "0.00", 4, &sz) && sz.cx > vw)
                vw = sz.cx;
            if (GetTextExtentPoint32A(dc, "8888", 4, &sz) && sz.cx > vw)
                vw = sz.cx;
            SelectObject(dc, old);
            ReleaseDC(h, dc);
        }
        /* Fallback if GetDC or the measurements somehow fail - should
           not happen on any real host, but this file compiles at the
           3.51 floor and two lines of defence cost nothing: a failure
           degrades to a shorter slider rather than to a zero- or
           negative-width control CreateWindow would silently accept. */
        if (lw <= 0) lw = 99;
        if (vw <= 0) vw = LZ_DLU_X(40) - 12;
        label_w = lw + pad;
        edit_w = vw + pad;
    }

    child(h, "BUTTON", lz_str_display(LZ_STR_DLG_THINK),
          BS_AUTOCHECKBOX | WS_TABSTOP, PAD, y, DLG_W - 2 * PAD, ROW_H,
          ID_THINK, inst);
    ctl = GetDlgItem(h, ID_THINK);
    SendMessage(ctl, BM_SETCHECK,
                set->think ? BST_CHECKED : BST_UNCHECKED, 0);
    /* Every control in this window is untheme'd, not only the edit. A
       scrollbar drawn XP-flat next to buttons and a checkbox drawn
       Win9x-classic is still half-and-half, just a different half, so
       the whole dialog matches the main window's standard rather than
       a one-off fix for one pair of controls. See gui/compat40.h's
       lz_ui_untheme for why the function is named for any class. */
    g_untheme.think = lz_ui_untheme(ctl);
    y += ROW_H + PAD;

    /* The value rows, in visual order - which is also creation
       order, which is also Tab order, which is why they are all calls
       to one function and not five blocks of similar code. */

    /* Base temperature comes directly under the Think box, THEN the
       think-block temperature - the user's own ordering: the general
       knob above the sub-case it governs. Same shape, same [0,
       LZ_COMMON_TEMP_MAX] slider mapping on both. */
    lz_gui_format_temp(set->temp, buf, (int)sizeof buf);
    value_row(h, inst, y, ID_TEMP_LABEL, LZ_STR_DLG_TEMPERATURE, label_w,
              ID_TEMP_SCROLL, 0, 100, lz_gui_temp_to_scroll(set->temp),
              ID_TEMP, buf, edit_w, 0,
              &g_untheme.temp_scroll, &g_untheme.temp);
    y += ROW_H + 4;

    /* Think-block temperature, under the base temperature.
       The row has no "off" because the GUI always enables the feature
       (the value box IS the whole control surface). */
    lz_gui_format_temp(set->think_temp, buf, (int)sizeof buf);
    value_row(h, inst, y, ID_THINK_TEMP_LABEL, LZ_STR_DLG_THINK_TEMP,
              label_w,
              ID_THINK_TEMP_SCROLL, 0, 100,
              lz_gui_temp_to_scroll(set->think_temp),
              ID_THINK_TEMP, buf, edit_w, 0,
              &g_untheme.think_temp_scroll, &g_untheme.think_temp);
    y += ROW_H + 4;

    child(h, "STATIC", lz_str_display(LZ_STR_DLG_TEMP_CAP), 0,
          PAD, y, DLG_W - 2 * PAD, ROW_H, 0, inst);
    y += ROW_H + 6;

    lz_gui_format_temp(set->topp, buf, (int)sizeof buf);
    value_row(h, inst, y, ID_TOPP_LABEL, LZ_STR_DLG_TOPP, label_w,
              ID_TOPP_SCROLL, LZ_COMMON_TOPP_SCROLL_MIN, LZ_COMMON_TOPP_SCROLL_MAX,
              lz_common_topp_to_scroll(set->topp),
              ID_TOPP, buf, edit_w, 0,
              &g_untheme.topp_scroll, &g_untheme.topp);
    y += ROW_H + 4;

    lz_gui_format_temp(set->rep, buf, (int)sizeof buf);
    value_row(h, inst, y, ID_REP_LABEL, LZ_STR_DLG_REP, label_w,
              ID_REP_SCROLL, 0, LZ_COMMON_REP_SCROLL_MAX,
              lz_common_rep_to_scroll(set->rep),
              ID_REP, buf, edit_w, 0,
              &g_untheme.rep_scroll, &g_untheme.rep);
    /* +6, not the +4 between slider rows: rep is the LAST sampling row
       and ctx is the FIRST length row, so this is a category boundary
       (the same gap a note leaves before its next row). With +4 the
       context box read as part of the sampling block, and its top gap
       (4) disagreed with its own note's bottom gap (6). */
    y += ROW_H + 6;

    /* ---- context window ----

       A plain box, NO slider: the range is 512..32768, which is 64
       stepped positions - a coarse picker whose thumb is meaningless -
       and the box accepts any integer in range anyway. The context row
       is the SAME shape as the max-new-tokens row directly under it -
       the two "length" settings, one category, two plain boxes. */
    sprintf(buf, "%d", set->ctx);
    value_row(h, inst, y, ID_CTX_LABEL, LZ_STR_DLG_CONTEXT, label_w,
              0, 0, 0, 0,
              ID_CTX, buf, edit_w, LZ_ES_NUMBER,
              NULL, &g_untheme.ctx);
    /* No note under the context box: the two "length" rows - context
       + max new tokens - are one category, and a note between them
       would push the max-new row a full line away from its partner.
       The context range is already named by the refusal message
       (LZ_STR_ERR_BAD_CTX) when a value is out of range, so a note
       would carry nothing the dialog does not say on demand. The enum
       member LZ_STR_DLG_CTX_NOTE stays in localized_strings.h so the
       ids after it do not shift. */
    y += ROW_H + 4;

    /* ---- maximum new tokens ----

       NO SLIDER, and no LZ_ES_NUMBER either. Both absences are the same
       fact: this box has to be able to hold -1.

       LZ_ES_NUMBER admits digits and nothing else - not even a minus
       sign - so a box carrying it could never be typed into. And a
       slider would have to span "unlimited" plus 16..ctx, a range with
       a hole in it at one end; a thumb position that means something
       categorically different from its neighbours is worse than no
       thumb.

       SHOWN AS -1, NOT AS THE WORD "unlimited", and this is
       deliberate. A box that displays a localised word is a box whose
       reader has to parse that word back, in whichever language, with
       whatever spacing the user leaves around it. The note underneath
       carries the meaning at no such cost. */
    sprintf(buf, "%d", set->max_new);
    value_row(h, inst, y, ID_MAXNEW_LABEL, LZ_STR_DLG_MAXNEW, label_w,
              0, 0, 0, 0,
              ID_MAXNEW, buf, edit_w, 0,
              NULL, &g_untheme.maxnew);
    y += ROW_H + 4;

    child(h, "STATIC", lz_str_display(LZ_STR_DLG_MAXNEW_NOTE), 0,
          PAD, y, DLG_W - 2 * PAD, ROW_H, 0, inst);
    y += ROW_H + 6;

    /* ---- custom system prompt ----
       A MULTILINE box (ES_MULTILINE | ES_AUTOVSCROLL). The value it
       carries is the user's custom text, EMPTY means "built-in identity"
       (engine Item 9b). The box is INITIALISED empty; the placeholder is
       not the value. */
    child(h, "STATIC", lz_str_display(LZ_STR_DLG_SYS), 0,
          PAD, y + 3, label_w, ROW_H, ID_SYS_LABEL, inst);
    {
        /* Three row-heights: one boxful of prompt is more than one line
           of label, and the dialog grew for it (see DLG_H). */
        int sh = 3 * ROW_H;
        /* GBK INTO THE BOX, UTF-8 OUT OF IT (the two-forms trap in
           localized_strings.h): a window is ANSI, so SetWindowTextA and
           GetWindowTextA talk the DISPLAY code page, while
           LZGuiSettings.system is UTF-8. sys_set_box does the one
           direction; lz_gui_settings_dialog_read does the other. */
        ctl = child_ex(h, lz_ex_style(WS_EX_CLIENTEDGE), "EDIT", "",
                       WS_TABSTOP | ES_MULTILINE | ES_AUTOVSCROLL |
                       ES_WANTRETURN,
                       PAD + label_w, y, DLG_W - PAD - (PAD + label_w),
                       sh, ID_SYS, inst);
        sys_set_box(ctl, set->system);
        lz_edit_use_font_margins(ctl);
        /* Cap what can be typed: the reader copies into
           LZGuiSettings.system (LZ_COMMON_SYSTEM_MAX+1 bytes) and is
           safe either way, but a box that refuses the excess at the
           keyboard is a box whose user learns the limit exists. */
        SendMessage(ctl, EM_LIMITTEXT, (WPARAM)LZ_COMMON_SYSTEM_MAX, 0);
        g_untheme.sys = lz_ui_untheme(ctl);
    }
    y += ROW_H + 4;

    /* No system-prompt note under the label: it rendered
       truncated/incomplete next to the label and read as a broken
       fragment. The label + empty box already say "leave it alone for
       the default"; a note earns no space it cannot draw correctly. */

    /* Cancel hard right, OK one gap to its left, and the wide one on the
       far left - the Win9x arrangement for a dialog whose third button
       is not part of the accept/dismiss pair. The gap between OK and
       Cancel is BTN_GAP (6), not the dialog margin: the two belong
       together and the margin would read as three separate buttons.

       CREATION ORDER IS TAB ORDER, and it runs left-to-right here:
       Restore (far left) -> OK (middle) -> Cancel (right), matching
       the screen. The bx positions are computed for each button
       independently of creation order, so a reorder that swaps the
       child() calls without touching bx fixes Tab without moving a
       pixel. */
    /* The restore-defaults label is six characters wide; 75 does not
       hold it, and the guide's answer for a label that does not fit is
       a wider button, not a smaller font. Widened in whole DLU so it
       stays on the grid. */
    ctl = child(h, "BUTTON", lz_str_display(LZ_STR_BTN_RESTORE_DEFAULT),
               BS_PUSHBUTTON | WS_TABSTOP, PAD, DLG_H - PAD - BTN_H,
               LZ_DLU_X(64), BTN_H, ID_RESTORE, inst);
    g_untheme.restore = lz_ui_untheme(ctl);
    bx = DLG_W - PAD - BTN_W;
    bx -= BTN_GAP + BTN_W;
    ctl = child(h, "BUTTON", lz_str_display(LZ_STR_BTN_OK),
               BS_DEFPUSHBUTTON | WS_TABSTOP, bx, DLG_H - PAD - BTN_H,
               BTN_W, BTN_H, ID_OK, inst);
    g_untheme.ok = lz_ui_untheme(ctl);
    bx = DLG_W - PAD - BTN_W;
    ctl = child(h, "BUTTON", lz_str_display(LZ_STR_BTN_CANCEL),
               BS_PUSHBUTTON | WS_TABSTOP, bx, DLG_H - PAD - BTN_H, BTN_W,
               BTN_H, ID_CANCEL, inst);
    g_untheme.cancel = lz_ui_untheme(ctl);
    return h;
}

LZGuiSettingsUntheme lz_gui_settings_last_untheme(void) {
    return g_untheme;
}

int lz_gui_settings_dialog_read(HWND dlg, const LZGuiSettings *cur,
                                LZGuiSettings *out) {
    /* Big enough for the largest box this dialog owns (the custom
       system prompt, capped at LZ_COMMON_SYSTEM_MAX by EM_LIMITTEXT) so a
       single buffer serves every field. A smaller buffer here would
       truncate a long system prompt silently. */
    char buf[LZ_COMMON_SYSTEM_MAX + 1];
    LZGuiSettings s;
    float t = 0.0f;
    HWND e;
    int sel;

    if (!dlg || !cur || !out) return 1;
    /* START FROM THE CURRENT SETTINGS. This is the reason `cur` is a
       parameter at all: `s` must start life fully initialised, because
       the fields this dialog does NOT show (seed_mode, seed) cannot be
       allowed to hold garbage. A non-zero seed_mode reads as
       LZ_COMMON_SEED_FIXED, which would pin the session to a fixed
       nonsense seed that nothing reports because it produces perfectly
       ordinary-looking replies.

       Passing the current settings IN, rather than documenting a
       precondition on `out`, is what makes this un-forgettable: the
       compiler now demands a source for the fields this dialog does
       not own, and the next field added to LZGuiSettings inherits the
       right behaviour by default instead of the wrong one. */
    s = *cur;
    s.think = (int)SendMessage(GetDlgItem(dlg, ID_THINK), BM_GETCHECK, 0, 0)
              == BST_CHECKED;
    /* Both preset-following values start from the mode's default with
       their flag clear, and the setters below raise the flag again for
       whichever ones the boxes actually carry a value for. Doing it
       this way round rather than trusting *cur's flags is what makes a
       think toggle inside the dialog behave: the box contents are the
       authority for the VALUE, and having just been rewritten by the
       toggle they are the mode default, which is exactly the state
       "not manual" describes. */
    s.temp = lz_common_settings_default_temp(s.think);
    s.manual_temp = 0;
    s.topp = lz_common_settings_default_topp(s.think);
    s.manual_topp = 0;
    s.think_temp = lz_common_settings_default_think_temp(s.think);
    s.manual_think_temp = 0;

    /* EVERY box is REFUSED, never clamped, and each names itself in the
       return code. The reasoning for the temperature generalises
       without change: a clamp turns "1.5" or "9000" into a
       silently different request, a refusal tells the user the limit
       exists. And a dialog that reports the wrong field sends the user
       to correct a value that was fine, which is why these are six
       codes and not one.

       sscanf, not atof: atof cannot tell "no number" from "zero", and
       zero is a legal temperature (greedy decoding). */
    e = GetDlgItem(dlg, ID_THINK_TEMP);
    buf[0] = '\0';
    if (e) GetWindowTextA(e, buf, (int)sizeof buf);
    if (sscanf(buf, "%f", &t) != 1) return LZ_GUI_SET_BAD_THINK_TEMP;
    if (lz_common_settings_set_think_temp(&s, t) != 0)
        return LZ_GUI_SET_BAD_THINK_TEMP;

    e = GetDlgItem(dlg, ID_TEMP);
    buf[0] = '\0';
    if (e) GetWindowTextA(e, buf, (int)sizeof buf);
    if (sscanf(buf, "%f", &t) != 1) return LZ_GUI_SET_BAD_TEMP;
    if (lz_common_settings_set_temp(&s, t) != 0) return LZ_GUI_SET_BAD_TEMP;

    e = GetDlgItem(dlg, ID_TOPP);
    buf[0] = '\0';
    if (e) GetWindowTextA(e, buf, (int)sizeof buf);
    if (sscanf(buf, "%f", &t) != 1) return LZ_GUI_SET_BAD_TOPP;
    if (lz_common_settings_set_topp(&s, t) != 0) return LZ_GUI_SET_BAD_TOPP;

    e = GetDlgItem(dlg, ID_REP);
    buf[0] = '\0';
    if (e) GetWindowTextA(e, buf, (int)sizeof buf);
    if (sscanf(buf, "%f", &t) != 1) return LZ_GUI_SET_BAD_REP;
    if (lz_common_settings_set_rep(&s, t) != 0) return LZ_GUI_SET_BAD_REP;

    e = GetDlgItem(dlg, ID_CTX);
    buf[0] = '\0';
    if (e) GetWindowTextA(e, buf, (int)sizeof buf);
    if (sscanf(buf, "%d", &sel) != 1) return LZ_GUI_SET_BAD_CTX;
    if (sel < LZ_COMMON_CTX_MIN || sel > LZ_COMMON_CTX_MAX)
        return LZ_GUI_SET_BAD_CTX;
    s.ctx = sel;

    /* Capped against the context THIS DIALOG is about to commit, not
       against the one currently in force: the two rows are read in the
       same pass, and a user who raises both at once should not be told
       their generation cap is too large for a window they just
       enlarged. s.ctx is already parsed above, which is the only
       reason this row has to come last. */
    e = GetDlgItem(dlg, ID_MAXNEW);
    buf[0] = '\0';
    if (e) GetWindowTextA(e, buf, (int)sizeof buf);
    if (sscanf(buf, "%d", &sel) != 1) return LZ_GUI_SET_BAD_MAXNEW;
    if (lz_common_settings_set_max_new(&s, sel, s.ctx) != 0)
        return LZ_GUI_SET_BAD_MAXNEW;

    /* The system prompt box. The buffer the box owns is capped by the
       EM_LIMITTEXT set at creation, so it cannot overflow the
       destination - this read is safe against a box someone pasted
       megabytes into.
       GBK OUT OF THE BOX, UTF-8 INTO s.system: the box speaks the
       display code page (see the create side's own note), the field is
       UTF-8. Without the round-trip the reader hands back mojibake. */
    e = GetDlgItem(dlg, ID_SYS);
    buf[0] = '\0';
    if (e) GetWindowTextA(e, buf, (int)sizeof buf);
    if (buf[0]) {
        /* lz_gbk_to_utf8 of pure ASCII is the identity, so no ASCII
           fallback is needed - the box's GBK is converted, and ASCII
           passes through unchanged. */
        static char sysutf8[LZ_COMMON_SYSTEM_MAX + 1];
        int n2 = lz_gbk_to_utf8(buf, (int)strlen(buf), sysutf8,
                                (int)sizeof sysutf8, NULL);
        if (n2 < 0 || n2 >= (int)sizeof sysutf8) {
            s.system[0] = '\0';
        } else {
            size_t n3 = (size_t)n2;
            if (n3 > LZ_COMMON_SYSTEM_MAX) n3 = LZ_COMMON_SYSTEM_MAX;
            memcpy(s.system, sysutf8, n3);
            s.system[n3] = '\0';
        }
    } else {
        s.system[0] = '\0';
    }

    *out = s;
    return LZ_GUI_SET_OK;
}

int lz_gui_settings_dialog(HWND owner, HINSTANCE inst, LZGuiSettings *set) {
    HWND h;
    MSG msg;
    LZGuiSettings edited;

    if (!set) return 0;
    h = lz_gui_settings_dialog_create(owner, inst, set);
    if (!h) return 0;

    /* Modal by hand: disable the owner, pump until the window is gone,
       enable it again. The enable MUST happen before the owner is
       activated or the system hands focus to some other application -
       which is why it is here and not in WM_DESTROY. */
    g_result = -1;
    EnableWindow(owner, FALSE);
    ShowWindow(h, SW_SHOW);
    SetFocus(GetDlgItem(h, ID_TEMP));
    while (g_result < 0 && IsWindow(h) &&
           GetMessage(&msg, NULL, 0, 0) > 0) {
        /* IsDialogMessage gives tab order and Esc for free, and it is
           the reason a popup can behave like a dialog without being
           one. */
        if (!IsDialogMessage(h, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (g_result >= 0) break;
    }
    EnableWindow(owner, TRUE);
    SetActiveWindow(owner);

    if (g_result == 1) {
        int rc = lz_gui_settings_dialog_read(h, set, &edited);
        if (rc == 0) {
            *set = edited;
            if (IsWindow(h)) DestroyWindow(h);
            return 1;
        }
        /* The value was refused. Say so - naming the box that refused
           rather than defaulting to the temperature - and treat it as a
           cancel rather than committing half of it.
           A table rather than a chain of ternaries, indexed by the
           code, so adding the seventh setting is one row. Index 0 is
           LZ_GUI_SET_OK, which cannot get here; it holds the
           temperature message so that a code this table has not heard
           of degrades to a real sentence instead of reading past the
           end. */
        static const int MSG[] = {
            LZ_STR_ERR_BAD_TEMP,    /* LZ_GUI_SET_OK - unreachable */
            LZ_STR_ERR_BAD_TEMP,
            LZ_STR_ERR_BAD_CTX,
            LZ_STR_ERR_BAD_TOPP,
            LZ_STR_ERR_BAD_REP,
            LZ_STR_ERR_BAD_MAXNEW,
            LZ_STR_ERR_BAD_THINK_TEMP
        };
        int which = (rc >= 0 && rc < (int)(sizeof MSG / sizeof MSG[0]))
                    ? MSG[rc] : LZ_STR_ERR_BAD_TEMP;
        MessageBoxA(owner, lz_str_display((LZStr)which),
                    lz_str_display(LZ_STR_ERR_TITLE),
                    MB_ICONEXCLAMATION | MB_OK);
    }
    if (IsWindow(h)) DestroyWindow(h);
    return 0;
}
