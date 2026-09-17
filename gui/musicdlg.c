/* The sound window. See musicdlg.h. */
#include "musicdlg.h"

#include <stdio.h>
#include <string.h>

#include "compat40.h"
#include "layout.h"
#include "localized_strings.h"
#include "midiplay.h"

#define DLG_CLASS "Kunkun98Sound"

/* ES_NUMBER, spelled out for the same reason gui/settingsdlg.c spells
 * it: the constant is hidden below the API floor, and the box it goes
 * on holds digits and a point. NOT used on the baseline box - a point
 * is not a digit and ES_NUMBER would refuse it. */

/* The control ids start at 3101, a block of their own above the
 * settings dialog's 3001..3024. Appended, never inserted, for the
 * reason that file's own enum gives: a selftest that addresses a
 * control by number does not FAIL when the number moves, it measures
 * a different control. */
enum {
    ID_MUS_ON = 3101,
    ID_MUS_ENG_LABEL,
    ID_MUS_ENG_SYS,
    ID_MUS_ENG_WAVE,
    ID_MUS_INST_LABEL,
    ID_MUS_INST,
    ID_MUS_SPD_LABEL,
    ID_MUS_SPD_FOLLOW,
    ID_MUS_SPD_FIXED,
    ID_MUS_BASE_LABEL,
    ID_MUS_BASE,
    ID_MUS_VOL_LABEL,
    ID_MUS_VOL_SCROLL,
    ID_MUS_VOL,
    ID_MUS_FIN_LABEL,
    ID_MUS_FIN_CHIME,
    ID_MUS_FIN_BEEP,
    ID_MUS_FIN_NONE,
    ID_MUS_OK,
    ID_MUS_CANCEL
};

#define DLG_W   340
#define DLG_H   322
#define PAD     LZ_GUI_DLG_MARGIN
#define ROW_H     20
#define ROW_GAP    4
#define LABEL_W   84
/* The value column, matching gui/settingsdlg.c's: "1000.00" and "127"
 * are the widest things either box holds, and a wider one steals the
 * slider's room without showing anything more. */
#define VALUE_W   48
#define BTN_W   LZ_GUI_BTN_W
#define BTN_H   LZ_GUI_BTN_H
#define BTN_GAP LZ_GUI_BTN_GAP

static int g_registered;
static int g_result;

/* EVERY control goes through here, and every one is untheme'd. That is
 * not belt and braces: gui/compat40.h says outright that the settings
 * dialog needs the same call on a SCROLLBAR and a BUTTON, and a window
 * where some controls are classic and some are themed does not read as
 * a style, it reads as a mistake. Doing it in one place is what makes
 * it impossible to forget on the next control added. */
static HWND child_ex(HWND p, DWORD ex, const char *cls, const char *text,
                     DWORD style, int x, int y, int w, int hgt, int id,
                     HINSTANCE inst) {
    HWND c = CreateWindowExA(ex, cls, text, WS_CHILD | WS_VISIBLE | style,
                             x, y, w, hgt, p, (HMENU)(LONG_PTR)id, inst,
                             NULL);
    if (c) {
        SendMessage(c, WM_SETFONT, (WPARAM)lz_ui_font(),
                    MAKELPARAM(TRUE, 0));
        lz_ui_untheme(c);
    }
    return c;
}

static HWND child(HWND p, const char *cls, const char *text, DWORD style,
                  int x, int y, int w, int hgt, int id, HINSTANCE inst) {
    return child_ex(p, 0, cls, text, style, x, y, w, hgt, id, inst);
}

/* A radio. WS_GROUP on the FIRST of each set and on nothing else: it is
 * what makes BS_AUTORADIOBUTTON clear only its own siblings and what
 * makes the arrow keys move within the set rather than across every
 * radio in the window. Three sets here, so exactly three WS_GROUPs. */
static HWND radio(HWND p, int str, int x, int y, int w, int id,
                  int first, HINSTANCE inst) {
    return child(p, "BUTTON", lz_str_display((LZStr)str),
                 BS_AUTORADIOBUTTON | (first ? (WS_GROUP | WS_TABSTOP) : 0),
                 x, y, w, ROW_H, id, inst);
}

static void set_check(HWND h, int id, int on) {
    HWND c = GetDlgItem(h, id);
    if (c) SendMessage(c, BM_SETCHECK, on ? BST_CHECKED : BST_UNCHECKED, 0);
}

static int get_check(HWND h, int id) {
    HWND c = GetDlgItem(h, id);
    return c && SendMessage(c, BM_GETCHECK, 0, 0) == BST_CHECKED;
}

/* The volume box and its scrollbar, kept in step. Volume is 0..127 -
 * MIDI's own scale, so the number in the box is the number on the
 * wire and there is no curve to disagree about. */
static void vol_show(HWND h, int v) {
    char buf[16];
    HWND sb = GetDlgItem(h, ID_MUS_VOL_SCROLL);
    HWND ed = GetDlgItem(h, ID_MUS_VOL);
    if (v < 0) v = 0;
    if (v > LZ_MUSIC_VOL_MAX) v = LZ_MUSIC_VOL_MAX;
    if (sb) SetScrollPos(sb, SB_CTL, v, TRUE);
    if (ed) {
        sprintf(buf, "%d", v);
        SetWindowTextA(ed, buf);
    }
}

static int vol_read(HWND h) {
    char buf[16];
    int v = LZ_MUSIC_VOL_DEFAULT;
    HWND ed = GetDlgItem(h, ID_MUS_VOL);
    if (ed && GetWindowTextA(ed, buf, (int)sizeof buf) > 0) {
        v = 0;
        {
            const char *p = buf;
            while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
        }
    }
    if (v < 0) v = 0;
    if (v > LZ_MUSIC_VOL_MAX) v = LZ_MUSIC_VOL_MAX;
    return v;
}

static LRESULT CALLBACK dlgproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_COMMAND:
        switch (LOWORD(wp)) {
        case ID_MUS_OK:     g_result = 1; return 0;
        case ID_MUS_CANCEL: g_result = 0; return 0;
        default: break;
        }
        return 0;

    case WM_HSCROLL: {
        HWND sb = (HWND)lp;
        int pos;
        if (!sb || GetDlgCtrlID(sb) != ID_MUS_VOL_SCROLL) break;
        pos = GetScrollPos(sb, SB_CTL);
        switch (LOWORD(wp)) {
        case SB_LINELEFT:      pos -= 1; break;
        case SB_LINERIGHT:     pos += 1; break;
        case SB_PAGELEFT:      pos -= 16; break;
        case SB_PAGERIGHT:     pos += 16; break;
        case SB_THUMBTRACK:
        case SB_THUMBPOSITION: pos = (int)(short)HIWORD(wp); break;
        case SB_TOP:           pos = 0; break;
        case SB_BOTTOM:        pos = LZ_MUSIC_VOL_MAX; break;
        /* SB_ENDSCROLL is deliberately absent: it arrives after
           SB_THUMBPOSITION has already delivered the final value, so
           handling it would re-read a position that is current. The
           same reasoning gui/settingsdlg.c gives for its own rows. */
        default: return 0;
        }
        vol_show(h, pos);
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

HWND lz_gui_music_dialog_create(HWND owner, HINSTANCE inst,
                                const LZMusicSettings *set) {
    HWND h, ctl;
    RECT rc, orc;
    char buf[LZ_MUSIC_TPS_TEXT];
    int y = PAD, dx, dy, dw, dh, i;
    int dlg_w = DLG_W, label_w = LABEL_W, col2;
    const char *list;

    if (!set || !register_class(inst)) return NULL;

    h = CreateWindowExA(0, DLG_CLASS,
                        lz_str_display(LZ_STR_DLG_SOUND_TITLE),
                        WS_POPUP | WS_CAPTION, 0, 0, DLG_W, DLG_H,
                        owner, NULL, inst, NULL);
    if (!h) return NULL;

    /* MEASURED, like gui/settingsdlg.c's rows: the label column is the
       widest of its own labels in the font this window actually uses,
       not a number chosen at a desk. A hard-coded width clips the
       longest label in whichever language was not the one it was
       chosen in, and the whole row after it moves. */
    {
        static const int LABELS[] = {
            LZ_STR_DLG_ENGINE, LZ_STR_DLG_INSTRUMENT, LZ_STR_DLG_SPEED,
            LZ_STR_DLG_BASE_TPS, LZ_STR_DLG_MUS_VOLUME
        };
        HDC dc = GetDC(h);
        if (dc) {
            HGDIOBJ old = SelectObject(dc, lz_ui_font());
            SIZE sz;
            int w = 0;
            for (i = 0; i < (int)(sizeof LABELS / sizeof LABELS[0]); i++) {
                const char *s = lz_str_display((LZStr)LABELS[i]);
                if (GetTextExtentPoint32A(dc, s, (int)strlen(s), &sz) &&
                    sz.cx > w)
                    w = sz.cx;
            }
            /* The longest radio decides the WINDOW's width the same
               way: the wavetable row carries its own "no bank yet". */
            {
                const char *s = lz_str_display(LZ_STR_DLG_ENGINE_WAVE);
                if (GetTextExtentPoint32A(dc, s, (int)strlen(s), &sz)) {
                    int need = PAD + w + BTN_GAP * 2 + 20 + sz.cx + PAD;
                    if (need > dlg_w) dlg_w = need;
                }
            }
            if (w > 0) label_w = w + BTN_GAP * 2;
            SelectObject(dc, old);
            ReleaseDC(h, dc);
        }
    }
    col2 = PAD + label_w;

    /* ---- play at all ---- */
    child(h, "BUTTON", lz_str_display(LZ_STR_DLG_MUSIC_ON),
          BS_AUTOCHECKBOX | WS_TABSTOP | WS_GROUP,
          PAD, y, dlg_w - 2 * PAD, ROW_H, ID_MUS_ON, inst);
    set_check(h, ID_MUS_ON, set->on);
    y += ROW_H + ROW_GAP * 2;

    /* ---- engine ---- */
    child(h, "STATIC", lz_str_display(LZ_STR_DLG_ENGINE), 0,
          PAD, y + 3, label_w, ROW_H, ID_MUS_ENG_LABEL, inst);
    radio(h, LZ_STR_DLG_ENGINE_SYS, col2, y, dlg_w - col2 - PAD,
          ID_MUS_ENG_SYS, 1, inst);
    y += ROW_H;
    ctl = radio(h, LZ_STR_DLG_ENGINE_WAVE, col2, y, dlg_w - col2 - PAD,
                ID_MUS_ENG_WAVE, 0, inst);
    /* Greyed while there is no bank, and the LABEL says why. The
       fall-back in lz_music_engine_pick is silent, so a row that could
       be chosen and then quietly played on the other engine would
       leave the user comparing two engines that were one. */
    if (ctl && !lz_midi_wave_ok()) EnableWindow(ctl, FALSE);
    set_check(h, ID_MUS_ENG_SYS, set->engine != LZ_MUSIC_ENGINE_WAVE);
    set_check(h, ID_MUS_ENG_WAVE, set->engine == LZ_MUSIC_ENGINE_WAVE);
    y += ROW_H + ROW_GAP * 2;

    /* ---- instrument ---- */
    child(h, "STATIC", lz_str_display(LZ_STR_DLG_INSTRUMENT), 0,
          PAD, y + 3, label_w, ROW_H, ID_MUS_INST_LABEL, inst);
    /* CBS_DROPDOWNLIST, so the only values are the ones offered: the
       bank has samples for exactly these, and a typed eleventh would
       resolve to whichever region it falls back on. */
    ctl = child_ex(h, lz_ex_style(WS_EX_CLIENTEDGE), "COMBOBOX", "",
                   WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST,
                   col2, y, dlg_w - col2 - PAD, ROW_H * 8,
                   ID_MUS_INST, inst);
    /* ONE string, split with the playlist's own splitter. Row 0 is
       "as written"; rows 1.. are lz_music_instrument[] in order, which
       tests/test_gui_music.c holds against this table. */
    list = lz_str_display(LZ_STR_DLG_INST_LIST);
    for (i = 0; i <= LZ_MUSIC_INSTRUMENTS; i++) {
        char name[64];
        if (!lz_music_list_at(list, i, name, (int)sizeof name)) break;
        SendMessage(ctl, CB_ADDSTRING, 0, (LPARAM)name);
    }
    {
        int sel = 0;
        if (set->instrument != LZ_MUSIC_AS_WRITTEN) {
            int k = lz_music_instrument_index(set->instrument);
            if (k >= 0) sel = k + 1;
        }
        SendMessage(ctl, CB_SETCURSEL, (WPARAM)sel, 0);
    }
    y += ROW_H + ROW_GAP * 2;

    /* ---- tempo ---- */
    child(h, "STATIC", lz_str_display(LZ_STR_DLG_SPEED), 0,
          PAD, y + 3, label_w, ROW_H, ID_MUS_SPD_LABEL, inst);
    radio(h, LZ_STR_DLG_SPEED_FOLLOW, col2, y, dlg_w - col2 - PAD,
          ID_MUS_SPD_FOLLOW, 1, inst);
    y += ROW_H;
    radio(h, LZ_STR_DLG_SPEED_FIXED, col2, y, dlg_w - col2 - PAD,
          ID_MUS_SPD_FIXED, 0, inst);
    set_check(h, ID_MUS_SPD_FOLLOW, set->speed != LZ_MUSIC_FIXED);
    set_check(h, ID_MUS_SPD_FIXED, set->speed == LZ_MUSIC_FIXED);
    y += ROW_H + ROW_GAP;

    /* ---- baseline ---- */
    child(h, "STATIC", lz_str_display(LZ_STR_DLG_BASE_TPS), 0,
          PAD, y + 3, label_w + 60, ROW_H, ID_MUS_BASE_LABEL, inst);
    lz_music_format_tps(set->base_tps_milli, buf, (int)sizeof buf);
    /* NOT ES_NUMBER: a decimal point is not a digit, and the box that
       refuses one is the box a user cannot enter 4.2 in. The value is
       validated on OK instead, by the same reader that will parse it
       for real. */
    ctl = child_ex(h, lz_ex_style(WS_EX_CLIENTEDGE), "EDIT", buf,
                   WS_TABSTOP | ES_LEFT | ES_AUTOHSCROLL,
                   dlg_w - PAD - VALUE_W, y, VALUE_W, LZ_GUI_EDIT_H,
                   ID_MUS_BASE, inst);
    lz_edit_use_font_margins(ctl);
    y += ROW_H + ROW_GAP * 2;

    /* ---- volume ---- */
    child(h, "STATIC", lz_str_display(LZ_STR_DLG_MUS_VOLUME), 0,
          PAD, y + 3, label_w, ROW_H, ID_MUS_VOL_LABEL, inst);
    /* SCROLLBAR, not the trackbar common control: SCROLLBAR is a
       user32 class that has existed since Windows 3.0 and is certainly
       on the floor, while the trackbar came with comctl32 and 95.
       gui/settingsdlg.c makes the same choice for the same reason. */
    ctl = child(h, "SCROLLBAR", "", WS_TABSTOP | WS_GROUP | SBS_HORZ,
                col2, y, dlg_w - col2 - PAD - VALUE_W - BTN_GAP,
                LZ_GUI_EDIT_H, ID_MUS_VOL_SCROLL, inst);
    SetScrollRange(ctl, SB_CTL, 0, LZ_MUSIC_VOL_MAX, FALSE);
    ctl = child_ex(h, lz_ex_style(WS_EX_CLIENTEDGE), "EDIT", "",
                   WS_TABSTOP | ES_LEFT | ES_AUTOHSCROLL,
                   dlg_w - PAD - VALUE_W, y, VALUE_W, LZ_GUI_EDIT_H,
                   ID_MUS_VOL, inst);
    lz_edit_use_font_margins(ctl);
    vol_show(h, set->volume);
    y += ROW_H + ROW_GAP * 2;

    /* ---- what is heard when the reply lands ----
     *
     * Label on the SAME row as the first radio, like the engine and
     * tempo groups above. On its own row it read as a heading for the
     * rest of the window rather than for its own three. */
    child(h, "STATIC", lz_str_display(LZ_STR_DLG_FINISH), 0,
          PAD, y + 3, label_w, ROW_H, ID_MUS_FIN_LABEL, inst);
    radio(h, LZ_STR_DLG_FINISH_CHIME, col2, y, dlg_w - col2 - PAD,
          ID_MUS_FIN_CHIME, 1, inst);
    y += ROW_H;
    radio(h, LZ_STR_DLG_FINISH_BEEP, col2, y, dlg_w - col2 - PAD,
          ID_MUS_FIN_BEEP, 0, inst);
    y += ROW_H;
    radio(h, LZ_STR_DLG_FINISH_NONE, col2, y, dlg_w - col2 - PAD,
          ID_MUS_FIN_NONE, 0, inst);
    set_check(h, ID_MUS_FIN_CHIME, set->finish == LZ_MUSIC_FINISH_CHIME);
    set_check(h, ID_MUS_FIN_BEEP, set->finish == LZ_MUSIC_FINISH_BEEP);
    set_check(h, ID_MUS_FIN_NONE, set->finish == LZ_MUSIC_FINISH_NONE);
    y += ROW_H + ROW_GAP * 2;

    /* ---- buttons ---- */
    child(h, "BUTTON", lz_str_display(LZ_STR_BTN_OK),
          BS_DEFPUSHBUTTON | WS_TABSTOP | WS_GROUP,
          dlg_w - PAD - BTN_W * 2 - BTN_GAP, y, BTN_W, BTN_H,
          ID_MUS_OK, inst);
    child(h, "BUTTON", lz_str_display(LZ_STR_BTN_CANCEL),
          BS_PUSHBUTTON | WS_TABSTOP,
          dlg_w - PAD - BTN_W, y, BTN_W, BTN_H, ID_MUS_CANCEL, inst);
    y += BTN_H + PAD;

    /* The window is sized to WHAT WAS LAID OUT, not to a constant
       beside the layout. The two get out of step on the first row
       added - the first build of this window was four pixels short and
       the buttons fell off the bottom - and the height also depends on
       the font, which depends on the language and on the system. The
       centring happens here for the same reason: it needs the final
       size, and a window centred on a guess is off by half the
       difference. */
    rc.left = 0; rc.top = 0; rc.right = dlg_w; rc.bottom = y;
    AdjustWindowRect(&rc, WS_POPUP | WS_CAPTION, FALSE);
    dw = rc.right - rc.left;
    dh = rc.bottom - rc.top;
    orc.left = orc.top = orc.right = orc.bottom = 0;
    if (owner) GetWindowRect(owner, &orc);
    /* CW_USEDEFAULT is honoured only for overlapped windows; a WS_POPUP
       given it opens at 0,0 - gui/settingsdlg.c measured exactly that,
       which is why the arithmetic is in gui/layout.c and not here. */
    lz_gui_center_rect(orc.left, orc.top, orc.right - orc.left,
                       orc.bottom - orc.top, dw, dh,
                       GetSystemMetrics(SM_CXSCREEN),
                       GetSystemMetrics(SM_CYSCREEN), &dx, &dy);
    SetWindowPos(h, NULL, dx, dy, dw, dh, SWP_NOZORDER | SWP_NOACTIVATE);
    return h;
}

int lz_gui_music_dialog_read(HWND dlg, LZMusicSettings *out) {
    LZMusicSettings s;
    char buf[LZ_MUSIC_TPS_TEXT];
    HWND ctl;
    int base = 0, sel;

    if (!dlg || !out) return LZ_GUI_MUSIC_BAD_BASE;
    lz_music_defaults(&s);

    ctl = GetDlgItem(dlg, ID_MUS_BASE);
    if (!ctl) return LZ_GUI_MUSIC_BAD_BASE;
    GetWindowTextA(ctl, buf, (int)sizeof buf);
    /* Read BEFORE anything is written into *out. A dialog that commits
       half of itself and then reports a bad field has already changed
       the settings the user was told were rejected. */
    if (!lz_music_parse_tps(buf, &base)) return LZ_GUI_MUSIC_BAD_BASE;
    s.base_tps_milli = base;

    s.on = get_check(dlg, ID_MUS_ON);
    s.engine = get_check(dlg, ID_MUS_ENG_WAVE) ? LZ_MUSIC_ENGINE_WAVE
                                               : LZ_MUSIC_ENGINE_SYSTEM;
    s.speed = get_check(dlg, ID_MUS_SPD_FIXED) ? LZ_MUSIC_FIXED
                                               : LZ_MUSIC_FOLLOW;
    if (get_check(dlg, ID_MUS_FIN_NONE))       s.finish = LZ_MUSIC_FINISH_NONE;
    else if (get_check(dlg, ID_MUS_FIN_BEEP))  s.finish = LZ_MUSIC_FINISH_BEEP;
    else                                       s.finish = LZ_MUSIC_FINISH_CHIME;
    s.volume = vol_read(dlg);

    ctl = GetDlgItem(dlg, ID_MUS_INST);
    sel = ctl ? (int)SendMessage(ctl, CB_GETCURSEL, 0, 0) : 0;
    /* Row 0 is "as written"; rows 1.. index lz_music_instrument[]. A
       CB_ERR (-1, nothing selected) lands on as-written too, which is
       the answer that changes nothing. */
    if (sel >= 1 && sel <= LZ_MUSIC_INSTRUMENTS)
        s.instrument = (int)lz_music_instrument[sel - 1];
    else
        s.instrument = LZ_MUSIC_AS_WRITTEN;

    lz_music_clamp(&s);
    *out = s;
    return LZ_GUI_MUSIC_OK;
}

int lz_gui_music_dialog(HWND owner, HINSTANCE inst, LZMusicSettings *set) {
    HWND h;
    MSG msg;
    LZMusicSettings edited;

    if (!set) return 0;
    h = lz_gui_music_dialog_create(owner, inst, set);
    if (!h) return 0;

    g_result = -1;
    EnableWindow(owner, FALSE);
    ShowWindow(h, SW_SHOW);
    SetFocus(GetDlgItem(h, ID_MUS_ON));
    while (g_result < 0 && IsWindow(h) &&
           GetMessage(&msg, NULL, 0, 0) > 0) {
        if (!IsDialogMessage(h, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (g_result >= 0) break;
    }
    /* The enable MUST happen before the owner is activated, or the
       system hands focus to some other application - which is why it
       is here and not in WM_DESTROY. */
    EnableWindow(owner, TRUE);
    SetActiveWindow(owner);

    if (g_result == 1) {
        if (lz_gui_music_dialog_read(h, &edited) == LZ_GUI_MUSIC_OK) {
            *set = edited;
            if (IsWindow(h)) DestroyWindow(h);
            return 1;
        }
        MessageBoxA(owner, lz_str_display(LZ_STR_ERR_BAD_BASE),
                    lz_str_display(LZ_STR_ERR_TITLE),
                    MB_ICONEXCLAMATION | MB_OK);
    }
    if (IsWindow(h)) DestroyWindow(h);
    return 0;
}
