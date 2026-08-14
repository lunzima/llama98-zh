#ifndef LZ_GUI_LAYOUT_H
#define LZ_GUI_LAYOUT_H

/* Where every control goes, as arithmetic rather than as window code.
 *
 * Split out from the window procedure on purpose. Layout is the part
 * most likely to be wrong - a control 3 pixels tall, two controls on top
 * of each other, a status bar that walks off the bottom when the window
 * is resized - and it is also the only part of a GUI that can be checked
 * without a display. Keeping it a pure function of the client size means
 * it can be swept across hundreds of window sizes and assert
 * the invariants that actually matter, none of which need an HWND.
 *
 * No windows.h here, deliberately: this header must compile anywhere,
 * including a build with no Win32 at all.
 *
 * ARRANGEMENT: menu bar (window menu, not a control) at the top, a
 * ToolbarWindow32 below it, transcript filling the middle, the input box
 * above the status bar, and a standard status bar along the bottom. The
 * right sidebar carries a framed avatar, a name under it, and the model
 * state under that.
 *
 * The toolbar is a ToolbarWindow32 with 24x24 icons, not a row of
 * standard text buttons: Win9x put commands on a toolbar, and a strip
 * of text buttons under the menu bar is a DIALOG's shape, not an
 * application's.
 *
 * WHAT IS ON THE TOOLBAR, AND WHY SEND IS NOT. The bar carries every
 * menu command except Exit and About - open, save, clear, stop,
 * settings - which is the rule a Win9x toolbar followed: it is a
 * shortcut row for the menu, not a place for a control that belongs to
 * a specific field.
 *
 * Send belongs to the input box. It is the verb for the text sitting in
 * that box and nothing else, and the box is multi-line, so the button
 * goes BESIDE it, on the same row, sharing its bottom edge. Not
 * floating over the text the way a modern client does - a Win9x button
 * occupies its own rectangle and the field shrinks to make room.
 */

typedef struct { int x, y, w, h; } LZRect;

typedef enum {
    LZ_GUI_TRANSCRIPT = 0,   /* RichEdit 2.0, the conversation */
    LZ_GUI_INPUT,            /* multi-line edit, what the user types */
    LZ_GUI_SEND,             /* push button beside the input box */
    LZ_GUI_SPLIT,            /* the drag band; NOT a control, see below */
    LZ_GUI_TOOLBAR,          /* ToolbarWindow32, 24x24 icons + labels */
    LZ_GUI_SIDE_CHICKEN,     /* right sidebar: framed avatar */
    LZ_GUI_SIDE_NAME,        /* right sidebar: the name under it */
    LZ_GUI_SIDE_INFO,        /* right sidebar: model / KV / state */
    /* Inference inspector. Both are bounding boxes, not individual
     * controls: LZ_GUI_SIDE_LAMPS is the whole
     * 8x2 expert-lamp grid's footprint (gui/main.c places the 16
     * individual STATIC children inside it), and LZ_GUI_SIDE_CAND is
     * the candidate LISTBOX. Either can come back zero-sized - see
     * lz_gui_layout's own `panel_mode` parameter - and a zero-sized
     * rect is this file's existing convention for "not present"
     * (LZ_GUI_SIDE_INFO already does this via imax(0, ...) when the
     * window is short). */
    LZ_GUI_SIDE_LAMPS,
    LZ_GUI_SIDE_CAND,
    LZ_GUI_STATUS,           /* standard status bar, bottom */
    LZ_GUI_PART_COUNT
} LZGuiPart;

/* lz_gui_layout's panel_mode argument - what the right sidebar shows
 * below the model/status line:
 *   NONE  - no model loaded. LZ_GUI_SIDE_INFO fills all the way down
 *           to the input row - the three-item sidebar
 *           ("sidebar keeps its existing three items").
 *   PLAIN - a model is loaded but has no MoE (num_experts == 0, which
 *           is also what an unloaded model's zeroed config reads as -
 *           the two cases need no separate flag). LZ_GUI_SIDE_INFO is
 *           bounded, LZ_GUI_SIDE_LAMPS is zero-sized, and
 *           LZ_GUI_SIDE_CAND fills the space the lamps would have
 *           used - "the candidate list moves up to fill the space".
 *   LAMPS - a model with MoE is loaded. LZ_GUI_SIDE_LAMPS gets the
 *           8x2 grid's footprint and LZ_GUI_SIDE_CAND starts below it.
 * Which of PLAIN/LAMPS applies is a fact about the loaded model
 * (config.num_experts), not about generation being in progress - the
 * panel's SHAPE is decided at load time, its CONTENTS arrive later,
 * frame by frame, over WM_APP_INSPECT. */
#define LZ_GUI_PANEL_NONE  0
#define LZ_GUI_PANEL_PLAIN 1
#define LZ_GUI_PANEL_LAMPS 2

/* The 16 expert lamps: 8 columns x 2 rows, sharing the 14px bitmap and
 * 3px gap the status bar's own twin lamps use (gui/main.c's LZ_LAMP_PX
 * / LZ_LAMP_GAP) - one appearance for "a lamp" in this front end, not
 * two independently chosen ones. Given its own name rather than reusing
 * LZ_LAMP_PX directly: those constants are private to gui/main.c
 * (status-bar lamps are its own concern), while this grid's footprint
 * has to be known here too, to reserve room for it. Exposed as the
 * grid's total pixel size, not just the per-lamp size, because that is
 * the number lz_gui_layout actually needs. */
/* The CELL, not the lamp: assets/lamp-*.bmp are 14x14 and each cell
   carries a one-pixel WS_EX_STATICEDGE on every side, which is
   NONCLIENT - so a 14 cell leaves a 12x12 client and a STATIC clips the
   bitmap to it. That is not a lamp drawn small inside a frame, it is a
   lamp with its right two columns and bottom two rows cut off, and at
   10x magnification the disc is visibly sliced flat on two sides.
   16 = 14 + 1 + 1. The grid still fits: 8*16 + 7*3 = 149 against the
   152 a 160-wide sidebar leaves after its margins. */
#define LZ_GUI_ELAMP_PX    16
#define LZ_GUI_ELAMP_GAP    3
#define LZ_GUI_ELAMP_COLS   8
#define LZ_GUI_ELAMP_ROWS   2
#define LZ_GUI_ELAMP_GRID_W (LZ_GUI_ELAMP_COLS * LZ_GUI_ELAMP_PX + \
                             (LZ_GUI_ELAMP_COLS - 1) * LZ_GUI_ELAMP_GAP)
#define LZ_GUI_ELAMP_GRID_H (LZ_GUI_ELAMP_ROWS * LZ_GUI_ELAMP_PX + \
                             (LZ_GUI_ELAMP_ROWS - 1) * LZ_GUI_ELAMP_GAP)

/* The avatar, and the static that frames it.
 *
 * 48 and not 112. An early instant messenger put a SMALL framed head in
 * the corner of the chat window - OICQ and MSN Messenger were both in
 * the 32-to-48 range - and a 112-pixel photograph occupying an eighth of
 * the window is a poster, which is a shape that era did not have. The
 * frame is the other half: an avatar of the period was always inside a
 * sunken border, never floating on the dialog background.
 *
 * FRAME is what the control is created with; the sunken edge eats two
 * pixels on each side, so the client is exactly AVATAR and the bitmap
 * lands with no scaling and no SS_CENTERIMAGE (which is 4.0-only). */
#define LZ_GUI_AVATAR 48
#define LZ_GUI_AVATAR_FRAME (LZ_GUI_AVATAR + 4)

/* One line of sidebar label text - what LZ_GUI_SIDE_NAME's own height
 * (gui/layout.c's private NAME_H, sourced from this) already measures
 * for this exact font and width. Exposed here because gui/main.c
 * needs the same number for the candidate list's title strip, which
 * layout.c does not carve out as its own LZGuiPart (see
 * LZ_GUI_SIDE_CAND's own comment - one rect, two purposes). */
#define LZ_GUI_SIDE_LABEL_H 18

/* Dialog metrics, in the units the Windows 95 UI guide gives them in.
 *
 * A dialog unit is defined against the dialog font: 4 horizontal DLU is
 * one average character width and 8 vertical DLU is one character
 * height. For MS Sans Serif 8pt - the 9x dialog font, and what
 * DEFAULT_GUI_FONT gives - that average is 6x13 pixels, so one
 * horizontal DLU is 1.5px and one vertical DLU is 1.625px.
 *
 * Spelled as the DLU arithmetic rather than as the pixel answers,
 * because the pixel answers are what somebody edits when a label looks
 * tight. The settings dialog had 84x24 buttons 12px apart - numbers with
 * no provenance, and 12% wider and 4% taller than anything Windows drew.
 * The guide's values come out as the ones every Win9x dialog used:
 *
 *     button        50 x 14 DLU  ->  75 x 23 px
 *     button gap     4 DLU       ->   6 px
 *     dialog margin  7 DLU       ->  11 px
 */
#define LZ_DLU_X(n) (((n) * 3 + 1) / 2)
#define LZ_DLU_Y(n) (((n) * 13 + 4) / 8)

#define LZ_GUI_BTN_W      LZ_DLU_X(50)   /* 75 */
#define LZ_GUI_BTN_H      LZ_DLU_Y(14)   /* 23 */
#define LZ_GUI_BTN_GAP    LZ_DLU_X(4)    /*  6 */
#define LZ_GUI_DLG_MARGIN LZ_DLU_X(7)    /* 11 */

/* A single-line text box: 12 DLU, which is 20 pixels, which is exactly
 * what the contents need - 13 for the line, two for the control's own
 * padding, two more each side for the sunken edge. Any extra height
 * pools UNDER the text, because a single-line edit draws at the top of
 * its client area instead of centring - that is what makes a taller box
 * look inflated: not the border, the empty strip below the digits.
 * Measured off the rendered pixels, not chosen. */
#define LZ_GUI_EDIT_H     LZ_DLU_Y(12)   /* 20 */

/* Smallest client area the layout is defined for. The window enforces
 * the matching window size through WM_GETMINMAXINFO - a layout minimum
 * the window does not enforce is a layout that produces rectangles
 * outside the client area the first time somebody drags a corner. */
#define LZ_GUI_MIN_CW 360
#define LZ_GUI_MIN_CH 260

/* The input box is resizable, between these. The splitter band sits
 * in the plain gap, so making the box draggable costs no extra pixels.
 *
 * A MAXIMUM because the transcript is the point of the window: without
 * one, a drag to the top leaves a one-line conversation above a
 * half-screen text box, and nothing but another drag gets it back.
 * MAX is also clamped against the client height inside lz_gui_layout,
 * so a short window cannot be dragged into a negative transcript. */
/* EXACTLY THREE LINES of the target's UI font, plus the sunken edge.
 *
 * The line pitch is measured, not assumed. SimSun 9pt is what
 * DEFAULT_GUI_FONT resolves to on Simplified Chinese Win9x, and at 96
 * DPI GDI reports tmHeight 12 with tmExternalLeading 2 - so a multi-line
 * edit advances 14 pixels per line, and three lines of text need 42.
 * WS_EX_CLIENTEDGE takes two pixels top and bottom, hence 46.
 * (build/gate/fontmetric.c is the probe; MS Sans Serif 8pt comes out at
 * 13 + 0, which is why guessing from the Latin font would have been two
 * pixels short per line.) */
#define LZ_GUI_INPUT_LINE   14                      /* SimSun 9pt, measured */
#define LZ_GUI_INPUT_MIN_H  (LZ_GUI_INPUT_LINE * 3 + 4)   /* 46 */
#define LZ_GUI_INPUT_MAX_H 200
#define LZ_GUI_SPLIT_H       4

/* The etched groove that closes the tool bar dock at the BOTTOM, in
 * pixels. Word 95 has one there as well as the one under the menu, so
 * the band is two pixels taller than the control and the parent paints
 * the difference. The control cannot: it clips to its own rectangle.
 *
 * Measured on the real Word 95 window, x=700, dock bottom:
 *     y=107 160,160,160   COLOR_3DSHADOW
 *     y=108 105,105,105   COLOR_3DDKSHADOW
 *     y=109 255,255,255   COLOR_3DHILIGHT
 *
 * out[LZ_GUI_TOOLBAR] is the CONTROL's rectangle and is this much
 * shorter than the band; everything below uses the band height, so
 * changing this moves the transcript and the sidebar with it. */
#define LZ_GUI_DOCK_GROOVE   2

/* The status bar's default/fallback height, in pixels. Exposed (not a
 * private constant in layout.c) because a caller with no real control
 * to measure - a fresh window before its first resize, a test sweep
 * with no HWND at all - needs a number, and this is that number: the
 * height comctl32 v6 (themed - what a manifest build always gets on a
 * modern host) actually gives a status bar built with today's UI font.
 * It is NOT what comctl32 v5 (unthemed - Windows 98's own comctl32,
 * and what a manifest-free review build gets on ANY host, themed or
 * not) gives the same control: v5's status bar ignores the height it
 * is asked to be and substitutes its own font-derived one, 20 rather
 * than this file's 22, while keeping the bottom edge where it was
 * asked to be (measured on the kunkun98-noman build: asked for
 * y=438/h=22 - bottom 460 - got back y=440/h=20, bottom still 460).
 * lz_gui_layout's status_h parameter exists so a caller that CAN
 * measure the real control passes that instead of trusting this
 * number for a build the number does not describe. */
#define LZ_GUI_STATUS_H 22

/* Fill out[LZ_GUI_PART_COUNT]. A client size below the minimum is
 * treated as the minimum rather than producing degenerate rectangles;
 * the window will not deliver one, and clamping keeps every caller from
 * having to decide what a 12-pixel-tall transcript means.
 *
 * cw/ch are the CLIENT size as GetClientRect reports it, which on a
 * window with a menu bar already excludes the menu bar - Windows
 * shrinks the client area when SetMenu installs one, so the layout
 * needs no menu-bar term of its own (measured: a 480-tall window laid
 * out with an explicit SM_CYMENU offset landed every bottom-anchored
 * control 20 px too low).
 *
 * input_h is the height the user has dragged the input box to; pass 0
 * for the default. It is clamped here rather than by the caller, so
 * every route into the layout gets the same answer for the same drag.
 *
 * status_h is the status bar's REAL height, measured by the caller
 * (GetWindowRect after creating the control - see gui/main.c's
 * create_children); pass 0 for LZ_GUI_STATUS_H, the only answer
 * available with no control to measure. This is deliberately NOT
 * queried inside this function: layout.c is a pure function of
 * integers on purpose (no windows.h, so it can be swept with no display
 * at all), and a control's real height is a
 * Win32 fact this file is not allowed to go find out for itself. The
 * bug this parameter fixes stays invisible because the ONLY build ever
 * gated (the shipped, manifest-carrying one, comctl32 v6) is also the
 * one where LZ_GUI_STATUS_H happens to already be correct; comctl32 v5,
 * what Windows 98 itself runs and what a manifest-free review build
 * gets on ANY host, is exercised by --selftest only because the caller
 * can pass the measured height.
 *
 * panel_mode is one of the LZ_GUI_PANEL_* constants above.
 * Decided by the CALLER from whether a model is loaded and whether its
 * config has any experts - this file stays a pure function of integers,
 * so it takes the already-decided mode rather than a model pointer it
 * has no business dereferencing. */
void lz_gui_layout(int cw, int ch, int input_h, int status_h,
                   int panel_mode, LZRect *out);

/* Part name for test output. English: it is a debug string. */
const char *lz_gui_part_name(LZGuiPart p);

/* Centre a w x h popup on its owner, kept inside a scr_w x scr_h screen.
 *
 * Here rather than in settingsdlg.c because it is arithmetic, and this
 * is the file whose arithmetic gets swept without a display.
 *
 * It exists because the settings dialog passed CW_USEDEFAULT for x and
 * y. That is documented to be honoured only for overlapped windows; for
 * a WS_POPUP the system substitutes zero, so the dialog opened at the
 * top-left corner of the SCREEN every time, regardless of where its
 * owner was (measured: popup at 0,0 while the owner sat at 60,60).
 *
 * An owner of zero size means "no owner" and centres on the screen.
 * The clamp is what keeps a dialog reachable when the owner is half
 * off-screen, which on a 640x480 target is not an edge case. */
void lz_gui_center_rect(int ow_x, int ow_y, int ow_w, int ow_h,
                        int w, int h, int scr_w, int scr_h,
                        int *out_x, int *out_y);

#endif
