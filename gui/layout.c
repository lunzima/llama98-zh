/* Control geometry. Pure arithmetic - see layout.h for why it is here
 * and not in the window procedure.
 *
 * The numbers are Win9x metrics: a standard button is 23 pixels tall
 * (14 dialog units at MS Sans Serif 8pt) and the 4-pixel margin is what
 * the standard dialog spacing works out to. The design specifies the
 * ARRANGEMENT (menu bar, tool strip, transcript, input, status bar)
 * and no pixel counts, so these are chosen here. The tests assert the
 * invariants, not the numbers, so changing one of these constants is
 * meant to be cheap.
 */
#include "layout.h"

#define MARGIN     4
/* 46, measured off the rendered window rather than derived or guessed.
 *
 * The wrong candidates each fail differently:
 *   50 - seven pixels of dead band under the buttons, read as a gap.
 *   43 - what TB_GETMAXSIZE reports. The control UNDER-reports, and at
 *        43 the bar and the conversation collide.
 *   45 - flush against the button row's last painted pixel. Wrong for a
 *        reason the arithmetic could not see: the toolbar keeps a
 *        margin ABOVE its buttons too, so a flush bottom puts a gap at
 *        the top and none at the bottom.
 *
 * The measurement (window captured, ink located per row, divided back
 * out of the 1.333 display scale): client top at 70.0, button ink from
 * 72 to 128, so a 2-pixel top margin and a 42.75-logical button row.
 * Matching the bottom margin to the top one gives 45.75, and 46 is the
 * whole pixel above it. Both gaps are then the same, which is what
 * "sitting on its content" actually looks like.
 *
 * lz_gui_toolbar_needed_height() adds the two pixels the control leaves
 * out of its own answer, so the selftest compares like with like; the
 * one pixel between that and this value is the symmetry margin. */
/* TOOLBAR_CTL_H is 51, in three named pieces rather than one number,
 * because they answer to three different things.
 *
 *   46  the button row, measured; see the paragraphs above.
 *   +2  the divider the bar draws at its own TOP (Word 95 has one
 *       right under the menu).
 *   ---  TOOLBAR_CTL_H: what the CONTROL gets.
 *   +LZ_GUI_DOCK_GROOVE  the groove that closes the dock at the bottom,
 *       painted by the parent because a child clips to its own rect.
 *   +1  one pixel of COLOR_BTNFACE between that groove and the
 *       transcript. Chosen by eye, and it
 *       is the term to change if the gap wants adjusting again -- the
 *       other three are measurements and this one is the taste. */
#define TOOLBAR_CTL_H (46 + 2)
#define TOOLBAR_H     (TOOLBAR_CTL_H + LZ_GUI_DOCK_GROOVE + 1)
#define INPUT_H   52          /* about three lines; Ctrl+Enter inserts one */
/* 160, not 128: the inference-inspector design requires "widen the
 * sidebar to 160px", not negotiable in this file. The extra 32px is
 * what makes the 8-lamp-wide grid (LZ_GUI_ELAMP_GRID_W, 133px) fit
 * inside SIDE_W - 2*MARGIN (152px at 160) at all; 128 left only 120px
 * of content width. */
#define SIDE_W   160          /* right sidebar width */
/* The name line under the avatar - LZ_GUI_SIDE_LABEL_H is the same
   number, exposed publicly so gui/main.c's candidate-list title strip
   does not carry a second, independently guessed copy. */
#define NAME_H    LZ_GUI_SIDE_LABEL_H
/* Two lines - the model line and the status line set_status's own
 * sprintf produces ("%s\n%s") - fixed rather than filling to the
 * bottom, because more sidebar content below it needs the room.
 * NAME_H (18px) already holds one line of this same font at this same
 * width comfortably, so two of them is the cheapest honest answer
 * without a fresh GetTextExtentPoint32A measurement. Unlike NAME_H,
 * this box is not a tight fit (it can fill to the input row), so no
 * pixel measurement is being matched. */
#define SIDE_INFO_H (NAME_H * 2)
/* Gap between the lamp grid and the candidate list beneath it (user
 * report: the candidate list sat too close to the lamps). A NAMED
 * constant of its own rather than widening MARGIN - MARGIN is the
 * whole window's general internal spacing (avatar-to-name,
 * name-to-info, the transcript's own edges, the input row), and
 * enlarging it would nudge every one of those along with this one gap
 * nobody complained about. Double MARGIN: the lamp grid and the
 * candidate list are two visually distinct sections (a grid of small
 * squares next to a bordered list box), not two lines of the same
 * running text the way NAME/INFO's own MARGIN-sized gaps are - a
 * doubled gap is what reads as "a new section starts here" rather
 * than "the next line of the same block". Not independently
 * re-measured against a live render (no interactive desktop in this
 * environment - team-lead's own report already pixel-measured the
 * 4px MARGIN gap as the problem); the rect NxM geometry sweep is what
 * proves this constant is what the layout actually uses rather than
 * a hardcoded old number surviving somewhere. */
#define SIDE_CAND_GAP (MARGIN * 2)

static int imax(int a, int b) { return a > b ? a : b; }

void lz_gui_layout(int cw, int ch, int input_h, int status_h,
                   int panel_mode, LZRect *out) {
    int strip_y, row_y, side_x, side_y, side_b, top, cap, sh;
    int info_h, lamps_h, cand_y;

    if (!out) return;
    if (cw < LZ_GUI_MIN_CW) cw = LZ_GUI_MIN_CW;
    if (ch < LZ_GUI_MIN_CH) ch = LZ_GUI_MIN_CH;

    /* status_h: the caller's measurement, or LZ_GUI_STATUS_H when there
       is no control to have measured (see layout.h - comctl32 v5's
       status bar does not honour the height it is given, v6's does, and
       this file has no way to ask which one is running). */
    sh = status_h > 0 ? status_h : LZ_GUI_STATUS_H;

    /* Clamp the drag. The absolute cap keeps the transcript the point of
       the window; the relative one keeps a short window usable, and
       without it a 260-tall client with a 200-tall input box would have
       no transcript at all. */
    /* ZERO is the sentinel, and only zero. The drag produces NEGATIVE
       requests as soon as the pointer goes far enough below the box
       (want = bottom - bar_y, and bar_y passes bottom). Treating
       anything at or below zero as "use the default" would make a
       small downward drag clamp to the minimum while a further drag
       jumped the box back to its default height (measured: release at
       client y=400 gave 32, y=500 and beyond gave 52). Anything that
       is not the sentinel is a request, and requests get clamped by
       the two lines below - including negative ones. */
    if (input_h == 0) input_h = INPUT_H;
    cap = (ch - TOOLBAR_H - sh) / 2;
    if (cap > LZ_GUI_INPUT_MAX_H) cap = LZ_GUI_INPUT_MAX_H;
    if (cap < LZ_GUI_INPUT_MIN_H) cap = LZ_GUI_INPUT_MIN_H;
    if (input_h > cap) input_h = cap;
    if (input_h < LZ_GUI_INPUT_MIN_H) input_h = LZ_GUI_INPUT_MIN_H;

    /* Status bar: the full width of the client, hard against the bottom
       edge. */
    out[LZ_GUI_STATUS].x = 0;
    out[LZ_GUI_STATUS].y = ch - sh;
    out[LZ_GUI_STATUS].w = cw;
    out[LZ_GUI_STATUS].h = sh;

    /* Input row above the status bar: the box, and Send at its right
       end sharing its bottom edge. Enter sends, Ctrl+Enter inserts a
       newline (main.c) - the button is the visible way to do the same
       thing, which a multi-line box needs because Enter is ambiguous
       the moment the field can hold two lines. */
    row_y = ch - sh - MARGIN - input_h;
    out[LZ_GUI_INPUT].x = MARGIN;
    out[LZ_GUI_INPUT].y = row_y;
    out[LZ_GUI_INPUT].w = cw - SIDE_W - 2 * MARGIN
                          - LZ_GUI_BTN_W - MARGIN;
    out[LZ_GUI_INPUT].h = input_h;

    out[LZ_GUI_SEND].x = out[LZ_GUI_INPUT].x + out[LZ_GUI_INPUT].w + MARGIN;
    out[LZ_GUI_SEND].y = row_y + input_h - LZ_GUI_BTN_H;
    out[LZ_GUI_SEND].w = LZ_GUI_BTN_W;
    out[LZ_GUI_SEND].h = LZ_GUI_BTN_H;

    /* The drag band, occupying the gap between the input box and the
       transcript. Not a control: a child window there would have to be
       painted, and the thing being asked for is a drag that does not
       flicker. The window hit-tests this rectangle itself and tracks
       with an inverted bar, which is what Win9x splitters did and why
       they never flickered - nothing is laid out again until the
       button comes up. */
    out[LZ_GUI_SPLIT].x = MARGIN;
    out[LZ_GUI_SPLIT].y = row_y - LZ_GUI_SPLIT_H;
    out[LZ_GUI_SPLIT].w = cw - SIDE_W - 2 * MARGIN;
    out[LZ_GUI_SPLIT].h = LZ_GUI_SPLIT_H;

    /* Toolbar: hard against the top-left of the client area, which
       already excludes the menu bar (see layout.h). Full width, like
       every toolbar of the era - the bar's own background continues to
       the right edge even though the buttons stop. */
    strip_y = 0;
    out[LZ_GUI_TOOLBAR].x = 0;
    out[LZ_GUI_TOOLBAR].y = strip_y;
    out[LZ_GUI_TOOLBAR].w = cw;
    /* The CONTROL's height, not the band's. The groove and the pixel
       under it are the parent's; everything below uses TOOLBAR_H, so
       they cost the transcript three pixels and cost the bar none. */
    out[LZ_GUI_TOOLBAR].h = TOOLBAR_CTL_H;

    /* Right sidebar, in the shape an early instant messenger used: a
       small framed head at the top, the name directly under it, and the
       model state under that. Runs from below the toolbar down to the
       input row. */
    side_x = cw - SIDE_W;
    side_y = strip_y + TOOLBAR_H;
    side_b = row_y - MARGIN;
    out[LZ_GUI_SIDE_CHICKEN].x = side_x + (SIDE_W - LZ_GUI_AVATAR_FRAME) / 2;
    out[LZ_GUI_SIDE_CHICKEN].y = side_y;
    out[LZ_GUI_SIDE_CHICKEN].w = LZ_GUI_AVATAR_FRAME;
    out[LZ_GUI_SIDE_CHICKEN].h = LZ_GUI_AVATAR_FRAME;

    out[LZ_GUI_SIDE_NAME].x = side_x + MARGIN;
    out[LZ_GUI_SIDE_NAME].y = side_y + LZ_GUI_AVATAR_FRAME + MARGIN;
    out[LZ_GUI_SIDE_NAME].w = SIDE_W - 2 * MARGIN;
    out[LZ_GUI_SIDE_NAME].h = NAME_H;

    /* panel_mode == NONE: SIDE_INFO fills to side_b (info_h computed
       AFTER assigning y) - the sidebar keeps its existing three
       items. Otherwise it is bounded to SIDE_INFO_H, freeing the rest
       of the column for the lamp grid (if any) and the candidate list
       below it. */
    out[LZ_GUI_SIDE_INFO].x = side_x + MARGIN;
    out[LZ_GUI_SIDE_INFO].y = out[LZ_GUI_SIDE_NAME].y + NAME_H + MARGIN;
    out[LZ_GUI_SIDE_INFO].w = SIDE_W - 2 * MARGIN;
    if (panel_mode == LZ_GUI_PANEL_NONE) {
        info_h = imax(0, side_b - out[LZ_GUI_SIDE_INFO].y);
    } else {
        info_h = imax(0, side_b - out[LZ_GUI_SIDE_INFO].y);
        if (info_h > SIDE_INFO_H) info_h = SIDE_INFO_H;
    }
    out[LZ_GUI_SIDE_INFO].h = info_h;

    /* Lamp grid: LZ_GUI_PANEL_LAMPS only. Centred in the sidebar's
       content width the same way the avatar is - LZ_GUI_ELAMP_GRID_W
       (133px) is narrower than SIDE_W - 2*MARGIN (152px), so it does
       not just fill the column the way NAME/INFO's boxes do.
       Zero-sized in both other modes - PLAIN because there is no MoE
       to show, NONE because there is no model at all - which is this
       file's existing convention for "not present" (imax(0, ...) on
       SIDE_INFO already relies on the same reading elsewhere). The
       gate here asserts that gui/main.c creates NO lamp controls at
       all in that case; this rect being zero-sized is what a caller
       would consult to decide that, if it needed to (gui/main.c's own
       decision instead comes straight from config.num_experts, which
       is the more direct source - see create_children's own comment). */
    out[LZ_GUI_SIDE_LAMPS].y = out[LZ_GUI_SIDE_INFO].y + info_h + MARGIN;
    if (panel_mode == LZ_GUI_PANEL_LAMPS) {
        out[LZ_GUI_SIDE_LAMPS].x = side_x + (SIDE_W - LZ_GUI_ELAMP_GRID_W) / 2;
        out[LZ_GUI_SIDE_LAMPS].w = LZ_GUI_ELAMP_GRID_W;
        lamps_h = LZ_GUI_ELAMP_GRID_H;
    } else {
        out[LZ_GUI_SIDE_LAMPS].x = side_x + MARGIN;
        out[LZ_GUI_SIDE_LAMPS].w = 0;
        lamps_h = 0;
    }
    out[LZ_GUI_SIDE_LAMPS].h = lamps_h;

    /* Candidate list: fills whatever the lamp grid (if present) left,
       down to side_b - "the candidate list moves up to fill the
       space" when there is no lamp grid to sit under. Zero-sized in
       NONE, same convention as above - "the whole panel does not
       display". */
    cand_y = lamps_h > 0
             ? out[LZ_GUI_SIDE_LAMPS].y + lamps_h + SIDE_CAND_GAP
             : out[LZ_GUI_SIDE_LAMPS].y;
    out[LZ_GUI_SIDE_CAND].x = side_x + MARGIN;
    out[LZ_GUI_SIDE_CAND].y = cand_y;
    if (panel_mode == LZ_GUI_PANEL_NONE) {
        out[LZ_GUI_SIDE_CAND].w = 0;
        out[LZ_GUI_SIDE_CAND].h = 0;
    } else {
        out[LZ_GUI_SIDE_CAND].w = SIDE_W - 2 * MARGIN;
        out[LZ_GUI_SIDE_CAND].h = imax(0, side_b - cand_y);
    }

    /* Transcript: directly under the toolbar, left of the sidebar.
       No gap above it. A Win9x tool bar sits ON the content - the bar
       draws its own bottom edge and the client area starts there - and
       a four-pixel margin here would read as a gap somebody forgot to
       close. */
    top = strip_y + TOOLBAR_H;
    out[LZ_GUI_TRANSCRIPT].x = MARGIN;
    out[LZ_GUI_TRANSCRIPT].y = top;
    out[LZ_GUI_TRANSCRIPT].w = cw - SIDE_W - 2 * MARGIN;
    out[LZ_GUI_TRANSCRIPT].h = imax(0, out[LZ_GUI_SPLIT].y - top);
}

const char *lz_gui_part_name(LZGuiPart p) {
    static const char *const names[] = {
        "transcript", "input", "send", "split", "toolbar",
        "side_chicken", "side_name", "side_info",
        "side_lamps", "side_cand", "status"
    };
    if (p < 0 || p >= LZ_GUI_PART_COUNT) return "?";
    return names[p];
}

void lz_gui_center_rect(int ow_x, int ow_y, int ow_w, int ow_h,
                        int w, int h, int scr_w, int scr_h,
                        int *out_x, int *out_y) {
    int x, y;

    if (ow_w <= 0 || ow_h <= 0) {       /* no owner: centre on the screen */
        ow_x = 0; ow_y = 0;
        ow_w = scr_w; ow_h = scr_h;
    }
    x = ow_x + (ow_w - w) / 2;
    y = ow_y + (ow_h - h) / 2;

    /* Clamp the far edge first, then the near one. Doing it in the other
       order loses on a dialog wider than the screen: the right-edge
       clamp would then push the title bar off the left, where there is
       nothing to grab. */
    if (x + w > scr_w) x = scr_w - w;
    if (y + h > scr_h) y = scr_h - h;
    if (x < 0) x = 0;
    if (y < 0) y = 0;

    if (out_x) *out_x = x;
    if (out_y) *out_y = y;
}
