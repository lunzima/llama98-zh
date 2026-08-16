/* kunkun98 - the Win32 front end. Window, controls, menu, status strip.
 *
 * No engine yet: this is the skeleton. Menu entries whose action needs a
 * loaded model are created GRAYED rather than live. A live item that
 * silently does nothing looks exactly like a bug, and empty menu items
 * are not allowed; greying is how "not wired yet" is said out loud.
 *
 * Text goes through gui/localized_strings.c in its DISPLAY form. Win9x
 * has no CP_UTF8, so an ANSI window reads whatever bytes it is handed
 * with code page 936 - handing it the UTF-8 source draws mojibake and
 * reports nothing.
 *
 * --selftest <path> builds the window, checks every control against
 * gui/layout.c and against the string table, writes a report and exits.
 * It is a mode of the SHIPPED binary rather than a separate build: a
 * console-subsystem copy would be a different executable, and the thing
 * worth checking is this one.
 */
#include <windows.h>
#include <richedit.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "chatfile.h"
#include "command.h"
#include "aboutdlg.h"       /* Help > About window */
#include "caption.h"        /* title-bar self-drawing */
#include "compat.h"     /* lz_time_ms - the throughput clock */
#include "compat40.h"
#include "cpucheck.h"
#include "err.h"
#include "inifile.h"
#include "savechat.h"
#include "session.h"
#include "settings.h"
#include "settingsdlg.h"
#include "resource.h"
#include "splash.h"
#include "toolbar.h"
#include "gbk.h"
#include "layout.h"
#include "localized_strings.h"
#include "modelload.h"
#include "mru.h"
#include "stream.h"
#include "worker.h"

#define LZ_CLASS_NAME "Kunkun98Main"
#define LZ_TRANSCRIPT_LIMIT (1024L * 1024L)

/* Think text grey, everything else back to plain black.
 * Both are set explicitly. Leaving the non-think runs on CFE_AUTOCOLOR
 * would be more polite to the system colour scheme, but it also makes
 * "did the colour get restored" unanswerable - the field reads back as
 * whatever the control feels like.
 *
 * EXPLICIT LITERALS, not GetSysColor (user report: a
 * custom desktop colour scheme could make speaker/system colours
 * unreadable). They are fixed, explicit colours regardless of scheme -
 * the user asked for the opposite of following the desktop - the
 * SAME reasoning that motivates the transcript's background also
 * being pinned to white below (EM_SETBKGNDCOLOR) rather than following
 * COLOR_WINDOW. Pinning one side and not the other would risk exactly
 * the unreadable combination (say, light grey text on a scheme with a
 * light window colour) a user picking their own palette can hit.
 *
 * The values match the STOCK Win98 scheme's own colours (000000,
 * 808080), so nothing changes there; USER/ASSISTANT/
 * SYS below are new colours, chosen dark enough to stay readable on
 * the now-fixed white background - a pure RGB(0,0,255)/RGB(0,255,0)
 * reads as neon on white and the green in particular loses contrast,
 * so these are deliberately dark (navy / dark green / maroon) rather
 * than saturated primaries.
 *
 * Known limit: the colours are baked into each run's character
 * formatting as it is appended, so text already on screen does not
 * retint when one of these constants changes; changing a colour means
 * editing source, not switching a theme. A live WM_SYSCOLORCHANGE
 * would not retint old text either. */
#define LZ_COLOR_TEXT       RGB(0x00, 0x00, 0x00)   /* body: black */
#define LZ_COLOR_THINK      RGB(0x80, 0x80, 0x80)   /* think: grey */
#define LZ_COLOR_USER       RGB(0x00, 0x00, 0x80)   /* user speaker line: navy */
#define LZ_COLOR_ASSISTANT  RGB(0x00, 0x64, 0x00)   /* assistant speaker line: dark green */
#define LZ_COLOR_SYS        RGB(0x80, 0x00, 0x00)   /* system line: maroon */
/* Behind code, inline and fenced alike. C0C0C0 and not a lighter grey
 * because it has to be a PALETTE colour: the target can be running 256
 * colours, where anything outside the default static palette is
 * dithered, and a dither pattern behind small text is worse than no
 * shading at all. It is also the Win95 shading grey - the same silver
 * as the toolbar - so it reads as furniture rather than as a
 * highlight. */
#define LZ_COLOR_CODE_BG    RGB(0xC0, 0xC0, 0xC0)

enum {
    ID_TRANSCRIPT = 1001,
    ID_INPUT,
    ID_SEND,
    /* No menu item and no toolbar button carries this id - IDM_STOP_GEN
       is what those wear, and is what cmd_enable is now called with.
       ID_STOP is kept anyway: the WM_COMMAND switch still has
       "case ID_STOP: case IDM_STOP_GEN:", a spare path for a Stop control
       of its own beside the input box, and deleting the enum would turn
       that case silently into dead code with nobody noticing. */
    ID_STOP,
    ID_SIDE_CHICKEN,
    ID_SIDE_INFO,
    ID_STATUS,
    ID_TOOLBAR,
    ID_SIDE_NAME,
    ID_LAMP0,
    ID_LAMP1,
    ID_PROGRESS,
    /* Inference inspector, part one. ID_ELAMP0 ..
       ID_ELAMP0+15 are the 16 expert lamps, contiguous and not
       individually named - the same shape ID_LAMP0/ID_LAMP1 use for
       the status bar's own pair, extended to 16 the way IDM_MRU0
       (gui/resource.h) extends to a 4-deep list: each lamp is created
       with (HMENU)(LZ_IPTR)(ID_ELAMP0 + i), and skipping straight to
       ID_ELAMP0 + 16 for the next id reserves the run without naming
       every member of it. */
    ID_ELAMP0,
    ID_CAND = ID_ELAMP0 + 16,
    ID_CAND_TITLE
};

/* Status-bar lamps. 16 pixels, two of them, in a reserved cell on the
   right - the geometry a Win9x status indicator had. */
enum { LZ_LAMP_OFF = 0, LZ_LAMP_READY, LZ_LAMP_BUSY, LZ_LAMP_ERROR,
       LZ_LAMP_KINDS };
/* 14, not 16. SS_BITMAP clips, it does not scale, so the control's size
   IS the bitmap's size - and a 16-pixel lamp centred in a 22-pixel status
   bar covered the bar's own border at the top and the bottom. The bead
   itself did not shrink; the cell's dead one-pixel margin did (see
   build/watcom/make_lamps.py). */
#define LZ_LAMP_PX     14
/* 3, up from 2: at 14 pixels the two beads' rings nearly touched and the
   pair read as one wide object. The reserved cell widens with it, since
   LZ_LAMP_CELL_W is derived rather than written down twice. */
#define LZ_LAMP_GAP     3
#define LZ_LAMP_PAD     3
#define LZ_LAMP_CELL_W (LZ_LAMP_PAD * 2 + LZ_LAMP_PX * 2 + LZ_LAMP_GAP)
/* ONE periodic tick for the whole window, and the reason is not
 * tidiness. There used to be three - a lamp timer, a token timer and a
 * debug-ramp timer - and each of them wrote to the status line with its
 * own idea of which phase the job was in. The token timer said
 * "generating" every 100 ms while the lamp timer put the prefill
 * progress back every 400 ms; the debug timer existed only because the
 * lamp timer's lifetime was job-scoped and the ramp had to run when no
 * job did. Three writers, three phase tests, one line - the disagreement
 * was the bug, not a symptom of it.
 * Now: ui_tick decides the phase once and every periodic consumer hangs
 * off it, ui_timer_sync owns the single lifetime. Sub-rates come from
 * the CLOCK, not from counting ticks, so a consumer's interval does not
 * depend on how often this happens to fire. */
#define LZ_UI_TIMER     1
/* Selftest-only pump watchdog. Its own id so it cannot collide with
   the display tick above; see st_pump. */
#define LZ_ST_PUMP_TIMER 99
#define LZ_LAMP_BLINK 400        /* ms; WinZip's activity lamp flickered */

/* The display is refreshed on a TIMER, not once per token (user
 * request). A token is a few bytes; pushing each one straight to the
 * control would cost an EM_SETCHARFORMAT + EM_SETPARAFORMAT +
 * EM_REPLACESEL and the repaint that follows - three messages and a
 * redraw to add three characters. Buffering the arrivals and emptying
 * the buffer on a tick
 * gives the same visible behaviour (text appearing as it is generated)
 * for a fraction of the work, which is the whole point on a machine
 * where the repaint can cost more than the token did.
 *
 * IT IS A KNOB, not a constant: how long a tick should be depends
 * entirely on the machine, and this project's own iron law nine says a
 * value that is arguable across the target family becomes an option
 * rather than a decision made here. `stream_ms` in kunkun98.ini.
 * ZERO restores the old push-per-token behaviour exactly, which is what
 * makes it a control rather than only a tuning value - the selftest
 * drives both settings. */
#define LZ_TOK_MS       100      /* default tick */
#define LZ_TOK_MS_MAX   2000     /* an ini typo must not freeze the view */
#define LZ_TOK_BUF      4096

/* One window, so one instance. Static rather than a GWL_USERDATA block:
   nothing here is per-window, and a heap block would need a lifetime
   story for no benefit.
 *
 * Declared here, ahead of the menu tables below, because build_menu_bar
 * reads g.mru while drawing the File popup - a menu table is DATA, not a
 * function, so there is no forward declaration for "the struct this file
 * uses everywhere" the way there is for a function. */
static struct {
    HWND main;
    HWND part[LZ_GUI_PART_COUNT];
    HINSTANCE inst;
    HMODULE riched;
    HACCEL accel;               /* Ctrl+O / Ctrl+S / F1, see kunkun98.rc */
    int  status_is_sbar;        /* status bar is comctl32's SBAR */
    /* The status bar's REAL height, measured once in create_children
     * (after apply_font, not right after creation - see that measurement
     * site's own comment for why the order matters) and fed to every
     * lz_gui_layout call from here on - comctl32 v5 (unthemed, what
     * Windows 98's own comctl32 and any manifest-free build give)
     * overrides whatever height it is asked for, v6 (themed) does not,
     * and layout.c has no way to tell which one it is talking to
     * (gui/layout.h has the measurement and why this exists). 0 until
     * create_children runs that far; lz_gui_layout treats that the same
     * as any other non-positive value, its own LZ_GUI_STATUS_H default. */
    int  status_h;
    /* The input box's dragged height, 0 for the layout's default, and
     * the band the user grabs to change it. `dragging` is the tracking
     * state: while it is set the layout is NOT recomputed, only an
     * inverted bar moves, which is the whole reason the drag does not
     * flicker. */
    int  input_h;
    HWND lamp[2];
    HWND progress;          /* prefill indicator, comctl32 only */
    int  progress_untheme;  /* did SetWindowTheme report success */
    HBITMAP lamp_bmp[LZ_LAMP_KINDS];
    int  lamp_phase;            /* activity lamp: 0 lit, 1 dark */
    /* Inference inspector, part one. elamp[16]: NULL
     * unless the currently loaded model has MoE (config.num_experts >
     * 0) - see side_panel_sync's own comment for the create/destroy
     * lifecycle. A SEPARATE set of bitmap handles from lamp_bmp[]
     * above rather than sharing them: lamp_bmp is only ever loaded
     * inside `if (g.status_is_sbar)` (the status bar's twin lamps need
     * comctl32 to have somewhere to live), while these 16 are plain
     * children of the MAIN WINDOW and have to work on the 3.51 floor
     * with no comctl32 at all - sharing would mean moving that load
     * out of its guard, a second change with its own risk, for a
     * bitmap resource cheap enough that loading it twice is not worth
     * that.
     *
     * ALL FOUR lamp kinds: team-lead's own real-model measurement
     * (200 tokens) found "how many layers picked this
     * expert" has a real, short spread - 60.0% of hits are 1 layer,
     * 31.5% are 2, 7.7% are 3, 0.8% are 4+ - so a lamp reads
     * OFF/GREEN/AMBER/RED for 0/1/2/3+ layers, the
     * Passat B2 instrument-cluster reading the user asked for. Routing
     * WEIGHT was considered and rejected for this: 79.6% of per-expert
     * maxima fall in a narrow 0.3-0.7 band, too tight to draw as
     * anything but even, uninformative brightness - see inspect.h's
     * own comment on expert_hits for the fuller account. */
    HWND elamp[16];
    HBITMAP elamp_bmp[LZ_LAMP_KINDS];
    /* The candidate list's own title strip - "Candidates (N)" or the
     * Chinese equivalent, set once at creation and again on every
     * WM_APP_INSPECT (repaint_candidates).
     * Not tracked in g.part[]/LZGuiPart: LZ_GUI_SIDE_CAND is the whole
     * area layout.c reserves, and this is a top slice of it this file
     * carves out itself, the same relationship elamp[] has to
     * LZ_GUI_SIDE_LAMPS - one rect, two purposes, only one of which
     * layout.c needs to know about by name. */
    HWND cand_title;
    int  load_failed;           /* last model load ended badly */
    LZRect split;
    int  dragging;
    int  drag_y;                /* where the inverted bar is right now */
    char side_model[400];       /* sidebar model line, set at load */
    /* Current conversation file's last segment, for the title bar. The
       project had no "current conversation file" concept before - the file
       dialogs are one-shot, the path is discarded. The caption module only
       reads it; the truth lives here. GBK display form. */
    char chat_name[LZ_CAP_CHAT];
    LZStream stream;
    /* Where the last exchange STARTS in the transcript, so it can be
     * taken back: [0] is just before the user turn's header, [1] just
     * before the assistant's (the rollback commands).
     *
     * CHARACTER POSITIONS as the control counts them, read back with
     * EM_EXGETSEL - NOT GetWindowTextLengthA, which is wrong here for
     * a reason worth keeping written down:
     * that call returns the length of the ANSI text, and this control
     * holds GBK, where a Chinese character is two bytes and one cp. A
     * transcript with any Chinese in it would truncate in the middle of
     * the wrong turn, and the selftest's own ASCII scripts would never
     * have shown it.
     *
     * -1 means "there is no exchange on screen this refers to", which
     * is the state transcript_clear leaves behind and the state a
     * conversation loaded from a file is in (its turns were drawn, but
     * not by do_send, so nothing recorded where they began). The three
     * rollback commands are greyed while either is -1 - see
     * rollback_ready. */
    LONG turn_cp[2];
    /* Tokens that have arrived but are not on screen yet, and how often
       the timer empties this into the control. See tokens_arrived. */
    char tok_buf[LZ_TOK_BUF];
    int  tok_n;
    int  tok_ms;
    /* Spec 3.1: how many tokens THIS turn's generation has produced, and
       the wall-clock time at which the job started. Accumulated in
       tokens_arrived (every emitted token reaches it once), so it is
       authoritative over the running time even when the display lags
       behind by a throttle. `tok_live` says a generate job is currently
       running - it is what lets the throttled tick keep the rate cell
       fresh while a long turn is in flight. */
    int   tok_gen;
    double tok_start_ms;
    /* When the GENERATION phase began, which is not when the job did.
       The cell is labelled "generating N tok, X tok/s", so its
       denominator has to be the span that produced those tokens: with
       the job start as the denominator, a 30 s prefill followed by a
       20-token reply in 2 s reads 0.6 tok/s for a decode running at 10.
       cli_main.c faces the same split and answers it by printing BOTH
       "s turn" and "s gen" - see its own comment on why the whole-turn
       number is the honest one for the reuse A/B. One cell cannot say
       two things, so it says the one its label claims.
       Equal to tok_start_ms until prefill ends, so a turn with no
       prefill is unchanged. */
    double gen_start_ms;
    int   tok_live;
    /* The repetition penalty's WINDOW, in tokens. Ini-only, deliberately
     * not in LZGuiSettings and not in the settings dialog: it and the
     * penalty itself are two knobs
     * for one symptom, and a user whose model is repeating would reach
     * for both and then not know which one did anything. Lives here
     * rather than beside the other sampling values for exactly that
     * reason - "set once per machine", the same shelf `prefill` sits
     * on. 0 = off, -1 = the whole generation, >0 = a window. */
    int  repeat_last_n;
    int  done_seen;             /* a WM_APP_GEN_DONE has been handled */
    int  done_rc;
    /* The status line's RESTING text - what it says when nothing is
     * happening. Not "whatever it said a moment ago": errors are
     * transient and are never written here, so a failure never lingers
     * into the next job's idle text. The selftest caught it; by
     * inspection it reads as obviously correct. */
    char idle_status[160];
    /* Which kind of job is running. finish_job has to know: a load that
     * succeeds changes what the window IS, a generation that succeeds
     * changes nothing but the transcript. One handler, two meanings, so
     * the meaning is carried explicitly rather than guessed from state
     * that the job itself just changed. */
    int  job_kind;
    LZGuiModel mdl;
    LZMru mru;                  /* the File menu's recent-model list */
    LZGuiSession sess;
    LZGuiSettings set;
    WNDPROC input_proc;         /* the EDIT's original window procedure */
    WNDPROC transcript_proc;    /* the RichEdit's original window procedure */
    /* Forces create_toolbar to return NULL, as if comctl32 were absent.
     * This field exists ONLY for the selftest: "no comctl32" cannot be
     * produced on a machine that has the class, so the only way to walk
     * that path at all is to fake the one call that fails on NT 3.51. */
    int  no_toolbar;
    /* lz_drop_accept's own return from WM_CREATE - whether
     * DragAcceptFiles actually resolved. Not a visible control, so
     * nothing else can read this state back the way the idle-state
     * checks read a menu or a button; the selftest wiring gate below
     * is what this field exists for. */
    int  drop_on;
    /* lz_ui_untheme's own return from create_children, for each of
     * the two controls it is called on - same reason
     * drop_on exists just above: neither is a visible control state
     * anything else can read back, and the selftest wiring gate below
     * is what these fields exist for. input_untheme_ok also lets that
     * gate compare the transcript's outcome against the input box's -
     * the two must agree, since SetWindowTheme is not RichEdit-specific
     * (gui/compat40.c's lz_ui_untheme has the fuller reasoning). */
    int  input_untheme_ok;
    int  transcript_untheme_ok;
    /* Same field, third control - side_panel_sync's own
     * candidate LISTBOX, created and destroyed with the model rather
     * than once at WM_CREATE like the two above, so this is
     * overwritten on every (re)creation rather than set once. Only
     * meaningful while g.part[LZ_GUI_SIDE_CAND] is non-NULL - the
     * enumeration gate below knows to skip it otherwise. */
    int  cand_untheme_ok;
    int  cand_title_untheme_ok;
    /* Every OTHER g.part[] control's own lz_ui_untheme return - the
     * structural half of the same fix. A stored value, not a fresh
     * probe, is what makes the consistency gate actually sensitive to
     * "this call site never ran": a fresh probe in the test would
     * silently repair a forgotten call and never go red - the whole
     * point of storing it here. Parallel to g.part[] itself, filled
     * at the SAME creation sites (create_children's own mandatory-parts
     * loop, right next to the apply_font call already there) rather
     * than one field per control.
     * SPLIT/SIDE_LAMPS/SIDE_CAND entries stay at 0 (never written) -
     * SIDE_CAND has its own field above, SIDE_LAMPS has elamp_untheme_
     * ok below, SPLIT has no window ever. */
    int  part_untheme_ok[LZ_GUI_PART_COUNT];
    int  elamp_untheme_ok[16];
    /* Status bar part 1: tokens the current history renders
     * to, or < 0 when there is no model to count with - see
     * update_ctx_cell/set_ctx_cell. Kept here rather than recomputed
     * inline at paint time because lz_gui_session_token_count runs a
     * full BPE pass; the cell is refreshed at the handful of points
     * where the answer could have changed, not on every WM_PAINT. */
    int  ctx_tokens;
    /* The modeless Find dialog, or NULL when none is open.
     * comdlg32 owns its window; this is only the handle FindTextA
     * returned, read by the message loop (IsDialogMessage) and cleared
     * on FR_DIALOGTERM. */
    HWND find_dlg;
    /* How many WM_APP_INSPECT messages the window has handled, ever
     * Not read by
     * segment C's own repaint - that reads (LZInspect *)lp directly out
     * of the message, same as WM_APP_TOKENS reads its buffer - this
     * exists purely so a test can tell "the throttle sent one" apart
     * from "the throttle sent one AND the unconditional finishing send
     * added a second", the same role done_seen plays for
     * WM_APP_GEN_DONE just above. */
    int  inspect_seen;
    /* A copy of the most recent WM_APP_INSPECT's payload, taken in the
     * case below before the message is freed. Segment C (the panel
     * itself, not built yet) would read straight off (LZInspect *)lp at
     * dispatch time and never need this; it exists now purely so a test
     * can tell WHICH frame the finishing send delivered, not merely
     * that a send happened - see inspect_seen's own comment for the
     * counting half of that. */
    LZInspect last_inspect;
} g;

/* IDM_* moved to gui/resource.h: the accelerator table needs them and a
   resource compiler cannot read an enum. */

/* Defined much further down, next to the commands it greys; called from
   build_menu_bar (immediately below) and from start_job, both of which
   come first in this file. */
static void rollback_sync(void);
/* Same shape: push_caption is defined beside apply_language far below, but
   finish_job (model load) and WM_CREATE call it before that point. */
static void push_caption(void);

/* Menu bar layout: five popups, each a NULL-terminated list of
   (id, label, live). id 0 ends the list. `live` is whether the item
   does anything yet - see the file banner; a live item that silently
   does nothing looks exactly like a bug, and empty menu items are not
   allowed. */
typedef struct { UINT id; int label; int live; } LZMenuItem;
/* id 0xFFFF is the separator marker; id 0 ends the list. */
#define LZ_MENU_SEP { 0xFFFF, 0, 0 }

/* Clear is here and not only on the tool bar. On NT 3.51 there is no
   comctl32 and therefore no tool bar at all, and a command that lives
   nowhere else would simply vanish on the floor - the same rule already
   applied to accelerators, applied here to the other kind of invisible
   command. */
static const LZMenuItem MENU_FILE[] = {
    { IDM_OPEN_CHAT,  LZ_STR_MENU_OPEN_CHAT,  1 },
    { IDM_SAVE_CHAT,  LZ_STR_MENU_SAVE_CHAT,  1 },
    { IDM_CLEAR,      LZ_STR_MENU_CLEAR,      1 },
    LZ_MENU_SEP,
    { IDM_EXIT,       LZ_STR_MENU_EXIT,       1 },
    { 0, 0, 0 }
};
/* Two commands, forwarded to whichever of the transcript or the input
   box has focus - see the IDM_COPY/IDM_SELECT_ALL case in wndproc. The
   same pair is also what the right-click popup in show_edit_popup
   carries, so a menu edit here has to stay in step with that array too. */
static const LZMenuItem MENU_EDIT[] = {
    { IDM_COPY,       LZ_STR_MENU_COPY,       1 },
    { IDM_SELECT_ALL, LZ_STR_MENU_SELECT_ALL, 1 },
    { IDM_FIND,       LZ_STR_MENU_FIND,       1 },
    LZ_MENU_SEP,
    /* The three rollback commands. Under Edit rather than
       File: they edit the conversation, and File in this program is
       about the model and the .txt on disk. */
    { IDM_REGEN,      LZ_STR_MENU_REGEN,      1 },
    { IDM_EDIT_LAST,  LZ_STR_MENU_EDIT_LAST,  1 },
    { IDM_DEL_LAST,   LZ_STR_MENU_DEL_LAST,   1 },
    { 0, 0, 0 }
};
static const LZMenuItem MENU_MODEL[] = {
    { IDM_OPEN_MODEL, LZ_STR_MENU_OPEN_MODEL, 1 },
    { IDM_STOP_GEN,   LZ_STR_MENU_STOP,       1 },
    { 0, 0, 0 }
};
/* Language lives under Settings rather than in a bar of its own: a
   fifth top-level popup holding two items is not what a Win9x menu bar
   looked like, and this is a setting. */
static const LZMenuItem MENU_SETTINGS_POPUP[] = {
    { IDM_SETTINGS,   LZ_STR_MENU_SETTINGS,   1 },
    LZ_MENU_SEP,
    { IDM_LANG_ZH,    LZ_STR_MENU_LANG_ZH,    1 },
    { IDM_LANG_EN,    LZ_STR_MENU_LANG_EN,    1 },
    { 0, 0, 0 }
};
static const LZMenuItem MENU_HELP[] = {
    { IDM_ABOUT,      LZ_STR_MENU_ABOUT,      1 },
    { 0, 0, 0 }
};

static void build_menu_bar(HWND hwnd) {
    static const struct {
        int title;
        const LZMenuItem *items;
    } BAR[] = {
        { LZ_STR_MENU_FILE,           MENU_FILE },
        { LZ_STR_MENU_EDIT,           MENU_EDIT },
        { LZ_STR_MENU_MODEL,          MENU_MODEL },
        { LZ_STR_MENU_SETTINGS_TITLE, MENU_SETTINGS_POPUP },
        { LZ_STR_MENU_HELP,           MENU_HELP }
    };
    HMENU bar = CreateMenu();
    int i, j;
    if (!bar) return;
    for (i = 0; i < (int)(sizeof BAR / sizeof BAR[0]); i++) {
        HMENU m = CreatePopupMenu();
        if (!m) { DestroyMenu(bar); return; }
        for (j = 0; BAR[i].items[j].id; j++) {
            const LZMenuItem *it = &BAR[i].items[j];
            if (it->id == 0xFFFF) {
                /* The recent list, right above the separator that leads
                   to Exit - which is where Win9x put it, and why Exit
                   stays last in MENU_FILE rather than growing its own
                   entry here. Nothing is drawn when the list is empty -
                   an "1." with no path is worse than no block - so a
                   fresh install's File menu omits the recent list. */
                if (BAR[i].items == MENU_FILE && g.mru.n > 0) {
                    int k;
                    AppendMenuA(m, MF_SEPARATOR, 0, NULL);
                    for (k = 0; k < g.mru.n; k++) {
                        char label[LZ_MRU_LEN + 8];
                        /* "&1 " gives the item its mnemonic; the path's
                           own ampersands must be doubled or the menu
                           eats them and underlines a letter of the
                           directory name. */
                        int p = sprintf(label, "&%d ", k + 1);
                        const char *s = g.mru.item[k];
                        while (*s && p < (int)sizeof label - 2) {
                            if (*s == '&')
                                label[p++] = '&';
                            label[p++] = *s++;
                        }
                        label[p] = '\0';
                        AppendMenuA(m, MF_STRING | MF_ENABLED,
                                    (UINT)(IDM_MRU0 + k), label);
                    }
                }
                AppendMenuA(m, MF_SEPARATOR, 0, NULL);
            } else {
                /* The language pair carries its own tick. Baked in here
                   because the whole bar is rebuilt on a change anyway,
                   and because CheckMenuRadioItem is a 4.0 call that the
                   3.51 floor hides - MF_CHECKED is 3.1-era. It draws a
                   check rather than a bullet on the floor; that is the
                   difference between the two, and it is not worth an
                   owner-drawn menu. */
                UINT tick = 0;
                if (it->id == IDM_LANG_ZH)
                    tick = lz_str_lang_english() ? 0 : MF_CHECKED;
                else if (it->id == IDM_LANG_EN)
                    tick = lz_str_lang_english() ? MF_CHECKED : 0;
                AppendMenuA(m, MF_STRING | tick |
                            (it->live ? MF_ENABLED : MF_GRAYED),
                            it->id, lz_str_display((LZStr)it->label));
            }
        }
        AppendMenuA(bar, MF_POPUP, (UINT_PTR)m,
                    lz_str_display((LZStr)BAR[i].title));
    }
    /* SetMenu does NOT destroy the menu it replaces, so rebuilding on
       every language change would leak an HMENU and its four popups per
       switch. Read the old one first, swap, then destroy. */
    {
        HMENU old = GetMenu(hwnd);
        SetMenu(hwnd, bar);
        if (old) DestroyMenu(old);
    }
    DrawMenuBar(hwnd);
    /* A FRESH bar has every item live, so anything whose grey state
       follows the window's own state has to be re-applied right here -
       the same argument apply_language already makes, in the same
       words, for the tool strip it rebuilds.
       Only the rollback three. ID_SEND and IDM_STOP_GEN are re-applied
       by their own callers instead, which leaves a real (small) gap:
       finish_job greys Stop and THEN, on a successful load, rebuilds
       this bar, so the menu item comes back live while the toolbar
       button stays grey. Harmless - lz_worker_request_stop with no job
       is a documented no-op - and moving those two here is a separate
       change with its own gate to write, so it is named rather than
       done in passing. */
    rollback_sync();
}

enum { JOB_NONE = 0, JOB_LOAD, JOB_GENERATE };

static LRESULT CALLBACK input_subclass(HWND h, UINT msg, WPARAM wp,
                                       LPARAM lp);
static LRESULT CALLBACK transcript_subclass(HWND h, UINT msg, WPARAM wp,
                                            LPARAM lp);
static void show_edit_popup(HWND ctl, POINT pt);
static void sys_line(LZStr id);
static void sys_line_fmt(const char *utf8, int len);

/* FINDMSGSTRING's registered value, 0 until create_children
 * registers it. Declared up here, ahead of create_children's own
 * definition, rather than beside transcript_find/open_find further
 * down - those come after create_children in this file, and a plain
 * static variable (unlike a function) has no separate forward-
 * declaration form to lean on. */
static UINT g_findmsg;

/* How wide the state cell has to be to hold the widest thing it will
 * ever say, in the language that is loaded.
 *
 * Not two thirds of the bar: on a 656-pixel window that would be 400
 * pixels around LZ_STR_STATE_NO_MODEL, and a status bar whose first panel is
 * mostly empty is the thing that reads as "not a status bar" - Word
 * 95's panels are sized to their contents (Page 1 / Sec 1 / 1/1 fills
 * its own panel), so its bar reads as a row of fields rather than as a
 * strip with a word at one end.
 *
 * Measured rather than tabulated because the four strings are not the
 * same width in the two languages: LZ_STR_STATE_GENERATING is 60 pixels
 * of Latin ("generating") and 36 of SimSun in Chinese, and a constant
 * tuned on one of them is wrong on the other. */
static int status_state_w(HWND sb) {
    static const int ids[5] = {
        LZ_STR_STATE_NO_MODEL, LZ_STR_STATE_LOADING,
        LZ_STR_STATE_READY,    LZ_STR_STATE_GENERATING,
        LZ_STR_STATE_TOKCELL   /* the throughput cell replaces the state
                                  string in part 0 while a generate runs,
                                  and it is the WIDEST thing part 0 ever
                                  shows - filled in it reads "generating
                                  12345 tok, 123.4 tok/s", far longer than
                                  LZ_STR_STATE_READY. Without it here the
                                  panel clips the rate mid-run. */
    };
    HDC dc;
    HFONT f, of = NULL;
    int i, w = 0;

    if (!sb) return 0;
    dc = GetDC(sb);
    if (!dc) return 0;
    f = (HFONT)SendMessage(sb, WM_GETFONT, 0, 0);
    if (f) of = (HFONT)SelectObject(dc, f);
    for (i = 0; i < 5; i++) {
        SIZE s;
        const char *t = lz_str_display(ids[i]);
        if (!t) continue;
        if (GetTextExtentPoint32A(dc, t, (int)strlen(t), &s) && s.cx > w)
            w = (int)s.cx;
    }
    if (of) SelectObject(dc, of);
    ReleaseDC(sb, dc);
    /* The panel's own inset on both sides plus the breathing room Win9x
       left around a status field. Without it the text touches the
       sunken edge it sits in. */
    return w + 14;
}

/* THE BAR'S OWN PANEL BORDER IS NOT THE ONE WORD 95 HAS.
 *
 * comctl32 here draws a part boundary as a single flat grey line
 * (measured: 180,180,180 at the boundary, 184/220 at the left edge);
 * Word 95's panels are recessed boxes - COLOR_3DSHADOW along the top and
 * left, COLOR_3DHILIGHT along the bottom and right, which is
 * BDR_SUNKENOUTER. Held side by side that is the whole difference, and
 * it is why sizing the panels to their contents did not by itself make
 * the bar read as a status bar.
 *
 * So the bevel is painted here, after the control has finished. Parts 0
 * and 1 only: part 2 holds the two lamps, and the point of its
 * SBT_NOBORDERS is that they are NOT sitting in a well.
 *
 * SB_GETRECT rather than arithmetic over the widths handed to
 * SB_SETPARTS: the bar keeps its own margins and the grip's corner, and
 * a second opinion about where a part is would drift from the first.
 *
 * WM_PRINTCLIENT as well as WM_PAINT, and not only so that screenshots
 * agree with the screen: anything that asks a window to render into a DC
 * it supplies goes down that path and would otherwise get the bar
 * without its bevels. */
static WNDPROC g_sb_orig;

#ifndef WM_PRINTCLIENT
#define WM_PRINTCLIENT 0x0318
#endif

static LRESULT CALLBACK sb_bevel_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    LRESULT r = CallWindowProc(g_sb_orig, h, m, w, l);
    if (m == WM_PAINT || m == WM_PRINTCLIENT) {
        HDC dc = (m == WM_PRINTCLIENT) ? (HDC)w : GetDC(h);
        if (dc) {
            RECT pr;
            int i;
            /* Parts 0 and 1 only. Part 2 is the prefill indicator's
               segment and the CONTROL sitting in it draws its own well -
               bevelling the part as well put a second frame around the
               first. Part 3 holds the lamps and gets none either; the
               point of its SBT_NOBORDERS is that they are NOT in a well.
               The simulated strip draws part 2's well itself for the
               same reason in reverse: there is no control there to do
               it. */
            for (i = 0; i < 2; i++)
                if (SendMessage(h, LZ_SB_GETRECT, (WPARAM)i, (LPARAM)&pr))
                    lz_draw_edge(dc, &pr, LZ_BDR_SUNKENOUTER, LZ_BF_RECT);
            if (m == WM_PAINT) ReleaseDC(h, dc);
        }
    }
    return r;
}

/* One way in for every part's text, because SBT_NOBORDERS is a flag on
   the MESSAGE: send SB_SETTEXT without it anywhere and that part gets
   the control's own flat border back on the next update, next to the
   bevels drawn above - a defect that would appear only after whatever
   event calls that one site. */
static char g_sb_cache[3][160];

static void sb_text(int part, const char *s) {
    HWND h = g.part[LZ_GUI_STATUS];
    if (part < 0 || part > 2) return;
    lstrcpynA(g_sb_cache[part], s ? s : "", (int)sizeof g_sb_cache[part]);
    if (!h) return;
    if (!g.status_is_sbar) {
        /* The fallback strip paints from this cache, so a write to it is
           what makes the cell change - there is no control to tell. */
        InvalidateRect(h, NULL, FALSE);
        return;
    }
    SendMessage(h, LZ_SB_SETTEXT,
                (WPARAM)(part | LZ_SBT_NOBORDERS),
                (LPARAM)g_sb_cache[part]);
}

/* SB_SETPARTS CLEARS EVERY PART'S FLAGS, so all three have to be sent
 * again or the control's own borders come back the moment the window is
 * resized - which is also the first thing that happens after the bar is
 * created. That is why SBT_NOBORDERS looked inert: relayout set the
 * parts and then re-sent only part 2, so parts 0 and 1 lost the flag in
 * the same breath they were given it. Proved by turning the bevel
 * drawing off and looking: the borders at x=8/9 and x=72/73 were still
 * there, so the control had never stopped drawing them.
 *
 * Sends straight from the cache instead of calling sb_text, which would
 * hand each buffer to lstrcpynA as both source and destination. That
 * self-copy empties the bar: the text vanishes and the borders go with
 * it, which reads as "the fix broke the status bar" rather than as one
 * overlapping copy. */
static void sb_reflag(void) {
    HWND h = g.part[LZ_GUI_STATUS];
    int i;
    if (!h || !g.status_is_sbar) return;
    /* All FOUR parts: SB_SETPARTS clears every part's flags, and a part
       left unflagged gets the control's own border back beside the
       hand-drawn bevels. Parts 2 and 3 carry no text of their own - the
       indicator and the lamps are drawn into them - so they are flagged
       with an empty string rather than skipped. */
    for (i = 0; i < 4; i++)
        SendMessage(h, LZ_SB_SETTEXT,
                    (WPARAM)(i | LZ_SBT_NOBORDERS),
                    (LPARAM)(i < 3 ? g_sb_cache[i] : ""));
}

/* ---------------------------------------------- the fallback strip
 *
 * What the status bar is where comctl32 is not: NT 3.51, and any host
 * running with classic_ui (gui/compat40.h). A plain STATIC was the old
 * answer and it lost the two extra cells, the lamps and the panel
 * bevels; this draws them.
 *
 * The part BOUNDARIES are not computed here. relayout computes them once
 * for both strips and leaves them in g_sb_p0/g_sb_p1, so the simulated
 * bar and comctl32's cannot disagree about where a cell begins - which
 * is also what makes the two comparable pixel for pixel.
 *
 * Bevels are drawn by hand rather than through lz_draw_edge: that
 * degrades to drawing NOTHING on the floor (compat40.h), which is
 * exactly the system this strip exists for. COLOR_BTNSHADOW and
 * COLOR_BTNHIGHLIGHT are Win3.x-era and present everywhere. */
#define LZ_SBCLASS "Kunkun98Status"
static int g_sb_p0, g_sb_p1;        /* right edge of part 0, of part 1 */
/* A custom class answers WM_SETFONT/WM_GETFONT itself or it has no font
   at all: DefWindowProc stores nothing, WM_GETFONT then returns 0, and
   status_state_w - which measures part 0's text through exactly that
   message - sizes the cell with the system default instead of the UI
   font. That put the simulated bar's first boundary 11 pixels off
   comctl32's, which the strip comparison is what caught. */
static HFONT g_sb_font;
/* comctl32's own cell metrics, measured through the strip comparison:
   the gap before a cell's left edge, and the face rows above the cells. */
#define LZ_SB_GAP 2
#define LZ_SB_TOP 2
/* The prefill indicator's own width. It is a SHORT bar sitting between
   the context cell and the lamps, not a fill of the whole cell - the
   context reading stays visible while a prefill runs, and a status bar
   whose middle turns into one long bar reads as a modal progress dialog
   rather than as an indicator. */
/* Sized so the well holds a WHOLE number of chunks and nothing over.
   A chunk is 8 wide with a 2-wide gap after it, but the last one needs
   no trailing gap, so an exact fit is 10n - 2 of inner width. Nine
   chunks is 88, and 88 plus the inset on each side is 94.
   Picked that way rather than round, because a well two pixels wider
   than its content leaves a sliver that reads as a rendering fault. */
#define LZ_PROG_W  94
/* How far the chunks sit inside the segment. Three, from the v5
   control's own client area: 14 pixels tall, which is the 20-pixel
   segment less three a side. The number describes the CHUNKS - what
   the comctl32 control does inside its own window is its business. */
#define LZ_PROG_INSET 3

/* Prefill progress: written by the ENGINE on the worker thread through
   LZGenOpts.on_prefill, read by a timer on the UI thread. Two plain
   ints, because the handler must not touch a control - worker.h's rule
   for the token path applies here too - so it records and the tick
   draws. A tick that catches one field from the previous slice and one
   from the next is a frame stale, which the tick-based display already
   accepts everywhere else. */
/* Prefill progress, ACCUMULATED ACROSS SEGMENTS. One turn can prefill
   in more than one go: on a prefix-cache miss lz_prefix_prepare
   forwards the reusable part and reports 0..n1, and then the resume
   path forwards the generation-prompt tail and reports 0..n2 - two
   independent ranges through one callback. Reported raw, the bar ran to
   100% and jumped back to 0%. g_pf_base carries the earlier segments so
   `done` only ever moves forward; the denominator grows when a new
   segment appears, because that is the moment the work becomes known -
   neither end of the chain can know n2 before the tokeniser runs. */
static int g_pf_done, g_pf_total, g_pf_base, g_pf_seen;

/* Is a prefill in progress? Three places ask - the fallback strip's
   paint, the indicator tick, and set_status, which has to know that the
   throughput cell is not the thing to show yet. One spelling, because
   three copies of "> 0 && <" drift and the drift is invisible: each
   site would simply disagree about what phase the job is in. */
static int prefill_active(void) {
    return g_pf_total > 0 && g_pf_done < g_pf_total;
}

/* `debug_prefill_ms` in kunkun98.ini: loop a fake prefill of that many
 * milliseconds, so the indicator can be looked at.
 *
 * A real prefill finishes well inside one refresh tick on any prompt
 * short enough to type, so it is gone before it has been drawn twice.
 * Front-end only: it drives the same two counters the engine's callback
 * writes and changes nothing below the GUI, so what it exercises is the
 * paint path itself and it needs no model. 0 (default) is off. */
static int   g_dbg_prefill_ms;
static double g_dbg_prefill_t0;

/* 4.0-era, hidden at the 3.51 floor. Value is fixed by the ABI, same
   argument gui/captionwnd.c makes for the constants it spells out. */
#ifndef DT_END_ELLIPSIS
#define DT_END_ELLIPSIS 0x00008000
#endif

/* GetSysColorBrush is a 4.0 EXPORT and the floor does not have it - the
   Watcom build reports it as an implicit int-returning function, which
   is the shape this project's -we floor exists to catch. GetSysColor and
   CreateSolidBrush are both Win3.x. Cached for the life of the process,
   the way the system's own brushes are, so no caller has to track
   ownership; the set is small and fixed. */
/* One cached brush per system colour this strip paints with, keyed by
   INDEX rather than by a chain of comparisons with a default arm - a
   default arm aliases an unlisted index onto whatever sits in the last
   slot. An index with no slot gets no brush, so a mistake shows up as
   nothing drawn rather than as the wrong thing drawn. */
static const int LZ_SB_COLORS[] = {
    COLOR_BTNSHADOW, COLOR_BTNHIGHLIGHT, COLOR_BTNFACE, COLOR_HIGHLIGHT
};
#define LZ_SB_NCOLORS ((int)(sizeof LZ_SB_COLORS / sizeof LZ_SB_COLORS[0]))
static HBRUSH g_sb_br[LZ_SB_NCOLORS];

static HBRUSH sb_brush(int idx) {
    int k;
    for (k = 0; k < LZ_SB_NCOLORS; k++)
        if (LZ_SB_COLORS[k] == idx) break;
    if (k == LZ_SB_NCOLORS) return NULL;
    if (!g_sb_br[k]) g_sb_br[k] = CreateSolidBrush(GetSysColor(idx));
    return g_sb_br[k];
}

/* Drop the cache so the next paint asks the system again.
 *
 * The colours were never literals - they come from GetSysColor - but a
 * brush cached for the life of the process freezes whatever the scheme
 * was at startup, which is the same thing as hardcoding it one step
 * later. The strip has to follow a scheme change, and the only moment
 * it can learn of one is WM_SYSCOLORCHANGE. */
static void lamps_reload(void);

static void sb_brush_reset(void) {
    int i;
    for (i = 0; i < LZ_SB_NCOLORS; i++) {
        if (g_sb_br[i]) { DeleteObject(g_sb_br[i]); g_sb_br[i] = NULL; }
    }
}

/* One sunken cell, BDR_SUNKENOUTER's shape: shadow along the top and
   left, highlight along the bottom and right. */
static void sb_sink(HDC dc, int x0, int y0, int x1, int y1) {
    RECT e;
    HBRUSH sh = sb_brush(COLOR_BTNSHADOW);
    HBRUSH hi = sb_brush(COLOR_BTNHIGHLIGHT);
    if (x1 - x0 < 2 || y1 - y0 < 2) return;
    e.left = x0; e.top = y0; e.right = x1; e.bottom = y0 + 1;
    FillRect(dc, &e, sh);
    e.right = x0 + 1; e.bottom = y1;
    FillRect(dc, &e, sh);
    e.left = x0; e.top = y1 - 1; e.right = x1; e.bottom = y1;
    FillRect(dc, &e, hi);
    e.left = x1 - 1; e.top = y0;  e.bottom = y1;
    FillRect(dc, &e, hi);
}

/* The Win95 sizing grip: three RIDGES running parallel to the corner's
   anti-diagonal, the one furthest from the corner the longest, each a
   two-pixel shadow with a one-pixel highlight up-left of it - that pair
   is what gives a ridge its round rather than reading as a flat line.
   Ridges that would not fit the strip's height are dropped whole, so a
   short bar loses a ridge instead of bleeding past its own edge. */
static void sb_grip(HDC dc, const RECT *rc) {
    HBRUSH sh = sb_brush(COLOR_BTNSHADOW);
    HBRUSH hi = sb_brush(COLOR_BTNHIGHLIGHT);
    int i, j;
    /* Read off comctl32's own grip rather than guessed: the ridges are
       FIVE apart, not four; each is one highlight pixel followed by two
       shadow, in that order left to right; the feet sit two rows up from
       the bottom; and the whole thing is clipped two columns in from the
       right, which is why the shortest ridge ends in a bare highlight
       with its shadow cut off. */
    int right = rc->right - 2;          /* last column the grip may touch */
    int base  = rc->bottom - 2;         /* the row the ridge feet stand on */
    for (i = 0; i < 3; i++) {
        int d = 3 + i * 5;
        if (base - d < rc->top) break;
        for (j = 0; j <= d; j++) {
            int x = right - d + j;
            int y = base - j;
            RECT p;
            if (x < rc->left || y < rc->top) continue;
            p.left = x; p.top = y; p.right = x + 1; p.bottom = y + 1;
            FillRect(dc, &p, hi);
            p.left  = x + 1;
            p.right = (x + 3 > right + 1) ? right + 1 : x + 3;
            if (p.right > p.left) FillRect(dc, &p, sh);
        }
    }
}

/* Where the prefill indicator sits, in the strip's client coordinates.
 * ONE function, called by the native MoveWindow and by the fallback
 * paint, for the same reason g_sb_p0/g_sb_p1 are computed once: two
 * copies of a rectangle drift, and the drift shows up as the simulated
 * bar sitting a pixel off the real one.
 * Returns 0 when there is no room for it. */
static int sb_prog_rect(int strip_h, RECT *out) {
    int pw = LZ_PROG_W;
    if (pw > (g_sb_p1 - g_sb_p0) - LZ_SB_GAP * 3)
        pw = (g_sb_p1 - g_sb_p0) - LZ_SB_GAP * 3;
    if (pw <= 0 || strip_h <= LZ_SB_TOP + 2) return 0;
    /* A CELL: the same top inset and flush bottom as the other two. A
       sunken rectangle nested inside another one reads as a hole
       punched in the bar rather than as a segment of it. */
    out->right  = g_sb_p1;
    out->left   = out->right - pw;
    out->top    = LZ_SB_TOP;
    out->bottom = strip_h;
    return 1;
}

/* The bar's own area inside that segment. Both paths use it: the
 * fallback fills chunks into it, the comctl32 build puts the control
 * there - so the two cannot end up with the bar at different heights.
 * Returns 0 when the inset leaves nothing. */
static int sb_prog_inner(const RECT *cell, RECT *out) {
    out->left   = cell->left + LZ_PROG_INSET;
    out->right  = cell->right - LZ_PROG_INSET;
    out->top    = cell->top + LZ_PROG_INSET;
    out->bottom = cell->bottom - LZ_PROG_INSET;
    return (out->right - out->left) > 4 && (out->bottom - out->top) > 2;
}

/* The Win9x progress bar, drawn.
 *
 * Measured off comctl32 v5 rather than remembered - the manifest was
 * temporarily removed to get a v5 control on this host, since v6 draws
 * one smooth fill and the target has no v6 at all. What that showed:
 * chunks eight wide, two apart, starting one pixel in, in
 * COLOR_HIGHLIGHT on COLOR_BTNFACE. Both are asked for by index, so the
 * target's own navy-on-silver comes out without being written down.
 * The sunken edge is this bar's own: it sits inside the status strip,
 * where a flat rectangle would read as a gap rather than as a well. */
static void sb_prog_paint(HDC dc, const RECT *r, int done, int total) {
    RECT fill, in;
    int x, chunks;

    /* The segment: the same sunken treatment parts 0 and 1 get, from
       the same function, so the three read as three segments. */
    sb_sink(dc, r->left, r->top, r->right, r->bottom);
    if (!sb_prog_inner(r, &in)) return;
    if (total <= 0 || done <= 0) return;
    if (done > total) done = total;

    /* THE COUNT IS ROUNDED, NOT THE PIXEL SPAN. comctl32 divides the
       well into whole chunks first and then scales that COUNT - at 40%
       of a 92-pixel well it draws four, where scaling the span and
       fitting chunks into it gives three. Measured against the v5
       control, not reasoned from the geometry. */
    /* +2 because the final chunk carries no trailing gap - without it
       the well loses its last chunk and never reads as full. */
    chunks = (in.right - in.left + 2) / 10;
    if (chunks <= 0) return;
    chunks = (int)(((long)chunks * done + total / 2) / total);
    for (x = in.left; chunks > 0; x += 10, chunks--) {
        if (x + 8 > in.right) break;
        fill.left = x; fill.right = x + 8;
        fill.top = in.top; fill.bottom = in.bottom;
        FillRect(dc, &fill, sb_brush(COLOR_HIGHLIGHT));
    }
}

static void sb_fallback_paint(HWND h, HDC dc) {
    RECT rc, t;
    HFONT f = g_sb_font ? g_sb_font : lz_ui_font(), old = NULL;
    char text0[160];
    int mid;

    GetClientRect(h, &rc);
    FillRect(dc, &rc, sb_brush(COLOR_BTNFACE));

    /* Part 0's text is the control's own: set_status writes it with
       SetWindowTextA on this path, and keeping that the authority means
       nothing else has to change to feed this strip. Part 1 comes from
       the cache every part is written through. */
    text0[0] = '\0';
    GetWindowTextA(h, text0, (int)sizeof text0);

    /* Cell rects as comctl32 actually lays them out, read off the strip
       comparison rather than assumed: two rows of face above them,
       flush with the bottom, and the gap between two cells sits on the
       RIGHT cell's left edge only - the left cell's highlight stays at
       the boundary. Each of those three was wrong in an earlier guess
       and each showed up as a full-width or full-height band in the
       difference image. */
    mid = g_sb_p1 > g_sb_p0 ? g_sb_p1 : g_sb_p0;
    /* Cell 1 ends where the indicator's cell begins - three segments in
       a row, not two with something laid over the second. */
    {
        RECT pcell;
        if (sb_prog_rect(rc.bottom - rc.top, &pcell) &&
            pcell.left - LZ_SB_GAP > g_sb_p0 + LZ_SB_GAP)
            mid = pcell.left - LZ_SB_GAP;
    }
    sb_sink(dc, 0, LZ_SB_TOP, g_sb_p0, rc.bottom);
    sb_sink(dc, g_sb_p0 + LZ_SB_GAP, LZ_SB_TOP, mid, rc.bottom);

    if (f) old = (HFONT)SelectObject(dc, f);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, GetSysColor(COLOR_BTNTEXT));
    /* Centred in the CELL, not in the strip: comctl32 measures from the
       cell's own top, so centring on the full height puts the text one
       pixel high. The left inset is the cell's edge plus two. */
    t = rc; t.top = LZ_SB_TOP;
    t.left = LZ_SB_GAP; t.right = g_sb_p0 - LZ_SB_GAP;
    DrawTextA(dc, text0, -1, &t, DT_SINGLELINE | DT_VCENTER | DT_LEFT |
                                 DT_NOPREFIX | DT_END_ELLIPSIS);
    t.left = g_sb_p0 + LZ_SB_GAP * 2; t.right = mid - LZ_SB_GAP;
    DrawTextA(dc, g_sb_cache[1], -1, &t, DT_SINGLELINE | DT_VCENTER |
                                         DT_LEFT | DT_NOPREFIX |
                                         DT_END_ELLIPSIS);
    if (old) SelectObject(dc, old);

    /* Always drawn, empty when there is nothing to report: a segment
       that came and went would re-flow the bar underneath the reader. */
    {
        RECT pr;
        if (sb_prog_rect(rc.bottom - rc.top, &pr))
            sb_prog_paint(dc, &pr, prefill_active() ? g_pf_done : 0,
                          g_pf_total > 0 ? g_pf_total : 1);
    }

    sb_grip(dc, &rc);
}

static LRESULT CALLBACK sb_fallback_proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_ERASEBKGND) return 1;      /* WM_PAINT covers every pixel */
    if (m == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        if (dc) sb_fallback_paint(h, dc);
        EndPaint(h, &ps);
        return 0;
    }
    /* Same reason the comctl32 bar answers it: anything rendering this
       window into a caller-supplied DC - the pixel comparison included -
       goes down this path and would otherwise get a blank strip. */
    if (m == WM_PRINTCLIENT) { sb_fallback_paint(h, (HDC)w); return 0; }
    if (m == WM_SETTEXT) {
        LRESULT r = DefWindowProcA(h, m, w, l);
        InvalidateRect(h, NULL, FALSE);
        return r;
    }
    if (m == WM_SYSCOLORCHANGE) {
        /* The strip's own colours, and its lamps'. The bitmaps carry
           the button face baked in by lz_mapped_bitmap, so a scheme
           change makes them as stale as the brushes - reloading them is
           the same fix, not an extra one. */
        sb_brush_reset();
        lamps_reload();
        InvalidateRect(h, NULL, TRUE);
        return 0;
    }
    if (m == WM_SETFONT) {
        g_sb_font = (HFONT)w;
        if (LOWORD(l)) InvalidateRect(h, NULL, FALSE);
        return 0;
    }
    if (m == WM_GETFONT) return (LRESULT)g_sb_font;
    return DefWindowProcA(h, m, w, l);
}

static void sb_register_fallback(void) {
    static int done;
    WNDCLASSA wc;
    if (done) return;
    done = 1;
    memset(&wc, 0, sizeof wc);
    wc.lpfnWndProc   = sb_fallback_proc;
    wc.hInstance     = g.inst;
    wc.hCursor       = LoadCursorA(NULL, IDC_ARROW);
    wc.lpszClassName = LZ_SBCLASS;
    RegisterClassA(&wc);
}

static void set_status(const char *display_utf8) {
    HWND h = g.part[LZ_GUI_STATUS];
    char cell[80];
    char status_gbk[512];
    /* The status bar and its sidebar mirror are ANSI controls (GBK on
       the target). The text they are handed here is UTF-8 (see the
       caller contract below), so convert once here and write the GBK
       form to the windows. `shown` is the GBK form, so the sidebar
       mirror below (also ANSI/GBK) always gets the converted text,
       never the raw UTF-8 that mojibakes. */
    lz_gbk_from_utf8(display_utf8 ? display_utf8 : "",
                     (int)strlen(display_utf8 ? display_utf8 : ""),
                     status_gbk, (int)sizeof status_gbk, NULL);
    const char *shown = status_gbk;
    if (!h) return;
    /* The throughput cell, on BOTH strips.
       While a generate job runs, and after it ends until the next turn,
       part 0 shows this cell rather than the text passed in: the state
       line ("generating" / "ready") carries no per-turn information and
       this does. The caller is start_job or the throttled tick, so it
       cannot sit stale for the length of a long generation.
       `tok_live` alone is the gate - finish_job leaves it set for a
       successful generation so the FINAL reading stays until the next
       turn, start_job clears it for a load, and errors clear it so an
       error message is never hidden by a stale rate.
       status_gbk is rewritten IN PLACE, which is also what keeps
       `shown` (a pointer into it) following the cell, so the sidebar
       mirror below paints the same text the strip does.

       NOT DURING PREFILL. tok_live is set by start_job, which runs
       before the first token exists - so through the whole prefill this
       cell would answer "0 tok, 0.0 tok/s", and it would do it by
       silently discarding the text the caller passed. That is what made
       the indicator move while the strip and the sidebar said nothing
       about a prefill: the bar is driven by SETPOS, the words come
       through here. During prefill the caller's text is the informative
       one and a rate over zero tokens is not. */
    if (g.tok_live && !prefill_active()) {
        lz_common_tokcell(cell, (int)sizeof cell, g.tok_gen,
                       lz_time_ms() - g.gen_start_ms,
                       lz_str_utf8(LZ_STR_STATE_TOKCELL));
        lz_gbk_from_utf8(cell, (int)strlen(cell),
                         status_gbk, (int)sizeof status_gbk, NULL);
    }
    if (g.status_is_sbar) sb_text(0, status_gbk);
    else                  SetWindowTextA(h, status_gbk);
    /* Sidebar mirrors the same state line under the model info.

       No blank line when there is no model yet: the missing model line
       is not substituted with "\n", so before a model is loaded the
       state sits on the first line rather than two lines below the name
       with nothing in between - a gap that reads as a layout mistake
       rather than as an empty field. */
    if (g.part[LZ_GUI_SIDE_INFO]) {
        char side[912];   /* 400 (side_model) + 1 + 511 (status_gbk) + slack */
        const char *state = shown;
        if (g.side_model[0])
            snprintf(side, sizeof side, "%s\n%s", g.side_model, state);
        else
            snprintf(side, sizeof side, "%s", state);
        SetWindowTextA(g.part[LZ_GUI_SIDE_INFO], side);
    }
}

/* Change the resting text and show it. Whoever changes what the window
 * IS - a model loaded, a model closed - calls this.
 * Stores UTF-8 (set_status converts to GBK for the windows), matching
 * the caller contract set_status now documents. */
/* Store the resting text and show it. Nothing else - see
   set_idle_status for the other half, and why they are two functions. */
static void set_idle_text(const char *display_utf8) {
    strncpy(g.idle_status, display_utf8 ? display_utf8 : "",
            sizeof g.idle_status - 1);
    g.idle_status[sizeof g.idle_status - 1] = '\0';
    set_status(g.idle_status);
}

static void set_idle_status(const char *display_utf8) {
    /* A new RESTING text also retires the previous generate's
       throughput cell. set_idle_status means "what the window IS has
       changed" - a model loaded or closed, the conversation cleared -
       and a rate from the last turn has no business covering that. The
       successful-generation path keeps its rate by NOT going through
       here: finish_job's JOB_GENERATE tail calls set_status directly.
       Split from set_idle_text because that retirement is a SIDE
       EFFECT, and a caller that only re-spells the resting text - a
       language switch - was silently killing the throughput cell of a
       job that was still running, for the whole rest of that job. */
    g.tok_live = 0;
    set_idle_text(display_utf8);
}

/* Status bar part 1 - the cell SB_SETPARTS reserves and nothing else
 * writes to. The number that belongs here is the one the user cannot
 * otherwise see: how close the conversation is to the point where the
 * oldest exchange gets dropped.
 *
 * No fallback for the STATIC-control case (unlike set_status's
 * LZ_SB_SETTEXT/SetWindowTextA split): a plain STATIC label here is
 * part 0's own control on NT 3.51 (see create_children - the
 * comctl32-absent path has no second cell to write into at all), so
 * this simply does nothing there, same as the lamps already do. */
/* The denominator of the context cell, and the answer to "how much
 * context is there".
 *
 * Not LZGuiSettings.ctx, and not a constant. With a model up it is what
 * the run state was ACTUALLY allocated with, which differs from the
 * setting whenever lz_common_ctx_clamp took the model's own cfg->seq_len
 * as the smaller number; with nothing up there is no allocation, so the
 * only honest answer is what the next load will ask for.
 *
 * "The setting moved" and "the setting took effect" are two different
 * claims, and a status bar that reports the first while looking like it
 * reports the second is worse than one that reports neither. */
static int effective_ctx(void) {
    if (g.mdl.have_state && g.mdl.seq_len > 0) return g.mdl.seq_len;
    return g.set.ctx;
}

static void set_ctx_cell(void) {
    char cell[64];
    if (!g.status_is_sbar) return;
    if (g.ctx_tokens < 0) cell[0] = '\0';
    else sprintf(cell, "%s %d/%d", lz_str_display(LZ_STR_STATE_CTX),
                 g.ctx_tokens, effective_ctx());
    sb_text(1, cell);
}

/* Recompute and repaint the context cell. Called at every point the
 * answer could have changed - a turn finishing, a load completing, the
 * conversation being cleared or replaced - never per token: the count
 * is a full BPE pass (lz_gui_session_token_count), the same one
 * lz_generate itself would run, and paying it on every token would be
 * the cost this function's own comment warns against. */
static void update_ctx_cell(void) {
    g.ctx_tokens = lz_gui_session_token_count(&g.sess, NULL, 0);
    set_ctx_cell();
}

/* ------------------------------------------------------- transcript */

/* The UI font's own face name AND its size in twips, for
 * append_run's non-code runs - same GetObjectA(HFONT, LOGFONTA) pattern
 * gui/compat40.c's own lz_richedit_use_font already uses to read a font
 * back, and the same lfHeight-to-twips conversion, because that is the
 * value the control's base format was set FROM.
 *
 * ONE function returning both, not two: the
 * heading sizes are multiples of this height, and a second reader of the
 * same HFONT would be a second authority - they would agree until a
 * language switch changed the font under one of them.
 *
 * Queried fresh every call rather than cached: lz_ui_font() can change
 * under a language switch (apply_font/apply_language), and a stale
 * cached name would silently stop matching once it did - a run every few
 * hundred milliseconds during generation is cheap enough on the target
 * that this is not a hot loop the way a per-BYTE cost would be.
 *
 * `twips` may be NULL when only the face is wanted. */
static void ui_font_metrics(char *out, int cap, int *twips) {
    LOGFONTA lf;
    HFONT hf = lz_ui_font();
    if (hf && GetObjectA((HGDIOBJ)hf, (int)sizeof lf, &lf)) {
        lstrcpynA(out, lf.lfFaceName, cap);
        if (twips) {
            HDC dc = GetDC(NULL);
            int dpi = dc ? GetDeviceCaps(dc, LOGPIXELSY) : 96;
            long px = lf.lfHeight < 0 ? -lf.lfHeight : lf.lfHeight;
            if (dc) ReleaseDC(NULL, dc);
            if (dpi <= 0) dpi = 96;
            *twips = (int)(px * 1440 / dpi);
            /* A font with no height at all would otherwise scale every
               heading to nothing; 9pt is lz_ui_font's own request. */
            if (*twips <= 0) *twips = 9 * 20;
        }
    } else {
        lstrcpynA(out, "MS Sans Serif", cap);   /* matches lz_ui_font's
                                                    own 3.51 fallback */
        if (twips) *twips = 9 * 20;
    }
}

/* Push the whole settings block into the session. ONE function with
 * four callers, not the same two-or-three lines written out four times:
 * the next setting that has to reach the session touches one place, and
 * a gate that wants to exercise "the settings were applied" can call
 * what the product calls instead of copying its body.
 *
 * The seed is a POLICY here, not a value - lz_gui_session_begin draws
 * the actual number per turn (gui/session.h). */
static void apply_settings(void) {
    lz_gui_session_apply(&g.sess, &g.set);
    lz_gui_session_set_seed(&g.sess, g.set.seed_mode, g.set.seed);
    /* repeat_last_n is ini-only and is applied HERE rather than inside
       lz_gui_session_apply, because it is not in LZGuiSettings: it is
       kept out of the dialog on purpose - it and the repetition penalty
       are two knobs for one symptom, and
       a user chasing a repeating model would turn both. Applied after
       the apply call because that call rewrites the whole sampling
       block from the preset first. */
    g.sess.opts.sample.repeat_last_n = g.repeat_last_n;
}

/* Make g.set.ctx real, re-allocating the run state if one exists.
 *
 * Returns 0 when the size in force is now the one that was asked for,
 * and a non-zero LZErr with errbuf filled when the allocation failed -
 * in which case the setting has ALREADY been rolled back to `prev` and
 * written to the ini, and the old run state is still the live one. The
 * caller's whole job on a non-zero return is to show the box.
 *
 * Split that way on purpose. The three things the failure criterion
 * asks for on failure - the old state survives, an error box appears,
 * the setting goes back - are three separate assertions, and a modal
 * MessageBox in the middle of this function would make two of the three
 * unreachable from a selftest. What decides whether the box appears is
 * this function's return code, so the return code is what a gate can
 * assert; the box itself is one `if` at the single call site.
 *
 * NOT lz_gui_session_reset. The conversation is TEXT and does not live
 * in the LZRunState - re-rendering it against a fresh state is what
 * every turn already does - so throwing it away would be a data loss
 * with no technical cause, and would additionally leave the transcript
 * on screen describing a history that is not in the state. The prefix
 * checkpoint IS stale (new state, new epoch) and is cleared: clear, not
 * drop-then-arm, because the MODEL has not changed and lz_prefix_init
 * sizes its 1.51 MB buffer off the model. That is gui/session.h's own
 * lifecycle rule. */
static int ctx_commit(HWND hwnd, int prev, char *errbuf, int errlen) {
    int rc;
    if (errbuf && errlen > 0) errbuf[0] = '\0';
    if (g.set.ctx == prev) return 0;
    /* Drained, not just joined: this frees the LZRunState a generation
       job would be writing into. Same hazard, same answer, as the two
       load call sites. */
    lz_worker_join_drain(hwnd);
    rc = lz_gui_model_resize(&g.mdl, g.set.ctx, errbuf, errlen);
    if (rc != 0) {
        /* Explicitly NOT "fall back to the largest size that does
           allocate" (user decision). That would be the cleverer
           behaviour and it
           would hand the user a number they never asked for, with the
           dialog afterwards showing the number they DID ask for. This
           project's ledger of silent downgrades is long enough. */
        g.set.ctx = prev;
        lz_ini_set_int("ctx", g.set.ctx);
        set_ctx_cell();
        return rc;
    }
    lz_gui_session_prefix_clear(&g.sess);
    /* Written on the success path too, not only on failure. The rule
       "the ini holds the size in force" is then true at every instant
       rather than only after a clean exit, which matters because the
       failure case is exactly the case where the next thing that
       happens might not be a clean exit. */
    lz_ini_set_int("ctx", g.set.ctx);
    set_ctx_cell();
    return 0;
}

static void ctx_apply(HWND hwnd, int prev) {
    static char err[1024];
    if (ctx_commit(hwnd, prev, err, (int)sizeof err) != 0)
        MessageBoxA(hwnd, err, lz_str_display(LZ_STR_ERR_TITLE),
                    MB_ICONHAND | MB_OK);
}

/* Fixed columns for a streamed table row.
 * Six because that is what the design settled on; the WIDTH is a
 * multiple of the body size rather than a twip count, so it follows the
 * font. Neither number has an automated gate - it is one of the three
 * things only the user's eye can settle. */
#define LZ_TABLE_COLS 6

/* Whether the control that got created is RichEdit 2.0-or-later rather
 * than the 1.0 fallback.
 *
 * THE VERSION MAP, from MS's own "About Rich Edit Controls", because
 * this project has now had to derive it twice:
 *
 *   1.0   Riched32.dll   class "RichEdit"
 *   2.0   Riched20.dll   class "RichEdit20A" / "RichEdit20W"
 *   3.0   Riched20.dll   SAME DLL NAME as 2.0 - the file was not
 *                        renamed, so loading riched20 gets whichever of
 *                        the two the machine has and the class name
 *                        cannot tell them apart
 *   4.1   Msftedit.dll   class "RICHEDIT50W"
 *
 * "Each version of rich edit is a superset of the preceding one", so
 * 2.0 is the floor worth targeting and 1.0 is the subset that degrades
 * (user decision: guarantee the 2.0 features first, do not
 * let the 1.0 fallback set the design). lz_richedit_class already asks
 * for riched20 before riched32, so this is a check on which one
 * answered, not a preference expressed here.
 *
 * What 2.0 buys that this file uses: CHARFORMAT2, and specifically its
 * background colour - listed in MS's own 2.0 feature table as one of
 * the additions. On a riched32-only machine the code shading does not
 * appear and everything else is unchanged, the same degradation
 * gui/compat40.c applies to every 4.0-era nicety.
 *
 * Cached: lz_richedit_class picks one DLL at startup and never changes
 * its mind, and this is read once per run during generation. */
static int richedit_v2(void) {
    static int known = -1;
    if (known < 0) {
        const char *c = lz_richedit_class();
        known = (c && strcmp(c, "RichEdit20A") == 0) ? 1 : 0;
    }
    return known;
}

/* Horizontal pixels to twips - the unit PARAFORMAT works in.
 * LOGPIXELSX, not the LOGPIXELSY ui_font_metrics uses: these are two
 * different quantities, not two readings of one, and a display with
 * non-square pixels would want them to disagree. */
static int px_to_twips(int px) {
    HDC dc = GetDC(NULL);
    int dpi = dc ? GetDeviceCaps(dc, LOGPIXELSX) : 96;
    if (dc) ReleaseDC(NULL, dc);
    if (dpi <= 0) dpi = 96;
    return (int)((long)px * 1440 / dpi);
}

static int client_width_twips(HWND h) {
    RECT rc;
    if (!h || !GetClientRect(h, &rc)) return 0;
    return px_to_twips(rc.right - rc.left);
}

/* How far a wrapped bullet line has to be pushed in to sit under the
 * TEXT rather than under the marker: the width of "\xA1\xF1 " - the
 * bullet and its space - in the font the control is actually using.
 *
 * MEASURED, not derived from the point size. A GBK full-width glyph is
 * the em square, so the bullet is base_twips wide, but the space after
 * it is proportional and its width is the font's business. Guessing
 * here would put every wrapped line a few pixels off the text above it,
 * which is precisely the misalignment a hanging indent exists to fix.
 *
 * Falls back to one and a half em when the DC cannot answer, which is
 * the right shape (a bullet plus about half a space) even though it is
 * not the measurement. */
static int bullet_indent_twips(HWND h, int base_twips) {
    HDC dc;
    HFONT f, old;
    SIZE sz;
    int w = 0;
    dc = h ? GetDC(h) : NULL;
    f = lz_ui_font();
    if (dc && f) {
        old = (HFONT)SelectObject(dc, (HGDIOBJ)f);
        if (GetTextExtentPoint32A(dc, "\xA1\xF1 ", 3, &sz)) w = (int)sz.cx;
        SelectObject(dc, (HGDIOBJ)old);
    }
    if (dc) ReleaseDC(h, dc);
    if (w <= 0) return base_twips * 3 / 2;
    return px_to_twips(w);
}

/* Whether the view is parked at the end. Checked BEFORE appending: once
 * the text has grown, "is the last line visible" answers a different
 * question. A user who scrolled up to read something is
 * not asking to be yanked back every time a token arrives. */
static void append_run(void *ud, const char *gbk, int n, int style) {
    HWND h = g.part[LZ_GUI_TRANSCRIPT];
    /* The 2.0 struct always, with cbSize saying which half is meant -
       that is how RichEdit tells the two apart, and it lets the 1.0
       fallback take the identical call with one field fewer. */
    CHARFORMAT2A cf;
    CHARRANGE end;
    static char buf[LZ_STREAM_RUN_MAX + 16];
    char face[LF_FACESIZE];
    int follow, base_twips;

    (void)ud;
    if (!h || n <= 0) return;
    if (n > (int)sizeof buf - 1) n = (int)sizeof buf - 1;
    memcpy(buf, gbk, (size_t)n);
    buf[n] = '\0';

    follow = lz_scroll_at_end(h);

    /* Append means "select nothing, at the end, then replace it". */
    end.cpMin = -1;
    end.cpMax = -1;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&end);

    /* ONE read of the UI font per run, feeding both the face below and
       the heading sizes - see ui_font_metrics on why not two. */
    ui_font_metrics(face, (int)sizeof face, &base_twips);

    memset(&cf, 0, sizeof cf);
    cf.cbSize = richedit_v2() ? (UINT)sizeof cf : (UINT)sizeof(CHARFORMATA);
    /* CFM_FACE always in the mask, not only for code runs - EM_
       SETCHARFORMAT leaves any attribute OUTSIDE dwMask exactly as the
       insertion point already has it, so a run right after a code run
       that did not ask for the face back would inherit Courier New
       forever. Explicit on every call is what keeps this stateless -
       no "was the last run code" flag to keep in sync with clears,
       resets, or a fresh conversation. */
    cf.dwMask = CFM_COLOR | CFM_BOLD | CFM_ITALIC | CFM_FACE | CFM_SIZE;
    cf.crTextColor = (style & LZ_STYLE_THINK) ? LZ_COLOR_THINK : LZ_COLOR_TEXT;
    cf.dwEffects = 0;
    if (style & LZ_STYLE_BOLD)   cf.dwEffects |= CFE_BOLD;
    if (style & LZ_STYLE_ITALIC) cf.dwEffects |= CFE_ITALIC;
    /* Headings are a SIZE, and bold on top - the Win9x idiom, where a
       heading in a document window was the body face two or four points
       up rather than a different family.
       CFM_SIZE is in the mask unconditionally for the same reason
       CFM_FACE is (see below): anything left out of dwMask keeps
       whatever the insertion point already had, so the run after a
       heading would inherit the heading's size forever. yHeight is in
       TWIPS. */
    {
        /* ONE heading size for h1 through h6, not three (user decision:
           headings do not need three sizes). This checkpoint only ever
           writes "###",
           so the difference between the levels would never appear on
           screen. gui/stream.c still PARSES the level
           and still clamps h4-h6 to 3 - that stays gated - the display
           just does not use it beyond heading-or-not.

           x1.5 of the BODY size, as an integer ratio so there is no
           float in a message-loop path, and derived from the running UI
           font rather than written as a point size: a hardcoded 14pt is
           right only on the display the number was read off, and this
           program follows the system font across a language switch and
           a DPI it did not choose. */
        if (style & LZ_STYLE_H_MASK) {
            cf.yHeight = base_twips * 3 / 2;
            cf.dwEffects |= CFE_BOLD;
        } else {
            cf.yHeight = base_twips;
        }
    }
    /* Code: a monospace face, stacked with whatever colour/bold/italic
       also apply - "**`bold code`**" is bold AND monospace, not
       either/or (all three are independent CHARFORMAT
       switches", plural). Courier New has shipped since Windows 3.1,
       so unlike the UI font this needs no probe. */
    if (style & LZ_STYLE_CODE)
        lstrcpynA(cf.szFaceName, "Courier New", (int)sizeof cf.szFaceName);
    else
        lstrcpynA(cf.szFaceName, face, (int)sizeof cf.szFaceName);
    /* Grey behind code (user request). Inline spans get it
       too, not only fenced blocks, and that is the point rather than a
       liberty: Courier New has no CJK glyphs, so the system font-links
       Chinese back to the UI face and a backticked Chinese word comes out
       looking exactly like the prose around it. The face alone marks
       only ASCII code;
       the background marks all of it.
       Explicit in BOTH directions, like every other attribute here -
       CFE_AUTOBACKCOLOR is what puts a non-code run back on the
       window's own colour instead of inheriting the last code span's
       grey. */
    if (richedit_v2()) {
        cf.dwMask |= CFM_BACKCOLOR;
        if (style & LZ_STYLE_CODE) cf.crBackColor = LZ_COLOR_CODE_BG;
        else                       cf.dwEffects |= CFE_AUTOBACKCOLOR;
    }
    SendMessage(h, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);

    /* The one PARAGRAPH property this program sets. A table row's cells
       arrive separated by tabs, and a
       tab with no tab stop lands wherever RichEdit's default grid puts
       it, which is not a column.

       Sent on EVERY run, table or not, for the same reason CFM_FACE is
       always in the character mask: the explicit version is stateless.
       The alternative is a "was the last run a table" flag that has to
       stay in sync with clear, reset and a new conversation - three
       places that have each already broken something else in this file.

       SCF_SELECTION, and the selection is the EMPTY insertion point at
       the end. That confines it to the paragraph being built. It is
       safe to switch back to no-tabs on the run after a table only
       because the row's own '\n' carried the TABLE bit (gui/stream.c's
       line-scoped-bits branch), so by the time a plain run gets here
       the insertion point is already in the NEXT paragraph. Those two
       halves have to be read together; either one alone is wrong.

       Six columns, and the width is derived from the body size so it
       tracks the same font everything else here does - but CLAMPED so
       that all six fit inside the control.

       The clamp is a RichEdit limitation worked around, not a
       refinement. A row wider than the control does not just overflow:
       RichEdit WRAPS it, the continuation starts at x=0, and its next
       tab jumps to the FIRST stop - so the overflowing cell lands under
       COLUMN TWO, interleaved with the row below it. Measured on a
       six-column table at the default 640x480: not "misaligned" but
       unreadable. Six columns at the unclamped inch each need six
       inches; the transcript has about four.

       Clamping to avail/LZ_TABLE_COLS leaves exactly one column's worth
       of room past the last stop a six-column row uses, which is what
       the trailing cell needs.

       What this does NOT survive is a RESIZE: a paragraph keeps the
       stops it was written with, and RichEdit will not recompute them.
       A table written wide and then read in a narrow window wraps
       again. That is strictly better than wrapping at every size, and
       reflowing every earlier paragraph on WM_SIZE is not something
       this front end can afford on the target. */
    {
        PARAFORMAT2 pf;
        int k;
        memset(&pf, 0, sizeof pf);
        pf.cbSize = richedit_v2() ? (UINT)sizeof pf : (UINT)sizeof(PARAFORMAT);
        /* All three fields on every run, zeros included - the same
           stateless-explicit rule the character mask follows. A run
           that left PFM_OFFSET out would inherit the previous
           paragraph's hanging indent. */
        pf.dwMask = PFM_TABSTOPS | PFM_STARTINDENT | PFM_OFFSET;
        /* NO PFM_BORDER, and no EM_INSERTTABLE either. Both are
           measurements (probes run inside this selftest and then
           removed), not omissions:

           PFM_BORDER - riched20 ACCEPTS the border fields and reads
           them back unchanged (wBorders=001F, wBorderWidth=000F,
           exactly what was written) and paints nothing at all. The
           reference agrees, in as many words: MS's "About Rich Edit
           Controls" lists PARAFORMAT2's "border space/width/sides"
           under "FOR RTF ROUNDTRIPPING ONLY". Measurement and document
           converge, which is the only time either is worth much.

           The shape is worth keeping: a gate asserting "the border
           took" would read back the right values and stay green forever
           while no line was ever drawn. Only the screen says otherwise.
           (Paragraph SHADING is in that same roundtripping-only list,
           so it is not an alternative route to a shaded code block
           either - the character background this file does use is.)

           EM_INSERTTABLE (WM_USER+232, with TABLEROWPARMS /
           TABLECELLPARMS - which do carry per-cell border widths and
           colours) - riched20 ignores it: text length 0 before, 0
           after, with ES_READONLY cleared first so a refusal on that
           ground could not be mistaken for the feature being absent.
           The SAME structs and the SAME message inserted a row into an
           msftedit RICHEDIT50W control in the same run (length 0 -> 8),
           which is what makes the zero a fact about riched20 rather
           than about this code.

           MSDN's Requirements table is NOT what settles this, and must
           not be quoted as if it were: it says "Windows 8" for
           TABLECELLPARMS and "Windows Vista" for PARAFORMAT - and
           PARAFORMAT is Rich Edit 1.0, which this program uses through
           an NT4-era header on a Win98 target. Those tables describe
           the SDK the docs were built from. What carries real version
           information is the page BODY ("Rich Edit 2.0:", "earlier than
           Rich Edit 3.0", the per-version feature tables in "About Rich
           Edit Controls") - and past that, a probe.

           Nor would 3.0 help: its "simple tables" are, in its own
           description, "simulated by tabs" - which is exactly what this
           code already does. Real tables with cell borders start at 4.1
           (Msftedit.dll), and that is a different DLL, a different
           class, and far past the floor.

           So table rules, if ever wanted, come from box-drawing
           CHARACTERS (all measured encodable in GBK) or from painting
           over the control. */
        pf.dxStartIndent = 0;
        /* Hanging indent for a bullet: the first line starts at the
           margin, where the marker is, and dxOffset pushes every
           CONTINUATION line in to where the text begins. dxOffset is
           relative to the first line, so a positive value is the
           hanging shape - no negative dxStartIndent trick needed. */
        pf.dxOffset = (style & LZ_STYLE_BULLET)
                      ? (LONG)bullet_indent_twips(h, base_twips) : 0;
        if (style & LZ_STYLE_TABLE) {
            int colw = base_twips * 8;
            int avail = client_width_twips(h);
            if (avail > 0 && colw * LZ_TABLE_COLS > avail)
                colw = avail / LZ_TABLE_COLS;
            if (colw < 1) colw = 1;
            pf.cTabCount = LZ_TABLE_COLS;
            for (k = 0; k < LZ_TABLE_COLS; k++)
                pf.rgxTabs[k] = (LONG)(colw * (k + 1));
        } else {
            pf.cTabCount = 0;
        }
        SendMessage(h, EM_SETPARAFORMAT, 0, (LPARAM)&pf);
    }

    SendMessage(h, EM_REPLACESEL, FALSE, (LPARAM)buf);

    if (follow) SendMessage(h, WM_VSCROLL, SB_BOTTOM, 0);
}

/* Append `len` bytes of UTF-8 as ONE run with an explicit colour, no
 * bold/italic/code, and WITHOUT going through gui/stream.c's Markdown/
 * think scanner - speaker labels, clocks and system lines
 * are UI chrome this program wrote itself, never model output, so
 * there is nothing in them for that scanner to find and no reason to
 * pay for looking. append_run (the scanner's own sink) is not reused
 * here because its colour choice is hardwired to the think/plain pair
 * carried in its `style` bitmask; this needs a THIRD, independent axis
 * (which SPEAKER, or "system") the bitmask was never built to carry,
 * and mixing that into the scanner's own state machine would make it
 * responsible for two unrelated things.
 *
 * `utf8` is converted to GBK here exactly the way gui/stream.c's own
 * emit() does for the scanner's runs - same lz_gbk_from_utf8 call,
 * same "clamp after a possible truncation" guard - so this is, in
 * effect, that one call plus a CHARFORMAT set, with the scanner left
 * out. */
static void append_colored_line(const char *utf8, int len, COLORREF color) {
    HWND h = g.part[LZ_GUI_TRANSCRIPT];
    CHARFORMAT cf;
    CHARRANGE end;
    static char gbk[LZ_STREAM_RUN_MAX + 16];
    int n, follow;

    if (!h || len <= 0) return;
    n = lz_gbk_from_utf8(utf8, len, gbk, (int)sizeof gbk, NULL);
    if (n > (int)sizeof gbk - 1) n = (int)sizeof gbk - 1;
    if (n <= 0) return;

    follow = lz_scroll_at_end(h);

    end.cpMin = -1;
    end.cpMax = -1;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&end);

    memset(&cf, 0, sizeof cf);
    cf.cbSize = sizeof cf;
    /* CFM_SIZE here for the same reason append_run carries it, and the
       case is not hypothetical: a reply whose LAST run is a heading
       leaves the insertion point at the heading's size, and the very
       next thing written is one of these chrome lines. Without the bit
       the speaker label for the following turn comes out in heading
       type. */
    cf.dwMask = CFM_COLOR | CFM_BOLD | CFM_ITALIC | CFM_FACE | CFM_SIZE;
    cf.crTextColor = color;
    cf.dwEffects = 0;
    {
        int twips;
        ui_font_metrics(cf.szFaceName, (int)sizeof cf.szFaceName, &twips);
        cf.yHeight = twips;
    }
    SendMessage(h, EM_SETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    SendMessage(h, EM_REPLACESEL, FALSE, (LPARAM)gbk);

    if (follow) SendMessage(h, WM_VSCROLL, SB_BOTTOM, 0);
}

/* One line naming who is speaking, in that speaker's own colour, its
 * own line, optionally followed by a clock reading (user request: a
 * coloured "who and when" header above each turn). The
 * user's own sketch wrote it as two bracketed placeholders - speaker,
 * then time - on one line, followed by an indented line of bracketed
 * content underneath. The brackets are read as PLACEHOLDER notation,
 * not literal characters - the content line below is bracketed the
 * exact same way in that sketch, and content plainly should not carry
 * literal brackets, so the marks can only mean "a value goes here".
 * Read this way, the shape is "label, a space, HH:MM" on its own line,
 * with the
 * turn's own content starting on the NEXT line, indented (the two
 * leading spaces do_send and load_chat_from push right after calling
 * this, not part of this function - a header has no content of its
 * own to indent).
 *
 * The label is one of the EXISTING localized_strings.h speaker
 * strings (LZ_STR_SPEAKER_USER / LZ_STR_SPEAKER_ASSISTANT), not a new
 * "user/assistant" pair - those are already translated, and a second
 * name for the same role would fork the translation the first time
 * either one changed.
 *
 * `with_time` off is for load_chat_from's replay path: LZChatMsg
 * (src/chat.h) carries no timestamp - the engine layer has no display
 * concerns to begin with (iron law one) - and a message loaded back
 * from a file was not sent "now", so inventing a clock reading for it
 * would put a fact on screen that is simply not true. A live turn
 * (do_send) always passes 1: GetLocalTime is read fresh, right here,
 * at the moment the header is actually drawn. */
static void turn_header(LZStr speaker, COLORREF color, int with_time) {
    char line[128];
    const char *label = lz_str_utf8(speaker);
    int n = sprintf(line, "%s", label);
    if (with_time) {
        SYSTEMTIME t;
        GetLocalTime(&t);
        n += sprintf(line + n, " %02u:%02u", (unsigned)t.wHour,
                    (unsigned)t.wMinute);
    }
    n += sprintf(line + n, "\r\n");
    append_colored_line(line, n, color);
}

/* Feed generated UTF-8 to the transcript. Chunk boundaries are
 * arbitrary; gui/stream.c owns the reassembly. */
static void transcript_push(const char *utf8, int len) {
    lz_stream_push(&g.stream, utf8, len, append_run, NULL);
}

/* Generation finished: flush whatever the stream is still holding, or a
 * reply ending in a partial tag never appears at all. */
static void transcript_end(void) {
    lz_stream_end(&g.stream, append_run, NULL);
}

/* The two-space indent every turn's content sits behind (turn_header's
 * own shape). One function because there are three callers - do_send's
 * user turn, do_send's assistant turn and load_chat_from's replay - and
 * because WHICH WAY the spaces are written turns out to matter a great
 * deal.
 *
 * CHROME, not content: written straight into the control like the
 * speaker label above, NOT pushed through gui/stream.c. These two bytes
 * are this program's own layout, not something the model said, and the
 * scanner has no business seeing them - the same argument
 * append_colored_line's own comment already makes for the label and the
 * clock.
 *
 * Written directly (not via transcript_push) for a reason that matters:
 * transcript_push("  ", 2) would leave s->at_bol FALSE, and EVERY
 * block-level construct is recognised only at a line start. So the
 * first line of every turn could not be a heading, a bullet, a quote, a
 * rule, a table row or a fenced block - it would print its own marker
 * instead. That is the line a structured reply is most likely to open a
 * heading on, and this front end's checkpoint writes structured
 * replies. Nine gates covered the scanner and four covered the
 * rendering; none of them fed a turn the way do_send does. */
static void turn_indent(void) {
    append_colored_line("  ", 2, LZ_COLOR_TEXT);
}

/* Empty whatever has arrived since the last tick into the control. */
static void tokens_flush(void) {
    int n = g.tok_n;
    g.tok_n = 0;               /* cleared FIRST: transcript_push runs the
                                  scanner and the sink, and neither has
                                  any business seeing a buffer that is
                                  about to be re-entered */
    if (n > 0) transcript_push(g.tok_buf, n);
}

/* A batch of generated bytes has reached the UI thread.
 *
 * With the throttle on they are held for the next tick; with it off
 * (stream_ms = 0) they go straight through, unthrottled. Correctness
 * does not depend on
 * where the boundaries fall - gui/stream.c is chunk-independent by
 * construction - so merging
 * arrivals is free of meaning, only of cost.
 *
 * A batch too big for the remaining room does not get truncated and
 * does not grow the buffer: the buffer is flushed and, if the batch
 * still does not fit, it is pushed on its own. Buffering is an
 * optimisation, and an optimisation that can drop output is not one. */
static void tokens_arrived(const char *utf8, int n) {
    if (n <= 0) return;
    /* Refresh the throughput cell's token count from the
       worker's own counter, which counts SINK CALLS (one per sampled
       token) - not the byte length here, which is what the transcript
       needs and would overstate a CJK-heavy reply's token count by up
       to 3x. The worker writes it concurrently while a job runs, so a
       mid-generation read is a display-layer race (a 32-bit aligned
       long read is atomic on x86, so the number is a moment old, never
       torn); the exact final count is read after lz_worker_join in
       finish_job. */
    g.tok_gen = (int)lz_worker_tokens_sent();
    if (g.tok_ms <= 0) { transcript_push(utf8, n); return; }
    if (g.tok_n + n > LZ_TOK_BUF) tokens_flush();
    if (n > LZ_TOK_BUF) { transcript_push(utf8, n); return; }
    memcpy(g.tok_buf + g.tok_n, utf8, (size_t)n);
    g.tok_n += n;
}

/* Paint the prefill indicator from whatever the counters say now.
 * Called from ui_tick and from WM_APP_PREFILL - the tick for the
 * regular refresh, the message so the first update lands as prefill
 * STARTS rather than up to one tick later. */
static void prefill_paint_tick(void) {
    /* Whether the well was last painted with something in it. Only the
       fallback strip needs it: emptying that one means invalidating it,
       and this runs on every tick of the whole job, so without the
       transition test a generate with no prefill would repaint the
       strip every LZ_LAMP_BLINK for nothing. The comctl32 control takes
       a redundant SETPOS without repainting. */
    static int shown;

    if (!prefill_active()) {
        /* Prefill over: the well EMPTIES, it does not disappear. The
           segment is part of the strip's layout - a well that came and
           went would re-flow the bar underneath the reader, and a well
           left full would read as a job still running. */
        if (g.progress)
            SendMessage(g.progress, LZ_PBM_SETPOS, (WPARAM)0, 0);
        else if (shown && g.part[LZ_GUI_STATUS])
            InvalidateRect(g.part[LZ_GUI_STATUS], NULL, FALSE);
        shown = 0;
        return;
    }
    shown = 1;
    {
        char pf[96];
        sprintf(pf, "%s %d/%d", lz_str_utf8(LZ_STR_STATE_PREFILL),
                g_pf_done, g_pf_total);
        set_status(pf);
    }
    /* Both strips, one condition. The comctl32 bar is a control to
       drive; the fallback paints itself and only needs to be told the
       numbers moved. */
    if (g.progress) {
        SendMessage(g.progress, LZ_PBM_SETRANGE, 0, MAKELPARAM(0, 100));
        SendMessage(g.progress, LZ_PBM_SETPOS,
                    (WPARAM)((g_pf_done * 100) / g_pf_total), 0);
    } else if (g.part[LZ_GUI_STATUS]) {
        InvalidateRect(g.part[LZ_GUI_STATUS], NULL, FALSE);
    }
}

/* The two lamps, from the state the program is already in rather than
   from a copy of it. Left = the model: dark when none is open, green
   when one is ready, red when the last load failed. Right = activity:
   amber while a job runs, dark otherwise, and it alternates on a timer
   because a lamp that is merely ON does not say "working" - WinZip's
   flickered for the same reason.
   Safe before the lamps exist and on the STATIC fallback, where they
   never do. */


static void gui_prefill_progress(int done, int total, void *ctx) {
    (void)ctx;
    /* Every segment opens with done == 0 (gen_prefill_raw reports before
       its first slice, on purpose). A second one means the previous
       segment's total is now behind us. */
    if (done == 0 && g_pf_seen) g_pf_base = g_pf_total;
    g_pf_seen  = 1;
    g_pf_total = g_pf_base + total;
    g_pf_done  = g_pf_base + done;
    /* Posted, not drawn: this runs on the WORKER thread. The UI thread
       redraws when it handles the message, which is what makes the
       indicator appear as prefill STARTS rather than at the next 400 ms
       tick - on a fast machine the whole prefill fits inside one tick,
       so the tick alone never showed it. */
    if (g.main) PostMessage(g.main, WM_APP_PREFILL, 0, 0);
}

/* ---- the window's single periodic tick (see LZ_UI_TIMER) ---- */

static int    g_ui_timer_ms;      /* 0 = not armed */
static double g_ui_last_flush;

/* The tick's period: fine enough for the fastest consumer, which is the
   token flush when stream_ms asks for one shorter than a lamp blink.
   Everything slower is gated on the clock inside ui_tick, so a long
   stream_ms does not slow the lamps down and a short one does not make
   them blink faster. */
static int ui_tick_ms(void) {
    int ms = LZ_LAMP_BLINK;
    if (g.tok_ms > 0 && g.tok_ms < ms) ms = g.tok_ms;
    return ms;
}

/* Arm or disarm the one timer from the state that needs it. Called
   wherever that state changes - both ends of a job, and once at startup
   for the demo ramp - so no caller has to remember a matching
   KillTimer. Re-arms when the period changes, which is what makes
   stream_ms editable without a restart. */
static void ui_timer_sync(HWND hwnd) {
    int want = (g.job_kind != JOB_NONE || g_dbg_prefill_ms > 0)
               ? ui_tick_ms() : 0;
    if (want == g_ui_timer_ms) return;
    if (g_ui_timer_ms) KillTimer(hwnd, LZ_UI_TIMER);
    if (want) SetTimer(hwnd, LZ_UI_TIMER, (UINT)want, NULL);
    g_ui_timer_ms = want;
}

static void set_lamps(void) {
    int left, right;
    if (!g.lamp[0] || !g.lamp[1]) return;
    if (g.job_kind == JOB_LOAD)          left = LZ_LAMP_OFF;
    else if (lz_gui_model_ready(&g.mdl)) left = LZ_LAMP_READY;
    else if (g.load_failed)              left = LZ_LAMP_ERROR;
    else                                 left = LZ_LAMP_OFF;
    right = (g.job_kind != JOB_NONE && !g.lamp_phase) ? LZ_LAMP_BUSY
                                                      : LZ_LAMP_OFF;
    SendMessage(g.lamp[0], STM_SETIMAGE, IMAGE_BITMAP,
                (LPARAM)g.lamp_bmp[left]);
    SendMessage(g.lamp[1], STM_SETIMAGE, IMAGE_BITMAP,
                (LPARAM)g.lamp_bmp[right]);
}

/* Everything this window does on a clock, in the order it has to happen
 * and with the phase decided ONCE - see LZ_UI_TIMER.
 *
 * Every sub-rate is a deadline against lz_time_ms, not a count of
 * ticks: the tick's period follows stream_ms (ui_tick_ms), so counting
 * would make the lamp rate depend on a setting that has nothing to do
 * with it. On the target that clock moves in ~55 ms steps, which is
 * finer than any deadline here. */
static void ui_tick(void) {
    double now = lz_time_ms();

    /* 1. Buffered tokens. stream_ms == 0 means the sink pushes straight
          to the control and there is nothing held back to flush. */
    if (g.job_kind == JOB_GENERATE && g.tok_ms > 0 &&
        now - g_ui_last_flush >= (double)g.tok_ms) {
        tokens_flush();
        g_ui_last_flush = now;
    }

    /* 2. The activity lamp. Phase off the wall clock so it blinks at
          LZ_LAMP_BLINK whatever the tick period is. */
    {
        int ph = g.job_kind != JOB_NONE
                 ? (int)(((long)(now / LZ_LAMP_BLINK)) & 1) : 0;
        if (ph != g.lamp_phase) { g.lamp_phase = ph; set_lamps(); }
    }

    /* 3. The demo ramp, and only with no real job: a generation owns
          these counters and the ramp must not overwrite what the engine
          reports. t0 is zeroed while a job runs so the sweep restarts
          from empty afterwards rather than resuming mid-way. */
    if (g_dbg_prefill_ms > 0) {
        if (g.job_kind != JOB_NONE) {
            g_dbg_prefill_t0 = 0.0;
        } else {
            double dt;
            if (g_dbg_prefill_t0 <= 0.0) g_dbg_prefill_t0 = now;
            dt = now - g_dbg_prefill_t0;
            if (dt >= (double)g_dbg_prefill_ms) {
                g_dbg_prefill_t0 = now;   /* loop, so it can be watched
                                             more than once */
                dt = 0.0;
            }
            g_pf_base  = 0;
            g_pf_total = 1000;
            g_pf_done  = (int)((dt * 1000.0) / (double)g_dbg_prefill_ms);
            if (g_pf_done >= g_pf_total) g_pf_done = g_pf_total - 1;
        }
    }

    /* 4. The prefill -> generation transition, seen once. The
          throughput cell's denominator starts here, not at the job's
          start - see LZGuiState.gen_start_ms. */
    {
        static int was_prefill;
        int now_prefill = prefill_active();
        if (was_prefill && !now_prefill && g.job_kind == JOB_GENERATE)
            g.gen_start_ms = now;
        was_prefill = now_prefill;
    }

    /* 5. THE STATUS LINE, one decision, one writer. prefill_paint_tick
          empties the well by itself when there is no prefill, so the
          resting/idle case needs no branch here. */
    prefill_paint_tick();
    if (g.job_kind == JOB_GENERATE && !prefill_active())
        set_status(lz_str_utf8(LZ_STR_STATE_GENERATING));
}

/* (Re)load the lamp artwork and repaint the pair.
 *
 * One place, because it runs twice: once when the controls are built,
 * and again on WM_SYSCOLORCHANGE - lz_mapped_bitmap bakes the button
 * face into the bitmap as it loads, so a scheme change makes the old
 * handles wrong in exactly the way it makes a cached brush wrong. */
static void lamps_reload(void) {
    int i;
    for (i = 0; i < LZ_LAMP_KINDS; i++) {
        if (g.lamp_bmp[i]) { DeleteObject(g.lamp_bmp[i]); g.lamp_bmp[i] = NULL; }
    }
    g.lamp_bmp[LZ_LAMP_OFF]   = lz_mapped_bitmap(g.inst, IDB_LAMP_OFF);
    g.lamp_bmp[LZ_LAMP_READY] = lz_mapped_bitmap(g.inst, IDB_LAMP_READY);
    g.lamp_bmp[LZ_LAMP_BUSY]  = lz_mapped_bitmap(g.inst, IDB_LAMP_BUSY);
    g.lamp_bmp[LZ_LAMP_ERROR] = lz_mapped_bitmap(g.inst, IDB_LAMP_ERROR);
    set_lamps();
}

/* ------------------------------------------------- splitter tracking
 *
 * The Win9x splitter, and the reason it never flickered: while the
 * button is down NOTHING is laid out. An inverted bar is XORed onto the
 * window, moved by erasing it (XOR again) and drawing it at the new
 * place, and only when the button comes up does the layout run once.
 *
 * Live-resizing the two controls on every WM_MOUSEMOVE is the other
 * way, and on the target it is the flickering way: a RichEdit holding a
 * conversation repaints its whole client area for each move, and the
 * gap between the erase and the repaint is the flash.
 */
/* The 50% dither of the era. A solid inverted band reads as a black
   bar rather than as something being moved; every Win9x splitter and
   the outline of a window being dragged used this checkerboard. Eight
   rows of alternating bits is the pattern GDI itself uses for a
   halftone brush. */
static HBRUSH dither_brush(void) {
    static HBRUSH b;
    static const unsigned short BITS[8] = {
        0x5555, 0xAAAA, 0x5555, 0xAAAA, 0x5555, 0xAAAA, 0x5555, 0xAAAA
    };
    if (!b) {
        HBITMAP bm = CreateBitmap(8, 8, 1, 1, BITS);
        if (bm) { b = CreatePatternBrush(bm); DeleteObject(bm); }
        if (!b) b = (HBRUSH)GetStockObject(BLACK_BRUSH);
    }
    return b;
}

/* A solid white brush for the input EDIT's own WM_CTLCOLOREDIT reply
 * - created once and cached, the same "process-lifetime
 * singleton, never freed until the process ends" shape dither_brush
 * just above already uses. Not created fresh inside the message
 * handler: WM_CTLCOLOREDIT arrives on every redraw of that control
 * (every keystroke, every caret blink), and a fresh CreateSolidBrush
 * there would leak one GDI object per redraw rather than one for the
 * whole run. */
static HBRUSH input_bkg_brush(void) {
    static HBRUSH b;
    if (!b) b = CreateSolidBrush(RGB(0xFF, 0xFF, 0xFF));
    return b;
}

/* The sunken edge on the transcript and the input box is two pixels
   wide. The bar is inset by that much at each end so it never lands ON
   a border: XOR against a dark border comes back light, so the last two
   pixels of each end inverted the wrong way and the bar read as though
   it were mis-measured. It is not - the band's rectangle is the
   transcript's rectangle by construction, and its right edge coincides
   with the Send button's - but the ends looked wrong, and looking wrong
   is the whole job of a tracking bar. */
#define LZ_DRAG_INSET 2

static void drag_bar(HWND hwnd, int y) {
    HDC dc = GetDC(hwnd);
    if (!dc) return;
    /* PATINVERT: XOR, so the same call undoes it. That is what makes the
       tracking flicker-free - nothing is erased and repainted, one
       rectangle is inverted and then inverted back. */
    {
        HGDIOBJ old = SelectObject(dc, dither_brush());
        PatBlt(dc, g.split.x + LZ_DRAG_INSET, y,
               g.split.w - 2 * LZ_DRAG_INSET, g.split.h, PATINVERT);
        SelectObject(dc, old);
    }
    ReleaseDC(hwnd, dc);
}

/* What the sidebar shows below the model/status line, as one of
 * layout.h's LZ_GUI_PANEL_* constants - decided from the currently
 * loaded model, nowhere else, so every lz_gui_layout call site and
 * side_panel_sync agree with each other by construction rather than
 * by each recomputing "is a model loaded, does it have MoE" the same
 * way independently. */
static int side_panel_mode(void) {
    if (!lz_gui_model_ready(&g.mdl)) return LZ_GUI_PANEL_NONE;
    return g.mdl.model.config.num_experts > 0 ? LZ_GUI_PANEL_LAMPS
                                              : LZ_GUI_PANEL_PLAIN;
}

/* Turn a bar position into the input height it means, clamped by the
   layout so the tracking bar cannot be dragged somewhere the layout
   would refuse to follow. */
static int drag_to_input_h(HWND hwnd, int bar_y) {
    RECT rc;
    LZRect r[LZ_GUI_PART_COUNT];
    int delta, bottom, want;
    GetClientRect(hwnd, &rc);
    /* Ask the layout rather than restating its arithmetic here. The gap
       between the band and the box, and the box's bottom edge, are both
       the layout's business; a second copy of either would drift the
       first time a margin changed. */
    lz_gui_layout(rc.right - rc.left, rc.bottom - rc.top, g.input_h,
                 g.status_h, side_panel_mode(), r);
    delta = r[LZ_GUI_INPUT].y - r[LZ_GUI_SPLIT].y;
    bottom = r[LZ_GUI_INPUT].y + r[LZ_GUI_INPUT].h;
    want = bottom - (bar_y + delta);
    lz_gui_layout(rc.right - rc.left, rc.bottom - rc.top, want,
                 g.status_h, side_panel_mode(), r);
    return r[LZ_GUI_INPUT].h;
}

/* Grey or ungrey a command wherever it appears.
 *
 * One function instead of EnableWindow on a button, because a command
 * now has up to two faces: a toolbar button and a menu item. They must
 * agree - a greyed toolbar button beside a live menu entry for the same
 * thing is how a user learns not to trust either.
 *
 * It also carries the NT 3.51 case. There is no comctl32 there, so
 * g.part[LZ_GUI_TOOLBAR] is NULL and only the menu half runs; that is
 * the whole degradation, and it is why no command may live on the
 * toolbar alone. */
static void cmd_enable(UINT id, int on) {
    HMENU bar;
    /* Three faces, not two. Send lives on a real BUTTON beside the
       input box, so a command with a control of its own is enabled
       where that control is; the toolbar and the menu are handled
       below for every command. */
    if (id == ID_SEND && g.part[LZ_GUI_SEND])
        EnableWindow(g.part[LZ_GUI_SEND], on ? TRUE : FALSE);
    if (g.part[LZ_GUI_TOOLBAR])
        lz_gui_toolbar_enable(g.part[LZ_GUI_TOOLBAR], (int)id, on);
    if (!g.main) return;
    bar = GetMenu(g.main);
    if (bar)
        EnableMenuItem(bar, id,
                       MF_BYCOMMAND | (UINT)(on ? MF_ENABLED : MF_GRAYED));
}

/* The read half of cmd_enable: is this command id allowed to run right
 * now? Used once, at the top of WM_COMMAND (wndproc, below), so that
 * every way a command can arrive - a live click (already stopped by the
 * disabled control or the greyed menu item itself), an accelerator
 * (which is NOT stopped by a greyed menu item - Windows fires the
 * WM_COMMAND anyway; see IDM_STOP_GEN's Esc case), or a synthetic
 * PostMessage like input_subclass's Enter key - is held to the exact
 * state cmd_enable last drew. Without this read-back, cmd_enable's
 * state would only change how three controls LOOKED; nothing would read
 * it back before acting, and input_subclass's PostMessage could reach
 * do_send no matter what the Send button showed.
 *
 * ID_SEND has a real control, so that control is the source of truth -
 * the same one cmd_enable's own EnableWindow call sets. Every other id
 * this file's WM_COMMAND switch handles always carries a menu item
 * ("no command lives on the toolbar alone"
 * rule; ID_STOP is the one exception, and it is dead - see its own enum
 * comment), so the menu's grey state is authoritative for those, and
 * doubles as the right answer for an unpopulated MRU slot: it is not
 * in the menu at all, which reads as disabled here exactly as it
 * should. */
static int cmd_is_enabled(UINT id) {
    HMENU bar;
    UINT st;

    if (id == ID_SEND)
        return g.part[LZ_GUI_SEND] ?
               IsWindowEnabled(g.part[LZ_GUI_SEND]) != 0 : 1;
    if (!g.main) return 1;
    bar = GetMenu(g.main);
    if (!bar) return 1;
    st = GetMenuState(bar, id, MF_BYCOMMAND);
    return st != (UINT)-1 && !(st & MF_GRAYED);
}

/* ----------------------------------------------------- generation */

static int start_job(HWND hwnd, LZWorkerJob job, void *ud, int kind) {
    g.done_seen = 0;
    g.done_rc = 0;
    g.inspect_seen = 0;
    g.job_kind = kind;
    /* &g.sess.ins only for JOB_GENERATE - a model load has no
       candidate list to report, and worker.c's own NULL check on ins
       is what makes passing it a genuine no-cost skip there, not a
       write into memory nothing is reading. */
    if (lz_worker_start(hwnd, job, ud,
                        kind == JOB_GENERATE ? &g.sess.ins : NULL)) {
        g.job_kind = JOB_NONE;
        return 1;
    }
    /* A generate job's throughput cell starts counting from
       the moment the job STARTS, not from the first token - a slow
       prompt or a cold cache spends time before any token, and that time
       is part of the answer. For a LOAD there is no such cell, and
       tok_live is cleared so set_status falls through to the passed
       text - a load starting must also retire the PREVIOUS generate's
       final reading, or the rate from the last turn would keep covering
       the bar across a model swap. Recorded before the set_status
       below, which now reads tok_live. */
    g.tok_live = (kind == JOB_GENERATE);
    if (kind == JOB_GENERATE) {
        g.tok_gen = 0;
        g.tok_start_ms = g.gen_start_ms = lz_time_ms();
    }
    set_status(lz_str_utf8(kind == JOB_LOAD ? LZ_STR_STATE_LOADING
                                               : LZ_STR_STATE_GENERATING));
    /* Loading is not interruptible - lz_read_weights takes no cont
       callback - so offering a live Stop button during it would be a
       button that does nothing. */
    cmd_enable(IDM_STOP_GEN, kind == JOB_GENERATE);
    /* Busy is one of rollback_ready's own conditions, so this greys all
       three - rolling the conversation back underneath a job that is
       still writing into it would leave the transcript and the history
       describing different conversations. */
    rollback_sync();
    g.lamp_phase = 0;
    set_lamps();
    /* This job's prefill starts from nothing, whatever the last one
       left. */
    g_pf_base = 0;
    g_pf_seen = 0;
    g_ui_last_flush = lz_time_ms();
    /* One timer, armed from the state that needs it. A job running is
       one of the two things that need it - the demo ramp is the other,
       and ui_timer_sync is the only place that decides. */
    ui_timer_sync(hwnd);
    return 0;
}

/* Everything a model load needs set up before the job is handed to the
 * worker, and NOT the start_job call - split there so the selftest can
 * run this half without a thread and without a directory that has to
 * exist. Both callers go through start_model_load below; neither writes
 * these three lines itself.
 *
 * One function because both callers share the same three lines, and
 * seq_want is a line that is invisible when forgotten: a load with a
 * stale seq_want allocates the previous size, the status bar reports
 * the previous size, and everything is self-consistent and wrong. Same
 * shape as lz_gui_session_begin calling regen instead of repeating its
 * body. */
static void model_load_prepare(const char *dir) {
    strncpy(g.mdl.dir, dir, sizeof g.mdl.dir - 1);
    g.mdl.dir[sizeof g.mdl.dir - 1] = '\0';
    /* The setting, not the clamp: the clamp needs cfg->seq_len, which
       is not knowable until lz_open has run. lz_gui_model_load_job
       applies it on the far side. */
    g.mdl.seq_want = g.set.ctx;
    lz_gui_session_prefix_drop(&g.sess);  /* replacing the model */
}

static void start_model_load(HWND hwnd, const char *dir) {
    model_load_prepare(dir);
    start_job(hwnd, lz_gui_model_load_job, &g.mdl, JOB_LOAD);
}

/* Defined below relayout(), which it calls; forward-declared here
   because finish_job (immediately below) is the only caller and comes
   first in the file. */
static void side_panel_sync(HWND hwnd);

static void finish_job(HWND hwnd, int rc) {
    int kind = g.job_kind;
    /* BEFORE transcript_end, and that order is the whole contract: the
       last tick can easily land after the final token, so whatever the
       throttle is still holding has to go through the scanner while the
       scanner is still in this turn's state. Ending first would flush
       the held tail, reset the stream, and only then hand it the last
       few bytes of the reply - which would come out unstyled, after the
       reset, or not at all. */
    tokens_flush();
    /* The stream may still be holding a partial tag; without this a
       reply that ends mid-tag is never shown at all. */
    transcript_end();
    lz_worker_join();
    /* Spec 3.1: NO read-back of lz_worker_tokens_sent() here. The
       token count for the status cell comes from tokens_arrived, which
       reads the worker's counter on every WM_APP_TOKENS and therefore
       always carries the FINAL count by the time the last one is
       dispatched - and every token the worker emits posts one
       WM_APP_TOKENS before it posts GEN_DONE, so by the time this
       handler runs the last tokens_arrived has already set g.tok_gen.
       A defensive read-back here is dead: a mutation test proved it
       does nothing - deleting it changes nothing, the counter's value
       already being correct. The selftest's
       "g.tok_gen == 11" check pins the chain that IS live - worker
       counts, tokens_arrived carries it, the status cell reads it. */
    /* Retire the prefill counters with the job that produced them.
       A run that finishes normally ends on done == total and the tick
       empties the well by itself; a run that was STOPPED part-way ends
       on done < total, and with the timer gone nothing ever comes back
       to clear it. The numbers would then sit there - the fallback
       strip repaints from these globals on any invalidate, and the next
       job's first tick would show the previous job's percentage over a
       turn that may have no prefill at all. Cleared through
       prefill_paint_tick rather than by hand so the emptying stays in
       the one function that knows how to do it for both strips. */
    g_pf_done = g_pf_total = g_pf_base = g_pf_seen = 0;
    cmd_enable(IDM_STOP_GEN, 0);
    g.done_rc = rc;
    g.done_seen = 1;
    g.job_kind = JOB_NONE;
    /* AFTER job_kind, which is half of what ui_timer_sync reads - the
       tick stops here unless the demo ramp still wants it. Then the
       repaint, so nothing can put the retired counters back. */
    ui_timer_sync(hwnd);
    prefill_paint_tick();

    if (kind == JOB_LOAD) {
        /* What the window IS has changed either way: a failed load
           leaves nothing open, so the resting text has to follow even on
           failure - otherwise the status line still claims a model. */
        int ok = (rc == 0) && lz_gui_model_ready(&g.mdl);
        g.load_failed = !ok;
        /* The inference inspector's side panel: this is
           the ONE place "is a model loaded, does it have MoE" changes
           from the UI's own point of view - a successful load builds
           the candidate list (and the lamp grid, if the model has
           MoE), a failed one or a reload tears down whatever the
           PREVIOUS model left, and side_panel_sync's own
           lz_gui_model_ready check is what tells those two apart, so
           `ok` is not threaded through as a separate argument. */
        side_panel_sync(hwnd);
        /* UTF-8, matching set_idle_status's contract: g.idle_status is
           now UTF-8 everywhere, and set_status converts to GBK. Using
           lz_str_display here (GBK) would double-convert through
           set_status and turn every Chinese state string into '?'. */
        const char *rest = lz_str_utf8(ok ? LZ_STR_STATE_READY
                                          : LZ_STR_STATE_NO_MODEL);
        strncpy(g.idle_status, rest, sizeof g.idle_status - 1);
        g.idle_status[sizeof g.idle_status - 1] = '\0';
        /* Unconditional, not "ok": a load that just FAILED still leaves
           Send meaningful - /load and /help do not need the model that
           did not come up. See create_children's own cmd_enable(ID_SEND,
           ...) call for the fuller reasoning; this call exists mainly so
           a future condition that SHOULD grey Send (busy, say) has an
           obvious place to be added without hunting for it. */
        cmd_enable(ID_SEND, 1);
        if (ok) {
            /* Sized off &g.mdl.model, so it has to be rebuilt
               for whichever model just finished loading -
               prefix_teardown (called before this load started, at
               both start_job(..., JOB_LOAD) call sites) already
               dropped whatever the PREVIOUS model's cache held.
               Failure here is not a load failure: pc_ready just stays
               0, which is exactly what makes lz_gui_session_job's own
               prefix branch never fire - the same leniency the
               prepare-failure fallback has, one level earlier. */
            char pcerr[256];
            lz_gui_session_prefix_arm(&g.sess, &g.mdl, pcerr,
                                      (int)sizeof pcerr);
            /* Remembered here, not in open_model_dir: only a load that
               actually SUCCEEDED belongs on the recent list - pushing it
               at the click would put a directory that just failed to
               load right back on the menu. */
            lz_mru_push(&g.mru, g.mdl.dir);
            build_menu_bar(hwnd);
            /* The directory name came from a shell dialog, so it is
               ANSI - GBK on the target. The transcript takes UTF-8 and
               converts on the way out, so handing it those bytes raw
               would mojibake any model path with a Chinese character in
               it. This is the two-forms trap in localized_strings.h,
               met in the wild. */
            const char *name = lz_gui_model_name(&g.mdl);
            char nameu[300], line[512];
            lz_gbk_to_utf8(name, (int)strlen(name), nameu, (int)sizeof nameu,
                           NULL);
            sprintf(line, lz_str_utf8(LZ_STR_SYS_MODEL_LOADED), nameu,
                    (int)(g.mdl.state.bytes_alloc / (1024 * 1024)));
            /* Through sys_line_fmt, not a direct push (user report -
               see that function's own comment): it is a system message
               and so gets the system line's colour and blank line, the
               same as every other system message. */
            sys_line_fmt(line, (int)strlen(line));
            /* Sidebar model info: name + KV footprint, one line. The
               status line under it is refreshed by set_status.
               `nameu` is UTF-8 (the transcript path needs it that way);
               the sidebar is an ANSI control, so SetWindowTextA would
               draw the UTF-8 bytes as GBK and mojibake any model path
               with a Chinese character. Convert back to GBK here, the
               mirror of the gbk_to_utf8 above - the two-forms trap in
               localized_strings.h, met twice in the wild. */
            sprintf(g.side_model, lz_str_utf8(LZ_STR_SIDE_MODEL), nameu,
                    (int)(g.mdl.state.bytes_alloc / (1024 * 1024)));
            {
                char side_gbk[sizeof g.side_model];
                lz_gbk_from_utf8(g.side_model, (int)strlen(g.side_model),
                               side_gbk, (int)sizeof side_gbk, NULL);
                memcpy(g.side_model, side_gbk, sizeof side_gbk);
            }
            set_status(g.idle_status);   /* repaint the sidebar too */
            /* The model segment of the title bar changed with the load. */
            push_caption();
        } else {
            /* A FAILED load still changed what the window is: the job
               unloaded whatever was open before it started reading, so
               lz_gui_model_ready is false now. Both of these were
               written only on the success path, and both survive the
               failure - the sidebar kept naming a model that is gone,
               with its memory figure, directly above a status line
               saying the load failed, and the title bar kept it too.
               Cleared here so the two halves of the window agree. */
            g.side_model[0] = '\0';
            set_status(g.idle_status);
            push_caption();
        }
    } else if (kind == JOB_GENERATE) {
        static char err[1024];
        /* Backfill happens on EVERY ending, stop included: a stopped
           reply is still on screen and still part of the conversation,
           and a history that omits it renders a different prompt from
           what the user is looking at.
           TWO newlines, not one - the reply's own final
           line, then a blank line before whatever comes next (the next
           turn's header, or LZ_STR_SYS_CTX_TRIMMED below), the same
           trailing-blank-line shape transcript_line's own comment
           gives user/replayed turn content. */
        transcript_push("\r\n\r\n", 4);
        transcript_end();
        lz_gui_session_end(&g.sess, err, (int)sizeof err);
        /* Said here rather than by the worker: anything the worker
           writes to the sink lands in the reply, and from there in
           history, and the model would read its own status line back on
           the next turn. */
        if (lz_gui_session_trimmed(&g.sess) > 0)
            sys_line(LZ_STR_SYS_CTX_TRIMMED);
        /* The user pressed Stop: say so. The engine records the finish
           reason in opts.out_finish (generate.c's own write), and
           LZ_STR_SYS_GEN_STOPPED existed in the string table for
           exactly this but was never read - a stopped reply otherwise
           looked identical to one that ended on its own. */
        if (g.sess.opts.out_finish == LZ_FINISH_CANCELLED)
            sys_line(LZ_STR_SYS_GEN_STOPPED);
    }

    /* AFTER the branch, not before it. The model lamp's third state is
       "the last load failed", and g.load_failed is written inside the
       JOB_LOAD branch above - lighting the lamps first meant a failed
       load painted them from the PREVIOUS load's verdict, and a lamp is
       not repainted on a timer, so it stayed wrong until some unrelated
       event happened to call this again. */
    set_lamps();

    /* The throughput cell is LIVE ONLY while a generate job runs. It
       retires the moment the job ends and the resting text comes back:
       a status line that sticks at "N tok, X tok/s" after generation
       ends reads as hung, not as informative, and the status line must
       return to "ready" once the model finishes. */
    if (rc != 0) {
        g.tok_live = 0;
        const char *msg = lz_worker_error();
        set_status(msg && msg[0] ? msg
                                 : lz_str_display(LZ_STR_ERR_TITLE));
        /* A modal box owned by a window nobody can see has no owner
           context and no way to be dismissed, so the status line carries
           the failure on its own in that case. */
        if (IsWindowVisible(hwnd) && msg && msg[0]) {
            char msg_gbk[512];
            lz_gbk_from_utf8(msg, (int)strlen(msg),
                             msg_gbk, (int)sizeof msg_gbk, NULL);
            MessageBoxA(hwnd, msg_gbk, lz_str_display(LZ_STR_ERR_TITLE),
                        MB_ICONHAND | MB_OK);
        }
    } else {
        g.tok_live = 0;
        set_status(g.idle_status);
    }
    /* Covers both branches above in one call, whatever rc was -
       a load changes whether there is a model to count with at all, a
       generate turn changes what history there is to count. Placed
       after the status line above rather than inside either kind's own
       branch so a failed load (which still changes "is there a model"
       from whatever it was) is not missed. */
    update_ctx_cell();
    /* Both halves of rollback_ready: the job is over, and a
       JOB_GENERATE has pushed its reply into history. After the
       backfill above, deliberately - reading history before
       lz_gui_session_end has pushed would grey the three commands on
       the very turn they become meaningful. */
    rollback_sync();
    /* Tell an unattended window it has finished. */
    if (GetForegroundWindow() != hwnd) FlashWindow(hwnd, TRUE);
}

/* Where the end of the transcript is, in the units EM_EXSETSEL speaks.
 *
 * Asked of the control rather than computed, and asked in CHARACTERS
 * rather than bytes - see g.turn_cp's own comment for what
 * GetWindowTextLengthA would have got wrong. The (-1, -1) selection is
 * the same "collapse to the end" append_run already uses, so this
 * leaves the control in exactly the state the next append wants; the
 * EM_EXGETSEL right after is what turns that into a number.
 *
 * A side effect on the selection is not a side effect worth avoiding
 * here: both callers are about to write at the end anyway. */
static LONG transcript_cp_end(void) {
    HWND h = g.part[LZ_GUI_TRANSCRIPT];
    CHARRANGE cr;
    if (!h) return 0;
    cr.cpMin = -1;
    cr.cpMax = -1;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    SendMessage(h, EM_EXGETSEL, 0, (LPARAM)&cr);
    return cr.cpMin;
}

/* Delete everything from `cp` to the end.
 *
 * EM_REPLACESEL with an empty string, fUndo FALSE - the same call
 * append_run makes to put text in, which is what keeps this symmetric
 * with how the text got there. NOT a redraw of the whole transcript
 * from history: that would re-run every Markdown run through the
 * scanner and, worse, would have to invent clock readings for turn
 * headers that were drawn live (turn_header's own with_time comment). */
static void transcript_truncate(LONG cp) {
    HWND h = g.part[LZ_GUI_TRANSCRIPT];
    CHARRANGE cr;
    if (!h || cp < 0) return;
    cr.cpMin = cp;
    cr.cpMax = -1;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    SendMessage(h, EM_REPLACESEL, FALSE, (LPARAM)"");
}

/* Reset the recorded turn boundaries - there is no exchange on screen. */
static void turn_cp_forget(void) {
    g.turn_cp[0] = -1;
    g.turn_cp[1] = -1;
}

static void transcript_clear(void) {
    if (g.part[LZ_GUI_TRANSCRIPT])
        SetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], "");
    lz_stream_init(&g.stream);
    /* The positions were positions IN the text that just went away. */
    turn_cp_forget();
    /* DROPPED, not flushed: the window this text was going into has
       just been emptied, so pushing it now would put the tail of the
       previous conversation at the top of the new one. Same reasoning
       as lz_stream_init on the line above - per-turn state does not
       survive a clear. */
    g.tok_n = 0;
}

static void apply_font(HWND h) {
    SendMessage(h, WM_SETFONT, (WPARAM)lz_ui_font(),
                MAKELPARAM(TRUE, 0));
}

/* The status bar's REAL height, not LZ_GUI_STATUS_H - see g.status_h's
 * own field comment and gui/layout.h for why a guessed constant is
 * wrong on comctl32 v5. Shared by create_children (right after the
 * status bar's font is first set) and apply_language (right after a
 * language switch re-sets it) - the same measurement is needed both
 * times, for the same reason: comctl32 computes a status bar's height
 * from whatever font it CURRENTLY has, so the answer can change
 * whenever the font does, not only at creation.
 *
 * MUST be called AFTER the status bar already has its real, current
 * font - not right after CreateWindowExA, and not before apply_font
 * has run following a language switch. Measuring early once already
 * cost a debugging round: asked immediately after
 * creation, before WM_SETFONT ever reached the control, and got back
 * 26 - neither LZ_GUI_STATUS_H (22) nor the real post-font answer (20).
 *
 * One resize is enough to make comctl32 reveal what it actually gives,
 * whether or not that matches what was asked; a plain STATIC (the 3.51
 * floor, no comctl32) does not auto-size at all and simply reports
 * back what it is told, so this is one code path, not one per control
 * type. LZ_GUI_STATUS_H is the probe height, not because it is
 * trusted but because SOME height has to be asked for before there is
 * anything to measure - the number that comes back out is what
 * matters, not the number that went in. */
static void measure_status_h(void) {
    if (!g.part[LZ_GUI_STATUS]) return;
    {
        RECT r;
        MoveWindow(g.part[LZ_GUI_STATUS], 0, 0, 100, LZ_GUI_STATUS_H, TRUE);
        GetWindowRect(g.part[LZ_GUI_STATUS], &r);
        g.status_h = r.bottom - r.top;
        if (g.status_h <= 0) g.status_h = LZ_GUI_STATUS_H;
    }
}

/* The tool strip, built from the string table.
 *
 * A function rather than a block inside create_children because the
 * labels are LANGUAGE, and a toolbar's strings go in at creation
 * (TB_ADDSTRING) - there is no "retitle button 3". Switching language
 * therefore destroys this control and builds a new one, which is only
 * safe if there is exactly one place that knows how.
 *
 * Every menu command except Exit and About, which is the rule a Win9x
 * tool bar followed - it is a shortcut row for the menu. Send is NOT
 * here: it is the verb for the text in the input box and belongs beside
 * that box, not in a row of file commands.
 *
 * NO SEPARATORS, and the reason is measured rather than a preference.
 * The shipped binary carries a Microsoft.Windows.Common-Controls 6.0
 * manifest ("passive theming"), so on XP and later
 * comctl32 draws this bar borderless and flat - and a themed separator
 * is not a line, it is nothing at all. Grouped the way the menus are
 * grouped, that produced five icons with two inexplicable gaps in the
 * middle. The grouping is a nice-to-have; a gap nobody can explain is
 * not.
 *
 * Glyphs are comctl32's own, chosen for what the era's strip HAS.
 * FILEOPEN, FILESAVE and PROPERTIES are literal. It has no "stop", so
 * that one takes the period's own reading: Internet Explorer's Stop
 * button was an X. FILENEW for clearing is "start a fresh page". */
static HWND create_toolbar(HWND hwnd) {
    /* See the comment on g.no_toolbar: this is the only way the selftest
       can walk the "comctl32 is absent" path on a machine that has it. */
    if (g.no_toolbar) return NULL;
    /* IDM_REGEN - the one rollback command that gets
       a button, because it is the one a user reaches for repeatedly.
       LZ_STD_REDOW is comctl32's own redo arrow, which is as close as
       the system strip comes to "do that again". */
    static const int CMDS[6] = {
        IDM_OPEN_MODEL, IDM_SAVE_CHAT, IDM_CLEAR, IDM_STOP_GEN,
        IDM_REGEN, IDM_SETTINGS };
    static const int GLYPHS[6] = {
        LZ_STD_FILEOPEN, LZ_STD_FILESAVE, LZ_STD_FILENEW,
        LZ_STD_DELETE, LZ_STD_REDOW, LZ_STD_PROPERTIES };
    const char *labels[6];
    labels[0] = lz_str_display(LZ_STR_BTN_OPEN);
    labels[1] = lz_str_display(LZ_STR_BTN_SAVE);
    labels[2] = lz_str_display(LZ_STR_BTN_CLEAR);
    labels[3] = lz_str_display(LZ_STR_BTN_STOP);
    labels[4] = lz_str_display(LZ_STR_BTN_REGEN);
    labels[5] = lz_str_display(LZ_STR_BTN_SETTINGS);
    return lz_gui_toolbar_create(hwnd, g.inst, CMDS, GLYPHS, labels, 6,
                                 ID_TOOLBAR);
}

/* kunkun98.ini persistence for the recent list: mru0 is the newest,
 * mru3 the oldest, matching g.mru.item[]. Kept here rather than in
 * gui/mru.c so that module stays free of Win32 and the ini format - the
 * same split gui/inifile.c itself follows. */
static void mru_load(void) {
    char path[LZ_MRU_MAX][LZ_MRU_LEN];
    char key[8];
    int i;

    lz_mru_init(&g.mru);
    for (i = 0; i < LZ_MRU_MAX; i++) {
        sprintf(key, "mru%d", i);
        lz_ini_get_str(key, path[i], (int)sizeof path[i], "");
    }
    /* Pushed oldest first: lz_mru_push always inserts at the front, so
       pushing mru3 .. mru0 in that order leaves item[0] as mru0 again -
       the saved order survives the round trip. A directory that no
       longer has a model in it is dropped here rather than shown greyed
       - a menu item that does nothing when clicked is worse than one
       fewer item (gui/mru.h). */
    for (i = LZ_MRU_MAX - 1; i >= 0; i--) {
        if (path[i][0] && lz_gui_model_dir_ok(path[i]))
            lz_mru_push(&g.mru, path[i]);
    }
}

static void mru_save(void) {
    char key[8];
    int i;
    for (i = 0; i < LZ_MRU_MAX; i++) {
        sprintf(key, "mru%d", i);
        lz_ini_set_str(key, i < g.mru.n ? g.mru.item[i] : "");
    }
}

static int create_children(HWND hwnd) {
    /* BEFORE build_menu_bar, which ends in rollback_sync, which reads
       these. The struct's own zero-init would leave them at 0 - a
       perfectly valid position, the very start of the transcript -
       rather than at the "nothing recorded" sentinel, and the only
       reason that is not already a live bug is that history is empty
       here too. One line beats depending on that coincidence. */
    turn_cp_forget();
    /* Before the menu is built: the File popup draws the list at
       creation time, same as every other menu label in this file. */
    mru_load();
    build_menu_bar(hwnd);
    /* Loaded here rather than in WinMain so that --selftest, which never
       reaches the message loop, can still check it is there. */
    g.accel = LoadAcceleratorsA(g.inst, MAKEINTRESOURCEA(IDR_ACCEL));
    /* The literal string, not the FINDMSGSTRING macro:
       commdlg.h supplies that macro and cannot compile at this file's
       3.51 floor (see the comment above open_find, further down).
       FINDMSGSTRINGA at commdlg.h:493 in this host's MinGW headers and
       commdlg.h:245 in Watcom's - both "commdlg_FindReplace", checked
       rather than assumed, the same way VK_ESCAPE's value was.
       RegisterWindowMessageA never returns 0 for a real registration
       (0 means the call itself failed), which is what lets g_findmsg's
       own guard below tell "not yet registered" apart from "registered,
       but this isn't it". */
    g_findmsg = RegisterWindowMessageA("commdlg_FindReplace");
    const char *RICH_CLASS;
    int i;

    /* riched20 is loaded by hand rather than through an import library:
       Watcom has no import lib for it, and a missing DLL must surface as
       a message the user can act on instead of a loader failure box in
       the system's own words. RichEdit is mandatory - a
       plain EDIT caps out at 32 KB of conversation. */
    RICH_CLASS = lz_richedit_class();
    if (!RICH_CLASS) {
        MessageBoxA(hwnd, lz_str_display(LZ_STR_ERR_NO_RICHEDIT),
                    lz_str_display(LZ_STR_ERR_TITLE), MB_ICONHAND | MB_OK);
        return 0;
    }

    g.part[LZ_GUI_TRANSCRIPT] = CreateWindowExA(
        lz_ex_style(WS_EX_CLIENTEDGE), RICH_CLASS, "",
        WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_READONLY |
        ES_AUTOVSCROLL,
        0, 0, 0, 0, hwnd, (HMENU)ID_TRANSCRIPT, g.inst, NULL);
    /* WM_SETFONT alone is not enough for a RichEdit - see
       lz_richedit_use_font's own comment (gui/compat40.c): IMF_AUTOFONT
       substitutes a face per character by script, so a Chinese font
       selected via WM_SETFONT still comes out ASCII-fixed-pitch on any
       English text sharing the line (model names, numbers, the KV
       figure). The apply_font loop further down still sends WM_SETFONT
       here too - that sets the default RichEdit falls back to when
       IMF_AUTOFONT is off but a run's own CHARFORMAT was never set, so
       both calls stay, not just this one. */
    lz_richedit_use_font(g.part[LZ_GUI_TRANSCRIPT], lz_ui_font());
    /* The RichEdit's scrollbar, not its border - WS_VSCROLL
       above is a standard NONCLIENT scrollbar, themed by uxtheme the
       same generic way any other window's is, regardless of riched20's
       own client-area drawing (see lz_ui_untheme's own comment in
       gui/compat40.c for the fuller reasoning). Only observable on
       kunkun98-gui, the manifest build -
       kunkun98-noman has no application manifest and therefore no
       theming to turn off on anything, so there is nothing here for a
       before/after comparison to find on that binary. */
    g.transcript_untheme_ok = lz_ui_untheme(g.part[LZ_GUI_TRANSCRIPT]);
    /* Fixed white background (user request: a custom
       desktop colour scheme could make the transcript hard to read,
       same motivation as the explicit LZ_COLOR_* literals above - see
       that comment). EM_SETBKGNDCOLOR is a RichEdit-only message and
       riched32 (the 3.51-era DLL lz_richedit_class can fall back to)
       supports it too, so this needs no floor check. The input EDIT
       below has no such message - see its own comment and the
       WM_CTLCOLOREDIT case in wndproc for why it is a completely
       different mechanism. */
    SendMessage(g.part[LZ_GUI_TRANSCRIPT], EM_SETBKGNDCOLOR, 0,
               RGB(0xFF, 0xFF, 0xFF));

    g.part[LZ_GUI_INPUT] = CreateWindowExA(
        lz_ex_style(WS_EX_CLIENTEDGE), "EDIT", "",
        WS_CHILD | WS_VISIBLE | WS_TABSTOP | WS_VSCROLL | ES_MULTILINE |
        ES_AUTOVSCROLL | ES_WANTRETURN,
        0, 0, 0, 0, hwnd, (HMENU)ID_INPUT, g.inst, NULL);
    lz_edit_use_font_margins(g.part[LZ_GUI_INPUT]);
    /* Down to the RichEdit's border, not up to the theme's. Theming the
       RichEdit's border is possible - subclass WM_NCPAINT and
       DrawThemeBackground an EP_EDITBORDER - but that pulls the window
       towards a modern look, and this one is going the other way. */
    g.input_untheme_ok = lz_ui_untheme(g.part[LZ_GUI_INPUT]);
    /* No EM_SETBKGNDCOLOR call here - a plain EDIT control does not
       understand that message at all (it is RichEdit-specific). Its
       fixed white background is set from the PARENT's own
       WM_CTLCOLOREDIT instead - see that case in wndproc, and
       input_bkg_brush's own comment for why the brush is cached
       rather than created here. Read as covering the SAME user
       request as the RichEdit's background above: "a text box's
       background stays white regardless of the desktop scheme"
       applies to both of this window's text boxes, not just the one
       that happens to have a one-line API for it. */

    /* Tool bar. Command ids are reused (ID_SEND, ID_STOP, IDM_CLEAR,
       IDM_SETTINGS) so the WM_COMMAND switch needs no new cases.

       NULL on NT 3.51, where there is no comctl32 - see cmd_enable and
       gui/toolbar.h. Every one of these four is also on a menu, so the
       floor loses a convenience and no function. */
    g.part[LZ_GUI_TOOLBAR] = create_toolbar(hwnd);

    /* Send, beside the input box. A real button in its own rectangle -
       the field shrank to make room - rather than something floating
       over the text, which is a shape this era did not have. */
    g.part[LZ_GUI_SEND] = CreateWindowExA(
        0, "BUTTON", lz_str_display(LZ_STR_BTN_SEND),
        WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_DEFPUSHBUTTON,
        0, 0, 0, 0, hwnd, (HMENU)ID_SEND, g.inst, NULL);

    /* The splitter is a RECTANGLE, not a window - see gui/layout.c. The
       part exists so the layout owns its geometry and the tests can
       sweep it; nothing is created here, and g.part stays NULL.
       Parking the parent's own handle here to keep the "every part
       exists" loop quiet would be caught immediately: the geometry
       checks compare the parent's rect against the band's. The checks
       below name the exception instead of hiding it. */
    g.part[LZ_GUI_SPLIT] = NULL;

    /* Right sidebar in the shape an early instant messenger used: a
       small framed head, the name under it, then the model state.

       The frame is WS_EX_CLIENTEDGE rather than a bare STATIC. An avatar
       of that era was always inside a sunken border; floating on the
       dialog background is what made this read as a poster rather than
       as a buddy. lz_ex_style drops the bit on 3.51, where the picture
       simply has no frame. */
    g.part[LZ_GUI_SIDE_CHICKEN] = CreateWindowExA(
        lz_ex_style(WS_EX_CLIENTEDGE), "STATIC", "",
        WS_CHILD | WS_VISIBLE | SS_BITMAP,
        0, 0, 0, 0, hwnd, (HMENU)ID_SIDE_CHICKEN, g.inst, NULL);
    if (g.part[LZ_GUI_SIDE_CHICKEN]) {
        HBITMAP hb = LoadBitmapA(g.inst, MAKEINTRESOURCE(IDB_CHICKEN));
        if (hb)
            SendMessage(g.part[LZ_GUI_SIDE_CHICKEN], STM_SETIMAGE,
                        IMAGE_BITMAP, (LPARAM)hb);
    }
    g.part[LZ_GUI_SIDE_NAME] = CreateWindowExA(
        0, "STATIC", lz_str_display(LZ_STR_APP_TITLE),
        WS_CHILD | WS_VISIBLE | SS_CENTER,
        0, 0, 0, 0, hwnd, (HMENU)ID_SIDE_NAME, g.inst, NULL);
    /* SS_CENTER, not SS_LEFT: the sidebar items are horizontally
       centred in the new width - the third of the three items being
       this one. SIDE_NAME just above already centres this way;
       SIDE_CHICKEN's centring is geometry (its rect is centred in
       SIDE_W, not a style bit - a bitmap has no alignment style to
       set), so this is the only sidebar item whose centring is a style
       bit. */
    g.part[LZ_GUI_SIDE_INFO] = CreateWindowExA(
        0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_CENTER,
        0, 0, 0, 0, hwnd, (HMENU)ID_SIDE_INFO, g.inst, NULL);

    /* Status bar: comctl32's standard sunken one when available, a
       STATIC fallback otherwise (3.51 without comctl32).

       SBARS_SIZEGRIP is hand-defined for the same reason
       INITCOMMONCONTROLSEX is in compat40.c: commctrl.h is not included
       anywhere in this front end, and pulling it in for one constant
       would drag a header the 3.51 floor has to be argued with. The
       value is 0x0100 and has been since Windows 95.

       It matters more than it looks. The window is WS_OVERLAPPEDWINDOW,
       so it resizes; a resizable Win9x window whose status bar has no
       gripper in the bottom-right corner reads as wrong at a glance,
       and the corner is also where people grab. */
    {
        const char *sbar = lz_statusbar_class();
#ifndef SBARS_SIZEGRIP
#define SBARS_SIZEGRIP 0x0100
#endif
        if (sbar) {
            g.part[LZ_GUI_STATUS] = CreateWindowExA(
                0, sbar, "", WS_CHILD | WS_VISIBLE | SBARS_SIZEGRIP,
                0, 0, 0, 0, hwnd, (HMENU)ID_STATUS, g.inst, NULL);
            g.status_is_sbar = g.part[LZ_GUI_STATUS] != NULL;
            if (g.status_is_sbar) {
                g_sb_orig = (WNDPROC)(LZ_IPTR)SetWindowLongPtrA(
                    g.part[LZ_GUI_STATUS], GWLP_WNDPROC,
                    (LZ_IPTR)sb_bevel_proc);
            }
        }
        /* The twin lamps, WinZip's idiom: left = model state, right =
           activity. Children of the STATUS BAR, not of the window - they
           belong to a reserved cell in the bar, and a child of the bar
           moves and clips with it for free.
           Only when the bar is comctl32's. The STATIC fallback (3.51
           without comctl32) has no parts to reserve one of, and a lamp
           floating on a plain text strip is worse than no lamp. */
        /* The fallback strip is created BEFORE the lamps, because the
           lamps are children of whichever strip exists and need a parent
           to be created into. */
        if (!g.part[LZ_GUI_STATUS]) {
            sb_register_fallback();
            g.part[LZ_GUI_STATUS] = CreateWindowExA(
                0, LZ_SBCLASS,
                lz_str_display(LZ_STR_STATE_NO_MODEL),
                WS_CHILD | WS_VISIBLE,
                0, 0, 0, 0, hwnd, (HMENU)ID_STATUS, g.inst, NULL);
        }
        if (g.part[LZ_GUI_STATUS]) {
            int i2;
            for (i2 = 0; i2 < 2; i2++)
                g.lamp[i2] = CreateWindowExA(
                    0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_BITMAP,
                    0, 0, LZ_LAMP_PX, LZ_LAMP_PX,
                    g.part[LZ_GUI_STATUS], (HMENU)(LZ_IPTR)(ID_LAMP0 + i2),
                    g.inst, NULL);
            lamps_reload();
        }
        /* The prefill indicator. comctl32 only - the fallback strip has
           no control to create and will draw its own; this is the arm
           the simulation gets measured against. Created hidden: it is
           shown only while a prefill is running. */
        {
            const char *pcls = lz_progress_class();
            if (pcls && g.part[LZ_GUI_STATUS]) {
                g.progress = CreateWindowExA(
                    0, pcls, "", WS_CHILD | WS_VISIBLE,
                    0, 0, 0, 0, g.part[LZ_GUI_STATUS],
                    (HMENU)(LZ_IPTR)ID_PROGRESS, g.inst, NULL);
                /* Untheme it, for the same reason the transcript and the
                   input box are untheme'd: this program ships a
                   common-controls 6.0 manifest, so on XP and later
                   uxtheme repaints the bar as one smooth fill. The
                   deliverable's look is Win9x, where a progress bar is
                   a row of separate chunks - and the 3.51 simulation has
                   to be built against THAT, not against whatever the
                   host's theme draws. No-op on the target, which has no
                   uxtheme to turn off. */
                g.progress_untheme = lz_ui_untheme(g.progress);
            }
        }
    }

    /* Four parts may be NULL, for three different reasons, and all are
       named here rather than left to a reader to infer:
         LZ_GUI_SPLIT   is a band of background, not a control at all.
         LZ_GUI_TOOLBAR is comctl32's, and every command on it is also
                        on a menu, so a machine without the class loses
                        a shortcut row and no function.
         LZ_GUI_SIDE_LAMPS/SIDE_CAND do not exist yet AT
                        ALL - no model is loaded this early, and
                        side_panel_sync is what creates them, later,
                        once one is. LZ_GUI_SIDE_LAMPS never gets a
                        g.part[] entry regardless (see its own comment
                        in layout.h) - listed anyway so this loop's own
                        exception set stays self-documenting rather
                        than relying on that fact holding forever.
       The toolbar is left out of the mandatory set: requiring it would
       turn "no comctl32" into "the window does not open" - the
       opposite of what gui/toolbar.c's own comment promises. */
    for (i = 0; i < LZ_GUI_PART_COUNT; i++) {
        if (i == LZ_GUI_SPLIT || i == LZ_GUI_TOOLBAR ||
            i == LZ_GUI_SIDE_LAMPS || i == LZ_GUI_SIDE_CAND)
            continue;
        if (!g.part[i]) return 0;
        apply_font(g.part[i]);
        /* Same call the transcript and input box already need - every
           OTHER mandatory control gets it here, in the one place that
           already touches all of them, so a future addition to this
           loop cannot repeat the omission a screenshot caught three
           times over. */
        g.part_untheme_ok[i] = lz_ui_untheme(g.part[i]);
    }
    /* LZ_GUI_TOOLBAR is the documented exception the roster gate's own
       comment promises - measured, not assumed: calling
       lz_ui_untheme on it changes what TB_GETMAXSIZE reports (46 -> 50
       needed), because unlike EDIT/LISTBOX/BUTTON/STATIC a comctl32
       ToolbarWindow32's theme is not purely border/scrollbar chrome -
       it is load-bearing for the control's own button metrics. Tried
       once, TOOLBAR_H's own 46 (already itself a measured, not chosen,
       value - see its own comment) broke immediately ("toolbar: the
       layout is tall enough for its buttons" went red). Left themed on
       purpose rather than re-measuring TOOLBAR_H against the unthemed
       metric and re-verifying every geometry test that depends on it
       for a control whose classic/themed distinction is genuinely
       different in kind from the others on this roster. */
    if (g.part[LZ_GUI_TOOLBAR]) apply_font(g.part[LZ_GUI_TOOLBAR]);

    /* The status bar's real height - see measure_status_h's own
       comment for why this has to run AFTER apply_font, not right
       after creation. */
    measure_status_h();

    /* Without this the transcript stops accepting text at 64 KB, and
       the symptom is a conversation that quietly stops updating. */
    SendMessage(g.part[LZ_GUI_TRANSCRIPT], EM_EXLIMITTEXT, 0,
                (LPARAM)LZ_TRANSCRIPT_LIMIT);
    /* Stop needs a job in flight - a live control that does nothing is
       indistinguishable from a broken one, which is why it starts
       greyed the way the menu item does.
       Send does NOT wait for a model the same way: do_send's own
       "commands come first" comment says /load and /help are exactly
       what a user with no model open needs to reach, so greying Send
       for "no model" told that user their own commands did not work.
       It stays live from here on - cmd_is_enabled (above) is what
       actually stops WM_COMMAND/ID_SEND from doing anything while
       do_send's own lz_worker_busy() guard applies; a busy job is
       between commands too, so it does not need Send greyed either. */
    cmd_enable(ID_SEND, 1);
    cmd_enable(IDM_STOP_GEN, 0);
    /* AGAIN, after the tool strip exists. build_menu_bar ran near the
       top of this function and ended in this same call, but the strip
       is created further down and every button on a fresh strip is
       live - so without this the Retry BUTTON would be usable while
       its own menu item was correctly greyed, which is precisely the
       disagreement cmd_enable's own comment says teaches a user not to
       trust either face. */
    rollback_sync();
    lz_stream_init(&g.stream);
    lz_common_settings_init(&g.set);
    lz_gui_session_init(&g.sess, &g.mdl, g.set.think);
    /* Set once, on the session's own opts - common/session.c takes
       configuration from s->opts before each job, so nothing has to be
       threaded through lz_session_job's signature for this. */
    g.sess.opts.on_prefill = gui_prefill_progress;
    /* Multi-turn prefix reuse, ON by default.
     *
     * The prefix matcher validates the cached prefix BY CONTENT and
     * rejects a stale one on its own, so a missed invalidation costs a
     * wasted match, not a wrong answer. All three invalidation points
     * (clear / replace history / replace model) are driven through real
     * generation and compared PREFIX against FULL byte for byte.
     *
     * The key stays: prefill=0 restores the full-prefill path exactly,
     * and it is the control arm the parity gate needs. NOT in the
     * settings dialog - it is a "set once per machine" knob, not
     * something to reach for when a reply disappoints.
     *
     * Read here rather than folded into lz_gui_session_init's own
     * zero-init, the same shape as g.set.think/temp being applied right
     * after that call - lz_gui_session_init has no ini access of its
     * own, by design (gui/session.c stays pure logic, no Win32, no
     * file I/O). */
    g.sess.prefill = lz_ini_get_int("prefill", 1) ? LZ_PREFILL_PREFIX
                                                  : LZ_PREFILL_FULL;
    /* SetWindowLongPtrA / GWLP_WNDPROC rather than the older spelling:
       Watcom defines the Ptr form as a macro straight onto
       SetWindowLongA, which is exactly the 32-bit call Win98 has, while
       MinGW's 64-bit headers no longer declare GWL_WNDPROC at all. One
       spelling, both toolchains, and the target still gets the API it
       actually owns. */
    g.input_proc = (WNDPROC)SetWindowLongPtrA(g.part[LZ_GUI_INPUT],
                                              GWLP_WNDPROC,
                                              (LONG_PTR)input_subclass);
    /* Same idea, on the transcript: WM_CONTEXTMENU is the only message
       this subclass wants, everything else falls straight through to
       riched32/riched20's own procedure. */
    g.transcript_proc = (WNDPROC)SetWindowLongPtrA(
        g.part[LZ_GUI_TRANSCRIPT], GWLP_WNDPROC,
        (LONG_PTR)transcript_subclass);
    /* Restored settings. Each one falls back to what the program would
       have used anyway, so a missing or truncated ini is not a special
       case - it is a fresh install. */
    g.input_h = lz_ini_get_int("input_h", 0);   /* 0 = the layout default */
    /* How often the transcript catches up with the generator. 0 means
       "every token, immediately" - the pre-throttle behaviour, kept
       reachable so there is something to compare against. Clamped
       because a hand-edited ini is untrusted input and a tick of an
       hour would look exactly like a hung window. */
    g.tok_ms = lz_ini_get_int("stream_ms", LZ_TOK_MS);
    if (g.tok_ms < 0) g.tok_ms = 0;
    if (g.tok_ms > LZ_TOK_MS_MAX) g.tok_ms = LZ_TOK_MS_MAX;
    /* Seed policy. Stored as an int rather than the full 64 bits the
       engine takes: a reproducibility seed is a number a person types,
       and an ini a person edited by hand is the only thing that writes
       this. The widening on the way out is deliberate and lossless. */
    g.set.seed_mode = lz_ini_get_int("seed_mode", LZ_COMMON_SEED_RANDOM)
                      ? LZ_COMMON_SEED_FIXED : LZ_COMMON_SEED_RANDOM;
    g.set.seed = (unsigned long long)(unsigned)lz_ini_get_int("seed", 1);
    g.set.think = lz_ini_get_int("think", g.set.think) ? 1 : 0;
    /* Context window. Through the clamp with no model cap - there is no
       model at this point in startup, and the cap is applied again at
       load time where cfg->seq_len is finally knowable. Clamping here
       as well is not redundant: a hand-edited ini is untrusted input,
       and an out-of-range number would otherwise reach the slider,
       where lz_common_ctx_to_scroll would snap it into range without
       anyone having decided that it should. */
    g.set.ctx = lz_common_ctx_clamp(lz_ini_get_int("ctx", LZ_COMMON_CTX_DEFAULT), 0);
    /* Ini-only, no dialog row - see g.repeat_last_n. Read straight from
       the engine's own default so the two cannot drift, and NOT clamped:
       0, -1 and any positive window are all meaningful to the sampler
       (off / whole generation / a window), so there is no range to
       enforce here that sampler.h does not already define. */
    {
        LZSampleParams sp;
        lz_sample_defaults(&sp);
        g.repeat_last_n = lz_ini_get_int("repeat_last_n", sp.repeat_last_n);
    }
    {
        /* Through the setters, not into the fields: the ranges are
           rules and an ini a user edited by hand is untrusted input. A
           refused value leaves the mode default in place.
           milli / 1000.0f is FRONT-END code, not inside
           lz_fpu_float_begin/end and not in src/, so iron law six's
           rule 1 (never divide by a float constant) does not apply
           here - that rule protects the engine's cross-compiler
           bit-exactness, which these lines have no part in.

           THOUSANDTHS because the profile API has no float form
           (gui/inifile.h) and parsing one back would go through the C
           library's locale-dependent decimal point - on a machine set
           to a comma locale, "0.60" reads as 0. */
        int milli = lz_ini_get_int("temp_milli", -1);
        if (milli >= 0)
            if (lz_common_settings_set_temp(&g.set, (float)milli / 1000.0f) == 0)
                g.set.manual_temp = 1;
        /* The manual flag is NOT restored from the ini for either of
           these, it is INFERRED from the key being present at all:
           writing the value is what the dialog does when the user
           commits one, so a key that exists means a user chose it. The
           alternative - a separate manual_topp key - would be a second
           authority for "did the user set this", which is exactly what
           the two-flags note in settings.h exists to prevent. */
        milli = lz_ini_get_int("topp_milli", -1);
        if (milli >= 0)
            if (lz_common_settings_set_topp(&g.set, (float)milli / 1000.0f) == 0)
                g.set.manual_topp = 1;
        milli = lz_ini_get_int("rep_milli", -1);
        if (milli >= 0)
            lz_common_settings_set_rep(&g.set, (float)milli / 1000.0f);
        /* think-block temperature, same key-present-means-user-chose
           rule as topp: written only when manual_think_temp, so a key
           that exists restores a value the user actually set, not the
           0.3 default re-read as a manual choice. */
        milli = lz_ini_get_int("think_temp_milli", -1);
        if (milli >= 0)
            if (lz_common_settings_set_think_temp(&g.set, (float)milli / 1000.0f) == 0)
                g.set.manual_think_temp = 1;
        /* Capped against the context just read, not against 0: an ini
           holding a max_new larger than its own ctx is a file someone
           edited, and the dialog would refuse that pair too. */
        lz_common_settings_set_max_new(&g.set,
            lz_ini_get_int("max_new", LZ_COMMON_MAXNEW_UNLIMITED), g.set.ctx);
        /* Custom system prompt. The ini holds it as a plain
           string - newlines survive because the profile API reads a
           value verbatim between the key's = and the next \n. Read into
           the field, clamped, so a hand-edited ini cannot overflow it
           into the next field. */
        {
            char tmp[LZ_COMMON_SYSTEM_MAX + 1];
            lz_ini_get_str("system", tmp, (int)sizeof tmp, "");
            strncpy(g.set.system, tmp, sizeof g.set.system - 1);
            g.set.system[sizeof g.set.system - 1] = '\0';
        }
    }
    apply_settings();
    set_idle_status(lz_str_utf8(LZ_STR_STATE_NO_MODEL));
    update_ctx_cell();          /* empty: no model yet */
    return 1;
}

static void relayout(HWND hwnd) {
    RECT rc;
    LZRect r[LZ_GUI_PART_COUNT];
    int i;
    GetClientRect(hwnd, &rc);
    lz_gui_layout(rc.right - rc.left, rc.bottom - rc.top, g.input_h,
                 g.status_h, side_panel_mode(), r);
    for (i = 0; i < LZ_GUI_PART_COUNT; i++) {
        /* LZ_GUI_SPLIT holds the parent's own handle - it is a band of
           background the window hit-tests, not a control - so moving it
           would move the window. LZ_GUI_SIDE_LAMPS has no single HWND
           either - see the loop right below instead - g.part[] simply
           never has an entry for it (permanently NULL), so this check
           already skips it for free. LZ_GUI_SIDE_CAND is excluded on
           purpose, not for lack of a handle: it gets its own block
           below, because that rect actually holds TWO controls
           (cand_title on top, the LISTBOX under it), not one. */
        if (g.part[i] && i != LZ_GUI_SPLIT && i != LZ_GUI_SIDE_CAND)
            MoveWindow(g.part[i], r[i].x, r[i].y, r[i].w, r[i].h, TRUE);
    }
    g.split = r[LZ_GUI_SPLIT];
    /* LZ_GUI_SIDE_CAND's rect, split top to bottom: the title strip
       (LZ_GUI_SIDE_LABEL_H, the same one-line height LZ_GUI_SIDE_NAME
       uses) and the LISTBOX taking whatever is left. Both NULL when
       no model is loaded (side_panel_sync never created them), so
       this is a no-op then - MoveWindow is simply never called. */
    if (g.cand_title) {
        int title_h = LZ_GUI_SIDE_LABEL_H;
        int rest_h;
        if (title_h > r[LZ_GUI_SIDE_CAND].h) title_h = r[LZ_GUI_SIDE_CAND].h;
        rest_h = r[LZ_GUI_SIDE_CAND].h - title_h;
        if (rest_h < 0) rest_h = 0;
        MoveWindow(g.cand_title, r[LZ_GUI_SIDE_CAND].x, r[LZ_GUI_SIDE_CAND].y,
                  r[LZ_GUI_SIDE_CAND].w, title_h, TRUE);
        if (g.part[LZ_GUI_SIDE_CAND])
            MoveWindow(g.part[LZ_GUI_SIDE_CAND], r[LZ_GUI_SIDE_CAND].x,
                      r[LZ_GUI_SIDE_CAND].y + title_h,
                      r[LZ_GUI_SIDE_CAND].w, rest_h, TRUE);
    }
    /* The 16 expert lamps: an 8x2 grid inside the bounding box
       layout.c reserved for it, positioned individually because
       g.elamp[] holds 16 handles, not one - the same reason the
       status bar's own twin lamps get their own positioning pass just
       below rather than a g.part[] entry. A NULL g.elamp[i] (no
       model, or a model with no MoE) is simply skipped - side_
       panel_sync is what decides whether they exist at all, this
       function only ever answers "where", never "whether". */
    {
        int col, row;
        for (i = 0; i < 16; i++) {
            if (!g.elamp[i]) continue;
            col = i % LZ_GUI_ELAMP_COLS;
            row = i / LZ_GUI_ELAMP_COLS;
            MoveWindow(g.elamp[i],
                      r[LZ_GUI_SIDE_LAMPS].x +
                          col * (LZ_GUI_ELAMP_PX + LZ_GUI_ELAMP_GAP),
                      r[LZ_GUI_SIDE_LAMPS].y +
                          row * (LZ_GUI_ELAMP_PX + LZ_GUI_ELAMP_GAP),
                      LZ_GUI_ELAMP_PX, LZ_GUI_ELAMP_PX, TRUE);
        }
    }
    /* Three status-bar parts: state (0), model info (1), and a reserved
       cell on the right for the two lamps (2). The boundaries follow the
       width, so a resize keeps the model info and the lamps where they
       belong. */
    if (g.status_is_sbar) {
        /* Assigned, not initialised: Watcom rejects a non-constant
           initialiser here (E1054), and the boundary is runtime data. */
        /* The sizing grip owns the bottom-right corner - it is drawn by
           the bar itself, over whatever the last part contains - so the
           reserved cell stops short of it. Without this the second lamp
           sat under the grip's diagonal ridges. SM_CXVSCROLL is what a
           status-bar gripper is sized by. */
        int parts[4], w = rc.right - rc.left;
        int grip = GetSystemMetrics(SM_CXVSCROLL);
        int cell = w - LZ_LAMP_CELL_W - grip;
        /* Sized to its own text, not to a fraction of the bar - see
           status_state_w. Clamped so a narrow window shrinks the state
           cell instead of pushing the model info off the right end. */
        parts[0] = status_state_w(g.part[LZ_GUI_STATUS]);
        if (parts[0] <= 0 || parts[0] > cell) parts[0] = cell;
        /* The same boundaries the fallback strip paints from, so neither
           can drift from the other. */
        g_sb_p0 = parts[0];
        g_sb_p1 = cell;
        /* FOUR parts: the prefill indicator is a segment of its own on
           this path too, so both strips have the same structure. */
        {
            RECT sr, pc;
            int pl = cell;
            GetClientRect(g.part[LZ_GUI_STATUS], &sr);
            if (sb_prog_rect(sr.bottom - sr.top, &pc) &&
                pc.left > parts[0] + LZ_SB_GAP)
                pl = pc.left;
            /* Minus the gap: comctl32 starts the next part two pixels
               past the boundary it is given, so the segment's own left
               edge would land the part two pixels narrow. */
            parts[1] = pl - LZ_SB_GAP;
            parts[2] = cell;        /* the indicator's own segment */
            parts[3] = -1;          /* lamps, no border */
        }
        SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_SETPARTS, 4, (LPARAM)parts);
        /* Every part with no border of the control's own - 0 and 1 get
           the Win95 bevel from sb_bevel_proc instead, and 2 holds the
           lamps and gets none at all. Sent after SB_SETPARTS because the
           parts have to exist first, and re-sent on every relayout
           because SB_SETPARTS resets the flags - all three of them, see
           sb_reflag. */
        sb_reflag();
        /* The lamps live inside the bar, so their geometry comes off the
           bar and not off the client area - a child of the status bar
           moves with it and is clipped by it. */
    } else if (g.part[LZ_GUI_STATUS]) {
        /* The fallback strip's own copy of the boundaries above, by the
           same arithmetic - it has no SB_SETPARTS to be told them. */
        int w = rc.right - rc.left;
        int grip = GetSystemMetrics(SM_CXVSCROLL);
        int cell = w - LZ_LAMP_CELL_W - grip;
        int p0 = status_state_w(g.part[LZ_GUI_STATUS]);
        if (p0 <= 0 || p0 > cell) p0 = cell;
        g_sb_p0 = p0;
        g_sb_p1 = cell;
        InvalidateRect(g.part[LZ_GUI_STATUS], NULL, FALSE);
    }
    /* The lamps are placed for BOTH strips - they are children of
       whichever one exists. The comctl32 bar answers SB_GETRECT for its
       reserved cell; the fallback has no parts to ask about, so the cell
       is computed the same way SB_SETPARTS was told to reserve it, which
       keeps one arithmetic rather than two. */
    if (g.lamp[0] && g.lamp[1] && g.part[LZ_GUI_STATUS]) {
        RECT pr;
        int y, x;
        int w = rc.right - rc.left;
        int grip = GetSystemMetrics(SM_CXVSCROLL);
        int cell = w - LZ_LAMP_CELL_W - grip;
        {
            /* Centred in the PART, not in the client rect. Those are not
               the same rectangle: the bar keeps its top border and each
               part keeps a sunken edge, and a 16-pixel lamp centred on
               the client rect ran over both of them. Asking SB_GETRECT
               is what stops that inset from being guessed. */
            if (!g.status_is_sbar ||
                !SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETRECT, 3,
                             (LPARAM)&pr)) {
                GetClientRect(g.part[LZ_GUI_STATUS], &pr);
                pr.left = cell;
            }
            y = pr.top + (pr.bottom - pr.top - LZ_LAMP_PX) / 2;
            if (y < pr.top) y = pr.top;
            x = pr.left + LZ_LAMP_PAD;
            MoveWindow(g.lamp[0], x, y, LZ_LAMP_PX, LZ_LAMP_PX, TRUE);
            MoveWindow(g.lamp[1], x + LZ_LAMP_PX + LZ_LAMP_GAP, y,
                       LZ_LAMP_PX, LZ_LAMP_PX, TRUE);
            /* The indicator fills cell 1, inset by the cell's own sunken
               edge so it does not sit on the bevel. */
            if (g.progress) {
                /* The STRIP's own client height, not the window's - `rc`
                   up at the top of this function is the main window and
                   using it made the bar 456 pixels tall inside a
                   22-pixel strip. */
                /* The SAME rectangle the fallback paints into - see
                   sb_prog_rect. */
                /* The control fills its segment: its own frame is the
                   well, which is why the strip does not bevel part 2.
                   The part rect is asked for rather than derived - the
                   bar keeps its own margins, and a second opinion about
                   where a part is would drift from the first. */
                RECT q;
                if (SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETRECT,
                                (WPARAM)2, (LPARAM)&q))
                    MoveWindow(g.progress, q.left, q.top,
                               q.right - q.left, q.bottom - q.top, TRUE);
            }
        }
    }
}

/* Dialog-unit tab stop for the candidate LISTBOX's two columns (token
 * text, probability) - LB_SETTABSTOPS positions are dialog template
 * units (MapDialogRect's own unit), the same convention LZ_DLU_X
 * converts FROM; this goes the other way, an approximate inverse
 * (2*px/3, since LZ_DLU_X's own forward direction is roughly
 * px = dlu*1.5). 60 is chosen for room, not measured against a live
 * control - a probability like "0.1234" is 6 ASCII characters
 * regardless of what the token column holds, so the second column
 * only ever needs to be sized against that, not against the widest
 * token text (which the layout does not control anyway). */
#define LZ_GUI_CAND_TAB 60

/* Create or destroy the candidate LISTBOX and the 16 expert lamps to
 * match whatever model is currently loaded - called once, right after
 * a load job finishes (finish_job's JOB_LOAD branch, ok or not), which
 * is the only place "is a model loaded, does it have MoE" changes
 * from the UI's point of view. Nothing else needs to call this: a
 * generation cannot change which model is loaded, and IDM_CLEAR does
 * not touch g.mdl at all (the model stays loaded across a
 * clear).
 *
 * Idempotent, and always tears down before rebuilding rather than
 * diffing old against new - a model swap can change num_experts from
 * >0 to 0 or back, which is rare but not impossible, and diffing it
 * would be a second, mostly-untested path for a case the simple
 * tear-down-then-rebuild already covers by construction. Safe to call
 * before anything exists yet (WM_CREATE has not run relayout(), and
 * every handle here starts NULL) - every DestroyWindow/DeleteObject
 * below is itself a no-op on NULL. */
static void side_panel_sync(HWND hwnd) {
    int i;
    int num_experts;

    for (i = 0; i < 16; i++) {
        if (g.elamp[i]) { DestroyWindow(g.elamp[i]); g.elamp[i] = NULL; }
    }
    for (i = 0; i < LZ_LAMP_KINDS; i++)
        if (g.elamp_bmp[i]) { DeleteObject(g.elamp_bmp[i]); g.elamp_bmp[i] = NULL; }
    if (g.part[LZ_GUI_SIDE_CAND]) {
        DestroyWindow(g.part[LZ_GUI_SIDE_CAND]);
        g.part[LZ_GUI_SIDE_CAND] = NULL;
    }
    if (g.cand_title) { DestroyWindow(g.cand_title); g.cand_title = NULL; }

    if (!lz_gui_model_ready(&g.mdl)) return;   /* LZ_GUI_PANEL_NONE */

    /* "Candidates (N)" or the Chinese equivalent, N = 0 until the
       first WM_APP_INSPECT arrives - repaint_candidates keeps this
       current from there. Created
       before the LISTBOX itself only because that reads more
       naturally top to bottom; relayout is what actually decides
       which one sits where in LZ_GUI_SIDE_CAND's rect. */
    {
        char title[64];
        sprintf(title, "%s (0)", lz_str_display(LZ_STR_SIDE_CAND));
        g.cand_title = CreateWindowExA(
            0, "STATIC", title, WS_CHILD | WS_VISIBLE | SS_LEFTNOWORDWRAP,
            0, 0, 0, 0, hwnd, (HMENU)ID_CAND_TITLE, g.inst, NULL);
        /* create_children's own mandatory-parts loop is what applies
           this to every part BUILT THERE - these two do not exist yet
           at that point (this function is what creates them, later),
           so they apply it themselves. */
        if (g.cand_title) {
            apply_font(g.cand_title);
            g.cand_title_untheme_ok = lz_ui_untheme(g.cand_title);
        }
    }

    /* LBS_USETABSTOPS: without this bit LB_SETTABSTOPS below is
       accepted but does nothing - '\t' never expands, and the
       probability column lands glued straight onto the token text
       ("<|im_end|>1.0000", from a user's own screenshot). The style bit
       is what tells the control to honour tab stops at all; the call
       below only supplies WHERE they are. */
    g.part[LZ_GUI_SIDE_CAND] = CreateWindowExA(
        lz_ex_style(WS_EX_CLIENTEDGE), "LISTBOX", "",
        WS_CHILD | WS_VISIBLE | LBS_NOTIFY | LBS_NOINTEGRALHEIGHT |
        LBS_USETABSTOPS | WS_VSCROLL,
        0, 0, 0, 0, hwnd, (HMENU)ID_CAND, g.inst, NULL);
    if (g.part[LZ_GUI_SIDE_CAND]) {
        int tabs[1];
        apply_font(g.part[LZ_GUI_SIDE_CAND]);
        /* Untheme too - the third control in this file to ship without
           it (the input box, the settings dialog's six, this one) - see
           the enumeration gate further down in this
           function's own selftest for why that pattern is not supposed
           to be able to recur silently again. */
        g.cand_untheme_ok = lz_ui_untheme(g.part[LZ_GUI_SIDE_CAND]);
        tabs[0] = LZ_GUI_CAND_TAB;
        /* Two columns via a real tab stop, not "STATIC + space
           padding" - SimSun's ASCII is monospace but a CJK glyph is
           double-width, so a space-padded column goes crooked the
           instant a Chinese token shows up. Measured, not assumed
           (team-lead's own screenshot, a real model, 160px sidebar):
           60 gives clean two-column output for real token/probability
           pairs, nothing more to correct here. */
        SendMessage(g.part[LZ_GUI_SIDE_CAND], LB_SETTABSTOPS, 1,
                   (LPARAM)tabs);
    }

    num_experts = g.mdl.model.config.num_experts;
    if (num_experts > 0) {
        /* All four kinds - see g.elamp_bmp's own field comment for the
           measurement behind OFF/GREEN/AMBER/RED. */
        g.elamp_bmp[LZ_LAMP_OFF]   = lz_mapped_bitmap(g.inst, IDB_LAMP_OFF);
        g.elamp_bmp[LZ_LAMP_READY] = lz_mapped_bitmap(g.inst, IDB_LAMP_READY);
        g.elamp_bmp[LZ_LAMP_BUSY]  = lz_mapped_bitmap(g.inst, IDB_LAMP_BUSY);
        g.elamp_bmp[LZ_LAMP_ERROR] = lz_mapped_bitmap(g.inst, IDB_LAMP_ERROR);
        for (i = 0; i < 16; i++) {
            /* WS_EX_STATICEDGE (user request: a Passat B2 instrument-
               cluster look wants a frame around each lamp, not a bare
               bitmap floating on the sidebar background). lz_ex_style
               strips it below the 3.51 floor, same as WS_EX_CLIENTEDGE
               elsewhere in this file.

               The border is NONCLIENT, so the cell has to be BIGGER
               than the bitmap by its width - see LZ_GUI_ELAMP_PX. A
               STATIC clips the bitmap to its client area, so at 14 the
               14x14 lamp loses its right two columns and bottom two
               rows and the disc comes out sliced flat on two sides. A
               dashboard lamp is a whole circle behind a frame, and a
               frame that eats the lamp is not the same thing seen
               small. */
            g.elamp[i] = CreateWindowExA(
                lz_ex_style(WS_EX_STATICEDGE), "STATIC", "",
                WS_CHILD | WS_VISIBLE | SS_BITMAP,
                0, 0, LZ_GUI_ELAMP_PX, LZ_GUI_ELAMP_PX,
                hwnd, (HMENU)(LZ_IPTR)(ID_ELAMP0 + i), g.inst, NULL);
            /* OFF, not READY - no LZInspect data has arrived for this
               model yet (that only starts once a generation actually
               runs), and an unlit lamp is the correct read of "nothing
               known yet", the same way a fresh activity lamp starts
               dark. repaint_lamps is what ever calls STM_SETIMAGE with
               a lit bitmap again. */
            if (g.elamp[i] && g.elamp_bmp[LZ_LAMP_OFF])
                SendMessage(g.elamp[i], STM_SETIMAGE, IMAGE_BITMAP,
                           (LPARAM)g.elamp_bmp[LZ_LAMP_OFF]);
            g.elamp_untheme_ok[i] = lz_ui_untheme(g.elamp[i]);
        }
    }

    relayout(hwnd);
}

/* Switch the interface language while the window is up.
 *
 * The string table can be re-initialised in one call, but a control does
 * not re-read it: every string below was FROZEN into a window at
 * creation time. So this function is really the list of those places,
 * and the failure mode when one is missing is a control still speaking
 * the old language with nothing anywhere reporting a problem. The
 * selftest therefore switches, reads several of them back, and switches
 * again - a language switch that half works looks exactly like one that
 * works if you only check the title bar.
 *
 * The TRANSCRIPT is deliberately left alone. It is conversation, not
 * furniture: retranslating what the model said is not possible, and
 * clearing it to keep the window internally consistent would throw away
 * the user's session for a cosmetic reason.
 *
 * Neither is the RUNNING JOB touched. Nothing in the worker reads the
 * table; the strings it can produce (engine errbuf) are looked up by
 * src/err.c against its OWN language, which lz_set_error_lang selects -
 * hence the call below, without which the buttons change language and
 * the error boxes do not. */
static void apply_language(HWND hwnd, int english) {
    int i;
    if (english == lz_str_lang_english()) return;
    lz_str_init(english);
    lz_set_error_lang(english);
    /* Before every control is re-touched below, same reason as
       WinMain's own call: the font has to be the new language's before
       WM_SETFONT hands it out, or every control would be told to keep
       using the font the OLD language built. */
    lz_ui_set_font_lang(english);
    /* Same loop shape create_children uses (LZ_GUI_SPLIT is background,
       not a control; LZ_GUI_TOOLBAR is skipped here on purpose - it
       gets REBUILT a few lines down and that call site already sends
       it the new font via apply_font(fresh)). WM_SETFONT is enough for
       every part except the transcript, so lz_richedit_use_font is the
       one extra call - see its own comment on why WM_SETFONT alone
       would leave English text in the transcript still drawing with
       whatever RichEdit substituted for the old font's script. */
    for (i = 0; i < LZ_GUI_PART_COUNT; i++) {
        if (i == LZ_GUI_SPLIT || i == LZ_GUI_TOOLBAR || !g.part[i]) continue;
        apply_font(g.part[i]);
    }
    if (g.part[LZ_GUI_TRANSCRIPT])
        lz_richedit_use_font(g.part[LZ_GUI_TRANSCRIPT], lz_ui_font());
    /* g.status_h is a MEASUREMENT, not a constant that only needed
       taking once - comctl32 recomputes a status bar's height from
       whatever font it currently has, and the loop above just gave it
       a new one. Skipping this leaves relayout() below placing the
       status bar (and everything measured against it) at the OLD
       language's height, which on a host where the two languages
       resolve to genuinely different fonts overlaps the input box -
       the same failure the creation-time measurement exists to
       prevent, at the one other place the active font changes after
       the window exists. Must run after the apply_font loop above,
       same ordering requirement measure_status_h's own comment
       explains for create_children. */
    measure_status_h();

    push_caption();
    build_menu_bar(hwnd);          /* also moves the radio tick */
    if (g.part[LZ_GUI_SIDE_NAME])
        SetWindowTextA(g.part[LZ_GUI_SIDE_NAME],
                       lz_str_display(LZ_STR_APP_TITLE));
    if (g.part[LZ_GUI_SEND])
        SetWindowTextA(g.part[LZ_GUI_SEND],
                       lz_str_display(LZ_STR_BTN_SEND));

    /* The tool strip is REBUILT, not relabelled. Its captions went in
       through TB_ADDSTRING, which appends to a pool the buttons index
       into - there is no message that replaces one. */
    if (g.part[LZ_GUI_TOOLBAR]) {
        HWND fresh = create_toolbar(hwnd);
        if (fresh) {
            DestroyWindow(g.part[LZ_GUI_TOOLBAR]);
            g.part[LZ_GUI_TOOLBAR] = fresh;
            apply_font(fresh);
            /* No lz_ui_untheme call here either - see create_children's
               own comment on LZ_GUI_TOOLBAR for why it stays themed on
               purpose. */
            /* A fresh strip has every button live, so the two commands
               that follow the window's state are re-applied. Written as
               the same calls start_job/finish_job/create_children make
               (ID_SEND unconditionally 1 - see create_children's own
               comment for why "no model" stopped meaning "grey Send"),
               so this cannot disagree with them. */
            cmd_enable(ID_SEND, 1);
            cmd_enable(IDM_STOP_GEN, g.job_kind == JOB_GENERATE);
            /* The rollback three have a button too now (Retry), and
               build_menu_bar's own call further up happened BEFORE this
               strip existed. */
            rollback_sync();
        }
    }

    /* The resting status line is a COPY of display bytes, so it cannot
       be transcoded where it lies - it is re-derived from the state it
       was describing. A job in flight has written a transient over it;
       that one is restored from idle_status by finish_job, in the new
       language, which is why this is safe to do mid-job.
       set_idle_TEXT, though: a language switch re-spells what the
       window is, it does not change it. set_idle_status would also
       retire the throughput cell, and it does not come back until the
       NEXT job starts - so switching language mid-generation used to
       cost that job its rate display for good, and with stream_ms=0
       (no token timer) left the bar reading "ready" while tokens were
       still arriving. */
    set_idle_text(lz_str_utf8(lz_gui_model_ready(&g.mdl)
                                 ? LZ_STR_STATE_READY
                                 : LZ_STR_STATE_NO_MODEL));
    relayout(hwnd);
}

/* The last path segment (after the last '/' or '\\'), same split
   lz_gui_model_name does for the model directory. Used for the title bar's
   chat segment, which is a filename, not a full path. */
static const char *path_tail(const char *p)
{
    const char *last = p;
    for (; *p; p++)
        if (*p == '/' || *p == '\\') last = p + 1;
    return last;
}

/* Assemble and push the title-bar's three segments. The caption module
   knows nothing about chatfile / modelload / localized_strings - this is
   the single place the three GBK segments are gathered and handed over. */
static void push_caption(void)
{
    LZCaption c;
    const char *m;

    memset(&c, 0, sizeof c);
    lstrcpynA(c.brand, lz_str_display(LZ_STR_APP_TITLE), LZ_CAP_BRAND);
    lstrcpynA(c.chat,
              g.chat_name[0] ? g.chat_name
                             : lz_str_display(LZ_STR_CAPTION_UNTITLED),
              LZ_CAP_CHAT);
    m = lz_gui_model_ready(&g.mdl) ? lz_gui_model_name(&g.mdl) : "";
    lstrcpynA(c.model, m ? m : "", LZ_CAP_MODEL);
    lz_caption_set(&c);
}

/* Append a whole line to the transcript, given UTF-8, followed by a
 * BLANK line (user request: breathing room between turns,
 * the same "content ends, then one more \r\n" shape sys_line_fmt
 * already gives system messages). Both of this function's own callers
 * are turn CONTENT - do_send's user turn and load_chat_from's replay
 * loop - so this is where that spacing lives rather than at each call
 * site; the assistant's own reply ending is a THIRD, separate site
 * (finish_job's JOB_GENERATE branch, which streams content in over
 * time via transcript_push rather than calling this function once)
 * and gets the identical treatment there.
 *
 * TRAILING, not a leading blank line before the NEXT turn's own
 * header - team-lead's own reading of the request was "before each
 * header", with an explicit exception carved out for the very first
 * one (nothing needs to precede it, or a system line's own trailing
 * blank line already does). A trailing blank line after every turn's
 * content produces the exact same picture without that exception: the
 * transcript starts empty, so there is nothing for a first turn to
 * trail after either. */
static void transcript_line(const char *utf8, int len) {
    transcript_push(utf8, len);
    transcript_push("\r\n\r\n", 4);
    transcript_end();
}

/* Send whatever is in the input box.
 *
 * The control holds GBK (it is an ANSI control), the engine wants UTF-8,
 * so the conversion happens here and the UTF-8 is what reaches history.
 * The reverse direction - reading the reply back out of the transcript -
 * is the one gui/session.h refuses to do, for the same reason in mirror
 * image. */
/* A system line: no speaker label, its own colour, its own line, PLUS
 * one blank line after it (user report: "model loaded:
 * ..." sat flush against the very next turn's own speaker header,
 * with no separation at all).
 *
 * THE single exit point for every system message this window ever
 * shows - a formatted-string counterpart to sys_line below, for the
 * handful of callers that have arguments to fill into an
 * LZ_STR_SYS_xxx or LZ_STR_ERR_xxx pattern before the text is ready
 * (sys_line itself only ever takes a bare id because most callers
 * have nothing to fill in).
 * Both END UP HERE, and there is exactly one place in this file that
 * mentions LZ_COLOR_SYS - this function's own body - which is what a
 * static check can hold onto: a future system message that coloured
 * itself red directly, bypassing this function, would show up as a
 * SECOND occurrence of that name. */
static void sys_line_fmt(const char *utf8, int len) {
    append_colored_line(utf8, len, LZ_COLOR_SYS);
    append_colored_line("\r\n\r\n", 4, LZ_COLOR_SYS);
}

/* A system line from a plain LZ_STR_* id, no arguments to format -
 * every sys_line(...) call site already reads this way; sys_line_fmt
 * above is for the few that do not. */
static void sys_line(LZStr id) {
    const char *u = lz_str_utf8(id);
    sys_line_fmt(u, (int)strlen(u));
}

/* The prefix drop/arm/clear trio lives in gui/session.c as
 * lz_gui_session_prefix_drop / _arm / _clear. See session.h for the
 * rule the three of them share and for why a headless gate has to be
 * able to reach them. */

static void open_model(HWND hwnd);
static void open_model_dir(HWND hwnd, const char *dir);

/* Write the conversation to a file, GBK, CRLF, from HISTORY rather than
 * from the transcript control - see gui/chatfile.h for what is and is
 * not in that file, and why.
 *
 * Split from the menu handler so the selftest can drive it: a path that
 * only exists behind a modal dialog is a path that only gets tested by
 * hand, and this one has two details worth a gate - the header line and
 * RichEdit's bare-CR line separators (lz_common_crlf still runs here even
 * though the source is history, not RichEdit text, because GBK output
 * must end in CRLF for the same Notepad reason regardless of where it
 * came from).
 *
 * The header line itself is not written here - lz_chatfile_encode
 * already put it there, and LZ_STR_SAVE_HEADER exists only so the
 * selftest has the same bytes to check against without repeating the
 * literal.
 *
 * All THREE conversions below (lz_chatfile_encode, lz_gbk_from_utf8,
 * lz_common_crlf) have snprintf return semantics - the length the FULL
 * result needs, not how much was written - and all three return values
 * are checked. enc is the reachable one: an ordinary
 * conversation stays well under 1 MB, but load_chat_from can read in up
 * to ~1 MB of history from a file (raw[] below), and saving that right
 * back out needs headroom this buffer does not have - lz_chatfile_encode
 * would silently stop at cap and save_chat_to would write a truncated
 * file with no error. gbk and crlf are checked too, even though they
 * cannot overflow given enc already fits (GBK output is never longer
 * than its UTF-8 input, and crlf[] is sized for the worst case of gbk[]
 * doubling): asserting the invariant for real costs one comparison and
 * means nobody has to re-derive the size argument to trust it.
 *
 * Returns 0 on success, 1 if any step would not fit (nothing is
 * written in that case) or the file could not be opened. */
static int save_chat_to(const char *path) {
    static char enc[1024 * 1024 + 8];
    static char gbk[1024 * 1024 + 8];
    static char crlf[2 * (1024 * 1024 + 8)];
    FILE *f;
    int n;

    if (!path) return 1;
    /* The custom system prompt is rendered into the prompt but never
       lives in hist, so encoding hist alone would silently
       drop it from the file. Build the same synthetic prefix the
       render uses - one LZChatMsg in front of the history - and encode
       that, so a saved conversation carries its identity. */
    enc[0] = '\0';
    if (g.set.system[0]) {
        LZChatMsg tmp[LZ_CHAT_HIST_MAX + 1];
        LZChatMsg sys;
        int i;
        sys.role = LZ_ROLE_SYSTEM;
        sys.content = g.set.system;
        sys.len = (int)strlen(g.set.system);
        tmp[0] = sys;
        for (i = 0; i < g.sess.hist.n; i++) tmp[i + 1] = g.sess.hist.msgs[i];
        n = lz_chatfile_encode(tmp, g.sess.hist.n + 1, enc,
                               (int)sizeof enc);
    } else {
        n = lz_chatfile_encode(g.sess.hist.msgs, g.sess.hist.n, enc,
                               (int)sizeof enc);
    }
    if (n >= (int)sizeof enc) return 1;
    n = lz_gbk_from_utf8(enc, (int)strlen(enc), gbk, (int)sizeof gbk, NULL);
    if (n >= (int)sizeof gbk) return 1;
    n = lz_common_crlf(gbk, (int)strlen(gbk), crlf, (int)sizeof crlf);
    if (n >= (int)sizeof crlf) return 1;

    f = fopen(path, "wb");            /* binary: the CRLFs are already in */
    if (!f) return 1;
    fwrite(crlf, 1, strlen(crlf), f);
    fclose(f);
    return 0;
}

/* Read a saved conversation back and make it the live one.
 *
 * On success the history is REPLACED wholesale and the transcript is
 * cleared and redrawn from it. On failure both are left exactly as they
 * were - lz_chatfile_decode's own contract (chatfile.h), and redrawing
 * only after a success is what keeps that promise true here too, not
 * just inside the decoder.
 *
 * Split from the menu handler for the same reason save_chat_to is: the
 * selftest cannot drive a modal GetOpenFileName.
 *
 * Returns 0, or an LZ_ERR_* code with errbuf set. */
static int load_chat_from(const char *path, char *errbuf, int errlen) {
    static char raw[1024 * 1024 + 8];
    static char utf8[2 * (1024 * 1024 + 8)];
    FILE *f;
    long sz;
    size_t n;
    int un, i, rc;

    if (!path) { LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG); return rc; }
    f = fopen(path, "rb");
    if (!f) { LZ_ERR_SET1(rc, errbuf, errlen, LZ_ERR_OPEN_FILE, path);
             return rc; }
    /* Size it before reading rather than discover truncation after the
       fact: fread(raw, 1, sizeof raw - 1, f) alone silently stops at
       the cap on anything bigger, the read-side twin of the encode-side
       bug save_chat_to had (see there) - this file is not necessarily
       one this program wrote (chatfile.h: it is a .txt people edit by
       hand), so an oversized one is an input, not an impossibility.
       feof() right after a full-capacity fread is NOT a reliable
       substitute: a file whose remaining bytes exactly equal the
       request also leaves feof() unset, because nothing ever tried to
       read past it - that would misreport a complete file as
       truncated. Getting the size up front has no such edge case. */
    if (fseek(f, 0, SEEK_END) != 0 || (sz = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        LZ_ERR_SET1(rc, errbuf, errlen, LZ_ERR_OPEN_FILE, path);
        return rc;
    }
    if (sz > (long)(sizeof raw - 1)) {
        fclose(f);
        LZ_ERR_SET1(rc, errbuf, errlen, LZ_ERR_TRUNC, (int)sz);
        return rc;
    }
    n = fread(raw, 1, sizeof raw - 1, f);
    fclose(f);
    raw[n] = '\0';

    un = lz_gbk_to_utf8(raw, (int)n, utf8, (int)sizeof utf8, NULL);
    rc = lz_chatfile_decode(utf8, un, &g.sess.hist, path, errbuf, errlen);
    if (rc != 0) return rc;

    /* A system message in the FILE feeds back into the SETTINGS, not
       into hist (user decision: opening a conversation writes it back
       into the settings). The render path
       synthesises the system prompt at render time from the settings,
       so a system message parked in hist index 0 would be a SECOND
       system - and lz_chat_render refuses any system past index 0
       (LZ_ERR_SYSTEM_FIRST). A file this program saved carries the
       identity at index 0; pull it out and clear it. A hand-edited
       file could put one anywhere; it still becomes the setting, which
       is the closest honest reading. */
    if (g.sess.hist.n > 0 &&
        g.sess.hist.msgs[0].role == LZ_ROLE_SYSTEM) {
        const LZChatMsg *m = &g.sess.hist.msgs[0];
        int mlen = m->len < 0 ? (int)strlen(m->content) : m->len;
        if (mlen > LZ_COMMON_SYSTEM_MAX) mlen = LZ_COMMON_SYSTEM_MAX;
        memcpy(g.set.system, m->content, (size_t)mlen);
        g.set.system[mlen] = '\0';
        /* lz_chat_hist_pop is exactly "drop the newest" - but the
           system is the OLDEST. Drop it by shifting, the way the trim
           does. */
        free(g.sess.hist.owned[0]);
        g.sess.hist.owned[0] = NULL;
        for (i = 1; i < g.sess.hist.n; i++) {
            g.sess.hist.msgs[i - 1] = g.sess.hist.msgs[i];
            g.sess.hist.owned[i - 1] = g.sess.hist.owned[i];
        }
        g.sess.hist.n--;
        g.sess.hist.owned[g.sess.hist.n] = NULL;
        memset(&g.sess.hist.msgs[g.sess.hist.n], 0,
               sizeof g.sess.hist.msgs[g.sess.hist.n]);
        /* The loaded identity is now the active setting, and the
           conversation render must pick it up. */
        lz_gui_session_set_system(&g.sess, g.set.system);
    }

    transcript_clear();
    for (i = 0; i < g.sess.hist.n; i++) {
        const LZChatMsg *m = &g.sess.hist.msgs[i];
        int mlen = m->len < 0 ? (int)strlen(m->content) : m->len;
        int is_user = m->role == LZ_ROLE_USER;
        /* No speaker label exists for LZ_ROLE_SYSTEM - gui/session.c
           never pushes one, so this can only be reached by a hand-edited
           file, and it renders under the assistant label rather than
           being silently dropped. */
        /* with_time 0: a replayed turn was not sent NOW -
           LZChatMsg carries no timestamp to replay honestly (see
           turn_header's own comment) - so this omits the clock rather
           than making one up. */
        turn_header(is_user ? LZ_STR_SPEAKER_USER : LZ_STR_SPEAKER_ASSISTANT,
                   is_user ? LZ_COLOR_USER : LZ_COLOR_ASSISTANT, 0);
        turn_indent();
        transcript_line(m->content, mlen);
    }
    update_ctx_cell();          /* the loaded conversation's size */
    /* History was just replaced wholesale (lz_chatfile_decode above,
       not lz_gui_session_reset - that would also wipe the hist this
       function just built), so the prefix cached before this load does
       not correspond to what is on screen. Direct
       call, not lz_gui_session_reset: that one also clears hist/
       reply_len, which here would undo the decode two lines up.
       CLEAR and not drop - the model did not change, only the history
       did (session.h's rule). */
    lz_gui_session_prefix_clear(&g.sess);
    /* transcript_clear above forgot the turn positions, and the turns
       redrawn since were drawn by this loop rather than by do_send, so
       nothing recorded where they begin. The three rollback commands
       stay greyed for a loaded conversation until the user sends
       something of their own - see rollback_ready's own comment for
       why that is the honest answer rather than a limitation to work
       around. */
    rollback_sync();
    return 0;
}

static void save_chat(HWND hwnd) {
    char path[MAX_PATH];
    /* Stop first. Saving a conversation that is
       still growing writes a file that matches nothing the user ever
       saw. Drained, not just joined: a stopped job's tail (the
       WM_APP_GEN_DONE backfill into g.sess) must land before this
       function reads history, not whenever the queue next happens to be
       pumped - see worker.h. */
    lz_worker_join_drain(hwnd);
    if (!lz_pick_save_file(hwnd, lz_str_display(LZ_STR_DLG_SAVE_CHAT_TITLE),
                           lz_str_display(LZ_STR_DLG_TEXT_FILTER), "*.txt",
                           "txt", path, (int)sizeof path))
        return;
    if (save_chat_to(path) != 0)
        MessageBoxA(hwnd, lz_str_display(LZ_STR_ERR_SAVE_FAILED),
                    lz_str_display(LZ_STR_ERR_TITLE), MB_ICONHAND | MB_OK);
    else {
        lstrcpynA(g.chat_name, path_tail(path), LZ_CAP_CHAT);
        push_caption();
    }
}

/* Open a saved conversation and make it the live one.
 *
 * Deliberately NOT lz_gui_session_reset before the decode: reset clears
 * history, and load_chat_from has already put the new turns there by
 * the time it returns success - resetting first would either be
 * redundant on success or, on a failed decode, throw away whatever the
 * user had instead of leaving it alone. "the decode succeeding is what
 * makes it live" is the whole ordering. */
static void open_chat(HWND hwnd) {
    char path[MAX_PATH];
    char err[1024];

    /* Drained, not just joined - see worker.h. load_chat_from below
       replaces g.sess.hist outright, but only after a decode succeeds;
       an undrained GEN_DONE arriving afterward would still backfill a
       stale reply into whichever history happens to be live then.
       Unlike IDM_CLEAR, this path genuinely is at risk of that
       backfill landing in hist, not just the transcript: nothing here
       calls lz_gui_session_reset (load_chat_from's contract is to
       leave history alone on failure, so resetting first would throw
       away a conversation the user still has on a cancel or a bad
       file), so g.sess.reply_len is never zeroed the way IDM_CLEAR's
       handler zeroes it - a belated lz_gui_session_end would find
       reply_len > 0 and actually push. A modal dialog's own internal
       message pump tends to drain the queue - including this window's
       WM_APP_GEN_DONE - before it returns; that is incidental, not a
       guarantee, which is why this call is drained explicitly rather
       than left to that pump. */
    lz_worker_join_drain(hwnd);
    if (!lz_pick_open_file(hwnd, lz_str_display(LZ_STR_DLG_OPEN_CHAT_TITLE),
                           lz_str_display(LZ_STR_DLG_TEXT_FILTER), "*.txt",
                           path, (int)sizeof path))
        return;
    if (load_chat_from(path, err, (int)sizeof err) != 0)
        MessageBoxA(hwnd, err, lz_str_display(LZ_STR_ERR_TITLE),
                    MB_ICONHAND | MB_OK);
    else {
        lstrcpynA(g.chat_name, path_tail(path), LZ_CAP_CHAT);
        push_caption();
    }
}

/* Run a slash command. Returns 1 when the line WAS one, whatever the
 * outcome - the caller must not then send it as a message.
 *
 * Every branch goes through the same function the menu item goes
 * through, so the menu and the commands are one source. A second
 * implementation of "clear the
 * conversation" is a second thing to forget to join the worker in. */
static int do_command(HWND hwnd, const char *utf8, int len) {
    const char *arg = NULL;
    int alen = 0;
    LZCmd c = lz_common_parse_command(utf8, len, &arg, &alen);

    if (c == LZ_CMD_NONE) return 0;

    switch (c) {
    case LZ_CMD_HELP:
        sys_line(LZ_STR_HELP_BODY);
        break;
    case LZ_CMD_CLEAR:
        SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_CLEAR, 0), 0);
        break;
    case LZ_CMD_STOP:
        SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_STOP_GEN, 0), 0);
        break;
    case LZ_CMD_LOAD:
        /* An argument means "this path"; no argument means "ask me",
           which is the menu item. */
        if (alen > 0) {
            char dir[512];
            int n = alen < (int)sizeof dir - 1 ? alen : (int)sizeof dir - 1;
            memcpy(dir, arg, (size_t)n);
            dir[n] = '\0';
            /* Drained, not just joined - see worker.h. About to replace
               g.mdl and start a load; a GEN_DONE left in the queue would
               otherwise arrive after start_job has already changed
               g.job_kind, and finish_job would then read the NEW job's
               kind for the OLD job's result. */
            lz_worker_join_drain(hwnd);
            if (!lz_gui_model_dir_ok(dir)) {
                sys_line(LZ_STR_ERR_NO_MODEL_BIN);
                break;
            }
            start_model_load(hwnd, dir);
        } else {
            open_model(hwnd);
        }
        break;
    case LZ_CMD_TEMP: {
        char num[32], line[160];
        float t;
        int n = alen < (int)sizeof num - 1 ? alen : (int)sizeof num - 1;
        memcpy(num, arg, (size_t)n);
        num[n] = '\0';
        /* sscanf rather than atof: atof cannot distinguish "no number"
           from "the number zero", and zero is a legal temperature. */
        if (n == 0 || sscanf(num, "%f", &t) != 1 ||
            lz_common_settings_set_temp(&g.set, t) != 0) {
            sys_line(LZ_STR_ERR_BAD_TEMP);
            break;
        }
        apply_settings();
        sprintf(line, lz_str_utf8(LZ_STR_SYS_TEMP_SET), num);
        /* Through sys_line_fmt, same reasoning as the "model loaded"
           call site - this is the second system message that must not
           bypass it. */
        sys_line_fmt(line, (int)strlen(line));
        break;
    }
    case LZ_CMD_THINK: {
        int on = lz_common_parse_onoff(arg, alen);
        if (on < 0) { sys_line(LZ_STR_ERR_BAD_ONOFF); break; }
        lz_common_settings_set_think(&g.set, on);
        apply_settings();
        sys_line(on ? LZ_STR_SYS_THINK_ON : LZ_STR_SYS_THINK_OFF);
        break;
    }
    case LZ_CMD_SAVE:
        SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_SAVE_CHAT, 0), 0);
        break;
    default: {
        char line[256];
        int n = alen;
        (void)n;
        sprintf(line, lz_str_utf8(LZ_STR_ERR_UNKNOWN_CMD), "");
        /* Through sys_line_fmt - the third system message that must
           not bypass it. */
        sys_line_fmt(line, (int)strlen(line));
        break;
    }
    }
    SetWindowTextA(g.part[LZ_GUI_INPUT], "");
    return 1;
}

/* Everything between the input control and the conversation, as one
   function so the selftest can walk the same path the user does.
   Returns the UTF-8 length, or <= 0 when there is nothing to send.

   The lz_common_lf step is not cosmetic. A multi-line EDIT separates its
   lines with CRLF, and without this conversion the CR would go straight
   into LZChatMsg.content, into the prompt, and into the saved file -
   where gui/chatfile.c's decoder would take it for part of the line
   separator and hand back one byte less than it was given. */
static int read_input_utf8(char *out, int cap) {
    static char gbk[16384];
    static char raw[49152];
    int n, un;

    n = GetWindowTextA(g.part[LZ_GUI_INPUT], gbk, (int)sizeof gbk);
    if (n <= 0) return 0;
    un = lz_gbk_to_utf8(gbk, n, raw, (int)sizeof raw, NULL);
    if (un <= 0) return un;
    return lz_common_lf(raw, un, out, cap);
}

/* AFTER the "assistant: " label, not before: the label is UI chrome,
 * not part of the reply, and must stay plain regardless of think
 * state - do_send (the only caller) already sequences it that way,
 * this is just the seed itself, factored out so the selftest can
 * drive it without a real model or worker job.
 *
 * Derived, not re-decided: see lz_chat_gen_prompt_starts_in_think's
 * own comment in src/chat.h for why this is not `g.sess.think`
 * written out as 0/1 a second time - the exact duplication that
 * file's own comments warn drifts silently the first time either
 * literal in lz_chat_gen_prompt_tail changes shape. */
static void seed_reply_style(void) {
    lz_stream_set_in_think(&g.stream,
                           lz_chat_gen_prompt_starts_in_think(g.sess.think));
}

/* ------------------------------------------ rolling one turn back
 *
 * Three commands - Regenerate, Edit Last Message, Delete Last Exchange -
 * over ONE mechanism, written together on purpose. Done separately they
 * would be three copies of "pop the history, cut the control, forget the
 * cached prefix", and the third copy would be the one that forgot a
 * step. That shape has cost this repository real defects more than once.
 *
 * The three pieces the mechanism is made of:
 *
 *   1. HISTORY  lz_chat_hist_pop (src/chat.h), one message at a time.
 *   2. CONTROL  transcript_truncate to a position recorded when the turn
 *               was drawn, NOT a redraw from history - see that
 *               function's own comment.
 *   3. PREFIX   lz_gui_session_prefix_clear. CLEAR, not drop-then-arm:
 *               the history moved and the model did not, which is
 *               exactly the case session.h's rule names. Clear is what
 *               load_chat_from, the other "history was replaced under a
 *               valid cache" path, already does.
 */

/* The user message the last exchange was an answer to, or NULL.
 *
 * Two positions because a turn can end two ways: normally the reply is
 * the newest message and the question is under it, but a generation
 * stopped before its first token pushes NO assistant turn at all
 * (lz_gui_session_end returns early on reply_len == 0), leaving the
 * question itself newest. Both are exchanges the user can take back. */
static const LZChatMsg *last_user_msg(void) {
    int n = g.sess.hist.n;
    if (n > 0 && g.sess.hist.msgs[n - 1].role == LZ_ROLE_USER)
        return &g.sess.hist.msgs[n - 1];
    if (n > 1 && g.sess.hist.msgs[n - 2].role == LZ_ROLE_USER)
        return &g.sess.hist.msgs[n - 2];
    return NULL;
}

/* Is there an exchange to take back right now?
 *
 * Four conditions, and the two positional ones are not redundant with
 * each other: history says the conversation HAS a last exchange,
 * g.turn_cp says THIS PROGRAM RUN drew it and knows where. A
 * conversation loaded from a .txt satisfies the first and not the
 * second, and rolling it back would cut the control at a position
 * belonging to a conversation this program did not draw. */
static int rollback_ready(void) {
    int n = g.sess.hist.n;
    if (lz_worker_busy()) return 0;
    if (g.turn_cp[0] < 0 || g.turn_cp[1] < 0) return 0;
    if (n <= 0) return 0;
    if (g.sess.hist.msgs[n - 1].role == LZ_ROLE_USER) return 1;
    return n >= 2 && g.sess.hist.msgs[n - 1].role == LZ_ROLE_ASSISTANT &&
           g.sess.hist.msgs[n - 2].role == LZ_ROLE_USER;
}

/* Grey or ungrey all three, from the state above.
 *
 * Regenerate carries one extra condition the other two do not: it ends
 * in a generation, so it needs a model to generate with. Putting that
 * in the GREY STATE rather than in the command's own body is what lets
 * the central cmd_is_enabled guard stop it - no second error box, and
 * no code path that starts a job against weights that are not there. */
static void rollback_sync(void) {
    int on = rollback_ready();
    cmd_enable(IDM_REGEN, on && lz_gui_model_ready(&g.mdl));
    cmd_enable(IDM_EDIT_LAST, on);
    cmd_enable(IDM_DEL_LAST, on);
}

/* Take the conversation back one turn.
 *
 * keep_user 1 leaves the question standing, in history and on screen -
 * that is Regenerate, and the caller draws a fresh assistant header
 * afterwards. keep_user 0 takes the question too, back to where the
 * exchange began, which is what Edit and Delete share.
 *
 * reply_len is deliberately NOT zeroed here. lz_worker_join_drain above
 * guarantees this turn's WM_APP_GEN_DONE has already been dispatched,
 * so lz_gui_session_end has already run and nothing else reads that
 * buffer until the next begin or regen, both of which zero it
 * themselves. (IDM_CLEAR's handler leans on lz_gui_session_reset for
 * the same field precisely because it has an UNdrained case to worry
 * about; this one does not.) */
static void rollback_last(HWND hwnd, int keep_user) {
    LZChatHist *h = &g.sess.hist;

    lz_worker_join_drain(hwnd);
    if (h->n > 0 && h->msgs[h->n - 1].role == LZ_ROLE_ASSISTANT)
        lz_chat_hist_pop(h);
    if (!keep_user && h->n > 0 && h->msgs[h->n - 1].role == LZ_ROLE_USER)
        lz_chat_hist_pop(h);

    transcript_truncate(keep_user ? g.turn_cp[1] : g.turn_cp[0]);
    /* The scanner's per-turn state describes text that is now gone, and
       so does anything the throttle is still holding - the same pair
       transcript_clear resets, for the same reason. */
    lz_stream_init(&g.stream);
    g.tok_n = 0;
    lz_gui_session_prefix_clear(&g.sess);

    if (keep_user) g.turn_cp[1] = -1;   /* the caller re-records it */
    else turn_cp_forget();
    update_ctx_cell();
    rollback_sync();
}

/* Draw a turn's header and remember where it started.
 *
 * Split out of do_send so that recording the position and drawing the
 * thing it points at cannot be separated: a header written without its
 * cp is a turn that silently cannot be rolled back, and nothing on
 * screen would say so. */
static void turn_begin_user(const char *utf8, int un) {
    g.turn_cp[0] = transcript_cp_end();
    /* The assistant half of this turn has not begun. Left stale, a
       failed lz_gui_session_begin below would leave [0] pointing at
       this turn and [1] at the PREVIOUS one - two positions from two
       different exchanges, which is the one way this pair can be
       wrong without being obviously wrong. */
    g.turn_cp[1] = -1;
    /* turn_header + a two-space indent (user request - see
       turn_header's own comment for the header/content shape and why
       the brackets in the user's sketch are read as placeholders).
       with_time 1: this turn is happening right now. */
    turn_header(LZ_STR_SPEAKER_USER, LZ_COLOR_USER, 1);
    turn_indent();
    transcript_line(utf8, un);
}

static void turn_begin_assistant(void) {
    g.turn_cp[1] = transcript_cp_end();
    turn_header(LZ_STR_SPEAKER_ASSISTANT, LZ_COLOR_ASSISTANT, 1);
    turn_indent();
}

static void do_regen(HWND hwnd) {
    static char err[1024];

    rollback_last(hwnd, 1);
    turn_begin_assistant();
    seed_reply_style();
    if (lz_gui_session_regen(&g.sess, err, (int)sizeof err) != 0) {
        MessageBoxA(hwnd, err, lz_str_display(LZ_STR_ERR_TITLE),
                    MB_ICONHAND | MB_OK);
        rollback_sync();
        return;
    }
    start_job(hwnd, lz_gui_session_job, &g.sess, JOB_GENERATE);
}

static void do_edit_last(HWND hwnd) {
    /* Sized off do_send's own input buffer: whatever fits going in has
       to fit coming back out. static, not stack - iron law six. */
    static char gbk[49152];
    static char crlf[2 * 49152 + 8];
    const LZChatMsg *m = last_user_msg();
    int ulen, n;

    if (!m || !m->content) return;
    ulen = m->len < 0 ? (int)strlen(m->content) : m->len;
    /* Read BEFORE the pop: lz_chat_hist_pop frees the copy this points
       into, and the bytes are the whole point of this command. */
    n = lz_gbk_from_utf8(m->content, ulen, gbk, (int)sizeof gbk, NULL);
    if (n >= (int)sizeof gbk) return;
    /* Back to CRLF. read_input_utf8 ran lz_common_lf on the way in, and a
       multi-line EDIT draws a lone LF as a box rather than a line
       break - the round trip is only closed if both halves exist. */
    n = lz_common_crlf(gbk, (int)strlen(gbk), crlf, (int)sizeof crlf);
    if (n >= (int)sizeof crlf) return;

    rollback_last(hwnd, 0);
    SetWindowTextA(g.part[LZ_GUI_INPUT], crlf);
    /* Caret at the end, nothing selected: the user asked to EDIT this
       line, and a fully selected block that the next keystroke wipes
       out is the opposite of that.
       A huge start AND end rather than `n`, deliberately - n is a BYTE
       count, and whether this ANSI EDIT counts its selection in bytes
       or in characters is the same question g.turn_cp answers for the
       transcript, with the same GBK trap behind it. Out-of-range
       positions are clamped to the end of the text, so a number no
       text can reach means "the end" without having to know which unit
       it is in. Measured, not assumed - st_rollback reads the caret
       back and compares it against where select-all says the end is,
       and that measurement answers the question too: the caret gate's
       own mutation prints "end 5" for "q2 " plus two Chinese
       characters, which is CHARACTERS (7 bytes in GBK). So `n` really
       would have been the wrong number, not merely an unproven one. */
    SendMessage(g.part[LZ_GUI_INPUT], EM_SETSEL,
                (WPARAM)0x7FFFFFFF, (LPARAM)0x7FFFFFFF);
    SetFocus(g.part[LZ_GUI_INPUT]);
}

static void do_send(HWND hwnd) {
    static char utf8[49152];
    static char err[1024];
    int un, rc;

    if (lz_worker_busy()) return;
    un = read_input_utf8(utf8, (int)sizeof utf8);
    if (un <= 0) return;

    /* Commands come first, and BEFORE the model check: /load and /help
       are exactly what a user with no model loaded needs to reach. */
    if (do_command(hwnd, utf8, un)) return;

    if (!lz_gui_model_ready(&g.mdl)) {
        /* LZ_STR_ERR_NO_MODEL_LOADED, not LZ_STR_ERR_NO_MODEL_BIN - this
           check names no directory at all (lz_gui_model_ready asks
           "is a model loaded into this process", not "does some path
           have model.bin in it"). A user with a perfectly good model
           directory who has simply not opened it yet must not be told
           their directory is missing a file it has - the wrong wording
           also drew the box at screen centre rather than over the main
           window, which read as a crash. */
        MessageBoxA(hwnd, lz_str_display(LZ_STR_ERR_NO_MODEL_LOADED),
                    lz_str_display(LZ_STR_ERR_TITLE),
                    MB_ICONEXCLAMATION | MB_OK);
        return;
    }

    turn_begin_user(utf8, un);

    rc = lz_gui_session_begin(&g.sess, utf8, un, err, (int)sizeof err);
    if (rc != 0) {
        MessageBoxA(hwnd, err, lz_str_display(LZ_STR_ERR_TITLE),
                    MB_ICONHAND | MB_OK);
        /* turn_begin_user just invalidated the assistant position, so
           this redraws the three rollback commands as greyed rather
           than leaving them offering to roll back a turn that never
           started. */
        rollback_sync();
        return;
    }
    SetWindowTextA(g.part[LZ_GUI_INPUT], "");
    turn_begin_assistant();
    seed_reply_style();
    start_job(hwnd, lz_gui_session_job, &g.sess, JOB_GENERATE);
}

/* Neither modifier is down - the one condition under which Enter sends
 * rather than inserts a newline. Shift joined Ctrl here:
 * Shift+Enter is the near-universal "insert a newline" binding in a
 * chat box, and without this check it would fall through to "send"
 * like plain Enter - almost certainly what a user who reported a
 * "crash" actually hit (see do_send's LZ_STR_ERR_NO_MODEL_LOADED:
 * that box is real, just misdiagnosed as a crash because of where it
 * draws and what it says). */
static int input_send_key(void) {
    return GetKeyState(VK_CONTROL) >= 0 && GetKeyState(VK_SHIFT) >= 0;
}

/* Enter sends, Ctrl+Enter or Shift+Enter inserts a newline. The
   composition test lives in gui/compat40.c - and it is NOT
   ImmGetOpenStatus: that answers "is an IME switched on", and a
   Chinese user has one on permanently.

   WM_CHAR is intercepted here too, not only WM_KEYDOWN (added alongside
   the Shift check above). TranslateMessage - in the message
   loop, well before this window procedure ever sees the WM_KEYDOWN -
   already queues a WM_CHAR('\r') for it unconditionally, so swallowing
   WM_KEYDOWN below does not un-queue that WM_CHAR: it still arrives, on
   the NEXT pass through the message loop, ahead of the WM_COMMAND this
   function posts (PostMessage appends to the tail; TranslateMessage's
   own post happened earlier in the same loop iteration, before
   DispatchMessage even reached here). Left unhandled, a plain-Enter
   send picked up a trailing CRLF the edit control inserted from that
   queued WM_CHAR before do_send ever read the box - every message sent
   by pressing Enter carried a stray newline. Checking '\n' alongside
   '\r' is defensive, not load-bearing: VK_RETURN's own translation is
   documented to produce '\r' alone, never a separate '\n' - but the
   edit control inserts a break for either one (measured, both send and
   receive), so blocking only one would leave a matching gap open. */
static LRESULT CALLBACK input_subclass(HWND h, UINT msg, WPARAM wp,
                                       LPARAM lp) {
    if (msg == WM_KEYDOWN && wp == VK_RETURN) {
        if (input_send_key() && !lz_ime_composing(h)) {
            PostMessage(GetParent(h), WM_COMMAND, MAKEWPARAM(ID_SEND, 0), 0);
            return 0;
        }
    }
    if (msg == WM_CHAR && (wp == '\r' || wp == '\n')) {
        if (input_send_key() && !lz_ime_composing(h))
            return 0;
    }
    return CallWindowProc(g.input_proc, h, msg, wp, lp);
}

/* The same two commands the Edit menu carries, as a popup.
 *
 * Built each time rather than kept: it is two items, and a cached HMENU
 * would be one more thing apply_language has to remember to rebuild.
 *
 * The accelerator tail is cut off. A context menu in this era did not
 * print "Ctrl+C" in a right-hand column - the menu bar did - and the
 * string table carries one form for both. */
static void show_edit_popup(HWND ctl, POINT pt) {
    HMENU m = CreatePopupMenu();
    int i;
    static const int ITEMS[2] = { IDM_COPY, IDM_SELECT_ALL };
    static const int LABELS[2] = { LZ_STR_MENU_COPY, LZ_STR_MENU_SELECT_ALL };
    if (!m) return;
    for (i = 0; i < 2; i++) {
        char buf[64];
        char *tab;
        strncpy(buf, lz_str_display((LZStr)LABELS[i]), sizeof buf - 1);
        buf[sizeof buf - 1] = '\0';
        tab = strchr(buf, '\t');
        if (tab) *tab = '\0';
        AppendMenuA(m, MF_STRING | MF_ENABLED, (UINT)ITEMS[i], buf);
    }
    /* Owner is the PARENT, not the control: WM_COMMAND has to land in
       the window procedure that has the cases for these ids. */
    TrackPopupMenu(m, TPM_LEFTALIGN | TPM_RIGHTBUTTON, pt.x, pt.y, 0,
                   GetParent(ctl), NULL);
    DestroyMenu(m);
}

/* The transcript has no context menu of its own - RichEdit does not
   offer one without an OLE callback the rest of this front end has no
   use for - and the input box's built-in one does not carry Copy or
   Select All the way the Edit menu does. This subclass gives both
   controls the SAME popup, the one show_edit_popup builds. */
static LRESULT CALLBACK transcript_subclass(HWND h, UINT msg, WPARAM wp,
                                            LPARAM lp) {
    if (msg == WM_CONTEXTMENU) {
        /* lParam is -1 when the menu came from the keyboard (Shift+F10
           or the menu key), and using those coordinates puts the popup
           in the top-left corner of the screen. */
        POINT pt;
        pt.x = (short)LOWORD(lp);
        pt.y = (short)HIWORD(lp);
        if (lp == (LPARAM)-1) GetCursorPos(&pt);
        show_edit_popup(h, pt);
        return 0;
    }
    return CallWindowProc(g.transcript_proc, h, msg, wp, lp);
}

/* Validate and start loading `dir` - the half of open_model that does not
 * need a folder-browse dialog, so the MRU's WM_COMMAND handler can share
 * it instead of growing its own copy. That copy is exactly the kind of
 * place a second lz_worker_join_drain() call goes missing.
 *
 * `dir` may alias g.mru.item[k] (the MRU handler passes it directly): the
 * only mutation of g.mru here is lz_mru_remove, and it is called only
 * AFTER the directory has already been matched by value, so the pointer
 * is never read again once the array underneath it starts shifting. */
static void open_model_dir(HWND hwnd, const char *dir) {
    /* The unified sequence before anything that touches shared state -
       and a load replaces the LZRunState the worker may be generating
       into. Idempotent, so open_model's own join before the folder
       dialog and this one do not conflict. Drained, not just joined -
       see worker.h: start_job below overwrites g.job_kind for the NEW
       job, so an old GEN_DONE left in the queue would be read by
       finish_job under the wrong kind if it were not flushed first. */
    lz_worker_join_drain(hwnd);
    /* Refuse up front rather than let the engine
       fall back to f32 safetensors and run a 128 MB machine out of
       memory several seconds later. Reached from the MRU, this is also
       how a directory that stopped having a model in it gets dropped
       from the list instead of sitting on the menu doing nothing when
       clicked. */
    if (!lz_gui_model_dir_ok(dir)) {
        lz_mru_remove(&g.mru, dir);
        build_menu_bar(hwnd);
        MessageBoxA(hwnd, lz_str_display(LZ_STR_ERR_NO_MODEL_BIN),
                    lz_str_display(LZ_STR_ERR_TITLE), MB_ICONEXCLAMATION | MB_OK);
        return;
    }
    start_model_load(hwnd, dir);
}

static void open_model(HWND hwnd) {
    char dir[512];
    /* Joined AND drained here too: a folder-browse dialog is modal, and
       starting it while the worker is still generating into the
       RichEdit behind it is the same hazard open_model_dir's own join
       guards against - see worker.h. */
    lz_worker_join_drain(hwnd);
    /* gui/compat40.c owns the choice of dialog: NT 3.51's shell32 has
       no folder browser, so there it asks for model.bin itself and
       keeps the directory. */
    if (!lz_pick_folder(hwnd, lz_str_display(LZ_STR_DLG_OPEN_MODEL_TITLE),
                        lz_str_display(LZ_STR_FILTER_MODEL),
                        "model.bin", dir, (int)sizeof dir))
        return;
    open_model_dir(hwnd, dir);
}

/* Find in the conversation, via comdlg32's FindTextA - the
 * period-appropriate find box rather than a hand-drawn one, and
 * comdlg32 is already one of the four DLLs this deliverable allows.
 *
 * FindText is MODELESS: it returns immediately and talks back through
 * a registered message (g_findmsg). Two consequences, both silent when
 * missed - the dialog either eats every keystroke or answers none:
 *   1. IsDialogMessage(g.find_dlg, &msg) has to run in WinMain's own
 *      loop, or Tab and the dialog's own Enter/Escape never reach it.
 *   2. FINDMSGSTRING is a RegisterWindowMessage value, not a compile-
 *      time constant, so it cannot be a case label in wndproc's
 *      switch - it is an `if` checked before the switch runs, below.
 *
 * FINDREPLACEA itself, and the FR_DIALOGTERM/FR_FINDNEXT flags, are
 * commdlg.h types - a header this file cannot include (see
 * gui/compat40.c's own top comment: it drags in prsht.h, which the
 * 3.51 floor hides). gui/compat40.c already raises WINVER for exactly
 * this reason (the file-dialog wrappers lz_pick_open_file and friends),
 * so lz_find_open/lz_find_parse live there and hand this file nothing
 * but plain C types - this file never names FINDREPLACEA at all.
 * FR_DOWN and FR_MATCHCASE are the one exception: EM_FINDTEXT's own
 * wParam bits are commdlg.h's FR_DOWN/FR_MATCHCASE values BY DESIGN
 * (so a dialog's Flags can be handed to EM_FINDTEXT with no
 * translation), and richedit.h - which this file DOES include - does
 * not redeclare them, so gui/compat40.h hand-declares just those two
 * the same way gui/resource.h hand-declares VK_ESCAPE.
 *
 * g_findmsg itself is declared earlier in this file (ahead of
 * create_children, which registers it) - see the comment there. */
/* comdlg32 writes the user's typed search text directly into this
 * buffer for as long as the dialog is open - it is not a copy-in,
 * copy-out API - so it has to outlive the call that opened the dialog.
 * Static storage, not the stack, for exactly that reason. */
static char g_find_buf[128];

/* Find `needle` in the conversation, starting at `from`.
 *
 * EM_FINDTEXTW when riched20 (RichEdit 2.0+) is loaded, not EM_FINDTEXTA
 * - EM_FINDTEXTA fails for Chinese text even on a machine whose system
 * locale (GetACP()=936, GBK) plainly matches what this file writes
 * everywhere else: the active keyboard layout can be English US
 * (0x04090409) while the layout list also carries Chinese (0x08040804).
 *
 * VERIFIED FACTS, from a controlled create/seed/find matrix
 * (a fresh RichEdit
 * control per row, the three moments varied independently, each
 * result reproduced on repeated runs):
 *   - What EM_REPLACESEL actually STORES is correct regardless of the
 *     active layout at any of the three moments - GetWindowTextA reads
 *     back the right GBK bytes every time, in every row tried,
 *     including with gui/compat40.c's own EM_SETCHARFORMAT(SCF_ALL,
 *     bCharSet=GB2312_CHARSET) reproduced ahead of the text (the exact
 *     call lz_richedit_use_font makes before this control ever holds a
 *     character). This matches what this program actually shows: real
 *     Chinese text displays correctly under an English layout, not as
 *     mojibake.
 *   - EM_FINDTEXTA's own ANSI->Unicode conversion for the search TERM
 *     is the part that is wrong, and it does not track GetACP(), the
 *     layout at window creation, or the layout active at the find call
 *     itself - only the layout active the first time ANSI text was
 *     written into the control (EM_REPLACESEL, append_run above).
 *     Reproduced with EM_SETCHARFORMAT(bCharSet=134) set beforehand
 *     too: identical to the no-charformat matrix, so the document's own
 *     charset setting does not reach this code path either.
 * NOT ESTABLISHED: why the needle conversion is pinned to that one
 * moment specifically, or what riched20 is actually keying it on
 * internally. Do not extend this comment with a mechanism claim
 * unless it is backed by its own discriminating experiment.
 *
 * Practical result: a user typing Chinese through an IME while the
 * active keyboard layout is English - an entirely ordinary way to use
 * one - gets every Chinese Find silently reported "not found" instead
 * of finding the wrong thing. This is the real, shipped Find feature,
 * not only a selftest artifact.
 *
 * The fix sidesteps the whole question instead of chasing it: the
 * needle is converted to UTF-16 with an EXPLICIT code page - 936, the
 * same one this file's GBK convention already assumes everywhere else
 * - and searched with EM_FINDTEXTW, which never goes through
 * EM_FINDTEXTA's ANSI conversion at all, whatever it turns out to be
 * keyed on. Verified to find the needle in all six create/seed/find
 * timing combinations the probe tried, including the two that broke
 * EM_FINDTEXTA - this is exactly why the fix does not need the
 * mechanism nailed down to be correct.
 *
 * riched32 (RichEdit 1.0, the NT 3.51 floor) has no EM_FINDTEXTW at
 * all - it predates Unicode messages entirely - so on that floor, and
 * only there, this still falls back to EM_FINDTEXTA and inherits
 * whatever this bug's real cause turns out to be. Not a silently
 * reintroduced bug: there is no better message riched32 answers, and a
 * Chinese find that sometimes reports "not found" is what the floor
 * actually is on that control.
 *
 * Both messages take the needle as one contiguous buffer, and the
 * needle came out of an ANSI control (the find dialog) or this file's
 * own selftest, so it is GBK on both the ANSI path and as the source
 * for the UTF-16 conversion - nothing upstream of this function
 * converts it.
 *
 * Returns the character offset, or -1. NOT 0 for "not found" - 0 is a
 * legitimate hit at the very start, and conflating them is how a
 * search that finds nothing scrolls the user to the top and looks
 * like it worked. */
/* Byte compare for the RichEdit 1.0 search path.
   Case folding is ASCII-only and DELIBERATELY not locale-aware: folding
   through the locale is the very thing that made EM_FINDTEXT's answer
   depend on the keyboard layout. A byte >= 0x80 is either a GBK lead or
   a trail and is compared exactly - case has no meaning for it, and a
   trail byte can land on 'A'..'Z', which is how a locale-aware fold
   corrupts the pair. */
static int find_eq(const char *a, const char *b, long n, int match_case) {
    long i;
    for (i = 0; i < n; i++) {
        unsigned char x = (unsigned char)a[i], y = (unsigned char)b[i];
        if (x == y) continue;
        if (match_case || x >= 0x80 || y >= 0x80) return 0;
        if (x >= 'A' && x <= 'Z') x = (unsigned char)(x - 'A' + 'a');
        if (y >= 'A' && y <= 'Z') y = (unsigned char)(y - 'A' + 'a');
        if (x != y) return 0;
    }
    return 1;
}

static long transcript_find(const char *needle, long from, int down,
                            int match_case) {
    FINDTEXTA ft;
    WPARAM flags;
    long hit;
    if (!needle || !needle[0] || !g.part[LZ_GUI_TRANSCRIPT]) return -1;
    flags = (WPARAM)((down ? FR_DOWN : 0) | (match_case ? FR_MATCHCASE : 0));

    if (strcmp(lz_richedit_class(), "RichEdit20A") == 0) {
        FINDTEXTW ftw;
        WCHAR needle_w[128];   /* g_find_buf is 128 bytes; GBK->UTF-16
                                   never produces more wchars than
                                   source bytes, so this is never tight */
        int wn = MultiByteToWideChar(936, 0, needle, -1, needle_w,
                                     (int)(sizeof needle_w /
                                           sizeof needle_w[0]));
        if (wn > 0) {
            ftw.chrg.cpMin = from;
            ftw.chrg.cpMax = down ? -1 : 0;
            ftw.lpstrText = needle_w;
            hit = (long)SendMessage(g.part[LZ_GUI_TRANSCRIPT], EM_FINDTEXTW,
                                    flags, (LPARAM)&ftw);
            return hit;
        }
        /* Conversion failed - should not happen for a needle that came
           out of a GBK control to begin with, but fall through to the
           ANSI path rather than silently claiming "not found". */
    }

    /* THE 1.0 PATH SEARCHES THE TEXT ITSELF rather than asking
       EM_FINDTEXT.
       RichEdit 1.0's ANSI find compares through the thread's locale, and
       the result depends on the active keyboard layout: the same needle
       in the same buffer is found under a Chinese layout and not found
       under an English one. That was reproduced directly - the selftest
       forces 00000409, and GetWindowTextA still shows the needle sitting
       at the offset the search reports as absent. Adding FR_MATCHCASE
       does not fix it, so this is not only case folding.
       For an ANSI control a character position IS a byte offset, so a
       plain byte search over the control's own text answers the exact
       question EM_FINDTEXT was being asked, with no locale in it. Find
       is a user action, not a per-token path, so reading the buffer per
       search is affordable; the transcript is capped at
       LZ_TRANSCRIPT_LIMIT and this is heap, not stack. */
    {
        long len = (long)SendMessage(g.part[LZ_GUI_TRANSCRIPT],
                                     WM_GETTEXTLENGTH, 0, 0);
        long nlen = (long)strlen(needle);
        char *buf;
        long i, last;

        (void)ft;
        if (len <= 0 || nlen <= 0 || nlen > len) return -1;
        buf = (char *)malloc((size_t)len + 1);
        if (!buf) return -1;
        buf[0] = '\0';
        SendMessage(g.part[LZ_GUI_TRANSCRIPT], WM_GETTEXT,
                    (WPARAM)(len + 1), (LPARAM)buf);
        last = (long)strlen(buf) - nlen;
        hit = -1;
        if (down) {
            for (i = (from > 0 ? from : 0); i <= last; i++)
                if (find_eq(buf + i, needle, nlen, match_case)) { hit = i; break; }
        } else {
            long start = (from - 1 < last) ? from - 1 : last;
            for (i = start; i >= 0; i--)
                if (find_eq(buf + i, needle, nlen, match_case)) { hit = i; break; }
        }
        free(buf);
        return hit;
    }
}

/* Opens the dialog, or - if one is already open - gives it focus
 * instead of leaking a second one. FindText has no "already open"
 * query of its own; g.find_dlg standing for NULL is the only signal. */
static void open_find(HWND hwnd) {
    if (g.find_dlg) { SetFocus(g.find_dlg); return; }
    g.find_dlg = lz_find_open(hwnd, g_find_buf, (int)sizeof g_find_buf);
}

/* GBK lead-byte range (gui/mru.c's own comment explains why this is a
 * duplicated pair of constants rather than a #include of gbk_tables.h
 * - a ~4700-line generated decode table pulled in for two numbers).
 * Needed here so truncating a token for display never splits a
 * double-byte character in half. */
#define GBK_LEAD_LO 0x81
#define GBK_LEAD_HI 0xFE

/* Truncate `gbk` (NUL-terminated, modified in place) so it renders no
 * wider than `max_px` pixels in the candidate LISTBOX's own current
 * font, appending "..." when it had to cut. Without it a token like
 * "<|im_end|>" (10 ASCII characters) would run past LZ_GUI_CAND_TAB's
 * own tab stop and glue straight onto the probability column.
 *
 * Measured against the REAL rendering width (GetTextExtentPoint32A
 * against lz_ui_font(), the same font apply_font already put on this
 * control) - the same "measure, do not guess" approach the settings
 * dialog's slider/value-box row already established - not a byte or
 * character COUNT: a Chinese character is roughly twice the pixel
 * width of an ASCII one in this font, so a count-based cutoff would
 * both cut Chinese too early and Latin too late.
 *
 * Walks FORWARD from the start, one whole GBK character (1 or 2
 * bytes) at a time, tracking the longest prefix that still fits
 * alongside the ellipsis - not backward from the end, which cannot
 * tell a lead byte from a trail byte in isolation: GBK's trail-byte
 * range (0x40-0xFE) overlaps its lead-byte range (0x81-0xFE), so only
 * a scan that starts from a KNOWN boundary (the string's own start,
 * since it came from a real decode) can stay synchronised - the same
 * resynchronisation reasoning gui/stream.c's own seq_len() and
 * gui/mru.c's own ci_eq() already rely on for their own byte walks. */
static void truncate_gbk_to_width(HWND lb, char *gbk, int max_px) {
    HDC dc;
    HFONT old_font;
    SIZE sz;
    int len = (int)strlen(gbk);

    dc = GetDC(lb);
    if (!dc) return;
    old_font = (HFONT)SelectObject(dc, lz_ui_font());

    if (GetTextExtentPoint32A(dc, gbk, len, &sz) && sz.cx > max_px) {
        SIZE ell;
        int i = 0, last_ok = 0;
        GetTextExtentPoint32A(dc, "...", 3, &ell);
        while (i < len) {
            unsigned char c = (unsigned char)gbk[i];
            int step = (c >= GBK_LEAD_LO && c <= GBK_LEAD_HI && i + 1 < len)
                       ? 2 : 1;
            i += step;
            if (GetTextExtentPoint32A(dc, gbk, i, &sz) &&
                sz.cx + ell.cx <= max_px)
                last_ok = i;
        }
        gbk[last_ok] = '\0';
        strcat(gbk, "...");
    }

    SelectObject(dc, old_font);
    ReleaseDC(lb, dc);
}

/* Which of the four lamp bitmaps a hit count maps to - a three-colour
 * Passat B2 instrument-cluster reading, the user's own request backed
 * by team-lead's real-model measurement: of every expert a token used
 * at all, 60.0% were picked by exactly one MoE layer, 31.5% by two,
 * 7.7% by three, 0.8% by four - see g.elamp_bmp's own field comment
 * and inspect.h's expert_hits comment for the fuller numbers.
 *
 * 0 layers -> OFF, 1 -> GREEN (common), 2 -> AMBER (less common),
 * 3 OR MORE -> RED (rare and therefore worth flagging - a Passat
 * dashboard has three telltale colours, not four, so 3 and 4+ share
 * the top one rather than getting a colour each). Broken out as its
 * own function, not inlined into repaint_lamps' loop, so the
 * selftest can name the exact boundary (>=3, not >=4 or >2) as a
 * value to assert on rather than re-deriving it from control flow. */
static int elamp_tier(unsigned char hits) {
    if (hits == 0) return LZ_LAMP_OFF;
    if (hits == 1) return LZ_LAMP_READY;
    if (hits == 2) return LZ_LAMP_BUSY;
    return LZ_LAMP_ERROR;
}

/* Light the 16 expert lamps from one WM_APP_INSPECT frame's own
 * expert_hits (Part Two's GUI half - the engine half, src/forward.c
 * filling expert_hits via lz_moe_hits_add). hits[i] is how many of
 * this token's MoE layers chose expert i - the union-across-layers
 * reading ("lamp = expert identity, union across all layers") is
 * refined from "any layer at all" to "how many", already computed on
 * the engine side; this function only ever reads the result and maps
 * it through elamp_tier.
 *
 * PER-LAMP, not "is anything lit at all": an all-zero hits array and
 * "this model has no MoE" render as the SAME picture (a row of dark
 * lamps), and the degenerate case (side_panel_sync) already tells
 * those apart by never creating the lamp array at all when
 * num_experts == 0. Once the array DOES exist, a check that only
 * asked "is anything lit" could not tell a genuinely all-zero tally
 * apart from a bug that lit the wrong lamp (or the wrong COLOUR) and
 * left the right one dark - checking every lamp's own tier against
 * its own hit count is what an "is anything lit" reading cannot
 * catch, and now additionally covers the wrong-colour case a plain
 * on/off reading never could even in principle.
 *
 * No-op when g.elamp[0] is NULL - no model loaded, or the loaded one
 * has no MoE - the same "nothing to paint into" reasoning
 * repaint_candidates' own NULL check uses below. */
static void repaint_lamps(const LZInspect *ins) {
    int i;
    if (!g.elamp[0]) return;
    for (i = 0; i < 16; i++) {
        int tier = elamp_tier(ins->expert_hits[i]);
        if (g.elamp[i] && g.elamp_bmp[tier])
            SendMessage(g.elamp[i], STM_SETIMAGE, IMAGE_BITMAP,
                       (LPARAM)g.elamp_bmp[tier]);
    }
}

/* Repaint the candidate LISTBOX from one WM_APP_INSPECT frame.
 *
 * Numbers only came across the wire (LZInspect's own design, iron law
 * one) - turning cand_id[i] into text needs the tokenizer, which is
 * why this lives here and not in gui/worker.c or src/. lz_decode_into
 * is the thread-safe form (its own header comment): the worker thread
 * is running concurrently, sampling and posting the NEXT frame while
 * this one paints, so a decode that shared LZTokenizer's own internal
 * buffer (plain lz_decode) would be a real race, not a theoretical
 * one.
 *
 * No-op if the panel is not up (g.part[LZ_GUI_SIDE_CAND] NULL) - a
 * frame can arrive after the panel was torn down (a reload in
 * flight, the window closing) if the worker posted it before
 * side_panel_sync ran; see gui/worker.h's own ordering notes for why
 * that race is closed for WM_APP_GEN_DONE but this message does not
 * have the same synchronous-drain guarantee end to end. Silently
 * dropping the frame is the correct read of "nothing to paint it
 * into", not a bug to chase. */
static void repaint_candidates(const LZInspect *ins) {
    HWND lb = g.part[LZ_GUI_SIDE_CAND];
    int i, tab_px;
    char title[64];

    if (!lb) return;

    /* LZ_GUI_CAND_TAB (60), converted from dialog template units to
       THIS host's actual pixels via GetDialogBaseUnits() - the same
       conversion LB_SETTABSTOPS itself uses internally, measured
       fresh rather than assumed via LZ_DLU_X (that macro is
       calibrated for "MS Sans Serif 8pt", layout.h's own documented
       assumption, which is not necessarily the SAME font
       GetDialogBaseUnits reports on every host). One DLU is a quarter
       of the average character WIDTH horizontally (the documented
       conversion LB_SETTABSTOPS's own tab positions use). */
    {
        DWORD base = GetDialogBaseUnits();
        tab_px = LZ_GUI_CAND_TAB * (int)LOWORD(base) / 4;
    }

    SendMessage(lb, WM_SETREDRAW, FALSE, 0);
    SendMessage(lb, LB_RESETCONTENT, 0, 0);
    for (i = 0; i < ins->n_cand && i < LZ_INSPECT_CAND_MAX; i++) {
        char utf8[64], gbk[64], row[80];
        int ulen = lz_decode_into(&g.mdl.tok, ins->cand_id[i], utf8,
                                  (int)sizeof utf8, NULL);
        int glen = lz_gbk_from_utf8(utf8, ulen, gbk, (int)sizeof gbk - 1,
                                    NULL);
        gbk[glen] = '\0';
        /* Over-wide tokens truncate with an ellipsis rather than
           running past the tab stop (team-lead's own screenshot:
           "<|im_end|>1.0000" glued together). By RENDERED WIDTH, not
           byte count - see truncate_gbk_to_width's own comment for why
           a Chinese token needs this and a byte cutoff would get it
           wrong either direction. */
        truncate_gbk_to_width(lb, gbk, tab_px);
        /* %.4f, not more: cand_p is a probability in [0,1], and four
           places is already finer than a 14px lamp-sized column can
           usefully show. A literal tab (\t), not spaces - LB_SETTABSTOPS
           above is what the column position actually comes from. */
        sprintf(row, "%s\t%.4f", gbk, (double)ins->cand_p[i]);
        SendMessage(lb, LB_ADDSTRING, 0, (LPARAM)row);
    }
    SendMessage(lb, WM_SETREDRAW, TRUE, 0);
    InvalidateRect(lb, NULL, TRUE);

    /* n_survived, not n_cand: n_cand is
       how many ROWS fit (capped at LZ_INSPECT_CAND_MAX), n_survived is
       the true count after top_k/top_p/min_p, and the title reports
       the latter so "candidates (37)" with 10 rows showing is read as
       truncation, not as the model only having ever produced 10. */
    if (g.cand_title) {
        sprintf(title, "%s (%d)", lz_str_display(LZ_STR_SIDE_CAND),
               ins->n_survived);
        /* experts_truncated (Part Two): n_experts >
           LZ_INSPECT_EXPERT_MAX means expert_hits (a 32-entry array)
           cannot represent every expert this model has, so the lamp
           grid can only ever show a SUBSET - this must not be silent
           (the exact wording of how to say so is left to this call).
           Appended to the SAME title the candidate count already uses
           rather than a new control: new panel chrome is ruled out for
           the normal case, and this is the one abnormal case that
           still has to say SOMETHING - reusing existing text surface
           is the smallest footprint that is not silent. */
        if (ins->experts_truncated) {
            char suffix[64];
            sprintf(suffix, " [%d experts, 32 shown]", ins->n_experts);
            strncat(title, suffix, (sizeof title) - strlen(title) - 1);
        }
        SetWindowTextA(g.cand_title, title);
    }
}

static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    /* FINDMSGSTRING - see the comment on g_findmsg above open_find.
       comdlg32 posts this to FINDREPLACEA.hwndOwner, which open_find
       set to hwnd, so it always arrives here rather than at g.find_dlg
       itself. g_findmsg guards this: it reads 0 until create_children
       registers it, and RegisterWindowMessageA never returns 0 for a
       successful registration, so the check cannot mistake an
       unregistered state for a real match. */
    if (g_findmsg && msg == g_findmsg) {
        int down = 0, match_case = 0;
        const char *needle = NULL;
        /* lz_find_parse hides the FINDREPLACEA cast (gui/compat40.c) -
           see the comment above open_find for why this file cannot
           name that type. Returns 1 for FR_DIALOGTERM, 2 for
           FR_FINDNEXT (with down/match_case/needle filled in), 0 for
           anything else. */
        int kind = lz_find_parse(lp, &down, &match_case, &needle);
        if (kind == 1) {
            g.find_dlg = NULL;
        } else if (kind == 2) {
            CHARRANGE sel;
            long from, hit;
            /* Continues from the live selection, not from 0 every time
               - otherwise Find Next would keep re-finding the first hit
               instead of advancing. Down search continues from the
               selection's far end, up search from its near end, so a
               fresh dialog (nothing selected, cpMin==cpMax==0) starts
               at the top either way. */
            SendMessage(g.part[LZ_GUI_TRANSCRIPT], EM_EXGETSEL, 0,
                       (LPARAM)&sel);
            from = down ? sel.cpMax : sel.cpMin;
            hit = transcript_find(needle, from, down, match_case);
            if (hit >= 0) {
                CHARRANGE found;
                found.cpMin = hit;
                found.cpMax = hit + (long)strlen(needle);
                SendMessage(g.part[LZ_GUI_TRANSCRIPT], EM_EXSETSEL, 0,
                           (LPARAM)&found);
                SendMessage(g.part[LZ_GUI_TRANSCRIPT], EM_SCROLLCARET, 0, 0);
            } else {
                MessageBoxA(g.find_dlg,
                           lz_str_display(LZ_STR_FIND_NOT_FOUND),
                           lz_str_display(LZ_STR_ERR_TITLE),
                           MB_ICONEXCLAMATION | MB_OK);
            }
        }
        return 0;
    }
    switch (msg) {
    case WM_CREATE:
        g.main = hwnd;
        if (!create_children(hwnd)) return -1;
        /* Lay out here, not only from WM_SIZE. An overlapped window that
           is created and never resized may receive no WM_SIZE at all,
           and every child then sits at 0x0 - an empty grey window with
           six invisible controls in the corner. The selftest caught
           exactly that: 640x480 failed while 900x700 passed, because the
           second size was a CHANGE and the first was not. */
        relayout(hwnd);
        /* Drag and drop. 0 on a system with no shell32 export
           for it - the window then simply never receives WM_DROPFILES,
           which is what NT 3.51 did; g.drop_on tracks that (see the
           selftest wiring gate below), not IsWindowEnabled or anything
           painted, because DragAcceptFiles changes no visible state. */
        g.drop_on = lz_drop_accept(hwnd, 1);
        /* The demo ramp is the other thing that wants the tick when no
           job is running; ui_timer_sync reads both and is the only
           place that arms it. Off unless the ini asked for it. */
        ui_timer_sync(hwnd);
        /* Title-bar self-drawing. Attach the subclass after
           create_children so the caption exists before the first repaint,
           then push the initial three segments. The subclass is attached
           unconditionally; whether it PAINTS is lz_caption_can_paint's
           answer, which is no on the 3.51 floor - there the system
           caption is left alone. */
        lz_caption_attach(hwnd);
        push_caption();
        return 0;

    case WM_SIZE:
        relayout(hwnd);
        return 0;

    /* The tool bar dock's BOTTOM groove. The top one is the control's
       own divider (gui/toolbar.c); this one cannot be, because a child
       window clips to its own rectangle and the groove is outside it -
       gui/layout.c hands the control TOOLBAR_H minus LZ_GUI_DOCK_GROOVE
       and leaves those two pixels to the parent.
       Painting it here rather than in WM_ERASEBKGND on purpose: the
       erase runs before the bar paints and the bar's own background
       would not cover it, but an erase is also skipped whenever the
       update region misses it, and a groove that comes and goes with
       the scroll is worse than none. */
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(hwnd, &ps);
        if (g.part[LZ_GUI_TOOLBAR]) {
            RECT rc, band;
            LZRect r[LZ_GUI_PART_COUNT];
            GetClientRect(hwnd, &rc);
            lz_gui_layout(rc.right - rc.left, rc.bottom - rc.top,
                          g.input_h, g.status_h, side_panel_mode(), r);
            band.left   = 0;
            band.right  = rc.right;
            band.bottom = r[LZ_GUI_TOOLBAR].y + r[LZ_GUI_TOOLBAR].h
                        + LZ_GUI_DOCK_GROOVE;
            band.top    = band.bottom - LZ_GUI_DOCK_GROOVE;
            lz_draw_edge(dc, &band, LZ_EDGE_ETCHED, LZ_BF_BOTTOM);
        }
        EndPaint(hwnd, &ps);
        return 0;
    }

    /* Never sent unless lz_drop_accept's DragAcceptFiles call in
       WM_CREATE actually took - NT 3.51 with no shell32 export for it
       simply never fires this message, so no g.drop_on guard is needed
       here any more than IDM_STOP_GEN needs one for a job that is not
       running: the source of the message already did the gating.

       What was dropped may be the model's directory or the
       model.bin file inside it - lz_drop_dir_of tells them apart
       (gui/compat40.c) - and either way this reuses open_model_dir
       rather than growing a second copy of the join-drain +
       lz_gui_model_dir_ok + start_job sequence, the same reasoning
       that made the MRU handler share it too.

       lz_drop_finish runs unconditionally, including when
       lz_drop_first_path or lz_drop_dir_of found nothing usable: the
       HDROP is shell-allocated and has to be freed either way, or
       every failed or unrecognised drop leaks one. */
    case WM_DROPFILES: {
        char path[MAX_PATH], dir[MAX_PATH];
        if (lz_drop_first_path(wp, path, (int)sizeof path) &&
            lz_drop_dir_of(path, dir, (int)sizeof dir))
            open_model_dir(hwnd, dir);
        lz_drop_finish(wp);
        return 0;
    }

    case WM_SYSCOLORCHANGE:
        /* Forwarded: only top-level windows are sent this, and the
           status strip is a child - without this it keeps painting in
           the scheme that was in force when the process started. */
        if (g.part[LZ_GUI_STATUS] && !g.status_is_sbar)
            SendMessage(g.part[LZ_GUI_STATUS], WM_SYSCOLORCHANGE, 0, 0);
        else
            lamps_reload();
        return 0;

    case WM_APP_PREFILL:
        /* Redraw the indicator now rather than at the next tick - see
           WM_APP_PREFILL in worker.h. Only while a job runs: a message
           that outlived its job would repaint from counters the next
           turn has not written yet. */
        if (g.job_kind == JOB_GENERATE) prefill_paint_tick();
        return 0;

    case WM_TIMER:
        /* One tick, one handler. Every periodic consumer - the token
           flush, the activity lamp, the demo ramp, the status line -
           hangs off ui_tick, which decides the phase once. Three timers
           each deciding it separately is what put "generating" over the
           prefill progress; see LZ_UI_TIMER. */
        if (wp == LZ_UI_TIMER) { ui_tick(); return 0; }
        break;

    /* ---- splitter ---- */
    case WM_SETCURSOR:
        if ((HWND)wp == hwnd) {
            /* A plain local, not a compound literal. gcc took
               `&(RECT){...}` without a word; wcc386 -za99 does not have
               them, and the floor gate said so in three lines. */
            POINT p;
            RECT band;
            GetCursorPos(&p);
            ScreenToClient(hwnd, &p);
            band.left = g.split.x;
            band.top = g.split.y;
            band.right = g.split.x + g.split.w;
            band.bottom = g.split.y + g.split.h;
            if (PtInRect(&band, p)) {
                SetCursor(LoadCursor(NULL, IDC_SIZENS));
                return TRUE;
            }
        }
        break;

    case WM_LBUTTONDOWN: {
        POINT p;
        p.x = (short)LOWORD(lp);
        p.y = (short)HIWORD(lp);
        if (p.x >= g.split.x && p.x < g.split.x + g.split.w &&
            p.y >= g.split.y && p.y < g.split.y + g.split.h) {
            SetCapture(hwnd);
            g.dragging = 1;
            g.drag_y = g.split.y;
            drag_bar(hwnd, g.drag_y);
            return 0;
        }
        break;
    }

    case WM_MOUSEMOVE:
        if (g.dragging) {
            /* Snap the bar to a position the layout would actually
               accept, so releasing never moves it somewhere the user
               did not see it. */
            int want = drag_to_input_h(hwnd, (short)HIWORD(lp));
            RECT rc;
            LZRect r[LZ_GUI_PART_COUNT];
            GetClientRect(hwnd, &rc);
            lz_gui_layout(rc.right - rc.left, rc.bottom - rc.top, want,
                         g.status_h, side_panel_mode(), r);
            if (r[LZ_GUI_SPLIT].y != g.drag_y) {
                drag_bar(hwnd, g.drag_y);          /* XOR it away */
                g.drag_y = r[LZ_GUI_SPLIT].y;
                drag_bar(hwnd, g.drag_y);          /* and back down */
            }
            return 0;
        }
        break;

    case WM_LBUTTONUP:
        if (g.dragging) {
            drag_bar(hwnd, g.drag_y);              /* erase the tracker */
            g.dragging = 0;
            ReleaseCapture();
            g.input_h = drag_to_input_h(hwnd, (short)HIWORD(lp));
            relayout(hwnd);
            InvalidateRect(hwnd, NULL, TRUE);
            return 0;
        }
        break;

    case WM_CAPTURECHANGED:
        /* Somebody else took the mouse - an Alt-Tab, a message box.
           Leave the tracker erased and the layout untouched rather than
           committing a drag the user never finished. */
        if (g.dragging) {
            drag_bar(hwnd, g.drag_y);
            g.dragging = 0;
        }
        return 0;

    case WM_GETMINMAXINFO: {
        /* Derived from the layout's own minimum, not typed in twice.
           AdjustWindowRect turns the client minimum into the window
           minimum, so a border style change cannot desynchronise them. */
        RECT rc;
        MINMAXINFO *mmi = (MINMAXINFO *)lp;
        rc.left = 0; rc.top = 0;
        rc.right = LZ_GUI_MIN_CW; rc.bottom = LZ_GUI_MIN_CH;
        AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
        mmi->ptMinTrackSize.x = rc.right - rc.left;
        mmi->ptMinTrackSize.y = rc.bottom - rc.top +
                                GetSystemMetrics(SM_CYMENU);
        return 0;
    }

    case WM_SETFOCUS:
        if (g.part[LZ_GUI_INPUT]) SetFocus(g.part[LZ_GUI_INPUT]);
        return 0;

    case WM_CTLCOLOREDIT:
        /* Fixed white background for the input box - see
           its own comment in create_children for why this is a
           SEPARATE mechanism from the transcript's EM_SETBKGNDCOLOR:
           a plain EDIT control has no such message, and can only be
           repainted from the PARENT's WM_CTLCOLOREDIT, which is what
           this is. Scoped to ID_INPUT alone via GetDlgCtrlID rather
           than answering it for every EDIT this window might ever
           host - the settings dialog's own edit controls are a
           SEPARATE window and never reach this wndproc at all, but a
           future EDIT added to the main window should not silently
           inherit this without its own decision. Text colour is set
           too, not just background: pinning one and not the other
           would risk the exact unreadable combination (say, light
           text on a scheme whose window colour is also light) this
           whole task exists to rule out. */
        if (GetDlgCtrlID((HWND)lp) == ID_INPUT) {
            HDC dc = (HDC)wp;
            SetBkColor(dc, RGB(0xFF, 0xFF, 0xFF));
            SetTextColor(dc, LZ_COLOR_TEXT);
            return (LRESULT)input_bkg_brush();
        }
        break;

    case WM_APP_TOKENS:
        /* Ownership arrives with the message and leaves with the free.
           Taking a COPY (tokens_arrived buffers) and freeing after
           means the buffer is live for exactly the span that reads it -
           the throttle must not extend that span, which is why the
           bytes are copied into g.tok_buf rather than the pointer
           being held until the tick. */
        tokens_arrived((const char *)lp, (int)wp);
        lz_worker_free_tokens((void *)lp);
        return 0;

    case WM_APP_INSPECT:
        /* Segment C built the candidate LISTBOX; Part Two (GUI half -
           the engine half, src/forward.c filling expert_hits) adds the
           16 expert lamps, a three-colour tally (repaint_lamps' own
           comment). Both readers get (LZInspect *)lp BEFORE the free,
           the same push-first-free-after shape WM_APP_TOKENS above
           already uses. */
        g.inspect_seen++;
        g.last_inspect = *(LZInspect *)lp;
        repaint_candidates((const LZInspect *)lp);
        repaint_lamps((const LZInspect *)lp);
        lz_worker_free_tokens((void *)lp);
        return 0;

    case WM_APP_GEN_DONE:
        finish_job(hwnd, (int)wp);
        return 0;

    case WM_COMMAND:
        /* The central guard - see cmd_is_enabled's own comment for what
           it protects and why one check here has to stand in for every
           path a command can arrive by, rather than teaching each path
           to check for itself. */
        if (!cmd_is_enabled((UINT)LOWORD(wp)))
            return 0;
        switch (LOWORD(wp)) {
        case IDM_OPEN_MODEL:
            open_model(hwnd);
            return 0;
        case IDM_MRU0: case IDM_MRU0 + 1:
        case IDM_MRU0 + 2: case IDM_MRU0 + 3:
            open_model_dir(hwnd, g.mru.item[LOWORD(wp) - IDM_MRU0]);
            return 0;
        case ID_SEND:
            do_send(hwnd);
            return 0;
        case IDM_SAVE_CHAT:
            save_chat(hwnd);
            return 0;
        case IDM_OPEN_CHAT:
            open_chat(hwnd);
            return 0;
        case IDM_COPY:
        case IDM_SELECT_ALL: {
            /* Accelerators fire BEFORE the control sees the key, so an
               Edit menu with Ctrl+C on it takes the shortcut away from
               the controls that already implemented it. Forwarding puts
               it back, and makes one menu item work for both the
               conversation and the input box.
               EM_SETSEL rather than a RichEdit-only message: the floor
               may be riched32 (RichEdit 1.0), and EM_SETSEL is shared
               with the plain EDIT the input box is. */
            HWND focus = GetFocus();
            if (focus != g.part[LZ_GUI_TRANSCRIPT] &&
                focus != g.part[LZ_GUI_INPUT])
                focus = g.part[LZ_GUI_TRANSCRIPT];
            if (LOWORD(wp) == IDM_COPY) SendMessage(focus, WM_COPY, 0, 0);
            else SendMessage(focus, EM_SETSEL, 0, (LPARAM)-1);
            return 0;
        }
        case IDM_FIND:
            open_find(hwnd);
            return 0;
        /* The three rollback commands. No guard of their own: the
           cmd_is_enabled check at the top of WM_COMMAND is what holds
           them to what rollback_sync last drew, which is the whole
           reason that guard exists (see its own comment). */
        case IDM_REGEN:
            do_regen(hwnd);
            return 0;
        case IDM_EDIT_LAST:
            do_edit_last(hwnd);
            return 0;
        case IDM_DEL_LAST:
            rollback_last(hwnd, 0);
            return 0;
        case IDM_SETTINGS: {
            /* Captured BEFORE the dialog, because the dialog writes
               straight into g.set and there is then nothing left to
               compare against - and the rollback on an allocation
               failure needs the old number, not the old-number-shaped
               thing the ini happens to hold. */
            int prev_ctx = g.set.ctx;
            char prev_sys[LZ_COMMON_SYSTEM_MAX + 1];
            strcpy(prev_sys, g.set.system);
            if (lz_gui_settings_dialog(hwnd, g.inst, &g.set)) {
                /* Before ANY of the writes below, because every one of
                   them lands in state a running job is reading:
                   apply_settings rewrites g.sess.opts, the prefix clear
                   resets the LZPrefixCache lz_prefix_prepare may be
                   inside of. ctx_commit drains for exactly this reason
                   but returns early when the context did not change, so
                   changing only the system prompt reached the clear
                   with the worker still running. The dialog is modal
                   and pumps messages, so the job kept going the whole
                   time it was open. */
                lz_worker_join_drain(hwnd);
                apply_settings();
                ctx_apply(hwnd, prev_ctx);
                /* The system prompt changed but the context did not, so
                   ctx_apply's prefix_clear never ran - and a changed
                   system prompt changes every RENDER, which changes the
                   token prefix the cache is keyed on.
                   Clear it explicitly; the model did not change, so
                   this is a clear, not a drop-then-arm. */
                if (strcmp(prev_sys, g.set.system) != 0)
                    lz_gui_session_prefix_clear(&g.sess);
            }
            return 0;
        }
        case IDM_CLEAR:
            /* Drained, not just joined - see worker.h. A stop here does
               not open a modal dialog the way Open/Save do, so nothing
               else pumps the queue for us: without the drain, a job
               stopped by this very call still has its WM_APP_TOKENS and
               WM_APP_GEN_DONE sitting undispatched when transcript_clear
               runs below, and the message loop delivers them afterward -
               the leftover token text and finish_job's own
               transcript_push("\r\n") land in the box the user just
               watched go blank. (lz_gui_session_reset below also zeroes
               reply_len, so the belated lz_gui_session_end no-ops rather
               than pushing that reply into hist - the damage here is the
               visible transcript, not the history struct.) */
            lz_worker_join_drain(hwnd);
            lz_gui_session_reset(&g.sess);
            transcript_clear();
            update_ctx_cell();  /* back to 0, or empty with no model */
            /* The conversation has no file now: drop the chat name, the
               title bar falls back to the "untitled" segment. */
            g.chat_name[0] = '\0';
            push_caption();
            /* transcript_clear forgot the positions; this is the half
               that redraws the commands that read them. */
            rollback_sync();
            return 0;
        case ID_STOP:
        case IDM_STOP_GEN:
            /* Only the flag. The job stops at its next check and the
               window learns about it from WM_APP_GEN_DONE - joining here
               would block the UI thread inside a message handler, and
               the messages it is waiting to drain arrive on this thread.

               Reached by Esc through the accelerator table (gui/kunkun98.rc)
               even while the input box or the transcript has focus: EDIT
               controls do not consume VK_ESCAPE the way they consume, say,
               Tab, so TranslateAcceleratorA still sees it and routes it
               here regardless of which child has focus. No guard is needed
               for the case where nothing is running -
               lz_worker_request_stop() is a no-op then (it only sets a
               flag a job's own loop polls), so Esc with no generation in
               flight does nothing observable rather than something that
               needs to be prevented. */
            lz_worker_request_stop();
            return 0;
        case IDM_LANG_ZH:
            apply_language(hwnd, 0);
            return 0;
        case IDM_LANG_EN:
            apply_language(hwnd, 1);
            return 0;
        case IDM_ABOUT:
            /* Its OWN title, not the menu item's. The menu string carries
               a mnemonic - "&About" / its Chinese counterpart - and a
               title bar prints the ampersand literally instead of
               underlining the next letter the way a menu does. The About
               WINDOW does this itself - it is what this call opens,
               replacing the MessageBox.
               Non-modal by being modal-over-the-owner: it disables this
               window and pumps until it is gone, the same shape the
               settings dialog uses. */
            lz_gui_about_dialog(hwnd, g.inst);
            return 0;
        case IDM_EXIT:
            /* Through WM_CLOSE, not straight to DestroyWindow: the
               stop-and-join lives there, and two ways out of the program
               means one of them eventually forgets. */
            SendMessage(hwnd, WM_CLOSE, 0, 0);
            return 0;
        default:
            break;
        }
        break;

    case WM_CLOSE:
        /* The unified sequence, at the one place it matters most: the
           window is about to stop existing, and a worker still posting
           into it is posting into a freed HWND. Drained too, while hwnd
           is still valid, so nothing of the worker's is left in the
           queue for DestroyWindow to orphan. */
        lz_worker_join_drain(hwnd);
        /* Written on the way out, not on every change: an ini rewritten
           per keystroke is a floppy-era machine writing to disk in the
           middle of a conversation. */
        {
            RECT wr;
            if (GetWindowRect(hwnd, &wr) && !IsIconic(hwnd) &&
                !IsZoomed(hwnd)) {
                lz_ini_set_int("win_x", (int)wr.left);
                lz_ini_set_int("win_y", (int)wr.top);
                lz_ini_set_int("win_w", (int)(wr.right - wr.left));
                lz_ini_set_int("win_h", (int)(wr.bottom - wr.top));
            }
            lz_ini_set_int("input_h", g.input_h);
            lz_ini_set_int("lang", lz_str_lang_english());
            lz_ini_set_int("think", g.set.think);
            lz_ini_set_int("temp_milli", (int)(g.set.temp * 1000.0f + 0.5f));
            lz_ini_set_int("seed_mode", g.set.seed_mode);
            lz_ini_set_int("seed", (int)(unsigned)g.set.seed);
            lz_ini_set_int("ctx", g.set.ctx);
            /* top_p is written only when the user set one. That is what
               makes "the key is present" mean "a user chose it" on the
               way back in - see the read side. Temperature does not get
               the same treatment because it has always been written
               unconditionally and the close-time gate pins that. */
            if (g.set.manual_topp)
                lz_ini_set_int("topp_milli",
                               (int)(g.set.topp * 1000.0f + 0.5f));
            /* think-block temp: same key-present-means-user-chose rule
               as topp, so only write it when the user actually set one.
               The GUI default-ON behaviour (0.3, manual flag clear) must
               NOT persist a key - a key would restore manual on the way
               back in and freeze the next preset change out. */
            if (g.set.manual_think_temp)
                lz_ini_set_int("think_temp_milli",
                               (int)(g.set.think_temp * 1000.0f + 0.5f));
            lz_ini_set_int("rep_milli", (int)(g.set.rep * 1000.0f + 0.5f));
            lz_ini_set_int("max_new", g.set.max_new);
            lz_ini_set_int("repeat_last_n", g.repeat_last_n);
            lz_ini_set_str("system", g.set.system);
            mru_save();
        }
        DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        /* riched20 is NOT unloaded here, and that is not an oversight.
           WM_DESTROY reaches the parent BEFORE its children, so the
           RichEdit child is still alive at this point and its window
           procedure lives inside the module being freed. Unloading it
           here crashes with STATUS_STACK_BUFFER_OVERRUN (0xC0000409);
           after a clean, all-green selftest report the only symptom is
           an exit code, because the checks have all finished and the
           file is already closed.

           It is never unloaded: the class it registers is process-wide
           and the process is ending anyway. */
        PostQuitMessage(0);
        return 0;

    default:
        break;
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

static int register_class(HINSTANCE inst) {
    WNDCLASSA wc;
    memset(&wc, 0, sizeof wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = wndproc;
    wc.hInstance = inst;
    /* The app icon from the resource, falling back to the system
       one: a build without resources still runs. */
    wc.hIcon = LoadIconA(inst, MAKEINTRESOURCEA(IDI_APP));
    if (!wc.hIcon) wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wc.lpszClassName = LZ_CLASS_NAME;
    return RegisterClassA(&wc) != 0;
}

static HWND create_main(HINSTANCE inst, int cw, int ch) {
    RECT rc;
    rc.left = 0; rc.top = 0; rc.right = cw; rc.bottom = ch;
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    return CreateWindowExA(
        0, LZ_CLASS_NAME, lz_str_display(LZ_STR_APP_TITLE),
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top, NULL, NULL, inst, NULL);
}

/* ---------------------------------------------------------- selftest */

static int st_fails;
static int st_skips;

static void st_check(FILE *f, int ok, const char *what) {
    fprintf(f, "%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) st_fails++;
}

/* A check that could not be RUN, which is not the same thing as one
   that ran and failed. Kept out of `checks` so checks+skips stays
   constant and a report diff shows a check going missing; kept out of
   `fails` so a system resource another process happened to be holding
   cannot fail the build. The reason is written into the report because
   a silent skip is exactly how a check that quietly stopped running
   for good goes unnoticed. */
static void st_skip(FILE *f, const char *what, const char *why) {
    fprintf(f, "SKIP %s: %s\n", what, why);
    st_skips++;
}

/* The clipboard is ONE object shared by every process on the desktop,
   and any of them - a clipboard viewer, a remote-desktop channel, the
   shell - may hold it open at the instant we ask. A single
   OpenClipboard therefore measures the rest of the desktop rather than
   this program, which is what made the Copy check flap.
   ~200 ms total: long enough to outlast another process's own open/
   read/close round trip, short enough that a desktop where the
   clipboard is genuinely wedged does not stall the selftest. */
#define LZ_CLIP_TRIES 20
#define LZ_CLIP_WAIT  10
static int st_clip_open(HWND hwnd) {
    int i;
    for (i = 0; i < LZ_CLIP_TRIES; i++) {
        if (OpenClipboard(hwnd)) return 1;
        Sleep(LZ_CLIP_WAIT);
    }
    return 0;
}

/* True when `path` was last written no earlier than this executable.
   The strip comparison is only meaningful between two images the SAME
   binary produced, and the counterpart is an ordinary file sitting next
   to the report from whenever the other run happened - possibly a build
   ago. A stale one still loads, still has the right dimensions, and
   still yields a percentage: iron law four's shape exactly, a number
   that is self-consistent and means nothing.
   Timestamps rather than a build stamp because the executable's own
   mtime IS the fact being asked about, and baking __DATE__ into the
   binary to answer it would trade a reproducible build for it.
   FindFirstFile rather than GetFileAttributesEx: the latter is 4.0. */
static int st_newer_than_exe(const char *path) {
    char exe[MAX_PATH];
    WIN32_FIND_DATAA me, other;
    HANDLE h;
    if (!GetModuleFileNameA(NULL, exe, (DWORD)sizeof exe)) return 0;
    h = FindFirstFileA(exe, &me);
    if (h == INVALID_HANDLE_VALUE) return 0;
    FindClose(h);
    h = FindFirstFileA(path, &other);
    if (h == INVALID_HANDLE_VALUE) return 0;
    FindClose(h);
    return CompareFileTime(&other.ftLastWriteTime, &me.ftLastWriteTime) >= 0;
}

/* Whatever the status line is showing, from whichever control is
   carrying it - set_status's own split, read back. */
static void st_status_text(char *out, int cap) {
    HWND h = g.part[LZ_GUI_STATUS];
    out[0] = '\0';
    if (!h) return;
    if (g.status_is_sbar) SendMessage(h, LZ_SB_GETTEXT, 0, (LPARAM)out);
    else                  GetWindowTextA(h, out, cap);
}

/* The mnemonic letter in a menu label, uppercased, or 0 if there is
   none. "&&" is a literal ampersand and is not one. */
static int st_mnemonic(const char *s) {
    int i;
    for (i = 0; s[i]; i++) {
        if (s[i] == '&' && s[i + 1] && s[i + 1] != '&') {
            int c = (unsigned char)s[i + 1];
            if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
            return c;
        }
    }
    return 0;
}

/* Every child window in z-order, which is NOT the same list as g.part[].
   That difference is the point: g.part[] holds the handles the code
   kept, and a control the code created and then lost still exists, still
   answers to its id, and is still in the tab chain. Walking the real
   children is the only way to see one. */
static int st_children(HWND hwnd, HWND *out, int cap) {
    HWND c = GetWindow(hwnd, GW_CHILD);
    int n = 0;
    while (c && n < cap) {
        out[n++] = c;
        c = GetWindow(c, GW_HWNDNEXT);
    }
    return n;
}

static void st_check_text(FILE *f, int part, LZStr id, const char *what) {
    char got[256];
    HWND h = g.part[part];
    got[0] = '\0';
    if (h) GetWindowTextA(h, got, (int)sizeof got);
    st_check(f, strcmp(got, lz_str_display(id)) == 0, what);
    if (strcmp(got, lz_str_display(id)) != 0)
        fprintf(f, "  got %s want %s\n", got, lz_str_display(id));
}

/* Colour of a character range, or 0xFFFFFFFF when the range is not one
   uniform colour - RichEdit clears the mask bit for a mixed selection,
   which is itself the answer to "did the run end where it should". */
static COLORREF st_color(HWND h, int from, int to) {
    CHARFORMAT cf;
    CHARRANGE cr;
    cr.cpMin = from;
    cr.cpMax = to;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    memset(&cf, 0, sizeof cf);
    cf.cbSize = sizeof cf;
    cf.dwMask = CFM_COLOR;
    SendMessage(h, EM_GETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    if (!(cf.dwMask & CFM_COLOR)) return 0xFFFFFFFFu;
    return cf.crTextColor;
}

/* CFE_BOLD/CFE_ITALIC over a character range, 0 for a mixed selection
 * (same CFM_ bit-clearing RichEdit uses for st_color's own "not one
 * colour" case) - the read-back half of append_run's bold/italic
 * wiring. */
static DWORD st_effects(HWND h, int from, int to) {
    CHARFORMAT cf;
    CHARRANGE cr;
    cr.cpMin = from;
    cr.cpMax = to;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    memset(&cf, 0, sizeof cf);
    cf.cbSize = sizeof cf;
    cf.dwMask = CFM_BOLD | CFM_ITALIC;
    SendMessage(h, EM_GETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    if (!(cf.dwMask & (CFM_BOLD | CFM_ITALIC))) return 0;
    return cf.dwEffects & (CFE_BOLD | CFE_ITALIC);
}

/* Whether a range's face is exactly append_run's own "Courier New" -
 * the read-back half of the code-span wiring. Mixed faces read back
 * with CFM_FACE cleared, which fails the strcmp the same way a mixed
 * colour fails st_color's mask check. */
static int st_is_code_face(HWND h, int from, int to) {
    CHARFORMAT cf;
    CHARRANGE cr;
    cr.cpMin = from;
    cr.cpMax = to;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    memset(&cf, 0, sizeof cf);
    cf.cbSize = sizeof cf;
    cf.dwMask = CFM_FACE;
    SendMessage(h, EM_GETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    return (cf.dwMask & CFM_FACE) && strcmp(cf.szFaceName, "Courier New") == 0;
}

/* A range's size in twips, 0 for a mixed selection - the read-back half
 * of the heading wiring. Asserted as a
 * RATIO to the body size rather than against a point number: the body
 * size comes from whatever font the running system gave lz_ui_font, so
 * "is it 14pt" is a question about the host, while "is it half again the
 * body" is the question the design actually answers. */
static int st_yheight(HWND h, int from, int to) {
    CHARFORMAT cf;
    CHARRANGE cr;
    cr.cpMin = from;
    cr.cpMax = to;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    memset(&cf, 0, sizeof cf);
    cf.cbSize = sizeof cf;
    cf.dwMask = CFM_SIZE;
    SendMessage(h, EM_GETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    if (!(cf.dwMask & CFM_SIZE)) return 0;
    return (int)cf.yHeight;
}

/* Tab stops on the paragraph a character position belongs to. The table
 * gate reads this back on the line BEFORE a table row, which is the only
 * form of the assertion that can fail while the code is wrong: PARAFORMAT
 * applies to a whole paragraph, so "did EM_SETPARAFORMAT get sent" stays
 * green even when it lands on the wrong one. */
static int st_tab_count(HWND h, int at) {
    PARAFORMAT pf;
    CHARRANGE cr;
    cr.cpMin = at;
    cr.cpMax = at;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    memset(&pf, 0, sizeof pf);
    pf.cbSize = sizeof pf;
    pf.dwMask = PFM_TABSTOPS;
    SendMessage(h, EM_GETPARAFORMAT, 0, (LPARAM)&pf);
    return (int)pf.cTabCount;
}

/* The hanging indent (PARAFORMAT.dxOffset) on the paragraph at `at`. */
static int st_offset(HWND h, int at) {
    PARAFORMAT pf;
    CHARRANGE cr;
    cr.cpMin = at;
    cr.cpMax = at;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    memset(&pf, 0, sizeof pf);
    pf.cbSize = sizeof pf;
    pf.dwMask = PFM_OFFSET;
    SendMessage(h, EM_GETPARAFORMAT, 0, (LPARAM)&pf);
    return (int)pf.dxOffset;
}

/* The background behind a range, or 0xFFFFFFFF when the range is not
 * one uniform colour / the control has no CHARFORMAT2 at all. */
static COLORREF st_backcolor(HWND h, int from, int to) {
    CHARFORMAT2A cf;
    CHARRANGE cr;
    if (!richedit_v2()) return 0xFFFFFFFFu;
    cr.cpMin = from;
    cr.cpMax = to;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    memset(&cf, 0, sizeof cf);
    cf.cbSize = sizeof cf;
    cf.dwMask = CFM_BACKCOLOR;
    SendMessage(h, EM_GETCHARFORMAT, SCF_SELECTION, (LPARAM)&cf);
    if (!(cf.dwMask & CFM_BACKCOLOR)) return 0xFFFFFFFFu;
    if (cf.dwEffects & CFE_AUTOBACKCOLOR) return GetSysColor(COLOR_WINDOW);
    return cf.crBackColor;
}

/* The LAST tab stop on the paragraph at `at`, in twips, or 0 when there
 * are none. Read back rather than recomputed: the question the table
 * gate asks is whether what the control ENDED UP WITH fits inside the
 * control, and recomputing the expected value here would only prove
 * this file agrees with itself. */
static int st_last_tab(HWND h, int at) {
    PARAFORMAT pf;
    CHARRANGE cr;
    cr.cpMin = at;
    cr.cpMax = at;
    SendMessage(h, EM_EXSETSEL, 0, (LPARAM)&cr);
    memset(&pf, 0, sizeof pf);
    pf.cbSize = sizeof pf;
    pf.dwMask = PFM_TABSTOPS;
    SendMessage(h, EM_GETPARAFORMAT, 0, (LPARAM)&pf);
    if (pf.cTabCount <= 0) return 0;
    return (int)pf.rgxTabs[pf.cTabCount - 1];
}

/* Push a scripted reply through the real display path and read back what
   landed in the control.

   No model is needed for this and none should be: what is being checked
   is that arbitrary chunk boundaries survive the trip into RichEdit with
   the right colours, and a real generator would only make the boundaries
   less controllable. */
static int st_transcript(FILE *f) {
    /* ASCII, so a character index equals a byte index and the colour
       ranges below can be written down. */
    static const char SCRIPT[] = "<think>ok</think>hi";
    /* A tagged reply with two CJK characters inside and two after, as
       bytes: a source literal here would be testing the compiler's
       source charset rather than the pipeline. */
    static const char CJK[] =
        "<think>\xE6\x80\x9D\xE8\x80\x83</think>\xE5\x9B\x9E\xE7\xAD\x94";
    /* Same as CJK, tags stripped - the display bytes gui/stream.c is
       expected to actually produce (tags are consumed, not shown - see
       gui/stream.h's own comment). */
    static const char CJK_CONTENT[] =
        "\xE6\x80\x9D\xE8\x80\x83\xE5\x9B\x9E\xE7\xAD\x94";
    HWND h = g.part[LZ_GUI_TRANSCRIPT];
    char got[512], want[512];
    int i, n, checks = 0;

    /* Three bytes at a time splits "<think>" and "</think>" both. */
    for (i = 0; i < (int)sizeof SCRIPT - 1; i += 3) {
        int take = (int)sizeof SCRIPT - 1 - i;
        transcript_push(SCRIPT + i, take < 3 ? take : 3);
    }
    transcript_end();

    got[0] = '\0';
    GetWindowTextA(h, got, (int)sizeof got);
    /* "okhi" - both tags consumed, never reaching the control at all
       (user decision: tags are consumed rather than shown). */
    st_check(f, strcmp(got, "okhi") == 0, "transcript: chunked script "
             "arrives intact, tags consumed rather than shown");
    if (strcmp(got, "okhi") != 0) fprintf(f, "  got %s\n", got);
    checks++;

    /* "ok" (2 characters) is the think region's content; "hi" is not.
       Neither tag's own bytes exist in the control to have a colour
       at all - that absence IS the check above. */
    st_check(f, st_color(h, 0, 2) == LZ_COLOR_THINK,
             "transcript: the think region's content is grey");
    checks++;
    st_check(f, st_color(h, 2, 4) == LZ_COLOR_TEXT,
             "transcript: text after the block is back to plain");
    checks++;
    st_check(f, st_color(h, 0, 4) == 0xFFFFFFFFu,
             "transcript: the two regions really are different colours");
    checks++;

    transcript_clear();
    got[0] = '\0';
    GetWindowTextA(h, got, (int)sizeof got);
    st_check(f, got[0] == '\0', "transcript: clear empties it");
    checks++;

    /* One byte at a time: every multi-byte character is split, and so is
       every tag. */
    for (i = 0; i < (int)sizeof CJK - 1; i++) transcript_push(CJK + i, 1);
    transcript_end();
    n = lz_gbk_from_utf8(CJK_CONTENT, (int)sizeof CJK_CONTENT - 1, want,
                         (int)sizeof want, NULL);
    (void)n;
    got[0] = '\0';
    GetWindowTextA(h, got, (int)sizeof got);
    st_check(f, strcmp(got, want) == 0,
             "transcript: byte-at-a-time Chinese matches the codec");
    checks++;

    transcript_clear();

    /* Inline Markdown reaching the REAL control - the
       parser's own output is gui/stream.c's job; this is
       only proof that append_run's CHARFORMAT wiring actually carries
       what the parser decided into RichEdit, the same relationship
       the think colour checks above have to LZ_COLOR_THINK. */
    {
        static const char MDSCRIPT[] = "**bold** and *italic* and `code`";
        static const char MDWANT[]   = "bold and italic and code";
        transcript_push(MDSCRIPT, (int)sizeof MDSCRIPT - 1);
        transcript_end();
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        st_check(f, strcmp(got, MDWANT) == 0,
                 "transcript: inline Markdown markers are consumed, "
                 "not shown");
        if (strcmp(got, MDWANT) != 0) fprintf(f, "  got %s\n", got);
        checks++;

        st_check(f, (st_effects(h, 0, 4) & CFE_BOLD) != 0,
                 "transcript: **bold** reaches the control as real "
                 "bold"); checks++;
        st_check(f, (st_effects(h, 9, 15) & CFE_ITALIC) != 0,
                 "transcript: *italic* reaches the control as real "
                 "italic"); checks++;
        st_check(f, st_is_code_face(h, 20, 24),
                 "transcript: `code` reaches the control in a "
                 "monospace face"); checks++;
        st_check(f, (st_effects(h, 4, 9) & (CFE_BOLD | CFE_ITALIC)) == 0,
                 "transcript: plain text between markers carries "
                 "neither effect"); checks++;

        transcript_clear();
    }

    /* Block Markdown reaching the real control. Same relationship to
       gui/stream.c as the inline block above:
       the parser's own decisions are covered exhaustively elsewhere;
       what can only be checked HERE is
       whether append_run turns those decisions into RichEdit state. */
    {
        static const char HSCRIPT[] = "# H\nbody\n";
        int body, head;
        transcript_push(HSCRIPT, (int)sizeof HSCRIPT - 1);
        transcript_end();
        head = st_yheight(h, 0, 1);          /* the 'H' */
        body = st_yheight(h, 2, 6);          /* "body" */
        /* A RATIO, not a point size: the body size comes from whatever
           font this host gave lz_ui_font, so "is it 14pt" would be a
           question about the host. x1.5 is the heading/body ratio.
           Both halves are read back rather than one being computed, so
           a build that hardcoded a size and a build that derived it can
           be told apart. */
        st_check(f, body > 0 && head == body * 3 / 2,
                 "transcript: a heading is half again the body size");
        if (!(body > 0 && head == body * 3 / 2))
            fprintf(f, "  head %d body %d twips\n", head, body);
        checks++;
        st_check(f, (st_effects(h, 0, 1) & CFE_BOLD) != 0,
                 "transcript: a heading is bold on top of the size");
        checks++;
        transcript_clear();

        /* One size for every level (user decision). gui/stream.c still
           parses "###" as level 3 and still
           clamps h4-h6 - so this is
           the assertion that the DISPLAY deliberately ignores the level.
           Its mutation is restoring the per-level x1.5/x1.3/x1.15 table,
           which reddens this and leaves the x1.5 check above green,
           because h1 was the level that table agreed with. */
        {
            static const char H3[] = "### H\n";
            int h3;
            transcript_push(H3, (int)sizeof H3 - 1);
            transcript_end();
            h3 = st_yheight(h, 0, 1);
            st_check(f, h3 == head,
                     "transcript: ### is the same size as #, not a "
                     "third level");
            if (h3 != head) fprintf(f, "  h3 %d h1 %d twips\n", h3, head);
            checks++;
            transcript_clear();
        }
    }
    {
        /* The seed, both policies. RANDOM assigns rng_seed: without it
           lz_generate's "0 -> the sampler substitutes 1" leaves every
           session on seed 1 and two independent launches with the same
           prompt produce byte-identical replies at temperature 0.8.

           The assertion is on opts.rng_seed, NOT on generated text.
           Identical text can also mean the model simply repeats itself,
           which would make this gate a statement about the checkpoint
           instead of about the seed - and on a weak model that reads
           green either way. The seed is the quantity this change
           controls, so the seed is what gets asserted.

           lz_gui_session_begin is what draws it (not apply), because a
           re-run of one turn has to differ from the first run. */
        LZGuiModel empty;
        LZGuiSession ss;
        unsigned long long a, b;
        char err[128];
        memset(&empty, 0, sizeof empty);
        lz_gui_session_init(&ss, &empty, 1);

        lz_gui_session_set_seed(&ss, LZ_COMMON_SEED_FIXED, 4242);
        lz_gui_session_begin(&ss, "x", 1, err, (int)sizeof err);
        a = ss.opts.rng_seed;
        lz_gui_session_begin(&ss, "x", 1, err, (int)sizeof err);
        b = ss.opts.rng_seed;
        st_check(f, a == 4242ULL && b == 4242ULL,
                 "seed: the fixed policy gives the typed value, every turn");
        if (!(a == 4242ULL && b == 4242ULL))
            fprintf(f, "  got %lu / %lu\n", (unsigned long)a, (unsigned long)b);
        checks++;

        lz_gui_session_set_seed(&ss, LZ_COMMON_SEED_RANDOM, 4242);
        lz_gui_session_begin(&ss, "x", 1, err, (int)sizeof err);
        a = ss.opts.rng_seed;
        lz_gui_session_begin(&ss, "x", 1, err, (int)sizeof err);
        b = ss.opts.rng_seed;
        /* Two begins back to back land in the SAME millisecond here -
           which is exactly why seed_turn exists. Without it this pair
           would be equal and the defect would be back, unnoticed,
           because the clock alone looks like it is doing the job. */
        st_check(f, a != b,
                 "seed: the random policy gives a different seed each turn");
        if (a == b)
            fprintf(f, "  both %lu\n", (unsigned long)a);
        checks++;
        st_check(f, a != 0 && b != 0,
                 "seed: ... and never the engine's zero, which means 1");
        checks++;
        lz_gui_session_free(&ss);

        /* A FRESH session is deterministic - the random policy belongs
           to the settings, not to lz_gui_session_init. This is the half
           that keeps every headless caller reproducible. */
        lz_gui_session_init(&ss, &empty, 1);
        lz_gui_session_begin(&ss, "x", 1, err, (int)sizeof err);
        a = ss.opts.rng_seed;
        lz_gui_session_begin(&ss, "x", 1, err, (int)sizeof err);
        st_check(f, a == 0ULL && ss.opts.rng_seed == 0ULL,
                 "seed: a session nobody configured stays deterministic");
        checks++;
        lz_gui_session_free(&ss);
    }
    /* ... and the PRODUCT is configured. The three checks above all call
       lz_gui_session_set_seed themselves, so none of them can tell
       whether startup ever does - and "the front end forgot to set a
       seed" is the entire defect being fixed here. g.sess has been
       through the real startup path by the time the selftest runs.
       Mutation: drop the set_seed line from apply_settings - only this
       reddens. */
    st_check(f, g.sess.seed_mode == LZ_COMMON_SEED_RANDOM,
             "seed: startup left the live session on the random policy");
    checks++;
    /* Same shape, same reason: the failure mode is that nothing ever
       turns the knob on - a default is a caller.
       Mutation: put the ini default back to 0 - only this reddens. */
    st_check(f, g.sess.prefill == LZ_PREFILL_PREFIX,
             "prefill: startup left the live session on prefix reuse");
    checks++;
    {
        /* The throttle, and BOTH settings of it - iron law nine's "one
           option plus one gate proving it really changes something".
           The knob is worth nothing if the only thing checked is that
           the text eventually arrives, which it does either way.

           Driven through tokens_arrived/tokens_flush, the functions
           WM_APP_TOKENS and the timer actually call, not through
           transcript_push - the point is precisely what happens
           BETWEEN a token arriving and the tick. */
        int saved = g.tok_ms;
        char got[64];

        g.tok_ms = 100;
        tokens_arrived("ab", 2);
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        st_check(f, got[0] == '\0',
                 "transcript: with the throttle on, a token is not on "
                 "screen before the tick");
        if (got[0]) fprintf(f, "  got %s\n", got);
        checks++;
        tokens_flush();
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        st_check(f, strcmp(got, "ab") == 0,
                 "transcript: ... and it is there after the tick");
        if (strcmp(got, "ab") != 0) fprintf(f, "  got %s\n", got);
        checks++;
        transcript_clear();

        /* stream_ms = 0 is the control: the old behaviour, still
           reachable, and it has to differ from the one above or the
           knob is decorative. */
        g.tok_ms = 0;
        tokens_arrived("cd", 2);
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        st_check(f, strcmp(got, "cd") == 0,
                 "transcript: with the throttle off, a token is on "
                 "screen immediately");
        if (strcmp(got, "cd") != 0) fprintf(f, "  got %s\n", got);
        checks++;
        transcript_clear();

        /* A batch bigger than the buffer is pushed, not truncated -
           the one way a buffering optimisation can lose output. */
        {
            static char big[LZ_TOK_BUF + 64];
            int i;
            for (i = 0; i < (int)sizeof big; i++) big[i] = 'x';
            g.tok_ms = 100;
            tokens_arrived(big, (int)sizeof big);
            tokens_flush();
            st_check(f, GetWindowTextLengthA(h) == (int)sizeof big,
                     "transcript: a batch larger than the throttle "
                     "buffer is not truncated");
            if (GetWindowTextLengthA(h) != (int)sizeof big)
                fprintf(f, "  %d of %d bytes\n", GetWindowTextLengthA(h),
                        (int)sizeof big);
            checks++;
            transcript_clear();
        }
        g.tok_ms = saved;
    }
    {
        /* Bullets get a hanging indent, plain lines do not (user
           request, after a wrapped bullet's second line was seen
           starting left of its own marker). Asserted as "positive, and
           gone again on the next line" rather than against a twip
           count: the width is MEASURED from the running font, so a
           number here would be a number about this desk.

           Two mutations, one each: drop LZ_STYLE_BULLET from the
           dxOffset test (first assertion), and drop PFM_OFFSET from
           dwMask so the next paragraph inherits it (second). */
        static const char S[] = "- item\nplain\n";
        transcript_push(S, (int)sizeof S - 1);
        transcript_end();
        st_check(f, st_offset(h, 0) > 0,
                 "transcript: a bullet line has a hanging indent");
        if (!(st_offset(h, 0) > 0))
            fprintf(f, "  dxOffset %d twips\n", st_offset(h, 0));
        checks++;
        st_check(f, st_offset(h, 8) == 0,
                 "transcript: the line after it does not");
        if (st_offset(h, 8) != 0)
            fprintf(f, "  dxOffset %d twips on the plain line\n",
                    st_offset(h, 8));
        checks++;
        transcript_clear();
    }
    if (richedit_v2()) {
        /* Grey behind code, and the window's own colour behind
           everything else (user request). Skipped entirely
           on the RichEdit 1.0 fallback, which has no CHARFORMAT2 - and
           skipped rather than weakened, so this says nothing at all
           there instead of saying something that is always true.

           Mutations: drop CFM_BACKCOLOR from the mask (first
           assertion), and drop the CFE_AUTOBACKCOLOR branch so the
           grey leaks past the code span (second). */
        static const char S[] = "`x` y";
        transcript_push(S, (int)sizeof S - 1);
        transcript_end();
        st_check(f, st_backcolor(h, 0, 1) == LZ_COLOR_CODE_BG,
                 "transcript: code is on a grey background");
        if (st_backcolor(h, 0, 1) != LZ_COLOR_CODE_BG)
            fprintf(f, "  backcolor %08lx want %08lx\n",
                    (unsigned long)st_backcolor(h, 0, 1),
                    (unsigned long)LZ_COLOR_CODE_BG);
        checks++;
        st_check(f, st_backcolor(h, 2, 4) == GetSysColor(COLOR_WINDOW),
                 "transcript: the text after it is back on the window "
                 "colour");
        checks++;
        transcript_clear();
    }
    {
        /* ONE BYTE AT A TIME, which is closer to how a reply actually
           arrives than any check above: a token is an arbitrary number
           of bytes and append_run is called once per run, so a heading
           line reaches the control as two or three separate
           EM_SETCHARFORMAT + EM_SETPARAFORMAT + EM_REPLACESEL triples.
           The SCANNER is chunk-independent; nothing proved the
           RENDERER was, and the runs it
           sees are exactly what the chunking changes.

           NO MUTATION REDDENS THIS ONE ALONE, and that is worth writing
           down rather than papering over with a mutation that reddens
           everything. Three were tried: moving EM_SETPARAFORMAT after
           EM_REPLACESEL, skipping EM_EXSETSEL for table runs, and the
           "only send when the TABLE bit changed" flag the design warns
           about. The first and third changed nothing anywhere; the
           second broke the insert itself and reddened half the file,
           which proves nothing about this assertion.

           The reason is structural and is the property worth keeping:
           EVERY run carries its complete style and append_run holds no
           state between calls, so where the run boundaries fall cannot
           change the result. This check is therefore a guard on that
           invariant - the day someone caches "was the last run a table"
           to save two messages per token, the whole-buffer checks above
           still pass and this is where it should show up - not a check
           that is sensitive today. An unproven gate is worth keeping
           only when it says so about itself. */
        static const char S[] = "### H\n|a|b|\n";
        int i, head, body;
        for (i = 0; i < (int)sizeof S - 1; i++) transcript_push(S + i, 1);
        transcript_end();
        head = st_yheight(h, 0, 1);          /* the 'H' */
        body = st_yheight(h, 2, 3);          /* the 'a' of the table row */
        st_check(f, body > 0 && head == body * 3 / 2,
                 "transcript: byte-at-a-time, the heading is still "
                 "half again the body");
        if (!(body > 0 && head == body * 3 / 2))
            fprintf(f, "  head %d body %d twips\n", head, body);
        checks++;
        st_check(f, st_tab_count(h, 2) == LZ_TABLE_COLS &&
                    st_tab_count(h, 0) == 0,
                 "transcript: byte-at-a-time, the table row got the tab "
                 "stops and the heading line did not");
        if (!(st_tab_count(h, 2) == LZ_TABLE_COLS && st_tab_count(h, 0) == 0))
            fprintf(f, "  heading %d table %d\n",
                    st_tab_count(h, 0), st_tab_count(h, 2));
        checks++;
        transcript_clear();
    }
    {
        /* A TURN, the way do_send builds one - not transcript_push on
           its own. Every check above feeds the scanner directly, and
           that is exactly the shape of hole this found: a turn is
           turn_header (chrome) + turn_indent() + the content, and while
           the indent went THROUGH the scanner it left at_bol false, so
           the FIRST LINE of every turn could not be a heading, a
           bullet, a quote, a rule, a table row or a fence. Caught by
           looking at the screen, not by any of the gates above - a
           reply opening with a "### " heading showed the hashes.

           Its own mutation is turn_indent() going back to
           transcript_push("  ", 2), which reddens this and nothing
           else. */
        static const char TURN[] = "### H\nbody\n";
        int i;
        char got[128];
        turn_indent();
        transcript_push(TURN, (int)sizeof TURN - 1);
        transcript_end();
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        for (i = 0; got[i]; i++)
            if (got[i] == '#') break;
        st_check(f, got[i] == '\0',
                 "transcript: a turn's FIRST line can still be a heading");
        if (got[i] == '#') fprintf(f, "  got %s\n", got);
        checks++;
        /* The other half, and it fails for the opposite reason: a fix
           that got the heading back by dropping the indent would pass
           the check above and lose the turn's shape. Two spaces, then
           the heading's own first character. */
        st_check(f, got[0] == ' ' && got[1] == ' ' && got[2] == 'H',
                 "transcript: ... and the two-space indent is still there");
        checks++;
        transcript_clear();
    }
    {
        /* This gate asserts the RESULT on a paragraph that must NOT
           have changed, not that EM_SETPARAFORMAT was sent. Sending it
           is what a broken version does too - the failure mode is
           landing it on the wrong paragraph, and only a read-back of the
           earlier line can see that. Its own mutation is SCF_ALL in
           place of the empty-selection send, which leaves the message
           going out unchanged and reddens only this. */
        static const char TSCRIPT[] = "one\ntwo\n|a|b|\n";
        transcript_push(TSCRIPT, (int)sizeof TSCRIPT - 1);
        transcript_end();
        st_check(f, st_tab_count(h, 0) == 0,
                 "transcript: a table row leaves the earlier lines' tab "
                 "stops alone");
        if (st_tab_count(h, 0) != 0)
            fprintf(f, "  line 1 cTabCount %d\n", st_tab_count(h, 0));
        checks++;
        /* The other direction, so a build that simply never sets any tab
           stop at all cannot pass the assertion above by doing nothing -
           that is this project's own "prove the instrument is live"
           rule, and without it the gate would survive the feature being
           deleted outright. */
        st_check(f, st_tab_count(h, 9) == LZ_TABLE_COLS,
                 "transcript: the table row itself did get tab stops");
        if (st_tab_count(h, 9) != LZ_TABLE_COLS)
            fprintf(f, "  table row cTabCount %d\n", st_tab_count(h, 9));
        checks++;
        /* ... and every one of them is inside the control. RichEdit
           wraps a row that runs past the right edge and restarts the
           continuation at x=0, where the next tab jumps to the FIRST
           stop - so an overflowing cell lands under column two, mixed
           into the row below. Measured on screen with six columns.
           Asserted against the control's own
           width, not against a number, so it holds at any window size
           and on any DPI. Mutation: drop the clamp (always
           base_twips*8) - at 640x480 six inches of stops do not fit in
           four inches of control and only this reddens. */
        {
            int last = st_last_tab(h, 9), wide = client_width_twips(h);
            st_check(f, wide > 0 && last > 0 && last <= wide,
                     "transcript: the table's tab stops fit inside the "
                     "control");
            if (!(wide > 0 && last > 0 && last <= wide))
                fprintf(f, "  last stop %d twips, control %d twips\n",
                        last, wide);
            checks++;
        }
        transcript_clear();
    }

    /* Think-region seeding: a thinking-enabled prompt
       already ends "<think>\n" - the model resumes INSIDE the block
       and its reply stream never contains an opening tag, only the
       closing one. Without seed_reply_style, the parser starts
       assuming "not in think", the "</think>" below never matches
       anything to close, and the whole reasoning span - AND the tag
       text itself - renders as ordinary black text. ASCII, matching
       the SCRIPT test above's own "a character index equals a byte
       index" reasoning - the CJK-by-bytes test earlier in this
       function already proves the GBK conversion independently, this
       one is about the seed, not the codec. */
    {
        static const char REPLY[] = "reasoning</think>answer";
        g.sess.think = 1;
        seed_reply_style();
        transcript_push(REPLY, (int)sizeof REPLY - 1);
        transcript_end();

        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        st_check(f, strchr(got, '<') == NULL,
                 "transcript: a seeded reply's closing think tag never "
                 "reaches the display"); checks++;
        if (strchr(got, '<'))
            fprintf(f, "  got %s\n", got);
        st_check(f, strcmp(got, "reasoninganswer") == 0,
                 "transcript: the seeded reply's content is exactly "
                 "the two halves with the tag removed");
        if (strcmp(got, "reasoninganswer") != 0)
            fprintf(f, "  got %s\n", got);
        checks++;

        /* "reasoning" (9 chars) is the think region; "answer" (6
           chars) is not - the actual per-half comparison, not just
           "no '<' anywhere". */
        st_check(f, st_color(h, 0, 9) == LZ_COLOR_THINK,
                 "transcript: the seeded reply's FIRST half renders "
                 "grey"); checks++;
        st_check(f, st_color(h, 9, 15) == LZ_COLOR_TEXT,
                 "transcript: the seeded reply's SECOND half renders "
                 "plain, after the unseen close tag flipped it back");
        checks++;

        g.sess.think = 0;
        transcript_clear();
    }

    /* Turn headers (user request: colour-coded speaker + clock above
       each turn). Read back the same way the think-region
       colours above are - via st_color on the real control, not "was
       a function called". */
    {
        /* lz_str_DISPLAY, not lz_str_utf8: GetWindowTextA below reads
           back whatever landed in this ANSI control, which is GBK -
           turn_header converts its UTF-8 argument to GBK on the way
           in (append_colored_line's own lz_gbk_from_utf8 call), so
           the on-screen bytes must be compared against the GBK form,
           the same lz_str_display/lz_str_utf8 split st_check_text
           already relies on for every other control in this file. */
        const char *label = lz_str_display(LZ_STR_SPEAKER_USER);
        int llen = (int)strlen(label);
        int fmt_ok;

        transcript_clear();
        turn_header(LZ_STR_SPEAKER_USER, LZ_COLOR_USER, 1);
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        /* label, one space, HH:MM (digit digit colon digit digit),
           CRLF - checked structurally rather than against a captured
           clock reading: turn_header reads GetLocalTime itself, live,
           so this test has no way to know in advance what time it
           will have read. */
        fmt_ok = (int)strlen(got) == llen + 8 &&
                 strncmp(got, label, (size_t)llen) == 0 &&
                 got[llen]     == ' '  &&
                 got[llen + 1] >= '0' && got[llen + 1] <= '9' &&
                 got[llen + 2] >= '0' && got[llen + 2] <= '9' &&
                 got[llen + 3] == ':'  &&
                 got[llen + 4] >= '0' && got[llen + 4] <= '9' &&
                 got[llen + 5] >= '0' && got[llen + 5] <= '9' &&
                 got[llen + 6] == '\r' && got[llen + 7] == '\n';
        st_check(f, fmt_ok, "transcript: the user turn header is the "
                 "speaker label, a space, and a HH:MM clock");
        if (!fmt_ok) fprintf(f, "  got %s\n", got);
        checks++;

        /* The LABEL's own character span, not the whole line and not
           `llen` (measured): the default UI language is Chinese, and
           RichEdit's character positions are DECODED characters, not
           ANSI bytes, so a DBCS label's strlen() overshoots. Also
           measured: RichEdit collapses a "\r\n" pair into ONE
           character position, not two, so a range computed as
           "decoded label chars + 1 space + 5 clock chars + 2 CRLF
           bytes" overshoots too and lands one position INTO the
           implicit trailing run, which reads back mixed - the exact
           failure this replaced. Restricting the checked range to
           just the label (never touching the space, the clock or the
           CRLF) sidesteps both counting questions entirely; it is
           still a real proof that the LABEL itself - not "something,
           somewhere on this line" - carries the user colour. */
        {
            int lchars = MultiByteToWideChar(CP_ACP, 0, label, llen,
                                             NULL, 0);
            st_check(f, lchars > 0 &&
                     st_color(h, 0, lchars) == LZ_COLOR_USER,
                     "transcript: the user turn header renders in the "
                     "user colour"); checks++;
        }

        transcript_clear();
    }
    {
        const char *ulabel = lz_str_display(LZ_STR_SPEAKER_USER);
        const char *alabel = lz_str_display(LZ_STR_SPEAKER_ASSISTANT);
        int allen = (int)strlen(alabel);

        transcript_clear();
        turn_header(LZ_STR_SPEAKER_ASSISTANT, LZ_COLOR_ASSISTANT, 1);
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        st_check(f, strncmp(got, alabel, (size_t)allen) == 0 &&
                 got[allen] == ' ',
                 "transcript: the assistant turn header starts with "
                 "its own speaker label, distinct from the user's");
        if (strcmp(alabel, ulabel) == 0)
            fprintf(f, "  speaker labels are not even distinct: %s\n",
                   alabel);
        checks++;

        /* The label's own span only - same reasoning as the user
           header check above. */
        {
            int achars = MultiByteToWideChar(CP_ACP, 0, alabel, allen,
                                             NULL, 0);
            st_check(f, achars > 0 &&
                     st_color(h, 0, achars) == LZ_COLOR_ASSISTANT,
                     "transcript: the assistant turn header renders "
                     "in the assistant colour"); checks++;
        }
        /* The actual pairwise comparison, not just each colour
           individually matching its own #define - a build where
           LZ_COLOR_USER and LZ_COLOR_ASSISTANT happened to be the
           same literal would still pass both checks above. */
        st_check(f, LZ_COLOR_USER != LZ_COLOR_ASSISTANT,
                 "transcript: user and assistant colours are not the "
                 "same constant"); checks++;

        transcript_clear();
    }
    {
        /* with_time == 0: load_chat_from's replay path - NO clock at
           all, not a blank one (turn_header's own comment: LZChatMsg
           carries no timestamp to replay honestly). */
        const char *label = lz_str_display(LZ_STR_SPEAKER_USER);
        char want[160];

        transcript_clear();
        turn_header(LZ_STR_SPEAKER_USER, LZ_COLOR_USER, 0);
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        sprintf(want, "%s\r\n", label);
        st_check(f, strcmp(got, want) == 0,
                 "transcript: with_time off omits the clock entirely, "
                 "rather than leaving a blank one");
        if (strcmp(got, want) != 0) fprintf(f, "  got %s want %s\n",
                                            got, want);
        checks++;

        transcript_clear();
    }
    {
        /* transcript_line's own blank-line-after (user request:
           breathing room between turns). Its own two callers are
           do_send's user turn and load_chat_from's replay loop - direct
           here, in addition to the two st_worker checks that both
           expect "%s\r\n\r\n" for the assistant side; those two only
           ever exercised the assistant path, and this is the
           user/replay path's own proof.
           Mutation: reverting the "\r\n\r\n" back to "\r\n" in
           transcript_line's own body reddens only this check. */
        static const char CONTENT[] = "hi";
        char want[16];

        transcript_clear();
        transcript_line(CONTENT, (int)sizeof CONTENT - 1);
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        sprintf(want, "%s\r\n\r\n", CONTENT);
        st_check(f, strcmp(got, want) == 0,
                 "transcript: transcript_line ends its content with a "
                 "blank line, not just its own line break");
        if (strcmp(got, want) != 0) fprintf(f, "  got %s\n", got);
        checks++;

        transcript_clear();
    }

    /* sys_line/sys_line_fmt: red, and a REAL blank line after - a
       second CRLF, not just the one every other transcript
       line already ends with, or the next turn's own header sits
       flush against it (the exact bug the "model loaded" line had). */
    {
        static const char SYSMSG[] = "sysmsg";
        char want[64];

        transcript_clear();
        sys_line_fmt(SYSMSG, (int)sizeof SYSMSG - 1);
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        sprintf(want, "%s\r\n\r\n", SYSMSG);
        st_check(f, strcmp(got, want) == 0,
                 "transcript: sys_line_fmt ends with a blank line "
                 "after it, not just its own CRLF");
        if (strcmp(got, want) != 0) fprintf(f, "  got %s\n", got);
        checks++;

        st_check(f, st_color(h, 0, (int)sizeof SYSMSG - 1) == LZ_COLOR_SYS,
                 "transcript: sys_line_fmt renders in the system "
                 "colour"); checks++;

        transcript_clear();
    }
    {
        /* sys_line(id) gets the SAME treatment as sys_line_fmt - the
           behavioural half of "single exit point" (the static half -
           that the colour-setting function is called with the SYSTEM
           colour constant at exactly sys_line_fmt's own two lines, and
           nowhere else in this file - is a check_zh_comments.py-style
           source scan of the kind this
           file's other structural gates already use). Compared byte
           for byte
           against the SAME construction sys_line_fmt's own check
           above uses, not just "is it also red" - a sys_line that
           forgot the blank line, or coloured itself independently
           with a second RGB literal that happened to render the same
           on screen, would still look right to a weaker check. */
        const char *msg = lz_str_display(LZ_STR_SYS_CTX_TRIMMED);
        char want[512];

        transcript_clear();
        sys_line(LZ_STR_SYS_CTX_TRIMMED);
        got[0] = '\0';
        GetWindowTextA(h, got, (int)sizeof got);
        sprintf(want, "%s\r\n\r\n", msg);
        st_check(f, strcmp(got, want) == 0,
                 "transcript: sys_line(id) matches sys_line_fmt's own "
                 "output byte for byte - the same exit point");
        if (strcmp(got, want) != 0) fprintf(f, "  got %s\n", got);
        checks++;

        /* The message's own character span, not a byte count and not
           -1 (measured): the default UI language is Chinese,
           LZ_STR_SYS_CTX_TRIMMED's Chinese text is DBCS so strlen(msg)
           overshoots RichEdit's own DECODED-character positions, and
           separately RichEdit collapses each "\r\n" pair into ONE
           position rather than two, so even a correctly DBCS-aware
           byte-to-char conversion of the WHOLE "message\r\n\r\n"
           string overshoots by landing one position into the
           trailing implicit run. Checking just the message's own span
           avoids both counting questions. */
        {
            int mchars = MultiByteToWideChar(CP_ACP, 0, msg,
                                             (int)strlen(msg), NULL, 0);
            st_check(f, mchars > 0 &&
                     st_color(h, 0, mchars) == LZ_COLOR_SYS,
                     "transcript: sys_line(id) also renders in the "
                     "system colour"); checks++;
        }

        transcript_clear();
    }

    return checks;
}

/* ---- find: transcript_find, without ever driving comdlg32's dialog --

   FindTextA's own window cannot be exercised end to end here (a
   modeless comdlg32 dialog is not something a selftest drives through).
   What CAN be checked without one is the
   function every Find Next click ultimately calls: transcript_find,
   the same function whether the caller is a real dialog or this file. */
static int st_find(FILE *f) {
    /* U+4F60 U+597D U+FF0C U+4E16 U+754C ("hello, comma, world" in
       Chinese), UTF-8, so the transcript's own GBK conversion is
       exercised the same way a real reply would be, rather than a byte
       string picked to already agree with the assertion below. */
    static const char TEXT_UTF8[] =
        "before \xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95"
        "\x8C after";
    /* U+4F60 U+597D, the "hello" prefix of the two-word string above. */
    static const char NEEDLE_UTF8[] = "\xE4\xBD\xA0\xE5\xA5\xBD";
    char got[256], needle_gbk[64], *p;
    int checks = 0;
    long expect, hit;

    transcript_clear();
    transcript_push(TEXT_UTF8, (int)sizeof TEXT_UTF8 - 1);
    transcript_end();
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);

    lz_gbk_from_utf8(NEEDLE_UTF8, (int)sizeof NEEDLE_UTF8 - 1, needle_gbk,
                     (int)sizeof needle_gbk, NULL);
    /* The "right" offset is read off the SAME bytes transcript_find
       searches (a plain strstr on what GetWindowTextA actually
       returned), not hand-computed from the UTF-8 source - hand-
       computing it would mean assuming how many character-index units
       RichEdit spends per GBK double-byte character, which is exactly
       the question this check exists to answer rather than assume. */
    p = strstr(got, needle_gbk);
    expect = p ? (long)(p - got) : -1;

    hit = transcript_find(needle_gbk, 0, 1, 0);
    st_check(f, expect >= 0 && hit == expect,
             "find: a Chinese substring is located at the right offset");
    if (hit != expect) fprintf(f, "  got %ld want %ld\n", hit, expect);
    checks++;

    /* Explicit layout-independence gate - see transcript_find's own
       comment for exactly what was verified. The check just
       above reddens on a machine that boots with a non-Chinese layout;
       a machine that boots already-Chinese would pass it whether or
       not the fix works, since EM_FINDTEXTA would coincidentally agree
       with EM_FINDTEXTW there too. This block forces the worst case
       regardless of this host's own boot state: it switches the
       keyboard layout to English US BEFORE the text is written, so it
       reproduces the findprobe matrix's seed=en rows (the ones that
       actually redden). The layout switch is per-thread, nothing
       global is touched, and it is restored below; a fresh
       transcript_clear/push/end runs under the forced layout and a
       fresh expect is computed off what that seed actually produced
       (not reused from the first check's own seed, which may have run
       under a different layout entirely).
       !en || ...: if this host has no English layout installed to
       load at all (plausible on a single-language machine, which is
       exactly the real target's own likely configuration), there is
       nothing here to force - checks stays a fixed number across
       machines either way, the same way the real-model block further
       down already has to. */
    {
        HKL orig = lz_kbd_layout_get();   /* compat40.c - see its own
                                              comment for why this is
                                              not called directly here */
        HKL en = LoadKeyboardLayoutA("00000409", 0);
        long expect2 = expect;

        if (en) {
            ActivateKeyboardLayout(en, 0);
            transcript_clear();
            transcript_push(TEXT_UTF8, (int)sizeof TEXT_UTF8 - 1);
            transcript_end();
            got[0] = '\0';
            GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);
            p = strstr(got, needle_gbk);
            expect2 = p ? (long)(p - got) : -1;
        }
        hit = transcript_find(needle_gbk, 0, 1, 0);
        st_check(f, !en || (expect2 >= 0 && hit == expect2),
                 "find: a Chinese substring is found when it was "
                 "written under a forced English keyboard layout");
        if (en && hit != expect2)
            fprintf(f, "  got %ld want %ld\n", hit, expect2);
        checks++;
        if (en && orig) ActivateKeyboardLayout(orig, 0);
    }

    /* Not 0: 0 is a legitimate hit at the very start of the transcript,
       and a search that returns 0 for "not found" would scroll the
       user to the top and look exactly like a match at position 0. */
    hit = transcript_find("kk98-find-not-present-xyzzy", 0, 1, 0);
    st_check(f, hit == -1,
             "find: a string that is not there reports not-found "
             "rather than 0");
    checks++;

    transcript_clear();
    return checks;
}

/* ---- worker: the threading contract, with a scripted producer ----

   No engine here on purpose. What is being checked is that bytes cross
   the thread boundary exactly once, that stop is honoured, and that
   every posted buffer is freed - none of which a real generator would
   make easier to observe, and all of which are the things that corrupt
   memory rather than merely look wrong. */

/* Substring search over bytes with an explicit length: the haystack can
   contain anything, and strstr would stop at the first NUL. */
static int memmem_present(const char *hay, int hlen, const char *needle,
                          int nlen) {
    int i;
    if (nlen <= 0 || hlen < nlen) return 0;
    for (i = 0; i + nlen <= hlen; i++)
        if (memcmp(hay + i, needle, (size_t)nlen) == 0) return 1;
    return 0;
}

typedef struct { const char *s; int len; int chunk; } StJob;

static int st_job_finite(void *ud, LZTokenSink sink, LZShouldContinue cont,
                         void *cb, char *errbuf, int errlen) {
    StJob *j = (StJob *)ud;
    int pos = 0;
    (void)errbuf; (void)errlen;
    while (pos < j->len) {
        int take = j->len - pos < j->chunk ? j->len - pos : j->chunk;
        if (cont && !cont(cb)) return 0;
        sink(j->s + pos, take, cb);
        pos += take;
    }
    return 0;
}

/* gate 4:
   emits n "tokens" - one byte each, the byte itself unused, sink only
   needs to fire - all in a tight loop with no delay between them (real
   wall-clock time, deliberately not mocked: a tight loop of a handful
   of iterations completes in well under the 500ms window on any real
   host, which is what makes this a genuine test of the throttle rather
   than a hope about it). Before each sink() call, writes a token-
   number marker into *ins so the FINAL WM_APP_INSPECT posted can be
   checked against "the last token's data", not merely "some token's
   data" - a job that just left ins alone could not tell those two
   apart.

   After the burst it blocks, exactly like st_job_blocking just below -
   not incidental copy-paste, but the one thing that makes the test's
   two assertions independently observable at all. The throttle's one
   send and thread_main's own unconditional finishing send both happen
   in response to this same job, and without a pause between "burst
   done" and "job returns" the two would land in the queue microseconds
   apart with nothing for a test to peek at in between - a PeekMessage
   call could never tell "the burst produced one message" apart from
   "the burst produced one message AND the job had already finished by
   the time anyone looked". The pause turns that into two separate,
   individually inspectable moments. */
typedef struct { int n; LZInspect *ins; } StInspectJob;

static int st_job_inspect(void *ud, LZTokenSink sink, LZShouldContinue cont,
                          void *cb, char *errbuf, int errlen) {
    StInspectJob *j = (StInspectJob *)ud;
    int i;
    (void)errbuf; (void)errlen;
    for (i = 0; i < j->n; i++) {
        if (cont && !cont(cb)) return 0;
        j->ins->n_survived = i + 1;   /* the marker - see this function's
                                          own comment */
        sink("x", 1, cb);
    }
    while (cont && cont(cb)) Sleep(1);
    return 0;
}

/* Emits one piece, then refuses to end until stopped. That is what makes
   the stop test a test: a job on a timer can finish on its own and the
   check passes without stop having done anything. */
static int st_job_blocking(void *ud, LZTokenSink sink, LZShouldContinue cont,
                           void *cb, char *errbuf, int errlen) {
    StJob *j = (StJob *)ud;
    (void)errbuf; (void)errlen;
    if (cont && !cont(cb)) return 0;
    sink(j->s, j->chunk, cb);
    while (cont && cont(cb)) Sleep(1);
    return 0;
}

/* Like st_job_blocking, but writes through lz_gui_session_append_reply
 * first, the way the real job (lz_gui_session_job -> lz_session_job ->
 * session_sink in common/session.c) accumulates into g.sess before ever
 * reaching the UI sink. st_job_blocking alone cannot exercise the
 * mid-clear drain check below it: it never touches g.sess, so
 * g.sess.reply_len stays 0 and lz_gui_session_end() has nothing to push
 * regardless of when the WM_APP_GEN_DONE behind it gets drained - the
 * very thing that check needs to be sensitive to. */
static int st_job_session_blocking(void *ud, LZTokenSink sink,
                                   LZShouldContinue cont, void *cb,
                                   char *errbuf, int errlen) {
    static const char REPLY[] = "an interrupted reply";
    (void)ud; (void)errbuf; (void)errlen;
    if (cont && !cont(cb)) return 0;
    lz_gui_session_append_reply(&g.sess, REPLY, (int)sizeof REPLY - 1);
    sink(REPLY, (int)sizeof REPLY - 1, cb);
    while (cont && cont(cb)) Sleep(1);
    return 0;
}

static int st_job_fails(void *ud, LZTokenSink sink, LZShouldContinue cont,
                        void *cb, char *errbuf, int errlen) {
    (void)ud; (void)sink; (void)cont; (void)cb;
    if (errbuf && errlen > 0) {
        strncpy(errbuf, "scripted failure", (size_t)errlen - 1);
        errbuf[errlen - 1] = '\0';
    }
    return LZ_ERR_FORWARD;
}

/* A scripted JOB_LOAD success - no file is read, model/tok/state stay
 * zeroed. finish_job's `ok = (rc==0) && lz_gui_model_ready(&g.mdl)`
 * branch needs exercising with ok actually TRUE: st_model's
 * loads all fail on purpose (a real model is needed for the success
 * path and none is used, by that function's own comment), and every
 * existing worker test above uses JOB_GENERATE. Flipping the three
 * have_* flags is what lz_gui_model_ready checks and nothing else -
 * everything finish_job's ok-branch reads off a model this way
 * (state.bytes_alloc, dir via lz_gui_model_name) tolerates the zeroed
 * rest, and lz_gui_model_unload is documented safe on a zeroed
 * struct, so tearing this back down afterward needs no special case. */
static int st_job_load_ok(void *ud, LZTokenSink sink, LZShouldContinue cont,
                          void *cb, char *errbuf, int errlen) {
    LZGuiModel *m = (LZGuiModel *)ud;
    (void)sink; (void)cont; (void)cb; (void)errbuf; (void)errlen;
    m->have_model = m->have_tok = m->have_state = 1;
    return 0;
}

/* Pump until the job reports done. `stop_at_first` requests a stop when
   the first token message shows up - from the UI thread, which is where
   the stop button lives. */
static int st_pump(HWND hwnd, int stop_at_first) {
    MSG msg;
    /* Named, and deliberately NOT the window's display tick: this one
       is a watchdog whose message the loop below eats itself, so it
       never reaches the window procedure and drives nothing. Reusing
       LZ_UI_TIMER's id here would cancel the real tick. */
    SetTimer(hwnd, LZ_ST_PUMP_TIMER, 20000, NULL);
    while (!g.done_seen && GetMessage(&msg, NULL, 0, 0) > 0) {
        if (msg.message == WM_APP_TOKENS && stop_at_first) {
            lz_worker_request_stop();
            stop_at_first = 0;
        }
        if (msg.message == WM_TIMER && msg.wParam == LZ_ST_PUMP_TIMER) break;
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    KillTimer(hwnd, LZ_ST_PUMP_TIMER);
    return g.done_seen;
}

static int st_worker(FILE *f, HWND hwnd) {
    static const char SCRIPT[] =
        "the quick brown fox jumps over the lazy dog, twice over";
    StJob job;
    char got[512], want[512];
    int checks = 0;

    job.s = SCRIPT;
    job.len = (int)sizeof SCRIPT - 1;
    job.chunk = 5;

    /* 1. A job that runs to completion. */
    transcript_clear();
    /* The context-cell wiring, planted here rather than as its own
       block: no model ever becomes ready in selftest, so g.ctx_tokens
       reads -1
       both before AND after a correctly-wired finish_job call - a
       before/after comparison of the VALUE cannot tell "update_ctx_cell
       ran and computed -1 again" apart from "finish_job's call site was
       never there at all" (verified: deleting that call site left this
       whole file's checks 176 fails 0, completely unseen). A sentinel
       sidesteps the value entirely: set
       ctx cell to prose no code path would ever write, and ask whether
       finish_job's tail touched it AT ALL, regardless of what it wrote. */
    if (g.status_is_sbar)
        sb_text(1, "kk98-ctx-sentinel");
    st_check(f, start_job(hwnd, st_job_finite, &job, JOB_GENERATE) == 0,
             "worker: a job starts"); checks++;
    st_check(f, st_pump(hwnd, 0), "worker: the job reports done"); checks++;
    if (g.status_is_sbar) {
        char cell[64];
        cell[0] = '\0';
        SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETTEXT, 1, (LPARAM)cell);
        st_check(f, strcmp(cell, "kk98-ctx-sentinel") != 0,
                 "status: finish_job's tail really calls update_ctx_cell");
        checks++;
    }
    st_check(f, g.done_rc == 0, "worker: a clean job reports rc 0"); checks++;
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);
    /* The trailing break is part of the contract, not noise to tolerate:
       a reply that does not end a line runs into the next turn's
       speaker label. It is asserted here rather than trimmed away.
       TWO newlines, not one (turn spacing - see finish_job's own
       comment): the reply's own line ending, then a blank line
       before whatever comes next. */
    sprintf(want, "%s\r\n\r\n", SCRIPT);
    st_check(f, strcmp(got, want) == 0,
             "worker: every byte crossed the thread boundary, in order");
    if (strcmp(got, want) != 0) fprintf(f, "  got %s\n", got);
    checks++;
    st_check(f, lz_worker_posts_sent() == lz_worker_posts_freed(),
             "worker: every posted buffer was freed exactly once");
    fprintf(f, "  sent %ld freed %ld dropped %ld\n", lz_worker_posts_sent(),
            lz_worker_posts_freed(), lz_worker_posts_dropped());
    checks++;
    st_check(f, lz_worker_posts_dropped() == 0,
             "worker: no token buffer was dropped"); checks++;
    st_check(f, lz_worker_posts_sent() > 1,
             "worker: the run really was chunked, so order was tested");
    checks++;
    /* Spec 3.1: the token counter counted each sink() call, and the
       last tokens_arrived carried it into g.tok_gen. SCRIPT is 55 bytes
       at chunk 5, so st_job_finite calls sink exactly 11 times (11 full
       chunks, 0 remainder) - the count is that number, NOT the 55
       bytes a byte-counting implementation would report, which is
       exactly the CJK-bloat this counter exists to avoid. Deleting the
       w.tokens++ line in worker_sink, or tokens_arrived's read of it,
       each reddens one of these. */
    st_check(f, lz_worker_tokens_sent() == 11,
             "spec 3.1: the worker counted 11 sink calls for 55 bytes "
             "in 5-byte chunks, not 55");
    if (lz_worker_tokens_sent() != 11)
        fprintf(f, "  worker counted %ld\n", lz_worker_tokens_sent());
    checks++;
    st_check(f, g.tok_gen == 11,
             "spec 3.1: g.tok_gen carries the worker's token count into "
             "the status cell");
    if (g.tok_gen != 11) fprintf(f, "  g.tok_gen %d\n", g.tok_gen);
    checks++;
    /* The throughput cell is LIVE ONLY while a generate job runs.
       finish_job clears tok_live the moment the job ends, so the
       status line returns to its resting text - keeping the final
       reading until the next round would leave the bar stuck at
       "N tok, X tok/s" after a reply, which reads as the status not
       updating. Deleting finish_job's clear reddens this. */
    st_check(f, g.tok_live == 0,
             "spec 3.1: a finished generation clears the throughput "
             "cell, so the status line returns to its resting text");
    checks++;
    /* The status line, after a finished generation, is back to the
       RESTING text - not the "N tok, X tok/s" cell. Only when the status
       bar is comctl32's (a plain STATIC control has no parts to read
       back); the tok_live field check above guards the value regardless
       of which control is present, this one guards the DISPLAY. */
    if (g.status_is_sbar) {
        char cell[64];
        cell[0] = '\0';
        SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETTEXT, 0, (LPARAM)cell);
        st_check(f, strstr(cell, "tok/s") == NULL,
                 "spec 3.1: after the job ends, part 0 is back to the "
                 "resting text, not the throughput cell");
        if (strstr(cell, "tok/s") != NULL)
            fprintf(f, "  part 0 reads [%s]\n", cell);
        checks++;
    }
    /* PREFILL OUTRANKS THE THROUGHPUT CELL, both windows.
       tok_live is set by start_job - before the first token exists - so
       without set_status's prefill_active() gate the cell answered
       "0 tok, 0.0 tok/s" for the whole prefill AND discarded the text
       the caller passed. The bar moved (it is driven by SETPOS) while
       neither the strip nor the sidebar said what was happening.
       Both directions are asserted: without the second one this would
       also pass on a build that never shows the cell at all. */
    {
        const char *probe = "kk98-prefill-probe";
        int was_live = g.tok_live, was_done = g_pf_done,
            was_total = g_pf_total;
        char cell[80], side[512];

        g.tok_live = 1;
        g_pf_done = 3;
        g_pf_total = 10;
        set_status(probe);
        /* BOTH strips, read the way each one stores text: comctl32 has
           parts, the fallback is an ordinary window set_status writes
           with SetWindowTextA. Reading only the first would leave this
           untested on exactly the path the degraded front end uses. */
        st_status_text(cell, (int)sizeof cell);
        st_check(f, strstr(cell, probe) != NULL &&
                    strstr(cell, "tok/s") == NULL,
                 "prefill: the progress text holds the status line "
                 "against the throughput cell");
        if (strstr(cell, probe) == NULL)
            fprintf(f, "  part 0 reads [%s]\n", cell);
        checks++;

        if (g.part[LZ_GUI_SIDE_INFO]) {
            side[0] = '\0';
            GetWindowTextA(g.part[LZ_GUI_SIDE_INFO], side, (int)sizeof side);
            st_check(f, strstr(side, probe) != NULL,
                     "prefill: the sidebar mirror shows it too");
            if (strstr(side, probe) == NULL)
                fprintf(f, "  sidebar reads [%s]\n", side);
            checks++;
        }

        g_pf_done = g_pf_total = 0;
        set_status(probe);
        st_status_text(cell, (int)sizeof cell);
        st_check(f, strstr(cell, "tok/s") != NULL,
                 "prefill: once it ends the throughput cell takes the "
                 "line back");
        if (strstr(cell, "tok/s") == NULL)
            fprintf(f, "  part 0 reads [%s]\n", cell);
        checks++;

        /* THE PERIODIC TICK MUST NOT TAKE THE LINE EITHER. It runs for
           the whole job, so during prefill it used to hand set_status a
           constant that knows nothing about the phase - the gate inside
           set_status cannot see that, because the caller supplied the
           text rather than letting the throughput cell substitute one.
           ui_tick deciding the phase once is what this asserts.
           Reddens if that branch goes back to an unconditional
           set_status(GENERATING). */
        g.tok_live = 1;
        g.job_kind = JOB_GENERATE;
        g_pf_done = 3;
        g_pf_total = 10;
        SendMessage(hwnd, WM_TIMER, (WPARAM)LZ_UI_TIMER, 0);
        g.job_kind = JOB_NONE;
        st_status_text(cell, (int)sizeof cell);
        st_check(f, strstr(cell, "3/10") != NULL,
                 "prefill: the token tick refreshes the progress line "
                 "instead of overwriting it");
        if (strstr(cell, "3/10") == NULL)
            fprintf(f, "  part 0 reads [%s]\n", cell);
        checks++;

        /* TWO SEGMENTS, ONE BAR. A prefix-cache miss prefills in two
           goes and each reports its own 0..n through the same callback;
           taken raw the bar ran to 100% and jumped back to 0%. `done`
           must only ever move forward. Reddens if g_pf_base goes. */
        {
            int seq_ok = 1, prev;
            g_pf_base = g_pf_seen = g_pf_done = g_pf_total = 0;
            gui_prefill_progress(0, 100, NULL);
            gui_prefill_progress(60, 100, NULL);
            prev = g_pf_done;
            gui_prefill_progress(100, 100, NULL);
            if (g_pf_done < prev) seq_ok = 0;
            prev = g_pf_done;
            gui_prefill_progress(0, 5, NULL);      /* second segment */
            if (g_pf_done < prev) seq_ok = 0;
            if (g_pf_total != 105) seq_ok = 0;
            prev = g_pf_done;
            gui_prefill_progress(5, 5, NULL);
            if (g_pf_done < prev || g_pf_done != 105) seq_ok = 0;
            st_check(f, seq_ok,
                     "prefill: a second segment continues the bar "
                     "instead of restarting it");
            if (!seq_ok)
                fprintf(f, "  ended at %d/%d\n", g_pf_done, g_pf_total);
            checks++;
            g_pf_base = g_pf_seen = 0;
        }

        g.tok_live = was_live;
        g_pf_done = was_done;
        g_pf_total = was_total;
        /* set_status, not set_idle_status: the latter would strncpy
           g.idle_status onto itself. Same visible result - tok_live is
           back to what finish_job left it. */
        set_status(g.idle_status);
    }
    st_check(f, !lz_worker_busy(), "worker: not busy after the job ends");
    checks++;

    /* 2. Stop. The job cannot end by itself, so finishing at all proves
          the flag crossed threads and the job checked it. */
    transcript_clear();
    st_check(f, start_job(hwnd, st_job_blocking, &job, JOB_GENERATE) == 0,
             "worker: the blocking job starts"); checks++;
    /* The ON branch of cmd_enable(IDM_STOP_GEN, ...), read off the real
       menu while a job is genuinely running - not while nothing is
       happening, which is all the idle-state check below covers. This is
       what a review found missing: start_job's cmd_enable call had never
       once been asserted with a live job behind it, only finish_job's
       "turn it back off" call had a check (the idle-state block further
       down). Checked here, right after start_job returns, because
       start_job makes the cmd_enable call synchronously - no message has
       to be pumped first for this to be true. */
    {
        HMENU bar = GetMenu(hwnd);
        UINT st = bar ? GetMenuState(bar, IDM_STOP_GEN, MF_BYCOMMAND)
                      : (UINT)-1;
        st_check(f, st != (UINT)-1 && !(st & MF_GRAYED),
                 "worker: Stop ungreys while a generate job is running");
        checks++;
    }
    st_check(f, lz_worker_start(hwnd, st_job_finite, &job, NULL) != 0,
             "worker: a second start is refused while one runs"); checks++;
    st_check(f, st_pump(hwnd, 1), "worker: stop ends the job"); checks++;
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);
    /* TWO trailing newlines (turn spacing) - a stopped job
       still ends the way any other JOB_GENERATE ending does, backfill
       and all (finish_job's own "stop included" comment, unchanged by
       this). */
    sprintf(want, "%.*s\r\n\r\n", job.chunk, SCRIPT);
    st_check(f, strcmp(got, want) == 0,
             "worker: a stopped job leaves a prefix, not a truncation");
    if (strcmp(got, want) != 0) fprintf(f, "  got %s\n", got);
    checks++;
    st_check(f, lz_worker_posts_sent() == lz_worker_posts_freed(),
             "worker: buffers balance after a stop"); checks++;

    /* 3. A failing job's code and message reach the window. */
    transcript_clear();
    st_check(f, start_job(hwnd, st_job_fails, &job, JOB_GENERATE) == 0,
             "worker: the failing job starts"); checks++;
    st_check(f, st_pump(hwnd, 0), "worker: the failing job reports done");
    checks++;
    st_check(f, g.done_rc == LZ_ERR_FORWARD,
             "worker: the job's return code arrives intact"); checks++;
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_STATUS], got, (int)sizeof got);
    st_check(f, strcmp(got, "scripted failure") == 0,
             "worker: the job's message reaches the status line");
    if (strcmp(got, "scripted failure") != 0) fprintf(f, "  got %s\n", got);
    checks++;

    /* 4. And the window recovers: a fourth job runs normally. */
    transcript_clear();
    st_check(f, start_job(hwnd, st_job_finite, &job, JOB_GENERATE) == 0,
             "worker: a job runs again after a failed one"); checks++;
    st_check(f, st_pump(hwnd, 0) && g.done_rc == 0,
             "worker: the fourth job completes cleanly"); checks++;
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_STATUS], got, (int)sizeof got);
    /* NOT "the status line is back to 'no model'". Spec 3.1 leaves the
       throughput cell up after a successful generation ("keep the final
       reading until the next round"), and this selftest runs with no
       model and no real tokens, so a clean fourth run's part 0 is
       "N tok, 0.0 tok/s", not the resting text. What this check is
       FOR is that the previous job's FAILURE message does not survive
       into the next job's outcome - the failure was written to part 0
       by finish_job's error branch, and a correct success path must
       have overwritten it. */
    st_check(f, strcmp(got, "scripted failure") != 0,
             "worker: a clean run's status does not carry the previous "
             "job's error");
    if (strcmp(got, "scripted failure") == 0) fprintf(f, "  got %s\n", got);
    checks++;

    /* 5. Clearing mid-generation - the lz_worker_join_drain regression
          (worker.h). IDM_CLEAR is exercised through SendMessage, not
          PostMessage: on the window's own thread, SendMessage calls
          wndproc directly and synchronously, the same way a real menu
          click or accelerator does, so no GetMessage/DispatchMessage
          loop runs underneath it. Nothing pumps the worker's queue here
          except whatever IDM_CLEAR's own handler does - which is
          exactly what the checks below need to be sensitive to.

          Three things are checked, and they are NOT redundant:
          - g.done_seen catches the regression directly: with a plain
            join (no drain), the job's GEN_DONE is still undispatched
            the instant SendMessage returns, so done_seen is still 0.
          - the transcript catches the user-visible half: an undrained
            WM_APP_TOKENS writes the interrupted reply, and finish_job's
            own transcript_push("\r\n"), into the box the user just
            watched go blank - "cleared, but half a sentence pops back
            up".
          - hist.n == 0 is asserted too, but by inspection it is NOT
            independently sensitive to this regression: IDM_CLEAR resets
            reply_len along with hist, so even an undrained,
            belatedly-dispatched GEN_DONE finds lz_gui_session_end()
            with nothing to push. It is kept as a real invariant (the
            clear must leave no history behind) and as a tripwire should
            that reset ever stop covering reply_len - not as evidence
            against this particular bug. */
    transcript_clear();
    lz_gui_session_reset(&g.sess);
    st_check(f, start_job(hwnd, st_job_session_blocking, NULL,
                          JOB_GENERATE) == 0,
             "worker: mid-clear setup - a job starts"); checks++;
    {
        /* PM_NOREMOVE: wait for the one chunk to actually be queued,
           without dispatching it - dispatching it here would be the very
           pump this check exists to route around. */
        MSG peek;
        int waited = 0;
        while (!PeekMessage(&peek, hwnd, WM_APP_TOKENS, WM_APP_TOKENS,
                            PM_NOREMOVE) && waited < 5000) {
            Sleep(1);
            waited++;
        }
        st_check(f, PeekMessage(&peek, hwnd, WM_APP_TOKENS, WM_APP_TOKENS,
                                PM_NOREMOVE) != 0,
                 "worker: mid-clear setup - a token message is queued, "
                 "undispatched");
        checks++;
    }
    SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_CLEAR, 0), 0);
    st_check(f, g.done_seen == 1,
             "worker: clearing mid-generation reaps the interrupted job "
             "before IDM_CLEAR returns, not on a later turn of the "
             "message loop"); checks++;
    st_check(f, g.sess.hist.n == 0,
             "worker: clearing mid-generation does not push the old "
             "reply into the new history"); checks++;
    /* Harmless once IDM_CLEAR already drained: done_seen is already 1
       from inside the SendMessage above, so this returns at once
       without blocking. Under the regression it is what finally lets
       the still-queued GEN_DONE arrive - run before the transcript
       check below so that check is not passing merely because nothing
       has been dispatched yet. */
    st_pump(hwnd, 0);
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);
    st_check(f, got[0] == '\0',
             "worker: clearing mid-generation leaves no leftover text in "
             "the transcript");
    if (got[0] != '\0') fprintf(f, "  got %s\n", got);
    checks++;

    /* The IDM_CLEAR call site, idle this time - not the
       mid-generation case just above. That case cannot isolate it: the
       command's own lz_worker_join_drain reaps the interrupted job's
       GEN_DONE, which runs finish_job (already gated above), and
       finish_job's OWN update_ctx_cell call consumes any sentinel
       planted beforehand before IDM_CLEAR's explicit call ever gets a
       chance to - confirmed by running the sentinel there first: it
       passed even with IDM_CLEAR's own call deleted, for exactly this
       reason. Idle, nothing to drain, isolates the one line this check
       is actually about. */
    if (g.status_is_sbar)
        sb_text(1, "kk98-ctx-sentinel");
    SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_CLEAR, 0), 0);
    if (g.status_is_sbar) {
        char cell2[64];
        cell2[0] = '\0';
        SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETTEXT, 1, (LPARAM)cell2);
        st_check(f, strcmp(cell2, "kk98-ctx-sentinel") != 0,
                 "status: IDM_CLEAR's handler really calls update_ctx_cell");
        checks++;
    }

    /* 6. job_kind misattribution - the /load path (LZ_CMD_LOAD). Of the
          three call sites that start a NEW job while an old one may
          still be finishing (open_model, open_model_dir, /load), this
          is the only one with no modal dialog in front of it: the other
          two show a folder browser first, and that browser's own
          message pump can accidentally drain the old job before
          start_job ever runs. /load <path> goes straight to start_job,
          so it is the one call site where an undrained join is not
          accidentally safe.

          Without the drain: start_job overwrites g.job_kind to
          JOB_LOAD before the interrupted GENERATE job's belated
          GEN_DONE is dispatched, so finish_job reads JOB_LOAD for the
          OLD job's result - and because finish_job unconditionally
          resets g.job_kind to JOB_NONE regardless of which branch it
          took, that misattributed call also consumes the "a job just
          finished" state. When the REAL load job's own GEN_DONE
          arrives afterward, g.job_kind is already JOB_NONE, so neither
          branch in finish_job matches and its result is dropped
          silently - g.load_failed is never written for it.

          g.done_rc is what is asserted, not g.load_failed: with the
          fake one-line "model.bin" below, lz_gui_model_load_job
          fails either way (see st_model), so load_failed ends up 1
          whichever job's completion sets it - not a discriminating
          signal by itself. But st_pump stops at the FIRST WM_APP_DONE
          it dispatches, and that is a real difference: under the
          regression it is the misattributed OLD job's clean stop
          (rc 0), and the real load job's own DONE is left undispatched
          in the queue; drained, it is the real load job's own failure
          (rc != 0, no old job left to misattribute). */
    {
        char dir[MAX_PATH], bin[MAX_PATH], cmd[MAX_PATH + 8];
        FILE *out;

        GetTempPathA((DWORD)sizeof dir, dir);
        strcat(dir, "kunkun98-selftest-load");
        CreateDirectoryA(dir, NULL);
        /* Forward slash, matching lz_gui_model_dir_ok (which builds
           "%.500s/model.bin") - Win32 accepts both, but the pre-check and
           the selftest must agree on the SAME path or the test only
           passes by the grace of backslash==slash. */
        sprintf(bin, "%s/model.bin", dir);
        out = fopen(bin, "wb");
        if (out) { fwrite("not a model", 1, 11, out); fclose(out); }

        transcript_clear();
        lz_gui_session_reset(&g.sess);
        g.load_failed = 0;
        st_check(f, start_job(hwnd, st_job_blocking, &job,
                              JOB_GENERATE) == 0,
                 "worker: mid-load setup - a generate job starts");
        checks++;
        {
            MSG peek;
            int waited = 0;
            while (!PeekMessage(&peek, hwnd, WM_APP_TOKENS, WM_APP_TOKENS,
                                PM_NOREMOVE) && waited < 5000) {
                Sleep(1);
                waited++;
            }
            st_check(f, PeekMessage(&peek, hwnd, WM_APP_TOKENS,
                                    WM_APP_TOKENS, PM_NOREMOVE) != 0,
                     "worker: mid-load setup - a token message is "
                     "queued, undispatched");
            checks++;
        }
        sprintf(cmd, "/load %s", dir);
        do_command(hwnd, cmd, (int)strlen(cmd));
        st_pump(hwnd, 0);
        st_check(f, g.done_rc != 0,
                 "worker: /load reaps the interrupted generate job "
                 "before starting the load, so the load job's own "
                 "failure - not the old job's clean stop - is what "
                 "st_pump sees first");
        if (g.done_rc == 0) fprintf(f, "  done_rc 0 (old job, misread)\n");
        checks++;
        /* Cleanup, not part of the assertion: whichever job st_pump
           above did not reach is still live or still queued under the
           regression, and this leaves neither behind for later checks
           to trip over. Harmless no-op once everything above is
           already drained. */
        lz_worker_join_drain(hwnd);
        g.load_failed = 0;

        DeleteFileA(bin);
        RemoveDirectoryA(dir);
    }

    /* 7. finish_job's JOB_LOAD branch, with ok genuinely TRUE - not just
          cmd_enable(ID_SEND, 1) called directly (the "enable: Send
          ungreys..." check above tests cmd_enable's own on-branch, not
          finish_job's decision to call it). st_job_load_ok scripts a
          successful load without a real model, so this is the first
          place in this file finish_job's ok computation itself is
          exercised with ok actually 1.

          Send's own enabled state stopped being a discriminator here
          once cmd_enable(ID_SEND, 1) became unconditional in finish_job
          (the WM_COMMAND-bypass fix - a failed load leaves Send just
          as meaningful as a successful one, for /load and /help): both
          branches now call it the same way, so checking it here would
          pass whether ok's computation ran at all. The idle status
          text is what still depends on ok, so that is the assertion
          below. The paired block right after drives st_job_fails
          instead (ok genuinely FALSE) and checks the one thing that
          DOES still have to hold either way - Send stays enabled;
          losing that second block would let a regression back to
          cmd_enable(ID_SEND, ok) pass unnoticed, since this block's own
          ok is always 1. */
    {
        memset(&g.mdl, 0, sizeof g.mdl);
        strncpy(g.mdl.dir, "C:\\kk98-scripted-load", sizeof g.mdl.dir - 1);
        EnableWindow(g.part[LZ_GUI_SEND], FALSE);   /* known baseline */
        st_check(f, start_job(hwnd, st_job_load_ok, &g.mdl, JOB_LOAD) == 0,
                 "worker: a scripted load job starts"); checks++;
        st_check(f, st_pump(hwnd, 0),
                 "worker: the scripted load reports done"); checks++;
        st_check(f, strcmp(g.idle_status,
                           lz_str_utf8(LZ_STR_STATE_READY)) == 0,
                 "enable: finish_job's ok branch sets the READY idle "
                 "status after a real load succeeds");
        checks++;

        /* Torn all the way back down: every side effect finish_job's
           ok branch has (the MRU push, the resting status text, the
           sidebar model line, g.ctx_tokens) has to be undone, or later
           checks in this file - several assume "no model" - would be
           reading state this scripted success left behind rather than
           their own setup. Send is NOT reset to FALSE here: idle state
           IS Send-enabled now, so leaving it live is the correct
           restore, not a leftover. */
        lz_mru_remove(&g.mru, g.mdl.dir);
        lz_gui_model_unload(&g.mdl);
        memset(g.mdl.dir, 0, sizeof g.mdl.dir);
        g.load_failed = 0;
        g.side_model[0] = '\0';
        build_menu_bar(hwnd);
        EnableWindow(g.part[LZ_GUI_SEND], TRUE);
        set_idle_status(lz_str_utf8(LZ_STR_STATE_NO_MODEL));
        update_ctx_cell();
        /* One more side effect finish_job's ok branch has, same as
           every other line in this teardown: the
           candidate panel side_panel_sync built. lz_gui_model_unload
           just above already made lz_gui_model_ready(&g.mdl) false, so
           this call tears the panel back down - it does not run
           itself, unlike everything finish_job calls through the
           worker's own message. */
        side_panel_sync(hwnd);
    }

    /* The paired case block #7's own comment promises: a scripted load
       that FAILS (ok genuinely FALSE), checking that Send stays enabled
       anyway - the actual new behaviour the WM_COMMAND-bypass fix
       introduced. st_job_fails already exists (st_model's own JOB_
       GENERATE checks use it); nothing about it is generate-specific,
       so it doubles as a failing JOB_LOAD here. */
    {
        EnableWindow(g.part[LZ_GUI_SEND], FALSE);   /* known baseline */
        st_check(f, start_job(hwnd, st_job_fails, NULL, JOB_LOAD) == 0,
                 "worker: a scripted failing load job starts"); checks++;
        st_check(f, st_pump(hwnd, 0),
                 "worker: the scripted failing load reports done");
        checks++;
        st_check(f, IsWindowEnabled(g.part[LZ_GUI_SEND]) != 0,
                 "enable: finish_job leaves Send enabled after a load "
                 "fails too");
        checks++;
        g.load_failed = 0;
        set_idle_status(lz_str_utf8(LZ_STR_STATE_NO_MODEL));
    }

    /* 8. Gate 5 ("degenerate cases"): num_experts == 0 must not
          create a single lamp control, and the candidate list must sit
          HIGHER (a smaller y) than it does when a lamp grid IS present
          - "the candidate list moves up to fill the space", checked as
          the actual comparison the words describe, not just "some rect
          is zero". st_job_load_ok does
          not touch config at all (only the have_* flags), so
          config.num_experts is set by hand before each start_job call,
          the same way block #7 above sets g.mdl.dir by hand. */
    {
        RECT cand_no_lamps, cand_with_lamps;
        int i, any_elamp;

        /* 8a. num_experts == 0 - a model, but no MoE. */
        memset(&g.mdl, 0, sizeof g.mdl);
        g.mdl.model.config.num_experts = 0;
        strncpy(g.mdl.dir, "C:\\kk98-scripted-load-plain",
               sizeof g.mdl.dir - 1);
        st_check(f, start_job(hwnd, st_job_load_ok, &g.mdl, JOB_LOAD) == 0,
                 "worker: a scripted MoE-less load job starts"); checks++;
        st_check(f, st_pump(hwnd, 0),
                 "worker: the scripted MoE-less load reports done");
        checks++;

        any_elamp = 0;
        for (i = 0; i < 16; i++) if (g.elamp[i]) any_elamp = 1;
        st_check(f, !any_elamp,
                 "panel: num_experts == 0 creates no expert lamps at "
                 "all"); checks++;
        st_check(f, g.part[LZ_GUI_SIDE_CAND] != NULL,
                 "panel: the candidate list itself still exists with "
                 "no MoE"); checks++;
        /* Without LBS_USETABSTOPS the LB_SETTABSTOPS call below never
           takes effect and '\t' does not expand, so the probability
           column lands glued onto the token text ("<|im_end|>1.0000"
           in a user's own screenshot). No pixel judge is available in
           this environment (no interactive desktop), but the STYLE BIT
           itself is directly readable, so that is what this asserts,
           rather than anything about how it renders. */
        st_check(f, (GetWindowLong(g.part[LZ_GUI_SIDE_CAND], GWL_STYLE) &
                    LBS_USETABSTOPS) != 0,
                 "panel: the candidate LISTBOX actually honours "
                 "LB_SETTABSTOPS (LBS_USETABSTOPS is set)"); checks++;
        GetWindowRect(g.part[LZ_GUI_SIDE_CAND], &cand_no_lamps);

        lz_mru_remove(&g.mru, g.mdl.dir);
        lz_gui_model_unload(&g.mdl);
        memset(g.mdl.dir, 0, sizeof g.mdl.dir);
        g.load_failed = 0;
        g.side_model[0] = '\0';
        build_menu_bar(hwnd);
        set_idle_status(lz_str_utf8(LZ_STR_STATE_NO_MODEL));
        update_ctx_cell();
        side_panel_sync(hwnd);

        /* 8b. num_experts > 0 - the paired case, same model otherwise. */
        memset(&g.mdl, 0, sizeof g.mdl);
        g.mdl.model.config.num_experts = 16;
        strncpy(g.mdl.dir, "C:\\kk98-scripted-load-moe",
               sizeof g.mdl.dir - 1);
        st_check(f, start_job(hwnd, st_job_load_ok, &g.mdl, JOB_LOAD) == 0,
                 "worker: a scripted MoE load job starts"); checks++;
        st_check(f, st_pump(hwnd, 0),
                 "worker: the scripted MoE load reports done"); checks++;

        any_elamp = 1;
        for (i = 0; i < 16; i++) if (!g.elamp[i]) any_elamp = 0;
        st_check(f, any_elamp,
                 "panel: num_experts > 0 creates all 16 expert lamps");
        checks++;

        /* WS_EX_STATICEDGE really reached the window, not just the
           CreateWindowExA call site (the user's own "frame around each
           lamp" request) - the same "read the
           style bit back" shape the LBS_USETABSTOPS gate elsewhere in
           this file uses, for the same reason: this kind of defect
           only shows up on screen, and this environment has no pixel
           judge, so the style bit itself is what gets asserted.
           lz_ex_style can strip the bit below the 3.51 floor, which
           this host is not, so it is expected present here. */
        {
            int all_edged = 1, bad = -1;
            for (i = 0; i < 16; i++) {
                LONG ex = GetWindowLong(g.elamp[i], GWL_EXSTYLE);
                if (!(ex & WS_EX_STATICEDGE)) { all_edged = 0; bad = i; break; }
            }
            st_check(f, all_edged,
                     "panel: every expert lamp carries WS_EX_STATICEDGE "
                     "- the frame the user asked for");
            if (!all_edged) fprintf(f, "  lamp %d has no static edge\n", bad);
            checks++;
        }

        /* And that the frame did not eat the lamp. The
           check above asserts the border EXISTS; it can stay green while
           the border clips the bitmap's right two columns and
           bottom two rows, because a style bit says nothing about what
           is left to draw in. WS_EX_STATICEDGE is nonclient, a STATIC
           clips SS_BITMAP to its client area, so the invariant is
           "client area still holds the whole bitmap" - stated against
           the bitmap's own measured size rather than against
           LZ_GUI_ELAMP_PX, or it would just be re-deriving the cell
           size from itself and would follow any future mistake. */
        {
            HBITMAP lb = (HBITMAP)SendMessage(g.elamp[0], STM_GETIMAGE,
                                              IMAGE_BITMAP, 0);
            BITMAP bm;
            RECT cl;
            int fits = 0, bw = 0, bh = 0, cw = 0, ch = 0;
            if (lb && GetObjectA(lb, (int)sizeof bm, &bm) &&
                GetClientRect(g.elamp[0], &cl)) {
                bw = (int)bm.bmWidth;  bh = (int)bm.bmHeight;
                cw = cl.right - cl.left; ch = cl.bottom - cl.top;
                fits = cw >= bw && ch >= bh;
            }
            st_check(f, fits,
                     "panel: the lamp frame leaves room for the whole "
                     "bitmap - a clipped disc is not a small one");
            if (!fits)
                fprintf(f, "  bitmap %dx%d, client %dx%d\n", bw, bh, cw, ch);
            checks++;
        }

        /* elamp_tier itself, as a pure function (the three-colour
           rework) - the exact boundary the panel's own
           gate below depends on, pinned here as values rather than
           left to be re-derived from repaint_lamps' control flow.
           4 is checked alongside 3 specifically because ">= 3" and
           "== 3" agree at 3 and disagree at 4 - a mutation that
           narrowed the RED tier to exactly 3 would still pass a test
           that only tried 0/1/2/3. 255 (the saturation ceiling
           lz_moe_hits_add's own contract guarantees)
           is checked too: a count that can never exceed 255 must
           still read as RED, not overflow into some fifth reading
           this function does not have. */
        st_check(f, elamp_tier(0) == LZ_LAMP_OFF &&
                 elamp_tier(1) == LZ_LAMP_READY &&
                 elamp_tier(2) == LZ_LAMP_BUSY &&
                 elamp_tier(3) == LZ_LAMP_ERROR &&
                 elamp_tier(4) == LZ_LAMP_ERROR &&
                 elamp_tier(255) == LZ_LAMP_ERROR,
                 "panel: elamp_tier reads 0/1/2/3-or-more layers as "
                 "off/green/amber/red"); checks++;

        /* repaint_lamps' own wiring, gate: PER LAMP, checked against
           an INDEPENDENTLY written expected table (want_tier below),
           not by calling elamp_tier() a second time inside this test
           - that would only prove repaint_lamps agrees with itself,
           and a wrong threshold shared by both would still pass.
           expert_hits == 0 and "this model has no MoE" are the SAME
           picture on screen (a row of dark lamps) - side_panel_sync
           already tells them apart by never creating the array at all
           in the second case (the check above), but once the array
           DOES exist, "some lamp is lit" cannot tell a correct tally
           from one where the wrong lamp lit, or lit the wrong COLOUR.
           {0,1,2,3,5} deliberately touches every tier AND the exact
           3-vs-4 boundary (index 3 holds 3, the smallest RED value;
           index 4 holds 5, confirming RED does not stop at exactly 3)
           - not a synthetic all-or-nothing pattern, so a mutation that
           lit every lamp the same colour would already fail
           obviously. */
        {
            LZInspect probe;
            static const unsigned char HITS[16] =
                { 0, 1, 2, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            static const int WANT_TIER[16] = {
                LZ_LAMP_OFF, LZ_LAMP_READY, LZ_LAMP_BUSY, LZ_LAMP_ERROR,
                LZ_LAMP_ERROR, LZ_LAMP_OFF, LZ_LAMP_OFF, LZ_LAMP_OFF,
                LZ_LAMP_OFF, LZ_LAMP_OFF, LZ_LAMP_OFF, LZ_LAMP_OFF,
                LZ_LAMP_OFF, LZ_LAMP_OFF, LZ_LAMP_OFF, LZ_LAMP_OFF
            };
            int all_correct = 1, bad_lamp = -1;
            memset(&probe, 0, sizeof probe);
            memcpy(probe.expert_hits, HITS, sizeof HITS);
            repaint_lamps(&probe);
            for (i = 0; i < 16; i++) {
                HGDIOBJ got = (HGDIOBJ)SendMessage(g.elamp[i], STM_GETIMAGE,
                                                   IMAGE_BITMAP, 0);
                HGDIOBJ want = (HGDIOBJ)g.elamp_bmp[WANT_TIER[i]];
                if (got != want) { all_correct = 0; bad_lamp = i; break; }
            }
            st_check(f, all_correct,
                     "panel: every lamp renders the bitmap its own hit "
                     "count's tier calls for, independently"); checks++;
            if (!all_correct)
                fprintf(f, "  lamp %d wrong (hits %d, want tier %d)\n",
                       bad_lamp, HITS[bad_lamp], WANT_TIER[bad_lamp]);
        }
        GetWindowRect(g.part[LZ_GUI_SIDE_CAND], &cand_with_lamps);
        /* The comparison gate 5 is actually named for: PLAIN's
           candidate list starts higher up (smaller y) than LAMPS'
           does - not merely "different", which a coordinate system
           flip or an unrelated layout regression could also produce. */
        st_check(f, cand_no_lamps.top < cand_with_lamps.top,
                 "panel: the candidate list moves UP to fill the lamp "
                 "grid's space when there is no MoE");
        if (cand_no_lamps.top >= cand_with_lamps.top)
            fprintf(f, "  no-lamps top %ld, with-lamps top %ld\n",
                   (long)cand_no_lamps.top, (long)cand_with_lamps.top);
        checks++;

        /* experts_truncated (Part Two): a model with more
           experts than expert_hits has entries has no room to record
           them all, and team-lead's own instruction is that the panel
           must say so
           rather than silently acting as if the model only had 32.
           Checked by CONTENT (the count appears in the title), not by
           "the title changed" - a title that changed for the wrong
           reason would still pass a weaker check. */
        {
            LZInspect probe;
            char got[128];
            memset(&probe, 0, sizeof probe);
            probe.n_survived = 3;
            probe.n_experts = 40;
            probe.experts_truncated = 1;
            repaint_candidates(&probe);
            got[0] = '\0';
            if (g.cand_title) GetWindowTextA(g.cand_title, got, (int)sizeof got);
            st_check(f, strstr(got, "40") != NULL,
                     "panel: a truncated expert mask says how many "
                     "experts the model actually has, not just 32");
            if (!strstr(got, "40")) fprintf(f, "  got %s\n", got);
            checks++;
        }

        /* Untheme consistency across EVERY control this window creates
           - extends the settings dialog's own falsifiable agreement
           check from six controls to the whole main window, right here
           because this is the one place in this file both the candidate
           LISTBOX and all 16 expert lamps are guaranteed to exist at
           once. Three controls have shipped without lz_ui_untheme (the
           input box, the settings dialog's six, the candidate LISTBOX
           just above).

           READS BACK g.part_untheme_ok[]/g.elamp_untheme_ok[]/the
           three named scalars - what production ACTUALLY stored at
           each control's own creation site - rather than calling
           lz_ui_untheme fresh here. That distinction is load-bearing,
           not stylistic: lz_ui_untheme is idempotent, so a FRESH call
           in the test would "fix" a control production forgot to
           untheme and this gate would never be able to tell the two
           apart - exactly the failure mode the mutation below is
           designed to catch, and exactly why it would not have caught
           it under a fresh-probing design. A forgotten call leaves its
           stored field at the struct's zero-init value, which reads as
           0 same as a genuine failure would - indistinguishable from
           "uxtheme is absent on this host" ONLY when EVERY entry reads
           0 together; one lone 0 among a field of 1s is what "does not
           agree" is built to catch.
           NULLs skipped, not asserted absent: LZ_GUI_TOOLBAR's window
           itself can be NULL (no comctl32), and this is not the check
           for that (the floor tests further down are).

           EXCEPTION, named per this comment's own promise: LZ_GUI_
           TOOLBAR is NOT on this roster. Measured, not assumed: calling
           lz_ui_untheme on it changed what TB_GETMAXSIZE reports
           (46 -> 50 needed), which broke "the layout is tall enough for
           its buttons" outright - unlike EDIT/LISTBOX/BUTTON/STATIC, a
           comctl32 ToolbarWindow32's theme is not purely border/
           scrollbar chrome, it is load-bearing for the control's own
           button metrics. See create_children's own comment at the
           toolbar's apply_font call for the fuller account. Every
           other control on this roster unthemes the same way the
           transcript and input box do. */
        {
            int roster[32];
            int nr = 0, i3, first, all_agree;
            if (g.part[LZ_GUI_TRANSCRIPT]) roster[nr++] = g.transcript_untheme_ok;
            if (g.part[LZ_GUI_INPUT])      roster[nr++] = g.input_untheme_ok;
            if (g.part[LZ_GUI_SEND])       roster[nr++] = g.part_untheme_ok[LZ_GUI_SEND];
            if (g.part[LZ_GUI_SIDE_CHICKEN]) roster[nr++] = g.part_untheme_ok[LZ_GUI_SIDE_CHICKEN];
            if (g.part[LZ_GUI_SIDE_NAME])  roster[nr++] = g.part_untheme_ok[LZ_GUI_SIDE_NAME];
            if (g.part[LZ_GUI_SIDE_INFO])  roster[nr++] = g.part_untheme_ok[LZ_GUI_SIDE_INFO];
            if (g.part[LZ_GUI_STATUS])     roster[nr++] = g.part_untheme_ok[LZ_GUI_STATUS];
            if (g.part[LZ_GUI_SIDE_CAND])  roster[nr++] = g.cand_untheme_ok;
            if (g.cand_title)              roster[nr++] = g.cand_title_untheme_ok;
            for (i3 = 0; i3 < 16; i3++)
                if (g.elamp[i3]) roster[nr++] = g.elamp_untheme_ok[i3];

            first = nr > 0 ? roster[0] : 1;
            all_agree = 1;
            for (i3 = 1; i3 < nr; i3++)
                if (roster[i3] != first) { all_agree = 0; break; }
            st_check(f, nr >= 19 && all_agree,
                    "theme: every control this window creates unthemes "
                    "consistently (transcript, input, send, chicken, "
                    "name, info, status, candidate list, candidate "
                    "title, 16 lamps - toolbar excepted, see comment "
                    "above)");
            fprintf(f, "  untheme roster: %d controls, first=%d, "
                   "agree=%d\n", nr, first, all_agree);
            checks++;
        }

        /* Fixed white backgrounds (user request - see turn_header's
           own comment for the fuller task). Two
           DIFFERENT mechanisms, two different verifiability levels,
           stated honestly rather than pretended-equal:

           - The transcript is RichEdit: EM_SETBKGNDCOLOR. It has NO
             GET counterpart at all - the only observable side channel
             is its OWN documented return value, the PREVIOUS
             background colour. Probed with a sentinel FIRST (not by
             re-setting white and reading white back, which would pass
             even if create_children never called this - the same
             idempotent-probe trap the untheme gate hit) so the SECOND
             call's return proves the mechanism
             genuinely changed state between the two calls, then
             restored to white immediately so production's own state
             is left exactly as this probe found it.
           - The input box is a plain EDIT, driven from the PARENT's
             WM_CTLCOLOREDIT (gui/main.c's own wndproc case) - GetBk/
             TextColor DO exist, so those two are a real, direct
             read-back rather than an inference. Sentinel colours are
             set on the DC BEFORE calling it, for the same reason as
             above - but even that turned out NOT to be enough here
             (found by an actual mutation run, see the check below's
             own comment): DefWindowProcA's OWN default WM_CTLCOLOREDIT
             handling also happens to set white/black on this host's
             stock scheme, so a colour-only sentinel could not tell
             "this wndproc case ran" apart from "it did not, and
             DefWindowProcA's fallback coincidentally looked the
             same". The check that actually catches that is the
             returned BRUSH HANDLE compared against this file's own
             cached singleton, not just non-NULL. */
        {
            /* TWO calls, not one: the first sets a sentinel and
               returns whatever was there BEFORE (production's own
               setting, if any - not checked here, see the comment
               above for why that half is the WEAKER claim); the
               SECOND sets white and returns what the FIRST call just
               stored. Checking THAT return against the sentinel is
               what actually proves the mechanism changed live state
               between the two calls, rather than e.g. silently
               ignoring wParam/lParam and always reporting some fixed
               value regardless of what was asked. */
            COLORREF orig = SendMessage(g.part[LZ_GUI_TRANSCRIPT],
                                        EM_SETBKGNDCOLOR, 0,
                                        RGB(0x01, 0x02, 0x03));
            COLORREF prev = SendMessage(g.part[LZ_GUI_TRANSCRIPT],
                                        EM_SETBKGNDCOLOR, 0,
                                        RGB(0xFF, 0xFF, 0xFF));
            (void)orig;
            st_check(f, prev == RGB(0x01, 0x02, 0x03),
                     "panel: EM_SETBKGNDCOLOR on the transcript really "
                     "changes what SendMessage reports as \"previous\" "
                     "- the mechanism is live on this control");
            if (prev != RGB(0x01, 0x02, 0x03))
                fprintf(f, "  orig=%08lX prev=%08lX\n",
                       (unsigned long)orig, (unsigned long)prev);
            checks++;
            /* White is already restored above (the second call) - no
               separate restore step needed here. */
        }
        {
            HDC dc = GetDC(hwnd);
            if (dc) {
                HBRUSH got_brush;
                SetBkColor(dc, RGB(0x01, 0x02, 0x03));
                SetTextColor(dc, RGB(0x04, 0x05, 0x06));
                got_brush = (HBRUSH)SendMessage(hwnd, WM_CTLCOLOREDIT,
                                                (WPARAM)dc,
                                                (LPARAM)g.part[LZ_GUI_INPUT]);
                st_check(f, GetBkColor(dc) == RGB(0xFF, 0xFF, 0xFF),
                         "panel: WM_CTLCOLOREDIT sets the input box's "
                         "DC background to white"); checks++;
                st_check(f, GetTextColor(dc) == LZ_COLOR_TEXT,
                         "panel: WM_CTLCOLOREDIT sets the input box's "
                         "DC text colour explicitly"); checks++;
                /* == input_bkg_brush(), not just non-NULL (found by an
                   actual mutation run, not reasoned out in advance):
                   DefWindowProcA's
                   OWN default WM_CTLCOLOREDIT handling ALSO sets the
                   DC to COLOR_WINDOW/COLOR_WINDOWTEXT and returns a
                   valid stock brush, and on THIS host's stock desktop
                   scheme COLOR_WINDOW/COLOR_WINDOWTEXT already ARE
                   white/black - so a mutation that deleted this
                   case's entire body left every check above green,
                   "non-NULL" included, because the fallthrough to
                   DefWindowProcA coincidentally produced the same
                   colours and a non-NULL (just different) brush. Only
                   comparing the exact HANDLE against this file's own
                   cached singleton - which DefWindowProcA can never
                   return, whatever the desktop scheme - is what
                   distinguishes "this wndproc case ran" from "the
                   fallthrough happened to look the same". */
                st_check(f, got_brush == input_bkg_brush(),
                         "panel: WM_CTLCOLOREDIT returns THIS file's "
                         "own cached white brush, not a stock one from "
                         "DefWindowProcA's own fallback handling");
                checks++;
                ReleaseDC(hwnd, dc);
            }
        }

        lz_mru_remove(&g.mru, g.mdl.dir);
        lz_gui_model_unload(&g.mdl);
        memset(g.mdl.dir, 0, sizeof g.mdl.dir);
        g.load_failed = 0;
        g.side_model[0] = '\0';
        build_menu_bar(hwnd);
        set_idle_status(lz_str_utf8(LZ_STR_STATE_NO_MODEL));
        update_ctx_cell();
        side_panel_sync(hwnd);
    }

    /* 8b. truncate_gbk_to_width (team-lead's own screenshot:
          "<|im_end|>1.0000" glued together - LB_SETTABSTOPS
          taking effect (block 8's own LBS_USETABSTOPS check) is only
          half the fix, since a token wide enough still overruns the
          tab stop regardless). No tokenizer needed - this calls the
          truncation function directly on a plain GBK C string, which
          is exactly what it operates on; driving it through the real
          repaint_candidates would need a real loaded tokenizer to turn
          a token id into text, which is not available here. */
    {
        DWORD base = GetDialogBaseUnits();
        int tab_px = LZ_GUI_CAND_TAB * (int)LOWORD(base) / 4;
        char tok[64];
        HDC dc;
        SIZE sz;
        HFONT old_font;

        /* A short token must NOT be touched - proof the function is
           not simply always appending "...". */
        strcpy(tok, "ok");
        truncate_gbk_to_width(hwnd, tok, tab_px);
        st_check(f, strcmp(tok, "ok") == 0,
                 "panel: a short token is left alone by truncation");
        checks++;

        /* An unambiguously over-wide ASCII token - long enough to
           exceed tab_px on any plausible font/DPI, not borderline the
           way the real "<|im_end|>" (10 chars) might be on some hosts. */
        strcpy(tok, "averyveryverylongtokenthatoverflowsthetabstop");
        truncate_gbk_to_width(hwnd, tok, tab_px);
        st_check(f, strstr(tok, "...") != NULL,
                 "panel: an over-wide token is truncated with an "
                 "ellipsis"); checks++;
        dc = GetDC(hwnd);
        old_font = (HFONT)SelectObject(dc, lz_ui_font());
        GetTextExtentPoint32A(dc, tok, (int)strlen(tok), &sz);
        st_check(f, sz.cx <= tab_px,
                 "panel: the truncated token (with its ellipsis) does "
                 "not cross the tab stop");
        if (sz.cx > tab_px)
            fprintf(f, "  truncated width %ld, tab stop %d\n",
                   (long)sz.cx, tab_px);
        checks++;
        SelectObject(dc, old_font);
        ReleaseDC(hwnd, dc);

        /* An over-wide CHINESE token - team-lead's own named concern:
           a byte-count cutoff would bisect a double-byte character.
           20 real GBK characters (40 bytes), each the same byte pair
           repeated (0xC4 0xE3, GBK for the pronoun "you") -
           unambiguously wider than tab_px on any
           plausible font/DPI this host might report (8 characters was
           tried first and measured borderline-fitting on at least one
           real host, which is exactly the kind of thing this project's
           own "measure, do not guess" rule means to catch before it
           ships as a flaky gate).

           The check does NOT re-walk the string with the same lead-
           byte logic truncate_gbk_to_width itself uses - that would
           only prove the function agrees with itself, not that it
           avoided a split (see this project's own iron law four on
           exactly that trap). Instead it uses a fact independent of
           the algorithm: every one of the 8 source characters is
           EXACTLY 2 bytes, all identical, so a CORRECT truncation's
           prefix (everything before the appended "...") can only ever
           be an EVEN number of bytes - 0, 2, 4, ... whole characters.
           A naive byte-count cutoff has no reason to land on an even
           boundary and would fail this about half the time. */
        {
            int k;
            tok[0] = '\0';
            for (k = 0; k < 20; k++) strcat(tok, "\xC4\xE3");  /* GBK "you" */
            truncate_gbk_to_width(hwnd, tok, tab_px);
            st_check(f, strstr(tok, "...") != NULL,
                     "panel: an over-wide Chinese token is truncated "
                     "with an ellipsis"); checks++;
            {
                /* Located explicitly, not assumed to be the last 3
                   bytes - independent of whatever the check just above
                   found, so a mutation that breaks ONLY the ellipsis
                   (leaves the truncated length correct but never
                   appends "...") does not ALSO misreport a split here
                   just because this arithmetic silently assumed a
                   suffix that mutation removed. Falls back to the
                   whole string's own length when "..." is absent -
                   still a meaningful parity check on this specific
                   input (20 identical 2-byte characters), and the
                   ellipsis check above is what is responsible for
                   catching the ellipsis's own absence. */
                char *ell = strstr(tok, "...");
                int before_ellipsis = ell ? (int)(ell - tok)
                                          : (int)strlen(tok);
                st_check(f, before_ellipsis % 2 == 0,
                         "panel: truncating a Chinese token never "
                         "splits a double-byte character");
                if (before_ellipsis % 2 != 0)
                    fprintf(f, "  got %s (prefix %d bytes)\n", tok,
                           before_ellipsis);
                checks++;
            }
        }
    }

    /* 9. The inference inspector's pipeline (segment B, the design
          named in st_job_inspect's own comment): worker.c's send-side
          throttle, and the unconditional finishing send that implements
          "stop on the last token once generation ends". This test does
          not go through
          start_job - that
          helper always passes &g.sess.ins for JOB_GENERATE (see its own
          comment), which is exactly right for production but would give
          this test no control over what the job writes into it, and no
          way to plant a fresh LZInspect this block owns outright. */
    {
        StInspectJob ij;
        LZInspect ins;
        MSG peek;
        int waited, n_during;

        memset(&ins, 0, sizeof ins);
        ij.n = 10;
        ij.ins = &ins;

        transcript_clear();
        /* start_job always resets done_seen/done_rc/inspect_seen before
           a job starts (see there) - this path calls lz_worker_start
           directly instead, precisely so it can pass &ins rather than
           &g.sess.ins, so nothing else zeroes these first. Without
           this, st_pump below would read g.done_seen == 1 left over
           from block #7's job, return true on its very first condition
           check without pumping a single message, and both assertions
           after it would fail reading state nothing this block had
           touched yet. */
        g.done_seen = 0;
        g.inspect_seen = 0;
        st_check(f, lz_worker_start(hwnd, st_job_inspect, &ij, &ins) == 0,
                 "worker: an inspect-carrying job starts"); checks++;

        /* PM_NOREMOVE: wait for the burst's one throttled send to
           actually land, without dispatching it - dispatching here
           would already decide assertion 1 by a race instead of by the
           throttle. Same idiom as #5/#6 above. */
        waited = 0;
        while (!PeekMessage(&peek, hwnd, WM_APP_INSPECT, WM_APP_INSPECT,
                            PM_NOREMOVE) && waited < 5000) {
            Sleep(1);
            waited++;
        }
        st_check(f, PeekMessage(&peek, hwnd, WM_APP_INSPECT, WM_APP_INSPECT,
                                PM_NOREMOVE) != 0,
                 "worker: inspect setup - a WM_APP_INSPECT is queued, "
                 "undispatched"); checks++;

        /* gate 4, assertion 1: the throttle, not the job, decides how
           many WM_APP_INSPECT messages a ten-"token" burst produces
           before the job's own end - one, not ten. Drained by hand with
           PM_REMOVE and freed the same way lz_worker_free_tokens's other
           callers do, rather than through DispatchMessage - counting it
           here AND letting st_pump below count the SAME message again
           would make this assertion and the next one non-independent.
           st_job_inspect's own block-after-burst is what makes "before
           the job's own end" a real boundary rather than a hope: the
           job is paused, not finished, at this point. */
        n_during = 0;
        while (PeekMessage(&peek, hwnd, WM_APP_INSPECT, WM_APP_INSPECT,
                           PM_REMOVE)) {
            n_during++;
            lz_worker_free_tokens((void *)peek.lParam);
        }
        st_check(f, n_during == 1,
                 "worker: the throttle collapses a fast burst to one "
                 "WM_APP_INSPECT, not one per token"); checks++;

        /* Release the job - same mechanism st_job_blocking's own callers
           use - and let it run to completion. */
        lz_worker_request_stop();
        st_check(f, st_pump(hwnd, 0),
                 "worker: the inspect job reports done"); checks++;

        /* gate 4, assertion 2 - the one singled out by name.
           thread_main's unconditional finishing send fires regardless
           of how the job ended (normal / stop / error - see thread_main's
           own comment); here it ends by request_stop, the cheapest of
           the three to script. g.inspect_seen was reset to 0 above and
           the only message counted since then is this one - the burst's
           own message was drained by hand just above and never reached
           the window procedure, so 1 here can only be the finishing
           send that st_pump just dispatched.
           MUTATION: commenting out thread_main's own
           `if (w.ins) post_inspect(w.ins);` line, rebuilding, and
           rerunning --selftest turns this exact check red while
           assertion 1 above stays green - confirming the two are
           sensitive to their own mechanisms, not to each other's. */
        st_check(f, g.inspect_seen == 1,
                 "worker: the finishing send delivers the panel's last "
                 "frame, unconditionally, even though the throttle's own "
                 "timer had nothing left to say");
        if (g.inspect_seen != 1)
            fprintf(f, "  inspect_seen %d\n", g.inspect_seen);
        checks++;
        /* And it is genuinely the LAST token's frame, not the burst's
           already-drained one replayed - n_survived was still being
           written up to i == ij.n - 1 (see st_job_inspect), so the
           finishing send's own copy, taken after the burst ends, can
           only carry ij.n. */
        st_check(f, g.last_inspect.n_survived == ij.n,
                 "worker: the frame the finishing send delivered is the "
                 "LAST token's, not a stale one");
        if (g.last_inspect.n_survived != ij.n)
            fprintf(f, "  n_survived %d want %d\n",
                   g.last_inspect.n_survived, ij.n);
        checks++;
    }

    transcript_clear();
    return checks;
}

/* ---- rolling one turn back ----
 *
 * Everything here runs without a model. That is not a compromise: the
 * three commands are pure front-end bookkeeping - pop, cut, forget -
 * and the only piece that needs weights is the generation Regenerate
 * ends in, which start_job owns and st_worker already gates.
 *
 * The turns are played through THE PRODUCT'S OWN functions, in
 * do_send's order, rather than by writing text into the control and
 * messages into history by hand. A gate that assembles its own
 * fixture cannot see a defect in how the program assembles one - the
 * failure mode where thirteen green gates coexisted with every turn's
 * first line failing to render, because none of them fed a turn the
 * way do_send does.
 *
 * MEASURED SENSITIVITY - fourteen mutations, run one at a time, each
 * reverted before the next. Every check below is red under at least
 * one of them, and where a mutation reddens more than one the reason
 * is written out rather than left as "close enough":
 *
 *   1  truncate always to turn_cp[0]         -> 1 (regen cut)
 *   2  drop lz_gui_session_prefix_clear      -> 1 (prefix)
 *   3  cp from GetWindowTextLengthA          -> 3 (all three cuts)
 *   4  regen without its role check          -> 1 (regen refuses)
 *   5  edit without SetWindowTextA           -> 1 (input box)
 *   6  start_job without rollback_sync       -> 1 (greyed while busy)
 *   7  drop turn_cp_forget                   -> 1 (second rollback)
 *   8  IDM_REGEN not gated on a model        -> 1 (greyed, no model)
 *   9  regenerate pops the question too      -> 2
 *  10  edit/delete pop only the reply        -> 2 (edit AND delete -
 *      one line, two scenarios, which is what those two gates are for)
 *  11  truncate to 0                         -> 4
 *  12  edit without the EM_SETSEL            -> 1 (caret)
 *  13  apply_language's strip not re-synced  -> 1 (button after switch)
 *  14  create_children's strip not re-synced -> 1 (button at creation,
 *      and ONLY from the check placed up in selftest() itself - see
 *      the comment there for why a later one reads a repaired state)
 *
 * Number 3 is the one worth keeping: it is the implementation that was
 * asked for in writing, and it is wrong for any transcript containing
 * a Chinese character. All-ASCII fixtures would have shipped it.
 *
 * Number 9's second red is real coupling, not a leaky gate: with the
 * question popped, history ends with an assistant turn, and regen
 * refuses to render that - the target check is red on its own terms
 * (it prints "hist.n 2 want 3") and the second is a consequence.
 * Number 11's four are the same thing at a larger radius. */
static void st_turn(const char *u, const char *r) {
    char err[512];
    turn_begin_user(u, (int)strlen(u));
    lz_gui_session_begin(&g.sess, u, (int)strlen(u), err, (int)sizeof err);
    turn_begin_assistant();
    /* finish_job's JOB_GENERATE branch, in its order: the reply, its
       two trailing newlines, the flush, then the backfill. */
    transcript_push(r, (int)strlen(r));
    transcript_push("\r\n\r\n", 4);
    transcript_end();
    lz_gui_session_append_reply(&g.sess, r, (int)strlen(r));
    lz_gui_session_end(&g.sess, err, (int)sizeof err);
    rollback_sync();
}

static int st_rollback(FILE *f, HWND hwnd) {
    /* Each string is an ASCII tag the readback can be searched for,
       followed by U+4F60 U+597D. The Chinese is load-bearing: it makes
       a character position differ from a byte offset, which is the one
       thing that tells a cp recorded with EM_EXGETSEL apart from one
       recorded with GetWindowTextLengthA (g.turn_cp's own comment).
       All-ASCII fixtures pass either way. */
    static const char USER1[]  = "q1 \xE4\xBD\xA0\xE5\xA5\xBD";
    static const char REPLY1[] = "a1 \xE4\xBD\xA0\xE5\xA5\xBD";
    static const char USER2[]  = "q2 \xE4\xBD\xA0\xE5\xA5\xBD";
    static const char REPLY2[] = "a2 \xE4\xBD\xA0\xE5\xA5\xBD";
    static char prompt1[4096];
    StJob job;
    char got[2048], err[512];
    const char *p;
    int plen1, checks = 0;

    job.s = REPLY1;
    job.len = (int)sizeof REPLY1 - 1;
    job.chunk = 3;

    /* ---- 1. Regenerate: the reply goes, the question stays ---- */
    lz_gui_session_reset(&g.sess);
    transcript_clear();
    st_turn(USER1, REPLY1);
    st_turn(USER2, REPLY2);
    /* The prompt lz_gui_session_begin built for THIS turn, kept for the
       comparison further down. Read after the turn rather than during
       it because nothing overwrites it in between - the render is the
       last thing begin does, and the job that consumed it never
       rebuilds it. */
    p = lz_gui_session_prompt(&g.sess, &plen1);
    if (plen1 > (int)sizeof prompt1) plen1 = (int)sizeof prompt1;
    memcpy(prompt1, p, (size_t)plen1);

    st_check(f, rollback_ready(),
             "rollback: a completed exchange can be taken back");
    checks++;
    /* A prefix cache holding SOMETHING, so that "it was forgotten"
       below is a change of state rather than a reading of the zero it
       started at. These are the two fields lz_prefix_reset clears; set
       here rather than earned by a real prepare because earning one
       needs a model, and what is being gated is whether rollback_last
       makes the call at all. */
    g.sess.pc.n = 7;
    g.sess.pc.have = 1;

    rollback_last(hwnd, 1);

    st_check(f, g.sess.hist.n == 3 &&
                g.sess.hist.msgs[2].role == LZ_ROLE_USER,
             "rollback: regenerate drops the reply and keeps the "
             "question");
    if (g.sess.hist.n != 3) fprintf(f, "  hist.n %d want 3\n",
                                    g.sess.hist.n);
    checks++;
    st_check(f, g.sess.pc.n == 0 && g.sess.pc.have == 0,
             "rollback: the cached prefix is forgotten, because the "
             "history it described just moved"); checks++;

    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);
    st_check(f, strstr(got, "q2 ") != NULL && strstr(got, "a2 ") == NULL,
             "rollback: regenerate cuts at the assistant header, not at "
             "the user's");
    checks++;
    st_check(f, strstr(got, "q1 ") != NULL && strstr(got, "a1 ") != NULL,
             "rollback: the exchange BEFORE the last one is untouched");
    checks++;

    /* What regenerating actually re-renders. Not the seed (its
       randomness is checked by its own gate, and a fixed seed would
       defeat that check) but the prompt,
       which must come back byte for byte the same as the one begin
       built for this same question, or the second attempt is answering
       a different conversation from the first. */
    {
        int plen2 = 0;
        int rc = lz_gui_session_regen(&g.sess, err, (int)sizeof err);
        st_check(f, rc == 0, "rollback: regen renders after the pop");
        checks++;
        p = lz_gui_session_prompt(&g.sess, &plen2);
        st_check(f, plen2 == plen1 && memcmp(p, prompt1, (size_t)plen1) == 0,
                 "rollback: the re-render is byte for byte the prompt "
                 "the first attempt used");
        if (plen2 != plen1) fprintf(f, "  len %d want %d\n", plen2, plen1);
        checks++;
    }
    /* And the invariant that makes the pop mandatory: asked to render a
       history whose newest message is the assistant's, regen refuses
       rather than appending a generation prompt to the model's own
       reply and letting it answer itself. */
    st_turn(USER2, REPLY2);
    st_check(f, lz_gui_session_regen(&g.sess, err, (int)sizeof err) != 0,
             "rollback: regen refuses a history that does not end with "
             "the user"); checks++;

    /* ---- 2. Edit: both halves go, the text comes back ---- */
    lz_gui_session_reset(&g.sess);
    transcript_clear();
    st_turn(USER1, REPLY1);
    st_turn(USER2, REPLY2);
    SetWindowTextA(g.part[LZ_GUI_INPUT], "");
    SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_EDIT_LAST, 0), 0);

    st_check(f, g.sess.hist.n == 2,
             "rollback: edit takes back both halves of the exchange");
    if (g.sess.hist.n != 2) fprintf(f, "  hist.n %d want 2\n",
                                    g.sess.hist.n);
    checks++;
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_INPUT], got, (int)sizeof got);
    {
        char want[256];
        lz_gbk_from_utf8(USER2, (int)sizeof USER2 - 1, want,
                         (int)sizeof want, NULL);
        st_check(f, strcmp(got, want) == 0,
                 "rollback: edit puts the question back in the input box");
        if (strcmp(got, want) != 0) fprintf(f, "  got %s\n", got);
        checks++;
    }
    {
        /* Where the caret ended up, against where THIS control says its
           text ends - read out of it both times rather than computed,
           because "is that bytes or characters" is precisely what a GBK
           input box must not be guessed at (do_edit_last's own comment
           on the 0x7FFFFFFF clamp). Select-all is the reference: its end
           is the end, in whatever unit the control is counting in. */
        HWND h = g.part[LZ_GUI_INPUT];
        DWORD a = 0, b = 0, s0 = 0, s1 = 0;
        SendMessage(h, EM_GETSEL, (WPARAM)&a, (LPARAM)&b);
        SendMessage(h, EM_SETSEL, 0, (LPARAM)-1);
        SendMessage(h, EM_GETSEL, (WPARAM)&s0, (LPARAM)&s1);
        SendMessage(h, EM_SETSEL, (WPARAM)a, (LPARAM)a);
        st_check(f, s1 > 0 && a == b && a == s1,
                 "rollback: the caret sits at the end of the restored "
                 "text, with nothing selected");
        if (!(s1 > 0 && a == b && a == s1))
            fprintf(f, "  caret %lu..%lu end %lu\n", (unsigned long)a,
                    (unsigned long)b, (unsigned long)s1);
        checks++;
    }
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);
    st_check(f, strstr(got, "q2 ") == NULL && strstr(got, "a1 ") != NULL,
             "rollback: edit cuts back to the user header"); checks++;
    /* Only ONE turn's positions are ever recorded, so a second rollback
       with nothing in between has nowhere to cut to and is refused.
       Asserted rather than left implicit: this is the limitation the
       whole g.turn_cp design accepts, and a version that silently cut
       at a stale position instead would look identical until the day it
       ate the wrong turn. */
    st_check(f, !rollback_ready(),
             "rollback: a second rollback in a row is refused - only one "
             "turn's position is remembered"); checks++;

    /* ---- 3. Greyed while a job runs ---- */
    lz_gui_session_reset(&g.sess);
    transcript_clear();
    SetWindowTextA(g.part[LZ_GUI_INPUT], "");
    st_turn(USER1, REPLY1);
    /* The before-state, built here rather than inherited: a command
       greyed for the ordinary reason (nothing to roll back) passes a
       "greyed while busy" check without the busy rule existing at all. */
    st_check(f, cmd_is_enabled(IDM_EDIT_LAST) && cmd_is_enabled(IDM_DEL_LAST),
             "rollback: idle and with an exchange behind it, the "
             "commands are live"); checks++;
    /* st_job_blocking does not go through the session, so nothing will
       overwrite this - and without it finish_job's backfill would push
       the turn above into history a SECOND time. In the product a
       generate job is always preceded by begin or regen, both of which
       zero it; here there is no such call, so the selftest stands in
       for one. */
    g.sess.reply_len = 0;
    if (start_job(hwnd, st_job_blocking, &job, JOB_GENERATE) == 0) {
        st_check(f, !cmd_is_enabled(IDM_REGEN) &&
                    !cmd_is_enabled(IDM_EDIT_LAST) &&
                    !cmd_is_enabled(IDM_DEL_LAST),
                 "rollback: all three are greyed while a job is running");
        checks++;
        st_pump(hwnd, 1);
        st_check(f, cmd_is_enabled(IDM_EDIT_LAST) &&
                    cmd_is_enabled(IDM_DEL_LAST),
                 "rollback: and live again once it ends"); checks++;
    }
    /* IDM_REGEN's extra condition, which the pair above cannot show
       because they are never subject to it: no model is loaded in this
       whole run, so Regenerate stays greyed even now that the other two
       are live. That is what keeps its handler free of a model check
       and free of a second error box. */
    st_check(f, !cmd_is_enabled(IDM_REGEN),
             "rollback: regenerate stays greyed with no model to "
             "regenerate with"); checks++;

    /* ---- 3b. The BUTTON agrees with the menu item ----
     *
     * Regenerate is the one rollback command with a face on the tool
     * strip, and a strip is REBUILT rather than relabelled (see
     * apply_language) - a fresh one comes back with every button live.
     * So the disagreement this looks for is not hypothetical: it is
     * what happens by default unless three separate call sites
     * re-apply the state after building a strip. The language switch is
     * exercised because it is the only one of the three that can be
     * driven from here.
     *
     * Skipped when there is no strip at all (NT 3.51, and the selftest's
     * own no-toolbar branch), which lz_gui_toolbar_enabled reports as
     * -1 rather than as a state. */
    {
        HWND tb = g.part[LZ_GUI_TOOLBAR];
        int was_en = lz_str_lang_english();
        int before = lz_gui_toolbar_enabled(tb, IDM_REGEN);
        st_check(f, before == -1 || before == (cmd_is_enabled(IDM_REGEN) != 0),
                 "rollback: the Retry button and its menu item agree");
        if (before != -1)
            fprintf(f, "  button %d menu %d\n", before,
                    cmd_is_enabled(IDM_REGEN) != 0);
        checks++;
        apply_language(hwnd, !was_en);
        tb = g.part[LZ_GUI_TOOLBAR];   /* rebuilt: the old handle is gone */
        {
            int after = lz_gui_toolbar_enabled(tb, IDM_REGEN);
            st_check(f, after == -1 ||
                        after == (cmd_is_enabled(IDM_REGEN) != 0),
                     "rollback: and still agree after the strip is "
                     "rebuilt by a language switch");
            if (after != -1)
                fprintf(f, "  button %d menu %d\n", after,
                        cmd_is_enabled(IDM_REGEN) != 0);
            checks++;
        }
        apply_language(hwnd, was_en);
    }

    /* ---- 4. Delete, through the menu command ---- */
    lz_gui_session_reset(&g.sess);
    transcript_clear();
    st_turn(USER1, REPLY1);
    st_turn(USER2, REPLY2);
    SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_DEL_LAST, 0), 0);
    st_check(f, g.sess.hist.n == 2,
             "rollback: delete takes back exactly one exchange");
    if (g.sess.hist.n != 2) fprintf(f, "  hist.n %d want 2\n",
                                    g.sess.hist.n);
    checks++;
    got[0] = '\0';
    GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);
    st_check(f, strstr(got, "q2 ") == NULL && strstr(got, "a2 ") == NULL &&
                strstr(got, "a1 ") != NULL,
             "rollback: delete leaves the previous exchange whole");
    checks++;

    lz_gui_session_reset(&g.sess);
    transcript_clear();
    SetWindowTextA(g.part[LZ_GUI_INPUT], "");
    return checks;
}

/* ---- model loading: the pre-check and the failure path ----

   No real model is needed and none is used. What can be checked without
   one is everything that goes wrong, which is the half that normally
   ships untested: a directory that holds no model, and a model.bin
   that is not one. The success path needed a trained model and waited -
   st_real_model below is that wait ending. */
static int st_model(FILE *f) {
    char dir[MAX_PATH], bin[MAX_PATH];
    LZGuiModel m;
    char err[512];
    FILE *out;
    int rc, checks = 0;

    memset(&m, 0, sizeof m);
    GetTempPathA((DWORD)sizeof dir, dir);
    strcat(dir, "kunkun98-selftest");
    CreateDirectoryA(dir, NULL);

    /* An empty directory must be refused BEFORE anything is opened. */
    st_check(f, !lz_gui_model_dir_ok(dir),
             "model: a directory with no model.bin is refused"); checks++;

    sprintf(bin, "%s/model.bin", dir);
    out = fopen(bin, "wb");
    if (out) { fwrite("not a model", 1, 11, out); fclose(out); }
    /* ... and accepted once the file is there. Without this the check
       above passes for a pre-check that refuses everything, including
       real models. */
    st_check(f, lz_gui_model_dir_ok(dir),
             "model: the pre-check accepts a directory that has one");
    checks++;

    /* The file is eleven bytes of text. The engine must say so and leave
       nothing allocated - a tokenizer that fails after the weights
       loaded is the leak this teardown exists for. */
    strncpy(m.dir, dir, sizeof m.dir - 1);
    err[0] = '\0';
    rc = lz_gui_model_load_job(&m, NULL, NULL, NULL, err, (int)sizeof err);
    st_check(f, rc != 0, "model: loading eleven bytes of text fails");
    checks++;
    st_check(f, err[0] != '\0', "model: the failure fills errbuf");
    if (err[0] == '\0') fprintf(f, "  errbuf was empty\n");
    checks++;
    st_check(f, !lz_gui_model_ready(&m),
             "model: nothing is left loaded after a failure"); checks++;
    lz_gui_model_unload(&m);
    st_check(f, !lz_gui_model_ready(&m),
             "model: unload is safe to repeat"); checks++;

    /* A missing directory is a different failure from a bad file, and
       both have to be refusals rather than crashes. */
    memset(&m, 0, sizeof m);
    strcpy(m.dir, "Z:\\no\\such\\place");
    st_check(f, !lz_gui_model_dir_ok(m.dir),
             "model: a missing directory is refused"); checks++;
    err[0] = '\0';
    rc = lz_gui_model_load_job(&m, NULL, NULL, NULL, err, (int)sizeof err);
    st_check(f, rc != 0 && !lz_gui_model_ready(&m),
             "model: loading a missing directory fails cleanly"); checks++;
    lz_gui_model_unload(&m);

    DeleteFileA(bin);
    RemoveDirectoryA(dir);
    return checks;
}

/* ---- loading a REAL model, when one is available ----

   st_model above (and st_job_load_ok, used across this file) cover
   everything about loading that does NOT need a trained model: the
   pre-check, weight-load failure, cleanup. Success has always been
   SIMULATED - st_job_load_ok flips have_model/have_tok/have_state by
   hand rather than calling lz_open/lz_read_weights/lz_tokenizer_load/
   lz_state_alloc, so finish_job's own
   `ok = (rc == 0) && lz_gui_model_ready(&g.mdl)` line, and everything
   that only exists once a real tokenizer does (a real token count, a
   real KV footprint), has had zero coverage. This is that.

   LZ_TEST_MODEL: the same env var every gate in this repo that needs a
   real model reads - one name for every gate, not a second one to
   remember. Falls back to a known-good local path.

   EVERY CHECK BELOW STILL RUNS AND COUNTS WHEN THERE IS NO MODEL -
   `!have_model || <real assertion>`, never a skipped st_check call.
   `checks` has to be the same number on every machine whether or not
   LZ_TEST_MODEL happens to point at something real there -
   EXPECT_CHECKS is a fixed constant; a check that reads `-1` only
   because no model is ready in THIS environment must not be mistaken
   for a check that always reads `-1` on principle. */
static int st_real_model(FILE *f, HWND hwnd) {
    const char *dir;
    int checks = 0;
    int have_model;

    dir = getenv("LZ_TEST_MODEL");
    if (!dir || !dir[0]) dir = "E:\\LLM\\models\\recover-r10-q8";
    /* The exact pre-check open_model_dir itself calls, not a second
       fopen written here - this can never disagree with what the
       production path considers "a model is there". */
    have_model = lz_gui_model_dir_ok(dir);

    if (have_model) {
        /* A clean slate regardless of what earlier blocks in this file
           left behind - this check has to prove what IT did, not
           inherit a history or a cache from something upstream. */
        lz_gui_session_reset(&g.sess);
        memset(&g.mdl, 0, sizeof g.mdl);
        EnableWindow(g.part[LZ_GUI_SEND], FALSE);   /* known baseline */
        open_model_dir(hwnd, dir);   /* the REAL production path - not
                                        st_job_load_ok's flag flip */
        st_pump(hwnd, 0);
    }

    st_check(f, !have_model || lz_gui_model_ready(&g.mdl),
             "real model: lz_gui_model_ready is true after a real load");
    checks++;

    /* THE point of this whole function: finish_job's OWN
       `ok = (rc==0) && lz_gui_model_ready(&g.mdl)` line has never run
       with a real load behind it before now - "enable: finish_job's ok
       branch sets the READY idle status..." above uses st_job_load_ok,
       which sets ok's INPUTS by hand rather than earning them through a
       real lz_open/lz_read_weights/lz_tokenizer_load/lz_state_alloc
       sequence. Proven to cover something that check does not, by
       mutation, not by this comment's say-so.

       Checked through the idle status text, not Send's enabled state,
       for the same reason the scripted block switched: cmd_enable(
       ID_SEND, 1) in finish_job's JOB_LOAD branch does not depend on
       ok (the WM_COMMAND-bypass fix), so Send would read enabled here
       whether or not this line's ok computation is even reached. A
       real load that FAILS is not separately tested here - it exercises
       the exact same cmd_enable(ID_SEND, 1) call site the scripted
       failing-load block above already covers; nothing about that call
       depends on the load having been real. */
    st_check(f, !have_model ||
             strcmp(g.idle_status, lz_str_utf8(LZ_STR_STATE_READY))
             == 0,
             "real model: finish_job's ok branch sets the READY idle "
             "status after a REAL load, not a scripted one");
    checks++;

    st_check(f, !have_model || g.side_model[0] != '\0',
             "real model: the sidebar carries the model's name");
    checks++;

    st_check(f, !have_model || g.ctx_tokens >= 0,
             "real model: the context cell leaves -1 behind (a real "
             "model with an empty conversation reads 0, not absent)");
    checks++;

    {
        int found = 0, i;
        if (have_model)
            for (i = 0; i < g.mru.n; i++)
                if (strcmp(g.mru.item[i], g.mdl.dir) == 0) found = 1;
        st_check(f, !have_model || found,
                 "real model: the directory lands on the recent list");
        checks++;
    }

    {
        /* An untouched session (hist.n == 0) is a documented SHORTCUT
           inside lz_gui_session_token_count - it returns 0 without
           ever calling lz_encode, so asking right after load would
           pass whether or not a real BPE pass can run at all. Pushing
           one turn is what actually exercises lz_chat_render +
           lz_encode against the real tokenizer this model finally
           provides, not the shortcut. */
        char errbuf[256];
        int n = -1;
        if (have_model) {
            /* U+4F60 U+597D, "hello", UTF-8. */
            static const char TURN[] = "\xe4\xbd\xa0\xe5\xa5\xbd";
            lz_gui_session_begin(&g.sess, TURN, (int)strlen(TURN),
                                 errbuf, (int)sizeof errbuf);
            n = lz_gui_session_token_count(&g.sess, errbuf,
                                           (int)sizeof errbuf);
        }
        st_check(f, !have_model || n > 0,
                 "real model: a rendered turn counts to a real, "
                 "positive number of tokens - not the hist.n==0 "
                 "shortcut, a real BPE pass");
        checks++;
    }

    {
        /* do_send's own turn_header wiring, through the
           REAL entry point rather than by calling turn_header a
           second time by hand - the direct calls in st_transcript
           already prove turn_header itself is correct; this proves
           do_send actually CALLS it, with the right speaker/colour, at
           the right two places. Stopped at the first token
           (st_pump(hwnd, 1), the exact shape st_worker's own "2. Stop"
           block uses) rather than run to completion: what is being
           checked here - the two headers - is pushed to the
           transcript SYNCHRONOUSLY, before start_job is even called,
           so waiting out a full real generation would only add
           wall-clock time for no more coverage.

           lz_gui_session_reset first: the token-count block just
           above began a turn (lz_gui_session_begin) and never ended
           it, and do_send calls lz_gui_session_begin again inside
           itself - a clean slate is what makes that call succeed
           rather than error on an already-open turn, and an error
           here would pop a MessageBoxA, which hangs an unattended
           selftest run stone dead.

           `!have_model || <cond>`, checks++ unconditional throughout
           (fixed from a version that wrapped this whole block in
           `if (have_model)`, checks++ included -
           exactly the drift-per-machine bug this function's own
           surrounding comment warns about: EXPECT_CHECKS is a FIXED
           constant, and a conditionally-skipped st_check call would
           make the real total depend on whether THIS machine happens
           to have the model at LZ_TEST_MODEL/the fallback path). */
        static const char MSG[] = "hi";
        /* lz_str_DISPLAY - GetWindowTextA below reads back GBK, the
           same reason st_transcript's own turn_header checks use
           lz_str_display rather than lz_str_utf8. */
        const char *ulabel = lz_str_display(LZ_STR_SPEAKER_USER);
        const char *alabel = lz_str_display(LZ_STR_SPEAKER_ASSISTANT);
        char got[2048];
        char *up = NULL, *ap = NULL;
        int up_ok = 1, up_color_ok = 1, ap_ok = 1;

        if (have_model) {
            transcript_clear();
            lz_gui_session_reset(&g.sess);
            SetWindowTextA(g.part[LZ_GUI_INPUT], MSG);
            do_send(hwnd);
            st_pump(hwnd, 1);

            got[0] = '\0';
            GetWindowTextA(g.part[LZ_GUI_TRANSCRIPT], got, (int)sizeof got);
            up = strstr(got, ulabel);
            up_ok = (up == got);
            if (up_ok) {
                /* The LABEL's own character span only, converted from
                   BYTES to RichEdit's DECODED-character positions via
                   MultiByteToWideChar - not strlen(ulabel), and not
                   the whole line up to its CRLF (measured): the default
                   UI language is Chinese, so a
                   DBCS label's byte count overshoots its real
                   character span, and separately RichEdit collapses
                   "\r\n" into ONE character position rather than two,
                   so even a correct byte-to-char conversion of "up to
                   and including the CRLF" overshoots by one.
                   Restricting to just the label sidesteps both, and
                   the ASSISTANT header - a DIFFERENT colour later in
                   this SAME buffer - is exactly why this cannot
                   become "the whole line" (an unbounded range)
                   either. */
                int lchars = MultiByteToWideChar(CP_ACP, 0, ulabel,
                                                 (int)strlen(ulabel), NULL,
                                                 0);
                up_color_ok = lchars > 0 &&
                             st_color(g.part[LZ_GUI_TRANSCRIPT], 0, lchars)
                             == LZ_COLOR_USER;
            }
            ap = strstr(got, alabel);
            ap_ok = (ap != NULL && up != NULL && ap > up);

            /* The Stop path, exercised here for real (st_pump(hwnd, 1)
               stops at the first token) must ALSO say it was stopped -
               the engine records the reason in opts.out_finish, and
               finish_job's JOB_GENERATE branch turns CANCELLED into the
               LZ_STR_SYS_GEN_STOPPED line. Read back from the transcript
               in GBK (lz_str_display), like the labels above. */
            {
                const char *stoplabel = lz_str_display(LZ_STR_SYS_GEN_STOPPED);
                int stop_ok = strstr(got, stoplabel) != NULL;
                st_check(f, !have_model || stop_ok,
                         "real model: stopping a generation says it was "
                         "stopped, not silently ends like a normal EOS");
                checks++;
            }

            transcript_clear();
            lz_gui_session_reset(&g.sess);
        }

        st_check(f, !have_model || up_ok,
                 "real model: do_send's user turn header starts the "
                 "transcript with the real speaker label");
        checks++;
        st_check(f, !have_model || up_color_ok,
                 "real model: do_send's user turn header is coloured "
                 "through the real entry point, not just in "
                 "turn_header's own direct-call test");
        checks++;
        st_check(f, !have_model || ap_ok,
                 "real model: do_send's assistant turn header also "
                 "reaches the transcript, after the user's");
        checks++;
    }

    if (have_model) {
        /* Full teardown, same shape st_job_load_ok's own block above
           uses: every side effect a real load leaves (the MRU entry,
           the sidebar line, the ctx cell, the pushed turn, the prefix
           cache finish_job's ok branch also builds) has to come back
           out, or later checks in this file - several assume "no
           model" - would read this load's leftovers instead of their
           own setup. Send is left enabled, not reset to FALSE - idle
           state IS Send-enabled now. */
        transcript_clear();
        lz_gui_session_reset(&g.sess);
        lz_gui_session_prefix_drop(&g.sess);
        lz_mru_remove(&g.mru, g.mdl.dir);
        lz_gui_model_unload(&g.mdl);
        memset(g.mdl.dir, 0, sizeof g.mdl.dir);
        g.load_failed = 0;
        g.side_model[0] = '\0';
        build_menu_bar(hwnd);
        EnableWindow(g.part[LZ_GUI_SEND], TRUE);
        set_idle_status(lz_str_utf8(LZ_STR_STATE_NO_MODEL));
        update_ctx_cell();
        /* Same reasoning as st_job_load_ok's own teardown block above
           - the candidate panel is another side effect
           that does not tear itself down here. */
        side_panel_sync(hwnd);
    }

    return checks;
}

/* ---- the conversation layer, without a generator ----

   Everything between "the user pressed Enter" and "the assistant turn is
   in history" is front-end logic. Waiting for a trained model to check
   it would mean checking it never, and the one thing that CANNOT be
   checked later is the backfill encoding: by the time a real model
   exists, a reply that lost a character on the way through the control
   looks like a model that produced a question mark. */
/* ---- prefill: the prefix-reuse knob, without a real model --

   The parity gate needs a real model and is
   skipped until one is available. THIS gate cannot go further than it
   does either: "does lz_gui_session_job's own branch (gui/session.c)
   really call lz_prefix_prepare" cannot be exercised end to end through
   lz_gui_session_job itself, even with a flag-flipped
   (have_model/have_tok/have_state = 1, everything else zeroed) fake
   model the way st_job_load_ok already does for finish_job - tried
   once, and it segfaulted before lz_gui_session_job even returned
   (lz_gen_opts_set_eos, called unconditionally before the retry loop
   starts, dereferences the tokenizer for real - same category of risk
   as lz_gui_session_token_count's crash).
   That attempt is not committed.

   What CAN be checked safely is the mechanism the gate names:
   lz_prefix_prepare, the SAME function gui/session.c's branch calls,
   driven directly with a split <= 0 - its own documented "nothing to
   reuse... take the full path" shortcut. Verified line by line before
   relying on it: along that path the function touches only pc (already
   a real, lz_prefix_init'd cache) and the caller's own out-parameters:
   m/t/s are checked for non-NULL and never dereferenced. Passing
   pointers to zeroed-but-valid LZModel/LZTokenizer/LZRunState locals is
   therefore safe - their CONTENTS are never read on this path.

   gui/session.c's own three-line branch
   (`if (s->prefill == LZ_PREFILL_PREFIX && s->pc_ready)`) is reviewed
   by inspection, not exercised by a runtime mutation here - it is short
   and unambiguous, and there is no way to run it further without a real
   model. */
static int st_prefill(FILE *f) {
    LZModel fake_model;
    LZTokenizer fake_tok;
    LZRunState fake_state;
    LZPrefixCache pc;
    char err[256];
    int rc, checks = 0;
    long calls = -1, hits = 0, tsaved = 0, mismatch = 0, unsplit = 0;

    memset(&fake_model, 0, sizeof fake_model);
    memset(&fake_tok, 0, sizeof fake_tok);
    memset(&fake_state, 0, sizeof fake_state);
    /* lz_ckpt_alloc (inside lz_prefix_init) sizes its four buffers off
       m->config alone - never touches weight data - so these six ints
       are enough to make it allocate for real without a real model.
       NOT left at 0: malloc(0) is well-defined C for crash safety, but
       "safe" is not "returns non-NULL", and malloc(0) returning NULL is
       ALSO well-defined C. Watcom's runtime does exactly that, gcc's
       does not - lz_ckpt_alloc reads a NULL buffer as an allocation
       failure regardless of which C rule produced it, so a 0-size
       "lz_prefix_init succeeds" check would pass on gcc and fail on
       Watcom. Small positive sizes sidestep the ambiguity entirely
       rather than asserting around it. */
    /* n_q8 = n_linear_layers * lin_n_v_heads * lin_k_head_dim *
       lin_v_head_dim, and n_s = n_q8 / 32 (INTEGER division -
       ckpt_sizes, src/forward.c) - the first fix used all-1 dims,
       giving n_q8=1 and n_s = 1/32 = 0, so ssm_s's own malloc(0) still
       hit the exact same Watcom-vs-gcc gap this whole block exists to
       avoid. lin_n_v_heads=32 (n_linear_layers/k/v_head_dim=1) makes
       n_q8=32, n_s=1: every one of the four buffers gets a genuinely
       positive size on both toolchains. */
    fake_model.config.n_linear_layers = 1;
    fake_model.config.lin_n_v_heads = 32;
    fake_model.config.lin_k_head_dim = 1;
    fake_model.config.lin_v_head_dim = 1;
    fake_model.config.lin_conv_dim = 1;
    fake_model.config.conv_kernel = 2;

    rc = lz_prefix_init(&pc, &fake_model, err, (int)sizeof err);
    st_check(f, rc == 0,
             "prefill: lz_prefix_init succeeds on a model whose own "
             "sizing reduces to nothing"); checks++;

    lz_prefix_stats(&pc, &calls, &hits, &tsaved, &mismatch, &unsplit);
    st_check(f, calls == 0,
             "prefill: the prefix cache is untouched in full mode");
    checks++;

    {
        /* "hi" (2 bytes) against split=0: tail_len = 2 - 0 = 2 > 0, but
           split <= 0 is what matters here - the generation-prompt tail
           for either think state is always longer than 2 bytes in the
           real template, so this stands in for "the render is shorter
           than what would need to be reusable", the shortcut's own
           documented case, without needing the real template rendered. */
        int start_pos = 0, suffix_off = 0, reused = 0;
        rc = lz_prefix_prepare(&pc, &fake_model, &fake_tok, &fake_state,
                               "hi", 2, 0, &start_pos, &suffix_off,
                               &reused, NULL, NULL, NULL, err,
                               (int)sizeof err);
        st_check(f, rc == 0,
                 "prefill: prepare's own no-reuse shortcut still "
                 "reports success"); checks++;
    }

    lz_prefix_stats(&pc, &calls, &hits, &tsaved, &mismatch, &unsplit);
    st_check(f, calls > 0,
             "prefill: the prefix path is actually entered in prefix "
             "mode"); checks++;

    lz_prefix_free(&pc);

    /* ---- what prefix_drop is actually FOR ----
     *
     * The clear half is gated by counting the
     * matcher's rejections, and MEASURED that the drop half is not
     * gated there at all: making lz_gui_session_prefix_drop a no-op
     * leaves that whole gate green, because the arm that follows it
     * re-initialises the cache anyway. Drop is therefore redundant for
     * CORRECTNESS in the drop-then-arm sequence, and its real job is
     * the other one: a load that FAILS never reaches arm, and pc_ready
     * has to be 0 for the whole window in which no model is loaded -
     * otherwise lz_gui_session_job's prefix branch fires against
     * weights that are gone.
     *
     * That is a state assertion and needs no model, so it belongs
     * here. Mutation: make prefix_drop a no-op - only these redden. */
    {
        LZGuiModel fake_mdl;
        LZGuiSession ps;
        memset(&fake_mdl, 0, sizeof fake_mdl);
        fake_mdl.model = fake_model;
        lz_gui_session_init(&ps, &fake_mdl, 0);

        rc = lz_gui_session_prefix_arm(&ps, &fake_mdl, err, (int)sizeof err);
        st_check(f, rc == 0 && ps.pc_ready,
                 "prefill: arm reports ready after a successful init");
        checks++;
        lz_gui_session_prefix_drop(&ps);
        /* Wording note: no check description may contain the substring
           "FAIL" - the whole report is asserted to lack it, so a PASS
           line that merely mentions the word turns the run red with
           `fails 0` printed right below it. Say "a load that did not
           finish" instead. */
        st_check(f, ps.pc_ready == 0,
                 "prefill: drop leaves the cache not-ready, which is what "
                 "a load that did not finish depends on");
        checks++;
        /* Two checks here, not three. A third - "a refused arm leaves
           it not-ready too", driven by passing a NULL model - only
           reddened because the no-op drop above had left pc_ready at
           1, so it read a state the previous assertion built rather
           than one of its own; and the behaviour it asserted is not
           obviously right - refusing a call because the CALLER passed
           NULL is no reason to throw away a cache that is still valid.
           The real failed-load path is drop-then-no-arm, which is what
           the check above covers. */
        lz_gui_session_free(&ps);
    }
    return checks;
}

static int st_session(FILE *f) {
    /* U+4F60 U+597D, then an emoji GBK cannot represent. */
    static const char USER[] = "\xE4\xBD\xA0\xE5\xA5\xBD";
    static const char REPLY[] =
        "<think>\xE6\x83\xB3</think>\xE5\x9B\x9E\xE7\xAD\x94"
        "\xF0\x9F\x90\x94";
    static const char EMOJI[] = "\xF0\x9F\x90\x94";
    LZGuiSession s;
    LZGuiModel empty;
    char err[1024];
    const char *p;
    int plen, rc, checks = 0;

    memset(&empty, 0, sizeof empty);
    lz_gui_session_init(&s, &empty, 1);

    err[0] = '\0';
    rc = lz_gui_session_begin(&s, USER, (int)sizeof USER - 1, err,
                              (int)sizeof err);
    st_check(f, rc == 0, "session: the first turn renders"); checks++;
    p = lz_gui_session_prompt(&s, &plen);
    st_check(f, plen > 0 && memchr(p, USER[0], (size_t)plen) != NULL &&
             strstr(p, USER) != NULL,
             "session: the prompt contains the user's UTF-8"); checks++;
    st_check(f, lz_gui_session_turns(&s) == 1,
             "session: one turn after begin"); checks++;

    /* No model: the job must refuse rather than dereference one. */
    err[0] = '\0';
    rc = lz_gui_session_job(&s, NULL, NULL, NULL, err, (int)sizeof err);
    st_check(f, rc != 0 && err[0] != '\0',
             "session: generating with no model fails with a message");
    checks++;

    /* Now the part that matters. A scripted reply carrying a character
       GBK cannot represent goes through the SAME path a generated one
       does. The transcript will show '?' for it - that is correct and
       expected - and history must not. */
    lz_gui_session_append_reply(&s, REPLY, (int)sizeof REPLY - 1);
    err[0] = '\0';
    rc = lz_gui_session_end(&s, err, (int)sizeof err);
    st_check(f, rc == 0, "session: backfill succeeds"); checks++;
    st_check(f, lz_gui_session_turns(&s) == 2,
             "session: the assistant turn landed in history"); checks++;
    if (lz_gui_session_turns(&s) == 2) {
        const char *a = s.hist.msgs[1].content;
        int alen = s.hist.msgs[1].len;
        if (alen < 0) alen = (int)strlen(a);
        st_check(f, memmem_present(a, alen, EMOJI, 4),
                 "session: history keeps the byte the display cannot show");
        checks++;
        /* lz_chat_norm_history drops reasoning entirely for history
           turns - the tag must not survive into the next render. */
        st_check(f, !memmem_present(a, alen, "<think>", 7),
                 "session: the think block is stripped from history");
        checks++;
    }

    /* Second turn: the render must carry BOTH turns, which is what
       full-history mode means and what a resume path would have to
       reproduce byte for byte. */
    err[0] = '\0';
    rc = lz_gui_session_begin(&s, "second", 6, err, (int)sizeof err);
    st_check(f, rc == 0 && lz_gui_session_turns(&s) == 3,
             "session: the second turn appends rather than replaces");
    checks++;
    p = lz_gui_session_prompt(&s, &plen);
    st_check(f, strstr(p, USER) != NULL && strstr(p, "second") != NULL,
             "session: the second render still contains the first turn");
    checks++;

    lz_gui_session_reset(&s);
    st_check(f, lz_gui_session_turns(&s) == 0,
             "session: reset clears the conversation"); checks++;

    /* ---- the context-full trim ----
       No model needed: what a full context DOES is drop the oldest
       exchange and render again, and both halves are ours. */
    {
        int i;
        for (i = 0; i < 3; i++) {
            char msg[32];
            sprintf(msg, "u%d", i);
            lz_gui_session_begin(&s, msg, (int)strlen(msg), err,
                                 (int)sizeof err);
            lz_gui_session_append_reply(&s, "a", 1);
            lz_gui_session_end(&s, err, (int)sizeof err);
        }
        st_check(f, lz_gui_session_turns(&s) == 6,
                 "trim: three exchanges make six turns"); checks++;
        st_check(f, lz_gui_session_trim(&s) == 1, "trim: drops something");
        checks++;
        st_check(f, lz_gui_session_turns(&s) == 4,
                 "trim: drops a PAIR, not a lone user turn - a reply with "
                 "no question in front of it reads as the conversation "
                 "having started that way");
        checks++;
        st_check(f, memmem_present(s.hist.msgs[0].content,
                                   (int)strlen(s.hist.msgs[0].content),
                                   "u1", 2),
                 "trim: the OLDEST exchange is the one that went");
        checks++;
        /* Down to one turn it must refuse: a single message longer than
           the whole context is not a trimming problem, and a loop that
           kept trying would never end. */
        while (lz_gui_session_trim(&s)) { /* keep going */ }
        st_check(f, lz_gui_session_turns(&s) >= 1 &&
                 lz_gui_session_turns(&s) <= 2,
                 "trim: stops with the newest turn still there"); checks++;
        st_check(f, lz_gui_session_trim(&s) == 0,
                 "trim: refuses rather than looping forever"); checks++;
    }

    lz_gui_session_free(&s);
    return checks;
}

/* ---- chat file: what a user opens afterwards, and what refuses to load --

   save_chat_to() and load_chat_from() are both split out from their menu
   handlers for the same reason: a path that only exists behind a modal
   dialog (GetSaveFileName / GetOpenFileName) is a path that only gets
   tested by hand otherwise. What is being checked is not "did fopen
   work" but the things that make the round trip honest - the header
   line, the CRLF conversion (still needed even though the source is
   history now rather than RichEdit text: GBK output still has to end in
   CRLF for Notepad), and the two contracts gui/chatfile.h documents -
   a good file loads back with the same turns, and a bad one changes
   nothing. */
static int st_save(FILE *f) {
    char dir[MAX_PATH];
    char path[MAX_PATH + 64];      /* GetTempPath can fill dir entirely */
    static char got[8192];
    char errb[256];
    const char *header;
    FILE *r, *w;
    size_t n;
    int i, cr_alone = 0, checks = 0, n_before;

    lz_gui_session_reset(&g.sess);
    lz_chat_hist_push(&g.sess.hist, LZ_ROLE_USER, "one\ntwo\nthree", -1,
                      errb, (int)sizeof errb);
    lz_chat_hist_push(&g.sess.hist, LZ_ROLE_ASSISTANT, "reply", -1,
                      errb, (int)sizeof errb);
    /* A custom system prompt, so the save/load round-trip has to carry
       it - the reason the save prepends a synthetic [system] line and
       the load feeds it back into the settings. Empty here would let
       a save that silently drops the identity pass. */
    strcpy(g.set.system,
           /* UTF-8 for "you are a test assistant." */
           "\xe4\xbd\xa0\xe6\x98\xaf\xe6\xb5\x8b\xe8\xaf\x95"
           "\xe5\x8a\xa9\xe6\x89\x8b\xe3\x80\x82");
    lz_gui_session_set_system(&g.sess, g.set.system);

    GetTempPathA((DWORD)sizeof dir, dir);
    sprintf(path, "%skunkun98-save.txt", dir);
    DeleteFileA(path);
    st_check(f, save_chat_to(path) == 0, "save: writes a file");
    checks++;

    got[0] = '\0';
    r = fopen(path, "rb");
    n = r ? fread(got, 1, sizeof got - 1, r) : 0;
    if (r) fclose(r);
    got[n] = '\0';

    header = lz_str_display(LZ_STR_SAVE_HEADER);
    st_check(f, n > 0 && strncmp(got, header, strlen(header)) == 0,
             "save: the file starts with the header line");
    checks++;

    /* Every CR must be followed by an LF. A lone CR is what RichEdit
       hands out and what Notepad cannot read. */
    for (i = 0; i < (int)n; i++)
        if (got[i] == '\r' && (i + 1 >= (int)n || got[i + 1] != '\n'))
            cr_alone++;
    st_check(f, cr_alone == 0, "save: no bare CR survives into the file");
    if (cr_alone) fprintf(f, "  %d bare CRs\n", cr_alone);
    checks++;

    /* And the reverse: a lone LF would mean the conversion ran the wrong
       way and Notepad still cannot read it. */
    {
        int lf_alone = 0;
        for (i = 0; i < (int)n; i++)
            if (got[i] == '\n' && (i == 0 || got[i - 1] != '\r'))
                lf_alone++;
        st_check(f, lf_alone == 0, "save: no bare LF either");
        checks++;
    }

    st_check(f, strstr(got, "three") != NULL,
             "save: the message text is actually in there");
    checks++;

    /* Round trip: load the file just written back and confirm the turns
       match. This is the whole reason the format has a magic line and a
       decoder, not just an assertion that fopen worked. */
    n_before = g.sess.hist.n;
    /* CLEAR THE SETTING BEFORE THE LOAD, so "the system prompt rode
       the round-trip" has a real baseline: whatever g.set.system holds
       after load_chat_from must have come from the FILE, not from the
       value the test set before saving. Without this, a save that
       silently dropped the system prompt would leave the pre-set value
       in place and the check would pass on the setting the test wrote,
       not on the file. (This is the same shape as the ctx-ini gate's
       own baseline trick - see st_ctx_commit.) */
    g.set.system[0] = '\0';
    /* The load_chat_from call site, planted right on this real
       successful load rather than a synthetic one - g.status_is_sbar's
       true here just like everywhere else in this run. */
    if (g.status_is_sbar)
        sb_text(1, "kk98-ctx-sentinel");
    st_check(f, load_chat_from(path, errb, (int)sizeof errb) == 0 &&
                g.sess.hist.n == n_before &&
                g.sess.hist.msgs[0].role == LZ_ROLE_USER &&
                strcmp(g.sess.hist.msgs[0].content, "one\ntwo\nthree") == 0 &&
                g.sess.hist.msgs[1].role == LZ_ROLE_ASSISTANT &&
                strcmp(g.sess.hist.msgs[1].content, "reply") == 0,
             "chat: a saved conversation loads back with the same turns");
    checks++;
    /* The SYSTEM PROMPT rides the round-trip: written as a synthetic
       [system] first line and fed back into the SETTINGS on load (the
       same user decision as load_chat_from: opening a conversation
       writes it back into the settings). The turns above stay
       user-first because the system line never enters hist - it
       becomes the setting instead. */
    st_check(f, strcmp(g.set.system,
                       "\xe4\xbd\xa0\xe6\x98\xaf\xe6\xb5\x8b\xe8\xaf\x95"
                       "\xe5\x8a\xa9\xe6\x89\x8b\xe3\x80\x82") == 0,
             "chat: the custom system prompt rides the round-trip into "
             "the settings");
    if (strcmp(g.set.system,
               "\xe4\xbd\xa0\xe6\x98\xaf\xe6\xb5\x8b\xe8\xaf\x95"
               "\xe5\x8a\xa9\xe6\x89\x8b\xe3\x80\x82") != 0)
        fprintf(f, "  system [%s]\n", g.set.system);
    checks++;
    /* Back to empty so the custom prompt set above does not leak into
       every subsequent test's renders. */
    g.set.system[0] = '\0';
    lz_gui_session_set_system(&g.sess, "");
    if (g.status_is_sbar) {
        char cell3[64];
        cell3[0] = '\0';
        SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETTEXT, 1, (LPARAM)cell3);
        st_check(f, strcmp(cell3, "kk98-ctx-sentinel") != 0,
                 "status: load_chat_from really calls update_ctx_cell");
        checks++;
    }

    /* A file with no magic line must be refused, and must not touch
       whatever history already existed - a half-loaded conversation
       renders a prompt that matches nothing on screen. */
    w = fopen(path, "wb");
    if (w) { fwrite("not a chat file\r\n", 1, 17, w); fclose(w); }
    n_before = g.sess.hist.n;
    st_check(f, load_chat_from(path, errb, (int)sizeof errb) != 0 &&
                g.sess.hist.n == n_before,
             "chat: a file without the header line is refused and "
             "changes nothing");
    checks++;

    /* save_chat_to discarded lz_chatfile_encode's return value (and its
       own two conversions' after it). Direct assertion, not round trip:
       encoding a truncated history and then decoding it back can stay
       green even when both sides silently agree on the same cut point -
       see the chatfile CRLF history two functions up. What has to be
       checked is that save_chat_to REFUSES an encode that would not
       fit, not that whatever it did write reads back consistently. */
    {
        static char big[1024 * 1024 + 256];
        int save_rc;
        memset(big, 'a', sizeof big - 1);
        big[sizeof big - 1] = '\0';

        lz_gui_session_reset(&g.sess);
        lz_chat_hist_push(&g.sess.hist, LZ_ROLE_USER, big,
                          (int)sizeof big - 1, errb, (int)sizeof errb);

        DeleteFileA(path);
        save_rc = save_chat_to(path);
        st_check(f, save_rc != 0,
                 "save: an encode too big for the buffer is refused, "
                 "not silently truncated");
        checks++;
        r = fopen(path, "rb");
        st_check(f, r == NULL,
                 "save: a refused save leaves no half-written file "
                 "behind");
        if (r) fclose(r);
        checks++;
        DeleteFileA(path);
        lz_gui_session_reset(&g.sess);
    }

    /* load_chat_from's read-side twin: a file bigger than raw[]'s cap
       must not be read up to the cap and silently cut there. Built
       directly with fopen/fwrite, not through save_chat_to - the check
       just above means save_chat_to now refuses to WRITE anything this
       large, so this is the only way left to put an oversized file in
       front of the reader (which still has to cope: chatfile.h says
       this is a .txt people hand-edit, not only ever this program's own
       output). */
    {
        /* NOT "huge": that identifier is Watcom's legacy __huge memory-
           model keyword (16-bit segmented pointers), still recognised
           by wcc386 even in this 32-bit flat build, and it breaks the
           declaration silently into cascading syntax errors rather than
           a clean "reserved word" diagnostic - gcc has no such keyword,
           so this only shows up on the Watcom build. */
        static char oversized[1024 * 1024 + 4096];
        FILE *w2;
        size_t hn = 0;
        int load_rc;

        memcpy(oversized + hn, header, strlen(header));
        hn += strlen(header);
        oversized[hn++] = '\n';
        memcpy(oversized + hn, "[user]\n", 7);
        hn += 7;
        memset(oversized + hn, 'b', sizeof oversized - hn - 1);
        hn = sizeof oversized - 1;
        oversized[hn] = '\0';

        DeleteFileA(path);
        w2 = fopen(path, "wb");
        if (w2) { fwrite(oversized, 1, hn, w2); fclose(w2); }

        n_before = g.sess.hist.n;
        load_rc = load_chat_from(path, errb, (int)sizeof errb);
        st_check(f, load_rc != 0 && g.sess.hist.n == n_before,
                 "chat: a file bigger than the read buffer is refused, "
                 "not silently truncated");
        checks++;
        DeleteFileA(path);
    }

    lz_gui_session_reset(&g.sess);
    transcript_clear();
    return checks;
}

/* ---- the settings window, without clicking anything ----

   The dialog is built and read back directly; only the modal loop is
   left uncovered, and it is four lines. What this catches is the pair
   that has to agree: the box shows a number, and the box's own reader
   has to accept it. A formatter and a parser that disagree produce a
   dialog which refuses the value it just offered - and the only way to
   find that out is to open it. */
static int st_settings_dialog(FILE *f, HINSTANCE inst) {
    LZGuiSettings in, out;
    HWND dlg;
    char buf[64];
    int checks = 0;

    lz_common_settings_init(&in);
    lz_common_settings_set_temp(&in, 0.37f);
    /* Non-default, same rule as the rest: a row whose box is never
       written and never read would otherwise round-trip its default and
       pass. 0.42 is not 0.30 (the think-temp default in both presets). */
    lz_common_settings_set_think_temp(&in, 0.42f);
    /* Load-bearing, all three: the context row has to round-trip a
       value that is NOT the default (or a reader that ignores the
       control entirely would pass), and the seed pair has to hold
       values nothing in this dialog can produce, so that "the reader
       invented them" and "the reader carried them through" are
       distinguishable. 0xC0FFEE is not reachable from any control. */
    in.ctx = 1536;
    /* Every value fed in is one the row's own DEFAULT is not, so a row
       whose box is never written and never read still round-trips its
       default and would pass. 0.35 is not 0.95 (think top-p), 1.25 is
       not 1.10, 256 is not -1, 1536 is not 2048. */
    lz_common_settings_set_topp(&in, 0.35f);
    lz_common_settings_set_rep(&in, 1.25f);
    lz_common_settings_set_max_new(&in, 256, in.ctx);
    /* A CUSTOM system prompt, in UTF-8 with Chinese (hex-escaped per
       iron law seven). The value is NOT the default (empty), so a
       reader that ignores the box entirely would pass. The
       distinguishing pair: non-empty carries this, empty means the
       built-in. */
    strcpy(in.system,
           /* UTF-8 for "you are a test assistant." */
           "\xe4\xbd\xa0\xe6\x98\xaf\xe6\xb5\x8b\xe8\xaf\x95"
           "\xe5\x8a\xa9\xe6\x89\x8b\xe3\x80\x82");
    in.seed_mode = LZ_COMMON_SEED_FIXED;
    in.seed = 0xC0FFEEull;

    dlg = lz_gui_settings_dialog_create(NULL, inst, &in);
    st_check(f, dlg != NULL, "settings: the window is created"); checks++;
    if (!dlg) return checks;

    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 0,
             "settings: what the window shows, the window accepts");
    checks++;
    st_check(f, strcmp(out.system, in.system) == 0,
             "settings: the system prompt box round-trips");
    if (strcmp(out.system, in.system) != 0)
        fprintf(f, "  got [%s]\n", out.system);
    checks++;
    /* THE THINK TOGGLE MUST NOT TOUCH THE SYSTEM PROMPT. Its handler
       reads every control and writes them all back, so the prompt makes
       a full round trip through the box on a click that has nothing to
       do with it - and the box is ANSI while the settings are UTF-8, so
       a missing conversion on the way back shows up as mojibake rather
       than as a lost setting. Driven through the real WM_COMMAND, not
       by calling the handler, so the wiring is covered too. */
    SendMessage(dlg, WM_COMMAND,
                MAKEWPARAM(3001 /* ID_THINK */, BN_CLICKED),
                (LPARAM)GetDlgItem(dlg, 3001));
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 0 &&
                strcmp(out.system, in.system) == 0,
             "settings: toggling think leaves the system prompt intact");
    if (strcmp(out.system, in.system) != 0)
        fprintf(f, "  got [%s] want [%s]\n", out.system, in.system);
    checks++;
    /* Back, so the checks below see the state they were written for. */
    SendMessage(dlg, WM_COMMAND,
                MAKEWPARAM(3001, BN_CLICKED), (LPARAM)GetDlgItem(dlg, 3001));

    /* Empty in, empty out - the box carries the USER's text and empty
       is the state that means "built-in identity". A box that shows
       the built-in constant instead would make "restore defaults"
       commit the constant as a custom prompt. */
    SetWindowTextA(GetDlgItem(dlg, 3020 /* ID_SYS */), "");
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 0 &&
                out.system[0] == '\0',
             "settings: an empty system box means built-in identity, "
             "not a copy of it");
    checks++;
    st_check(f, out.think == in.think,
             "settings: the think box round-trips"); checks++;
    st_check(f, out.ctx == 1536,
             "settings: the context box round-trips");
    if (out.ctx != 1536) fprintf(f, "  got %d\n", out.ctx);
    checks++;
    {
        float dp = out.topp - 0.35f, dr = out.rep - 1.25f;
        if (dp < 0) dp = -dp;
        if (dr < 0) dr = -dr;
        st_check(f, dp < 0.005f, "settings: the top-p box round-trips");
        if (dp >= 0.005f) fprintf(f, "  got %.3f\n", (double)out.topp);
        checks++;
        st_check(f, dr < 0.005f,
                 "settings: the repetition penalty box round-trips");
        if (dr >= 0.005f) fprintf(f, "  got %.3f\n", (double)out.rep);
        checks++;
    }
    st_check(f, out.max_new == 256,
             "settings: the max-new-tokens box round-trips");
    if (out.max_new != 256) fprintf(f, "  got %d\n", out.max_new);
    checks++;

    /* THE THUMBS AGREE WITH THE BOXES. The round-trip checks above go
       box -> reader -> struct and never look at a scrollbar, so a row
       whose SetScrollRange or SetScrollPos is wrong passes every one of
       them - the thumb sits at the wrong place on screen and nothing
       says so. This is the half only a person looking at the window
       would otherwise catch, which is exactly the half worth
       automating. Compared against gui/settings.c's own mapping, which
       is separately tabled in st_sampling, so the two are not one
       observation counted twice. ctx is NOT here: it has no slider. */
    {
        static const struct { int sb_id; int want; } THUMB[] = {
            { 3004, 0 }, { 3023, 0 }, { 3013, 0 }, { 3016, 0 }
        };
        int want[4], got[4], i, bad = 0;
        want[0] = lz_gui_temp_to_scroll(in.temp);
        want[1] = lz_gui_temp_to_scroll(in.think_temp);
        want[2] = lz_common_topp_to_scroll(in.topp);
        want[3] = lz_common_rep_to_scroll(in.rep);
        for (i = 0; i < 4; i++) {
            got[i] = GetScrollPos(GetDlgItem(dlg, THUMB[i].sb_id), SB_CTL);
            if (got[i] != want[i]) bad++;
        }
        st_check(f, bad == 0,
                 "settings: every slider's thumb sits where its own "
                 "value says it should");
        fprintf(f, "  thumbs temp %d/%d thinktemp %d/%d topp %d/%d "
                   "rep %d/%d\n",
                got[0], want[0], got[1], want[1],
                got[2], want[2], got[3], want[3]);
        checks++;
    }
    /* The fields this dialog does not show must come back untouched.
       If the reader filled an uninitialised local and copied it out
       whole, every OK would write
       stack garbage into the seed policy - a non-zero seed_mode reads
       as FIXED, which would pin the session to a fixed nonsense seed
       and then write it to the ini at exit. Nothing would show it,
       because a fixed seed produces perfectly ordinary replies. */
    st_check(f, out.seed_mode == LZ_COMMON_SEED_FIXED &&
                out.seed == 0xC0FFEEull,
             "settings: the reader carries through the fields the "
             "dialog does not own");
    if (out.seed_mode != LZ_COMMON_SEED_FIXED || out.seed != 0xC0FFEEull)
        fprintf(f, "  seed_mode %d seed %lu\n", out.seed_mode,
                (unsigned long)out.seed);
    checks++;
    {
        float d = out.temp - in.temp;
        if (d < 0) d = -d;
        st_check(f, d < 0.01f,
                 "settings: the temperature round-trips through the box");
        if (d >= 0.01f)
            fprintf(f, "  in %.3f out %.3f\n", (double)in.temp,
                    (double)out.temp);
        checks++;
    }
    {
        float d = out.think_temp - in.think_temp;
        if (d < 0) d = -d;
        st_check(f, d < 0.01f,
                 "settings: the think-temp round-trips through the box");
        if (d >= 0.01f)
            fprintf(f, "  in %.3f out %.3f\n", (double)in.think_temp,
                    (double)out.think_temp);
        checks++;
    }

    /* Tab order (the slider/value-box swap): IsDialogMessage
       walks WS_TABSTOP siblings by Z-order, which is creation order,
       not screen position - a position swap that forgets to swap
       creation order to match leaves Tab going right-to-left across a
       row that looks left-to-right, exactly the kind of regression a
       screenshot cannot show and only actually pressing Tab finds.
       GetNextDlgTabItem is the same call IsDialogMessage itself uses
       to build the chain, so this asks the real mechanism rather than
       a proxy for it. Mutation-verified: swapping settingsdlg.c's two
       CreateWindowExA calls back to their old (pre-swap) order while
       leaving the x-coordinates as they are now - i.e. reintroducing
       exactly the regression this check exists for - reddens only
       this check. */
    {
        /* THE WHOLE CHAIN, not the first two links. Two hand-written
           pairs break silently when rows are added in the middle: one
           breaks, the other keeps checking a row that has moved.
           Walking the full order is the same
           amount of code and it covers every row that will ever be
           added, provided whoever adds one extends CHAIN - which they
           will notice, because leaving it alone reddens this. */
        static const int CHAIN[] = {
            3004, 3003,   /* temperature: slider, box */
            3023, 3022,   /* think-temp: slider, box */
            3013, 3012,   /* top-p */
            3016, 3015,   /* repetition penalty */
            3009,         /* context - box only, no slider */
            3018,         /* max new tokens - box only, no slider */
            3020,         /* custom system prompt - box only */
            3007, 3005, 3006,   /* Restore defaults, OK, Cancel - left to right */
            3001          /* wraps back to the think box */
        };
        HWND cur = GetDlgItem(dlg, 3001 /* ID_THINK */);
        int i, bad = -1;
        for (i = 0; cur && i < (int)(sizeof CHAIN / sizeof CHAIN[0]); i++) {
            cur = GetNextDlgTabItem(dlg, cur, FALSE);
            if (cur != GetDlgItem(dlg, CHAIN[i])) { bad = i; break; }
        }
        st_check(f, bad < 0,
                 "settings: Tab visits every row in visual order - "
                 "slider, then value box, row by row");
        if (bad >= 0) fprintf(f, "  diverges at step %d (want id %d)\n",
                              bad, CHAIN[bad]);
        checks++;
    }

    /* Slider longer than the value box (round two of user
       feedback on the swap: the slider read as too short). Compared as
       a RELATIONSHIP, not a pixel count - a hardcoded number would
       break the moment this host's font or DPI differs from whatever
       produced it, exactly what settingsdlg.c's own new "measure, do
       not guess" comment exists to avoid. GetWindowRect's width needs
       no coordinate translation (width is the same in screen and
       client space), so this reads the REAL, already-created controls
       rather than recomputing the geometry a second time. */
    {
        RECT sr, er;
        GetWindowRect(GetDlgItem(dlg, 3004 /* ID_TEMP_SCROLL */), &sr);
        GetWindowRect(GetDlgItem(dlg, 3003 /* ID_TEMP */), &er);
        st_check(f, (sr.right - sr.left) > (er.right - er.left),
                 "settings: the slider is wider than the value box");
        fprintf(f, "  slider %ld px, value box %ld px\n",
                (long)(sr.right - sr.left), (long)(er.right - er.left));
        checks++;

        /* The two "length" rows (context + max-new-tokens) are the same
           shape - a plain box, no slider - so they must be
           the same geometry: one column for the labels, one for the
           value boxes. Left edges AND widths, because equal widths at
           different x would still be two columns. The context row's
           slider is gone (its box is the whole control now), so it is
           compared against the max-new box, not the temperature row. */
        {
            RECT cr, mr;
            GetWindowRect(GetDlgItem(dlg, 3009 /* ID_CTX */), &cr);
            GetWindowRect(GetDlgItem(dlg, 3018 /* ID_MAXNEW */), &mr);
            st_check(f, cr.left == mr.left && cr.right == mr.right,
                     "settings: the context box lines up with the "
                     "max-new box, column for column");
            fprintf(f, "  ctx box %ld..%ld / maxnew box %ld..%ld\n",
                    (long)cr.left, (long)cr.right,
                    (long)mr.left, (long)mr.right);
            checks++;
        }
    }

    /* Theme consistency across the WHOLE dialog (the
       user's second complaint: a classic-drawn scrollbar next to
       themed buttons is still half-and-half, only a different half).
       lz_ui_untheme's own return - S_OK-ness, the same falsifiable
       signal used for the RichEdit/input-box pairing - is what
       gui/settingsdlg.c's g_untheme captures for all six controls;
       this reads it back and asserts every one agrees with the first.
       Pixels are not available here either (no interactive desktop in
       this environment), so agreement is the strongest provable claim:
       either this host has uxtheme and ALL SIX succeed, or it does not
       and ALL SIX report 0 - a host where some controls succeed and
       others do not would mean the "any window, any class" claim in
       lz_ui_untheme's own comment is wrong for at least one class. */
    {
        LZGuiSettingsUntheme u = lz_gui_settings_last_untheme();
        st_check(f, u.temp_scroll == u.think && u.temp == u.think &&
                    u.think_temp == u.think &&
                    u.think_temp_scroll == u.think &&
                    u.topp == u.think && u.topp_scroll == u.think &&
                    u.rep == u.think && u.rep_scroll == u.think &&
                    u.ctx == u.think &&
                    u.maxnew == u.think &&
                    u.ok == u.think && u.cancel == u.think &&
                    u.restore == u.think,
                 "settings: every control's untheme call agrees - the "
                 "dialog is not half classic, half themed");
        fprintf(f, "  untheme think=%d temp=%d/%d thinktemp=%d/%d "
                   "topp=%d/%d rep=%d/%d "
                   "ctx=%d maxnew=%d ok=%d cancel=%d restore=%d\n",
                u.think, u.temp, u.temp_scroll, u.think_temp,
                u.think_temp_scroll, u.topp, u.topp_scroll,
                u.rep, u.rep_scroll, u.ctx, u.maxnew,
                u.ok, u.cancel, u.restore);
        checks++;
    }

    /* Wiring proof for one representative control - the scrollbar,
       the one this round's actual report was about - same shape as
       gui/main.c's own g.transcript_untheme_ok check: the stored value
       from lz_gui_settings_dialog_create's real call, compared against
       an independent, side-effect-free (SetWindowTheme("","") is
       idempotent) fresh call on the SAME control. Proof the real call
       site actually ran, not just that calling lz_ui_untheme on a
       SCROLLBAR happens to work when this file does it directly - the
       cross-control check above cannot tell those two apart on its
       own, the same reason lz_drop_accept's own wiring check exists
       beside its capability check. */
    {
        LZGuiSettingsUntheme u = lz_gui_settings_last_untheme();
        st_check(f, u.temp_scroll ==
                    lz_ui_untheme(GetDlgItem(dlg, 3004 /* ID_TEMP_SCROLL */)),
                 "settings: the scrollbar's untheme call really ran, "
                 "not just declared");
        checks++;
    }

    /* A value the rules refuse must be refused HERE too, at the point
       the user typed it - not clamped on the way out. */
    SetWindowTextA(GetDlgItem(dlg, 3003 /* ID_TEMP */), "1.5");
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) != 0,
             "settings: an over-cap value is refused by the reader");
    checks++;
    SetWindowTextA(GetDlgItem(dlg, 3003), "not a number");
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) != 0,
             "settings: text that is not a number is refused");
    checks++;
    /* And the context box's own refusal, with a DISTINCT code - the
       caller picks the message from it, so a shared 1 would send the
       user to fix the temperature they never touched. Temperature is
       put back first: the reader checks it before the context, so a
       still-broken 1.5 would return 1 and this check would pass while
       measuring nothing about the context box at all. */
    SetWindowTextA(GetDlgItem(dlg, 3003), "0.37");
    SetWindowTextA(GetDlgItem(dlg, 3009 /* ID_CTX */), "40000");
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 2,
             "settings: a context outside the range is refused, and "
             "says which box");
    if (lz_gui_settings_dialog_read(dlg, &in, &out) != 2)
        fprintf(f, "  rc %d\n", lz_gui_settings_dialog_read(dlg, &in, &out));
    checks++;
    SetWindowTextA(GetDlgItem(dlg, 3009), "");
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 2,
             "settings: an empty context box is refused too");
    checks++;
    SetWindowTextA(GetDlgItem(dlg, 3009), "1536");

    /* EVERY box names ITSELF, as a table. Five codes and five boxes:
       the mapping is the whole point, because the caller picks the
       message from the code, and a dialog that reports the wrong field
       sends the user to correct a value that was fine. Each case puts
       ONE box out of range and restores it before the next, so a code
       can only come from the box under test - a leftover bad value in
       an earlier-checked box would shadow every case after it (the
       reader returns at the first refusal), which is exactly how a
       still-broken 1.5 in the temperature box would make the other
       four cases pass while measuring nothing. */
    {
        static const struct { int id; const char *bad; int want;
                              const char *ok; } CASES[] = {
            { 3003, "1.5",  1, "0.37" },     /* temperature, over cap  */
            { 3009, "40000", 2, "1536" },    /* context, over max (32768) */
            { 3012, "1.4",  3, "0.35" },     /* top-p, over 1.0        */
            { 3015, "2.0",  4, "1.25" },     /* rep penalty, over 1.5  */
            { 3018, "8",    5, "256"  },     /* max new, under floor   */
            { 3022, "1.5",  6, "0.30" }      /* think-temp, over cap   */
        };
        int i, bad = 0;
        for (i = 0; i < (int)(sizeof CASES / sizeof CASES[0]); i++) {
            int rc;
            SetWindowTextA(GetDlgItem(dlg, CASES[i].id), CASES[i].bad);
            rc = lz_gui_settings_dialog_read(dlg, &in, &out);
            if (rc != CASES[i].want) {
                bad++;
                fprintf(f, "  id %d = \"%s\" gave rc %d, want %d\n",
                        CASES[i].id, CASES[i].bad, rc, CASES[i].want);
            }
            SetWindowTextA(GetDlgItem(dlg, CASES[i].id), CASES[i].ok);
        }
        st_check(f, bad == 0,
                 "settings: each box's refusal reports that box, not "
                 "another one");
        checks++;
        st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 0,
                 "settings: every box is accepted again once restored - "
                 "the loop above left nothing behind");
        checks++;
    }

    /* -1 is a legal max-new and is NOT a refusal, which is the one
       thing about that box a reader could plausibly get wrong: it is
       the only box in this dialog whose valid set is not an interval. */
    SetWindowTextA(GetDlgItem(dlg, 3018 /* ID_MAXNEW */), "-1");
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 0 &&
                out.max_new == LZ_COMMON_MAXNEW_UNLIMITED,
             "settings: -1 in the max-new box means unlimited, not an "
             "error");
    checks++;
    /* Capped against the context THIS dialog is committing, not the one
       in force. 1600 is above the 1536 in the context box, so it must
       be refused; the same number with a larger context in the same
       pass must not be. That second half is what proves the cap reads
       the box rather than a snapshot taken before the dialog opened. */
    SetWindowTextA(GetDlgItem(dlg, 3018), "1600");
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 5,
             "settings: a max-new above the context being committed is "
             "refused");
    checks++;
    SetWindowTextA(GetDlgItem(dlg, 3009 /* ID_CTX */), "4096");
    st_check(f, lz_gui_settings_dialog_read(dlg, &in, &out) == 0 &&
                out.max_new == 1600,
             "settings: the same max-new is accepted when the context "
             "in the SAME dialog is raised to fit it");
    checks++;
    SetWindowTextA(GetDlgItem(dlg, 3009), "1536");
    SetWindowTextA(GetDlgItem(dlg, 3018), "256");

    lz_gui_format_temp(0.6f, buf, (int)sizeof buf);
    st_check(f, strcmp(buf, "0.60") == 0,
             "settings: the formatter is stable");
    if (strcmp(buf, "0.60") != 0) fprintf(f, "  got %s\n", buf);
    checks++;

    /* NO TWO CONTROLS IN THIS WINDOW MAY SHARE A MNEMONIC. The menu bar
       carries this rule and the dialog does too (its seven are
       T P O E N M R) - a collision is both easy to introduce and
       completely silent: Windows cycles focus between the clashing
       controls instead of activating one, which looks like the key
       doing nothing.
       Read off the REAL controls rather than the string table: a
       label's '&' only matters if that label is actually in the
       window, and the table holds strings for menus and buttons that
       are not. Both languages, because the two tables are written
       independently and only one of them is on screen at a time. */
    {
        int lang, bad = 0;
        int was_english = lz_str_lang_english();
        for (lang = 0; lang < 2; lang++) {
            HWND probe;
            HWND c;
            char seen[256];
            memset(seen, 0, sizeof seen);
            lz_str_init(lang);
            probe = lz_gui_settings_dialog_create(NULL, inst, &in);
            if (!probe) { bad++; continue; }
            for (c = GetWindow(probe, GW_CHILD); c;
                 c = GetWindow(c, GW_HWNDNEXT)) {
                char t[128], *amp;
                t[0] = '\0';
                GetWindowTextA(c, t, (int)sizeof t);
                amp = strchr(t, '&');
                /* "&&" is a literal ampersand, not a mnemonic. */
                while (amp && amp[1] == '&') amp = strchr(amp + 2, '&');
                if (amp && amp[1]) {
                    unsigned char k = (unsigned char)amp[1];
                    if (k >= 'a' && k <= 'z') k = (unsigned char)(k - 32);
                    if (seen[k]) {
                        bad++;
                        fprintf(f, "  lang %d: '%c' is on two controls "
                                   "(%s)\n", lang, (char)k, t);
                    }
                    seen[k] = 1;
                }
            }
            DestroyWindow(probe);
        }
        lz_str_init(was_english);
        st_check(f, bad == 0,
                 "settings: no two controls in the dialog share a "
                 "mnemonic, in either language");
        checks++;
    }

    DestroyWindow(dlg);
    return checks;
}

/* ---- the custom system prompt at render time ----
 *
 * This is the criterion that matters, and the dialog round-trip in
 * st_settings_dialog cannot see it: whether the system prompt the
 * user typed actually lands in the RENDERED conversation. It is
 * exercised through the product's own begin path, not through a copy
 * of render_conv's body.
 *
 * The distinguishing pair, driven through the real session:
 *   empty    -> the render contains the ENGINE's built-in identity,
 *               byte for byte - lz_chat_default_system(), not a copy
 *               of the text
 *   non-empty -> the render contains the USER text, and does NOT
 *               contain the built-in (REPLACE, not append)
 * Both against a session whose history is real turns, because a
 * system message that renders on an empty history but collides with
 * a turn would be a different bug.
 *
 * The built-in is checked against the GETTER's return, byte for
 * byte, rather than against a literal here - this pins the renderer
 * to the single source of truth, not to a second copy of the text.
 *
 * MUTATION MATRIX - each applied alone, each reverted:
 *   A  render_conv never prepends the custom system     -> 1
 *      (the REPLACE criterion)
 *   B  render_conv skips the empty branch                -> 2
 *      (empty->built-in, and clear->restored; the empty branch
 *      IS the "built-in identity" path)
 *   C  save_chat_to drops the system prefix              -> 1
 *      (the round-trip criterion - the gate's own baseline must
 *      clear g.set.system before the load, or a dropped write
 *      passes on the value the test set itself)
 *   D  load_chat_from does not feed back into settings   -> 1
 *      (the same round-trip criterion, other direction)
 */
static int st_system_prompt(FILE *f) {
    int checks = 0;
    char err[256];
    const char *builtin = lz_chat_default_system();
    int blen = (int)strlen(builtin);

    lz_gui_session_reset(&g.sess);
    lz_gui_session_set_system(&g.sess, "");
    g.sess.hist.n = 0;

    /* Empty -> built-in identity, byte for byte. */
    if (lz_gui_session_begin(&g.sess,
                             "hi", 2, err, (int)sizeof err) == 0) {
        int plen;
        const char *p = lz_gui_session_prompt(&g.sess, &plen);
        int found = 0, i;
        for (i = 0; i + blen <= plen; i++)
            if (memcmp(p + i, builtin, (size_t)blen) == 0) { found = 1; break; }
        st_check(f, found,
                 "system: empty settings prompt -> the built-in identity "
                 "renders, byte for byte");
        checks++;
    }

    lz_gui_session_reset(&g.sess);
    g.sess.hist.n = 0;

    /* Non-empty -> user text, no built-in. */
    lz_gui_session_set_system(&g.sess,
        /* UTF-8 for "you are a test assistant." */
        "\xe4\xbd\xa0\xe6\x98\xaf\xe6\xb5\x8b\xe8\xaf\x95"
        "\xe5\x8a\xa9\xe6\x89\x8b\xe3\x80\x82");
    if (lz_gui_session_begin(&g.sess,
                             "hi", 2, err, (int)sizeof err) == 0) {
        int plen;
        const char *p = lz_gui_session_prompt(&g.sess, &plen);
        int has_custom = 0, has_builtin = 0, i;
        static const char custom[] =
            "\xe4\xbd\xa0\xe6\x98\xaf\xe6\xb5\x8b\xe8\xaf\x95"
            "\xe5\x8a\xa9\xe6\x89\x8b\xe3\x80\x82";
        int clen = (int)sizeof custom - 1;
        for (i = 0; i + clen <= plen; i++)
            if (memcmp(p + i, custom, (size_t)clen) == 0) { has_custom = 1; break; }
        for (i = 0; i + blen <= plen; i++)
            if (memcmp(p + i, builtin, (size_t)blen) == 0) { has_builtin = 1; break; }
        st_check(f, has_custom && !has_builtin,
                 "system: a custom prompt REPLACES the built-in identity "
                 "in the render");
        if (!has_custom || has_builtin)
            fprintf(f, "  custom %d builtin %d\n", has_custom, has_builtin);
        checks++;
    }

    /* Clearing the setting (restore defaults) goes back to the built-in.
       Same render path, so this proves the toggle is live in both
       directions. */
    lz_gui_session_reset(&g.sess);
    g.sess.hist.n = 0;
    lz_gui_session_set_system(&g.sess, "");
    if (lz_gui_session_begin(&g.sess,
                             "hi", 2, err, (int)sizeof err) == 0) {
        int plen, found = 0, i;
        const char *p = lz_gui_session_prompt(&g.sess, &plen);
        for (i = 0; i + blen <= plen; i++)
            if (memcmp(p + i, builtin, (size_t)blen) == 0) { found = 1; break; }
        st_check(f, found,
                 "system: clearing the custom prompt restores the built-in");
        checks++;
    }

    lz_gui_session_reset(&g.sess);
    return checks;
}

/* ---- the sampling settings ----

   THE RULES, not the controls - the dialog half is st_settings_dialog
   above. Three things live here because each is a place where a
   plausible implementation is wrong in a way no screenshot shows:

     - top_p and temperature must follow the preset INDEPENDENTLY. One
       shared "the user set something" flag passes every round-trip
       check ever written and still freezes the wrong value.
     - -1 and 0 mean the same thing to the engine, so a max_new of -1
       has to arrive at opts as something <= 0 and NOT be normalised on
       the way.
     - the repetition penalty has to be sent even when it equals 1.0,
       because 1.0 is the identity and three upstream projects disagree
       about what "not sent" means (sampler.h).

   MUTATIONS, each alone, each reverted:

     1  set_think uses one flag for both      -> 1 (independence)
     2  apply_settings drops topp             -> 1 (reaches opts)
     3  apply_settings drops rep              -> 2
        (reaches-the-engine, and the explicit-1.0 check with it: with
        the assignment gone the preset's 1.1 is what stays, which is
        precisely the failure that check exists for. Real coupling,
        not a duplicate observation - one reads a value the user set,
        the other reads the identity value that looks like "not set")
     4  apply_settings drops max_new          -> 1
     5  apply_settings drops repeat_last_n    -> 1
     6  set_max_new normalises -1 to 0        -> 2
        (the rule check and the dialog's own -1 case; the ENGINE would
        behave identically, which is exactly why both read the stored
        value rather than the model's output - this defect is
        unobservable downstream)
     7  set_rep accepts anything              -> 2 (the range table,
        and the dialog's per-box refusal table - the second is what
        proves the dialog's refusal comes from the setter and not from
        a range the dialog re-states for itself)
     8  restore leaves rep/max_new alone      -> 1 */
static int st_sampling(FILE *f) {
    int checks = 0;
    LZGuiSettings s;

    /* Independence of the two preset-following values. Set ONLY the
       temperature, then toggle: top_p must move, the temperature must
       not. Then the mirror image. A shared flag fails both halves; a
       missing flag fails the first. */
    {
        float think_topp = lz_common_settings_default_topp(1);
        float inst_topp  = lz_common_settings_default_topp(0);
        int ok;

        lz_common_settings_init(&s);            /* think on */
        lz_common_settings_set_temp(&s, 0.42f);
        lz_common_settings_set_think(&s, 0);
        ok = (s.temp > 0.415f && s.temp < 0.425f) &&
             (s.topp > inst_topp - 0.001f && s.topp < inst_topp + 0.001f);

        lz_common_settings_init(&s);
        lz_common_settings_set_topp(&s, 0.42f);
        lz_common_settings_set_think(&s, 0);
        ok = ok && (s.topp > 0.415f && s.topp < 0.425f) &&
             (s.temp > lz_common_settings_default_temp(0) - 0.001f &&
              s.temp < lz_common_settings_default_temp(0) + 0.001f);

        st_check(f, ok,
                 "sampling: temperature and top-p follow the preset "
                 "independently - setting one does not freeze the other");
        if (!ok)
            fprintf(f, "  temp %.3f topp %.3f (think topp %.3f, "
                       "instruct topp %.3f)\n",
                    (double)s.temp, (double)s.topp,
                    (double)think_topp, (double)inst_topp);
        checks++;
    }

    /* The three refusal ranges, as a table. Each row is a value the
       setter must take and one it must refuse, so a setter that
       accepts everything and a setter that refuses everything are both
       red - one predicate cannot be fooled by either. */
    {
        int bad = 0;
        lz_common_settings_init(&s);
        if (lz_common_settings_set_topp(&s, 0.05f) != 0) bad++;
        if (lz_common_settings_set_topp(&s, 1.0f)  != 0) bad++;
        if (lz_common_settings_set_topp(&s, 0.04f) == 0) bad++;
        if (lz_common_settings_set_topp(&s, 1.01f) == 0) bad++;
        if (lz_common_settings_set_rep(&s, 1.0f)   != 0) bad++;
        if (lz_common_settings_set_rep(&s, 1.5f)   != 0) bad++;
        if (lz_common_settings_set_rep(&s, 0.99f)  == 0) bad++;
        if (lz_common_settings_set_rep(&s, 1.51f)  == 0) bad++;
        if (lz_common_settings_set_max_new(&s, 16, 0)     != 0) bad++;
        if (lz_common_settings_set_max_new(&s, 15, 0)     == 0) bad++;
        if (lz_common_settings_set_max_new(&s, 4096, 2048) == 0) bad++;
        if (lz_common_settings_set_max_new(&s, 2048, 2048) != 0) bad++;
        st_check(f, bad == 0,
                 "sampling: each setter takes its own range and refuses "
                 "just outside it");
        if (bad) fprintf(f, "  %d of the range cases went the wrong way\n",
                         bad);
        checks++;
    }

    /* -1 is normalised HERE and only here, and it stays -1. Any
       negative means unlimited; nothing downstream should ever see -7,
       and nothing should turn -1 into 0. */
    {
        int ok;
        lz_common_settings_init(&s);
        ok = lz_common_settings_set_max_new(&s, -7, 2048) == 0 &&
             s.max_new == LZ_COMMON_MAXNEW_UNLIMITED;
        ok = ok && lz_common_settings_set_max_new(&s, -1, 2048) == 0 &&
             s.max_new == LZ_COMMON_MAXNEW_UNLIMITED;
        st_check(f, ok,
                 "sampling: any negative max-new becomes -1 exactly, "
                 "not 0 and not itself");
        if (!ok) fprintf(f, "  max_new %d\n", s.max_new);
        checks++;
    }

    /* Restore covers EVERY field. The table is what a half-restore
       looks like from outside: one of these staying put is a "restore
       defaults" that quietly did not. */
    {
        int ok;
        lz_common_settings_init(&s);
        lz_common_settings_set_temp(&s, 0.11f);
        lz_common_settings_set_topp(&s, 0.22f);
        lz_common_settings_set_think_temp(&s, 0.19f);
        lz_common_settings_set_rep(&s, 1.45f);
        lz_common_settings_set_max_new(&s, 64, 0);
        s.ctx = 512;
        lz_common_settings_restore(&s);
        ok = s.manual_temp == 0 && s.manual_topp == 0 &&
             s.manual_think_temp == 0 &&
             s.max_new == LZ_COMMON_MAXNEW_UNLIMITED &&
             s.ctx == LZ_COMMON_CTX_DEFAULT &&
             s.temp == lz_common_settings_default_temp(s.think) &&
             s.topp == lz_common_settings_default_topp(s.think) &&
             s.rep  == lz_common_settings_default_rep(s.think) &&
             s.think_temp == lz_common_settings_default_think_temp(s.think);
        st_check(f, ok,
                 "sampling: restore puts back every setting, not only "
                 "the temperature");
        if (!ok)
            fprintf(f, "  temp %.2f topp %.2f thinktemp %.2f rep %.2f "
                       "max_new %d "
                       "ctx %d manual %d/%d/%d\n",
                    (double)s.temp, (double)s.topp, (double)s.think_temp,
                    (double)s.rep,
                    s.max_new, s.ctx, s.manual_temp, s.manual_topp,
                    s.manual_think_temp);
        checks++;
    }

    /* Every value reaches the ENGINE's option block, through the
       product's own apply_settings and not through a copy of its body.
       g.set is driven directly and put back afterwards. */
    {
        LZGuiSettings saved = g.set;
        int saved_rln = g.repeat_last_n;
        int ok;

        lz_common_settings_init(&g.set);
        lz_common_settings_set_topp(&g.set, 0.33f);
        lz_common_settings_set_rep(&g.set, 1.45f);
        lz_common_settings_set_think_temp(&g.set, 0.19f);
        lz_common_settings_set_max_new(&g.set, 123, 0);
        g.repeat_last_n = 77;
        apply_settings();
        ok = g.sess.opts.sample.topp > 0.325f &&
             g.sess.opts.sample.topp < 0.335f &&
             g.sess.opts.sample.repetition_penalty > 1.445f &&
             g.sess.opts.sample.repetition_penalty < 1.455f &&
             g.sess.opts.sample.temp_think > 0.185f &&
             g.sess.opts.sample.temp_think < 0.195f &&
             g.sess.opts.sample.think_temp_enabled == 1 &&
             g.sess.opts.max_new_tokens == 123 &&
             g.sess.opts.sample.repeat_last_n == 77;
        st_check(f, ok,
                 "sampling: top-p, the penalty, its window, the "
                 "generation cap and the think-block temperature all "
                 "reach the engine's options");
        if (!ok)
            fprintf(f, "  topp %.3f rep %.3f thinktemp %.3f/%d "
                       "max_new %d window %d\n",
                    (double)g.sess.opts.sample.topp,
                    (double)g.sess.opts.sample.repetition_penalty,
                    (double)g.sess.opts.sample.temp_think,
                    g.sess.opts.sample.think_temp_enabled,
                    g.sess.opts.max_new_tokens,
                    g.sess.opts.sample.repeat_last_n);
        checks++;

        /* Unlimited goes down as something the engine reads as "no
           cap". <= 0 rather than == -1 on purpose: llama_zh.h's
           contract is "<=0 means the model's seq_len", and pinning the
           exact spelling here would be this check asserting a
           convention rather than a behaviour. */
        lz_common_settings_set_max_new(&g.set, -1, 0);
        apply_settings();
        st_check(f, g.sess.opts.max_new_tokens <= 0,
                 "sampling: unlimited reaches the engine as no cap");
        if (g.sess.opts.max_new_tokens > 0)
            fprintf(f, "  max_new_tokens %d\n", g.sess.opts.max_new_tokens);
        checks++;

        /* A penalty of exactly 1.0 is the identity, and it must still
           be sent - sampler.h: HuggingFace defaults to 1.0, llama.cpp
           to 1.1, vLLM to 1.0, so "leave it out" means three different
           things. The preset writes 1.1, so seeing 1.0 here can only
           mean the explicit value arrived. */
        lz_common_settings_set_rep(&g.set, 1.0f);
        apply_settings();
        st_check(f, g.sess.opts.sample.repetition_penalty > 0.995f &&
                    g.sess.opts.sample.repetition_penalty < 1.005f,
                 "sampling: a penalty of 1.0 is sent explicitly, not "
                 "left to whatever the preset put there");
        if (!(g.sess.opts.sample.repetition_penalty > 0.995f &&
              g.sess.opts.sample.repetition_penalty < 1.005f))
            fprintf(f, "  rep %.3f\n",
                    (double)g.sess.opts.sample.repetition_penalty);
        checks++;

        g.set = saved;
        g.repeat_last_n = saved_rln;
        apply_settings();
    }

    return checks;
}

/* ---- the context window ----

   WHAT THIS HOST CAN AND CANNOT REACH. There is no model in this
   environment, so "set 1024 and watch the run state come back 1024" is
   not available end to end; lz_state_alloc needs real weights. What IS
   available is every step on the way there, and the instrument that
   makes them observable is gui/modelload.c's state_alloc wrapper:
   lz_gui_state_alloc_last records the seq_len the allocator was handed,
   at the single point both callers funnel through, so it cannot drift
   from what lz_state_alloc really receives. Asserting on it is strictly
   stronger than asserting on the setting - the criterion is "read back
   the allocated value, not the setting".

   THE FAILURE PATH IS THE POINT OF THE WHOLE ITEM and it is fully
   reachable, because a synthesized failure needs no allocator at all -
   lz_gui_state_alloc_fail short-circuits before lz_state_alloc is
   called, which is also why a fake LZGuiModel with have_model set and a
   zeroed LZModel behind it is safe here: nothing dereferences the model
   on that path. Asking this machine for a genuinely impossible
   allocation is not an option; 2 GB succeeds here, so a gate written
   that way would be green forever.

   THREE SEPARATE ASSERTIONS on failure, not one &&. The split is
   deliberate and right: the mutation that matters most
   ("fall back to the largest size that does allocate") leaves the old
   state alive and the error reported, and changes only whether the
   setting rolls back. An && would hide which third of the behaviour
   broke.

   MUTATION MATRIX - each applied alone, each reverted, with the checks
   it reddens:

     1  lz_common_ctx_clamp: drop the model_cap clamp        -> 2
        (the clamp table, and the "clamped to the model" check that
        reads it back through resize - the second is what proves the
        clamp is WIRED, not merely correct)
     2  set_ctx_cell: denominator back to a constant      -> 2
        (both status-cell checks; one constant cannot be right for two
        different windows at once)
     3  effective_ctx: return g.set.ctx unconditionally   -> 1
        (only the allocated-size denominator)
     4  lz_gui_model_resize: free the old state first     -> 1
        The literal mutation - move lz_state_free above state_alloc -
        CRASHES on the fake state (0xA5 pointers handed to free), and a
        crash proves nothing about an assertion that never ran. The
        surgical form is `memset(&m->state, 0, sizeof m->state)` in the
        same position, which is not an approximation: lz_state_free
        ends with exactly that memset, so it is the identical
        post-state minus the free() calls. Reddens the "untouched on
        failure" check alone.
     5  ctx_commit: do not roll g.set.ctx back            -> 2
        (the struct and the ini; the ini is written FROM the struct, so
        a wrong struct necessarily gives a wrong ini - real coupling in
        one direction, and 5b below is the other direction)
     5b ctx_commit: roll the struct back, not the ini     -> 1
        This one reddens only after the ini baseline is pinned: the ini
        file usually already holds 2048, so "the ini says 2048" reads
        ambient state rather than anything this code did. Writing 12345
        first (see st_ctx_commit) is what makes it discriminating. A
        gate whose baseline is the environment tests the environment.
     6  ctx_commit: treat a failed resize as success      -> 3
     7  model_load_prepare: drop the seq_want line        -> 1
     8  lz_gui_settings_dialog_read: drop `s = *cur`      -> 1
        (the carried-through-fields check; think/temp/ctx are all
        assigned afterwards, so nothing else moves)
     9  (removed: lz_common_ctx_to_scroll and its slider table died with
        the context slider; the box is the whole control now)
    10  the reader accepts any context text               -> 2
        (both refusal checks - one predicate, two inputs)
    10b the context refusal returns 1, the temperature's  -> 2
        code, instead of 2 (both refusal checks again; what it proves
        is that they read the CODE and not merely non-zero, which is
        what decides which box the message box names)
    13  the context row measures its own field width      -> 1
        (the column-alignment check - and this one existed only as a
        screenshot until the row was rebuilt; see the check itself)
    14  the box is created showing the default, not the
        incoming value                                    -> 1
    11  ctx_commit: drop the unchanged-size early-out     -> 0
        Also nothing, and this one is NOT a gate defect: resize has its
        own `seq_len == seq` early-out, so the outcome the check asserts
        still holds. Removing BOTH (11b) reddens it. Recorded because
        "the mutation did nothing" and "the check is asleep" look the
        same from the outside and are not the same thing.
    12  ctx_commit: drop lz_gui_session_prefix_clear      -> 1

   TWO THINGS HERE HAVE NO AUTOMATED GATE, said plainly rather than
   left to be discovered:

     - That WM_COMMAND's IDM_SETTINGS actually calls ctx_apply. The
       handler opens a modal dialog, so nothing in a selftest can drive
       it; deleting the ctx_apply call reddens nothing.
     - That the error box appears, as opposed to ctx_commit reporting
       that it should. A modal MessageBox in this process would hang
       the run.

   ONE TOOLING NOTE, because it cost two mutations' worth of confusion:
   `make gui` fails with "cannot open output file ... Permission denied"
   while a copy of the window is still running (a leftover screenshot
   process, say), and the run right after it then exercises the
   PREVIOUS build. The symptom is a mutation whose reds belong to the
   mutation before it. Grep the make output for "error" before trusting
   any mutation result. */
static int st_ctx_window(FILE *f) {
    int checks = 0;

    /* The clamp, as a table. Pure arithmetic, so every interesting
       pair is cheap - and "interesting" here means each boundary and
       each of the two clamps in isolation, because a single case that
       happens to exercise both cannot tell which one is missing. */
    {
        static const struct { int want, cap, expect; } CASES[] = {
            /* in range, no model */
            { 2048,    0, 2048 },
            /* below the floor / above the ceiling */
            {   16,    0,  512 },
            {    0,    0,  512 },
            {   -1,    0,  512 },
            { 99999,   0, 32768 },
            /* the model is the smaller number - the step the CLI has
               and the GUI did not */
            { 32768, 4096, 4096 },
            { 2048, 1000, 1000 },
            /* the model is the LARGER number: it does not raise the
               ceiling, because the ceiling is the model's own
               max_position_embeddings (32768 on KunMoe) */
            { 99999, 32768, 32768 },
            /* both clamps at once, ceiling first then cap */
            { 99999, 4096, 4096 },
            /* exactly on a boundary, both ends */
            {  512,    0,  512 },
            { 32768,    0, 32768 }
        };
        int i, bad = 0;
        for (i = 0; i < (int)(sizeof CASES / sizeof CASES[0]); i++) {
            int got = lz_common_ctx_clamp(CASES[i].want, CASES[i].cap);
            if (got != CASES[i].expect) {
                bad++;
                fprintf(f, "  clamp(%d, %d) = %d, want %d\n",
                        CASES[i].want, CASES[i].cap, got, CASES[i].expect);
            }
        }
        st_check(f, bad == 0,
                 "ctx: the clamp takes the floor, the ceiling and the "
                 "model's own limit");
        checks++;
    }

    /* The slider's mapping died with the slider: the context is a plain
       box now, so there is no scroll position to
       round-trip and no step to snap to. The clamp table above is what
       remains to check - the box takes any integer in [MIN, MAX] and
       lz_common_ctx_clamp still bounds the allocation. */

    /* The load path takes its size from the settings. One line, and
       invisible when forgotten: a stale seq_want allocates the previous
       size, the bar reports the previous size, and the two agree. */
    {
        char saved_dir[512];
        int saved_want = g.mdl.seq_want, old_ctx = g.set.ctx;
        memcpy(saved_dir, g.mdl.dir, sizeof saved_dir);

        g.set.ctx = 1536;
        g.mdl.seq_want = -12345;
        model_load_prepare("C:\\kk98-ctx-probe");
        st_check(f, g.mdl.seq_want == 1536,
                 "ctx: a model load asks for the size the settings hold");
        if (g.mdl.seq_want != 1536)
            fprintf(f, "  seq_want %d\n", g.mdl.seq_want);
        checks++;

        memcpy(g.mdl.dir, saved_dir, sizeof saved_dir);
        g.mdl.seq_want = saved_want;
        g.set.ctx = old_ctx;
    }

    /* What the allocator is handed, on a fake model, with the
       allocation short-circuited. The fake is local - not g.mdl - so
       nothing this block does can be seen by anything after it. */
    {
        LZGuiModel fake;
        char err[256];
        int rc;

        memset(&fake, 0, sizeof fake);
        fake.have_model = 1;
        fake.have_state = 1;
        fake.seq_len = 2048;
        fake.model.config.seq_len = 0;   /* no cap */

        lz_gui_state_alloc_last = -1;
        lz_gui_state_alloc_fail = 1;
        err[0] = '\0';
        rc = lz_gui_model_resize(&fake, 1024, err, (int)sizeof err);
        st_check(f, rc != 0 && lz_gui_state_alloc_last == 1024 &&
                    lz_gui_state_alloc_fail == 0,
                 "ctx: the size the settings hold is the size the "
                 "allocator is asked for");
        fprintf(f, "  rc %d asked %d fail-counter %d\n", rc,
                lz_gui_state_alloc_last, lz_gui_state_alloc_fail);
        checks++;

        /* The model's own ceiling wins, and it is read back from what
           the ALLOCATOR was asked for rather than from the setting.
           3000 is not a
           multiple of the step, so it can only have come from the cap. */
        fake.model.config.seq_len = 3000;
        lz_gui_state_alloc_last = -1;
        lz_gui_state_alloc_fail = 1;
        rc = lz_gui_model_resize(&fake, 32768, err, (int)sizeof err);
        st_check(f, rc != 0 && lz_gui_state_alloc_last == 3000,
                 "ctx: a request above the model's own limit is clamped "
                 "to the model, not to the request");
        if (lz_gui_state_alloc_last != 3000)
            fprintf(f, "  asked %d\n", lz_gui_state_alloc_last);
        checks++;

        /* Failure leaves the model byte-for-byte alone. Stronger than
           "it can still generate", and the only form of that claim
           this host can make: a memcmp of the whole struct sees a
           freed-then-reallocated state, a changed seq_len, and a
           cleared have_state alike. Filled with a recognisable pattern
           first so the comparison is against something, not against
           two identical zero pages. */
        {
            LZGuiModel before;
            memset(&fake.state, 0xA5, sizeof fake.state);
            fake.seq_len = 2048;
            fake.model.config.seq_len = 0;
            before = fake;
            lz_gui_state_alloc_fail = 1;
            rc = lz_gui_model_resize(&fake, 4096, err, (int)sizeof err);
            st_check(f, rc != 0 &&
                        memcmp(&before, &fake, sizeof fake) == 0,
                     "ctx: a failed resize leaves the run state exactly "
                     "as it was");
            checks++;
            st_check(f, err[0] != '\0',
                     "ctx: a failed resize reports why");
            checks++;
            memset(&fake.state, 0, sizeof fake.state);
        }
        lz_gui_state_alloc_fail = 0;
    }

    return checks;
}

/* ---- ctx_commit's three outcomes on a synthesized failure ----

   Separate from st_ctx_window because this one drives the REAL g.mdl
   and the REAL g.set, and has to put both back. It is also the only
   part that can speak to "the user is told": the box itself is a modal
   MessageBox that cannot be shown here without hanging, so what is
   asserted is the tuple that decides it - ctx_commit's return code and
   the message it filled - and the box is one `if` in ctx_apply, the
   single caller. Same construction as measuring a control's state and
   gating the measured tuple when the control itself will not
   reproduce. */
static int st_ctx_commit(FILE *f, HWND hwnd) {
    int checks = 0;
    int old_ctx = g.set.ctx;
    int old_have_model = g.mdl.have_model, old_have_state = g.mdl.have_state;
    int old_seq = g.mdl.seq_len, old_want = g.mdl.seq_want;
    int old_ini = lz_ini_get_int("ctx", LZ_COMMON_CTX_DEFAULT);
    char err[512];
    int rc;

    /* Enough of a model for lz_gui_model_ready and for resize to get as
       far as the allocator, with the allocation short-circuited so the
       zeroed LZModel behind it is never dereferenced. have_tok too:
       lz_gui_model_ready wants all three. */
    g.mdl.have_model = 1;
    g.mdl.have_tok = 1;
    g.mdl.have_state = 1;
    g.mdl.seq_len = 2048;

    g.set.ctx = 4096;
    /* A value nothing here can produce, written FIRST. Without it the
       ini check below reads whatever the file already held - and 2048
       is exactly what it usually holds, so the check passed with the
       write deleted. Caught by running that mutation, not by reading
       the check: a gate whose "before" is ambient state is testing the
       environment, not the code. */
    lz_ini_set_int("ctx", 12345);
    lz_gui_state_alloc_fail = 1;
    err[0] = '\0';
    rc = ctx_commit(hwnd, 2048, err, (int)sizeof err);

    st_check(f, rc != 0 && err[0] != '\0',
             "ctx: a failed resize is reported to the caller, which is "
             "what puts the error box on screen");
    checks++;
    st_check(f, g.set.ctx == 2048,
             "ctx: a failed resize puts the setting back to what it was");
    if (g.set.ctx != 2048) fprintf(f, "  ctx %d\n", g.set.ctx);
    checks++;
    st_check(f, g.mdl.seq_len == 2048 && g.mdl.have_state == 1,
             "ctx: a failed resize leaves the old window in force");
    checks++;
    st_check(f, lz_ini_get_int("ctx", -1) == 2048,
             "ctx: the rolled-back value reaches the ini, not just the "
             "struct");
    if (lz_ini_get_int("ctx", -1) != 2048)
        fprintf(f, "  ini ctx %d\n", lz_ini_get_int("ctx", -1));
    checks++;

    /* No change requested is not a resize: the counter must survive
       untouched, which proves the early-out is an early-out and not a
       resize that happened to land on the same number. */
    lz_gui_state_alloc_fail = 1;
    rc = ctx_commit(hwnd, g.set.ctx, err, (int)sizeof err);
    st_check(f, rc == 0 && lz_gui_state_alloc_fail == 1,
             "ctx: committing an unchanged size allocates nothing");
    checks++;
    lz_gui_state_alloc_fail = 0;

    /* The SUCCESS path, reached the only way this host can reach it:
       with have_state clear, lz_gui_model_resize has nothing to
       re-allocate and returns 0 without touching the allocator - so
       everything ctx_commit does AFTER a successful resize runs, which
       is the half a synthesized failure can never exercise.
       What is being checked is the prefix cache. The new run state
       carries a new epoch, so lz_ckpt_restore would refuse a stale
       checkpoint anyway; relying on that refusal is relying on a
       downstream guard to cover an upstream mistake, which is how the
       lifecycle rule in gui/session.h got written in the first place. */
    {
        int old_n = g.sess.pc.n, old_have_pc = g.sess.pc.have;
        int old_bytes = g.sess.pc.bytes;
        g.mdl.have_state = 0;
        g.sess.pc.n = 7;
        g.sess.pc.bytes = 99;
        g.sess.pc.have = 1;
        g.set.ctx = 4096;
        rc = ctx_commit(hwnd, 2048, err, (int)sizeof err);
        st_check(f, rc == 0 && g.sess.pc.n == 0 && g.sess.pc.have == 0,
                 "ctx: a successful resize forgets the cached prefix");
        if (rc != 0 || g.sess.pc.n != 0 || g.sess.pc.have != 0)
            fprintf(f, "  rc %d pc.n %d pc.have %d\n", rc, g.sess.pc.n,
                    g.sess.pc.have);
        checks++;
        g.sess.pc.n = old_n;
        g.sess.pc.bytes = old_bytes;
        g.sess.pc.have = old_have_pc;
        g.mdl.have_state = 1;
    }

    g.set.ctx = old_ctx;
    g.mdl.have_model = old_have_model;
    g.mdl.have_tok = 0;
    g.mdl.have_state = old_have_state;
    g.mdl.seq_len = old_seq;
    g.mdl.seq_want = old_want;
    lz_ini_set_int("ctx", old_ini);
    set_ctx_cell();
    return checks;
}

/* ---- the temperature scrollbar, mouse half included ----

   The temperature scrollbar (temperature -> horizontal scrollbar +
   edit box, gui/settingsdlg.c) is built the exact way st_settings_dialog
   above is - lz_gui_settings_dialog_create only, no ShowWindow, no
   modal loop, GetDlgItem/SendMessage/DestroyWindow. That split exists
   specifically so a selftest can do this; settingsdlg.h says so.

   WHAT "responds to the mouse" COVERS, AND WHAT IT DOES NOT. dlgproc's
   WM_HSCROLL handler has no path back into the scrollbar control's own
   hit-testing - on a real click, deciding WHICH notification a given
   pixel corresponds to happens entirely inside USER32's built-in
   SCROLLBAR window procedure, which then sends WM_HSCROLL to the
   PARENT as its own responsibility. dlgproc only ever reacts to that
   notification; it has no way to tell a synthesized one from a real
   one. So SendMessage(dlg, WM_HSCROLL, MAKEWPARAM(code, hiword),
   (LPARAM)sb) below exercises exactly the code this file owns. What it
   does NOT prove is that a given screen pixel really produces the
   notification code these checks assume - that hit-testing is platform
   code nothing here touches, the same trust boundary this file already
   extends to BM_GETCHECK / EM_GETSEL / every other stock control
   message elsewhere in this selftest.

   OS DEFAULTS BELOW ARE MEASURED, NOT ASSUMED (throwaway probe, not
   part of this tree): a freshly-created SB_CTL
   SCROLLBAR, before any SetScrollRange/SetScrollPos call, reports
   GetScrollRange lo=0 hi=0 and GetScrollPos 0. So "the range really is
   0..100" and "the position really is 50" are not things a deleted
   SetScrollRange/SetScrollPos call could produce by coincidence - they
   are strong checks on their own. The mutations below exist for the
   ordinary iron-law-four reason (prove each check is wired to the
   right call, not that it escapes a lucky default). */
static int st_settings_scroll(FILE *f, HINSTANCE inst) {
    LZGuiSettings in;
    HWND dlg, sb, e;
    int checks = 0;
    int lo = -1, hi = -1, pos;
    char buf[64];

    lz_common_settings_init(&in);
    lz_common_settings_set_temp(&in, 0.5f);   /* -> scroll position 50 */

    dlg = lz_gui_settings_dialog_create(NULL, inst, &in);
    st_check(f, dlg != NULL, "scroll: the dialog opens"); checks++;
    if (!dlg) return checks;

    sb = GetDlgItem(dlg, 3004 /* ID_TEMP_SCROLL */);
    st_check(f, sb != NULL, "scroll: the control exists"); checks++;
    if (!sb) { DestroyWindow(dlg); return checks; }
    e = GetDlgItem(dlg, 3003 /* ID_TEMP */);

    GetScrollRange(sb, SB_CTL, &lo, &hi);
    st_check(f, lo == 0 && hi == 100,
             "scroll: the range is 0..100 (a fresh SCROLLBAR's own "
             "default is 0..0, measured - this is not that default)");
    checks++;

    pos = GetScrollPos(sb, SB_CTL);
    buf[0] = '\0';
    if (e) GetWindowTextA(e, buf, (int)sizeof buf);
    st_check(f, pos == 50 && strcmp(buf, "0.50") == 0,
             "scroll: the initial position and the text box agree with "
             "the temperature the dialog opened with (0.50 -> 50, "
             "against a fresh SCROLLBAR's own default position of 0, "
             "measured)");
    checks++;

    /* Position-correctness, one row per notification family. Each
       hardcodes the position an untampered dlgproc must reach from a
       known start, so a wrong delta or a missing case arm turns
       exactly ONE row red - none of them read the text box, which is
       what keeps a defect here from being indistinguishable from the
       separate sync check below.
       SB_THUMBPOSITION is not a separate row: it is literally the same
       switch arm as SB_THUMBTRACK (`case SB_THUMBTRACK: case
       SB_THUMBPOSITION:` fall through together), so a second row would
       exercise the identical line, not additional coverage. That row
       is POSITIVE-ONLY, no mutation, per team-lead's correction: at
       range 100 a position never exceeds 100, so (short) casting
       HIWORD changes nothing observable here - it is the correct read
       of WM_HSCROLL's contract for when the range grows, not a guard
       against anything reachable today. */
    {
        typedef struct {
            int start;
            WORD code, hiword;
            int want;
            const char *what;
        } ScrollCase;
        static const ScrollCase CASES[] = {
            { 50, SB_LINERIGHT,  0, 51,
              "scroll: SB_LINERIGHT advances the position by 1" },
            { 50, SB_LINELEFT,   0, 49,
              "scroll: SB_LINELEFT retreats the position by 1" },
            { 50, SB_PAGERIGHT,  0, 60,
              "scroll: SB_PAGERIGHT advances the position by 10" },
            { 50, SB_PAGELEFT,   0, 40,
              "scroll: SB_PAGELEFT retreats the position by 10" },
            { 50, SB_TOP,        0, 0,
              "scroll: SB_TOP (Home) jumps to the low end" },
            { 50, SB_BOTTOM,     0, 100,
              "scroll: SB_BOTTOM (End) jumps to the high end" },
            { 50, SB_THUMBTRACK, 73, 73,
              "scroll: SB_THUMBTRACK reads the position from HIWORD "
              "(not mutation-verified - see the comment above)" },
        };
        int i;
        for (i = 0; i < (int)(sizeof CASES / sizeof CASES[0]); i++) {
            SetScrollPos(sb, SB_CTL, CASES[i].start, TRUE);
            SendMessage(dlg, WM_HSCROLL,
                        MAKEWPARAM(CASES[i].code, CASES[i].hiword),
                        (LPARAM)sb);
            st_check(f, GetScrollPos(sb, SB_CTL) == CASES[i].want,
                     CASES[i].what);
            checks++;
        }
    }

    /* Text-box sync, deliberately separate from the table above and
       computed from whatever GetScrollPos ACTUALLY reads rather than a
       hardcoded target. Deleting the handler's SetWindowTextA call
       desyncs the two even though the position itself stays correct;
       a wrong delta moves the position without desyncing it from
       whatever (also wrong, but consistent) text got written for it.
       Reading the live position rather than asserting "51" here is
       what keeps this check from going red for the SAME reason a
       position-table mutation would - the two mutations are
       independent, and this is what keeps what they catch independent
       too. */
    {
        float want_t;
        char want_buf[32];
        SetScrollPos(sb, SB_CTL, 50, TRUE);
        SendMessage(dlg, WM_HSCROLL, MAKEWPARAM(SB_LINERIGHT, 0),
                   (LPARAM)sb);
        pos = GetScrollPos(sb, SB_CTL);
        want_t = lz_gui_scroll_to_temp(pos);
        lz_gui_format_temp(want_t, want_buf, (int)sizeof want_buf);
        buf[0] = '\0';
        if (e) GetWindowTextA(e, buf, (int)sizeof buf);
        st_check(f, strcmp(buf, want_buf) == 0,
                 "scroll: the text box tracks whatever position a "
                 "scroll message actually produced");
        checks++;
    }

    /* Low-end boundary: SB_LINELEFT from position 0 must not go
       negative. NOT mutation-verified, and not for the SB_THUMBTRACK
       reason (never reachable) - the opposite one: checked empirically
       (a second throwaway probe, not in this tree) that deleting
       dlgproc's own `if (pos < 0) pos = 0` changes nothing observable,
       because BOTH downstream users of the value already clamp it
       themselves. SetScrollPos(sb, SB_CTL, -1, TRUE) followed by
       GetScrollPos reads back 0, not -1 - Win32 re-clamps to the live
       SetScrollRange internally. And lz_gui_scroll_to_temp has its own
       `if (pos < 0) pos = 0` before its division, so the text box
       would read "0.00" either way. This is the mirror case of the
       temperature->scroll direction's NaN guard (that one is NOT
       redundant: nothing downstream can rescue a value that already
       went through undefined behaviour on the way in). Kept anyway,
       same reasoning lz_gui_format_temp's own comment gives for its
       unreachable guard - a function should not depend on its own
       downstream calls having been read closely enough to rely on. */
    SetScrollPos(sb, SB_CTL, 0, TRUE);
    SendMessage(dlg, WM_HSCROLL, MAKEWPARAM(SB_LINELEFT, 0), (LPARAM)sb);
    st_check(f, GetScrollPos(sb, SB_CTL) == 0,
             "scroll: the position does not go negative at the low end "
             "(not mutation-verified - see the comment above)");
    checks++;

    DestroyWindow(dlg);
    return checks;
}

/* ---- the About window ----
 *
 * NO gate can tell "looks like Word 95's about box" apart from "does
 * not" - the appearance comparison
 * is a screenshot next to the reference for the user. What a gate CAN
 * cover is the four things that are actual failures rather than taste:
 * the window builds, the three controls are all there, the body text and
 * caption come from the string table (not a copy typed here), and Esc /
 * OK / the close box all dismiss it and leave nothing leaked.
 *
 * The same create-without-loop split as st_settings_dialog: build the
 * window, read it, drive the dismiss messages, assert IsWindow goes
 * false. A window that lingers is a leak that shows up as nothing else
 * in a selftest. */
static int st_about(FILE *f, HINSTANCE inst) {
    HWND dlg;
    int checks = 0;

    dlg = lz_gui_about_create(NULL, inst);
    st_check(f, dlg != NULL, "about: the window is created"); checks++;
    if (!dlg) return checks;

    /* All the controls exist. The icon control is a STATIC with SS_ICON
       carrying IDI_APP; the OK button, the four text controls and the
       divider all live. */
    st_check(f, GetDlgItem(dlg, 3102 /* ID_ABT_ICON */) != NULL,
             "about: the icon control exists"); checks++;
    st_check(f, GetDlgItem(dlg, 3101 /* ID_ABT_OK */) != NULL,
             "about: the OK button exists"); checks++;
    /* The divider is NOT a control - it is drawn in WM_PAINT with
       lz_draw_edge, because the SS_* separator styles are Windows 95
       additions the 3.51 floor hides (aboutdlg.c's own comment). So the
       "controls are all there" check covers the icon, the OK button and
       the four text controls, and the divider's PRESENCE is what the
       WM_PRINTCLIENT check below asserts - a window that refuses to
       paint its own divider would render with it missing. */

    /* The caption and body come from the string table, not re-typed in
       aboutdlg.c. A dialog that typed its own product name is two
       authorities for the same word (see aboutdlg.h). */
    {
        char cap[64];
        GetWindowTextA(dlg, cap, (int)sizeof cap);
        st_check(f, strcmp(cap, lz_str_display(LZ_STR_DLG_ABOUT_TITLE)) == 0,
                 "about: the caption comes from the string table");
        if (strcmp(cap, lz_str_display(LZ_STR_DLG_ABOUT_TITLE)) != 0)
            fprintf(f, "  caption [%s]\n", cap);
        checks++;

        /* The four text controls split the SAME LZ_STR_ABOUT_BODY the
           string table owns - the title is its first line, the version
           its second, the description lines 4-6, the credits the last
           line. The split lives in aboutdlg.c, so the checks assert
           each control holds the corresponding SLICE of the live
           string, not a re-typed copy. */
        {
            const char *body = lz_str_display(LZ_STR_ABOUT_BODY);
            const char *nl1 = strchr(body, '\n');
            const char *nl2 = nl1 ? strchr(nl1 + 1, '\n') : NULL;
            char want[512];   /* the license string is ~470 bytes, not the
                                 body's ~70 - sized for the longest it holds */
            HWND b;
            char got[512];
            b = GetDlgItem(dlg, 3103 /* ID_ABT_BODY */);
            got[0] = '\0';
            if (b) GetWindowTextA(b, got, (int)sizeof got);
            if (nl1) {
                int n = (int)(nl1 - body);
                if (n > (int)sizeof want - 1) n = (int)sizeof want - 1;
                memcpy(want, body, (size_t)n);
                want[n] = '\0';
            } else { lstrcpynA(want, body, (int)sizeof want); }
            st_check(f, strcmp(got, want) == 0,
                     "about: the title is the body's first line");
            if (strcmp(got, want) != 0) fprintf(f, "  title [%s] want [%s]\n",
                                                got, want);
            checks++;

            /* The title font: it must be a REAL font answering
               WM_GETFONT, and it must match the UI font's charset (SimSun
               GB2312 in Chinese, whatever stock face in English) - NOT
               the FIXEDFONT a title degrades to when its font handle
               is freed right after WM_SETFONT. Comparing against the LIVE
               UI font's charset
               rather than a hardcoded 134 keeps this true across both
               languages. */
            {
                HFONT tf = (HFONT)SendMessage(b, WM_GETFONT, 0, 0);
                LOGFONTA lf, uf;
                int ui_charset;
                if (!tf || !GetObjectA((HGDIOBJ)tf, (int)sizeof lf, &lf)) {
                    st_check(f, 0, "about: the title control answers "
                             "WM_GETFONT with a real font");
                } else {
                    ui_charset = DEFAULT_CHARSET;
                    if (lz_ui_font() &&
                        GetObjectA((HGDIOBJ)lz_ui_font(),
                                   (int)sizeof uf, &uf))
                        ui_charset = uf.lfCharSet;
                    st_check(f, lf.lfCharSet == ui_charset ||
                                lf.lfCharSet == DEFAULT_CHARSET,
                             "about: the title font matches the UI font's "
                             "charset, not FIXEDFONT");
                    fprintf(f, "  title charset %d (UI %d)\n", lf.lfCharSet,
                            ui_charset);
                }
                checks++;
            }

            /* version = second line */
            b = GetDlgItem(dlg, 3104 /* ID_ABT_VER */);
            got[0] = '\0';
            if (b) GetWindowTextA(b, got, (int)sizeof got);
            if (nl2 && nl1) {
                int n = (int)(nl2 - nl1 - 1);
                if (n > (int)sizeof want - 1) n = (int)sizeof want - 1;
                memcpy(want, nl1 + 1, (size_t)n);
                want[n] = '\0';
            } else { want[0] = '\0'; }
            st_check(f, strcmp(got, want) == 0,
                     "about: the version is the body's second line");
            if (strcmp(got, want) != 0)
                fprintf(f, "  version [%s] want [%s]\n", got, want);
            checks++;

            /* license = LZ_STR_ABOUT_LICENSE, its own string, NOT the
               body's last line - the multi-line Apache-2.0
               boilerplate split off from the body when it grew a full
               copyright notice + Small Fonts rendering. */
            b = GetDlgItem(dlg, 3106 /* ID_ABT_CRED */);
            got[0] = '\0';
            if (b) GetWindowTextA(b, got, (int)sizeof got);
            lstrcpynA(want, lz_str_display(LZ_STR_ABOUT_LICENSE),
                      (int)sizeof want);
            st_check(f, strcmp(got, want) == 0,
                     "about: the license line is the license string");
            if (strcmp(got, want) != 0)
                fprintf(f, "  license [%s] want [%s]\n", got, want);
            checks++;
        }
    }

    /* The OK button carries the table's OK label. */
    {
        char ok[64];
        GetWindowTextA(GetDlgItem(dlg, 3101 /* ID_ABT_OK */), ok,
                       (int)sizeof ok);
        st_check(f, strcmp(ok, lz_str_display(LZ_STR_BTN_OK)) == 0,
                 "about: the OK label comes from the string table");
        checks++;
    }

    /* The System Info button, right of OK - same table
       wording rule. It must be a real control that carries the button
       label. */
    {
        char si[64];
        HWND b = GetDlgItem(dlg, 3108 /* ID_ABT_SYSINFO */);
        st_check(f, b != NULL, "about: the System Info button exists");
        checks++;
        si[0] = '\0';
        if (b) GetWindowTextA(b, si, (int)sizeof si);
        st_check(f, strcmp(si, lz_str_display(LZ_STR_BTN_SYSINFO)) == 0,
                 "about: the System Info label comes from the string "
                 "table");
        if (strcmp(si, lz_str_display(LZ_STR_BTN_SYSINFO)) != 0)
            fprintf(f, "  sysinfo [%s]\n", si);
        checks++;
    }

    /* The divider is drawn in WM_PAINT/WM_PRINTCLIENT (aboutdlg.c's
       own comment). Render the window into a memory DC and read the
       pixels back - this is the DETERMINISTIC way to see where the
       divider actually landed, unlike a screen grab (DPI scaling) or
       asking the user. The two-row groove is light (160) over dark
       (100), the reference's own colours, spanning the text column. */
    {
        HDC hdc = GetDC(dlg);
        HDC mdc = CreateCompatibleDC(hdc);
        if (mdc) {
            RECT cr;
            HBITMAP bmp;
            GetClientRect(dlg, &cr);
            bmp = CreateCompatibleBitmap(hdc, cr.right, cr.bottom);
            if (bmp) {
                HGDIOBJ old = SelectObject(mdc, bmp);
                /* WM_PRINTCLIENT paints controls but not the client
                   background, so a fresh bitmap comes out black. Fill
                   the dialog's own background colour first, exactly as
                   WM_ERASEBKGND would. */
                {
                    HBRUSH br = (HBRUSH)(COLOR_BTNFACE + 1);
                    RECT fill;
                    fill.left = 0; fill.top = 0;
                    fill.right = cr.right; fill.bottom = cr.bottom;
                    FillRect(mdc, &fill, br);
                }
                SendMessage(dlg, WM_PRINTCLIENT, (WPARAM)mdc, 0);
                {
                    /* The divider's real position, read back from the
                       rendered pixels - not guessed from a screen. It
                       must be a light-over-dark two-row groove spanning
                       the FULL client width (the reference draws it
                       x9..371, below the logo), so find the widest run
                       of the light-pen colour and assert it reaches
                       from near the left margin to near the right one.
                       DETERMINISTIC: client coordinates, no DPI, no
                       screen grab. */
                    int y, x;
                    int best_w = 0, best_y = 0, best_x0 = 0;
                    for (y = 0; y < cr.bottom; y++) {
                        int run = 0, run_start = 0;
                        for (x = 0; x < cr.right; x++) {
                            COLORREF c = GetPixel(mdc, x, y);
                            int r = GetRValue(c), g = GetGValue(c), b = GetBValue(c);
                            int is_light = (r == g && g == b && r >= 150 && r <= 170);
                            if (is_light) {
                                if (run == 0) run_start = x;
                                run++;
                            } else {
                                if (run > best_w) {
                                    best_w = run; best_y = y; best_x0 = run_start;
                                }
                                run = 0;
                            }
                        }
                        if (run > best_w) {
                            best_w = run; best_y = y; best_x0 = run_start;
                        }
                    }
                    fprintf(f, "  divider: %dpx at y=%d x%d..%d (client %ldx%ld)\n",
                            best_w, best_y, best_x0, best_x0 + best_w,
                            (long)cr.right, (long)cr.bottom);
                    /* Full-width: from the left margin to the right
                       margin, within a margin of tolerance. The logo
                       frame's own top edge is also a light run, but it
                       is narrower (67px) than the divider (the full
                       width minus two margins), so the WIDEST run is
                       the divider. */
                    st_check(f, best_w >= cr.right - 2 * LZ_GUI_DLG_MARGIN - 4,
                             "about: the divider spans the full client "
                             "width below the logo");
                    checks++;
                }
                SelectObject(mdc, old);
                DeleteObject(bmp);
            }
            DeleteDC(mdc);
        }
        if (hdc) ReleaseDC(dlg, hdc);
    }

    /* Three ways out, one check each: OK, Esc (IDCANCEL via the modal
       loop's IsDialogMessage), and WM_CLOSE. All three must destroy the
       window - a lingering window is a leak no other check sees. */
    SendMessage(dlg, WM_COMMAND, MAKEWPARAM(3101 /* ID_ABT_OK */, 0), 0);
    st_check(f, !IsWindow(dlg), "about: OK destroys the window");
    checks++;

    dlg = lz_gui_about_create(NULL, inst);
    st_check(f, dlg != NULL, "about: the window can be rebuilt"); checks++;
    if (dlg) {
        SendMessage(dlg, WM_COMMAND, MAKEWPARAM(IDCANCEL, 0), 0);
        st_check(f, !IsWindow(dlg), "about: Esc destroys the window");
        checks++;
    }

    dlg = lz_gui_about_create(NULL, inst);
    if (dlg) {
        SendMessage(dlg, WM_CLOSE, 0, 0);
        st_check(f, !IsWindow(dlg), "about: the close box destroys the "
                 "window");
        checks++;
    }

    /* ---- the System Info dialog ----
       Build it without its modal loop (the same create-without-loop
       split the about window uses), check the window and its OK exist,
       and that it carries real fact lines - a dialog whose GetVersionExA
       / GetSystemInfo / GlobalMemoryStatus / GetDiskFreeSpaceA calls all
       failed would show only empty or '?' text, which is a mechanism
       failure no control-existence check can see. Then OK must destroy
       it. */
    {
        HWND si = lz_gui_sysinfo_create(NULL, inst);
        st_check(f, si != NULL, "sysinfo: the window is created");
        checks++;
        if (si) {
            st_check(f, GetDlgItem(si, 3201 /* ID_SYS_OK */) != NULL,
                     "sysinfo: the OK button exists");
            checks++;
            {
                /* The OS line (3202) is the first fact. It must carry
                   real text - "Windows: ..." (or the Chinese label) -
                   not the '?' a failed GetVersionExA would leave. The
                   '?' fallback exists (sys_os in aboutdlg.c); a window
                   that shows four '?'s is a mechanism failure no
                   control-existence check can see. */
                char txt[320];
                txt[0] = '\0';
                GetWindowTextA(GetDlgItem(si, 3202 /* ID_SYS_OS */), txt,
                               (int)sizeof txt);
                st_check(f, txt[0] && strcmp(txt, "?") != 0,
                         "sysinfo: the OS line carries real text");
                if (txt[0] && strcmp(txt, "?") != 0)
                    fprintf(f, "  os line [%s]\n", txt);
                checks++;
            }
            {
                /* The CPU line (3203). Word 95's own matching (from the
                   decompile, case 1) names 386/486/Pentium and falls to
                   an "other" string for anything else - it does NOT
                   print a question mark. The check asserts the line
                   carries the CPU label and a non-'?' value, so a
                   regression back to the '?'-for-unknown shape is
                   caught. */
                char txt[320];
                txt[0] = '\0';
                GetWindowTextA(GetDlgItem(si, 3203 /* ID_SYS_CPU */), txt,
                               (int)sizeof txt);
                st_check(f, txt[0] && strcmp(txt, "?") != 0,
                         "sysinfo: the CPU line carries a value, not '?'");
                if (txt[0]) fprintf(f, "  cpu line [%s]\n", txt);
                checks++;
            }
            SendMessage(si, WM_COMMAND, MAKEWPARAM(3201 /* ID_SYS_OK */, 0),
                        0);
            st_check(f, !IsWindow(si), "sysinfo: OK destroys the window");
            checks++;
        }
    }

    return checks;
}

/* ---- status-strip pixel capture, for comparing the simulated bar
 * against comctl32's.
 *
 * Both strips answer WM_PRINTCLIENT - the comctl32 one through
 * sb_bevel_proc, the fallback through sb_fallback_proc - so both render
 * into a caller-supplied DC and are captured the same way.
 *
 * TWO RUNS, not one. classic_ui makes the same binary on the same
 * machine produce the other strip, which already holds the font, the DPI
 * and the system colours constant; building a second main window inside
 * one run would complicate the selftest's window lifetime and control
 * nothing further. Each run writes its own image and, when the
 * counterpart from the other run is already there, compares.
 *
 * PIXEL IDENTITY IS NOT THE CRITERION AND MUST NOT BECOME ONE. The
 * fallback is a reimplementation, not a copy of comctl32; demanding
 * identity would mean reimplementing comctl32. Only what must hold is
 * asserted - the capture worked, the file was written, both strips span
 * the same width. Height, cell boundaries and the differing-byte count
 * are REPORTED, because a difference there is a fact to look at rather
 * than a failure, and the images are written so the appearance can be
 * judged by looking at it. */
#ifndef PRF_CLIENT
#define PRF_CLIENT 0x00000004L
#endif
#ifndef PRF_CHILDREN
#define PRF_CHILDREN 0x00000010L
#endif

static long sb_stride(int w) { return (long)(((w * 3) + 3) & ~3); }

static unsigned char *sb_capture(HWND strip, int *w, int *h) {
    RECT rc;
    HDC wdc, mdc;
    BITMAPINFO bi;
    void *bits = NULL;
    HBITMAP dib, old;
    unsigned char *copy = NULL;
    long n;

    GetClientRect(strip, &rc);
    *w = rc.right - rc.left;
    *h = rc.bottom - rc.top;
    if (*w <= 0 || *h <= 0) return NULL;
    wdc = GetDC(strip);
    if (!wdc) return NULL;
    mdc = CreateCompatibleDC(wdc);
    memset(&bi, 0, sizeof bi);
    bi.bmiHeader.biSize        = sizeof bi.bmiHeader;
    bi.bmiHeader.biWidth       = *w;
    bi.bmiHeader.biHeight      = *h;
    bi.bmiHeader.biPlanes      = 1;
    bi.bmiHeader.biBitCount    = 24;
    bi.bmiHeader.biCompression = BI_RGB;
    dib = CreateDIBSection(wdc, &bi, DIB_RGB_COLORS, &bits, NULL, 0);
    ReleaseDC(strip, wdc);
    if (!dib || !bits) { if (dib) DeleteObject(dib); DeleteDC(mdc); return NULL; }
    old = (HBITMAP)SelectObject(mdc, dib);
    /* Pre-filled with the button face: a DIB section starts as zeros,
       i.e. black, and comctl32's status bar leaves its topmost rows
       unpainted under WM_PRINTCLIENT - those rows would otherwise read
       as a difference neither strip draws. */
    {
        RECT fr;
        fr.left = 0; fr.top = 0; fr.right = *w; fr.bottom = *h;
        FillRect(mdc, &fr, sb_brush(COLOR_BTNFACE));
    }
    /* PRF_CHILDREN as well as PRF_CLIENT: the lamps are child
       controls, so a client-only render leaves them out and the
       comparison would be blind to exactly the cell they sit in. */
    SendMessage(strip, WM_PRINTCLIENT, (WPARAM)mdc,
                PRF_CLIENT | PRF_CHILDREN);
    GdiFlush();
    n = sb_stride(*w) * (*h);
    copy = (unsigned char *)malloc((size_t)n);
    if (copy) memcpy(copy, bits, (size_t)n);
    SelectObject(mdc, old);
    DeleteObject(dib);
    DeleteDC(mdc);
    return copy;
}

static int sb_bmp_write(const char *path, const unsigned char *bits,
                        int w, int h) {
    BITMAPFILEHEADER fh;
    BITMAPINFOHEADER ih;
    long n = sb_stride(w) * h;
    FILE *o = fopen(path, "wb");
    int ok;
    if (!o) return 0;
    memset(&fh, 0, sizeof fh);
    memset(&ih, 0, sizeof ih);
    fh.bfType    = 0x4D42;                       /* "BM" */
    fh.bfOffBits = sizeof fh + sizeof ih;
    fh.bfSize    = fh.bfOffBits + (DWORD)n;
    ih.biSize    = sizeof ih;
    ih.biWidth   = w;
    ih.biHeight  = h;
    ih.biPlanes  = 1;
    ih.biBitCount = 24;
    ih.biCompression = BI_RGB;
    ih.biSizeImage   = (DWORD)n;
    ok = fwrite(&fh, sizeof fh, 1, o) == 1 &&
         fwrite(&ih, sizeof ih, 1, o) == 1 &&
         fwrite(bits, 1, (size_t)n, o) == (size_t)n;
    fclose(o);
    return ok;
}

static unsigned char *sb_bmp_load(const char *path, int *w, int *h) {
    BITMAPFILEHEADER fh;
    BITMAPINFOHEADER ih;
    unsigned char *bits;
    long n;
    FILE *i = fopen(path, "rb");
    if (!i) return NULL;
    if (fread(&fh, sizeof fh, 1, i) != 1 || fread(&ih, sizeof ih, 1, i) != 1 ||
        ih.biBitCount != 24 || ih.biWidth <= 0 || ih.biHeight <= 0) {
        fclose(i); return NULL;
    }
    *w = (int)ih.biWidth;
    *h = (int)ih.biHeight;
    n = sb_stride(*w) * (*h);
    bits = (unsigned char *)malloc((size_t)n);
    if (!bits) { fclose(i); return NULL; }
    if (fseek(i, (long)fh.bfOffBits, SEEK_SET) != 0 ||
        fread(bits, 1, (size_t)n, i) != (size_t)n) {
        free(bits); fclose(i); return NULL;
    }
    fclose(i);
    return bits;
}

static void sb_compare(FILE *f, const char *path, int *checks);

static int selftest(HINSTANCE inst, const char *path) {
    static const int SIZES[][2] = { { 640, 480 }, { 900, 700 },
                                    { LZ_GUI_MIN_CW, LZ_GUI_MIN_CH } };
    FILE *f = fopen(path, "wb");
    HWND hwnd;
    int s, i, checks = 0;

    if (!f) return 2;
    if (!register_class(inst)) { fprintf(f, "FAIL register_class\n");
                                 fclose(f); return 1; }
    hwnd = create_main(inst, 640, 480);
    if (!hwnd) { fprintf(f, "FAIL create_main\n"); fclose(f); return 1; }

    /* LZ_GUI_SPLIT is a band of the parent's background that the window
       hit-tests, not a control - checked below by comparing g.split
       against the layout instead. LZ_GUI_SIDE_LAMPS/SIDE_CAND
       do not exist yet either, for a different reason: no
       model is loaded this early, and side_panel_sync is what creates
       them, later, once one is (st_model/st_real_model, further down,
       are what exercise that side). All three are named here rather
       than skipped silently. */
    for (i = 0; i < LZ_GUI_PART_COUNT; i++) {
        char what[64];
        if (i == LZ_GUI_SPLIT || i == LZ_GUI_SIDE_LAMPS ||
            i == LZ_GUI_SIDE_CAND)
            continue;
        sprintf(what, "control exists: %s", lz_gui_part_name((LZGuiPart)i));
        st_check(f, g.part[i] != NULL && IsWindow(g.part[i]), what);
        checks++;
    }
    st_check(f, g.part[LZ_GUI_SPLIT] == NULL,
             "control exists: split is deliberately not a window");
    checks++;
    st_check(f, g.part[LZ_GUI_SIDE_LAMPS] == NULL &&
             g.part[LZ_GUI_SIDE_CAND] == NULL,
             "control exists: no model loaded yet, so the candidate "
             "panel does not exist yet either"); checks++;

    /* The strip is captured HERE, not at the end: this is the one point
       where the window is in its just-built state at a known size, so
       the two runs compare like with like. Later the selftest walks the
       degradation branches deliberately, and what the strip looks like
       after that is not what either system shows a user. */
    relayout(hwnd);
    sb_compare(f, path, &checks);

    /* Two invariants about the REAL child list. g.part[] alone is not
       enough: a control created twice leaves the first instance orphaned
       at 0x0, and every check that looks at g.part[] still passes. The
       orphan keeps its id and stays enabled while the visible pair is
       disabled, and - the part a user would actually hit - takes two
       slots in the tab chain the visible pair does not, so Tab from
       the input box makes the focus vanish twice. */
    {
        HWND kids[64];
        int n = st_children(hwnd, kids, (int)(sizeof kids / sizeof kids[0]));
        int a, b, dup = 0, ghost = 0;
        for (a = 0; a < n; a++) {
            RECT r;
            for (b = a + 1; b < n; b++)
                if (GetDlgCtrlID(kids[a]) == GetDlgCtrlID(kids[b])) dup++;
            GetClientRect(kids[a], &r);
            if ((GetWindowLong(kids[a], GWL_STYLE) & WS_TABSTOP)
                && (r.right <= 0 || r.bottom <= 0))
                ghost++;
        }
        /* -3, not -1: SPLIT never has a window, and neither do
           LZ_GUI_SIDE_LAMPS (never - see its own comment) or
           LZ_GUI_SIDE_CAND (not yet - no model is loaded at this
           point in the selftest). */
        st_check(f, n >= LZ_GUI_PART_COUNT - 3,   /* SPLIT, LAMPS, CAND */
                 "children: the walk found at least as many as g.part[]");
        checks++;
        st_check(f, dup == 0,
                 "children: no two share a control id"); checks++;
        st_check(f, ghost == 0,
                 "children: no zero-sized tab stop"); checks++;
    }

    /* The drag-drop wiring, not the pure logic underneath it (that is
       lz_drop_dir_of's 8 cases). "Cannot really drag" rules out
       exercising WM_DROPFILES end to end, but WM_CREATE's call to
       lz_drop_accept can be checked without one: g.drop_on is what it
       returned there, and calling lz_drop_accept(hwnd, 1) again here is
       side-effect-free (DragAcceptFiles is just a flag on the window,
       idempotent to set twice) and gives an independent second answer
       to compare it against.

       Deliberately NOT "g.drop_on == 1": that would assume this host's
       shell32 has the export, which is true on every machine this runs
       on today but is not what the wiring is actually responsible for.
       Comparing against a fresh call is true whether or not the export
       exists - on NT 3.51 both calls come back 0 and this still passes -
       and is exactly what goes false if WM_CREATE's own call is ever
       deleted: g.drop_on would stay at its struct-zero-init 0 while the
       fresh call here still answers whatever this host really has. */
    st_check(f, g.drop_on == lz_drop_accept(hwnd, 1),
             "drop: WM_CREATE's lz_drop_accept call really ran, not "
             "just declared"); checks++;

    /* create_children's OWN rollback_sync, and it has to be checked
       HERE, before anything else runs.
       The strip is created after the menu bar, and a fresh strip has
       every button live, so create_children re-applies the state at
       the end. Measured: deleting that call reddens nothing at all if
       the check is placed down in st_rollback, because every
       build_menu_bar in between - and the selftest triggers several -
       ends in the same rollback_sync and quietly repairs the state
       before it can be observed. In the shipped program nothing
       repairs it until the user changes language or loads a model, so
       the window opens with a live Retry button that does nothing.
       An assertion whose subject is fixed before it looks is not an
       assertion; this one looks first. */
    {
        int btn = lz_gui_toolbar_enabled(g.part[LZ_GUI_TOOLBAR], IDM_REGEN);
        st_check(f, btn == -1 || btn == (cmd_is_enabled(IDM_REGEN) != 0),
                 "rollback: at creation, the Retry button already agrees "
                 "with its menu item");
        if (btn != -1)
            fprintf(f, "  button %d menu %d\n", btn,
                    cmd_is_enabled(IDM_REGEN) != 0);
        checks++;
    }

    /* The transcript's own uxtheme opt-out, same wiring
       shape as lz_drop_accept just above: g.transcript_untheme_ok is
       what create_children's call returned, and calling
       lz_ui_untheme(g.part[LZ_GUI_TRANSCRIPT]) again here is side-
       effect-free (SetWindowTheme("","") is idempotent, same as
       DragAcceptFiles) and gives an independent second answer to
       compare it against - proof the call in create_children really
       ran, not just that the source line exists. */
    st_check(f, g.transcript_untheme_ok ==
                lz_ui_untheme(g.part[LZ_GUI_TRANSCRIPT]),
             "theme: the transcript's untheme call really ran, not "
             "just declared"); checks++;

    /* THE actual claim this task fixed: RichEdit's scrollbar is not
       exempt from uxtheme the way its border is - the transcript has a
       theme to turn off. If it did not, SetWindowTheme would have had
       nothing to turn off on the transcript and could plausibly fail or
       no-op differently than it does on the input box's ordinary EDIT.
       It does not: on any host, uxtheme absent or present, both calls
       report the same outcome, because SetWindowTheme is not RichEdit-
       specific - proven here rather than assumed, on whichever this
       host happens to be (uxtheme.dll is XP+; this repo's own floor is
       NT 3.51, so both sides of the comparison have to hold on a host
       with neither, same reasoning as the drop_on check above). */
    st_check(f, g.transcript_untheme_ok == g.input_untheme_ok,
             "theme: SetWindowTheme reaches the transcript's scrollbar "
             "exactly as it reaches the input box's border");
    checks++;

    /* Three lines really fit in the smallest input box.
     *
     * LZ_GUI_INPUT_MIN_H is a constant derived from SimSun 9pt, which is
     * the target's font and not this machine's. Asking the font that is
     * actually loaded turns "46 was right on the box it was measured on"
     * into "46 is right here too, or this goes red" - and a minimum that
     * silently holds two and a half lines is exactly the kind of thing
     * nobody reports as a bug. */
    {
        HDC dc = GetDC(hwnd);
        TEXTMETRICA tm;
        int pitch = 0;
        if (dc) {
            HGDIOBJ old = SelectObject(dc, lz_ui_font());
            GetTextMetricsA(dc, &tm);
            SelectObject(dc, old);
            ReleaseDC(hwnd, dc);
            pitch = (int)(tm.tmHeight + tm.tmExternalLeading);
        }
        fprintf(f, "  ui font line pitch %d, three lines + edge = %d,"
                   " minimum is %d\n",
                pitch, pitch * 3 + 4, LZ_GUI_INPUT_MIN_H);
        st_check(f, pitch > 0 && pitch * 3 + 4 <= LZ_GUI_INPUT_MIN_H,
                 "input: the smallest box holds three lines of the UI font");
        checks++;
    }

    /* What the user types is what the model gets - through the real
       control, not a re-run of the conversion in the test.
       A multi-line EDIT separates its lines with CRLF, and every step
       after GetWindowText treats the CR as ordinary text: it reaches
       LZChatMsg.content, the rendered prompt, and the saved file, where
       gui/chatfile.c's decoder mistakes it for part of the line separator
       and returns one byte less than it was handed - a byte the training
       corpus does not contain goes into every multi-line prompt.
       THE TEXT PUT IN USES CRLF; a version of this check that used LF
       passed against the broken code too. A
       standard EDIT stores what SetWindowText hands it; it does not
       convert LF to CRLF on the way in. So "0 CR came out" is true
       because no CR ever went in, and the check measures nothing.
       Hence the raw byte count below: it is the proof the instrument is
       connected. 15 bytes go in, 13 come out, and the two that are gone
       are exactly the CRs.
       NOT PINNED TO 15, on purpose: MSDN documents GetWindowTextLength
       as possibly returning a value LARGER than the true length when an
       ANSI app has a common dialog in the picture - and this one does
       (lz_pick_open_file/lz_pick_save_file, gui/compat40.c). A binding
       assertion of "raw_n == 15" would then fail on a machine where the
       control genuinely holds the CRLF text this check exists to set
       up, for a reason that has nothing to do with what is being
       tested - worse than an insensitive check, per iron law four: a
       check that goes red for the wrong reason. What this step actually
       needs to prove is only that the control holds MORE than what the
       CR-stripped conversion below produces - "some bytes were still in
       there to strip" - and raw_n > un says exactly that without
       depending on the exact count. Which of the missing bytes were CR,
       and that none survived, is what the second check below (an exact
       count, an exact CR total, an exact string) actually pins down. */
    {
        static char got[256];
        static const char CRLF_IN[] = "one\r\ntwo\r\nthree";
        char keep[256];
        int un, i, cr = 0, raw_n;
        GetWindowTextA(g.part[LZ_GUI_INPUT], keep, (int)sizeof keep);
        SetWindowTextA(g.part[LZ_GUI_INPUT], CRLF_IN);
        raw_n = GetWindowTextLengthA(g.part[LZ_GUI_INPUT]);
        un = read_input_utf8(got, (int)sizeof got);
        for (i = 0; i < un; i++) if (got[i] == '\r') cr++;
        fprintf(f, "  input holds %d bytes, read back %d, %d of them CR\n",
                raw_n, un, cr);
        st_check(f, raw_n > un,
                 "input: the control really is holding CRLF to begin with");
        checks++;
        st_check(f, un == 13 && cr == 0 &&
                    strcmp(got, "one\ntwo\nthree") == 0,
                 "input: a multi-line message reaches the model without CR");
        checks++;
        SetWindowTextA(g.part[LZ_GUI_INPUT], keep);
    }

    /* Enter's own key bindings: Shift+Enter must insert a
       newline, not send, and a plain Enter's queued WM_CHAR must not
       leave a stray newline behind either - see input_subclass's own
       comment for both mechanisms. Driven IN-PROCESS, directly calling
       input_subclass with SetKeyboardState faking the modifier
       GetKeyState reads inside it: there is no interactive desktop in
       the environment this selftest runs in, and both SendKeys and
       keybd_event were tried against this app from the outside and
       neither delivers a key (build/gate/shot.ps1's own -Chars comment
       hits the same limit from the other direction). This also tests
       the right thing rather than a proxy for it - input_subclass's
       OWN decision, not whatever Windows' real translation machinery
       happens to do underneath it. */
    {
        BYTE keys[256];
        MSG msg;
        HWND h = g.part[LZ_GUI_INPUT];
        int posted;

        memset(keys, 0, sizeof keys);
        SetKeyboardState(keys);   /* known baseline: nothing held */

        /* Plain Enter still sends - unchanged behaviour, re-proven here
           because input_send_key replaces the inline check this used
           to be, and WM_COMMAND is observed via PeekMessage rather than
           a return value: input_subclass's swallow path returns 0
           explicitly, but CallWindowProc's real return for an EDIT
           control's WM_KEYDOWN is typically 0 too, so the return value
           alone cannot tell "posted" apart from "did not". */
        while (PeekMessage(&msg, hwnd, WM_COMMAND, WM_COMMAND, PM_REMOVE)) {}
        input_subclass(h, WM_KEYDOWN, VK_RETURN, 1);
        posted = PeekMessage(&msg, hwnd, WM_COMMAND, WM_COMMAND, PM_REMOVE) &&
                 LOWORD(msg.wParam) == ID_SEND;
        st_check(f, posted, "enter: plain Enter still posts Send");
        checks++;

        /* Ctrl+Enter does not send - the ORIGINAL rule, re-proven with
           the same instrument as Shift+Enter below so a regression in
           input_send_key's Ctrl half would be caught the same way a
           regression in its new Shift half is. */
        keys[VK_CONTROL] = 0x80;
        SetKeyboardState(keys);
        while (PeekMessage(&msg, hwnd, WM_COMMAND, WM_COMMAND, PM_REMOVE)) {}
        input_subclass(h, WM_KEYDOWN, VK_RETURN, 1);
        posted = PeekMessage(&msg, hwnd, WM_COMMAND, WM_COMMAND, PM_REMOVE) &&
                 LOWORD(msg.wParam) == ID_SEND;
        st_check(f, !posted, "enter: Ctrl+Enter does not send"); checks++;
        keys[VK_CONTROL] = 0;
        SetKeyboardState(keys);

        /* Shift+Enter does not send - the fix. Before Shift joined Ctrl
           in input_send_key, this reddened: Shift+Enter posted
           WM_COMMAND/ID_SEND exactly like plain Enter (verified). */
        keys[VK_SHIFT] = 0x80;
        SetKeyboardState(keys);
        while (PeekMessage(&msg, hwnd, WM_COMMAND, WM_COMMAND, PM_REMOVE)) {}
        input_subclass(h, WM_KEYDOWN, VK_RETURN, 1);
        posted = PeekMessage(&msg, hwnd, WM_COMMAND, WM_COMMAND, PM_REMOVE) &&
                 LOWORD(msg.wParam) == ID_SEND;
        st_check(f, !posted, "enter: Shift+Enter does not send"); checks++;
        keys[VK_SHIFT] = 0;
        SetKeyboardState(keys);

        /* The queued-WM_CHAR fix: with no modifier held (the "send"
           case), input_subclass must swallow WM_CHAR('\r') too, or the
           edit control inserts the newline TranslateMessage already
           queued before this window procedure ever saw WM_KEYDOWN -
           swallowing WM_KEYDOWN alone cannot stop a WM_CHAR that was
           queued before it ran. Driven with the control's real text and
           length, not through PostMessage: a swallowed WM_CHAR must
           leave the box exactly as it was. */
        {
            char keep2[256];
            int before, after;
            GetWindowTextA(h, keep2, (int)sizeof keep2);
            SetWindowTextA(h, "hi");
            before = GetWindowTextLengthA(h);
            input_subclass(h, WM_CHAR, '\r', 1);
            after = GetWindowTextLengthA(h);
            st_check(f, after == before,
                     "enter: a plain Enter's queued WM_CHAR is "
                     "swallowed, not inserted as a stray newline");
            checks++;

            /* The paired case: WITH a modifier held, WM_CHAR must NOT
               be swallowed - that is Ctrl+Enter/Shift+Enter's whole
               point, inserting the newline the user asked for. Without
               this pairing, a mutation that swallowed WM_CHAR
               unconditionally (rather than only for the send case)
               would pass the check above and go undetected. */
            keys[VK_SHIFT] = 0x80;
            SetKeyboardState(keys);
            input_subclass(h, WM_CHAR, '\r', 1);
            after = GetWindowTextLengthA(h);
            st_check(f, after > before,
                     "enter: WM_CHAR still reaches the control when a "
                     "modifier is held, so Ctrl/Shift+Enter can still "
                     "insert a newline");
            checks++;
            keys[VK_SHIFT] = 0;
            SetKeyboardState(keys);

            SetWindowTextA(h, keep2);
        }
    }

    /* The lamps: the controls exist and every bitmap loaded.
     *
     * The bitmap half is the one that matters. A new BITMAP can go into
     * gui/kunkun98.rc and not into the review build's copy of it; the
     * symptom is a control that is there, is visible, has the right
     * size - and draws nothing, because STM_SETIMAGE was handed a NULL.
     * Nothing else notices. */
    {
        int i2, missing = 0;
        for (i2 = 0; i2 < LZ_LAMP_KINDS; i2++)
            if (!g.lamp_bmp[i2]) missing++;
        st_check(f, !g.status_is_sbar || (g.lamp[0] && g.lamp[1]),
                 "lamps: both controls exist"); checks++;
        st_check(f, !g.status_is_sbar || missing == 0,
                 "lamps: every state bitmap loaded"); checks++;
        if (missing) fprintf(f, "  %d of %d lamp bitmaps are NULL\n",
                             missing, LZ_LAMP_KINDS);

        /* And that they CHANGE. Everything above passes on a pair of
           lamps welded to grey - which is what they would be if
           set_lamps read a state nobody updates, and there is no model
           here to light them for real. So drive the state and read the
           bitmap back off the control. */
        if (g.status_is_sbar && g.lamp[0] && g.lamp[1] && !missing) {
            HGDIOBJ idle, lit;
            int saved = g.job_kind;
            g.job_kind = JOB_NONE;
            g.lamp_phase = 0;
            set_lamps();
            idle = (HGDIOBJ)SendMessage(g.lamp[1], STM_GETIMAGE,
                                        IMAGE_BITMAP, 0);
            g.job_kind = JOB_GENERATE;
            set_lamps();
            lit = (HGDIOBJ)SendMessage(g.lamp[1], STM_GETIMAGE,
                                       IMAGE_BITMAP, 0);
            g.job_kind = saved;
            set_lamps();
            st_check(f, idle == (HGDIOBJ)g.lamp_bmp[LZ_LAMP_OFF] &&
                     lit == (HGDIOBJ)g.lamp_bmp[LZ_LAMP_BUSY],
                     "lamps: the activity lamp lights while a job runs");
            checks++;

            /* Clear of the sizing grip. The grip is painted by the bar
               over the last part's own area, so "inside the part" is not
               the same as "visible" - without this margin the right lamp
               sits under the grip's ridges. */
            {
                RECT sb, l1;
                int grip = GetSystemMetrics(SM_CXVSCROLL);
                GetClientRect(g.part[LZ_GUI_STATUS], &sb);
                GetWindowRect(g.lamp[1], &l1);
                MapWindowPoints(NULL, g.part[LZ_GUI_STATUS],
                                (POINT *)&l1, 2);
                st_check(f, l1.right <= sb.right - grip,
                         "lamps: clear of the status bar's sizing grip");
                checks++;
                if (l1.right > sb.right - grip)
                    fprintf(f, "  lamp ends at %ld, grip starts at %ld\n",
                            (long)l1.right, (long)(sb.right - grip));
            }

            /* And clear of the bar's border top and bottom. The same
               defect as the grip, on the other axis and found the same
               way - by looking at it. A 16-pixel lamp centred on the
               bar's CLIENT rect sat on the border at both ends, because
               the client rect is not the part rect. The numbers are
               reported whatever the verdict: the check can only fail
               when it fails, and knowing the actual clearance is what
               makes the next change to STATUS_H reviewable. */
            {
                RECT pr, l0;
                int ok;
                if (!SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETRECT, 2,
                                 (LPARAM)&pr)) {
                    pr.top = 0; pr.bottom = 0;
                }
                GetWindowRect(g.lamp[0], &l0);
                MapWindowPoints(NULL, g.part[LZ_GUI_STATUS],
                                (POINT *)&l0, 2);
                ok = (pr.bottom > pr.top) && l0.top >= pr.top &&
                     l0.bottom <= pr.bottom;
                st_check(f, ok,
                         "lamps: inside the status bar's part, not on its "
                         "border"); checks++;
                fprintf(f, "  part y %ld..%ld, lamp y %ld..%ld\n",
                        (long)pr.top, (long)pr.bottom,
                        (long)l0.top, (long)l0.bottom);
            }
        }
    }

    /* A resizable window's status bar has a gripper. Checked on the
       style rather than by looking at pixels, and written so it is one
       check either way - a check that only exists when comctl32 does
       would make the total depend on the machine. */
    st_check(f, !g.status_is_sbar ||
                (GetWindowLong(g.part[LZ_GUI_STATUS], GWL_STYLE)
                 & SBARS_SIZEGRIP) != 0,
             "status: the comctl32 bar carries a sizing grip"); checks++;

    /* Status bar part 1, the context cell. Checked on the REAL
       bar via LZ_SB_GETTEXT, same reasoning as the greying rule right
       below: a check that only read g.ctx_tokens would agree with a
       broken wParam (part 1 vs part 0) or a dropped SendMessage call. */
    {
        char cell[128];

        /* Empty with no model - true for this entire run, since no load
           in selftest ever succeeds (st_model's own comment says why).
           A sentinel byte first, not a blank buffer: if LZ_SB_GETTEXT
           were ever silently wrong (bad wParam, message unanswered),
           an uninitialised buffer could already read as empty for the
           wrong reason and this check would agree with a broken
           message by accident. */
        cell[0] = 'X'; cell[1] = '\0';
        if (g.status_is_sbar)
            SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETTEXT, 1,
                       (LPARAM)cell);
        st_check(f, !g.status_is_sbar || cell[0] == '\0',
                 "status: the context cell is empty with no model");
        checks++;

        /* Built on g.sess directly, not st_session's own local session:
           the point here is whether update_ctx_cell's call sites really
           reach the REAL bar, not whether lz_gui_session_token_count's
           own counting logic is correct - that half has no model to
           exercise it with here either, and is not what this check is
           for.

           !ready is true for the whole run - same fact as the check
           above - so the right side of the || is never actually
           exercised on this machine; recorded here rather than hidden.
           It still runs unconditionally, so `checks` does not depend on
           whether some other environment can make ready true - on one
           that can, this same line starts checking the real thing. */
        {
            int ready;
            char err[256];
            err[0] = '\0';
            lz_gui_session_begin(&g.sess, "u", 1, err, (int)sizeof err);
            update_ctx_cell();
            ready = lz_gui_model_ready(g.sess.mdl);
            cell[0] = 'X'; cell[1] = '\0';
            if (g.status_is_sbar)
                SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETTEXT, 1,
                           (LPARAM)cell);
            st_check(f, !ready || cell[0] != '\0',
                     "status: the context cell counts tokens once a "
                     "conversation exists");
            checks++;
            /* Restored: nothing after this point may see g.sess holding
               a turn nobody asked for. */
            lz_gui_session_reset(&g.sess);
            update_ctx_cell();
        }

        /* set_ctx_cell's OWN rendering, decoupled from readiness - the
           half the check above cannot reach on this machine, since
           ready is never true here. g.ctx_tokens is driven directly,
           bypassing lz_gui_session_token_count entirely, so this IS
           sensitive on this host to what the check above is not:
           whether a real token count, once there is one,
           actually reaches the bar as the right text. */
        /* THE DENOMINATOR IS SPELLED OUT AS A LITERAL, not built by
           calling effective_ctx() a second time. A check that computes
           its expectation with the same function the product uses
           agrees with that function by construction and is blind to
           every way it could be wrong - which is exactly how a
           hardcoded LZ_GUI_SEQ_LEN survived here until the size stopped
           being a constant. 1024 and 768 are
           written out below because they are what was just assigned. */
        {
            char want[64];
            int old_ctx = g.set.ctx;
            int old_have = g.mdl.have_state, old_seq = g.mdl.seq_len;

            g.ctx_tokens = 42;
            g.set.ctx = 1024;
            g.mdl.have_state = 0;
            g.mdl.seq_len = 0;
            set_ctx_cell();
            cell[0] = '\0';
            if (g.status_is_sbar)
                SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETTEXT, 1,
                           (LPARAM)cell);
            sprintf(want, "%s 42/1024", lz_str_display(LZ_STR_STATE_CTX));
            st_check(f, !g.status_is_sbar || strcmp(cell, want) == 0,
                     "status: the context cell renders a known token "
                     "count over the requested window");
            if (g.status_is_sbar && strcmp(cell, want) != 0)
                fprintf(f, "  got %s want %s\n", cell, want);
            checks++;

            /* With a state allocated, the denominator is what was
               ALLOCATED, not what was asked for - the two differ
               whenever the model's own cfg->seq_len was the smaller
               number, and reporting the request there would be the
               status bar claiming a window that does not exist. 768 is
               deliberately not a round number the user could type in
               the context box, so it cannot have come from the setting
               or from a rounded version of it. */
            g.mdl.have_state = 1;
            g.mdl.seq_len = 768;
            set_ctx_cell();
            cell[0] = '\0';
            if (g.status_is_sbar)
                SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETTEXT, 1,
                           (LPARAM)cell);
            sprintf(want, "%s 42/768", lz_str_display(LZ_STR_STATE_CTX));
            st_check(f, !g.status_is_sbar || strcmp(cell, want) == 0,
                     "status: with a state allocated the denominator is "
                     "the allocated size, not the requested one");
            if (g.status_is_sbar && strcmp(cell, want) != 0)
                fprintf(f, "  got %s want %s\n", cell, want);
            checks++;

            g.mdl.have_state = old_have;
            g.mdl.seq_len = old_seq;
            g.set.ctx = old_ctx;
            g.ctx_tokens = -1;
            set_ctx_cell();
        }
    }

    /* The greying rule, checked on the REAL menu and the REAL button
       rather than on the call that is supposed to do it. The failure
       mode this guards against: cmd_enable handed control ids that no
       menu item and no toolbar button carries - every call succeeds and
       nothing changes, and a check that watched cmd_enable's arguments
       would agree with it.

       Send stays live with no model open: a user with no model open
       still has /load and /help to
       reach, and greying Send would tell them otherwise. See
       create_children's cmd_enable(ID_SEND, 1) call for the fuller
       reasoning. */
    {
        HMENU bar = GetMenu(hwnd);
        UINT st = bar ? GetMenuState(bar, IDM_STOP_GEN, MF_BYCOMMAND)
                      : (UINT)-1;
        st_check(f, st != (UINT)-1 && (st & MF_GRAYED),
                 "enable: Stop is greyed while nothing is generating");
        checks++;
        st_check(f, IsWindowEnabled(g.part[LZ_GUI_SEND]) != 0,
                 "enable: Send stays live with no model open");
        checks++;
    }

    /* The ON branch of cmd_enable(ID_SEND, ...). Every call site now
       passes 1 unconditionally (create_children, finish_job,
       apply_language - see their own comments), so by the time
       selftest reaches here the button is already live, and flipping
       ON a control that is already on would prove nothing: it cannot
       tell "cmd_enable's ID_SEND branch works" apart from "nobody
       touched it since create_children". The block below forces OFF
       first with a raw EnableWindow, not cmd_enable(ID_SEND, 0) - which
       no call site uses anymore either - so the ON call afterward is
       the only thing that can flip it back. This is the same
       decoupling the original version of this check used against a
       different degenerate case: a mutation that broke cmd_enable's
       ID_SEND branch entirely, in both directions behind one guarded
       `if`, leaves Send disabled by construction and looks identical
       either way until OFF is established independently of the
       function under test. */
    {
        EnableWindow(g.part[LZ_GUI_SEND], FALSE);
        cmd_enable(ID_SEND, 1);
        st_check(f, IsWindowEnabled(g.part[LZ_GUI_SEND]) != 0,
                 "enable: Send ungreys when cmd_enable turns it on");
        checks++;
        /* Left live, not forced back to FALSE - idle state IS
           Send-enabled now, per the check above. */
    }

    /* The central WM_COMMAND guard (cmd_is_enabled, called at the top
       of the WM_COMMAND case in wndproc): a disabled command id must
       be rejected AT DISPATCH, not just look disabled. Driven through
       the exact message input_subclass posts - WM_COMMAND/ID_SEND to
       the main window - rather than by calling do_send directly, so a
       mutation that deletes the guard but leaves do_send untouched is
       still caught here; calling do_send directly would test the
       wrong function.

       "/help" is the probe: do_command's LZ_CMD_HELP branch pushes
       LZ_STR_HELP_BODY into the transcript synchronously, needs no
       model, and cannot block on the worker - see do_send's own
       "commands come first" comment for why this line reaches
       do_command before do_send's model-ready check ever runs. */
    {
        HWND tr = g.part[LZ_GUI_TRANSCRIPT];
        int before, after;

        transcript_clear();
        SetWindowTextA(g.part[LZ_GUI_INPUT], "/help");
        EnableWindow(g.part[LZ_GUI_SEND], FALSE);
        before = GetWindowTextLength(tr);
        SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(ID_SEND, 0), 0);
        after = GetWindowTextLength(tr);
        st_check(f, after == before,
                 "dispatch: WM_COMMAND is rejected for a disabled id");
        checks++;

        EnableWindow(g.part[LZ_GUI_SEND], TRUE);
        SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(ID_SEND, 0), 0);
        after = GetWindowTextLength(tr);
        st_check(f, after > before,
                 "dispatch: the same id reaches do_send once re-enabled");
        checks++;

        transcript_clear();
        SetWindowTextA(g.part[LZ_GUI_INPUT], "");
    }

    /* A round trip through the real API, not a check that the setter
       returned. The failure this catches is a key written under one
       name and read under another, which no amount of reading the code
       makes visible. Values are restored afterwards so the selftest
       does not leave the developer's own ini rewritten.

       This does NOT prove the value reached disk. Both the write and
       the read go through GetPrivateProfileIntA / WritePrivateProfileStringA,
       which cache the file in-process, and a round trip that never
       leaves that cache passes exactly the same way whether or not the
       real WriteFile underneath it succeeded - lz_ini_set_int's own
       WritePrivateProfileStringA return value is discarded (see
       gui/inifile.c), so nothing else in the process would ever see
       that write fail either. That is the real failure mode on the
       target: the exe installed to a directory the process account
       cannot write to (an NT 3.51/NT4 ACL, not a hypothetical - it is
       how the target platform's file permissions actually work), where
       every in-process read after a failed write still returns the
       value that failed to save. The block right after this one closes
       that gap by reading the file's own bytes. */
    {
        int old_h = lz_ini_get_int("input_h", 0);
        int got;
        lz_ini_set_int("input_h", 137);
        got = lz_ini_get_int("input_h", -1);
        lz_ini_set_int("input_h", old_h);
        st_check(f, got == 137, "ini: an int survives a write and a read");
        checks++;
    }
    {
        /* The write forced out of WritePrivateProfileString's cache
           (NULL/NULL/NULL is the documented way to flush it) and then
           read back with a plain fopen, never through any profile API,
           so an in-process cache cannot make this pass on its own -
           the literal bytes on disk are the judge. */
        char path[MAX_PATH + 8];
        char raw[4096];
        int old_h2 = lz_ini_get_int("input_h", 0);
        int on_disk = 0;
        if (lz_ini_path(path, (int)sizeof path)) {
            FILE *rf;
            lz_ini_set_int("input_h", 731);
            WritePrivateProfileStringA(NULL, NULL, NULL, path);
            rf = fopen(path, "rb");
            if (rf) {
                size_t n = fread(raw, 1, sizeof raw - 1, rf);
                raw[n] = '\0';
                fclose(rf);
                on_disk = strstr(raw, "input_h=731") != NULL;
            }
            lz_ini_set_int("input_h", old_h2);
            WritePrivateProfileStringA(NULL, NULL, NULL, path);
        }
        st_check(f, on_disk,
                 "ini: a write reaches the file's own bytes, not just "
                 "the in-process profile cache");
        checks++;
    }
    {
        char old[512], got[512];
        lz_ini_get_str("model", old, (int)sizeof old, "");
        lz_ini_set_str("model", "C:\\x\\model dir");
        lz_ini_get_str("model", got, (int)sizeof got, "");
        lz_ini_set_str("model", old);
        st_check(f, strcmp(got, "C:\\x\\model dir") == 0,
                 "ini: a path with a space survives a write and a read");
        checks++;
    }

    /* The text must be the DISPLAY form. Comparing against the UTF-8
       source instead would pass on an ASCII label and fail on every
       Chinese one, which is the bug this checks for. The sidebar name
       is the one plain label left after the tool strip became a
       toolbar; the toolbar's own labels are checked below by asking the
       control, because they live inside it and not on child windows. */
    st_check_text(f, LZ_GUI_SIDE_NAME, LZ_STR_APP_TITLE,
                  "text: sidebar name"); checks++;

    /* The UI font. Three checks, not one - the obvious suggestion
       (WM_GETFONT on the transcript, read back with GetObjectA) turned
       out not to work, and MEASURING that rather than assuming it
       changed the design:
       WM_GETFONT on g.part[LZ_GUI_TRANSCRIPT] (a RichEdit) returns NULL
       even immediately after create_children runs, regardless of
       WM_SETFONT having been sent to it - confirmed by comparing it to
       g.part[LZ_GUI_INPUT] (a plain EDIT) at the same point, which
       correctly answers WM_GETFONT with exactly lz_ui_font()'s HFONT.
       RichEdit simply does not answer that message the way a stock
       EDIT does; nothing here was ever wired wrong.
       So the font ITSELF is checked directly (lz_ui_font(), not by
       asking a control for it), and REAL WIRING is proven on the
       control that can actually answer for itself - the input box -
       plus one RichEdit-specific signal that is genuinely provable
       without being a round-trip of the same assumption. */
    if (!lz_str_lang_english()) {
        LOGFONTA lf;
        HFONT ifont;
        LONG opt;

        /* 1. The font itself. Chinese only: the English branch keeps
           DEFAULT_GUI_FONT's stock object, whose face name is whatever
           the running system supplies and is not this project's to
           assert. 134 is GB2312_CHARSET's value, used as a literal for
           the same reason gui/compat40.c's own font code does -
           wingdi.h's charset constants are not guaranteed present at
           the 3.51 declaration level that file compiles against, and a
           second #define here could silently drift from its value
           instead of matching it. */
        memset(&lf, 0, sizeof lf);
        st_check(f, GetObjectA((HGDIOBJ)lz_ui_font(), (int)sizeof lf,
                              &lf) != 0 &&
                 strcmp(lf.lfFaceName, "\xCB\xCE\xCC\xE5") == 0 &&
                 lf.lfCharSet == 134 /* GB2312_CHARSET */,
                 "font: lz_ui_font() builds SimSun by name and charset "
                 "for Chinese, not the stock object");
        checks++;

        /* 2. WM_SETFONT wiring, proven where it CAN be: the input box
           answers WM_GETFONT normally (a plain EDIT, unlike the
           RichEdit above), so this is real evidence create_children's
           apply_font loop reached a real control with the real font
           object, not two functions that happen to agree with each
           other in isolation. */
        ifont = (HFONT)SendMessage(g.part[LZ_GUI_INPUT], WM_GETFONT, 0, 0);
        st_check(f, ifont == lz_ui_font(),
                 "font: WM_SETFONT wiring reaches a real control (the "
                 "input box) with the real font object, not just two "
                 "functions agreeing in isolation");
        checks++;

        /* 3. The one RichEdit-specific signal that is genuinely
           provable: EM_GETLANGOPTIONS's IMF_AUTOFONT bit, which
           lz_richedit_use_font turns off, is NOT the same trap as
           reading back the CHARFORMAT lz_richedit_use_font itself just
           wrote (team-lead's own "round-trip is blind to a shared
           assumption" caution) - this reads a DIFFERENT flag from a
           DIFFERENT starting state (RichEdit's own default, ON) than
           the one the call sets, so it can fail for a real reason:
           lz_richedit_use_font's call never reaching this control. It
           still does not prove what gets PAINTED for an ASCII run
           sharing a line with Chinese text - that is CHARFORMAT
           territory, genuinely left unverified rather than faked with
           a check that cannot fail for the reason it claims to. */
        opt = (LONG)SendMessage(g.part[LZ_GUI_TRANSCRIPT], WM_USER + 121,
                                0, 0);
        st_check(f, (opt & 0x0002L) == 0,
                 "font: lz_richedit_use_font turned IMF_AUTOFONT off on "
                 "the transcript (RichEdit's own default is on)");
        checks++;
    }

    /* The toolbar. Four things, and the last is the one that would
       otherwise be found by a user rather than by a test: the layout
       gives the bar a fixed height, and if a button needs more than
       that the labels are clipped - which reads as a font problem. */
    {
        HWND tb = g.part[LZ_GUI_TOOLBAR];
        LZRect r[LZ_GUI_PART_COUNT];
        int need = lz_gui_toolbar_needed_height(tb);
        lz_gui_layout(640, 480, 0, g.status_h, side_panel_mode(), r);
        st_check(f, tb != NULL && IsWindow(tb),
                 "toolbar: the control exists"); checks++;
        st_check(f, tb == NULL || need > 0,
                 "toolbar: it reports a button size"); checks++;
        st_check(f, tb == NULL || need <= r[LZ_GUI_TOOLBAR].h,
                 "toolbar: the layout is tall enough for its buttons");
        checks++;
        /* Always printed, not only on failure: this number is how
           TOOLBAR_H gets chosen, and a value nobody can read is a value
           somebody guesses. */
        fprintf(f, "  toolbar needs %d, layout gives %d\n",
                need, r[LZ_GUI_TOOLBAR].h);
    }

    /* Proof the check above can actually go red, the same way
       measure_status_h's own big-font block (further down) proves
       itself against LZ_GUI_STATUS_H: a synthetic, obviously-oversized
       font forced onto the toolbar with WM_SETFONT directly, bypassing
       lz_ui_font()/apply_font entirely, so this does not depend on
       this host's two languages resolving to different fonts (they do
       not - see apply_language's own font-collapse note). If
       TB_GETMAXSIZE genuinely reflects the control's real font (the
       thing "toolbar: the layout is tall enough for its buttons"
       relies on), an obviously oversized font must make it report more
       than the layout reserves - the SAME comparison the check above
       makes, on data engineered to fail it, which is what "the check
       can go red" actually means rather than an assumption. */
    {
        HWND tb = g.part[LZ_GUI_TOOLBAR];
        HFONT big = CreateFontA(-60, 0, 0, 0, FW_NORMAL, 0, 0, 0,
                                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
                                DEFAULT_PITCH | FF_DONTCARE, NULL);
        LZRect r[LZ_GUI_PART_COUNT];
        int need_big = 0;

        lz_gui_layout(640, 480, 0, g.status_h, side_panel_mode(), r);
        if (big && tb) {
            SendMessage(tb, WM_SETFONT, (WPARAM)big, MAKELPARAM(TRUE, 0));
            need_big = lz_gui_toolbar_needed_height(tb);
        }
        st_check(f, !big || !tb || need_big > r[LZ_GUI_TOOLBAR].h,
                 "toolbar: an artificially oversized font makes the "
                 "layout's reserved height genuinely too small");
        if (big && tb) fprintf(f, "  oversized font needs %d, layout "
                               "gives %d\n", need_big, r[LZ_GUI_TOOLBAR].h);
        checks++;

        /* Real font back before anything else in this file reads the
           toolbar - apply_font is the same call create_children itself
           uses, so this restores exactly the state that call leaves,
           not a guess at it. */
        if (tb) apply_font(tb);
        if (big) DeleteObject((HGDIOBJ)big);
    }
    /* The menu bar: the window menu
       is a real menu bar with four popups. */
    {
        HMENU bar = GetMenu(hwnd);
        st_check(f, bar != NULL, "menu: a menu bar exists"); checks++;
        if (bar) {
            st_check(f, GetMenuItemCount(bar) == 5,
                     "menu: five popups"); checks++;
        }

        /* Mnemonics, read back off the REAL menu rather than off the
           string table. The table is where they are written, but the
           thing a user presses Alt against is the menu, and between the
           two sits AppendMenuA and the GBK conversion. Read the menu. */
        {
            char buf[256];
            int barm[16], nb = bar ? GetMenuItemCount(bar) : 0;
            int i2, j2, missing = 0, dup = 0;
            if (nb > 16) nb = 16;
            for (i2 = 0; i2 < nb; i2++) {
                HMENU pop = GetSubMenu(bar, i2);
                int pm[32], nm = 0, np, k;
                buf[0] = '\0';
                GetMenuStringA(bar, i2, buf, (int)sizeof buf, MF_BYPOSITION);
                barm[i2] = st_mnemonic(buf);
                if (!barm[i2]) missing++;
                np = pop ? GetMenuItemCount(pop) : 0;
                for (k = 0; k < np && nm < 32; k++) {
                    buf[0] = '\0';
                    if (GetMenuStringA(pop, k, buf, (int)sizeof buf,
                                       MF_BYPOSITION) <= 0)
                        continue;              /* separator */
                    pm[nm] = st_mnemonic(buf);
                    if (!pm[nm]) missing++;
                    for (j2 = 0; j2 < nm; j2++)
                        if (pm[j2] && pm[j2] == pm[nm]) dup++;
                    nm++;
                }
            }
            /* Across the bar too: Alt+F must not be ambiguous. */
            for (i2 = 0; i2 < nb; i2++)
                for (j2 = i2 + 1; j2 < nb; j2++)
                    if (barm[i2] && barm[i2] == barm[j2]) dup++;

            st_check(f, nb > 0 && missing == 0,
                     "menu: every item carries a mnemonic"); checks++;
            st_check(f, dup == 0,
                     "menu: no two items in one menu share a mnemonic");
            checks++;
        }
    }

    /* MRU: two fake paths pushed straight into g.mru - no
       loader involved, this checks the MENU, not the load path - the
       bar rebuilt, and the File popup read back with GetMenuStringA,
       the only way to see what a user's Alt+F actually shows. Saved and
       restored so this does not leave the real menu holding fake paths
       for the checks that follow. */
    {
        LZMru saved = g.mru;
        HMENU bar, file_pop;
        char buf[600];

        lz_mru_init(&g.mru);
        lz_mru_push(&g.mru, "C:\\models\\second");
        /* Pushed last, so it lands at item[0] / IDM_MRU0 - the one this
           block reads back. */
        lz_mru_push(&g.mru, "C:\\models\\a & b");
        build_menu_bar(hwnd);

        bar = GetMenu(hwnd);
        file_pop = bar ? GetSubMenu(bar, 0) : NULL;   /* BAR[0] is File */
        buf[0] = '\0';
        if (file_pop)
            GetMenuStringA(file_pop, IDM_MRU0, buf, (int)sizeof buf,
                           MF_BYCOMMAND);
        st_check(f, strstr(buf, "C:\\models\\a") != NULL,
                 "mru: the recent list appears in the File menu");
        checks++;
        st_check(f, strstr(buf, "a && b") != NULL,
                 "mru: an ampersand in a path is not eaten by the menu");
        checks++;

        g.mru = saved;
        build_menu_bar(hwnd);
    }

    /* The accelerator table. A menu that advertises Ctrl+O in its own
       label and then does not answer to it is worse than no accelerator
       at all, so both halves are checked - here that the table exists
       and carries exactly the entries gui/kunkun98.rc declares, and
       that its keys are the ones the menu text
       promises. Exact count, not a loose lower bound: Ctrl+O, Ctrl+S,
       Ctrl+C, Ctrl+A, Esc, Ctrl+N and Ctrl+F - a bound that
       only ever grows (">= 2" survived two more entries being added on
       top of it without anyone having to look) would silently stop
       meaning anything the day an entry is removed instead of added. */
    st_check(f, g.accel != NULL &&
                CopyAcceleratorTable(g.accel, NULL, 0) == 7,
             "accel: the accelerator table has exactly the entries "
             "gui/kunkun98.rc declares"); checks++;
    /* The application icon resource must load; the chicken lives in
       the title bar (and Alt-Tab) now, not on a button. */
    {
        HICON hmi = lz_ui_icon_16(g.inst);
        st_check(f, hmi != NULL, "icon: app 16x16 loads"); checks++;
        if (hmi) DestroyIcon(hmi);
    }
    st_check_text(f, LZ_GUI_STATUS, LZ_STR_STATE_NO_MODEL,
                  "text: status"); checks++;
    {
        char title[256];
        LZCaption want;
        GetWindowTextA(hwnd, title, (int)sizeof title);
        /* The title is now the COMPOSED three segments, not the bare app
           title: no model and no chat file at this point, so it is
           "brand - untitled". Build the same way push_caption does and
           compare, rather than hardcoding a second copy of the format. */
        memset(&want, 0, sizeof want);
        lstrcpynA(want.brand, lz_str_display(LZ_STR_APP_TITLE), LZ_CAP_BRAND);
        lstrcpynA(want.chat, lz_str_display(LZ_STR_CAPTION_UNTITLED),
                  LZ_CAP_CHAT);
        {
            char flat[LZ_CAP_BRAND + LZ_CAP_CHAT + LZ_CAP_MODEL + 8];
            lz_caption_compose(&want, flat, (int)sizeof flat);
            st_check(f, strcmp(title, flat) == 0, "text: window title");
        }
        checks++;
    }

    /* Every control must sit exactly where layout.c says, at every size.
       MoveWindow taking a rectangle and the window ending up elsewhere
       is what a forgotten WM_SIZE, or a child of the wrong parent, looks
       like. */
    for (s = 0; s < (int)(sizeof SIZES / sizeof SIZES[0]); s++) {
        RECT wr, rc;
        LZRect want[LZ_GUI_PART_COUNT];
        wr.left = 0; wr.top = 0;
        wr.right = SIZES[s][0]; wr.bottom = SIZES[s][1];
        AdjustWindowRect(&wr, WS_OVERLAPPEDWINDOW, FALSE);
        MoveWindow(hwnd, 0, 0, wr.right - wr.left, wr.bottom - wr.top, FALSE);
        /* The real client area, not SIZES: the menu bar shrinks it and
           that is the size relayout actually laid out with. */
        GetClientRect(hwnd, &rc);
        lz_gui_layout(rc.right, rc.bottom, g.input_h, g.status_h,
                     side_panel_mode(), want);
        for (i = 0; i < LZ_GUI_PART_COUNT; i++) {
            RECT got;
            char what[96];
            if (i == LZ_GUI_SPLIT) {
                /* No window to measure; what has to agree is the band
                   the mouse code hit-tests against. */
                sprintf(what, "rect %dx%d: split band matches the layout",
                        SIZES[s][0], SIZES[s][1]);
                st_check(f, g.split.x == want[i].x && g.split.y == want[i].y
                         && g.split.w == want[i].w && g.split.h == want[i].h,
                         what);
                checks++;
                continue;
            }
            if (i == LZ_GUI_SIDE_LAMPS) {
                /* No single HWND either - the 16 expert lamps live in
                   g.elamp[], not g.part[] (layout.h's own comment on
                   this part explains why). This sweep never loads a
                   model (st_worker/st_model/st_real_model all run
                   later), so g.elamp[0] is NULL and want[i] is zero-
                   sized every time through - asserted, not assumed, so
                   a regression that made a model appear loaded this
                   early would still be caught. The non-NULL branch is
                   still written correctly (checked against the grid's
                   own origin + footprint, from relayout's placement of
                   elamp[0] - not a readback of something this loop
                   computed itself) for whichever future caller reaches
                   this code with elamp[] actually populated. */
                if (!g.elamp[0]) {
                    sprintf(what, "rect %dx%d: %s is absent (no MoE "
                            "model loaded)", SIZES[s][0], SIZES[s][1],
                            lz_gui_part_name((LZGuiPart)i));
                    st_check(f, want[i].w <= 0 && want[i].h <= 0, what);
                } else {
                    RECT g0;
                    GetWindowRect(g.elamp[0], &g0);
                    MapWindowPoints(NULL, hwnd, (POINT *)&g0, 2);
                    sprintf(what, "rect %dx%d: %s (grid origin + "
                            "footprint)", SIZES[s][0], SIZES[s][1],
                            lz_gui_part_name((LZGuiPart)i));
                    st_check(f, g0.left == want[i].x && g0.top == want[i].y
                             && LZ_GUI_ELAMP_GRID_W == want[i].w
                             && LZ_GUI_ELAMP_GRID_H == want[i].h, what);
                }
                checks++;
                continue;
            }
            if (!g.part[i]) {
                /* LZ_GUI_SIDE_CAND, same reasoning as LZ_GUI_SIDE_LAMPS
                   just above - not created until a model is, and this
                   sweep never loads one. */
                sprintf(what, "rect %dx%d: %s is absent (no model "
                        "loaded)", SIZES[s][0], SIZES[s][1],
                        lz_gui_part_name((LZGuiPart)i));
                st_check(f, want[i].w <= 0 && want[i].h <= 0, what);
                checks++;
                continue;
            }
            GetWindowRect(g.part[i], &got);
            MapWindowPoints(NULL, hwnd, (POINT *)&got, 2);
            sprintf(what, "rect %dx%d: %s", SIZES[s][0], SIZES[s][1],
                    lz_gui_part_name((LZGuiPart)i));
            st_check(f, got.left == want[i].x && got.top == want[i].y &&
                     got.right - got.left == want[i].w &&
                     got.bottom - got.top == want[i].h, what);
            checks++;
        }
    }

    /* The window must refuse to go below the layout's minimum. */
    {
        MINMAXINFO mmi;
        RECT rc;
        memset(&mmi, 0, sizeof mmi);
        SendMessage(hwnd, WM_GETMINMAXINFO, 0, (LPARAM)&mmi);
        rc.left = 0; rc.top = 0;
        rc.right = LZ_GUI_MIN_CW; rc.bottom = LZ_GUI_MIN_CH;
        AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
        st_check(f, mmi.ptMinTrackSize.x == rc.right - rc.left &&
                 mmi.ptMinTrackSize.y == rc.bottom - rc.top +
                        GetSystemMetrics(SM_CYMENU),
                 "minimum window size matches the layout minimum");
        checks++;
    }

    checks += st_prefill(f);
    checks += st_transcript(f);
    checks += st_find(f);
    checks += st_worker(f, hwnd);
    checks += st_rollback(f, hwnd);
    checks += st_model(f);
    checks += st_real_model(f, hwnd);
    checks += st_session(f);
    checks += st_save(f);
    checks += st_system_prompt(f);
    checks += st_sampling(f);
    checks += st_ctx_window(f);
    checks += st_ctx_commit(f, hwnd);
    checks += st_settings_dialog(f, inst);
    checks += st_settings_scroll(f, inst);
    checks += st_about(f, inst);

    /* Resources. Both are things that go missing by a build step being
       forgotten rather than by code being wrong, and the symptom of a
       missing icon is a generic icon nobody looks twice at. */
    {
        int sw = 0, sh = 0;
        st_check(f, lz_gui_splash_size(inst, &sw, &sh),
                 "resource: the splash bitmap is in the binary");
        checks++;
        /* The size is fixed by design, and a wrong one means some other
           bitmap got compiled in. */
        st_check(f, sw == 492 && sh == 283,
                 "resource: the splash is 492x283");
        if (!(sw == 492 && sh == 283)) fprintf(f, "  got %dx%d\n", sw, sh);
        checks++;
        st_check(f, LoadIconA(inst, MAKEINTRESOURCEA(IDI_APP)) != NULL,
                 "resource: the application icon is in the binary");
        checks++;
    }

    /* g.status_h staying current after a language switch, not just at
       creation. Before the text-only language-switch block below, for
       the same "changes the window under everything above" reason -
       and restores BOTH the font and the language before that block
       runs, so its own zh_* baselines are still captured against the
       state it expects.

       WHY THIS CANNOT BE "switch language, assert geometry": on THIS
       machine (and on any host where the two languages resolve to the
       SAME underlying font - measured with shot.ps1 -Font:
       zh-CN's DEFAULT_GUI_FONT stock object, MinGW's dev host, and
       SimSun by name are literally the same font under two different
       byte encodings of the same face) a real language switch changes
       NOTHING about the status bar's real height, so that check would
       be permanently green whether or not g.status_h re-measurement
       ever ran - the exact "judge was never sensitive to begin with"
       shape. Two
       checks instead, neither depending on what this host's fonts
       happen to be:

       1. measure_status_h() itself reacts to a REAL font-height
          change - manufactured directly (a synthetic 60px font forced
          onto the control with WM_SETFONT, bypassing lz_ui_font()/
          apply_font entirely) rather than hoped for from a language
          switch. This is the part that would stay broken if
          measure_status_h() itself always returned the same number
          regardless of what font the control actually has.
       2. apply_language() really CALLS measure_status_h() - a
          sentinel, not a value comparison, for the same reason
          the update_ctx_cell checks use one: g.status_h before
          and after could legitimately be the SAME real number (as it
          is on this host), so a before/after VALUE comparison cannot
          tell "re-measured and got the same real answer" apart from
          "the call site was never reached at all". Poisoning
          g.status_h with a value no real measurement could produce,
          then checking it is gone, can tell them apart. */
    {
        HFONT big = CreateFontA(-60, 0, 0, 0, FW_NORMAL, 0, 0, 0,
                                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                                CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY,
                                DEFAULT_PITCH | FF_DONTCARE, NULL);
        int normal_h, big_h, comctl32_varies;

        measure_status_h();
        normal_h = g.status_h;
        big_h = normal_h;
        if (big && g.part[LZ_GUI_STATUS]) {
            SendMessage(g.part[LZ_GUI_STATUS], WM_SETFONT, (WPARAM)big,
                       MAKELPARAM(TRUE, 0));
            measure_status_h();
            big_h = g.status_h;
        }
        /* MEASURED, not assumed: comctl32 v6 (themed -
           what this build gets whenever it runs with a manifest, on
           any host) does not auto-size a status bar from its font at
           all - it honours whatever height measure_status_h's own
           probe asks for, exactly, so normal_h == LZ_GUI_STATUS_H on
           that comctl32 version REGARDLESS of what font the control
           has. v5 (unthemed - kunkun98-noman) is the one that
           substitutes its own font-derived answer, which is the whole
           reason this file's status_h exists. Confirmed on both:
           this exact block reported normal=22 big=22 on kunkun98-gui
           and normal=20 big=26 on kunkun98-noman, same 60px font,
           same code, same host.
           So "does a bigger font produce a bigger measurement" is
           only a MEANINGFUL question when comctl32_varies is true -
           on the build where it is false, asserting `>` would fail
           for a reason that has nothing to do with this file being
           right or wrong, the same shape st_real_model's
           `!have_model ||` guards elsewhere in this file exist for.
           comctl32_varies is discovered from the SAME baseline
           measurement the assertion needs anyway, not hardcoded per
           build - this is the fact that tells the two comctl32
           versions apart, not an assumption about which one is
           running.

           BLIND SPOT, found by mutating measure_status_h() itself
           (not just apply_language's call to it) to always assign
           LZ_GUI_STATUS_H: this check does not catch that mutation on
           its own, because comctl32_varies is computed from the very
           function the mutation broke - a hardcoded measure_status_h
           reports normal_h == LZ_GUI_STATUS_H unconditionally, which
           reads as "this comctl32 legitimately does not vary" rather
           than "the measurement is broken", so the guard degrades to
           vacuously true. That specific mutation is NOT invisible
           overall, though: it is exactly the pre-existing "rect NxM:
           status" checks firing again on kunkun98-noman (3 fails, the
           original bug), since those run earlier in
           this function and read g.status_h through relayout rather
           than through this check's own local variables. Recorded
           here rather than left to be rediscovered, because a check
           whose own success condition is computed from the function
           it is checking is a shape worth being suspicious of on
           sight. */
        comctl32_varies = normal_h != LZ_GUI_STATUS_H;
        st_check(f, big != NULL &&
                 (!comctl32_varies || big_h > normal_h),
                 "font: measure_status_h reacts to a real font-height "
                 "change where this comctl32 version varies by font "
                 "at all");
        fprintf(f, "  normal=%d big=%d varies=%d\n", normal_h, big_h,
               comctl32_varies);
        checks++;

        /* Put the real font back before anything else in this file
           reads the status bar, and before the sentinel test below -
           that one needs g.status_h to start from a real value, not
           the manufactured 60px one. */
        if (g.part[LZ_GUI_STATUS]) apply_font(g.part[LZ_GUI_STATUS]);
        measure_status_h();
        if (big) DeleteObject((HGDIOBJ)big);
    }
    {
        int poison = -999999;
        g.status_h = poison;
        apply_language(hwnd, 1);
        st_check(f, g.status_h != poison,
                 "font: apply_language re-measures g.status_h, not "
                 "only create_children");
        checks++;
        apply_language(hwnd, 0);   /* back to the state the block below expects */
    }

    /* The language switch. Last, because it changes the window under
       everything above.
     *
     * Every check here compares against the bytes CAPTURED BEFORE the
     * switch, not against the table. Comparing a control to
     * lz_str_display() after switching is the version of this check that
     * proves nothing: it passes when the control was never touched and
     * the table moved underneath it, which is exactly the bug - a
     * caption frozen at creation time. The saved bytes are what makes a
     * "the switch changed nothing" result visible. */
    {
        char zh_send[128], zh_status[160], en_send[128], back[128];
        char zh_tb[128], en_tb[128];
        HWND tb;
        zh_send[0] = zh_status[0] = en_send[0] = back[0] = '\0';
        zh_tb[0] = en_tb[0] = '\0';
        GetWindowTextA(g.part[LZ_GUI_SEND], zh_send, (int)sizeof zh_send);
        strncpy(zh_status, g.idle_status, sizeof zh_status - 1);
        zh_status[sizeof zh_status - 1] = '\0';
        tb = g.part[LZ_GUI_TOOLBAR];
        if (tb) lz_gui_toolbar_text(tb, IDM_OPEN_MODEL, zh_tb,
                                    (int)sizeof zh_tb);

        apply_language(hwnd, 1);
        GetWindowTextA(g.part[LZ_GUI_SEND], en_send, (int)sizeof en_send);
        if (g.part[LZ_GUI_TOOLBAR])
            lz_gui_toolbar_text(g.part[LZ_GUI_TOOLBAR], IDM_OPEN_MODEL,
                                en_tb, (int)sizeof en_tb);

        st_check(f, strcmp(en_send, zh_send) != 0 &&
                    strcmp(en_send, lz_str_display(LZ_STR_BTN_SEND)) == 0,
                 "language: the Send button follows the switch"); checks++;
        st_check(f, strcmp(g.idle_status, zh_status) != 0 &&
                    strcmp(g.idle_status,
                           lz_str_display(LZ_STR_STATE_NO_MODEL)) == 0,
                 "language: the resting status line follows the switch");
        checks++;
        /* The tool strip is the one that cannot be relabelled in place,
           so it is the one most likely to miss the switch. */
        st_check(f, !tb || (strcmp(en_tb, zh_tb) != 0 &&
                            strcmp(en_tb,
                                   lz_str_display(LZ_STR_BTN_OPEN)) == 0),
                 "language: the tool strip is rebuilt in the new language");
        checks++;
        if (tb && strcmp(en_tb, zh_tb) == 0)
            fprintf(f, "  tool strip still says %s\n", zh_tb);

        /* The SAME assertion "toolbar: the layout is tall enough for
           its buttons" makes, re-run against the REAL toolbar this
           switch just rebuilt in English - a gap: that check runs
           once, in create_children's initial (Chinese) state, and
           create_children happens long before this block, so it has
           never once been evaluated against the English labels/font
           apply_language actually builds. g.part[LZ_GUI_TOOLBAR], not
           the local `tb` above - apply_language destroys the old
           control and replaces it, so `tb` is a stale handle by this
           point (the same reason the toolbar-text checks just above
           read g.part[LZ_GUI_TOOLBAR] directly instead of `tb` too).
           The oversized-font block further up already proves this
           exact comparison can go red for a real reason; this is
           whether it actually does, on the real English toolbar. */
        {
            HWND tb_en = g.part[LZ_GUI_TOOLBAR];
            LZRect r_en[LZ_GUI_PART_COUNT];
            int need_en = lz_gui_toolbar_needed_height(tb_en);
            lz_gui_layout(640, 480, 0, g.status_h, side_panel_mode(), r_en);
            st_check(f, tb_en == NULL || need_en <= r_en[LZ_GUI_TOOLBAR].h,
                     "toolbar: the layout is tall enough for its "
                     "buttons in English too");
            checks++;
            fprintf(f, "  toolbar (English) needs %d, layout gives %d\n",
                    need_en, r_en[LZ_GUI_TOOLBAR].h);
        }

        /* The tick, read off the real menu - the same argument as the
           mnemonic check above. */
        {
            HMENU bar = GetMenu(hwnd);
            UINT szh = bar ? GetMenuState(bar, IDM_LANG_ZH, MF_BYCOMMAND)
                           : (UINT)-1;
            UINT sen = bar ? GetMenuState(bar, IDM_LANG_EN, MF_BYCOMMAND)
                           : (UINT)-1;
            st_check(f, szh != (UINT)-1 && sen != (UINT)-1 &&
                        !(szh & MF_CHECKED) && (sen & MF_CHECKED),
                     "language: the menu tick moves to the live language");
            checks++;
        }

        apply_language(hwnd, 0);
        GetWindowTextA(g.part[LZ_GUI_SEND], back, (int)sizeof back);
        st_check(f, strcmp(back, zh_send) == 0 &&
                    strcmp(g.idle_status, zh_status) == 0,
                 "language: switching back restores the original text");
        checks++;

        /* A language switch RE-SPELLS what the window is; it does not
           change it. Going through set_idle_status would also clear
           tok_live, and nothing sets that again until the NEXT job
           starts - so switching language mid-generation cost that job
           its throughput cell for good. Reddens if apply_language goes
           back to set_idle_status. */
        {
            int was = g.tok_live;
            g.tok_live = 1;
            apply_language(hwnd, 1);
            st_check(f, g.tok_live == 1,
                     "language: switching mid-job leaves the throughput "
                     "cell live");
            apply_language(hwnd, 0);
            g.tok_live = was;
            checks++;
        }
    }

    /* Edit menu: IDM_COPY and IDM_SELECT_ALL both forward to
       GetFocus(), so the whole point is exercised only if focus is
       actually moved onto the control under test rather than left to
       whatever the default happens to be.

       Iron law four's second note applies to the clipboard itself: a
       readback that matches what was just written could just as well be
       leftover content nobody put there this run, so the FIRST check
       below proves the instrument is connected - a cleared clipboard
       really does read back empty - before either command is trusted to
       have put anything into it.

       Everything here goes through st_clip_open and can end in st_skip:
       the clipboard belongs to the desktop, not to this process, and a
       contended shared resource must not be reported as a broken Copy
       command. */
    {
        HWND tr = g.part[LZ_GUI_TRANSCRIPT];
        /* GetTickCount alone is not unique enough to stamp with: it
           moves in 15.6 ms steps here (iron law three's note at
           lz_time_ms), and the classic_ui harness runs this twice.
           The counter makes the string differ even inside one tick. */
        static int probe_no;
        char unique[64];
        const char *busy = "another process held the clipboard for the "
                           "whole retry window";
        const char *what_empty = "edit: the clipboard instrument reads back "
                                 "empty right after EmptyClipboard";
        const char *what_copy = "edit: Copy reaches the control that has "
                                "focus";

        sprintf(unique, "kk98-clip-probe-%lu-%d",
                (unsigned long)GetTickCount(), ++probe_no);

        if (!st_clip_open(hwnd)) {
            st_skip(f, what_empty, busy);
        } else {
            /* Emptied and read back inside ONE open. Closing in between
               leaves a window for another process to put something on
               the clipboard, and then this check would be reporting on
               that process. */
            int empty_ok;
            EmptyClipboard();
            empty_ok = GetClipboardData(CF_TEXT) == NULL;
            CloseClipboard();
            st_check(f, empty_ok, what_empty);
            checks++;
        }

        transcript_clear();
        transcript_push(unique, (int)strlen(unique));
        transcript_end();
        SendMessage(tr, EM_SETSEL, 0, (LPARAM)-1);
        SetFocus(tr);
        {
            int ok = 0, opened = 0, attempt;
            /* The Copy and the readback retry AS A PAIR: a clipboard
               viewer taking ownership in between would wipe what was
               just written, and re-reading alone would then never
               recover. This does not weaken the assertion - an
               IDM_COPY that does not reach the focused control fails
               all four attempts just as it failed one. */
            for (attempt = 0; attempt < 4 && !ok; attempt++) {
                SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_COPY, 0), 0);
                if (!st_clip_open(hwnd)) continue;
                opened = 1;
                {
                    HANDLE hg = GetClipboardData(CF_TEXT);
                    if (hg) {
                        char *p = (char *)GlobalLock(hg);
                        if (p) ok = strstr(p, unique) != NULL;
                        GlobalUnlock(hg);
                    }
                }
                CloseClipboard();
            }
            /* Never opened it at all -> nothing was measured. Opened it
               and did not find the probe -> Copy really is broken. */
            if (!ok && !opened) {
                st_skip(f, what_copy, busy);
            } else {
                st_check(f, ok, what_copy);
                checks++;
            }
        }

        /* Collapsed first, so the check below is proof SELECT_ALL did
           the expanding rather than a leftover selection from the Copy
           check above happening to already cover the same range. */
        transcript_clear();
        transcript_push(unique, (int)strlen(unique));
        transcript_end();
        SendMessage(tr, EM_SETSEL, 0, 0);
        SetFocus(tr);
        SendMessage(hwnd, WM_COMMAND, MAKEWPARAM(IDM_SELECT_ALL, 0), 0);
        {
            DWORD start = 0, end = 0;
            int want = (int)strlen(unique);
            SendMessage(tr, EM_GETSEL, (WPARAM)&start, (LPARAM)&end);
            /* Two invariants: the control holds what was written into
               it, and the selection covers all of that. The end may sit
               past the length by at most one line ending - EM_SETSEL
               (0,-1) reports the position after the buffer's final
               break and WM_GETTEXTLENGTH does not count it, which is
               len+1 on RichEdit 2.0 (a lone CR) and len+2 on 1.0
               (CRLF). An under-selection still fails, and so does any
               real over-selection. */
            int ctrl = (int)SendMessage(tr, WM_GETTEXTLENGTH, 0, 0);
            fprintf(f, "  selall start=%d end=%d want=%d ctrl=%d\n",
                    (int)start, (int)end, want, ctrl);
            st_check(f, ctrl == want,
                     "edit: the transcript holds the whole conversation");
            checks++;
            st_check(f, (int)start == 0 &&
                        (int)end >= ctrl && (int)end <= ctrl + 2,
                     "edit: Select All selects the whole conversation");
            checks++;
        }
        transcript_clear();
    }

    /* Settings persistence, the READ side: ini -> create_children's
       temp_milli block, not the setter it calls (lz_common_settings_set_temp
       is pinned on its own elsewhere). Three things live only in that
       few lines of create_children and nowhere else - the `milli >= 0`
       guard, the /1000.0f conversion, and trusting the setter's return
       rather than the field - and none of them had a gate: deleting the
       guard, for instance, moves no PASS/FAIL line, because nothing had
       ever put a negative or out-of-range temp_milli in front of it.

       A fresh probe window per case, not the first window: create_children
       fully re-initialises g.set (lz_common_settings_init) before applying
       the ini, so there is no state to reset between cases, and reusing
       the first window's hwnd here would run into the same "still needed
       afterward" problem WM_CLOSE has below. think is pinned to 1 for
       every case so the expected default does not depend on whatever the
       developer's own ini happens to have saved. */
    {
        typedef struct { int milli; float want_temp; int want_manual;
                         const char *what; } TempCase;
        static const TempCase CASES[] = {
            { -1,      0.0f, 0, "negative is skipped, default stands" },
            { -500,    0.0f, 0, "far negative is skipped too" },
            { 500,     0.5f, 1, "in range: applied, marked manual" },
            { 1000,    1.0f, 1, "exactly the cap: accepted" },
            { 1001,    0.0f, 0, "just over the cap: refused, default stands" },
            { 2000000, 0.0f, 0, "far over the cap: refused, default stands" }
        };
        int old_think = lz_ini_get_int("think", -1);
        int old_tempm = lz_ini_get_int("temp_milli", -1);
        float def_temp = lz_common_settings_default_temp(1);
        int c;

        for (c = 0; c < (int)(sizeof CASES / sizeof CASES[0]); c++) {
            HWND wc;
            lz_ini_set_int("think", 1);
            lz_ini_set_int("temp_milli", CASES[c].milli);
            wc = CreateWindowExA(0, LZ_CLASS_NAME, "stc",
                                 WS_OVERLAPPEDWINDOW, 40, 40, 500, 400,
                                 NULL, NULL, inst, NULL);
            st_check(f, wc != NULL,
                     "ini->settings: a probe window opens"); checks++;
            if (wc) {
                float want = CASES[c].want_manual ? CASES[c].want_temp
                                                  : def_temp;
                char what[176];
                sprintf(what, "ini->settings: temp_milli=%d - %s",
                        CASES[c].milli, CASES[c].what);
                st_check(f, g.set.manual_temp == CASES[c].want_manual &&
                            g.set.temp > want - 0.0005f &&
                            g.set.temp < want + 0.0005f, what);
                if (g.set.manual_temp != CASES[c].want_manual ||
                    g.set.temp <= want - 0.0005f ||
                    g.set.temp >= want + 0.0005f)
                    fprintf(f, "  got manual=%d temp=%f, want manual=%d "
                               "temp~=%f\n", g.set.manual_temp,
                            (double)g.set.temp, CASES[c].want_manual,
                            (double)want);
                checks++;
                DestroyWindow(wc);
            }
        }
        lz_ini_set_int("think", old_think);
        lz_ini_set_int("temp_milli", old_tempm);
    }

    /* Settings persistence, the SAVE side. WM_CLOSE is where g.set
       actually reaches the ini (see the handler); nothing else in this
       process ever calls it. selftest's own teardown at the bottom of
       this function calls DestroyWindow directly, and IDM_EXIT is the
       only real caller that goes through WM_CLOSE - so without this
       block, everything downstream of that handler (the *1000+0.5f
       conversion, the IsIconic/IsZoomed skip on the geometry branch,
       mru_save) had no gate at all: break the arithmetic and no
       PASS/FAIL line moves.

       A window of its own, not the first one: WM_CLOSE destroys
       whatever it is sent to, and the rest of this function still
       needs hwnd afterward. Placed beside the floor test for the same
       reason that one is last - this also overwrites g.main and
       g.part[] through its own create_children call, so nothing after
       either block may assume the FIRST window's controls.

       The temperature is driven directly with lz_common_settings_set_temp,
       not through the ini, so this test stands only on the SAVE side -
       the READ side (ini -> g.set on the way in) is a different
       integration point with its own gate elsewhere. */
    {
        HWND w3;
        char path[MAX_PATH + 8];
        char raw[4096];
        int old_wx = lz_ini_get_int("win_x", -32768);
        int old_wy = lz_ini_get_int("win_y", -32768);
        int old_ww = lz_ini_get_int("win_w", 0);
        int old_wh = lz_ini_get_int("win_h", 0);
        int old_lang = lz_ini_get_int("lang", 0);
        int old_think = lz_ini_get_int("think", -1);
        int old_tempm = lz_ini_get_int("temp_milli", -1);

        /* Poisoned before the window exists, not read back through the
           status bar cell: there is no control to plant a sentinel in
           until WM_CREATE has already run, so unlike the finish_job
           wiring gate above, the field itself is what has to carry the
           before/after difference. 999999 is a value neither
           lz_gui_session_token_count (no model: -1; a model with empty
           history: 0) nor any context window (LZ_COMMON_CTX_MAX is 32768)
           could ever legitimately
           produce, so surviving unchanged past WM_CREATE means
           create_children's own call never ran. */
        g.ctx_tokens = 999999;
        w3 = CreateWindowExA(0, LZ_CLASS_NAME, "st3", WS_OVERLAPPEDWINDOW,
                             40, 40, 500, 400, NULL, NULL, inst, NULL);
        st_check(f, w3 != NULL,
                 "close: a window for the save-side test opens"); checks++;
        if (w3) {
            st_check(f, g.ctx_tokens == -1,
                     "status: create_children's own tail calls "
                     "update_ctx_cell");
            checks++;
        }
        if (w3) {
            int found = 0;
            lz_common_settings_set_temp(&g.set, 0.5f);
            SendMessage(w3, WM_CLOSE, 0, 0);   /* destroys w3 itself */

            if (lz_ini_path(path, (int)sizeof path)) {
                FILE *rf = fopen(path, "rb");
                if (rf) {
                    size_t n = fread(raw, 1, sizeof raw - 1, rf);
                    raw[n] = '\0';
                    fclose(rf);
                    /* Read from the file's own bytes, same reasoning as
                       the disk-persistence check above - a getter could
                       be answering from the in-process cache. */
                    found = strstr(raw, "temp_milli=500") != NULL;
                }
            }
            st_check(f, found,
                     "close: WM_CLOSE writes temp_milli matching the "
                     "live temperature (0.5 -> 500)");
            checks++;

            lz_ini_set_int("win_x", old_wx);
            lz_ini_set_int("win_y", old_wy);
            lz_ini_set_int("win_w", old_ww);
            lz_ini_set_int("win_h", old_wh);
            lz_ini_set_int("lang", old_lang);
            lz_ini_set_int("think", old_think);
            lz_ini_set_int("temp_milli", old_tempm);
            WritePrivateProfileStringA(NULL, NULL, NULL, path);
        }
    }

    /* The floor's shape, forced. Nothing here has comctl32 missing, so
       the case is exercised by building a second window with the tool
       strip suppressed - which is the only way this path gets walked on
       a machine that has the class. LAST on purpose: WM_CREATE on the
       new window runs create_children again, which overwrites g.main
       and every g.part[] with the second window's own - so nothing
       after this block may read them and expect the first window's
       controls. */
    {
        HWND w2;
        g.no_toolbar = 1;               /* read by create_toolbar */
        w2 = CreateWindowExA(0, LZ_CLASS_NAME, "st2",
                             WS_OVERLAPPEDWINDOW, 0, 0, 700, 554,
                             NULL, NULL, inst, NULL);
        g.no_toolbar = 0;
        /* Renamed from "the window opens with no tool strip at all":
           the assertion below it is only w2 != NULL, i.e. that WM_CREATE
           did not crash - it never checks that create_toolbar actually
           took the g.no_toolbar branch, so the name promises more than
           the check delivers. The intent is a smoke test (a missing
           tool strip must not stop the window from opening), so the
           scope is right; only the name overclaimed. The second check
           below closes the remaining gap cheaply: without it, w2 !=
           NULL would hold just as well if g.no_toolbar were silently
           ignored and a real tool strip got built anyway. */
        st_check(f, w2 != NULL,
                 "floor: WM_CREATE survives create_toolbar returning NULL");
        checks++;
        if (w2) {
            st_check(f, g.part[LZ_GUI_TOOLBAR] == NULL,
                     "floor: the no-toolbar branch was really taken, "
                     "not just survived");
            checks++;
        }
        if (w2) DestroyWindow(w2);
    }

    fprintf(f, "checks %d\nfails %d\nskips %d\n", checks, st_fails, st_skips);
    fclose(f);
    DestroyWindow(hwnd);
    return st_fails ? 1 : 0;
}

static void sb_compare(FILE *f, const char *path, int *checks) {
    {
        HWND strip = g.part[LZ_GUI_STATUS];
        int cw = 0, ch = 0;
        unsigned char *shot = strip ? sb_capture(strip, &cw, &ch) : NULL;
        int classic = lz_compat_classic();

        /* The mapped lamp artwork, dumped so the c0c0c0-to-button-face
           substitution can be checked rather than assumed. */
        {
            HDC ddc = CreateCompatibleDC(NULL);
            BITMAPINFO lbi;
            unsigned char *lb;
            char lpath[MAX_PATH + 40];
            memset(&lbi, 0, sizeof lbi);
            lbi.bmiHeader.biSize = sizeof lbi.bmiHeader;
            lbi.bmiHeader.biWidth = LZ_LAMP_PX;
            lbi.bmiHeader.biHeight = LZ_LAMP_PX;
            lbi.bmiHeader.biPlanes = 1;
            lbi.bmiHeader.biBitCount = 24;
            lbi.bmiHeader.biCompression = BI_RGB;
            lb = (unsigned char *)malloc((size_t)(sb_stride(LZ_LAMP_PX) *
                                                  LZ_LAMP_PX));
            if (ddc && lb && g.lamp_bmp[LZ_LAMP_OFF]) {
                if (GetDIBits(ddc, g.lamp_bmp[LZ_LAMP_OFF], 0, LZ_LAMP_PX,
                              lb, &lbi, DIB_RGB_COLORS)) {
                    sprintf(lpath, "%.*s.lamp-%s.bmp", MAX_PATH, path,
                            lz_compat_classic() ? "classic" : "native");
                    sb_bmp_write(lpath, lb, LZ_LAMP_PX, LZ_LAMP_PX);
                    fprintf(f, "  lamp image -> %s\n", lpath);
                }
            }
            if (lb) free(lb);
            if (ddc) DeleteDC(ddc);
        }
        /* The native progress bar, driven to a known position and
           captured on its own. This is the observation the 3.51
           simulation gets built against - height, border, chunk width
           and gap all have to come off a real one rather than memory. */
        /* The SIMULATED bar, rendered on its own so it can be compared
           with the native one captured just below. Driven to the same
           40% the native probe uses. */
        if (!g.progress && g.part[LZ_GUI_STATUS]) {
            RECT sr, pr;
            GetClientRect(g.part[LZ_GUI_STATUS], &sr);
            if (sb_prog_rect(sr.bottom - sr.top, &pr)) {
                int pw2 = pr.right - pr.left, ph2 = pr.bottom - pr.top;
                HDC wdc = GetDC(g.part[LZ_GUI_STATUS]);
                HDC mdc = wdc ? CreateCompatibleDC(wdc) : NULL;
                BITMAPINFO bi2;
                void *bits2 = NULL;
                HBITMAP dib2, old2;
                memset(&bi2, 0, sizeof bi2);
                bi2.bmiHeader.biSize = sizeof bi2.bmiHeader;
                bi2.bmiHeader.biWidth = pw2;
                bi2.bmiHeader.biHeight = ph2;
                bi2.bmiHeader.biPlanes = 1;
                bi2.bmiHeader.biBitCount = 24;
                bi2.bmiHeader.biCompression = BI_RGB;
                dib2 = mdc ? CreateDIBSection(wdc, &bi2, DIB_RGB_COLORS,
                                              &bits2, NULL, 0) : NULL;
                if (wdc) ReleaseDC(g.part[LZ_GUI_STATUS], wdc);
                if (dib2 && bits2) {
                    RECT local;
                    old2 = (HBITMAP)SelectObject(mdc, dib2);
                    local.left = 0; local.top = 0;
                    local.right = pw2; local.bottom = ph2;
                    /* Pre-filled, because sb_prog_paint does NOT paint
                       its own background - the strip has already filled
                       the whole client with the button face by the time
                       it runs. Without this the DIB's zero fill shows
                       through as black and the capture reports a
                       difference the screen does not have. */
                    FillRect(mdc, &local, sb_brush(COLOR_BTNFACE));
                    sb_prog_paint(mdc, &local, 40, 100);
                    GdiFlush();
                    {
                        static char qpath[MAX_PATH + 40];
                        sprintf(qpath, "%.*s.prog-sim.bmp", MAX_PATH, path);
                        sb_bmp_write(qpath, (unsigned char *)bits2, pw2, ph2);
                        fprintf(f, "  progress-sim %dx%d at 40%% -> %s\n",
                                pw2, ph2, qpath);
                    }
                    SelectObject(mdc, old2);
                    DeleteObject(dib2);
                }
                if (mdc) DeleteDC(mdc);
            }
        }
        if (g.progress) {
            RECT pr;
            int pw, ph;
            unsigned char *pshot;
            /* No Show/Hide around this: the control is visible for the
               life of the window now, and hiding it at the end of a
               probe would leave the segment empty of its own well for
               the rest of the run. Only the position is driven, and it
               is put back afterwards. */
            SendMessage(g.progress, LZ_PBM_SETRANGE, 0, MAKELPARAM(0, 100));
            SendMessage(g.progress, LZ_PBM_SETPOS, (WPARAM)40, 0);
            UpdateWindow(g.progress);
            GetClientRect(g.progress, &pr);
            pw = pr.right - pr.left; ph = pr.bottom - pr.top;
            pshot = sb_capture(g.progress, &pw, &ph);
            if (pshot) {
                static char ppath[MAX_PATH + 40];
                sprintf(ppath, "%.*s.prog-native.bmp", MAX_PATH, path);
                sb_bmp_write(ppath, pshot, pw, ph);
                fprintf(f, "  progress %dx%d at 40%% untheme=%d -> %s\n",
                        pw, ph, g.progress_untheme, ppath);
                free(pshot);
            }
            SendMessage(g.progress, LZ_PBM_SETPOS, (WPARAM)0, 0);
        }
        /* Geometry of every segment on both paths, so the simulated
           rects get aligned to comctl32's own answers rather than to an
           assumption about them. */
        {
            RECT sr, pc, pin;
            GetClientRect(g.part[LZ_GUI_STATUS], &sr);
            fprintf(f, "  strip client %dx%d p0=%d p1=%d\n",
                    (int)(sr.right - sr.left), (int)(sr.bottom - sr.top),
                    g_sb_p0, g_sb_p1);
            if (g.status_is_sbar) {
                int i;
                for (i = 0; i < 4; i++) {
                    RECT q;
                    if (SendMessage(g.part[LZ_GUI_STATUS], LZ_SB_GETRECT,
                                    (WPARAM)i, (LPARAM)&q))
                        fprintf(f, "  part%d %d,%d..%d,%d (%dx%d)\n", i,
                                (int)q.left, (int)q.top, (int)q.right,
                                (int)q.bottom, (int)(q.right - q.left),
                                (int)(q.bottom - q.top));
                }
                if (g.progress) {
                    RECT w;
                    GetWindowRect(g.progress, &w);
                    MapWindowPoints(NULL, g.part[LZ_GUI_STATUS],
                                    (POINT *)&w, 2);
                    fprintf(f, "  ctrl  %d,%d..%d,%d (%dx%d)\n",
                            (int)w.left, (int)w.top, (int)w.right,
                            (int)w.bottom, (int)(w.right - w.left),
                            (int)(w.bottom - w.top));
                }
            }
            if (sb_prog_rect(sr.bottom - sr.top, &pc)) {
                fprintf(f, "  simseg %d,%d..%d,%d (%dx%d)\n",
                        (int)pc.left, (int)pc.top, (int)pc.right,
                        (int)pc.bottom, (int)(pc.right - pc.left),
                        (int)(pc.bottom - pc.top));
                if (sb_prog_inner(&pc, &pin))
                    fprintf(f, "  simbar %d,%d..%d,%d (%dx%d)\n",
                            (int)pin.left, (int)pin.top, (int)pin.right,
                            (int)pin.bottom, (int)(pin.right - pin.left),
                            (int)(pin.bottom - pin.top));
            }
        }
            /* The engine's prefill callback has to still be on the
           session's opts by the time a job runs - anything that
           re-defaults LZGenOpts after create_children would silently
           take it off and the indicator would never be fed. */
        st_check(f, g.sess.opts.on_prefill == gui_prefill_progress,
                 "prefill: the session still carries the progress callback");
        (*checks)++;
    st_check(f, shot != NULL,
                 "statusbar: the strip renders into a supplied DC");
        (*checks)++;
        if (shot) {
            static char mine[MAX_PATH + 40], other[MAX_PATH + 40];
            sprintf(mine,  "%.*s.sb-%s.bmp", MAX_PATH, path,
                    classic ? "classic" : "native");
            sprintf(other, "%.*s.sb-%s.bmp", MAX_PATH, path,
                    classic ? "native" : "classic");
            st_check(f, sb_bmp_write(mine, shot, cw, ch),
                     "statusbar: the strip image was written");
            (*checks)++;
            fprintf(f, "  strip %s %dx%d cells %d/%d -> %s\n",
                    classic ? "simulated" : "comctl32", cw, ch,
                    g_sb_p0, g_sb_p1, mine);
            {
                int ow = 0, oh = 0;
                unsigned char *ref = sb_bmp_load(other, &ow, &oh);
                int stale = ref && !st_newer_than_exe(other);
                static char why[MAX_PATH + 80];
                if (stale) {
                    free(ref);
                    ref = NULL;
                }
                if (!ref) {
                    /* Skipped, not silently dropped: without this the
                       report's check count moved between runs depending
                       on what happened to be lying next to it, which is
                       the one thing a report meant to be diffed must
                       not do. Missing and stale get different words -
                       they need different actions from whoever reads
                       this, and one is a leftover that has to go. */
                    sprintf(why, stale
                            ? "the counterpart predates this build; run "
                              "again with classic_ui=%d to replace %.*s"
                            : "no counterpart yet; run again with "
                              "classic_ui=%d to produce %.*s",
                            classic ? 0 : 1, MAX_PATH, other);
                    st_skip(f, "statusbar: both strips span the same width",
                            why);
                } else {
                    /* Width must match - both strips span the client
                       area, and a difference there is a layout fault
                       rather than a drawing choice. Everything else is
                       reported. */
                    st_check(f, ow == cw,
                             "statusbar: both strips span the same width");
                    (*checks)++;
                    if (ow == cw && oh == ch) {
                        /* Split at the progress cell, because a
                           difference there is NOT the same fact as a
                           difference anywhere else. Everything on the
                           strip is drawn by this program on both
                           paths; the progress cell is not - the
                           comctl32 build puts a real control there,
                           and on a host with comctl32 v6 that control
                           keeps drawing its flat accent-coloured self
                           whether or not SetWindowTheme succeeded.
                           There is no v6 on the target, so that part
                           of the count says something about this
                           machine and nothing about the port; folding
                           it into one number makes the number
                           unreadable, and dropping it would hide a
                           real difference on a host where the
                           untheming does work. Report both. */
                        long n = sb_stride(cw) * ch, k, diff = 0, inprog = 0;
                        RECT pc2;
                        int have_pc = sb_prog_rect(ch, &pc2);
                        for (k = 0; k < n; k++) {
                            if (ref[k] == shot[k]) continue;
                            diff++;
                            if (have_pc) {
                                long row = k / sb_stride(cw);
                                long x   = (k % sb_stride(cw)) / 3;
                                /* Rows run bottom-up in a DIB. */
                                long y   = ch - 1 - row;
                                if (x >= pc2.left && x < pc2.right &&
                                    y >= pc2.top  && y < pc2.bottom) inprog++;
                            }
                        }
                        fprintf(f, "  pixel bytes differing: %ld / %ld "
                                   "(%.1f%%)\n", diff, n,
                                n ? (double)diff * 100.0 / (double)n : 0.0);
                        fprintf(f, "  of those, inside the progress cell: "
                                   "%ld; drawn by this program: %ld\n",
                                inprog, diff - inprog);
                    } else {
                        fprintf(f, "  heights differ: %d vs %d - compare "
                                   "the two images by eye\n", ch, oh);
                    }
                    free(ref);
                }
            }
            free(shot);
        }
    }
}

/* ------------------------------------------------------------ entry */

static const char *selftest_path(const char *cmdline) {
    /* Deliberately not CommandLineToArgvW: it lives in shell32 and is a
       wide entry point, and src/cli_main.c already records that both of
       those are unverified on Win9x. One flag with one argument does not
       need an argv parser. */
    const char *p = strstr(cmdline, "--selftest");
    if (!p) return NULL;
    p += strlen("--selftest");
    while (*p == ' ' || *p == '\t') p++;
    return *p ? p : NULL;
}

int WINAPI WinMain(HINSTANCE inst, HINSTANCE prev, LPSTR cmdline, int show) {
    MSG msg;
    HWND hwnd;
    const char *st;

    (void)prev;
    g.inst = inst;

    /* `classic_ui=1` forces the NT 3.51 appearance path here; see
       gui/compat40.h for which answers it forces.
       FIRST, ahead of the language block and the selftest: everything
       after this asks compat40 questions, and one answered before the
       switch is read is answered for the wrong system. Applying to the
       selftest too is deliberate - that is how the degraded layout gets
       assertions rather than only a screenshot. */
    lz_compat_force_classic(lz_ini_get_int("classic_ui", 0));
    /* Display-only debug aid; see g_dbg_prefill_ms. Clamped rather than
       trusted: a negative or absurd value would either never advance or
       divide the ramp into nothing. */
    g_dbg_prefill_ms = lz_ini_get_int("debug_prefill_ms", 0);
    if (g_dbg_prefill_ms < 0) g_dbg_prefill_ms = 0;
    if (g_dbg_prefill_ms > 600000) g_dbg_prefill_ms = 600000;

    /* Both tables, one language. src/err.c defaults to ENGLISH, so
       without the second call the buttons would be Chinese while the
       engine's errbuf - which this window shows verbatim in the same
       message box - is English. Both tables must follow the one
       language selected.
       Language BEFORE the window, because the menu bar and the tool
       strip take their captions at creation time - switching afterwards
       works (apply_language exists) but would build the whole bar twice
       on every start. */
    {
        int lang = lz_ini_get_int("lang", 0);
        if (lang != 0 && lang != 1) lang = 0;
        lz_str_init(lang);
        lz_set_error_lang(lang);
        /* Same reason as the comment above: before the window, because
           every control takes its font at creation time. lz_str_init's
           own convention (0 Chinese, 1 English) is exactly
           lz_ui_set_font_lang's "english" argument - no translation
           needed between the two. */
        lz_ui_set_font_lang(lang);
    }

    /* The hardware floor, before a window exists. Socket 7 is the floor
       and Socket 7 is NOT "has MMX" - the original Pentium, the K5 and
       the plain 6x86 have none - so the only hard requirements are
       CPUID and an FPU, and this says which one is missing.
       The message comes from err.c in UTF-8 and the box is ANSI, so it
       goes through the codec like every other string that crosses that
       boundary. */
    {
        static char cerr[1024];
        static char cdisp[1024];
        if (lz_cpu_check(cerr, (int)sizeof cerr) != 0) {
            lz_gbk_from_utf8(cerr, (int)strlen(cerr), cdisp,
                             (int)sizeof cdisp, NULL);
            MessageBoxA(NULL, cdisp, lz_str_display(LZ_STR_ERR_TITLE),
                        MB_ICONHAND | MB_OK);
            return 1;
        }
    }

    st = selftest_path(cmdline ? cmdline : "");
    if (st) return selftest(inst, st);

    {
        /* Up before the window is built, down after: on the target that
           span is riched20 loading plus six controls, which is the part
           of startup a person actually waits through. */
        HWND splash = lz_gui_splash_show(inst);
        if (!register_class(inst)) { lz_gui_splash_close(splash); return 1; }
        hwnd = create_main(inst, 640, 480);
        if (!hwnd) { lz_gui_splash_close(splash); return 1; }
        lz_gui_splash_close(splash);
    }
    /* Position, clamped to the work area. A saved rectangle from a
       machine with a second monitor - or from before the taskbar moved -
       otherwise puts the window somewhere it cannot be dragged back
       from, and the user's only repair is deleting the ini. */
    {
        int x = lz_ini_get_int("win_x", -32768);
        int y = lz_ini_get_int("win_y", -32768);
        int w = lz_ini_get_int("win_w", 0);
        int h = lz_ini_get_int("win_h", 0);
        RECT wa;
        if (x != -32768 && w >= LZ_GUI_MIN_CW && h >= LZ_GUI_MIN_CH &&
            SystemParametersInfoA(SPI_GETWORKAREA, 0, &wa, 0)) {
            if (w > wa.right - wa.left)  w = wa.right - wa.left;
            if (h > wa.bottom - wa.top)  h = wa.bottom - wa.top;
            if (x < wa.left)             x = wa.left;
            if (y < wa.top)              y = wa.top;
            if (x + w > wa.right)        x = wa.right - w;
            if (y + h > wa.bottom)       y = wa.bottom - h;
            MoveWindow(hwnd, x, y, w, h, FALSE);
        }
    }
    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);

    /* g.accel is loaded in create_children, so --selftest sees it too.
       IsDialogMessage(hwnd, ...) (the MAIN window's own, below) runs
       AFTER accelerator translation: it eats keystrokes for tab
       navigation and would swallow Ctrl+O before the accelerator table
       ever saw it. A NULL table (no resources linked) just means the
       branch is skipped.

       g.find_dlg's own IsDialogMessage is a DIFFERENT call
       with the opposite ordering requirement, and one that cannot be
       reasoned out from the paragraph above: that one is about the
       MAIN window, this one is about the modeless Find dialog, and
       TranslateAcceleratorA does not check msg.hwnd against the hWnd
       it is given - confirmed against a real find dialog and a real
       accelerator table:
       TranslateAcceleratorA(owner, accel, &msg) fired the table's
       WM_COMMAND even with msg.hwnd pointed at the find dialog's own
       edit control, and separately, IsDialogMessage(find_dlg, &msg)
       returned TRUE for that same plain Ctrl+O keydown - not only for
       Tab/Enter/Escape navigation keys, which is what makes checking
       it FIRST actually prevent the leak rather than just reorder it:
       the message is fully consumed before TranslateAcceleratorA ever
       runs. Skipped this check second (accelerator table, unconditional
       on msg.hwnd) would let Ctrl+O open the model-load dialog while
       the user is mid-search - silent on this host, since no window
       here refuses focus, and exactly the failure mode a dialog eating
       every keystroke or answering none looks like from outside. */
    while (GetMessage(&msg, NULL, 0, 0) > 0) {
        if (g.find_dlg && IsDialogMessage(g.find_dlg, &msg)) continue;
        if (g.accel && TranslateAcceleratorA(hwnd, g.accel, &msg)) continue;
        if (!IsDialogMessage(hwnd, &msg)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
    return (int)msg.wParam;
}
