#ifndef LZ_GUI_COMPAT40_H
#define LZ_GUI_COMPAT40_H

#include <windows.h>

/* Everything the front end does differently on Windows NT 3.51.
 *
 * The floor is 3.51; the LOOK is 4.0. Those are not in conflict as long
 * as every 4.0-era nicety is asked for at run time and degrades when the
 * answer is no - which is the whole point of gathering them here rather
 * than sprinkling GetVersion() through the window code. This file is
 * also the honest list of what 3.51 actually loses.
 *
 * WHAT THE BUILD-TIME FLOOR CAN AND CANNOT SEE
 * --------------------------------------------
 * build/watcom/winver.sh compiles at the floor, and the compiler hides
 * anything newer - as a W131 "no prototype found", which -we turns
 * fatal. That catches FUNCTIONS.
 *
 * It cannot catch CONSTANTS. A window style or a message number that
 * the headers hide is just an undeclared identifier, and the moment
 * this file supplies it the gate has nothing left to object to. So the
 * supplied set below is deliberately small, each entry says what
 * happens on 3.51, and this header declares NO functions - a
 * self-declared prototype would silently undo
 * the floor for the one thing the floor is good at.
 */

/* ---- constants the 3.51 headers do not have ----
 * Values are the 4.0 ones. On 3.51 they are either ignored by the API
 * that receives them (the WS_EX_* bits, TPM_BOTTOMALIGN) or belong to a
 * message range this program owns outright (WM_APP). */

#ifndef WM_APP
/* Our own window class's private range. 3.51 has WM_USER but not the
 * WM_APP block; the number is not special to the system either way. */
#define WM_APP 0x8000
#endif

#ifndef WS_EX_CLIENTEDGE
#define WS_EX_CLIENTEDGE 0x00000200L   /* 3.51: no sunken border */
#endif

#ifndef WS_EX_STATICEDGE
#define WS_EX_STATICEDGE 0x00020000L   /* 3.51: no etched border */
#endif

#ifndef TPM_BOTTOMALIGN
/* 3.51 ignores it and opens the menu downward - off the bottom edge -
 * at which point Windows repositions a popup that does not fit. The
 * flag makes it deliberate rather than lucky. */
#define TPM_BOTTOMALIGN 0x0020L
#endif

#ifndef DEFAULT_GUI_FONT
#define DEFAULT_GUI_FONT 17            /* 3.51: absent, see lz_ui_font */
#endif

/* BS_ICON + BM_SETIMAGE/BM_GETIMAGE: the menu button is the chicken.
   These are 4.0 additions; on 3.51 the style bit is
   ignored by the button (it renders its label text instead) and the
   messages go unanswered (SendMessage returns 0), so the floor gets a
   plain labelled button - a graceful degradation, not a missing
   control. LR_DEFAULTCOLOR is 0x0000, so LoadImage is called with a
   plain 0 where the 4.0 name would be. */
#ifndef BS_ICON
#define BS_ICON 0x0040L
#endif

/* SS_BITMAP / STM_SETIMAGE: both are Win3.1-era (the sidebar chicken
   uses them), but Watcom's winuser.h hides every SS_* behind
   `WINVER >= 0x0400`. The values are stable; a 3.51 static control
   answers STM_SETIMAGE and draws the bitmap exactly like a 4.0 one. */
#ifndef SS_BITMAP
#define SS_BITMAP 0x0000000EL
#endif
#ifndef STM_SETIMAGE
#define STM_SETIMAGE 0x0172
#endif
#ifndef STM_GETIMAGE
#define STM_GETIMAGE 0x0173
#endif

/* Standard status bar (comctl32's msctls_statusbar32). SB_SETTEXT /
   SB_SETPARTS are commctrl messages the 3.51 headers hide; the values
   are stable and a non-statusbar control ignores them (returns 0), so
   the STATIC fallback path simply never sends them.
   Named LZ_* rather than SB_* on purpose: compat40.c compiles at the
   4.0 level where commctrl.h defines SB_SETTEXT as SB_SETTEXTA, and a
   same-name fallback would collide with it (W140 under -we, which the
   build treats as fatal). */
#define LZ_SB_SETTEXT  (WM_USER + 1)
/* SB_GETTEXTA, same reasoning as LZ_SB_SETTEXT above - the one this
   file needs because a part other than 0 carries real content:
   GetWindowTextA's own answer for a status bar is part 0's
   text (confirmed against the running window elsewhere in this file),
   so reading part 1 back needs the message, not the window text. */
#define LZ_SB_GETTEXT  (WM_USER + 2)
#define LZ_SB_SETPARTS (WM_USER + 4)
/* SBT_NOBORDERS, OR'd into SB_SETTEXT's wParam beside the part index:
   that part is drawn with no sunken edge. Used for the cell that holds
   the two lamps - a bitmap sitting in a sunken well reads as a hole in
   the bar, and Win9x put its own status-bar indicators flat. It is a
   flag on the message, not a style, so a status bar that predates it
   ignores the high bits and draws the border, which is the old
   appearance and not a failure. */
#define LZ_SBT_NOBORDERS 0x0100
/* SB_GETRECT: the bar's own answer for where a part lives, in the bar's
   client coordinates. Worth asking rather than deriving from the client
   rect, because the part is inset from it by the bar's top border and by
   its own sunken edge, and those insets are the bar's business - deriving
   them is how the lamps came to sit on top of the border. */
#define LZ_SB_GETRECT  (WM_USER + 10)
#ifndef BM_SETIMAGE
#define BM_SETIMAGE 0x00F7
#endif
#ifndef BM_GETIMAGE
#define BM_GETIMAGE 0x00F8
#endif

/* BM_GETCHECK / BM_SETCHECK exist back to 3.1 and take these values;
 * only the NAMES are 4.0. Supplied here so the settings window can use
 * the 4.0 names at the 3.51 floor - the same file compiles clean on
 * gcc, which is the whole argument for having the floor at all. */
/* WM_CAPTURECHANGED tells a window that somebody else took the mouse -
   an Alt-Tab or a message box in the middle of a splitter drag. 3.51
   never sends it, so the case is dead code there and a drag that loses
   the capture ends the ordinary way, on the button-up that follows.
   The cost on the floor is a tracking bar left on screen until the next
   repaint, not a wedged window. */
#ifndef WM_CAPTURECHANGED
#define WM_CAPTURECHANGED 0x0215
#endif

#ifndef BST_UNCHECKED
#define BST_UNCHECKED 0x0000
#endif
#ifndef BST_CHECKED
#define BST_CHECKED   0x0001
#endif

/* MB_ICONERROR / MB_ICONWARNING are 4.0 SPELLINGS of MB_ICONHAND and
 * MB_ICONEXCLAMATION, same values, and the older names exist back to
 * 3.1. Nothing to supply - the front end just uses the old names. */

/* SPI_GETWORKAREA: SystemParametersInfoA itself is 3.1-era and needs no
 * fallback here, but Watcom's winuser.h hides this uiAction value behind
 * `WINVER >= 0x0400` because the "work area" (screen minus taskbar) is a
 * Windows 95 shell concept - NT 3.51's Program Manager has none. On 3.51
 * the call itself is safe (the function exists); it simply answers FALSE
 * for an action it does not recognise, and the caller already treats a
 * FALSE return as "skip the clamp, use the window's default placement" -
 * the same shape of degradation as every other entry above. */
#ifndef SPI_GETWORKAREA
#define SPI_GETWORKAREA 0x0030
#endif

#ifndef WM_DROPFILES
/* 3.1-era shell drag/drop notification. NT 3.51 has no Explorer-style
 * drag source to send it, so on that floor this branch is simply never
 * reached - the window just never receives the message, not a crash
 * and not a missing constant at runtime. */
#define WM_DROPFILES 0x0233
#endif

/* EM_FINDTEXT's wParam bits. Not a 3.51-vs-4.0 gap like the
 * rest of this section - richedit.h supplies EM_FINDTEXT itself and
 * gui/main.c already includes it - but these two bits are commdlg.h's
 * FR_DOWN/FR_MATCHCASE, defined there and nowhere richedit.h reaches,
 * and commdlg.h itself cannot compile at the 3.51 floor (see this
 * file's own top comment: it drags in prsht.h, which the floor hides).
 * The values are not a coincidence: RichEdit's EM_FINDTEXT deliberately
 * reuses the FINDREPLACE dialog's own Flags bits so a dialog's Flags
 * can be handed to EM_FINDTEXT with no translation - which is exactly
 * what lz_find_parse below does. */
#ifndef FR_DOWN
#define FR_DOWN      0x0001L
#endif
#ifndef FR_MATCHCASE
#define FR_MATCHCASE 0x0004L
#endif

/* DrawEdge's styles and flags. A genuine 3.51-vs-4.0 gap: the whole
 * function is a Windows 95 addition, so winuser.h hides both the
 * prototype and these constants behind `WINVER >= 0x0400` (the values
 * here are copied from that same block, not from memory). The call
 * itself goes through lz_draw_edge below - a constant may be
 * hand-declared here, a prototype may not.
 *
 * Only the four spellings the front end actually uses are given, and
 * the composites are written out of their parts the way the SDK writes
 * them, so a reader can check them against winuser.h line for line
 * rather than against four bare magic numbers. */
#ifndef BDR_SUNKENOUTER
#define LZ_BDR_SUNKENOUTER 0x0002
#define LZ_BDR_RAISEDINNER 0x0004
#define LZ_EDGE_ETCHED (LZ_BDR_SUNKENOUTER | LZ_BDR_RAISEDINNER)
#define LZ_BF_LEFT     0x0001
#define LZ_BF_TOP      0x0002
#define LZ_BF_RIGHT    0x0004
#define LZ_BF_BOTTOM   0x0008
#define LZ_BF_RECT     (LZ_BF_LEFT | LZ_BF_TOP | LZ_BF_RIGHT | LZ_BF_BOTTOM)
#else
#define LZ_BDR_SUNKENOUTER BDR_SUNKENOUTER
#define LZ_BDR_RAISEDINNER BDR_RAISEDINNER
#define LZ_EDGE_ETCHED     EDGE_ETCHED
#define LZ_BF_LEFT         BF_LEFT
#define LZ_BF_TOP          BF_TOP
#define LZ_BF_RIGHT        BF_RIGHT
#define LZ_BF_BOTTOM       BF_BOTTOM
#define LZ_BF_RECT         BF_RECT
#endif

/* Gradient caption colours - a 4.0-era addition the 3.51 headers do
 * not name. Constants, which this header may supply (a hidden FUNCTION
 * prototype is the thing it must not); GetSysColor answers 0 for an
 * index the running system does not know, which is what callers test. */
#ifndef COLOR_GRADIENTACTIVECAPTION
#define LZ_COLOR_GRAD_ACTIVE   27
#define LZ_COLOR_GRAD_INACTIVE 28
#else
#define LZ_COLOR_GRAD_ACTIVE   COLOR_GRADIENTACTIVECAPTION
#define LZ_COLOR_GRAD_INACTIVE COLOR_GRADIENTINACTIVECAPTION
#endif

/* ---- run-time capability ---- */

/* Major version of the running system: 3 on NT 3.51, 4 on 95/98/ME/NT4.
 * GetVersion, not GetVersionEx: the latter is itself 3.51-and-later, and
 * asking a capability question with a call that might not exist is the
 * shape of bug this file is for. */
int lz_os_major(void);

/* The calling thread's active keyboard layout - GetKeyboardLayout(0),
 * routed through here because Watcom's winuser.h hides the
 * declaration below WINVER 0x0400 despite the export itself
 * predating Win32. See gui/compat40.c's own comment on
 * GetKeyboardLayoutFn for why this is not a hand-declared prototype.
 * Callers save and restore a thread's layout around a deliberate,
 * temporary switch (gui/main.c's find selftest) - never
 * to test a capability, so it has no "is this available" counterpart
 * the way lz_os_major does; a NULL is just "nothing to restore to". */
HKL lz_kbd_layout_get(void);

/* 1 when the 4.0 window furniture is available. */
int lz_os_has_40(void);

/* ---- forced downgrade, for looking at the 3.51 path on a host ----
 *
 * Makes every APPEARANCE capability here answer the way NT 3.51 does, on
 * any system. Off unless turned on; gui/main.c reads kunkun98.ini's
 * `classic_ui` at startup. The degraded window's appearance cannot be
 * judged without being looked at, and the floor's hardware is not on the
 * desk.
 *
 * Forces, and nothing else:
 *   lz_os_has_40()        0
 *   lz_statusbar_class()  NULL, so the status bar takes its fallback
 *                         (which also removes the toolbar)
 *   lz_draw_edge()        draws nothing
 *   the file dialogs      3.1 style, and lz_pick_folder skips
 *                         SHBrowseForFolder for its fallback branch
 *   lz_richedit_class()   riched32 ("RichEdit", 1.0), the floor's own
 *
 * The RichEdit class is the one FUNCTIONAL capability forced, because
 * the transcript is where the floor's differences actually show and
 * every message this front end sends exists in 1.0 (see that function's
 * own declaration below). The IME is left alone: it changes behaviour
 * rather than appearance, and an observation made through a knob that
 * changed behaviour says nothing about this program.
 *
 * Reproduces the ANSWERS, not the system: a 3.51 comdlg32 is still a
 * different binary, and only the machine settles what it does. */
void lz_compat_force_classic(int on);
int  lz_compat_classic(void);

/* DrawEdge, resolved from user32 at run time. Returns 0 and draws
 * nothing when the system does not have it (NT 3.51), which is the
 * ordinary degradation for this file: the toolbar's dock groove and the
 * status bar's panel bevels simply do not appear, and everything they
 * separate still has its own position and its own background. Nothing
 * downstream depends on the return value; it is here so a caller CAN
 * tell, not because one currently does.
 *
 * Every 4.0 call in this front end has to be reachable only this way -
 * a direct DrawEdge call compiles clean on gcc. */
int lz_draw_edge(HDC dc, RECT *rc, unsigned edge, unsigned flags);

/* ---- title-bar self-drawing's four 4.0-era capabilities ----
 *
 * All four are hidden behind WINVER=0x0351 in the headers, or are 4.0-era
 * additions whose export is genuinely absent on NT 3.51, so they are
 * probed at run time through this file; the dynamic lookup lives only here. */

/* The system caption font's LOGFONT. Returns 1 on success; 0 when
 * SPI_GETNONCLIENTMETRICS does not exist (3.51), in which case the caller
 * falls back to GetStockObject(DEFAULT_GUI_FONT) - MSO95 has the same
 * fallback (FUN_5062515c). */
int lz_caption_logfont(LOGFONTA *out);

/* One caption button. which: 0 close / 1 maximize / 2 minimize.
 * Returns 1 when drawn; 0 when DrawFrameControl does not exist (3.51), in
 * which case the caller does not paint buttons and lets DefWindowProc do
 * it - 3.51's system buttons are already classic, so there is no "retro
 * text next to modern buttons" half-way there. */
int lz_caption_button(HDC dc, const RECT *r, int which,
                      int pushed, int inactive);

/* Display colour depth flag (MSO95's DAT_506d04bc). 1 when > 8bpp. It
 * decides three things: 4-vs-8-pixel gradient bands, whether the Office
 * scheme applies to inactive windows, and whether the bright-flip can
 * trigger. GetDeviceCaps itself is not 4.0-era; this wrapper is here only
 * to keep "asking the system about its capabilities" in one file. */
int lz_caption_hicolor(void);

/* The caption icon at SM_CXSMICON size. DrawIconEx is a 4.0-era function
 * (3.51 only has DrawIcon, which cannot size), hidden behind the floor, so
 * it is routed here. Returns 1 when drawn, 0 when not available or no icon -
 * the caller then leaves the icon slot empty, which is the same graceful
 * degradation as every other 4.0 nicety in this file. */
int lz_caption_icon(HDC dc, HICON ic, int x, int y, int size);

/* The caption button width, asked of the frame.
 *
 * SM_CXSIZE is documented as "title bar button width", and on Win9x it is;
 * on modern frames it lies - this box reports 36 while the frame's real
 * buttons are 24 wide. Drawing at 36 puts the buttons where they do not
 * actually hit (Word 95 itself is caught by this). Only the WIDTH comes
 * from here - height, top edge, right alignment and the close-gap keep
 * MSO95's metrics, which still mean what they always meant. Win9x does not
 * answer WM_GETTITLEBARINFOEX, the struct stays zero, and it falls back to
 * SM_CXSIZE - the correct answer there.
 *
 * This is a message, not an exported function, so no dynamic lookup; it is
 * here because "can this platform answer" is the same question the rest of
 * this file owns. */
int lz_caption_button_width(HWND h);

/* The UI font. Never NULL.
 *
 * Chinese: SimSun 9pt, asked for BY NAME rather than taken from
 * DEFAULT_GUI_FONT. The stock object is whatever the running system
 * decided, and the two systems this program meets disagree: Chinese
 * Windows 98 puts SimSun 9pt there - the font gui/layout.h's row
 * heights were derived from - while a modern host puts MS Shell Dlg
 * there, and RichEdit ignores it entirely in favour of its own
 * fixed-pitch default. Laying the window out for one font and letting
 * the system pick another is a mismatch nothing reports; it just looks
 * wrong.
 *
 * English: the stock object, because MS Sans Serif is what a Win9x
 * English system puts there, and demanding SimSun on a machine with no
 * Chinese font installed produces a substitution nobody chose.
 *
 * On 3.51 GetStockObject returns NULL for DEFAULT_GUI_FONT, and the
 * fallback chain below it is the API's own answer, not a version test. */
HFONT lz_ui_font(void);

/* Rebuild the UI font for a language. Cheap and idempotent - returns
 * immediately when the language has not changed.
 *
 * Must be called BEFORE the controls are created (they take the font at
 * creation) and again from the language switch, which re-sends
 * WM_SETFONT to every child. Not called automatically from lz_str_init
 * because this file is the platform layer and does not depend on the
 * string table; the default here is Chinese, matching lz_str_init's. */
void lz_ui_set_font_lang(int english);

/* Make a RichEdit actually use a font, which WM_SETFONT alone does not
 * do: IMF_AUTOFONT lets it substitute a face per character according to
 * script, so a Chinese font produces SimSun for the Chinese and a
 * fixed-pitch fallback for every ASCII run on the same line. This turns
 * that off and states the format for all text. Safe with NULL. */
void lz_richedit_use_font(HWND h, HFONT f);

/* The menu button's chicken icon at 16x16, or NULL when the system has
   no LoadImage (NT 3.51) - the caller then leaves the button as plain
   labelled text. Win95+ always answers. */
HICON lz_ui_icon_16(HINSTANCE inst);

/* The standard status bar class name ("msctls_statusbar32"), or NULL
   when comctl32's InitCommonControlsEx is unavailable (then the caller
   falls back to a STATIC control with the same text). comctl32 is
   loaded by hand, never imported: Win95/NT4 carry it, but the floor
   must not assume it. */
const char *lz_statusbar_class(void);

/* Strip extended styles the running system does not understand. Unknown
 * bits are ignored in practice, but "in practice" is not a thing to
 * build on when the alternative is one AND. */
DWORD lz_ex_style(DWORD want);

/* The progress-bar class name, or NULL where comctl32 has none (the
 * 3.51 floor, and any host under classic_ui). Registered by the same
 * InitCommonControlsEx call the status bar uses, so asking for one
 * implies the other is available too.
 * PBM_SETRANGE/PBM_SETPOS are commctrl messages the 3.51 headers hide;
 * the values are stable and named LZ_* here for the same reason
 * LZ_SB_SETTEXT is - compat40.c compiles at the 4.0 level where
 * commctrl.h already defines the real names. */
const char *lz_progress_class(void);
#define LZ_PBM_SETRANGE (WM_USER + 1)
#define LZ_PBM_SETPOS   (WM_USER + 2)

/* Register/locate a rich edit control and return its class name.
 * riched20 ("RichEdit20A") when present, else riched32 ("RichEdit",
 * RichEdit 1.0) - which is all NT 3.51 ships. Every message this front
 * end sends (EM_EXLIMITTEXT, EM_EXSETSEL, EM_SETCHARFORMAT,
 * EM_GETCHARFORMAT, EM_REPLACESEL) exists in 1.0.
 * Returns NULL when neither loads. */
const char *lz_richedit_class(void);

/* Choose a directory. SHBrowseForFolder when shell32 has it; otherwise
 * a file dialog over `filter_file` whose directory is returned, because
 * 3.51's shell32 predates the folder browser entirely.
 * Returns 1 on success. */
int lz_pick_folder(HWND owner, const char *title, const char *filter_desc,
                   const char *filter_file, char *out, int cap);

/* Choose a file to write. GetSaveFileName is 3.1-era and needs no
 * fallback; it lives here because commdlg.h is one of the headers that
 * will not compile at the floor, and this is the file that raises it.
 * Returns 1 on success. */
int lz_pick_save_file(HWND owner, const char *title, const char *filter_desc,
                      const char *filter_pattern, const char *default_ext,
                      char *out, int cap);

/* Choose a file to open. GetOpenFileName is 3.1-era, same as
 * GetSaveFileName above - and it costs no new import table entry:
 * lz_pick_folder's own NT 3.51 fallback (no SHBrowseForFolder) already
 * calls it, so the symbol is statically linked either way.
 * Returns 1 on success. */
int lz_pick_open_file(HWND owner, const char *title, const char *filter_desc,
                      const char *filter_pattern, char *out, int cap);

/* Load a BITMAP resource with comctl32's colour mapping applied:
 * c0c0c0 becomes the user's COLOR_BTNFACE, 808080 the button shadow,
 * ffffff the button highlight. Artwork drawn against the Win9x button
 * face therefore sits on whatever face the running system has - and
 * that is not c0c0c0 anywhere after Windows 2000.
 * Falls back to LoadBitmap where comctl32 has no such export. */
HBITMAP lz_mapped_bitmap(HINSTANCE inst, int id);

/* Turn XP+ theming off for one window, of ANY class - not only edits.
 *
 * The name is lz_ui_untheme, not lz_edit_untheme, because the call is
 * not edit-specific: user32's EDIT gets a themed flat border under the
 * common-controls 6.0 manifest, while the conversation is a RichEdit -
 * riched20 is neither user32 nor comctl32 and draws its own classic
 * sunken edge, which uxtheme cannot override. So the edit comes down to
 * the RichEdit's BORDER rather than the other way round, which is also
 * the direction the whole front end is going. The settings dialog needs
 * the SAME call on a SCROLLBAR and a BUTTON (checkbox and pushbuttons)
 * too - a name that says "edit" invites exactly the "buttons don't need
 * this" reasoning that is precisely wrong here.
 *
 * NOT the whole control, though, for the RichEdit case specifically:
 * the RichEdit is created with WS_VSCROLL, a standard Windows
 * scrollbar living in the NONCLIENT area - uxtheme paints that
 * generically for any window carrying the style, and riched20's own
 * client-area drawing (the border, above) has no say in it. Not
 * verified with a screenshot - this environment has no interactive
 * desktop to take one on - but SetWindowTheme(h, "", "") is not
 * RichEdit-specific and reports success on this control the same way
 * it does on the input box (gui/main.c's selftest checks exactly that
 * pairing), which is what a control whose scrollbar genuinely had no
 * theme to remove would not do. This same call classic-ifies that
 * scrollbar too - see gui/main.c's call site on the transcript
 * control.
 *
 * No-op on the target: uxtheme.dll is XP and later.
 *
 * No themed-border subclassing path exists for the RichEdit's border:
 * subclassing it and painting EP_EDITBORDER over its frame on
 * WM_NCPAINT is the documented way to give it a themed border; built
 * behind a flag and measured, it changed nothing - the conversation's
 * left edge stayed 190/139/124 while the untheme'd edit matched it
 * exactly. One measurement on one host is not proof it cannot be done,
 * but it is enough not to keep a code path whose name promises what it
 * does not deliver.
 *
 * Returns 1 if uxtheme's SetWindowTheme was reached and reported
 * success, 0 if uxtheme is absent (the Windows 9x/NT 3.51 floor, where
 * "this does nothing" is correct) or the call itself failed - the
 * return lets two call sites be told apart instead of one silently
 * stopping, the same gap gui/compat40.c's lz_drop_accept closes for
 * DragAcceptFiles. */
int lz_ui_untheme(HWND h);


/* Give an edit control the left and right margins a dialog-template
 * edit gets for free.
 *
 * An EDIT created by CreateWindow has margins of ZERO; a dialog
 * template asks for EC_USEFONTINFO without anyone noticing, which is
 * why edits look right in programs that use one and cramped in the ones
 * that do not. This front end builds its controls by hand, so it has to
 * ask.
 *
 * EM_SETMARGINS is 4.0; on 3.51 the message goes unanswered and the
 * text sits against the frame, which is what that system's own edits
 * did.
 *
 * This is a correctness-of-appearance change, NOT the fix for the
 * ").80" review-screenshot artifact (the text caret photographed on
 * top of the first digit). */
void lz_edit_use_font_margins(HWND edit);

/* Is an IME mid-composition? 0 when imm32 is absent, which is the
 * correct answer on 3.51: there is no IMM there to be composing. */
int lz_ime_composing(HWND h);

/* Is the view parked at the end of a scrolling control? Uses
 * GetScrollInfo when available; on 3.51 falls back to comparing
 * GetScrollPos against GetScrollRange's maximum, which is coarser -
 * it cannot see the page size - but answers the same question for a
 * control whose range is set the usual way.
 *
 * The 3.51 branch is NOT the same shape and is deliberately left alone:
 * without a page there is no proportional bar, so GetScrollRange's
 * maximum really is the maximum POSITION and comparing against it is
 * right. The bug fixed in lz_scroll_range_at_end is specific to a
 * proportional bar reporting a content extent with the page zeroed.
 * There is no NT 3.51 here to measure that on, so it is recorded as
 * unmeasured rather than "fixed" alongside. */
int lz_scroll_at_end(HWND h);

/* The page-based half of the above, taking the numbers instead of the
 * window so a gate can feed it tuples read off the running program.
 * Returns 1 when the view is at the end, which includes "the text fits,
 * there is nothing to scroll". See the comment on the definition for the
 * measurement that made this a separate function. */
int lz_scroll_range_at_end(int max, unsigned page, int pos);

/* Drag and drop, through shell32 at run time.
 *
 * NOT a static import: the deliverable's import table is four DLLs.
 * shell32 is already
 * loaded by hand here for the folder browser, so this is one more
 * dynamically-resolved symbol on a module this file already owns.
 *
 * 0 when the system has no such export - the window then simply does
 * not accept drops, which is what NT 3.51 did. */
int  lz_drop_accept(HWND h, int on);
/* Path of the first dropped item. Returns 0 when there is none (no
 * such export, or the drop carried nothing DragQueryFileA could read). */
int  lz_drop_first_path(WPARAM hdrop, char *out, int cap);
void lz_drop_finish(WPARAM hdrop);

/* What was dropped -> which directory to open.
 *
 * Both forms reach here in practice: people drag the folder, and people
 * drag the file they can see inside it. GetFileAttributesA answers which
 * one this is; when it cannot (the item vanished between the drop and
 * this call, or the path never existed), fall back to "has a
 * model.bin tail" rather than guessing, because a directory that no
 * longer exists is refused by lz_gui_model_dir_ok two lines later
 * anyway - lz_drop_dir_of only has to produce a plausible directory
 * string, not confirm one exists.
 *
 * Exported (not a gui/main.c static) because it is the only logic in
 * this Task that can be verified without a window.
 * GetFileAttributesA is kernel32, so this needs
 * no probe() and no dynamic resolution; only the three functions above
 * do.
 *
 * Returns 1 when out holds a directory. */
int  lz_drop_dir_of(const char *path, char *out, int cap);

/* Find in the conversation, via comdlg32's FindTextA.
 *
 * Both functions exist so gui/main.c never has to name FINDREPLACEA:
 * commdlg.h - the header that declares it, and FR_DIALOGTERM/
 * FR_FINDNEXT with it - cannot compile at the 3.51 floor gui/main.c
 * builds at (same reason as lz_pick_open_file's own header comment).
 * This file already raises WINVER and already includes commdlg.h for
 * the file-dialog wrappers above, so it is the natural home for this
 * one too - a STATIC import, unlike the drag/drop functions above:
 * FindTextA is 3.1-era comdlg32, and comdlg32 is already one of the
 * four DLLs this deliverable's import table allows.
 *
 * lz_find_open: opens the dialog. `needle_buf`/`needle_cap` name the
 * CALLER's own persistent buffer - comdlg32 writes the user's typed
 * search text directly into it for as long as the dialog stays open,
 * so it must outlive this call, which is why it is not owned here.
 * Returns the dialog's HWND, or NULL on failure. */
HWND lz_find_open(HWND owner, char *needle_buf, int needle_cap);

/* Interprets one FINDMSGSTRING message's lParam (a FINDREPLACEA*).
 * Returns 0 for a message this caller has nothing to do for, 1 for
 * FR_DIALOGTERM (the dialog is closing - the caller should forget its
 * HWND), 2 for FR_FINDNEXT (down/match_case/needle are filled in;
 * `needle` points into the SAME buffer lz_find_open was given, valid
 * only until the next message from this dialog). */
int lz_find_parse(LPARAM lp, int *down, int *match_case,
                  const char **needle);

#endif
