#ifndef LZ_GUI_SETTINGSDLG_H
#define LZ_GUI_SETTINGSDLG_H

#include <windows.h>

#include "settings.h"

/* The inference settings window - the menu entry between "clear
 * conversation" and "about".
 *
 * A plain popup with a hand-run modal loop rather than DialogBoxIndirect.
 * A dialog template is always UNICODE even for the ANSI entry points -
 * strings, class names, the lot - and building one in memory at the
 * 3.51 floor, for two toolchains, to be verified on hardware nobody
 * here has, is a lot of risk for a window with four controls. Everything
 * this way round is something the main window already does.
 *
 * The RULES are not here. gui/settings.c owns them and is tested on its
 * own; this file is the part that cannot be tested without a window, and
 * it is deliberately the smaller half.
 */

/* Run it. Returns 1 when the user accepted, in which case *set has been
 * updated; 0 on cancel, with *set untouched. */
int lz_gui_settings_dialog(HWND owner, HINSTANCE inst, LZGuiSettings *set);

/* Create the window without running the loop, and read the controls back
 * into a settings struct. Split out so the selftest can build the thing,
 * look at it, and destroy it: the modal loop is then the only part with
 * no gate, and it is four lines. */
HWND lz_gui_settings_dialog_create(HWND owner, HINSTANCE inst,
                                   const LZGuiSettings *set);

/* Read the controls. Returns 0 on success; non-zero when one of the two
 * value boxes does not hold an acceptable number, in which case *out is
 * untouched - the same refusal gui/settings.c makes, surfaced where the
 * user typed it. WHICH box is the LZGuiSettingsBad code below. Distinct
 * codes rather than a bare 1 because the caller names the offending
 * field in the message box, and a dialog that reports the wrong field
 * sends the user to correct a value that was fine.
 *
 * `cur` supplies the fields this dialog does NOT show. It is a
 * parameter rather than a documented precondition on *out so the fields
 * this dialog does not own always come from a real source - the
 * compiler demands them, and the next field added to LZGuiSettings
 * inherits the right behaviour by default instead of the wrong one.
 * Passing the caller's live settings is always correct; a caller
 * that genuinely has none (the dialog's own WM_COMMAND handlers) passes
 * an lz_gui_settings_init'd struct and says why. */
int lz_gui_settings_dialog_read(HWND dlg, const LZGuiSettings *cur,
                                LZGuiSettings *out);

/* lz_gui_settings_dialog_read's return. The numbers matter: gui/main.c's
 * selftest asserts on them by value (it cannot see this enum's name any
 * more than it can see the control ids), so entries are APPENDED. */
typedef enum {
    LZ_GUI_SET_OK = 0,
    LZ_GUI_SET_BAD_TEMP,
    LZ_GUI_SET_BAD_CTX,
    LZ_GUI_SET_BAD_TOPP,
    LZ_GUI_SET_BAD_REP,
    LZ_GUI_SET_BAD_MAXNEW,
    LZ_GUI_SET_BAD_THINK_TEMP   /* appended, not inserted - see above */
} LZGuiSettingsBad;

/* Format a temperature for the edit box. Kept beside the parser so the
 * two agree: a box that shows what its own reader cannot accept is a
 * dialog that refuses the value it just offered. */
void lz_gui_format_temp(float t, char *out, int cap);

/* Temperature <-> horizontal scrollbar position (0..100, SBS_HORZ's own
 * range for a two-decimal value capped at LZ_GUI_TEMP_MAX). Exported so
 * the mapping can be exercised without a window - the
 * same reason lz_gui_format_temp lives here. Both clamp: an out-of-range
 * argument produces an in-range result rather than propagating it. */
int   lz_gui_temp_to_scroll(float t);
float lz_gui_scroll_to_temp(int pos);

/* lz_ui_untheme's own return (gui/compat40.h), for each control
 * lz_gui_settings_dialog_create calls it on - not a visible control
 * state anything else can read back, so the selftest wiring gate needs
 * somewhere to read it from, the same reason gui/main.c's
 * g.transcript_untheme_ok/g.input_untheme_ok exist. One dialog is ever
 * open at a time by this file's own design (g_result is already the
 * precedent for "state describing the single live instance"), so "the
 * last dialog lz_gui_settings_dialog_create built" is unambiguous. */
typedef struct {
    int think;
    int beep;                   /* the reply-finished checkbox */
    int think_temp, think_temp_scroll;
    int temp, temp_scroll;
    int topp, topp_scroll;
    int rep, rep_scroll;
    int ctx;                    /* box only - no slider */
    int maxnew;                 /* no slider - see settingsdlg.c */
    int sys;                    /* the custom system prompt box */
    int ok, cancel, restore;
} LZGuiSettingsUntheme;
LZGuiSettingsUntheme lz_gui_settings_last_untheme(void);

#endif
