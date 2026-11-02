#ifndef LZ_GUI_MUSICDLG_H
#define LZ_GUI_MUSICDLG_H

#include <windows.h>

#include "music.h"

/* The sound window.
 *
 * Its OWN window rather than more rows on the settings dialog, which
 * the owner asked for and which the two dialogs' subjects justify:
 * everything on the settings dialog is about what the MODEL does -
 * temperature, context, how many tokens - and how loud the machine is
 * while it does it is not one of those. Putting them together would
 * also have made a tall dialog taller than the 3.51 floor's smallest
 * supported screen.
 *
 * Built exactly like gui/settingsdlg.h describes: a plain popup with a
 * hand-run modal loop rather than DialogBoxIndirect, for the same
 * reason - a dialog template is always UNICODE even for the ANSI entry
 * points, and building one in memory at the floor, for two toolchains,
 * to be verified on hardware nobody here has, is a lot of risk for a
 * window with a dozen controls.
 *
 * The RULES are not here either. common/music.c owns them and is
 * tested on its own; this file is the part that cannot be tested
 * without a window.
 */

/* Run it. Returns 1 when the user accepted, in which case *set has
 * been updated; 0 on cancel, with *set untouched. */
int lz_gui_music_dialog(HWND owner, HINSTANCE inst, LZMusicSettings *set);

/* Create the window without running the loop. Split out for the same
 * reason settingsdlg's is: the selftest can build the thing, look at
 * it and destroy it, which leaves the modal loop as the only part
 * with no gate. */
HWND lz_gui_music_dialog_create(HWND owner, HINSTANCE inst,
                                const LZMusicSettings *set);

/* Read the controls back. Returns 0 on success; non-zero when the
 * baseline box does not hold a number this accepts, in which case
 * *out is untouched - the same refusal common/music.c makes, surfaced
 * where the user typed it. */
#define LZ_GUI_MUSIC_OK       0
#define LZ_GUI_MUSIC_BAD_BASE 1
int lz_gui_music_dialog_read(HWND dlg, LZMusicSettings *out);

#endif /* LZ_GUI_MUSICDLG_H */
