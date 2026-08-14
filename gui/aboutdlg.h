#ifndef LZ_GUI_ABOUTDLG_H
#define LZ_GUI_ABOUTDLG_H

#include <windows.h>

/* The About window - Help > About, replacing the MessageBox.
 *
 * A plain popup with a hand-run modal loop, the same shape as the settings
 * dialog (gui/settingsdlg.c) - see that file's header for why not
 * DialogBoxIndirect.
 *
 * Unlike the settings dialog this one has NOTHING to read back, so its
 * handlers destroy the window directly (g_result in settingsdlg exists to
 * defer the teardown until after the controls are read; there is nothing to
 * read here). That is what makes "Esc / OK / close all dismiss it" checkable
 * from the selftest: the window is simply gone when the message is handled,
 * and the modal loop exits because IsWindow(h) turns false.
 *
 * The ICON and the TEXT both come from the string/resource tables, never
 * typed here: the product name in the body is the string table's
 * LZ_STR_ABOUT_BODY, the caption is LZ_STR_DLG_ABOUT_TITLE, and the icon is
 * the same IDI_APP the main window uses. A dialog that re-types its own
 * product name is two authorities for the same word, and the two drift the
 * first time the brand does. */

/* Run it, modally, centred on the owner. Returns immediately once the
 * window is gone. */
void lz_gui_about_dialog(HWND owner, HINSTANCE inst);

/* Create the window without running the loop, so the selftest can build it,
 * check every control, send it a dismiss message, and watch it die - the
 * modal loop is then the only part with no gate, and it is four lines. */
HWND lz_gui_about_create(HWND owner, HINSTANCE inst);

/* The System Info dialog, opened by the About dialog's System Info...
 * button. Mechanism follows the decompiled Winword.exe (D:\ghidra\
 * projects\W95Winword.gpr): an enumerator over system facts - OS
 * version (GetVersionExA), CPU (GetSystemInfo), memory
 * (GlobalMemoryStatus), disk (GetDiskFreeSpaceA) - with kunkun98's own
 * wording. Same popup + modal loop shape as the About window. */
void lz_gui_sysinfo_dialog(HWND owner, HINSTANCE inst);

/* Build the System Info window without running its loop, so the selftest
 * can create it, read the fact lines, and destroy it - the same
 * create-without-loop split the About window has. */
HWND lz_gui_sysinfo_create(HWND owner, HINSTANCE inst);

#endif
