#ifndef LZ_GUI_RESOURCE_H
#define LZ_GUI_RESOURCE_H

/* Resource ids, shared by gui/kunkun98.rc and the code that loads them.
 *
 * Small numbers on purpose: the icon has to be the LOWEST-numbered icon
 * in the binary, because that is the one the shell shows for the file.
 * Nothing enforces that but the numbering. */
#define IDI_APP     1
#define IDB_SPLASH  2
#define IDB_CHICKEN 3
/* About-dialog logo background: a 68x191 crop of the splash backdrop
 * (field + ghosted photo, NO 3D rooster), dithered to the Win9x 256
 * palette like the splash. */
#define IDB_ABOUT_LOGO 4
/* Status-bar lamps (build/watcom/make_lamps.py). Four states, one
 * bitmap each - a strip would need splitting at run time for no gain. */
#define IDB_LAMP_OFF   5
#define IDB_LAMP_READY 6
#define IDB_LAMP_BUSY  7
#define IDB_LAMP_ERROR 8

#define IDR_ACCEL   10

/* The virtual-key code windows.h would normally supply (winuser.h).
 * Defined by hand instead of pulling in windows.h here: a resource
 * compiler (windres, wrc) reads this header through its own
 * preprocessor, and VK_ESCAPE is not a keyword either one understands -
 * without a definition in scope, "VK_ESCAPE" in an ACCELERATORS entry
 * is an undefined token and both compilers reject it with a bare
 * "syntax error", naming no symbol. 0x1B is VK_ESCAPE's fixed value on
 * every Windows version back to 3.1; it does not change.
 *
 * Guarded, unlike IDI_APP/IDM_* above: those live in this project's own
 * numbering and cannot collide with anything, but gui/main.c includes
 * both windows.h and this header (in that order), and windows.h already
 * declares the real VK_ESCAPE - verified identical in both toolchains'
 * own headers (MinGW winuser.h and Watcom h/nt/winuser.h both give
 * 0x1B), so redefining it to the same token sequence is standards-legal
 * and silent either way. The guard is not fixing a live bug; it is
 * removing the redefinition from happening at all, matching how every
 * other entry in gui/compat40.h that could plausibly already be
 * declared is written. */
#ifndef VK_ESCAPE
#define VK_ESCAPE 0x1B
#endif

/* Menu command ids. Here rather than in an enum in gui/main.c because
 * the ACCELERATORS table in gui/kunkun98.rc needs them too, and a
 * resource compiler cannot read an enum. One definition, two consumers -
 * the alternative was a second hand-kept copy of the numbers inside the
 * .rc, which is the shape of drift this project exists to punish
 * elsewhere in this tree.
 *
 * Only the ids an accelerator or a menu refers to live here. The control
 * ids (ID_SEND and friends) stay in main.c: no resource mentions them. */
#define IDM_OPEN_MODEL 2001
#define IDM_SAVE_CHAT  2002
#define IDM_STOP_GEN   2003
#define IDM_CLEAR      2004
#define IDM_SETTINGS   2005
#define IDM_ABOUT      2006
#define IDM_EXIT       2007
/* Language. Two items rather than one toggle: a checked pair says which
 * one you are in without being read, and a single "English" item is
 * ambiguous about whether it names the current state or the action.
 * Radio-checked at build time rather than with CheckMenuRadioItem, which
 * is a 4.0 call the 3.51 floor hides. */
#define IDM_LANG_ZH    2008
#define IDM_LANG_EN    2009
/* The four-deep recent-model list under File (gui/mru.h). Contiguous
 * ids: IDM_MRU0 .. IDM_MRU0 + LZ_MRU_MAX - 1, so the WM_COMMAND handler
 * can index g.mru.item[] with LOWORD(wp) - IDM_MRU0 instead of a
 * four-way switch. */
#define IDM_MRU0       2010

/* Edit menu: forwarded to whichever of the transcript or the input box
   has focus (gui/main.c). */
#define IDM_COPY        2020
#define IDM_SELECT_ALL  2021

/* Load a saved conversation (gui/chatfile.h). Not contiguous with the
 * ids above - the gap that is left is reserved for File commands;
 * Copy and Select All already have ids above, next to IDM_MRU0. */
#define IDM_OPEN_CHAT  2022

/* Find in the conversation, via comdlg32's FindTextA - no
 * FINDREPLACE-specific ids are needed here, this is only the menu
 * command that opens the dialog. */
#define IDM_FIND       2023

/* Roll the conversation back one turn, three ways. Contiguous and in
 * the order they appear under Edit, but nothing indexes them the way
 * IDM_MRU0 is indexed - each has its own case in the WM_COMMAND
 * switch, because each does a different thing with the user turn it
 * uncovers.
 *
 * No accelerators. Every one of them destroys text the user is looking
 * at, and this era's Undo does not reach into a RichEdit that was
 * written by EM_REPLACESEL with fUndo FALSE (gui/main.c's append_run) -
 * so a mistyped Ctrl-key would take a reply away with nothing to bring
 * it back. The menu is the whole surface, deliberately. */
#define IDM_REGEN      2024
#define IDM_EDIT_LAST  2025
#define IDM_DEL_LAST   2026

#endif
