#ifndef LZ_LOCALIZED_STRINGS_H
#define LZ_LOCALIZED_STRINGS_H

/* Every user-visible string the front end draws (iron law 7).
 *
 * Bilingual, and laid out exactly like src/err.c: an enum, two parallel
 * arrays indexed by it, one switch. Same shape on purpose - the front
 * end shows its own text and the engine's errbuf in the same dialog, and
 * a window whose buttons are Chinese and whose error text is English is
 * what happens when the two tables have independent switches.
 *
 * TWO FORMS PER STRING, and choosing the wrong one is the failure this
 * API exists to prevent:
 *
 *   lz_str_display()  GBK - what goes to a window, a menu, a control.
 *                     Win9x has no CP_UTF8, so an ANSI window reads
 *                     these bytes with code page 936.
 *   lz_str_utf8()     UTF-8 - what goes into conversation history, into
 *                     the engine, or into a file meant to be read
 *                     anywhere else.
 *
 * Passing display bytes to the engine, or UTF-8 bytes to a control, both
 * fail quietly: one tokenizes into U+FFFD, the other draws mojibake.
 * Neither raises anything.
 */

typedef enum {
    /* --- window furniture --- */
    LZ_STR_APP_TITLE = 0,

    /* --- menu bar titles --- */
    LZ_STR_MENU_FILE,
    LZ_STR_MENU_EDIT,
    LZ_STR_MENU_MODEL,
    LZ_STR_MENU_SETTINGS_TITLE,
    LZ_STR_MENU_HELP,

    /* --- main menu --- */
    LZ_STR_MENU_OPEN_MODEL,
    LZ_STR_MENU_OPEN_CHAT,
    LZ_STR_MENU_SAVE_CHAT,
    LZ_STR_MENU_STOP,
    LZ_STR_MENU_CLEAR,
    LZ_STR_MENU_SETTINGS,
    LZ_STR_MENU_ABOUT,
    LZ_STR_MENU_EXIT,
    /* Each language names ITSELF in ITSELF - the item for Chinese stays
     * written in Chinese even inside the English menu, and "English"
     * stays English inside the Chinese one. A translated language name
     * is how someone who cannot read the current language loses the
     * way back out. */
    LZ_STR_MENU_LANG_ZH,
    LZ_STR_MENU_LANG_EN,

    /* The Edit menu and the popup that shares its two commands
       (gui/main.c's show_edit_popup). */
    LZ_STR_MENU_COPY,
    LZ_STR_MENU_SELECT_ALL,
    LZ_STR_MENU_FIND,                 /* opens comdlg32's FindTextA */

    /* Roll the conversation back one turn. Three commands over one
     * mechanism - see gui/main.c's rollback_last. */
    LZ_STR_MENU_REGEN,
    LZ_STR_MENU_EDIT_LAST,
    LZ_STR_MENU_DEL_LAST,

    /* --- buttons --- */
    /* Tool bar captions. Short on purpose - a tool button's label sits
     * under a 24-pixel glyph, so two characters is the budget, and the
     * menu carries the long form (LZ_STR_MENU_OPEN_MODEL vs
     * LZ_STR_BTN_OPEN, e.g. "Open Model" vs "Open"). */
    LZ_STR_BTN_OPEN,
    LZ_STR_BTN_SAVE,
    LZ_STR_BTN_SEND,
    LZ_STR_BTN_STOP,
    LZ_STR_BTN_CLEAR,
    LZ_STR_BTN_REGEN,
    LZ_STR_BTN_SETTINGS,
    LZ_STR_BTN_OK,
    LZ_STR_BTN_CANCEL,
    LZ_STR_BTN_RESTORE_DEFAULT,

    /* --- settings dialog --- */
    LZ_STR_DLG_ABOUT_TITLE,
    LZ_STR_DLG_SETTINGS_TITLE,
    LZ_STR_DLG_THINK,
    LZ_STR_DLG_TEMPERATURE,
    LZ_STR_DLG_TEMP_CAP,
    LZ_STR_DLG_CONTEXT,
    LZ_STR_DLG_CTX_NOTE,   /* unused: the note control is gone
                              (see settingsdlg.c); member kept so the ids
                              after it do not shift. */
    LZ_STR_DLG_TOPP,
    LZ_STR_DLG_REP,
    LZ_STR_DLG_MAXNEW,
    LZ_STR_DLG_MAXNEW_NOTE,
    LZ_STR_DLG_SYS,

    /* --- file dialogs --- */
    LZ_STR_DLG_OPEN_MODEL_TITLE,
    LZ_STR_DLG_SAVE_CHAT_TITLE,
    LZ_STR_DLG_OPEN_CHAT_TITLE,
    LZ_STR_DLG_TEXT_FILTER,
    LZ_STR_FILTER_MODEL,

    /* --- status bar --- */
    LZ_STR_STATE_NO_MODEL,
    LZ_STR_STATE_LOADING,
    LZ_STR_STATE_READY,
    LZ_STR_STATE_GENERATING,
    /* Prefill is NOT generation - no token has been produced yet - and
       it is NOT "the prompt" either. What gets forwarded depends on the
       turn: the whole render on the first one, and with prefix reuse
       only the new suffix on later ones. "Context" is the one word true
       of all of them; naming the prompt would be wrong from turn two
       onward, which is most of a conversation. */
    LZ_STR_STATE_PREFILL,
    LZ_STR_STATE_CTX,                 /* status bar part 1: "context 137/2048" */
    /* The state word IS localised ("generating" in English); what stays
       fixed is the separator and the unit. U+00B7 (MIDDLE DOT) is in
       the GBK tables (A1A4) and round-trips through gui/gbk.c, so the
       · is safe - verified by a compile-time probe. The "tok" unit is
       deliberately never translated: the context cell and the
       candidate list already use the ASCII "tok", and a unit that
       changes language would read as two different units. */
    LZ_STR_STATE_TOKCELL,

    /* --- inference inspector --- */
    /* The candidate list's own title, combined with n_survived the
       same way LZ_STR_STATE_CTX combines with a token count -
       "%s (%d)" in gui/main.c, giving "Candidates (16)" in English or
       the Chinese label with the same count in parentheses. No
       fullwidth punctuation: this is a label using the same sprintf
       convention as the status cell already uses, not new ground. */
    LZ_STR_SIDE_CAND,

    /* --- transcript --- */
    LZ_STR_SPEAKER_USER,
    LZ_STR_SPEAKER_ASSISTANT,
    /* Both carry %s = model name and %d = LZRunState.bytes_alloc in MB.
       That field is the WHOLE runtime state - activation scratch, the
       quantization buffers, the MoE and MTP working set, the KV planes
       among them - so it is named for what it is. It was labelled "KV"
       here and nowhere else; cli_main.c has always called the same
       number "Runtime state". */
    LZ_STR_SYS_MODEL_LOADED,          /* %s = model name, %d = state MB */
    LZ_STR_SIDE_MODEL,                /* %s = model name, %d = state MB */
    LZ_STR_SYS_CTX_TRIMMED,
    LZ_STR_SYS_GEN_STOPPED,
    LZ_STR_SYS_TEMP_SET,              /* %s = the number, already formatted */
    LZ_STR_SYS_THINK_ON,
    LZ_STR_SYS_THINK_OFF,

    /* --- errors --- */
    LZ_STR_ERR_TITLE,
    LZ_STR_ERR_NO_RICHEDIT,
    LZ_STR_ERR_NO_MODEL_BIN,
    /* Distinct from LZ_STR_ERR_NO_MODEL_BIN on purpose: that one means
     * "this SPECIFIC directory has no model.bin" (lz_gui_model_dir_ok
     * refused it - the /load <path> and Open Model pre-checks use it
     * correctly). This one means "no model has been loaded into THIS
     * PROCESS yet" - do_send's lz_gui_model_ready check, which names no
     * directory at all. A user with a perfectly good model directory who
     * simply has not opened it yet gets told their directory is missing
     * a file it has - a problem that does not exist instead of the one
     * that does. */
    LZ_STR_ERR_NO_MODEL_LOADED,
    LZ_STR_ERR_BUSY,
    LZ_STR_ERR_UNKNOWN_CMD,           /* %s = what the user typed */
    LZ_STR_ERR_BAD_TEMP,
    LZ_STR_ERR_BAD_CTX,
    LZ_STR_ERR_BAD_TOPP,
    LZ_STR_ERR_BAD_REP,
    LZ_STR_ERR_BAD_MAXNEW,
    LZ_STR_ERR_BAD_ONOFF,
    LZ_STR_ERR_SAVE_FAILED,
    LZ_STR_FIND_NOT_FOUND,            /* EM_FINDTEXT came back -1 */

    /* --- slash commands --- */
    LZ_STR_HELP_BODY,

    /* --- about / saved file --- */
    LZ_STR_ABOUT_BODY,
    LZ_STR_SAVE_HEADER,

    /* --- about dialog's System Info button and dialog ---
       The System Info feature is implemented the way the Winword.exe
       Ghidra project (D:\ghidra\projects\W95Winword.gpr) shows Word 95
       does it: an enumerator over system facts - OS version, CPU,
       memory, disk - fetched with the matching Win32 calls and shown in
       its own dialog. kunkun98 uses its own wording (the 0x9a000x
       strings are Word 95's, not ours) but the mechanism follows the
       decompile. */
    LZ_STR_BTN_SYSINFO,          /* the About dialog's System Info button */
    LZ_STR_DLG_SYSINFO_TITLE,    /* the System Info dialog's caption */
    LZ_STR_SYSINFO_OS,           /* OS version line: label %s */
    LZ_STR_SYSINFO_CPU,          /* CPU line */
    LZ_STR_SYSINFO_MEM,          /* memory line */
    LZ_STR_SYSINFO_DISK,         /* disk line */

    /* --- settings dialog: think-block dynamic temperature ---
       Appended, not inserted, so the ids above never shift - the same
       discipline that keeps the removed LZ_STR_DLG_SYS_NOTE member in
       place. */
    LZ_STR_DLG_THINK_TEMP,       /* the value box's label, under Thinking */
    LZ_STR_ERR_BAD_THINK_TEMP,   /* think-temp out of [0, 1.0] */

    /* --- about dialog: the Apache-2.0 boilerplate ---
       Rendered in the bottom-left corner below the divider, in Small
       Fonts 7pt (a bitmap face, multi-line). NOT the one-line LZ_STR_ABOUT_BODY
       credit: the owner asked for the full appendix boilerplate. */
    LZ_STR_ABOUT_LICENSE,

    /* --- title bar: the "untitled" chat name ---
       Shown when the current conversation has no file. The caption module
       invents no user-visible text (iron law seven); it receives three GBK
       segments, and this is the chat segment when there is no file. */
    LZ_STR_CAPTION_UNTITLED,

    LZ_STR_COUNT
} LZStr;

/* Select the language and transcode the whole table once.
 *
 * Returns the number of strings that LOST characters on the way to GBK.
 * It must be zero: a
 * string with a character GBK cannot represent turns into '?' at
 * runtime, silently, in a dialog nobody is looking at. Catching it here
 * costs one comparison per string, once.
 *
 * Returns -1 if the arena is too small, which is a build-time mistake
 * (raise LZ_STR_ARENA) rather than a runtime condition.
 *
 * Calling it is optional: the first lz_str_display() initialises to
 * Chinese if nobody has. Forgetting the call must not be the difference
 * between readable text and mojibake. */
int lz_str_init(int english);

/* GBK bytes, for windows, menus and controls. Never NULL. */
const char *lz_str_display(LZStr id);

/* UTF-8 bytes, for history, the engine, and files. Never NULL.
 *
 * Not affected by lz_str_init's transcoding - it is the table source. */
const char *lz_str_utf8(LZStr id);

/* Which language is selected (1 = English). */
int lz_str_lang_english(void);

#endif
