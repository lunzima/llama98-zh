/* Every user-visible string the front end draws, bilingual.
 *
 * Laid out like src/err.c: enum-indexed parallel arrays, POSITIONAL.
 * Row N is LZStr N, and the tag in each row's comment is the only thing
 * that says so to a reader - the compiler binds by position alone.
 * Designated initialisers would say it to the compiler too, but they are
 * C99 and Visual C++ 4.0, in the target family, cannot compile them.
 * build/str_table_order_gate.sh reads these comments against the enum
 * and requires them equal, which is the safety the designated form gave
 * in a shape that survives losing it.
 *
 * The Chinese here is DATA, not annotation - same exemption err.c's
 * table has. Comments in this file are still English, and the checker
 * still enforces that: the exemption is per line, not per file.
 *
 * Some wording is specified by the design documents; the button labels
 * and the dialog captions are not specified anywhere and are written
 * here; they are the ordinary Win9x wording and are meant to be
 * changed by whoever disagrees, not treated as decided.
 */
#include <string.h>

#include "localized_strings.h"
#include "chatfile.h"
#include "gbk.h"

/* Room for the transcoded table. GBK is never longer than the UTF-8 it
 * came from - two bytes per CJK character against three, one against one
 * for ASCII - so the UTF-8 total is a safe bound. Measured usage is
 * reported so this stays a number
 * somebody chose rather than a number nobody has looked at since. */
#define LZ_STR_ARENA 8192

static const char *const LZ_STR_EN[LZ_STR_COUNT] = {
    /* LZ_STR_APP_TITLE */ "Kunkun98",

    /* Mnemonics. Every Win9x menu had them; a menu bar with no
       underlined letters is the single most visible way this front end
       fails to look like its target. The accelerator text after \t is
       right-aligned by the menu and must agree with the ACCELERATORS
       table in gui/kunkun98.rc - the selftest checks that both exist. */
    /* LZ_STR_MENU_FILE */ "&File",
    /* LZ_STR_MENU_EDIT */ "&Edit",
    /* LZ_STR_MENU_MODEL */ "&Model",
    /* LZ_STR_MENU_SETTINGS_TITLE */ "&Settings",
    /* LZ_STR_MENU_HELP */ "&Help",

    /* LZ_STR_MENU_OPEN_MODEL */ "&Open Model...\tCtrl+O",
    /* LZ_STR_MENU_OPEN_CHAT */ "&Open Conversation...",
    /* LZ_STR_MENU_SAVE_CHAT */ "&Save Conversation...\tCtrl+S",
    /* LZ_STR_MENU_STOP */ "S&top Generating\tEsc",
    /* LZ_STR_MENU_CLEAR */ "&Clear Conversation\tCtrl+N",
    /* LZ_STR_MENU_SETTINGS */ "&Inference Settings...",
    /* LZ_STR_MENU_ABOUT */ "&About",
    /* LZ_STR_MENU_EXIT */ "E&xit",
    /* LZ_STR_MENU_LANG_ZH */ "\xe4\xb8\xad\xe6\x96\x87(&C)",
    /* LZ_STR_MENU_LANG_EN */ "&English",

    /* LZ_STR_MENU_COPY */ "&Copy\tCtrl+C",
    /* LZ_STR_MENU_SELECT_ALL */ "Select &All\tCtrl+A",
    /* LZ_STR_MENU_FIND */ "&Find...\tCtrl+F",

    /* No "\tCtrl+..." tail on any of the three - see the note beside
       IDM_REGEN in gui/resource.h for why they have no accelerator to
       advertise. */
    /* LZ_STR_MENU_REGEN */ "&Regenerate",
    /* LZ_STR_MENU_EDIT_LAST */ "&Edit Last Message",
    /* LZ_STR_MENU_DEL_LAST */ "&Delete Last Exchange",

    /* LZ_STR_BTN_OPEN */ "Open",
    /* LZ_STR_BTN_SAVE */ "Save",
    /* LZ_STR_BTN_SEND */ "Send",
    /* LZ_STR_BTN_STOP */ "Stop",
    /* LZ_STR_BTN_CLEAR */ "Clear",
    /* Shorter than the menu's own word, same reason LZ_STR_BTN_SETTINGS
       is shorter than LZ_STR_MENU_SETTINGS just below: this caption sits
       under a 24-pixel glyph in a button sized for two Chinese
       characters, and "Regenerate" does not fit in it. */
    /* LZ_STR_BTN_REGEN */ "Retry",
    /* "Options", not "Settings", and the toolbar is the reason: the
       strip's captions sit under a 24-pixel glyph in a button whose
       width floor is set for two Chinese characters, and "Settings"
       comes back out of the control as "Sett...". It is also the period's
       own word - Tools > Options is where a 1997 program put this. The
       MENU keeps the longer "Inference Settings...", which has the room
       and needs the qualifier. */
    /* LZ_STR_BTN_SETTINGS */ "Options",
    /* LZ_STR_BTN_OK */ "OK",
    /* LZ_STR_BTN_CANCEL */ "Cancel",
    /* LZ_STR_BTN_RESTORE_DEFAULT */ "&Restore Defaults",

    /* The dialog gets mnemonics too - its loop calls IsDialogMessage,
       which is what makes Alt+letter reach a control. The tool-strip
       buttons deliberately do NOT: they duplicate menu commands, Alt
       goes to the menu bar first, and a Win9x toolbar had no mnemonics
       of its own either. */
    /* LZ_STR_DLG_ABOUT_TITLE */ "About Kunkun98",
    /* LZ_STR_DLG_SETTINGS_TITLE */ "Inference Settings",
    /* LZ_STR_DLG_THINK */ "&Thinking (think)",
    /* LZ_STR_DLG_TEMPERATURE */ "Tem&perature",
    /* LZ_STR_DLG_TEMP_CAP */ "Temperature is capped at 1.0",
    /* THE SETTINGS DIALOG'S MNEMONICS, all of them, in one place -
       every other entry in this table points here rather than
       repeating a subset, because two hand-kept lists drift and the
       shorter one then reads as permission to reuse a letter:

           T think        P temperature   B think-block temp
           O top-p        E repetition    N context
           M max new      S system prompt F beep on finish
           R restore defaults

       The ZH table takes the SAME letter for each, so this list
       governs both. OK and Cancel carry none in either language.
       A collision is silent - Windows cycles focus between the
       clashing controls instead of activating one - so gui/main.c's
       selftest reads the letters off the real controls of the real
       dialog and reddens on a repeat.
       A STATIC's mnemonic moves focus to the next tab stop in Z-order,
       which is why each label is created immediately before its own
       controls. */
    /* LZ_STR_DLG_CONTEXT */ "Co&ntext",
    /* LZ_STR_DLG_CTX_NOTE */ "512 to 32768; larger needs more memory",
    /* LZ_STR_DLG_TOPP */ "T&op-p",
    /* LZ_STR_DLG_REP */ "Repetition p&enalty",
    /* LZ_STR_DLG_MAXNEW */ "&Max new tokens",
    /* LZ_STR_DLG_MAXNEW_NOTE */ "-1 = unlimited (until EOS or the "
                                    "context fills)",
    /* LZ_STR_DLG_SYS */ "&System prompt",

    /* LZ_STR_DLG_OPEN_MODEL_TITLE */ "Select a model directory",
    /* LZ_STR_DLG_SAVE_CHAT_TITLE */ "Save conversation",
    /* LZ_STR_DLG_OPEN_CHAT_TITLE */ "Open conversation",
    /* LZ_STR_DLG_TEXT_FILTER */ "Text files (*.txt)",
    /* LZ_STR_FILTER_MODEL */ "Kunkun98 model (model.bin)",

    /* LZ_STR_STATE_NO_MODEL */ "no model",
    /* LZ_STR_STATE_LOADING */ "loading...",
    /* LZ_STR_STATE_READY */ "ready",
    /* LZ_STR_STATE_GENERATING */ "generating",
    /* LZ_STR_STATE_PREFILL */ "processing context",
    /* LZ_STR_STATE_CTX */ "context",
    /* LZ_STR_STATE_TOKCELL */ "generating %d tok, %.1f tok/s",

    /* LZ_STR_SIDE_CAND */ "Candidates",

    /* LZ_STR_SPEAKER_USER */ "You",
    /* LZ_STR_SPEAKER_ASSISTANT */ "Kunkun98",
    /* LZ_STR_SYS_MODEL_LOADED */ "model loaded: %s (state %dMB)",
    /* LZ_STR_SIDE_MODEL */ "%s (state %dMB)",
    /* LZ_STR_SYS_CTX_TRIMMED */ "context trimmed",
    /* LZ_STR_SYS_GEN_STOPPED */ "generation stopped",
    /* LZ_STR_SYS_TEMP_SET */ "temperature set to %s",
    /* LZ_STR_SYS_THINK_ON */ "thinking: on",
    /* LZ_STR_SYS_THINK_OFF */ "thinking: off",

    /* LZ_STR_ERR_TITLE */ "Error",
    /* LZ_STR_ERR_NO_RICHEDIT */ "neither riched20.dll nor riched32.dll "
                                    "could be loaded; Kunkun98 cannot show "
                                    "a conversation without one",
    /* LZ_STR_ERR_NO_MODEL_BIN */ "that directory has no model.bin",
    /* LZ_STR_ERR_NO_MODEL_LOADED */ "no model is loaded yet - use "
                                    "Open Model (Ctrl+O) or /load <path>",
    /* LZ_STR_ERR_BUSY */ "still generating; stop it first",
    /* LZ_STR_ERR_UNKNOWN_CMD */ "unknown command: %s",
    /* LZ_STR_ERR_BAD_TEMP */ "temperature must be between 0 and 1.0",
    /* LZ_STR_ERR_BAD_CTX */ "context must be between 512 and 32768",
    /* LZ_STR_ERR_BAD_TOPP */ "top-p must be between 0.05 and 1.0",
    /* LZ_STR_ERR_BAD_REP */ "repetition penalty must be between "
                                    "1.0 and 1.5",
    /* LZ_STR_ERR_BAD_MAXNEW */ "max new tokens must be -1, or between "
                                    "16 and the context window",
    /* LZ_STR_ERR_BAD_ONOFF */ "use on or off",
    /* LZ_STR_ERR_SAVE_FAILED */ "could not write that file",
    /* LZ_STR_FIND_NOT_FOUND */ "cannot find that text",

    /* LZ_STR_HELP_BODY */
        "/load <path>   open a model\n"
        "/save          save the conversation\n"
        "/clear         clear the conversation\n"
        "/stop          stop generating\n"
        "/temp <0-1>    temperature (1.0 is a hard cap)\n"
        "/think on|off  thinking mode\n"
        "/help          this list",

    /* Laid out the way a Win9x about box is: product, version, then
       the credits, one thing per line. A single pipe-separated line
       would carry the same facts and read like a status bar. */
    /* LZ_STR_ABOUT_BODY */
        "Kunkun98\n"
        "for Windows, version 0.1\n"
        "\n"
        "latent mixture of experts + linear attention\n"
        "engine: llama98",
    /* Not translated: this is the chat file's magic line, and
       lz_chatfile_decode compares it byte-for-byte against
       LZ_CHATFILE_MAGIC regardless of which language wrote the file -
       a Chinese-language save has to open under English and vice
       versa. Sourced from chatfile.h rather than typed out a second
       time so the two cannot drift apart. */
    /* LZ_STR_SAVE_HEADER */ LZ_CHATFILE_MAGIC,

    /* System Info - see localized_strings.h. Own wording, Word 95's
       mechanism. The %s slots are the values the code fetches
       with GetVersionExA / GetSystemInfo / GlobalMemoryStatus /
       GetDiskFreeSpaceA, the same calls the decompiled Winword.exe uses. */
    /* LZ_STR_BTN_SYSINFO */ "System Info...",
    /* LZ_STR_DLG_SYSINFO_TITLE */ "Kunkun98 System Info",
    /* LZ_STR_SYSINFO_OS */ "Windows: %s",
    /* LZ_STR_SYSINFO_CPU */ "CPU: %s",
    /* LZ_STR_SYSINFO_MEM */ "Memory: %s",
    /* LZ_STR_SYSINFO_DISK */ "Disk: %s",
    /* LZ_STR_SYSINFO_TIER */ "Operators: %s",

    /* Think-block dynamic temperature. Mnemonic B; the whole set is
       listed at LZ_STR_DLG_TEMP_CAP above. */
    /* LZ_STR_DLG_THINK_TEMP */ "Think-&block temp",
    /* LZ_STR_ERR_BAD_THINK_TEMP */ "think-block temperature must be "
                                    "between 0 and 1.0",

    /* The Apache-2.0 appendix boilerplate, rendered bottom-left of the
       About dialog in Small Fonts 7pt. The copyright owner is the
       project's own attribution (see the about body's own comment). */
    /* LZ_STR_ABOUT_LICENSE */
        "Copyright (c) 2026 Lunzima\n"
        "Licensed under the Apache License, Version 2.0",
    /* LZ_STR_CAPTION_UNTITLED */ "Untitled Chat",
    /* Mnemonic F, not the P of "Bee&p": temperature already holds P
       (see the set at LZ_STR_DLG_TEMP_CAP), and this checkbox is the
       newcomer. */
    /* LZ_STR_DLG_BEEP */ "Beep when a reply &finishes"
};

static const char *const LZ_STR_ZH[LZ_STR_COUNT] = {
    /* LZ_STR_APP_TITLE */ "\xe6\x98\x86\xe6\x98\x86\x39\x38",

    /* Chinese Windows puts the mnemonic in parentheses after the word -
       the label renders as the word followed by (F) with F underlined.
       Not a liberty taken in translation: every menu in Simplified
       Chinese Win9x looks like this, because a Han character has no
       Latin letter of its own to underline. */
    /* LZ_STR_MENU_FILE */ "\xe6\x96\x87\xe4\xbb\xb6(&F)",
    /* LZ_STR_MENU_EDIT */ "\xe7\xbc\x96\xe8\xbe\x91(&E)",
    /* LZ_STR_MENU_MODEL */ "\xe6\xa8\xa1\xe5\x9e\x8b(&M)",
    /* LZ_STR_MENU_SETTINGS_TITLE */ "\xe8\xae\xbe\xe7\xbd\xae(&S)",
    /* LZ_STR_MENU_HELP */ "\xe5\xb8\xae\xe5\x8a\xa9(&H)",

    /* LZ_STR_MENU_OPEN_MODEL */ "\xe6\x89\x93\xe5\xbc\x80\xe6\xa8\xa1\xe5\x9e\x8b(&O)\xe2\x80\xa6\tCtrl+O",
    /* LZ_STR_MENU_OPEN_CHAT */ "\xe6\x89\x93\xe5\xbc\x80\xe5\xaf\xb9\xe8\xaf\x9d(&O)\xe2\x80\xa6",
    /* LZ_STR_MENU_SAVE_CHAT */ "\xe4\xbf\x9d\xe5\xad\x98\xe5\xaf\xb9\xe8\xaf\x9d(&S)\xe2\x80\xa6\tCtrl+S",
    /* LZ_STR_MENU_STOP */ "\xe5\x81\x9c\xe6\xad\xa2\xe7\x94\x9f\xe6\x88\x90(&T)\tEsc",
    /* LZ_STR_MENU_CLEAR */ "\xe6\xb8\x85\xe7\xa9\xba\xe5\xaf\xb9\xe8\xaf\x9d(&C)\tCtrl+N",
    /* LZ_STR_MENU_SETTINGS */ "\xe6\x8e\xa8\xe7\x90\x86\xe8\xae\xbe\xe7\xbd\xae(&I)\xe2\x80\xa6",
    /* LZ_STR_MENU_ABOUT */ "\xe5\x85\xb3\xe4\xba\x8e(&A)",
    /* LZ_STR_MENU_EXIT */ "\xe9\x80\x80\xe5\x87\xba(&X)",
    /* LZ_STR_MENU_LANG_ZH */ "\xe4\xb8\xad\xe6\x96\x87(&C)",
    /* LZ_STR_MENU_LANG_EN */ "&English",

    /* LZ_STR_MENU_COPY */ "\xe5\xa4\x8d\xe5\x88\xb6(&C)\tCtrl+C",
    /* LZ_STR_MENU_SELECT_ALL */ "\xe5\x85\xa8\xe9\x80\x89(&A)\tCtrl+A",
    /* LZ_STR_MENU_FIND */ "\xe6\x9f\xa5\xe6\x89\xbe(&F)\xe2\x80\xa6\tCtrl+F",

    /* LZ_STR_MENU_REGEN */ "\xe9\x87\x8d\xe6\x96\xb0\xe7\x94\x9f\xe6\x88\x90(&R)",
    /* LZ_STR_MENU_EDIT_LAST */ "\xe4\xbf\xae\xe6\x94\xb9\xe4\xb8\x8a\xe4\xb8\x80\xe5\x8f\xa5(&E)",
    /* LZ_STR_MENU_DEL_LAST */ "\xe5\x88\xa0\xe9\x99\xa4\xe4\xb8\x8a\xe4\xb8\x80\xe8\xbd\xae(&D)",

    /* LZ_STR_BTN_OPEN */ "\xe6\x89\x93\xe5\xbc\x80",
    /* LZ_STR_BTN_SAVE */ "\xe4\xbf\x9d\xe5\xad\x98",
    /* LZ_STR_BTN_SEND */ "\xe5\x8f\x91\xe9\x80\x81",
    /* LZ_STR_BTN_STOP */ "\xe5\x81\x9c\xe6\xad\xa2",
    /* LZ_STR_BTN_CLEAR */ "\xe6\xb8\x85\xe7\xa9\xba",
    /* LZ_STR_BTN_REGEN */ "\xe9\x87\x8d\xe8\xaf\x95",
    /* LZ_STR_BTN_SETTINGS */ "\xe8\xae\xbe\xe7\xbd\xae",
    /* LZ_STR_BTN_OK */ "\xe7\xa1\xae\xe5\xae\x9a",
    /* LZ_STR_BTN_CANCEL */ "\xe5\x8f\x96\xe6\xb6\x88",
    /* LZ_STR_BTN_RESTORE_DEFAULT */ "\xe6\x81\xa2\xe5\xa4\x8d\xe9\xbb\x98\xe8\xae\xa4(&R)",

    /* LZ_STR_DLG_ABOUT_TITLE */ "\xe5\x85\xb3\xe4\xba\x8e \xe6\x98\x86\xe6\x98\x86\x39\x38",
    /* LZ_STR_DLG_SETTINGS_TITLE */ "\xe6\x8e\xa8\xe7\x90\x86\xe8\xae\xbe\xe7\xbd\xae",
    /* Half-width parentheses and colons throughout the Chinese table.
       Full-width ones are a full em wide with the glyph hugging one
       side, and in SimSun's 12-pixel BITMAP faces - which is what the
       target actually renders with, not the outline face a modern host
       substitutes - that reads as a hole in the line. Chinese commas and
       full stops are left alone; the complaint is specific to the marks
       whose full-width form is mostly whitespace. */
    /* LZ_STR_DLG_THINK */ "\xe6\xb7\xb1\xe5\xba\xa6\xe6\x80\x9d\xe8\x80\x83(think)(&T)",
    /* LZ_STR_DLG_TEMPERATURE */ "\xe6\xb8\xa9\xe5\xba\xa6(&P)",
    /* LZ_STR_DLG_TEMP_CAP */ "\xe6\xb8\xa9\xe5\xba\xa6\xe4\xb8\x8a\xe9\x99\x90 1.0",
    /* LZ_STR_DLG_CONTEXT */ "\xe4\xb8\x8a\xe4\xb8\x8b\xe6\x96\x87(&N)",
    /* LZ_STR_DLG_CTX_NOTE */ "\xe8\x8c\x83\xe5\x9b\xb4 512 - 32768\xef\xbc\x8c\xe8\xb6\x8a\xe5\xa4\xa7\xe8\xb6\x8a\xe5\x8d\xa0\xe5\x86\x85\xe5\xad\x98",
    /* LZ_STR_DLG_TOPP */ "Top-p(&O)",
    /* LZ_STR_DLG_REP */ "\xe9\x87\x8d\xe5\xa4\x8d\xe6\x83\xa9\xe7\xbd\x9a(&E)",
    /* LZ_STR_DLG_MAXNEW */ "\xe6\x9c\x80\xe5\xa4\xa7\xe7\x94\x9f\xe6\x88\x90\xe9\x95\xbf\xe5\xba\xa6(&M)",
    /* LZ_STR_DLG_MAXNEW_NOTE */ "-1 = \xe4\xb8\x8d\xe9\x99\x90\xef\xbc\x88\xe7\x9b\xb4\xe5\x88\xb0 EOS \xe6\x88\x96\xe4\xb8\x8a\xe4\xb8\x8b\xe6\x96\x87\xe5\x8d\xa0\xe6\xbb\xa1\xef\xbc\x89",
    /* LZ_STR_DLG_SYS */ "\xe7\xb3\xbb\xe7\xbb\x9f\xe6\x8f\x90\xe7\xa4\xba\xe8\xaf\x8d(&S)",

    /* LZ_STR_DLG_OPEN_MODEL_TITLE */ "\xe9\x80\x89\xe6\x8b\xa9\xe6\xa8\xa1\xe5\x9e\x8b\xe7\x9b\xae\xe5\xbd\x95",
    /* LZ_STR_DLG_SAVE_CHAT_TITLE */ "\xe4\xbf\x9d\xe5\xad\x98\xe5\xaf\xb9\xe8\xaf\x9d",
    /* LZ_STR_DLG_OPEN_CHAT_TITLE */ "\xe6\x89\x93\xe5\xbc\x80\xe5\xaf\xb9\xe8\xaf\x9d",
    /* LZ_STR_DLG_TEXT_FILTER */ "\xe6\x96\x87\xe6\x9c\xac\xe6\x96\x87\xe4\xbb\xb6 (*.txt)",
    /* LZ_STR_FILTER_MODEL */ "\xe6\x98\x86\xe6\x98\x86\x39\x38 \xe6\xa8\xa1\xe5\x9e\x8b (model.bin)",

    /* LZ_STR_STATE_NO_MODEL */ "\xe6\x9c\xaa\xe5\x8a\xa0\xe8\xbd\xbd",
    /* LZ_STR_STATE_LOADING */ "\xe5\x8a\xa0\xe8\xbd\xbd\xe4\xb8\xad\xe2\x80\xa6",
    /* LZ_STR_STATE_READY */ "\xe5\xb7\xb2\xe5\x8a\xa0\xe8\xbd\xbd",
    /* LZ_STR_STATE_GENERATING */ "\xe7\x94\x9f\xe6\x88\x90\xe4\xb8\xad",
    /* LZ_STR_STATE_PREFILL */ "\xe5\xa4\x84\xe7\x90\x86\xe4\xb8\x8a\xe4\xb8\x8b\xe6\x96\x87",
    /* LZ_STR_STATE_CTX */ "\xe4\xb8\x8a\xe4\xb8\x8b\xe6\x96\x87",
    /* The ZH table DOES translate this one - the state word is ordinary
       Han and GBK-encodable. The cell uses a comma rather than a
       middle-dot separator (the owner asked to avoid dot separators),
       so it needs
       no mapping and cannot mojibake. What stays ASCII is the TOKEN
       UNIT "tok" and the "/s" - there is no standard localisation for
       a token count unit, and every token-count display in this
       program (the context cell, the candidate list) already uses the
       ASCII "tok". */
    /* LZ_STR_STATE_TOKCELL */
        "\xe7\x94\x9f\xe6\x88\x90\xe4\xb8\xad %d tok, %.1f tok/s",

    /* LZ_STR_SIDE_CAND */ "\xe5\x80\x99\xe9\x80\x89",

    /* LZ_STR_SPEAKER_USER */ "\xe4\xbd\xa0",
    /* LZ_STR_SPEAKER_ASSISTANT */ "\xe6\x98\x86\xe6\x98\x86\x39\x38",
    /* LZ_STR_SYS_MODEL_LOADED */ "\xe6\xa8\xa1\xe5\x9e\x8b\xe5\xb7\xb2\xe5\x8a\xa0\xe8\xbd\xbd: %s (\xe8\xbf\x90\xe8\xa1\x8c\xe7\x8a\xb6\xe6\x80\x81 %dMB)",
    /* LZ_STR_SIDE_MODEL */ "%s (\xe8\xbf\x90\xe8\xa1\x8c\xe7\x8a\xb6\xe6\x80\x81 %dMB)",
    /* LZ_STR_SYS_CTX_TRIMMED */ "\xe4\xb8\x8a\xe4\xb8\x8b\xe6\x96\x87\xe5\xb7\xb2\xe8\xa3\x81\xe5\x89\xaa",
    /* LZ_STR_SYS_GEN_STOPPED */ "\xe5\xb7\xb2\xe5\x81\x9c\xe6\xad\xa2\xe7\x94\x9f\xe6\x88\x90",
    /* LZ_STR_SYS_TEMP_SET */ "\xe6\xb8\xa9\xe5\xba\xa6\xe5\xb7\xb2\xe8\xae\xbe\xe4\xb8\xba %s",
    /* LZ_STR_SYS_THINK_ON */ "\xe6\xb7\xb1\xe5\xba\xa6\xe6\x80\x9d\xe8\x80\x83: \xe5\xbc\x80",
    /* LZ_STR_SYS_THINK_OFF */ "\xe6\xb7\xb1\xe5\xba\xa6\xe6\x80\x9d\xe8\x80\x83: \xe5\x85\xb3",

    /* LZ_STR_ERR_TITLE */ "\xe9\x94\x99\xe8\xaf\xaf",
    /* LZ_STR_ERR_NO_RICHEDIT */ "riched20.dll \xe4\xb8\x8e riched32.dll \xe9\x83\xbd\xe5\x8a\xa0\xe8\xbd\xbd\xe4\xb8\x8d\xe4\xba\x86\xef\xbc\x8c"
                                    "\xe6\x98\x86\xe6\x98\x86\x39\x38 \xe6\xb2\xa1\xe6\x9c\x89\xe5\xae\x83\xe4\xbb\xac\xe5\xb0\xb1\xe6\x97\xa0\xe6\xb3\x95\xe6\x98\xbe\xe7\xa4\xba\xe5\xaf\xb9\xe8\xaf\x9d",
    /* LZ_STR_ERR_NO_MODEL_BIN */ "\xe8\xaf\xa5\xe7\x9b\xae\xe5\xbd\x95\xe9\x87\x8c\xe6\xb2\xa1\xe6\x9c\x89 model.bin",
    /* LZ_STR_ERR_NO_MODEL_LOADED */ "\xe8\xbf\x98\xe6\xb2\xa1\xe6\x9c\x89\xe5\x8a\xa0\xe8\xbd\xbd\xe6\xa8\xa1\xe5\x9e\x8b \xe2\x80\x94\xe2\x80\x94 \xe7\x94\xa8\xe3\x80\x8c\xe6\x89\x93\xe5\xbc\x80\xe6\xa8\xa1\xe5\x9e\x8b\xe3\x80\x8d"
                                    "(Ctrl+O) \xe6\x88\x96 /load <\xe8\xb7\xaf\xe5\xbe\x84>",
    /* LZ_STR_ERR_BUSY */ "\xe6\xad\xa3\xe5\x9c\xa8\xe7\x94\x9f\xe6\x88\x90\xef\xbc\x8c\xe8\xaf\xb7\xe5\x85\x88\xe5\x81\x9c\xe6\xad\xa2",
    /* LZ_STR_ERR_UNKNOWN_CMD */ "\xe6\x9c\xaa\xe7\x9f\xa5\xe5\x91\xbd\xe4\xbb\xa4: %s",
    /* LZ_STR_ERR_BAD_TEMP */ "\xe6\xb8\xa9\xe5\xba\xa6\xe5\xbf\x85\xe9\xa1\xbb\xe5\x9c\xa8 0 \xe5\x88\xb0 1.0 \xe4\xb9\x8b\xe9\x97\xb4",
    /* LZ_STR_ERR_BAD_CTX */ "\xe4\xb8\x8a\xe4\xb8\x8b\xe6\x96\x87\xe7\xaa\x97\xe5\x8f\xa3\xe5\xbf\x85\xe9\xa1\xbb\xe5\x9c\xa8 512 \xe5\x88\xb0 32768 \xe4\xb9\x8b\xe9\x97\xb4",
    /* LZ_STR_ERR_BAD_TOPP */ "Top-p \xe5\xbf\x85\xe9\xa1\xbb\xe5\x9c\xa8 0.05 \xe5\x88\xb0 1.0 \xe4\xb9\x8b\xe9\x97\xb4",
    /* LZ_STR_ERR_BAD_REP */ "\xe9\x87\x8d\xe5\xa4\x8d\xe6\x83\xa9\xe7\xbd\x9a\xe5\xbf\x85\xe9\xa1\xbb\xe5\x9c\xa8 1.0 \xe5\x88\xb0 1.5 \xe4\xb9\x8b\xe9\x97\xb4",
    /* LZ_STR_ERR_BAD_MAXNEW */ "\xe6\x9c\x80\xe5\xa4\xa7\xe7\x94\x9f\xe6\x88\x90\xe9\x95\xbf\xe5\xba\xa6\xe5\xbf\x85\xe9\xa1\xbb\xe6\x98\xaf -1\xef\xbc\x8c"
                                    "\xe6\x88\x96\xe4\xbb\x8b\xe4\xba\x8e 16 \xe4\xb8\x8e\xe4\xb8\x8a\xe4\xb8\x8b\xe6\x96\x87\xe7\xaa\x97\xe5\x8f\xa3\xe4\xb9\x8b\xe9\x97\xb4",
    /* LZ_STR_ERR_BAD_ONOFF */ "\xe8\xaf\xb7\xe7\x94\xa8 on \xe6\x88\x96 off",
    /* LZ_STR_ERR_SAVE_FAILED */ "\xe5\x86\x99\xe4\xb8\x8d\xe4\xba\x86\xe8\xbf\x99\xe4\xb8\xaa\xe6\x96\x87\xe4\xbb\xb6",
    /* LZ_STR_FIND_NOT_FOUND */ "\xe6\x89\xbe\xe4\xb8\x8d\xe5\x88\xb0\xe8\xbf\x99\xe6\xae\xb5\xe6\x96\x87\xe5\xad\x97",

    /* LZ_STR_HELP_BODY */
        "/load <\xe8\xb7\xaf\xe5\xbe\x84>   \xe6\x89\x93\xe5\xbc\x80\xe6\xa8\xa1\xe5\x9e\x8b\n"
        "/save          \xe4\xbf\x9d\xe5\xad\x98\xe5\xaf\xb9\xe8\xaf\x9d\n"
        "/clear         \xe6\xb8\x85\xe7\xa9\xba\xe5\xaf\xb9\xe8\xaf\x9d\n"
        "/stop          \xe5\x81\x9c\xe6\xad\xa2\xe7\x94\x9f\xe6\x88\x90\n"
        "/temp <0-1>    \xe6\xb8\xa9\xe5\xba\xa6 (1.0 \xe6\x98\xaf\xe7\xa1\xac\xe4\xb8\x8a\xe9\x99\x90)\n"
        "/think on|off  \xe6\xb7\xb1\xe5\xba\xa6\xe6\x80\x9d\xe8\x80\x83\xe5\xbc\x80\xe5\x85\xb3\n"
        "/help          \xe5\x88\x97\xe5\x87\xba\xe5\x91\xbd\xe4\xbb\xa4",

    /* LZ_STR_ABOUT_BODY */
        "\xe6\x98\x86\xe6\x98\x86\x39\x38 (Kunkun98)\n"
        "for Windows, \xe7\x89\x88\xe6\x9c\xac 0.1\n"
        "\n"
        "\xe9\x9a\x90\xe7\xa9\xba\xe9\x97\xb4\xe6\xb7\xb7\xe5\x90\x88\xe4\xb8\x93\xe5\xae\xb6 + \xe7\xba\xbf\xe6\x80\xa7\xe6\xb3\xa8\xe6\x84\x8f\xe5\x8a\x9b\n"
        "\xe5\xbc\x95\xe6\x93\x8e: llama98",
    /* Same magic line as the English table, on purpose - see the EN
       table's comment on this entry. */
    /* LZ_STR_SAVE_HEADER */ LZ_CHATFILE_MAGIC,

    /* System Info - see the EN table's comment. Mechanism from the
       decompiled Winword.exe, wording ours. */
    /* LZ_STR_BTN_SYSINFO */ "\xe7\xb3\xbb\xe7\xbb\x9f\xe4\xbf\xa1\xe6\x81\xaf(&I)...",
    /* LZ_STR_DLG_SYSINFO_TITLE */ "\xe6\x98\x86\xe6\x98\x86\x39\x38 \xe7\xb3\xbb\xe7\xbb\x9f\xe4\xbf\xa1\xe6\x81\xaf",
    /* LZ_STR_SYSINFO_OS */ "Windows: %s",
    /* LZ_STR_SYSINFO_CPU */ "CPU: %s",
    /* LZ_STR_SYSINFO_MEM */ "\xe5\x86\x85\xe5\xad\x98: %s",
    /* LZ_STR_SYSINFO_DISK */ "\xe7\xa3\x81\xe7\x9b\x98: %s",
    /* LZ_STR_SYSINFO_TIER */ "\xe7\xae\x97\xe5\xad\x90: %s",

    /* Think-block dynamic temperature - same B mnemonic as
       the English table, kept in parens per the ZH convention. */
    /* LZ_STR_DLG_THINK_TEMP */ "\xe6\x80\x9d\xe8\x80\x83\xe5\x9d\x97\xe6\xb8\xa9\xe5\xba\xa6(&B)",
    /* LZ_STR_ERR_BAD_THINK_TEMP */ "\xe6\x80\x9d\xe8\x80\x83\xe5\x9d\x97\xe6\xb8\xa9\xe5\xba\xa6\xe5\xbf\x85\xe9\xa1\xbb\xe5\x9c\xa8 0 \xe5\x88\xb0 1.0 \xe4\xb9\x8b\xe9\x97\xb4",

    /* Apache-2.0 appendix boilerplate. The license text stays in its
       original English in BOTH tables - a legal text is not translated
       (a translation would change its legal effect); the copyright line
       carries the project's attribution. Identical EN and ZH entries are
       deliberate, so the two cannot drift apart. */
    /* LZ_STR_ABOUT_LICENSE */
        "Copyright (c) 2026 Lunzima\n"
        "Licensed under the Apache License, Version 2.0",
    /* LZ_STR_CAPTION_UNTITLED */ "\xe6\x9c\xaa\xe5\x91\xbd\xe5\x90\x8d\xe5\xaf\xb9\xe8\xaf\x9d",
    /* LZ_STR_DLG_BEEP */
        /* U+56DE U+590D U+5B8C U+6210 U+540E U+63D0 U+793A U+97F3
           "hui fu wan cheng hou ti shi yin" */
        "\xe5\x9b\x9e\xe5\xa4\x8d\xe5\xae\x8c\xe6\x88\x90\xe5\x90\x8e"
        "\xe6\x8f\x90\xe7\xa4\xba\xe9\x9f\xb3(&F)"
};

/* Static, not stack: the target's default stack is small enough that a
 * 4 KB local would matter. */
static char g_arena[LZ_STR_ARENA];
static char g_check[LZ_STR_ARENA];
static const char *g_display[LZ_STR_COUNT];
static int g_english = 0;      /* the deliverable is a Chinese app */
static int g_ready = 0;

static const char *const *table(int english) {
    return english ? LZ_STR_EN : LZ_STR_ZH;
}

int lz_str_init(int english) {
    const char *const *src;
    int used = 0, lossy = 0, i, overflow = 0;

    g_english = english ? 1 : 0;
    src = table(g_english);
    for (i = 0; i < LZ_STR_COUNT; i++) g_display[i] = 0;

    for (i = 0; i < LZ_STR_COUNT; i++) {
        const char *s = src[i] ? src[i] : "";
        int len = (int)strlen(s);
        int need = lz_gbk_from_utf8(s, len, NULL, 0, NULL);
        int back;

        if (used + need + 1 > LZ_STR_ARENA || len >= LZ_STR_ARENA) {
            overflow = 1;
            continue;
        }
        lz_gbk_from_utf8(s, len, g_arena + used, LZ_STR_ARENA - used, NULL);
        g_display[i] = g_arena + used;
        used += need + 1;

        /* Round-tripping is the only honest check for "did this string
           survive". Counting '?' in the output would flag every string
           that legitimately contains a question mark, and comparing
           lengths alone would miss a string that lost one character and
           whose replacement happened to restore the byte count. */
        back = lz_gbk_to_utf8(g_display[i], need, g_check, LZ_STR_ARENA,
                              NULL);
        if (back != len || memcmp(g_check, s, (size_t)len) != 0) lossy++;
    }

    /* Ready either way: a table that did not fit must not make every
       later call retry the whole conversion. The missing entries come
       back as empty strings, which is conspicuous, and the return value
       signals the loss. */
    g_ready = 1;
    return overflow ? -1 : lossy;
}

const char *lz_str_display(LZStr id) {
    if (!g_ready) lz_str_init(g_english);
    if ((int)id < 0 || (int)id >= LZ_STR_COUNT) return "";
    return g_display[id] ? g_display[id] : "";
}

const char *lz_str_utf8(LZStr id) {
    const char *s;
    if ((int)id < 0 || (int)id >= LZ_STR_COUNT) return "";
    s = table(g_english)[id];
    return s ? s : "";
}

int lz_str_lang_english(void) { return g_english; }
