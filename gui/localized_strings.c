/* User-visible string table for the front end (iron law 7), bilingual.
 *
 * Laid out like src/err.c: enum-indexed parallel arrays, designated
 * initialisers so a reordered enum cannot silently shift the text.
 *
 * The Chinese here is DATA, not annotation - same exemption err.c's
 * table has. Comments in this file are
 * still English, and the checker still enforces that: the exemption is
 * per line, not per file.
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
    [LZ_STR_APP_TITLE]            = "Kunkun98",

    /* Mnemonics. Every Win9x menu had them; a menu bar with no
       underlined letters is the single most visible way this front end
       fails to look like its target. The accelerator text after \t is
       right-aligned by the menu and must agree with the ACCELERATORS
       table in gui/kunkun98.rc - the selftest checks that both exist. */
    [LZ_STR_MENU_FILE]            = "&File",
    [LZ_STR_MENU_EDIT]            = "&Edit",
    [LZ_STR_MENU_MODEL]           = "&Model",
    [LZ_STR_MENU_SETTINGS_TITLE]  = "&Settings",
    [LZ_STR_MENU_HELP]            = "&Help",

    [LZ_STR_MENU_OPEN_MODEL]      = "&Open Model...\tCtrl+O",
    [LZ_STR_MENU_OPEN_CHAT]       = "&Open Conversation...",
    [LZ_STR_MENU_SAVE_CHAT]       = "&Save Conversation...\tCtrl+S",
    [LZ_STR_MENU_STOP]            = "S&top Generating\tEsc",
    [LZ_STR_MENU_CLEAR]           = "&Clear Conversation\tCtrl+N",
    [LZ_STR_MENU_SETTINGS]        = "&Inference Settings...",
    [LZ_STR_MENU_ABOUT]           = "&About",
    [LZ_STR_MENU_EXIT]            = "E&xit",
    [LZ_STR_MENU_LANG_ZH]         = "中文(&C)",
    [LZ_STR_MENU_LANG_EN]         = "&English",

    [LZ_STR_MENU_COPY]            = "&Copy\tCtrl+C",
    [LZ_STR_MENU_SELECT_ALL]      = "Select &All\tCtrl+A",
    [LZ_STR_MENU_FIND]            = "&Find...\tCtrl+F",

    /* No "\tCtrl+..." tail on any of the three - see the note beside
       IDM_REGEN in gui/resource.h for why they have no accelerator to
       advertise. */
    [LZ_STR_MENU_REGEN]           = "&Regenerate",
    [LZ_STR_MENU_EDIT_LAST]       = "&Edit Last Message",
    [LZ_STR_MENU_DEL_LAST]        = "&Delete Last Exchange",

    [LZ_STR_BTN_OPEN]             = "Open",
    [LZ_STR_BTN_SAVE]             = "Save",
    [LZ_STR_BTN_SEND]             = "Send",
    [LZ_STR_BTN_STOP]             = "Stop",
    [LZ_STR_BTN_CLEAR]            = "Clear",
    /* Shorter than the menu's own word, same reason LZ_STR_BTN_SETTINGS
       is shorter than LZ_STR_MENU_SETTINGS just below: this caption sits
       under a 24-pixel glyph in a button sized for two Chinese
       characters, and "Regenerate" does not fit in it. */
    [LZ_STR_BTN_REGEN]            = "Retry",
    /* "Options", not "Settings", and the toolbar is the reason: the
       strip's captions sit under a 24-pixel glyph in a button whose
       width floor is set for two Chinese characters, and "Settings"
       comes back out of the control as "Sett...". It is also the period's
       own word - Tools > Options is where a 1997 program put this. The
       MENU keeps the longer "Inference Settings...", which has the room
       and needs the qualifier. */
    [LZ_STR_BTN_SETTINGS]         = "Options",
    [LZ_STR_BTN_OK]               = "OK",
    [LZ_STR_BTN_CANCEL]           = "Cancel",
    [LZ_STR_BTN_RESTORE_DEFAULT]  = "&Restore Defaults",

    /* The dialog gets mnemonics too - its loop calls IsDialogMessage,
       which is what makes Alt+letter reach a control. The tool-strip
       buttons deliberately do NOT: they duplicate menu commands, Alt
       goes to the menu bar first, and a Win9x toolbar had no mnemonics
       of its own either. */
    [LZ_STR_DLG_ABOUT_TITLE]      = "About Kunkun98",
    [LZ_STR_DLG_SETTINGS_TITLE]   = "Inference Settings",
    [LZ_STR_DLG_THINK]            = "&Thinking (think)",
    [LZ_STR_DLG_TEMPERATURE]      = "Tem&perature",
    [LZ_STR_DLG_TEMP_CAP]         = "Temperature is capped at 1.0",
    /* Mnemonics inside this dialog must not collide with each other:
       T think, P temperature, O top-p, E repetition, N context,
       M max new, R restore defaults. A STATIC's mnemonic moves focus
       to the next tab stop in Z-order, which is why each label is
       created immediately before its own controls. */
    [LZ_STR_DLG_CONTEXT]          = "Co&ntext",
    [LZ_STR_DLG_CTX_NOTE]         = "512 to 32768; larger needs more memory",
    [LZ_STR_DLG_TOPP]             = "T&op-p",
    [LZ_STR_DLG_REP]              = "Repetition p&enalty",
    [LZ_STR_DLG_MAXNEW]           = "&Max new tokens",
    [LZ_STR_DLG_MAXNEW_NOTE]      = "-1 = unlimited (until EOS or the "
                                    "context fills)",
    [LZ_STR_DLG_SYS]              = "&System prompt",
    /* No LZ_STR_DLG_SYS_NOTE entry: the note control it fed is gone
       (settingsdlg.c), and the enum member stays in localized_strings.h
       so the ids do not shift. */

    [LZ_STR_DLG_OPEN_MODEL_TITLE] = "Select a model directory",
    [LZ_STR_DLG_SAVE_CHAT_TITLE]  = "Save conversation",
    [LZ_STR_DLG_OPEN_CHAT_TITLE]  = "Open conversation",
    [LZ_STR_DLG_TEXT_FILTER]      = "Text files (*.txt)",
    [LZ_STR_FILTER_MODEL]         = "Kunkun98 model (model.bin)",

    [LZ_STR_STATE_NO_MODEL]       = "no model",
    [LZ_STR_STATE_LOADING]        = "loading...",
    [LZ_STR_STATE_READY]          = "ready",
    [LZ_STR_STATE_GENERATING]     = "generating",
    [LZ_STR_STATE_PREFILL]        = "processing context",
    [LZ_STR_STATE_CTX]            = "context",
    [LZ_STR_STATE_TOKCELL]        = "generating %d tok, %.1f tok/s",

    [LZ_STR_SIDE_CAND]            = "Candidates",

    [LZ_STR_SPEAKER_USER]         = "You",
    [LZ_STR_SPEAKER_ASSISTANT]    = "Kunkun98",
    [LZ_STR_SYS_MODEL_LOADED]     = "model loaded: %s (KV %dMB)",
    [LZ_STR_SYS_CTX_TRIMMED]      = "context trimmed",
    [LZ_STR_SYS_GEN_STOPPED]      = "generation stopped",
    [LZ_STR_SYS_TEMP_SET]         = "temperature set to %s",
    [LZ_STR_SYS_THINK_ON]         = "thinking: on",
    [LZ_STR_SYS_THINK_OFF]        = "thinking: off",

    [LZ_STR_ERR_TITLE]            = "Error",
    [LZ_STR_ERR_NO_RICHEDIT]      = "neither riched20.dll nor riched32.dll "
                                    "could be loaded; Kunkun98 cannot show "
                                    "a conversation without one",
    [LZ_STR_ERR_NO_MODEL_BIN]     = "that directory has no model.bin",
    [LZ_STR_ERR_NO_MODEL_LOADED]  = "no model is loaded yet - use "
                                    "Open Model (Ctrl+O) or /load <path>",
    [LZ_STR_ERR_BUSY]             = "still generating; stop it first",
    [LZ_STR_ERR_UNKNOWN_CMD]      = "unknown command: %s",
    [LZ_STR_ERR_BAD_TEMP]         = "temperature must be between 0 and 1.0",
    [LZ_STR_ERR_BAD_CTX]          = "context must be between 512 and 32768",
    [LZ_STR_ERR_BAD_TOPP]         = "top-p must be between 0.05 and 1.0",
    [LZ_STR_ERR_BAD_REP]          = "repetition penalty must be between "
                                    "1.0 and 1.5",
    [LZ_STR_ERR_BAD_MAXNEW]       = "max new tokens must be -1, or between "
                                    "16 and the context window",
    [LZ_STR_ERR_BAD_ONOFF]        = "use on or off",
    [LZ_STR_ERR_SAVE_FAILED]      = "could not write that file",
    [LZ_STR_FIND_NOT_FOUND]       = "cannot find that text",

    [LZ_STR_HELP_BODY] =
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
    [LZ_STR_ABOUT_BODY] =
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
    [LZ_STR_SAVE_HEADER]          = LZ_CHATFILE_MAGIC,

    /* System Info - see localized_strings.h. Own wording, Word 95's
       mechanism. The %s slots are the values the code fetches
       with GetVersionExA / GetSystemInfo / GlobalMemoryStatus /
       GetDiskFreeSpaceA, the same calls the decompiled Winword.exe uses. */
    [LZ_STR_BTN_SYSINFO]          = "System Info...",
    [LZ_STR_DLG_SYSINFO_TITLE]    = "Kunkun98 System Info",
    [LZ_STR_SYSINFO_OS]           = "Windows: %s",
    [LZ_STR_SYSINFO_CPU]          = "CPU: %s",
    [LZ_STR_SYSINFO_MEM]          = "Memory: %s",
    [LZ_STR_SYSINFO_DISK]         = "Disk: %s",

    /* Think-block dynamic temperature. Mnemonic B - the
       dialog's other controls use T P O E N M R S, and the collision
       check in gui/main.c's selftest enforces that. */
    [LZ_STR_DLG_THINK_TEMP]       = "Think-&block temp",
    [LZ_STR_ERR_BAD_THINK_TEMP]   = "think-block temperature must be "
                                    "between 0 and 1.0",

    /* The Apache-2.0 appendix boilerplate, rendered bottom-left of the
       About dialog in Small Fonts 7pt. The copyright owner is the
       project's own attribution (see the about body's own comment). */
    [LZ_STR_ABOUT_LICENSE] =
        "Copyright (c) 2026 Lunzima\n"
        "Licensed under the Apache License, Version 2.0",
    [LZ_STR_CAPTION_UNTITLED]     = "Untitled Chat",
};

static const char *const LZ_STR_ZH[LZ_STR_COUNT] = {
    [LZ_STR_APP_TITLE]            = "昆昆98",

    /* Chinese Windows puts the mnemonic in parentheses after the word -
       the label renders as the word followed by (F) with F underlined.
       Not a liberty taken in translation: every menu in Simplified
       Chinese Win9x looks like this, because a Han character has no
       Latin letter of its own to underline. */
    [LZ_STR_MENU_FILE]            = "文件(&F)",
    [LZ_STR_MENU_EDIT]            = "编辑(&E)",
    [LZ_STR_MENU_MODEL]           = "模型(&M)",
    [LZ_STR_MENU_SETTINGS_TITLE]  = "设置(&S)",
    [LZ_STR_MENU_HELP]            = "帮助(&H)",

    [LZ_STR_MENU_OPEN_MODEL]      = "打开模型(&O)…\tCtrl+O",
    [LZ_STR_MENU_OPEN_CHAT]       = "打开对话(&O)…",
    [LZ_STR_MENU_SAVE_CHAT]       = "保存对话(&S)…\tCtrl+S",
    [LZ_STR_MENU_STOP]            = "停止生成(&T)\tEsc",
    [LZ_STR_MENU_CLEAR]           = "清空对话(&C)\tCtrl+N",
    [LZ_STR_MENU_SETTINGS]        = "推理设置(&I)…",
    [LZ_STR_MENU_ABOUT]           = "关于(&A)",
    [LZ_STR_MENU_EXIT]            = "退出(&X)",
    [LZ_STR_MENU_LANG_ZH]         = "中文(&C)",
    [LZ_STR_MENU_LANG_EN]         = "&English",

    [LZ_STR_MENU_COPY]            = "复制(&C)\tCtrl+C",
    [LZ_STR_MENU_SELECT_ALL]      = "全选(&A)\tCtrl+A",
    [LZ_STR_MENU_FIND]            = "查找(&F)…\tCtrl+F",

    [LZ_STR_MENU_REGEN]           = "重新生成(&R)",
    [LZ_STR_MENU_EDIT_LAST]       = "修改上一句(&E)",
    [LZ_STR_MENU_DEL_LAST]        = "删除上一轮(&D)",

    [LZ_STR_BTN_OPEN]             = "打开",
    [LZ_STR_BTN_SAVE]             = "保存",
    [LZ_STR_BTN_SEND]             = "发送",
    [LZ_STR_BTN_STOP]             = "停止",
    [LZ_STR_BTN_CLEAR]            = "清空",
    [LZ_STR_BTN_REGEN]            = "重试",
    [LZ_STR_BTN_SETTINGS]         = "设置",
    [LZ_STR_BTN_OK]               = "确定",
    [LZ_STR_BTN_CANCEL]           = "取消",
    [LZ_STR_BTN_RESTORE_DEFAULT]  = "恢复默认(&R)",

    [LZ_STR_DLG_ABOUT_TITLE]      = "关于 昆昆98",
    [LZ_STR_DLG_SETTINGS_TITLE]   = "推理设置",
    /* Half-width parentheses and colons throughout the Chinese table.
       Full-width ones are a full em wide with the glyph hugging one
       side, and in SimSun's 12-pixel BITMAP faces - which is what the
       target actually renders with, not the outline face a modern host
       substitutes - that reads as a hole in the line. Chinese commas and
       full stops are left alone; the complaint is specific to the marks
       whose full-width form is mostly whitespace. */
    [LZ_STR_DLG_THINK]            = "深度思考(think)(&T)",
    [LZ_STR_DLG_TEMPERATURE]      = "温度(&P)",
    [LZ_STR_DLG_TEMP_CAP]         = "温度上限 1.0",
    [LZ_STR_DLG_CONTEXT]          = "上下文(&N)",
    [LZ_STR_DLG_CTX_NOTE]         = "范围 512 - 32768，越大越占内存",
    [LZ_STR_DLG_TOPP]             = "Top-p(&O)",
    [LZ_STR_DLG_REP]              = "重复惩罚(&E)",
    [LZ_STR_DLG_MAXNEW]           = "最大生成长度(&M)",
    [LZ_STR_DLG_MAXNEW_NOTE]      = "-1 = 不限（直到 EOS 或上下文占满）",
    [LZ_STR_DLG_SYS]              = "系统提示词(&S)",

    [LZ_STR_DLG_OPEN_MODEL_TITLE] = "选择模型目录",
    [LZ_STR_DLG_SAVE_CHAT_TITLE]  = "保存对话",
    [LZ_STR_DLG_OPEN_CHAT_TITLE]  = "打开对话",
    [LZ_STR_DLG_TEXT_FILTER]      = "文本文件 (*.txt)",
    [LZ_STR_FILTER_MODEL]         = "昆昆98 模型 (model.bin)",

    [LZ_STR_STATE_NO_MODEL]       = "未加载",
    [LZ_STR_STATE_LOADING]        = "加载中…",
    [LZ_STR_STATE_READY]          = "已加载",
    [LZ_STR_STATE_GENERATING]     = "生成中",
    [LZ_STR_STATE_PREFILL]        = "处理上下文",
    [LZ_STR_STATE_CTX]            = "上下文",
    /* The ZH table DOES translate this one - the state word is ordinary
       Han and GBK-encodable. The cell uses a comma rather than a `·`
       separator (the user asked to avoid dot separators), so it needs
       no mapping and cannot mojibake. What stays ASCII is the TOKEN
       UNIT "tok" and the "/s" - there is no standard localisation for
       a token count unit, and every token-count display in this
       program (the context cell, the candidate list) already uses the
       ASCII "tok". */
    [LZ_STR_STATE_TOKCELL] =
        "\xe7\x94\x9f\xe6\x88\x90\xe4\xb8\xad %d tok, %.1f tok/s",

    [LZ_STR_SIDE_CAND]            = "候选",

    [LZ_STR_SPEAKER_USER]         = "你",
    [LZ_STR_SPEAKER_ASSISTANT]    = "昆昆98",
    [LZ_STR_SYS_MODEL_LOADED]     = "模型已加载: %s (KV %dMB)",
    [LZ_STR_SYS_CTX_TRIMMED]      = "上下文已裁剪",
    [LZ_STR_SYS_GEN_STOPPED]      = "已停止生成",
    [LZ_STR_SYS_TEMP_SET]         = "温度已设为 %s",
    [LZ_STR_SYS_THINK_ON]         = "深度思考: 开",
    [LZ_STR_SYS_THINK_OFF]        = "深度思考: 关",

    [LZ_STR_ERR_TITLE]            = "错误",
    [LZ_STR_ERR_NO_RICHEDIT]      = "riched20.dll 与 riched32.dll 都加载不了，"
                                    "昆昆98 没有它们就无法显示对话",
    [LZ_STR_ERR_NO_MODEL_BIN]     = "该目录里没有 model.bin",
    [LZ_STR_ERR_NO_MODEL_LOADED]  = "还没有加载模型 —— 用「打开模型」"
                                    "(Ctrl+O) 或 /load <路径>",
    [LZ_STR_ERR_BUSY]             = "正在生成，请先停止",
    [LZ_STR_ERR_UNKNOWN_CMD]      = "未知命令: %s",
    [LZ_STR_ERR_BAD_TEMP]         = "温度必须在 0 到 1.0 之间",
    [LZ_STR_ERR_BAD_CTX]          = "上下文窗口必须在 512 到 32768 之间",
    [LZ_STR_ERR_BAD_TOPP]         = "Top-p 必须在 0.05 到 1.0 之间",
    [LZ_STR_ERR_BAD_REP]          = "重复惩罚必须在 1.0 到 1.5 之间",
    [LZ_STR_ERR_BAD_MAXNEW]       = "最大生成长度必须是 -1，"
                                    "或介于 16 与上下文窗口之间",
    [LZ_STR_ERR_BAD_ONOFF]        = "请用 on 或 off",
    [LZ_STR_ERR_SAVE_FAILED]      = "写不了这个文件",
    [LZ_STR_FIND_NOT_FOUND]       = "找不到这段文字",

    [LZ_STR_HELP_BODY] =
        "/load <路径>   打开模型\n"
        "/save          保存对话\n"
        "/clear         清空对话\n"
        "/stop          停止生成\n"
        "/temp <0-1>    温度 (1.0 是硬上限)\n"
        "/think on|off  深度思考开关\n"
        "/help          列出命令",

    [LZ_STR_ABOUT_BODY] =
        "昆昆98 (Kunkun98)\n"
        "for Windows, 版本 0.1\n"
        "\n"
        "隐空间混合专家 + 线性注意力\n"
        "引擎: llama98",
    /* Same magic line as the English table, on purpose - see the EN
       table's comment on this entry. */
    [LZ_STR_SAVE_HEADER]          = LZ_CHATFILE_MAGIC,

    /* System Info - see the EN table's comment. Mechanism from the
       decompiled Winword.exe, wording ours. */
    [LZ_STR_BTN_SYSINFO]          = "系统信息(&I)...",
    [LZ_STR_DLG_SYSINFO_TITLE]    = "昆昆98 系统信息",
    [LZ_STR_SYSINFO_OS]           = "Windows: %s",
    [LZ_STR_SYSINFO_CPU]          = "CPU: %s",
    [LZ_STR_SYSINFO_MEM]          = "内存: %s",
    [LZ_STR_SYSINFO_DISK]         = "磁盘: %s",

    /* Think-block dynamic temperature - same B mnemonic as
       the English table, kept in parens per the ZH convention. */
    [LZ_STR_DLG_THINK_TEMP]       = "思考块温度(&B)",
    [LZ_STR_ERR_BAD_THINK_TEMP]   = "思考块温度必须在 0 到 1.0 之间",

    /* Apache-2.0 appendix boilerplate. The license text stays in its
       original English in BOTH tables - a legal text is not translated
       (a translation would change its legal effect); the copyright line
       carries the project's attribution. Identical EN and ZH entries are
       deliberate, so the two cannot drift apart. */
    [LZ_STR_ABOUT_LICENSE] =
        "Copyright (c) 2026 Lunzima\n"
        "Licensed under the Apache License, Version 2.0",
    [LZ_STR_CAPTION_UNTITLED]     = "未命名对话",
};

/* Static, not stack: iron law six, and the target's default stack is
 * small enough that a 4 KB local would matter. */
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
