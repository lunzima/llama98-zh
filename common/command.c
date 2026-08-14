/* Slash command parsing. See command.h.
 *
 * Byte-oriented on purpose: the input arrives as GBK from an ANSI
 * control, and a GBK trail byte can land in the visible ASCII range
 * (0x40-0x7E). Nothing here inspects a byte beyond the command word,
 * which is ASCII by construction, so the trail-byte hazard never
 * applies - but the argument is copied out as BYTES, never scanned for
 * separators, which is what keeps it from applying later either.
 */
#include <string.h>

#include "command.h"

static const struct { const char *name; LZCmd id; } CMDS[] = {
    { "load",  LZ_CMD_LOAD },
    { "save",  LZ_CMD_SAVE },
    { "clear", LZ_CMD_CLEAR },
    { "stop",  LZ_CMD_STOP },
    { "temp",  LZ_CMD_TEMP },
    { "think", LZ_CMD_THINK },
    { "help",  LZ_CMD_HELP }
};
#define N_CMDS (int)(sizeof CMDS / sizeof CMDS[0])

static int is_blank(char c) { return c == ' ' || c == '\t' ||
                                     c == '\r' || c == '\n'; }

const char *lz_common_command_name(LZCmd c) {
    int i;
    for (i = 0; i < N_CMDS; i++)
        if (CMDS[i].id == c) return CMDS[i].name;
    return "";
}

LZCmd lz_common_parse_command(const char *line, int len,
                           const char **arg, int *arglen) {
    int i, word, a, b;
    LZCmd id = LZ_CMD_UNKNOWN;

    if (!line) return LZ_CMD_NONE;
    if (len < 0) len = (int)strlen(line);
    /* Leading blanks are skipped, but the slash still has to be the
       first non-blank: a reply that happens to mention a path partway
       through a sentence is not a command. */
    i = 0;
    while (i < len && is_blank(line[i])) i++;
    if (i >= len || line[i] != '/') return LZ_CMD_NONE;
    i++;

    word = i;
    while (i < len && !is_blank(line[i])) i++;

    for (a = 0; a < N_CMDS; a++) {
        int n = (int)strlen(CMDS[a].name);
        if (i - word == n &&
            memcmp(line + word, CMDS[a].name, (size_t)n) == 0) {
            id = CMDS[a].id;
            break;
        }
    }

    /* Argument: everything after the word, blanks trimmed off both
       ends. /load keeps its interior spaces because a path has them. */
    a = i;
    while (a < len && is_blank(line[a])) a++;
    b = len;
    while (b > a && is_blank(line[b - 1])) b--;
    if (arg) *arg = line + a;
    if (arglen) *arglen = b - a;
    return id;
}

int lz_common_parse_onoff(const char *arg, int arglen) {
    static const struct { const char *s; int v; } WORDS[] = {
        { "on", 1 }, { "off", 0 },
        { "1", 1 },  { "0", 0 },
        { "true", 1 }, { "false", 0 },
        { "yes", 1 },  { "no", 0 }
    };
    int i, n = (int)(sizeof WORDS / sizeof WORDS[0]);
    if (!arg || arglen <= 0) return -1;
    for (i = 0; i < n; i++) {
        int L = (int)strlen(WORDS[i].s);
        if (arglen == L && memcmp(arg, WORDS[i].s, (size_t)L) == 0)
            return WORDS[i].v;
    }
    return -1;
}
