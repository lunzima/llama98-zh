#ifndef LZ_GUI_COMMAND_H
#define LZ_GUI_COMMAND_H

/* The slash command layer.
 *
 * Parsing only - no window, no engine, no state. The dispatch lives in
 * gui/main.c next to the menu handlers, because the two should be the
 * same code path: a command that does something the menu
 * cannot, or a menu item with no command, is how the two drift into
 * disagreeing about what the program does.
 *
 * A line is a command when its FIRST character is '/'. That is the whole
 * rule, and it means a message beginning with a slash cannot be sent -
 * the same bargain every chat client of this era made.
 */

typedef enum {
    LZ_CMD_NONE = 0,     /* not a command line at all */
    LZ_CMD_UNKNOWN,      /* starts with '/' but names nothing */
    LZ_CMD_LOAD,         /* /load <path>  - argument is the WHOLE rest */
    LZ_CMD_SAVE,
    LZ_CMD_CLEAR,
    LZ_CMD_STOP,
    LZ_CMD_TEMP,         /* /temp <0-1> */
    LZ_CMD_THINK,        /* /think on|off */
    LZ_CMD_HELP,
    LZ_CMD_COUNT
} LZCmd;

/* Parse one line. `arg` / `arglen` receive the argument text with
 * surrounding blanks removed; for /load that is the entire remainder,
 * spaces and all, because a path may contain them and quoting rules are
 * one more thing to get wrong.
 *
 * Neither output is written when there is no argument; pass NULL for
 * either if it is not wanted. `len` < 0 means NUL-terminated. */
LZCmd lz_common_parse_command(const char *line, int len,
                           const char **arg, int *arglen);

/* The command's name without the slash, for a message or a test.
 * Never NULL. */
const char *lz_common_command_name(LZCmd c);

/* on / off / 1 / 0 / true / false -> 1 or 0; -1 when it is none of
 * those. /think with an unparseable argument must say so rather than
 * pick a side. */
int lz_common_parse_onoff(const char *arg, int arglen);

#endif
