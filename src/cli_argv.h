#ifndef LZ_CLI_ARGV_H
#define LZ_CLI_ARGV_H

/* Re-decodes a Windows process's narrow argv as GBK -> UTF-8.
 *
 * Pure: takes the ANSI code page as a PARAMETER rather than calling
 * GetACP() itself, so it is testable on any machine regardless of that
 * machine's own locale, and so cli_main.c stays the only file in this
 * translation unit pair that touches windows.h.
 *
 * Only known correct when cp == 936 (GBK) - a Windows process's narrow
 * argv is the OS's canonical (wide) command line down-converted to the
 * current ANSI code page), and lz_gbk_to_utf8 only knows GBK. Returns
 * NULL for any other cp, and NULL on allocation failure; either way the
 * caller's own argv is unchanged and must keep being used - falling
 * back to the untouched narrow argv is the existing, long-standing
 * behaviour, not a new failure mode this function introduces.
 *
 * On success, returns a freshly malloc'd array of argc entries, each a
 * freshly malloc'd UTF-8 C string. Leaks by design, same convention as
 * the rest of cli_main.c's argv handling: the result lives for the life
 * of the process and a pile of `const char *` end up pointing into it.
 */
char **lz_gbk_argv_convert(int cp, int argc, char **argv);

#endif
