#ifndef LZ_CLI_ARGV_H
#define LZ_CLI_ARGV_H

/* Re-decodes a narrow argv as GBK -> UTF-8.
 *
 * NOT "a Windows process's", which this line said and cli_main.c:1084
 * had already corrected in prose from the DOS side: there is no Win32
 * in the body, and the DOS arm calls it for the same reason with a
 * literal 936.
 *
 * NAMED lz_argv_*, NOT lz_gbk_*. llama98.def's "3.7.1 GBK transcoding"
 * section makes lz_gbk_* the DLL's exported transcoder pair -
 * lz_gbk_from_utf8 and lz_gbk_to_utf8, both in src/gbk.c, which opens
 * "no OS calls, no allocation". This is an unexported CLI-entry helper
 * that mallocs and leaks by design; it uses that pair, it is not one of
 * them. The new name also says the direction, which every other member
 * of that family does and `_convert` did not.
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
char **lz_argv_gbk_to_utf8(int cp, int argc, char **argv);

#endif
