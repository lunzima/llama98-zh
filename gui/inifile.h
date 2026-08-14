#ifndef LZ_GUI_INIFILE_H
#define LZ_GUI_INIFILE_H

/* kunkun98.ini - settings that survive a restart.
 *
 * BESIDE THE EXECUTABLE, not in %APPDATA%. That is where a 1997 program
 * kept them, and it is also the only place that exists on every target:
 * NT 3.51 has no reliable per-user application data directory, and the
 * deliverable is often a directory someone copied onto a machine.
 *
 * The profile API (GetPrivateProfileString and friends) is Win3.1-era,
 * so it needs nothing from compat40.c. It is also the reason values are
 * INTS and STRINGS and nothing else: the API has no float form, and
 * parsing one back would go through the C library's locale-dependent
 * decimal point. Temperature is therefore stored in thousandths.
 *
 * lz_ini_beside is separated from lz_ini_path so the naming rule can be
 * tested without Win32. */

/* Replace the extension of `module` with ".ini". Returns 1 on success,
 * 0 when `cap` is too small (nothing is written to out in that case
 * beyond a terminator). Pure: no Win32, no filesystem. */
int lz_ini_beside(char *out, int cap, const char *module);

/* lz_ini_beside applied to this process's own module path. Cached after
 * the first call. Returns 1 on success. */
int lz_ini_path(char *out, int cap);

/* All four read and write section "kunkun98" of that file. A read with
 * no file, no section or no key returns `def` - which is why every
 * caller passes the value the program would have used anyway, and why a
 * missing ini is indistinguishable from a fresh install. */
int  lz_ini_get_int(const char *key, int def);
void lz_ini_set_int(const char *key, int v);
int  lz_ini_get_str(const char *key, char *out, int cap, const char *def);
void lz_ini_set_str(const char *key, const char *v);

#endif
