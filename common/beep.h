#ifndef LZ_COMMON_BEEP_H
#define LZ_COMMON_BEEP_H

/* An audible "the reply is finished", for a machine slow enough that the
 * user goes and does something else while it generates.
 *
 * OPT-IN. Every caller holds its own flag and only calls this when the
 * user asked for it: a CLI that beeps by default is one that beeps in
 * the middle of a script, and a window that chimes uninvited is worse.
 *
 * PLATFORM, NOT FRONT END, picks the sound - the machine decides what it
 * has, not whether there is a window:
 *
 *   Win32 (gui AND the NT command line)   MessageBeep(MB_OK), the system
 *                                         sound the user configured. A
 *                                         console process gets it too.
 *   MS-DOS, and anything else             '\a' to stderr, which is the
 *                                         PC speaker on a DOS box and
 *                                         whatever the terminal does
 *                                         elsewhere.
 *
 * stderr rather than stdout on purpose: stdout is the generated text and
 * may be redirected into a file, where a stray 0x07 becomes part of the
 * output.
 *
 * Never fails and never blocks long enough to matter, so it returns
 * nothing - a caller that had to check would only be able to ignore it.
 */
void lz_beep(void);

#endif /* LZ_COMMON_BEEP_H */
