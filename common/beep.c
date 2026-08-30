#include "beep.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <stdio.h>
#endif /* _WIN32 */

void lz_beep(void) {
#ifdef _WIN32
    /* MB_OK is the "default beep" event, which is what a user who has
       themed their system expects to hear; MB_ICONASTERISK would be the
       information chime and is a louder claim than "done". */
    MessageBeep(MB_OK);
#else
    /* fflush because stderr is unbuffered on every target here EXCEPT a
       DOS build redirected to a file, and a beep that arrives at exit is
       not the signal that was asked for. */
    fputc('\a', stderr);
    fflush(stderr);
#endif /* _WIN32 */
}
