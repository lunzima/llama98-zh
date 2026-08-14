/* See savechat.h. */
#include <string.h>

#include "savechat.h"

int lz_common_crlf(const char *in, int len, char *out, int cap) {
    int i = 0, need = 0, wrote = 0;

    if (!in) return 0;
    if (len < 0) len = (int)strlen(in);

    while (i < len) {
        char c = in[i];
        int emit_break = 0;

        if (c == '\r') {
            /* CRLF is one break, not two. Treating them separately is
               how a file ends up with blank lines between every line. */
            i += (i + 1 < len && in[i + 1] == '\n') ? 2 : 1;
            emit_break = 1;
        } else if (c == '\n') {
            i++;
            emit_break = 1;
        } else {
            i++;
        }

        if (emit_break) {
            if (out && wrote + 2 < cap) {
                out[wrote++] = '\r';
                out[wrote++] = '\n';
            }
            need += 2;
        } else {
            if (out && wrote + 1 < cap) out[wrote++] = c;
            need += 1;
        }
    }
    if (out && cap > 0) out[wrote] = '\0';
    return need;
}

int lz_common_lf(const char *in, int len, char *out, int cap) {
    int i = 0, need = 0, wrote = 0;

    if (!in) return 0;
    if (len < 0) len = (int)strlen(in);

    while (i < len) {
        char c = in[i];

        if (c == '\r') {
            /* CRLF is one break, and so is a lone CR - the same rule
               lz_common_crlf applies in the other direction. Emitting LF
               for the CR and then again for the LF would double every
               line break instead of normalising it. */
            i += (i + 1 < len && in[i + 1] == '\n') ? 2 : 1;
            c = '\n';
        } else {
            i++;
        }

        if (out && wrote + 1 < cap) out[wrote++] = c;
        need += 1;
    }
    if (out && cap > 0) out[wrote] = '\0';
    return need;
}
