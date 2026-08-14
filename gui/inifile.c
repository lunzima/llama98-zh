#include <windows.h>
#include <stdio.h>
#include <string.h>

#include "inifile.h"

static const char SECTION[] = "kunkun98";

int lz_ini_beside(char *out, int cap, const char *module) {
    int n, cut, i;
    if (!out || cap <= 0) return 0;
    out[0] = '\0';
    if (!module) return 0;
    n = (int)strlen(module);
    /* The extension is the last dot AFTER the last separator. A dot in
       a directory name ("C:\v1.0\kunkun98") is not an extension, and a
       plain strrchr would cut the path in half there. */
    cut = n;
    for (i = n - 1; i >= 0; i--) {
        if (module[i] == '\\' || module[i] == '/') break;
        if (module[i] == '.') { cut = i; break; }
    }
    if (cut + 4 + 1 > cap) return 0;
    memcpy(out, module, (size_t)cut);
    memcpy(out + cut, ".ini", 5);
    return 1;
}

int lz_ini_path(char *out, int cap) {
    static char cached[MAX_PATH + 8];
    static int ready = 0;
    if (!ready) {
        char mod[MAX_PATH];
        DWORD n = GetModuleFileNameA(NULL, mod, (DWORD)sizeof mod);
        if (n == 0 || n >= sizeof mod) return 0;
        if (!lz_ini_beside(cached, (int)sizeof cached, mod)) return 0;
        ready = 1;
    }
    if (!out || cap <= (int)strlen(cached)) return 0;
    strcpy(out, cached);
    return 1;
}

int lz_ini_get_int(const char *key, int def) {
    char path[MAX_PATH + 8];
    if (!lz_ini_path(path, (int)sizeof path)) return def;
    return (int)GetPrivateProfileIntA(SECTION, key, def, path);
}

void lz_ini_set_int(const char *key, int v) {
    char path[MAX_PATH + 8], buf[32];
    if (!lz_ini_path(path, (int)sizeof path)) return;
    sprintf(buf, "%d", v);
    WritePrivateProfileStringA(SECTION, key, buf, path);
}

int lz_ini_get_str(const char *key, char *out, int cap, const char *def) {
    char path[MAX_PATH + 8];
    if (!out || cap <= 0) return 0;
    out[0] = '\0';
    if (!lz_ini_path(path, (int)sizeof path)) {
        if (def) { strncpy(out, def, (size_t)cap - 1); out[cap - 1] = '\0'; }
        return (int)strlen(out);
    }
    return (int)GetPrivateProfileStringA(SECTION, key, def ? def : "",
                                         out, (DWORD)cap, path);
}

void lz_ini_set_str(const char *key, const char *v) {
    char path[MAX_PATH + 8];
    if (!lz_ini_path(path, (int)sizeof path)) return;
    WritePrivateProfileStringA(SECTION, key, v ? v : "", path);
}
