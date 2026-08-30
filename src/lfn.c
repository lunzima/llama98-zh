/* Long-file-name resolution. See lfn.h for what this exists for and why
   the directory is read rather than a short name constructed. */

#include <stdio.h>
#include <string.h>

/* Directory walk, one shape over two platform interfaces.
 *
 * dirent.h is POSIX and Visual C++ 4.0 does not have it - the compiler
 * stops at the include, so this was the one file in the no-assembly set
 * that would not build for the NT target. MSVC has had _findfirst in
 * <io.h> since long before 4.0, and it needs no windows.h.
 *
 * Three calls rather than an #ifdef around the loop: the matching logic
 * below is the part that is easy to get wrong, and it should exist once
 * regardless of which interface hands it the names. */
#if defined(_MSC_VER)
#include <io.h>
typedef struct { long h; struct _finddata_t fd; int first; } lz_dir;
#else
#include <dirent.h>
typedef struct { DIR *d; } lz_dir;
#endif /* _MSC_VER */

static int dir_open(lz_dir *it, const char *path) {
#if defined(_MSC_VER)
    char pat[520];
    size_t n = strlen(path);
    if (n + 5 >= sizeof pat) return 0;
    strcpy(pat, path);
    /* A trailing separator would double up; the caller passes "." for
       the current directory, which needs one added. */
    if (n > 0 && pat[n - 1] != '\\' && pat[n - 1] != '/') strcat(pat, "\\");
    strcat(pat, "*.*");
    it->h = _findfirst(pat, &it->fd);
    it->first = 1;
    return it->h != -1L;
#else
    it->d = opendir(path);
    return it->d != NULL;
#endif /* _MSC_VER */
}

static const char *dir_next(lz_dir *it) {
#if defined(_MSC_VER)
    if (it->first) { it->first = 0; return it->fd.name; }
    if (_findnext(it->h, &it->fd) != 0) return NULL;
    return it->fd.name;
#else
    struct dirent *e = readdir(it->d);
    return e ? e->d_name : NULL;
#endif /* _MSC_VER */
}

static void dir_close(lz_dir *it) {
#if defined(_MSC_VER)
    if (it->h != -1L) _findclose(it->h);
#else
    if (it->d) closedir(it->d);
#endif /* _MSC_VER */
}

#include "err.h"
#include "lfn.h"

/* The 8.3 shape, spelled out once so the matcher below cannot drift from
   the thing it is matching against. */
#define LZ_83_BASE 8
#define LZ_83_EXT  3

/* Longest directory entry this carries. Not NAME_MAX: that is 12 on a
   plain DOS build and 259 elsewhere, so the same source would size two
   different buffers, and the larger is the one that has to fit. */
#define LZ_LFN_ENTRY 260

/* ASCII case fold. Not tolower(), which is locale-dependent - a Chinese
   locale's case rules have no business deciding whether a filename
   matched. Not stricmp/strcasecmp either: the two toolchains spell it
   differently and neither spelling is C99. */
static int fold(int c) {
    return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c;
}

/* Case-insensitive compare of at most n bytes; also requires both sides
   to END at n, so "jso" does not match "json" merely by prefix. */
static int ci_eq_n(const char *a, const char *b, int n) {
    int i;
    for (i = 0; i < n; i++) {
        if (!a[i] || !b[i]) return 0;
        if (fold((unsigned char)a[i]) != fold((unsigned char)b[i])) return 0;
    }
    return a[n] == '\0';
}

/* Is `p` (length pn) a case-insensitive prefix of `s`? */
static int ci_prefix(const char *p, int pn, const char *s) {
    int i;
    for (i = 0; i < pn; i++) {
        if (!s[i]) return 0;
        if (fold((unsigned char)p[i]) != fold((unsigned char)s[i])) return 0;
    }
    return 1;
}

/* Split at the LAST dot: "model.safetensors" -> base "model", ext
   "safetensors". A name with no dot gets an empty extension, and a
   leading dot is part of the base (".gitignore" has no extension), which
   is what both DOS and Win32 do. *ext points into `name`. */
static void split_name(const char *name, int *base_len, const char **ext) {
    const char *dot = NULL;
    const char *p;
    for (p = name; *p; p++)
        if (*p == '.' && p != name) dot = p;
    if (dot) {
        *base_len = (int)(dot - name);
        *ext = dot + 1;
    } else {
        *base_len = (int)strlen(name);
        *ext = name + strlen(name);
    }
}

/* Does entry `e` look like the 8.3 mangling of `want`?
 *
 * The extension must equal the wanted one's first three characters -
 * that is why .json breaks, "json" being four and becoming "JSO".
 *
 * The base is accepted in either shape a filesystem produces: plain
 * truncation to eight, or <prefix>~<digits> where the part before the
 * tilde is a prefix of the wanted base. The digits are not checked
 * against anything - there is nothing to check them against (lfn.h). */
static int is_83_of(const char *e, const char *want) {
    int wb, eb;
    const char *wext, *eext;
    int keep, i;

    split_name(want, &wb, &wext);
    split_name(e, &eb, &eext);

    keep = (int)strlen(wext);
    if (keep > LZ_83_EXT) keep = LZ_83_EXT;
    if (!ci_eq_n(eext, wext, keep)) return 0;

    /* plain truncation */
    keep = wb;
    if (keep > LZ_83_BASE) keep = LZ_83_BASE;
    if (eb == keep && ci_prefix(e, eb, want)) return 1;

    /* numbered form */
    if (eb > LZ_83_BASE) return 0;
    for (i = 0; i < eb; i++) {
        if (e[i] != '~') continue;
        if (i == 0) return 0;                    /* "~1.JSO" names nothing */
        if (!ci_prefix(e, i, want)) return 0;    /* prefix must be the base's */
        if (i + 1 >= eb) return 0;               /* a tilde with no number */
        for (++i; i < eb; i++)
            if (e[i] < '0' || e[i] > '9') return 0;
        return 1;
    }
    return 0;
}

/* Build <dir>/<name> into out. Returns 0, or the byte count needed when
   it does not fit (which the caller reports as LZ_ERR_TRUNC). */
static int join(const char *dir, const char *name, char *out, int cap) {
    int dn = (int)strlen(dir);
    int nn = (int)strlen(name);
    int sep, need;

    /* Trailing separators collapse, EXCEPT the one that IS the root.
       "C:\" names the root; "C:" is drive-relative - the current
       directory of drive C, a different place. lz_pick_folder hands back
       the root with its separator for this reason. */
    while (dn > 0 && (dir[dn - 1] == '/' || dir[dn - 1] == '\\')) {
        if (dn == 1) break;                       /* "\" or "/" */
        if (dn == 3 && dir[1] == ':') break;      /* "C:\" */
        dn--;
    }

    /* Only add a separator when the directory does not already end in
       one - the root cases above kept theirs. */
    sep = (dn > 0 && dir[dn - 1] != '/' && dir[dn - 1] != '\\') ? 1 : 0;
    need = dn + sep + nn + 1;
    if (need > cap) return need;

    if (dn > 0) memcpy(out, dir, (size_t)dn);
    /* Forward slash, matching every path this engine already builds by
       hand - src/model.c records why the two must be identical. */
    if (sep) out[dn] = '/';
    memcpy(out + dn + sep, name, (size_t)nn + 1);
    return 0;
}

static int can_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

int lz_lfn_path(const char *dir, const char *want,
                char *out, int cap, char *errbuf, int errlen) {
    lz_dir it;
    const char *nm;
    char found[LZ_LFN_ENTRY];
    int have_exact = 0, n_83 = 0;
    int rc, need;

    if (!dir || !want || !out || cap <= 0) {
        LZ_ERR_SET(rc, errbuf, errlen, LZ_ERR_NULL_ARG);
        return rc;
    }

    /* 1. The name as asked for. Every LFN-capable volume answers here,
          so the ordinary case never opens the directory at all. */
    need = join(dir, want, out, cap);
    if (need != 0) {
        LZ_ERR_SET1(rc, errbuf, errlen, LZ_ERR_TRUNC, need);
        return rc;
    }
    if (can_open(out)) return 0;

    /* 2/3. Read the directory. A directory that cannot be opened reports
           "not found": from the caller's side the two are the same fact,
           and a second code would add a branch that changes nothing. */
    found[0] = '\0';
    if (dir_open(&it, dir[0] ? dir : ".")) {
        while ((nm = dir_next(&it)) != NULL) {
            if (nm[0] == '\0') continue;
            if ((int)strlen(nm) >= LZ_LFN_ENTRY) continue;
            /* An exact hit ignoring case wins outright and stops the
               scan: it is the name that was asked for, so no 8.3
               candidate can be a better answer and the ambiguity count
               below must not be able to veto it. */
            if (ci_eq_n(nm, want, (int)strlen(want))) {
                strcpy(found, nm);
                have_exact = 1;
                break;
            }
            if (is_83_of(nm, want)) {
                n_83++;
                if (n_83 == 1) strcpy(found, nm);
            }
        }
        dir_close(&it);
    }

    if (!have_exact) {
        if (n_83 > 1) {
            /* Two files in one directory mangle to the same 8.3 name -
               tokenizer.json and tokenizer_config.json are the pair this
               was written for. Nothing here can say which was meant, so
               nothing here picks. */
            LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_LFN_AMBIGUOUS,
                        want, n_83);
            return rc;
        }
        if (n_83 == 0) {
            LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_LFN_NOT_FOUND, want, dir);
            return rc;
        }
    }

    need = join(dir, found, out, cap);
    if (need != 0) {
        LZ_ERR_SET1(rc, errbuf, errlen, LZ_ERR_TRUNC, need);
        return rc;
    }
    /* The match is confirmed by opening it, which is also what keeps a
       SUBDIRECTORY whose name happens to mangle the same way from being
       handed back as a file. dirent carries no portable type field -
       Watcom has d_attr, POSIX has an optional d_type - so the open is
       both the cheaper test and the only one that means the same thing
       on every target this builds for. */
    if (!can_open(out)) {
        out[0] = '\0';
        LZ_ERR_SET2(rc, errbuf, errlen, LZ_ERR_LFN_NOT_FOUND, want, dir);
        return rc;
    }
    return 0;
}

int lz_lfn_exists(const char *dir, const char *want) {
    char path[LZ_LFN_ENTRY + 512];
    return lz_lfn_path(dir, want, path, (int)sizeof path, NULL, 0) == 0;
}
