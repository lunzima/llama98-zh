#ifndef LZ_LFN_H
#define LZ_LFN_H

/* Resolving <dir>/<name> on a volume that cannot store long names.
 *
 * A model directory copied onto such a volume keeps only the 8.3 form of
 * every name that does not fit: "tokenizer.json" becomes "TOKENI~1.JSO",
 * and appending the literal long name then opens nothing. Of the four
 * names the loader wants only "model.bin" is 8.3-legal, so that one
 * survives - a property of that filename, not something to rely on.
 *
 * ENUMERATION, NOT A CONSTRUCTED SHORT NAME. The "~1" index is assigned
 * in directory-creation order, not derived from the long name:
 * "tokenizer.json" and "tokenizer_config.json" both mangle to
 * TOKENI~n.JSO and which gets ~1 depends on which was written first.
 * Building "TOKENI~1.JSO" and opening it would therefore load the wrong
 * file for one of the two orderings - a silently wrong answer, which is
 * worse here than an error.
 *
 * AMBIGUITY IS REFUSED for the same reason: when two entries both match
 * there is nothing in the directory that says which was meant, so
 * LZ_ERR_LFN_AMBIGUOUS reports the count and lets one be renamed.
 *
 * <dirent.h> covers all four build configurations - Watcom -bt=dos,
 * Watcom -bt=nt, MinGW and Linux gcc - with no per-target #ifdef. Watcom
 * ships it in the common include directory and sizes dirent.d_name from
 * NAME_MAX, which is 12 on a plain DOS build and 259 where long names
 * exist; 8.3 is exactly what readdir has to offer on the volume this
 * exists for.
 */

/* Resolve `want` inside `dir` and write the openable path to `out`.
 *
 * Tried in order, first hit wins:
 *   1. <dir>/<want> exactly, so the ordinary case costs one fopen and
 *      never reads the directory;
 *   2. an entry of `dir` equal to `want` ignoring case;
 *   3. an entry whose name is the 8.3 mangling of `want`, either plain
 *      truncation (config.json -> CONFIG.JSO) or numbered
 *      (tokenizer.json -> TOKENI~1.JSO).
 *
 * The separator written into `out` is '/', matching what every call site
 * in this engine builds by hand; src/model.c says why that must be
 * identical everywhere.
 *
 * Returns 0, or LZ_ERR_LFN_NOT_FOUND / LZ_ERR_LFN_AMBIGUOUS (`out`
 * untouched) / LZ_ERR_TRUNC / LZ_ERR_NULL_ARG, with errbuf filled as
 * everywhere else in this engine.
 *
 * `dir` may end with a separator or not; an empty `dir` resolves against
 * the current directory. */
int lz_lfn_path(const char *dir, const char *want,
                char *out, int cap, char *errbuf, int errlen);

/* Probe form, for callers that only ask whether the name is there -
 * lz_open's model.bin test, which decides between the bin and
 * safetensors loaders. Returns 1 or 0 and writes no errbuf: a probe's
 * negative answer is not an error. */
int lz_lfn_exists(const char *dir, const char *want);

#endif
