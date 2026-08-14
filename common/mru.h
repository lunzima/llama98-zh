#ifndef LZ_GUI_MRU_H
#define LZ_GUI_MRU_H

/* The File menu's recent-model list.
 *
 * Pure: no Win32, no ini, no menu. Everything that goes wrong with an
 * MRU goes wrong in these three rules - newest first, no duplicates,
 * bounded - and all three are testable without a window.
 *
 * Four entries because that is what a Win9x File menu carried, and
 * because the block sits above the separator and pushes Exit down. */
#define LZ_MRU_MAX 4
#define LZ_MRU_LEN 512

typedef struct {
    char item[LZ_MRU_MAX][LZ_MRU_LEN];
    int  n;
} LZMru;

void lz_mru_init(LZMru *m);
/* Move to the front, or insert there; drops the oldest when full.
 * Comparison is case-insensitive: Windows paths are. */
void lz_mru_push(LZMru *m, const char *path);
/* Forget one - what a menu entry that no longer loads gets. */
void lz_mru_remove(LZMru *m, const char *path);

#endif
