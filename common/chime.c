/* The finish chime, as bytes. A byte-for-byte dump of
 * sfx/output/success_arr.mid from parlour-midi, the repository the
 * music ships from - not edited here, because an edit here is a file
 * that no longer matches the one it came from.
 *
 * LICENCE. The music is CC BY 4.0, attributed to Lunzima; the rest of
 * this tree is Apache-2.0. These 139 bytes are the one piece of that
 * music inside this licence, and this comment is the credit CC BY asks
 * for. It is also why the music is not share-alike.
 *
 * The chime is not a file and is not shipped as one: at 139 bytes it
 * is cheaper to carry than to look for, and it is the one sound the
 * program owes the user at a moment they may not be watching. An
 * installation where music/ was never copied across still says the
 * reply has landed.
 *
 * Three tones from a Standard MIDI File like any other - E6, G6 from
 * 70 ms, C7 from 140, over at 440 - and it goes through the same
 * parser as every other piece, so nothing here is a second code path.
 */
#include "music.h"

const int lz_music_chime_len = 139;

const unsigned char lz_music_chime[139] = {
     77,  84, 104, 100,   0,   0,   0,   6,   0,   1,   0,   3,
      0, 192,  77,  84, 114, 107,   0,   0,   0,  37,   0, 255,
     81,   3,   7,  83,   0,   0, 255,   3,  13,  83, 117,  99,
     99, 101, 115, 115,  32,  99, 104, 105, 109, 101,   0, 255,
     88,   4,   4,   2,  24,   8, 131,  80, 255,  47,   0,  77,
     84, 114, 107,   0,   0,   0,  36,   0, 192,   0,  48, 144,
     88,  96,  27, 128,  88,   0,   1, 144,  91,  96,  27, 128,
     91,   0,   1, 144,  96,  96, 114, 128,  96,   0, 129, 118,
    176, 123,   0,   0, 255,  47,   0,  77,  84, 114, 107,   0,
      0,   0,  28,   0, 193,   0,  76, 145,  88,  72,  28, 145,
     91,  72,  21, 129,  88,   0,  36, 129,  91,   0, 130,  47,
    177, 123,   0,   0, 255,  47,   0
};
