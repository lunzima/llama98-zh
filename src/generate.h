#ifndef LZ_GENERATE_H
#define LZ_GENERATE_H

/* Internal dynamic-temperature helpers, exported from generate.c so
   tests can pin the <think> tracker directly. The sampler-side
   temperature substitution has its own gate (test_sampler.c), but WHICH
   region the sampler is told it is in is decided here and must be gated
   too - a wrong region silently applies the wrong temperature with no
   error.

   NOT part of the public API: llama98.def does not export these, so they
   stay internal to the engine library while remaining linkable to tests
   that compile $(SRC) directly (the same arrangement apply_penalties_
   assumed / lz_target_dist have in sampler.h). */

/* Think-block tracking: whether generation is currently inside a <think>
   block. A tag can be split across BPE tokens, so the last
   LZ_THINK_PEND-1 bytes are held between feeds - the longest marker is
   </think>, 8 bytes. */
#define LZ_THINK_PEND 8

typedef struct {
    int  in_think;         /* read before each sample */
    char pend[LZ_THINK_PEND];
    int  n_pend;
} LZThinkTrack;

/* Feed decoded bytes to the think-block tracker; flips `in_think` on a
   complete <think>/</think> tag. */
void lz_think_track_feed(LZThinkTrack *tr, const char *bytes, int len);

#endif
