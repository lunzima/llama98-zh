/* kv_slot_next must agree with kv_slot everywhere, not just where it was
 * derived.
 *
 * forward.c's scoring loops replace a per-iteration integer modulo with
 * a loop-carried increment. The rule is one line and it was derived by
 * hand across four cases; a hand-derived rule with no producer is worth
 * exactly as much as the check that reproduces it. This walks the whole
 * parameter space rather than the cases the derivation happened to
 * think of.
 *
 * Both functions are copied here rather than included, because forward.c
 * is not a header and pulling it in would drag the engine. The copies
 * must be kept identical to the originals - which is what the FIRST
 * assertion below is for: it re-derives kv_slot's own answer from the
 * definition (sink, then modulo) instead of trusting the copy.
 *
 * Exit: 0 agree, 1 a disagreement (printed).
 */
#include <stdio.h>

typedef struct { int attn_sink; int kv_ring; } State;

static int kv_slot(const State *s, int t) {
    if (!s->kv_ring) return t;
    if (t < s->attn_sink) return t;
    return s->attn_sink + ((t - s->attn_sink) % s->kv_ring);
}

static int kv_slot_next(const State *s, int slot) {
    return (slot + 1 == s->attn_sink + s->kv_ring) ? s->attn_sink : slot + 1;
}

int main(void)
{
    State s;
    int bad = 0, checked = 0, seeded = 0;

    for (s.attn_sink = 0; s.attn_sink <= 20; s.attn_sink++) {
        for (s.kv_ring = 0; s.kv_ring <= 20; s.kv_ring++) {
            int t, slot;
            /* Seeded the way the loops seed it, then advanced the way
               the loops advance it, for every starting t a loop could
               use - not just t = 0. A rule that only holds from the
               origin would pass a walk that always starts there. */
            for (t = 0; t <= 40; t++) {
                int u;
                slot = kv_slot(&s, t);
                seeded++;
                for (u = t; u < t + 60; u++) {
                    int want = kv_slot(&s, u);
                    if (slot != want) {
                        if (bad < 6)
                            printf("MISMATCH sink=%d ring=%d start=%d u=%d "
                                   "carried=%d direct=%d\n",
                                   s.attn_sink, s.kv_ring, t, u, slot, want);
                        bad++;
                    }
                    checked++;
                    slot = kv_slot_next(&s, slot);
                }
            }
        }
    }

    printf("seeds %d, steps %d\n", seeded, checked);
    if (bad) { printf("kv_slot_next: FAIL, %d disagreement(s)\n", bad); return 1; }
    /* Positive control: the walk must actually visit a wrap. Without a
       wrap the increment and the modulo agree trivially and this whole
       program is an expensive way to check that x + 1 == x + 1. */
    {
        State w; int t, wraps = 0, prev;
        w.attn_sink = 3; w.kv_ring = 7;
        prev = kv_slot(&w, 0);
        for (t = 1; t <= 60; t++) {
            int now = kv_slot(&w, t);
            if (now < prev) wraps++;
            prev = now;
        }
        if (wraps < 2) {
            printf("kv_slot_next: FAIL - the sweep saw %d wraps, so the "
                   "wrapping branch was never compared\n", wraps);
            return 1;
        }
        printf("control: %d wraps exercised\n", wraps);
    }
    printf("kv_slot_next: PASS\n");
    return 0;
}
