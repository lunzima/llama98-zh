#include <string.h>

#include "forward.h"
#include "ops.h"
#include "ops_quant.h"  /* lz_sigmoid_q15_run, lz_quantize_q8 */
#include "fwht.h"        /* lz_fwht, the block-diagonal Hadamard kernel */

/* Attn helpers, moved out of forward.c (the four-path split).
   Pure code motion: every body is letter-for-letter what forward.c held,
   only `static` dropped where the remaining forward.c paths still call it
   (the rest stay static and are declared extern in forward.h only when a
   caller outside this file needs them). */

/* One row of the Q8 weighted sum: dst += a * dequantised(vt). Split from
   forward_attn for the same reason score_q8_row was - the body ran twice,
   once per KV plane, and the plane is now an argument. */
void wsum_q8_row(float *dst, const int8_t *vt, const float *vts,
                 float a, int hd) {
    int d;
    for (d = 0; d < hd; d++) dst[d] += a * (float)vt[d] * vts[d >> 5];
}

/* One row of the Q8 score: q . dequantised(kt), grouped by 32.
 *
 * The scoring twin of wsum_q8_row, and the same four-parameter reason.
 * The high and low planes ran this letter for letter; the accumulation
 * ORDER is what has to survive, since a float sum reassociated is a
 * different number, and that is why the group loop comes along rather
 * than being left at the call sites. */
float score_q8_row(const float *qhh, const int8_t *kt,
                   const float *kts, int hd) {
    float sum = 0.0f;
    int d;
    for (d = 0; d < hd; d += 32) {
        float acc = 0.0f;
        int e;
        for (e = 0; e < 32; e++) acc += qhh[d + e] * (float)kt[d + e];
        sum += acc * kts[d >> 5];
    }
    return sum;
}

int kv_slot_next(const LZRunState *s, int slot) {
    return (slot + 1 == s->attn_sink + s->kv_ring) ? s->attn_sink : slot + 1;
}

int kv_slot(const LZRunState *s, int t) {
    if (!s->kv_ring) return t;
    if (t < s->attn_sink) return t;
    return s->attn_sink + ((t - s->attn_sink) % s->kv_ring);
}

/* Stored scale groups for a row of N elements at group width GS. A GS of
   0 is "one scale for the whole row", which stores none - callers that
   pass a width straight from lz_act_gs rely on that, since it returns 0
   for a width this path cannot honour. Written once because ten sites
   had it inline and a reader had to check each one for the guard. */
int scale_groups(int n, int gs) {
    return (gs > 0) ? n / gs : 0;
}

/* The slot after SLOT in a ring of DEPTH. SLOT is already reduced, so
   the successor is a compare; recomputing (base + tk + 1) % depth spends
   a second runtime modulo to reach the same answer. Distinct from
   kv_slot_next, which walks the KV ring and its attention sink. */
int ring_slot_next(int slot, int depth) {
    return (slot + 1 == depth) ? 0 : slot + 1;
}

/* One Q8 KV plane's weighted sum: dst = sum_t att[t] * dequant(v[t]).
 *
 * WRITTEN ONCE BECAUSE IT WAS WRITTEN TWICE. forward_attn had this body
 * for the high plane and again, under LZ_KV_2PLANE, for the low one -
 * the fixed fast path and the scalar walk both duplicated, differing
 * only in which buffer they write and which plane they read. The second
 * copy is compiled out of every shipping build, so it was a body nobody
 * built and nobody ran, kept in step with this one by hand. Now the
 * plane is an argument and the two callers reach the same code.
 *
 * noring IS SEPARATE FROM walk_t0 and the low plane is why. The fixed
 * kernel cannot express win_t0 - it walks from 0 - so the caller guards
 * it with LZ_ATTN_NORING. The low plane's scalar walk, though, starts at
 * a literal 0 rather than at win_t0, so its guard and its start were
 * already two different values in the original. Preserved rather than
 * unified: they may well be the same thing, but the path that would
 * prove it is not in any build that runs here.
 *
 * dead0/dead1 are parameters for the same reason the skip is spelled out
 * instead of using LZ_ATTN_SKIP_DEAD - that macro reads them from the
 * caller's scope and is #undef'd at the end of forward_attn, so it does
 * not exist out here. Same behaviour, including the debug counter. */
void attn_wsum_plane(LZRunState *s, float *dst,
                     const int8_t *vplane, const float *vs,
                     int hd, int kvd, int kvh, int pos,
                     int walk_t0, int noring, int dead0, int dead1) {
    int t, rslot;
#if (LZ_ATTN_FIXED & 2)
    if ((lz_attn_mode() & 2) && s->wsum_cbuf && noring &&
        hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
        lz_attn_wsum_q8(dst, s->att, hd,
                        vplane + (size_t)kvh * hd,
                        vs + (size_t)(kvh * hd) / 32,
                        kvd, pos, s->wsum_cbuf, s->wsum_cq,
                        s->attn_sink, s->kv_ring);
        return;
    }
#else
    (void)noring;
#endif /* LZ_ATTN_FIXED & 2 */
    memset(dst, 0, (size_t)hd * sizeof(float));
    LZ_KV_WALK(s, pos, t, rslot, walk_t0) {
        const int8_t *vt;
        const float *vts;
        float a;
        if (t >= dead0 && t < dead1) { lz_debug_attn_skip++; continue; }
        vt  = vplane + (size_t)rslot * kvd + (size_t)kvh * hd;
        vts = vs + (size_t)rslot * kvd / 32 + (size_t)(kvh * hd) / 32;
        a   = s->att[t];
        /* Convert, two multiplies and an add per element. */
        LZ_FCX(LZ_FC_ATTN_WSUM, 2 * hd, hd, 0, hd, 0);
        wsum_q8_row(dst, vt, vts, a, hd);
    }
}

/* Block-diagonal Hadamard over one head: hd/n independent n-wide
   transforms. n divides hd by construction (lz_state_alloc picks it). */
void rot_head(float *v, int hd, int n) {
    int o;
    for (o = 0; o < hd; o += n) lz_fwht(v + o, n);
    lz_debug_n_kv_rot++;
}
static void attn_score_loop(LZRunState *s, const LZModelConfig *c,
                            int *attn_int,
                            const float *ks4, const float *vs4,
                            const float *ks, const float *vs,
                            const int8_t *kc, const int8_t *vc,
                            const float *kf, const float *vf,
                            const unsigned char *k4, const unsigned char *v4,
#if LZ_KV_2PLANE
                            const int8_t *kc_lo, const int8_t *vc_lo,
#endif /* LZ_KV_2PLANE */
                            int layer, int pos0, int nt,
                            int qd, int qgd, int hd, int nh, int nkv,
                            int kv_mul, int kvd, int gso, float scale) {
    int h, d, t, tk;
    int kvh, kvc;                /* h / kv_mul and h % kv_mul, carried */
    float invn = 0.0f;           /* 1/kv_rot_v, hoisted: invariant across h and tk */

    if (s->kv_rot_v) invn = 1.0f / (float)s->kv_rot_v;

    for (tk = 0; tk < nt; tk++) {
        const float *qht = s->qh + (size_t)tk * qd;
        const float *qgt = s->qg + (size_t)tk * qgd;
        float *aot = s->attn_out + (size_t)tk * qd;
        int pos = pos0 + tk;
        /* win_t0: lz_debug_mtp_attn_window's own comment above - 0
           (normal) unless this is the MTP's own attention AND the
           investigative window override is active. */
        int win_t0 = 0;
        int dead0, dead1;            /* see LZ_ATTN_DEAD below */
        if (layer < 0 && lz_debug_mtp_attn_window > 0 &&
            pos - lz_debug_mtp_attn_window + 1 > win_t0)
            win_t0 = pos - lz_debug_mtp_attn_window + 1;
        /* lz_debug_mtp_attn_rows's own comment above - real ALU-cost
           proxy for the MTP's own attention (score loop runs once per
           head, (pos-win_t0+1) rows each), not just a byte count. Body
           layers (layer >= 0) never increment this. */
        if (layer < 0) lz_debug_mtp_attn_rows += (lz_i64)(pos - win_t0 + 1) * nh;

        /* Positions [dead0, dead1) fall outside sink+window. The mask
           block further down already sets their att[] to -1e30f, so
           SCORING them is dead work and their contribution to the value
           sum is exactly zero - exp(-1e30 - max) underflows to +0.0f and
           a*0 adds nothing. Skipping them is therefore bit-identical.

           This skip is the compute half of the eviction change. The
           memory half landed first (kunkun98-pilot at 2048 tokens:
           4.6 -> 3.0 MB at window 256); the loops kept walking the
           whole prefix, and the compute half measured nothing (2.173 vs
           2.123 ms/token of attention, window off vs on).

           dead1 stays 0 when no window is set, so LZ_ATTN_DEAD is false
           for every t on the default path and this costs one predictable
           compare per row. */
        {
            int d0 = win_t0 > s->attn_sink ? win_t0 : s->attn_sink;
            dead0 = d0;
            dead1 = 0;
            if (s->attn_win > 0) {
                dead1 = pos + 1 - s->attn_win;
                if (dead1 < s->attn_sink) dead1 = s->attn_sink;
            }
        }
#define LZ_ATTN_DEAD(t) ((t) >= dead0 && (t) < dead1)

/* Drop an evicted position and record that it was dropped. Seven walks
   below do this identically, and the counter is what keeps the drop
   visible in --debug rather than silent.
   Not wrapped in do/while(0): the continue has to reach the caller's
   loop, and a do/while would swallow it. So this expands to a bare if
   and the call sites carry no semicolon, same as LZ_KV_WALK. */
#define LZ_ATTN_SKIP_DEAD(t) \
    if (LZ_ATTN_DEAD(t)) { lz_debug_attn_skip++; continue; }
/* The kernels take kv_slot()'s two parameters and walk the ring
   themselves; what they still cannot express is win_t0, the
   investigative override that leaves att[0..win_t0-1] unscored. Rows the
   eviction MASK drops need no guard: the mask runs after scoring and
   overwrites whatever the kernel put there, and after softmax their
   weight is exactly zero. */
#define LZ_ATTN_NORING (win_t0 == 0)

        /* --attn-int: the int wsum keeps the exact int64 accumulator for
           lz_quantize_q8_int64 instead of the float dequant + re-quantize.
           Active only where the float chain would otherwise be the wsum's
           fixed tier: o_proj must quantize at gs==32 so each Q8 group is
           one 32-wide wsum group with ONE sscale; kv_rot_v==0 keeps the
           float un-rotate out (the int path has no Hadamard); v4/vf and
           the 2-plane low term are float-only consumers of `out`. */
#if (LZ_ATTN_FIXED & 2)
        /* wsum_cbuf and LZ_ATTN_MAX_HD only exist when the fixed wsum is
           compiled in, and the int path rides on it - so the predicate
           is compiled out with it rather than merely evaluating false. */
        /* !use_subn: the int tail leaves the result in attn_acc as
           per-group ints and never materializes attn_out, but SubLN has
           to run a per-ELEMENT weight multiply over that vector before
           o_proj reads it. Same reason the KDA block computes its own
           o_proj norm in float on every tier. */
        attn_int[tk] = lz_attn_int() && (lz_attn_mode() & 2) &&
                       s->wsum_cbuf && LZ_ATTN_NORING && !c->use_subn &&
                       hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD &&
                       s->kv_rot_v == 0 && gso == 32 && !v4 && !vf &&
                       !LZ_KV_2PLANE;
#else
        attn_int[tk] = 0;
#endif /* LZ_ATTN_FIXED & 2 */

        /* kvh is h / kv_mul, advanced rather than divided: h steps by one,
           so the quotient steps once every kv_mul heads. kv_mul is a
           runtime value, so the compiler cannot strength-reduce the
           division on its own, and it is one idiv per head per layer.
           The advance sits in the third clause, where a continue in the
           body cannot skip it. */
        kvh = 0;
        kvc = 0;
        for (h = 0; h < nh; h++,
             kvc = (kvc + 1 == kv_mul) ? (kvh++, 0) : kvc + 1) {
            const float *qhh = qht + (size_t)h * hd;
            float *out = aot + (size_t)h * hd;
            /* Cache row for the position being scored. Assigned once per
               iteration in each scoring loop below rather than called
               twice: kv_slot ends in an integer modulo by a runtime
               value, several of these loops used it for both the data
               pointer and the scale pointer, and the calls straddle the
               dot product so the compiler will not fold them. Declared
               out here because C89 wants declarations at the top of a
               block and these loop bodies already open with a statement.
               It depends only on t, never on h - hoisting it out of the
               HEAD loop as well would need a per-token table, which is
               a buffer this function does not have. */
            int rslot;

            if (k4) {
                /* score = sum_d q[d] * (scale_t * cents[code]) - the row
                   scale factors straight out of the sum, so it costs one
                   multiply per (row, head), not per coordinate. The
                   codebook is 64 bytes and stays in L1; that is what
                   makes a table lookup per coordinate affordable here
                   and would not be with a per-coordinate table. */
                LZ_KV_WALK(s, pos, t, rslot, win_t0) {
                    const unsigned char *kt;
                    float sum = 0.0f;
                    LZ_ATTN_SKIP_DEAD(t)
                    kt = k4 + (size_t)rslot * kvd / 2 +
                         (size_t)kvh * (hd >> 1);
                    for (d = 0; d < hd; d += 2) {
                        unsigned char pk = kt[d >> 1];
                        sum += qhh[d]     * lz_kv4_cents[pk & 0x0F];
                        sum += qhh[d + 1] * lz_kv4_cents[pk >> 4];
                    }
                    s->att[t] = sum * ks4[(size_t)rslot * nkv + kvh] * scale;
                }
            } else
            if (kf) {
                /* Reference scoring: no quantization anywhere on this
                   path, so any gap between this and the Q8 arm IS the
                   cache format's cost - which is the whole point of
                   having the arm. */
                LZ_KV_WALK(s, pos, t, rslot, win_t0) {
                    const float *kt;
                    float sum = 0.0f;
                    LZ_ATTN_SKIP_DEAD(t)
                    kt = kf + (size_t)rslot * kvd + (size_t)kvh * hd;
                    for (d = 0; d < hd; d++) sum += qhh[d] * kt[d];
                    s->att[t] = sum * scale;
                }
            } else
            {
#if (LZ_ATTN_FIXED & 1)
            /* The fixed path needs head_dim to be a multiple of 32 and
               within its static per-head buffers; anything else keeps the
               float loops rather than being silently truncated. The
               shape test is ANDed with the runtime tier, so a model the
               kernel cannot take falls back without the selector having
               to know about head_dim.
               LZ_ATTN_NORING is the third condition and the expensive one
               to have learned: both fixed kernels index the cache by
               ABSOLUTE t, so they read the wrong row the moment kv_slot()
               stops being the identity, and they score positions the
               eviction mask has dropped. */
            if ((lz_attn_mode() & 1) && LZ_ATTN_NORING &&
                hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                lz_attn_score_q8(s->att, qhh, hd,
                                 kc + (size_t)kvh * hd,
                                 ks + (size_t)(kvh * hd) / 32,
                                 kvd, pos, scale, s->attn_sink, s->kv_ring);
            } else
#endif /* LZ_ATTN_FIXED & 1 */
            LZ_KV_WALK(s, pos, t, rslot, win_t0) {
                /* scoring: q (f32) against k (Q8). float accumulation within
                   a group, one scale multiply at the group tail
                   (head_dim is a multiple of 32; saves a per-element
                   scale multiply). win_t0 is 0 (t starts at 0, as
                   always) unless the investigative window override
                   above is active. */
                const int8_t *kt;
                const float *kts;
                float sum = 0.0f;
                LZ_ATTN_SKIP_DEAD(t)
                kt  = kc + (size_t)rslot * kvd + (size_t)kvh * hd;
                kts = ks + (size_t)rslot * kvd / 32 +
                      (size_t)(kvh * hd) / 32;
                /* A convert, a multiply and an add per element; a
                   multiply and an add per group; one output scale. */
                LZ_FCX(LZ_FC_ATTN_SCORE, hd + (hd >> 5) + 1, hd + (hd >> 5), 0, hd, 0);
                sum = score_q8_row(qhh, kt, kts, hd);
                s->att[t] = sum * scale;
            }
#if LZ_KV_2PLANE
            /* Scoring is linear in k, so score(hi + lo/254) = score(hi)
               + score(lo)/254. This is a SECOND PASS over the same
               kernels - no new kernel and no second pair of assembly
               implementations to keep bit-identical.
               Folded BEFORE softmax: the low plane corrects the logit,
               not the probability. */
            {
                int tt;
                /* Declared HERE, not beside rslot above: every use is
                   inside this LZ_KV_2PLANE block, which is off by
                   default, so a declaration outside it is an unused
                   variable in every shipping build. rslot stays up
                   there because the k4 loop does use it. */
                int lslot;
#if (LZ_ATTN_FIXED & 1)
                if ((lz_attn_mode() & 1) && LZ_ATTN_NORING &&
                    hd > 0 && (hd % 32) == 0 && hd <= LZ_ATTN_MAX_HD) {
                    lz_attn_score_q8(s->att_lo, qhh, hd,
                                     kc_lo + (size_t)kvh * hd,
                                     ks + (size_t)(kvh * hd) / 32,
                                     kvd, pos, scale,
                                     s->attn_sink, s->kv_ring);
                } else
#endif /* LZ_ATTN_FIXED & 1 */
                LZ_KV_WALK(s, pos, tt, lslot, 0) {
                    const int8_t *kt = kc_lo + (size_t)lslot * kvd + (size_t)kvh * hd;
                    const float *kts = ks + (size_t)lslot * kvd / 32 +
                                       (size_t)(kvh * hd) / 32;
                    float sum = 0.0f;
                    /* Same shape as the scoring loop above: convert +
                       multiply + add per element, multiply + add per
                       group, one output scale. */
                    LZ_FCX(LZ_FC_ATTN_SCORE, hd + (hd >> 5) + 1, hd + (hd >> 5), 0, hd, 0);
                    sum = score_q8_row(qhh, kt, kts, hd);
                    s->att_lo[tt] = sum * scale;
                }
                for (tt = 0; tt <= pos; tt++)
                    s->att[tt] += s->att_lo[tt] * (1.0f / LZ_GDN_LO_SCALE);
            }
#endif /* LZ_KV_2PLANE */
            }
            /* win_t0>0 means s->att[0..win_t0-1] were never scored this
               call (investigative window override) - softmax only the
               slice actually filled, [win_t0..pos]. Ordinary case
               (win_t0==0) is unchanged: lz_softmax(s->att, pos+1). */
            /* StreamingLLM eviction, as a MASK rather than an eviction
               (--attn-sink / --attn-window). Keeping the first sink_n
               positions and the most recent window_w, and masking what
               falls between, reproduces the attention that a
               start+recent cache would compute - without touching the
               cache layout, the hot scoring loops, or any of the four
               format branches above.

               That ordering is deliberate. The memory saving is the
               point of the technique, but it is not the RISK: the risk
               is that a small hybrid model loses too much by forgetting
               the middle. Measure the quality with an instrument that
               cannot introduce bugs of its own, and only then pay for
               the eviction machinery.

               -1e30f rather than -INFINITY: the window always contains
               `pos`, so at least one entry survives, but a sentinel that
               cannot produce inf-minus-inf inside lz_softmax is worth
               more than the exactness of a true infinity. */
            if (s->attn_win > 0 || s->attn_sink > 0) {
                int lo = (s->attn_win > 0) ? pos + 1 - s->attn_win : 0;
                if (lo < s->attn_sink) lo = s->attn_sink;
                for (t = win_t0 > s->attn_sink ? win_t0 : s->attn_sink;
                     t < lo; t++)
                    s->att[t] = -1e30f;
            }
            lz_softmax(s->att + win_t0, pos + 1 - win_t0);

            if (v4) {
                memset(out, 0, (size_t)hd * sizeof(float));
                LZ_KV_WALK(s, pos, t, rslot, win_t0) {
                    const unsigned char *vt;
                    /* Row scale folded into the attention weight, same
                       trick as the scoring pass. */
                    float a;
                    LZ_ATTN_SKIP_DEAD(t)
                    vt = v4 + (size_t)rslot * kvd / 2 +
                         (size_t)kvh * (hd >> 1);
                    a = s->att[t] * vs4[(size_t)rslot * nkv + kvh];
                    for (d = 0; d < hd; d += 2) {
                        unsigned char pk = vt[d >> 1];
                        out[d]     += a * lz_kv4_cents[pk & 0x0F];
                        out[d + 1] += a * lz_kv4_cents[pk >> 4];
                    }
                }
            } else
            if (vf) {
                memset(out, 0, (size_t)hd * sizeof(float));
                LZ_KV_WALK(s, pos, t, rslot, win_t0) {
                    const float *vt;
                    float a;
                    LZ_ATTN_SKIP_DEAD(t)
                    vt = vf + (size_t)rslot * kvd + (size_t)kvh * hd;
                    a  = s->att[t];
                    LZ_FCX(LZ_FC_ATTN_WSUM, hd, hd, 0, 0, 0);   /* f32 V: no convert */
                    for (d = 0; d < hd; d++) out[d] += a * vt[d];
                }
            } else
            {
#if (LZ_ATTN_FIXED & 2)
            /* s->wsum_cbuf is in the condition, not asserted: an embedder
               may call lz_float_ref_select after the state exists, and the
               allocation above cannot see that. Reading the buffer the
               kernel needs is the only test that cannot drift from it. */
            if (attn_int[tk]) {
                lz_attn_wsum_q8_int(s->attn_acc + (size_t)tk * qd + (size_t)h * hd,
                                    s->attn_ss + (size_t)tk * (qd >> 5) + (size_t)h * (hd >> 5),
                                    s->att, hd,
                                    vc + (size_t)kvh * hd,
                                    vs + (size_t)(kvh * hd) / 32,
                                    kvd, pos, s->wsum_cbuf, s->wsum_cq,
                                    s->attn_sink, s->kv_ring);
            } else
#endif /* LZ_ATTN_FIXED & 2 */
            attn_wsum_plane(s, out, vc, vs, hd, kvd, kvh, pos,
                            win_t0, LZ_ATTN_NORING, dead0, dead1);
#if LZ_KV_2PLANE
            /* Same argument as the scoring pass: the weighted sum is
               linear in v. */
            {
                attn_wsum_plane(s, s->wsum_lo, vc_lo, vs, hd, kvd, kvh, pos,
                                0, LZ_ATTN_NORING, dead0, dead1);
                for (d = 0; d < hd; d++)
                    out[d] += s->wsum_lo[d] * (1.0f / LZ_GDN_LO_SCALE);
            }
#endif /* LZ_KV_2PLANE */
            }
            /* Rotate the attention output back. It is a linear combination
               of rotated V rows, so H applied once more undoes the
               rotation up to the factor kv_rot_v that lz_fwht carries
               (H H = n I). This has to happen BEFORE the output gate
               below: the gate is elementwise against qgt, which lives in
               the original basis. */
            if (s->kv_rot_v) {
                rot_head(out, hd, s->kv_rot_v);
                for (d = 0; d < hd; d++) out[d] *= invn;
            }
        }

        /* output gate: same source as q, take the second half of each head.
           The int path gates in the int domain - acc * Q15(sigmoid) >> 15
           keeps the same sscale. The Q15 sigmoid is the FLOAT path's own
           sigmoid (lz_sigmoid, whatever tier) rounded to 32768-scale, so
           the gate semantics match the float chain exactly; only the
           integer-domain accumulator and the >> 15 divide change.

           Fixed tier: lz_sigmoid(g) is (float)sigmoid_q15(g) * 2^-15, so
           `... * 32768.0f` below just undoes that scaling - both factors
           are exact powers of two (sigmoid_q15's range is [0, 32768], well
           inside float's 24-bit mantissa), so the round trip is exact and
           sigmoid_q15(g) can be called directly, dropping the cvt+2mul.
           Float tier keeps the original expression: lz_sigmoid there is
           exp-based, a different value from sigmoid_q15's table. */
        if (attn_int[tk]) {
            int sig_fixed = lz_sig_mode();
            for (h = 0; h < nh; h++) {
                const float *gate = qgt + (size_t)h * 2 * hd + hd;
                int64_t *ac = s->attn_acc + (size_t)tk * qd + (size_t)h * hd;
                if (sig_fixed) {
                    /* A RUN, not an element at a time. sigmoid_q15's
                       interpolation tail has a SIMD cell and a scalar
                       caller could never reach it - the kernel needs
                       more than one value to have anything to
                       vectorise. lz_sigmoid_q15_run is bit-identical
                       to the per-element call; see its own comment for
                       how the saturating exits are split out. */
                    int32_t s15[64];
                    int base;
                    for (base = 0; base < hd; base += 64) {
                        int m = hd - base;
                        if (m > 64) m = 64;
                        lz_sigmoid_q15_run(gate + base, s15, m);
                        for (d = 0; d < m; d++)
                            ac[base + d] =
                                (ac[base + d] * (int64_t)s15[d]) >> 15;
                    }
                } else {
                    for (d = 0; d < hd; d++) {
                        int32_t s15 =
                            (int32_t)(lz_sigmoid(gate[d]) * 32768.0f);
                        ac[d] = (ac[d] * (int64_t)s15) >> 15;
                    }
                }
            }
        } else {
            for (h = 0; h < nh; h++) {
                const float *gate = qgt + (size_t)h * 2 * hd + hd;
                float *out = aot + (size_t)h * hd;
                for (d = 0; d < hd; d++) out[d] *= lz_sigmoid(gate[d]);
            }
        }
    }
#undef LZ_ATTN_SKIP_DEAD
#undef LZ_ATTN_DEAD
}

/* Run nt tokens through this layer together. Batching reuses the
   WEIGHT LOADS: each output row's weights stream from memory once,
   serving nt tokens (arithmetic intensity 1 -> nt).

   Each token's own algorithm and ordering are untouched, so nt=1 is
   bit-identical to batch slice t - a hard gate, not a soft ask (see
   forward.h).

   `layer` is either a real index into m->layers[] or LZ_MTP_CACHE_LAYER
   (the MTP block, see above); everything else about this function is
   unchanged either way - the MTP block's q/k/v/o/q_norm/k_norm tensors
   are checked at load time (model.c's model_walk) to have exactly the
   body's attn_qgate_dim/attn_kv_dim/attn_q_dim/head_dim, so nothing here
   needs to special-case its shapes, only which cache slot it writes. */
void forward_attn(const LZModel *m, LZRunState *s,
                         const LZLayer *L, int layer, int pos0, int nt) {
    const LZModelConfig *c = &m->config;
    int hd = c->head_dim, nh = c->n_heads, nkv = c->n_kv_heads;
    int kv_mul = nh / nkv;
    float rms_eps = c->rms_norm_eps;
    int ci = (layer >= 0) ? s->cache_idx[layer] : c->n_full_layers;
    size_t loff = (size_t)ci * s->kv_slots * c->attn_kv_dim;
    size_t soff = loff / 32;
    int8_t *kc = s->kq8 ? s->kq8 + loff : NULL;
    int8_t *vc = s->vq8 ? s->vq8 + loff : NULL;
    /* Reference-arm slices; NULL unless --kv f32. Same (layer, pos,
       kv_dim) layout as kq8/vq8 so `loff` indexes both. */
    float *kf = s->kf32 ? s->kf32 + loff : NULL;
    float *vf = s->vf32 ? s->vf32 + loff : NULL;
    /* 4-bit slices: half the bytes, plus one scale per (pos, kv head). */
    unsigned char *k4 = s->k4 ? s->k4 + loff / 2 : NULL;
    unsigned char *v4 = s->v4 ? s->v4 + loff / 2 : NULL;
    float *ks4 = s->ks4 ? s->ks4 + (size_t)ci * s->kv_slots * c->n_kv_heads : NULL;
    float *vs4 = s->vs4 ? s->vs4 + (size_t)ci * s->kv_slots * c->n_kv_heads : NULL;
    float *ks = s->ksq ? s->ksq + soff : NULL;
    float *vs = s->vsq ? s->vsq + soff : NULL;
#if LZ_KV_2PLANE
    int8_t *kc_lo = s->kq8_lo ? s->kq8_lo + loff : NULL;
    int8_t *vc_lo = s->vq8_lo ? s->vq8_lo + loff : NULL;
#endif /* LZ_KV_2PLANE */
    /* (H q) . (H k) = kv_rot_k * (q . k) because lz_fwht is the
       unnormalized Hadamard (see its comment in ops.h). Fold that factor
       out here rather than normalizing the transform: kv_rot_k is a power
       of two, so this division is exact and identical on both compilers,
       whereas dividing by a float constant makes the two compilers
       disagree - one strength-reduces it to a reciprocal multiply. */
    float scale = s->attn_scale;
    int hdim = c->hidden_size, kvd = c->attn_kv_dim, qd = c->attn_q_dim;
    int qgd = c->attn_qgate_dim;
    int gsa = lz_act_gs(&L->q_proj, hdim);
    int nsa = scale_groups(hdim, gsa);
    int gso = lz_act_gs(&L->o_proj, qd);
    int nso = scale_groups(qd, gso);
    /* Per-token flag for the fixed-tier attention int path (--attn-int).
       The quantize step after the tk loop must know which tokens took the
       int wsum; only the MTP window override can split a batch. */
    int attn_int[LZ_BATCH_MAX];
    /* int-pipeline milestone 7. v_may is the capability, v_int is "the
       int buffer holds it" and is raised only by proj_i16 - two
       objects, for the reason that function's header gives. The SubLN
       branch below never calls it, so v_int stays 0 there and the KV
       write reads the float row it actually got.

       The consumer conditions are part of the capability, not of the
       matmul: vc is the plane lz_quantize_q8_i16 would write, and
       kv_rot_v / vf / v4 each read the float row element by element, so
       any of them means the exit would have to rebuild what it declined
       to build. Same set --attn-int refuses on.

       kv_rot_v STAYS IN THAT LIST, and the reason is now measured
       rather than assumed, because the obvious objection is right: the
       rotation DOES have an exact integer form (widen the int16 row to
       int32, lz_fwht_i32 in pure add/sub, out through
       lz_quantize_q8_int), so the exit does not strictly have to rebuild
       anything. Built and measured, kmr20, whole-sequence logit rms
       against --kv f32 on both gate corpora:

         --kv q8 --kv-rot on    zh          en
           float rotation     7.107e-02   7.161e-02   (this code)
           int rotation       7.296e-02   7.837e-02   +2.7% / +9.4%

       WHY IT LOSES, and it is not the transform: vtmp_i16 is at ONE
       row-wide exponent, so its error is uniform in ABSOLUTE terms,
       while the per-32 group scale downstream is not. The Hadamard
       conserves that error's energy and spreads it evenly, so a quiet
       group that would have been quantized against its own small scale
       inherits the row's average error instead. The float row has no
       such floor to spread. The exit and the rotation are antagonistic
       for that reason, not for want of an integer kernel.

       The refusal does cost the gate something, and it is paid there:
       on q8 the `off` arm takes this exit and the `on` arm does not, so
       build/kv_rot_gate.sh's two arms differ in the exit as well as in
       the rotation. That gate now prints this counter per arm and says
       so, rather than reading the sum as the rotation's own effect. */
    int v_int = 0, v_may = 0;
    int h, tk;

    if (s->kv_rot_k) scale /= (float)s->kv_rot_k;
#if LZ_VPROJ_I16
    v_may = s->vtmp_i16 && vc && vs && !vf && !v4 && s->kv_rot_v == 0 &&
            !LZ_KV_2PLANE && kvd >= 32 && (kvd % 32) == 0 &&
            lz_matmul_xq_i16_ok(&L->v_proj, hdim, nt);
#endif /* LZ_VPROJ_I16 */

    /* Quantize xb once; the q/k/v projections share the same xq/xqs
       (avoid rescanning activations).

       Under mixed precision q/k/v may have different formats and
       different weight gs, but sharing one quantization still holds -
       lz_act_gs only looks at whether in_dim is a multiple of 32, and
       all three share the same in_dim, so the answer is identical.
       Asking q_proj once suffices. */
    if (c->use_subn) {
        /* SubLN: the two pre-layer norms are gone, so q/k/v each
           normalize the raw residual with their OWN weight. That is
           exactly what the shared quantization above cannot express -
           three different inputs, so three norms, three quantizes,
           three matmuls. s->xb2 is free through this whole function.

           Plain norm + quantize on every tier: the norms tier's int
           shortcut (used by the KDA block) is an optimization, and the
           three inputs here differ whether or not it is on. */
        struct { const LZTensor *nrm; const LZTensor *w; float *out; int od; }
            p[3];
        int pi;
        p[0].nrm = &L->attn_q_subn_norm; p[0].w = &L->q_proj;
        p[0].out = s->qg;   p[0].od = qgd;
        p[1].nrm = &L->attn_k_subn_norm; p[1].w = &L->k_proj;
        p[1].out = s->ktmp; p[1].od = kvd;
        p[2].nrm = &L->attn_v_subn_norm; p[2].w = &L->v_proj;
        p[2].out = s->vtmp; p[2].od = kvd;
        for (pi = 0; pi < 3; pi++) {
            for (tk = 0; tk < nt; tk++)
                lz_rmsnorm(s->xb2 + (size_t)tk * hdim,
                           s->xb + (size_t)tk * hdim,
                           lz_t_f32(p[pi].nrm, s->wscr), hdim,
                           rms_eps);
            for (tk = 0; tk < nt; tk++)
                lz_quantize_q8(s->xb2 + (size_t)tk * hdim, hdim, gsa,
                               s->xq + (size_t)tk * hdim,
                               s->xqs + (size_t)tk * nsa);
            lz_matmul_xq_nt(p[pi].out, s->xb2, s->xq, s->xqs, p[pi].w,
                            hdim, p[pi].od, nt);
        }
    } else {
    for (tk = 0; tk < nt; tk++)
        lz_quantize_q8(s->xb + (size_t)tk * hdim, hdim, gsa,
                       s->xq + (size_t)tk * hdim, s->xqs + (size_t)tk * nsa);
    lz_matmul_xq_nt(s->qg, s->xb, s->xq, s->xqs, &L->q_proj, hdim, qgd, nt);
    lz_matmul_xq_nt(s->ktmp, s->xb, s->xq, s->xqs, &L->k_proj, hdim, kvd, nt);
    proj_i16(s, v_may, &v_int, s->vtmp, s->vtmp_i16,
                 s->xb, &L->v_proj, hdim, kvd, nt, LZ_VPROJ_ES, 32767);
    }

    /* QK-Norm + RoPE + write KV cache. **All cache rows of the batch
       must be written before scoring**: token t in the batch must see
       rows 0..t; merging the two loops would read rows not yet written. */
    for (tk = 0; tk < nt; tk++) {
        float *qgt = s->qg + (size_t)tk * qgd;
        float *qht = s->qh + (size_t)tk * qd;
        float *ktt = s->ktmp + (size_t)tk * kvd;
        float *vtt = s->vtmp + (size_t)tk * kvd;
        int pos = pos0 + tk;
        /* The cache row this token writes to. The writes below asked for
           it eighteen times with the same argument, and kv_slot ends in
           an integer modulo by a runtime value - which the compiler
           cannot always fold, since lz_gdn_quantize_2p and memcpy sit
           between the calls and it has no way to know they leave *s
           alone. It folds MOST of them: hoisting removes six idiv from
           this translation unit, not seventeen, so the other twelve were
           already gone. Six is still worth it on the in-order cores in
           the target family, where idiv r/m32 is around forty cycles and
           does not pipeline. */
        int wslot = kv_slot(s, pos);
        /* q_proj output is laid out PER HEAD as [q(hd), gate(hd)], not
           [all q][all gate]. The reference does view(..., n_heads, hd*2)
           then chunks along the last dim; treating it as two contiguous
           halves feeds every head misaligned data. */
        {
            const float *q_norm_w = lz_t_f32(&L->q_norm, s->wscr);
            for (h = 0; h < nh; h++)
                lz_rmsnorm(qht + (size_t)h * hd, qgt + (size_t)h * 2 * hd,
                           q_norm_w, hd, rms_eps);
        }
        {
            const float *k_norm_w = lz_t_f32(&L->k_norm, s->wscr);
            for (h = 0; h < nkv; h++)
                lz_rmsnorm(ktt + (size_t)h * hd, ktt + (size_t)h * hd,
                           k_norm_w, hd, rms_eps);
        }
        /* QK-Norm first, then RoPE, only then quantize into the cache -
           the cache holds the rotated k. head_dim is a multiple of 32,
           so quantization groups never cross heads. */
        lz_rope(qht, nh, hd, c->rotary_dim, pos, s->rope_cs);
        lz_rope(ktt, nkv, hd, c->rotary_dim, pos, s->rope_cs);
        /* Hadamard AFTER RoPE, on q and k alike - the cache holds rotated
           k, and a query rotated by the same H still scores against it
           (rotations preserve dot products). Doing it before RoPE would
           not: RoPE mixes coordinate i with i+rotary_dim/2, which does
           not commute with H. V is rotated too, and the attention output
           is rotated back below. */
        if (s->kv_rot_k) {
            for (h = 0; h < nh;  h++) rot_head(qht + (size_t)h * hd, hd, s->kv_rot_k);
            for (h = 0; h < nkv; h++) rot_head(ktt + (size_t)h * hd, hd, s->kv_rot_k);
        }
        if (s->kv_rot_v)
            for (h = 0; h < nkv; h++) rot_head(vtt + (size_t)h * hd, hd, s->kv_rot_v);
        /* Q8 writes, per side. Guarded on the pointer rather than on the
           format so a side whose plane was not allocated can never be
           written through - that is the shape of the segfault --kv q4
           already caused once, in lz_state_reset. */
#if LZ_KV_2PLANE
        /* hi and the scale come out byte-identical to lz_quantize_q8's -
           that is lz_gdn_quantize_2p's documented contract - so enabling
           the low plane cannot perturb the high one. fc_site -1: this is
           the KV cache write, not the recurrence write-back, and has no
           census site of its own yet. */
        if (kc) lz_gdn_quantize_2p(ktt, kvd, 32, kc + (size_t)wslot * kvd,
                                   kc_lo + (size_t)wslot * kvd,
                                   ks + (size_t)wslot * kvd / 32, -1);
        if (vc) lz_gdn_quantize_2p(vtt, kvd, 32, vc + (size_t)wslot * kvd,
                                   vc_lo + (size_t)wslot * kvd,
                                   vs + (size_t)wslot * kvd / 32, -1);
#else
        if (kc) lz_quantize_q8(ktt, kvd, 32, kc + (size_t)wslot * kvd,
                               ks + (size_t)wslot * kvd / 32);
        /* v_int says the projection left this token in vtmp_i16 at
           LZ_VPROJ_ES and vtt holds nothing - the int16 twin of the
           quantize below, folding 2^-ES into the stored group scale. */
        if (v_int) {
            lz_quantize_q8_i16(s->vtmp_i16 + (size_t)tk * kvd, kvd, 32,
                               pow2f(-LZ_VPROJ_ES),
                               vc + (size_t)wslot * kvd,
                               vs + (size_t)wslot * kvd / 32);
            lz_debug_vproj_i16 += kvd;
        } else if (vc) {
            lz_quantize_q8(vtt, kvd, 32, vc + (size_t)wslot * kvd,
                           vs + (size_t)wslot * kvd / 32);
        }
#endif /* LZ_KV_2PLANE */
        /* Reference arm: the same post-QK-norm, post-RoPE (and, when
           enabled, post-Hadamard) vectors the Q8 path just quantized,
           kept exactly. Written alongside rather than instead of the Q8
           planes so the two differ ONLY in the cache format. */
        if (kf) memcpy(kf + (size_t)wslot * kvd, ktt, (size_t)kvd * sizeof(float));
        if (vf) memcpy(vf + (size_t)wslot * kvd, vtt, (size_t)kvd * sizeof(float));
        /* Per HEAD, not per row: the norm that gets split off has to be
           the norm of the vector the attention actually dots against,
           and that is one head's worth. */
        if (v4)
            for (h = 0; h < nkv; h++)
                lz_kv4_quantize(vtt + (size_t)h * hd, hd,
                                v4 + (size_t)wslot * kvd / 2 + (size_t)h * (hd >> 1),
                                vs4 + (size_t)wslot * nkv + h);
        if (k4) {
            for (h = 0; h < nkv; h++) {
                {
                    unsigned char *kc4 = k4 + (size_t)wslot * kvd / 2 +
                                         (size_t)h * (hd >> 1);
                    float *ksc = ks4 + (size_t)wslot * nkv + h;
                    lz_kv4_quantize(ktt + (size_t)h * hd, hd, kc4, ksc);
                }
            }
        }
    }
    /* Post QK-Norm, post RoPE, pre-cache-quantization. These two are the
       last taps on the attention path that are still exact floats, so
       they are where a rotary_dim / head interleave / start_pos error has
       to show up before the KV cache's Q8 noise can hide it. */
    LZ_TAP("aq", layer, s->qh, qd);
    LZ_TAP("ak", layer, s->ktmp, kvd);

    attn_score_loop(s, c, attn_int, ks4, vs4, ks, vs, kc, vc, kf, vf, k4, v4,
#if LZ_KV_2PLANE
                    kc_lo, vc_lo,
#endif /* LZ_KV_2PLANE */
                    layer, pos0, nt, qd, qgd, hd, nh, nkv, kv_mul, kvd, gso,
                    scale);
    LZ_TAP("ax", layer, s->attn_out, qd);

    /* o_proj input is the attention output (attn_q_dim); quantize once more.
       The int path quantizes each 32-group from attn_acc with its own
       sscale: attn_int requires gso==32, so one Q8 group == one wsum group
       and the xq/xqs layout is identical to the float call below. */
    if (c->use_subn) {
        /* SubLN: o_proj's own input RMSNorm over the attention output.
           In place - attn_out is dead after this, and the tap above
           already recorded the pre-norm value as "ax". */
        for (tk = 0; tk < nt; tk++)
            lz_rmsnorm(s->attn_out + (size_t)tk * qd,
                       s->attn_out + (size_t)tk * qd,
                       lz_t_f32(&L->attn_o_subn_norm, s->wscr), qd,
                       rms_eps);
    }
    for (tk = 0; tk < nt; tk++) {
        if (attn_int[tk]) {
            int g;
            for (g = 0; g < (qd >> 5); g++)
                lz_quantize_q8_int64(s->attn_acc + (size_t)tk * qd + g * 32,
                                     32, 32,
                                     s->attn_ss[(size_t)tk * (qd >> 5) + g],
                                     s->xq + (size_t)tk * qd + g * 32,
                                     s->xqs + (size_t)tk * nso + g);
        } else {
            lz_quantize_q8(s->attn_out + (size_t)tk * qd, qd, gso,
                           s->xq + (size_t)tk * qd, s->xqs + (size_t)tk * nso);
        }
    }
    lz_matmul_xq_nt(s->xb2, s->attn_out, s->xq, s->xqs, &L->o_proj, qd,
                    hdim, nt);
}

