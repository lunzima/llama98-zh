#ifndef LZ_OPS_SSE_H
#define LZ_OPS_SSE_H
/* The SSE1-on-%xmm kernels: declarations for the part of
 * src/ops_mmx_sse.c that touches no MMX register at all.
 *
 * WHY A HEADER OF ITS OWN. src/ops_mmx_sse.c is a separate translation
 * unit from src/ops_mmx.c because gcc treats -msse as license to assume
 * CMOV for the WHOLE unit it is passed to, so the build separates the
 * two on ISA level - and the declarations did not. Everything here sat
 * in src/ops_mmx.h, which made a caller's include line say "MMX" while
 * the kernel it reached needs an SSE1 machine.
 *
 * WHAT DID NOT MOVE, and it is the distinction the split turns on.
 * There are two kinds of _sse kernel in this tree:
 *
 *   SSE1 on %mm    lz_p2_split32_sse, lz_q8round32_sse,
 *                  lz_p2_rows_sse, lz_q8round_group_sse and its 2p
 *                  sibling. Named for the ISA a machine must HAVE, but
 *                  the register file they work in is MMX's - measured,
 *                  not assumed: __m64 in the gcc bodies, an emms in the
 *                  group loops. They stay in src/ops_mmx.h, next to the
 *                  plain-MMX kernels they share a register file and an
 *                  emms discipline with.
 *
 *   SSE1 on %xmm   the five below. __m128 only on gcc, no `8087` in any
 *                  __modify list on Watcom.
 *
 * Moving all of the first kind out on the strength of the name would
 * have separated them from the emms rule that governs them.
 *
 * SSE2 does not need a second copy of any of these: mulps, addps and
 * maxps are SSE1, and what SSE2 adds is double and integer width that
 * none of these loops uses. src/ops_kernel_norm.h fills both tier slots
 * with one body for that reason and says so.
 */

#if defined(LZ_MMX_TU)

/* lz_matmul's F32-weight row. BOTH toolchains from one macro, and
   declared outside the __WATCOMC__ split below because the wrapper has
   the same name and prototype on each side. Returns how many elements
   it consumed (a multiple of 8) and leaves the C's EIGHT accumulators
   in acc8 for the caller to combine - see the body for why the combine
   does not belong in the kernel. */
#define LZ_MATMUL_F32_SSE_EXTERN 1
int lz_matmul_row_sse(const float *row, const float *x, int in_dim,
                      float *acc8);

/* lz_fwht's butterfly stage over FLOATS - the SSE1 sibling of the int32
   cell in src/ops_mmx.c. Mirrored ISA answers: this needs addps/subps
   (SSE1, and MMX cannot express it), that one needs paddd/psubd (MMX,
   and SSE1 cannot). len >= 4. */
#define LZ_FWHT_F32_SSE_EXTERN 1
void lz_fwht_stage_f32_sse(float *y, int n, int len);

/* Attention wsum's chunk fold - accf[d] += (float)acc32[d], 32 wide.
   SSE1 converts one at a time (cvtsi2ss); the packed convert at this
   tier is cvtpi2ps, whose source is an %mm register, and the call site
   has just paid an emms. So this cell belongs here and not with the
   %mm family - and it still replaces lz_i32f's six-operation x87 split
   with three instructions an element. src/ops_sse2.c holds the packed
   twin for machines that have SSE2. */
#define LZ_HAVE_I32FACC_SSE 1
void lz_i32f_acc32_sse(float *accf, const int32_t *acc32);

#if defined(__WATCOMC__)
/* RMSNorm's output loop and softmax's max scan and normalize loop. The
   _w wrappers sit between the #pragma aux bodies and the three dispatch
   tables in src/ops_kernel_norm.h, which reach them by token paste
   (LZ_NORM_SLOTS pastes _w onto the base name) - so a search for these
   spellings finds the definitions and no call site. */
void lz_rmsnorm_out_sse_w(float *o, const float *x, const float *w,
                          int n4, const float *k2);
void lz_vmax_sse_w(const float *x, int n4, float *pmax);
void lz_vscale_sse_w(float *x, int n4, const float *pk);
#endif /* __WATCOMC__ */

#endif /* LZ_MMX_TU */

#endif /* LZ_OPS_SSE_H */
