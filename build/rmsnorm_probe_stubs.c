/* gcc-side kernel for build/rmsnorm_out_probe.c.
 *
 * The probe drives lz_rmsnorm_out_sse_w, and that name exists ONLY under
 * Watcom: src/ops_mmx_sse.c defines it as the addressable wrapper around
 * a #pragma aux body. A gcc build of the probe therefore had no kernel
 * to call, and the gcc intrinsics twin - lz_rmsnorm_out_sse in
 * src/ops_kernel_norm.h, which is what every gcc-built binary actually
 * runs - was never exercised by the probe that carries its name.
 *
 * This supplies the name from that twin, so each toolchain's build tests
 * its own half against the same scalar reference and the same fixed-seed
 * inputs. The two halves are then compared to each other through the
 * kernel hash the probe prints, which is a function of the kernel alone.
 *
 * NOT LINKED INTO THE WATCOM BUILD: there the real wrapper is in
 * ops_mmx_sse.c and a second definition would collide. The #if is on
 * __WATCOMC__ rather than on the build script so that adding this file
 * to the wrong link line fails loudly at the missing symbol rather than
 * quietly testing the wrong thing.
 */
#if !defined(__WATCOMC__)

#include "lz_int.h"
#include "ops.h"
/* BEFORE ops_kernel_norm.h, which builds LZ_NORMOUT_TAB[LZ_ROW_N] and
   calls LZ_DEFINE_PICK without including what defines either - the same
   ordering .prof/rowdot_xcheck.c documents for ops_matmul.h. */
#include "ops_kernel_shared.h"
#include "ops_kernel_norm.h"

void lz_rmsnorm_out_sse_w(float *o, const float *x, const float *w,
                          int n4, const float *k2);
/* Declared so the no-SSE arm is an UNDEFINED SYMBOL at link rather than
   an implicit declaration the compiler merely warns about. */
void lz_rmsnorm_out_sse_w_has_no_kernel_in_this_build(void);

void lz_rmsnorm_out_sse_w(float *o, const float *x, const float *w,
                          int n4, const float *k2) {
#if defined(LZ_HAVE_NORM_SSE) || defined(__SSE__) || defined(__SSE2__)
    lz_rmsnorm_out_sse(o, x, w, n4, k2);
#else
    /* No SSE in this build: there is no twin to test, and returning the
       scalar answer would make the probe compare the reference with
       itself and print a green that means nothing. */
    (void)o; (void)x; (void)w; (void)n4; (void)k2;
    lz_rmsnorm_out_sse_w_has_no_kernel_in_this_build();
#endif
}

#endif /* !__WATCOMC__ */
