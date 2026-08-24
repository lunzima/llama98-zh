/* Does Open Watcom's mmintrin.h give the four MMX intrinsics epi's
 * kernel needs? If it does, the Watcom twin is the same body rather
 * than a second hand-written one.
 *
 * Build:
 *   INCLUDE='C:\WATCOM\h;C:\WATCOM\h\nt' wcc386 -6r -oxi -zq -bt=nt \
 *     .prof/watcom_mmx_intrin_probe.c -fo=/tmp/wmmx.obj
 */
#include <mmintrin.h>

__m64 probe_mmx(__m64 av, __m64 mv, __m64 mlo, __m64 mhi) {
    __m64 hi = _mm_or_si64(_mm_and_si64(_mm_srli_pi16(av, 15), mlo),
                           _mm_and_si64(av, mhi));
    return _mm_add_pi32(_mm_madd_pi16(hi, mv),
                        _mm_unpacklo_pi16(mv, mv));
}
