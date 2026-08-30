/* float transcendentals (lz_sinf/lz_cosf/lz_powf/lz_logf/lz_expf),
   ported verbatim from musl libm (powf/logf/expf) and glibc float
   sinf/cosf (sincosf_table), then converted to ALL-FLOAT internals:
   every double_t/double intermediate and every double table constant
   became float.  The engine's ARM build must not pull libgcc's
   double soft-float, and x86-64 (SSE) and ARM (soft-float) then run
   the same 24-bit float operation sequence -> bit-identical output.
   See lz_mathf.h for the accuracy contract. */

#include "lz_mathf.h"   /* pulls lz_int.h: the fixed-width types, and NOT
                           <stdint.h>, which VC++ 4.0 does not have */
#include <float.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* libm.h shim: bit helpers and FP-environment glue the musl bodies    */
/* reference. Verbatim from musl src/internal/libm.h, trimmed to the    */
/* parts the four functions below use.                                  */
/* ------------------------------------------------------------------ */

/* C89 has no `inline`. gcc spells the extension __inline__, Watcom
   spells it __inline, and both accept it in C89 mode - so the keyword is
   named once here rather than at each of the definitions below.
   (ops_arm_prim.h's __inline__ is not a precedent for portability: it
   sits inside #if __arm__ && __GNUC__, which Watcom never reaches.) */
#if defined(__GNUC__)
#define LZ_MATHF_INLINE __inline__
#elif defined(__WATCOMC__)
#define LZ_MATHF_INLINE __inline
#else
#define LZ_MATHF_INLINE
#endif /* __GNUC__ */

#define hidden
#define WANT_ROUNDING 1
#define WANT_SNAN 0
#define issignalingf_inline(x) 0
#define TOINT_INTRINSICS 0

#if defined(__WATCOMC__)
#define predict_true(x)  (x)
#define predict_false(x) (x)
#else
#define predict_true(x) __builtin_expect(!!(x), 1)
#define predict_false(x) __builtin_expect(x, 0)
#endif /* defined(__WATCOMC__) */

static LZ_MATHF_INLINE float eval_as_float(float x) { float y = x; return y; }
static LZ_MATHF_INLINE float fp_barrierf(float x) { volatile float y = x; return y; }
static LZ_MATHF_INLINE void fp_force_evalf(float x) { volatile float y; y = x; (void)y; }

/* union bit-punning, C89-safe (no compound literals - Watcom is C90). */
static LZ_MATHF_INLINE uint32_t asuint(float f) {
    union { float f; uint32_t u; } u; u.f = f; return u.u;
}
static LZ_MATHF_INLINE float asfloat(uint32_t u) {
    union { uint32_t u; float f; } uu; uu.u = u; return uu.f;
}
#define GET_FLOAT_WORD(w,d) do { (w) = asuint(d); } while(0)
#define SET_FLOAT_WORD(d,w) do { (d) = asfloat(w); } while(0)

/* C89 has no lz_inff() and cannot spell one as 1.0f/0.0f in a constant
   expression, so it comes from the bit pattern - the same 0x7F800000
   ops_quant.c compares against. */
static LZ_MATHF_INLINE float lz_inff(void) { return asfloat(0x7F800000u); }

/* error paths - the musl bodies call these on inf/nan/overflow inputs.
   They raise the FP exception and return the IEEE-mandated value. */
hidden float lz_math_uflowf(uint32_t sign) { return sign ? -0.0f : 0.0f; }
hidden float lz_math_oflowf(uint32_t sign) { return sign ? -lz_inff() : lz_inff(); }
hidden float lz_math_divzerof(uint32_t sign) { return sign ? -lz_inff() : lz_inff(); }
hidden float lz_math_invalidf(float x) { return x - x; }

/* ------------------------------------------------------------------ */
/* data tables (from musl powf_data.c / exp2f_data.c / logf_data.c)    */
/* ------------------------------------------------------------------ */
#define _LZ_MATHF_DATA_H


#define POWF_LOG2_TABLE_BITS 4
#define POWF_LOG2_POLY_ORDER 5
#define POWF_SCALE_BITS 0
#define POWF_SCALE ((float)(1 << POWF_SCALE_BITS))
/* static: these tables must not collide with the libm archive's own
   __powf_log2_data/__exp2f_data/__logf_data symbols (glibc's powf/logf
   members would then read OUR table layout, which for the float build
   has float fields where glibc expects double - garbage out).  */
static const struct powf_log2_data {
    struct {
        float invc, logc;
    } tab[1 << POWF_LOG2_TABLE_BITS];
    float poly[POWF_LOG2_POLY_ORDER];
} __powf_log2_data;


#define EXP2F_TABLE_BITS 5
#define EXP2F_POLY_ORDER 3
static const struct exp2f_data {
    uint32_t tab[1 << EXP2F_TABLE_BITS];
    float poly[EXP2F_POLY_ORDER];
    float invln2;   /* 1/ln2, unscaled: expf computes x*invln2 and hands it
                       to exp2_inline, which itself scales by N and does the
                       float toint (no double shift trick here). */
} __exp2f_data;


#define LOGF_TABLE_BITS 4
#define LOGF_POLY_ORDER 4
static const struct logf_data {
    struct {
        float invc, logc;
    } tab[1 << LOGF_TABLE_BITS];
    float ln2;
    float poly[LOGF_POLY_ORDER - 1];
} __logf_data;

/* --- powf_data.c --- */


static const struct powf_log2_data __powf_log2_data = {
  /* tab */ {
  { 1.39890718460083, -0.4843002259731293 * POWF_SCALE },
  { 1.3403141498565674, -0.42257124185562134 * POWF_SCALE },
  { 1.2864322662353516, -0.3633754253387451 * POWF_SCALE },
  { 1.2367150783538818, -0.30651310086250305 * POWF_SCALE },
  { 1.1906976699829102, -0.25180721282958984 * POWF_SCALE },
  { 1.147982120513916, -0.19910015165805817 * POWF_SCALE },
  { 1.1082251071929932, -0.148251011967659 * POWF_SCALE },
  { 1.0711297988891602, -0.09913323819637299 * POWF_SCALE },
  { 1.0364372730255127, -0.05163281410932541 * POWF_SCALE },
  { 1.0, 0.0 * POWF_SCALE },
  { 0.9492859840393066, 0.07508531957864761 * POWF_SCALE },
  { 0.8951049447059631, 0.15987126529216766 * POWF_SCALE },
  { 0.8476821780204773, 0.23840466141700745 * POWF_SCALE },
  { 0.8050314784049988, 0.3128829002380371 * POWF_SCALE },
  { 0.7664670944213867, 0.38370421528816223 * POWF_SCALE },
  { 0.7314286231994629, 0.45121103525161743 * POWF_SCALE },
  },
  /* poly */ {
  0.28845757246017456 * POWF_SCALE, -0.3609260618686676 * POWF_SCALE,
  0.4808984696865082 * POWF_SCALE, -0.721347451210022 * POWF_SCALE,
  1.4426950216293335 * POWF_SCALE,
  }
};

/* --- exp2f_data.c --- */



static const struct exp2f_data __exp2f_data = {
  /* tab[i] = float bit pattern of 2^(i/N), i in [0,N); value in [1,2)
     with exponent 127. exp2_inline adds (k>>5)+127 to the exponent
     field to obtain 2^(k/N) for an integer k. */
  /* tab */ {
0x3f800000, 0x3f82cd87, 0x3f85aac3, 0x3f88980f, 0x3f8b95c2, 0x3f8ea43a, 0x3f91c3d3, 0x3f94f4f0,
0x3f9837f0, 0x3f9b8d3a, 0x3f9ef532, 0x3fa27043, 0x3fa5fed7, 0x3fa9a15b, 0x3fad583f, 0x3fb123f6,
0x3fb504f3, 0x3fb8fbaf, 0x3fbd08a4, 0x3fc12c4d, 0x3fc5672a, 0x3fc9b9be, 0x3fce248c, 0x3fd2a81e,
0x3fd744fd, 0x3fdbfbb8, 0x3fe0ccdf, 0x3fe5b907, 0x3feac0c7, 0x3fefe4ba, 0x3ff5257d, 0x3ffa83b3,
  },
  /* poly */ {
  0.055503614246845245f, 0.24022845923900604f, 0.6931471824645996f,
  },
  /* invln2 */ 1.4426950216293335f,
};

/* --- logf_data.c --- */


static const struct logf_data __logf_data = {
  /* tab */ {
  { 1.39890718460083, -0.33569133281707764 },
  { 1.3403141498565674, -0.2929040491580963 },
  { 1.2864322662353516, -0.2518726587295532 },
  { 1.2367150783538818, -0.21245868504047394 },
  { 1.1906976699829102, -0.1745394468307495 },
  { 1.147982120513916, -0.13800570368766785 },
  { 1.1082251071929932, -0.10275976359844208 },
  { 1.0711297988891602, -0.06871392577886581 },
  { 1.0364372730255127, -0.035789139568805695 },
  { 1.0, 0.0 },
  { 0.9492859840393066, 0.052045177668333054 },
  { 0.8951049447059631, 0.11081431061029434 },
  { 0.8476821780204773, 0.1652495265007019 },
  { 0.8050314784049988, 0.2168738842010498 },
  { 0.7664670944213867, 0.26596349477767944 },
  { 0.7314286231994629, 0.31275567412376404 },
  },
  /* ln2 */ 0.6931471824645996,
  /* poly */ {
  -0.25089341402053833, 0.33345675468444824, -0.4999997615814209,
  }
};



/* exp2 table size: musl's EXP2F_TABLE_BITS=5 -> 32 entries. Used by the
   exp2_inline helper (in powf) and lz_expf. */
enum { LZ_EXP2F_N = 32 };

/* ------------------------------------------------------------------ */
/* function bodies (musl powf.c / logf.c / expf.c) */
/* ------------------------------------------------------------------ */

/*
POWF_LOG2_POLY_ORDER = 5
EXP2F_TABLE_BITS = 5

ULP error: 0.82 (~ 0.5 + relerr*2^24)
relerr: 1.27 * 2^-26 (Relative error ~= 128*Ln2*relerr_log2 + relerr_exp2)
relerr_log2: 1.83 * 2^-33 (Relative error of logx.)
relerr_exp2: 1.69 * 2^-34 (Relative error of exp2(ylogx).)
*/

#define N (1 << POWF_LOG2_TABLE_BITS)
#define T __powf_log2_data.tab
#define A __powf_log2_data.poly
#define OFF 0x3f330000

/* Subnormal input is normalized so ix has negative biased exponent.
   Output is multiplied by N (POWF_SCALE) if TOINT_INTRINICS is set.  */
static LZ_MATHF_INLINE float log2_inline(uint32_t ix)
{
	float z, r, r2, r4, p, q, y, yl0, invc, logc;
	uint32_t iz, top, tmp;
	int k, i;

	/* x = 2^k z; where z is in range [OFF,2*OFF] and exact.
	   The range is split into N subintervals.
	   The ith subinterval contains z and c is near its center.  */
	tmp = ix - OFF;
	i = (tmp >> (23 - POWF_LOG2_TABLE_BITS)) % N;
	top = tmp & 0xff800000;
	iz = ix - top;
	k = (int32_t)top >> (23 - POWF_SCALE_BITS); /* arithmetic shift */
	invc = T[i].invc;
	logc = T[i].logc;
	z = asfloat(iz);

	/* log2(x) = log1p(z/c-1)/ln2 + log2(c) + k */
	r = z * invc - 1;
	yl0 = logc + (float)k;

	/* Pipelined polynomial evaluation to approximate log1p(r)/ln2.  */
	r2 = r * r;
	y = A[0] * r + A[1];
	p = A[2] * r + A[3];
	r4 = r2 * r2;
	q = A[4] * r + yl0;
	q = p * r2 + q;
	y = y * r4 + q;
	return y;
}

#undef N
#undef T
#define T __exp2f_data.tab
#define SIGN_BIAS (1 << (EXP2F_TABLE_BITS + 11))

/* The output of log2 and thus the input of exp2 is either scaled by N
   (in case of fast toint intrinsics) or not.  The unscaled xd must be
   in [-1021,1023], sign_bias sets the sign of the result.  */
/* 2^xd for xd in (-128,128): xd = k/32 + r, k an integer, r in
   [-1/64,1/64].  The 2^23 magic does the float->int round without libm:
   at 2^23 the float ulp is 1, so (v + 2^23) - 2^23 rounds v to an
   integer (RNE).  No double, no 2^52 shift trick.  The unscaled xd must
   be in [-1021,1023]; sign_bias (pow of a negative base) is applied by
   the caller, not here. */
#define C __exp2f_data.poly
static LZ_MATHF_INLINE float exp2_inline(float xd)
{
	uint32_t t;
	float x32, frac, r, r2, y, z, s;
	int k, kexp, overflow2 = 0;

	/* k = round-to-nearest-even of xd*32, r = xd - k/32 in [-1/64,1/64].
	   Explicit RNE instead of a 2^23 magic: for negative xd the magic
	   lands below 2^23, where the float ulp is 1/2, so kd comes out a
	   half-integer and (int)kd truncates it wrongly.  musl's double
	   magic survives because its xd is tiny relative to 2^52; a float
	   2^23 is not large enough. */
	x32 = xd * 32.0f;
	k = (int)x32;
	frac = x32 - (float)k;
	if (frac > 0.5f || (frac == 0.5f && (k & 1)))
		k += 1;
	else if (frac < -0.5f || (frac == -0.5f && (k & 1)))
		k -= 1;
	r = xd - (float)k * 0.03125f;

	/* exp2(xd) = 2^(k/32) * 2^r ~= s * (C0*r^3 + C1*r^2 + C2*r + 1) */
	t = T[k & 31];
	kexp = k >> 5;
	if (kexp > 127) {
		/* k/32 = 128: float cannot hold 2^128, but any result that
		   reaches here has r = xd - 128 < 0 (the callers already
		   oflow on xd > 128), so 2^(128+r) is representable.  Build
		   2^127 * 2^r and scale by 2 at the end.  */
		kexp = 127;
		overflow2 = 1;
	}
	if (kexp >= -126) {
		uint32_t e = (uint32_t)(kexp + 127);   /* <= 254 after clamp */
		/* Replace the table value's exponent field: ADDING the
		   offset would carry into the mantissa and overflow to
		   NaN once 127+kexp hits 255.  */
		s = asfloat((t & 0x007fffff) | (e << 23));
	} else {
		/* 2^(k>>5) is subnormal: scale the [1,2) table value by the
		   bit pattern of the subnormal 2^(k>>5) (kexp >= -150 here;
		   below that the caller's uflow check has already fired). */
		int shift = kexp + 149;
		uint32_t m = shift > 0 ? 1u << shift : 1u;
		s = asfloat(t) * asfloat(m);
	}
	r2 = r * r;
	y = C[0] * r + C[1];
	z = C[2] * r + 1;
	z = y * r2 + z;
	y = z * s;
	if (overflow2) y *= 2.0f;
	return eval_as_float(y);
}
#undef C

/* Returns 0 if not int, 1 if odd int, 2 if even int.  The argument is
   the bit representation of a non-zero finite floating-point value.  */
static LZ_MATHF_INLINE int checkint(uint32_t iy)
{
	int e = iy >> 23 & 0xff;
	if (e < 0x7f)
		return 0;
	if (e > 0x7f + 23)
		return 2;
	if (iy & ((1 << (0x7f + 23 - e)) - 1))
		return 0;
	if (iy & (1 << (0x7f + 23 - e)))
		return 1;
	return 2;
}

static LZ_MATHF_INLINE int zeroinfnan(uint32_t ix)
{
	return 2 * ix - 1 >= 2u * 0x7f800000 - 1;
}


float lz_powf(float x, float y)
{
	uint32_t sign_bias = 0;
	uint32_t ix, iy;
	float logx, ylogx, x2;

	ix = asuint(x);
	iy = asuint(y);
	if (predict_false(ix - 0x00800000 >= 0x7f800000 - 0x00800000 ||
			  zeroinfnan(iy))) {
		/* Either (x < 1.1754943508222875e-38 or inf or nan) or (y is 0 or inf or nan).  */
		if (predict_false(zeroinfnan(iy))) {
			if (2 * iy == 0)
				return issignalingf_inline(x) ? x + y : 1.0f;
			if (ix == 0x3f800000)
				return issignalingf_inline(y) ? x + y : 1.0f;
			if (2 * ix > 2u * 0x7f800000 ||
			    2 * iy > 2u * 0x7f800000)
				return x + y;
			if (2 * ix == 2 * 0x3f800000)
				return 1.0f;
			if ((2 * ix < 2 * 0x3f800000) == !(iy & 0x80000000))
				return 0.0f; /* |x|<1 && y==inf or |x|>1 && y==-inf.  */
			return y * y;
		}
		if (predict_false(zeroinfnan(ix))) {
			x2 = x * x;
			if (ix & 0x80000000 && checkint(iy) == 1)
				x2 = -x2;
			/* Without the barrier some versions of clang hoist the 1/x2 and
			   thus division by zero exception can be signaled spuriously.  */
			return iy & 0x80000000 ? fp_barrierf(1 / x2) : x2;
		}
		/* x and y are non-zero finite.  */
		if (ix & 0x80000000) {
			/* Finite x < 0.  */
			int yint = checkint(iy);
			if (yint == 0)
				return lz_math_invalidf(x);
			if (yint == 1)
				sign_bias = SIGN_BIAS;
			ix &= 0x7fffffff;
		}
		if (ix < 0x00800000) {
			/* Normalize subnormal x so exponent becomes negative.  */
			ix = asuint(x * 8388608.0f);
			ix &= 0x7fffffff;
			ix -= 23 << 23;
		}
	}
	logx = log2_inline(ix);
	ylogx = y * logx; /* cannot overflow, y is single prec.  */
	if (predict_false(ylogx >= 126.0f || ylogx <= -126.0f)) {
		/* |y*log(x)| >= 126.  */
		if (ylogx > 128.0f)
			return lz_math_oflowf(sign_bias);
		if (ylogx <= -150.0f)
			return lz_math_uflowf(sign_bias);
	}
	return sign_bias ? -exp2_inline(ylogx) : exp2_inline(ylogx);
}

#undef N
#undef T
#undef A
#undef OFF
#undef C
#undef SHIFT

/*
LOGF_TABLE_BITS = 4
LOGF_POLY_ORDER = 4

ULP error: 0.818 (nearest rounding.)
Relative error: 1.957 * 2^-26 (before rounding.)
*/

#define T __logf_data.tab
#define A __logf_data.poly
#define Ln2 __logf_data.ln2
#define N (1 << LOGF_TABLE_BITS)
#define OFF 0x3f330000


float lz_logf(float x)
{
	float z, r, r2, y, yl0, invc, logc;
	uint32_t ix, iz, tmp;
	int k, i;

	ix = asuint(x);
	/* Fix sign of zero with downward rounding when x==1.  */
	if (WANT_ROUNDING && predict_false(ix == 0x3f800000))
		return 0;
	if (predict_false(ix - 0x00800000 >= 0x7f800000 - 0x00800000)) {
		/* x < 1.1754943508222875e-38 or inf or nan.  */
		if (ix * 2 == 0)
			return lz_math_divzerof(1);
		if (ix == 0x7f800000) /* log(inf) == inf.  */
			return x;
		if ((ix & 0x80000000) || ix * 2 >= 0xff000000)
			return lz_math_invalidf(x);
		/* x is subnormal, normalize it.  */
		ix = asuint(x * 8388608.0f);
		ix -= 23 << 23;
	}

	/* x = 2^k z; where z is in range [OFF,2*OFF] and exact.
	   The range is split into N subintervals.
	   The ith subinterval contains z and c is near its center.  */
	tmp = ix - OFF;
	i = (tmp >> (23 - LOGF_TABLE_BITS)) % N;
	k = (int32_t)tmp >> 23; /* arithmetic shift */
	iz = ix - (tmp & 0xff800000);
	invc = T[i].invc;
	logc = T[i].logc;
	z = asfloat(iz);

	/* log(x) = log1p(z/c-1) + log(c) + k*Ln2 */
	r = z * invc - 1;
	yl0 = logc + (float)k * Ln2;

	/* Pipelined polynomial evaluation to approximate log1p(r).  */
	r2 = r * r;
	y = A[1] * r + A[2];
	y = A[0] * r2 + y;
	y = y * r2 + (yl0 + r);
	return eval_as_float(y);
}

#undef T
#undef A
#undef Ln2
#undef N
#undef OFF

/*
EXP2F_TABLE_BITS = 5
EXP2F_POLY_ORDER = 3

ULP error: 0.502 (nearest rounding.)
Relative error: 1.69 * 2^-34 in [-ln2/64, ln2/64] (before rounding.)
Wrong count: 170635 (all nearest rounding wrong results with fma.)
Non-nearest ULP error: 1 (rounded ULP error)
*/

#define InvLn2 __exp2f_data.invln2
#define T __exp2f_data.tab

static LZ_MATHF_INLINE uint32_t top12(float x)
{
	return asuint(x) >> 20;
}


float lz_expf(float x)
{
	uint32_t abstop;

	abstop = top12(x) & 0x7ff;
	if (predict_false(abstop >= top12(88.0f))) {
		/* |x| >= 88 or x is nan.  */
		if (asuint(x) == asuint(-lz_inff()))
			return 0.0f;
		if (abstop >= top12(lz_inff()))
			return x + x;
		if (x > 88.72283172607422f) /* x > log(inf) ~= 88.72 */
			return lz_math_oflowf(0);
		if (x < -103.97207641601562f) /* x < log(0.0) ~= -103.97 */
			return lz_math_uflowf(0);
	}

	/* exp(x) = 2^(x*log2(e)); exp2_inline does the k/32 split and the
	   float toint itself.  */
	return exp2_inline(x * InvLn2);
}

#undef N
#undef T
#undef InvLn2

/* ------------------------------------------------------------------ */
/* sin/cos (glibc 2.39 s_sinf.c / s_cosf.c + s_sincosf.h +            */
/*          sincosf_poly.h + s_sincosf_data.c, verbatim)              */
/* ------------------------------------------------------------------ */

/* float sin/cos, ported verbatim from glibc's float sinf/cosf
   (__sincosf_table + reduce_fast/reduce_large, worst-case 0.5607 ulp).
   This is the implementation arm-linux-gnueabi's libm actually runs
   (glibc >= 2.28), so lz_sinf/lz_cosf are bit-identical to the target
   libm -- verified 0/5.4M bit-mismatch under qemu-arm. TOINT_INTRINSICS=0
   (ARMv5 soft-float has no round-to-int instruction), so hpi_inv carries
   the 2^24 prescale. Data path is float-only, all intermediates float. */

typedef struct
{
  float sign[4];		/* Sign of sine in quadrants 0..3.  */
  float hpi_inv;		/* 2 / PI * 2^24 (TOINT_INTRINSICS=0).  */
  float hpi;			/* PI / 2.  */
  float c0, c1, c2, c3, c4;	/* Cosine polynomial.  */
  float s1, s2, s3;		/* Sine polynomial.  */
} sincos_t;

static const sincos_t lz_sincosf_table[2] =
{
  {
    { 1.0f, -1.0f, -1.0f, 1.0f },
    10680707.0f,
    1.5707963705062866f,
    1.0f,
    -0.5f,
    0.04166662320494652f,
    -0.0013886763481423259f,
    2.4390450562350452e-05f,
    -0.16666655242443085f,
    0.008332177996635437f,
    -0.0001951729936990887f
  },
  {
    { 1.0f, -1.0f, -1.0f, 1.0f },
    10680707.0f,
    1.5707963705062866f,
    -1.0f,
    0.5f,
    -0.04166662320494652f,
    0.0013886763481423259f,
    -2.4390450562350452e-05f,
    -0.16666655242443085f,
    0.008332177996635437f,
    -0.0001951729936990887f
  }
};

/* Table with 4/PI to 192 bit precision.  */
static const uint32_t lz_inv_pio4[24] =
{
  0xa2,       0xa2f9,      0xa2f983,   0xa2f9836e,
  0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529,
  0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1,
  0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0,
  0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599,
  0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041
};

static const float pi63 = 3.4061216748705226e-19f;
static const float pio4 = 0.7853981852531433f;

/* Top 12 bits of the float representation with the sign bit cleared.  */
static LZ_MATHF_INLINE uint32_t
abstop12 (float x)
{
  return (asuint (x) >> 20) & 0x7ff;
}

/* Fast range reduction using single multiply-subtract.  Return the modulo of
   X as a value between -PI/4 and PI/4 and store the quadrant in NP.
   Accurate for |X| <= 120.0.  */
static LZ_MATHF_INLINE float
reduce_fast (float x, const sincos_t *p, int *np)
{
  float r = x * p->hpi_inv;
  int n = ((int32_t)r + 0x800000) >> 24;
  *np = n;
  return x - (float)n * p->hpi;
}

/* Reduce the range of XI (reinterpreted float, |X| >= 2) using fast integer
   arithmetic and a 192-bit 4/PI table.  A 32x96->128 bit multiply computes
   the exact 2.62-bit fixed-point modulo.  */
static LZ_MATHF_INLINE float
reduce_large (uint32_t xi, int *np)
{
  const uint32_t *arr = &lz_inv_pio4[(xi >> 26) & 15];
  int shift = (xi >> 23) & 7;
  uint64_t n, res0, res1, res2;
  float x;

  xi = (xi & 0xffffff) | 0x800000;
  xi <<= shift;

  res0 = xi * arr[0];
  res1 = (uint64_t)xi * arr[4];
  res2 = (uint64_t)xi * arr[8];
  res0 = (res2 >> 32) | (res0 << 32);
  res0 += res1;

  n = (res0 + ((uint64_t)1 << 61)) >> 62;
  res0 -= n << 62;
  /* THROUGH THE BARRIER, and PC=24 is not a substitute for it.  A 387
     loads this 62-bit integer with `fild`, which is exact and which the
     precision control word does not touch - so without the barrier the
     x87 build multiplies the EXACT integer by pi63 and rounds once,
     while SSE and the ARM soft-float round the integer to float first
     and then round the product: one rounding against two.  The results
     differ by 1 ulp on roughly 0.6% of arguments.  That is not
     academic - forward.c builds the RoPE table from lz_sinf/lz_cosf at
     every position, so the first argument where the two disagree makes
     the whole float tier (--fixed off) diverge between gcc and Watcom
     from that token onward; on kmr20 it is position 125, sin(125).
     Same class as lz_i32f in ops_kernel_shared.h, which forces the
     int32 rounding before a multiply for exactly this reason. */
  x = fp_barrierf ((float)(int64_t)res0);
  *np = (int)n;
  return x * pi63;
}

/* Polynomial for sine or cosine of reduced X, selected by quadrant N.  */
static LZ_MATHF_INLINE float
sinf_poly (float x, float x2, const sincos_t *p, int n)
{
  float x3, x4, x6, x7, s, c, c1, c2, s1;

  if ((n & 1) == 0)
    {
      x3 = x * x2;
      s1 = p->s2 + x2 * p->s3;

      x7 = x3 * x2;
      s = x + x3 * p->s1;

      return s + x7 * s1;
    }
  else
    {
      x4 = x2 * x2;
      c2 = p->c3 + x2 * p->c4;
      c1 = p->c0 + x2 * p->c1;

      x6 = x4 * x2;
      c = c1 + x4 * p->c2;

      return c + x6 * c2;
    }
}

float
lz_sinf (float y)
{
  float x = y;
  float s;
  int n;
  const sincos_t *p = &lz_sincosf_table[0];

  if (abstop12 (y) < abstop12 (pio4))
    {
      s = x * x;

      if (predict_false (abstop12 (y) < abstop12 (0.000244140625f)))
	{
	  /* Force underflow for tiny y.  */
	  if (predict_false (abstop12 (y) < abstop12 (1.1754943508222875e-38f)))
	    fp_force_evalf (s);
	  return y;
	}

      return sinf_poly (x, s, p, 0);
    }
  else if (predict_true (abstop12 (y) < abstop12 (120.0f)))
    {
      x = reduce_fast (x, p, &n);

      s = p->sign[n & 3];

      if (n & 2)
	p = &lz_sincosf_table[1];

      return sinf_poly (x * s, x * x, p, n);
    }
  else if (abstop12 (y) < abstop12 (lz_inff()))
    {
      uint32_t xi = asuint (y);
      int sign = xi >> 31;

      x = reduce_large (xi, &n);

      s = p->sign[(n + sign) & 3];

      if ((n + sign) & 2)
	p = &lz_sincosf_table[1];

      return sinf_poly (x * s, x * x, p, n);
    }
  else
    return lz_math_invalidf (y);
}

float
lz_cosf (float y)
{
  float x = y;
  float s;
  float x2;
  int n;
  const sincos_t *p = &lz_sincosf_table[0];

  if (abstop12 (y) < abstop12 (pio4))
    {
      x2 = x * x;

      if (predict_false (abstop12 (y) < abstop12 (0.000244140625f)))
	return 1.0f;

      return sinf_poly (x, x2, p, 1);
    }
  else if (predict_true (abstop12 (y) < abstop12 (120.0f)))
    {
      x = reduce_fast (x, p, &n);

      s = p->sign[n & 3];

      if (n & 2)
	p = &lz_sincosf_table[1];

      return sinf_poly (x * s, x * x, p, n ^ 1);
    }
  else if (abstop12 (y) < abstop12 (lz_inff()))
    {
      uint32_t xi = asuint (y);
      int sign = xi >> 31;

      x = reduce_large (xi, &n);

      s = p->sign[(n + sign) & 3];

      if ((n + sign) & 2)
	p = &lz_sincosf_table[1];

      return sinf_poly (x * s, x * x, p, n ^ 1);
    }
  else
    return lz_math_invalidf (y);
}
