#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

#define POINTS_MAX         2000             /* max number of data points in one lc. */
#define MAX_N_OBS         20000             /* max number of data points */
#define MAX_LC              200             /* max number of lightcurves */
#define MAX_LINE_LENGTH    1000             /* max length of line in an input file */
#define MAX_N_FPOINTS    500000             /* max number of frequency points */
#define MAX_N_FAC          1000             /* max number of facets */
#define MAX_N_ITER          100             /* maximum number of iterations */
#define MAX_N_PAR           200             /* maximum number of parameters */
#define MAX_LM               10             /* maximum degree and order of sph. harm. */
#define N_PHOT_PAR            5             /* maximum number of parameters in scattering  law */
#define TINY                  1e-8          /* precision parameter for mu, mu0*/
#define N_POLES              10             /* number of initial poles */

/* dytemp is stored transposed - dytemp[(jp-1)*DYT_STRIDE + l], l = 1..ma - so
   consecutive work-items reading consecutive parameters hit consecutive
   addresses. Requires ma <= DYT_STRIDE-1 (spherical-harmonics degree <= 6,
   i.e. every production workunit); enforced on the host. */
#define DYT_STRIDE           64

/* normal-equation accumulation tile: points per rank-K update in
   mrqcof_curve2 */
#define CURVE2_K             8

#define PI                 M_PI             /* 3.14159265358979323846 */
#define AU            149597870.691         /* Astronomical Unit [km] */
#define C_SPEED       299792458             /* speed of light [m/s]*/

#define DEG2RAD      (PI / 180)
#define RAD2DEG      (180 / PI)

#define BLOCK_DIM 128
/* SoftFP64.cl - IEEE-754 binary64 arithmetic in software, for devices
   without cl_khr_fp64 (the PS_FP32 build, see Real.cl).

   A value is the double's own 64-bit pattern in a ulong; every operation
   is computed with integer arithmetic and rounded to nearest-even exactly
   as the hardware does, with full subnormal support: add, sub, mul, fma,
   div and sqrt are correctly rounded. On top of them sit ports of the AMD
   device-library (ocml) routines the kernels call - exp, log, acos,
   sincos, fmod - reproduced operation by operation from the library, so
   the PS_FP32 build returns bit-for-bit what the FP64 build computes on
   AMD hardware.

   The file is also compiled as C++ by the host-side test harness
   (SF_HOST), hence the few portability macros. */

#ifdef PS_FP32

#ifdef SF_HOST
#define SF_CONST static const
typedef long long sf_i64;
#define sf_clz64(x) ((x) ? __builtin_clzll(x) : 64)
#define sf_mulhi(a, b) ((ulong)(((unsigned __int128)(a) * (b)) >> 64))
#define sf_as_uint(f) sf_host_as_uint(f)
#else
#define SF_CONST __constant
typedef long sf_i64;
#define sf_clz64(x) ((int)clz((ulong)(x)))
#define sf_mulhi(a, b) mul_hi((ulong)(a), (ulong)(b))
#define sf_as_uint(f) as_uint(f)
#endif

#define SF_SIGN     0x8000000000000000UL
#define SF_INF      0x7FF0000000000000UL
#define SF_HIDDEN   0x0010000000000000UL
#define SF_FRAC     0x000FFFFFFFFFFFFFUL
#define SF_QUIET    0x0008000000000000UL
#define SF_NAN      0x7FF8000000000000UL
#define SF_ONE      0x3FF0000000000000UL

#define sf_sign(a)  ((uint)((a) >> 63))
#define sf_expf(a)  ((int)(((a) >> 52) & 0x7FF))
#define sf_frac(a)  ((a) & SF_FRAC)
#define sf_pack(s, e, sig) (((ulong)(s) << 63) + ((ulong)(e) << 52) + (sig))
#define sf_isnan(a) (((a) & ~SF_SIGN) > SF_INF)
#define sf_iszero(a) (((a) << 1) == 0)
#define sf_neg(a)   ((a) ^ SF_SIGN)
#define sf_abs(a)   ((a) & ~SF_SIGN)

/* ================================================================== */
/* rounding core (Berkeley SoftFloat 3 conventions)                    */
/* ================================================================== */

ulong sf_prop_nan(ulong a, ulong b)
{
	return (sf_isnan(a) ? a : b) | SF_QUIET;
}

ulong sf_shr_jam(ulong a, uint dist)
{
	if (dist == 0)
		return a;
	if (dist < 64)
		return (a >> dist) | (ulong)((a << (64 - dist)) != 0);
	return (ulong)(a != 0);
}

/* sig: leading one at bit 62, value = sig * 2^(exp - 1084) */
ulong sf_round_pack(uint sign, int exp, ulong sig)
{
	uint roundBits = (uint)(sig & 0x3FF);
	if ((uint)exp >= 0x7FDu)
	{
		if (exp < 0)
		{
			sig = sf_shr_jam(sig, (uint)(-exp));
			exp = 0;
			roundBits = (uint)(sig & 0x3FF);
		}
		else if (exp > 0x7FD || sig + 0x200 >= SF_SIGN)
		{
			return sf_pack(sign, 0x7FF, 0);
		}
	}
	sig = (sig + 0x200) >> 10;
	if (roundBits == 0x200)
		sig &= ~(ulong)1;
	if (!sig)
		exp = 0;
	return sf_pack(sign, exp, sig);
}

ulong sf_norm_round_pack(uint sign, int exp, ulong sig)
{
	int shift = sf_clz64(sig) - 1;
	exp -= shift;
	if (shift >= 10 && (uint)exp < 0x7FDu)
		return sf_pack(sign, sig ? exp : 0, sig << (shift - 10));
	return sf_round_pack(sign, exp, sig << shift);
}

/* subnormal significand -> normalized (hidden bit at 52), biased exponent */
ulong sf_norm_sub(ulong sig, int* exp)
{
	int shift = sf_clz64(sig) - 11;
	*exp = 1 - shift;
	return sig << shift;
}

/* 128-bit value (hi:lo) * 2^e, nonzero: round to double */
ulong sf_round128(uint sign, int e, ulong hi, ulong lo)
{
	int k;
	if (hi)
		k = sf_clz64(hi) - 1;
	else
		k = 63 + sf_clz64(lo);
	if (k >= 64)
	{
		hi = lo << (k - 64);
		lo = 0;
	}
	else if (k > 0)
	{
		hi = (hi << k) | (lo >> (64 - k));
		lo <<= k;
	}
	return sf_round_pack(sign, e - k + 64 + 1084, hi | (ulong)(lo != 0));
}

/* ================================================================== */
/* add / sub                                                            */
/* ================================================================== */

ulong sf_add_mags(ulong a, ulong b, uint signZ)
{
	int expA = sf_expf(a), expB = sf_expf(b);
	ulong sigA = sf_frac(a), sigB = sf_frac(b);
	int expDiff = expA - expB;
	int expZ;
	ulong sigZ;
	if (!expDiff)
	{
		if (!expA)
			return a + sigB;
		if (expA == 0x7FF)
		{
			if (sigA | sigB)
				return sf_prop_nan(a, b);
			return a;
		}
		expZ = expA;
		sigZ = (0x0020000000000000UL + sigA + sigB) << 9;
	}
	else
	{
		sigA <<= 9;
		sigB <<= 9;
		if (expDiff < 0)
		{
			if (expB == 0x7FF)
			{
				if (sigB)
					return sf_prop_nan(a, b);
				return sf_pack(signZ, 0x7FF, 0);
			}
			expZ = expB;
			if (expA)
				sigA += 0x2000000000000000UL;
			else
				sigA <<= 1;
			sigA = sf_shr_jam(sigA, (uint)(-expDiff));
		}
		else
		{
			if (expA == 0x7FF)
			{
				if (sigA)
					return sf_prop_nan(a, b);
				return a;
			}
			expZ = expA;
			if (expB)
				sigB += 0x2000000000000000UL;
			else
				sigB <<= 1;
			sigB = sf_shr_jam(sigB, (uint)expDiff);
		}
		sigZ = 0x2000000000000000UL + sigA + sigB;
		if (sigZ < 0x4000000000000000UL)
		{
			--expZ;
			sigZ <<= 1;
		}
	}
	return sf_round_pack(signZ, expZ, sigZ);
}

ulong sf_sub_mags(ulong a, ulong b, uint signZ)
{
	int expA = sf_expf(a), expB = sf_expf(b);
	ulong sigA = sf_frac(a), sigB = sf_frac(b);
	int expDiff = expA - expB;
	if (!expDiff)
	{
		if (expA == 0x7FF)
		{
			if (sigA | sigB)
				return sf_prop_nan(a, b);
			return SF_NAN;
		}
		sf_i64 sigDiff = (sf_i64)sigA - (sf_i64)sigB;
		if (!sigDiff)
			return 0;	/* +0 under round-to-nearest */
		if (expA)
			--expA;
		if (sigDiff < 0)
		{
			signZ ^= 1;
			sigDiff = -sigDiff;
		}
		int shift = sf_clz64((ulong)sigDiff) - 11;
		int expZ = expA - shift;
		if (expZ < 0)
		{
			shift = expA;
			expZ = 0;
		}
		return sf_pack(signZ, expZ, (ulong)sigDiff << shift);
	}
	sigA <<= 10;
	sigB <<= 10;
	int expZ;
	ulong sigZ;
	if (expDiff < 0)
	{
		signZ ^= 1;
		if (expB == 0x7FF)
		{
			if (sigB)
				return sf_prop_nan(a, b);
			return sf_pack(signZ, 0x7FF, 0);
		}
		sigA += expA ? 0x4000000000000000UL : sigA;
		sigA = sf_shr_jam(sigA, (uint)(-expDiff));
		sigB |= 0x4000000000000000UL;
		expZ = expB;
		sigZ = sigB - sigA;
	}
	else
	{
		if (expA == 0x7FF)
		{
			if (sigA)
				return sf_prop_nan(a, b);
			return a;
		}
		sigB += expB ? 0x4000000000000000UL : sigB;
		sigB = sf_shr_jam(sigB, (uint)expDiff);
		sigA |= 0x4000000000000000UL;
		expZ = expA;
		sigZ = sigA - sigB;
	}
	return sf_norm_round_pack(signZ, expZ - 1, sigZ);
}

ulong sf_add(ulong a, ulong b)
{
	uint sA = sf_sign(a);
	return sA == sf_sign(b) ? sf_add_mags(a, b, sA) : sf_sub_mags(a, b, sA);
}

ulong sf_sub(ulong a, ulong b)
{
	uint sA = sf_sign(a);
	return sA == sf_sign(b) ? sf_sub_mags(a, b, sA) : sf_add_mags(a, b, sA);
}

/* ================================================================== */
/* mul / fma                                                            */
/* ================================================================== */

ulong sf_mul(ulong a, ulong b)
{
	int expA = sf_expf(a), expB = sf_expf(b);
	ulong sigA = sf_frac(a), sigB = sf_frac(b);
	uint signZ = sf_sign(a) ^ sf_sign(b);
	if (expA == 0x7FF || expB == 0x7FF)
	{
		if (sf_isnan(a) || sf_isnan(b))
			return sf_prop_nan(a, b);
		if (sf_iszero(a) || sf_iszero(b))
			return SF_NAN;	/* inf * 0 */
		return sf_pack(signZ, 0x7FF, 0);
	}
	if (sf_iszero(a) || sf_iszero(b))
		return sf_pack(signZ, 0, 0);
	if (expA)
		sigA |= SF_HIDDEN;
	else
		sigA = sf_norm_sub(sigA, &expA);
	if (expB)
		sigB |= SF_HIDDEN;
	else
		sigB = sf_norm_sub(sigB, &expB);
	return sf_round128(signZ, expA + expB - 2150, sf_mulhi(sigA, sigB), sigA * sigB);
}

/* a * b + c with a single rounding */
ulong sf_fma(ulong a, ulong b, ulong c)
{
	int expA = sf_expf(a), expB = sf_expf(b), expC = sf_expf(c);
	ulong sigA = sf_frac(a), sigB = sf_frac(b), sigC = sf_frac(c);
	uint signP = sf_sign(a) ^ sf_sign(b), signC = sf_sign(c);

	if (sf_isnan(a) || sf_isnan(b) || sf_isnan(c))
		return sf_isnan(a) ? a | SF_QUIET : sf_prop_nan(b, c);
	if (expA == 0x7FF || expB == 0x7FF)
	{
		if (sf_iszero(a) || sf_iszero(b))
			return SF_NAN;
		if (expC == 0x7FF && signC != signP)
			return SF_NAN;
		return sf_pack(signP, 0x7FF, 0);
	}
	if (expC == 0x7FF)
		return c;
	if (sf_iszero(a) || sf_iszero(b))
	{
		/* exact zero product */
		if (sf_iszero(c))
			return signP == signC ? c : 0;
		return c;
	}

	if (expA)
		sigA |= SF_HIDDEN;
	else
		sigA = sf_norm_sub(sigA, &expA);
	if (expB)
		sigB |= SF_HIDDEN;
	else
		sigB = sf_norm_sub(sigB, &expB);

	/* product P in [2^104, 2^106), shifted to [2^124, 2^126) */
	ulong pHi = sf_mulhi(sigA, sigB), pLo = sigA * sigB;
	pHi = (pHi << 20) | (pLo >> 44);
	pLo <<= 20;
	int expP = expA + expB - 2170;	/* value = P * 2^expP */

	if (sf_iszero(c))
		return sf_round128(signP, expP, pHi, pLo);

	if (expC)
		sigC |= SF_HIDDEN;
	else
		sigC = sf_norm_sub(sigC, &expC);
	/* C in [2^124, 2^125) */
	ulong cHi = sigC << 8, cLo = 0;
	int expCv = expC - 1147;

	int d = expP - expCv, e;
	if (d >= 0)
	{
		/* C >>= d with jam */
		if (d >= 128)
		{
			cLo = (ulong)((cHi | cLo) != 0);
			cHi = 0;
		}
		else if (d >= 64)
		{
			ulong stick = (ulong)(cLo != 0 || (d > 64 && (cHi << (128 - d)) != 0));
			cLo = (d == 64 ? cHi : cHi >> (d - 64)) | stick;
			cHi = 0;
		}
		else if (d > 0)
		{
			ulong stick = (ulong)((cLo << (64 - d)) != 0);
			cLo = (cLo >> d) | (cHi << (64 - d)) | stick;
			cHi >>= d;
		}
		e = expP;
	}
	else
	{
		int n = -d;
		if (n >= 128)
		{
			pLo = (ulong)((pHi | pLo) != 0);
			pHi = 0;
		}
		else if (n >= 64)
		{
			ulong stick = (ulong)(pLo != 0 || (n > 64 && (pHi << (128 - n)) != 0));
			pLo = (n == 64 ? pHi : pHi >> (n - 64)) | stick;
			pHi = 0;
		}
		else
		{
			ulong stick = (ulong)((pLo << (64 - n)) != 0);
			pLo = (pLo >> n) | (pHi << (64 - n)) | stick;
			pHi >>= n;
		}
		e = expCv;
	}

	ulong sHi, sLo;
	uint signZ;
	if (signP == signC)
	{
		sLo = pLo + cLo;
		sHi = pHi + cHi + (ulong)(sLo < pLo);
		signZ = signP;
	}
	else
	{
		if (pHi > cHi || (pHi == cHi && pLo >= cLo))
		{
			sLo = pLo - cLo;
			sHi = pHi - cHi - (ulong)(pLo < cLo);
			signZ = signP;
		}
		else
		{
			sLo = cLo - pLo;
			sHi = cHi - pHi - (ulong)(cLo < pLo);
			signZ = signC;
		}
		if (!(sHi | sLo))
			return 0;	/* exact cancellation: +0 */
	}
	return sf_round128(signZ, e, sHi, sLo);
}

/* ================================================================== */
/* div / sqrt                                                           */
/* ================================================================== */

ulong sf_div(ulong a, ulong b)
{
	int expA = sf_expf(a), expB = sf_expf(b);
	ulong sigA = sf_frac(a), sigB = sf_frac(b);
	uint signZ = sf_sign(a) ^ sf_sign(b);
	if (sf_isnan(a) || sf_isnan(b))
		return sf_prop_nan(a, b);
	if (expA == 0x7FF)
		return expB == 0x7FF ? SF_NAN : sf_pack(signZ, 0x7FF, 0);
	if (expB == 0x7FF)
		return sf_pack(signZ, 0, 0);
	if (sf_iszero(b))
		return sf_iszero(a) ? SF_NAN : sf_pack(signZ, 0x7FF, 0);
	if (sf_iszero(a))
		return sf_pack(signZ, 0, 0);
	if (expA)
		sigA |= SF_HIDDEN;
	else
		sigA = sf_norm_sub(sigA, &expA);
	if (expB)
		sigB |= SF_HIDDEN;
	else
		sigB = sf_norm_sub(sigB, &expB);

	int e = expA - expB;
	if (sigA < sigB)
	{
		sigA <<= 1;
		--e;
	}
	/* long division, 11 quotient bits per step; the float estimate of
	   each step is corrected exactly with integer arithmetic */
	ulong rem = sigA - sigB, q = 1;
	const float rb = 1.0f / (float)sigB;
	for (int i = 0; i < 5; i++)
	{
		ulong n = rem << 11;
		ulong dq = (ulong)((float)n * rb);
		if (dq > 2047)
			dq = 2047;
		ulong t = dq * sigB;
		while (t > n)
		{
			--dq;
			t -= sigB;
		}
		while (n - t >= sigB)
		{
			++dq;
			t += sigB;
		}
		rem = n - t;
		q = (q << 11) | dq;
	}
	/* q in [2^55, 2^56) */
	return sf_round_pack(signZ, e + 1022, (q << 7) | (ulong)(rem != 0));
}

ulong sf_sqrt(ulong a)
{
	int exp = sf_expf(a);
	ulong sig = sf_frac(a);
	if (sf_isnan(a))
		return a | SF_QUIET;
	if (sf_iszero(a))
		return a;
	if (sf_sign(a))
		return SF_NAN;
	if (exp == 0x7FF)
		return a;
	if (exp)
		sig |= SF_HIDDEN;
	else
		sig = sf_norm_sub(sig, &exp);
	int E = exp - 1075;
	if (E & 1)
	{
		sig <<= 1;
		--E;
	}
	/* r = floor(sqrt(sig * 2^58)), 56 bits; radicand (hi:lo) */
	ulong hi = sig >> 6, lo = sig << 58;
	ulong rem = 0, root = 0;
	for (int i = 0; i < 64; i++)
	{
		ulong two = (i < 32 ? hi >> (62 - 2 * i) : lo >> (126 - 2 * i)) & 3;
		rem = (rem << 2) | two;
		ulong trial = (root << 2) | 1;
		if (rem >= trial)
		{
			rem -= trial;
			root = (root << 1) | 1;
		}
		else
		{
			root <<= 1;
		}
	}
	return sf_round_pack(0, (E - 58) / 2 - 7 + 1084, (root << 7) | (ulong)(rem != 0));
}

/* ================================================================== */
/* comparisons, min/max                                                 */
/* ================================================================== */

int sf_lt(ulong a, ulong b)
{
	if (sf_isnan(a) || sf_isnan(b))
		return 0;
	uint sA = sf_sign(a), sB = sf_sign(b);
	if (sA != sB)
		return sA && ((a | b) << 1) != 0;
	return a != b && (sA ^ (uint)(a < b));
}

int sf_le(ulong a, ulong b)
{
	if (sf_isnan(a) || sf_isnan(b))
		return 0;
	uint sA = sf_sign(a), sB = sf_sign(b);
	if (sA != sB)
		return sA || ((a | b) << 1) == 0;
	return a == b || (sA ^ (uint)(a < b));
}

int sf_eq(ulong a, ulong b)
{
	if (sf_isnan(a) || sf_isnan(b))
		return 0;
	return a == b || ((a | b) << 1) == 0;
}

#define sf_gt(a, b) sf_lt((b), (a))
#define sf_ge(a, b) sf_le((b), (a))
#define sf_ne(a, b) (!sf_eq((a), (b)))

/* llvm.minnum / maxnum (IEEE mode): a NaN operand yields the other one,
   -0 orders below +0 */
ulong sf_minnum(ulong a, ulong b)
{
	if (sf_isnan(a))
		return sf_isnan(b) ? a | SF_QUIET : b;
	if (sf_isnan(b))
		return a;
	if (sf_iszero(a) && sf_iszero(b))
		return a | b;
	return sf_lt(b, a) ? b : a;
}

ulong sf_maxnum(ulong a, ulong b)
{
	if (sf_isnan(a))
		return sf_isnan(b) ? a | SF_QUIET : b;
	if (sf_isnan(b))
		return a;
	if (sf_iszero(a) && sf_iszero(b))
		return a & b;
	return sf_lt(a, b) ? b : a;
}

/* ================================================================== */
/* conversions and exact helpers                                        */
/* ================================================================== */

ulong sf_from_u53(ulong v)	/* exact, v < 2^53 */
{
	if (!v)
		return 0;
	int shift = sf_clz64(v) - 11;
	return sf_pack(0, 0x432 - shift, v << shift);
}

ulong sf_from_i32(int i)
{
	if (i == 0)
		return 0;
	uint s = i < 0;
	ulong m = s ? (ulong)(-(sf_i64)i) : (ulong)i;
	return sf_from_u53(m) | ((ulong)s << 63);
}

ulong sf_from_f32(float f)	/* exact */
{
	uint u = sf_as_uint(f);
	uint s = u >> 31;
	int e = (int)((u >> 23) & 0xFF);
	ulong m = u & 0x7FFFFF;
	if (e == 0xFF)
		return sf_pack(s, 0x7FF, m << 29);
	if (e == 0)
	{
		if (!m)
			return (ulong)s << 63;
		int shift = sf_clz64(m) - 40;	/* hidden bit to 23 */
		m <<= shift;
		e = 1 - shift;
		m &= 0x7FFFFF;
	}
	return sf_pack(s, e - 127 + 1023, m << 29);
}

/* round toward zero to int, saturating, NaN -> 0 (v_cvt_i32_f64) */
int sf_to_i32(ulong a)
{
	if (sf_isnan(a))
		return 0;
	int exp = sf_expf(a);
	uint s = sf_sign(a);
	if (exp < 0x3FF)
		return 0;
	if (exp >= 0x3FF + 31)
		return s ? (int)0x80000000 : 0x7FFFFFFF;
	ulong sig = sf_frac(a) | SF_HIDDEN;
	sf_i64 v = (sf_i64)(sig >> (0x433 - exp));
	return (int)(s ? -v : v);
}

/* approximate conversion, only used for Newton seeds */
float sf_to_f32_approx(ulong a)
{
	int exp = sf_expf(a);
	if (exp == 0x7FF || exp < 0x3FF - 126 || exp > 0x3FF + 127)
		return 0.0f;
	float m = (float)((sf_frac(a) | SF_HIDDEN) >> 29) * 1.1920928955078125e-7f;	/* 2^-23 */
	m = ldexp(m, exp - 0x3FF);
	return sf_sign(a) ? -m : m;
}

ulong sf_ldexp(ulong a, int n)
{
	int exp = sf_expf(a);
	ulong sig = sf_frac(a);
	uint s = sf_sign(a);
	if (exp == 0x7FF)
		return sf_isnan(a) ? a | SF_QUIET : a;
	if (sf_iszero(a))
		return a;
	if (exp)
		sig |= SF_HIDDEN;
	else
		sig = sf_norm_sub(sig, &exp);
	if (n > 4000)
		n = 4000;
	if (n < -4000)
		n = -4000;
	return sf_round_pack(s, exp - 1 + n, sig << 10);
}

/* mantissa in [0.5, 1) with the sign of a; 0/inf/nan: (a, 0) */
ulong sf_frexp(ulong a, int* e)
{
	int exp = sf_expf(a);
	ulong sig = sf_frac(a);
	*e = 0;
	if (exp == 0x7FF || sf_iszero(a))
		return exp == 0x7FF && sf_isnan(a) ? a | SF_QUIET : a;
	if (!exp)
		sig = sf_norm_sub(sig, &exp) & SF_FRAC;
	*e = exp - 1022;
	return sf_pack(sf_sign(a), 1022, sig);
}

/* round to nearest-even integer (llvm.rint) */
ulong sf_rint(ulong a)
{
	int exp = sf_expf(a);
	if (exp <= 0x3FE)
	{
		if (sf_iszero(a))
			return a;
		if (exp == 0x3FE && sf_frac(a))
			return (a & SF_SIGN) | SF_ONE;
		return a & SF_SIGN;
	}
	if (exp >= 0x433)
		return sf_isnan(a) ? a | SF_QUIET : a;
	ulong lastBit = (ulong)1 << (0x433 - exp);
	ulong roundMask = lastBit - 1;
	ulong z = a + (lastBit >> 1);
	if (!(z & roundMask))
		z &= ~lastBit;
	return z & ~roundMask;
}

/* round toward -inf (llvm.floor) */
ulong sf_floor(ulong a)
{
	int exp = sf_expf(a);
	if (exp <= 0x3FE)
	{
		if (sf_iszero(a))
			return a;
		return sf_sign(a) ? (SF_SIGN | SF_ONE) : 0;
	}
	if (exp >= 0x433)
		return sf_isnan(a) ? a | SF_QUIET : a;
	ulong roundMask = ((ulong)1 << (0x433 - exp)) - 1;
	if (!(a & roundMask))
		return a;
	ulong z = a & ~roundMask;
	return sf_sign(a) ? sf_sub(z, SF_ONE) : z;
}

ulong sf_copysign(ulong mag, ulong sgn)
{
	return (mag & ~SF_SIGN) | (sgn & SF_SIGN);
}

/* ================================================================== */
/* approximate hardware instructions used as Newton seeds by ocml       */
/* (the library refines them with fma steps to full accuracy, so any    */
/* seed of ~2^-22 relative error gives the same final result)           */
/* ================================================================== */

ulong sf_rcp_seed(ulong a)
{
	if (sf_iszero(a))
		return sf_pack(sf_sign(a), 0x7FF, 0);
	if (sf_isnan(a))
		return a | SF_QUIET;
	if (sf_expf(a) == 0x7FF)
		return a & SF_SIGN;
	int e2;
	ulong m = sf_frexp(a & ~SF_SIGN, &e2);	/* |a| = m * 2^e2, m in [0.5, 1) */
	ulong r = sf_from_f32(1.0f / sf_to_f32_approx(m));
	return sf_ldexp(r, -e2) | (a & SF_SIGN);
}

ulong sf_rsq_seed(ulong a)
{
	if (sf_iszero(a))
		return sf_pack(sf_sign(a), 0x7FF, 0);
	if (sf_isnan(a) || sf_sign(a))
		return SF_NAN;
	if (sf_expf(a) == 0x7FF)
		return 0;
	int e2;
	ulong m = sf_frexp(a, &e2);	/* a = m * 2^e2, m in [0.5, 1) */
	if (e2 & 1)
	{
		m = sf_ldexp(m, 1);
		--e2;
	}
	ulong r = sf_from_f32(rsqrt(sf_to_f32_approx(m)));
	return sf_ldexp(r, -e2 / 2);
}

/* v_trig_preop_f64: 53-bit segment of 2/pi selected by the exponent of x */
SF_CONST uint SF_2OVERPI[38] = {
	0xA2F9836Eu, 0x4E441529u, 0xFC2757D1u, 0xF534DDC0u, 0xDB629599u, 0x3C439041u, 0xFE5163ABu, 0xDEBBC561u,
	0xB7246E3Au, 0x424DD2E0u, 0x06492EEAu, 0x09D1921Cu, 0xFE1DEB1Cu, 0xB129A73Eu, 0xE88235F5u, 0x2EBB4484u,
	0xE99C7026u, 0xB45F7E41u, 0x3991D639u, 0x835339F4u, 0x9C845F8Bu, 0xBDF9283Bu, 0x1FF897FFu, 0xDE05980Fu,
	0xEF2F118Bu, 0x5A0A6D1Fu, 0x6D367ECFu, 0x27CB09B7u, 0x4F463F66u, 0x9E5FEA2Du, 0x7527BAC7u, 0xEBE5F17Bu,
	0x3D0739F7u, 0x8A5292EAu, 0x6BFB5FB1u, 0x1F8D5D08u, 0x56033046u, 0xFC7B0000u };

ulong sf_trig_preop(ulong x, int seg)
{
	int E = sf_expf(x);
	int shift = seg * 53 + (E > 1077 ? E - 1077 : 0);
	int w = shift >> 5, o = shift & 31;
	ulong hi = ((ulong)SF_2OVERPI[w] << 32) | SF_2OVERPI[w + 1];
	ulong lo = SF_2OVERPI[w + 2];
	ulong win = o ? (hi << o) | (lo >> (32 - o)) : hi;
	int scale = -53 - shift + (E >= 1968 ? 128 : 0);
	return sf_ldexp(sf_from_u53(win >> 11), scale);
}

/* ================================================================== */
/* ocml ports (ROCm device libraries, f64 variants with the OpenCL      */
/* defaults: finite_only_opt = 0). Each line mirrors one IR operation.  */
/* ================================================================== */

#define F_ sf_fma
#define M_ sf_mul
#define A_ sf_add
#define S_ sf_sub
#define N_ sf_neg

ulong sf_ocml_exp(ulong x)
{
	ulong t = sf_rint(M_(x, 0x3FF71547652B82FEUL));
	ulong nt = N_(t);
	ulong r = F_(nt, 0x3FE62E42FEFA39EFUL, x);
	r = F_(nt, 0x3C7ABC9E3B39803FUL, r);
	ulong p = F_(r, 0x3E5ADE156A5DCB37UL, 0x3E928AF3FCA7AB0CUL);
	p = F_(r, p, 0x3EC71DEE623FDE64UL);
	p = F_(r, p, 0x3EFA01997C89E6B0UL);
	p = F_(r, p, 0x3F2A01A014761F6EUL);
	p = F_(r, p, 0x3F56C16C1852B7B0UL);
	p = F_(r, p, 0x3F81111111122322UL);
	p = F_(r, p, 0x3FA55555555502A1UL);
	p = F_(r, p, 0x3FC5555555555511UL);
	p = F_(r, p, 0x3FE000000000000BUL);
	p = F_(r, p, SF_ONE);
	p = F_(r, p, SF_ONE);
	ulong e = sf_ldexp(p, sf_to_i32(t));
	if (sf_gt(x, 0x4090000000000000UL))	/* 1024 */
		e = SF_INF;
	if (sf_lt(x, 0xC090CC0000000000UL))	/* -1075 */
		e = 0;
	return e;
}

ulong sf_ocml_log(ulong x)
{
	int ex;
	ulong m = sf_frexp(x, &ex);
	int small = sf_lt(m, 0x3FE5555555555555UL);
	ulong v7 = M_(m, small ? 0x4000000000000000UL : SF_ONE);
	int v9 = ex - small;
	ulong v10 = A_(v7, 0xBFF0000000000000UL);
	ulong v11 = A_(v7, SF_ONE);
	ulong v12 = A_(v11, 0xBFF0000000000000UL);
	ulong v13 = S_(v7, v12);
	ulong v14 = sf_rcp_seed(v11);
	ulong v15 = N_(v11);
	ulong v16 = F_(v15, v14, SF_ONE);
	ulong v17 = F_(v16, v14, v14);
	ulong v18 = F_(v15, v17, SF_ONE);
	ulong v19 = F_(v18, v17, v17);
	ulong v20 = M_(v10, v19);
	ulong v21 = M_(v11, v20);
	ulong v22 = N_(v21);
	ulong v23 = F_(v20, v11, v22);
	ulong v24 = F_(v20, v13, v23);
	ulong v25 = A_(v21, v24);
	ulong v26 = S_(v25, v21);
	ulong v27 = S_(v24, v26);
	ulong v28 = S_(v10, v25);
	ulong v29 = S_(v10, v28);
	ulong v30 = S_(v29, v25);
	ulong v31 = S_(v30, v27);
	ulong v32 = A_(v28, v31);
	ulong v33 = M_(v19, v32);
	ulong v34 = A_(v20, v33);
	ulong v35 = S_(v34, v20);
	ulong v36 = S_(v33, v35);
	ulong v37 = M_(v34, v34);
	ulong v38 = F_(v37, 0x3FC3AB76BF559E2BUL, 0x3FC385386B47B09AUL);
	ulong v39 = F_(v37, v38, 0x3FC7474DD7F4DF2EUL);
	ulong v40 = F_(v37, v39, 0x3FCC71C016291751UL);
	ulong v41 = F_(v37, v40, 0x3FD249249B27ACF1UL);
	ulong v42 = F_(v37, v41, 0x3FD99999998EF7B6UL);
	ulong v43 = F_(v37, v42, 0x3FE5555555555780UL);
	ulong v44 = sf_ldexp(v34, 1);
	ulong v45 = sf_ldexp(v36, 1);
	ulong v46 = M_(v34, v37);
	ulong v47 = M_(v46, v43);
	ulong v48 = A_(v44, v47);
	ulong v49 = S_(v48, v44);
	ulong v50 = S_(v47, v49);
	ulong v51 = A_(v45, v50);
	ulong v52 = A_(v48, v51);
	ulong v53 = S_(v52, v48);
	ulong v54 = S_(v51, v53);
	ulong v55 = sf_from_i32(v9);
	ulong v56 = M_(v55, 0x3FE62E42FEFA39EFUL);
	ulong v57 = N_(v56);
	ulong v58 = F_(v55, 0x3FE62E42FEFA39EFUL, v57);
	ulong v59 = F_(v55, 0x3C7ABC9E3B39803FUL, v58);
	ulong v60 = A_(v56, v59);
	ulong v61 = S_(v60, v56);
	ulong v62 = S_(v59, v61);
	ulong v63 = A_(v60, v52);
	ulong v64 = S_(v63, v60);
	ulong v65 = S_(v63, v64);
	ulong v66 = S_(v60, v65);
	ulong v67 = S_(v52, v64);
	ulong v68 = A_(v67, v66);
	ulong v69 = A_(v62, v54);
	ulong v70 = S_(v69, v62);
	ulong v71 = S_(v69, v70);
	ulong v72 = S_(v62, v71);
	ulong v73 = S_(v54, v70);
	ulong v74 = A_(v73, v72);
	ulong v75 = A_(v69, v68);
	ulong v76 = A_(v63, v75);
	ulong v77 = S_(v76, v63);
	ulong v78 = S_(v75, v77);
	ulong v79 = A_(v74, v78);
	ulong v80 = A_(v76, v79);
	ulong r = v80;
	if (sf_eq(sf_abs(x), SF_INF))
		r = x;
	if (sf_lt(x, 0))
		r = SF_NAN;
	if (sf_eq(x, 0))
		r = 0xFFF0000000000000UL;
	return r;
}

ulong sf_ocml_acos(ulong x)
{
	ulong v2 = sf_abs(x);
	int v3 = sf_ge(v2, 0x3FE0000000000000UL);
	ulong v4 = F_(v2, 0xBFE0000000000000UL, 0x3FE0000000000000UL);
	ulong v5 = M_(x, x);
	ulong v6 = v3 ? v4 : v5;
	ulong v7 = F_(v6, 0x3FA059859FEA6A70UL, 0xBF90A5A378A05EAFUL);
	ulong v8 = F_(v6, v7, 0x3F94052137024D6AUL);
	ulong v9 = F_(v6, v8, 0x3F7AB3A098A70509UL);
	ulong v10 = F_(v6, v9, 0x3F88ED60A300C8D2UL);
	ulong v11 = F_(v6, v10, 0x3F8C6FA84B77012BUL);
	ulong v12 = F_(v6, v11, 0x3F91C6C111DCCB70UL);
	ulong v13 = F_(v6, v12, 0x3F96E89F0A0ADACFUL);
	ulong v14 = F_(v6, v13, 0x3F9F1C72C668963FUL);
	ulong v15 = F_(v6, v14, 0x3FA6DB6DB41CE4BDUL);
	ulong v16 = F_(v6, v15, 0x3FB333333336FD5BUL);
	ulong v17 = F_(v6, v16, 0x3FC5555555555380UL);
	ulong v18 = M_(v6, v17);
	if (!v3)
	{
		ulong v19 = F_(x, v18, x);
		return F_(0x3FEDD9AD336A0500UL, 0x3FFAF154EEB562D6UL, N_(v19));
	}
	ulong v23 = sf_rsq_seed(v4);
	ulong v24 = M_(v4, v23);
	ulong v25 = M_(v23, 0x3FE0000000000000UL);
	ulong v26 = N_(v25);
	ulong v27 = F_(v26, v24, 0x3FE0000000000000UL);
	ulong v28 = F_(v25, v27, v25);
	ulong v29 = F_(v24, v27, v24);
	ulong v30 = N_(v29);
	ulong v31 = F_(v30, v29, v4);
	ulong v32 = F_(v31, v28, v29);
	int v33 = sf_eq(v4, 0);
	ulong v34 = v33 ? v4 : v32;
	ulong v35 = M_(v34, v34);
	ulong v36 = N_(v35);
	ulong v37 = F_(v34, v34, v36);
	ulong v38 = S_(v4, v35);
	ulong v39 = S_(v4, v38);
	ulong v40 = S_(v39, v35);
	ulong v41 = S_(v40, v37);
	ulong v42 = A_(v38, v41);
	ulong v43 = M_(v34, 0x4000000000000000UL);
	ulong v44 = sf_rcp_seed(v43);
	ulong v45 = N_(v43);
	ulong v46 = F_(v45, v44, SF_ONE);
	ulong v47 = F_(v46, v44, v44);
	ulong v48 = F_(v45, v47, SF_ONE);
	ulong v49 = F_(v48, v47, v47);
	ulong v50 = M_(v42, v49);
	ulong v51 = F_(v45, v50, v42);
	ulong v52 = F_(v51, v49, v50);
	ulong v53 = v33 ? 0 : v52;
	ulong v54 = A_(v34, v53);
	ulong v55 = S_(v54, v34);
	ulong v56 = S_(v53, v55);
	ulong v57 = F_(v54, v18, v54);
	ulong v58 = M_(v57, 0xC000000000000000UL);
	ulong v59 = F_(0x3FFDD9AD336A0500UL, 0x3FFAF154EEB562D6UL, v58);
	ulong v60 = F_(v54, v18, v56);
	ulong v61 = A_(v54, v60);
	ulong v62 = M_(v61, 0x4000000000000000UL);
	ulong r = sf_lt(x, 0) ? v59 : v62;
	if (sf_eq(x, 0xBFF0000000000000UL))
		r = 0x400921FB54442D18UL;
	if (sf_eq(x, SF_ONE))
		r = 0;
	return r;
}

/* trig argument reduction: x = q * pi/2 + (hi + lo), q = result & 3 */
int sf_ocml_trigredsmall(ulong x, ulong* rhi, ulong* rlo)
{
	ulong v3 = sf_rint(M_(x, 0x3FE45F306DC9C883UL));
	ulong v4 = F_(v3, 0xBFF921FB54442D18UL, x);
	ulong v5 = F_(v3, 0xBC91A62633145C00UL, v4);
	ulong v6 = M_(v3, 0x3C91A62633145C00UL);
	ulong v7 = N_(v6);
	ulong v8 = F_(v3, 0x3C91A62633145C00UL, v7);
	ulong v9 = S_(v4, v6);
	ulong v10 = S_(v4, v9);
	ulong v11 = S_(v10, v6);
	ulong v12 = S_(v9, v5);
	ulong v13 = A_(v12, v11);
	ulong v14 = S_(v13, v8);
	ulong v15 = F_(v3, 0xB97B839A252049C0UL, v14);
	ulong v16 = A_(v5, v15);
	ulong v17 = S_(v16, v5);
	*rlo = S_(v15, v17);
	*rhi = v16;
	return sf_to_i32(v3) & 3;
}

int sf_ocml_trigredlarge(ulong x, ulong* rhi, ulong* rlo)
{
	ulong v2 = sf_trig_preop(x, 0);
	ulong v3 = sf_trig_preop(x, 1);
	ulong v6 = sf_ge(x, 0x7B00000000000000UL) ? sf_ldexp(x, -128) : x;
	ulong v7 = M_(v3, v6);
	ulong v8 = M_(v2, v6);
	ulong v9 = N_(v8);
	ulong v10 = F_(v2, v6, v9);
	ulong v11 = A_(v7, v10);
	ulong v12 = A_(v8, v11);
	ulong v13 = sf_ldexp(v12, -2);
	ulong v14 = sf_floor(v13);
	ulong v15 = S_(v13, v14);
	ulong v16 = sf_minnum(v15, 0x3FEFFFFFFFFFFFFFUL);
	ulong v20 = sf_isnan(v13) ? v13 : v16;
	ulong v24 = sf_eq(sf_abs(v13), SF_INF) ? 0 : v20;
	ulong v25 = S_(v11, v7);
	ulong v26 = S_(v10, v25);
	ulong v27 = S_(v11, v25);
	ulong v28 = S_(v7, v27);
	ulong v29 = A_(v26, v28);
	ulong v30 = N_(v7);
	ulong v31 = F_(v3, v6, v30);
	ulong v32 = sf_trig_preop(x, 2);
	ulong v33 = M_(v32, v6);
	ulong v34 = A_(v33, v31);
	ulong v35 = A_(v34, v29);
	ulong v36 = S_(v12, v8);
	ulong v37 = S_(v11, v36);
	ulong v38 = A_(v37, v35);
	ulong v39 = S_(v38, v37);
	ulong v40 = S_(v35, v39);
	ulong v41 = S_(v35, v34);
	ulong v42 = S_(v29, v41);
	ulong v43 = S_(v35, v41);
	ulong v44 = S_(v34, v43);
	ulong v45 = A_(v42, v44);
	ulong v46 = S_(v34, v33);
	ulong v47 = S_(v31, v46);
	ulong v48 = S_(v34, v46);
	ulong v49 = S_(v33, v48);
	ulong v50 = A_(v47, v49);
	ulong v51 = A_(v50, v45);
	ulong v52 = N_(v33);
	ulong v53 = F_(v32, v6, v52);
	ulong v54 = A_(v53, v51);
	ulong v55 = A_(v40, v54);
	ulong v56 = sf_ldexp(v24, 2);
	ulong v57 = A_(v38, v56);
	ulong v59 = sf_lt(v57, 0) ? 0x4010000000000000UL : 0;
	ulong v60 = A_(v56, v59);
	ulong v61 = A_(v38, v60);
	int v62 = sf_to_i32(v61);
	ulong v63 = sf_from_i32(v62);
	ulong v64 = S_(v60, v63);
	ulong v65 = A_(v38, v64);
	ulong v66 = S_(v65, v64);
	ulong v67 = S_(v38, v66);
	ulong v68 = A_(v55, v67);
	int v69 = sf_ge(v65, 0x3FE0000000000000UL);
	int v71 = v69 + v62;
	ulong v73 = S_(v65, v69 ? SF_ONE : 0);
	ulong v74 = A_(v73, v68);
	ulong v75 = S_(v74, v73);
	ulong v76 = S_(v68, v75);
	ulong v77 = M_(v74, 0x3FF921FB54442D18UL);
	ulong v78 = N_(v77);
	ulong v79 = F_(v74, 0x3FF921FB54442D18UL, v78);
	ulong v80 = F_(v74, 0x3C91A62633145C07UL, v79);
	ulong v81 = F_(v76, 0x3FF921FB54442D18UL, v80);
	ulong v82 = A_(v77, v81);
	ulong v83 = S_(v82, v77);
	*rlo = S_(v81, v83);
	*rhi = v82;
	return v71 & 3;
}

/* sin and cos of the reduced argument (hi, lo) */
void sf_ocml_sincosred2(ulong x, ulong y, ulong* s, ulong* c)
{
	ulong v3 = M_(x, x);
	ulong v4 = M_(v3, 0x3FE0000000000000UL);
	ulong v5 = S_(SF_ONE, v4);
	ulong v6 = S_(SF_ONE, v5);
	ulong v7 = S_(v6, v4);
	ulong v8 = M_(v3, v3);
	ulong v9 = F_(v3, 0xBDA907DB46CC5E42UL, 0x3E21EEB69037AB78UL);
	ulong v10 = F_(v3, v9, 0xBE927E4FA17F65F6UL);
	ulong v11 = F_(v3, v10, 0x3EFA01A019F4EC90UL);
	ulong v12 = F_(v3, v11, 0xBF56C16C16C16967UL);
	ulong v13 = F_(v3, v12, 0x3FA5555555555555UL);
	ulong v14 = N_(y);
	ulong v15 = F_(x, v14, v7);
	ulong v16 = F_(v8, v13, v15);
	*c = A_(v5, v16);
	ulong v18 = F_(v3, 0x3DE5E0B2F9A43BB8UL, 0xBE5AE600B42FDFA7UL);
	ulong v19 = F_(v3, v18, 0x3EC71DE3796CDE01UL);
	ulong v20 = F_(v3, v19, 0xBF2A01A019E83E5CUL);
	ulong v21 = F_(v3, v20, 0x3F81111111110BB3UL);
	ulong v22 = N_(v3);
	ulong v23 = M_(x, v22);
	ulong v24 = M_(y, 0x3FE0000000000000UL);
	ulong v25 = F_(v23, v21, v24);
	ulong v26 = F_(v3, v25, v14);
	ulong v27 = F_(v23, 0xBFC5555555555555UL, v26);
	*s = S_(x, v27);
}

/* returns sin(x), stores cos(x) */
ulong sf_ocml_sincos(ulong x, ulong* cp)
{
	ulong ax = sf_abs(x);
	ulong rhi, rlo;
	int q = sf_lt(ax, 0x41D0000000000000UL)	/* 2^30 */
		? sf_ocml_trigredsmall(ax, &rhi, &rlo)
		: sf_ocml_trigredlarge(ax, &rhi, &rlo);
	ulong s9, c10;
	sf_ocml_sincosred2(rhi, rlo, &s9, &c10);
	ulong flip = q > 1 ? SF_SIGN : 0;
	ulong sv = (q & 1) == 0 ? s9 : c10;
	sv ^= (x & SF_SIGN) ^ flip;
	ulong cv = (q & 1) == 0 ? c10 : N_(s9);
	cv ^= flip;
	/* fcmp one(ax, inf): false for inf and NaN */
	int finite = !sf_isnan(ax) && ax != SF_INF;
	*cp = finite ? cv : 0x7FF8000000000000UL;
	return finite ? sv : 0x7FF8000000000000UL;
}

ulong sf_ocml_fmod(ulong x, ulong y)
{
	ulong v3 = sf_abs(x), v4 = sf_abs(y);
	ulong r;
	if (sf_gt(v3, v4))
	{
		int v8, v12;
		ulong v9 = sf_frexp(v3, &v8);
		ulong v10 = sf_ldexp(v9, 26);
		ulong v14 = sf_frexp(v4, &v12);
		int v13 = v12 - 1;
		ulong v15 = sf_ldexp(v14, 1);
		int v16 = v8 - v12;
		ulong v17 = sf_div(SF_ONE, v15);
		ulong v34 = v10;
		int v33 = v16;
		if (v16 > 26)
		{
			ulong v20 = v10;
			int v21 = v16;
			for (;;)
			{
				ulong v22 = M_(v17, v20);
				ulong v23 = sf_rint(v22);
				ulong v25 = F_(N_(v23), v15, v20);
				ulong v28 = sf_lt(v25, 0) ? A_(v15, v25) : v25;
				ulong v29 = sf_ldexp(v28, 26);
				int v30 = v21 - 26;
				int more = (uint)v21 > 52u;
				v20 = v29;
				v21 = v30;
				if (!more)
					break;
			}
			v34 = v20;
			v33 = v21;
		}
		ulong v36 = sf_ldexp(v34, v33 - 25);
		ulong v37 = M_(v17, v36);
		ulong v38 = sf_rint(v37);
		ulong v40 = F_(N_(v38), v15, v36);
		ulong v43 = sf_lt(v40, 0) ? A_(v15, v40) : v40;
		ulong v44 = sf_ldexp(v43, v13);
		r = (x & SF_SIGN) ^ v44;
	}
	else
	{
		r = sf_eq(v3, v4) ? sf_copysign(0, x) : x;
	}
	if (sf_eq(y, 0))
		r = SF_NAN;
	if (!(!sf_isnan(y) && !sf_isnan(v3) && v3 != SF_INF))	/* ord(y) && one(|x|, inf) */
		r = SF_NAN;
	return r;
}

#undef F_
#undef M_
#undef A_
#undef S_
#undef N_

#endif /* PS_FP32 */
/* Real.cl - the floating-point type of the model computation.

   Every double-precision quantity of the kernels is a `real`, and every
   operation on one is written with the R_* macros below.

   Default build (the device supports cl_khr_fp64):
     real == double and each macro expands textually to the plain C
     expression it replaces (R_ADD(a, b) -> ((a) + (b)), R_DIV -> ddiv,
     R_EXP -> exp, ...), so the compiler sees exactly the code it always
     compiled.

   -D PS_FP32 (set by the host when the device has no FP64 support):
     real holds the IEEE double's bit pattern and every macro calls the
     software binary64 implementation in SoftFP64.cl: correctly rounded
     add/sub/mul/fma/div/sqrt and bit-exact ports of the AMD device-library
     exp/log/acos/sincos/fmod. The results are identical, bit for bit, to
     the FP64 build on AMD hardware. A real still occupies 8 bytes with
     8-byte alignment - it IS the double - so structs, buffers and the host
     side are unchanged. `real` is a struct on purpose: a plain operator or
     literal applied to it is a compile error instead of a silent
     integer/float operation.

   Fused multiply-add: under FP_CONTRACT ON clang fuses an add/sub with a
   multiply that is its direct operand (left operand checked first) into
   one fma - a single rounding. Those spots are written with the explicit
   R_MADD/R_ADDM/R_MSUB/R_SUBM/R_ADDTOM/R_SUBFROMM forms: in the FP64 build
   they are the same expressions (so the compiler fuses them as before),
   in the PS_FP32 build they are software fma calls. Keep them in sync
   when editing: an R_ADD with an R_MUL operand would be fused by the FP64
   compiler but rounded twice by the emulation.

   Rules for kernel code:
     - arithmetic, comparisons and math functions on reals only through the
       R_* macros;
     - literals through R_C() (a float-exact floating literal, e.g.
       R_C(1.0), R_C(90.0)); other constants have named macros (R_PI,
       R_DEG2RAD, ...);
     - double kernel arguments are real_arg, read with R_ARG(). */

#ifndef PS_FP32

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef double real;
typedef double real_arg;

#define R_ARG(x)            (x)
#define R_C(x)              (x)
#define R_FROM_INT(i)       ((double)(i))

#define R_ADD(a, b)         ((a) + (b))
#define R_SUB(a, b)         ((a) - (b))
#define R_MUL(a, b)         ((a) * (b))
#define R_DIV(a, b)         ddiv((a), (b))
#define R_NEG(a)            (-(a))
#define R_ADDTO(a, b)       ((a) += (b))
#define R_SUBFROM(a, b)     ((a) -= (b))

/* the contracted (fma) forms, see above */
#define R_MADD(a, b, c)     (((a) * (b)) + (c))
#define R_ADDM(c, a, b)     ((c) + ((a) * (b)))
#define R_MSUB(a, b, c)     (((a) * (b)) - (c))
#define R_SUBM(c, a, b)     ((c) - ((a) * (b)))
#define R_ADDTOM(x, a, b)   ((x) += ((a) * (b)))
#define R_SUBFROMM(x, a, b) ((x) -= ((a) * (b)))

#define R_LT(a, b)          ((a) < (b))
#define R_GT(a, b)          ((a) > (b))
#define R_LE(a, b)          ((a) <= (b))
#define R_GE(a, b)          ((a) >= (b))
#define R_EQ(a, b)          ((a) == (b))
#define R_NE(a, b)          ((a) != (b))

#define R_FABS(x)           fabs(x)
#define R_SQRT(x)           sqrt(x)
#define R_EXP(x)            exp(x)
#define R_LOG(x)            log(x)
#define R_ACOS(x)           acos(x)
#define R_SINCOS(x, pc)     sincos((x), (pc))
#define R_FMOD_2PI(x)       fmod((x), (2 * PI))
#define R_CLAMP(x, lo, hi)  clamp((x), (lo), (hi))
#define R_POW2(x)           pow((x), 2.0)
#define R_ISNAN(x)          isnan(x)

#define R_PI                PI
#define R_2PI               (2 * PI)
#define R_48PI              (24.0 * 2.0 * PI)
#define R_DEG2RAD           DEG2RAD
#define R_RAD2DEG           RAD2DEG
#define R_TINY              TINY
#define R_1E_10             1e-10
#define R_1E30              1e30
#define R_1E40              1e40

#else /* PS_FP32 */

typedef struct { ulong u; } real;
typedef ulong real_arg;

real sf_r(ulong u)
{
	real r;
	r.u = u;
	return r;
}

/* sincos with the OpenCL signature: returns sin, stores cos */
real sf_sincos_r(real x, real* c)
{
	ulong cu;
	ulong s = sf_ocml_sincos(x.u, &cu);
	c->u = cu;
	return sf_r(s);
}

#define R_ARG(v)            sf_r(v)
#define R_C_(x)             sf_r(sf_from_f32(x##f))
#define R_C(x)              R_C_(x)
#define R_FROM_INT(i)       sf_r(sf_from_i32(i))

#define R_ADD(a, b)         sf_r(sf_add((a).u, (b).u))
#define R_SUB(a, b)         sf_r(sf_sub((a).u, (b).u))
#define R_MUL(a, b)         sf_r(sf_mul((a).u, (b).u))
#define R_DIV(a, b)         sf_r(sf_div((a).u, (b).u))
#define R_NEG(a)            sf_r(sf_neg((a).u))
#define R_ADDTO(a, b)       ((a) = R_ADD((a), (b)))
#define R_SUBFROM(a, b)     ((a) = R_SUB((a), (b)))

#define R_MADD(a, b, c)     sf_r(sf_fma((a).u, (b).u, (c).u))
#define R_ADDM(c, a, b)     sf_r(sf_fma((a).u, (b).u, (c).u))
#define R_MSUB(a, b, c)     sf_r(sf_fma((a).u, (b).u, sf_neg((c).u)))
#define R_SUBM(c, a, b)     sf_r(sf_fma(sf_neg((a).u), (b).u, (c).u))
#define R_ADDTOM(x, a, b)   ((x) = R_ADDM((x), (a), (b)))
#define R_SUBFROMM(x, a, b) ((x) = R_SUBM((x), (a), (b)))

#define R_LT(a, b)          sf_lt((a).u, (b).u)
#define R_GT(a, b)          sf_gt((a).u, (b).u)
#define R_LE(a, b)          sf_le((a).u, (b).u)
#define R_GE(a, b)          sf_ge((a).u, (b).u)
#define R_EQ(a, b)          sf_eq((a).u, (b).u)
#define R_NE(a, b)          sf_ne((a).u, (b).u)

#define R_FABS(x)           sf_r(sf_abs((x).u))
#define R_SQRT(x)           sf_r(sf_sqrt((x).u))
#define R_EXP(x)            sf_r(sf_ocml_exp((x).u))
#define R_LOG(x)            sf_r(sf_ocml_log((x).u))
#define R_ACOS(x)           sf_r(sf_ocml_acos((x).u))
#define R_SINCOS(x, pc)     sf_sincos_r((x), (pc))
#define R_FMOD_2PI(x)       sf_r(sf_ocml_fmod((x).u, 0x401921FB54442D18UL))
#define R_CLAMP(x, lo, hi)  sf_r(sf_minnum(sf_maxnum((x).u, (lo).u), (hi).u))
#define R_POW2(x)           sf_r(sf_mul((x).u, (x).u))	/* the compiler turns pow(x, 2.0) into x * x */
#define R_ISNAN(x)          sf_isnan((x).u)

/* the bit patterns of the FP64 build's (constant-folded) values */
#define R_PI                sf_r(0x400921FB54442D18UL)
#define R_2PI               sf_r(0x401921FB54442D18UL)
#define R_48PI              sf_r(0x4062D97C7F3321D2UL)
#define R_DEG2RAD           sf_r(0x3F91DF46A2529D39UL)
#define R_RAD2DEG           sf_r(0x404CA5DC1A63C1F8UL)
#define R_TINY              sf_r(0x3E45798EE2308C3AUL)
#define R_1E_10             sf_r(0x3DDB7CDFD9D7BDBBUL)
#define R_1E30              sf_r(0x46293E5939A08CEAUL)
#define R_1E40              sf_r(0x483D6329F1C35CA5UL)

#endif /* PS_FP32 */
#pragma OPENCL FP_CONTRACT ON

/* cl_khr_fp64 is enabled in Real.cl (FP64 build only) */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

//struct __attribute__((packed)) freq_context
//struct mfreq_context
//struct __attribute__((aligned(8))) mfreq_context
typedef struct mfreq_context
{
	//double* Area;
	//double* Dg;
	//double* alpha;
	//double* covar;
	//double* dytemp;
	//double* ytemp;

	real Area[MAX_N_FAC + 1];
	/* The point- and fit-dimensioned work arrays (alpha, covar, dytemp,
	   ytemp, jp_*, e_*, de, de0) live in a separate runtime-sized scratch
	   buffer - one slice of freq_context.scrStride doubles per work-group,
	   at the offsets recorded in freq_context - instead of compile-time
	   worst-case arrays here. That cuts per-context memory ~6x (2.27 MB ->
	   ~0.4 MB for typical workunits). */
	real beta[MAX_N_PAR + 1];
	real atry[MAX_N_PAR + 1];
	real da[MAX_N_PAR + 1];
	real cg[MAX_N_PAR + 1];
	real Blmat[4][4];
	real Dblm[3][4][4];
	real dave[MAX_N_PAR + 1];
	real dyda[MAX_N_PAR + 1];

	real sh_big[BLOCK_DIM];
	real chck[4];
	real pivinv;
	real ave;
	real freq;
	real Alamda;
	real Chisq;
	real Ochisq;
	real rchisq;
	real trial_chisq;
	real iter_diff, dev_old, dev_new;

	int Niter;
	int np, np1, np2;
	int isInvalid, isAlamda, isNiter;
	int icol;
	//double conw_r;

	int ipiv[MAX_N_PAR + 1];
	int indxc[MAX_N_PAR + 1];
	int indxr[MAX_N_PAR + 1];
	int sh_icol[BLOCK_DIM];
	int sh_irow[BLOCK_DIM];
} CUDA_LCC;

//struct freq_context
//typedef struct __attribute__((aligned(8))) freq_context
struct freq_context
{
	real Phi_0;
	real logCl;
	real cl;
	//double logC;
	real lambda_pole[N_POLES + 1];
	real beta_pole[N_POLES + 1];


	real par[4];
	real Alamda_start;
	real Alamda_incr;

	//double cgFirst[MAX_N_PAR + 1];
	real tim[MAX_N_OBS + 1];
	real ee[MAX_N_OBS + 1][3];	// double* ee;
	real ee0[MAX_N_OBS + 1][3];	// double* ee0;
	real Sig[MAX_N_OBS + 1];
	real Weight[MAX_N_OBS + 1];
	real Brightness[MAX_N_OBS + 1];
	real Fc[MAX_N_FAC + 1][MAX_LM + 1];
	real Fs[MAX_N_FAC + 1][MAX_LM + 1];
	real Darea[MAX_N_FAC + 1];
	real Nor[MAX_N_FAC + 1][3];
	real Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
	real Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
	real conw_r;

	int ia[MAX_N_PAR + 1];

	int Dg_block;
	int lastone;
	int lastma;
	int ma;
	int Mfit, Mfit1;
	int Mmax, Lmax;
	int n;
	int Ncoef, Ncoef0;
	int Numfac;
	int Numfac1;
	int Nphpar;
	int ndata;
	int Is_Precalc;

	/* runtime dimensions + per-context offsets (in doubles) into the
	   scratch buffer that replaced the fixed-size work arrays */
	int lcPoints1;
	int scrStride;
	int offAlpha;
	int offCovar;
	int offDytemp;
	int offYtemp;
	int offJpScale;
	int offJpDphp1;
	int offJpDphp2;
	int offJpDphp3;
	int offE1;
	int offE2;
	int offE3;
	int offE01;
	int offE02;
	int offE03;
	int offDe;
	int offDe0;
};

//struct freq_result
//struct __attribute__((aligned(8))) freq_result
struct freq_result
{
	real dark_best, per_best, dev_best, dev_best_x2, la_best, be_best, freq;
	int isReported, isInvalid, isNiter;
};
/* double-only helpers: the FP32 (df64) build has its own division in
   Real.cl and no double type */
#ifndef PS_FP32

/* WORKAROUND(rusticl / aco): runtime f64 '/' returns results with ~3*2^-29
   relative error (verified by [DIVTEST]); fma() and '*' are exact.
   Markstein sequence: two Newton steps refine the reciprocal, the final
   fused correction restores correct rounding.
   NATIVE_DIV_OK=1 (set by the host after the startup probe) replaces the
   helper with plain '/' on drivers whose division is correctly rounded;
   both paths then produce identical bits, so determinism is preserved. */
#ifndef NATIVE_DIV_OK
#define NATIVE_DIV_OK 0
#endif
#if NATIVE_DIV_OK
#define ddiv(a, b) ((a) / (b))
#else
inline double ddiv(double a, double b)
{
    double r = 1.0 / b;
    double e = fma(-b, r, 1.0);
    r = fma(r, e, r);
    e = fma(-b, r, 1.0);
    r = fma(r, e, r);
    double q = a * r;
    return fma(fma(-b, q, a), r, q);
}
#endif

/*
    FROM stackoverflow: https://stackoverflow.com/questions/42856717/intrinsics-equivalent-to-the-cuda-type-casting-intrinsics-double2loint-doub
    You can express these operations via a union. This will not create extra overhead with modern compilers as long as optimization is on (nvcc -O3 ...).
*/

//struct HiLo
//{
//    int lo;
//    int hi;
//};
//
//typedef struct HiLo hilo;
//
//union U {
//    double val;
//    hilo hiLo;
//};
//
//double HiLoint2double(int hi, int lo)
//{
//    union U u;
//
//    u.hiLo.hi = hi;
//    u.hiLo.lo = lo;
//
//    return u.val;
//}

typedef union {
    double val;
    struct {
        int lo;
        int hi;
    };
} un;

double HiLoint2double(int hi, int lo)
{
    /*union {
        double val;
        struct {
            int lo;
            int hi;
        };
    } u;*/
    un u;

    u.hi = hi;
    u.lo = lo;
    return u.val;
}


int double2hiint(double val)
{
    un u;
    u.val = val;
    return u.hi;
}

int double2loint(double val)
{
    un u;
    u.val = val;
    return u.lo;
}

//int __double2hiint(double val)
//{
//    union {
//        double val;
//        struct {
//            int lo;
//            int hi;
//        };
//    } u;
//    u.val = val;
//
//    return u.hi;
//}
//
//int __double2loint(double val)
//{
//    union {
//        double val;
//        struct {
//            int lo;
//            int hi;
//        };
//    } u;
//    u.val = val;
//
//    return u.lo;
//}
//
//int2 __double2int2(double val) {
//    int2 result;
//
//    result.x = __double2hiint(val);
//    result.y = __double2loint(val);
//
//    return result;
//}

#endif /* !PS_FP32 */
void SwapDouble(real a, real b) 
{ 
	real temp = a; 
	a = b; 
	b = temp; 
} //beta, lambda rotation matrix and its derivatives

 //  8.11.2006


//#include <math.h>
//#include "globals_CUDA.h"

void blmatrix(__global struct mfreq_context* CUDA_LCC, real bet, real lam)
{
	real cb, sb, cl, sl;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	sb = R_SINCOS(bet, &cb);
  	sl = R_SINCOS(lam, &cl);
	(*CUDA_LCC).Blmat[1][1] = R_MUL(cb, cl);
	(*CUDA_LCC).Blmat[1][2] = R_MUL(cb, sl);
	(*CUDA_LCC).Blmat[1][3] = R_NEG(sb);
	(*CUDA_LCC).Blmat[2][1] = R_NEG(sl);
	(*CUDA_LCC).Blmat[2][2] = cl;
	(*CUDA_LCC).Blmat[2][3] = R_C(0.0);
	(*CUDA_LCC).Blmat[3][1] = R_MUL(sb, cl);
	(*CUDA_LCC).Blmat[3][2] = R_MUL(sb, sl);
	(*CUDA_LCC).Blmat[3][3] = cb;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("bet: %10.7f, lam: %10.7f\n", bet, lam);
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	/* Ders. of Blmat w.r.t. bet */
	(*CUDA_LCC).Dblm[1][1][1] = R_MUL(R_NEG(sb), cl);
	(*CUDA_LCC).Dblm[1][1][2] = R_MUL(R_NEG(sb), sl);
	(*CUDA_LCC).Dblm[1][1][3] = R_NEG(cb);
	(*CUDA_LCC).Dblm[1][2][1] = R_C(0.0);
	(*CUDA_LCC).Dblm[1][2][2] = R_C(0.0);
	(*CUDA_LCC).Dblm[1][2][3] = R_C(0.0);
	(*CUDA_LCC).Dblm[1][3][1] = R_MUL(cb, cl);
	(*CUDA_LCC).Dblm[1][3][2] = R_MUL(cb, sl);
	(*CUDA_LCC).Dblm[1][3][3] = R_NEG(sb);
	/* Ders. w.r.t. lam */
	(*CUDA_LCC).Dblm[2][1][1] = R_MUL(R_NEG(cb), sl);
	(*CUDA_LCC).Dblm[2][1][2] = R_MUL(cb, cl);
	(*CUDA_LCC).Dblm[2][1][3] = R_C(0.0);
	(*CUDA_LCC).Dblm[2][2][1] = R_NEG(cl);
	(*CUDA_LCC).Dblm[2][2][2] = R_NEG(sl);
	(*CUDA_LCC).Dblm[2][2][3] = R_C(0.0);
	(*CUDA_LCC).Dblm[2][3][1] = R_MUL(R_NEG(sb), sl);
	(*CUDA_LCC).Dblm[2][3][2] = R_MUL(sb, cl);
	(*CUDA_LCC).Dblm[2][3][3] = R_C(0.0);
}
 //Curvature function (and hence facet area) from Laplace series

 //  8.11.2006


void curv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* cg,
	int brtmpl,
	int brtmph)
{
	int n;
	real fsum, g;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	//        brtmpl:  1, 4, 7... 382
	//		  brtmph:  3, 6, 9... 288
	int q = 0;
	for (int i = brtmpl; i <= brtmph; i++, q++)
	{
		//if (blockIdx.x == 0)
		//	printf("i: %d\n", i);

		g = R_C(0.0);
		n = 0;
		for (int m = 0; m <= (*CUDA_CC).Mmax; m++) // Mmax = 6
		{
			for (int l = m; l <= (*CUDA_CC).Lmax; l++)  // Lmax = 6
			{
				n++;
				//if (blockIdx.x == 0 && threadIdx.x == 0)
				//	printf("cg[%3d]: %10.7f\n", n, cg[n]);

				fsum = R_MUL(cg[n], (*CUDA_CC).Fc[i][m]);
				if (m != 0)
				{
					n++;
					//if (blockIdx.x == 0 && threadIdx.x == 0)
					//	printf("cg[%3d]: %10.7f\n", n, cg[n]);

					fsum = R_ADDM(fsum, cg[n], (*CUDA_CC).Fs[i][m]);
				}

				g = R_ADDM(g, (*CUDA_CC).Pleg[i][l][m], fsum);
			}
		}

		g = R_EXP(g);
		(*CUDA_LCC).Area[i] = R_MUL((*CUDA_CC).Darea[i], g);

		//if (blockIdx.x == 0)
		//	printf("[%3d - %3d] i: %3d\n", q, threadIdx.x, i);

		//if (blockIdx.x == 0)
		//	printf("Area[%d]: %.7f\n", i, Area[i]);

		/* Dg is no longer materialized: Dg[i][k] == g * Dsph[i][k] folds into the
		   facet weights through Area (= Darea * g) - see bright.cl and conv.cl */
	}
}

/* main-triangle elements per work-item: (DYT_STRIDE-1) rows at most */
#define C2_EPT (((DYT_STRIDE - 1) * DYT_STRIDE / 2 + BLOCK_DIM - 1) / BLOCK_DIM)

void mrqcof_curve2(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* alpha,
	__global real* beta,
	__local real (*dydaT)[DYT_STRIDE],
	__local real* s2wS,
	__local real* dwsS,
	__local real* dyS,
	__local real* coefS,		/* 2 * CURVE2_K: per-point coef, coef1 */
	int inrel,
	int lpoints,
	__global real* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global real* dytempG = scr + (*CUDA_CC).offDytemp;
	__global real* ytempG = scr + (*CUDA_CC).offYtemp;
	int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
	real dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);


	//precalc thread boundaries
	int tmph, tmpl;
	tmph = lpoints / BLOCK_DIM;
	if (lpoints % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	lnp1 = (*CUDA_LCC).np1 + tmpl;
	tmph = tmpl + tmph;
	if (tmph > lpoints) tmph = lpoints;
	tmpl++;

	int matmph, matmpl;									// threadIdx.x == 1
	matmph = (*CUDA_CC).ma / BLOCK_DIM;					// 0
	if ((*CUDA_CC).ma % BLOCK_DIM) matmph++;			// 1
	matmpl = threadIdx.x * matmph;						// 1
	matmph = matmpl + matmph;							// 2
	if (matmph > (*CUDA_CC).ma) matmph = (*CUDA_CC).ma;
	matmpl++;											// 2

	int latmph, latmpl;
	latmph = (*CUDA_CC).lastone / BLOCK_DIM;
	if ((*CUDA_CC).lastone % BLOCK_DIM) latmph++;
	latmpl = threadIdx.x * latmph;
	latmph = latmpl + latmph;
	if (latmph > (*CUDA_CC).lastone) latmph = (*CUDA_CC).lastone;
	latmpl++;

	/* The relative-lightcurve renormalization (ytemp *= coef, dytemp column 1
	   zeroed, dytemp[l] = coef * (dytemp[l] - coef1 * dave[l]) for l >= 2) used
	   to be a separate in-place pass over the whole curve in global memory
	   before the tiles re-read it. It is now applied while each tile is staged
	   into local memory - same expressions, same values - which saves a full
	   read + write of dytemp per curve (port of the HIP tree's fused I1
	   renormalization). dytemp/ytemp are per-curve scratch: nothing reads the
	   renormalized global copies afterwards. */
	const int lnp1b = (*CUDA_LCC).np1;	/* point jp uses Sig[lnp1b + jp] */
	const real ave = (*CUDA_LCC).ave;
	__local real* coef1S = coefS + CURVE2_K;

	/* everyone has read np1 before work-item 0 advances it */
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); 	//__syncthreads();

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np1 += lpoints;
	}

	lnp2 = (*CUDA_LCC).np2;
	ltrial_chisq = (*CUDA_LCC).trial_chisq;

	/* 2026 rewrite: the normal equations are accumulated once per
	   CURVE2_K-point tile (a rank-K update from a local-memory-staged dyda
	   tile) instead of once per data point. The old code swept the whole
	   triangular alpha matrix in global memory with a read-modify-write per
	   point, plus TWO work-group barriers per matrix row per point; those
	   barriers protected nothing (the staged derivatives are read-only during
	   the sweep and every alpha/beta slot has exactly one writer), so the
	   tile needs just two barriers total. Both original index variants -
	   absolute (ia[1]!=0) and relative (ia[1]==0, column shift m-1, frozen
	   first parameter, gated tail rows) - are reproduced element for element;
	   within a tile only the summation order over the K points changes.

	   dydaT[p][l] is point jp0+p's staged derivative row (renormalization
	   applied while staging for a relative curve), 1-based parameter l. */
	int jp0, p, P;
	real wp[CURVE2_K];

	/* Main triangle (rows l = l0..lastone, columns m = l0..l) flattened over
	   the work-group: work-item t owns elements t, t + BLOCK_DIM, ... for the
	   whole curve and keeps their running alpha values in registers, adding
	   each tile's contribution exactly as the in-memory
	   alpha = alpha + acc update did (same values, same order), and writes
	   them back once at the end. Before, rows were swept one at a time with
	   only l of BLOCK_DIM work-items busy and every tile did a global
	   read-modify-write of the whole triangle. The two original index
	   variants, element for element:
	     absolute (ia[1] != 0): l0 = 1, alpha[l][m], beta[l]
	     relative (ia[1] == 0): l0 = 2, alpha[l-1][m-1], beta[l-1]
	   (frozen size scale). The ia-gated tail rows keep their code. */
	const int rel = !(*CUDA_CC).ia[1];
	const int l0 = rel ? 2 : 1;
	const int nrows = (*CUDA_CC).lastone - l0 + 1;
	const int ntri = nrows > 0 ? nrows * (nrows + 1) / 2 : 0;
	const int Mfit1 = (*CUDA_CC).Mfit1;
	int triLM[C2_EPT];		/* l * 64 + m of each owned element */
	real triA[C2_EPT];	/* its running alpha value */
	#pragma unroll
	for (int t = 0; t < C2_EPT; t++)
	{
		int e = threadIdx.x + t * BLOCK_DIM;
		triLM[t] = 0;
		triA[t] = R_C(0.0);
		if (e < ntri)
		{
			/* row r of the triangle holds r + 1 elements */
			int r = (int)((sqrt(8.0f * (float)e + 1.0f) - 1.0f) * 0.5f);
			while ((r + 1) * (r + 2) / 2 <= e) r++;
			while (r * (r + 1) / 2 > e) r--;
			int lr = l0 + r, mr = l0 + (e - r * (r + 1) / 2);
			triLM[t] = lr * 64 + mr;
			triA[t] = alpha[(lr - rel) * Mfit1 + (mr - rel)];
		}
	}
	const int browL = l0 + threadIdx.x;	/* beta row owned by this work-item */
	const int ownB = browL <= (*CUDA_CC).lastone;
	real betaR = ownB ? beta[browL - rel] : R_C(0.0);

	for (jp0 = 1; jp0 <= lpoints; jp0 += CURVE2_K)
	{
		P = lpoints - jp0 + 1;
		if (P > CURVE2_K) P = CURVE2_K;

		/* per-point scalars; for a relative curve also the renormalization
		   factors, with the expressions of the former in-place pass */
		if (threadIdx.x < P)
		{
			jp = jp0 + threadIdx.x;
			ymod = ytempG[jp];
			if (inrel)
			{
				coef = R_DIV(R_MUL((*CUDA_CC).Sig[lnp1b + jp], R_FROM_INT(lpoints)), ave);
				coef1 = R_DIV(ymod, ave);
				coefS[threadIdx.x] = coef;
				coef1S[threadIdx.x] = coef1;
				ymod = R_MUL(coef, ymod);
			}
			sig2i = R_DIV(R_C(1.0), R_MUL((*CUDA_CC).Sig[lnp2 + jp], (*CUDA_CC).Sig[lnp2 + jp]));
			wght = (*CUDA_CC).Weight[lnp2 + jp];
			dy = R_SUB((*CUDA_CC).Brightness[lnp2 + jp], ymod);
			real sig2iwght = R_MUL(sig2i, wght);
			s2wS[threadIdx.x] = sig2iwght;
			dwsS[threadIdx.x] = R_MUL(dy, sig2iwght);
			dyS[threadIdx.x] = dy;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		/* stage the tile (consecutive work-items copy consecutive addresses),
		   renormalizing on the way for a relative curve */
		for (m = threadIdx.x; m < P * DYT_STRIDE; m += BLOCK_DIM)
		{
			real v = dytempG[(jp0 - 1) * DYT_STRIDE + m];
			if (inrel)
			{
				p = m / DYT_STRIDE;
				l = m % DYT_STRIDE;
				if (l == 1)
					v = R_C(0.0);	/* size-scale derivative is explicitly zero */
				else if (l >= 2 && l <= (*CUDA_CC).ma)
					v = R_MUL(coefS[p], R_SUBM(v, coef1S[p], (*CUDA_LCC).dave[l]));
			}
			((__local real*)&dydaT[0][0])[m] = v;
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		/* main triangle: register-resident elements */
		#pragma unroll
		for (int t = 0; t < C2_EPT; t++)
		{
			if (threadIdx.x + t * BLOCK_DIM < ntri)
			{
				int lr = triLM[t] / 64, mr = triLM[t] % 64;
				real acc = R_C(0.0);
				for (int pp = 0; pp < P; pp++)
				{
					real w = R_MUL(dydaT[pp][lr], s2wS[pp]);
					R_ADDTOM(acc, w, dydaT[pp][mr]);
				}
				triA[t] = R_ADD(triA[t], acc);
			}
		}
		if (ownB)
		{
			real bacc = R_C(0.0);
			for (int pp = 0; pp < P; pp++)
				R_ADDTOM(bacc, dwsS[pp], dydaT[pp][browL]);
			betaR = R_ADD(betaR, bacc);
		}

		/* ia-gated tail rows l = lastone+1..lastma: unchanged */
		j = nrows > 0 ? nrows : 0;
		l = (*CUDA_CC).lastone + 1;
		if (l < l0) l = l0;
		for (; l <= (*CUDA_CC).lastma; l++)
		{
			if ((*CUDA_CC).ia[l])
			{
				j++;
				for (p = 0; p < P; p++)
					wp[p] = R_MUL(dydaT[p][l], s2wS[p]);

				tmpl = latmpl;
				if (rel && tmpl == 1) tmpl++;	//m==1
				for (m = tmpl; m <= latmph; m++)
				{
					real acc = R_C(0.0);
					for (p = 0; p < P; p++)
						R_ADDTOM(acc, wp[p], dydaT[p][m]);
					alpha[j * Mfit1 + m - rel] = R_ADD(alpha[j * Mfit1 + m - rel], acc);
				} /* m */
				if (threadIdx.x == 0)
				{
					k = (*CUDA_CC).lastone - rel;
					for (m = (*CUDA_CC).lastone + 1; m <= l; m++)
					{
						if ((*CUDA_CC).ia[m])
						{
							k++;
							real acc = R_C(0.0);
							for (p = 0; p < P; p++)
								R_ADDTOM(acc, wp[p], dydaT[p][m]);
							alpha[j * Mfit1 + k] = R_ADD(alpha[j * Mfit1 + k], acc);
						}
					} /* m */
					real bacc = R_C(0.0);
					for (p = 0; p < P; p++)
						R_ADDTOM(bacc, dwsS[p], dydaT[p][l]);
					beta[j] = R_ADD(beta[j], bacc);
				}
			}
		} /* l */

		/* chi-square: same per-point terms in the same ascending order */
		for (p = 0; p < P; p++)
		{
			ltrial_chisq = R_ADDM(ltrial_chisq, R_MUL(dyS[p], dyS[p]), s2wS[p]);
		}

		/* everyone must finish reading dydaT before the next tile overwrites it */
		barrier(CLK_LOCAL_MEM_FENCE);
	} /* jp0 */

	#pragma unroll
	for (int t = 0; t < C2_EPT; t++)
	{
		if (threadIdx.x + t * BLOCK_DIM < ntri)
		{
			int lr = triLM[t] / 64, mr = triLM[t] % 64;
			alpha[(lr - rel) * Mfit1 + (mr - rel)] = triA[t];
		}
	}
	if (ownB)
		beta[browL - rel] = betaR;

	lnp2 += lpoints;

	if (threadIdx.x == 0)
	{
		//printf("[%d] ltrial_chisq: %10.7f\n", blockIdx.x, ltrial_chisq);

		(*CUDA_LCC).np2 = lnp2;
		(*CUDA_LCC).trial_chisq = ltrial_chisq;
	}
}

//computes integrated brightness of all visible and iluminated areas
//  and its derivatives

//  8.11.2006


void matrix_neo(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* cg,
	int lnp1,
	int Lpoints,
	int num,
	__global real* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global real* jp_ScaleG = scr + (*CUDA_CC).offJpScale;
	__global real* jp_dphp_1G = scr + (*CUDA_CC).offJpDphp1;
	__global real* jp_dphp_2G = scr + (*CUDA_CC).offJpDphp2;
	__global real* jp_dphp_3G = scr + (*CUDA_CC).offJpDphp3;
	__global real* e_1G = scr + (*CUDA_CC).offE1;
	__global real* e_2G = scr + (*CUDA_CC).offE2;
	__global real* e_3G = scr + (*CUDA_CC).offE3;
	__global real* e0_1G = scr + (*CUDA_CC).offE01;
	__global real* e0_2G = scr + (*CUDA_CC).offE02;
	__global real* e0_3G = scr + (*CUDA_CC).offE03;
	__global real* deG = scr + (*CUDA_CC).offDe;
	__global real* de0G = scr + (*CUDA_CC).offDe0;
	__private real f, cf, sf, pom, pom0, alpha;
	__private real ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
	__private int lnp;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	lnp = lnp1 + brtmpl - 1;
	//printf("lnp: %3d = lnp1: %3d + brtmpl: %3d - 1 | lnp++: %3d\n", lnp, lnp1, brtmpl, lnp + 1);

	int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%3d] cg[%3d]: %10.7f\n", blockIdx.x,  q, (*CUDA_LCC).cg[q]);

	for (int jp = brtmpl; jp <= brtmph; jp++)
	{
		lnp++;

		ee_1 = (*CUDA_CC).ee[lnp][0];		// position vectors
		ee0_1 = (*CUDA_CC).ee0[lnp][0];
		ee_2 = (*CUDA_CC).ee[lnp][1];
		ee0_2 = (*CUDA_CC).ee0[lnp][1];
		ee_3 = (*CUDA_CC).ee[lnp][2];
		ee0_3 = (*CUDA_CC).ee0[lnp][2];
		t = (*CUDA_CC).tim[lnp];

		//if (blockIdx.x == 0)
		//	printf("jp[%3d] lnp[%3d], %10.7f, %10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",
		//		jp, lnp, ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3);

		//printf("tim[%3d]: %10.7f\n", lnp, t);
		//printf("lnp: %3d, ee[%d]: %.7f, ee0[%d]: %.7f\n", lnp, lnp * 3 + 0, (*CUDA_CC).ee[lnp][0], lnp, (*CUDA_CC).ee0[lnp][0]);

		alpha = R_ACOS(R_CLAMP(R_ADDM(R_MADD(ee_1, ee0_1, R_MUL(ee_2, ee0_2)), ee_3, ee0_3), R_C(-1.0), R_C(1.0)));


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] alpha[%3d]: %.7f, cg[%3d]: %10.7f\n", jp, alpha, q, (*CUDA_LCC).cg[q]);

		/* Exp-lin model (const.term=1.) */
		real f = R_EXP(R_NEG(R_DIV(alpha, cg[(*CUDA_CC).Ncoef0 + 2])));	//f is temp here

		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] [%2d][%3d] jp[%3d] f: %10.7f, cg[%3d] %10.7f, alpha %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, f, (*CUDA_CC).Ncoef0 + 2, cg[(*CUDA_CC).Ncoef0 + 2], alpha);

		jp_ScaleG[jp] = R_ADDM(R_ADDM(R_C(1.0), cg[(*CUDA_CC).Ncoef0 + 1], f), cg[(*CUDA_CC).Ncoef0 + 3], alpha);
		jp_dphp_1G[jp] = f;
		jp_dphp_2G[jp] = R_DIV(R_MUL(R_MUL(cg[(*CUDA_CC).Ncoef0 + 1], f), alpha), R_MUL(cg[(*CUDA_CC).Ncoef0 + 2], cg[(*CUDA_CC).Ncoef0 + 2]));
		jp_dphp_3G[jp] = alpha;

		//if (blockIdx.x == 0)
		//	printf("[neo] [%d][%3d] jp_Scale[%3d]: %10.7f, jp_dphp_1[]: %10.7F, jp_dphp_2[]: %10.7f, jp_dphp_3[]: %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, jp_ScaleG[jp], jp_dphp_1G[jp], jp_dphp_2G[jp], jp_dphp_3G[jp]);

		//  matrix start
		f = R_MADD(cg[(*CUDA_CC).Ncoef0], t, (*CUDA_CC).Phi_0);
		f = R_FMOD_2PI(f); /* may give little different results than Mikko's */
		sf = R_SINCOS(f, &cf);

		//if (threadIdx.x == 0)
		//	printf("jp[%3d] [%3d] cf: %10.7f, sf: %10.7f\n", jp, blockIdx.x, cf, sf);

		//if (num == 1 && blockIdx.x == 0 && jp == brtmpl)
		//{
		//	printf("[%2d][%3d][%3d] f: % .6f, cosF: % .6f, sinF: % .6f\n", blockIdx.x, threadIdx.x, jp, f, cf, sf);
		//}

		//	/* rotation matrix, Z axis, angle f */

		tmat = R_MADD(cf, (*CUDA_LCC).Blmat[1][1], R_MUL(sf, (*CUDA_LCC).Blmat[2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(cf, (*CUDA_LCC).Blmat[1][2], R_MUL(sf, (*CUDA_LCC).Blmat[2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(cf, (*CUDA_LCC).Blmat[1][3], R_MUL(sf, (*CUDA_LCC).Blmat[2][3]));
		e_1G[jp] = R_ADDM(pom, tmat, ee_3);
		e0_1G[jp] = R_ADDM(pom0, tmat, ee0_3);

		//if (blockIdx.x == 0)
		//	printf("[%3d] jp[%3d] %10.7f, %10.7f\n", threadIdx.x, jp, e_1G[jp], e0_1G[jp]);

		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Blmat[1][1], R_MUL(cf, (*CUDA_LCC).Blmat[2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Blmat[1][2], R_MUL(cf, (*CUDA_LCC).Blmat[2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Blmat[1][3], R_MUL(cf, (*CUDA_LCC).Blmat[2][3]));
		e_2G[jp] = R_ADDM(pom, tmat, ee_3);
		e0_2G[jp] = R_ADDM(pom0, tmat, ee0_3);

		tmat = (*CUDA_LCC).Blmat[3][1];
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = (*CUDA_LCC).Blmat[3][2];
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = (*CUDA_LCC).Blmat[3][3];
		e_3G[jp] = R_ADDM(pom, tmat, ee_3);
		e0_3G[jp] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[1][1][1], R_MUL(sf, (*CUDA_LCC).Dblm[1][2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[1][1][2], R_MUL(sf, (*CUDA_LCC).Dblm[1][2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[1][1][3], R_MUL(sf, (*CUDA_LCC).Dblm[1][2][3]));
		deG[(jp) * 16 + (1) * 4 + (1)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (1) * 4 + (1)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[2][1][1], R_MUL(sf, (*CUDA_LCC).Dblm[2][2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[2][1][2], R_MUL(sf, (*CUDA_LCC).Dblm[2][2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[2][1][3], R_MUL(sf, (*CUDA_LCC).Dblm[2][2][3]));
		deG[(jp) * 16 + (1) * 4 + (2)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (1) * 4 + (2)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[1][1], R_MUL(R_MUL(t, cf), (*CUDA_LCC).Blmat[2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[1][2], R_MUL(R_MUL(t, cf), (*CUDA_LCC).Blmat[2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[1][3], R_MUL(R_MUL(t, cf), (*CUDA_LCC).Blmat[2][3]));
		deG[(jp) * 16 + (1) * 4 + (3)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (1) * 4 + (3)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[1][1][1], R_MUL(cf, (*CUDA_LCC).Dblm[1][2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[1][1][2], R_MUL(cf, (*CUDA_LCC).Dblm[1][2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[1][1][3], R_MUL(cf, (*CUDA_LCC).Dblm[1][2][3]));
		deG[(jp) * 16 + (2) * 4 + (1)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (2) * 4 + (1)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[2][1][1], R_MUL(cf, (*CUDA_LCC).Dblm[2][2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[2][1][2], R_MUL(cf, (*CUDA_LCC).Dblm[2][2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[2][1][3], R_MUL(cf, (*CUDA_LCC).Dblm[2][2][3]));
		deG[(jp) * 16 + (2) * 4 + (2)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (2) * 4 + (2)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(R_MUL(R_NEG(t), cf), (*CUDA_LCC).Blmat[1][1], R_MUL(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_MUL(R_NEG(t), cf), (*CUDA_LCC).Blmat[1][2], R_MUL(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_MUL(R_NEG(t), cf), (*CUDA_LCC).Blmat[1][3], R_MUL(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[2][3]));
		deG[(jp) * 16 + (2) * 4 + (3)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (2) * 4 + (3)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = (*CUDA_LCC).Dblm[1][3][1];
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = (*CUDA_LCC).Dblm[1][3][2];
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = (*CUDA_LCC).Dblm[1][3][3];
		deG[(jp) * 16 + (3) * 4 + (1)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (3) * 4 + (1)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = (*CUDA_LCC).Dblm[2][3][1];
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = (*CUDA_LCC).Dblm[2][3][2];
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = (*CUDA_LCC).Dblm[2][3][3];
		deG[(jp) * 16 + (3) * 4 + (2)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (3) * 4 + (2)] = R_ADDM(pom0, tmat, ee0_3);


		deG[(jp) * 16 + (3) * 4 + (3)] = R_C(0.0);
		de0G[(jp) * 16 + (3) * 4 + (3)] = R_C(0.0);
	}
}

void bright(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* cg,
	int jp,
	int Lpoints1,
	int Inrel,
	__global real* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global real* dytempG = scr + (*CUDA_CC).offDytemp;
	__global real* ytempG = scr + (*CUDA_CC).offYtemp;
	__global real* jp_ScaleG = scr + (*CUDA_CC).offJpScale;
	__global real* jp_dphp_1G = scr + (*CUDA_CC).offJpDphp1;
	__global real* jp_dphp_2G = scr + (*CUDA_CC).offJpDphp2;
	__global real* jp_dphp_3G = scr + (*CUDA_CC).offJpDphp3;
	__global real* e_1G = scr + (*CUDA_CC).offE1;
	__global real* e_2G = scr + (*CUDA_CC).offE2;
	__global real* e_3G = scr + (*CUDA_CC).offE3;
	__global real* e0_1G = scr + (*CUDA_CC).offE01;
	__global real* e0_2G = scr + (*CUDA_CC).offE02;
	__global real* e0_3G = scr + (*CUDA_CC).offE03;
	__global real* deG = scr + (*CUDA_CC).offDe;
	__global real* de0G = scr + (*CUDA_CC).offDe0;
	real cl, cls, dnom, s, Scale;
	real e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];
	int ncoef0, ncoef, i, j, incl_count = 0;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	ncoef0 = (*CUDA_CC).Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = (*CUDA_CC).ma;
	cl = R_EXP(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

	/* matrix from neo */
	/* derivatives */
	e_1 = e_1G[jp];
	e_2 = e_2G[jp];
	e_3 = e_3G[jp];
	e0_1 = e0_1G[jp];
	e0_2 = e0_2G[jp];
	e0_3 = e0_3G[jp];
	de[1][1] = deG[(jp) * 16 + (1) * 4 + (1)];
	de[1][2] = deG[(jp) * 16 + (1) * 4 + (2)];
	de[1][3] = deG[(jp) * 16 + (1) * 4 + (3)];
	de[2][1] = deG[(jp) * 16 + (2) * 4 + (1)];
	de[2][2] = deG[(jp) * 16 + (2) * 4 + (2)];
	de[2][3] = deG[(jp) * 16 + (2) * 4 + (3)];
	de[3][1] = deG[(jp) * 16 + (3) * 4 + (1)];
	de[3][2] = deG[(jp) * 16 + (3) * 4 + (2)];
	de[3][3] = deG[(jp) * 16 + (3) * 4 + (3)];
	de0[1][1] = de0G[(jp) * 16 + (1) * 4 + (1)];
	de0[1][2] = de0G[(jp) * 16 + (1) * 4 + (2)];
	de0[1][3] = de0G[(jp) * 16 + (1) * 4 + (3)];
	de0[2][1] = de0G[(jp) * 16 + (2) * 4 + (1)];
	de0[2][2] = de0G[(jp) * 16 + (2) * 4 + (2)];
	de0[2][3] = de0G[(jp) * 16 + (2) * 4 + (3)];
	de0[3][1] = de0G[(jp) * 16 + (3) * 4 + (1)];
	de0[3][2] = de0G[(jp) * 16 + (3) * 4 + (2)];
	de0[3][3] = de0G[(jp) * 16 + (3) * 4 + (3)];

	/*Integrated brightness (phase coeff. used later) */
	real lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	real br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	short int incl[MAX_N_FAC];
	real dbr[MAX_N_FAC];

	br = R_C(0.0);
	tmp1 = R_C(0.0);
	tmp2 = R_C(0.0);
	tmp3 = R_C(0.0);
	tmp4 = R_C(0.0);
	tmp5 = R_C(0.0);

	/* Two passes: the cheap visibility test first builds this work-item's
	   list of visible facets, then the division-heavy terms run over that
	   list. In a single pass a wavefront executed the heavy block for every
	   facet that ANY of its lanes could see, i.e. for nearly all facets;
	   now it runs max(incl_count) times per wavefront. lmu/lmu0 are
	   recomputed with the same expressions and the sums still run over the
	   visible facets in ascending order. */
	for (i = 1; i <= (*CUDA_CC).Numfac; i++)
	{
		lmu = R_ADDM(R_MADD(e_1, (*CUDA_CC).Nor[i][0], R_MUL(e_2, (*CUDA_CC).Nor[i][1])), e_3, (*CUDA_CC).Nor[i][2]);
		lmu0 = R_ADDM(R_MADD(e0_1, (*CUDA_CC).Nor[i][0], R_MUL(e0_2, (*CUDA_CC).Nor[i][1])), e0_3, (*CUDA_CC).Nor[i][2]);
		if (R_GT(lmu, R_TINY) && R_GT(lmu0, R_TINY))
		{
			incl[incl_count] = i;
			incl_count++;
		}
	}

	for (int c = 0; c < incl_count; c++)
	{
		i = incl[c];
		j = i;
		lmu = R_ADDM(R_MADD(e_1, (*CUDA_CC).Nor[i][0], R_MUL(e_2, (*CUDA_CC).Nor[i][1])), e_3, (*CUDA_CC).Nor[i][2]);
		lmu0 = R_ADDM(R_MADD(e0_1, (*CUDA_CC).Nor[i][0], R_MUL(e0_2, (*CUDA_CC).Nor[i][1])), e0_3, (*CUDA_CC).Nor[i][2]);
		{
			dnom = R_ADD(lmu, lmu0);
			s = R_MUL(R_MUL(lmu, lmu0), R_ADD(cl, R_DIV(cls, dnom)));
			ar = (*CUDA_LCC).Area[j];
			R_ADDTOM(br, ar, s);

			/* Darea[i] * s * Dg[i][k] == Darea[i] * s * g * Dsph[i][k]
			   == (Area[i] * s) * Dsph[i][k]: fold g into the weight and
			   gather from the one read-only, facet-major Dsph shared by
			   all work-groups instead of the per-context Dg matrix */
			dbr[c] = R_MUL(ar, s);

			real lmu0_dnom = R_DIV(lmu0, dnom);
			dsmu = R_MADD(cls, R_MUL(lmu0_dnom, lmu0_dnom), R_MUL(cl, lmu0));
			real lmu_dnom = R_DIV(lmu, dnom);
			dsmu0 = R_MADD(cls, R_MUL(lmu_dnom, lmu_dnom), R_MUL(cl, lmu));


			sum1 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de[1][1], R_MUL((*CUDA_CC).Nor[i][1], de[2][1])), (*CUDA_CC).Nor[i][2], de[3][1]);
			sum10 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de0[1][1], R_MUL((*CUDA_CC).Nor[i][1], de0[2][1])), (*CUDA_CC).Nor[i][2], de0[3][1]);
			R_ADDTOM(tmp1, ar, R_MADD(dsmu, sum1, R_MUL(dsmu0, sum10)));
			sum2 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de[1][2], R_MUL((*CUDA_CC).Nor[i][1], de[2][2])), (*CUDA_CC).Nor[i][2], de[3][2]);
			sum20 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de0[1][2], R_MUL((*CUDA_CC).Nor[i][1], de0[2][2])), (*CUDA_CC).Nor[i][2], de0[3][2]);
			R_ADDTOM(tmp2, ar, R_MADD(dsmu, sum2, R_MUL(dsmu0, sum20)));
			sum3 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de[1][3], R_MUL((*CUDA_CC).Nor[i][1], de[2][3])), (*CUDA_CC).Nor[i][2], de[3][3]);
			sum30 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de0[1][3], R_MUL((*CUDA_CC).Nor[i][1], de0[2][3])), (*CUDA_CC).Nor[i][2], de0[3][3]);
			R_ADDTOM(tmp3, ar, R_MADD(dsmu, sum3, R_MUL(dsmu0, sum30)));

			R_ADDTOM(tmp4, R_MUL(lmu, lmu0), ar);
			R_ADDTOM(tmp5, ar, R_DIV(R_MUL(lmu, lmu0), R_ADD(lmu, lmu0)));
		}
	}

	Scale = jp_ScaleG[jp];
	i = (jp - 1) * DYT_STRIDE + (ncoef0 - 3 + 1);
	/* Ders. of brightness w.r.t. rotation parameters */
	dytempG[i] = R_MUL(Scale, tmp1);

	i++;
	dytempG[i] = R_MUL(Scale, tmp2);
	i++;
	dytempG[i] = R_MUL(Scale, tmp3);

	i++;
	/* Ders. of br. w.r.t. phase function params. */
	dytempG[i] = R_MUL(br, jp_dphp_1G[jp]);
	i++;
	dytempG[i] = R_MUL(br, jp_dphp_2G[jp]);
	i++;
	dytempG[i] = R_MUL(br, jp_dphp_3G[jp]);

	/* Ders. of br. w.r.t. cl, cls */
	dytempG[(jp - 1) * DYT_STRIDE + (ncoef - 1)] = R_MUL(R_MUL(Scale, tmp4), cl);
	dytempG[(jp - 1) * DYT_STRIDE + (ncoef)] = R_MUL(Scale, tmp5);

	/* Scaled brightness */
	ytempG[jp] = R_MUL(br, Scale);

	ncoef0 -= 3;
	int iStart;
	int d;

	iStart = Inrel + 1;
	d = (jp - 1) * DYT_STRIDE + iStart;


	/* Derivatives of brightness w.r.t. g-coeffs: BRIGHT_GB columns per pass
	   over the visible-facet list (was 2); each column is still
	   dbr[0] * Dsph[..] followed by the fma chain over the visible facets
	   in ascending order. Up to BRIGHT_GB - 1 columns past ncoef0 are read
	   (inside the Dsph row) but not stored. */
#define BRIGHT_GB 16
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += BRIGHT_GB)
		{
			real t[BRIGHT_GB];
			{
				real l_dbr = dbr[0];
				__global real* row = (*CUDA_CC).Dsph[incl[0]] + i;
				for (int b = 0; b < BRIGHT_GB; b++)
					t[b] = R_MUL(l_dbr, row[b]);
			}

			for (j = 1; j < incl_count; j++)
			{
				real l_dbr = dbr[j];
				__global real* row = (*CUDA_CC).Dsph[incl[j]] + i;
				for (int b = 0; b < BRIGHT_GB; b++)
					R_ADDTOM(t[b], l_dbr, row[b]);
			}

			for (int b = 0; b < BRIGHT_GB; b++)
			{
				if (i + b <= ncoef0)
					dytempG[(jp - 1) * DYT_STRIDE + i + b] = R_MUL(Scale, t[b]);
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d++)
			dytempG[d] = R_C(0.0);
	}

	//return(0);
}
//Convexity regularization function

//  8.11.2006


real conv(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local real* res,
	int nc,
	int brtmpl,
	int brtmph)
{
	int i, j, k;
	real tmp = R_C(0.0);
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//j = blockIdx.x * (CUDA_Numfac1)+brtmpl;
	j = brtmpl;
	for (i = brtmpl; i <= brtmph; i++, j++)
	{
		//tmp += CUDA_Area[j] * CUDA_Nor[i][nc];
		R_ADDTOM(tmp, (*CUDA_LCC).Area[j], (*CUDA_CC).Nor[i][nc]);
	}

	res[threadIdx.x] = tmp;

	//if (threadIdx.x == 0)
	//    printf("conv>>> [%d] jp-1[%3d] res[%3d]: %10.7f\n", blockIdx.x, nc, threadIdx.x, res[threadIdx.x]);

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	//parallel reduction
	k = BLOCK_DIM >> 1;
	while (k > 1)
	{
		if (threadIdx.x < k)
			R_ADDTO(res[threadIdx.x], res[threadIdx.x + k]);
		k = k >> 1;
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		tmp = R_ADD(res[0], res[1]);
	}

	/* the derivatives w.r.t. the shape coefficients are computed for all
	   points at once in mrqcof_curve1_last */

	return (tmp);
}
 //slighly changed code from Numerical Recipes
 //  converted from Mikko's fortran code

 //  8.11.2006


//#include <stdio.h>
//#include <stdlib.h>
//#include "globals_CUDA.h"
//#include "declarations_CUDA.h"


/* comment the following line if no YORP */
/*#define YORP*/

void mrqcof_start(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* cg,
	__global real* alpha,
	__global real* beta)
{
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);
	int x = threadIdx.x;

	int brtmph, brtmpl;
	// brtmph = 288 / 128 = 2 (2.25)
	brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
	if ((*CUDA_CC).Numfac % BLOCK_DIM)
	{
		brtmph++; // brtmph = 3
	}

	brtmpl = threadIdx.x * brtmph;	// 0 * 3 = 0, 1 * 3 = 3, 6,  9, 12, 15, 18... 381(127 * 3)
	brtmph = brtmpl + brtmph;		//		   3,         6, 9, 12, 15, 18, 21... 384(381 + 3)
	if (brtmph > (*CUDA_CC).Numfac) //  97 * 3 = 201 > 288
	{
		brtmph = (*CUDA_CC).Numfac; // 3, 6, ... max 288
	}

	brtmpl++; // 1..382
	//if(blockIdx.x == 0)
	//	printf("Idx: %d | Numfac: %d | brtmpl: %d | brtmph: %d\n", threadIdx.x, (*CUDA_CC).Numfac, brtmpl, brtmph);

		/*  ---   CURV  ---  */
	curv(CUDA_LCC, CUDA_CC, cg, brtmpl, brtmph);

	if (threadIdx.x == 0)
	{
		//   #ifdef YORP
		//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
		  // #else

		//if (blockIdx.x == 0)
		//	printf("[mrqcof_start] a[%3d]: %10.7f, a[%3d]: %10.7f\n",
		//		(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar, cg[(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar],
		//		(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar, cg[(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar]);

		  /*  ---  BLMATRIX ---  */
		blmatrix(CUDA_LCC, cg[(*CUDA_CC).ma - 4 - (*CUDA_CC).Nphpar], cg[(*CUDA_CC).ma - 3 - (*CUDA_CC).Nphpar]);
		//   #endif
		(*CUDA_LCC).trial_chisq = R_C(0.0);
		(*CUDA_LCC).np = 0;
		(*CUDA_LCC).np1 = 0;
		(*CUDA_LCC).np2 = 0;
		(*CUDA_LCC).ave = R_C(0.0);
	}

	brtmph = (*CUDA_CC).Mfit / BLOCK_DIM;
	if ((*CUDA_CC).Mfit % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Mfit) brtmph = (*CUDA_CC).Mfit;
	brtmpl++;

	__private int idx, k, j;

	for (j = brtmpl; j <= brtmph; j++)
	{
		for (k = 1; k <= j; k++)
		{
			idx = j * (*CUDA_CC).Mfit1 + k;
			alpha[idx] = R_C(0.0);
			//if (blockIdx.x == 0 && j < 3)
			//	printf("[%3d] j: %d, k: %d, Mfit1: %2d, alpha[%3d]: %.7f\n", threadIdx.x, j, k, (*CUDA_CC).Mfit1, idx, alpha[idx]);
		}
		beta[j] = R_C(0.0);
	}


	//int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%d][%3d] cg[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, q, (*CUDA_LCC).cg[q]);


}

void mrqcof_matrix(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* cg,
	int Lpoints,
	int num,
	__global real* scr)
{
	matrix_neo(CUDA_LCC, CUDA_CC, cg, (*CUDA_LCC).np, Lpoints, num, scr);
}

void mrqcof_curve1(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* cg,
	__local real* tmave,
	int Inrel,
	int Lpoints,
	int num,
	__global real* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global real* dytempG = scr + (*CUDA_CC).offDytemp;
	__global real* ytempG = scr + (*CUDA_CC).offYtemp;
	//__local double tmave[BLOCK_DIM];  // __shared__
	__private int Lpoints1 = Lpoints + 1;
	__private int k, lnp, jp;
	__private real lave;

	lnp = (*CUDA_LCC).np;
	lave = (*CUDA_LCC).ave;

	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//precalc thread boundaries
	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;

	/* points are dealt out round-robin (jp = t+1, t+1+BLOCK_DIM, ...) rather than
	   in contiguous blocks of ceil(Lpoints/BLOCK_DIM): every point is
	   independent, and this packs the last partial round into as few
	   wavefronts as possible (156 points: 5 wave32 rounds instead of 6). The
	   ytemp partial sums below keep the contiguous blocks, so ave is summed
	   in the same order as before. */
	for (jp = threadIdx.x + 1; jp <= Lpoints; jp += BLOCK_DIM)
	{
			/*  ---  BRIGHT  ---  */
		bright(CUDA_LCC, CUDA_CC, cg, jp, Lpoints1, Inrel, scr);
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	if (Inrel == 1)
	{
		int tmph, tmpl;
		tmph = (*CUDA_CC).ma / BLOCK_DIM;
		if ((*CUDA_CC).ma % BLOCK_DIM) tmph++;
		tmpl = threadIdx.x * tmph;
		tmph = tmpl + tmph;
		if (tmph > (*CUDA_CC).ma) tmph = (*CUDA_CC).ma;
		tmpl++;
		if (tmpl == 1) tmpl++;

		int ixx;
		for (int l = tmpl; l <= tmph; l++)
		{
			//jp==1
			ixx = l;
			(*CUDA_LCC).dave[l] = dytempG[ixx];

			//jp>=2
			ixx += DYT_STRIDE;
			for (int jp = 2; jp <= Lpoints; jp++, ixx += DYT_STRIDE)
			{
				//(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + dytempG[ixx];
				(*CUDA_LCC).dave[l] = R_ADD((*CUDA_LCC).dave[l], dytempG[ixx]);

				//if (threadIdx.x == 1)
				//	printf("[Device | mrqcof_curv1] [%3d] dytemp[%3d]: %10.7f, dave[%3d]: %10.7f\n", blockIdx.x, ixx, dytempG[ixx], l, (*CUDA_LCC).dave[l]);
			}
		}

		tmave[threadIdx.x] = R_C(0.0);
		for (int jp = brtmpl; jp <= brtmph; jp++)
		{
			R_ADDTO(tmave[threadIdx.x], ytempG[jp]);
		}

		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		//parallel reduction
		k = BLOCK_DIM >> 1;
		while (k > 1)
		{
			if (threadIdx.x < k) R_ADDTO(tmave[threadIdx.x], tmave[threadIdx.x + k]);
			k = k >> 1;
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
		}

		if (threadIdx.x == 0)
		{
			lave = R_ADD(tmave[0], tmave[1]);
		}
		//parallel reduction end
	}

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp + Lpoints;
		(*CUDA_LCC).ave = lave;
	}
}

void mrqcof_curve1_last(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* a,
	__global real* alpha,
	__global real* beta,
	__local real* res,
	int Inrel,
	int Lpoints,
	__global real* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global real* dytempG = scr + (*CUDA_CC).offDytemp;
	__global real* ytempG = scr + (*CUDA_CC).offYtemp;
	int l, jp, lnp;
	real ymod, lave;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	lnp = (*CUDA_LCC).np;
	//
	if (threadIdx.x == 0)
	{
		if (Inrel == 1) /* is the LC relative? */
		{
			lave = R_C(0.0);
			for (l = 1; l <= (*CUDA_CC).ma; l++)
				(*CUDA_LCC).dave[l] = R_C(0.0);
		}
		else
			lave = (*CUDA_LCC).ave;
	}
	//precalc thread boundaries
	int tmph, tmpl;
	tmph = (*CUDA_CC).ma / BLOCK_DIM;
	if ((*CUDA_CC).ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > (*CUDA_CC).ma) tmph = (*CUDA_CC).ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
	if ((*CUDA_CC).Numfac % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Numfac) brtmph = (*CUDA_CC).Numfac;
	brtmpl++;

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();
	//if (threadIdx.x == 0)
	//	printf("conv>>> [%d] \n", blockIdx.x);

	/* convexity derivatives dyda[l] = sum_i Area[i] * Dsph[i][l] * Nor[i][nc]
	   of every point (nc = jp-1) in ONE pass over the facets - it used to be
	   one pass per point inside conv(). Area/Dsph are read once for all
	   points and the per-point sums become independent chains; each sum keeps
	   its facet order and operand rounding, and dave[l] still accumulates the
	   points in ascending order (each l belongs to one work-item). */
	for (l = tmpl; l <= tmph; l++)
	{
		for (int jb = 0; jb < Lpoints; jb += 3)
		{
			real d[3] = { R_C(0.0), R_C(0.0), R_C(0.0) };
			if (l <= (*CUDA_CC).Ncoef)
			{
				for (int i = 1; i <= (*CUDA_CC).Numfac; i++)
				{
					/* Darea[i] * Dg[i][l] == Area[i] * Dsph[i][l] (Area = Darea*g) */
					real ad = R_MUL((*CUDA_LCC).Area[i], (*CUDA_CC).Dsph[i][l]);
					for (int q = 0; q < 3; q++)
						R_ADDTOM(d[q], ad, (*CUDA_CC).Nor[i][jb + q]);
				}
			}
			for (int q = 0; q < 3 && jb + q < Lpoints; q++)
			{
				dytempG[(jb + q) * DYT_STRIDE + l] = d[q];
				if (Inrel == 1)
					(*CUDA_LCC).dave[l] = R_ADD((*CUDA_LCC).dave[l], d[q]);
			}
		}
	}

	for (jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;
		// *--- CONV() ---* //
		ymod = conv(CUDA_LCC, CUDA_CC, res, jp - 1, brtmpl, brtmph);

		if (threadIdx.x == 0)
		{
			ytempG[jp] = ymod;

			if (Inrel == 1)
				lave = R_ADD(lave, ymod);
		}
		/* save lightcurves */
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

		/*         if ((*CUDA_LCC).Lastcall == 1) always ==0
					 (*CUDA_LCC).Yout[np] = ymod;*/
	} /* jp, lpoints */

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp;
		(*CUDA_LCC).ave = lave;
	}
}

real mrqcof_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* alpha)
{
	int j, k;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	/* mirror the lower triangle; each row is split over the work-group
	   (reads are contiguous, source and destination never overlap) */
	int lsize = get_local_size(0);
	for (int j = 2; j <= (*CUDA_CC).Mfit; j++)
	{
		for (k = 1 + threadIdx.x; k <= j - 1; k += lsize)
		{
			alpha[k * (*CUDA_CC).Mfit1 + j] = alpha[j * (*CUDA_CC).Mfit1 + k];
			//if (blockIdx.x ==0 && threadIdx.x == 0)
			//	printf("[mrqcof_end] [%d][%3d] alpha[%3d]: %10.7f\n", blockIdx.x, threadIdx.x, k * (*CUDA_CC).Mfit1 + j, alpha[k * (*CUDA_CC).Mfit1 + j]);
		}
	}

	return (*CUDA_LCC).trial_chisq;
}

//from Numerical Recipes

/* 2026: the damped normal matrix is staged into local memory and the whole
   Gauss-Jordan elimination runs there; global memory is only touched to read
   alpha/beta on entry and to write the step vector da at the end. The old
   version swept covar in global memory on every pivot step. Two consequences
   of the caller's structure are used:

   * the inverted matrix itself is dead - ClCalculateIter1Mrqcof2Start rezeroes
	 covar before mrqcof2 accumulates into it, and mrqmin_2_end copies that
	 fresh accumulation - so neither the solved matrix nor the final
	 column-unscramble pass (and its indxr/indxc bookkeeping) is needed;
	 only da and the return code leave this function;

   * the icol/pivinv broadcast scalars and the pivot-reduction arrays move
	 from per-context global struct members to local memory.

   The local buffers are declared at kernel scope (OpenCL requirement) in
   ClCalculateIter1Mrqmin1End and passed through mrqmin_1_end. Pivot choice
   and elimination order are unchanged, so the computed step is bit-identical
   to the global-memory version. */
int gauss_errc(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local real* covL,   /* [DYT_STRIDE * DYT_STRIDE], indexed with Mfit1 stride */
	__local real* daL,    /* [DYT_STRIDE] */
	__local int* ipivL,     /* [DYT_STRIDE] */
	__local real* shBig,  /* [BLOCK_DIM] */
	__local int* shIrow,    /* [BLOCK_DIM] */
	__local int* shIcol,    /* [BLOCK_DIM] */
	__local real* pivBC,  /* [1] pivinv broadcast */
	__local int* icolBC,    /* [1] icol broadcast */
	__global real* alphaG)
{
	real big, dum;
	real tmpSwap;
	int i, licol = 0, irow = 0, j, k, l, ll;
	int n = (*CUDA_CC).Mfit;
	int mfit1 = (*CUDA_CC).Mfit1;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = n / BLOCK_DIM;
	if (n % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > n) brtmph = n;
	brtmpl++;

	/* stage the damped matrix and the right-hand side straight from
	   alpha/beta (this replaces the covar staging that mrqmin_1_end used to
	   do in global memory; covar itself is no longer written at all) */
	for (j = brtmpl; j <= brtmph; j++)
	{
		int ixx = j * mfit1 + 1;
		for (k = 1; k <= n; k++, ixx++)
		{
			covL[ixx] = alphaG[ixx];
		}
		int qq = j * mfit1 + j;
		covL[qq] = R_MUL(alphaG[qq], R_ADD(R_C(1.0), (*CUDA_LCC).Alamda));
		daL[j] = (*CUDA_LCC).beta[j];
	}

	if (threadIdx.x == 0)
	{
		for (j = 1; j <= n; j++) ipivL[j] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

	for (i = 1; i <= n; i++)
	{
		big = R_C(-1.0);
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)
		{
			if (ipivL[j] != 1)
			{
				int ixx = j * mfit1 + 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					if (ipivL[k] == 0)
					{
						real tmpcov = R_FABS(covL[ixx]);
						if (R_GE(tmpcov, big))
						{
							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if (ipivL[k] > 1)
					{
						barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
						return(1);
					}
				}
			}
		}
		shBig[threadIdx.x] = big;
		shIrow[threadIdx.x] = irow;
		shIcol[threadIdx.x] = licol;

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		if (threadIdx.x == 0)
		{
			big = shBig[0];
			icolBC[0] = shIcol[0];
			irow = shIrow[0];

			for (j = 1; j < BLOCK_DIM; j++)
			{
				if (R_GE(shBig[j], big))
				{
					big = shBig[j];
					irow = shIrow[j];
					icolBC[0] = shIcol[j];
				}
			}

			++ipivL[icolBC[0]];

			if (irow != icolBC[0])
			{
				for (l = 1; l <= n; l++)
				{
					tmpSwap = covL[irow * mfit1 + l];
					covL[irow * mfit1 + l] = covL[icolBC[0] * mfit1 + l];
					covL[icolBC[0] * mfit1 + l] = tmpSwap;
				}

				tmpSwap = daL[irow];
				daL[irow] = daL[icolBC[0]];
				daL[icolBC[0]] = tmpSwap;
			}

			int covarIdx = icolBC[0] * mfit1 + icolBC[0];

			if (R_EQ(covL[covarIdx], R_C(0.0)))
			{
				for (int l2 = 1; l2 <= (*CUDA_CC).ma; l2++)
				{
					(*CUDA_LCC).atry[l2] = (*CUDA_LCC).cg[l2];
				}

				icolBC[0] = -1;
			}
			else
			{
				pivBC[0] = R_DIV(R_C(1.0), covL[covarIdx]);
				covL[covarIdx] = R_C(1.0);

				daL[icolBC[0]] = R_MUL(daL[icolBC[0]], pivBC[0]);
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		if (icolBC[0] < 0)
		{
			return(2);
		}

		for (l = brtmpl; l <= brtmph; l++)
		{
			int qq = icolBC[0] * mfit1 + l;
			real covar1 = R_MUL(covL[qq], pivBC[0]);
			covL[qq] = covar1;
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();

		for (ll = brtmpl; ll <= brtmph; ll++)
		{
			if (ll != icolBC[0])
			{
				int ixx = ll * mfit1;
				int jxx = icolBC[0] * mfit1;
				dum = covL[ixx + icolBC[0]];
				covL[ixx + icolBC[0]] = R_C(0.0);
				ixx++;
				jxx++;
				for (l = 1; l <= n; l++, ixx++, jxx++)
				{
					R_SUBFROMM(covL[ixx], covL[jxx], dum);
				}

				R_SUBFROMM(daL[ll], daL[icolBC[0]], dum);
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE); //__syncthreads();
	}

	/* only the step vector leaves the solver (the column unscramble of the
	   classic routine acted on the inverse, which nothing reads) */
	for (j = brtmpl; j <= brtmph; j++)
	{
		(*CUDA_LCC).da[j] = daL[j];
	}

	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); //__syncthreads();

	return(0);
}
//N.B. The foll. L-M routines are modified versions of Press et al.
//  converted from Mikko's fortran code

//  8.11.2006


int mrqmin_1_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__local real* covL,
	__local real* daL,
	__local int* ipivL,
	__local real* shBig,
	__local int* shIrow,
	__local int* shIcol,
	__local real* pivBC,
	__local int* icolBC,
	__global real* alphaG)
{
	int j;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int ma = (*CUDA_CC).ma;

	//precalc thread boundaries
	int tmph, tmpl;
	tmph = ma / BLOCK_DIM;
	if (ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > ma) tmph = ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = (*CUDA_CC).Mfit / BLOCK_DIM;
	if ((*CUDA_CC).Mfit % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > (*CUDA_CC).Mfit) brtmph = (*CUDA_CC).Mfit;
	brtmpl++;

	// <<< Iter1Mrqmin1EndPre1
	if ((*CUDA_LCC).isAlamda)
	{
		for (j = tmpl; j <= tmph; j++)
		{
			(*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];
		}
	}

	// The damped matrix is staged straight from alpha into local memory by
	// gauss_errc; covar is not touched at all (it is rezeroed by
	// ClCalculateIter1Mrqcof2Start before mrqcof2 accumulates into it).

	// <<< gauss_errc    ---- GAUS ERROR CODE ----
	int err_code = gauss_errc(CUDA_LCC, CUDA_CC, covL, daL, ipivL, shBig, shIrow, shIcol, pivBC, icolBC, alphaG);
	if (err_code)
	{
		return err_code;
	}
	//     __syncthreads(); inside gauss
	// <<< gaus_errc END

	// >>> Iter1Mrqmin1EndPost
	if (threadIdx.x == 0)
	{
		//		if (err_code != 0) return(err_code);  "bacha na sync threads" - Watch out for Sync Threads
		j = 0;
		for (int l = 1; l <= ma; l++)
			if ((*CUDA_CC).ia[l])
			{
				j++;
				(*CUDA_LCC).atry[l] = R_ADD((*CUDA_LCC).cg[l], (*CUDA_LCC).da[j]);
			}
	}

	return err_code;
}

void mrqmin_2_end(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* scr)
{
	__global real* alphaG = scr + (*CUDA_CC).offAlpha;
	__global real* covarG = scr + (*CUDA_CC).offCovar;

	int j, k, l;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	/* the copies are split over the work-group, the scalar updates are done
	   by work-item 0. The branch stays uniform: the only write to Chisq /
	   Ochisq (else branch) sets Chisq = Ochisq, which keeps the test false. */
	int lsize = get_local_size(0);

	if (R_LT((*CUDA_LCC).Chisq, (*CUDA_LCC).Ochisq))
	{
		if (threadIdx.x == 0)
			(*CUDA_LCC).Alamda = R_DIV((*CUDA_LCC).Alamda, (*CUDA_CC).Alamda_incr);
		for (j = 1; j <= (*CUDA_CC).Mfit; j++)
		{
			for (k = 1 + threadIdx.x; k <= (*CUDA_CC).Mfit; k += lsize)
			{
				alphaG[j * (*CUDA_CC).Mfit1 + k] = covarG[j * (*CUDA_CC).Mfit1 + k];

				//if (blockIdx.x == 0)
				//	printf("alpha[%3d]: %10.7f\n", alphaG[j * (*CUDA_CC).Mfit1 + k]);
			}
		}
		for (j = 1 + threadIdx.x; j <= (*CUDA_CC).Mfit; j += lsize)
		{
			(*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
		}
		for (l = 1 + threadIdx.x; l <= (*CUDA_CC).ma; l += lsize)
		{
			(*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
		}
	}
	else if (threadIdx.x == 0)
	{
		(*CUDA_LCC).Alamda = R_MUL((*CUDA_CC).Alamda_incr, (*CUDA_LCC).Alamda);
		(*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
	}


}
__kernel void ClCalculatePrepare(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_result* CUDA_FR,
    __global int* CUDA_End,
    real_arg freq_start,
    real_arg freq_step,
    int n_max,
    int n_start)
{
    int3 blockIdx;
    blockIdx.x = get_group_id(0);
    int x = blockIdx.x;

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

    /* one work-group per (frequency, pole) pair: N_POLES consecutive groups
       share the same trial frequency and each of them will run one of the
       initial poles, all concurrently (the poles used to be a serial host-side
       loop) */
    int n = n_start + blockIdx.x / N_POLES;


    //zero context
    if (n > n_max)
    {
        //CUDA_mCC[x].isInvalid = 1;
        (*CUDA_LCC).isInvalid = 1;
        (*CUDA_FR).isInvalid = 1;
        return;
    }
    else
    {
        //CUDA_mCC[x].isInvalid = 0;
        (*CUDA_LCC).isInvalid = 0;
        (*CUDA_FR).isInvalid = 0;
    }

    //printf("[%d] n_start: %d | n_max: %d | n: %d \n", blockIdx.x, n_start, n_max, n);

    //printf("Idx: %d | isInvalid: %d\n", x, CUDA_CC[x].isInvalid);
    //printf("Idx: %d | isInvalid: %d\n", x, (*CUDA_LCC).isInvalid);

    //CUDA_mCC[x].freq = freq_start - (n - 1) * freq_step;
    (*CUDA_LCC).freq = R_SUBM(R_ARG(freq_start), R_FROM_INT(n - 1), R_ARG(freq_step));

    ///* initial poles */
    (*CUDA_LFR).per_best = R_C(0.0);
    (*CUDA_LFR).dark_best = R_C(0.0);
    (*CUDA_LFR).la_best = R_C(0.0);
    (*CUDA_LFR).be_best = R_C(0.0);
    (*CUDA_LFR).dev_best = R_1E40;

    //printf("n: %4d, CUDA_CC[%3d].freq: %10.7f, CUDA_FR[%3d].la_best: %10.7f, isInvalid: %4d \n", n, x, (*CUDA_LCC).freq, x, (*CUDA_LFR).la_best, (*CUDA_LCC).isInvalid);

    //if (blockIdx.x == 0)
        //printf("Prepare CUDA_End: %2d\n", *CUDA_End);
}

__kernel void ClCalculatePreparePole(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global struct freq_result* CUDA_FR,
    __global real* CUDA_cg_first,
    __global int* CUDA_End,
    __global struct freq_context* CUDA_CC2)
{
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    //const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

    /* launched with BLOCK_DIM work-items per group: the per-group loops are
       strided over the group, the scalar setup is done by work-item 0 */
    const int lsize = get_local_size(0);

    /* CUDA_CC2 is no longer used: it only received a Brightness copy for a
       host-side debug check that has been removed (kept so the kernel
       argument indices stay unchanged) */

    //*CUDA_End = 13;
    //printf("[%d] PreparePole t: %d, CUDA_End: %d\n", x, t, *CUDA_End);


    /* invalid contexts (n > n_max) are not counted here: the host knows how
       many there are and starts CUDA_End at that count; isReported is already
       0 for every context (host initialises CUDA_FR before each batch) */
    if ((*CUDA_LCC).isInvalid)
        return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("[Device] PreparePole > ma: %d\n", (*CUDA_CC).ma);

    //* starts from the initial ellipsoid */
    for (int i = 1 + threadIdx.x; i <= (*CUDA_CC).Ncoef; i += lsize)
    {
        (*CUDA_LCC).cg[i] = CUDA_cg_first[i];
        //if(blockIdx.x == 0)
        //	printf("cg[%3d]: %10.7f\n", i, CUDA_cg_first[i]);
    }
    //printf("Idx: %d | m: %d | Ncoef: %d\n", x, m, (*CUDA_CC).Ncoef);
    //printf("cg[%d]: %.7f\n", x, CUDA_CC[x].cg[CUDA_CC[x].Ncoef + 1]);
    //printf("Idx: %d | beta_pole[%d]: %.7f\n", x, m, CUDA_CC[x].beta_pole[m]);

    for (int i = 1 + threadIdx.x; i <= (*CUDA_CC).Nphpar; i += lsize)
    {
        (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i] = (*CUDA_CC).par[i];
        //              ia[Ncoef+3+i] = ia_par[i]; moved to global
        //if (blockIdx.x == 0)
        //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + i, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + i]);

    }

    /* the remaining cg entries and the scalar state: work-item 0 only (the
       indices are disjoint from the strided loops above) */
    if (threadIdx.x != 0)
        return;

    real period = R_DIV(R_C(1.0), (*CUDA_LCC).freq);

    /* which of the initial poles this group runs (see ClCalculatePrepare) */
    const int m = blockIdx.x % N_POLES + 1;

    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = (*CUDA_CC).beta_pole[m];
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2] = (*CUDA_CC).lambda_pole[m];
    //if (blockIdx.x == 0)
    //{
    //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);
    //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);
    //}
    //printf("cg[%d]: %.7f | cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1], (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);

    /* The formulas use beta measured from the pole */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = R_SUB(R_C(90.0), (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);
    //printf("90 - cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);

    /* conversion of lambda, beta to radians */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1] = R_MUL(R_DEG2RAD, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2] = R_MUL(R_DEG2RAD, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);
    //printf("cg[%d]: %.7f | cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1], (*CUDA_CC).Ncoef + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);

    /* Use omega instead of period */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3] = R_DIV(R_48PI, period);

    //if (threadIdx.x == 0)
    //{
    //	printf("[%3d] cg[%3d]: %10.7f, period: %10.7f\n", blockIdx.x, (*CUDA_CC).Ncoef + 3, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3], period);
    //}

    /* Lommel-Seeliger part */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2] = R_C(1.0);
    //if (blockIdx.x == 0)
    //{
    //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 2]);
    //}

    /* Use logarithmic formulation for Lambert to keep it positive */
    (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1] = R_LOG((*CUDA_CC).cl);
    //(*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1] = (*CUDA_CC).logCl;   //log((*CUDA_CC).cl);


    //if (blockIdx.x == 0)
    //{
    //	printf("cg[%3d]: %10.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1]);
    //}
    //printf("cg[%d]: %.7f\n", (*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3 + (*CUDA_CC).Nphpar + 1]);

    /* Levenberg-Marquardt loop */
    // moved to global iter_max,iter_min,iter_dif_max
    //
    (*CUDA_LCC).rchisq = R_C(-1.0);
    (*CUDA_LCC).Alamda = R_C(-1.0);
    (*CUDA_LCC).Niter = 0;
    (*CUDA_LCC).iter_diff = R_1E40;
    (*CUDA_LCC).dev_old = R_1E30;
    (*CUDA_LCC).dev_new = R_C(0.0);
    //	(*CUDA_LCC).Lastcall=0; always ==0
    (*CUDA_LFR).isReported = 0;
}

__kernel void ClCalculateIter1Begin(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_result* CUDA_FR,
    __global int* CUDA_End,
    int CUDA_n_iter_min,
    int CUDA_n_iter_max,
    real_arg CUDA_iter_diff_max,
    real_arg CUDA_Alamda_start,
    int n_contexts)
{
    int x = get_global_id(0);
    if (x >= n_contexts)
        return;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    //const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[x];

    if ((*CUDA_LCC).isInvalid)
    {
        return;
    }

    //                                   ?    < 50                                 ?       > 0                                   ?      < 0
    (*CUDA_LCC).isNiter = (((*CUDA_LCC).Niter < CUDA_n_iter_max) && R_GT((*CUDA_LCC).iter_diff, R_ARG(CUDA_iter_diff_max))) || ((*CUDA_LCC).Niter < CUDA_n_iter_min);
    (*CUDA_FR).isNiter = (*CUDA_LCC).isNiter;

    //printf("[%d] isNiter: %d, Alamda: %10.7f\n", blockIdx.x, (*CUDA_LCC).isNiter, (*CUDA_LCC).Alamda);

    if ((*CUDA_LCC).isNiter)
    {
        if (R_LT((*CUDA_LCC).Alamda, R_C(0.0)))
        {
            (*CUDA_LCC).isAlamda = 1;
            (*CUDA_LCC).Alamda = R_ARG(CUDA_Alamda_start); /* initial alambda */
        }
        else
        {
            (*CUDA_LCC).isAlamda = 0;
        }
    }
    else
    {
        if (!(*CUDA_LFR).isReported)
        {
            //int oldEnd = *CUDA_End;
            //atomic_add(CUDA_End, 1);
            int t = *CUDA_End;
            atomic_inc(CUDA_End);

            //printf("[%d] t: %2d, Begin %2d\n", blockIdx.x, t, *CUDA_End);

            (*CUDA_LFR).isReported = 1;
        }
    }

    //if (threadIdx.x == 1)
    //	printf("[begin] Alamda: %10.7f\n", (*CUDA_LCC).Alamda);
    //barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); // TEST
}

__kernel void ClCalculateIter1Mrqcof1Start(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global real* scratch)
    //__global int* CUDA_End)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    //double* dytemp = &CUDA_Dytemp[blockIdx.x];

    //double* Area = &CUDA_mCC[0].Area;

    //if (blockIdx.x == 0)
    //	printf("[%d][%3d] [Mrqcof1Start]\n", blockIdx.x, threadIdx.x);
        //printf("isInvalid: %3d, isNiter: %3d, isAlamda: %3d\n", (*CUDA_LCC).isInvalid, (*CUDA_LCC).isNiter, (*CUDA_LCC).isAlamda);

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return; //>> 0

    // => mrqcof_start(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
    mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, scr + (*CUDA_CC).offAlpha, (*CUDA_LCC).beta);
}

__kernel void ClCalculateIter1Mrqcof1Matrix(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int lpoints,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx;
    blockIdx.x = get_group_id(0);
    int x = blockIdx.x;

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return;

    __local int num; // __shared__

    int3 localIdx;
    localIdx.x = get_local_id(0);
    if (localIdx.x == 0)
    {
        num = 0;
    }

    mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, lpoints, num, scr);
}

/* mrqcof pass over one lightcurve, curve1 part. trial == 0: the current
   parameters (cg), skipped when the previous trial was rejected (isAlamda
   == 0); trial == 1: the trial parameters (atry). One kernel for both passes
   gives mrqcof_curve1 (and its bright()) a single call site, which the
   compiler then inlines - with two callers it could emit a real function
   call, costing the kernel ~248 VGPRs (occupancy 5 instead of 12). */
__kernel void ClCalculateIter1MrqcofCurve1(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints,
    __global real* scratch,
    const int trial)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!trial && !(*CUDA_LCC).isAlamda) return;

    __local int num;  // __shared__
    __local real tmave[BLOCK_DIM];

    if (threadIdx.x == 0)
    {
        num = 0;
    }

    mrqcof_curve1(CUDA_LCC, CUDA_CC, trial ? (*CUDA_LCC).atry : (*CUDA_LCC).cg, tmave, inrel, lpoints, num, scr);
}

__kernel void ClCalculateIter1Mrqcof1Curve1Last(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    //double* dytemp = &CUDA_Dytemp[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return;

    __local real res[BLOCK_DIM];

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof1Curve1Last\n");

    mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, scr + (*CUDA_CC).offAlpha, (*CUDA_LCC).beta, res, inrel, lpoints, scr);
    //if (threadIdx.x == 0)
    //{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++) {
    //		printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    //}
}

/* mrqcof pass over one lightcurve, curve2 part (normal equations).
   trial == 0: into alpha/beta, skipped when isAlamda == 0; trial == 1: into
   covar/da. One kernel for both passes gives mrqcof_curve2 a single call
   site (see ClCalculateIter1MrqcofCurve1). */
__kernel void ClCalculateIter1MrqcofCurve2(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints,
    __global real* scratch,
    const int trial)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!trial && !(*CUDA_LCC).isAlamda) return;

    /* OpenCL requires __local declarations at kernel scope */
    __local real dydaT[CURVE2_K][DYT_STRIDE];
    __local real tileS[5 * CURVE2_K];   /* s2w, dws, dy, coef, coef1 */

    mrqcof_curve2(CUDA_LCC, CUDA_CC,
        scr + (trial ? (*CUDA_CC).offCovar : (*CUDA_CC).offAlpha),
        trial ? (*CUDA_LCC).da : (*CUDA_LCC).beta,
        dydaT, tileS, tileS + CURVE2_K, tileS + 2 * CURVE2_K, tileS + 3 * CURVE2_K, inrel, lpoints, scr);
}

__kernel void ClCalculateIter1Mrqcof1End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    if (!(*CUDA_LCC).isAlamda) return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof1End\n");


    real ochisq = mrqcof_end(CUDA_LCC, CUDA_CC, scr + (*CUDA_CC).offAlpha);
    if (threadIdx.x == 0)
        (*CUDA_LCC).Ochisq = ochisq;


    ////if (threadIdx.x == 0)
    ////{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++) {
    //	printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    ////}
}

__kernel void ClCalculateIter1Mrqmin1End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    /* runtime-sized by the host to Mfit1*Mfit1 doubles (~24 KB for real
       workunits) so the kernel also fits GCN's 32 KB local-memory limit */
    __local real* covL,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    //if (threadIdx.x == 0)
    //{
    //	int i = 56;
    //	//for (int i = 1; i <= 60; i++)
    //	//{
    //		printf("[%d] alpha[%2d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).alpha[i]);
    //	//}
    //}

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqmin1End\n");

    // gauss_err =
    //mrqmin_1_end(CUDA_LCC, CUDA_CC, sh_icol, sh_irow, sh_big, icol, pivinv);


    /* OpenCL requires __local declarations at kernel scope; the solver runs
       entirely in local memory (see gauss_errc.cl). covL comes in as a
       runtime-sized kernel argument. */
    __local real daL[DYT_STRIDE];
    __local int ipivL[DYT_STRIDE];
    __local real shBig[BLOCK_DIM];
    __local int shIrow[BLOCK_DIM];
    __local int shIcol[BLOCK_DIM];
    __local real pivBC[1];
    __local int icolBC[1];

    mrqmin_1_end(CUDA_LCC, CUDA_CC, covL, daL, ipivL, shBig, shIrow, shIcol, pivBC, icolBC, scr + (*CUDA_CC).offAlpha);

    //if (blockIdx.x == 0) {
    //	printf("[%3d] sh_icol[%3d]: %3d\n", threadIdx.x, threadIdx.x, sh_icol[threadIdx.x]);
    //}
}

__kernel void ClCalculateIter1Mrqcof2Start(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof2Start\n");


    //mrqcof_start(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);
    mrqcof_start(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, scr + (*CUDA_CC).offCovar, (*CUDA_LCC).da);

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("alpha[56]: %10.7f\n", (*CUDA_LCC).alpha[56]);
}

__kernel void ClCalculateIter1Mrqcof2Matrix(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int lpoints,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    __local int num; // __shared__

    int3 localIdx;
    localIdx.x = get_local_id(0);
    if (localIdx.x == 0)
    {
        num = 0;
    }

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqcof2Matrix\n");

    //mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).atry, lpoints);
    mrqcof_matrix(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, lpoints, num, scr);
}

__kernel void ClCalculateIter1Mrqcof2Curve1Last(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    const int inrel,
    const int lpoints,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    //double* dytemp = &CUDA_Dytemp[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    __local real res[BLOCK_DIM];

    //mrqcof_curve1_last(CUDA_LCC, CUDA_CC, dytemp, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, res, inrel, lpoints);
    mrqcof_curve1_last(CUDA_LCC, CUDA_CC, (*CUDA_LCC).atry, scr + (*CUDA_CC).offCovar, (*CUDA_LCC).da, res, inrel, lpoints, scr);
}

__kernel void ClCalculateIter1Mrqcof2End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    real chisq = mrqcof_end(CUDA_LCC, CUDA_CC, scr + (*CUDA_CC).offCovar);
    if (threadIdx.x == 0)
        (*CUDA_LCC).Chisq = chisq;

    //if (blockIdx.x == 0)
    //	printf("[%3d] Chisq: %10.7f\n", threadIdx.x, (*CUDA_LCC).Chisq);
}

__kernel void ClCalculateIter1Mrqmin2End(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global real* scratch)
{
    __global real* scr = scratch + get_group_id(0) * (ulong)(*CUDA_CC).scrStride;

    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    if (!(*CUDA_LCC).isNiter) return;

    //if (blockIdx.x == 0 && threadIdx.x == 0)
    //	printf("Mrqmin2End\n");

    //mrqmin_2_end(CUDA_LCC, CUDA_ia, CUDA_ma);
    mrqmin_2_end(CUDA_LCC, CUDA_CC, scr);

    if (threadIdx.x == 0)
        (*CUDA_LCC).Niter++;

    //if (blockIdx.x == 0)
    //	printf("[%3d] Niter: %d\n", threadIdx.x, (*CUDA_LCC).Niter);
    //printf("|");
}

__kernel void ClCalculateIter2(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC)
{
    int i, j;
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];

    if ((*CUDA_LCC).isInvalid)
    {
        return;
    }

    //if (blockIdx.x == 0)
    //	printf("[%3d] isNiter: %d\n", threadIdx.x, (*CUDA_LCC).isNiter);

    if ((*CUDA_LCC).isNiter)
    {
        /* evaluated once, before anyone updates Ochisq: work-item 0 used to
           write Ochisq inside this branch while other wavefronts could still
           be evaluating the condition, which made the branch - and the
           barriers in it - divergent */
        const int improved = (*CUDA_LCC).Niter == 1 || R_LT((*CUDA_LCC).Chisq, (*CUDA_LCC).Ochisq);
        if (improved)
        {
            int brtmph = (*CUDA_CC).Numfac / BLOCK_DIM;
            if ((*CUDA_CC).Numfac % BLOCK_DIM) brtmph++;
            int brtmpl = threadIdx.x * brtmph;
            brtmph = brtmpl + brtmph;
            if (brtmph > (*CUDA_CC).Numfac) brtmph = (*CUDA_CC).Numfac;
            brtmpl++;

            curv(CUDA_LCC, CUDA_CC, (*CUDA_LCC).cg, brtmpl, brtmph);
            /* work-item 0 sums the Area of every facet; this also orders every
               read of Ochisq above before the write below */
            barrier(CLK_GLOBAL_MEM_FENCE);

            if (threadIdx.x == 0)
            {
                (*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;

                for (i = 1; i <= 3; i++)
                {
                    (*CUDA_LCC).chck[i] = R_C(0.0);


                    for (j = 1; j <= (*CUDA_CC).Numfac; j++)
                    {
                        real qq;
                        qq = R_ADDM((*CUDA_LCC).chck[i], (*CUDA_LCC).Area[j], (*CUDA_CC).Nor[j][i - 1]);

                        //if (blockIdx.x == 0)
                        //	printf("[%d] [%d][%3d] qq: %10.7f, chck[%d]: %10.7f, Area[%3d]: %10.7f, Nor[%3d][%d]: %10.7f\n",
                        //		blockIdx.x, i, j, qq, i, (*CUDA_LCC).chck[i], j, (*CUDA_LCC).Area[j], j, i - 1, (*CUDA_CC).Nor[j][i - 1]);

                        (*CUDA_LCC).chck[i] = qq;
                    }

                    //if (blockIdx.x == 0)
                    //	printf("[%d] chck[%d]: %10.7f\n", blockIdx.x, i, (*CUDA_LCC).chck[i]);
                }

                //printf("[%d] chck[1]: %10.7f, chck[2]: %10.7f, chck[3]: %10.7f\n", blockIdx.x, (*CUDA_LCC).chck[1], (*CUDA_LCC).chck[2], (*CUDA_LCC).chck[3]);

                (*CUDA_LCC).rchisq = R_SUBM((*CUDA_LCC).Chisq, R_ADD(R_ADD(R_POW2((*CUDA_LCC).chck[1]), R_POW2((*CUDA_LCC).chck[2])), R_POW2((*CUDA_LCC).chck[3])), R_POW2((*CUDA_CC).conw_r));
                //(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - ((*CUDA_LCC).chck[1] * (*CUDA_LCC).chck[1] + (*CUDA_LCC).chck[2] * (*CUDA_LCC).chck[2] + (*CUDA_LCC).chck[3] * (*CUDA_LCC).chck[3]) * ((*CUDA_CC).conw_r * (*CUDA_CC).conw_r);
            }
        }


        if (threadIdx.x == 0)
        {
            //if (blockIdx.x == 0)
            //	printf("ndata - 3: %3d\n", (*CUDA_CC).ndata - 3);

            (*CUDA_LCC).dev_new = R_SQRT(R_DIV((*CUDA_LCC).rchisq, R_FROM_INT((*CUDA_CC).ndata - 3)));

            //if (blockIdx.x == 233)
            //{
            //	double dev_best = (*CUDA_LCC).dev_new * (*CUDA_LCC).dev_new * ((*CUDA_CC).ndata - 3);
            //	printf("[%3d] rchisq: %12.8f, ndata-3: %3d, dev_new: %12.8f, dev_best: %12.8f\n",
            //		blockIdx.x, (*CUDA_LCC).rchisq, (*CUDA_CC).ndata - 3, (*CUDA_LCC).dev_new, dev_best);
            //}

            // NOTE: only if this step is better than the previous, 1e-10 is for numeric errors
            if (R_GT(R_SUB((*CUDA_LCC).dev_old, (*CUDA_LCC).dev_new), R_1E_10))
            {
                (*CUDA_LCC).iter_diff = R_SUB((*CUDA_LCC).dev_old, (*CUDA_LCC).dev_new);
                (*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
            }
            //		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
        }

    }
}

__kernel void ClCalculateFinishPole(
    __global struct mfreq_context* CUDA_mCC,
    __global struct freq_context* CUDA_CC,
    __global struct freq_result* CUDA_FR)
{
    int i;
    int3 blockIdx;
    blockIdx.x = get_group_id(0);

    //const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
    //const auto CUDA_LFR = &CUDA_FR[blockIdx.x];
    __global struct mfreq_context* CUDA_LCC = &CUDA_mCC[blockIdx.x];
    __global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

    if ((*CUDA_LCC).isInvalid) return;

    real totarea = R_C(0.0);
    for (i = 1; i <= (*CUDA_CC).Numfac; i++)
    {
        totarea = R_ADD(totarea, (*CUDA_LCC).Area[i]);
    }

    //if(blockIdx.x == 2)
    //	printf("[%d] chck[1]: %10.7f, chck[2]: %10.7f, chck[3]: %10.7f, conw_r: %10.7f\n", blockIdx.x, (*CUDA_LCC).chck[1], (*CUDA_LCC).chck[2], (*CUDA_LCC).chck[3], (*CUDA_CC).conw_r);

    //if (blockIdx.x == 2)
    //	printf("rchisq: %10.7f, Chisq: %10.7f \n", (*CUDA_LCC).rchisq, (*CUDA_LCC).Chisq);

    //const double sum = pow((*CUDA_LCC).chck[1], 2.0) + pow((*CUDA_LCC).chck[2], 2.0) + pow((*CUDA_LCC).chck[3], 2.0);
    const real sum = R_ADDM(R_MADD((*CUDA_LCC).chck[1], (*CUDA_LCC).chck[1], R_MUL((*CUDA_LCC).chck[2], (*CUDA_LCC).chck[2])), (*CUDA_LCC).chck[3], (*CUDA_LCC).chck[3]);
    //printf("[FinishPole] [%d] sum: %10.7f\n", blockIdx.x, sum);

    const real dark = R_SQRT(sum);

    //if (blockIdx.x == 232 || blockIdx.x == 233)
    //	printf("[%d] sum: %12.8f, dark: %12.8f, totarea: %12.8f, dark_best: %12.8f\n", blockIdx.x, sum, dark, totarea, dark / totarea * 100);

    /* period solution */
    const real period = R_DIV(R_2PI, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 3]);

    /* pole solution */
    const real la_tmp = R_MUL(R_RAD2DEG, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 2]);

    //if (la_tmp < 0.0)
    //	printf("[CalculateFinishPole] la_best: %4.0f\n", la_tmp);

    const real be_tmp = R_SUBM(R_C(90.0), R_RAD2DEG, (*CUDA_LCC).cg[(*CUDA_CC).Ncoef + 1]);

    //if (blockIdx.x == 2)
        //printf("[%d] dev_new: %10.7f, dev_best: %10.7f\n", blockIdx.x, (*CUDA_LCC).dev_new, (*CUDA_LFR).dev_best);

    if (R_LT((*CUDA_LCC).dev_new, (*CUDA_LFR).dev_best))
    {
        (*CUDA_LFR).dev_best = (*CUDA_LCC).dev_new;
        (*CUDA_LFR).dev_best_x2 = (*CUDA_LCC).rchisq;
        (*CUDA_LFR).per_best = period;
        (*CUDA_LFR).dark_best = R_MUL(R_DIV(dark, totarea), R_C(100.0));
        (*CUDA_LFR).la_best = R_LT(la_tmp, R_C(0.0)) ? R_ADD(la_tmp, R_C(360.0)) : la_tmp;
        (*CUDA_LFR).be_best = be_tmp;

        //printf("[%d] dev_best: %12.8f\n", blockIdx.x, (*CUDA_LFR).dev_best);

        //if (blockIdx.x == 232)
        //{
        //	double dev_best = (*CUDA_LFR).dev_best * (*CUDA_LFR).dev_best * ((*CUDA_CC).ndata - 3);
        //	printf("[%3d] rchisq: %12.8f, ndata-3: %3d, dev_new: %12.8f, dev_best: %12.8f\n",
        //		blockIdx.x, (*CUDA_LCC).rchisq, (*CUDA_CC).ndata - 3, (*CUDA_LFR).dev_best, dev_best);
        //}
    }

    if (R_ISNAN((*CUDA_LFR).dark_best) == 1)
    {
        (*CUDA_LFR).dark_best = R_C(1.0);
    }

    //if (blockIdx.x == 2)
    //	printf("dark_best: %10.7f \n", (*CUDA_LFR).dark_best);

    //debug
    /*	(*CUDA_LFR).dark=dark;
    (*CUDA_LFR).totarea=totarea;
    (*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
    (*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
    (*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
}
