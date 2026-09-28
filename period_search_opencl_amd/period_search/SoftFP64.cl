/* SoftFP64.cl - IEEE-754 binary64 arithmetic in software, for devices
   without cl_khr_fp64 (see Real.cl).

   A value is the double's own 64-bit pattern in a ulong; every operation
   is computed with integer arithmetic and rounded to nearest-even exactly
   as the hardware does, with full subnormal support: add, sub, mul, fma,
   div and sqrt are correctly rounded. On top of them sit ports of the AMD
   device-library (ocml) routines the kernels call - exp, log, acos,
   sincos, fmod - reproduced operation by operation from the library */

#ifdef PS_SWFP64

#define SF_CONST __constant
typedef long sf_i64;
#define sf_clz64(x) ((int)clz((ulong)(x)))
#define sf_mulhi(a, b) mul_hi((ulong)(a), (ulong)(b))
#define sf_as_uint(f) as_uint(f)

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

/* a + b with an Inf or NaN operand; b carries the effective sign (negated
   for a subtraction), bNan is b as given, for the NaN propagation */
ulong sf_addsub_special(ulong a, ulong b, ulong bNan)
{
	if (sf_isnan(a) || sf_isnan(b))
		return sf_prop_nan(a, bNan);
	if (sf_abs(a) == SF_INF)
		return sf_abs(b) == SF_INF && a != b ? SF_NAN : a;	/* inf - inf */
	return b;
}

/* a + b for finite a, b on a single path (no data-dependent branches but
   the rare over/underflow in sf_round_pack), so the lanes of a wavefront do
   not diverge on sign and exponent differences: order by magnitude, align
   the smaller significand with a sticky bit, add or subtract, normalize,
   round (10 guard bits below the significand, SoftFloat style). */
ulong sf_add_finite(ulong a, ulong b)
{
	const ulong absA = sf_abs(a), absB = sf_abs(b);
	const int swap = absA < absB;
	const ulong x = swap ? b : a, y = swap ? a : b;
	const ulong absX = swap ? absB : absA, absY = swap ? absA : absB;
	int eX = (int)(absX >> 52), eY = (int)(absY >> 52);
	ulong sX = (absX & SF_FRAC) | (eX ? SF_HIDDEN : 0);
	ulong sY = (absY & SF_FRAC) | (eY ? SF_HIDDEN : 0);
	eX += !eX;
	eY += !eY;
	sX <<= 10;	/* leading one at bit 62 */
	sY <<= 10;
	/* sY >>= d with jam (d = 0 .. 2045) */
	const uint d = (uint)(eX - eY);
	const ulong lost = sY << ((64 - d) & 63);
	const ulong sYj = d == 0 ? sY : d < 64 ? (sY >> d) | (ulong)(lost != 0) : (ulong)(sY != 0);
	const uint signZ = sf_sign(x);
	ulong sZ = signZ != sf_sign(y) ? sX - sYj : sX + sYj;
	if (!sZ)
		return a & b & SF_SIGN;	/* exact zero: -0 only for (-0) + (-0) */
	/* normalize to the leading one at bit 62 */
	const int lz = sf_clz64(sZ);
	int eZ;
	if (lz == 0)
	{
		sZ = (sZ >> 1) | (sZ & 1);
		eZ = eX + 1;
	}
	else
	{
		sZ <<= lz - 1;
		eZ = eX - (lz - 1);
	}
	return sf_round_pack(signZ, eZ - 1, sZ);
}

ulong sf_add(ulong a, ulong b)
{
	if (sf_abs(a) >= SF_INF || sf_abs(b) >= SF_INF)
		return sf_addsub_special(a, b, b);
	return sf_add_finite(a, b);
}

ulong sf_sub(ulong a, ulong b)
{
	if (sf_abs(a) >= SF_INF || sf_abs(b) >= SF_INF)
		return sf_addsub_special(a, b ^ SF_SIGN, b);
	return sf_add_finite(a, b ^ SF_SIGN);
}

/* ================================================================== */
/* mul / fma                                                            */
/* ================================================================== */

/* significand (hidden bit at 52 - a subnormal is normalized, its exponent
   then <= 0) and biased exponent of a nonzero finite value */
ulong sf_unpack(ulong a, int* exp)
{
	const int e = sf_expf(a);
	*exp = e;
	return e ? sf_frac(a) | SF_HIDDEN : sf_norm_sub(sf_frac(a), exp);
}

ulong sf_mul_slow(ulong a, ulong b)
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

/* a * b for normal a, b: the exact product of the two 53-bit significands
   has its leading one at bit 104 or 105, so normalizing it is a fixed
   shift - no clz and no 128-bit case analysis (same rounding as the general
   path via sf_round128) */
ulong sf_mul_normal(ulong a, ulong b)
{
	const ulong sigA = sf_frac(a) | SF_HIDDEN, sigB = sf_frac(b) | SF_HIDDEN;
	const uint signZ = sf_sign(a) ^ sf_sign(b);
	const ulong hi = sf_mulhi(sigA, sigB), lo = sigA * sigB;
	/* hi in [2^40, 2^42): leading one to bit 62 */
	const int top = (int)(hi >> 41);	/* 1: leading one at bit 105 */
	const uint k = top ? 21 : 22;
	const ulong sig = (hi << k) | (lo >> (64 - k)) | (ulong)((lo << k) != 0);
	/* sf_round128: e - k' + 64 + 1084 with e = expA + expB - 2150, k' = k */
	return sf_round_pack(signZ, sf_expf(a) + sf_expf(b) - 2150 - (int)k + 64 + 1084, sig);
}

ulong sf_mul(ulong a, ulong b)
{
	if ((uint)(sf_expf(a) - 1) >= 0x7FEu || (uint)(sf_expf(b) - 1) >= 0x7FEu)
		return sf_mul_slow(a, b);
	return sf_mul_normal(a, b);
}

/* a * b + c with a single rounding (general path) */
ulong sf_fma_slow(ulong a, ulong b, ulong c)
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

/* 128-bit (hi:lo) >> n with jam, n >= 0, branch-free */
void sf_shr128_jam(ulong* hi, ulong* lo, uint n)
{
	const ulong h = *hi, l = *lo;
	/* a whole word first */
	const int big = n >= 64;
	const ulong h1 = big ? 0 : h, l1 = big ? h : l;
	ulong st = big ? (ulong)(l != 0) : 0;
	const uint m = big ? n - 64 : n;	/* < 64 unless n >= 128 */
	const uint mm = m & 63, inv = (64 - mm) & 63;
	const ulong l2 = mm ? (l1 >> mm) | (h1 << inv) : l1;
	const ulong h2 = h1 >> mm;
	st |= mm ? (ulong)((l1 << inv) != 0) : 0;
	const int gone = n >= 128;
	*hi = gone ? 0 : h2;
	*lo = gone ? (ulong)((h | l) != 0) : (l2 | st);
}

/* sf_round128 without data-dependent branches (same result) */
ulong sf_round128_bf(uint sign, int e, ulong hi, ulong lo)
{
	const int k = hi ? sf_clz64(hi) - 1 : 63 + sf_clz64(lo);
	const int big = k >= 64;
	const uint km = (uint)(big ? k - 64 : k) & 63, inv = (64 - km) & 63;
	const ulong hs = big ? lo << km : (km ? (hi << km) | (lo >> inv) : hi);
	const ulong ls = big ? 0 : lo << km;
	return sf_round_pack(sign, e - k + 64 + 1084, hs | (ulong)(ls != 0));
}

/* a * b + c for normal a, b and normal or zero c on a single path: exact
   128-bit product, the smaller of product / addend aligned with jam,
   add or subtract (negating on borrow), normalize, round. Same layout and
   rounding as the general sf_fma below. */
ulong sf_fma_normal(ulong a, ulong b, ulong c)
{
	const int expA = sf_expf(a), expB = sf_expf(b);
	const ulong sigA = sf_frac(a) | SF_HIDDEN, sigB = sf_frac(b) | SF_HIDDEN;
	const uint signP = sf_sign(a) ^ sf_sign(b), signC = sf_sign(c);

	/* P in [2^124, 2^126), value P * 2^expP */
	ulong pHi = sf_mulhi(sigA, sigB), pLo = sigA * sigB;
	pHi = (pHi << 20) | (pLo >> 44);
	pLo <<= 20;
	const int expP = expA + expB - 2170;

	/* C in [2^124, 2^125), value C * 2^expCv; a zero c never wins the
	   alignment and contributes nothing */
	const int czero = sf_iszero(c);
	const ulong cHi = czero ? 0 : (sf_frac(c) | SF_HIDDEN) << 8;
	const int expCv = czero ? expP - 200 : sf_expf(c) - 1147;

	const int d = expP - expCv;
	const int pBig = d >= 0;
	ulong xHi = pBig ? pHi : cHi, xLo = pBig ? pLo : 0;
	ulong yHi = pBig ? cHi : pHi, yLo = pBig ? 0 : pLo;
	const int e = pBig ? expP : expCv;
	sf_shr128_jam(&yHi, &yLo, (uint)(pBig ? d : -d));

	const uint signX = pBig ? signP : signC;
	/* x + y or x - y as one 128-bit two's-complement add */
	const int sub = signP != signC;
	const ulong nyLo = ~yLo + 1, nyHi = ~yHi + (ulong)(nyLo == 0);
	yLo = sub ? nyLo : yLo;
	yHi = sub ? nyHi : yHi;
	ulong sLo = xLo + yLo;
	ulong sHi = xHi + yHi + (ulong)(sLo < xLo);
	/* borrow: |y| > |x|, negate */
	const int neg = (long)sHi < 0;
	const ulong mLo = ~sLo + 1, mHi = ~sHi + (ulong)(mLo == 0);
	sLo = neg ? mLo : sLo;
	sHi = neg ? mHi : sHi;
	const ulong r = sf_round128_bf(signX ^ (uint)neg, e, sHi, sLo);
	return (sHi | sLo) ? r : 0;	/* exact cancellation: +0 */
}

ulong sf_fma(ulong a, ulong b, ulong c)
{
	/* the general path handles nan/inf, zero or subnormal a/b, and a
	   subnormal/inf/nan c */
	if ((uint)(sf_expf(a) - 1) >= 0x7FEu || (uint)(sf_expf(b) - 1) >= 0x7FEu
		|| ((uint)(sf_expf(c) - 1) >= 0x7FEu && !sf_iszero(c)))
		return sf_fma_slow(a, b, c);
	return sf_fma_normal(a, b, c);
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
	sigA = sf_unpack(a, &expA);
	sigB = sf_unpack(b, &expB);

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
	sig = sf_unpack(a, &exp);
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
	sig = sf_unpack(a, &exp);
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

#endif /* PS_SWFP64 */
