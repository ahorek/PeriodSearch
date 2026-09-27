#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include <immintrin.h>
#include "CalcStrategyAvx512.hpp"

#if defined(__GNUC__)
__attribute__((target("avx512f")))
#endif
inline static __m512d blendv_pd(__m512d a, __m512d b, __m512d c) {
	__m512i result = _mm512_ternarylogic_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(b), _mm512_srai_epi64(_mm512_castpd_si512(c), 63), 0xd8);

	return _mm512_castsi512_pd(result);
}

// -mavx512dq
#if defined(__GNUC__)
__attribute__((target("avx512dq,avx512f")))
#endif
inline static __m512d cmp_pd(__m512d a, __m512d b) {
	__m512i result = _mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_GT_OS));

	return _mm512_castsi512_pd(result);
}

#if defined(__GNUC__)
__attribute__((target("avx512f")))
#endif
inline static int movemask_pd(__m512d a) {
	return (int)_mm512_cmpneq_epi64_mask(_mm512_setzero_si512(), _mm512_and_si512(_mm512_set1_epi64(0x8000000000000000ULL), _mm512_castpd_si512(a)));
}

#if defined(__GNUC__)
__attribute__((target("avx512f")))
#endif
inline static double reduce_pd(__m512d a) {
	__m256d b = _mm256_add_pd(_mm512_castpd512_pd256(a), _mm512_extractf64x4_pd(a, 1));
	__m128d d = _mm_add_pd(_mm256_castpd256_pd128(b), _mm256_extractf128_pd(b, 1));
	double* f = (double*)&d;
	return _mm_cvtsd_f64(d) + f[1];
}


#define INNER_CALC \
		 res_br=_mm512_add_pd(res_br,avx_pbr);	\
			__m512d avx_sum1,avx_sum10,avx_sum2,avx_sum20,avx_sum3,avx_sum30; \
			\
				avx_sum1=_mm512_mul_pd(avx_Nor1,avx_de11); \
				avx_sum1=_mm512_fmadd_pd(avx_Nor2,avx_de21, avx_sum1); \
				avx_sum1=_mm512_fmadd_pd(avx_Nor3,avx_de31, avx_sum1); \
\
				avx_sum10=_mm512_mul_pd(avx_Nor1,avx_de011); \
				avx_sum10=_mm512_fmadd_pd(avx_Nor2,avx_de021, avx_sum10); \
				avx_sum10=_mm512_fmadd_pd(avx_Nor3,avx_de031, avx_sum10); \
				\
				avx_sum2=_mm512_mul_pd(avx_Nor1,avx_de12); \
				avx_sum2=_mm512_fmadd_pd(avx_Nor2,avx_de22, avx_sum2); \
				avx_sum2=_mm512_fmadd_pd(avx_Nor3,avx_de32, avx_sum2); \
				\
				avx_sum20=_mm512_mul_pd(avx_Nor1,avx_de012); \
				avx_sum20=_mm512_fmadd_pd(avx_Nor2,avx_de022, avx_sum20); \
				avx_sum20=_mm512_fmadd_pd(avx_Nor3,avx_de032, avx_sum20); \
				\
				avx_sum3=_mm512_mul_pd(avx_Nor1,avx_de13); \
				avx_sum3=_mm512_fmadd_pd(avx_Nor2,avx_de23, avx_sum3); \
				avx_sum3=_mm512_fmadd_pd(avx_Nor3,avx_de33, avx_sum3); \
				\
				avx_sum30=_mm512_mul_pd(avx_Nor1,avx_de013); \
				avx_sum30=_mm512_fmadd_pd(avx_Nor2,avx_de023, avx_sum30); \
				avx_sum30=_mm512_fmadd_pd(avx_Nor3,avx_de033, avx_sum30); \
				\
			avx_sum1=_mm512_mul_pd(avx_sum1,avx_dsmu); \
			avx_sum2=_mm512_mul_pd(avx_sum2,avx_dsmu); \
			avx_sum3=_mm512_mul_pd(avx_sum3,avx_dsmu); \
			avx_sum10=_mm512_mul_pd(avx_sum10,avx_dsmu0); \
			avx_sum20=_mm512_mul_pd(avx_sum20,avx_dsmu0); \
			avx_sum30=_mm512_mul_pd(avx_sum30,avx_dsmu0); \
			\
			avx_dyda1=_mm512_fmadd_pd(avx_Area,_mm512_add_pd(avx_sum1,avx_sum10), avx_dyda1); \
			avx_dyda2=_mm512_fmadd_pd(avx_Area,_mm512_add_pd(avx_sum2,avx_sum20), avx_dyda2); \
			avx_dyda3=_mm512_fmadd_pd(avx_Area,_mm512_add_pd(avx_sum3,avx_sum30), avx_dyda3); \
			\
			avx_d=_mm512_fmadd_pd(_mm512_mul_pd(avx_lmu,avx_lmu0),avx_Area, avx_d); \
			avx_d1=_mm512_add_pd(avx_d1,_mm512_div_pd(_mm512_mul_pd(_mm512_mul_pd(avx_Area,avx_lmu),avx_lmu0),_mm512_add_pd(avx_lmu,avx_lmu0)));
// end of inner_calc

#define INNER_CALC_DSMU \
	  avx_Area=_mm512_load_pd(&gl.Area[i]); \
	  avx_dnom=_mm512_add_pd(avx_lmu,avx_lmu0); \
	  avx_s=_mm512_mul_pd(_mm512_mul_pd(avx_lmu,avx_lmu0),_mm512_add_pd(avx_cl,_mm512_div_pd(avx_cls,avx_dnom))); \
	  avx_pdbr=_mm512_mul_pd(_mm512_load_pd(&gl.Darea[i]),avx_s); \
	  avx_pbr=_mm512_mul_pd(avx_Area,avx_s); \
	  avx_powdnom=_mm512_div_pd(avx_lmu0,avx_dnom); \
	  avx_powdnom=_mm512_mul_pd(avx_powdnom,avx_powdnom); \
	  avx_dsmu=_mm512_fmadd_pd(avx_cl,avx_lmu0, _mm512_mul_pd(avx_cls,avx_powdnom)); \
	  avx_powdnom=_mm512_div_pd(avx_lmu,avx_dnom); \
	  avx_powdnom=_mm512_mul_pd(avx_powdnom,avx_powdnom); \
	  avx_dsmu0=_mm512_fmadd_pd(avx_cl,avx_lmu, _mm512_mul_pd(avx_cls,avx_powdnom));
// end of inner_calc_dsmu

#if defined(__GNUC__)
__attribute__((target("avx512dq,avx512f")))
#endif

/**
 * @brief Computes integrated brightness of all visible and illuminated areas and its derivatives.
 *
 * This function calculates the integrated brightness of all visible and illuminated areas based on the provided time `t`,
 * coefficient vector `cg`, and global data. It also computes the derivatives of the brightness with respect to the coefficients.
 *
 * @param t The time at which the brightness is evaluated.
 * @param cg A reference to a vector of doubles containing the coefficients for the brightness calculation.
 * @param ncoef An integer representing the number of coefficients.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables `ymod` and `dyda`.
 *
 * @date 8.11.2006
 * @author Josef Durec
 *
 * @date 25.3.2024 modified by Pavel Rosicky
 */
void CalcStrategyAvx512::bright(const double t, std::vector<double>& cg, const int ncoef, globals &gl)
{
	int i, j, k;
	incl_count = 0;
	double *ee = gl.xx1;
	double *ee0 = gl.xx2;

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);				/* Lambert */
	cls = cg[ncoef];						/* Lommel-Seeliger */
	dot_product_new(ee, ee0, cos_alpha);
	alpha = acos(cos_alpha);
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php);				/* computes also Scale */

	matrix(cg[ncoef0], t, tmat, dtm);

	/* Directions (and derivatives) in the rotating system */
	for (i = 1; i <= 3; i++)
	{
		e[i] = 0;
		e0[i] = 0;
		for (j = 1; j <= 3; j++)
		{
			e[i] += tmat[i][j] * ee[j];
			e0[i] += tmat[i][j] * ee0[j];
			de[i][j] = 0;
			de0[i][j] = 0;
			for (k = 1; k <= 3; k++)
			{
				de[i][j] += dtm[j][i][k] * ee[k];
				de0[i][j] += dtm[j][i][k] * ee0[k];
			}
		}
	}

	/*Integrated brightness (phase coefficients used later) */
	__m512d avx_e1 = _mm512_set1_pd(e[1]);
	__m512d avx_e2 = _mm512_set1_pd(e[2]);
	__m512d avx_e3 = _mm512_set1_pd(e[3]);
	__m512d avx_e01 = _mm512_set1_pd(e0[1]);
	__m512d avx_e02 = _mm512_set1_pd(e0[2]);
	__m512d avx_e03 = _mm512_set1_pd(e0[3]);
	__m512d avx_de11 = _mm512_set1_pd(de[1][1]);
	__m512d avx_de12 = _mm512_set1_pd(de[1][2]);
	__m512d avx_de13 = _mm512_set1_pd(de[1][3]);
	__m512d avx_de21 = _mm512_set1_pd(de[2][1]);
	__m512d avx_de22 = _mm512_set1_pd(de[2][2]);
	__m512d avx_de23 = _mm512_set1_pd(de[2][3]);
	__m512d avx_de31 = _mm512_set1_pd(de[3][1]);
	__m512d avx_de32 = _mm512_set1_pd(de[3][2]);
	__m512d avx_de33 = _mm512_set1_pd(de[3][3]);
	__m512d avx_de011 = _mm512_set1_pd(de0[1][1]);
	__m512d avx_de012 = _mm512_set1_pd(de0[1][2]);
	__m512d avx_de013 = _mm512_set1_pd(de0[1][3]);
	__m512d avx_de021 = _mm512_set1_pd(de0[2][1]);
	__m512d avx_de022 = _mm512_set1_pd(de0[2][2]);
	__m512d avx_de023 = _mm512_set1_pd(de0[2][3]);
	__m512d avx_de031 = _mm512_set1_pd(de0[3][1]);
	__m512d avx_de032 = _mm512_set1_pd(de0[3][2]);
	__m512d avx_de033 = _mm512_set1_pd(de0[3][3]);

	__m512d avx_tiny = _mm512_set1_pd(TINY);
	__m512d avx_cl = _mm512_set1_pd(cl);
	__m512d avx_cls = _mm512_set1_pd(cls);
	__m512d avx_11 = _mm512_set1_pd(1.0);
	__m512d avx_Scale = _mm512_set1_pd(Scale);
	__m512d res_br = _mm512_setzero_pd();
	__m512d avx_dyda1 = _mm512_setzero_pd();
	__m512d avx_dyda2 = _mm512_setzero_pd();
	__m512d avx_dyda3 = _mm512_setzero_pd();
	__m512d avx_d = _mm512_setzero_pd();
	__m512d avx_d1 = _mm512_setzero_pd();

#ifdef __GNUC__
	double g[8] __attribute__((aligned(64)));
#else
	alignas(64) double g[8];
#endif

	for (i = 0; i < Numfac; i += 8)
	{
		__m512d avx_lmu, avx_lmu0, cmpe, cmpe0, cmp;
		__m512d avx_Nor1 = _mm512_load_pd(&gl.Nor[0][i]);
		__m512d avx_Nor2 = _mm512_load_pd(&gl.Nor[1][i]);
		__m512d avx_Nor3 = _mm512_load_pd(&gl.Nor[2][i]);
		__m512d avx_s, avx_dnom, avx_dsmu, avx_dsmu0, avx_powdnom, avx_pdbr, avx_pbr;
		__m512d avx_Area;

		avx_lmu = _mm512_mul_pd(avx_e1, avx_Nor1);
		avx_lmu = _mm512_fmadd_pd(avx_e2, avx_Nor2, avx_lmu);
		avx_lmu = _mm512_fmadd_pd(avx_e3, avx_Nor3, avx_lmu);
		avx_lmu0 = _mm512_mul_pd(avx_e01, avx_Nor1);
		avx_lmu0 = _mm512_fmadd_pd(avx_e02, avx_Nor2, avx_lmu0);
		avx_lmu0 = _mm512_fmadd_pd(avx_e03, avx_Nor3, avx_lmu0);

		cmpe = cmp_pd(avx_lmu, avx_tiny);
		cmpe0 = cmp_pd(avx_lmu0, avx_tiny);
		cmp = _mm512_and_pd(cmpe, cmpe0);
		int icmp = movemask_pd(cmp);

		if (icmp)
		{
			INNER_CALC_DSMU

				avx_pbr = blendv_pd(_mm512_setzero_pd(), avx_pbr, cmp);
			avx_dsmu = blendv_pd(_mm512_setzero_pd(), avx_dsmu, cmp);
			avx_dsmu0 = blendv_pd(_mm512_setzero_pd(), avx_dsmu0, cmp);
			avx_lmu = blendv_pd(_mm512_setzero_pd(), avx_lmu, cmp);
			avx_lmu0 = blendv_pd(avx_11, avx_lmu0, cmp); // Note: So that it is not divisible by zero (abychom nedelili nulou)

			_mm512_store_pd(g, avx_pdbr);
			if (icmp & 1)
			{
				Dg_row[incl_count] = (__m512d*)&gl.Dg[i];
				dbr[incl_count++] = _mm512_set1_pd(g[0]);
			}
			if (icmp & 2)
			{
				Dg_row[incl_count] = (__m512d*)&gl.Dg[i + 1];
				dbr[incl_count++] = _mm512_set1_pd(g[1]);
			}
			if (icmp & 4)
			{
				Dg_row[incl_count] = (__m512d*)&gl.Dg[i + 2];
				dbr[incl_count++] = _mm512_set1_pd(g[2]);
			}
			if (icmp & 8)
			{
				Dg_row[incl_count] = (__m512d*)&gl.Dg[i + 3];
				dbr[incl_count++] = _mm512_set1_pd(g[3]);
			}
			if (icmp & 16)
			{
				Dg_row[incl_count] = (__m512d*)&gl.Dg[i + 4];
				dbr[incl_count++] = _mm512_set1_pd(g[4]);
			}
			if (icmp & 32)
			{
				Dg_row[incl_count] = (__m512d*)&gl.Dg[i + 5];
				dbr[incl_count++] = _mm512_set1_pd(g[5]);
			}
			if (icmp & 64)
			{
				Dg_row[incl_count] = (__m512d*)&gl.Dg[i + 6];
				dbr[incl_count++] = _mm512_set1_pd(g[6]);
			}
			if (icmp & 128)
			{
				Dg_row[incl_count] = (__m512d*)&gl.Dg[i + 7];
				dbr[incl_count++] = _mm512_set1_pd(g[7]);
			}
			INNER_CALC
		}
	}

	dbr[incl_count] = _mm512_setzero_pd();
	Dg_row[incl_count] = Dg_row[0];
	gl.ymod = reduce_pd(res_br);

	/* Derivatives of brightness w.r.t. g-coefficients */
	int ncoef03 = ncoef0 - 3, dgi = 0;
	i = 0;
	const int nvec = (ncoef03 + 7) / 8;	/* 8-column vectors, the last may be partial */

	/* 8 vectors per pass over the visible facets (was fewer):
	   more independent accumulator chains, and far fewer passes over the
	   dbr/Dg_row lists. Every column keeps the original operation sequence
	   (fma(dbr[j+1], D, fma(dbr[j], D, t))), over the
	   facets in ascending order, and the same vectors are read and stored. */
	for (; dgi + 8 <= nvec; dgi += 8, i += 64)
	{
		__m512d t0, t1, t2, t3;
		__m512d t4, t5, t6, t7;
		t0 = _mm512_setzero_pd();
		t1 = _mm512_setzero_pd();
		t2 = _mm512_setzero_pd();
		t3 = _mm512_setzero_pd();
		t4 = _mm512_setzero_pd();
		t5 = _mm512_setzero_pd();
		t6 = _mm512_setzero_pd();
		t7 = _mm512_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			const __m512d* r0 = &Dg_row[j][dgi];
			const __m512d* r1 = &Dg_row[j + 1][dgi];
			const __m512d d0 = dbr[j];
			const __m512d d1 = dbr[j + 1];

			t0 = _mm512_fmadd_pd(d1, r1[0], _mm512_fmadd_pd(d0, r0[0], t0));
			t1 = _mm512_fmadd_pd(d1, r1[1], _mm512_fmadd_pd(d0, r0[1], t1));
			t2 = _mm512_fmadd_pd(d1, r1[2], _mm512_fmadd_pd(d0, r0[2], t2));
			t3 = _mm512_fmadd_pd(d1, r1[3], _mm512_fmadd_pd(d0, r0[3], t3));
			t4 = _mm512_fmadd_pd(d1, r1[4], _mm512_fmadd_pd(d0, r0[4], t4));
			t5 = _mm512_fmadd_pd(d1, r1[5], _mm512_fmadd_pd(d0, r0[5], t5));
			t6 = _mm512_fmadd_pd(d1, r1[6], _mm512_fmadd_pd(d0, r0[6], t6));
			t7 = _mm512_fmadd_pd(d1, r1[7], _mm512_fmadd_pd(d0, r0[7], t7));
		}

		_mm512_store_pd(&gl.dyda[i + 0], _mm512_mul_pd(t0, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 8], _mm512_mul_pd(t1, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 16], _mm512_mul_pd(t2, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 24], _mm512_mul_pd(t3, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 32], _mm512_mul_pd(t4, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 40], _mm512_mul_pd(t5, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 48], _mm512_mul_pd(t6, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 56], _mm512_mul_pd(t7, avx_Scale));
	}
	for (; dgi + 4 <= nvec; dgi += 4, i += 32)
	{
		__m512d t0, t1, t2, t3;
		t0 = _mm512_setzero_pd();
		t1 = _mm512_setzero_pd();
		t2 = _mm512_setzero_pd();
		t3 = _mm512_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			const __m512d* r0 = &Dg_row[j][dgi];
			const __m512d* r1 = &Dg_row[j + 1][dgi];
			const __m512d d0 = dbr[j];
			const __m512d d1 = dbr[j + 1];

			t0 = _mm512_fmadd_pd(d1, r1[0], _mm512_fmadd_pd(d0, r0[0], t0));
			t1 = _mm512_fmadd_pd(d1, r1[1], _mm512_fmadd_pd(d0, r0[1], t1));
			t2 = _mm512_fmadd_pd(d1, r1[2], _mm512_fmadd_pd(d0, r0[2], t2));
			t3 = _mm512_fmadd_pd(d1, r1[3], _mm512_fmadd_pd(d0, r0[3], t3));
		}

		_mm512_store_pd(&gl.dyda[i + 0], _mm512_mul_pd(t0, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 8], _mm512_mul_pd(t1, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 16], _mm512_mul_pd(t2, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 24], _mm512_mul_pd(t3, avx_Scale));
	}
	for (; dgi + 2 <= nvec; dgi += 2, i += 16)
	{
		__m512d t0, t1;
		t0 = _mm512_setzero_pd();
		t1 = _mm512_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			const __m512d* r0 = &Dg_row[j][dgi];
			const __m512d* r1 = &Dg_row[j + 1][dgi];
			const __m512d d0 = dbr[j];
			const __m512d d1 = dbr[j + 1];

			t0 = _mm512_fmadd_pd(d1, r1[0], _mm512_fmadd_pd(d0, r0[0], t0));
			t1 = _mm512_fmadd_pd(d1, r1[1], _mm512_fmadd_pd(d0, r0[1], t1));
		}

		_mm512_store_pd(&gl.dyda[i + 0], _mm512_mul_pd(t0, avx_Scale));
		_mm512_store_pd(&gl.dyda[i + 8], _mm512_mul_pd(t1, avx_Scale));
	}
	for (; dgi + 1 <= nvec; dgi += 1, i += 8)
	{
		__m512d t0;
		t0 = _mm512_setzero_pd();

		for (j = 0; j < incl_count; j += 2)
		{
			const __m512d* r0 = &Dg_row[j][dgi];
			const __m512d* r1 = &Dg_row[j + 1][dgi];
			const __m512d d0 = dbr[j];
			const __m512d d1 = dbr[j + 1];

			t0 = _mm512_fmadd_pd(d1, r1[0], _mm512_fmadd_pd(d0, r0[0], t0));
		}

		_mm512_store_pd(&gl.dyda[i + 0], _mm512_mul_pd(t0, avx_Scale));
	}

	/* Derivatives of brightness w.r.t. rotation parameters */
	gl.dyda[ncoef0 - 3 + 1 - 1] = reduce_pd(avx_dyda1) * Scale;
	gl.dyda[ncoef0 - 3 + 2 - 1] = reduce_pd(avx_dyda2) * Scale;
	gl.dyda[ncoef0 - 3 + 3 - 1] = reduce_pd(avx_dyda3) * Scale;

	/* Derivatives of br. w.r.t. cl, cls */
	gl.dyda[ncoef - 1 - 1] = reduce_pd(avx_d) * Scale * cl;
	gl.dyda[ncoef - 1] = reduce_pd(avx_d1) * Scale;

	/* Derivatives of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
	{
		gl.dyda[ncoef0 + i - 1] = gl.ymod * dphp[i];
	}

	/* Scaled brightness */
	gl.ymod *= Scale;
}
