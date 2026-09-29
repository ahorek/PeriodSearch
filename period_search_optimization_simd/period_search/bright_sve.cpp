/* computes integrated brightness of all visible and illuminated areas
   and its derivatives

   8.11.2006 - Josef Durec
   25.3.2024 - Pavel Rosicky
*/

#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "CalcStrategySve.hpp"

// Everything below runs on the all-true predicate. The tail lanes of the last iteration
// are loaded as zeros, so lmu / lmu0 are zero there, the visibility test rejects them and
// their contribution to every accumulator is zero as well - which is what lets the final
// svaddv_f64 reduce over the whole vector.
#define INNER_CALC \
    res_br = svadd_f64_x(pt, res_br, avx_pbr); \
    svfloat64_t avx_sum1, avx_sum10, avx_sum2, avx_sum20, avx_sum3, avx_sum30; \
    \
    avx_sum1 = svmul_f64_x(pt, avx_Nor1, avx_de11); \
    avx_sum1 = svmla_f64_x(pt, avx_sum1, avx_Nor2, avx_de21); \
    avx_sum1 = svmla_f64_x(pt, avx_sum1, avx_Nor3, avx_de31); \
    \
    avx_sum10 = svmul_f64_x(pt, avx_Nor1, avx_de011); \
    avx_sum10 = svmla_f64_x(pt, avx_sum10, avx_Nor2, avx_de021); \
    avx_sum10 = svmla_f64_x(pt, avx_sum10, avx_Nor3, avx_de031); \
    \
    avx_sum2 = svmul_f64_x(pt, avx_Nor1, avx_de12); \
    avx_sum2 = svmla_f64_x(pt, avx_sum2, avx_Nor2, avx_de22); \
    avx_sum2 = svmla_f64_x(pt, avx_sum2, avx_Nor3, avx_de32); \
    \
    avx_sum20 = svmul_f64_x(pt, avx_Nor1, avx_de012); \
    avx_sum20 = svmla_f64_x(pt, avx_sum20, avx_Nor2, avx_de022); \
    avx_sum20 = svmla_f64_x(pt, avx_sum20, avx_Nor3, avx_de032); \
    \
    avx_sum3 = svmul_f64_x(pt, avx_Nor1, avx_de13); \
    avx_sum3 = svmla_f64_x(pt, avx_sum3, avx_Nor2, avx_de23); \
    avx_sum3 = svmla_f64_x(pt, avx_sum3, avx_Nor3, avx_de33); \
    \
    avx_sum30 = svmul_f64_x(pt, avx_Nor1, avx_de013); \
    avx_sum30 = svmla_f64_x(pt, avx_sum30, avx_Nor2, avx_de023); \
    avx_sum30 = svmla_f64_x(pt, avx_sum30, avx_Nor3, avx_de033); \
    \
    avx_sum1 = svmul_f64_x(pt, avx_sum1, avx_dsmu); \
    avx_sum2 = svmul_f64_x(pt, avx_sum2, avx_dsmu); \
    avx_sum3 = svmul_f64_x(pt, avx_sum3, avx_dsmu); \
    avx_sum10 = svmul_f64_x(pt, avx_sum10, avx_dsmu0); \
    avx_sum20 = svmul_f64_x(pt, avx_sum20, avx_dsmu0); \
    avx_sum30 = svmul_f64_x(pt, avx_sum30, avx_dsmu0); \
    \
    avx_dyda1 = svmla_f64_x(pt, avx_dyda1, avx_Area, svadd_f64_x(pt, avx_sum1, avx_sum10)); \
    avx_dyda2 = svmla_f64_x(pt, avx_dyda2, avx_Area, svadd_f64_x(pt, avx_sum2, avx_sum20)); \
    avx_dyda3 = svmla_f64_x(pt, avx_dyda3, avx_Area, svadd_f64_x(pt, avx_sum3, avx_sum30)); \
    \
    avx_d = svmla_f64_x(pt, avx_d, avx_Area, svmul_f64_x(pt, avx_lmu, avx_lmu0)); \
    avx_d1 = svmla_f64_x(pt, avx_d1, svmul_f64_x(pt, svmul_f64_x(pt, avx_Area, avx_lmu), avx_lmu0), avx_inv);
// end of inner_calc

// 1/dnom is forced to 1 wherever the facet is rejected, so a zero dnom can never turn into
// an inf that a later multiplication by zero would make a NaN. The area of a rejected facet is
// zero, so all its (finite) contributions vanish without masking every term separately.
#define INNER_CALC_DSMU \
    avx_Area = svsel_f64(cmp, svld1_f64(pg, &gl.Area[i]), avx_zero); \
    avx_dnom = svadd_f64_x(pt, avx_lmu, avx_lmu0); \
    avx_inv = svsel_f64(cmp, svdiv_f64_x(pt, avx_11, avx_dnom), avx_11); \
    avx_s = svmul_f64_x(pt, svmul_f64_x(pt, avx_lmu, avx_lmu0), svmla_f64_x(pt, avx_cl, avx_cls, avx_inv)); \
    avx_pdbr = svmul_f64_x(pt, svld1_f64(pg, &gl.Darea[i]), avx_s); \
    avx_pbr = svmul_f64_x(pt, avx_Area, avx_s); \
    avx_powdnom = svmul_f64_x(pt, avx_lmu0, avx_inv); \
    avx_powdnom = svmul_f64_x(pt, avx_powdnom, avx_powdnom); \
    avx_dsmu = svmla_f64_x(pt, svmul_f64_x(pt, avx_cls, avx_powdnom), avx_cl, avx_lmu0); \
    avx_powdnom = svmul_f64_x(pt, avx_lmu, avx_inv); \
    avx_powdnom = svmul_f64_x(pt, avx_powdnom, avx_powdnom); \
    avx_dsmu0 = svmla_f64_x(pt, svmul_f64_x(pt, avx_cls, avx_powdnom), avx_cl, avx_lmu);
// end of inner_calc_dsmu

#if defined(_MSC_VER) && !defined(__clang__)
  #include <intrin.h>
  #if defined(_M_ARM64)
    #define DG_PREFETCH(p) __prefetch(p)
  #else
    #define DG_PREFETCH(p) _mm_prefetch(p, _MM_HINT_T0)
  #endif
#else
  #define DG_PREFETCH(p) __builtin_prefetch(p, 0, 3)
#endif

#if defined(__GNUC__) && !(defined __x86_64__ || defined(__i386__) || defined(_WIN32))
  #define SVE_TARGET __attribute__((__target__("+sve")))
  #define SVE_TARGET_INLINE __attribute__((__target__("+sve"), always_inline))
#elif defined(__GNUC__)
  #define SVE_TARGET
  #define SVE_TARGET_INLINE __attribute__((always_inline))
#else
  #define SVE_TARGET
  #define SVE_TARGET_INLINE
#endif

/**
 * @brief dyda[cnt * v .. cnt * (v + W)) = Scale * sum_j dbr[j] * Dg_row[j][cnt * v .. cnt * (v + W))
 *
 * W accumulators stay in registers, so each visible Dg row is streamed only once per chunk. Facets are summed
 * sequentially (including the zero-weight pair padding), i.e. in the same order as the original pairwise loop.
 * Only the last vector of the chunk can be partial, it is governed by pl (all-true when the chunk ends inside the row).
 * Relies on the padding entry at dbr[incl_count] and on valid row pointers up to Dg_row[incl_count + DG_PREFETCH_ROWS].
 */
template <int W>
SVE_TARGET_INLINE
static inline void dg_accumulate(double* const* Dg_row, const double* dbr, const int incl_count, const int v, double* dyda,
								 const svbool_t pt, const svbool_t pl, const svfloat64_t avx_Scale)
{
	const int cnt = static_cast<int>(svcntd());
	const int off = cnt * v;

	// Named accumulators (not an array) so that the compiler keeps all of them in registers; unused ones are optimized away.
	svfloat64_t a0 = svdup_n_f64(0.0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0;
	svfloat64_t a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0, a12 = a0, a13 = a0;

#define DG_MLA(k) if (W > k) a##k = svmla_n_f64_x(pt, a##k, svld1_f64(k == W - 1 ? pl : pt, p + cnt * k), pdbr)
	const int n = (incl_count + 1) & ~1;
	for (int j = 0; j < n; j++)
	{
		// Dg rows exceed L1 in total and are visited in a data dependent order, so fetch a few rows ahead
		const char* pf = reinterpret_cast<const char*>(Dg_row[j + DG_PREFETCH_ROWS] + off);
		for (int b = 0; b < W * cnt * 8; b += 64)
			DG_PREFETCH(pf + b);

		const double* p = Dg_row[j] + off;
		const double pdbr = dbr[j];
		DG_MLA(0); DG_MLA(1); DG_MLA(2); DG_MLA(3); DG_MLA(4); DG_MLA(5); DG_MLA(6);
		DG_MLA(7); DG_MLA(8); DG_MLA(9); DG_MLA(10); DG_MLA(11); DG_MLA(12); DG_MLA(13);
	}
#undef DG_MLA

#define DG_STORE(k) if (W > k) svst1_f64(k == W - 1 ? pl : pt, &dyda[off + cnt * k], svmul_f64_x(pt, a##k, avx_Scale))
	DG_STORE(0); DG_STORE(1); DG_STORE(2); DG_STORE(3); DG_STORE(4); DG_STORE(5); DG_STORE(6);
	DG_STORE(7); DG_STORE(8); DG_STORE(9); DG_STORE(10); DG_STORE(11); DG_STORE(12); DG_STORE(13);
#undef DG_STORE
}

SVE_TARGET
static void dg_chunk(const int w, double* const* Dg_row, const double* dbr, const int incl_count, const int v, double* dyda,
					 const svbool_t pt, const svbool_t pl, const svfloat64_t avx_Scale)
{
	switch (w)
	{
		case 14: dg_accumulate<14>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 13: dg_accumulate<13>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 12: dg_accumulate<12>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 11: dg_accumulate<11>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 10: dg_accumulate<10>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 9: dg_accumulate<9>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 8: dg_accumulate<8>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 7: dg_accumulate<7>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 6: dg_accumulate<6>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 5: dg_accumulate<5>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 4: dg_accumulate<4>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 3: dg_accumulate<3>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 2: dg_accumulate<2>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		case 1: dg_accumulate<1>(Dg_row, dbr, incl_count, v, dyda, pt, pl, avx_Scale); break;
		default: break;
	}
}

/**
 * @brief Computes integrated brightness of all visible and illuminated areas and its derivatives.
 *
 * This function calculates the integrated brightness of all visible and illuminated areas based on
 * the provided time t, coefficient vector cg, and global data. It also computes the derivatives of
 * the brightness with respect to the coefficients.
 *
 * @param t The time at which the brightness is evaluated.
 * @param cg A reference to a vector of doubles containing the coefficients for the brightness calculation.
 * @param ncoef An integer representing the number of coefficients.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables ymod and dyda.
 *
 * @date 8.11.2006
 * @author Josef Durec
 */
SVE_TARGET
void CalcStrategySve::bright(const double t, std::vector<double>& cg, const int ncoef, globals &gl)
{
	int i, j, k;
	incl_count = 0;
	double *ee = gl.xx1;
	double *ee0 = gl.xx2;

	ncoef0 = ncoef - 2 - Nphpar;
	cl = exp(cg[ncoef - 1]);		/* Lambert */
	cls = cg[ncoef];				/* Lommel-Seeliger */
	dot_product_new(ee, ee0, cos_alpha);
	alpha = acos(cos_alpha);
	for (i = 1; i <= Nphpar; i++)
		php[i] = cg[ncoef0 + i];

	phasec(dphp, alpha, php);		/* computes also Scale */

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

	/* Integrated brightness (phase coefficients used later) */
	const svbool_t pt = svptrue_b64();
	const int cnt = static_cast<int>(svcntd());

	svfloat64_t avx_e1 = svdup_n_f64(e[1]);
	svfloat64_t avx_e2 = svdup_n_f64(e[2]);
	svfloat64_t avx_e3 = svdup_n_f64(e[3]);
	svfloat64_t avx_e01 = svdup_n_f64(e0[1]);
	svfloat64_t avx_e02 = svdup_n_f64(e0[2]);
	svfloat64_t avx_e03 = svdup_n_f64(e0[3]);
	svfloat64_t avx_de11 = svdup_n_f64(de[1][1]);
	svfloat64_t avx_de12 = svdup_n_f64(de[1][2]);
	svfloat64_t avx_de13 = svdup_n_f64(de[1][3]);
	svfloat64_t avx_de21 = svdup_n_f64(de[2][1]);
	svfloat64_t avx_de22 = svdup_n_f64(de[2][2]);
	svfloat64_t avx_de23 = svdup_n_f64(de[2][3]);
	svfloat64_t avx_de31 = svdup_n_f64(de[3][1]);
	svfloat64_t avx_de32 = svdup_n_f64(de[3][2]);
	svfloat64_t avx_de33 = svdup_n_f64(de[3][3]);
	svfloat64_t avx_de011 = svdup_n_f64(de0[1][1]);
	svfloat64_t avx_de012 = svdup_n_f64(de0[1][2]);
	svfloat64_t avx_de013 = svdup_n_f64(de0[1][3]);
	svfloat64_t avx_de021 = svdup_n_f64(de0[2][1]);
	svfloat64_t avx_de022 = svdup_n_f64(de0[2][2]);
	svfloat64_t avx_de023 = svdup_n_f64(de0[2][3]);
	svfloat64_t avx_de031 = svdup_n_f64(de0[3][1]);
	svfloat64_t avx_de032 = svdup_n_f64(de0[3][2]);
	svfloat64_t avx_de033 = svdup_n_f64(de0[3][3]);
	svfloat64_t avx_Scale = svdup_n_f64(Scale);

	svfloat64_t avx_tiny = svdup_n_f64(TINY);
	svfloat64_t avx_cl = svdup_n_f64(cl);
	svfloat64_t avx_cls = svdup_n_f64(cls);
	svfloat64_t avx_11 = svdup_n_f64(1.0);
	svfloat64_t avx_zero = svdup_n_f64(0.0);
	svfloat64_t res_br = svdup_n_f64(0.0);
	svfloat64_t avx_dyda1 = svdup_n_f64(0.0);
	svfloat64_t avx_dyda2 = svdup_n_f64(0.0);
	svfloat64_t avx_dyda3 = svdup_n_f64(0.0);
	svfloat64_t avx_d = svdup_n_f64(0.0);
	svfloat64_t avx_d1 = svdup_n_f64(0.0);

	double s_vis[SVE_MAX_LANES];
	double s_pdbr[SVE_MAX_LANES];

	for (i = 0; i < Numfac; i += cnt)
	{
		const svbool_t pg = svwhilelt_b64(static_cast<int64_t>(i), static_cast<int64_t>(Numfac));
		const int active = (Numfac - i) < cnt ? (Numfac - i) : cnt;

		svfloat64_t avx_lmu, avx_lmu0;
		svfloat64_t avx_Nor1 = svld1_f64(pg, &gl.Nor[0][i]);
		svfloat64_t avx_Nor2 = svld1_f64(pg, &gl.Nor[1][i]);
		svfloat64_t avx_Nor3 = svld1_f64(pg, &gl.Nor[2][i]);
		svfloat64_t avx_s, avx_dnom, avx_dsmu, avx_dsmu0, avx_powdnom, avx_pdbr, avx_pbr, avx_inv;
		svfloat64_t avx_Area;

		avx_lmu = svmul_f64_x(pt, avx_e1, avx_Nor1);
		avx_lmu = svmla_f64_x(pt, avx_lmu, avx_e2, avx_Nor2);
		avx_lmu = svmla_f64_x(pt, avx_lmu, avx_e3, avx_Nor3);

		avx_lmu0 = svmul_f64_x(pt, avx_e01, avx_Nor1);
		avx_lmu0 = svmla_f64_x(pt, avx_lmu0, avx_e02, avx_Nor2);
		avx_lmu0 = svmla_f64_x(pt, avx_lmu0, avx_e03, avx_Nor3);

		const svbool_t cmp = svand_z(pt, svcmpgt_f64(pt, avx_lmu, avx_tiny),
										 svcmpgt_f64(pt, avx_lmu0, avx_tiny));

		if (svptest_any(pt, cmp))
		{
			INNER_CALC_DSMU

			/* The per-facet bookkeeping stays scalar: the vector length is not a compile
			   time constant, so the predicate cannot be folded into a lane bitmask. */
			svst1_f64(pg, s_vis, svsel_f64(cmp, avx_11, avx_zero));
			svst1_f64(pg, s_pdbr, avx_pdbr);

			// Branchless compaction of the visible facets: always write the slot, advance only when the lane is visible.
			for (j = 0; j < active; j++)
			{
				Dg_row[incl_count] = gl.Dg[i + j];
				dbr[incl_count] = s_pdbr[j];
				incl_count += s_vis[j] != 0.0;
			}

			INNER_CALC
		}
	}

	// zero-weight padding entry for the pairwise order, valid (unused) rows for the prefetch look-ahead
	dbr[incl_count] = 0.0;
	for (j = 0; j <= DG_PREFETCH_ROWS; j++)
		Dg_row[incl_count + j] = gl.Dg[0];

	gl.ymod = svaddv_f64(pt, res_br);

	/* Derivatives of brightness w.r.t. g-coefficients, in balanced chunks of at most 14 vectors (accumulators in registers) */
	const int ncoef03 = ncoef0 - 3;
	const int nvec = (ncoef03 + cnt - 1) / cnt;
	const int nchunks = (nvec + 13) / 14;
	const int wchunk = nchunks > 0 ? (nvec + nchunks - 1) / nchunks : 1;
	for (int v = 0; v < nvec; v += wchunk)
	{
		const int w = nvec - v < wchunk ? nvec - v : wchunk;
		const svbool_t pl = svwhilelt_b64(static_cast<int64_t>(cnt) * (v + w - 1), static_cast<int64_t>(ncoef03));
		dg_chunk(w, Dg_row, dbr, incl_count, v, gl.dyda, pt, pl, avx_Scale);
	}

	/* Derivatives of brightness w.r.t. rotation parameters */
	gl.dyda[ncoef0 - 3 + 1 - 1] = svaddv_f64(pt, avx_dyda1) * Scale;
	gl.dyda[ncoef0 - 3 + 2 - 1] = svaddv_f64(pt, avx_dyda2) * Scale;
	gl.dyda[ncoef0 - 3 + 3 - 1] = svaddv_f64(pt, avx_dyda3) * Scale;

	/* Derivatives of br. w.r.t. cl, cls */
	gl.dyda[ncoef - 1 - 1] = svaddv_f64(pt, avx_d) * Scale * cl;
	gl.dyda[ncoef - 1] = svaddv_f64(pt, avx_d1) * Scale;

	/* Derivatives of br. w.r.t. phase function params. */
	for (i = 1; i <= Nphpar; i++)
		gl.dyda[ncoef0 + i - 1] = gl.ymod * dphp[i];

	/* Scaled brightness */
	gl.ymod *= Scale;
}
