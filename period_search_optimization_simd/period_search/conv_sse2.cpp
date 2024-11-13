/* Convexity regularization function

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include <emmintrin.h>
#include "CalcStrategySse2.hpp"
#include "arrayHelpers.hpp"

#if defined(__GNUC__)
__attribute__((target("sse2")))
#endif

//void CalcStrategySse2::conv(int nc, double dres[], int ma, double &result, globals &gl)
void CalcStrategySse2::conv(int nc, int ma, globals &gl)
{
	int i, j;

	gl.ymod = 0;
	for (j = 1; j <= ma; j++)
		gl.dyda[j] = 0;

	for (i = 0; i < Numfac; i++)
	{
		gl.ymod += gl.Area[i] * gl.Nor[nc - 1][i];
		__m128d avx_Darea = _mm_set1_pd(gl.Darea[i]);
		__m128d avx_Nor = _mm_set1_pd(gl.Nor[nc - 1][i]);
		double* Dg_row = gl.Dg[i];
		for (j = 0; j < Ncoef; j += 2)
		{
			__m128d avx_dres = _mm_load_pd(&gl.dyda[j]);
			__m128d avx_Dg = _mm_load_pd(&Dg_row[j]);

			avx_dres = _mm_add_pd(avx_dres, _mm_mul_pd(_mm_mul_pd(avx_Darea, avx_Dg), avx_Nor));

			_mm_store_pd(&gl.dyda[j], avx_dres);
		}
	}
}
