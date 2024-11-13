/* Convexity regularization function

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.h"
#include <immintrin.h>
#include "CalcStrategyFma.hpp"
#include "arrayHelpers.hpp"

#if defined(__GNUC__)
__attribute__((target("avx,fma")))
#endif

//void CalcStrategyFma::conv(int nc, double dres[], int ma, double &result, globals &gl)
void CalcStrategyFma::conv(int nc, int ma, globals &gl)
{
    int i, j;

	gl.ymod = 0;
    for (j = 1; j <= ma; j++)
        gl.dyda[j] = 0;

    for (i = 0; i < Numfac; i++)
    {
		gl.ymod += gl.Area[i] * gl.Nor[nc - 1][i];
        __m256d avx_Darea = _mm256_set1_pd(gl.Darea[i]);
        __m256d avx_Nor = _mm256_set1_pd(gl.Nor[nc - 1][i]);
        double *Dg_row = gl.Dg[i];
        for (j = 0; j < Ncoef; j += 4)
        {
            __m256d avx_dres = _mm256_load_pd(&gl.dyda[j]);
            __m256d avx_Dg = _mm256_load_pd(&Dg_row[j]);

            avx_dres = _mm256_fmadd_pd(_mm256_mul_pd(avx_Darea, avx_Dg), avx_Nor, avx_dres);

            _mm256_store_pd(&gl.dyda[j], avx_dres);
        }
    }
}
