/* Convexity regularization function

   8.11.2006
*/

#include "globals.h"
#include <immintrin.h>
#include "CalcStrategyAvx512.hpp"
#include "arrayHelpers.hpp"

#if defined(__GNUC__)
__attribute__((target("avx512f")))
#endif

void CalcStrategyAvx512::conv(const int nc, const int ma, globals &gl)
{
	gl.ymod = 0;

    for (auto j = 1; j <= ma; j++)
    {
        gl.dyda[j] = 0;
    }

    for (auto i = 0; i < Numfac; i++)
    {
		gl.ymod += gl.Area[i] * gl.Nor[nc - 1][i];
        __m512d avx_Darea = _mm512_set1_pd(gl.Darea[i]);
        __m512d avx_Nor = _mm512_set1_pd(gl.Nor[nc - 1][i]);
        double *Dg_row = gl.Dg[i];

        for (auto j = 0; j < Ncoef; j += 8)
        {
            __m512d avx_dres = _mm512_load_pd(&gl.dyda[j]);
            __m512d avx_Dg = _mm512_load_pd(&Dg_row[j]);

            avx_dres = _mm512_fmadd_pd(_mm512_mul_pd(avx_Darea, avx_Dg), avx_Nor, avx_dres);

            _mm512_store_pd(&gl.dyda[j], avx_dres);
        }
    }
}
