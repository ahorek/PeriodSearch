#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include "CalcStrategyAsimd.hpp"
#include "arrayHelpers.hpp"

#if defined __GNUG__ && !defined __clang__
__attribute__((__target__("arch=armv8-a+simd")))
#elif defined __GNUG__ && __clang__
// NOTE: The following generates warning: unsupported architecture 'armv8-a+simd' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// __attribute__((target("arch=armv8-a+simd")))
#endif

/**
 * @brief Computes the convexity regularization function.
 *
 * This function calculates the convexity regularization function, updating the global variables `ymod` and `dyda` based on the given parameters and the global data.
 *
 * @param nc An integer representing the current coefficient index.
 * @param ma An integer representing the number of coefficients.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables `ymod` and `dyda`.
 *
 * @date 8.11.2006
 */
void CalcStrategyAsimd::conv(const int nc, const int ma, globals &gl)
{
	gl.ymod = 0;

    for (auto j = 1; j <= ma; j++)
    {
        gl.dyda[j] = 0;
    }

    for (auto i = 0; i < Numfac; i++)
    {
		gl.ymod += gl.Area[i] * gl.Nor[nc - 1][i];
        double *Dg_row = gl.Dg[i];
        float64x2_t avx_Darea = vdupq_n_f64(gl.Darea[i]);
        float64x2_t avx_Nor = vdupq_n_f64(gl.Nor[nc - 1][i]);

        for (auto j = 0; j < Ncoef; j += 2)
        {
            float64x2_t avx_dres = vld1q_f64(&gl.dyda[j]);
            float64x2_t avx_Dg = vld1q_f64(&Dg_row[j]);

            avx_dres = vfmaq_f64(avx_dres, vmulq_f64(avx_Darea, avx_Dg), avx_Nor);
            vst1q_f64(&gl.dyda[j], avx_dres);
        }
    }
}
