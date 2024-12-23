#include <cmath>
#include <vector>
#include "globals.h"
#include "CalcStrategyAsimd.hpp"
#include "arrayHelpers.hpp"

#if defined __GNUG__ && !defined __clang__
__attribute__((__target__("arch=armv8-a+simd")))
#elif defined __GNUG__ && __clang__
// NOTE: The following generates warning: unsupported architecture 'armv8-a+simd' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// __attribute__((target("arch=armv8-a+simd")))
#endif

/**
 * @brief Computes the curvature function and facet area from the Laplace series.
 *
 * This function calculates the curvature function and hence the facet area based on the Laplace series
 * using the provided coefficients and global data. The results are stored in the global variables.
 *
 * @param cg A reference to a vector of doubles containing the coefficients for the Laplace series.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables `Area` and `Dg`.
 *
 * @date 8.11.2006
 */
void CalcStrategyAsimd::curv(std::vector<double>& cg, globals& gl)
{
    int k;

    for (auto i = 1; i <= Numfac; i++)
    {
        double g = 0;
        int n = 0;
        //m=0
        for (auto l = 0; l <= Lmax; l++)
        {
            n++;
            const double fsum = cg[n] * Fc[i][0];
            g = g + Pleg[i][l][0] * fsum;
        }
        //
        for (auto m = 1; m <= Mmax; m++)
        {
            for (auto l = m; l <= Lmax; l++)
            {
                n++;
                double fsum = cg[n] * Fc[i][m];
                n++;
                fsum = fsum + cg[n] * Fs[i][m];
                g = g + Pleg[i][l][m] * fsum;
            }
        }

        g = exp(g);
        gl.Area[i - 1] = gl.Darea[i - 1] * g;
        float64x2_t avx_g = vdupq_n_f64(g);

        for (k = 1; k < n; k += 2) {
            float64x2_t avx_pom = vld1q_f64(&Dsph[i][k]);
            avx_pom = vmulq_f64(avx_pom, avx_g);
            vst1q_f64(&gl.Dg[i - 1][k - 1], avx_pom);
        }

        if (k == n)
        {
            gl.Dg[i - 1][k - 1] = g * Dsph[i][k]; //last odd value
        }
    }
}
