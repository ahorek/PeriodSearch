/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"
#include "CalcStrategyAsimd.hpp"
#include "arrayHelpers.hpp"

#if defined(__GNUC__)
__attribute__((__target__("arch=armv8-a+simd")))
#endif
void CalcStrategyAsimd::curv(double cg[])
{
   int i, m, l, k;

   for (i = 1; i <= Numfac; i++)
   {
      double g = 0;
      int n = 0;
      //m=0

      float64x2_t avx_g = vdupq_n_f64(0.0);
      float64x2_t avx_Fc = vdupq_n_f64(Fc[i][0]);
      for (l = 0; l <= Lmax + 1; l += 2) {
        float64x2_t avx_cg = vld1q_f64(&cg[l + 1]);
        float64x2_t avx_fsum = vmulq_f64(avx_cg, avx_Fc);
        float64x2_t avx_coeff = vsetq_lane_f64(Pleg[i][l + 1][0], Pleg[i][l][0], 0);

        avx_g = vfmaq_f64(avx_fsum, avx_g, avx_coeff);
      }
      n += Lmax + 1;

      //
      for (m = 1; m <= Mmax; m++) {
        float64x2_t  avx_Fc = vdupq_n_f64(Fc[i][m]);
        float64x2_t  avx_Fs = vdupq_n_f64(Fs[i][m]);

        int offset = 0;
        for (l = m; l <= Lmax; l += 2) {
          int n1 = n + (8 * offset++);
          float64x2_t  avx_cg = vsetq_lane_f64(cg[n1 + 7], cg[n1 + 5], cg[n1 + 3], cg[n1 + 1]);
          float64x2_t  avx_cg2 = vsetq_lane_f64(cg[n1 + 8], cg[n1 + 6], cg[n1 + 4], cg[n1 + 2]);
          float64x2_t  avx_fsum = vmulq_f64(avx_cg, avx_Fc);
          avx_fsum = vfmaq_f64(avx_fsum, avx_cg2, avx_Fs);
          float64x2_t  avx_coeff = vsetq_lane_f64(Pleg[i][l + 3][m], Pleg[i][l + 2][m], Pleg[i][l + 1][m], Pleg[i][l][m]);
          avx_g = vfmaq_f64(avx_fsum, avx_g, avx_coeff);
         }

         n += (2 * (Lmax - m + 1));
      }

      avx_g = vpaddq_f64(avx_g, avx_g);
      vst1q_lane_f64(&g, avx_g, 0);

      g = exp(g);
      Area[i-1] = Darea[i-1] * g;

      avx_g = vdupq_n_f64(g);
      for (k = 1; k < n; k += 2) {
        float64x2_t avx_pom = vld1q_f64(&Dsph[i][k]);
        avx_pom = vmulq_f64(avx_pom, avx_g);
        vst1q_f64(&Dg[i-1][k-1], avx_pom);
      }
      if (k==n) Dg[i-1][k-1] = g * Dsph[i][k]; //last odd value
   }
}
