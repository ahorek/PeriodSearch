/* Convexity regularization function

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include <arm_neon.h>

/*
double conv(int nc, double dres[], int ma)
{
   int i, j;

   double res;

   res = 0;
   for (j = 1; j <= ma; j++)
      dres[j] = 0;

   for (i = 0; i < Numfac; i++)
   {
      res += Area[i] * Nor[nc-1][i];
      for (j = 0; j < Ncoef; j++) {
         dres[j] += Darea[i] * Dg[i][j] * Nor[nc-1][i];
      }
   }

   return(res);
}
*/

double conv(int nc, double dres[], int ma) {
    int i, j;
    double res = 0;

    //for (j = 1; j <= ma; j++)
    //    dres[j] = 0;
    memset(dres + 1, 0, ma * sizeof(double));

    for (i = 0; i < Numfac; i++) {
        res += Area[i] * Nor[nc - 1][i];
        double *Dg_row = Dg[i];
        float64x2_t avx_Darea = vdupq_n_f64(Darea[i]);
        float64x2_t avx_Nor = vdupq_n_f64(Nor[nc - 1][i]);
        for (j = 0; j < Ncoef; j += 2) {
            float64x2_t avx_dres = vld1q_f64(&dres[j]);
            float64x2_t avx_Dg = vld1q_f64(&Dg_row[j]);

            avx_dres = vaddq_f64(avx_dres, vmulq_f64(vmulq_f64(avx_Darea, avx_Dg), avx_Nor));
            vst1q_f64(&dres[j], avx_dres);
        }
    }
    return res;
}