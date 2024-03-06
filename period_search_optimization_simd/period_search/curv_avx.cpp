/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"
#include <immintrin.h>
#include "CalcStrategyAvx.hpp"

#if defined(__GNUC__)
__attribute__((target("avx")))
#endif
void CalcStrategyAvx::curv(double cg[])
{
    int i, m, l, k;

    for (i = 1; i <= Numfac; i++)
    {
        double g = 0;
        int n = 0;
        //m=0

        __m256d avx_g = _mm256_setzero_pd();
        __m256d avx_Fc = _mm256_set1_pd(Fc[i][0]);
        for (l = 0; l <= Lmax + 1; l += 4) {
          __m256d avx_cg = _mm256_loadu_pd(&cg[l + 1]);
          __m256d avx_fsum = _mm256_mul_pd(avx_cg, avx_Fc);
          __m256d avx_coeff = _mm256_set_pd(Pleg[i][l + 3][0], Pleg[i][l + 2][0], Pleg[i][l + 1][0], Pleg[i][l][0]);
          avx_g = _mm256_add_pd(avx_g, _mm256_mul_pd(avx_coeff, avx_fsum));
        }
        n += Lmax + 1;
          
        for (m = 1; m <= Mmax; m++) {
          __m256d avx_Fc = _mm256_set1_pd(Fc[i][m]);
          __m256d avx_Fs = _mm256_set1_pd(Fs[i][m]);

          int offset = 0;
          for (l = m; l <= Lmax; l += 4) {
            int n1 = n + (8 * offset++);
            __m256d avx_cg = _mm256_set_pd(cg[n1 + 7], cg[n1 + 5], cg[n1 + 3], cg[n1 + 1]);
            __m256d avx_cg2 = _mm256_set_pd(cg[n1 + 8], cg[n1 + 6], cg[n1 + 4], cg[n1 + 2]);
            __m256d avx_fsum = _mm256_mul_pd(avx_cg, avx_Fc);
            avx_fsum = _mm256_add_pd(avx_fsum, _mm256_mul_pd(avx_cg2, avx_Fs));
            __m256d avx_coeff = _mm256_set_pd(Pleg[i][l + 3][m], Pleg[i][l + 2][m], Pleg[i][l + 1][m], Pleg[i][l][m]);
            avx_g = _mm256_add_pd(avx_g, _mm256_mul_pd(avx_coeff, avx_fsum));
           }

           n += (2 * (Lmax - m + 1));
        }

        avx_g = _mm256_hadd_pd(avx_g, avx_g);
	    avx_g = _mm256_add_pd(avx_g, _mm256_permute2f128_pd(avx_g, avx_g, 1));
        g = _mm256_cvtsd_f64(avx_g);
          
        g = exp(g);
        Area[i - 1] = Darea[i - 1] * g;

        avx_g = _mm256_set1_pd(g);
        int cyklus = (n >> 2) << 2;
        for (k = 1; k <= cyklus; k += 4)
        {
            __m256d avx_pom = _mm256_loadu_pd(&Dsph[i][k]);
            avx_pom = _mm256_mul_pd(avx_pom, avx_g);
            _mm256_store_pd(&Dg[i - 1][k - 1], avx_pom);
        }
        if (k <= n) Dg[i - 1][k - 1] = g * Dsph[i][k]; //last odd value
        if (k + 1 <= n) Dg[i - 1][k - 1 + 1] = g * Dsph[i][k + 1]; //last odd value
        if (k + 2 <= n) Dg[i - 1][k - 1 + 2] = g * Dsph[i][k + 2]; //last odd value
    }

    // For Unit tests
    /*printf("\nDg[%d][%d:\n", 288, 24);
    for(int q = 0; q <= 16; q++)
    {
        printf("_dg_%d[] = { ", q);
        for(int p = 0; p <= 288; p++)
        {
            printf("%.30f, ", Dg[p][q]);
        }
        printf("};\n");
    }

    printf("\nArea[%d]:\n", 288);
    printf("_area[] = { ");
    for(int p = 0; p <= 288; p++)
    {
        printf("%.30f, ", Area[p]);
    }
    printf("};\n");*/
}