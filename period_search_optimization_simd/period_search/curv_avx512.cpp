/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "constants.h"
#include <immintrin.h>
#include "CalcStrategyAvx512.hpp"

#if defined(__GNUC__)
__attribute__((target("avx512f")))
#endif
void CalcStrategyAvx512::curv(double cg[])
{
    int i, m, l, k;

    for (i = 1; i <= Numfac; i++)
    {
        double g = 0;
        int n = 0;
        //m=0

        __m512d avx_g = _mm512_setzero_pd();
        __m512d avx_Fc = _mm512_set1_pd(Fc[i][0]);
        for (l = 0; l <= Lmax + 1; l += 8) {
          __m512d avx_cg = _mm512_loadu_pd(&cg[l + 1]);
          __m512d avx_fsum = _mm512_mul_pd(avx_cg, avx_Fc);
          __m512d avx_coeff = _mm512_set_pd(Pleg[i][l + 7][0], Pleg[i][l + 6][0], Pleg[i][l + 5][0], Pleg[i][l + 4][0], Pleg[i][l + 3][0], Pleg[i][l + 2][0], Pleg[i][l + 1][0], Pleg[i][l][0]);
          avx_g = _mm512_fmadd_pd(avx_coeff, avx_fsum, avx_g);
        }
        n += Lmax + 1;

        //
        for (m = 1; m <= Mmax; m++) {
          __m512d avx_Fc = _mm512_set1_pd(Fc[i][m]);
          __m512d avx_Fs = _mm512_set1_pd(Fs[i][m]);

          int offset = 0;
          for (l = m; l <= Lmax; l += 8) {
            int n1 = n + (16 * offset++);
             __m512d avx_cg = _mm512_set_pd(cg[n1 + 15], cg[n1 + 13], cg[n1 + 11], cg[n1 + 9], cg[n1 + 7], cg[n1 + 5], cg[n1 + 3], cg[n1 + 1]);
             __m512d avx_cg2 = _mm512_set_pd(cg[n1 + 16], cg[n1 + 14], cg[n1 + 12], cg[n1 + 10], cg[n1 + 8], cg[n1 + 6], cg[n1 + 4], cg[n1 + 2]);
             __m512d avx_fsum = _mm512_mul_pd(avx_cg, avx_Fc);
             avx_fsum = _mm256_fmadd_pd(avx_cg2, avx_Fs, avx_fsum);
             __m512d avx_coeff = _mm512_set_pd(Pleg[i][l + 7][m], Pleg[i][l + 6][m], Pleg[i][l + 5][m], Pleg[i][l + 4][m], Pleg[i][l + 3][m], Pleg[i][l + 2][m], Pleg[i][l + 1][m], Pleg[i][l][m]);
             avx_g = _mm512_fmadd_pd(avx_coeff, avx_fsum, avx_g);
           }

           n += (2 * (Lmax - m + 1));
        }

        // sum all elements in a vector and store it to a double
        // avx_g = hadd_pd(avx_g, avx_g);
        // avx_g = hpermute_add_pd(avx_g);
        __m256d sum_256 = _mm256_add_pd(_mm512_castpd512_pd256(avx_g), _mm512_extractf64x4_pd(avx_g, 1));
        __m128d sum_128 = _mm_add_pd(_mm256_castpd256_pd128(sum_256), _mm256_extractf128_pd(sum_256, 1));
        double *f = (double*)&sum_128;
        g = _mm_cvtsd_f64(sum_128) + f[1];

        g = exp(g);
        Area[i - 1] = Darea[i - 1] * g;

        avx_g = _mm512_set1_pd(g);
        int cyklus = (n >> 2) << 2;
        for (k = 1; k <= cyklus; k += 8)
        {
            __m512d avx_pom = _mm512_loadu_pd(&Dsph[i][k]);
            avx_pom = _mm512_mul_pd(avx_pom, avx_g);
            _mm512_store_pd(&Dg[i - 1][k - 1], avx_pom);
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