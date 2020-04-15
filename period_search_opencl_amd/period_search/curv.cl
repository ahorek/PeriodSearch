/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/
//
//#include <cuda.h>
//#include <math.h>
//#include "globals_CUDA.h"


void curv(__global struct freq_context2 *CUDA_LCC, struct funcarrays FA, double cg[], int brtmpl, int brtmph)
{
    int i, m, n, l, k;
    double fsum, g;

    for (i = brtmpl; i <= brtmph; i++)
    {
        g = 0;
        n = 0;
        for (m = 0; m <= FA.Mmax; m++)
        {
            for (l = m; l <= FA.Lmax; l++)
            {
                n++;
                fsum = cg[n] * FA.Fc[i][m];
                if (m != 0)
                {
                    n++;
                    fsum = fsum + cg[n] * FA.Fs[i][m];
                }
                g = g + FA.Pleg[i][l][m] * fsum;
            }
        }

        g = exp(g);
        (*CUDA_LCC).Area[i] = FA.Darea[i] * g;
        for (k = 1; k <= n; k++)
        {
            (*CUDA_LCC).Dg[i + k * FA.Numfac1] = g * FA.Dsph[i][k];
        }
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
