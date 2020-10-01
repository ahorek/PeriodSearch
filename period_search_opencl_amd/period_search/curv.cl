/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

void curv(
    __global struct freq_context2 *CUDA_LCC, 
    __global struct FuncArrays* Fa, 
    double cg[],
    int brtmpl, 
    int brtmph)
    //int i) 
    //__read_only int Numfac,
    //__read_only int Mmax,
    //__read_only int Lmax)
{
    int i;
    int m, n, l, k;
    int Numfac1 = Fa->Numfac + 1;
    double fsum, g;
    //int brtmph, brtmpl;
    int3 blockIdx, threadIdx;
    blockIdx.x = get_group_id(0);
    threadIdx.x = get_local_id(0);

    for (i = brtmpl; i <= brtmph; i++)
    {
        g = 0;
        n = 0;
        for (m = 0; m <= Fa->Mmax; m++)
            for (l = m; l <= Fa->Lmax; l++)
            {
                n++;
                fsum = cg[n] * Fa->Fc[i][m];
                if (m != 0)
                {
                    n++;
                    fsum = fsum + cg[n] * Fa->Fs[i][m];
                }
                g = g + Fa->Pleg[i][l][m] * fsum;
            }
        

        g = exp(g);
        (*CUDA_LCC).Area[i] = Fa->Darea[i] * g;

        for (k = 1; k <= n; k++)
        {
            (*CUDA_LCC).Dg[i + k * Numfac1] = g * Fa->Dsph[i][k];
            //if (blockIdx.x == 2)
            //    printf("curv >>> [%d][%d] i: %d, k: %d, Numfac1: %d, Dg[%d]: % .6f\n", blockIdx.x, threadIdx.x, i, k, Numfac1, i + k * Numfac1, (*CUDA_LCC).Dg[i + k * Numfac1]);
                
            //printf("curv >> [%d][%d]  \ti: %d\tbrtmpl: %d\tbrtmph: %d\n", blockIdx.x, threadIdx.x, i, brtmpl, brtmph);
        }

        //if (blockIdx.x == 2)
        //    printf("curv >>> [%d][%d] \ti: %d\tArea[%d]: % .6f\n", blockIdx.x, threadIdx.x, i, i, (*CUDA_LCC).Area[i]);
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
