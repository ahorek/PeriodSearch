/* beta, lambda rotation matrix and its derivatives

   8.11.2006
*/

//#include <math.h>
//#include "globals_CUDA.h"

void blmatrix(__global struct freq_context2* CUDA_LCC, const double bet, const double lam)
{
    double cb, sb, cl, sl;
    int3 threadIdx, blockIdx;
    threadIdx.x = get_local_id(0);
    blockIdx.x = get_group_id(0);

    cb = cos(bet);
    sb = sin(bet);
    cl = cos(lam);
    sl = sin(lam);
    (*CUDA_LCC).Blmat[1][1] = cb * cl;
    (*CUDA_LCC).Blmat[1][2] = cb * sl;
    (*CUDA_LCC).Blmat[1][3] = -sb;
    (*CUDA_LCC).Blmat[2][1] = -sl;
    (*CUDA_LCC).Blmat[2][2] = cl;
    (*CUDA_LCC).Blmat[2][3] = 0;
    (*CUDA_LCC).Blmat[3][1] = sb * cl;
    (*CUDA_LCC).Blmat[3][2] = sb * sl;
    (*CUDA_LCC).Blmat[3][3] = cb;

    //printf("blmatrix >>> [%d][%d]: \tbet: % .6f, lam: % .6f\n", blockIdx.x, threadIdx.x, bet, lam);
    //printf("blmatrix >>> [%d][%d]: \t% .6f, % .6f, % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).Blmat[3][1], (*CUDA_LCC).Blmat[3][1], (*CUDA_LCC).Blmat[3][1]);

    /* Ders. of Blmat w.r.t. bet */
    (*CUDA_LCC).Dblm[1][1][1] = -sb * cl;
    (*CUDA_LCC).Dblm[1][1][2] = -sb * sl;
    (*CUDA_LCC).Dblm[1][1][3] = -cb;
    (*CUDA_LCC).Dblm[1][2][1] = 0;
    (*CUDA_LCC).Dblm[1][2][2] = 0;
    (*CUDA_LCC).Dblm[1][2][3] = 0;
    (*CUDA_LCC).Dblm[1][3][1] = cb * cl;
    (*CUDA_LCC).Dblm[1][3][2] = cb * sl;
    (*CUDA_LCC).Dblm[1][3][3] = -sb;
    /* Ders. w.r.t. lam */
    (*CUDA_LCC).Dblm[2][1][1] = -cb * sl;
    (*CUDA_LCC).Dblm[2][1][2] = cb * cl;
    (*CUDA_LCC).Dblm[2][1][3] = 0;
    (*CUDA_LCC).Dblm[2][2][1] = -cl;
    (*CUDA_LCC).Dblm[2][2][2] = -sl;
    (*CUDA_LCC).Dblm[2][2][3] = 0;
    (*CUDA_LCC).Dblm[2][3][1] = -sb * sl;
    (*CUDA_LCC).Dblm[2][3][2] = sb * cl;
    (*CUDA_LCC).Dblm[2][3][3] = 0;
}
