/* Convexity regularization function

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include "cuda_vars.cuh"
#include <device_launch_parameters.h>


__device__ double __forceinline__ conv(freq_context *__restrict__ CUDA_LCC, int nc, double *__restrict__ dyda, int bid)
{
  int i, j;
  //__shared__ double res[CUDA_BLOCK_DIM];
  double tmp, tmp2; //, dtmp, dtmp2;
  int nf = CUDA_Numfac, nf1 = CUDA_Numfac1, nco = CUDA_Ncoef;

  j = bid * nf1 + threadIdx.x + 1;
  int xx = threadIdx.x + 1;
  tmp = 0, tmp2 = 0;
  // double * __restrict__ areap = CUDA_Area + j;
  double *__restrict__ areap = &(Areag[bid][threadIdx.x + 1]);
  double *__restrict__ norp = CUDA_Nor[nc] + xx;
#pragma unroll 4
  while (xx <= nf - CUDA_BLOCK_DIM)
  {
    double a0, a1, n0, n1;
    a0 = areap[0];
    n0 = norp[0];
    a1 = areap[CUDA_BLOCK_DIM];
    n1 = norp[CUDA_BLOCK_DIM];
    tmp += a0 * n0;  // areap[0] * norp[0];
    tmp2 += a1 * n1; // areap[CUDA_BLOCK_DIM] * norp[CUDA_BLOCK_DIM];
    xx += 2 * CUDA_BLOCK_DIM;
    areap += 2 * CUDA_BLOCK_DIM;
    norp += 2 * CUDA_BLOCK_DIM;
  }
  // #pragma unroll 1
  if (xx <= nf)
  {
    tmp += areap[0] * norp[0]; // CUDA_Area[j] * CUDA_Nor[nc][xx];
  }

  tmp += tmp2;

  tmp += __shfl_down_sync(0xffffffff, tmp, 16);
  tmp += __shfl_down_sync(0xffffffff, tmp, 8);
  tmp += __shfl_down_sync(0xffffffff, tmp, 4);
  tmp += __shfl_down_sync(0xffffffff, tmp, 2);
  tmp += __shfl_down_sync(0xffffffff, tmp, 1);
  /*
#if CUDA_BLOCK_DIM == 128
  __shared__ double mm, nn, vv;
  if(threadIdx.x == 96)
    vv = tmp;
  if(threadIdx.x == 64)
    nn = tmp;
  if(threadIdx.x == 32)
    mm = tmp;
  __syncthreads();
  if(threadIdx.x == 0)
    tmp += mm + nn + vv;
#endif
#if CUDA_BLOCK_DIM == 64
  __shared__ double nn;
  if(threadIdx.x == 32)
    nn = tmp;
  __syncthreads();
  if(threadIdx.x == 0)
    tmp += nn;
#endif
  */
  int ma = CUDA_ma, dg_block = CUDA_Dg_block;
  double *__restrict__ dg = CUDA_Dg, *__restrict__ darea = CUDA_Darea, *__restrict__ nor = CUDA_Nor[nc];
#pragma unroll 1
  for (j = 1; j <= ma; j++)
  {
    int m = blockIdx() * dg_block + j * nf1;
    double dtmp = 0, dtmp2 = 0;
    if (j <= nco)
    {
      int mm = m + threadIdx.x + 1;

      i = threadIdx.x + 1;
      double *__restrict__ dgp = dg + mm;
      double *__restrict__ dareap = darea + i;
      double *__restrict__ norp = nor + i;

#pragma unroll 4
      while (i <= nf - CUDA_BLOCK_DIM)
      {
        double g0, g1, a0, a1, n0, n1;
        g0 = dgp[0];
        a0 = dareap[0];
        n0 = norp[0];
        g1 = dgp[CUDA_BLOCK_DIM];
        a1 = dareap[CUDA_BLOCK_DIM];
        n1 = norp[CUDA_BLOCK_DIM];
        dtmp += (g0 * a0) * n0;
        dtmp2 += (g1 * a1) * n1;
        i += 2 * CUDA_BLOCK_DIM;
        dgp += 2 * CUDA_BLOCK_DIM;
        dareap += 2 * CUDA_BLOCK_DIM;
        ;
        norp += 2 * CUDA_BLOCK_DIM;
      }
      // #pragma unroll 1
      if (i <= nf) //; i += CUDA_BLOCK_DIM, mm += CUDA_BLOCK_DIM)
      {
        dtmp += dgp[0] * dareap[0] * norp[0]; // CUDA_Dg[mm] * CUDA_Darea[i] * CUDA_Nor[nc][i];
      }

      dtmp += dtmp2;

      dtmp += __shfl_down_sync(0xffffffff, dtmp, 16);
      dtmp += __shfl_down_sync(0xffffffff, dtmp, 8);
      dtmp += __shfl_down_sync(0xffffffff, dtmp, 4);
      dtmp += __shfl_down_sync(0xffffffff, dtmp, 2);
      dtmp += __shfl_down_sync(0xffffffff, dtmp, 1);
    }

    if (threadIdx.x == 0)
      dyda[j - 1] = dtmp;
  }

  return (tmp);
}
