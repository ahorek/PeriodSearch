/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include <cuda.h>
#include <math.h>
#include "globals_CUDA.h"


__device__ void __forceinline__ curv(freq_context const * __restrict__ CUDA_LCC, double * __restrict__ cg, int bid)
{
  int i, m, n, l, k;
  double g;
  
  int numfac = CUDA_Numfac, nf1 = CUDA_Numfac1, mm = CUDA_Mmax, lm = CUDA_Lmax;
  i = threadIdx.x + 1;
  double * __restrict__ CUDA_Fcp = CUDA_Fc[0] + i;
  double * __restrict__ CUDA_Fsp = CUDA_Fs[0] + i;
  double * __restrict__ CUDA_Dareap = CUDA_Darea + i;

#pragma unroll 1
  while(i <= numfac)
    {
      g = 0;
      n = 0;
      double const * __restrict__ cgp = cg + 1;
      double const * __restrict__ fcp = CUDA_Fcp;
      double const * __restrict__ fsp = CUDA_Fsp;

#pragma unroll 2
      for(m = 0; m <= mm; m++)
	{ 
	  double fcim = __ldca(&fcp[0]); //* //[m*(MAX_N_FAC + 1)]; //CUDA_Fc[m][i];
	  double fsim = __ldca(&fsp[0]); //[m*(MAX_N_FAC + 1)]; //CUDA_Fs[m][i];
	  double * __restrict__ CUDA_Plegp = &CUDA_Pleg[m][m][i]; //[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
#pragma unroll 3
	  for(l = m; l <= lm; l++)
	    {
	      n++;
	      double fsum = __ldca(cgp++) * fcim; //CUDA_Fc[i][m];
	      if(m > 0)
		{
		  n++;
		  fsum += __ldca(cgp++) * fsim; //CUDA_Fs[i][m];
		}
	      g += CUDA_Plegp[0] * fsum; //[m][l][i] * fsum; //CUDA_Pleg[m][l][i] * fsum;
	      CUDA_Plegp += (MAX_N_FAC + 1);
	    }
	  fcp += MAX_N_FAC + 1;
	  fsp += MAX_N_FAC + 1;
	}
      double dd = CUDA_Dareap[0];
      g = exp(g);
      dd *= g;
      double * __restrict__ dgp = CUDA_LCC->Dg + (nf1 + i);
      double const * __restrict__ dsphp = CUDA_Dsph[0] + i + MAX_N_FAC + 1;
      
      Areag[bid][i] = dd;
      k = 1;
#pragma unroll 1
      while(k <= n - (UNRL - 1))
	{
	  double a[UNRL];

#pragma unroll 
	  for(int nn = 0; nn < UNRL; nn++)
	    {
	      a[nn] = __ldca(dsphp) * g;
	      dsphp += (MAX_N_FAC + 1);
	    }
#pragma unroll 
	  for(int nn = 0; nn < UNRL; nn++)
	    {
	      __stwb(dgp, a[nn]);
	      dgp += nf1;
	    }
	  k += UNRL;
	}
#pragma unroll 3
      while(k <= n)
	{
	  __stwb(dgp, dsphp[0] * g);
	  dsphp += (MAX_N_FAC + 1);
	  k++;
	  dgp += nf1;
	}

      i += CUDA_BLOCK_DIM;
      CUDA_Fcp += CUDA_BLOCK_DIM;
      CUDA_Fsp += CUDA_BLOCK_DIM;
      CUDA_Dareap += CUDA_BLOCK_DIM;
    }
  //__syncwarp();
}