#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}
#define SWAP4(a,b) {double x[4],y[4];for(int t1=0;t1<4;t1++) x[t1]=(a)[t1];for(int r1=0;r1<4;r1++) y[r1]=(b)[r1];for(int t2=0;t2<4;t2++)(b)[t2]=(x)[t2];for(int t3=0;t3<4;t3++)(a)[t3]=y[t3];}
#define SWAP8(a,b) {double x[8];for(int t1=0;t1<8;t1++) x[t1]=(a)[t1];for(int t2=0;t2<8;t2++)(a)[t2]=(b)[t2];for(int t3=0;t3<8;t3++)(b)[t3]=x[t3];}
#define SWAP4n(a,b,n) {double x[4],y[4];for(int t1=0;t1<4;t1++)x[t1]=(a)[t1*n];for(int r1=0;r1<4;r1++)y[r1]=(b)[r1*n];for(int t2=0;t2<4;t2++)(b)[t2*n]=x[t2];for(int t3=0;t3<4;t3++)(a)[t3*n]=y[t3];}
#define SWAP8n(a,b,n) {double x[8];for(int t1=0;t1<8;t1++)x[t1]=(a)[t1*n];for(int t2=0;t2<8;t2++)(a)[t2*n]=(b)[t2*n];for(int t3=0;t3<8;t3++)(b)[t3*n]=x[t3];}

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ int __forceinline__ gauss_errc(freq_context * __restrict__ CUDA_LCC, int ma)
{
  __shared__ int16_t sh_icol[N80]; //[CUDA_BLOCK_DIM];
  __shared__ int16_t sh_irow[N80]; //[CUDA_BLOCK_DIM];
  __shared__ double sh_big[N80]; //[CUDA_BLOCK_DIM];
  __shared__ double pivinv;
  __shared__ int icol;

  __shared__ int16_t indxr[MAX_N_PAR + 1];
  __shared__ int16_t indxc[MAX_N_PAR + 1];
  __shared__ int16_t ipiv[MAX_N_PAR + 1];

  int mf1 = CUDA_mfit1;
  int i, licol = 0, irow = 0, j, k, l, ll;
  double big, dum, temp;
  int mf = CUDA_mfit;
  
  j = threadIdx.x + 1;

#pragma unroll 9
  while(j <= mf)
    {
      ipiv[j] = 0;
      j += CUDA_BLOCK_DIM;
    }

  __syncwarp();

  double * __restrict__ covarp = CUDA_LCC->covar;

#pragma unroll 1
  for(i = 1; i <= mf; i++)
    {
      big = 0.0;
      irow = 0;
      licol = 0;
      j = threadIdx.x + 1;

#pragma unroll 2
      while(j <= mf)
	{
	  if(ipiv[j] != 1)
	    {
	      int ixx = j * mf1 + 1;
#pragma unroll 4
	      for(k = 1; k <= mf; k++, ixx++)
		{
		  int ii = ipiv[k];
		  if(ii == 0)
		    {
		      double tmpcov = fabs(__ldg(&covarp[ixx]));
		      if(tmpcov >= big)
			{
			  irow = j;
			  licol = k;
			  big = tmpcov;
			}
		    }
		  else if(ii > 1)
		    {
		      return(1);
		    }
		}
	    }
	  j += CUDA_BLOCK_DIM;
	}
      //      sh_big[threadIdx.x] = big;
      //      sh_irow[threadIdx.x] = irow;
      //      sh_icol[threadIdx.x] = licol;
      j = threadIdx.x;
      while(j <= mf)
	{      
	  sh_big[j] = big;
	  sh_irow[j] = irow;
	  sh_icol[j] = licol;
	  j += CUDA_BLOCK_DIM;
	}
      
      __syncwarp();
      
      if(threadIdx.x == 0)
	{
	  big = sh_big[0];
	  icol = sh_icol[0];
	  irow = sh_irow[0];
#pragma unroll 2
	  for(j = 1; j <= mf; j++)
	    {
	      if(sh_big[j] >= big)
		{
		  big = sh_big[j];
		  irow = sh_irow[j];
		  icol = sh_icol[j];
		}
	    }
	  ++(ipiv[icol]);

	  double * __restrict__ dapp = CUDA_LCC->da;

	  if(irow != icol)
	    {
	      double * __restrict__ cvrp = covarp + irow * mf1; 
	      double * __restrict__ cvcp = covarp + icol * mf1; 
#pragma unroll 4
	      for(l = 1; l <= mf - 3; l += 4)
		{
		  SWAP4(cvrp, cvcp);
		  cvrp += 4;
		  cvcp += 4;
		}
	      
#pragma unroll 3
	      for(; l <= mf; l++)
		{
		  SWAP(cvrp[0], cvcp[0]);
		  cvrp++;
		  cvcp++;
		}
	      
	      SWAP(dapp[irow], dapp[icol]);
	      //SWAP(b[irow],b[icol])
	    }
	  //CUDA_LCC->indxr[i] = irow;
	  indxr[i] = irow;
	  //CUDA_LCC->indxc[i] = icol;
	  indxc[i] = icol;
	  double cov = covarp[icol * mf1 + icol];
	  if(cov == 0.0) 
	    {
	      int bid = blockIdx();
	      j = 0;
	      
	      int    const * __restrict__ iap = CUDA_ia + 1;
	      double * __restrict__ atp = atry[bid] + 1; //CUDA_LCC->atry + 1;
	      double * __restrict__ cgp = CUDA_LCC->cg + 1;
	      double * __restrict__ dap = dapp;
#pragma unroll 4
	      for(int l = 1; l <= ma; l++)
		{
		  if(*iap)
		    {
		      dap++;
		      __stwb(atp,  *cgp + *dap);
		    }
		  iap++;
		  atp++;
		  cgp++;
		}
	      
	      return(2);
	    }
	  pivinv = ___drcp_rn(cov);
	  covarp[icol * mf1 + icol] = 1.0;
	  dapp[icol] *= pivinv;
	}
      
      __syncwarp();
      
      int x = threadIdx.x + 1;
      double * __restrict__ p = &covarp[icol * mf1];
#pragma unroll 2
      while(x <= mf)
	{
	  //if(x != 0)
	  __stwb(&p[x], __ldg(&p[x]) * pivinv);
	  x += CUDA_BLOCK_DIM;
	}
      
      __syncwarp();
      
#pragma unroll 2
      for(ll = 1; ll <= mf; ll++)
	if(ll != icol)
	  {
	    int ixx = ll * mf1, jxx = icol * mf1;
	    dum = __ldg(&covarp[ixx + icol]);
	    covarp[ixx + icol] = 0.0;
	    ixx++;
	    jxx++;
	    ixx += threadIdx.x;
	    jxx += threadIdx.x;
	    l = threadIdx.x + 1;
#pragma unroll 2
	    while(l <= mf)
	      {
		__stwb(&covarp[ixx],  __ldg(&covarp[ixx]) - __ldg(&covarp[jxx]) * dum);
		l += CUDA_BLOCK_DIM;
		ixx += CUDA_BLOCK_DIM;
		jxx += CUDA_BLOCK_DIM;
	      }
	    double *dapp = CUDA_LCC->da;
	    __stwb(&dapp[ll], __ldg(&dapp[ll]) - __ldg(&dapp[icol]) * dum);
	  }
      
      __syncwarp();
    }

  l = mf - threadIdx.x;

  while(l >= 1)
    {
      //int r = CUDA_LCC->indxr[l];
      int r = indxr[l];
      //int c = CUDA_LCC->indxc[l];
      int c = indxc[l];
      if(r != c)
	{
	  double * __restrict__ cvp1 = &(covarp[0]), * __restrict__ cvp2;
	  cvp2 = cvp1;
	  int i1 = mf1 + r;
	  int i2 = mf1 + c;
	  cvp1 = cvp1 + i1;
	  cvp2 = cvp2 + i2;
#pragma unroll 4
	  for(k = 1; k <= mf - 3; k += 4)
	    {
	      SWAP4n(cvp1, cvp2, mf1);
	      cvp1 += mf1 * 4;
	      cvp2 += mf1 * 4;
	    }
#pragma unroll 3
	  for(; k <= mf; k++)
	    {
	      SWAP(cvp1[0], cvp2[0]);
	      cvp1 += mf1;
	      cvp2 += mf1;
	    }
	}
      l -= CUDA_BLOCK_DIM;
    }

  __syncwarp();

  return(0);
}
#undef SWAP