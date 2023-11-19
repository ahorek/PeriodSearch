//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
//#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void __forceinline__ MrqcofCurve23I0IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid)
{
  int lpoints = 3;
  int mf1 = CUDA_mfit1;
  int l, jp, j, k, m, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, wght, ltrial_chisq;
  __shared__ double dyda[BLOCKX4][N80];
  double * __restrict__ dydap = dyda[threadIdx.y];
  //__syncthreads();

  if(threadIdx.x == 0)
    {
      npg1[bid] += lpoints;
    }

  lnp2 = npg2[bid];
  ltrial_chisq = trial_chisqg[bid];

  int ma = CUDA_ma, lma = CUDA_lastma;
  int lastone = CUDA_lastone;
  int * __restrict__ iapp = CUDA_ia;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;;
  
#pragma unroll 
  for(jp = 1; jp <= lpoints; jp++)
    {
      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ bad, strided read, BAD
      double * __restrict__ c = &(dytemp[ixx]);
      l = threadIdx.x;
#pragma unroll 2
      while(l < ma)
	{
	  dydap[l] = __ldca(c); // YYYY
	  l += CUDA_BLOCK_DIM;
	  c += CUDA_BLOCK_DIM * Lpoints1;
	}
      
      __syncwarp();
      
      lnp2++;
      double s = __ldg(&CUDA_sig[lnp2]);
      ymod = __ldca(&ytemp[jp]);
      sig2i = ___drcp_rn(s * s);
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;
      
      j = 0;
      double sig2iwght = sig2i * wght;
      
#pragma unroll 
      for(l = 2; l <= lastone; l++)
	{
	  j++;
	  wt = dydap[l-1] * sig2iwght;
	  
	  int xx = threadIdx.x + 1;
	  double * __restrict__ alp = &alpha[j * mf1 + xx - 1];
#pragma unroll 2
	  while(xx <= l)
	    {
	      //if(xx != 0)
	      double const * __restrict__ alp2 = alp;
	      __stwb(alp, *alp2 + wt * dydap[xx-1]);
	      xx  += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } /* m */
	  //__syncthreads();
	  if(threadIdx.x == 0)
	    {
	      beta[j] = beta[j] + dy * wt;
	    }
	  //__syncthreads();
	} /* l */
      
#pragma unroll 
      for(; l <= lma; l++)
	{
	  if(iapp[l])
	    {
	      j++;
	      wt = dydap[l-1] * sig2iwght;
	      
	      int xx = threadIdx.x + 1;
	      double * __restrict__ alph = &alpha[j * mf1 - 1];
#pragma unroll 2
	      while(xx <= lastone)
		{
		  //if(xx != 0)
		  __stwb(&alph[xx], alph[xx] + wt * dydap[xx-1]);
		  xx += CUDA_BLOCK_DIM;
		} /* m */
	      //__syncthreads();
	      if(threadIdx.x == 0)
		{
		  k = lastone - 1;
		  m = lastone + 1;
		  int * __restrict__ iap = iapp + m;
		  double * __restrict__ alp = alpha + j * mf1 + k;
#pragma unroll 4
		  for(; m <= l; m++)
		    {
		      if(*iap)
			{
			  alp++;
			  __stwb(alp, *alp + wt * dydap[m-1]);
			}
		      iap++;
		    } /* m */
		  beta[j] = beta[j] + dy * wt;
		}
	      //__syncthreads();
	    }
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg2[bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}

__device__ void __forceinline__ MrqcofCurve23I0IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid)
{
  int lpoints = 3;
  int mf1 = CUDA_mfit1;
  int l, jp, j, k, m, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, wght, ltrial_chisq;
  __shared__ double dyda[N80];
  
  __syncwarp();

  if(threadIdx.x == 0)
    {
      npg1[bid] += lpoints;
    }

  lnp2 = npg2[bid];
  ltrial_chisq = trial_chisqg[bid];

  int ma = CUDA_ma, lma = CUDA_lastma;
  int lastone = CUDA_lastone;
  int * __restrict__ iapp = CUDA_ia;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
#pragma unroll 
  for(jp = 1; jp <= lpoints; jp++) 
    {
      lnp2++;
      double s = __ldg(&CUDA_sig[lnp2]);
      ymod = __ldca(&(ytemp[jp]));
      sig2i = ___drcp_rn(s * s); 
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;
      
      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ, bad, strided read, BAD!
      double * __restrict__ c = &(dytemp[ixx]); //  bad c
      l = threadIdx.x + 1;
#pragma unroll 4
      while(l <= ma - CUDA_BLOCK_DIM)
	{
	  double a, b;
	  a = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  b = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  dyda[l-1] = a;
	  dyda[l-1 + CUDA_BLOCK_DIM] = b;
	  l += 2*CUDA_BLOCK_DIM;
	}
#pragma unroll 1
      while(l <= ma)
	{
	  dyda[l-1] = __ldca(c);
	  l += CUDA_BLOCK_DIM;
	  c += CUDA_BLOCK_DIM * Lpoints1;
	}
      
      __syncwarp();
      
      j = 0;
      double sig2iwght = sig2i * wght;
      
#pragma unroll 4
      for(l = 1; l <= lastone; l++)
	{
	  j++;
	  wt = dyda[l-1] * sig2iwght;
	  int xx = threadIdx.x + 1;
#pragma unroll 2
	  while(xx <= l)
	    {
	      alpha[j * mf1 + xx] += wt * dyda[xx-1];
	      xx += CUDA_BLOCK_DIM;
	    } /* m */
	  //__syncthreads();
	  if(threadIdx.x == 0)
	    {
	      beta[j] = beta[j] + dy * wt;
	    }
	  //__syncthreads();
	} /* l */
      
#pragma unroll 4
      for(; l <= lma; l++)
	{
	  if(iapp[l])
	    {
	      j++;
	      wt = dyda[l-1] * sig2iwght;
	      int xx = threadIdx.x + 1;
#pragma unroll 2
	      while(xx <= lastone)
		{
		  //if(xx != 0)
		  alpha[j * mf1 + xx] += wt * dyda[xx-1];
		  xx += CUDA_BLOCK_DIM;
		} /* m */
	      //__syncthreads();
	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone + 1;
		  int * __restrict__ iap = iapp + m;
		  double * __restrict__ alp = alpha + j * mf1 + k;
#pragma unroll 4
		  for(; m <= l; m++)
		    {
		      if(*iap)
			{
			  alp++;
			  __stwb(alp, *alp + wt * dyda[m-1]);
			}
		      iap++;
		    } /* m */
		  beta[j] = beta[j] + dy * wt;
		}
	      //__syncthreads();
	    }
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg2[bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}

// SLOW (only 3 threads participate -> 1/10 perf))
  __device__ void __forceinline__ MrqcofCurve23I1IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid)
{
  int lpoints = 3;
  int mf1 = CUDA_mfit1;
  int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  __shared__ double dydat[3][N80];
  
  lnp1 = npg1[bid] + threadIdx.x + 1;

  int ma = CUDA_ma;
  //int bid = blockIdx();
  jp = threadIdx.x + 1;
  double rave = ___drcp_rn(aveg[bid]);
  double * __restrict__ dytmpp = CUDA_LCC->dytemp, * __restrict__ cuda_sig = CUDA_sig, * __restrict__ ytemp = CUDA_LCC->ytemp;
  double * __restrict__ cuda_weight = CUDA_Weight, * __restrict__ cuda_brightness = CUDA_brightness;
  //double * __restrict__ dave = CUDA_LCC->dave;
  double * __restrict__ davep = &(dave[bid][0]);
  long int lpadd = sizeof(double) * Lpoints1;
  
  //#pragma unroll 
  if(jp <= lpoints)
    {
      int ixx = jp + Lpoints1;
      // Set the size scale coeff. deriv. explicitly zero for relative lcurves 
      dytmpp[ixx] = 0; // YYY, good, consecutive
      coef = cuda_sig[lnp1] * lpoints * rave; // / CUDA_LCC->ave;
      
      double yytmp = ytemp[jp];
      coef1 = yytmp * rave; // / CUDA_LCC->ave;
      ytemp[jp] = coef * yytmp;
      
      ixx += Lpoints1;
      double * __restrict__ dyp = dytmpp + ixx; //&(CUDA_LCC->dytemp[ixx]);
      double * __restrict__ dap = &(davep[2]);
#pragma unroll 
      for(l = 2; l <= ma - (UNRL - 1); l += UNRL, ixx += UNRL * Lpoints1)
	{
	  double dd[UNRL], dy[UNRL];
	  int ii;
	  double * __restrict__ dypp = dyp;
	  for(ii = 0; ii < UNRL; ii++)
	    {
	      dy[ii] = __ldg(dypp);
	      //dypp += Lpoints1;
	      dypp = (double *)(((char *)dypp) + lpadd);

	      dd[ii] = __ldca(dap);
	      dap++;
	    }
	  for(ii = 0; ii < UNRL; ii++)
	    {
	      __stwb(dyp, coef * (dy[ii] - coef1 * dd[ii])); //WXX
	      //dyp += Lpoints1;
	      dyp = (double *)(((char *)dyp) + lpadd);
	    }
	}
#pragma unroll 
      for(; l <= ma; l++, dyp += Lpoints1, dap++)
	__stwb(dyp, coef * ( __ldg(dyp) - coef1 * __ldca(dap))); //WXX
	//*dyp = __ldg(dyp) * coef - coef1 * __ldg(dap);

      jp += CUDA_BLOCK_DIM;
      lnp1 += CUDA_BLOCK_DIM;
    }

  __syncwarp();

  if(threadIdx.x == 0)
    {
      npg1[bid] += lpoints;
    }

  lnp2 = npg2[bid];
  ltrial_chisq = trial_chisqg[bid];
  int lastone = CUDA_lastone;
  int * __restrict__ iapp = CUDA_ia;

#pragma unroll 
  for(jp = 1; jp <= lpoints; jp++)
    {
      if(jp == 1)
	{
	  int ixx = jp + (threadIdx.x + 1) * Lpoints1; // RXX bad, strided read, BAD
	  double * __restrict__ c = dytmpp + ixx;  //&(CUDA_LCC->dytemp[ixx]);
	  l = threadIdx.x;
#pragma unroll 2
	  while(l < ma)
	    {
	      dydat[0][l] = c[0]; // YYYY RXX
	      dydat[1][l] = c[1]; // YYYY
	      dydat[2][l] = c[2]; // YYYY
	      l += CUDA_BLOCK_DIM;
	      c += CUDA_BLOCK_DIM * Lpoints1;
	    }
	  __syncwarp();
	}
      
      double * __restrict__ dyda = &dydat[jp-1][0];
      
      lnp2++;
      double s = cuda_sig[lnp2];
      ymod = ytemp[jp];
      sig2i = ___drcp_rn(s * s);
      wght = cuda_weight[lnp2];
      dy = cuda_brightness[lnp2] - ymod;
      
      j = 0;
      double sig2iwght = sig2i * wght;

      double * __restrict__ dydap = dyda + 1;
#pragma unroll 
      for(l = 2; l <= lastone; l++)
	{
	  j++;
	  wt = *dydap * sig2iwght;
	  dydap++;
	  
	  int xx = threadIdx.x + 1;
	  double * __restrict__ alp = &(alpha[j * mf1 - 1 + xx]);
#pragma unroll 2
	  while(xx <= l)
	    {
	      //if(xx != 0)
	      double * __restrict__ alp2 = alp;
	      __stwb(alp, *alp2 + wt * dyda[xx-1]);
	      xx += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } /* m */
	  //__syncthreads();
	  if(threadIdx.x == 0)
	    {
	      beta[j] += dy * wt;
	    }
	  //__syncthreads();
	} /* l */
      
#pragma unroll 
      for(; l <= CUDA_lastma; l++)
	{
	  if(iapp[l])
	    {
	      j++;
	      wt = *dydap * sig2iwght;
	      
	      int xx = threadIdx.x + 1;
	      double * __restrict__ alp = &alpha[j * mf1 - 1];
#pragma unroll 2
	      while(xx <= lastone)
		{
		  //if(xx != 0)
		  double const * __restrict__ alp2 = alp;
		  __stwb(alp, *alp2 + wt * dyda[xx-1]);
		  xx += CUDA_BLOCK_DIM;
		  alp += CUDA_BLOCK_DIM;
		} /* m */
	      //__syncthreads();
	      if(threadIdx.x == 0)
		{
		  k = lastone - 1;
		  m = lastone + 1;
		  int * __restrict__ iap = iapp + m;
		  double * __restrict__ alp = alpha + j * mf1 + k;
#pragma unroll 4
		  for(; m <= l; m++)
		    {
		      if(*iap)
			{
			  //k++;
			  alp++;
			  double const * __restrict__ alp2 = alp;
			  __stwb(alp, *alp2 + wt * dyda[m - 1]);
			}
		      iap++;
		    } /* m */
		  beta[j] = beta[j] + dy * wt;
		}
	      //__syncthreads();
	    }
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg2[bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}

__device__ void __forceinline__ MrqcofCurve23I1IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid)
{
  int lpoints = 3;
  int mf1 = CUDA_mfit1;
  //int bid = blockIdx();
  int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  __shared__ double dyda[N80];
  
  lnp1 = npg1[bid] + 1;

  int ma = CUDA_ma;
  double rave = ___drcp_rn(aveg[bid]);
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 
  for(jp = 1; jp <= lpoints; jp++, lnp1++)
    {
      int ixx = jp + Lpoints1;
      // Set the size scale coeff. deriv. explicitly zero for relative lcurves 
      dytemp[ixx] = 0; // YYY, good?, same for all threads??
      coef = __ldg(&CUDA_sig[lnp1]) * lpoints * rave; // / CUDA_LCC->ave;
      
      double yytmp = ytemp[jp];
      coef1 = yytmp * rave; // / CUDA_LCC->ave;
      ytemp[jp] = coef * yytmp;
      
      ixx += Lpoints1;
      double * __restrict__ dyp = &(dytemp[ixx]);
      double * __restrict__ dap = &(dave[bid][2]);
      l = 2 + threadIdx.x;
#pragma unroll 2
      while(l <= ma)
	{
	  double dy = __ldg(dyp);
	  double dd = __ldca(dap);
	  dap += CUDA_BLOCK_DIM;
	  __stwb(dyp, coef * (dy - coef1 * dd));
	  dyp += Lpoints1 * CUDA_BLOCK_DIM;
	  l += CUDA_BLOCK_DIM;
	  ixx += CUDA_BLOCK_DIM * Lpoints1;
	}
    }

  __syncwarp();

  if(threadIdx.x == 0)
    {
      npg1[bid] += lpoints;
    }

  lnp2 = npg2[bid];
  ltrial_chisq = trial_chisqg[bid];

  int lastone = CUDA_lastone, lma = CUDA_lastma;
  int * __restrict__ iapp = CUDA_ia;
  
#pragma unroll 
  for(jp = 1; jp <= lpoints; jp++) 
    {
      lnp2++;
      double s = __ldg(&CUDA_sig[lnp2]);
      ymod = __ldca(&(ytemp[jp]));
      sig2i = ___drcp_rn(s * s); 
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;
      
      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ, bad, strided read, BAD!
      double * __restrict__ c = &(dytemp[ixx]); //  bad c
      l = threadIdx.x + 1;
#pragma unroll 4
      while(l <= ma - CUDA_BLOCK_DIM)
	{
	  double a, b;
	  a = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  b = __ldca(c);
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  dyda[l-1] = a;
	  dyda[l-1 + CUDA_BLOCK_DIM] = b;
	  l += 2*CUDA_BLOCK_DIM;
	}
#pragma unroll 1
      while(l <= ma)
	{
	  dyda[l-1] = __ldca(c);
	  l += CUDA_BLOCK_DIM;
	  c += CUDA_BLOCK_DIM * Lpoints1;
	}
      
      __syncwarp();
      
      j = 0;
      double sig2iwght = sig2i * wght;
      
#pragma unroll 4
      for(l = 1; l <= lastone; l++)
	{
	  j++;
	  wt = dyda[l-1] * sig2iwght;
	  int xx = threadIdx.x + 1;
	  double * __restrict__ alp = &alpha[j * mf1 + xx]; 
#pragma unroll 2
	  while(xx <= l)
	    {
	      double const * __restrict alp2 = alp;
	      __stwb(alp, *alp2 +  wt * dyda[xx-1]);
	      xx += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } /* m */
	  //__syncthreads();
	  if(threadIdx.x == 0)
	    {
	      beta[j] = beta[j] + dy * wt;
	    }
	  //__syncthreads();
	} /* l */
      
#pragma unroll 4
      while(l <= lma)
	{
	  if(iapp[l])
	    {
	      j++;
	      wt = dyda[l-1] * sig2iwght;
	      int xx = threadIdx.x + 1;
	      double * __restrict__ alp = &alpha[j * mf1 + xx]; 
#pragma unroll 2
	      while(xx <= lastone)
		{
		  //if(xx != 0)
		  double const * __restrict alp2 = alp;
		  __stwb(alp, *alp2 + wt * dyda[xx-1]);
		  xx += CUDA_BLOCK_DIM;
		  alp += CUDA_BLOCK_DIM;
		} /* m */
	      //__syncthreads();
	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone + 1;
		  int * __restrict__ iap = iapp + m;
		  double * __restrict__ alp = alpha + j * mf1 + k;
#pragma unroll 4
		  while(m <= l)
		    {
		      if(*iap)
			{
			  alp++;
			  __stwb(alp, *alp + wt * dyda[m-1]);
			}
		      iap++;
		      m++;
		    } /* m */
		  beta[j] = beta[j] + dy * wt;
		}
	      //__syncthreads();
	    }
	  l++;
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */
  
  if(threadIdx.x == 0)
    {
      npg2[bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}

__device__ void __forceinline__ MrqcofCurve2I0IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid)
{
  //inrel = 0;
  int l, jp, j, /*k, m,*/ lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, wght, ltrial_chisq;
  int mf1 = CUDA_mfit1;
  
  __shared__ double dydat[4][N80];
  
  __syncwarp(); // remove sync ?

  if(threadIdx.x == 0)
    {
      npg1[bid] += lpoints;
    }

  lnp2 = npg2[bid];
  ltrial_chisq = trial_chisqg[bid];

  int ma = CUDA_ma, lma = CUDA_lastma;
  int lastone = CUDA_lastone;
  int * __restrict__ iapp = CUDA_ia;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 2
  for(jp = 1; jp <= lpoints; jp++)
    {
      if(((jp-1)&3) == 0)
	{
	  int tid = threadIdx.x >> 2;
	  int u = threadIdx.x & 3;
	  int ixx = jp + (tid + 1) * Lpoints1; // ZZZ bad, strided read dytemp, BAD
	  double * __restrict__ c = &(dytemp[ixx]);//, *dddc = ddd + ixx;
	  c += u;
	  l = tid;
#pragma unroll 4
	  while(l < ma)
	    {
#ifdef DYTEMP_NEW
	      dydat[u][l] = dytemp2[bid][jp][(l << 2) + u + 1];
#else
	      dydat[u][l] = __ldca(c); //*dddc //__ldca(c); // YYYY
#endif
	      l += CUDA_BLOCK_DIM/4;
	      c += CUDA_BLOCK_DIM/4 * Lpoints1;
	    }
	}
      __syncwarp();
      
      double * __restrict__ dyda = &(dydat[(jp-1) & 3][0]);	  

      /*
      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ bad, strided read, BAD
      double *c = &(CUDA_LCC->dytemp[ixx]);
#pragma unroll 2
      for(l = threadIdx.x; l < ma; l += CUDA_BLOCK_DIM, c += CUDA_BLOCK_DIM * Lpoints1)
	dyda[l] = __ldca(c); // YYYY
      */  

      lnp2++;
      double s = __ldg(&CUDA_sig[lnp2]);
      ymod = __ldca(&ytemp[jp]);
      sig2i = ___drcp_rn(s * s);
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;

      j = 0;
      double sig2iwght = sig2i * wght;

#pragma unroll 2
      for(l = 2; l <= lastone; l++)
	{
	  j++;
	  wt = dyda[l-1] * sig2iwght;

	  int xx = threadIdx.x + 1;
	  double * __restrict__ alph = (&alpha[j * mf1 - 1]) + xx;
#pragma unroll 2
	  while(xx <= l)
	    {
	      //if(xx != 0)
	      //alpha[j * mf1 - 1 + xx] += wt * dyda[xx-1];
	      double const * __restrict__ alph2 = alph;
	      __stwb(alph, __ldca(alph2) + wt * dyda[xx-1]); //ldg
	      //*alpha += wt * dyda[xx-1];
	      //alpha += CUDA_BLOCK_DIM;
	      //xx += CUDA_BLOCK_DIM;
	      alph  += CUDA_BLOCK_DIM;
	    } /* m */
	  //__syncthreads();
	  if(threadIdx.x == 0)
	    {
	      beta[j] = beta[j] + dy * wt;
	    }
	  //__syncthreads();
	  l++;
	} /* l */
	  
#pragma unroll 1
      while(l <= lma)
	{
	  if(iapp[l])
	    {
	      j++;
	      wt = dyda[l-1] * sig2iwght;

	      int xx = threadIdx.x + 1;
	      double * __restrict__ alph = &alpha[j * mf1 - 1 + xx]; // + xx;
#pragma unroll 2
	      while(xx <= lastone)
		{
		  //if(xx != 0)
		  double const * __restrict__ alph2 = alph;
		  __stwb(alph, __ldca(alph2) + wt * dyda[xx-1]); //ldg
		  //*alpha += wt * dyda[xx-1];
		  //alpha += CUDA_BLOCK_DIM;
		  alph  += CUDA_BLOCK_DIM;
		} /* m */
	      //__syncthreads();
	      if(threadIdx.x == 0)
		{
		  int k = lastone - 1;
		  int m = lastone + 1;
		  int * __restrict__ iap = iapp + m;
		  double * __restrict__ alp = alpha + j * mf1 + k;
		  beta[j] = beta[j] + dy * wt;
#pragma unroll 4
		  while(m <= l)
		    {
		      if(*iap)
			{
			  //k++;
			  alp++;
			  double const * __restrict__ alp2 = alp;
			  __stwb(alp, __ldca(alp2) + wt * dyda[m - 1]);
			}
		      iap++;
		      m++;
		    } /* m */
		}
	      //__syncthreads();
	    }
	  l++;
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg2[bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}

// SLOWW
__device__ void __forceinline__ MrqcofCurve2I1IA0(freq_context *__restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid)
{
  int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  int mf1 = CUDA_mfit1;
  __shared__ double dydat[4][N80];
  //__shared__ double ddd[4500];
  
  lnp1 = npg1[bid] + threadIdx.x + 1;

  int ma = CUDA_ma;
  //int bid = blockIdx();
  jp = threadIdx.x + 1;
  double rave = ___drcp_rn(aveg[bid]);
  double * __restrict__ dytempp = CUDA_LCC->dytemp, * __restrict__ ytempp = CUDA_LCC->ytemp;
  double * __restrict__ cuda_sig = CUDA_sig;
  //double * __restrict__ davep = CUDA_LCC->dave;
  double * __restrict__ davep = &(dave[bid][0]);
  long int lpadd = sizeof(double) * Lpoints1;
  
#pragma unroll 1
  while(jp <= lpoints)
    {
      int ixx = jp + Lpoints1;
      // Set the size scale coeff. deriv. explicitly zero for relative lcurves 
      dytempp[ixx] = 0; // YYY, good, consecutive

      //ddd[ixx] = 0;
      coef = __ldca(&cuda_sig[lnp1]) * lpoints * rave; // / CUDA_LCC->ave;
      
      double yytmp = __ldca(&ytempp[jp]);
      coef1 = yytmp * rave; // / CUDA_LCC->ave;
      ytempp[jp] = coef * yytmp;
      
      ixx += Lpoints1;
      double * __restrict__ dyp = &(dytempp[ixx]), *__restrict__ dypp; //, *ddyp = ddd + ixx, *ddypp; 
      double * __restrict__ dap = &(davep[2]);
      //dypp = dyp;

#pragma unroll 1
      for(l = 2; l <= ma - (2 - 1); l += 2, ixx += 2 * Lpoints1)
	{
	  double dd[2], dy[2];
	  int ii;
	  dypp = dyp;
	  //ddypp = ddyp;
	  for(ii = 0; ii < 2; ii++)
	    {
	      dy[ii] = __ldca(dypp);
	      //dypp += Lpoints1;
	      dypp = (double *)(((char *)dypp) + lpadd);

	      //ddyp += Lpoints1;
	      dd[ii] = __ldca(dap);
	      dap++;
	    }
	  for(ii = 0; ii < 2; ii++)
	    {
	      double d = coef * (dy[ii] - coef1 * dd[ii]);
	      *dyp = d;
	      //dyp += Lpoints1;
	      dyp = (double *)(((char *)dyp) + lpadd);

	      //*ddypp = d;
	      //ddypp += Lpoints1;
	    }
	}
#pragma unroll 
      while(l <= ma)
	{
	  double d = coef * __ldca(&dyp[0]) - coef1 * __ldca(&dap[0]);
	  *dyp = d;
	  l++;
	  //dyp += Lpoints1;
	  dyp = (double *)(((char *)dyp) + lpadd);

	  dap++;
	}
      jp += CUDA_BLOCK_DIM;
      lnp1 += CUDA_BLOCK_DIM;
    }
  
  __syncwarp();
  
  if(threadIdx.x == 0)
    {
      npg1[bid] += lpoints;
    }
  
  lnp2 = npg2[bid];
  ltrial_chisq = trial_chisqg[bid];
  
  int lastone = CUDA_lastone, lma = CUDA_lastma;
  int * __restrict__ iapp = CUDA_ia;
  double * __restrict__ cuda_weight = CUDA_Weight, * __restrict__ cuda_brightness = CUDA_brightness;
  
#pragma unroll 4
  for(jp = 1; jp <= lpoints; jp++)
    {
      if(((jp-1)&3) == 0)
	{
	  int tid = threadIdx.x >> 2;
	  int u = threadIdx.x & 3;
	  int ixx = jp + (tid + 1) * Lpoints1; // ZZZ bad, strided read dytemp, BAD
	  double * __restrict__ c = &(dytempp[ixx]);//, *dddc = ddd + ixx;
	  c += u;
	  l = tid;
#pragma unroll 4
	  while(l < ma)
	    {
#ifdef DYTEMP_NEW
	      dydat[u][l] = dytemp2[bid][jp][(l << 2) + u + 1];
#else
	      dydat[u][l] = __ldca(c); //*dddc //__ldca(c); // YYYY
#endif
	      l += CUDA_BLOCK_DIM/4;
	      c += CUDA_BLOCK_DIM/4 * Lpoints1;
	    }
	}
      __syncwarp();
      double * __restrict__ dyda = &(dydat[(jp-1) & 3][0]);	  
      lnp2++;
      double s = cuda_sig[lnp2];
      ymod = ytempp[jp];
      sig2i = ___drcp_rn(s * s);
      wght = cuda_weight[lnp2];
      dy = cuda_brightness[lnp2] - ymod;

      j = 0;
      double sig2iwght = sig2i * wght;


#pragma unroll 2
      for(l = 2; l <= lastone; l++)
	{
	  j++;
	  wt = dyda[l-1] * sig2iwght;

	  int xx = threadIdx.x;
	  double *__restrict__ alph = &alpha[j * mf1 + xx];
#pragma unroll 2
	  while(xx < l)
	    {
	      //if(xx != 0)
	      __stwb(alph, __ldca(alph) +  wt * dyda[xx]);
	      alph += CUDA_BLOCK_DIM;
	      xx += CUDA_BLOCK_DIM;
	    } /* m */
	  //__syncthreads();
	  if(threadIdx.x == 0)
	    {
	      beta[j] = beta[j] + dy * wt;
	    }
	  //	  __syncthreads();
	} /* l */
	  
#pragma unroll 1
      for(; l <= lma; l++)
	{
	  if(iapp[l])
	    {
	      j++;
	      wt = dyda[l-1] * sig2iwght;

	      int xx = threadIdx.x;
	      double * __restrict__ alph = &alpha[j * mf1 + xx];
#pragma unroll 2
	      while(xx < lastone)
		{
		  //if(xx != 0)
		  __stwb(alph, __ldca(alph) + wt * dyda[xx]);
		  alph += CUDA_BLOCK_DIM;
		  xx += CUDA_BLOCK_DIM;
		} /* m */
	      //__syncthreads();
	      if(threadIdx.x == 0)
		{
		  k = lastone - 1;
		  m = lastone + 1;
		  int * __restrict__ iap = iapp + m;
		  double * __restrict__ alp = &(alpha[j * mf1 + k]);
#pragma unroll 4
		  for(; m <= l; m++)
		    {
		      if(*iap)
			{
			  //k++;
			  ++alp;
			  __stwb(alp, __ldca(alp) + wt * dyda[m-1]);
			}
		      iap++;
		    } /* m */
		  beta[j] +=  dy * wt;
		}
	      //__syncthreads();
	    }
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg2[bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}

__device__ void __forceinline__ MrqcofCurve2I0IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid)
{
  int l, jp, j, k, m, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, wght, ltrial_chisq;
  int mf1 = CUDA_mfit1;
  __shared__ double dyda[N80];
  
  __syncwarp(); // remove

  if(threadIdx.x == 0)
    {
      npg1[bid] += lpoints;
    }

  lnp2 = npg2[bid];
  ltrial_chisq = trial_chisqg[bid];

  int ma = CUDA_ma, lma = CUDA_lastma;
  int lastone = CUDA_lastone;
  int * __restrict__ iapp = CUDA_ia;
  double * __restrict__ dytemp = CUDA_LCC->dytemp, *ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 2
  for(jp = 1; jp <= lpoints; jp++) // CHANGE LOOP threadIdx.x ?
    {
      lnp2++;
      double s = __ldg(&CUDA_sig[lnp2]);
      ymod = __ldca(&(ytemp[jp]));
      sig2i = ___drcp_rn(s * s); 
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;

      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ, bad, strided read, BAD!
      double * __restrict__ c = &(dytemp[ixx]); //  bad c
      l = threadIdx.x + 1;
#pragma unroll 2
      while(l <= ma - CUDA_BLOCK_DIM)
	{
	  double a, b;
#ifdef DYTEMP_NEW
	  a = dytemp2[bid][jp][l];
#else
	  a = __ldca(c);
#endif
	  c += CUDA_BLOCK_DIM * Lpoints1;
#ifdef DYTEMP_NEW
	  b = dytemp2[bid][jp][l + CUDA_BLOCK_DIM];
#else
	  b = __ldca(c);
#endif
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  dyda[l-1] = a;
	  dyda[l-1 + CUDA_BLOCK_DIM] = b;
	  l += 2*CUDA_BLOCK_DIM;
	}
      //#pragma unroll 1
      //for( ; l <= ma; l += CUDA_BLOCK_DIM, c += CUDA_BLOCK_DIM * Lpoints1)
      if(l <= ma)
#ifdef DYTEMP_NEW
	dyda[l - 1] = dytemp2[bid[jp][l];
#else
	dyda[l-1] = __ldca(c);
#endif
	    
      __syncwarp();

      j = 0;
      double sig2iwght = sig2i * wght;

#pragma unroll 2
      for(l = 1; l <= lastone; l++)
	{
	  j++;
	  wt = dyda[l-1] * sig2iwght;
	  int xx = threadIdx.x + 1;
	  double * __restrict__ alp = alpha + mf1 + xx;
#pragma unroll 2
	  while(xx <= l)
	    {
	      double const * __restrict__ alp2 = alp;
	      __stwb(alp, __ldca(alp2) + wt * dyda[xx-1]);
	      alp += mf1;
	      xx += CUDA_BLOCK_DIM;
	    } /* m */
	  //__syncthreads();
	  if(threadIdx.x == 0)
	    {
	      beta[j] = beta[j] + dy * wt;
	    }
	  //__syncthreads();
	} /* l */
	  
#pragma unroll 2
      for(; l <= lma; l++)
	{
	  if(iapp[l])
	    {
	      j++;
	      wt = dyda[l-1] * sig2iwght;
	      int xx = threadIdx.x + 1;
	      double * __restrict__ alp = &alpha[j * mf1 + xx];
#pragma unroll 2
	      while(xx <= lastone)
		{
		  //if(xx != 0)
		  //alpha[j * mf1 + xx] += wt * dyda[xx-1];
		  double const * __restrict__ alp2 = alp;
		  __stwb(alp, __ldca(alp2) + wt * dyda[xx-1]);
		  xx += CUDA_BLOCK_DIM;
		  alp  += CUDA_BLOCK_DIM;
		} /* m */
	      //__syncthreads();
	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone + 1;
		  int * __restrict__ iap = iapp + m;
		  double * __restrict__ alp = alpha + j * mf1 + k;
#pragma unroll 4
		  for(; m <= l; m++)
		    {
		      if(*iap)
			{
			  alp++;
			  __stwb(alp, __ldca(alp) + wt * dyda[m-1]);
			}
		      iap++;
		    } /* m */
		  beta[j] = beta[j] + dy * wt;
		}
	      //__syncthreads();
	    }
	} /* l */
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } /* jp */

  if(threadIdx.x == 0)
    {
      npg2[bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}

// WORKING, SLOW
  __device__ void __forceinline__ MrqcofCurve2I1IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid)
{
  int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
  double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
  int mf1 = CUDA_mfit1;
  __shared__ double dyda[N80];
  
  lnp1 = npg1[bid] + threadIdx.x + 1;
  
  int ma = CUDA_ma;
  //int bid = blockIdx();
  jp = threadIdx.x + 1;
  double rave = ___drcp_rn(aveg[bid]);
  double * __restrict__ dytemp = CUDA_LCC->dytemp, * __restrict__ ytemp = CUDA_LCC->ytemp;
  
#pragma unroll 1
  while(jp <= lpoints)
    {
      int ixx = jp + Lpoints1;
      // Set the size scale coeff. deriv. explicitly zero for relative lcurves 
      dytemp[ixx] = 0; // YYY, good, consecutive
      coef = __ldg(&CUDA_sig[lnp1]) * lpoints * rave; // / CUDA_LCC->ave;

      double yytmp = ytemp[jp];
      coef1 = yytmp * rave; // / CUDA_LCC->ave;
      ytemp[jp] = coef * yytmp;

      ixx += Lpoints1;
      double * __restrict__ dyp = &(dytemp[ixx]);
      double * __restrict__ dap = &(dave[bid][2]);
#pragma unroll 2
      for(l = 2; l <= ma - (UNRL - 1); l += UNRL, ixx += UNRL * Lpoints1)
	{
	  double dd[UNRL], dy[UNRL];
	  int ii;
	  double * __restrict__ dypp = dyp;
#pragma unroll 4
	  for(ii = 0; ii < UNRL; ii++)
	    {
	      dy[ii] = __ldca(dyp);
	      dyp += Lpoints1;
	      dd[ii] = __ldg(dap);
	      dap++;
	    }
#pragma unroll 4
	  for(ii = 0; ii < UNRL; ii++)
	    {
	      __stwb(dypp, coef * (dy[ii] - coef1 * dd[ii]));
	      dypp += Lpoints1;
	    }
	}
#pragma unroll 3
      for(; l <= ma; l++, dyp += Lpoints1, dap++)
	__stwb(dyp, __ldca(dyp) * coef - coef1 * __ldg(dap));
      
      jp += CUDA_BLOCK_DIM;
      lnp1 += CUDA_BLOCK_DIM;
    }

  __syncwarp();

  if(threadIdx.x == 0)
    {
      npg1[bid] += lpoints;
    }

  lnp2 = npg2[bid];
  ltrial_chisq = trial_chisqg[bid];

  int lastone = CUDA_lastone, lma = CUDA_lastma;
  int * __restrict__ iapp = CUDA_ia;
  
#pragma unroll 2
  for(jp = 1; jp <= lpoints; jp++) // CHANGE LOOP threadIDx.x ?
    {
      lnp2++;
      double s = __ldg(&CUDA_sig[lnp2]);
      ymod = __ldca(&(ytemp[jp]));
      sig2i = ___drcp_rn(s * s); 
      wght = __ldg(&CUDA_Weight[lnp2]);
      dy = __ldg(&CUDA_brightness[lnp2]) - ymod;

      int ixx = jp + (threadIdx.x + 1) * Lpoints1; // ZZZ, bad, strided read, BAD!
      double * __restrict__ c = &(dytemp[ixx]); //  bad c
      l = threadIdx.x + 1;
#pragma unroll 2
      while(l <= ma - CUDA_BLOCK_DIM)
	{
	  double a, b;
#ifdef DYTEMP_NEW
	  a = dytemp2[bid][jp][l];
#else
	  a = __ldca(c);
#endif
	  c += CUDA_BLOCK_DIM * Lpoints1;
#ifdef DYTEMP_NEW
	  b = dytemp2[bid][jp][l + CUDA_BLOCK_DIM];
#else
	  b = __ldca(c);
#endif
	  c += CUDA_BLOCK_DIM * Lpoints1;
	  dyda[l-1] = a;
	  dyda[l-1 + CUDA_BLOCK_DIM] = b;
	  l += 2*CUDA_BLOCK_DIM;
	}
      //#pragma unroll 2
      //for( ; l <= ma; l += CUDA_BLOCK_DIM, c += CUDA_BLOCK_DIM * Lpoints1)
      if(l < ma)
#ifdef DYTEMP_NEW
	dyda[l - 1] = dytemp2[bid][jp][l];
#else
	dyda[l-1] = __ldca(c);
#endif
	    
      __syncwarp();

      j = 0;
      double sig2iwght = sig2i * wght;

#pragma unroll 2
      for(l = 1; l <= lastone; l++)
	{
	  j++;
	  wt = dyda[l-1] * sig2iwght;
	  int xx = threadIdx.x + 1;
	  double * __restrict__ alp = &alpha[j * mf1 + xx];
#pragma unroll 2
	  while(xx <= l)
	    {
	      double const * __restrict__ alp2 = alp;
	      __stwb(alp, *alp2 + wt * dyda[xx-1]);
	      xx += CUDA_BLOCK_DIM;
	      alp += CUDA_BLOCK_DIM;
	    } // m 
	  //__syncthreads();
	  if(threadIdx.x == 0)
	    {
	      beta[j] = beta[j] + dy * wt;
	    }
	  //__syncthreads();
	} // l
	  
#pragma unroll 1
      for(; l <= lma; l++)
	{
	  if(iapp[l])
	    {
	      j++;
	      wt = dyda[l-1] * sig2iwght;
	      int xx = threadIdx.x + 1;
	      double * __restrict__ alp = &alpha[j * mf1 + xx];
#pragma unroll 2
	      while(xx <= lastone)
		{
		  //if(xx != 0)
		  double const * __restrict__ alp2 = alp;
		  //alpha[j * mf1 + xx] += wt * dyda[xx-1];
		  *alp = *alp2 + wt * dyda[xx-1];
		  xx += CUDA_BLOCK_DIM;
		} // m 
	      //__syncthreads();
	      if(threadIdx.x == 0)
		{
		  k = lastone;
		  m = lastone + 1;
		  int * __restrict__ iap = iapp + m;
		  double * __restrict__ alp = alpha + j * mf1 + k;
#pragma unroll 4
		  for(; m <= l; m++)
		    {
		      if(*iap)
			{
			  alp++;
			  double const * __restrict__ alp2 = alp;
			  *alp = *alp2 + wt * dyda[m-1];
			}
		      iap++;
		    } // m 
		  beta[j] = beta[j] + dy * wt;
		}
	      //__syncthreads();
	    }
	} // l 
      ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
    } // jp 

  if(threadIdx.x == 0)
    {
      npg2[bid] = lnp2;
      trial_chisqg[bid] = ltrial_chisq;
    }
}

__global__ void
__launch_bounds__(768) 
CudaCalculateIter1Mrqcof1Curve2I0IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I0IA0(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}


__global__ void
__launch_bounds__(768) 
CudaCalculateIter1Mrqcof1Curve2I0IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I0IA1(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}


__global__ void
__launch_bounds__(768) 
CudaCalculateIter1Mrqcof1Curve2I1IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I1IA0(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}

__global__ void
__launch_bounds__(768) 
CudaCalculateIter1Mrqcof1Curve2I1IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I1IA1(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}

__global__ 
void CudaCalculateIter1Mrqcof2CurveM12I0IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  double *atryp = atry[bid]; //CUDA_LCC->atry;
  mrqcof_matrix(CUDA_LCC, atryp, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, atryp, 0, lpoints, bid);
  MrqcofCurve2I0IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

__global__ 
void CudaCalculateIter1Mrqcof2CurveM12I0IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  double *atryp = atry[bid]; //CUDA_LCC->atry;
  mrqcof_matrix(CUDA_LCC, atryp, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, atryp, 0, lpoints, bid);
  MrqcofCurve2I0IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  MrqcofCurve23I0IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}

__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  MrqcofCurve23I0IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}


// SLOW
__global__ void CudaCalculateIter1Mrqcof2Curve2I1IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  MrqcofCurve23I1IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}



__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof2Curve2I1IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I1IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}



__global__ 
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1CurveM12I0IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  double *cg = CUDA_LCC->cg;
  mrqcof_matrix(CUDA_LCC, cg, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, cg, 0, lpoints, bid);
  MrqcofCurve2I0IA0(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, lpoints, bid);
}


__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1CurveM12I0IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  double *cg = CUDA_LCC->cg;
  mrqcof_matrix(CUDA_LCC, cg, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, cg, 0, lpoints, bid);
  MrqcofCurve2I0IA1(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, lpoints, bid);
}



__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1CurveM12I1IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  double *cg = CUDA_LCC->cg;
  mrqcof_matrix(CUDA_LCC, cg, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, cg, 1, lpoints, bid);
  MrqcofCurve2I1IA0(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, lpoints, bid);
}


__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1CurveM12I1IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  double *cg = CUDA_LCC->cg;
  mrqcof_matrix(CUDA_LCC, cg, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, cg, 1, lpoints, bid);
  MrqcofCurve2I1IA1(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, lpoints, bid);
}


__global__ 
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1Curve1LastI0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  if(CUDA_LCC->ytemp == NULL) return;

  mrqcof_curve1_lastI0(CUDA_LCC, CUDA_LCC->cg, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}


__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1Curve1LastI1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  mrqcof_curve1_lastI1(CUDA_LCC, CUDA_LCC->cg, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}

__global__ 
void CudaCalculateIter1Mrqcof2CurveM12I1IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  double *atryp = atry[bid]; //CUDA_LCC->atry;
  mrqcof_matrix(CUDA_LCC, atryp, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, atryp, 1, lpoints, bid);
  MrqcofCurve2I1IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

__global__ 
__launch_bounds__(384) 
void CudaCalculateIter1Mrqcof2CurveM12I1IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  double *atryp = atry[bid]; //CUDA_LCC->atry;
  mrqcof_matrix(CUDA_LCC, atryp, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, atryp, 1, lpoints, bid);
  MrqcofCurve2I1IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

__global__
__launch_bounds__(768) 
void CudaCalculateIter1Mrqcof2Curve1LastI0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  mrqcof_curve1_lastI0(CUDA_LCC, atry[bid], CUDA_LCC->covar, CUDA_LCC->da, bid);
}

__global__
__launch_bounds__(1024) 
void CudaCalculateIter1Mrqcof2Curve1LastI1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  mrqcof_curve1_lastI1(CUDA_LCC, atry[bid], CUDA_LCC->covar, CUDA_LCC->da, bid);
}