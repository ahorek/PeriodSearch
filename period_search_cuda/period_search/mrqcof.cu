/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <device_launch_parameters.h>



__device__ void __forceinline__ mrqcof_start(freq_context * __restrict__ CUDA_LCC,
											 double * __restrict__ a,
											 double * __restrict__ alpha,
											 double * __restrict__ beta,
											 int bid)
{
	int j, k;
	int mf = CUDA_mfit, mf1 = CUDA_mfit1;

	/* N.B. curv and blmatrix called outside bright
	   because output same for all points */
	curv(CUDA_LCC, a, bid);

#pragma unroll 4
	for (j = 1; j <= mf; j++)
	{
		alpha += mf1;
		k = threadIdx.x + 1;
#pragma unroll
		while (k <= j)
		{
			__stwb(&alpha[k], 0.0);
			k += CUDA_BLOCK_DIM;
		}
	}

	j = threadIdx.x + 1;
#pragma unroll 2
	while (j <= mf)
	{
		__stwb(&beta[j], 0.0);
		j += CUDA_BLOCK_DIM;
	}

	// __syncthreads(); //pro jistotu
}

__device__ double __forceinline__ mrqcof_end(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha)
{
	int j, k, mf = CUDA_mfit, mf1 = CUDA_mfit1;
	int tid = threadIdx.x;
	double *__restrict__ app = alpha + mf1 + 2 + tid;
	;
	double const *__restrict__ ap2 = alpha + (2 + tid) * mf1;
	long int mf1add = sizeof(double) * mf1;
#pragma unroll
	for (j = 2 + tid; j <= mf; j += blockDim.x)
	{
		double *__restrict__ ap = app;
#pragma unroll
		for (k = 1; k <= j - 1; k++)
		{
			__stwb(ap, __ldca(&ap2[k]));
			// ap  += mf1;
			ap = (double *)(((char *)ap) + mf1add);
		}
		app += blockDim.x;
		// ap2 += mf1;
		ap2 = (double *)(((char *)ap2) + mf1add * blockDim.x);
	}

	return 0; // trial_chisqg[bid];
}

__device__ void __forceinline__ mrqcof_matrix(freq_context *__restrict__ CUDA_LCC,
											  double *__restrict__ a,
											  int Lpoints, int bid)
{
	matrix_neo(CUDA_LCC, a, npg[bid], Lpoints, bid);
}

__device__ void __forceinline__ mrqcof_curve1(freq_context *__restrict__ CUDA_LCC,
											  double *__restrict__ a,
											  int Inrel, int Lpoints, int bid)
{
	int lnp, Lpoints1 = Lpoints + 1;
	double lave = 0;

	int n = threadIdx.x;
	if (Inrel == 1)
	{
#pragma unroll 1
		while (n <= Lpoints)
		{
			bright(CUDA_LCC, a, n, Lpoints1, 1); // jp <-- n, OK, consecutive
			n += CUDA_BLOCK_DIM;
		}
	}

	int ma = CUDA_ma;

	__syncwarp();
	double *__restrict__ dytemp = CUDA_LCC->dytemp, *__restrict__ ytemp = CUDA_LCC->ytemp;

	if (Inrel == 1)
	{
		double const *__restrict__ pp = &(dytemp[2 * Lpoints1 + threadIdx.x + 1]); // good, consecutive
		int bid = blockIdx();
#pragma unroll 1
		for (int i = 2; i <= ma; i++)
		{
			double dl = 0, dl2 = 0;
			int nn = threadIdx.x + 1;
			double const *__restrict__ p = pp;

#pragma unroll 2
			while (nn <= Lpoints - CUDA_BLOCK_DIM)
			{
				dl += p[0];
				dl2 += p[CUDA_BLOCK_DIM];
				p += 2 * CUDA_BLOCK_DIM;
				nn += 2 * CUDA_BLOCK_DIM;
			}
			// #pragma unroll 1
			if (nn <= Lpoints)
			{
				dl += p[0];
				// p  += CUDA_BLOCK_DIM;
				// nn += CUDA_BLOCK_DIM;
			}

			dl += dl2;

			dl += __shfl_down_sync(0xffffffff, dl, 16);
			dl += __shfl_down_sync(0xffffffff, dl, 8);
			dl += __shfl_down_sync(0xffffffff, dl, 4);
			dl += __shfl_down_sync(0xffffffff, dl, 2);
			dl += __shfl_down_sync(0xffffffff, dl, 1);

			pp += Lpoints1;

			if (threadIdx.x == 0)
				dave[bid][i] = dl;
		}

		double d = 0, d2 = 0;
		int n = threadIdx.x + 1;
		double const *__restrict__ p2 = &(ytemp[n]);

#pragma unroll 2
		while (n <= Lpoints - CUDA_BLOCK_DIM)
		{
			d += p2[0];
			d2 += p2[CUDA_BLOCK_DIM];
			p2 += 2 * CUDA_BLOCK_DIM;
			n += 2 * CUDA_BLOCK_DIM;
		}

		if (n <= Lpoints)
		{
			d += p2[0];
		}
		d += d2;

		d += __shfl_down_sync(0xffffffff, d, 16);
		d += __shfl_down_sync(0xffffffff, d, 8);
		d += __shfl_down_sync(0xffffffff, d, 4);
		d += __shfl_down_sync(0xffffffff, d, 2);
		d += __shfl_down_sync(0xffffffff, d, 1);

		lave = d;
	}

	if (threadIdx.x == 0)
	{
		lnp = npg[bid];
		aveg[bid] = lave;
		npg[bid] = lnp + Lpoints;
	}
	__syncwarp();
}

// __device__ void mrqcof_curve1_last(freq_context *CUDA_LCC, double a[],
// 	      double *alpha, double beta[],int Inrel,int Lpoints)
// {
// 	int l,jp, lnp;
//    double ymod, lave;

//    lnp=(*CUDA_LCC).np;
//    //
//    if (threadIdx.x==0)
//    {
// 	   if (Inrel == 1) /* is the LC relative? */
// 	   {
// 		  lave = 0;
// 		  for (l = 1; l <= CUDA_ma; l++)
// 		  (*CUDA_LCC).dave[l]=0;
// 	   }
// 	   else
// 		  lave=(*CUDA_LCC).ave;
//    }
// //precalc thread boundaries
//     int tmph,tmpl;
// 	tmph=CUDA_ma/CUDA_BLOCK_DIM;
// 	if(CUDA_ma%CUDA_BLOCK_DIM) tmph++;
// 	tmpl=threadIdx.x*tmph;
// 	tmph=tmpl+tmph;
// 	if (tmph>CUDA_ma) tmph=CUDA_ma;
// 	tmpl++;
// //
//     int brtmph,brtmpl;
// 	brtmph=CUDA_Numfac/CUDA_BLOCK_DIM;
// 	if(CUDA_Numfac%CUDA_BLOCK_DIM) brtmph++;
// 	brtmpl=threadIdx.x*brtmph;
// 	brtmph=brtmpl+brtmph;
// 	if (brtmph>CUDA_Numfac) brtmph=CUDA_Numfac;
// 	brtmpl++;

// 	__syncthreads();

//       for (jp = 1; jp <= Lpoints; jp++)
//       {
//          lnp++;

//          ymod = conv(CUDA_LCC,jp-1,tmpl,tmph,brtmpl,brtmph);

// 		 if (threadIdx.x==0)
// 		 {
// 			 (*CUDA_LCC).ytemp[jp] = ymod;

// 			 if (Inrel == 1)
// 				lave = lave + ymod;
// 		 }
// 		for (l=tmpl; l <= tmph; l++)
// 		{
// 			(*CUDA_LCC).dytemp[jp+l*(Lpoints+1)] = (*CUDA_LCC).dyda[l];
// 			if (Inrel == 1)
// 				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dyda[l];
// 		}
// 		/* save lightcurves */
// 		 __syncthreads();

// /*         if ((*CUDA_LCC).Lastcall == 1) always ==0
// 			 (*CUDA_LCC).Yout[np] = ymod;*/
//       } /* jp, lpoints */
// 	 if (threadIdx.x==0)
// 	 {
// 		  (*CUDA_LCC).np=lnp;
// 		  (*CUDA_LCC).ave=lave;
// 	 }
// }

__device__ void __forceinline__ mrqcof_curve1_lastI1(
	freq_context * __restrict__ CUDA_LCC,
	double * __restrict__ a,
	double * __restrict__ alpha,
	double * __restrict__ beta,
	int bid)
{
	int Lpoints = 3;
	int Lpoints1 = Lpoints + 1;
	int jp, lnp;
	double ymod, lave;
	__shared__ double dyda[BLOCKX4][N80];
	double *__restrict__ dydap = dyda[threadIdx.y];
	// int bid = blockIdx();

	lnp = npg[bid];

	int n = threadIdx.x + 1, ma = CUDA_ma;
	double *__restrict__ p = &(dave[bid][n]);
#pragma unroll 2
	while (n <= ma)
	{
		*p = 0;
		p += CUDA_BLOCK_DIM;
		n += CUDA_BLOCK_DIM;
	}
	lave = 0;

	//__syncthreads();

	double *__restrict__ dytemp = CUDA_LCC->dytemp, *ytemp = CUDA_LCC->ytemp;
	long int lpadd = sizeof(double) * Lpoints1;

#pragma unroll 1
	for (jp = 1; jp <= Lpoints; jp++)
	{
		ymod = conv(CUDA_LCC, (jp - 1), dydap, bid);

		lnp++;

		if (threadIdx.x == 0)
		{
			ytemp[jp] = ymod;
			lave = lave + ymod;
		}

		int n = threadIdx.x + 1;
		double const *__restrict__ a;
		double *__restrict__ b, *__restrict__ c;

		a = &(dydap[n - 1]);
		b = &(dave[bid][n]);
#ifdef DYTEMP_NEW
		// c = &(dytemp2[blockIdx()][jp][n]);
#else
		c = &(dytemp[jp + Lpoints1 * n]); // ZZZ bad store order, strided
#endif
		// unrl2
#pragma unroll 2
		while (n <= ma - CUDA_BLOCK_DIM)
		{
			double d = a[0], bb = b[0];
			double d2 = a[CUDA_BLOCK_DIM], bb2 = b[CUDA_BLOCK_DIM];
#ifdef DYTEMP_NEW
			dytemp2[bid][jp][n] = d;
#else
			c[0] = d;
#endif
			// c += Lpoints1;
			c = (double *)(((char *)c) + lpadd);
			b[0] = bb + d;
#ifdef DYTEMP_NEW
			dytemp2[bid][jp][n + CUDA_BLOCK_DIM] = d2;
#else
			c[0] = d2;
#endif
			// c += Lpoints1;
			c = (double *)(((char *)c) + lpadd);
			b[CUDA_BLOCK_DIM] = bb2 + d2;
			a += 2 * CUDA_BLOCK_DIM;
			b += 2 * CUDA_BLOCK_DIM;
			n += 2 * CUDA_BLOCK_DIM;
		}
		// #pragma unroll 1
		if (n <= ma)
		{
			double d = a[0], bb = b[0];
#ifdef DYTEMP_NEW
			dytemp2[bid][jp][n] = d;
#else
			c[0] = d;
#endif
			b[0] = bb + d;
		}
	} /* jp, lpoints */

	if (threadIdx.x == 0)
	{
		npg[bid] = lnp;
		aveg[bid] = lave;
	}

	/* save lightcurves */
	__syncwarp();
}

__device__ void __forceinline__ mrqcof_curve1_lastI0(freq_context * __restrict__ CUDA_LCC,
													 double * __restrict__ a,
													 double * __restrict__ alpha,
													 double * __restrict__ beta,
													 int bid)
{
	int Lpoints = 3;
	int Lpoints1 = Lpoints + 1;
	int jp, lnp;
	double ymod;
	__shared__ double dyda[BLOCKX4][N80];
	// int bid = blockIdx();
	double *__restrict__ dydap = dyda[threadIdx.y];

	lnp = npg[bid];

	//  if(threadIdx.x == 0)
	//  lave = CUDA_LCC->ave;

	//__syncthreads();

	int ma = CUDA_ma;
	double *__restrict__ dytemp = CUDA_LCC->dytemp, *ytemp = CUDA_LCC->ytemp;

#pragma unroll 3
	for (jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;

		ymod = conv(CUDA_LCC, (jp - 1), dydap, bid);

		if (threadIdx.x == 0)
			ytemp[jp] = ymod;

		int n = threadIdx.x + 1;
		double *__restrict__ p = &dytemp[jp + Lpoints1 * n]; // ZZZ bad store order, strided
#pragma unroll 2
		while (n <= ma - CUDA_BLOCK_DIM)
		{
			double d = dydap[n - 1];
			double d2 = dydap[n + CUDA_BLOCK_DIM - 1];
#ifdef DYTEMP_NEW
			dytemp2[bid][jp][n] = d;
#else
			*p = d; //  YYYY
#endif
			p += Lpoints1 * CUDA_BLOCK_DIM;
#ifdef DYTEMP_NEW
			dytemp2[bid][jp][n + CUDA_BLOCK_DIM] = d2;
#else
			*p = d2;
#endif
			p += Lpoints1 * CUDA_BLOCK_DIM;
			n += 2 * CUDA_BLOCK_DIM;
		}
		// #pragma unroll 1
		if (n <= ma)
		{
			double d = dydap[n - 1];
#ifdef DYTEMP_NEW
			dytemp2[bid][jp][n] = d;
#else
			*p = d;
#endif
			// p += Lpoints1 * CUDA_BLOCK_DIM;
			// n += CUDA_BLOCK_DIM;
		}
	} /* jp, lpoints */

	if (threadIdx.x == 0)
	{
		npg[bid] = Lpoints; // lnp;
		//      CUDA_LCC->ave = lave;
	}

	/* save lightcurves */
	//__syncthreads();
}