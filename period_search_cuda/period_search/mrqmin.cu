/* N.B. The foll. L-M routines are modified versions of Press et al.
   converted from Mikko's fortran code

   8.11.2006
*/

#include <cuda.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include <device_launch_parameters.h>
#include <stdio.h>

__device__ int __forceinline__ mrqmin_1_end(freq_context *__restrict__ CUDA_LCC, int ma, int mfit, int mfit1, const int block)
{
	int bid = blockIdx();

	if (__ldg(&isAlamda[bid]))
	{
		int n = threadIdx.x + 1;
		double *__restrict__ ap = atry[bid] + n;
		double const *__restrict__ cgp = CUDA_LCC->cg + n;
#pragma unroll 1
		while (n <= ma - block)
		{
			ap[0] = cgp[0];
			ap[block] = cgp[block];
			n += 2 * block;
			ap += 2 * block;
			cgp += 2 * block;
		}
		if (n <= ma)
		{
			ap[0] = cgp[0];
		}
	}

	double ccc = 1 + __ldg(&Alamda[bid]);

	int ixx = mfit1 + threadIdx.x + 1;

	double *__restrict__ a = CUDA_LCC->covar + ixx;
	double const *__restrict__ b = CUDA_LCC->alpha + ixx;
#pragma unroll 2
	while (ixx < mfit1 * mfit1 - (UNRL - 1) * block)
	{
		int i;
		double t[UNRL];
		for (i = 0; i < UNRL; i++)
		{
			t[i] = b[0];
			b += block;
		}
		for (i = 0; i < UNRL; i++)
		{
			if ((ixx + i * block) % (mfit1 + 1) == 0)
				a[0] = ccc * t[i];
			else
				a[0] = t[i];
			a += block;
		}
		ixx += UNRL * block;
	}
#pragma unroll 3
	while (ixx < mfit1 * mfit1)
	{
		double t = b[0];
		if (ixx % (mfit1 + 1) == 0)
			*a = ccc * t;
		else
			*a = t;

		a += block;
		b += block;
		ixx += block;
	}

	int xx = threadIdx.x + 1;
	double const *__restrict__ bp;
	double *__restrict__ dap;
	bp = CUDA_LCC->beta + xx;
	dap = CUDA_LCC->da + xx;
#pragma unroll 1
	while (xx <= mfit - block)
	{
		dap[0] = bp[0];
		dap[block] = bp[block];
		bp += 2 * block;
		dap += 2 * block;
		xx += 2 * block;
	}
	if (xx <= mfit)
	{
		*dap = bp[0];
		bp += block;
		dap += block;
		xx += block;
	}

	__syncwarp();

	int err_code = gauss_errc(CUDA_LCC, ma);
	if (err_code)
	{
		return err_code;
	}

	int n = threadIdx.x + 1;
	int const *__restrict__ iap = CUDA_ia + n;
	double *__restrict__ ap = atry[bid] + n;
	double const *__restrict__ cgp = CUDA_LCC->cg + n;
	double const *__restrict__ ddap = CUDA_LCC->da + n - 1;
#pragma unroll 1
	while (n <= ma - block)
	{
		if (*iap)
			*ap = cgp[0] + ddap[0];
		if (iap[block])
			ap[block] = cgp[block] + ddap[block];
		n += 2 * block;
		iap += 2 * block;
		ap += 2 * block;
		cgp += 2 * block;
		ddap += 2 * block;
	}
	// #pragma unroll 2
	if (n <= ma)
	{
		if (*iap)
			*ap = cgp[0] + ddap[0];
	}
	//__syncthreads();

	return err_code;
}

// clean pointers and []'s
// threadify loops
__device__ void __forceinline__ mrqmin_2_end(freq_context *__restrict__ CUDA_LCC, int ma, int bid)
{
	int j, k, l; //, bid = blockIdx();
	int mf = CUDA_mfit, mf1 = CUDA_mfit1;

	if (Chisq[bid] < Ochisq[bid])
	{
		double rai = CUDA_Alamda_incr;
		double const *__restrict__ dap = CUDA_LCC->da + 1 + threadIdx.x;
		double *__restrict__ dbp = CUDA_LCC->beta + 1 + threadIdx.x;
#pragma unroll 1
		for (j = threadIdx.x; j < mf - CUDA_BLOCK_DIM; j += CUDA_BLOCK_DIM)
		{
			double v1 = dap[0];
			double v2 = dap[CUDA_BLOCK_DIM];
			dbp[0] = v1;
			dbp[CUDA_BLOCK_DIM] = v2;
			dbp += 2 * CUDA_BLOCK_DIM;
			dap += 2 * CUDA_BLOCK_DIM;
		}
		if (j < mf)
			*dbp = dap[0];

		rai = __drcp_rn(rai); /// 1.0/rai;

		double const *__restrict__ cvp = CUDA_LCC->covar + mf1 + threadIdx.x;

		double *__restrict__ ap = CUDA_LCC->alpha + mf1 + threadIdx.x;

		double const *__restrict__ cvpo = cvp + 1;

		double *apo = ap + 1;

		Alamda[bid] = __ldg(&Alamda[bid]) * rai;

#pragma unroll 1
		for (j = 0; j < mf; j++)
		{
			cvp = cvpo;
			ap = apo;
#pragma unroll 1
			for (k = threadIdx.x; k < mf - CUDA_BLOCK_DIM; k += CUDA_BLOCK_DIM)
			{
				double v1 = cvp[0];
				double v2 = cvp[CUDA_BLOCK_DIM];
				ap[0] = v1;
				ap[CUDA_BLOCK_DIM] = v2;
				cvp += 2 * CUDA_BLOCK_DIM;
				ap += 2 * CUDA_BLOCK_DIM;
			}

			if (k < mf)
				__stwb(ap, __ldca(cvp)); //[0]; //ldcs

			cvpo += mf + 1;
			apo += mf + 1;
		}

		double const *__restrict__ atp = atry[bid] + 1 + threadIdx.x;

		double *__restrict__ cgp = CUDA_LCC->cg + 1 + threadIdx.x;

#pragma unroll 1
		for (l = threadIdx.x; l < ma - CUDA_BLOCK_DIM; l += CUDA_BLOCK_DIM)
		{
			double v1 = atp[0];
			double v2 = atp[CUDA_BLOCK_DIM];
			cgp[0] = v1;
			cgp[CUDA_BLOCK_DIM] = v2;
			atp += CUDA_BLOCK_DIM;
			cgp += CUDA_BLOCK_DIM;
		}

		if (l < ma)
			*cgp = atp[0];
	}
	else if (threadIdx.x == 0)
	{
		double a, c;
		a = CUDA_Alamda_incr * __ldg(&Alamda[bid]);
		c = Ochisq[bid];
		Alamda[bid] = a;
		Chisq[bid] = c;
	}

	return;
}