/* Convexity regularization function

   8.11.2006
*/

//#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
////#include "globals_CUDA.h"
//#include "declarations_OpenCl.h

//#define __device

double conv(struct freq_context2* CUDA_LCC, struct funcarrays FA, int2* texArea, int2* texDg, int nc, int tmpl, int tmph, int brtmpl, int brtmph)
{
	int i, j, k;
	double res[BLOCK_DIM]; // NOTE: __shared__
	double tmp, dtmp;
	int2 bfr;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_global_id(0);
	threadIdx.x = get_local_id(0);

	tmp = 0;
	j = blockIdx.x * (FA.Numfac1) + brtmpl;
	for (i = brtmpl; i <= brtmph; i++, j++)
	{
		bfr = texArea[j];
		//bfr = tex1Dfetch(texArea, j);
		tmp += __hiloint2double(bfr.y, bfr.x) * FA.Nor[i][nc];
	}

	res[threadIdx.x] = tmp;

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//parallel reduction
	k = BLOCK_DIM >> 1;
	while (k > 1)
	{
		if (threadIdx.x < k) {
			res[threadIdx.x] += res[threadIdx.x + k];
		}

		k = k >> 1;

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	if (threadIdx.x == 0)
	{
		tmp = res[0] + res[1];
	}

	//parallel reduction end
	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int2 xx;
	int m = blockIdx.x * FA.Dg_block + tmpl * FA.Numfac1;
	for (j = tmpl; j <= tmph; j++, m += FA.Numfac1)
	{
		dtmp = 0;
		if (j <= FA.Ncoef)
		{
			int mm = m + 1;
			for (i = 1; i <= FA.Numfac; i++, mm++)
			{
				xx = texDg[mm];
				//xx = tex1Dfetch(texDg, mm);
				dtmp += FA.Darea[i] * __hiloint2double(xx.y, xx.x) * FA.Nor[i][nc];
			}
		}
		(*CUDA_LCC).dyda[j] = dtmp;
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	return (tmp);
}
