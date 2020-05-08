/* Convexity regularization function

   8.11.2006
*/

//#include <math.h>
//#include <stdlib.h>
//#include <stdio.h>
////#include "globals_CUDA.h"
//#include "declarations_OpenCl.h

//#define __device

double conv(
	__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	__global int2* texArea, 
	__global int2* texDg, 
	__local double* res,
	int nc, int tmpl, int tmph, int brtmpl, int brtmph)
{
	int i, j, k;
	// NOTE: variable length arrays are not supported in OpenCL, also it is "__shared__"
	//double res[Fa->blockDim]; 
	double tmp, dtmp;
	int2 bfr;
	int3 blockIdx, threadIdx;
	blockIdx.x = get_global_id(0);
	threadIdx.x = get_local_id(0);

	tmp = 0;
	j = blockIdx.x * (Fa->Numfac1) + brtmpl;
	for (i = brtmpl; i <= brtmph; i++, j++)
	{
		bfr = texArea[j];
		//bfr = tex1Dfetch(texArea, j);
		tmp += HiLoint2double(bfr.y, bfr.x) * Fa->Nor[i][nc];
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
	int m = blockIdx.x * Fa->Dg_block + tmpl * Fa->Numfac1;
	for (j = tmpl; j <= tmph; j++, m += Fa->Numfac1)
	{
		dtmp = 0;
		if (j <= Fa->Ncoef)
		{
			int mm = m + 1;
			for (i = 1; i <= Fa->Numfac; i++, mm++)
			{
				xx = texDg[mm];
				//xx = tex1Dfetch(texDg, mm);
				dtmp += Fa->Darea[i] * HiLoint2double(xx.y, xx.x) * Fa->Nor[i][nc];
			}
		}
		(*CUDA_LCC).dyda[j] = dtmp;
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	return (tmp);
}
