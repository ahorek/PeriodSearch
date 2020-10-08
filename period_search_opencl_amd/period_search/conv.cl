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
	int nc, int tmpl, int tmph, int brtmpl, int brtmph)
{
	int i, j, k;
	// NOTE: variable length arrays are not supported in OpenCL, also it is "__shared__"
	__local double res[BLOCK_DIM];
	double tmp, dtmp;
	//int2 bfr;
	double bfr;
	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	tmp = 0;
	j = blockIdx.x * (Fa->Numfac1) + brtmpl;
	for (i = brtmpl; i <= brtmph; i++, j++)
	{
		//bfr = texArea[j];
		//bfr = tex1Dfetch(texArea, j);
		//tmp += HiLoint2double(bfr.y, bfr.x) * Fa->Nor[i][nc];

		bfr = (*CUDA_LCC).Area[i];
		tmp += bfr * Fa->Nor[i][nc];

	}

	res[threadIdx.x] = tmp;

	//if (blockIdx.x == 2)
	//	printf("[%d][%d] tmp: % .6f\n", blockIdx.x, threadIdx.x, tmp);

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

	//int2 xx;
	double xx;
	//int m = blockIdx.x * Fa->Dg_block + tmpl * Fa->Numfac1;
	int m = tmpl * Fa->Numfac1;
	for (j = tmpl; j <= tmph; j++, m += Fa->Numfac1)
	{
		dtmp = 0;
		if (j <= Fa->Ncoef)
		{
			int mm = m + 1;
			for (i = 1; i <= Fa->Numfac; i++, mm++)
			{
				//xx = texDg[mm];
				//xx = tex1Dfetch(texDg, mm);
				//dtmp += Fa->Darea[i] * HiLoint2double(xx.y, xx.x) * Fa->Nor[i][nc];
				
				xx = (*CUDA_LCC).Dg[mm];
				dtmp += Fa->Darea[i] * xx * Fa->Nor[i][nc];

				//if (blockIdx.x == 2 && threadIdx.x == 5)
				//	printf("[%d][%d] dtmp[%d]: % .16f, bfr: % .16f\n", blockIdx.x, threadIdx.x, i, dtmp, xx);
					//printf("[%d][%d] Darea[%d]: % .16f, bfr: % .16f, Nor[%d][%d]: % .16f\n", blockIdx.x, threadIdx.x, i, Fa->Darea[i], xx, i, nc, Fa->Nor[i][nc]);

			}
		}
		(*CUDA_LCC).dyda[j] = dtmp;

		// NOTE: Here we get some tiny differences against CUDA calculations in 13 - 16 symbol after decimal place
		//if (blockIdx.x == 2)
		//	printf("[%d][%d] dyda[%d]: % .16f\n", blockIdx.x, threadIdx.x, j, (*CUDA_LCC).dyda[j]);
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	return (tmp);
}
