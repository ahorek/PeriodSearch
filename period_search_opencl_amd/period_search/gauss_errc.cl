#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include "globals_CUDA.h"
//#include "declarations_CUDA.h"
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

int gauss_errc(
	__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa)
	/*__local int* icol,
	__local double* pivinv,
	__local int* sh_icol,
	__local int* sh_irow,
	__local double* sh_big,*/
	//int ma)
{
	__local int icol;
	__local double pivinv;
	__local int sh_icol[BLOCK_DIM];
	__local int sh_irow[BLOCK_DIM];
	__local double sh_big[BLOCK_DIM];

	__private double covar;
	__private int ma = *(*CUDA_LCC).da;

	// __shared__ int icol;
	// __shared__ double pivinv;
	// __shared__ int sh_icol[CUDA_BLOCK_DIM];
	// __shared__ int sh_irow[CUDA_BLOCK_DIM];
	// __shared__ double sh_big[CUDA_BLOCK_DIM];

	int3 threadIdx;
	threadIdx.x = get_local_id(0);

	//	__shared__ int indxc[MAX_N_PAR+1],indxr[MAX_N_PAR+1],ipiv[MAX_N_PAR+1];
	int i, licol = 0, irow = 0, j, k, l, ll;
	double big, dum, temp;
	int n = Fa->Lmfit;

	int brtmph, brtmpl;
	brtmph = n / BLOCK_DIM;
	if (n % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > n) brtmph = n;
	brtmpl++;

	/*        indxc=vector_int(n+1);
		indxr=vector_int(n+1);
		ipiv=vector_int(n+1);*/

	if (threadIdx.x == 0)
	{
		for (j = 1; j <= n; j++) (*CUDA_LCC).ipiv[j] = 0;
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	for (i = 1; i <= n; i++)
	{
		big = 0.0;
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)
			if ((*CUDA_LCC).ipiv[j] != 1)
			{
				int ixx = j * Fa->Lmfit1 + 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					if ((*CUDA_LCC).ipiv[k] == 0)
					{
						//  double n = *(double *)num;
						covar = (*CUDA_LCC).covar[ixx];
						double tmpcov = fabs(covar);
						if (tmpcov >= big)
						{
							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if ((*CUDA_LCC).ipiv[k] > 1)
					{
						//printf("-");

						//__syncthreads();
						barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
						/*					        deallocate_vector((void *) ipiv);
												deallocate_vector((void *) indxc);
												deallocate_vector((void *) indxr);*/
						return(1);
					}
				}
			}

		sh_big[threadIdx.x] = big;
		sh_irow[threadIdx.x] = irow;
		sh_icol[threadIdx.x] = licol;

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (threadIdx.x == 0)
		{
			big = sh_big[0];
			icol = sh_icol[0];
			irow = sh_irow[0];
			for (j = 1; j < BLOCK_DIM; j++)
			{
				if (sh_big[j] >= big)
				{
					big = sh_big[j];
					irow = sh_irow[j];
					icol = sh_icol[j];
				}
			}
			++((*CUDA_LCC).ipiv[icol]);
			if (irow != icol)
			{
				int index = irow * Fa->Lmfit1 + l;
				for (l = 1; l <= n; l++)
				{
					SWAP((*CUDA_LCC).covar[index], (*CUDA_LCC).covar[index])
				}

				SWAP((*CUDA_LCC).da[irow], (*CUDA_LCC).da[icol])
					//SWAP(b[irow],b[icol])
			}

			(*CUDA_LCC).indxr[i] = irow;
			(*CUDA_LCC).indxc[i] = icol;
			int colIdx = icol * Fa->Lmfit1 + icol;
			if ((*CUDA_LCC).covar[colIdx] == 0.0)
			{
				j = 0;
				for (int l = 1; l <= ma; l++)
				{
					if (Fa->ia[l])
					{
						j++;
						(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
					}
				}
				//printf("+");
				/*					    deallocate_vector((void *) ipiv);
												deallocate_vector((void *) indxc);
												deallocate_vector((void *) indxr);*/
				return(2);
			}
			pivinv = 1.0 / (*CUDA_LCC).covar[colIdx];
			(*CUDA_LCC).covar[colIdx] = 1.0;
			(*CUDA_LCC).da[icol] *= pivinv;
			//b[icol] *= pivinv;
		}

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		for (l = brtmpl; l <= brtmph; l++)
		{
			(*CUDA_LCC).covar[icol * Fa->Lmfit1 + l] *= pivinv;
		}

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		for (ll = brtmpl; ll <= brtmph; ll++)
			if (ll != icol)
			{
				int ixx = ll * Fa->Lmfit1;
				int jxx = icol * Fa->Lmfit1;
				dum = (*CUDA_LCC).covar[ixx + icol];
				(*CUDA_LCC).covar[ixx + icol] = 0.0;
				ixx++;
				jxx++;
				for (l = 1; l <= n; l++, ixx++, jxx++)
				{
					(*CUDA_LCC).covar[ixx] -= (*CUDA_LCC).covar[jxx] * dum;
				}

				(*CUDA_LCC).da[ll] -= (*CUDA_LCC).da[icol] * dum;
				//b[ll] -= b[icol]*dum;
			}

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}
	if (threadIdx.x == 0)
	{
		for (l = n; l >= 1; l--)
		{
			if ((*CUDA_LCC).indxr[l] != (*CUDA_LCC).indxc[l])
			{
				for (k = 1; k <= n; k++)
				{
					SWAP((*CUDA_LCC).covar[k * Fa->Lmfit1 + (*CUDA_LCC).indxr[l]], (*CUDA_LCC).covar[k * Fa->Lmfit1 + (*CUDA_LCC).indxc[l]]);
				}
			}
		}
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	/*        deallocate_vector((void *) ipiv);
		deallocate_vector((void *) indxc);
		deallocate_vector((void *) indxr);*/

	return(0);
}
#undef SWAP
/* from Numerical Recipes */
