//#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

//#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>
//#include "globals_CUDA.h"
//#include "declarations_CUDA.h"
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>

void swap(__global double* a, __global double* b) {
	double temp = *a;
	*a = *b;
	*b = temp;
}

int gauss_errc_begin(
	__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	double* big,
	int* irow,
	int* licol,
	int brtmpl,
	int brtmph,
	int i,
	int j)
{
	int3 threadIdx;
	threadIdx.x = get_local_id(0);

	int ixx = j * Fa->Lmfit1 + 1;
	int k;
	int n = Fa->Lmfit;
	double covar;

	for (k = 1; k <= n; k++, ixx++)
	{
		if ((*CUDA_LCC).ipiv[k] == 0)
		{
			//printf("[%d][%d][%d] ipiv[%d]: %d\n", blockIdx.x, threadIdx.x, i, k, (*CUDA_LCC).ipiv[k]);
			//  double n = *(double *)num;
			covar = (*CUDA_LCC).covar[ixx];

			//if (threadIdx.x == 2 && i == 2)
			//	printf("j[%d], i[%d], k[%d], covar[%d]: % .9f\n", j, i, k, ixx, covar);

			double tmpcov = fabs(covar);
			//if (threadIdx.x == 9)
			//	printf("tmpcov: % .9f\n", tmpcov);

			if (tmpcov >= *big)
			{
				*big = tmpcov;
				*irow = j;
				*licol = k;
			}
		}
		else if ((*CUDA_LCC).ipiv[k] > 1)
		{
			//printf("-");
			//if (blockIdx.x == 2)
			//	printf("[%d][%d][%d] ipiv[%d]: %d\n", blockIdx.x, threadIdx.x, i, k, (*CUDA_LCC).ipiv[k]);
			//__syncthreads();
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

			return(1);
		}


		// ----------------*****************---------------------

		//switch ((*CUDA_LCC).ipiv[k])
		//{
		//case 0:
		//	covar = (*CUDA_LCC).covar[ixx];
		//	double tmpcov = fabs(covar);
		//	//if (threadIdx.x == 9)
		//	//	printf("tmpcov: % .9f\n", tmpcov);
		//
		//	if (tmpcov >= *big)
		//	{
		//		*big = tmpcov;
		//		*irow = j;
		//		*licol = k;
		//	}
		//	break;
		//
		//case 1:
		//	break;
		//
		//default:
		//	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		//	return;
		//}
	}

	//if (threadIdx.x == 2)
	//	printf("[%d] i[%d] big: % .9f, irow: %d, icol: %d\n", threadIdx.x, i, *big, *irow, *licol);
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//if (threadIdx.x == 40 && i == 1)
	//	printf(">>> big[%d]: % .9f\n", threadIdx.x, *big);

	return (0);
}

int gauss_errc_mid(
	__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	double* big,
	int* irow,
	__local int* icol,
	__local double* pivinv,
	int i)
{
	int3 threadIdx;
	threadIdx.x = get_local_id(0);

	int j, l;
	int n = Fa->Lmfit;
	__private int ma = Fa->ma;

	++((*CUDA_LCC).ipiv[*icol]);
	if (*irow != *icol)
	{
		int index = *irow * Fa->Lmfit1 + l;
		for (l = 1; l <= n; l++)
		{
			swap(&((*CUDA_LCC).covar[index]), &((*CUDA_LCC).covar[index]));
			//SWAP((*CUDA_LCC).covar[index], (*CUDA_LCC).covar[index])
		}

		swap(&((*CUDA_LCC).da[*irow]), &((*CUDA_LCC).da[*icol]));
		//SWAP((*CUDA_LCC).da[irow], (*CUDA_LCC).da[icol])
			//SWAP(b[irow],b[icol])
	}

	(*CUDA_LCC).indxr[i] = *irow;
	(*CUDA_LCC).indxc[i] = *icol;
	int colIdx = *icol * Fa->Lmfit1 + *icol;
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

		return (2);
	}

	*pivinv = 1.0 / (*CUDA_LCC).covar[colIdx];
	(*CUDA_LCC).covar[colIdx] = 1.0;
	(*CUDA_LCC).da[*icol] *= *pivinv;
	//b[icol] *= pivinv;

	return(0);
}

void gauss_errc_end(
	__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	int licol,
	int ll,
	int i)
{

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	double dum;

	if (ll != licol)
	{
		int ixx = ll * Fa->Lmfit1;
		int jxx = licol * Fa->Lmfit1;
		dum = (*CUDA_LCC).covar[ixx + licol];
		(*CUDA_LCC).da[ll] -= (*CUDA_LCC).da[licol] * dum;

		(*CUDA_LCC).covar[ixx + licol] = 0.0;
		ixx++;
		jxx++;

		//if (threadIdx.x == 0)
		//	printf("[%d][%d] i[%d] covar[%d]: % .9f, dum: % .9f, irow: %d, icol: %d\n", blockIdx.x, threadIdx.x, i, ixx + licol, (*CUDA_LCC).covar[ixx + licol], dum, ixx, jxx);


		//for (l = 1; l <= n; l++)
		for (int l = 1; l <= Fa->Lmfit; l++)     // Fa->Lmfit: 54
		{
			ixx++;
			jxx++;
			(*CUDA_LCC).covar[ixx] -= (*CUDA_LCC).covar[jxx] * dum;
		}

		//b[ll] -= b[icol]*dum;
	}
}

void gauss_errc(
	__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	int brtmpl,
	int brtmph,
	int3 threadIdx,
	int3 blockIdx)
	//__global int* ipiv,
	//int i)
	/*__local int* icol,
	__local double* pivinv,
	__local int* sh_icol,
	__local int* sh_irow,
	__local double* sh_big,*/
	//int ma)
{
	//int3 threadIdx, blockIdx;
	//threadIdx.x = get_local_id(0);
	//blockIdx.x = get_group_id(0);

	__local int icol;
	__local double pivinv;
	__local int sh_icol[BLOCK_DIM];
	__local int sh_irow[BLOCK_DIM];
	__local double sh_big[BLOCK_DIM];

	__private double covar;
	__private int ma = Fa->ma;

	// __shared__ int icol;
	// __shared__ double pivinv;
	// __shared__ int sh_icol[CUDA_BLOCK_DIM];
	// __shared__ int sh_irow[CUDA_BLOCK_DIM];
	// __shared__ double sh_big[CUDA_BLOCK_DIM];



	//	__shared__ int indxc[MAX_N_PAR+1],indxr[MAX_N_PAR+1],ipiv[MAX_N_PAR+1];
	icol = 0;
	int irow = 0;
	int j, k, l, ll;
	__private double big = 0;
	double dum, temp;
	__private int n = Fa->Lmfit;
	__private int ixx, jxx;

	/*int brtmph, brtmpl;
	brtmph = Fa->Lmfit / BLOCK_DIM;
	if (Fa->Lmfit % BLOCK_DIM)
	{
		brtmph++;
	}
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Fa->Lmfit)
	{
		brtmph = Fa->Lmfit;
	}
	brtmpl++;*/

	//if (threadIdx.x == 0)
	//{
	//	for (j = 1; j <= Fa->Lmfit; j++)
	//	{
	//		(*CUDA_LCC).ipiv[j] = 0;
	//	}
	//}

	////__syncthreads();
	//barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	//if (blockIdx.x == 2)
	//	printf("[%d][%d][%d] brtmpl: %d, brtmph: %d, Lmfit: %d\n", blockIdx.x, threadIdx.x, i, brtmpl, brtmph, Fa->Lmfit);

	for (int i = 1; i <= n; i++)
	{
		big = 0.0;
		irow = 0;
		icol = 0;
		for (j = brtmpl; j <= brtmph; j++)
		{
			if ((*CUDA_LCC).ipiv[j] != 1)
			{
				int ixx = j * Fa->Lmfit1 + 1;
				k = 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					//if (blockIdx.x == 2 & i == 1)
					//	printf("[%d][%d] i: %d, j: %d, ipiv[%d]: %d\n", blockIdx.x, threadIdx.x, i, j, k, (*CUDA_LCC).ipiv[k]);

					if ((*CUDA_LCC).ipiv[k] == 0)
					{
						//printf("[%d][%d][%d] ipiv[%d]: %d\n", blockIdx.x, threadIdx.x, i, k, (*CUDA_LCC).ipiv[k]);
						//  double n = *(double *)num;
						covar = (*CUDA_LCC).covar[ixx];
						double tmpcov = fabs(covar);
						//if (threadIdx.x == 9)
						//	printf("tmpcov: % .9f\n", tmpcov);

						if (tmpcov >= big)
						{
							big = tmpcov;
							irow = j;
							icol = k;
						}
					}
					else if ((*CUDA_LCC).ipiv[k] > 1)
					{
						//printf("-");
						//if (blockIdx.x == 2)
						//	printf("[%d][%d][%d] ipiv[%d]: %d\n", blockIdx.x, threadIdx.x, i, k, (*CUDA_LCC).ipiv[k]);
						//__syncthreads();
						barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

						return;
					}
				}
			}
		}

		sh_big[threadIdx.x] = big;
		sh_irow[threadIdx.x] = irow;
		sh_icol[threadIdx.x] = icol;

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (blockIdx.x == 2 && threadIdx.x == 2)
			printf("[%d][%d] big: % .9f, irow: %d, icol: %d\n", blockIdx.x, threadIdx.x, big, irow, icol);

		// --------*******************----------------

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
					swap(&((*CUDA_LCC).covar[index]), &((*CUDA_LCC).covar[index]));
					//SWAP((*CUDA_LCC).covar[index], (*CUDA_LCC).covar[index])
				}

				swap(&((*CUDA_LCC).da[irow]), &((*CUDA_LCC).da[icol]));
				//SWAP((*CUDA_LCC).da[irow], (*CUDA_LCC).da[icol])
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
				printf("+");

				return;
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

		// -----------*****************---------------------------------

		//if (threadIdx.x == 2 && blockIdx.x == 2)
		//	printf("[%d][%d] i[%d] brtmph: %d, ll: %d, icol: %d, ll!=icol: %d\n", blockIdx.x, threadIdx.x, i, brtmph, ll, icol, ll != icol);

		//for (ll = brtmpl; ll <= brtmph; ll++)
		ll = brtmpl;
		while (ll <= brtmph)
		{

			if (ll != icol)
			{
				ixx = ll * Fa->Lmfit1;
				jxx = icol * Fa->Lmfit1;
				dum = (*CUDA_LCC).covar[ixx + icol];
				(*CUDA_LCC).da[ll] -= (*CUDA_LCC).da[icol] * dum;

				(*CUDA_LCC).covar[ixx + icol] = 0.0;

				//if (threadIdx.x == 2 && i == 1)
				//	printf("[%d][%d] covar[%d]: % .9f, dum: % .9f, irow: %d, icol: %d\n", blockIdx.x, threadIdx.x, ixx + icol, (*CUDA_LCC).covar[ixx + icol], dum, ixx, jxx);

				ixx++;
				jxx++;

				//for (l = 1; l <= n; l++)
				for (l = 1; l <= n; l++)     // Fa->Lmfit: 54
				{
					ixx++;
					jxx++;
					(*CUDA_LCC).covar[ixx] -= (*CUDA_LCC).covar[jxx] * dum;
				}

				//b[ll] -= b[icol]*dum;
			}
			ll++;
		}

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	if (threadIdx.x == 0)
	{
		for (l = Fa->Lmfit; l >= 1; l--)
		{
			int indxr = (*CUDA_LCC).indxr[l];
			int indxc = (*CUDA_LCC).indxc[l];
			if (indxr != indxc)
			{
				for (k = 1; k <= Fa->Lmfit; k++)
				{
					int a = k * Fa->Lmfit1 + indxr;
					int b = k * Fa->Lmfit1 + indxc;

					swap(&((*CUDA_LCC).covar[a]), &((*CUDA_LCC).covar[b]));
					//SWAP((*CUDA_LCC).covar[a], (*CUDA_LCC).covar[b]);
				}
			}
		}
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	return;
}
//#undef SWAP
/* from Numerical Recipes */
