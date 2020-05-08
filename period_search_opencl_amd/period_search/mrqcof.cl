/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

//#include <stdio.h>
//#include <stdlib.h>
//#include "globals_CUDA.h"
//#include "declarations_CUDA.h"


/* comment the following line if no YORP */
/*#define YORP*/

void mrqcof_start(__global struct freq_context2* CUDA_LCC, __global varholder* Fa, double a[], __global double alpha[], __global double beta[])
{
	int j, k;
	int brtmph, brtmpl;
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	int3 threadIdx;
	threadIdx.x = get_local_id(0);

	brtmph = Fa->Numfac / BLOCK_DIM;
	if (Fa->Numfac % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Fa->Numfac) brtmph = Fa->Numfac;
	brtmpl++;

	/* N.B. curv and blmatrix called outside bright
	   because output same for all points */
	curv(CUDA_LCC, Fa, a, brtmpl, brtmph);

	if (threadIdx.x == 0)
	{
		//   #ifdef YORP
		//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
		  // #else
		blmatrix(CUDA_LCC, a[Fa->ma - 4 - Fa->Nphpar], a[Fa->ma - 3 - Fa->Nphpar]);
		//   #endif
		(*CUDA_LCC).trial_chisq = 0;
		(*CUDA_LCC).np = 0;
		(*CUDA_LCC).np1 = 0;
		(*CUDA_LCC).np2 = 0;
		(*CUDA_LCC).ave = 0;
	}

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

	brtmpl++;
	for (j = brtmpl; j <= brtmph; j++)
	{
		for (k = 1; k <= j; k++)
		{
			alpha[j * (Fa->Lmfit1) + k] = 0;
		}

		beta[j] = 0;
	}

	//__syncthreads(); //for sure
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

double mrqcof_end(__global struct freq_context2* CUDA_LCC, __global varholder* Fa, __global double* alpha)
{
	int j, k;

	for (j = 2; j <= Fa->Lmfit; j++)
	{
		for (k = 1; k <= j - 1; k++)
		{
			alpha[k * Fa->Lmfit1 + j] = alpha[j * Fa->Lmfit1 + k];
		}
	}

	// TODO: Check this insain return
	return (*CUDA_LCC).trial_chisq;
}

void mrqcof_matrix(__global struct freq_context2* CUDA_LCC, __global varholder* Fa, double a[], int Lpoints)
{
	matrix_neo(CUDA_LCC, Fa, a, (*CUDA_LCC).np, Lpoints);
}

void mrqcof_curve1(
	__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	__global int2* texArea,
	__global int2* texDg,
	double a[], double beta[], int Inrel, int Lpoints) //double* alpha,
{
	int l, k, jp, lnp, Lpoints1 = Lpoints + 1;
	double lave;
	double tmave[BLOCK_DIM]; // NOTE: __shared__
	int3 threadIdx;
	threadIdx.x = get_local_id(0);

	lnp = (*CUDA_LCC).np;
	lave = (*CUDA_LCC).ave;
	//precalc thread boundaries
	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;
	//

	for (jp = brtmpl; jp <= brtmph; jp++)
	{
		bright(CUDA_LCC, Fa, texArea, texDg, a, jp, Lpoints1, Inrel);
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if (Inrel == 1) {
		int tmph, tmpl;
		tmph = Fa->ma / BLOCK_DIM;
		if (Fa->ma % BLOCK_DIM) tmph++;
		tmpl = threadIdx.x * tmph;
		tmph = tmpl + tmph;
		if (tmph > Fa->ma) tmph = Fa->ma;
		tmpl++;
		if (tmpl == 1) tmpl++;

		int ixx;
		ixx = tmpl * Lpoints1;
		for (l = tmpl; l <= tmph; l++)
		{
			//jp==1
			ixx++;
			(*CUDA_LCC).dave[l] = (*CUDA_LCC).dytemp[ixx];
			//jp>=2
			ixx++;
			for (jp = 2; jp <= Lpoints; jp++, ixx++)
			{
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dytemp[ixx];
			}

		}

		tmave[threadIdx.x] = 0;
		for (jp = brtmpl; jp <= brtmph; jp++)
		{
			tmave[threadIdx.x] += (*CUDA_LCC).ytemp[jp];
		}

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//parallel reduction
		k = BLOCK_DIM >> 1;
		while (k > 1)
		{
			if (threadIdx.x < k)
			{
				tmave[threadIdx.x] += tmave[threadIdx.x + k];
			}

			k = k >> 1;
			//__syncthreads();
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		}
		if (threadIdx.x == 0) lave = tmave[0] + tmave[1];
		//parallel reduction end

	}
	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp + Lpoints;
		(*CUDA_LCC).ave = lave;
	}
}

void mrqcof_curve1_last(
	__global struct freq_context2* CUDA_LCC, 
	__global varholder* Fa, 
	__global int2* texArea, 
	__global int2* texDg, 
	__local double* res,
	double a[], 
	__global double alpha[], 
	double beta[], 
	int Inrel, 
	int Lpoints)
{
	int l, jp, lnp;
	double ymod, lave;
	int3 threadIdx;
	threadIdx.x = get_local_id(0);

	lnp = (*CUDA_LCC).np;
	//
	if (threadIdx.x == 0)
	{
		if (Inrel == 1) /* is the LC relative? */
		{
			lave = 0;
			for (l = 1; l <= Fa->ma; l++)
			{
				(*CUDA_LCC).dave[l] = 0;
			}
		}
		else
		{
			lave = (*CUDA_LCC).ave;
		}
	}
	//precalc thread boundaries
	int tmph, tmpl;
	tmph = Fa->ma / BLOCK_DIM;
	if (Fa->ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > Fa->ma) tmph = Fa->ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = Fa->Numfac / BLOCK_DIM;
	if (Fa->Numfac % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Fa->Numfac) brtmph = Fa->Numfac;
	brtmpl++;

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	for (jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;

		ymod = conv(CUDA_LCC, Fa, texArea, texDg, res, jp - 1, tmpl, tmph, brtmpl, brtmph);

		if (threadIdx.x == 0)
		{
			(*CUDA_LCC).ytemp[jp] = ymod;

			if (Inrel == 1)
			{
				lave = lave + ymod;
			}
		}
		for (l = tmpl; l <= tmph; l++)
		{
			(*CUDA_LCC).dytemp[jp + l * (Lpoints + 1)] = (*CUDA_LCC).dyda[l];
			if (Inrel == 1)
			{
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dyda[l];
			}
		}

		/* save lightcurves */
		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		/*         if ((*CUDA_LCC).Lastcall == 1) always ==0
					 (*CUDA_LCC).Yout[np] = ymod;*/

	} /* jp, lpoints */

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp;
		(*CUDA_LCC).ave = lave;
	}
}

