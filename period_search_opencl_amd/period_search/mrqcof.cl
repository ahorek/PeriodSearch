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

void mrqcof_start(__global struct freq_context2* CUDA_LCC, struct funcarrays FA, double a[], __global double alpha[], __global double beta[])
{
	int j, k;
	int brtmph, brtmpl;
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	int3 threadIdx;
	threadIdx.x = get_local_id(0);

	brtmph = FA.Numfac / BLOCK_DIM;
	if (FA.Numfac % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > FA.Numfac) brtmph = FA.Numfac;
	brtmpl++;

	/* N.B. curv and blmatrix called outside bright
	   because output same for all points */
	curv(CUDA_LCC, FA, a, brtmpl, brtmph);

	if (threadIdx.x == 0)
	{
		//   #ifdef YORP
		//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
		  // #else
		blmatrix(CUDA_LCC, a[FA.ma - 4 - FA.Nphpar], a[FA.ma - 3 - FA.Nphpar]);
		//   #endif
		(*CUDA_LCC).trial_chisq = 0;
		(*CUDA_LCC).np = 0;
		(*CUDA_LCC).np1 = 0;
		(*CUDA_LCC).np2 = 0;
		(*CUDA_LCC).ave = 0;
	}

	brtmph = FA.Lmfit / BLOCK_DIM;
	if (FA.Lmfit % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > FA.Lmfit) brtmph = FA.Lmfit;
	brtmpl++;

	for (j = brtmpl; j <= brtmph; j++)
	{
		for (k = 1; k <= j; k++)
		{
			alpha[j * (FA.Lmfit1) + k] = 0;
		}

		beta[j] = 0;
	}

	//__syncthreads(); //for sure
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

double mrqcof_end(struct freq_context2* CUDA_LCC, struct funcarrays FA, double* alpha)
{
	int j, k;

	for (j = 2; j <= FA.Lmfit; j++)
	{
		for (k = 1; k <= j - 1; k++)
		{
			alpha[k * FA.Lmfit1 + j] = alpha[j * FA.Lmfit1 + k];
		}
	}

	return (*CUDA_LCC).trial_chisq;
}

void mrqcof_matrix(struct freq_context2* CUDA_LCC, struct funcarrays FA, double a[], int Lpoints)
{
	matrix_neo(CUDA_LCC, FA, a, (*CUDA_LCC).np, Lpoints);
}

void mrqcof_curve1(struct freq_context2* CUDA_LCC, struct funcarrays FA, int2* texArea, int2* texDg,
	double a[], double* alpha, double beta[], int Inrel, int Lpoints)
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
		bright(CUDA_LCC, FA, texArea, texDg, a, jp, Lpoints1, Inrel);

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if (Inrel == 1) {
		int tmph, tmpl;
		tmph = FA.ma / BLOCK_DIM;
		if (FA.ma % BLOCK_DIM) tmph++;
		tmpl = threadIdx.x * tmph;
		tmph = tmpl + tmph;
		if (tmph > FA.ma) tmph = FA.ma;
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

void mrqcof_curve1_last(struct freq_context2* CUDA_LCC, struct funcarrays FA, int2* texArea, int2* texDg, double a[], double* alpha, double beta[], int Inrel, int Lpoints)
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
			for (l = 1; l <= FA.ma; l++)
				(*CUDA_LCC).dave[l] = 0;
		}
		else
			lave = (*CUDA_LCC).ave;
	}
	//precalc thread boundaries
	int tmph, tmpl;
	tmph = FA.ma / BLOCK_DIM;
	if (FA.ma % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > FA.ma) tmph = FA.ma;
	tmpl++;
	//
	int brtmph, brtmpl;
	brtmph = FA.Numfac / BLOCK_DIM;
	if (FA.Numfac % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > FA.Numfac) brtmph = FA.Numfac;
	brtmpl++;

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	for (jp = 1; jp <= Lpoints; jp++)
	{
		lnp++;

		ymod = conv(CUDA_LCC, FA, texArea, texDg, jp - 1, tmpl, tmph, brtmpl, brtmph);

		if (threadIdx.x == 0)
		{
			(*CUDA_LCC).ytemp[jp] = ymod;

			if (Inrel == 1)
				lave = lave + ymod;
		}
		for (l = tmpl; l <= tmph; l++)
		{
			(*CUDA_LCC).dytemp[jp + l * (Lpoints + 1)] = (*CUDA_LCC).dyda[l];
			if (Inrel == 1)
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dyda[l];
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

