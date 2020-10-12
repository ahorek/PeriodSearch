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

void mrqcof_start(
	__global struct freq_context2* CUDA_LCC,
	__global struct FuncArrays* Fa,
	__global double a[],
	__global double* alpha,
	__global double* beta,
	int mrqnum)
	//__read_only int Numfac,
	//__read_only int Mmax,
	//__read_only int Lmax)
{
	int j, k;
	int brtmph, brtmpl;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	/*brtmph = Fa->Numfac / BLOCK_DIM;
	if (Fa->Numfac % BLOCK_DIM)
	{
		brtmph++;
	}

	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Fa->Numfac)
	{
		brtmph = Fa->Numfac;
	}

	brtmpl++;*/
	//printf("brtmpl: %d, brtmph: %d\n", brtmpl, brtmph);

	/* N.B. curv and blmatrix called outside bright
	   because output same for all points */
	   //curv(CUDA_LCC, Fa, a, brtmpl, brtmph, Fa->Numfac, Fa->Mmax, Fa->Lmax);
		//curv(CUDA_LCC, Fa, a, brtmpl, brtmph);

	if (threadIdx.x == 0)
	{
		/* ---  blmatrix  --- */
		double cb, sb, cl, sl, bet, lam;
		bet = a[Fa->ma - 4 - Fa->Nphpar];
		lam = a[Fa->ma - 3 - Fa->Nphpar];

		cb = cos(bet);
		sb = sin(bet);
		cl = cos(lam);
		sl = sin(lam);
		int x = blockIdx.x;
		Fa->Blmat[x][1][1] = cb * cl;
		Fa->Blmat[x][1][2] = cb * sl;
		Fa->Blmat[x][1][3] = -sb;
		Fa->Blmat[x][2][1] = -sl;
		Fa->Blmat[x][2][2] = cl;
		Fa->Blmat[x][2][3] = 0;
		Fa->Blmat[x][3][1] = sb * cl;
		Fa->Blmat[x][3][2] = sb * sl;
		Fa->Blmat[x][3][3] = cb;

		/* Ders. of Blmat w.r.t. bet */
		Fa->Dblm[x][1][1][1] = -sb * cl;
		Fa->Dblm[x][1][1][2] = -sb * sl;
		Fa->Dblm[x][1][1][3] = -cb;
		Fa->Dblm[x][1][2][1] = 0;
		Fa->Dblm[x][1][2][2] = 0;
		Fa->Dblm[x][1][2][3] = 0;
		Fa->Dblm[x][1][3][1] = cb * cl;
		Fa->Dblm[x][1][3][2] = cb * sl;
		Fa->Dblm[x][1][3][3] = -sb;
		/* Ders. w.r.t. lam */
		Fa->Dblm[x][2][1][1] = -cb * sl;
		Fa->Dblm[x][2][1][2] = cb * cl;
		Fa->Dblm[x][2][1][3] = 0;
		Fa->Dblm[x][2][2][1] = -cl;
		Fa->Dblm[x][2][2][2] = -sl;
		Fa->Dblm[x][2][2][3] = 0;
		Fa->Dblm[x][2][3][1] = -sb * sl;
		Fa->Dblm[x][2][3][2] = sb * cl;
		Fa->Dblm[x][2][3][3] = 0;


		//printf("blmatrix >>> [%d][%d]: cb: %.6f, cl: %.6f, sb: %.6f, Dblm[1][3][3]: % .6f\n", blockIdx.x, threadIdx.x, cb, cl, sb, Fa->Dblm[x][1][3][3]);
		/*  ---- END blmatrix ---  */

		//blmatrix(CUDA_LCC, a[Fa->ma - 4 - Fa->Nphpar], a[Fa->ma - 3 - Fa->Nphpar]);
		

		(*CUDA_LCC).trial_chisq = 0;
		(*CUDA_LCC).np = 0;
		(*CUDA_LCC).np1 = 0;
		(*CUDA_LCC).np2 = 0;
		(*CUDA_LCC).ave = 0;

		/*if (blockIdx.x == 1) {
			printf("CUDA_ma: %d, CUDA_Nphpar: %d, (*CUDA_LCC).np: %d\n", Fa->ma, Fa->Nphpar, (*CUDA_LCC).np);
			printf("%f, %f\n", a[Fa->ma - 4 - Fa->Nphpar], a[Fa->ma - 3 - Fa->Nphpar]);
			printf("cg[1]: %.6f\n", a[1]);
			printf("cg[%d]: %.6f\n cg[%d]: %.6f\n", Fa->ma - 4 - Fa->Nphpar, a[Fa->ma - 4 - Fa->Nphpar], Fa->ma - 3 - Fa->Nphpar, a[Fa->ma - 3 - Fa->Nphpar]);
		}*/

		//if (blockIdx.x == 1)
		//{
		//	printf("mrqcof >>> [%d][%d]: \t% .6f, % .6f, % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).Blmat[3][1], (*CUDA_LCC).Blmat[3][1], (*CUDA_LCC).Blmat[3][1]);
		//	//printf("[%d][%d]: \t% .6f, % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).e_3[jp], (*CUDA_LCC).e0_3[jp]);
		//}
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
			alpha[j * (Fa->Lmfit1) + k] = 0.0;

			//if (blockIdx.x == 1 && threadIdx.x == 0)
			//{
			//	printf("mrqcof_start >>> [%d][%d] alpha[%d]: % .6f\n", blockIdx.x, threadIdx.x, j * (Fa->Lmfit1) + k, alpha[j * (Fa->Lmfit1) + k]);
			//}
		}

		// TODO: Coment this in when done investigating beta[39] - beta[46] issue!
		beta[j] = 0.0;
		//Fa->beta[blockIdx.x][j] = 0;

		//if (blockIdx.x == 0)
		//	printf("mrqcof_start >>> [%d][%d][idx]  beta[%d]: % .6f\n", blockIdx.x, threadIdx.x, j, beta[idx]);
	}

	//int idx = (blockIdx.x * (MAX_N_PAR + 1) + threadIdx.x);
	//beta[idx] = 1.0;
	//write_mem_fence(CLK_GLOBAL_MEM_FENCE);

	////__syncthreads(); //for sure
	//write_mem_fence(CLK_GLOBAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

double mrqcof_end(__global struct freq_context2* CUDA_LCC, 
	__global varholder* Fa, 
	__global double* alpha)
{
	int j, k;
	int3 blockIdx;
	blockIdx.x = get_global_id(0);

	for (j = 2; j <= Fa->Lmfit; j++)
	{
		for (k = 1; k <= j - 1; k++)
		{
			alpha[k * Fa->Lmfit1 + j] = alpha[j * Fa->Lmfit1 + k];

			//if(blockIdx.x == 0)
			//	printf("[%d] alpha[%d]: % .9f\n", blockIdx.x, k * Fa->Lmfit1 + j, alpha[k * Fa->Lmfit1 + j]);
		}
	}

	// TODO: Check this insain return
	return (*CUDA_LCC).trial_chisq;
}

void mrqcof_matrix(__global struct freq_context2* CUDA_LCC, __global varholder* Fa, double a[], int Lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);

	// NOTE: AMD APP SDK OpenCL Optimization Guide (2015): http://developer.amd.com/wordpress/media/2013/12/AMD_OpenCL_Programming_Optimization_Guide2.pdf
	/*
		3.1.2.2 Reads Of The Same Address:
		Under certain conditions, one unexpected case of a channel conflict is that
		reading from the same address is a conflict, even on the FastPath.

		This does not happen on the read-only memories, such as constant buffers,
		textures, or shader resource view (SRV); but it is possible on the read/write UAV
		memory or OpenCL global memory.
		From a hardware standpoint, reads from a fixed address have the same upper
		bits, so they collide and are serialized. To read in a single value, read the value
		in a single work-item, place it in local memory, and then use that location:
		Avoid:
			temp = input[3] // if input is from global space

		Use:
			if (get_local_id(0) == 0) {
			local = input[3]
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			temp = local

	*/

	/*__local int Np;
	if (get_group_id(0) == 0)
	{
		Np = (*CUDA_LCC).np;
	}

	barrier(CLK_LOCAL_MEM_FENCE);*/

	printf("mrqcof_matrix >>>\n");

	//matrix_neo(CUDA_LCC, Fa, a, (*CUDA_LCC).np, Lpoints);
	//matrix_neo(CUDA_LCC, Fa, a, Np, Lpoints);
}

void mrqcof_curve1(
	__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	/*__global int2* texArea,
	__global int2* texDg,*/
	double a[],
	//double alpha,
	//double beta[],
	int brtmpl,
	int brtmph, 
	int Inrel,
	int Lpoints, 
	int num)
{
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	__local	double tmave[BLOCK_DIM]; // NOTE: __shared__
	//int j_p;
	__private int l, k, j_p, lnp, Lpoints1 = Lpoints + 1;
	__private double lave;

	lnp = (*CUDA_LCC).np;
	lave = (*CUDA_LCC).ave;
	//barrier(CLK_LOCAL_MEM_FENCE) // TODO: Test it. If it is needed put everithing inside first check for globalIdx.x == 0). Otherwise just delete this line.

	//if (threadIdx.x == 0) {
	//	printf("mrqcof_curve1 >>> [%d][%d]\t%d,  % .6f\n", blockIdx.x, threadIdx.x, lnp, (*CUDA_LCC).ave);
	//}

	//precalc thread boundaries
	/*__private int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM)
	{
		brtmph++;
	}

	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints)
	{
		brtmph = Lpoints;
	}

	brtmpl++;*/

	/*if (blockIdx.x == 1) 
		printf("[%d][%d]\tbrtmpl: %d\n", blockIdx.x, threadIdx.x, brtmpl);*/

	/*if(blockIdx.x == 0)
		printf("[%d][%d]  \tbrtmpl: %d,\tbrtmph: %d\n", blockIdx.x, threadIdx.x, brtmpl, brtmph);*/

	// NOTE: Don't use function calls inside for lops with OpenCl 1.2 as there is an issue https://stackoverflow.com/questions/23986825/opencl-for-loop-execution-model
	// TODO:Have to be tested with OpenCl 2.0 and higher versions
	//for (j_p = brtmpl; j_p <= brtmph; j_p++)
	//{
	//	bright(CUDA_LCC, Fa, (*CUDA_LCC).cg, j_p, Lpoints1, Inrel);
	//	//bright(CUDA_LCC, Fa, a, brtmpl, brtmph, Lpoints1, Inrel);
	//	if (blockIdx.x == 1)
	//		printf("thread[%d], brtmpl: %d, brtmph: %d, jp: %d\n", threadIdx.x, brtmpl, brtmph, j_p);
	//}

	//j_p = brtmph;
	//barrier(CLK_LOCAL_MEM_FENCE);
	//bright(CUDA_LCC, Fa, a, j_p, Lpoints1, Inrel);
	//do {
	//	//if (jp > brtmph) return;
	//	//bright(CUDA_LCC, Fa, a, jp, Lpoints1, Inrel);	

	//	bright(CUDA_LCC, Fa, a, j_p, Lpoints1, Inrel);	
	//	//if(blockIdx.x == 2)
	//	//	printf("thread[%d], brtmpl: %d, brtmph: %d, jp: %d\n", threadIdx.x, brtmpl, brtmph, jp);

	//	j_p--;
	//} while (j_p >= brtmpl);

	//__syncthreads();
	//barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

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

			// TODO: there are couple of differences compared to cuda_app
			//  2/49  dytemp[7851]:  0.198331
			//  2/50  dytemp[8008]: -1.785283
//  
			//if (blockIdx.x == 9 && num == 1)
			//	printf("%2d/%2d  dytemp[%3d]: % .6f\n", blockIdx.x, threadIdx.x, ixx, (*CUDA_LCC).dytemp[ixx]);

			//jp>=2
			ixx++;
			for (int jp = 2; jp <= Lpoints; jp++, ixx++)
			{
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dytemp[ixx];
			}

		}

		tmave[threadIdx.x] = 0;
		for (int jp = brtmpl; jp <= brtmph; jp++)
		{
			tmave[threadIdx.x] += (*CUDA_LCC).ytemp[jp];
			
			//if (blockIdx.x == 2 && num == 1)
			//	printf("[%d][%d]  \tytemp[%d]: % .6f\n", blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).ytemp[jp]);
				//printf("[%d][%d]  \ttmave[%d]: % .6f\n", blockIdx.x, threadIdx.x, threadIdx.x, tmave[threadIdx.x]);
		}
		

		//__syncthreads();
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		
		//if (blockIdx.x == 2 && num == 1) {
		//	printf("[%2d][%3d] tmave[%d]: % .6f\n", blockIdx.x, threadIdx.x, threadIdx.x, tmave[threadIdx.x]);
		//}

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
		if (threadIdx.x == 0)
		{
			lave = tmave[0] + tmave[1];
			//if (blockIdx.x == 2 && num == 1)
			//	printf("[%d][%d]  \tlave: % .6f\n", blockIdx.x, threadIdx.x, lave);
		}
		//parallel reduction end

	}

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np = lnp + Lpoints;
		(*CUDA_LCC).ave = lave;

		//if (blockIdx.x == 2)
		//	printf("mrqcof_curve1 >> [%d][%d]  \tave: % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).ave);
	}
}

void mrqcof_curve1_last(
	__global struct freq_context2* CUDA_LCC, 
	__global varholder* Fa, 
	//double a[], 
	//__global double alpha[], 
	//__global double beta[], 
	int Inrel, 
	int Lpoints)
{
	int l, jp, lnp;
	double ymod, lave;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

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

		//ymod = conv(CUDA_LCC, Fa, texArea, texDg, jp - 1, tmpl, tmph, brtmpl, brtmph);
		ymod = conv(CUDA_LCC, Fa, jp - 1, tmpl, tmph, brtmpl, brtmph);

		if (threadIdx.x == 0)
		{
			(*CUDA_LCC).ytemp[jp] = ymod;
			if (Inrel == 1)
			{
				lave = lave + ymod;
			}

			//if (blockIdx.x == 2)
			//	printf("[%d][%d] ytemp[%d]: % .16f lave: % .16f\n", blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).ytemp[jp], lave);
		}

		for (l = tmpl; l <= tmph; l++)
		{
			(*CUDA_LCC).dytemp[jp + l * (Lpoints + 1)] = (*CUDA_LCC).dyda[l];
			if (Inrel == 1)
			{
				(*CUDA_LCC).dave[l] = (*CUDA_LCC).dave[l] + (*CUDA_LCC).dyda[l];
			}

			// NOTE: Here we get some tiny differences against CUDA calculations in 13 - 16 symbol after decimal place
			//if (blockIdx.x == 2) {
			//	int idx = jp + l * (Lpoints + 1);
			//	printf("[%d][%d] dytemp[%d]: % .16f dave[%d]: % .16f\n", blockIdx.x, threadIdx.x, idx, (*CUDA_LCC).dytemp[idx], l, (*CUDA_LCC).dave[l]);
			//}
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

		//if (blockIdx.x == 2)
		//	printf("mrqcof_curve1_last >> [%d][%d] ave: % .6f, np: %5d\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).ave, (*CUDA_LCC).np);
	}
}

