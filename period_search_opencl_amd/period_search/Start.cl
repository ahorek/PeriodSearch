#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

// void PrintSizes() {
// 	int3 groupIdx;
// 	int3 globalIdx;
// 	int3 localIdx;
// 	int3 numGroups;
// 	int3 globalSize;
// 	int3 localSize;
// 	groupIdx.x = get_group_id(0);
// 	groupIdx.y = get_group_id(1);
// 	globalIdx.x = get_global_id(0);
// 	globalIdx.y = get_global_id(1);
// 	localIdx.x = get_local_id(0);
// 	int dims = get_work_dim();
// 	numGroups.x = get_num_groups(0);
// 	numGroups.y = get_num_groups(1);
// 	globalSize.x = get_local_size(0);
// 	globalSize.y = get_local_size(1);
// 	localSize.x = get_local_size(0);
// 	localSize.y = get_local_size(1);

// 	if (globalIdx.x == 0 && globalIdx.y == 0)
// 	{
// 		printf("Number of work dimensions: %d\n", dims);
// 		printf("Number of goroups in dimensions{0, 1}: %d, %d\n", numGroups.x, numGroups.y);
// 		printf("Number of GLOBAL work items in group per dimension{0, 1}: %d, %d\n", globalSize.x, globalSize.y);
// 		printf("Number of LOCAL work items in group per dimention{0, 1}: %d, %d\n", localSize.x, localIdx.y);
// 	}
// 	if (globalIdx.x == 0)
// 	{
// 		printf("groupIdx.x: %d, groupIdx.y: %d\n", groupIdx.x, groupIdx.y);
// 	}
// }

__kernel void CLCalculatePrepare(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__read_only int max_test_periods,
	__read_only int n_start,
	double freq_start,
	double freq_step)
{
	//PrintSizes();

	int3 blockIdx, threadIdx;
	//blockIdx.x = get_local_id(0);
	threadIdx.x = get_group_id(0);

	//printf("localId[%d]\n", blockIdx.x);

	//struct freq_result* CUDA_LCC = &CUDA_CC[idx];
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[threadIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[threadIdx.x];

	/*if (blockIdx.x == 0)
		printf("n_start: %d\n", n_start);*/

	int n = n_start + threadIdx.x;

	//zero context
	//	CUDA_CC is zeroed itself as global memory but need to reset between freq TODO
	if (n > max_test_periods)
	{
		(*CUDA_LCC).isInvalid = 1;
		{
			return;
		}
	}
	else
	{
		(*CUDA_LCC).isInvalid = 0;
	}

	(*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;
	printf("CUDA_LCC[%d].freq = %.6f\n", threadIdx.x, (*CUDA_LCC).freq);

	/* initial poles */
	(*CUDA_FR).per_best = 0;
	(*CUDA_FR).dark_best = 0;
	(*CUDA_FR).la_best = 0;
	(*CUDA_FR).be_best = 0;
	(*CUDA_FR).dev_best = 1e40;

	// Remove when done testing!
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


__kernel void CLCalculatePreparePole(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__global struct FuncArrays* Fa,
	__global double* lambda_pole,
	__global double* beta_pole,
	__global int* CUDA_End,
	__read_only int m,
	__global double* cgFirst)
	//int Ncoef,
	//int Nphpar,
	//int Numfac,
	//__global struct* FuncArrays FaRes)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];
	
	int localId = get_local_id(0);
	//if ( blockIdx.x == 0)
		printf("[%d] Ncoef = %d, Nphpar = %d, isInvalid = %d\n", blockIdx.x, Fa->Ncoef, Fa->Nphpar, (*CUDA_LCC).isInvalid);


	if ((*CUDA_LCC).isInvalid)
	{
		printf("END!\n");
		atomic_add(CUDA_End, 1);
		(*CUDA_LFR).isReported = 0; //signal not to read result

		return;
	}

	double period = 1 / (*CUDA_LCC).freq;

	/* starts from the initial ellipsoid */
	for (int i = 1; i <= Fa->Ncoef; i++)
	{
		(*CUDA_LCC).cg[i] = cgFirst[i];
		//if (get_local_id(0) == 0)
		//	printf("cg[%d]: %.6f\n", i, (*CUDA_LCC).cg[i]); // <<<<<<<
	}

	(*CUDA_LCC).cg[Fa->Ncoef + 1] = beta_pole[m];
	(*CUDA_LCC).cg[Fa->Ncoef + 2] = lambda_pole[m];

	/* The formulas use beta measured from the pole */
	(*CUDA_LCC).cg[Fa->Ncoef + 1] = 90 - (*CUDA_LCC).cg[Fa->Ncoef + 1];

	/* conversion of lambda, beta to radians */
	(*CUDA_LCC).cg[Fa->Ncoef + 1] = DEG2RAD * (*CUDA_LCC).cg[Fa->Ncoef + 1];
	(*CUDA_LCC).cg[Fa->Ncoef + 2] = DEG2RAD * (*CUDA_LCC).cg[Fa->Ncoef + 2];
	//if (get_local_id(0) == 0) {
	//
	//	printf("cg[%d]: %.6f\n", Fa->Ncoef + 1, (*CUDA_LCC).cg[Fa->Ncoef + 1]); // <<<<<<<<<<
	//	printf("cg[%d]: %.6f\n", Fa->Ncoef + 2, (*CUDA_LCC).cg[Fa->Ncoef + 2]); // <<<<<<<<<<
	//}

	/* Use omega instead of period */
	(*CUDA_LCC).cg[Fa->Ncoef + 3] = 24 * 2 * M_PI / period;
	//if (get_local_id(0) == 0)
	//	printf("cg[%d]: %.6f\n", Fa->Ncoef + 3, (*CUDA_LCC).cg[Fa->Ncoef + 3]); // <<<<<<<<<<

	for (int i = 1; i <= Fa->Nphpar; i++)
	{
		(*CUDA_LCC).cg[Fa->Ncoef + 3 + i] = Fa->par[i];
		//              ia[Ncoef+3+i] = ia_par[i]; moved to global
		//if (get_local_id(0) == 0)
		//	printf("cg[%d]: %.6f\t(cg[Fa->Ncoef + 3 + i])\n", Fa->Ncoef + 3 + i, (*CUDA_LCC).cg[Fa->Ncoef + 3 + i]); // <<<<<<<<
	}

	//if (blockIdx.x == 0)
	//	printf("cg[%d]: %.6f\n", 129, (*CUDA_LCC).cg[129]); // <<<<<<<<


	/* Lommel-Seeliger part */
	(*CUDA_LCC).cg[Fa->Ncoef + 3 + Fa->Nphpar + 2] = 1;

	/* Use logarithmic formulation for Lambert to keep it positive */
	(*CUDA_LCC).cg[Fa->Ncoef + 3 + Fa->Nphpar + 1] = Fa->logCl;

	//if (get_local_id(0) == 0) {
	//	printf("cg[%d]: %.6f\n", Fa->Ncoef + 3 + Fa->Nphpar + 2, (*CUDA_LCC).cg[Fa->Ncoef + 3 + Fa->Nphpar + 2]); // <<<<<<<<
	//	printf("cg[%d]: %.6f\n", Fa->Ncoef + 3 + Fa->Nphpar + 1, (*CUDA_LCC).cg[Fa->Ncoef + 3 + Fa->Nphpar + 1]); // <<<<<<<<
	//
	//}

	/* Levenberg-Marquardt loop */
	// moved to global iter_max,iter_min,iter_dif_max
	//
	(*CUDA_LCC).rchisq = -1;
	(*CUDA_LCC).Alamda = -1;
	(*CUDA_LCC).Niter = 0;
	(*CUDA_LCC).iter_diff = 1e40;
	(*CUDA_LCC).dev_old = 1e30;
	(*CUDA_LCC).dev_new = 0;
	//	(*CUDA_LCC).Lastcall=0; always ==0
	(*CUDA_LFR).isReported = 0;
	//printf("IsInvalid: %d\t", (*CUDA_LCC).isInvalid);
	//printf(".");
}

__kernel void CLCalculateIter1Begin(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__global int* CUDA_End,
	__read_only int n_iter_min,
	__read_only int n_iter_max,
	__read_only double iter_diff_max,
	__read_only double aLambda_start)
{
	int3 globalIdx;
	globalIdx.x = get_group_id(0);
	//printf("%d\n", b);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[globalIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[globalIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	(*CUDA_LCC).isNiter = (((*CUDA_LCC).Niter < n_iter_max) && ((*CUDA_LCC).iter_diff > iter_diff_max)) || ((*CUDA_LCC).Niter < n_iter_min);
	if ((*CUDA_LCC).isNiter)
	{
		if ((*CUDA_LCC).Alamda < 0)
		{
			(*CUDA_LCC).isAlamda = 1;
			(*CUDA_LCC).Alamda = aLambda_start; /* initial alambda */
		}
		else
		{
			(*CUDA_LCC).isAlamda = 0;
		}
	}
	else
	{
		if (!(*CUDA_LFR).isReported)
		{
			atomic_add(CUDA_End, 1);
			(*CUDA_LFR).isReported = 1;
		}
	}

	//printf("IsInvalid: %d\t", (*CUDA_LCC).isInvalid);
}

__kernel void CLCalculateIter1Mrqcof1Start(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
	//__read_only int Numfac,
	//__read_only int Mmax,
	//__read_only int Lmax)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);
	//printf("gID[%d] ", get_group_id(0));
	//if (get_global_id(0) == 0) // && get_local_id(0) == 0)
	//{
	//	printf("CalculateIter1Mrqcof1Start >>>\n");
	//	printf("global_size(x.y.z): %d, %d, %d\n", get_global_size(0), get_global_size(1), get_global_size(2));
	//	printf("local_size(x.y.z): %d, %d, %d\n", get_local_size(0), get_local_size(1), get_local_size(2));
	//	printf("number_of_groups(x.y.z): %d, %d, %d\n", get_num_groups(0), get_num_groups(1), get_num_groups(2));
	//}

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_CC).isInvalid) return;

	if (!(*CUDA_CC).isNiter) return;

	if (!(*CUDA_CC).isAlamda) return;

	//printf("m=group_%d\n", blockIdx.y);
	//mrqcof_start(CUDA_LCC, Fa, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, Numfac, Mmax, Lmax);
	//mrqcof_start((*CUDA_LCC).isAlamda);

	//printf("groupId[%d], localId[%d], globalId[%d]\n", get_group_id(0), get_local_id(0), get_global_id(0));
	mrqcof_start(CUDA_LCC, Fa, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);

	//if (blockIdx.x == 1)
	//{
	//	printf("mrqcof >>> [%d][%d]: \t% .6f, % .6f, % .6f\n", blockIdx.x, get_local_id(0), (*CUDA_LCC).Blmat[3][1], (*CUDA_LCC).Blmat[3][1], (*CUDA_LCC).Blmat[3][1]);
	//	//printf("[%d][%d]: \t% .6f, % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).e_3[jp], (*CUDA_LCC).e0_3[jp]);
	//}
}

__kernel void CLCalculateIter1Mrqcof1Matrix(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);

	//if (get_global_id(0) == 0) // && get_local_id(0) == 0)
	//{
	//	printf("CalculateIter1Mrqcof1Matrix >>>\n");
	//	printf("global_size(x.y.z): %d, %d, %d\n", get_global_size(0), get_global_size(1), get_global_size(2));
	//	printf("local_size(x.y.z): %d, %d, %d\n", get_local_size(0), get_local_size(1), get_local_size(2));
	//	printf("number_of_groups(x.y.z): %d, %d, %d\n", get_num_groups(0), get_num_groups(1), get_num_groups(2));
	//}


	//printf("WDim[%d], Dim[%d, %d, %d]\n", get_work_dim(), get_global_size(0), get_num_groups(0), get_local_size(0));
	//printf("groupId[%d], localId[%d], globalId[%d]\n", get_group_id(0), get_local_id(0), get_global_id(0));
	
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	//barrier(CLK_LOCAL_MEM_FENCE);

	/*if (get_local_id(0) == 1) {
		printf("mrqcof_matrix[%d] >> (*CUDA_LCC).cg[129] = %.6f\n", get_group_id(0), (*CUDA_LCC).cg[129]);
	}*/

	mrqcof_matrix(CUDA_LCC, Fa, (*CUDA_LCC).cg, lpoints);
}

__kernel void CLCalculateIter1Mrqcof1Curve1(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	__global int2* texArea,
	__global int2* texDg,
	const int inrel,
	const int lpoints)
{

	int3 blockIdx;
	blockIdx.x = get_group_id(0); //get_global_id(0);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).cg, (*CUDA_LCC).beta, inrel, lpoints); //(*CUDA_LCC).alpha,
}

__kernel void CLCalculateIter1Mrqcof1Curve1Last(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	__global int2* texArea,
	__global int2* texDg,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	//mrqcof_curve1_last(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__kernel void CLCalculateIter1Mrqcof1End(
	__global struct freq_context2* CUDA_CC,
	__global struct FuncArrays* Fa)
	//__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	//(*CUDA_LCC).Ochisq = mrqcof_end(CUDA_LCC, Fa, (*CUDA_LCC).alpha);
}


int next2(int a, int b) {
	return a + b;
}

__kernel void CLCalculateIter1Mrqmin1End(
	__global struct freq_context2* CUDA_CC,
	__global struct FuncArrays* Fa,
	__global int* CUDA_End)
	//__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	//__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	//printf("IsInvalid: %d\t", CUDA_CC[blockIdx.x].isInvalid);
	//if ((*CUDA_LCC).isInvalid)
	//{
	//	return;
	//}
	/*
	if (!(*CUDA_LCC).isNiter)
	{
		return;
	}*/

	//gauss_err=
	//mrqmin_1_end(CUDA_LCC, Fa);  //CUDA_ma, CUDA_mfit, CUDA_mfit1, blockDim);
	int ans = next2((*CUDA_End), blockIdx.x);
	printf("> %d\n", ans);
}

__kernel void CLCalculateIter1Mrqcof2Start(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
	//__read_only int Numfac)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//mrqcof_start(CUDA_LCC, Fa, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, Numfac, 0, 0);
	mrqcof_start(CUDA_LCC, Fa, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);
}

__kernel void CLCalculateIter1Mrqcof2Matrix(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//mrqcof_matrix(CUDA_LCC, Fa, (*CUDA_LCC).atry, lpoints);
}

__kernel void CLCalculateIter1Mrqcof2Curve1(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	__global int2* texArea,
	__global int2* texDg,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//mrqcof_curve1(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).atry, (*CUDA_LCC).da, inrel, lpoints); // (*CUDA_LCC).covar,
}

__kernel void CLCalculateIter1Mrqcof2Curve1Last(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	__global int2* texArea,
	__global int2* texDg,
	//__local double* res,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//mrqcof_curve1_last(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}

__kernel void CLCalculateIter1Mrqcof2End(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	//(*CUDA_LCC).Chisq = mrqcof_end(CUDA_LCC, Fa, (*CUDA_LCC).covar);
}

__kernel void CLCalculateIter1Mrqmin2End(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	mrqmin_2_end(CUDA_LCC, Fa);

	(*CUDA_LCC).Niter++;
}


__kernel void CLCalculateIter2(
	__global struct freq_context2* CUDA_CC,
	__global struct FuncArrays* Fa)
{
	int3 blockIdx;
	int3 threadIdx;
	blockIdx.x = get_global_id(0);
	threadIdx.x = get_local_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	if ((*CUDA_LCC).isNiter)
	{
		if ((*CUDA_LCC).Niter == 1 || (*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
		{
			if (threadIdx.x == 0)
			{
				(*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;
			}

			//__syncthreads();
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

			int brtmph = (*Fa).Numfac / BLOCK_DIM;
			if ((*Fa).Numfac % BLOCK_DIM) brtmph++;
			int brtmpl = threadIdx.x * brtmph;
			brtmph = brtmpl + brtmph;
			if (brtmph > (*Fa).Numfac) brtmph = (*Fa).Numfac;
			brtmpl++;

			//curv(CUDA_LCC, Fa, (*CUDA_LCC).cg, brtmpl, brtmph);

			if (threadIdx.x == 0)
			{
				for (int i = 1; i <= 3; i++)
				{
					(*CUDA_LCC).chck[i] = 0;
					for (int j = 1; j <= (*Fa).Numfac; j++)
					{
						(*CUDA_LCC).chck[i] = (*CUDA_LCC).chck[i] + (*CUDA_LCC).Area[j] * Fa->Nor[j][i - 1];
					}
				}
				(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - (pow((*CUDA_LCC).chck[1], 2) + pow((*CUDA_LCC).chck[2], 2) + pow((*CUDA_LCC).chck[3], 2)) * pow(Fa->Conw_r, 2);
			}
		}

		if (threadIdx.x == 0)
		{
			(*CUDA_LCC).dev_new = sqrt((*CUDA_LCC).rchisq / (Fa->Ndata - 3));
			/* only if this step is better than the previous,
				1e-10 is for numeric errors */
			if ((*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new > 1e-10)
			{
				(*CUDA_LCC).iter_diff = (*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new;
				(*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
			}
			//		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
		}
	}
}

__kernel void CLCalculateFinishPole(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	double totarea = 0;
	for (int i = 1; i <= Fa->Numfac; i++)
	{
		totarea = totarea + (*CUDA_LCC).Area[i];
	}

	double sum = pow((*CUDA_LCC).chck[1], 2) + pow((*CUDA_LCC).chck[2], 2) + pow((*CUDA_LCC).chck[3], 2);
	double dark = sqrt(sum);

	/* period solution */
	double period = 2 * PI / (*CUDA_LCC).cg[Fa->Ncoef + 3];

	/* pole solution */
	double la_tmp = RAD2DEG * (*CUDA_LCC).cg[Fa->Ncoef + 2];
	double be_tmp = 90 - RAD2DEG * (*CUDA_LCC).cg[Fa->Ncoef + 1];

	if ((*CUDA_LCC).dev_new < (*CUDA_LFR).dev_best)
	{
		(*CUDA_LFR).dev_best = (*CUDA_LCC).dev_new;
		(*CUDA_LFR).per_best = period;
		(*CUDA_LFR).dark_best = dark / totarea * 100;
		(*CUDA_LFR).la_best = la_tmp;
		(*CUDA_LFR).be_best = be_tmp;
	}
	//debug
	/*	(*CUDA_LFR).dark=dark;
	(*CUDA_LFR).totarea=totarea;
	(*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
	(*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
	(*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
}

__kernel void CLCalculateFinish(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((*CUDA_LCC).isInvalid)
	{
		return;
	}

	if ((*CUDA_LFR).la_best < 0)
	{
		(*CUDA_LFR).la_best += 360;
	}

	if (isnan((*CUDA_LFR).dark_best) == 1)
	{
		(*CUDA_LFR).dark_best = 1.0;
	}
}
