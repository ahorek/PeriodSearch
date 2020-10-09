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
	double freq_step,
	__global struct FuncArrays* Fa)
{
	//PrintSizes();

	int3 blockIdx, threadIdx;
	blockIdx.x = get_local_id(0);
	threadIdx.x = get_group_id(0);

	//printf("localId[%d]\n", blockIdx.x);

	//struct freq_result* CUDA_LCC = &CUDA_CC[idx];
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[threadIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[threadIdx.x];

	if (blockIdx.x == 0)
		printf("n_start: %d, id: %d, max_test_periods: %d\n", n_start, threadIdx.x, max_test_periods);

	int n = n_start + threadIdx.x;

	//zero context
	//	CUDA_CC is zeroed itself as global memory but need to reset between freq TODO
	if (n > max_test_periods)
	{
		Fa->isInvalid[blockIdx.x] = 1;
		{
			return;
		}
	}
	else
	{
		Fa->isInvalid[blockIdx.x] = 0;
	}

	(*CUDA_LCC).freq = freq_start - (n - 1) * freq_step;
	printf("CUDA_LCC[%d].freq = %.6f\n", threadIdx.x, (*CUDA_LCC).freq);

	/* initial poles */
	(*CUDA_FR).per_best = 0;
	(*CUDA_FR).dark_best = 0;
	(*CUDA_FR).la_best = 0;
	(*CUDA_FR).be_best = 0;
	(*CUDA_FR).dev_best = 1e40;

	// TODO: Remove when done testing!
	//barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
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
	printf("[%d] Ncoef = %d, Nphpar = %d, isInvalid = %d\n", blockIdx.x, Fa->Ncoef, Fa->Nphpar, Fa->isInvalid[blockIdx.x]);


	if (Fa->isInvalid[blockIdx.x])
	{
		printf("END!\n");
		atomic_add(CUDA_End, 1);
		Fa->isReported[blockIdx.x] = 0; //signal not to read result

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
	Fa->isReported[blockIdx.x] = 0;
	//printf("IsInvalid: %d\t", Fa->isInvalid[blockIdx.x]);
	//printf(".");
}

__kernel void CLCalculateIter1Begin(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__global int* CUDA_End,
	__read_only int n_iter_min,
	__read_only int n_iter_max,
	__read_only double iter_diff_max,
	__read_only double aLambda_start,
	__global varholder* Fa)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);
	//printf("%d\n", b);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Iter1Mrqcof1Matrix >>> isInvalid: %d\n", Fa->isInvalid[blockIdx.x]);

	if (Fa->isInvalid[blockIdx.x])
	{
		return;
	}

	//Fa->isNiter[blockIdx.x] = (((*CUDA_LCC).Niter < n_iter_max) && ((*CUDA_LCC).iter_diff > iter_diff_max)) || ((*CUDA_LCC).Niter < n_iter_min);
	Fa->isNiter[blockIdx.x] = (((*CUDA_LCC).Niter < n_iter_max) && ((*CUDA_LCC).iter_diff > iter_diff_max)) || ((*CUDA_LCC).Niter < n_iter_min);

	if (Fa->isNiter[blockIdx.x])
	{
		if ((*CUDA_LCC).Alamda < 0)
		{
			Fa->isAlamda[blockIdx.x] = 1;
			(*CUDA_LCC).Alamda = aLambda_start; /* initial alambda */
		}
		else
		{
			Fa->isAlamda[blockIdx.x] = 0;
		}
	}
	else
	{
		if (!Fa->isReported[blockIdx.x])
		{
			atomic_add(CUDA_End, 1);
			Fa->isReported[blockIdx.x] = 1;
		}
	}

	//printf("IsInvalid: %d\t", Fa->isInvalid[blockIdx.x]);
	//if (blockIdx.x == 0) // && threadIdx.x == 0)
	//	printf("Iter1Mrqcof1Matrix >>> isInvalid: %d, isNiter: %d, isAlamda: %d, Alamda: % .9f\n", Fa->isInvalid[blockIdx.x], Fa->isNiter[blockIdx.x], Fa->isAlamda[blockIdx.x], (*CUDA_LCC).Alamda);
}

__kernel void CLCalculateIter1Mrqcof1Start(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
	//__global double* alpha,
	//__global double* beta)
	//__read_only int Numfac,
	//__read_only int Mmax,
	//__read_only int Lmax)
{
	int brtmph, brtmpl;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);
	//printf("gID[%d] ", get_group_id(0));
	//if (get_global_id(0) == 0) // && get_local_id(0) == 0)
	//{
	//	printf("CalculateIter1Mrqcof1Start >>>\n");
	//	printf("global_size(x.y.z): %d, %d, %d\n", get_global_size(0), get_global_size(1), get_global_size(2));
	//	printf("local_size(x.y.z): %d, %d, %d\n", get_local_size(0), get_local_size(1), get_local_size(2));
	//	printf("number_of_groups(x.y.z): %d, %d, %d\n", get_num_groups(0), get_num_groups(1), get_num_groups(2));
	//}

	__global struct freq_context2* CUDA_LCC;
	CUDA_LCC = &CUDA_CC[blockIdx.x];

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//	printf("Iter1Mrqcof1Start >>> isInvalid: %d, isNiter: %d, isAlamda: %d\n", Fa->isInvalid[blockIdx.x], Fa->isNiter[blockIdx.x], Fa->isAlamda[blockIdx.x]);

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	if (!Fa->isAlamda[blockIdx.x]) return;

	brtmph = Fa->Numfac / BLOCK_DIM;
	if (Fa->Numfac % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Fa->Numfac) brtmph = Fa->Numfac;
	brtmpl++;

	// TODO: Call curv() here and take that call from mrqcof_start() out!
	//for (int i = brtmpl; i <= brtmph; i++)
	//{
	//	if (blockIdx.x == 1)
	//		printf("curv >> [%d][%d]  \ti: %d\tbrtmpl: %d\tbrtmph: %d\n", blockIdx.x, threadIdx.x, i, brtmpl, brtmph);*/
	curv(CUDA_LCC, Fa, (*CUDA_LCC).cg, brtmpl, brtmph);
	//}

	//__syncthreads();
	//barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


	//printf("m=group_%d\n", blockIdx.y);
	//mrqcof_start(CUDA_LCC, Fa, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, Numfac, Mmax, Lmax);
	//mrqcof_start(Fa->isAlamda[blockIdx.x]);

	//printf("groupId[%d], localId[%d], globalId[%d]\n", get_group_id(0), get_local_id(0), get_global_id(0));

	mrqcof_start(CUDA_LCC, Fa, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);

	//int idx = blockIdx.x * (MAX_N_PAR + 1);// +threadIdx.x;
	//__global double* l_beta = &beta[idx];

	//mrqcof_start(CUDA_LCC, Fa, (*CUDA_LCC).cg, alpha, l_beta);
}

__kernel void CLCalculateIter1Mrqcof1Matrix(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	/*__local int isNiter;
	if (get_group_id(0) == 0)
	{
		isNiter = Fa->isNiter[blockIdx.x];
	}*/

	if (Fa->isInvalid[blockIdx.x]) return;

	//if(blockIdx.x == 1 && threadIdx.x == 0)
	//	printf("Iter1Mrqcof1Matrix >>> isInvalid: %d, isNiter: %d, isAlamda: %d, np: %d, lpoints: %d\n", Fa->isInvalid[blockIdx.x], Fa->isNiter[blockIdx.x], Fa->isAlamda[blockIdx.x], (*CUDA_LCC).np, lpoints);

	if (!Fa->isNiter[blockIdx.x]) return;

	if (!Fa->isAlamda[blockIdx.x]) return;

	//mrqcof_matrix(CUDA_LCC, Fa, (*CUDA_LCC).cg, lpoints);
	matrix_neo(CUDA_LCC, Fa, (*CUDA_LCC).cg, (*CUDA_LCC).np, lpoints);
}

__kernel void CLCalculateIter1Mrqcof1Curve1(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	/*__global int2* texArea,
	__global int2* texDg,*/
	const int inrel,
	const int Lpoints)
{

	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	if (!Fa->isAlamda[blockIdx.x]) return;

	__private int brtmph, brtmpl;
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

	brtmpl++;

	//__local int _inrel, _lpoints, _np;
	//__local double _ave;
	//if (blockIdx.x == 0)
	//{
	//	_inrel = inrel;
	//	_lpoints = lpoints;
	//	//_np = (*CUDA_LCC).np;
	//	//_ave = (*CUDA).ave;
	//}

	for (int jp = brtmpl; jp <= brtmph; jp++) {
		bright(CUDA_LCC, Fa, (*CUDA_LCC).cg, jp, Lpoints + 1, inrel);
	}
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	/*if(threadIdx.x == 0)
		printf("1Mrqcof1Curve1 >> [%d][%d]\n", blockIdx.x, threadIdx.x);*/

		//mrqcof_curve1(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).cg, (*CUDA_LCC).beta, _inrel, _lpoints); //, _np, _ave); //(*CUDA_LCC).alpha,
	mrqcof_curve1(CUDA_LCC, Fa, (*CUDA_LCC).cg, brtmpl, brtmph, inrel, Lpoints); //, _np, _ave); //(*CUDA_LCC).alpha,
}

__kernel void CLCalculateIter1Mrqcof1Curve1Last(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	if (!Fa->isAlamda[blockIdx.x]) return;

	//mrqcof_curve1_last(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
	mrqcof_curve1_last(CUDA_LCC, Fa, inrel, lpoints);
}

__kernel void CLCalculateIter1Mrqcof1End(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	if (!Fa->isAlamda[blockIdx.x]) return;

	(*CUDA_LCC).Ochisq = mrqcof_end(CUDA_LCC, Fa, (*CUDA_LCC).alpha);

	//printf("[%d] Ochisq: % .9f\n", blockIdx.x, (*CUDA_LCC).Ochisq);
}


__kernel void CLCalculateIter1Mrqmin1End_(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
{
	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	//printf("[%d][%d]\n", blockIdx.x, threadIdx.x);

	//if(threadIdx.x == 0)
	//	printf("[%d][%d] isInvalid: %d, isNiter: %d\n", blockIdx.x, threadIdx.x, Fa->isInvalid[blockIdx.x], Fa->isNiter[blockIdx.x]);

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	//int block = CUDA_BLOCK_DIM;
	//gauss_err=
	//mrqmin_1_end(CUDA_LCC, Fa, Fa->isAlamda[blockIdx.x], threadIdx, blockIdx);  //CUDA_ma, CUDA_mfit, CUDA_mfit1, blockDim);
	mrqmin_1_end_old(CUDA_LCC, Fa, Fa->isAlamda[blockIdx.x]);  //CUDA_ma, CUDA_mfit, CUDA_mfit1, blockDim);
}

void CLCalculateIter1Mrqmin1End_01(__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa)
{
	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	/*if (Fa->isInvalid[blockIdx.x]) return;
	if (!Fa->isNiter[blockIdx.x]) return;*/

	//printf("[%d] Alamda: % .9f\n", blockIdx.x, (*CUDA_LCC).Alamda);

	// ---------- Init -----------
	__private int i, j, k, l, ll, err_code;
	
	/*precalc thread boundaries*/
	__private int tmph, tmpl;
	tmph = Fa->ma / BLOCK_DIM;
	if (Fa->ma % BLOCK_DIM) tmph++;

	tmpl = threadIdx.x * tmph;
	tmph = tmpl + tmph;
	if (tmph > Fa->ma) tmph = Fa->ma;
	tmpl++;

	int brtmph, brtmpl;
	brtmph = Fa->Lmfit / BLOCK_DIM;
	if (Fa->Lmfit % BLOCK_DIM) brtmph++;

	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Fa->Lmfit) brtmph = Fa->Lmfit;
	brtmpl++;

	if (Fa->isAlamda[blockIdx.x])
	{
		for (j = tmpl; j <= tmph; j++)
		{
			(*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];

			//if (blockIdx.x == 2)
			 //  printf("[%d][%d] atry[%d]: % .9f\n", blockIdx.x, threadIdx.x, j, (*CUDA_LCC).atry[j]);
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	//if(threadIdx.x == 0)
	//	printf("[%d] Alamda: % .9f\n", blockIdx.x, (*CUDA_LCC).Alamda);

	for (j = brtmpl; j <= brtmph; j++)
	{
		int ixx = j * Fa->Lmfit1 + 1;
		for (k = 1; k <= Fa->Lmfit; k++, ixx++)
		{
			(*CUDA_LCC).covar[ixx] = (*CUDA_LCC).alpha[ixx];
			//if (ixx == 166)
			//	printf("[%d][%d] covar[%d]: % .9f\n", blockIdx.x, threadIdx.x, ixx, (*CUDA_LCC).covar[ixx]);
		}

		int idx = j * Fa->Lmfit1 + j;
		(*CUDA_LCC).covar[idx] = (*CUDA_LCC).alpha[idx] * (1 + (*CUDA_LCC).Alamda);
		(*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];

		//if (blockIdx.x == 0)// && threadIdx.x == 5)
		//	  printf("[%d][%d] covar[%d]: % .9f, alpha[%d]: % .9f, Alamda: % .9f\n", blockIdx.x, threadIdx.x, idx, (*CUDA_LCC).covar[idx], idx, (*CUDA_LCC).alpha[idx], (*CUDA_LCC).Alamda);
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	/* 
		START of Gauss-Jordan Elimination (gauss_errc)
	*/
	__private int n = Fa->Lmfit;	// __private?
	__private int ma = Fa->ma;		// __private?
	__private double big, dum, temp;
	__private int index;
	__private int indxr;
	__private int indxc;
	__private int a;
	__private int b;
	__private double covar;
	__private double tmpcov;
	__private int irow = 0;
	__private int licol = 0;
	__private int colIdx;

	__local int icol;
	__local int ixx, jxx;
	__local double pivinv;
	__local int sh_icol[BLOCK_DIM];
	__local int sh_irow[BLOCK_DIM];
	__local double sh_big[BLOCK_DIM];
	
	if (threadIdx.x == 0)
	{
		for (j = 1; j <= n; j++)
		{
			(*CUDA_LCC).ipiv[j] = 0;
			//printf("ipiv[%d][%d]: %d\n", blockIdx.x, j, (*CUDA_LCC).ipiv[j]);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	

	/*
	*	Here we need every single thread (out of 128 in our case) to run it's own iteration of 'i' (e.g. 54).
	*	
	*   So: 128 x 54 = 6912 per block
	*	    6912 x 10 = 69120 in total
	*/
	for (i = 1; i <= n; i++)
	{
		big = 0.0f;
		irow = 0;
		licol = 0;
		for (j = brtmpl; j <= brtmph; j++)
		{
			if ((*CUDA_LCC).ipiv[j] != 1)
			{
				//err_code = gauss_errc_begin(CUDA_LCC, Fa, &big, &irow, &licol, brtmpl, brtmph, i, j);
				//if (err_code) goto end;

				ixx = j * Fa->Lmfit1 + 1;
				for (k = 1; k <= n; k++, ixx++)
				{
					//if ((*CUDA_LCC).ipiv[k] > 1)
					//	printf("ipiv[%d][%d]: %d\n", blockIdx.x, k, (*CUDA_LCC).ipiv[k]);
					if ((*CUDA_LCC).ipiv[k] == 0)
					{
						covar = (*CUDA_LCC).covar[ixx];

						//if (blockIdx.x == 0 && threadIdx.x == 2 && i == 2)
						//	printf("[%d][%d] j[%d], i[%d], k[%d], covar[%d][%d]: % .9f\n", blockIdx.x, threadIdx.x, j, i, k, blockIdx.x, ixx, (*CUDA_LCC).covar[ixx]);

						tmpcov = fabs(covar);
						if (tmpcov >= big)
						{
							big = tmpcov;
							irow = j;
							licol = k;
						}
					}
					else if ((*CUDA_LCC).ipiv[k] > 1)
					{
						printf("-");
						//if (blockIdx.x == 2)
						//	printf("[%d][%d][%d] ipiv[%d]: %d\n", blockIdx.x, threadIdx.x, i, k, (*CUDA_LCC).ipiv[k]);
						//__syncthreads();
						barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

						return;
					}
				}

				//int ixx = j * Fa->Lmfit1 + 1;
				//for (k = 1; k <= Fa->Lmfit; k++, ixx++)
				//{
				//	if (threadIdx.x == 2 && i == 2)
				//		printf("[%d][%d] j[%d], i[%d], k[%d], covar[%d][%d]: % .9f\n", blockIdx.x, threadIdx.x, j, i, k, blockIdx.x, ixx, (*CUDA_LCC).covar[ixx]);
				//}
			}
		}

		sh_big[threadIdx.x] = big;
		sh_irow[threadIdx.x] = irow;
		sh_icol[threadIdx.x] = licol;

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		//if (threadIdx.x == 12 && i == 2)
		//	printf("[%d][%d] i[%d] big: % .9f, irow: %d, licol: %d\n", blockIdx.x, threadIdx.x, i, big, irow, licol);

		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[%d][%d] i[%d] licol[0]: %d\n", blockIdx.x, threadIdx.x, i,  sh_icol[0]);
		
	//}
		// <<<<<<<<<<<<<<<<<<<<<<  gauss_errc_mid >>>>>>>>>>>>>>>>>>>>>>>>>>
	//for (i = 1; i <= n; i++)
	//{
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
			//if(blockIdx.x == 0)
			//	printf("ipiv[%d][%d]: %d\n", blockIdx.x, icol, (*CUDA_LCC).ipiv[icol]);
			if (irow != icol)
			{
				index = irow * Fa->Lmfit1 + l;
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
			colIdx = icol * Fa->Lmfit1 + icol;
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

		for (ll = brtmpl; ll <= brtmph; ll++)
		{
			if (ll != icol)
			{
				ixx = ll * (Fa->Lmfit1);
				jxx = icol * (Fa->Lmfit1);
				dum = (*CUDA_LCC).covar[ixx + icol];
				(*CUDA_LCC).covar[ixx + icol] = 0.0;

				//if (threadIdx.x == 2 && i == 1)
				//	printf("[%d][%d] covar[%d]: % .9f, dum: % .9f, irow: %d, icol: %d\n", blockIdx.x, threadIdx.x, ixx + icol, (*CUDA_LCC).covar[ixx + icol], dum, ixx, jxx);
				ixx++;
				jxx++;

				for (l = 1; l <= n; l++, ixx++, jxx++)
				{
					(*CUDA_LCC).covar[ixx] -= (*CUDA_LCC).covar[jxx] * dum;
				}

				(*CUDA_LCC).da[ll] -= (*CUDA_LCC).da[icol] * dum;
				//b[ll] -= b[icol]*dum;
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	} /* End 'i'  */

	/* ------ END GAUSS_ERRC ------  */
}


__kernel void CLCalculateIter1Mrqmin1End(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
{
	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);
	int l, j, k, indxr, indxc, a, b;
	int n = Fa->Lmfit;
	int ma = Fa->ma;

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;
	if (!Fa->isNiter[blockIdx.x]) return;
	
	CLCalculateIter1Mrqmin1End_01(CUDA_LCC, Fa);

	if (threadIdx.x == 0)
	{
		//printf("[%d][%d] <<<<<< \n", blockIdx.x, threadIdx.x);
		for (l = n; l >= 1; l--)
		{
			indxr = (*CUDA_LCC).indxr[l];
			indxc = (*CUDA_LCC).indxc[l];
			if (indxr != indxc)
			{
				for (k = 1; k <= n; k++)
				{
					a = k * Fa->Lmfit1 + indxr;
					b = k * Fa->Lmfit1 + indxc;

					swap(&((*CUDA_LCC).covar[a]), &((*CUDA_LCC).covar[b]));
					//SWAP((*CUDA_LCC).covar[a], (*CUDA_LCC).covar[b]);
				}
			}
		}
	}
	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	/* TEST - OK */ 

	/*int brtmph, brtmpl;
	brtmph = Fa->Lmfit / BLOCK_DIM;
	if (Fa->Lmfit % BLOCK_DIM) brtmph++;

	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Fa->Lmfit) brtmph = Fa->Lmfit;
	brtmpl++;

	for (int i = 1; i <= n; i++)
	{
		for (j = brtmpl; j <= brtmph; j++)
		{
			int ixx = j * Fa->Lmfit1 + 1;
			for (k = 1; k <= n; k++, ixx++)
			{
				if (blockIdx.x == 5 && threadIdx.x == 2 && i == 2)
					printf("[%d][%d] j[%d], i[%d], k[%d], covar[%d][%d]: % .9f\n", blockIdx.x, threadIdx.x, j, i, k, blockIdx.x, ixx, (*CUDA_LCC).covar[ixx]);
			}
		}
	}*/
	/* END TEST */

	if (threadIdx.x == 0)
	{
		//		if (err_code != 0) return(err_code); bacha na sync threads (cz)/ watch the sync threads
		j = 0;
		for (int l = 1; l <= ma; l++)
		{
			if (Fa->ia[l])
			{
				j++;
				(*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
				if(blockIdx.x == 5)
					printf("[%d][%d] atry[%d]: % .9f\n", blockIdx.x, threadIdx.x, l, (*CUDA_LCC).atry[l]);
			}
		}
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

__kernel void CLCalculateIter1Mrqcof2Start(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
	//__read_only int Numfac)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

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

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	//mrqcof_matrix(CUDA_LCC, Fa, (*CUDA_LCC).atry, lpoints);
}

__kernel void CLCalculateIter1Mrqcof2Curve1(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	//__global int2* texArea,
	//__global int2* texDg,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	//mrqcof_curve1(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).atry, (*CUDA_LCC).da, inrel, lpoints); // (*CUDA_LCC).covar,
}

__kernel void CLCalculateIter1Mrqcof2Curve1Last(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	//__global int2* texArea,
	//__global int2* texDg,
	//__local double* res,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	//mrqcof_curve1_last(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}

__kernel void CLCalculateIter1Mrqcof2End(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

	//(*CUDA_LCC).Chisq = mrqcof_end(CUDA_LCC, Fa, (*CUDA_LCC).covar);
}

__kernel void CLCalculateIter1Mrqmin2End(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x]) return;

	if (!Fa->isNiter[blockIdx.x]) return;

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

	if (Fa->isInvalid[blockIdx.x])
	{
		return;
	}

	if (Fa->isNiter[blockIdx.x])
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

	if (Fa->isInvalid[blockIdx.x]) return;

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
	__global struct freq_result* CUDA_FR,
	__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];

	if (Fa->isInvalid[blockIdx.x])
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
