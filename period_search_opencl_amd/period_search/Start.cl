__kernel void CLCalculatePrepare(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__read_only int max_test_periods,
	__read_only int n_start,
	double freq_start,
	double freq_step)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	printf("blockIdx.x = ", blockIdx.x);

	//struct freq_result* CUDA_LCC = &CUDA_CC[idx];
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];
	__global struct freq_result* CUDA_LFR = &CUDA_FR[blockIdx.x];
	int n = n_start + blockIdx.x;

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
	printf("CUDA_CC2[%d].freq = %.6f\n", blockIdx.x, (*CUDA_LCC).freq);

	/* initial poles */
	(*CUDA_FR).per_best = 0;
	(*CUDA_FR).dark_best = 0;
	(*CUDA_FR).la_best = 0;
	(*CUDA_FR).be_best = 0;
	(*CUDA_FR).dev_best = 1e40;
}

__kernel void CLCalculateFinish(
	__global struct freq_context2* CUDA_CC2,
	__global struct freq_result* CUDA_FR)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	//const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	//const auto CUDA_LFR = &CUDA_FR[blockIdx.x];

	if ((CUDA_CC2[blockIdx.x]).isInvalid)
	{
		return;
	}

	if ((CUDA_FR[blockIdx.x]).la_best < 0)
	{
		(CUDA_FR[blockIdx.x]).la_best += 360;
	}

	if (isnan((CUDA_FR[blockIdx.x]).dark_best) == 1)
	{
		(CUDA_FR[blockIdx.x]).dark_best = 1.0;
	}
}

__kernel void CLCalculatePreparePole(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__global int* CUDA_End,
	__global double* CUDA_cg_first,
	__global double* CUDA_beta_pole,
	__global double* CUDA_lambda_pole,
	__global double* CUDA_par,
	double log_cl,
	int m,
	int n_coef,
	int n_ph_par)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	/*const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];*/

	if (CUDA_CC[blockIdx.x].isInvalid)
	{
		atomic_add(CUDA_End, 1);
		CUDA_FR[blockIdx.x].isReported = 0; //signal not to read result

		return;
	}

	double period = 1 / CUDA_CC[blockIdx.x].freq;

	/* starts from the initial ellipsoid */
	for (int i = 1; i <= n_coef; i++)
	{
		CUDA_CC[blockIdx.x].cg[i] = CUDA_cg_first[i];
	}

	CUDA_CC[blockIdx.x].cg[n_coef + 1] = CUDA_beta_pole[m];
	CUDA_CC[blockIdx.x].cg[n_coef + 2] = CUDA_lambda_pole[m];

	/* The formulas use beta measured from the pole */
	CUDA_CC[blockIdx.x].cg[n_coef + 1] = 90 - CUDA_CC[blockIdx.x].cg[n_coef + 1];

	/* conversion of lambda, beta to radians */
	CUDA_CC[blockIdx.x].cg[n_coef + 1] = DEG2RAD * CUDA_CC[blockIdx.x].cg[n_coef + 1];
	CUDA_CC[blockIdx.x].cg[n_coef + 2] = DEG2RAD * CUDA_CC[blockIdx.x].cg[n_coef + 2];

	/* Use omega instead of period */
	CUDA_CC[blockIdx.x].cg[n_coef + 3] = 24 * 2 * M_PI / period;

	for (int i = 1; i <= n_ph_par; i++)
	{
		CUDA_CC[blockIdx.x].cg[n_coef + 3 + i] = CUDA_par[i];
		//              ia[Ncoef+3+i] = ia_par[i]; moved to global
	}

	/* Lommel-Seeliger part */
	CUDA_CC[blockIdx.x].cg[n_coef + 3 + n_ph_par + 2] = 1;
	/* Use logarithmic formulation for Lambert to keep it positive */
	CUDA_CC[blockIdx.x].cg[n_coef + 3 + n_ph_par + 1] = log_cl; // log(CUDA_cl);

	/* Levenberg-Marquardt loop */
	// moved to global iter_max,iter_min,iter_dif_max
	//
	CUDA_CC[blockIdx.x].rchisq = -1;
	CUDA_CC[blockIdx.x].Alamda = -1;
	CUDA_CC[blockIdx.x].Niter = 0;
	CUDA_CC[blockIdx.x].iter_diff = 1e40;
	CUDA_CC[blockIdx.x].dev_old = 1e30;
	CUDA_CC[blockIdx.x].dev_new = 0;
	//	(*CUDA_LCC).Lastcall=0; always ==0
	CUDA_FR[blockIdx.x].isReported = 0;
}

__kernel void CLCalculateIter1Begin(
	__global struct freq_context2* CUDA_CC,
	__global struct freq_result* CUDA_FR,
	__global int* CUDA_End,
	int n_iter_min,
	int n_iter_max,
	double iter_diff_max,
	double aLambda_start)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	if (CUDA_CC[blockIdx.x].isInvalid)
	{
		return;
	}

	CUDA_CC[blockIdx.x].isNiter = ((CUDA_CC[blockIdx.x].Niter < n_iter_max) && (CUDA_CC[blockIdx.x].iter_diff > iter_diff_max)) || (CUDA_CC[blockIdx.x].Niter < n_iter_min);
	if (CUDA_CC[blockIdx.x].isNiter)
	{
		if (CUDA_CC[blockIdx.x].Alamda < 0)
		{
			CUDA_CC[blockIdx.x].isAlamda = 1;
			CUDA_CC[blockIdx.x].Alamda = aLambda_start; /* initial alambda */
		}
		else
		{
			CUDA_CC[blockIdx.x].isAlamda = 0;
		}
	}
	else
	{
		if (!CUDA_FR[blockIdx.x].isReported)
		{
			atomic_add(CUDA_End, 1);
			CUDA_FR[blockIdx.x].isReported = 1;
		}
	}
}

__kernel void CLCalculateIter1Mrqcof1Start(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC;
	CUDA_LCC = &CUDA_CC[blockIdx.x];
	//printf("%d", Fa->Dg_block);

	if ((*CUDA_CC).isInvalid) return;

	if (!(*CUDA_CC).isNiter) return;

	if (!(*CUDA_CC).isAlamda) return;

	mrqcof_start(CUDA_LCC, Fa, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta);
}

__kernel void CLCalculateIter1Mrqcof1Matrix(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_matrix(CUDA_LCC, Fa, (*CUDA_LCC).cg, lpoints);
}

__kernel void CLCalculateIter1Mrqcof1Curve1(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	__global int2* texArea,
	__global int2* texDg,
	const int inrel, const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if ((*CUDA_LCC).isInvalid) return;

	if (!(*CUDA_LCC).isNiter) return;

	if (!(*CUDA_LCC).isAlamda) return;

	mrqcof_curve1(CUDA_LCC, Fa, texArea, texDg, (*CUDA_LCC).cg, (*CUDA_LCC).beta, inrel, lpoints); //(*CUDA_LCC).alpha,
}

//__kernel void CudaCalculateIter1Mrqmin1End(void)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	int block = CUDA_BLOCK_DIM;
//	/*gauss_err=*/mrqmin_1_end(CUDA_LCC, CUDA_ma, CUDA_mfit, CUDA_mfit1, block);
//}
//
//__kernel void CudaCalculateIter1Mrqmin2End(void)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	mrqmin_2_end(CUDA_LCC, CUDA_ia, CUDA_ma);
//	(*CUDA_LCC).Niter++;
//}

//__kernel void CudaCalculateIter1Mrqcof1Curve1Last(const int inrel, const int lpoints)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	if (!(*CUDA_LCC).isAlamda) return;
//
//	mrqcof_curve1_last(CUDA_LCC, (*CUDA_LCC).cg, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
//}
//
//__kernel void CudaCalculateIter1Mrqcof1End(void)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	if (!(*CUDA_LCC).isAlamda) return;
//
//	(*CUDA_LCC).Ochisq = mrqcof_end(CUDA_LCC, (*CUDA_LCC).alpha);
//}
//
//__kernel void CudaCalculateIter1Mrqcof2Start(void)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	mrqcof_start(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da);
//}
//
//__kernel void CudaCalculateIter1Mrqcof2Matrix(const int lpoints)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	mrqcof_matrix(CUDA_LCC, (*CUDA_LCC).atry, lpoints);
//}
//
//__kernel void CudaCalculateIter1Mrqcof2Curve1(const int inrel, const int lpoints)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	mrqcof_curve1(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
//}
//
//__kernel void CudaCalculateIter1Mrqcof2Curve1Last(const int inrel, const int lpoints)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	mrqcof_curve1_last(CUDA_LCC, (*CUDA_LCC).atry, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
//}
//
//__kernel void CudaCalculateIter1Mrqcof2End(void)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if (!(*CUDA_LCC).isNiter) return;
//
//	(*CUDA_LCC).Chisq = mrqcof_end(CUDA_LCC, (*CUDA_LCC).covar);
//}
//
//__kernel void CudaCalculateIter2(void)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid)
//	{
//		return;
//	}
//
//	if ((*CUDA_LCC).isNiter)
//	{
//		if ((*CUDA_LCC).Niter == 1 || (*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
//		{
//			if (threadIdx.x == 0)
//			{
//				(*CUDA_LCC).Ochisq = (*CUDA_LCC).Chisq;
//			}
//			__syncthreads();
//
//			auto brtmph = CUDA_Numfac / CUDA_BLOCK_DIM;
//			if (CUDA_Numfac % CUDA_BLOCK_DIM) brtmph++;
//			int brtmpl = threadIdx.x * brtmph;
//			brtmph = brtmpl + brtmph;
//			if (brtmph > CUDA_Numfac) brtmph = CUDA_Numfac;
//			brtmpl++;
//
//			curv(CUDA_LCC, (*CUDA_LCC).cg, brtmpl, brtmph);
//
//			if (threadIdx.x == 0)
//			{
//				for (auto i = 1; i <= 3; i++)
//				{
//					(*CUDA_LCC).chck[i] = 0;
//					for (auto j = 1; j <= CUDA_Numfac; j++)
//					{
//						(*CUDA_LCC).chck[i] = (*CUDA_LCC).chck[i] + (*CUDA_LCC).Area[j] * CUDA_Nor[j][i - 1];
//					}
//				}
//				(*CUDA_LCC).rchisq = (*CUDA_LCC).Chisq - (pow((*CUDA_LCC).chck[1], 2) + pow((*CUDA_LCC).chck[2], 2) + pow((*CUDA_LCC).chck[3], 2)) * pow(CUDA_conw_r, 2);
//			}
//		}
//		if (threadIdx.x == 0)
//		{
//			(*CUDA_LCC).dev_new = sqrt((*CUDA_LCC).rchisq / (CUDA_ndata - 3));
//			/* only if this step is better than the previous,
//				1e-10 is for numeric errors */
//			if ((*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new > 1e-10)
//			{
//				(*CUDA_LCC).iter_diff = (*CUDA_LCC).dev_old - (*CUDA_LCC).dev_new;
//				(*CUDA_LCC).dev_old = (*CUDA_LCC).dev_new;
//			}
//			//		(*CUDA_LFR).Niter=(*CUDA_LCC).Niter;
//		}
//	}
//}
//
//__kernel void CudaCalculateFinishPole(void)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	double totarea = 0;
//	for (auto i = 1; i <= CUDA_Numfac; i++)
//	{
//		totarea = totarea + (*CUDA_LCC).Area[i];
//	}
//
//	const auto sum = pow((*CUDA_LCC).chck[1], 2) + pow((*CUDA_LCC).chck[2], 2) + pow((*CUDA_LCC).chck[3], 2);
//	const auto dark = sqrt(sum);
//
//	/* period solution */
//	const auto period = 2 * PI / (*CUDA_LCC).cg[CUDA_Ncoef + 3];
//
//	/* pole solution */
//	const auto la_tmp = RAD2DEG * (*CUDA_LCC).cg[CUDA_Ncoef + 2];
//	const auto be_tmp = 90 - RAD2DEG * (*CUDA_LCC).cg[CUDA_Ncoef + 1];
//
//	if ((*CUDA_LCC).dev_new < (*CUDA_LFR).dev_best)
//	{
//		(*CUDA_LFR).dev_best = (*CUDA_LCC).dev_new;
//		(*CUDA_LFR).per_best = period;
//		(*CUDA_LFR).dark_best = dark / totarea * 100;
//		(*CUDA_LFR).la_best = la_tmp;
//		(*CUDA_LFR).be_best = be_tmp;
//	}
//	//debug
//	/*	(*CUDA_LFR).dark=dark;
//	(*CUDA_LFR).totarea=totarea;
//	(*CUDA_LFR).chck[1]=(*CUDA_LCC).chck[1];
//	(*CUDA_LFR).chck[2]=(*CUDA_LCC).chck[2];
//	(*CUDA_LFR).chck[3]=(*CUDA_LCC).chck[3];*/
//}
//
//__kernel void CudaCalculateFinish(void)
//{
//	const auto CUDA_LCC = &CUDA_CC[blockIdx.x];
//	const auto CUDA_LFR = &CUDA_FR[blockIdx.x];
//
//	if ((*CUDA_LCC).isInvalid) return;
//
//	if ((*CUDA_LFR).la_best < 0)
//		(*CUDA_LFR).la_best += 360;
//
//	if (isnan((*CUDA_LFR).dark_best) == 1)
//		(*CUDA_LFR).dark_best = 1.0;
//}