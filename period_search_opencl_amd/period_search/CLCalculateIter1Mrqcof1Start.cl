//void curv(__global struct freq_context2* CUDA_LCC, struct funcarrays FA, double cg[], int brtmpl, int brtmph)
//{
//	int i, m, n, l, k;
//	double fsum, g;
//
//	for (i = brtmpl; i <= brtmph; i++)
//	{
//		g = 0;
//		n = 0;
//		for (m = 0; m <= FA.Mmax; m++)
//		{
//			for (l = m; l <= FA.Lmax; l++)
//			{
//				n++;
//				fsum = cg[n] * FA.Fc[i][m];
//				if (m != 0)
//				{
//					n++;
//					fsum = fsum + cg[n] * FA.Fs[i][m];
//				}
//				g = g + FA.Pleg[i][l][m] * fsum;
//			}
//		}
//
//		g = exp(g);
//		(*CUDA_LCC).Area[i] = FA.Darea[i] * g;
//		for (k = 1; k <= n; k++)
//		{
//			(*CUDA_LCC).Dg[i + k * FA.Numfac1] = g * FA.Dsph[i][k];
//		}
//	}
//
//	//__syncthreads();
//	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
//}

//void blmatrix(__global struct freq_context2* CUDA_LCC, double bet, double lam)
//{
//	double cb, sb, cl, sl;
//
//	cb = cos(bet);
//	sb = sin(bet);
//	cl = cos(lam);
//	sl = sin(lam);
//	(*CUDA_LCC).Blmat[1][1] = cb * cl;
//	(*CUDA_LCC).Blmat[1][2] = cb * sl;
//	(*CUDA_LCC).Blmat[1][3] = -sb;
//	(*CUDA_LCC).Blmat[2][1] = -sl;
//	(*CUDA_LCC).Blmat[2][2] = cl;
//	(*CUDA_LCC).Blmat[2][3] = 0;
//	(*CUDA_LCC).Blmat[3][1] = sb * cl;
//	(*CUDA_LCC).Blmat[3][2] = sb * sl;
//	(*CUDA_LCC).Blmat[3][3] = cb;
//	/* Ders. of Blmat w.r.t. bet */
//	(*CUDA_LCC).Dblm[1][1][1] = -sb * cl;
//	(*CUDA_LCC).Dblm[1][1][2] = -sb * sl;
//	(*CUDA_LCC).Dblm[1][1][3] = -cb;
//	(*CUDA_LCC).Dblm[1][2][1] = 0;
//	(*CUDA_LCC).Dblm[1][2][2] = 0;
//	(*CUDA_LCC).Dblm[1][2][3] = 0;
//	(*CUDA_LCC).Dblm[1][3][1] = cb * cl;
//	(*CUDA_LCC).Dblm[1][3][2] = cb * sl;
//	(*CUDA_LCC).Dblm[1][3][3] = -sb;
//	/* Ders. w.r.t. lam */
//	(*CUDA_LCC).Dblm[2][1][1] = -cb * sl;
//	(*CUDA_LCC).Dblm[2][1][2] = cb * cl;
//	(*CUDA_LCC).Dblm[2][1][3] = 0;
//	(*CUDA_LCC).Dblm[2][2][1] = -cl;
//	(*CUDA_LCC).Dblm[2][2][2] = -sl;
//	(*CUDA_LCC).Dblm[2][2][3] = 0;
//	(*CUDA_LCC).Dblm[2][3][1] = -sb * sl;
//	(*CUDA_LCC).Dblm[2][3][2] = sb * cl;
//	(*CUDA_LCC).Dblm[2][3][3] = 0;
//}

//void mrqcof_start(__global struct freq_context2* CUDA_LCC, __global double* alpha, __global double* beta, struct funcarrays FA, double a[])
//{
//	int j, k;
//	int brtmph, brtmpl;
//	int3 blockIdx;
//	blockIdx.x = get_global_id(0);
//	int3 threadIdx;
//	threadIdx.x = get_local_id(0);
//
//	brtmph = FA.Numfac / BLOCK_DIM;
//	if (FA.Numfac % BLOCK_DIM)
//	{
//		brtmph++;
//	}
//
//	brtmpl = threadIdx.x * brtmph;
//	brtmph = brtmpl + brtmph;
//	if (brtmph > FA.Numfac)
//	{
//		brtmph = FA.Numfac;
//	}
//
//	brtmpl++;
//
//	/* N.B. curv and blmatrix called outside bright
//	   because output same for all points */
//	//curv(CUDA_LCC, FA, a, brtmpl, brtmph);
//
//	if (threadIdx.x == 0)
//	{
//		//   #ifdef YORP
//		//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
//		  // #else
//		//blmatrix(CUDA_LCC, a[FA.ma - 4 - FA.Nphpar], a[FA.ma - 3 - FA.Nphpar]);
//		//   #endif
//		(*CUDA_LCC).trial_chisq = 0;
//		(*CUDA_LCC).np = 0;
//		(*CUDA_LCC).np1 = 0;
//		(*CUDA_LCC).np2 = 0;
//		(*CUDA_LCC).ave = 0;
//	}
//
//	brtmph = FA.Lmfit / BLOCK_DIM;
//	if (FA.Lmfit % BLOCK_DIM)
//	{
//		brtmph++;
//	}
//
//	brtmpl = threadIdx.x * brtmph;
//	brtmph = brtmpl + brtmph;
//	if (brtmph > FA.Lmfit)
//	{
//		brtmph = FA.Lmfit;
//	}
//
//	brtmpl++;
//	for (j = brtmpl; j <= brtmph; j++)
//	{
//		for (k = 1; k <= j; k++)
//		{
//			alpha[j * (FA.Lmfit1) + k] = 0;
//		}
//
//		beta[j] = 0;
//	}
//
//	//__syncthreads(); //for sure
//	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
//}

__kernel void CLCalculateIter1Mrqcof1Start(
	__global struct freq_context2* CUDA_CC,
	struct funcarrays FA)
{
	int3 blockIdx;
	blockIdx.x = get_global_id(0);
	/*__global struct freq_context2* CUDA_LCC;
	CUDA_LCC = &CUDA_CC[blockIdx.x];*/


	if (CUDA_CC[blockIdx.x].isInvalid) return;

	if (!CUDA_CC[blockIdx.x].isNiter) return;

	if (!CUDA_CC[blockIdx.x].isAlamda) return;

	//mrqcof_start(CUDA_LCC, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, FA, (*CUDA_LCC).cg);
}