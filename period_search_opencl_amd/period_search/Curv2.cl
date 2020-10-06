
void MrqcofCurve2(__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	//__global int2* texsig,
	//__global int2* texWeight,
	//__global int2* texbrightness,
	double alpha[], double beta[], int inrel, int lpoints)
{
	int l, jp, j, k, m, lnp1, lnp2, Lpoints1 = lpoints + 1;
	double dy, sig2i, wt, ymod, coef1, coef, wght, ltrial_chisq;
	int2 xx;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	//precalc thread boundaries
	int tmph, tmpl;
	tmph = lpoints / BLOCK_DIM;
	if (lpoints % BLOCK_DIM) tmph++;
	tmpl = threadIdx.x * tmph;
	lnp1 = (*CUDA_LCC).np1 + tmpl;
	tmph = tmpl + tmph;
	if (tmph > lpoints) tmph = lpoints;
	tmpl++;

	int matmph, matmpl;
	matmph = Fa->ma / BLOCK_DIM;
	if (Fa->ma % BLOCK_DIM) matmph++;
	matmpl = threadIdx.x * matmph;
	matmph = matmpl + matmph;
	if (matmph > Fa->ma) matmph = Fa->ma;
	matmpl++;

	int latmph, latmpl;
	latmph = Fa->lastone / BLOCK_DIM;
	if (Fa->lastone % BLOCK_DIM) latmph++;
	latmpl = threadIdx.x * latmph;
	latmph = latmpl + latmph;
	if (latmph > Fa->lastone) latmph = Fa->lastone;
	latmpl++;

	//if (blockIdx.x == 2)
	//	printf("% 2d/% 2d  tmpl: %3d, tmph: %3d Lpoints1: %5d\n", blockIdx.x, threadIdx.x, tmpl, tmph, Lpoints1);
	//	printf("[%d][%d]  \tave: % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).ave);

	/*   if ((*CUDA_LCC).Lastcall != 1) always ==0
	   {*/
	if (inrel /*==1*/)
	{
		for (jp = tmpl; jp <= tmph; jp++)
		{
			lnp1++;
			int ixx = jp + 1 * Lpoints1;
			/* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
			(*CUDA_LCC).dytemp[ixx] = 0;

			//xx = texsig[lnp1];
			//xx = tex1Dfetch(texsig, lnp1);
			//coef = HiLoint2double(xx.y, xx.x) * lpoints / (*CUDA_LCC).ave;
			coef = Fa->Sig[lnp1] * lpoints / (*CUDA_LCC).ave;

			double yytmp = (*CUDA_LCC).ytemp[jp];
			coef1 = yytmp / (*CUDA_LCC).ave;
			(*CUDA_LCC).ytemp[jp] = coef * yytmp;

			ixx += Lpoints1;
			if (blockIdx.x == 2)
				printf("%2d/%2d  dytemp[%3d]: % .6f\n", blockIdx.x, threadIdx.x, ixx, (*CUDA_LCC).dytemp[ixx]);
				//printf("%2d/%2d  sig[%3d]: % .6f\tytemp[%3d]: % .6f\tcoef: % .6f\tcoef1: %.6f\n", blockIdx.x, threadIdx.x, lnp1, Fa->Sig[lnp1], jp, (*CUDA_LCC).ytemp[jp], coef, coef1);

			for (l = 2; l <= Fa->ma; l++, ixx += Lpoints1)
			{
				(*CUDA_LCC).dytemp[ixx] = coef * ((*CUDA_LCC).dytemp[ixx] - coef1 * (*CUDA_LCC).dave[l]);
				//if (blockIdx.x == 2 && threadIdx.x == 2)
				//	printf("%2d/%2d  dytemp[%3d]: % .6f\n", blockIdx.x, threadIdx.x, ixx, (*CUDA_LCC).dytemp[ixx]);
			}
		}
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np1 += lpoints;
	}

	lnp2 = (*CUDA_LCC).np2;
	ltrial_chisq = (*CUDA_LCC).trial_chisq;

	if (blockIdx.x == 2 && threadIdx.x == 2)
		printf("%3d/%3d  np1: %d, lnp2: %d, ia[1]: %d\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).np1, lnp2, Fa->ia[1]);

	if (Fa->ia[1]) //not relative
	{
		for (jp = 1; jp <= lpoints; jp++)
		{
			ymod = (*CUDA_LCC).ytemp[jp];

			int ixx = jp + matmpl * Lpoints1;
			for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
			{
				(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];
			}

			//__syncthreads();
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

			lnp2++;
			//xx = texsig[lnp2];
			//xx = tex1Dfetch(texsig, lnp2);
			//sig2i = 1 / (HiLoint2double(xx.y, xx.x) * HiLoint2double(xx.y, xx.x));
			double bfr = Fa->Sig[lnp2];
			sig2i = 1 / (bfr * bfr);

			//if (blockIdx.x == 0)
			//	printf("bright >>> [%3d][%3d][%3d] bfr: % .6f, sig2i: % .6f\n", blockIdx.x, threadIdx.x, jp, bfr, sig2i);

			//xx = texweight[lnp2];
			//xx = tex1Dfetch(texWeight, lnp2);
			//wght = HiLoint2double(xx.y, xx.x);
			wght = Fa->Weight[lnp2];

			//xx = texbrightness[lnp2];
			//xx = tex1Dfetch(texbrightness, lnp2);
			//dy = HiLoint2double(xx.y, xx.x) - ymod;
			dy = Fa->Brightness[lnp2] - ymod;

			j = 0;
			//
			double sig2iwght = sig2i * wght;
			for (l = 1; l <= Fa->lastone; l++)
			{
				j++;
				wt = (*CUDA_LCC).dyda[l] * sig2iwght;
				//				   k = 0;
				//precalc thread boundaries
				tmph = l / BLOCK_DIM;
				if (l % BLOCK_DIM) tmph++;
				tmpl = threadIdx.x * tmph;
				tmph = tmpl + tmph;
				if (tmph > l) tmph = l;
				tmpl++;
				for (m = tmpl; m <= tmph; m++)
				{
					//				  k++;
					alpha[j * (Fa->Lmfit1) + m] = alpha[j * (Fa->Lmfit1) + m] + wt * (*CUDA_LCC).dyda[m];
				} /* m */

				//__syncthreads();
				barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
				if (threadIdx.x == 0)
				{
					beta[j] = beta[j] + dy * wt;
					printf("[%d][%d]  \tbeta[%d]: % .6f\n", blockIdx.x, threadIdx.x, j, beta[j]);
				}

				//__syncthreads();
				barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
			} /* l */

			for (; l <= Fa->lastma; l++)
			{
				if (Fa->ia[l])
				{
					j++;
					wt = (*CUDA_LCC).dyda[l] * sig2iwght;
					//				   k = 0;

					for (m = latmpl; m <= latmph; m++)
					{
						//					  k++;
						alpha[j * (Fa->Lmfit1) + m] = alpha[j * (Fa->Lmfit1) + m] + wt * (*CUDA_LCC).dyda[m];
					} /* m */

					//__syncthreads();
					barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
					if (threadIdx.x == 0)
					{
						k = Fa->lastone;
						m = Fa->lastone + 1;
						for (; m <= l; m++)
						{
							if (Fa->ia[m])
							{
								k++;
								alpha[j * (Fa->Lmfit1) + k] = alpha[j * (Fa->Lmfit1) + k] + wt * (*CUDA_LCC).dyda[m];
							}
						} /* m */

						beta[j] = beta[j] + dy * wt;
						printf("[%d][%d]  \tbeta[%d]: % .6f\n", blockIdx.x, threadIdx.x, j, beta[j]);
					}

					//__syncthreads();
					barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
				}
			} /* l */

			ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
		} /* jp */
	}
	else //relative ia[1]==0
	{
		for (jp = 1; jp <= lpoints; jp++)
		{
			ymod = (*CUDA_LCC).ytemp[jp];

			int ixx = jp + matmpl * Lpoints1;
			for (l = matmpl; l <= matmph; l++, ixx += Lpoints1)
			{
				(*CUDA_LCC).dyda[l] = (*CUDA_LCC).dytemp[ixx];

				if (blockIdx.x == 0 && threadIdx.x == 2)
					printf("[%2d][%2d][%3d][%3d]  \tdyda[%d]: % .6f\n", blockIdx.x, threadIdx.x, jp, j, l, (*CUDA_LCC).dyda[l]);
			}

			//__syncthreads();
			barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

			lnp2++;

			//xx = texsig[lnp2];
			//xx = tex1Dfetch(texsig, lnp2);
			//sig2i = 1 / (HiLoint2double(xx.y, xx.x) * HiLoint2double(xx.y, xx.x));
			double bfr = Fa->Sig[lnp2];
			sig2i = 1 / (bfr * bfr);

			//if (blockIdx.x == 1)
			//	printf("bright >>> [%3d][%3d][%3d][%3d] bfr: % .6f, sig2i: % .6f\n", blockIdx.x, threadIdx.x, jp, lnp2, Fa->Sig[lnp2], sig2i);

			//xx = (*CUDA_LCC).Weight[lnp2];
			//xx = tex1Dfetch(texWeight, lnp2);
			//wght = HiLoint2double(xx.y, xx.x);
			wght = Fa->Weight[lnp2];

			//if (blockIdx.x == 1)
			//	printf("bright >>> [%3d][%3d][%3d][%3d] Weight: % .6f\n", blockIdx.x, threadIdx.x, jp, lnp2, Fa->Weight[lnp2]);

			//xx = texbrightness[lnp2];
			//xx = tex1Dfetch(texbrightness, lnp2);
			//dy = HiLoint2double(xx.y, xx.x) - ymod;
			bfr = Fa->Brightness[lnp2];
			dy = bfr - ymod;

			//if (blockIdx.x == 1 && threadIdx.x == 9)
			//	printf("bright >>> [%3d][%3d][%3d][%3d] bfr: % .6f, ymod: % .6f\n", blockIdx.x, threadIdx.x, jp, lnp2, bfr, ymod);

			j = 0;
			//
			double sig2iwght = sig2i * wght;
			//l==1
			//
			for (l = 2; l <= Fa->lastone; l++)
			{
				j++;
				wt = (*CUDA_LCC).dyda[l] * sig2iwght;

				//if (blockIdx.x == 2 && threadIdx.x == 2 && l == 2 && jp == 2)
				//	printf("[%2d][%2d][%3d][%3d]  \tdyda[%d]: % .6f\n", blockIdx.x, threadIdx.x, jp, j, l, (*CUDA_LCC).dyda[l]);

				//				   k = 0;
				//precalc thread boundaries
				tmph = l / BLOCK_DIM;
				if (l % BLOCK_DIM) tmph++;
				tmpl = threadIdx.x * tmph;
				tmph = tmpl + tmph;
				if (tmph > l) tmph = l;
				tmpl++;
				//m==1
				if (tmpl == 1) tmpl++;
				//
				for (m = tmpl; m <= tmph; m++)
				{
					//					  k++;
					alpha[j * (Fa->Lmfit1) + m - 1] = alpha[j * (Fa->Lmfit1) + m - 1] + wt * (*CUDA_LCC).dyda[m];
				} /* m */

				//__syncthreads();
				barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
				if (threadIdx.x == 0)
				{
					beta[j] = beta[j] + dy * wt;

					//if (blockIdx.x == 2 && jp == 1)
					//	printf("[%2d][%2d][%3d][%3d]  \tbeta[%d]: % .6f\n", blockIdx.x, threadIdx.x, jp, j, l, beta[j]);
				}

				__syncthreads();
				barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
			} /* l */

			for (; l <= Fa->lastma; l++)
			{
				if (Fa->ia[l])
				{
					j++;
					wt = (*CUDA_LCC).dyda[l] * sig2iwght;
					//				   k = 0;

					tmpl = latmpl;
					//m==1
					if (tmpl == 1) tmpl++;
					//
					for (m = tmpl; m <= latmph; m++)
					{
						//k++;
						alpha[j * (Fa->Lmfit1) + m - 1] = alpha[j * (Fa->Lmfit1) + m - 1] + wt * (*CUDA_LCC).dyda[m];
					} /* m */

					//__syncthreads();
					barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
					if (threadIdx.x == 0)
					{
						k = Fa->lastone - 1;
						m = Fa->lastone + 1;
						for (; m <= l; m++)
						{
							if (Fa->ia[m])
							{
								k++;
								alpha[j * (Fa->Lmfit1) + k] = alpha[j * (Fa->Lmfit1) + k] + wt * (*CUDA_LCC).dyda[m];
							}
						} /* m */

						beta[j] = beta[j] + dy * wt;
						//printf("[%d][%d]  \tbeta[%d]: % .6f\n", blockIdx.x, threadIdx.x, j, beta[j]);
					}

					//__syncthreads();
					barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
				}
			} /* l */

			ltrial_chisq = ltrial_chisq + dy * dy * sig2iwght;
		} /* jp */
	}
	/*     } always ==0  Lastcall != 1 */

	   /*  if (((*CUDA_LCC).Lastcall == 1) && (CUDA_Inrel[i] == 1)) always ==0
			(*CUDA_LCC).Sclnw[i] = (*CUDA_LCC).Scale * CUDA_Lpoints[i] * CUDA_sig[np]/ave;*/

	if (threadIdx.x == 0)
	{
		(*CUDA_LCC).np2 = lnp2;
		(*CUDA_LCC).trial_chisq = ltrial_chisq;
	}
}


__kernel void CLCalculateIter1Mrqcof1Curve2(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	//__global int2* texsig,
	//__global int2* texWeight,
	//__global int2* texbrightness,
	int inrel, int lpoints)
{
	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid) return;

	if (!Fa->isNiter) return;

	if (!Fa->isAlamda) return;

	//if (blockIdx.x == 1)
	//	printf("(*CUDA_LCC).beta[threadIdx.x]: % .6f\n", (*CUDA_LCC).beta[threadIdx.x]);
	//MrqcofCurve2(CUDA_LCC, Fa, texsig, texWeight, texbrightness, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
	MrqcofCurve2(CUDA_LCC, Fa, (*CUDA_LCC).alpha, (*CUDA_LCC).beta, inrel, lpoints);
}

__kernel void CLCalculateIter1Mrqcof2Curve2(
	__global struct freq_context2* CUDA_CC,
	__global varholder* Fa,
	//__global int2* texsig,
	//__global int2* texWeight,
	//__global int2* texbrightness,
	const int inrel,
	const int lpoints)
{
	int3 blockIdx;
	blockIdx.x = get_group_id(0);
	__global struct freq_context2* CUDA_LCC = &CUDA_CC[blockIdx.x];

	if (Fa->isInvalid) return;

	if (!Fa->isNiter) return;

	//MrqcofCurve2(CUDA_LCC, Fa, texsig, texWeight, texbrightness, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
	MrqcofCurve2(CUDA_LCC, Fa, (*CUDA_LCC).covar, (*CUDA_LCC).da, inrel, lpoints);
}
