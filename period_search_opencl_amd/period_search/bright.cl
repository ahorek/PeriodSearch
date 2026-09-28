//computes integrated brightness of all visible and iluminated areas
//  and its derivatives

//  8.11.2006


void matrix_neo(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* cg,
	int lnp1,
	int Lpoints,
	int num,
	__global real* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global real* jp_ScaleG = scr + (*CUDA_CC).offJpScale;
	__global real* jp_dphp_1G = scr + (*CUDA_CC).offJpDphp1;
	__global real* jp_dphp_2G = scr + (*CUDA_CC).offJpDphp2;
	__global real* jp_dphp_3G = scr + (*CUDA_CC).offJpDphp3;
	__global real* e_1G = scr + (*CUDA_CC).offE1;
	__global real* e_2G = scr + (*CUDA_CC).offE2;
	__global real* e_3G = scr + (*CUDA_CC).offE3;
	__global real* e0_1G = scr + (*CUDA_CC).offE01;
	__global real* e0_2G = scr + (*CUDA_CC).offE02;
	__global real* e0_3G = scr + (*CUDA_CC).offE03;
	__global real* deG = scr + (*CUDA_CC).offDe;
	__global real* de0G = scr + (*CUDA_CC).offDe0;
	__private real f, cf, sf, pom, pom0, alpha;
	__private real ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
	__private int lnp;

	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) brtmph++;
	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) brtmph = Lpoints;
	brtmpl++;

	//if (blockIdx.x == 0 && threadIdx.x == 0)
	//{
	//	printf("Blmat[1][1]: %10.7f, Blmat[2][1]: %10.7f, Blmat[3][1]: %10.7f\n", (*CUDA_LCC).Blmat[1][1], (*CUDA_LCC).Blmat[2][1], (*CUDA_LCC).Blmat[3][1]);
	//	printf("Blmat[1][2]: %10.7f, Blmat[2][2]: %10.7f, Blmat[3][2]: %10.7f\n", (*CUDA_LCC).Blmat[1][2], (*CUDA_LCC).Blmat[2][2], (*CUDA_LCC).Blmat[3][2]);
	//	printf("Blmat[1][3]: %10.7f, Blmat[2][3]: %10.7f, Blmat[3][3]: %10.7f\n", (*CUDA_LCC).Blmat[1][3], (*CUDA_LCC).Blmat[2][3], (*CUDA_LCC).Blmat[3][3]);
	//}

	lnp = lnp1 + brtmpl - 1;
	//printf("lnp: %3d = lnp1: %3d + brtmpl: %3d - 1 | lnp++: %3d\n", lnp, lnp1, brtmpl, lnp + 1);

	int q = (*CUDA_CC).Ncoef0 + 2;
	//if (blockIdx.x == 0)
	//	printf("[neo] [%3d] cg[%3d]: %10.7f\n", blockIdx.x,  q, (*CUDA_LCC).cg[q]);

	for (int jp = brtmpl; jp <= brtmph; jp++)
	{
		lnp++;

		ee_1 = (*CUDA_CC).ee[lnp][0];		// position vectors
		ee0_1 = (*CUDA_CC).ee0[lnp][0];
		ee_2 = (*CUDA_CC).ee[lnp][1];
		ee0_2 = (*CUDA_CC).ee0[lnp][1];
		ee_3 = (*CUDA_CC).ee[lnp][2];
		ee0_3 = (*CUDA_CC).ee0[lnp][2];
		t = (*CUDA_CC).tim[lnp];

		//if (blockIdx.x == 0)
		//	printf("jp[%3d] lnp[%3d], %10.7f, %10.7f, %10.7f, %10.7f, %10.7f, %10.7f\n",
		//		jp, lnp, ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3);

		//printf("tim[%3d]: %10.7f\n", lnp, t);
		//printf("lnp: %3d, ee[%d]: %.7f, ee0[%d]: %.7f\n", lnp, lnp * 3 + 0, (*CUDA_CC).ee[lnp][0], lnp, (*CUDA_CC).ee0[lnp][0]);

		alpha = R_ACOS(R_CLAMP(R_ADDM(R_MADD(ee_1, ee0_1, R_MUL(ee_2, ee0_2)), ee_3, ee0_3), R_C(-1.0), R_C(1.0)));


		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] alpha[%3d]: %.7f, cg[%3d]: %10.7f\n", jp, alpha, q, (*CUDA_LCC).cg[q]);

		/* Exp-lin model (const.term=1.) */
		real f = R_EXP(R_NEG(R_DIV(alpha, cg[(*CUDA_CC).Ncoef0 + 2])));	//f is temp here

		//if (blockIdx.x == 0 && threadIdx.x == 0)
		//	printf("[neo] [%2d][%3d] jp[%3d] f: %10.7f, cg[%3d] %10.7f, alpha %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, f, (*CUDA_CC).Ncoef0 + 2, cg[(*CUDA_CC).Ncoef0 + 2], alpha);

		jp_ScaleG[jp] = R_ADDM(R_ADDM(R_C(1.0), cg[(*CUDA_CC).Ncoef0 + 1], f), cg[(*CUDA_CC).Ncoef0 + 3], alpha);
		jp_dphp_1G[jp] = f;
		jp_dphp_2G[jp] = R_DIV(R_MUL(R_MUL(cg[(*CUDA_CC).Ncoef0 + 1], f), alpha), R_MUL(cg[(*CUDA_CC).Ncoef0 + 2], cg[(*CUDA_CC).Ncoef0 + 2]));
		jp_dphp_3G[jp] = alpha;

		//if (blockIdx.x == 0)
		//	printf("[neo] [%d][%3d] jp_Scale[%3d]: %10.7f, jp_dphp_1[]: %10.7F, jp_dphp_2[]: %10.7f, jp_dphp_3[]: %10.7f\n",
		//		blockIdx.x, threadIdx.x, jp, jp_ScaleG[jp], jp_dphp_1G[jp], jp_dphp_2G[jp], jp_dphp_3G[jp]);

		//  matrix start
		f = R_MADD(cg[(*CUDA_CC).Ncoef0], t, (*CUDA_CC).Phi_0);
		f = R_FMOD_2PI(f); /* may give little different results than Mikko's */
		sf = R_SINCOS(f, &cf);

		//if (threadIdx.x == 0)
		//	printf("jp[%3d] [%3d] cf: %10.7f, sf: %10.7f\n", jp, blockIdx.x, cf, sf);

		//if (num == 1 && blockIdx.x == 0 && jp == brtmpl)
		//{
		//	printf("[%2d][%3d][%3d] f: % .6f, cosF: % .6f, sinF: % .6f\n", blockIdx.x, threadIdx.x, jp, f, cf, sf);
		//}

		//	/* rotation matrix, Z axis, angle f */

		tmat = R_MADD(cf, (*CUDA_LCC).Blmat[1][1], R_MUL(sf, (*CUDA_LCC).Blmat[2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(cf, (*CUDA_LCC).Blmat[1][2], R_MUL(sf, (*CUDA_LCC).Blmat[2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(cf, (*CUDA_LCC).Blmat[1][3], R_MUL(sf, (*CUDA_LCC).Blmat[2][3]));
		e_1G[jp] = R_ADDM(pom, tmat, ee_3);
		e0_1G[jp] = R_ADDM(pom0, tmat, ee0_3);

		//if (blockIdx.x == 0)
		//	printf("[%3d] jp[%3d] %10.7f, %10.7f\n", threadIdx.x, jp, e_1G[jp], e0_1G[jp]);

		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Blmat[1][1], R_MUL(cf, (*CUDA_LCC).Blmat[2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Blmat[1][2], R_MUL(cf, (*CUDA_LCC).Blmat[2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Blmat[1][3], R_MUL(cf, (*CUDA_LCC).Blmat[2][3]));
		e_2G[jp] = R_ADDM(pom, tmat, ee_3);
		e0_2G[jp] = R_ADDM(pom0, tmat, ee0_3);

		tmat = (*CUDA_LCC).Blmat[3][1];
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = (*CUDA_LCC).Blmat[3][2];
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = (*CUDA_LCC).Blmat[3][3];
		e_3G[jp] = R_ADDM(pom, tmat, ee_3);
		e0_3G[jp] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[1][1][1], R_MUL(sf, (*CUDA_LCC).Dblm[1][2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[1][1][2], R_MUL(sf, (*CUDA_LCC).Dblm[1][2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[1][1][3], R_MUL(sf, (*CUDA_LCC).Dblm[1][2][3]));
		deG[(jp) * 16 + (1) * 4 + (1)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (1) * 4 + (1)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[2][1][1], R_MUL(sf, (*CUDA_LCC).Dblm[2][2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[2][1][2], R_MUL(sf, (*CUDA_LCC).Dblm[2][2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(cf, (*CUDA_LCC).Dblm[2][1][3], R_MUL(sf, (*CUDA_LCC).Dblm[2][2][3]));
		deG[(jp) * 16 + (1) * 4 + (2)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (1) * 4 + (2)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[1][1], R_MUL(R_MUL(t, cf), (*CUDA_LCC).Blmat[2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[1][2], R_MUL(R_MUL(t, cf), (*CUDA_LCC).Blmat[2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[1][3], R_MUL(R_MUL(t, cf), (*CUDA_LCC).Blmat[2][3]));
		deG[(jp) * 16 + (1) * 4 + (3)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (1) * 4 + (3)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[1][1][1], R_MUL(cf, (*CUDA_LCC).Dblm[1][2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[1][1][2], R_MUL(cf, (*CUDA_LCC).Dblm[1][2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[1][1][3], R_MUL(cf, (*CUDA_LCC).Dblm[1][2][3]));
		deG[(jp) * 16 + (2) * 4 + (1)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (2) * 4 + (1)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[2][1][1], R_MUL(cf, (*CUDA_LCC).Dblm[2][2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[2][1][2], R_MUL(cf, (*CUDA_LCC).Dblm[2][2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_NEG(sf), (*CUDA_LCC).Dblm[2][1][3], R_MUL(cf, (*CUDA_LCC).Dblm[2][2][3]));
		deG[(jp) * 16 + (2) * 4 + (2)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (2) * 4 + (2)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = R_MADD(R_MUL(R_NEG(t), cf), (*CUDA_LCC).Blmat[1][1], R_MUL(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[2][1]));
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = R_MADD(R_MUL(R_NEG(t), cf), (*CUDA_LCC).Blmat[1][2], R_MUL(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[2][2]));
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = R_MADD(R_MUL(R_NEG(t), cf), (*CUDA_LCC).Blmat[1][3], R_MUL(R_MUL(R_NEG(t), sf), (*CUDA_LCC).Blmat[2][3]));
		deG[(jp) * 16 + (2) * 4 + (3)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (2) * 4 + (3)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = (*CUDA_LCC).Dblm[1][3][1];
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = (*CUDA_LCC).Dblm[1][3][2];
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = (*CUDA_LCC).Dblm[1][3][3];
		deG[(jp) * 16 + (3) * 4 + (1)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (3) * 4 + (1)] = R_ADDM(pom0, tmat, ee0_3);

		tmat = (*CUDA_LCC).Dblm[2][3][1];
		pom = R_MUL(tmat, ee_1);
		pom0 = R_MUL(tmat, ee0_1);
		tmat = (*CUDA_LCC).Dblm[2][3][2];
		R_ADDTOM(pom, tmat, ee_2);
		R_ADDTOM(pom0, tmat, ee0_2);
		tmat = (*CUDA_LCC).Dblm[2][3][3];
		deG[(jp) * 16 + (3) * 4 + (2)] = R_ADDM(pom, tmat, ee_3);
		de0G[(jp) * 16 + (3) * 4 + (2)] = R_ADDM(pom0, tmat, ee0_3);


		deG[(jp) * 16 + (3) * 4 + (3)] = R_C(0.0);
		de0G[(jp) * 16 + (3) * 4 + (3)] = R_C(0.0);
	}
}

void bright(
	__global struct mfreq_context* CUDA_LCC,
	__global struct freq_context* CUDA_CC,
	__global real* cg,
	int jp,
	int Lpoints1,
	int Inrel,
	__global real* scr)
{
	/* runtime-sized work arrays, one slice per work-group */
	__global real* dytempG = scr + (*CUDA_CC).offDytemp;
	__global real* ytempG = scr + (*CUDA_CC).offYtemp;
	__global real* jp_ScaleG = scr + (*CUDA_CC).offJpScale;
	__global real* jp_dphp_1G = scr + (*CUDA_CC).offJpDphp1;
	__global real* jp_dphp_2G = scr + (*CUDA_CC).offJpDphp2;
	__global real* jp_dphp_3G = scr + (*CUDA_CC).offJpDphp3;
	__global real* e_1G = scr + (*CUDA_CC).offE1;
	__global real* e_2G = scr + (*CUDA_CC).offE2;
	__global real* e_3G = scr + (*CUDA_CC).offE3;
	__global real* e0_1G = scr + (*CUDA_CC).offE01;
	__global real* e0_2G = scr + (*CUDA_CC).offE02;
	__global real* e0_3G = scr + (*CUDA_CC).offE03;
	__global real* deG = scr + (*CUDA_CC).offDe;
	__global real* de0G = scr + (*CUDA_CC).offDe0;
	real cl, cls, dnom, s, Scale;
	real e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];
	int ncoef0, ncoef, i, j, incl_count = 0;

	int3 blockIdx, threadIdx;
	blockIdx.x = get_group_id(0);
	threadIdx.x = get_local_id(0);

	ncoef0 = (*CUDA_CC).Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = (*CUDA_CC).ma;
	cl = R_EXP(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

	/* matrix from neo */
	/* derivatives */
	e_1 = e_1G[jp];
	e_2 = e_2G[jp];
	e_3 = e_3G[jp];
	e0_1 = e0_1G[jp];
	e0_2 = e0_2G[jp];
	e0_3 = e0_3G[jp];
	de[1][1] = deG[(jp) * 16 + (1) * 4 + (1)];
	de[1][2] = deG[(jp) * 16 + (1) * 4 + (2)];
	de[1][3] = deG[(jp) * 16 + (1) * 4 + (3)];
	de[2][1] = deG[(jp) * 16 + (2) * 4 + (1)];
	de[2][2] = deG[(jp) * 16 + (2) * 4 + (2)];
	de[2][3] = deG[(jp) * 16 + (2) * 4 + (3)];
	de[3][1] = deG[(jp) * 16 + (3) * 4 + (1)];
	de[3][2] = deG[(jp) * 16 + (3) * 4 + (2)];
	de[3][3] = deG[(jp) * 16 + (3) * 4 + (3)];
	de0[1][1] = de0G[(jp) * 16 + (1) * 4 + (1)];
	de0[1][2] = de0G[(jp) * 16 + (1) * 4 + (2)];
	de0[1][3] = de0G[(jp) * 16 + (1) * 4 + (3)];
	de0[2][1] = de0G[(jp) * 16 + (2) * 4 + (1)];
	de0[2][2] = de0G[(jp) * 16 + (2) * 4 + (2)];
	de0[2][3] = de0G[(jp) * 16 + (2) * 4 + (3)];
	de0[3][1] = de0G[(jp) * 16 + (3) * 4 + (1)];
	de0[3][2] = de0G[(jp) * 16 + (3) * 4 + (2)];
	de0[3][3] = de0G[(jp) * 16 + (3) * 4 + (3)];

	/*Integrated brightness (phase coeff. used later) */
	real lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	real br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	short int incl[MAX_N_FAC];
	real dbr[MAX_N_FAC];

	br = R_C(0.0);
	tmp1 = R_C(0.0);
	tmp2 = R_C(0.0);
	tmp3 = R_C(0.0);
	tmp4 = R_C(0.0);
	tmp5 = R_C(0.0);

#ifdef PS_SWFP64
	/* Software FP64 build: the software-FP64 visibility test costs ~6 emulated
	   operations per facet, so a float pre-scan first drops the facets it
	   can prove hidden or unlit: lmu < -margin or lmu0 < -margin, margin =
	   1e-5 of the sum of the term magnitudes plus an absolute 1e-30, far
	   above the float error of the truncated operands (~5e-7 of that sum),
	   so the exact lmu, lmu0 are then < 0 < TINY. The loop below applies
	   the exact test to the remaining candidates and compacts incl / dbr in
	   place, so the heavy terms still run over exactly the visible facets
	   in ascending order: the results are bit-identical. */
	{
		const float fe_1 = R_TO_F32(e_1), fe_2 = R_TO_F32(e_2), fe_3 = R_TO_F32(e_3);
		const float fe0_1 = R_TO_F32(e0_1), fe0_2 = R_TO_F32(e0_2), fe0_3 = R_TO_F32(e0_3);
		for (i = 1; i <= (*CUDA_CC).Numfac; i++)
		{
			const float n1 = R_TO_F32((*CUDA_CC).Nor[i][0]);
			const float n2 = R_TO_F32((*CUDA_CC).Nor[i][1]);
			const float n3 = R_TO_F32((*CUDA_CC).Nor[i][2]);
			const float a1 = fe_1 * n1, a2 = fe_2 * n2, a3 = fe_3 * n3;
			const float b1 = fe0_1 * n1, b2 = fe0_2 * n2, b3 = fe0_3 * n3;
			const float fmu = a1 + a2 + a3;
			const float fmu0 = b1 + b2 + b3;
			const float m = 1e-5f * (fabs(a1) + fabs(a2) + fabs(a3)) + 1e-30f;
			const float m0 = 1e-5f * (fabs(b1) + fabs(b2) + fabs(b3)) + 1e-30f;
			if (!(fmu < -m) && !(fmu0 < -m0))
			{
				incl[incl_count] = i;
				incl_count++;
			}
		}
	}
	const int ncand = incl_count;
	incl_count = 0;
#define BRIGHT_NLIST ncand
#define BRIGHT_SLOT incl_count
#else
	/* Two passes: the cheap visibility test first builds this work-item's
	   list of visible facets, then the division-heavy terms run over that
	   list. In a single pass a wavefront executed the heavy block for every
	   facet that ANY of its lanes could see, i.e. for nearly all facets;
	   now it runs max(incl_count) times per wavefront. lmu/lmu0 are
	   recomputed with the same expressions and the sums still run over the
	   visible facets in ascending order. */
	for (i = 1; i <= (*CUDA_CC).Numfac; i++)
	{
		lmu = R_ADDM(R_MADD(e_1, (*CUDA_CC).Nor[i][0], R_MUL(e_2, (*CUDA_CC).Nor[i][1])), e_3, (*CUDA_CC).Nor[i][2]);
		lmu0 = R_ADDM(R_MADD(e0_1, (*CUDA_CC).Nor[i][0], R_MUL(e0_2, (*CUDA_CC).Nor[i][1])), e0_3, (*CUDA_CC).Nor[i][2]);
		if (R_GT(lmu, R_TINY) && R_GT(lmu0, R_TINY))
		{
			incl[incl_count] = i;
			incl_count++;
		}
	}

#define BRIGHT_NLIST incl_count
#define BRIGHT_SLOT c
#endif

	for (int c = 0; c < BRIGHT_NLIST; c++)
	{
		i = incl[c];
		j = i;
		lmu = R_ADDM(R_MADD(e_1, (*CUDA_CC).Nor[i][0], R_MUL(e_2, (*CUDA_CC).Nor[i][1])), e_3, (*CUDA_CC).Nor[i][2]);
		lmu0 = R_ADDM(R_MADD(e0_1, (*CUDA_CC).Nor[i][0], R_MUL(e0_2, (*CUDA_CC).Nor[i][1])), e0_3, (*CUDA_CC).Nor[i][2]);
#ifdef PS_SWFP64
		if (R_GT(lmu, R_TINY) && R_GT(lmu0, R_TINY))
#endif
		{
#ifdef PS_SWFP64
			incl[incl_count] = i;
#endif
			dnom = R_ADD(lmu, lmu0);
			s = R_MUL(R_MUL(lmu, lmu0), R_ADD(cl, R_DIV(cls, dnom)));
			ar = (*CUDA_LCC).Area[j];
			R_ADDTOM(br, ar, s);

			/* Darea[i] * s * Dg[i][k] == Darea[i] * s * g * Dsph[i][k]
			   == (Area[i] * s) * Dsph[i][k]: fold g into the weight and
			   gather from the one read-only, facet-major Dsph shared by
			   all work-groups instead of the per-context Dg matrix */
			dbr[BRIGHT_SLOT] = R_MUL(ar, s);

			real lmu0_dnom = R_DIV(lmu0, dnom);
			dsmu = R_MADD(cls, R_MUL(lmu0_dnom, lmu0_dnom), R_MUL(cl, lmu0));
			real lmu_dnom = R_DIV(lmu, dnom);
			dsmu0 = R_MADD(cls, R_MUL(lmu_dnom, lmu_dnom), R_MUL(cl, lmu));


			sum1 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de[1][1], R_MUL((*CUDA_CC).Nor[i][1], de[2][1])), (*CUDA_CC).Nor[i][2], de[3][1]);
			sum10 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de0[1][1], R_MUL((*CUDA_CC).Nor[i][1], de0[2][1])), (*CUDA_CC).Nor[i][2], de0[3][1]);
			R_ADDTOM(tmp1, ar, R_MADD(dsmu, sum1, R_MUL(dsmu0, sum10)));
			sum2 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de[1][2], R_MUL((*CUDA_CC).Nor[i][1], de[2][2])), (*CUDA_CC).Nor[i][2], de[3][2]);
			sum20 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de0[1][2], R_MUL((*CUDA_CC).Nor[i][1], de0[2][2])), (*CUDA_CC).Nor[i][2], de0[3][2]);
			R_ADDTOM(tmp2, ar, R_MADD(dsmu, sum2, R_MUL(dsmu0, sum20)));
			sum3 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de[1][3], R_MUL((*CUDA_CC).Nor[i][1], de[2][3])), (*CUDA_CC).Nor[i][2], de[3][3]);
			sum30 = R_ADDM(R_MADD((*CUDA_CC).Nor[i][0], de0[1][3], R_MUL((*CUDA_CC).Nor[i][1], de0[2][3])), (*CUDA_CC).Nor[i][2], de0[3][3]);
			R_ADDTOM(tmp3, ar, R_MADD(dsmu, sum3, R_MUL(dsmu0, sum30)));

			R_ADDTOM(tmp4, R_MUL(lmu, lmu0), ar);
			R_ADDTOM(tmp5, ar, R_DIV(R_MUL(lmu, lmu0), R_ADD(lmu, lmu0)));
#ifdef PS_SWFP64
			incl_count++;
#endif
		}
	}

	Scale = jp_ScaleG[jp];
	i = (jp - 1) * DYT_STRIDE + (ncoef0 - 3 + 1);
	/* Ders. of brightness w.r.t. rotation parameters */
	dytempG[i] = R_MUL(Scale, tmp1);

	i++;
	dytempG[i] = R_MUL(Scale, tmp2);
	i++;
	dytempG[i] = R_MUL(Scale, tmp3);

	i++;
	/* Ders. of br. w.r.t. phase function params. */
	dytempG[i] = R_MUL(br, jp_dphp_1G[jp]);
	i++;
	dytempG[i] = R_MUL(br, jp_dphp_2G[jp]);
	i++;
	dytempG[i] = R_MUL(br, jp_dphp_3G[jp]);

	/* Ders. of br. w.r.t. cl, cls */
	dytempG[(jp - 1) * DYT_STRIDE + (ncoef - 1)] = R_MUL(R_MUL(Scale, tmp4), cl);
	dytempG[(jp - 1) * DYT_STRIDE + (ncoef)] = R_MUL(Scale, tmp5);

	/* Scaled brightness */
	ytempG[jp] = R_MUL(br, Scale);

	ncoef0 -= 3;
	int iStart;
	int d;

	iStart = Inrel + 1;
	d = (jp - 1) * DYT_STRIDE + iStart;


	/* Derivatives of brightness w.r.t. g-coeffs: BRIGHT_GB columns per pass
	   over the visible-facet list (was 2); each column is still
	   dbr[0] * Dsph[..] followed by the fma chain over the visible facets
	   in ascending order. Up to BRIGHT_GB - 1 columns past ncoef0 are read
	   (inside the Dsph row) but not stored. */
#define BRIGHT_GB 16
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += BRIGHT_GB)
		{
			real t[BRIGHT_GB];
			{
				real l_dbr = dbr[0];
				__global real* row = (*CUDA_CC).Dsph[incl[0]] + i;
				for (int b = 0; b < BRIGHT_GB; b++)
					t[b] = R_MUL(l_dbr, row[b]);
			}

			for (j = 1; j < incl_count; j++)
			{
				real l_dbr = dbr[j];
				__global real* row = (*CUDA_CC).Dsph[incl[j]] + i;
				for (int b = 0; b < BRIGHT_GB; b++)
					R_ADDTOM(t[b], l_dbr, row[b]);
			}

			for (int b = 0; b < BRIGHT_GB; b++)
			{
				if (i + b <= ncoef0)
					dytempG[(jp - 1) * DYT_STRIDE + i + b] = R_MUL(Scale, t[b]);
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d++)
			dytempG[d] = R_C(0.0);
	}

	//return(0);
}
