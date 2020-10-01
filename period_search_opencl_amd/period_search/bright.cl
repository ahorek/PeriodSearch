/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006
*/

//#include <CL/cl.hpp>
////#include <math.h>
////#include "globals_CUDA.h"
//#include "globals.h"
//#include "declarations_OpenCl.h"

void matrix_neo(__global struct freq_context2* CUDA_LCC, __global varholder* Fa, double cg[], const int lnp1, int Lpoints)
{
	double f, cf, sf, pom, pom0, alpha;
	double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
	int lnp;
	int3 threadIdx, blockIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);

	int brtmph, brtmpl;
	brtmph = Lpoints / BLOCK_DIM;
	if (Lpoints % BLOCK_DIM) {
		brtmph++;
	}

	brtmpl = threadIdx.x * brtmph;
	brtmph = brtmpl + brtmph;
	if (brtmph > Lpoints) {
		brtmph = Lpoints;
	}
	brtmpl++;

	/*if (blockIdx.x == 0)
	{
		printf("bright >>> [%d][%d] \tbrtmpl[%d]: %d,\tbrtmph: %d\tlnp1: %d\n", blockIdx.x, threadIdx.x, brtmpl, brtmph, lnp1);
	}*/

	// TODO: Check this out. May be it needs to run against __local vars and only for get_group_id(0) == 0 ?
	lnp = lnp1 + brtmpl - 1;
	for (int jp = brtmpl; jp <= brtmph; jp++)
	{
		lnp++;
		ee_1 = Fa->ee[lnp][0];// position vectors
		ee0_1 = Fa->ee0[lnp][0];
		ee_2 = Fa->ee[lnp][1];
		ee0_2 = Fa->ee0[lnp][1];
		ee_3 = Fa->ee[lnp][2];
		ee0_3 = Fa->ee0[lnp][2];
		t = Fa->tim[lnp];
				
		//if (get_group_id(0) == 0) {
		//	printf("ee[%d](%.6f %.6f %.6f)\n", lnp, Fa->ee[lnp][0], Fa->ee[lnp][1], Fa->ee[lnp][2]);
		//	//printf("ee0[%d](%.6f %.6f %.6f)  ", lnp * 3, Fa->ee0[lnp * 3][0], Fa->ee0[lnp * 3][1], Fa->ee0[lnp * 3][2]);
		//	//printf("tim[%d](%.6f)\n", lnp, Fa->tim[lnp]);
		//}

		alpha = acos(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3);
		
		//if (blockIdx.x == 0)
		//{
			//double ff = (alpha * -1) / cg[Fa->Ncoef0 + 2];
			//printf("[%d]: %.6f\n", jp, alpha);
			
		//}
		
		/* Exp-lin model (const.term=1.) */ 
		f = exp(( alpha * -1.0) / cg[Fa->Ncoef0 + 2]);	//f is temp here
		
		(*CUDA_LCC).jp_Scale[jp] = 1 + cg[Fa->Ncoef0 + 1] * f + cg[Fa->Ncoef0 + 3] * alpha;
		(*CUDA_LCC).jp_dphp_1[jp] = f;
		(*CUDA_LCC).jp_dphp_2[jp] = cg[Fa->Ncoef0 + 1] * f * alpha / (cg[Fa->Ncoef0 + 2] * cg[Fa->Ncoef0 + 2]);
		(*CUDA_LCC).jp_dphp_3[jp] = alpha;

		//if(blockIdx.x == 1)
		//	printf("[%d][%d][%d]: %.6f %.6f\n", blockIdx.x, threadIdx.x, jp, cg[Fa->Ncoef0 + 1], (*CUDA_LCC).jp_Scale[jp]);

		//  matrix start
		f = cg[Fa->Ncoef0] * t + Fa->Phi_0;
		f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
		cf = cos(f);
		sf = sin(f);

		/*if (blockIdx.x == 0 && jp == brtmpl)
		{
			printf("[%d][%d]: \tf: % .6f, cosF: % .6f, sinF: % .6f\n", blockIdx.x, threadIdx.x, f, cf, sf);
		}*/

		///* rotation matrix, Z axis, angle f */

		tmat = cf * (*CUDA_LCC).Blmat[1][1] + sf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Blmat[1][2] + sf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Blmat[1][3] + sf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_1[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_1[jp] = pom0 + tmat * ee0_3;

		//if (blockIdx.x == 2)
		//	printf("[%d][%d]: \t% .6f, % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).e_1[jp], (*CUDA_LCC).e0_1[jp]);

		tmat = (-sf) * (*CUDA_LCC).Blmat[1][1] + cf * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][2] + cf * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-sf) * (*CUDA_LCC).Blmat[1][3] + cf * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_2[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_2[jp] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 1 * (*CUDA_LCC).Blmat[3][1];
		tmat = (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 1 * (*CUDA_LCC).Blmat[3][2];
		tmat = (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 1 * (*CUDA_LCC).Blmat[3][3];
		tmat = (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_3[jp] = pom + tmat * ee_3;
		(*CUDA_LCC).e0_3[jp] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[1][1][1] + sf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][2] + sf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[1][1][3] + sf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][1][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][1] = pom0 + tmat * ee0_3;

		tmat = cf * (*CUDA_LCC).Dblm[2][1][1] + sf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][2] + sf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = cf * (*CUDA_LCC).Dblm[2][1][3] + sf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][1][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][2] = pom0 + tmat * ee0_3;

		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][1] + (t * cf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][2] + (t * cf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * sf) * (*CUDA_LCC).Blmat[1][3] + (t * cf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][1][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][1][3] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[1][1][1] + cf * (*CUDA_LCC).Dblm[1][2][1] + 0 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][2] + cf * (*CUDA_LCC).Dblm[1][2][2] + 0 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[1][1][3] + cf * (*CUDA_LCC).Dblm[1][2][3] + 0 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][2][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][1] = pom0 + tmat * ee0_3;

		tmat = -sf * (*CUDA_LCC).Dblm[2][1][1] + cf * (*CUDA_LCC).Dblm[2][2][1] + 0 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][2] + cf * (*CUDA_LCC).Dblm[2][2][2] + 0 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = -sf * (*CUDA_LCC).Dblm[2][1][3] + cf * (*CUDA_LCC).Dblm[2][2][3] + 0 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][2][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][2] = pom0 + tmat * ee0_3;

		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][1] + (-t * sf) * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][2] + (-t * sf) * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (-t * cf) * (*CUDA_LCC).Blmat[1][3] + (-t * sf) * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][2][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][2][3] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[1][1][1] + 0 * (*CUDA_LCC).Dblm[1][2][1] + 1 * (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][2] + 0 * (*CUDA_LCC).Dblm[1][2][2] + 1 * (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[1][1][3] + 0 * (*CUDA_LCC).Dblm[1][2][3] + 1 * (*CUDA_LCC).Dblm[1][3][3];
		(*CUDA_LCC).de[jp][3][1] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][1] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Dblm[2][1][1] + 0 * (*CUDA_LCC).Dblm[2][2][1] + 1 * (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][2] + 0 * (*CUDA_LCC).Dblm[2][2][2] + 1 * (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Dblm[2][1][3] + 0 * (*CUDA_LCC).Dblm[2][2][3] + 1 * (*CUDA_LCC).Dblm[2][3][3];
		(*CUDA_LCC).de[jp][3][2] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][2] = pom0 + tmat * ee0_3;

		tmat = 0 * (*CUDA_LCC).Blmat[1][1] + 0 * (*CUDA_LCC).Blmat[2][1] + 0 * (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = 0 * (*CUDA_LCC).Blmat[1][2] + 0 * (*CUDA_LCC).Blmat[2][2] + 0 * (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = 0 * (*CUDA_LCC).Blmat[1][3] + 0 * (*CUDA_LCC).Blmat[2][3] + 0 * (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).de[jp][3][3] = pom + tmat * ee_3;
		(*CUDA_LCC).de0[jp][3][3] = pom0 + tmat * ee0_3;

		//if (blockIdx.x == 1 && jp == brtmpl)
		//{
		//	printf("matrix_neo >>> [%d][%d]: \t% .6f, % .6f, % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).Blmat[3][1], (*CUDA_LCC).Blmat[3][1], (*CUDA_LCC).Blmat[3][1]);
		//	//printf("[%d][%d]: \t% .6f, % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).de[jp][3][3], (*CUDA_LCC).de0[jp][3][3]);
		//}
	}

	//__syncthreads();
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}


void bright_test(__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	double cg[],
	int jp,
	int Lpoints1,
	int Inrel) {

	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);
	
	//printf("jp: %d\n", jp);

	int ncoef0, ncoef, i, j, incl_count = 0;
	double cl, cls, dnom, s, Scale;


	ncoef0 = Fa->Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = Fa->ma;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */


	// <<<<<<<<<<<<<<<<<
	double cgt;
	cgt = cg[threadIdx.x];
	if (threadIdx.x == 0)
		printf("bright_test >>> [%d][%d] \tncoef: %d, ma: %d\tcgt: % .6f\n", blockIdx.x, threadIdx.x, ncoef0, ncoef, cgt);

}

//__device__ double bright(freq_context2* CUDA_LCC, double cg[], int jp, int Lpoints1, int Inrel)

void bright(__global struct freq_context2* CUDA_LCC,
	__global varholder* Fa,
	double cg[],
	int jp,
	//const int brtmph, 
	//const int brtmpl,
	int Lpoints1, 
	int Inrel)
{
	int ncoef0, ncoef, i, j, incl_count = 0;// , jp;
	double cl, cls, dnom, s, Scale;
	double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];

	int3 blockIdx, threadIdx;
	threadIdx.x = get_local_id(0);
	blockIdx.x = get_group_id(0);
	//jp = jpp;

	//if (blockIdx.x == 1)
	//	printf("[%d]  %d\n", threadIdx.x, jp);

	ncoef0 = Fa->Ncoef0;//ncoef - 2 - CUDA_Nphpar;
	ncoef = Fa->ma;
	cl = exp(cg[ncoef - 1]); /* Lambert */
	cls = cg[ncoef];       /* Lommel-Seeliger */

	//if (threadIdx.x == 0)
	//	printf("bright >>> [%d][%d] \tncoef: %d, ma: %d\ttest: %d\n", blockIdx.x, threadIdx.x, ncoef0, ncoef, ncoef + threadIdx.x + blockIdx.x);

   /* matrix from neo */
		 /* derivatives */

	e_1 = (*CUDA_LCC).e_1[jp];
	e_2 = (*CUDA_LCC).e_2[jp];
	e_3 = (*CUDA_LCC).e_3[jp];
	e0_1 = (*CUDA_LCC).e0_1[jp];
	e0_2 = (*CUDA_LCC).e0_2[jp];
	e0_3 = (*CUDA_LCC).e0_3[jp];
	de[1][1] = (*CUDA_LCC).de[jp][1][1];
	de[1][2] = (*CUDA_LCC).de[jp][1][2];
	de[1][3] = (*CUDA_LCC).de[jp][1][3];
	de[2][1] = (*CUDA_LCC).de[jp][2][1];
	de[2][2] = (*CUDA_LCC).de[jp][2][2];
	de[2][3] = (*CUDA_LCC).de[jp][2][3];
	de[3][1] = (*CUDA_LCC).de[jp][3][1];
	de[3][2] = (*CUDA_LCC).de[jp][3][2];
	de[3][3] = (*CUDA_LCC).de[jp][3][3];
	de0[1][1] = (*CUDA_LCC).de0[jp][1][1];
	de0[1][2] = (*CUDA_LCC).de0[jp][1][2];
	de0[1][3] = (*CUDA_LCC).de0[jp][1][3];
	de0[2][1] = (*CUDA_LCC).de0[jp][2][1];
	de0[2][2] = (*CUDA_LCC).de0[jp][2][2];
	de0[2][3] = (*CUDA_LCC).de0[jp][2][3];
	de0[3][1] = (*CUDA_LCC).de0[jp][3][1];
	de0[3][2] = (*CUDA_LCC).de0[jp][3][2];
	de0[3][3] = (*CUDA_LCC).de0[jp][3][3];

	/* Directions (and ders.) in the rotating system */

	
	/*Integrated brightness (phase coeff. used later) */
	double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
	double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
	//   short int *incl=&(*CUDA_LCC).incl[threadIdx.x*MAX_N_FAC];
	//   double *dbr=&(*CUDA_LCC).dbr[threadIdx.x*MAX_N_FAC];
	short int incl[MAX_N_FAC];
	double dbr[MAX_N_FAC];
	int2 bfr;
	//__global float4* in4 = (__global float4*)in;
	//__global int2* texArea = &(__global double)(*CUDA_LCC).Area[0];
	//__global int2 texArea = &(*CUDA_LCC).Area;
	

	br = 0;
	tmp1 = 0;
	tmp2 = 0;
	tmp3 = 0;
	tmp4 = 0;
	tmp5 = 0;

	j = blockIdx.x * (Fa->Numfac1) + 1;
	//j = blockIdx.x;
	//if (blockIdx.x == 1 && threadIdx.x == 0)
	//	printf("Fa->Numfac: %d, Fa->Numfac1: %d\n", Fa->Numfac, Fa->Numfac1);
	
	/*if (blockIdx.x == 1) {
		printf("[%d][%d]  \tNumfac: %d\t jp: %d\n", blockIdx.x, threadIdx.x, Fa->Numfac, jp);
	}*/
	for (i = 1; i <= Fa->Numfac; i++, j++)
	{
		//j++;
		lmu = e_1 * Fa->Nor[i][0] + e_2 * Fa->Nor[i][1] + e_3 * Fa->Nor[i][2];
		//if (blockIdx.x == 9) {
		//	//printf("[%d][%d]  \ti: %d\n", blockIdx.x, threadIdx.x, i);
		////	printf("[%d][%d]  \te_1[%d]: % .6f\tNor[%d][0]: % .6f\n", blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).e_1[jp], i, Fa->Nor[i][0]);
		//	printf("[%d][%d]  \tlmu: %d\n", blockIdx.x, threadIdx.x, lmu);
		//}
		lmu0 = e0_1 * Fa->Nor[i][0] + e0_2 * Fa->Nor[i][1] + e0_3 * Fa->Nor[i][2];
		if ((lmu > TINY) && (lmu0 > TINY))
		{
			dnom = lmu + lmu0;
			s = lmu * lmu0 * (cl + cls / dnom);
			//bfr = texArea[j];
			
			//bfr.x = double2loint((*CUDA_LCC).Area[j]);
			//bfr.y = double2hiint((*CUDA_LCC).Area[j]);
			//bfr = tex1Dfetch(texArea, j);
			//ar = HiLoint2double(bfr.y, bfr.x);
			ar = (*CUDA_LCC).Area[i];

			br += ar * s;
			//if (blockIdx.x == 2 && threadIdx.x == 0)
			//	printf("bright >>> [%d][%d] \tArea[%d]: % .16f\tbr: % .6f\n", blockIdx.x, threadIdx.x, j, ar, br);
				//printf("bright >>> [%d][%d] \tArea[%d]: % .16f\n", blockIdx.x, threadIdx.x, j, ar);
				//printf("bright >>> [%d][%d] \tArea[%d]: % .16f\tbfr.y: %d\t bfr.x: %d\n", blockIdx.x, threadIdx.x, j + 289, ar, bfr.y, bfr.x);

			//
			incl[incl_count] = i;
			dbr[incl_count] = Fa->Darea[i] * s;
			incl_count++;
			//
			dsmu = cls * pow(lmu0 / dnom, 2) + cl * lmu0;
			dsmu0 = cls * pow(lmu / dnom, 2) + cl * lmu;

			sum1 = Fa->Nor[i][0] * (*CUDA_LCC).de[jp][1][1] + Fa->Nor[i][1] * (*CUDA_LCC).de[jp][2][1] + Fa->Nor[i][2] * (*CUDA_LCC).de[jp][3][1];
			sum10 = Fa->Nor[i][0] * (*CUDA_LCC).de0[jp][1][1] + Fa->Nor[i][1] * (*CUDA_LCC).de0[jp][2][1] + Fa->Nor[i][2] * (*CUDA_LCC).de0[jp][3][1];
			tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);

			sum2 = Fa->Nor[i][0] * (*CUDA_LCC).de[jp][1][2] + Fa->Nor[i][1] * (*CUDA_LCC).de[jp][2][2] + Fa->Nor[i][2] * (*CUDA_LCC).de[jp][3][2];
			sum20 = Fa->Nor[i][0] * (*CUDA_LCC).de0[jp][1][2] + Fa->Nor[i][1] * (*CUDA_LCC).de0[jp][2][2] + Fa->Nor[i][2] * (*CUDA_LCC).de0[jp][3][2];
			tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);

			sum3 = Fa->Nor[i][0] * (*CUDA_LCC).de[jp][1][3] + Fa->Nor[i][1] * (*CUDA_LCC).de[jp][2][3] + Fa->Nor[i][2] * (*CUDA_LCC).de[jp][3][3];
			sum30 = Fa->Nor[i][0] * (*CUDA_LCC).de0[jp][1][3] + Fa->Nor[i][1] * (*CUDA_LCC).de0[jp][2][3] + Fa->Nor[i][2] * (*CUDA_LCC).de0[jp][3][3];
			tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

			tmp4 += lmu * lmu0 * ar;
			tmp5 += ar * lmu * lmu0 / (lmu + lmu0);
		}
	}

	Scale = (*CUDA_LCC).jp_Scale[jp];
	i = jp + (ncoef0 - 3 + 1) * Lpoints1;
	/* Ders. of brightness w.r.t. rotation parameters */
	(*CUDA_LCC).dytemp[i] = Scale * tmp1;

	//if(blockIdx.x == 1)
	//		printf("bright >>> [%d][%d] \tbr: % .6f\tjp_Scale[%d]: % .6f\n", blockIdx.x, threadIdx.x, br, jp, Scale);
	
	//printf("bright >>> [%d][%d] \tjp_Scale[%d]: % .6f\tdytemp[%d]% .6f\n", blockIdx.x, threadIdx.x, jp, (*CUDA_LCC).jp_Scale[jp], i, (*CUDA_LCC).dytemp[i]);
	//printf("bright >>> [%d][%d] \tdytemp[%d]: % .6f\n", blockIdx.x, threadIdx.x, i, (*CUDA_LCC).dytemp[i]);

	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = Scale * tmp2;
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = Scale * tmp3;
	i += Lpoints1;
	/* Ders. of br. w.r.t. phase function params. */
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_1[jp];
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_2[jp];
	i += Lpoints1;
	(*CUDA_LCC).dytemp[i] = br * (*CUDA_LCC).jp_dphp_3[jp];

	/* Ders. of br. w.r.t. cl, cls */
	(*CUDA_LCC).dytemp[jp + (ncoef - 1) * (Lpoints1)] = Scale * tmp4 * cl;
	(*CUDA_LCC).dytemp[jp + (ncoef) * (Lpoints1)] = Scale * tmp5;
	/* Scaled brightness */
	(*CUDA_LCC).ytemp[jp] = br * Scale;

	//if (blockIdx.x == 2)
	//	printf("bright >>> [%3d][%3d] dytemp[%3d]: % .6f\n", blockIdx.x, threadIdx.x, i, (*CUDA_LCC).dytemp[i]);
		//printf("bright >>> [%3d][%3d] dytemp[%3d]: % .6f\n", blockIdx.x, threadIdx.x, jp + (ncoef) * (Lpoints1), (*CUDA_LCC).dytemp[jp + (ncoef) * (Lpoints1)]);
		//printf("bright >>> [%d][%d] \tbr: % .6f\tjp_Scale[%d]: % .6f\t ytemp[%d]: % .6f\n", blockIdx.x, threadIdx.x, br, jp, Scale, jp, (*CUDA_LCC).ytemp[jp]);


	ncoef0 -= 3;
	int m, m1, mr, iStart;
	int d, d1, dr;
	if (Inrel)
	{
		iStart = 2;
		//m = blockIdx.x * Fa->Dg_block + 2 * (Fa->Numfac1);
		m = 2 * Fa->Numfac1;
		d = jp + 2 * (Lpoints1);
	}
	else
	{
		iStart = 1;
		//m = blockIdx.x * Fa->Dg_block + (Fa->Numfac1);
		m = Fa->Numfac1;
		d = jp + (Lpoints1);
	}

	m1 = m + (Fa->Numfac1);
	mr = 2 * Fa->Numfac1;
	//mr = Fa->Numfac1;
	d1 = d + (Lpoints1);
	dr = 2 * Lpoints1;

	/* Derivatives of brightness w.r.t. g-coeffs */
	if (incl_count)
	{
		for (i = iStart; i <= ncoef0; i += 2, m += mr, m1 += mr, d += dr, d1 += dr)
		{
			double tmp = 0, tmp1 = 0;

			double l_dbr = dbr[0];
			int l_incl = incl[0];

			int2 xx;
			//xx = texDg[m + l_incl];
			//xx = tex1Dfetch(texDg, m + l_incl);
			//tmp = l_dbr * HiLoint2double(xx.y, xx.x);
			//tmp = l_dbr * (*CUDA_LCC).Dg[(m + l_incl)];
			tmp = l_dbr * (*CUDA_LCC).Dg[m + l_incl];

			//if (blockIdx.x == 2 && i == 10)
			//	printf("bright >>> [%3d][%3d][%3d][%d] bfr: % .6f, l_dbr: % .6f, l_incl: %5d\ttmp: % .6f\n", blockIdx.x, threadIdx.x, i, jp, (*CUDA_LCC).Dg[m + l_incl], l_dbr, l_incl, tmp);

			if ((i + 1) <= ncoef0)
			{
				//xx = texDg[m1 + l_incl];
				//xx = tex1Dfetch(texDg, m1 + l_incl);
				//tmp1 = l_dbr * HiLoint2double(xx.y, xx.x);
				tmp1 = l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];

				//if (blockIdx.x == 2 && threadIdx.x == 0)
				//	printf("bright >>> [%3d][%3d] bfr: % .6f, l_dbr: % .6f, l_incl: %5d\ttmp1: % .6f\n", blockIdx.x, threadIdx.x, (*CUDA_LCC).Dg[m1 + l_incl], l_dbr, l_incl, tmp1);

			}

			for (j = 1; j < incl_count; j++)
			{
				double l_dbr = dbr[j];
				int l_incl = incl[j];

				int2 xx1;
				//xx1 = texDg[m + l_incl];
				//xx = tex1Dfetch(texDg, m + l_incl);
				//tmp += l_dbr * HiLoint2double(xx1.y, xx1.x);
				tmp += l_dbr * (*CUDA_LCC).Dg[m + l_incl];

				if ((i + 1) <= ncoef0)
				{
					//xx1 = texDg[m1 + l_incl];
					//xx = tex1Dfetch(texDg, m1 + l_incl);
					//tmp1 += l_dbr * HiLoint2double(xx1.y, xx1.x);
					tmp1 += l_dbr * (*CUDA_LCC).Dg[m1 + l_incl];
				}
			}

			(*CUDA_LCC).dytemp[d] = Scale * tmp;
			if ((i + 1) <= ncoef0)
			{
				(*CUDA_LCC).dytemp[d1] = Scale * tmp1;

				//if (blockIdx.x == 2)
				//	printf("bright >>> [%3d][%3d] Scale: % .6f\ttmp1: % .6f\tdytemp[%3d]: % .6f\n", blockIdx.x, threadIdx.x, Scale, tmp1, d1, (*CUDA_LCC).dytemp[d1]);
			}
		}
	}
	else
	{
		for (i = 1; i <= ncoef0; i++, d += Lpoints1)
			(*CUDA_LCC).dytemp[d] = 0;
	}

	return;
}
