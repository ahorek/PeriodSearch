/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006
*/

#include <cmath>
#include "globals_CUDA.h"
#include <device_launch_parameters.h>

__device__ void matrix_neo(freq_context* CUDA_LCC, double cg[], int lnp1, int Lpoints)
{
    double f, cf, sf, pom, pom0, alpha;
    double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat;
    int lnp, jp;

    int brtmph, brtmpl, index;
    brtmph = Lpoints / CUDA_BLOCK_DIM;
    if (Lpoints % CUDA_BLOCK_DIM) brtmph++;
    brtmpl = threadIdx.x * brtmph;
    brtmph = brtmpl + brtmph;
    if (brtmph > Lpoints) brtmph = Lpoints;
    brtmpl++;
    double inv_coef = 1.0 / cg[CUDA_ncoef0 + 2];


    lnp = lnp1 + brtmpl - 1;
    for (jp = brtmpl; jp <= brtmph; jp++)
    {
        lnp++;
        ee_1 = CUDA_ee[lnp * 3 + 0];// position vectors
        ee0_1 = CUDA_ee0[lnp * 3 + 0];
        ee_2 = CUDA_ee[lnp * 3 + 1];
        ee0_2 = CUDA_ee0[lnp * 3 + 1];
        ee_3 = CUDA_ee[lnp * 3 + 2];
        ee0_3 = CUDA_ee0[lnp * 3 + 2];
        t = CUDA_tim[lnp];

        alpha = acos(ee_1 * ee0_1 + ee_2 * ee0_2 + ee_3 * ee0_3);

        /* Exp-lin model (const.term=1.) */
		f = exp(-alpha * inv_coef);
        (*CUDA_LCC).jp_Scale[jp] = __fma_rn(cg[CUDA_ncoef0 + 3], alpha, __fma_rn(cg[CUDA_ncoef0 + 1], f, 1));
		(*CUDA_LCC).jp_dphp_1[jp] = f;
		(*CUDA_LCC).jp_dphp_2[jp] = cg[CUDA_ncoef0 + 1] * f * alpha / (cg[CUDA_ncoef0 + 2] * cg[CUDA_ncoef0 + 2]);
		(*CUDA_LCC).jp_dphp_3[jp] = alpha;

        //  matrix start
        double f = __fma_rn(cg[CUDA_ncoef0], t, CUDA_Phi_0);
		f = fmod(f, 2 * PI);
        sincos(f, &sf, &cf);

        /* rotation matrix, Z axis, angle f */
        tmat = __fma_rn(cf, (*CUDA_LCC).Blmat[1][1], sf * (*CUDA_LCC).Blmat[2][1]);
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = __fma_rn(cf, (*CUDA_LCC).Blmat[1][2], sf * (*CUDA_LCC).Blmat[2][2]);
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = __fma_rn(cf, (*CUDA_LCC).Blmat[1][3], sf * (*CUDA_LCC).Blmat[2][3]);
		(*CUDA_LCC).e_1[jp] = __fma_rn(tmat, ee_3, pom);
		(*CUDA_LCC).e0_1[jp] = __fma_rn(tmat, ee0_3, pom0);

        tmat = __fma_rn(-sf, (*CUDA_LCC).Blmat[1][1], cf * (*CUDA_LCC).Blmat[2][1]);
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = __fma_rn(-sf, (*CUDA_LCC).Blmat[1][2], cf * (*CUDA_LCC).Blmat[2][2]);
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = __fma_rn(-sf, (*CUDA_LCC).Blmat[1][3], cf * (*CUDA_LCC).Blmat[2][3]);
		(*CUDA_LCC).e_2[jp] = __fma_rn(tmat, ee_3, pom);
		(*CUDA_LCC).e0_2[jp] = __fma_rn(tmat, ee0_3, pom0);

		tmat = (*CUDA_LCC).Blmat[3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Blmat[3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Blmat[3][3];
		(*CUDA_LCC).e_3[jp] = __fma_rn(tmat, ee_3, pom);
		(*CUDA_LCC).e0_3[jp] = __fma_rn(tmat, ee0_3, pom0);

		tmat = __fma_rn(cf, (*CUDA_LCC).Dblm[1][1][1], sf * (*CUDA_LCC).Dblm[1][2][1]);
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = __fma_rn(cf, (*CUDA_LCC).Dblm[1][1][2], sf * (*CUDA_LCC).Dblm[1][2][2]);
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = __fma_rn(cf, (*CUDA_LCC).Dblm[1][1][3], sf * (*CUDA_LCC).Dblm[1][2][3]);
        index = jp * 16 + 1 * 4 + 1;
        (*CUDA_LCC).de[index] = __fma_rn(tmat, ee_3, pom);
        (*CUDA_LCC).de0[index] = __fma_rn(tmat, ee0_3, pom0);

		tmat = __fma_rn(cf, (*CUDA_LCC).Dblm[2][1][1], sf * (*CUDA_LCC).Dblm[2][2][1]);
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = __fma_rn(cf, (*CUDA_LCC).Dblm[2][1][2], sf * (*CUDA_LCC).Dblm[2][2][2]);
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = __fma_rn(cf, (*CUDA_LCC).Dblm[2][1][3], sf * (*CUDA_LCC).Dblm[2][2][3]);
        index++; // index = jp * 16 + 1 * 4 + 2;
        (*CUDA_LCC).de[index] = __fma_rn(tmat, ee_3, pom);
        (*CUDA_LCC).de0[index] = __fma_rn(tmat, ee0_3, pom0);

		tmat = __fma_rn(-t * sf, (*CUDA_LCC).Blmat[1][1], t * cf * (*CUDA_LCC).Blmat[2][1]);
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = __fma_rn(-t * sf, (*CUDA_LCC).Blmat[1][2], t * cf * (*CUDA_LCC).Blmat[2][2]);
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = __fma_rn(-t * sf, (*CUDA_LCC).Blmat[1][3], t * cf * (*CUDA_LCC).Blmat[2][3]);
        index++; // index = jp * 16 + 1 * 4 + 3;
        (*CUDA_LCC).de[index] = __fma_rn(tmat, ee_3, pom);
        (*CUDA_LCC).de0[index] = __fma_rn(tmat, ee0_3, pom0);

		tmat = __fma_rn(-sf, (*CUDA_LCC).Dblm[1][1][1], cf * (*CUDA_LCC).Dblm[1][2][1]);
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = __fma_rn(-sf, (*CUDA_LCC).Dblm[1][1][2], cf * (*CUDA_LCC).Dblm[1][2][2]);
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = __fma_rn(-sf, (*CUDA_LCC).Dblm[1][1][3], cf * (*CUDA_LCC).Dblm[1][2][3]);
        index = jp * 16 + 2 * 4 + 1;
        (*CUDA_LCC).de[index] = __fma_rn(tmat, ee_3, pom);
        (*CUDA_LCC).de0[index] = __fma_rn(tmat, ee0_3, pom0);

		tmat = __fma_rn(-sf, (*CUDA_LCC).Dblm[2][1][1], cf * (*CUDA_LCC).Dblm[2][2][1]);
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = __fma_rn(-sf, (*CUDA_LCC).Dblm[2][1][2], cf * (*CUDA_LCC).Dblm[2][2][2]);
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = __fma_rn(-sf, (*CUDA_LCC).Dblm[2][1][3], cf * (*CUDA_LCC).Dblm[2][2][3]);
        index++; // index = jp * 16 + 2 * 4 + 2;
        (*CUDA_LCC).de[index] = __fma_rn(tmat, ee_3, pom);
        (*CUDA_LCC).de0[index] = __fma_rn(tmat, ee0_3, pom0);

		tmat = __fma_rn(-t * cf, (*CUDA_LCC).Blmat[1][1], (-t * sf) * (*CUDA_LCC).Blmat[2][1]);
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = __fma_rn(-t * cf, (*CUDA_LCC).Blmat[1][2], (-t * sf) * (*CUDA_LCC).Blmat[2][2]);
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = __fma_rn(-t * cf, (*CUDA_LCC).Blmat[1][3], (-t * sf) * (*CUDA_LCC).Blmat[2][3]);
        index++; // index = jp * 16 + 2 * 4 + 3;
        (*CUDA_LCC).de[index] = __fma_rn(tmat, ee_3, pom);
        (*CUDA_LCC).de0[index] = __fma_rn(tmat, ee0_3, pom0);

		tmat = (*CUDA_LCC).Dblm[1][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Dblm[1][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Dblm[1][3][3];
        index = jp * 16 + 3 * 4 + 1;
        (*CUDA_LCC).de[index] = __fma_rn(tmat, ee_3, pom);
        (*CUDA_LCC).de0[index] = __fma_rn(tmat, ee0_3, pom0);

		tmat = (*CUDA_LCC).Dblm[2][3][1];
		pom = tmat * ee_1;
		pom0 = tmat * ee0_1;
		tmat = (*CUDA_LCC).Dblm[2][3][2];
		pom += tmat * ee_2;
		pom0 += tmat * ee0_2;
		tmat = (*CUDA_LCC).Dblm[2][3][3];
        index++; // index = jp * 16 + 3 * 4 + 2;
        (*CUDA_LCC).de[index] = __fma_rn(tmat, ee_3, pom);
        (*CUDA_LCC).de0[index] = __fma_rn(tmat, ee0_3, pom0);

        index++; // index = jp * 16 + 3 * 4 + 3;
        (*CUDA_LCC).de[index] = 0;
        (*CUDA_LCC).de0[index] = 0;
    }
    __syncthreads();
}
__device__ double bright(freq_context* CUDA_LCC, double cg[], int jp, int Lpoints1, int Inrel)
{
    int ncoef0, ncoef, i, j, incl_count = 0;
    double cl, cls, dnom, s, Scale;
    double e_1, e_2, e_3, e0_1, e0_2, e0_3, de[4][4], de0[4][4];

    ncoef0 = CUDA_ncoef0;//ncoef - 2 - CUDA_Nphpar;
    ncoef = CUDA_ma;
    cl = exp(cg[ncoef - 1]); /* Lambert */
    cls = cg[ncoef];       /* Lommel-Seeliger */

    /* matrix from neo */
    /* derivatives */

    e_1 = (*CUDA_LCC).e_1[jp];
    e_2 = (*CUDA_LCC).e_2[jp];
    e_3 = (*CUDA_LCC).e_3[jp];
    e0_1 = (*CUDA_LCC).e0_1[jp];
    e0_2 = (*CUDA_LCC).e0_2[jp];
    e0_3 = (*CUDA_LCC).e0_3[jp];

    // Loop over the indices to map to flattened array
    for (int i = 1; i <= 3; ++i) 
    {
        for (int j = 1; j <= 3; ++j) 
        {
            // Calculate the flattened index for 'de' and 'de0'
            int index = jp * (4 * 4) + i * 4 + j;
            de[i][j] = (*CUDA_LCC).de[index];
            de0[i][j] = (*CUDA_LCC).de0[index];
        }
    }

    // index = x * 16 + y * 4 + z;

    /* Directions (and ders.) in the rotating system */

    //
    /*Integrated brightness (phase coeff. used later) */
    double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
    double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
    //   short int *incl=&(*CUDA_LCC).incl[threadIdx.x*MAX_N_FAC];
    //   double *dbr=&(*CUDA_LCC).dbr[threadIdx.x*MAX_N_FAC];
    short int incl[MAX_N_FAC];
    double dbr[MAX_N_FAC];
    //int2 bfr;

    br = 0;
    tmp1 = 0;
    tmp2 = 0;
    tmp3 = 0;
    tmp4 = 0;
    tmp5 = 0;
    j = blockIdx.x * (CUDA_Numfac1)+1;
    for (i = 1; i <= CUDA_Numfac; i++, j++)
    {
        lmu = __fma_rn(e_1, CUDA_Nor[i][0], __fma_rn(e_2, CUDA_Nor[i][1], e_3 * CUDA_Nor[i][2]));
		lmu0 = __fma_rn(e0_1, CUDA_Nor[i][0], __fma_rn(e0_2, CUDA_Nor[i][1], e0_3 * CUDA_Nor[i][2]));
        if ((lmu > TINY) && (lmu0 > TINY))
        {
            dnom = lmu + lmu0;
            s = lmu * lmu0 * (cl + cls / dnom);
            ar = CUDA_Area[j];
            br += ar * s;
            //
            incl[incl_count] = i;
            dbr[incl_count] = CUDA_Darea[i] * s;
            incl_count++;

            double lmu0_dnom = lmu0 / dnom;
            dsmu = __fma_rn(cls, lmu0_dnom * lmu0_dnom, cl * lmu0);
            double lmu_dnom = lmu / dnom;
            dsmu0 = __fma_rn(cls, lmu_dnom * lmu_dnom, cl * lmu);

            sum1 = __fma_rn(CUDA_Nor[i][0], de[1][1], __fma_rn(CUDA_Nor[i][1], de[2][1], (CUDA_Nor[i][2] * de[3][1])));
			sum10 = __fma_rn(CUDA_Nor[i][0], de0[1][1], __fma_rn(CUDA_Nor[i][1], de0[2][1], (CUDA_Nor[i][2] * de0[3][1])));
			tmp1 += ar * __fma_rn(dsmu, sum1, dsmu0 * sum10);
			sum2 = __fma_rn(CUDA_Nor[i][0], de[1][2], __fma_rn(CUDA_Nor[i][1], de[2][2], (CUDA_Nor[i][2] * de[3][2])));
			sum20 = __fma_rn(CUDA_Nor[i][0], de0[1][2], __fma_rn(CUDA_Nor[i][1], de0[2][2], (CUDA_Nor[i][2] * de0[3][2])));
			tmp2 += ar * __fma_rn(dsmu, sum2, dsmu0 * sum20);
			sum3 = __fma_rn(CUDA_Nor[i][0], de[1][3], __fma_rn(CUDA_Nor[i][1], de[2][3], (CUDA_Nor[i][2] * de[3][3])));
			sum30 = __fma_rn(CUDA_Nor[i][0], de0[1][3], __fma_rn(CUDA_Nor[i][1], de0[2][3], (CUDA_Nor[i][2] * de0[3][3])));
			tmp3 += ar * __fma_rn(dsmu, sum3, dsmu0 * sum30);

			tmp4 = __fma_rn(lmu * lmu0, ar, tmp4);
			double inv_sum = 1.0 / (lmu + lmu0);
			tmp5 = __fma_rn(ar * lmu * lmu0, inv_sum, tmp5);
        }
    }
    Scale = (*CUDA_LCC).jp_Scale[jp];
    i = jp + (ncoef0 - 3 + 1) * Lpoints1;

    /* Ders. of brightness w.r.t. rotation parameters */
    (*CUDA_LCC).dytemp[i] = Scale * tmp1;
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

    ncoef0 -= 3;
    int m, m1, mr, iStart;
    int d, d1, dr;

    iStart = Inrel + 1;
    m = blockIdx.x * CUDA_Dg_block + iStart * (CUDA_Numfac1);
    d = jp + (Lpoints1 << Inrel);

    m1 = m + (CUDA_Numfac1);
    mr = 2 * CUDA_Numfac1;
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
            int is_next_coef_valid = (i + 1) <= ncoef0;

            tmp = l_dbr * CUDA_Dg[m + l_incl];
            if (is_next_coef_valid)
            {
                tmp1 = l_dbr * CUDA_Dg[m1 + l_incl];
            }

            for (j = 1; j < incl_count; j++)
            {
                double l_dbr = dbr[j];
                int l_incl = incl[j];
                tmp += l_dbr * CUDA_Dg[m + l_incl];
                if (is_next_coef_valid)
                {
                    tmp1 += l_dbr * CUDA_Dg[m1 + l_incl];
                }
            }

            (*CUDA_LCC).dytemp[d] = Scale * tmp;
            if (is_next_coef_valid)
            {
                (*CUDA_LCC).dytemp[d1] = Scale * tmp1;
            }
        }
    }
    else
    {
        for (i = 1; i <= ncoef0; i++, d += Lpoints1)
            (*CUDA_LCC).dytemp[d] = 0;
    }

    return(0);
}
