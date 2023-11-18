/* computes integrated brightness of all visible and iluminated areas
   and its derivatives

   8.11.2006
*/

#include <cmath>
#include "globals_CUDA.h"
#include <device_launch_parameters.h>

__device__ void __forceinline__ matrix_neo(freq_context *__restrict__ CUDA_LCC, double const *__restrict__ cg, int lnp1, int Lpoints, int bid)
{
  int lnp, jp;
  int blockidx = bid;

  jp = threadIdx.x + 1;

  double nc02r = ___drcp_rn(cg[CUDA_ncoef0 + 2]);
  double phi0 = CUDA_Phi_0;
  double nc02r2 = nc02r * nc02r;

#pragma unroll 1
  while (jp <= Lpoints)
  {
    double f, cf, sf, pom, pom0, alpha;
    double ee_1, ee_2, ee_3, ee0_1, ee0_2, ee0_3, t, tmat1, tmat2, tmat3;

    lnp = lnp1 + jp;

    ee_1 = CUDA_ee[0][lnp]; // position vectors
    ee0_1 = CUDA_ee0[0][lnp];
    ee_2 = CUDA_ee[1][lnp];
    ee0_2 = CUDA_ee0[1][lnp];
    ee_3 = CUDA_ee[2][lnp];
    ee0_3 = CUDA_ee0[2][lnp];
    t = CUDA_tim[lnp];
    double nc00 = cg[CUDA_ncoef0 + 0];

    alpha = acos(((ee_1 * ee0_1) + ee_2 * ee0_2) + ee_3 * ee0_3);
    f = nc00 * t + phi0;

    /* Exp-lin model (const.term=1.) */
    double ff = exp2(-1.44269504088896 * (alpha * nc02r));

    double nc01 = cg[CUDA_ncoef0 + 1];
    double nc03 = cg[CUDA_ncoef0 + 3];

    /* fmod may give little different results than Mikko's */
    f = f - 2.0 * PI * round(f * (1.0 / (2.0 * PI))); // 3:41.9

    double scale = 1.0 + nc01 * ff + nc03 * alpha;
    double d2 = nc01 * ff * alpha * nc02r2;

    //  matrix start

    __builtin_assume(f > (-2.0 * PI) && f < (2.0 * PI));
    sincos(f, &sf, &cf);

    CUDA_scale[blockidx][jp] = scale;

    jp_dphp[0][blockidx][jp] = ff;
    jp_dphp[1][blockidx][jp] = d2;
    jp_dphp[2][blockidx][jp] = alpha;

    /* rotation matrix, Z axis, angle f */

    double Blmat00 = __ldg(&Blmat[0][0][blockidx]);
    double Blmat10 = __ldg(&Blmat[1][0][blockidx]);
    double Blmat01 = __ldg(&Blmat[0][1][blockidx]);
    double Blmat11 = __ldg(&Blmat[1][1][blockidx]);
    double Blmat02 = __ldg(&Blmat[0][2][blockidx]);

    tmat1 = cf * Blmat00;
    tmat2 = cf * Blmat01;
    tmat3 = cf * Blmat02;
    tmat1 += sf * Blmat10;
    tmat2 += sf * Blmat11;
    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;
    pom += tmat3 * ee_3;
    pom0 += tmat3 * ee0_3;
    double pom1 = pom;
    double pom1_0 = pom0;
    double pom1_t = -t * pom;
    double pom1_t0 = -t * pom0;
    ge[0][0][blockidx][jp] = pom1;
    ge[1][0][blockidx][jp] = pom1_0;
    gde[0][1][2][blockidx][jp] = pom1_t;
    gde[1][1][2][blockidx][jp] = pom1_t0;

    double msf = -sf;

    tmat1 = msf * Blmat00;
    tmat2 = msf * Blmat01;
    tmat3 = msf * Blmat02;
    tmat1 += cf * Blmat10;
    tmat2 += cf * Blmat11;
    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;
    pom += tmat3 * ee_3;
    pom0 += tmat3 * ee0_3;
    double pom2 = pom;
    double pom2_0 = pom0;
    double pom2_t = t * pom;
    double pom2_t0 = t * pom0;
    ge[0][1][blockidx][jp] = pom2;
    ge[1][1][blockidx][jp] = pom2_0;
    gde[0][0][2][blockidx][jp] = pom2_t;
    gde[1][0][2][blockidx][jp] = pom2_t0;

    tmat1 = __ldg(&Blmat[2][0][blockidx]);
    tmat2 = __ldg(&Blmat[2][1][blockidx]);
    tmat3 = __ldg(&Blmat[2][2][blockidx]);

    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;
    pom += tmat3 * ee_3;
    pom0 += tmat3 * ee0_3;

    double Dblm000 = __ldg(&Dblm[0][0][0][blockidx]);
    double Dblm001 = __ldg(&Dblm[0][0][1][blockidx]);
    double Dblm002 = __ldg(&Dblm[0][0][2][blockidx]);

    ge[0][2][blockidx][jp] = pom;
    ge[1][2][blockidx][jp] = pom0;

    tmat1 = cf * Dblm000;
    tmat2 = cf * Dblm001;
    tmat3 = cf * Dblm002;

    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;
    pom += tmat3 * ee_3;
    pom0 += tmat3 * ee0_3;
    gde[0][0][0][blockidx][jp] = pom;
    gde[1][0][0][blockidx][jp] = pom0;

    tmat1 = msf * Dblm000;
    tmat2 = msf * Dblm001;
    tmat3 = msf * Dblm002;
    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;
    pom += tmat3 * ee_3;
    pom0 += tmat3 * ee0_3;

    double Dblm100 = __ldg(&Dblm[1][0][0][blockidx]);
    double Dblm101 = __ldg(&Dblm[1][0][1][blockidx]);
    double Dblm110 = __ldg(&Dblm[1][1][0][blockidx]);
    double Dblm111 = __ldg(&Dblm[1][1][1][blockidx]);

    gde[0][1][0][blockidx][jp] = pom;
    gde[1][1][0][blockidx][jp] = pom0;

    tmat1 = cf * Dblm100;
    tmat2 = cf * Dblm101;
    tmat1 += sf * Dblm110;
    tmat2 += sf * Dblm111;

    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;

    tmat1 = msf * Dblm100 + cf * Dblm110;
    tmat2 = msf * Dblm101 + cf * Dblm111;

    gde[0][0][1][blockidx][jp] = pom;
    gde[1][0][1][blockidx][jp] = pom0;

    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;

    double Dblm020 = __ldg(&Dblm[0][2][0][blockidx]);
    double Dblm021 = __ldg(&Dblm[0][2][1][blockidx]);
    double Dblm022 = __ldg(&Dblm[0][2][2][blockidx]);

    gde[0][1][1][blockidx][jp] = pom;
    gde[1][1][1][blockidx][jp] = pom0;

    tmat1 = Dblm020;
    tmat2 = Dblm021;
    tmat3 = Dblm022;

    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;
    pom += tmat3 * ee_3;
    pom0 += tmat3 * ee0_3;

    double Dblm120 = __ldg(&Dblm[1][2][0][blockidx]);
    double Dblm121 = __ldg(&Dblm[1][2][1][blockidx]);

    gde[0][2][0][blockidx][jp] = pom;
    gde[1][2][0][blockidx][jp] = pom0;

    tmat1 = Dblm120;
    tmat2 = Dblm121;

    pom = tmat1 * ee_1;
    pom0 = tmat1 * ee0_1;
    pom += tmat2 * ee_2;
    pom0 += tmat2 * ee0_2;

    gde[0][2][2][blockidx][jp] = 0;
    gde[1][2][2][blockidx][jp] = 0;

    gde[0][2][1][blockidx][jp] = pom;
    gde[1][2][1][blockidx][jp] = pom0;

    jp += CUDA_BLOCK_DIM;
  }
  __syncwarp();
}

__device__ double __forceinline__ bright(freq_context *__restrict__ CUDA_LCC,
                                         double *__restrict__ cg,
                                         int jp /*threadIdx, ok!*/, int Lpoints1, int Inrel)
{
  int ncoef0, ncoef, incl_count = 0;
  int i, j, blockidx = blockIdx();
  double cl, cls, dnom, s, Scale;
  double e_1, e_2, e_3, e0_1, e0_2, e0_3;
  double de[3][3], de0[3][3];

  ncoef0 = CUDA_ncoef0; // ncoef - 2 - CUDA_Nphpar;
  ncoef = CUDA_ma;
  cl = exp(cg[ncoef - 1]); /* Lambert */
  cls = cg[ncoef];         /* Lommel-Seeliger */

  /* matrix from neo */
  /* derivatives */

  e_1 = __ldg(&ge[0][0][blockidx][jp]);
  e_2 = __ldg(&ge[0][1][blockidx][jp]);
  e_3 = __ldg(&ge[0][2][blockidx][jp]);
  e0_1 = __ldg(&ge[1][0][blockidx][jp]);
  e0_2 = __ldg(&ge[1][1][blockidx][jp]);
  e0_3 = __ldg(&ge[1][2][blockidx][jp]);

  de[0][0] = __ldg(&gde[0][0][0][blockidx][jp]);
  de[0][1] = __ldg(&gde[0][0][1][blockidx][jp]);
  de[0][2] = __ldg(&gde[0][0][2][blockidx][jp]);
  de[1][0] = __ldg(&gde[0][1][0][blockidx][jp]);
  de[1][1] = __ldg(&gde[0][1][1][blockidx][jp]);
  de[1][2] = __ldg(&gde[0][1][2][blockidx][jp]);
  de[2][0] = __ldg(&gde[0][2][0][blockidx][jp]);
  de[2][1] = __ldg(&gde[0][2][1][blockidx][jp]);
  de[2][2] = 0; // CUDA_LCC->de[2][2][jp];

  de0[0][0] = __ldg(&gde[1][0][0][blockidx][jp]);
  de0[0][1] = __ldg(&gde[1][0][1][blockidx][jp]);
  de0[0][2] = __ldg(&gde[1][0][2][blockidx][jp]);
  de0[1][0] = __ldg(&gde[1][1][0][blockidx][jp]);
  de0[1][1] = __ldg(&gde[1][1][1][blockidx][jp]);
  de0[1][2] = __ldg(&gde[1][1][2][blockidx][jp]);
  de0[2][0] = __ldg(&gde[1][2][0][blockidx][jp]);
  de0[2][1] = __ldg(&gde[1][2][1][blockidx][jp]);
  de0[2][2] = 0; // CUDA_LCC->de0[2][2][jp];

  /* Directions (and ders.) in the rotating system */

  //
  /*Integrated brightness (phase coeff. used later) */
  double lmu, lmu0, dsmu, dsmu0, sum1, sum10, sum2, sum20, sum3, sum30;
  double br, ar, tmp1, tmp2, tmp3, tmp4, tmp5;
  //   short int *incl=&CUDA_LCC->incl[threadIdx.x*MAX_N_FAC];
  //   double *dbr=&CUDA_LCC->dbr[threadIdx.x*MAX_N_FAC];

  short int incl[MAX_N_FAC];
  double dbr[MAX_N_FAC];
  // int2 bfr;
  int nf = CUDA_Numfac, nf1 = CUDA_Numfac1;

  int bid = blockidx;
  br = 0;
  tmp1 = 0;
  tmp2 = 0;
  tmp3 = 0;
  tmp4 = 0;
  tmp5 = 0;
  j = bid * nf1 + 1;
  double const *__restrict__ norp0;
  double const *__restrict__ norp1;
  double const *__restrict__ norp2;
  double const *__restrict__ areap;
  double const *__restrict__ dareap;
  norp0 = CUDA_Nor[0];
  norp1 = CUDA_Nor[1];
  norp2 = CUDA_Nor[2];
  // areap = CUDA_Area;
  areap = &(Areag[bid][0]);
  dareap = CUDA_Darea;

#pragma unroll 1
  for (i = 1; i <= nf && i <= MAX_N_FAC; i++, j++)
  {
    double n0 = norp0[i], n1 = norp1[i], n2 = norp2[i];
    lmu = e_1 * n0 + e_2 * n1 + e_3 * n2;
    lmu0 = e0_1 * n0 + e0_2 * n1 + e0_3 * n2;
    // if((lmu > TINY) && (lmu0 > TINY))
    //{
    if ((lmu <= TINY) || (lmu0 <= TINY))
      continue;
    dnom = lmu + lmu0;
    ar = __ldca(&areap[i]);

    double dnom_1 = ___drcp_rn(dnom);

    s = lmu * lmu0 * (cl + cls * dnom_1);
    double lmu0_dnom = lmu0 * dnom_1;

    br += ar * s;
    //
    dbr[incl_count] = __ldca(&dareap[i]) * s;
    incl[incl_count] = i;
    incl_count++;

    double lmu_dnom = lmu * dnom_1;
    dsmu = cls * (lmu0_dnom * lmu0_dnom) + cl * lmu0;
    dsmu0 = cls * (lmu_dnom * lmu_dnom) + cl * lmu;
    //	  double n0 = CUDA_Nor[0][i], n1 = CUDA_Nor[1][i], n2 = CUDA_Nor[2][i];

    sum1 = n0 * de[0][0] + n1 * de[1][0] + n2 * de[2][0];
    sum10 = n0 * de0[0][0] + n1 * de0[1][0] + n2 * de0[2][0];
    sum2 = n0 * de[0][1] + n1 * de[1][1] + n2 * de[2][1];
    sum20 = n0 * de0[0][1] + n1 * de0[1][1] + n2 * de0[2][1];
    sum3 = n0 * de[0][2] + n1 * de[1][2];    // + n2 * de[2][2];
    sum30 = n0 * de0[0][2] + n1 * de0[1][2]; // + n2 * de0[2][2];

    tmp1 += ar * (dsmu * sum1 + dsmu0 * sum10);
    tmp2 += ar * (dsmu * sum2 + dsmu0 * sum20);
    tmp3 += ar * (dsmu * sum3 + dsmu0 * sum30);

    tmp4 += ar * lmu * lmu0;
    tmp5 += ar * lmu * lmu0 * dnom_1; // lmu0 * __drcp_rn(lmu + lmu0);
    //}
  }

  // Scale = CUDA_LCC->jp_Scale[jp];
  Scale = __ldg(&CUDA_scale[bid][jp]);
  i = jp + (ncoef0 - 3 + 1) * Lpoints1;
#ifndef NEWDYTMP
  double *__restrict__ dytempp = CUDA_LCC->dytemp, *__restrict__ ytemp = CUDA_LCC->ytemp;
#else
  double *__restrict__ dytempp = dytemp[jp][0][bid], *__restrict__ ytemp = CUDA_LCC->ytemp;
#endif
  /* Ders. of brightness w.r.t. rotation parameters */
  dytempp[i] = Scale * tmp1;
  i += Lpoints1;
  dytempp[i] = Scale * tmp2;
  i += Lpoints1;
  dytempp[i] = Scale * tmp3;
  i += Lpoints1;

  /* Ders. of br. w.r.t. phase function params. */
  dytempp[i] = br * __ldg(&jp_dphp[0][bid][jp]);
  i += Lpoints1;
  dytempp[i] = br * __ldg(&jp_dphp[1][bid][jp]);
  i += Lpoints1;
  dytempp[i] = br * __ldg(&jp_dphp[2][bid][jp]);

  /* Ders. of br. w.r.t. cl, cls */
  dytempp[jp + (ncoef) * (Lpoints1)-Lpoints1] = Scale * tmp4 * cl;
  dytempp[jp + (ncoef) * (Lpoints1)] = Scale * tmp5;

  /* Scaled brightness */
  ytemp[jp] = br * Scale;

  ncoef0 -= 3;
  int m, m1, mr, iStart;
  int d, d1, dr;

  iStart = Inrel + 1;
  m = bid * CUDA_Dg_block + iStart * nf1;
  d = jp + Inrel * 2 * Lpoints1;

  m1 = m + nf1;
  mr = 2 * nf1;
  d1 = d + Lpoints1;
  dr = 2 * Lpoints1;

  /* Derivatives of brightness w.r.t. g-coeffs */
  if (incl_count)
  {
    double const *__restrict__ pCUDA_Dg = CUDA_Dg + m;
    double const *__restrict__ pCUDA_Dg1 = CUDA_Dg + m1;

#pragma unroll 1
    for (i = iStart; i <= ncoef0; i += 2, /*m += mr, m1 += mr,*/ d += dr, d1 += dr)
    {
      double tmp = 0, tmp1 = 0;

      if ((i + 1) <= ncoef0)
      {
#pragma unroll 2
        for (j = 0; j < incl_count - (UNRL - 1); j += UNRL)
        {
          double l_dbr[UNRL], l_tmp[UNRL], l_tmp1[UNRL];
          int l_incl[UNRL], ii;

          for (ii = 0; ii < UNRL; ii++)
          {
            l_incl[ii] = incl[j + ii];
            l_dbr[ii] = dbr[j + ii];
          }
          for (ii = 0; ii < UNRL; ii++)
          {
            l_tmp[ii] = pCUDA_Dg[l_incl[ii]];
            l_tmp1[ii] = pCUDA_Dg1[l_incl[ii]];
          }
          for (ii = 0; ii < UNRL; ii++)
          {
            double qq = l_dbr[ii];
            tmp += qq * l_tmp[ii];
            tmp1 += qq * l_tmp1[ii];
          }
        }
#pragma unroll 3
        for (; j < incl_count; j++)
        {
          int l_incl = incl[j];
          double l_dbr = dbr[j];
          double v1 = pCUDA_Dg[l_incl];
          double v2 = pCUDA_Dg1[l_incl];

          tmp += l_dbr * v1;
          tmp1 += l_dbr * v2;
        }
        __stwb(&dytempp[d], Scale * tmp);
        __stwb(&dytempp[d1], Scale * tmp1);
      }
      else
      {
#pragma unroll 2
        for (j = 0; j < incl_count - (UNRL - 1); j += UNRL)
        {
          double l_dbr[UNRL], l_tmp[UNRL];
          int l_incl[UNRL], ii;

          for (ii = 0; ii < UNRL; ii++)
          {
            l_incl[ii] = incl[j + ii];
          }

          for (ii = 0; ii < UNRL; ii++)
          {
            l_dbr[ii] = dbr[j + ii];
            l_tmp[ii] = pCUDA_Dg[l_incl[ii]];
          }

          for (ii = 0; ii < UNRL; ii++)
            tmp += l_dbr[ii] * l_tmp[ii];
        }
#pragma unroll 3
        for (; j < incl_count; j++)
        {
          int l_incl = incl[j];
          double l_dbr = dbr[j];

          tmp += l_dbr * pCUDA_Dg[l_incl];
        }
        __stwb(&dytempp[d], Scale * tmp);
      }
      pCUDA_Dg += mr;
      pCUDA_Dg1 += mr;
    }
  }
  else
  {
    double *__restrict__ p = dytempp + d;
#pragma unroll
    for (i = 1; i <= ncoef0 - (UNRL - 1); i += UNRL)
      for (int t = 0; t < UNRL; t++, p += Lpoints1)
        __stwb(p, 0.0);
#pragma unroll
    for (; i <= ncoef0; i++, p += Lpoints1)
      __stwb(p, 0.0);
  }

  return 0;
}