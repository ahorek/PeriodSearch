/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006

   Numerical recipes: 'mrqcof' is used by 'mrqmin' to evaluate coefficients
*/

#include <stdio.h>
#include <stdlib.h>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include <immintrin.h>
#include "CalcStrategyAvx.hpp"
#include "arrayHelpers.hpp"

// #define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
template<typename T1, typename T2>
constexpr auto MIN(T1 X, T2 Y) { return ((X) < (Y) ? (X) : (Y)); }

/* comment the following line if no YORP */
/*#define YORP*/

#if defined(__GNUC__)
__attribute__((target("avx")))
#endif

void CalcStrategyAvx::mrqcof(double** x1, double** x2, double x3[], double y[],
	double sig[], double a[], int ia[], int ma,
	double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq, globals &gl)
{
	int i, j, k, l, m, np, np1, np2, jp, ic;

	/* N.B. curv and blmatrix called outside bright because output same for all points */
	CalcStrategyAvx::curv(a, gl);

	//   #ifdef YORP
	//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
	//   #else
	blmatrix(a[ma - 4 - Nphpar], a[ma - 3 - Nphpar]);
	//   #endif

	for (j = 0; j < mfit; j++)
	{
		for (k = 0; k <= j; k++)
			alpha[j][k] = 0;
		beta[j] = 0;
	}

	trial_chisq = 0;
	np = 0;
	np1 = 0;
	np2 = 0;

	for (i = 1; i <= gl.Lcurves; i++)
	{
		if (gl.Inrel[i]/* == 1*/) /* is the LC relative? */
		{
			gl.ave = 0;
			for (l = 1; l <= ma; l++)
				gl.dave[l] = 0;
		}
		for (jp = 1; jp <= gl.Lpoints[i]; jp++)
		{
			np++;
			for (ic = 1; ic <= 3; ic++) /* position vectors */
			{
				gl.xx1[ic] = x1[np][ic];
				gl.xx2[ic] = x2[np][ic];
			}

			if (i < gl.Lcurves)
			{
				//CalcStrategyAvx::bright(gl.xx1, gl.xx2, x3[np], a, dyda, ma, gl.ymod, gl);
				CalcStrategyAvx::bright(gl.xx1, gl.xx2, x3[np], a, ma, gl);
			}
			else
			{
				//CalcStrategyAvx::conv(jp, dyda, ma, gl.ymod, gl);
				CalcStrategyAvx::conv(jp, ma, gl);
			}

			gl.ytemp[jp] = gl.ymod;

			if (gl.Inrel[i]/* == 1*/)
			{
				gl.ave += gl.ymod;
				for (l = 1; l <= ma; l += 4) //last odd value is not problem
				{
					__m256d avx_dyda = _mm256_load_pd(&gl.dyda[l - 1]);
					__m256d avx_dave = _mm256_loadu_pd(&gl.dave[l]);
					avx_dave = _mm256_add_pd(avx_dave, avx_dyda);
					_mm256_storeu_pd(&gl.dave[l], avx_dave);
				}
			}

			for (l = 1; l <= ma; l++)
			{
				gl.dytemp[jp][l] = gl.dyda[l - 1];
			}
			/* save lightcurves */

			//if (Lastcall == 1)
			//	Yout[np] = ymod;
		} /* jp, lpoints */

		if (Lastcall != 1)
		{
			__m256d avx_ave, avx_coef, avx_ytemp;
			avx_ave = _mm256_set1_pd(gl.ave);
			for (jp = 1; jp <= gl.Lpoints[i]; jp++)
			{
				np1++;
				if (gl.Inrel[i] /*== 1*/)
				{
					gl.coef = sig[np1] * gl.Lpoints[i] / gl.ave;
					avx_coef = _mm256_set1_pd(gl.coef);
					avx_ytemp = _mm256_broadcast_sd(&gl.ytemp[jp]);
					for (l = 1; l <= ma; l += 4)
					{
						__m256d avx_dytemp = _mm256_loadu_pd(&gl.dytemp[jp][l]);
						__m256d avx_dave = _mm256_loadu_pd(&gl.dave[l]);
						avx_dytemp = _mm256_sub_pd(avx_dytemp, _mm256_div_pd(_mm256_mul_pd(avx_ytemp, avx_dave), avx_ave));
						avx_dytemp = _mm256_mul_pd(avx_dytemp, avx_coef);
						_mm256_storeu_pd(&gl.dytemp[jp][l], avx_dytemp);
					}

					gl.ytemp[jp] *= gl.coef;
					/* Set the size scale coeff. deriv. explicitly zero for relative lcurves */
					gl.dytemp[jp][1] = 0;
				}
			}
			if (ia[0]) //not relative
			{
				for (jp = 1; jp <= gl.Lpoints[i]; jp++)
				{
					gl.ymod = gl.ytemp[jp];
					for (l = 1; l <= ma; l++)
						gl.dyda[l - 1] = gl.dytemp[jp][l];
					np2++;
					gl.sig2i = 1 / (sig[np2] * sig[np2]);
					gl.wght = gl.Weight[np2];
					gl.dy = y[np2] - gl.ymod;
					j = 0;
					//
					double sig2iwght = gl.sig2i * gl.wght;
					//l=0
					gl.wt = gl.dyda[0] * sig2iwght;
					alpha[j][0] += gl.wt * gl.dyda[0];
					beta[j] += gl.dy * gl.wt;
					j++;
					//
					for (l = 1; l <= lastone; l++)  //line of ones
					{
						gl.wt = gl.dyda[l] * sig2iwght;
						__m256d avx_wt = _mm256_set1_pd(gl.wt);
						k = 0;
						//m=0
						alpha[j][k] += gl.wt * gl.dyda[0];
						k++;
						for (m = 1; m <= l; m += 4)
						{
							__m256d avx_alpha = _mm256_loadu_pd(&alpha[j][k]);
							__m256d avx_dyda = _mm256_loadu_pd(&gl.dyda[m]);
							avx_alpha = _mm256_add_pd(avx_alpha, _mm256_mul_pd(avx_wt, avx_dyda));
							_mm256_storeu_pd(&alpha[j][k], avx_alpha);
							k += 4;
						} /* m */
						beta[j] += gl.dy * gl.wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							gl.wt = gl.dyda[l] * sig2iwght;
							__m256d avx_wt = _mm256_set1_pd(gl.wt);
							k = 0;
							//m=0
							alpha[j][k] += gl.wt * gl.dyda[0];
							k++;
							int kk = k;
							for (m = 1; m <= lastone; m += 4)
							{
								__m256d avx_alpha = _mm256_loadu_pd(&alpha[j][kk]);
								__m256d avx_dyda = _mm256_loadu_pd(&gl.dyda[m]);
								avx_alpha = _mm256_add_pd(avx_alpha, _mm256_mul_pd(avx_wt, avx_dyda));
								_mm256_storeu_pd(&alpha[j][kk], avx_alpha);
								kk += 4;
							} /* m */
							k += lastone;
							for (m = lastone + 1; m <= l; m++)
								if (ia[m])
								{
									alpha[j][k] += gl.wt * gl.dyda[m];
									k++;
								}
							beta[j] += gl.dy * gl.wt;
							j++;
						}
					} /* l */

					trial_chisq += gl.dy * gl.dy * sig2iwght;
				} /* jp */
			}
			else //relative ia[0]==0
			{
				for (jp = 1; jp <= gl.Lpoints[i]; jp++)
				{
					gl.ymod = gl.ytemp[jp];
					for (l = 1; l <= ma; l++)
						gl.dyda[l - 1] = gl.dytemp[jp][l];
					np2++;
					gl.sig2i = 1 / (sig[np2] * sig[np2]);
					gl.wght = gl.Weight[np2];
					gl.dy = y[np2] - gl.ymod;
					j = 0;
					//
					double sig2iwght = gl.sig2i * gl.wght;
					// l=0
					//
					for (l = 1; l <= lastone; l++)  //line of ones
					{
						gl.wt = gl.dyda[l] * sig2iwght;
						__m256d avx_wt = _mm256_set1_pd(gl.wt);
						k = 0;
						//m=0
						//
						for (m = 1; m <= l; m += 4)
						{
							__m256d avx_alpha = _mm256_load_pd(&alpha[j][k]);
							__m256d avx_dyda = _mm256_loadu_pd(&gl.dyda[m]);
							avx_alpha = _mm256_add_pd(avx_alpha, _mm256_mul_pd(avx_wt, avx_dyda));
							_mm256_store_pd(&alpha[j][k], avx_alpha);
							k += 4;
						} /* m */
						beta[j] += gl.dy * gl.wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							gl.wt = gl.dyda[l] * sig2iwght;
							__m256d avx_wt = _mm256_set1_pd(gl.wt);
							//m=0
							//
							int kk = 0;
							for (m = 1; m <= lastone; m += 4)
							{
								__m256d avx_alpha = _mm256_load_pd(&alpha[j][kk]);
								__m256d avx_dyda = _mm256_loadu_pd(&gl.dyda[m]);
								avx_alpha = _mm256_add_pd(avx_alpha, _mm256_mul_pd(avx_wt, avx_dyda));
								_mm256_store_pd(&alpha[j][kk], avx_alpha);
								kk += 4;
							} /* m */
							k = lastone;
							for (m = lastone + 1; m <= l; m++)
								if (ia[m])
								{
									alpha[j][k] += gl.wt * gl.dyda[m];
									k++;
								}
							beta[j] += gl.dy * gl.wt;
							j++;
						}
					} /* l */

					trial_chisq += gl.dy * gl.dy * sig2iwght;
				} /* jp */
			}
		} /* Lastcall != 1 */

		//if ((Lastcall == 1) && (Inrel[i] == 1))
		//	Sclnw[i] = Scale * Lpoints[i] * sig[np] / gl.ave;

	} /* i,  lcurves */

	for (j = 1; j < mfit; j++)
		for (k = 0; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];
}

