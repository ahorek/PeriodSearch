/* slightly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include "globals.h"
#include "declarations.h"
#include <pmmintrin.h>
#include "CalcStrategySse3.hpp"
#include "arrayHelpers.hpp"

// #define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
template<typename T1, typename T2>
constexpr auto MIN(T1 X, T2 Y) { return ((X) < (Y) ? (X) : (Y)); }

/* comment the following line if no YORP */
/*#define YORP*/

#if defined(__GNUC__)
__attribute__((target("sse3")))
#endif

void CalcStrategySse3::mrqcof(double** x1, double** x2, double x3[], double y[],
			  double sig[], double a[], int ia[], int ma,
		  double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq, globals& gl)
{
	int i, j, k, l, m, np, np1, np2, jp, ic;

	/* N.B. curv and blmatrix called outside bright because output same for all points */
	CalcStrategySse3::curv(a, gl);

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
				CalcStrategySse3::bright(x3[np], a, ma, gl);
			}
			else
			{
				CalcStrategySse3::conv(jp, ma, gl);
			}

			gl.ytemp[jp] = gl.ymod;

			if (gl.Inrel[i]/* == 1*/)
			{
				gl.ave += gl.ymod;
				for (l = 1; l <= ma; l += 2) //last odd value is not problem
				{
					__m128d avx_dyda = _mm_load_pd(&gl.dyda[l - 1]), avx_dave = _mm_loadu_pd(&gl.dave[l]);
					avx_dave = _mm_add_pd(avx_dave, avx_dyda);
					_mm_storeu_pd(&gl.dytemp[jp][l], avx_dyda);
					_mm_storeu_pd(&gl.dave[l], avx_dave);
				}
			}
			else
			{
				for (l = 1; l <= ma; l++)
				{
					gl.dytemp[jp][l] = gl.dyda[l - 1];
				}
			}
			/* save lightcurves */
		} /* jp, lpoints */

		if (Lastcall != 1)
		{
			__m128d avx_ave, avx_coef, avx_ytemp;
			avx_ave = _mm_set1_pd(gl.ave);
			for (jp = 1; jp <= gl.Lpoints[i]; jp++)
			{
				np1++;
				if (gl.Inrel[i] /*== 1*/)
				{
					gl.coef = sig[np1] * gl.Lpoints[i] / gl.ave;
					avx_coef = _mm_set1_pd(gl.coef);
					avx_ytemp = _mm_loaddup_pd(&gl.ytemp[jp]);
					for (l = 1; l <= ma; l += 2)
					{
						__m128d avx_dytemp = _mm_loadu_pd(&gl.dytemp[jp][l]), avx_dave = _mm_loadu_pd(&gl.dave[l]);
						avx_dytemp = _mm_sub_pd(avx_dytemp, _mm_div_pd(_mm_mul_pd(avx_ytemp, avx_dave), avx_ave));
						avx_dytemp = _mm_mul_pd(avx_dytemp, avx_coef);
						_mm_storeu_pd(&gl.dytemp[jp][l], avx_dytemp);
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
						__m128d avx_wt = _mm_set1_pd(gl.wt);
						k = 0;
						//m=0
						alpha[j][k] += gl.wt * gl.dyda[0];
						k++;
						for (m = 1; m <= l; m += 2)
						{
							__m128d avx_alpha = _mm_loadu_pd(&alpha[j][k]), avx_dyda = _mm_loadu_pd(&gl.dyda[m]);
							avx_alpha = _mm_add_pd(avx_alpha, _mm_mul_pd(avx_wt, avx_dyda));
							_mm_storeu_pd(&alpha[j][k], avx_alpha);
							k += 2;
						} /* m */
						beta[j] += gl.dy * gl.wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							gl.wt = gl.dyda[l] * sig2iwght;
							__m128d avx_wt = _mm_set1_pd(gl.wt);
							k = 0;
							//m=0
							alpha[j][k] += gl.wt * gl.dyda[0];
							k++;
							int kk = k;
							for (m = 1; m <= lastone; m += 2)
							{
								__m128d avx_alpha = _mm_loadu_pd(&alpha[j][kk]), avx_dyda = _mm_loadu_pd(&gl.dyda[m]);
								avx_alpha = _mm_add_pd(avx_alpha, _mm_mul_pd(avx_wt, avx_dyda));
								_mm_storeu_pd(&alpha[j][kk], avx_alpha);
								kk += 2;
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
						__m128d avx_wt = _mm_set1_pd(gl.wt);
						k = 0;
						//m=0
						//
						for (m = 1; m <= l; m += 2)
						{
							__m128d avx_alpha = _mm_load_pd(&alpha[j][k]), avx_dyda = _mm_loadu_pd(&gl.dyda[m]);
							avx_alpha = _mm_add_pd(avx_alpha, _mm_mul_pd(avx_wt, avx_dyda));
							_mm_store_pd(&alpha[j][k], avx_alpha);
							k += 2;
						} /* m */
						beta[j] += gl.dy * gl.wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							gl.wt = gl.dyda[l] * sig2iwght;
							__m128d avx_wt = _mm_set1_pd(gl.wt);
							//m=0
							//
							int kk = 0;
							for (m = 1; m <= lastone; m += 2)
							{
								__m128d avx_alpha = _mm_load_pd(&alpha[j][kk]), avx_dyda = _mm_loadu_pd(&gl.dyda[m]);
								avx_alpha = _mm_add_pd(avx_alpha, _mm_mul_pd(avx_wt, avx_dyda));
								_mm_store_pd(&alpha[j][kk], avx_alpha);
								kk += 2;
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
	} /* i,  lcurves */

	for (j = 1; j < mfit; j++)
		for (k = 0; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];
}

