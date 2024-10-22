/* slighly changed code from Numerical Recipes
   converted from Mikko's fortran code

   8.11.2006
*/

#include <cstdio>
#include <cstdlib>
#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "CalcStrategyNone.hpp"
#include "arrayHelpers.hpp"

/* comment the following line if no YORP */
/*#define YORP*/

void CalcStrategyNone::mrqcof(double** x1, double** x2, double x3[], double y[],
	double sig[], double a[], int ia[], int ma,
	double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq, globals &gl)
{
	int i, j, k, l, m, np, np1, np2, jp, ic;

	/* N.B. curv and blmatrix called outside bright because output same for all points */
	CalcStrategyNone::curv(a, gl);

	// #ifdef YORP
	//      blmatrix(a[ma-5-Nphpar],a[ma-4-Nphpar]);
	// #else
	blmatrix(a[ma - 4 - Nphpar], a[ma - 3 - Nphpar]);
	// #endif

	for (j = 0; j < mfit; j++)
	{
		for (k = 0; k <= j; k++)
		{
			alpha[j][k] = 0;
		}
		beta[j] = 0;
	}

	trial_chisq = 0;
	np = 0;
	np1 = 0;
	np2 = 0;

	for (i = 1; i <= gl.Lcurves; i++)
	{
		if (gl.Inrel[i] == 1) /* is the LC relative? */
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
				CalcStrategyNone::bright(gl.xx1, gl.xx2, x3[np], a, dyda, ma, gl.ymod, gl);
			}
			else
			{
				CalcStrategyNone::conv(jp, dyda, ma, gl.ymod, gl);
			}

			gl.ytemp[jp] = gl.ymod;

			if (gl.Inrel[i] == 1)
				gl.ave += gl.ymod;

			for (l = 1; l <= ma; l++)
			{
				gl.dytemp[jp][l] = dyda[l - 1];
				gl.dave[l] += dyda[l - 1];
			}
			/* save lightcurves */

			//if (Lastcall == 1)
			//{
			//	Yout[np] = ymod;
			//}
		} /* jp, lpoints */

		if (Lastcall != 1)
		{
			for (jp = 1; jp <= gl.Lpoints[i]; jp++)
			{
				np1++;
				if (gl.Inrel[i] == 1)
				{
					gl.coef = sig[np1] * gl.Lpoints[i] / gl.ave;
					for (l = 1; l <= ma; l++)
					{
						gl.dytemp[jp][l] = gl.coef * (gl.dytemp[jp][l] - gl.ytemp[jp] * gl.dave[l] / gl.ave);
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
						dyda[l - 1] = gl.dytemp[jp][l];
					np2++;
					gl.sig2i = 1 / (sig[np2] * sig[np2]);
					gl.wght = gl.Weight[np2];
					gl.dy = y[np2] - gl.ymod;
					j = 0;
					//
					double sig2iwght = gl.sig2i * gl.wght;
					//l=0
					gl.wt = dyda[0] * sig2iwght;
					alpha[j][0] += gl.wt * dyda[0];
					beta[j] += gl.dy * gl.wt;
					j++;
					//
					for (l = 1; l <= lastone; l++)  //line of ones
					{
						gl.wt = dyda[l] * sig2iwght;
						k = 0;
						//m=0
						alpha[j][k] += gl.wt * dyda[0];
						k++;
						for (m = 1; m <= l; m++)
						{
							alpha[j][k] += gl.wt * dyda[m];
							k++;
						} /* m */
						beta[j] += gl.dy * gl.wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							gl.wt = dyda[l] * sig2iwght;
							k = 0;
							//m=0
							alpha[j][k] += gl.wt * dyda[0];
							k++;
							int kk = k;
							for (m = 1; m <= lastone; m++)
							{
								alpha[j][k] = alpha[j][kk] + gl.wt * dyda[m];
								kk++;
							} /* m */
							k += lastone;
							for (m = lastone + 1; m <= l; m++)
							{
								if (ia[m])
								{
									alpha[j][k] += gl.wt * dyda[m];
									k++;
								}
							} /* m */
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
						dyda[l - 1] = gl.dytemp[jp][l];
					np2++;
					gl.sig2i = 1 / (sig[np2] * sig[np2]);
					gl.wght = gl.Weight[np2];
					gl.dy = y[np2] - gl.ymod;
					j = 0;
					//
					double sig2iwght = gl.sig2i * gl.wght;
					//l=0
					//
					for (l = 1; l <= lastone; l++)  //line of ones
					{
						gl.wt = dyda[l] * sig2iwght;
						k = 0;
						//m=0
						//
						for (m = 1; m <= l; m++)
						{
							alpha[j][k] += gl.wt * dyda[m];
							k++;
						} /* m */
						beta[j] += gl.dy * gl.wt;
						j++;
					} /* l */
					for (; l <= lastma; l++)  //rest parameters
					{
						if (ia[l])
						{
							gl.wt = dyda[l] * sig2iwght;
							//m=0
							//
							int kk = 0;
							for (m = 1; m <= lastone; m++)
							{
								alpha[j][kk] += gl.wt * dyda[m];
								kk++;
							} /* m */
							// k += lastone;
							k = lastone;
							for (m = lastone + 1; m <= l; m++)
							{
								if (ia[m])
								{
									alpha[j][k] += gl.wt * dyda[m];
									k++;
								}
							} /* m */
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

	//printf("np: %d\n", np);

	for (j = 1; j < mfit; j++)
		for (k = 0; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];
}
