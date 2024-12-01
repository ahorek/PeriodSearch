/* N.B. The foll. L-M routines are modified versions of Press et al.
   converted from Mikko's fortran code

   8.11.2006

   Numerical recipes: Nonlinear least-squares fit, Marquardt�s method
*/

#include "globals.h"
#include "declarations.h"
#include "constants.h"
#include "arrayHelpers.hpp"

int mrqmin(double** x1, double** x2, double x3[], double y[],
	double sig[], double a[], int ia[], int ma,
	double** covar, double** alpha, globals &gl)
{
	int j, k, l, err_code;
	static int mfit, lastone, lastma; /* it is set in the first call*/
	static double* atry, * beta, * da; //beta, da are zero indexed
	double trial_chisq;

	/* deallocates memory when used in period_search */
	if (Deallocate == 1)
	{
		deallocate_vector((void*)atry);
		deallocate_vector((void*)beta);
		deallocate_vector((void*)da);
		return(0);
	}

	if (Lastcall != 1)
	{
		if (Alamda < 0)
		{
			atry = vector_double(ma);
			beta = vector_double(ma);
			da = vector_double(ma);

			/* number of fitted parameters */
			mfit = 0;
			lastma = 0;
			for (j = 0; j < ma; j++)
			{
				if (ia[j])
				{
					mfit++;
					lastma = j;
				}
			}

			lastone = 0;
			for (j = 1; j <= lastma; j++) //ia[0] is skipped because ia[0]=0 is acceptable inside mrqcof
			{
				if (!ia[j]) break;
				lastone = j;
			}

			Alamda = Alamda_start; /* initial alambda */

			calcCtx.CalculateMrqcof(x1, x2, x3, y, sig, a, ia, ma, alpha, beta, mfit, lastone, lastma, trial_chisq, gl);

			Ochisq = trial_chisq;
			for (j = 1; j <= ma; j++)
			{
				atry[j] = a[j];
			}
		}

		for (j = 0; j < mfit; j++)
		{
			for (k = 0; k < mfit; k++)
			{
				covar[j][k] = alpha[j][k];
			}

			covar[j][j] = alpha[j][j] * (1 + Alamda);
			da[j] = beta[j];
		}

		calcCtx.CalculateGaussErrc(covar, mfit, da, err_code);

		if (err_code != 0) return(err_code);

		j = 0;
		for (l = 1; l <= ma; l++)
		{
			if (ia[l - 1])
			{
				atry[l] = a[l] + da[j];
				j++;
			}
		}
	} /* Lastcall != 1 */

	if (Lastcall == 1)
	{
		for (l = 1; l <= ma; l++)
		{
			atry[l] = a[l];
		}
	}

	calcCtx.CalculateMrqcof(x1, x2, x3, y, sig, atry, ia, ma, covar, da, mfit, lastone, lastma, trial_chisq, gl);

	Chisq = trial_chisq;

	if (Lastcall == 1)
	{
		deallocate_vector((void*)atry);
		deallocate_vector((void*)beta);
		deallocate_vector((void*)da);
		return(0);
	}

	if (trial_chisq < Ochisq)
	{
		Alamda = Alamda / Alamda_incr;
		for (j = 0; j < mfit; j++)
		{
			for (k = 0; k < mfit; k++)
			{
				alpha[j][k] = covar[j][k];
			}
			beta[j] = da[j];
		}

		for (l = 1; l <= ma; l++)
		{
			a[l] = atry[l];
		}
	}
	else
	{
		Alamda = Alamda_incr * Alamda;
		Chisq = Ochisq;
	}

	return(0);
}

