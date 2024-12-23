// ReSharper disable IdentifierTypo
#include <vector>
#include "globals.h"
#include "declarations.h"
#include "arrayHelpers.hpp"

/**
 * @brief Performs a nonlinear least-squares fit using the Marquardt method.
 *
 * This function implements the Marquardt method for nonlinear least-squares fitting.
 * It adjusts the parameters to minimize the chi-squared value, based on the given data and initial parameters.
 *
 * @param x1 A reference to a 2D vector of doubles representing the independent variable data.
 * @param x2 A reference to a 2D vector of doubles representing additional independent variable data.
 * @param x3 A reference to a vector of doubles representing additional data points.
 * @param y A reference to a vector of doubles representing the dependent variable data.
 * @param sig A reference to a vector of doubles representing the standard deviations of the data points.
 * @param a A reference to a vector of doubles representing the initial parameters.
 * @param ia A reference to a vector of integers indicating which parameters are to be fitted.
 * @param ma An integer representing the total number of parameters.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @return An integer error code:
 *         - 0: No error
 *         - Non-zero: Indicates an error occurred during fitting
 *
 * @note The function modifies the global variables related to the fitting process. Converted from Mikko's Fortran code.
 *
 * @source Numerical Recipes: Nonlinear least-squares fit, Marquardt’s method.
 *
 * @date 8.11.2006
 */
int mrqmin(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
		std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, const int ma, globals& gl)
{
	int j, k, l, err_code;
	static int mfit, lastone, lastma; /* it is set in the first call*/
	//static double* beta, * da, * atry; //beta, da are zero indexed

	double trial_chisq;
	//double** alpha, ** covar; // Dummy placeholders

	/* deallocates memory when used in period_search */
	if (Deallocate == 1)
	{
		//deallocate_vector((void*)atry);
		//deallocate_vector((void*)beta);
		//deallocate_vector((void*)da);
		return(0);
	}

	if (Lastcall != 1)
	{
		if (Alamda < 0)
		{
			//atry = vector_double(ma);
			//beta = vector_double(ma);
			//da = vector_double(ma);

			std::fill(atry.begin(), atry.end(), 0.0);
			std::fill(beta.begin(), beta.end(), 0.0);
			std::fill(da.begin(), da.end(), 0.0);

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

			//gl.isCovar = false;
			// NOTE: Use gl.alpha
			calcCtx.CalculateMrqcof(x1, x2, x3, y, sig, a, ia, ma, beta, mfit, lastone, lastma, trial_chisq, gl, false);

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
				gl.covar[j][k] = gl.alpha[j][k];
			}

			gl.covar[j][j] = gl.alpha[j][j] * (1 + Alamda);
			da[j] = beta[j];
		}

		//calcCtx.CalculateGaussErrc(covar, mfit, da.data(), err_code);

		//auto flat2DCover = flatten2Dvector(gl.covar);
		//// Create double** from the flattened vector
		//auto rows = gl.covar.size();
		//auto cols = gl.covar[0].size();
		//double** flatCovar = new double*[rows];
	 //   for (size_t i = 0; i < rows; ++i)
	 //   {
		//	flatCovar[i] = &flat2DCover[i * cols];
	 //   }

		//calcCtx.CalculateGaussErrc(flatCovar, mfit, da.data(), err_code);
		calcCtx.CalculateGaussErrc(gl, mfit, da, err_code);

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

	//gl.isCovar = true;
	// NOTE: Use gl.covar
	calcCtx.CalculateMrqcof(x1, x2, x3, y, sig, atry, ia, ma, da, mfit, lastone, lastma, trial_chisq, gl, true);

	Chisq = trial_chisq;

	if (Lastcall == 1)
	{
		//deallocate_vector((void*)atry);
		//deallocate_vector((void*)beta);
		//deallocate_vector((void*)da);
		return(0);
	}

	if (trial_chisq < Ochisq)
	{
		Alamda = Alamda / Alamda_incr;
		for (j = 0; j < mfit; j++)
		{
			for (k = 0; k < mfit; k++)
			{
				gl.alpha[j][k] = gl.covar[j][k];
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

