#include "globals.h"
#include "CalcStrategyNone.hpp"
#include "arrayHelpers.hpp"

/**
 * @brief Computes the convexity regularization function.
 *
 * This function calculates the convexity regularization function, updating the global variables `ymod` and `dyda` based on the given parameters and the global data.
 *
 * @param nc An integer representing the current coefficient index.
 * @param ma An integer representing the number of coefficients.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note The function modifies the global variables `ymod` and `dyda`.
 *
 * @date 8.11.2006
 */
void CalcStrategyNone::conv(const int nc, const int ma, globals &gl)
{
	gl.ymod = 0;

	for (auto j = 1; j <= ma; j++)
	{
		gl.dyda[j] = 0;
	}

	for (int i = 0; i < Numfac; i++)
	{
		gl.ymod += gl.Area[i] * gl.Nor[nc - 1][i];

		for (auto j = 0; j < Ncoef; j++)
		{
			gl.dyda[j] += gl.Darea[i] * gl.Dg[i][j] * gl.Nor[nc - 1][i];
		}
	}
}
