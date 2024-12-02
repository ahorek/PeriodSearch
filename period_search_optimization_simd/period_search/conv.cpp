/* Convexity regularization function

   8.11.2006
*/

#include "globals.h"
#include "CalcStrategyNone.hpp"
#include "arrayHelpers.hpp"

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
