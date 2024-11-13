/* Convexity regularization function

   8.11.2006
*/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "globals.h"
#include "declarations.h"
#include "CalcStrategyNone.hpp"
#include "arrayHelpers.hpp"

//void CalcStrategyNone::conv(int nc, double dres[], int ma, double &result, globals &gl)
void CalcStrategyNone::conv(int nc, int ma, globals &gl)
{
	int i, j;

	gl.ymod = 0;
	for (j = 1; j <= ma; j++)
		gl.dyda[j] = 0;

	for (i = 0; i < Numfac; i++)
	{
		gl.ymod += gl.Area[i] * gl.Nor[nc - 1][i];

		for (j = 0; j < Ncoef; j++)
		{
			gl.dyda[j] += gl.Darea[i] * gl.Dg[i][j] * gl.Nor[nc - 1][i];
		}
	}
}
