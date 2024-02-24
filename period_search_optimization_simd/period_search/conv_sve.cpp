/* Convexity regularization function

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"
#include "CalcStrategySve.hpp"

#if defined(__GNUC__)
__attribute__((__target__("+sve")))
#endif
double CalcStrategySve::conv(int nc, double dres[], int ma)
{
    int i, j;
    double res = 0;

    for (j = 1; j <= ma; j++)
        dres[j] = 0;

    for (i = 0; i < Numfac; i++) {
        res += Area[i] * Nor[nc - 1][i];
        double *Dg_row = Dg[i];
		float64x2_t sve_Darea = svdup_n_f64(Darea[i]);
		float64x2_t sve_Nor = svdup_n_f64(Nor[nc - 1][i]);
		for (j = 0; j < Ncoef; j += svcntd()) {
    		float64x2_t sve_dres = svld1_f64(&dres[j]);
    		float64x2_t sve_Dg = svld1_f64(&Dg_row[j]);

    		sve_dres = svadd_f64_x(sve_dres, svmul_f64_x(svmul_f64_x(sve_Darea, sve_Dg), sve_Nor));
    		svst1_f64(&dres[j], sve_dres);
		}
    }
    return res;
}
