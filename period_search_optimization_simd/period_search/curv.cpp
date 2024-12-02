/* Curvature function (and hence facet area) from Laplace series

   8.11.2006
*/

#include "globals.h"
#include "CalcStrategyNone.hpp"
#include "arrayHelpers.hpp"

void CalcStrategyNone::curv(double cg[], globals& gl)
{
    for (auto i = 1; i <= Numfac; i++)
    {
        double g = 0;
        int n = 0;
        //m=0
        for (auto l = 0; l <= Lmax; l++)
        {
            n++;
            const double fsum = cg[n] * Fc[i][0];
            g += Pleg[i][l][0] * fsum;
        }
        //
        for (auto m = 1; m <= Mmax; m++)
        {
            for (auto l = m; l <= Lmax; l++)
            {
                n++;
                double fsum = cg[n] * Fc[i][m];
                n++;
                fsum += cg[n] * Fs[i][m];
                g += Pleg[i][l][m] * fsum;
            }
        }

        g = exp(g);
        gl.Area[i - 1] = gl.Darea[i - 1] * g;

        for (auto k = 1; k <= n; k++)
        {
            gl.Dg[i - 1][k - 1] = g * Dsph[i][k];
        }
    }
}
