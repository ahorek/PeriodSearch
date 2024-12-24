/* Find the curv. fn. Laplace series for given ellipsoid
   converted from Mikko's fortran code

   8.11.2006
*/

#include "stdafx.h"
#include <cmath>
#include <vector>

#include "arrayHelpers.hpp"
#include "globals.h"
#include "declarations.h"

/**
 * @brief Finds the curvature function Laplace series for a given ellipsoid.
 *
 * This function computes the curvature function Laplace series for a given ellipsoid
 * by fitting spherical harmonics. It converts the original Fortran code from Mikko (J.Durech, 8.11.2006).
 *
 * @param cg A vector to store the coefficients of the fitted series.
 * @param a The semi-major axis of the ellipsoid.
 * @param b The semi-minor axis of the ellipsoid.
 * @param c The vertical axis of the ellipsoid.
 * @param ndir The number of directions.
 * @param ncoef The number of coefficients in the series.
 * @param at A vector of theta angles (in radians) for the directions.
 * @param af A vector of phi angles (in radians) for the directions.
 */
void ellfit(std::vector<double>& cg, const double a, const double b, const double c, const int ndir, const int ncoef,
    const std::vector<double>& at, const std::vector<double>& af)
{
    std::vector<int> indx(ncoef + 1, 0);
    std::vector<double> fitvec(ncoef + 1, 0.0);
    std::vector<double> er(ndir + 1, 0.0);
    std::vector<double> d(2, 0.0);

    std::vector<std::vector<double>> fmat;
    init_matrix(fmat, ndir + 1, ncoef + 1, 0.0);

    std::vector<std::vector<double>> fitmat;
    init_matrix(fitmat, ncoef +  1, ncoef + 1, 0.0);

    /* Compute the LOGcurv.func. of the ellipsoid */
    for (int i = 1; i <= ndir; i++)
    {
        const double st = sin(at[i]);
        const double sum = pow(a * st * cos(af[i]), 2) + pow(b * st * sin(af[i]), 2) + pow(c * cos(at[i]), 2);
        er[i] = 2 * (log(a * b * c) - log(sum));
    }

    /* Compute the sph. harm. values at each direction and
       construct the matrix fmat from them */
    for (int i = 1; i <= ndir; i++)
    {
        int n = 0;
        for (int m = 0; m <= m_max; m++)
        {
            for (int l = m; l <= l_max; l++)
            {
                n++;
                if (m != 0)
                {
                    fmat[i][n] = pleg[i][l][m] * cos(m * af[i]);
                    n++;
                    fmat[i][n] = pleg[i][l][m] * sin(m * af[i]);
                }
                else
                {
                    fmat[i][n] = pleg[i][l][m];
                }
            }
        }
    }

    /* Fit the coefficients r from fmat[ndir,ncoef]*r[ncoef]=er[ndir] */
    for (int i = 1; i <= ncoef; i++)
    {
        for (int j = 1; j <= ncoef; j++)
        {
            fitmat[i][j] = 0;

            for (int k = 1; k <= ndir; k++)
            {
                fitmat[i][j] = fitmat[i][j] + fmat[k][i] * fmat[k][j];
            }

        }

        fitvec[i] = 0;

        for (int j = 1; j <= ndir; j++)
        {
            fitvec[i] = fitvec[i] + fmat[j][i] * er[j];
        }
    }

    ludcmp(fitmat, ncoef, indx, d);
    lubksb(fitmat, ncoef, indx, fitvec);

    for (int i = 1; i <= ncoef; i++)
    {
        cg[i] = fitvec[i];
    }
}

