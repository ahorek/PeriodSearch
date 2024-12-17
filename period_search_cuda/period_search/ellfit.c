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

//void ellfit(double cg[], double a, double b, double c, int ndir, int ncoef, double at[], double af[])
void ellfit(std::vector<double>& cg, const double a, const double b, const double c, const int ndir, const int ncoef,
    const std::vector<double>& at, const std::vector<double>& af)
{
    //int i, j;
    //int *indx;

    // double ** fmat, ** fitmat; //*fitvec, *er, * d,

    //indx = vector_int(ncoef);
    std::vector<int> indx(ncoef + 1, 0);

    //fitvec = vector_double(ncoef);
    std::vector<double> fitvec(ncoef + 1, 0.0);

    //er = vector_double(ndir);
    std::vector<double> er(ndir + 1, 0.0);

    //d = vector_double(1);
    std::vector<double> d(2, 0.0);

    //fmat = matrix_double(ndir, ncoef);
    std::vector<std::vector<double>> fmat;
    init_matrix(fmat, ndir, ncoef, 0.0);

    //fitmat = matrix_double(ncoef, ncoef);
    std::vector<std::vector<double>> fitmat;
    init_matrix(fitmat, ncoef, ncoef, 0.0);

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

    //deallocate_matrix_double(fitmat, ncoef);
    //deallocate_matrix_double(fmat, ndir);
    //deallocate_vector((void*)fitvec);
    //deallocate_vector((void*)d);
    //deallocate_vector((void *) indx);
    //deallocate_vector((void*)er);
}

