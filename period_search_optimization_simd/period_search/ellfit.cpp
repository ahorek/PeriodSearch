#include <cmath>
#include "globals.h"
#include "declarations.h"
#include "arrayHelpers.hpp"

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
void ellfit(std::vector<double>& cg, double a, double b, double c, int ndir, int ncoef, const std::vector<double>& at, const std::vector<double>& af)
{
    int i, m, l, n, j, k;
    // int *indx;

    double sum, st;
    //double *fitvec, *d, *er, **fmat, **fitmat;

    //indx = vector_int(ncoef);
    //fitvec = vector_double(ncoef);
    //er = vector_double(ndir);
    //d = vector_double(1);

    //fmat = matrix_double(ndir, ncoef);
    //fitmat = matrix_double(ncoef, ncoef);

    std::vector<int> indx(ncoef + 1, 0);
    std::vector<double> fitvec(ncoef + 1, 0.0);
    std::vector<double> er(ndir + 1, 0.0);
    std::vector<double> d(2, 0.0);

    std::vector<std::vector<double>> fmat;
    init_matrix(fmat, ndir + 1, ncoef + 1, 0.0);

    std::vector<std::vector<double>> fitmat;
    init_matrix(fitmat, ncoef + 1, ncoef + 1, 0.0);

    /* Compute the LOGcurv.func. of the ellipsoid */
    for (i = 1; i <= ndir; i++)
    {
        st = sin(at[i]);
        sum = pow(a * st * cos(af[i]), 2) + pow(b * st * sin(af[i]), 2) +
            pow(c * cos(at[i]), 2);
        er[i] = 2 * (log(a * b * c) - log(sum));
    }
    /* Compute the sph. harm. values at each direction and construct the matrix fmat from them */
    for (i = 1; i <= ndir; i++)
    {
        n = 0;
        for (m = 0; m <= Mmax; m++)
            for (l = m; l <= Lmax; l++)
            {
                n++;
                if (m != 0)
                {
                    fmat[i][n] = Pleg[i][l][m] * cos(m * af[i]);
                    n++;
                    fmat[i][n] = Pleg[i][l][m] * sin(m * af[i]);
                }
                else
                    fmat[i][n] = Pleg[i][l][m];
            }
    }

    /* Fit the coefficients r from fmat[ndir,ncoef]*r[ncoef]=er[ndir] */
    for (i = 1; i <= ncoef; i++)
    {
        for (j = 1; j <= ncoef; j++)
        {
            fitmat[i][j] = 0;
            for (k = 1; k <= ndir; k++)
                fitmat[i][j] = fitmat[i][j] + fmat[k][i] * fmat[k][j];

        }
        fitvec[i] = 0;

        for (j = 1; j <= ndir; j++)
            fitvec[i] = fitvec[i] + fmat[j][i] * er[j];
    }

    // For Unit test reference only
    //printArray(fitmat, ncoef, ncoef, "fitmat[x][y]:");

    ludcmp(fitmat, ncoef, indx, d);

    //printArray(fitmat, ncoef, ncoef, "fitvec_after_lubksb");
    //printArray(fitvec, ncoef, "fitvec_before_lubksb");

    lubksb(fitmat, ncoef, indx, fitvec);

    // For Unit test reference only
    //printArray(fitvec, ncoef, "fitvec");

    for (i = 1; i <= ncoef; i++)
        cg[i] = fitvec[i];

    // For Unit tests reference only
    //printArray(cg, ncoef, "cg");

    //deallocate_matrix_double(fitmat, ncoef);
    //deallocate_matrix_double(fmat, ndir);
    //deallocate_vector((void *)fitvec);
    //deallocate_vector((void *)d);
    //deallocate_vector((void *)indx);
    //deallocate_vector((void *)er);

}

