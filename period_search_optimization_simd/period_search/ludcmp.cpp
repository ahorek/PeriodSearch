// ReSharper disable IdentifierTypo
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "declarations.h"

#if _MSC_VER >= 1900 // Visual Studio 2015 or later
constexpr auto tiny = 1.0e-20;
#else
const auto tiny = 1.0e-20;
#endif

//TODO: LUdcmp:LUdcmp()

/**
 * @brief Performs LU decomposition of a matrix.
 *
 * This function decomposes a given matrix into lower and upper triangular matrices
 * using LU decomposition as described in "Numerical Recipes".
 * It is used for solving linear systems, inverting matrices, and determining the determinant.
 *
 * The function operates on 1-based indexed arrays, and it does not use the zero
 * elements of the vectors and matrix.
 *
 * @param a A 2D vector representing the matrix to be decomposed. The matrix should be
 *          of size (n+1) x (n+1) to account for 1-based indexing.
 * @param n The dimension of the matrix.
 * @param indx A vector of integers that will be filled with the permutation information
 *             produced during the decomposition. The vector should be of size n+1.
 * @param d A vector of doubles to be updated with the parity of the permutation matrix. The
 *          first element (d[0]) will be set to 1.0 or -1.0.
 */
//void ludcmp(double **a, int n, int indx[], double d[])
void ludcmp(std::vector<std::vector<double>>& a, const int n, std::vector<int>& indx, std::vector<double>& d)
{
    int i, imax = -999, j, k;
    double big, dum, sum, temp;
    //double *v;

    //v = vector_double(n);
    std::vector<double> v(n + 1);

    //*d = 1.0;
    d[0] = 1.0;
    for (i = 1; i <= n; i++)
    {
        big = 0.0;
        for (j = 1; j <= n; j++)
            if ((temp = fabs(a[i][j])) > big) big = temp;
        if (big == 0.0) { fprintf(stderr, "Singular matrix in routine ludcmp\n"); fflush(stderr); exit(4); }
        v[i] = 1.0 / big;
    }
    for (j = 1; j <= n; j++)
    {
        for (i = 1; i < j; i++)
        {
            sum = a[i][j];
            for (k = 1; k < i; k++) sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
        }
        big = 0.0;
        for (i = j; i <= n; i++)
        {
            sum = a[i][j];
            for (k = 1; k < j; k++)
                sum -= a[i][k] * a[k][j];
            a[i][j] = sum;
            if ((dum = v[i] * fabs(sum)) >= big)
            {
                big = dum;
                imax = i;
            }
        }
        if (j != imax)
        {
            for (k = 1; k <= n; k++)
            {
                dum = a[imax][k];
                a[imax][k] = a[j][k];
                a[j][k] = dum;
            }

            //*d = -(*d);
            d[0] = -(d[0]);
            v[imax] = v[j];
        }
        indx[j] = imax;
        if (a[j][j] == 0.0) a[j][j] = tiny;
        if (j != n)
        {
            dum = 1.0 / (a[j][j]);
            for (i = j + 1; i <= n; i++) a[i][j] *= dum;
        }
    }

    // For Unit tests reference only
    //printArray(indx, n, "indx");

    //deallocate_vector((void *)v);
}
