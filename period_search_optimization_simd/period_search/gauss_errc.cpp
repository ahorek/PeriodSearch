#include <cmath>
#include <cstdlib>
#include <vector>
//#include "declarations.h"
#include "CalcStrategyNone.hpp"

#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

/**
* @brief Solves a linear system of equations using Gaussian elimination with partial pivoting.
*
* This function implements the Gaussian elimination algorithm with partial pivoting to solve a
* linear system of equations. It rearranges the covariance matrix and the right-hand side vector
* to find the solution.
*
* @param gl A reference to a globals structure containing the covariance matrix and other global data.
* @param n The dimension of the system (number of equations/variables).
* @param b A vector of doubles representing the right-hand side vector of the system.
* @param error An integer reference to store error codes:
*              - 0: No error
*              - 1: Singular matrix
*              - 2: Zero pivot element
*
* @note The function modifies the covariance matrix `covar` in place.
*
* @source Numerical Recipes
*
* @date 8.11.2006
*/
void CalcStrategyNone::gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int& error)
{
    int i, icol = 0, irow = 0, j, k, l, ll;
    double big, dum, pivinv, temp;

    auto& a = gl.covar;

    std::vector<int> indxc(n + 1 + 1, 0);
    std::vector<int> indxr(n + 1 + 1, 0);
    std::vector<int> ipiv(n + 1 + 1, 0);

    for (i = 1; i <= n; i++)
    {
        big = 0.0;
        for (j = 0; j < n; j++) //* 1 -> 0
        {
            if (ipiv[j] != 1)
            {
                for (k = 0; k < n; k++) //* 1 -> 0
                {
                    if (ipiv[k] == 0)
                    {
                        if (fabs(a[j][k]) >= big)
                        {
                            big = fabs(a[j][k]);
                            irow = j;
                            icol = k;
                        }
                    }
                    else if (ipiv[k] > 1)
                    {
                        error = 1;

                        return;
                    }
                }
            }
        }

        ++(ipiv[icol]);
        if (irow != icol)
        {
            for (l = 0; l < n; l++) SWAP(a[irow][l], a[icol][l])
                SWAP(b[irow], b[icol])
        }

        indxr[i] = irow;
        indxc[i] = icol;

        if (a[icol][icol] == 0.0)
        {
            error = 2;

            return;
        }

        pivinv = 1.0 / a[icol][icol];
        a[icol][icol] = 1.0;

        for (l = 0; l < n; l++)
        {
            a[icol][l] *= pivinv;
        }

        b[icol] *= pivinv;
        for (ll = 0; ll < n; ll++)
        {
            if (ll != icol)
            {
                dum = a[ll][icol];
                a[ll][icol] = 0.0;
                for (l = 0; l < n; l++)
                {
                    a[ll][l] -= a[icol][l] * dum;
                }

                b[ll] -= b[icol] * dum;
            }
        }
    }

    for (l = n; l >= 1; l--)
    {
        if (indxr[l] != indxc[l])
            for (k = 0; k < n; k++)
            {
                SWAP(a[k][indxr[l]], a[k][indxc[l]]);
            }
    }

    error = 0;

    return;
}
#undef SWAP
