// ReSharper disable IdentifierTypo
#include <vector>

// TODO: LUdcmp::solve()

/**
 * @brief Solves the set of linear equations Ax = b using LU decomposition.
 *
 * This function solves the set of linear equations Ax = b where A is a matrix,
 * and x and b are vectors. It uses the LU decomposition solution method from
 * the "Numerical Recipes" book. It is assumed that the input matrix `a` has already
 * been decomposed into its LU form.
 *
 * The function operates on 1-based indexed arrays, and it does not use the zero
 * elements of the vectors and matrix.
 *
 * @param a A 2D vector representing the LU-decomposed matrix A. The matrix should be
 *          of size (n+1) x (n+1) to account for 1-based indexing.
 * @param n The dimension of the matrix A.
 * @param indx A vector of integers representing the permutation vector produced
 *             during the LU decomposition. The vector should be of size n+1.
 * @param b A vector of doubles representing the right-hand side of the equation.
 *          The vector should be of size n+1. The solution vector x will be stored
 *          in b after the function completes.
 */
void lubksb(const std::vector<std::vector<double>>& a, const int n, const std::vector<int>& indx, std::vector<double>& b)
{
   int i, ii=0, ip, j;
   double sum;

   for (i = 1; i <= n; i++)
   {
      ip = indx[i];
      sum = b[ip];
      b[ip] = b[i];
      if (ii)
         for (j = ii; j <= i-1; j++) sum -= a[i][j] * b[j];
      else if (sum) ii = i;
      b[i] = sum;
   }
   for (i = n; i >= 1; i--)
   {
      sum = b[i];
      for (j = i + 1;j <= n; j ++) sum -= a[i][j] * b[j];
      b[i] = sum / a[i][i];
   }
}
