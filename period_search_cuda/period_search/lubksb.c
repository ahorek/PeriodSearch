/* Numerical Recipes */

#include <vector>

#include "stdafx.h"

//void lubksb(double **a, int n, int indx[], double b[])
void lubksb(const std::vector<std::vector<double>>& a, const int n, const std::vector<int>& indx, std::vector<double>& b)
{
   int i, ii=0, j;
   double sum;

   for (i = 1; i <= n; i++)
   {
      const int ip = indx[i];
      sum = b[ip];
      b[ip] = b[i];
      if (ii)
      {
          for (j = ii; j <= i - 1; j++)
          {
              sum -= a[i][j] * b[j];
          }          
      }
      else if (sum != 0.0)
      {
          ii = i;
      }

      b[i] = sum;
   }

   for (i = n; i >= 1; i--)
   {
      sum = b[i];
      for (j = i + 1; j <= n; j++)
      {
          sum -= a[i][j] * b[j];
      }

      b[i] = sum / a[i][i];
   }
}
