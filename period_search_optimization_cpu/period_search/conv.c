/* Convexity regularization function

   8.11.2006
*/

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "globals.h"
#include "declarations.h"

double conv(int nc, double dres[], int ma)
{
   int i, j;

   double res;

   res = 0;
   for (j = 1; j <= ma; j++)
      dres[j] = 0;

   for (i = 0; i < Numfac; i++)
   {
      res += Area[i] * Nor[nc-1][i];
      for (j = 0; j < Ncoef; j++) {
         dres[j] += Darea[i] * Dg[i][j] * Nor[nc-1][i];
      }
   }

   return(res);
}
