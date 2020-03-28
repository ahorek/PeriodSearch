/* Numerical Recipes */

#define TINY 1.0e-20;
#include <stdio.h>
#include <stdlib.h>
#include "declarations.h"
#include <math.h>
#include "../period_search/arrayHelpers.hpp"

void ludcmp(double **a, int n, int indx[], double d[])
{
    int i, imax = -999, j, k;
    double big, dum, sum, temp;
    double *v;

    v = vector_double(n);

    *d = 1.0;
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
            *d = -(*d);
            v[imax] = v[j];
        }
        indx[j] = imax;
        if (a[j][j] == 0.0) a[j][j] = TINY;
        if (j != n)
        {
            dum = 1.0 / (a[j][j]);
            for (i = j + 1; i <= n; i++) a[i][j] *= dum;
        }
    }

    // For Unit tests reference only
    //printArray(indx, n, "indx");
    
    deallocate_vector((void *)v);
}
#undef TINY
