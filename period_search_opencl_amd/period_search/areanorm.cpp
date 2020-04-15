/* Areas and normals of the triangulated Gaussian image sphere

   8.11.2006
*/
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "stdafx.h"
#include <math.h>
#include "globals.h"
#include "declarations.hpp"

void areanorm(double t[], double f[], int ndir, int nfac, int** ifp, double at[], double af[])
{
    int i, j;

    double  st, clen2, clen;

    double c[4], vx[4], vy[4], vz[4],
        * x, * y, * z;

    x = vector_double(ndir);
    y = vector_double(ndir);
    z = vector_double(ndir);

    for (i = 1; i <= ndir; i++)
    {
        st = sin(t[i]);
        x[i] = st * cos(f[i]);
        y[i] = st * sin(f[i]);
        z[i] = cos(t[i]);
    }
    for (i = 1; i <= nfac; i++)
    {
        /* vectors of triangle edges */
        for (j = 2; j <= 3; j++)
        {
            vx[j] = x[ifp[i][j]] - x[ifp[i][1]];
            vy[j] = y[ifp[i][j]] - y[ifp[i][1]];
            vz[j] = z[ifp[i][j]] - z[ifp[i][1]];
        }
        /* The cross product for each triangle */
        c[1] = vy[2] * vz[3] - vy[3] * vz[2];
        c[2] = vz[2] * vx[3] - vz[3] * vx[2];
        c[3] = vx[2] * vy[3] - vx[3] * vy[2];
        /* Areas (on the unit sphere) and normals */
        clen2 = c[1] * c[1] + c[2] * c[2] + c[3] * c[3];
        clen = sqrt(clen2);
        /* normal */
        normal[i][0] = c[1] / clen;
        normal[i][1] = c[2] / clen;
        normal[i][2] = c[3] / clen;
        /* direction angles of normal */
        at[i] = acos(normal[i][2]);
        af[i] = atan2(normal[i][1], normal[i][0]);
        /* triangle area */
        d_area[i] = 0.5 * clen;
    }

    deallocate_vector((void*)x);
    deallocate_vector((void*)y);
    deallocate_vector((void*)z);
}