/* Areas and normals of the triangulated Gaussian image sphere

   8.11.2006
*/

#include <math.h>
#include "globals.h"
#include "declarations.hpp"
#include "VectorT.hpp"

using namespace std;

void areanorm(double t[], double f[], int ndir, int nfac, int **ifp, double at[], double af[])
{
    int i, j;
    double  st, clen2, clen, clen_t;
    double vx[4], vy[4], vz[4];
    math::VectorT<double> vector_c(3);
    vector<double> x(ndir + 1), y(ndir + 1), z(ndir + 1);

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
        vector_c[0] = vy[2] * vz[3] - vy[3] * vz[2];
        vector_c[1] = vz[2] * vx[3] - vz[3] * vx[2];
        vector_c[2] = vx[2] * vy[3] - vx[3] * vy[2];

        /* Areas (on the unit sphere) and normals */
        clen = vector_c.magnitude();

        /* normal */
        Nor[0][i - 1] = vector_c[0] / clen;
        Nor[1][i - 1] = vector_c[1] / clen;
        Nor[2][i - 1] = vector_c[2] / clen;

        /* direction angles of normal */
        at[i] = acos(Nor[2][i - 1]);
        af[i] = atan2(Nor[1][i - 1], Nor[0][i - 1]);

        /* triangle area */
        Darea[i - 1] = 0.5 * clen;
    }
}

