#include <cmath>
#include "declarations.h"
#include "arrayHelpers.hpp"

/**
 * @brief Calculates areas and normals of the triangulated Gaussian image sphere.
 *
 * This function computes the areas and normals for each facet of a triangulated
 * Gaussian image sphere. It takes spherical coordinates (theta, phi) and converts them
 * to Cartesian coordinates, then calculates the cross product for each triangle to find
 * the normal vector and the area of the triangle on the unit sphere.
 *
 * @param t A vector of theta angles (in radians) for the directions.
 * @param f A vector of phi angles (in radians) for the directions.
 * @param ndir The number of directions.
 * @param nfac The number of facets (triangles).
 * @param ifp A 2D vector of indices defining the vertices of each facet.
 * @param at A vector to store the theta angles of the normal vectors.
 * @param af A vector to store the phi angles of the normal vectors.
 * @param gl A reference to a globals structure containing necessary global data.
 *
 * @note This function was originally written on 8.11.2006.
 */
void areanorm(const std::vector<double>& t, const std::vector<double>& f, const int ndir, const int nfac, const std::vector<std::vector<int>>& ifp,
    std::vector<double>& at, std::vector<double>& af, globals &gl)
{
    int i;
    double c[4]{}, vx[4]{}, vy[4]{}, vz[4]{};

    //double* x = vector_double(ndir);
    //double* y = vector_double(ndir);
    //double* z = vector_double(ndir);
    std::vector<double> x(ndir + 1, 0.0);
    std::vector<double> y(ndir + 1, 0.0);
    std::vector<double> z(ndir + 1, 0.0);

    for (i = 1; i <= ndir; i++)
    {
        const double st = sin(t[i]);
        x[i] = st * cos(f[i]);
        y[i] = st * sin(f[i]);
        z[i] = cos(t[i]);
		//printf("x[%3d] % 0.6f\ty[%3d] % 0.6f\tz[%3d] % 0.6f\n", i, x[i], i, y[i], i, z[i]);
    }

    for (i = 1; i <= nfac; i++)
    {
        /* vectors of triangle edges */
        for (int j = 2; j <= 3; j++)
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
        const double clen2 = c[1] * c[1] + c[2] * c[2] + c[3] * c[3];
        const double clen = sqrt(clen2);
		//printf("[%3d] % 0.6f\n", i, clen);

        /* normal */
        gl.Nor[0][i - 1] = c[1] / clen;
        gl.Nor[1][i - 1] = c[2] / clen;
        gl.Nor[2][i - 1] = c[3] / clen;

        /* direction angles of normal */
        at[i] = acos(gl.Nor[2][i - 1]);
        af[i] = atan2(gl.Nor[1][i - 1], gl.Nor[0][i - 1]);

        /* triangle area */
        gl.Darea[i - 1] = 0.5 * clen;
		//printf("[%3d] % 0.6f\n", i-1, Darea[i - 1]);
    }

    // NOTE: For unit tests reference only
    //printArray(at, i - 1, "at");
    //printArray(af, i - 1, "af");
	//printArray(Darea, nfac, "Darea");

    //deallocate_vector((void *)x);
    //deallocate_vector((void *)y);
    //deallocate_vector((void *)z);
}