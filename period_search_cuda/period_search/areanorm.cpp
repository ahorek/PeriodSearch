#include "stdafx.h"
#include <cmath>

#include "arrayHelpers.hpp"
#include "globals.h"
#include "declarations.h"

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
 *
 * @note This function was originally written on 8.11.2006.
 */
void areanorm(const std::vector<double>& t, const std::vector<double>& f, const int ndir, const int nfac, 
                const std::vector<std::vector<int>>& ifp, std::vector<double>& at, std::vector<double>& af)
{
    double c[4], vx[4], vy[4], vz[4];

    std::vector<double> x(ndir + 1, 0.0);
    std::vector<double> y(ndir + 1, 0.0);
    std::vector<double> z(ndir + 1, 0.0);

    for (int i = 1; i <= ndir; i++)
    {
        const double st = sin(t[i]);
        x[i] = st * cos(f[i]);
        y[i] = st * sin(f[i]);
        z[i] = cos(t[i]);
    }

    for (int i = 1; i <= nfac; i++)
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
        const double clen2 = dot_product(c, c);
        const double clen = sqrt(clen2);

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
}