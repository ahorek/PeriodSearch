#include <cmath>
#include "globals.h"
#include "constants.h"

/**
 * @brief Computes the rotation matrix and its derivatives.
 *
 * This function calculates the rotation matrix for a given angular velocity (`omg`) and time (`t`),
 * as well as the derivatives of the rotation matrix with respect to the angular velocity.
 *
 * @param omg The angular velocity in radians per unit time.
 * @param t The time at which the rotation matrix is evaluated.
 * @param tmat A 2D array to store the computed rotation matrix.
 * @param dtm A 3D array to store the derivatives of the rotation matrix with respect to angular velocity.
 *
 * @note The function modifies the global variables `Blmat` and `Dblm`.
 *
 * @source Converted from Mikko's Fortran code
 *
 * @date 8.11.2006
 */
void matrix(const double omg, const double t, double tmat[][4], double dtm[][4][4])
{
    double f, cf, sf, dfm[4][4], fmat[4][4];

    int i, j, k;

    /* phase of rotation */
    f = omg * t + Phi_0;
    f = fmod(f, 2 * PI); /* may give little different results than Mikko's */
    cf = cos(f);
    sf = sin(f);
    mtsf = -t * sf;
    mtcf = -t * cf;
    tcf = t * cf;
    
    /* rotation matrix, Z axis, angle f */
    alignas(64) double fmat[4][4] = {
        {0,  0,   0,  0},  
        {0,  cf,  sf, 0},  
        {0, -sf,  cf, 0},  
        {0,  0,   0,  1}   
      };
  
    /* Ders. w.r.t omg */
    alignas(64) double dfm[4][4] = {
        {0,       0,       0, 0},  
        {0,    mtsf,     tcf, 0},  
        {0,    mtcf,    mtsf, 0},  
        {0,       0,       0, 0}   
    };
    /* Construct tmat (complete rotation matrix) and its derivatives */
    // double tmat[4][4] = {0};
    // double dtm[4][4][4] = {0};

    for (int i = 1; i <= 3; i++) {
        for (j = 1; j <= 3; j++) {
            __m256 rowC = _mm256_setzero_ps();
        
            for (int k = 1; k <= 3; k++) {
                __m256 a = _mm256_set1_ps(fmat[i][k]);
                __m256 b = _mm256_loadu_ps(&Blmat[k][j]);
                rowC = _mm256_fmadd_ps(a, b, rowC);
            }
        }
    }

    for (i = 1; i <= 3; i++)
        for (j = 1; j <= 3; j++)
        {
            for (k = 1; k <= 3; k++)
            {
                tmat[i][j] = tmat[i][j] + fmat[i][k] * Blmat[k][j];
                dtm[1][i][j] = dtm[1][i][j] + fmat[i][k] * Dblm[1][k][j];
                dtm[2][i][j] = dtm[2][i][j] + fmat[i][k] * Dblm[2][k][j];
                dtm[3][i][j] = dtm[3][i][j] + dfm[i][k] * Blmat[k][j];
            }
        }

/*
    __m512d fmat_row, blmat_row, dtm_row, dfm_row, tmat_row;

    for (int i = 1; i <= 3; i++) {
        for (int j = 1; j <= 3; j++) {
            fmat_row = _mm512_load_pd(&fmat[i][1]);
            dfm_row = _mm512_load_pd(&dfm[i][1]);

            for (int k = 1; k <= 3; k++) {
                blmat_row = _mm512_load_pd(&Blmat[k][1]);
                dtm_row = _mm512_load_pd(&Dblm[1][k][1]);

                tmat_row = _mm512_fmadd_pd(fmat_row, blmat_row, _mm512_setzero_pd());
                dtm[1][i][j] += _mm512_reduce_add_pd(_mm512_mul_pd(fmat_row, dtm_row));
                dtm[2][i][j] += _mm512_reduce_add_pd(_mm512_mul_pd(fmat_row, _mm512_load_pd(&Dblm[2][k][1])));
                dtm[3][i][j] += _mm512_reduce_add_pd(_mm512_mul_pd(dfm_row, blmat_row));
            }

            tmat[i][j] = _mm512_reduce_add_pd(tmat_row);
        }
    }
    */


    /*printf("\ntmat[4][4]:\n");
    for(int q = 0; q <= 4; q++)
    {
        printf("double _tmat_%d[] ={", q);
        for(int p = 0; p <= 4; p++)
        {
            printf("%.30f, ", tmat[p][q]);
        }
        printf("};\n");
    }

    printf("\ndtm[4][4][4]:\n");
    for (int r = 0; r <= 4; r++) {
        for (int q = 0; q <= 4; q++)
        {
            printf("double _dtm_%d_%d[] ={", q, r);
            for (int p = 0; p <= 4; p++)
            {
                printf("%.30f, ", dtm[p][q][r]);
            }
            printf("};\n");
        }
    }*/
}
