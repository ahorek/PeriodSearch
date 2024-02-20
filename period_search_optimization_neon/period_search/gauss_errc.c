#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include "declarations.h"
#include <arm_neon.h>


int gauss_errc(double **a, int n, double b[])
{
	int *indxc, *indxr, *ipiv;
    int i, icol = 0, irow = 0, j, k, l, ll;
    double big, dum, pivinv, temp;

    indxc = vector_int(n + 1);
    indxr = vector_int(n + 1);
    ipiv = vector_int(n + 1);

    memset(ipiv, 0, n * sizeof(int));

    for (i = 1; i <= n; i++) {
        big = 0.0;
        for (j = 0; j < n; j++)
            if (ipiv[j] != 1) {
                for (k = 0; k < n; k++) {
                    if (ipiv[k] == 0) {
                        if (fabs(a[j][k]) >= big) {
                            big = fabs(a[j][k]);
                            irow = j;
                            icol = k;
                        }
                    }
                    else if (ipiv[k] > 1) {
                        deallocate_vector((void *)ipiv);
                        deallocate_vector((void *)indxc);
                        deallocate_vector((void *)indxr);
                        return(1);
                    }
                }
            }
        ++(ipiv[icol]);
        if (irow != icol) {
            for (l = 0; l < n; l++) SWAP(a[irow][l], a[icol][l])
                SWAP(b[irow], b[icol])
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if (a[icol][icol] == 0.0) {
            deallocate_vector((void *)ipiv);
            deallocate_vector((void *)indxc);
            deallocate_vector((void *)indxr);
            return(2);
        }


		pivinv=1.0/a[icol][icol];
		//__m128d avx_pivinv;
		//avx_pivinv=_mm_set1_pd(pivinv);
        float64x2_t avx_pivinv = vdupq_n_f64(pivinv);
		a[icol][icol] = 1.0;

        for (l = 0; l < (n - 1); l += 2) {
            float64x2_t avx_a1 = vld1q_f64(&a[icol][l]);
            avx_a1 = vmulq_f64(avx_a1, avx_pivinv);
            vst1q_f64(&a[icol][l], avx_a1);
        }
/*
		for (l=0;l<(n-1);l+=2)
		{
			__m128d avx_a1=_mm_load_pd(&a[icol][l]);
			avx_a1=_mm_mul_pd(avx_a1,avx_pivinv);
			_mm_store_pd(&a[icol][l],avx_a1);
		}
        */

		if (l==(n-1)) a[icol][l] *= pivinv; //last odd value


		//pivinv=1.0/a[icol][icol];
		//a[icol][icol]=1.0;
		//for (l=0;l<n;l++)
        //   a[icol][l] *= pivinv;

		b[icol] *= pivinv;
		for (ll=0;ll<n;ll++)
			if (ll != icol) {
				dum=a[ll][icol];
				a[ll][icol]=0.0;

				//__m128d avx_dum;
				//avx_dum=_mm_set1_pd(dum);
                float64x2_t avx_dum = vdupq_n_f64(dum);

                for (l = 0; l < (n - 1); l += 2) {
                    float64x2_t avx_a = vld1q_f64(&a[ll][l]);
                    float64x2_t avx_aa = vld1q_f64(&a[icol][l]);
                    float64x2_t avx_result = vmlsq_f64(avx_a, avx_aa, avx_dum);
                    vst1q_f64(&a[ll][l], avx_result);
                }
/*
				for (l=0;l<(n-1);l+=2)
				{
					__m128d avx_a=_mm_load_pd(&a[ll][l]);
                    __m128d avx_aa=_mm_load_pd(&a[icol][l]);
					avx_a=_mm_sub_pd(avx_a,_mm_mul_pd(avx_aa,avx_dum));
					_mm_store_pd(&a[ll][l],avx_a);
				}
                */
				if (l==(n-1)) a[ll][l] -= a[icol][l]*dum; //last odd value

				//for (l=0;l<n;l++) a[ll][l] -= a[icol][l]*dum;
				b[ll] -= b[icol]*dum;
			}
	}
	for (l=n;l>=1;l--) {
		if (indxr[l] != indxc[l])
			for (k=0;k<n;k++)
				SWAP(a[k][indxr[l]],a[k][indxc[l]]);
	}
    deallocate_vector((void *)ipiv);
    deallocate_vector((void *)indxc);
    deallocate_vector((void *)indxr);
	
	return(0);
}
#undef SWAP
/* from Numerical Recipes */
