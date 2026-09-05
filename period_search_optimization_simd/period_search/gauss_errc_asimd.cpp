#define SWAP(a,b) {temp=(a);(a)=(b);(b)=temp;}

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string.h>
#include "declarations.h"
#include "CalcStrategyAsimd.hpp"

#if defined __GNUG__ && !defined __clang__
__attribute__((__target__("arch=armv8-a+simd")))
#elif defined __GNUG__ && __clang__
// NOTE: The following generates warning: unsupported architecture 'armv8-a+simd' in the 'target' attribute string; 'target' attribute ignored [-Wignored-attributes]
// __attribute__((target("arch=armv8-a+simd")))
#endif

/**
* @brief Solves a linear system of equations using Gaussian elimination with partial pivoting.
*
* This function implements the Gaussian elimination algorithm with partial pivoting to solve a
* linear system of equations. It rearranges the covariance matrix and the right-hand side vector
* to find the solution.
*
* @param gl A reference to a globals structure containing the covariance matrix and other global data.
* @param n The dimension of the system (number of equations/variables).
* @param b A vector of doubles representing the right-hand side vector of the system.
* @param error An integer reference to store error codes:
*              - 0: No error
*              - 1: Singular matrix
*              - 2: Zero pivot element
*
* @note The function modifies the covariance matrix `covar` in place.
*
* @source Numerical Recipes
*
* @date 8.11.2006
*/
void CalcStrategyAsimd::gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int &error)
{
	//int *indxc,  *indxr, *ipiv;
	int i, icol = 0, irow = 0, j, k, l, ll;
	double big, dum, pivinv, temp;

	auto& a = gl.covar;

	//indxc = vector_int(n + 1);
	std::vector<int> indxc(n + 1 + 1, 0);
	//indxr = vector_int(n + 1);
	std::vector<int> indxr(n + 1 + 1, 0);
	//ipiv = vector_int(n + 1);
	//memset(ipiv, 0, n * sizeof(int));
	std::vector<int> ipiv(n + 1 + 1, 0);

	/* Pivot search. The ipiv state changes only once per pivot step, so the column
	   bookkeeping is hoisted out of the row loop (O(n) per step instead of O(n*n))
	   and the inner loop is left branch-free: a per-lane running maximum plus the
	   index that produced it, reduced once per step. */
	const int P = (n + 1) & ~1;					// columns rounded up to a whole vector
	std::vector<uint64_t> colmask(P, 0);		// all-ones = column still free, 0 = taken
	double max_val[2], max_idx[2];
	double best;

	const float64x2_t avx_reject = vdupq_n_f64(-1.0);	// |a| >= 0, a negative never wins
	const float64x2_t avx_step = vdupq_n_f64(2.0);
	const double lane_init[2] = { 0.0, 1.0 };
	const float64x2_t avx_lane = vld1q_f64(lane_init);

	for (i = 1; i <= n; i++)
	{
		for (k = 0; k < n; k++)
		{
			if (ipiv[k] > 1)
			{
				//deallocate_vector((void*)indxc);
				//deallocate_vector((void*)indxr);
				//deallocate_vector((void*)ipiv);
				error = 1;
				return;
			}

			colmask[k] = ipiv[k] ? 0 : ~UINT64_C(0);
		}

		for (k = n; k < P; k++)
			colmask[k] = 0;						// padding lane can never win

		float64x2_t avx_big = vdupq_n_f64(0.0);		// per-lane running 'big', starts at 0.0
		float64x2_t avx_idx = vdupq_n_f64(-1.0);	// per-lane winning index j*P+k, -1 = none

		for (j = 0; j < n; j++)
		{
			if (ipiv[j] == 1) continue;

			float64x2_t avx_cur = vaddq_f64(vdupq_n_f64(static_cast<double>(j) * P), avx_lane);

			for (k = 0; k < P; k += 2)
			{
				float64x2_t avx_abs = vabsq_f64(vld1q_f64(&a[j][k]));
				avx_abs = vbslq_f64(vld1q_u64(&colmask[k]), avx_abs, avx_reject);

				uint64x2_t avx_ge = vcgeq_f64(avx_abs, avx_big);
				avx_big = vbslq_f64(avx_ge, avx_abs, avx_big);
				avx_idx = vbslq_f64(avx_ge, avx_cur, avx_idx);
				avx_cur = vaddq_f64(avx_cur, avx_step);
			}
		}

		/* Largest value wins; ties go to the largest index, which reproduces the
		   scalar '>=' ("the last one seen wins") exactly. */
		vst1q_f64(max_val, avx_big);
		vst1q_f64(max_idx, avx_idx);
		big = max_val[0];
		best = max_idx[0];

		for (l = 1; l < 2; l++)
		{
			if (max_val[l] > big || (max_val[l] == big && max_idx[l] > best))
			{
				big = max_val[l];
				best = max_idx[l];
			}
		}

		if (best >= 0.0)
		{
			const int idx = static_cast<int>(best);
			irow = idx / P;
			icol = idx - irow * P;
		}

		++(ipiv[icol]);
		if (irow != icol)
		{
			for (l = 0; l < n; l++) SWAP(a[irow][l], a[icol][l])
				SWAP(b[irow], b[icol])
		}

		indxr[i] = irow;
		indxc[i] = icol;

		if (a[icol][icol] == 0.0)
		{
			//deallocate_vector((void*)indxc);
			//deallocate_vector((void*)indxr);
			//deallocate_vector((void*)ipiv);
			error = 2;
			return;
		}

		pivinv = 1.0 / a[icol][icol];
		float64x2_t avx_pivinv = vdupq_n_f64(pivinv);
		a[icol][icol] = 1.0;

        for (l = 0; l < (n - 1); l += 2) {
            float64x2_t avx_a1 = vld1q_f64(&a[icol][l]);
            avx_a1 = vmulq_f64(avx_a1, avx_pivinv);
            vst1q_f64(&a[icol][l], avx_a1);
        }

		if (l==(n-1)) a[icol][l] *= pivinv; //last odd value

		b[icol] *= pivinv;
		for (ll = 0; ll < n; ll++)
		{
			if (ll != icol)
			{
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				float64x2_t avx_dum = vdupq_n_f64(dum);

                for (l = 0; l < (n - 1); l += 2) {
                    float64x2_t avx_a = vld1q_f64(&a[ll][l]);
                    float64x2_t avx_aa = vld1q_f64(&a[icol][l]);
                    float64x2_t avx_result = vmlsq_f64(avx_a, avx_aa, avx_dum);
                    vst1q_f64(&a[ll][l], avx_result);
                }

				if (l == (n - 1)) a[ll][l] -= a[icol][l] * dum; //last odd value

				b[ll] -= b[icol] * dum;
			}
		}
	}

	for (l = n; l >= 1; l--)
	{
		if (indxr[l] != indxc[l])
			for (k = 0; k < n; k++)
			{
				SWAP(a[k][indxr[l]], a[k][indxc[l]]);
			}
	}

	//deallocate_vector((void*)indxc);
	//deallocate_vector((void*)indxr);
	//deallocate_vector((void*)ipiv);
	error = 0;

	return;
}
#undef SWAP
