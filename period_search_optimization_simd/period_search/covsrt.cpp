#define SWAP(a,b) {swap=(a);(a)=(b);(b)=swap;}

/**
 * @brief Rearranges the covariance matrix to its original form.
 *
 * This function rearranges the covariance matrix `covar` to its original form by swapping elements
 * based on the permutation indices stored in `ia`. This is typically used after a least-squares fit
 * to restore the covariance matrix to its original order.
 *
 * @param covar A pointer to a double array representing the covariance matrix.
 * @param ma The dimension of the covariance matrix (number of rows/columns).
 * @param ia An array of integers representing the permutation indices.
 * @param mfit The number of fitted parameters (number of rows/columns to be rearranged).
 *
 * @note The function modifies the `covar` matrix in place.
 *
 * @source Numerical Recipes
 *
 * @date 8.11.2006
 */
void covsrt(double **covar, int ma, int ia[], int mfit)
{
	int i,j,k;
	double swap;

	for (i=mfit+1;i<=ma;i++)
		for (j=1;j<=i;j++) covar[i][j]=covar[j][i]=0.0;
	k=mfit;
	for (j=ma;j>=1;j--) {
		if (ia[j]) {
			for (i=1;i<=ma;i++) SWAP(covar[i][k],covar[i][j])
			for (i=1;i<=ma;i++) SWAP(covar[k][i],covar[j][i])
			k--;
		}
	}
}
#undef SWAP
