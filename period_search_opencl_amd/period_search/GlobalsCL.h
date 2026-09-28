#pragma OPENCL FP_CONTRACT ON

/* cl_khr_fp64 is enabled in Real.cl (FP64 build only) */
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

//struct __attribute__((packed)) freq_context
//struct mfreq_context
//struct __attribute__((aligned(8))) mfreq_context
typedef struct mfreq_context
{
	//double* Area;
	//double* Dg;
	//double* alpha;
	//double* covar;
	//double* dytemp;
	//double* ytemp;

	real Area[MAX_N_FAC + 1];
	/* The point- and fit-dimensioned work arrays (alpha, covar, dytemp,
	   ytemp, jp_*, e_*, de, de0) live in a separate runtime-sized scratch
	   buffer - one slice of freq_context.scrStride doubles per work-group,
	   at the offsets recorded in freq_context - instead of compile-time
	   worst-case arrays here. That cuts per-context memory ~6x (2.27 MB ->
	   ~0.4 MB for typical workunits). */
	real beta[MAX_N_PAR + 1];
	real atry[MAX_N_PAR + 1];
	real da[MAX_N_PAR + 1];
	real cg[MAX_N_PAR + 1];
	real Blmat[4][4];
	real Dblm[3][4][4];
	real dave[MAX_N_PAR + 1];
	real dyda[MAX_N_PAR + 1];

	real sh_big[BLOCK_DIM];
	real chck[4];
	real pivinv;
	real ave;
	real freq;
	real Alamda;
	real Chisq;
	real Ochisq;
	real rchisq;
	real trial_chisq;
	real iter_diff, dev_old, dev_new;

	int Niter;
	int np, np1, np2;
	int isInvalid, isAlamda, isNiter;
	int icol;
	//double conw_r;

	int ipiv[MAX_N_PAR + 1];
	int indxc[MAX_N_PAR + 1];
	int indxr[MAX_N_PAR + 1];
	int sh_icol[BLOCK_DIM];
	int sh_irow[BLOCK_DIM];
} CUDA_LCC;

//struct freq_context
//typedef struct __attribute__((aligned(8))) freq_context
struct freq_context
{
	real Phi_0;
	real logCl;
	real cl;
	//double logC;
	real lambda_pole[N_POLES + 1];
	real beta_pole[N_POLES + 1];


	real par[4];
	real Alamda_start;
	real Alamda_incr;

	//double cgFirst[MAX_N_PAR + 1];
	real tim[MAX_N_OBS + 1];
	real ee[MAX_N_OBS + 1][3];	// double* ee;
	real ee0[MAX_N_OBS + 1][3];	// double* ee0;
	real Sig[MAX_N_OBS + 1];
	real Weight[MAX_N_OBS + 1];
	real Brightness[MAX_N_OBS + 1];
	real Fc[MAX_N_FAC + 1][MAX_LM + 1];
	real Fs[MAX_N_FAC + 1][MAX_LM + 1];
	real Darea[MAX_N_FAC + 1];
	real Nor[MAX_N_FAC + 1][3];
	real Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
	real Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
	real conw_r;

	int ia[MAX_N_PAR + 1];

	int Dg_block;
	int lastone;
	int lastma;
	int ma;
	int Mfit, Mfit1;
	int Mmax, Lmax;
	int n;
	int Ncoef, Ncoef0;
	int Numfac;
	int Numfac1;
	int Nphpar;
	int ndata;
	int Is_Precalc;

	/* runtime dimensions + per-context offsets (in doubles) into the
	   scratch buffer that replaced the fixed-size work arrays */
	int lcPoints1;
	int scrStride;
	int offAlpha;
	int offCovar;
	int offDytemp;
	int offYtemp;
	int offJpScale;
	int offJpDphp1;
	int offJpDphp2;
	int offJpDphp3;
	int offE1;
	int offE2;
	int offE3;
	int offE01;
	int offE02;
	int offE03;
	int offDe;
	int offDe0;
};

//struct freq_result
//struct __attribute__((aligned(8))) freq_result
struct freq_result
{
	real dark_best, per_best, dev_best, dev_best_x2, la_best, be_best, freq;
	int isReported, isInvalid, isNiter;
};
