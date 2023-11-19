#pragma once

// vars
__device__ extern double Dblm[2][3][3][N_BLOCKS]; // OK, set by [tid], read by [bid]
__device__ extern double Blmat[3][3][N_BLOCKS];   // OK, set by [tid], read by [bid]

__device__ extern double CUDA_scale[N_BLOCKS][POINTS_MAX + 1];   // OK [bid][tid]
__device__ extern double ge[2][3][N_BLOCKS][POINTS_MAX + 1];     // OK [bid][tid]
__device__ extern double gde[2][3][3][N_BLOCKS][POINTS_MAX + 1]; // OK [bid][tid]
__device__ extern double jp_dphp[3][N_BLOCKS][POINTS_MAX + 1];   // OK [bid][tid]

__device__ extern double dave[N_BLOCKS][MAX_N_PAR + 1];
__device__ extern double atry[N_BLOCKS][MAX_N_PAR + 1];

__device__ extern double chck[N_BLOCKS];
__device__ extern int    isInvalid[N_BLOCKS];
__device__ extern int    isNiter[N_BLOCKS];
__device__ extern int    isAlamda[N_BLOCKS];
__device__ extern double Alamda[N_BLOCKS];
__device__ extern int    Niter[N_BLOCKS];
__device__ extern double iter_diffg[N_BLOCKS];
__device__ extern double rchisqg[N_BLOCKS]; // not needed
__device__ extern double dev_oldg[N_BLOCKS];
__device__ extern double dev_newg[N_BLOCKS];

__device__ extern double trial_chisqg[N_BLOCKS];
__device__ extern double aveg[N_BLOCKS];
__device__ extern int    npg[N_BLOCKS];
__device__ extern int    npg1[N_BLOCKS];
__device__ extern int    npg2[N_BLOCKS];

__device__ extern double Ochisq[N_BLOCKS];
__device__ extern double Chisq[N_BLOCKS];
__device__ extern double Areag[N_BLOCKS][MAX_N_FAC + 1];

//LFR
__managed__ extern int isReported[N_BLOCKS];
__managed__ extern double dark_best[N_BLOCKS];
__managed__ extern double per_best[N_BLOCKS];
__managed__ extern double dev_best[N_BLOCKS];
__managed__ extern double la_best[N_BLOCKS];
__managed__ extern double be_best[N_BLOCKS];

#ifdef NEWDYTEMP
__device__ extern double dytemp[POINTS_MAX + 1][40][N_BLOCKS];
#endif

// vars

__device__ double Dblm[2][3][3][N_BLOCKS]; // OK, set by [tid], read by [bid]
__device__ double Blmat[3][3][N_BLOCKS];   // OK, set by [tid], read by [bid]

__device__ double CUDA_scale[N_BLOCKS][POINTS_MAX + 1];   // OK [bid][tid]
__device__ double ge[2][3][N_BLOCKS][POINTS_MAX + 1];     // OK [bid][tid]
__device__ double gde[2][3][3][N_BLOCKS][POINTS_MAX + 1]; // OK [bid][tid]
__device__ double jp_dphp[3][N_BLOCKS][POINTS_MAX + 1];   // OK [bid][tid]

__device__ double dave[N_BLOCKS][MAX_N_PAR + 1];
__device__ double atry[N_BLOCKS][MAX_N_PAR + 1];

__device__ double chck[N_BLOCKS];
__device__ int    isInvalid[N_BLOCKS];
__device__ int    isNiter[N_BLOCKS];
__device__ int    isAlamda[N_BLOCKS];
__device__ double Alamda[N_BLOCKS];
__device__ int    Niter[N_BLOCKS];
__device__ double iter_diffg[N_BLOCKS];
__device__ double rchisqg[N_BLOCKS]; // not needed
__device__ double dev_oldg[N_BLOCKS];
__device__ double dev_newg[N_BLOCKS];

__device__ double trial_chisqg[N_BLOCKS];
__device__ double aveg[N_BLOCKS];
__device__ int    npg[N_BLOCKS];
__device__ int    npg1[N_BLOCKS];
__device__ int    npg2[N_BLOCKS];

__device__ double Ochisq[N_BLOCKS];
__device__ double Chisq[N_BLOCKS];
__device__ double Areag[N_BLOCKS][MAX_N_FAC + 1];

//LFR
__managed__ int isReported[N_BLOCKS];
__managed__ double dark_best[N_BLOCKS];
__managed__ double per_best[N_BLOCKS];
__managed__ double dev_best[N_BLOCKS];
__managed__ double la_best[N_BLOCKS];
__managed__ double be_best[N_BLOCKS];


#ifdef NEWDYTEMP
__device__ double dytemp[POINTS_MAX + 1][40][N_BLOCKS];
#endif

#define CUDA_Nphpar 3

//global to all freq
__constant__ int CUDA_Ncoef, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ int CUDA_ma, CUDA_mfit, CUDA_mfit1, CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__constant__ double CUDA_cg_first[MAX_N_PAR + 1];
__constant__ int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__constant__ double CUDA_iter_diff_max;
__constant__ double CUDA_conw_r;
__constant__ int CUDA_Lmax, CUDA_Mmax;
__constant__ double CUDA_lcl, CUDA_Alamda_start, CUDA_Alamda_incr;  //, CUDA_Alamda_incrr;
__constant__ double CUDA_Phi_0;
__constant__ double CUDA_beta_pole[N_POLES + 1];
__constant__ double CUDA_lambda_pole[N_POLES + 1];

__device__ double CUDA_par[4];
__device__ int CUDA_ia[MAX_N_PAR + 1];
__device__ double CUDA_Nor[3][MAX_N_FAC + 1];
__device__ double CUDA_Fc[MAX_LM+1][MAX_N_FAC + 1];
__device__ double CUDA_Fs[MAX_LM+1][MAX_N_FAC + 1];
__device__ double CUDA_Pleg[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
__device__ double CUDA_Darea[MAX_N_FAC + 1];
__device__ double CUDA_Dsph[MAX_N_PAR + 1][MAX_N_FAC + 1];
__device__ double CUDA_ee[3][MAX_N_OBS + 1]; //[3][MAX_N_OBS+1];
__device__ double CUDA_ee0[3][MAX_N_OBS+1];
__device__ double CUDA_tim[MAX_N_OBS + 1];
__device__ double *CUDA_brightness/*[MAX_N_OBS+1]*/;
__device__ double *CUDA_sig/*[MAX_N_OBS+1]*/;
__device__ double *CUDA_Weight/*[MAX_N_OBS+1]*/;
//__device__ double *CUDA_Area;
__device__ double *CUDA_Dg;
__device__ int CUDA_End;
__device__ int CUDA_Is_Precalc;

//global to one thread
__device__ freq_context *CUDA_CC;