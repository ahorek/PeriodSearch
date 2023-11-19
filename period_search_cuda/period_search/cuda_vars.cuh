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
