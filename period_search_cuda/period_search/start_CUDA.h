#pragma once
int CUDAPrepare(int cudadev, double *beta_pole, double *lambda_pole, double *par, double cl,
		double Alamda_start, double Alamda_incr, double Alamda_incrr, double ee[][MAX_N_OBS+1],
		double ee0[][MAX_N_OBS+1], double *tim, double Phi_0, int checkex, int ndata);

int CUDAStart(int cudadev, int n_start_from,double freq_start,double freq_end,double freq_step,double stop_condition,int n_iter_min,double conw_r,
			  int ndata,int *ia,int *ia_par,double *cg_first,MFILE& mf,double escl,double *sig,int Numfac,double *brightness);

int CUDAPrecalc(int cudadev, double freq_start,double freq_end,double freq_step,double stop_condition,int n_iter_min,double *conw_r,
			  int ndata,int *ia,int *ia_par,int *new_conw,double *cg_first,double *sig,int Numfac,double *brightness);

int DoCheckpoint(MFILE& mf, int nlines, int newConw, double conwr);

void CUDAUnprepare(void);

void GetCUDAOccupancy(const int cudaDevice);

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