#pragma once

__global__ void CudaCalculatePrepare(int n_start, int n_max);

__global__ void CudaCalculatePreparePole(int m, double freq_start, double freq_step, int n);

__global__ void CudaCalculateIter1Begin(int n_max);

__global__ void CudaCalculateIter1Mrqmin1End(void);

__global__ void CudaCalculateIter1Mrqmin2End(void);

__global__ void CudaCalculateIter1Mrqcof1Start(void);

__global__ void CudaCalculateIter1Mrqcof1Matrix(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I0IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I0IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve2I0IA0(void);

__global__ void CudaCalculateIter1Mrqcof1Curve2I0IA1(void);

__global__ void CudaCalculateIter1Mrqcof1Curve2I1IA0(void);

__global__ void CudaCalculateIter1Mrqcof1Curve2I1IA1(void);

__global__ void CudaCalculateIter1Mrqcof1CurveM1(int inrel, int lpoints);

__global__ void CudaCalculateIter1Mrqcof1CurveM12I0IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1CurveM12I0IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1CurveM12I1IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1CurveM12I1IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1LastI0(void);

__global__ void CudaCalculateIter1Mrqcof1Curve1LastI1(void);

__global__ void CudaCalculateIter1Mrqcof1End(void);

__global__ void CudaCalculateIter1Mrqcof2Start(void);

__global__ void CudaCalculateIter1Mrqcof2Matrix(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I0IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I0IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I1IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof1Curve1I1IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve1I0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve1I1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA0(void);

__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA1(void);

__global__ void CudaCalculateIter1Mrqcof2Curve2I1IA0(void);

__global__ void CudaCalculateIter1Mrqcof2Curve2I1IA1(void);

__global__ void CudaCalculateIter1Mrqcof2CurveM1(int inrel, int lpoints);

__global__ void CudaCalculateIter1Mrqcof2CurveM12I0IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2CurveM12I0IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2CurveM12I1IA0(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2CurveM12I1IA1(int lpoints);

__global__ void CudaCalculateIter1Mrqcof2Curve1LastI1(void);

__global__ void CudaCalculateIter1Mrqcof2Curve1LastI0(void);

__global__ void CudaCalculateIter1Mrqcof2End(void);

__global__ void CudaCalculateIter2(void);

__global__ void CudaCalculateFinishPole(void);

__global__ void CudaCalculateFinish(void);


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