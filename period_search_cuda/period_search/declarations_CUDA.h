#pragma once

#define BLOCKX4 4
#define BLOCKX8 8
#define BLOCKX16 16
#define BLOCKX32 32

#define blockIdx() (blockIdx.x + gridDim.x * threadIdx.y) 

__device__ void curv(freq_context * __restrict__ CUDA_LCC,
		     double * __restrict__ cg,
		     int bid);
__device__ int mrqmin_1_end(freq_context * __restrict__ CUDA_LCC,
			    int ma, int mfit, int mfit1, int block);
__device__ void mrqmin_2_end(freq_context * __restrict__ CUDA_LCC, int ma, int bid);
__device__ void mrqcof_start(freq_context * __restrict__ CUDA_LCC,
			     double * __restrict__ a,
			     double * __restrict__ alpha,
			     double * __restrict__ beta,
			     int bid);
__device__ void mrqcof_matrix(freq_context * __restrict__ CUDA_LCC,
					      double * __restrict__ a,
					      int Lpoints, int bid);
__device__ void mrqcof_curve1(freq_context * __restrict__ CUDA_LCC,
					      double * __restrict__ a,
					      int Inrel, int Lpoints, int bid);

__device__ double mrqcof_end(freq_context * __restrict__ CUDA_LCC,
			     double * __restrict__ alpha);

__device__ void mrqcof_curve1_lastI1(
	      freq_context * __restrict__ CUDA_LCC,
	      double * __restrict__ a,
	      double * __restrict__ alpha,
	      double * __restrict__ beta,
	      int bid);

__device__ void MrqcofCurve23I1IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid);
__device__ void MrqcofCurve23I1IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid);
__device__ void MrqcofCurve23I0IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid);
__device__ void MrqcofCurve23I0IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid);
__device__ void MrqcofCurve2I0IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid);
__device__ void MrqcofCurve2I1IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid);
__device__ void MrqcofCurve2I0IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid);
__device__ void MrqcofCurve2I1IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int lpoints, int bid);

__device__ void mrqcof_curve1_lastI0(freq_context * __restrict__ CUDA_LCC,
													 double * __restrict__ a,
													 double * __restrict__ alpha,
													 double * __restrict__ beta,
													 int bid);

__device__ int gauss_errc(freq_context * __restrict__ CUDA_LCC, int ma);
__device__ void blmatrix(double bet, double lam, int tid);
__device__ double conv(freq_context * __restrict__ CUDA_LCC,
		       int nc,
		       double *dyda,
		       int bid);
__device__ double bright(freq_context * __restrict__ CUDA_LCC,
			 double * __restrict__ cg,
			 int jp, int Lpoints1, int Inrel);
__device__ void matrix_neo(freq_context * __restrict__ CUDA_LCC,
			   double const * __restrict__ cg,
			   int lnp1, int Lpoints, int bid);

__global__ void CudaCalculateIter1Mrqcof1Curve2I0IA0(void);
__global__ void CudaCalculateIter1Mrqcof1Curve2I0IA0(void);
__global__ void CudaCalculateIter1Mrqcof1Curve2I0IA1(void);
__global__ void CudaCalculateIter1Mrqcof1Curve2I1IA0(void);
__global__ void CudaCalculateIter1Mrqcof1Curve2I1IA1(void);
__global__ void CudaCalculateIter1Mrqcof2CurveM12I0IA1(const int lpoints);
__global__ void CudaCalculateIter1Mrqcof2CurveM12I0IA0(const int lpoints);
__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA0(void);
__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA1(void);
__global__ void CudaCalculateIter1Mrqcof2Curve2I1IA0(void);
__global__ void CudaCalculateIter1Mrqcof2Curve2I1IA1(void);
__global__ void CudaCalculateIter1Mrqcof1CurveM12I0IA0(const int lpoints);
__global__ void CudaCalculateIter1Mrqcof1CurveM12I0IA1(const int lpoints);
__global__ void CudaCalculateIter1Mrqcof1CurveM12I1IA0(const int lpoints);
__global__ void CudaCalculateIter1Mrqcof1CurveM12I1IA1(const int lpoints);
__global__ void CudaCalculateIter1Mrqcof1Curve1LastI0(void);
__global__ void CudaCalculateIter1Mrqcof1Curve1LastI1(void);
__global__ void CudaCalculateIter1Mrqcof2CurveM12I1IA1(const int lpoints);
__global__ void CudaCalculateIter1Mrqcof2CurveM12I1IA0(const int lpoints);
__global__ void CudaCalculateIter1Mrqcof2Curve1LastI0(void);
__global__ void CudaCalculateIter1Mrqcof2Curve1LastI1(void);

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