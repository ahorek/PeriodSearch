#pragma once

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

__device__ void mrqcof_curve1_lastI0(freq_context * __restrict__ CUDA_LCC,
													 double * __restrict__ a,
													 double * __restrict__ alpha,
													 double * __restrict__ beta,
													 int bid);

__device__ void MrqcofCurve23I0IA0(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid);
__device__ void MrqcofCurve23I0IA1(freq_context * __restrict__ CUDA_LCC, double * __restrict__ alpha, double * __restrict__ beta, int bid);

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