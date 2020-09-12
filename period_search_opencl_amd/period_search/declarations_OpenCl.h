#pragma once
#include "CL/cl.hpp"
#include <CL/cl_platform.h>
#include "globals.h"
#include "globals_OpenCl.h"


//void curv(freq_context* CUDA_LCC, double cg[], int brtmpl, int brtmph);
//extern int mrqmin_1_end(freq_context* CUDA_LCC, int ma, int mfit, int mfit1, int block);
 //void mrqmin_2_end(freq_context* CUDA_LCC, int ia[], int ma);
//void mrqcof_start(__global struct freq_context2* CUDA_LCC, varholder* Fa, double a[], double* alpha, double* beta);
 //void mrqcof_matrix(freq_context* CUDA_LCC, double a[], int Lpoints);
 //void mrqcof_curve1(freq_context* CUDA_LCC, double a[], double* alpha, double beta[], int Inrel, int Lpoints);
 //void mrqcof_curve1_last(freq_context* CUDA_LCC, double a[], double* alpha, double beta[], int Inrel, int Lpoints);
 //void MrqcofCurve2(freq_context* CUDA_LCC, double* alpha, double beta[], int inrel, int lpoints);
 //double mrqcof_end(freq_context* CUDA_LCC, double* alpha);

//double mrqcof(freq_context* CUDA_LCC, double a[], int ia[], int ma, double alpha[/*MAX_N_PAR+1*/][MAX_N_PAR + 1], double beta[], int mfit, int lastone, int lastma);
//__device__ int gauss_errc(freq_context *CUDA_LCC,int n, double b[]);
//extern int gauss_errc(freq_context* CUDA_LCC, int ma);
//void blmatrix(freq_context* CUDA_LCC, double bet, double lam);
//double conv(freq_context* CUDA_LCC, int nc, int tmpl, int tmph, int brtmpl, int brtmph);
//double bright(freq_context* CUDA_LCC, double cg[], int jp, int Lpoints1, int Inrel);
//void matrix_neo(freq_context* CUDA_LCC, double cg[], int lnp1, int Lpoints);
//void CudaCalculateIter1Mrqcof2Curve2(int inrel, int lpoints);
//void CudaCalculateIter1Mrqcof1Curve2(int inrel, int lpoints);





//int mrqmin_1_end(FreqContext* CUDA_LCC);
//void mrqmin_2_end(FreqContext* CUDA_LCC, int ia[], int ma);
//void mrqcof_start(FreqContext* CUDA_LCC, double a[], double* alpha, double beta[]);
//void mrqcof_matrix(FreqContext* CUDA_LCC, double a[], int Lpoints);
//void mrqcof_curve1(FreqContext* CUDA_LCC, double a[], double* alpha, double beta[], int Inrel, int Lpoints);
//void mrqcof_curve1_last(FreqContext* CUDA_LCC, double a[], double* alpha, double beta[], int Inrel, int Lpoints);
//void mrqcof_curve2(FreqContext* CUDA_LCC, double a[], double* alpha, double beta[], int Inrel, int Lpoints);
//double mrqcof_end(FreqContext* CUDA_LCC, double* alpha);
//double mrqcof(FreqContext* CUDA_LCC, double a[], int ia[], int ma, double alpha[/*MAX_N_PAR+1*/][MAX_N_PAR + 1], double beta[], int mfit, int lastone, int lastma);
//int gauss_errc(FreqContext* CUDA_LCC, int n, double b[]);
//void blmatrix(FreqContext* CUDA_LCC, double bet, double lam);
//double conv(FreqContext* CUDA_LCC, int nc, int tmpl, int tmph, int brtmpl, int brtmph);
//double bright(FreqContext* CUDA_LCC, double cg[], int jp, int Lpoints1, int Inrel);
//void matrix_neo(FreqContext* CUDA_LCC, double cg[], int lnp1, int Lpoints);
//void CUDACalculateIter1_mrqcof2_curve2(int Inrel, int Lpoints);
//void CUDACalculateIter1_mrqcof1_curve2(int Inrel, int Lpoints);

//__device__ int mrqmin_1_end(FreqContext* CUDA_LCC);
//__device__ void mrqmin_2_end(FreqContext* CUDA_LCC, int ia[], int ma);
//__device__ void mrqcof_start(FreqContext* CUDA_LCC, double a[],
//    double* alpha, double beta[]);
//__device__ void mrqcof_matrix(FreqContext* CUDA_LCC, double a[], int Lpoints);
//__device__ void mrqcof_curve1(FreqContext* CUDA_LCC, double a[],
//    double* alpha, double beta[], int Inrel, int Lpoints);
//__device__ void mrqcof_curve1_last(FreqContext* CUDA_LCC, double a[],
//    double* alpha, double beta[], int Inrel, int Lpoints);
//__device__ void mrqcof_curve2(FreqContext* CUDA_LCC, double a[],
//    double* alpha, double beta[], int Inrel, int Lpoints);
//__device__ double mrqcof_end(FreqContext* CUDA_LCC, double* alpha);
//
//__device__ double mrqcof(FreqContext* CUDA_LCC, double a[], int ia[], int ma,
//    double alpha[/*MAX_N_PAR+1*/][MAX_N_PAR + 1], double beta[], int mfit, int lastone, int lastma);
//__device__ int gauss_errc(FreqContext* CUDA_LCC, int n, double b[]);
//__device__ void blmatrix(FreqContext* CUDA_LCC, double bet, double lam);
//__device__ double conv(FreqContext* CUDA_LCC, int nc, int tmpl, int tmph, int brtmpl, int brtmph);
//__device__ double bright(FreqContext* CUDA_LCC, double cg[], int jp, int Lpoints1, int Inrel);
//__device__ void matrix_neo(FreqContext* CUDA_LCC, double cg[], int lnp1, int Lpoints);
//__global__ void CUDACalculateIter1_mrqcof2_curve2(int Inrel, int Lpoints);
//__global__ void CUDACalculateIter1_mrqcof1_curve2(int Inrel, int Lpoints);