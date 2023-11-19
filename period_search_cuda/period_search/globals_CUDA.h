#pragma once
//#define NEWDYTEMP
#include <cuda_runtime_api.h>

//  NOTE Fake declaration to satisfy intellisense. See https://stackoverflow.com/questions/39980645/enable-code-indexing-of-cuda-in-clion/39990500
#ifndef __CUDACC__
//#define __host__
//#define __device__
//#define __shared__
//#define __constant__
//#define __global__
//#define __host__
#include <device_functions.h>
#include <vector_types.h>
#include <driver_types.h>
#include <texture_types.h>
#include <cuda_texture_types.h>
//#define __CUDACC__
#define __CUDA__
inline void __syncthreads() {};
inline void atomicAdd(int*, int) {};

//template <class T>
//static __device__ T tex1Dfetch(texture<int2, 1> texObject, int x) { return {}; };

__device__ __device_builtin__ double __hiloint2double(int hi, int lo);

//template<class T, int texType = cudaTextureType1D, enum cudaTextureReadMode mode = cudaReadModeElementType>
//struct texture {};
//	int                          norm;
//	enum cudaTextureFilterMode   fMode;
//	enum cudaTextureAddressMode  aMode;
//	struct cudaChannelFormatDesc desc;
//};

//#include <__clang_cuda_builtin_vars.h>
//#include <__clang_cuda_intrinsics.h>
//#include <__clang_cuda_math_forward_declares.h>
//#include <__clang_cuda_complex_builtins.h>
//#include <../../../../../../2019/Professional/VC/Tools/Llvm/lib/clang/9.0.0/include/__clang_cuda_cmath.h>
#endif

//#ifdef __INTELLISENSE__
////#define __device__ \
////			__location__(device)
//#endif

#include "constants.h"
#include "cudamemasm.h"

//NOTE: https://devtalk.nvidia.com/default/topic/517801/-34-texture-is-not-a-template-34-error-mvs-2010/

#define N_BLOCKS 512

//global to all freq
__constant__ extern int CUDA_Ncoef, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ extern int CUDA_ma, CUDA_mfit, CUDA_mfit1, CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__constant__ extern double CUDA_cg_first[MAX_N_PAR + 1];
__constant__ extern int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__constant__ extern double CUDA_iter_diff_max;
__constant__ extern double CUDA_conw_r;
__constant__ extern int CUDA_Lmax, CUDA_Mmax;
__constant__ extern double CUDA_lcl, CUDA_Alamda_start, CUDA_Alamda_incr;
__constant__ extern double CUDA_Phi_0;
__constant__ extern double CUDA_beta_pole[N_POLES + 1];
__constant__ extern double CUDA_lambda_pole[N_POLES + 1];

__device__ extern double CUDA_par[4];
__device__ extern double CUDA_ee[3][MAX_N_OBS+1]; 
__device__ extern double CUDA_ee0[3][MAX_N_OBS+1]; 
__device__ extern double CUDA_tim[MAX_N_OBS + 1];
__device__ extern int CUDA_ia[MAX_N_PAR + 1];
__device__ extern double CUDA_Nor[3][MAX_N_FAC + 1];
__device__ extern double CUDA_Fc[MAX_LM + 1][MAX_N_FAC + 1];
__device__ extern double CUDA_Fs[MAX_LM + 1][MAX_N_FAC + 1];

__device__ extern double CUDA_Pleg[MAX_LM + 1][MAX_LM + 1][MAX_N_FAC + 1];
__device__ extern double CUDA_Darea[MAX_N_FAC + 1]; 
__device__ extern double CUDA_Dsph[MAX_N_PAR + 1][MAX_N_FAC + 1];
__device__ extern double *CUDA_brightness/*[MAX_N_OBS+1]*/;
__device__ extern double *CUDA_sig/*[MAX_N_OBS+1]*/;
__device__ extern double *CUDA_Weight/*[MAX_N_OBS+1]*/;
__device__ extern int CUDA_End;
__device__ extern int CUDA_Is_Precalc;


#ifdef NEWDYTEMP
__device__ extern double dytemp[POINTS_MAX + 1][40][N_BLOCKS];
#endif


//global to one thread
struct freq_context
{
  //	double Area[MAX_N_FAC+1];
  //double *Area;
  //	double Dg[(MAX_N_FAC+1)*(MAX_N_PAR+1)];
  double *Dg;
  //	double alpha[MAX_N_PAR+1][MAX_N_PAR+1];
  double *alpha;
  //	double covar[MAX_N_PAR+1][MAX_N_PAR+1];
  double *covar;
  //
#ifndef NEWDYTEMP
  double *dytemp;
#endif
  //	double ytemp[POINTS_MAX+1],
  double *ytemp;
  double cg[MAX_N_PAR + 1];
  double beta[MAX_N_PAR + 1];
  double da[MAX_N_PAR + 1];
};

//extern __device__ double *CUDA_Area;
extern __device__ double *CUDA_Dg;
//extern texture<int2, 1> texArea;
//extern texture<int2, 1> texDg;

__device__ extern freq_context *CUDA_CC;

/*
struct freq_result
{
	int isReported;
	double dark_best, per_best, dev_best, la_best, be_best;
};
*/

//__device__ extern freq_result *CUDA_FR;
//LFR
__managed__ extern int isReported[N_BLOCKS];
__managed__ extern double dark_best[N_BLOCKS];
__managed__ extern double per_best[N_BLOCKS];
__managed__ extern double dev_best[N_BLOCKS];
__managed__ extern double la_best[N_BLOCKS];
__managed__ extern double be_best[N_BLOCKS];


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

#define UNRL 4
#define blockIdx() (blockIdx.x + gridDim.x * threadIdx.y)

#define BLOCKX4 4
#define BLOCKX8 8
#define BLOCKX16 16
#define BLOCKX32 32