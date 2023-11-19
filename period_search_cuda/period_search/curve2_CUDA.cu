//#ifndef __CUDACC__
//#define __CUDACC__
//#endif

#include <stdio.h>
#include <stdlib.h>
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
//#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void
__launch_bounds__(768) 
CudaCalculateIter1Mrqcof1Curve2I0IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I0IA0(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}


__global__ void
__launch_bounds__(768) 
CudaCalculateIter1Mrqcof1Curve2I0IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I0IA1(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}


__global__ void
__launch_bounds__(768) 
CudaCalculateIter1Mrqcof1Curve2I1IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I1IA0(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}

__global__ void
__launch_bounds__(768) 
CudaCalculateIter1Mrqcof1Curve2I1IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I1IA1(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}

__global__ 
void CudaCalculateIter1Mrqcof2CurveM12I0IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  double *atryp = atry[bid]; //CUDA_LCC->atry;
  mrqcof_matrix(CUDA_LCC, atryp, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, atryp, 0, lpoints, bid);
  MrqcofCurve2I0IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

__global__ 
void CudaCalculateIter1Mrqcof2CurveM12I0IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  double *atryp = atry[bid]; //CUDA_LCC->atry;
  mrqcof_matrix(CUDA_LCC, atryp, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, atryp, 0, lpoints, bid);
  MrqcofCurve2I0IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  MrqcofCurve23I0IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}

__global__ void CudaCalculateIter1Mrqcof2Curve2I0IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  MrqcofCurve23I0IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}


// SLOW
__global__ void CudaCalculateIter1Mrqcof2Curve2I1IA0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  MrqcofCurve23I1IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}



__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof2Curve2I1IA1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  MrqcofCurve23I1IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, bid);
}



__global__ 
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1CurveM12I0IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  double *cg = CUDA_LCC->cg;
  mrqcof_matrix(CUDA_LCC, cg, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, cg, 0, lpoints, bid);
  MrqcofCurve2I0IA0(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, lpoints, bid);
}


__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1CurveM12I0IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  double *cg = CUDA_LCC->cg;
  mrqcof_matrix(CUDA_LCC, cg, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, cg, 0, lpoints, bid);
  MrqcofCurve2I0IA1(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, lpoints, bid);
}



__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1CurveM12I1IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  double *cg = CUDA_LCC->cg;
  mrqcof_matrix(CUDA_LCC, cg, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, cg, 1, lpoints, bid);
  MrqcofCurve2I1IA0(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, lpoints, bid);
}


__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1CurveM12I1IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  double *cg = CUDA_LCC->cg;
  mrqcof_matrix(CUDA_LCC, cg, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, cg, 1, lpoints, bid);
  MrqcofCurve2I1IA1(CUDA_LCC, CUDA_LCC->alpha, CUDA_LCC->beta, lpoints, bid);
}


__global__ 
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1Curve1LastI0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  if(CUDA_LCC->ytemp == NULL) return;

  mrqcof_curve1_lastI0(CUDA_LCC, CUDA_LCC->cg, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}


__global__
__launch_bounds__(512) 
void CudaCalculateIter1Mrqcof1Curve1LastI1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  mrqcof_curve1_lastI1(CUDA_LCC, CUDA_LCC->cg, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}

__global__ 
void CudaCalculateIter1Mrqcof2CurveM12I1IA1(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  double *atryp = atry[bid]; //CUDA_LCC->atry;
  mrqcof_matrix(CUDA_LCC, atryp, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, atryp, 1, lpoints, bid);
  MrqcofCurve2I1IA1(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

__global__ 
__launch_bounds__(384) 
void CudaCalculateIter1Mrqcof2CurveM12I1IA0(const int lpoints)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  double *atryp = atry[bid]; //CUDA_LCC->atry;
  mrqcof_matrix(CUDA_LCC, atryp, lpoints, bid);
  mrqcof_curve1(CUDA_LCC, atryp, 1, lpoints, bid);
  MrqcofCurve2I1IA0(CUDA_LCC, CUDA_LCC->covar, CUDA_LCC->da, lpoints, bid);
}

__global__
__launch_bounds__(768) 
void CudaCalculateIter1Mrqcof2Curve1LastI0(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  mrqcof_curve1_lastI0(CUDA_LCC, atry[bid], CUDA_LCC->covar, CUDA_LCC->da, bid);
}

__global__
__launch_bounds__(1024) 
void CudaCalculateIter1Mrqcof2Curve1LastI1(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  mrqcof_curve1_lastI1(CUDA_LCC, atry[bid], CUDA_LCC->covar, CUDA_LCC->da, bid);
}