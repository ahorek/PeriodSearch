#include <cuda.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <math.h>

#include "constants.h"
#include "globals_CUDA.h"
#include "declarations_CUDA.h"
#include "cuda_vars.cuh"
#include <cstdio>

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

__global__ void CudaCalculatePrepare(int n_start, int n_max)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int n = n_start + tid;

  if(n > n_max)
    {
      isInvalid[tid] = 1;
      return;
    }
  else
    {
      isInvalid[tid] = 0;
    }

  per_best[tid] = 0; 
  dark_best[tid] = 0;
  la_best[tid] = 0;
  be_best[tid] = 0;
  dev_best[tid] = 1e40;
}

__global__
__launch_bounds__(1024)
  void CudaCalculatePreparePole(int m, double freq_start, double freq_step, int n_start)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  n_start += tid;
  auto CUDA_LCC = &CUDA_CC[tid];
  //auto CUDA_LFR = &CUDA_FR[tid];

  if(__ldg(&isInvalid[tid]))  
    {
      atomicAdd(&CUDA_End, 1);
      isReported[tid] = 0; //signal not to read result

      return;
    }

  //double period = ___drcp_rn(__ldg(&CUDA_freq[tid]));
  double period = ___drcp_rn(freq_start - (n_start - 1) * freq_step);
  double * __restrict__ cgp = CUDA_LCC->cg + 1;
  double const * __restrict__ cfp = CUDA_cg_first + 1;
  /* starts from the initial ellipsoid */
  int i;
  int ncoef = CUDA_Ncoef;
#pragma unroll 4
  for(i = 1; i <= ncoef - (UNRL - 1); i += UNRL)
    {
      double d[UNRL];
      int ii;
      for(ii = 0; ii < UNRL; ii++)
	d[ii] = *cfp++;
      for(ii = 0; ii < UNRL; ii++)
	*cgp++ = d[ii];
    }
#pragma unroll 3
  for( ; i <= ncoef; i++)
    {
      *cgp++ = *cfp++; //CUDA_cg_first[i];
    }

  
  /* The formulae use beta measured from the pole */
  /* conversion of lambda, beta to radians */
  *cgp++ = DEG2RAD * 90 - DEG2RAD * CUDA_beta_pole[m];
  *cgp++ = DEG2RAD * CUDA_lambda_pole[m];
   
  /* Use omega instead of period */
  *cgp++ = (24.0 * 2.0 * PI) / period;

#pragma unroll
  for(i = 1; i <= CUDA_Nphpar; i++)
    {
      *cgp++ = CUDA_par[i];
    }
  
  /* Use logarithmic formulation for Lambert to keep it positive */
  *cgp++ = CUDA_lcl; //log(CUDA_cl); 
  /* Lommel-Seeliger part */
  *cgp++ = 1;

  /* Levenberg-Marquardt loop */
  // moved to global iter_max,iter_min,iter_dif_max
  //
  rchisqg[tid] = -1;
  Alamda[tid] = -1;
  Niter[tid] = 0;
  iter_diffg[tid] = 1e40;
  dev_oldg[tid] = 1e30;
  dev_newg[tid] = 0;
  isReported[tid] = 0;
}

__global__ void CudaCalculateIter1Begin(int n_max)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(tid > n_max) return;

  if(__ldg(&isInvalid[tid])) 
    {
      return;
    }

  int niter = __ldg(&Niter[tid]);
  bool b_isniter = ((niter < CUDA_n_iter_max) && (iter_diffg[tid] > CUDA_iter_diff_max)) || (niter < CUDA_n_iter_min);
  isNiter[tid] = b_isniter;

  if(b_isniter)
    {
      if(__ldg(&Alamda[tid]) < 0)
	{
	  isAlamda[tid] = 1;
	  Alamda[tid] = CUDA_Alamda_start; /* initial alambda */
	}
      else
	isAlamda[tid] = 0;
    }
  else
    {
      if(!(__ldg(&isReported[tid])))
	{
	  atomicAdd(&CUDA_End, 1);
#ifdef _DEBUG
	  /*const int is_precalc = CUDA_Is_Precalc;
	    if(is_precalc)
	    {
	    printf("%d ", CUDA_End);
	    }*/
#endif
	  isReported[tid] = 1;
	}
    }
}

__global__
__launch_bounds__(768)
void CudaCalculateIter1Mrqmin1End(void)
{
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];

  if(__ldg(&isInvalid[bid])) return; //CUDA_LCC->isInvalid) return;

  if(!__ldg(&isNiter[bid])) return;

  /*gauss_err=*/
  mrqmin_1_end(CUDA_LCC, CUDA_ma, CUDA_mfit, CUDA_mfit1, CUDA_BLOCK_DIM);
}

__global__ void CudaCalculateIter1Mrqmin2End(void)
{
  //int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];
  if(__ldg(&isInvalid[bid])) return;

  if(!__ldg(&isNiter[bid])) return;

  mrqmin_2_end(CUDA_LCC, CUDA_ma, bid);

  __syncwarp();
  if(threadIdx.x == 0)
    Niter[bid]++;
  //CUDA_LCC->Niter++;
}

__global__
__launch_bounds__(512)
void CudaCalculateIter1Mrqcof1Start(void)
{
  int tid = blockIdx() * blockDim.x + threadIdx.x;

  if(tid < blockDim.y * gridDim.x)
    {
      auto CUDA_LCC = &CUDA_CC[tid];
 
      double *a = CUDA_LCC->cg;
      blmatrix(a[CUDA_ma-4-CUDA_Nphpar], a[CUDA_ma-3-CUDA_Nphpar], tid);
    }

  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];
  
  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  if(!__ldg(&isAlamda[bid])) return;

  if(threadIdx.x == 0)
    {
      trial_chisqg[bid] = 0;
      npg[bid] = 0;
      npg1[bid] = 0;
      npg2[bid] = 0;
      aveg[bid] = 0;
    }

  mrqcof_start(CUDA_LCC, CUDA_LCC->cg, CUDA_LCC->alpha, CUDA_LCC->beta, bid);
}

__global__ void CudaCalculateIter1Mrqcof1End(void)
{
  int tid = blockIdx.x * blockDim.y + threadIdx.y;
  auto CUDA_LCC = &CUDA_CC[tid];

  if(__ldg(&isInvalid[tid])) return;
  if(!__ldg(&isNiter[tid])) return;
  if(!__ldg(&isAlamda[tid])) return;

  mrqcof_end(CUDA_LCC, CUDA_LCC->alpha);
  Ochisq[tid] = trial_chisqg[tid];
}

__global__
__launch_bounds__(768)
void CudaCalculateIter1Mrqcof2Start(void)
{
  int tid = blockIdx() * blockDim.x + threadIdx.x;

  if(tid < blockDim.y * gridDim.x)
    {
      //auto CUDA_LCC = &CUDA_CC[tid];
 
      double *a = atry[tid]; //CUDA_LCC->atry;
      blmatrix(a[CUDA_ma - CUDA_Nphpar - 4], a[CUDA_ma - CUDA_Nphpar - 3], tid);
    }

  int bid = blockIdx();
  auto CUDA_LCC = &CUDA_CC[bid];
  
  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;

  if(threadIdx.x == 0)
    {
      trial_chisqg[bid] = 0;
      npg[bid] = 0;
      npg1[bid] = 0;
      npg2[bid] = 0;
      aveg[bid] = 0;
    }
  
  mrqcof_start(CUDA_LCC, atry[bid], CUDA_LCC->covar, CUDA_LCC->da, bid);
}

__global__ void CudaCalculateIter1Mrqcof2End(void)
{
  int tid = blockIdx.x * blockDim.y + threadIdx.y;
  auto CUDA_LCC = &CUDA_CC[tid];

  if(__ldg(&isInvalid[tid])) return;
  if(!__ldg(&isNiter[tid])) return;

  mrqcof_end(CUDA_LCC, CUDA_LCC->covar);
  Chisq[tid] = __ldg(&trial_chisqg[tid]);
}

__global__
__launch_bounds__(768) //768
void CudaCalculateIter2(void)
{
  //bool beenThere = false;
  int bid = blockIdx();
  if(__ldg(&isInvalid[bid])) return;
  if(!__ldg(&isNiter[bid])) return;
  
  int nf = CUDA_Numfac;
  auto CUDA_LCC = &CUDA_CC[bid];

  double chisq = __ldg(&Chisq[bid]);
  
  if(Niter[bid] == 1 || chisq < Ochisq[bid])
    {
      curv(CUDA_LCC, CUDA_LCC->cg, bid);
      
      double a[3] = {0, 0, 0};

      int j = threadIdx.x + 1;

      double const * __restrict__ areap = Areag[bid];
#pragma unroll 9
      while(j <= nf)
	{
	  double dd = areap[j];
#pragma unroll 3
	  for(int i = 0; i < 3; i++)
	    {
	      double const * __restrict__ norp = CUDA_Nor[i];
	      a[i] += dd * norp[j];
	    }
	  j += CUDA_BLOCK_DIM;
	}
      
#pragma unroll
      for(int off = CUDA_BLOCK_DIM/2; off > 0; off >>= 1)
	{
	  double b[3];
#pragma unroll 3
	  for(int i = 0; i < 3; i++)
	    b[i] = __shfl_down_sync(0xffffffff, a[i], off);
#pragma unroll 3
	  for(int i = 0; i < 3; i++)
	    a[i] += b[i];
	}
      
      //__syncwarp();
      if(threadIdx.x == 0)
	{
	  double conwr2 = CUDA_conw_r, aa = 0;
	  
	  Ochisq[bid] = chisq;
	  conwr2 *= conwr2;

#pragma unroll 3
	  for(int i = 0; i < 3; i++)
	    {
	      aa += a[i]*a[i];
	    }
	  
	  double rchisq = chisq - aa * conwr2; //(CUDA_conw_r * CUDA_conw_r);
	  double dev_old = dev_oldg[bid];
	  double dev_new = __dsqrt_rn(rchisq / (CUDA_ndata - 3));
	  chck[bid] = norm3d(a[0], a[1], a[2]);

	  dev_newg[bid]  = dev_new;
	  double diff    = dev_old - dev_new;
	  
	  /* 
	  // only if this step is better than the previous,
	  // 1e-10 is for numeric errors 
	  */
	  
	  if(diff > 1e-10)
	    {
	      iter_diffg[bid] = diff; 
	      dev_oldg[bid] = dev_new; 
	    }
	}
    }
}

__global__ void CudaCalculateFinishPole(void)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto CUDA_LCC = &CUDA_CC[tid];
  //auto CUDA_LFR = &CUDA_FR[tid];

  if(__ldg(&isInvalid[tid])) return;
  
  double dn = __ldg(&dev_newg[tid]);
  int nf = CUDA_Numfac;
  
  if(dn >= __ldg(&dev_best[tid]))
    return;

  double dark = __ldg(&chck[tid]); 

  register double tot = 0, tot2 = 0;
  double const * __restrict__ p = &(Areag[tid][1]); //??????????????????
#pragma unroll 4
  for(int i = 0; i < nf - 1; i++)
    {
      tot  += __ldca(p++);
      i++;
      tot2 += __ldca(p++);
    }
  if(nf & 1)
    tot += __ldca(p); //LDG_d_ca(CUDA_LCC->Area, (nf - 1));
  //tot += CUDA_LCC->Area[nf - 1];
  
  tot = __drcp_rn(tot + tot2);
  
  /* period solution */
  double period = 2.0 * PI * __drcp_rn(CUDA_LCC->cg[CUDA_Ncoef + 3]);

  /* pole solution */
  double la_tmp = RAD2DEG * CUDA_LCC->cg[CUDA_Ncoef + 2];
  double be_tmp = 90.0 - RAD2DEG * CUDA_LCC->cg[CUDA_Ncoef + 1];

  dev_best[tid] = dn;
  dark_best[tid] = dark * 100.0 * tot;
  per_best[tid] = period;
  la_best[tid] = la_tmp;
  be_best[tid] = be_tmp;
}

__global__ void CudaCalculateFinish(void)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  //  auto CUDA_LCC = &CUDA_CC[tid];
  //auto CUDA_LFR = &CUDA_FR[tid];

  if(__ldg(&isInvalid[tid])) return;

  double lla_best = la_best[tid];
  if(lla_best < 0)
    la_best[tid] = lla_best + 360;

  if(isnan(__ldg(&dark_best[tid])) == 1)
    dark_best[tid] = 1.0;
}
