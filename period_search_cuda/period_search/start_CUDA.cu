//#ifndef NVML_NO_UNVERSIONED_FUNC_DEFS
//#define NVML_NO_UNVERSIONED_FUNC_DEFS
//#endif

//#define NEWDYTEMP

int msleep(long ms)
{
  struct timespec ts;
  int ret;
  
  if(ms < 0)
    {
      return -1;
    }
  
  ts.tv_sec = ms / 1000;
  ts.tv_nsec = (ms % 1000) * 1000000L;
  
  while(0 != (ret = nanosleep(&ts, &ts)));
  //nop
  
  return ret;
}

#include <cuda.h>
#include <cstdio>
#include "mfile.h"
#include "globals.h"
#include "globals_CUDA.h"
#include "start_CUDA.h"
#include "declarations_CUDA.h"
#include "boinc_api.h"
#include "Start.cuh"
//#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
//#include <cuda_occupancy.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
//#include <nvml.h>

#include <sys/time.h>
#include <sys/resource.h>

#ifdef __GNUC__
#include <time.h>
#endif
#include "ComputeCapability.h"

/*
int sched_yield(void) __THROW
{
  usleep(0);
}
*/

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

int CUDA_grid_dim;
extern int Nfactor; // default 1, usage: --N number  where number is 2 - 16
cudaStream_t stream1;
cudaStream_t stream2;
cudaEvent_t event1, event2;

double *pee, *pee0, *pWeight;
//bool nvml_enabled = false;

//bool if_freq_measured = false;

//void GetPeakClock(const int cudadev)
//{
//	unsigned int currentSmClock;
//	unsigned int currentMemoryClock;
//	const unsigned int devId = cudadev;
//	nvmlDevice_t nvmlDevice;
//	nvmlDeviceGetHandleByIndex(devId, &nvmlDevice);
//	nvmlDeviceGetClock(nvmlDevice, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT, &currentSmClock);
//	nvmlDeviceGetClock(nvmlDevice, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &currentMemoryClock);
//	currentMemoryClock /= 2;
//	cudaDeviceProp deviceProp;
//	cudaGetDeviceProperties(&deviceProp, cudadev);
//	const auto deviceClock = deviceProp.clockRate / 1000;
//	const auto memoryClock = deviceProp.memoryClockRate / 1000 /2;
//	fprintf(stderr, "CUDA Device SM clock [base|current]: %u MHz | %u MHz\n", deviceClock, currentSmClock);
//	fprintf(stderr, "CUDA Device Memory clock [base|current]: %u MHz | %u MHz\n", memoryClock, currentMemoryClock);
//
//	if_freq_measured = true;
//}

// NOTE: https://boinc.berkeley.edu/trac/wiki/CudaApps
bool SetCUDABlockingSync(const int device)
{
  CUdevice  hcuDevice;
  CUcontext hcuContext;
  
  CUresult status = cuInit(0);
  if (status != CUDA_SUCCESS)
    return false;
  
  status = cuDeviceGet(&hcuDevice, device);
  if (status != CUDA_SUCCESS)
    return false;
  
  // 0x4
  status = cuCtxCreate(&hcuContext, CU_CTX_SCHED_BLOCKING_SYNC, hcuDevice);
  //status = cuCtxCreate(&hcuContext, CU_CTX_SCHED_YIELD, hcuDevice);
  if (status != CUDA_SUCCESS)
    return false;
  
  return true;
}


int *theEnd = NULL;


int CUDAPrepare(int cudadev, double* beta_pole, double* lambda_pole, double* par, double cl,
		double Alamda_start, double Alamda_incr, double Alamda_incrr,
		double ee[][MAX_N_OBS + 1], double ee0[][MAX_N_OBS + 1], double* tim, double Phi_0, int checkex, int ndata)
{
  //init gpu
  auto initResult = SetCUDABlockingSync(cudadev);
  if (!initResult)
    {
      fprintf(stderr, "CUDA: Error while initialising CUDA\n");
      exit(999);
    }

  cudaSetDevice(cudadev);
  // TODO: Check if this is obsolete when calling SetCUDABlockingSync()
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync|cudaDeviceLmemResizeToMax);
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  // TODO: Check if this will help to free some CPU core utilization
  //cudaSetDeviceFlags(cudaDeviceScheduleYield);

  /*
  try
    {
      nvmlInit();
      nvml_enabled = true;
    }
  catch (...)
    {
      nvml_enabled = false;
    }
  */

  //determine gridDim
  cudaDeviceProp deviceProp;

  cudaGetDeviceProperties(&deviceProp, cudadev);
  if (!checkex)
    {
      auto cudaVersion = CUDA_VERSION;
      auto totalGlobalMemory = deviceProp.totalGlobalMem / 1048576;
      auto sharedMemorySm = deviceProp.sharedMemPerMultiprocessor;
      auto sharedMemoryBlock = deviceProp.sharedMemPerBlock;
      /*
      char drv_version_str[NVML_DEVICE_PART_NUMBER_BUFFER_SIZE + 1];
      if (nvml_enabled) 
	{
	  auto retval = nvmlSystemGetDriverVersion(drv_version_str,
						   NVML_DEVICE_PART_NUMBER_BUFFER_SIZE);
	  if (retval != NVML_SUCCESS) {
	    fprintf(stderr, "%s\n", nvmlErrorString(retval));
	    return 1;
	  }
	}
      */

      /*auto peakClk = 1;
	cudaDeviceGetAttribute(&peakClk, cudaDevAttrClockRate, cudadev);
	auto devicePeakClock = peakClk / 1024;*/

      fprintf(stderr, "CUDA version: %d\n", cudaVersion);
      fprintf(stderr, "CUDA Device number: %d\n", cudadev);
      fprintf(stderr, "CUDA Device: %s %luMB\n", deviceProp.name, totalGlobalMemory);
      //fprintf(stderr, "CUDA Device driver: %s\n", drv_version_str);
      fprintf(stderr, "Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
      //fprintf(stderr, "Device peak clock: %d MHz\n", devicePeakClock);
      fprintf(stderr, "Shared memory per Block | per SM: %lu | %lu\n", sharedMemoryBlock, sharedMemorySm);
      fprintf(stderr, "Multiprocessors per task under cuda-mps: %d\n\n", deviceProp.multiProcessorCount);
    }


  //int cudaBlockDim = CUDA_BLOCK_DIM;
  // NOTE: See this https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities , Table 15.
  // NOTE: Also this https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
  // NOTE: NB - Always set MaxUsedRegisters to 32 in order to achieve 100% SM occupancy (project's Configuration properties -> CUDA C/C++ -> Device)

  Cc cc(deviceProp);
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

  // Maximum number of resident thread blocks per multiprocessor
  auto smxBlock = cc.GetSmxBlock();

  // CUDA_grid_dim = N_BLOCKS; //3072;
  CUDA_grid_dim = Nfactor * deviceProp.multiProcessorCount * smxBlock;

  if (!checkex)
    {
      fprintf(stderr, "Resident blocks per multiprocessor: %d\n", smxBlock);
      fprintf(stderr, "Grid dim (x%d): %d = %d*%d\n", Nfactor, CUDA_grid_dim, deviceProp.multiProcessorCount * Nfactor, smxBlock);
      fprintf(stderr, "Block dim: %d\n", CUDA_BLOCK_DIM);
    }

  cudaError_t res;

  //Global parameters
  res = cudaMemcpyToSymbol(CUDA_beta_pole, beta_pole, sizeof(double) * (N_POLES + 1));
  res = cudaMemcpyToSymbol(CUDA_lambda_pole, lambda_pole, sizeof(double) * (N_POLES + 1));
  res = cudaMemcpyToSymbol(CUDA_par, par, sizeof(double) * 4);
  // cl = log(cl);
  res = cudaMemcpyToSymbol(CUDA_lcl, &cl, sizeof(cl));
  res = cudaMemcpyToSymbol(CUDA_Alamda_start, &Alamda_start, sizeof(Alamda_start));
  res = cudaMemcpyToSymbol(CUDA_Alamda_incr, &Alamda_incr, sizeof(Alamda_incr));
  // res = cudaMemcpyToSymbol(CUDA_Alamda_incrr, &Alamda_incrr, sizeof(Alamda_incrr));
  res = cudaMemcpyToSymbol(CUDA_Mmax, &m_max, sizeof(m_max));
  res = cudaMemcpyToSymbol(CUDA_Lmax, &l_max, sizeof(l_max));
  res = cudaMemcpyToSymbol(CUDA_tim, tim, sizeof(double) * (MAX_N_OBS + 1));
  res = cudaMemcpyToSymbol(CUDA_Phi_0, &Phi_0, sizeof(Phi_0));

  res = cudaMalloc(&pWeight, (ndata + 3 + 1) * sizeof(double));
  res = cudaMemcpy(pWeight, weight, (ndata + 3 + 1) * sizeof(double), cudaMemcpyHostToDevice);
  res = cudaMemcpyToSymbol(CUDA_Weight, &pWeight, sizeof(pWeight));
  res = cudaMemcpyToSymbol(CUDA_ee, ee, 3 * (MAX_N_OBS + 1) * sizeof(double)); //, cudaMemcpyHostToDevice);
  res = cudaMemcpyToSymbol(CUDA_ee0, ee0, 3 * (MAX_N_OBS + 1) * sizeof(double)); //, cudaMemcpyHostToDevice);
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(&event1, cudaEventBlockingSync|cudaEventDisableTiming);
  cudaEventCreateWithFlags(&event2, cudaEventBlockingSync|cudaEventDisableTiming);

  cudaMallocHost(&theEnd, sizeof(int));

  return (res == cudaSuccess) ? 1 : 0;
}


void CUDAUnprepare(void)
{
  //cudaUnbindTexture(texWeight);
  //cudaFree(pee);
  //cudaFree(pee0);
  cudaFree(pWeight);

  cudaFreeHost(theEnd);
  theEnd = NULL;

  cudaStreamDestroy(stream1);
  cudaStreamDestroy(stream2);
  cudaEventDestroy(event1);
  cudaEventDestroy(event2);
}


static double precalcpct = 0;
int CUDAPrecalc(int cudadev, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double *conw_r,
		int ndata, int *ia, int *ia_par, int* new_conw, double* cg_first, double* sig, int Numfac, double* brightness)
{
  //int* endPtr;
  int max_test_periods, iC;
  double sum_dark_facet, ave_dark_facet;
  int i, n, m;
  int n_iter_max;
  double iter_diff_max;
  //freq_result *res;
  void *pcc, *pbrightness, *psig;

  setpriority(PRIO_PROCESS, 0, -20);

  // NOTE: max_test_periods dictates the CUDA_Grid_dim_precalc value which is actual Threads-per-Block
  /*	Cuda Compute profiler gives the following advice for almost every kernel launched:
	"Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute 16 threads per block.
	Consequently, some threads in a warp are masked off and those hardware resources are unused. Try changing the number of threads per block to be a multiple of 32 threads.
	Between 128 and 256 threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one large thread block per multiprocessor
	if latency affects performance. This is particularly beneficial to kernels that frequently call __syncthreads().*/

  max_test_periods = 10; //10;
  sum_dark_facet = 0.0;
  ave_dark_facet = 0.0;

  //#ifdef _DEBUG
  //	int n_max = (int)((freq_start - freq_end) / freq_step) + 1;
  //	if (n_max < max_test_periods)
  //	{
  //		max_test_periods = n_max;
  //		fprintf(stderr, "n_max(%d) < max_test_periods (%d)\n", n_max, max_test_periods);
  //	}
  //	else
  //	{
  //		fprintf(stderr, "n_max(%d) > max_test_periods (%d)\n", n_max, max_test_periods);
  //	}
  //
  //	fprintf(stderr, "freq_start (%.3f) - freq_end (%.3f) / freq_step (%.3f) = n_max (%d)\n", freq_start, freq_end, freq_step, n_max);
  //#endif

  for (i = 1; i <= n_ph_par; i++)
    {
      ia[n_coef + 3 + i] = ia_par[i];
    }

  n_iter_max = 0;
  iter_diff_max = -1;
  if (stop_condition > 1)
    {
      n_iter_max = (int)stop_condition;
      iter_diff_max = 0;
      n_iter_min = 0; /* to not overwrite the n_iter_max value */
    }
  if (stop_condition < 1)
    {
      n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
      iter_diff_max = stop_condition;
    }

  cudaError_t err;
  int isPrecalc = 1;
  /*int i_col, sh_icol_local[CUDA_BLOCK_DIM], sh_irow_local[CUDA_BLOCK_DIM];
    double piv_inv, sh_big_local[CUDA_BLOCK_DIM];*/

  //here move data to device
  cudaMemcpyToSymbolAsync(CUDA_Ncoef, &n_coef, sizeof(n_coef), 0, cudaMemcpyHostToDevice, stream1);
  // cudaMemcpyToSymbolAsync(CUDA_Nphpar, &n_ph_par, sizeof(n_ph_par), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Numfac, &Numfac, sizeof(Numfac), 0, cudaMemcpyHostToDevice, stream1);
  m = Numfac + 1;
  cudaMemcpyToSymbolAsync(CUDA_Numfac1, &m, sizeof(m), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_ia, ia, sizeof(int) * (MAX_N_PAR + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_cg_first, cg_first, sizeof(double) * (MAX_N_PAR + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_n_iter_max, &n_iter_max, sizeof(n_iter_max), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_n_iter_min, &n_iter_min, sizeof(n_iter_min), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_ndata, &ndata, sizeof(ndata), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_iter_diff_max, &iter_diff_max, sizeof(iter_diff_max), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_conw_r, conw_r, sizeof(conw_r), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3, 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Darea, d_area, sizeof(double) * (MAX_N_FAC + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Is_Precalc, &isPrecalc, sizeof isPrecalc, 0, cudaMemcpyHostToDevice, stream1);

  err = cudaMalloc(&pbrightness, (ndata + 1) * sizeof(double));
  err = cudaMemcpyAsync(pbrightness, brightness, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice, stream1);
  err = cudaMemcpyToSymbolAsync(CUDA_brightness, &pbrightness, sizeof(pbrightness), 0, cudaMemcpyHostToDevice, stream1);
  err = cudaMalloc(&psig, (ndata + 1) * sizeof(double));
  err = cudaMemcpyAsync(psig, sig, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice, stream1);
  err = cudaMemcpyToSymbolAsync(CUDA_sig, &psig, sizeof(psig), 0, cudaMemcpyHostToDevice, stream1);
  if (err) printf("Error: %s\n", cudaGetErrorString(err));

  /* number of fitted parameters */
  int lmfit = 0, llastma = 0, llastone = 1, ma = n_coef + 5 + n_ph_par;
  for (m = 1; m <= ma; m++)
    {
      if (ia[m])
	{
	  lmfit++;
	  llastma = m;
	}
    }

  llastone = 1;
  for (m = 2; m <= llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
    {
      if (!ia[m]) break;
      llastone = m;
    }

  cudaMemcpyToSymbolAsync(CUDA_ma, &ma, sizeof(ma), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_mfit, &lmfit, sizeof(lmfit), 0, cudaMemcpyHostToDevice, stream1);
  m = lmfit + 1;
  cudaMemcpyToSymbolAsync(CUDA_mfit1, &m, sizeof(m), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_lastma, &llastma, sizeof(llastma), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_lastone, &llastone, sizeof(llastone), 0, cudaMemcpyHostToDevice, stream1);
  // printf("ma = %d, mfit = %d, m = %d, lastma = %d, lastone = %d\n", ma, lmfit, m, llastma, llastone);
  m = ma - 2 - n_ph_par;
  cudaMemcpyToSymbolAsync(CUDA_ncoef0, &m, sizeof(m), 0, cudaMemcpyHostToDevice, stream1);

  int CUDA_Grid_dim_precalc = CUDA_grid_dim;
  if (max_test_periods < CUDA_Grid_dim_precalc)
    {
      CUDA_Grid_dim_precalc = max_test_periods;
      //#ifdef _DEBUG
      //		fprintf(stderr, "CUDA_Grid_dim_precalc = %d\n", CUDA_Grid_dim_precalc);
      //#endif
    }

  //cudaMallocHost(&res, CUDA_Grid_dim_precalc * sizeof(freq_result));

  err = cudaMalloc(&pcc, (CUDA_Grid_dim_precalc + 32) * sizeof(freq_context));
  cudaMemcpyToSymbolAsync(CUDA_CC, &pcc, sizeof(pcc), 0, cudaMemcpyHostToDevice, stream1);
  //err = cudaMalloc(&pfr, (CUDA_Grid_dim_precalc + 32) * sizeof(freq_result));
  //cudaMemcpyToSymbolAsync(CUDA_FR, &pfr, sizeof(pfr), 0, cudaMemcpyHostToDevice, stream1);

  m = (Numfac + 1) * (n_coef + 1);
  cudaMemcpyToSymbolAsync(CUDA_Dg_block, &m, sizeof(m), 0, cudaMemcpyHostToDevice, stream1);

  //  double *pa,
  double *pg, *pal, *pco, *pdytemp, *pytemp;

  //err = cudaMalloc(&pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
  //cudaMemcpyToSymbolAsync(CUDA_Area, &pa, sizeof(pa), 0, cudaMemcpyHostToDevice, stream1);
  err = cudaMalloc(&pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
  err = cudaMemcpyToSymbolAsync(CUDA_Dg, &pg, sizeof(pg), 0, cudaMemcpyHostToDevice, stream1);
  err = cudaMalloc(&pal, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 2) * sizeof(double));
  err = cudaMalloc(&pco, (CUDA_Grid_dim_precalc) * (lmfit + 1) * (lmfit + 2) * sizeof(double));
#ifndef NEWDYTEMP
  err = cudaMalloc(&pdytemp, (CUDA_Grid_dim_precalc + 1) * (max_l_points + 1) * (ma + 1) * sizeof(double));
#endif
  err = cudaMalloc(&pytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * sizeof(double));

  dim3 dim_3(32, CUDA_Grid_dim_precalc, 1);

  for(m = 0; m < CUDA_Grid_dim_precalc; m++)
    {
      freq_context ps;
      // ps.Area = &pa[m * (Numfac + 1)];
      ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];
      ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];
      ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];
#ifndef NEWDYTEMP
      ps.dytemp = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
#endif
      ps.ytemp = &pytemp[m * (max_l_points + 1)];
      freq_context *pt = &((freq_context*)pcc)[m];
      err = cudaMemcpyAsync(pt, &ps, sizeof(void*) * 7, cudaMemcpyHostToDevice, m&1 ? stream1 : stream2);
    }

  //cudaStreamSynchronize(stream2);

  //printf("MaxTestPeriods %d %d\n", max_test_periods, CUDA_Grid_dim_precalc);

  for(n = 1; n <= max_test_periods; n += CUDA_Grid_dim_precalc)
    {
      CudaCalculatePrepare<<<1, CUDA_Grid_dim_precalc, 0, stream1>>>(n, max_test_periods);

      for(m = 1; m <= N_POLES; m++)
	{
	  //zero global End signal
	  *theEnd = 0;
	  cudaMemcpyToSymbolAsync(CUDA_End, theEnd, sizeof(int), 0, cudaMemcpyHostToDevice, stream1);
	  CudaCalculatePreparePole<<<1, CUDA_Grid_dim_precalc, 0, stream1>>>(m, freq_start, freq_step, n); 


#ifdef _DEBUG
	  printf("."); // printf("ia[1] = %d\r\n", ia[1]);
#endif

	  int loop = 0;
	  while(!*theEnd)
	    {
	      loop++;
	      CudaCalculateIter1Begin<<<1, CUDA_Grid_dim_precalc, 0, stream1>>>(CUDA_Grid_dim_precalc);
	      // 1, dim_3 works
	      // CudaCalculateIter1Mrqcof1Start<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
	      CudaCalculateIter1Mrqcof1Start<<<1, dim_3, 0, stream1>>>();
	      cudaEventRecord(event1, stream1);
	      cudaStreamWaitEvent(stream2, event1);
	      cudaMemcpyFromSymbolAsync(theEnd, CUDA_End, sizeof(int), 0, cudaMemcpyDeviceToHost, stream2);
	      cudaEventRecord(event2, stream2);

	      for(iC = 1; iC < l_curves; iC++)
		{
		  if(in_rel[iC])
		    if(ia[1])
		      CudaCalculateIter1Mrqcof1CurveM12I1IA1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		    else
		      CudaCalculateIter1Mrqcof1CurveM12I1IA0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		  else
		    if(ia[1])
		      CudaCalculateIter1Mrqcof1CurveM12I0IA1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		    else
		      CudaCalculateIter1Mrqcof1CurveM12I0IA0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		}

	      if(in_rel[l_curves])
		{ // 1, dim_3 x NO NO NO
		  CudaCalculateIter1Mrqcof1Curve1LastI1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		  if(ia[1])
		    CudaCalculateIter1Mrqcof1Curve2I1IA1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		  else // 1, dim_3 ???? works
		    CudaCalculateIter1Mrqcof1Curve2I1IA0<<<1, dim_3, 0, stream1>>>();		    
		  // CudaCalculateIter1Mrqcof1Curve2I1IA0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();		    
		}
	      else
		{ // 1, dim_3 This can not be changed!!!!
		  CudaCalculateIter1Mrqcof1Curve1LastI0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		  if(ia[1])
		    CudaCalculateIter1Mrqcof1Curve2I0IA1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		  else
		    CudaCalculateIter1Mrqcof1Curve2I0IA0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		}

	      CudaCalculateIter1Mrqcof1End<<<1, dim_3, 0, stream1>>>();
	      CudaCalculateIter1Mrqmin1End<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
	      // 1, dim_3 OK
	      // CudaCalculateIter1Mrqcof2Start<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
	      cudaEventSynchronize(event2);
	      // cudaStreamSynchronize(stream1);

	      CudaCalculateIter1Mrqcof2Start<<<1, dim_3, 0, stream1>>>();

	      for(iC = 1; iC < l_curves; iC++)
		{
		  if(in_rel[iC])
		    if(ia[1])
		      CudaCalculateIter1Mrqcof2CurveM12I1IA1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		    else
		      CudaCalculateIter1Mrqcof2CurveM12I1IA0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		  else
		    if(ia[1])
		      CudaCalculateIter1Mrqcof2CurveM12I0IA1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		    else
		      CudaCalculateIter1Mrqcof2CurveM12I0IA0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		}
	      
	      if(in_rel[l_curves])
		{ // 1, dim_3 OK
		  // CudaCalculateIter1Mrqcof2Curve1LastI1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		  CudaCalculateIter1Mrqcof2Curve1LastI1<<<1, dim_3, 0, stream1>>>();
		  if(ia[1])
		    CudaCalculateIter1Mrqcof2Curve2I1IA1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		  else // 1, dim_3, ok
		    CudaCalculateIter1Mrqcof2Curve2I1IA0<<<1, dim_3, 0, stream1>>>();
		  // CudaCalculateIter1Mrqcof2Curve2I1IA0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		}
	      else
		{ // 1, dim_3 no no no
		  CudaCalculateIter1Mrqcof2Curve1LastI0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		  // CudaCalculateIter1Mrqcof2Curve1LastI0<<<1, dim_3, 0, stream1>>>();
		  if(ia[1]) //ok
		    CudaCalculateIter1Mrqcof2Curve2I0IA1<<<1, dim_3, 0, stream1>>>();
		  // CudaCalculateIter1Mrqcof2Curve2I0IA1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		  else //1, dim_3 no no no
		    //CudaCalculateIter1Mrqcof2Curve2I0IA0<<<1, dim_3, 0, stream1>>>();
		    CudaCalculateIter1Mrqcof2Curve2I0IA0<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
		}

	      CudaCalculateIter1Mrqcof2End<<<1, dim_3, 0, stream1>>>();
	      //CudaCalculateIter1Mrqmin2End<<<1, CUDA_Grid_dim_precalc, 0, stream1>>>();

	      // ok
	      // CudaCalculateIter1Mrqmin2End<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
	      CudaCalculateIter1Mrqmin2End<<<1, dim_3, 0, stream1>>>();

	      // 1, dim_3, ok
	      // CudaCalculateIter2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM, 0, stream1>>>();
	      CudaCalculateIter2<<<1, dim_3, 0, stream1>>>();
	      // cudaStreamSynchronize(stream1);

	      *theEnd = (*theEnd >= CUDA_Grid_dim_precalc);
	      precalcpct += 0.00001;
	      boinc_fraction_done(precalcpct > 0.02 ? 0.02 : precalcpct);
	    }
	  printf("."); fflush(stdout);
	  CudaCalculateFinishPole<<<1, CUDA_Grid_dim_precalc, 0, stream1>>>();
	  // cudaStreamSynchronize(stream1); // just to keep queue short
	}

      CudaCalculateFinish<<<1, CUDA_Grid_dim_precalc, 0, stream1>>>();
      //read results here+
      //err = cudaMemcpyAsync(res, pfr, sizeof(freq_result) * CUDA_Grid_dim_precalc, cudaMemcpyDeviceToHost, stream1);
      cudaStreamSynchronize(stream1);

      for (m = 1; m <= CUDA_Grid_dim_precalc; m++)
	{
	  // if(res[m - 1].isReported == 1)
	  if (isReported[m-1] == 1)
	    sum_dark_facet = sum_dark_facet + dark_best[m-1];
	  // sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
	}
    } /* period loop */

  isPrecalc = 0;

  cudaMemcpyToSymbolAsync(CUDA_Is_Precalc, &isPrecalc, sizeof(isPrecalc), 0, cudaMemcpyHostToDevice, stream1);
  cudaStreamSynchronize(stream1);
  //cudaFree(pa);
  cudaFree(pg);
  cudaFree(pal);
  cudaFree(pco);
#ifndef NEWDYTEMP
  cudaFree(pdytemp);
#endif
  cudaFree(pytemp);
  cudaFree(pcc);
  //cudaFree(pfr);
  cudaFree(pbrightness);
  cudaFree(psig);
  //cudaFreeHost(res);


  ave_dark_facet = sum_dark_facet / max_test_periods;

  if (ave_dark_facet < 1.0)
    *new_conw = 1; /* new correct conwexity weight */
  if (ave_dark_facet >= 1.0)
    *conw_r = *conw_r * 2; /* still not good */

  return 1;
}


int CUDAStart(int cudadev, int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
	      int ndata, int* ia, int* ia_par, double* cg_first, MFILE& mf, double escl, double* sig, int Numfac, double* brightness)
{
  int retval, i, n, m, iC, n_max = (int)((freq_start - freq_end) / freq_step) + 1;

  setpriority(PRIO_PROCESS, 0, 19);

  if(n_max < CUDA_grid_dim)
    CUDA_grid_dim = 32 * ((n_max + 31) / 32);
  
  int n_iter_max, LinesWritten;
  double iter_diff_max;
  //freq_result *res;
  void *pcc, *pbrightness, *psig;
  char buf[256];

  for (i = 1; i <= n_ph_par; i++)
    {
      ia[n_coef + 3 + i] = ia_par[i];
    }

  n_iter_max = 0;
  iter_diff_max = -1;
  if (stop_condition > 1)
    {
      n_iter_max = (int)stop_condition;
      iter_diff_max = 0;
      n_iter_min = 0; /* to not overwrite the n_iter_max value */
    }
  if (stop_condition < 1)
    {
      n_iter_max = MAX_N_ITER; /* to avoid neverending loop */
      iter_diff_max = stop_condition;
    }

  cudaError_t err;

  //cudaMallocHost(&res, CUDA_grid_dim * sizeof(freq_result));

  //here move data to device
  cudaMemcpyToSymbolAsync(CUDA_Ncoef, &n_coef, sizeof(n_coef), 0, cudaMemcpyHostToDevice, stream1); 
  //  cudaMemcpyToSymbolAsync(CUDA_Nphpar, &n_ph_par, sizeof(n_ph_par), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Numfac, &Numfac, sizeof(Numfac), 0, cudaMemcpyHostToDevice, stream1);
  m = Numfac + 1;
  cudaMemcpyToSymbolAsync(CUDA_Numfac1, &m, sizeof(m), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_ia, ia, sizeof(int) * (MAX_N_PAR + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_cg_first, cg_first, sizeof(double) * (MAX_N_PAR + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_n_iter_max, &n_iter_max, sizeof(n_iter_max), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_n_iter_min, &n_iter_min, sizeof(n_iter_min), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_ndata, &ndata, sizeof(ndata), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_iter_diff_max, &iter_diff_max, sizeof(iter_diff_max), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_conw_r, &conw_r, sizeof(conw_r), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3, 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Darea, d_area, sizeof(double) * (MAX_N_FAC + 1), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0, cudaMemcpyHostToDevice, stream1);

  err = cudaMalloc(&pbrightness, (ndata + 1) * sizeof(double));
  err = cudaMemcpyAsync(pbrightness, brightness, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice, stream1);
  err = cudaMemcpyToSymbolAsync(CUDA_brightness, &pbrightness, sizeof(pbrightness), 0, cudaMemcpyHostToDevice, stream1);
  //err = cudaBindTexture(0, texbrightness, pbrightness, (ndata + 1) * sizeof(double));

  err = cudaMalloc(&psig, (ndata + 1) * sizeof(double));
  err = cudaMemcpyAsync(psig, sig, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice, stream1);
  err = cudaMemcpyToSymbolAsync(CUDA_sig, &psig, sizeof(psig), 0, cudaMemcpyHostToDevice, stream1);
  //err = cudaBindTexture(0, texsig, psig, (ndata + 1) * sizeof(double));
  if (err) printf("Error: %s", cudaGetErrorString(err));

  /* number of fitted parameters */
  int lmfit = 0, llastma = 0, llastone = 1, ma = n_coef + 5 + n_ph_par;
  for (m = 1; m <= ma; m++)
    {
      if (ia[m])
	{
	  lmfit++;
	  llastma = m;
	}
    }
  llastone = 1;
  for (m = 2; m <= llastma; m++) //ia[1] is skipped because ia[1]=0 is acceptable inside mrqcof
    {
      if (!ia[m]) break;
      llastone = m;
    }
  cudaMemcpyToSymbolAsync(CUDA_ma, &ma, sizeof(ma), 0, cudaMemcpyHostToDevice, stream1); 
  cudaMemcpyToSymbolAsync(CUDA_mfit, &lmfit, sizeof(lmfit), 0, cudaMemcpyHostToDevice, stream1);
  m = lmfit + 1;
  cudaMemcpyToSymbolAsync(CUDA_mfit1, &m, sizeof(m), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_lastma, &llastma, sizeof(llastma), 0, cudaMemcpyHostToDevice, stream1);
  cudaMemcpyToSymbolAsync(CUDA_lastone, &llastone, sizeof(llastone), 0, cudaMemcpyHostToDevice, stream1);
  m = ma - 2 - n_ph_par;
  cudaMemcpyToSymbolAsync(CUDA_ncoef0, &m, sizeof(m), 0, cudaMemcpyHostToDevice, stream1);

  err = cudaMalloc(&pcc, (CUDA_grid_dim + 32) * sizeof(freq_context));
  cudaMemcpyToSymbolAsync(CUDA_CC, &pcc, sizeof(pcc), 0, cudaMemcpyHostToDevice, stream1);
  //err = cudaMalloc(&pfr, (CUDA_grid_dim + 32) * sizeof(freq_result));
  //cudaMemcpyToSymbolAsync(CUDA_FR, &pfr, sizeof(pfr), 0, cudaMemcpyHostToDevice, stream1);

  m = (Numfac + 1) * (n_coef + 1);
  cudaMemcpyToSymbolAsync(CUDA_Dg_block, &m, sizeof(m), 0, cudaMemcpyHostToDevice, stream1);

  //double *pa;
  double *pg, *pal, *pco, *pdytemp, *pytemp;

  //err = cudaMalloc(&pa, CUDA_grid_dim * (Numfac + 1) * sizeof(double));
  //err = cudaMemcpyToSymbolAsync(CUDA_Area, &pa, sizeof(pa), 0, cudaMemcpyHostToDevice, stream1);
	
  err = cudaMalloc(&pg, CUDA_grid_dim * (Numfac + 1) * (n_coef + 1) * sizeof(double));
  err = cudaMemcpyToSymbolAsync(CUDA_Dg, &pg, sizeof(pg), 0, cudaMemcpyHostToDevice, stream1);
  //err = cudaBindTexture(0, texDg, pg, CUDA_grid_dim * (Numfac + 1) * (n_coef + 1) * sizeof(double));
  
  err = cudaMalloc(&pal, CUDA_grid_dim * (lmfit + 1) * (lmfit + 1) * sizeof(double));
  err = cudaMalloc(&pco, CUDA_grid_dim * (lmfit + 1) * (lmfit + 1) * sizeof(double));
#ifndef NEWDYTEMP
  err = cudaMalloc(&pdytemp, (CUDA_grid_dim + 1) * (max_l_points + 1) * (ma + 1) * sizeof(double));
#endif
  err = cudaMalloc(&pytemp, CUDA_grid_dim * (max_l_points + 1) * sizeof(double));

  for (m = 0; m < CUDA_grid_dim; m++)
    {
      freq_context ps;
      //ps.Area    = &pa[m * (Numfac + 1)];
      ps.Dg      = &pg[m * (Numfac + 1) * (n_coef + 1)];
      ps.alpha   = &pal[m * (lmfit + 1) * (lmfit + 1)];
      ps.covar   = &pco[m * (lmfit + 1) * (lmfit + 1)];
#ifndef NEWDYTEMP
      ps.dytemp  = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
#endif
      ps.ytemp   = &pytemp[m * (max_l_points + 1)];
      freq_context *pt = &((freq_context*)pcc)[m];
      err = cudaMemcpyAsync(pt, &ps, sizeof(void*) * 7, cudaMemcpyHostToDevice, m&1 ? stream1 : stream2);
    }

  err = cudaStreamSynchronize(stream2);

  //int firstreport = 0;//beta debug
  // auto oldFractionDone = 0.0001;

  // printf("N %d %d %d\n", n_start_from, n_max, CUDA_grid_dim);

  n = n_start_from;
  int dim1 = CUDA_grid_dim;
  int dim2 = 1;
  if(CUDA_grid_dim % 32 == 0)
    {
      dim1 = CUDA_grid_dim / 32;
      dim2 = 32;
    }

  dim3 dim_3(32, dim2, 1);

  dim3 block4(CUDA_BLOCK_DIM, BLOCKX4, 1);
  // dim3 block8(CUDA_BLOCK_DIM, BLOCKX8, 1);
  // dim3 block16(CUDA_BLOCK_DIM, BLOCKX16, 1);
  // dim3 block32(CUDA_BLOCK_DIM, BLOCKX32, 1);

  while(n <= n_max)
    {
      double fractionDone = (double)n / (double)n_max;

      CudaCalculatePrepare<<<dim1, dim2, 0, stream1>>>(n, n_max);

      for(m = 1; m <= N_POLES; m++)
	{
	  double q = n_max - n; q = q > CUDA_grid_dim ? CUDA_grid_dim : q;
	  double fractionDone2 = (double)(n-1)/(double)n_max + q/(double)n_max * (double)(m-1)/(double)N_POLES;
	  fractionDone = fractionDone2 > 0.99990 ? 0.99990 : fractionDone2;
	  // printf("\r                            %d %d %d %9.6f \r", n, N_POLES, m, fractionDone); fflush(stdout);

      //#if _DEBUG
      //		float fraction = fractionDone * 100;
      //		std::time_t t = std::time(nullptr);   // get time now
      //		std::tm* now = std::localtime(&t);
      //
      //		printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
      //		fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
      //#endif

	  boinc_fraction_done(fractionDone);

#ifdef _DEBUG
	  float fraction2 = fractionDone2 * 100;
	  //float fraction = fractionDone * 100;
	  std::time_t t = std::time(nullptr);   // get time now
	  std::tm* now = std::localtime(&t);

	  printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
	  fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
#endif

	  //zero global End signal
	  *theEnd = 0;
	  cudaMemcpyToSymbolAsync(CUDA_End, theEnd, sizeof(int), 0, cudaMemcpyHostToDevice, stream1);
	  CudaCalculatePreparePole<<<dim1, dim2, 0, stream1>>>(m, freq_start, freq_step, n); // RRRR
	  int loop = 0;

	  while(!*theEnd)
	    {
	      //usleep(1); // allow higher priority threads (stage 1) run
	      CudaCalculateIter1Begin<<<dim1, dim2, 0, stream1 >>>(CUDA_grid_dim); // RRRR
	      //usleep(1); // allow higher priority threads (stage 1) run
	      CudaCalculateIter1Mrqcof1Start<<<CUDA_grid_dim/BLOCKX4, block4/*CUDA_BLOCK_DIM*/, 0, stream1>>>();
	      cudaEventRecord(event1, stream1);
	      cudaStreamWaitEvent(stream2, event1);
	      cudaMemcpyFromSymbolAsync(theEnd, CUDA_End, sizeof(int), 0, cudaMemcpyDeviceToHost, stream2);
	      cudaEventRecord(event2, stream2);

	      for(iC = 1; iC < l_curves; iC++)
		{
		  if(in_rel[iC])
		    if(ia[1])
		      CudaCalculateIter1Mrqcof1CurveM12I1IA1<<<CUDA_grid_dim/1, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		    else
		      CudaCalculateIter1Mrqcof1CurveM12I1IA0<<<CUDA_grid_dim/1, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		  else
		    if(ia[1])
		      CudaCalculateIter1Mrqcof1CurveM12I0IA1<<<CUDA_grid_dim/1, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		    else
		      CudaCalculateIter1Mrqcof1CurveM12I0IA0<<<CUDA_grid_dim/1, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		}

	      //usleep(1); // allow higher priority threads (stage 1) run

	      if(in_rel[l_curves])
		{
		  CudaCalculateIter1Mrqcof1Curve1LastI1<<<CUDA_grid_dim/BLOCKX4, block4, 0, stream1>>>(); //4 max, shared
		  //usleep(1); // allow higher priority threads (stage 1) run
		  if(ia[1])
		    CudaCalculateIter1Mrqcof1Curve2I1IA1<<<CUDA_grid_dim/1, CUDA_BLOCK_DIM, 0, stream1>>>(); 
		  else
		    CudaCalculateIter1Mrqcof1Curve2I1IA0<<<CUDA_grid_dim/BLOCKX4, block4, 0, stream1>>>();
		}
	      else
		{
		  CudaCalculateIter1Mrqcof1Curve1LastI0<<<CUDA_grid_dim/BLOCKX4, block4, 0, stream1>>>();
		  //usleep(1); // allow higher priority threads (stage 1) run
		  if(ia[1])
		    CudaCalculateIter1Mrqcof1Curve2I0IA1<<<CUDA_grid_dim, CUDA_BLOCK_DIM, 0, stream1>>>();
		  else
		    CudaCalculateIter1Mrqcof1Curve2I0IA0<<<CUDA_grid_dim/1, CUDA_BLOCK_DIM, 0, stream1>>>();
		}
			 
	      //usleep(1); // allow higher priority threads (stage 1) run
	      CudaCalculateIter1Mrqcof1End<<<dim1, dim_3, 0, stream1>>>(); //RRRR
	      //usleep(1); // allow higher priority threads (stage 1) run
	      CudaCalculateIter1Mrqmin1End<<<CUDA_grid_dim/1, CUDA_BLOCK_DIM, 0, stream1>>>(); // 1 max?, gauss, shared

	      cudaEventSynchronize(event2);
	      //cudaStreamSynchronize(stream1);
	      
	      //usleep(1); // allow higher priority threads (stage 1) run
	      CudaCalculateIter1Mrqcof2Start<<<CUDA_grid_dim/4, block4 /*CUDA_BLOCK_DIM*/, 0, stream1>>>();
	      for(iC = 1; iC < l_curves; iC++)
		{
		  //usleep(1); // allow higher priority threads (stage 1) run
		  if(in_rel[iC])
		    if(ia[1])
		      CudaCalculateIter1Mrqcof2CurveM12I1IA1<<<CUDA_grid_dim, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		    else
		      CudaCalculateIter1Mrqcof2CurveM12I1IA0<<<CUDA_grid_dim, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);

		  else
		    if(ia[1])
		      CudaCalculateIter1Mrqcof2CurveM12I0IA1<<<CUDA_grid_dim, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		    else
		      CudaCalculateIter1Mrqcof2CurveM12I0IA0<<<CUDA_grid_dim, CUDA_BLOCK_DIM, 0, stream1>>>(l_points[iC]);
		}

	      //cudaStreamSynchronize(stream1);

	      //usleep(1); // allow higher priority threads (stage 1) run
	      if(in_rel[l_curves])
		{
		  CudaCalculateIter1Mrqcof2Curve1LastI1<<<CUDA_grid_dim/BLOCKX4, block4, 0, stream1>>>();
		  //usleep(1); // allow higher priority threads (stage 1) run
		  if(ia[1])
		    CudaCalculateIter1Mrqcof2Curve2I1IA1<<<CUDA_grid_dim, CUDA_BLOCK_DIM, 0, stream1>>>();
		  else
		    CudaCalculateIter1Mrqcof2Curve2I1IA0<<<CUDA_grid_dim/BLOCKX4, block4, 0, stream1>>>();
		}
	      else // last
		{
		  CudaCalculateIter1Mrqcof2Curve1LastI0<<<CUDA_grid_dim/BLOCKX4, block4, 0, stream1>>>();
		  //usleep(1); // allow higher priority threads (stage 1) run
		  if(ia[1])
		    CudaCalculateIter1Mrqcof2Curve2I0IA1<<<CUDA_grid_dim, CUDA_BLOCK_DIM, 0, stream1>>>();
		  else
		    CudaCalculateIter1Mrqcof2Curve2I0IA0<<<CUDA_grid_dim/BLOCKX4, block4, 0, stream1>>>();
		}

	      //msleep(1); // allow higher priority threads (stage 1) run
	      CudaCalculateIter1Mrqcof2End<<<dim1, dim_3, 0, stream1>>>(); // RRRR
	      CudaCalculateIter1Mrqmin2End<<<CUDA_grid_dim/1, CUDA_BLOCK_DIM, 0, stream1>>>(); // RRRR

	      CudaCalculateIter2<<<CUDA_grid_dim/BLOCKX4, block4, 0, stream1>>>();

	      //if((loop & 7) == 7)
	      //msleep(250);
	      if((loop & 3) == 3)
		{
		  //msleep(70);
		  double cp = fractionDone2 + ((double)loop / (double)64.0 * (double)q/(double)(n_max * N_POLES));
		  cp = cp > 0.99990 ? 0.99990 : cp;
		  fractionDone = cp;
		  boinc_fraction_done(fractionDone);

		  printf("%9.6f \r ", fractionDone); fflush(stdout);
		}

	      //msleep(1); // allow higher priority threads (stage 1) run

	      *theEnd = (*theEnd >= CUDA_grid_dim);
	      loop++;
	    }

	  printf("."); fflush(stdout);

	  CudaCalculateFinishPole<<<dim1, dim2, 0, stream1>>>();
	}

      CudaCalculateFinish<<<dim1, dim2, 0, stream1>>>();

      //read results here synchronously

      //err = cudaMemcpyAsync(res, pfr, sizeof(freq_result) * CUDA_grid_dim, cudaMemcpyDeviceToHost, stream1);
      if(boinc_time_to_checkpoint() || boinc_is_standalone())
	{
	  retval = DoCheckpoint(mf, (n - 1) + LinesWritten, 1, conw_r); //zero lines
	  if (retval) { fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval); exit(retval); }
	  boinc_checkpoint_completed();
	  boinc_fraction_done(fractionDone);
	}

      printf("\n"); fflush(stdout);

      //err = cudaStreamSynchronize(stream1);

      LinesWritten = 0;

      for(m = 1; m <= CUDA_grid_dim; m++)
	{
	  // if(res[m - 1].isReported == 1)
	  if(isReported[m - 1] == 1)
	    {
	      LinesWritten++;
	      /* output file */
	      if(n == 1 && m == 1)
		{ // res[m - 1].
		  mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * per_best[m - 1], dev_best[m - 1], dev_best[m - 1] * dev_best[m - 1] * (ndata - 3), conw_r * escl * escl, round(la_best[m - 1]), round(be_best[m - 1]));
		}
	      else
		{
		  mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * per_best[m - 1], dev_best[m - 1], dev_best[m - 1] * dev_best[m - 1] * (ndata - 3), dark_best[m - 1], round(la_best[m - 1]), round(be_best[m - 1]));
		}
	    }
	}

      n += CUDA_grid_dim;
    } /* period loop */
	
  boinc_fraction_done(0.99992);
  printf("cuda DONE\n"); fflush(stdout);
	
  //cudaFree(pa);      
  cudaFree(pg);      
  cudaFree(pal);     
  cudaFree(pco);     
#ifndef NEWDYTEMP
  cudaFree(pdytemp); 
#endif
  cudaFree(pytemp);  
  cudaFree(pcc);     
  //cudaFree(pfr);     
  cudaFree(pbrightness);
  cudaFree(psig);      
  //cudaFreeHost(res);   


  boinc_fraction_done(0.99993);
  //nvmlShutdown();

  return 1;
}
