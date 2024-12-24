#ifndef NVML_NO_UNVERSIONED_FUNC_DEFS
#define NVML_NO_UNVERSIONED_FUNC_DEFS
#endif

#if defined _MSC_VER & _MSC_VER < 1900
//#define _WIN32_WINNT _WIN32_WINNT_WINXP 
//#define NTDDI_VERSION NTDDI_WINXPSP3 

#include <Windows.h>

//#include <wtypes.h> 
//#include <unknwn.h> 
//#include <objbase.h>

#endif

// NOTE: CUDA 11.8 supports the following compute capabilities (CC):
// compute_50,sm_50;compute_52,sm_52;compute_53,sm_53;compute_60,sm_60;compute_61,sm_61;compute_62,sm_62;compute_70,sm_70;compute_72,sm_72;compute_75,sm_75;compute_80,sm_80;compute_86,sm_86;compute_87,sm_87;compute_89,sm_89,compute_90,sm_90
// NOTE: CUDA 9.0 supports the following compute capabilities (CC):
// compute_10,sm_10;compute_11,sm_11;compute_12,sm_12;compute_20,sm_20;compute_30,sm_30;compute_32,sm_32;compute_35,sm_35

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

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 10020)
#include <nvml.h>
#endif

#ifdef __GNUC__
#include <ctime>
#include <execinfo.h>
#else

#endif
#include "ComputeCapability.h"
#include "arrayHelpers.hpp"

//global to all freq
__constant__ int CUDA_Ncoef, CUDA_Nphpar, CUDA_Numfac, CUDA_Numfac1, CUDA_Dg_block;
__constant__ int CUDA_ia[MAX_N_PAR + 1];
__constant__ int CUDA_ma, CUDA_mfit, CUDA_mfit1, CUDA_lastone, CUDA_lastma, CUDA_ncoef0;
__device__ double* CUDA_cg_first;   //[MAX_N_PAR + 1];
__device__ double CUDA_beta_pole[N_POLES + 1];
__device__ double CUDA_lambda_pole[N_POLES + 1];
__device__ double CUDA_par[4];
__device__ double CUDA_cl, CUDA_Alamda_start, CUDA_Alamda_incr;
__device__ int CUDA_n_iter_max, CUDA_n_iter_min, CUDA_ndata;
__device__ double CUDA_iter_diff_max;
__constant__ double CUDA_Nor[MAX_N_FAC + 1][3];
__constant__ double CUDA_conw_r;
__constant__ int CUDA_Lmax, CUDA_Mmax;
__device__ double CUDA_Fc[MAX_N_FAC + 1][MAX_LM + 1];
__device__ double CUDA_Fs[MAX_N_FAC + 1][MAX_LM + 1];
__device__ double CUDA_Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
__device__ double CUDA_Darea[MAX_N_FAC + 1];
__device__ double CUDA_Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
__device__ double* CUDA_ee          /*[MAX_N_OBS+1][3]*/;
__device__ double* CUDA_ee0         /*[MAX_N_OBS+1][3]*/;
__device__ double* CUDA_tim;
__device__ double* CUDA_brightness  /*[MAX_N_OBS+1]*/;
__device__ double* CUDA_sig         /*[MAX_N_OBS+1]*/;
__device__ double* CUDA_Weight      /*[MAX_N_OBS+1]*/;
__device__ double* CUDA_Area;
__device__ double* CUDA_Dg;

__constant__ double CUDA_Phi_0;
__device__ int CUDA_End;
__device__ int CUDA_Is_Precalc;

// global to one thread
__device__ freq_context* CUDA_CC;
__device__ freq_result* CUDA_FR;


int CUDA_grid_dim;
double* pee, * pee0, * pWeight; 
bool nvml_enabled = false;
double* d_CUDA_tim;

// Global variable to track the total size of allocations
size_t totalAllocatedSize = 0;

//bool if_freq_measured = false;

// ReSharper disable All

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

    status = cuCtxCreate(&hcuContext, 0x4, hcuDevice);
    if (status != CUDA_SUCCESS)
        return false;

    return true;
}

#if defined _DEBUG
#if defined __GNUC__
void printCallStack()
{
    const int maxFrames = 64;
    void* frames[maxFrames];
    int numFrames = backtrace(frames, maxFrames);
    char** symbols = backtrace_symbols(frames, numFrames);

    std::cerr << "Call stack:" << std::endl;
    for (int i = 0; i < numFrames; ++i) {
        std::cerr << symbols[i] << std::endl;
    }

    free(symbols);
}
#else
#define _WIN32_WINNT 0x0501 // Target Windows XP or later
#include <Windows.h>
#include <DbgHelp.h>
#include <iostream>
#include <stdexcept>
#include <vector>


extern "C"
USHORT WINAPI CaptureStackBackTrace(
    ULONG FramesToSkip,
    ULONG FramesToCapture,
    PVOID * BackTrace,
    PULONG BackTraceHash
);

void printCallStack()
{
    const int maxFrames = 64;
    void* frames[maxFrames];
    USHORT numFrames = CaptureStackBackTrace(0, maxFrames, frames, NULL);
    SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
    symbol->MaxNameLen = 255;
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

    HANDLE process = GetCurrentProcess();
    SymInitialize(process, NULL, TRUE);

    std::cerr << "Call stack:" << std::endl;
    for (USHORT i = 0; i < numFrames; ++i) {
        SymFromAddr(process, (DWORD64)(frames[i]), 0, symbol);
        std::cerr << i << ": " << symbol->Name << " - 0x" << std::hex << symbol->Address << std::dec << std::endl;
    }

    free(symbol);
}
#endif
#endif

void CUDAGlobalsFree()
{
    cudaFree(pee);
    cudaFree(pee0);
    cudaFree(pWeight);
    cudaFree(d_CUDA_tim);
}

// Template function for safe CUDA memory allocation
template <typename T>
cudaError_t safeCudaMalloc(T*& d_ptr, size_t size)
{
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), size * sizeof(T));

    if (err != cudaSuccess) {
        std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "Attempting CPU memory allocation..." << std::endl;
        d_ptr = static_cast<T*>(malloc(size * sizeof(T)));

        if (d_ptr == nullptr) {
            std::cerr << "CPU memory allocation also failed." << std::endl;
            return cudaErrorMemoryAllocation;
        }
    }
    else
    {
        totalAllocatedSize += size * sizeof(T);
    }

    return cudaSuccess;
}

void handleCudaError(cudaError_t err, const char* function, const char* symbol_name)
{
    if (err != cudaSuccess) {
        //std::cerr << "CUDA error in " << function << ": " << cudaGetErrorString(err) << std::endl;
        std::cerr << "CUDA error in "<< function << " | memory copy to symbol \"" << symbol_name << "\" failed: " << cudaGetErrorString(err) << std::endl;
#if defined _DEBUG
        printCallStack();
#endif
        // Handle the error (e.g., free resources, exit the program, etc.)
        CUDAGlobalsFree();
        exit(999);
    }
}

//void copyToSymbol(double* beta_pole)
//{
//    cudaError_t err = cudaMemcpyToSymbol(CUDA_beta_pole, beta_pole, sizeof(double) * (N_POLES + 1));
//    handleCudaError(err, "cudaMemcpyToSymbol");
//}

// _____________________________


// Template function for safe CUDA memory copy to symbol
template <typename T>
cudaError_t safeCudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, const char* symbol_name)
{
    cudaError_t err = cudaMemcpyToSymbol(symbol, src, count);
    handleCudaError(err, "cudaMemcpyToSymbol", symbol_name);

    totalAllocatedSize += count * sizeof(T);

    return err;
}

//// Wrapper function to handle the type deduction for arrays
//template <typename T, size_t N>
//cudaError_t copyToSymbol(T(&symbol)[N], const T* src, size_t count)
//{
//    return safeCudaMemcpyToSymbol(reinterpret_cast<const void*>(&symbol), src, count * sizeof(T));
//}

// Wrapper function to handle the type deduction for arrays
template <typename T, size_t N>
cudaError_t copyToSymbol(T(&symbol)[N], const T* src, size_t count, const char* symbol_name = {})
{
    return safeCudaMemcpyToSymbol(reinterpret_cast<const void*>(&symbol), src, count * sizeof(T), symbol_name);
}

// Overload for single numerical types
template <typename T>
cudaError_t copyToSymbol(T& symbol, const T* src, size_t count, const char* symbol_name = {})
{
    return safeCudaMemcpyToSymbol(reinterpret_cast<const void*>(&symbol), src, count * sizeof(T), symbol_name);
}

// _________________

// Template function for safe CUDA memory copy
template <typename T>
cudaError_t safeMemcopy(T* dest, const T* src, size_t count, cudaMemcpyKind kind)
{
    cudaError_t err = cudaMemcpy(dest, src, count * sizeof(T), kind);

    if (err != cudaSuccess) {
        std::cerr << "CUDA memory copy failed: " << cudaGetErrorString(err) << std::endl;
        // Handle the error appropriately
    }

    return err;
}

// Template wrapper function for copying a device pointer to a device symbol
template <typename T>
cudaError_t copyPointerToSymbol(T** symbol, T* d_ptr, const char* symbol_name)
{
    return safeCudaMemcpyToSymbol(*symbol, &d_ptr, 1 * sizeof(T), symbol_name);
}

// Macro to simplify calling copyToSymbol with the symbol name and optional count
#define CopyToSymbol(symbol, src, count) copyToSymbol(symbol, src, count, #symbol)

// Macro to simplify calling copyPointerToSymbol with the symbol name
#define CopyPointerToSymbol(symbol, d_ptr) copyPointerToSymbol(&symbol, d_ptr, #symbol)

int CUDAPrepare(int cudadev, double* beta_pole, double* lambda_pole, double* par, double cl, double Alamda_start, double Alamda_incr,
    std::vector<std::vector<double>>& ee, std::vector<std::vector<double>>& ee0, std::vector<double>& tim, 
    double Phi_0, int checkex, int ndata, struct globals& gl)
{
    // init gpu
    auto initResult = SetCUDABlockingSync(cudadev);
    if (!initResult)
    {
        fprintf(stderr, "CUDA: Error while initialising CUDA\n");
        exit(999);
    }

    cudaSetDevice(cudadev);
    // TODO: Check if this is obsolete when calling SetCUDABlockingSync()
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
    // TODO: Check if this will help to free some CPU core utilization
    //cudaSetDeviceFlags(cudaDeviceScheduleYield);

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 10020)
    try
    {
        nvmlInit();
        nvml_enabled = true;
    }
    catch (...)
    {
        nvml_enabled = false;
    }
#endif

    // determine gridDim
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, cudadev);
    if (!checkex)
    {
        auto cudaVersion = CUDA_VERSION;
        auto totalGlobalMemory = deviceProp.totalGlobalMem / 1048576;
        auto sharedMemorySm = deviceProp.sharedMemPerMultiprocessor;
        auto sharedMemoryBlock = deviceProp.sharedMemPerBlock;

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 10020)
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
#endif

        /*auto peakClk = 1;
      cudaDeviceGetAttribute(&peakClk, cudaDevAttrClockRate, cudadev);
      auto devicePeakClock = peakClk / 1024;*/

        fprintf(stderr, "CUDA version: %d\n", cudaVersion);
        fprintf(stderr, "CUDA Device number: %d\n", cudadev);
        fprintf(stderr, "CUDA Device: %s %lluMB \n", deviceProp.name, totalGlobalMemory);
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 10020)
        fprintf(stderr, "CUDA Device driver: %s\n", drv_version_str);
#endif
        fprintf(stderr, "Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        //fprintf(stderr, "Device peak clock: %d MHz\n", devicePeakClock);
        fprintf(stderr, "Shared memory per Block | per SM: %llu | %llu\n", sharedMemoryBlock, sharedMemorySm);
        fprintf(stderr, "Multiprocessors: %d\n", deviceProp.multiProcessorCount);

    }

    // NOTE: See this https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities , Table 15.
    // NOTE: Also this https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
    // NOTE: NB - Always set MaxUsedRegisters to 32 in order to achieve 100% SM occupancy (project's Configuration properties -> CUDA C/C++ -> Device)

    Cc cc(deviceProp);
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#endif

    // Maximum number of resident thread blocks per multiprocessor
    auto smxBlock = cc.GetSmxBlock();
	if(smxBlock <= 0)
	{
		fprintf(stderr, "Unsupported number of resident bloks per multiporcessor: %d!\n", smxBlock);
		exit(999);
	}

    CUDA_grid_dim = 2 * deviceProp.multiProcessorCount * smxBlock;

    if (!checkex)
    {
        fprintf(stderr, "Resident blocks per multiprocessor: %d\n", smxBlock);
        fprintf(stderr, "Grid dim (x2): %d = %d*%d\n", CUDA_grid_dim, deviceProp.multiProcessorCount * 2, smxBlock);
        fprintf(stderr, "Block dim: %d\n", CUDA_BLOCK_DIM);
    }

    //Global parameters - old approach
    //cudaMemcpyToSymbol(CUDA_beta_pole, beta_pole, sizeof(double) * (N_POLES + 1));
    //cudaMemcpyToSymbol(CUDA_lambda_pole, lambda_pole, sizeof(double) * (N_POLES + 1));
    //cudaMemcpyToSymbol(CUDA_par, par, sizeof(double) * 4);
    //cudaMemcpyToSymbol(CUDA_cl, &cl, sizeof(cl));
    //cudaMemcpyToSymbol(CUDA_Alamda_start, &Alamda_start, sizeof(Alamda_start));
    //cudaMemcpyToSymbol(CUDA_Alamda_incr, &Alamda_incr, sizeof(Alamda_incr));
    //cudaMemcpyToSymbol(CUDA_Mmax, &m_max, sizeof(m_max));
    //cudaMemcpyToSymbol(CUDA_Lmax, &l_max, sizeof(l_max));
    //cudaMemcpyToSymbol(CUDA_Phi_0, &Phi_0, sizeof(Phi_0));

    //copyToSymbol(CUDA_beta_pole, beta_pole, (N_POLES + 1));
    //copyToSymbol(CUDA_lambda_pole, lambda_pole, (N_POLES + 1));
    //copyToSymbol(CUDA_par, par, 4);
    //copyToSymbol(CUDA_cl, &cl, sizeof(cl));
    //copyToSymbol(CUDA_Alamda_start, &Alamda_start, sizeof(Alamda_start));
    //copyToSymbol(CUDA_Alamda_incr, &Alamda_incr, sizeof(Alamda_incr));
    //copyToSymbol(CUDA_Mmax, &m_max, sizeof(m_max));
    //copyToSymbol(CUDA_Lmax, &l_max, sizeof(l_max));
    //copyToSymbol(CUDA_Phi_0, &Phi_0, sizeof(Phi_0));

#if defined _DEBUG
    // Get the current heap size limit using cudaDeviceGetLimit()
    size_t limit = 0; 
    cudaError err = cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to get CUDA Malloc Heap Size Limit: " << cudaGetErrorString(err) << std::endl;
        return 1;

        // Exit with error code
    }
    std::cerr << "Default heap size limit is: " << limit / (1024 * 1024) << " MB" << std::endl;

    //limit = 1024 * 1024 * 1024; // 1GB
    //cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);

    //// Get the new heap size limit using cudaDeviceGetLimit() 
    //err = cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    //if (err != cudaSuccess)
    //{
    //    std::cerr << "Failed to get CUDA Malloc Heap Size Limit: " << cudaGetErrorString(err) << std::endl;
    //    return 1;
    //    // Exit with error code 
    //}
    //std::cout << "New heap size limit is: " << limit / (1024 * 1024) << " MB" << std::endl;

    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);
    int freeMemoryMB = static_cast<int>(std::ceil(static_cast<double>(freeMemory) / (1024.0 * 1024.0)));
    std::cerr << "Free memory:" << freeMemoryMB << " MB" << std::endl;
    int totalMemoryMB = static_cast<int>(std::ceil(static_cast<double>(totalMemory) / (1024.0 * 1024.0)));
    std::cerr << "Total memory:" << totalMemoryMB << " MB" << std::endl;
#endif

    // Using macro to achieve verbose error output
    CopyToSymbol(CUDA_beta_pole, beta_pole, N_POLES + 1);
    CopyToSymbol(CUDA_lambda_pole, lambda_pole, (N_POLES + 1));
    CopyToSymbol(CUDA_par, par, 4);
    CopyToSymbol(CUDA_cl, &cl, 1);
    CopyToSymbol(CUDA_Alamda_start, &Alamda_start, 1);
    CopyToSymbol(CUDA_Alamda_incr, &Alamda_incr, 1);
    CopyToSymbol(CUDA_Mmax, &m_max, 1);
    CopyToSymbol(CUDA_Lmax, &l_max, 1);
    CopyToSymbol(CUDA_Phi_0, &Phi_0, 1);

    //cudaMalloc(reinterpret_cast<void**>(&d_CUDA_tim), tim.size() * sizeof(double));
    //cudaMemcpy(d_CUDA_tim, tim.data(), tim.size() * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(CUDA_tim, &d_CUDA_tim, sizeof(double*));
    safeCudaMalloc(d_CUDA_tim, tim.size());
    safeMemcopy(d_CUDA_tim, tim.data(), tim.size(), cudaMemcpyHostToDevice);
    CopyPointerToSymbol(CUDA_tim, d_CUDA_tim);

    //cudaMalloc(reinterpret_cast<void**>(&pWeight), gl.Weight.size() * sizeof(double));
    //cudaMemcpy(pWeight, gl.Weight.data(), gl.Weight.size() * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(CUDA_Weight, &pWeight, sizeof(double*));
    safeCudaMalloc(pWeight, gl.Weight.size());
    safeMemcopy(pWeight, gl.Weight.data(), gl.Weight.size(), cudaMemcpyHostToDevice);
    CopyPointerToSymbol(CUDA_Weight, pWeight);
    
    auto flattened_ee = flatten2Dvector<double>(ee);
    //cudaMalloc(reinterpret_cast<void**>(&pee), flattened_ee.size() * sizeof(double));
    //cudaMemcpy(pee, flattened_ee.data(), flattened_ee.size() * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(CUDA_ee, &pee, sizeof(double*));
    safeCudaMalloc(pee, flattened_ee.size());
    safeMemcopy(pee, flattened_ee.data(), flattened_ee.size(), cudaMemcpyHostToDevice);
    CopyPointerToSymbol(CUDA_ee, pee);

    auto flattened_ee0 = flatten2Dvector<double>(ee0);
    //cudaMalloc(reinterpret_cast<void**>(&pee0), flattened_ee0.size() * sizeof(double));
    //cudaMemcpy(pee0, flattened_ee0.data(), flattened_ee0.size() * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpyToSymbol(CUDA_ee0, &pee0, sizeof(double*));
    safeCudaMalloc(pee0, flattened_ee0.size());
    safeMemcopy(pee0, flattened_ee0.data(), flattened_ee0.size(), cudaMemcpyHostToDevice);
    CopyPointerToSymbol(CUDA_ee0, pee0);

    cudaError_t _err = cudaGetLastError();
    if (_err != cudaSuccess)
    {
        std::cerr << "Failed to launch kernel: " << cudaGetErrorString(_err) << std::endl;
        return 0;
    }

#if defined _DEBUG
    // Print the total allocated size
    std::cerr << "Total allocated size: " << totalAllocatedSize << " bytes" << std::endl;
#endif

    return 1;
}

int CUDAPrecalc(int cudadev, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double* conw_r,
    int ndata, std::vector<int>& ia, int* ia_par, int* new_conw, std::vector<double>& cg_first, std::vector<double>& sig, int Numfac,
    std::vector<double>& brightness, struct globals& gl)
{
    //int* endPtr;
    int max_test_periods, iC, theEnd;
    double sum_dark_facet, ave_dark_facet;
    int i, n, m;
    int n_iter_max;
    double iter_diff_max;
    freq_result* res;
    void* pcc, * pfr, * pbrightness, * psig, * p_cg_first;

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

    // here move data to device
    cudaMemcpyToSymbol(CUDA_Ncoef, &n_coef, sizeof(n_coef));
    cudaMemcpyToSymbol(CUDA_Nphpar, &n_ph_par, sizeof(n_ph_par));
    cudaMemcpyToSymbol(CUDA_Numfac, &Numfac, sizeof(Numfac));
    m = Numfac + 1;
    cudaMemcpyToSymbol(CUDA_Numfac1, &m, sizeof(m));
    cudaMemcpyToSymbol(CUDA_n_iter_max, &n_iter_max, sizeof(n_iter_max));
    cudaMemcpyToSymbol(CUDA_n_iter_min, &n_iter_min, sizeof(n_iter_min));
    cudaMemcpyToSymbol(CUDA_ndata, &ndata, sizeof(ndata));
    cudaMemcpyToSymbol(CUDA_iter_diff_max, &iter_diff_max, sizeof(iter_diff_max));
    cudaMemcpyToSymbol(CUDA_conw_r, &conw_r, sizeof(conw_r));
    cudaMemcpyToSymbol(CUDA_Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
    cudaMemcpyToSymbol(CUDA_Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
    cudaMemcpyToSymbol(CUDA_Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
    cudaMemcpyToSymbol(CUDA_Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
    cudaMemcpyToSymbol(CUDA_Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
    cudaMemcpyToSymbol(CUDA_Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
    cudaMemcpyToSymbol(CUDA_Is_Precalc, &isPrecalc, sizeof isPrecalc, 0, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(CUDA_ia, ia.data(), ia.size() * sizeof(int));

    err = cudaMalloc(&p_cg_first, cg_first.size() * sizeof(double));
    err = cudaMemcpy(p_cg_first, cg_first.data(), cg_first.size() * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(CUDA_cg_first, &p_cg_first, sizeof(p_cg_first));

    err = cudaMalloc(&pbrightness, brightness.size() * sizeof(double));
    err = cudaMemcpy(pbrightness, brightness.data(), brightness.size() * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(CUDA_brightness, &pbrightness, sizeof(pbrightness));

    err = cudaMalloc(&psig, sig.size() * sizeof(double));
    err = cudaMemcpy(psig, sig.data(), sig.size() * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(CUDA_sig, &psig, sizeof(psig));

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
    cudaMemcpyToSymbol(CUDA_ma, &ma, sizeof(ma));
    cudaMemcpyToSymbol(CUDA_mfit, &lmfit, sizeof(lmfit));
    m = lmfit + 1;
    cudaMemcpyToSymbol(CUDA_mfit1, &m, sizeof(m));
    cudaMemcpyToSymbol(CUDA_lastma, &llastma, sizeof(llastma));
    cudaMemcpyToSymbol(CUDA_lastone, &llastone, sizeof(llastone));
    m = ma - 2 - n_ph_par;
    cudaMemcpyToSymbol(CUDA_ncoef0, &m, sizeof(m));

    int CUDA_Grid_dim_precalc = CUDA_grid_dim;
    if (max_test_periods < CUDA_Grid_dim_precalc)
    {
        CUDA_Grid_dim_precalc = max_test_periods;
        //#ifdef _DEBUG
        //		fprintf(stderr, "CUDA_Grid_dim_precalc = %d\n", CUDA_Grid_dim_precalc);
        //#endif
    }

    err = cudaMalloc(&pcc, CUDA_Grid_dim_precalc * sizeof(freq_context));
    cudaMemcpyToSymbol(CUDA_CC, &pcc, sizeof(pcc));
    err = cudaMalloc(&pfr, CUDA_Grid_dim_precalc * sizeof(freq_result));
    cudaMemcpyToSymbol(CUDA_FR, &pfr, sizeof(pfr));

    m = (Numfac + 1) * (n_coef + 1);
    cudaMemcpyToSymbol(CUDA_Dg_block, &m, sizeof(m));

    double* pa, * pg, * pal, * pco, * pdytemp, * pytemp;
    double* pe_1, * pe_2, * pe_3, * pe0_1, * pe0_2, * pe0_3;
    double* pjp_Scale, * pjp_dphp_1, * pjp_dphp_2, * pjp_dphp_3;
    double* pde, * pde0;

    err = cudaMalloc(&pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
    cudaMemcpyToSymbol(CUDA_Area, &pa, sizeof(pa));
    err = cudaMalloc(&pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
    err = cudaMemcpyToSymbol(CUDA_Dg, &pg, sizeof(pg));
    err = cudaMalloc(&pal, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
    err = cudaMalloc(&pco, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
    err = cudaMalloc(&pdytemp, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * (ma + 1) * sizeof(double));
    err = cudaMalloc(&pytemp, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe_1, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe_2, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe_3, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe0_1, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe0_2, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe0_3, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pjp_Scale, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pjp_dphp_1, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pjp_dphp_2, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pjp_dphp_3, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pde, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * 4 * 4 * sizeof(double));
    err = cudaMalloc(&pde0, CUDA_Grid_dim_precalc * (gl.maxLcPoints + 1) * 4 * 4 * sizeof(double));

    for (m = 0; m < CUDA_Grid_dim_precalc; m++)
    {
        freq_context ps;
        ps.Area = &pa[m * (Numfac + 1)];                            // 1st pointer
        ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];               // 2nd...
        ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];             // 3
        ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];             // 4
        ps.dytemp = &pdytemp[m * (gl.maxLcPoints + 1) * (ma + 1)];  // 5
        ps.ytemp = &pytemp[m * (gl.maxLcPoints + 1)];               // 6
        ps.e_1 = &pe_1[m * (gl.maxLcPoints + 1)];                   // 7
        ps.e_2 = &pe_2[m * (gl.maxLcPoints + 1)];                   // 8
        ps.e_3 = &pe_3[m * (gl.maxLcPoints + 1)];                   // 9
        ps.e0_1 = &pe0_1[m * (gl.maxLcPoints + 1)];                 // 10
        ps.e0_2 = &pe0_2[m * (gl.maxLcPoints + 1)];                 // 11
        ps.e0_3 = &pe0_3[m * (gl.maxLcPoints + 1)];                 // 12
        ps.jp_Scale = &pjp_Scale[m * (gl.maxLcPoints + 1)];         // 13
        ps.jp_dphp_1 = &pjp_dphp_1[m * (gl.maxLcPoints + 1)];       // 14
        ps.jp_dphp_2 = &pjp_dphp_2[m * (gl.maxLcPoints + 1)];       // 15
        ps.jp_dphp_3 = &pjp_dphp_3[m * (gl.maxLcPoints + 1)];       // 16
        ps.de = &pde[m * (gl.maxLcPoints + 1) * 4 * 4];             // 17
        ps.de0 = &pde0[m * (gl.maxLcPoints + 1) * 4 * 4];           // 18

        freq_context* pt = &((freq_context*)pcc)[m];
        err = cudaMemcpy(pt, &ps, sizeof(void*) * 18, cudaMemcpyHostToDevice);   // <- 18 pointers!
        // NOTE: We could use the following approach that will give us more readability, flexibility and safety, but it may have a significant performance impact.
        // TODO: So, this needs to be analyzed using Profiler
        //err = cudaMemcpy(pt, &ps, sizeof(freq_context), cudaMemcpyHostToDevice);
    }

    res = static_cast<freq_result*>(malloc(CUDA_Grid_dim_precalc * sizeof(freq_result)));

    for (n = 1; n <= max_test_periods; n += CUDA_Grid_dim_precalc)
    {
        CudaCalculatePrepare<<<CUDA_Grid_dim_precalc, 1>>>(n, max_test_periods, freq_start, freq_step);
        //err = cudaThreadSynchronize();
        err = cudaDeviceSynchronize();

        for (m = 1; m <= N_POLES; m++)
        {
            //zero global End signal
            theEnd = 0;
            cudaMemcpyToSymbol(CUDA_End, &theEnd, sizeof(theEnd), 0, cudaMemcpyHostToDevice);
            //cudaGetSymbolAddress((void**)&endPtr, CUDA_End);
            //
            CudaCalculatePreparePole<<<CUDA_Grid_dim_precalc, 1>>>(m);
            //
#ifdef _DEBUG
            printf(".");
#endif
            auto count = 0;
            while (!theEnd)
            {
                count++;
                if (count > 51)
                {
                    std::cerr << "CUDA Precalc routine went out of bounds!" << std::endl;
                    CUDAGlobalsFree();
                    // TODO: CUDALocalFree();
                    exit(999);
                }

                CudaCalculateIter1Begin<<<CUDA_Grid_dim_precalc, 1>>>();
                //mrqcof
                CudaCalculateIter1Mrqcof1Start<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>();
                for (iC = 1; iC < gl.Lcurves; iC++)
                {
                    CudaCalculateIter1Mrqcof1Matrix<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Lpoints[iC]);
                    CudaCalculateIter1Mrqcof1Curve1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Inrel[iC], gl.Lpoints[iC]);
                    CudaCalculateIter1Mrqcof1Curve2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Inrel[iC], gl.Lpoints[iC]);
                }
                CudaCalculateIter1Mrqcof1Curve1Last<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Inrel[gl.Lcurves], gl.Lpoints[gl.Lcurves]);
                CudaCalculateIter1Mrqcof1Curve2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Inrel[gl.Lcurves], gl.Lpoints[gl.Lcurves]);
                CudaCalculateIter1Mrqcof1End<<<CUDA_Grid_dim_precalc, 1>>>();
                //mrqcof
                CudaCalculateIter1Mrqmin1End<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>();
                //mrqcof
                CudaCalculateIter1Mrqcof2Start<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>();
                for (iC = 1; iC < gl.Lcurves; iC++)
                {
                    CudaCalculateIter1Mrqcof2Matrix<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Lpoints[iC]);
                    CudaCalculateIter1Mrqcof2Curve1<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Inrel[iC], gl.Lpoints[iC]);
                    CudaCalculateIter1Mrqcof2Curve2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Inrel[iC], gl.Lpoints[iC]);
                }
                CudaCalculateIter1Mrqcof2Curve1Last<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Inrel[gl.Lcurves], gl.Lpoints[gl.Lcurves]);
                CudaCalculateIter1Mrqcof2Curve2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>(gl.Inrel[gl.Lcurves], gl.Lpoints[gl.Lcurves]);
                CudaCalculateIter1Mrqcof2End<<<CUDA_Grid_dim_precalc, 1>>>();
                //mrqcof
                CudaCalculateIter1Mrqmin2End<<<CUDA_Grid_dim_precalc, 1>>>();
                CudaCalculateIter2<<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM>>>();
                //err=cudaThreadSynchronize(); memcpy is synchro itself
                err = cudaDeviceSynchronize();
                //cudaMemcpy(&theEnd, endPtr, sizeof(theEnd), cudaMemcpyDeviceToHost);
                cudaMemcpyFromSymbolAsync(&theEnd, CUDA_End, sizeof theEnd, 0, cudaMemcpyDeviceToHost);
                theEnd = theEnd == CUDA_Grid_dim_precalc;

                //break;//debug
            }
            CudaCalculateFinishPole<<<CUDA_Grid_dim_precalc, 1>>>();
            //err = cudaThreadSynchronize();
            err = cudaDeviceSynchronize();
            //			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_Grid_dim_precalc);
            //			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_Grid_dim_precalc);
            //break; //debug
        }
        printf("\n");

        CudaCalculateFinish<<<CUDA_Grid_dim_precalc, 1>>>();
        //err=cudaThreadSynchronize(); memcpy is synchro itself

        //read results here
        err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_Grid_dim_precalc, cudaMemcpyDeviceToHost);

        for (m = 1; m <= CUDA_Grid_dim_precalc; m++)
        {
            if (res[m - 1].isReported == 1)
                sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
        }
    } /* period loop */

    isPrecalc = 0;
    cudaMemcpyToSymbol(CUDA_Is_Precalc, &isPrecalc, sizeof(isPrecalc), 0, cudaMemcpyHostToDevice);

    cudaFree(pjp_Scale);
    cudaFree(pjp_dphp_1);
    cudaFree(pjp_dphp_2);
    cudaFree(pjp_dphp_3);
    cudaFree(pde);
    cudaFree(pde0);
    cudaFree(pe_1);
    cudaFree(pe_2);
    cudaFree(pe_3);
    cudaFree(pe0_1);
    cudaFree(pe0_2);
    cudaFree(pe0_3);
    cudaFree(pa);
    cudaFree(pg);
    cudaFree(pal);
    cudaFree(pco);
    cudaFree(pdytemp);
    cudaFree(pytemp);
    cudaFree(pcc);
    cudaFree(pfr);
    cudaFree(pbrightness);
    cudaFree(psig);
    cudaFree(p_cg_first);

    free(res);

    ave_dark_facet = sum_dark_facet / max_test_periods;

    if (ave_dark_facet < 1.0)
        *new_conw = 1; /* new correct conwexity weight */
    if (ave_dark_facet >= 1.0)
        *conw_r = *conw_r * 2; /* still not good */

    return 1;
}


int CUDAStart(int cudadev, int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
    int ndata, std::vector<int>& ia, int* ia_par, std::vector<double>& cg_first, MFILE& mf, double escl, std::vector<double>& sig, int Numfac, 
    std::vector<double>& brightness, struct globals& gl)
{
    int retval, i, n, m, iC, n_max = (int)((freq_start - freq_end) / freq_step) + 1;
    int n_iter_max, theEnd, LinesWritten;
    double iter_diff_max;
    freq_result* res;
    void* pcc, * pfr, * pbrightness, * psig, * p_cg_first;
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

    // here move data to device
    cudaMemcpyToSymbol(CUDA_Ncoef, &n_coef, sizeof(n_coef));
    cudaMemcpyToSymbol(CUDA_Nphpar, &n_ph_par, sizeof(n_ph_par));
    cudaMemcpyToSymbol(CUDA_Numfac, &Numfac, sizeof(Numfac));
    m = Numfac + 1;
    cudaMemcpyToSymbol(CUDA_Numfac1, &m, sizeof(m));
    cudaMemcpyToSymbol(CUDA_n_iter_max, &n_iter_max, sizeof(n_iter_max));
    cudaMemcpyToSymbol(CUDA_n_iter_min, &n_iter_min, sizeof(n_iter_min));
    cudaMemcpyToSymbol(CUDA_ndata, &ndata, sizeof(ndata));
    cudaMemcpyToSymbol(CUDA_iter_diff_max, &iter_diff_max, sizeof(iter_diff_max));
    cudaMemcpyToSymbol(CUDA_conw_r, &conw_r, sizeof(conw_r));
    cudaMemcpyToSymbol(CUDA_Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);
    cudaMemcpyToSymbol(CUDA_Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
    cudaMemcpyToSymbol(CUDA_Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
    cudaMemcpyToSymbol(CUDA_Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
    cudaMemcpyToSymbol(CUDA_Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
    cudaMemcpyToSymbol(CUDA_Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
    cudaMemcpyToSymbol(CUDA_ia, ia.data(), ia.size() * sizeof(int));

    err = cudaMalloc(&p_cg_first, cg_first.size() * sizeof(double));
    err = cudaMemcpy(p_cg_first, cg_first.data(), cg_first.size() * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(CUDA_cg_first, &p_cg_first, sizeof(p_cg_first));

    err = cudaMalloc(&pbrightness, brightness.size() * sizeof(double));
    err = cudaMemcpy(pbrightness, brightness.data(), brightness.size() * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(CUDA_brightness, &pbrightness, sizeof(pbrightness));

    err = cudaMalloc(&psig, sig.size() * sizeof(double));
    err = cudaMemcpy(psig, sig.data(), sig.size() * sizeof(double), cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(CUDA_sig, &psig, sizeof(psig));

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
    cudaMemcpyToSymbol(CUDA_ma, &ma, sizeof(ma));
    cudaMemcpyToSymbol(CUDA_mfit, &lmfit, sizeof(lmfit));
    m = lmfit + 1;
    cudaMemcpyToSymbol(CUDA_mfit1, &m, sizeof(m));
    cudaMemcpyToSymbol(CUDA_lastma, &llastma, sizeof(llastma));
    cudaMemcpyToSymbol(CUDA_lastone, &llastone, sizeof(llastone));
    m = ma - 2 - n_ph_par;
    cudaMemcpyToSymbol(CUDA_ncoef0, &m, sizeof(m));

    err = cudaMalloc(&pcc, CUDA_grid_dim * sizeof(freq_context));
    cudaMemcpyToSymbol(CUDA_CC, &pcc, sizeof(pcc));
    err = cudaMalloc(&pfr, CUDA_grid_dim * sizeof(freq_result));
    cudaMemcpyToSymbol(CUDA_FR, &pfr, sizeof(pfr));

    m = (Numfac + 1) * (n_coef + 1);
    cudaMemcpyToSymbol(CUDA_Dg_block, &m, sizeof(m));

    double* pa, * pg, * pal, * pco, * pdytemp, * pytemp;
    double* pe_1, * pe_2, * pe_3, * pe0_1, * pe0_2, * pe0_3;
    double* pjp_Scale, * pjp_dphp_1, * pjp_dphp_2, * pjp_dphp_3;
    double* pde, * pde0;

    err = cudaMalloc(&pa, CUDA_grid_dim * (Numfac + 1) * sizeof(double));
    err = cudaMemcpyToSymbol(CUDA_Area, &pa, sizeof(pa));

    err = cudaMalloc(&pg, CUDA_grid_dim * (Numfac + 1) * (n_coef + 1) * sizeof(double));
    err = cudaMemcpyToSymbol(CUDA_Dg, &pg, sizeof(pg));

    err = cudaMalloc(&pal, CUDA_grid_dim * (lmfit + 1) * (lmfit + 1) * sizeof(double));
    err = cudaMalloc(&pco, CUDA_grid_dim * (lmfit + 1) * (lmfit + 1) * sizeof(double));
    err = cudaMalloc(&pdytemp, CUDA_grid_dim * (gl.maxLcPoints + 1) * (ma + 1) * sizeof(double));
    err = cudaMalloc(&pytemp, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe_1, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe_2, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe_3, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe0_1, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe0_2, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pe0_3, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pjp_Scale, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pjp_dphp_1, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pjp_dphp_2, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pjp_dphp_3, CUDA_grid_dim * (gl.maxLcPoints + 1) * sizeof(double));
    err = cudaMalloc(&pde, CUDA_grid_dim * (gl.maxLcPoints + 1) * 4 * 4 * sizeof(double));
    err = cudaMalloc(&pde0, CUDA_grid_dim * (gl.maxLcPoints + 1) * 4 * 4 * sizeof(double));

    for (m = 0; m < CUDA_grid_dim; m++)
    {
        freq_context ps;
        ps.Area = &pa[m * (Numfac + 1)];
        ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];
        ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];
        ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];
        ps.dytemp = &pdytemp[m * (gl.maxLcPoints + 1) * (ma + 1)];
        ps.ytemp = &pytemp[m * (gl.maxLcPoints + 1)];
        ps.e_1 = &pe_1[m * (gl.maxLcPoints + 1)];
        ps.e_2 = &pe_2[m * (gl.maxLcPoints + 1)];
        ps.e_3 = &pe_3[m * (gl.maxLcPoints + 1)];
        ps.e0_1 = &pe0_1[m * (gl.maxLcPoints + 1)];
        ps.e0_2 = &pe0_2[m * (gl.maxLcPoints + 1)];
        ps.e0_3 = &pe0_3[m * (gl.maxLcPoints + 1)];
        ps.jp_Scale = &pjp_Scale[m * (gl.maxLcPoints + 1)];
        ps.jp_dphp_1 = &pjp_dphp_1[m * (gl.maxLcPoints + 1)];
        ps.jp_dphp_2 = &pjp_dphp_2[m * (gl.maxLcPoints + 1)];
        ps.jp_dphp_3 = &pjp_dphp_3[m * (gl.maxLcPoints + 1)];
        ps.de = &pde[m * (gl.maxLcPoints + 1) * 4 * 4];              
        ps.de0 = &pde0[m * (gl.maxLcPoints + 1) * 4 * 4];               

        freq_context* pt = &((freq_context*)pcc)[m];
        err = cudaMemcpy(pt, &ps, sizeof(void*) * 18, cudaMemcpyHostToDevice);
    }

    res = static_cast<freq_result*>(malloc(CUDA_grid_dim * sizeof(freq_result)));

    //int firstreport = 0;//beta debug
    auto oldFractionDone = 0.0001;

    for (n = n_start_from; n <= n_max; n += CUDA_grid_dim)
    {
        auto fractionDone = (double)n / (double)n_max;
        //boinc_fraction_done(fractionDone);

        //#if _DEBUG
        //		float fraction = fractionDone * 100;
        //		std::time_t t = std::time(nullptr);   // get time now
        //		std::tm* now = std::localtime(&t);
        //
        //		printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
        //		fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction);
        //#endif

        CudaCalculatePrepare<<<CUDA_grid_dim, 1>>>(n, n_max, freq_start, freq_step);
        //err = cudaThreadSynchronize();
        //err = cudaDeviceSynchronize();

        for (m = 1; m <= N_POLES; m++)
        {
            auto mid = fractionDone - oldFractionDone;
            auto inner = mid / static_cast<double>(N_POLES) * m;
            //printf("mid: %.4f, inner: %.4f\n", mid, inner);
            auto fractionDone2 = oldFractionDone + inner;
            boinc_fraction_done(fractionDone2);

#ifdef _DEBUG
            auto fraction2 = fractionDone2 * 100;
            //float fraction = fractionDone * 100;
            std::time_t t = std::time(nullptr);   // get time now
            std::tm* now = std::localtime(&t);

            printf("%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
            fprintf(stderr, "%02d:%02d:%02d | Fraction done: %.4f%%\n", now->tm_hour, now->tm_min, now->tm_sec, fraction2);
#endif
            //zero global End signal
            theEnd = 0;
            cudaMemcpyToSymbol(CUDA_End, &theEnd, sizeof(theEnd));
            //
            CudaCalculatePreparePole<<<CUDA_grid_dim, 1>>>(m);
            //
            while (!theEnd)
            {
                CudaCalculateIter1Begin<<<CUDA_grid_dim, 1>>>();
                //mrqcof
                CudaCalculateIter1Mrqcof1Start<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>();
                for (iC = 1; iC < gl.Lcurves; iC++)
                {
                    CudaCalculateIter1Mrqcof1Matrix<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Lpoints[iC]);
                    CudaCalculateIter1Mrqcof1Curve1<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Inrel[iC], gl.Lpoints[iC]);
                    CudaCalculateIter1Mrqcof1Curve2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Inrel[iC], gl.Lpoints[iC]);
                }
                CudaCalculateIter1Mrqcof1Curve1Last<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Inrel[gl.Lcurves], gl.Lpoints[gl.Lcurves]);
                CudaCalculateIter1Mrqcof1Curve2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Inrel[gl.Lcurves], gl.Lpoints[gl.Lcurves]);
                CudaCalculateIter1Mrqcof1End<<<CUDA_grid_dim, 1>>>();
                //mrqcof
                CudaCalculateIter1Mrqmin1End<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>();

                /*if (!if_freq_measured && nvml_enabled && n == n_start_from && m == N_POLES)
              {
              GetPeakClock(cudadev);
              }*/

              //mrqcof
                CudaCalculateIter1Mrqcof2Start<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>();
                for (iC = 1; iC < gl.Lcurves; iC++)
                {
                    CudaCalculateIter1Mrqcof2Matrix<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Lpoints[iC]);
                    CudaCalculateIter1Mrqcof2Curve1<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Inrel[iC], gl.Lpoints[iC]);
                    CudaCalculateIter1Mrqcof2Curve2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Inrel[iC], gl.Lpoints[iC]);
                }
                CudaCalculateIter1Mrqcof2Curve1Last<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Inrel[gl.Lcurves], gl.Lpoints[gl.Lcurves]);
                CudaCalculateIter1Mrqcof2Curve2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>(gl.Inrel[gl.Lcurves], gl.Lpoints[gl.Lcurves]);
                CudaCalculateIter1Mrqcof2End<<<CUDA_grid_dim, 1>>>();
                //mrqcof
                CudaCalculateIter1Mrqmin2End<<<CUDA_grid_dim, 1>>>();
                CudaCalculateIter2<<<CUDA_grid_dim, CUDA_BLOCK_DIM>>>();
                //err=cudaThreadSynchronize(); memcpy is synchro itself
                //err = cudaDeviceSynchronize();
                cudaMemcpyFromSymbolAsync(&theEnd, CUDA_End, sizeof theEnd, 0, cudaMemcpyDeviceToHost);
                //cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd));
                err = cudaDeviceSynchronize();
                theEnd = theEnd == CUDA_grid_dim;

                //break;//debug
            }

            CudaCalculateFinishPole<<<CUDA_grid_dim, 1>>>();
            //err = cudaThreadSynchronize();
            //err = cudaDeviceSynchronize();
            //			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_grid_dim);
            //			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_grid_dim);
            //break; //debug
        }

        CudaCalculateFinish<<<CUDA_grid_dim, 1>>>();
        //err=cudaThreadSynchronize(); memcpy is synchro itself

        //read results here synchronously
        err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_grid_dim, cudaMemcpyDeviceToHost);

        oldFractionDone = fractionDone;
        LinesWritten = 0;
        for (m = 1; m <= CUDA_grid_dim; m++)
        {
            if (res[m - 1].isReported == 1)
            {
                LinesWritten++;
                double dark_best = n == 1 && m == 1
                    ? conw_r * escl * escl
                    : res[m - 1].dark_best;

                /* output file */
                mf.printf("%.8f  %.6f  %.6f %4.1f %4.0f %4.0f\n", 24 * res[m - 1].per_best, res[m - 1].dev_best, res[m - 1].dev_best * res[m - 1].dev_best * (ndata - 3), dark_best, round(res[m - 1].la_best), round(res[m - 1].be_best));
            }
        }

        if (boinc_time_to_checkpoint() || boinc_is_standalone())
        {
            retval = DoCheckpoint(mf, (n - 1) + LinesWritten, 1, conw_r); //zero lines
            if (retval)
            {
                fprintf(stderr, "%s APP: period_search checkpoint failed %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval);
                exit(retval);
            }
            boinc_checkpoint_completed();
        }

        //		break;//debug
        printf("\n");
        fflush(stdout);
    } /* period loop */

    printf("\n");

    cudaFree(pjp_Scale);
    cudaFree(pjp_dphp_1);
    cudaFree(pjp_dphp_2);
    cudaFree(pjp_dphp_3);
    cudaFree(pde);
    cudaFree(pde0);
    cudaFree(pe_1);
    cudaFree(pe_2);
    cudaFree(pe_3);
    cudaFree(pe0_1);
    cudaFree(pe0_2);
    cudaFree(pe0_3);
    cudaFree(pa);
    cudaFree(pg);
    cudaFree(pal);
    cudaFree(pco);
    cudaFree(pdytemp);
    cudaFree(pytemp);
    cudaFree(pcc);
    cudaFree(pfr);
    cudaFree(pbrightness);
    cudaFree(psig);
    cudaFree(p_cg_first);

    free(res);

    //nvmlShutdown();

    return 1;
}
