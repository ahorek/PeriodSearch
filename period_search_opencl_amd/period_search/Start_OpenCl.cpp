#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
//#include <CL/cl.h>
#include <CL/cl.hpp>

//#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#elif defined(cl_amd_fp64)
//#pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#else
//#error "Double precision floating point not supported by OpenCL implementation."
//#endif

//#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <algorithm>
#include "globals.h"
#include "declarations.hpp"
#include "declarations_OpenCl.h"
#include "Start_OpenCl.h"


#ifdef _WIN32
//#include "boinc_win.h"
#include <Shlwapi.h>
#endif


#include "Start_OpenCL.h"


using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;

// NOTE: global to all freq

vector<cl::Platform> platforms;
vector<cl::Device> devices;
cl::Context context;
cl::Program program;
cl::Program programIter1Mrqcof1Start;
cl::CommandQueue queue;


int CUDA_grid_dim;
//double CUDA_cl;
double logCl;
double aLambda_start;


cl_int2* texWeight;
cl_int2* texArea;
cl_int2* texDg;
cl_int2* texbrightness;
cl_int2* texsig;

//cl::Image1D texWeight;  //NOTE: CUDA's 'texture<int2, 1>{}' structure equivalent
//cl::Image1D texArea;
//cl::Image1D texDg;
//cl::Buffer CUDA_cg_first;
cl::Buffer CUDA_lambda_pole;
cl::Buffer CUDA_beta_pole;
cl::Buffer CUDA_par;
//void* pFa;

// NOTE: global to one thread
#ifdef __GNUC__
freq_result* CUDA_FR __attribute__((aligned(4)));
FuncArrays* Fa __attribute__((aligned(4)));
#else
__declspec(align(4)) freq_result* CUDA_FR;
__declspec(align(4)) FuncArrays* Fa;
#endif

double* pee, * pee0, * pWeight;

cl_int ClPrepare(int deviceId, double* beta_pole, double* lambda_pole, double* par, double ia, double Alambda_start, double Alambda_incr,
	double ee[][3], double ee0[][3], double* tim, double Phi_0, int checkex, int ndata)
{
	Fa = static_cast<FuncArrays*>(malloc(sizeof(FuncArrays)));
	
	try {
		cl::Platform::get(&platforms);
		vector<cl::Platform>::iterator iter;
		for (iter = platforms.begin(); iter != platforms.end(); ++iter)
		{
			auto name = (*iter).getInfo<CL_PLATFORM_NAME>();
			auto vendor = (*iter).getInfo<CL_PLATFORM_VENDOR>();
			std::cerr << "Platform vendor: " << vendor << endl;
			if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc."))
			{
				break;
			}
			if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "NVIDIA Corporation"))
			{
				break;
			}
			//if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Intel(R) Corporation"))
			//{
			//	break;
			//}
		}



		// Create an OpenCL context
		cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, cl_context_properties((*iter)()), 0 };
		context = cl::Context(CL_DEVICE_TYPE_GPU, cps);

		// Detect OpenCL devices
		devices = context.getInfo<CL_CONTEXT_DEVICES>();
		deviceId = 0;
		const auto device = devices[deviceId];
		const auto openClVersion = device.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
		const auto clDeviceVersion = device.getInfo<CL_DEVICE_VERSION>();
		const auto clDeviceExtensionCapabilities = device.getInfo<CL_DEVICE_EXECUTION_CAPABILITIES>();
		const auto clDeviceGlobalMemSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
		const auto clDeviceLocalMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
		const auto clDeviceMaxConstantArgs = device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>();
		const auto clDeviceMaxConstantBufferSize = device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
		const auto clDeviceMaxParameterSize = device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();
		const auto clDeviceMaxMemAllocSize = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
		const auto deviceName = device.getInfo<CL_DEVICE_NAME>();
		const auto deviceMaxWorkItemDims = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>();
		const auto clGlobalMemory = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
		const auto globalMemory = clGlobalMemory / 1048576;
		const auto msCount = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
		const auto block = device.getInfo<CL_DEVICE_MAX_SAMPLERS>();
		const auto deviceExtensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
		const auto devMaxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
		const auto devMaxWorkItemSizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		bool is64CUDA = deviceExtensions.find("cl_khr_fp64") != std::string::npos;
		bool is64AMD = deviceExtensions.find("cl_amd_fp64") == std::string::npos;
		//auto doesNotSupportsFp64 = (deviceExtensions.find("cl_khr_fp64") == std::string::npos) || (deviceExtensions.find("cl_amd_fp64") == std::string::npos);
		//if(doesNotSupportsFp64)
		if(!is64CUDA || !is64AMD)
		{
			fprintf(stderr, "Double precision floating point not supported by OpenCL implementation. Exiting...\n");
			exit(-1);
		}

		std::cerr << "OpenCL version: " << openClVersion << " | " << clDeviceVersion << endl;
		std::cerr << "OpenCL Device number : " << deviceId << endl;
		std::cerr << "OpenCl Device name: " << deviceName << " " << globalMemory << "MB" << endl;
		std::cerr << "Multiprocessors: " << msCount << endl;
		std::cerr << "Max Samplers: " << block << endl;
		std::cerr << "Max work item dimensions: " << deviceMaxWorkItemDims << endl;
#ifdef _DEBUG
		std::cerr << "Debug info:" << endl;
		std::cerr << "CL_DEVICE_EXTENSIONS: " << deviceExtensions << endl;
		std::cerr << "CL_DEVICE_GLOBAL_MEM_SIZE: " << clDeviceGlobalMemSize << " B" << endl;
		std::cerr << "CL_DEVICE_LOCAL_MEM_SIZE: " << clDeviceLocalMemSize << " B" << endl;
		std::cerr << "CL_DEVICE_MAX_CONSTANT_ARGS: " << clDeviceMaxConstantArgs << endl;
		std::cerr << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << clDeviceMaxConstantBufferSize << " B" << endl;
		std::cerr << "CL_DEVICE_MAX_PARAMETER_SIZE: " << clDeviceMaxParameterSize << " B" << endl;
		std::cerr << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << clDeviceMaxMemAllocSize << " B" << endl;
		std::cerr << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << devMaxWorkGroupSize << endl;
		std::cerr << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << devMaxWorkItemSizes[0] << " | " << devMaxWorkItemSizes[1] << " | " << devMaxWorkItemSizes[2] << endl;
#endif


		// TODO: Calculate this:
		auto SMXBlock = block; // 128;
		CUDA_grid_dim = msCount * SMXBlock;
		cl_int* err = nullptr;
		CUDA_lambda_pole = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * (N_POLES + 1), lambda_pole, err);
		CUDA_beta_pole = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * (N_POLES + 1), beta_pole, err);
		//CUDA_par = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * 4, par, err);
				
		memcpy((*Fa).par, par, sizeof(double) * 4);
		memcpy((*Fa).ia, &ia, sizeof(double));
		
		logCl = log(ia);
		(*Fa).logCl = log(ia);
		//auto clCl = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof cl, &cl, err);
		aLambda_start = Alambda_start;
		//auto clAlambdaStart = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Alambda_start, &Alambda_start, err);
		auto clAlambdaIncr = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Alambda_incr, &Alambda_incr, err);
		(*Fa).Mmax = m_max;
		(*Fa).Lmax = l_max;
		//auto clMmax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m_max, &m_max, err);
		//auto clLmax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof l_max, &l_max, err);
		//auto clTim = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_OBS + 1), tim, err);
		//auto clPhi0 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Phi_0, &Phi_0);
		(*Fa).Phi_0 = Phi_0;

		queue = cl::CommandQueue(context, devices[0]);

		auto pWsize = (ndata + 3 + 1) * sizeof(double);
		auto pWeightBuf = cl::Buffer(context, CL_MEM_READ_ONLY, pWsize, err);
		double* pWeight; //???
		//queue.enqueueWriteBuffer(pWeightBuf, CL_BLOCKING, 0, pWsize, texWeight);
		// texWeight = &pWeight; ??

		/*res = cudaMalloc(&pWeight, (ndata + 3 + 1) * sizeof(double));
		res = cudaMemcpy(pWeight, weight, (ndata + 3 + 1) * sizeof(double), cudaMemcpyHostToDevice);
		res = cudaBindTexture(0, texWeight, pWeight, (ndata + 3 + 1) * sizeof(double)); */

		auto pEeSize = (ndata + 1) * 3 * sizeof(double);
		//auto pEeBuf = cl::Buffer(context, CL_MEM_READ_ONLY, pEeSize, err);
		//queue.enqueueWriteBuffer(pEeBuf, CL_BLOCKING, 0, pEeSize, ee);

		//auto pee = malloc((MAX_N_OBS + 1) * 3 * sizeof(double));
		//memcpy(pee, &ee, (MAX_N_OBS + 1) * 3 * sizeof(double));

		memcpy((*Fa).ee, &ee[0], (MAX_N_OBS + 1) * sizeof(double)); 
		//memcpy((*Fa).ee, &ee[0], (ndata + 3 + 1) * sizeof(double)); // (MAX_N_OBS + 1) * 3 * sizeof(double));
		memcpy((*Fa).ee0, &ee0[0], (MAX_N_OBS + 1) * 3 * sizeof(double));
		//memcpy((*Fa).ee0, &ee0[0], (ndata + 3 + 1) * sizeof(double));
		memcpy((*Fa).tim, tim , sizeof(double) * (MAX_N_OBS + 1));

		/*for(int tt = 0; tt < 128; tt++)
		{
			printf("%.6f\t%.6f\n", ee[tt][0], (*Fa).ee[tt][0]);
		}*/


		/*
		res = cudaMalloc(&pee, (ndata + 1) * 3 * sizeof(double));
		res = cudaMemcpy(pee, ee, (ndata + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
		res = cudaMemcpyToSymbol(CUDA_ee, &pee, sizeof(void*));0

		res = cudaMalloc(&pee0, (ndata + 1) * 3 * sizeof(double));
		res = cudaMemcpy(pee0, ee0, (ndata + 1) * 3 * sizeof(double), cudaMemcpyHostToDevice);
		res = cudaMemcpyToSymbol(CUDA_ee0, &pee0, sizeof(void*));

		if (res == cudaSuccess) return 1; else return 0;*/
		if (err)
		{
			return (cl_int)*err;
		}

		return 0;
	}
	catch (cl::Error& err)
	{
		// Catch OpenCL errors and print log if it is a build error
		cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
		cout << "ERROR: " << err.what() << "(" << err.err() << ")" << endl;
		if (err.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			const auto str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
			cout << "Program Info: " << str << endl;
		}
		//cleanupHost();
		return 1;
	}
	catch (string& msg)
	{
		cerr << "Exception caught in main(): " << msg << endl;
		//cleanupHost();
		return 1;
	}
}

void PrintFreqResult(const int maxItterator, void* pcc2, void* pfr)
{
	for (auto l = 0; l < maxItterator; l++)
	{
		const auto freq = static_cast<freq_context2*>(pcc2)[l].freq;
		const auto la_best = static_cast<freq_result*>(pfr)[l].la_best;
		std::cerr << "freq[" << l << "] = " << freq << " | la_best[" << l << "] = " << la_best << std::endl;
	}
}

cl_int ClPrecalc(double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double* conw_r,
	int ndata, int* ia, int* ia_par, int* new_conw, double* cg_first, double* sig, int Numfac, double* brightness)
{
	auto blockDim = BLOCK_DIM;
	int max_test_periods, iC;
	int theEnd = 0;
	double sum_dark_facet, ave_dark_facet;
	int i, n, m;
	auto n_max = static_cast<int>((freq_start - freq_end) / freq_step) + 1;
	int n_iter_max;
	double iter_diff_max;
	freq_result* res;
	//freq_context2* pcc2;
	void* pcc2; // *pcc
	void* pfr, * pbrightness, * psig;

	max_test_periods = 10;
	sum_dark_facet = 0.0;
	ave_dark_facet = 0.0;

	if (n_max < max_test_periods)
		max_test_periods = n_max;

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

	cl_int* err = nullptr;
	//cudaError_t err;

	(*Fa).Ncoef = n_coef;
	(*Fa).Nphpar = n_ph_par;
	(*Fa).Numfac = Numfac;
	m = Numfac + 1;
	(*Fa).Numfac1 = m;
	//(*Fa).cg[1] = 1.111111111;

	double cg[MAX_N_PAR + 1];

	auto CUDA_End = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &theEnd, err);
	queue.enqueueWriteBuffer(CUDA_End, CL_TRUE, 0, sizeof(int), &theEnd);
	auto cgFirst = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
	queue.enqueueWriteBuffer(cgFirst, CL_TRUE, 0, sizeof(double) * (MAX_N_PAR + 1), cg_first);
	
	auto clIa = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * (MAX_N_PAR + 1), ia);
	auto clNdata = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof ndata, &ndata, err);
	auto clConwR = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof conw_r, conw_r, err);
	
	memcpy((*Fa).Sig, sig, sizeof(double) * (MAX_N_OBS + 1));
	memcpy((*Fa).Fc, f_c, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	memcpy((*Fa).Fs, f_s, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1));
	memcpy((*Fa).Pleg, pleg, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1));
	memcpy((*Fa).Darea, d_area, sizeof(double) * (MAX_N_FAC + 1));
	memcpy((*Fa).Dsph, d_sphere, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1));
	memcpy((*Fa).Nor, normal, sizeof(double) * (MAX_N_FAC + 1) * 3);

	auto brightnesSize = sizeof(double) * (ndata + 1);
	auto clBrightness = cl::Buffer(context, CL_MEM_READ_ONLY, brightnesSize, brightness, err);
	queue.enqueueWriteBuffer(clBrightness, CL_BLOCKING, 0, brightnesSize, brightness);

	//auto sigSize = sizeof(double) * (ndata + 1);
	//auto clSig = cl::Buffer(context, CL_MEM_READ_ONLY, sigSize, sig, err);
	//queue.enqueueWriteBuffer(clBrightness, CL_BLOCKING, 0, sigSize, sig);

	/* number of fitted parameters */
	int lmfit = 0, llastma = 0, llastone = 1, ma = n_coef + 5 + n_ph_par;
	int lmfit1 = 0;
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

	auto clMa = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof ma, &ma, err);
	(*Fa).ma = ma;
	auto clMFit = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof lmfit, & lmfit, err);
	m = lmfit + 1;
	lmfit1 = m;
	auto clMFit1 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);
	auto clLastMa = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof llastma, &llastma, err);
	auto clMLastOne = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof llastone, &llastone, err);
	m = ma - 2 - n_ph_par;
	(*Fa).Ncoef0 = m;
	//auto clNCoef0 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);

	auto CUDA_Grid_dim_precalc = CUDA_grid_dim;
	if (max_test_periods < CUDA_Grid_dim_precalc) 
	{
		CUDA_Grid_dim_precalc = max_test_periods;
	}
	//auto totalWorkItems = CUDA_Grid_dim_precalc;
	auto totalWorkItems = CUDA_Grid_dim_precalc * BLOCK_DIM;
	

	auto pcc2Size = CUDA_Grid_dim_precalc * sizeof(freq_context2);
	//pcc2 = static_cast<freq_context2*>(malloc(pcc2Size));
	pcc2 = static_cast<freq_context2*>(malloc(CUDA_Grid_dim_precalc * sizeof(freq_context2)));
	
	auto CUDA_CC2 = cl::Buffer(context, CL_MEM_READ_WRITE, pcc2Size, pcc2, err);
	//queue.enqueueWriteBuffer(CUDA_CC2, CL_TRUE, 0, pcc2Size, pcc2);

	auto frSize = CUDA_Grid_dim_precalc * sizeof(freq_result);
	pfr = static_cast<freq_result*>(malloc(frSize));
	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, frSize, &pfr, err);
	queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, pfr);

	m = (Numfac + 1) * (n_coef + 1);
	(*Fa).Dg_block = m;
	//auto clDbBlock = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);
	
	double* pa, * pg, * pal, * pco, * pdytemp, * pytemp;
	
	pa = vector_double(CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	pg = vector_double(CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	pal = vector_double(CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	pco = vector_double(CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	pdytemp = vector_double(CUDA_Grid_dim_precalc * (max_l_points + 1) * (ma + 1) * sizeof(double));
	pytemp = vector_double(CUDA_Grid_dim_precalc * (max_l_points + 1) * sizeof(double));
	
	// TODO: FIX THIS
	/*texArea = nullptr;
	texDg = nullptr;*/
	
	/*
	err = cudaMalloc(&pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	err = cudaBindTexture(0, texArea, pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	err = cudaMalloc(&pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	err = cudaBindTexture(0, texDg, pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	err = cudaMalloc(&pal, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = cudaMalloc(&pco, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	err = cudaMalloc(&pdytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * (ma + 1) * sizeof(double));
	err = cudaMalloc(&pytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * sizeof(double));
	*/

	//auto paSize = CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double);
	//auto clTexArea = cl::Buffer(context, CL_MEM_READ_WRITE, paSize, &pa, err);

	auto pFaSize = sizeof(FuncArrays);
	//pFa = static_cast<varholder*>(malloc(pFaSize));
	auto clFa = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, pFaSize, Fa, err);
	queue.enqueueWriteBuffer(clFa, CL_BLOCKING, 0, pFaSize, Fa);
	/*FuncArrays faRes;
	auto clFaRes = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(FuncArrays), &faRes);*/
	

	/*auto clTexArea = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int2), &texArea, err);
	auto clTexDg = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int2), &texDg, err);
	auto clTexWeight = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int2), &texWeight, err);
	auto clTexSig = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int2), &texsig, err);
	auto clTexBrightness = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int2), &texbrightness, err);*/

	/*for (m = 0; m < CUDA_Grid_dim_precalc; m++)
	{
		freq_context2 ps{};
		ps.Area = &pa[m * (Numfac + 1)];
	}*/

	//for (m = 0; m < CUDA_Grid_dim_precalc; m++)
	//{
	//	freq_context2 ps{};
	//	/*ps.Area = &pa[m * (Numfac + 1)];
	//	ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];
	//	ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];
	//	ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];
	//	ps.dytemp = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
	//	ps.ytemp = &pytemp[m * (max_l_points + 1)];
	//	
	//	auto pt = &static_cast<freq_context2*>(pcc2)[m];*/
	//	std::fill_n(ps.cg, (MAX_N_PAR + 1), 0.0);
	//	auto pt = &static_cast<freq_context2*>(pcc2)[m];
	//	memcpy(pt, &ps, sizeof(void*));
	//	//err = cudaMemcpy(pt, &ps, sizeof(void*) * 6, cudaMemcpyHostToDevice);
	//}
	
	// NOTE: NOTA BENE - In contrast to Cuda, where global memory is zeroed by itself, here we need to initialize the values in each dimension. GV-26.09.2020
	for (m = 0; m < CUDA_Grid_dim_precalc; m++)
	{
		auto pt = &static_cast<freq_context2*>(pcc2)[m];
		pt->isInvalid = 0;
		pt->isAlamda = 0;
		pt->isNiter = 0;
				
		std::fill_n(pt->Area, (MAX_N_FAC + 1), 0.0);
		std::fill_n(pt->cg, (MAX_N_PAR + 1), 0.0);
		std::fill_n(pt->dytemp, (POINTS_MAX + 1) * (MAX_N_PAR + 1), 0.0);
		std::fill_n(pt->Dg, (MAX_N_FAC + 1) * (MAX_N_PAR + 1), 0.0);
	}

	
	queue.enqueueWriteBuffer(CUDA_CC2, CL_TRUE, 0, pcc2Size, pcc2);
	//queue.enqueueWriteBuffer(CUDA_CC2, CL_BLOCKING, 0, pFaSize, pcc2);
	// Allocate result space
	res = (freq_result*)malloc(CUDA_Grid_dim_precalc * sizeof(freq_result));

#pragma region Load kernel files
	// Load CL file, build CL program object, create CL kernel object
	std::ifstream constantsFile("period_search/constants.h");
	std::ifstream globalsFile("period_search/Globals.hcl");
	std::ifstream intrinsicsFile("period_search/Intrinsics.cl");
	std::ifstream curvFile("period_search/curv.cl");
	std::ifstream brightFile("period_search/bright.cl");
	//std::ifstream convFile("period_search/conv.cl");
	std::ifstream blmatrixFile("period_search/blmatrix.cl");
	std::ifstream gauserrcFile("period_search/gauss_errc.cl");
	std::ifstream mrqminFile("period_search/mrqmin.cl");
	std::ifstream mrqcofFile("period_search/mrqcof.cl");
	std::ifstream curv2File("period_search/Curv2.cl");
	std::ifstream kernelFile("period_search/Start.cl");

	std::stringstream st;
	st << constantsFile.rdbuf();
	st << globalsFile.rdbuf();
	st << intrinsicsFile.rdbuf();
	st << curvFile.rdbuf();
	st << brightFile.rdbuf();
	//st << convFile.rdbuf();
	st << blmatrixFile.rdbuf();
	st << gauserrcFile.rdbuf();
	st << mrqminFile.rdbuf();
	st << mrqcofFile.rdbuf();
	st << curv2File.rdbuf();
	st << kernelFile.rdbuf();

	auto KernelStart = st.str();
	st.flush();
	
	constantsFile.close();
	globalsFile.close();
	intrinsicsFile.close();
	curvFile.close();
	brightFile.close();
	blmatrixFile.close();
	gauserrcFile.close();
	mrqminFile.close();
	mrqcofFile.close();
	curv2File.close();
	kernelFile.close();
#pragma endregion

	cl::Program::Sources sources(1, std::make_pair(KernelStart.c_str(), KernelStart.length()));
	program = cl::Program(context, sources, err);

	try
	{
		program.build(devices);
		for (cl::Device dev : devices)
		{
			std::string name = dev.getInfo<CL_DEVICE_NAME>();
			std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
			std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
		}
		/*program.getInfo()*/
		
		/*size_t log;
		program.getBuildInfo()
		clGetProgramBuildInfo(program, NULL, CL_PROGRAM_BUILD_LOG, 0, NULL, &log);
		char* buildlog = malloc(log * sizeof(char));
		clGetProgramBuildInfo(program, NULL, CL_PROGRAM_BUILD_LOG, log, buildlog, NULL);
		printf(buildlog);*/
	}
	catch (cl::Error& e)
	{
		if (e.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			for (cl::Device dev : devices)
			{
				// Check the build status
				cl_build_status status1 = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
				//cl_build_status status2 = curv2Program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
				if (status1 != CL_BUILD_ERROR) // && status2 != CL_BUILD_ERROR)
					continue;

				// Get the build log
				std::string name = dev.getInfo<CL_DEVICE_NAME>();
				std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
				//buildlog = curv2Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				//std::cerr << buildlog << std::endl;
				}
		}
		else
		{
			for (cl::Device dev : devices)
			{
				std::string name = dev.getInfo<CL_DEVICE_NAME>();
				std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
				std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
			}
			throw e;
		}
	}

#pragma region Kernel creation
	cl::Kernel kernelCalculatePrepare = cl::Kernel(program, "CLCalculatePrepare");
	cl::Kernel kernelCalculatePreparePole = cl::Kernel(program, "CLCalculatePreparePole");
	cl::Kernel kernelCalculateIter1Begin = cl::Kernel(program, "CLCalculateIter1Begin");
	cl::Kernel kernelCalculateIter1Mrqcof1Start = cl::Kernel(program, "CLCalculateIter1Mrqcof1Start");
	cl::Kernel kernelCalculateIter1Mrqcof1Matrix = cl::Kernel(program, "CLCalculateIter1Mrqcof1Matrix");
	cl::Kernel kernelCalculateIter1Mrqcof1Curve1 = cl::Kernel(program, "CLCalculateIter1Mrqcof1Curve1");
	cl::Kernel kernelCalculateIter1Mrqcof1Curve2 = cl::Kernel(program, "CLCalculateIter1Mrqcof1Curve2");
	cl::Kernel kernelCalculateIter1Mrqcof1Curve1Last = cl::Kernel(program, "CLCalculateIter1Mrqcof1Curve1Last");
	cl::Kernel kernelCalculateIter1Mrqcof1End = cl::Kernel(program, "CLCalculateIter1Mrqcof1End");
	cl::Kernel kernelCalculateIter1Mrqmin1End = cl::Kernel(program, "CLCalculateIter1Mrqmin1End");
	cl::Kernel kernelCalculateIter1Mrqcof2Start = cl::Kernel(program, "CLCalculateIter1Mrqcof2Start");
	cl::Kernel kernelCalculateIter1Mrqcof2Matrix = cl::Kernel(program, "CLCalculateIter1Mrqcof2Matrix");
	cl::Kernel kernelCalculateIter1Mrqcof2Curve1 = cl::Kernel(program, "CLCalculateIter1Mrqcof2Curve1");
	cl::Kernel kernelCalculateIter1Mrqcof2Curve2 = cl::Kernel(program, "CLCalculateIter1Mrqcof2Curve2");
	cl::Kernel kernelCalculateIter1Mrqcof2Curve1Last = cl::Kernel(program, "CLCalculateIter1Mrqcof2Curve1Last");
	cl::Kernel kernelCalculateIter1Mrqcof2End = cl::Kernel(program, "CLCalculateIter1Mrqcof2End");
	cl::Kernel kernelCalculateIter1Mrqmin2End = cl::Kernel(program, "CLCalculateIter1Mrqmin2End");
	cl::Kernel kernelCalculateIter2 = cl::Kernel(program, "CLCalculateIter2");
	cl::Kernel kernelCalculateFinishPole = cl::Kernel(program, "CLCalculateFinishPole");
	cl::Kernel kernelCalculateFinish = cl::Kernel(program, "CLCalculateFinish");
#pragma endregion

#pragma region SetKernelArgs
	kernelCalculateFinish.setArg(0, CUDA_CC2);
	kernelCalculateFinish.setArg(1, CUDA_FR);
	
	kernelCalculatePrepare.setArg(0, CUDA_CC2);
	kernelCalculatePrepare.setArg(1, CUDA_FR);
	kernelCalculatePrepare.setArg(2, sizeof max_test_periods, &max_test_periods);
	kernelCalculatePrepare.setArg(4, sizeof freq_start, &freq_start);
	kernelCalculatePrepare.setArg(5, sizeof freq_step, &freq_step);
	
	kernelCalculatePreparePole.setArg(0, CUDA_CC2);
	kernelCalculatePreparePole.setArg(1, CUDA_FR);
	kernelCalculatePreparePole.setArg(2, clFa);
	kernelCalculatePreparePole.setArg(3, CUDA_lambda_pole);
	kernelCalculatePreparePole.setArg(4, CUDA_beta_pole);
	kernelCalculatePreparePole.setArg(5, CUDA_End);
	// NOTE: 7th arg 'm' is set a little further as 'm' is an iterator counter
	kernelCalculatePreparePole.setArg(7, cgFirst);
	
	kernelCalculateIter1Begin.setArg(0, CUDA_CC2);
	kernelCalculateIter1Begin.setArg(1, CUDA_FR);
	kernelCalculateIter1Begin.setArg(2, CUDA_End);
	kernelCalculateIter1Begin.setArg(3, sizeof(int), &n_iter_min);
	kernelCalculateIter1Begin.setArg(4, sizeof(int), &n_iter_max);
	kernelCalculateIter1Begin.setArg(5, sizeof(double), &iter_diff_max);
	kernelCalculateIter1Begin.setArg(6, sizeof(double), &aLambda_start);

	kernelCalculateIter1Mrqcof1Start.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof1Start.setArg(1, clFa);

	kernelCalculateIter1Mrqcof1Matrix.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof1Matrix.setArg(1, clFa);

	kernelCalculateIter1Mrqcof2Matrix.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof2Matrix.setArg(1, clFa);
		
	kernelCalculateIter1Mrqcof1Curve1.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof1Curve1.setArg(1, clFa);
	/*kernelCalculateIter1Mrqcof1Curve1.setArg(2, clTexArea);
	kernelCalculateIter1Mrqcof1Curve1.setArg(3, clTexDg);*/
	
	kernelCalculateIter1Mrqcof1Curve2.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof1Curve2.setArg(1, clFa);
	/*kernelCalculateIter1Mrqcof1Curve2.setArg(2, clTexSig);
	kernelCalculateIter1Mrqcof1Curve2.setArg(3, clTexWeight);
	kernelCalculateIter1Mrqcof1Curve2.setArg(4, clTexBrightness);*/
	
	kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof1Curve1Last.setArg(1, clFa);
	/*kernelCalculateIter1Mrqcof1Curve1Last.setArg(2, clTexArea);
	kernelCalculateIter1Mrqcof1Curve1Last.setArg(3, clTexDg);*/
	kernelCalculateIter1Mrqcof1Curve1Last.setArg(2, sizeof in_rel[l_curves], &(in_rel[l_curves]));
	kernelCalculateIter1Mrqcof1Curve1Last.setArg(3, sizeof l_points[l_curves], &(l_points[l_curves]));

	kernelCalculateIter1Mrqmin1End.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqmin1End.setArg(1, clFa);
	kernelCalculateIter1Mrqmin1End.setArg(2, CUDA_End);

	kernelCalculateIter1Mrqcof1End.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof1End.setArg(1, clFa);

	kernelCalculateIter1Mrqcof2Start.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof2Start.setArg(1, clFa);
	//kernelCalculateIter1Mrqcof2Start.setArg(2, sizeof l_points[iC], &(l_points[iC]));

	kernelCalculateIter1Mrqcof2Curve1.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof2Curve1.setArg(1, clFa);
	/*kernelCalculateIter1Mrqcof2Curve1.setArg(2, clTexArea);
	kernelCalculateIter1Mrqcof2Curve1.setArg(3, clTexDg); */
	
	kernelCalculateIter1Mrqcof2Curve2.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof2Curve2.setArg(1, clFa);
	/*kernelCalculateIter1Mrqcof2Curve2.setArg(2, clTexSig);
	kernelCalculateIter1Mrqcof2Curve2.setArg(3, clTexWeight);
	kernelCalculateIter1Mrqcof2Curve2.setArg(4, clTexBrightness);*/
	
	kernelCalculateIter1Mrqcof2Curve1Last.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof2Curve1Last.setArg(1, clFa);
	/*kernelCalculateIter1Mrqcof2Curve1Last.setArg(2, clTexArea);
	kernelCalculateIter1Mrqcof2Curve1Last.setArg(3, clTexDg);*/

	kernelCalculateIter1Mrqcof2End.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqcof2End.setArg(1, clFa);

	kernelCalculateIter1Mrqmin2End.setArg(0, CUDA_CC2);
	kernelCalculateIter1Mrqmin2End.setArg(1, clFa);

	kernelCalculateIter2.setArg(0, CUDA_CC2);
	kernelCalculateIter2.setArg(1, clFa);

	kernelCalculateFinishPole.setArg(0, CUDA_CC2);
	kernelCalculateFinishPole.setArg(1, CUDA_FR);
	kernelCalculateFinishPole.setArg(2, clFa);

	kernelCalculateFinish.setArg(0, CUDA_CC2);
	kernelCalculateFinish.setArg(1, CUDA_FR);
#pragma endregion

	int ce;

	cl::Device device = devices[0];
	const auto clDeviceGlobalMemSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
	const auto clDeviceLocalMemSize = device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
	const auto clDeviceMaxConstantArgs = device.getInfo<CL_DEVICE_MAX_CONSTANT_ARGS>();
	const auto clDeviceMaxConstantBufferSize = device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>();
	const auto clDeviceMaxParameterSize = device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();
	const auto clDeviceMaxMemAllocSize = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(); std::cerr << "CL_DEVICE_GLOBAL_MEM_SIZE: " << clDeviceGlobalMemSize << " B" << endl;
	std::cerr << "*********************" << std::endl;
	std::cerr << "CL_DEVICE_LOCAL_MEM_SIZE: " << clDeviceLocalMemSize << " B" << endl;
	std::cerr << "CL_DEVICE_MAX_CONSTANT_ARGS: " << clDeviceMaxConstantArgs << endl;
	std::cerr << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: " << clDeviceMaxConstantBufferSize << " B" << endl;
	std::cerr << "CL_DEVICE_MAX_PARAMETER_SIZE: " << clDeviceMaxParameterSize << " B" << endl;
	std::cerr << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << clDeviceMaxMemAllocSize << " B" << endl;

	size_t result[3];
	kernelCalculatePrepare.getWorkGroupInfo(device, CL_KERNEL_LOCAL_MEM_SIZE, result);
	kernelCalculatePrepare.getWorkGroupInfo(device, CL_KERNEL_PRIVATE_MEM_SIZE, result);

	std::cerr << "*********************" << std::endl;

	for (n = 1; n <= max_test_periods; n += CUDA_Grid_dim_precalc)
	{
		kernelCalculatePrepare.setArg(3, sizeof n, &n);
		// NOTE: CudaCalculatePrepare(n, max_test_periods, freq_start, freq_step); // << <CUDA_Grid_dim_precalc, 1 >> >
		queue.enqueueNDRangeKernel(kernelCalculatePrepare, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
		queue.finish();
		/*queue.enqueueReadBuffer(CUDA_CC2, CL_BLOCKING, 0, pcc2Size, pcc2);
		queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, pfr);*/
		//PrintFreqResult(CUDA_Grid_dim_precalc, pcc2, pfr);

		queue.enqueueBarrierWithWaitList();
		// cuda sync err = cudaThreadSynchronize();
		
		for (m = 1; m <= N_POLES; m++)
		{
			//zero global End signal
			theEnd = 0;
			kernelCalculatePreparePole.setArg(6, sizeof m, &m);
			queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
			//cudaMemcpyToSymbol(CUDA_End, &theEnd, sizeof(theEnd));

			// NOTE: CudaCalculatePreparePole(m);													<< <CUDA_Grid_dim_precalc, 1 >> >
			queue.enqueueNDRangeKernel(kernelCalculatePreparePole, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
			queue.finish();
			
			//queue.enqueueReadBuffer(CUDA_CC2, CL_BLOCKING, 0, pcc2Size, pcc2);
			/*std::memcpy(&cg[0], &(*(freq_context2*)pcc2).cg[0], (MAX_N_PAR + 1) * sizeof(double));
			auto arrcg = (*(freq_context2*)pcc2).cg;*/
			
			queue.enqueueReadBuffer(clFa, CL_BLOCKING, 0, pFaSize, Fa);
			queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
			//size_t out = clFinish(queue());
			//queue.enqueueBarrierWithWaitList();

#ifdef _DEBUG
			printf(".");
#endif
			while (!theEnd)
			{
																// NOTE: CudaCalculateIter1Begin(); // << <CUDA_Grid_dim_precalc, 1 >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Begin, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				clFinish(queue());
				//queue.enqueueBarrierWithWaitList();
																// NOTE: CudaCalculateIter1Mrqcof1Start(); // << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				ce = queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Start, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				clFinish(queue());
				//queue.enqueueReadBuffer(CUDA_CC2, CL_BLOCKING, 0, pcc2Size, pcc2);
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				//queue.enqueueBarrierWithWaitList();

				for (iC = 1; iC < l_curves; iC++)
				{
					kernelCalculateIter1Mrqcof1Matrix.setArg(2, sizeof (l_points[iC]), &(l_points[iC]));
																// NOTE: CudaCalculateIter1Mrqcof1Matrix(l_points[iC]);					//<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>
					ce = queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Matrix, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
					clFinish(queue());
					queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
					//queue.enqueueReadBuffer(CUDA_CC2, CL_BLOCKING, 0, sizeof pcc2, &pcc2);
					queue.enqueueBarrierWithWaitList();

					kernelCalculateIter1Mrqcof1Curve1.setArg(2, sizeof (in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof1Curve1.setArg(3, sizeof (l_points[iC]), &(l_points[iC]));
																// NOTE: CudaCalculateIter1Mrqcof1Curve1(in_rel[iC], l_points[iC]);		// << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					ce = queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
					clFinish(queue());
					queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
					queue.enqueueBarrierWithWaitList();

					kernelCalculateIter1Mrqcof1Curve2.setArg(2, sizeof (in_rel[iC]), &(in_rel[iC]));
					kernelCalculateIter1Mrqcof1Curve2.setArg(3, sizeof (l_points[iC]), &(l_points[iC]));
																// NOTE: CudaCalculateIter1Mrqcof1Curve2(in_rel[iC], l_points[iC]);		// << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					ce = queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
					clFinish(queue());
					queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
					queue.enqueueBarrierWithWaitList();
				}
				
																// NOTE: CudaCalculateIter1Mrqcof1Curve1Last(in_rel[l_curves], l_points[l_curves]);	//  << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1Last, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
																// NOTE: CudaCalculateIter1Mrqcof1Curve2(in_rel[l_curves], l_points[l_curves]);		//  << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
																// NOTE: CudaCalculateIter1Mrqcof1End();	<< <CUDA_Grid_dim_precalc, 1 >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1End, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);

																// NOTE: CudaCalculateIter1Mrqmin1End();   << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqmin1End, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				clFinish(queue());
				kernelCalculateIter1Mrqmin1End.getWorkGroupInfo(device, CL_KERNEL_LOCAL_MEM_SIZE, result);
				kernelCalculateIter1Mrqmin1End.getWorkGroupInfo(device, CL_KERNEL_PRIVATE_MEM_SIZE, result);
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
																// NOTE: CudaCalculateIter1Mrqcof2Start();  	<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Start, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				for (iC = 1; iC < l_curves; iC++)
				{
					kernelCalculateIter1Mrqcof2Matrix.setArg(2, l_points[iC]);				// NOTE: CudaCalculateIter1Mrqcof2Matrix(l_points[iC]);	<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Matrix, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
					clFinish(queue());
					queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
					
					kernelCalculateIter1Mrqcof2Curve1.setArg(2, in_rel[iC]);
					kernelCalculateIter1Mrqcof2Curve1.setArg(3, l_points[l_curves]);		// NOTE: CudaCalculateIter1Mrqcof2Curve1(in_rel[iC], l_points[iC]);	<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve1, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
					clFinish(queue());
					queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
					
					kernelCalculateIter1Mrqcof2Curve2.setArg(2, in_rel[iC]);
					kernelCalculateIter1Mrqcof2Curve2.setArg(3, l_points[iC]);				// NOTE: CudaCalculateIter1Mrqcof2Curve2(in_rel[iC], l_points[iC]); << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
					clFinish(queue());
					queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				}
				
				kernelCalculateIter1Mrqcof2Curve1Last.setArg(2, in_rel[l_curves]);	//  l_points.at(l_curves)	
				kernelCalculateIter1Mrqcof2Curve1Last.setArg(3, l_points[l_curves]);		// NOTE: CudaCalculateIter1Mrqcof2Curve1Last(in_rel[l_curves], l_points[l_curves]);	//	 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve1Last, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				
				kernelCalculateIter1Mrqcof2Curve2.setArg(2, in_rel[l_curves]);
				kernelCalculateIter1Mrqcof2Curve2.setArg(3, l_points[l_curves]); 			// NOTE: CudaCalculateIter1Mrqcof2Curve2(in_rel[l_curves], l_points[l_curves]);		//	 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Curve2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
																							// NOTE: CudaCalculateIter1Mrqcof2End();	<<<CUDA_Grid_dim_precalc, 1 >>>
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2End, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
																							// NOTE: CudaCalculateIter1Mrqmin2End(); <<<CUDA_Grid_dim_precalc, 1 >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqmin2End, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
																							// NOTE:CudaCalculateIter2();  <<<CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter2, cl::NDRange(), cl::NDRange(totalWorkItems), cl::NDRange(BLOCK_DIM));
				clFinish(queue());
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				
				// TODO: Sync kernels here - waith for event
				//err=cudaThreadSynchronize(); memcpy is synchro itself
				queue.enqueueBarrierWithWaitList();

				//cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd));
				queue.enqueueReadBuffer(CUDA_End, CL_BLOCKING, 0, sizeof(int), &theEnd);
				theEnd = theEnd == CUDA_Grid_dim_precalc;
			}
																							// NOTE: CudaCalculateFinishPole();	<<<CUDA_Grid_dim_precalc, 1 >> >
			queue.enqueueNDRangeKernel(kernelCalculateFinishPole, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
			clFinish(queue());
			// TODO: Sync threads on device -> kernel(s)?
			//err = cudaThreadSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_Grid_dim_precalc);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context2)*CUDA_Grid_dim_precalc);
						//break; //debug

		}
		printf("\n");

																							// NOTE: CudaCalculateFinish();	<<<CUDA_Grid_dim_precalc, 1 >> >
		queue.enqueueNDRangeKernel(kernelCalculateFinish, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange());
		clFinish(queue());
		queue.enqueueReadBuffer(CUDA_CC2, CL_BLOCKING, 0, pcc2Size, pcc2);
		queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, pfr);
		//err=cudaThreadSynchronize(); memcpy is synchro itself

		//read results here
		// TODO: Read result Buffer
		//err = cudaMemcpy(res, pfr, sizeof(freq_result) * CUDA_Grid_dim_precalc, cudaMemcpyDeviceToHost);

		for (m = 1; m <= CUDA_Grid_dim_precalc; m++)
		{
			if (res[m - 1].isReported == 1)
				sum_dark_facet = sum_dark_facet + res[m - 1].dark_best;
		}

	} /* period loop */


	// TODO: Free Buffers
	/*cudaUnbindTexture(texArea);
	cudaUnbindTexture(texDg);
	cudaUnbindTexture(texbrightness);
	cudaUnbindTexture(texsig);
	cudaFree(pa);
	cudaFree(pg);
	cudaFree(pal);
	cudaFree(pco);
	cudaFree(pdytemp);
	cudaFree(pytemp);
	cudaFree(pcc);
	cudaFree(pfr);
	cudaFree(pbrightness);
	cudaFree(psig);*/

	free((void*)res);

	ave_dark_facet = sum_dark_facet / max_test_periods;

	if (ave_dark_facet < 1.0)
		*new_conw = 1; /* new correct conwexity weight */
	if (ave_dark_facet >= 1.0)
		*conw_r = *conw_r * 2; /* still not good */

	return 1;
}

