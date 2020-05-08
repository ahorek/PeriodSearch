#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
//#include <CL/cl.h>
#include <CL/cl.hpp>

//#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
//#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
//#else
//#error "Double precision floating point not supported by OpenCL implementation."
//#endif

//#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
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
double CUDA_cl;
double log_cl;
double aLambda_start;


cl_int2* texWeight;
cl_int2* texArea;
cl_int2* texDg;
cl_int2* texbrightness;
cl_int2* texsig;

//cl::Image1D texWeight;  //NOTE: CUDA's 'texture<int2, 1>{}' structure equivalent
//cl::Image1D texArea;
//cl::Image1D texDg;
cl::Buffer CUDA_cg_first;
cl::Buffer CUDA_beta_pole;
cl::Buffer CUDA_lambda_pole;
cl::Buffer CUDA_par;

// NOTE: global to one thread
#ifdef __GNUC__
freq_result* CUDA_FR __attribute__((aligned(16)));
#else
__declspec(align(16)) freq_result* CUDA_FR;
#endif

double* pee, * pee0, * pWeight;

cl_int ClPrepare(int deviceId, double* beta_pole, double* lambda_pole, double* par, double cl, double Alambda_start, double Alambda_incr,
	double ee[][3], double ee0[][3], double* tim, double Phi_0, int checkex, int ndata)
{
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
		auto doesNotSupportsFp64 = (deviceExtensions.find("cl_khr_fp64") == std::string::npos) || (deviceExtensions.find("cl_khr_fp64") == std::string::npos);
		if(doesNotSupportsFp64)
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
#endif

		// TODO: Calculate this:
		auto SMXBlock = 128;
		CUDA_grid_dim = msCount * SMXBlock;

		cl_int* err = nullptr;
		CUDA_beta_pole = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * (N_POLES + 1), beta_pole, err);
		CUDA_lambda_pole = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(double) * (N_POLES + 1), lambda_pole, err);
		CUDA_par = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * 4, par, err);
		CUDA_cl = cl;
		log_cl = log(cl);
		//auto clCl = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof cl, &cl, err);
		aLambda_start = Alambda_start;
		//auto clAlambdaStart = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Alambda_start, &Alambda_start, err);
		auto clAlambdaIncr = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Alambda_incr, &Alambda_incr, err);
		auto clMmax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m_max, &m_max, err);
		auto clLmax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof l_max, &l_max, err);
		auto clTim = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_OBS + 1), tim, err);
		auto clPhi0 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Phi_0, &Phi_0);

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
		auto clEe = cl::Buffer(context, CL_MEM_READ_ONLY, pEeSize, ee, err);
		auto clEe0 = cl::Buffer(context, CL_MEM_READ_ONLY, pEeSize, ee, err);

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
	int max_test_periods, iC, theEnd;
	double sum_dark_facet, ave_dark_facet;
	int i, n, m;
	auto n_max = static_cast<int>((freq_start - freq_end) / freq_step) + 1;
	int n_iter_max;
	double iter_diff_max;
	freq_result* res;
	//freq_context* pcc;
	void* pcc, *pcc2;
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

	//auto CUDA_Ncoef = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof n_coef, &n_coef, err);
	//CUDA_Nphpar = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof n_ph_par, &n_ph_par, err);
	auto clNumfac = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof Numfac, &Numfac, err);
	m = Numfac + 1;
	auto clNumfac1 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);
	auto clIa = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int) * (MAX_N_PAR + 1), ia);
	CUDA_cg_first = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_PAR + 1), cg_first, err);
	//auto clNIterMax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof n_iter_max, &n_iter_max, err);
	//auto clNIterMin = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof n_iter_min, &n_iter_min, err);
	auto clNdata = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof ndata, &ndata, err);
	//auto clIterDiffMax = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof iter_diff_max, &iter_diff_max, err);
	auto clConwR = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof conw_r, conw_r, err);
	auto clNor = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * 3, normal, err);
	auto clFc = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1), f_c, err);
	auto clFs = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1), f_s, err);
	auto clPleg = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * (MAX_LM + 1) * (MAX_LM + 1), pleg, err);
	auto clDArea = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1), d_area, err);
	auto clDSphere = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double) * (MAX_N_FAC + 1) * (MAX_N_PAR + 1), d_sphere, err);

	auto brightnesSize = sizeof(double) * (ndata + 1);
	auto clBrightness = cl::Buffer(context, CL_MEM_READ_ONLY, brightnesSize, brightness, err);
	queue.enqueueWriteBuffer(clBrightness, CL_BLOCKING, 0, brightnesSize, brightness);

	//  err = cudaMalloc(&pbrightness, (ndata + 1) * sizeof(double));
	//  err = cudaMemcpy(pbrightness, brightness, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice);
	//  err = cudaBindTexture(0, texbrightness, pbrightness, (ndata + 1) * sizeof(double));

	auto sigSize = sizeof(double) * (ndata + 1);
	auto clSig = cl::Buffer(context, CL_MEM_READ_ONLY, sigSize, sig, err);
	queue.enqueueWriteBuffer(clBrightness, CL_BLOCKING, 0, sigSize, sig);

	//  err = cudaMalloc(&psig, (ndata + 1) * sizeof(double));
	//  err = cudaMemcpy(psig, sig, (ndata + 1) * sizeof(double), cudaMemcpyHostToDevice);
	//  err = cudaBindTexture(0, texsig, psig, (ndata + 1) * sizeof(double));
	//

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
	auto clMFit = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof lmfit, &lmfit, err);
	m = lmfit + 1;
	lmfit1 = m;
	auto clMFit1 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);
	auto clLastMa = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof llastma, &llastma, err);
	auto clMLastOne = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof llastone, &llastone, err);
	m = ma - 2 - n_ph_par;
	auto clNCoef0 = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);

	//		cudaMemcpyToSymbol(CUDA_ma, &ma, sizeof(ma));
	//		cudaMemcpyToSymbol(CUDA_mfit, &lmfit, sizeof(lmfit));
	//		m = lmfit + 1;
	//		cudaMemcpyToSymbol(CUDA_mfit1, &m, sizeof(m));
	//		cudaMemcpyToSymbol(CUDA_lastma, &llastma, sizeof(llastma));
	//		cudaMemcpyToSymbol(CUDA_lastone, &llastone, sizeof(llastone));
	//		m = ma - 2 - n_ph_par;
	//		cudaMemcpyToSymbol(CUDA_ncoef0, &m, sizeof(m));

	auto CUDA_Grid_dim_precalc = CUDA_grid_dim;
	if (max_test_periods < CUDA_Grid_dim_precalc) CUDA_Grid_dim_precalc = max_test_periods;

	auto gdpcSize = CUDA_Grid_dim_precalc * sizeof(freq_context);
	pcc = static_cast<freq_context*>(malloc(gdpcSize));
	auto CUDA_CC = cl::Buffer(context, CL_MEM_READ_ONLY, gdpcSize, &pcc, err);
	queue.enqueueWriteBuffer(CUDA_CC, CL_TRUE, 0, gdpcSize, pcc);

	// err = cudaMalloc(&pcc, CUDA_Grid_dim_precalc * sizeof(freq_context));
	//	cudaMemcpyToSymbol(CUDA_CC, &pcc, sizeof(pcc));


	auto pcc2Size = CUDA_Grid_dim_precalc * sizeof(freq_context2);
	pcc2 = static_cast<freq_context2*>(malloc(pcc2Size));
	auto CUDA_CC2 = cl::Buffer(context, CL_MEM_READ_ONLY, pcc2Size, &pcc2, err);
	queue.enqueueWriteBuffer(CUDA_CC2, CL_TRUE, 0, pcc2Size, pcc2);

	auto frSize = CUDA_Grid_dim_precalc * sizeof(freq_result);
	pfr = static_cast<freq_result*>(malloc(frSize));
	auto CUDA_FR = cl::Buffer(context, CL_MEM_READ_ONLY, frSize, &pfr, err);
	queue.enqueueWriteBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, pfr);

	//err = cudaMalloc(&pfr, CUDA_Grid_dim_precalc * sizeof(freq_result));
	//cudaMemcpyToSymbol(CUDA_FR, &pfr, sizeof(pfr));

	m = (Numfac + 1) * (n_coef + 1);
	auto clDbBlock = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof m, &m, err);

	/*m = (Numfac + 1) * (n_coef + 1);
	cudaMemcpyToSymbol(CUDA_Dg_block, &m, sizeof(m));*/

	double* pa, * pg, * pal, * pco, * pdytemp, * pytemp;

	auto paSize = CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double);
	auto clPa = cl::Buffer(context, CL_MEM_READ_WRITE, paSize, &pa, err);


	//err = cudaMalloc(&pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	//err = cudaBindTexture(0, texArea, pa, CUDA_Grid_dim_precalc * (Numfac + 1) * sizeof(double));
	//err = cudaMalloc(&pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	//err = cudaBindTexture(0, texDg, pg, CUDA_Grid_dim_precalc * (Numfac + 1) * (n_coef + 1) * sizeof(double));
	//err = cudaMalloc(&pal, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	//err = cudaMalloc(&pco, CUDA_Grid_dim_precalc * (lmfit + 1) * (lmfit + 1) * sizeof(double));
	//err = cudaMalloc(&pdytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * (ma + 1) * sizeof(double));
	//err = cudaMalloc(&pytemp, CUDA_Grid_dim_precalc * (max_l_points + 1) * sizeof(double));

	for (m = 0; m < CUDA_Grid_dim_precalc; m++)
	{
		freq_context ps;
		ps.Area = &pa[m * (Numfac + 1)];
		ps.Dg = &pg[m * (Numfac + 1) * (n_coef + 1)];
		ps.alpha = &pal[m * (lmfit + 1) * (lmfit + 1)];
		ps.covar = &pco[m * (lmfit + 1) * (lmfit + 1)];
		ps.dytemp = &pdytemp[m * (max_l_points + 1) * (ma + 1)];
		ps.ytemp = &pytemp[m * (max_l_points + 1)];
		auto pt = &static_cast<freq_context*>(pcc)[m];

		//err = cudaMemcpy(pt, &ps, sizeof(void*) * 6, cudaMemcpyHostToDevice);
	}

	res = (freq_result*)malloc(CUDA_Grid_dim_precalc * sizeof(freq_result));

	//CHAR filepath[MAX_PATH]; // = getenv("_");
	//GetModuleFileName(nullptr, filepath, MAX_PATH);
	//auto filename = PathFindFileName(filepath);

	// Load CL file, build CL program object, create CL kernel object
	std::ifstream constantsFile("period_search/constants.h");
	std::ifstream globalsFile("period_search/Globals.hcl");
	std::ifstream intrinsicsFile("period_search/Intrnsics.cl");
	std::ifstream curvFile("period_search/curv.cl");
	std::ifstream convFile("period_search/conv.cl");
	std::ifstream blmatrixFile("period_search/blmatrix.cl");
	std::ifstream gauserrcFile("period_search/gauss_errc.cl");
	std::ifstream mrqminFile("period_search/mrqmin.cl");
	std::ifstream brightFile("period_search/bright.cl");
	std::ifstream mrqcofFile("period_search/mrqcof.cl");
	//std::ifstream curv2File("period_search/curv2_2.cl");
	std::ifstream kernelFile("period_search/Start.cl");

	std::stringstream st;
	st << constantsFile.rdbuf();
	st << globalsFile.rdbuf();
	st << intrinsicsFile.rdbuf();
	st << curvFile.rdbuf();
	st << convFile.rdbuf();
	st << blmatrixFile.rdbuf();
	st << brightFile.rdbuf();
	st << gauserrcFile.rdbuf();
	st << mrqminFile.rdbuf();
	st << mrqcofFile.rdbuf();
	//st << curv2File.rdbuf();
	st << kernelFile.rdbuf();

	auto KernelStart = st.str();
	kernelFile.close();
	constantsFile.close();
	globalsFile.close();
	st.flush();

	cl::Program::Sources sources(1, std::make_pair(KernelStart.c_str(), KernelStart.length()));
	program = cl::Program(context, sources, err);

	/*std::ifstream curv2File("period_search/Curv2.cl");
	std::stringstream curv2String;
	curv2String << curv2File.rdbuf();
	auto KernelCurv2 = curv2String.str();
	curv2File.close();
	curv2String.flush();

	cl::Program::Sources curv2Source(1, std::make_pair(KernelCurv2.c_str(), KernelCurv2.length()));
	auto curv2Program = cl::Program(context, curv2Source, err);*/

	try
	{
		program.build(devices);
		//curv2Program.build(devices);
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
			throw e;
		}
	}

	//PrintFreqResult(CUDA_Grid_dim_precalc, pcc2, pfr);
	auto freqResultSize = CUDA_Grid_dim_precalc * sizeof(freq_result);
	//auto bufCudaFr = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, freqResultSize, &CUDA_FR, err);
	cl::Kernel kernelCalculatePrepare = cl::Kernel(program, "CLCalculatePrepare");
	cl::Kernel kernelCalculateFinish = cl::Kernel(program, "CLCalculateFinish");
	cl::Kernel kernelCalculatePreparePole = cl::Kernel(program, "CLCalculatePreparePole");
	cl::Kernel kernelCalculateIter1Begin = cl::Kernel(program, "CLCalculateIter1Begin");
	cl::Kernel kernelCalculateIter1Mrqcof1Start = cl::Kernel(program, "CLCalculateIter1Mrqcof1Start");
	cl::Kernel kernelCalculateIter1Mrqcof1Matrix = cl::Kernel(program, "CLCalculateIter1Mrqcof1Matrix");
	cl::Kernel kernelCalculateIter1Mrqcof1Curve1 = cl::Kernel(program, "CLCalculateIter1Mrqcof1Curve1");
	cl::Kernel kernelCalculateIter1Mrqcof1Curve2 = cl::Kernel(program, "CLCalculateIter1Mrqcof1Curve2");
	cl::Kernel kernelCalculateIter1Mrqcof1Curve1Last = cl::Kernel(program, "CLCalculateIter1Mrqcof1Curve1Last");
	cl::Kernel kernelCalculateIter1Mrqcof1End = cl::Kernel(program, "CLCalculateIter1Mrqcof1End");
	cl::Kernel kernelCalculateIter1Mrqmin1End = cl::Kernel(program, "CLCalculateIter1Mrqmin1End");


	kernelCalculateFinish.setArg(0, CUDA_CC2);
	kernelCalculateFinish.setArg(1, CUDA_FR);
	kernelCalculatePrepare.setArg(0, CUDA_CC2);
	kernelCalculatePrepare.setArg(1, CUDA_FR);
	kernelCalculatePrepare.setArg(2, sizeof max_test_periods, &max_test_periods);
	kernelCalculatePreparePole.setArg(0, CUDA_CC2);
	kernelCalculatePreparePole.setArg(1, CUDA_FR);

	for (n = 1; n <= max_test_periods; n += CUDA_Grid_dim_precalc)
	{
		kernelCalculatePrepare.setArg(3, sizeof n, &n);
		kernelCalculatePrepare.setArg(4, sizeof freq_start, &freq_start);
		kernelCalculatePrepare.setArg(5, sizeof freq_step, &freq_step);

		queue.enqueueNDRangeKernel(kernelCalculatePrepare, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
		//auto result = queue.finish();
		queue.enqueueReadBuffer(CUDA_CC2, CL_BLOCKING, 0, pcc2Size, pcc2);
		queue.enqueueReadBuffer(CUDA_FR, CL_BLOCKING, 0, frSize, pfr);
		PrintFreqResult(CUDA_Grid_dim_precalc, pcc2, pfr);

		//CudaCalculatePrepare(n, max_test_periods, freq_start, freq_step); // << <CUDA_Grid_dim_precalc, 1 >> >

		// TODO: Sync kernels here - waith for event
		// cuda sync err = cudaThreadSynchronize();
		//
		kernelCalculateIter1Begin.setArg(0, CUDA_CC2);
		kernelCalculateIter1Begin.setArg(1, CUDA_FR);
		kernelCalculateIter1Begin.setArg(3, sizeof n_iter_min, &n_iter_min);
		kernelCalculateIter1Begin.setArg(4, sizeof n_iter_max, &n_iter_max);
		kernelCalculateIter1Begin.setArg(5, sizeof iter_diff_max, &iter_diff_max);
		kernelCalculateIter1Begin.setArg(6, sizeof aLambda_start, &aLambda_start);

		for (m = 1; m <= N_POLES; m++)
		{

			//zero global End signal
			theEnd = 0;
			auto CUDA_End = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof theEnd, &theEnd, err);
			queue.enqueueWriteBuffer(CUDA_End, CL_BLOCKING, 0, sizeof theEnd, &theEnd);
			//cudaMemcpyToSymbol(CUDA_End, &theEnd, sizeof(theEnd));

			// NOTE: CudaCalculatePreparePole(m);																 << <CUDA_Grid_dim_precalc, 1 >> >
			kernelCalculatePreparePole.setArg(2, CUDA_End);
			kernelCalculatePreparePole.setArg(3, CUDA_cg_first);
			kernelCalculatePreparePole.setArg(4, CUDA_beta_pole);
			kernelCalculatePreparePole.setArg(5, CUDA_lambda_pole);
			kernelCalculatePreparePole.setArg(6, CUDA_par);
			kernelCalculatePreparePole.setArg(7, sizeof log_cl, &log_cl);
			kernelCalculatePreparePole.setArg(8, sizeof m, &m);
			kernelCalculatePreparePole.setArg(9, sizeof n_coef, &n_coef);
			kernelCalculatePreparePole.setArg(10, sizeof n_ph_par, &n_ph_par);

			queue.enqueueNDRangeKernel(kernelCalculatePreparePole, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));

			while (!theEnd)
			{
				// NOTE: CudaCalculateIter1Begin(); // << <CUDA_Grid_dim_precalc, 1 >> >
				kernelCalculateIter1Begin.setArg(2, CUDA_End);
				queue.enqueueNDRangeKernel(kernelCalculateIter1Begin, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
				//mrqcof
				// NOTE: CudaCalculateIter1Mrqcof1Start(); // << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				kernelCalculateIter1Mrqcof1Start.setArg(0, CUDA_CC);
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Start, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(BLOCK_DIM));

				for (iC = 1; iC < l_curves; iC++)
				{
					// NOTE: CudaCalculateIter1Mrqcof1Matrix(l_points[iC]);					//<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >>
					kernelCalculateIter1Mrqcof1Matrix.setArg(0, CUDA_CC);
					kernelCalculateIter1Mrqcof1Matrix.setArg(0, CUDA_FR);
					kernelCalculateIter1Mrqcof1Matrix.setArg(1, sizeof l_points[iC], &l_points[iC]);
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Matrix, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(BLOCK_DIM));

					// NOTE: CudaCalculateIter1Mrqcof1Curve1(in_rel[iC], l_points[iC]);		// << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(BLOCK_DIM));
					kernelCalculateIter1Mrqcof1Curve1.setArg(0, CUDA_CC);
					kernelCalculateIter1Mrqcof1Curve1.setArg(1, CUDA_FR);
					kernelCalculateIter1Mrqcof1Curve1.setArg(2, texArea);
					kernelCalculateIter1Mrqcof1Curve1.setArg(3, texDg);
					kernelCalculateIter1Mrqcof1Curve1.setArg(4, sizeof in_rel[iC], &in_rel[iC]);
					kernelCalculateIter1Mrqcof1Curve1.setArg(5, sizeof l_points[iC], &l_points[iC]);

					// NOTE: CudaCalculateIter1Mrqcof1Curve2(in_rel[iC], l_points[iC]);		// << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					kernelCalculateIter1Mrqcof1Curve2.setArg(0, CUDA_CC);
					kernelCalculateIter1Mrqcof1Curve2.setArg(0, CUDA_FR);
					kernelCalculateIter1Mrqcof1Curve2.setArg(0, texsig);
					kernelCalculateIter1Mrqcof1Curve2.setArg(0, texWeight);
					kernelCalculateIter1Mrqcof1Curve2.setArg(0, texbrightness);
					kernelCalculateIter1Mrqcof1Curve2.setArg(0, sizeof in_rel[iC], &in_rel[iC]);
					kernelCalculateIter1Mrqcof1Curve2.setArg(0, sizeof l_points[iC], &l_points[iC]);
					queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(BLOCK_DIM));
				}
				
				// NOTE: CudaCalculateIter1Mrqcof1Curve1Last(in_rel[l_curves], l_points[l_curves]);	//  << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, CUDA_CC);
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, CUDA_FR);
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, texArea);
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, texDg);
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, sizeof in_rel[iC], &in_rel[iC]);
				kernelCalculateIter1Mrqcof1Curve1Last.setArg(0, sizeof l_points[iC], &l_points[iC]);
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve1Last, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(BLOCK_DIM));
				
				// NOTE: CudaCalculateIter1Mrqcof1Curve2(in_rel[l_curves], l_points[l_curves]);		//  << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1Curve2, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(BLOCK_DIM));

				// NOTE: CudaCalculateIter1Mrqcof1End();												//	<< <CUDA_Grid_dim_precalc, 1 >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof1End, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
				
				//mrqcof
				// NOTE: CudaCalculateIter1Mrqmin1End();												//  << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqmin1End, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(BLOCK_DIM));
				//mrqcof

				// NOTE: CudaCalculateIter1Mrqcof2Start();											//	<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				//queue.enqueueNDRangeKernel(kernelCalculateIter1Mrqcof2Start, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(BLOCK_DIM));
				
				/*for (iC = 1; iC < l_curves; iC++)
				{
					CudaCalculateIter1Mrqcof2Matrix(l_points[iC]);							//	<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					CudaCalculateIter1Mrqcof2Curve1(in_rel[iC], l_points[iC]);				//	<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
					CudaCalculateIter1Mrqcof2Curve2(in_rel[iC], l_points[iC]);				//	<< <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				}
				CudaCalculateIter1Mrqcof2Curve1Last(in_rel[l_curves], l_points[l_curves]);	//	 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				CudaCalculateIter1Mrqcof2Curve2(in_rel[l_curves], l_points[l_curves]);		//	 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				CudaCalculateIter1Mrqcof2End();												//	 << <CUDA_Grid_dim_precalc, 1 >> >
				//mrqcof
				CudaCalculateIter1Mrqmin2End();												//	 << <CUDA_Grid_dim_precalc, 1 >> >
				CudaCalculateIter2();														//	 << <CUDA_Grid_dim_precalc, CUDA_BLOCK_DIM >> >
				//err=cudaThreadSynchronize(); memcpy is synchro itself

				// TODO: Read scalar buffer
				//cudaMemcpyFromSymbol(&theEnd, CUDA_End, sizeof(theEnd));
				theEnd = theEnd == CUDA_Grid_dim_precalc;

				//break;//debug
				*/
			}
			// NOTE: CudaCalculateFinishPole();														//	 << <CUDA_Grid_dim_precalc, 1 >> >

			// TODO: Sync threads on device -> kernel(s)?
			//err = cudaThreadSynchronize();
			//			err=cudaMemcpyFromSymbol(&res,CUDA_FR,sizeof(freq_result)*CUDA_Grid_dim_precalc);
			//			err=cudaMemcpyFromSymbol(&resc,CUDA_CC,sizeof(freq_context)*CUDA_Grid_dim_precalc);
						//break; //debug

		}

		// NOTE: CudaCalculateFinish();																						 << <CUDA_Grid_dim_precalc, 1 >> >
		queue.enqueueNDRangeKernel(kernelCalculateFinish, cl::NDRange(), cl::NDRange(CUDA_Grid_dim_precalc), cl::NDRange(1));
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

	//}
	//catch (cl::Error& e)
	//{
	//	if (e.err() == CL_BUILD_PROGRAM_FAILURE)
	//	{
	//		for (cl::Device dev : devices)
	//		{
	//			// Check the build status
	//			cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
	//			if (status != CL_BUILD_ERROR)
	//				continue;

	//			// Get the build log
	//			std::string name = dev.getInfo<CL_DEVICE_NAME>();
	//			std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
	//			std::cerr << "Build log for " << name << ":" << std::endl
	//				<< buildlog << std::endl;
	//		}
	//	}
	//	else if(e.err() != CL_SUCCESS)
	//	{
	//		std::string error = e.what();
	//		std::cerr << "Error " << e.err() << " | " << error << std::endl;
	//	}
	//	else
	//	{
	//		throw e;
	//	}
	//}
}

