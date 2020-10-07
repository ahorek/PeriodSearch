#pragma once
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.h>
#include <CL/cl.hpp>
#include "constants.h"
#include <vector>
#include <iostream>
#include <cstring>


#define __kernel

//#ifdef __GNUC__
//#define PACK( __Declaration__ ) __Declaration__ __attribute__((__packed__))
//#endif
//
//#ifdef _MSC_VER
//#define PACK( __Declaration__ ) __pragma( pack(push, 1) ) __Declaration__ __pragma( pack(pop))
//#endif


//     global to one thread
//struct freq_context
//{
//	//	double Area[MAX_N_FAC+1];
//	double* Area;
//	//	double Dg[(MAX_N_FAC+1)*(MAX_N_PAR+1)];
//	double* Dg;
//	//	double alpha[MAX_N_PAR+1][MAX_N_PAR+1];
//	double* alpha;
//	//	double covar[MAX_N_PAR+1][MAX_N_PAR+1];
//	double* covar;
//	//	double dytemp[(POINTS_MAX+1)*(MAX_N_PAR+1)]
//	double* dytemp;
//	//	double ytemp[POINTS_MAX+1],
//	double* ytemp;
//	double cg[MAX_N_PAR + 1];
//	double Ochisq, Chisq, Alamda;
//	double atry[MAX_N_PAR + 1], beta[MAX_N_PAR + 1], da[MAX_N_PAR + 1];
//	double Blmat[4][4];
//	double Dblm[3][4][4];
//	//mrqcof locals
//	double dyda[MAX_N_PAR + 1], dave[MAX_N_PAR + 1];
//	double trial_chisq, ave;
//	int n, np, np1, np2;
//	//bright
//	double e_1[POINTS_MAX + 1], e_2[POINTS_MAX + 1], e_3[POINTS_MAX + 1], e0_1[POINTS_MAX + 1], e0_2[POINTS_MAX + 1], e0_3[POINTS_MAX + 1], de[POINTS_MAX + 1][4][4], de0[POINTS_MAX + 1][4][4];
//	double jp_Scale[POINTS_MAX + 1];
//	double jp_dphp_1[POINTS_MAX + 1], jp_dphp_2[POINTS_MAX + 1], jp_dphp_3[POINTS_MAX + 1];
//	// gaus
//	int indxc[MAX_N_PAR + 1], indxr[MAX_N_PAR + 1], ipiv[MAX_N_PAR + 1];
//	//global
//	double freq;
//	int isNiter;
//	double iter_diff, rchisq, dev_old, dev_new;
//	int Niter;
//	double chck[4];
//	int isAlamda; //Alamda<0 for init
//	//
//	int isInvalid;
//	//test
//};
//
//extern __declspec(align(4)) freq_context* CUDA_CC;

struct freq_context2
{
	//double Sig[MAX_N_OBS + 1], Weight[MAX_N_OBS + 1], Brightness[MAX_N_OBS + 1];

	cl_int4 texDg[(MAX_N_FAC + 1) * (MAX_N_PAR + 1)];
	double dytemp[(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
	double Dg[(MAX_N_FAC + 1) * (MAX_N_PAR + 1)];
	double alpha[(MAX_N_PAR + 1) * (MAX_N_PAR + 1)];
	double de[POINTS_MAX + 1][4][4];
	double de0[POINTS_MAX + 1][4][4];
	double e_1[POINTS_MAX + 1];
	double e_2[POINTS_MAX + 1];
	double e_3[POINTS_MAX + 1];
	double e0_1[POINTS_MAX + 1];
	double e0_2[POINTS_MAX + 1];
	double e0_3[POINTS_MAX + 1];
	double jp_Scale[POINTS_MAX + 1];
	double jp_dphp_1[POINTS_MAX + 1];
	double jp_dphp_2[POINTS_MAX + 1];
	double jp_dphp_3[POINTS_MAX + 1];
	double ytemp[POINTS_MAX + 1];
	double Area[MAX_N_FAC + 1];

	double cg[MAX_N_PAR + 1];
	double da[MAX_N_PAR + 1];
	double dyda[MAX_N_PAR + 1];
	double dave[MAX_N_PAR + 1];
	double atry[MAX_N_PAR + 1];
	double covar[MAX_N_PAR + 1];
	double beta[MAX_N_PAR + 1];
	/*double Blmat[4][4];
	double Dblm[4][4][4];*/
	int indxc[MAX_N_PAR + 1];
	int indxr[MAX_N_PAR + 1];
	int ipiv[MAX_N_PAR + 1];
	double chck[4];
	double iter_diff, rchisq, dev_old, dev_new;
	double trial_chisq, ave;
	double freq;
	double Ochisq, Chisq, Alamda;
	//int isInvalid, isAlamda, isNiter;
	int Niter;
	int n, np, np1, np2;
};

//extern __declspec(align(4)) freq_context2* CUDA_CC2;
//extern freq_context2* CUDA_CC2;

// TODO: 1) Define "Texture" like Texture<int2> as it is pointless to define multi dimensional vector type. Use this:
// TODO: 2) Rename form "Texture" to something more suitable for the case like 'Matrix'...
struct Texture
{
	int x;
	int y;
};

//extern cl_int2* texArea;
//extern cl_int2* texDg;
//extern cl_int2* texWeight;
//extern cl_int2* texsig;
//extern cl_int2* texbrightness;

//extern cl::Image1D textWeight;
//extern cl::Image1D texbrightness;
//extern cl::Image1D texsig;


//extern texture<int2, 1> texWeight;
//extern texture<int2, 1> texbrightness;
//extern texture<int2, 1> texsig;


//extern cl::Image1D texArea;
//extern cl::Image1D texDg;

//extern texture<int2, 1> texArea;
//extern texture<int2, 1> texDg;

// NOTE: Check here https://docs.microsoft.com/en-us/cpp/preprocessor/pack?redirectedfrom=MSDN&view=vs-2019
//#pragma pack(4)
struct freq_result
{
	int isReported;
	double dark_best, per_best, dev_best, la_best, be_best;
};

//extern freq_result CUDA_FR;
//extern __declspec(align(4)) freq_result* CUDA_FR;

typedef struct FuncArrays
{
	int Mmax, Lmax;
	int ma, Nphpar;
	int Ncoef, Ncoef0;
	int Numfac;
	int Numfac1;
	int Lmfit, Lmfit1;
	int Dg_block;
	int lastone;
	int lastma;
	int Ndata;
	double cl;
	double Conw_r;
	double Phi_0;
	double Alamda_incr;
	double logCl;
	int ia[MAX_N_PAR + 1];
	//int* ia;
	double par[4];
	//double* lambda_pole;
	//double* beta_pole;
	////double cgFirst[MAX_N_PAR + 1];
	//double* cgFirst;
	int isNiter[MAX_N_ITER];
	int isInvalid[MAX_N_ITER];
	int isAlamda[MAX_N_ITER];
	int isReported[MAX_N_ITER];
	double tim[MAX_N_OBS + 1];
	double ee[MAX_N_OBS + 1][3];
	double ee0[MAX_N_OBS + 1][3];
	double Darea[MAX_N_FAC + 1];
	double Nor[MAX_N_FAC + 1][3];
	double Fc[MAX_N_FAC + 1][MAX_LM + 1];
	double Fs[MAX_N_FAC + 1][MAX_LM + 1];
	double Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
	double Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
	double cg[MAX_N_PAR + 1];
	double Sig[MAX_N_OBS + 1];
	double Weight[MAX_N_OBS + 1];
	double Brightness[MAX_N_OBS + 1];
	double Blmat[MAX_N_ITER][4][4];
	double Dblm[MAX_N_ITER][3][4][4];
} varholder;

//typedef struct FuncArrays varholder;

//extern varholder fa;