// ReSharper disable IdentifierTypo
#pragma once
#include <cstdio>
#include <vector>
#include "constants.h"
#include "CalcStrategy.hpp"

// Shared by all threads - set up before the fits start and only read during them
extern int Lmax, Mmax, Ncoef, Numfac, Nphpar;

extern double Alamda_incr, Alamda_start, Phi_0,

Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];

// Per-thread state of one Levenberg-Marquardt fit. Every OpenMP thread fits its own pole,
// so everything written during a fit (mrqmin/mrqcof/bright/phasec/blmatrix) is thread_local.
extern thread_local int Niter, Lastcall, Deallocate;

extern thread_local double Ochisq, Chisq, Alamda, Scale,
Blmat[4][4],
Dblm[3][4][4];

extern thread_local std::vector<double> atry;
extern thread_local std::vector<double> beta;
extern thread_local std::vector<double> da;

// The SIMD strategy objects cache intermediate results in their members -> one per thread
extern thread_local CalcContext calcCtx;

extern struct SIMDSupport
{
	bool hasAVX512dq = false;
	bool hasAVX512 = false;
	bool hasFMA = false;
	bool hasAVX = false;
	bool hasSSE3 = false;
	bool hasSSE2 = false;
	bool hasASIMD = false;
	bool hasSVE = false;
	bool isBulldozer = false;
} CPUopt;