#pragma once
//#include <stdio.h>
#include "constants.h"

#ifdef NO_SSE3
#include <emmintrin.h>
#else
#include <pmmintrin.h>
#endif

extern int Lmax, Mmax, Niter, Lastcall,
Ncoef, Numfac, Lcurves, Nphpar,
//Lpoints[MAX_LC+1], Inrel[MAX_LC+1],
* Lpoints, * Inrel,
Deallocate;

extern double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale,
Sclnw[MAX_LC + 1],
Yout[MAX_N_OBS + 1],
Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1],
Blmat[4][4],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
Dblm[3][4][4],

Weight[MAX_N_OBS + 1];

#ifdef __GNUC__
extern double Nor[3][MAX_N_FAC + 2] __attribute__((aligned(16)));
//Area[MAX_N_FAC + 2] __attribute__((aligned(16)));
			  //Darea[MAX_N_FAC + 2] __attribute__((aligned(16)));
			  //Dg[MAX_N_FAC+4][MAX_N_PAR+10] __attribute__ ((aligned (16)));
#else
extern __declspec(align(16)) double Nor[3][MAX_N_FAC + 2] , Area[MAX_N_FAC + 2] , Darea[MAX_N_FAC + 2] , Dg[MAX_N_FAC + 4][MAX_N_PAR + 10]; //All are zero indexed
#endif

//extern double** Dg, * Darea, * Area, * Weight;

struct Data {
	double per_start, per_step_coef, per_end,
		freq, freq_start, freq_step, freq_end, jd_min, jd_max,
		dev_old, dev_new, iter_diff, iter_diff_max, stop_condition,
		totarea, sum, dark, dev_best, per_best, dark_best, la_tmp, be_tmp, la_best, be_best,
		sum_dark_facet, ave_dark_facet;
};

__declspec(align(16)) struct Bright {
	__m128d Dg_row[MAX_N_FAC + 3];
	__m128d dbr[MAX_N_FAC + 3];
};