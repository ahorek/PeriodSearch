#pragma once

#include "CalcStrategy.hpp"
#include "constants.h"
#include <arm_neon.h>
#include "arrayHelpers.hpp"

#ifndef CSASIMD
#define CSASIMD

class CalcStrategyAsimd final : public CalcStrategy
{
public:

	CalcStrategyAsimd() = default;
	//~CalcStrategyAsimd() = default;

	void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq, globals& gl) override;

	void bright(double t, double cg[], int ncoef, globals &gl) override;

	void conv(int nc, int ma, globals &gl) override;

	void curv(double cg[], globals &gl) override;

	void gauss_errc(double** a, int n, double b[], int &error) override;

private:
	float64x2_t* Dg_row[MAX_N_FAC + 3];
	float64x2_t dbr[MAX_N_FAC + 3];

	double php[N_PHOT_PAR + 1];
	double dphp[N_PHOT_PAR + 1];

	double e[4], e0[4];
	double de[4][4];
	double de0[4][4];
	double tmat[4][4];
	double dtm[4][4][4];

	double cos_alpha;
	double cl;
	double cls;
	double alpha;

	int ncoef0;
	int incl_count;
};

#endif