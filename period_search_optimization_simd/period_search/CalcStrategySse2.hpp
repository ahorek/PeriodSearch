#pragma once

#include <immintrin.h>
#include "CalcStrategy.hpp"
#include "constants.h"
#include "arrayHelpers.hpp"

#ifndef CSS2
#define CSS2

class CalcStrategySse2 : public CalcStrategy
{
public:
#if defined _WIN32
#pragma warning(disable:26495)
#endif

	CalcStrategySse2() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq, globals& gl);

	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br);

	virtual void conv(int nc, double dres[], int ma, double &result);

	virtual void curv(double cg[]);

	virtual void gauss_errc(double** a, int n, double b[], int &error);

private:
	__m128d* Dg_row[MAX_N_FAC + 3];
	__m128d dbr[MAX_N_FAC + 3];

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