#pragma once

#include <immintrin.h>
#include "CalcStrategy.hpp"
#include "CalcStrategyAvx.hpp"
#include "constants.h"
#include "arrayHelpers.hpp"

#ifndef CSF
#define CSF

class CalcStrategyFma : public CalcStrategyAvx
{
public:
#if defined _WIN32
#pragma warning(disable:26495)
#endif

	CalcStrategyFma() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq, globals& gl);

	//virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br, globals &gl);
	virtual void bright(double ee[], double ee0[], double t, double cg[], int ncoef, globals &gl);

	//virtual void conv(int nc, double dres[], int ma, double &result, globals &gl);
	virtual void conv(int nc, int ma, globals &gl);

	virtual void curv(double cg[], globals &gl);

	virtual void gauss_errc(double** a, int n, double b[], int &error);

private:
	__m256d* Dg_row[MAX_N_FAC + 3];
	__m256d dbr[MAX_N_FAC + 3];

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