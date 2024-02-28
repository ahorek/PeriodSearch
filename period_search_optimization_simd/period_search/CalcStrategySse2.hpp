#pragma once

#include <immintrin.h>
#include "CalcStrategy.hpp"

#ifndef CSS2
#define CSS2

class CalcStrategySse2 : public CalcStrategy
{
public:

	CalcStrategySse2() {};

	virtual void mrqcof(double** x1, double** x2, double x3[], double y[],
		double sig[], double a[], int ia[], int ma,
		double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq);

	virtual void bright(double ee[], double ee0[], double t, double cg[], double dyda[], int ncoef, double &br);

	virtual void conv(int nc, double dres[], int ma, double &result);

	virtual void curv(double cg[]);

	virtual void gauss_errc(double** a, int n, double b[], int &error);
};

#endif