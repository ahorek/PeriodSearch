// ReSharper disable CppInconsistentNaming
#pragma once

#include <immintrin.h>
//#include "CalcStrategy.hpp"
#include "CalcStrategyAvx.hpp"
#include "constants.h"
#include "arrayHelpers.hpp"

#ifndef CSF
#define CSF

class CalcStrategyFma final : public CalcStrategyAvx
{
public:
#if defined _WIN32
#pragma warning(disable:26495)
#endif

	CalcStrategyFma() = default;

	//void mrqcof(double** x1, double** x2, double x3[], double y[],
	//	double sig[], double a[], int ia[], int ma,
	//	double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq, globals& gl) override;
	void mrqcof(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
		std::vector<double>& sig, std::vector<double>& a, std::vector<int>& ia, int ma,
		std::vector<double>& beta, int mfit, int lastone, int lastma, double& trial_chisq, globals& gl, const bool isCovar) override;

	//void mrqcof(std::vector<std::vector<double>>& x1, std::vector<std::vector<double>>& x2, std::vector<double>& x3, std::vector<double>& y,
	//	std::vector<double>& sig, double a[], int ia[], int ma,
	//	double** alpha, double beta[], int mfit, int lastone, int lastma, double& trial_chisq, globals& gl) override;

	void bright(double t, std::vector<double>& cg, int ncoef, globals &gl) override;
	//void bright(double t, double cg[], int ncoef, globals& gl) override;

	void conv(int nc, int ma, globals &gl) override;

	void curv(std::vector<double>& cg, globals &gl) override;
	//void curv(double cg[], globals& gl) override;

	//void gauss_errc(double** a, int n, double b[], int &error) override;
	void gauss_errc(struct globals& gl, const int n, std::vector<double>& b, int &error) override;

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