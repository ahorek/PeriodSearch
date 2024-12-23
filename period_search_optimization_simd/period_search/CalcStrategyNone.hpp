// ReSharper disable CppInconsistentNaming
#pragma once

#include "CalcStrategy.hpp"
#include "constants.h"
#include "arrayHelpers.hpp"

#ifndef CSNO
#define CSNO

class CalcStrategyNone final : public CalcStrategy
{
public:
#if defined _WIN32
#pragma warning(disable:26495)
#endif

	CalcStrategyNone() = default;

    //void mrqcof(double** x1, double** x2, double x3[], double y[],
    //            double sig[], double a[], int ia[], int ma,
    //            double** alpha, double beta[], int mfit, int lastone, int lastma, double &trial_chisq, globals& gl) override;

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
	double php[N_PHOT_PAR + 1];
	double dphp[N_PHOT_PAR + 1];
	double dbr[MAX_N_FAC]; //IS ZERO INDEXED
	double e[4];
	double e0[4];
	double de[4][4];
	double de0[4][4];
	double tmat[4][4];
	double dtm[4][4][4];

	double cos_alpha;
	double alpha;
	double cl;
	double cls;
	double dnom;
	double tmpdyda;
	double s;

	double tmpdyda1;
	double tmpdyda2;
	double tmpdyda3;
	double tmpdyda4;
	double tmpdyda5;

	int	incl[MAX_N_FAC]; //array of indexes of facets to Area, Dg, Nor. !!!!!!!!!!!incl IS ZERO INDEXED
	int ncoef0;
	int incl_count = 0;
};

#endif