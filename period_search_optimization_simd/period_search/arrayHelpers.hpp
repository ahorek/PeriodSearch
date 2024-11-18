#pragma once
#include "constants.h"

void init2Darray(double**& matrix, int dytemp_siszeX, int dytemp_sizeY);
void delete2Darray(double**& ary, int sizeY);
void printArray(int array[], int iMax, char msg[]);
void printArray(double array[], int iMax, char msg[]);
void printArray(double **array, int iMax, int jMax, char msg[]);
void printArray(double ***array, int iMax, int jMax, int kMax, char msg[]);

struct globals
{
#ifdef __GNUC__
	double Nor[3][MAX_N_FAC + 8] __attribute__((aligned(64))),
		Area[MAX_N_FAC + 8] __attribute__((aligned(64))),
		Darea[MAX_N_FAC + 8] __attribute__((aligned(64))),
		Dg[MAX_N_FAC + 16][MAX_N_PAR + 8] __attribute__((aligned(64)));
	double dyda[MAX_N_PAR + 16] __attribute__((aligned(64)));
#else
	// NOTE: About MSVC - https://learn.microsoft.com/en-us/cpp/cpp/alignment-cpp-declarations?view=msvc-170
	alignas(64) double Nor[3][MAX_N_FAC + 8];
	alignas(64) double Area[MAX_N_FAC + 8];
	alignas(64) double Darea[MAX_N_FAC + 8];
	alignas(64) double Dg[MAX_N_FAC + 16][MAX_N_PAR + 8];
	alignas(64) double dyda[MAX_N_PAR + 16];
#endif

	int Lcurves;
	int maxLcPoints;	// replaces MAX_LC_POINTS
	int maxDataPoints;	// replaces MAX_N_OBS
	int dytemp_sizeX;
	int dytemp_sizeY;
	int* Lpoints;
	int* Inrel;

	double ymod;
	double wt;
	double sig2i;
	double dy;
	double coef;
	double wght;
	double ave;
	double xx1[4];
	double xx2[4];
	double dave[MAX_N_PAR + 1 + 4];
	double* ytemp;
	double* Weight;
	double** dytemp;
};

